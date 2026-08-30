# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-side layout contract of the head-aligned QKV GEMV kernel (`mv_heads.cc`):
the augmented weight the kernel's TAG/KIND row padding defines, and the slot
layout its per-iteration head outputs land in. numpy + ml_dtypes only -- no air,
so the host tests need no toolchain. Port of the layout half of the study
branch's 2-launch Qwen3 QKV stage (rms_qkv_qknorm_rope_multi.py on tag
pre-port-20260829); the ELF builder that consumes these lands separately.
"""

import numpy as np
from ml_dtypes import bfloat16

K_PAD = 64  # row padding of the augmented weight matrix: [tag, kind, 0 ...]
QKV2_TILE_M = 8
QKV2_HERD_M = 8


def qkv_heads_store_perm(m, herd_m, tile_m):
    """Index array P with A_stored = A_logical[P]: stored row
    i*herd_m*tile_m + tx*tile_m + r = logical row tx*rows_per_col + i*tile_m + r."""
    rows_per_col = m // herd_m
    n_iter = rows_per_col // tile_m
    perm = np.empty(m, dtype=np.int64)
    for i in range(n_iter):
        for tx in range(herd_m):
            base = i * herd_m * tile_m + tx * tile_m
            perm[base : base + tile_m] = (
                tx * rows_per_col + i * tile_m + np.arange(tile_m)
            )
    return perm


def qkv_heads_slot_gather(m, herd_m, head_dim, tile_m=8):
    """Index array G with out_logical = slots[G] for out_slots=True: head h of
    column tx is final in the slot of its last chunk's iteration
    (h*chunks_per_head + chunks_per_head - 1), at [tx*head_dim, +head_dim)."""
    rows_per_col = m // herd_m
    cph = head_dim // tile_m
    g = np.empty(m, dtype=np.int64)
    for tx in range(herd_m):
        for h in range(rows_per_col // head_dim):
            it = h * cph + cph - 1
            lo = tx * rows_per_col + h * head_dim
            g[lo : lo + head_dim] = (
                it * herd_m * head_dim + tx * head_dim + np.arange(head_dim)
            )
    return g


def qkv_heads_augment_weight(w_logical, q_rows, qk_rows, head_dim, herd_m=8, tile_m=8):
    """The static A of `_build_qkv_heads_gemv` from the logical [wq; wk; wv]
    (M, K): rows permuted iteration-major and padded by K_PAD with
    [tag, kind, 0...] per row (tag = the row's chunk index within its head,
    kind = 0 Q / 1 K / 2 V). Done once per layer by the host."""
    m, k = w_logical.shape
    perm = qkv_heads_store_perm(m, herd_m, tile_m)
    aug = np.zeros((m, k + K_PAD), dtype=bfloat16)
    aug[:, :k] = np.asarray(w_logical, dtype=bfloat16)[perm]
    logical = perm  # logical row of each stored row
    tag = (logical % head_dim) // tile_m
    kind = np.where(logical < q_rows, 0, np.where(logical < qk_rows, 1, 2))
    aug[:, k] = tag.astype(bfloat16)
    aug[:, k + 1] = kind.astype(bfloat16)
    return np.ascontiguousarray(aug)


def qkv2_prep_weight(w_logical, q_dim, qk_dim, head_dim):
    """Static weight of the 2-launch ELF from the logical [wq; wk; wv] (M, K)."""
    return qkv_heads_augment_weight(
        w_logical, q_dim, qk_dim, head_dim, QKV2_HERD_M, QKV2_TILE_M
    )


def qkv2_out_total(qkv_dim, head_dim):
    """Length of the ELF's output arg (per-iteration head slots)."""
    rows_per_col = qkv_dim // QKV2_HERD_M
    return (rows_per_col // QKV2_TILE_M) * QKV2_HERD_M * head_dim


def qkv2_gather(qkv_dim, head_dim):
    """Index array mapping the output arg to q_roped | k_roped | v (or None)."""
    return qkv_heads_slot_gather(qkv_dim, QKV2_HERD_M, head_dim, QKV2_TILE_M)
