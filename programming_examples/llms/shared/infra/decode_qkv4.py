# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host side of the 4-launch decode QKV stage (`rms_qkv_qknorm_rope_gemv4`).

`[2026-08-21]` Doc 57 O1 first cut. The builder is
`shared.builders.rms_qkv_qknorm_rope_multi.build_rms_qkv_qknorm_rope_gemv4_module`:
RMSNorm, ONE GEMV over the row-packed `[wq; wk; wv]`, ONE per-row-weighted QK-norm
over the Q|K rows, ONE RoPE over Q|K -- the 8-launch stage's kernels with four
fewer `air.launch` boundaries (~107 us each, doc 57 section 1.5). This module is
the single owner of the ELF's 9-arg layout for every Qwen3 decode driver, so the
index table lives in one place:

    0 x_in (emb)        1 norm_w (emb, static)      2 normed (emb)
    3 wqkv (q+2kv, emb, static)                     4 qkv = q | k | v  (output)
    5 qk_norm_w (q+kv, static)                      6 qk_n (q+kv)
    7 lut (q+kv, position-dependent, NOT static)    8 qk_roped = q_roped | k_roped (output)

Requires the per-layer weights object to carry `_wq_t`, `_wk_t`, `_wv_t`
(out, in) bf16 transposes, `attn_norm`, `q_norm`, `k_norm` -- what every Qwen3
driver's `prepare_runtime` already produces.
"""

import numpy as np
from ml_dtypes import bfloat16

OUTPUT_INDICES = [4, 8]
STATIC_INDICES = {1, 3, 5}
INTERMEDIATE_INDICES = {2, 4, 6, 8}

# 3-launch form (host_rmsnorm=True in the builder): the RMSNorm launch is gone
# and arg0 is the host-normalized vector.
#     0 normed (emb)   1 wqkv (static)   2 qkv (output)   3 qk_norm_w (static)
#     4 qk_n           5 lut             6 qk_roped (output)
OUTPUT_INDICES_3 = [2, 6]
STATIC_INDICES_3 = {1, 3}
INTERMEDIATE_INDICES_3 = {2, 4, 6}


def host_rmsnorm(x_bf16, norm_w, eps):
    """bf16(x * rsqrt(mean(x^2) + eps) * w) in f32 -- the device RMSNorm's math."""
    xf = np.asarray(x_bf16, dtype=np.float32).reshape(-1)
    rstd = 1.0 / np.sqrt(np.mean(xf * xf) + eps)
    return (xf * rstd * np.asarray(norm_w, dtype=np.float32).reshape(-1)).astype(bfloat16)


def call_args_3(lw, x_bf16, lut, config, eps):
    """The 7 positional inputs of the 3-launch form; RMSNorm done here."""
    prep_weights(lw, config)
    emb_dim = config.emb_dim
    q_dim = config.n_heads * config.head_dim
    kv_dim = config.n_kv_heads * config.head_dim
    qk_dim = q_dim + kv_dim
    return [
        host_rmsnorm(x_bf16, lw.attn_norm.reshape(emb_dim), eps),  # 0 normed (host)
        lw._wqkv_t,  # 1 wqkv (static)
        np.zeros(qk_dim + kv_dim, dtype=bfloat16),  # 2 qkv
        lw._qk_norm_w,  # 3 qk_norm_w (static)
        np.zeros(qk_dim, dtype=bfloat16),  # 4 qk_n
        np.asarray(lut, bfloat16),  # 5 lut (DYNAMIC)
        np.zeros(qk_dim, dtype=bfloat16),  # 6 qk_roped
    ]


def split_outputs_3(res, config):
    q_dim = config.n_heads * config.head_dim
    kv_dim = config.n_kv_heads * config.head_dim
    qkv = np.asarray(res[2]).astype(bfloat16)
    qk_roped = np.asarray(res[6]).astype(bfloat16)
    return (
        np.ascontiguousarray(qkv[q_dim + kv_dim :]),
        np.ascontiguousarray(qk_roped[:q_dim]),
        np.ascontiguousarray(qk_roped[q_dim:]),
    )


def run_3(cache, kernel_name, backend_kwargs, lw, x_bf16, lut, config, layer_idx, eps=1e-6):
    """One call of the 3-launch stage (host RMSNorm) -> (v, q_roped, k_roped)."""
    res = cache.load_and_run(
        kernel_name,
        backend_kwargs,
        *call_args_3(lw, x_bf16, lut, config, eps),
        output_indices=OUTPUT_INDICES_3,
        static_input_indices=STATIC_INDICES_3,
        intermediate_indices=INTERMEDIATE_INDICES_3,
        bo_key=f"{kernel_name}_L{layer_idx}" if layer_idx is not None else None,
    )
    return split_outputs_3(res, config)


def prep_weights(lw, config):
    """Host-side, once per layer: the two packed static buffers. Idempotent.

    `_wqkv_t` = [wq_t; wk_t; wv_t] (q_dim+2*kv_dim, emb_dim);
    `_qk_norm_w` = [q_norm x n_heads; k_norm x n_kv_heads] (q_dim+kv_dim,).
    """
    if getattr(lw, "_wqkv_t", None) is not None:
        return
    head_dim = config.head_dim
    lw._wqkv_t = np.ascontiguousarray(
        np.concatenate([lw._wq_t, lw._wk_t, lw._wv_t], axis=0)
    )
    lw._qk_norm_w = np.ascontiguousarray(
        np.concatenate(
            [
                np.tile(np.asarray(lw.q_norm, bfloat16).reshape(head_dim), config.n_heads),
                np.tile(np.asarray(lw.k_norm, bfloat16).reshape(head_dim), config.n_kv_heads),
            ]
        ).astype(bfloat16)
    )


def position_lut(rope_lut_bf16, current_pos, config):
    """The position's RoPE LUT row tiled over the n_heads+n_kv_heads Q|K rows."""
    pos = rope_lut_bf16[current_pos : current_pos + 1]  # (1, head_dim)
    return np.tile(pos, (config.n_heads + config.n_kv_heads, 1)).flatten().astype(bfloat16)


def call_args(lw, x_bf16, lut, config):
    """The 9 positional inputs, in ELF arg order."""
    prep_weights(lw, config)
    emb_dim = config.emb_dim
    q_dim = config.n_heads * config.head_dim
    kv_dim = config.n_kv_heads * config.head_dim
    qk_dim = q_dim + kv_dim
    return [
        np.asarray(x_bf16).flatten().astype(bfloat16),  # 0 x_in
        lw.attn_norm.reshape(emb_dim).astype(bfloat16),  # 1 norm_w (static)
        np.zeros(emb_dim, dtype=bfloat16),  # 2 normed
        lw._wqkv_t,  # 3 wqkv (static)
        np.zeros(qk_dim + kv_dim, dtype=bfloat16),  # 4 qkv
        lw._qk_norm_w,  # 5 qk_norm_w (static)
        np.zeros(qk_dim, dtype=bfloat16),  # 6 qk_n
        np.asarray(lut, bfloat16),  # 7 lut (DYNAMIC -- position)
        np.zeros(qk_dim, dtype=bfloat16),  # 8 qk_roped
    ]


def split_outputs(res, config):
    """(v, q_roped, k_roped) from the ELF's outputs (arg 4 and arg 8)."""
    q_dim = config.n_heads * config.head_dim
    kv_dim = config.n_kv_heads * config.head_dim
    qkv = np.asarray(res[4]).astype(bfloat16)
    qk_roped = np.asarray(res[8]).astype(bfloat16)
    return (
        np.ascontiguousarray(qkv[q_dim + kv_dim :]),
        np.ascontiguousarray(qk_roped[:q_dim]),
        np.ascontiguousarray(qk_roped[q_dim:]),
    )


def run(cache, kernel_name, backend_kwargs, lw, x_bf16, lut, config, layer_idx):
    """One call of the 4-launch stage -> (v, q_roped, k_roped)."""
    res = cache.load_and_run(
        kernel_name,
        backend_kwargs,
        *call_args(lw, x_bf16, lut, config),
        output_indices=OUTPUT_INDICES,
        static_input_indices=STATIC_INDICES,
        intermediate_indices=INTERMEDIATE_INDICES,
        bo_key=f"{kernel_name}_L{layer_idx}" if layer_idx is not None else None,
    )
    return split_outputs(res, config)
