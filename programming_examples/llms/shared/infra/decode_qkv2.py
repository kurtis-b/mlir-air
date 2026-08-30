# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host side of the 2-launch decode QKV stage: the head-aligned GEMV with QK-norm
+ RoPE as an in-core epilogue
(`shared.builders.rms_qkv_qknorm_rope_multi.build_rms_qkv_qknorm_rope_gemv2_module`).
The single owner of the ELF's 5-arg layout, so the index table lives in one place:

    0 x_in (emb)          1 norm_w (emb, static)
    2 bvec (emb + 3*head_dim) = [normed | lut | q_norm | k_norm]: the device
      writes [0, emb) (RMSNorm launch), the host fills the tail every call
      (the LUT is position-dependent), so it is a DYNAMIC input, not an
      intermediate
    3 wqkv2 (static): the builder form's storage of [wq; wk; wv]
      (`qkv2_prep_weight`)   4 out (output, device-written)

The GEMV's output buffer may be larger than q|k|v (per-iteration slots);
`qkv2_gather` (None for logical-order forms) maps it to q_roped | k_roped | v.
Requires the per-layer weights object to carry `_wq_t`, `_wk_t`, `_wv_t`
(out, in) bf16 transposes, `attn_norm`, `q_norm`, `k_norm` -- what every Qwen3
driver's `prepare_runtime` already produces. numpy + ml_dtypes only.
"""

import numpy as np
from ml_dtypes import bfloat16

from shared.infra.qkv2_layout import qkv2_gather, qkv2_out_total, qkv2_prep_weight

OUTPUT_INDICES_2 = [4]
STATIC_INDICES_2 = {1, 3}
INTERMEDIATE_INDICES_2 = {4}


def prep_weights_2(lw, config):
    """Host-side, once per layer: the 2-launch form's static weight. Idempotent.
    Needs `_wq_t`, `_wk_t`, `_wv_t`."""
    if getattr(lw, "_wqkv2", None) is not None:
        return
    q_dim = config.n_heads * config.head_dim
    kv_dim = config.n_kv_heads * config.head_dim
    wqkv_t = np.concatenate([lw._wq_t, lw._wk_t, lw._wv_t], axis=0)  # logical
    lw._wqkv2 = qkv2_prep_weight(wqkv_t, q_dim, q_dim + kv_dim, config.head_dim)
    lw._q_norm_row = np.asarray(lw.q_norm, bfloat16).reshape(config.head_dim)
    lw._k_norm_row = np.asarray(lw.k_norm, bfloat16).reshape(config.head_dim)


def call_args_2(lw, x_bf16, lut_row, config):
    """The 5 positional inputs of the 2-launch form, in ELF arg order.
    `lut_row` is one position's RoPE LUT row (head_dim,)."""
    prep_weights_2(lw, config)
    emb_dim, head_dim = config.emb_dim, config.head_dim
    q_dim = config.n_heads * head_dim
    kv_dim = config.n_kv_heads * head_dim
    qkv_dim = q_dim + 2 * kv_dim
    bvec = np.zeros(emb_dim + 3 * head_dim, dtype=bfloat16)
    bvec[emb_dim : emb_dim + head_dim] = np.asarray(lut_row, bfloat16).reshape(head_dim)
    bvec[emb_dim + head_dim : emb_dim + 2 * head_dim] = lw._q_norm_row
    bvec[emb_dim + 2 * head_dim :] = lw._k_norm_row
    return [
        np.asarray(x_bf16).flatten().astype(bfloat16),  # 0 x_in
        lw.attn_norm.reshape(emb_dim).astype(bfloat16),  # 1 norm_w (static)
        bvec,  # 2 bvec (DYNAMIC: lut)
        lw._wqkv2,  # 3 wqkv2 (static)
        np.zeros(qkv2_out_total(qkv_dim, head_dim), dtype=bfloat16),  # 4 out
    ]


def split_outputs_2(res, config):
    """(v, q_roped, k_roped) from the ELF's output (arg 4)."""
    head_dim = config.head_dim
    q_dim = config.n_heads * head_dim
    kv_dim = config.n_kv_heads * head_dim
    qkv_dim = q_dim + 2 * kv_dim
    out = np.asarray(res[4]).astype(bfloat16)
    g = qkv2_gather(qkv_dim, head_dim)
    if g is not None:
        out = out[g]
    return (
        np.ascontiguousarray(out[q_dim + kv_dim :]),
        np.ascontiguousarray(out[:q_dim]),
        np.ascontiguousarray(out[q_dim : q_dim + kv_dim]),
    )
