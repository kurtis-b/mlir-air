# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared references and packing helpers for Gemma3 dataflow kernels."""

from __future__ import annotations

import numpy as np
from ml_dtypes import bfloat16


Q4NX_ROWS = 32
Q4NX_COLS = 256


def pack_int4_low_first(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.uint8).reshape(-1)
    if flat.size % 2:
        raise ValueError("int4 value count must be even")
    if np.any(flat > 15):
        raise ValueError("int4 values must be in [0, 15]")
    packed = (flat[0::2] & 0x0F) | ((flat[1::2] & 0x0F) << 4)
    return packed.astype(np.uint8)


def unpack_int4_low_first(packed: np.ndarray, count: int) -> np.ndarray:
    packed_u8 = np.asarray(packed, dtype=np.uint8).reshape(-1)
    out = np.empty(count, dtype=np.uint8)
    out[0::2] = packed_u8[: (count + 1) // 2] & 0x0F
    out[1::2] = (packed_u8[: count // 2] >> 4) & 0x0F
    return out


def random_q4nx_block(rows: int, cols: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    q = rng.integers(0, 16, size=(rows, cols), dtype=np.uint8)
    scale = rng.uniform(0.005, 0.05, size=(cols,)).astype(bfloat16)
    min_offset = rng.uniform(-0.4, 0.2, size=(cols,)).astype(bfloat16)
    packed = pack_int4_low_first(q)
    return q, packed.view(np.int8), scale, min_offset


def q4nx_dequant_reference(
    packed_i8: np.ndarray, scale: np.ndarray, min_offset: np.ndarray, rows: int, cols: int
) -> np.ndarray:
    q = unpack_int4_low_first(packed_i8.view(np.uint8), rows * cols).reshape(
        rows, cols
    )
    ref = q.astype(np.float32) * scale.astype(np.float32)[None, :]
    ref += min_offset.astype(np.float32)[None, :]
    return ref.astype(bfloat16)


def fused_dqp_reference(
    packed_i8: np.ndarray,
    scale: np.ndarray,
    min_offset: np.ndarray,
    activation: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    w = q4nx_dequant_reference(packed_i8, scale, min_offset, rows, cols)
    out = w.astype(np.float32) @ activation.astype(np.float32)
    return out.astype(bfloat16)


def attention_reference(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    query_base: int = 0,
    causal: bool = False,
    window_len: int = 0,
) -> np.ndarray:
    qf = q.astype(np.float32)
    kf = k.astype(np.float32)
    vf = v.astype(np.float32)
    q_chunk, head_dim = qf.shape
    kv_len = kf.shape[0]
    scale = 1.0 / np.sqrt(float(head_dim))
    out = np.zeros((q_chunk, vf.shape[1]), dtype=np.float32)

    for qi in range(q_chunk):
        end = kv_len
        if causal:
            end = min(end, query_base + qi + 1)
        start = 0
        if window_len > 0:
            start = max(0, end - window_len)
        if end <= start:
            continue
        scores = qf[qi] @ kf[start:end].T * scale
        mx = np.max(scores)
        weights = np.exp(scores - mx)
        weights /= np.sum(weights)
        out[qi] = weights @ vf[start:end]
    return out.astype(bfloat16)
