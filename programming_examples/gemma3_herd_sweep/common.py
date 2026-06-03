# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared references and packing helpers for Gemma3 dataflow kernels."""

from __future__ import annotations

import numpy as np
from ml_dtypes import bfloat16


Q4NX_ROWS = 32
Q4NX_COLS = 256
SUPPORTED_HERD_SHAPES = ("2x4", "4x4", "8x4")
SCHEDULE_MODES = ("smoke", "paper")
FLOW_VARIANTS = ("causal", "swa")
FLOW_KV_STAGING_MODES = ("replicated", "shared", "pipeline")
OUTPUT_MODES = ("auto", "direct", "l2-gather")
DIAGNOSTIC_OUTPUT_MODES = ("packet-direct",)
ALL_OUTPUT_MODES = OUTPUT_MODES + DIAGNOSTIC_OUTPUT_MODES
OUTPUT_MODE_KERNELS = ("q4nx", "fused_dqp", "flowqkv", "flowkv")

_OUTPUT_MODE_SUPPORT = {
    "q4nx": {
        "2x4": ("direct", "l2-gather"),
        "4x4": ("direct", "l2-gather"),
        "8x4": ("l2-gather",),
    },
    "fused_dqp": {
        "2x4": ("direct", "l2-gather"),
        "4x4": ("direct", "l2-gather"),
        "8x4": ("l2-gather",),
    },
    "flowqkv": {
        "2x4": ("direct", "l2-gather"),
        "4x4": ("direct", "l2-gather"),
        "8x4": ("l2-gather",),
    },
    "flowkv": {
        "2x4": ("direct",),
        "4x4": ("direct",),
        "8x4": ("l2-gather",),
    },
}

_AUTO_OUTPUT_MODE = {
    "q4nx": "l2-gather",
    "fused_dqp": "direct",
    "flowqkv": "direct",
    "flowkv": "direct",
}

_OUTPUT_MODE_UNSUPPORTED_REASONS = {
    ("q4nx", "8x4", "direct"): (
        "direct output from the full 8x4 herd exceeds the shim S2MM DMA "
        "channel budget; use l2-gather"
    ),
    ("fused_dqp", "8x4", "direct"): (
        "direct output from the full 8x4 herd exceeds the shim S2MM DMA "
        "channel budget; use l2-gather"
    ),
    ("flowqkv", "8x4", "direct"): (
        "direct output from the full 8x4 herd exceeds the shim S2MM DMA "
        "channel budget; use l2-gather"
    ),
    ("flowkv", "8x4", "direct"): (
        "direct output from the full 8x4 herd exceeds the shim S2MM DMA "
        "channel budget; use l2-gather"
    ),
    ("flowkv", "2x4", "l2-gather"): (
        "FlowKV small-shape L2 gather is diagnostic-only: channel-staged "
        "compile passes, but hardware validation is not clean; use direct"
    ),
    ("flowkv", "4x4", "l2-gather"): (
        "FlowKV small-shape L2 gather is diagnostic-only: channel-staged "
        "compile passes, but hardware validation is not clean; use direct"
    ),
    ("q4nx", "8x4", "packet-direct"): (
        "packet-direct is diagnostic-only: shared shim S2MM packet output "
        "currently corrupts hardware validation; use l2-gather"
    ),
    ("fused_dqp", "8x4", "packet-direct"): (
        "packet-direct is diagnostic-only: shared shim S2MM packet output "
        "currently corrupts hardware validation; use l2-gather"
    ),
}


def parse_herd_shape(shape: str) -> tuple[int, int]:
    if shape not in SUPPORTED_HERD_SHAPES:
        supported = ", ".join(SUPPORTED_HERD_SHAPES)
        raise ValueError(f"herd shape must be one of: {supported}")
    rows, cols = shape.split("x", 1)
    return int(rows), int(cols)


def herd_shape_name(herd_rows: int, herd_cols: int) -> str:
    return f"{herd_rows}x{herd_cols}"


def supported_output_modes(
    kernel: str, herd_rows: int, herd_cols: int
) -> tuple[str, ...]:
    shape = herd_shape_name(herd_rows, herd_cols)
    try:
        return _OUTPUT_MODE_SUPPORT[kernel][shape]
    except KeyError as exc:
        if kernel not in OUTPUT_MODE_KERNELS:
            supported = ", ".join(OUTPUT_MODE_KERNELS)
            raise ValueError(f"kernel must be one of: {supported}") from exc
        supported = ", ".join(SUPPORTED_HERD_SHAPES)
        raise ValueError(f"herd shape must be one of: {supported}") from exc


def is_output_mode_supported(
    mode: str, herd_rows: int, herd_cols: int, kernel: str
) -> bool:
    if mode == "auto":
        return True
    if mode not in ALL_OUTPUT_MODES:
        return False
    return mode in supported_output_modes(kernel, herd_rows, herd_cols)


def unsupported_output_mode_reason(
    mode: str, herd_rows: int, herd_cols: int, kernel: str
) -> str | None:
    shape = herd_shape_name(herd_rows, herd_cols)
    reason = _OUTPUT_MODE_UNSUPPORTED_REASONS.get((kernel, shape, mode))
    if reason:
        return reason
    if mode == "packet-direct" and kernel in ("q4nx", "fused_dqp"):
        return (
            "packet-direct is diagnostic-only: shared shim S2MM packet output "
            "does not have passing hardware validation; use l2-gather"
        )
    if mode == "packet-direct":
        return (
            "packet-direct is diagnostic-only and has no passing hardware "
            "validation for this kernel/shape"
        )
    return None


def resolve_output_mode(
    mode: str, herd_rows: int, herd_cols: int, kernel: str
) -> str:
    if mode not in ALL_OUTPUT_MODES:
        supported = ", ".join(ALL_OUTPUT_MODES)
        raise ValueError(f"output mode must be one of: {supported}")
    modes = supported_output_modes(kernel, herd_rows, herd_cols)
    if mode == "auto":
        mode = _AUTO_OUTPUT_MODE[kernel]
        if mode not in modes:
            mode = "l2-gather"
    if mode in modes:
        return mode

    supported = ", ".join(modes)
    shape = herd_shape_name(herd_rows, herd_cols)
    reason = unsupported_output_mode_reason(mode, herd_rows, herd_cols, kernel)
    if reason:
        reason = f" ({reason})"
    else:
        reason = ""
    raise ValueError(
        f"output mode {mode!r} is not supported for {kernel} {shape}{reason}; "
        f"supported modes: {supported}"
    )


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


def random_q4nx_blocks(
    row_blocks: int, col_blocks: int, rows: int, cols: int, seed: int = 0
):
    rng = np.random.default_rng(seed)
    packed = np.empty((row_blocks, col_blocks, rows * cols // 2), dtype=np.int8)
    scale = np.empty((row_blocks, col_blocks, cols), dtype=bfloat16)
    min_offset = np.empty((row_blocks, col_blocks, cols), dtype=bfloat16)

    for rb in range(row_blocks):
        for cb in range(col_blocks):
            q = rng.integers(0, 16, size=(rows, cols), dtype=np.uint8)
            packed[rb, cb] = pack_int4_low_first(q).view(np.int8)
            scale[rb, cb] = rng.uniform(0.005, 0.05, size=(cols,)).astype(bfloat16)
            min_offset[rb, cb] = rng.uniform(-0.4, 0.2, size=(cols,)).astype(
                bfloat16
            )
    return packed, scale, min_offset


def q4nx_dequant_reference(
    packed_i8: np.ndarray, scale: np.ndarray, min_offset: np.ndarray, rows: int, cols: int
) -> np.ndarray:
    q = unpack_int4_low_first(packed_i8.view(np.uint8), rows * cols).reshape(
        rows, cols
    )
    ref = q.astype(np.float32) * scale.astype(np.float32)[None, :]
    ref += min_offset.astype(np.float32)[None, :]
    return ref.astype(bfloat16)


def q4nx_dequant_blocks_reference(
    packed_i8: np.ndarray,
    scale: np.ndarray,
    min_offset: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    row_blocks, col_blocks = packed_i8.shape[:2]
    out = np.empty((row_blocks * rows, col_blocks * cols), dtype=bfloat16)
    for rb in range(row_blocks):
        for cb in range(col_blocks):
            block = q4nx_dequant_reference(
                packed_i8[rb, cb], scale[rb, cb], min_offset[rb, cb], rows, cols
            )
            out[rb * rows : (rb + 1) * rows, cb * cols : (cb + 1) * cols] = block
    return out


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


def fused_dqp_blocks_reference(
    packed_i8: np.ndarray,
    scale: np.ndarray,
    min_offset: np.ndarray,
    activation: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    row_blocks = packed_i8.shape[0]
    out = np.empty((row_blocks, rows), dtype=bfloat16)
    for rb in range(row_blocks):
        out[rb] = fused_dqp_reference(
            packed_i8[rb], scale[rb], min_offset[rb], activation, rows, cols
        )
    return out


def fused_dqp_paper_reference(
    packed_i8: np.ndarray,
    scale: np.ndarray,
    min_offset: np.ndarray,
    activation: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    row_blocks, col_blocks = packed_i8.shape[:2]
    out = np.zeros((row_blocks, rows), dtype=np.float32)
    for rb in range(row_blocks):
        for cb in range(col_blocks):
            block = fused_dqp_reference(
                packed_i8[rb, cb],
                scale[rb, cb],
                min_offset[rb, cb],
                activation[cb],
                rows,
                cols,
            )
            out[rb] += block.astype(np.float32)
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


def tiled_attention_reference(
    q_tiles: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    kv_groups: int,
    heads_per_kv: int,
    query_base: int = 0,
    causal: bool = False,
    window_len: int = 0,
) -> np.ndarray:
    herd_rows, herd_cols = q_tiles.shape[:2]
    out = np.empty_like(q_tiles)
    tiles_per_query_chunk = kv_groups * heads_per_kv
    for tx in range(herd_rows):
        for ty in range(herd_cols):
            linear = tx * herd_cols + ty
            q_slot = linear // tiles_per_query_chunk
            rem = linear % tiles_per_query_chunk
            kv_group = rem // heads_per_kv
            tile_query_base = query_base + q_slot * q_tiles.shape[2]
            out[tx, ty] = attention_reference(
                q_tiles[tx, ty],
                k[kv_group],
                v[kv_group],
                query_base=tile_query_base,
                causal=causal,
                window_len=window_len,
            )
    return out
