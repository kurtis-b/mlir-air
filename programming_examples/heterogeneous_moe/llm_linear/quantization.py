# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from numerics import quantize_array

QuantKind = Literal["int4", "uint4"]


@dataclass(frozen=True)
class PackedWeightMetadata:
    quant_kind: QuantKind
    shape: tuple[int, int]
    block_size: int
    quant_axis: int
    signed: bool
    packing: str = "two_values_per_byte_low_nibble_first"


@dataclass(frozen=True)
class PackedLinearWeights:
    packed: np.ndarray
    scales: np.ndarray
    metadata: PackedWeightMetadata
    zero_points: np.ndarray | None = None

    def descriptor(self) -> dict[str, object]:
        return {
            "quant_kind": self.metadata.quant_kind,
            "shape": list(self.metadata.shape),
            "block_size": self.metadata.block_size,
            "quant_axis": self.metadata.quant_axis,
            "signed": self.metadata.signed,
            "packing": self.metadata.packing,
            "packed_bytes": int(self.packed.nbytes),
            "scale_shape": [int(dim) for dim in self.scales.shape],
            "scale_bytes": int(self.scales.nbytes),
            "zero_point_shape": (
                None
                if self.zero_points is None
                else [int(dim) for dim in self.zero_points.shape]
            ),
        }


def pack_4bit(values: np.ndarray, *, signed: bool = False) -> np.ndarray:
    ints = np.asarray(values, dtype=np.int16).reshape(-1)
    if signed:
        if np.any(ints < -8) or np.any(ints > 7):
            raise ValueError("signed int4 values must be in [-8, 7]")
        nibbles = (ints & 0xF).astype(np.uint8)
    else:
        if np.any(ints < 0) or np.any(ints > 15):
            raise ValueError("uint4 values must be in [0, 15]")
        nibbles = ints.astype(np.uint8)
    if nibbles.size % 2:
        nibbles = np.concatenate((nibbles, np.zeros(1, dtype=np.uint8)))
    low = nibbles[0::2]
    high = nibbles[1::2] << np.uint8(4)
    return np.ascontiguousarray(low | high)


def unpack_4bit(packed: np.ndarray, *, count: int, signed: bool = False) -> np.ndarray:
    data = np.asarray(packed, dtype=np.uint8).reshape(-1)
    low = data & np.uint8(0xF)
    high = (data >> np.uint8(4)) & np.uint8(0xF)
    unpacked = np.empty(data.size * 2, dtype=np.int8 if signed else np.uint8)
    if signed:
        low_signed = np.where(low >= 8, low.astype(np.int8) - 16, low.astype(np.int8))
        high_signed = np.where(
            high >= 8, high.astype(np.int8) - 16, high.astype(np.int8)
        )
        unpacked[0::2] = low_signed
        unpacked[1::2] = high_signed
    else:
        unpacked[0::2] = low
        unpacked[1::2] = high
    return np.ascontiguousarray(unpacked[:count])


def metadata_for_packed_weights(
    *,
    quant_kind: QuantKind,
    shape: tuple[int, int],
    block_size: int,
    quant_axis: int,
) -> PackedWeightMetadata:
    if quant_kind not in {"int4", "uint4"}:
        raise ValueError(f"Unsupported quant_kind: {quant_kind}")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if quant_axis not in {0, 1}:
        raise ValueError("quant_axis must be 0 or 1")
    return PackedWeightMetadata(
        quant_kind=quant_kind,
        shape=shape,
        block_size=block_size,
        quant_axis=quant_axis,
        signed=quant_kind == "int4",
    )


def quantize_weight_matrix(
    weights: np.ndarray,
    *,
    quant_kind: QuantKind = "int4",
    block_size: int = 32,
    quant_axis: int = 0,
) -> PackedLinearWeights:
    matrix = np.asarray(weights, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("weights must be a rank-2 matrix")
    metadata = metadata_for_packed_weights(
        quant_kind=quant_kind,
        shape=(int(matrix.shape[0]), int(matrix.shape[1])),
        block_size=block_size,
        quant_axis=quant_axis,
    )
    axis_extent = int(matrix.shape[quant_axis])
    block_count = (axis_extent + block_size - 1) // block_size
    qvalues = np.zeros(matrix.shape, dtype=np.int8 if metadata.signed else np.uint8)
    scale_shape = (
        (block_count, int(matrix.shape[1]))
        if quant_axis == 0
        else (int(matrix.shape[0]), block_count)
    )
    scales = np.ones(scale_shape, dtype=np.float32)
    zero_points = None if metadata.signed else np.zeros(scale_shape, dtype=np.uint8)

    for block_index in range(block_count):
        start = block_index * block_size
        stop = min(start + block_size, axis_extent)
        if quant_axis == 0:
            selectors = (slice(start, stop), slice(None))
            block = matrix[selectors]
            scale_slot = (block_index, slice(None))
        else:
            selectors = (slice(None), slice(start, stop))
            block = matrix[selectors]
            scale_slot = (slice(None), block_index)

        if metadata.signed:
            max_abs = np.max(np.abs(block), axis=quant_axis, keepdims=True)
            scale = np.where(max_abs == 0.0, 1.0, max_abs / np.float32(7.0))
            quantized = np.clip(np.rint(block / scale), -8, 7).astype(np.int8)
            scales[scale_slot] = np.squeeze(scale, axis=quant_axis)
        else:
            min_value = np.min(block, axis=quant_axis, keepdims=True)
            max_value = np.max(block, axis=quant_axis, keepdims=True)
            scale = np.where(
                max_value == min_value,
                1.0,
                (max_value - min_value) / np.float32(15.0),
            )
            zp = np.clip(np.rint(-min_value / scale), 0, 15).astype(np.uint8)
            quantized = np.clip(np.rint(block / scale) + zp, 0, 15).astype(np.uint8)
            scales[scale_slot] = np.squeeze(scale, axis=quant_axis)
            assert zero_points is not None
            zero_points[scale_slot] = np.squeeze(zp, axis=quant_axis)
        qvalues[selectors] = quantized

    return PackedLinearWeights(
        packed=pack_4bit(qvalues, signed=metadata.signed),
        scales=np.ascontiguousarray(scales),
        zero_points=None if zero_points is None else np.ascontiguousarray(zero_points),
        metadata=metadata,
    )


def dequantize_packed_weights(
    packed_weights: PackedLinearWeights, *, dtype_name: str | None = None
) -> np.ndarray:
    metadata = packed_weights.metadata
    count = int(metadata.shape[0] * metadata.shape[1])
    qvalues = unpack_4bit(
        packed_weights.packed, count=count, signed=metadata.signed
    ).reshape(metadata.shape)
    output = np.zeros(metadata.shape, dtype=np.float32)
    axis_extent = int(metadata.shape[metadata.quant_axis])
    block_count = (axis_extent + metadata.block_size - 1) // metadata.block_size

    for block_index in range(block_count):
        start = block_index * metadata.block_size
        stop = min(start + metadata.block_size, axis_extent)
        if metadata.quant_axis == 0:
            selectors = (slice(start, stop), slice(None))
            scale = packed_weights.scales[block_index, :][None, :]
            zp = (
                None
                if packed_weights.zero_points is None
                else packed_weights.zero_points[block_index, :][None, :]
            )
        else:
            selectors = (slice(None), slice(start, stop))
            scale = packed_weights.scales[:, block_index][:, None]
            zp = (
                None
                if packed_weights.zero_points is None
                else packed_weights.zero_points[:, block_index][:, None]
            )

        block_q = qvalues[selectors].astype(np.float32)
        if metadata.signed:
            output[selectors] = block_q * scale
        else:
            assert zp is not None
            output[selectors] = (block_q - zp.astype(np.float32)) * scale

    if dtype_name is None:
        return np.ascontiguousarray(output)
    return quantize_array(output, dtype_name)


def decode_gemv_fused_dequant(
    prefill_last_row: np.ndarray,
    packed_weights: PackedLinearWeights,
    dtype_name: str,
) -> tuple[np.ndarray, dict[str, float]]:
    import time

    dequant_start = time.perf_counter_ns()
    metadata = packed_weights.metadata
    count = int(metadata.shape[0] * metadata.shape[1])
    qvalues = unpack_4bit(
        packed_weights.packed, count=count, signed=metadata.signed
    ).reshape(metadata.shape)
    dequant_ms = (time.perf_counter_ns() - dequant_start) / 1_000_000.0

    linear_start = time.perf_counter_ns()
    vector = np.asarray(prefill_last_row, dtype=np.float32)
    output = np.zeros(metadata.shape[1], dtype=np.float32)
    axis_extent = int(metadata.shape[metadata.quant_axis])
    block_count = (axis_extent + metadata.block_size - 1) // metadata.block_size
    for block_index in range(block_count):
        start = block_index * metadata.block_size
        stop = min(start + metadata.block_size, axis_extent)
        if metadata.quant_axis == 0:
            selectors = (slice(start, stop), slice(None))
            scale = packed_weights.scales[block_index, :][None, :]
            zp = (
                None
                if packed_weights.zero_points is None
                else packed_weights.zero_points[block_index, :][None, :]
            )
            block_vector = vector[start:stop]
            block_q = qvalues[selectors].astype(np.float32)
            if metadata.signed:
                block_weights = block_q * scale
            else:
                assert zp is not None
                block_weights = (block_q - zp.astype(np.float32)) * scale
            output += block_vector @ block_weights
        else:
            selectors = (slice(None), slice(start, stop))
            scale = packed_weights.scales[:, block_index][:, None]
            zp = (
                None
                if packed_weights.zero_points is None
                else packed_weights.zero_points[:, block_index][:, None]
            )
            block_q = qvalues[selectors].astype(np.float32)
            if metadata.signed:
                block_weights = block_q * scale
            else:
                assert zp is not None
                block_weights = (block_q - zp.astype(np.float32)) * scale
            output[start:stop] = vector @ block_weights
    linear_ms = (time.perf_counter_ns() - linear_start) / 1_000_000.0
    return quantize_array(output, dtype_name), {
        "dequant_ms": float(dequant_ms),
        "linear_ms": float(linear_ms),
        "packed_weight_bytes_read": float(packed_weights.packed.nbytes),
        "scale_bytes_read": float(packed_weights.scales.nbytes),
        "zero_point_bytes_read": float(
            0
            if packed_weights.zero_points is None
            else packed_weights.zero_points.nbytes
        ),
    }
