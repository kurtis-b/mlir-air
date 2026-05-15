# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

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
        plan = decode_quantization_plan_for_metadata(self.metadata)
        return {
            "storage": self.metadata.quant_kind,
            "quant_kind": self.metadata.quant_kind,
            "shape": list(self.metadata.shape),
            "block_size": self.metadata.block_size,
            "quant_axis": self.metadata.quant_axis,
            "signed": self.metadata.signed,
            "packing": self.metadata.packing,
            "kernel_key": plan.kernel_key,
            "hardware_fused": plan.hardware_fused,
            "packed_shape": [int(dim) for dim in plan.packed_shape],
            "packed_bytes": int(self.packed.nbytes),
            "scale_shape": [int(dim) for dim in self.scales.shape],
            "scale_bytes": int(self.scales.nbytes),
            "zero_point_shape": (
                None
                if self.zero_points is None
                else [int(dim) for dim in self.zero_points.shape]
            ),
        }


@dataclass(frozen=True)
class DecodeQuantizationPlan:
    storage: QuantKind
    block_size: int
    quant_axis: int
    shape: tuple[int, int] | None = None
    npu_decode_tile_n: int | None = None

    @property
    def quant_kind(self) -> QuantKind:
        return self.storage

    @property
    def signed(self) -> bool:
        return self.storage == "int4"

    @property
    def hardware_fused(self) -> bool:
        if self.shape is None:
            return False
        h, n = self.shape
        return (
            self.storage == "int4"
            and self.quant_axis == 0
            and self.block_size > 0
            and h % self.block_size == 0
            and n % 8 == 0
        )

    @property
    def kernel_key(self) -> str:
        return "decode_int4" if self.hardware_fused else "decode"

    @property
    def packed_shape(self) -> tuple[int, ...]:
        if self.shape is None:
            return ()
        h, n = self.shape
        if self.hardware_fused:
            return (h, n // 8)
        return ((h * n + 1) // 2,)

    @property
    def packed_bytes(self) -> int:
        if self.shape is None:
            return 0
        h, n = self.shape
        return (h * n + 1) // 2

    @property
    def scale_shape(self) -> tuple[int, ...]:
        if self.shape is None:
            return ()
        if self.block_size <= 0:
            return ()
        h, n = self.shape
        if self.quant_axis == 0:
            return ((h + self.block_size - 1) // self.block_size, n)
        return (h, (n + self.block_size - 1) // self.block_size)

    @property
    def scale_bytes(self) -> int:
        if not self.scale_shape:
            return 0
        return int(np.prod(self.scale_shape)) * np.dtype(np.float32).itemsize

    @property
    def zero_point_shape(self) -> tuple[int, ...] | None:
        return None if self.signed else self.scale_shape

    @property
    def zero_point_bytes(self) -> int:
        if self.zero_point_shape is None:
            return 0
        return int(np.prod(self.zero_point_shape)) * np.dtype(np.uint8).itemsize

    def quantize_kwargs(self) -> dict[str, Any]:
        return {
            "quant_kind": self.storage,
            "block_size": self.block_size,
            "quant_axis": self.quant_axis,
        }

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "storage": self.storage,
            "quant_kind": self.storage,
            "block_size": self.block_size,
            "quant_axis": self.quant_axis,
            "kernel_key": self.kernel_key,
            "hardware_fused": self.hardware_fused,
            "packed_shape": [int(dim) for dim in self.packed_shape],
            "packed_bytes": self.packed_bytes,
            "scale_shape": [int(dim) for dim in self.scale_shape],
            "scale_bytes": self.scale_bytes,
            "npu_decode_tile_n": self.npu_decode_tile_n,
        }


@dataclass(frozen=True)
class DecodeHardwareWeightArrays:
    packed: np.ndarray
    scales: np.ndarray
    plan: DecodeQuantizationPlan


def decode_quantization_plan_from_manifest(
    manifest: dict[str, Any],
    *,
    shape: tuple[int, int] | None = None,
    npu_decode_tile_n: int | None = None,
) -> DecodeQuantizationPlan | None:
    decode = manifest.get("weights", {}).get("decode", {})
    if not isinstance(decode, dict):
        return None
    storage = decode.get("storage", "bf16")
    if storage in {None, "bf16", "dense"}:
        return None
    if storage not in {"int4", "uint4"}:
        raise ValueError(f"weights.decode.storage is invalid: {storage}")
    return DecodeQuantizationPlan(
        storage=storage,
        block_size=int(decode.get("block_size", 32)),
        quant_axis=int(decode.get("quant_axis", 0)),
        shape=shape,
        npu_decode_tile_n=npu_decode_tile_n,
    )


def decode_quantization_plan_for_metadata(
    metadata: PackedWeightMetadata,
    *,
    npu_decode_tile_n: int | None = None,
) -> DecodeQuantizationPlan:
    return DecodeQuantizationPlan(
        storage=metadata.quant_kind,
        block_size=metadata.block_size,
        quant_axis=metadata.quant_axis,
        shape=metadata.shape,
        npu_decode_tile_n=npu_decode_tile_n,
    )


def validate_accelerator_decode_plan(
    plan: DecodeQuantizationPlan | None, *, require_int4: bool = False
) -> DecodeQuantizationPlan | None:
    if plan is None:
        if require_int4:
            raise ValueError("accelerator quantized decode requires storage == int4")
        return None
    if plan.storage != "int4":
        raise ValueError("accelerator quantized decode supports only signed int4")
    if plan.block_size <= 0:
        raise ValueError("accelerator int4 decode requires positive block_size")
    if plan.quant_axis != 0:
        raise ValueError("accelerator int4 decode requires quant_axis == 0")
    if plan.shape is None:
        raise ValueError("accelerator int4 decode requires a concrete HxN shape")
    h, n = plan.shape
    if h % plan.block_size != 0:
        raise ValueError("accelerator int4 decode requires H % block_size == 0")
    if n % 8 != 0:
        raise ValueError("accelerator int4 decode requires N divisible by 8")
    return plan


def hardware_decode_weight_arrays(
    packed_weights: PackedLinearWeights,
    *,
    npu_decode_tile_n: int | None = None,
) -> DecodeHardwareWeightArrays:
    plan = decode_quantization_plan_for_metadata(
        packed_weights.metadata, npu_decode_tile_n=npu_decode_tile_n
    )
    validate_accelerator_decode_plan(plan, require_int4=True)
    packed_bytes = np.ascontiguousarray(
        np.asarray(packed_weights.packed, dtype=np.uint8)
    )
    if int(packed_bytes.nbytes) != plan.packed_bytes:
        raise ValueError("packed int4 decode buffer has an unexpected byte count")
    packed_matrix = np.ascontiguousarray(
        packed_bytes.view(np.uint32).reshape(plan.packed_shape)
    )
    scales = np.ascontiguousarray(np.asarray(packed_weights.scales, dtype=np.float32))
    if tuple(int(dim) for dim in scales.shape) != plan.scale_shape:
        raise ValueError("int4 decode scales have an unexpected shape")
    return DecodeHardwareWeightArrays(packed=packed_matrix, scales=scales, plan=plan)


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
