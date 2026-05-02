# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np


def normalize_dtype_name(dtype_name: str) -> str:
    normalized = dtype_name.lower()
    aliases = {
        "float16": "f16",
        "half": "f16",
        "bfloat16": "bf16",
    }
    return aliases.get(normalized, normalized)


def is_bf16_dtype(dtype_name: str) -> bool:
    return normalize_dtype_name(dtype_name) == "bf16"


def host_array_dtype(dtype_name: str) -> np.dtype:
    return np.dtype(np.float32 if is_bf16_dtype(dtype_name) else np.float16)


def npu_buffer_dtype(dtype_name: str) -> np.dtype:
    return np.dtype(np.uint16 if is_bf16_dtype(dtype_name) else np.float16)


def float32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float32)
    raw = data.view(np.uint32)
    lsb = (raw >> 16) & np.uint32(1)
    rounding_bias = np.uint32(0x7FFF) + lsb
    rounded = raw + rounding_bias
    return (rounded >> 16).astype(np.uint16)


def bf16_bits_to_float32(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return raw.view(np.float32)


def quantize_array(values: np.ndarray, dtype_name: str) -> np.ndarray:
    normalized = normalize_dtype_name(dtype_name)
    if normalized == "bf16":
        return bf16_bits_to_float32(float32_to_bf16_bits(values))
    if normalized == "f16":
        return np.asarray(values, dtype=np.float16)
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def quantize_scalar(value: float, dtype_name: str) -> float:
    normalized = normalize_dtype_name(dtype_name)
    if normalized == "bf16":
        return float(
            bf16_bits_to_float32(
                float32_to_bf16_bits(np.asarray([value], dtype=np.float32))
            )[0]
        )
    if normalized == "f16":
        return float(np.float16(value))
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def encode_npu_array(values: np.ndarray, dtype_name: str) -> np.ndarray:
    normalized = normalize_dtype_name(dtype_name)
    if normalized == "bf16":
        return np.ascontiguousarray(float32_to_bf16_bits(values))
    if normalized == "f16":
        return np.ascontiguousarray(np.asarray(values, dtype=np.float16))
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def decode_npu_array(values: np.ndarray, dtype_name: str) -> np.ndarray:
    normalized = normalize_dtype_name(dtype_name)
    if normalized == "bf16":
        return np.ascontiguousarray(bf16_bits_to_float32(values))
    if normalized == "f16":
        return np.ascontiguousarray(np.asarray(values, dtype=np.float16))
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def array_error_metrics(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> dict[str, float | bool]:
    actual_f32 = np.asarray(actual, dtype=np.float32)
    expected_f32 = np.asarray(expected, dtype=np.float32)
    delta = actual_f32 - expected_f32
    abs_delta = np.abs(delta)
    delta_f64 = np.asarray(delta, dtype=np.float64)
    return {
        "max_abs_error": float(np.max(abs_delta)),
        "mean_abs_error": float(np.mean(abs_delta)),
        "rmse": float(np.sqrt(np.mean(np.square(delta_f64)))),
        "max_abs_expected": float(np.max(np.abs(expected_f32))),
        "atol": float(atol),
        "rtol": float(rtol),
        "allclose": bool(np.allclose(actual_f32, expected_f32, atol=atol, rtol=rtol)),
    }


def encoded_array_summary(values: np.ndarray, dtype_name: str) -> dict[str, object]:
    encoded = encode_npu_array(values, dtype_name)
    return {
        "shape": list(values.shape),
        "host_dtype": str(np.asarray(values).dtype),
        "encoded_dtype": str(encoded.dtype),
        "elements": int(encoded.size),
        "nbytes": int(encoded.nbytes),
        "contiguous": bool(np.ascontiguousarray(values).flags.c_contiguous),
    }
