# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from numerics import array_error_metrics, quantize_array
from .quantization import (
    PackedLinearWeights,
    dequantize_packed_weights,
    quantize_weight_matrix,
)

DEFAULT_INPUT_SCALE = 0.25
DEFAULT_WEIGHT_SCALE = 0.125


@dataclass(frozen=True)
class LinearConfig:
    M: int = 4
    K: int = 64
    H: int = 64
    N: int = 32
    dtype: str = "bf16"
    shape_tier: str = "tiny_ci"

    @property
    def element_bytes(self) -> int:
        return 2


@dataclass(frozen=True)
class LinearWeights:
    prefill: np.ndarray
    decode: np.ndarray
    decode_quantized: PackedLinearWeights | None = None


def decode_quantization_from_manifest(
    manifest: dict[str, Any],
) -> dict[str, Any] | None:
    decode = manifest.get("weights", {}).get("decode", {})
    if not isinstance(decode, dict):
        return None
    storage = decode.get("storage", "bf16")
    if storage in {None, "bf16", "dense"}:
        return None
    if storage not in {"int4", "uint4"}:
        raise ValueError(f"weights.decode.storage is invalid: {storage}")
    return {
        "quant_kind": storage,
        "block_size": int(decode.get("block_size", 32)),
        "quant_axis": int(decode.get("quant_axis", 0)),
    }


def config_from_manifest(manifest: dict[str, Any]) -> LinearConfig:
    model = manifest["model"]
    return LinearConfig(
        M=int(model["M"]),
        K=int(model["K"]),
        H=int(model["H"]),
        N=int(model["N"]),
        dtype=str(model["dtype"]),
        shape_tier=str(model.get("shape_tier", "custom")),
    )


def random_inputs(
    cfg: LinearConfig, seed: int, scale: float = DEFAULT_INPUT_SCALE
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((cfg.M, cfg.K), dtype=np.float32) * np.float32(scale)
    return quantize_array(values, cfg.dtype)


def random_weights(
    cfg: LinearConfig,
    seed: int,
    scale: float = DEFAULT_WEIGHT_SCALE,
    decode_quantization: dict[str, Any] | None = None,
) -> LinearWeights:
    rng = np.random.default_rng(seed)

    def arr(shape: tuple[int, ...]) -> np.ndarray:
        values = rng.standard_normal(shape, dtype=np.float32) * np.float32(scale)
        return quantize_array(values, cfg.dtype)

    prefill = arr((cfg.K, cfg.H))
    decode = arr((cfg.H, cfg.N))
    if decode_quantization is None:
        return LinearWeights(prefill=prefill, decode=decode)

    packed = quantize_weight_matrix(decode, **decode_quantization)
    dequantized = dequantize_packed_weights(packed, dtype_name=cfg.dtype)
    return LinearWeights(prefill=prefill, decode=dequantized, decode_quantized=packed)


def prefill_gemm(
    inputs: np.ndarray, weights: np.ndarray, dtype_name: str
) -> np.ndarray:
    output = np.asarray(inputs, dtype=np.float32) @ np.asarray(
        weights, dtype=np.float32
    )
    return quantize_array(output, dtype_name)


def decode_gemv(
    prefill_last_row: np.ndarray, weights: np.ndarray, dtype_name: str
) -> np.ndarray:
    output = np.asarray(prefill_last_row, dtype=np.float32) @ np.asarray(
        weights, dtype=np.float32
    )
    return quantize_array(output, dtype_name)


def run_reference(
    cfg: LinearConfig, inputs: np.ndarray, weights: LinearWeights
) -> dict[str, np.ndarray]:
    prefill = prefill_gemm(inputs, weights.prefill, cfg.dtype)
    decode_input = np.ascontiguousarray(prefill[cfg.M - 1, :])
    output = decode_gemv(decode_input, weights.decode, cfg.dtype)
    return {
        "prefill": prefill,
        "decode_input": decode_input,
        "output": output,
    }


def validation_tolerances(dtype_name: str) -> dict[str, float]:
    normalized = dtype_name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return {"atol": 0.35, "rtol": 0.08}
    return {"atol": 0.05, "rtol": 0.02}


def stage_metrics(
    actual: dict[str, np.ndarray],
    expected: dict[str, np.ndarray],
    dtype_name: str,
) -> dict[str, dict[str, float | bool]]:
    tolerances = validation_tolerances(dtype_name)
    return {
        name: array_error_metrics(actual[name], expected[name], **tolerances)
        for name in ("prefill", "decode_input", "output")
    }


def workload_bytes(cfg: LinearConfig) -> dict[str, int]:
    element_bytes = cfg.element_bytes
    values = {
        "input": cfg.M * cfg.K * element_bytes,
        "prefill_weights": cfg.K * cfg.H * element_bytes,
        "prefill_output": cfg.M * cfg.H * element_bytes,
        "decode_input": cfg.H * element_bytes,
        "decode_weights": cfg.H * cfg.N * element_bytes,
        "output": cfg.N * element_bytes,
    }
    values["total_tensor_bytes"] = int(sum(values.values()))
    return {key: int(value) for key, value in values.items()}
