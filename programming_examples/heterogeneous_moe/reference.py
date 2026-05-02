# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from kernels import KernelConfig
from numerics import array_error_metrics, quantize_array, quantize_scalar

DEFAULT_INPUT_SCALE = 0.5
DEFAULT_WEIGHT_SCALE = 0.5
DEFAULT_ROUTING_PROFILE = "balanced"


@dataclass(frozen=True)
class MoEWeights:
    router: np.ndarray
    expert0_w1: np.ndarray
    expert0_w2: np.ndarray
    expert1_w1: np.ndarray
    expert1_w2: np.ndarray


def random_inputs(
    cfg: KernelConfig,
    seed: int,
    scale: float = DEFAULT_INPUT_SCALE,
    routing_profile: str = DEFAULT_ROUTING_PROFILE,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal(
        (cfg.batch_tokens, cfg.hidden_size), dtype=np.float32
    ) * np.float32(scale)
    if routing_profile == "expert0_hot":
        data = np.abs(data)
    elif routing_profile == "expert1_hot":
        data = np.abs(data)
    elif routing_profile == "alternating":
        for row in range(cfg.batch_tokens):
            if row % 2 == 0:
                data[row] = np.abs(data[row])
            else:
                data[row] = -np.abs(data[row])
    elif routing_profile != "balanced":
        raise ValueError(f"Unsupported routing profile: {routing_profile}")
    return quantize_array(data, cfg.dtype)


def _router_weights(
    cfg: KernelConfig,
    rng: np.random.Generator,
    scale: float,
    routing_profile: str,
) -> np.ndarray:
    if routing_profile == "balanced":
        data = rng.standard_normal((cfg.hidden_size, 2), dtype=np.float32) * np.float32(
            scale
        )
        return quantize_array(data, cfg.dtype)

    profile_scale = np.float32(max(scale * 6.0, 1.0))
    if routing_profile == "expert0_hot":
        data = np.stack(
            (
                np.full(cfg.hidden_size, profile_scale, dtype=np.float32),
                np.full(cfg.hidden_size, -profile_scale, dtype=np.float32),
            ),
            axis=1,
        )
        return quantize_array(data, cfg.dtype)
    if routing_profile == "expert1_hot":
        data = np.stack(
            (
                np.full(cfg.hidden_size, -profile_scale, dtype=np.float32),
                np.full(cfg.hidden_size, profile_scale, dtype=np.float32),
            ),
            axis=1,
        )
        return quantize_array(data, cfg.dtype)
    if routing_profile == "alternating":
        data = np.stack(
            (
                np.full(cfg.hidden_size, profile_scale, dtype=np.float32),
                np.full(cfg.hidden_size, -profile_scale, dtype=np.float32),
            ),
            axis=1,
        )
        return quantize_array(data, cfg.dtype)
    raise ValueError(f"Unsupported routing profile: {routing_profile}")


def random_weights(
    cfg: KernelConfig,
    seed: int,
    scale: float = DEFAULT_WEIGHT_SCALE,
    routing_profile: str = DEFAULT_ROUTING_PROFILE,
) -> MoEWeights:
    rng = np.random.default_rng(seed)

    def arr(shape: tuple[int, ...]) -> np.ndarray:
        data = rng.standard_normal(shape, dtype=np.float32) * np.float32(scale)
        return quantize_array(data, cfg.dtype)

    return MoEWeights(
        router=_router_weights(cfg, rng, scale, routing_profile),
        expert0_w1=arr((cfg.hidden_size, cfg.ffn_size)),
        expert0_w2=arr((cfg.ffn_size, cfg.hidden_size)),
        expert1_w1=arr((cfg.hidden_size, cfg.ffn_size)),
        expert1_w2=arr((cfg.ffn_size, cfg.hidden_size)),
    )


def router_logits(
    inputs: np.ndarray, weights: np.ndarray, dtype_name: str
) -> np.ndarray:
    logits = np.asarray(inputs, dtype=np.float32) @ np.asarray(
        weights, dtype=np.float32
    )
    return quantize_array(logits, dtype_name)


def softmax_rows(logits: np.ndarray, dtype_name: str) -> np.ndarray:
    logits_f32 = logits.astype(np.float32)
    shifted = logits_f32 - np.max(logits_f32, axis=1, keepdims=True)
    expd = np.exp(shifted)
    probs = expd / np.sum(expd, axis=1, keepdims=True)
    return quantize_array(probs, dtype_name)


def topk_weights(logits: np.ndarray, mode: str, dtype_name: str) -> np.ndarray:
    probs = softmax_rows(logits, dtype_name)
    if mode == "top2":
        return probs
    if mode != "top1":
        raise ValueError(f"Unsupported router mode: {mode}")
    indices = np.argmax(probs.astype(np.float32), axis=1)
    out = np.zeros_like(probs)
    out[np.arange(probs.shape[0]), indices] = quantize_array(
        np.asarray([1.0], dtype=np.float32), dtype_name
    )[0]
    return out


def routed_inputs(
    inputs: np.ndarray, weights: np.ndarray, dtype_name: str
) -> tuple[np.ndarray, np.ndarray]:
    expert0 = quantize_array(
        inputs.astype(np.float32) * weights[:, 0:1].astype(np.float32), dtype_name
    )
    expert1 = quantize_array(
        inputs.astype(np.float32) * weights[:, 1:2].astype(np.float32), dtype_name
    )
    return expert0, expert1


def expert_mlp(
    inputs: np.ndarray, w1: np.ndarray, w2: np.ndarray, dtype_name: str
) -> np.ndarray:
    hidden = np.asarray(inputs, dtype=np.float32) @ np.asarray(w1, dtype=np.float32)
    hidden = quantize_array(np.maximum(hidden, 0.0), dtype_name)
    output = np.asarray(hidden, dtype=np.float32) @ np.asarray(w2, dtype=np.float32)
    return quantize_array(output, dtype_name)


def pack_expert_outputs(
    expert0: np.ndarray, expert1: np.ndarray, dtype_name: str
) -> np.ndarray:
    return quantize_array(np.concatenate((expert0, expert1), axis=1), dtype_name)


def aggregate_outputs(
    expert0: np.ndarray, expert1: np.ndarray, weights: np.ndarray, dtype_name: str
) -> np.ndarray:
    lhs = quantize_array(
        np.asarray(expert0, dtype=np.float32)
        * np.asarray(weights[:, 0:1], dtype=np.float32),
        dtype_name,
    )
    rhs = quantize_array(
        np.asarray(expert1, dtype=np.float32)
        * np.asarray(weights[:, 1:2], dtype=np.float32),
        dtype_name,
    )
    return quantize_array(
        np.asarray(lhs, dtype=np.float32) + np.asarray(rhs, dtype=np.float32),
        dtype_name,
    )


def aggregate_packed_outputs(
    experts: np.ndarray, weights: np.ndarray, dtype_name: str
) -> np.ndarray:
    split = experts.shape[1] // 2
    return aggregate_outputs(
        experts[:, :split], experts[:, split:], weights, dtype_name
    )


def run_reference(
    cfg: KernelConfig, inputs: np.ndarray, weights: MoEWeights, mode: str
) -> dict[str, np.ndarray]:
    logits = router_logits(inputs, weights.router, cfg.dtype)
    routes = topk_weights(logits, mode, cfg.dtype)
    expert0_in, expert1_in = routed_inputs(inputs, routes, cfg.dtype)
    expert0_out = expert_mlp(
        expert0_in, weights.expert0_w1, weights.expert0_w2, cfg.dtype
    )
    expert1_out = expert_mlp(
        expert1_in, weights.expert1_w1, weights.expert1_w2, cfg.dtype
    )
    packed_expert_outputs = pack_expert_outputs(expert0_out, expert1_out, cfg.dtype)
    aggregated = aggregate_packed_outputs(packed_expert_outputs, routes, cfg.dtype)
    return {
        "logits": logits,
        "weights": routes,
        "expert0_input": expert0_in,
        "expert1_input": expert1_in,
        "expert0_output": expert0_out,
        "expert1_output": expert1_out,
        "packed_expert_outputs": packed_expert_outputs,
        "output": aggregated,
    }


def torch_reference(
    inputs: np.ndarray, weights: MoEWeights, mode: str
) -> dict[str, np.ndarray]:
    import torch

    x = torch.tensor(inputs.astype(np.float32))
    router_w = torch.tensor(weights.router.astype(np.float32))
    logits = x @ router_w
    probs = torch.softmax(logits, dim=-1)
    if mode == "top1":
        indices = torch.argmax(probs, dim=-1)
        routed = torch.zeros_like(probs)
        routed.scatter_(1, indices.unsqueeze(-1), 1.0)
    elif mode == "top2":
        routed = probs
    else:
        raise ValueError(f"Unsupported router mode: {mode}")

    expert0_in = x * routed[:, 0:1]
    expert1_in = x * routed[:, 1:2]
    e0 = torch.relu(
        expert0_in @ torch.tensor(weights.expert0_w1.astype(np.float32))
    ) @ torch.tensor(weights.expert0_w2.astype(np.float32))
    e1 = torch.relu(
        expert1_in @ torch.tensor(weights.expert1_w1.astype(np.float32))
    ) @ torch.tensor(weights.expert1_w2.astype(np.float32))
    packed = torch.cat((e0, e1), dim=1)
    out = e0 * routed[:, 0:1] + e1 * routed[:, 1:2]
    return {
        "logits": logits.detach().cpu().numpy().astype(np.float32, copy=False),
        "weights": routed.detach().cpu().numpy().astype(np.float32, copy=False),
        "expert0_input": expert0_in.detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False),
        "expert1_input": expert1_in.detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False),
        "expert0_output": e0.detach().cpu().numpy().astype(np.float32, copy=False),
        "expert1_output": e1.detach().cpu().numpy().astype(np.float32, copy=False),
        "packed_expert_outputs": packed.detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False),
        "output": out.detach().cpu().numpy().astype(np.float32, copy=False),
    }


def validation_tolerances(dtype_name: str) -> dict[str, float]:
    normalized = dtype_name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return {"atol": 0.35, "rtol": 0.08}
    return {"atol": 0.05, "rtol": 0.02}


def optional_torch_validation(
    inputs: np.ndarray,
    weights: MoEWeights,
    mode: str,
    dtype_name: str,
    *,
    actual: dict[str, np.ndarray] | None = None,
    quantized_reference: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    try:
        torch_bundle = torch_reference(inputs, weights, mode)
    except ImportError:
        return {"ran": False, "ok": False, "message": "torch not installed"}

    tolerances = validation_tolerances(dtype_name)
    result: dict[str, Any] = {
        "ran": True,
        "ok": True,
        "message": "ok",
        "tolerances": tolerances,
    }

    if quantized_reference is not None:
        quantized_metrics = {}
        for name, expected in quantized_reference.items():
            quantized_metrics[name] = array_error_metrics(
                expected,
                torch_bundle[name],
                atol=tolerances["atol"],
                rtol=tolerances["rtol"],
            )
        result["quantized_reference_vs_torch"] = quantized_metrics

    if actual is not None:
        actual_metrics = {}
        for name, observed in actual.items():
            actual_metrics[name] = array_error_metrics(
                observed,
                torch_bundle[name],
                atol=tolerances["atol"],
                rtol=tolerances["rtol"],
            )
        result["actual_vs_torch"] = actual_metrics
        required_metrics = ["output"]
        result["required_metrics"] = required_metrics
        result["ok"] = all(
            actual_metrics[name]["allclose"] for name in required_metrics
        )
        if not result["ok"]:
            result["message"] = "actual outputs differ from torch reference"

    return result
