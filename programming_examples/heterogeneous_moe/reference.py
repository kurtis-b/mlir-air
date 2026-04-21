# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kernels import KernelConfig
from numerics import quantize_array, quantize_scalar


@dataclass(frozen=True)
class MoEWeights:
    router: np.ndarray
    expert0_w1: np.ndarray
    expert0_w2: np.ndarray
    expert1_w1: np.ndarray
    expert1_w2: np.ndarray


def random_inputs(cfg: KernelConfig, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((cfg.batch_tokens, cfg.hidden_size), dtype=np.float32)
    return quantize_array(data, cfg.dtype)


def random_weights(cfg: KernelConfig, seed: int) -> MoEWeights:
    rng = np.random.default_rng(seed)

    def arr(shape: tuple[int, ...]) -> np.ndarray:
        return quantize_array(rng.standard_normal(shape, dtype=np.float32), cfg.dtype)

    return MoEWeights(
        router=arr((cfg.hidden_size, 2)),
        expert0_w1=arr((cfg.hidden_size, cfg.ffn_size)),
        expert0_w2=arr((cfg.ffn_size, cfg.hidden_size)),
        expert1_w1=arr((cfg.hidden_size, cfg.ffn_size)),
        expert1_w2=arr((cfg.ffn_size, cfg.hidden_size)),
    )


def router_logits(inputs: np.ndarray, weights: np.ndarray, dtype_name: str) -> np.ndarray:
    logits = np.zeros((inputs.shape[0], weights.shape[1]), dtype=np.float32)
    for row in range(inputs.shape[0]):
        for col in range(weights.shape[1]):
            acc = quantize_scalar(0.0, dtype_name)
            for k in range(inputs.shape[1]):
                prod = quantize_scalar(float(inputs[row, k]) * float(weights[k, col]), dtype_name)
                acc = quantize_scalar(acc + prod, dtype_name)
            logits[row, col] = acc
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
    out[np.arange(probs.shape[0]), indices] = quantize_array(np.asarray([1.0], dtype=np.float32), dtype_name)[0]
    return out


def routed_inputs(inputs: np.ndarray, weights: np.ndarray, dtype_name: str) -> tuple[np.ndarray, np.ndarray]:
    expert0 = quantize_array(inputs.astype(np.float32) * weights[:, 0:1].astype(np.float32), dtype_name)
    expert1 = quantize_array(inputs.astype(np.float32) * weights[:, 1:2].astype(np.float32), dtype_name)
    return expert0, expert1


def expert_mlp(inputs: np.ndarray, w1: np.ndarray, w2: np.ndarray, dtype_name: str) -> np.ndarray:
    hidden = np.zeros((inputs.shape[0], w1.shape[1]), dtype=np.float32)
    for row in range(inputs.shape[0]):
        for col in range(w1.shape[1]):
            acc = quantize_scalar(0.0, dtype_name)
            for k in range(inputs.shape[1]):
                prod = quantize_scalar(float(inputs[row, k]) * float(w1[k, col]), dtype_name)
                acc = quantize_scalar(acc + prod, dtype_name)
            hidden[row, col] = quantize_scalar(max(acc, 0.0), dtype_name)

    output = np.zeros((inputs.shape[0], w2.shape[1]), dtype=np.float32)
    for row in range(inputs.shape[0]):
        for col in range(w2.shape[1]):
            acc = quantize_scalar(0.0, dtype_name)
            for k in range(w2.shape[0]):
                prod = quantize_scalar(float(hidden[row, k]) * float(w2[k, col]), dtype_name)
                acc = quantize_scalar(acc + prod, dtype_name)
            output[row, col] = acc
    return quantize_array(output, dtype_name)


def pack_expert_weights(w1: np.ndarray, w2: np.ndarray, dtype_name: str) -> np.ndarray:
    packed = np.empty((w1.shape[0], w1.shape[1] + w2.shape[0]), dtype=np.float32)
    packed[:, : w1.shape[1]] = w1
    packed[:, w1.shape[1] :] = w2.T
    return quantize_array(packed, dtype_name)


def expert_mlp_packed(inputs: np.ndarray, packed_weights: np.ndarray, dtype_name: str) -> np.ndarray:
    split = packed_weights.shape[1] // 2
    w1 = packed_weights[:, :split]
    w2 = packed_weights[:, split:].T
    return expert_mlp(inputs, w1, w2, dtype_name)


def pack_expert_outputs(expert0: np.ndarray, expert1: np.ndarray, dtype_name: str) -> np.ndarray:
    return quantize_array(np.concatenate((expert0, expert1), axis=1), dtype_name)


def aggregate_outputs(expert0: np.ndarray, expert1: np.ndarray, weights: np.ndarray, dtype_name: str) -> np.ndarray:
    out = np.zeros_like(expert0, dtype=np.float32)
    for row in range(expert0.shape[0]):
        for col in range(expert0.shape[1]):
            lhs = quantize_scalar(float(expert0[row, col]) * float(weights[row, 0]), dtype_name)
            rhs = quantize_scalar(float(expert1[row, col]) * float(weights[row, 1]), dtype_name)
            out[row, col] = quantize_scalar(lhs + rhs, dtype_name)
    return quantize_array(out, dtype_name)


def aggregate_packed_outputs(experts: np.ndarray, weights: np.ndarray, dtype_name: str) -> np.ndarray:
    split = experts.shape[1] // 2
    return aggregate_outputs(experts[:, :split], experts[:, split:], weights, dtype_name)


def run_reference(cfg: KernelConfig, inputs: np.ndarray, weights: MoEWeights, mode: str) -> dict[str, np.ndarray]:
    logits = router_logits(inputs, weights.router, cfg.dtype)
    routes = topk_weights(logits, mode, cfg.dtype)
    expert0_in, expert1_in = routed_inputs(inputs, routes, cfg.dtype)
    expert0_out = expert_mlp(expert0_in, weights.expert0_w1, weights.expert0_w2, cfg.dtype)
    expert1_out = expert_mlp(expert1_in, weights.expert1_w1, weights.expert1_w2, cfg.dtype)
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


def optional_torch_validation(inputs: np.ndarray, weights: MoEWeights, mode: str) -> tuple[bool, str]:
    try:
        import torch
    except ImportError:
        return False, "torch not installed"

    x = torch.tensor(inputs.astype(np.float32))
    router_w = torch.tensor(weights.router.astype(np.float32))
    logits = x @ router_w
    probs = torch.softmax(logits, dim=-1)
    if mode == "top1":
        indices = torch.argmax(probs, dim=-1)
        routed = torch.zeros_like(probs)
        routed.scatter_(1, indices.unsqueeze(-1), 1.0)
    else:
        routed = probs

    expert0_in = x * routed[:, 0:1]
    expert1_in = x * routed[:, 1:2]
    e0 = torch.relu(expert0_in @ torch.tensor(weights.expert0_w1.astype(np.float32))) @ torch.tensor(
        weights.expert0_w2.astype(np.float32)
    )
    e1 = torch.relu(expert1_in @ torch.tensor(weights.expert1_w1.astype(np.float32))) @ torch.tensor(
        weights.expert1_w2.astype(np.float32)
    )
    out = e0 * routed[:, 0:1] + e1 * routed[:, 1:2]
    finite = bool(torch.isfinite(out).all())
    return finite, "ok" if finite else "non-finite output"
