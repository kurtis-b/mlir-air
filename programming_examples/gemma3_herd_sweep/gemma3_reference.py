#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU-only Gemma3 synthetic references for model-loop bring-up."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
from ml_dtypes import bfloat16

from common import attention_reference, q4nx_dequant_blocks_reference
from gemma3_config import Gemma3LayerConfig, Gemma3TextConfig, describe_kernel_sequence, synthetic_text_config
from gemma3_weights import Gemma3LayerWeights, Gemma3SyntheticWeights, Q4NXProjectionWeights, synthetic_weights


@dataclass
class Gemma3KVCache:
    k: list[np.ndarray]
    v: list[np.ndarray]
    cache_len: list[int]
    token_base: list[int]
    layout_version: str = "synthetic-seq-kvgroup-head"

    @classmethod
    def allocate(cls, config: Gemma3TextConfig) -> "Gemma3KVCache":
        k = [
            np.zeros((config.kv_len, config.n_kv_heads, config.head_dim), dtype=bfloat16)
            for _ in range(config.n_layers)
        ]
        v = [
            np.zeros((config.kv_len, config.n_kv_heads, config.head_dim), dtype=bfloat16)
            for _ in range(config.n_layers)
        ]
        return cls(k=k, v=v, cache_len=[0] * config.n_layers, token_base=[0] * config.n_layers)

    def append(self, layer_index: int, k_new: np.ndarray, v_new: np.ndarray) -> None:
        rows = k_new.shape[0]
        start = self.cache_len[layer_index]
        end = start + rows
        if end > self.k[layer_index].shape[0]:
            raise ValueError("KV cache append exceeds configured kv_len")
        self.k[layer_index][start:end] = k_new.astype(bfloat16)
        self.v[layer_index][start:end] = v_new.astype(bfloat16)
        self.cache_len[layer_index] = end

    def layer_view(self, layer_index: int, window_len: int = 0) -> tuple[np.ndarray, np.ndarray, int]:
        end = self.cache_len[layer_index]
        start = max(0, end - window_len) if window_len > 0 else 0
        return self.k[layer_index][start:end], self.v[layer_index][start:end], start


@dataclass
class LayerReferenceResult:
    output: np.ndarray
    intermediates: dict[str, np.ndarray] = field(default_factory=dict)


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    xf = np.asarray(x, dtype=np.float32)
    wf = np.asarray(weight, dtype=np.float32)
    rms = np.sqrt(np.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return ((xf / rms) * wf).astype(bfloat16)


def qk_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    return rms_norm(x, weight, eps)


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    xf = np.asarray(x, dtype=np.float32)
    inner = 0.7978845608 * (xf + 0.044715 * xf * xf * xf)
    return (0.5 * xf * (1.0 + np.tanh(inner))).astype(bfloat16)


def geglu(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    return (gelu_tanh(gate).astype(np.float32) * up.astype(np.float32)).astype(bfloat16)


def bf16_mm_reference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a.astype(np.float32) @ b.astype(np.float32)).astype(bfloat16)


def generate_rope_lut(config: Gemma3TextConfig, seq_len: int) -> np.ndarray:
    half = config.head_dim // 2
    idx = np.arange(half, dtype=np.float32)
    inv_freq = 1.0 / (config.rope_base ** (idx / float(half)))
    positions = np.arange(seq_len, dtype=np.float32)[:, None]
    angles = positions * inv_freq[None, :]
    lut = np.concatenate([np.cos(angles), np.sin(angles)], axis=1)
    return lut.astype(bfloat16)


def apply_rope_halfsplit(x: np.ndarray, lut: np.ndarray) -> np.ndarray:
    xf = np.asarray(x, dtype=np.float32)
    lf = np.asarray(lut, dtype=np.float32)
    if xf.shape[-1] != lf.shape[-1]:
        raise ValueError("RoPE LUT head_dim must match input head_dim")
    half = xf.shape[-1] // 2
    cos_vals = lf[..., :half]
    sin_vals = lf[..., half:]
    x1 = xf[..., :half]
    x2 = xf[..., half:]
    out = np.empty_like(xf)
    out[..., :half] = x1 * cos_vals - x2 * sin_vals
    out[..., half:] = x1 * sin_vals + x2 * cos_vals
    return out.astype(bfloat16)


def q4nx_projection_reference(x: np.ndarray, projection: Q4NXProjectionWeights) -> np.ndarray:
    projection.validate()
    weight = q4nx_dequant_blocks_reference(
        projection.packed,
        projection.scale,
        projection.min_offset,
        projection.rows,
        projection.cols,
    )
    if weight.shape != (projection.out_dim, projection.in_dim):
        raise ValueError(f"{projection.name}: dequantized shape mismatch {weight.shape}")
    xf = np.asarray(x, dtype=np.float32)
    if xf.shape[-1] != projection.in_dim:
        raise ValueError(f"{projection.name}: input last dimension mismatch")
    out = xf.reshape(-1, projection.in_dim) @ weight.astype(np.float32).T
    return out.reshape(xf.shape[:-1] + (projection.out_dim,)).astype(bfloat16)


def _gqa_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    config: Gemma3TextConfig,
    *,
    query_base: int,
    causal: bool,
    window_len: int,
) -> np.ndarray:
    q = np.asarray(q, dtype=bfloat16)
    k = np.asarray(k, dtype=bfloat16)
    v = np.asarray(v, dtype=bfloat16)
    out = np.empty((q.shape[0], config.n_heads, config.head_dim), dtype=bfloat16)
    for head in range(config.n_heads):
        kv_group = head // config.heads_per_kv
        out[:, head, :] = attention_reference(
            q[:, head, :],
            k[:, kv_group, :],
            v[:, kv_group, :],
            query_base=query_base,
            causal=causal,
            window_len=window_len,
        )
    return out.reshape(q.shape[0], config.emb_dim)


def _window_for_layer(layer_cfg: Gemma3LayerConfig) -> int:
    return layer_cfg.window_len if layer_cfg.attention_kind == "local_swa" else 0


def _project_qkv(
    x: np.ndarray,
    layer_weights: Gemma3LayerWeights,
    config: Gemma3TextConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = q4nx_projection_reference(x, layer_weights.q_proj).reshape(
        x.shape[0], config.n_heads, config.head_dim
    )
    k = q4nx_projection_reference(x, layer_weights.k_proj).reshape(
        x.shape[0], config.n_kv_heads, config.head_dim
    )
    v = q4nx_projection_reference(x, layer_weights.v_proj).reshape(
        x.shape[0], config.n_kv_heads, config.head_dim
    )
    return q, k, v


def _apply_qk_norm_and_rope(
    q: np.ndarray,
    k: np.ndarray,
    layer_weights: Gemma3LayerWeights,
    rope_lut: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    qn = qk_norm(q, layer_weights.q_norm)
    kn = qk_norm(k, layer_weights.k_norm)
    q_rope = apply_rope_halfsplit(qn, rope_lut[:, None, :])
    k_rope = apply_rope_halfsplit(kn, rope_lut[:, None, :])
    return q_rope, k_rope


def ffn_reference(x: np.ndarray, layer_weights: Gemma3LayerWeights) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    normed = rms_norm(x, layer_weights.ffn_norm)
    gate = q4nx_projection_reference(normed, layer_weights.gate_proj)
    up = q4nx_projection_reference(normed, layer_weights.up_proj)
    activated = geglu(gate, up)
    down = q4nx_projection_reference(activated, layer_weights.down_proj)
    return (x.astype(np.float32) + down.astype(np.float32)).astype(bfloat16), {
        "ffn_norm": normed,
        "gate": gate,
        "up": up,
        "mlp_activation": activated,
        "down": down,
    }


def prefill_layer_reference(
    x: np.ndarray,
    weights: Gemma3SyntheticWeights,
    cache: Gemma3KVCache,
    layer_cfg: Gemma3LayerConfig,
    *,
    token_base: int = 0,
) -> LayerReferenceResult:
    config = weights.config
    layer_weights = weights.layers[layer_cfg.layer_index]
    if x.shape[-1] != config.emb_dim:
        raise ValueError("prefill input emb_dim mismatch")
    if x.shape[0] > config.kv_len:
        raise ValueError("prefill sequence exceeds synthetic kv_len")

    intermediates: dict[str, np.ndarray] = {}
    rope_lut = generate_rope_lut(config, token_base + x.shape[0])[token_base:]
    normed = rms_norm(x, layer_weights.attn_norm)
    q, k, v = _project_qkv(normed, layer_weights, config)
    q_rope, k_rope = _apply_qk_norm_and_rope(q, k, layer_weights, rope_lut)
    cache.append(layer_cfg.layer_index, k_rope, v)
    k_view, v_view, view_base = cache.layer_view(layer_cfg.layer_index, _window_for_layer(layer_cfg))
    attn = _gqa_attention(
        q_rope,
        k_view,
        v_view,
        config,
        query_base=token_base - view_base,
        causal=layer_cfg.causal,
        window_len=0,
    )
    proj = q4nx_projection_reference(attn, layer_weights.o_proj)
    residual = (x.astype(np.float32) + proj.astype(np.float32)).astype(bfloat16)
    output, ffn_intermediates = ffn_reference(residual, layer_weights)
    intermediates.update(
        {
            "attn_norm": normed,
            "q": q,
            "k": k,
            "v": v,
            "q_roped": q_rope,
            "k_roped": k_rope,
            "attention": attn,
            "o_proj": proj,
            "residual_attn": residual,
        }
    )
    intermediates.update(ffn_intermediates)
    return LayerReferenceResult(output=output, intermediates=intermediates)


def decode_layer_reference(
    x: np.ndarray,
    weights: Gemma3SyntheticWeights,
    cache: Gemma3KVCache,
    layer_cfg: Gemma3LayerConfig,
    *,
    current_pos: int,
) -> LayerReferenceResult:
    config = weights.config
    layer_weights = weights.layers[layer_cfg.layer_index]
    if x.shape != (config.emb_dim,):
        raise ValueError("decode input must be one emb_dim vector")
    x2 = x.reshape(1, config.emb_dim)
    rope_lut = generate_rope_lut(config, current_pos + 1)[current_pos: current_pos + 1]
    normed = rms_norm(x2, layer_weights.attn_norm)
    q, k, v = _project_qkv(normed, layer_weights, config)
    q_rope, k_rope = _apply_qk_norm_and_rope(q, k, layer_weights, rope_lut)
    cache.append(layer_cfg.layer_index, k_rope, v)
    k_view, v_view, view_base = cache.layer_view(layer_cfg.layer_index, _window_for_layer(layer_cfg))
    attn = _gqa_attention(
        q_rope,
        k_view,
        v_view,
        config,
        query_base=current_pos - view_base,
        causal=layer_cfg.causal,
        window_len=0,
    )
    proj = q4nx_projection_reference(attn, layer_weights.o_proj)
    residual = (x2.astype(np.float32) + proj.astype(np.float32)).astype(bfloat16)
    output, ffn_intermediates = ffn_reference(residual, layer_weights)
    intermediates = {
        "attn_norm": normed,
        "q": q,
        "k": k,
        "v": v,
        "q_roped": q_rope,
        "k_roped": k_rope,
        "attention": attn,
        "o_proj": proj,
        "residual_attn": residual,
    }
    intermediates.update(ffn_intermediates)
    return LayerReferenceResult(output=output.reshape(config.emb_dim), intermediates=intermediates)


def logits_reference(x: np.ndarray, weights: Gemma3SyntheticWeights) -> np.ndarray:
    normed = rms_norm(x, weights.final_norm)
    return normed.astype(np.float32) @ weights.lm_head.astype(np.float32).T


def synthetic_token_ids(config: Gemma3TextConfig, seq_len: int) -> np.ndarray:
    return (np.arange(seq_len, dtype=np.int64) * 17 + 3) % config.vocab_size


def embedding_reference(token_ids: np.ndarray, weights: Gemma3SyntheticWeights) -> np.ndarray:
    return weights.embed_table[np.asarray(token_ids, dtype=np.int64)].astype(bfloat16)


def run_synthetic_prefill_decode_smoke() -> dict[str, object]:
    config = synthetic_text_config(n_layers=2)
    weights = synthetic_weights(config, seed=42)
    cache = Gemma3KVCache.allocate(config)
    token_ids = synthetic_token_ids(config, config.q_chunk)
    x = embedding_reference(token_ids, weights)
    for layer_cfg in config.layers:
        result = prefill_layer_reference(x, weights, cache, layer_cfg)
        x = result.output
    logits = logits_reference(x[-1], weights)
    decode_x = embedding_reference(np.array([7], dtype=np.int64), weights)[0]
    for layer_cfg in config.layers:
        result = decode_layer_reference(
            decode_x, weights, cache, layer_cfg, current_pos=config.q_chunk
        )
        decode_x = result.output
    decode_logits = logits_reference(decode_x, weights)
    return {
        "config": config,
        "weights": weights,
        "cache": cache,
        "prefill_logits_shape": logits.shape,
        "decode_logits_shape": decode_logits.shape,
        "prefill_checksum": float(np.sum(logits.astype(np.float32))),
        "decode_checksum": float(np.sum(decode_logits.astype(np.float32))),
    }


def _self_test() -> None:
    smoke = run_synthetic_prefill_decode_smoke()
    config = smoke["config"]
    cache = smoke["cache"]
    assert smoke["prefill_logits_shape"] == (config.vocab_size,)
    assert smoke["decode_logits_shape"] == (config.vocab_size,)
    assert cache.cache_len == [config.q_chunk + 1, config.q_chunk + 1]
    assert config.layers[0].attention_kind == "local_swa"
    assert config.layers[1].attention_kind == "global_full"
    sequence = describe_kernel_sequence(config)
    assert "prefill:L0:q4nx" in sequence
    assert "decode:L1:flowkv" in sequence
    print(sequence)
    print(f"prefill_checksum={smoke['prefill_checksum']:.6f}")
    print(f"decode_checksum={smoke['decode_checksum']:.6f}")
    print("GEMMA3_CPU_REFERENCE_SELF_TEST: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma3 synthetic CPU reference")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--print-sequence", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return 0
    if args.print_sequence:
        print(describe_kernel_sequence(synthetic_text_config(n_layers=2)))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
