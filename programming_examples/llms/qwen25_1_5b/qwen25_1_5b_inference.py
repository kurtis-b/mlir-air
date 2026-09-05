# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Qwen2.5-1.5B BF16 Inference on MLIR-AIR (NPU2).

Unified driver: NPU prefill (28 layers) + NPU decode (KV cache) + NPU LM-head.
Mirrors qwen3_0_6b_inference.py with the Qwen2.5 deltas handled in the prefill
and decode block runners (fused on-device QKV bias instead of QK-norm, dims
emb=1536/hidden=8960/kv_dim=256, head_dim=128, eps=1e-6, tied embeddings,
vocab=151936 LM-head partitioning).

Usage:
    cd build_peano
    python3 ../qwen25_1_5b_inference.py --compile-only
    python3 ../qwen25_1_5b_inference.py --run-only --n-tokens 32 --prompt "..."
    python3 ../qwen25_1_5b_inference.py --run-only --n-tokens 32 --profile
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qwen25_1_5b_weights import LlamaConfig, load_weights, generate_rope_lut
from qwen25_1_5b_cpu_helpers import rms_norm
from shared.infra.driver import (  # noqa: F401
    build_session as _build_session,
    repl_loop as _shared_repl_loop,
    run_once as _shared_run_once,
    tokenize_prompt as _tokenize_prompt,
    generate as _shared_generate,
    run_npu_prefill as _shared_run_npu_prefill,
    run_lm_head as _shared_run_lm_head,
    free_original_weight_numpy as _free_original_weight_numpy,
)
from qwen25_1_5b_prefill import (
    compile_all_kernels,
    run_transformer_block_qwen25,
    preload_prefill_weights,
)
from qwen25_1_5b_decode import (
    compile_decode_kernels,
    run_decode_block,
    _gemv_backend,
    _lm_gemv_backend,
    _LM_N_PARTITIONS,
    _LM_N_PART,
    _GEMV_QO,
    _GEMV_GATEUP,
    _GEMV_DOWN,
)
import qwen25_1_5b_decode as _decode_mod

EPS = 1e-6


# ---------------------------------------------------------------------------
# Streaming-decode helpers (BPE-safe incremental output)
# ---------------------------------------------------------------------------


def generate(*args, **kw):
    """This model's NPU prefill/decode steps, through the shared driver."""
    return _shared_generate(
        *args,
        run_npu_prefill=run_npu_prefill,
        run_npu_decode_step=run_npu_decode_step,
        label="Qwen2.5",
        **kw,
    )


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


@dataclass
class Session:
    config: Any
    seq_len: int
    weights: Any
    tokenizer: Any
    prefill_cache: Any
    decode_cache: Any
    rope_lut_bf16: np.ndarray
    model_variant: str


# ---------------------------------------------------------------------------
# Runtime preparation
# ---------------------------------------------------------------------------


def prepare_runtime(
    prefill_cache, decode_cache, weights, config, seq_len, rope_lut_bf16
):
    """One-time runtime init: transpose decode GEMV weights, tag layer idx,
    pre-load prefill + decode + LM-head BOs."""
    print(f"\n{'='*60}")
    print("Preparing runtime (one-time init, outside profiling scope)...")
    print(f"{'='*60}")
    t0 = time.time()

    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    # 1. Pre-transpose decode GEMV weights. GEMV expects W[out, in]; HF/our
    #    loader stores projections as (in, out) (y = x @ W), so transpose.
    if not hasattr(weights, "_decode_weights_transposed"):
        print("  Pre-transposing weights for decode GEMV...")
        for lw in weights.layers:
            lw._wq_t = np.ascontiguousarray(
                lw.wq.astype(bfloat16).reshape(emb_dim, q_dim).T
            )  # (q_dim, emb)
            lw._wk_t = np.ascontiguousarray(
                lw.wk.astype(bfloat16).reshape(emb_dim, kv_dim).T
            )  # (kv_dim, emb)
            lw._wv_t = np.ascontiguousarray(
                lw.wv.astype(bfloat16).reshape(emb_dim, kv_dim).T
            )  # (kv_dim, emb)
            lw._wo_t = np.ascontiguousarray(
                lw.wo.astype(bfloat16).reshape(q_dim, emb_dim).T
            )  # (emb, q_dim)
            lw._wgate_t = np.ascontiguousarray(
                lw.w_gate.astype(bfloat16).reshape(emb_dim, hidden_dim).T
            )  # (hidden, emb)
            lw._wup_t = np.ascontiguousarray(
                lw.w_up.astype(bfloat16).reshape(emb_dim, hidden_dim).T
            )  # (hidden, emb)
            lw._wdown_t = np.ascontiguousarray(
                lw.w_down.astype(bfloat16).reshape(hidden_dim, emb_dim).T
            )  # (emb, hidden)
        weights._decode_weights_transposed = True

    # 2. Tag layer index for per-layer BO isolation.
    for i, lw in enumerate(weights.layers):
        lw._layer_idx = i

    # 3. Pre-load prefill block weights into per-layer BOs (skipped on the real
    #    prefill pass via static_input_indices).
    preload_prefill_weights(weights, config, prefill_cache, seq_len, rope_lut_bf16)

    # Originals are resident in the prefill per-layer BOs; the block runner
    # rebuilds arg lists via np.asarray(lw.wX).reshape, so swap each for a
    # same-shape zero-stride broadcast (reshape stays a no-op view, buffer
    # collapses to one element) instead of dropping the attribute.
    _free_original_weight_numpy(weights, config)

    # 4. Pre-load decode weights into per-layer BOs + LM-head GEMV.
    _preload_decode_weights(decode_cache, weights, config)

    t_prep = time.time() - t0
    print(f"  Runtime prepared in {t_prep:.1f}s")
    prefill_cache.profiler.preprocessing_s = t_prep
    decode_cache.profiler.preprocessing_s = t_prep


def _run_lm_head(decode_cache, weights, x_normed_bf16, vocab_size):
    """This model's vocabulary split and backend, through the shared driver."""
    return _shared_run_lm_head(
        decode_cache,
        weights,
        x_normed_bf16,
        vocab_size,
        lm_gemv_backend=_lm_gemv_backend,
        n_partitions=_LM_N_PARTITIONS,
        n_part=_LM_N_PART,
    )


def _preload_decode_weights(decode_cache, weights, config):
    """Pre-load all decode block weights into per-layer BOs (skipped on
    subsequent calls via static_input_indices)."""
    if hasattr(weights, "_decode_weights_preloaded_to_bos"):
        return

    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim
    vocab_size = weights.lm_head.shape[0]

    print("  Pre-loading decode weights into per-layer BOs...")
    _was = decode_cache.profiler.enabled
    decode_cache.profiler.enabled = False

    lut_q_dummy = np.zeros(n_heads * head_dim, dtype=bfloat16)
    lut_k_dummy = np.zeros(n_kv_heads * head_dim, dtype=bfloat16)

    for li in range(config.n_layers):
        lw = weights.layers[li]

        # One fused ELF: RMSNorm + Q/K/V GEMV + bias-add + RoPE.
        _decode_mod._fused_bias_rope_gemv_call(
            decode_cache,
            lw,
            config,
            lut_q_dummy,
            lut_k_dummy,
            f"_L{li}",
            np.zeros(emb_dim, dtype=bfloat16),
        )

        # o_gemv: weight static {0}.
        decode_cache.load_and_run(
            "o_gemv",
            _gemv_backend(False, "o_gemv"),
            lw._wo_t,
            np.zeros(q_dim, dtype=bfloat16),
            np.zeros(emb_dim, dtype=bfloat16),
            output_indices=[2],
            static_input_indices={0},
            intermediate_indices={2},
            bo_key=f"o_gemv_L{li}",
        )
        # gate_gemv / up_gemv: weight static {0}.
        decode_cache.load_and_run(
            "gate_gemv",
            _gemv_backend(False, "gate_gemv"),
            lw._wgate_t,
            np.zeros(emb_dim, dtype=bfloat16),
            np.zeros(hidden_dim, dtype=bfloat16),
            output_indices=[2],
            static_input_indices={0},
            intermediate_indices={2},
            bo_key=f"gate_gemv_L{li}",
        )
        decode_cache.load_and_run(
            "up_gemv",
            _gemv_backend(False, "up_gemv"),
            lw._wup_t,
            np.zeros(emb_dim, dtype=bfloat16),
            np.zeros(hidden_dim, dtype=bfloat16),
            output_indices=[2],
            static_input_indices={0},
            intermediate_indices={2},
            bo_key=f"up_gemv_L{li}",
        )
        # down_gemv: weight static {0}.
        decode_cache.load_and_run(
            "down_gemv",
            _gemv_backend(False, "down_gemv"),
            lw._wdown_t,
            np.zeros(hidden_dim, dtype=bfloat16),
            np.zeros(emb_dim, dtype=bfloat16),
            output_indices=[2],
            static_input_indices={0},
            intermediate_indices={2},
            bo_key=f"down_gemv_L{li}",
        )

    # LM-head GEMV weights (19 partitions, n_part=8192).
    weights._lm_weight_parts_gemv = []
    for p in range(_LM_N_PARTITIONS):
        n_start = p * _LM_N_PART
        n_end = min(n_start + _LM_N_PART, vocab_size)
        w = np.zeros((_LM_N_PART, emb_dim), dtype=bfloat16)
        if n_end > n_start:
            w[: n_end - n_start, :] = np.ascontiguousarray(
                weights.lm_head[n_start:n_end, :]
            ).astype(bfloat16)
        weights._lm_weight_parts_gemv.append(w)

    lm_inputs = [np.zeros(emb_dim, dtype=bfloat16)]
    for p in range(_LM_N_PARTITIONS):
        lm_inputs.append(weights._lm_weight_parts_gemv[p])
        lm_inputs.append(np.zeros(_LM_N_PART, dtype=bfloat16))
    decode_cache.load_and_run(
        "lm_head_gemv",
        _lm_gemv_backend(),
        *lm_inputs,
        output_indices=[2 + 2 * p for p in range(_LM_N_PARTITIONS)],
        static_input_indices={1 + 2 * p for p in range(_LM_N_PARTITIONS)},
        intermediate_indices={2 + 2 * p for p in range(_LM_N_PARTITIONS)},
    )

    decode_cache.profiler.enabled = _was
    weights._decode_weights_preloaded_to_bos = True
    print(f"  Pre-loaded {config.n_layers} decode layers + LM Head.")


# ---------------------------------------------------------------------------
# NPU LM-head (19-partition GEMV) — shared by prefill end + decode.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# NPU Prefill with KV cache extraction
# ---------------------------------------------------------------------------


def run_npu_prefill(*args, **kw):
    """This model's transformer block and helpers, through the shared driver."""
    return _shared_run_npu_prefill(
        *args,
        run_transformer_block=run_transformer_block_qwen25,
        rms_norm=rms_norm,
        run_lm_head=_run_lm_head,
        eps=EPS,
        **kw,
    )


# ---------------------------------------------------------------------------
# Single decode step
# ---------------------------------------------------------------------------


def run_npu_decode_step(
    x_decode_bf16,
    weights,
    config,
    decode_cache,
    rope_lut_bf16,
    k_cache,
    v_cache,
    current_pos,
):
    """Run one NPU decode step: 28 blocks + final RMSNorm + LM-head."""
    vocab_size = weights.lm_head.shape[0]
    x = x_decode_bf16.copy()
    for layer_idx in range(config.n_layers):
        t0 = decode_cache.profiler.start_layer()
        x, _ = run_decode_block(
            x,
            weights.layers[layer_idx],
            decode_cache,
            config,
            k_cache[layer_idx],
            v_cache[layer_idx],
            current_pos,
            rope_lut_bf16,
        )
        decode_cache.profiler.end_layer(layer_idx, t0)

    with decode_cache.profiler.time_cpu("final_rms_norm"):
        x_normed = rms_norm(
            x.astype(np.float32).reshape(1, config.emb_dim), weights.final_norm, eps=EPS
        )
    logits = _run_lm_head(
        decode_cache, weights, x_normed.flatten().astype(bfloat16), vocab_size
    )
    next_token = int(np.argmax(logits))
    return next_token, logits


# ---------------------------------------------------------------------------
# Full generation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

MODEL_CHOICES = {"base": "Qwen/Qwen2.5-1.5B", "instruct": "Qwen/Qwen2.5-1.5B-Instruct"}


def build_session(args) -> Session:
    """Build this model's Session through the shared driver."""
    return _build_session(
        args,
        config_cls=LlamaConfig,
        session_cls=Session,
        model_choices=MODEL_CHOICES,
        load_weights=load_weights,
        generate_rope_lut=generate_rope_lut,
        compile_all_kernels=compile_all_kernels,
        compile_decode_kernels=compile_decode_kernels,
        prepare_runtime=prepare_runtime,
    )


def run_once(session, prompt_text, **kw):
    """This model's `generate`, through the shared driver."""
    return _shared_run_once(session, prompt_text, generate=generate, **kw)


def repl_loop(session, args):
    """This model's `generate`, through the shared driver."""
    _shared_repl_loop(session, args, generate=generate)


def _print_one_shot_output(session, prompt_text, generated, prompt_len_actual):
    print(f"\n{'='*60}")
    if session.model_variant == "instruct":
        response = session.tokenizer.decode(generated, skip_special_tokens=True).strip()
        print(f"Q: {prompt_text}")
        print(f"A: {response}")
    else:
        prompt_tokens = _tokenize_prompt(session, prompt_text)
        all_tokens = prompt_tokens[:prompt_len_actual] + generated
        print("Generated text:")
        print(session.tokenizer.decode(all_tokens))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen2.5-1.5B Inference (NPU)")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--run-only", action="store_true")
    parser.add_argument("--n-tokens", type=int, default=10)
    parser.add_argument("--profile", action="store_true")
    # Default: NPU head-first FlashAttention. Pass --cpu-attn to fall back to
    # the FP32 host attention reference.
    parser.add_argument("--cpu-attn", action="store_true", default=False)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--prompt", type=str, default="What is the capital of France?")
    parser.add_argument(
        "--model", type=str, choices=["base", "instruct"], default="instruct"
    )
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    if args.interactive:
        if args.compile_only:
            parser.error("--interactive cannot be combined with --compile-only")
        if not args.run_only:
            parser.error("--interactive requires --run-only")
        args.profile = False

    session = build_session(args)

    if args.interactive:
        repl_loop(session, args)
    else:
        generated, plen = run_once(
            session,
            args.prompt,
            n_tokens=args.n_tokens,
            profile=args.profile,
            cpu_attn=args.cpu_attn,
        )
        _print_one_shot_output(session, args.prompt, generated, plen)
