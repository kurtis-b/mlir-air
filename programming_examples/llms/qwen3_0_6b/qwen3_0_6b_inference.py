# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Qwen3-0.6B BF16 Inference on MLIR-AIR (NPU2).

Unified driver: NPU prefill (28 layers) + NPU decode (KV cache) + NPU LM-head.
Mirrors llama32_1b_inference.py with the Qwen3 deltas handled in the prefill
and decode block runners (QK-norm split, decoupled dims, eps=1e-6,
vocab=151936 LM-head partitioning).

Usage:
    cd build_peano
    python3 ../qwen3_0_6b_inference.py --compile-only
    python3 ../qwen3_0_6b_inference.py --run-only --n-tokens 32 --prompt "..."
    python3 ../qwen3_0_6b_inference.py --run-only --n-tokens 32 --profile
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qwen3_0_6b_weights import LlamaConfig, load_weights, generate_rope_lut
from qwen3_0_6b_cpu_helpers import rms_norm
from shared.infra.cache import KernelCache, Profiler
from qwen3_0_6b_prefill import (
    compile_all_kernels,
    run_transformer_block_qwen3,
    preload_prefill_weights,
)
from qwen3_0_6b_decode import (
    compile_decode_kernels,
    run_decode_block,
    run_rms_qkv4,
    prep_rms_qkv4_weights,
    _o_gemv_ffn_backend,
    _lm_gemv_backend,
    _LM_PARTS,
    lm_head_partition_slices,
    _W4_DECODE,
    require_decode_artifacts,
    _run_o_gemv_ffn_int4,
)

EPS = 1e-6


# ---------------------------------------------------------------------------
# Streaming-decode helpers (BPE-safe incremental output)
# ---------------------------------------------------------------------------


class _StreamState:
    def __init__(self) -> None:
        self.printed_len: int = 0


def _delta_text(tokenizer: Any, ids: list, state: _StreamState) -> str:
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    delta = decoded[state.printed_len :]
    state.printed_len = len(decoded)
    return delta


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
    # `[2026-08-26]` queue item 24: the ONE place the production driver, the
    # verify adapter and `model_adapter.prepare` all pass through -- so the
    # "this cache was built for the other precision" refusal is stated once,
    # before any weight transposition or device work.
    require_decode_artifacts(decode_cache)
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
            )  # (q_dim, emb_dim)
            lw._wk_t = np.ascontiguousarray(
                lw.wk.astype(bfloat16).reshape(emb_dim, kv_dim).T
            )  # (kv_dim, emb_dim)
            lw._wv_t = np.ascontiguousarray(
                lw.wv.astype(bfloat16).reshape(emb_dim, kv_dim).T
            )  # (kv_dim, emb_dim)
            # `[2026-08-26]` doc 56 H2b: under w4_decode the O+FFN decode
            # weights are the packed int4 BOs the loader attached
            # (`_wo_packed` / `_wgateup_packed` / `_wdown_packed`); the bf16
            # transposes below would copy the dequantized prefill weights for
            # nothing (~630 MB host RAM).
            if not _W4_DECODE:
                lw._wo_t = np.ascontiguousarray(
                    lw.wo.astype(bfloat16).reshape(q_dim, emb_dim).T
                )  # (emb_dim, q_dim) DECOUPLED
                lw._wgate_t = np.ascontiguousarray(
                    lw.w_gate.astype(bfloat16).reshape(emb_dim, hidden_dim).T
                )  # (hidden, emb)
                lw._wup_t = np.ascontiguousarray(
                    lw.w_up.astype(bfloat16).reshape(emb_dim, hidden_dim).T
                )  # (hidden, emb)
                lw._wdown_t = np.ascontiguousarray(
                    lw.w_down.astype(bfloat16).reshape(hidden_dim, emb_dim).T
                )  # (emb, hidden)
            # 4-launch QKV stage: packed [wq; wk; wv] + tiled QK-norm weight,
            # then drop the three separate transposes (one resident copy).
            prep_rms_qkv4_weights(lw, config)
            del lw._wq_t, lw._wk_t, lw._wv_t
        weights._decode_weights_transposed = True

    # 2. Tag layer index for per-layer BO isolation.
    for i, lw in enumerate(weights.layers):
        lw._layer_idx = i

    # 3. Pre-load prefill block weights into per-layer BOs (skipped on the real
    #    prefill pass via static_input_indices — keeps weight host->device
    #    writes out of the timed prefill region). Mirrors llama.
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


def _free_original_weight_numpy(weights, config):
    """Collapse host numpy originals to zero-stride broadcasts after prefill
    preload. Weights are resident in the prefill BOs and passed as static
    inputs, so only their dtype/shape metadata is read afterward."""
    import gc

    z = np.zeros((), dtype=bfloat16)
    for layer_idx in range(config.n_layers):
        lw = weights.layers[layer_idx]
        for attr in ("wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down"):
            a = getattr(lw, attr, None)
            if a is not None and getattr(a, "size", 0) > 1:
                setattr(lw, attr, np.broadcast_to(z, a.shape))
    gc.collect()


def _preload_decode_weights(decode_cache, weights, config):
    """Pre-load all decode block weights into per-layer BOs (skipped on
    subsequent calls via static_input_indices). Mirrors llama."""
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

    lut_dummy = np.zeros((n_heads + n_kv_heads) * head_dim, dtype=bfloat16)

    for li in range(config.n_layers):
        lw = weights.layers[li]

        # Fused decode ELF warmup (4 launches: RMSNorm+QKV GEMV+QK-norm+RoPE).
        # The LUT (arg 7) is position-dependent -> NOT static.
        run_rms_qkv4(decode_cache, lw, np.zeros(emb_dim, dtype=bfloat16), lut_dummy, config, li)

        # o_gemv_ffn: packed RMS-input buffer (both paths); interleaved
        # w_gateup only on the bf16 path (w4_decode preloads the packed BOs
        # the loader attached, through the SAME run helper the decode loop
        # uses -- one owner of the arg layout).
        packed = np.empty((2, emb_dim), dtype=bfloat16)
        packed[0] = 0.0
        packed[1] = lw.ffn_norm.reshape(emb_dim).astype(bfloat16)
        lw._packed_rms_buf = packed

        if _W4_DECODE:
            z_q = np.zeros(q_dim, dtype=bfloat16)
            z_emb = np.zeros(emb_dim, dtype=bfloat16)
            _run_o_gemv_ffn_int4(z_q, z_emb, lw, config, decode_cache, li)
            continue

        wgateup = np.empty((2 * hidden_dim, emb_dim), dtype=bfloat16)
        wgateup[0::2] = lw._wgate_t
        wgateup[1::2] = lw._wup_t
        lw._wgateup_t = wgateup
        del lw._wgate_t
        del lw._wup_t

        z_emb = np.zeros(emb_dim, dtype=bfloat16)
        z_q = np.zeros(q_dim, dtype=bfloat16)
        z_hidden = np.zeros(hidden_dim, dtype=bfloat16)
        z_hidden_emb = np.zeros((hidden_dim, emb_dim), dtype=bfloat16)

        decode_cache.load_and_run(
            "o_gemv_ffn",
            _o_gemv_ffn_backend(),
            lw._wo_t,  # arg0 wo (static, decoupled)
            z_q,  # arg1 attn_out (q_dim)
            z_emb,  # arg2 (dead)
            z_emb,  # arg3 x_residual
            z_emb,  # arg4 (dead)
            z_emb,  # arg5 (dead)
            lw._packed_rms_buf,  # arg6 packed (static)
            lw._wgateup_t,  # arg7 gate/up (static)
            z_hidden,  # arg8 (dead)
            z_hidden_emb,  # arg9 (dead)
            z_hidden,  # arg10 (dead)
            z_hidden,  # arg11 swiglu
            lw._wdown_t,  # arg12 wdown (static)
            z_emb,  # arg13 (dead)
            z_emb,  # arg14 output
            output_indices=[14],
            static_input_indices={0, 6, 7, 12},
            intermediate_indices={2, 4, 5, 8, 9, 10, 11, 13, 14},
            bo_key=f"o_gemv_ffn_L{li}",
        )

    # LM-head GEMV weights (19 partitions, n_part=8192).
    weights._lm_weight_parts_gemv = []
    for rows, (n_start, n_end) in zip(_LM_PARTS, lm_head_partition_slices(vocab_size)):
        w = np.zeros((rows, emb_dim), dtype=bfloat16)
        if n_end > n_start:
            w[: n_end - n_start, :] = np.ascontiguousarray(
                weights.lm_head[n_start:n_end, :]
            ).astype(bfloat16)
        weights._lm_weight_parts_gemv.append(w)

    n_parts = len(_LM_PARTS)
    lm_inputs = [np.zeros(emb_dim, dtype=bfloat16)]
    for p, rows in enumerate(_LM_PARTS):
        lm_inputs.append(weights._lm_weight_parts_gemv[p])
        lm_inputs.append(np.zeros(rows, dtype=bfloat16))
    decode_cache.load_and_run(
        "lm_head_gemv",
        _lm_gemv_backend(),
        *lm_inputs,
        output_indices=[2 + 2 * p for p in range(n_parts)],
        static_input_indices={1 + 2 * p for p in range(n_parts)},
        intermediate_indices={2 + 2 * p for p in range(n_parts)},
    )

    decode_cache.profiler.enabled = _was
    weights._decode_weights_preloaded_to_bos = True
    print(f"  Pre-loaded {config.n_layers} decode layers + LM Head.")


# ---------------------------------------------------------------------------
# NPU LM-head (19-partition GEMV) — shared by prefill end + decode.
# ---------------------------------------------------------------------------


def _run_lm_head(decode_cache, weights, x_normed_bf16, vocab_size):
    n_parts = len(_LM_PARTS)
    lm_inputs = [x_normed_bf16.flatten().astype(bfloat16)]
    out_idx = []
    for p, rows in enumerate(_LM_PARTS):
        lm_inputs.append(weights._lm_weight_parts_gemv[p])
        lm_inputs.append(np.zeros(rows, dtype=bfloat16))
        out_idx.append(2 + 2 * p)
    res = decode_cache.load_and_run(
        "lm_head_gemv",
        _lm_gemv_backend(),
        *lm_inputs,
        output_indices=out_idx,
        static_input_indices={1 + 2 * p for p in range(n_parts)},
        intermediate_indices={2 + 2 * p for p in range(n_parts)},
    )
    logits = np.zeros(vocab_size, dtype=np.float32)
    for p, (n_start, n_end) in enumerate(lm_head_partition_slices(vocab_size)):
        logits[n_start:n_end] = res[2 + 2 * p][: n_end - n_start].astype(np.float32)
    return logits


# ---------------------------------------------------------------------------
# NPU Prefill with KV cache extraction
# ---------------------------------------------------------------------------


def run_npu_prefill(
    token_ids,
    weights,
    config,
    prefill_cache,
    decode_cache,
    rope_lut_bf16,
    max_seq,
    tokenizer,
    cpu_attn=True,
    profile=False,
    quiet=False,
):
    """Run NPU prefill (28 Qwen3 layers) and extract KV cache.

    Returns: (prefill_token, logits_row, k_cache, v_cache, prompt_len).
    K cache stores k_roped (AFTER QK-norm AND RoPE); V stores raw projection.
    """
    seq_len = len(token_ids)
    emb_dim = config.emb_dim
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    vocab_size = weights.lm_head.shape[0]

    k_cache = np.zeros((config.n_layers, n_kv_heads, max_seq, head_dim), dtype=bfloat16)
    v_cache = np.zeros((config.n_layers, n_kv_heads, max_seq, head_dim), dtype=bfloat16)

    with prefill_cache.profiler.time_cpu("embed_lookup"):
        x_bf16 = weights.embed_table[token_ids].astype(np.float32).astype(bfloat16)

    if not quiet:
        print(f"Running NPU prefill ({config.n_layers} layers, seq_len={seq_len})...")
    t_start = time.time()

    for layer_idx in range(config.n_layers):
        t0 = prefill_cache.profiler.start_layer()
        x_bf16, inter = run_transformer_block_qwen3(
            x_bf16,
            weights.layers[layer_idx],
            rope_lut_bf16,
            config,
            prefill_cache,
            layer_idx=layer_idx,
            cpu_attn=cpu_attn,
            verbose=profile,
        )
        with prefill_cache.profiler.time_cpu("kv_append"):  # the plan's stage name (was kv_cache_extract)
            k_roped = inter["k_roped"]
            v_raw = inter["v"]
            k_cache[layer_idx, :, :seq_len, :] = (
                k_roped.astype(bfloat16)
                .reshape(seq_len, n_kv_heads, head_dim)
                .transpose(1, 0, 2)
            )
            v_cache[layer_idx, :, :seq_len, :] = (
                v_raw.astype(bfloat16)
                .reshape(seq_len, n_kv_heads, head_dim)
                .transpose(1, 0, 2)
            )
        prefill_cache.profiler.end_layer(layer_idx, t0)

    # Final RMSNorm (eps=1e-6) on the prediction-position row + NPU LM-head.
    prompt_len = len([t for t in token_ids if t != tokenizer.eos_token_id])
    pred_pos = prompt_len - 1
    with prefill_cache.profiler.time_cpu("final_rms_norm"):
        last_hidden = np.asarray(x_bf16, dtype=np.float32)[pred_pos : pred_pos + 1]
        last_normed = (
            rms_norm(last_hidden, weights.final_norm, eps=EPS)
            .flatten()
            .astype(bfloat16)
        )

    logits_row = _run_lm_head(decode_cache, weights, last_normed, vocab_size)
    prefill_token = int(np.argmax(logits_row))

    t_prefill = time.time() - t_start
    if not quiet:
        print(f"NPU prefill done in {t_prefill:.2f}s. First token: {prefill_token}")
    return prefill_token, logits_row, k_cache, v_cache, prompt_len


# ---------------------------------------------------------------------------
# NPU Chunked (incremental) prefill -- doc 56 H1b: chunk-outer / layer-inner.
# ---------------------------------------------------------------------------


def run_npu_prefill_chunked(
    token_ids,
    weights,
    config,
    prefill_cache,
    decode_cache,
    rope_lut_bf16,
    max_seq,
    tokenizer,
    ubatch,
    profile=False,
    quiet=False,
    on_chunk=None,
):
    """Incremental NPU prefill: the prompt in `ubatch`-token chunks, chunk-outer /
    layer-inner, per-layer KV append, each chunk attending to all earlier
    chunks' KV plus its own (square FA at chunk 1, rectangular after), RoPE at
    absolute positions. The ONE owner of the chunked path (doc 56 H1b): the
    model adapter's `prefill(ubatch_policy="chunked")` and the verify adapter's
    `LLMS_VERIFY_UBATCH` branch both call THIS.

    `token_ids` is the VALID prompt only -- no EOS padding, and the length must
    be a multiple of `ubatch` (this scheduler has no padding path; a partial
    tail is a refusal, not a silent pad). The whole-prompt `run_npu_prefill`
    keeps its EOS-padded contract.

    `on_chunk(chunk_idx, context_start, context_end, elapsed_s)` fires after
    each chunk; the LAST chunk's callback fires after the final RMSNorm + LM
    head, so per-chunk times sum to the forward clock.

    Returns (prefill_token, logits_row, k_cache, v_cache, prompt_len) --
    exactly `run_npu_prefill`'s tuple; the KV state handed to decode IS the
    incrementally built cache.
    """
    prompt_len = len(token_ids)
    if ubatch <= 0 or prompt_len % ubatch != 0:
        raise ValueError(
            f"chunked prefill: prompt of {prompt_len} tokens is not a whole "
            f"number of ubatch={ubatch} chunks (no padding path; doc 56 H1b)"
        )
    if any(t == tokenizer.eos_token_id for t in token_ids):
        raise ValueError(
            "chunked prefill: token_ids is the valid prompt only (no EOS "
            "padding), but it contains the EOS id"
        )
    if prompt_len > max_seq:
        raise ValueError(f"prompt of {prompt_len} tokens exceeds max_seq {max_seq}")
    if rope_lut_bf16.shape[0] < prompt_len:
        raise ValueError(
            f"rope LUT covers {rope_lut_bf16.shape[0]} positions < prompt {prompt_len}"
        )
    n_chunks = prompt_len // ubatch
    emb_dim = config.emb_dim
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    vocab_size = weights.lm_head.shape[0]

    k_cache = np.zeros((config.n_layers, n_kv_heads, max_seq, head_dim), dtype=bfloat16)
    v_cache = np.zeros((config.n_layers, n_kv_heads, max_seq, head_dim), dtype=bfloat16)

    if not quiet:
        print(
            f"Running NPU chunked prefill ({config.n_layers} layers, "
            f"prompt {prompt_len} = {n_chunks} x ubatch {ubatch})..."
        )
    from qwen3_0_6b_prefill import run_transformer_block_qwen3_chunk

    t_start = time.time()
    prefill_token = None
    logits_row = None
    for c in range(n_chunks):
        t_chunk = time.perf_counter()
        start, end = c * ubatch, (c + 1) * ubatch
        with prefill_cache.profiler.time_cpu("embed_lookup"):
            x_bf16 = (
                weights.embed_table[list(token_ids[start:end])]
                .astype(np.float32)
                .astype(bfloat16)
            )
        for layer_idx in range(config.n_layers):
            t0 = prefill_cache.profiler.start_layer()
            x_bf16 = run_transformer_block_qwen3_chunk(
                x_bf16,
                weights.layers[layer_idx],
                rope_lut_bf16,
                config,
                prefill_cache,
                layer_idx,
                k_cache[layer_idx],
                v_cache[layer_idx],
                start,
                verbose=profile,
            )
            prefill_cache.profiler.end_layer(layer_idx, t0)
        if c == n_chunks - 1:
            # Final RMSNorm (eps=1e-6) on the last valid row + NPU LM-head.
            with prefill_cache.profiler.time_cpu("final_rms_norm"):
                last_hidden = np.asarray(x_bf16, dtype=np.float32)[-1:]
                last_normed = (
                    rms_norm(last_hidden, weights.final_norm, eps=EPS)
                    .flatten()
                    .astype(bfloat16)
                )
            logits_row = _run_lm_head(decode_cache, weights, last_normed, vocab_size)
            prefill_token = int(np.argmax(logits_row))
        if on_chunk is not None:
            on_chunk(c, start, end, time.perf_counter() - t_chunk)

    t_prefill = time.time() - t_start
    if not quiet:
        print(f"NPU chunked prefill done in {t_prefill:.2f}s. First token: {prefill_token}")
    return prefill_token, logits_row, k_cache, v_cache, prompt_len


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
    """Run one NPU decode step: 28 blocks + final RMSNorm + LM-head.

    (The O2 runlist-pairs prototype that once hooked here measured -2 to -4 ms/token and was
    removed in the 2026-08-22 cleanup; doc 57 §5 records it, tag pre-cleanup-20260821 holds it.)
    """
    vocab_size = weights.lm_head.shape[0]
    x = x_decode_bf16.copy()
    for layer_idx in range(config.n_layers):
        t0 = decode_cache.profiler.start_layer()
        x = run_decode_block(
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


def generate(
    prompt_tokens,
    weights,
    config,
    prefill_cache,
    decode_cache,
    rope_lut_bf16,
    tokenizer,
    n_tokens=10,
    profile=False,
    cpu_attn=True,
    on_token=None,
    ttft_start=None,
):
    seq_len = len(prompt_tokens)
    max_seq = seq_len + n_tokens
    streaming = on_token is not None
    if ttft_start is None:
        ttft_start = time.perf_counter()

    if not streaming:
        print(f"\n{'='*60}")
        print(f"Qwen3 Inference: prompt_len={seq_len}, n_tokens={n_tokens}")
        print(f"{'='*60}\n")

    prefill_token, _logits, k_cache, v_cache, prompt_len = run_npu_prefill(
        prompt_tokens,
        weights,
        config,
        prefill_cache,
        decode_cache,
        rope_lut_bf16,
        max_seq,
        tokenizer=tokenizer,
        cpu_attn=cpu_attn,
        profile=profile,
        quiet=True,
    )

    ttft = time.perf_counter() - ttft_start
    if not streaming:
        print(f"Time to first token (TTFT): {ttft:.2f}s. First token: {prefill_token}")

    generated_tokens = [prefill_token]
    current_pos = prompt_len
    x_decode = weights.embed_table[prefill_token].astype(bfloat16)

    stream_state = _StreamState() if streaming else None
    if streaming:
        on_token(prefill_token, _delta_text(tokenizer, generated_tokens, stream_state))

    if not streaming:
        print(f"\nDecoding {n_tokens} tokens...")
    t_dec = time.time()

    eos_ids = {tokenizer.eos_token_id}
    eot = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(eot, int) and eot >= 0:
        eos_ids.add(eot)

    for _ in range(n_tokens):
        next_token, _ = run_npu_decode_step(
            x_decode,
            weights,
            config,
            decode_cache,
            rope_lut_bf16,
            k_cache,
            v_cache,
            current_pos,
        )
        generated_tokens.append(next_token)
        current_pos += 1
        with decode_cache.profiler.time_cpu("embed_lookup"):
            x_decode = weights.embed_table[next_token].astype(bfloat16)
        if streaming:
            on_token(next_token, _delta_text(tokenizer, generated_tokens, stream_state))
        if next_token in eos_ids:
            break

    t_decode = time.time() - t_dec
    n_gen = len(generated_tokens) - 1
    if not streaming and n_gen > 0:
        print(
            f"\nGenerated {n_gen} tokens in {t_decode:.2f}s ({n_gen / t_decode:.2f} tok/s)"
        )

    if prefill_cache.profiler.enabled:
        print(f"\n{'='*60}\nPREFILL detail")
        prefill_cache.profiler.report()
    if decode_cache.profiler.enabled:
        print(f"\n{'='*60}\nDECODE detail")
        decode_cache.profiler.report()

    return generated_tokens


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

MODEL_CHOICES = {"base": "Qwen/Qwen3-0.6B", "instruct": "Qwen/Qwen3-0.6B"}


def build_session(args) -> Session:
    config = LlamaConfig()
    seq_len = 2048

    prefill_cache = KernelCache(
        "prefill_kernel_cache",
        verbose=args.verbose,
        profiler=Profiler(enabled=args.profile),
    )
    decode_cache = KernelCache(
        "decode_kernel_cache",
        verbose=args.verbose,
        profiler=Profiler(enabled=args.profile),
    )

    if not args.run_only:
        print("Compiling prefill kernels...")
        compile_all_kernels(
            prefill_cache, config, seq_len, verbose=args.verbose, cpu_attn=args.cpu_attn
        )
        print("\nCompiling decode kernels...")
        compile_decode_kernels(decode_cache, config, verbose=args.verbose)

    if args.compile_only:
        print("\nCompilation passed.")
        sys.exit(0)

    if args.run_only:
        prefill_cache.load_manifest()
        decode_cache.load_manifest()
        # `[2026-08-23]` the QKV ELF's scratch-arg layout, which only the
        # compile path used to set (doc 56 H1a review: 17 args to an 18-arg ELF)
        from qwen3_0_6b_prefill import restore_scratch_layout

        restore_scratch_layout(config, seq_len)

    model_id = MODEL_CHOICES.get(args.model, args.model)
    print(f"\nLoading weights ({model_id})...")
    weights = load_weights(model_id, config=config)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    rope_lut_bf16 = generate_rope_lut(
        config=config, seq_len=seq_len + args.n_tokens
    ).astype(bfloat16)

    prepare_runtime(
        prefill_cache, decode_cache, weights, config, seq_len, rope_lut_bf16
    )

    return Session(
        config=config,
        seq_len=seq_len,
        weights=weights,
        tokenizer=tokenizer,
        prefill_cache=prefill_cache,
        decode_cache=decode_cache,
        rope_lut_bf16=rope_lut_bf16,
        model_variant=args.model,
    )


def _tokenize_prompt(session: Session, prompt_text: str) -> list:
    if session.model_variant == "instruct":
        messages = [{"role": "user", "content": prompt_text}]
        chat_text = session.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return session.tokenizer.encode(chat_text)
    return session.tokenizer.encode(prompt_text)


def run_once(
    session, prompt_text, *, n_tokens, profile=False, cpu_attn=True, on_token=None
):
    ttft_start = time.perf_counter()
    with session.prefill_cache.profiler.time_cpu("tokenize"):
        tokens = _tokenize_prompt(session, prompt_text)
    prompt_len_actual = len(tokens)
    with session.prefill_cache.profiler.time_cpu("eos_pad"):
        if len(tokens) < session.seq_len:
            tokens = tokens + [session.tokenizer.eos_token_id] * (
                session.seq_len - len(tokens)
            )
    generated = generate(
        tokens,
        session.weights,
        session.config,
        session.prefill_cache,
        session.decode_cache,
        session.rope_lut_bf16,
        tokenizer=session.tokenizer,
        n_tokens=n_tokens,
        profile=profile,
        cpu_attn=cpu_attn,
        on_token=on_token,
        ttft_start=ttft_start,
    )
    return generated, prompt_len_actual


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


def repl_loop(session, args):
    print("\nInteractive mode — Ctrl-D or /quit to exit.\n")

    def _cb(_tid, delta):
        sys.stdout.write(delta)
        sys.stdout.flush()

    while True:
        try:
            prompt = input("Prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not prompt:
            continue
        if prompt in ("/quit", "/exit"):
            return
        sys.stdout.write("\nResponse: ")
        sys.stdout.flush()
        try:
            run_once(
                session,
                prompt,
                n_tokens=args.n_tokens,
                profile=False,
                cpu_attn=args.cpu_attn,
                on_token=_cb,
            )
        except KeyboardInterrupt:
            print("\n[interrupted]")
            continue
        print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-0.6B Inference (NPU)")
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
