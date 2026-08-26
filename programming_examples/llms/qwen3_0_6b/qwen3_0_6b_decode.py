# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Qwen3-0.6B Decode on MLIR-AIR (NPU2).

Single-token autoregressive generation with KV cache. Mirrors
llama32_1b_decode.py but applies the same two Qwen3 deltas the Phase-2
prefill handled:

  1. QK-norm: a per-head RMSNorm over head_dim on Q and K AFTER the GEMV
     projection and BEFORE RoPE. RoPE's linearity does NOT let us commute
     the (nonlinear) QK-norm past it, so we CANNOT use the llama
     `rms_gemv_rope` ELF (which fuses RoPE right after the GEMV). We instead
     build a Qwen-specific fused decode ELF that does RMSNorm + Q/K/V GEMV +
     per-head QK-norm + RoPE (M=1) entirely on the NPU
     (rms_qkv_qknorm_rope_gemv).

  2. Decoupled head_dim: n_heads*head_dim = 2048 != hidden_size = 1024.
        q_proj : 1024 -> 2048   (16 heads x 128)
        k/v    : 1024 -> 1024   (8 heads x 128)
        o_proj : 2048 -> 1024   (NOT square)
     The llama `rms_gemv_rope` asserts q_total==emb_dim; the llama
     `o_gemv_ffn` stage-1 O-GEMV is square (emb x emb). We build Qwen
     variants: the Q GEMV is M=q_dim, the O GEMV is M=emb_dim, K=q_dim.

  3. LM-head vocab = 151936 (not 128256). We split the vocab across
     19 partitions of n_part=8192 each (19*8192 = 155648 >= 151936;
     8192 % 64 == 0). n_part is capped at 8192 so the DMA repeat count
     n_part/32 - 1 = 255 stays at the hardware limit. The trailing
     partitions carry zero rows (logits truncated to vocab on host).

Decode attention is CPU (decode_attention_cpu), matching llama.
"""

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_PROG_EXAMPLES = str(Path(__file__).resolve().parent.parent.parent)
if _PROG_EXAMPLES not in sys.path:
    sys.path.insert(0, _PROG_EXAMPLES)
_LLMS_DIR = str(Path(__file__).resolve().parent.parent)
if _LLMS_DIR not in sys.path:
    sys.path.insert(0, _LLMS_DIR)

from qwen3_0_6b_weights import LlamaConfig
from shared.infra.cache import KernelCache


# The decode QKV stage ELF. `[2026-08-21]` 4 launches (doc 57 O1 first cut):
# RMSNorm, ONE GEMV over [wq; wk; wv], ONE per-row-weighted QK-norm over Q|K,
# ONE RoPE over Q|K -- the same kernels as the 8-launch form, bit-identical
# outputs (probe_o1_rms_qkv4.py, devq 461), 1.125 -> 0.680 ms per layer
# because each air.launch boundary costs ~107 us (doc 57 section 1.5). The
# artifact name carries the launch count so a stale 8-launch cache can never
# be bound to this arg layout.
# `[2026-08-22]` 2 launches (doc 57 O1 second half), PRODUCTION since
# `make verify` PASS (devq 556): RMSNorm, then ONE head-aligned GEMV whose
# cores apply QK-norm + RoPE in L1 (kernel mv_heads.cc, evidence
# results/o1-epilogue-20260822/; stage 0.672 -> 0.494 ms per layer, devq 555).
# QWEN3_RMS_QKV_LAUNCHES=4 selects the 4-launch form for A/B; the launch
# count is in the artifact name.
import os as _os  # noqa: E402

_RMS_QKV_LAUNCHES = int(_os.environ.get("QWEN3_RMS_QKV_LAUNCHES", "2"))
assert _RMS_QKV_LAUNCHES in (2, 4), _RMS_QKV_LAUNCHES
_RMS_QKV_KERNEL = f"rms_qkv_qknorm_rope_gemv{_RMS_QKV_LAUNCHES}"


def build_rms_qkv_qknorm_rope_gemv_module(config, n_launches=None):
    """Fused decode ELF: RMSNorm + Q/K/V GEMV + per-head QK-norm + RoPE (M=1).

    n_launches: 4 (the 2026-08-21 production form), 2 (the head-aligned GEMV
    with the in-core epilogue), 8 (the pre-2026-08-21 form, kept for A/B).
    Default: `_RMS_QKV_LAUNCHES`.
    """
    from shared.builders.rms_qkv_qknorm_rope_multi import (
        build_rms_qkv_qknorm_rope_gemv_module as _build8,
        build_rms_qkv_qknorm_rope_gemv4_module as _build4,
        build_rms_qkv_qknorm_rope_gemv2_module as _build2,
    )

    if n_launches is None:
        n_launches = _RMS_QKV_LAUNCHES

    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim
    build = {2: _build2, 4: _build4, 8: _build8}[n_launches]
    return build(
        emb_dim, q_dim, kv_dim, n_heads, n_kv_heads, head_dim, qknorm_eps=1e-6
    )


# Host side of the 4-launch stage: shared.infra.decode_qkv4 owns the 9-arg
# layout (and the 2-launch form's 5-arg layout); these are the model driver's
# thin names for it. The `rms_qkv4_*` names stay for the inference driver;
# they dispatch on `_RMS_QKV_LAUNCHES`.
from shared.infra import decode_qkv4 as _qkv4  # noqa: E402


def prep_rms_qkv4_weights(lw, config):
    if _RMS_QKV_LAUNCHES == 2:
        _qkv4.prep_weights_2(lw, config)
    else:
        _qkv4.prep_weights(lw, config)


rms_qkv4_lut = _qkv4.position_lut  # the tiled LUT; the 2-launch form takes its first head_dim


def rms_qkv4_args(lw, x_bf16, lut, config):
    """(inputs, output_indices, static_input_indices, intermediate_indices) of
    the QKV stage in the form `_RMS_QKV_KERNEL` names -- the 2-launch ABI
    (5 args, one LUT row) when `QWEN3_RMS_QKV_LAUNCHES` is 2, else the
    4-launch ABI (9 args, the tiled LUT). Mirrors `run_rms_qkv4`."""
    if _RMS_QKV_LAUNCHES == 2:
        return (
            _qkv4.call_args_2(
                lw, x_bf16, np.asarray(lut).reshape(-1)[: config.head_dim], config
            ),
            _qkv4.OUTPUT_INDICES_2,
            _qkv4.STATIC_INDICES_2,
            _qkv4.INTERMEDIATE_INDICES_2,
        )
    return (
        _qkv4.call_args(lw, x_bf16, lut, config),
        _qkv4.OUTPUT_INDICES,
        _qkv4.STATIC_INDICES,
        _qkv4.INTERMEDIATE_INDICES,
    )


def run_rms_qkv4(cache, lw, x_bf16, lut, config, layer_idx, verbose=False):
    """One call of the QKV stage (4- or 2-launch form) -> (v, q_roped, k_roped).
    `lut` is the tiled (n_heads+n_kv_heads) x head_dim LUT of `rms_qkv4_lut`;
    the 2-launch form takes one row of it."""
    if _RMS_QKV_LAUNCHES == 2:
        return _qkv4.run_2(
            cache, _RMS_QKV_KERNEL, _rms_qkv_qknorm_rope_gemv_backend(verbose),
            lw, x_bf16, np.asarray(lut).reshape(-1)[: config.head_dim], config, layer_idx,
        )
    return _qkv4.run(
        cache, _RMS_QKV_KERNEL, _rms_qkv_qknorm_rope_gemv_backend(verbose),
        lw, x_bf16, lut, config, layer_idx,
    )


def _rms_qkv_qknorm_rope_gemv_backend(verbose=False):
    return {
        "verbose": verbose,
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "rms_qkv_qknorm_rope_gemv",
    }


# LM-head decode partitioning. vocab=151936.
# Per-partition GEMV broadcasts the K=emb_dim input vector with a hardware
# push_queue repeat_count ~= n_part/32 - 1, capped at the [0:255] range. So
# n_part must be <= 8192 (8192/32 - 1 = 255). 19 * 8192 = 155648 >= 151936;
# the final partition carries 3712 zero-padded rows (logits truncated to
# vocab on host).
# `[2026-08-21]` Mixed partitions: 9 full partitions at the BD-repeat cap for
# m_input 8 (16384 rows: 16384/64 launch iterations x 1 kernel call = 256
# broadcasts, repeat 255) plus one 4480-row tail on the 64-row tile grid --
# 10 launches and 64 pad rows against the former 19 x 8192 (3712 pad rows).
# Measured 9.35 -> 8.25 ms per token on the probe (devq 476); the planner
# (shared/plan, doc 56 H0) derived it. Gated by `make verify`.
_LM_PARTS = tuple([16384] * 9 + [4480])   # sum 151936 == vocab
_LM_N_PARTITIONS = len(_LM_PARTS)
_LM_N_PART = 8192  # the pre-2026-08-21 uniform partition (repeat cap at m_input 4); kept for A/B


def lm_head_partition_slices(vocab_size):
    """[(start, end)] of the vocab rows each partition carries (end clipped to vocab)."""
    out, start = [], 0
    for rows in _LM_PARTS:
        out.append((start, min(start + rows, vocab_size)))
        start += rows
    return out


# ---------------------------------------------------------------------------
# Builder 1: o_gemv_ffn (decoupled O GEMV) + Residual + RMSNorm + SwiGLU FFN.
#   Copy of shared build_o_gemv_ffn_module but stage 1's O GEMV is
#   M=emb_dim, K=q_dim (attn_out is q_dim wide), wo is (emb_dim, q_dim).
#   Stages 2/3 (RMSNorm+SwiGLU, down GEMV) stay emb/hidden.
# ---------------------------------------------------------------------------


def build_o_gemv_ffn_qwen_module(emb_dim, q_dim, hidden_dim):
    """3-launch decode ELF: O-proj(decoupled) + residual + RMSNorm + SwiGLU.

    15-arg ABI mirrors the shared o_gemv_ffn (dead args kept), with two
    decoupled shapes:
      %arg0  wo        (emb_dim, q_dim)   <- DECOUPLED (was emb x emb)
      %arg1  attn_out  (q_dim,)           <- DECOUPLED (was emb)
      ... rest identical to shared o_gemv_ffn.
    """
    # Import o_gemv_ffn_multi first: its module-level sys.path.insert adds the
    # matvec_2tile_add / matvec_swiglu_rms source dirs to the path.
    from shared.builders.o_gemv_ffn_multi import (
        _STAGE2_TILE_M,
        _STAGE2_M_INPUT,
        _STAGE2_HERD_COLS,
        _STAGE2_N_CASCADE,
        _EXTERNS,
    )
    from matvec_2tile_add import build_module as build_2tile_add
    from matvec_swiglu_rms import build_module as build_swiglu_rms
    from shared.infra.stitching import stitch_elf, KernelSlice, FuncArg

    # Stage 1: O GEMV is M=emb_dim (output), K=q_dim (input). DECOUPLED.
    stage1 = build_2tile_add(emb_dim, q_dim, m=8, k=512, n_cores=8)
    # Stage 2: RMSNorm + interleaved gate/up GEMV + SwiGLU. emb/hidden.
    stage2 = build_swiglu_rms(
        2 * hidden_dim,
        emb_dim,
        _STAGE2_TILE_M,
        _STAGE2_M_INPUT,
        _STAGE2_HERD_COLS,
        _STAGE2_N_CASCADE,
        bfloat16,
        bfloat16,
    )
    # Stage 3: down GEMV M=emb_dim, K=hidden_dim.
    stage3 = build_2tile_add(emb_dim, hidden_dim, m=8, k=512, n_cores=8)

    base_args = [
        FuncArg("%arg0", f"memref<{emb_dim}x{q_dim}xbf16>"),  # wo (DECOUPLED)
        FuncArg("%arg1", f"memref<{q_dim}xbf16>"),  # attn_out (DECOUPLED)
        FuncArg("%arg2", f"memref<{emb_dim}xbf16>"),
        FuncArg("%arg3", f"memref<{emb_dim}xbf16>"),  # x_residual
        FuncArg("%arg4", f"memref<{emb_dim}xbf16>"),
        FuncArg("%arg5", f"memref<{emb_dim}xbf16>"),
        FuncArg("%arg6", f"memref<2x{emb_dim}xbf16>"),  # packed RMS input
        FuncArg("%arg7", f"memref<{2 * hidden_dim}x{emb_dim}xbf16>"),  # gate/up
        FuncArg("%arg8", f"memref<{hidden_dim}xbf16>"),
        FuncArg("%arg9", f"memref<{hidden_dim}x{emb_dim}xbf16>"),
        FuncArg("%arg10", f"memref<{hidden_dim}xbf16>"),
        FuncArg("%arg11", f"memref<{hidden_dim}xbf16>"),  # swiglu
        FuncArg("%arg12", f"memref<{emb_dim}x{hidden_dim}xbf16>"),  # wdown
        FuncArg("%arg13", f"memref<{emb_dim}xbf16>"),
        FuncArg("%arg14", f"memref<{emb_dim}xbf16>"),  # output
    ]
    prelude = (
        f"    %arg6_row0_strided = memref.subview %arg6[0, 0] [1, {emb_dim}] [1, 1]\n"
        f"        : memref<2x{emb_dim}xbf16> to memref<{emb_dim}xbf16, strided<[1]>>\n"
        f"    %arg6_row0 = memref.cast %arg6_row0_strided\n"
        f"        : memref<{emb_dim}xbf16, strided<[1]>> to memref<{emb_dim}xbf16>"
    )
    slices = [
        KernelSlice(
            str(stage1),
            "s1",
            {0: 0, 1: 1, 2: 3},
            arg_aliases={3: "%arg6_row0"},
            extern_syms=_EXTERNS,
        ),
        KernelSlice(str(stage2), "s2", {0: 7, 1: 6, 2: 11}, extern_syms=_EXTERNS),
        KernelSlice(
            str(stage3),
            "s3",
            {0: 12, 1: 11, 3: 14},
            arg_aliases={2: "%arg6_row0"},
            extern_syms=_EXTERNS,
        ),
    ]
    module = stitch_elf(
        "o_gemv_ffn",
        base_args,
        slices,
        prelude=prelude,
        allow_unreferenced_args={2, 4, 5, 8, 9, 10, 13},
    )
    print(f"  o_gemv_ffn_qwen module: {len(str(module).splitlines())} lines, parsed OK")
    return module


# ---------------------------------------------------------------------------
# Builder 2: LM-head GEMV (19 partitions, n_part=8192 for vocab 151936).
# ---------------------------------------------------------------------------


def build_lm_head_gemv_qwen_module(emb_dim):
    from shared.builders.lm_head_gemv_multi import build_lm_head_gemv_module

    return build_lm_head_gemv_module(
        emb_dim=emb_dim,
        parts=_LM_PARTS,
        tile_m=8,
        # m_input 8 (one kernel call per 8-row tile, B-broadcast repeat ~127)
        # measured 8.5 % faster than m_input 4 at the same launch count and
        # bytes (doc 57 section 1.4, devq 449: 9.12 vs 9.96 ms); gated by
        # `make verify`.
        m_input=8,
        herd_m=8,
    )


# ---------------------------------------------------------------------------
# Backend kwargs
# ---------------------------------------------------------------------------


def _o_gemv_ffn_backend(verbose=False):
    return {
        "verbose": verbose,
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "o_gemv_ffn",
        "use_lock_race_condition_fix": False,
    }


def _lm_gemv_backend(verbose=False):
    return {
        "verbose": verbose,
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "lm_head_gemv",
    }


# ---------------------------------------------------------------------------
# Decode kernel compilation
# ---------------------------------------------------------------------------


def compile_decode_kernels(cache, config, verbose=False):
    """Compile the Qwen3 decode kernels."""
    from shared.infra.external_kernels import (
        compile_mv,
        compile_mv_bf16,
        compile_rope,
        compile_silu_and_mul,
    )

    emb_dim = config.emb_dim
    hidden_dim = config.hidden_dim
    q_dim = config.n_heads * config.head_dim

    print(f"\n{'='*60}\nCompiling Qwen3 decode kernels...\n{'='*60}\n")

    # External .o kernels: GEMV (mv.o), 2tile-add/swiglu (mv_bf16.o), RoPE.
    compile_mv()
    compile_mv_bf16()
    compile_rope()
    compile_silu_and_mul()

    if _RMS_QKV_LAUNCHES == 2:
        from shared.infra.external_kernels import compile_mv_heads

        compile_mv_heads(config.head_dim)
    print(
        f"\n--- {_RMS_QKV_KERNEL} (FUSED: RMSNorm+QKV+QK-norm+RoPE, "
        f"{_RMS_QKV_LAUNCHES} launches) ---"
    )
    cache.compile_and_cache(
        _RMS_QKV_KERNEL,
        build_rms_qkv_qknorm_rope_gemv_module(config),
        _rms_qkv_qknorm_rope_gemv_backend(verbose),
    )

    print("\n--- o_gemv_ffn (O GEMV decoupled + Residual + FFN) ---")
    cache.compile_and_cache(
        "o_gemv_ffn",
        build_o_gemv_ffn_qwen_module(emb_dim, q_dim, hidden_dim),
        _o_gemv_ffn_backend(verbose),
    )

    print("\n--- lm_head_gemv (19-partition, vocab 151936) ---")
    cache.compile_and_cache(
        "lm_head_gemv",
        build_lm_head_gemv_qwen_module(emb_dim),
        _lm_gemv_backend(verbose),
    )

    cache._save_manifest()
    print(f"\nAll {len(cache.artifacts)} decode kernels compiled.")


# ---------------------------------------------------------------------------
# CPU decode attention (with KV cache)
# ---------------------------------------------------------------------------


def decode_attention_cpu(
    q, k_cache, v_cache, current_pos, n_heads, n_kv_heads, head_dim
):
    """Single-query GQA attention with KV cache.

    Args:
        q: (q_dim,) — RoPE'd query vector for the current token.
        k_cache: (n_kv_heads, max_seq, head_dim) — cached keys [0:current_pos+1].
        v_cache: (n_kv_heads, max_seq, head_dim) — cached values.
        current_pos: current token position (0-indexed).
    Returns:
        attn_out: (q_dim,) bfloat16.
    """
    group_size = n_heads // n_kv_heads
    scale = 1.0 / np.sqrt(head_dim)
    seq_len = current_pos + 1

    q_heads = q.astype(np.float32).reshape(n_heads, head_dim)
    k_cached = k_cache[:, :seq_len, :].astype(np.float32)
    v_cached = v_cache[:, :seq_len, :].astype(np.float32)

    out = np.zeros((n_heads, head_dim), dtype=np.float32)
    for h in range(n_heads):
        kv_h = h // group_size
        scores = (q_heads[h] @ k_cached[kv_h].T) * scale
        probs = np.exp(scores - scores.max())
        probs = probs / probs.sum()
        out[h] = probs @ v_cached[kv_h]

    return out.reshape(-1).astype(bfloat16)


# ---------------------------------------------------------------------------
# Single decode transformer block
# ---------------------------------------------------------------------------


def run_decode_block(
    x_bf16,
    layer_weights,
    cache,
    config,
    k_cache_layer,
    v_cache_layer,
    current_pos,
    rope_lut_bf16,
    verbose=False,
):
    """Run one Qwen3 transformer block for a single decode token.

    Stages: rms_qkv_qknorm_rope_gemv4 (NPU, 4 launches: RMSNorm + QKV GEMV +
    per-head QK-norm + RoPE) -> KV-cache write -> CPU attention -> o_gemv_ffn (NPU).
    """
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim

    layer_idx = getattr(layer_weights, "_layer_idx", None)

    # --- One ELF (4 launches) = RMSNorm + QKV GEMV + per-head QK-norm + RoPE ---
    lut = rms_qkv4_lut(rope_lut_bf16, current_pos, config)
    v, q_roped, k_roped = run_rms_qkv4(
        cache, layer_weights, x_bf16, lut, config, layer_idx, verbose
    )

    # --- Update KV cache (K after qk-norm AND rope; V raw projection) ---
    # `[2026-08-25]` bucketed so host_ops equals the plan's host-stage count
    # (doc 56 s3.6; the kv_append stage was the uncounted one).
    with cache.profiler.time_cpu("kv_append"):
        k_cache_layer[:, current_pos, :] = k_roped.reshape(n_kv_heads, head_dim)
        v_cache_layer[:, current_pos, :] = v.reshape(n_kv_heads, head_dim)

    # --- CPU attention ---
    with cache.profiler.time_cpu("decode_attention_cpu"):
        attn_out = decode_attention_cpu(
            q_roped,
            k_cache_layer,
            v_cache_layer,
            current_pos,
            n_heads,
            n_kv_heads,
            head_dim,
        )

    # --- Stage E: O-proj (decoupled) + Residual + RMSNorm + SwiGLU ---
    return _run_o_gemv_ffn(
        attn_out, x_bf16, layer_weights, config, cache, layer_idx, verbose
    )


def _run_o_gemv_ffn(
    attn_out, x_bf16, layer_weights, config, cache, layer_idx, verbose=False
):
    """Decode Stage E: O-proj(decoupled) + Residual + RMSNorm + SwiGLU FFN.

    Shared by the fused and legacy decode paths so the o_gemv_ffn arg layout +
    BO indices have a single owner.
    """
    emb_dim = config.emb_dim
    hidden_dim = config.hidden_dim
    z_emb = np.zeros(emb_dim, dtype=bfloat16)
    z_hidden = np.zeros(hidden_dim, dtype=bfloat16)
    z_hidden_emb = np.zeros((hidden_dim, emb_dim), dtype=bfloat16)
    results = cache.load_and_run(
        "o_gemv_ffn",
        _o_gemv_ffn_backend(verbose),
        layer_weights._wo_t,  # arg0 wo (static, decoupled)
        attn_out,  # arg1 attn_out (q_dim)
        z_emb,  # arg2 (dead)
        x_bf16.flatten().astype(bfloat16),  # arg3 x_residual
        z_emb,  # arg4 (dead)
        z_emb,  # arg5 (dead)
        layer_weights._packed_rms_buf,  # arg6 packed (static)
        layer_weights._wgateup_t,  # arg7 gate/up (static)
        z_hidden,  # arg8 (dead)
        z_hidden_emb,  # arg9 (dead)
        z_hidden,  # arg10 (dead)
        z_hidden,  # arg11 swiglu
        layer_weights._wdown_t,  # arg12 wdown (static)
        z_emb,  # arg13 (dead)
        z_emb,  # arg14 output
        output_indices=[14],
        static_input_indices={0, 6, 7, 12},
        intermediate_indices={2, 4, 5, 8, 9, 10, 11, 13, 14},
        bo_key=f"o_gemv_ffn_L{layer_idx}" if layer_idx is not None else None,
    )
    return results[14].astype(bfloat16)
