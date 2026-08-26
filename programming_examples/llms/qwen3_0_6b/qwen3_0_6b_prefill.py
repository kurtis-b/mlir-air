# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Qwen3-0.6B Prefill on MLIR-AIR (NPU2) — single-block (Phase 2) path.

Qwen3 diverges from LLAMA-3.2 in two ways that break the llama
`build_rms_gemms_rope_module` fusion:

  1. QK-norm: a per-head RMSNorm over head_dim is applied to Q and K AFTER
     the projection GEMM and BEFORE RoPE. RoPE linearity does NOT let us
     commute the (nonlinear) QK-norm past RoPE, so the llama RMSNorm+QKV+RoPE
     fused ELF (RoPE immediately after the GEMM) is wrong. We instead build a
     Qwen-specific 8-launch ELF that does RMSNorm + Q/K/V GEMM + per-head
     QK-norm(Q,K) + RoPE(Q,K) all on the NPU (rms_qkv_qknorm_rope).

  2. Decoupled head_dim: n_heads*head_dim = 2048 != hidden_size = 1024.
        q_proj : 1024 -> 2048   (16 heads x 128)
        k/v    : 1024 -> 1024   (8 heads x 128)
        o_proj : 2048 -> 1024   (NOT square — llama's o_ffn assumes square wo)
     We build a Qwen-specific o_ffn ELF whose O GEMM is K=q_dim=2048,
     N=emb_dim=1024; the residual/RMSNorm/FFN tail stays emb_dim=1024.

Attention uses the CPU fallback (cpu_attn=True), matching llama prefill.
"""

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

# Add programming_examples/ and llms/ to path for shared.* + registry imports.
_PROG_EXAMPLES = str(Path(__file__).resolve().parent.parent.parent)
if _PROG_EXAMPLES not in sys.path:
    sys.path.insert(0, _PROG_EXAMPLES)
_LLMS_DIR = str(Path(__file__).resolve().parent.parent)
if _LLMS_DIR not in sys.path:
    sys.path.insert(0, _LLMS_DIR)

from qwen3_0_6b_weights import LlamaConfig
from qwen3_0_6b_cpu_helpers import attention_reference
from shared.infra.cache import KernelCache, Profiler

# ---------------------------------------------------------------------------
# Builder 1 (FUSED): RMSNorm + Q/K/V GEMM + per-head QK-norm(Q,K) + RoPE(Q,K).
#   8-launch ELF that does the entire attention-input stage on the NPU. See
#   shared/builders/rms_qkv_qknorm_rope_multi.py.
# ---------------------------------------------------------------------------


def build_rms_qkv_qknorm_rope_module(seq_len, config):
    from shared.builders.rms_qkv_qknorm_rope_multi import (
        build_rms_qkv_qknorm_rope_module as _build,
    )

    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim
    return _build(
        seq_len,
        emb_dim,
        q_dim,
        kv_dim,
        n_heads,
        n_kv_heads,
        head_dim,
        qknorm_eps=1e-6,
    )


# ---------------------------------------------------------------------------
# Builder 2: O proj (decoupled wo: q_dim->emb_dim) + Residual + FFN.
#   Delegates to the shared _build_o_ffn (q_dim=, per-GEMM registry method /
#   tile_n since queue item 14): the O GEMM is K=q_dim, N=emb_dim, attn_out
#   arg shape (seq, q_dim), wo (q_dim, emb_dim). Everything from the residual
#   add onward stays emb_dim.
# ---------------------------------------------------------------------------


def build_o_ffn_qwen_module(
    seq_len,
    emb_dim,
    q_dim,
    hidden_dim,
    gemm_method=None,
):
    """O-proj(q_dim->emb_dim) + Residual + FFN. Returns (module, scratch_for).

    `[2026-08-25]` PER-GEMM (queue item 14): a thin delegate to the shared
    `o_ffn_multi._build_o_ffn`, which now takes each GEMM's registry method +
    tile_n independently (drain = 3-arg, 1 launch; fused-cast = 4-arg + f32
    scratch + cast launch) with per-GEMM mm.o objects co-linked in one ELF —
    the doc 56 H1a walls (drain at M=512/1024; the M=512 tile_n 96/128 mix)
    are gone. `gemm_method` forces every GEMM's registry method and is
    test-only now; a forced method is a plan deviation and must be recorded
    as one (`compile.json` `artifact_deviation`).

    Func args 0..14 as before (attn_out (seq, q_dim) / wo (q_dim, emb_dim)
    DECOUPLED; output at 14); the f32 C-scratch tail (base index 15) exists
    per FUSED-CAST GEMM only — `scratch_for` (order O/gate/up/down, None =
    drain) is the ELF's arg-layout contract, threaded to the call sites via
    `_OFFN_SCRATCH_FOR`.
    """
    from shared.builders.o_ffn_multi import _build_o_ffn

    return _build_o_ffn(
        seq_len=seq_len,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        q_dim=q_dim,
        gemm_method=gemm_method,
        func_name="o_ffn_qwen",
        debug_dump_path="/tmp/debug_o_ffn_qwen.mlir",
    )


# ---------------------------------------------------------------------------
# Backend kwargs
# ---------------------------------------------------------------------------


def _rms_qkv_qknorm_rope_backend(verbose=False):
    return {
        "verbose": verbose,
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "rms_qkv_qknorm_rope",
        "runtime_loop_tiling_sizes": [2, 2],
    }


def _o_ffn_backend(verbose=False):
    return {
        "verbose": verbose,
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "o_ffn_qwen",
        "runtime_loop_tiling_sizes": [2, 2],
    }


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

# Scratch-arg indices for the rms_qkv_qknorm_rope ELF (registry-driven;
# GQA -> 1 fused-cast scratch on Q). Set by compile_all_kernels so the block
# runner + preload know the fused ELF's scratch-arg layout.
_FUSED_SCRATCH_FOR = None

# Scratch-arg indices for the o_ffn_qwen ELF (registry-driven, PER GEMM in
# order O/gate/up/down; None = drain, no scratch — queue item 14). Set by
# compile_all_kernels / restore_scratch_layout; `_o_ffn_call` is the one
# consumer.
_OFFN_SCRATCH_FOR = None


def restore_scratch_layout(config, seq_len, o_ffn_gemm_method=None):
    """`[2026-08-23]` Set `_FUSED_SCRATCH_FOR` WITHOUT compiling: the layout
    `alloc_gemm_scratch` gives the QKV ELF at this seq_len from the registry's
    Q/K/V methods (base arg 17), exactly as `compile_all_kernels` leaves it.
    A process that loads the manifest (`--run-only`, the verify adapter on a
    timed artifact set, the model study) passed 17 args to an 18-arg ELF at
    seq_len 2048 (Q is fused-cast there) until this existed. Returns the layout.

    `[2026-08-25]` Also sets `_OFFN_SCRATCH_FOR` the same way (queue item 14:
    the o_ffn cascade is per-GEMM now, so its scratch tail is registry-driven
    too — all-drain at M=512/1024 means NO scratch args, and a loaded artifact
    set must be called with exactly its own arg count).

    `o_ffn_gemm_method`: the artifact set's RECORDED deviation (compile.json
    `artifact_deviation.o_ffn_gemm_method`), when the loaded set was compiled
    with a forced method. The ELF's arg layout is the deviation's, not the
    registry's: restoring the registry layout against a forced fused-cast ELF
    sets 15 of its 19 args and leaves the f32 scratch args UNBOUND — a
    nondeterministic wrong answer the token gate catches only sometimes
    (observed live, devq 583, on the pre-item-14 forced M=1024 set)."""
    global _FUSED_SCRATCH_FOR, _OFFN_SCRATCH_FOR
    from shared.builders.gemm_builder import gemm_registry_config, o_ffn_gemm_layout
    from shared.infra.stitching import alloc_gemm_scratch

    q_dim = config.n_heads * config.head_dim
    kv_dim = config.n_kv_heads * config.head_dim
    _args, scratch_for = alloc_gemm_scratch(
        [
            (gemm_registry_config(seq_len, config.emb_dim, q_dim, "bf16", "high"), seq_len, q_dim),
            (gemm_registry_config(seq_len, config.emb_dim, kv_dim, "bf16", "high"), seq_len, kv_dim),
            (gemm_registry_config(seq_len, config.emb_dim, kv_dim, "bf16", "high"), seq_len, kv_dim),
        ],
        base_arg_count=17,
    )
    _FUSED_SCRATCH_FOR = scratch_for
    _OFFN_SCRATCH_FOR = o_ffn_gemm_layout(
        seq_len, config.emb_dim, config.hidden_dim, q_dim=q_dim,
        method=o_ffn_gemm_method,
    )["scratch_for"]
    return list(scratch_for)


def compile_all_kernels(cache, config, seq_len, verbose=False, cpu_attn=False, o_ffn_gemm_method=None):
    global _FUSED_SCRATCH_FOR, _OFFN_SCRATCH_FOR
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    print(
        f"\n{'='*60}\nCompiling Qwen3 prefill kernels (seq_len={seq_len})...\n{'='*60}\n"
    )

    from shared.builders.gemm_builder import o_ffn_gemm_layout
    from shared.infra.external_kernels import compile_gemm_mm_variant, compile_rope

    # mm.o variants for GEMM co-linking (the two the QKV stage can name at any
    # M, plus whatever the o_ffn layout's per-GEMM specs name — e.g. a forced
    # fused-cast M=512 needs mm_m64n96.o); rope.o (head_dim=128) for the ropes.
    offn_layout = o_ffn_gemm_layout(
        seq_len, emb_dim, hidden_dim, q_dim=q_dim, method=o_ffn_gemm_method
    )
    variants = {(32, 128, 32), (64, 128, 32)}
    for spec in (offn_layout["o"], offn_layout["gate_up"], offn_layout["down"]):
        variants.add((spec["tile_m"], spec["tile_n"], spec["tile_k_l1"]))
    for tile_m, tile_n, tile_k_l1 in sorted(variants):
        compile_gemm_mm_variant(tile_m=tile_m, tile_n=tile_n, tile_k_l1=tile_k_l1)
    compile_rope()

    print("\n--- rms_qkv_qknorm_rope (FUSED: RMSNorm+QKV+QK-norm+RoPE, 8 launches) ---")
    fused_mod, fused_scratch = build_rms_qkv_qknorm_rope_module(seq_len, config)
    _FUSED_SCRATCH_FOR = fused_scratch
    cache.compile_and_cache(
        "rms_qkv_qknorm_rope", fused_mod, _rms_qkv_qknorm_rope_backend(verbose)
    )

    print("\n--- o_ffn_qwen (O proj decoupled + Residual + FFN, per-GEMM method) ---")
    offn_mod, offn_scratch = build_o_ffn_qwen_module(
        seq_len, emb_dim, q_dim, hidden_dim, gemm_method=o_ffn_gemm_method
    )
    _OFFN_SCRATCH_FOR = offn_scratch
    cache.compile_and_cache("o_ffn_qwen", offn_mod, _o_ffn_backend(verbose))

    # Flash Attention (head-first, head_dim=128). Skip if using CPU fallback.
    if not cpu_attn:
        print("\n--- flash_attn (head-first FA, head_dim=128) ---")
        from shared.infra.fa_headfirst import compile_headfirst_fa

        compile_headfirst_fa(cache, seq_len, n_heads, n_kv_heads, head_dim, verbose)
    else:
        print("\n--- Skipping flash_attn (CPU attention fallback) ---")

    cache._save_manifest()
    print(f"\nAll {len(cache.artifacts)} kernels compiled to {cache.cache_dir}/")


# ---------------------------------------------------------------------------
# Prefill weight pre-load (BO reuse — opt-buffer-object-reuse B1)
# ---------------------------------------------------------------------------


def _fused_qknorm_rope_call(
    cache, lw, config, seq_len, lut_q, lut_k, layer_idx, x_in, verbose=False
):
    """Issue one rms_qkv_qknorm_rope ELF call (fused prefill attention-input).

    Used by BOTH preload_prefill_weights (warmup, x_in zeroed) and the block
    runner (x_in = real hidden). Returns the cache.load_and_run result tuple
    (output_indices=[8, 15, 16] -> v, q_roped, k_roped). The single owner of
    the fused arg layout + index sets so the warmup BO set lines up exactly.
    """
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    args = [
        np.asarray(x_in, dtype=bfloat16).reshape(seq_len, emb_dim),  # 0 x_in (dynamic)
        np.asarray(lw.attn_norm, dtype=bfloat16).reshape(emb_dim),  # 1 norm_w (static)
        np.zeros((seq_len, emb_dim), dtype=bfloat16),  # 2 normed (inter)
        np.asarray(lw.wq, dtype=bfloat16).reshape(emb_dim, q_dim),  # 3 wq (static)
        np.zeros((seq_len, q_dim), dtype=bfloat16),  # 4 q (inter)
        np.asarray(lw.wk, dtype=bfloat16).reshape(emb_dim, kv_dim),  # 5 wk (static)
        np.zeros((seq_len, kv_dim), dtype=bfloat16),  # 6 k (inter)
        np.asarray(lw.wv, dtype=bfloat16).reshape(emb_dim, kv_dim),  # 7 wv (static)
        np.zeros((seq_len, kv_dim), dtype=bfloat16),  # 8 v (inter/out)
        np.asarray(lw.q_norm, dtype=bfloat16).reshape(head_dim),  # 9 q_norm (static)
        np.asarray(lw.k_norm, dtype=bfloat16).reshape(head_dim),  # 10 k_norm (static)
        np.zeros((seq_len, q_dim), dtype=bfloat16),  # 11 q_n (inter)
        np.zeros((seq_len, kv_dim), dtype=bfloat16),  # 12 k_n (inter)
        lut_q,  # 13 lut_q (static)
        lut_k,  # 14 lut_k (static)
        np.zeros((seq_len, q_dim), dtype=bfloat16),  # 15 q_roped (inter/out)
        np.zeros((seq_len, kv_dim), dtype=bfloat16),  # 16 k_roped (inter/out)
    ]
    inter = {2, 4, 6, 8, 11, 12, 15, 16}
    nxt = 17
    for sc, cols in zip(_FUSED_SCRATCH_FOR or [], (q_dim, kv_dim, kv_dim)):
        if sc is not None:
            args.append(np.zeros((seq_len, cols), dtype=np.float32))
            inter.add(nxt)
            nxt += 1
    return cache.load_and_run(
        "rms_qkv_qknorm_rope",
        _rms_qkv_qknorm_rope_backend(verbose),
        *args,
        output_indices=[8, 15, 16],
        static_input_indices={1, 3, 5, 7, 9, 10, 13, 14},
        intermediate_indices=inter,
        bo_key=f"rms_qkv_qknorm_rope_L{layer_idx}",
        shared_nonstatic=True,
    )


def _o_ffn_call(cache, lw, config, seq_len, attn_out, x_resid, layer_idx, verbose=False):
    """Issue one o_ffn_qwen ELF call (O proj + Residual + FFN).

    Used by BOTH preload_prefill_weights (warmup, activations zeroed) and the
    block runner. Returns the cache.load_and_run result tuple (output index 14).
    The single owner of the o_ffn arg layout: 15 base args plus one f32 scratch
    per FUSED-CAST GEMM as `_OFFN_SCRATCH_FOR` says (order O/gate/up/down;
    None = drain, no arg — queue item 14, the cascade is per-GEMM), so the
    warmup BO set and the inference call line up exactly at every layout.
    """
    if _OFFN_SCRATCH_FOR is None:
        raise RuntimeError(
            "o_ffn_qwen scratch layout unset: call compile_all_kernels or "
            "restore_scratch_layout(config, seq_len) before running the block"
        )
    emb_dim = config.emb_dim
    hidden_dim = config.hidden_dim
    q_dim = config.n_heads * config.head_dim
    n_total = seq_len * emb_dim

    args = [
        np.asarray(attn_out, dtype=bfloat16).reshape(seq_len, q_dim),  # 0 attn_out (dynamic)
        np.asarray(lw.wo, dtype=bfloat16).reshape(q_dim, emb_dim),  # 1 wo (static)
        np.zeros((seq_len, emb_dim), dtype=bfloat16),  # 2 proj (inter)
        np.asarray(x_resid, dtype=bfloat16).reshape(seq_len, emb_dim),  # 3 x_resid (dynamic)
        np.zeros((seq_len, emb_dim), dtype=bfloat16),  # 4 res1 (inter)
        np.asarray(lw.ffn_norm, dtype=bfloat16).reshape(emb_dim),  # 5 ffn_norm (static)
        np.zeros((seq_len, emb_dim), dtype=bfloat16),  # 6 normed2 (inter)
        np.asarray(lw.w_gate, dtype=bfloat16).reshape(emb_dim, hidden_dim),  # 7 w_gate (static)
        np.zeros((seq_len, hidden_dim), dtype=bfloat16),  # 8 gate (inter)
        np.asarray(lw.w_up, dtype=bfloat16).reshape(emb_dim, hidden_dim),  # 9 w_up (static)
        np.zeros((seq_len, hidden_dim), dtype=bfloat16),  # 10 up (inter)
        np.zeros((seq_len, hidden_dim), dtype=bfloat16),  # 11 swiglu (inter)
        np.asarray(lw.w_down, dtype=bfloat16).reshape(hidden_dim, emb_dim),  # 12 w_down (static)
        np.zeros((seq_len, emb_dim), dtype=bfloat16),  # 13 down (inter)
        np.zeros(n_total, dtype=bfloat16),  # 14 output (inter/out)
    ]
    inter = {2, 4, 6, 8, 10, 11, 13, 14}
    scratch_cols = (emb_dim, hidden_dim, hidden_dim, emb_dim)  # O, gate, up, down
    for sc, cols in zip(_OFFN_SCRATCH_FOR, scratch_cols):
        if sc is not None:
            assert sc == len(args), (sc, len(args))
            args.append(np.zeros((seq_len, cols), dtype=np.float32))
            inter.add(sc)
    return cache.load_and_run(
        "o_ffn_qwen",
        _o_ffn_backend(verbose),
        *args,
        output_indices=[14],
        static_input_indices={1, 5, 7, 9, 12},
        intermediate_indices=inter,
        bo_key=f"o_ffn_qwen_L{layer_idx}",
        shared_nonstatic=True,
    )


def preload_prefill_weights(weights, config, cache, seq_len, rope_lut_bf16):
    """Pre-load all prefill block weights into per-layer BOs once.

    Mirrors llama's preload_prefill_weights / IRON's prepare_runtime: a warmup
    XRT call per layer per ELF allocates the bo_key-keyed BO set and performs
    the host->device write of the *static* weight args. During the real prefill
    pass, ``static_input_indices`` then skips those weight writes (they are
    unchanged), so only the small dynamic activation inputs are re-written.

    Without this, the first (and only) prefill pass writes every weight inside
    the timed region: o_ffn_qwen alone moves 154 MB/layer of BO data on call 1.

    The warmup call layout MUST match ``run_transformer_block_qwen3`` exactly
    (same arg count, same static_input_indices / intermediate_indices, same
    bo_key) or the reused BO set would not line up at inference time.
    """
    if hasattr(weights, "_prefill_weights_preloaded"):
        return

    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    print("Pre-loading prefill block weights (per-layer BOs)...")
    profiler_enabled = cache.profiler.enabled
    cache.profiler.enabled = False

    # RoPE LUTs (seq-first, repeated per head) — same for all layers.
    lut_q = np.repeat(rope_lut_bf16[:seq_len], n_heads, axis=0).flatten()
    lut_k = np.repeat(rope_lut_bf16[:seq_len], n_kv_heads, axis=0).flatten()

    for layer_idx in range(config.n_layers):
        lw = weights.layers[layer_idx]

        # One fused ELF: RMSNorm + Q/K/V GEMM + per-head QK-norm + RoPE.
        _fused_qknorm_rope_call(
            cache,
            lw,
            config,
            seq_len,
            lut_q,
            lut_k,
            layer_idx,
            np.zeros((seq_len, emb_dim), dtype=bfloat16),
        )

        # o_ffn_qwen: allocate + write wo/ffn_norm/w_gate/w_up/w_down ({1,5,7,9,12}).
        _o_ffn_call(
            cache,
            lw,
            config,
            seq_len,
            np.zeros((seq_len, q_dim), dtype=bfloat16),
            np.zeros((seq_len, emb_dim), dtype=bfloat16),
            layer_idx,
        )

    cache.profiler.enabled = profiler_enabled
    weights._prefill_weights_preloaded = True
    print(f"  Pre-loaded {config.n_layers} prefill layers.")


# ---------------------------------------------------------------------------
# Single transformer block
# ---------------------------------------------------------------------------


def run_transformer_block_qwen3(
    x_bf16,
    layer_weights,
    rope_lut_bf16,
    config,
    cache,
    layer_idx=0,
    cpu_attn=True,
    verbose=False,
):
    """Run one Qwen3 transformer block on NPU (kernels pre-compiled in cache)."""
    seq_len = x_bf16.shape[0]
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    inter = {}

    # RoPE LUTs (seq-first, repeated per head).
    lut_q = np.repeat(rope_lut_bf16[:seq_len], n_heads, axis=0).flatten()
    lut_k = np.repeat(rope_lut_bf16[:seq_len], n_kv_heads, axis=0).flatten()

    # ---- Stages A-C: one ELF = RMSNorm + Q/K/V GEMM + QK-norm(Q,K) + RoPE(Q,K).
    res = _fused_qknorm_rope_call(
        cache,
        layer_weights,
        config,
        seq_len,
        lut_q,
        lut_k,
        layer_idx,
        x_bf16,
        verbose=verbose,
    )
    v = res[8].reshape(seq_len, kv_dim)
    q_roped = res[15].reshape(seq_len, q_dim)
    k_roped = res[16].reshape(seq_len, kv_dim)
    inter["v"] = v
    inter["q_roped"] = q_roped
    inter["k_roped"] = k_roped

    # ---- Stage D: GQA attention ----
    if cpu_attn:
        # HOST cpu fallback (FP32 causal GQA reference).
        with cache.profiler.time_cpu("prefill_cpu_attention"):
            attn_out = attention_reference(
                q_roped.astype(np.float32),
                k_roped.astype(np.float32),
                v.astype(np.float32),
                n_heads,
                n_kv_heads,
            ).astype(bfloat16)
    else:
        # NPU head-first FlashAttention (head_dim=128). q_roped/k_roped are
        # post-QK-norm post-RoPE seq-first; v is the raw projection seq-first.
        from shared.infra.fa_headfirst import npu_fa_headfirst

        attn_out = npu_fa_headfirst(
            cache,
            np.ascontiguousarray(q_roped),
            np.ascontiguousarray(k_roped),
            np.ascontiguousarray(v),
            n_heads,
            n_kv_heads,
            head_dim,
            seq_len,
            verbose=verbose,
        )
    inter["attn_out"] = attn_out

    # ---- Stage E: O proj + Residual + FFN ----
    results = _o_ffn_call(
        cache,
        layer_weights,
        config,
        seq_len,
        attn_out,
        x_bf16,
        layer_idx,
        verbose=verbose,
    )
    output_bf16 = results[14].reshape(seq_len, emb_dim)
    inter["ffn_out"] = output_bf16
    return output_bf16, inter
