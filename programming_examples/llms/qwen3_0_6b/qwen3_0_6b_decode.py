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
     per-head QK-norm + RoPE (M=1) entirely on the NPU: 2 launches
     (rms_qkv_qknorm_rope_gemv2, the head-aligned GEMV with the in-core
     epilogue) by default, 8 launches (rms_qkv_qknorm_rope_gemv) under
     QWEN3_RMS_QKV_LAUNCHES=8 for A/B.

  2. Decoupled head_dim: n_heads*head_dim = 2048 != hidden_size = 1024.
        q_proj : 1024 -> 2048   (16 heads x 128)
        k/v    : 1024 -> 1024   (8 heads x 128)
        o_proj : 2048 -> 1024   (NOT square)
     The llama `rms_gemv_rope` asserts q_total==emb_dim; the llama
     `o_gemv_ffn` stage-1 O-GEMV is square (emb x emb). We build Qwen
     variants: the Q GEMV is M=q_dim, the O GEMV is M=emb_dim, K=q_dim.

  3. LM-head vocab = 151936 (not 128256). We split it as 9 x 16384 + 4480,
     which sums to the vocabulary EXACTLY -- ten launches and no padding.
     A partition is capped at (255 + 1) * herd_m * m_input rows by the DMA
     repeat count, so 16384 needs m_input 8; at m_input 4 the ceiling is
     8192, which is why this was 19 x 8192 = 155648 with 3712 padded rows.

Decode attention is CPU (decode_attention_cpu), matching llama.
"""

import os
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

from shared.infra.cpu_attn import decode_attention_cpu
from qwen3_0_6b_weights import LlamaConfig
from shared.infra import decode_qkv2 as _qkv2
from shared.infra.cache import KernelCache

# The decode QKV stage ELF. 2 launches: RMSNorm, then ONE head-aligned GEMV
# whose cores apply QK-norm + RoPE in L1 (kernel mv_heads.cc; host layout in
# shared.infra.qkv2_layout, host ABI in shared.infra.decode_qkv2).
# QWEN3_RMS_QKV_LAUNCHES=8 selects the 8-launch form for A/B. Each form's
# artifact name is bound to its own ABI: the 8-launch ELF keeps the name its
# 17-arg caches already carry, the 2-launch ELF gets its own, so a cache can
# never bind one ABI to the other's ELF.
_RMS_QKV_LAUNCHES = int(os.environ.get("QWEN3_RMS_QKV_LAUNCHES", "2"))
assert _RMS_QKV_LAUNCHES in (2, 8), _RMS_QKV_LAUNCHES
_RMS_QKV_KERNEL = {2: "rms_qkv_qknorm_rope_gemv2", 8: "rms_qkv_qknorm_rope_gemv"}[
    _RMS_QKV_LAUNCHES
]

# `[2026-08-26]` doc 56 H2b (queue items 18, 24): the w4_decode path,
# selected by QWEN3_W4_DECODE -- default ON (W4_DEFAULT; the three-arm
# verify lit in this dir is the gate); QWEN3_W4_DECODE=0 selects bf16. What
# changes: the O+FFN stage compiles/dispatches `o_gemv_ffn_int4` (the llama
# 3-launch int4 cascade at q_dim=2048 / k_chunk=1024, SAME launch structure)
# from the packed BOs `w4_decode_pack.quantize_decode_weights` attaches; the
# QKV stage and the LM head stay bf16 (priced negatives -- doc 57 section 5
# item 6). Read at import time like _RMS_QKV_LAUNCHES.
from w4_decode_pack import (
    W4_DEFAULT,
    w4_decode_selected as _w4_decode_selected,
)  # noqa: E402

_W4_DECODE = _w4_decode_selected()
_O_FFN_KERNEL = "o_gemv_ffn_int4" if _W4_DECODE else "o_gemv_ffn"


def required_decode_artifacts():
    """The decode ELF names the CURRENT precision selection dispatches.

    ONE derivation of the set: the same module-level `_RMS_QKV_KERNEL` /
    `_O_FFN_KERNEL` the compile and dispatch paths use, so the check below
    can never drift from what `compile_decode_kernels` writes.
    """
    return (_RMS_QKV_KERNEL, _O_FFN_KERNEL, _LM_KERNEL)


def require_decode_artifacts(cache):
    """`[2026-08-26]` queue item 24 (the w4_decode default flip): refuse a
    decode cache that does not hold the selected precision's ELFs, with the
    fix named.

    Why here rather than at the dispatch: `load_and_run` indexes
    `cache.artifacts[name]`, so a cache compiled BEFORE the flip (it has
    `o_gemv_ffn`, not `o_gemv_ffn_int4`) surfaces as a bare `KeyError` deep
    inside the first decode step -- after the weights loaded, after a prefill
    ran, and with nothing in the message about precision. A stale cache is the
    single most likely consequence of flipping a default, so it gets a
    sentence instead of a traceback.
    """
    missing = [n for n in required_decode_artifacts() if n not in cache.artifacts]
    if not missing:
        return
    sel = "w4_decode" if _W4_DECODE else "bf16"
    other = "QWEN3_W4_DECODE=0" if _W4_DECODE else "QWEN3_W4_DECODE=1"
    raise RuntimeError(
        f"decode cache {str(cache.cache_dir)!r} does not contain {missing} -- it was "
        f"not compiled for the selected precision ({sel}: QWEN3_W4_DECODE is "
        f"{'1' if _W4_DECODE else '0'} here; unset default "
        f"{'1' if W4_DEFAULT else '0'}). It holds {sorted(cache.artifacts)}. Recompile "
        f"(`make compile` with the same flag), or select the other precision "
        f"with {other}."
    )


def build_rms_qkv_qknorm_rope_gemv_module(config, n_launches=None):
    """Fused decode ELF: RMSNorm + Q/K/V GEMV + per-head QK-norm + RoPE (M=1).

    n_launches: 2 (the head-aligned GEMV with the in-core epilogue) or 8 (the
    separate QK-norm and RoPE launches, kept for A/B). Default: `_RMS_QKV_LAUNCHES`.
    """
    from shared.builders.rms_qkv_qknorm_rope_multi import (
        build_rms_qkv_qknorm_rope_gemv_module as _build8,
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
    build = {2: _build2, 8: _build8}[n_launches]
    return build(emb_dim, q_dim, kv_dim, n_heads, n_kv_heads, head_dim, qknorm_eps=1e-6)


def _rms_qkv_qknorm_rope_gemv_backend(verbose=False):
    return {
        "verbose": verbose,
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "rms_qkv_qknorm_rope_gemv",
    }


# The 8-launch ELF's host ABI, in one place: the decode step and the inference
# pre-load used to carry two hand-copied 17-argument lists that had to agree by
# inspection. Positions 13/14 are the RoPE LUTs, which are position-dependent and
# therefore never static. (The 2-launch ABI is `shared.infra.decode_qkv2`'s.)
RMS_QKV_OUTPUT_INDICES = [8, 15, 16]  # v, q_roped, k_roped
RMS_QKV_STATIC_INDICES = {1, 3, 5, 7, 9, 10}  # norm_w, wq, wk, wv, q_norm, k_norm
RMS_QKV_INTERMEDIATE_INDICES = {2, 4, 6, 8, 11, 12, 15, 16}


def rms_qkv_luts(rope_lut_bf16, current_pos, config):
    """Per-head RoPE LUTs for one position: (lut_q, lut_k), each flat bf16."""
    rope_lut_pos = rope_lut_bf16[current_pos : current_pos + 1]  # (1, head_dim)
    lut_q = np.tile(rope_lut_pos, (config.n_heads, 1)).flatten().astype(bfloat16)
    lut_k = np.tile(rope_lut_pos, (config.n_kv_heads, 1)).flatten().astype(bfloat16)
    return lut_q, lut_k


def prep_rms_qkv_weights(layer_weights, config):
    """Once per layer, host side: the static weight of the selected form. Idempotent.
    The 8-launch form binds `_wq_t`/`_wk_t`/`_wv_t` as they are."""
    if _RMS_QKV_LAUNCHES == 2:
        _qkv2.prep_weights_2(layer_weights, config)


def rms_qkv_args(layer_weights, x_bf16, lut_q, lut_k, config):
    """(inputs, output_indices, static_input_indices, intermediate_indices) of
    the QKV stage in the form `_RMS_QKV_KERNEL` names: the 2-launch ABI (5 args,
    one LUT row) or the 8-launch ABI (17 args, the per-head LUTs). `run_rms_qkv`
    and the inference pre-load both go through here, so the args can never
    disagree with the ELF the launch count selects."""
    if _RMS_QKV_LAUNCHES == 2:
        lut_row = np.asarray(lut_q, bfloat16).reshape(-1)[: config.head_dim]
        return (
            _qkv2.call_args_2(layer_weights, x_bf16, lut_row, config),
            _qkv2.OUTPUT_INDICES_2,
            _qkv2.STATIC_INDICES_2,
            _qkv2.INTERMEDIATE_INDICES_2,
        )
    emb_dim = config.emb_dim
    head_dim = config.head_dim
    q_dim = config.n_heads * head_dim
    kv_dim = config.n_kv_heads * head_dim
    inputs = [
        np.asarray(
            x_bf16, bfloat16
        ).flatten(),  # 0 x_in (flatten: always a contiguous copy)
        layer_weights.attn_norm.reshape(emb_dim).astype(bfloat16),  # 1 norm_w (static)
        np.zeros(emb_dim, dtype=bfloat16),  # 2 normed
        layer_weights._wq_t,  # 3 wq (static)
        np.zeros(q_dim, dtype=bfloat16),  # 4 q
        layer_weights._wk_t,  # 5 wk (static)
        np.zeros(kv_dim, dtype=bfloat16),  # 6 k
        layer_weights._wv_t,  # 7 wv (static)
        np.zeros(kv_dim, dtype=bfloat16),  # 8 v
        np.asarray(layer_weights.q_norm, bfloat16).reshape(
            head_dim
        ),  # 9 q_norm (static)
        np.asarray(layer_weights.k_norm, bfloat16).reshape(
            head_dim
        ),  # 10 k_norm (static)
        np.zeros(q_dim, dtype=bfloat16),  # 11 q_n
        np.zeros(kv_dim, dtype=bfloat16),  # 12 k_n
        lut_q,  # 13 lut_q (DYNAMIC -- position-dependent)
        lut_k,  # 14 lut_k (DYNAMIC)
        np.zeros(q_dim, dtype=bfloat16),  # 15 q_roped
        np.zeros(kv_dim, dtype=bfloat16),  # 16 k_roped
    ]
    return (
        inputs,
        RMS_QKV_OUTPUT_INDICES,
        RMS_QKV_STATIC_INDICES,
        RMS_QKV_INTERMEDIATE_INDICES,
    )


def run_rms_qkv(
    cache, layer_weights, x_bf16, lut_q, lut_k, config, layer_idx, verbose=False
):
    """One call of the QKV stage for one layer -> (v, q_roped, k_roped)."""
    inputs, outs, statics, inters = rms_qkv_args(
        layer_weights, x_bf16, lut_q, lut_k, config
    )
    res = cache.load_and_run(
        _RMS_QKV_KERNEL,
        _rms_qkv_qknorm_rope_gemv_backend(verbose),
        *inputs,
        output_indices=outs,
        static_input_indices=statics,
        intermediate_indices=inters,
        bo_key=f"{_RMS_QKV_KERNEL}_L{layer_idx}" if layer_idx is not None else None,
    )
    if _RMS_QKV_LAUNCHES == 2:
        return _qkv2.split_outputs_2(res, config)
    return res[8].astype(bfloat16), res[15].astype(bfloat16), res[16].astype(bfloat16)


# LM-head decode partitioning. vocab=151936.
# Per-partition GEMV broadcasts the K=emb_dim input vector with a hardware
# push_queue repeat_count capped at the [0:255] range, so a partition may be
# at most (255 + 1) * herd_m * m_input rows. At m_input 4 that is 8192, which
# is why this head was 19 x 8192 = 155648: 3712 rows of padding past the
# 151936-row vocabulary, and 19 launch boundaries.
#
# At m_input 8 the ceiling doubles to 16384, and the vocabulary then divides
# EXACTLY as 9 x 16384 + 4480 -- ten launches, no padding at all. The tail
# still sits on the tile grid (4480 % (tile_m 8 * herd_m 8) == 0), which
# `build_lm_head_gemv_module` checks.
_LM_M_INPUT = 8
_LM_PARTS = [16384] * 9 + [4480]  # sums to 151936, the vocabulary exactly
_LM_N_PARTITIONS = len(_LM_PARTS)
# Kept because the shared `run_lm_head` and this model's weight slicing take
# it as the equal-split shorthand; `_LM_PARTS` is what actually decides the
# partitioning now.
_LM_N_PART = _LM_PARTS[0]

# The cache artifact key is versioned with the partitioning, exactly as
# `_RMS_QKV_KERNEL` is versioned with its launch count and for the same
# reason: the host ABI is 1 + 2 * len(_LM_PARTS) buffers, so a decode cache
# compiled before this change holds a 39-argument ELF while the caller now
# passes 21. `load_manifest()` would accept it -- same toolchain, same key --
# and the mismatch surfaces on the device, not at load. A distinct key makes a
# stale cache a clean "recompile" error from `require_decode_artifacts`
# instead. The ELF's own instance_name stays `lm_head_gemv`: it must match the
# func the builder emits.
_LM_KERNEL = f"lm_head_gemv_p{len(_LM_PARTS)}"


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


def build_o_gemv_ffn_int4_qwen_module(emb_dim, q_dim, hidden_dim):
    """w4_decode O+FFN: the llama 3-launch int4 cascade (matvec_int4_packed_add
    / swiglu_rms / packed_add over one `mv_int4_bf16.o`), decoupled exactly as
    `build_o_gemv_ffn_qwen_module` decouples the bf16 cascade (O GEMV M=emb,
    K=q_dim) and at k_chunk=emb_dim (stage 2 requires K == K_CHUNK; O and
    down split into 2 / 3 chunks). Same 15-arg ABI, arg1 is q_dim wide,
    arg0/7/12 are packed-uint8 BOs. Thin delegate -- the llama builder is the
    one owner (doc 56 H2b: REUSE the existing int4 builders)."""
    from llama32_1b_int4.multi_launch_builder.o_gemv_ffn_int4_multi import (
        build_o_gemv_ffn_int4_module,
    )
    from w4_decode_pack import GROUP_SIZE, M_TILE, K_CHUNK, N_CORES

    assert K_CHUNK == emb_dim, (K_CHUNK, emb_dim)
    return build_o_gemv_ffn_int4_module(
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        gs=GROUP_SIZE,
        m_tile=M_TILE,
        k_chunk=K_CHUNK,
        n_cores=N_CORES,
        q_dim=q_dim,
        # Qwen3's RMS eps is 1e-6 (the model contract; matches qknorm_eps at
        # build_rms_qkv and inference EPS). Llama callers keep the builder
        # default 1e-5. NOTE: the bf16 sibling stage 2 (matvec_swiglu_rms)
        # still hard-codes 1e-5 on main -- pre-existing, logged as a
        # follow-up, outside this PR's diff.
        eps=1e-6,
    )


# ---------------------------------------------------------------------------
# Builder 2: LM-head GEMV (9 x 16384 + 4480 for vocab 151936, m_input 8).
# ---------------------------------------------------------------------------


def build_lm_head_gemv_qwen_module(emb_dim):
    from shared.builders.lm_head_gemv_multi import build_lm_head_gemv_module

    return build_lm_head_gemv_module(
        emb_dim=emb_dim,
        parts=_LM_PARTS,
        tile_m=8,
        m_input=_LM_M_INPUT,
        herd_m=8,
    )


# ---------------------------------------------------------------------------
# Backend kwargs
# ---------------------------------------------------------------------------


def _o_gemv_ffn_backend(verbose=False):
    if _W4_DECODE:
        return _o_gemv_ffn_int4_backend(verbose)
    return {
        "verbose": verbose,
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "o_gemv_ffn",
        "use_lock_race_condition_fix": False,
    }


def _o_gemv_ffn_int4_backend(verbose=False):
    """The llama int4 cascade's preset (ping-pong on; the llama study measured
    a large e2e decode regression without it -- artifact not ported, so no
    number is claimed here; the preset is owned by shared.infra)."""
    from shared.infra.backend_presets import OGF_INT4_BACKEND

    return {"verbose": verbose, **OGF_INT4_BACKEND}


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


def _sibling_o_ffn_entry(cache):
    """`[2026-08-26]` queue item 24: the manifest entry for the OTHER
    precision's O+FFN ELF, if this cache already holds one.

    `QWEN3_W4_DECODE` is documented as an A/B knob, and `o_gemv_ffn` /
    `o_gemv_ffn_int4` is the ONE artifact that differs between the two
    precisions. `_save_manifest` writes exactly what the current compile
    produced, so without this a `make compile` at the default would erase the
    bf16 entry (the ELF stays on disk; the manifest stops naming it) and every
    bf16 consumer -- `QWEN3_W4_DECODE=0 make run`, a bf16 A/B rung on
    `build_peano` -- would refuse until someone recompiled.

    Deliberately NARROW: exactly this one name is carried across, never the
    whole previous manifest. Resurrecting every stale entry is the hazard this
    avoids while still letting one build tree serve both precisions.
    """
    import json as _json

    name = "o_gemv_ffn" if _W4_DECODE else "o_gemv_ffn_int4"
    man = Path(cache.cache_dir) / cache.MANIFEST_FILE
    if not man.is_file():
        return None
    try:
        data = _json.loads(man.read_text())
    except (ValueError, OSError):
        return None
    # Review of #33, P1: never carry across a toolchain change. The normal
    # cache path treats a different or missing `_toolchain` stamp as cold
    # (KernelCache.load_manifest); the carry must not be a side door that
    # re-stamps a stale ELF as current after `_save_manifest`.
    if data.get("_toolchain") != cache._toolchain_id():
        return None
    info = data.get("entries", {}).get(name)
    if not info or not info.get("output_binary"):
        return None
    for cand in (
        Path(info["output_binary"]),
        Path(cache.cache_dir) / Path(info["output_binary"]).name,
    ):
        if cand.is_file():
            return name, dict(info, output_binary=str(cand))
    return None


def _restore_sibling_o_ffn(cache, carried):
    """Put the entry `_sibling_o_ffn_entry` found back, after the compile."""
    if not carried:
        return
    from air.backend.xrt import XRTCompileArtifact

    name, info = carried
    cache.artifacts[name] = XRTCompileArtifact(
        info["output_binary"], info["kernel"], info.get("insts")
    )
    print(
        f"  carried over the other precision's O+FFN ELF: {name} "
        f"({info['output_binary']}) -- this cache serves both precisions"
    )


def compile_decode_kernels(cache, config, verbose=False):
    """Compile the Qwen3 decode kernels."""
    from shared.infra.external_kernels import (
        compile_mv,
        compile_mv_bf16,
        compile_mv_heads,
        compile_rope,
        compile_silu_and_mul,
    )

    emb_dim = config.emb_dim
    hidden_dim = config.hidden_dim
    q_dim = config.n_heads * config.head_dim

    # read BEFORE anything is written; restored after (queue item 24)
    carried = _sibling_o_ffn_entry(cache)

    print(
        f"\n{'='*60}\nCompiling Qwen3 decode kernels "
        f"({'w4_decode int4' if _W4_DECODE else 'bf16'} O+FFN)...\n{'='*60}\n"
    )

    # External .o kernels: GEMV (mv.o), head-aligned GEMV + epilogue
    # (mv_heads_hd{head_dim}.o), 2tile-add/swiglu (mv_bf16.o), RoPE.
    compile_mv()
    compile_mv_heads(config.head_dim)
    compile_mv_bf16()
    compile_rope()
    compile_silu_and_mul()

    print(
        f"\n--- {_RMS_QKV_KERNEL} (FUSED: RMSNorm+QKV+QK-norm+RoPE, "
        f"{_RMS_QKV_LAUNCHES} launches) ---"
    )
    cache.compile_and_cache(
        _RMS_QKV_KERNEL,
        build_rms_qkv_qknorm_rope_gemv_module(config),
        _rms_qkv_qknorm_rope_gemv_backend(verbose),
    )

    if _W4_DECODE:
        from w4_decode_pack import GROUP_SIZE, K_CHUNK

        print("\n--- o_gemv_ffn_int4 (w4_decode: int4 O GEMV decoupled + FFN) ---")
        cache.compile_and_cache(
            "o_gemv_ffn_int4",
            build_o_gemv_ffn_int4_qwen_module(emb_dim, q_dim, hidden_dim),
            # int4_gs / int4_k_chunk ride the backend kwargs so the per-compile
            # kernel sweep stages THIS model's mv_int4_bf16.o (DIM_K=1024), not
            # llama's 2048 default (cache.compile_and_cache pops them).
            {
                **_o_gemv_ffn_int4_backend(verbose),
                "int4_gs": GROUP_SIZE,
                "int4_k_chunk": K_CHUNK,
            },
        )
    else:
        print("\n--- o_gemv_ffn (O GEMV decoupled + Residual + FFN) ---")
        cache.compile_and_cache(
            "o_gemv_ffn",
            build_o_gemv_ffn_qwen_module(emb_dim, q_dim, hidden_dim),
            _o_gemv_ffn_backend(verbose),
        )

    print("\n--- lm_head_gemv (9 x 16384 + 4480, vocab 151936) ---")
    cache.compile_and_cache(
        _LM_KERNEL,
        build_lm_head_gemv_qwen_module(emb_dim),
        _lm_gemv_backend(verbose),
    )

    _restore_sibling_o_ffn(cache, carried)
    cache._save_manifest()
    print(f"\nAll {len(cache.artifacts)} decode kernels compiled.")


# ---------------------------------------------------------------------------
# CPU decode attention (with KV cache)
# ---------------------------------------------------------------------------


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

    Stages: the QKV stage ELF `_RMS_QKV_KERNEL` (NPU: RMSNorm + Q/K/V GEMV +
    per-head QK-norm + RoPE) -> KV-cache write -> CPU attention -> o_gemv_ffn (NPU).
    """
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    layer_idx = getattr(layer_weights, "_layer_idx", None)

    # --- One ELF = RMSNorm + Q/K/V GEMV + per-head QK-norm + RoPE ---
    lut_q, lut_k = rms_qkv_luts(rope_lut_bf16, current_pos, config)
    v, q_roped, k_roped = run_rms_qkv(
        cache, layer_weights, x_bf16, lut_q, lut_k, config, layer_idx, verbose
    )

    # --- Update KV cache (K after qk-norm AND rope; V raw projection) ---
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


# Cache of dead-ABI placeholders for the w4 path (the llama int4 pattern:
# reallocating the hidden x emb buffer per call is pure host glue).
_DEAD_PLACEHOLDERS = {}


def _dead_buf(shape):
    key = shape if isinstance(shape, tuple) else (shape,)
    buf = _DEAD_PLACEHOLDERS.get(key)
    if buf is None:
        buf = np.zeros(shape, dtype=bfloat16)
        _DEAD_PLACEHOLDERS[key] = buf
    return buf


def _run_o_gemv_ffn_int4(
    attn_out, x_bf16, layer_weights, config, cache, layer_idx, verbose=False
):
    """w4_decode Stage E: int4 O-proj(decoupled) + Residual + RMSNorm + SwiGLU.

    Same 15-arg ABI and BO indices as the bf16 cascade; slots 0/7/12 hold the
    packed-uint8 BOs `w4_decode_pack.quantize_decode_weights` attached."""
    emb_dim = config.emb_dim
    hidden_dim = config.hidden_dim
    z_emb = _dead_buf(emb_dim)
    z_hidden = _dead_buf(hidden_dim)
    z_hidden_emb = _dead_buf((hidden_dim, emb_dim))
    results = cache.load_and_run(
        "o_gemv_ffn_int4",
        _o_gemv_ffn_int4_backend(verbose),
        layer_weights._wo_packed,  # arg0 wo (static, packed-i8, decoupled K=q_dim)
        attn_out,  # arg1 attn_out (q_dim)
        z_emb,  # arg2 (dead)
        x_bf16.flatten().astype(bfloat16),  # arg3 x_residual
        z_emb,  # arg4 (dead)
        z_emb,  # arg5 (dead)
        layer_weights._packed_rms_buf,  # arg6 packed RMS input (static)
        layer_weights._wgateup_packed,  # arg7 gate/up (static, packed-i8)
        z_hidden,  # arg8 (dead)
        z_hidden_emb,  # arg9 (dead)
        z_hidden,  # arg10 (dead)
        z_hidden,  # arg11 swiglu
        layer_weights._wdown_packed,  # arg12 wdown (static, packed-i8)
        z_emb,  # arg13 (dead)
        z_emb,  # arg14 output
        output_indices=[14],
        static_input_indices={0, 6, 7, 12},
        intermediate_indices={2, 4, 5, 8, 9, 10, 11, 13, 14},
        bo_key=f"o_gemv_ffn_int4_L{layer_idx}" if layer_idx is not None else None,
    )
    return results[14].astype(bfloat16)


def _run_o_gemv_ffn(
    attn_out, x_bf16, layer_weights, config, cache, layer_idx, verbose=False
):
    """Decode Stage E: O-proj(decoupled) + Residual + RMSNorm + SwiGLU FFN.

    Shared by the fused and legacy decode paths so the o_gemv_ffn arg layout +
    BO indices have a single owner. Dispatches the w4_decode int4 cascade when
    QWEN3_W4_DECODE selected it (same launch structure, packed weights).
    """
    if _W4_DECODE:
        return _run_o_gemv_ffn_int4(
            attn_out, x_bf16, layer_weights, config, cache, layer_idx, verbose
        )
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
