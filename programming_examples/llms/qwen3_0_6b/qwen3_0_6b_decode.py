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

# `[2026-08-26]` doc 56 H2b (queue items 18, 24): the w4_decode path,
# selected by QWEN3_W4_DECODE -- **default ON**, the production default for
# this model since 2026-08-26; QWEN3_W4_DECODE=0 selects bf16 for A/B. What
# changes: the O+FFN stage compiles/dispatches `o_gemv_ffn_int4` (the llama
# 3-launch int4 cascade at q_dim=2048 / k_chunk=1024, SAME launch structure)
# from the packed BOs `w4_decode_pack.quantize_decode_weights` attaches; the
# QKV stage and the LM head stay bf16 (priced negatives -- see
# results/item18-h2b-20260826/PREDICTION.md section 2 and doc 57 section 5
# item 6). Read at import time like _RMS_QKV_LAUNCHES; model_adapter.prepare
# sets the env from `precision_plan` before importing the driver.
from w4_decode_pack import w4_decode_selected as _w4_decode_selected  # noqa: E402

_W4_DECODE = _w4_decode_selected()
_O_FFN_KERNEL = "o_gemv_ffn_int4" if _W4_DECODE else "o_gemv_ffn"


def required_decode_artifacts():
    """The decode ELF names the CURRENT precision selection dispatches.

    ONE derivation of the set: the same module-level `_RMS_QKV_KERNEL` /
    `_O_FFN_KERNEL` the compile and dispatch paths use, so the check below
    can never drift from what `compile_decode_kernels` writes.
    """
    return (_RMS_QKV_KERNEL, _O_FFN_KERNEL, "lm_head_gemv")


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
        f"not compiled for the selected precision ({sel}; QWEN3_W4_DECODE "
        f"default is {'1' if _W4_DECODE else '0'} since 2026-08-26, doc 56 H2b "
        f"queue item 24). It holds {sorted(cache.artifacts)}. Recompile "
        f"(`make compile`, or `run_model.py compile-decode --precision-plan "
        f"{sel}`), or select the other precision with {other}."
    )


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
# push_queue repeat_count, capped at the [0:255] range. The repeat is
#     M / (herd_m * m_input * herd_rows) - 1
# so at m_input 8 and ONE core row a partition may carry at most 16384 rows;
# past that aircc refuses with
#     error: 'aiex.npu.push_queue' op Repeat count exceeds the [0:255] range
# (item 28, devq 691 leg 5, the exact text). 19 * 8192 was the pre-2026-08-21
# uniform form at m_input 4 (3712 pad rows, logits truncated on host).
# `[2026-08-21]` Mixed partitions: full partitions at the cap plus one tail on
# the tile grid -- 10 launches and 64 pad rows at one row. Measured 9.35 ->
# 8.25 ms per token on the probe (devq 476); the planner (shared/plan, doc 56
# H0) derived it. Gated by `make verify`.
#
# `[2026-08-26]` queue item 28 -- THE LM HEAD FILLS MORE THAN ONE CORE ROW, AND
# THE PARTITIONING FOLLOWS FROM THAT.
#
# The GEMV built `@herd(sizes=[herd_m, 1])`: 8 columns x ONE compute row, i.e.
# 8 of the device's 32 compute tiles, with all eight shim/DDR paths already in
# use (item 27, commit 2e14f533). Item 27 measured its standalone bf16 GEMV at
# 35.72 / 44.43 / 50.03 GB/s over 8 / 16 / 32 cores and predicted -12 % here.
# MEASURED ON THIS HEAD (devq 688), rows alone are a NULL: 7.58 -> 7.55 ms at
# two rows, +9.1 % at four. The reason is that this head, unlike that harness,
# was ALREADY near the device's read ceiling -- 311.16 MB in 7.58 ms is
# 41.1 GB/s end to end and ~48 GB/s once its ten launch boundaries are charged,
# against item 27's measured device-wide 50-54 GB/s.
#
# What rows DO buy is the repeat cap: it scales with `herd_rows`, so a
# partition may carry 16384 / 32768 / 65536 rows at 1 / 2 / 4 rows, and the
# head needs 10 / 5 / 3 launches to cover the vocab. At ~130 us per launch
# boundary that is the win: devq 691 measured the same 311.16 MB at
# 7.663 / 6.844 / 6.470 ms over 10 / 5 / 3 launches (-10.7 % and -15.6 %), all
# at 0 violations of a derived per-element bound. **The partitioning is
# therefore DERIVED from the row count rather than written down**, so the two
# cannot disagree and `QWEN3_LM_HERD_ROWS=1` reproduces the pre-item-28 head
# byte for byte.
#
# A value > 1 REQUIRES aircc's `--use-lock-race-condition-fix` or the device
# hangs with ERT_CMD_STATE_TIMEOUT (item 27 section 6.1). `[2026-08-27, review
# round 6]` The flag is NOT derived from this constant, and the comment that
# used to stand here described a rule that was WITHDRAWN: it said every
# multi-row herd gets the flag "with no exemption list", which is what devq
# 812/813 measured FAULTING the device on the QKV split-cast form. What
# actually happens is narrower and does not consult this constant at all --
# `matvec.py`, which builds this kernel, stamps `air.lock_race_fix_required` on
# any herd it emits above one row, and `KernelCache.compile_and_cache` supplies
# the flag iff that mark is on the module. So `lm_head_gemv` is covered because
# of what its BUILDER emitted (verified on the real module), not because of
# anything written here; a herd built some other way at rows > 1 would NOT be
# covered. Stating it at the call site keeps the requirement legible.
_LM_TILE_M = 8
_LM_HERD_M = 8
_LM_M_INPUT = 8
_LM_BD_REPEAT_CAP = 255  # aiex.npu.push_queue's [0:255]
_LM_VOCAB = 151936
_LM_HERD_ROWS = int(_os.environ.get("QWEN3_LM_HERD_ROWS", "4"))
assert _LM_HERD_ROWS in (1, 2, 4), (
    f"QWEN3_LM_HERD_ROWS={_LM_HERD_ROWS}: NPU2 has four core rows and the herd "
    "grid is a power of two (1, 2 or 4)"
)


def lm_head_parts(herd_rows=None):
    """Partition row counts at `herd_rows`: full partitions at the BD repeat
    cap, plus one tail rounded up to the tile grid.

    At 1 / 2 / 4 rows this is 9x16384+4480 / 4x32768+20864 / 2x65536+20864 --
    10, 5 and 3 launches, and every one of them covers the vocab exactly (0 pad
    rows at 2 and 4 rows, 64 at 1).
    """
    herd_rows = _LM_HERD_ROWS if herd_rows is None else herd_rows
    cap = (_LM_BD_REPEAT_CAP + 1) * _LM_HERD_M * _LM_M_INPUT * herd_rows
    grid = _LM_TILE_M * _LM_HERD_M
    full, rem = divmod(_LM_VOCAB, cap)
    return tuple([cap] * full + ([-(-rem // grid) * grid] if rem else []))


def lm_head_herd_rows(parts=None, want=None):
    """Per-partition `herd_rows`, halved for any partition it does not divide.

    A partition of M rows can only use R rows when M % (tile_m*herd_m*R) == 0.
    The 20864-row tail divides by 128 but not by 256, so at `want=4` it caps at
    2 while the 65536-row partitions take 4. Returning a per-partition tuple
    (which `build_lm_head_gemv_module` accepts) is what keeps the tail legal
    without shrinking the whole head to the tail's limit.
    """
    want = _LM_HERD_ROWS if want is None else want
    out = []
    for rows in lm_head_parts(want) if parts is None else parts:
        r = want
        while r > 1 and rows % (_LM_TILE_M * _LM_HERD_M * r):
            r //= 2
        # `[2026-08-26]` The two caps pull in OPPOSITE directions and a tail can
        # fall between them: halving `r` for divisibility DOUBLES the broadcast
        # repeat, so a partition that divides at 2 rows may exceed the [0:255]
        # range that only 4 rows would have satisfied. It does not happen at
        # this model's m_input 8 (the 20864 tail reads 162), and it DOES happen
        # at m_input 4 (the same tail would read 325) -- which is why this is an
        # assertion and not a comment. The fix when it fires is to round the
        # tail up to `tile_m * herd_m * want` in `lm_head_parts` so it can take
        # the full row count, at the cost of a few pad rows.
        repeat = rows // (_LM_HERD_M * _LM_M_INPUT * r) - 1
        assert repeat <= _LM_BD_REPEAT_CAP, (
            f"partition of {rows} rows at herd_rows={r} (capped down from "
            f"{want} for divisibility) needs a broadcast repeat of {repeat}, "
            f"past the [0:{_LM_BD_REPEAT_CAP}] range -- aircc refuses with "
            f"\"'aiex.npu.push_queue' op Repeat count exceeds the [0:255] "
            f"range\". Round the tail up to tile_m*herd_m*{want} in "
            f"lm_head_parts so it can take {want} rows."
        )
        out.append(r)
    return tuple(out)


_LM_PARTS = lm_head_parts()
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
    )


# ---------------------------------------------------------------------------
# Builder 2: LM-head GEMV (19 partitions, n_part=8192 for vocab 151936).
# ---------------------------------------------------------------------------


def build_lm_head_gemv_qwen_module(emb_dim):
    from shared.builders.lm_head_gemv_multi import build_lm_head_gemv_module

    return build_lm_head_gemv_module(
        emb_dim=emb_dim,
        parts=_LM_PARTS,
        tile_m=_LM_TILE_M,
        # m_input 8 (one kernel call per 8-row tile, B-broadcast repeat ~127)
        # measured 8.5 % faster than m_input 4 at the same launch count and
        # bytes (doc 57 section 1.4, devq 449: 9.12 vs 9.96 ms); gated by
        # `make verify`.
        m_input=_LM_M_INPUT,
        herd_m=_LM_HERD_M,
        # queue item 28: 4 core rows per full partition (32 of the 32 compute
        # tiles), 2 on the tail. Coupled to `use_lock_race_condition_fix` in
        # `_lm_gemv_backend` below -- see `_LM_HERD_ROWS`.
        herd_rows=lm_head_herd_rows(_LM_PARTS),
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
        # `[2026-08-27]` queue item 28: no explicit False. This cascade's stage 2
        # (`matvec_swiglu_rms`, n_cascade=4) is an 8 x 4 herd, so the compile
        # chokepoint supplies the lock-race fix; writing False here would be
        # refused as a contradiction rather than silently overridden.
    }


def _o_gemv_ffn_int4_backend(verbose=False):
    """The llama int4 cascade's preset (ping-pong on -- dropping it regressed
    the llama e2e 12.4 -> 7.8 tok/s)."""
    from shared.infra.backend_presets import OGF_INT4_BACKEND

    return {"verbose": verbose, **OGF_INT4_BACKEND}


def _lm_gemv_backend(verbose=False):
    # `[2026-08-27]` queue item 28: this dict says NOTHING about the lock-race
    # fix, on purpose. `matvec.py` marks its own herd at `herd_rows > 1` and
    # `KernelCache.compile_and_cache` supplies the flag for that mark -- one
    # trigger, in one place. A row-count-driven helper used to live here too,
    # and a second trigger is how the flag reached kernels it faults (devq
    # 812/813).
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
    bf16 consumer -- `QWEN3_W4_DECODE=0 make run`, the study's bf16 decode
    rungs on `build_peano` -- would refuse until someone recompiled.

    Deliberately NARROW: exactly this one name is carried across, never the
    whole previous manifest. Resurrecting every stale entry (this cache also
    holds `rms_qkv_qknorm_rope_gemv` / `_gemv4` ELFs from an older selection)
    is the hazard this avoids while still letting one build tree serve both
    precisions.
    """
    import json as _json

    name = "o_gemv_ffn" if _W4_DECODE else "o_gemv_ffn_int4"
    man = Path(cache.cache_dir) / cache.MANIFEST_FILE
    if not man.is_file():
        return None
    try:
        info = _json.loads(man.read_text()).get(name)
    except (ValueError, OSError):
        return None
    if not info or not info.get("output_binary"):
        return None
    for cand in (Path(info["output_binary"]),
                 Path(cache.cache_dir) / Path(info["output_binary"]).name):
        if cand.is_file():
            info = dict(info, output_binary=str(cand))
            return name, info
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
    if info.get("launches"):
        cache.launch_counts[name] = info["launches"]
    if info.get("n_args") is not None:
        cache.arg_counts[name] = int(info["n_args"])
    print(f"  carried over the other precision's O+FFN ELF: {name} "
          f"({info['output_binary']}) -- this cache serves both precisions")


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

    # read BEFORE anything is written; restored after (queue item 24)
    carried = _sibling_o_ffn_entry(cache)

    print(f"\n{'='*60}\nCompiling Qwen3 decode kernels "
          f"({'w4_decode int4' if _W4_DECODE else 'bf16'} O+FFN)...\n{'='*60}\n")

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

    if _W4_DECODE:
        from w4_decode_pack import GROUP_SIZE, K_CHUNK

        print("\n--- o_gemv_ffn_int4 (w4_decode: int4 O GEMV decoupled + FFN) ---")
        cache.compile_and_cache(
            "o_gemv_ffn_int4",
            build_o_gemv_ffn_int4_qwen_module(emb_dim, q_dim, hidden_dim),
            # int4_gs / int4_k_chunk ride the backend kwargs so the per-compile
            # kernel sweep stages THIS model's mv_int4_bf16.o (DIM_K=1024), not
            # llama's 2048 default (cache.compile_and_cache pops them).
            {**_o_gemv_ffn_int4_backend(verbose), "int4_gs": GROUP_SIZE, "int4_k_chunk": K_CHUNK},
        )
    else:
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

    _restore_sibling_o_ffn(cache, carried)
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
