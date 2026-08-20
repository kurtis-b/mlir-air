# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``fused`` — MLIR-level fusion via ``stitch_elf``: one runlist, three ELFs.

CONTRACT
    ``prepare_fused(shape, seed=...)`` is this mode's entry in the ``SPECS``
    catalogue: the D2 layer prepared for ``opcheck.py``'s ``dispatch`` seam,
    executed as ONE ``KernelCache.run_sequence`` call over three entries —
    forced single-submission — so the whole layer is one host submission::

        entry 1  qkv_proj    the D2 module, unchanged     (fused-cast GEMM)
        entry 2  mha_out_proj  the D2 module, unchanged   (attention + o_proj)
        entry 3  fused_tail  ONE stitched module: the norm_tail PIPELINE
                 (ln1), the whole staged FFN, the norm_tail PIPELINE (ln2)

    `[2026-08-08]` Both normalization points are J7a's three-herd pipeline
    (``builders/norm_tail.py``), not the decomposed add/LayerNorm/gamma-mul
    launches this mode first shipped. That deletes four whole [seq, emb]
    intermediates from L3 -- ln1_sum, ln1_norm, ln2_sum, ln2_norm -- and drops
    both gammas from a host-materialized [seq, emb] broadcast to the [emb]
    weight itself.

    IT BUYS NO MEASURABLE LATENCY, and the first write-up of this claimed it
    did. At 1024 the pipeline measures 98.5/99.2/100.2 ms against the
    decomposed tail's 97.7/99.0/106.5 -- overlapping ranges, +1.0% on means
    once the decomposed tail's own high outlier is dropped. The -5.9% first
    reported here came from comparing three fresh runs against that single
    outlier, which is the same mistake, on the same day, as the 731 ms entry
    doc 09 carries a correction for. Compare distributions, not a run against
    a number.

    What it does buy is verified correctness (10/10 boundaries clean at 256,
    512 and 1024) and a structurally shorter tail. Whether that is worth
    anything is not yet measurable here: see the bytes_transferred footgun --
    the traffic this removes is device-resident and nothing in the port
    records it. BOUNDED TO 256..1024; see the packing and floor notes below.

    THREE THINGS THIS COST, all of them recorded because none is obvious:
      - ``packed1``/``packed2`` are PLANE-MAJOR, so plane 0 is contiguous at
        offset 0 and the GEMM that writes it needs no change. That caps the
        mode at rows*cols <= 2^20 (1024 rows at emb 768). Row-interleaving has
        no cap but demands a strided producer, which means the SHARED GEMM
        builder.
      - ``hidden`` is written TWICE by ln1's stage_scale (``mirror_out``): once
        to its own buffer, which the FFN reads at a base, and once into
        packed2's plane 1, which ln2 reads. The FFN cannot take a plane offset
        without that same shared-builder change.
      - ``ffn_out`` and ``packed2`` are two typed func args over ONE buffer
        (``_TAIL_BUFFER_ALIAS``); stitch_elf wires args by position with
        matching types, so the 2-D output view cannot map onto the 3-D packed
        arg inside the module, and needs no reconciling outside it.

    Every intermediate stays device-resident: q/k/v flow from entry 1 to entry
    2 and ``attn_out`` from entry 2 to entry 3 through the BO pool, and inside
    ``fused_tail`` the normalization sums, the FFN staging and ``hidden``
    (which feeds the FFN AND the second residual add) are shared func args no
    host sync ever touches. That is the boundary this mode isolates: coarse
    crosses 402 sync boundaries because its row-blocked ``addnorm`` restages
    64 bands per normalization point through the host; fusing the norms into
    a stitched module over whole tensors removes all of it.

WHY THREE ELFs AND NOT ONE, WHICH IS THE MODE'S OWN FINDING
    A whole-layer single-module stitch is not blocked by symbol collisions —
    E1's ``(method, tile_n)`` naming fix removed those — but by BACKEND
    SETTINGS: one ELF is one aircc invocation, and FlashAttention requires
    ``runtime_loop_tiling_sizes=[1, 1]`` while the 4096-row GEMMs are built at
    ``[2, 2]`` for BD-ID recycling. So attention keeps its own ELF, qkv (which
    must run BEFORE it) keeps its own, and everything after it fuses into one.
    The ELF ABI then aggregates all three into one runlist (05a §5), which is
    what makes the layer a single submission anyway.

    ``[2026-08-08]`` **This is now measured, and it is a HANG rather than
    either thing the paragraph used to predict.** ``mha_out_proj`` at 4096
    under ``[2, 2]`` compiles and then returns ``ERT_CMD_STATE_TIMEOUT``, 3/3,
    against 3/3 clean passes at ``[1, 1]``. "A placement failure at best, wrong
    numbers at worst" was the right instinct and neither disjunct: it places,
    it never returns.

    Two corrections that came with the measurement, so nobody re-derives them:
    ``omit_pingpong="all"`` is **not** part of the conflict — ping-pong ON
    passes at this shape with byte-identical statistics — and the claim that
    the two settings produce different programs is **not** visible in the
    lowered IR, which is identical op-for-op between them. A compile-only
    comparison of the two therefore "refutes" this paragraph and is wrong to;
    that is exactly what doc 26 §4 did before its retraction. Reproduce with
    ``agents/probes/probe_backend_preset_hardware.py``.

WHY THE NORMALIZATION IS STREAMED, NOT ROW-BLOCKED
    ``build_addnorm_module`` caps a launch at 104 rows of 768 (L1), so a
    faithful re-use of coarse's operator inside one module would need 64
    launches per normalization point, each reading a 64-row band of a whole
    tensor — and a band at a nonzero row offset cannot be routed into a
    slice's args clause: ``memref.cast`` cannot cast an offset subview back
    to the identity layout the launch signature declares (the row-0 trick in
    ``o_gemv_ffn_multi.py`` works only at offset 0). The decomposed
    ``elementwise_add`` / ``layer_norm`` / ``elementwise_mul`` builders have
    no such cap — each walks all 4096 rows in one launch, and all three are
    validated standalone at exactly 4096x768 in the SPECS catalogue. They are
    the same decomposition ``runlist`` measured clean at every boundary, at
    streaming rather than banded granularity: the per-row arithmetic is
    identical, only the dispatch shape differs — which is precisely the
    variable this mode is supposed to move.

FOOTGUNS
    - ``FUSED_CACHE_DIR`` is this mode's OWN directory, in
      ``transformer_layer/.gitignore`` AND the Makefile ``clean`` target.
      ``KernelCache`` picks the directory by NAME; two modes sharing one can
      trade ELFs whose fingerprints happen to agree.
    - NO ELF EXECUTES TWICE. Each of the three entries is its own artifact
      with its own ``hw_context``; the reused-context corruption ``runlist``
      measured (wrong numbers from a GEMM ELF's second execution onward)
      is unreachable here by construction.
    - The dispatch vector is recorded on the fault-injected path too. The
      driver requires the fault artifact's summed totals to EQUAL the clean
      run's; anything conditional on the injected flag fails that.
    - The mode computes; the oracle checks. Nothing here imports a function
      of ``pattern/reference.py`` that computes a boundary — the device does
      all of it, and the golden model only supplies inputs and expectations.
    - ``execution_mode`` comes from ``pattern.EXECUTION_MODE_CSV`` — the CSV
      value is ``fused_elf``. Do not inline the string.
    - ``bytes_transferred`` CANNOT SEE what the pipeline saves. It counts
      host<->device sync traffic only, and the four intermediates removed were
      device-resident, so the column rises here (verification reads back both
      whole packed buffers for two boundary tensors, and x is uploaded twice)
      while the traffic that actually fell is unmeasured. Latency is the only
      proxy this port has for it today.
    - ln2's gamma is now the ``[emb]`` weight, like ln1's. Both are static and
      content-keyed; under fault injection the key changes and the perturbed
      weight is re-uploaded, and nothing special-cases the injected path.
"""

import os
import re
import sys

import numpy as np
from ml_dtypes import bfloat16

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE_ROOT = os.path.dirname(os.path.dirname(_HERE))  # transformer_layer/
_PROJ_ROOT = os.path.dirname(_EXAMPLE_ROOT)  # programming_examples/
for _p in (_PROJ_ROOT, os.path.join(_PROJ_ROOT, "llms"), _EXAMPLE_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import shared.infra.external_kernels as ek  # noqa: E402
from shared.infra.bo_pool import BufferSpec, DispatchStep, content_key  # noqa: E402
from shared.infra.stitching import KernelSlice, stitch_elf  # noqa: E402

from builders.block import block_config  # noqa: E402
from builders.block_cache import (  # noqa: E402
    block_artifact_fingerprint,
    load_fingerprints,
    save_fingerprints,
)
from builders.elementwise_add import build_elementwise_add_module  # noqa: E402
from builders.elementwise_mul import (  # noqa: E402
    broadcast_row_weight,
    build_elementwise_mul_module,
)
from builders.ffn import build_ffn_module, ffn_arg_layout  # noqa: E402
from builders.layer_norm import build_layer_norm_module  # noqa: E402
from builders.norm_tail import build_norm_tail_module  # noqa: E402
from builders.mha_out_proj import (  # noqa: E402
    MHA_OUT_PROJ_RUNNER_KWARGS,
    build_mha_out_proj_module,
    compile_mha_out_proj_kernels,
)
from builders.qkv_proj import build_qkv_proj_module  # noqa: E402
from opcheck_layer import (  # noqa: E402
    BLOCK_STAGE_ATOL,
    DECODER_STAGE_ATOL,
    print_dispatch_totals,
    reconfiguration_delta,
)
from opcheck_prepare import _spec_digest  # noqa: E402
from pattern import EXECUTION_MODE_CSV  # noqa: E402
from pattern.blocked_attention import round_bf16  # noqa: E402
from pattern.reference import (  # noqa: E402
    DECODER_BOUNDARIES,
    ENCODER_BOUNDARIES,
    fuse_qkv_weight,
    generate_golden_reference,
)

#: This mode's ELF cache, relative to the working directory. Its OWN — see the
#: module footguns.
FUSED_CACHE_DIR = "fused_cache"

#: The host tensors the layer takes, in the order ``opcheck.py`` indexes them
#: for fault injection. Identical to the block's: the QKV weight is FUSED,
#: because entry 1 is the block's own qkv_proj module.
FUSED_INPUT_NAMES = (
    "x",
    "w_qkv",
    "w_o",
    "ln1_weight",
    "w_up",
    "w_down",
    "ln2_weight",
)

#: Backend settings for the two GEMM-backed ELFs (qkv_proj and fused_tail):
#: BD-ID recycling for 4096-row herds and the ELF ABI. The settings D1/D2
#: validated, unchanged — and the reason the attention ELF cannot join the
#: stitch (it needs MHA_OUT_PROJ_RUNNER_KWARGS instead; see module docstring).
_GEMM_BACKEND = {
    "runtime_loop_tiling_sizes": [2, 2],
    "output_format": "elf",
    "omit_while_true_loop": False,
}

#: Tail func args that are backed by another arg's buffer. ``ffn_out`` is the
#: FFN slice's 2-D output view of ``packed2``'s plane 0: one BO, two typed args,
#: because the types cannot be reconciled inside the stitched module and do not
#: need to be outside it.
_TAIL_BUFFER_ALIAS = {"ffn_out": "packed2"}

#: The stitched tail's combined func name, and therefore its instance_name — a
#: mismatch does not fail to load, it times out far from the cause.
_TAIL_FUNC = "fused_tail"

#: The DECODER's two stitched func names. The decoder variant has no packed
#: plane anywhere: its pre-norm normalizes `x` alone (no residual add), and
#: its raw residual stream flows through bare add slices over separate args,
#: so the plane-major packing — and its rows*cols cap — never enters.
_PRE_NORM_FUNC = "decoder_pre_norm"
_DECODER_TAIL_FUNC = "decoder_tail"


def _private_syms(ir_text):
    """Every ``func.func private`` symbol in a slice's IR — its extern set.

    These are the external-kernel entry points the slice links from ``.o``
    files, so ``stitch_elf`` must NOT prefix-rename them. Reading them off the
    text rather than reconstructing them per method keeps a drain-vs-fused-cast
    (or a future method's) symbol difference from silently breaking the link.
    """
    return {
        "@" + m for m in re.findall(r"func\.func private @([A-Za-z0-9_]+)", ir_text)
    }


def fused_tail_arg_layout(seq_len, emb_dim, ffn_dim, up_spec, down_spec):
    """The stitched tail's signature and the index of each named buffer.

    Returns ``(args, idx)``. The f32 scratch entries exist only for a GEMM
    method that stages one, which is why this is computed rather than written
    out — the same bug class ``ffn_arg_layout`` exists to kill (and whose
    entries this layout embeds at the FFN slice's positions).
    """
    from shared.infra.stitching import FuncArg

    args, idx = [], {}

    def add(name, type_str):
        idx[name] = len(args)
        args.append(FuncArg(f"%arg{len(args)}", type_str))

    act = f"memref<{seq_len}x{emb_dim}xbf16>"
    wide = f"memref<{seq_len}x{ffn_dim}xbf16>"
    # ln1 is the norm_tail PIPELINE (J7a), not three staged launches. Its two
    # operands arrive in one plane-major buffer: plane 0 is entry 2's output,
    # written contiguously at offset 0 by a GEMM that needs no change, and
    # plane 1 is the layer input, filled by the host. ln1_sum and ln1_norm are
    # gone -- they were the L3 round trips the pipeline exists to delete.
    add("packed1", f"memref<2x{seq_len}x{emb_dim}xbf16>")
    add("gamma1", f"memref<{emb_dim}xbf16>")  # the weight itself, NOT broadcast
    add("hidden", act)  # LayerNorm(p0 + p1) * gamma1 — FFN in AND ln2 residual
    add("w_up", f"memref<{emb_dim}x{ffn_dim}xbf16>")
    if up_spec["needs_f32_scratch"]:
        add("ffn_up_f32", f"memref<{seq_len}x{ffn_dim}xf32>")
    add("ffn_up", wide)
    add("ffn_gelu", wide)
    add("w_down", f"memref<{ffn_dim}x{emb_dim}xbf16>")
    if down_spec["needs_f32_scratch"]:
        add("ffn_out_f32", f"memref<{seq_len}x{emb_dim}xf32>")
    # ln2 is the pipeline too. Plane 0 is the FFN's output and plane 1 is
    # `hidden`, mirrored there by ln1's own stage_scale so the FFN can still
    # read it at a buffer base.
    #
    # ffn_out and packed2 are TWO func args backed by ONE buffer -- see
    # _TAIL_BUFFER_ALIAS. The aliasing has to happen at the buffer level
    # because it cannot happen at the MLIR level: the FFN slice's output arg is
    # `memref<seq x emb>` and stitch_elf wires args by position with matching
    # types, so mapping it onto a 3-D packed arg fails to parse. At the BO
    # level there is nothing to reconcile -- the FFN writes seq*emb elements
    # from the base, which IS plane 0.
    add("ffn_out", act)
    add("packed2", f"memref<2x{seq_len}x{emb_dim}xbf16>")
    add("gamma2", f"memref<{emb_dim}xbf16>")
    add("output", act)
    return args, idx


def build_fused_tail_module(cfg):
    """Stitch ln1 + FFN + ln2 into one multi-launch module over whole tensors.

    Seven slices in execution order — add, LayerNorm, gamma-mul, the whole
    staged FFN (itself three launches plus the fused-cast tail), add,
    LayerNorm, gamma-mul. Each slice is a builder's parsed-and-reprinted
    module, so its affine aliases and SSA names are canonical before
    ``stitch_elf`` renames them per prefix; the FFN slice's own stitch is
    transparent here because the whole func body is spliced.
    """
    seq_len, emb_dim, ffn_dim = cfg["seq_len"], cfg["emb_dim"], cfg["ffn_dim"]
    up_spec, down_spec = cfg["ffn_up_spec"], cfg["ffn_down_spec"]

    args, idx = fused_tail_arg_layout(seq_len, emb_dim, ffn_dim, up_spec, down_spec)
    _, ffn_idx = ffn_arg_layout(seq_len, emb_dim, ffn_dim, up_spec, down_spec)

    # One text per builder, reused for both normalization points; the per-slice
    # prefix is what keeps the two copies' symbols and channels distinct.
    add_ir = str(build_elementwise_add_module(seq_len, emb_dim, bfloat16))
    ln_ir = str(build_layer_norm_module(seq_len, emb_dim, bfloat16))
    mul_ir = str(build_elementwise_mul_module(seq_len, emb_dim, bfloat16))
    ffn_ir = str(build_ffn_module(seq_len, emb_dim, ffn_dim))

    # The FFN slice's wiring: its own layout's positions onto this signature.
    ffn_pairs = [
        ("x", "hidden"),
        ("w_up", "w_up"),
        ("h", "ffn_up"),
        ("a", "ffn_gelu"),
        ("w_down", "w_down"),
        ("y", "ffn_out"),
    ]
    if "h_f32" in ffn_idx:
        ffn_pairs.append(("h_f32", "ffn_up_f32"))
    if "y_f32" in ffn_idx:
        ffn_pairs.append(("y_f32", "ffn_out_f32"))
    ffn_map = {ffn_idx[src]: idx[dst] for src, dst in ffn_pairs}

    def norm_slices(tag, x_name, r_name, gamma, s_name, n_name, o_name):
        return [
            KernelSlice(
                add_ir,
                f"{tag}a",
                {0: idx[x_name], 1: idx[r_name], 2: idx[s_name]},
                extern_syms=_private_syms(add_ir),
            ),
            KernelSlice(
                ln_ir,
                f"{tag}n",
                {0: idx[s_name], 1: idx[n_name]},
                extern_syms=_private_syms(ln_ir),
            ),
            KernelSlice(
                mul_ir,
                f"{tag}m",
                {0: idx[n_name], 1: idx[gamma], 2: idx[o_name]},
                extern_syms=_private_syms(mul_ir),
            ),
        ]

    # ln1: ONE slice, three herds joined L1->L1 inside its own segment. Its
    # per-column shim budget (packed fetch + gamma) is scoped to that segment,
    # so it does not contend with the FFN's.
    nt1_ir = str(
        build_norm_tail_module(seq_len, emb_dim, plane_major=True, mirror_out=True)
    )
    nt2_ir = str(build_norm_tail_module(seq_len, emb_dim, plane_major=True))
    slices = [
        KernelSlice(
            nt1_ir,
            "l1",
            {
                0: idx["packed1"],
                1: idx["gamma1"],
                2: idx["hidden"],
                3: idx["packed2"],
            },
            extern_syms=_private_syms(nt1_ir),
        ),
        KernelSlice(ffn_ir, "ff", ffn_map, extern_syms=_private_syms(ffn_ir)),
        KernelSlice(
            nt2_ir,
            "l2",
            {0: idx["packed2"], 1: idx["gamma2"], 2: idx["output"]},
            extern_syms=_private_syms(nt2_ir),
        ),
    ]
    module = stitch_elf(
        _TAIL_FUNC,
        args,
        slices,
        debug_dump_path="/tmp/debug_fused_tail.mlir",
    )
    print(f"  fused_tail module: {len(str(module).splitlines())} lines, parsed OK")
    return module


def decoder_pre_norm_arg_layout(seq_len, emb_dim):
    """The pre-attention norm's signature: LN(x) * gamma, no residual.

    ``ln1_norm`` is the LayerNorm's own output, a real func arg because the
    two slices hand off through it; ``gamma1b`` is the [seq, emb] broadcast
    the decomposed mul launch takes (the norm_tail PIPELINE takes the [emb]
    weight, but the decoder deliberately uses the decomposed slices -- no
    packed planes, no rows*cols cap).
    """
    from shared.infra.stitching import FuncArg

    args, idx = [], {}

    def add(name, type_str):
        idx[name] = len(args)
        args.append(FuncArg(f"%arg{len(args)}", type_str))

    act = f"memref<{seq_len}x{emb_dim}xbf16>"
    add("x", act)
    add("ln1_norm", act)
    add("gamma1b", act)
    add("ln_in", act)
    return args, idx


def build_decoder_pre_norm_module(cfg):
    """Stitch LayerNorm + gamma-mul into one module: ``ln_in = LN(x) * g1``."""
    seq_len, emb_dim = cfg["seq_len"], cfg["emb_dim"]
    args, idx = decoder_pre_norm_arg_layout(seq_len, emb_dim)
    ln_ir = str(build_layer_norm_module(seq_len, emb_dim, bfloat16))
    mul_ir = str(build_elementwise_mul_module(seq_len, emb_dim, bfloat16))
    slices = [
        KernelSlice(
            ln_ir,
            "pn",
            {0: idx["x"], 1: idx["ln1_norm"]},
            extern_syms=_private_syms(ln_ir),
        ),
        KernelSlice(
            mul_ir,
            "pm",
            {0: idx["ln1_norm"], 1: idx["gamma1b"], 2: idx["ln_in"]},
            extern_syms=_private_syms(mul_ir),
        ),
    ]
    module = stitch_elf(_PRE_NORM_FUNC, args, slices)
    print(f"  pre_norm module: {len(str(module).splitlines())} lines, parsed OK")
    return module


def decoder_tail_arg_layout(seq_len, emb_dim, ffn_dim, up_spec, down_spec):
    """The decoder tail's signature. No packed planes, no aliases.

    add(attn_out, x) -> residual; LN(residual) -> ln2_norm; ln2_norm * g2 ->
    ffn_in; FFN(ffn_in) -> ffn_out; add(ffn_out, residual) -> output. The
    f32 scratch entries exist only for a GEMM method that stages one, same
    as the encoder tail's layout.
    """
    from shared.infra.stitching import FuncArg

    args, idx = [], {}

    def add(name, type_str):
        idx[name] = len(args)
        args.append(FuncArg(f"%arg{len(args)}", type_str))

    act = f"memref<{seq_len}x{emb_dim}xbf16>"
    wide = f"memref<{seq_len}x{ffn_dim}xbf16>"
    add("attn_out", act)
    add("x", act)
    add("residual", act)
    add("ln2_norm", act)
    add("gamma2b", act)
    add("ffn_in", act)
    add("w_up", f"memref<{emb_dim}x{ffn_dim}xbf16>")
    if up_spec["needs_f32_scratch"]:
        add("ffn_up_f32", f"memref<{seq_len}x{ffn_dim}xf32>")
    add("ffn_up", wide)
    add("ffn_gelu", wide)
    add("w_down", f"memref<{ffn_dim}x{emb_dim}xbf16>")
    if down_spec["needs_f32_scratch"]:
        add("ffn_out_f32", f"memref<{seq_len}x{emb_dim}xf32>")
    add("ffn_out", act)
    add("output", act)
    return args, idx


def build_decoder_tail_module(cfg):
    """Stitch the decoder's raw-residual tail into one multi-launch module.

    Five slices in execution order -- add, LayerNorm, gamma-mul, the whole
    staged FFN, add -- each a builder's parsed-and-reprinted module, exactly
    the mechanism the encoder tail uses, minus the packed-plane pipeline the
    pre-norm structure cannot express (and therefore minus its rows*cols
    cap; the decoder rides the encoder's seq bound anyway, via the profile
    skip rule, until someone measures past it).
    """
    seq_len, emb_dim, ffn_dim = cfg["seq_len"], cfg["emb_dim"], cfg["ffn_dim"]
    up_spec, down_spec = cfg["ffn_up_spec"], cfg["ffn_down_spec"]
    args, idx = decoder_tail_arg_layout(seq_len, emb_dim, ffn_dim, up_spec, down_spec)
    _, ffn_idx = ffn_arg_layout(seq_len, emb_dim, ffn_dim, up_spec, down_spec)

    add_ir = str(build_elementwise_add_module(seq_len, emb_dim, bfloat16))
    ln_ir = str(build_layer_norm_module(seq_len, emb_dim, bfloat16))
    mul_ir = str(build_elementwise_mul_module(seq_len, emb_dim, bfloat16))
    ffn_ir = str(build_ffn_module(seq_len, emb_dim, ffn_dim))

    ffn_pairs = [
        ("x", "ffn_in"),
        ("w_up", "w_up"),
        ("h", "ffn_up"),
        ("a", "ffn_gelu"),
        ("w_down", "w_down"),
        ("y", "ffn_out"),
    ]
    if "h_f32" in ffn_idx:
        ffn_pairs.append(("h_f32", "ffn_up_f32"))
    if "y_f32" in ffn_idx:
        ffn_pairs.append(("y_f32", "ffn_out_f32"))
    ffn_map = {ffn_idx[src]: idx[dst] for src, dst in ffn_pairs}

    slices = [
        KernelSlice(
            add_ir,
            "r1",
            {0: idx["attn_out"], 1: idx["x"], 2: idx["residual"]},
            extern_syms=_private_syms(add_ir),
        ),
        KernelSlice(
            ln_ir,
            "dn",
            {0: idx["residual"], 1: idx["ln2_norm"]},
            extern_syms=_private_syms(ln_ir),
        ),
        KernelSlice(
            mul_ir,
            "dm",
            {0: idx["ln2_norm"], 1: idx["gamma2b"], 2: idx["ffn_in"]},
            extern_syms=_private_syms(mul_ir),
        ),
        KernelSlice(ffn_ir, "ff", ffn_map, extern_syms=_private_syms(ffn_ir)),
        KernelSlice(
            add_ir,
            "r2",
            {0: idx["ffn_out"], 1: idx["residual"], 2: idx["output"]},
            extern_syms=_private_syms(add_ir),
        ),
    ]
    module = stitch_elf(
        _DECODER_TAIL_FUNC,
        args,
        slices,
        debug_dump_path="/tmp/debug_decoder_tail.mlir",
    )
    print(
        f"  decoder_tail module: {len(str(module).splitlines())} lines, parsed OK"
    )
    return module


def fused_config(seq_len, emb_dim, ffn_dim, num_heads, head_dim, causal=False):
    """Resolve every operator's configuration without building anything.

    Delegates the spec resolution to ``block_config`` — same registry rows,
    same attention configuration, same arg layouts as the code D2 validated —
    and replaces the artifact plan with this mode's three ELFs.
    """
    base = block_config(seq_len, emb_dim, ffn_dim, num_heads, head_dim, causal=causal)
    suffix = f"{seq_len}x{emb_dim}"
    if causal:
        artifacts = {
            "pre_norm": f"fused_pre_norm_{suffix}",
            "qkv_proj": f"fused_qkv_proj_{suffix}",
            "mha_out_proj": f"fused_mha_out_proj_{suffix}x{num_heads}h_causal",
            "decoder_tail": f"fused_decoder_tail_{suffix}x{ffn_dim}",
        }
    else:
        artifacts = {
            "qkv_proj": f"fused_qkv_proj_{suffix}",
            "mha_out_proj": f"fused_mha_out_proj_{suffix}x{num_heads}h",
            "fused_tail": f"fused_tail_{suffix}x{ffn_dim}",
        }
    cfg = {
        k: v
        for k, v in base.items()
        if k not in ("artifacts", "backend_kwargs", "norm_rows", "norm_blocks")
    }
    if causal:
        tail_args, tail_idx = decoder_tail_arg_layout(
            seq_len, emb_dim, ffn_dim, cfg["ffn_up_spec"], cfg["ffn_down_spec"]
        )
        _, pre_idx = decoder_pre_norm_arg_layout(seq_len, emb_dim)
        cfg.update(
            {
                "artifacts": artifacts,
                "tail_idx": tail_idx,
                # No aliases in the decoder layout: every name is its own BO.
                "tail_order": tuple(
                    name for name, _ in sorted(tail_idx.items(), key=lambda kv: kv[1])
                ),
                "pre_idx": pre_idx,
                "backend_kwargs": {
                    artifacts["pre_norm"]: dict(
                        _GEMM_BACKEND, instance_name=_PRE_NORM_FUNC
                    ),
                    artifacts["qkv_proj"]: dict(
                        _GEMM_BACKEND, instance_name="qkv_proj"
                    ),
                    artifacts["mha_out_proj"]: dict(
                        MHA_OUT_PROJ_RUNNER_KWARGS, instance_name="mha_out_proj"
                    ),
                    artifacts["decoder_tail"]: dict(
                        _GEMM_BACKEND, instance_name=_DECODER_TAIL_FUNC
                    ),
                },
            }
        )
        # Conditional, never defaulted: block_artifact_fingerprint hashes the
        # whole cfg, and a new key on the encoder path would invalidate every
        # cached encoder ELF (same argument as builders/block.py).
        cfg["causal"] = True
        return cfg
    tail_args, tail_idx = fused_tail_arg_layout(
        seq_len, emb_dim, ffn_dim, cfg["ffn_up_spec"], cfg["ffn_down_spec"]
    )
    cfg.update(
        {
            "artifacts": artifacts,
            "tail_idx": tail_idx,
            "tail_order": tuple(
                _TAIL_BUFFER_ALIAS.get(name, name)
                for name, _ in sorted(tail_idx.items(), key=lambda kv: kv[1])
            ),
            "backend_kwargs": {
                artifacts["qkv_proj"]: dict(_GEMM_BACKEND, instance_name="qkv_proj"),
                artifacts["mha_out_proj"]: dict(
                    MHA_OUT_PROJ_RUNNER_KWARGS, instance_name="mha_out_proj"
                ),
                artifacts["fused_tail"]: dict(_GEMM_BACKEND, instance_name=_TAIL_FUNC),
            },
        }
    )
    return cfg


def describe_fused(cfg):
    """One line per resolved decision, for the run log and the lit gate."""
    n_tail = len(cfg["tail_order"])
    if cfg.get("causal"):
        variant, entries = "decoder_gpt2, causal", 4
    else:
        variant, entries = "encoder_bert, non-causal", 3
    print(
        f"  fused {cfg['seq_len']}x{cfg['emb_dim']} ffn {cfg['ffn_dim']} "
        f"{cfg['num_heads']}h x {cfg['head_dim']} ({variant}, "
        f"{entries} entries over {entries} ELFs in one submission)"
    )
    print(
        f"    qkv_proj  {cfg['qkv_spec']['method']} ({cfg['qkv_source']}), "
        f"ffn up {cfg['ffn_up_spec']['method']} / down "
        f"{cfg['ffn_down_spec']['method']} ({cfg['ffn_source']}), "
        f"o_proj {cfg['o_proj_spec']['method']} ({cfg['o_proj_source']})"
    )
    print(
        f"    fused_tail: add+ln+mul / ffn / add+ln+mul stitched over "
        f"{n_tail} whole-tensor args (streamed norms, no row banding)"
    )


#: Per-artifact (external objects, module builder), mirroring
#: ``builders/block.py::_ARTIFACT_BUILD``. The attention object is force-rebuilt
#: with THIS design's tile flags by ``compile_mha_out_proj_kernels``; a stale
#: generic one does not fail to link, it hangs.
_ARTIFACT_BUILD = {
    "qkv_proj": (
        lambda cfg: ek.compile_gemm_mm(
            tile_m=cfg["qkv_spec"]["tile_m"],
            tile_n=cfg["qkv_spec"]["tile_n"],
            tile_k_l1=cfg["qkv_spec"]["tile_k_l1"],
            sym_suffix=cfg["qkv_spec"]["sym_suffix"],
            out_name=cfg["qkv_spec"]["obj"],
        ),
        lambda cfg: build_qkv_proj_module(cfg["seq_len"], cfg["emb_dim"]),
    ),
    "mha_out_proj": (
        lambda cfg: compile_mha_out_proj_kernels(cfg["attn_cfg"], cfg["o_proj_spec"]),
        lambda cfg: build_mha_out_proj_module(
            cfg["seq_len"],
            cfg["head_dim"],
            cfg["num_heads"],
            causal=cfg.get("causal", False),
        ),
    ),
    "fused_tail": (
        lambda cfg: _compile_tail_kernels(cfg),
        build_fused_tail_module,
    ),
    # Decoder only (`fused_config(causal=True)` swaps the artifact set): the
    # pre-attention norm and the raw-residual tail. The tail's externs are
    # the SAME set _compile_tail_kernels builds (FFN micro-kernels + GeLU +
    # multi-row LayerNorm); the pre-norm links the LayerNorm kernel alone.
    "pre_norm": (
        lambda cfg: ek.compile_layer_norm(),
        build_decoder_pre_norm_module,
    ),
    "decoder_tail": (
        lambda cfg: _compile_tail_kernels(cfg),
        build_decoder_tail_module,
    ),
}


def _compile_tail_kernels(cfg):
    """External objects the stitched tail links: both FFN micro-kernels,
    encoder.cc's FFN half (the GeLU), and the multi-row LayerNorm kernel. The
    elementwise add/mul launches are direct vector codegen — nothing to build.
    """
    for spec in (cfg["ffn_up_spec"], cfg["ffn_down_spec"]):
        ek.compile_gemm_mm(
            tile_m=spec["tile_m"],
            tile_n=spec["tile_n"],
            tile_k_l1=spec["tile_k_l1"],
            sym_suffix=spec["sym_suffix"],
            out_name=spec["obj"],
        )
    ek.compile_encoder(build_ffn=True, build_addnorm=False, out_name="encoder_ffn.o")
    ek.compile_layer_norm()


def compile_fused_artifacts(cache, cfg, run_only=False):
    """Compile the three ELFs into ``cache``, reusing only exact matches.

    Same shape as ``compile_block_artifacts`` and reusing its fingerprint
    machinery verbatim: every module is built and hashed on every call, a
    cached ELF is reused only when its recorded fingerprint matches, and only
    a miss rebuilds — including that artifact's external objects, whose names
    carry their tile shapes so variants cannot overwrite each other.
    """
    names = cfg["artifacts"]
    have_manifest = bool(run_only and cache.load_manifest())
    recorded = load_fingerprints(cache) if have_manifest else {}

    fingerprints = {}
    reused = []
    modules = {}
    stale = []
    for key, (_, build_module) in _ARTIFACT_BUILD.items():
        # The build table is the union of both variants; the cfg's artifact
        # set names which subset applies (same guard as the block's).
        if key not in names:
            continue
        name = names[key]
        module = build_module(cfg)
        fingerprints[name] = block_artifact_fingerprint(cfg, key, module)
        if name in cache.artifacts and recorded.get(name) == fingerprints[name]:
            reused.append(name)
            continue
        modules[key] = module
        stale.append(key)

    for key in stale:
        print(f"== external objects for {names[key]} ==")
        _ARTIFACT_BUILD[key][0](cfg)
    for key in stale:
        name = names[key]
        print(f"== compiling {name} ==")
        cache.compile_and_cache(name, modules[key], cfg["backend_kwargs"][name])
    if reused:
        print(f"  reusing {len(reused)} cached fused artifacts: {', '.join(reused)}")
    cache._save_manifest()
    save_fingerprints(cache, fingerprints)


def _spec_buf(name, array, static=False, host_output=False):
    """One ``BufferSpec``, sized and keyed from the array that backs it."""
    return BufferSpec(
        name=name,
        nbytes=array.size * array.itemsize,
        static=static,
        host_output=host_output,
        content_key=content_key(array) if static else None,
    )


def _ordered_args(idx, rename):
    """A step's argument names in signature order, from a layout's index map.

    Same helper as the block's: reading the order back out of the layout is
    what keeps this correct when a registry method change adds or drops an
    f32 scratch.
    """
    return tuple(rename[n] for n, _ in sorted(idx.items(), key=lambda kv: kv[1]))


def prepare_fused(shape, seed=42):
    """The ``fused`` mode's ``SPECS`` preparer: the D2 layer, one submission.

    Same golden model, same per-boundary comparisons at ``BLOCK_STAGE_ATOL``,
    same measured injection target (``ln1_weight`` — here it feeds the first
    gamma multiply inside the stitched tail, scaling one column of ``hidden``
    and cascading through both residual paths exactly as in the block). What
    differs is the execution boundary: three ELFs, one runlist, no
    intermediate host sync.
    """
    seq_len, emb_dim = shape["seq_len"], shape["emb_dim"]
    ffn_dim, num_heads = shape["ffn_dim"], shape["num_heads"]
    head_dim = shape["head_dim"]
    # The variant rides in the shape dict (run_mode._shape_for stamps it).
    # The decoder is FOUR entries in the one submission -- pre_norm, qkv,
    # causal mha, decoder_tail -- with NO packed plane anywhere; see the
    # decoder builders above.
    variant = shape.get("workload_variant", "encoder_bert")
    causal = variant == "decoder_gpt2"
    boundary_names = DECODER_BOUNDARIES if causal else ENCODER_BOUNDARIES
    stage_atol = DECODER_STAGE_ATOL if causal else BLOCK_STAGE_ATOL

    cfg = fused_config(seq_len, emb_dim, ffn_dim, num_heads, head_dim, causal=causal)
    describe_fused(cfg)

    golden = generate_golden_reference(
        seq_len, emb_dim, ffn_dim, num_heads, seed=seed, workload_variant=variant
    )
    weights = golden["weights"]
    reference = golden["boundaries"]
    # The order is FUSED_INPUT_NAMES; `inject` below indexes into it.
    inputs = [
        golden["input"],
        fuse_qkv_weight(weights),
        weights["attn_output_weight"],
        weights["ln1_weight"],
        weights["ffn_up_weight"],
        weights["ffn_down_weight"],
        weights["ln2_weight"],
    ]

    from shared.infra.cache import KernelCache, Profiler

    cache = KernelCache(
        cache_dir=FUSED_CACHE_DIR, verbose=False, profiler=Profiler(enabled=False)
    )
    compile_fused_artifacts(cache, cfg, run_only=True)

    names = cfg["artifacts"]
    mha_idx = cfg["mha_idx"]
    tail_idx, tail_order = cfg["tail_idx"], cfg["tail_order"]

    def dispatch(device_inputs, stage_stats):
        reconfig_baseline = cache.reconfiguration_counts()
        x, w_qkv, w_o, ln1_weight, w_up, w_down, ln2_weight = device_inputs

        def zeros_act():
            return np.zeros((seq_len, emb_dim), dtype=bfloat16)

        # ln1's pipeline takes gamma as the [emb] weight; ln2's mul launch still
        # takes the [seq, emb] broadcast. Two shapes on purpose, until ln2 moves
        # to the pipeline too.
        gamma1 = round_bf16(np.asarray(ln1_weight, dtype=bfloat16))
        gamma2 = round_bf16(np.asarray(ln2_weight, dtype=bfloat16))
        # Plane 1 is x; plane 0 is written on device by entry 2. x is uploaded
        # twice -- here and as qkv_proj's own operand -- because qkv_proj reads
        # it at a buffer base and cannot take a plane offset. That is a host
        # upload, not a sync boundary.
        packed1 = np.zeros((2, seq_len, emb_dim), dtype=bfloat16)
        packed1[1] = x
        # Both planes are device-written: plane 0 by the FFN, plane 1 by ln1.
        packed2 = np.zeros((2, seq_len, emb_dim), dtype=bfloat16)
        arrays = {
            "x": x,
            "packed1": packed1,
            "packed2": packed2,
            "w_qkv": w_qkv,
            "w_o": w_o,
            "gamma1": gamma1,
            "gamma2": gamma2,
            "w_up": w_up,
            "w_down": w_down,
            "qkv_f32": np.zeros((seq_len, 3 * emb_dim), dtype=np.float32),
            "q": zeros_act(),
            "k": zeros_act(),
            "v": zeros_act(),
            "attn_context": zeros_act(),
            "hidden": zeros_act(),
            "ffn_up": np.zeros((seq_len, ffn_dim), dtype=bfloat16),
            "ffn_gelu": np.zeros((seq_len, ffn_dim), dtype=bfloat16),
            "output": zeros_act(),
        }
        # The attention module's layout names onto this mode's buffer names —
        # identical to the block's sequence A.
        mha_rename = {
            "q": "q",
            "k": "k",
            "v": "v",
            "attn_out": "attn_context",
            "w_o": "w_o",
            "y_f32": "attn_f32",
            "y": "packed1",
        }
        if "y_f32" in mha_idx:
            arrays["attn_f32"] = np.zeros((seq_len, emb_dim), dtype=np.float32)
        if "ffn_up_f32" in tail_idx:
            arrays["ffn_up_f32"] = np.zeros((seq_len, ffn_dim), dtype=np.float32)
        if "ffn_out_f32" in tail_idx:
            arrays["ffn_out_f32"] = np.zeros((seq_len, emb_dim), dtype=np.float32)

        host_filled = {
            "x",
            "packed1",
            "w_qkv",
            "w_o",
            "gamma1",
            "w_up",
            "w_down",
            "gamma2",
        }
        tail_writes = tuple(
            sorted(tail_idx[n] for n in tail_idx if n not in host_filled)
        )
        steps = [
            DispatchStep(
                names["qkv_proj"],
                ("x", "w_qkv", "qkv_f32", "q", "k", "v"),
                writes=(2, 3, 4, 5),
            ),
            DispatchStep(
                names["mha_out_proj"],
                _ordered_args(mha_idx, mha_rename),
                writes=tuple(
                    sorted(
                        mha_idx[n] for n in ("attn_out", "y_f32", "y") if n in mha_idx
                    )
                ),
            ),
            DispatchStep(names["fused_tail"], tail_order, writes=tail_writes),
        ]
        outputs = (
            "q",
            "k",
            "v",
            "attn_context",
            "packed1",
            "hidden",
            "ffn_up",
            "ffn_gelu",
            "packed2",
            "output",
        )
        statics = {"w_qkv", "w_o", "gamma1", "w_up", "w_down", "gamma2"}
        specs = {
            name: _spec_buf(
                name, arr, static=name in statics, host_output=name in outputs
            )
            for name, arr in arrays.items()
        }
        # The f32 scratches are zero-filled by the host, exactly as every D1
        # run handed them to XRTRunner; they appear only at written positions,
        # so they must be named here or the upload is skipped.
        host_writes = (host_filled | {"qkv_f32"}) | (
            {"attn_f32", "ffn_up_f32", "ffn_out_f32"} & set(arrays)
        )
        print("  [fused 1/1] qkv_proj + mha_out_proj + fused_tail (one submission)")
        results, vector = cache.run_sequence(
            steps,
            specs,
            cfg["backend_kwargs"],
            arrays,
            host_writes=host_writes,
            require_single_submission=True,
        )
        boundaries = {n: np.array(results[n], copy=True) for n in outputs}
        # attn_out is no longer its own buffer: it is what entry 2 wrote into
        # plane 0 of packed1. Reshape from the flat readback view, then name it
        # so ENCODER_BOUNDARIES compares the same tensor it always did.
        boundaries["attn_out"] = boundaries.pop("packed1").reshape(2, seq_len, emb_dim)[
            0
        ]
        boundaries["ffn_out"] = boundaries.pop("packed2").reshape(2, seq_len, emb_dim)[
            0
        ]

        vector_rows = [vector.as_row()]
        stages = []
        for name in ENCODER_BOUNDARIES:
            atol = BLOCK_STAGE_ATOL[name]
            stats = stage_stats(boundaries[name], reference[name], atol=atol)
            stages.append(dict(stats, name=name, atol=atol))
            print(
                f"  [stage] {name:13s} {stats['n_elements']:>9d} elements  "
                f"mismatch {stats['n_mismatch']:>7d}  "
                f"mean_rel_L1 {stats['mean_rel_L1']:.3e}  "
                f"atol_required {stats['atol_required']:.3e} (atol {atol:.1e})"
            )
        clean = sum(1 for s in stages if s["n_mismatch"] == 0)
        print(f"[fused] stages: {clean}/{len(stages)} clean")
        # On the fault path too — the FAULT half of the lit recipe pins the
        # printed totals to the same literals as the clean half.
        print_dispatch_totals("fused", vector_rows)
        return [boundaries["output"]], {
            "stages": stages,
            "stages_passed": clean == len(stages),
            "dispatch_vectors": vector_rows,
            # The same decomposition the host-mediated modes report. This mode
            # runs NO host compute, so host_cpu_ms is empty BY CONSTRUCTION --
            # which is the comparison: offload's attention is host time, this
            # mode's is device time, and until now both landed in one number.
            "device_ms": sum(
                float(r.get("device_submission_ms", 0.0)) for r in vector_rows
            ),
            "sync_ms": sum(float(r.get("host_sync_ms", 0.0)) for r in vector_rows),
            "host_cpu_ms": {},
            # What THIS dispatch loaded and attached (schema v2's
            # reconfiguration columns). Three ELFs, loaded once, contexts
            # standing for the process: a warmed dispatch honestly reports 0,
            # which is the packaged mode's whole claim on this axis.
            **reconfiguration_delta(cache, reconfig_baseline),
        }

    def decoder_dispatch(device_inputs, stage_stats):
        # `[2026-08-19]` the re-execution wall this dispatch sat behind is
        # ROOT-CAUSED AND FIXED. The wall: dispatch 1 was 12/12 stages clean
        # (devq 382) while every later dispatch computed UNMASKED attention
        # (corr 0.9994 with the non-causal reference, devq 383) with q/k/v
        # clean, surviving a full hw-context unload/reload (devq 384). The
        # cause was NOT composition-level runlist state (the H10 frozen-BD
        # framing is retracted): the causal mha keeps its Q-block base in an
        # UNINITIALIZED L1 counter behind a boot flag that only fires on
        # zeroed memory, and the base only ever advanced -- so a completed
        # execution left it past every kv block, and on re-execution
        # `apply_causal_mask` never fired. Coarse re-executed causally only
        # because its per-dispatch flow re-zeroes the partition; the context
        # reload did not heal it because the counter is not CDO-initialized
        # state. Fixed in attn_npu2*.py by wrapping the advance modulo the
        # execution's total Q blocks (identity within one execution, boot
        # state restored at its end); measured after the fix: three
        # dispatches of this stitch 12/12 clean each, attn_context corr
        # 0.9995 with the CAUSAL reference on every one, d2 == d3 bytes.
        reconfig_baseline = cache.reconfiguration_counts()
        x, w_qkv, w_o, ln1_weight, w_up, w_down, ln2_weight = device_inputs

        def zeros_act():
            return np.zeros((seq_len, emb_dim), dtype=bfloat16)

        # Both gammas are the [seq, emb] broadcast the decomposed mul slices
        # take (the norm_tail PIPELINE takes the [emb] weight; the decoder
        # deliberately uses the decomposed slices -- no packed planes).
        gamma1b = round_bf16(broadcast_row_weight(ln1_weight, seq_len))
        gamma2b = round_bf16(broadcast_row_weight(ln2_weight, seq_len))
        arrays = {
            "x": x,
            "w_qkv": w_qkv,
            "w_o": w_o,
            "gamma1b": gamma1b,
            "gamma2b": gamma2b,
            "w_up": w_up,
            "w_down": w_down,
            "ln1_norm": zeros_act(),
            "ln_in": zeros_act(),
            "qkv_f32": np.zeros((seq_len, 3 * emb_dim), dtype=np.float32),
            "q": zeros_act(),
            "k": zeros_act(),
            "v": zeros_act(),
            "attn_context": zeros_act(),
            "attn_out": zeros_act(),
            "residual": zeros_act(),
            "ln2_norm": zeros_act(),
            "ffn_in": zeros_act(),
            "ffn_up": np.zeros((seq_len, ffn_dim), dtype=bfloat16),
            "ffn_gelu": np.zeros((seq_len, ffn_dim), dtype=bfloat16),
            "ffn_out": zeros_act(),
            "output": zeros_act(),
        }
        # attn_out is its own buffer here -- the encoder packs it into
        # packed1 plane 0 for the norm_tail pipeline; the decoder's residual
        # add takes two plain 2-D args.
        mha_rename = {
            "q": "q",
            "k": "k",
            "v": "v",
            "attn_out": "attn_context",
            "w_o": "w_o",
            "y_f32": "attn_f32",
            "y": "attn_out",
        }
        if "y_f32" in mha_idx:
            arrays["attn_f32"] = np.zeros((seq_len, emb_dim), dtype=np.float32)
        if "ffn_up_f32" in tail_idx:
            arrays["ffn_up_f32"] = np.zeros((seq_len, ffn_dim), dtype=np.float32)
        if "ffn_out_f32" in tail_idx:
            arrays["ffn_out_f32"] = np.zeros((seq_len, emb_dim), dtype=np.float32)

        host_filled = {
            "x",
            "w_qkv",
            "w_o",
            "gamma1b",
            "gamma2b",
            "w_up",
            "w_down",
        }
        tail_writes = tuple(
            sorted(tail_idx[n] for n in tail_idx if n not in host_filled)
        )
        pre_idx = cfg["pre_idx"]
        pre_order = tuple(
            name for name, _ in sorted(pre_idx.items(), key=lambda kv: kv[1])
        )
        pre_writes = tuple(sorted(pre_idx[n] for n in ("ln1_norm", "ln_in")))
        steps = [
            DispatchStep(names["pre_norm"], pre_order, writes=pre_writes),
            DispatchStep(
                names["qkv_proj"],
                ("ln_in", "w_qkv", "qkv_f32", "q", "k", "v"),
                writes=(2, 3, 4, 5),
            ),
            DispatchStep(
                names["mha_out_proj"],
                _ordered_args(mha_idx, mha_rename),
                writes=tuple(
                    sorted(
                        mha_idx[n] for n in ("attn_out", "y_f32", "y") if n in mha_idx
                    )
                ),
            ),
            DispatchStep(names["decoder_tail"], cfg["tail_order"], writes=tail_writes),
        ]
        outputs = (
            "ln_in",
            "q",
            "k",
            "v",
            "attn_context",
            "attn_out",
            "residual",
            "ffn_in",
            "ffn_up",
            "ffn_gelu",
            "ffn_out",
            "output",
        )
        statics = {"w_qkv", "w_o", "gamma1b", "gamma2b", "w_up", "w_down"}
        specs = {
            name: _spec_buf(
                name, arr, static=name in statics, host_output=name in outputs
            )
            for name, arr in arrays.items()
        }
        host_writes = (host_filled | {"qkv_f32"}) | (
            {"attn_f32", "ffn_up_f32", "ffn_out_f32"} & set(arrays)
        )
        print(
            "  [fused 1/1] pre_norm + qkv_proj + mha_out_proj + decoder_tail "
            "(one submission)"
        )
        results, vector = cache.run_sequence(
            steps,
            specs,
            cfg["backend_kwargs"],
            arrays,
            host_writes=host_writes,
            require_single_submission=True,
        )
        boundaries = {n: np.array(results[n], copy=True) for n in outputs}

        vector_rows = [vector.as_row()]
        stages = []
        for name in boundary_names:
            atol = stage_atol[name]
            stats = stage_stats(boundaries[name], reference[name], atol=atol)
            stages.append(dict(stats, name=name, atol=atol))
            print(
                f"  [stage] {name:13s} {stats['n_elements']:>9d} elements  "
                f"mismatch {stats['n_mismatch']:>7d}  "
                f"mean_rel_L1 {stats['mean_rel_L1']:.3e}  "
                f"atol_required {stats['atol_required']:.3e} (atol {atol:.1e})"
            )
        clean = sum(1 for s_ in stages if s_["n_mismatch"] == 0)
        print(f"[fused] stages: {clean}/{len(stages)} clean")
        print_dispatch_totals("fused", vector_rows)
        return [boundaries["output"]], {
            "stages": stages,
            "stages_passed": clean == len(stages),
            "dispatch_vectors": vector_rows,
            "device_ms": sum(
                float(r.get("device_submission_ms", 0.0)) for r in vector_rows
            ),
            "sync_ms": sum(float(r.get("host_sync_ms", 0.0)) for r in vector_rows),
            "host_cpu_ms": {},
            **reconfiguration_delta(cache, reconfig_baseline),
        }

    if causal:
        dispatch = decoder_dispatch

    record_extra = {
        "variant": variant,
        "causal": causal,
        "golden_seed": seed,
        "execution_mode": EXECUTION_MODE_CSV["fused"],
        "elf_count": len(names),
        "fusion": (
            "stitch_elf: pre_norm / qkv / causal mha / decoder_tail "
            "(add-ln-mul-ffn-add), no packed planes"
            if causal
            else "stitch_elf: norm_tail(J7a) / ffn / norm_tail(J7a) in one module"
        ),
        "gemm_spec_source": cfg["qkv_source"],
        "gemm_spec_qkv": _spec_digest(cfg["qkv_spec"]),
        "gemm_spec_ffn_up": _spec_digest(cfg["ffn_up_spec"]),
        "gemm_spec_ffn_down": _spec_digest(cfg["ffn_down_spec"]),
        "gemm_spec_o_proj": _spec_digest(cfg["o_proj_spec"]),
        "attention_config": {
            key: cfg["attn_cfg"][key]
            for key in (
                "parallel_seq",
                "parallel_heads",
                "kv_seq_tile",
                "q_seq_tile",
                "num_q_tiles",
                "cascade_stages",
                "gqa_group_size",
            )
        },
    }
    prepared = {
        "inputs": inputs,
        # ln1_weight, index 3. Same measured target as the block's; see
        # opcheck_layer.py for the numbers. For the decoder it feeds the
        # pre-attention gamma multiply, upstream of every boundary.
        "inject": (FUSED_INPUT_NAMES.index("ln1_weight"), (0,)),
        "expected": [reference["output"]],
        "dispatch": dispatch,
        "record_extra": record_extra,
    }
    if causal:
        # Same seam as opcheck_layer's: the decoder's whole-layer comparison
        # runs at its own output boundary's atol.
        prepared["atol"] = stage_atol["output"]
    return prepared
