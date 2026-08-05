# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How each transformer-layer operator is built and what data it is given.

CONTRACT
    One ``prepare_<operator>(shape, seed=...)`` per operator. Each returns the
    four things a run needs, plus two optional extras:

        module        the built ``air.ir.Module``
        inputs        every argument the host fills, in signature order
        expected      the FP32-derived expected outputs
        inject        ``(input_index, position)`` for the negative control
        runner_kwargs per-operator backend settings (optional)
        record_extra  provenance merged into the results artifact (optional)

    A multi-artifact operator returns ``dispatch`` instead of ``module``; see
    ``opcheck.py``'s module docstring for what that seam does and does not
    change. ``opcheck_specs.py`` names these functions in its ``SPECS`` entries
    and adds the per-shape ``atol``; nothing here knows what shapes exist.

WHY THIS IS A SEPARATE MODULE FROM ``opcheck_specs.py``
    Porting convention 5 caps a module at ~800 lines, and the two together
    reached 1081 with four more execution modes' specs still to land in the
    catalogue (Phase E2-E5). The seam is the one D1 named and the one this
    file's own neighbours already draw -- ``opcheck.py`` / ``opcheck_specs.py``,
    ``registry_sweep.py`` / ``sweep_families.py``: MECHANISM in one module,
    CATALOGUE in another.

    So the split is three ways, not two, and each module knows one thing:

        opcheck.py          what counts as evidence (the recording runner, the
                            injection, the results artifact, the CLI)
        opcheck_prepare.py  how each operator is built and fed  <- this file
        opcheck_specs.py    which (operator, shape) the port claims, and at what
                            tolerance

    Adding a shape of an existing operator touches only the catalogue. Adding an
    operator touches this file and the catalogue. Changing what counts as
    evidence touches only ``opcheck.py``.

    THE REFERENCE IS ALWAYS COMPUTED HERE, from the CLEAN inputs, before any
    injection. ``opcheck.py`` perturbs afterwards. That ordering is the whole
    negative control: perturbing the reference instead would satisfy the letter
    of the check and destroy its purpose.

FOOTGUNS
    - The perturbed element is picked strictly BELOW the diagonal for
      ``causal_mask``. Above it the reference is ``-10000``, and
      ``rtol * 10000 = 160`` swallows any perturbation worth making -- the
      negative control would silently pass and prove nothing.
    - ``mha_out_proj`` injects into ``w_o``, not into Q/K/V, and ``block`` into
      ``ln1_weight``, for the same class of reason. Both section headers below
      carry the measurements that chose them.
    - External kernel objects are written to the CURRENT WORKING DIRECTORY,
      because that is where aiecc's ``link_with`` search looks. Every
      ``prepare`` here builds its objects as a side effect. Run from a scratch
      directory.
    - encoder.cc is built TWICE, to two objects, each with one half:
      ``encoder.o`` (addnorm half, for ``addnorm``) and ``encoder_ffn.o`` (FFN
      half, for ``ffn``). Building both halves into one object collides with
      ``addnorm_ffn.o`` on ``ffn_gelu_bf16`` and
      ``ffn_eltwise_add_bf16_vector``, and neither operator needs the other's
      half. ``compile_kernels.py`` checks that the split actually happened.
    - The GEMM-backed operators need extra backend settings that the C1
      operators do not -- BD-ID recycling and ELF output -- and
      ``mha_out_proj`` needs FlashAttention's instead, which are NOT the same.
      They are declared per spec, so an operator that forgets them fails to
      place rather than quietly inheriting the wrong ones.
"""

import os
import sys

import numpy as np
from ml_dtypes import bfloat16

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_HERE)  # programming_examples/
for _p in (_PROJ_ROOT, os.path.join(_PROJ_ROOT, "llms"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import shared.infra.external_kernels as ek  # noqa: E402
from builders.addnorm import (  # noqa: E402
    addnorm_pre_add_reference,
    addnorm_reference,
    build_addnorm_module,
    compile_addnorm_kernel,
)
from builders.elementwise_add import (  # noqa: E402
    build_elementwise_add_module,
    causal_mask_bias,
    elementwise_add_reference,
)
from builders.ffn import (  # noqa: E402
    build_ffn_module,
    ffn_device_inputs,
    ffn_gemm_specs,
    ffn_reference,
)
from builders.layer_norm import (  # noqa: E402
    build_layer_norm_module,
    layer_norm_reference,
)
from builders.mha_out_proj import (  # noqa: E402
    MHA_OUT_PROJ_RUNNER_KWARGS,
    build_mha_out_proj_module,
    compile_mha_out_proj_kernels,
    mha_out_proj_arg_layout,
    mha_out_proj_config,
    mha_out_proj_device_inputs,
    mha_out_proj_reference,
)
from builders.block import (  # noqa: E402
    BLOCK_BOUNDARIES,
    BLOCK_INPUT_NAMES,
    block_config,
    compile_block_artifacts,
    describe_block,
    run_block,
)
from builders.qkv_proj import (  # noqa: E402
    build_qkv_proj_module,
    qkv_gemm_spec,
    qkv_proj_reference,
)
from pattern.reference import (  # noqa: E402
    fuse_qkv_weight,
    generate_golden_reference,
)

# ---------------------------------------------------------------------------
# Per-operator preparation
#
# Each returns the four things a run needs: the built module, the device inputs,
# the FP32-derived expected outputs, and where to inject a fault. The reference
# is ALWAYS computed here, from the clean inputs, before any injection.
# ---------------------------------------------------------------------------


def prepare_elementwise_add(shape, seed=0):
    rows, cols = shape["rows"], shape["cols"]
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((rows, cols)).astype(bfloat16)
    b = rng.standard_normal((rows, cols)).astype(bfloat16)
    return {
        "module": build_elementwise_add_module(rows, cols, bfloat16),
        "inputs": [a, b],
        "expected": [elementwise_add_reference(a, b)],
        "inject": (0, (rows - 1, 0)),
    }


def prepare_causal_mask(shape, seed=1):
    seq = shape["rows"]
    rng = np.random.default_rng(seed)
    scores = rng.standard_normal((seq, seq)).astype(bfloat16)
    mask = causal_mask_bias(seq, bfloat16)
    return {
        # causal_mask=True changes no MLIR; it asserts the square shape and
        # records that `b` is the static mask rather than a residual.
        "module": build_elementwise_add_module(seq, seq, bfloat16, causal_mask=True),
        "inputs": [scores, mask],
        "expected": [elementwise_add_reference(scores, mask)],
        # Strictly below the diagonal: see the module footgun about rtol
        # swallowing a perturbation of a -10000 masked element.
        "inject": (0, (seq - 1, 0)),
    }


def prepare_layer_norm(shape, seed=2):
    rows, cols = shape["rows"], shape["cols"]
    ek.compile_layer_norm()
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((rows, cols)).astype(bfloat16)
    return {
        "module": build_layer_norm_module(rows, cols, bfloat16),
        "inputs": [x],
        "expected": [layer_norm_reference(x)],
        "inject": (0, (rows - 1, 0)),
    }


def prepare_addnorm(shape, seed=3):
    """Both residual orderings. ``shape["pre_add"]`` picks which.

    The variant selects the kernel OBJECT, the entry point and the reference
    together -- see ``builders/addnorm.py``. Taking the reference from the same
    flag that built the module is what stops the two from drifting apart into a
    check that passes against the wrong function.
    """
    rows, cols = shape["rows"], shape["cols"]
    pre_add = shape.get("pre_add", False)
    # addnorm half only -- the FFN half collides on ffn_gelu_bf16.
    compile_addnorm_kernel(pre_add=pre_add)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((rows, cols)).astype(bfloat16)
    residual = rng.standard_normal((rows, cols)).astype(bfloat16)
    # A trained LayerNorm gamma sits near 1; uniform(0.5, 1.5) is that range
    # without being exactly 1, which would hide a dropped weight multiply.
    weight = rng.uniform(0.5, 1.5, size=cols).astype(bfloat16)
    reference = addnorm_pre_add_reference if pre_add else addnorm_reference
    return {
        "module": build_addnorm_module(rows, cols, bfloat16, pre_add=pre_add),
        "inputs": [x, residual, weight],
        "expected": [reference(x, residual, weight)],
        "inject": (0, (rows - 1, 0)),
    }


# ---------------------------------------------------------------------------
# GEMM-backed operators (C2)
#
# HOW THEIR DATA IS SCALED, AND WHY IT IS NOT A FREE CHOICE
#     The external GEMM's error is dominated by its bfp16 MMUL emulation, not by
#     the bf16 output rounding: the registry records mean_rel_L1 = 9.3e-3 for it
#     and these runs measure 9.7e-3, i.e. roughly 1% of the OUTPUT's own
#     magnitude. So the absolute error a run reports is proportional to how big
#     the output is, and quoting an `atol` only means something alongside the
#     scale it was measured at.
#
#     The scale used here is the one the registry's own GEMM sweep uses
#     (matrix_multiplication/bf16_in_bf16_out/run.py): operands ~ N(0, 1/sqrt(K))
#     for a reduction of depth K, which puts the product at 1/sqrt(K). Every
#     `atol` below is then the MEASURED worst-case absolute error at that scale
#     rounded up, ~2-3x, exactly as the registry's GEMM rows are sized -- and it
#     makes the mean_rel_L1 recorded here directly comparable with them.
#
#     The one departure is the FFN's up-projection, which is scaled to put its
#     output at unit variance instead. GeLU is only interesting on |x| ~ 1-3;
#     at 1/sqrt(K) the whole tensor would sit inside +/-0.15 where the
#     activation is indistinguishable from 0.5x, and the operator's own stage
#     would go untested.
#
# WHY THEIR RUNS NEED EXTRA BACKEND SETTINGS
#     `runtime_loop_tiling_sizes=[2,2]` for BD-ID recycling, and ELF output for
#     multi-segment designs. Both are placement/packaging concerns; neither
#     touches the comparison. See _GEMM_RUNNER_KWARGS.
# ---------------------------------------------------------------------------

# Backend settings both GEMM-backed operators need.
#   runtime_loop_tiling_sizes: BD-ID recycling. A fused-cast GEMM herd runs out
#     of buffer descriptors without it, which surfaces as a placement failure.
#   output_format "elf": these designs are MULTI-SEGMENT -- one aie.device per
#     air.launch, driven by an aiex.configure/aiex.run runtime sequence. The
#     xclbin path names a single instruction blob on the aircc command line, so
#     a second segment collides on it and aiecc stops with "produced duplicate
#     output path". ELF is the format the shipped multi-launch llama builders
#     use for exactly this reason, and it is not a relaxation of anything: the
#     comparison downstream is identical.
_GEMM_RUNNER_KWARGS = {
    "runtime_loop_tiling_sizes": [2, 2],
    "output_format": "elf",
}


def _registry_gemm_scale(k):
    """Operand scale the registry's GEMM sweep uses for a depth-``k`` reduction.

    Both operands at ``1/sqrt(k)`` put the product at ``1/sqrt(k)`` too. Copied
    from ``matrix_multiplication/bf16_in_bf16_out/run.py`` so the error figures
    recorded here sit on the same axis as the registry's GEMM rows.
    """
    return 1.0 / np.sqrt(k)


def _unit_output_scale(k):
    """Operand scale that puts a depth-``k`` product at unit variance.

    Both operands at ``k^-0.25``: the reduction multiplies the two scales ``k``
    times. Used only where a downstream stage needs activation-sized inputs --
    the FFN's GeLU.
    """
    return k**-0.25


# E[gelu(x)^2] for x ~ N(0, 1). Used to size the down-projection weight so its
# output lands where the registry's GEMM sweep would put a depth-`ffn_dim`
# reduction, rather than sqrt(1/0.42) ~ 1.5x above it.
_GELU_SECOND_MOMENT = 0.42


def prepare_qkv_proj(shape, seed=4):
    seq_len, emb_dim = shape["seq_len"], shape["emb_dim"]
    n_total = 3 * emb_dim
    spec, source = qkv_gemm_spec(seq_len, emb_dim)
    ek.compile_gemm_mm(
        tile_m=spec["tile_m"],
        tile_n=spec["tile_n"],
        tile_k_l1=spec["tile_k_l1"],
        sym_suffix=spec["sym_suffix"],
        out_name=spec["obj"],
    )
    rng = np.random.default_rng(seed)
    scale = _registry_gemm_scale(emb_dim)
    x = (rng.standard_normal((seq_len, emb_dim)) * scale).astype(bfloat16)
    w = (rng.standard_normal((emb_dim, n_total)) * scale).astype(bfloat16)
    return {
        "module": build_qkv_proj_module(seq_len, emb_dim),
        # arg2 is the f32 C scratch: an input slot the device writes, not an
        # output. See builders/qkv_proj.py on why it precedes q/k/v.
        "inputs": [x, w, np.zeros((seq_len, n_total), dtype=np.float32)],
        "expected": list(qkv_proj_reference(x, w)),
        "inject": (0, (seq_len - 1, 0)),
        "runner_kwargs": _GEMM_RUNNER_KWARGS,
        "record_extra": {"gemm_spec_source": source, "gemm_spec": _spec_digest(spec)},
    }


def prepare_ffn(shape, seed=5):
    seq_len, emb_dim = shape["seq_len"], shape["emb_dim"]
    ffn_dim = shape["ffn_dim"]
    up_spec, down_spec, source = ffn_gemm_specs(seq_len, emb_dim, ffn_dim)
    for spec in (up_spec, down_spec):
        ek.compile_gemm_mm(
            tile_m=spec["tile_m"],
            tile_n=spec["tile_n"],
            tile_k_l1=spec["tile_k_l1"],
            sym_suffix=spec["sym_suffix"],
            out_name=spec["obj"],
        )
    # encoder.cc's FFN half only. Building the addnorm half too would collide
    # with addnorm_ffn.o on ffn_gelu_bf16, and this ELF needs neither.
    ek.compile_encoder(build_ffn=True, build_addnorm=False, out_name="encoder_ffn.o")

    rng = np.random.default_rng(seed)
    # Up-projection at unit output variance so GeLU is exercised where it is
    # nonlinear; down-projection weight sized to land y where the registry's
    # sweep puts a depth-ffn_dim reduction. See the section header above.
    up_scale = _unit_output_scale(emb_dim)
    down_scale = _registry_gemm_scale(ffn_dim) / np.sqrt(ffn_dim * _GELU_SECOND_MOMENT)
    x = (rng.standard_normal((seq_len, emb_dim)) * up_scale).astype(bfloat16)
    w_up = (rng.standard_normal((emb_dim, ffn_dim)) * up_scale).astype(bfloat16)
    w_down = (rng.standard_normal((ffn_dim, emb_dim)) * down_scale).astype(bfloat16)
    return {
        "module": build_ffn_module(seq_len, emb_dim, ffn_dim),
        "inputs": ffn_device_inputs(x, w_up, w_down, up_spec, down_spec),
        "expected": [ffn_reference(x, w_up, w_down)],
        "inject": (0, (seq_len - 1, 0)),
        "runner_kwargs": _GEMM_RUNNER_KWARGS,
        "record_extra": {
            "gemm_spec_source": source,
            "gemm_spec_up": _spec_digest(up_spec),
            "gemm_spec_down": _spec_digest(down_spec),
        },
    }


# ---------------------------------------------------------------------------
# Fused attention + output projection (C3)
#
# HOW ITS DATA IS SCALED
#     Q, K and V are N(0, 1) -- the standard the FlashAttention registry rows
#     were measured at, which is PyTorch's own SDPA test distribution. W_o is
#     N(0, 1/sqrt(emb_dim)), the registry GEMM sweep's scale for a depth-emb_dim
#     reduction, so the projection's contribution sits on the same axis as the
#     registry's GEMM rows.
#
# WHY THE FAULT IS INJECTED INTO W_o AND NOT INTO Q, K OR V
#     Softmax normalisation DAMPS a single-element perturbation of Q, K or V.
#     Measured on this reference at 512x512, 8 heads, with the shared
#     FAULT_DELTA of 2.0: perturbing one element of K or V moves the output by
#     at most 4.9e-3, and one element of Q by at most 3.5e-2 -- both inside, or
#     within a factor of two of, an `atol` sized to this datapath's honest
#     error. A negative control that close to the tolerance band proves nothing
#     about whether the check discriminates; it just measures how the two
#     numbers happened to land.
#
#     W_o has no such averaging in front of it: one perturbed weight moves an
#     entire output column by `attn_out[:, c] * FAULT_DELTA`, measured at 3.7e-1
#     over 416 elements (1.5 over 460 in the causal variant), one to two orders
#     of magnitude clear of the band. That is the property the control needs --
#     and it is still a DEVICE input perturbed after the reference was computed
#     from the clean one, which is what makes it a control rather than a
#     rescaling of the reference.
# ---------------------------------------------------------------------------


def prepare_mha_out_proj(shape, seed=6):
    seq_len, head_dim = shape["seq_len"], shape["head_dim"]
    num_heads = shape["num_heads"]
    num_kv_heads = shape.get("num_kv_heads")
    causal = shape["causal"]
    emb_dim = num_heads * head_dim

    attn_cfg, gemm_spec, gemm_source = mha_out_proj_config(
        seq_len,
        head_dim,
        num_heads,
        num_kv_heads=num_kv_heads,
        causal=causal,
    )
    compile_mha_out_proj_kernels(attn_cfg, gemm_spec)

    rng = np.random.default_rng(seed)
    q = rng.standard_normal((seq_len, emb_dim)).astype(bfloat16)
    k = rng.standard_normal((seq_len, attn_cfg["kv_emb_dim"])).astype(bfloat16)
    v = rng.standard_normal((seq_len, attn_cfg["kv_emb_dim"])).astype(bfloat16)
    w_o = (
        rng.standard_normal((emb_dim, emb_dim)) * _registry_gemm_scale(emb_dim)
    ).astype(bfloat16)

    inputs = mha_out_proj_device_inputs(q, k, v, w_o, attn_cfg, gemm_spec)
    _, arg_idx = mha_out_proj_arg_layout(attn_cfg, gemm_spec)
    return {
        "module": build_mha_out_proj_module(
            seq_len,
            head_dim,
            num_heads,
            num_kv_heads=num_kv_heads,
            causal=causal,
        ),
        "inputs": inputs,
        "expected": [
            mha_out_proj_reference(
                q, k, v, w_o, num_heads, num_kv_heads=num_kv_heads, causal=causal
            )
        ],
        # w_o, whose index moves with the arg layout -- ask the layout rather
        # than hard-coding it. See the section header on why it is w_o.
        "inject": (arg_idx["w_o"], (0, 0)),
        "runner_kwargs": dict(MHA_OUT_PROJ_RUNNER_KWARGS),
        "record_extra": {
            "gemm_spec_source": gemm_source,
            "gemm_spec": _spec_digest(gemm_spec),
            "attention_config": {
                key: attn_cfg[key]
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
        },
    }


def _spec_digest(spec):
    """The part of a GEMM spec worth recording: method and the four tiles.

    Written into the results artifact so a spec that came from ``gemm_spec_fn``
    rather than the registry is VISIBLE there. The phase allows that injection
    hook for an unmeasured shape precisely on the condition that the guess is
    not silent.
    """
    return {
        "method": spec["method"],
        "tile_m": spec["tile_m"],
        "tile_k_l2": spec["tile_k_l2"],
        "tile_k_l1": spec["tile_k_l1"],
        "tile_n": spec["tile_n"],
    }


# ---------------------------------------------------------------------------
# The whole encoder_bert layer (D2)
#
# WHY THIS ONE OWNS ITS OWN DISPATCH
#     Four ELFs over four `KernelCache.run_sequence` calls, not one module: see
#     `builders/block.py` on why the layer cannot be a single sequence, and
#     `opcheck.py`'s module docstring on what the `dispatch` seam does and does
#     not change. Everything that makes this a check -- reference from the clean
#     inputs, injection after it, `_RecordingRunner`'s verdict at RTOL and this
#     spec's `atol`, zero permitted mismatches -- is untouched.
#
# THE DATA IS THE GOLDEN MODEL'S, NOT A SCALE CHOSEN HERE
#     Every other operator above scales its operands to where the registry's
#     GEMM sweep measured them, because a standalone operator has no other
#     defensible scale. A whole layer does: `pattern/reference.py` draws exactly
#     what iron's `generate_golden_reference` draws, in the same order, at
#     `val_range = 0.05`. That is a much SMALLER scale than the registry's, and
#     it has a consequence worth knowing before reading the stage errors:
#     attention scores land around 5e-3, so the softmax is nearly uniform, the
#     attention output is an average of V, and `attn_out` comes out around 1e-3
#     against a residual `x` around 5e-2. The attention half therefore
#     contributes a few percent of what the first LayerNorm sees.
#
# WHY THE FAULT GOES INTO ln1_weight, MEASURED AND NOT ASSUMED
#     The consequence above is exactly the trap the phase document warns about,
#     and it is worse than "damped": measured on this golden model at
#     512x768x3072, 12 heads, with the shared FAULT_DELTA of 2.0, perturbing one
#     element of `w_o` or of the fused QKV weight moves the layer output by
#     3.1e-2 and puts ZERO elements outside the tolerance band. A negative
#     control injected there would PASS under injection and prove nothing.
#
#     The same measurement over every candidate input:
#
#         w_o[0,0]        max|d| 3.1e-2      0 elements outside the band
#         w_qkv[0,0]      max|d| 3.1e-2      0
#         w_up[0,0]       max|d| 1.6e-1     49
#         w_down[0,0]     max|d| 1.3e+0    213
#         x[0,0]          max|d| 3.1e+0    491
#         ln2_weight[0]   max|d| 5.9e+0    488
#         ln1_weight[0]   max|d| 1.5e+0  36855
#
#     `ln1_weight` wins on the axis that matters, which is not the largest
#     single deviation but how much of the output moves: it scales one column of
#     `hidden`, and `hidden` is BOTH the FFN's input and the second addnorm's
#     residual, so the perturbation reaches 9% of the output through two
#     independent paths. Nothing averages the weight itself -- the normalization
#     that would is upstream of the multiply. `ln2_weight` is the same shape of
#     argument one stage later and moves 1.3% of the output; it is the fallback
#     if this one ever stops discriminating.
#
# THE STAGE TOLERANCES ARE PER BOUNDARY AND MEASURED
#     A single `atol` across ten boundaries would mean nothing: they span three
#     orders of magnitude, from `attn_out` at 1e-3 to `output` at 4. Each entry
#     in BLOCK_STAGE_ATOL is that boundary's measured `atol_required` rounded up,
#     the same methodology `kernel_registry` uses, and each is recorded in the
#     results artifact beside the statistics it was checked at.
# ---------------------------------------------------------------------------

# Per-boundary `atol` for the block's stage comparisons. `rtol` is RTOL for all
# of them, as everywhere else, and each entry is that boundary's MEASURED
# `atol_required` at 4096x768x3072 rounded up by the registry's usual 2-3x --
# except `q`/`k`/`v`, which keep the 5e-3 the `qkv_proj` rows above already use
# (a 1.6x margin here), and `output`, which is pinned to the spec's own `atol`
# because it is the same tensor compared the same way.
#
#     boundary       atol_required   atol    margin
#     q / k / v          3.1e-3      5e-3     1.6x
#     attn_context       2.3e-4      1e-3     4.4x
#     attn_out           7.4e-4      2.5e-3   3.4x
#     hidden             1.2e-2      3.5e-2   3.0x
#     ffn_up             5.0e-2      1.5e-1   3.0x
#     ffn_gelu           4.5e-2      1.5e-1   3.3x
#     ffn_out            1.1e-1      3.0e-1   2.6x
#     output             7.4e-2      1.0e-1   1.35x
#
# They span three orders of magnitude because the BOUNDARIES do: `attn_out` sits
# around 1e-3 at this golden model's scale and `output` around 4. A single atol
# across all ten would be vacuous at one end and unsatisfiable at the other.
BLOCK_STAGE_ATOL = {
    "q": 5e-3,
    "k": 5e-3,
    "v": 5e-3,
    "attn_context": 1e-3,
    "attn_out": 2.5e-3,
    "hidden": 3.5e-2,
    "ffn_up": 1.5e-1,
    "ffn_gelu": 1.5e-1,
    "ffn_out": 3e-1,
    "output": 1e-1,
}

# Where the four block ELFs are cached, relative to the working directory. It
# sits under the working directory rather than beside opcheck.py so `make
# clean` takes it with the rest of the build, and so a clean and a
# fault-injected run of the same shape share it: compilation depends on the
# shape and not on the data, and rebuilding four ELFs to perturb one weight
# would double the gate's hardware time for nothing.
#
# What makes that sharing safe is that reuse is keyed by FINGERPRINT and not by
# name -- the resolved registry specs, the built MLIR, the device kernel sources
# and the backend kwargs, per `builders/block_cache.py`.
# The two runs of a gate agree on all of them; a registry re-sweep or a builder
# edit does not, and recompiles rather than running the old ELF against the new
# recorded specs.
BLOCK_CACHE_DIR = "block_cache"


def prepare_block(shape, seed=42):
    """One whole ``encoder_bert`` layer against the golden model.

    Compiles and dispatches inside the returned ``dispatch`` callable's closure
    rather than here, so the injection -- which ``opcheck.py`` applies to
    ``inputs`` after this function has returned -- reaches the device buffers.
    """
    return prepare_layer_dispatch(shape, seed=seed)


def prepare_layer_dispatch(
    shape, seed=42, cache_dir=BLOCK_CACHE_DIR, label="block", extra=None
):
    """The full-layer preparation ``prepare_block`` and the Phase E modes share.

    One implementation, parameterized rather than copied, because everything in
    it IS the artifact contract the modes are measured against: the golden
    model's draws, the injection target, the per-boundary comparisons at
    ``BLOCK_STAGE_ATOL``, and the ``dispatch_vectors`` recorded straight from
    ``run_block``. A mode that re-implemented this glue could drift from the
    contract in exactly the ways the driver's cross-mode comparison cannot see.

    Args:
        cache_dir: the mode's OWN ELF cache directory. Never share one between
            modes: the directory is chosen by NAME, so two modes pointed at one
            can trade ELFs whose fingerprints happen to agree, and the result is
            numerically valid output attributed to the wrong execution boundary
            -- a failure no equivalence check would surface. See
            08b-phase-e2-coarse-and-instrumentation.md.
        label: prefix for the stage-summary prints, and nothing else. The lit
            recipes match on it, so it is the mode's own name.
        extra: merged into ``record_extra`` -- a mode adds its
            ``execution_mode`` CSV value here, from the one mapping in
            ``pattern/__init__.py``.

    The vectors are recorded UNCONDITIONALLY, on the fault-injected path as
    well as the clean one. The driver compares the two runs' summed totals and
    requires them EQUAL; a "skip instrumentation when injecting" shortcut would
    fail that check, not dodge it.
    """
    seq_len, emb_dim = shape["seq_len"], shape["emb_dim"]
    ffn_dim, num_heads = shape["ffn_dim"], shape["num_heads"]
    head_dim = shape["head_dim"]

    cfg = block_config(seq_len, emb_dim, ffn_dim, num_heads, head_dim)
    describe_block(cfg)

    golden = generate_golden_reference(
        seq_len, emb_dim, ffn_dim, num_heads, seed=seed, workload_variant="encoder_bert"
    )
    weights = golden["weights"]
    reference = golden["boundaries"]
    # The order is BLOCK_INPUT_NAMES; `inject` below indexes into it.
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
        cache_dir=cache_dir, verbose=False, profiler=Profiler(enabled=False)
    )
    compile_block_artifacts(cache, cfg, run_only=True)

    def dispatch(device_inputs, stage_stats):
        boundaries, vector_rows = run_block(cache, cfg, device_inputs)
        stages = []
        for name in BLOCK_BOUNDARIES:
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
        print(f"[{label}] stages: {clean}/{len(stages)} clean")
        # On the fault path too -- the lit recipes' FAULT half matches this
        # line, so an instrumentation made conditional on the injected flag
        # fails in the suite before the driver's totals comparison sees it.
        print(f"[{label}] recorded {len(vector_rows)} dispatch vectors")
        return [boundaries["output"]], {
            "stages": stages,
            "stages_passed": clean == len(stages),
            "dispatch_vectors": vector_rows,
        }

    record_extra = {
        "variant": "encoder_bert",
        "causal": False,
        "golden_seed": seed,
        "gemm_spec_source": cfg["qkv_source"],
        "gemm_spec_qkv": _spec_digest(cfg["qkv_spec"]),
        "gemm_spec_ffn_up": _spec_digest(cfg["ffn_up_spec"]),
        "gemm_spec_ffn_down": _spec_digest(cfg["ffn_down_spec"]),
        "gemm_spec_o_proj": _spec_digest(cfg["o_proj_spec"]),
        "addnorm_rows": cfg["norm_rows"],
        "addnorm_dispatches": cfg["norm_blocks"],
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
    record_extra.update(extra or {})
    return {
        "inputs": inputs,
        "expected": [reference["output"]],
        # ln1_weight, index 3. See the section header for the measurement that
        # rules out every attention-side target.
        "inject": (BLOCK_INPUT_NAMES.index("ln1_weight"), (0,)),
        "dispatch": dispatch,
        "record_extra": record_extra,
    }
