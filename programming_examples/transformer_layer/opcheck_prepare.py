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

    The same cap split THIS file in Phase E4: it stood at 799 lines and two
    more operators (``transpose``, ``elementwise_mul``) had to land. The
    full-layer preparation -- the D2 block behind ``opcheck.py``'s ``dispatch``
    seam, ``BLOCK_STAGE_ATOL``, the dispatch-vector contract -- moved to
    ``opcheck_layer.py`` along its own section boundary.

    So the split is four ways, and each module knows one thing:

        opcheck.py          what counts as evidence (the recording runner, the
                            injection, the results artifact, the CLI)
        opcheck_prepare.py  how each STANDALONE operator is built and fed  <- this
        opcheck_layer.py    how the WHOLE-LAYER checks are prepared (block, and
                            the glue the Phase E modes share)
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
from builders.ffn_accum import (  # noqa: E402
    FFN_ACCUM_HERD_X,
    FFN_ACCUM_TILE_K,
    build_ffn_accum_module,
    compile_ffn_accum_kernel,
    ffn_accum_device_inputs,
    ffn_accum_reference,
)
from builders.ffn import (  # noqa: E402
    build_ffn_module,
    ffn_device_inputs,
    ffn_gemm_specs,
    ffn_reference,
)
from builders.elementwise_mul import (  # noqa: E402
    build_elementwise_mul_module,
    elementwise_mul_reference,
)
from builders.layer_norm import (  # noqa: E402
    build_layer_norm_module,
    layer_norm_reference,
)
from builders.norm_tail import (  # noqa: E402
    build_norm_tail_module,
    compile_norm_tail_kernels,
    norm_tail_device_inputs,
    norm_tail_reference,
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
from builders.qkv_proj import (  # noqa: E402
    build_qkv_proj_module,
    qkv_gemm_spec,
    qkv_proj_reference,
)
from builders.transpose import (  # noqa: E402
    build_transpose_module,
    compile_transpose_kernel,
    transpose_reference,
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


def prepare_elementwise_mul(shape, seed=7):
    """``out = a * b``, the operator Phase E4 built from nothing.

    ``b`` is drawn from ``uniform(0.5, 1.5)`` rather than a standard normal,
    and that is load-bearing twice over: it is the gamma-shaped operand the
    ``runlist`` mode actually feeds this kernel (a LayerNorm weight sits near
    1), and it bounds ``b`` away from zero at the injection site -- a fault in
    ``a`` moves the output by ``delta * b`` there, so a near-zero ``b`` would
    let the negative control pass inside the tolerance band.
    """
    rows, cols = shape["rows"], shape["cols"]
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((rows, cols)).astype(bfloat16)
    b = rng.uniform(0.5, 1.5, size=(rows, cols)).astype(bfloat16)
    return {
        "module": build_elementwise_mul_module(rows, cols, bfloat16),
        "inputs": [a, b],
        "expected": [elementwise_mul_reference(a, b)],
        "inject": (0, (rows - 1, 0)),
    }


def prepare_norm_tail(shape, seed=9):
    """The J7a three-herd pipeline: ``LayerNorm(x + residual) * gamma``.

    x and residual are drawn from deliberately DIFFERENT distributions --
    x standard normal, residual ``normal(0.75, 1.5)`` -- because the pipeline's
    one unproven premise is whether the AIE lowering hands the add kernel a
    base pointer that includes the plane-1 subview offset. If it passed the
    plane-0 base for both operands the device would compute
    ``LayerNorm(x + x) * gamma``, and LayerNorm is scale-invariant --
    ``LN(x + x) == LN(x)`` -- so the failure survives every check that a
    symmetric draw would soften. Distinct distributions make the wrong read
    disagree at nearly every element.

    The injection perturbs the x half of the PACKED buffer (input 0,
    position ``(rows-1, 0, 0)`` of its ``[rows, 2, cols]`` layout), last row:
    the whole row's statistics shift and the perturbed element lands far
    outside the band, same reasoning as ``prepare_layer_norm``'s.

    A shape carrying ``offset_regime`` draws x at a COMMON OFFSET large next
    to its spread (mean 8, sigma 0.25 -- mean/sigma 32) with residual
    identically zero. This is the regime where the one-pass variance
    (``E[x^2] - E[x]^2``) the norm kernel first shipped loses the variance
    entirely -- it cancels below zero, clamps, and normalizes by
    ``1/sqrt(eps)``, putting ~700 of every 768 elements outside tolerance --
    and where ``layer_norm_rows``'s two-pass f32 statistics must not. The
    zero residual is deliberate and is NOT the asymmetric-input discipline
    above: adding zero is exact in bf16, so the row isolates the norm stage's
    statistics from stage_add's own sum rounding. Plane addressing keeps its
    guard from the asymmetric rows, which still run.
    """
    rows, cols = shape["rows"], shape["cols"]
    compile_norm_tail_kernels()
    rng = np.random.default_rng(seed)
    if shape.get("offset_regime"):
        x = (8.0 + 0.25 * rng.standard_normal((rows, cols))).astype(bfloat16)
        residual = np.zeros((rows, cols), dtype=bfloat16)
    else:
        x = rng.standard_normal((rows, cols)).astype(bfloat16)
        residual = (0.75 + 1.5 * rng.standard_normal((rows, cols))).astype(bfloat16)
    # Gamma-shaped: a trained LayerNorm weight sits near 1, and uniform(0.5,
    # 1.5) is bounded away from zero so no element's gamma can swallow a fault.
    gamma = rng.uniform(0.5, 1.5, size=cols).astype(bfloat16)
    return {
        "module": build_norm_tail_module(rows, cols, bfloat16),
        "inputs": norm_tail_device_inputs(x, residual) + [gamma],
        "expected": [norm_tail_reference(x, residual, gamma)],
        "inject": (0, (rows - 1, 0, 0)),
    }


def prepare_transpose(shape, seed=8):
    """``out = a.T``, pure data movement -- the check is exactness, not tolerance.

    A fault anywhere in ``a`` lands bit-identically at the mirrored position,
    so the injection needs no magnitude argument: ``delta`` passes through
    unattenuated wherever it is put.
    """
    rows, cols = shape["rows"], shape["cols"]
    tile_rows = shape.get("tile_rows", 64)
    herd_x = shape.get("herd_x", 8)
    compile_transpose_kernel(tile_rows, cols // herd_x)
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((rows, cols)).astype(bfloat16)
    return {
        "module": build_transpose_module(
            rows, cols, bfloat16, herd_x=herd_x, tile_rows=tile_rows
        ),
        "inputs": [a],
        "expected": [transpose_reference(a)],
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


def prepare_ffn_accum(shape, seed=11):
    """The J7b accumulator-ring down-projection: ``y = a @ w``, bf16.

    Operands at the registry GEMM sweep's scale (both ``1/sqrt(K)`` for the
    depth-``ffn_dim`` reduction) so the recorded error sits on the same axis
    as the registry's GEMM rows -- this operator's accumulator rounds the
    running sum to bf16 every K step, and that cost is only comparable to
    the registry tier at the tier's own scale.

    The inputs are handed over PRE-TILED by ``ffn_accum_device_inputs`` (the
    device's blocked layouts; see builders/ffn_accum.py), so the injection
    position indexes the packed FLAT a. Index 0 is element (0, 0) of ``a``
    in every layout this packing can produce -- block (0,0), microtile
    (0,0), element (0,0) -- so the control does not depend on the packing
    arithmetic it is guarding. A delta there moves ``y[0, :]`` by
    ``delta * w[0, :]`` ~ 2/sqrt(3072) ~ 3.6e-2 per element, an order above
    the tolerance band at this scale.

    ``y`` need not enter zeroed: the builder zeroes the accumulator tile on
    device (a guarded ``ffn_zero_bf16_up_proj`` call on the K loop's first
    iteration), so a stale output BO cannot leak into the result.
    ``XRTRunner.run_test`` zero-fills its output placeholders regardless --
    which is exactly why this numeric arm could never have detected a
    reliance on that, and why the structural arm checks the zero is
    dispatched (review round 2).
    """
    seq_len, ffn_dim = shape["seq_len"], shape["ffn_dim"]
    emb_dim = shape["emb_dim"]
    # ONE source of truth for the tile geometry: the builder's own constants
    # (set by measured walls, not a registry sweep -- their definition in
    # builders/ffn_accum.py records why kernel_registry is not the source
    # here). The kernel object bakes DIM_M/DIM_K/DIM_N in as -D flags, so
    # compiling it at a different tile_n from the one the module declares
    # links the wrong microkernel and produces garbage that no import error
    # announces -- so tile_n is passed explicitly, derived from THIS row's
    # emb_dim, rather than trusting compile_ffn_accum_kernel's default to
    # track the shape (the default is derived from the module's DEFAULT
    # shape, which happens to coincide here; explicit is what stays right
    # when a second catalogue row lands at another width).
    herd_x, tile_k = FFN_ACCUM_HERD_X, FFN_ACCUM_TILE_K
    tile_n = emb_dim // herd_x
    compile_ffn_accum_kernel(tile_k=tile_k, tile_n=tile_n)
    rng = np.random.default_rng(seed)
    scale = _registry_gemm_scale(ffn_dim)
    a = (rng.standard_normal((seq_len, ffn_dim)) * scale).astype(bfloat16)
    w = (rng.standard_normal((ffn_dim, emb_dim)) * scale).astype(bfloat16)
    return {
        "module": build_ffn_accum_module(
            seq_len, ffn_dim, emb_dim, herd_x=herd_x, tile_k=tile_k
        ),
        "inputs": ffn_accum_device_inputs(a, w, herd_x=herd_x, tile_k=tile_k),
        "expected": [ffn_accum_reference(a, w)],
        "inject": (0, (0,)),
        "runner_kwargs": _GEMM_RUNNER_KWARGS,
    }


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
