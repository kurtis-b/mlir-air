# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The operator catalogue ``opcheck.py`` runs: what to build, and with what data.

CONTRACT
    ``SPECS`` is the list of every ``(operator, shape_key)`` the port claims.
    Each entry carries the shape, its ``atol``, and a ``prepare`` callable
    returning the four things a run needs:

        module        the built ``air.ir.Module``
        inputs        every argument the host fills, in signature order
        expected      the FP32-derived expected outputs
        inject        ``(input_index, position)`` for the negative control

    plus the optional ``runner_kwargs`` (per-operator backend settings) and
    ``record_extra`` (provenance merged into the results artifact).

WHY THIS IS A SEPARATE MODULE FROM ``opcheck.py``
    Porting convention 5 caps a module at ~800 lines, and the seam that was
    already here is the one between the CHECK MECHANISM and the CATALOGUE it
    runs. ``opcheck.py`` keeps the mechanism -- the recording runner, the
    injection, the results artifact, the negative-control verdict and the CLI --
    and none of it knows what an operator is. This file knows every operator and
    nothing about how a verdict is reached. Adding an operator touches only this
    file; changing what counts as evidence touches only that one.

    The reference is ALWAYS computed here, from the CLEAN inputs, before any
    injection. ``opcheck.py`` perturbs afterwards. That ordering is the whole
    negative control: perturbing the reference instead would satisfy the letter
    of the check and destroy its purpose.

FOOTGUNS
    - The perturbed element is picked strictly BELOW the diagonal for
      ``causal_mask``. Above it the reference is ``-10000``, and
      ``rtol * 10000 = 160`` swallows any perturbation worth making -- the
      negative control would silently pass and prove nothing.
    - ``mha_out_proj`` injects into ``w_o``, not into Q/K/V, for the same
      class of reason. See the section header above ``_prepare_mha_out_proj``.
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
from builders.addnorm import addnorm_reference, build_addnorm_module  # noqa: E402
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
from builders.qkv_proj import (  # noqa: E402
    build_qkv_proj_module,
    qkv_gemm_spec,
    qkv_proj_reference,
)

# ---------------------------------------------------------------------------
# Per-operator preparation
#
# Each returns the four things a run needs: the built module, the device inputs,
# the FP32-derived expected outputs, and where to inject a fault. The reference
# is ALWAYS computed here, from the clean inputs, before any injection.
# ---------------------------------------------------------------------------


def _prepare_elementwise_add(shape, seed=0):
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


def _prepare_causal_mask(shape, seed=1):
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


def _prepare_layer_norm(shape, seed=2):
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


def _prepare_addnorm(shape, seed=3):
    rows, cols = shape["rows"], shape["cols"]
    # addnorm half only -- the FFN half collides with addnorm_ffn.o.
    ek.compile_encoder(build_ffn=False, build_addnorm=True)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((rows, cols)).astype(bfloat16)
    residual = rng.standard_normal((rows, cols)).astype(bfloat16)
    # A trained LayerNorm gamma sits near 1; uniform(0.5, 1.5) is that range
    # without being exactly 1, which would hide a dropped weight multiply.
    weight = rng.uniform(0.5, 1.5, size=cols).astype(bfloat16)
    return {
        "module": build_addnorm_module(rows, cols, bfloat16),
        "inputs": [x, residual, weight],
        "expected": [addnorm_reference(x, residual, weight)],
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


def _prepare_qkv_proj(shape, seed=4):
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


def _prepare_ffn(shape, seed=5):
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


def _prepare_mha_out_proj(shape, seed=6):
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


# Every (operator, shape) C1 and C2 claim. `atol` is the measured worst-case
# absolute error rounded up, per the kernel_registry methodology; `rtol` is fixed
# at RTOL for all of them.
SPECS = [
    {
        "operator": "elementwise_add",
        "shape_key": "512x512",
        "shape": {"rows": 512, "cols": 512},
        "atol": 5e-2,
        "prepare": _prepare_elementwise_add,
    },
    {
        "operator": "causal_mask",
        "shape_key": "512x512",
        "shape": {"rows": 512, "cols": 512},
        "atol": 5e-2,
        "prepare": _prepare_causal_mask,
    },
    {
        "operator": "layer_norm",
        "shape_key": "512x512",
        "shape": {"rows": 512, "cols": 512},
        "atol": 5e-2,
        "prepare": _prepare_layer_norm,
    },
    {
        # 64 rows, not 512: addnorm needs one kernel call per tile, which caps
        # rows at herd_x * (what fits L1). See builders/addnorm.py.
        "operator": "addnorm",
        "shape_key": "64x512",
        "shape": {"rows": 64, "cols": 512},
        "atol": 5e-2,
        "prepare": _prepare_addnorm,
    },
    {
        # The only registered projection shapes are (M, K, 3K) for K in
        # {1024, 2048}; 2048x1024x3072 is the smaller of the two. See the
        # structured report for the case-matrix shapes C4 still has to measure.
        "operator": "qkv_proj",
        "shape_key": "2048x1024",
        "shape": {"seq_len": 2048, "emb_dim": 1024},
        # Measured abs_err max 1.95e-3 over 6.3M elements at the registry GEMM
        # scale, mean_rel_L1 9.9e-3 -- which is the registry's own 9.3e-3 for
        # this GEMM, so the split-cast launches add nothing measurable. atol is
        # that worst case rounded up, 2.6x.
        "atol": 5e-3,
        "prepare": _prepare_qkv_proj,
    },
    {
        # The larger of the two registered (M, K, 3K) triples.
        "operator": "qkv_proj",
        "shape_key": "2048x2048",
        "shape": {"seq_len": 2048, "emb_dim": 2048},
        "atol": 5e-3,
        "prepare": _prepare_qkv_proj,
    },
    {
        "operator": "ffn",
        "shape_key": "2048x1024x3072",
        "shape": {"seq_len": 2048, "emb_dim": 1024, "ffn_dim": 3072},
        # Measured abs_err max 1.59e-3 over 2.1M elements, mean_rel_L1 1.6e-2 --
        # roughly 1.6x the single GEMM's, which is where the bf16 staging of h
        # and the activation's bf16 intermediates show up against an FP32
        # reference. atol is that worst case rounded up, 3.1x.
        "atol": 5e-3,
        "prepare": _prepare_ffn,
    },
    {
        # Fused attention + output projection, non-causal. 8 heads x head_dim
        # 64 = emb_dim 512, so the projection is 512x512x512 -- a registered
        # shape, which the registry resolves to `drain`.
        "operator": "mha_out_proj",
        "shape_key": "512x512x8h",
        "shape": {
            "seq_len": 512,
            "head_dim": 64,
            "num_heads": 8,
            "causal": False,
        },
        # Measured mean_rel_L1 4.6e-2 over 262144 elements -- the
        # FlashAttention tier (3.9e-2), which is where an operator whose
        # attention half IS that kernel belongs; the projection on top adds
        # nothing to the relative error. atol_required, the smallest atol this
        # run would have passed at, measured 1.85e-2; atol is that rounded up,
        # 2.7x, and stays well inside the 1e-1 ceiling.
        "atol": 5e-2,
        "prepare": _prepare_mha_out_proj,
    },
    {
        # Same shape with causal masking, as a SEPARATE key: masking changes
        # the arithmetic (early rows attend to a handful of keys, so |y| runs
        # an order of magnitude wider) and it is a distinct device path, so it
        # is distinct evidence.
        "operator": "mha_out_proj",
        "shape_key": "512x512x8h_causal",
        "shape": {
            "seq_len": 512,
            "head_dim": 64,
            "num_heads": 8,
            "causal": True,
        },
        # Measured mean_rel_L1 3.6e-2, atol_required 4.88e-2 over 262144
        # elements. A LOWER relative error than the non-causal row but a much
        # larger absolute one, because masking concentrates the softmax: the
        # first rows attend to a handful of keys, so |y| runs to 4.1 instead of
        # 0.35 and the same relative error lands further from zero.
        #
        # atol is 1.64x atol_required, below the registry's usual 2-3x margin
        # and deliberately so: 1e-1 is a HARD ceiling (the driver's objective
        # check rejects anything above it) and this datapath's honest error
        # gets within a factor of two of it. Widening to the FlashAttention
        # tier's own 1e-1 would buy the usual margin and nothing else; keeping
        # 8e-2 keeps this row strictly tighter than the tier it inherits.
        "atol": 8e-2,
        "prepare": _prepare_mha_out_proj,
    },
    {
        # Prefill-sized: 2048 positions, 16 heads x head_dim 64 = emb_dim 1024,
        # causal. The projection is 2048x1024x1024, also a registered shape.
        # Four times the sequence and twice the heads of the rows above, which
        # is what makes it evidence rather than a repeat: the launch grid grows
        # from 2x4 iterations to 8x8 and the reference's chunking loop from 2
        # blocks to 8.
        "operator": "mha_out_proj",
        "shape_key": "2048x1024x16h_causal",
        "shape": {
            "seq_len": 2048,
            "head_dim": 64,
            "num_heads": 16,
            "causal": True,
        },
        # Measured mean_rel_L1 4.1e-2, atol_required 4.81e-2 over 2097152
        # elements -- the same band as the 512 causal row over 8x the output,
        # so the error is set by the datapath and not by the shape, exactly as
        # the registry's FlashAttention rows record. Same 8e-2 for the same
        # reason; here the margin is 1.66x.
        "atol": 8e-2,
        "prepare": _prepare_mha_out_proj,
    },
]
