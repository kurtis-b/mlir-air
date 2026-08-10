# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Every ``(operator, shape)`` the transformer-layer port claims, and its ``atol``.

CONTRACT
    ``SPECS`` is a list of dicts, one per claimed point:

        operator    the operator name ``opcheck.py --operator`` selects on
        shape_key   the shape's name, unique within the operator
        shape       the shape as a DICT of named dimensions -- read from here,
                    never parsed back out of ``shape_key``
        atol        the measured worst-case absolute error rounded up
        prepare     the ``prepare_*`` callable from ``opcheck_prepare.py``

    A row may also carry, optionally:

        mean_rel_L1_max
                    a ceiling on the run's AGGREGATE mean relative L1 error,
                    enforced by ``opcheck.py`` on top of the element-wise
                    verdict (conjoined, so it can only make the verdict
                    stricter). For an operator whose claim is an aggregate --
                    norm_tail's is that keeping intermediates resident beats a
                    stated whole-layer figure -- the element-wise check alone
                    can pass while the claim fails; this key is how the row
                    states the claim where the run can check it.

    ``rtol`` is not here: it is fixed at ``opcheck.RTOL`` for every kernel in the
    registry and ``atol`` is what moves. See ``kernel_registry/README.md``.

    ``opcheck.py`` imports ``SPECS`` from this module and nothing else from it.

WHY THIS IS A SEPARATE MODULE FROM ``opcheck_prepare.py``
    See that file's docstring for the three-way split and the ~800-line cap that
    forced it. In short: this module is the CATALOGUE. It knows which shapes are
    claimed and at what tolerance, and nothing about how any of them is built.

EVERY ``atol`` HERE IS MEASURED, AND THE COMMENT SAYS SO
    Each entry carries the run that set it: ``n_elements``, ``mean_rel_L1``,
    ``abs_err_max`` and ``atol_required`` -- the smallest ``atol`` that run would
    have passed at -- with the margin stated as ``atol / atol_required``. That is
    the ``kernel_registry`` methodology, and the reason it is written down beside
    the number is that a tolerance nobody can check is not a gate. ``1e-1`` is a
    HARD CEILING; the driver's objective check rejects anything above it.

ORDERING MATTERS. The driver's per-operator negative control injects an
operator's FIRST declared shape, so a new row is appended AFTER its operator's
existing ones: the cheap early shape keeps the standing control. A row that needs
an injection of its own gets an explicit one -- ``check-ffn-ladder-fault`` in the
Makefile is the seq-64 ``ffn`` point's, and the comment there says why.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from opcheck_layer import prepare_block  # noqa: E402
from opcheck_prepare import (  # noqa: E402
    prepare_addnorm,
    prepare_ffn_accum,
    prepare_causal_mask,
    prepare_elementwise_add,
    prepare_elementwise_mul,
    prepare_ffn,
    prepare_layer_norm,
    prepare_softmax,
    prepare_mha_out_proj,
    prepare_norm_tail,
    prepare_qkv_proj,
    prepare_transpose,
)
from pattern.coarse import prepare_coarse  # noqa: E402
from pattern.coarse_c2 import prepare_coarse_c2  # noqa: E402
from pattern.coarse_c3 import prepare_coarse_c3  # noqa: E402
from pattern.fused import prepare_fused  # noqa: E402
from pattern.offload import prepare_offload  # noqa: E402
from pattern.runlist import prepare_runlist  # noqa: E402

# ---------------------------------------------------------------------------
# THE baseline_768 SET (D1)
#
# Phase C validated each operator at whatever width was cheapest to bring it up.
# The block runs at `baseline_768` -- hidden 768, ffn 3072, 12 heads x head_dim
# 64, encoder_bert -- and almost none of those points were there. The rows
# tagged "baseline_768" below close that gap so that a Phase D2 block failure
# localizes to the integration rather than to an operator nobody had run at that
# width.
#
# WHY THE THREE GEMM-BACKED ROWS ARE PINNED TO seq_len 4096 AND THE OTHER THREE
# ARE NOT
#     They are pinned to 4096 because that is what the D2 block runs, and
#     localizing a D2 failure to an operator is the whole point of the set.
#
#     WHEN THESE ROWS WERE WRITTEN there was also no choice: 4096 was the only
#     sequence length at which `build_ffn_module` built at hidden 768 at all. The
#     up-projection takes tile_n 128 and the down-projection tile_n 96 at every
#     point on the ladder, and two SAME-METHOD GEMMs at different tile_n declared
#     `f32_to_bf16_mn_<suffix>` twice with different memref types, which
#     `stitch_elf` rejects; 4096 escaped only because it is where the registry
#     puts them on different methods (drain up, fused-cast down), so the suffixes
#     differed for the wrong reason. PHASE E1 REMOVED THAT CONSTRAINT -- the
#     suffix and the mm.o name are functions of (method, tile_n) now -- and the
#     `ffn` 64x768x3072 row below is the second ladder point. The 4096 rows stay
#     pinned for the D2 reason, which was always the real one.
#
#     layer_norm, elementwise_add and addnorm are row-parallel: their builders
#     DERIVE the legal row count from L1 rather than accepting one, so only
#     `cols` is part of the family for them.
#
# WHY THERE IS NO baseline_768 causal_mask ROW
#     Its shape is seq x seq, not seq x hidden, so "the 768 width" does not name
#     a shape for it -- and `encoder_bert` never builds one: the golden
#     reference uses an all-ones attention mask for the encoder variant and a
#     `tril` one only for `decoder_gpt2`. A 4096x4096 point would cost real
#     hardware time to prove something the block does not exercise.
#
# ORDERING MATTERS HERE. The driver's per-operator negative control takes each
# operator's FIRST declared shape, so each new row is appended AFTER its
# operator's existing ones: the cheap Phase C shape keeps the standing control
# and every baseline_768 point gets an injection of its own. That is deliberate
# for `addnorm` above all -- its pre-add variant is the one function in this
# file that had never run on hardware, and it is the one whose reference is
# most likely to agree with the device by construction, since both were written
# here.
# ---------------------------------------------------------------------------

# Every (operator, shape) C1, C2, C3 and D1 claim. `atol` is the measured
# worst-case absolute error rounded up, per the kernel_registry methodology;
# `rtol` is fixed at RTOL for all of them.
SPECS = [
    {
        "operator": "elementwise_add",
        "shape_key": "512x512",
        "shape": {"rows": 512, "cols": 512},
        "atol": 5e-2,
        "prepare": prepare_elementwise_add,
    },
    {
        # baseline_768. Rows are free for this operator, so they are the
        # block's own 4096 rather than a cheaper number -- the tile shape is
        # `cols` wide and the row count only sets the herd loop's trip count,
        # so the full sequence costs almost nothing extra to check.
        "operator": "elementwise_add",
        "shape_key": "4096x768",
        "shape": {"rows": 4096, "cols": 768},
        # Measured over 3145728 elements: mean_rel_L1 1.879e-3 -- the same
        # 1.9e-3 the registry records for this kernel, so widening the row
        # count and the width changes nothing -- abs_err_max 3.125e-2, and
        # atol_required 0.0. Zero is not a rounding: a single bf16 rounding of
        # an f32 sum is always within rtol of the f32 value, so `rtol` alone
        # accounts for every element and `atol` is doing no work here. It stays
        # at the 5e-2 of the operator's other row rather than being driven to
        # something arbitrarily small, which would only measure how close two
        # unconstrained numbers happened to land.
        "atol": 5e-2,
        "prepare": prepare_elementwise_add,
    },
    {
        # PHASE E4: the operator built from nothing. 512x512 first for the
        # cheap standing negative control (the driver injects an operator's
        # FIRST declared shape); `b` is uniform(0.5, 1.5) so the injected
        # delta cannot be swallowed by a near-zero multiplicand -- see
        # prepare_elementwise_mul.
        "operator": "elementwise_mul",
        "shape_key": "512x512",
        "shape": {"rows": 512, "cols": 512},
        # Measured over 262144 elements: mean_rel_L1 2.745e-3, abs_err_max
        # 3.125e-2, atol_required 0.0. Zero for the same reason as
        # elementwise_add's: a single bf16 rounding of an exact-in-f32 product
        # is always within rtol alone, so atol does no work and stays at the
        # tier's 5e-2 rather than being driven arbitrarily small. The relative
        # error is ~1.4x add's 1.9e-3 because `b`'s uniform(0.5, 1.5) draw
        # shifts the product's exponent distribution, not because the kernel
        # rounds differently.
        "atol": 5e-2,
        "prepare": prepare_elementwise_mul,
    },
    {
        # baseline_768, the exact shape the runlist mode's two gamma
        # multiplies dispatch. Rows are free for this operator (the row count
        # only sets each tile's streaming trip count), so they are the block's
        # own 4096, same as elementwise_add's row above.
        "operator": "elementwise_mul",
        "shape_key": "4096x768",
        "shape": {"rows": 4096, "cols": 768},
        # Measured over 3145728 elements: mean_rel_L1 2.734e-3, abs_err_max
        # 3.125e-2, atol_required 0.0 -- within 0.4% of the 512x512 row over
        # 12x the elements, so widening the rows and narrowing the width
        # changes nothing, exactly as a single-rounding kernel should behave.
        # Same 5e-2 for the same reason.
        "atol": 5e-2,
        "prepare": prepare_elementwise_mul,
    },
    {
        # PHASE E4: pure data movement -- the transpose iron's runlist
        # dispatches as k_transpose. Validated standalone at the k^T shape the
        # gate configuration would give it; the runlist mode's README records
        # why it is not on that mode's dataflow (its consumer, the scores
        # GEMM, is host torch on this device). Every output element must be
        # BIT-identical to its input element, so n_mismatch 0 here is
        # exactness, not tolerance.
        "operator": "transpose",
        "shape_key": "4096x768",
        "shape": {"rows": 4096, "cols": 768},
        # Measured over 3145728 elements: mean_rel_L1 0.0, rel_err_max 0.0,
        # abs_err_max 0.0, atol_required 0.0 -- bit-exact, as data movement
        # must be. atol keeps the tier's 5e-2 because there is nothing to size
        # it against; any nonzero error here is a moved byte, and n_mismatch
        # catches it at any tolerance.
        "atol": 5e-2,
        "prepare": prepare_transpose,
    },
    {
        "operator": "causal_mask",
        "shape_key": "512x512",
        "shape": {"rows": 512, "cols": 512},
        "atol": 5e-2,
        "prepare": prepare_causal_mask,
    },
    {
        "operator": "layer_norm",
        "shape_key": "512x512",
        "shape": {"rows": 512, "cols": 512},
        "atol": 5e-2,
        "prepare": prepare_layer_norm,
    },
    # ---------------------------------------------------------------------
    # SOFTMAX. `[2026-08-09]` The operator corrected `runlist` needs, over the
    # streaming family already in programming_examples/softmax/softmax.cc.
    #
    # WHY EVERY ROW HERE CARRIES A mean_rel_L1_max AND THE OTHER ROW-WISE
    # OPERATORS DO NOT. A softmax row spans three orders of magnitude -- at
    # cols 768 the largest output is ~1.9e-2 and the mean is 1/768 = 1.3e-3 --
    # so an `atol` sized the registry way, from the worst ELEMENT, is set by
    # the handful of large outputs and is loose against the small ones: at
    # atol 7.5e-3 an expected 1e-4 would accept a returned 0. That is true of
    # any single atol over a distribution this skewed and it is not fixable by
    # choosing a different number.
    #
    # The element-wise check is kept as-is, per convention, and an AGGREGATE
    # ceiling is conjoined -- the mechanism opcheck.run_spec already has for
    # exactly this ("the element-wise check cannot enforce an aggregate
    # claim"), and which norm_tail uses. Measured mean_rel_L1 is 1.60-1.63e-2
    # across all three shapes, consistent with bf16 through an exp and a
    # divide; the ceiling is 2.5e-2, ~1.5x. A device that passed elementwise
    # while being globally wrong fails there.
    #
    # THE CONTROL HAD TO BE FIXED BEFORE THESE ROWS MEANT ANYTHING. Injecting
    # at (rows-1, 0), which is what every other row-wise operator here does,
    # left the injected run PASSING at 512x512 and 4096x768 -- +2.0 on a
    # low-probability element moves the tensor by 1.06e-3 / 7.43e-3 against an
    # atol of 7.5e-3. softmax is a normalization, so a perturbation's absolute
    # effect depends on where it lands, and at 512x512 the injected run's
    # abs_err_max equalled the clean run's exactly: no atol admitting the clean
    # run could reject that injection. opcheck_prepare.prepare_softmax now
    # targets the last row's argmax, measured per shape, which moves the tensor
    # 12-18x over atol at all three.
    {
        "operator": "softmax",
        "shape_key": "512x512",
        "shape": {"rows": 512, "cols": 512, "rows_per_call": 8},
        # Measured over 262144 elements: mean_rel_L1 1.630e-2, abs_err_max
        # 4.883e-3, atol_required 2.570e-3. atol is that rounded up 2.9x.
        "atol": 7.5e-3,
        "mean_rel_L1_max": 2.5e-2,
        "prepare": prepare_softmax,
    },
    {
        # baseline_768 width, at the block's row count.
        "operator": "softmax",
        "shape_key": "4096x768",
        "shape": {"rows": 4096, "cols": 768, "rows_per_call": 8},
        # Measured over 3145728 elements: mean_rel_L1 1.615e-2, abs_err_max
        # 4.883e-3, atol_required 2.586e-3. atol is that rounded up 2.9x.
        "atol": 7.5e-3,
        "mean_rel_L1_max": 2.5e-2,
        "prepare": prepare_softmax,
    },
    {
        # ATTENTION WIDTH. This row is the one corrected `runlist` actually
        # needs -- its per-head score matrix is [seq, seq], so the row width is
        # the sequence length and not the hidden size. Kept at 64 rows because
        # what this row exists to prove is that a 4096-wide row works at all;
        # the full [4096, 4096] is 32 MiB per tensor and belongs to the mode's
        # own gate rather than to an operator row.
        #
        # rows_per_call drops 8 -> 2: three [rows_per_call, cols] bf16 buffers
        # live in L1, so the row width is what bounds the row count. At 8 this
        # shape would ask for 192 KiB of a 64 KiB L1.
        "operator": "softmax",
        "shape_key": "64x4096",
        "shape": {"rows": 64, "cols": 4096, "rows_per_call": 2},
        # Measured over 262144 elements: mean_rel_L1 1.596e-2, abs_err_max
        # 8.545e-4, atol_required 5.459e-4 -- an order tighter than the 768
        # rows because the outputs are an order smaller (1/4096 against
        # 1/768). atol is that rounded up 2.7x.
        "atol": 1.5e-3,
        "mean_rel_L1_max": 2.5e-2,
        "prepare": prepare_softmax,
    },
    {
        # baseline_768, at the block's own row count for the same reason as
        # elementwise_add above.
        "operator": "layer_norm",
        "shape_key": "4096x768",
        "shape": {"rows": 4096, "cols": 768},
        # Measured over 3145728 elements: mean_rel_L1 7.117e-5, abs_err_max
        # 1.563e-2, rel_err_max 7.813e-3 and atol_required 0.0 -- rtol alone
        # covers every element since J7a's round-3 review moved
        # layer_norm_rows to f32 two-pass statistics with one rounding at the
        # store (the run that sized this atol, on the one-pass kernel,
        # measured mean_rel_L1 1.969e-3 and atol_required 1.419e-3). atol
        # stays that run's 5e-3 -- ten times tighter than the 512x512 row
        # above, which was sized before `atol_required` was recorded --
        # rather than being driven to something arbitrarily small.
        "atol": 5e-3,
        "prepare": prepare_layer_norm,
    },
    {
        # DOC 23 ITEM 2, the layer_norm half: rows at a COMMON OFFSET large
        # next to their spread (mean 8, sigma 0.25 -- |mean|/sigma 32), the
        # regime where a one-pass bf16 variance loses the row's statistics
        # entirely and the measured addnorm collapse boundary sits between
        # |mean|/sigma 2 and 4. J7a's round-3 move to two-pass f32 statistics
        # is what makes this kernel pass here, and until this row nothing
        # pinned that: doc 23 measured it clean (0/49152, mean_rel_L1 1.1e-4,
        # probe_addnorm_variance_cliff.py) and a revert to one-pass would have
        # failed only a probe nobody runs. 128 rows to match the sibling
        # norm_tail offset row, NOT first in declaration order, so the
        # standing negative control keeps its measured 512x512 target.
        "operator": "layer_norm",
        "shape_key": "128x768_offset",
        "shape": {"rows": 128, "cols": 768, "offset_regime": True},
        # Measured over 98304 elements: mean_rel_L1 9.819e-5, abs_err_max
        # 1.562e-2, rel_err_max 7.812e-3 and atol_required 0.0 -- rtol ALONE
        # covers every element, the same signature as the 4096x768 row,
        # because the f32 two-pass statistics are exact at any common offset
        # and what remains is per-element bf16 rounding on O(1) normalized
        # outputs. Within 1.4x of the zero-mean row's 7.117e-5, so the offset
        # costs the kernel nothing measurable. atol stays the sibling row's
        # 5e-3 rather than being driven to something arbitrarily small.
        "atol": 5e-3,
        "prepare": prepare_layer_norm,
    },
    {
        # 64 rows, not 512: addnorm needs one kernel call per tile, which caps
        # rows at herd_x * (what fits L1). See builders/addnorm.py.
        "operator": "addnorm",
        "shape_key": "64x512",
        "shape": {"rows": 64, "cols": 512},
        "atol": 5e-2,
        "prepare": prepare_addnorm,
    },
    {
        # baseline_768, PRE-ADD -- LayerNorm(x + residual) * weight, which is
        # what encoder_bert computes at both of its normalization points and
        # which nothing had ever dispatched. The row above is the post-add form
        # iron's operator carried; they are DIFFERENT FUNCTIONS, so this is
        # separate evidence and not a wider shape of the same evidence.
        #
        # Rows are 64 because that is what fits, DERIVED not carried over: at
        # cols 768 the pre-add path's three L1 tile buffers (x, residual, one
        # output -- the post-add entry point needs a fourth) cap
        # `addnorm_max_rows` at 104, and the post-add form at the same width at
        # 80. Both are ALLOCATION bounds that ignore aircc's ping-pong of the
        # DMA-fed buffers, so sitting at the cap risks a placement failure for
        # nothing; 64 keeps the same row count as the 512-wide row, which makes
        # the two directly comparable.
        "operator": "addnorm",
        "shape_key": "64x768_pre_add",
        "shape": {"rows": 64, "cols": 768, "pre_add": True},
        # Measured over 49152 elements: mean_rel_L1 2.687e-3, abs_err_max
        # 6.25e-2, atol_required 6.646e-4. atol is that rounded up, 3.0x.
        #
        # 26x tighter than the post-add row above (atol_required 1.747e-2) at a
        # HIGHER relative error, and the ordering is the whole reason. Post-add
        # finishes with `+ residual` in bf16, so a cancelling element -- one
        # where `norm * weight` nearly negates the residual -- carries an
        # absolute error set by the residual's magnitude while its own value is
        # near zero, and rtol covers none of it. Pre-add has no trailing add:
        # every error is proportional to the output that carries it. Do not
        # read the tighter number as the pre-add path being a better kernel; it
        # is the same kernel with the cancellation removed.
        "atol": 2e-3,
        "prepare": prepare_addnorm,
    },
    {
        # PHASE J7A: the norm tail as a three-herd pipeline -- add, norm and
        # gamma-scale in one segment joined by L1->L1 channels, x|residual
        # packed as planes into ONE strided L3 fetch. Same function as the
        # addnorm pre-add row above (LayerNorm(x + residual) * gamma), computed
        # by a different dataflow; the evidence it adds is the dataflow's.
        #
        # 128x768 FIRST for the standing negative control (the driver injects
        # an operator's first declared shape): two trips per tile, so the
        # channel feed order past trip one -- the class of defect J1 lost a
        # night to -- is exercised at a fraction of the 4096-row cost. The
        # inputs are deliberately asymmetric; see prepare_norm_tail on why a
        # symmetric draw would soften the pipeline's one unproven premise (the
        # plane-1 subview offset reaching the add kernel's base pointer).
        "operator": "norm_tail",
        "shape_key": "128x768",
        "shape": {"rows": 128, "cols": 768},
        # Measured over 98304 elements: mean_rel_L1 3.590e-3, abs_err_max
        # 6.25e-2, atol_required 2.637e-3. atol is the tier's 5e-2, a 19x
        # margin. The relative error is ~1.3x the fused addnorm pre-add row's
        # 2.687e-3, which is the pipeline's one extra bf16 rounding (the
        # normalized tensor is materialized between stage_norm and
        # stage_scale) behaving as expected. (First shipped at 4.354e-3;
        # the round-3 move of layer_norm_rows to f32 two-pass statistics
        # took the norm stage's share of the error down.)
        "atol": 5e-2,
        # The phase's actual claim, enforced as an aggregate ceiling: the
        # resident pipeline must beat the 1.688e-2 the block measures at this
        # width (the decomposed tail it replaces measures 1.784e-2). A
        # pipeline that is element-wise correct but round-trips its
        # intermediates through L3 passes np.isclose and fails here. Measured
        # 3.590e-3, a 4.7x margin.
        "mean_rel_L1_max": 1.688e-2,
        "prepare": prepare_norm_tail,
    },
    {
        # baseline_768, the block's own activation shape and the shape the
        # phase's precision claim is measured at: 128 trips per tile at
        # herd_x=8 (rows_per_call 4 -- 8 overflows L1 once aircc ping-pongs
        # both of stage_add's tiles; see builders/norm_tail.py).
        "operator": "norm_tail",
        "shape_key": "4096x768",
        "shape": {"rows": 4096, "cols": 768},
        # Measured over 3145728 elements: mean_rel_L1 3.620e-3 -- within 1% of
        # the 128-row point over 32x the elements, so the error is set by the
        # datapath and not the trip count, and comfortably under the driver's
        # 1.688e-2 clause (the whole-layer figure the resident pipeline must
        # beat) -- abs_err_max 6.25e-2, atol_required 4.375e-3. atol stays
        # the tier's 5e-2, an 11x margin, well under the 1e-1 ceiling.
        # (First shipped at 4.478e-3 / atol_required 2.425e-2; the round-3
        # move of layer_norm_rows to f32 two-pass statistics is the change.)
        "atol": 5e-2,
        # The driver's 1.688e-2 clause, enforced by the run itself at the
        # shape the phase's precision claim is measured at: see the 128x768
        # row above. Measured 3.620e-3, a 4.7x margin.
        "mean_rel_L1_max": 1.688e-2,
        "prepare": prepare_norm_tail,
    },
    {
        # J7A REVIEW ROUND 3: the norm stage on rows with a COMMON OFFSET
        # large next to their spread (x at mean 8, sigma 0.25; residual
        # identically zero so the bf16 sum is exact and the row isolates the
        # statistics -- see prepare_norm_tail). The one-pass variance
        # (E[x^2] - E[x]^2, bf16 row sum) layer_norm_rows first shipped loses
        # this row's variance ENTIRELY: it cancels below zero, clamps, and
        # normalizes by 1/sqrt(eps), putting ~700 of every 768 elements
        # outside tolerance while the two claimed shapes above -- zero-mean-
        # ish activations -- pass untouched. The kernel now keeps f32
        # statistics and computes variance two-pass; this row is what keeps
        # it that way.
        #
        # No mean_rel_L1_max: that ceiling is the phase's RESIDENCY claim,
        # measured at the block's activation distribution. This row's claim
        # is the LayerNorm contract itself, and conflating the two would let
        # a distribution change masquerade as a residency regression.
        "operator": "norm_tail",
        "shape_key": "128x768_offset",
        "shape": {"rows": 128, "cols": 768, "offset_regime": True},
        # Measured over 98304 elements: mean_rel_L1 2.799e-3, abs_err_max
        # 3.125e-2, rel_err_max 1.29e-2 and atol_required 0.0 -- rtol ALONE
        # covers every element, because residual-zero makes the bf16 sum
        # exact and the f32 statistics make the deviations exact at any
        # common offset, so what remains is per-element bf16 rounding on
        # O(1) normalized outputs, which is proportional by construction.
        # atol stays the tier's 5e-2 rather than being driven to something
        # arbitrarily small, same reasoning as elementwise_add's zero.
        "atol": 5e-2,
        "prepare": prepare_norm_tail,
    },
    {
        # The two model-registered projection shapes are (M, K, 3K) for K in
        # {1024, 2048}; 2048x1024x3072 is the smaller of the two. The C4 sweep
        # added the study's own (seq, 768, 2304) ladder beside them, and its
        # shortest point runs below.
        "operator": "qkv_proj",
        "shape_key": "2048x1024",
        "shape": {"seq_len": 2048, "emb_dim": 1024},
        # Measured abs_err max 1.95e-3 over 6.3M elements at the registry GEMM
        # scale, mean_rel_L1 9.9e-3 -- which is the registry's own 9.3e-3 for
        # this GEMM, so the split-cast launches add nothing measurable. atol is
        # that worst case rounded up, 2.6x.
        "atol": 5e-3,
        "prepare": prepare_qkv_proj,
    },
    {
        # The larger of the two registered (M, K, 3K) triples.
        "operator": "qkv_proj",
        "shape_key": "2048x2048",
        "shape": {"seq_len": 2048, "emb_dim": 2048},
        "atol": 5e-3,
        "prepare": prepare_qkv_proj,
    },
    {
        # The shortest point of the study's own ladder, and the one shape this
        # operator could not build at all until the C4 sweep found a fused-cast
        # configuration for it: 64x768x2304 was registered with `drain` and
        # `direct` only, and this builder pins fused-cast. It is here because a
        # resolution check alone cannot show that the row it found actually
        # computes the right thing -- the row it replaced was numerically wrong
        # (two of nine cast sub-tiles returned zeros), not absent.
        #
        # It is also the only entry here at herd 1x4 rather than 8x4: M = 64
        # cannot hold 8 rows of fused-cast's forced tile_m = 64, so the herd
        # comes from the registry row rather than the file-level default.
        #
        # Measured over the 147k elements of q, k and v together:
        # mean_rel_L1 9.9e-3 -- the registry's own value for this GEMM, so the
        # three split-cast launches add nothing measurable -- and atol_required
        # 1.32e-3. atol stays the 5e-3 the two shapes above use, which is a 3.8x
        # margin here and is also, to 2%, the K-scaled high-tier bound the sweep
        # itself checks at K = 768 (1.5e-3 * sqrt(8192/768) = 4.9e-3; see
        # sweep/sweep_measure.py on why that constant does not transfer across
        # K unscaled).
        "operator": "qkv_proj",
        "shape_key": "64x768",
        "shape": {"seq_len": 64, "emb_dim": 768},
        "atol": 5e-3,
        "prepare": prepare_qkv_proj,
    },
    {
        # baseline_768, at the block's sequence length. The GEMM is
        # 4096x768x2304 and the registry resolves fused-cast for it at the
        # file-level 8x4 herd, unlike the 64-row point above. `64x768` does NOT
        # substitute for this: the driver reads `seq_len` from the shape dict
        # and requires 4096, because a 64-row projection exercises one herd
        # iteration and the block's exercises sixty-four.
        "operator": "qkv_proj",
        "shape_key": "4096x768",
        "shape": {"seq_len": 4096, "emb_dim": 768},
        # Measured over the 9437184 elements of q, k and v together:
        # mean_rel_L1 9.863e-3 -- the registry's own 9.3e-3 for this GEMM, so
        # the three split-cast launches still add nothing measurable at 64x the
        # rows of the 64x768 point -- abs_err_max 1.953e-3 and atol_required
        # 1.773e-3. atol stays the 5e-3 the operator's other three rows use,
        # which is a 2.8x margin here.
        "atol": 5e-3,
        "prepare": prepare_qkv_proj,
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
        "prepare": prepare_ffn,
    },
    {
        # baseline_768, and until Phase E1 THE ONLY POINT ON THE LADDER WHERE
        # THIS OPERATOR BUILT AT hidden 768. The up-projection (N = 3072) takes
        # tile_n 128 and the down-projection (N = 768) takes tile_n 96
        # everywhere; two same-method GEMMs with different tile_n declared
        # `f32_to_bf16_mn_<suffix>` twice with different memref types and
        # stitch_elf rejected the redefinition. seq 4096 is where the registry
        # happens to put them on different methods -- drain for the up,
        # fused-cast for the down -- so the pair co-linked while 64..2048
        # (drain/drain) and 8192/16384 (fused-cast/fused-cast) did not.
        #
        # E1 made the symbol suffix and object name functions of
        # (method, tile_n) in `llms/shared/builders/gemm_builder.py`, so the
        # same-method pairs co-link too and the row below is the second point.
        # This one is KEPT AS IT IS: it is the only mixed-method pairing in the
        # file (drain up, fused-cast down), which is a different device path
        # from two drains, and it is the shape the D2 block runs.
        "operator": "ffn",
        "shape_key": "4096x768x3072",
        "shape": {"seq_len": 4096, "emb_dim": 768, "ffn_dim": 3072},
        # Measured over 3145728 elements: mean_rel_L1 1.569e-2 -- within 2% of
        # the 2048x1024x3072 row's 1.602e-2, so the bf16 staging of `h` and the
        # activation's intermediates dominate here too and the mixed-method
        # pairing (drain up, fused-cast down) costs nothing -- abs_err_max
        # 1.709e-3 and atol_required 1.472e-3. atol stays 5e-3, a 3.4x margin.
        "atol": 5e-3,
        "prepare": prepare_ffn,
    },
    {
        # PHASE E1: THE SECOND POINT ON THE SEQUENCE LADDER, and the reason this
        # sub-phase exists. Both registry rows here -- 64x768x3072 and
        # 64x3072x768 -- resolve to `drain`, at tile_n 128 and 96 respectively,
        # which is EXACTLY the pairing that could not be built before E1 made the
        # symbol suffix and mm.o name functions of (method, tile_n). No laxer test
        # could have produced this row: the module did not parse.
        #
        # seq 64 rather than 128..2048 because it is the cheapest such point and
        # every one of them is drain/drain, so the collision is equally exercised
        # and the hardware time is not. It is also the shortest M the registry
        # measured, so both GEMMs run at herd 2x4 rather than the file-level 8x4:
        # drain's forced tile_m = 32 admits at most herd_m = 2 at M = 64, and
        # `resolve_gemm_spec` reads that herd off the row rather than defaulting.
        # E2-E5 each need more than one sequence length; this is the row that
        # proves there is more than one.
        "operator": "ffn",
        "shape_key": "64x768x3072",
        "shape": {"seq_len": 64, "emb_dim": 768, "ffn_dim": 3072},
        # Measured over 49152 elements: mean_rel_L1 1.561e-2 -- within 0.6% of
        # the 4096-row point's 1.569e-2 and 2.6% of the 2048x1024x3072 row's
        # 1.602e-2, so this operator's relative error is set by the bf16 staging
        # of `h` and the activation's intermediates and not by the sequence
        # length, and two drains cost no more than a drain paired with a
        # fused-cast -- abs_err_max 1.465e-3 and atol_required 1.150e-3.
        #
        # atol stays the 5e-3 the operator's other two rows use, a 4.3x margin
        # here. It is the most headroom of the three, which is what 64x fewer
        # rows of the same arithmetic should give: the same per-element error
        # distribution sampled 64 times less often reaches a lower maximum.
        "atol": 5e-3,
        "prepare": prepare_ffn,
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
        "prepare": prepare_mha_out_proj,
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
        "prepare": prepare_mha_out_proj,
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
        "prepare": prepare_mha_out_proj,
    },
    {
        # baseline_768: 4096 positions, 12 heads x head_dim 64 = emb_dim 768,
        # NON-CAUSAL. encoder_bert is bidirectional -- `generate_golden_
        # reference` builds an all-ones attention mask for it and a `tril` one
        # only for decoder_gpt2 -- so the causal rows above are not this
        # operator's baseline_768 evidence, they are a different device path.
        # The projection is 4096x768x768, which the registry resolves to drain.
        #
        # The expensive row here: four times the sequence of the largest Phase
        # C point at three quarters of its heads, and non-causal, so every one
        # of the 16.8M score entries is computed rather than half of them.
        "operator": "mha_out_proj",
        "shape_key": "4096x768x12h",
        "shape": {
            "seq_len": 4096,
            "head_dim": 64,
            "num_heads": 12,
            "causal": False,
        },
        # Measured over 3145728 elements: mean_rel_L1 5.335e-2, abs_err_max
        # 9.033e-3, atol_required 8.706e-3. atol is that rounded up, 2.9x.
        #
        # The tightest atol of this operator's four rows and its LOOSEST
        # relative error, which is not a contradiction: softmax over 4096 keys
        # averages V eight times harder than over 512, so |y| shrinks by
        # roughly sqrt(8) and the same relative error lands closer to zero. The
        # relative error rising from 4.6e-2 to 5.3e-2 over that 8x reduction
        # depth is the FlashAttention tier behaving as its registry rows
        # record. atol is 4x below the 1e-1 ceiling, so this row -- the one the
        # block actually runs -- has the most headroom of the four, not the
        # least.
        "atol": 2.5e-2,
        "prepare": prepare_mha_out_proj,
    },
    {
        # PHASE D2: the whole encoder_bert layer, at the one configuration the
        # phase forces. `seq_len 4096` because that is the only point on the
        # ladder where `build_ffn_module` builds at hidden 768 (see the ffn row
        # above); `baseline_768` because it is the only family whose GEMMs
        # resolve; non-causal because encoder_bert is bidirectional.
        #
        # This shape is not a wider version of the rows above -- it is the first
        # thing here that runs several operators against EACH OTHER, with q, k
        # and v produced by one artifact and consumed by the next without
        # touching the host. What it can catch that D1 could not is a layout
        # transition, an argument map, an external-kernel collision across a
        # multi-operator sequence, or BO reuse under the Phase B allocator.
        "operator": "block",
        "shape_key": "4096x768_encoder_bert",
        "shape": {
            "seq_len": 4096,
            "emb_dim": 768,
            "ffn_dim": 3072,
            "num_heads": 12,
            "head_dim": 64,
        },
        # Measured over the 3145728 elements of the layer output: mean_rel_L1
        # 1.688e-2, abs_err_max 7.812e-2, atol_required 7.398e-2.
        #
        # atol is 1e-1, the HARD CEILING, at a margin of 1.35x -- the thinnest
        # in this file, and stated rather than padded because there is nowhere
        # to pad to. The whole layer's relative error, 1.688e-2, is within 8% of
        # the `ffn` operator's own 1.569e-2 at the same shape, so the FFN
        # dominates and nothing downstream is amplifying it; what makes the
        # ABSOLUTE number large is scale, not error. This golden model's
        # activations run around 1 where the registry's GEMM sweep puts a
        # depth-3072 reduction at 1/sqrt(3072), roughly 60x smaller, and the
        # `ffn` row's 1.472e-3 scaled by that factor is 9e-2 -- which is what
        # arrives at the second LayerNorm and, divided by its ~1.2 row standard
        # deviation and multiplied by a gamma in [0, 1), is the 7.4e-2 measured.
        #
        # The final LayerNorm renormalizing to roughly unit scale is the only
        # reason a chain of eight GEMMs, two LayerNorms and a softmax fits under
        # the ceiling at all. If a future change pushes this over 1e-1 the
        # answer is a defect report or a smaller `val_range`, not a wider
        # tolerance; the driver rejects anything above 1e-1 anyway.
        "atol": 1e-1,
        "prepare": prepare_block,
    },
    {
        # PHASE E2: the same layer as `block`, measured as the `coarse`
        # execution strategy -- few fused kernels over one runlist per
        # sequence. The CSV value it records is iron's old name for the mode,
        # confined by convention rule 7 to
        # pattern/__init__.py::EXECUTION_MODE_CSV.
        # The device path IS builders/block.py, through pattern/coarse's own
        # ELF cache; what this row adds is the mode's artifact -- operator
        # name, execution_mode, and the four per-sequence dispatch vectors the
        # Phase E distinguishability gate calibrates against (4 submissions /
        # 131 runlist entries, 128 of them addnorm's row blocking).
        "operator": "coarse",
        "shape_key": "4096x768_encoder_bert",
        "shape": {
            "seq_len": 4096,
            "emb_dim": 768,
            "ffn_dim": 3072,
            "num_heads": 12,
            "head_dim": 64,
        },
        # Same tensor as the block row, compared the same way at the same
        # golden seed, so the measurement is the block's: mean_rel_L1
        # 1.688e-2, atol_required 7.398e-2 over 3145728 elements, and atol is
        # the 1e-1 HARD CEILING at that row's 1.35x margin. See the block
        # entry above for why there is nowhere to pad to and why exceeding the
        # ceiling is a defect report, never a wider tolerance.
        "atol": 1e-1,
        "prepare": prepare_coarse,
    },
    {
        # PHASE E3: the same layer, measured as the `offload` execution
        # strategy — the host owns every intermediate and dispatches six
        # registry GEMMs one at a time, each its own one-step run_sequence
        # call, so the driver-summed vector is six submissions over six
        # entries: the mode aggregates NOTHING, which is its clause in the
        # distinguishability gate. Attention stays in host torch (its two
        # GEMMs resolve in no registry — pattern/offload/README.md has the
        # derivation), recorded as `attention_path` in the artifact: this is
        # a hybrid boundary, stated rather than discovered from the code.
        "operator": "offload",
        "shape_key": "4096x768_encoder_bert",
        "shape": {
            "seq_len": 4096,
            "emb_dim": 768,
            "ffn_dim": 3072,
            "num_heads": 12,
            "head_dim": 64,
        },
        # Same tensor as the block row, compared the same way at the same
        # golden seed, so the 1e-1 HARD CEILING carries over — and this mode
        # sits FURTHER from it than the block does. Measured over 3145728
        # elements: mean_rel_L1 1.396e-2, atol_required 5.489e-2, a 1.82x
        # margin against the block's 1.35x (mean_rel_L1 1.688e-2,
        # atol_required 7.398e-2): host FP32 attention and host norms land
        # closer to the FP32 oracle than the device path, so of the four
        # modes this one has the most headroom, not the least. See the block
        # entry for why exceeding the ceiling is a defect report, never a
        # wider tolerance.
        "atol": 1e-1,
        "prepare": prepare_offload,
    },
    {
        # PHASE E4: the same layer, measured as the `runlist` execution
        # strategy — the fine-grained point of the taxonomy. coarse's
        # dispatch schedule held fixed and every unit refined into
        # single-operator kernels: q/k/v; per head attn_scores -> softmax ->
        # attn_output, device-resident inside one submission; output_proj;
        # each normalization point as 64 bands of add + LayerNorm + gamma
        # multiply at coarse's own row granularity (builders.block.norm_rows,
        # imported); up_proj, GeLU, down_proj. The driver-summed vector is 17
        # submissions over 427 entries — above coarse's 131 BY CONSTRUCTION,
        # since every coarse unit maps onto one or more finer entries, which
        # is the ordinal clause this mode owns. pattern/runlist/README.md
        # records the refinement, its measured restage cost, and why the
        # first structure tried (13 streaming entries) was rejected.
        #
        # `[2026-08-09]` THIS COMMENT WAS STALE and is corrected here. It
        # described "host torch attention through the SAME blocked
        # implementation offload uses" and "5 submissions over 391 entries",
        # which is the mode as it stood BEFORE the corrected-taxonomy rebuild
        # (52b93e1a) put every operator on the device. The lit recipe was
        # moved with the rebuild and pins 17/427; this row was not, so the
        # catalogue asserted a structure the code had stopped having — and
        # the tolerance figures below travelled with it. Both are re-measured.
        "operator": "runlist",
        "shape_key": "4096x768_encoder_bert",
        "shape": {
            "seq_len": 4096,
            "emb_dim": 768,
            "ffn_dim": 3072,
            "num_heads": 12,
            "head_dim": 64,
        },
        # Same tensor as the block row, compared the same way at the same
        # golden seed, so the 1e-1 HARD CEILING carries over. `[2026-08-09]`
        # Re-measured on the REBUILT mode over 3145728 elements: mean_rel_L1
        # 1.746e-2, atol_required 6.981e-2, a 1.43x margin, zero mismatching
        # elements.
        #
        # The figures this row carried before — 1.732e-2 / 7.077e-2 / 1.41x,
        # and 1.755e-2 / 7.011e-2 / 1.43x before that — were measured on the
        # HOST-ATTENTION mode and are not comparable: the arithmetic changed
        # when both attention matmuls and the softmax moved to the device. Do
        # not read 1.732 -> 1.746 as a regression; it is a different
        # computation. The explanation that travelled with the old number
        # ("device norms but host f32 attention") no longer describes
        # anything, since nothing here runs on the host.
        #
        # See the block entry for why exceeding the ceiling is a defect
        # report, never a wider tolerance.
        "atol": 1e-1,
        "prepare": prepare_runlist,
    },
    {
        # PHASE E5: the same layer, measured as the `fused` execution
        # strategy — MLIR-level fusion via stitch_elf, the most fused point
        # of the taxonomy and the CSV value `fused_elf`. Three ELFs in ONE
        # runlist submission: the D2 qkv_proj and mha_out_proj modules
        # unchanged, then a stitched ln1+ffn+ln2 module over whole
        # [4096, 768] tensors — streamed norms, no row banding — which is
        # what removes the 386 sync boundaries coarse pays restaging its 64
        # addnorm bands per normalization point through the host. One ELF for
        # the whole layer is NOT available: attention requires
        # runtime_loop_tiling_sizes [1,1] and the 4096-row GEMMs are built at
        # [2,2], and one ELF is one aircc invocation — pattern/fused/fused.py
        # has the derivation. `[2026-08-08]` Measured: at [2,2] mha_out_proj
        # compiles and then hangs (ERT_CMD_STATE_TIMEOUT, 3/3), so the conflict
        # is real; omit_pingpong is NOT part of it and used to be cited here as
        # if it were. The driver-summed vector
        # is 1 submission over 3 entries, against coarse's 4 over 131; the
        # sync-boundary drop is the mode's gating clause in the Phase E
        # distinguishability check.
        "operator": "fused",
        # `[2026-08-08]` MOVED 4096 -> 1024, because the row could not build at
        # all. fused.py:37 has said "BOUNDED TO 256..1024" since the mode was
        # written and this row was never moved with it, so `make check-fused`
        # RAISED before aircc was reached:
        #
        #   ValueError: plane_major packing needs a plane stride of rows*cols
        #   (4096*768 = 3145728), over the shim aie.dma_bd cap of 1048576
        #
        # -- from builders/norm_tail.py, via fused.py's build_fused_tail_module
        # (plane_major=True). Confirmed by running the gate, which nobody had
        # done: it never reached the device. compile_fused_artifacts rebuilds
        # every module on every call even with run_only=True, so the cached
        # 4096 tail ELF could not rescue it either.
        #
        # 1024 is the top of the mode's own supported range, so this is the
        # widest shape the row can honestly claim. The alternative -- keeping
        # 4096 and switching the tail to row-interleaved packing, whose strides
        # stay at or under 2*cols at any row count -- is a real option and a
        # larger change: it requires the PRODUCER to write interleaved, which
        # is the whole reason plane_major exists.
        "shape_key": "1024x768_encoder_bert",
        "shape": {
            "seq_len": 1024,
            "emb_dim": 768,
            "ffn_dim": 3072,
            "num_heads": 12,
            "head_dim": 64,
        },
        # Same tensor as the block row, compared the same way at the same
        # golden seed, so the 1e-1 HARD CEILING carries over. Measured at 1024
        # over 786432 elements: mean_rel_L1 1.756e-2, atol_required 5.813e-2, a
        # 1.72x margin, every boundary n_mismatch 0. See the block entry for
        # why exceeding the ceiling is a defect report, never a wider
        # tolerance.
        #
        # The 4096 figures this row used to carry -- mean_rel_L1 1.784e-2,
        # atol_required 7.896e-2, a 1.27x margin described as "the thinnest of
        # the four modes" -- were produced before the row stopped building, and
        # are NOT comparable to the line above: different sequence length,
        # different element count. Do not read 1.27x -> 1.72x as an
        # improvement. What the composition argument said still holds and is
        # worth keeping: fused stacks device attention (whose error the
        # host-attention modes avoided) on the decomposed norm tail (which
        # stages bf16 between add, LayerNorm and gamma-mul where the fused
        # addnorm does not), and at 4096 that put it above block 1.688e-2 and
        # runlist 1.732e-2.
        "atol": 1e-1,
        "prepare": prepare_fused,
    },
    {
        # `coarse` CELL C2: the block front, the decomposed tail.
        #
        # 28-coarse-blend-space.md derives `coarse`'s blend space from the
        # artifact plans and finds it is two axes over six cells -- front
        # {block, runlist} x tail {stitched, banded, decomposed} -- of which
        # two ARE modes: (block, stitched) is `fused` and (runlist,
        # decomposed) is `runlist`. C1 = (block, banded) is today's `coarse`,
        # inherited from D2 rather than chosen. This row and the C3 row below
        # are the two INTERIOR cells, and they exist so the blend has
        # provenance: a measurement selects the cell.
        #
        # C2 holds `coarse`'s front and refines only the tail. Predicted 4
        # submissions over 389 entries at this shape (2 for the block front,
        # 3 + 6*64 for the decomposed tail), which is the ordinal claim the
        # pair owns: coarse's 131 < C3's 169 < C2's 389 < runlist's 427. Every
        # coarse tail unit maps onto finer units and the front is untouched,
        # so the bracket holds BY CONSTRUCTION.
        #
        # THE MEASUREMENT DOES NOT BELONG AT 1024. `fused`'s stitched tail
        # caps at 1365 rows (builders/norm_tail.py's plane_major stride
        # check), so at seq >= 2048 the whole stitched row of the table is
        # unbuildable and `coarse` is the mode you use where `fused` does not
        # fit. At 1024 every cell is dominated by one that already has a mode
        # name. This row gates at 4096 -- where the incumbent C1 and `runlist`
        # are also gated -- so the four points are compared where they are all
        # real.
        "operator": "coarse_c2",
        "shape_key": "4096x768_encoder_bert",
        "shape": {
            "seq_len": 4096,
            "emb_dim": 768,
            "ffn_dim": 3072,
            "num_heads": 12,
            "head_dim": 64,
        },
        # Same tensor as the block row, compared the same way at the same
        # golden seed, so the 1e-1 HARD CEILING carries over. `[2026-08-09]`
        # Measured over 3145728 elements: mean_rel_L1 1.784e-2, atol_required
        # 7.896e-2, a 1.27x margin, zero mismatching elements.
        #
        # THE THINNEST OF THE FOUR CORNERS, and the composition says why: the
        # error tracks the TAIL, not the front. The two decomposed-tail cells
        # land at 1.784e-2 (this one) and 1.746e-2 (`runlist`); the two
        # banded-tail cells at 1.688e-2 (`coarse`) and 1.654e-2 (C3). A
        # decomposed tail stages bf16 between the add, the LayerNorm and the
        # gamma multiply where the fused addnorm keeps it in one kernel, which
        # is the same effect `fused`'s row records for its own tail. 1.27x is
        # the least headroom any whole-layer row has ever carried; if a future
        # change pushes it past the ceiling the answer is a recorded defect,
        # never a wider tolerance. See the block entry.
        "atol": 1e-1,
        "prepare": prepare_coarse_c2,
    },
    {
        # `coarse` CELL C3: the runlist front, the banded tail. The other
        # interior cell -- see the C2 row above for the blend space and why
        # these two exist. C3 varies the axis C2 holds fixed: the front
        # becomes `runlist`'s three projections plus per-head attn_scores ->
        # softmax -> attn_output plus output_proj, and the tail stays
        # `coarse`'s banded one. Varying one axis per cell is what keeps the
        # comparison a comparison; the first `runlist` structure moved two at
        # once and its entry count measured the schedule change rather than
        # the decomposition.
        #
        # Predicted 17 submissions over 169 entries at this shape (4 + 3*12
        # for the runlist front, 1 + 2*64 for the banded tail). The 17
        # submissions are the front's memory bound -- one per head, ~800 MiB
        # if batched -- and not a schedule choice, exactly as in `runlist`.
        "operator": "coarse_c3",
        "shape_key": "4096x768_encoder_bert",
        "shape": {
            "seq_len": 4096,
            "emb_dim": 768,
            "ffn_dim": 3072,
            "num_heads": 12,
            "head_dim": 64,
        },
        # The 1e-1 hard ceiling, for the reason the C2 row states.
        # `[2026-08-09]` Measured over 3145728 elements: mean_rel_L1
        # 1.654e-2, atol_required 7.266e-2, a 1.38x margin, zero mismatching
        # elements. The BEST mean of the four corners, which is the other half
        # of the tail effect the C2 row describes -- this cell keeps the fused
        # addnorm and refines the front, and the front is not where the error
        # is made.
        "atol": 1e-1,
        "prepare": prepare_coarse_c3,
    },
    {
        # PHASE J7B: the FFN down-projection as a COMPILER-FORMED accumulator
        # ring -- the naive K loop (fetch C, in-place accumulate, store C per
        # step) that air-hoist-dma-in-accum-pattern collapses so C stays
        # L1-resident across K. First (and only) shape: the ladder's shortest
        # point's down half, 96 K steps at tile_k 32 over 4 columns. The
        # numbers are identical whether the ring formed or not, which is why
        # this operator also has a structural arm (ffn_accum_structure.py and
        # the driver's own --debug-ir clause), and why this row's evidence is
        # only clause one of three.
        "operator": "ffn_accum",
        "shape_key": "64x3072x768",
        "shape": {"seq_len": 64, "ffn_dim": 3072, "emb_dim": 768},
        # Measured over 49152 elements: mean_rel_L1 1.417e-2, abs_err_max
        # 1.831e-3, atol_required 1.383e-3 -- atol stays the GEMM tier's 5e-3,
        # a 3.6x margin. The relative error is an order above the other GEMM
        # rows and that is the ring's honest cost, not a defect: the in-place
        # kernel's C is bf16, so the running sum is rounded to bf16 once per K
        # step, 96 times here, where a drain-to-f32 GEMM rounds once. The
        # FP32 reference deliberately reproduces none of it.
        "atol": 5e-3,
        "prepare": prepare_ffn_accum,
    },
]
