# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How the WHOLE-LAYER checks are prepared: the D2 block and the Phase E modes.

CONTRACT
    ``prepare_layer_dispatch(shape, seed=..., cache_dir=..., label=...,
    extra=...)`` is the full-layer preparation ``prepare_block`` and the
    coarse-path Phase E modes share -- the golden model's draws, the measured
    injection target, the per-boundary comparisons at ``BLOCK_STAGE_ATOL``, and
    the ``dispatch_vectors`` recorded straight from ``run_block``.
    ``dispatch_vector_totals`` validates and sums recorded vectors exactly the
    way the driver's independent copy does. A mode with its own device path
    (``offload``, ``runlist``) imports the pieces rather than the whole:
    ``BLOCK_STAGE_ATOL`` and ``dispatch_vector_totals`` from here, its dispatch
    from its own ``pattern/<mode>/`` module.

WHY THIS IS A SEPARATE MODULE FROM ``opcheck_prepare.py``
    The same ~800-line cap that split ``opcheck_specs.py`` out of the old
    single file (see ``opcheck_prepare.py``'s docstring for that history).
    Phase E4 needed two more operator preparers and ``opcheck_prepare.py``
    stood at 799 lines, so the split happened along the seam that was already
    visible in its section headers: PER-OPERATOR preparation (one module, one
    hardware artifact each) stays there; the FULL-LAYER preparation (several
    ELFs behind ``opcheck.py``'s ``dispatch`` seam, per-boundary stages, the
    dispatch-vector contract) is this file. Adding an operator touches
    ``opcheck_prepare.py`` and the catalogue; adding an execution mode touches
    ``pattern/<mode>/`` and the catalogue; changing what a full-layer check
    records touches only this file.

THE MEASUREMENTS BEHIND THE CONSTANTS
    ``BLOCK_STAGE_ATOL`` and the ``ln1_weight`` injection choice are measured,
    not assumed, and the measurements are written where the numbers are
    declared below -- see the section comments, which moved here with the code.
"""

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_HERE)  # programming_examples/
for _p in (_PROJ_ROOT, os.path.join(_PROJ_ROOT, "llms"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from builders.block import (  # noqa: E402
    BLOCK_BOUNDARIES,
    BLOCK_INPUT_NAMES,
    DECODER_BLOCK_BOUNDARIES,
    block_config,
    compile_block_artifacts,
    describe_block,
    run_block,
    run_decoder_block,
)
from opcheck_prepare import _spec_digest  # noqa: E402
from pattern.reference import (  # noqa: E402
    generate_golden_reference,
    layer_inputs,
)

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
#     Every standalone operator in `opcheck_prepare.py` scales its operands to
#     where the registry's GEMM sweep measured them, because a standalone
#     operator has no other defensible scale. A whole layer does:
#     `pattern/reference.py` draws exactly what iron's
#     `generate_golden_reference` draws, in the same order, at
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
# except `q`/`k`/`v`, which keep the 5e-3 the `qkv_proj` rows already use
# (a 1.6x margin here), and `output`, which is pinned to the spec's own `atol`
# because it is the same tensor compared the same way.
#
#     boundary       atol_required   atol    margin
#     q / k / v          3.1e-3      5e-3     1.6x
#     attn_context       2.3e-4      4e-4     see below
#
# `[2026-08-19]` attn_context TIGHTENED 1e-3 -> 4e-4, and the number is set
# by a real recorded defect, not by the usual 2-3x rounding. The runlist
# attention ran UNSCALED (softmax(QK^T), no 1/sqrt(d)) from its landing
# until 2026-08-19 and this boundary's 1e-3 ceiling passed it at every
# length -- the absolute error stayed small while the relative error ran
# 25x the other modes. The measured table (Turbo, per-mode):
#
#     honest paths                unscaled defect (recorded, pre-fix)
#     FA kernel  2.288e-4 @4096   runlist  4.631e-4 @4096
#     FA kernel  2.716e-4 @1024   runlist  5.706e-4 @1024
#     FA kernel  3.165e-4 @512    runlist  5.452e-4 @512
#     offload    1.655e-4 @512
#     runlist(fixed) 0.996e-4 @4096, 1.388e-4 @1024, 1.617e-4 @512
#
# 4e-4 sits in the separation window: 1.26x over the honest maximum (the
# FlashAttention path at 512 -- thin, and deliberately so: widening past
# 4.6e-4 re-admits the defect) and every recorded defect point is >=1.16x
# OVER it. The window is narrow because the defect's absolute error is
# genuinely small; the RELATIVE error (3.7-9.8e-2 vs ~1.4e-2) is the wide
# signal, and a per-boundary mean_rel guard remains the named design task
# (it changes gate semantics the fault-injection lit halves pin).
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
    "attn_context": 4e-4,
    "attn_out": 2.5e-3,
    "hidden": 3.5e-2,
    "ffn_up": 1.5e-1,
    "ffn_gelu": 1.5e-1,
    "ffn_out": 3e-1,
    "output": 1e-1,
}

# Per-boundary `atol` for the DECODER's stage comparisons, MEASURED at
# 512x768x3072x12h causal (devq 359, first walk; devq 360, the audit), each
# entry the boundary's atol_required rounded up. THE DECODER'S NUMBERS ARE
# STRUCTURALLY LARGER THAN THE ENCODER'S AND THE AUDIT SAYS WHY: the encoder's
# chain has a shared raw input at q/k/v (device and reference feed the SAME x,
# so that boundary shows kernel error alone) and a final norm that shrinks
# accumulated drift; the pre-norm decoder has neither, so every boundary past
# `ln_in` carries upstream drift THROUGH the comparison.
#
# devq 360's stage-transfer audit is what licenses these as tolerances rather
# than as a defect report: for every stage it compared the device output
# against the reference STEP applied to the device's OWN upstream input
# (kernel-grade error, chaining removed) and against the reference chain
# (the number below). Transfer error: ln_in / residual / ffn_in / output
# EXACT within rtol (atol_required 0.0); attn_context 3.9e-2 and attn_out
# 2.2e-2, inside the causal row's 8e-2; ffn_up/gelu/out 3.6e-2 / 3.5e-2 /
# 9.5e-2, inside the encoder's FFN tiers. So the graph is correct and the
# totals are chain accumulation. THE ONE KERNEL-LEVEL FINDING: q/k/v transfer
# error is 3.5e-2 (mean_rel_L1 1.03e-2) against the encoder boundary's 5e-3 --
# the SAME fused-cast kernel at ~1% under the pre-norm input regime
# (LayerNorm x gamma widens the per-column dynamic range and the cancellation
# in the bf16 partial re-accumulation) where the encoder's raw N(0,1) draw
# holds it under 5e-3. Regime-dependence of this method, measured; the same
# shape of finding as addnorm's offset rows.
#
#     boundary       atol_required   atol    margin
#     ln_in              0.0         3.5e-2  (addnorm tier kept, not driven
#                                            arbitrarily small)
#     q / k / v          3.7e-2      7.5e-2   2.0x
#     attn_context       4.4e-2      8e-2     1.8x
#     attn_out           7.3e-2      1.5e-1   2.1x
#     residual           7.3e-2      1.5e-1   2.1x
#     ffn_in             1.5e-1      3.0e-1   2.0x
#     ffn_up             1.6e-1      3.5e-1   2.1x
#     ffn_gelu           1.6e-1      3.5e-1   2.2x
#     ffn_out            2.9e-1      4.5e-1   1.5x
#     output             3.0e-1      4.5e-1   1.5x
#
# The two end boundaries run the thin margins end boundaries run everywhere
# in this file (`output` 1.35x above, the causal rows 1.5-1.6x). `output` is
# also the whole-layer comparison's atol via the preparer's `atol` key: same
# tensor, same comparison, one number.
DECODER_STAGE_ATOL = {
    "ln_in": 3.5e-2,
    "q": 7.5e-2,
    "k": 7.5e-2,
    "v": 7.5e-2,
    "attn_context": 8e-2,
    "attn_out": 1.5e-1,
    "residual": 1.5e-1,
    "ffn_in": 3e-1,
    "ffn_up": 3.5e-1,
    "ffn_gelu": 3.5e-1,
    "ffn_out": 4.5e-1,
    "output": 4.5e-1,
}

# `[2026-08-20]` The table above was measured at ONE width, and the first walk
# of the 1024-wide decoder family (gpt2_medium_1024: 1024x4096, devq 430)
# crossed its last two ceilings on the two device-norm modes while the
# mean-relative error stayed where the 768 family reads it: `coarse` ffn_out
# atol_required 4.869e-1 (2 of 1,048,576 elements over, mean_rel 4.95e-2),
# `fused` 5.412e-1 (40 over, mean_rel 5.57e-2); `offload` and `runlist` 12/12
# clean (offload's tail 2.40e-1 -- its pre-norms are host f32). That is the
# element-wise tail of a 4096-deep bf16 reduction against a ceiling sized at
# 3072, not a 25x relative excess of the kind the attention-scale defect read
# as. Per-width entries are therefore MEASURED and rounded up, like the base
# table: 1024 -> 6e-1 on ffn_out/output (1.11x over fused's 5.412e-1). A
# width with no entry uses the base table unchanged, which is why gpt2_512
# (hidden 512, ffn 2048; 4/4 clean at devq 429) needs none. Confirmation
# walk: the second gpt2_medium_1024 smoke must read 4/4 under these.
DECODER_STAGE_ATOL_BY_HIDDEN = {
    1024: {"ffn_out": 6e-1, "output": 6e-1},
}


def decoder_stage_atol(hidden_size):
    """The decoder's per-boundary atol table at ``hidden_size``.

    The one authority the four modes read, so a per-width entry lands in all
    of them at once and none can carry its own copy of the base table.
    """
    table = dict(DECODER_STAGE_ATOL)
    table.update(DECODER_STAGE_ATOL_BY_HIDDEN.get(int(hidden_size), {}))
    return table

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

# `DispatchVector.as_row()` (llms/shared/infra/dispatch.py): five counts and
# one derived MEAN. The driver checks the same contract with its own copy
# (agents/scripts/port-loop/phase_e_checks.py::vector_totals), deliberately --
# neither side trusts the other's arithmetic.
_VECTOR_COUNT_KEYS = (
    "host_submissions_per_layer",
    "air_launches_per_elf",
    "herd_launches",
    "sync_boundaries",
    "bytes_transferred",
)
_VECTOR_MEAN_KEY = "runlist_entries_per_submission"
_VECTOR_KEYS = _VECTOR_COUNT_KEYS + (_VECTOR_MEAN_KEY,)


def dispatch_vector_totals(rows):
    """Validate recorded dispatch vectors and sum them the way the driver does.

    Every row must be ``DispatchVector.as_row()`` verbatim: all six keys,
    finite non-negative values, whole-number counts, at least one submission
    per row, some bytes moved overall. ``runlist_entries_per_submission`` is a
    derived MEAN (``dispatch.py``), so total entries are
    ``sum(round(mean * submissions))`` -- never a naive sum of the means --
    and a product that is not a whole number of entries is rejected, because
    that is the shape a fabricated number takes.

    Raises ``ValueError`` on any violation, failing the opcheck run and its
    lit gate before the driver's independent arithmetic sees the artifact.
    The lit recipes pin the returned totals to one set of literals in BOTH
    halves, clean and fault-injected, so wrong contents or a fault run whose
    totals drift from the clean run's fail in the suite too.
    """
    if not rows:
        raise ValueError("no dispatch vectors were recorded")
    totals = {
        "host_submissions": 0,
        "runlist_entries": 0,
        "air_launches": 0,
        "herd_launches": 0,
        "sync_boundaries": 0,
        "bytes_transferred": 0,
    }
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"dispatch_vectors[{i}] is not a dict")
        missing = [k for k in _VECTOR_KEYS if k not in row]
        if missing:
            raise ValueError(
                f"dispatch_vectors[{i}] is missing {', '.join(missing)}; "
                "record DispatchVector.as_row(), never a hand-built dict"
            )
        for key in _VECTOR_KEYS:
            value = row[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"dispatch_vectors[{i}][{key!r}]={value!r} is not a number"
                )
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"dispatch_vectors[{i}][{key!r}]={value!r} is negative/non-finite"
                )
        for key in _VECTOR_COUNT_KEYS:
            if float(row[key]) != int(row[key]):
                raise ValueError(
                    f"dispatch_vectors[{i}][{key!r}]={row[key]!r} is a fractional count"
                )
        subs = int(row["host_submissions_per_layer"])
        if subs < 1:
            raise ValueError(
                f"dispatch_vectors[{i}] records {subs} host submissions; "
                "every recorded sequence submitted at least once"
            )
        product = float(row[_VECTOR_MEAN_KEY]) * subs
        if abs(product - round(product)) > 1e-9:
            raise ValueError(
                f"dispatch_vectors[{i}]: {row[_VECTOR_MEAN_KEY]!r} entries "
                f"per submission over {subs} submission(s) is not a whole "
                "number of runlist entries"
            )
        totals["host_submissions"] += subs
        totals["runlist_entries"] += int(round(product))
        totals["air_launches"] += int(row["air_launches_per_elf"])
        totals["herd_launches"] += int(row["herd_launches"])
        totals["sync_boundaries"] += int(row["sync_boundaries"])
        totals["bytes_transferred"] += int(row["bytes_transferred"])
    if totals["bytes_transferred"] <= 0:
        raise ValueError("every recorded dispatch vector moved zero bytes")
    return totals


# ---------------------------------------------------------------------------
# The fault-injection delta, DERIVED for an injected weight set (doc 58 M1)
#
# WHY THE SHIPPED CONSTANT CANNOT SIMPLY BE CARRIED OVER
#     `opcheck.py:145-148` sizes FAULT_DELTA = 2.0 "two orders of magnitude
#     above the tolerance band AT THESE INPUT SCALES", where the scales are
#     `pattern/reference.py`'s VAL_RANGE 0.05 `randn` weights and `torch.rand`
#     gammas. `atol` is ABSOLUTE, so the constant's discriminating power is
#     really `atol / (layer output scale)` -- and a weight set with a different
#     scale moves that ratio without touching a line of code.
#
#     MEASURED, doc 58 M1, at 512x768x3072x12 with real layer-0 tensors sliced
#     to these shapes (results/item31-4a-m1-20260827/evidence/fault-delta-*):
#
#         variant   weights            output rms   ln1+2.0 max|d|   outside band
#         encoder   generated             0.5716         1.526e+0          36855
#         encoder   real Qwen3-0.6B       0.5428         8.523e+0          19621
#         encoder   real Llama-3.2-1B     0.2163         3.488e+0           2295
#         decoder   generated             1.4153         1.842e+0          43948
#         decoder   real Qwen3-0.6B       0.3130         6.914e-1             25
#         decoder   real Llama-3.2-1B     0.0789         7.202e-2              0
#
#     The last row is the trap doc 58a section 3.2(c) named, arriving: the
#     injected run PASSES and the negative control proves nothing. The mechanism
#     is not the gamma's magnitude (it is O(1) in every source) but the
#     DECODER's output scale. The encoder is post-norm, so its final boundary is
#     renormalized and its scale is set by the gamma; the pre-norm decoder's
#     final boundary is the RAW residual sum, so its scale is set by the weights
#     -- and real projections (rms 1.2e-2 to 3.1e-2) are 2-4x smaller than the
#     generated 5.0e-2 over a four-multiply chain. The decoder's atol is also
#     4.5x wider. At Llama's scale the band (4.5e-1) exceeds the layer output's
#     own absmax (3.75e-1): EVERY candidate reads zero elements outside, and a
#     device returning zeros would pass. That is a statement about the absolute
#     tolerance table, not about the delta, and it is recorded in doc 58 as the
#     thing M2 must settle before it trusts a real-weight cell.
#
# WHAT THIS DOES INSTEAD
#     Derives the delta from the RESPONSE, using the same oracle the preparer
#     already owns: perturb, recompute the layer, measure how far outside the
#     band the worst element lands, and double until it clears MIN_EXCESS. The
#     generated path never reaches here -- it keeps FAULT_DELTA exactly, which
#     is what "no behaviour change for any existing caller" means.
# ---------------------------------------------------------------------------

#: How far outside the band the derived delta must push the worst element.
#: 2x, and the reason is the device's own error rather than a round number: a
#: clean run that PASSES has every element inside the band by definition, so a
#: perturbation whose response exceeds the band by 2x cannot be closed by the
#: device error the clean run just demonstrated is smaller than 1x. Below that
#: the host margin can be real and the device verdict still a coin flip -- the
#: 25-elements-of-393216 row above is exactly that case.
FAULT_EXCESS_MIN = 2.0

#: Doublings allowed before the search gives up. 8 doublings from the shipped
#: 2.0 reaches 512, well past the 32 the widest measured case needed. A cap hit
#: is a REFUSAL, not a fallback: it means no perturbation of this target leaves
#: this tolerance band, i.e. the band is vacuous at this scale, and silently
#: running on would be the vacuous negative control this whole section exists to
#: prevent.
FAULT_DELTA_MAX_DOUBLINGS = 8


def band_excess(actual, expected, atol, rtol):
    """How far outside ``np.isclose``'s band the worst element lands, as a ratio.

    ``max |a - e| / (atol + rtol * |e|)``, elementwise. ``1.0`` is exactly on
    the band; ``opcheck.py``'s comparison rejects a run iff this exceeds 1.0
    somewhere. Returning the RATIO rather than a count is what makes a margin
    comparable across two tolerance tables and two output scales -- a count
    depends on how many elements happen to sit near the edge.
    """
    a = np.asarray(actual, np.float32)
    e = np.asarray(expected, np.float32)
    band = atol + rtol * np.abs(e)
    return float(np.max(np.abs(a - e) / band))


def derive_fault_delta(
    golden_kwargs, weights, target_key, target_index, clean_output, atol, rtol, base
):
    """The smallest ``base * 2^k`` whose injection provably leaves the band.

    Args:
        golden_kwargs: what ``generate_golden_reference`` was called with, minus
            ``weights`` -- so the recomputation is the SAME layer, not a second
            transcription of it.
        weights: the injected weight set. Never mutated; each probe perturbs a
            fresh copy of the one tensor, exactly as ``opcheck.py::_inject``
            does, so the content key moves for the same reason it does there.
        target_key, target_index: the weight and element the injection hits.
        clean_output: the layer output the reference already computed.
        atol, rtol: the tolerances the whole-layer comparison will actually use.
        base: ``opcheck.py``'s ``FAULT_DELTA``, the starting point.

    Returns ``(delta, excess)``. Raises ``ValueError`` at the cap, with the
    measured responses, because a target that cannot be pushed outside this band
    is a tolerance defect and not something to work around.
    """
    delta = float(base)
    trace = []
    for _ in range(FAULT_DELTA_MAX_DOUBLINGS + 1):
        perturbed = dict(weights)
        buf = np.array(weights[target_key], copy=True)
        before = float(buf[target_index])
        buf[target_index] = np.asarray(before + delta, dtype=buf.dtype)
        perturbed[target_key] = buf
        probe = generate_golden_reference(**golden_kwargs, weights=perturbed)["output"]
        excess = band_excess(probe, clean_output, atol, rtol)
        trace.append((delta, excess))
        if excess >= FAULT_EXCESS_MIN:
            return delta, excess
        delta *= 2.0
    raise ValueError(
        f"no delta up to {trace[-1][0]:g} pushes {target_key}{list(target_index)} "
        f"more than {FAULT_EXCESS_MIN}x outside the band "
        f"(atol {atol:.3e}, rtol {rtol:.3e}); measured excess "
        + ", ".join(f"{d:g}->{e:.3f}" for d, e in trace)
        + ". The negative control cannot discriminate at this weight scale: the "
        "band is wide relative to the layer output, which is a defect report "
        "about the tolerance table, not a delta to widen further."
    )


def fault_delta_hook(golden_kwargs, weights, target_key, target_index, clean_output, atol):
    """A callable ``opcheck.py`` resolves ONLY when it is actually injecting.

    ``None`` for the generated path, so that path pays nothing and behaves
    exactly as before. For an injected weight set this returns a zero-argument
    callable -- deferred because the derivation costs one whole-layer oracle
    evaluation per doubling, and a clean run must not pay for a control it is
    not running.
    """
    if weights is None:
        return None

    def resolve():
        # opcheck.py imports opcheck_specs, which imports this module, so the
        # import must be deferred to call time. Under `python3 opcheck.py` the
        # script is `__main__` and this loads a second module object: harmless
        # (module level is imports and constants; `main()` is guarded) and it
        # keeps ONE definition of RTOL and FAULT_DELTA rather than a mirror
        # here that could drift from the file the gate actually reads.
        import opcheck

        delta, excess = derive_fault_delta(
            golden_kwargs,
            weights,
            target_key,
            target_index,
            clean_output,
            atol,
            opcheck.RTOL,
            opcheck.FAULT_DELTA,
        )
        print(
            f"[fault-delta] injected weights: {target_key}{list(target_index)} "
            f"+{delta:g} puts the worst element {excess:.2f}x outside the band "
            f"(atol {atol:.1e}); shipped FAULT_DELTA is {opcheck.FAULT_DELTA:g}"
        )
        return delta

    return resolve


def reconfiguration_delta(cache, baseline):
    """The ``(context_loads, kernel_attaches)`` THIS dispatch performed, as a dict.

    ``KernelCache.reconfiguration_counts()`` is cumulative since the cache was
    built -- the right number for ``offload``'s gated ``reconfiguration:``
    line, which ``run_npu2_offload_peano.lit`` pins ACROSS dispatches, and the
    wrong one for a results row, which records one steady-state layer dispatch
    (schema v2's ``context_loads`` / ``kernel_attaches``). Every whole-layer
    ``dispatch`` snapshots the counters at entry and reports the difference at
    exit through this one helper, so what counts as a reconfiguration stays
    defined in exactly one place -- ``ensure_loaded``'s single increment,
    which counts an ELF ``backend.load`` and an xclbin load identically, and
    counts an evicted context's reload AGAIN. That reload is the number:
    ``offload``-ELF's 30 per layer and the ``runlist`` front's per-head
    attention reloads are this delta, where the standing-context modes read 0.

    Args:
        cache: the mode's ``KernelCache``.
        baseline: ``cache.reconfiguration_counts()`` captured at dispatch entry.
    """
    loads, attaches = cache.reconfiguration_counts()
    return {
        "context_loads": loads - baseline[0],
        "kernel_attaches": attaches - baseline[1],
    }


def print_dispatch_totals(label, vector_rows):
    """Validate, sum and print a mode's recorded vectors, one shared format.

    The two lines here are what every mode's lit recipe pins to literals, in
    BOTH halves (clean and fault-injected): instrumentation conditional on the
    injected flag, a malformed row (``dispatch_vector_totals`` raises, failing
    the run), or drifted totals fail in the suite before the driver's
    comparison sees them. One implementation so a mode cannot drift the format
    while keeping the semantics -- or the reverse.
    """
    print(f"[{label}] recorded {len(vector_rows)} dispatch vectors")
    totals = dispatch_vector_totals(vector_rows)
    print(
        f"[{label}] dispatch totals: "
        f"submissions {totals['host_submissions']} "
        f"entries {totals['runlist_entries']} "
        f"air {totals['air_launches']} "
        f"herd {totals['herd_launches']} "
        f"sync {totals['sync_boundaries']} "
        f"bytes {totals['bytes_transferred']}"
    )
    return totals


def prepare_block(shape, seed=42, weights=None):
    """One whole ``encoder_bert`` layer against the golden model.

    Compiles and dispatches inside the returned ``dispatch`` callable's closure
    rather than here, so the injection -- which ``opcheck.py`` applies to
    ``inputs`` after this function has returned -- reaches the device buffers.
    """
    return prepare_layer_dispatch(shape, seed=seed, weights=weights)


def prepare_layer_dispatch(
    shape, seed=42, cache_dir=BLOCK_CACHE_DIR, label="block", extra=None, weights=None
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
        weights: doc 58 phase M1's injection seam. ``None`` draws the weight set
            as every pre-M1 caller does; a dict replaces it. The input
            activation, the draw order, the oracle, the injection target and the
            per-boundary tolerances are untouched either way -- only the tensors
            move. See ``pattern/reference.py::generate_golden_reference`` for why
            the arrays are kept BY IDENTITY, which is what keeps the negative
            control's content key content-derived.

    The vectors are recorded UNCONDITIONALLY, on the fault-injected path as
    well as the clean one, then validated and summed by
    ``dispatch_vector_totals``. The driver requires the two runs' summed
    totals EQUAL, and the lit gate pins the printed totals to one set of
    literals in both halves; a "skip instrumentation when injecting" shortcut
    fails both, not dodges them.
    """
    seq_len, emb_dim = shape["seq_len"], shape["emb_dim"]
    ffn_dim, num_heads = shape["ffn_dim"], shape["num_heads"]
    head_dim = shape["head_dim"]
    # The variant rides in the shape dict (run_mode._shape_for stamps it from
    # the family) so the per-mode preparers pass it through unchanged. Absent
    # means the encoder, which is every pre-family caller.
    variant = shape.get("workload_variant", "encoder_bert")
    causal = variant == "decoder_gpt2"
    boundary_names = DECODER_BLOCK_BOUNDARIES if causal else BLOCK_BOUNDARIES
    stage_atol = decoder_stage_atol(emb_dim) if causal else BLOCK_STAGE_ATOL
    run_layer = run_decoder_block if causal else run_block

    cfg = block_config(seq_len, emb_dim, ffn_dim, num_heads, head_dim, causal=causal)
    describe_block(cfg)

    golden_kwargs = dict(
        seq_len=seq_len,
        hidden_size=emb_dim,
        intermediate_size=ffn_dim,
        num_heads=num_heads,
        seed=seed,
        workload_variant=variant,
    )
    golden = generate_golden_reference(**golden_kwargs, weights=weights)
    reference = golden["boundaries"]
    # The order is BLOCK_INPUT_NAMES, read back OUT of the tuple rather than
    # transcribed beside it; `inject` below indexes into the same tuple.
    inputs = layer_inputs(golden, BLOCK_INPUT_NAMES)

    from shared.infra.cache import KernelCache, Profiler

    cache = KernelCache(
        cache_dir=cache_dir, verbose=False, profiler=Profiler(enabled=False)
    )
    compile_block_artifacts(cache, cfg, run_only=True)

    def dispatch(device_inputs, stage_stats, forward_done=None):
        reconfig_baseline = cache.reconfiguration_counts()
        boundaries, vector_rows = run_layer(cache, cfg, device_inputs)
        stages = []
        # The forward is DONE here: every boundary is a host array. The study's
        # clock stops at this instant (operator rule, 2026-08-22); the per-boundary
        # comparison below is verification and runs outside it.
        if forward_done is not None:
            forward_done()
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
        clean = sum(1 for s in stages if s["n_mismatch"] == 0)
        print(f"[{label}] stages: {clean}/{len(stages)} clean")
        print_dispatch_totals(label, vector_rows)
        return [boundaries["output"]], {
            "stages": stages,
            "stages_passed": clean == len(stages),
            "dispatch_vectors": vector_rows,
            # The same latency decomposition the pattern modes report. This is
            # the shared seam `coarse` dispatches through (builders/block.py),
            # which runs every stage on the device -- so host_cpu_ms is empty
            # BY CONSTRUCTION, exactly as in `fused`, and the comparison
            # against the host-mediated modes is the point of recording it.
            "device_ms": sum(
                float(r.get("device_submission_ms", 0.0)) for r in vector_rows
            ),
            "sync_ms": sum(float(r.get("host_sync_ms", 0.0)) for r in vector_rows),
            "host_cpu_ms": {},
            # What THIS dispatch loaded and attached (schema v2's
            # reconfiguration columns). The block path loads its four ELFs on
            # the first dispatch and keeps every context standing, so a warmed
            # dispatch honestly reports 0 -- which is the mode's steady-state
            # per-layer reconfiguration cost, not a missing measurement.
            **reconfiguration_delta(cache, reconfig_baseline),
        }

    record_extra = {
        "variant": variant,
        "causal": causal,
        "golden_seed": seed,
        # Which of the two weight sources this run used. Recorded rather than
        # inferred: a results tree that mixes generated and injected weights is
        # mixing two experiments, and the seed alone no longer identifies the
        # tensors once injection exists.
        "weight_source": "injected" if weights is not None else "generated",
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
    prepared = {
        "inputs": inputs,
        "expected": [reference["output"]],
        # ln1_weight, index 3. See the section header for the measurement that
        # rules out every attention-side target. The decoder keeps the same
        # target: its ln1 is the pre-norm feeding attention, so the
        # perturbation is upstream of every boundary rather than two of ten.
        "inject": (BLOCK_INPUT_NAMES.index("ln1_weight"), (0,)),
        "dispatch": dispatch,
        "record_extra": record_extra,
    }
    # Injected weights only: the delta is DERIVED from the response, because the
    # shipped constant is calibrated against the generated scale and a real
    # weight set moves that calibration. Deferred -- `opcheck.py` resolves it
    # only when it is actually injecting. See the section header above for the
    # measurement, and for the decoder case where the shipped 2.0 stops
    # discriminating entirely.
    hook = fault_delta_hook(
        golden_kwargs,
        weights,
        "ln1_weight",
        (0,),
        reference["output"],
        stage_atol["output"],
    )
    if hook is not None:
        prepared["fault_delta"] = hook
    if causal:
        # The decoder's whole-layer comparison must run at ITS output
        # boundary's atol, not the spec row's encoder-measured one: same
        # tensor, same comparison, one number -- the symmetry the encoder's
        # stage table pins in the other direction. run_mode prefers this key
        # over spec["atol"] when present.
        prepared["atol"] = stage_atol["output"]
    return prepared
