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

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_HERE)  # programming_examples/
for _p in (_PROJ_ROOT, os.path.join(_PROJ_ROOT, "llms"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from builders.block import (  # noqa: E402
    BLOCK_BOUNDARIES,
    BLOCK_INPUT_NAMES,
    block_config,
    compile_block_artifacts,
    describe_block,
    run_block,
)
from opcheck_prepare import _spec_digest  # noqa: E402
from pattern.reference import (  # noqa: E402
    fuse_qkv_weight,
    generate_golden_reference,
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
    well as the clean one, then validated and summed by
    ``dispatch_vector_totals``. The driver requires the two runs' summed
    totals EQUAL, and the lit gate pins the printed totals to one set of
    literals in both halves; a "skip instrumentation when injecting" shortcut
    fails both, not dodges them.
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
        reconfig_baseline = cache.reconfiguration_counts()
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
