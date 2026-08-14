# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Are a fresh run's numbers consistent with a known-good one?

    python3 study/compare_roots.py --baseline results/a --candidate results/b \\
        --csv coarse.csv --csv runlist.csv

CONTRACT
    Reads the named CSVs from both roots through ``results_io``, keys rows on
    ``(study_case_id, execution_mode, seq_len)``, and reports a tiered verdict:

    - **Identifier columns must match exactly.** A difference there is a
      structural change, not noise, and it fails.
    - **Only mean-based columns gate.** ``avg_latency_ms``,
      ``effective_gflops_per_sec`` and ``avg_power_w``. ``min_*`` and ``max_*``
      are single samples from a noisy distribution and are reported, never
      gating -- one lucky sample must not turn a healthy run red.
    - **Gating is on the MEDIAN and p90 of the per-mode drift distribution**,
      not on any single row's drift. With enough rows a per-row threshold fires
      on a handful of ``offload`` rows even when nothing is wrong.
    - ``offload`` gets a wider band because it genuinely is noisier run to run;
      doc 03 records an XRT version change alone moving it 19-39% at
      ``seq_len >= 4096`` while leaving the others within 0.6%.

    Exit 0 when nothing failed. The manifests are compared first and their
    provenance differences -- git commit, XRT version, platform -- are printed
    as NOTES, because "the latency moved" and "the toolchain moved" are the same
    observation until you look.

THE MEASUREMENT CONDITION IS CHECKED BEFORE ANY DRIFT IS GATED `[2026-08-12]`
    The NPU power mode is a measurement condition and it silently resets: it is
    non-persistent across every reboot and every ``amdxdna`` reload, and at
    ``Default`` this host measures ~15-20x slow (README trap 0, doc 32). A
    Turbo-vs-``Default`` pair therefore drifts ~1500-2000%, which is thirty to
    a hundred times the 15%/35% fail bands below -- so before this guard existed,
    **a power-mode change alone printed ``VERDICT: PROBLEM``**, and it printed it
    in the vocabulary of a code regression. That is the precise failure trap 0's
    closing sentence forbids: re-measure a whole comparison after a pmode change,
    never splice one across it. This module is the tool that splices.

    It cannot be fixed the way the two live gates were. ``require_turbo()`` reads
    the mode NOW; these are two runs recorded EARLIER, and no live query says
    anything about either of them. It needs the condition to have been RECORDED,
    which is why doc 34's M4 (``schema.CONDITION_FIELDS``, written into the
    manifest by ``manifest.py``) is this guard's prerequisite rather than a
    parallel nicety.

    Three outcomes, and the difference between them is whether an operator can
    act on it:

    - **Both roots record a mode and the modes DIFFER -> REFUSE.** A failure,
      and the drift below stops gating: every gating field is reported
      ``[SPLICED]`` instead of ``[GATE]``, so the tool cannot hand back a red
      latency verdict for a condition change. The verdict is still PROBLEM --
      "the comparison proved nothing" must never print green, the same rule
      ``smoke_gate`` and the empty-``--csv`` case already follow -- but the
      failure names the power mode, and the fix it asks for is the one trap 0
      already prescribes: re-walk one side.
    - **Either root does not record a mode -> FLAG, and keep gating.** Every
      root recorded before `[2026-08-12]` is in this state and CANNOT be moved
      out of it: the mode a walk ran at is not recoverable from its files, and
      stamping one after the fact would write an inference into a data field.
      Refusing would make the module unusable against the entire recorded corpus
      -- which is the same reasoning ``port-loop/pmode_guard.py`` used to flag
      rather than refuse the throughput floor's ``unknown``, and it is the one
      place that module's choice transfers here unchanged.
    - **Both record the same mode -> one line saying so, and gate normally.**

    THE ASYMMETRY IS THE POINT, and it is where this differs from the floor
    guard. That guard flags a KNOWN mismatch nowhere, because the floor is one
    shipped driver-owned file and refusing on it would leave "re-seed the floor
    until the gate passes" as the only exit -- the exact pressure the file
    exists to remove. A results root is neither shipped nor driver-owned: the
    exit from a mismatch is to re-measure, which is what should happen anyway.
    So a mismatch refuses here and an unknown flags in both.

    NO OVERRIDE. There is deliberately no ``--allow-pmode-splice``, for
    ``pmode_guard``'s stated reason: nothing added for a guard may become the way
    to defeat it. What a caller who genuinely wants the pmode-independent half
    gets instead is that it still runs -- bytes, counts and every identifier
    field are pmode-independent (trap 0), they are compared on a refused
    comparison exactly as on an accepted one, and a structural change still
    fails on its own terms.

THE BUILD CONDITION IS CHECKED TOO, AND IT FLAGS RATHER THAN REFUSING `[2026-08-12]`
    Queue item 16. The docstring above has always said the manifests are compared
    first so that "the latency moved" and "the toolchain moved" stop being the
    same observation -- and until today that was not true: ``compare_manifests``
    diffed a ``toolchain`` block ``manifest.py`` had never written, so the loop
    iterated an empty key and the toolchain half compared nothing. It is written
    now (``schema.TOOLCHAIN_FIELDS``: XRT version, the mlir-aie and Peano wheel
    pins, and the ``build-xrt``-vs-``install-xrt`` resolution), the existing diff
    prints it field by field, and ``compare_toolchain`` gives it a verdict-level
    reading.

    It WARNS on a mismatch and keeps gating, where the power mode REFUSES. That
    asymmetry is argued in full at ``compare_toolchain``; in one line, a pmode
    mismatch guarantees a false verdict and has an actionable exit, while a
    toolchain mismatch may not invalidate anything and frequently has no exit at
    all, because the toolchain it would re-walk against has been overwritten.

WHAT IS DELIBERATELY DIFFERENT FROM IRON'S COMPARATOR
    - **No file list is baked in.** iron hardcodes its ``RESULT_CSVS`` and
      ``DERIVED_CSVS`` tuples, which silently skip a file a run stopped
      producing. Here the caller names them, exactly as ``smoke_gate`` and
      ``manifest`` take ``--expect``: a comparison of nothing must not be able
      to pass.
    - **No ``pattern_label`` rename table.** iron carries an intended-difference
      map because it renamed ``Hybrid`` to ``Coarse runlist`` mid-study.
      Convention 7 settled that naming once, in the schema module, and the port
      has no back-compatibility obligation -- so there is nothing to except, and
      an unexpected identifier difference is unambiguously a failure.
    - **Rows are read through the schema.** A CSV whose header is not this
      schema fails here rather than being compared column-by-name against
      something it is not. Reading an iron tree is ``iron_adapter``'s job.

FOOTGUNS
    - **A row present on one side only is a FAILURE, not a skip.** That is the
      shape a case that stopped building takes, and it is the thing most worth
      catching.
    - **The tolerances are per CSV execution_mode**, read off the rows
      themselves. A CSV whose rows carry several modes gets several drift
      distributions, which is intended -- a ladder CSV per mode gets one.
    - **A drift distribution over one row is a median over one number.** It is
      still gated, and it should be: one rung moving 40% is worth failing on.
      But do not read "median" as a robust statistic at that count; ``n`` is
      printed on every line for exactly that reason.
    - **Power tolerances are wide (15% / 50%) and the columns are SoC rails.**
      Doc 09's finding that no sensor here measures the NPU applies to every
      power comparison this makes; see ``power.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import results_io  # noqa: E402
import schema  # noqa: E402

#: What makes two rows the same measurement. Doc 03's key.
KEY_FIELDS = ("study_case_id", "execution_mode", "seq_len")

#: Must match exactly. Shape, configuration, structural counts and outcome:
#: everything whose change means the runs are not comparable rather than that
#: the machine was busier. The dispatch vector is HERE and not in the drift
#: block on purpose -- those are counts, and a mode whose submission count moved
#: is a different mode, not a noisier one.
IDENTIFIER_FIELDS = (
    "schema_version",
    "study_case_id",
    "study_case_label",
    "workload_variant",
    "backend",
    "execution_mode",
    "attention_path",
    "seq_len",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "attention_head_size",
    "batch_size",
    "dtype",
    "use_bias",
    "weights_source",
    "warmup_runs",
    "runs_per_sample",
    "measured_inference_count",
    "latency_sample_count",
    "host_submissions_per_layer",
    "air_launches_per_elf",
    "herd_launches",
    "sync_boundaries",
    "bytes_transferred",
    "context_loads",
    "kernel_attaches",
    "power_backend",
    "npu_dispatch_count",
    "npu_unique_xclbin_count",
    "validation_error_count",
    "run_status",
)

#: Reported as drift. Only GATING_FIELDS can fail the run.
LATENCY_FIELDS = (
    "avg_latency_ms",
    "min_latency_ms",
    "max_latency_ms",
    "device_ms",
    "sync_ms",
    "host_cpu_ms",
    "effective_gflops_per_sec",
)
POWER_FIELDS = (
    "avg_power_w",
    "min_power_w",
    "max_power_w",
    "power_std_w",
    "raw_avg_power_w",
)
GATING_FIELDS = ("avg_latency_ms", "effective_gflops_per_sec", "avg_power_w")

#: (warn, fail) percent, applied to the median AND the p90 of the drift spread.
#: Doc 03's table.
LATENCY_TOLERANCE = {
    "hybrid": (5.0, 15.0),
    "runlist": (5.0, 15.0),
    "fused_elf": (5.0, 15.0),
    "offload": (20.0, 35.0),
}
DEFAULT_LATENCY_TOLERANCE = (10.0, 25.0)
POWER_TOLERANCE = (15.0, 50.0)


class Report:
    """Report lines plus warning and failure counts."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.warnings = 0
        self.failures = 0
        #: Reasons the comparison itself is invalid, as opposed to results
        #: within it being wrong. Counted as failures AND surfaced verbatim
        #: above the verdict, because a reader who sees only `PROBLEM` will
        #: read it as the code having regressed -- which is the entire defect
        #: the pmode guard exists to stop.
        self.refusals: list[str] = []

    def say(self, text: str = "") -> None:
        self.lines.append(text)

    def warn(self, text: str) -> None:
        self.warnings += 1
        self.lines.append(f"  WARN  {text}")

    def fail(self, text: str) -> None:
        self.failures += 1
        self.lines.append(f"  FAIL  {text}")

    def refuse(self, headline: str, detail: str) -> None:
        self.refusals.append(headline)
        self.fail(f"{headline} {detail}")

    def render(self) -> str:
        return "\n".join(self.lines)


def _number(value: object) -> float | None:
    """A float, or ``None`` for empty, unset or NaN. Never raises."""
    text = str("" if value is None else value).strip()
    if text in ("", "None", "nan", "NaN"):
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed if parsed == parsed else None


def relative_percent(baseline: float, candidate: float) -> float | None:
    """|drift| as a percentage of the baseline.

    ``None`` when both are zero -- no drift and no denominator -- and ``inf``
    when only the baseline is, which is a real and reportable change rather than
    a division to be swallowed.
    """
    if baseline == 0.0:
        return None if candidate == 0.0 else float("inf")
    return abs(candidate - baseline) / abs(baseline) * 100.0


def signed_percent(baseline: float, candidate: float) -> float | None:
    """Which WAY it moved. Reported beside the magnitude, never gated on."""
    if baseline == 0.0:
        return None
    return (candidate - baseline) / abs(baseline) * 100.0


def percentile_90(values: list[float]) -> float:
    """p90 by nearest rank, so a short list still has a defined value."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))
    return ordered[index]


def row_key(row: dict) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")).strip() for field in KEY_FIELDS)


#: The manifest filenames a root may carry, newest writer first.
#:
#: `[2026-08-14]` THIS MODULE READ THE WRONG NAME, and it made items 15 and 16
#: inert on every root the Phase G runner produces. ``run_profile.py`` writes
#: ``results_manifest.json`` (its ``MANIFEST_NAME``) and this module looked only
#: for ``manifest.json``, so a comparison of two real walks reported
#: ``npu_power_mode: unknown (absent)`` and warned that the mode was "NOT
#: recoverable from its files" -- while a fully populated manifest recording
#: ``turbo`` (observed), plus the whole toolchain block, sat in the same
#: directory under the other name.
#:
#: Found by running the two-walk comparison for the first time (Phase G's "two
#: walks into two roots", which nothing had done). It is the same shape as item
#: 16, one level over: that item found the toolchain diff comparing NOTHING
#: because the writer never wrote the block; this is the reader and the writer
#: disagreeing about the FILENAME, which no test caught because every test built
#: its own fixture and named it whatever it read.
MANIFEST_NAMES = ("results_manifest.json", "manifest.json")


def manifest_path(root: Path) -> Path | None:
    """The first manifest name this root actually has, or None."""
    for name in MANIFEST_NAMES:
        path = root / name
        if path.exists():
            return path
    return None


def load_manifest(root: Path) -> tuple[dict | None, str | None]:
    """``(payload, why_not)`` for one root's manifest. Never raises.

    ``why_not`` is a phrase, not a sentence, so both callers can embed it in
    their own vocabulary: the provenance diff wants "there is nothing to diff"
    and the condition guard wants "the measurement condition cannot be read".
    """
    path = manifest_path(root)
    if path is None:
        return None, f"no {' or '.join(MANIFEST_NAMES)} in the root"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return None, f"its {path.name} could not be read ({e})"
    if not isinstance(payload, dict):
        return None, f"its {path.name} is not a JSON object"
    return payload, None


def compare_conditions(report: Report, baseline: Path, candidate: Path) -> bool:
    """The measurement-condition guard. Returns whether drift may GATE.

    See the module docstring for the three outcomes and for why a mismatch
    refuses while an unknown flags. This runs whatever the manifests look like
    -- in particular it does NOT sit behind the "manifest missing on one side"
    early-out the provenance diff uses, because the roots this study actually
    compares (the recorded ladder walks) carry no manifest at all, and a guard
    that skipped there would be a guard that never ran on real evidence.
    """
    report.say("\n=== measurement condition ===")
    left_payload, left_why = load_manifest(baseline)
    right_payload, right_why = load_manifest(candidate)

    left = schema.conditions_from_manifest(left_payload)
    right = schema.conditions_from_manifest(right_payload)
    left_mode = schema.normalise_power_mode(left["npu_power_mode"])
    right_mode = schema.normalise_power_mode(right["npu_power_mode"])

    report.say(
        f"  npu_power_mode: baseline={left_mode} "
        f"({left['npu_power_mode_source']})  candidate={right_mode} "
        f"({right['npu_power_mode_source']})"
    )

    unknown_sides = [
        (label, block, why)
        for label, mode, block, why in (
            ("baseline", left_mode, left, left_why),
            ("candidate", right_mode, right, right_why),
        )
        if mode == schema.UNKNOWN_CONDITION
    ]

    if unknown_sides:
        for label, block, why in unknown_sides:
            reason = why or block["npu_power_mode_detail"] or "no reason recorded"
            report.warn(
                f"the {label} root does not record the NPU power mode it was "
                f"measured at -- {reason}"
            )
        report.say(
            "        This comparison is UNCONDITIONED on the axis that costs "
            "15-20x on this host: a large\n"
            "        latency drift below may be a power-mode change rather than "
            "a code change, and nothing\n"
            "        here can tell you which (README trap 0). Flagged rather "
            "than refused because the mode\n"
            "        a recorded run used is NOT recoverable from its files -- "
            "re-walk to condition it, and\n"
            "        never stamp a mode you did not observe."
        )
        # Gating survives an unknown on purpose: refusing would make every
        # recorded root uncomparable, and a flagged comparison that still
        # catches a real regression beats a refused one that catches nothing.
        return True

    if left_mode != right_mode:
        report.refuse(
            f"COMPARISON REFUSED: the two roots were measured at different NPU "
            f"power modes (baseline={left_mode}, candidate={right_mode}).",
            "They are not a comparison. On this host a dispatch-bound "
            "measurement differs ~15-20x across that boundary -- ~1500-2000% "
            "drift against fail bands of 15-35% -- so every latency and power "
            "number below would fail on the power mode while reading as a code "
            "regression. README trap 0: re-measure a whole comparison after a "
            "pmode change, never splice across one. Re-walk one side at the "
            "other's mode. The identifier comparison below is pmode-independent "
            "and still holds.",
        )
        return False

    report.say(
        f"  both roots were measured at `{left_mode}` -- the comparison is "
        f"conditioned and the drift below may be read as a code difference"
    )
    return True


def compare_toolchain(report: Report, baseline: Path, candidate: Path) -> None:
    """The build-condition guard. FLAGS; never withdraws the gate.

    `[2026-08-12]`, queue item 16. Its prerequisite was that anything wrote the
    block at all: ``compare_manifests`` below has diffed ``manifest["toolchain"]``
    since it was written and ``manifest.py`` never wrote one, so the toolchain
    half of every root comparison compared NOTHING. That half is now populated by
    ``manifest.observe_toolchain`` and prints as NOTE lines; this function is the
    verdict-level reading of the same fact.

    WHY THIS FLAGS WHERE THE POWER MODE REFUSES. The split is not inherited, and
    it is the decision worth arguing. Four reasons, and the first two are
    decisive.

      1. MAGNITUDE. A pmode mismatch drifts ~1500-2000% against fail bands of
         15-35% -- a 30-100x overshoot, so a mismatch GUARANTEES the latency
         verdict is false, and withdrawing it costs nothing that was true. The
         recorded toolchain effect is doc 03's XRT change: ``offload`` 19-39%
         against its own 35% band, the other three modes within 0.6% against
         15%. That straddles the band rather than dwarfing it. A guard that
         withdrew the verdict here would discard true regressions -- most fields,
         most of the time, are still inside tolerance -- to suppress a false one
         that may not be there.
      2. THE EXIT IS OFTEN NOT AVAILABLE. Item 15's rule for choosing is whether
         the operator can act: it refuses a pmode mismatch because "re-walk one
         side" is actionable, and ``pmode_guard`` flags the throughput floor
         because refusing left "re-seed until it passes" as the only exit. A
         toolchain is INSTALLED, singular and destructive -- doc 15's four-layer
         upgrade replaces LLVM, mlir-aie, the stale tablegen output and the
         bindings bottom-up, and `install-xrt` overwrites. A baseline recorded
         against a toolchain no longer on the host CANNOT be re-walked at that
         toolchain, so the refusal's only real exit is "delete the baseline".
         That is item 15's own argument against refusing, transferred.
      3. THE TOOLCHAIN IS OFTEN THE INDEPENDENT VARIABLE. The pmode is never what
         a study is measuring; the toolchain frequently is. The README's own
         provenance check -- ``runlist``'s non-attention total byte-identical to
         the figure its gate pinned BEFORE the rebuild -- is a comparison taken
         deliberately across a toolchain change. Refusing would drop exactly the
         latency fields to ``[SPLICED]`` for the operator who ran it to see them.
      4. IT IS ALREADY HOW PROVENANCE IS TREATED HERE. ``compare_manifests``
         diffs the git sha as a NOTE and does not refuse, because a code change
         is the normal reason to run a comparison at all. The toolchain sits on
         the same axis one layer down.

    So a mismatch WARNS, names the fields that moved, and gating continues. What
    a reader gets is the one thing they did not have before: when a latency moves
    and the toolchain moved with it, the tool now says so instead of leaving
    "the latency moved" and "the toolchain moved" indistinguishable -- which is
    what the module docstring has always claimed this section was for.

    An ABSENT or UNKNOWN side also warns, for ``compare_conditions``' reason
    exactly: a recorded root cannot be stamped after the fact, and refusing there
    would make the module useless against the whole recorded corpus. Both roots
    recording nothing is NOT agreement, and is not reported as any.

    Returns None on purpose -- there is no ``may_gate`` to return, because this
    guard never withdraws the verdict.
    """
    report.say("\n=== build condition (toolchain) ===")
    left_payload, left_why = load_manifest(baseline)
    right_payload, right_why = load_manifest(candidate)
    left = schema.toolchain_from_manifest(left_payload)
    right = schema.toolchain_from_manifest(right_payload)

    for name in schema.TOOLCHAIN_IDENTITY_FIELDNAMES:
        report.say(f"  {name}: baseline={left[name]}  candidate={right[name]}")

    blind = [
        (label, block, why)
        for label, block, why in (
            ("baseline", left, left_why),
            ("candidate", right, right_why),
        )
        if any(
            block[name] == schema.UNKNOWN_CONDITION
            for name in schema.TOOLCHAIN_IDENTITY_FIELDNAMES
        )
    ]

    if blind:
        for label, block, why in blind:
            missing = [
                name
                for name in schema.TOOLCHAIN_IDENTITY_FIELDNAMES
                if block[name] == schema.UNKNOWN_CONDITION
            ]
            reason = why or block["toolchain_detail"] or "no reason recorded"
            report.warn(
                f"the {label} root does not record {', '.join(missing)} -- " f"{reason}"
            )
        report.say(
            "        This comparison is UNCONDITIONED on the toolchain: doc 03 "
            "records an XRT version\n"
            "        change ALONE moving `offload` 19-39% at seq_len >= 4096, "
            "which is inside its own fail\n"
            "        band, so a drift below may be a toolchain change rather "
            "than a code change and nothing\n"
            "        here can tell you which. Flagged rather than refused, and "
            "still gating: a recorded\n"
            "        root's toolchain is NOT recoverable from its files and must "
            "not be stamped after the\n"
            "        fact. Two roots that both record nothing are not two roots "
            "that agree."
        )

    differences = schema.toolchain_differences(left, right)
    if differences:
        for name, before, after in differences:
            report.warn(
                f"the two roots were built against different {name} "
                f"(baseline={before}, candidate={after})"
            )
        report.say(
            "        The drift below is NOT conditioned on the toolchain and is "
            "still gated -- see this\n"
            "        function's docstring for why a toolchain mismatch flags "
            "where a power-mode mismatch\n"
            "        refuses. Read a red latency verdict here as `code OR "
            "toolchain`, and settle it by\n"
            "        re-walking one side at the other's toolchain if that "
            "toolchain still exists."
        )
    elif not blind:
        report.say(
            "  both roots were built against the same toolchain -- the "
            "comparison is conditioned on this axis too"
        )


def compare_manifests(report: Report, baseline: Path, candidate: Path) -> None:
    """Provenance first: a toolchain change explains drift that code does not."""
    left, right = manifest_path(baseline), manifest_path(candidate)
    seen = sorted({p.name for p in (left, right) if p is not None})
    report.say(f"\n=== {' / '.join(seen) if seen else MANIFEST_NAMES[0]} ===")
    if left is None or right is None:
        report.say("  SKIP (missing on one side)")
        return
    a, left_why = load_manifest(baseline)
    b, right_why = load_manifest(candidate)
    if a is None or b is None:
        # Present but unparsable is a FAILURE, not a skip -- unchanged from
        # before the guard landed. Only ABSENT is skippable here; the condition
        # section above treats absent and unparsable alike, because for its
        # question both mean the same thing.
        report.fail(f"a manifest could not be read: {left_why or right_why}")
        return

    report.say(
        f"  complete: baseline={a.get('complete')} candidate={b.get('complete')}"
    )
    if not b.get("complete"):
        report.fail(
            "candidate manifest reports complete=false -- see its "
            "incomplete_reasons before reading any drift below"
        )
    for label, block in (("git", "git"), ("toolchain", "toolchain")):
        left_block = a.get(block) or {}
        right_block = b.get(block) or {}
        for key in sorted(set(left_block) | set(right_block)):
            before, after = left_block.get(key), right_block.get(key)
            if before != after:
                report.say(f"  NOTE  {label}.{key}: {before} -> {after}")
    if (a.get("git") or {}).get("dirty") or (b.get("git") or {}).get("dirty"):
        report.say("  NOTE  a run was measured from a dirty tree")


def compare_csv(
    report: Report,
    rel: str,
    baseline: Path,
    candidate: Path,
    may_gate: bool = True,
) -> None:
    """One CSV, both sides. Appends to ``report``; never raises for bad data.

    ``may_gate=False`` when the measurement condition was refused: the drift is
    still COMPUTED and PRINTED -- a reader who has just been told the pmode
    moved wants to see how far the numbers moved with it -- but no gating field
    may warn or fail on it. Suppressing the computation instead would hide the
    evidence for the refusal; suppressing only the verdict is what keeps the
    tool from reporting a condition change as a code regression.
    """
    report.say(f"\n=== {rel} ===")
    left, right = baseline / rel, candidate / rel
    if not left.exists():
        report.fail(
            "absent from the baseline. It was named as a file to compare, so "
            "its absence is a failure rather than something to skip"
        )
        return
    if not right.exists():
        report.fail("missing in candidate")
        return
    try:
        baseline_rows = results_io.read_rows(left)
        candidate_rows = results_io.read_rows(right)
    except Exception as e:
        report.fail(f"unreadable as the current schema -- {e}")
        return

    by_key_left = {row_key(r): r for r in baseline_rows}
    by_key_right = {row_key(r): r for r in candidate_rows}
    only_left = sorted(set(by_key_left) - set(by_key_right))
    only_right = sorted(set(by_key_right) - set(by_key_left))
    shared = sorted(set(by_key_left) & set(by_key_right))
    report.say(
        f"  rows: baseline={len(baseline_rows)} candidate={len(candidate_rows)} "
        f"matched={len(shared)}"
    )
    for key in only_left[:10]:
        report.fail(f"row only in baseline: {key}")
    for key in only_right[:10]:
        report.fail(f"row only in candidate: {key}")

    mismatches = 0
    drift: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    signed: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for key in shared:
        before_row, after_row = by_key_left[key], by_key_right[key]
        mode = str(before_row.get("execution_mode") or "unknown")

        for field in IDENTIFIER_FIELDS:
            before = str(before_row.get(field, "")).strip()
            after = str(after_row.get(field, "")).strip()
            if before == after:
                continue
            mismatches += 1
            if mismatches <= 10:
                report.fail(f"{key} {field}: baseline={before!r} candidate={after!r}")

        for field in LATENCY_FIELDS + POWER_FIELDS:
            before = _number(before_row.get(field))
            after = _number(after_row.get(field))
            if before is None or after is None:
                continue
            percent = relative_percent(before, after)
            if percent is None:
                continue
            drift[mode][field].append(percent)
            direction = signed_percent(before, after)
            if direction is not None:
                signed[mode][field].append(direction)

    report.say(f"  identifier mismatches: {mismatches}")
    if drift:
        report_drift(report, drift, signed, may_gate=may_gate)


def report_drift(
    report: Report,
    drift: dict[str, dict[str, list[float]]],
    signed: dict[str, dict[str, list[float]]],
    may_gate: bool = True,
) -> None:
    report.say("  drift by execution_mode (median / p90 / max, % relative):")
    for mode in sorted(drift):
        warn_at_latency, fail_at_latency = LATENCY_TOLERANCE.get(
            mode, DEFAULT_LATENCY_TOLERANCE
        )
        for field in sorted(drift[mode]):
            values = drift[mode][field]
            if not values:
                continue
            median = statistics.median(values)
            p90 = percentile_90(values)
            largest = max(values)
            gating = field in GATING_FIELDS
            warn_at, fail_at = (
                POWER_TOLERANCE
                if field in POWER_FIELDS
                else (warn_at_latency, fail_at_latency)
            )
            directions = signed[mode].get(field) or []
            direction = statistics.median(directions) if directions else 0.0
            if not gating:
                tag = "info"
            elif may_gate:
                tag = "GATE"
            else:
                # A gating field whose verdict was withdrawn by the condition
                # guard. Named distinctly from `info` so nobody reads the
                # absence of a FAIL as the number having been fine.
                tag = "SPLICED"
            # Padded as a whole so the columns stay aligned across a tag that is
            # not four characters, while `[GATE]` and `[info]` stay literal.
            report.say(
                f"    {f'[{tag}]':<9} {mode:>9} {field:<28} "
                f"med={median:7.2f} p90={p90:7.2f} max={largest:8.2f} "
                f"signed_med={direction:+7.2f} n={len(values)}"
            )
            if not gating or not may_gate:
                continue
            if median > fail_at or p90 > fail_at:
                report.fail(
                    f"{mode}/{field}: median={median:.2f}% p90={p90:.2f}% "
                    f"exceeds the fail threshold {fail_at}%"
                )
            elif median > warn_at or p90 > warn_at:
                report.warn(
                    f"{mode}/{field}: median={median:.2f}% p90={p90:.2f}% "
                    f"exceeds the warn threshold {warn_at}%"
                )


def compare_roots(baseline: Path, candidate: Path, csvs: list[str]) -> Report:
    """The whole comparison. An empty ``csvs`` is itself a failure."""
    report = Report()
    report.say(f"baseline : {baseline}")
    report.say(f"candidate: {candidate}")
    if not csvs:
        report.fail(
            "no CSVs were named, so this comparison proved nothing. Name the "
            "files both runs were supposed to produce."
        )
        report.say("")
        report.say(f"warnings: {report.warnings}   failures: {report.failures}")
        report.say("VERDICT: PROBLEM")
        return report

    compare_manifests(report, baseline, candidate)
    # BEFORE any CSV: whether these two roots may be compared at all is a
    # different question from whether their numbers agree, and it has to be
    # settled first or the second question's answer is meaningless.
    may_gate = compare_conditions(report, baseline, candidate)
    # Beside it, and also before any CSV: the toolchain is the other half of the
    # condition. It FLAGS rather than refusing, so it returns nothing and cannot
    # move `may_gate` -- the argument is in its docstring.
    compare_toolchain(report, baseline, candidate)
    for rel in csvs:
        compare_csv(report, rel, baseline, candidate, may_gate=may_gate)

    report.say("")
    for headline in report.refusals:
        report.say(headline)
    report.say(f"warnings: {report.warnings}   failures: {report.failures}")
    report.say(f"VERDICT: {'OK' if report.failures == 0 else 'PROBLEM'}")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument(
        "--csv",
        action="append",
        default=[],
        dest="csvs",
        metavar="REL/PATH.csv",
        help="a results CSV present in both roots, relative to each. "
        "Repeatable. Required: with none, the comparison checks nothing.",
    )
    args = ap.parse_args(argv)

    report = compare_roots(
        args.baseline.expanduser().resolve(),
        args.candidate.expanduser().resolve(),
        args.csvs,
    )
    print(report.render())
    return 0 if report.failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
