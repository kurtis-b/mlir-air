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

    def say(self, text: str = "") -> None:
        self.lines.append(text)

    def warn(self, text: str) -> None:
        self.warnings += 1
        self.lines.append(f"  WARN  {text}")

    def fail(self, text: str) -> None:
        self.failures += 1
        self.lines.append(f"  FAIL  {text}")

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


def compare_manifests(report: Report, baseline: Path, candidate: Path) -> None:
    """Provenance first: a toolchain change explains drift that code does not."""
    report.say("\n=== manifest.json ===")
    left, right = baseline / "manifest.json", candidate / "manifest.json"
    if not left.exists() or not right.exists():
        report.say("  SKIP (missing on one side)")
        return
    try:
        a = json.loads(left.read_text(encoding="utf-8"))
        b = json.loads(right.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        report.fail(f"a manifest could not be read: {e}")
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


def compare_csv(report: Report, rel: str, baseline: Path, candidate: Path) -> None:
    """One CSV, both sides. Appends to ``report``; never raises for bad data."""
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
        report_drift(report, drift, signed)


def report_drift(
    report: Report,
    drift: dict[str, dict[str, list[float]]],
    signed: dict[str, dict[str, list[float]]],
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
            report.say(
                f"    [{'GATE' if gating else 'info'}] {mode:>9} {field:<28} "
                f"med={median:7.2f} p90={p90:7.2f} max={largest:8.2f} "
                f"signed_med={direction:+7.2f} n={len(values)}"
            )
            if not gating:
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
    for rel in csvs:
        compare_csv(report, rel, baseline, candidate)

    report.say("")
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
