# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only checks on the run-to-run comparator.

    python3 study/test_compare_roots.py

Every test builds two small results roots on disk and asserts the VERDICT, not
just the arithmetic: the module's job is to decide, and a tiering rule that
computes the right percentage and then fails the wrong way is still wrong.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import compare_roots  # noqa: E402
import results_io  # noqa: E402
import schema  # noqa: E402


def _row(mode="hybrid", seq=1024, latency=100.0, **overrides):
    row = schema.empty_row("results")
    row.update(
        {
            "study_id": "s",
            "study_case_id": "baseline_768",
            "study_case_label": "BERT-Base",
            "workload_variant": "encoder_bert",
            "backend": "xrt",
            "execution_mode": mode,
            "attention_path": "device",
            "seq_len": seq,
            "hidden_size": 768,
            "batch_size": 1,
            "dtype": "bf16",
            "warmup_runs": 1,
            "runs_per_sample": 1,
            "latency_sample_count": 3,
            "avg_latency_ms": latency,
            "min_latency_ms": latency * 0.9,
            "max_latency_ms": latency * 1.1,
            "host_submissions_per_layer": 4,
            "run_status": "passed",
            "failure_message": "",
        }
    )
    row.update(overrides)
    return row


def _roots(baseline_rows, candidate_rows, *, name="coarse.csv", manifests=None):
    """Two results roots on disk. Returns a TemporaryDirectory to keep alive."""
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    (root / "a").mkdir()
    (root / "b").mkdir()
    results_io.write_rows(root / "a" / name, baseline_rows)
    results_io.write_rows(root / "b" / name, candidate_rows)
    if manifests:
        for side, payload in zip(("a", "b"), manifests):
            (root / side / "manifest.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
    return tmp, root / "a", root / "b"


def test_relative_percent_handles_a_zero_baseline_without_dividing():
    assert compare_roots.relative_percent(0.0, 0.0) is None
    assert compare_roots.relative_percent(0.0, 1.0) == float("inf")
    assert compare_roots.relative_percent(100.0, 110.0) == 10.0
    assert compare_roots.relative_percent(100.0, 90.0) == 10.0


def test_signed_percent_keeps_the_direction_relative_percent_discards():
    assert compare_roots.signed_percent(100.0, 90.0) == -10.0
    assert compare_roots.signed_percent(100.0, 110.0) == 10.0
    assert compare_roots.signed_percent(0.0, 5.0) is None


def test_percentile_90_is_defined_for_a_short_list():
    assert compare_roots.percentile_90([]) == 0.0
    assert compare_roots.percentile_90([7.0]) == 7.0
    assert compare_roots.percentile_90([1.0, 2.0, 3.0]) == 3.0


def test_number_never_raises_on_junk():
    for junk in (None, "", "None", "nan", "NaN", "not-a-number", []):
        assert compare_roots._number(junk) is None
    assert compare_roots._number("2.5") == 2.5


def test_identical_roots_pass():
    tmp, a, b = _roots([_row()], [_row()])
    with tmp:
        report = compare_roots.compare_roots(a, b, ["coarse.csv"])
    assert report.failures == 0, report.render()
    assert "VERDICT: OK" in report.render()


def test_naming_no_csvs_is_a_failure_not_a_pass():
    """The same rule as smoke_gate: a check of nothing must not be cheerful."""
    tmp, a, b = _roots([_row()], [_row()])
    with tmp:
        report = compare_roots.compare_roots(a, b, [])
    assert report.failures == 1
    assert "proved nothing" in report.render()


def test_a_small_latency_drift_passes_for_a_quiet_mode():
    tmp, a, b = _roots([_row(latency=100.0)], [_row(latency=103.0)])
    with tmp:
        report = compare_roots.compare_roots(a, b, ["coarse.csv"])
    assert report.failures == 0, report.render()


def test_a_large_latency_drift_fails_for_a_quiet_mode():
    tmp, a, b = _roots([_row(latency=100.0)], [_row(latency=130.0)])
    with tmp:
        report = compare_roots.compare_roots(a, b, ["coarse.csv"])
    assert report.failures >= 1
    assert "avg_latency_ms" in report.render()


def test_the_same_drift_passes_for_offload_and_fails_for_runlist():
    """The per-mode band, verified as a DIFFERENCE rather than in one direction."""
    tmp, a, b = _roots(
        [_row(mode="offload", latency=100.0)],
        [_row(mode="offload", latency=125.0)],
        name="offload.csv",
    )
    with tmp:
        loose = compare_roots.compare_roots(a, b, ["offload.csv"])
    tmp, a, b = _roots(
        [_row(mode="runlist", latency=100.0)],
        [_row(mode="runlist", latency=125.0)],
        name="runlist.csv",
    )
    with tmp:
        tight = compare_roots.compare_roots(a, b, ["runlist.csv"])
    assert loose.failures == 0, loose.render()
    assert tight.failures >= 1, tight.render()


def test_min_and_max_latency_are_reported_and_never_gate():
    """A single unlucky sample must not turn a healthy run red."""
    tmp, a, b = _roots(
        [_row(latency=100.0)],
        [_row(latency=100.0, min_latency_ms=10.0, max_latency_ms=900.0)],
    )
    with tmp:
        report = compare_roots.compare_roots(a, b, ["coarse.csv"])
    text = report.render()
    assert report.failures == 0, text
    assert "min_latency_ms" in text
    assert "[info]" in text


def test_a_moved_structural_count_is_a_failure_not_drift():
    """A mode whose submission count changed is a different mode."""
    tmp, a, b = _roots([_row()], [_row(host_submissions_per_layer=6)])
    with tmp:
        report = compare_roots.compare_roots(a, b, ["coarse.csv"])
    assert report.failures >= 1
    assert "host_submissions_per_layer" in report.render()


def test_a_row_present_on_one_side_only_fails():
    tmp, a, b = _roots([_row(seq=512), _row(seq=1024)], [_row(seq=1024)])
    with tmp:
        report = compare_roots.compare_roots(a, b, ["coarse.csv"])
    assert report.failures >= 1
    assert "only in baseline" in report.render()


def test_a_csv_absent_from_the_baseline_fails_rather_than_skipping():
    """iron skips this case; a file a run stopped producing is the thing to catch."""
    tmp, a, b = _roots([_row()], [_row()])
    with tmp:
        report = compare_roots.compare_roots(a, b, ["nope.csv"])
    assert report.failures >= 1
    assert "absent from the baseline" in report.render()


def test_a_status_flip_to_failed_is_an_identifier_mismatch():
    tmp, a, b = _roots([_row()], [_row(run_status="failed")])
    with tmp:
        report = compare_roots.compare_roots(a, b, ["coarse.csv"])
    assert report.failures >= 1
    assert "run_status" in report.render()


def test_an_unreadable_csv_fails_rather_than_being_skipped():
    tmp, a, b = _roots([_row()], [_row()])
    with tmp:
        (b / "coarse.csv").write_text("not,a,schema\n1,2,3\n", encoding="utf-8")
        report = compare_roots.compare_roots(a, b, ["coarse.csv"])
    assert report.failures >= 1
    assert "unreadable as the current schema" in report.render()


def test_provenance_differences_are_notes_not_failures():
    manifests = (
        {"complete": True, "git": {"sha": "aaa", "dirty": False}, "toolchain": {}},
        {"complete": True, "git": {"sha": "bbb", "dirty": False}, "toolchain": {}},
    )
    tmp, a, b = _roots([_row()], [_row()], manifests=manifests)
    with tmp:
        report = compare_roots.compare_roots(a, b, ["coarse.csv"])
    text = report.render()
    assert report.failures == 0, text
    assert "NOTE  git.sha: aaa -> bbb" in text


def test_an_incomplete_candidate_manifest_fails():
    manifests = (
        {"complete": True, "git": {}, "toolchain": {}},
        {"complete": False, "git": {}, "toolchain": {}},
    )
    tmp, a, b = _roots([_row()], [_row()], manifests=manifests)
    with tmp:
        report = compare_roots.compare_roots(a, b, ["coarse.csv"])
    assert report.failures >= 1
    assert "complete=false" in report.render()


def test_there_is_no_intended_rename_exception_table():
    """Convention 7 settled the naming once; nothing needs excepting."""
    assert not hasattr(compare_roots, "RENAMED_VALUES")
    assert "pattern_label" not in compare_roots.IDENTIFIER_FIELDS


def test_every_gating_field_is_also_reported():
    reported = set(compare_roots.LATENCY_FIELDS) | set(compare_roots.POWER_FIELDS)
    assert set(compare_roots.GATING_FIELDS) <= reported


def test_every_compared_field_is_a_real_schema_column():
    names = {f.name for f in schema.fields_for("results")}
    for field in (
        compare_roots.IDENTIFIER_FIELDS
        + compare_roots.LATENCY_FIELDS
        + compare_roots.POWER_FIELDS
        + compare_roots.KEY_FIELDS
    ):
        assert field in names, field


def test_every_csv_execution_mode_has_a_tolerance_band():
    """A mode with no band silently gets the default; that must be a choice."""
    for mode in schema.EXECUTION_MODES:
        assert mode in compare_roots.LATENCY_TOLERANCE, mode


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"compare-roots tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
