# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only checks on the results manifest.

    python3 study/test_manifest.py

The load-bearing one is ``test_all_failed_tree_is_not_complete``: iron's manifest
calls a tree complete when the files exist, so a run whose every measurement
failed reports complete. Phase G's gate is "a full profile run completes with a
complete results_manifest.json", so that definition would make the gate unable
to fail -- the same shape as the smoke test doc 09 records.

Its sibling is ``test_one_row_where_nine_were_expected_is_not_complete``, which
pins the SECOND version of that hole: "at least one passed row" is still a file
rule wearing a row's clothes, so a truncated walk reported complete until the
row counts landed. The pair is deliberate -- the first shows the file rule is
not enough, the second shows the passed-row rule is not either.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import manifest  # noqa: E402
import results_io  # noqa: E402
import schema  # noqa: E402


def _row(status, msg=None):
    row = schema.empty_row("results")
    row["execution_mode"] = "hybrid"
    row["run_status"] = status
    row["failure_message"] = msg
    return row


def test_complete_tree_is_complete():
    with tempfile.TemporaryDirectory() as d:
        results_io.write_rows(Path(d) / "m.csv", [_row("passed")])
        m = manifest.build_manifest(d, ["m.csv"])
        assert m["complete"] is True
        assert m["incomplete_reasons"] == []
        assert m["missing_files"] == []


def test_all_failed_tree_is_not_complete():
    """iron would call this complete: the file exists. It measured nothing."""
    with tempfile.TemporaryDirectory() as d:
        results_io.write_rows(
            Path(d) / "m.csv", [_row("failed", "ERT_CMD_STATE_TIMEOUT")]
        )
        m = manifest.build_manifest(d, ["m.csv"])
        assert m["complete"] is False
        assert m["missing_files"] == []  # the file IS present
        assert any("none with run_status=passed" in r for r in m["incomplete_reasons"])
        assert any("ERT_CMD_STATE_TIMEOUT" in r for r in m["incomplete_reasons"])


def test_missing_file_is_reported_twice_over():
    with tempfile.TemporaryDirectory() as d:
        m = manifest.build_manifest(d, ["gone.csv"])
        assert m["complete"] is False
        assert m["missing_files"] == ["gone.csv"]
        assert m["incomplete_reasons"]


def test_empty_expectation_is_not_complete():
    """Describing an empty tree as a finished run is the failure to avoid."""
    with tempfile.TemporaryDirectory() as d:
        m = manifest.build_manifest(d, [])
        assert m["complete"] is False


def test_file_records_carry_size_and_mtime():
    with tempfile.TemporaryDirectory() as d:
        results_io.write_rows(Path(d) / "m.csv", [_row("passed")])
        rec = manifest.build_manifest(d, ["m.csv"])["expected_files"][0]
        assert rec["exists"] and rec["bytes"] > 0 and rec["mtime_utc"]


def test_provenance_is_present_and_best_effort():
    with tempfile.TemporaryDirectory() as d:
        m = manifest.build_manifest(d, [], repo=d)  # not a git repo
        assert "git" in m and "system" in m
        assert m["git"]["sha"] is None  # best effort, not a crash
        assert m["system"]["python"]


def test_write_manifest_round_trips_as_sorted_json():
    with tempfile.TemporaryDirectory() as d:
        results_io.write_rows(Path(d) / "m.csv", [_row("passed")])
        m = manifest.build_manifest(d, ["m.csv"])
        out = manifest.write_manifest(Path(d) / "manifest.json", m)
        back = json.loads(out.read_text())
        assert back["complete"] is True
        keys = list(back)
        assert keys == sorted(keys), "keys must be sorted so runs diff cleanly"


def test_cli_exit_codes():
    with tempfile.TemporaryDirectory() as d:
        results_io.write_rows(Path(d) / "m.csv", [_row("passed")])
        out = str(Path(d) / "m.json")
        assert manifest.main([d, "--expect", "m.csv", "-o", out]) == 0
        assert manifest.main([d, "--expect", "nope.csv", "-o", out]) == 1


def _rungs(passed=0, failed=0, skipped=0):
    return (
        [_row("passed") for _ in range(passed)]
        + [_row("failed", "ERT_CMD_STATE_TIMEOUT") for _ in range(failed)]
        + [
            _row("skipped", "skipped: outside the mode's supported range")
            for _ in range(skipped)
        ]
    )


def test_one_row_where_nine_were_expected_is_not_complete():
    """M3, both directions in one test: the count is what closes it."""
    with tempfile.TemporaryDirectory() as d:
        results_io.write_rows(Path(d) / "m.csv", _rungs(passed=1))

        # Without counts this is the old verdict, and it is wrong about the run.
        assert manifest.build_manifest(d, ["m.csv"])["complete"] is True

        m = manifest.build_manifest(
            d, ["m.csv"], expected_rows={"m.csv": {"rows": 9, "measured": 9}}
        )
        assert m["complete"] is False
        assert m["row_counts_checked"] is True
        assert any("expected 9 row(s), found 1" in r for r in m["incomplete_reasons"])


def test_matching_counts_are_complete_and_recorded():
    with tempfile.TemporaryDirectory() as d:
        results_io.write_rows(Path(d) / "m.csv", _rungs(passed=3, skipped=2))
        m = manifest.build_manifest(
            d,
            ["m.csv"],
            expected_rows={"m.csv": {"rows": 5, "measured": 3, "skipped": 2}},
        )
        assert m["complete"] is True, m["incomplete_reasons"]
        record = m["expected_files"][0]
        assert record["observed_rows"]["total"] == 5
        assert record["observed_rows"]["skipped"] == 2
        assert record["expected_rows"]["measured"] == 3


def test_an_inapplicable_rung_recorded_as_failed_is_reported():
    """M6: the clause that needs `skipped` to be emitted to mean anything."""
    with tempfile.TemporaryDirectory() as d:
        results_io.write_rows(Path(d) / "m.csv", _rungs(passed=3, failed=2))
        m = manifest.build_manifest(
            d,
            ["m.csv"],
            expected_rows={"m.csv": {"rows": 5, "measured": 3, "skipped": 2}},
        )
        assert m["complete"] is False
        joined = " ".join(m["incomplete_reasons"])
        assert "expected 2 skipped row(s), found 0" in joined
        assert "indistinguishable from a regression" in joined


def test_a_regression_fails_the_measured_clause_not_the_row_clause():
    with tempfile.TemporaryDirectory() as d:
        results_io.write_rows(Path(d) / "m.csv", _rungs(passed=2, failed=1))
        m = manifest.build_manifest(
            d, ["m.csv"], expected_rows={"m.csv": {"rows": 3, "measured": 3}}
        )
        assert m["complete"] is False
        joined = " ".join(m["incomplete_reasons"])
        assert "expected 3 passed row(s), found 2" in joined
        assert "row(s), found" in joined and "expected 3 row(s)" not in joined


def test_counts_for_a_file_nobody_expected_are_a_failure():
    with tempfile.TemporaryDirectory() as d:
        results_io.write_rows(Path(d) / "m.csv", _rungs(passed=1))
        m = manifest.build_manifest(
            d, ["m.csv"], expected_rows={"m.csv": {"rows": 1}, "ghost.csv": {"rows": 4}}
        )
        assert m["complete"] is False
        assert any("not in the expected list" in r for r in m["incomplete_reasons"])


def test_a_missing_file_is_not_double_reported_by_the_count_clause():
    with tempfile.TemporaryDirectory() as d:
        m = manifest.build_manifest(
            d, ["gone.csv"], expected_rows={"gone.csv": {"rows": 4}}
        )
        assert m["complete"] is False
        assert len(m["incomplete_reasons"]) == 1, m["incomplete_reasons"]
        assert "MISSING" in m["incomplete_reasons"][0]
        assert m["expected_files"][0]["observed_rows"] is None


def test_no_expected_rows_leaves_the_verdict_and_the_keys_alone():
    with tempfile.TemporaryDirectory() as d:
        results_io.write_rows(Path(d) / "m.csv", _rungs(passed=1))
        m = manifest.build_manifest(d, ["m.csv"])
        assert m["complete"] is True
        assert m["row_counts_checked"] is False
        assert m["expected_files"][0]["expected_rows"] is None
        # observed counts are still recorded: describing what is there costs
        # nothing and is what makes a manifest worth diffing.
        assert m["expected_files"][0]["observed_rows"]["total"] == 1


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"manifest tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
