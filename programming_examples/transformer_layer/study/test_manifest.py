# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only checks on the results manifest.

    python3 study/test_manifest.py

The load-bearing one is ``test_all_failed_tree_is_not_complete``: iron's manifest
calls a tree complete when the files exist, so a run whose every measurement
failed reports complete. Phase G's gate is "a full profile run completes with a
complete results_manifest.json", so that definition would make the gate unable
to fail -- the same shape as the smoke test doc 09 records.
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


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"manifest tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
