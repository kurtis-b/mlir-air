# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only checks on the results CSV layer and Phase F's smoke gate.

    python3 study/test_smoke_gate.py

No NPU, no MLIR. Plain test_* functions with a main() runner (convention 11).

The gate's whole value is in what it REJECTS, so most of these are negatives:
an all-failed CSV, an empty one, a missing one, an unreadable one, and being
handed nothing to check. A smoke gate that only ever passes is the artifact doc
09 warns about -- iron shipped one that reported 21/21 on an environment where
every measurement had failed.
"""

import os
import sys
import tempfile
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import results_io  # noqa: E402
import schema  # noqa: E402
import smoke_gate  # noqa: E402


def _row(status, msg=None, mode="hybrid"):
    row = schema.empty_row("results")
    row["execution_mode"] = mode
    row["run_status"] = status
    row["failure_message"] = msg
    return row


def _raises(exc, match, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc as e:
        assert match in str(e), f"expected {match!r} in {str(e)!r}"
        return
    raise AssertionError(f"expected {exc.__name__} containing {match!r}")


# --- the CSV layer ---------------------------------------------------------


def test_round_trip_preserves_values_and_none():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "r.csv"
        row = _row("passed")
        row["seq_len"] = 4096
        results_io.write_rows(p, [row])
        back = results_io.read_rows(p)
        assert len(back) == 1
        assert back[0]["run_status"] == "passed"
        assert str(back[0]["seq_len"]) == "4096"
        assert back[0]["avg_latency_ms"] is None  # empty cell -> None


def test_write_refuses_an_empty_row_list():
    """A header-only CSV reads as 'ran and measured nothing'."""
    with tempfile.TemporaryDirectory() as d:
        _raises(
            ValueError,
            "measured nothing",
            results_io.write_rows,
            Path(d) / "r.csv",
            [],
        )


def test_write_validates_every_row():
    with tempfile.TemporaryDirectory() as d:
        bad = _row("passed")
        del bad["avg_latency_ms"]
        _raises(
            ValueError,
            "row 0 is not writable",
            results_io.write_rows,
            Path(d) / "r.csv",
            [bad],
        )


def test_read_rejects_a_foreign_header():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "r.csv"
        p.write_text("a,b\n1,2\n", encoding="utf-8")
        _raises(ValueError, "header does not match", results_io.read_rows, p)


# --- the gate --------------------------------------------------------------


def test_gate_passes_when_a_row_passed():
    with tempfile.TemporaryDirectory() as d:
        results_io.write_rows(
            Path(d) / "m.csv", [_row("failed", "boom"), _row("passed")]
        )
        assert smoke_gate.check_results_root(d, ["m.csv"]) == []


def test_gate_fails_when_every_row_failed_and_quotes_the_reason():
    """The headline case: complete, well-formed CSVs full of failures."""
    with tempfile.TemporaryDirectory() as d:
        results_io.write_rows(
            Path(d) / "m.csv",
            [
                _row("failed", "ERT_CMD_STATE_TIMEOUT"),
                _row("failed", "ERT_CMD_STATE_TIMEOUT"),
            ],
        )
        problems = smoke_gate.check_results_root(d, ["m.csv"])
        assert len(problems) == 1
        assert "none with run_status=passed" in problems[0]
        assert "ERT_CMD_STATE_TIMEOUT" in problems[0]


def test_gate_exempts_a_file_the_plan_expects_fully_skipped_and_only_that():
    """`[2026-08-19]` Per-(family, mode) reachability's one new case: `fused`
    at a decoder family expects ZERO measured rows, so the one-passed-row rule
    would demand a measurement the plan says cannot exist. Three clauses, and
    each is load-bearing: the all-skipped file passes WITH the expectation, a
    non-skipped row in it is flagged (a rung ran against a plan that said it
    could not), and WITHOUT the expectation the old rule is untouched."""
    with tempfile.TemporaryDirectory() as d:
        results_io.write_rows(
            os.path.join(d, "fused.csv"),
            [_row("skipped", "skipped: fused cannot build decoder_gpt2")],
        )
        expect = {"fused.csv": {"rows": 1, "measured": 0, "skipped": 1}}
        assert (
            smoke_gate.check_results_root(d, ["fused.csv"], expected_rows=expect)
            == []
        )
        # A passed row where the plan said nothing could run is a finding.
        results_io.write_rows(
            os.path.join(d, "fused.csv"), [_row("passed"), _row("skipped")]
        )
        problems = smoke_gate.check_results_root(
            d, ["fused.csv"], expected_rows=expect
        )
        assert len(problems) == 1 and "ran against a plan" in problems[0]
        # Without the expectation the old rule stands: all-skipped fails.
        results_io.write_rows(
            os.path.join(d, "fused.csv"),
            [_row("skipped", "skipped: fused cannot build decoder_gpt2")],
        )
        problems = smoke_gate.check_results_root(d, ["fused.csv"])
        assert len(problems) == 1 and "none with run_status=passed" in problems[0]


def test_gate_fails_on_a_missing_file():
    """iron's version skips these; a measurement that never ran must fail."""
    with tempfile.TemporaryDirectory() as d:
        problems = smoke_gate.check_results_root(d, ["never_written.csv"])
        assert len(problems) == 1 and "MISSING" in problems[0]


def test_gate_fails_on_an_unreadable_file():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "m.csv").write_text("not,a,schema\n1,2,3\n", encoding="utf-8")
        problems = smoke_gate.check_results_root(d, ["m.csv"])
        assert len(problems) == 1 and "unreadable" in problems[0]


def test_gate_fails_when_given_nothing_to_check():
    """A gate that checked nothing must not report success."""
    with tempfile.TemporaryDirectory() as d:
        problems = smoke_gate.check_results_root(d, [])
        assert len(problems) == 1 and "proved" in problems[0]


def test_gate_reports_every_bad_csv_not_just_the_first():
    with tempfile.TemporaryDirectory() as d:
        results_io.write_rows(Path(d) / "a.csv", [_row("failed", "one")])
        results_io.write_rows(Path(d) / "b.csv", [_row("failed", "two")])
        problems = smoke_gate.check_results_root(d, ["a.csv", "b.csv", "c.csv"])
        assert len(problems) == 3


def test_cli_exit_codes():
    with tempfile.TemporaryDirectory() as d:
        results_io.write_rows(Path(d) / "m.csv", [_row("passed")])
        assert smoke_gate.main([d, "--expect", "m.csv"]) == 0
        assert smoke_gate.main([d, "--expect", "missing.csv"]) == 1
        assert smoke_gate.main([d]) == 1  # nothing to check


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"smoke-gate tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
