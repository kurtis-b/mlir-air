# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only checks on reading and writing results CSVs.

    python3 study/test_results_io.py

The load-bearing one is ``test_read_then_write_round_trips``. ``read_rows`` and
``write_rows`` are a pair, and until `[2026-08-08]` they did not compose: csv
hands back ``schema_version`` as the string ``"1"``, ``validate_row`` compares it
against the int ``1``, so a row this module had just written could not be written
to a second file. The error even said "use the adapter to read it", pointing at
iron for a file MLIR-AIR wrote itself.

It went unnoticed because nothing round-tripped: every writer built rows from
``schema.empty_row``. ``run_ladder`` reads each rung's row back from a child
process and rewrites them as one CSV per mode, which is the first caller to
compose the pair -- and it failed on the first rung. Hence this module.
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


def _row(status="passed", **over):
    row = schema.empty_row("results")
    row["execution_mode"] = "hybrid"
    row["run_status"] = status
    row["failure_message"] = "" if status == "passed" else "boom"
    row["seq_len"] = 4096
    row["avg_latency_ms"] = 488.308
    row.update(over)
    return row


def test_read_then_write_round_trips():
    """The pair must compose: run_ladder aggregates rows it read from children."""
    with tempfile.TemporaryDirectory() as d:
        first = Path(d) / "a.csv"
        results_io.write_rows(first, [_row()])
        back = results_io.read_rows(first)
        second = results_io.write_rows(Path(d) / "b.csv", back)  # must not raise
        again = results_io.read_rows(second)
        assert len(again) == 1
        assert float(again[0]["avg_latency_ms"]) == 488.308


def test_schema_version_reads_back_as_an_int():
    """A string "1" here is what broke the round trip; validate_row wants int 1."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "a.csv"
        results_io.write_rows(path, [_row()])
        row = results_io.read_rows(path)[0]
        assert row["schema_version"] == schema.SCHEMA_VERSION
        assert isinstance(row["schema_version"], int)
        schema.validate_row(row)  # the check the round trip actually needs


def test_blank_cells_read_back_as_none():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "a.csv"
        results_io.write_rows(path, [_row(avg_latency_ms=None)])
        assert results_io.read_rows(path)[0]["avg_latency_ms"] is None


def test_multiple_rows_keep_their_order():
    """A ladder's CSV is one row per rung and the rungs are ordered."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "a.csv"
        rows = [_row(seq_len=s) for s in (512, 1024, 2048, 4096)]
        results_io.write_rows(path, rows)
        assert [r["seq_len"] for r in results_io.read_rows(path)] == [
            "512",
            "1024",
            "2048",
            "4096",
        ]


def test_writing_no_rows_is_refused():
    """A header-only CSV reads as 'ran and measured nothing'."""
    with tempfile.TemporaryDirectory() as d:
        try:
            results_io.write_rows(Path(d) / "a.csv", [])
        except ValueError as e:
            assert "no rows" in str(e)
        else:
            raise AssertionError("an empty row list must be refused")


def test_an_invalid_row_is_refused_with_its_index():
    with tempfile.TemporaryDirectory() as d:
        bad = _row()
        bad["execution_mode"] = "coarse"  # a CODE name, not a CSV value
        try:
            results_io.write_rows(Path(d) / "a.csv", [_row(), bad])
        except ValueError as e:
            assert "row 1" in str(e), "the failing row must be identified"
            assert "hybrid" in str(e), "and it must name the value to use"
        else:
            raise AssertionError("a row failing validation must be refused")


def test_a_foreign_header_is_refused_and_points_at_the_adapter():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "iron.csv"
        path.write_text("some,other,columns\n1,2,3\n", encoding="utf-8")
        try:
            results_io.read_rows(path)
        except ValueError as e:
            assert "iron_adapter" in str(e)
        else:
            raise AssertionError("a non-schema header must be refused")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"results-io tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
