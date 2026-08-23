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

import csv
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


def _v1_header():
    """The 65 v1 names: every current field except the five v2 appended."""
    names = [f.name for f in schema.fields_for("results")]
    return names[: len(names) - 5]


def test_the_v2_columns_really_are_appended_at_the_end():
    """``read_rows_compatible`` rests on schema.py's claim that v2's columns are
    appended AFTER every v1 one. If a later version ever inserts a column in the
    middle, the prefix rule silently starts misreading archived trees -- so the
    property is pinned here rather than trusted."""
    names = [f.name for f in schema.fields_for("results")]
    v2_suffix = ["device_ms", "sync_ms", "host_cpu_ms", "context_loads", "kernel_attaches"]
    v3 = list(schema.MODEL_SCOPE_FIELDNAMES)
    # `[2026-08-23]` v3 appended thirteen model-scope columns AFTER the v2 five,
    # so a v2 file is a strict prefix of a v3 header exactly as a v1 file was
    # of a v2 one. Both suffixes are pinned by position.
    assert names[-len(v3):] == v3, names[-len(v3):]
    assert names[-len(v3) - 5 : -len(v3)] == v2_suffix, names[-len(v3) - 5 : -len(v3)]


def test_compatible_read_accepts_a_v1_file_and_fills_none():
    """Fifteen of this project's nineteen result trees are v1 and are not
    re-measurable at will. The absent columns must come back None, never 0."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "v1.csv"
        names = _v1_header()
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=names)
            w.writeheader()
            row = {n: "" for n in names}
            row["schema_version"] = 1
            row["run_status"] = "passed"
            row["avg_latency_ms"] = "12.5"
            w.writerow(row)
        rows = results_io.read_rows_compatible(path)
    assert len(rows) == 1
    assert rows[0]["avg_latency_ms"] == "12.5"
    assert rows[0]["device_ms"] is None, "an absent column must not read as 0"
    assert rows[0]["context_loads"] is None


def test_the_exact_reader_still_refuses_that_same_v1_file():
    """The control: writers stay strict. If ``read_rows`` also accepted v1 this
    whole function would be pointless, and a v1-shaped row could be written."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "v1.csv"
        names = _v1_header()
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=names)
            w.writeheader()
            w.writerow({n: ("1" if n == "schema_version" else "") for n in names})
        try:
            results_io.read_rows(path)
        except ValueError as e:
            assert "does not match schema" in str(e)
        else:
            raise AssertionError("the exact reader must still refuse a v1 header")


def test_compatible_read_refuses_a_REORDERED_header():
    """A prefix is not the same as a subset. Swapping two columns keeps the name
    set identical and changes every value's meaning."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "swapped.csv"
        names = _v1_header()
        names[3], names[4] = names[4], names[3]
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=names)
            w.writeheader()
            w.writerow({n: ("1" if n == "schema_version" else "") for n in names})
        try:
            results_io.read_rows_compatible(path)
        except ValueError as e:
            assert "not a prefix" in str(e)
        else:
            raise AssertionError("a reordered header must be refused")


def test_compatible_read_refuses_a_NEWER_version():
    """Reading a v3 file here would silently drop columns this code cannot see."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "future.csv"
        rows = [schema.empty_row("results")]
        rows[0]["run_status"] = "passed"
        rows[0]["execution_mode"] = "hybrid"
        results_io.write_rows(path, rows)
        text = path.read_text(encoding="utf-8").splitlines()
        head, first = text[0], text[1].split(",")
        first[0] = str(schema.SCHEMA_VERSION + 1)
        path.write_text(head + "\n" + ",".join(first) + "\n", encoding="utf-8")
        try:
            results_io.read_rows_compatible(path)
        except ValueError as e:
            assert "NEWER" in str(e)
        else:
            raise AssertionError("a newer schema version must be refused")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"results-io tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
