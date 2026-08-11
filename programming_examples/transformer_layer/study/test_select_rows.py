# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only checks on row selection.

python3 study/test_select_rows.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import cases  # noqa: E402
import results_io  # noqa: E402
import schema  # noqa: E402
import select_rows as select_mod  # noqa: E402


def _row(mode="hybrid", seq=1024, family="baseline_768", status="passed", **kw):
    row = schema.empty_row("results")
    row.update(
        {
            "study_id": "s",
            "study_case_id": family,
            "study_case_label": family,
            "workload_variant": cases.FAMILY_SPECS[family].workload_variant,
            "backend": "xrt",
            "execution_mode": mode,
            "seq_len": seq,
            "avg_latency_ms": 100.0,
            "run_status": status,
            "failure_message": "" if status == "passed" else "boom",
        }
    )
    row.update(kw)
    return row


def test_numeric_never_raises_and_distinguishes_empty_from_zero():
    assert select_mod.numeric({"a": ""}, "a") is None
    assert select_mod.numeric({"a": None}, "a") is None
    assert select_mod.numeric({"a": "nan"}, "a") is None
    assert select_mod.numeric({"a": "junk"}, "a") is None
    assert select_mod.numeric({}, "a") is None
    assert select_mod.numeric({"a": "0"}, "a") == 0.0


def test_integer_goes_through_numeric_so_a_float_string_still_parses():
    assert select_mod.integer({"seq_len": "1024.0"}, "seq_len") == 1024
    assert select_mod.integer({"seq_len": ""}, "seq_len") is None


def test_eligibility_checks_identity_not_outcome():
    """A failed rung is a measurement; hiding it is how a ladder looks complete."""
    assert select_mod.is_eligible(_row(status="failed"))
    assert not select_mod.is_eligible(_row(mode="coarse"))  # a CODE name
    assert not select_mod.is_eligible(_row(seq=None))


def test_failed_rows_are_included_by_default():
    rows = [_row(status="passed"), _row(seq=512, status="failed")]
    assert len(select_mod.select(rows)) == 2
    assert len(select_mod.select(rows, status="passed")) == 1


def test_mode_accepts_the_code_name_and_matches_the_csv_value():
    """The most common way to select nothing by accident."""
    rows = [_row(mode="hybrid"), _row(mode="runlist")]
    assert len(select_mod.select(rows, mode="coarse")) == 1
    assert len(select_mod.select(rows, mode="hybrid")) == 1
    assert len(select_mod.select(rows, mode="runlist")) == 1


def test_filters_compose():
    rows = [
        _row(mode="hybrid", seq=512),
        _row(mode="hybrid", seq=1024),
        _row(mode="runlist", seq=512),
    ]
    picked = select_mod.select(rows, mode="coarse", seq_len=512)
    assert len(picked) == 1
    assert picked[0]["execution_mode"] == "hybrid"
    assert picked[0]["seq_len"] == 512


def test_variant_filters_by_the_family_the_row_declares():
    rows = [_row(family="baseline_768"), _row(family="gpt2_512")]
    assert len(select_mod.select(rows, workload_variant="decoder_gpt2")) == 1


def test_rows_come_back_in_matrix_order_not_file_order():
    rows = [
        _row(family="gpt2_512", mode="runlist", seq=4096),
        _row(family="baseline_768", mode="runlist", seq=512),
        _row(family="baseline_768", mode="offload", seq=1024),
    ]
    picked = select_mod.select(rows)
    assert [
        (r["study_case_id"], r["execution_mode"], r["seq_len"]) for r in picked
    ] == [
        ("baseline_768", "offload", 1024),
        ("baseline_768", "runlist", 512),
        ("gpt2_512", "runlist", 4096),
    ]


def test_an_unknown_family_sorts_last_rather_than_raising():
    """A tree from an older run is still worth reading."""
    rows = [_row(family="baseline_768")]
    stranger = _row()
    stranger["study_case_id"] = "retired_family"
    picked = select_mod.select(rows + [stranger])
    assert picked[-1]["study_case_id"] == "retired_family"


def test_load_tree_reads_the_root_and_one_level_down():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "sub").mkdir()
        results_io.write_rows(root / "coarse.csv", [_row(mode="hybrid")])
        results_io.write_rows(root / "sub" / "runlist.csv", [_row(mode="runlist")])
        rows, skipped = select_mod.load_tree(root)
    assert len(rows) == 2
    assert skipped == []


def test_load_tree_does_not_recurse_without_limit():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        deep = root / "archive" / "older"
        deep.mkdir(parents=True)
        results_io.write_rows(deep / "coarse.csv", [_row()])
        rows, _ = select_mod.load_tree(root)
    assert rows == []


def test_load_tree_returns_what_it_skipped_rather_than_printing_it():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        results_io.write_rows(root / "good.csv", [_row()])
        (root / "foreign.csv").write_text("a,b\n1,2\n", encoding="utf-8")
        rows, skipped = select_mod.load_tree(root)
    assert len(rows) == 1
    assert len(skipped) == 1
    assert "foreign.csv" in skipped[0]


def test_an_empty_tree_is_empty_and_not_an_error():
    with tempfile.TemporaryDirectory() as d:
        rows, skipped = select_mod.load_tree(d)
    assert (rows, skipped) == ([], [])


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"select tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
