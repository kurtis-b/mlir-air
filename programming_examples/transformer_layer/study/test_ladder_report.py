# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only checks on the ladder analysis.

    python3 study/test_ladder_report.py

This module is here because its subject produces *conclusions*, not artifacts. A
bug in ``results_io`` shows up as an exception; a bug here shows up as a
plausible sentence about which execution mode wins, which is exactly the kind of
wrong answer that gets quoted into a document and believed. So both claims the
report makes are pinned in both directions: the exponent recovers a known
synthetic slope, and a crossover is reported when and only when the ranking
genuinely swaps between two rungs that both passed.
"""

import io
import os
import sys
import tempfile
from contextlib import redirect_stdout

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import ladder_report  # noqa: E402
import results_io  # noqa: E402
import schema  # noqa: E402

#: The keys ``load`` attaches and every function below reads. Named once so the
#: in-memory fixture and the on-disk one cannot drift apart silently --
#: ``test_the_in_memory_fixture_matches_what_load_produces`` asserts they agree.
_DERIVED = ("_seq", "_ms", "_ok")


def _rows(pairs, ok=True):
    """pairs: [(seq, ms)]; ms=None marks a failed rung."""
    out = []
    for seq, ms in pairs:
        passed = ok and ms is not None
        out.append(
            {
                "_seq": seq,
                "_ms": ms,
                "_ok": passed,
                "run_status": "passed" if passed else "failed",
                "failure_message": "" if passed else "synthetic failure",
                "host_submissions_per_layer": 1,
                "herd_launches": 2,
                "sync_boundaries": 3,
                "bytes_transferred": 4,
            }
        )
    return out


def _csv_row(mode, seq, ms, status="passed", message=None):
    """A real schema row, built the way ``run_ladder`` builds one."""
    row = schema.empty_row("results")
    row["execution_mode"] = schema.EXECUTION_MODE_CSV.get(mode, mode)
    row["study_id"] = "test-ladder-report"
    row["seq_len"] = seq
    row["study_case_id"] = f"{seq}x768_encoder_bert"
    row["study_case_label"] = f"{mode} seq {seq}"
    row["run_status"] = status
    row["avg_latency_ms"] = ms
    row["failure_message"] = message
    row["host_submissions_per_layer"] = 1
    row["herd_launches"] = 2
    row["sync_boundaries"] = 3
    row["bytes_transferred"] = 4
    return row


def _tree(spec):
    """Write ``{mode: [(seq, ms, status)]}`` as a real results tree.

    Returns the TemporaryDirectory object -- keep it alive for the test's
    duration. Rows go through ``results_io.write_rows``, so they are validated
    against the shipped schema: a fixture that the real writer would refuse is
    not a fixture this module may test ``load`` against.
    """
    tmp = tempfile.TemporaryDirectory()
    for mode, rungs in spec.items():
        rows = [
            _csv_row(
                mode,
                seq,
                ms,
                status=status,
                message=None if status == "passed" else f"{status}: synthetic",
            )
            for seq, ms, status in rungs
        ]
        results_io.write_rows(os.path.join(tmp.name, f"{mode}.csv"), rows)
    return tmp


def test_exponent_recovers_a_linear_slope():
    rows = _rows([(512, 10.0), (1024, 20.0), (2048, 40.0)])
    assert abs(ladder_report.exponent(rows) - 1.0) < 1e-9


def test_exponent_recovers_a_quadratic_slope():
    rows = _rows([(512, 10.0), (1024, 40.0), (2048, 160.0)])
    assert abs(ladder_report.exponent(rows) - 2.0) < 1e-9


def test_exponent_needs_three_points():
    """A two-point slope is an interpolation, not a trend. It must not print."""
    assert ladder_report.exponent(_rows([(512, 10.0), (1024, 20.0)])) is None
    assert ladder_report.exponent(_rows([(512, 10.0)])) is None


def test_exponent_ignores_failed_rungs():
    """Three rows, one failed: two usable points, so no slope."""
    rows = _rows([(512, 10.0), (1024, None), (2048, 40.0)])
    assert ladder_report.exponent(rows) is None


def test_exponent_uses_only_passing_rungs_for_the_fit():
    rows = _rows([(512, 10.0), (1024, 20.0), (2048, 40.0), (4096, None)])
    assert (
        abs(ladder_report.exponent(rows) - 1.0) < 1e-9
    ), "the failed rung must not tilt it"


def test_crossover_is_found_when_the_ranking_swaps():
    data = {
        "a": _rows([(512, 10.0), (1024, 100.0)]),
        "b": _rows([(512, 50.0), (1024, 60.0)]),
    }
    found = ladder_report.crossovers(data)
    assert len(found) == 1
    assert "a leads b at seq 512" in found[0]
    assert "trails it at seq 1024" in found[0]


def test_no_crossover_when_the_ranking_holds():
    data = {
        "a": _rows([(512, 10.0), (1024, 20.0)]),
        "b": _rows([(512, 50.0), (1024, 60.0)]),
    }
    assert ladder_report.crossovers(data) == []


def test_crossover_needs_both_sides_to_pass():
    """The gap must not be bridged: b never measured 1024, so nothing crossed."""
    data = {
        "a": _rows([(512, 10.0), (1024, 100.0)]),
        "b": _rows([(512, 50.0), (1024, None)]),
    }
    assert ladder_report.crossovers(data) == []


def test_crossover_is_reported_once_per_pair_and_interval():
    data = {
        "a": _rows([(512, 10.0), (1024, 100.0), (2048, 10.0)]),
        "b": _rows([(512, 50.0), (1024, 60.0), (2048, 50.0)]),
    }
    found = ladder_report.crossovers(data)
    assert len(found) == 2, "one per adjacent interval where the sign flips"


def test_render_names_failed_rungs_and_does_not_invent_latency():
    data = {"a": _rows([(512, 10.0), (1024, None)])}
    text = ladder_report.render(data)
    assert "**FAILED**" in text
    assert "synthetic failure" in text
    assert "n/a" in text, "two points, so no slope column value"


def test_render_reports_no_crossovers_explicitly():
    data = {"a": _rows([(512, 10.0), (1024, 20.0)])}
    assert "none: the ranking is the same at every rung" in ladder_report.render(data)


# ---------------------------------------------------------------------------
# ``load`` and ``main`` -- the two entry points. Everything above this line runs
# on a fixture this module builds by hand, which means every check above holds
# whether or not ``load`` produces that shape at all. These close that gap.
# ---------------------------------------------------------------------------


def test_load_reads_a_real_results_tree():
    """The entry point, against CSVs the shipped writer produced."""
    with _tree(
        {
            "coarse": [(512, 10.0, "passed"), (1024, 20.0, "passed")],
            "fused": [(512, 5.0, "passed"), (1024, 40.0, "passed")],
        }
    ) as root:
        data = ladder_report.load(root, ["coarse", "fused"])
    assert sorted(data) == ["coarse", "fused"]
    assert [r["_seq"] for r in data["coarse"]] == [512, 1024]
    assert [r["_ms"] for r in data["fused"]] == [5.0, 40.0]
    assert all(r["_ok"] for rows in data.values() for r in rows)


def test_the_in_memory_fixture_matches_what_load_produces():
    """The fixture above must carry the keys ``load`` attaches, with the same
    types. Otherwise every check in this module is written against a shape the
    production reader does not produce, and a rename in ``load`` moves nothing
    red -- the exact failure this section exists to close."""
    with _tree({"coarse": [(512, 10.0, "passed")]}) as root:
        loaded = ladder_report.load(root, ["coarse"])["coarse"][0]
    handmade = _rows([(512, 10.0)])[0]
    for key in _DERIVED:
        assert key in loaded, f"load stopped attaching {key}"
        assert key in handmade, f"the fixture stopped carrying {key}"
        assert type(loaded[key]) is type(handmade[key]), (
            f"{key}: load gives {type(loaded[key]).__name__}, "
            f"the fixture gives {type(handmade[key]).__name__}"
        )
    # The keys ``render`` indexes directly off the row must survive the CSV
    # round-trip too, or the table renders on a mode nobody measured.
    for key, _label in ladder_report._STRUCT:
        assert loaded[key] is not None, f"{key} did not survive load()"
    assert loaded["run_status"] == "passed"


def test_load_does_not_count_a_skipped_rung_as_passed():
    """``run_status='skipped'`` is a rung that could not apply, and
    ``run_ladder.walk`` has emitted it since queue item 12. It has no latency,
    so treating it as a measurement would put a hole in a fit and report the
    slope anyway."""
    with _tree(
        {
            "fused": [
                (512, 10.0, "passed"),
                (1024, 20.0, "passed"),
                (2048, None, "skipped"),
                (4096, None, "skipped"),
            ]
        }
    ) as root:
        rows = ladder_report.load(root, ["fused"])["fused"]
    assert [r["_ok"] for rows_ in (rows,) for r in rows_] == [True, True, False, False]
    assert [r["_ms"] for r in rows] == [10.0, 20.0, None, None]
    # Two usable points out of four, so no slope -- not a slope fitted on two.
    assert ladder_report.exponent(rows) is None


def test_load_orders_by_seq_len_not_by_row_order():
    """The module's own contract: 'Rows are matched on seq_len, not on row
    order, so a partially written or reordered CSV lines up correctly.'"""
    with _tree(
        {"coarse": [(2048, 40.0, "passed"), (512, 10.0, "passed"), (1024, 20.0, "passed")]}
    ) as root:
        rows = ladder_report.load(root, ["coarse"])["coarse"]
    assert [r["_seq"] for r in rows] == [512, 1024, 2048]
    # And the fit is the linear one, which it would not be if the reader had
    # paired 2048 with 10.0.
    assert abs(ladder_report.exponent(rows) - 1.0) < 1e-9


def test_load_skips_a_mode_with_no_csv_rather_than_raising():
    with _tree({"coarse": [(512, 10.0, "passed")]}) as root:
        data = ladder_report.load(root, ["coarse", "offload", "runlist"])
    assert sorted(data) == ["coarse"], "a mode that never ran is absent, not empty"


def test_main_reports_the_numbers_that_are_in_the_csvs():
    with _tree(
        {
            "coarse": [(512, 10.0, "passed"), (1024, 100.0, "passed")],
            "fused": [(512, 50.0, "passed"), (1024, 60.0, "passed")],
        }
    ) as root:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = ladder_report.main([root, "--modes", "coarse,fused"])
    out = buf.getvalue()
    assert rc == 0
    assert "10.0" in out and "100.0" in out
    # coarse leads at 512 and trails at 1024: a real crossover, end to end.
    assert "coarse leads fused at seq 512" in out


def test_main_writes_the_markdown_it_was_asked_for():
    with _tree({"coarse": [(512, 10.0, "passed")]}) as root:
        md = os.path.join(root, "report.md")
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = ladder_report.main([root, "--modes", "coarse", "--md", md])
        assert rc == 0
        assert os.path.isfile(md), "--md was accepted and wrote nothing"
        written = open(md, encoding="utf-8").read()
    assert "Latency by sequence length" in written
    assert "10.0" in written


def test_main_refuses_an_empty_tree_instead_of_printing_an_empty_table():
    """A tree with no <mode>.csv is a run that did not happen. Returning 0 with
    an empty table is how a walk that produced nothing reads as a result."""
    with tempfile.TemporaryDirectory() as root:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = ladder_report.main([root])
    assert rc == 1
    assert "no <mode>.csv found" in buf.getvalue()


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"ladder-report tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
