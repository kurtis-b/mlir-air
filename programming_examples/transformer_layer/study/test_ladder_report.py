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

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import ladder_report  # noqa: E402


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


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"ladder-report tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
