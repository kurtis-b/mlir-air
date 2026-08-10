# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase F's gate: every expected measurement CSV has a row that actually passed.

    python3 study/smoke_gate.py <results-root> [--expect rel/path.csv ...]

CONTRACT
    Given a results root and the CSVs a run was supposed to produce, return 0
    only if EVERY one of them exists, is readable as the current schema, and carries at
    least one row with ``run_status=passed``. Anything else is a failure, and
    the report names the file and quotes the first ``failure_message`` verbatim
    -- which is usually enough to identify the cause without opening the tree.

WHY THE BAR IS "A PASSED ROW" AND NOT "A FILE"
    Doc 09: "at least one row per measurement CSV has ``run_status=passed``, not
    merely that the expected files exist and are non-empty. That distinction is
    not pedantry: iron shipped a smoke test that checked only file existence and
    it reported 21/21 passed on an environment where every measurement had
    failed." A broken environment still writes complete, well-formed CSVs full
    of failed rows -- that is by design (see ``results_io``), and it is what
    makes existence worthless as evidence.

TWO HOLES IN IRON'S VERSION, CLOSED HERE
    ``unattended_smoke_job.py::_measured_row_failures`` is the corrected shape --
    it does look for a passed row -- but it ``continue``s past a file that does
    not exist, and past one whose rows are empty or lack the column. So a
    measurement that never ran at all, or produced nothing, passes its gate.
    That is the same "passed having measured nothing" failure from the other
    direction. Here a missing, empty, or unreadable expected CSV is a FAILURE:
    the caller said it should be there.

FOOTGUNS
    - **The expected list is the contract.** This gate cannot know what a run
      was supposed to produce; passing no ``--expect`` checks nothing and says
      so rather than returning a cheerful 0.
    - A CSV whose header is not the current schema fails here rather than being skipped.
      Silently ignoring an unreadable results file is how a gate ends up
      measuring nothing.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import results_io  # noqa: E402


def check_results_root(root: str | Path, expected: list[str]) -> list[str]:
    """Problems with ``root``; empty means the gate passes.

    ``expected`` are paths relative to ``root``. An empty list is itself a
    problem: a gate given nothing to check has checked nothing.
    """
    root = Path(root)
    if not expected:
        return [
            "no expected measurement CSVs were named, so this gate proved "
            "nothing. Pass the files the run was supposed to produce."
        ]

    problems: list[str] = []
    for rel in expected:
        path = root / rel
        if not path.exists():
            problems.append(
                f"{rel}: MISSING. The run was supposed to produce it, so its "
                "absence is a failure, not something to skip -- a measurement "
                "that never ran is the case this gate exists to catch."
            )
            continue
        try:
            rows = results_io.read_rows(path)
        except Exception as e:
            problems.append(f"{rel}: unreadable as the current schema -- {e}")
            continue
        if not rows:
            problems.append(
                f"{rel}: 0 rows. A failed measurement still writes a complete "
                "row, so an empty CSV means nothing was attempted."
            )
            continue

        passed = [r for r in rows if r.get("run_status") == "passed"]
        if passed:
            continue

        reasons = sorted(
            {
                str(r.get("failure_message") or "").strip()
                for r in rows
                if str(r.get("failure_message") or "").strip()
            }
        )
        detail = f" First failure: {reasons[0]}" if reasons else ""
        problems.append(
            f"{rel}: {len(rows)} row(s), none with run_status=passed.{detail}"
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results_root")
    ap.add_argument(
        "--expect",
        action="append",
        default=[],
        metavar="REL/PATH.csv",
        help="a measurement CSV the run should have produced, relative to the "
        "results root. Repeatable. Required: with none, the gate checks nothing.",
    )
    args = ap.parse_args(argv)

    problems = check_results_root(args.results_root, args.expect)
    for p in problems:
        print(f"[smoke-gate] {p}")
    if problems:
        print(f"[smoke-gate] FAIL ({len(problems)} problem(s))")
        return 1
    print(f"[smoke-gate] PASS ({len(args.expect)} CSV(s), each with a passed row)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
