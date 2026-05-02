#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


FAIL_UNDER_LINES = 90.0


def _run(cmd: list[str], *, stdout_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    if stdout_path is None:
        return subprocess.run(cmd, check=False, text=True)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as handle:
        return subprocess.run(cmd, check=False, text=True, stdout=handle)


def _checked(cmd: list[str], *, stdout_path: Path | None = None) -> None:
    completed = _run(cmd, stdout_path=stdout_path)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _totals(json_path: Path) -> dict[str, float]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    totals = payload["totals"]
    statements = float(totals.get("num_statements", 0))
    covered_lines = float(totals.get("covered_lines", 0))
    branches = float(totals.get("num_branches", 0))
    covered_branches = float(totals.get("covered_branches", 0))
    return {
        "line_percent": 100.0 if statements == 0 else covered_lines * 100.0 / statements,
        "branch_percent": 100.0 if branches == 0 else covered_branches * 100.0 / branches,
    }


def main(argv: list[str] | None = None) -> int:
    del argv
    root = Path(__file__).resolve().parent
    os.chdir(root)
    out_dir = root / "artifacts" / "coverage" / "latest"
    out_dir.mkdir(parents=True, exist_ok=True)

    coverage = [sys.executable, "-m", "coverage"]
    _checked([*coverage, "erase"])
    _checked([*coverage, "run", "-m", "pytest", "tests"])

    report_path = out_dir / "coverage.txt"
    _checked([*coverage, "report"], stdout_path=report_path)
    _checked([*coverage, "xml", "-o", str(out_dir / "coverage.xml")])
    _checked([*coverage, "html", "-d", str(out_dir / "html")])

    json_path = out_dir / "coverage.json"
    _checked([*coverage, "json", "-o", str(json_path)])
    totals = _totals(json_path)

    report_text = report_path.read_text(encoding="utf-8")
    print(report_text, end="" if report_text.endswith("\n") else "\n")
    print(f"Line coverage: {totals['line_percent']:.2f}% (fail-under {FAIL_UNDER_LINES:.0f}%)")
    print(f"Branch coverage: {totals['branch_percent']:.2f}% (reported only)")
    print(f"Coverage reports written to {out_dir}")

    if totals["line_percent"] < FAIL_UNDER_LINES:
        print(
            f"ERROR: line coverage {totals['line_percent']:.2f}% is below "
            f"{FAIL_UNDER_LINES:.0f}%",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
