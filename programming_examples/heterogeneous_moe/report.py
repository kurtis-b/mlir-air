#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
from pathlib import Path

from manifest import load_json, project_dir
from reports import matrix_report_markdown, write_markdown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a markdown report from benchmark matrix outputs."
    )
    parser.add_argument(
        "--summary",
        default="artifacts/benchmarks/latest/summary.json",
        help="Summary JSON path relative to this directory.",
    )
    parser.add_argument(
        "--out",
        default="artifacts/benchmarks/latest/report.md",
        help="Markdown report path relative to this directory.",
    )
    parser.add_argument("--title", default="Heterogeneous MoE CPU/GPU Report")
    args = parser.parse_args(argv)

    summary_path = Path(args.summary)
    if not summary_path.is_absolute():
        summary_path = project_dir() / summary_path
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = project_dir() / out_path

    write_markdown(
        matrix_report_markdown(load_json(summary_path), args.title), out_path
    )
    print(f"Wrote markdown report to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
