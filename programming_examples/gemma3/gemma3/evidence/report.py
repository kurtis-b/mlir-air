#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gemma3 paper-parity Markdown report generator."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gemma3.probes.kernel_parity import diagnostic_exclusions
from gemma3.evidence.paper_compare import Comparison, compare_one, load_targets, target_by_id

PASS_CLASSES = {"PAPER_MATCH"}
EXPLAINED_CLASSES = {"PAPER_MATCH", "EXPLAINED_DEVIATION"}


@dataclass(frozen=True)
class ReportRow:
    target_id: str
    metric: str
    local_value: float | None
    paper_value: float | None
    delta_pct: float | None
    classification: str
    correctness: str
    command: str
    log_path: str
    note: str


def _load_result_file(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "results" in data:
        rows = data.get("results", [])
    elif "target_id" in data:
        rows = [data]
    else:
        rows = []
    for row in rows:
        row.setdefault("log_path", str(path))
    return rows


def load_result_rows(paths: list[Path]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in _load_result_file(path):
            target_id = row.get("target_id")
            if target_id:
                rows[str(target_id)] = row
    return rows


def _paper_value(target: dict[str, Any]) -> float | None:
    if "paper_value" in target:
        return target.get("paper_value")
    return None


def build_rows(target_data: dict[str, Any], local_by_target: dict[str, dict[str, Any]]) -> list[ReportRow]:
    rows: list[ReportRow] = []
    threshold = float(target_data.get("similarity_threshold_pct", 20.0))
    for target in target_data["targets"]:
        target_id = target["id"]
        local = local_by_target.get(target_id)
        if local is None:
            comparison = Comparison(
                target_id=target_id,
                metric=target["metric"],
                local_value=None,
                paper_value=_paper_value(target),
                delta_pct=None,
                classification="MISSING_LOCAL_RESULT",
                note="no local result JSON cell",
            )
            correctness = "MISSING"
            command = "n/a"
            log_path = "n/a"
        else:
            comparison = compare_one(target, local, threshold)
            correctness = str(local.get("correctness", "PASS"))
            command = str(local.get("command", "n/a"))
            log_path = str(local.get("log_path", "n/a"))
        rows.append(
            ReportRow(
                target_id=target_id,
                metric=comparison.metric,
                local_value=comparison.local_value,
                paper_value=comparison.paper_value,
                delta_pct=comparison.delta_pct,
                classification=comparison.classification,
                correctness=correctness,
                command=command,
                log_path=log_path,
                note=comparison.note,
            )
        )
    return rows


def overall_status(rows: list[ReportRow]) -> str:
    classes = {row.classification for row in rows}
    if classes <= PASS_CLASSES and all(row.correctness == "PASS" for row in rows):
        return "MATCHES_PAPER"
    if classes <= EXPLAINED_CLASSES and all(row.correctness == "PASS" for row in rows):
        return "MATCHES_WITH_EXPLAINED_DEVIATIONS"
    return "DOES_NOT_MATCH_PAPER"


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:g}"


def _fmt_delta(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}%"


def render_markdown(target_data: dict[str, Any], rows: list[ReportRow]) -> str:
    class_counts = Counter(row.classification for row in rows)
    correctness_counts = Counter(row.correctness for row in rows)
    status = overall_status(rows)
    lines = [
        "# Gemma3 Paper-Parity Report",
        "",
        f"Overall status: `{status}`",
        "",
        "## Correctness Summary",
        "",
    ]
    for name, count in sorted(correctness_counts.items()):
        lines.append(f"- `{name}`: {count}")
    lines.extend(["", "## Performance Summary", ""])
    for name, count in sorted(class_counts.items()):
        lines.append(f"- `{name}`: {count}")

    lines.extend(
        [
            "",
            "## Result Cells",
            "",
            "| Target | Metric | Paper | Local | Delta | Classification | Correctness | Command | Log | Note |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row.target_id} | {row.metric} | {_fmt(row.paper_value)} | {_fmt(row.local_value)} | "
            f"{_fmt_delta(row.delta_pct)} | {row.classification} | {row.correctness} | "
            f"{row.command} | {row.log_path} | {row.note} |"
        )

    lines.extend(["", "## Source Conflicts", ""])
    for conflict in target_data.get("headline_conflicts", []):
        lines.append(
            f"- `{conflict['metric']}`: primary PDF v2={conflict.get('primary_pdf_v2')} "
            f"secondary abstract={conflict.get('secondary_abstract')} {conflict.get('unit', '')}"
        )

    lines.extend(["", "## Unsupported And Diagnostic Modes", ""])
    for exclusion in diagnostic_exclusions():
        lines.append(
            f"- `{exclusion.kernel}` `{exclusion.herd_shape}` `{exclusion.output_mode}`: "
            f"{exclusion.failure_class}; {exclusion.reason}"
        )
    return "\n".join(lines) + "\n"


def generate_report(result_paths: list[Path]) -> str:
    targets = load_targets()
    # Validate target IDs once before rendering to catch stale result JSON.
    known = target_by_id(targets)
    local_by_target = load_result_rows(result_paths)
    stale = sorted(set(local_by_target) - set(known))
    if stale:
        raise ValueError(f"local result references unknown paper targets: {stale}")
    rows = build_rows(targets, local_by_target)
    return render_markdown(targets, rows)


def _self_test() -> None:
    report = generate_report([])
    if "Overall status: `DOES_NOT_MATCH_PAPER`" not in report:
        raise AssertionError("missing overall status")
    if "MISSING_LOCAL_RESULT" not in report:
        raise AssertionError("missing local-result diagnostics")
    if "## Source Conflicts" not in report or "## Unsupported And Diagnostic Modes" not in report:
        raise AssertionError("missing required report sections")
    print("GEMMA3_REPORT_STATUS: DOES_NOT_MATCH_PAPER")
    print("GEMMA3_REPORT_MISSING_LOCAL_RESULT: present")
    print("GEMMA3_REPORT_SELF_TEST: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Gemma3 paper-parity report")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--result", action="append", type=Path, default=[])
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return 0
    report = generate_report(args.result)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(report, encoding="utf-8")
        print(f"GEMMA3_REPORT_MD: {args.output_md}")
    else:
        print(report, end="")
    print(f"GEMMA3_REPORT_STATUS: {overall_status(build_rows(load_targets(), load_result_rows(args.result)))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
