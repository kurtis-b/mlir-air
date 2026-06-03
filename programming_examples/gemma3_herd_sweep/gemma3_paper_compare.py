#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gemma3 paper-target validation and local-result comparison."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_TARGETS = Path(__file__).with_name("paper_targets.json")
_GEMMA_DIR = Path(__file__).resolve().parent
if str(_GEMMA_DIR) not in sys.path:
    sys.path.insert(0, str(_GEMMA_DIR))
from gemma3_environment import validate_environment_for_paper
REQUIRED_METRICS = {
    "prefill_ttft_seconds",
    "decode_tps",
    "vision_ttft_seconds",
    "vision_speedup",
    "average_power_watts",
    "prefill_tps_per_watt_speedup",
    "decode_tps_per_watt_speedup",
}
REQUIRED_HEADLINE_CONFLICTS = {
    "prefill_speedup_npu_vs_igpu",
    "decode_speedup_npu_vs_igpu",
    "prefill_speedup_npu_vs_cpu",
    "decode_speedup_npu_vs_cpu",
    "power_efficiency_npu_vs_igpu",
    "power_efficiency_npu_vs_cpu",
}
TIMED_PAPER_METRICS = {
    "prefill_ttft_seconds",
    "decode_tps",
    "vision_ttft_seconds",
    "prefill_tps_per_watt_speedup",
    "decode_tps_per_watt_speedup",
}


@dataclass(frozen=True)
class Comparison:
    target_id: str
    metric: str
    local_value: float | None
    paper_value: float | None
    delta_pct: float | None
    classification: str
    note: str = ""

    def format(self) -> str:
        delta = "n/a" if self.delta_pct is None else f"{self.delta_pct:.2f}%"
        paper = "n/a" if self.paper_value is None else f"{self.paper_value:g}"
        local = "n/a" if self.local_value is None else f"{self.local_value:g}"
        note = f" note={self.note}" if self.note else ""
        return (
            f"compare {self.target_id} metric={self.metric} local={local} "
            f"paper={paper} delta={delta} class={self.classification}{note}"
        )


def load_targets(path: Path = DEFAULT_TARGETS) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError(f"unsupported paper target schema: {data.get('schema_version')}")
    return data


def validate_targets(data: dict[str, Any]) -> None:
    targets = data.get("targets", [])
    if not targets:
        raise ValueError("paper target file has no targets")
    ids = [target["id"] for target in targets]
    duplicates = sorted({target_id for target_id in ids if ids.count(target_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate paper target ids: {duplicates}")

    metrics = {target.get("metric") for target in targets}
    missing_metrics = REQUIRED_METRICS - metrics
    if missing_metrics:
        raise ValueError(f"missing paper target metrics: {sorted(missing_metrics)}")

    conflicts = {item.get("metric") for item in data.get("headline_conflicts", [])}
    missing_conflicts = REQUIRED_HEADLINE_CONFLICTS - conflicts
    if missing_conflicts:
        raise ValueError(f"missing headline conflict metrics: {sorted(missing_conflicts)}")

    _require_scalar_cells(targets, "prefill_ttft_seconds", "gemma3-1b", 18)
    _require_scalar_cells(targets, "prefill_ttft_seconds", "gemma3-4b", 18)
    _require_scalar_cells(targets, "decode_tps", "gemma3-1b", 20)
    _require_scalar_cells(targets, "decode_tps", "gemma3-4b", 24)
    _require_scalar_cells(targets, "vision_ttft_seconds", "gemma3-4b-vision", 3)
    _require_scalar_cells(targets, "average_power_watts", None, 48)


def _require_scalar_cells(targets: list[dict[str, Any]], metric: str, model: str | None, expected: int) -> None:
    cells = [target for target in targets if target.get("metric") == metric]
    if model is not None:
        cells = [target for target in cells if target.get("model_variant") == model]
    if len(cells) != expected:
        label = f"{metric}/{model or 'all'}"
        raise ValueError(f"expected {expected} target cells for {label}, got {len(cells)}")


def target_by_id(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {target["id"]: target for target in data["targets"]}


def compare_one(target: dict[str, Any], local: dict[str, Any], threshold_pct: float) -> Comparison:
    if local.get("correctness", "PASS") != "PASS":
        return Comparison(target["id"], target["metric"], local.get("local_value"), target.get("paper_value"), None, "LOCAL_FAIL", str(local.get("correctness")))
    if target.get("classification_hint") != "OOC" and target.get("metric") in TIMED_PAPER_METRICS:
        fallback_names = unmeasured_host_fallbacks(local)
        if fallback_names:
            return Comparison(
                target["id"],
                target["metric"],
                local.get("local_value"),
                target.get("paper_value"),
                None,
                "UNMEASURED_HOST_FALLBACK",
                "unmeasured_host_fallbacks=" + ",".join(fallback_names),
            )
    if target.get("classification_hint") == "OOC":
        classification = "PAPER_MATCH" if local.get("classification") == "OOC" else "EXPLAINED_DEVIATION"
        return Comparison(target["id"], target["metric"], None, None, None, classification, "paper target is out-of-capacity")

    local_value = local.get("local_value")
    if local_value is None:
        return Comparison(target["id"], target["metric"], None, target.get("paper_value"), None, "MISSING_LOCAL_RESULT")
    local_value = float(local_value)

    if "paper_min" in target and "paper_max" in target:
        low = float(target["paper_min"])
        high = float(target["paper_max"])
        if low <= local_value <= high:
            return Comparison(target["id"], target["metric"], local_value, None, 0.0, "PAPER_MATCH", f"range={low:g}-{high:g}")
        nearest = low if local_value < low else high
        delta_pct = abs(local_value - nearest) / nearest * 100.0
        classification = "PAPER_MATCH" if delta_pct <= threshold_pct else _deviation_class(local)
        return Comparison(target["id"], target["metric"], local_value, nearest, delta_pct, classification, f"range={low:g}-{high:g}")

    paper_value = target.get("paper_value")
    if paper_value is None:
        return Comparison(target["id"], target["metric"], local_value, None, None, "EXPLAINED_DEVIATION", "non-scalar paper target")
    paper_value = float(paper_value)
    if paper_value == 0.0:
        delta_pct = 0.0 if local_value == 0.0 else float("inf")
    else:
        delta_pct = abs(local_value - paper_value) / paper_value * 100.0
    classification = "PAPER_MATCH" if delta_pct <= threshold_pct else _deviation_class(local)
    return Comparison(target["id"], target["metric"], local_value, paper_value, delta_pct, classification)


def _deviation_class(local: dict[str, Any]) -> str:
    return "EXPLAINED_DEVIATION" if local.get("explanation") else "LOCAL_FAIL"


def _fallback_name(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return str(item.get("name") or item.get("operation") or item.get("stage") or "unknown")
    return str(item)


def _fallback_is_unmeasured(item: Any) -> bool:
    if isinstance(item, str):
        return True
    if not isinstance(item, dict):
        return True
    contributes = bool(item.get("contributes_to_timing", item.get("timed_window", True)))
    measured = bool(item.get("measured", False))
    status = str(item.get("status", ""))
    if status in {
        "measured",
        "measured-host-fallback",
        "npu",
        "validated",
        "hardware-validated",
    }:
        measured = True
    return contributes and not measured


def unmeasured_host_fallbacks(local: dict[str, Any]) -> tuple[str, ...]:
    names: list[str] = []
    for key in ("unmeasured_host_fallbacks", "host_fallbacks"):
        for item in local.get(key, []) or []:
            if _fallback_is_unmeasured(item):
                names.append(_fallback_name(item))
    return tuple(dict.fromkeys(names))


def _result_rows(local_results: dict[str, Any]) -> list[dict[str, Any]]:
    if "results" in local_results:
        return list(local_results.get("results", []))
    if "target_id" in local_results:
        return [local_results]
    return []


def compare_results(data: dict[str, Any], local_results: dict[str, Any]) -> list[Comparison]:
    threshold = float(data.get("similarity_threshold_pct", 20.0))
    targets = target_by_id(data)
    results = []
    for local in _result_rows(local_results):
        target_id = local.get("target_id")
        if target_id not in targets:
            raise ValueError(f"local result references unknown paper target: {target_id}")
        results.append(compare_one(targets[target_id], local, threshold))
    return results


def write_markdown_summary(comparisons: list[Comparison], path: Path) -> None:
    lines = [
        "| Target | Metric | Local | Paper | Delta | Classification | Note |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in comparisons:
        delta = "n/a" if item.delta_pct is None else f"{item.delta_pct:.2f}%"
        paper = "n/a" if item.paper_value is None else f"{item.paper_value:g}"
        local = "n/a" if item.local_value is None else f"{item.local_value:g}"
        lines.append(
            f"| {item.target_id} | {item.metric} | {local} | {paper} | "
            f"{delta} | {item.classification} | {item.note} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv_summary(comparisons: list[Comparison], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target_id", "metric", "local_value", "paper_value", "delta_pct", "classification", "note"])
        for item in comparisons:
            writer.writerow([item.target_id, item.metric, item.local_value, item.paper_value, item.delta_pct, item.classification, item.note])


def _fixture() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "results": [
            {
                "target_id": "prefill_ttft_seconds_gemma3_1b_npu_1024",
                "local_value": 1.0,
                "correctness": "PASS",
            },
            {
                "target_id": "decode_tps_gemma3_4b_npu_131072",
                "local_value": 7.0,
                "correctness": "PASS",
                "explanation": "synthetic fixture deliberately outside tolerance",
            },
            {
                "target_id": "decode_tps_gemma3_1b_npu_65536",
                "classification": "OOC",
                "correctness": "PASS",
            },
            {
                "target_id": "prefill_tps_per_watt_speedup_gemma3_4b_npu_vs_cpu",
                "local_value": 100.0,
                "correctness": "PASS",
            },
            {
                "target_id": "decode_tps_gemma3_1b_npu_1024",
                "local_value": 1000.0,
                "correctness": "PASS",
                "host_fallbacks": [
                    {
                        "name": "rms_norm",
                        "contributes_to_timing": True,
                        "measured": False,
                    }
                ],
            },
            {
                "target_id": "decode_tps_gemma3_1b_npu_2048",
                "local_value": 40.5,
                "correctness": "PASS",
                "host_fallbacks": [
                    {
                        "name": "rms_norm",
                        "status": "measured-host-fallback",
                        "contributes_to_timing": True,
                        "measured": True,
                    }
                ],
            },
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Gemma3 paper targets and compare local result JSON")
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--summary-md", type=Path)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--allow-incomplete-environment", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    data = load_targets(args.targets)
    validate_targets(data)
    if args.validate:
        print(f"GEMMA3_PAPER_TARGETS_VALID: targets={len(data['targets'])} conflicts={len(data['headline_conflicts'])}")
    if args.environment and not args.allow_incomplete_environment:
        validate_environment_for_paper(json.loads(args.environment.read_text(encoding="utf-8")))
    if args.compare:
        local_results = json.loads(args.compare.read_text(encoding="utf-8"))
        comparisons = compare_results(data, local_results)
        for comparison in comparisons:
            print(comparison.format())
        if args.summary_md:
            write_markdown_summary(comparisons, args.summary_md)
            print(f"GEMMA3_COMPARE_SUMMARY_MD: {args.summary_md}")
        if args.summary_csv:
            write_csv_summary(comparisons, args.summary_csv)
            print(f"GEMMA3_COMPARE_SUMMARY_CSV: {args.summary_csv}")
    if args.self_test:
        comparisons = compare_results(data, _fixture())
        classes = {comparison.classification for comparison in comparisons}
        if (
            "PAPER_MATCH" not in classes
            or "EXPLAINED_DEVIATION" not in classes
            or "UNMEASURED_HOST_FALLBACK" not in classes
        ):
            raise AssertionError(f"fixture did not exercise expected comparison classes: {classes}")
        for comparison in comparisons:
            print(comparison.format())
        print("GEMMA3_PAPER_COMPARE_SELF_TEST: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
