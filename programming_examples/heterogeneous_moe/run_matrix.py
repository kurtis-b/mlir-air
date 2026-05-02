#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from case_runner import RunCaseOptions, contains_npu as _contains_npu, run_case_with_trace
from manifest import EDGE_STUDY_SCHEMA_VERSION, load_json, project_dir, save_json
from results import CSV_FIELDNAMES, result_csv_row


def _run_case(
    manifest_path: Path,
    case: dict,
    iterations: int | None,
    warmup: int | None,
    measurement_mode: str,
) -> tuple[dict, object | None]:
    return run_case_with_trace(
        load_json(manifest_path),
        case,
        RunCaseOptions(
            manifest_path=manifest_path,
            case_name=case["name"],
            iterations=iterations,
            warmup=warmup,
            measurement_mode=measurement_mode,
            command_line=[sys.executable, *sys.argv],
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a benchmark matrix for the heterogeneous MoE example.")
    parser.add_argument("--manifest", default="default_manifest.json", help="Manifest path relative to this directory.")
    parser.add_argument(
        "--matrix",
        default="default_benchmark_matrix.json",
        help="Benchmark matrix JSON path relative to this directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/benchmarks/latest",
        help="Output directory for result JSON, CSV, traces, and report inputs.",
    )
    parser.add_argument("--iterations", type=int, default=None, help="Override iteration count for every case.")
    parser.add_argument("--warmup", type=int, default=None, help="Override warmup count for every case.")
    parser.add_argument(
        "--measurement-mode",
        choices=["cold", "warm", "both"],
        default="warm",
        help="Measure cold start, warm steady-state, or both. Validation is always untimed.",
    )
    parser.add_argument("--allow-npu", action="store_true", help="Run NPU-tagged cases in the matrix.")
    parser.add_argument("--require-torch", action="store_true", help="Fail if torch validation is unavailable or fails.")
    parser.add_argument(
        "--case-filter",
        nargs="*",
        default=[],
        help="Only run benchmark cases whose names exactly match one of these values.",
    )
    args = parser.parse_args()

    manifest_path = (project_dir() / args.manifest).resolve()
    matrix_path = (project_dir() / args.matrix).resolve()
    matrix = load_json(matrix_path)
    output_dir = (project_dir() / args.output_dir).resolve()
    traces_dir = output_dir / "traces"
    output_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)

    selected_cases = matrix["cases"]
    if args.case_filter:
        allowed_cases = set(args.case_filter)
        selected_cases = [case for case in selected_cases if case["name"] in allowed_cases]

    aggregate = {
        "schema_version": EDGE_STUDY_SCHEMA_VERSION,
        "measurement_mode": args.measurement_mode,
        "cases": [],
        "skipped": [],
    }
    for case in selected_cases:
        stage_backends = {
            "router": case["router_backend"],
            "expert0": case["expert0_backend"],
            "expert1": case["expert1_backend"],
            "aggregation": case["aggregation_backend"],
        }
        if _contains_npu(stage_backends) and not args.allow_npu:
            aggregate["skipped"].append({"case_name": case["name"], "reason": "NPU disabled for this run"})
            continue

        result, trace = _run_case(manifest_path, case, args.iterations, args.warmup, args.measurement_mode)
        if args.require_torch and not result["torch_validation"]["ok"]:
            raise SystemExit(f"torch validation failed for {case['name']}: {result['torch_validation']['message']}")

        case_dir = output_dir / case["name"]
        case_dir.mkdir(parents=True, exist_ok=True)
        save_json(case_dir / "results.json", result)
        save_json(case_dir / "stage_metrics.json", result["stage_metrics"])
        save_json(case_dir / "trace_summary.json", result["trace_summary"])
        save_json(case_dir / "transfer_summary.json", result["transfer_summary"])
        save_json(case_dir / "device_events.json", result["device_events"])
        save_json(case_dir / "npu_development.json", result["npu_development"])
        if trace is not None:
            trace.dump(traces_dir / f"{case['name']}.json")
        aggregate["cases"].append(result)

    save_json(output_dir / "summary.json", aggregate)

    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CSV_FIELDNAMES,
        )
        writer.writeheader()
        for result in aggregate["cases"]:
            writer.writerow(result_csv_row(result))

    print(f"Wrote benchmark matrix outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
