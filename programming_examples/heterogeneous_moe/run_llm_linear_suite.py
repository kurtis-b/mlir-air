#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import copy
import csv
import sys
import time
from pathlib import Path
from typing import Any

from llm_linear.compile import populate_artifacts
from llm_linear.manifest import (
    SCHEMA_VERSION,
    apply_case_to_manifest,
    collect_run_metadata,
    load_json,
    resolve_package_path,
    save_json,
)
from llm_linear.reports import suite_summary_markdown
from llm_linear.results import (
    CSV_FIELDNAMES,
    benchmark_runtime,
    build_case_result,
    correctness_failure_message,
    generate_inputs,
    result_csv_row,
    validate_runtime,
)
from llm_linear.runtime import LinearRuntime
from llm_linear.schema import case_stage_backends, contains_npu, required_backends
from llm_linear.suites import suite_workloads


def _run_case(
    manifest: dict[str, Any],
    manifest_path: Path,
    case: dict[str, Any],
    *,
    suite: str,
    workload_name: str,
    iterations: int,
    warmup: int,
    measurement_mode: str,
    command_line: list[str],
) -> dict[str, Any]:
    setup_start = time.perf_counter_ns()
    with LinearRuntime(manifest) as runtime:
        runtime.prepare()
        setup_ms = (time.perf_counter_ns() - setup_start) / 1_000_000.0
        inputs, input_phase, _input_scale = generate_inputs(runtime, manifest)
        timing = benchmark_runtime(
            runtime,
            inputs,
            iterations=iterations,
            warmup=warmup,
            measurement_mode=measurement_mode,
        )
        last_run, validation_ms = validate_runtime(runtime, inputs)
    metadata = collect_run_metadata(manifest_path, manifest, command_line=command_line)
    return build_case_result(
        metadata=metadata,
        case_name=case["name"],
        manifest=manifest,
        iterations=iterations,
        requested_warmup=warmup,
        measurement_mode=measurement_mode,
        timing=timing,
        last_run=last_run,
        validation_ms=validation_ms,
        phase_timings_ms={"compile_load_setup_ms": setup_ms, **input_phase},
        suite=suite,
        workload_name=workload_name,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run LLM-linear shape suites for heterogeneous Ryzen execution."
    )
    parser.add_argument(
        "--manifest",
        default="default_linear_manifest.json",
        help="Base manifest path relative to llm_linear/.",
    )
    parser.add_argument(
        "--matrix",
        default="default_linear_matrix.json",
        help="Benchmark matrix path relative to llm_linear/.",
    )
    parser.add_argument(
        "--suite",
        nargs="+",
        choices=["tiny_ci", "medium", "llm_like"],
        default=["tiny_ci"],
        help="Shape suites to run.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/benchmarks/latest",
        help="Output directory relative to llm_linear/.",
    )
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument(
        "--measurement-mode",
        choices=["cold", "warm", "both"],
        default="warm",
        help="Measure cold start, warm steady-state, or both.",
    )
    parser.add_argument(
        "--transfer-mode",
        choices=["host", "direct"],
        default="host",
        help="Transfer mode for all selected cases.",
    )
    parser.add_argument(
        "--allow-npu", action="store_true", help="Run NPU-tagged cases in each suite."
    )
    parser.add_argument(
        "--require-correctness",
        action="store_true",
        help="Fail if NumPy reference validation is outside dtype tolerances.",
    )
    parser.add_argument(
        "--workload-filter",
        nargs="*",
        default=[],
        help="Only run workloads whose names contain any of these substrings.",
    )
    parser.add_argument(
        "--case-filter",
        nargs="*",
        default=[],
        help="Only run benchmark cases whose names exactly match one of these values.",
    )
    args = parser.parse_args(argv)

    if args.transfer_mode == "direct":
        raise SystemExit(
            "transfer_mode=direct is unsupported in llm_linear Milestone 1; "
            "host-staged mixed paths are the only mixed paths in this implementation"
        )

    base_manifest = load_json(resolve_package_path(args.manifest))
    matrix = load_json(resolve_package_path(args.matrix))
    workloads = suite_workloads(args.suite, base_manifest, matrix)
    if args.workload_filter:
        workloads = [
            workload
            for workload in workloads
            if any(fragment in workload["name"] for fragment in args.workload_filter)
        ]
    if args.case_filter:
        allowed = set(args.case_filter)
        for workload in workloads:
            workload["cases"] = [
                case for case in workload["cases"] if case["name"] in allowed
            ]

    output_dir = resolve_package_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "measurement_mode": args.measurement_mode,
        "transfer_mode": args.transfer_mode,
        "workloads": [],
    }
    csv_rows: list[dict[str, Any]] = []
    artifact_cache: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    command_line = (
        [sys.executable, *sys.argv]
        if argv is None
        else [sys.executable, "run_llm_linear_suite.py", *argv]
    )

    for workload in workloads:
        manifest = workload["manifest"]
        needed_backends = required_backends(workload["cases"], args.allow_npu)
        if needed_backends:
            cache_key = (manifest["paths"]["artifacts"], tuple(sorted(needed_backends)))
            cached_artifacts = artifact_cache.get(cache_key)
            if cached_artifacts is None:
                manifest = populate_artifacts(manifest, needed_backends)
                artifact_cache[cache_key] = copy.deepcopy(manifest["artifacts"])
            else:
                manifest["artifacts"] = copy.deepcopy(cached_artifacts)

        suite_dir = output_dir / workload["suite"] / workload["name"]
        suite_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = suite_dir / "compiled_linear_manifest.json"
        save_json(manifest_path, manifest)
        cases_dir = suite_dir / "cases"
        cases_dir.mkdir(parents=True, exist_ok=True)
        workload_summary = {
            "suite": workload["suite"],
            "name": workload["name"],
            "shape": workload["shape"],
            "manifest": str(manifest_path),
            "cases": [],
            "skipped": [],
        }

        for case in workload["cases"]:
            stage_backends = case_stage_backends(case)
            if contains_npu(stage_backends) and not args.allow_npu:
                workload_summary["skipped"].append(
                    {"case_name": case["name"], "reason": "NPU disabled for this run"}
                )
                continue

            case_manifest = apply_case_to_manifest(
                copy.deepcopy(manifest), case, transfer_mode=args.transfer_mode
            )
            iterations = (
                int(args.iterations)
                if args.iterations is not None
                else int(case_manifest["benchmark"]["iterations"])
            )
            warmup = (
                int(args.warmup)
                if args.warmup is not None
                else int(case_manifest["benchmark"]["warmup"])
            )
            result = _run_case(
                case_manifest,
                manifest_path,
                case,
                suite=workload["suite"],
                workload_name=workload["name"],
                iterations=iterations,
                warmup=warmup,
                measurement_mode=args.measurement_mode,
                command_line=command_line,
            )
            if args.require_correctness:
                failure = correctness_failure_message(result)
                if failure is not None:
                    raise SystemExit(
                        f"correctness validation failed for {case['name']}: {failure}"
                    )
            save_json(cases_dir / f"{case['name']}.json", result)
            workload_summary["cases"].append(result)
            csv_rows.append(result_csv_row(result))

        save_json(suite_dir / "summary.json", workload_summary)
        summary["workloads"].append(workload_summary)

    save_json(output_dir / "summary.json", summary)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    (output_dir / "report.md").write_text(
        suite_summary_markdown(summary, "LLM-Linear Heterogeneous Suite") + "\n",
        encoding="utf-8",
    )
    print(f"Wrote llm_linear suite outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
