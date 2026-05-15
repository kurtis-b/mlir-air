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
from llm_linear.direct_bridge import (
    bootstrap_direct_bridge,
    cleanup_direct_bridge,
    probe_direct_bridge,
    reexec_with_direct_bridge_bootstrap,
)

if __name__ == "__main__":
    reexec_with_direct_bridge_bootstrap(__file__, sys.argv[1:])
bootstrap_direct_bridge()

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
from llm_linear.schema import case_stage_backends, contains_npu
from llm_linear.suites import suite_workloads
from llm_linear.transfer import DirectTransferUnsupported


def _required_artifact_stage_backends(
    cases: list[dict[str, Any]], allow_npu: bool
) -> dict[str, set[str]]:
    needed: dict[str, set[str]] = {"prefill": set(), "decode": set()}
    for case in cases:
        for stage, backend in case_stage_backends(case).items():
            if backend == "gpu" or (backend == "npu" and allow_npu):
                needed[stage].add(backend)
    return {stage: backends for stage, backends in needed.items() if backends}


def _artifact_cache_key(
    manifest: dict[str, Any], stage_backends: dict[str, set[str]]
) -> tuple[str, str, tuple[tuple[str, tuple[str, ...]], ...]]:
    decode = manifest.get("weights", {}).get("decode", {})
    decode_key = (
        str(decode.get("storage", "bf16")) if isinstance(decode, dict) else "bf16"
    )
    if isinstance(decode, dict) and decode_key in {"int4", "uint4"}:
        decode_key += (
            f":b{int(decode.get('block_size', 32))}"
            f":axis{int(decode.get('quant_axis', 0))}"
        )
    return (
        manifest["paths"]["artifacts"],
        decode_key,
        tuple(
            (stage, tuple(sorted(backends)))
            for stage, backends in sorted(stage_backends.items())
        ),
    )


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
        default=None,
        help="Override transfer mode for all selected cases.",
    )
    parser.add_argument(
        "--decode-weight-storage",
        choices=["bf16", "int4", "uint4"],
        default="bf16",
        help="Decode GEMV weight storage for this run.",
    )
    parser.add_argument(
        "--decode-quant-block-size",
        type=int,
        default=32,
        help="Block size for int4/uint4 decode weight quantization.",
    )
    parser.add_argument(
        "--decode-quant-axis",
        type=int,
        choices=[0, 1],
        default=0,
        help="Quantization axis for packed decode weights.",
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

    base_manifest = load_json(resolve_package_path(args.manifest))
    if args.decode_weight_storage != "bf16":
        base_manifest.setdefault("weights", {})["decode"] = {
            "storage": args.decode_weight_storage,
            "block_size": int(args.decode_quant_block_size),
            "quant_axis": int(args.decode_quant_axis),
        }
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
    if args.transfer_mode == "direct":
        for workload in workloads:
            for case in workload["cases"]:
                backends = case_stage_backends(case)
                if set(backends.values()) != {"gpu", "npu"}:
                    raise SystemExit(
                        "transfer_mode=direct only supports GPU/NPU mixed cases; "
                        f"{case['name']} uses {backends['prefill']}->{backends['decode']}"
                    )
    direct_cases_selected = []
    for workload in workloads:
        for case in workload["cases"]:
            backends = case_stage_backends(case)
            effective_transfer = args.transfer_mode or case.get("transfer_mode", "host")
            if effective_transfer == "direct" and (
                args.allow_npu or not contains_npu(backends)
            ):
                direct_cases_selected.append(case["name"])
    if direct_cases_selected:
        status = probe_direct_bridge()
        if not status.available:
            raise SystemExit(
                "direct GPU/NPU cases require a native bridge before compilation: "
                f"{status.diagnostic}"
            )

    output_dir = resolve_package_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "measurement_mode": args.measurement_mode,
        "transfer_mode": args.transfer_mode or "case",
        "decode_weight_storage": args.decode_weight_storage,
        "workloads": [],
    }
    csv_rows: list[dict[str, Any]] = []
    artifact_cache: dict[
        tuple[str, str, tuple[tuple[str, tuple[str, ...]], ...]], dict[str, Any]
    ] = {}
    command_line = (
        [sys.executable, *sys.argv]
        if argv is None
        else [sys.executable, "run_llm_linear_suite.py", *argv]
    )

    for workload in workloads:
        manifest = workload["manifest"]
        needed_artifacts = _required_artifact_stage_backends(
            workload["cases"], args.allow_npu
        )
        if needed_artifacts:
            needed_backends = {
                backend
                for backends in needed_artifacts.values()
                for backend in backends
            }
            cache_key = _artifact_cache_key(manifest, needed_artifacts)
            cached_artifacts = artifact_cache.get(cache_key)
            if cached_artifacts is None:
                manifest = populate_artifacts(
                    manifest,
                    needed_backends,
                    stage_backends=needed_artifacts,
                )
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
            try:
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
            except DirectTransferUnsupported as exc:
                raise SystemExit(str(exc)) from exc
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
        cleanup_direct_bridge()

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
