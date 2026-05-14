#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import copy
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_linear.compile import populate_artifacts
from llm_linear.external_baselines import (
    BASELINE_RUNNERS,
    EXTERNAL_BASELINE_SCHEMA_VERSION,
    apply_comparison_ratios,
    make_context,
    metadata_for_external_run,
    selected_baselines,
    write_outputs,
)
from llm_linear.manifest import (
    apply_case_to_manifest,
    load_json,
    resolve_package_path,
    save_json,
)
from llm_linear.results import (
    benchmark_runtime,
    build_case_result,
    validate_runtime,
)
from llm_linear.runtime import LinearRuntime
from llm_linear.suites import suite_workloads

AIR_REFERENCE_CASES = {
    "cpu": {"name": "cpu_only", "prefill_backend": "cpu", "decode_backend": "cpu"},
    "gpu": {"name": "gpu_only", "prefill_backend": "gpu", "decode_backend": "gpu"},
    "npu": {"name": "npu_only", "prefill_backend": "npu", "decode_backend": "npu"},
}


def _default_output_dir() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"artifacts/benchmarks/external_kernel_gap_{stamp}"


def _apply_decode_storage(
    manifest: dict[str, Any],
    *,
    storage: str,
    block_size: int,
    quant_axis: int,
) -> None:
    manifest.setdefault("weights", {})["decode"] = (
        {"storage": "bf16"}
        if storage == "bf16"
        else {
            "storage": storage,
            "block_size": int(block_size),
            "quant_axis": int(quant_axis),
        }
    )


def _run_air_reference(
    *,
    backend: str,
    manifest: dict[str, Any],
    manifest_path: Path,
    inputs,
    iterations: int,
    warmup: int,
    suite: str,
    workload_name: str,
    command_line: list[str],
) -> dict[str, Any]:
    reference_manifest = copy.deepcopy(manifest)
    if backend in {"gpu", "npu"}:
        reference_manifest.setdefault("runtime", {})["resident_weights"] = True
        needed = {"prefill": {backend}, "decode": {backend}}
        reference_manifest = populate_artifacts(
            reference_manifest,
            {backend},
            stage_backends=needed,
        )
    case = AIR_REFERENCE_CASES[backend]
    case_manifest = apply_case_to_manifest(
        reference_manifest, case, transfer_mode="host"
    )
    setup_ms = 0.0
    import time

    setup_start = time.perf_counter_ns()
    with LinearRuntime(case_manifest) as runtime:
        runtime.prepare()
        setup_ms = (time.perf_counter_ns() - setup_start) / 1_000_000.0
        timing = benchmark_runtime(
            runtime,
            inputs,
            iterations=iterations,
            warmup=warmup,
            measurement_mode="warm",
        )
        last_run, validation_ms = validate_runtime(runtime, inputs)
    metadata = metadata_for_external_run(manifest_path, case_manifest, command_line)
    return build_case_result(
        metadata=metadata,
        case_name=case["name"],
        manifest=case_manifest,
        iterations=iterations,
        requested_warmup=warmup,
        measurement_mode="warm",
        timing=timing,
        last_run=last_run,
        validation_ms=validation_ms,
        phase_timings_ms={"compile_load_setup_ms": setup_ms},
        suite=suite,
        workload_name=workload_name,
    )


def _reference_ms(result: dict[str, Any] | None) -> dict[str, float | None] | None:
    if result is None:
        return None
    if result.get("correctness", {}).get("validation_status") != "pass":
        return None
    return {
        "pipeline": float(result["latency_ms"]["mean"]),
        "prefill": float(result["stage_latency_ms"]["prefill"]["mean"]),
        "decode": float(result["stage_latency_ms"]["decode"]["mean"]),
    }


def _eligible_device_row(rows: list[dict[str, Any]], device: str) -> bool:
    return any(
        row.get("device") == device
        and row.get("fallback_status") == "native"
        and row.get("validation_status") == "pass"
        for row in rows
    )


def _run_air_references(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    inputs,
    iterations: int,
    warmup: int,
    suite: str,
    workload_name: str,
    command_line: list[str],
    need_gpu: bool,
    need_npu: bool,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    timings: dict[str, Any] = {"cpu": None, "gpu": None, "npu": None}
    diagnostics: dict[str, dict[str, Any]] = {}
    for backend, should_run in (
        ("cpu", True),
        ("gpu", need_gpu),
        ("npu", need_npu),
    ):
        if not should_run:
            diagnostics[backend] = {"status": "skipped"}
            continue
        try:
            result = _run_air_reference(
                backend=backend,
                manifest=manifest,
                manifest_path=manifest_path,
                inputs=inputs,
                iterations=iterations,
                warmup=warmup,
                suite=suite,
                workload_name=workload_name,
                command_line=command_line,
            )
        except Exception as exc:
            diagnostics[backend] = {"status": "failed", "reason": str(exc)}
            continue
        diagnostics[backend] = {
            "status": "pass",
            "case_name": result["case_name"],
            "validation_status": result.get("correctness", {}).get("validation_status"),
            "mean_end_to_end_ms": result["latency_ms"]["mean"],
            "mean_prefill_ms": result["stage_latency_ms"]["prefill"]["mean"],
            "mean_decode_ms": result["stage_latency_ms"]["decode"]["mean"],
        }
        timings[backend] = _reference_ms(result)
    return timings, diagnostics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run external/local kernel baselines for LLM-linear workloads."
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
        "--workload-filter",
        nargs="*",
        default=[],
        help="Only run workloads whose names contain any of these substrings.",
    )
    parser.add_argument(
        "--baseline-filter",
        nargs="*",
        default=[],
        help="Only run baselines whose names contain any of these substrings.",
    )
    parser.add_argument(
        "--decode-weight-storage",
        choices=["bf16", "int4", "uint4"],
        default="bf16",
        help="Decode GEMV weight storage for this run.",
    )
    parser.add_argument("--decode-quant-block-size", type=int, default=32)
    parser.add_argument("--decode-quant-axis", type=int, choices=[0, 1], default=0)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument(
        "--allow-npu",
        action="store_true",
        help="Permit NPU baselines and AIR NPU gap references.",
    )
    parser.add_argument(
        "--require-correctness",
        action="store_true",
        help="Fail if a supported baseline fails NumPy reference validation.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory relative to llm_linear/. Defaults to "
            "artifacts/benchmarks/external_kernel_gap_<timestamp>."
        ),
    )
    args = parser.parse_args(argv)
    if args.iterations is not None and args.iterations <= 0:
        raise SystemExit("--iterations must be positive")
    if args.warmup is not None and args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")

    try:
        baselines = selected_baselines(args.baseline_filter)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    base_manifest = load_json(resolve_package_path(args.manifest))
    _apply_decode_storage(
        base_manifest,
        storage=args.decode_weight_storage,
        block_size=args.decode_quant_block_size,
        quant_axis=args.decode_quant_axis,
    )
    matrix = load_json(resolve_package_path(args.matrix))
    workloads = suite_workloads(args.suite, base_manifest, matrix)
    if args.workload_filter:
        workloads = [
            workload
            for workload in workloads
            if any(fragment in workload["name"] for fragment in args.workload_filter)
        ]

    output_dir = resolve_package_path(args.output_dir or _default_output_dir())
    output_dir.mkdir(parents=True, exist_ok=True)
    command_line = (
        [sys.executable, *sys.argv]
        if argv is None
        else [sys.executable, "run_llm_linear_external_baselines.py", *argv]
    )

    summary: dict[str, Any] = {
        "schema_version": EXTERNAL_BASELINE_SCHEMA_VERSION,
        "decode_weight_storage": args.decode_weight_storage,
        "selected_baselines": baselines,
        "allow_npu": bool(args.allow_npu),
        "workloads": [],
    }
    all_rows: list[dict[str, Any]] = []

    for workload in workloads:
        manifest = workload["manifest"]
        iterations = (
            int(args.iterations)
            if args.iterations is not None
            else int(manifest["benchmark"]["iterations"])
        )
        warmup = (
            int(args.warmup)
            if args.warmup is not None
            else int(manifest["benchmark"]["warmup"])
        )
        workload_dir = output_dir / workload["suite"] / workload["name"]
        workload_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = workload_dir / "baseline_linear_manifest.json"
        save_json(manifest_path, manifest)
        ctx = make_context(
            suite=workload["suite"],
            workload=workload["name"],
            manifest=manifest,
            iterations=iterations,
            warmup=warmup,
            decode_weight_storage=args.decode_weight_storage,
            output_dir=output_dir,
            allow_npu=args.allow_npu,
        )

        rows: list[dict[str, Any]] = []
        for baseline in baselines:
            rows.extend(BASELINE_RUNNERS[baseline](ctx))

        need_gpu = _eligible_device_row(rows, "gpu")
        need_npu = bool(args.allow_npu and _eligible_device_row(rows, "npu"))
        air_timings, air_diagnostics = _run_air_references(
            manifest=manifest,
            manifest_path=manifest_path,
            inputs=ctx.inputs,
            iterations=iterations,
            warmup=warmup,
            suite=workload["suite"],
            workload_name=workload["name"],
            command_line=command_line,
            need_gpu=need_gpu,
            need_npu=need_npu,
        )
        cpu_ms = air_timings.get("cpu")
        if cpu_ms is None:
            cpu_row = next(
                (
                    row
                    for row in rows
                    if row.get("baseline_name") == "cpu_numpy"
                    and row.get("validation_status") == "pass"
                ),
                None,
            )
            if cpu_row is not None:
                cpu_ms = {
                    "pipeline": cpu_row.get("mean_end_to_end_ms"),
                    "prefill": cpu_row.get("mean_prefill_ms"),
                    "decode": cpu_row.get("mean_decode_ms"),
                }

        apply_comparison_ratios(
            rows,
            cpu_ms=cpu_ms,
            air_gpu_ms=air_timings.get("gpu"),
            air_npu_ms=air_timings.get("npu"),
        )
        for row in rows:
            if args.require_correctness and row.get("fallback_status") == "native":
                if row.get("validation_status") != "pass":
                    raise SystemExit(
                        "correctness validation failed for "
                        f"{row['baseline_name']} on {workload['name']}: "
                        f"status={row.get('validation_status')}, "
                        f"output_max_abs_error={row.get('output_max_abs_error')}"
                    )
            save_json(
                workload_dir / f"{row['baseline_name']}_{row['scope']}.json",
                row,
            )
        workload_summary = {
            "suite": workload["suite"],
            "name": workload["name"],
            "shape": workload["shape"],
            "manifest": str(manifest_path),
            "iterations": iterations,
            "warmup": warmup,
            "air_references": air_diagnostics,
            "rows": rows,
        }
        save_json(workload_dir / "summary.json", workload_summary)
        summary["workloads"].append(workload_summary)
        all_rows.extend(rows)

    write_outputs(output_dir, summary=summary, rows=all_rows)
    print(f"Wrote llm_linear external baseline outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
