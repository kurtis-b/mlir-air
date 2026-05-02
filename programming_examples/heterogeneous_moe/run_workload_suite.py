#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import copy
import csv
import sys
from typing import Any

from case_runner import RunCaseOptions, contains_npu, run_case_with_trace
from compile import populate_artifacts
from manifest import EDGE_STUDY_SCHEMA_VERSION, load_json, project_dir, save_json
from reference import DEFAULT_INPUT_SCALE, DEFAULT_ROUTING_PROFILE, DEFAULT_WEIGHT_SCALE
from reports import suite_summary_markdown
from results import correctness_failure_message
from workloads import required_backends, routing_stats, suite_workloads


def _csv_row(
    workload: dict[str, Any], manifest: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    model_preset = workload["model_preset"]
    return {
        "suite": workload["suite"],
        "workload": workload["name"],
        "model_id": model_preset["model_id"],
        "model_class": model_preset["model_class"],
        "weight_storage": model_preset["weight_storage"],
        "compute_dtype": model_preset["compute_dtype"],
        "num_experts": model_preset["num_experts"],
        "active_experts": model_preset["active_experts"],
        "routing_profile": workload["routing_profile"],
        "context_length": workload["context_length"],
        "routed_tokens": workload["routed_tokens"],
        "kernel_chunk_tokens": workload["kernel_chunk_tokens"],
        "input_scale": workload["input_scale"],
        "batch_tokens": manifest["model"]["batch_tokens"],
        "hidden_size": manifest["model"]["hidden_size"],
        "ffn_size": manifest["model"]["ffn_size"],
        "case_name": result["case_name"],
        "router_mode": result["router_mode"],
        "router_backend": result["stage_backends"]["router"],
        "expert0_backend": result["stage_backends"]["expert0"],
        "expert1_backend": result["stage_backends"]["expert1"],
        "aggregation_backend": result["stage_backends"]["aggregation"],
        "transfer_mode": result["transfer_mode"],
        "mean_latency_ms": result["latency_ms"]["mean"],
        "p50_latency_ms": result["latency_ms"]["p50"],
        "p95_latency_ms": result["latency_ms"]["p95"],
        "max_abs_error": result["stage_metrics"]["output"]["max_abs_error"],
        "torch_ok": result["torch_validation"]["ok"],
        "expert_overlap_us": result["trace_summary"]["overlap"]["expert0_expert1_us"],
        "transfer_bytes": result["transfer_summary"]["total_bytes"],
        "transfer_elapsed_us": result["transfer_summary"]["total_elapsed_us"],
        "npu_executed": result["npu_development"]["executed"],
    }


def _workload_summary(
    workload: dict[str, Any], manifest_path: str, manifest: dict[str, Any]
) -> dict[str, Any]:
    workload_cfg = manifest.get("workload", {})
    return {
        "suite": workload["suite"],
        "name": workload["name"],
        "model": manifest["model"],
        "model_preset": {
            "model_id": workload_cfg.get("model_id"),
            "model_class": workload_cfg.get("model_class"),
            "weight_storage": workload_cfg.get("weight_storage"),
            "compute_dtype": workload_cfg.get("compute_dtype"),
            "num_experts": workload_cfg.get("num_experts"),
            "active_experts": workload_cfg.get("active_experts"),
            "shared_expert_ffn_size": workload_cfg.get("shared_expert_ffn_size"),
        },
        "routing_profile": workload_cfg.get("routing_profile", DEFAULT_ROUTING_PROFILE),
        "context_length": workload_cfg.get("context_length"),
        "routed_tokens": int(
            workload_cfg.get("routed_tokens", manifest["model"]["batch_tokens"])
        ),
        "kernel_chunk_tokens": int(manifest["model"]["batch_tokens"]),
        "input_scale": float(
            manifest.get("inputs", {}).get("scale", DEFAULT_INPUT_SCALE)
        ),
        "weight_scale": float(
            manifest.get("weights", {}).get("scale", DEFAULT_WEIGHT_SCALE)
        ),
        "route_stats": routing_stats(manifest),
        "cases": [],
        "skipped": [],
        "manifest": manifest_path,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run expanded workload suites for the heterogeneous MoE example."
    )
    parser.add_argument(
        "--manifest",
        default="default_manifest.json",
        help="Base manifest relative to this directory.",
    )
    parser.add_argument(
        "--matrix",
        default="default_benchmark_matrix.json",
        help="Base benchmark matrix relative to this directory.",
    )
    parser.add_argument(
        "--suite",
        nargs="+",
        choices=["shape_sweep", "routing_sweep", "model_presets"],
        default=["shape_sweep", "routing_sweep", "model_presets"],
        help="Workload suites to run.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/benchmarks/workload_suites/latest",
        help="Output directory relative to this directory.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override iteration count for every case.",
    )
    parser.add_argument(
        "--warmup", type=int, default=None, help="Override warmup count for every case."
    )
    parser.add_argument(
        "--measurement-mode",
        choices=["cold", "warm", "both"],
        default="warm",
        help="Measure cold start, warm steady-state, or both. Validation is always untimed.",
    )
    parser.add_argument(
        "--allow-npu", action="store_true", help="Run NPU-tagged cases in each suite."
    )
    parser.add_argument(
        "--require-correctness",
        action="store_true",
        help="Fail if final output validation is outside dtype tolerances.",
    )
    parser.add_argument(
        "--require-torch",
        action="store_true",
        help="Fail if torch validation is unavailable or fails.",
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

    root = project_dir()
    base_manifest = load_json((root / args.manifest).resolve())
    matrix = load_json((root / args.matrix).resolve())
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

    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    suite_summary: dict[str, Any] = {
        "schema_version": EDGE_STUDY_SCHEMA_VERSION,
        "measurement_mode": args.measurement_mode,
        "workloads": [],
    }
    csv_rows: list[dict[str, Any]] = []
    artifact_cache: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}

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
        manifest_path = suite_dir / "compiled_manifest.json"
        save_json(manifest_path, manifest)
        case_results_dir = suite_dir / "cases"
        case_results_dir.mkdir(parents=True, exist_ok=True)

        workload_summary = _workload_summary(workload, str(manifest_path), manifest)

        for case in workload["cases"]:
            stage_backends = {
                "router": case["router_backend"],
                "expert0": case["expert0_backend"],
                "expert1": case["expert1_backend"],
                "aggregation": case["aggregation_backend"],
            }
            if contains_npu(stage_backends) and not args.allow_npu:
                workload_summary["skipped"].append(
                    {"case_name": case["name"], "reason": "NPU disabled for this run"}
                )
                continue

            result, _trace = run_case_with_trace(
                manifest,
                case,
                RunCaseOptions(
                    manifest_path=manifest_path,
                    case_name=case["name"],
                    iterations=args.iterations,
                    warmup=args.warmup,
                    measurement_mode=args.measurement_mode,
                    command_line=(
                        [sys.executable, *sys.argv]
                        if argv is None
                        else [sys.executable, "run_workload_suite.py", *argv]
                    ),
                ),
            )
            if args.require_torch and not result["torch_validation"]["ok"]:
                raise SystemExit(
                    f"torch validation failed for {case['name']}: {result['torch_validation']['message']}"
                )
            if args.require_correctness:
                failure = correctness_failure_message(result)
                if failure is not None:
                    raise SystemExit(
                        f"correctness validation failed for {case['name']}: {failure}"
                    )
            save_json(case_results_dir / f"{case['name']}.json", result)
            workload_summary["cases"].append(result)
            csv_rows.append(_csv_row(workload_summary, manifest, result))

        save_json(suite_dir / "summary.json", workload_summary)
        suite_summary["workloads"].append(workload_summary)

    save_json(output_dir / "summary.json", suite_summary)

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "suite",
                "workload",
                "model_id",
                "model_class",
                "weight_storage",
                "compute_dtype",
                "num_experts",
                "active_experts",
                "routing_profile",
                "context_length",
                "routed_tokens",
                "kernel_chunk_tokens",
                "input_scale",
                "batch_tokens",
                "hidden_size",
                "ffn_size",
                "case_name",
                "router_mode",
                "router_backend",
                "expert0_backend",
                "expert1_backend",
                "aggregation_backend",
                "transfer_mode",
                "mean_latency_ms",
                "p50_latency_ms",
                "p95_latency_ms",
                "max_abs_error",
                "torch_ok",
                "expert_overlap_us",
                "transfer_bytes",
                "transfer_elapsed_us",
                "npu_executed",
            ],
        )
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    (output_dir / "report.md").write_text(
        suite_summary_markdown(suite_summary, "Heterogeneous MoE Workload Suites")
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote workload suite outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
