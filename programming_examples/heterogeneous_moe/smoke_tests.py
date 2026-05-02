#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

from case_runner import RunCaseOptions, contains_npu, run_case_with_trace
from compile import populate_artifacts
from manifest import load_json, project_dir, save_json
from results import correctness_failure_message
from test_golden_air import main as golden_air_main


CPU_CASES = ["cpu_top1", "cpu_top2"]
GPU_CASES = ["gpu_top1", "gpu_top2"]
MIXED_GPU_CASES = [
    "cpu_router_gpu_experts_cpu_agg_top2",
    "gpu_router_cpu_experts_gpu_agg_top2",
]
NPU_CASES = [
    "npu_top1",
    "npu_top2",
    "cpu_router_npu_experts_cpu_agg_top2",
    "cpu_router_npu_gpu_experts_cpu_agg_top2",
]


def _case_backends(case: dict[str, Any]) -> dict[str, str]:
    return {
        "router": case["router_backend"],
        "expert0": case["expert0_backend"],
        "expert1": case["expert1_backend"],
        "aggregation": case["aggregation_backend"],
    }


def _case_uses(case: dict[str, Any], backend: str) -> bool:
    return backend in _case_backends(case).values()


def _matrix_cases(matrix: dict[str, Any], names: list[str]) -> list[dict[str, Any]]:
    by_name = {case["name"]: case for case in matrix["cases"]}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise SystemExit(f"matrix is missing expected cases: {', '.join(missing)}")
    return [by_name[name] for name in names]


def _expanded_lanes(lanes: list[str]) -> set[str]:
    expanded: set[str] = set()
    for lane in lanes:
        if lane == "ci":
            expanded.update({"golden", "cpu"})
        elif lane == "gpu-all":
            expanded.update({"golden", "cpu", "gpu-compile", "gpu", "mixed-gpu"})
        elif lane == "all":
            expanded.update({"golden", "cpu", "gpu-compile", "gpu", "mixed-gpu", "npu"})
        else:
            expanded.add(lane)
    return expanded


def _save_result(output_dir: Path, result: dict[str, Any]) -> None:
    case_dir = output_dir / result["case_name"]
    case_dir.mkdir(parents=True, exist_ok=True)
    save_json(case_dir / "results.json", result)
    save_json(case_dir / "stage_metrics.json", result["stage_metrics"])


def _run_cases(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    cases: list[dict[str, Any]],
    iterations: int,
    warmup: int,
    measurement_mode: str,
    output_dir: Path,
    require_correctness: bool,
    require_torch: bool,
) -> list[str]:
    passed: list[str] = []
    for case in cases:
        result, _trace = run_case_with_trace(
            manifest,
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
        if require_torch and not result["torch_validation"]["ok"]:
            raise SystemExit(f"torch validation failed for {case['name']}: {result['torch_validation']['message']}")
        if require_correctness:
            failure = correctness_failure_message(result)
            if failure is not None:
                raise SystemExit(f"correctness validation failed for {case['name']}: {failure}")
        _save_result(output_dir, result)
        print(f"PASS {case['name']}")
        passed.append(case["name"])
    return passed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run lightweight heterogeneous MoE smoke-test lanes.")
    parser.add_argument(
        "--lane",
        nargs="+",
        choices=["golden", "cpu", "gpu-compile", "gpu", "mixed-gpu", "npu", "ci", "gpu-all", "all"],
        default=["ci"],
        help="Smoke-test lanes to run. The default 'ci' lane runs golden AIR and CPU top1/top2.",
    )
    parser.add_argument("--manifest", default="default_manifest.json", help="Base manifest path relative to this directory.")
    parser.add_argument("--matrix", default="default_benchmark_matrix.json", help="Benchmark matrix path relative to this directory.")
    parser.add_argument(
        "--compiled-manifest-out",
        default="artifacts/compiled_manifest.json",
        help="Compiled manifest output path relative to this directory.",
    )
    parser.add_argument(
        "--use-existing-compiled-manifest",
        action="store_true",
        help="Use --compiled-manifest-out for non-CPU lanes instead of compiling artifacts first.",
    )
    parser.add_argument("--output-dir", default="artifacts/smoke_tests/latest", help="Output directory relative to this directory.")
    parser.add_argument("--iterations", type=int, default=1, help="Iterations for benchmark smoke cases.")
    parser.add_argument("--warmup", type=int, default=0, help="Warmup iterations for benchmark smoke cases.")
    parser.add_argument(
        "--measurement-mode",
        choices=["cold", "warm", "both"],
        default="warm",
        help="Measure cold start, warm steady-state, or both. Validation is always untimed.",
    )
    parser.add_argument("--allow-npu", action="store_true", help="Allow NPU smoke cases and artifact compilation.")
    parser.add_argument("--no-require-correctness", action="store_true", help="Do not fail on final output correctness.")
    parser.add_argument("--require-torch", action="store_true", help="Fail if torch validation is unavailable or fails.")
    args = parser.parse_args(argv)

    lanes = _expanded_lanes(args.lane)
    if "npu" in lanes and not args.allow_npu:
        raise SystemExit("NPU smoke lanes require --allow-npu")

    root = project_dir()
    manifest_path = (root / args.manifest).resolve()
    compiled_manifest_path = (root / args.compiled_manifest_out).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_manifest = load_json(manifest_path)
    matrix = load_json((root / args.matrix).resolve())

    passed: list[str] = []
    if "golden" in lanes:
        if golden_air_main() != 0:
            return 1
        passed.append("golden_air")

    case_names: list[str] = []
    if "cpu" in lanes:
        case_names.extend(CPU_CASES)
    if "gpu" in lanes:
        case_names.extend(GPU_CASES)
    if "mixed-gpu" in lanes:
        case_names.extend(MIXED_GPU_CASES)
    if "npu" in lanes:
        case_names.extend(NPU_CASES)

    selected_cases = _matrix_cases(matrix, case_names)
    compiled_manifest = copy.deepcopy(base_manifest)
    compiled_manifest_for_cases = compiled_manifest
    needs_compiled_manifest = any(_case_uses(case, "gpu") or _case_uses(case, "npu") for case in selected_cases)
    if "gpu-compile" in lanes or needs_compiled_manifest:
        if args.use_existing_compiled_manifest:
            compiled_manifest = load_json(compiled_manifest_path)
            compiled_manifest_for_cases = compiled_manifest
            print(f"PASS compiled_manifest: {compiled_manifest_path}")
            passed.append("compiled_manifest")
        else:
            backends = {"gpu"} if "gpu-compile" in lanes or any(_case_uses(case, "gpu") for case in selected_cases) else set()
            if any(_case_uses(case, "npu") for case in selected_cases):
                backends.add("npu")
            compiled_manifest = populate_artifacts(copy.deepcopy(base_manifest), backends)
            save_json(compiled_manifest_path, compiled_manifest)
            print(f"PASS compile_{'_'.join(sorted(backends))}: {compiled_manifest_path}")
            passed.append(f"compile_{'_'.join(sorted(backends))}")
            compiled_manifest_for_cases = compiled_manifest

    cpu_cases = [case for case in selected_cases if not _case_uses(case, "gpu") and not _case_uses(case, "npu")]
    device_cases = [case for case in selected_cases if case not in cpu_cases]
    if cpu_cases:
        passed.extend(
            _run_cases(
                manifest=base_manifest,
                manifest_path=manifest_path,
                cases=cpu_cases,
                iterations=args.iterations,
                warmup=args.warmup,
                measurement_mode=args.measurement_mode,
                output_dir=output_dir,
                require_correctness=not args.no_require_correctness,
                require_torch=args.require_torch,
            )
        )
    if device_cases:
        for case in device_cases:
            if contains_npu(_case_backends(case)) and not args.allow_npu:
                raise SystemExit(f"{case['name']} requires --allow-npu")
        passed.extend(
            _run_cases(
                manifest=compiled_manifest_for_cases,
                manifest_path=compiled_manifest_path,
                cases=device_cases,
                iterations=args.iterations,
                warmup=args.warmup,
                measurement_mode=args.measurement_mode,
                output_dir=output_dir,
                require_correctness=not args.no_require_correctness,
                require_torch=args.require_torch,
            )
        )

    print(f"Completed {len(passed)} smoke checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
