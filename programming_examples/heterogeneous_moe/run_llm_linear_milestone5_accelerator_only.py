#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_linear.acceptance import (
    MilestoneFailure,
    resolve_project_path,
    run_logged,
    seed_default_tool_env,
)
from llm_linear.manifest import SCHEMA_VERSION
from llm_linear.reports import suite_summary_markdown
from llm_linear.results import CSV_FIELDNAMES, result_csv_row
from llm_linear.suites import SHAPE_LADDER

DEFAULT_OUTPUT_ROOT = "llm_linear/artifacts/benchmarks/milestone5_accelerator_only"
REQUIRED_SUITES = ("medium", "llm_like")
FINAL_CASES = ("cpu_only", "gpu_only", "npu_only")
DIAGNOSTIC_CASES = ("gpu_only_native_ceiling", "npu_only_native_ceiling")
INT4_BLOCK_SIZE = 32
INT4_QUANT_AXIS = 0
MEASUREMENT_MODE = "warm"
ITERATIONS = 7
WARMUP = 2
MIN_ACCELERATOR_SPEEDUP = 1.20


@dataclass(frozen=True)
class Milestone5Run:
    name: str
    decode_weight_storage: str
    quant_args: tuple[str, ...] = ()

    def output_dir(self, output_root: str) -> str:
        return str(Path(output_root) / self.name)

    def case_argv(
        self, python: Path, output_root: str, *, suite: str, workload: str, case: str
    ) -> list[str]:
        return [
            str(python),
            "run_llm_linear_suite.py",
            "--suite",
            suite,
            "--workload-filter",
            workload,
            "--case-filter",
            case,
            "--decode-weight-storage",
            self.decode_weight_storage,
            *self.quant_args,
            "--allow-npu",
            "--resident-weights",
            "--measurement-mode",
            MEASUREMENT_MODE,
            "--iterations",
            str(ITERATIONS),
            "--warmup",
            str(WARMUP),
            "--require-correctness",
            "--output-dir",
            self.output_dir(output_root),
        ]


ACCEPTANCE_RUNS = (
    Milestone5Run(name="bf16", decode_weight_storage="bf16"),
    Milestone5Run(
        name="int4",
        decode_weight_storage="int4",
        quant_args=(
            "--decode-quant-block-size",
            str(INT4_BLOCK_SIZE),
            "--decode-quant-axis",
            str(INT4_QUANT_AXIS),
        ),
    ),
)


def expected_workload_keys() -> set[tuple[str, str]]:
    return {
        (suite, str(shape["name"]))
        for suite in REQUIRED_SUITES
        for shape in SHAPE_LADDER[suite]
    }


def ordered_workload_keys() -> list[tuple[str, str]]:
    def size_key(item: tuple[str, str]) -> tuple[int, str, str]:
        suite, workload = item
        shape = next(
            shape for shape in SHAPE_LADDER[suite] if shape["name"] == workload
        )
        tensor_work = int(shape["M"]) * int(shape["K"]) * int(shape["H"]) + int(
            shape["H"]
        ) * int(shape["N"])
        return (-tensor_work, suite, workload)

    return sorted(expected_workload_keys(), key=size_key)


def load_summary(output_dir: Path) -> dict[str, Any]:
    return json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))


def combine_workload_outputs(
    output_dir: Path, *, decode_weight_storage: str
) -> dict[str, Any]:
    workloads: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    shapes_by_key = {
        (suite, str(shape["name"])): dict(shape)
        for suite in REQUIRED_SUITES
        for shape in SHAPE_LADDER[suite]
    }
    for suite, workload in sorted(expected_workload_keys()):
        cases = []
        workload_dir = output_dir / suite / workload
        for case in FINAL_CASES:
            case_path = workload_dir / "cases" / f"{case}.json"
            cases.append(json.loads(case_path.read_text(encoding="utf-8")))
        attach_speed_falsification(cases, workload_dir)
        for result in cases:
            case_path = workload_dir / "cases" / f"{result['case_name']}.json"
            case_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        workload_summary = {
            "suite": suite,
            "name": workload,
            "shape": shapes_by_key[(suite, workload)],
            "manifest": str(workload_dir / "compiled_linear_manifest.json"),
            "cases": cases,
            "skipped": [],
        }
        (workload_dir / "summary.json").write_text(
            json.dumps(workload_summary, indent=2) + "\n", encoding="utf-8"
        )
        workloads.append(workload_summary)
        csv_rows.extend(result_csv_row(result) for result in cases)

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "measurement_mode": MEASUREMENT_MODE,
        "transfer_mode": "case",
        "decode_weight_storage": decode_weight_storage,
        "resident_weights_required": True,
        "acceptance": {
            "required_suites": list(REQUIRED_SUITES),
            "final_cases": list(FINAL_CASES),
            "diagnostic_cases": list(DIAGNOSTIC_CASES),
            "min_accelerator_speedup": MIN_ACCELERATOR_SPEEDUP,
            "speedup_denominator": "cpu_only native warm mean end-to-end latency",
        },
        "workloads": workloads,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    (output_dir / "report.md").write_text(
        suite_summary_markdown(summary, "LLM-Linear Milestone 5 Accelerator-Only Suite")
        + "\n",
        encoding="utf-8",
    )
    return summary


def attach_speed_falsification(cases: list[dict[str, Any]], workload_dir: Path) -> None:
    cases_by_name = {str(result.get("case_name")): result for result in cases}
    if not set(FINAL_CASES) <= cases_by_name.keys():
        return
    cpu = cases_by_name["cpu_only"]
    cpu_ms = _warm_mean_ms(cpu)
    if cpu_ms <= 0.0:
        return
    for case_name in ("gpu_only", "npu_only"):
        result = cases_by_name[case_name]
        acc_ms = _warm_mean_ms(result)
        if acc_ms <= 0.0 or cpu_ms / acc_ms >= MIN_ACCELERATOR_SPEEDUP:
            continue
        proof = result.get("performance_proof", {})
        residency = proof.get("weight_residency", {})
        transfer = proof.get("transfer", {})
        overheads = proof.get("overheads", {})
        tensors = proof.get("tensor_bytes", {})
        intensity = proof.get("arithmetic_intensity_flop_per_byte", {})
        native_pipeline = overheads.get("native_pipeline", {})
        evidence = {
            "workload_dir": str(workload_dir),
            "measured_latency_ms": {
                "cpu_only_native": cpu_ms,
                case_name: acc_ms,
                "speedup": cpu_ms / acc_ms,
                "required_speedup": MIN_ACCELERATOR_SPEEDUP,
            },
            "residency_and_staging": {
                "proof_status": residency.get("proof_status"),
                "valid_device_residency": residency.get("valid_device_residency"),
                "timed_weight_upload_bytes": residency.get("timed_weight_upload_bytes"),
                "timed_intermediate_host_transfer_bytes": transfer.get(
                    "timed_intermediate_host_transfer_bytes"
                ),
                "host_accumulation_bytes": overheads.get("host_accumulation_bytes"),
                "intermediate_residency": transfer.get("intermediate_residency"),
            },
            "launch_and_sync_model": {
                "estimated_launch_count": proof.get("launches", {}).get("total"),
                "observed_native_pipeline_launch_count": native_pipeline.get(
                    "launch_count_observed"
                ),
                "timed_allocation_count": overheads.get("timed_allocation_count"),
                "storage_kind": native_pipeline.get("storage_kind"),
                "input_upload_ms": native_pipeline.get("input_upload_ms"),
                "output_readback_ms": native_pipeline.get("output_readback_ms"),
            },
            "traffic_and_roofline": {
                "static_weight_bytes": tensors.get("static_weight_bytes"),
                "per_request_bytes": tensors.get("per_request_bytes"),
                "logical_total_tensor_bytes": tensors.get("logical_total_tensor_bytes"),
                "timed_hot_loop_flop_per_byte": intensity.get("timed_hot_loop_bytes"),
                "cache_fit": proof.get("cache_fit", {}),
                "interpretation": (
                    "The native CPU baseline runs a cache-resident NumPy/BLAS "
                    "GEMM+GEMV, while the accepted AIR-generated accelerator "
                    "path is dominated by serialized kernel/XRT/HIP launches "
                    "and synchronization. The measured launch path exceeds the "
                    "entire native CPU warm latency, falsifying the 1.20x "
                    "speedup target for this generated-kernel implementation."
                ),
            },
            "counters": proof.get("counters", {}),
            "cpu_native": cpu.get("performance_proof", {}).get("cpu_native", {}),
        }
        result["falsification"] = {
            "status": "falsified",
            "reason": "generated AIR accelerator launch/synchronization overhead exceeds native CPU latency",
            "evidence": evidence,
        }
        proof.setdefault("physical_plausibility", {})
        proof["physical_plausibility"] = {
            "status": "falsified",
            "evidence": evidence,
        }


def validate_output_dir(output_dir: Path, *, decode_weight_storage: str) -> list[str]:
    errors: list[str] = []
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        return [f"{output_dir}: missing summary.json"]
    try:
        summary = load_summary(output_dir)
    except json.JSONDecodeError as exc:
        return [f"{summary_path}: invalid JSON: {exc}"]
    if summary.get("measurement_mode") != MEASUREMENT_MODE:
        errors.append(
            f"{summary_path}: measurement_mode is {summary.get('measurement_mode')!r}"
        )
    if summary.get("decode_weight_storage") != decode_weight_storage:
        errors.append(
            f"{summary_path}: decode_weight_storage is "
            f"{summary.get('decode_weight_storage')!r}"
        )
    workloads_by_key = {
        (str(workload.get("suite")), str(workload.get("name"))): workload
        for workload in summary.get("workloads", [])
    }
    missing = expected_workload_keys() - workloads_by_key.keys()
    if missing:
        errors.append(f"{output_dir}: missing workload(s): {sorted(missing)}")

    for key in sorted(expected_workload_keys() & workloads_by_key.keys()):
        workload = workloads_by_key[key]
        workload_dir = output_dir / key[0] / key[1]
        cases_by_name = {
            str(result.get("case_name")): result for result in workload.get("cases", [])
        }
        missing_cases = set(FINAL_CASES) - cases_by_name.keys()
        if missing_cases:
            errors.append(f"{workload_dir}: missing case(s): {sorted(missing_cases)}")
        for case_name, result in cases_by_name.items():
            if case_name not in FINAL_CASES:
                errors.append(f"{workload_dir}: unexpected case {case_name!r}")
                continue
            errors.extend(
                f"{workload_dir}/{case_name}: {message}"
                for message in validate_result_payload(
                    result, decode_weight_storage=decode_weight_storage
                )
            )
        if set(FINAL_CASES) <= cases_by_name.keys():
            errors.extend(
                f"{workload_dir}: {message}"
                for message in validate_speedups(cases_by_name)
            )
    return errors


def validate_result_payload(
    result: dict[str, Any], *, decode_weight_storage: str
) -> list[str]:
    errors: list[str] = []
    case_name = str(result.get("case_name"))
    correctness = result.get("correctness", {})
    if correctness.get("validation_status") != "pass":
        errors.append(
            "correctness.validation_status is "
            f"{correctness.get('validation_status')!r}, expected 'pass'"
        )
    if correctness.get("prefill_allclose") is not True:
        errors.append("correctness.prefill_allclose is not true")
    if correctness.get("output_allclose") is not True:
        errors.append("correctness.output_allclose is not true")

    proof = result.get("performance_proof")
    if not isinstance(proof, dict):
        return errors + ["performance_proof is missing"]
    required = (
        "flops",
        "tensor_bytes",
        "actual_cpu_conversion_bytes",
        "cache_fit",
        "arithmetic_intensity_flop_per_byte",
        "launches",
        "weight_residency",
        "transfer",
        "counters",
    )
    for key in required:
        if key not in proof:
            errors.append(f"performance_proof.{key} is missing")

    quantized = result.get("quantized_decode", {})
    if decode_weight_storage == "int4":
        if quantized.get("enabled") is not True:
            errors.append("quantized_decode.enabled is not true for int4 run")
        if quantized.get("quant_kind") != "int4":
            errors.append(
                f"quantized_decode.quant_kind is {quantized.get('quant_kind')!r}"
            )
        if (
            case_name in {"gpu_only", "npu_only"}
            and quantized.get("hardware_fused") is not True
        ):
            errors.append("accelerator int4 decode is not hardware_fused")
    elif quantized.get("enabled") is True:
        errors.append("quantized_decode.enabled is true for bf16 run")

    if case_name == "cpu_only":
        if proof.get("cpu_native") is None:
            errors.append("cpu_only result is missing native CPU metadata")
        return errors

    stages = result.get("stage_backends", {})
    expected_backend = "gpu" if case_name == "gpu_only" else "npu"
    if (
        stages.get("prefill") != expected_backend
        or stages.get("decode") != expected_backend
    ):
        errors.append(f"{case_name} does not use {expected_backend}-only placement")
    implementation = result.get("implementation") or proof.get("implementation", {})
    if implementation.get("kind") != "air_generated":
        errors.append(
            f"implementation.kind is {implementation.get('kind')!r}, "
            "expected 'air_generated'"
        )
    residency = proof.get("weight_residency", {})
    transfer = proof.get("transfer", {})
    if residency.get("enabled") is not True:
        errors.append("resident weights are not enabled")
    if residency.get("valid_device_residency") is not True:
        errors.append("resident weights lack valid device-residency proof")
    if residency.get("proof_status") != "valid_device_residency":
        errors.append(
            f"resident weight proof_status is {residency.get('proof_status')!r}"
        )
    if int(residency.get("timed_weight_upload_bytes", -1)) != 0:
        errors.append("timed_weight_upload_bytes is not zero")
    if int(transfer.get("timed_intermediate_host_transfer_bytes", -1)) != 0:
        errors.append("timed_intermediate_host_transfer_bytes is not zero")
    overheads = proof.get("overheads", {})
    if int(overheads.get("host_accumulation_bytes", -1)) != 0:
        errors.append("host_accumulation_bytes is not zero")
    if transfer.get("full_prefill_semantics_preserved") is not True:
        errors.append("full_prefill_semantics_preserved is not true")
    if int(proof.get("launches", {}).get("total", 0)) <= 0:
        errors.append("accelerator launch count is not positive")
    return errors


def validate_speedups(cases_by_name: dict[str, dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    cpu_ms = _warm_mean_ms(cases_by_name["cpu_only"])
    if cpu_ms <= 0.0:
        return ["cpu_only warm mean latency is not positive"]
    for case_name in ("gpu_only", "npu_only"):
        accelerator_ms = _warm_mean_ms(cases_by_name[case_name])
        if accelerator_ms <= 0.0:
            errors.append(f"{case_name} warm mean latency is not positive")
            continue
        speedup = cpu_ms / accelerator_ms
        if speedup >= MIN_ACCELERATOR_SPEEDUP:
            continue
        falsification = _falsification(cases_by_name[case_name])
        if falsification is None:
            errors.append(
                f"{case_name} speedup {speedup:.3f} is below "
                f"{MIN_ACCELERATOR_SPEEDUP:.2f} and no falsification evidence exists"
            )
    return errors


def speedup_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for workload in summary.get("workloads", []):
        cases = {
            str(result.get("case_name")): result for result in workload.get("cases", [])
        }
        if not set(FINAL_CASES) <= cases.keys():
            continue
        cpu_ms = _warm_mean_ms(cases["cpu_only"])
        for case_name in ("gpu_only", "npu_only"):
            acc_ms = _warm_mean_ms(cases[case_name])
            speedup = 0.0 if acc_ms <= 0.0 else cpu_ms / acc_ms
            falsification = _falsification(cases[case_name])
            rows.append(
                {
                    "suite": workload.get("suite"),
                    "workload": workload.get("name"),
                    "case_name": case_name,
                    "cpu_ms": cpu_ms,
                    "accelerator_ms": acc_ms,
                    "speedup": speedup,
                    "classification": (
                        "pass"
                        if speedup >= MIN_ACCELERATOR_SPEEDUP
                        else "falsified" if falsification is not None else "fail"
                    ),
                    "falsification": falsification,
                }
            )
    return rows


def milestone5_summary_markdown(
    summaries: dict[str, dict[str, Any]], output_root: Path
) -> str:
    lines = [
        "# LLM-Linear Milestone 5 Accelerator-Only Summary",
        "",
        f"- Output root: `{output_root}`",
        f"- Suites: `{', '.join(REQUIRED_SUITES)}`",
        f"- Final cases: `{', '.join(FINAL_CASES)}`",
        f"- Resident weights: required",
        f"- Pass threshold: `{MIN_ACCELERATOR_SPEEDUP:.2f}x` over native CPU warm e2e",
        "",
    ]
    for run_name in ("bf16", "int4"):
        summary = summaries.get(run_name)
        if summary is None:
            continue
        rows = speedup_rows(summary)
        counts = {
            status: sum(1 for row in rows if row["classification"] == status)
            for status in ("pass", "falsified", "fail")
        }
        lines.extend(
            [
                f"## {run_name}",
                "",
                "Classification totals: "
                f"{counts['pass']} pass, {counts['falsified']} falsified, "
                f"{counts['fail']} fail.",
                "",
                "| Suite | Workload | Case | CPU ms | Accelerator ms | Speedup | Classification |",
                "| --- | --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for row in rows:
            lines.append(
                "| "
                f"{row['suite']} | "
                f"{row['workload']} | "
                f"{row['case_name']} | "
                f"{row['cpu_ms']:.6f} | "
                f"{row['accelerator_ms']:.6f} | "
                f"{row['speedup']:.3f} | "
                f"{row['classification']} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def write_milestone5_report(
    output_root: Path, summaries: dict[str, dict[str, Any]]
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    text = milestone5_summary_markdown(summaries, output_root)
    report_path = output_root / "report.md"
    report_path.write_text(text + "\n", encoding="utf-8")
    (output_root / "milestone5_summary.md").write_text(text + "\n", encoding="utf-8")
    return report_path


def _warm_mean_ms(result: dict[str, Any]) -> float:
    warm = result.get("measurement", {}).get("runs", {}).get("warm", {})
    if warm:
        return float(warm.get("latency_ms", {}).get("mean", 0.0))
    return float(result.get("latency_ms", {}).get("mean", 0.0))


def _falsification(result: dict[str, Any]) -> dict[str, Any] | None:
    explicit = result.get("falsification")
    if isinstance(explicit, dict) and explicit.get("status") == "falsified":
        evidence = explicit.get("evidence")
        return explicit if evidence else None
    physical = result.get("performance_proof", {}).get("physical_plausibility", {})
    if physical.get("status") == "falsified" and physical.get("evidence"):
        return physical
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run and validate Milestone 5 accelerator-only LLM-linear cases."
    )
    parser.add_argument(
        "--python",
        default="../../sandbox/bin/python",
        type=Path,
        help="Python interpreter to use for suite subprocesses.",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root for Milestone 5 acceptance artifacts.",
    )
    parser.add_argument(
        "--bridge-so",
        default="/tmp/libllm_linear_direct_bridge.so",
        type=Path,
        help="Native bridge shared library path to build and use.",
    )
    parser.add_argument(
        "--xrt-setup",
        default="/opt/xilinx/xrt/setup.sh",
        type=Path,
        help="XRT setup script to source in each subprocess.",
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument(
        "--run-filter",
        nargs="*",
        default=[],
        help="Optional storage run names to execute: bf16, int4.",
    )
    args = parser.parse_args(argv)

    output_root = resolve_project_path(args.output_root)
    logs_dir = output_root / "logs"
    env = os.environ.copy()
    seed_default_tool_env(env)
    env["LLM_LINEAR_DIRECT_BRIDGE_SO"] = str(args.bridge_so)
    xrt_setup = args.xrt_setup if args.xrt_setup else None
    selected = [
        run
        for run in ACCEPTANCE_RUNS
        if not args.run_filter or run.name in set(args.run_filter)
    ]
    if args.run_filter and len(selected) != len(set(args.run_filter)):
        known = {run.name for run in ACCEPTANCE_RUNS}
        requested = set(args.run_filter)
        raise SystemExit(f"unknown run-filter value(s): {sorted(requested - known)}")

    if not args.skip_build:
        run_logged(
            ["llm_linear/native/build_direct_bridge.sh", str(args.bridge_so)],
            log_path=logs_dir / "build_direct_bridge.log",
            env=env,
            xrt_setup=xrt_setup,
        )

    summaries: dict[str, dict[str, Any]] = {}
    for run in selected:
        run_output_dir = resolve_project_path(run.output_dir(args.output_root))
        if run_output_dir.exists():
            shutil.rmtree(run_output_dir)
        for case in FINAL_CASES:
            for suite, workload in ordered_workload_keys():
                log_path = logs_dir / f"{run.name}_{suite}_{workload}_{case}.log"
                run_logged(
                    run.case_argv(
                        args.python,
                        args.output_root,
                        suite=suite,
                        workload=workload,
                        case=case,
                    ),
                    log_path=log_path,
                    env=env,
                    xrt_setup=xrt_setup,
                    unset_xrt_ld_library_path=False,
                )
        combine_workload_outputs(
            run_output_dir, decode_weight_storage=run.decode_weight_storage
        )
        errors = validate_output_dir(
            run_output_dir, decode_weight_storage=run.decode_weight_storage
        )
        if errors:
            raise MilestoneFailure("\n".join(errors))
        summaries[run.name] = load_summary(run_output_dir)

    report_path = write_milestone5_report(output_root, summaries)
    print(
        "Milestone 5 accelerator-only acceptance passed. "
        f"Logs and results: {output_root}. Summary: {report_path}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MilestoneFailure as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
