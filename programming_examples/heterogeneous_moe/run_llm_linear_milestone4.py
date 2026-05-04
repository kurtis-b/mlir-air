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
    scan_log_for_blockers,
    seed_default_tool_env,
    validate_direct_result_payload,
)
from llm_linear.crossover import crossover_rows
from llm_linear.manifest import SCHEMA_VERSION
from llm_linear.reports import suite_summary_markdown
from llm_linear.results import CSV_FIELDNAMES, result_csv_row
from llm_linear.suites import SHAPE_LADDER

DEFAULT_OUTPUT_ROOT = "llm_linear/artifacts/benchmarks/milestone4_crossover"
REQUIRED_CASES = (
    "cpu_only",
    "gpu_only",
    "npu_only",
    "gpu_prefill_npu_decode_host",
    "npu_prefill_gpu_decode_host",
    "gpu_prefill_npu_decode_direct",
    "npu_prefill_gpu_decode_direct",
)
DIRECT_CASES = (
    "gpu_prefill_npu_decode_direct",
    "npu_prefill_gpu_decode_direct",
)
REQUIRED_SUITES = ("tiny_ci", "medium", "llm_like")
INT4_BLOCK_SIZE = 32
INT4_QUANT_AXIS = 0
MEASUREMENT_MODE = "both"
ITERATIONS = 5
WARMUP = 2


@dataclass(frozen=True)
class Milestone4Run:
    name: str
    decode_weight_storage: str
    quant_args: tuple[str, ...] = ()
    suites: tuple[str, ...] = REQUIRED_SUITES
    cases: tuple[str, ...] = REQUIRED_CASES

    def output_dir(self, output_root: str) -> str:
        return str(Path(output_root) / self.name)

    def argv(self, python: Path, output_root: str) -> list[str]:
        return [
            str(python),
            "run_llm_linear_suite.py",
            "--suite",
            *self.suites,
            "--case-filter",
            *self.cases,
            "--decode-weight-storage",
            self.decode_weight_storage,
            *self.quant_args,
            "--allow-npu",
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

    def workload_argv(
        self, python: Path, output_root: str, *, suite: str, workload: str
    ) -> list[str]:
        return [
            str(python),
            "run_llm_linear_suite.py",
            "--suite",
            suite,
            "--workload-filter",
            workload,
            "--case-filter",
            *self.cases,
            "--decode-weight-storage",
            self.decode_weight_storage,
            *self.quant_args,
            "--allow-npu",
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
    Milestone4Run(name="bf16", decode_weight_storage="bf16"),
    Milestone4Run(
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


def _workload_size_key(item: tuple[str, str]) -> tuple[int, str, str]:
    suite, workload = item
    shape = next(shape for shape in SHAPE_LADDER[suite] if shape["name"] == workload)
    tensor_work = int(shape["M"]) * int(shape["K"]) * int(shape["H"]) + int(
        shape["H"]
    ) * int(shape["N"])
    return (-tensor_work, suite, workload)


def ordered_workload_keys() -> list[tuple[str, str]]:
    return sorted(expected_workload_keys(), key=_workload_size_key)


def ordered_cases(cases: tuple[str, ...]) -> list[str]:
    direct = [case for case in cases if case in DIRECT_CASES]
    non_direct = [case for case in cases if case not in DIRECT_CASES]
    return non_direct + direct


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
        for case in REQUIRED_CASES:
            case_path = workload_dir / "cases" / f"{case}.json"
            cases.append(json.loads(case_path.read_text(encoding="utf-8")))
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
        for result in cases:
            csv_rows.append(result_csv_row(result))

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "measurement_mode": MEASUREMENT_MODE,
        "transfer_mode": "case",
        "decode_weight_storage": decode_weight_storage,
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
        suite_summary_markdown(summary, "LLM-Linear Heterogeneous Suite") + "\n",
        encoding="utf-8",
    )
    return summary


def validate_output_dir(
    output_dir: Path,
    *,
    expected_cases: tuple[str, ...] = REQUIRED_CASES,
    decode_weight_storage: str,
) -> list[str]:
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

    expected_workloads = expected_workload_keys()
    workloads_by_key = {
        (str(workload.get("suite")), str(workload.get("name"))): workload
        for workload in summary.get("workloads", [])
    }
    missing_workloads = expected_workloads - workloads_by_key.keys()
    if missing_workloads:
        errors.append(
            f"{output_dir}: missing expected workload(s): "
            f"{sorted(missing_workloads)}"
        )

    for key in sorted(expected_workloads & workloads_by_key.keys()):
        workload = workloads_by_key[key]
        workload_dir = output_dir / key[0] / key[1]
        seen: set[str] = set()
        skipped = workload.get("skipped", [])
        if skipped:
            skipped_names = [str(item.get("case_name")) for item in skipped]
            errors.append(f"{workload_dir}: skipped case(s): {skipped_names}")
        for payload in workload.get("cases", []):
            case_name = str(payload.get("case_name", ""))
            seen.add(case_name)
            if case_name not in expected_cases:
                errors.append(f"{workload_dir}: unexpected case_name {case_name!r}")
            case_path = workload_dir / "cases" / f"{case_name}.json"
            if not case_path.exists():
                errors.append(f"{case_path}: missing per-case result JSON")
            require_direct = case_name.endswith("_direct")
            errors.extend(
                f"{workload_dir}/{case_name}: {message}"
                for message in validate_result_payload(
                    payload,
                    decode_weight_storage=decode_weight_storage,
                    require_direct=require_direct,
                )
            )
        missing_cases = set(expected_cases) - seen
        if missing_cases:
            errors.append(
                f"{workload_dir}: missing expected case(s): " f"{sorted(missing_cases)}"
            )

    errors.extend(validate_required_crossover_rows(summary, expected_workloads))
    return errors


def validate_result_payload(
    result: dict[str, Any], *, decode_weight_storage: str, require_direct: bool
) -> list[str]:
    errors = validate_direct_result_payload(result, require_direct=require_direct)
    if decode_weight_storage == "int4":
        errors.extend(validate_int4_payload(result))
    elif result.get("quantized_decode", {}).get("enabled") is True:
        errors.append("quantized_decode.enabled is true for bf16 run")
    return errors


def validate_int4_payload(result: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    quantized = result.get("quantized_decode", {})
    if quantized.get("enabled") is not True:
        errors.append("quantized_decode.enabled is not true")
    if quantized.get("quant_kind") != "int4":
        errors.append(f"quantized_decode.quant_kind is {quantized.get('quant_kind')!r}")
    if int(quantized.get("packed_bytes", 0)) <= 0:
        errors.append("quantized_decode.packed_bytes is not positive")
    if int(quantized.get("scale_bytes", 0)) <= 0:
        errors.append("quantized_decode.scale_bytes is not positive")
    metadata = quantized.get("metadata") or {}
    if metadata.get("quant_kind") != "int4":
        errors.append("quantized_decode.metadata.quant_kind is not int4")
    if int(metadata.get("block_size", 0)) != INT4_BLOCK_SIZE:
        errors.append(f"quantized_decode.metadata.block_size is not {INT4_BLOCK_SIZE}")
    if int(metadata.get("quant_axis", -1)) != INT4_QUANT_AXIS:
        errors.append(f"quantized_decode.metadata.quant_axis is not {INT4_QUANT_AXIS}")

    decode_backend = str(
        result.get("stage_backends", {}).get(
            "decode", result.get("placement", {}).get("decode", "")
        )
    )
    if decode_backend in {"gpu", "npu"} and quantized.get("hardware_fused") is not True:
        errors.append(
            "quantized_decode.hardware_fused is not true for accelerator decode"
        )
    return errors


def validate_required_crossover_rows(
    summary: dict[str, Any], expected_workloads: set[tuple[str, str]]
) -> list[str]:
    errors: list[str] = []
    rows = crossover_rows(summary)
    row_keys = {
        (
            str(row.get("suite")),
            str(row.get("workload")),
            str(row.get("mixed_case")),
            str(row.get("baseline")),
        )
        for row in rows
    }
    for suite, workload in sorted(expected_workloads):
        for mixed in DIRECT_CASES:
            baselines = (
                "cpu_only",
                "gpu_only",
                "npu_only",
                mixed.replace("_direct", "_host"),
            )
            for baseline in baselines:
                key = (suite, workload, mixed, baseline)
                if key not in row_keys:
                    errors.append(
                        "missing crossover row for "
                        f"{suite}/{workload}: {mixed} vs {baseline}"
                    )
    return errors


def milestone4_summary_markdown(
    summaries: dict[str, dict[str, Any]], output_root: Path
) -> str:
    lines = [
        "# LLM-Linear Milestone 4 Crossover Summary",
        "",
        f"- Output root: `{output_root}`",
        f"- Suites: `{', '.join(REQUIRED_SUITES)}`",
        f"- Cases: `{', '.join(REQUIRED_CASES)}`",
        f"- Measurement: `{MEASUREMENT_MODE}`, iterations={ITERATIONS}, warmup={WARMUP}",
        "",
    ]
    for run_name in ("bf16", "int4"):
        summary = summaries.get(run_name)
        if summary is None:
            continue
        rows = crossover_rows(summary)
        lines.extend([f"## {run_name}", ""])
        if not rows:
            lines.extend(
                [
                    "No audited direct mixed crossover rows were present.",
                    f"Bottleneck: {_dominant_direct_bottleneck(summary)}.",
                    "",
                ]
            )
            continue
        counts = {
            classification: sum(
                1 for row in rows if row["classification"] == classification
            )
            for classification in ("wins", "loses", "inconclusive")
        }
        lines.append(
            "Classification totals: "
            f"{counts['wins']} wins, {counts['loses']} loses, "
            f"{counts['inconclusive']} inconclusive."
        )
        if counts["wins"] == 0:
            lines.append(
                "No direct mixed crossover win was observed for this storage mode."
            )
            lines.append(f"Bottleneck: {_dominant_direct_bottleneck(summary)}.")
        lines.extend(
            [
                "",
                "| Suite | Workload | Direct case | Baseline | Speedup | Classification |",
                "| --- | --- | --- | --- | ---: | --- |",
            ]
        )
        for row in rows:
            lines.append(
                "| "
                f"{row['suite']} | "
                f"{row['workload']} | "
                f"{row['mixed_case']} | "
                f"{row['baseline']} | "
                f"{row['speedup']:.3f} | "
                f"{row['classification']} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def _dominant_direct_bottleneck(summary: dict[str, Any]) -> str:
    counts: dict[str, int] = {"prefill stage": 0, "decode stage": 0, "handoff/sync": 0}
    total = 0
    for workload in summary.get("workloads", []):
        for result in workload.get("cases", []):
            if result.get("case_name") not in DIRECT_CASES:
                continue
            components = {
                "prefill stage": _stage_mean_ms(result, "prefill"),
                "decode stage": _stage_mean_ms(result, "decode"),
                "handoff/sync": _transfer_elapsed_ms(result),
            }
            dominant = max(components, key=components.get)
            counts[dominant] += 1
            total += 1
    if total == 0:
        return "no direct mixed case results were available"
    dominant = max(counts, key=counts.get)
    return f"{dominant} dominated {counts[dominant]} of {total} direct mixed cases"


def _stage_mean_ms(result: dict[str, Any], stage: str) -> float:
    return float(result.get("stage_latency_ms", {}).get(stage, {}).get("mean", 0.0))


def _transfer_elapsed_ms(result: dict[str, Any]) -> float:
    transfer = result.get("transfer_summary", {})
    if "total_elapsed_us" in transfer:
        return float(transfer.get("total_elapsed_us", 0.0)) / 1000.0
    return (
        sum(
            float(event.get("elapsed_us", 0.0))
            for event in result.get("transfer_events", [])
        )
        / 1000.0
    )


def write_milestone4_report(
    output_root: Path, summaries: dict[str, dict[str, Any]]
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "report.md"
    text = milestone4_summary_markdown(summaries, output_root)
    report_path.write_text(text + "\n", encoding="utf-8")
    (output_root / "milestone4_summary.md").write_text(text + "\n", encoding="utf-8")
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run and validate the Milestone 4 LLM-linear crossover study."
    )
    parser.add_argument(
        "--python",
        default="../../sandbox/bin/python",
        type=Path,
        help="Python interpreter to use for suite subprocesses.",
    )
    parser.add_argument(
        "--bridge-so",
        default="/tmp/libllm_linear_direct_bridge.so",
        type=Path,
        help="Native bridge shared library path to build and use.",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root for Milestone 4 acceptance artifacts.",
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
        help="Optional run names to execute: bf16, int4.",
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
        for case in ordered_cases(run.cases):
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
            run_output_dir,
            expected_cases=run.cases,
            decode_weight_storage=run.decode_weight_storage,
        )
        if errors:
            raise MilestoneFailure("\n".join(errors))
        summaries[run.name] = load_summary(run_output_dir)

    report_path = write_milestone4_report(output_root, summaries)
    print(
        "Milestone 4 crossover acceptance passed. "
        f"Logs and results: {output_root}. Summary: {report_path}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MilestoneFailure as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
