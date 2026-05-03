#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import run_llm_linear_milestone2 as milestone2

DEFAULT_OUTPUT_ROOT = "llm_linear/artifacts/benchmarks/milestone3_int4_hw"
REQUIRED_CASES = (
    "gpu_only",
    "npu_only",
    "gpu_prefill_npu_decode_host",
    "npu_prefill_gpu_decode_host",
    "gpu_prefill_npu_decode_direct",
    "npu_prefill_gpu_decode_direct",
)
REQUIRED_SUITES = ("tiny_ci", "medium", "llm_like")


class MilestoneFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class AcceptanceRun:
    name: str
    suite: str
    cases: tuple[str, ...] = REQUIRED_CASES

    def output_dir(self, output_root: str) -> str:
        return str(Path(output_root) / self.suite)

    def argv(self, python: Path, output_root: str) -> list[str]:
        return [
            str(python),
            "run_llm_linear_suite.py",
            "--suite",
            self.suite,
            "--case-filter",
            *self.cases,
            "--decode-weight-storage",
            "int4",
            "--decode-quant-block-size",
            "32",
            "--decode-quant-axis",
            "0",
            "--allow-npu",
            "--iterations",
            "1",
            "--warmup",
            "0",
            "--require-correctness",
            "--output-dir",
            self.output_dir(output_root),
        ]


ACCEPTANCE_RUNS = tuple(
    AcceptanceRun(name=f"{suite}_int4_hw", suite=suite) for suite in REQUIRED_SUITES
)


def iter_workload_case_results(output_dir: Path) -> dict[Path, list[Path]]:
    grouped: dict[Path, list[Path]] = {}
    for path in sorted(output_dir.glob("*/*/cases/*.json")):
        grouped.setdefault(path.parent.parent, []).append(path)
    return grouped


def validate_output_dir(output_dir: Path, expected_cases: tuple[str, ...]) -> list[str]:
    errors: list[str] = []
    workloads = iter_workload_case_results(output_dir)
    if not workloads:
        return [f"{output_dir}: no case result JSON files found"]
    expected = set(expected_cases)
    for workload_dir, paths in workloads.items():
        seen: set[str] = set()
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            case_name = str(payload.get("case_name", ""))
            seen.add(case_name)
            if case_name not in expected:
                errors.append(f"{path}: unexpected case_name {case_name!r}")
            require_direct = case_name.endswith("_direct")
            errors.extend(
                f"{path}: {message}"
                for message in validate_result_payload(
                    payload, require_direct=require_direct
                )
            )
        missing = expected - seen
        if missing:
            errors.append(
                f"{workload_dir}: missing expected case(s): {sorted(missing)}"
            )
    return errors


def validate_result_payload(
    result: dict[str, Any], *, require_direct: bool
) -> list[str]:
    errors = milestone2.validate_result_payload(result, require_direct=require_direct)
    quantized = result.get("quantized_decode", {})
    if quantized.get("enabled") is not True:
        errors.append("quantized_decode.enabled is not true")
    if quantized.get("quant_kind") != "int4":
        errors.append(f"quantized_decode.quant_kind is {quantized.get('quant_kind')!r}")
    if quantized.get("hardware_fused") is not True:
        errors.append("quantized_decode.hardware_fused is not true")
    if int(quantized.get("packed_bytes", 0)) <= 0:
        errors.append("quantized_decode.packed_bytes is not positive")
    if int(quantized.get("scale_bytes", 0)) <= 0:
        errors.append("quantized_decode.scale_bytes is not positive")
    metadata = quantized.get("metadata") or {}
    if metadata.get("quant_kind") != "int4":
        errors.append("quantized_decode.metadata.quant_kind is not int4")
    if int(metadata.get("block_size", 0)) != 32:
        errors.append("quantized_decode.metadata.block_size is not 32")
    if int(metadata.get("quant_axis", -1)) != 0:
        errors.append("quantized_decode.metadata.quant_axis is not 0")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run and validate Milestone 3 int4 hardware decode acceptance."
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
        help="Output root for Milestone 3 acceptance artifacts.",
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
        help="Optional acceptance run names to execute.",
    )
    args = parser.parse_args(argv)

    output_root = milestone2.resolve_project_path(args.output_root)
    logs_dir = output_root / "logs"
    env = os.environ.copy()
    milestone2.seed_default_tool_env(env)
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
        milestone2.run_logged(
            ["llm_linear/native/build_direct_bridge.sh", str(args.bridge_so)],
            log_path=logs_dir / "build_direct_bridge.log",
            env=env,
            xrt_setup=xrt_setup,
        )

    for run in selected:
        log_path = logs_dir / f"{run.name}.log"
        milestone2.run_logged(
            run.argv(args.python, args.output_root),
            log_path=log_path,
            env=env,
            xrt_setup=xrt_setup,
            unset_xrt_ld_library_path=False,
        )
        errors = validate_output_dir(
            milestone2.resolve_project_path(run.output_dir(args.output_root)),
            expected_cases=run.cases,
        )
        if errors:
            raise MilestoneFailure("\n".join(errors))

    print(
        f"Milestone 3 int4 hardware acceptance passed. Logs and results: {output_root}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (MilestoneFailure, milestone2.MilestoneFailure) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
