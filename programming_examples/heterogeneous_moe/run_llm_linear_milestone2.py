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

from llm_linear.acceptance import (
    DIRECT_CLASS_DEVICE_RESIDENT,
    DIRECT_CONTRACT,
    DIRECT_MECHANISM,
    MilestoneFailure,
    resolve_project_path,
    run_logged,
    scan_log_for_blockers,
    seed_default_tool_env,
    validate_direct_result_payload as validate_result_payload,
)

DEFAULT_OUTPUT_ROOT = "llm_linear/artifacts/benchmarks/milestone2_e2e"


@dataclass(frozen=True)
class AcceptanceRun:
    name: str
    suite: str
    cases: tuple[str, ...]
    output_leaf: str
    require_direct: bool
    unset_xrt_ld_library_path: bool
    workload_filters: tuple[str, ...] = ()
    isolate_cases: bool = False

    def output_dir(self, output_root: str) -> str:
        return str(Path(output_root) / self.output_leaf)

    def argv(
        self, python: Path, output_root: str, cases: tuple[str, ...] | None = None
    ) -> list[str]:
        argv = [
            str(python),
            "run_llm_linear_suite.py",
            "--suite",
            self.suite,
        ]
        if self.workload_filters:
            argv.extend(["--workload-filter", *self.workload_filters])
        argv.extend(
            [
                "--case-filter",
                *(cases or self.cases),
                "--allow-npu",
                "--iterations",
                "1",
                "--warmup",
                "0",
                "--require-correctness",
                "--output-dir",
                self.output_dir(output_root),
            ]
        )
        return argv


ACCEPTANCE_RUNS = (
    AcceptanceRun(
        name="tiny_g2n_direct",
        suite="tiny_ci",
        cases=("gpu_prefill_npu_decode_direct",),
        output_leaf="tiny_g2n_direct",
        require_direct=True,
        unset_xrt_ld_library_path=True,
    ),
    AcceptanceRun(
        name="tiny_n2g_direct",
        suite="tiny_ci",
        cases=("npu_prefill_gpu_decode_direct",),
        output_leaf="tiny_n2g_direct",
        require_direct=True,
        unset_xrt_ld_library_path=True,
    ),
    AcceptanceRun(
        name="medium_g2n_direct",
        suite="medium",
        cases=("gpu_prefill_npu_decode_direct",),
        output_leaf="medium_g2n_direct",
        require_direct=True,
        unset_xrt_ld_library_path=True,
        workload_filters=("medium_m8",),
    ),
    AcceptanceRun(
        name="medium_n2g_direct",
        suite="medium",
        cases=("npu_prefill_gpu_decode_direct",),
        output_leaf="medium_n2g_direct",
        require_direct=True,
        unset_xrt_ld_library_path=True,
        workload_filters=("medium_m8",),
    ),
    AcceptanceRun(
        name="medium_host_mixed",
        suite="medium",
        cases=(
            "gpu_prefill_npu_decode_host",
            "npu_prefill_gpu_decode_host",
        ),
        output_leaf="medium_host_mixed",
        require_direct=False,
        unset_xrt_ld_library_path=False,
        workload_filters=("medium_m8",),
        isolate_cases=True,
    ),
)


def iter_case_results(output_dir: Path) -> list[Path]:
    return sorted(output_dir.glob("*/*/cases/*.json"))


def validate_output_dir(
    output_dir: Path,
    *,
    expected_cases: tuple[str, ...],
    require_direct: bool,
) -> list[str]:
    errors: list[str] = []
    result_paths = iter_case_results(output_dir)
    if not result_paths:
        return [f"{output_dir}: no case result JSON files found"]
    allowed = set(expected_cases)
    seen: set[str] = set()
    for path in result_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        case_name = str(payload.get("case_name", ""))
        seen.add(case_name)
        if case_name not in allowed:
            errors.append(f"{path}: unexpected case_name {case_name!r}")
        errors.extend(
            f"{path}: {message}"
            for message in validate_result_payload(
                payload, require_direct=require_direct
            )
        )
    missing = allowed - seen
    if missing:
        errors.append(f"{output_dir}: missing expected case(s): {sorted(missing)}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run and validate the Milestone 2 LLM-linear direct acceptance suite."
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
        help="Output root for Milestone 2 acceptance artifacts.",
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
        build_log = logs_dir / "build_direct_bridge.log"
        run_logged(
            ["llm_linear/native/build_direct_bridge.sh", str(args.bridge_so)],
            log_path=build_log,
            env=env,
            xrt_setup=xrt_setup,
        )

    for run in selected:
        case_batches = (
            tuple((case,) for case in run.cases) if run.isolate_cases else (run.cases,)
        )
        for case_batch in case_batches:
            log_suffix = "_" + "_".join(case_batch) if run.isolate_cases else ""
            log_path = logs_dir / f"{run.name}{log_suffix}.log"
            run_logged(
                run.argv(args.python, args.output_root, cases=case_batch),
                log_path=log_path,
                env=env,
                xrt_setup=xrt_setup,
                unset_xrt_ld_library_path=run.unset_xrt_ld_library_path,
            )
        errors = validate_output_dir(
            resolve_project_path(run.output_dir(args.output_root)),
            expected_cases=run.cases,
            require_direct=run.require_direct,
        )
        if errors:
            raise MilestoneFailure("\n".join(errors))

    print(f"Milestone 2 acceptance passed. Logs and results: {output_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MilestoneFailure as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
