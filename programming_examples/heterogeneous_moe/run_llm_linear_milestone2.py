#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BLOCKER_MARKERS = (
    "Reverting to host copy of buffers",
    "exec_buf: Operation not supported",
)
DIRECT_MECHANISM = "hip_vmem_export_xrt_bo_import_fd"
DIRECT_CONTRACT = "no_host_copies"
DIRECT_CLASS_DEVICE_RESIDENT = "device_resident_zero_host_copy"
DEFAULT_OUTPUT_ROOT = "llm_linear/artifacts/benchmarks/milestone2_e2e"


class MilestoneFailure(RuntimeError):
    pass


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


def project_dir() -> Path:
    return Path(__file__).resolve().parent


def repo_dir() -> Path:
    return project_dir().parents[1]


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return project_dir() / candidate


def seed_default_tool_env(env: dict[str, str]) -> None:
    repo = repo_dir()
    sandbox_bin = repo / "sandbox" / "bin"
    if sandbox_bin.exists():
        env["PATH"] = str(sandbox_bin) + os.pathsep + env.get("PATH", "")
    defaults: dict[str, tuple[Path, ...]] = {
        "AIRCC_PATH": (
            repo / "install-xrt" / "bin" / "aircc",
            repo / "build-xrt" / "bin" / "aircc",
            repo / "install" / "bin" / "aircc",
            repo / "build" / "bin" / "aircc",
        ),
        "AIR_OPT_PATH": (
            repo / "build-gpu-lit" / "bin" / "air-opt",
            repo / "install-xrt" / "bin" / "air-opt",
            repo / "build-xrt" / "bin" / "air-opt",
            repo / "install" / "bin" / "air-opt",
            repo / "build" / "bin" / "air-opt",
        ),
        "LLVM_INSTALL_DIR": (repo / "llvm" / "install-amdgpu",),
    }
    for name, candidates in defaults.items():
        if env.get(name):
            continue
        for path in candidates:
            if path.exists():
                env[name] = str(path)
                break
    if not env.get("ROCM_PATH"):
        rocm = Path("/opt/rocm")
        if rocm.exists():
            env["ROCM_PATH"] = str(rocm)
    python_paths = [
        path
        for path in (
            repo / "build" / "python",
            repo / "install-xrt" / "python",
            repo / "build-xrt" / "python",
            repo / "python",
        )
        if path.exists()
    ]
    if python_paths:
        existing = env.get("PYTHONPATH")
        entries = [str(path) for path in python_paths]
        if existing:
            entries.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(entries)


def shell_command(
    argv: list[str], *, xrt_setup: Path | None, unset_xrt_ld_library_path: bool
) -> list[str]:
    argv_command = " ".join(shlex.quote(str(item)) for item in argv)
    if xrt_setup is not None:
        command = f"source {shlex.quote(str(xrt_setup))} && "
        if unset_xrt_ld_library_path:
            command += "unset LD_LIBRARY_PATH && "
        command += f"exec {argv_command}"
    else:
        command = argv_command
    return ["/bin/bash", "-lc", command]


def run_logged(
    argv: list[str],
    *,
    log_path: Path,
    env: dict[str, str],
    xrt_setup: Path | None,
    unset_xrt_ld_library_path: bool = False,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = shell_command(
        argv,
        xrt_setup=xrt_setup,
        unset_xrt_ld_library_path=unset_xrt_ld_library_path,
    )
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(shlex.quote(part) for part in command) + "\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=project_dir(),
            env=env,
            check=False,
            text=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    blockers = scan_log_for_blockers(log_path)
    if completed.returncode != 0:
        raise MilestoneFailure(
            f"{log_path.name} failed with exit code {completed.returncode}"
        )
    if blockers:
        raise MilestoneFailure(
            f"{log_path.name} contains blocked XRT fallback marker(s): "
            + ", ".join(blockers)
        )


def scan_log_for_blockers(log_path: Path) -> list[str]:
    if not log_path.exists():
        return []
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return [marker for marker in BLOCKER_MARKERS if marker in text]


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


def validate_result_payload(
    result: dict[str, Any], *, require_direct: bool
) -> list[str]:
    errors: list[str] = []
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
    if not require_direct:
        return errors

    shape = result.get("shape", {})
    try:
        m = int(shape["M"])
        h = int(shape["H"])
    except (KeyError, TypeError, ValueError):
        errors.append("shape.M/H missing or invalid")
        return errors
    item_bytes = dtype_item_bytes(str(result.get("dtype", "")))
    if item_bytes is None:
        errors.append(f"unsupported dtype for direct audit: {result.get('dtype')!r}")
        return errors
    expected_bytes = h * item_bytes
    expected_offset = (m - 1) * h * item_bytes

    transfer_summary = result.get("transfer_summary", {})
    direct_summary = transfer_summary.get("direct_handoff", {})
    if direct_summary.get("supported") is not True:
        errors.append("transfer_summary.direct_handoff.supported is not true")
    if direct_summary.get("contract") != DIRECT_CONTRACT:
        errors.append(
            "transfer_summary.direct_handoff.contract is "
            f"{direct_summary.get('contract')!r}"
        )
    if direct_summary.get("mechanism") != DIRECT_MECHANISM:
        errors.append(
            "transfer_summary.direct_handoff.mechanism is "
            f"{direct_summary.get('mechanism')!r}"
        )
    if direct_summary.get("direct_class") != DIRECT_CLASS_DEVICE_RESIDENT:
        errors.append(
            "transfer_summary.direct_handoff.direct_class is "
            f"{direct_summary.get('direct_class')!r}"
        )
    if direct_summary.get("zero_host_copy") is not True:
        errors.append("transfer_summary.direct_handoff.zero_host_copy is not true")
    if direct_summary.get("device_resident_buffers") is not True:
        errors.append(
            "transfer_summary.direct_handoff.device_resident_buffers is not true"
        )
    for field in (
        "numpy_host_materializations",
        "direct_handoff_numpy_host_materializations",
    ):
        if int(transfer_summary.get(field, 0)) != 0:
            errors.append(f"transfer_summary.{field} is not zero")

    execution_truth = result.get("execution_truth", {})
    if int(execution_truth.get("numpy_host_materializations", 0)) != 0:
        errors.append("execution_truth.numpy_host_materializations is not zero")

    direct_bridge = result.get("direct_bridge") or {}
    if direct_bridge.get("mechanism") != DIRECT_MECHANISM:
        errors.append(f"direct_bridge.mechanism is {direct_bridge.get('mechanism')!r}")
    if direct_bridge.get("direct_class") != DIRECT_CLASS_DEVICE_RESIDENT:
        errors.append(
            f"direct_bridge.direct_class is {direct_bridge.get('direct_class')!r}"
        )
    if direct_bridge.get("zero_host_copy") is not True:
        errors.append("direct_bridge.zero_host_copy is not true")
    if int(direct_bridge.get("import_method", -1)) != 3:
        errors.append(
            f"direct_bridge.import_method is {direct_bridge.get('import_method')!r}"
        )
    if int(direct_bridge.get("subview_offset_bytes", -1)) != expected_offset:
        errors.append(
            "direct_bridge.subview_offset_bytes is "
            f"{direct_bridge.get('subview_offset_bytes')!r}, expected {expected_offset}"
        )
    probe_report = direct_summary.get("probe_report") or direct_bridge.get(
        "probe_report"
    )
    errors.extend(validate_direct_probe_report(probe_report, DIRECT_MECHANISM))

    direct_events = [
        event
        for event in result.get("transfer_events", [])
        if event.get("actual_mode") == "device_resident_direct_handoff"
    ]
    if not direct_events:
        errors.append("no device_resident_direct_handoff transfer event found")
    for event in direct_events:
        errors.extend(
            validate_direct_event(event, result, expected_bytes, expected_offset)
        )
    return errors


def validate_direct_probe_report(probe_report: Any, mechanism: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(probe_report, dict):
        return ["direct handoff probe_report is missing"]
    if probe_report.get("contract") != DIRECT_CONTRACT:
        errors.append(f"probe_report.contract is {probe_report.get('contract')!r}")
    if probe_report.get("direct_supported") is not True:
        errors.append("probe_report.direct_supported is not true")
    if probe_report.get("selected_mechanism") != mechanism:
        errors.append(
            "probe_report.selected_mechanism is "
            f"{probe_report.get('selected_mechanism')!r}"
        )
    mechanisms = probe_report.get("mechanisms", [])
    selected = None
    if isinstance(mechanisms, list):
        selected = next(
            (
                item
                for item in mechanisms
                if isinstance(item, dict) and item.get("mechanism") == mechanism
            ),
            None,
        )
    if selected is None:
        return errors + [f"probe_report missing mechanism {mechanism!r}"]
    for field in ("supported", "direct_eligible", "zero_host_copy"):
        if selected.get(field) is not True:
            errors.append(f"probe_report.{mechanism}.{field} is not true")
    if selected.get("direct_class") != DIRECT_CLASS_DEVICE_RESIDENT:
        errors.append(
            f"probe_report.{mechanism}.direct_class is "
            f"{selected.get('direct_class')!r}"
        )
    if selected.get("bidirectional_visibility") is not True:
        errors.append(f"probe_report.{mechanism}.bidirectional_visibility is not true")
    if selected.get("npu_kernel_verification") is not True:
        errors.append(f"probe_report.{mechanism}.npu_kernel_verification is not true")
    if int(selected.get("host_materialization_count", -1)) != 0:
        errors.append(
            f"probe_report.{mechanism}.host_materialization_count is not zero"
        )
    return errors


def dtype_item_bytes(dtype: str) -> int | None:
    if dtype in {"bf16", "f16"}:
        return 2
    return None


def validate_direct_event(
    event: dict[str, Any],
    result: dict[str, Any],
    expected_bytes: int,
    expected_offset: int,
) -> list[str]:
    errors: list[str] = []
    if int(event.get("bytes", -1)) != expected_bytes:
        errors.append(
            f"direct event bytes is {event.get('bytes')!r}, expected {expected_bytes}"
        )
    if int(event.get("numpy_host_materializations", -1)) != 0:
        errors.append("direct event numpy_host_materializations is not zero")
    if event.get("mechanism") != DIRECT_MECHANISM:
        errors.append(f"direct event mechanism is {event.get('mechanism')!r}")
    if event.get("direct_class") != DIRECT_CLASS_DEVICE_RESIDENT:
        errors.append(f"direct event direct_class is {event.get('direct_class')!r}")
    if event.get("zero_host_copy") is not True:
        errors.append("direct event zero_host_copy is not true")
    if event.get("device_resident_buffers") is not True:
        errors.append("direct event device_resident_buffers is not true")

    tensor = event.get("tensor", {})
    if tensor.get("owner") != "hip_vmem":
        errors.append(f"direct tensor owner is {tensor.get('owner')!r}")
    if tensor.get("imported_view") != "xrt_bo":
        errors.append(f"direct tensor imported_view is {tensor.get('imported_view')!r}")
    if tensor.get("mechanism") != DIRECT_MECHANISM:
        errors.append(f"direct tensor mechanism is {tensor.get('mechanism')!r}")
    if tensor.get("direct_class") != DIRECT_CLASS_DEVICE_RESIDENT:
        errors.append(f"direct tensor direct_class is {tensor.get('direct_class')!r}")
    if tensor.get("zero_host_copy") is not True:
        errors.append("direct tensor zero_host_copy is not true")
    if int(tensor.get("byte_size", -1)) != expected_bytes:
        errors.append(
            "direct tensor byte_size is "
            f"{tensor.get('byte_size')!r}, expected {expected_bytes}"
        )
    if int(tensor.get("offset", -1)) != expected_offset:
        errors.append(
            f"direct tensor offset is {tensor.get('offset')!r}, expected {expected_offset}"
        )

    sync_names = [str(item.get("event", "")) for item in event.get("sync_events", [])]
    if "xrtBoCopy" in sync_names:
        errors.append("direct sync_events still include xrtBoCopy")
    placement = result.get("stage_backends", {})
    if placement.get("prefill") == "gpu" and placement.get("decode") == "npu":
        if not {"xrtBoSubview", "hipKernel", "hipMemcpyDtoD"} & set(sync_names):
            errors.append(
                "GPU->NPU direct sync_events do not include a no-XRT-copy row event"
            )
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
