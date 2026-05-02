# SPDX-License-Identifier: MIT

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from manifest import EDGE_STUDY_SCHEMA_VERSION, update_manifest_backends
from metadata import collect_run_metadata
from orchestrator import MoERuntime
from results import benchmark_runtime, build_case_result, generate_inputs, validate_runtime
from trace import TraceRecorder


@dataclass(frozen=True)
class RunCaseOptions:
    manifest_path: Path
    case_name: str = "adhoc"
    iterations: int | None = None
    warmup: int | None = None
    measurement_mode: str = "warm"
    command_line: list[str] | None = None


def case_stage_backends(case: dict[str, Any]) -> dict[str, str]:
    if "stage_backends" in case:
        return dict(case["stage_backends"])
    return {
        "router": case["router_backend"],
        "expert0": case["expert0_backend"],
        "expert1": case["expert1_backend"],
        "aggregation": case["aggregation_backend"],
    }


def contains_npu(stage_backends: dict[str, str]) -> bool:
    return any(backend == "npu" for backend in stage_backends.values())


def apply_case_to_manifest(manifest: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    stages = case_stage_backends(case)
    return update_manifest_backends(
        manifest,
        router_backend=stages.get("router"),
        expert0_backend=stages.get("expert0"),
        expert1_backend=stages.get("expert1"),
        aggregation_backend=stages.get("aggregation"),
        transfer_mode=case.get("transfer_mode"),
        router_mode=case.get("router_mode"),
    )


def run_case_with_trace(
    manifest: dict[str, Any],
    case: dict[str, Any],
    options: RunCaseOptions,
) -> tuple[dict[str, Any], TraceRecorder | None]:
    manifest = apply_case_to_manifest(copy.deepcopy(manifest), case)
    local_iterations = options.iterations or case.get("iterations") or manifest["benchmark"]["iterations"]
    local_warmup = options.warmup if options.warmup is not None else case.get("warmup", manifest["benchmark"]["warmup"])

    setup_start = time.perf_counter_ns()
    runtime = MoERuntime(manifest)
    runtime.prepare()
    setup_ms = (time.perf_counter_ns() - setup_start) / 1_000_000.0

    inputs, input_phase_timings, _ = generate_inputs(runtime, manifest)
    timing = benchmark_runtime(
        runtime,
        inputs,
        router_mode=manifest["runtime"]["router_mode"],
        iterations=local_iterations,
        warmup=local_warmup,
        measurement_mode=options.measurement_mode,
    )
    last_run, validation_ms = validate_runtime(runtime, inputs, router_mode=manifest["runtime"]["router_mode"])
    phase_timings_ms = {
        **input_phase_timings,
        **timing["phase_timings_ms"],
        "compile_load_setup_ms": setup_ms,
        "validation_ms": validation_ms,
    }
    result = build_case_result(
        schema_version=EDGE_STUDY_SCHEMA_VERSION,
        metadata=collect_run_metadata(options.manifest_path, manifest, command_line=options.command_line),
        case_name=options.case_name or case.get("name", "adhoc"),
        manifest=manifest,
        iterations=local_iterations,
        requested_warmup=local_warmup,
        measurement_mode=options.measurement_mode,
        timing=timing,
        last_run=last_run,
        validation_ms=validation_ms,
        phase_timings_ms=phase_timings_ms,
    )
    return result, last_run.get("trace")


def run_case(
    manifest: dict[str, Any],
    case: dict[str, Any],
    options: RunCaseOptions,
) -> dict[str, Any]:
    result, _ = run_case_with_trace(manifest, case, options)
    return result
