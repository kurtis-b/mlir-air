#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gemma3 paper-benchmark result JSON helpers."""

from __future__ import annotations

import argparse
import json
import shlex
from types import SimpleNamespace
import sys
from pathlib import Path
from typing import Any

from gemma3_artifacts import MODEL_SPECS, discover_model_artifacts, model_spec
from gemma3_environment import capture_environment
from gemma3_nonlinears import measure_cpu_contracts, paper_match_blockers
from gemma3_paper_compare import load_targets, target_by_id
from gemma3_power import capture_power_snapshot
from gemma3_real_execution import Gemma3ExecutionError, run_torch_benchmark

DEFAULT_POWER_WATTS = {"cpu": None, "gpu": None, "npu": None, "total": None}
TABLE_BY_METRIC = {
    "prefill_ttft_seconds": "prefill/decode text tables",
    "decode_tps": "prefill/decode text tables",
    "vision_ttft_seconds": "vision TTFT table",
}
UNIT_BY_METRIC = {
    "prefill_ttft_seconds": "seconds",
    "decode_tps": "tokens_per_second",
    "vision_ttft_seconds": "seconds",
}


def infer_metric(model_variant: str, decode_tokens: int, explicit_metric: str | None = None) -> str:
    if explicit_metric:
        return explicit_metric
    if model_variant.endswith("vision") and decode_tokens == 0:
        return "vision_ttft_seconds"
    if decode_tokens > 0:
        return "decode_tps"
    return "prefill_ttft_seconds"


def find_paper_target(
    *,
    model_variant: str,
    backend: str,
    metric: str,
    sequence_length: int,
) -> dict[str, Any]:
    data = load_targets()
    matches = [
        target
        for target in data["targets"]
        if target.get("model_variant") == model_variant
        and target.get("backend") == backend
        and target.get("metric") == metric
        and int(target.get("sequence_length", -1)) == sequence_length
    ]
    if len(matches) != 1:
        raise ValueError(
            "expected one paper target for "
            f"model={model_variant} backend={backend} metric={metric} "
            f"sequence_length={sequence_length}, got {len(matches)}"
        )
    return matches[0]


def missing_artifact_notes(inventory: Any) -> list[str]:
    notes: list[str] = []
    if not inventory.has_weight_files:
        notes.append("missing *.safetensors")
    if not inventory.config_exists:
        notes.append("missing config.json")
    if not inventory.tokenizer_exists:
        notes.append("missing tokenizer file")
    if getattr(inventory, "has_vision", False) and not inventory.processor_exists:
        notes.append("missing vision processor file")
    if not inventory.optional_packages.get("safetensors", False):
        notes.append("missing python:safetensors")
    if not any(inventory.optional_packages.get(pkg, False) for pkg in ("tokenizers", "sentencepiece", "transformers")):
        notes.append("missing python tokenizer package")
    return notes


def execution_wiring_record_from_plan(plan: Any) -> dict[str, Any]:
    return {
        "status": plan.status,
        "layers": plan.layers,
        "stage_count": plan.stage_count,
        "npu_candidate_count": plan.npu_candidate_count,
        "host_fallback_count": plan.host_fallback_count,
        "host_runtime_count": plan.host_runtime_count,
        "blockers": list(plan.blockers),
    }


def model_runner_record_from_plan(plan: Any) -> dict[str, Any]:
    return {
        "status": plan.status,
        "layers": plan.layers,
        "step_count": plan.step_count,
        "bo_allocation_status": plan.bo_allocation_status,
        "bo_requested_bytes": plan.bo_requested_bytes,
        "bo_allocated_bytes": plan.bo_allocated_bytes,
        "bo_allocation_count": plan.bo_allocation_count,
        "bo_skipped_count": plan.bo_skipped_count,
        "static_preload_tensor_count": plan.static_preload_tensor_count,
        "buffer_binding_status": plan.buffer_binding_status,
        "buffer_binding_count": plan.buffer_binding_count,
        "virtual_buffer_count": plan.virtual_buffer_count,
        "kernel_argument_binding_blocker_count": plan.kernel_argument_binding_blocker_count,
        "kernel_launch_count": plan.kernel_launch_count,
        "host_fallback_count": plan.host_fallback_count,
        "host_runtime_count": plan.host_runtime_count,
        "blockers": list(plan.blockers),
    }


def build_execution_wiring_record(
    *,
    model_variant: str,
    backend: str,
    weights_dir: Path | None,
    artifacts_ready: bool,
) -> dict[str, Any] | None:
    if backend != "npu" or not artifacts_ready:
        return None
    try:
        from gemma3_npu_wiring import build_wiring_plan

        return execution_wiring_record_from_plan(
            build_wiring_plan(model_variant, weights_dir=weights_dir)
        )
    except Exception as exc:
        return {
            "status": "WIRING_FAILED",
            "layers": None,
            "stage_count": None,
            "npu_candidate_count": None,
            "host_fallback_count": None,
            "host_runtime_count": None,
            "blockers": ["npu-wiring-preflight-failed"],
            "error": str(exc),
        }


def build_model_runner_record(
    *,
    model_variant: str,
    backend: str,
    weights_dir: Path | None,
    artifacts_ready: bool,
    prompt_len: int,
) -> dict[str, Any] | None:
    if backend != "npu" or not artifacts_ready:
        return None
    try:
        from gemma3_model_runner import build_model_runner_plan

        return model_runner_record_from_plan(
            build_model_runner_plan(
                model_variant,
                weights_dir=weights_dir,
                prompt_len=prompt_len,
            )
        )
    except Exception as exc:
        return {
            "status": "MODEL_RUNNER_PLAN_FAILED",
            "layers": None,
            "step_count": None,
            "bo_allocation_status": None,
            "bo_requested_bytes": None,
            "bo_allocated_bytes": None,
            "bo_allocation_count": None,
            "bo_skipped_count": None,
            "static_preload_tensor_count": None,
            "buffer_binding_status": None,
            "buffer_binding_count": None,
            "virtual_buffer_count": None,
            "kernel_argument_binding_blocker_count": None,
            "kernel_launch_count": None,
            "host_fallback_count": None,
            "host_runtime_count": None,
            "blockers": ["model-runner-plan-failed"],
            "error": str(exc),
        }


def _paper_delta_pct(target: dict[str, Any], local_value: float) -> tuple[float | None, float | None]:
    if "paper_min" in target and "paper_max" in target:
        low = float(target["paper_min"])
        high = float(target["paper_max"])
        if low <= local_value <= high:
            return 0.0, None
        nearest = low if local_value < low else high
        return abs(local_value - nearest) / nearest * 100.0, nearest
    paper_value = target.get("paper_value")
    if paper_value is None:
        return None, None
    paper_value = float(paper_value)
    if paper_value == 0.0:
        return (0.0 if local_value == 0.0 else float("inf")), paper_value
    return abs(local_value - paper_value) / paper_value * 100.0, paper_value


def _classification_for_delta(delta_pct: float | None) -> str:
    if delta_pct is None:
        return "EXPLAINED_DEVIATION"
    threshold = float(load_targets().get("similarity_threshold_pct", 20.0))
    return "PAPER_MATCH" if delta_pct <= threshold else "EXPLAINED_DEVIATION"


def fallback_records(
    *,
    measure_host_fallbacks: bool = True,
    fallback_timed_iters: int = 3,
) -> list[dict[str, Any]]:
    measurements = (
        measure_cpu_contracts(timed_iters=fallback_timed_iters)
        if measure_host_fallbacks
        else {}
    )
    records: list[dict[str, Any]] = []
    for blocker in paper_match_blockers():
        measurement = measurements.get(blocker.operation, {})
        measured = bool(measurement)
        records.append(
            {
                "name": blocker.operation,
                "status": "measured-host-fallback" if measured else blocker.timed_window_status,
                "contributes_to_timing": True,
                "measured": measured,
                "backend": "cpu-reference" if measured else "host-fallback",
                "elapsed_ms": measurement.get("elapsed_ms"),
                "timed_iters": measurement.get("timed_iters", 0),
                "measurement_source": measurement.get("measurement_source", "not-measured"),
                "tensor_contract": blocker.tensor_contract,
                "implementation_path": blocker.implementation_path,
                "hardware_status": blocker.hardware_status,
                "npu_promoted": blocker.hardware_status == "hardware-validated",
            }
        )
    return records


def build_paper_result(
    *,
    model_variant: str,
    backend: str,
    weights_dir: Path | None,
    tokenizer: Path | None,
    prompt_len: int,
    decode_tokens: int,
    metric: str | None,
    warmup_iters: int,
    timed_iters: int,
    artifact_format: str,
    compile_time_included: bool,
    command: list[str] | None = None,
    power_sample: bool = False,
    trace_size: int | None = None,
    debug_ir: bool = False,
    measure_host_fallbacks: bool = True,
    fallback_timed_iters: int = 3,
) -> dict[str, Any]:
    metric = infer_metric(model_variant, decode_tokens, metric)
    phase = "decode" if metric == "decode_tps" else "prefill"
    spec = model_spec(model_variant)
    spec.validate_sequence_length(prompt_len, phase=phase)
    target = find_paper_target(
        model_variant=model_variant,
        backend=backend,
        metric=metric,
        sequence_length=prompt_len,
    )
    inventory = discover_model_artifacts(model_variant, weights_dir=weights_dir, tokenizer=tokenizer)
    env = capture_environment(
        artifact_format=artifact_format,
        compile_time_included=compile_time_included,
        timing_window="runtime_only",
        require_hardware=False,
    )
    power = capture_power_snapshot(sample=power_sample, run_id=target["id"])
    execution_wiring = build_execution_wiring_record(
        model_variant=model_variant,
        backend=backend,
        weights_dir=weights_dir,
        artifacts_ready=inventory.can_load_real_artifacts,
    )
    model_runner = build_model_runner_record(
        model_variant=model_variant,
        backend=backend,
        weights_dir=weights_dir,
        artifacts_ready=inventory.can_load_real_artifacts,
        prompt_len=prompt_len,
    )

    notes = missing_artifact_notes(inventory)
    host_fallbacks = (
        fallback_records(
            measure_host_fallbacks=measure_host_fallbacks,
            fallback_timed_iters=fallback_timed_iters,
        )
        if backend == "npu"
        else []
    )
    if host_fallbacks:
        if all(item.get("measured") for item in host_fallbacks):
            notes.append(
                "nonlinear host fallbacks measured as CPU-reference microbenchmarks; "
                "not NPU-promoted"
            )
        else:
            notes.append("one or more nonlinear host fallbacks are unmeasured")

    local_value: float | None = None
    delta_pct: float | None = None
    real_benchmark: dict[str, Any] | None = None
    explanation: str | None = None
    if not inventory.can_load_real_artifacts:
        classification = "MISSING_REAL_ARTIFACTS"
        correctness = "BLOCKED_REAL_ARTIFACTS"
    elif backend in ("cpu", "igpu"):
        torch_backend_name = "CPU/HF" if backend == "cpu" else "iGPU/HF ROCm"
        try:
            benchmark = run_torch_benchmark(
                model_variant=model_variant,
                weights_dir=weights_dir,
                max_prompt_tokens=prompt_len,
                metric=metric,
                decode_tokens=decode_tokens,
                warmup_iters=warmup_iters,
                timed_iters=timed_iters,
                power_sample=power_sample,
                run_id=target["id"],
                torch_backend=backend,
            )
        except (Gemma3ExecutionError, ValueError) as exc:
            classification = "REAL_MODEL_EXECUTION_FAILED"
            correctness = "LOCAL_FAIL"
            notes.append(f"real {torch_backend_name} execution failed: {exc}")
        else:
            local_value = float(benchmark.local_value)
            delta_pct, nearest_paper_value = _paper_delta_pct(target, local_value)
            classification = _classification_for_delta(delta_pct)
            correctness = "PASS"
            if classification == "EXPLAINED_DEVIATION":
                explanation = (
                    f"local {torch_backend_name} measurement uses this Strix host and Transformers runtime; "
                    "it is a baseline cell, not validated NPU paper parity"
                )
            real_benchmark = benchmark.to_json_dict()
            notes.extend(benchmark.notes)
            notes.append(
                f"{torch_backend_name} baseline uses local Transformers execution; it is not an NPU paper-parity claim"
            )
            if nearest_paper_value is not None:
                notes.append(f"nearest_paper_value={nearest_paper_value:g}")
            if benchmark.power_snapshot:
                power_data = benchmark.power_snapshot
                power = SimpleNamespace(
                    watts=power_data.get("watts", DEFAULT_POWER_WATTS),
                    field_status=power_data.get("field_status", {}),
                    sampling_backend=power_data.get("sampling_backend"),
                    aligned_with_timed_window=power_data.get("aligned_with_timed_window", False),
                    notes=tuple(power_data.get("notes", ())),
                )
    else:
        classification = "REAL_MODEL_EXECUTION_NOT_IMPLEMENTED"
        correctness = "BLOCKED_EXECUTION_NOT_IMPLEMENTED"
        notes.append("real artifact inventory is present but execution is not implemented")
        if execution_wiring:
            notes.append(
                "execution wiring blockers: "
                + ",".join(execution_wiring.get("blockers", []))
            )
        if model_runner:
            notes.append(
                "model runner blockers: "
                + ",".join(model_runner.get("blockers", []))
            )
    if env.get("missing_paper_fields"):
        notes.append("environment is not paper-comparable: " + ",".join(env["missing_paper_fields"]))
    if power_sample:
        notes.extend(power.notes)
    if trace_size is not None:
        notes.append(f"trace_size={trace_size} requested for future NPU run")
    if debug_ir:
        notes.append("debug_ir requested for future compile run")

    command_text = shlex.join(command if command is not None else sys.argv)
    paper_value = target.get("paper_value")
    if paper_value is None and "paper_min" in target and "paper_max" in target:
        paper_value = None

    return {
        "schema_version": 1,
        "target_id": target["id"],
        "paper_source": target.get("paper_source", "arxiv_pdf_v2"),
        "paper_table": TABLE_BY_METRIC.get(metric, "paper_targets.json"),
        "model_variant": model_variant,
        "backend": backend,
        "metric": metric,
        "sequence_length": prompt_len,
        "decode_tokens": decode_tokens,
        "local_value": local_value,
        "paper_value": paper_value,
        "unit": target.get("unit", UNIT_BY_METRIC.get(metric, "unknown")),
        "delta_pct": delta_pct,
        "classification": classification,
        "correctness": correctness,
        "explanation": explanation,
        "host_fallbacks": host_fallbacks,
        "command": command_text,
        "git_commit": env["git"].get("commit"),
        "dirty_worktree": env["git"].get("dirty_worktree"),
        "xrt_version": env["xrt"].get("version"),
        "npu_power_mode": env["npu"].get("power_mode"),
        "artifact_format": artifact_format,
        "warmup_iters": warmup_iters,
        "timed_iters": timed_iters,
        "compile_time_included": compile_time_included,
        "power_watts": power.watts,
        "power_status": power.field_status,
        "power_sampling_backend": power.sampling_backend,
        "power_aligned_with_timed_window": power.aligned_with_timed_window,
        "environment_comparable": env.get("paper_comparable"),
        "missing_environment_fields": env.get("missing_paper_fields", []),
        "artifact_inventory": inventory.to_json_dict(),
        "execution_wiring": execution_wiring,
        "model_runner": model_runner,
        "real_benchmark": real_benchmark,
        "notes": notes,
    }


def write_result_json(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


def format_result(result: dict[str, Any]) -> str:
    notes = ";".join(result.get("notes", [])) or "none"
    return (
        f"GEMMA3_PAPER_RESULT target={result['target_id']} "
        f"class={result['classification']} correctness={result['correctness']} "
        f"local={result['local_value']} paper={result['paper_value']} notes={notes}"
    )


def _self_test() -> None:
    result = build_paper_result(
        model_variant="gemma3-1b",
        backend="npu",
        weights_dir=Path("/tmp/gemma3_missing_weights"),
        tokenizer=None,
        prompt_len=1024,
        decode_tokens=128,
        metric=None,
        warmup_iters=3,
        timed_iters=10,
        artifact_format="elf",
        compile_time_included=False,
        command=["gemma3_results.py", "--self-test"],
    )
    if result["target_id"] != "decode_tps_gemma3_1b_npu_1024":
        raise AssertionError(result["target_id"])
    if result["classification"] != "MISSING_REAL_ARTIFACTS":
        raise AssertionError(result["classification"])
    if not result["host_fallbacks"]:
        raise AssertionError("expected nonlinear fallback records")
    if any(not item.get("measured") for item in result["host_fallbacks"]):
        raise AssertionError("expected measured host fallback records")
    wiring = execution_wiring_record_from_plan(
        SimpleNamespace(
            status="BLOCKED",
            layers=2,
            stage_count=52,
            npu_candidate_count=20,
            host_fallback_count=28,
            host_runtime_count=4,
            blockers=("model-kernel-launch-not-wired",),
        )
    )
    if wiring["blockers"] != ["model-kernel-launch-not-wired"]:
        raise AssertionError(wiring)
    print(format_result(result))
    print(
        "GEMMA3_EXECUTION_WIRING_RECORD: "
        f"status={wiring['status']} stages={wiring['stage_count']} "
        f"blockers={','.join(wiring['blockers'])}"
    )
    print("GEMMA3_RESULT_JSON_SELF_TEST: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma3 paper-benchmark result JSON helper")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model-variant", choices=sorted(MODEL_SPECS), default="gemma3-1b")
    parser.add_argument("--backend", choices=["cpu", "igpu", "npu"], default="npu")
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--prompt-len", type=int, default=1024)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--metric", choices=sorted(TABLE_BY_METRIC))
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--timed-iters", type=int, default=10)
    parser.add_argument("--artifact-format", choices=["elf", "xclbin"], default="elf")
    parser.add_argument("--compile-time-included", action="store_true")
    parser.add_argument("--power-sample", action="store_true")
    parser.add_argument("--trace-size", type=int)
    parser.add_argument("--debug-ir", action="store_true")
    parser.add_argument("--skip-host-fallback-measurement", action="store_true")
    parser.add_argument("--fallback-timed-iters", type=int, default=3)
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return 0

    result = build_paper_result(
        model_variant=args.model_variant,
        backend=args.backend,
        weights_dir=args.weights_dir,
        tokenizer=args.tokenizer,
        prompt_len=args.prompt_len,
        decode_tokens=args.decode_tokens,
        metric=args.metric,
        warmup_iters=args.warmup_iters,
        timed_iters=args.timed_iters,
        artifact_format=args.artifact_format,
        compile_time_included=args.compile_time_included,
        power_sample=args.power_sample,
        trace_size=args.trace_size,
        debug_ir=args.debug_ir,
        measure_host_fallbacks=not args.skip_host_fallback_measurement,
        fallback_timed_iters=args.fallback_timed_iters,
        command=sys.argv,
    )
    print(format_result(result))
    if result["classification"] == "MISSING_REAL_ARTIFACTS":
        print("GEMMA3_PAPER_BENCHMARK_BLOCKED: missing_real_artifacts")
    if args.result_json:
        write_result_json(result, args.result_json)
        print(f"GEMMA3_RESULT_JSON: {args.result_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
