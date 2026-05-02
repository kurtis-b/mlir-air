# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
import statistics
import time
from typing import Any

import numpy as np

from kernels import KernelConfig
from manifest import EDGE_STUDY_SCHEMA_VERSION
from numerics import array_error_metrics
from reference import DEFAULT_INPUT_SCALE, DEFAULT_ROUTING_PROFILE, random_inputs

CSV_FIELDNAMES = [
    "case_name",
    "router_mode",
    "router_backend",
    "expert0_backend",
    "expert1_backend",
    "aggregation_backend",
    "transfer_mode",
    "routing_profile",
    "mean_latency_ms",
    "min_latency_ms",
    "max_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "output_max_abs_error",
    "torch_ok",
    "expert_overlap_us",
    "transfer_bytes",
    "transfer_elapsed_us",
    "npu_executed",
]


def stage_metrics(
    actual: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> dict[str, dict[str, float | bool]]:
    return {
        name: array_error_metrics(actual[name], reference[name], atol=atol, rtol=rtol)
        for name in actual.keys()
    }


def edge_study_limitations(
    manifest: dict[str, Any], transfer_summary: dict[str, Any]
) -> dict[str, Any]:
    workload = manifest.get("workload", {})
    limitations = [
        "Router top-k selection is CPU-side in the current runtime.",
        "Transfer events are measured in the NumPy host-array model, not true device-resident DMA timelines.",
        "Direct iGPU<->NPU peer transfer is not implemented; unsupported direct edges fall back to host staging in auto mode.",
        "This harness measures MoE routing and expert stages, not full transformer attention, KV-cache, or tokenizer overhead.",
    ]
    if (
        workload.get("weight_storage") == "quantized"
        and workload.get("compute_dtype") == "bf16"
    ):
        limitations.append(
            "Preset is labeled quantized storage, but the executable path computes bf16 tensors."
        )
    if workload.get("shared_expert_ffn_size"):
        limitations.append(
            "Shared expert metadata is recorded, but shared-expert execution is not modeled in this harness."
        )
    return {
        "study_readiness": "measurement_infrastructure",
        "routing_topk_location": "cpu",
        "transfer_model": transfer_summary.get("model"),
        "device_resident_buffers": bool(
            transfer_summary.get("device_resident_buffers")
        ),
        "direct_igpu_npu_peer": "unsupported",
        "limitations": limitations,
    }


def latency_stats(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        return {
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "stdev": 0.0,
        }
    ordered = sorted(float(value) for value in latencies_ms)
    return {
        "mean": float(sum(ordered) / len(ordered)),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "p50": _percentile(ordered, 50.0),
        "p95": _percentile(ordered, 95.0),
        "stdev": float(statistics.pstdev(ordered)) if len(ordered) > 1 else 0.0,
    }


def _percentile(ordered: list[float], percentile: float) -> float:
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _timed_run(runtime: Any, inputs: np.ndarray, router_mode: str) -> float:
    start = time.perf_counter_ns()
    runtime.run(
        inputs,
        router_mode=router_mode,
        validate=False,
        capture_details=False,
    )
    return (time.perf_counter_ns() - start) / 1_000_000.0


def logical_config(runtime: Any) -> KernelConfig:
    return KernelConfig(
        batch_tokens=runtime.logical_batch_tokens(),
        hidden_size=runtime.cfg.hidden_size,
        ffn_size=runtime.cfg.ffn_size,
        dtype=runtime.cfg.dtype,
    )


def generate_inputs(
    runtime: Any, manifest: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any], float]:
    start = time.perf_counter_ns()
    input_scale = float(manifest.get("inputs", {}).get("scale", DEFAULT_INPUT_SCALE))
    routing_profile = manifest.get("workload", {}).get(
        "routing_profile", DEFAULT_ROUTING_PROFILE
    )
    inputs = random_inputs(
        logical_config(runtime),
        manifest["inputs"]["seed"],
        scale=input_scale,
        routing_profile=routing_profile,
    )
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
    return inputs, {"input_generation_ms": elapsed_ms}, input_scale


def benchmark_runtime(
    runtime: Any,
    inputs: np.ndarray,
    *,
    router_mode: str,
    iterations: int,
    warmup: int,
    measurement_mode: str,
) -> dict[str, Any]:
    phase_timings_ms: dict[str, float] = {}
    measurement_runs: dict[str, Any] = {}

    if measurement_mode in {"cold", "both"}:
        cold_latency = _timed_run(runtime, inputs, router_mode)
        measurement_runs["cold_start"] = {
            "iterations": 1,
            "latency_ms": latency_stats([cold_latency]),
        }
        phase_timings_ms["cold_start_ms"] = cold_latency

    if measurement_mode == "cold":
        primary_latencies = [measurement_runs["cold_start"]["latency_ms"]["mean"]]
        effective_warmup = 0
    else:
        warmup_start = time.perf_counter_ns()
        for _ in range(warmup):
            runtime.run(
                inputs,
                router_mode=router_mode,
                validate=False,
                capture_details=False,
            )
        phase_timings_ms["warmup_total_ms"] = (
            time.perf_counter_ns() - warmup_start
        ) / 1_000_000.0
        effective_warmup = warmup

        primary_latencies = []
        timed_start = time.perf_counter_ns()
        for _ in range(iterations):
            primary_latencies.append(_timed_run(runtime, inputs, router_mode))
        phase_timings_ms["timed_total_ms"] = (
            time.perf_counter_ns() - timed_start
        ) / 1_000_000.0
        measurement_runs["warm"] = {
            "iterations": iterations,
            "latency_ms": latency_stats(primary_latencies),
        }

    return {
        "latencies_ms": primary_latencies,
        "latency_ms": latency_stats(primary_latencies),
        "measurement_runs": measurement_runs,
        "phase_timings_ms": phase_timings_ms,
        "effective_warmup": effective_warmup,
    }


def validate_runtime(
    runtime: Any, inputs: np.ndarray, *, router_mode: str
) -> tuple[dict[str, Any], float]:
    start = time.perf_counter_ns()
    last_run = runtime.run(
        inputs, router_mode=router_mode, validate=True, capture_details=True
    )
    return last_run, (time.perf_counter_ns() - start) / 1_000_000.0


def correctness_summary(last_run: dict[str, Any]) -> dict[str, Any]:
    output_metrics = last_run.get("stage_metrics", {}).get("output", {})
    torch_validation = last_run.get("torch_validation", {})
    return {
        "validated": bool(last_run.get("stage_metrics")),
        "output_max_abs_error": output_metrics.get("max_abs_error"),
        "output_atol": output_metrics.get("atol"),
        "output_rtol": output_metrics.get("rtol"),
        "output_allclose": output_metrics.get("allclose"),
        "torch_ran": bool(torch_validation.get("ran")),
        "torch_ok": bool(torch_validation.get("ok")),
        "torch_message": torch_validation.get("message"),
    }


def correctness_failure_message(result: dict[str, Any]) -> str | None:
    correctness = result.get("correctness", {})
    if not correctness.get("validated"):
        return "correctness validation did not produce stage metrics"
    if correctness.get("output_allclose"):
        return None

    output_metrics = result.get("stage_metrics", {}).get("output", {})
    max_abs_error = output_metrics.get("max_abs_error")
    atol = output_metrics.get("atol")
    rtol = output_metrics.get("rtol")
    return f"output correctness failed: max_abs_error={max_abs_error}, atol={atol}, rtol={rtol}"


def benchmark_measurement_block(
    *,
    measurement_mode: str,
    iterations: int,
    warmup: int,
    effective_warmup: int,
    validation_ms: float,
    measurement_runs: dict[str, Any],
    setup_ms: float,
) -> dict[str, Any]:
    return {
        "mode": measurement_mode,
        "iterations": iterations,
        "requested_warmup": warmup,
        "effective_warmup": effective_warmup,
        "validation_timed": False,
        "validation_ms": validation_ms,
        "setup_timed": False,
        "compile_load_setup_ms": setup_ms,
        "timing_source": "host_perf_counter_ns",
        "event_source": "untimed_validation_run",
        "runs": measurement_runs,
    }


def build_case_result(
    *,
    schema_version: str,
    metadata: dict[str, Any],
    case_name: str,
    manifest: dict[str, Any],
    iterations: int,
    requested_warmup: int,
    measurement_mode: str,
    timing: dict[str, Any],
    last_run: dict[str, Any],
    validation_ms: float,
    phase_timings_ms: dict[str, Any],
) -> dict[str, Any]:
    setup_ms = float(phase_timings_ms.get("compile_load_setup_ms", 0.0))
    transfer_summary = last_run["transfer_summary"]
    npu_development = last_run["npu_development"]
    return {
        "schema_version": schema_version,
        "metadata": metadata,
        "case_name": case_name,
        "router_mode": manifest["runtime"]["router_mode"],
        "stage_backends": manifest["runtime"]["stage_backends"],
        "transfer_mode": manifest["runtime"]["transfer_mode"],
        "workload": last_run["workload"],
        "iterations": iterations,
        "warmup": timing["effective_warmup"],
        "requested_warmup": requested_warmup,
        "measurement": benchmark_measurement_block(
            measurement_mode=measurement_mode,
            iterations=iterations,
            warmup=requested_warmup,
            effective_warmup=timing["effective_warmup"],
            validation_ms=validation_ms,
            measurement_runs=timing["measurement_runs"],
            setup_ms=setup_ms,
        ),
        "timing_breakdown_ms": {
            "measured_latency": timing["latency_ms"],
            "validation_ms": validation_ms,
            "compile_load_setup_ms": setup_ms,
            "input_generation_ms": float(
                phase_timings_ms.get("input_generation_ms", 0.0)
            ),
        },
        "phase_timings_ms": phase_timings_ms,
        "latency_ms": timing["latency_ms"],
        "latencies_ms": timing["latencies_ms"],
        "max_abs_error": last_run["max_abs_error"],
        "correctness": correctness_summary(last_run),
        "stage_metrics": last_run["stage_metrics"],
        "trace_summary": last_run["trace_summary"],
        "device_events": last_run["device_events"],
        "transfer_events": last_run["transfer_events"],
        "transfer_summary": transfer_summary,
        "data_movement": {
            "transfer_model": transfer_summary.get("model"),
            "device_resident_buffers": bool(
                transfer_summary.get("device_resident_buffers")
            ),
            "host_staged_count": int(transfer_summary.get("host_staged_count", 0)),
            "direct_igpu_npu_peer": "unsupported",
        },
        "torch_validation": last_run["torch_validation"],
        "npu_development": npu_development,
        "execution_truth": {
            "npu_executed": bool(npu_development.get("executed")),
            "timed_latency_includes_validation": False,
            "timed_latency_includes_compile_load_setup": False,
        },
        "limitations": last_run["limitations"],
    }


def result_csv_row(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_name": result["case_name"],
        "router_mode": result["router_mode"],
        "router_backend": result["stage_backends"]["router"],
        "expert0_backend": result["stage_backends"]["expert0"],
        "expert1_backend": result["stage_backends"]["expert1"],
        "aggregation_backend": result["stage_backends"]["aggregation"],
        "transfer_mode": result["transfer_mode"],
        "routing_profile": result["workload"]["routing_profile"],
        "mean_latency_ms": result["latency_ms"]["mean"],
        "min_latency_ms": result["latency_ms"]["min"],
        "max_latency_ms": result["latency_ms"]["max"],
        "p50_latency_ms": result["latency_ms"]["p50"],
        "p95_latency_ms": result["latency_ms"]["p95"],
        "output_max_abs_error": result["stage_metrics"]["output"]["max_abs_error"],
        "torch_ok": result["torch_validation"]["ok"],
        "expert_overlap_us": result["trace_summary"]["overlap"]["expert0_expert1_us"],
        "transfer_bytes": result["transfer_summary"]["total_bytes"],
        "transfer_elapsed_us": result["transfer_summary"]["total_elapsed_us"],
        "npu_executed": result["npu_development"]["executed"],
    }
