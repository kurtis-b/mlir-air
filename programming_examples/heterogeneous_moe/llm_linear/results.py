# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
import statistics
import time
from typing import Any

import numpy as np

from .manifest import SCHEMA_VERSION
from .reference import config_from_manifest, random_inputs, workload_bytes

CSV_FIELDNAMES = [
    "suite",
    "workload",
    "case_name",
    "shape_tier",
    "M",
    "K",
    "H",
    "N",
    "dtype",
    "prefill_backend",
    "decode_backend",
    "transfer_mode",
    "mean_end_to_end_ms",
    "mean_prefill_ms",
    "mean_decode_ms",
    "p50_end_to_end_ms",
    "p95_end_to_end_ms",
    "validation_status",
    "output_max_abs_error",
    "transfer_bytes",
    "host_staged_count",
    "device_resident_buffers",
    "direct_handoff_numpy_host_materializations",
    "decode_weight_storage",
    "decode_dequant_ms",
    "decode_linear_ms",
    "packed_weight_bytes_read",
]


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


def _timed_run(runtime: Any, inputs: np.ndarray) -> dict[str, float]:
    return runtime.run(inputs, validate=False, capture_details=False)["timing_ms"]


def _stage_latency_stats(
    samples: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    return {
        name: latency_stats([float(sample[name]) for sample in samples])
        for name in ("prefill", "decode", "end_to_end")
    }


def benchmark_runtime(
    runtime: Any,
    inputs: np.ndarray,
    *,
    iterations: int,
    warmup: int,
    measurement_mode: str,
) -> dict[str, Any]:
    measurement_runs: dict[str, Any] = {}
    phase_timings_ms: dict[str, float] = {}

    if measurement_mode in {"cold", "both"}:
        cold_sample = _timed_run(runtime, inputs)
        cold_stats = _stage_latency_stats([cold_sample])
        measurement_runs["cold_start"] = {
            "iterations": 1,
            "latency_ms": cold_stats["end_to_end"],
            "stage_latency_ms": cold_stats,
        }

    if measurement_mode == "cold":
        samples = [measurement_runs["cold_start"]["stage_latency_ms"]["end_to_end"]]
        primary = [
            {
                "prefill": measurement_runs["cold_start"]["stage_latency_ms"][
                    "prefill"
                ]["mean"],
                "decode": measurement_runs["cold_start"]["stage_latency_ms"]["decode"][
                    "mean"
                ],
                "end_to_end": measurement_runs["cold_start"]["stage_latency_ms"][
                    "end_to_end"
                ]["mean"],
            }
        ]
        effective_warmup = 0
    else:
        warmup_start = time.perf_counter_ns()
        for _ in range(warmup):
            _timed_run(runtime, inputs)
        phase_timings_ms["warmup_total_ms"] = (
            time.perf_counter_ns() - warmup_start
        ) / 1_000_000.0
        effective_warmup = warmup

        primary = []
        timed_start = time.perf_counter_ns()
        for _ in range(iterations):
            primary.append(_timed_run(runtime, inputs))
        phase_timings_ms["timed_total_ms"] = (
            time.perf_counter_ns() - timed_start
        ) / 1_000_000.0
        warm_stats = _stage_latency_stats(primary)
        measurement_runs["warm"] = {
            "iterations": iterations,
            "latency_ms": warm_stats["end_to_end"],
            "stage_latency_ms": warm_stats,
        }

    stage_stats = _stage_latency_stats(primary)
    return {
        "latencies_ms": [float(sample["end_to_end"]) for sample in primary],
        "latency_ms": stage_stats["end_to_end"],
        "stage_latency_ms": stage_stats,
        "measurement_runs": measurement_runs,
        "phase_timings_ms": phase_timings_ms,
        "effective_warmup": effective_warmup,
    }


def generate_inputs(
    runtime: Any, manifest: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any], float]:
    start = time.perf_counter_ns()
    cfg = runtime.cfg
    input_scale = float(manifest.get("inputs", {}).get("scale", 0.25))
    inputs = random_inputs(cfg, int(manifest["inputs"]["seed"]), scale=input_scale)
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
    return inputs, {"input_generation_ms": elapsed_ms}, input_scale


def validate_runtime(runtime: Any, inputs: np.ndarray) -> tuple[dict[str, Any], float]:
    start = time.perf_counter_ns()
    last_run = runtime.run(inputs, validate=True, capture_details=True)
    return last_run, (time.perf_counter_ns() - start) / 1_000_000.0


def correctness_summary(last_run: dict[str, Any]) -> dict[str, Any]:
    metrics = last_run.get("stage_metrics", {})
    output_metrics = metrics.get("output", {})
    prefill_metrics = metrics.get("prefill", {})
    validation = last_run.get("numpy_validation", {})
    return {
        "validated": bool(validation.get("ran")),
        "validation_status": validation.get("status", "skipped"),
        "prefill_allclose": prefill_metrics.get("allclose"),
        "output_allclose": output_metrics.get("allclose"),
        "output_max_abs_error": output_metrics.get("max_abs_error"),
        "output_atol": output_metrics.get("atol"),
        "output_rtol": output_metrics.get("rtol"),
    }


def correctness_failure_message(result: dict[str, Any]) -> str | None:
    correctness = result.get("correctness", {})
    if not correctness.get("validated"):
        return "correctness validation did not run"
    if correctness.get("prefill_allclose") and correctness.get("output_allclose"):
        return None
    return (
        "linear correctness failed: "
        f"prefill_allclose={correctness.get('prefill_allclose')}, "
        f"output_allclose={correctness.get('output_allclose')}, "
        f"output_max_abs_error={correctness.get('output_max_abs_error')}"
    )


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
        "compile_load_excluded": True,
        "compile_load_setup_ms": float(setup_ms),
        "validation_ms": float(validation_ms),
        "runs": measurement_runs,
    }


def build_case_result(
    *,
    metadata: dict[str, Any],
    case_name: str,
    manifest: dict[str, Any],
    iterations: int,
    requested_warmup: int,
    measurement_mode: str,
    timing: dict[str, Any],
    last_run: dict[str, Any],
    validation_ms: float,
    phase_timings_ms: dict[str, float],
    suite: str | None = None,
    workload_name: str | None = None,
) -> dict[str, Any]:
    cfg = config_from_manifest(manifest)
    stage_backends = dict(manifest["runtime"]["stage_backends"])
    transfer_summary = last_run["transfer_summary"]
    measurement = benchmark_measurement_block(
        measurement_mode=measurement_mode,
        iterations=iterations,
        warmup=requested_warmup,
        effective_warmup=timing["effective_warmup"],
        validation_ms=validation_ms,
        measurement_runs=timing["measurement_runs"],
        setup_ms=phase_timings_ms.get("compile_load_setup_ms", 0.0),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": metadata,
        "suite": suite,
        "workload_name": workload_name,
        "case_name": case_name,
        "shape_tier": cfg.shape_tier,
        "dtype": cfg.dtype,
        "shape": {"M": cfg.M, "K": cfg.K, "H": cfg.H, "N": cfg.N},
        "workload": last_run["workload"],
        "placement": {
            "prefill": stage_backends["prefill"],
            "decode": stage_backends["decode"],
        },
        "stage_backends": stage_backends,
        "transfer_mode": manifest["runtime"]["transfer_mode"],
        "transfer_semantics": {
            "requested": manifest["runtime"]["transfer_mode"],
            "actual": transfer_summary.get("transfer_semantics"),
            "device_resident_buffers": bool(
                transfer_summary.get("device_resident_buffers")
            ),
            "direct_handoff_supported": bool(
                transfer_summary.get("direct_handoff", {}).get("supported")
            ),
            "direct_handoff_claimed": bool(
                transfer_summary.get("device_resident_buffers")
            ),
        },
        "iterations": iterations,
        "warmup": timing["effective_warmup"],
        "requested_warmup": requested_warmup,
        "measurement": measurement,
        "timing_breakdown_ms": {
            "prefill_mean_ms": timing["stage_latency_ms"]["prefill"]["mean"],
            "decode_mean_ms": timing["stage_latency_ms"]["decode"]["mean"],
            "end_to_end_mean_ms": timing["stage_latency_ms"]["end_to_end"]["mean"],
        },
        "phase_timings_ms": {
            **timing["phase_timings_ms"],
            **phase_timings_ms,
            "validation_ms": validation_ms,
        },
        "latency_ms": timing["latency_ms"],
        "latencies_ms": timing["latencies_ms"],
        "stage_latency_ms": timing["stage_latency_ms"],
        "correctness": correctness_summary(last_run),
        "validation": last_run["numpy_validation"],
        "stage_metrics": last_run["stage_metrics"],
        "data_movement": {
            "bytes_moved": int(transfer_summary["total_bytes"]),
            "static_tensor_bytes": workload_bytes(cfg),
            "compile_load_excluded": True,
        },
        "transfer_events": last_run["transfer_events"],
        "transfer_summary": transfer_summary,
        "quantized_decode": last_run.get("quantized_decode", {"enabled": False}),
        "trace_summary": last_run["trace_summary"],
        "device_events": last_run["device_events"],
        "npu_development": last_run["npu_development"],
        "execution_truth": {
            "npu_executed": bool(last_run["npu_development"]["executed"]),
            "device_resident_buffers": bool(
                transfer_summary.get("device_resident_buffers")
            ),
            "numpy_host_materializations": int(
                transfer_summary.get("numpy_host_materializations", 0)
            ),
        },
        "limitations": last_run["limitations"],
    }


def result_csv_row(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("stage_metrics", {}).get("output", {})
    transfer = result.get("transfer_summary", {})
    quantized = result.get("quantized_decode", {})
    quant_detail = quantized.get("detail") or {}
    quant_metadata = quantized.get("metadata") or {}
    shape = result["shape"]
    return {
        "suite": result.get("suite"),
        "workload": result.get("workload_name"),
        "case_name": result["case_name"],
        "shape_tier": result["shape_tier"],
        "M": shape["M"],
        "K": shape["K"],
        "H": shape["H"],
        "N": shape["N"],
        "dtype": result["dtype"],
        "prefill_backend": result["stage_backends"]["prefill"],
        "decode_backend": result["stage_backends"]["decode"],
        "transfer_mode": result["transfer_mode"],
        "mean_end_to_end_ms": result["latency_ms"]["mean"],
        "mean_prefill_ms": result["stage_latency_ms"]["prefill"]["mean"],
        "mean_decode_ms": result["stage_latency_ms"]["decode"]["mean"],
        "p50_end_to_end_ms": result["latency_ms"]["p50"],
        "p95_end_to_end_ms": result["latency_ms"]["p95"],
        "validation_status": result["correctness"]["validation_status"],
        "output_max_abs_error": metrics.get("max_abs_error"),
        "transfer_bytes": transfer.get("total_bytes"),
        "host_staged_count": transfer.get("host_staged_count"),
        "device_resident_buffers": transfer.get("device_resident_buffers"),
        "direct_handoff_numpy_host_materializations": transfer.get(
            "direct_handoff_numpy_host_materializations"
        ),
        "decode_weight_storage": (
            quant_metadata.get("quant_kind") if quantized.get("enabled") else "bf16"
        ),
        "decode_dequant_ms": quant_detail.get("dequant_ms"),
        "decode_linear_ms": quant_detail.get("linear_ms"),
        "packed_weight_bytes_read": quant_detail.get("packed_weight_bytes_read"),
    }
