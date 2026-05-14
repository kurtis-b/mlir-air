# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path
from typing import Any

from .quantization import decode_quantization_plan_from_manifest
from .reference import LinearConfig, workload_bytes

ACCELERATOR_BACKENDS = {"gpu", "npu"}
WEIGHT_TRANSFER_LABELS = {
    "prefill_weights_to_backend",
    "decode_weights_to_backend",
    "decode_packed_weights_to_backend",
    "decode_scales_to_backend",
}


def element_bytes(dtype_name: str) -> int:
    if dtype_name in {"bf16", "f16"}:
        return 2
    raise ValueError(f"unsupported llm-linear dtype: {dtype_name}")


def workload_flops(cfg: LinearConfig) -> dict[str, int]:
    prefill_fma = int(cfg.M * cfg.K * cfg.H)
    decode_fma = int(cfg.H * cfg.N)
    return {
        "prefill_fma": prefill_fma,
        "decode_fma": decode_fma,
        "prefill_flops": int(2 * prefill_fma),
        "decode_flops": int(2 * decode_fma),
        "total_flops": int(2 * (prefill_fma + decode_fma)),
    }


def cpu_native_metadata() -> dict[str, Any]:
    affinity = _cpu_affinity()
    return {
        "backend": "numpy",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "isa_flags": _cpu_flags(),
        "threads": {
            "affinity_count": None if affinity is None else len(affinity),
            "affinity": (
                None if affinity is None else sorted(int(cpu) for cpu in affinity)
            ),
            "env": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
                if os.environ.get(name)
            },
        },
        "cache": cpu_cache_metadata(),
        "blas": numpy_blas_metadata(),
        "frequency": cpu_frequency_metadata(),
    }


def cpu_cache_metadata() -> dict[str, Any]:
    base = Path("/sys/devices/system/cpu/cpu0/cache")
    levels: dict[str, dict[str, Any]] = {}
    if base.exists():
        for index in sorted(base.glob("index*")):
            try:
                level = (index / "level").read_text(encoding="utf-8").strip()
                cache_type = (index / "type").read_text(encoding="utf-8").strip()
                size = _parse_cache_size((index / "size").read_text(encoding="utf-8"))
            except OSError:
                continue
            key = f"L{level}_{cache_type.lower()}"
            levels[key] = {"level": int(level), "type": cache_type, "bytes": size}
    return {
        "levels": levels,
        "l2_bytes": _first_cache_bytes(levels, "L2_"),
        "l3_bytes": _first_cache_bytes(levels, "L3_"),
    }


def numpy_blas_metadata() -> dict[str, Any]:
    try:
        import numpy as np

        config = getattr(np, "__config__", None)
        show_config = getattr(config, "show", None)
        if show_config is None:
            return {"available": False, "backend": "unknown"}
        # NumPy does not expose a stable structured BLAS descriptor across
        # versions. Capture the module path and known thread env instead of
        # scraping the human-oriented show() output.
        return {
            "available": True,
            "numpy_version": np.__version__,
            "config_module": str(getattr(config, "__name__", "numpy.__config__")),
        }
    except Exception as exc:  # pragma: no cover - defensive host metadata path
        return {"available": False, "error": str(exc)}


def cpu_frequency_metadata() -> dict[str, Any]:
    cpufreq = Path("/sys/devices/system/cpu/cpu0/cpufreq")
    fields = {}
    for name in (
        "scaling_governor",
        "scaling_cur_freq",
        "cpuinfo_cur_freq",
        "cpuinfo_min_freq",
        "cpuinfo_max_freq",
    ):
        path = cpufreq / name
        if path.exists():
            try:
                fields[name] = path.read_text(encoding="utf-8").strip()
            except OSError:
                pass
    return {"available": bool(fields), **fields}


def performance_counter_metadata() -> dict[str, Any]:
    return {
        "perf": _tool_counter_status("perf"),
        "rocm": {
            "rocprof": _tool_counter_status("rocprof"),
            "rocminfo": _tool_counter_status("rocminfo"),
            "rocm-smi": _tool_counter_status("rocm-smi"),
        },
        "xrt": {"xrt-smi": _tool_counter_status("xrt-smi")},
    }


def build_performance_proof(
    *,
    cfg: LinearConfig,
    manifest: dict[str, Any],
    last_run: dict[str, Any],
) -> dict[str, Any]:
    stages = dict(manifest["runtime"]["stage_backends"])
    dtype_bytes = element_bytes(cfg.dtype)
    flops = workload_flops(cfg)
    tensors = _tensor_byte_metadata(cfg, manifest, last_run)
    residency = weight_residency_metadata(cfg, manifest, last_run)
    transfer = hot_loop_transfer_metadata(cfg, manifest, residency)
    conversion = cpu_conversion_metadata(cfg, manifest, stages)
    launches = launch_count_metadata(cfg, manifest)
    hot_loop_bytes = max(
        1,
        int(transfer["timed_input_upload_bytes"])
        + int(transfer["timed_output_readback_bytes"])
        + int(transfer["timed_intermediate_host_transfer_bytes"])
        + int(residency["timed_weight_upload_bytes"]),
    )
    cache = cache_fit_metadata(
        int(tensors["static_weight_bytes"]) + int(tensors["per_request_bytes"])
    )
    return {
        "schema_version": 1,
        "implementation": implementation_metadata(manifest),
        "dtype_element_bytes": dtype_bytes,
        "flops": flops,
        "tensor_bytes": tensors,
        "actual_cpu_conversion_bytes": conversion,
        "cache_fit": cache,
        "arithmetic_intensity_flop_per_byte": {
            "logical_tensor_bytes": float(flops["total_flops"])
            / max(1, int(tensors["logical_total_tensor_bytes"])),
            "timed_hot_loop_bytes": float(flops["total_flops"]) / hot_loop_bytes,
        },
        "launches": launches,
        "weight_residency": residency,
        "transfer": transfer,
        "overheads": overhead_proof_metadata(
            last_run=last_run,
            residency=residency,
            transfer=transfer,
            launches=launches,
        ),
        "cpu_native": (
            cpu_native_metadata() if set(stages.values()) == {"cpu"} else None
        ),
        "counters": performance_counter_metadata(),
        "physical_plausibility": {
            "status": "not_evaluated",
            "reason": "hardware counters or calibrated roofline ceilings were not attached to this result",
        },
    }


def implementation_metadata(manifest: dict[str, Any]) -> dict[str, Any]:
    stages = dict(manifest["runtime"]["stage_backends"])
    by_stage = {
        stage: ("native_cpu_numpy" if backend == "cpu" else "air_generated")
        for stage, backend in stages.items()
    }
    backend_set = set(stages.values())
    if backend_set <= ACCELERATOR_BACKENDS:
        kind = "air_generated"
    elif backend_set == {"cpu"}:
        kind = "native_cpu_numpy"
    else:
        kind = "hybrid_air_generated"
    return {
        "kind": kind,
        "by_stage": by_stage,
        "final_winner_eligible": bool(
            kind == "air_generated" and len(backend_set) == 1
        ),
    }


def weight_residency_metadata(
    cfg: LinearConfig, manifest: dict[str, Any], last_run: dict[str, Any]
) -> dict[str, Any]:
    stages = dict(manifest["runtime"]["stage_backends"])
    runtime_report = last_run.get("resident_weights", {})
    requested = bool(
        runtime_report.get(
            "requested", manifest.get("runtime", {}).get("resident_weights", False)
        )
    )
    valid_device_residency = bool(runtime_report.get("valid_device_residency", False))
    static_by_stage = _static_weight_bytes_by_stage(cfg, manifest, last_run)
    resident_by_stage = {}
    for stage, backend in stages.items():
        stage_report = runtime_report.get(stage, {})
        resident_by_stage[stage] = bool(
            backend in ACCELERATOR_BACKENDS
            and stage_report.get("resident")
            and stage_report.get("device_resident")
        )
    timed_upload = 0
    for stage, backend in stages.items():
        if backend in ACCELERATOR_BACKENDS and not resident_by_stage[stage]:
            timed_upload += static_by_stage[stage]
    observed_labels = sorted(
        {
            str(event.get("label"))
            for event in last_run.get("transfer_events", [])
            if str(event.get("label")) in WEIGHT_TRANSFER_LABELS
        }
    )
    return {
        "requested": requested,
        "enabled": valid_device_residency,
        "valid_device_residency": valid_device_residency,
        "proof_status": runtime_report.get(
            "proof_status",
            "not_requested" if not requested else "invalid_or_unavailable",
        ),
        "resident_by_stage": resident_by_stage,
        "stage_proof": {
            stage: runtime_report.get(stage, {}) for stage in ("prefill", "decode")
        },
        "static_weight_bytes_by_stage": static_by_stage,
        "static_weight_bytes": int(sum(static_by_stage.values())),
        "timed_weight_upload_bytes": int(timed_upload),
        "static_upload_excluded_from_timed_region": bool(valid_device_residency),
        "observed_weight_transfer_labels": observed_labels,
        "observed_weight_transfer_count": len(observed_labels),
    }


def hot_loop_transfer_metadata(
    cfg: LinearConfig, manifest: dict[str, Any], residency: dict[str, Any]
) -> dict[str, Any]:
    stages = dict(manifest["runtime"]["stage_backends"])
    bytes_per_elem = element_bytes(cfg.dtype)
    prefill_backend = stages["prefill"]
    decode_backend = stages["decode"]
    same_resident_accelerator = (
        prefill_backend == decode_backend
        and prefill_backend in ACCELERATOR_BACKENDS
        and bool(residency["resident_by_stage"].get("prefill"))
        and bool(residency["resident_by_stage"].get("decode"))
    )
    input_upload = cfg.M * cfg.K * bytes_per_elem if prefill_backend != "cpu" else 0
    output_readback = cfg.N * bytes_per_elem if decode_backend != "cpu" else 0
    if same_resident_accelerator:
        intermediate = 0
        intermediate_residency = "device_same_backend"
    elif prefill_backend != "cpu" or decode_backend != "cpu":
        intermediate = (cfg.M * cfg.H + cfg.H) * bytes_per_elem
        intermediate_residency = "host_staged_numpy"
    else:
        intermediate = 0
        intermediate_residency = "cpu_numpy"
    return {
        "timed_input_upload_bytes": int(input_upload),
        "timed_output_readback_bytes": int(output_readback),
        "timed_input_output_bytes": int(input_upload + output_readback),
        "timed_intermediate_host_transfer_bytes": int(intermediate),
        "timed_weight_upload_bytes": int(residency["timed_weight_upload_bytes"]),
        "full_prefill_semantics_preserved": True,
        "resident_same_backend_decode_row": bool(same_resident_accelerator),
        "intermediate_residency": intermediate_residency,
    }


def overhead_proof_metadata(
    *,
    last_run: dict[str, Any],
    residency: dict[str, Any],
    transfer: dict[str, Any],
    launches: dict[str, Any],
) -> dict[str, Any]:
    execution = last_run.get("execution_overhead", {})
    result = {
        "implementation_kind": implementation_metadata_from_last_run(last_run),
        "timed_allocation_count": int(execution.get("timed_allocation_count", 0)),
        "timed_allocation_count_model": execution.get(
            "timed_allocation_count_model", "not_instrumented"
        ),
        "timed_weight_upload_bytes": int(residency["timed_weight_upload_bytes"]),
        "timed_input_output_transfer_bytes": int(transfer["timed_input_output_bytes"]),
        "timed_input_upload_bytes": int(transfer["timed_input_upload_bytes"]),
        "timed_output_readback_bytes": int(transfer["timed_output_readback_bytes"]),
        "kernel_run_launch_count": int(launches["total"]),
        "host_accumulation_bytes": int(execution.get("host_accumulation_bytes", 0)),
        "intermediate_residency": transfer.get("intermediate_residency"),
        "compile_load_excluded": bool(execution.get("compile_load_excluded", True)),
        "stage_timings_ms": execution.get("stage_timings_ms", {}),
        "residency_proof_status": residency.get("proof_status"),
    }
    native_pipeline = execution.get("native_pipeline")
    if isinstance(native_pipeline, dict):
        result["native_pipeline"] = dict(native_pipeline)
    return result


def implementation_metadata_from_last_run(last_run: dict[str, Any]) -> str:
    by_stage = last_run.get("execution_overhead", {}).get("implementation_by_stage")
    if not isinstance(by_stage, dict):
        return "unknown"
    values = set(str(value) for value in by_stage.values())
    if values == {"air_generated"}:
        return "air_generated"
    if values == {"native_cpu_numpy"}:
        return "native_cpu_numpy"
    return "hybrid_air_generated"


def cpu_conversion_metadata(
    cfg: LinearConfig, manifest: dict[str, Any], stages: dict[str, str]
) -> dict[str, Any]:
    bytes_per_elem = element_bytes(cfg.dtype)
    prefill = 0
    decode = 0
    if stages["prefill"] == "cpu":
        prefill = (cfg.M * cfg.K + cfg.K * cfg.H) * 4 + cfg.M * cfg.H * bytes_per_elem
    if stages["decode"] == "cpu":
        quant = decode_quantization_plan_from_manifest(manifest, shape=(cfg.H, cfg.N))
        if quant is None:
            decode = (cfg.H + cfg.H * cfg.N) * 4 + cfg.N * bytes_per_elem
        else:
            decode = (
                cfg.H * 4
                + quant.packed_bytes
                + quant.scale_bytes
                + quant.zero_point_bytes
                + cfg.H * cfg.N
                + cfg.H * cfg.N * 4
                + cfg.N * bytes_per_elem
            )
    return {
        "prefill_bytes": int(prefill),
        "decode_bytes": int(decode),
        "total_bytes": int(prefill + decode),
        "model": "numpy_float32_compute_with_dtype_quantized_outputs",
    }


def cache_fit_metadata(working_set_bytes: int) -> dict[str, Any]:
    cache = cpu_cache_metadata()
    l2 = cache.get("l2_bytes")
    l3 = cache.get("l3_bytes")
    if isinstance(l2, int) and working_set_bytes <= l2:
        classification = "fits_l2"
    elif isinstance(l3, int) and working_set_bytes <= l3:
        classification = "fits_l3"
    elif isinstance(l3, int):
        classification = "exceeds_l3"
    else:
        classification = "unknown"
    return {
        "classification": classification,
        "working_set_bytes": int(working_set_bytes),
        "l2_bytes": l2,
        "l3_bytes": l3,
    }


def launch_count_metadata(
    cfg: LinearConfig, manifest: dict[str, Any]
) -> dict[str, Any]:
    stages = dict(manifest["runtime"]["stage_backends"])
    prefill = _stage_launch_count("prefill", stages["prefill"], cfg, manifest)
    decode = _stage_launch_count("decode", stages["decode"], cfg, manifest)
    return {
        "prefill": prefill,
        "decode": decode,
        "total": int(prefill + decode),
        "model": "estimated_kernel_invocations_from_backend_tiles",
    }


def _stage_launch_count(
    stage: str, backend: str, cfg: LinearConfig, manifest: dict[str, Any]
) -> int:
    if backend == "cpu":
        return 0
    if stage == "prefill":
        if backend == "npu":
            artifact = manifest.get("artifacts", {}).get("prefill", {}).get("npu", {})
            tile_h = int(artifact.get("tile_h", min(cfg.H, 512)))
            return int(cfg.M * _ceildiv(cfg.H, tile_h))
        return 1

    decode_key = "decode"
    quant = decode_quantization_plan_from_manifest(manifest, shape=(cfg.H, cfg.N))
    if quant is not None and quant.kernel_key == "decode_int4":
        decode_key = "decode_int4"
    artifact = manifest.get("artifacts", {}).get(decode_key, {}).get(backend, {})
    if decode_key == "decode_int4":
        tile_n = int(artifact.get("tile_n", cfg.N))
        return int(_ceildiv(cfg.N, tile_n))
    tile_h = int(artifact.get("tile_h", cfg.H))
    tile_n = int(artifact.get("tile_n", cfg.N))
    return int(_ceildiv(cfg.H, tile_h) * _ceildiv(cfg.N, tile_n))


def _tensor_byte_metadata(
    cfg: LinearConfig, manifest: dict[str, Any], last_run: dict[str, Any]
) -> dict[str, Any]:
    logical = workload_bytes(cfg)
    static_by_stage = _static_weight_bytes_by_stage(cfg, manifest, last_run)
    per_request = int(
        logical["input"]
        + logical["prefill_output"]
        + logical["decode_input"]
        + logical["output"]
    )
    return {
        **logical,
        "logical_total_tensor_bytes": int(logical["total_tensor_bytes"]),
        "static_weight_bytes": int(sum(static_by_stage.values())),
        "per_request_bytes": per_request,
        "decode_storage_adjusted_weight_bytes": int(static_by_stage["decode"]),
    }


def _static_weight_bytes_by_stage(
    cfg: LinearConfig, manifest: dict[str, Any], last_run: dict[str, Any]
) -> dict[str, int]:
    bytes_per_elem = element_bytes(cfg.dtype)
    prefill = int(cfg.K * cfg.H * bytes_per_elem)
    quant = last_run.get("quantized_decode", {})
    if quant.get("enabled"):
        decode = int(quant.get("packed_bytes", 0)) + int(quant.get("scale_bytes", 0))
        decode += int(quant.get("zero_point_bytes", 0))
    else:
        decode = int(cfg.H * cfg.N * bytes_per_elem)
    return {"prefill": prefill, "decode": decode}


def _cpu_affinity() -> set[int] | None:
    try:
        return set(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return None


def _cpu_flags() -> list[str]:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return []
    try:
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            key, _, value = line.partition(":")
            if key.strip() in {"flags", "Features"}:
                return sorted(value.strip().split())
    except OSError:
        return []
    return []


def _parse_cache_size(raw: str) -> int:
    text = raw.strip().upper()
    multiplier = 1
    if text.endswith("K"):
        multiplier = 1024
        text = text[:-1]
    elif text.endswith("M"):
        multiplier = 1024 * 1024
        text = text[:-1]
    return int(float(text) * multiplier)


def _first_cache_bytes(levels: dict[str, dict[str, Any]], prefix: str) -> int | None:
    for key, value in sorted(levels.items()):
        if key.startswith(prefix):
            return int(value["bytes"])
    return None


def _tool_counter_status(name: str) -> dict[str, Any]:
    path = shutil.which(name)
    return {"available": path is not None, "path": path, "captured": False}


def _ceildiv(lhs: int, rhs: int) -> int:
    if rhs <= 0:
        return 0
    return (int(lhs) + int(rhs) - 1) // int(rhs)
