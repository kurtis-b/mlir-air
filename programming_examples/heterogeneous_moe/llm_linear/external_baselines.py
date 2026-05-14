# SPDX-License-Identifier: MIT

from __future__ import annotations

import csv
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from numerics import encode_npu_array

from .manifest import collect_run_metadata, package_dir, save_json
from .reference import (
    LinearConfig,
    LinearWeights,
    config_from_manifest,
    decode_gemv,
    decode_quantization_from_manifest,
    prefill_gemm,
    random_inputs,
    random_weights,
    run_reference,
    stage_metrics,
)

EXTERNAL_BASELINE_SCHEMA_VERSION = "llm-linear-external-baselines-v1"

CSV_FIELDNAMES = [
    "baseline_name",
    "framework",
    "device",
    "scope",
    "suite",
    "workload",
    "M",
    "K",
    "H",
    "N",
    "dtype",
    "decode_weight_storage",
    "iterations",
    "warmup",
    "mean_end_to_end_ms",
    "mean_prefill_ms",
    "mean_decode_ms",
    "speedup_vs_cpu",
    "gap_vs_air_gpu",
    "gap_vs_air_npu",
    "validation_status",
    "output_max_abs_error",
    "device_execution_proof",
    "fallback_status",
    "unsupported_reason",
]

BASELINE_NAMES = [
    "cpu_numpy",
    "rocblas_gpu",
    "torch_rocm",
    "ort_vitisai",
    "iron_npu",
]


@dataclass(frozen=True)
class BaselineContext:
    suite: str
    workload: str
    manifest: dict[str, Any]
    cfg: LinearConfig
    inputs: np.ndarray
    weights: LinearWeights
    reference: dict[str, np.ndarray]
    iterations: int
    warmup: int
    decode_weight_storage: str
    output_dir: Path
    allow_npu: bool = False


def selected_baselines(filters: list[str] | tuple[str, ...]) -> list[str]:
    if not filters:
        return list(BASELINE_NAMES)
    selected = [
        name
        for name in BASELINE_NAMES
        if any(fragment == name or fragment in name for fragment in filters)
    ]
    unknown = [
        fragment for fragment in filters if not any(fragment in n for n in selected)
    ]
    if unknown:
        raise ValueError(f"Unknown baseline filter(s): {', '.join(unknown)}")
    return selected


def make_context(
    *,
    suite: str,
    workload: str,
    manifest: dict[str, Any],
    iterations: int,
    warmup: int,
    decode_weight_storage: str,
    output_dir: Path,
    allow_npu: bool = False,
) -> BaselineContext:
    cfg = config_from_manifest(manifest)
    inputs = random_inputs(
        cfg,
        int(manifest["inputs"]["seed"]),
        scale=float(manifest.get("inputs", {}).get("scale", 0.25)),
    )
    weights = random_weights(
        cfg,
        int(manifest["weights"]["seed"]),
        scale=float(manifest.get("weights", {}).get("scale", 0.125)),
        decode_quantization=decode_quantization_from_manifest(manifest),
    )
    return BaselineContext(
        suite=suite,
        workload=workload,
        manifest=manifest,
        cfg=cfg,
        inputs=inputs,
        weights=weights,
        reference=run_reference(cfg, inputs, weights),
        iterations=iterations,
        warmup=warmup,
        decode_weight_storage=decode_weight_storage,
        output_dir=output_dir,
        allow_npu=allow_npu,
    )


def unsupported_row(
    ctx: BaselineContext,
    *,
    baseline_name: str,
    framework: str,
    device: str,
    scope: str = "pipeline",
    reason: str,
    fallback_status: str = "unsupported",
    proof: str = "",
) -> dict[str, Any]:
    return _row(
        ctx,
        baseline_name=baseline_name,
        framework=framework,
        device=device,
        scope=scope,
        mean_end_to_end_ms=None,
        mean_prefill_ms=None,
        mean_decode_ms=None,
        validation_status="unsupported",
        output_max_abs_error=None,
        device_execution_proof=proof,
        fallback_status=fallback_status,
        unsupported_reason=reason,
    )


def _row(
    ctx: BaselineContext,
    *,
    baseline_name: str,
    framework: str,
    device: str,
    scope: str,
    mean_end_to_end_ms: float | None,
    mean_prefill_ms: float | None,
    mean_decode_ms: float | None,
    validation_status: str,
    output_max_abs_error: float | None,
    device_execution_proof: str,
    fallback_status: str,
    unsupported_reason: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "baseline_name": baseline_name,
        "framework": framework,
        "device": device,
        "scope": scope,
        "suite": ctx.suite,
        "workload": ctx.workload,
        "M": ctx.cfg.M,
        "K": ctx.cfg.K,
        "H": ctx.cfg.H,
        "N": ctx.cfg.N,
        "dtype": ctx.cfg.dtype,
        "decode_weight_storage": ctx.decode_weight_storage,
        "iterations": ctx.iterations,
        "warmup": ctx.warmup,
        "mean_end_to_end_ms": _optional_float(mean_end_to_end_ms),
        "mean_prefill_ms": _optional_float(mean_prefill_ms),
        "mean_decode_ms": _optional_float(mean_decode_ms),
        "speedup_vs_cpu": None,
        "gap_vs_air_gpu": None,
        "gap_vs_air_npu": None,
        "validation_status": validation_status,
        "output_max_abs_error": _optional_float(output_max_abs_error),
        "device_execution_proof": device_execution_proof,
        "fallback_status": fallback_status,
        "unsupported_reason": unsupported_reason,
        "details": extra or {},
    }


def _optional_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _validation_status(
    ctx: BaselineContext, actual: dict[str, np.ndarray]
) -> tuple[str, float | None, dict[str, Any]]:
    metrics = stage_metrics(actual, ctx.reference, ctx.cfg.dtype)
    ok = all(bool(metrics[name]["allclose"]) for name in ("prefill", "output"))
    return (
        "pass" if ok else "fail",
        float(metrics["output"]["max_abs_error"]),
        metrics,
    )


def run_cpu_numpy(ctx: BaselineContext) -> list[dict[str, Any]]:
    prefill_samples: list[float] = []
    decode_samples: list[float] = []
    e2e_samples: list[float] = []
    actual: dict[str, np.ndarray] | None = None

    def run_once() -> dict[str, np.ndarray]:
        start = time.perf_counter_ns()
        prefill_start = start
        prefill = prefill_gemm(ctx.inputs, ctx.weights.prefill, ctx.cfg.dtype)
        prefill_end = time.perf_counter_ns()
        decode_input = np.ascontiguousarray(prefill[ctx.cfg.M - 1, :])
        output = decode_gemv(decode_input, ctx.weights.decode, ctx.cfg.dtype)
        decode_end = time.perf_counter_ns()
        prefill_samples.append((prefill_end - prefill_start) / 1_000_000.0)
        decode_samples.append((decode_end - prefill_end) / 1_000_000.0)
        e2e_samples.append((decode_end - start) / 1_000_000.0)
        return {"prefill": prefill, "decode_input": decode_input, "output": output}

    for _ in range(ctx.warmup):
        run_once()
        prefill_samples.pop()
        decode_samples.pop()
        e2e_samples.pop()
    for _ in range(ctx.iterations):
        actual = run_once()
    assert actual is not None
    validation_status, max_abs_error, metrics = _validation_status(ctx, actual)
    return [
        _row(
            ctx,
            baseline_name="cpu_numpy",
            framework="numpy",
            device="cpu",
            scope="pipeline",
            mean_end_to_end_ms=_mean(e2e_samples),
            mean_prefill_ms=_mean(prefill_samples),
            mean_decode_ms=_mean(decode_samples),
            validation_status=validation_status,
            output_max_abs_error=max_abs_error,
            device_execution_proof="numpy CPU execution; no device baseline claimed",
            fallback_status="native",
            extra={"stage_metrics": metrics, "latencies_ms": e2e_samples},
        )
    ]


def run_rocblas_gpu(ctx: BaselineContext) -> list[dict[str, Any]]:
    if ctx.cfg.dtype != "bf16":
        return [
            unsupported_row(
                ctx,
                baseline_name="rocblas_gpu",
                framework="rocBLAS",
                device="gpu",
                reason=f"rocBLAS baseline currently accepts bf16 only, got {ctx.cfg.dtype}",
            )
        ]
    if ctx.decode_weight_storage != "bf16":
        return [
            unsupported_row(
                ctx,
                baseline_name="rocblas_gpu",
                framework="rocBLAS",
                device="gpu",
                reason=(
                    "native low-bit rocBLAS path is unavailable; "
                    "dequantize-then-BLAS is intentionally not reported as native int4"
                ),
            )
        ]
    hipcc = shutil.which("hipcc")
    if hipcc is None:
        return [
            unsupported_row(
                ctx,
                baseline_name="rocblas_gpu",
                framework="rocBLAS",
                device="gpu",
                reason="hipcc not found on PATH",
            )
        ]

    try:
        executable, build_metadata = build_rocblas_runner(hipcc=hipcc)
    except Exception as exc:
        return [
            unsupported_row(
                ctx,
                baseline_name="rocblas_gpu",
                framework="rocBLAS",
                device="gpu",
                reason=f"rocBLAS runner build failed: {exc}",
            )
        ]

    work_dir = ctx.output_dir / "native_inputs" / ctx.suite / ctx.workload / "rocblas"
    work_dir.mkdir(parents=True, exist_ok=True)
    input_path = work_dir / "input.bf16"
    prefill_weight_path = work_dir / "prefill_weight.bf16"
    decode_weight_path = work_dir / "decode_weight.bf16"
    prefill_output_path = work_dir / "prefill_output.bf16"
    output_path = work_dir / "output.bf16"
    encode_npu_array(ctx.inputs, ctx.cfg.dtype).tofile(input_path)
    encode_npu_array(ctx.weights.prefill, ctx.cfg.dtype).tofile(prefill_weight_path)
    encode_npu_array(ctx.weights.decode, ctx.cfg.dtype).tofile(decode_weight_path)

    cmd = [
        str(executable),
        "--M",
        str(ctx.cfg.M),
        "--K",
        str(ctx.cfg.K),
        "--H",
        str(ctx.cfg.H),
        "--N",
        str(ctx.cfg.N),
        "--warmup",
        str(ctx.warmup),
        "--iterations",
        str(ctx.iterations),
        "--input",
        str(input_path),
        "--prefill-weight",
        str(prefill_weight_path),
        "--decode-weight",
        str(decode_weight_path),
        "--prefill-output",
        str(prefill_output_path),
        "--output",
        str(output_path),
    ]
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=120.0,
        )
    except Exception as exc:
        return [
            unsupported_row(
                ctx,
                baseline_name="rocblas_gpu",
                framework="rocBLAS",
                device="gpu",
                reason=f"rocBLAS runner execution failed: {exc}",
            )
        ]
    if completed.returncode != 0:
        return [
            unsupported_row(
                ctx,
                baseline_name="rocblas_gpu",
                framework="rocBLAS",
                device="gpu",
                reason=(completed.stderr or completed.stdout).strip(),
            )
        ]
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return [
            unsupported_row(
                ctx,
                baseline_name="rocblas_gpu",
                framework="rocBLAS",
                device="gpu",
                reason=f"rocBLAS runner emitted invalid JSON: {exc}",
            )
        ]

    prefill_bits = np.fromfile(prefill_output_path, dtype=np.uint16).reshape(
        ctx.cfg.M, ctx.cfg.H
    )
    output_bits = np.fromfile(output_path, dtype=np.uint16).reshape(ctx.cfg.N)
    from numerics import decode_npu_array

    prefill = decode_npu_array(prefill_bits, ctx.cfg.dtype)
    output = decode_npu_array(output_bits, ctx.cfg.dtype)
    actual = {
        "prefill": prefill,
        "decode_input": np.ascontiguousarray(prefill[ctx.cfg.M - 1, :]),
        "output": output,
    }
    validation_status, max_abs_error, metrics = _validation_status(ctx, actual)
    return [
        _row(
            ctx,
            baseline_name="rocblas_gpu",
            framework="rocBLAS",
            device="gpu",
            scope="pipeline",
            mean_end_to_end_ms=payload["mean_end_to_end_ms"],
            mean_prefill_ms=payload["mean_prefill_ms"],
            mean_decode_ms=payload["mean_decode_ms"],
            validation_status=validation_status,
            output_max_abs_error=max_abs_error,
            device_execution_proof=str(payload.get("device_execution_proof", "")),
            fallback_status="native",
            extra={
                "runner_stdout": payload,
                "build": build_metadata,
                "stage_metrics": metrics,
                "command": cmd,
            },
        )
    ]


def build_rocblas_runner(*, hipcc: str) -> tuple[Path, dict[str, Any]]:
    native_dir = package_dir() / "native"
    source = native_dir / "rocm_blas_baseline.cpp"
    build_dir = package_dir() / "artifacts" / "external_baselines" / "bin"
    build_dir.mkdir(parents=True, exist_ok=True)
    executable = build_dir / "llm_linear_rocm_blas_baseline"
    rocm_path = os.environ.get("ROCM_PATH")
    if not rocm_path:
        rocm_path = (
            "/opt/rocm-7.2.0" if Path("/opt/rocm-7.2.0").is_dir() else "/opt/rocm"
        )
    cmd = [
        hipcc,
        f"--rocm-path={rocm_path}",
        f"--hip-path={rocm_path}",
        "-std=c++17",
        str(source),
        f"-I{rocm_path}/include",
        f"-L{rocm_path}/lib",
        f"-Wl,-rpath,{rocm_path}/lib",
        "-D__AMDGCN_WAVEFRONT_SIZE=32",
        "-lrocblas",
        "-lhipblas",
        "-o",
        str(executable),
    ]
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout).strip())
    return executable, {
        "hipcc": hipcc,
        "rocm_path": rocm_path,
        "source": str(source),
        "executable": str(executable),
        "command": cmd,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def run_torch_rocm(ctx: BaselineContext) -> list[dict[str, Any]]:
    if ctx.decode_weight_storage != "bf16":
        return [
            unsupported_row(
                ctx,
                baseline_name="torch_rocm",
                framework="PyTorch",
                device="gpu",
                reason="PyTorch ROCm native low-bit decode path is not configured",
            )
        ]
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return [
            unsupported_row(
                ctx,
                baseline_name="torch_rocm",
                framework="PyTorch",
                device="gpu",
                reason="torch is not installed",
            )
        ]
    hip_version = getattr(getattr(torch, "version", None), "hip", None)
    cuda = getattr(torch, "cuda", None)
    if not hip_version or cuda is None or not bool(cuda.is_available()):
        return [
            unsupported_row(
                ctx,
                baseline_name="torch_rocm",
                framework="PyTorch",
                device="gpu",
                reason="torch is not a ROCm build with an available HIP device",
                proof=f"torch.version.hip={hip_version!r}",
                fallback_status="fallback",
            )
        ]

    try:
        device = torch.device("cuda")
        input_t = torch.tensor(ctx.inputs, dtype=torch.bfloat16, device=device)
        prefill_w = torch.tensor(
            ctx.weights.prefill, dtype=torch.bfloat16, device=device
        )
        decode_w = torch.tensor(ctx.weights.decode, dtype=torch.bfloat16, device=device)
        samples: list[dict[str, float]] = []
        output_cpu = None
        prefill_cpu = None
        with torch.no_grad():
            for _ in range(ctx.warmup):
                prefill = input_t @ prefill_w
                output = prefill[-1:, :] @ decode_w
                del output
            cuda.synchronize(device)
            for _ in range(ctx.iterations):
                start = torch.cuda.Event(enable_timing=True)
                mid = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                prefill = input_t @ prefill_w
                mid.record()
                output = prefill[-1:, :] @ decode_w
                end.record()
                end.synchronize()
                samples.append(
                    {
                        "prefill": float(start.elapsed_time(mid)),
                        "decode": float(mid.elapsed_time(end)),
                        "end_to_end": float(start.elapsed_time(end)),
                    }
                )
            prefill_cpu = prefill.detach().to(dtype=torch.float32).cpu().numpy()
            output_cpu = (
                output.reshape(-1).detach().to(dtype=torch.float32).cpu().numpy()
            )
    except Exception as exc:
        return [
            unsupported_row(
                ctx,
                baseline_name="torch_rocm",
                framework="PyTorch",
                device="gpu",
                reason=f"PyTorch ROCm execution failed: {exc}",
                fallback_status="fallback",
            )
        ]

    actual = {
        "prefill": np.asarray(prefill_cpu, dtype=np.float32),
        "decode_input": np.ascontiguousarray(prefill_cpu[ctx.cfg.M - 1, :]),
        "output": np.asarray(output_cpu, dtype=np.float32),
    }
    validation_status, max_abs_error, metrics = _validation_status(ctx, actual)
    proof = (
        f"torch.version.hip={hip_version}; "
        f"torch.cuda.is_available=True; device={cuda.get_device_name(device)}"
    )
    return [
        _row(
            ctx,
            baseline_name="torch_rocm",
            framework="PyTorch",
            device="gpu",
            scope="pipeline",
            mean_end_to_end_ms=_mean([sample["end_to_end"] for sample in samples]),
            mean_prefill_ms=_mean([sample["prefill"] for sample in samples]),
            mean_decode_ms=_mean([sample["decode"] for sample in samples]),
            validation_status=validation_status,
            output_max_abs_error=max_abs_error,
            device_execution_proof=proof,
            fallback_status="native",
            extra={"stage_metrics": metrics},
        )
    ]


def run_ort_vitisai(ctx: BaselineContext) -> list[dict[str, Any]]:
    if not ctx.allow_npu:
        return [
            unsupported_row(
                ctx,
                baseline_name="ort_vitisai",
                framework="ONNX Runtime",
                device="npu",
                reason="NPU baselines require --allow-npu",
            )
        ]
    try:
        ort = importlib.import_module("onnxruntime")
    except ImportError:
        return [
            unsupported_row(
                ctx,
                baseline_name="ort_vitisai",
                framework="ONNX Runtime",
                device="npu",
                reason="onnxruntime is not installed",
            )
        ]
    providers = list(getattr(ort, "get_available_providers", lambda: [])())
    if "VitisAIExecutionProvider" not in providers:
        return [
            unsupported_row(
                ctx,
                baseline_name="ort_vitisai",
                framework="ONNX Runtime",
                device="npu",
                reason="VitisAIExecutionProvider is not available",
                proof=f"providers={providers}",
            )
        ]
    return [
        unsupported_row(
            ctx,
            baseline_name="ort_vitisai",
            framework="ONNX Runtime",
            device="npu",
            reason=(
                "VitisAIExecutionProvider is present, but this runner did not get "
                "provider-assignment/profile proof for both MatMul nodes without CPU fallback"
            ),
            proof=f"providers={providers}",
            fallback_status="device_unproven",
        )
    ]


def run_iron_npu(ctx: BaselineContext) -> list[dict[str, Any]]:
    if not ctx.allow_npu:
        return [
            unsupported_row(
                ctx,
                baseline_name="iron_npu",
                framework="IRON",
                device="npu",
                reason="NPU baselines require --allow-npu",
            )
        ]
    if ctx.decode_weight_storage != "bf16":
        return [
            unsupported_row(
                ctx,
                baseline_name="iron_npu",
                framework="IRON",
                device="npu",
                reason="IRON native low-bit LLM-linear decode baseline is not configured",
            )
        ]

    iron_root = Path("/home/cj/iron")
    if not iron_root.is_dir():
        return [
            unsupported_row(
                ctx,
                baseline_name="iron_npu",
                framework="IRON",
                device="npu",
                reason="/home/cj/iron is not present",
            )
        ]

    sys.path.insert(0, str(iron_root))
    try:
        torch = importlib.import_module("torch")
        common = importlib.import_module("iron.common")
        gemm_module = importlib.import_module("iron.operators.gemm.op")
        gemv_module = importlib.import_module("iron.operators.gemv.op")
    except Exception as exc:
        return [
            unsupported_row(
                ctx,
                baseline_name="iron_npu",
                framework="IRON",
                device="npu",
                reason=f"IRON imports failed; source /home/cj/iron/ironenv/bin/activate first: {exc}",
            )
        ]

    try:
        AIEContext = getattr(common, "AIEContext")
        AIEGEMM = getattr(gemm_module, "AIEGEMM")
        AIEGEMV = getattr(gemv_module, "AIEGEMV")
        torch_dtype = torch.bfloat16
        context = AIEContext()
        context.build_dir = ctx.output_dir / "iron_build" / ctx.suite / ctx.workload
        prefill_op = AIEGEMM(
            M=ctx.cfg.M,
            K=ctx.cfg.K,
            N=ctx.cfg.H,
            use_static_weight=True,
            context=context,
        )
        prefill_op.weight = torch.tensor(
            ctx.weights.prefill, dtype=torch_dtype
        ).T.contiguous()
        decode_op = AIEGEMV(
            M=ctx.cfg.N,
            K=ctx.cfg.H,
            use_static_weight=True,
            is_mv=True,
            context=context,
        )
        decode_op.weight = torch.tensor(ctx.weights.decode, dtype=torch_dtype)
        context.compile_all()
        context.prepare_runtime()
        input_t = torch.tensor(ctx.inputs, dtype=torch_dtype)
        decode_input_t = torch.tensor(ctx.reference["decode_input"], dtype=torch_dtype)
    except Exception as exc:
        return [
            unsupported_row(
                ctx,
                baseline_name="iron_npu",
                framework="IRON",
                device="npu",
                reason=f"IRON NPU setup failed: {exc}",
            )
        ]

    rows: list[dict[str, Any]] = []
    try:
        prefill_samples = []
        for _ in range(ctx.warmup):
            prefill_op.forward(input_t)
        prefill_output = None
        for _ in range(ctx.iterations):
            start = time.perf_counter_ns()
            prefill_output = prefill_op.forward(input_t)
            prefill_samples.append((time.perf_counter_ns() - start) / 1_000_000.0)
        prefill_np = prefill_output.to(dtype=torch.float32).cpu().numpy()
        prefill_actual = {
            "prefill": np.asarray(prefill_np, dtype=np.float32),
            "decode_input": ctx.reference["decode_input"],
            "output": ctx.reference["output"],
        }
        validation_status, max_abs_error, metrics = _validation_status(
            ctx, prefill_actual
        )
        rows.append(
            _row(
                ctx,
                baseline_name="iron_npu",
                framework="IRON",
                device="npu",
                scope="prefill",
                mean_end_to_end_ms=_mean(prefill_samples),
                mean_prefill_ms=_mean(prefill_samples),
                mean_decode_ms=None,
                validation_status=validation_status,
                output_max_abs_error=max_abs_error,
                device_execution_proof="IRON AIEGEMM executed through XRT runlist",
                fallback_status="native",
                extra={"stage_metrics": metrics},
            )
        )
    except Exception as exc:
        rows.append(
            unsupported_row(
                ctx,
                baseline_name="iron_npu",
                framework="IRON",
                device="npu",
                scope="prefill",
                reason=f"IRON AIEGEMM execution failed: {exc}",
            )
        )

    try:
        decode_samples = []
        for _ in range(ctx.warmup):
            decode_op.forward(decode_input_t)
        decode_output = None
        for _ in range(ctx.iterations):
            start = time.perf_counter_ns()
            decode_output = decode_op.forward(decode_input_t)
            decode_samples.append((time.perf_counter_ns() - start) / 1_000_000.0)
        output_np = decode_output.to(dtype=torch.float32).cpu().numpy()
        decode_actual = {
            "prefill": ctx.reference["prefill"],
            "decode_input": ctx.reference["decode_input"],
            "output": np.asarray(output_np, dtype=np.float32),
        }
        validation_status, max_abs_error, metrics = _validation_status(
            ctx, decode_actual
        )
        rows.append(
            _row(
                ctx,
                baseline_name="iron_npu",
                framework="IRON",
                device="npu",
                scope="decode",
                mean_end_to_end_ms=_mean(decode_samples),
                mean_prefill_ms=None,
                mean_decode_ms=_mean(decode_samples),
                validation_status=validation_status,
                output_max_abs_error=max_abs_error,
                device_execution_proof="IRON AIEGEMV executed through XRT runlist",
                fallback_status="native",
                extra={"stage_metrics": metrics},
            )
        )
    except Exception as exc:
        rows.append(
            unsupported_row(
                ctx,
                baseline_name="iron_npu",
                framework="IRON",
                device="npu",
                scope="decode",
                reason=f"IRON AIEGEMV execution failed: {exc}",
            )
        )

    rows.append(
        unsupported_row(
            ctx,
            baseline_name="iron_npu",
            framework="IRON",
            device="npu",
            scope="pipeline",
            reason=(
                "device-resident AIEGEMM-to-AIEGEMV handoff is not implemented and "
                "therefore no NPU pipeline timing is claimed"
            ),
            proof="stage-level IRON rows may be present; pipeline proof absent",
            fallback_status="device_unproven",
        )
    )
    return rows


BASELINE_RUNNERS: dict[str, Callable[[BaselineContext], list[dict[str, Any]]]] = {
    "cpu_numpy": run_cpu_numpy,
    "rocblas_gpu": run_rocblas_gpu,
    "torch_rocm": run_torch_rocm,
    "ort_vitisai": run_ort_vitisai,
    "iron_npu": run_iron_npu,
}


def apply_comparison_ratios(
    rows: list[dict[str, Any]],
    *,
    cpu_ms: float | dict[str, float | None] | None,
    air_gpu_ms: float | dict[str, float | None] | None,
    air_npu_ms: float | dict[str, float | None] | None,
) -> None:
    for row in rows:
        if not _eligible_for_speedup(row):
            continue
        row_ms = _row_scope_ms(row)
        if row_ms is None or row_ms <= 0.0:
            continue
        cpu_ref = _reference_ms_for_scope(cpu_ms, row)
        gpu_ref = _reference_ms_for_scope(air_gpu_ms, row)
        npu_ref = _reference_ms_for_scope(air_npu_ms, row)
        if cpu_ref is not None and cpu_ref > 0.0:
            row["speedup_vs_cpu"] = float(cpu_ref / row_ms)
        if gpu_ref is not None and gpu_ref > 0.0:
            row["gap_vs_air_gpu"] = float(row_ms / gpu_ref)
        if npu_ref is not None and npu_ref > 0.0:
            row["gap_vs_air_npu"] = float(row_ms / npu_ref)


def _eligible_for_speedup(row: dict[str, Any]) -> bool:
    if row.get("validation_status") != "pass":
        return False
    if row.get("fallback_status") not in {"native"}:
        return False
    proof = str(row.get("device_execution_proof") or "")
    if row.get("device") != "cpu" and not proof:
        return False
    return True


def _row_scope_ms(row: dict[str, Any]) -> float | None:
    scope = row.get("scope")
    if scope == "prefill":
        return row.get("mean_prefill_ms") or row.get("mean_end_to_end_ms")
    if scope == "decode":
        return row.get("mean_decode_ms") or row.get("mean_end_to_end_ms")
    return row.get("mean_end_to_end_ms")


def _reference_ms_for_scope(
    reference: float | dict[str, float | None] | None, row: dict[str, Any]
) -> float | None:
    if reference is None:
        return None
    if not isinstance(reference, dict):
        return float(reference)
    scope = str(row.get("scope") or "pipeline")
    if scope in reference and reference[scope] is not None:
        return float(reference[scope])
    return None if reference.get("pipeline") is None else float(reference["pipeline"])


def write_outputs(
    output_dir: Path,
    *,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    title: str = "LLM-Linear External Kernel Gap Baselines",
) -> None:
    save_json(output_dir / "summary.json", summary)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=CSV_FIELDNAMES, extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    (output_dir / "report.md").write_text(
        external_baseline_report_markdown(summary, title) + "\n",
        encoding="utf-8",
    )


def external_baseline_report_markdown(summary: dict[str, Any], title: str) -> str:
    lines = [
        f"# {title}",
        "",
        f"- Schema: `{summary.get('schema_version')}`",
        f"- Decode weight storage: `{summary.get('decode_weight_storage')}`",
        "- `speedup_vs_cpu` is CPU mean divided by baseline mean.",
        "- `gap_vs_air_gpu` and `gap_vs_air_npu` are baseline mean divided by AIR mean; `1.0` is parity.",
        "",
        "| Suite | Workload | Baseline | Scope | Device | Mean ms | Validation | Speedup vs CPU | Gap vs AIR GPU | Gap vs AIR NPU | Status |",
        "| --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for workload in summary.get("workloads", []):
        for row in workload.get("rows", []):
            lines.append(
                "| "
                f"{row['suite']} | "
                f"{row['workload']} | "
                f"{row['baseline_name']} | "
                f"{row['scope']} | "
                f"{row['device']} | "
                f"{_format_optional(row.get('mean_end_to_end_ms'))} | "
                f"{row['validation_status']} | "
                f"{_format_optional(row.get('speedup_vs_cpu'))} | "
                f"{_format_optional(row.get('gap_vs_air_gpu'))} | "
                f"{_format_optional(row.get('gap_vs_air_npu'))} | "
                f"{row['fallback_status']} |"
            )
    unsupported = [
        row
        for workload in summary.get("workloads", [])
        for row in workload.get("rows", [])
        if row.get("unsupported_reason")
    ]
    if unsupported:
        lines.extend(["", "## Unsupported Or Unproven"])
        for row in unsupported:
            lines.append(
                f"- `{row['workload']}` / `{row['baseline_name']}` / `{row['scope']}`: "
                f"{row['unsupported_reason']}"
            )
    return "\n".join(lines)


def _format_optional(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def metadata_for_external_run(
    manifest_path: Path, manifest: dict[str, Any], command_line: list[str]
) -> dict[str, Any]:
    metadata = collect_run_metadata(
        manifest_path,
        manifest,
        command_line=command_line,
    )
    metadata["external_baselines"] = {
        "known_baselines": list(BASELINE_NAMES),
        "native_rocm_source": str(package_dir() / "native" / "rocm_blas_baseline.cpp"),
    }
    return metadata
