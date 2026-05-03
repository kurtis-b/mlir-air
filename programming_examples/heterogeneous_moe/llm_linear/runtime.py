# SPDX-License-Identifier: MIT

from __future__ import annotations

import ctypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from executors import _SharedLibraryWrapper, _ranked_memref_descriptor
from numerics import (
    decode_npu_array,
    encode_npu_array,
    encoded_array_summary,
    npu_buffer_dtype,
)
from trace import TraceRecorder, summarize_device_events

from .compile import ENTRYPOINTS, compile_gpu, compile_npu, resolve_air_sources
from .direct_bridge import probe_direct_bridge
from .manifest import artifact_root
from .quantization import PackedLinearWeights, decode_gemv_fused_dequant
from .reference import (
    LinearConfig,
    LinearWeights,
    config_from_manifest,
    decode_gemv,
    decode_quantization_from_manifest,
    prefill_gemm,
    random_weights,
    run_reference,
    stage_metrics,
    workload_bytes,
)
from .schema import validate_manifest
from .transfer import DirectTransferUnsupported, LinearTransferManager


class CpuLinearExecutor:
    def __init__(
        self,
        kind: str,
        dtype_name: str,
        decode_quantized: PackedLinearWeights | None = None,
    ) -> None:
        self.kind = kind
        self.dtype_name = dtype_name
        self.decode_quantized = decode_quantized
        self.last_quantized_detail: dict[str, float] | None = None

    def run(self, *arrays: np.ndarray) -> np.ndarray:
        self.last_quantized_detail = None
        if self.kind == "prefill":
            return prefill_gemm(arrays[0], arrays[1], self.dtype_name)
        if self.kind == "decode":
            if self.decode_quantized is not None:
                output, detail = decode_gemv_fused_dequant(
                    arrays[0], self.decode_quantized, self.dtype_name
                )
                self.last_quantized_detail = detail
                return output
            return decode_gemv(arrays[0], arrays[1], self.dtype_name)
        raise ValueError(f"Unknown CPU executor kind: {self.kind}")


class GpuLinearExecutor:
    def __init__(
        self,
        kind: str,
        source: Path,
        artifact: dict[str, Any],
        artifact_root_path: Path,
        arch: str,
        dtype_name: str,
    ) -> None:
        self.kind = kind
        self.source = source
        self.artifact = artifact
        self.artifact_root = artifact_root_path
        self.arch = arch
        self.dtype_name = dtype_name
        self.function_name = ENTRYPOINTS[kind]
        self._library: _SharedLibraryWrapper | None = None

    def prepare(self) -> None:
        if self._library is not None:
            return
        from compile import default_gpu_shared_libs

        compiled = self.artifact
        if not compiled or "so" not in compiled:
            compiled = compile_gpu(
                self.source,
                self.artifact_root / "gpu",
                self.arch,
                self.function_name,
            )
        self.function_name = compiled.get("entry", self.function_name)
        self._library = _SharedLibraryWrapper(
            Path(compiled["so"]), default_gpu_shared_libs()
        )

    def run(self, *arrays: np.ndarray) -> np.ndarray:
        self.prepare()
        output_shape = _output_shape(self.kind, arrays)
        output = np.zeros(output_shape, dtype=npu_buffer_dtype(self.dtype_name))
        descriptors: list[ctypes.Structure] = []
        for array in arrays:
            encoded = encode_npu_array(array, self.dtype_name)
            descriptors.append(_ranked_memref_descriptor(np.ascontiguousarray(encoded)))
        descriptors.append(_ranked_memref_descriptor(output))
        assert self._library is not None
        self._library.invoke(self.function_name, *descriptors)
        return decode_npu_array(output, self.dtype_name)


class NpuLinearExecutor:
    def __init__(
        self,
        kind: str,
        source: Path,
        artifact: dict[str, Any],
        artifact_root_path: Path,
        device: str,
        dtype_name: str,
    ) -> None:
        self.kind = kind
        self.source = source
        self.artifact = artifact
        self.artifact_root = artifact_root_path
        self.device = device
        self.dtype_name = dtype_name
        self._backend = None
        self._invoker = None

    def prepare(self) -> None:
        if self._invoker is not None:
            return
        from air.backend.xrt import XRTBackend, XRTCompileArtifact

        compiled = self.artifact
        if not compiled:
            compiled = compile_npu(self.source, self.artifact_root / "npu", self.device)
        self._backend = XRTBackend(target_device=self.device)
        artifact = XRTCompileArtifact(compiled["xclbin"], "MLIR_AIE", compiled["insts"])
        self._invoker = self._backend.load(artifact)

    def run(self, *arrays: np.ndarray) -> np.ndarray:
        self.prepare()
        output_shape = _output_shape(self.kind, arrays)
        output = np.zeros(output_shape, dtype=npu_buffer_dtype(self.dtype_name))
        encoded_args = [encode_npu_array(array, self.dtype_name) for array in arrays]
        result = self._invoker(*encoded_args, output)
        return decode_npu_array(
            np.asarray(result[-1]).reshape(output_shape), self.dtype_name
        )


@dataclass
class LinearStageExecutors:
    prefill: Any
    decode: Any


def _output_shape(kind: str, arrays: tuple[np.ndarray, ...]) -> tuple[int, ...]:
    if kind == "prefill":
        return (int(arrays[0].shape[0]), int(arrays[1].shape[1]))
    if kind == "decode":
        return (int(arrays[1].shape[1]),)
    raise ValueError(f"Unknown executor kind: {kind}")


class LinearRuntime:
    def __init__(self, manifest: dict[str, Any]) -> None:
        validate_manifest(manifest)
        self.manifest = manifest
        self.cfg = config_from_manifest(manifest)
        self.input_scale = float(manifest.get("inputs", {}).get("scale", 0.25))
        self.weight_scale = float(manifest.get("weights", {}).get("scale", 0.125))
        self.decode_quantization = decode_quantization_from_manifest(manifest)
        self.weights = random_weights(
            self.cfg,
            int(manifest["weights"]["seed"]),
            scale=self.weight_scale,
            decode_quantization=self.decode_quantization,
        )
        self.transfer = LinearTransferManager(manifest["runtime"]["transfer_mode"])
        self.artifact_root = artifact_root(manifest)
        self._sources: dict[str, dict[str, Path]] = {}
        self.executors = self._make_executors()

    def __enter__(self) -> "LinearRuntime":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False

    def _sources_for(self, backend: str) -> dict[str, Path]:
        if backend not in self._sources:
            self._sources[backend] = resolve_air_sources(self.manifest, backend)
        return self._sources[backend]

    def _make_executor(self, kind: str, backend: str) -> Any:
        if backend == "cpu":
            return CpuLinearExecutor(
                kind,
                self.cfg.dtype,
                self.weights.decode_quantized if kind == "decode" else None,
            )

        artifact = self.manifest["artifacts"].get(kind, {}).get(backend, {})
        source = self._sources_for(backend)[kind]
        if backend == "gpu":
            return GpuLinearExecutor(
                kind,
                source,
                artifact,
                self.artifact_root,
                self.manifest["compiler"]["gpu_arch"],
                self.cfg.dtype,
            )
        if backend == "npu":
            return NpuLinearExecutor(
                kind,
                source,
                artifact,
                self.artifact_root,
                self.manifest["compiler"]["npu_device"],
                self.cfg.dtype,
            )
        raise ValueError(f"Unsupported backend: {backend}")

    def _make_executors(self) -> LinearStageExecutors:
        stages = self.manifest["runtime"]["stage_backends"]
        return LinearStageExecutors(
            prefill=self._make_executor("prefill", stages["prefill"]),
            decode=self._make_executor("decode", stages["decode"]),
        )

    def prepare(self) -> None:
        if self.transfer.mode == "direct":
            self._require_direct_runtime()
        for executor in (self.executors.prefill, self.executors.decode):
            prepare = getattr(executor, "prepare", None)
            if prepare:
                prepare()

    def _require_direct_runtime(self) -> None:
        stages = self.manifest["runtime"]["stage_backends"]
        producer = stages["prefill"]
        consumer = stages["decode"]
        self.transfer.require_direct_edge(producer, consumer, "prefill_to_decode")
        status = probe_direct_bridge()
        if not status.available:
            raise DirectTransferUnsupported(
                "transfer_mode=direct requested a GPU/NPU device-resident "
                f"handoff for {producer}->{consumer}, but the native bridge is "
                f"not available: {status.diagnostic}"
            )
        raise DirectTransferUnsupported(
            "transfer_mode=direct found a native bridge probe, but the "
            "llm_linear direct executor is not enabled yet; refusing to fall "
            "back to host staging"
        )

    def _npu_stage_executed(self) -> bool:
        return any(
            backend == "npu"
            for backend in self.manifest["runtime"]["stage_backends"].values()
        )

    def _npu_sources_report(self) -> dict[str, str]:
        if "npu" not in self._sources:
            return {}
        return {name: str(path) for name, path in self._sources["npu"].items()}

    def _npu_development_report(
        self,
        inputs: np.ndarray,
        reference: dict[str, np.ndarray],
        *,
        executed: bool,
    ) -> dict[str, Any]:
        return {
            "executed": bool(executed),
            "dtype": self.cfg.dtype,
            "device": self.manifest["compiler"]["npu_device"],
            "sources": self._npu_sources_report(),
            "artifacts": self.manifest.get("artifacts", {}),
            "prefill": {
                "input": encoded_array_summary(inputs, self.cfg.dtype),
                "weights": encoded_array_summary(self.weights.prefill, self.cfg.dtype),
                "expected_output": encoded_array_summary(
                    reference["prefill"], self.cfg.dtype
                ),
            },
            "decode": {
                "input": encoded_array_summary(
                    reference["decode_input"], self.cfg.dtype
                ),
                "weights": encoded_array_summary(self.weights.decode, self.cfg.dtype),
                "weight_quantization": self._quantization_report(),
                "expected_output": encoded_array_summary(
                    reference["output"], self.cfg.dtype
                ),
            },
            "notes": [
                "Host transfer mode uses NumPy arrays for mixed paths.",
                "Direct device-resident GPU/NPU handoff is gated by the native bridge probe and refuses host fallback.",
            ],
        }

    def _quantization_report(self) -> dict[str, Any]:
        packed = self.weights.decode_quantized
        if packed is None:
            return {"storage": "bf16", "fused_decode": False}
        return {
            "storage": packed.metadata.quant_kind,
            "fused_decode": True,
            "metadata": packed.descriptor(),
            "dequantized_compute_dtype": self.cfg.dtype,
        }

    def _stage_run(
        self,
        executor: Any,
        trace: TraceRecorder | None,
        name: str,
        backend: str,
        *arrays: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        start_ns = time.perf_counter_ns()
        if trace is None:
            output = executor.run(*arrays)
        else:
            with trace.span(name, "stage", name, {"backend": backend}):
                output = executor.run(*arrays)
        return output, (time.perf_counter_ns() - start_ns) / 1_000_000.0

    def run(
        self,
        inputs: np.ndarray,
        *,
        validate: bool = True,
        capture_details: bool = True,
    ) -> dict[str, Any]:
        self.transfer.reset_events()
        stages = self.manifest["runtime"]["stage_backends"]
        trace = TraceRecorder() if capture_details else None
        e2e_start_ns = time.perf_counter_ns()
        if self.transfer.mode == "direct":
            self._require_direct_runtime()

        x_arg = self.transfer.transfer(
            "cpu", stages["prefill"], inputs, trace, "input_to_prefill"
        )
        wp_arg = self.transfer.transfer(
            "cpu",
            stages["prefill"],
            self.weights.prefill,
            trace,
            "prefill_weights_to_backend",
        )
        prefill, prefill_ms = self._stage_run(
            self.executors.prefill,
            trace,
            "prefill_gemm",
            stages["prefill"],
            x_arg,
            wp_arg,
        )
        prefill_host = self.transfer.transfer(
            stages["prefill"], "cpu", prefill, trace, "prefill_to_host"
        )

        decode_input = np.ascontiguousarray(prefill_host[self.cfg.M - 1, :])
        decode_arg = self.transfer.transfer(
            "cpu", stages["decode"], decode_input, trace, "decode_input_to_backend"
        )
        wd_arg = self.transfer.transfer(
            "cpu",
            stages["decode"],
            self.weights.decode,
            trace,
            "decode_weights_to_backend",
        )
        output, decode_ms = self._stage_run(
            self.executors.decode,
            trace,
            "decode_gemv",
            stages["decode"],
            decode_arg,
            wd_arg,
        )
        output_cpu = self.transfer.transfer(
            stages["decode"], "cpu", output, trace, "decode_to_host"
        )
        quantized_decode_detail = getattr(
            self.executors.decode, "last_quantized_detail", None
        )
        e2e_ms = (time.perf_counter_ns() - e2e_start_ns) / 1_000_000.0

        timing_ms = {
            "prefill": float(prefill_ms),
            "decode": float(decode_ms),
            "end_to_end": float(e2e_ms),
        }
        if not capture_details:
            return {
                "prefill": prefill_host,
                "decode_input": decode_input,
                "output": output_cpu,
                "timing_ms": timing_ms,
            }

        actual_bundle = {
            "prefill": prefill_host,
            "decode_input": decode_input,
            "output": output_cpu,
        }
        reference = run_reference(self.cfg, inputs, self.weights)
        if validate:
            per_stage_metrics = stage_metrics(actual_bundle, reference, self.cfg.dtype)
            validation_status = (
                "pass"
                if all(
                    bool(metrics["allclose"]) for metrics in per_stage_metrics.values()
                )
                else "fail"
            )
        else:
            per_stage_metrics = {}
            validation_status = "skipped"

        assert trace is not None
        transfer_summary = self.transfer.summary()
        return {
            "prefill": prefill_host,
            "decode_input": decode_input,
            "output": output_cpu,
            "reference": reference["output"],
            "workload": {
                "shape": {
                    "M": self.cfg.M,
                    "K": self.cfg.K,
                    "H": self.cfg.H,
                    "N": self.cfg.N,
                    "dtype": self.cfg.dtype,
                    "shape_tier": self.cfg.shape_tier,
                },
                "operation": "prefill_gemm_then_decode_gemv",
                "bytes": workload_bytes(self.cfg),
                "decode_weight_quantization": self._quantization_report(),
                "input_scale": self.input_scale,
                "weight_scale": self.weight_scale,
            },
            "timing_ms": timing_ms,
            "stage_metrics": per_stage_metrics,
            "numpy_validation": {
                "ran": bool(validate),
                "ok": validation_status == "pass",
                "status": validation_status,
            },
            "trace": trace,
            "trace_summary": trace.summary(),
            "transfer_events": self.transfer.snapshot(),
            "transfer_summary": transfer_summary,
            "quantized_decode": {
                "enabled": self.weights.decode_quantized is not None,
                "detail": quantized_decode_detail,
                "metadata": (
                    None
                    if self.weights.decode_quantized is None
                    else self.weights.decode_quantized.descriptor()
                ),
            },
            "device_events": summarize_device_events(trace),
            "npu_development": self._npu_development_report(
                inputs, reference, executed=self._npu_stage_executed()
            ),
            "limitations": linear_limitations(self.manifest, transfer_summary),
        }


def linear_limitations(
    manifest: dict[str, Any], transfer_summary: dict[str, Any]
) -> dict[str, Any]:
    return {
        "study_readiness": (
            "direct_handoff_ready"
            if transfer_summary.get("device_resident_buffers")
            else "host_staged_or_fail_closed_direct"
        ),
        "transfer_model": transfer_summary.get("model"),
        "device_resident_buffers": bool(
            transfer_summary.get("device_resident_buffers")
        ),
        "direct_igpu_npu_peer": (
            "supported"
            if transfer_summary.get("device_resident_buffers")
            else "fail_closed_without_native_bridge"
        ),
        "compile_load_excluded_from_timed_iterations": True,
        "limitations": [
            "Milestone 1 measures a two-stage linear pipeline, not full attention, KV-cache, tokenizer, or sampling.",
            "Host transfer mode mixed GPU/NPU paths are staged through NumPy arrays.",
            "transfer_mode=direct refuses host fallback unless a native DeviceResidentTensor bridge records an audited direct edge.",
            "Generated AIR sources are bring-up kernels and are not tuned LLM-scale tilings.",
        ],
        "placement": dict(manifest["runtime"]["stage_backends"]),
    }
