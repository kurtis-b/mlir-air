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

from .compile import (
    ENTRYPOINTS,
    GPU_DENSE_DECODE_TILE_H,
    compile_npu_stage,
    compile_gpu,
    decode_kernel_key,
    decode_quantization_plan,
    _npu_dense_decode_tile_metadata,
    resolve_air_sources,
)
from .direct_bridge import (
    DirectBridge,
    DirectBridgeArtifacts,
    DirectBridgeProbeReport,
    probe_direct_bridge,
)
from .kernels import LinearKernelConfig
from .manifest import artifact_root
from .quantization import (
    DecodeHardwareWeightArrays,
    PackedLinearWeights,
    decode_gemv_fused_dequant,
    hardware_decode_weight_arrays,
    validate_accelerator_decode_plan,
)
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
from .transfer import (
    DeviceResidentTensor,
    DirectTransferUnsupported,
    LinearTransferManager,
)


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
        self.last_host_accumulation_bytes = 0

    def run(self, *arrays: np.ndarray) -> np.ndarray:
        self.last_quantized_detail = None
        self.last_host_accumulation_bytes = 0
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
        decode_quantized: PackedLinearWeights | None = None,
        kernel_key: str | None = None,
    ) -> None:
        self.kind = kind
        self.kernel_key = kernel_key or kind
        self.source = source
        self.artifact = artifact
        self.artifact_root = artifact_root_path
        self.arch = arch
        self.dtype_name = dtype_name
        self.decode_quantized = decode_quantized
        self.function_name = ENTRYPOINTS[self.kernel_key]
        self._library: _SharedLibraryWrapper | None = None
        self.last_host_accumulation_bytes = 0

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
        self.last_host_accumulation_bytes = 0
        output_shape = _output_shape(self.kind, arrays)
        if self._uses_row_staged_prefill(arrays, output_shape):
            return self._run_prefill_rows(arrays, output_shape)
        if self._uses_tiled_dense_decode(arrays, output_shape):
            return self._run_decode_dense_tiles(arrays, output_shape)
        output = np.zeros(output_shape, dtype=npu_buffer_dtype(self.dtype_name))
        descriptors: list[ctypes.Structure] = []
        for encoded in _encode_hardware_stage_args(
            self.kind, self.kernel_key, arrays, self.dtype_name
        ):
            descriptors.append(_ranked_memref_descriptor(encoded))
        descriptors.append(_ranked_memref_descriptor(output))
        assert self._library is not None
        self._library.invoke(self.function_name, *descriptors)
        return decode_npu_array(output, self.dtype_name)

    def _uses_row_staged_prefill(
        self, arrays: tuple[np.ndarray, ...], output_shape: tuple[int, ...]
    ) -> bool:
        return (
            self.kind == "prefill"
            and int(output_shape[0]) > 1
            and int(self.artifact.get("source_m", int(output_shape[0]))) == 1
        )

    def _run_prefill_rows(
        self, arrays: tuple[np.ndarray, ...], output_shape: tuple[int, ...]
    ) -> np.ndarray:
        input_array, weights = arrays
        output_dtype = npu_buffer_dtype(self.dtype_name)
        final_output = np.zeros(output_shape, dtype=output_dtype)
        encoded_weights = np.ascontiguousarray(
            encode_npu_array(weights, self.dtype_name)
        )
        assert self._library is not None

        for row in range(int(input_array.shape[0])):
            row_input = np.ascontiguousarray(input_array[row : row + 1, :])
            row_output = np.zeros((1, output_shape[1]), dtype=output_dtype)
            row_args = [
                np.ascontiguousarray(encode_npu_array(row_input, self.dtype_name)),
                encoded_weights,
                row_output,
            ]
            descriptors = [_ranked_memref_descriptor(encoded) for encoded in row_args]
            self._library.invoke(self.function_name, *descriptors)
            final_output[row : row + 1, :] = row_output

        return decode_npu_array(final_output, self.dtype_name)

    def _uses_tiled_dense_decode(
        self, arrays: tuple[np.ndarray, ...], output_shape: tuple[int, ...]
    ) -> bool:
        return (
            self.kind == "decode"
            and self.kernel_key == "decode"
            and (
                int(self.artifact.get("tile_h", arrays[0].shape[0]))
                < int(arrays[0].shape[0])
                or int(self.artifact.get("tile_n", output_shape[0]))
                < int(output_shape[0])
            )
        )

    def _run_decode_dense_tiles(
        self, arrays: tuple[np.ndarray, ...], output_shape: tuple[int, ...]
    ) -> np.ndarray:
        decode_input, weights = arrays
        output_dtype = npu_buffer_dtype(self.dtype_name)
        final_output = np.zeros(output_shape, dtype=np.float32)
        tile_h = int(self.artifact.get("tile_h", int(decode_input.shape[0])))
        tile_n = int(self.artifact.get("tile_n", int(output_shape[0])))
        if tile_h <= 0:
            raise ValueError("GPU dense decode tile_h must be positive")
        if tile_n <= 0:
            raise ValueError("GPU dense decode tile_n must be positive")
        if int(decode_input.shape[0]) % tile_h != 0:
            raise ValueError("GPU dense decode requires H divisible by tile_h")
        if int(output_shape[0]) % tile_n != 0:
            raise ValueError("GPU dense decode requires N divisible by tile_n")
        assert self._library is not None

        for start in range(0, int(output_shape[0]), tile_n):
            accum = np.zeros((tile_n,), dtype=np.float32)
            for h_start in range(0, int(decode_input.shape[0]), tile_h):
                h_stop = h_start + tile_h
                tile_output = np.zeros((tile_n,), dtype=output_dtype)
                tile_args = [
                    np.ascontiguousarray(
                        encode_npu_array(decode_input[h_start:h_stop], self.dtype_name)
                    ),
                    np.ascontiguousarray(
                        encode_npu_array(
                            weights[h_start:h_stop, start : start + tile_n],
                            self.dtype_name,
                        )
                    ),
                    tile_output,
                ]
                descriptors = [
                    _ranked_memref_descriptor(encoded) for encoded in tile_args
                ]
                self._library.invoke(self.function_name, *descriptors)
                partial = decode_npu_array(tile_output, self.dtype_name)
                self.last_host_accumulation_bytes += int(
                    partial.astype(np.float32, copy=False).nbytes
                )
                accum += partial
            final_output[start : start + tile_n] = accum

        return final_output


class NpuLinearExecutor:
    def __init__(
        self,
        kind: str,
        source: Path,
        artifact: dict[str, Any],
        artifact_root_path: Path,
        device: str,
        dtype_name: str,
        decode_quantized: PackedLinearWeights | None = None,
        kernel_key: str | None = None,
    ) -> None:
        self.kind = kind
        self.kernel_key = kernel_key or kind
        self.source = source
        self.artifact = artifact
        self.artifact_root = artifact_root_path
        self.device = device
        self.dtype_name = dtype_name
        self.decode_quantized = decode_quantized
        self._backend = None
        self._invoker = None
        self.last_host_accumulation_bytes = 0

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
        self.last_host_accumulation_bytes = 0
        output_shape = _output_shape(self.kind, arrays)
        if self.kind == "prefill" and int(arrays[0].shape[0]) > 1:
            return self._run_prefill_rows(arrays, output_shape)
        if (
            self.kind == "decode"
            and self.kernel_key == "decode"
            and (
                int(self.artifact.get("tile_h", arrays[0].shape[0]))
                < int(arrays[0].shape[0])
                or int(self.artifact.get("tile_n", output_shape[0]))
                < int(output_shape[0])
            )
        ):
            return self._run_decode_dense_tiles(arrays, output_shape)
        if (
            self.kind == "decode"
            and self.kernel_key == "decode_int4"
            and int(self.artifact.get("tile_n", output_shape[0])) < int(output_shape[0])
        ):
            return self._run_decode_int4_tiles(arrays, output_shape)
        output = np.zeros(output_shape, dtype=npu_buffer_dtype(self.dtype_name))
        encoded_args = _encode_hardware_stage_args(
            self.kind, self.kernel_key, arrays, self.dtype_name
        )
        result = self._invoker(*encoded_args, output)
        return decode_npu_array(
            np.asarray(result[-1]).reshape(output_shape), self.dtype_name
        )

    def _run_prefill_rows(
        self, arrays: tuple[np.ndarray, ...], output_shape: tuple[int, ...]
    ) -> np.ndarray:
        input_array, weights = arrays
        output_dtype = npu_buffer_dtype(self.dtype_name)
        final_output = np.zeros(output_shape, dtype=output_dtype)
        encoded_weights = encode_npu_array(weights, self.dtype_name)
        tile_h = min(int(self.artifact.get("tile_h", 512)), int(output_shape[1]))
        row_output_shape = (1, tile_h)

        # Current NPU prefill kernels write row 0 for the medium multi-row
        # host baseline, so stage each requested row through row 0 explicitly.
        for row in range(int(input_array.shape[0])):
            staged_input = np.ascontiguousarray(input_array[row : row + 1, :])
            encoded_input = encode_npu_array(staged_input, self.dtype_name)
            for start in range(0, int(output_shape[1]), tile_h):
                stop = min(start + tile_h, int(output_shape[1]))
                width = stop - start
                staged_weights = np.zeros(
                    (weights.shape[0], tile_h), dtype=weights.dtype
                )
                staged_weights[:, :width] = weights[:, start:stop]
                row_output = np.zeros(row_output_shape, dtype=output_dtype)
                result = self._invoker(
                    encoded_input,
                    encode_npu_array(staged_weights, self.dtype_name),
                    row_output,
                )
                encoded_result = np.asarray(result[-1]).reshape(row_output_shape)
                final_output[row, start:stop] = encoded_result[0, :width]

        return decode_npu_array(final_output, self.dtype_name)

    def _run_decode_dense_tiles(
        self, arrays: tuple[np.ndarray, ...], output_shape: tuple[int, ...]
    ) -> np.ndarray:
        decode_input, weights = arrays
        output_dtype = npu_buffer_dtype(self.dtype_name)
        final_output = np.zeros(output_shape, dtype=np.float32)
        tile_h = int(self.artifact.get("tile_h", int(decode_input.shape[0])))
        tile_n = int(self.artifact.get("tile_n", output_shape[0]))
        if tile_h <= 0:
            raise ValueError("NPU dense decode tile_h must be positive")
        if tile_n <= 0:
            raise ValueError("NPU dense decode tile_n must be positive")
        if int(decode_input.shape[0]) % tile_h != 0:
            raise ValueError("NPU dense decode requires H divisible by tile_h")
        if int(output_shape[0]) % tile_n != 0:
            raise ValueError("NPU dense decode requires N divisible by tile_n")

        for start in range(0, int(output_shape[0]), tile_n):
            accum = np.zeros((tile_n,), dtype=np.float32)
            for h_start in range(0, int(decode_input.shape[0]), tile_h):
                h_stop = h_start + tile_h
                tile_output = np.zeros((tile_n,), dtype=output_dtype)
                tile_input = np.ascontiguousarray(
                    encode_npu_array(decode_input[h_start:h_stop], self.dtype_name)
                )
                tile_weights = np.ascontiguousarray(
                    encode_npu_array(
                        weights[h_start:h_stop, start : start + tile_n],
                        self.dtype_name,
                    )
                )
                result = self._invoker(tile_input, tile_weights, tile_output)
                partial = decode_npu_array(
                    np.asarray(result[-1]).reshape((tile_n,)), self.dtype_name
                )
                self.last_host_accumulation_bytes += int(
                    partial.astype(np.float32, copy=False).nbytes
                )
                accum += partial
            final_output[start : start + tile_n] = accum

        return final_output

    def _run_decode_int4_tiles(
        self, arrays: tuple[np.ndarray, ...], output_shape: tuple[int, ...]
    ) -> np.ndarray:
        decode_input, packed_weights, scales = arrays
        output_dtype = npu_buffer_dtype(self.dtype_name)
        final_output = np.zeros(output_shape, dtype=output_dtype)
        tile_n = int(self.artifact.get("tile_n", output_shape[0]))
        if tile_n <= 0 or tile_n % 8 != 0:
            raise ValueError("NPU int4 decode tile_n must be a positive multiple of 8")
        if int(output_shape[0]) % tile_n != 0:
            raise ValueError("NPU int4 decode requires N divisible by tile_n")

        encoded_input = np.ascontiguousarray(
            encode_npu_array(decode_input, self.dtype_name)
        )
        packed_words_per_tile = tile_n // 8
        for start in range(0, int(output_shape[0]), tile_n):
            packed_start = start // 8
            packed_stop = packed_start + packed_words_per_tile
            tile_output = np.zeros((tile_n,), dtype=output_dtype)
            result = self._invoker(
                encoded_input,
                np.ascontiguousarray(packed_weights[:, packed_start:packed_stop]),
                np.ascontiguousarray(scales[:, start : start + tile_n]),
                tile_output,
            )
            final_output[start : start + tile_n] = np.asarray(result[-1]).reshape(
                (tile_n,)
            )

        return decode_npu_array(final_output, self.dtype_name)


@dataclass
class LinearStageExecutors:
    prefill: Any
    decode: Any


def kernel_config_from_runtime_cfg(cfg: LinearConfig) -> LinearKernelConfig:
    return LinearKernelConfig(M=cfg.M, K=cfg.K, H=cfg.H, N=cfg.N, dtype=cfg.dtype)


@dataclass(frozen=True)
class DirectDecodeBuffers:
    storage: str
    weights: np.ndarray | None
    packed_weights: np.ndarray | None
    scales: np.ndarray | None
    block_size: int
    quant_axis: int


def _output_shape(kind: str, arrays: tuple[np.ndarray, ...]) -> tuple[int, ...]:
    if kind == "prefill":
        return (int(arrays[0].shape[0]), int(arrays[1].shape[1]))
    if kind == "decode":
        if len(arrays) == 3:
            return (int(arrays[2].shape[1]),)
        return (int(arrays[1].shape[1]),)
    raise ValueError(f"Unknown executor kind: {kind}")


def _encode_hardware_stage_args(
    kind: str,
    kernel_key: str,
    arrays: tuple[np.ndarray, ...],
    dtype_name: str,
) -> list[np.ndarray]:
    if kernel_key == "decode_int4":
        if kind != "decode" or len(arrays) != 3:
            raise ValueError("decode_int4 expects input, packed weights, and scales")
        return [
            np.ascontiguousarray(encode_npu_array(arrays[0], dtype_name)),
            np.ascontiguousarray(np.asarray(arrays[1], dtype=np.uint32)),
            np.ascontiguousarray(np.asarray(arrays[2], dtype=np.float32)),
        ]
    return [
        np.ascontiguousarray(encode_npu_array(array, dtype_name)) for array in arrays
    ]


def _raw_array_summary(array: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": str(contiguous.dtype),
        "shape": [int(dim) for dim in contiguous.shape],
        "nbytes": int(contiguous.nbytes),
    }


class LinearRuntime:
    def __init__(self, manifest: dict[str, Any]) -> None:
        validate_manifest(manifest)
        self.manifest = manifest
        self.cfg = config_from_manifest(manifest)
        self.input_scale = float(manifest.get("inputs", {}).get("scale", 0.25))
        self.weight_scale = float(manifest.get("weights", {}).get("scale", 0.125))
        self.decode_quantization_plan = decode_quantization_plan(manifest)
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
        self._direct_bridge: DirectBridge | None = None
        self._direct_probe_report: DirectBridgeProbeReport | None = None
        self._decode_hardware_arrays: DecodeHardwareWeightArrays | None = None
        self.executors = self._make_executors()

    def __enter__(self) -> "LinearRuntime":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        self.release()
        return False

    def _sources_for(self, backend: str) -> dict[str, Path]:
        if backend not in self._sources:
            stages = self.manifest["runtime"]["stage_backends"]
            include_decode_int4 = (
                stages.get("decode") == backend
                and self.decode_quantization_plan is not None
                and self._kernel_key_for_stage("decode", backend) == "decode_int4"
            )
            self._sources[backend] = resolve_air_sources(
                self.manifest,
                backend,
                include_decode_int4=include_decode_int4,
            )
        return self._sources[backend]

    def _kernel_key_for_stage(self, kind: str, backend: str) -> str:
        if kind == "decode" and backend in {"gpu", "npu"}:
            if self.decode_quantization_plan is not None:
                return decode_kernel_key(self.manifest)
        return kind

    def _decode_hardware_fused(self) -> bool:
        backend = self.manifest["runtime"]["stage_backends"]["decode"]
        return (
            backend in {"gpu", "npu"}
            and self.weights.decode_quantized is not None
            and self.decode_quantization_plan is not None
            and self.decode_quantization_plan.hardware_fused
            and self._kernel_key_for_stage("decode", backend) == "decode_int4"
        )

    def _decode_weight_arrays_for_backend(self, backend: str) -> tuple[np.ndarray, ...]:
        if backend in {"gpu", "npu"} and self.weights.decode_quantized is not None:
            arrays = self._decode_hardware_weight_arrays()
            return arrays.packed, arrays.scales
        return (self.weights.decode,)

    def _decode_hardware_weight_arrays(self) -> DecodeHardwareWeightArrays:
        if self._decode_hardware_arrays is not None:
            return self._decode_hardware_arrays
        validate_accelerator_decode_plan(
            self.decode_quantization_plan, require_int4=True
        )
        if self.weights.decode_quantized is None:
            raise ValueError("hardware fused decode requires quantized weights")
        self._decode_hardware_arrays = hardware_decode_weight_arrays(
            self.weights.decode_quantized,
            npu_decode_tile_n=(
                None
                if self.decode_quantization_plan is None
                else self.decode_quantization_plan.npu_decode_tile_n
            ),
        )
        return self._decode_hardware_arrays

    def _make_executor(self, kind: str, backend: str) -> Any:
        if backend == "cpu":
            return CpuLinearExecutor(
                kind,
                self.cfg.dtype,
                self.weights.decode_quantized if kind == "decode" else None,
            )

        kernel_key = self._kernel_key_for_stage(kind, backend)
        artifact = dict(self.manifest["artifacts"].get(kernel_key, {}).get(backend, {}))
        source = self._sources_for(backend)[kernel_key]
        if backend == "gpu":
            if kernel_key == "decode":
                artifact.setdefault("tile_h", min(self.cfg.H, GPU_DENSE_DECODE_TILE_H))
                artifact.setdefault("tile_n", self.cfg.N)
            return GpuLinearExecutor(
                kind,
                source,
                artifact,
                self.artifact_root,
                self.manifest["compiler"]["gpu_arch"],
                self.cfg.dtype,
                self.weights.decode_quantized if kernel_key == "decode_int4" else None,
                kernel_key=kernel_key,
            )
        if backend == "npu":
            if kernel_key == "decode":
                artifact.update(
                    {
                        key: artifact.get(key, value)
                        for key, value in _npu_dense_decode_tile_metadata(
                            kernel_config_from_runtime_cfg(self.cfg)
                        ).items()
                    }
                )
            return NpuLinearExecutor(
                kind,
                source,
                artifact,
                self.artifact_root,
                self.manifest["compiler"]["npu_device"],
                self.cfg.dtype,
                self.weights.decode_quantized if kernel_key == "decode_int4" else None,
                kernel_key=kernel_key,
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
            self._ensure_direct_artifacts()
            return
        for executor in (self.executors.prefill, self.executors.decode):
            prepare = getattr(executor, "prepare", None)
            if prepare:
                prepare()

    def run_hot(self, inputs: np.ndarray) -> dict[str, Any]:
        return self.run(inputs, validate=False, capture_details=False)

    def release(self) -> None:
        for executor in (self.executors.prefill, self.executors.decode):
            release = getattr(executor, "release", None)
            if release:
                release()

    def _require_direct_runtime(self) -> DirectBridge:
        stages = self.manifest["runtime"]["stage_backends"]
        producer = stages["prefill"]
        consumer = stages["decode"]
        self.transfer.require_direct_edge(producer, consumer, "prefill_to_decode")
        if self._direct_bridge is not None:
            return self._direct_bridge
        status = probe_direct_bridge()
        self._direct_probe_report = getattr(status, "probe_report", None)
        if not status.available:
            raise DirectTransferUnsupported(
                "transfer_mode=direct requested a GPU/NPU device-resident "
                f"handoff for {producer}->{consumer}, but the native bridge is "
                f"not available: {status.diagnostic}"
            )
        if self._direct_bridge is None:
            try:
                self._direct_bridge = DirectBridge(status.library_path)
            except RuntimeError as exc:
                raise DirectTransferUnsupported(str(exc)) from exc
        return self._direct_bridge

    def _ensure_direct_artifacts(self) -> None:
        stages = self.manifest["runtime"]["stage_backends"]
        artifacts = self.manifest.setdefault("artifacts", {})
        compiler_cfg = self.manifest["compiler"]
        if stages["prefill"] == "gpu" or stages["decode"] == "gpu":
            gpu_sources = self._sources_for("gpu")
            direct_dir = self.artifact_root / "gpu_direct"
            for kind in ("prefill", "decode"):
                if stages[kind] != "gpu":
                    continue
                kernel_key = self._kernel_key_for_stage(kind, "gpu")
                artifact_entry = artifacts.setdefault(kernel_key, {})
                if "gpu_direct" not in artifact_entry:
                    artifact_entry["gpu_direct"] = compile_gpu(
                        gpu_sources[kernel_key],
                        direct_dir,
                        compiler_cfg["gpu_arch"],
                        ENTRYPOINTS[kernel_key],
                        host_staging=False,
                    )
                if kernel_key == "decode":
                    artifact_entry["gpu_direct"].setdefault(
                        "tile_h", min(self.cfg.H, GPU_DENSE_DECODE_TILE_H)
                    )
                    artifact_entry["gpu_direct"].setdefault("tile_n", self.cfg.N)
        if stages["prefill"] == "npu" or stages["decode"] == "npu":
            npu_sources = self._sources_for("npu")
            npu_dir = self.artifact_root / "npu"
            for kind in ("prefill", "decode"):
                if stages[kind] != "npu":
                    continue
                kernel_key = self._kernel_key_for_stage(kind, "npu")
                artifact_entry = artifacts.setdefault(kernel_key, {})
                if "npu" not in artifact_entry:
                    artifact_entry["npu"] = compile_npu_stage(
                        kernel_key,
                        npu_sources[kernel_key],
                        npu_dir,
                        compiler_cfg["npu_device"],
                    )
                if kernel_key == "decode":
                    for key, value in _npu_dense_decode_tile_metadata(
                        kernel_config_from_runtime_cfg(self.cfg)
                    ).items():
                        artifact_entry["npu"].setdefault(key, value)

    def _direct_artifacts(self) -> DirectBridgeArtifacts:
        artifacts = self.manifest.get("artifacts", {})
        stages = self.manifest["runtime"]["stage_backends"]
        decode_key = self._kernel_key_for_stage("decode", stages["decode"])
        prefill_gpu = artifacts.get("prefill", {}).get("gpu_direct", {})
        decode_gpu = artifacts.get(decode_key, {}).get("gpu_direct", {})
        prefill_npu = artifacts.get("prefill", {}).get("npu", {})
        decode_npu = artifacts.get(decode_key, {}).get("npu", {})
        return DirectBridgeArtifacts(
            gpu_prefill_so=prefill_gpu.get("so"),
            gpu_decode_so=decode_gpu.get("so"),
            gpu_decode_tile_h=decode_gpu.get("tile_h"),
            gpu_decode_tile_n=decode_gpu.get("tile_n"),
            npu_prefill_xclbin=prefill_npu.get("xclbin"),
            npu_prefill_insts=prefill_npu.get("insts"),
            npu_decode_xclbin=decode_npu.get("xclbin"),
            npu_decode_insts=decode_npu.get("insts"),
            npu_decode_tile_h=decode_npu.get("tile_h"),
            npu_decode_tile_n=decode_npu.get("tile_n"),
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
                "weights": self._decode_weights_report(),
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

    def _decode_weights_report(self) -> dict[str, Any]:
        packed = self.weights.decode_quantized
        if packed is None:
            return {"dense": encoded_array_summary(self.weights.decode, self.cfg.dtype)}
        try:
            arrays = self._decode_hardware_weight_arrays()
            packed_values = arrays.packed
            scales = arrays.scales
        except ValueError:
            packed_values = np.ascontiguousarray(
                np.asarray(packed.packed, dtype=np.uint8)
            )
            scales = np.ascontiguousarray(np.asarray(packed.scales, dtype=np.float32))
        return {
            "dense_reference": encoded_array_summary(
                self.weights.decode, self.cfg.dtype
            ),
            "packed_weights": _raw_array_summary(packed_values),
            "scales": _raw_array_summary(scales),
        }

    def _quantization_report(self) -> dict[str, Any]:
        packed = self.weights.decode_quantized
        if packed is None:
            return {"storage": "bf16", "fused_decode": False}
        plan = self.decode_quantization_plan
        return {
            "storage": packed.metadata.quant_kind,
            "fused_decode": True,
            "hardware_fused": bool(plan.hardware_fused if plan is not None else False),
            "metadata": packed.descriptor(),
            "plan": None if plan is None else plan.to_metadata_dict(),
            "dequantized_compute_dtype": self.cfg.dtype,
        }

    def _quantized_decode_report(
        self,
        *,
        hardware_fused: bool,
        detail: dict[str, float] | None,
    ) -> dict[str, Any]:
        packed = self.weights.decode_quantized
        if packed is None:
            return {
                "enabled": False,
                "quant_kind": None,
                "hardware_fused": False,
                "detail": detail,
                "metadata": None,
                "packed_bytes": 0,
                "scale_bytes": 0,
                "zero_point_bytes": 0,
            }
        plan = self.decode_quantization_plan
        return {
            "enabled": True,
            "quant_kind": packed.metadata.quant_kind,
            "hardware_fused": bool(hardware_fused),
            "detail": detail,
            "metadata": packed.descriptor(),
            "plan": None if plan is None else plan.to_metadata_dict(),
            "packed_bytes": (
                int(packed.packed.nbytes) if plan is None else plan.packed_bytes
            ),
            "scale_bytes": (
                int(packed.scales.nbytes) if plan is None else plan.scale_bytes
            ),
            "zero_point_bytes": int(
                0 if packed.zero_points is None else packed.zero_points.nbytes
            ),
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

    def _execution_overhead_report(
        self,
        *,
        timing_ms: dict[str, float],
        transfer_summary: dict[str, Any],
    ) -> dict[str, Any]:
        host_accumulation_bytes = int(
            getattr(self.executors.decode, "last_host_accumulation_bytes", 0) or 0
        )
        return {
            "implementation_by_stage": self._implementation_by_stage(),
            "timed_allocation_count": int(transfer_summary.get("copied_count", 0)),
            "timed_allocation_count_model": (
                "observed_numpy_transfer_copies_only; backend-internal "
                "allocations require hardware counters or native runtime hooks"
            ),
            "host_accumulation_bytes": host_accumulation_bytes,
            "stage_timings_ms": {key: float(value) for key, value in timing_ms.items()},
            "compile_load_excluded": True,
        }

    def _implementation_by_stage(self) -> dict[str, str]:
        stages = self.manifest["runtime"]["stage_backends"]
        return {
            stage: ("native_cpu_numpy" if backend == "cpu" else "air_generated")
            for stage, backend in stages.items()
        }

    def _direct_decode_buffers(self) -> DirectDecodeBuffers:
        if self.weights.decode_quantized is None:
            return DirectDecodeBuffers(
                storage="dense",
                weights=encode_npu_array(self.weights.decode, self.cfg.dtype),
                packed_weights=None,
                scales=None,
                block_size=0,
                quant_axis=0,
            )
        arrays = self._decode_hardware_weight_arrays()
        return DirectDecodeBuffers(
            storage="int4",
            weights=None,
            packed_weights=arrays.packed,
            scales=arrays.scales,
            block_size=arrays.plan.block_size,
            quant_axis=arrays.plan.quant_axis,
        )

    def _run_direct(
        self,
        inputs: np.ndarray,
        *,
        validate: bool,
        capture_details: bool,
        trace: TraceRecorder | None,
        e2e_start_ns: int,
    ) -> dict[str, Any]:
        bridge = self._require_direct_runtime()
        self._ensure_direct_artifacts()
        stages = self.manifest["runtime"]["stage_backends"]
        direction = (
            "gpu_prefill_npu_decode"
            if stages["prefill"] == "gpu"
            else "npu_prefill_gpu_decode"
        )
        input_encoded = encode_npu_array(inputs, self.cfg.dtype)
        prefill_weights_encoded = encode_npu_array(self.weights.prefill, self.cfg.dtype)
        decode_buffers = self._direct_decode_buffers()
        output_encoded = np.zeros((self.cfg.N,), dtype=npu_buffer_dtype(self.cfg.dtype))
        prefill_encoded = (
            np.zeros((self.cfg.M, self.cfg.H), dtype=npu_buffer_dtype(self.cfg.dtype))
            if capture_details
            else None
        )
        decode_input_encoded = (
            np.zeros((self.cfg.H,), dtype=npu_buffer_dtype(self.cfg.dtype))
            if capture_details
            else None
        )

        def invoke_bridge() -> Any:
            return bridge.run(
                direction=direction,
                dtype=self.cfg.dtype,
                shape=(self.cfg.M, self.cfg.K, self.cfg.H, self.cfg.N),
                input_buffer=input_encoded,
                prefill_weights=prefill_weights_encoded,
                decode_weights=decode_buffers.weights,
                decode_packed_weights=decode_buffers.packed_weights,
                decode_scales=decode_buffers.scales,
                output_buffer=output_encoded,
                prefill_output_buffer=prefill_encoded,
                decode_input_buffer=decode_input_encoded,
                artifacts=self._direct_artifacts(),
                decode_storage=decode_buffers.storage,
                decode_block_size=decode_buffers.block_size,
                decode_quant_axis=decode_buffers.quant_axis,
            )

        try:
            if trace is None:
                bridge_result = invoke_bridge()
            else:
                with trace.span(
                    "direct_gpu_npu_linear",
                    "stage",
                    "direct_gpu_npu_linear",
                    {
                        "prefill_backend": stages["prefill"],
                        "decode_backend": stages["decode"],
                        "transfer_mode": "direct",
                    },
                ):
                    bridge_result = invoke_bridge()
        except RuntimeError as exc:
            raise DirectTransferUnsupported(str(exc)) from exc

        probe_report = (
            getattr(bridge_result, "probe_report", None) or self._direct_probe_report
        )
        if probe_report is not None:
            probe_report = probe_report.with_runtime_verification(
                mechanism=bridge_result.mechanism,
                sync_events=bridge_result.sync_events,
                diagnostic=bridge_result.diagnostic,
            )
        mechanism_report = (
            None
            if probe_report is None
            else probe_report.mechanism_report(bridge_result.mechanism)
        )
        direct_class = (
            getattr(bridge_result, "direct_class", None)
            or (None if mechanism_report is None else mechanism_report.direct_class)
            or "device_resident_zero_host_copy"
        )
        zero_host_copy = bool(
            getattr(
                bridge_result,
                "zero_host_copy",
                True if mechanism_report is None else mechanism_report.zero_host_copy,
            )
        )
        device_resident_buffers = bool(
            getattr(
                bridge_result,
                "device_resident_buffers",
                (
                    True
                    if mechanism_report is None
                    else mechanism_report.device_resident_buffers
                ),
            )
        )

        output_cpu = decode_npu_array(output_encoded, self.cfg.dtype)
        timing_ms = {
            "prefill": float(bridge_result.prefill_ms),
            "decode": float(bridge_result.decode_ms),
            "end_to_end": (time.perf_counter_ns() - e2e_start_ns) / 1_000_000.0,
        }

        if prefill_encoded is not None:
            prefill_host = decode_npu_array(prefill_encoded, self.cfg.dtype)
            assert decode_input_encoded is not None
            decode_input = decode_npu_array(decode_input_encoded, self.cfg.dtype)
        else:
            prefill_host = np.empty((self.cfg.M, self.cfg.H), dtype=inputs.dtype)
            decode_input = np.empty((self.cfg.H,), dtype=inputs.dtype)

        self.transfer.record_direct_handoff(
            producer=stages["prefill"],
            consumer=stages["decode"],
            tensor=DeviceResidentTensor(
                owner="hip_vmem",
                backend=stages["prefill"],
                dtype=str(input_encoded.dtype),
                shape=(self.cfg.H,),
                strides=(1,),
                byte_size=int(bridge_result.direct_bytes),
                offset=int(bridge_result.subview_offset_bytes),
                imported_view="xrt_bo",
                exported_handle_type="posix_fd",
                sync_state="producer_complete_consumer_waited",
                trace_id="prefill_to_decode",
                mechanism=bridge_result.mechanism,
                direct_class=direct_class,
                zero_host_copy=zero_host_copy,
                device_resident_buffers=device_resident_buffers,
            ),
            elapsed_us=float(bridge_result.handoff_us),
            label="prefill_to_decode",
            mechanism=bridge_result.mechanism,
            sync_events=bridge_result.sync_events,
            numpy_host_materializations=0,
            direct_class=direct_class,
            probe_report=None if probe_report is None else probe_report.to_dict(),
            zero_host_copy=zero_host_copy,
            device_resident_buffers=device_resident_buffers,
        )

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
        execution_overhead = self._execution_overhead_report(
            timing_ms=timing_ms,
            transfer_summary=transfer_summary,
        )
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
            "execution_overhead": execution_overhead,
            "quantized_decode": {
                **self._quantized_decode_report(
                    hardware_fused=self._decode_hardware_fused(),
                    detail=None,
                )
            },
            "device_events": summarize_device_events(trace),
            "direct_bridge": {
                "library_path": (
                    None if bridge.library_path is None else str(bridge.library_path)
                ),
                "mechanism": bridge_result.mechanism,
                "direct_class": direct_class,
                "zero_host_copy": zero_host_copy,
                "device_resident_buffers": device_resident_buffers,
                "bo_flag": bridge_result.bo_flag,
                "import_method": bridge_result.import_method,
                "subview_offset_bytes": bridge_result.subview_offset_bytes,
                "diagnostic": bridge_result.diagnostic,
                "probe_report": (
                    None if probe_report is None else probe_report.to_dict()
                ),
            },
            "npu_development": self._npu_development_report(
                inputs, reference, executed=self._npu_stage_executed()
            ),
            "limitations": linear_limitations(self.manifest, transfer_summary),
        }

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
            return self._run_direct(
                inputs,
                validate=validate,
                capture_details=capture_details,
                trace=trace,
                e2e_start_ns=e2e_start_ns,
            )

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

        decode_weight_arrays = self._decode_weight_arrays_for_backend(stages["decode"])
        wd_arg = self.transfer.transfer(
            "cpu",
            stages["decode"],
            decode_weight_arrays[0],
            trace,
            (
                "decode_packed_weights_to_backend"
                if self._decode_hardware_fused()
                else "decode_weights_to_backend"
            ),
        )
        decode_weight_args = [wd_arg]
        if self._decode_hardware_fused():
            scales_arg = self.transfer.transfer(
                "cpu",
                stages["decode"],
                decode_weight_arrays[1],
                trace,
                "decode_scales_to_backend",
            )
            decode_weight_args.append(scales_arg)
        output, decode_ms = self._stage_run(
            self.executors.decode,
            trace,
            "decode_gemv",
            stages["decode"],
            decode_arg,
            *decode_weight_args,
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
        execution_overhead = self._execution_overhead_report(
            timing_ms=timing_ms,
            transfer_summary=transfer_summary,
        )
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
            "execution_overhead": execution_overhead,
            "quantized_decode": {
                **self._quantized_decode_report(
                    hardware_fused=self._decode_hardware_fused(),
                    detail=quantized_decode_detail,
                )
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
            "The harness measures a two-stage linear pipeline, not full attention, KV-cache, tokenizer, or sampling.",
            "Host transfer mode mixed GPU/NPU paths are staged through NumPy arrays.",
            "transfer_mode=direct refuses host fallback unless a native DeviceResidentTensor bridge records an audited direct edge.",
            "Generated AIR sources are bring-up kernels and are not tuned LLM-scale tilings.",
        ],
        "placement": dict(manifest["runtime"]["stage_backends"]),
    }
