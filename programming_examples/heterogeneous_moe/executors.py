# SPDX-License-Identifier: MIT

from __future__ import annotations

import ctypes
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from compile import compile_gpu, compile_npu, compile_npu_parallel_expert, default_gpu_shared_libs
from kernels import KernelConfig
from numerics import decode_npu_array, encode_npu_array, npu_buffer_dtype, quantize_array
from reference import aggregate_packed_outputs, expert_mlp, router_logits

_NPU_SPLIT_COMPILE_LOCK = threading.Lock()


class _SharedLibraryWrapper:
    def __init__(self, library_path: Path, preload_paths: list[str]) -> None:
        mode = getattr(os, "RTLD_GLOBAL", 0) | getattr(os, "RTLD_NOW", 0)
        self._preloads = [ctypes.CDLL(path, mode=mode) for path in preload_paths if Path(path).exists()]
        self._library = ctypes.CDLL(str(library_path), mode=mode)

    def invoke(self, name: str, *args: ctypes.Structure) -> None:
        symbol = "_mlir_ciface_" + name
        try:
            func = getattr(self._library, symbol)
        except AttributeError as exc:
            raise RuntimeError(f"GPU shared library could not find function '{symbol}'") from exc
        func.restype = None
        func(*[ctypes.byref(arg) for arg in args])


def _as_ctype(dtype: np.dtype[Any]) -> Any:
    if dtype == np.dtype(np.float16):
        class F16(ctypes.Structure):
            _fields_ = [("value", ctypes.c_int16)]

        return F16
    return np.ctypeslib.as_ctypes_type(dtype)


def _ranked_memref_descriptor(array: np.ndarray) -> ctypes.Structure:
    c_type = _as_ctype(array.dtype)

    class MemRefDescriptor(ctypes.Structure):
        _fields_ = [
            ("allocated", ctypes.c_longlong),
            ("aligned", ctypes.POINTER(c_type)),
            ("offset", ctypes.c_longlong),
            ("shape", ctypes.c_longlong * array.ndim),
            ("strides", ctypes.c_longlong * array.ndim),
        ]

    descriptor = MemRefDescriptor()
    descriptor.allocated = array.ctypes.data
    descriptor.aligned = array.ctypes.data_as(ctypes.POINTER(c_type))
    descriptor.offset = 0
    descriptor.shape = (ctypes.c_longlong * array.ndim)(*array.shape)
    descriptor.strides = (ctypes.c_longlong * array.ndim)(*[stride // array.itemsize for stride in array.strides])
    return descriptor


class CpuExecutor:
    def __init__(self, kind: str, dtype_name: str) -> None:
        self.kind = kind
        self.dtype_name = dtype_name

    def run(self, *arrays: np.ndarray) -> np.ndarray:
        if self.kind == "router":
            return router_logits(arrays[0], arrays[1], self.dtype_name)
        if self.kind == "expert":
            return expert_mlp(arrays[0], arrays[1], arrays[2], self.dtype_name)
        if self.kind == "aggregation":
            return aggregate_packed_outputs(arrays[0], arrays[1], self.dtype_name)
        raise ValueError(f"Unknown CPU executor kind: {self.kind}")


class NpuExecutor:
    def __init__(
        self,
        kind: str,
        source: Path,
        artifact: dict[str, Any],
        artifact_root: Path,
        device: str,
        dtype_name: str,
        cfg: KernelConfig,
    ) -> None:
        self.kind = kind
        self.source = source
        self.artifact = artifact
        self.artifact_root = artifact_root
        self.device = device
        self.dtype_name = dtype_name
        self.cfg = cfg
        self._backend = None
        self._invoker = None
        self._split_backends: list[Any] = []
        self._split_invokers: dict[str, Any] | None = None
        self._weight_cache: dict[tuple[Any, ...], np.ndarray] = {}
        self._encoded_cache: dict[tuple[Any, ...], np.ndarray] = {}

    def prepare(self) -> None:
        if self._invoker is not None or self._split_invokers is not None:
            return
        from air.backend.xrt import XRTBackend, XRTCompileArtifact

        if self.kind == "expert":
            output_dir = self.artifact_root / "npu"
            with _NPU_SPLIT_COMPILE_LOCK:
                split_artifact = self.artifact if self.artifact.get("mode") in {"parallel_split", "tiled_split"} else compile_npu_parallel_expert(
                    self.cfg,
                    self.source.parent,
                    output_dir,
                    self.device,
                )
            self.artifact = split_artifact
            hidden_compiled = split_artifact["hidden"]["artifact"]
            output_compiled = split_artifact["output"]["artifact"]
            hidden_backend = XRTBackend(target_device=self.device)
            output_backend = XRTBackend(target_device=self.device)
            self._split_backends = [hidden_backend, output_backend]
            self._split_invokers = {
                "hidden": hidden_backend.load(
                    XRTCompileArtifact(hidden_compiled["xclbin"], "MLIR_AIE", hidden_compiled["insts"])
                ),
                "output": output_backend.load(
                    XRTCompileArtifact(output_compiled["xclbin"], "MLIR_AIE", output_compiled["insts"])
                ),
            }
            return

        if not self.artifact:
            compiled = compile_npu(self.source, self.artifact_root / "npu", self.device)
        else:
            compiled = self.artifact
        self._backend = XRTBackend(target_device=self.device)
        artifact = XRTCompileArtifact(compiled["xclbin"], "MLIR_AIE", compiled["insts"])
        self._invoker = self._backend.load(artifact)

    def run(self, *arrays: np.ndarray) -> np.ndarray:
        self.prepare()
        output_dtype = npu_buffer_dtype(self.dtype_name)
        if self.kind == "expert" and self._split_invokers is not None:
            split_mode = self.artifact.get("mode")
            if split_mode == "parallel_split":
                encoded_input = encode_npu_array(arrays[0], self.dtype_name)
                hidden_shape = (arrays[0].shape[0], arrays[1].shape[1])
                hidden_output = np.zeros(hidden_shape, dtype=output_dtype)
                hidden_key = ("w1_full", id(arrays[1]), arrays[1].shape)
                encoded_w1 = self._encoded_cache.get(hidden_key)
                if encoded_w1 is None:
                    encoded_w1 = encode_npu_array(arrays[1], self.dtype_name)
                    self._encoded_cache[hidden_key] = encoded_w1
                hidden_result = self._split_invokers["hidden"](encoded_input, encoded_w1, hidden_output)
                hidden_encoded = np.asarray(hidden_result[-1]).reshape(hidden_shape)

                output_shape = (arrays[0].shape[0], arrays[2].shape[1])
                output = np.zeros(output_shape, dtype=output_dtype)
                output_key = ("w2_full", id(arrays[2]), arrays[2].shape)
                encoded_w2 = self._encoded_cache.get(output_key)
                if encoded_w2 is None:
                    encoded_w2 = encode_npu_array(arrays[2], self.dtype_name)
                    self._encoded_cache[output_key] = encoded_w2
                result = self._split_invokers["output"](hidden_encoded, encoded_w2, output)
                return decode_npu_array(np.asarray(result[-1]).reshape(output_shape), self.dtype_name)

            encoded_input = encode_npu_array(arrays[0], self.dtype_name)
            tiling = self.artifact["tiling"]
            ffn_tile = int(tiling["ffn_tile"])
            output_tile = int(tiling["output_tile"])
            batch_tokens = arrays[0].shape[0]
            output_shape = arrays[0].shape
            accumulated = np.zeros(output_shape, dtype=np.float32)
            for ff0 in range(0, arrays[1].shape[1], ffn_tile):
                ff1 = ff0 + ffn_tile
                hidden_shape = (batch_tokens, ffn_tile)
                hidden_output = np.zeros(hidden_shape, dtype=output_dtype)
                hidden_key = ("w1", id(arrays[1]), ff0, ff1)
                encoded_w1 = self._weight_cache.get(hidden_key)
                if encoded_w1 is None:
                    encoded_w1 = encode_npu_array(np.ascontiguousarray(arrays[1][:, ff0:ff1]), self.dtype_name)
                    self._weight_cache[hidden_key] = encoded_w1
                hidden_result = self._split_invokers["hidden"](encoded_input, encoded_w1, hidden_output)
                hidden_encoded = np.asarray(hidden_result[-1]).reshape(hidden_shape)
                for out0 in range(0, arrays[2].shape[1], output_tile):
                    out1 = out0 + output_tile
                    partial_shape = (batch_tokens, output_tile)
                    partial_output = np.zeros(partial_shape, dtype=output_dtype)
                    output_key = ("w2", id(arrays[2]), ff0, ff1, out0, out1)
                    encoded_w2 = self._weight_cache.get(output_key)
                    if encoded_w2 is None:
                        encoded_w2 = encode_npu_array(np.ascontiguousarray(arrays[2][ff0:ff1, out0:out1]), self.dtype_name)
                        self._weight_cache[output_key] = encoded_w2
                    result = self._split_invokers["output"](hidden_encoded, encoded_w2, partial_output)
                    partial = decode_npu_array(np.asarray(result[-1]).reshape(partial_shape), self.dtype_name)
                    accumulated[:, out0:out1] += np.asarray(partial, dtype=np.float32)
            return quantize_array(accumulated, self.dtype_name)

        if self.kind == "router":
            output_shape = (arrays[0].shape[0], 2)
        elif self.kind == "expert":
            output_shape = arrays[0].shape
        elif self.kind == "aggregation":
            output_shape = (arrays[0].shape[0], arrays[0].shape[1] // 2)
        else:
            raise ValueError(f"Unknown NPU executor kind: {self.kind}")
        output = np.zeros(output_shape, dtype=output_dtype)
        encoded_args = []
        for index, array in enumerate(arrays):
            cache_key = None
            if self.kind == "router" and index == 1:
                cache_key = ("router_weights", id(array), array.shape)
            if cache_key is None:
                encoded_args.append(encode_npu_array(array, self.dtype_name))
                continue
            cached = self._encoded_cache.get(cache_key)
            if cached is None:
                cached = encode_npu_array(array, self.dtype_name)
                self._encoded_cache[cache_key] = cached
            encoded_args.append(cached)
        result = self._invoker(*encoded_args, output)
        return decode_npu_array(np.asarray(result[-1]).reshape(output_shape), self.dtype_name)


class GpuExecutor:
    def __init__(
        self,
        kind: str,
        source: Path,
        artifact: dict[str, Any],
        artifact_root: Path,
        arch: str,
        function_name: str,
        dtype_name: str,
    ) -> None:
        self.kind = kind
        self.source = source
        self.artifact = artifact
        self.artifact_root = artifact_root
        self.arch = arch
        self.function_name = function_name
        self.dtype_name = dtype_name
        self._library: _SharedLibraryWrapper | None = None
        self._split_libraries: dict[str, _SharedLibraryWrapper] | None = None
        self._split_entries: dict[str, str] = {}
        self._encoded_cache: dict[tuple[Any, ...], np.ndarray] = {}
        self._descriptor_cache: dict[tuple[Any, ...], ctypes.Structure] = {}

    def prepare(self) -> None:
        if self._library is not None or self._split_libraries is not None:
            return
        compiled = self.artifact
        if self.kind == "expert" and compiled.get("mode") == "parallel_split":
            self._split_libraries = {}
            preload_paths = default_gpu_shared_libs()
            for name in ("hidden", "output"):
                part = compiled[name]["artifact"]
                self._split_entries[name] = part.get(
                    "entry",
                    "expert_hidden" if name == "hidden" else "expert_output",
                )
                self._split_libraries[name] = _SharedLibraryWrapper(Path(part["so"]), preload_paths)
            return
        if not compiled or "so" not in compiled:
            if self.kind == "expert":
                raise RuntimeError("Missing compiled GPU expert artifact; run compile_kernels.py or populate_artifacts first.")
            compiled = compile_gpu(
                self.source,
                self.artifact_root / "gpu",
                self.arch,
                self.function_name,
            )
        else:
            compiled = self.artifact
        self.function_name = compiled.get("entry", self.function_name)
        self._library = _SharedLibraryWrapper(Path(compiled["so"]), default_gpu_shared_libs())

    def run(self, *arrays: np.ndarray) -> np.ndarray:
        self.prepare()
        if self.kind == "expert" and self._split_libraries is not None:
            hidden_shape = (arrays[0].shape[0], arrays[1].shape[1])
            output_shape = (arrays[0].shape[0], arrays[2].shape[1])
            hidden = np.zeros(hidden_shape, dtype=npu_buffer_dtype(self.dtype_name))
            output = np.zeros(output_shape, dtype=npu_buffer_dtype(self.dtype_name))
            input_encoded = encode_npu_array(arrays[0], self.dtype_name)
            w1_key = ("expert_w1_full", id(arrays[1]), arrays[1].shape)
            w2_key = ("expert_w2_full", id(arrays[2]), arrays[2].shape)
            encoded_w1 = self._encoded_cache.get(w1_key)
            desc_w1 = self._descriptor_cache.get(w1_key)
            if encoded_w1 is None or desc_w1 is None:
                encoded_w1 = encode_npu_array(arrays[1], self.dtype_name)
                desc_w1 = _ranked_memref_descriptor(encoded_w1)
                self._encoded_cache[w1_key] = encoded_w1
                self._descriptor_cache[w1_key] = desc_w1
            encoded_w2 = self._encoded_cache.get(w2_key)
            desc_w2 = self._descriptor_cache.get(w2_key)
            if encoded_w2 is None or desc_w2 is None:
                encoded_w2 = encode_npu_array(arrays[2], self.dtype_name)
                desc_w2 = _ranked_memref_descriptor(encoded_w2)
                self._encoded_cache[w2_key] = encoded_w2
                self._descriptor_cache[w2_key] = desc_w2

            hidden_descs = [
                _ranked_memref_descriptor(np.ascontiguousarray(input_encoded)),
                desc_w1,
                _ranked_memref_descriptor(hidden),
            ]
            self._split_libraries["hidden"].invoke(self._split_entries["hidden"], *hidden_descs)

            output_descs = [
                _ranked_memref_descriptor(hidden),
                desc_w2,
                _ranked_memref_descriptor(output),
            ]
            self._split_libraries["output"].invoke(self._split_entries["output"], *output_descs)
            return decode_npu_array(output, self.dtype_name)

        if self.kind == "router":
            output_shape = (arrays[0].shape[0], 2)
            cache_keys = [None, ("router_weights", id(arrays[1]), arrays[1].shape)]
        elif self.kind == "expert":
            output_shape = arrays[0].shape
            cache_keys = [
                None,
                ("expert_w1", id(arrays[1]), arrays[1].shape),
                ("expert_w2", id(arrays[2]), arrays[2].shape),
            ]
        elif self.kind == "aggregation":
            output_shape = (arrays[0].shape[0], arrays[0].shape[1] // 2)
            cache_keys = [None, None]
        else:
            raise ValueError(f"Unknown GPU executor kind: {self.kind}")
        encoded_args = []
        descriptors: list[ctypes.Structure] = []
        for array, cache_key in zip(arrays, cache_keys):
            if cache_key is None:
                encoded = encode_npu_array(array, self.dtype_name)
                encoded_args.append(encoded)
                descriptors.append(_ranked_memref_descriptor(np.ascontiguousarray(encoded)))
                continue
            encoded = self._encoded_cache.get(cache_key)
            descriptor = self._descriptor_cache.get(cache_key)
            if encoded is None or descriptor is None:
                encoded = encode_npu_array(array, self.dtype_name)
                descriptor = _ranked_memref_descriptor(encoded)
                self._encoded_cache[cache_key] = encoded
                self._descriptor_cache[cache_key] = descriptor
            encoded_args.append(encoded)
            descriptors.append(descriptor)
        output_dtype = npu_buffer_dtype(self.dtype_name)
        output = np.zeros(output_shape, dtype=output_dtype)
        descriptors.append(_ranked_memref_descriptor(output))
        assert self._library is not None
        self._library.invoke(self.function_name, *descriptors)
        return decode_npu_array(output, self.dtype_name)


@dataclass
class StageExecutors:
    router: Any
    expert0: Any
    expert1: Any
    aggregation: Any
