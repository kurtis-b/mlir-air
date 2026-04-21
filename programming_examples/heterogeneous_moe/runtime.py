# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kernels import (
    KernelConfig,
    default_air_filenames,
    default_gpu_air_filenames,
    write_default_air_sources,
    write_default_gpu_air_sources,
)
from numerics import decode_npu_array, encode_npu_array, npu_buffer_dtype
from reference import (
    aggregate_packed_outputs,
    expert_mlp_packed,
    optional_torch_validation,
    pack_expert_outputs,
    pack_expert_weights,
    random_inputs,
    random_weights,
    router_logits,
    routed_inputs,
    run_reference,
    topk_weights,
)


def _project_dir() -> Path:
    return Path(__file__).resolve().parent


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _require_tool(name: str) -> str:
    exe = shutil.which(name)
    if not exe:
        raise RuntimeError(f"Required tool '{name}' is not on PATH")
    return exe


def _resolve_air_sources(manifest: dict[str, Any], backend: str) -> dict[str, Path]:
    cfg = KernelConfig(
        batch_tokens=manifest["model"]["batch_tokens"],
        hidden_size=manifest["model"]["hidden_size"],
        ffn_size=manifest["model"]["ffn_size"],
        dtype=manifest["model"]["dtype"],
    )
    if backend == "gpu":
        gpu_dir = manifest["paths"].get("gpu_sources", manifest["paths"].get("gpu_air_sources", "air_gpu"))
        source_dir = _project_dir() / gpu_dir
        names = default_gpu_air_filenames(cfg)
        writer = write_default_gpu_air_sources
    elif backend == "npu":
        source_dir = _project_dir() / manifest["paths"]["air_sources"]
        names = default_air_filenames(cfg)
        writer = write_default_air_sources
    else:
        raise ValueError(f"Unsupported source backend: {backend}")
    paths = {key: source_dir / name for key, name in names.items()}
    if not all(path.exists() for path in paths.values()):
        writer(cfg, source_dir)
    return paths


def _artifact_root(manifest: dict[str, Any]) -> Path:
    return _project_dir() / manifest["paths"]["artifacts"]


def _llvm_lib_dir() -> Path:
    env = os.environ.get("LLVM_INSTALL_DIR")
    if env:
        return Path(env) / "lib"
    mlir_opt = shutil.which("mlir-opt")
    if not mlir_opt:
        raise RuntimeError("LLVM_INSTALL_DIR is unset and mlir-opt is not on PATH")
    return Path(mlir_opt).resolve().parent.parent / "lib"


def _llvm_bin_dir() -> Path:
    env = os.environ.get("LLVM_INSTALL_DIR")
    if env:
        return Path(env) / "bin"
    mlir_translate = shutil.which("mlir-translate")
    if not mlir_translate:
        raise RuntimeError("LLVM_INSTALL_DIR is unset and mlir-translate is not on PATH")
    return Path(mlir_translate).resolve().parent


def _llvm_clang() -> str:
    clang = _llvm_bin_dir() / "clang"
    if clang.exists():
        return str(clang)
    raise RuntimeError(
        "Could not find LLVM clang next to mlir-translate. "
        "Set LLVM_INSTALL_DIR to the local LLVM build used for GPU compilation."
    )


def _llvm_mlir_opt() -> str:
    mlir_opt = _llvm_bin_dir() / "mlir-opt"
    if mlir_opt.exists():
        return str(mlir_opt)
    raise RuntimeError("Could not find mlir-opt in LLVM_INSTALL_DIR/bin")


def _llvm_mlir_translate() -> str:
    mlir_translate = _llvm_bin_dir() / "mlir-translate"
    if mlir_translate.exists():
        return str(mlir_translate)
    raise RuntimeError("Could not find mlir-translate in LLVM_INSTALL_DIR/bin")


def _default_gpu_shared_libs() -> list[str]:
    llvm_lib_dir = _llvm_lib_dir()
    libs = [
        str(llvm_lib_dir / "libmlir_rocm_runtime.so"),
        str(llvm_lib_dir / "libmlir_runner_utils.so"),
        str(llvm_lib_dir / "libmlir_c_runner_utils.so"),
    ]
    rocm_root = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
    hip = rocm_root / "lib" / "libamdhip64.so"
    if hip.exists():
        libs.append(str(hip))
    return libs


def _sanitize_gpu_llvm_ir(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    filtered = [line for line in lines if not line.startswith("@llvm.global_dtors =")]
    if filtered != lines:
        path.write_text("\n".join(filtered) + "\n", encoding="utf-8")


def _compile_npu(source: Path, output_dir: Path, device: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    xclbin = output_dir / f"{source.stem}.{device}.xclbin"
    insts = output_dir / f"{source.stem}.{device}.insts.bin"
    cmd = [
        _require_tool("aircc"),
        "--device",
        device,
        str(source),
        "-o",
        str(xclbin),
        "-i",
        str(insts),
    ]
    subprocess.run(cmd, check=True)
    return {"xclbin": str(xclbin), "insts": str(insts)}


def _compile_gpu(source: Path, output_dir: Path, arch: str, entrypoint: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outlined_mlir = output_dir / f"{source.stem}.{arch}.outlined.mlir"
    output_mlir = output_dir / f"{source.stem}.{arch}.gpu.mlir"
    output_llvm = output_dir / f"{source.stem}.{arch}.ll"
    output_so = output_dir / f"{source.stem}.{arch}.so"
    outline_cmd = [
        _llvm_mlir_opt(),
        str(source),
        "--pass-pipeline=builtin.module(gpu-kernel-outlining)",
        "-o",
        str(outlined_mlir),
    ]
    subprocess.run(outline_cmd, check=True)
    pipeline = (
        f"builtin.module("
        f"rocdl-attach-target{{chip={arch} O=3}},"
        f"gpu.module(convert-gpu-to-rocdl{{chipset={arch} runtime=HIP}},reconcile-unrealized-casts),"
        f"gpu-module-to-binary,"
        f"func.func(gpu-async-region),"
        f"gpu-to-llvm,"
        f"convert-to-llvm,"
        f"reconcile-unrealized-casts)"
    )
    lower_cmd = [
        _llvm_mlir_opt(),
        str(outlined_mlir),
        f"--pass-pipeline={pipeline}",
        "-o",
        str(output_mlir),
    ]
    subprocess.run(lower_cmd, check=True)
    translate_cmd = [
        _llvm_mlir_translate(),
        "--mlir-to-llvmir",
        str(output_mlir),
        "-o",
        str(output_llvm),
    ]
    subprocess.run(translate_cmd, check=True)
    _sanitize_gpu_llvm_ir(output_llvm)

    shared_libs = _default_gpu_shared_libs()
    rpaths = [f"-Wl,-rpath,{Path(lib).parent}" for lib in shared_libs]
    clang_cmd = [
        _llvm_clang(),
        "-shared",
        "-fPIC",
        "-Wno-override-module",
        str(output_llvm),
        "-o",
        str(output_so),
        *shared_libs,
        *rpaths,
    ]
    subprocess.run(clang_cmd, check=True)
    return {"mlir": str(output_mlir), "llvm": str(output_llvm), "so": str(output_so), "entry": entrypoint}


def populate_artifacts(manifest: dict[str, Any], backends: set[str]) -> dict[str, Any]:
    cfg = KernelConfig(
        batch_tokens=manifest["model"]["batch_tokens"],
        hidden_size=manifest["model"]["hidden_size"],
        ffn_size=manifest["model"]["ffn_size"],
        dtype=manifest["model"]["dtype"],
    )
    npu_sources = write_default_air_sources(cfg, _project_dir() / manifest["paths"]["air_sources"])
    gpu_dir = manifest["paths"].get("gpu_sources", manifest["paths"].get("gpu_air_sources", "air_gpu"))
    gpu_sources = write_default_gpu_air_sources(cfg, _project_dir() / gpu_dir)
    artifact_dir = _artifact_root(manifest)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    compiler_cfg = manifest["compiler"]
    gpu_entries = {
        "router": "router_math_host",
        "expert": "expert_mlp_host",
        "aggregation": "aggregate_outputs_host",
    }
    for key, source in npu_sources.items():
        artifact_entry = manifest["artifacts"].setdefault(key, {})
        if "npu" in backends:
            artifact_entry["npu"] = _compile_npu(source, artifact_dir / "npu", compiler_cfg["npu_device"])
        if "gpu" in backends:
            artifact_entry["gpu"] = _compile_gpu(
                gpu_sources[key],
                artifact_dir / "gpu",
                compiler_cfg["gpu_arch"],
                gpu_entries[key],
            )
    return manifest


class TraceRecorder:
    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def record(self, name: str, category: str, start_ns: int, end_ns: int, tid: str, args: dict[str, Any] | None = None) -> None:
        event = {
            "name": name,
            "cat": category,
            "ph": "X",
            "ts": start_ns / 1000.0,
            "dur": (end_ns - start_ns) / 1000.0,
            "pid": 1,
            "tid": tid,
            "args": args or {},
        }
        with self._lock:
            self._events.append(event)

    @contextlib.contextmanager
    def span(self, name: str, category: str, tid: str, args: dict[str, Any] | None = None):
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            end = time.perf_counter_ns()
            self.record(name, category, start, end, tid, args)

    def dump(self, path: Path) -> None:
        payload = {"traceEvents": sorted(self._events, key=lambda event: event["ts"])}
        _save_json(path, payload)


class TransferManager:
    _PEER_SUPPORTED = {
        ("cpu", "cpu"),
        ("cpu", "gpu"),
        ("gpu", "cpu"),
        ("cpu", "npu"),
        ("npu", "cpu"),
        ("gpu", "gpu"),
        ("npu", "npu"),
    }

    def __init__(self, mode: str) -> None:
        self.mode = mode

    def transfer(
        self,
        producer: str,
        consumer: str,
        array: np.ndarray,
        trace: TraceRecorder,
        label: str,
    ) -> np.ndarray:
        with trace.span(label, "transfer", "transfer", {"producer": producer, "consumer": consumer, "mode": self.mode}):
            actual = self._resolve_mode(producer, consumer)
            if actual == "peer":
                return array if array.flags.c_contiguous else np.ascontiguousarray(array)
            return np.array(array, copy=True, order="C")

    def _resolve_mode(self, producer: str, consumer: str) -> str:
        if self.mode == "host":
            return "host"
        if self.mode == "peer":
            if (producer, consumer) not in self._PEER_SUPPORTED:
                raise RuntimeError(f"Peer transfer is not supported for edge {producer}->{consumer}")
            return "peer"
        if self.mode == "auto":
            return "peer" if (producer, consumer) in self._PEER_SUPPORTED else "host"
        raise ValueError(f"Unsupported transfer mode: {self.mode}")


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
            return expert_mlp_packed(arrays[0], arrays[1], self.dtype_name)
        if self.kind == "aggregation":
            return aggregate_packed_outputs(arrays[0], arrays[1], self.dtype_name)
        raise ValueError(f"Unknown CPU executor kind: {self.kind}")


class NpuExecutor:
    def __init__(self, kind: str, source: Path, artifact: dict[str, str], device: str, dtype_name: str) -> None:
        self.kind = kind
        self.source = source
        self.artifact = artifact
        self.device = device
        self.dtype_name = dtype_name
        self._backend = None
        self._invoker = None

    def prepare(self) -> None:
        if self._invoker is not None:
            return
        from air.backend.xrt import XRTBackend, XRTCompileArtifact

        if not self.artifact:
            compiled = _compile_npu(self.source, self.source.parent.parent / "artifacts" / "npu", self.device)
        else:
            compiled = self.artifact
        self._backend = XRTBackend(target_device=self.device)
        artifact = XRTCompileArtifact(compiled["xclbin"], "MLIR_AIE", compiled["insts"])
        self._invoker = self._backend.load(artifact)

    def run(self, *arrays: np.ndarray) -> np.ndarray:
        self.prepare()
        if self.kind == "router":
            output_shape = (arrays[0].shape[0], 2)
        elif self.kind == "expert":
            output_shape = arrays[0].shape
        elif self.kind == "aggregation":
            output_shape = (arrays[0].shape[0], arrays[0].shape[1] // 2)
        else:
            raise ValueError(f"Unknown NPU executor kind: {self.kind}")
        output_dtype = npu_buffer_dtype(self.dtype_name)
        output = np.zeros(output_shape, dtype=output_dtype)
        encoded_args = [encode_npu_array(array, self.dtype_name) for array in arrays]
        result = self._invoker(*encoded_args, output)
        return decode_npu_array(np.asarray(result[-1]).reshape(output_shape), self.dtype_name)


class GpuExecutor:
    def __init__(self, kind: str, source: Path, artifact: dict[str, str], arch: str, function_name: str, dtype_name: str) -> None:
        self.kind = kind
        self.source = source
        self.artifact = artifact
        self.arch = arch
        self.function_name = function_name
        self.dtype_name = dtype_name
        self._library: _SharedLibraryWrapper | None = None

    def prepare(self) -> None:
        if self._library is not None:
            return
        compiled = self.artifact
        if not compiled or "so" not in compiled:
            compiled = _compile_gpu(
                self.source,
                self.source.parent.parent / "artifacts" / "gpu",
                self.arch,
                self.function_name,
            )
        else:
            compiled = self.artifact
        self.function_name = compiled.get("entry", self.function_name)
        self._library = _SharedLibraryWrapper(Path(compiled["so"]), _default_gpu_shared_libs())

    def run(self, *arrays: np.ndarray) -> np.ndarray:
        self.prepare()
        if self.kind == "router":
            output_shape = (arrays[0].shape[0], 2)
            encoded_args = [encode_npu_array(array, self.dtype_name) for array in arrays]
        elif self.kind == "expert":
            output_shape = arrays[0].shape
            packed_weights = arrays[1]
            split = packed_weights.shape[1] // 2
            w1 = np.ascontiguousarray(packed_weights[:, :split])
            w2 = np.ascontiguousarray(packed_weights[:, split:].T)
            encoded_args = [
                encode_npu_array(arrays[0], self.dtype_name),
                encode_npu_array(w1, self.dtype_name),
                encode_npu_array(w2, self.dtype_name),
            ]
        elif self.kind == "aggregation":
            output_shape = (arrays[0].shape[0], arrays[0].shape[1] // 2)
            encoded_args = [encode_npu_array(array, self.dtype_name) for array in arrays]
        else:
            raise ValueError(f"Unknown GPU executor kind: {self.kind}")
        output_dtype = npu_buffer_dtype(self.dtype_name)
        output = np.zeros(output_shape, dtype=output_dtype)
        descriptors = [_ranked_memref_descriptor(np.ascontiguousarray(array)) for array in (*encoded_args, output)]
        assert self._library is not None
        self._library.invoke(self.function_name, *descriptors)
        return decode_npu_array(output, self.dtype_name)


@dataclass
class StageExecutors:
    router: Any
    expert0: Any
    expert1: Any
    aggregation: Any


class MoERuntime:
    def __init__(self, manifest: dict[str, Any]) -> None:
        self.manifest = manifest
        self.cfg = KernelConfig(
            batch_tokens=manifest["model"]["batch_tokens"],
            hidden_size=manifest["model"]["hidden_size"],
            ffn_size=manifest["model"]["ffn_size"],
            dtype=manifest["model"]["dtype"],
        )
        self.sources = {
            "npu": _resolve_air_sources(manifest, "npu"),
            "gpu": _resolve_air_sources(manifest, "gpu"),
        }
        self.weights = random_weights(self.cfg, manifest["weights"]["seed"])
        self.packed_expert_weights = {
            "expert0": pack_expert_weights(self.weights.expert0_w1, self.weights.expert0_w2, self.cfg.dtype),
            "expert1": pack_expert_weights(self.weights.expert1_w1, self.weights.expert1_w2, self.cfg.dtype),
        }
        self.transfer = TransferManager(manifest["runtime"]["transfer_mode"])
        self.executors = self._make_executors()

    def _make_executor(self, kind: str, backend: str) -> Any:
        if backend == "cpu":
            return CpuExecutor(kind, self.cfg.dtype)

        artifact = self.manifest["artifacts"].get(kind, {}).get(backend, {})
        if backend == "npu":
            return NpuExecutor(kind, self.sources["npu"][kind], artifact, self.manifest["compiler"]["npu_device"], self.cfg.dtype)
        if backend == "gpu":
            function_name = {
                "router": "router_math_host",
                "expert": "expert_mlp_host",
                "aggregation": "aggregate_outputs_host",
            }[kind]
            return GpuExecutor(
                kind,
                self.sources["gpu"][kind],
                artifact,
                self.manifest["compiler"]["gpu_arch"],
                function_name,
                self.cfg.dtype,
            )
        raise ValueError(f"Unsupported backend: {backend}")

    def _make_executors(self) -> StageExecutors:
        stages = self.manifest["runtime"]["stage_backends"]
        return StageExecutors(
            router=self._make_executor("router", stages["router"]),
            expert0=self._make_executor("expert", stages["expert0"]),
            expert1=self._make_executor("expert", stages["expert1"]),
            aggregation=self._make_executor("aggregation", stages["aggregation"]),
        )

    def prepare(self) -> None:
        for executor in (
            self.executors.router,
            self.executors.expert0,
            self.executors.expert1,
            self.executors.aggregation,
        ):
            prepare = getattr(executor, "prepare", None)
            if prepare:
                prepare()

    def _run_expert(
        self,
        executor: Any,
        inputs: np.ndarray,
        packed_weights: np.ndarray,
        trace: TraceRecorder,
        name: str,
        backend: str,
    ) -> np.ndarray:
        with trace.span(name, "stage", name, {"backend": backend}):
            return executor.run(inputs, packed_weights)

    def run(self, inputs: np.ndarray, router_mode: str | None = None) -> dict[str, Any]:
        router_mode = router_mode or self.manifest["runtime"]["router_mode"]
        stages = self.manifest["runtime"]["stage_backends"]
        trace = TraceRecorder()

        with trace.span("router_math", "stage", "router", {"backend": stages["router"]}):
            logits = self.executors.router.run(inputs, self.weights.router)
        logits_cpu = self.transfer.transfer(stages["router"], "cpu", logits, trace, "router_to_cpu")

        with trace.span("topk_select", "control", "cpu", {"mode": router_mode}):
            route_weights = topk_weights(logits_cpu, router_mode, self.cfg.dtype)
            expert0_in, expert1_in = routed_inputs(inputs, route_weights, self.cfg.dtype)

        expert0_arg = self.transfer.transfer("cpu", stages["expert0"], expert0_in, trace, "cpu_to_expert0")
        expert1_arg = self.transfer.transfer("cpu", stages["expert1"], expert1_in, trace, "cpu_to_expert1")
        expert0_weights = self.transfer.transfer(
            "cpu",
            stages["expert0"],
            self.packed_expert_weights["expert0"],
            trace,
            "expert0_weights_to_backend",
        )
        expert1_weights = self.transfer.transfer(
            "cpu",
            stages["expert1"],
            self.packed_expert_weights["expert1"],
            trace,
            "expert1_weights_to_backend",
        )

        with ThreadPoolExecutor(max_workers=2) as pool:
            future0 = pool.submit(
                self._run_expert,
                self.executors.expert0,
                expert0_arg,
                expert0_weights,
                trace,
                "expert0",
                stages["expert0"],
            )
            future1 = pool.submit(
                self._run_expert,
                self.executors.expert1,
                expert1_arg,
                expert1_weights,
                trace,
                "expert1",
                stages["expert1"],
            )
            expert0_out = future0.result()
            expert1_out = future1.result()

        aggregation_backend = stages["aggregation"]
        with trace.span("pack_aggregation_inputs", "control", "cpu", {"source0": stages["expert0"], "source1": stages["expert1"]}):
            packed_experts = pack_expert_outputs(expert0_out, expert1_out, self.cfg.dtype)
        agg_experts = self.transfer.transfer("cpu", aggregation_backend, packed_experts, trace, "experts_to_aggregation")
        agg_weights = self.transfer.transfer("cpu", aggregation_backend, route_weights, trace, "weights_to_aggregation")

        with trace.span("aggregation", "stage", "aggregation", {"backend": aggregation_backend}):
            output = self.executors.aggregation.run(agg_experts, agg_weights)
        output_cpu = self.transfer.transfer(aggregation_backend, "cpu", output, trace, "aggregation_to_cpu")

        reference = run_reference(self.cfg, inputs, self.weights, router_mode)
        max_abs_error = float(np.max(np.abs(output_cpu.astype(np.float32) - reference["output"].astype(np.float32))))
        torch_ok, torch_message = optional_torch_validation(inputs, self.weights, router_mode)

        return {
            "inputs": inputs,
            "logits": logits_cpu,
            "weights": route_weights,
            "expert0_output": expert0_out,
            "expert1_output": expert1_out,
            "output": output_cpu,
            "reference": reference["output"],
            "max_abs_error": max_abs_error,
            "torch_validation": {"ran": torch_ok or torch_message != "torch not installed", "ok": torch_ok, "message": torch_message},
            "trace": trace,
        }


def load_runtime(manifest_path: Path) -> MoERuntime:
    manifest = _load_json(manifest_path)
    return MoERuntime(manifest)


def update_manifest_backends(
    manifest: dict[str, Any],
    router_backend: str | None = None,
    expert0_backend: str | None = None,
    expert1_backend: str | None = None,
    aggregation_backend: str | None = None,
    transfer_mode: str | None = None,
    router_mode: str | None = None,
) -> dict[str, Any]:
    stage_backends = manifest["runtime"]["stage_backends"]
    if router_backend:
        stage_backends["router"] = router_backend
    if expert0_backend:
        stage_backends["expert0"] = expert0_backend
    if expert1_backend:
        stage_backends["expert1"] = expert1_backend
    if aggregation_backend:
        stage_backends["aggregation"] = aggregation_backend
    if transfer_mode:
        manifest["runtime"]["transfer_mode"] = transfer_mode
    if router_mode:
        manifest["runtime"]["router_mode"] = router_mode
    return manifest
