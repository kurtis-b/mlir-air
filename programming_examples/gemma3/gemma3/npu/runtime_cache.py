#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gemma3-owned NPU kernel cache and timed BO runner.

This mirrors the Llama 3.2 1B cache boundary without importing that example.
It records cached XRT artifacts, keeps loaded contexts alive, reuses per-layer
BO sets through explicit ``bo_key`` values, skips static/intermediate writes
after first use, and tracks launch timing/readback accounting for runtime
evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import time
from typing import Any, Iterable

import numpy as np
from ml_dtypes import bfloat16


MANIFEST_FILE = "gemma3_npu_kernel_manifest.json"


@dataclass(frozen=True)
class Gemma3CachedKernelArtifact:
    name: str
    output_binary: str
    kernel: str
    insts: str | None

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Gemma3KernelLaunchTiming:
    name: str
    bo_key: str
    first_call: bool
    write_ms: float
    kernel_ms: float
    read_ms: float
    elapsed_ms: float
    buffers_written: int
    bytes_written: int
    buffers_read: int

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Gemma3KernelCacheStats:
    cache_dir: str
    artifact_count: int
    loaded_context_count: int
    bo_set_count: int
    launch_count: int
    kernel_ms: float
    write_ms: float
    read_ms: float
    bytes_written: int

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


def default_cache_dir(model_variant: str = "gemma3-1b") -> Path:
    example_root = Path(__file__).resolve().parents[2]
    return example_root / "build_peano" / "runtime_cache" / model_variant


class Gemma3KernelCache:
    """Cached Gemma3 NPU artifacts, XRT contexts, and persistent BO sets."""

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        *,
        model_variant: str = "gemma3-1b",
        verbose: bool = False,
        lock_path: Path | str = "/tmp/gemma3-npu.lock",
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir(model_variant)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_variant = model_variant
        self.verbose = verbose
        self.lock_path = str(lock_path)
        self.artifacts: dict[str, Any] = {}
        self._loaded: dict[str, tuple[Any, Any]] = {}
        self._cached_bos: dict[str, list[Any]] = {}
        self.timings: list[Gemma3KernelLaunchTiming] = []

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"  [Gemma3KernelCache] {message}")

    @property
    def manifest_path(self) -> Path:
        return self.cache_dir / MANIFEST_FILE

    def register_artifact(
        self,
        name: str,
        *,
        output_binary: Path | str,
        kernel: str,
        insts: Path | str | None = None,
    ) -> None:
        """Register an already-built XRT artifact in this cache."""
        from air.backend.xrt import XRTCompileArtifact

        self.artifacts[name] = XRTCompileArtifact(
            str(output_binary),
            kernel,
            None if insts is None else str(insts),
        )

    def save_manifest(self) -> None:
        payload = []
        for name, artifact in sorted(self.artifacts.items()):
            payload.append(
                Gemma3CachedKernelArtifact(
                    name=name,
                    output_binary=str(artifact.output_binary),
                    kernel=str(artifact.kernel),
                    insts=None if artifact.insts is None else str(artifact.insts),
                ).to_json_dict()
            )
        self.manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        self._log(f"saved manifest with {len(payload)} artifacts")

    def load_manifest(self) -> bool:
        """Load cached artifact paths. Returns False when no manifest exists."""
        if not self.manifest_path.exists():
            return False
        from air.backend.xrt import XRTCompileArtifact

        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        artifacts: dict[str, Any] = {}
        for item in data:
            output_binary = Path(str(item["output_binary"]))
            insts = item.get("insts")
            insts_path = None if insts is None else Path(str(insts))
            if not output_binary.exists():
                raise FileNotFoundError(f"cached Gemma3 kernel binary is missing: {output_binary}")
            if insts_path is not None and not insts_path.exists():
                raise FileNotFoundError(f"cached Gemma3 kernel insts are missing: {insts_path}")
            artifacts[str(item["name"])] = XRTCompileArtifact(
                str(output_binary),
                str(item["kernel"]),
                None if insts_path is None else str(insts_path),
            )
        self.artifacts.update(artifacts)
        self._log(f"loaded manifest with {len(artifacts)} artifacts")
        return True

    def compile_and_cache(
        self,
        name: str,
        mlir_module: Any,
        backend_kwargs: dict[str, Any],
        *,
        output_binary_name: str = "air",
    ) -> Any:
        """Compile an MLIR module once and copy its outputs into the cache."""
        from air.backend.xrt import XRTBackend, XRTCompileArtifact

        backend = XRTBackend(**backend_kwargs)
        start = time.perf_counter()
        artifact = backend.compile(mlir_module, output_binary_name=output_binary_name)
        compile_seconds = time.perf_counter() - start

        source_binary = Path(artifact.output_binary)
        cached_binary = self.cache_dir / f"{name}{source_binary.suffix}"
        shutil.copy2(source_binary, cached_binary)
        cached_insts = None
        if artifact.insts and Path(artifact.insts).exists():
            cached_insts = self.cache_dir / f"{name}.insts.bin"
            shutil.copy2(artifact.insts, cached_insts)
        self.artifacts[name] = XRTCompileArtifact(
            str(cached_binary),
            artifact.kernel,
            None if cached_insts is None else str(cached_insts),
        )
        self.save_manifest()
        backend.unload()
        self._log(f"compiled {name} in {compile_seconds:.1f}s")
        return self.artifacts[name]

    def load_and_run(
        self,
        name: str,
        backend_kwargs: dict[str, Any],
        *inputs: np.ndarray,
        output_indices: Iterable[int] | None = None,
        static_input_indices: Iterable[int] | None = None,
        intermediate_indices: Iterable[int] | None = None,
        bo_key: str | None = None,
    ) -> tuple[np.ndarray, ...]:
        """Run a cached XRT artifact with persistent BO reuse."""
        import filelock
        import pyxrt as xrt
        from air.backend.xrt import XRTBackend

        if name not in self.artifacts:
            available = ", ".join(sorted(self.artifacts))
            raise RuntimeError(f"Gemma3 kernel '{name}' is not cached; available: {available}")

        if name not in self._loaded:
            artifact = self.artifacts[name]
            backend = XRTBackend(**backend_kwargs)
            with filelock.FileLock(self.lock_path):
                invoker = backend.load(artifact)
            self._loaded[name] = (backend, invoker)
            self._log(f"loaded {name}")

        backend, _invoker = self._loaded[name]
        artifact = self.artifacts[name]
        is_elf = str(artifact.output_binary).endswith(".elf")
        cache_key = bo_key or name
        static_indices = set(static_input_indices or ())
        intermediate_set = set(intermediate_indices or ())
        readback_set = {len(inputs) - 1} if output_indices is None else set(output_indices)
        sizes_in_bytes = [int(array.size * array.itemsize) for array in inputs]

        first_call = cache_key not in self._cached_bos
        if first_call:
            bos = []
            for index, byte_count in enumerate(sizes_in_bytes):
                if is_elf:
                    bos.append(xrt.ext.bo(backend.device, byte_count))
                else:
                    bos.append(
                        xrt.bo(
                            backend.device,
                            byte_count,
                            xrt.bo.host_only,
                            backend.kernel.group_id(index + 3),
                        )
                    )
            self._cached_bos[cache_key] = bos
            if not is_elf and backend.bo_instr is not None:
                backend.bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            self._log(f"allocated {len(bos)} BOs for {cache_key}")

        bos = self._cached_bos[cache_key]
        start = time.perf_counter()
        with filelock.FileLock(self.lock_path):
            write_start = time.perf_counter()
            buffers_written = 0
            bytes_written = 0
            for index, array in enumerate(inputs):
                if index in static_indices and not first_call:
                    continue
                if index in intermediate_set and not first_call:
                    continue
                payload = array.view(np.int16) if array.dtype == bfloat16 else array
                src = np.frombuffer(payload, dtype=np.uint8)
                dst = np.frombuffer(bos[index].map(), dtype=np.uint8, count=len(src))
                np.copyto(dst, src, casting="no")
                bos[index].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
                buffers_written += 1
                bytes_written += len(src)
            write_ms = (time.perf_counter() - write_start) * 1000.0

            kernel_start = time.perf_counter()
            if is_elf:
                run = xrt.run(backend.kernel)
                for index, bo in enumerate(bos):
                    run.set_arg(index, bo)
                run.start()
                run.wait2()
            else:
                handle = backend.kernel(3, backend.bo_instr, len(backend.instr_v), *bos)
                handle.wait()
            kernel_ms = (time.perf_counter() - kernel_start) * 1000.0

            read_start = time.perf_counter()
            for index in readback_set:
                bos[index].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            results = tuple(
                np.frombuffer(
                    bos[index].map(),
                    dtype=inputs[index].dtype,
                    count=inputs[index].size,
                )
                if index in readback_set
                else np.empty(0, dtype=inputs[index].dtype)
                for index in range(len(inputs))
            )
            read_ms = (time.perf_counter() - read_start) * 1000.0

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.timings.append(
            Gemma3KernelLaunchTiming(
                name=name,
                bo_key=cache_key,
                first_call=first_call,
                write_ms=write_ms,
                kernel_ms=kernel_ms,
                read_ms=read_ms,
                elapsed_ms=elapsed_ms,
                buffers_written=buffers_written,
                bytes_written=bytes_written,
                buffers_read=len(readback_set),
            )
        )
        return results

    def stats(self) -> Gemma3KernelCacheStats:
        return Gemma3KernelCacheStats(
            cache_dir=str(self.cache_dir),
            artifact_count=len(self.artifacts),
            loaded_context_count=len(self._loaded),
            bo_set_count=len(self._cached_bos),
            launch_count=len(self.timings),
            kernel_ms=sum(record.kernel_ms for record in self.timings),
            write_ms=sum(record.write_ms for record in self.timings),
            read_ms=sum(record.read_ms for record in self.timings),
            bytes_written=sum(record.bytes_written for record in self.timings),
        )

    def unload(self) -> None:
        for backend, _invoker in self._loaded.values():
            unload = getattr(backend, "unload", None)
            if unload is not None:
                unload()
        self._loaded.clear()
