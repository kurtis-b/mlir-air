# SPDX-License-Identifier: MIT

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DirectBridgeStatus:
    available: bool
    library_path: str | None
    diagnostic: str


@dataclass(frozen=True)
class DirectBridgeArtifacts:
    gpu_prefill_so: str | None = None
    gpu_decode_so: str | None = None
    npu_prefill_xclbin: str | None = None
    npu_prefill_insts: str | None = None
    npu_decode_xclbin: str | None = None
    npu_decode_insts: str | None = None
    npu_kernel_name: str = "MLIR_AIE"


@dataclass(frozen=True)
class DirectBridgeRunResult:
    prefill_ms: float
    decode_ms: float
    handoff_us: float
    direct_bytes: int
    subview_offset_bytes: int
    mechanism: str
    bo_flag: int
    import_method: int
    sync_events: list[dict[str, Any]]
    diagnostic: str


class _NativeRunConfig(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("direction", ctypes.c_uint32),
        ("dtype", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("m", ctypes.c_uint64),
        ("k", ctypes.c_uint64),
        ("h", ctypes.c_uint64),
        ("n", ctypes.c_uint64),
        ("input", ctypes.c_void_p),
        ("prefill_weights", ctypes.c_void_p),
        ("decode_weights", ctypes.c_void_p),
        ("output", ctypes.c_void_p),
        ("prefill_output", ctypes.c_void_p),
        ("decode_input", ctypes.c_void_p),
        ("gpu_prefill_so", ctypes.c_char_p),
        ("gpu_decode_so", ctypes.c_char_p),
        ("npu_prefill_xclbin", ctypes.c_char_p),
        ("npu_prefill_insts", ctypes.c_char_p),
        ("npu_decode_xclbin", ctypes.c_char_p),
        ("npu_decode_insts", ctypes.c_char_p),
        ("npu_kernel_name", ctypes.c_char_p),
    ]


class _NativeRunResult(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("bo_flag", ctypes.c_uint32),
        ("import_method", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("prefill_us", ctypes.c_uint64),
        ("handoff_us", ctypes.c_uint64),
        ("decode_us", ctypes.c_uint64),
        ("direct_bytes", ctypes.c_uint64),
        ("subview_offset_bytes", ctypes.c_uint64),
        ("mechanism", ctypes.c_char * 128),
        ("sync_events", ctypes.c_char * 512),
        ("diagnostic", ctypes.c_char * 512),
    ]


ABI_VERSION = 1
GPU_PREFILL_NPU_DECODE = 0
NPU_PREFILL_GPU_DECODE = 1
DTYPE_BF16 = 0
DTYPE_F16 = 1


def _load_library(
    path: str | None = None,
) -> tuple[Path | None, ctypes.CDLL | None, str | None]:
    raw_path = path or os.environ.get("LLM_LINEAR_DIRECT_BRIDGE_SO")
    if not raw_path:
        return (
            None,
            None,
            "LLM_LINEAR_DIRECT_BRIDGE_SO is unset; direct GPU/NPU handoff "
            "requires the native bridge library",
        )
    candidate = Path(raw_path).expanduser()
    if not candidate.exists():
        return None, None, f"LLM_LINEAR_DIRECT_BRIDGE_SO does not exist: {candidate}"
    try:
        return candidate, ctypes.CDLL(str(candidate)), None
    except OSError as exc:
        return candidate, None, f"failed to load direct bridge: {exc}"


def _last_error(library: ctypes.CDLL) -> str:
    last_error = getattr(library, "llm_linear_direct_bridge_last_error", None)
    if last_error is None:
        return ""
    last_error.argtypes = [ctypes.c_char_p, ctypes.c_uint64]
    last_error.restype = ctypes.c_int
    buffer = ctypes.create_string_buffer(512)
    last_error(buffer, ctypes.sizeof(buffer))
    return buffer.value.decode("utf-8", errors="replace")


def probe_direct_bridge() -> DirectBridgeStatus:
    candidate, library, error = _load_library()
    if error or library is None:
        return DirectBridgeStatus(
            available=False,
            library_path=None if candidate is None else str(candidate),
            diagnostic=str(error),
        )
    probe = getattr(library, "llm_linear_direct_bridge_probe", None)
    if probe is None:
        return DirectBridgeStatus(
            available=False,
            library_path=str(candidate),
            diagnostic=(
                "direct bridge library is missing llm_linear_direct_bridge_probe"
            ),
        )
    probe.restype = ctypes.c_int
    try:
        ok = int(probe()) == 0
    except Exception as exc:  # pragma: no cover - depends on native bridge
        return DirectBridgeStatus(
            available=False,
            library_path=str(candidate),
            diagnostic=f"direct bridge probe raised: {exc}",
        )
    return DirectBridgeStatus(
        available=ok,
        library_path=str(candidate),
        diagnostic=(
            _last_error(library)
            or (
                "direct bridge probe succeeded"
                if ok
                else "direct bridge probe reported unavailable"
            )
        ),
    )


class DirectBridge:
    def __init__(self, library_path: str | None = None) -> None:
        candidate, library, error = _load_library(library_path)
        if error or library is None:
            raise RuntimeError(str(error))
        self.library_path = candidate
        self._library = library
        self._run = getattr(library, "llm_linear_direct_bridge_run", None)
        if self._run is None:
            raise RuntimeError(
                "direct bridge library is missing llm_linear_direct_bridge_run"
            )
        self._run.argtypes = [
            ctypes.POINTER(_NativeRunConfig),
            ctypes.POINTER(_NativeRunResult),
        ]
        self._run.restype = ctypes.c_int

    def run(
        self,
        *,
        direction: str,
        dtype: str,
        shape: tuple[int, int, int, int],
        input_buffer: Any,
        prefill_weights: Any,
        decode_weights: Any,
        output_buffer: Any,
        prefill_output_buffer: Any | None,
        decode_input_buffer: Any | None,
        artifacts: DirectBridgeArtifacts,
    ) -> DirectBridgeRunResult:
        native_direction = {
            "gpu_prefill_npu_decode": GPU_PREFILL_NPU_DECODE,
            "npu_prefill_gpu_decode": NPU_PREFILL_GPU_DECODE,
        }[direction]
        native_dtype = {"bf16": DTYPE_BF16, "f16": DTYPE_F16}[dtype]
        m, k, h, n = [int(dim) for dim in shape]
        result = _NativeRunResult()
        config = _NativeRunConfig(
            abi_version=ABI_VERSION,
            direction=native_direction,
            dtype=native_dtype,
            reserved=0,
            m=m,
            k=k,
            h=h,
            n=n,
            input=_array_ptr(input_buffer),
            prefill_weights=_array_ptr(prefill_weights),
            decode_weights=_array_ptr(decode_weights),
            output=_array_ptr(output_buffer),
            prefill_output=_array_ptr(prefill_output_buffer),
            decode_input=_array_ptr(decode_input_buffer),
            gpu_prefill_so=_cstr(artifacts.gpu_prefill_so),
            gpu_decode_so=_cstr(artifacts.gpu_decode_so),
            npu_prefill_xclbin=_cstr(artifacts.npu_prefill_xclbin),
            npu_prefill_insts=_cstr(artifacts.npu_prefill_insts),
            npu_decode_xclbin=_cstr(artifacts.npu_decode_xclbin),
            npu_decode_insts=_cstr(artifacts.npu_decode_insts),
            npu_kernel_name=_cstr(artifacts.npu_kernel_name),
        )
        ok = int(self._run(ctypes.byref(config), ctypes.byref(result))) == 0
        diagnostic = _decode_char_array(result.diagnostic) or _last_error(self._library)
        if not ok:
            raise RuntimeError(diagnostic or "direct bridge run failed")
        return DirectBridgeRunResult(
            prefill_ms=float(result.prefill_us) / 1000.0,
            decode_ms=float(result.decode_us) / 1000.0,
            handoff_us=float(result.handoff_us),
            direct_bytes=int(result.direct_bytes),
            subview_offset_bytes=int(result.subview_offset_bytes),
            mechanism=_decode_char_array(result.mechanism),
            bo_flag=int(result.bo_flag),
            import_method=int(result.import_method),
            sync_events=_parse_sync_events(_decode_char_array(result.sync_events)),
            diagnostic=diagnostic or "ok",
        )


def _array_ptr(array: Any | None) -> int | None:
    if array is None:
        return None
    return int(array.ctypes.data)


def _cstr(value: str | None) -> bytes | None:
    if value is None:
        return None
    return str(value).encode("utf-8")


def _decode_char_array(value: Any) -> str:
    return bytes(value).split(b"\0", 1)[0].decode("utf-8", errors="replace")


def _parse_sync_events(encoded: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for chunk in encoded.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, _, detail = chunk.partition(":")
        event: dict[str, Any] = {"event": name}
        for item in detail.split(","):
            key, _, value = item.partition("=")
            if key and value:
                event[key] = value
        events.append(event)
    return events
