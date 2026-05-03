# SPDX-License-Identifier: MIT

from __future__ import annotations

import ctypes
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DIRECT_CONTRACT = "no_host_copies"
DIRECT_CLASS_DEVICE_RESIDENT = "device_resident_zero_host_copy"
DIRECT_CLASS_SHARED_HOST = "shared_host_zero_copy"
DIRECT_CLASS_HOST_STAGED = "host_staged_copy"
DIRECT_CLASS_UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class DirectBridgeMechanismReport:
    mechanism: str
    supported: bool
    direct_eligible: bool
    direct_class: str
    ownership: str | None
    handle_type: str | None
    import_view: str | None
    bidirectional_visibility: bool
    npu_kernel_verification: bool
    sync_events: list[dict[str, Any]]
    host_materialization_count: int
    zero_host_copy: bool
    device_resident_buffers: bool
    diagnostic: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DirectBridgeMechanismReport":
        return cls(
            mechanism=str(payload.get("mechanism") or payload.get("name") or ""),
            supported=bool(payload.get("supported")),
            direct_eligible=bool(payload.get("direct_eligible")),
            direct_class=str(payload.get("direct_class") or DIRECT_CLASS_UNSUPPORTED),
            ownership=_optional_str(payload.get("ownership")),
            handle_type=_optional_str(payload.get("handle_type")),
            import_view=_optional_str(payload.get("import_view")),
            bidirectional_visibility=bool(payload.get("bidirectional_visibility")),
            npu_kernel_verification=bool(payload.get("npu_kernel_verification")),
            sync_events=[
                dict(event)
                for event in payload.get("sync_events", [])
                if isinstance(event, dict)
            ],
            host_materialization_count=int(
                payload.get("host_materialization_count", 0)
            ),
            zero_host_copy=bool(payload.get("zero_host_copy")),
            device_resident_buffers=bool(payload.get("device_resident_buffers")),
            diagnostic=str(payload.get("diagnostic") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mechanism": self.mechanism,
            "supported": self.supported,
            "direct_eligible": self.direct_eligible,
            "direct_class": self.direct_class,
            "ownership": self.ownership,
            "handle_type": self.handle_type,
            "import_view": self.import_view,
            "bidirectional_visibility": self.bidirectional_visibility,
            "npu_kernel_verification": self.npu_kernel_verification,
            "sync_events": [dict(event) for event in self.sync_events],
            "host_materialization_count": int(self.host_materialization_count),
            "zero_host_copy": self.zero_host_copy,
            "device_resident_buffers": self.device_resident_buffers,
            "diagnostic": self.diagnostic,
        }

    def with_runtime_verification(
        self, *, sync_events: list[dict[str, Any]], diagnostic: str
    ) -> "DirectBridgeMechanismReport":
        npu_kernel_verification = any(
            str(event.get("event", "")) == "xrtRunWait" for event in sync_events
        )
        return DirectBridgeMechanismReport(
            mechanism=self.mechanism,
            supported=self.supported,
            direct_eligible=self.direct_eligible,
            direct_class=self.direct_class,
            ownership=self.ownership,
            handle_type=self.handle_type,
            import_view=self.import_view,
            bidirectional_visibility=self.bidirectional_visibility,
            npu_kernel_verification=npu_kernel_verification,
            sync_events=[dict(event) for event in sync_events],
            host_materialization_count=self.host_materialization_count,
            zero_host_copy=self.zero_host_copy,
            device_resident_buffers=self.device_resident_buffers,
            diagnostic=diagnostic,
        )


@dataclass(frozen=True)
class DirectBridgeProbeReport:
    contract: str
    direct_supported: bool
    selected_mechanism: str | None
    mechanisms: list[DirectBridgeMechanismReport]
    diagnostic: str
    library_path: str | None = None
    schema_version: int = 1

    @classmethod
    def from_dict(
        cls, payload: dict[str, Any], *, library_path: str | None = None
    ) -> "DirectBridgeProbeReport":
        mechanisms = [
            DirectBridgeMechanismReport.from_dict(item)
            for item in payload.get("mechanisms", [])
            if isinstance(item, dict)
        ]
        selected = _optional_str(payload.get("selected_mechanism"))
        if selected is None:
            for mechanism in mechanisms:
                if mechanism.supported and mechanism.direct_eligible:
                    selected = mechanism.mechanism
                    break
        direct_supported = bool(payload.get("direct_supported"))
        if not direct_supported and selected is not None:
            selected_report = next(
                (item for item in mechanisms if item.mechanism == selected), None
            )
            direct_supported = bool(
                selected_report
                and selected_report.supported
                and selected_report.direct_eligible
                and selected_report.zero_host_copy
                and selected_report.host_materialization_count == 0
            )
        return cls(
            contract=str(payload.get("contract") or DIRECT_CONTRACT),
            direct_supported=direct_supported,
            selected_mechanism=selected,
            mechanisms=mechanisms,
            diagnostic=str(payload.get("diagnostic") or ""),
            library_path=library_path,
            schema_version=int(payload.get("schema_version", 1)),
        )

    @classmethod
    def unavailable(
        cls, diagnostic: str, *, library_path: str | None = None
    ) -> "DirectBridgeProbeReport":
        return cls(
            contract=DIRECT_CONTRACT,
            direct_supported=False,
            selected_mechanism=None,
            mechanisms=[],
            diagnostic=diagnostic,
            library_path=library_path,
        )

    @classmethod
    def legacy(
        cls, *, ok: bool, diagnostic: str, library_path: str | None = None
    ) -> "DirectBridgeProbeReport":
        mechanism = DirectBridgeMechanismReport(
            mechanism="hip_vmem_export_xrt_bo_import_fd",
            supported=ok,
            direct_eligible=ok,
            direct_class=DIRECT_CLASS_DEVICE_RESIDENT,
            ownership="hip_vmem",
            handle_type="posix_fd",
            import_view="xrt_bo",
            bidirectional_visibility=ok,
            npu_kernel_verification=False,
            sync_events=[],
            host_materialization_count=0,
            zero_host_copy=ok,
            device_resident_buffers=ok,
            diagnostic=diagnostic,
        )
        return cls(
            contract=DIRECT_CONTRACT,
            direct_supported=ok,
            selected_mechanism=mechanism.mechanism if ok else None,
            mechanisms=[mechanism],
            diagnostic=diagnostic,
            library_path=library_path,
        )

    def selected_report(self) -> DirectBridgeMechanismReport | None:
        if self.selected_mechanism is None:
            return None
        return self.mechanism_report(self.selected_mechanism)

    def mechanism_report(self, mechanism: str) -> DirectBridgeMechanismReport | None:
        return next(
            (item for item in self.mechanisms if item.mechanism == mechanism), None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "contract": self.contract,
            "direct_supported": self.direct_supported,
            "selected_mechanism": self.selected_mechanism,
            "mechanisms": [item.to_dict() for item in self.mechanisms],
            "diagnostic": self.diagnostic,
            "library_path": self.library_path,
        }

    def with_runtime_verification(
        self,
        *,
        mechanism: str,
        sync_events: list[dict[str, Any]],
        diagnostic: str,
    ) -> "DirectBridgeProbeReport":
        updated: list[DirectBridgeMechanismReport] = []
        for report in self.mechanisms:
            if report.mechanism == mechanism:
                updated.append(
                    report.with_runtime_verification(
                        sync_events=sync_events, diagnostic=diagnostic
                    )
                )
            else:
                updated.append(report)
        return DirectBridgeProbeReport(
            contract=self.contract,
            direct_supported=self.direct_supported,
            selected_mechanism=self.selected_mechanism or mechanism,
            mechanisms=updated,
            diagnostic=diagnostic or self.diagnostic,
            library_path=self.library_path,
            schema_version=self.schema_version,
        )


@dataclass(frozen=True)
class DirectBridgeStatus:
    available: bool
    library_path: str | None
    diagnostic: str
    probe_report: DirectBridgeProbeReport | None = None


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
    direct_class: str = DIRECT_CLASS_DEVICE_RESIDENT
    zero_host_copy: bool = True
    device_resident_buffers: bool = True
    probe_report: DirectBridgeProbeReport | None = None


@dataclass(frozen=True)
class GpuStageRunResult:
    stage_ms: float
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


class _NativeGpuStageRunConfig(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("stage", ctypes.c_uint32),
        ("dtype", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("m", ctypes.c_uint64),
        ("k", ctypes.c_uint64),
        ("h", ctypes.c_uint64),
        ("n", ctypes.c_uint64),
        ("input", ctypes.c_void_p),
        ("weights", ctypes.c_void_p),
        ("output", ctypes.c_void_p),
        ("gpu_so", ctypes.c_char_p),
    ]


class _NativeGpuStageRunResult(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("reserved1", ctypes.c_uint32),
        ("reserved2", ctypes.c_uint32),
        ("stage_us", ctypes.c_uint64),
        ("diagnostic", ctypes.c_char * 512),
    ]


ABI_VERSION = 1
GPU_PREFILL_NPU_DECODE = 0
NPU_PREFILL_GPU_DECODE = 1
GPU_STAGE_PREFILL = 0
GPU_STAGE_DECODE = 1
DTYPE_BF16 = 0
DTYPE_F16 = 1

_LOADED_LIBRARIES: dict[str, ctypes.CDLL] = {}


@contextmanager
def _native_bridge_load_env():
    saved = {
        name: os.environ.pop(name, None) for name in ("LD_LIBRARY_PATH", "PYTHONPATH")
    }
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


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
    cache_key = str(candidate.resolve())
    cached = _LOADED_LIBRARIES.get(cache_key)
    if cached is not None:
        return candidate, cached, None
    try:
        mode = getattr(os, "RTLD_GLOBAL", 0) | getattr(os, "RTLD_NOW", 0)
        with _native_bridge_load_env():
            library = ctypes.CDLL(str(candidate), mode=mode)
        _LOADED_LIBRARIES[cache_key] = library
        return candidate, library, None
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
        report = DirectBridgeProbeReport.unavailable(
            str(error), library_path=None if candidate is None else str(candidate)
        )
        return DirectBridgeStatus(
            available=False,
            library_path=None if candidate is None else str(candidate),
            diagnostic=str(error),
            probe_report=report,
        )
    report_probe = getattr(library, "llm_linear_direct_bridge_probe_report", None)
    if report_probe is not None:
        report_probe.argtypes = [ctypes.c_char_p, ctypes.c_uint64]
        report_probe.restype = ctypes.c_int
        buffer = ctypes.create_string_buffer(16384)
        try:
            ok = int(report_probe(buffer, ctypes.sizeof(buffer))) == 0
            payload = json.loads(buffer.value.decode("utf-8", errors="replace"))
            report = DirectBridgeProbeReport.from_dict(
                payload, library_path=str(candidate)
            )
            available = ok and report.direct_supported
            return DirectBridgeStatus(
                available=available,
                library_path=str(candidate),
                diagnostic=report.diagnostic
                or (
                    "direct bridge probe report succeeded"
                    if available
                    else "direct bridge probe report did not validate a direct path"
                ),
                probe_report=report,
            )
        except Exception as exc:  # pragma: no cover - depends on native bridge
            report = DirectBridgeProbeReport.unavailable(
                f"direct bridge probe report raised: {exc}",
                library_path=str(candidate),
            )
            return DirectBridgeStatus(
                available=False,
                library_path=str(candidate),
                diagnostic=report.diagnostic,
                probe_report=report,
            )
    probe = getattr(library, "llm_linear_direct_bridge_probe", None)
    if probe is None:
        report = DirectBridgeProbeReport.unavailable(
            "direct bridge library is missing llm_linear_direct_bridge_probe",
            library_path=str(candidate),
        )
        return DirectBridgeStatus(
            available=False,
            library_path=str(candidate),
            diagnostic=report.diagnostic,
            probe_report=report,
        )
    probe.restype = ctypes.c_int
    try:
        ok = int(probe()) == 0
    except Exception as exc:  # pragma: no cover - depends on native bridge
        report = DirectBridgeProbeReport.unavailable(
            f"direct bridge probe raised: {exc}",
            library_path=str(candidate),
        )
        return DirectBridgeStatus(
            available=False,
            library_path=str(candidate),
            diagnostic=report.diagnostic,
            probe_report=report,
        )
    diagnostic = _last_error(library) or (
        "direct bridge probe succeeded"
        if ok
        else "direct bridge probe reported unavailable"
    )
    report = DirectBridgeProbeReport.legacy(
        ok=ok, diagnostic=diagnostic, library_path=str(candidate)
    )
    return DirectBridgeStatus(
        available=ok,
        library_path=str(candidate),
        diagnostic=diagnostic,
        probe_report=report,
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
        self._gpu_stage_run = getattr(
            library, "llm_linear_direct_bridge_run_gpu_stage", None
        )
        if self._gpu_stage_run is not None:
            self._gpu_stage_run.argtypes = [
                ctypes.POINTER(_NativeGpuStageRunConfig),
                ctypes.POINTER(_NativeGpuStageRunResult),
            ]
            self._gpu_stage_run.restype = ctypes.c_int

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
            direct_class=DIRECT_CLASS_DEVICE_RESIDENT,
            zero_host_copy=True,
            device_resident_buffers=True,
        )

    def run_gpu_stage(
        self,
        *,
        stage: str,
        dtype: str,
        shape: tuple[int, int, int, int],
        input_buffer: Any,
        weights: Any,
        output_buffer: Any,
        gpu_so: str,
    ) -> GpuStageRunResult:
        if self._gpu_stage_run is None:
            raise RuntimeError(
                "direct bridge library is missing "
                "llm_linear_direct_bridge_run_gpu_stage"
            )
        native_stage = {
            "prefill": GPU_STAGE_PREFILL,
            "decode": GPU_STAGE_DECODE,
        }[stage]
        native_dtype = {"bf16": DTYPE_BF16, "f16": DTYPE_F16}[dtype]
        m, k, h, n = [int(dim) for dim in shape]
        result = _NativeGpuStageRunResult()
        config = _NativeGpuStageRunConfig(
            abi_version=ABI_VERSION,
            stage=native_stage,
            dtype=native_dtype,
            reserved=0,
            m=m,
            k=k,
            h=h,
            n=n,
            input=_array_ptr(input_buffer),
            weights=_array_ptr(weights),
            output=_array_ptr(output_buffer),
            gpu_so=_cstr(gpu_so),
        )
        ok = int(self._gpu_stage_run(ctypes.byref(config), ctypes.byref(result))) == 0
        diagnostic = _decode_char_array(result.diagnostic) or _last_error(self._library)
        if not ok:
            raise RuntimeError(diagnostic or "native GPU stage run failed")
        return GpuStageRunResult(
            stage_ms=float(result.stage_us) / 1000.0,
            diagnostic=diagnostic or "ok",
        )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


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
