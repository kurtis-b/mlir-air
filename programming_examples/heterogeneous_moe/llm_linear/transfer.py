# SPDX-License-Identifier: MIT

from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from trace import TraceRecorder

DIRECT_CONTRACT = "no_host_copies"
DIRECT_CLASS_DEVICE_RESIDENT = "device_resident_zero_host_copy"
DIRECT_CLASS_SHARED_HOST = "shared_host_zero_copy"
DIRECT_CLASS_HOST_STAGED = "host_staged_copy"


class DirectTransferUnsupported(RuntimeError):
    pass


@dataclass(frozen=True)
class DeviceResidentTensor:
    owner: str
    backend: str
    dtype: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    byte_size: int
    offset: int = 0
    imported_view: str | None = None
    exported_handle_type: str | None = None
    exported_handle: int | None = None
    sync_state: str = "producer_complete"
    trace_id: str | None = None
    mechanism: str | None = None
    direct_class: str | None = None
    zero_host_copy: bool = True
    device_resident_buffers: bool = True

    def descriptor(self) -> dict[str, Any]:
        return {
            "owner": self.owner,
            "backend": self.backend,
            "dtype": self.dtype,
            "shape": [int(dim) for dim in self.shape],
            "strides": [int(dim) for dim in self.strides],
            "byte_size": int(self.byte_size),
            "offset": int(self.offset),
            "imported_view": self.imported_view,
            "exported_handle_type": self.exported_handle_type,
            "sync_state": self.sync_state,
            "trace_id": self.trace_id,
            "mechanism": self.mechanism,
            "direct_class": self.direct_class,
            "zero_host_copy": bool(self.zero_host_copy),
            "device_resident_buffers": bool(self.device_resident_buffers),
        }


class LinearTransferManager:
    def __init__(self, mode: str) -> None:
        if mode not in {"host", "direct"}:
            raise ValueError(f"Unsupported transfer mode: {mode}")
        self.mode = mode
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def reset_events(self) -> None:
        with self._lock:
            self._events = []

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [copy.deepcopy(event) for event in self._events]

    def summary(self) -> dict[str, Any]:
        events = self.snapshot()
        by_edge: dict[str, dict[str, Any]] = {}
        by_mode: dict[str, dict[str, Any]] = {}
        for event in events:
            edge_key = f"{event['producer']}->{event['consumer']}"
            edge = by_edge.setdefault(
                edge_key,
                {
                    "count": 0,
                    "bytes": 0,
                    "elapsed_us": 0.0,
                    "copied_count": 0,
                    "mechanisms": {},
                },
            )
            mode = by_mode.setdefault(
                event["actual_mode"],
                {"count": 0, "bytes": 0, "elapsed_us": 0.0, "copied_count": 0},
            )
            for bucket in (edge, mode):
                bucket["count"] += 1
                bucket["bytes"] += int(event["bytes"])
                bucket["elapsed_us"] += float(event["elapsed_us"])
                if event["copied"]:
                    bucket["copied_count"] += 1
            edge["mechanisms"][event["mechanism"]] = (
                edge["mechanisms"].get(event["mechanism"], 0) + 1
            )
        host_staged_count = int(
            sum(1 for event in events if event["actual_mode"] == "host_staged")
        )
        direct_events = [
            event
            for event in events
            if event["actual_mode"] == "device_resident_direct_handoff"
        ]
        direct_host_materializations = int(
            sum(
                int(event.get("numpy_host_materializations", 0))
                for event in direct_events
            )
        )
        direct_supported = (
            bool(direct_events)
            and direct_host_materializations == 0
            and all(bool(event.get("zero_host_copy")) for event in direct_events)
        )
        direct_device_resident = direct_supported and all(
            bool(event.get("device_resident_buffers")) for event in direct_events
        )
        direct_mechanisms = sorted(
            {
                str(event.get("mechanism"))
                for event in direct_events
                if event.get("mechanism")
            }
        )
        direct_classes = sorted(
            {
                str(event.get("direct_class"))
                for event in direct_events
                if event.get("direct_class")
            }
        )
        direct_probe_reports = [
            event["probe_report"]
            for event in direct_events
            if isinstance(event.get("probe_report"), dict)
        ]
        direct_model = (
            "device_resident_direct_handoff"
            if direct_device_resident
            else (
                "zero_host_copy_direct_handoff"
                if direct_supported
                else "numpy_host_staged_linear_transfer"
            )
        )
        return {
            "event_count": len(events),
            "total_bytes": int(sum(int(event["bytes"]) for event in events)),
            "total_elapsed_us": float(
                sum(float(event["elapsed_us"]) for event in events)
            ),
            "copied_count": int(sum(1 for event in events if event["copied"])),
            "host_staged_count": host_staged_count,
            "numpy_host_materializations": host_staged_count,
            "direct_handoff_numpy_host_materializations": direct_host_materializations,
            "model": direct_model,
            "transfer_semantics": direct_model if direct_supported else "host_staged",
            "device_resident_buffers": bool(direct_device_resident),
            "direct_handoff": {
                "requested": self.mode == "direct",
                "supported": direct_supported,
                "contract": DIRECT_CONTRACT,
                "mechanism": (
                    direct_mechanisms[0]
                    if len(direct_mechanisms) == 1
                    else direct_mechanisms if direct_mechanisms else None
                ),
                "direct_class": (
                    direct_classes[0]
                    if len(direct_classes) == 1
                    else direct_classes if direct_classes else None
                ),
                "probe_report": (
                    direct_probe_reports[0]
                    if len(direct_probe_reports) == 1
                    else direct_probe_reports if direct_probe_reports else None
                ),
                "zero_host_copy": direct_supported,
                "device_resident_buffers": bool(direct_device_resident),
                "edges": [
                    {
                        "producer": event["producer"],
                        "consumer": event["consumer"],
                        "mechanism": event["mechanism"],
                        "direct_class": event.get("direct_class"),
                        "zero_host_copy": bool(event.get("zero_host_copy")),
                        "device_resident_buffers": bool(
                            event.get("device_resident_buffers")
                        ),
                        "sync_events": event.get("sync_events", []),
                        "numpy_host_materializations": int(
                            event.get("numpy_host_materializations", 0)
                        ),
                    }
                    for event in direct_events
                ],
                "diagnostic": (
                    None
                    if direct_supported
                    else "No audited zero-host-copy GPU/NPU handoff event was recorded"
                ),
            },
            "by_edge": by_edge,
            "by_mode": by_mode,
        }

    def require_direct_edge(self, producer: str, consumer: str, label: str) -> None:
        if self.mode != "direct":
            return
        if {producer, consumer} != {"gpu", "npu"}:
            raise DirectTransferUnsupported(
                f"transfer_mode=direct only supports GPU/NPU interstage edges; "
                f"{label} requested {producer}->{consumer}"
            )

    def record_direct_handoff(
        self,
        *,
        producer: str,
        consumer: str,
        tensor: DeviceResidentTensor,
        elapsed_us: float,
        label: str,
        mechanism: str,
        sync_events: list[dict[str, Any]],
        numpy_host_materializations: int,
        direct_class: str = DIRECT_CLASS_DEVICE_RESIDENT,
        probe_report: dict[str, Any] | None = None,
        zero_host_copy: bool = True,
        device_resident_buffers: bool = True,
    ) -> None:
        self._record_event(
            {
                "label": label,
                "producer": producer,
                "consumer": consumer,
                "requested_mode": self.mode,
                "actual_mode": "device_resident_direct_handoff",
                "mechanism": mechanism,
                "direct_class": direct_class,
                "dtype": tensor.dtype,
                "shape": [int(dim) for dim in tensor.shape],
                "bytes": int(tensor.byte_size),
                "copied": False,
                "contiguous_before": True,
                "elapsed_us": float(elapsed_us),
                "device_resident": bool(device_resident_buffers),
                "zero_host_copy": bool(zero_host_copy),
                "device_resident_buffers": bool(device_resident_buffers),
                "tensor": tensor.descriptor(),
                "sync_events": sync_events,
                "numpy_host_materializations": int(numpy_host_materializations),
                "probe_report": probe_report,
            }
        )

    def transfer(
        self,
        producer: str,
        consumer: str,
        array: np.ndarray,
        trace: TraceRecorder | None,
        label: str,
    ) -> np.ndarray:
        if self.mode == "direct" and producer != consumer:
            raise DirectTransferUnsupported(
                "transfer_mode=direct refuses host-staged fallback for "
                f"{label}: {producer}->{consumer}"
            )
        actual = "same_backend_alias" if producer == consumer else "host_staged"
        bytes_moved = int(array.nbytes)
        copied = actual == "host_staged" or not array.flags.c_contiguous
        start_ns = time.perf_counter_ns()
        if trace is None:
            result = self._copy_or_alias(array, actual)
        else:
            with trace.span(
                label,
                "transfer",
                "transfer",
                {
                    "producer": producer,
                    "consumer": consumer,
                    "requested_mode": self.mode,
                    "actual_mode": actual,
                    "bytes": bytes_moved,
                    "device_resident_buffers": False,
                },
            ):
                result = self._copy_or_alias(array, actual)
        elapsed_us = (time.perf_counter_ns() - start_ns) / 1000.0
        self._record_event(
            {
                "label": label,
                "producer": producer,
                "consumer": consumer,
                "requested_mode": self.mode,
                "actual_mode": actual,
                "mechanism": (
                    "same_backend_host_array_alias"
                    if actual == "same_backend_alias" and not copied
                    else "numpy_host_copy"
                ),
                "dtype": str(array.dtype),
                "shape": [int(dim) for dim in array.shape],
                "bytes": bytes_moved,
                "copied": bool(copied),
                "contiguous_before": bool(array.flags.c_contiguous),
                "elapsed_us": elapsed_us,
                "device_resident": False,
            }
        )
        return result

    def _copy_or_alias(self, array: np.ndarray, actual: str) -> np.ndarray:
        if actual == "same_backend_alias" and array.flags.c_contiguous:
            return array
        return np.array(array, copy=True, order="C")

    def _record_event(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._events.append(event)
