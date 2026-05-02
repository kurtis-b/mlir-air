# SPDX-License-Identifier: MIT

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from trace import TraceRecorder


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
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def reset_events(self) -> None:
        with self._lock:
            self._events = []

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(event) for event in self._events]

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
                    "actual_modes": {},
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
            edge["actual_modes"][event["actual_mode"]] = (
                edge["actual_modes"].get(event["actual_mode"], 0) + 1
            )

        return {
            "event_count": len(events),
            "total_bytes": int(sum(int(event["bytes"]) for event in events)),
            "total_elapsed_us": float(
                sum(float(event["elapsed_us"]) for event in events)
            ),
            "copied_count": int(sum(1 for event in events if event["copied"])),
            "host_staged_count": int(
                sum(1 for event in events if event["actual_mode"] == "host_staged")
            ),
            "model": "numpy_host_array_transfer_model",
            "device_resident_buffers": False,
            "by_edge": by_edge,
            "by_mode": by_mode,
        }

    def transfer(
        self,
        producer: str,
        consumer: str,
        array: np.ndarray,
        trace: TraceRecorder | None,
        label: str,
    ) -> np.ndarray:
        actual = self._resolve_mode(producer, consumer)
        shape = tuple(int(dim) for dim in array.shape)
        bytes_moved = int(array.nbytes)
        copied = actual != "numpy_host_array_model" or not array.flags.c_contiguous
        start_ns = time.perf_counter_ns()
        if trace is None:
            if actual == "numpy_host_array_model":
                result = (
                    array if array.flags.c_contiguous else np.ascontiguousarray(array)
                )
            else:
                result = np.array(array, copy=True, order="C")
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
                },
            ):
                if actual == "numpy_host_array_model":
                    result = (
                        array
                        if array.flags.c_contiguous
                        else np.ascontiguousarray(array)
                    )
                else:
                    result = np.array(array, copy=True, order="C")
        end_ns = time.perf_counter_ns()
        self._record_event(
            {
                "label": label,
                "producer": producer,
                "consumer": consumer,
                "requested_mode": self.mode,
                "actual_mode": actual,
                "mechanism": self._mechanism(producer, consumer, actual, copied),
                "dtype": str(array.dtype),
                "shape": list(shape),
                "bytes": bytes_moved,
                "copied": bool(copied),
                "contiguous_before": bool(array.flags.c_contiguous),
                "elapsed_us": (end_ns - start_ns) / 1000.0,
            }
        )
        return result

    def _resolve_mode(self, producer: str, consumer: str) -> str:
        if self.mode == "host":
            return "host_staged"
        if self.mode == "peer":
            if (producer, consumer) not in self._PEER_SUPPORTED:
                raise RuntimeError(
                    f"Peer transfer is not supported for edge {producer}->{consumer}"
                )
            return "numpy_host_array_model"
        if self.mode == "auto":
            return (
                "numpy_host_array_model"
                if (producer, consumer) in self._PEER_SUPPORTED
                else "host_staged"
            )
        raise ValueError(f"Unsupported transfer mode: {self.mode}")

    def _mechanism(
        self, producer: str, consumer: str, actual: str, copied: bool
    ) -> str:
        if actual == "host_staged":
            return "numpy_host_copy"
        if producer == consumer and not copied:
            return "same_backend_alias"
        if producer == consumer:
            return "same_backend_contiguous_copy"
        if copied:
            return "numpy_contiguous_copy"
        return "modelled_host_array_alias_not_device_resident"

    def _record_event(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._events.append(event)
