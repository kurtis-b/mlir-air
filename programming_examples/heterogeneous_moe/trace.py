# SPDX-License-Identifier: MIT

from __future__ import annotations

import contextlib
import threading
import time
from pathlib import Path
from typing import Any

from manifest import save_json


def trace_duration_us(event: dict[str, Any]) -> float:
    return float(event["dur"])


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
        payload = {
            "displayTimeUnit": "ms",
            "traceEvents": sorted(self._events, key=lambda event: event["ts"]),
            "summary": self.summary(),
        }
        save_json(path, payload)

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(event) for event in self._events]

    def extend(self, events: list[dict[str, Any]], *, ts_offset_us: float = 0.0) -> None:
        if not events:
            return
        base_ts = min(float(event["ts"]) for event in events)
        rebased = []
        for event in events:
            cloned = dict(event)
            cloned["ts"] = ts_offset_us + (float(event["ts"]) - base_ts)
            rebased.append(cloned)
        with self._lock:
            self._events.extend(rebased)

    def summary(self) -> dict[str, Any]:
        events = self.snapshot()
        if not events:
            return {"event_count": 0, "total_duration_us": 0.0, "span_us": 0.0, "by_name": {}, "overlap": {}}

        by_name: dict[str, dict[str, float]] = {}
        for event in events:
            stats = by_name.setdefault(
                event["name"],
                {"count": 0.0, "total_us": 0.0, "max_us": 0.0, "min_us": float("inf")},
            )
            stats["count"] += 1.0
            stats["total_us"] += trace_duration_us(event)
            stats["max_us"] = max(stats["max_us"], trace_duration_us(event))
            stats["min_us"] = min(stats["min_us"], trace_duration_us(event))
        for stats in by_name.values():
            stats["mean_us"] = stats["total_us"] / stats["count"]
            stats["count"] = int(stats["count"])

        start_us = min(float(event["ts"]) for event in events)
        end_us = max(float(event["ts"]) + trace_duration_us(event) for event in events)
        expert0 = [event for event in events if event["name"] == "expert0"]
        expert1 = [event for event in events if event["name"] == "expert1"]
        overlap_us = 0.0
        for lhs in expert0:
            lhs_end = float(lhs["ts"]) + trace_duration_us(lhs)
            for rhs in expert1:
                rhs_end = float(rhs["ts"]) + trace_duration_us(rhs)
                overlap_us += max(0.0, min(lhs_end, rhs_end) - max(float(lhs["ts"]), float(rhs["ts"])))

        return {
            "event_count": len(events),
            "total_duration_us": float(sum(trace_duration_us(event) for event in events)),
            "span_us": float(end_us - start_us),
            "by_name": by_name,
            "overlap": {
                "expert0_expert1_us": float(overlap_us),
                "expert0_count": len(expert0),
                "expert1_count": len(expert1),
            },
        }


def summarize_device_events(trace: TraceRecorder) -> dict[str, Any]:
    events = trace.snapshot()
    by_backend: dict[str, dict[str, Any]] = {}
    by_category: dict[str, dict[str, Any]] = {}
    stage_events: list[dict[str, Any]] = []
    for event in events:
        duration_us = trace_duration_us(event)
        category = str(event.get("cat", "unknown"))
        args = dict(event.get("args", {}))
        backend = str(args.get("backend", event.get("tid", "unknown")))
        category_stats = by_category.setdefault(category, {"count": 0, "total_us": 0.0})
        category_stats["count"] += 1
        category_stats["total_us"] += duration_us
        if category == "stage":
            backend_stats = by_backend.setdefault(backend, {"count": 0, "total_us": 0.0, "stages": {}})
            backend_stats["count"] += 1
            backend_stats["total_us"] += duration_us
            stage_name = str(event.get("name", "unknown"))
            backend_stats["stages"][stage_name] = backend_stats["stages"].get(stage_name, 0.0) + duration_us
            stage_events.append(
                {
                    "stage": stage_name,
                    "backend": backend,
                    "duration_us": duration_us,
                    "source": "host_perf_counter",
                }
            )
    return {
        "source": "host_perf_counter",
        "device_timeline_available": False,
        "stage_events": stage_events,
        "by_backend": by_backend,
        "by_category": by_category,
        "notes": [
            "Stage timings are host perf_counter spans around executor calls.",
            "GPU/NPU device timelines are not merged into this summary yet.",
        ],
    }
