# SPDX-License-Identifier: MIT

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import pytest

MOE_DIR = Path(__file__).resolve().parents[1]
if str(MOE_DIR) not in sys.path:
    sys.path.insert(0, str(MOE_DIR))

from kernels import KernelConfig
from manifest import load_json


@pytest.fixture
def moe_dir() -> Path:
    return MOE_DIR


@pytest.fixture
def default_manifest(moe_dir: Path) -> dict[str, Any]:
    return load_json(moe_dir / "default_manifest.json")


@pytest.fixture
def default_matrix(moe_dir: Path) -> dict[str, Any]:
    return load_json(moe_dir / "default_benchmark_matrix.json")


@pytest.fixture
def small_cfg() -> KernelConfig:
    return KernelConfig(batch_tokens=2, hidden_size=4, ffn_size=8, dtype="f16")


def fake_last_run() -> dict[str, Any]:
    metrics = {
        "output": {
            "max_abs_error": 0.01,
            "mean_abs_error": 0.005,
            "rmse": 0.006,
            "max_abs_expected": 1.0,
            "atol": 0.05,
            "rtol": 0.02,
            "allclose": True,
        }
    }
    transfer_summary = {
        "event_count": 1,
        "total_bytes": 128,
        "total_elapsed_us": 7.5,
        "host_staged_count": 1,
        "model": "numpy_host_array_transfer_model",
        "device_resident_buffers": False,
    }
    return {
        "workload": {
            "shape": {"batch_tokens": 2, "hidden_size": 4, "ffn_size": 8, "dtype": "f16"},
            "routing_profile": "balanced",
        },
        "max_abs_error": 0.01,
        "stage_metrics": metrics,
        "torch_validation": {"ran": True, "ok": True, "message": "ok"},
        "trace_summary": {
            "event_count": 2,
            "total_duration_us": 10.0,
            "span_us": 8.0,
            "by_name": {},
            "overlap": {"expert0_expert1_us": 3.0, "expert0_count": 1, "expert1_count": 1},
        },
        "device_events": {"by_category": {"stage": {"count": 1, "total_us": 4.0}}},
        "transfer_events": [{"label": "x"}],
        "transfer_summary": transfer_summary,
        "npu_development": {"executed": False},
        "limitations": {"limitations": []},
    }


def fake_result(case_name: str = "case") -> dict[str, Any]:
    result = {
        "schema_version": "edge-study-v1",
        "metadata": {"command_line": []},
        "case_name": case_name,
        "router_mode": "top2",
        "stage_backends": {
            "router": "cpu",
            "expert0": "cpu",
            "expert1": "cpu",
            "aggregation": "cpu",
        },
        "transfer_mode": "host",
        "iterations": 1,
        "warmup": 0,
        "requested_warmup": 0,
        "measurement": {"runs": {}},
        "timing_breakdown_ms": {},
        "phase_timings_ms": {},
        "latency_ms": {"mean": 1.0, "min": 1.0, "max": 1.0, "p50": 1.0, "p95": 1.0, "stdev": 0.0},
        "latencies_ms": [1.0],
        "correctness": {
            "validated": True,
            "output_max_abs_error": 0.01,
            "output_atol": 0.05,
            "output_rtol": 0.02,
            "output_allclose": True,
            "torch_ran": True,
            "torch_ok": True,
            "torch_message": "ok",
        },
        "data_movement": {},
        "execution_truth": {"npu_executed": False},
    }
    result.update(fake_last_run())
    result["case_name"] = case_name
    return result


@pytest.fixture
def fake_result_factory():
    return fake_result


@pytest.fixture
def fake_last_run_factory():
    return fake_last_run


class FakeTrace:
    def __init__(self) -> None:
        self.dumped_to: Path | None = None

    def dump(self, path: Path) -> None:
        self.dumped_to = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"traceEvents": []}\n', encoding="utf-8")


@pytest.fixture
def fake_trace() -> FakeTrace:
    return FakeTrace()


def clone(value: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(value)
