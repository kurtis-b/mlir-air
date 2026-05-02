# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from manifest import (
    artifact_root,
    generated_air_source_root,
    load_json,
    project_dir,
    repo_dir,
    save_json,
    stable_json_hash,
    update_manifest_backends,
)
from trace import TraceRecorder, summarize_device_events, trace_duration_us
from transfer import TransferManager


def test_manifest_paths_hash_and_backend_updates(tmp_path: Path, default_manifest: dict) -> None:
    payload = {"b": 2, "a": {"x": 1}}
    path = tmp_path / "nested" / "payload.json"
    save_json(path, payload)

    assert load_json(path) == payload
    assert path.read_text(encoding="utf-8").endswith("\n")
    assert stable_json_hash({"a": 1, "b": 2}) == stable_json_hash({"b": 2, "a": 1})
    assert project_dir().name == "heterogeneous_moe"
    assert repo_dir().name == "mlir-air"

    manifest = update_manifest_backends(
        default_manifest,
        router_backend="gpu",
        expert0_backend="npu",
        expert1_backend="cpu",
        aggregation_backend="gpu",
        transfer_mode="peer",
        router_mode="top1",
    )
    assert manifest["runtime"]["stage_backends"] == {
        "router": "gpu",
        "expert0": "npu",
        "expert1": "cpu",
        "aggregation": "gpu",
    }
    assert manifest["runtime"]["transfer_mode"] == "peer"
    assert manifest["runtime"]["router_mode"] == "top1"
    assert artifact_root(manifest).name == "artifacts"

    manifest["paths"].pop("generated_air_sources")
    generated = generated_air_source_root(manifest)
    assert generated.name == "air_sources"
    assert manifest["paths"]["generated_air_sources"] == "artifacts/air_sources"


def test_transfer_manager_modes_and_summaries() -> None:
    contiguous = np.arange(6, dtype=np.float32).reshape(2, 3)
    noncontiguous = contiguous[:, ::2]

    host = TransferManager("host")
    host_copy = host.transfer("gpu", "npu", contiguous, None, "gpu_to_npu")
    assert host_copy is not contiguous
    assert host.summary()["host_staged_count"] == 1
    assert host.summary()["copied_count"] == 1

    peer = TransferManager("peer")
    alias = peer.transfer("cpu", "gpu", contiguous, None, "cpu_to_gpu")
    copied = peer.transfer("gpu", "gpu", noncontiguous, None, "gpu_to_gpu")
    assert alias is contiguous
    assert copied.flags.c_contiguous
    assert peer.summary()["by_edge"]["cpu->gpu"]["actual_modes"]["numpy_host_array_model"] == 1
    with pytest.raises(RuntimeError, match="Peer transfer is not supported"):
        peer.transfer("gpu", "npu", contiguous, None, "unsupported")

    auto = TransferManager("auto")
    staged = auto.transfer("gpu", "npu", contiguous, None, "auto_gpu_to_npu")
    assert staged is not contiguous
    assert auto.summary()["by_mode"]["host_staged"]["count"] == 1
    auto.reset_events()
    assert auto.snapshot() == []

    bad = TransferManager("bad")
    with pytest.raises(ValueError, match="Unsupported transfer mode"):
        bad.transfer("cpu", "cpu", contiguous, None, "bad")


def test_transfer_records_trace_spans() -> None:
    trace = TraceRecorder()
    data = np.arange(4, dtype=np.float32)
    manager = TransferManager("host")

    manager.transfer("cpu", "gpu", data, trace, "copy")
    snapshot = trace.snapshot()
    assert snapshot[0]["name"] == "copy"
    assert snapshot[0]["cat"] == "transfer"
    assert snapshot[0]["args"]["requested_mode"] == "host"
    assert manager.snapshot()[0]["mechanism"] == "numpy_host_copy"


def test_trace_summary_extend_dump_and_device_events(tmp_path: Path) -> None:
    trace = TraceRecorder()
    trace.record("expert0", "stage", 1_000, 6_000, "expert0", {"backend": "gpu"})
    trace.record("expert1", "stage", 3_000, 8_000, "expert1", {"backend": "npu"})
    trace.record("topk_select", "control", 8_000, 9_000, "cpu", {"backend": "cpu"})

    summary = trace.summary()
    assert summary["event_count"] == 3
    assert summary["by_name"]["expert0"]["mean_us"] == pytest.approx(5.0)
    assert summary["overlap"]["expert0_expert1_us"] == pytest.approx(3.0)
    assert trace_duration_us(trace.snapshot()[0]) == pytest.approx(5.0)

    other = TraceRecorder()
    other.extend(trace.snapshot(), ts_offset_us=100.0)
    assert min(event["ts"] for event in other.snapshot()) == pytest.approx(100.0)
    other.extend([])

    dumped = tmp_path / "trace.json"
    other.dump(dumped)
    payload = json.loads(dumped.read_text(encoding="utf-8"))
    assert payload["displayTimeUnit"] == "ms"
    assert payload["summary"]["event_count"] == 3

    device = summarize_device_events(trace)
    assert device["by_backend"]["gpu"]["stages"]["expert0"] == pytest.approx(5.0)
    assert device["by_category"]["control"]["total_us"] == pytest.approx(1.0)
    assert device["device_timeline_available"] is False


def test_empty_trace_summary() -> None:
    assert TraceRecorder().summary() == {
        "event_count": 0,
        "total_duration_us": 0.0,
        "span_us": 0.0,
        "by_name": {},
        "overlap": {},
    }
