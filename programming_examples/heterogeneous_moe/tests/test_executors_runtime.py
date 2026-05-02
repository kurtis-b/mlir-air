# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import executors
import runtime as runtime_api
import study as study_api
from executors import CpuExecutor, GpuExecutor, NpuExecutor, StageExecutors, _ranked_memref_descriptor
from kernels import KernelConfig
from orchestrator import MoERuntime, load_runtime
from reference import random_inputs, random_weights, run_reference


def test_cpu_executor_kinds_and_stage_container(small_cfg: KernelConfig) -> None:
    inputs = random_inputs(small_cfg, seed=10)
    weights = random_weights(small_cfg, seed=11)
    reference = run_reference(small_cfg, inputs, weights, "top2")

    np.testing.assert_allclose(CpuExecutor("router", small_cfg.dtype).run(inputs, weights.router), reference["logits"])
    np.testing.assert_allclose(
        CpuExecutor("expert", small_cfg.dtype).run(reference["expert0_input"], weights.expert0_w1, weights.expert0_w2),
        reference["expert0_output"],
    )
    np.testing.assert_allclose(
        CpuExecutor("aggregation", small_cfg.dtype).run(reference["packed_expert_outputs"], reference["weights"]),
        reference["output"],
    )
    with pytest.raises(ValueError, match="Unknown CPU executor kind"):
        CpuExecutor("bad", small_cfg.dtype).run(inputs)

    stages = StageExecutors(router=1, expert0=2, expert1=3, aggregation=4)
    assert stages.router == 1
    assert stages.aggregation == 4


def test_ranked_memref_descriptor_for_float_dtypes() -> None:
    f32 = np.arange(6, dtype=np.float32).reshape(2, 3)
    f16 = np.arange(4, dtype=np.float16).reshape(2, 2)
    f32_desc = _ranked_memref_descriptor(f32)
    f16_desc = _ranked_memref_descriptor(f16)
    assert tuple(f32_desc.shape) == (2, 3)
    assert tuple(f32_desc.strides) == (3, 1)
    assert tuple(f16_desc.shape) == (2, 2)
    assert f16_desc.offset == 0


def _filled_invoker(value: float):
    def invoke(*args):
        output = args[-1]
        output[...] = np.asarray(value, dtype=output.dtype)
        return (*args[:-1], output)

    return invoke


def test_npu_executor_simple_and_cache_paths(monkeypatch, tmp_path: Path, small_cfg: KernelConfig) -> None:
    executor = NpuExecutor(
        "router",
        tmp_path / "router.air.mlir",
        {"xclbin": "x", "insts": "i"},
        tmp_path,
        "npu2",
        small_cfg.dtype,
        small_cfg,
    )
    executor._invoker = _filled_invoker(1.0)
    monkeypatch.setattr(executor, "prepare", lambda: None)
    inputs = np.ones((2, 4), dtype=np.float16)
    weights = np.ones((4, 2), dtype=np.float16)
    output = executor.run(inputs, weights)
    assert output.shape == (2, 2)
    assert output.dtype == np.float16
    assert len(executor._encoded_cache) == 1
    executor.run(inputs, weights)
    assert len(executor._encoded_cache) == 1

    aggregation = NpuExecutor("aggregation", tmp_path / "agg.air.mlir", {}, tmp_path, "npu2", small_cfg.dtype, small_cfg)
    aggregation._invoker = _filled_invoker(2.0)
    monkeypatch.setattr(aggregation, "prepare", lambda: None)
    assert aggregation.run(np.ones((2, 8), dtype=np.float16), np.ones((2, 2), dtype=np.float16)).shape == (2, 4)

    bad = NpuExecutor("bad", tmp_path / "bad.air.mlir", {}, tmp_path, "npu2", small_cfg.dtype, small_cfg)
    bad._invoker = _filled_invoker(0.0)
    monkeypatch.setattr(bad, "prepare", lambda: None)
    with pytest.raises(ValueError, match="Unknown NPU executor kind"):
        bad.run(inputs)


def test_npu_executor_parallel_and_tiled_split(monkeypatch, tmp_path: Path, small_cfg: KernelConfig) -> None:
    inputs = np.ones((2, 4), dtype=np.float16)
    w1 = np.ones((4, 8), dtype=np.float16)
    w2 = np.ones((8, 4), dtype=np.float16)

    parallel = NpuExecutor(
        "expert",
        tmp_path / "expert.air.mlir",
        {"mode": "parallel_split"},
        tmp_path,
        "npu2",
        small_cfg.dtype,
        small_cfg,
    )
    parallel._split_invokers = {"hidden": _filled_invoker(1.0), "output": _filled_invoker(2.0)}
    monkeypatch.setattr(parallel, "prepare", lambda: None)
    parallel_out = parallel.run(inputs, w1, w2)
    assert parallel_out.shape == (2, 4)
    assert np.all(parallel_out == np.float16(2.0))

    tiled = NpuExecutor(
        "expert",
        tmp_path / "expert.air.mlir",
        {"mode": "tiled_split", "tiling": {"ffn_tile": 4, "output_tile": 2}},
        tmp_path,
        "npu2",
        small_cfg.dtype,
        small_cfg,
    )
    tiled._split_invokers = {"hidden": _filled_invoker(1.0), "output": _filled_invoker(0.5)}
    monkeypatch.setattr(tiled, "prepare", lambda: None)
    tiled_out = tiled.run(inputs, w1, w2)
    assert tiled_out.shape == (2, 4)
    assert np.all(tiled_out == np.float16(1.0))
    assert tiled._weight_cache


class FakeSharedLibrary:
    calls: list[str] = []

    def __init__(self, library_path: Path, preload_paths: list[str]) -> None:
        self.library_path = library_path
        self.preload_paths = preload_paths

    def invoke(self, name: str, *args) -> None:
        self.calls.append(name)


def test_gpu_executor_simple_split_and_errors(monkeypatch, tmp_path: Path, small_cfg: KernelConfig) -> None:
    FakeSharedLibrary.calls = []
    monkeypatch.setattr(executors, "_SharedLibraryWrapper", FakeSharedLibrary)
    monkeypatch.setattr(executors, "default_gpu_shared_libs", lambda: [])

    inputs = np.ones((2, 4), dtype=np.float16)
    weights = np.ones((4, 2), dtype=np.float16)
    router = GpuExecutor(
        "router",
        tmp_path / "router.air.mlir",
        {"so": str(tmp_path / "router.so"), "entry": "router_math"},
        tmp_path,
        "gfx1150",
        "router_math",
        small_cfg.dtype,
    )
    router_out = router.run(inputs, weights)
    assert router_out.shape == (2, 2)
    assert FakeSharedLibrary.calls[-1] == "router_math"
    router.run(inputs, weights)
    assert len(router._encoded_cache) == 1

    expert = GpuExecutor(
        "expert",
        tmp_path / "expert.air.mlir",
        {"mode": "parallel_split", "hidden": {"artifact": {"so": "hidden.so"}}, "output": {"artifact": {"so": "output.so"}}},
        tmp_path,
        "gfx1150",
        "expert_mlp",
        small_cfg.dtype,
    )
    expert_out = expert.run(inputs, np.ones((4, 8), dtype=np.float16), np.ones((8, 4), dtype=np.float16))
    assert expert_out.shape == (2, 4)
    assert FakeSharedLibrary.calls[-2:] == ["expert_hidden", "expert_output"]

    aggregation = GpuExecutor(
        "aggregation",
        tmp_path / "agg.air.mlir",
        {"so": str(tmp_path / "agg.so")},
        tmp_path,
        "gfx1150",
        "aggregate_outputs",
        small_cfg.dtype,
    )
    assert aggregation.run(np.ones((2, 8), dtype=np.float16), np.ones((2, 2), dtype=np.float16)).shape == (2, 4)

    missing = GpuExecutor("expert", tmp_path / "expert.air.mlir", {}, tmp_path, "gfx1150", "expert_mlp", small_cfg.dtype)
    with pytest.raises(RuntimeError, match="Missing compiled GPU expert artifact"):
        missing.prepare()

    bad = GpuExecutor("bad", tmp_path / "bad.air.mlir", {"so": "bad.so"}, tmp_path, "gfx1150", "bad", small_cfg.dtype)
    with pytest.raises(ValueError, match="Unknown GPU executor kind"):
        bad.run(inputs)


def test_moe_runtime_cpu_single_chunk_and_load(tmp_path: Path, default_manifest: dict) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(__import__("json").dumps(default_manifest), encoding="utf-8")

    with load_runtime(manifest_path) as runtime:
        inputs = random_inputs(runtime.cfg, default_manifest["inputs"]["seed"])
        single = runtime.run(inputs, router_mode="top2", validate=True, capture_details=True)
        assert single["stage_metrics"]["output"]["allclose"] is True
        assert single["workload"]["chunk_count"] == 1
        assert single["npu_development"]["executed"] is False
        assert runtime.prepare() is None

        quick = runtime.run(inputs, router_mode="top1", validate=False, capture_details=False)
        assert quick["output"].shape == inputs.shape

        longer = np.concatenate([inputs, inputs[:1]], axis=0)
        chunked = runtime.run(longer, router_mode="top2", validate=True, capture_details=True)
        assert chunked["workload"]["chunk_count"] == 2
        assert chunked["output"].shape[0] == longer.shape[0]
        assert chunked["stage_metrics"]["output"]["allclose"] is True

        quick_chunked = runtime.run(longer, router_mode="top2", validate=False, capture_details=False)
        assert quick_chunked == {"output": None}
    runtime.close()


def test_moe_runtime_reports_and_invalid_backend(default_manifest: dict) -> None:
    runtime = MoERuntime(default_manifest)
    assert runtime.logical_batch_tokens() == default_manifest["model"]["batch_tokens"]
    assert runtime._npu_sources_report() == {}
    runtime.close()
    runtime.close()

    bad_manifest = {**default_manifest, "runtime": {**default_manifest["runtime"], "stage_backends": dict(default_manifest["runtime"]["stage_backends"])}}
    bad_manifest["runtime"]["stage_backends"]["router"] = "bad"
    with pytest.raises(ValueError, match="Unsupported source backend|Unsupported backend"):
        MoERuntime(bad_manifest)


def test_runtime_and_study_reexports() -> None:
    assert runtime_api.EDGE_STUDY_SCHEMA_VERSION == "edge-study-v1"
    assert "MoERuntime" in runtime_api.__all__
    assert "benchmark_runtime" in study_api.__all__
