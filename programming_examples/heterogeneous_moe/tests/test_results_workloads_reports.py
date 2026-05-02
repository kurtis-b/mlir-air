# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from case_runner import (
    RunCaseOptions,
    apply_case_to_manifest,
    case_stage_backends,
    contains_npu,
    run_case,
)
from kernels import KernelConfig
from reference import random_inputs
from reports import (
    bottleneck,
    build_edge_study_summary,
    case_backend_name,
    compact_case,
    display,
    edge_study_markdown,
    flatten_suite_cases,
    format_case,
    is_mixed_backend,
    is_single_backend,
    matrix_report_markdown,
    suite_summary_markdown,
    summarize_workload,
    write_markdown,
)
from results import (
    benchmark_measurement_block,
    benchmark_runtime,
    build_case_result,
    correctness_failure_message,
    correctness_summary,
    edge_study_limitations,
    generate_inputs,
    latency_stats,
    logical_config,
    result_csv_row,
    stage_metrics,
    validate_runtime,
)
from workloads import (
    CONTEXT_LENGTHS,
    MODEL_PRESETS,
    case_map,
    model_preset_workloads,
    required_backends,
    routed_tokens_for_context,
    routing_stats,
    routing_workloads,
    select_cases,
    shape_workloads,
    suite_workloads,
)


class TinyRuntime:
    def __init__(self) -> None:
        self.cfg = KernelConfig(batch_tokens=2, hidden_size=4, ffn_size=8, dtype="f16")
        self.calls: list[dict[str, object]] = []

    def logical_batch_tokens(self) -> int:
        return 3

    def run(self, inputs, *, router_mode, validate, capture_details):
        self.calls.append(
            {
                "shape": tuple(inputs.shape),
                "router_mode": router_mode,
                "validate": validate,
                "capture_details": capture_details,
            }
        )
        if validate:
            return {
                "stage_metrics": {"output": {"allclose": True}},
                "torch_validation": {"ran": False, "ok": False},
            }
        return {"output": np.asarray(inputs)}


def test_latency_stats_and_runtime_measurement() -> None:
    assert latency_stats([])["mean"] == 0.0
    stats = latency_stats([1.0, 4.0, 2.0, 3.0])
    assert stats["mean"] == pytest.approx(2.5)
    assert stats["p50"] == pytest.approx(2.5)
    assert stats["p95"] == pytest.approx(3.85)
    assert latency_stats([7.0])["stdev"] == 0.0

    runtime = TinyRuntime()
    inputs = np.ones((3, 4), dtype=np.float32)
    timing = benchmark_runtime(
        runtime,
        inputs,
        router_mode="top2",
        iterations=2,
        warmup=1,
        measurement_mode="both",
    )
    assert timing["effective_warmup"] == 1
    assert timing["measurement_runs"]["cold_start"]["iterations"] == 1
    assert timing["measurement_runs"]["warm"]["iterations"] == 2

    cold = benchmark_runtime(
        runtime,
        inputs,
        router_mode="top1",
        iterations=9,
        warmup=9,
        measurement_mode="cold",
    )
    assert cold["effective_warmup"] == 0
    assert set(cold["measurement_runs"]) == {"cold_start"}


def test_generate_inputs_validate_and_stage_metrics(default_manifest: dict) -> None:
    runtime = TinyRuntime()
    inputs, phase, scale = generate_inputs(runtime, default_manifest)
    assert inputs.shape == (3, 4)
    assert scale == default_manifest["inputs"]["scale"]
    assert phase["input_generation_ms"] >= 0.0
    assert logical_config(runtime).batch_tokens == 3

    last_run, validation_ms = validate_runtime(runtime, inputs, router_mode="top2")
    assert last_run["stage_metrics"]["output"]["allclose"] is True
    assert validation_ms >= 0.0

    metrics = stage_metrics(
        {"output": np.asarray([1.0, 2.0])},
        {"output": np.asarray([1.0, 2.1])},
        atol=0.2,
        rtol=0.0,
    )
    assert metrics["output"]["allclose"] is True


def test_build_case_result_csv_and_correctness(
    default_manifest: dict, fake_last_run_factory
) -> None:
    last_run = fake_last_run_factory()
    timing = {
        "latencies_ms": [1.0],
        "latency_ms": latency_stats([1.0]),
        "measurement_runs": {
            "warm": {"iterations": 1, "latency_ms": latency_stats([1.0])}
        },
        "phase_timings_ms": {"timed_total_ms": 1.0},
        "effective_warmup": 0,
    }
    result = build_case_result(
        schema_version="edge-study-v1",
        metadata={"manifest_sha256": "abc"},
        case_name="cpu_top2",
        manifest=default_manifest,
        iterations=1,
        requested_warmup=0,
        measurement_mode="warm",
        timing=timing,
        last_run=last_run,
        validation_ms=0.2,
        phase_timings_ms={"compile_load_setup_ms": 0.3, "input_generation_ms": 0.4},
    )

    assert result["measurement"]["validation_timed"] is False
    assert result["correctness"]["output_allclose"] is True
    assert correctness_failure_message(result) is None
    row = result_csv_row(result)
    assert row["case_name"] == "cpu_top2"
    assert row["transfer_bytes"] == 128

    missing = {"correctness": {"validated": False}}
    assert (
        correctness_failure_message(missing)
        == "correctness validation did not produce stage metrics"
    )
    bad = {
        "correctness": {"validated": True, "output_allclose": False},
        "stage_metrics": {"output": {"max_abs_error": 9.0, "atol": 0.1, "rtol": 0.2}},
    }
    assert "max_abs_error=9.0" in correctness_failure_message(bad)
    assert correctness_summary({})["validated"] is False

    block = benchmark_measurement_block(
        measurement_mode="warm",
        iterations=1,
        warmup=0,
        effective_warmup=0,
        validation_ms=0.1,
        measurement_runs={},
        setup_ms=0.2,
    )
    assert block["compile_load_setup_ms"] == 0.2


def test_edge_limitations_and_workload_selection(
    default_manifest: dict, default_matrix: dict
) -> None:
    manifest = {
        **default_manifest,
        "workload": {
            "weight_storage": "quantized",
            "compute_dtype": "bf16",
            "shared_expert_ffn_size": 16,
        },
    }
    limitations = edge_study_limitations(
        manifest, {"model": "model", "device_resident_buffers": False}
    )
    assert limitations["routing_topk_location"] == "cpu"
    assert any("quantized storage" in item for item in limitations["limitations"])
    assert any("Shared expert" in item for item in limitations["limitations"])

    cases = select_cases(default_matrix, ["cpu_top1", "npu_top2"])
    assert [case["name"] for case in cases] == ["cpu_top1", "npu_top2"]
    assert case_map(default_matrix)["gpu_top1"]["router_backend"] == "gpu"
    assert required_backends(cases, allow_npu=False) == set()
    assert required_backends(cases, allow_npu=True) == {"npu"}

    preset = {"active_experts": 3, "num_experts": 8}
    assert routed_tokens_for_context(preset, 9) == 4
    assert routed_tokens_for_context(preset, 0) == 1


def test_workload_generation_and_routing_stats(
    default_manifest: dict, default_matrix: dict
) -> None:
    shapes = shape_workloads(default_manifest, default_matrix)
    assert shapes[0]["suite"] == "shape_sweep"
    assert shapes[0]["manifest"]["inputs"]["scale"] == 0.5
    assert "shape_sweep" in shapes[0]["manifest"]["paths"]["artifacts"]

    routing = routing_workloads(default_manifest, default_matrix)
    assert {
        workload["manifest"]["workload"]["routing_profile"] for workload in routing
    } == {
        "balanced",
        "expert0_hot",
        "expert1_hot",
        "alternating",
    }

    model_workloads = model_preset_workloads(default_manifest, default_matrix)
    assert len(model_workloads) == len(MODEL_PRESETS) * len(CONTEXT_LENGTHS)
    assert (
        model_workloads[0]["manifest"]["workload"]["context_length"]
        == CONTEXT_LENGTHS[0]
    )
    assert model_workloads[-1]["model_preset"]["name"] == MODEL_PRESETS[-1]["name"]

    combined = suite_workloads(
        ["shape_sweep", "routing_sweep"], default_manifest, default_matrix
    )
    assert len(combined) == len(shapes) + len(routing)

    stats = routing_stats(default_manifest)
    assert set(stats) == {"top1_token_counts", "top2_probability_mass"}
    assert (
        sum(stats["top1_token_counts"].values())
        == default_manifest["model"]["batch_tokens"]
    )


def test_case_runner_cpu_path(
    default_manifest: dict, default_matrix: dict, tmp_path: Path
) -> None:
    stage_case = {
        "name": "manual",
        "stage_backends": {
            "router": "cpu",
            "expert0": "gpu",
            "expert1": "cpu",
            "aggregation": "gpu",
        },
        "transfer_mode": "auto",
        "router_mode": "top2",
    }
    assert case_stage_backends(stage_case)["expert0"] == "gpu"
    assert contains_npu({"router": "cpu", "expert0": "npu"}) is True
    applied = apply_case_to_manifest(default_manifest, stage_case)
    assert applied["runtime"]["stage_backends"]["expert0"] == "gpu"

    result = run_case(
        default_manifest,
        default_matrix["cases"][0],
        RunCaseOptions(
            manifest_path=tmp_path / "manifest.json",
            case_name="cpu_top1",
            iterations=1,
            warmup=0,
            measurement_mode="warm",
            command_line=["pytest"],
        ),
    )
    assert result["case_name"] == "cpu_top1"
    assert result["correctness"]["output_allclose"] is True
    assert result["metadata"]["command_line"] == ["pytest"]


def test_matrix_and_suite_reports(fake_result_factory) -> None:
    cpu = fake_result_factory("cpu")
    gpu = fake_result_factory("gpu")
    gpu["stage_backends"] = {
        "router": "gpu",
        "expert0": "gpu",
        "expert1": "gpu",
        "aggregation": "gpu",
    }
    mixed = fake_result_factory("mixed")
    mixed["stage_backends"] = {
        "router": "cpu",
        "expert0": "gpu",
        "expert1": "cpu",
        "aggregation": "gpu",
    }
    mixed["latency_ms"]["mean"] = 0.5

    summary = {
        "cases": [cpu, gpu, mixed],
        "skipped": [{"case_name": "npu", "reason": "disabled"}],
    }
    text = matrix_report_markdown(summary, "Matrix")
    assert "# Matrix" in text
    assert "mixed" in text
    assert "NPU-tagged cases were skipped" in text
    assert "Skipped Cases" in text
    with pytest.raises(ValueError, match="No benchmark cases"):
        matrix_report_markdown({"cases": []}, "Empty")

    workload = {
        "suite": "shape_sweep",
        "name": "small",
        "model": {"batch_tokens": 2, "hidden_size": 4, "ffn_size": 8},
        "model_preset": {"model_id": "m", "model_class": "class"},
        "routing_profile": "balanced",
        "input_scale": 0.5,
        "cases": [cpu, mixed],
        "skipped": [],
    }
    suite = {"workloads": [workload]}
    suite_text = suite_summary_markdown(suite, "Suite")
    assert "Fastest Per Workload" in suite_text
    assert "mixed" in suite_text
    assert display(None) == "-"
    assert display("x") == "x"
    assert format_case(cpu) == "cpu / cpu / cpu / cpu"


def test_edge_study_report_helpers(tmp_path: Path, fake_result_factory) -> None:
    single = fake_result_factory("single")
    mixed = fake_result_factory("mixed")
    mixed["stage_backends"] = {
        "router": "cpu",
        "expert0": "gpu",
        "expert1": "cpu",
        "aggregation": "gpu",
    }
    mixed["latency_ms"]["mean"] = 0.5
    mixed["transfer_summary"]["total_elapsed_us"] = 600.0
    workload = {
        "suite": "shape_sweep",
        "name": "small",
        "model": {"batch_tokens": 2, "hidden_size": 4, "ffn_size": 8},
        "context_length": None,
        "routed_tokens": 2,
        "kernel_chunk_tokens": 2,
        "cases": [single, mixed],
        "skipped": [],
    }
    suite_summary = {"workloads": [workload]}

    assert is_single_backend(single) is True
    assert is_mixed_backend(mixed) is True
    assert case_backend_name(single) == "cpu"
    assert case_backend_name(mixed) == "mixed"
    assert bottleneck(mixed)["primary"] == "transfer_or_staging"
    assert compact_case(None) is None
    assert compact_case(single)["case_name"] == "single"
    assert flatten_suite_cases(suite_summary)[1]["backend_kind"] == "mixed"

    workload_summary = summarize_workload(workload)
    assert workload_summary["mixed_wins"] is True
    assert workload_summary["mixed_speedup_vs_best_single"] == pytest.approx(2.0)

    edge = build_edge_study_summary(suite_summary, ["cmd"])
    assert edge["mixed_case_count"] == 1
    markdown = edge_study_markdown(edge)
    assert "Heterogeneous MoE Edge-Efficiency Study" in markdown

    path = tmp_path / "report.md"
    write_markdown(markdown.rstrip("\n"), path)
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_summarize_workload_handles_empty_cases() -> None:
    workload = {
        "suite": "empty",
        "name": "none",
        "model": {},
        "cases": [],
        "skipped": [{"case_name": "x", "reason": "filtered"}],
    }
    summary = summarize_workload(workload)
    assert summary["fastest_overall"] is None
    assert summary["mixed_speedup_vs_best_single"] is None
