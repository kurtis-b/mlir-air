# SPDX-License-Identifier: MIT

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import run_llm_linear_external_baselines
from llm_linear.external_baselines import (
    CSV_FIELDNAMES,
    apply_comparison_ratios,
    build_rocblas_runner,
    make_context,
    run_cpu_numpy,
    selected_baselines,
    unsupported_row,
    write_outputs,
)
from llm_linear.manifest import load_json


def _tiny_manifest(moe_dir: Path, tmp_path: Path) -> dict:
    manifest = load_json(moe_dir / "llm_linear" / "default_linear_manifest.json")
    manifest["model"].update({"M": 2, "K": 8, "H": 6, "N": 4, "dtype": "f16"})
    manifest["paths"]["artifacts"] = str(tmp_path / "artifacts")
    manifest["paths"]["generated_air_sources"] = str(tmp_path / "air")
    manifest["runtime"]["stage_backends"] = {"prefill": "cpu", "decode": "cpu"}
    manifest["weights"]["decode"] = {"storage": "bf16"}
    manifest["benchmark"] = {"iterations": 1, "warmup": 0}
    return manifest


def test_external_baseline_selection_and_unknown_filters() -> None:
    assert selected_baselines(["gpu"]) == ["rocblas_gpu"]
    assert selected_baselines(["torch_rocm"]) == ["torch_rocm"]
    with pytest.raises(ValueError, match="Unknown baseline"):
        selected_baselines(["missing"])


def test_cpu_numpy_baseline_validates_and_gets_speedup_ratios(
    moe_dir: Path, tmp_path: Path
) -> None:
    manifest = _tiny_manifest(moe_dir, tmp_path)
    ctx = make_context(
        suite="tiny_ci",
        workload="unit",
        manifest=manifest,
        iterations=1,
        warmup=0,
        decode_weight_storage="bf16",
        output_dir=tmp_path,
    )
    rows = run_cpu_numpy(ctx)
    assert rows[0]["baseline_name"] == "cpu_numpy"
    assert rows[0]["validation_status"] == "pass"
    assert rows[0]["fallback_status"] == "native"
    apply_comparison_ratios(
        rows,
        cpu_ms={"pipeline": 10.0, "prefill": 4.0, "decode": 3.0},
        air_gpu_ms={"pipeline": 2.0, "prefill": 1.0, "decode": 1.0},
        air_npu_ms=None,
    )
    assert rows[0]["speedup_vs_cpu"] == pytest.approx(
        10.0 / rows[0]["mean_end_to_end_ms"]
    )
    assert rows[0]["gap_vs_air_gpu"] == pytest.approx(
        rows[0]["mean_end_to_end_ms"] / 2.0
    )


def test_unsupported_or_fallback_rows_do_not_receive_speedup(
    moe_dir: Path, tmp_path: Path
) -> None:
    manifest = _tiny_manifest(moe_dir, tmp_path)
    ctx = make_context(
        suite="tiny_ci",
        workload="unit",
        manifest=manifest,
        iterations=1,
        warmup=0,
        decode_weight_storage="bf16",
        output_dir=tmp_path,
    )
    rows = [
        unsupported_row(
            ctx,
            baseline_name="torch_rocm",
            framework="PyTorch",
            device="gpu",
            reason="CPU fallback",
            fallback_status="fallback",
        )
    ]
    apply_comparison_ratios(
        rows,
        cpu_ms={"pipeline": 10.0},
        air_gpu_ms={"pipeline": 2.0},
        air_npu_ms={"pipeline": 3.0},
    )
    assert rows[0]["speedup_vs_cpu"] is None
    assert rows[0]["gap_vs_air_gpu"] is None
    assert rows[0]["unsupported_reason"] == "CPU fallback"


def test_external_outputs_write_stable_json_csv_and_report(tmp_path: Path) -> None:
    row = {
        "baseline_name": "cpu_numpy",
        "framework": "numpy",
        "device": "cpu",
        "scope": "pipeline",
        "suite": "tiny_ci",
        "workload": "unit",
        "M": 1,
        "K": 2,
        "H": 3,
        "N": 4,
        "dtype": "bf16",
        "decode_weight_storage": "bf16",
        "iterations": 1,
        "warmup": 0,
        "mean_end_to_end_ms": 1.0,
        "mean_prefill_ms": 0.4,
        "mean_decode_ms": 0.5,
        "speedup_vs_cpu": 1.0,
        "gap_vs_air_gpu": None,
        "gap_vs_air_npu": None,
        "validation_status": "pass",
        "output_max_abs_error": 0.0,
        "device_execution_proof": "numpy CPU",
        "fallback_status": "native",
        "unsupported_reason": "",
    }
    summary = {
        "schema_version": "test",
        "decode_weight_storage": "bf16",
        "workloads": [{"suite": "tiny_ci", "name": "unit", "rows": [row]}],
    }
    write_outputs(tmp_path, summary=summary, rows=[row], title="External Test")
    assert (
        json.loads((tmp_path / "summary.json").read_text())["schema_version"] == "test"
    )
    with (tmp_path / "summary.csv").open(newline="", encoding="utf-8") as handle:
        csv_row = next(csv.DictReader(handle))
    assert list(csv_row.keys()) == CSV_FIELDNAMES
    assert csv_row["baseline_name"] == "cpu_numpy"
    assert "External Test" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_rocblas_build_metadata_is_captured_without_hardware(
    monkeypatch, tmp_path: Path
) -> None:
    class Completed:
        returncode = 0
        stdout = "ok"
        stderr = ""

    captured: dict[str, object] = {}

    def fake_run(cmd, check=False, capture_output=True, text=True):
        captured["cmd"] = cmd
        return Completed()

    monkeypatch.setenv("ROCM_PATH", str(tmp_path / "rocm"))
    monkeypatch.setattr("llm_linear.external_baselines.subprocess.run", fake_run)
    executable, metadata = build_rocblas_runner(hipcc="/opt/rocm/bin/hipcc")
    assert executable.name == "llm_linear_rocm_blas_baseline"
    assert "-lrocblas" in captured["cmd"]
    assert "-lhipblas" in captured["cmd"]
    assert metadata["hipcc"] == "/opt/rocm/bin/hipcc"
    assert metadata["source"].endswith("rocm_blas_baseline.cpp")


def test_external_baseline_cli_cpu_smoke(moe_dir: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "external_cpu"
    rc = run_llm_linear_external_baselines.main(
        [
            "--suite",
            "tiny_ci",
            "--workload-filter",
            "tiny_m1",
            "--baseline-filter",
            "cpu_numpy",
            "--iterations",
            "1",
            "--warmup",
            "0",
            "--require-correctness",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert rc == 0
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    rows = summary["workloads"][0]["rows"]
    assert rows[0]["baseline_name"] == "cpu_numpy"
    assert rows[0]["validation_status"] == "pass"
    assert summary["workloads"][0]["air_references"]["cpu"]["status"] == "pass"
