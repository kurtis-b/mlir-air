# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path

import pytest

import compile_llm_linear
import run_llm_linear_suite
from llm_linear.manifest import load_json
from llm_linear.reference import random_inputs
from llm_linear.results import (
    benchmark_runtime,
    build_case_result,
    correctness_failure_message,
    result_csv_row,
    validate_runtime,
)
from llm_linear.runtime import LinearRuntime
from llm_linear.transfer import DirectTransferUnsupported


def _tiny_cpu_manifest(moe_dir: Path, tmp_path: Path) -> dict:
    manifest = load_json(moe_dir / "llm_linear" / "default_linear_manifest.json")
    manifest["model"].update({"M": 2, "K": 8, "H": 6, "N": 4, "dtype": "f16"})
    manifest["paths"]["artifacts"] = str(tmp_path / "artifacts")
    manifest["paths"]["generated_air_sources"] = str(tmp_path / "air")
    manifest["runtime"]["stage_backends"] = {"prefill": "cpu", "decode": "cpu"}
    manifest["runtime"]["transfer_mode"] = "host"
    return manifest


def test_linear_runtime_result_schema(moe_dir: Path, tmp_path: Path) -> None:
    manifest = _tiny_cpu_manifest(moe_dir, tmp_path)
    with LinearRuntime(manifest) as runtime:
        inputs = random_inputs(runtime.cfg, seed=manifest["inputs"]["seed"])
        timing = benchmark_runtime(
            runtime, inputs, iterations=2, warmup=1, measurement_mode="both"
        )
        last_run, validation_ms = validate_runtime(runtime, inputs)
    result = build_case_result(
        metadata={"command_line": ["test"]},
        case_name="cpu_only",
        manifest=manifest,
        iterations=2,
        requested_warmup=1,
        measurement_mode="both",
        timing=timing,
        last_run=last_run,
        validation_ms=validation_ms,
        phase_timings_ms={"compile_load_setup_ms": 0.0, "input_generation_ms": 0.0},
        suite="tiny_ci",
        workload_name="unit",
    )
    assert result["measurement"]["validation_timed"] is False
    assert result["measurement"]["compile_load_excluded"] is True
    assert result["stage_latency_ms"]["prefill"]["mean"] >= 0.0
    assert result["transfer_summary"]["device_resident_buffers"] is False
    assert result["correctness"]["validation_status"] == "pass"
    assert correctness_failure_message(result) is None
    row = result_csv_row(result)
    assert row["case_name"] == "cpu_only"
    assert row["validation_status"] == "pass"


def test_linear_direct_transfer_fails_closed(moe_dir: Path, tmp_path: Path) -> None:
    manifest = _tiny_cpu_manifest(moe_dir, tmp_path)
    manifest["runtime"]["transfer_mode"] = "direct"
    with LinearRuntime(manifest) as runtime:
        inputs = random_inputs(runtime.cfg, seed=manifest["inputs"]["seed"])
        with pytest.raises(DirectTransferUnsupported, match="unsupported"):
            runtime.run(inputs)


def test_compile_llm_linear_cli(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_populate(manifest, backends):
        captured["backends"] = backends
        manifest["artifacts"]["prefill"]["gpu"] = {"so": "prefill.so"}
        return manifest

    monkeypatch.setattr(compile_llm_linear, "populate_artifacts", fake_populate)
    out_path = tmp_path / "compiled.json"
    assert (
        compile_llm_linear.main(["--backends", "gpu", "--manifest-out", str(out_path)])
        == 0
    )
    assert captured["backends"] == {"gpu"}
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["artifacts"]["prefill"]["gpu"]["so"] == "prefill.so"


def test_run_llm_linear_suite_cpu_filter_and_direct_failure(tmp_path: Path) -> None:
    out_dir = tmp_path / "suite"
    assert (
        run_llm_linear_suite.main(
            [
                "--suite",
                "tiny_ci",
                "--workload-filter",
                "tiny_m1",
                "--case-filter",
                "cpu_only",
                "--iterations",
                "1",
                "--warmup",
                "0",
                "--require-correctness",
                "--output-dir",
                str(out_dir),
            ]
        )
        == 0
    )
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["workloads"][0]["cases"][0]["case_name"] == "cpu_only"
    assert (out_dir / "summary.csv").read_text(encoding="utf-8").startswith("suite,")
    assert "Direct GPU/NPU handoff is unsupported" in (out_dir / "report.md").read_text(
        encoding="utf-8"
    )

    with pytest.raises(SystemExit, match="direct is unsupported"):
        run_llm_linear_suite.main(["--transfer-mode", "direct"])
