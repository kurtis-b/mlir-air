# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

import bench
import compile_kernels
import edge_study
import report
import run_matrix
import run_workload_suite
import smoke_tests
from case_runner import RunCaseOptions
from manifest import save_json


def test_bench_main_writes_outputs_and_prepare(
    monkeypatch, tmp_path: Path, default_manifest: dict, fake_result_factory, fake_trace
) -> None:
    monkeypatch.setattr(bench, "project_dir", lambda: tmp_path)
    monkeypatch.setattr(bench, "load_json", lambda path: default_manifest)

    captured: dict[str, Any] = {}

    def fake_run(manifest, case, options):
        captured["case"] = case
        captured["options"] = options
        return fake_result_factory(case["name"]), fake_trace

    monkeypatch.setattr(bench, "run_case_with_trace", fake_run)
    assert (
        bench.main(
            [
                "--case-name",
                "case",
                "--router-mode",
                "top1",
                "--router-backend",
                "cpu",
                "--results-out",
                str(tmp_path / "results.json"),
                "--csv-out",
                str(tmp_path / "rows.csv"),
                "--trace-out",
                str(tmp_path / "trace.json"),
                "--trace-summary-out",
                str(tmp_path / "trace_summary.json"),
                "--stage-metrics-out",
                str(tmp_path / "stage_metrics.json"),
                "--transfer-summary-out",
                str(tmp_path / "transfer_summary.json"),
                "--device-events-out",
                str(tmp_path / "device_events.json"),
                "--npu-dev-report-out",
                str(tmp_path / "npu.json"),
                "--require-correctness",
            ]
        )
        == 0
    )
    assert captured["case"]["router_mode"] == "top1"
    assert isinstance(captured["options"], RunCaseOptions)
    assert (
        json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))["case_name"]
        == "case"
    )
    assert (tmp_path / "rows.csv").read_text(encoding="utf-8").startswith("case_name")
    assert (tmp_path / "trace.json").exists()
    assert (tmp_path / "npu.json").exists()

    class FakeRuntime:
        def __init__(self, manifest) -> None:
            self.manifest = manifest

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def prepare(self) -> None:
            captured["prepared"] = True

    monkeypatch.setattr(bench, "MoERuntime", FakeRuntime)
    assert bench.main(["--prepare"]) == 0
    assert captured["prepared"] is True


def test_bench_main_validation_failures(
    monkeypatch, tmp_path: Path, default_manifest: dict, fake_result_factory
) -> None:
    monkeypatch.setattr(bench, "project_dir", lambda: tmp_path)
    monkeypatch.setattr(bench, "load_json", lambda path: default_manifest)

    bad_torch = fake_result_factory("bad")
    bad_torch["torch_validation"] = {"ok": False, "message": "no torch"}
    monkeypatch.setattr(
        bench, "run_case_with_trace", lambda *args, **kwargs: (bad_torch, None)
    )
    with pytest.raises(SystemExit, match="torch validation failed"):
        bench.main(["--require-torch"])

    bad_correctness = fake_result_factory("bad")
    bad_correctness["correctness"]["output_allclose"] = False
    monkeypatch.setattr(
        bench, "run_case_with_trace", lambda *args, **kwargs: (bad_correctness, None)
    )
    with pytest.raises(SystemExit, match="correctness validation failed"):
        bench.main(["--require-correctness"])


def test_compile_kernels_main(
    monkeypatch, tmp_path: Path, default_manifest: dict
) -> None:
    monkeypatch.setattr(compile_kernels, "project_dir", lambda: tmp_path)
    monkeypatch.setattr(compile_kernels, "load_json", lambda path: default_manifest)
    captured: dict[str, Any] = {}

    def fake_populate(manifest, backends):
        captured["backends"] = backends
        manifest["artifacts"]["router"]["gpu"] = {"so": "router.so"}
        return manifest

    monkeypatch.setattr(compile_kernels, "populate_artifacts", fake_populate)
    assert (
        compile_kernels.main(["--backends", "gpu", "--manifest-out", "compiled.json"])
        == 0
    )
    assert captured["backends"] == {"gpu"}
    assert (
        json.loads((tmp_path / "compiled.json").read_text(encoding="utf-8"))[
            "artifacts"
        ]["router"]["gpu"]["so"]
        == "router.so"
    )


def test_run_matrix_main_and_run_case(
    monkeypatch,
    tmp_path: Path,
    default_manifest: dict,
    default_matrix: dict,
    fake_result_factory,
    fake_trace,
) -> None:
    monkeypatch.setattr(run_matrix, "project_dir", lambda: tmp_path)
    monkeypatch.setattr(
        run_matrix,
        "load_json",
        lambda path: default_matrix if "matrix" in str(path) else default_manifest,
    )

    called: dict[str, Any] = {}

    def fake_run_case(
        manifest_path, case, iterations, warmup, measurement_mode, command_line=None
    ):
        called[case["name"]] = command_line
        return fake_result_factory(case["name"]), fake_trace

    monkeypatch.setattr(run_matrix, "_run_case", fake_run_case)
    assert (
        run_matrix.main(
            [
                "--matrix",
                "matrix.json",
                "--output-dir",
                str(tmp_path / "out"),
                "--case-filter",
                "cpu_top1",
                "npu_top1",
                "--iterations",
                "1",
                "--warmup",
                "0",
                "--require-correctness",
            ]
        )
        == 0
    )
    summary = json.loads(
        (tmp_path / "out" / "summary.json").read_text(encoding="utf-8")
    )
    assert [case["case_name"] for case in summary["cases"]] == ["cpu_top1"]
    assert summary["skipped"][0]["case_name"] == "npu_top1"
    assert "run_matrix.py" in called["cpu_top1"]
    assert (tmp_path / "out" / "traces" / "cpu_top1.json").exists()

    def fake_run_case_with_trace(manifest, case, options):
        return fake_result_factory(options.case_name), fake_trace

    monkeypatch.setattr(run_matrix, "run_case_with_trace", fake_run_case_with_trace)
    result, trace = run_matrix._run_case(
        tmp_path / "manifest.json", default_matrix["cases"][0], 1, 0, "warm", ["cmd"]
    )
    assert result["case_name"] == "cpu_top1"
    assert trace is fake_trace


def test_run_matrix_failure_paths(
    monkeypatch,
    tmp_path: Path,
    default_manifest: dict,
    default_matrix: dict,
    fake_result_factory,
) -> None:
    monkeypatch.setattr(run_matrix, "project_dir", lambda: tmp_path)
    monkeypatch.setattr(
        run_matrix,
        "load_json",
        lambda path: default_matrix if "matrix" in str(path) else default_manifest,
    )
    bad = fake_result_factory("cpu_top1")
    bad["torch_validation"] = {"ok": False, "message": "bad"}
    monkeypatch.setattr(run_matrix, "_run_case", lambda *args, **kwargs: (bad, None))
    with pytest.raises(SystemExit, match="torch validation failed"):
        run_matrix.main(
            ["--matrix", "matrix.json", "--case-filter", "cpu_top1", "--require-torch"]
        )


def test_run_workload_suite_main(
    monkeypatch,
    tmp_path: Path,
    default_manifest: dict,
    default_matrix: dict,
    fake_result_factory,
) -> None:
    monkeypatch.setattr(run_workload_suite, "project_dir", lambda: tmp_path)
    monkeypatch.setattr(
        run_workload_suite,
        "load_json",
        lambda path: default_matrix if "matrix" in str(path) else default_manifest,
    )
    cpu_case = default_matrix["cases"][0]
    npu_case = default_matrix["cases"][6]
    workload = {
        "suite": "shape_sweep",
        "name": "tiny",
        "manifest": default_manifest,
        "cases": [cpu_case, npu_case],
    }
    monkeypatch.setattr(
        run_workload_suite,
        "suite_workloads",
        lambda suites, manifest, matrix: [workload],
    )
    monkeypatch.setattr(
        run_workload_suite,
        "routing_stats",
        lambda manifest: {"top1_token_counts": {}, "top2_probability_mass": {}},
    )
    monkeypatch.setattr(
        run_workload_suite,
        "run_case_with_trace",
        lambda manifest, case, options: (fake_result_factory(case["name"]), None),
    )
    assert (
        run_workload_suite.main(
            [
                "--matrix",
                "matrix.json",
                "--suite",
                "shape_sweep",
                "--output-dir",
                str(tmp_path / "suite"),
                "--case-filter",
                "cpu_top1",
                "npu_top1",
                "--require-correctness",
            ]
        )
        == 0
    )
    summary = json.loads(
        (tmp_path / "suite" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["workloads"][0]["cases"][0]["case_name"] == "cpu_top1"
    assert summary["workloads"][0]["skipped"][0]["case_name"] == "npu_top1"
    assert (tmp_path / "suite" / "summary.csv").exists()
    assert (tmp_path / "suite" / "report.md").exists()


def test_run_workload_suite_compile_cache(
    monkeypatch,
    tmp_path: Path,
    default_manifest: dict,
    default_matrix: dict,
    fake_result_factory,
) -> None:
    monkeypatch.setattr(run_workload_suite, "project_dir", lambda: tmp_path)
    monkeypatch.setattr(
        run_workload_suite,
        "load_json",
        lambda path: default_matrix if "matrix" in str(path) else default_manifest,
    )
    gpu_case = default_matrix["cases"][2]
    workload = {
        "suite": "shape_sweep",
        "name": "gpu",
        "manifest": default_manifest,
        "cases": [gpu_case],
    }
    monkeypatch.setattr(
        run_workload_suite,
        "suite_workloads",
        lambda suites, manifest, matrix: [workload, {**workload, "name": "gpu2"}],
    )
    monkeypatch.setattr(
        run_workload_suite,
        "routing_stats",
        lambda manifest: {"top1_token_counts": {}, "top2_probability_mass": {}},
    )
    monkeypatch.setattr(
        run_workload_suite,
        "run_case_with_trace",
        lambda manifest, case, options: (fake_result_factory(case["name"]), None),
    )
    calls = {"populate": 0}

    def fake_populate(manifest, backends):
        calls["populate"] += 1
        manifest["artifacts"]["router"]["gpu"] = {"so": "router.so"}
        return manifest

    monkeypatch.setattr(run_workload_suite, "populate_artifacts", fake_populate)
    assert (
        run_workload_suite.main(
            [
                "--matrix",
                "matrix.json",
                "--output-dir",
                str(tmp_path / "gpu_suite"),
                "--suite",
                "shape_sweep",
            ]
        )
        == 0
    )
    assert calls["populate"] == 1


def test_report_main(tmp_path: Path, fake_result_factory) -> None:
    summary = {"cases": [fake_result_factory("cpu")], "skipped": []}
    save_json(tmp_path / "summary.json", summary)
    assert (
        report.main(
            [
                "--summary",
                str(tmp_path / "summary.json"),
                "--out",
                str(tmp_path / "report.md"),
                "--title",
                "Title",
            ]
        )
        == 0
    )
    assert "# Title" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_edge_study_main(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(edge_study, "project_dir", lambda: tmp_path)
    commands: list[list[str]] = []

    def fake_run(cmd, cwd, check):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(edge_study.subprocess, "run", fake_run)
    monkeypatch.setattr(edge_study, "load_json", lambda path: {"workloads": []})
    assert (
        edge_study.main(
            [
                "--profile",
                "smoke",
                "--output-dir",
                str(tmp_path / "edge"),
                "--iterations",
                "1",
                "--warmup",
                "0",
                "--allow-npu",
                "--require-correctness",
                "--require-torch",
                "--workload-filter",
                "small",
                "--case-filter",
                "cpu_top2",
            ]
        )
        == 0
    )
    assert "run_workload_suite.py" in commands[0]
    assert "--allow-npu" in commands[0]
    assert (tmp_path / "edge" / "edge_efficiency_summary.json").exists()
    assert (tmp_path / "edge" / "edge_efficiency_report.md").exists()

    monkeypatch.setattr(
        edge_study.subprocess,
        "run",
        lambda cmd, cwd, check: subprocess.CompletedProcess(cmd, 7),
    )
    assert edge_study.main(["--output-dir", str(tmp_path / "edge_fail")]) == 7


def test_smoke_tests_main_lanes(
    monkeypatch, tmp_path: Path, default_manifest: dict, default_matrix: dict
) -> None:
    monkeypatch.setattr(smoke_tests, "project_dir", lambda: tmp_path)
    monkeypatch.setattr(
        smoke_tests,
        "load_json",
        lambda path: default_matrix if "matrix" in str(path) else default_manifest,
    )
    monkeypatch.setattr(smoke_tests, "golden_air_main", lambda: 0)
    monkeypatch.setattr(
        smoke_tests,
        "_run_cases",
        lambda **kwargs: [case["name"] for case in kwargs["cases"]],
    )
    assert (
        smoke_tests.main(
            [
                "--lane",
                "ci",
                "--matrix",
                "matrix.json",
                "--output-dir",
                str(tmp_path / "smoke"),
            ]
        )
        == 0
    )

    with pytest.raises(SystemExit, match="NPU smoke lanes require --allow-npu"):
        smoke_tests.main(["--lane", "npu", "--matrix", "matrix.json"])

    assert smoke_tests._expanded_lanes(["gpu-all"]) == {
        "golden",
        "cpu",
        "gpu-compile",
        "gpu",
        "mixed-gpu",
    }
    assert smoke_tests._case_uses(default_matrix["cases"][2], "gpu") is True


def test_smoke_tests_compile_and_missing_cases(
    monkeypatch, tmp_path: Path, default_manifest: dict, default_matrix: dict
) -> None:
    monkeypatch.setattr(smoke_tests, "project_dir", lambda: tmp_path)
    monkeypatch.setattr(
        smoke_tests,
        "load_json",
        lambda path: default_matrix if "matrix" in str(path) else default_manifest,
    )
    monkeypatch.setattr(smoke_tests, "golden_air_main", lambda: 0)
    monkeypatch.setattr(
        smoke_tests,
        "_run_cases",
        lambda **kwargs: [case["name"] for case in kwargs["cases"]],
    )
    calls: dict[str, Any] = {}

    def fake_populate(manifest, backends):
        calls["backends"] = backends
        return manifest

    monkeypatch.setattr(smoke_tests, "populate_artifacts", fake_populate)
    assert (
        smoke_tests.main(
            [
                "--lane",
                "gpu-compile",
                "gpu",
                "--matrix",
                "matrix.json",
                "--output-dir",
                str(tmp_path / "gpu"),
            ]
        )
        == 0
    )
    assert calls["backends"] == {"gpu"}

    monkeypatch.setattr(
        smoke_tests,
        "load_json",
        lambda path: {"cases": []} if "matrix" in str(path) else default_manifest,
    )
    with pytest.raises(SystemExit, match="matrix is missing expected cases"):
        smoke_tests.main(["--lane", "cpu", "--matrix", "matrix.json"])

    monkeypatch.setattr(smoke_tests, "golden_air_main", lambda: 1)
    monkeypatch.setattr(
        smoke_tests,
        "load_json",
        lambda path: default_matrix if "matrix" in str(path) else default_manifest,
    )
    assert smoke_tests.main(["--lane", "golden", "--matrix", "matrix.json"]) == 1
