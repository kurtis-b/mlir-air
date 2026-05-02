# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

import compile as moe_compile
from kernels import (
    KernelConfig,
    _expert_tile_sizes,
    _largest_divisor_at_most,
    _streaming_tile_sizes,
    aggregation_air,
    default_air_filenames,
    emit_all_kernels,
    expert_air,
    expert_hidden_air,
    expert_output_air,
    router_math_air,
    split_expert_air_filenames,
    write_default_air_sources,
    write_split_expert_air_sources,
)


def test_kernel_tile_helpers_and_text_generation(small_cfg: KernelConfig) -> None:
    assert small_cfg.router_weights == 2
    assert _largest_divisor_at_most(18, 8) == 6
    assert _largest_divisor_at_most(7, 4) == 1
    assert _expert_tile_sizes(KernelConfig(hidden_size=16, ffn_size=32)) == (32, 16)
    assert _streaming_tile_sizes(64, 32, reduction_limit=16, output_limit=8, max_weight_elems=64) == (16, 4)

    router = router_math_air(small_cfg)
    expert = expert_air(small_cfg)
    aggregation = aggregation_air(small_cfg)
    hidden = expert_hidden_air(small_cfg)
    output = expert_output_air(small_cfg)

    assert "func.func @router_math" in router
    assert "memref<2x4xf16>" in router
    assert "func.func @expert_mlp" in expert
    assert "expert_hidden_herd" in hidden
    assert "expert_output_herd" in output
    assert "func.func @aggregate_outputs" in aggregation
    assert set(emit_all_kernels(small_cfg)) == {"router", "expert", "aggregation"}


def test_kernel_filenames_and_file_writes(tmp_path: Path, small_cfg: KernelConfig) -> None:
    names = default_air_filenames(small_cfg)
    assert names["router"] == "router_math_2x4x2_f16.air.mlir"
    assert names["expert"] == "expert_mlp_2x4x8_f16.air.mlir"
    assert names["aggregation"] == "aggregation_2x4_f16.air.mlir"

    paths = write_default_air_sources(small_cfg, tmp_path / "default")
    assert paths["router"].read_text(encoding="utf-8").startswith("//===- router_math.air.mlir")
    split_names = split_expert_air_filenames(small_cfg)
    assert split_names["expert_hidden"] == "expert_hidden_2x4x8_f16.air.mlir"
    split_paths = write_split_expert_air_sources(small_cfg, tmp_path / "split")
    assert "expert_output" in split_paths["expert_output"].read_text(encoding="utf-8")


def test_tool_resolution(monkeypatch, tmp_path: Path) -> None:
    tool = tmp_path / "aircc"
    tool.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("AIRCC_PATH", str(tool))
    assert moe_compile._aircc_tool() == str(tool.resolve())

    monkeypatch.setenv("AIRCC_PATH", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="AIRCC_PATH is set but does not exist"):
        moe_compile._aircc_tool()

    monkeypatch.delenv("AIRCC_PATH", raising=False)
    monkeypatch.setattr(moe_compile.shutil, "which", lambda name: f"/usr/bin/{name}" if name == "aircc" else None)
    assert moe_compile._aircc_tool() == "/usr/bin/aircc"
    with pytest.raises(RuntimeError, match="Required tool 'missing'"):
        moe_compile._require_tool("missing")


def test_llvm_tool_resolution_and_gpu_libs(monkeypatch, tmp_path: Path) -> None:
    install = tmp_path / "llvm"
    (install / "bin").mkdir(parents=True)
    (install / "lib").mkdir()
    for name in ("clang", "mlir-opt", "mlir-translate"):
        (install / "bin" / name).write_text("", encoding="utf-8")
    monkeypatch.setenv("LLVM_INSTALL_DIR", str(install))
    assert moe_compile._llvm_lib_dir() == install / "lib"
    assert moe_compile._llvm_bin_dir() == install / "bin"
    assert moe_compile._llvm_clang().endswith("clang")
    assert moe_compile._llvm_mlir_opt().endswith("mlir-opt")
    assert moe_compile._llvm_mlir_translate().endswith("mlir-translate")

    rocm = tmp_path / "rocm"
    (rocm / "lib").mkdir(parents=True)
    (rocm / "lib" / "libamdhip64.so").write_text("", encoding="utf-8")
    monkeypatch.setenv("ROCM_PATH", str(rocm))
    libs = moe_compile.default_gpu_shared_libs()
    assert str(rocm / "lib" / "libamdhip64.so") in libs

    (install / "bin" / "clang").unlink()
    with pytest.raises(RuntimeError, match="Could not find LLVM clang"):
        moe_compile._llvm_clang()
    (install / "bin" / "mlir-opt").unlink()
    with pytest.raises(RuntimeError, match="Could not find mlir-opt"):
        moe_compile._llvm_mlir_opt()
    (install / "bin" / "mlir-translate").unlink()
    with pytest.raises(RuntimeError, match="Could not find mlir-translate"):
        moe_compile._llvm_mlir_translate()


def test_resolve_air_sources_writes_missing_sources(tmp_path: Path, default_manifest: dict) -> None:
    manifest = default_manifest
    manifest["paths"]["generated_air_sources"] = str(tmp_path / "air_sources")
    paths = moe_compile.resolve_air_sources(manifest, "gpu")
    assert set(paths) == {"router", "expert", "aggregation"}
    assert all(path.exists() for path in paths.values())
    with pytest.raises(ValueError, match="Unsupported source backend"):
        moe_compile.resolve_air_sources(manifest, "cpu")


def test_compile_npu_commands_and_stale_reuse(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "router.air.mlir"
    source.write_text("module {}\n", encoding="utf-8")
    tool = tmp_path / "aircc"
    tool.write_text("", encoding="utf-8")
    monkeypatch.setenv("AIRCC_PATH", str(tool))
    commands: list[list[str]] = []

    def fake_run(cmd, check):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(moe_compile.subprocess, "run", fake_run)
    artifact = moe_compile.compile_npu_with_args(source, tmp_path / "out", "npu2", ["--extra"])
    assert commands[0][:2] == [str(tool.resolve()), "--extra"]
    assert artifact["xclbin"].endswith(".npu2.xclbin")

    Path(artifact["xclbin"]).write_text("x", encoding="utf-8")
    Path(artifact["insts"]).write_text("i", encoding="utf-8")
    os.utime(artifact["xclbin"], (source.stat().st_mtime + 10, source.stat().st_mtime + 10))
    os.utime(artifact["insts"], (source.stat().st_mtime + 10, source.stat().st_mtime + 10))
    commands.clear()
    reused = moe_compile.compile_npu_if_needed(source, tmp_path / "out", "npu2")
    assert reused == artifact
    assert commands == []

    Path(artifact["insts"]).unlink()
    moe_compile.compile_npu_if_needed_with_args(source, tmp_path / "out", "npu2", ["--again"])
    assert commands


def test_compile_npu_split_success_fallback_and_failure(monkeypatch, tmp_path: Path, small_cfg: KernelConfig) -> None:
    calls: list[tuple[str, list[str]]] = []

    def fake_compile(source: Path, output_dir: Path, device: str, extra_args: list[str]) -> dict[str, str]:
        calls.append((source.name, list(extra_args)))
        return {"xclbin": str(output_dir / f"{source.stem}.xclbin"), "insts": str(output_dir / f"{source.stem}.insts.bin")}

    monkeypatch.setattr(moe_compile, "compile_npu_if_needed_with_args", fake_compile)
    artifact = moe_compile.compile_npu_parallel_expert(small_cfg, tmp_path / "src", tmp_path / "out", "npu2")
    assert artifact["mode"] == "parallel_split"
    assert {entry[0].split("_")[1] for entry in calls} >= {"hidden", "output"}

    state = {"fail_first": True}

    def fail_then_compile(source: Path, output_dir: Path, device: str, extra_args: list[str]) -> dict[str, str]:
        if state["fail_first"]:
            state["fail_first"] = False
            raise subprocess.CalledProcessError(1, ["aircc"], output="failed\nhidden")
        return fake_compile(source, output_dir, device, extra_args)

    monkeypatch.setattr(moe_compile, "compile_npu_if_needed_with_args", fail_then_compile)
    fallback = moe_compile.compile_npu_parallel_expert(small_cfg, tmp_path / "src2", tmp_path / "out2", "npu2")
    assert fallback["mode"] == "tiled_split"
    assert fallback["tiling"]["ffn_tiles"] >= 1

    def always_fail(source: Path, output_dir: Path, device: str, extra_args: list[str]) -> dict[str, str]:
        raise subprocess.CalledProcessError(1, ["aircc"], output="line1\nline2")

    monkeypatch.setattr(moe_compile, "compile_npu_if_needed_with_args", always_fail)
    with pytest.raises(RuntimeError, match="Failed to compile tiled NPU expert"):
        moe_compile.compile_npu_tiled_expert(small_cfg, tmp_path / "src3", tmp_path / "out3", "npu2")


def test_compile_gpu_command_construction(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "router.air.mlir"
    source.write_text("module {}\n", encoding="utf-8")
    commands: list[list[str]] = []

    monkeypatch.setattr(moe_compile, "_air_opt_tool", lambda: "/tools/air-opt")
    monkeypatch.setattr(moe_compile, "_llvm_mlir_opt", lambda: "/tools/mlir-opt")
    monkeypatch.setattr(moe_compile, "_llvm_mlir_translate", lambda: "/tools/mlir-translate")
    monkeypatch.setattr(moe_compile, "_llvm_clang", lambda: "/tools/clang")
    monkeypatch.setattr(moe_compile, "default_gpu_shared_libs", lambda: [str(tmp_path / "libmlir_runner_utils.so")])

    def fake_run(cmd, check):
        commands.append(cmd)
        if "-o" in cmd:
            out = Path(cmd[cmd.index("-o") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.suffix == ".ll":
                out.write_text("@llvm.global_dtors = appending global []\ndefine void @x() {}\n", encoding="utf-8")
            else:
                out.write_text("generated\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(moe_compile.subprocess, "run", fake_run)
    artifact = moe_compile.compile_gpu(source, tmp_path / "gpu", "gfx1150", "router_math")
    assert artifact["entry"] == "router_math"
    assert artifact["so"].endswith(".so")
    assert len(commands) == 4
    assert commands[0][0] == "/tools/air-opt"
    assert commands[-1][0] == "/tools/clang"
    assert "@llvm.global_dtors" not in Path(artifact["llvm"]).read_text(encoding="utf-8")


def test_gpu_split_and_populate_artifacts(monkeypatch, tmp_path: Path, small_cfg: KernelConfig, default_manifest: dict) -> None:
    def fake_compile_gpu(source: Path, output_dir: Path, arch: str, entrypoint: str) -> dict[str, str]:
        return {"mlir": str(output_dir / f"{source.stem}.mlir"), "llvm": str(output_dir / f"{source.stem}.ll"), "so": str(output_dir / f"{source.stem}.so"), "entry": entrypoint}

    monkeypatch.setattr(moe_compile, "compile_gpu", fake_compile_gpu)
    split = moe_compile.compile_gpu_parallel_expert(small_cfg, tmp_path / "src", tmp_path / "out", "gfx1150")
    assert split["mode"] == "parallel_split"
    assert split["hidden"]["artifact"]["entry"] == "expert_hidden"

    manifest = default_manifest
    manifest["paths"]["artifacts"] = str(tmp_path / "artifacts")
    manifest["paths"]["generated_air_sources"] = str(tmp_path / "sources")
    monkeypatch.setattr(moe_compile, "compile_npu", lambda source, output_dir, device: {"xclbin": str(output_dir / "x"), "insts": str(output_dir / "i")})
    monkeypatch.setattr(
        moe_compile,
        "compile_npu_parallel_expert",
        lambda cfg, source_dir, output_dir, device: {"mode": "parallel_split", "hidden": {}, "output": {}},
    )
    populated = moe_compile.populate_artifacts(manifest, {"gpu", "npu"})
    assert "gpu" in populated["artifacts"]["router"]
    assert "npu" in populated["artifacts"]["expert"]
