# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pytest

from llm_linear import compile as linear_compile
from llm_linear.kernels import (
    LinearKernelConfig,
    decode_int4_gemv_air,
    decode_gemv_air,
    default_air_filenames,
    emit_all_kernels,
    prefill_gemm_air,
    write_default_air_sources,
)
from llm_linear.manifest import load_json


def test_linear_air_text_generation(tmp_path: Path) -> None:
    cfg = LinearKernelConfig(M=2, K=4, H=8, N=3, dtype="f16")
    prefill = prefill_gemm_air(cfg)
    decode = decode_gemv_air(cfg)
    npu_prefill = prefill_gemm_air(cfg, align_output_dma=True)
    npu_decode = decode_gemv_air(
        LinearKernelConfig(M=2, K=4, H=8, N=4, dtype="f16"),
        align_output_dma=True,
    )
    assert "func.func @llm_linear_prefill" in prefill
    assert "memref<2x4xf16>" in prefill
    assert "func.func @llm_linear_decode" in decode
    assert "memref<8x3xf16>" in decode
    assert "memref<4x8xf16" in npu_prefill
    assert "memref<8x2xf16" in npu_decode
    assert set(emit_all_kernels(cfg)) == {"prefill", "decode"}
    int4 = decode_int4_gemv_air(
        LinearKernelConfig(M=2, K=4, H=8, N=8, dtype="f16"),
        block_size=4,
        quant_axis=0,
    )
    assert "func.func @llm_linear_decode_int4" in int4
    assert "memref<8x1xi32>" in int4
    assert "memref<2x8xf32>" in int4
    assert "scf.for %bb" in int4
    assert "arith.shrui" in int4
    streamed = decode_int4_gemv_air(
        LinearKernelConfig(M=2, K=4, H=8, N=8, dtype="f16"),
        block_size=4,
        quant_axis=0,
        memory_strategy="streamed_l1",
    )
    assert "memref<2x8xf32>" in streamed
    assert set(
        emit_all_kernels(
            LinearKernelConfig(M=2, K=4, H=8, N=8, dtype="f16"),
            include_decode_int4=True,
            decode_int4_block_size=4,
        )
    ) == {"prefill", "decode", "decode_int4"}

    names = default_air_filenames(cfg)
    assert names["prefill"] == "prefill_gemm_m2_k4_h8_f16.air.mlir"
    assert names["decode"] == "decode_gemv_h8_n3_f16.air.mlir"
    int4_names = default_air_filenames(
        LinearKernelConfig(M=2, K=4, H=8, N=8, dtype="f16"),
        include_decode_int4=True,
        decode_int4_block_size=4,
    )
    assert int4_names["decode_int4"] == "decode_int4_h8_n8_b4_f16.air.mlir"
    paths = write_default_air_sources(cfg, tmp_path)
    assert (
        paths["prefill"]
        .read_text(encoding="utf-8")
        .startswith("//===- prefill_gemm.air.mlir")
    )


def test_linear_compile_populates_artifacts(
    monkeypatch, tmp_path: Path, moe_dir: Path
) -> None:
    manifest = load_json(moe_dir / "llm_linear" / "default_linear_manifest.json")
    manifest["paths"]["artifacts"] = str(tmp_path / "artifacts")
    manifest["paths"]["generated_air_sources"] = str(tmp_path / "air")
    calls: list[tuple[str, str]] = []

    def fake_gpu(
        source: Path, output_dir: Path, arch: str, entrypoint: str
    ) -> dict[str, str]:
        calls.append(("gpu", entrypoint))
        return {"so": str(output_dir / f"{source.stem}.so"), "entry": entrypoint}

    def fake_npu(source: Path, output_dir: Path, device: str) -> dict[str, str]:
        calls.append(("npu", device))
        return {
            "xclbin": str(output_dir / f"{source.stem}.xclbin"),
            "insts": str(output_dir / f"{source.stem}.insts.bin"),
        }

    monkeypatch.setattr(linear_compile, "compile_gpu", fake_gpu)
    monkeypatch.setattr(linear_compile, "compile_npu", fake_npu)
    populated = linear_compile.populate_artifacts(manifest, {"gpu", "npu"})
    assert populated["artifacts"]["prefill"]["gpu"]["entry"] == "llm_linear_prefill"
    assert populated["artifacts"]["decode"]["npu"]["xclbin"].endswith(".xclbin")
    assert ("gpu", "llm_linear_decode") in calls

    sources = linear_compile.resolve_air_sources(manifest, "gpu")
    assert set(sources) == {"prefill", "decode"}
    with pytest.raises(ValueError, match="Unsupported source backend"):
        linear_compile.resolve_air_sources(manifest, "cpu")


def test_linear_compile_routes_int4_decode_artifacts(
    monkeypatch, tmp_path: Path, moe_dir: Path
) -> None:
    manifest = load_json(moe_dir / "llm_linear" / "default_linear_manifest.json")
    manifest["model"].update({"H": 64, "N": 32})
    manifest["weights"]["decode"] = {
        "storage": "int4",
        "block_size": 32,
        "quant_axis": 0,
    }
    manifest["paths"]["artifacts"] = str(tmp_path / "artifacts")
    manifest["paths"]["generated_air_sources"] = str(tmp_path / "air")
    calls: list[tuple[str, str, str]] = []

    def fake_gpu(
        source: Path, output_dir: Path, arch: str, entrypoint: str
    ) -> dict[str, str]:
        calls.append(("gpu", source.name, entrypoint))
        return {"so": str(output_dir / f"{source.stem}.so"), "entry": entrypoint}

    def fake_npu(source: Path, output_dir: Path, device: str) -> dict[str, str]:
        calls.append(("npu", source.name, device))
        return {
            "xclbin": str(output_dir / f"{source.stem}.xclbin"),
            "insts": str(output_dir / f"{source.stem}.insts.bin"),
        }

    def fake_npu_with_args(
        source: Path, output_dir: Path, device: str, extra_args: list[str]
    ) -> dict[str, str]:
        calls.append(("npu", source.name, " ".join(extra_args)))
        return {
            "xclbin": str(output_dir / f"{source.stem}.xclbin"),
            "insts": str(output_dir / f"{source.stem}.insts.bin"),
        }

    monkeypatch.setattr(linear_compile, "compile_gpu", fake_gpu)
    monkeypatch.setattr(linear_compile, "compile_npu", fake_npu)
    monkeypatch.setattr(linear_compile, "compile_npu_with_args", fake_npu_with_args)
    populated = linear_compile.populate_artifacts(
        manifest,
        {"gpu", "npu"},
        stage_backends={"decode": {"gpu", "npu"}},
    )
    assert populated["artifacts"].get("decode", {}) == {}
    assert populated["artifacts"]["decode_int4"]["gpu"]["entry"] == (
        "llm_linear_decode_int4"
    )
    assert any("decode_int4" in source for _, source, _ in calls)
    assert (
        "npu",
        "decode_int4_h64_n32_b32_bf16.air.mlir",
        "--air-channel-multiplexing=L1",
    ) in calls

    sources = linear_compile.resolve_air_sources(manifest, "gpu")
    assert "decode_int4" in sources
    assert (
        sources["decode_int4"]
        .read_text(encoding="utf-8")
        .count("llm_linear_decode_int4")
    )


def test_linear_compile_does_not_reject_cpu_only_uint4_decode_for_prefill_artifacts(
    monkeypatch, tmp_path: Path, moe_dir: Path
) -> None:
    manifest = load_json(moe_dir / "llm_linear" / "default_linear_manifest.json")
    manifest["weights"]["decode"] = {
        "storage": "uint4",
        "block_size": 16,
        "quant_axis": 1,
    }
    manifest["paths"]["artifacts"] = str(tmp_path / "artifacts")
    manifest["paths"]["generated_air_sources"] = str(tmp_path / "air")

    def fake_gpu(
        source: Path, output_dir: Path, arch: str, entrypoint: str
    ) -> dict[str, str]:
        return {"so": str(output_dir / f"{source.stem}.so"), "entry": entrypoint}

    monkeypatch.setattr(linear_compile, "compile_gpu", fake_gpu)
    populated = linear_compile.populate_artifacts(
        manifest,
        {"gpu"},
        stage_backends={"prefill": {"gpu"}},
    )
    assert populated["artifacts"]["prefill"]["gpu"]["entry"] == "llm_linear_prefill"
    assert linear_compile.resolve_air_sources(
        manifest, "gpu", include_decode_int4=False
    ).keys() == {"prefill", "decode"}
