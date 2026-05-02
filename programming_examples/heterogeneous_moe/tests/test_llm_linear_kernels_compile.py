# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pytest

from llm_linear import compile as linear_compile
from llm_linear.kernels import (
    LinearKernelConfig,
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
    assert "func.func @llm_linear_prefill" in prefill
    assert "memref<2x4xf16>" in prefill
    assert "func.func @llm_linear_decode" in decode
    assert "memref<8x3xf16>" in decode
    assert set(emit_all_kernels(cfg)) == {"prefill", "decode"}

    names = default_air_filenames(cfg)
    assert names["prefill"] == "prefill_gemm_m2_k4_h8_f16.air.mlir"
    assert names["decode"] == "decode_gemv_h8_n3_f16.air.mlir"
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
