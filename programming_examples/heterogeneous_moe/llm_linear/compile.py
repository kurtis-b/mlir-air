# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path
from typing import Any

from compile import compile_gpu, compile_npu

from .kernels import (
    LinearKernelConfig,
    default_air_filenames,
    kernel_config_from_manifest,
    write_default_air_sources,
)
from .manifest import artifact_root, generated_air_source_root

ENTRYPOINTS = {
    "prefill": "llm_linear_prefill",
    "decode": "llm_linear_decode",
}


def resolve_air_sources(manifest: dict[str, Any], backend: str) -> dict[str, Path]:
    if backend not in {"gpu", "npu"}:
        raise ValueError(f"Unsupported source backend: {backend}")
    cfg = kernel_config_from_manifest(manifest)
    source_dir = generated_air_source_root(manifest) / backend
    names = default_air_filenames(cfg)
    paths = {key: source_dir / name for key, name in names.items()}
    if not all(path.exists() for path in paths.values()):
        write_default_air_sources(
            cfg,
            source_dir,
            align_output_dma=(backend in {"gpu", "npu"}),
        )
    return paths


def populate_artifacts(
    manifest: dict[str, Any],
    backends: set[str],
    *,
    cfg: LinearKernelConfig | None = None,
    stage_backends: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    cfg = cfg or kernel_config_from_manifest(manifest)
    stage_backends = stage_backends or {
        key: set(backends) for key in default_air_filenames(cfg)
    }
    source_root = generated_air_source_root(manifest)
    gpu_sources = (
        write_default_air_sources(
            cfg,
            source_root / "gpu",
            align_output_dma=True,
        )
        if any("gpu" in needed for needed in stage_backends.values())
        else {}
    )
    npu_sources = (
        write_default_air_sources(cfg, source_root / "npu", align_output_dma=True)
        if any("npu" in needed for needed in stage_backends.values())
        else {}
    )
    artifact_dir = artifact_root(manifest)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    compiler_cfg = manifest["compiler"]
    artifacts = manifest.setdefault("artifacts", {})

    for key in default_air_filenames(cfg):
        artifact_entry = artifacts.setdefault(key, {})
        needed = stage_backends.get(key, set())
        if "npu" in needed:
            artifact_entry["npu"] = compile_npu(
                npu_sources[key], artifact_dir / "npu", compiler_cfg["npu_device"]
            )
        if "gpu" in needed:
            artifact_entry["gpu"] = compile_gpu(
                gpu_sources[key],
                artifact_dir / "gpu",
                compiler_cfg["gpu_arch"],
                ENTRYPOINTS[key],
            )
    return manifest


def populate_direct_gpu_artifacts(
    manifest: dict[str, Any],
    *,
    cfg: LinearKernelConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or kernel_config_from_manifest(manifest)
    sources = write_default_air_sources(
        cfg, generated_air_source_root(manifest) / "gpu"
    )
    artifact_dir = artifact_root(manifest) / "gpu_direct"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    compiler_cfg = manifest["compiler"]
    artifacts = manifest.setdefault("artifacts", {})
    for key, source in sources.items():
        artifact_entry = artifacts.setdefault(key, {})
        artifact_entry["gpu_direct"] = compile_gpu(
            source,
            artifact_dir,
            compiler_cfg["gpu_arch"],
            ENTRYPOINTS[key],
            host_staging=False,
        )
    return manifest
