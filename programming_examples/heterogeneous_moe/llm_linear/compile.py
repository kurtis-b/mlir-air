# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path
from typing import Any

from compile import compile_gpu, compile_npu, compile_npu_with_args

from .kernels import (
    LinearKernelConfig,
    default_air_filenames,
    decode_gemv_air,
    decode_int4_gemv_air,
    kernel_config_from_manifest,
    prefill_gemm_air,
    write_default_air_sources,
)
from .manifest import artifact_root, generated_air_source_root
from .quantization import (
    DecodeQuantizationPlan,
    decode_quantization_plan_from_manifest,
    validate_accelerator_decode_plan,
)

ENTRYPOINTS = {
    "prefill": "llm_linear_prefill",
    "decode": "llm_linear_decode",
    "decode_int4": "llm_linear_decode_int4",
}

NPU_COMPILE_ARGS = {
    "decode": ["--air-channel-multiplexing=L1"],
    "decode_int4": ["--air-channel-multiplexing=L1"],
}

NPU_PREFILL_TILE_H = 512
NPU_DENSE_DECODE_TILE_H = 256
NPU_DENSE_DECODE_TILE_N = 256
NPU_DENSE_DECODE_MAX_OUTER_TILE_H = 4096
NPU_DECODE_TILE_N = 2048
GPU_DENSE_DECODE_TILE_H = 256
GPU_INT4_DECODE_MEMORY_STRATEGY = "streamed_l1"


def _npu_source_cfg(cfg: LinearKernelConfig) -> LinearKernelConfig:
    return LinearKernelConfig(
        M=1, K=cfg.K, H=min(cfg.H, NPU_PREFILL_TILE_H), N=cfg.N, dtype=cfg.dtype
    )


def _gpu_prefill_source_cfg(cfg: LinearKernelConfig) -> LinearKernelConfig:
    return LinearKernelConfig(M=1, K=cfg.K, H=cfg.H, N=cfg.N, dtype=cfg.dtype)


def _npu_decode_source_cfg(cfg: LinearKernelConfig) -> LinearKernelConfig:
    return LinearKernelConfig(
        M=cfg.M, K=cfg.K, H=cfg.H, N=min(cfg.N, NPU_DECODE_TILE_N), dtype=cfg.dtype
    )


def _require_dense_decode_tile_shape(
    cfg: LinearKernelConfig, *, tile_h: int, tile_n: int, backend: str
) -> None:
    if tile_h <= 0:
        raise ValueError(f"{backend} dense decode tile_h must be positive")
    if tile_n <= 0:
        raise ValueError(f"{backend} dense decode tile_n must be positive")
    if cfg.H > tile_h and cfg.H % tile_h != 0:
        raise ValueError(
            f"{backend} dense decode requires H divisible by tile_h "
            "when H exceeds tile_h"
        )
    if cfg.N > tile_n and cfg.N % tile_n != 0:
        raise ValueError(
            f"{backend} dense decode requires N divisible by tile_n "
            "when N exceeds tile_n"
        )


def _npu_dense_decode_outer_tile_h(cfg: LinearKernelConfig) -> int:
    if cfg.H <= NPU_DENSE_DECODE_TILE_H:
        return cfg.H
    candidates = []
    for tile_count in (1, 2, 4, 5, 6, 7, 8, 11, 16, 32, 43):
        if cfg.H % tile_count == 0:
            candidate = cfg.H // tile_count
            if candidate >= NPU_DENSE_DECODE_TILE_H:
                candidates.append(candidate)
    bounded = [
        value for value in candidates if value <= NPU_DENSE_DECODE_MAX_OUTER_TILE_H
    ]
    if bounded:
        return max(bounded)
    if cfg.H % NPU_DENSE_DECODE_TILE_H == 0:
        return NPU_DENSE_DECODE_TILE_H
    raise ValueError(
        "NPU dense decode requires H to be divisible by a supported outer tile"
    )


def _npu_dense_decode_inner_tile_h(outer_tile_h: int) -> int:
    if outer_tile_h <= NPU_DENSE_DECODE_TILE_H:
        return outer_tile_h
    for tile_count in (4, 5, 6, 7, 8):
        if outer_tile_h % tile_count == 0:
            return outer_tile_h // tile_count
    if outer_tile_h % NPU_DENSE_DECODE_TILE_H == 0:
        return NPU_DENSE_DECODE_TILE_H
    raise ValueError("NPU dense decode outer tile requires a supported inner H tile")


def _npu_dense_decode_source_cfg(cfg: LinearKernelConfig) -> LinearKernelConfig:
    outer_tile_h = _npu_dense_decode_outer_tile_h(cfg)
    _require_dense_decode_tile_shape(
        cfg,
        tile_h=outer_tile_h,
        tile_n=NPU_DENSE_DECODE_TILE_N,
        backend="NPU",
    )
    return LinearKernelConfig(
        M=cfg.M,
        K=cfg.K,
        H=outer_tile_h,
        N=min(cfg.N, NPU_DENSE_DECODE_TILE_N),
        dtype=cfg.dtype,
    )


def _gpu_dense_decode_source_cfg(cfg: LinearKernelConfig) -> LinearKernelConfig:
    _require_dense_decode_tile_shape(
        cfg,
        tile_h=GPU_DENSE_DECODE_TILE_H,
        tile_n=cfg.N,
        backend="GPU",
    )
    return LinearKernelConfig(
        M=cfg.M,
        K=cfg.K,
        H=min(cfg.H, GPU_DENSE_DECODE_TILE_H),
        N=cfg.N,
        dtype=cfg.dtype,
    )


def _gpu_dense_decode_tile_metadata(cfg: LinearKernelConfig) -> dict[str, int]:
    _require_dense_decode_tile_shape(
        cfg,
        tile_h=GPU_DENSE_DECODE_TILE_H,
        tile_n=cfg.N,
        backend="GPU",
    )
    return {"tile_h": min(cfg.H, GPU_DENSE_DECODE_TILE_H), "tile_n": cfg.N}


def _npu_dense_decode_tile_metadata(cfg: LinearKernelConfig) -> dict[str, int]:
    outer_tile_h = _npu_dense_decode_outer_tile_h(cfg)
    _require_dense_decode_tile_shape(
        cfg,
        tile_h=outer_tile_h,
        tile_n=NPU_DENSE_DECODE_TILE_N,
        backend="NPU",
    )
    return {
        "tile_h": outer_tile_h,
        "tile_n": min(cfg.N, NPU_DENSE_DECODE_TILE_N),
    }


def decode_quantization_plan(manifest: dict[str, Any]) -> DecodeQuantizationPlan | None:
    cfg = kernel_config_from_manifest(manifest)
    return decode_quantization_plan_from_manifest(
        manifest,
        shape=(cfg.H, cfg.N),
        npu_decode_tile_n=min(cfg.N, NPU_DECODE_TILE_N),
    )


def _write_npu_air_sources(
    manifest: dict[str, Any],
    *,
    include_decode_int4: bool,
) -> dict[str, Path]:
    cfg = kernel_config_from_manifest(manifest)
    prefill_cfg = _npu_source_cfg(cfg)
    quant = decode_quantization_plan(manifest)
    source_dir = generated_air_source_root(manifest) / "npu"
    source_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    prefill_name = default_air_filenames(prefill_cfg)["prefill"]
    paths["prefill"] = source_dir / prefill_name
    paths["prefill"].write_text(
        prefill_gemm_air(prefill_cfg, align_output_dma=True), encoding="utf-8"
    )

    dense_decode_cfg = _npu_dense_decode_source_cfg(cfg)
    decode_name = default_air_filenames(dense_decode_cfg)["decode"]
    paths["decode"] = source_dir / decode_name
    paths["decode"].write_text(
        decode_gemv_air(
            dense_decode_cfg,
            align_output_dma=True,
            dense_tile_h=_npu_dense_decode_inner_tile_h(dense_decode_cfg.H),
        ),
        encoding="utf-8",
    )

    if include_decode_int4:
        assert quant is not None
        decode_cfg = _npu_decode_source_cfg(cfg)
        int4_names = default_air_filenames(
            decode_cfg,
            include_decode_int4=True,
            decode_int4_block_size=quant.block_size,
        )
        paths["decode_int4"] = source_dir / int4_names["decode_int4"]
        paths["decode_int4"].write_text(
            decode_int4_gemv_air(
                decode_cfg,
                block_size=quant.block_size,
                quant_axis=quant.quant_axis,
                align_output_dma=True,
                memory_strategy="staged_l2",
            ),
            encoding="utf-8",
        )
    return paths


def _write_gpu_air_sources(
    manifest: dict[str, Any],
    *,
    include_decode_int4: bool,
) -> dict[str, Path]:
    cfg = kernel_config_from_manifest(manifest)
    prefill_cfg = _gpu_prefill_source_cfg(cfg)
    quant = decode_quantization_plan(manifest)
    source_dir = generated_air_source_root(manifest) / "gpu"
    source_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    prefill_name = default_air_filenames(prefill_cfg)["prefill"]
    paths["prefill"] = source_dir / prefill_name
    paths["prefill"].write_text(
        prefill_gemm_air(prefill_cfg, align_output_dma=True), encoding="utf-8"
    )

    dense_decode_cfg = _gpu_dense_decode_source_cfg(cfg)
    decode_name = default_air_filenames(dense_decode_cfg)["decode"]
    paths["decode"] = source_dir / decode_name
    paths["decode"].write_text(
        decode_gemv_air(
            dense_decode_cfg,
            align_output_dma=True,
            dense_tile_h=GPU_DENSE_DECODE_TILE_H,
        ),
        encoding="utf-8",
    )

    if include_decode_int4:
        assert quant is not None
        int4_names = default_air_filenames(
            cfg,
            include_decode_int4=True,
            decode_int4_block_size=quant.block_size,
        )
        paths["decode_int4"] = source_dir / int4_names["decode_int4"]
        paths["decode_int4"].write_text(
            decode_int4_gemv_air(
                cfg,
                block_size=quant.block_size,
                quant_axis=quant.quant_axis,
                align_output_dma=True,
                memory_strategy=GPU_INT4_DECODE_MEMORY_STRATEGY,
            ),
            encoding="utf-8",
        )
    return paths


def decode_quantization_config(manifest: dict[str, Any]) -> dict[str, Any] | None:
    plan = decode_quantization_plan(manifest)
    return None if plan is None else plan.to_metadata_dict()


def validate_accelerator_decode_quantization(
    manifest: dict[str, Any], *, require_int4: bool = False
) -> dict[str, Any] | None:
    plan = validate_accelerator_decode_plan(
        decode_quantization_plan(manifest), require_int4=require_int4
    )
    return None if plan is None else plan.to_metadata_dict()


def compile_npu_stage(
    kernel_key: str, source: Path, output_dir: Path, device: str
) -> dict[str, str]:
    extra_args = NPU_COMPILE_ARGS.get(kernel_key, [])
    if extra_args:
        return compile_npu_with_args(source, output_dir, device, extra_args)
    return compile_npu(source, output_dir, device)


def decode_kernel_key(manifest: dict[str, Any]) -> str:
    quant = decode_quantization_plan(manifest)
    if quant is None:
        return "decode"
    validate_accelerator_decode_plan(quant, require_int4=True)
    return quant.kernel_key


def _write_sources_for_manifest(
    manifest: dict[str, Any],
    backend: str,
    *,
    include_decode_int4: bool,
) -> dict[str, Path]:
    cfg = kernel_config_from_manifest(manifest)
    if backend == "npu":
        return _write_npu_air_sources(manifest, include_decode_int4=include_decode_int4)
    if backend == "gpu":
        return _write_gpu_air_sources(manifest, include_decode_int4=include_decode_int4)
    quant = validate_accelerator_decode_plan(
        decode_quantization_plan(manifest), require_int4=True
    )
    assert quant is not None
    return write_default_air_sources(
        cfg,
        generated_air_source_root(manifest) / backend,
        align_output_dma=(backend in {"gpu", "npu"}),
        include_decode_int4=include_decode_int4,
        decode_int4_block_size=quant.block_size,
        decode_int4_quant_axis=quant.quant_axis,
        decode_int4_memory_strategy=(
            GPU_INT4_DECODE_MEMORY_STRATEGY if backend == "gpu" else "staged_l2"
        ),
    )


def resolve_air_sources(
    manifest: dict[str, Any],
    backend: str,
    *,
    include_decode_int4: bool | None = None,
) -> dict[str, Path]:
    if backend not in {"gpu", "npu"}:
        raise ValueError(f"Unsupported source backend: {backend}")
    cfg = kernel_config_from_manifest(manifest)
    source_dir = generated_air_source_root(manifest) / backend
    quant = decode_quantization_plan(manifest)
    if include_decode_int4 is None:
        include_decode_int4 = bool(quant is not None and quant.storage == "int4")
    if include_decode_int4:
        validate_accelerator_decode_quantization(manifest, require_int4=True)
    prefill_cfg = _npu_source_cfg(cfg) if backend == "npu" else cfg
    names = default_air_filenames(
        cfg,
        include_decode_int4=include_decode_int4,
        decode_int4_block_size=(32 if quant is None else quant.block_size),
    )
    if backend == "npu":
        names["prefill"] = default_air_filenames(prefill_cfg)["prefill"]
        dense_decode_cfg = _npu_dense_decode_source_cfg(cfg)
        names["decode"] = default_air_filenames(dense_decode_cfg)["decode"]
        if include_decode_int4:
            decode_cfg = _npu_decode_source_cfg(cfg)
            names["decode_int4"] = default_air_filenames(
                decode_cfg,
                include_decode_int4=True,
                decode_int4_block_size=(32 if quant is None else quant.block_size),
            )["decode_int4"]
    if backend == "gpu":
        prefill_cfg = _gpu_prefill_source_cfg(cfg)
        names["prefill"] = default_air_filenames(prefill_cfg)["prefill"]
        dense_decode_cfg = _gpu_dense_decode_source_cfg(cfg)
        names["decode"] = default_air_filenames(dense_decode_cfg)["decode"]
    paths = {key: source_dir / name for key, name in names.items()}
    if not all(path.exists() for path in paths.values()):
        if backend == "npu":
            _write_npu_air_sources(manifest, include_decode_int4=include_decode_int4)
        elif backend == "gpu":
            _write_sources_for_manifest(
                manifest, backend, include_decode_int4=include_decode_int4
            )
        else:
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
    quant = decode_quantization_plan(manifest)
    include_decode_int4 = False
    if quant is not None and stage_backends.get("decode"):
        validate_accelerator_decode_plan(quant, require_int4=True)
        include_decode_int4 = True
        stage_backends = {
            ("decode_int4" if key == "decode" else key): set(value)
            for key, value in stage_backends.items()
        }
    source_root = generated_air_source_root(manifest)
    gpu_sources = (
        _write_gpu_air_sources(manifest, include_decode_int4=include_decode_int4)
        if any("gpu" in needed for needed in stage_backends.values())
        else {}
    )
    npu_sources = (
        _write_npu_air_sources(manifest, include_decode_int4=include_decode_int4)
        if any("npu" in needed for needed in stage_backends.values())
        else {}
    )
    artifact_dir = artifact_root(manifest)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    compiler_cfg = manifest["compiler"]
    artifacts = manifest.setdefault("artifacts", {})

    for key in stage_backends:
        artifact_entry = artifacts.setdefault(key, {})
        needed = stage_backends.get(key, set())
        if "npu" in needed:
            artifact_entry["npu"] = compile_npu_stage(
                key,
                npu_sources[key],
                artifact_dir / "npu",
                compiler_cfg["npu_device"],
            )
            if key == "prefill":
                artifact_entry["npu"]["tile_h"] = min(cfg.H, NPU_PREFILL_TILE_H)
            if key == "decode":
                artifact_entry["npu"].update(_npu_dense_decode_tile_metadata(cfg))
            if key == "decode_int4":
                artifact_entry["npu"]["tile_n"] = min(cfg.N, NPU_DECODE_TILE_N)
        if "gpu" in needed:
            artifact_entry["gpu"] = compile_gpu(
                gpu_sources[key],
                artifact_dir / "gpu",
                compiler_cfg["gpu_arch"],
                ENTRYPOINTS[key],
            )
            if key == "prefill":
                artifact_entry["gpu"]["source_m"] = 1
            if key == "decode":
                artifact_entry["gpu"].update(_gpu_dense_decode_tile_metadata(cfg))
    return manifest


def populate_direct_gpu_artifacts(
    manifest: dict[str, Any],
    *,
    cfg: LinearKernelConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or kernel_config_from_manifest(manifest)
    quant = decode_quantization_plan(manifest)
    include_decode_int4 = quant is not None
    if include_decode_int4:
        validate_accelerator_decode_plan(quant, require_int4=True)
    sources = _write_gpu_air_sources(manifest, include_decode_int4=include_decode_int4)
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
        if key == "decode":
            artifact_entry["gpu_direct"].update(_gpu_dense_decode_tile_metadata(cfg))
    return manifest
