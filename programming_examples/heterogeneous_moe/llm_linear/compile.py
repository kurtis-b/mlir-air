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

ENTRYPOINTS = {
    "prefill": "llm_linear_prefill",
    "decode": "llm_linear_decode",
    "decode_int4": "llm_linear_decode_int4",
}

NPU_COMPILE_ARGS = {
    "decode_int4": ["--air-channel-multiplexing=L1"],
}

NPU_PREFILL_TILE_H = 512
NPU_DECODE_TILE_N = 2048


def _npu_source_cfg(cfg: LinearKernelConfig) -> LinearKernelConfig:
    return LinearKernelConfig(
        M=1, K=cfg.K, H=min(cfg.H, NPU_PREFILL_TILE_H), N=cfg.N, dtype=cfg.dtype
    )


def _npu_decode_source_cfg(cfg: LinearKernelConfig) -> LinearKernelConfig:
    return LinearKernelConfig(
        M=cfg.M, K=cfg.K, H=cfg.H, N=min(cfg.N, NPU_DECODE_TILE_N), dtype=cfg.dtype
    )


def _write_npu_air_sources(
    manifest: dict[str, Any],
    *,
    include_decode_int4: bool,
) -> dict[str, Path]:
    cfg = kernel_config_from_manifest(manifest)
    prefill_cfg = _npu_source_cfg(cfg)
    quant = decode_quantization_config(manifest)
    source_dir = generated_air_source_root(manifest) / "npu"
    source_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    prefill_name = default_air_filenames(prefill_cfg)["prefill"]
    paths["prefill"] = source_dir / prefill_name
    paths["prefill"].write_text(
        prefill_gemm_air(prefill_cfg, align_output_dma=True), encoding="utf-8"
    )

    decode_name = default_air_filenames(cfg)["decode"]
    paths["decode"] = source_dir / decode_name
    paths["decode"].write_text(
        decode_gemv_air(cfg, align_output_dma=True), encoding="utf-8"
    )

    if include_decode_int4:
        assert quant is not None
        decode_cfg = _npu_decode_source_cfg(cfg)
        int4_names = default_air_filenames(
            decode_cfg,
            include_decode_int4=True,
            decode_int4_block_size=int(quant["block_size"]),
        )
        paths["decode_int4"] = source_dir / int4_names["decode_int4"]
        paths["decode_int4"].write_text(
            decode_int4_gemv_air(
                decode_cfg,
                block_size=int(quant["block_size"]),
                quant_axis=int(quant["quant_axis"]),
                align_output_dma=True,
            ),
            encoding="utf-8",
        )
    return paths


def decode_quantization_config(manifest: dict[str, Any]) -> dict[str, int | str] | None:
    decode = manifest.get("weights", {}).get("decode", {})
    if not isinstance(decode, dict):
        return None
    storage = decode.get("storage", "bf16")
    if storage in {None, "bf16", "dense"}:
        return None
    return {
        "storage": str(storage),
        "block_size": int(decode.get("block_size", 32)),
        "quant_axis": int(decode.get("quant_axis", 0)),
    }


def validate_accelerator_decode_quantization(
    manifest: dict[str, Any], *, require_int4: bool = False
) -> dict[str, int | str] | None:
    cfg = kernel_config_from_manifest(manifest)
    quant = decode_quantization_config(manifest)
    if quant is None:
        if require_int4:
            raise ValueError("accelerator quantized decode requires storage == int4")
        return None
    if quant["storage"] != "int4":
        raise ValueError("accelerator quantized decode supports only signed int4")
    block_size = int(quant["block_size"])
    if int(quant["quant_axis"]) != 0:
        raise ValueError("accelerator int4 decode requires quant_axis == 0")
    if cfg.H % block_size != 0:
        raise ValueError("accelerator int4 decode requires H % block_size == 0")
    if cfg.N % 8 != 0:
        raise ValueError("accelerator int4 decode requires N divisible by 8")
    return quant


def compile_npu_stage(
    kernel_key: str, source: Path, output_dir: Path, device: str
) -> dict[str, str]:
    extra_args = NPU_COMPILE_ARGS.get(kernel_key, [])
    if extra_args:
        return compile_npu_with_args(source, output_dir, device, extra_args)
    return compile_npu(source, output_dir, device)


def decode_kernel_key(manifest: dict[str, Any]) -> str:
    quant = decode_quantization_config(manifest)
    if quant is None:
        return "decode"
    validate_accelerator_decode_quantization(manifest, require_int4=True)
    return "decode_int4"


def _write_sources_for_manifest(
    manifest: dict[str, Any],
    backend: str,
    *,
    include_decode_int4: bool,
) -> dict[str, Path]:
    cfg = kernel_config_from_manifest(manifest)
    if backend == "npu":
        return _write_npu_air_sources(manifest, include_decode_int4=include_decode_int4)
    quant = validate_accelerator_decode_quantization(manifest, require_int4=True)
    assert quant is not None
    return write_default_air_sources(
        cfg,
        generated_air_source_root(manifest) / backend,
        align_output_dma=(backend in {"gpu", "npu"}),
        include_decode_int4=include_decode_int4,
        decode_int4_block_size=int(quant["block_size"]),
        decode_int4_quant_axis=int(quant["quant_axis"]),
        decode_int4_global_loads=(backend == "gpu"),
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
    quant = decode_quantization_config(manifest)
    if include_decode_int4 is None:
        include_decode_int4 = bool(quant is not None and quant["storage"] == "int4")
    if include_decode_int4:
        validate_accelerator_decode_quantization(manifest, require_int4=True)
    prefill_cfg = _npu_source_cfg(cfg) if backend == "npu" else cfg
    names = default_air_filenames(
        cfg,
        include_decode_int4=include_decode_int4,
        decode_int4_block_size=(32 if quant is None else int(quant["block_size"])),
    )
    if backend == "npu":
        names["prefill"] = default_air_filenames(prefill_cfg)["prefill"]
        if include_decode_int4:
            decode_cfg = _npu_decode_source_cfg(cfg)
            names["decode_int4"] = default_air_filenames(
                decode_cfg,
                include_decode_int4=True,
                decode_int4_block_size=(
                    32 if quant is None else int(quant["block_size"])
                ),
            )["decode_int4"]
    paths = {key: source_dir / name for key, name in names.items()}
    if not all(path.exists() for path in paths.values()):
        if backend == "npu":
            _write_npu_air_sources(manifest, include_decode_int4=include_decode_int4)
        elif include_decode_int4:
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
    quant = decode_quantization_config(manifest)
    include_decode_int4 = False
    if quant is not None and stage_backends.get("decode"):
        validate_accelerator_decode_quantization(manifest, require_int4=True)
        include_decode_int4 = True
        stage_backends = {
            ("decode_int4" if key == "decode" else key): set(value)
            for key, value in stage_backends.items()
        }
    source_root = generated_air_source_root(manifest)
    gpu_sources = (
        write_default_air_sources(
            cfg,
            source_root / "gpu",
            align_output_dma=True,
            include_decode_int4=include_decode_int4,
            decode_int4_block_size=(32 if quant is None else int(quant["block_size"])),
            decode_int4_quant_axis=(0 if quant is None else int(quant["quant_axis"])),
            decode_int4_global_loads=True,
        )
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
            if key == "decode_int4":
                artifact_entry["npu"]["tile_n"] = min(cfg.N, NPU_DECODE_TILE_N)
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
    quant = decode_quantization_config(manifest)
    include_decode_int4 = quant is not None
    if include_decode_int4:
        validate_accelerator_decode_quantization(manifest, require_int4=True)
    sources = write_default_air_sources(
        cfg,
        generated_air_source_root(manifest) / "gpu",
        include_decode_int4=include_decode_int4,
        decode_int4_block_size=(32 if quant is None else int(quant["block_size"])),
        decode_int4_quant_axis=(0 if quant is None else int(quant["quant_axis"])),
        decode_int4_global_loads=True,
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
