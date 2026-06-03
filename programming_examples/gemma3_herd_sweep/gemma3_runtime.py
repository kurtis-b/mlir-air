# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gemma3 synthetic runtime planning and cache manifest helpers.

This mirrors the Llama32 KernelCache/prepare_runtime boundary without importing
AIR/XRT. It records the kernel artifacts the Gemma model loop will need and
makes run-only mode fail early when the manifest is missing or inconsistent.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from gemma3_config import Gemma3KernelStep, Gemma3TextConfig, synthetic_text_config

MANIFEST_NAME = "gemma3_kernel_manifest.json"
MANIFEST_VERSION = 1


@dataclass(frozen=True)
class Gemma3KernelArtifact:
    cache_key: str
    phase: str
    layer_index: int
    kernel: str
    mode: str
    schedule_mode: str
    status: str
    static_inputs: tuple[str, ...]
    intermediate_outputs: tuple[str, ...]
    fallback: str | None = None

    @classmethod
    def from_step(cls, step: Gemma3KernelStep, config: Gemma3TextConfig) -> "Gemma3KernelArtifact":
        schedule = {
            "flowqkv": config.flowqkv_schedule_mode,
            "flowkv": config.flowkv_schedule_mode,
            "fused_dqp": config.fused_dqp_schedule_mode,
        }.get(step.kernel, "n/a")
        static_inputs, intermediate_outputs = _buffer_policy(step.kernel)
        cache_key = f"gemma3_{step.phase}_L{step.layer_index}_{step.kernel}"
        return cls(
            cache_key=cache_key,
            phase=step.phase,
            layer_index=step.layer_index,
            kernel=step.kernel,
            mode=step.mode,
            schedule_mode=schedule,
            status=step.status,
            static_inputs=static_inputs,
            intermediate_outputs=intermediate_outputs,
            fallback=step.fallback,
        )


def _buffer_policy(kernel: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if kernel == "q4nx":
        return ("packed_weights", "scale", "min_offset"), ("dequantized_weight",)
    if kernel == "bf16_mm":
        return ("weight_tile",), ("projection",)
    if kernel == "fused_dqp":
        return ("packed_weights", "scale", "min_offset"), ("projection",)
    if kernel in ("flowqkv", "flowkv"):
        return tuple(), ("attention",)
    if kernel in ("rms_norm", "qk_norm"):
        return ("norm_weight",), ("normalized",)
    if kernel == "rope":
        return ("rope_lut",), ("roped",)
    if kernel == "mlp_activation":
        return tuple(), ("activated",)
    if kernel == "residual_add":
        return tuple(), ("residual",)
    return tuple(), tuple()


def planned_artifacts(config: Gemma3TextConfig) -> tuple[Gemma3KernelArtifact, ...]:
    return tuple(Gemma3KernelArtifact.from_step(step, config) for step in config.kernel_sequence())


@dataclass
class Gemma3RuntimeManifest:
    version: int
    model_variant: str
    herd_shape: str
    artifacts: list[Gemma3KernelArtifact]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "model_variant": self.model_variant,
            "herd_shape": self.herd_shape,
            "artifacts": [asdict(artifact) for artifact in self.artifacts],
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, object]) -> "Gemma3RuntimeManifest":
        artifacts = [Gemma3KernelArtifact(**item) for item in data.get("artifacts", [])]
        return cls(
            version=int(data["version"]),
            model_variant=str(data["model_variant"]),
            herd_shape=str(data["herd_shape"]),
            artifacts=artifacts,
        )

    def validate_for(self, config: Gemma3TextConfig) -> None:
        if self.version != MANIFEST_VERSION:
            raise ValueError(f"unsupported manifest version: {self.version}")
        expected = planned_artifacts(config)
        expected_keys = [artifact.cache_key for artifact in expected]
        actual_keys = [artifact.cache_key for artifact in self.artifacts]
        if actual_keys != expected_keys:
            raise ValueError(
                "runtime manifest does not match current config: "
                f"expected {expected_keys}, got {actual_keys}"
            )


def manifest_path(cache_dir: Path | str) -> Path:
    return Path(cache_dir) / MANIFEST_NAME


def write_manifest(config: Gemma3TextConfig, cache_dir: Path | str) -> Gemma3RuntimeManifest:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    manifest = Gemma3RuntimeManifest(
        version=MANIFEST_VERSION,
        model_variant=config.model_variant,
        herd_shape=config.herd_shape,
        artifacts=list(planned_artifacts(config)),
    )
    manifest_path(cache_path).write_text(json.dumps(manifest.to_json_dict(), indent=2), encoding="utf-8")
    return manifest


def load_manifest(cache_dir: Path | str, config: Gemma3TextConfig) -> Gemma3RuntimeManifest:
    path = manifest_path(cache_dir)
    if not path.exists():
        raise FileNotFoundError(f"Gemma3 runtime manifest is missing: {path}")
    manifest = Gemma3RuntimeManifest.from_json_dict(json.loads(path.read_text(encoding="utf-8")))
    manifest.validate_for(config)
    return manifest


def prepare_runtime(
    config: Gemma3TextConfig | None = None,
    *,
    cache_dir: Path | str = "gemma3_kernel_cache",
    compile_only: bool = False,
    run_only: bool = False,
) -> Gemma3RuntimeManifest:
    if compile_only and run_only:
        raise ValueError("compile_only and run_only are mutually exclusive")
    config = config or synthetic_text_config()
    if compile_only:
        return write_manifest(config, cache_dir)
    if run_only:
        return load_manifest(cache_dir, config)
    raise ValueError("prepare_runtime requires compile_only or run_only")


def format_manifest(manifest: Gemma3RuntimeManifest) -> str:
    lines = [
        f"manifest_version={manifest.version}",
        f"model_variant={manifest.model_variant}",
        f"herd_shape={manifest.herd_shape}",
    ]
    for artifact in manifest.artifacts:
        fallback = f" fallback={artifact.fallback}" if artifact.fallback else ""
        lines.append(
            f"artifact {artifact.cache_key} phase={artifact.phase} "
            f"kernel={artifact.kernel} mode={artifact.mode} "
            f"schedule={artifact.schedule_mode} status={artifact.status}{fallback}"
        )
    return "\n".join(lines)


def artifact_keys(manifest: Gemma3RuntimeManifest | Iterable[Gemma3KernelArtifact]) -> tuple[str, ...]:
    artifacts = manifest.artifacts if isinstance(manifest, Gemma3RuntimeManifest) else manifest
    return tuple(artifact.cache_key for artifact in artifacts)
