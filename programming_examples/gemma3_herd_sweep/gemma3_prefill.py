# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Synthetic Gemma3 text prefill orchestration.

This module owns the host-driven prefill layer loop while the Gemma3 NPU model
path is still being assembled. It mirrors the Llama32 prefill split by keeping
per-layer execution, manifest validation, buffer names, and stage records
explicit, but it runs through the CPU reference until each sub-kernel is wired
to a validated artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from gemma3_config import Gemma3LayerConfig, Gemma3TextConfig, synthetic_text_config
from gemma3_reference import (
    Gemma3KVCache,
    embedding_reference,
    logits_reference,
    prefill_layer_reference,
    synthetic_token_ids,
)
from gemma3_runtime import Gemma3RuntimeManifest, load_manifest, prepare_runtime
from gemma3_weights import Gemma3SyntheticWeights, synthetic_weights


PREFILL_STAGE_ORDER = (
    "rms_norm",
    "q4nx",
    "bf16_mm",
    "rope",
    "qk_norm",
    "flowqkv",
    "o_proj",
    "residual_add",
    "mlp_activation",
)


@dataclass(frozen=True)
class Gemma3StageRecord:
    phase: str
    layer_index: int
    stage: str
    status: str
    mode: str = "n/a"
    fallback: str | None = None
    shape: tuple[int, ...] = ()
    checksum: float = 0.0
    token_base: int | None = None
    cache_len_before: int | None = None
    cache_len_after: int | None = None
    kv_read_start: int | None = None
    kv_read_end: int | None = None

    def format(self) -> str:
        shape = "x".join(str(dim) for dim in self.shape) if self.shape else "scalar"
        line = (
            f"{self.phase}:L{self.layer_index}:{self.stage} "
            f"status={self.status} mode={self.mode} shape={shape} "
            f"checksum={self.checksum:.6f}"
        )
        if self.fallback:
            line += f" fallback={self.fallback}"
        if self.token_base is not None:
            line += f" token_base={self.token_base}"
        if self.cache_len_before is not None and self.cache_len_after is not None:
            line += f" cache={self.cache_len_before}->{self.cache_len_after}"
        if self.kv_read_start is not None and self.kv_read_end is not None:
            line += f" kv_read={self.kv_read_start}:{self.kv_read_end}"
        return line


@dataclass
class Gemma3PrefillLayerReport:
    layer: Gemma3LayerConfig
    output: np.ndarray
    stages: list[Gemma3StageRecord] = field(default_factory=list)


@dataclass
class Gemma3PrefillReport:
    output: np.ndarray
    logits: np.ndarray
    stages: list[Gemma3StageRecord]

    def format(self) -> str:
        return "\n".join(stage.format() for stage in self.stages)


def _checksum(value: np.ndarray) -> float:
    return float(np.sum(np.asarray(value, dtype=np.float32)))


def _shape(value: np.ndarray) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value.shape)


def _artifact_map(
    manifest: Gemma3RuntimeManifest | None,
) -> dict[tuple[str, int, str], object]:
    if manifest is None:
        return {}
    return {
        (artifact.phase, artifact.layer_index, artifact.kernel): artifact
        for artifact in manifest.artifacts
    }


def _stage_from_artifact(
    artifacts: dict[tuple[str, int, str], object],
    *,
    phase: str,
    layer_index: int,
    stage: str,
    value: np.ndarray,
    token_base: int,
    cache_len_before: int,
    cache_len_after: int,
    kv_read_start: int | None = None,
    kv_read_end: int | None = None,
) -> Gemma3StageRecord:
    artifact = artifacts.get((phase, layer_index, stage))
    if artifact is None:
        status = "host-fallback"
        mode = "n/a"
        fallback = "synthetic CPU reference"
    else:
        status = artifact.status
        mode = artifact.mode
        fallback = artifact.fallback
    return Gemma3StageRecord(
        phase=phase,
        layer_index=layer_index,
        stage=stage,
        status=status,
        mode=mode,
        fallback=fallback,
        shape=_shape(value),
        checksum=_checksum(value),
        token_base=token_base,
        cache_len_before=cache_len_before,
        cache_len_after=cache_len_after,
        kv_read_start=kv_read_start,
        kv_read_end=kv_read_end,
    )


def _require_manifest_entries(
    config: Gemma3TextConfig,
    manifest: Gemma3RuntimeManifest | None,
    phase: str,
    kernels: Iterable[str],
) -> None:
    if manifest is None:
        return
    manifest.validate_for(config)
    keys = {
        (artifact.phase, artifact.layer_index, artifact.kernel)
        for artifact in manifest.artifacts
    }
    missing = []
    for layer in config.layers:
        for kernel in kernels:
            key = (phase, layer.layer_index, kernel)
            if key not in keys:
                missing.append(f"{phase}:L{layer.layer_index}:{kernel}")
    if missing:
        raise ValueError(f"Gemma3 manifest missing required artifacts: {missing}")


def compile_prefill_kernels(
    config: Gemma3TextConfig | None = None,
    *,
    cache_dir: str = "gemma3_kernel_cache",
) -> Gemma3RuntimeManifest:
    """Prepare the synthetic prefill artifact manifest.

    The real NPU kernel builders are not invoked here yet. This function exists
    so callers can use the same compile-only boundary as Llama32 while the
    Gemma3 substeps are promoted one at a time.
    """

    return prepare_runtime(config or synthetic_text_config(), cache_dir=cache_dir, compile_only=True)


def load_prefill_manifest(
    config: Gemma3TextConfig,
    *,
    cache_dir: str = "gemma3_kernel_cache",
) -> Gemma3RuntimeManifest:
    return load_manifest(cache_dir, config)


def run_prefill_layer(
    x: np.ndarray,
    weights: Gemma3SyntheticWeights,
    cache: Gemma3KVCache,
    layer_cfg: Gemma3LayerConfig,
    *,
    manifest: Gemma3RuntimeManifest | None = None,
    token_base: int = 0,
) -> Gemma3PrefillLayerReport:
    config = weights.config
    _require_manifest_entries(config, manifest, "prefill", ("q4nx", "bf16_mm", "flowqkv"))
    artifacts = _artifact_map(manifest)

    cache_before = cache.cache_len[layer_cfg.layer_index]
    result = prefill_layer_reference(x, weights, cache, layer_cfg, token_base=token_base)
    cache_after = cache.cache_len[layer_cfg.layer_index]
    view_window = layer_cfg.window_len if layer_cfg.attention_kind == "local_swa" else 0
    kv_read_start = max(0, cache_after - view_window) if view_window > 0 else 0

    intermediates = result.intermediates
    stage_values = {
        "rms_norm": intermediates["attn_norm"],
        "q4nx": intermediates["q"],
        "bf16_mm": intermediates["q"],
        "rope": intermediates["q_roped"],
        "qk_norm": intermediates["q_roped"],
        "flowqkv": intermediates["attention"],
        "o_proj": intermediates["o_proj"],
        "residual_add": intermediates["residual_attn"],
        "mlp_activation": intermediates["mlp_activation"],
    }

    stages = [
        _stage_from_artifact(
            artifacts,
            phase="prefill",
            layer_index=layer_cfg.layer_index,
            stage=stage,
            value=stage_values[stage],
            token_base=token_base,
            cache_len_before=cache_before,
            cache_len_after=cache_after,
            kv_read_start=kv_read_start if stage == "flowqkv" else None,
            kv_read_end=cache_after if stage == "flowqkv" else None,
        )
        for stage in PREFILL_STAGE_ORDER
    ]
    return Gemma3PrefillLayerReport(layer=layer_cfg, output=result.output, stages=stages)


def run_text_prefill(
    token_ids: np.ndarray,
    weights: Gemma3SyntheticWeights,
    cache: Gemma3KVCache,
    *,
    manifest: Gemma3RuntimeManifest | None = None,
    token_base: int = 0,
) -> Gemma3PrefillReport:
    x = embedding_reference(token_ids, weights)
    stages: list[Gemma3StageRecord] = []
    for layer_cfg in weights.config.layers:
        layer_report = run_prefill_layer(
            x,
            weights,
            cache,
            layer_cfg,
            manifest=manifest,
            token_base=token_base,
        )
        x = layer_report.output
        stages.extend(layer_report.stages)
    logits = logits_reference(x[-1], weights)
    return Gemma3PrefillReport(output=x, logits=logits, stages=stages)


def run_synthetic_prefill_smoke() -> dict[str, object]:
    config = synthetic_text_config(n_layers=2, local_window_len=4)
    weights = synthetic_weights(config, seed=42)
    cache = Gemma3KVCache.allocate(config)
    manifest = compile_prefill_kernels(config, cache_dir="/tmp/gemma3_prefill_manifest")

    first_tokens = synthetic_token_ids(config, config.q_chunk)
    first = run_text_prefill(first_tokens, weights, cache, manifest=manifest, token_base=0)
    second_tokens = synthetic_token_ids(config, config.q_chunk) + 1
    second = run_text_prefill(
        second_tokens,
        weights,
        cache,
        manifest=manifest,
        token_base=config.q_chunk,
    )
    return {
        "config": config,
        "cache": cache,
        "first": first,
        "second": second,
        "checksum": _checksum(second.logits),
    }


def assert_prefill_report(report: Gemma3PrefillReport, config: Gemma3TextConfig) -> None:
    if report.output.shape != (config.q_chunk, config.emb_dim):
        raise AssertionError(f"unexpected prefill output shape: {report.output.shape}")
    if report.logits.shape != (config.vocab_size,):
        raise AssertionError(f"unexpected logits shape: {report.logits.shape}")
    stages = {(stage.layer_index, stage.stage) for stage in report.stages}
    expected = {
        (layer.layer_index, stage)
        for layer in config.layers
        for stage in PREFILL_STAGE_ORDER
    }
    missing = expected - stages
    if missing:
        raise AssertionError(f"prefill report missing stages: {sorted(missing)}")


def _self_test() -> None:
    smoke = run_synthetic_prefill_smoke()
    config = smoke["config"]
    cache = smoke["cache"]
    assert_prefill_report(smoke["first"], config)
    assert_prefill_report(smoke["second"], config)
    assert cache.cache_len == [2 * config.q_chunk, 2 * config.q_chunk]
    local_flow = [
        stage
        for stage in smoke["second"].stages
        if stage.layer_index == 0 and stage.stage == "flowqkv"
    ][0]
    global_flow = [
        stage
        for stage in smoke["second"].stages
        if stage.layer_index == 1 and stage.stage == "flowqkv"
    ][0]
    assert (local_flow.kv_read_start, local_flow.kv_read_end) == (4, 8)
    assert (global_flow.kv_read_start, global_flow.kv_read_end) == (0, 8)
    print(smoke["second"].format())
    print(f"prefill_orchestration_checksum={smoke['checksum']:.6f}")
    print("GEMMA3_PREFILL_ORCHESTRATION_SELF_TEST: PASS")


if __name__ == "__main__":
    _self_test()
