# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Synthetic Gemma3 one-token decode orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gemma3_config import Gemma3LayerConfig, Gemma3TextConfig, synthetic_text_config
from gemma3_prefill import (
    Gemma3StageRecord,
    _artifact_map,
    _checksum,
    _require_manifest_entries,
    _shape,
    run_text_prefill,
)
from gemma3_reference import (
    Gemma3KVCache,
    decode_layer_reference,
    embedding_reference,
    logits_reference,
    synthetic_token_ids,
)
from gemma3_runtime import Gemma3RuntimeManifest, prepare_runtime
from gemma3_weights import Gemma3SyntheticWeights, synthetic_weights


DECODE_STAGE_ORDER = (
    "rms_norm",
    "fused_dqp",
    "rope",
    "qk_norm",
    "flowkv",
    "o_proj",
    "residual_add",
    "mlp_activation",
)


@dataclass
class Gemma3DecodeLayerReport:
    layer: Gemma3LayerConfig
    output: np.ndarray
    stages: list[Gemma3StageRecord] = field(default_factory=list)


@dataclass
class Gemma3DecodeReport:
    output: np.ndarray
    logits: np.ndarray
    stages: list[Gemma3StageRecord]

    def format(self) -> str:
        return "\n".join(stage.format() for stage in self.stages)


@dataclass
class Gemma3SyntheticLoopReport:
    prefill_checksums: tuple[float, ...]
    decode_checksums: tuple[float, ...]
    prefill_stage_count: int
    decode_stage_count: int
    cache_lengths: tuple[int, ...]
    last_decode: Gemma3DecodeReport

    def format(self) -> str:
        prefill = ",".join(f"{value:.6f}" for value in self.prefill_checksums)
        decode = ",".join(f"{value:.6f}" for value in self.decode_checksums)
        cache = ",".join(str(value) for value in self.cache_lengths)
        return (
            f"prefill_checksums={prefill}\n"
            f"decode_checksums={decode}\n"
            f"prefill_stage_count={self.prefill_stage_count}\n"
            f"decode_stage_count={self.decode_stage_count}\n"
            f"cache_lengths={cache}\n"
            f"{self.last_decode.format()}"
        )


def _decode_stage_from_artifact(
    artifacts,
    *,
    layer_index: int,
    stage: str,
    value: np.ndarray,
    current_pos: int,
    cache_len_before: int,
    cache_len_after: int,
    kv_read_start: int | None = None,
    kv_read_end: int | None = None,
) -> Gemma3StageRecord:
    artifact = artifacts.get(("decode", layer_index, stage))
    if artifact is None:
        status = "host-fallback"
        mode = "n/a"
        fallback = "synthetic CPU reference"
    else:
        status = artifact.status
        mode = artifact.mode
        fallback = artifact.fallback
    return Gemma3StageRecord(
        phase="decode",
        layer_index=layer_index,
        stage=stage,
        status=status,
        mode=mode,
        fallback=fallback,
        shape=_shape(value),
        checksum=_checksum(value),
        token_base=current_pos,
        cache_len_before=cache_len_before,
        cache_len_after=cache_len_after,
        kv_read_start=kv_read_start,
        kv_read_end=kv_read_end,
    )


def compile_decode_kernels(
    config: Gemma3TextConfig | None = None,
    *,
    cache_dir: str = "gemma3_kernel_cache",
) -> Gemma3RuntimeManifest:
    return prepare_runtime(config or synthetic_text_config(), cache_dir=cache_dir, compile_only=True)


def run_decode_layer(
    x: np.ndarray,
    weights: Gemma3SyntheticWeights,
    cache: Gemma3KVCache,
    layer_cfg: Gemma3LayerConfig,
    *,
    manifest: Gemma3RuntimeManifest | None = None,
    current_pos: int,
) -> Gemma3DecodeLayerReport:
    config = weights.config
    _require_manifest_entries(config, manifest, "decode", ("fused_dqp", "flowkv"))
    artifacts = _artifact_map(manifest)

    cache_before = cache.cache_len[layer_cfg.layer_index]
    result = decode_layer_reference(
        x,
        weights,
        cache,
        layer_cfg,
        current_pos=current_pos,
    )
    cache_after = cache.cache_len[layer_cfg.layer_index]
    view_window = layer_cfg.window_len if layer_cfg.attention_kind == "local_swa" else 0
    kv_read_start = max(0, cache_after - view_window) if view_window > 0 else 0

    intermediates = result.intermediates
    stage_values = {
        "rms_norm": intermediates["attn_norm"],
        "fused_dqp": intermediates["q"],
        "rope": intermediates["q_roped"],
        "qk_norm": intermediates["q_roped"],
        "flowkv": intermediates["attention"],
        "o_proj": intermediates["o_proj"],
        "residual_add": intermediates["residual_attn"],
        "mlp_activation": intermediates["mlp_activation"],
    }
    stages = [
        _decode_stage_from_artifact(
            artifacts,
            layer_index=layer_cfg.layer_index,
            stage=stage,
            value=stage_values[stage],
            current_pos=current_pos,
            cache_len_before=cache_before,
            cache_len_after=cache_after,
            kv_read_start=kv_read_start if stage == "flowkv" else None,
            kv_read_end=cache_after if stage == "flowkv" else None,
        )
        for stage in DECODE_STAGE_ORDER
    ]
    return Gemma3DecodeLayerReport(layer=layer_cfg, output=result.output, stages=stages)


def run_text_decode(
    token_id: int,
    weights: Gemma3SyntheticWeights,
    cache: Gemma3KVCache,
    *,
    manifest: Gemma3RuntimeManifest | None = None,
    current_pos: int,
) -> Gemma3DecodeReport:
    x = embedding_reference(np.asarray([token_id], dtype=np.int64), weights)[0]
    stages: list[Gemma3StageRecord] = []
    for layer_cfg in weights.config.layers:
        layer_report = run_decode_layer(
            x,
            weights,
            cache,
            layer_cfg,
            manifest=manifest,
            current_pos=current_pos,
        )
        x = layer_report.output
        stages.extend(layer_report.stages)
    logits = logits_reference(x, weights)
    return Gemma3DecodeReport(output=x, logits=logits, stages=stages)


def run_synthetic_text_loop(
    config: Gemma3TextConfig | None = None,
    *,
    manifest: Gemma3RuntimeManifest | None = None,
    prefill_chunks: int = 2,
    decode_tokens: int = 2,
    seed: int = 42,
) -> Gemma3SyntheticLoopReport:
    config = config or synthetic_text_config(n_layers=2, local_window_len=4)
    weights = synthetic_weights(config, seed=seed)
    cache = Gemma3KVCache.allocate(config)
    if manifest is None:
        manifest = compile_decode_kernels(config, cache_dir="/tmp/gemma3_decode_manifest")

    prefill_reports = []
    for chunk_index in range(prefill_chunks):
        base = chunk_index * config.q_chunk
        token_ids = synthetic_token_ids(config, config.q_chunk) + chunk_index
        prefill_reports.append(
            run_text_prefill(token_ids, weights, cache, manifest=manifest, token_base=base)
        )

    decode_reports = []
    current_pos = prefill_chunks * config.q_chunk
    for token_offset in range(decode_tokens):
        token_id = int((17 + token_offset * 13) % config.vocab_size)
        decode_reports.append(
            run_text_decode(
                token_id,
                weights,
                cache,
                manifest=manifest,
                current_pos=current_pos + token_offset,
            )
        )

    return Gemma3SyntheticLoopReport(
        prefill_checksums=tuple(_checksum(report.logits) for report in prefill_reports),
        decode_checksums=tuple(_checksum(report.logits) for report in decode_reports),
        prefill_stage_count=sum(len(report.stages) for report in prefill_reports),
        decode_stage_count=sum(len(report.stages) for report in decode_reports),
        cache_lengths=tuple(cache.cache_len),
        last_decode=decode_reports[-1],
    )


def assert_decode_report(report: Gemma3DecodeReport, config: Gemma3TextConfig) -> None:
    if report.output.shape != (config.emb_dim,):
        raise AssertionError(f"unexpected decode output shape: {report.output.shape}")
    if report.logits.shape != (config.vocab_size,):
        raise AssertionError(f"unexpected decode logits shape: {report.logits.shape}")
    stages = {(stage.layer_index, stage.stage) for stage in report.stages}
    expected = {
        (layer.layer_index, stage)
        for layer in config.layers
        for stage in DECODE_STAGE_ORDER
    }
    missing = expected - stages
    if missing:
        raise AssertionError(f"decode report missing stages: {sorted(missing)}")


def _self_test() -> None:
    config = synthetic_text_config(n_layers=2, local_window_len=4)
    report = run_synthetic_text_loop(config, prefill_chunks=2, decode_tokens=2)
    assert report.cache_lengths == (10, 10)
    assert report.prefill_stage_count == 2 * config.n_layers * 9
    assert report.decode_stage_count == 2 * config.n_layers * 8
    assert_decode_report(report.last_decode, config)
    local_flow = [
        stage
        for stage in report.last_decode.stages
        if stage.layer_index == 0 and stage.stage == "flowkv"
    ][0]
    global_flow = [
        stage
        for stage in report.last_decode.stages
        if stage.layer_index == 1 and stage.stage == "flowkv"
    ][0]
    assert (local_flow.kv_read_start, local_flow.kv_read_end) == (6, 10)
    assert (global_flow.kv_read_start, global_flow.kv_read_end) == (0, 10)
    print(report.format())
    print("GEMMA3_DECODE_ORCHESTRATION_SELF_TEST: PASS")


if __name__ == "__main__":
    _self_test()
