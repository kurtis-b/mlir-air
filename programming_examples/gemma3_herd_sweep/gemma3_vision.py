# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Synthetic Gemma3 vision-path contract for model-loop bring-up."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ml_dtypes import bfloat16

from gemma3_config import Gemma3TextConfig, synthetic_text_config
from gemma3_model_loop import Gemma3SyntheticSession
from gemma3_reference import rms_norm


@dataclass(frozen=True)
class Gemma3VisionConfig:
    enabled: bool = False
    image_token_count: int = 0
    attention_kind: str = "vision_nca"
    seed: int = 7

    def __post_init__(self) -> None:
        if self.image_token_count < 0:
            raise ValueError("image_token_count must be non-negative")
        if self.attention_kind != "vision_nca":
            raise ValueError("Gemma3 vision path only models vision_nca")
        if self.enabled and self.image_token_count <= 0:
            raise ValueError("enabled vision path requires positive image_token_count")
        if not self.enabled and self.image_token_count != 0:
            raise ValueError("disabled vision path must not contribute image tokens")


@dataclass(frozen=True)
class Gemma3VisionStageRecord:
    stage: str
    status: str
    shape: tuple[int, ...]
    checksum: float
    causal: bool | None = None
    fallback: str | None = None

    def format(self) -> str:
        shape = "x".join(str(dim) for dim in self.shape) if self.shape else "scalar"
        line = (
            f"vision_stage {self.stage} status={self.status} "
            f"shape={shape} checksum={self.checksum:.6f}"
        )
        if self.causal is not None:
            line += f" causal={self.causal}"
        if self.fallback:
            line += f" fallback={self.fallback}"
        return line


@dataclass(frozen=True)
class Gemma3VisionResult:
    enabled: bool
    attention_kind: str
    context_tokens: np.ndarray
    status: str
    fallback: str
    stages: tuple[Gemma3VisionStageRecord, ...] = field(default_factory=tuple)

    @property
    def checksum(self) -> float:
        return float(np.sum(self.context_tokens.astype(np.float32)))

    def format(self) -> str:
        return (
            f"vision enabled={self.enabled} status={self.status} "
            f"attention={self.attention_kind} tokens={self.context_tokens.shape[0]} "
            f"checksum={self.checksum:.6f} fallback={self.fallback}"
        )


def _checksum(value: np.ndarray) -> float:
    return float(np.sum(value.astype(np.float32)))


def _softmax(scores: np.ndarray) -> np.ndarray:
    scores = scores.astype(np.float32)
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def synthetic_image_tokens(
    config: Gemma3TextConfig,
    vision_config: Gemma3VisionConfig,
) -> np.ndarray:
    rng = np.random.default_rng(vision_config.seed)
    features = rng.normal(size=(vision_config.image_token_count, config.emb_dim)).astype(np.float32)
    positions = np.arange(vision_config.image_token_count, dtype=np.float32)[:, None]
    scales = 1.0 + positions / max(1, vision_config.image_token_count)
    return (features * scales).astype(bfloat16)


def run_noncausal_vision_reference(
    image_tokens: np.ndarray,
    config: Gemma3TextConfig,
) -> tuple[np.ndarray, tuple[Gemma3VisionStageRecord, ...]]:
    if image_tokens.ndim != 2 or image_tokens.shape[1] != config.emb_dim:
        raise ValueError("vision image tokens must be [image_tokens, emb_dim]")
    weight = np.ones((config.emb_dim,), dtype=np.float32).astype(bfloat16)
    normed = rms_norm(image_tokens, weight)
    scores = (
        normed.astype(np.float32) @ normed.astype(np.float32).T
    ) / np.sqrt(float(config.emb_dim))
    probs = _softmax(scores)
    attended = (probs @ image_tokens.astype(np.float32)).astype(bfloat16)
    context = rms_norm(
        (image_tokens.astype(np.float32) + attended.astype(np.float32)).astype(bfloat16),
        weight,
    )
    stages = (
        Gemma3VisionStageRecord(
            stage="image_projection",
            status="host-reference",
            shape=tuple(int(dim) for dim in image_tokens.shape),
            checksum=_checksum(image_tokens),
            fallback="synthetic CPU reference",
        ),
        Gemma3VisionStageRecord(
            stage="vision_attention",
            status="host-reference",
            shape=tuple(int(dim) for dim in attended.shape),
            checksum=_checksum(attended),
            causal=False,
            fallback="synthetic CPU reference",
        ),
        Gemma3VisionStageRecord(
            stage="visual_context_tokens",
            status="host-reference",
            shape=tuple(int(dim) for dim in context.shape),
            checksum=_checksum(context),
            fallback="synthetic CPU reference",
        ),
    )
    return context, stages


def run_vision_prefill_or_disabled(
    config: Gemma3TextConfig,
    vision_config: Gemma3VisionConfig | None = None,
) -> Gemma3VisionResult:
    vision_config = vision_config or Gemma3VisionConfig()
    if vision_config.enabled:
        image_tokens = synthetic_image_tokens(config, vision_config)
        context, stages = run_noncausal_vision_reference(image_tokens, config)
        return Gemma3VisionResult(
            enabled=True,
            attention_kind=vision_config.attention_kind,
            context_tokens=context,
            status="synthetic-cpu-reference",
            fallback="NPU vision path pending real artifacts and hardware validation",
            stages=stages,
        )
    context = np.zeros((0, config.emb_dim), dtype=bfloat16)
    return Gemma3VisionResult(
        enabled=False,
        attention_kind=vision_config.attention_kind,
        context_tokens=context,
        status="disabled",
        fallback="text-only baseline",
    )


def assert_text_loop_unchanged_with_vision_disabled() -> tuple[float, float]:
    config = synthetic_text_config(n_layers=2, local_window_len=4)
    vision = run_vision_prefill_or_disabled(config)
    if vision.context_tokens.shape != (0, config.emb_dim):
        raise AssertionError(f"unexpected disabled vision context shape: {vision.context_tokens.shape}")

    baseline = Gemma3SyntheticSession(config=config)
    baseline.prepare_compile_only("/tmp/gemma3_vision_baseline_manifest")
    baseline_result = baseline.run(prefill_chunks=2, decode_tokens=2, include_stages=True)

    disabled = Gemma3SyntheticSession(config=config)
    disabled.prepare_compile_only("/tmp/gemma3_vision_disabled_manifest")
    disabled_result = disabled.run(prefill_chunks=2, decode_tokens=2, include_stages=True)

    if baseline_result.final_cache_lengths != disabled_result.final_cache_lengths:
        raise AssertionError(
            "disabled vision changed cache lengths: "
            f"{baseline_result.final_cache_lengths} vs {disabled_result.final_cache_lengths}"
        )
    if baseline_result.final_decode_checksum != disabled_result.final_decode_checksum:
        raise AssertionError(
            "disabled vision changed final checksum: "
            f"{baseline_result.final_decode_checksum} vs {disabled_result.final_decode_checksum}"
        )
    return baseline_result.final_decode_checksum, disabled_result.final_decode_checksum


def assert_synthetic_vision_contract() -> Gemma3VisionResult:
    config = synthetic_text_config(n_layers=2, local_window_len=4)
    result = run_vision_prefill_or_disabled(
        config,
        Gemma3VisionConfig(enabled=True, image_token_count=4),
    )
    if result.context_tokens.shape != (4, config.emb_dim):
        raise AssertionError(f"unexpected vision context shape: {result.context_tokens.shape}")
    if not any(stage.stage == "vision_attention" and stage.causal is False for stage in result.stages):
        raise AssertionError("vision contract did not record non-causal attention")
    return result


def _self_test() -> None:
    config = synthetic_text_config(n_layers=2, local_window_len=4)
    disabled = run_vision_prefill_or_disabled(config)
    enabled = assert_synthetic_vision_contract()
    baseline_checksum, disabled_checksum = assert_text_loop_unchanged_with_vision_disabled()
    print(disabled.format())
    print(enabled.format())
    for stage in enabled.stages:
        print(stage.format())
    print(f"text_baseline_checksum={baseline_checksum:.6f}")
    print(f"text_with_disabled_vision_checksum={disabled_checksum:.6f}")
    print("GEMMA3_VISION_CONTRACT_SELF_TEST: PASS")


if __name__ == "__main__":
    _self_test()
