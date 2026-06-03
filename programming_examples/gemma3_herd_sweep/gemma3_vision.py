# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Disabled Gemma3 vision-path contract for text-only model-loop bring-up."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ml_dtypes import bfloat16

from gemma3_config import Gemma3TextConfig, synthetic_text_config
from gemma3_model_loop import Gemma3SyntheticSession


@dataclass(frozen=True)
class Gemma3VisionConfig:
    enabled: bool = False
    image_token_count: int = 0
    attention_kind: str = "vision_nca"

    def __post_init__(self) -> None:
        if self.image_token_count < 0:
            raise ValueError("image_token_count must be non-negative")
        if self.attention_kind != "vision_nca":
            raise ValueError("Gemma3 vision placeholder only models vision_nca")
        if not self.enabled and self.image_token_count != 0:
            raise ValueError("disabled vision path must not contribute image tokens")


@dataclass(frozen=True)
class Gemma3VisionResult:
    enabled: bool
    context_tokens: np.ndarray
    status: str
    fallback: str

    @property
    def checksum(self) -> float:
        return float(np.sum(self.context_tokens.astype(np.float32)))

    def format(self) -> str:
        return (
            f"vision enabled={self.enabled} status={self.status} "
            f"tokens={self.context_tokens.shape[0]} checksum={self.checksum:.6f} "
            f"fallback={self.fallback}"
        )


def run_vision_prefill_or_disabled(
    config: Gemma3TextConfig,
    vision_config: Gemma3VisionConfig | None = None,
) -> Gemma3VisionResult:
    vision_config = vision_config or Gemma3VisionConfig()
    if vision_config.enabled:
        raise NotImplementedError(
            "Gemma3 vision prefill is later scope; keep text-only loop as the baseline"
        )
    context = np.zeros((0, config.emb_dim), dtype=bfloat16)
    return Gemma3VisionResult(
        enabled=False,
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


def _self_test() -> None:
    config = synthetic_text_config(n_layers=2, local_window_len=4)
    vision = run_vision_prefill_or_disabled(config)
    baseline_checksum, disabled_checksum = assert_text_loop_unchanged_with_vision_disabled()
    print(vision.format())
    print(f"text_baseline_checksum={baseline_checksum:.6f}")
    print(f"text_with_disabled_vision_checksum={disabled_checksum:.6f}")
    print("GEMMA3_VISION_DISABLED_SELF_TEST: PASS")


if __name__ == "__main__":
    _self_test()
