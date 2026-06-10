# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unified synthetic Gemma3 model-loop session."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from gemma3.core.config import Gemma3TextConfig, synthetic_text_config
from gemma3.model.decode import Gemma3DecodeReport, run_text_decode
from gemma3.model.prefill import Gemma3PrefillReport, Gemma3StageRecord, run_text_prefill
from gemma3.core.reference import Gemma3KVCache, synthetic_token_ids
from gemma3.core.runtime import Gemma3RuntimeManifest, format_manifest, prepare_runtime
from gemma3.core.weights import Gemma3SyntheticWeights, synthetic_weights


@dataclass(frozen=True)
class Gemma3LoopEvent:
    phase: str
    token_base: int
    token_count: int
    checksum: float
    stage_count: int
    cache_lengths: tuple[int, ...]
    elapsed_ms: float = 0.0

    def format(self, *, profile: bool = False) -> str:
        cache = ",".join(str(value) for value in self.cache_lengths)
        line = (
            f"loop:{self.phase} token_base={self.token_base} "
            f"tokens={self.token_count} checksum={self.checksum:.6f} "
            f"stages={self.stage_count} cache_lengths={cache}"
        )
        if profile:
            line += f" elapsed_ms={self.elapsed_ms:.3f}"
        return line


@dataclass
class Gemma3LoopResult:
    events: list[Gemma3LoopEvent]
    prefill_reports: list[Gemma3PrefillReport]
    decode_reports: list[Gemma3DecodeReport]
    final_cache_lengths: tuple[int, ...]
    final_decode_checksum: float | None
    failure: str | None = None

    @property
    def all_stages(self) -> list[Gemma3StageRecord]:
        stages: list[Gemma3StageRecord] = []
        for report in self.prefill_reports:
            stages.extend(report.stages)
        for report in self.decode_reports:
            stages.extend(report.stages)
        return stages

    def format(self, *, profile: bool = False, include_stages: bool = False) -> str:
        lines = [event.format(profile=profile) for event in self.events]
        if include_stages:
            lines.extend(stage.format() for stage in self.all_stages)
        if self.final_decode_checksum is not None:
            lines.append(f"final_decode_checksum={self.final_decode_checksum:.6f}")
        lines.append(
            "final_cache_lengths="
            + ",".join(str(value) for value in self.final_cache_lengths)
        )
        if self.failure:
            lines.append(f"failure={self.failure}")
        return "\n".join(lines)


class Gemma3LoopError(RuntimeError):
    pass


@dataclass
class Gemma3SyntheticSession:
    config: Gemma3TextConfig = field(default_factory=synthetic_text_config)
    weights: Gemma3SyntheticWeights | None = None
    manifest: Gemma3RuntimeManifest | None = None
    seed: int = 42
    profile: bool = False

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = synthetic_weights(self.config, seed=self.seed)
        if self.weights.config != self.config:
            raise ValueError("Gemma3 session weights/config mismatch")
        self.cache = Gemma3KVCache.allocate(self.config)

    def prepare_compile_only(self, cache_dir) -> Gemma3RuntimeManifest:
        self.manifest = prepare_runtime(self.config, cache_dir=cache_dir, compile_only=True)
        return self.manifest

    def prepare_run_only(self, cache_dir) -> Gemma3RuntimeManifest:
        self.manifest = prepare_runtime(self.config, cache_dir=cache_dir, run_only=True)
        return self.manifest

    def manifest_summary(self) -> str:
        if self.manifest is None:
            return "manifest=not_loaded"
        return format_manifest(self.manifest)

    def _record_prefill(self, token_ids: np.ndarray, token_base: int) -> tuple[Gemma3PrefillReport, Gemma3LoopEvent]:
        start = perf_counter()
        report = run_text_prefill(
            token_ids,
            self.weights,
            self.cache,
            manifest=self.manifest,
            token_base=token_base,
        )
        elapsed_ms = (perf_counter() - start) * 1000.0
        event = Gemma3LoopEvent(
            phase="prefill",
            token_base=token_base,
            token_count=int(token_ids.shape[0]),
            checksum=float(np.sum(report.logits.astype(np.float32))),
            stage_count=len(report.stages),
            cache_lengths=tuple(self.cache.cache_len),
            elapsed_ms=elapsed_ms,
        )
        return report, event

    def _record_decode(self, token_id: int, current_pos: int) -> tuple[Gemma3DecodeReport, Gemma3LoopEvent]:
        start = perf_counter()
        report = run_text_decode(
            token_id,
            self.weights,
            self.cache,
            manifest=self.manifest,
            current_pos=current_pos,
        )
        elapsed_ms = (perf_counter() - start) * 1000.0
        event = Gemma3LoopEvent(
            phase="decode",
            token_base=current_pos,
            token_count=1,
            checksum=float(np.sum(report.logits.astype(np.float32))),
            stage_count=len(report.stages),
            cache_lengths=tuple(self.cache.cache_len),
            elapsed_ms=elapsed_ms,
        )
        return report, event

    def run(
        self,
        *,
        prefill_chunks: int = 2,
        decode_tokens: int = 2,
        include_stages: bool = False,
    ) -> Gemma3LoopResult:
        if prefill_chunks <= 0:
            raise ValueError("prefill_chunks must be positive")
        if decode_tokens <= 0:
            raise ValueError("decode_tokens must be positive")
        if prefill_chunks * self.config.q_chunk + decode_tokens > self.config.kv_len:
            raise ValueError("synthetic loop exceeds configured kv_len")

        events: list[Gemma3LoopEvent] = []
        prefill_reports: list[Gemma3PrefillReport] = []
        decode_reports: list[Gemma3DecodeReport] = []
        active_phase = "startup"
        active_token_base = 0
        try:
            for chunk_index in range(prefill_chunks):
                token_base = chunk_index * self.config.q_chunk
                active_phase = "prefill"
                active_token_base = token_base
                token_ids = synthetic_token_ids(self.config, self.config.q_chunk) + chunk_index
                report, event = self._record_prefill(token_ids, token_base)
                prefill_reports.append(report)
                events.append(event)

            current_pos = prefill_chunks * self.config.q_chunk
            for token_offset in range(decode_tokens):
                active_phase = "decode"
                active_token_base = current_pos + token_offset
                token_id = int((17 + token_offset * 13) % self.config.vocab_size)
                report, event = self._record_decode(token_id, active_token_base)
                decode_reports.append(report)
                events.append(event)
        except Exception as exc:
            modes = self.config.resolved_output_modes()
            failure = (
                f"phase={active_phase} token_base={active_token_base} "
                f"layers={self.config.n_layers} modes={modes} "
                f"error={type(exc).__name__}: {exc}"
            )
            raise Gemma3LoopError(failure) from exc

        final_decode_checksum = None
        if decode_reports:
            final_decode_checksum = float(np.sum(decode_reports[-1].logits.astype(np.float32)))
        result = Gemma3LoopResult(
            events=events,
            prefill_reports=prefill_reports,
            decode_reports=decode_reports,
            final_cache_lengths=tuple(self.cache.cache_len),
            final_decode_checksum=final_decode_checksum,
        )
        if include_stages:
            _assert_loop_stage_metadata(result, self.config)
        return result


def _assert_loop_stage_metadata(result: Gemma3LoopResult, config: Gemma3TextConfig) -> None:
    if not result.events:
        raise AssertionError("model loop produced no events")
    expected_cache_len = result.events[-1].token_base + result.events[-1].token_count
    expected_cache_lengths = tuple([expected_cache_len] * config.n_layers)
    if result.final_cache_lengths != expected_cache_lengths:
        raise AssertionError(
            "unexpected final cache lengths: "
            f"{result.final_cache_lengths}, expected {expected_cache_lengths}"
        )
    local_flow = [
        stage
        for stage in result.all_stages
        if stage.layer_index == 0 and stage.stage in ("flowqkv", "flowkv")
    ]
    global_flow = [
        stage
        for stage in result.all_stages
        if stage.layer_index == 1 and stage.stage in ("flowqkv", "flowkv")
    ]
    if config.n_layers >= 2 and local_flow and global_flow:
        final_cache_len = result.final_cache_lengths[0]
        if config.local_window_len < final_cache_len and not any(
            stage.kv_read_start is not None and stage.kv_read_start > 0
            for stage in local_flow
        ):
            raise AssertionError("local layer never recorded a clamped KV read")
        if not any(
            stage.kv_read_start == 0 and stage.kv_read_end is not None
            for stage in global_flow
        ):
            raise AssertionError("global layer never recorded a full KV read")


def run_synthetic_session_smoke() -> Gemma3LoopResult:
    config = synthetic_text_config(n_layers=2, local_window_len=4)
    session = Gemma3SyntheticSession(config=config, profile=True)
    session.prepare_compile_only("/tmp/gemma3_session_manifest")
    result = session.run(prefill_chunks=2, decode_tokens=2, include_stages=True)
    if result.final_cache_lengths != (10, 10):
        raise AssertionError(f"unexpected cache lengths: {result.final_cache_lengths}")
    return result


def _self_test() -> None:
    result = run_synthetic_session_smoke()
    print(result.format(profile=True, include_stages=False))
    print("GEMMA3_MODEL_LOOP_SESSION_SELF_TEST: PASS")


if __name__ == "__main__":
    _self_test()
