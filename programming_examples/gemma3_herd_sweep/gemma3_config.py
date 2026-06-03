# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Synthetic Gemma3 model-loop metadata.

This module intentionally imports no AIR/XRT modules. It mirrors the Llama32
example's config-first structure while keeping Gemma-specific kernel modes and
local/global attention metadata explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from common import Q4NX_COLS, Q4NX_ROWS, parse_herd_shape, resolve_output_mode


ATTENTION_KINDS = ("local_swa", "global_full", "vision_nca")
PREFILL_KERNELS = ("q4nx", "bf16_mm", "flowqkv")
DECODE_KERNELS = ("fused_dqp", "flowkv")
NONLINEAR_STAGES = ("rms_norm", "rope", "qk_norm", "mlp_activation", "residual_add")


@dataclass(frozen=True)
class Gemma3LayerConfig:
    layer_index: int
    attention_kind: str
    window_len: int

    def __post_init__(self):
        if self.attention_kind not in ATTENTION_KINDS:
            supported = ", ".join(ATTENTION_KINDS)
            raise ValueError(f"attention kind must be one of: {supported}")
        if self.window_len < 0:
            raise ValueError("window length must be non-negative")
        if self.attention_kind == "global_full" and self.window_len != 0:
            raise ValueError("global_full layers must use window_len=0")

    @property
    def causal(self) -> bool:
        return self.attention_kind in ("local_swa", "global_full")


@dataclass(frozen=True)
class Gemma3KernelStep:
    phase: str
    layer_index: int
    kernel: str
    mode: str
    status: str
    fallback: str | None = None

    def format(self) -> str:
        fallback = f" fallback={self.fallback}" if self.fallback else ""
        return (
            f"{self.phase}:L{self.layer_index}:{self.kernel} "
            f"mode={self.mode} status={self.status}{fallback}"
        )


@dataclass(frozen=True)
class Gemma3TextConfig:
    model_variant: str = "synthetic-gemma3-text"
    n_layers: int = 2
    vocab_size: int = 256
    emb_dim: int = Q4NX_COLS
    hidden_dim: int = 2 * Q4NX_COLS
    n_heads: int = 4
    n_kv_heads: int = 2
    head_dim: int = 64
    q_chunk: int = 4
    kv_len: int = 32
    kv_chunk: int = 32
    local_window_len: int = 16
    q4nx_rows: int = Q4NX_ROWS
    q4nx_cols: int = Q4NX_COLS
    rope_base: float = 10000.0
    herd_shape: str = "2x4"
    q4nx_output_mode: str = "auto"
    fused_dqp_output_mode: str = "auto"
    flowqkv_output_mode: str = "auto"
    flowkv_output_mode: str = "auto"
    flowqkv_schedule_mode: str = "smoke"
    flowkv_schedule_mode: str = "smoke"
    fused_dqp_schedule_mode: str = "smoke"
    host_fallbacks: tuple[str, ...] = NONLINEAR_STAGES
    layers: tuple[Gemma3LayerConfig, ...] = field(default_factory=tuple)

    def __post_init__(self):
        parse_herd_shape(self.herd_shape)
        if self.emb_dim != self.n_heads * self.head_dim:
            raise ValueError("emb_dim must equal n_heads * head_dim")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.emb_dim % self.q4nx_cols != 0:
            raise ValueError("emb_dim must be divisible by Q4NX_COLS")
        if self.hidden_dim % self.q4nx_cols != 0:
            raise ValueError("hidden_dim must be divisible by Q4NX_COLS")
        if self.emb_dim % self.q4nx_rows != 0:
            raise ValueError("emb_dim must be divisible by Q4NX_ROWS")
        if self.hidden_dim % self.q4nx_rows != 0:
            raise ValueError("hidden_dim must be divisible by Q4NX_ROWS")
        if self.kv_len <= 0 or self.kv_len % self.kv_chunk != 0:
            raise ValueError("kv_len must be divisible by a positive kv_chunk")
        if self.q_chunk <= 0 or self.q_chunk > self.kv_len:
            raise ValueError("q_chunk must be positive and no larger than kv_len")

        layers = self.layers or default_text_layers(self.n_layers, self.local_window_len)
        if len(layers) != self.n_layers:
            raise ValueError("layer metadata length must match n_layers")
        object.__setattr__(self, "layers", tuple(layers))

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def heads_per_kv(self) -> int:
        return self.n_heads // self.n_kv_heads

    def resolved_output_modes(self) -> dict[str, str]:
        herd_rows, herd_cols = parse_herd_shape(self.herd_shape)
        return {
            "q4nx": resolve_output_mode(
                self.q4nx_output_mode, herd_rows, herd_cols, "q4nx"
            ),
            "fused_dqp": resolve_output_mode(
                self.fused_dqp_output_mode, herd_rows, herd_cols, "fused_dqp"
            ),
            "flowqkv": resolve_output_mode(
                self.flowqkv_output_mode, herd_rows, herd_cols, "flowqkv"
            ),
            "flowkv": resolve_output_mode(
                self.flowkv_output_mode, herd_rows, herd_cols, "flowkv"
            ),
        }

    def kernel_sequence(self) -> tuple[Gemma3KernelStep, ...]:
        modes = self.resolved_output_modes()
        steps: list[Gemma3KernelStep] = []
        for layer in self.layers:
            steps.extend(
                [
                    Gemma3KernelStep("prefill", layer.layer_index, "q4nx", modes["q4nx"], "npu"),
                    Gemma3KernelStep("prefill", layer.layer_index, "bf16_mm", "n/a", "npu"),
                    Gemma3KernelStep("prefill", layer.layer_index, "rms_norm", "n/a", "host-fallback", "weighted_rms_norm candidate"),
                    Gemma3KernelStep("prefill", layer.layer_index, "rope", "n/a", "host-fallback", "rope candidate"),
                    Gemma3KernelStep("prefill", layer.layer_index, "qk_norm", "n/a", "host-fallback", "Gemma-specific candidate"),
                    Gemma3KernelStep("prefill", layer.layer_index, "flowqkv", modes["flowqkv"], "npu"),
                    Gemma3KernelStep("prefill", layer.layer_index, "residual_add", "n/a", "host-fallback", "Llama multi-launch pattern candidate"),
                    Gemma3KernelStep("prefill", layer.layer_index, "mlp_activation", "n/a", "host-fallback", "gemma3_dataflow_kernels/geglu candidate"),
                    Gemma3KernelStep("decode", layer.layer_index, "fused_dqp", modes["fused_dqp"], "npu"),
                    Gemma3KernelStep("decode", layer.layer_index, "rms_norm", "n/a", "host-fallback", "weighted_rms_norm candidate"),
                    Gemma3KernelStep("decode", layer.layer_index, "rope", "n/a", "host-fallback", "rope candidate"),
                    Gemma3KernelStep("decode", layer.layer_index, "qk_norm", "n/a", "host-fallback", "Gemma-specific candidate"),
                    Gemma3KernelStep("decode", layer.layer_index, "flowkv", modes["flowkv"], "npu"),
                    Gemma3KernelStep("decode", layer.layer_index, "residual_add", "n/a", "host-fallback", "Llama multi-launch pattern candidate"),
                    Gemma3KernelStep("decode", layer.layer_index, "mlp_activation", "n/a", "host-fallback", "gemma3_dataflow_kernels/geglu candidate"),
                ]
            )
        return tuple(steps)


def default_text_layers(n_layers: int, local_window_len: int) -> tuple[Gemma3LayerConfig, ...]:
    if n_layers <= 0:
        raise ValueError("n_layers must be positive")
    layers = []
    for idx in range(n_layers):
        if n_layers == 1:
            kind = "local_swa"
        elif n_layers == 2:
            kind = "local_swa" if idx == 0 else "global_full"
        else:
            kind = "global_full" if idx % 6 == 5 else "local_swa"
        window_len = 0 if kind == "global_full" else local_window_len
        layers.append(Gemma3LayerConfig(idx, kind, window_len))
    return tuple(layers)


def synthetic_text_config(**overrides) -> Gemma3TextConfig:
    return Gemma3TextConfig(**overrides)


def describe_kernel_sequence(config: Gemma3TextConfig) -> str:
    return "\n".join(step.format() for step in config.kernel_sequence())
