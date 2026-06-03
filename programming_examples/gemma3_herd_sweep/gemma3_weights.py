# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Synthetic Gemma3 weights for CPU-only model-loop bring-up."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ml_dtypes import bfloat16

from common import random_q4nx_blocks
from gemma3_config import Gemma3TextConfig


@dataclass
class Q4NXProjectionWeights:
    name: str
    in_dim: int
    out_dim: int
    rows: int
    cols: int
    packed: np.ndarray
    scale: np.ndarray
    min_offset: np.ndarray

    @property
    def row_blocks(self) -> int:
        return self.out_dim // self.rows

    @property
    def col_blocks(self) -> int:
        return self.in_dim // self.cols

    def validate(self) -> None:
        if self.in_dim % self.cols != 0:
            raise ValueError(f"{self.name}: in_dim must be divisible by cols")
        if self.out_dim % self.rows != 0:
            raise ValueError(f"{self.name}: out_dim must be divisible by rows")
        expected_packed = (self.row_blocks, self.col_blocks, self.rows * self.cols // 2)
        expected_params = (self.row_blocks, self.col_blocks, self.cols)
        if self.packed.shape != expected_packed:
            raise ValueError(f"{self.name}: packed shape {self.packed.shape} != {expected_packed}")
        if self.scale.shape != expected_params:
            raise ValueError(f"{self.name}: scale shape {self.scale.shape} != {expected_params}")
        if self.min_offset.shape != expected_params:
            raise ValueError(
                f"{self.name}: min_offset shape {self.min_offset.shape} != {expected_params}"
            )


@dataclass
class Gemma3LayerWeights:
    attn_norm: np.ndarray
    q_norm: np.ndarray
    k_norm: np.ndarray
    q_proj: Q4NXProjectionWeights
    k_proj: Q4NXProjectionWeights
    v_proj: Q4NXProjectionWeights
    o_proj: Q4NXProjectionWeights
    ffn_norm: np.ndarray
    gate_proj: Q4NXProjectionWeights
    up_proj: Q4NXProjectionWeights
    down_proj: Q4NXProjectionWeights

    def validate(self, config: Gemma3TextConfig) -> None:
        for vec_name in ("attn_norm", "ffn_norm"):
            vec = getattr(self, vec_name)
            if vec.shape != (config.emb_dim,):
                raise ValueError(f"{vec_name} shape {vec.shape} != {(config.emb_dim,)}")
        for vec_name in ("q_norm", "k_norm"):
            vec = getattr(self, vec_name)
            if vec.shape != (config.head_dim,):
                raise ValueError(f"{vec_name} shape {vec.shape} != {(config.head_dim,)}")
        for projection in (
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.o_proj,
            self.gate_proj,
            self.up_proj,
            self.down_proj,
        ):
            projection.validate()


@dataclass
class Gemma3SyntheticWeights:
    config: Gemma3TextConfig
    embed_table: np.ndarray
    final_norm: np.ndarray
    lm_head: np.ndarray
    layers: list[Gemma3LayerWeights] = field(default_factory=list)

    def validate(self) -> None:
        if self.embed_table.shape != (self.config.vocab_size, self.config.emb_dim):
            raise ValueError("embed_table shape does not match config")
        if self.final_norm.shape != (self.config.emb_dim,):
            raise ValueError("final_norm shape does not match config")
        if self.lm_head.shape != (self.config.vocab_size, self.config.emb_dim):
            raise ValueError("lm_head shape does not match config")
        if len(self.layers) != self.config.n_layers:
            raise ValueError("layer weight count does not match config")
        for layer in self.layers:
            layer.validate(self.config)


def _bf16_uniform(rng: np.random.Generator, shape, low=-0.5, high=0.5) -> np.ndarray:
    return rng.uniform(low, high, size=shape).astype(bfloat16)


def _norm_weight(rng: np.random.Generator, size: int) -> np.ndarray:
    return rng.uniform(0.8, 1.2, size=(size,)).astype(bfloat16)


def synthetic_q4nx_projection(
    name: str,
    in_dim: int,
    out_dim: int,
    config: Gemma3TextConfig,
    *,
    seed: int,
) -> Q4NXProjectionWeights:
    if in_dim % config.q4nx_cols != 0:
        raise ValueError(f"{name}: in_dim must be divisible by Q4NX_COLS")
    if out_dim % config.q4nx_rows != 0:
        raise ValueError(f"{name}: out_dim must be divisible by Q4NX_ROWS")
    packed, scale, min_offset = random_q4nx_blocks(
        out_dim // config.q4nx_rows,
        in_dim // config.q4nx_cols,
        config.q4nx_rows,
        config.q4nx_cols,
        seed=seed,
    )
    projection = Q4NXProjectionWeights(
        name,
        in_dim,
        out_dim,
        config.q4nx_rows,
        config.q4nx_cols,
        packed,
        scale,
        min_offset,
    )
    projection.validate()
    return projection


def synthetic_layer_weights(
    config: Gemma3TextConfig, layer_index: int, rng: np.random.Generator
) -> Gemma3LayerWeights:
    seed_base = 1000 + layer_index * 100
    return Gemma3LayerWeights(
        attn_norm=_norm_weight(rng, config.emb_dim),
        q_norm=_norm_weight(rng, config.head_dim),
        k_norm=_norm_weight(rng, config.head_dim),
        q_proj=synthetic_q4nx_projection(
            "q_proj", config.emb_dim, config.emb_dim, config, seed=seed_base + 1
        ),
        k_proj=synthetic_q4nx_projection(
            "k_proj", config.emb_dim, config.kv_dim, config, seed=seed_base + 2
        ),
        v_proj=synthetic_q4nx_projection(
            "v_proj", config.emb_dim, config.kv_dim, config, seed=seed_base + 3
        ),
        o_proj=synthetic_q4nx_projection(
            "o_proj", config.emb_dim, config.emb_dim, config, seed=seed_base + 4
        ),
        ffn_norm=_norm_weight(rng, config.emb_dim),
        gate_proj=synthetic_q4nx_projection(
            "gate_proj", config.emb_dim, config.hidden_dim, config, seed=seed_base + 5
        ),
        up_proj=synthetic_q4nx_projection(
            "up_proj", config.emb_dim, config.hidden_dim, config, seed=seed_base + 6
        ),
        down_proj=synthetic_q4nx_projection(
            "down_proj", config.hidden_dim, config.emb_dim, config, seed=seed_base + 7
        ),
    )


def synthetic_weights(config: Gemma3TextConfig, seed: int = 0) -> Gemma3SyntheticWeights:
    rng = np.random.default_rng(seed)
    weights = Gemma3SyntheticWeights(
        config=config,
        embed_table=_bf16_uniform(rng, (config.vocab_size, config.emb_dim), -0.25, 0.25),
        final_norm=_norm_weight(rng, config.emb_dim),
        lm_head=_bf16_uniform(rng, (config.vocab_size, config.emb_dim), -0.2, 0.2),
        layers=[synthetic_layer_weights(config, i, rng) for i in range(config.n_layers)],
    )
    weights.validate()
    return weights
