#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# fmt: off

"""Build, inspect, and optionally run the shared int8 GEMM benchmark."""

from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

M = N = K = 1024
TARGET_TOPS = {"cpu": 4.0, "gpu": 15.0, "npu": 36.0}
GPU_INT8_GEMM_BASE_WMMA_VARIANT = "lds_128x64_wmma4"
GPU_INT8_GEMM_AIR_TUNED_DIRECT_VARIANT = "global_128x128_bpack_w4_direct"
GPU_INT8_GEMM_DIRECT_CANONICAL_VARIANT = "global_128x128_bpack_w4_direct_canonical"
GPU_INT8_GEMM_DIRECT_PREFETCH_VARIANT = "global_128x128_bpack_w4_prefetch"
GPU_INT8_GEMM_DIRECT_RAWPTR_VARIANT = "global_128x128_bpack_w4_direct_rawptr"
GPU_INT8_GEMM_DIRECT_RAWPTR_U2_VARIANT = "global_128x128_bpack_w4_direct_rawptr_u2"
GPU_INT8_GEMM_ROCMLIR_LIKE_VARIANT = "lds_128x128_rocmlir_k32_pipe3"
GPU_INT8_GEMM_TENSILE_K32_PIPE2_VARIANT = "lds_128x128_tensile_k32_pipe2"
GPU_INT8_GEMM_TENSILE_K32_PIPE2_PAD_VARIANT = "lds_128x128_tensile_k32_pipe2_pad"
GPU_INT8_GEMM_TENSILE_K32_PIPE3_VARIANT = "lds_128x128_tensile_k32_pipe3"
GPU_INT8_GEMM_TENSILE_K32_PIPE3_PAD_VARIANT = "lds_128x128_tensile_k32_pipe3_pad"
GPU_INT8_GEMM_TENSILE_K32_PIPE3_WPE2_VARIANT = "lds_128x128_tensile_k32_pipe3_wpe2"
DEFAULT_GPU_INT8_GEMM_VARIANT = GPU_INT8_GEMM_BASE_WMMA_VARIANT


@dataclass(frozen=True)
class GpuInt8GemmVariantConfig:
    variant: str
    block_rows: int
    block_cols: int
    k_per_block: int
    workgroup_threads: int
    lds_stages: int
    swizzled_lds: bool
    packed_b: bool
    grouped_blocks: bool
    direct_b_from_global: bool
    pipeline: str
    default_group_m: int = 4
    wave_tile_rows: int = 16
    wave_tile_cols: int | None = None
    lds_k_padding: int = 0
    waves_per_eu: int = 0

    @property
    def wave_count(self) -> int:
        return self.workgroup_threads // 32

    @property
    def k_tiles(self) -> int:
        return K // self.k_per_block

    @property
    def lds_bytes_per_workgroup(self) -> int:
        k_stride = self.k_per_block + self.lds_k_padding
        a_bytes = self.block_rows * k_stride * self.lds_stages
        b_bytes = 0 if self.direct_b_from_global else self.block_cols * k_stride * self.lds_stages
        return a_bytes + b_bytes

    @property
    def effective_wave_tile_cols(self) -> int:
        return self.wave_tile_cols or self.block_cols

    @property
    def dynamic_wmma_per_wave(self) -> int:
        return (K // 16) * (self.wave_tile_rows // 16) * (self.effective_wave_tile_cols // 16)

    @property
    def dynamic_wmma_per_workgroup(self) -> int:
        return self.dynamic_wmma_per_wave * self.wave_count

    @property
    def dynamic_barriers(self) -> int:
        if self.lds_stages == 0:
            return 0
        if self.pipeline in {"rocmlir_like_pipe3", "tensile_like_pipe3"}:
            return 2 * ((self.k_tiles - 2 + 2) // 3) + 1
        return self.k_tiles if self.lds_stages == 2 else 2 * self.k_tiles


GPU_INT8_GEMM_VARIANT_CONFIGS = (
    GpuInt8GemmVariantConfig(GPU_INT8_GEMM_BASE_WMMA_VARIANT, 128, 64, 64, 256, 1, False, False, False, False, "single"),
    GpuInt8GemmVariantConfig("lds_128x64_bpack", 128, 64, 64, 256, 1, False, True, False, False, "single"),
    GpuInt8GemmVariantConfig("lds_128x64_bpack_swizzle", 128, 64, 64, 256, 1, True, True, False, False, "single"),
    GpuInt8GemmVariantConfig("lds_128x64_bpack_pipe2", 128, 64, 64, 256, 2, False, True, False, False, "pipe2_unrolled_copy"),
    GpuInt8GemmVariantConfig("lds_128x64_bpack_pipe2_grouped", 128, 64, 64, 256, 2, False, True, True, False, "pipe2_unrolled_copy"),
    GpuInt8GemmVariantConfig("lds_128x64_bpack_swizzle_grouped", 128, 64, 64, 256, 1, True, True, True, False, "single"),
    GpuInt8GemmVariantConfig("lds_128x64_bpack_frag", 128, 64, 64, 256, 1, False, True, False, True, "single"),
    GpuInt8GemmVariantConfig("lds_128x128_bpack_swizzle_pipe2", 128, 128, 64, 256, 2, True, True, False, False, "pipe2_unrolled_prefetch"),
    GpuInt8GemmVariantConfig("lds_128x64_bpack_swizzle_pipe2_looped", 128, 64, 64, 256, 2, True, True, False, False, "pipe2_looped_prefetch"),
    GpuInt8GemmVariantConfig("lds_128x128_bpack_swizzle_looped", 128, 128, 64, 256, 1, True, True, False, False, "single"),
    GpuInt8GemmVariantConfig("lds_64x128_bpack_swizzle_pipe2_looped", 64, 128, 64, 128, 2, True, True, False, False, "pipe2_looped_prefetch"),
    GpuInt8GemmVariantConfig("lds_128x64_bpack_swizzle_pipe2_k32_looped", 128, 64, 32, 256, 2, True, True, False, False, "pipe2_looped_prefetch"),
    GpuInt8GemmVariantConfig("lds_128x64_bpack_swizzle_pipe2_k128_looped", 128, 64, 128, 256, 2, True, True, False, False, "pipe2_looped_prefetch"),
    GpuInt8GemmVariantConfig("lds_128x128_bpack_swizzle_k32_looped", 128, 128, 32, 256, 1, True, True, False, False, "single"),
    GpuInt8GemmVariantConfig("lds_128x128_bpack_swizzle_k128_looped", 128, 128, 128, 256, 1, True, True, False, False, "single"),
    GpuInt8GemmVariantConfig("lds_128x64_bpack_swizzle_breg_k64_looped", 128, 64, 64, 256, 2, True, True, False, True, "pipe2_looped_prefetch"),
    GpuInt8GemmVariantConfig("lds_128x64_bpack_swizzle_k32_w4_pipe2", 128, 64, 32, 128, 2, True, True, True, False, "tensile_like_pipe2", 8, 32, 64),
    GpuInt8GemmVariantConfig("lds_128x64_bpack_swizzle_k32_w4_pipe2_pad", 128, 64, 32, 128, 2, True, True, True, False, "tensile_like_pipe2", 8, 32, 64, 16),
    GpuInt8GemmVariantConfig("lds_64x128_bpack_swizzle_k32_w4_pipe2", 64, 128, 32, 128, 2, True, True, True, False, "tensile_like_pipe2", 8, 32, 64),
    GpuInt8GemmVariantConfig("lds_64x128_bpack_swizzle_k32_w4_pipe2_pad", 64, 128, 32, 128, 2, True, True, True, False, "tensile_like_pipe2", 8, 32, 64, 16),
    GpuInt8GemmVariantConfig("lds_128x128_bpack_swizzle_k32_w4_pipe2", 128, 128, 32, 128, 2, True, True, True, False, "tensile_like_pipe2", 8, 64, 64),
    GpuInt8GemmVariantConfig("lds_128x128_bpack_swizzle_k32_w4_pipe2_pad", 128, 128, 32, 128, 2, True, True, True, False, "tensile_like_pipe2", 8, 64, 64, 16),
    GpuInt8GemmVariantConfig("lds_128x64_bpack_swizzle_k32_w4_pipe2_short", 128, 64, 32, 128, 2, True, True, True, False, "tensile_like_pipe2_short", 8, 32, 64),
    GpuInt8GemmVariantConfig("lds_128x64_bpack_swizzle_k32_w4_pipe2_short_pad", 128, 64, 32, 128, 2, True, True, True, False, "tensile_like_pipe2_short", 8, 32, 64, 16),
    GpuInt8GemmVariantConfig("lds_64x128_bpack_swizzle_k32_w4_pipe2_short", 64, 128, 32, 128, 2, True, True, True, False, "tensile_like_pipe2_short", 8, 32, 64),
    GpuInt8GemmVariantConfig("lds_64x128_bpack_swizzle_k32_w4_pipe2_short_pad", 64, 128, 32, 128, 2, True, True, True, False, "tensile_like_pipe2_short", 8, 32, 64, 16),
    GpuInt8GemmVariantConfig("lds_128x128_bpack_swizzle_k32_w4_pipe2_short", 128, 128, 32, 128, 2, True, True, True, False, "tensile_like_pipe2_short", 8, 64, 64),
    GpuInt8GemmVariantConfig("lds_128x128_bpack_swizzle_k32_w4_pipe2_short_pad", 128, 128, 32, 128, 2, True, True, True, False, "tensile_like_pipe2_short", 8, 64, 64, 16),
    GpuInt8GemmVariantConfig(GPU_INT8_GEMM_TENSILE_K32_PIPE2_VARIANT, 128, 128, 32, 128, 2, True, True, True, False, "tensile_like_pipe2", 8, 64, 64),
    GpuInt8GemmVariantConfig(GPU_INT8_GEMM_TENSILE_K32_PIPE2_PAD_VARIANT, 128, 128, 32, 128, 2, True, True, True, False, "tensile_like_pipe2", 8, 64, 64, 16),
    GpuInt8GemmVariantConfig(GPU_INT8_GEMM_ROCMLIR_LIKE_VARIANT, 128, 128, 32, 128, 3, False, True, True, False, "rocmlir_like_pipe3", 8, 64, 64, 0),
    GpuInt8GemmVariantConfig(GPU_INT8_GEMM_TENSILE_K32_PIPE3_VARIANT, 128, 128, 32, 128, 3, True, True, True, False, "tensile_like_pipe3", 8, 64, 64, 0),
    GpuInt8GemmVariantConfig(GPU_INT8_GEMM_TENSILE_K32_PIPE3_PAD_VARIANT, 128, 128, 32, 128, 3, True, True, True, False, "tensile_like_pipe3", 8, 64, 64, 16),
    GpuInt8GemmVariantConfig(GPU_INT8_GEMM_TENSILE_K32_PIPE3_WPE2_VARIANT, 128, 128, 32, 128, 3, True, True, True, False, "tensile_like_pipe3", 8, 64, 64, 0, 2),
    GpuInt8GemmVariantConfig(GPU_INT8_GEMM_AIR_TUNED_DIRECT_VARIANT, 128, 128, 32, 128, 0, False, True, True, True, "air_tuned_direct", 8, 64, 64),
    GpuInt8GemmVariantConfig(GPU_INT8_GEMM_DIRECT_CANONICAL_VARIANT, 128, 128, 16, 128, 0, False, True, True, True, "air_tuned_direct_canonical", 8, 64, 64),
    GpuInt8GemmVariantConfig(GPU_INT8_GEMM_DIRECT_PREFETCH_VARIANT, 128, 128, 32, 128, 0, False, True, True, True, "air_tuned_direct_prefetch", 8, 64, 64),
    GpuInt8GemmVariantConfig(GPU_INT8_GEMM_DIRECT_RAWPTR_VARIANT, 128, 128, 32, 128, 0, False, True, True, True, "air_tuned_direct_rawptr", 8, 64, 64),
    GpuInt8GemmVariantConfig(GPU_INT8_GEMM_DIRECT_RAWPTR_U2_VARIANT, 128, 128, 16, 128, 0, False, True, True, True, "air_tuned_direct_rawptr_u2", 8, 64, 64),
)
GPU_INT8_GEMM_VARIANT_BY_NAME = {config.variant: config for config in GPU_INT8_GEMM_VARIANT_CONFIGS}
GPU_INT8_GEMM_VARIANTS = tuple(GPU_INT8_GEMM_VARIANT_BY_NAME)
GPU_INT8_GEMM_SWEEP_VARIANTS = GPU_INT8_GEMM_VARIANTS
DEFAULT_GPU_INT8_GEMM_GROUP_SIZE = 4
GPU_INT8_GEMM_GROUP_SIZES = (2, 4, 8)
DEFAULT_GPU_INT8_GEMM_SWEEP_GROUP_SIZES = GPU_INT8_GEMM_GROUP_SIZES
DEFAULT_GPU_INT8_GEMM_SWEEP_REPETITIONS = 3
GPU_INT8_GEMM_GROUPED_SWIZZLE_VARIANT = "lds_128x64_bpack_swizzle_grouped"
GPU_INT8_GEMM_SWEEP_PROFILES = ("full", "default-decision", "gfx1150-rewrite", "gfx1150-next", "gfx1150-kshape", "gfx1150-breg", "gfx1150-tensile-like", "gfx1150-short-live", "gfx1150-air-tuned-direct", "gfx1150-rocmlir-like", "gfx1150-consistency", "gfx1150-opt-direct-canonical", "gfx1150-opt-direct-prefetch", "gfx1150-opt-tensile-pipe2", "gfx1150-opt-rawptr", "gfx1150-opt-rawptr-u2", "gfx1150-opt-chain")
DEFAULT_GPU_INT8_GEMM_SWEEP_PROFILE = "full"
DEFAULT_GPU_INT8_GEMM_DEFAULT_THRESHOLD_PCT = 3.0
DEFAULT_GPU_INT8_GEMM_DEFAULT_IMPROVEMENT_PCT = 5.0
GPU_INT8_GEMM_CONSISTENCY_MIN_TOPS = 8.35
GPU_INT8_GEMM_CONSISTENCY_TARGET_TOPS = 8.50
GPU_AIR_TUNED_ACCEPTANCE_PCT = 95.0
GPU_CANDIDATE_IMPROVEMENT_PCT = 5.0
GPU_INT8_GEMM_ACCEPTED_BEST_VARIANT = GPU_INT8_GEMM_ROCMLIR_LIKE_VARIANT
GPU_INT8_GEMM_ACCEPTED_BEST_GROUP_SIZE = 8
GPU_PROVIDER_BASELINES = ("hip_wmma", "rocwmma", "air_tuned", "rocblas_tensile", "ck_tile")
GPU_PROVIDER_EXECUTABLE = "hip_int8_gemm_baseline"
GPU_ROCMLIR_REFERENCE_PROVIDER = "rocmlir_reference"
GPU_ROCMLIR_REFERENCE_SOURCE = "compiler-reference"
GPU_PROVIDER_2X_TARGET_TOPS = 2.0 * 3.29
GPU_PROVIDER_ROCBLAS_PARITY_PCT = 95.0
DEFAULT_GPU_PROVIDER_VALIDATION_SAMPLES = 256
GEMM_INT8_OPS = 2.0 * M * N * K
GEMM_INT8_IDEAL_BYTES = M * K + K * N + M * N * 4
GPU_PROVIDER_BASELINE_FIELDNAMES = (
    "provider",
    "source",
    "status",
    "available",
    "validation",
    "mismatches",
    "repetitions",
    "median_mean_ms",
    "min_mean_ms",
    "mean_mean_ms",
    "stddev_mean_ms",
    "cv_mean_ms_pct",
    "best_kernel_min_ms",
    "median_tops",
    "mean_tops",
    "stddev_tops",
    "cv_tops_pct",
    "min_tops",
    "max_tops",
    "target_tops_2x",
    "target_pct_2x",
    "meets_2x",
    "ideal_bytes",
    "operational_intensity_ops_per_byte",
    "ideal_bandwidth_gbs",
    "mlir_air_pct_of_air_tuned",
    "passes_air_tuned_95pct",
    "candidate_improvement_pct",
    "keep_candidate",
    "wmma",
    "global_load_b128",
    "global_load_u8",
    "ds_read_b128",
    "ds_swizzle",
    "global_store_b32",
    "barriers",
    "waitcnt",
    "vgprs",
    "sgprs",
    "vgpr_spills",
    "sgpr_spills",
    "lds_bytes_per_workgroup",
    "global_load_lds",
    "ds_store_b128",
    "ds_store_b8",
    "ds_load_b128",
    "scratch_markers",
    "spills",
    "tops_by_rep",
    "mean_ms_by_rep",
    "mlir_air_pct_of_rocblas_tensile",
    "build_log",
    "run_log",
    "disassemble_log",
    "profile_log",
    "artifacts",
    "notes",
)
GPU_INT8_GEMM_GFX1150_REWRITE_CANDIDATES = (
    ("lds_128x64_wmma4", 4),
    ("lds_128x64_bpack_swizzle", 4),
    ("lds_128x64_bpack_swizzle_grouped", 2),
    ("lds_128x128_bpack_swizzle_pipe2", 4),
)
GPU_INT8_GEMM_GFX1150_NEXT_CANDIDATES = (
    ("lds_128x64_bpack_swizzle", 4),
    ("lds_128x64_bpack_swizzle_grouped", 2),
    ("lds_128x64_bpack_swizzle_pipe2_looped", 4),
    ("lds_128x128_bpack_swizzle_looped", 4),
    ("lds_64x128_bpack_swizzle_pipe2_looped", 4),
)
GPU_INT8_GEMM_GFX1150_KSHAPE_CANDIDATES = (
    ("lds_128x64_bpack_swizzle", 4),
    ("lds_128x64_bpack_swizzle_pipe2_looped", 4),
    ("lds_128x64_bpack_swizzle_pipe2_k32_looped", 4),
    ("lds_128x64_bpack_swizzle_pipe2_k128_looped", 4),
    ("lds_128x128_bpack_swizzle_looped", 4),
    ("lds_128x128_bpack_swizzle_k32_looped", 4),
    ("lds_128x128_bpack_swizzle_k128_looped", 4),
)
GPU_INT8_GEMM_GFX1150_BREG_CANDIDATES = (
    ("lds_128x64_bpack_swizzle_pipe2_looped", 4),
    ("lds_128x64_bpack_swizzle_breg_k64_looped", 4),
)
GPU_INT8_GEMM_GFX1150_TENSILE_LIKE_CANDIDATES = (
    ("lds_128x64_bpack_swizzle", 4),
    ("lds_128x128_bpack_swizzle_k32_looped", 4),
    ("lds_128x64_bpack_swizzle_k32_w4_pipe2", 8),
    ("lds_128x64_bpack_swizzle_k32_w4_pipe2_pad", 8),
    ("lds_64x128_bpack_swizzle_k32_w4_pipe2", 8),
    ("lds_64x128_bpack_swizzle_k32_w4_pipe2_pad", 8),
    ("lds_128x128_bpack_swizzle_k32_w4_pipe2", 8),
    ("lds_128x128_bpack_swizzle_k32_w4_pipe2_pad", 8),
    ("lds_128x64_bpack_swizzle_k32_w4_pipe2_short", 8),
    ("lds_128x64_bpack_swizzle_k32_w4_pipe2_short_pad", 8),
    ("lds_64x128_bpack_swizzle_k32_w4_pipe2_short", 8),
    ("lds_64x128_bpack_swizzle_k32_w4_pipe2_short_pad", 8),
    ("lds_128x128_bpack_swizzle_k32_w4_pipe2_short", 8),
    ("lds_128x128_bpack_swizzle_k32_w4_pipe2_short_pad", 8),
)
GPU_INT8_GEMM_GFX1150_SHORT_LIVE_CANDIDATES = (
    ("lds_128x64_bpack_swizzle", 4),
    ("lds_128x128_bpack_swizzle_k32_w4_pipe2_pad", 8),
    ("lds_128x64_bpack_swizzle_k32_w4_pipe2_short", 8),
    ("lds_128x64_bpack_swizzle_k32_w4_pipe2_short_pad", 8),
    ("lds_64x128_bpack_swizzle_k32_w4_pipe2_short", 8),
    ("lds_64x128_bpack_swizzle_k32_w4_pipe2_short_pad", 8),
    ("lds_128x128_bpack_swizzle_k32_w4_pipe2_short", 8),
    ("lds_128x128_bpack_swizzle_k32_w4_pipe2_short_pad", 8),
)
GPU_INT8_GEMM_GFX1150_AIR_TUNED_DIRECT_CANDIDATES = (
    ("lds_128x64_bpack_swizzle", 4),
    (GPU_INT8_GEMM_AIR_TUNED_DIRECT_VARIANT, 8),
)
GPU_INT8_GEMM_GFX1150_ROCMLIR_LIKE_CANDIDATES = (
    ("lds_128x64_bpack_swizzle", 4),
    (GPU_INT8_GEMM_ROCMLIR_LIKE_VARIANT, 8),
    (GPU_INT8_GEMM_AIR_TUNED_DIRECT_VARIANT, 8),
)
GPU_INT8_GEMM_GFX1150_CONSISTENCY_CANDIDATES = (
    (GPU_INT8_GEMM_ACCEPTED_BEST_VARIANT, GPU_INT8_GEMM_ACCEPTED_BEST_GROUP_SIZE),
    (GPU_INT8_GEMM_TENSILE_K32_PIPE3_VARIANT, 8),
    (GPU_INT8_GEMM_TENSILE_K32_PIPE3_PAD_VARIANT, 8),
    (GPU_INT8_GEMM_TENSILE_K32_PIPE3_WPE2_VARIANT, 8),
)
GPU_INT8_GEMM_GFX1150_OPT_COMMON_CANDIDATES = tuple(dict.fromkeys((
    (GPU_INT8_GEMM_ACCEPTED_BEST_VARIANT, GPU_INT8_GEMM_ACCEPTED_BEST_GROUP_SIZE),
    (GPU_INT8_GEMM_TENSILE_K32_PIPE3_VARIANT, 8),
    (GPU_INT8_GEMM_TENSILE_K32_PIPE3_WPE2_VARIANT, 8),
    (GPU_INT8_GEMM_AIR_TUNED_DIRECT_VARIANT, 8),
    (GPU_INT8_GEMM_ROCMLIR_LIKE_VARIANT, 8),
)))
GPU_INT8_GEMM_GFX1150_OPT_DIRECT_CANONICAL_CANDIDATES = (
    *GPU_INT8_GEMM_GFX1150_OPT_COMMON_CANDIDATES,
    (GPU_INT8_GEMM_DIRECT_CANONICAL_VARIANT, 8),
)
GPU_INT8_GEMM_GFX1150_OPT_DIRECT_PREFETCH_CANDIDATES = (
    *GPU_INT8_GEMM_GFX1150_OPT_COMMON_CANDIDATES,
    (GPU_INT8_GEMM_DIRECT_CANONICAL_VARIANT, 8),
    (GPU_INT8_GEMM_DIRECT_PREFETCH_VARIANT, 8),
)
GPU_INT8_GEMM_GFX1150_OPT_TENSILE_PIPE2_CANDIDATES = (
    *GPU_INT8_GEMM_GFX1150_OPT_COMMON_CANDIDATES,
    (GPU_INT8_GEMM_DIRECT_CANONICAL_VARIANT, 8),
    (GPU_INT8_GEMM_TENSILE_K32_PIPE2_VARIANT, 8),
    (GPU_INT8_GEMM_TENSILE_K32_PIPE2_PAD_VARIANT, 8),
)
GPU_INT8_GEMM_GFX1150_OPT_RAWPTR_CANDIDATES = (
    *GPU_INT8_GEMM_GFX1150_OPT_COMMON_CANDIDATES,
    (GPU_INT8_GEMM_DIRECT_RAWPTR_VARIANT, 8),
)
GPU_INT8_GEMM_GFX1150_OPT_RAWPTR_U2_CANDIDATES = (
    *GPU_INT8_GEMM_GFX1150_OPT_COMMON_CANDIDATES,
    (GPU_INT8_GEMM_DIRECT_RAWPTR_VARIANT, 8),
    (GPU_INT8_GEMM_DIRECT_RAWPTR_U2_VARIANT, 8),
)
GPU_INT8_GEMM_GFX1150_OPT_CHAIN_CANDIDATES = (
    *GPU_INT8_GEMM_GFX1150_OPT_COMMON_CANDIDATES,
    (GPU_INT8_GEMM_DIRECT_CANONICAL_VARIANT, 8),
    (GPU_INT8_GEMM_DIRECT_PREFETCH_VARIANT, 8),
    (GPU_INT8_GEMM_DIRECT_RAWPTR_VARIANT, 8),
    (GPU_INT8_GEMM_DIRECT_RAWPTR_U2_VARIANT, 8),
    (GPU_INT8_GEMM_TENSILE_K32_PIPE2_VARIANT, 8),
    (GPU_INT8_GEMM_TENSILE_K32_PIPE2_PAD_VARIANT, 8),
)
GPU_INT8_GEMM_DEFAULT_DECISION_CANDIDATES = (
    (GPU_INT8_GEMM_BASE_WMMA_VARIANT, 4),
    ("lds_128x64_bpack_swizzle", 4),
    ("lds_128x64_bpack_swizzle_grouped", 2),
    ("lds_128x64_bpack_swizzle_grouped", 4),
    ("lds_128x64_bpack_swizzle_grouped", 8),
)
GPU_STATIC_COUNTER_KEYS = (
    "wmma",
    "lds_bytes_per_workgroup",
    "barriers",
    "vgprs",
    "sgprs",
    "vgpr_spills",
    "sgpr_spills",
    "scratch_markers",
    "spills",
    "global_load_b128",
    "global_load_lds",
    "ds_store_b128",
    "ds_store_b8",
    "ds_load_b128",
    "global_store_b32",
    "waitcnt",
)
GPU_DYNAMIC_COUNTER_KEYS = (
    "dynamic_wmma_per_wave",
    "dynamic_wmma_per_workgroup",
    "dynamic_barriers",
    "dynamic_waitcnt_estimate",
    "wmma_per_barrier",
    "wmma_per_waitcnt",
)
GPU_RESOURCE_COUNTER_KEYS = (
    "lds_bytes_per_workgroup",
    "vgprs",
    "sgprs",
    "vgpr_spills",
    "sgpr_spills",
    "scratch_markers",
    "spills",
)
GPU_COMPARISON_COUNTER_KEYS = GPU_DYNAMIC_COUNTER_KEYS + GPU_RESOURCE_COUNTER_KEYS
GPU_STATIC_COUNTER_LABELS = {
    "wmma": "static WMMA instructions",
    "lds_bytes_per_workgroup": "LDS bytes/workgroup",
    "barriers": "static barriers",
    "vgprs": "VGPRs",
    "sgprs": "SGPRs",
    "vgpr_spills": "VGPR spills",
    "sgpr_spills": "SGPR spills",
    "scratch_markers": "scratch markers",
    "spills": "spills",
    "global_load_b128": "static global_load_b128",
    "global_load_lds": "static global_load_lds",
    "ds_store_b128": "static ds_store_b128",
    "ds_store_b8": "static ds_store_b8",
    "ds_load_b128": "static ds_load_b128",
    "global_store_b32": "static global_store_b32",
    "waitcnt": "static waitcnt",
}
GPU_DYNAMIC_COUNTER_LABELS = {
    "dynamic_wmma_per_wave": "dynamic WMMA/wave",
    "dynamic_wmma_per_workgroup": "dynamic WMMA/workgroup",
    "dynamic_barriers": "dynamic barriers",
    "dynamic_waitcnt_estimate": "dynamic waitcnt estimate",
    "wmma_per_barrier": "WMMA/barrier",
    "wmma_per_waitcnt": "WMMA/waitcnt",
    "lds_bytes_per_workgroup": "LDS bytes/workgroup",
    "vgprs": "VGPRs",
    "sgprs": "SGPRs",
    "vgpr_spills": "VGPR spills",
    "sgpr_spills": "SGPR spills",
    "scratch_markers": "scratch markers",
    "spills": "spills",
}

def positive_int(value: str) -> int:
    parsed = int(value, 10)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value, 10)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def gpu_group_size(value: str) -> int:
    parsed = positive_int(value)
    if parsed not in GPU_INT8_GEMM_GROUP_SIZES:
        choices = ", ".join(str(size) for size in GPU_INT8_GEMM_GROUP_SIZES)
        raise argparse.ArgumentTypeError(f"value must be one of: {choices}")
    return parsed


def gpu_group_sizes(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(gpu_group_size(item.strip()) for item in value.split(",") if item.strip())
    except argparse.ArgumentTypeError:
        raise
    if not parsed:
        raise argparse.ArgumentTypeError("at least one group size is required")
    return tuple(dict.fromkeys(parsed))


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def count_regex(text_or_path: str | Path, pattern: str) -> int:
    text = read_text(text_or_path) if isinstance(text_or_path, Path) else text_or_path
    return len(re.findall(pattern, text, flags=re.MULTILINE))


def last_kv_value(path: Path, key: str) -> str:
    matches = re.findall(rf"\b{re.escape(key)}=([^\s]+)", read_text(path))
    return matches[-1] if matches else ""


def timing_field(path: Path, domain: str, field_name: str) -> str:
    for line in read_text(path).splitlines():
        if f"timing_domain={domain}" in line:
            return dict(re.findall(r"([A-Za-z0-9_.-]+)=([^\s]+)", line)).get(field_name, "")
    return ""


def mlir_string_attr(path: Path, attr_name: str) -> str:
    matches = re.findall(rf'{re.escape(attr_name)} = "([^"]+)"', read_text(path))
    return matches[-1] if matches else ""


def mlir_int_attr(path: Path, attr_name: str) -> str:
    matches = re.findall(rf'{re.escape(attr_name)} = ([0-9]+)(?: : [a-z0-9]+)?', read_text(path))
    return matches[-1] if matches else ""


def summary_metric(path: Path, metric_name: str) -> str:
    matches = re.findall(rf'{re.escape(metric_name)}:\s*([^\s]+)', read_text(path))
    return matches[-1] if matches else ""


def gpu_summary_metadata_metric(path: Path, metric_name: str) -> str:
    text = read_text(path)
    for pattern in (
        rf'\.{re.escape(metric_name)}:\s*([^\s]+)',
        rf'\b{re.escape(metric_name)}\s*=\s*([0-9]+)\s*:',
    ):
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1]
    return ""


def gpu_variant_config(variant: str) -> GpuInt8GemmVariantConfig:
    try:
        return GPU_INT8_GEMM_VARIANT_BY_NAME[variant]
    except KeyError as exc:
        raise ValueError(f"unknown GPU INT8 GEMM variant: {variant}") from exc


def gpu_variant_uses_packed_b(variant: str) -> bool:
    return gpu_variant_config(variant).packed_b


def gpu_variant_uses_fragment_b(variant: str) -> bool:
    return variant == "lds_128x64_bpack_frag"


def gpu_b_pack_function(variant: str) -> str:
    return "mgpuPackBFragI8I32" if gpu_variant_uses_fragment_b(variant) else "mgpuPackBI8I32"


def gpu_static_lds_bytes_per_workgroup(variant: str) -> int:
    return gpu_variant_config(variant).lds_bytes_per_workgroup


def gpu_dynamic_waitcnt_estimate(static_waitcnt: int, static_wmma: int, config: GpuInt8GemmVariantConfig) -> int:
    if static_wmma <= 0:
        return 0
    return round(static_waitcnt * (config.dynamic_wmma_per_wave / static_wmma))


def fmt_ratio(numerator: float, denominator: float) -> str:
    return f"{numerator / denominator:.6f}" if denominator else ""


def to_tops(gops: str) -> str:
    try:
        return f"{float(gops) / 1000.0:.6f}" if gops else "n/a"
    except ValueError:
        return "n/a"


def parse_gops_tops(gops: str) -> float | None:
    try:
        return float(gops) / 1000.0 if gops else None
    except ValueError:
        return None


def parse_tops(value: str) -> float | None:
    try:
        return float(value) if value else None
    except ValueError:
        return None


def parse_float(value: str) -> float | None:
    try:
        return float(value) if value else None
    except ValueError:
        return None


def parse_int_value(value: str) -> int | None:
    try:
        return int(value) if value else None
    except ValueError:
        return None


def fmt_float(value: float | None) -> str:
    return f"{value:.6f}" if value is not None else ""


def stddev_or_zero(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def coefficient_of_variation_pct(mean: float, stddev: float) -> float:
    return (stddev / mean) * 100.0 if mean else 0.0


def set_perf_tops(result: "BackendResult", tops: float | None) -> None:
    result.perf_tops = tops
    if tops is None or result.target_tops is None:
        result.target_pct = "n/a"
        return
    result.target_pct = f"{(tops / result.target_tops) * 100.0:.1f}%"


def us_to_ms(us: str) -> str:
    try:
        return f"{float(us) / 1000.0:.6f}" if us else "n/a"
    except ValueError:
        return "n/a"


def sanitize_prefix(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "artifact"


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"expected exactly one occurrence of {old!r}")
    return text.replace(old, new, 1)


@dataclass
class RunContext:
    repo: Path
    out_dir: Path
    build_root: Path
    logs_dir: Path
    warmups: int
    iterations: int
    gpu_arch: str
    gpu_int8_gemm_variant: str
    gpu_int8_gemm_group_size: int
    run_enabled: bool
    cpu_threads: int
    npu_runtime_loop_tiling: str

    @property
    def disassemble(self) -> Path:
        return self.repo / "utils" / "isa_inspect" / "disassemble.sh"


@dataclass
class BackendResult:
    backend: str
    build_dir: Path
    artifacts_dir: Path
    status: str = "SKIP"
    evidence: str = "not selected"
    runtime: str = "not run"
    perf_domain: str = "not run"
    perf_count: str = "n/a"
    perf_latency: str = "n/a"
    perf_throughput: str = "n/a"
    perf_tops: float | None = None
    target_tops: float | None = None
    target_pct: str = "n/a"
    perf_notes: str = "not run"
    logs: dict[str, Path] = field(default_factory=dict)


def backend_result(ctx: RunContext, name: str, create: bool = False) -> BackendResult:
    build_dir = ctx.build_root / name
    result = BackendResult(name, build_dir, build_dir / "artifacts")
    result.target_tops = TARGET_TOPS.get(name)
    if create:
        result.build_dir.mkdir(parents=True, exist_ok=True)
        result.artifacts_dir.mkdir(parents=True, exist_ok=True)
    return result


def log_path(ctx: RunContext, result: BackendResult, stem: str) -> Path:
    path = ctx.logs_dir / f"{result.backend}_{stem}.log"
    result.logs[stem] = path
    return path


def run_capture(log: Path, argv: Sequence[object], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> tuple[bool, Path]:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as output:
        cd = f"cd {cwd} && " if cwd else ""
        output.write(f"+ {cd}{' '.join(shlex.quote(str(arg)) for arg in argv)}\n")
        output.flush()
        try:
            completed = subprocess.run([str(arg) for arg in argv], cwd=str(cwd) if cwd else None, env=env, stdout=output, stderr=subprocess.STDOUT, text=True, check=False)
            return completed.returncode == 0, log
        except OSError as exc:
            output.write(f"ERROR: {exc}\n")
            return False, log


def run_logged(ctx: RunContext, result: BackendResult, stem: str, argv: Sequence[object], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> tuple[bool, Path]:
    return run_capture(log_path(ctx, result, stem), argv, cwd=cwd, env=env)


def note_run_failure(result: BackendResult, log: Path, label: str = "run") -> None:
    result.runtime = f"{label} failed; see {log}"
    result.perf_notes = f"{label} failed; see {log}"


def parse_host_perf(ctx: RunContext, result: BackendResult, log: Path, domain: str) -> None:
    avg_us, min_us, max_us, gops = (last_kv_value(log, key) for key in ("avg_us", "min_us", "max_us", "gops"))
    result.perf_domain = domain
    result.perf_count = last_kv_value(log, "iterations") or str(ctx.iterations)
    result.perf_latency = f"mean {avg_us or 'n/a'} us ({us_to_ms(avg_us)} ms), min {min_us or 'n/a'} us, max {max_us or 'n/a'} us"
    result.perf_throughput = f"{gops or 'n/a'} GOPS ({to_tops(gops)} TOPS)"
    set_perf_tops(result, parse_gops_tops(gops))


def cpu_backend(ctx: RunContext) -> BackendResult:
    result = backend_result(ctx, "cpu", True)
    source_dir = ctx.repo / "test" / "cpu" / "int8_gemm"
    binary = result.build_dir / "int8_gemm_cpu"
    disasm = result.artifacts_dir / "cpu_int8_gemm.disasm.s"
    ok, log = run_logged(ctx, result, "build", ["make", "-C", source_dir, f"BUILD_DIR={result.build_dir}"])
    if not ok:
        result.status, result.evidence = "WARN", f"CPU benchmark build failed; see {log}"
        return result
    ok, log = run_logged(ctx, result, "disassemble", [ctx.disassemble, "cpu", "--output-dir", result.artifacts_dir, "--prefix", "cpu_int8_gemm", "--symbol", "cpu_i8_gemm_vnni", "--expect", "vpdpbusd", binary])
    if not ok:
        result.status, result.evidence = "FAIL", f"CPU disassembly did not show required VNNI marker; see {log}"
        return result
    if ctx.run_enabled:
        ok, log = run_logged(ctx, result, "run", [binary, "--warmups", ctx.warmups, "--iterations", ctx.iterations, "--threads", ctx.cpu_threads])
        result.runtime = f"ran; see {log}" if ok else result.runtime
        if not ok:
            note_run_failure(result, log)
    vnni = count_regex(disasm, r"\bvpdpbusd\b")
    zmm = count_regex(disasm, r"\bzmm[0-9]+")
    result.status = "PASS" if vnni else "FAIL"
    result.evidence = f"vpdpbusd={vnni}, zmm_refs={zmm}"
    if ctx.run_enabled and "failed" in result.runtime and result.status == "PASS":
        result.status = "WARN"
    if ctx.run_enabled and (run_log := result.logs.get("run")) and run_log.exists():
        parse_host_perf(ctx, result, run_log, last_kv_value(run_log, "timing_domain") or "host_steady_clock")
        result.perf_notes = f"threads={last_kv_value(run_log, 'threads') or ctx.cpu_threads}; warmups={ctx.warmups}; validation={last_kv_value(run_log, 'validation') or 'unknown'}"
    return result


def gpu_tools(repo: Path) -> tuple[str | None, str | None]:
    runner = os.environ.get("MLIR_RUNNER") or shutil.which("mlir-runner")
    if not runner and (candidate := repo / "llvm" / "install-amdgpu" / "bin" / "mlir-runner").exists():
        runner = str(candidate)
    airgpu = os.environ.get("AIRGPU_LIB")
    if not airgpu and (candidate := repo / "build-gpu" / "lib" / "libairgpu.so").exists():
        airgpu = str(candidate)
    if not airgpu and os.environ.get("MLIR_AIR_INSTALL_DIR"):
        airgpu = str(Path(os.environ["MLIR_AIR_INSTALL_DIR"]) / "lib" / "libairgpu.so")
    if not airgpu and (candidate := repo / "install-gpu" / "lib" / "libairgpu.so").exists():
        airgpu = str(candidate)
    return runner, airgpu


def gpu_compile_env(repo: Path) -> dict[str, str]:
    env = os.environ.copy()
    if not env.get("AIR_OPT") and (candidate := repo / "build-gpu" / "bin" / "air-opt").exists():
        env["AIR_OPT"] = str(candidate)
    if not env.get("MLIR_OPT") and (candidate := repo / "llvm" / "install-amdgpu" / "bin" / "mlir-opt").exists():
        env["MLIR_OPT"] = str(candidate)
    return env


def run_gpu_final_mlir(ctx: RunContext, result: BackendResult, final_mlir: Path, stem: str) -> tuple[bool, Path]:
    runner, airgpu = gpu_tools(ctx.repo)
    log = log_path(ctx, result, stem)
    if not runner or not Path(runner).exists():
        write_text(log, "ERROR: mlir-runner not found\n")
        return False, log
    if not airgpu or not Path(airgpu).exists():
        write_text(log, "ERROR: libairgpu.so not found\n")
        return False, log
    env = os.environ.copy()
    env.setdefault("AIRGPU_USE_HIP_MALLOC", "1")
    env.setdefault("AIRGPU_BENCHMARK_STREAM", "1")
    return run_capture(log, [runner, "--entry-point-result=void", f"--shared-libs={airgpu}", final_mlir], env=env)


def gpu_run_metrics(log: Path) -> dict[str, float]:
    fields = {
        "mean_ms": parse_float(timing_field(log, "kernel_event", "mean_ms")),
        "min_ms": parse_float(timing_field(log, "kernel_event", "min_ms")),
        "max_ms": parse_float(timing_field(log, "kernel_event", "max_ms")),
        "tops": parse_tops(timing_field(log, "kernel_event", "tops")),
    }
    return {key: value for key, value in fields.items() if value is not None}


def gpu_repetition_metric_entries(logs: Sequence[Path]) -> list[dict[str, float]]:
    metrics = [gpu_run_metrics(log) for log in logs]
    return [entry for entry in metrics if {"mean_ms", "min_ms", "tops"} <= set(entry)]


def format_metric_series(values: Sequence[float]) -> str:
    return ",".join(f"{value:.6f}" for value in values)


def gpu_repetition_series(logs: Sequence[Path]) -> dict[str, str]:
    metrics = gpu_repetition_metric_entries(logs)
    return {
        "tops_by_rep": format_metric_series([entry["tops"] for entry in metrics]),
        "mean_ms_by_rep": format_metric_series([entry["mean_ms"] for entry in metrics]),
    }


def summarize_gpu_repetition_metrics(logs: Sequence[Path]) -> dict[str, float]:
    metrics = gpu_repetition_metric_entries(logs)
    if not metrics:
        return {}
    mean_ms = [entry["mean_ms"] for entry in metrics]
    min_ms = [entry["min_ms"] for entry in metrics]
    tops = [entry["tops"] for entry in metrics]
    mean_mean_ms = statistics.mean(mean_ms)
    stddev_mean_ms = stddev_or_zero(mean_ms)
    mean_tops = statistics.mean(tops)
    stddev_tops = stddev_or_zero(tops)
    return {
        "repetitions": float(len(metrics)),
        "median_mean_ms": statistics.median(mean_ms),
        "min_mean_ms": min(mean_ms),
        "mean_mean_ms": mean_mean_ms,
        "stddev_mean_ms": stddev_mean_ms,
        "cv_mean_ms_pct": coefficient_of_variation_pct(mean_mean_ms, stddev_mean_ms),
        "best_kernel_min_ms": min(min_ms),
        "median_tops": statistics.median(tops),
        "mean_tops": mean_tops,
        "stddev_tops": stddev_tops,
        "cv_tops_pct": coefficient_of_variation_pct(mean_tops, stddev_tops),
        "min_tops": min(tops),
        "max_tops": max(tops),
    }


def apply_gpu_repetition_summary(ctx: RunContext, result: BackendResult, summary: dict[str, float], variant: str, group_size: int, run_logs: Sequence[Path]) -> None:
    if not summary:
        return
    repetitions = int(summary["repetitions"])
    result.perf_domain = "kernel_event"
    result.perf_count = f"{repetitions}x{ctx.iterations}"
    result.perf_latency = (
        f"median mean {summary['median_mean_ms']:.6f} ms, "
        f"mean mean {summary['mean_mean_ms']:.6f} ms, "
        f"stddev mean {summary['stddev_mean_ms']:.6f} ms, "
        f"cv {summary['cv_mean_ms_pct']:.3f}%, "
        f"min mean {summary['min_mean_ms']:.6f} ms, "
        f"best kernel min {summary['best_kernel_min_ms']:.6f} ms"
    )
    result.perf_throughput = (
        f"median {summary['median_tops']:.6f} TOPS, "
        f"mean {summary['mean_tops']:.6f} TOPS, "
        f"stddev {summary['stddev_tops']:.6f} TOPS, "
        f"cv {summary['cv_tops_pct']:.3f}%, "
        f"max {summary['max_tops']:.6f} TOPS"
    )
    set_perf_tops(result, summary["median_tops"])
    result.perf_notes = f"variant={variant}; group_m={group_size}; repetitions={repetitions}; warmups={ctx.warmups}"
    result.runtime = f"ran {repetitions} repetition(s); see {', '.join(str(log) for log in run_logs)}"


def render_gpu_mlir(source: Path, dest: Path, ctx: RunContext) -> None:
    text = read_text(source)
    for old, new in (
        ("%c10 = arith.constant 10 : index", f"%c10 = arith.constant {ctx.warmups} : index"),
        ("%c5 = arith.constant 5 : index", f"%c5 = arith.constant {ctx.iterations} : index"),
        ("%c10_i64 = arith.constant 10 : i64", f"%c10_i64 = arith.constant {ctx.warmups} : i64"),
        ("%c5_i64 = arith.constant 5 : i64", f"%c5_i64 = arith.constant {ctx.iterations} : i64"),
    ):
        text = replace_once(text, old, new)
    if gpu_variant_uses_packed_b(ctx.gpu_int8_gemm_variant):
        text = render_gpu_packed_b_mlir(text, ctx.gpu_int8_gemm_variant)
    write_text(dest, text)


def render_gpu_packed_b_mlir(text: str, variant: str) -> str:
    pack_function = gpu_b_pack_function(variant)
    text = replace_once(
        text,
        "  llvm.func @mgpuCheckOutputI8I32(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64) -> i32\n",
        "  llvm.func @mgpuCheckOutputI8I32(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64) -> i32\n"
        f"  llvm.func @{pack_function}(!llvm.ptr, !llvm.ptr, i64, i64)\n",
    )
    text = replace_once(
        text,
        "    %alloc_1 = memref.alloc() : memref<1024x1024xi32>\n",
        "    %alloc_1 = memref.alloc() : memref<1024x1024xi32>\n"
        "    %alloc_2 = memref.alloc() : memref<1024x1024xi8>\n",
    )
    text = replace_once(
        text,
        "    %b_ptr = llvm.inttoptr %b_ptr_i64 : i64 to !llvm.ptr\n",
        "    %b_ptr = llvm.inttoptr %b_ptr_i64 : i64 to !llvm.ptr\n"
        "    %bpack_intptr = memref.extract_aligned_pointer_as_index %alloc_2 : memref<1024x1024xi8> -> index\n"
        "    %bpack_ptr_i64 = arith.index_cast %bpack_intptr : index to i64\n"
        "    %bpack_ptr = llvm.inttoptr %bpack_ptr_i64 : i64 to !llvm.ptr\n",
    )
    text = replace_once(
        text,
        "    llvm.call @mgpuInitI8I32(%a_ptr, %b_ptr, %m64, %n64, %k64) : (!llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()\n",
        "    llvm.call @mgpuInitI8I32(%a_ptr, %b_ptr, %m64, %n64, %k64) : (!llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()\n"
        f"    llvm.call @{pack_function}(%b_ptr, %bpack_ptr, %n64, %k64) : (!llvm.ptr, !llvm.ptr, i64, i64) -> ()\n",
    )
    text = replace_once(
        text,
        "    gpu.memcpy %memref_2, %alloc_0 : memref<1024x1024xi8>, memref<1024x1024xi8>\n",
        "    gpu.memcpy %memref_2, %alloc_2 : memref<1024x1024xi8>, memref<1024x1024xi8>\n",
    )
    text = replace_once(
        text,
        "    memref.dealloc %alloc_0 : memref<1024x1024xi8>\n"
        "    memref.dealloc %alloc_1 : memref<1024x1024xi32>\n",
        "    memref.dealloc %alloc_0 : memref<1024x1024xi8>\n"
        "    memref.dealloc %alloc_1 : memref<1024x1024xi32>\n"
        "    memref.dealloc %alloc_2 : memref<1024x1024xi8>\n",
    )
    return text


def gpu_backend(ctx: RunContext) -> BackendResult:
    result = backend_result(ctx, "gpu", True)
    source = ctx.repo / "test" / "gpu" / "int8_gemm" / "air_sync.mlir"
    generated = result.build_dir / "int8_gemm.air_sync.mlir"
    isa = result.artifacts_dir / "gpu_int8_gemm.isa.s"
    summary = result.artifacts_dir / "gpu_int8_gemm.summary.txt"
    outline_mlir = result.artifacts_dir / "gpu_int8_gemm.outline.mlir"
    final_mlir = result.artifacts_dir / "gpu_int8_gemm.final.mlir"
    try:
        render_gpu_mlir(source, generated, ctx)
        write_text(log_path(ctx, result, "render"), f"generated {generated}\nwarmups={ctx.warmups}\niterations={ctx.iterations}\n")
    except Exception as exc:  # noqa: BLE001 - report rendering failures in the same log flow.
        log = log_path(ctx, result, "render")
        write_text(log, f"failed to render GPU MLIR: {exc}\n")
        result.status, result.evidence = "WARN", f"GPU MLIR render failed; see {log}"
        return result
    requested_config = gpu_variant_config(ctx.gpu_int8_gemm_variant)
    forbid = r"v_wmma_.*16x16x64|v_swmmac|swmmac"
    if requested_config.lds_stages == 0:
        forbid = rf"{forbid}|\bs_barrier\b|\bds_(read|load|store|write)_|uses_flat_scratch\s+1"
    ok, log = run_logged(ctx, result, "disassemble", [ctx.disassemble, "gpu", "--gpu-arch", ctx.gpu_arch, "--int8-gemm-variant", ctx.gpu_int8_gemm_variant, "--int8-gemm-group-size", ctx.gpu_int8_gemm_group_size, "--output-dir", result.artifacts_dir, "--prefix", "gpu_int8_gemm", "--expect", "v_wmma_i32_16x16x16_iu8", "--forbid", forbid, generated], env=gpu_compile_env(ctx.repo))
    if not ok:
        result.status, result.evidence = "WARN", f"GPU lowering/disassembly failed or required marker was absent; see {log}"
        return result
    runtime_logs: list[Path] = []
    if ctx.run_enabled:
        ok, log = run_gpu_final_mlir(ctx, result, final_mlir, "run")
        if ok:
            runtime_logs.append(log)
        else:
            note_run_failure(result, log)
    wmma = count_regex(isa, r"\bv_wmma_i32_16x16x16_iu8\b")
    barriers = count_regex(isa, r"\bs_barrier\b")
    scratch = count_regex(isa, r"uses_flat_scratch\s+1")
    variant = mlir_string_attr(outline_mlir, "air.gpu.int8_gemm_variant") or "unknown"
    lds_bytes_per_workgroup = gpu_static_lds_bytes_per_workgroup(variant)
    group_m = mlir_int_attr(outline_mlir, "air.gpu.int8_gemm_group_m") or str(ctx.gpu_int8_gemm_group_size)
    vgprs = gpu_summary_metadata_metric(summary, "vgpr_count") or "n/a"
    sgprs = gpu_summary_metadata_metric(summary, "sgpr_count") or "n/a"
    vgpr_spills = gpu_summary_metadata_metric(summary, "vgpr_spill_count") or "0"
    sgpr_spills = gpu_summary_metadata_metric(summary, "sgpr_spill_count") or "0"
    parsed_vgpr_spills = parse_int_value(vgpr_spills)
    parsed_sgpr_spills = parse_int_value(sgpr_spills)
    spills = (
        parsed_vgpr_spills + parsed_sgpr_spills
        if parsed_vgpr_spills is not None and parsed_sgpr_spills is not None
        else count_regex(summary, r"spill_count = [1-9]|_spill_count: [1-9]")
    )
    global_load_b128 = count_regex(isa, r"\bglobal_load_b128\b")
    global_load_lds = count_regex(isa, r"\bglobal_load(?:_async)?(?:_to)?_lds")
    ds_store_b128 = count_regex(isa, r"\bds_store_b128\b")
    ds_store_b8 = count_regex(isa, r"\bds_store_b8(?:_d16_hi)?\b")
    ds_load_b128 = count_regex(isa, r"\bds_(?:read|load)_b128\b")
    global_store_b32 = count_regex(isa, r"\bglobal_store_b32\b")
    waitcnt = count_regex(isa, r"\bs_waitcnt\b")
    config = gpu_variant_config(variant)
    dynamic_wmma_per_wave = config.dynamic_wmma_per_wave
    dynamic_wmma_per_workgroup = config.dynamic_wmma_per_workgroup
    dynamic_barriers = config.dynamic_barriers
    dynamic_waitcnt_estimate = gpu_dynamic_waitcnt_estimate(waitcnt, wmma, config)
    wmma_per_barrier = fmt_ratio(dynamic_wmma_per_wave, dynamic_barriers)
    wmma_per_waitcnt = fmt_ratio(dynamic_wmma_per_wave, dynamic_waitcnt_estimate)
    no_lds_direct_ok = True
    if config.lds_stages == 0:
        no_lds_direct_ok = (
            lds_bytes_per_workgroup == 0
            and barriers == 0
            and global_load_lds == 0
            and ds_store_b128 == 0
            and ds_store_b8 == 0
            and ds_load_b128 == 0
        )
    parsed_vgprs = parse_int_value(vgprs)
    vgpr_ok = parsed_vgprs is not None and parsed_vgprs < 256
    result.status = "PASS" if wmma and scratch == 0 and spills == 0 and vgpr_ok and no_lds_direct_ok else "FAIL"
    if ctx.run_enabled and "failed" in result.runtime and result.status == "PASS":
        result.status = "WARN"
    result.evidence = (
        f"variant={variant}, group_m={group_m}, wmma={wmma}, "
        f"lds_bytes_per_workgroup={lds_bytes_per_workgroup}, "
        f"barriers={barriers}, vgprs={vgprs}, "
        f"sgprs={sgprs}, vgpr_spills={vgpr_spills}, "
        f"sgpr_spills={sgpr_spills}, scratch_markers={scratch}, spills={spills}, "
        f"global_load_b128={global_load_b128}, global_load_lds={global_load_lds}, "
        f"ds_store_b128={ds_store_b128}, ds_store_b8={ds_store_b8}, "
        f"ds_load_b128={ds_load_b128}, global_store_b32={global_store_b32}, waitcnt={waitcnt}, "
        f"dynamic_wmma_per_wave={dynamic_wmma_per_wave}, "
        f"dynamic_wmma_per_workgroup={dynamic_wmma_per_workgroup}, "
        f"dynamic_barriers={dynamic_barriers}, "
        f"dynamic_waitcnt_estimate={dynamic_waitcnt_estimate}, "
        f"wmma_per_barrier={wmma_per_barrier}, wmma_per_waitcnt={wmma_per_waitcnt}"
    )
    if ctx.run_enabled and runtime_logs:
        apply_gpu_repetition_summary(
            ctx,
            result,
            summarize_gpu_repetition_metrics(runtime_logs),
            variant,
            int(group_m),
            runtime_logs,
        )
    return result



def evidence_map(result: BackendResult) -> dict[str, str]:
    return dict(re.findall(r"([A-Za-z0-9_.-]+)=([^,\s]+)", result.evidence))


def gpu_sweep_group_sizes_for_variant(
    ctx: RunContext, variant: str, group_sizes: Sequence[int]
) -> Sequence[int]:
    config = gpu_variant_config(variant)
    if not config.grouped_blocks:
        return (ctx.gpu_int8_gemm_group_size,)
    if config.pipeline in {"air_tuned_direct", "air_tuned_direct_canonical", "air_tuned_direct_prefetch", "air_tuned_direct_rawptr", "air_tuned_direct_rawptr_u2", "rocmlir_like_pipe3", "tensile_like_pipe3"}:
        return (config.default_group_m,)
    return group_sizes


def gpu_sweep_candidates(
    ctx: RunContext,
    sweep_profile: str,
    variants: Sequence[str],
    group_sizes: Sequence[int],
) -> list[tuple[str, int]]:
    if sweep_profile == "gfx1150-rewrite":
        return list(GPU_INT8_GEMM_GFX1150_REWRITE_CANDIDATES)
    if sweep_profile == "gfx1150-next":
        return list(GPU_INT8_GEMM_GFX1150_NEXT_CANDIDATES)
    if sweep_profile == "gfx1150-kshape":
        return list(GPU_INT8_GEMM_GFX1150_KSHAPE_CANDIDATES)
    if sweep_profile == "gfx1150-breg":
        return list(GPU_INT8_GEMM_GFX1150_BREG_CANDIDATES)
    if sweep_profile == "gfx1150-tensile-like":
        return list(GPU_INT8_GEMM_GFX1150_TENSILE_LIKE_CANDIDATES)
    if sweep_profile == "gfx1150-short-live":
        return list(GPU_INT8_GEMM_GFX1150_SHORT_LIVE_CANDIDATES)
    if sweep_profile == "gfx1150-air-tuned-direct":
        return list(GPU_INT8_GEMM_GFX1150_AIR_TUNED_DIRECT_CANDIDATES)
    if sweep_profile == "gfx1150-rocmlir-like":
        return list(GPU_INT8_GEMM_GFX1150_ROCMLIR_LIKE_CANDIDATES)
    if sweep_profile == "gfx1150-consistency":
        return list(GPU_INT8_GEMM_GFX1150_CONSISTENCY_CANDIDATES)
    if sweep_profile == "gfx1150-opt-direct-canonical":
        return list(GPU_INT8_GEMM_GFX1150_OPT_DIRECT_CANONICAL_CANDIDATES)
    if sweep_profile == "gfx1150-opt-direct-prefetch":
        return list(GPU_INT8_GEMM_GFX1150_OPT_DIRECT_PREFETCH_CANDIDATES)
    if sweep_profile == "gfx1150-opt-tensile-pipe2":
        return list(GPU_INT8_GEMM_GFX1150_OPT_TENSILE_PIPE2_CANDIDATES)
    if sweep_profile == "gfx1150-opt-rawptr":
        return list(GPU_INT8_GEMM_GFX1150_OPT_RAWPTR_CANDIDATES)
    if sweep_profile == "gfx1150-opt-rawptr-u2":
        return list(GPU_INT8_GEMM_GFX1150_OPT_RAWPTR_U2_CANDIDATES)
    if sweep_profile == "gfx1150-opt-chain":
        return list(GPU_INT8_GEMM_GFX1150_OPT_CHAIN_CANDIDATES)
    if sweep_profile == "default-decision":
        return list(GPU_INT8_GEMM_DEFAULT_DECISION_CANDIDATES)
    return [
        (variant, group_size)
        for variant in variants
        for group_size in gpu_sweep_group_sizes_for_variant(ctx, variant, group_sizes)
    ]


def row_float(row: dict[str, str], key: str) -> float | None:
    return parse_float(row.get(key, ""))


def row_int(row: dict[str, str], key: str) -> int | None:
    try:
        value = row.get(key, "")
        return int(value) if value else None
    except ValueError:
        return None


def gpu_row_label(row: dict[str, str]) -> str:
    return f"{row.get('variant', 'unknown')} group_m={row.get('group_m', 'n/a')}"


def is_current_gpu_default(row: dict[str, str]) -> bool:
    return row.get("variant") == DEFAULT_GPU_INT8_GEMM_VARIANT and row_int(row, "group_m") == DEFAULT_GPU_INT8_GEMM_GROUP_SIZE


def ranked_gpu_rows(rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    timed_rows = [row for row in rows if row_float(row, "median_tops") is not None]
    return sorted(
        timed_rows,
        key=lambda row: (-(row_float(row, "median_tops") or 0.0), row.get("variant", ""), row_int(row, "group_m") or 0),
    )


def is_accepted_gpu_best(row: dict[str, str]) -> bool:
    return row.get("variant") == GPU_INT8_GEMM_ACCEPTED_BEST_VARIANT and row_int(row, "group_m") == GPU_INT8_GEMM_ACCEPTED_BEST_GROUP_SIZE


def is_retained_gpu_candidate(row: dict[str, str]) -> bool:
    return row.get("keep_candidate") in {"accepted_best", "yes"}


def annotate_gpu_sweep_candidate_gates(rows: Sequence[dict[str, str]]) -> None:
    baseline = next((row for row in rows if is_accepted_gpu_best(row)), None)
    baseline_tops = row_float(baseline, "median_tops") if baseline else None
    for row in rows:
        row.setdefault("mlir_air_pct_of_air_tuned", "")
        row.setdefault("passes_air_tuned_95pct", "")
        row.setdefault("candidate_improvement_pct", "")
        row.setdefault("keep_candidate", "")
        row_tops = row_float(row, "median_tops")
        if baseline_tops is None or baseline_tops <= 0.0 or row_tops is None:
            continue
        improvement = ((row_tops - baseline_tops) / baseline_tops) * 100.0
        row["candidate_improvement_pct"] = f"{improvement:.3f}"
        if is_accepted_gpu_best(row):
            row["keep_candidate"] = "accepted_best"
        else:
            no_scratch = row_int(row, "scratch_markers") in (None, 0)
            no_vgpr_spills = row_int(row, "vgpr_spills") in (None, 0)
            no_sgpr_spills = row_int(row, "sgpr_spills") in (None, 0)
            no_spills = row_int(row, "spills") in (None, 0) and no_vgpr_spills and no_sgpr_spills
            vgprs = row_int(row, "vgprs")
            vgpr_ok = vgprs is not None and vgprs < 256
            keep = row.get("status") == "PASS" and no_scratch and no_spills and vgpr_ok and improvement >= GPU_CANDIDATE_IMPROVEMENT_PCT
            row["keep_candidate"] = "yes" if keep else "no"


def median_tops_gap_pct(top: dict[str, str], runner_up: dict[str, str]) -> float | None:
    top_tops = row_float(top, "median_tops")
    runner_tops = row_float(runner_up, "median_tops")
    if top_tops is None or runner_tops is None or runner_tops <= 0.0:
        return None
    return ((top_tops - runner_tops) / runner_tops) * 100.0


def comparison_counters_nearly_identical(top: dict[str, str], runner_up: dict[str, str]) -> bool:
    for key in GPU_COMPARISON_COUNTER_KEYS:
        top_value = row_float(top, key)
        runner_value = row_float(runner_up, key)
        if top_value is None or runner_value is None:
            if top.get(key, "") != runner_up.get(key, ""):
                return False
            continue
        min_tolerance = 2.0 if key in {"vgprs", "sgprs"} else 1.0
        tolerance = max(min_tolerance, max(abs(top_value), abs(runner_value)) * 0.05)
        if abs(top_value - runner_value) > tolerance:
            return False
    return True


def counter_delta(runner_up: dict[str, str], top: dict[str, str], key: str) -> str:
    runner_value = row_float(runner_up, key)
    top_value = row_float(top, key)
    if runner_value is None or top_value is None:
        return "n/a"
    delta = runner_value - top_value
    return f"{delta:+.0f}" if delta.is_integer() else f"{delta:+.3f}"


def counter_much_higher(slower: dict[str, str], faster: dict[str, str], key: str, min_abs_delta: float) -> bool:
    slower_value = row_float(slower, key)
    faster_value = row_float(faster, key)
    if slower_value is None or faster_value is None:
        return False
    delta = slower_value - faster_value
    if delta < min_abs_delta:
        return False
    if faster_value == 0.0:
        return True
    return (delta / abs(faster_value)) * 100.0 >= 10.0


def gpu_gap_analysis_notes(top: dict[str, str] | None, runner_up: dict[str, str] | None, gap_pct: float | None) -> list[str]:
    if not top or not runner_up:
        return ["Insufficient parseable runtime rows to compare the top candidates."]
    notes: list[str] = []
    top_cv = row_float(top, "cv_tops_pct") or 0.0
    runner_cv = row_float(runner_up, "cv_tops_pct") or 0.0
    cv_ceiling = max(top_cv, runner_cv)
    if gap_pct is not None and gap_pct < cv_ceiling:
        notes.append(f"Noise-limited: the top-vs-runner-up median TOPS gap ({gap_pct:.3f}%) is smaller than the larger top-candidate CV ({cv_ceiling:.3f}%).")
    elif gap_pct is not None:
        notes.append(f"The median TOPS gap ({gap_pct:.3f}%) is larger than the larger top-candidate CV ({cv_ceiling:.3f}%).")
    if comparison_counters_nearly_identical(top, runner_up):
        notes.append("The normalized execution/resource counters are nearly identical, so these counters do not explain a stable winner.")
    else:
        notes.append("The top candidates differ in normalized execution/resource counters; see the counter table below.")
    likely_causes = []
    for key, label, min_abs_delta in (("barriers", "barriers", 2.0), ("waitcnt", "waitcnt", 2.0), ("vgprs", "VGPRs", 8.0)):
        if counter_much_higher(runner_up, top, key, min_abs_delta):
            likely_causes.append(label)
    if likely_causes:
        notes.append(f"The runner-up is slower while carrying much higher {', '.join(likely_causes)}, which is a likely static reason for the gap.")
    return notes


def write_gpu_default_decision(
    ctx: RunContext,
    rows: Sequence[dict[str, str]],
    repetitions: int,
    threshold_pct: float,
    sweep_profile: str,
    csv_path: Path,
    sweep_md_path: Path,
) -> Path:
    decision_path = ctx.out_dir / "gpu_default_decision.md"
    ranked = ranked_gpu_rows(rows)
    top = ranked[0] if ranked else None
    runner_up = ranked[1] if len(ranked) > 1 else None
    gap_pct = median_tops_gap_pct(top, runner_up) if top and runner_up else None
    current_default = next((row for row in rows if is_current_gpu_default(row)), None)
    default_gap_pct = None
    all_pass = bool(rows) and all(row.get("status") == "PASS" for row in rows)
    all_parseable = bool(rows) and all(row_int(row, "repetitions") == repetitions for row in rows)
    gap_pass = gap_pct is not None and gap_pct >= threshold_pct
    top_is_current = bool(top and is_current_gpu_default(top))
    if top and current_default and not top_is_current:
        default_gap_pct = median_tops_gap_pct(top, current_default)
    elif top_is_current:
        default_gap_pct = 0.0
    default_gap_pass = (
        top_is_current
        or (default_gap_pct is not None and default_gap_pct >= DEFAULT_GPU_INT8_GEMM_DEFAULT_IMPROVEMENT_PCT)
    )
    promote = bool(all_pass and all_parseable and gap_pass and default_gap_pass and not top_is_current)
    recommendation = "PROMOTE" if promote else "KEEP_CURRENT_DEFAULT"
    blockers: list[str] = []
    if not all_pass:
        blockers.append("one or more sweep rows did not PASS")
    if not all_parseable:
        blockers.append(f"not every row produced {repetitions} parseable timing repetitions")
    if gap_pct is None:
        blockers.append("top-vs-runner-up gap is unavailable")
    elif not gap_pass:
        blockers.append(f"top-vs-runner-up gap {gap_pct:.3f}% is below the {threshold_pct:.3f}% threshold")
    if current_default is None:
        blockers.append("the current benchmark default is absent from this sweep")
    elif not default_gap_pass and not top_is_current:
        blockers.append(
            f"top-vs-current-default gap {default_gap_pct:.3f}% is below the {DEFAULT_GPU_INT8_GEMM_DEFAULT_IMPROVEMENT_PCT:.3f}% threshold"
        )
    if top_is_current:
        blockers.append("the top candidate is already the current benchmark default")
    if promote:
        decision_reason = (
            f"all rows passed, every row produced {repetitions} parseable timings, "
            f"the top candidate clears the {threshold_pct:.3f}% runner-up threshold, "
            f"and clears the {DEFAULT_GPU_INT8_GEMM_DEFAULT_IMPROVEMENT_PCT:.3f}% current-default threshold"
        )
    else:
        decision_reason = "; ".join(blockers) if blockers else "promotion conditions were not all satisfied"

    with decision_path.open("w", encoding="utf-8") as f:
        f.write("# GPU INT8 GEMM Default Decision\n\n")
        f.write(f"Recommendation: `{recommendation}`\n\n")
        f.write("## Decision Inputs\n\n| Field | Value |\n| --- | --- |\n")
        f.write(f"| Sweep profile | `{sweep_profile}` |\n")
        f.write(f"| Current benchmark default | `{DEFAULT_GPU_INT8_GEMM_VARIANT} group_m={DEFAULT_GPU_INT8_GEMM_GROUP_SIZE}` |\n")
        f.write(f"| Current default present | `{'yes' if current_default else 'no'}` |\n")
        f.write(f"| Runner-up promotion threshold | `{threshold_pct:.3f}%` |\n")
        f.write(f"| Current-default improvement threshold | `{DEFAULT_GPU_INT8_GEMM_DEFAULT_IMPROVEMENT_PCT:.3f}%` |\n")
        f.write(f"| Top candidate | `{gpu_row_label(top) if top else 'n/a'}` |\n")
        f.write(f"| Runner-up | `{gpu_row_label(runner_up) if runner_up else 'n/a'}` |\n")
        f.write(f"| Top-vs-runner-up gap | `{gap_pct:.3f}%` |\n" if gap_pct is not None else "| Top-vs-runner-up gap | `n/a` |\n")
        f.write(f"| Top-vs-current-default gap | `{default_gap_pct:.3f}%` |\n" if default_gap_pct is not None else "| Top-vs-current-default gap | `n/a` |\n")
        f.write(f"| Decision reason | {decision_reason} |\n")
        f.write("\n## Promotion Gates\n\n| Gate | Status |\n| --- | --- |\n")
        f.write(f"| All rows PASS | `{'yes' if all_pass else 'no'}` |\n")
        f.write(f"| Parseable repetitions per row | `{'yes' if all_parseable else 'no'}` |\n")
        f.write(f"| Top-vs-runner-up gap >= threshold | `{'yes' if gap_pass else 'no'}` |\n")
        f.write(f"| Current default present | `{'yes' if current_default else 'no'}` |\n")
        f.write(f"| Top-vs-current-default gap >= threshold | `{'yes' if default_gap_pass else 'no'}` |\n")
        f.write(f"| Top differs from current default | `{'yes' if not top_is_current else 'no'}` |\n")
        f.write("\n## Ranking\n\n")
        f.write("| Rank | Variant | Group M | Current Default | Status | Reps | Median TOPS | Mean TOPS | Stddev TOPS | CV TOPS % | Candidate delta % | Keep | Median Mean ms | Mean Mean ms | Stddev Mean ms | CV Mean ms % |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for rank, row in enumerate(ranked, start=1):
            f.write(
                f"| {rank} | `{row['variant']}` | {row['group_m']} | {'yes' if is_current_gpu_default(row) else 'no'} | {row['status']} | {row['repetitions']} | "
                f"{row['median_tops'] or 'n/a'} | {row['mean_tops'] or 'n/a'} | {row['stddev_tops'] or 'n/a'} | {row['cv_tops_pct'] or 'n/a'} | "
                f"{row['candidate_improvement_pct'] or 'n/a'} | {row['keep_candidate'] or 'n/a'} | "
                f"{row['median_mean_ms'] or 'n/a'} | {row['mean_mean_ms'] or 'n/a'} | {row['stddev_mean_ms'] or 'n/a'} | {row['cv_mean_ms_pct'] or 'n/a'} |\n"
            )
        f.write("\n## Gap Analysis\n\n")
        for note in gpu_gap_analysis_notes(top, runner_up, gap_pct):
            f.write(f"- {note}\n")
        if top and runner_up:
            f.write("\n### Dynamic/Normalized Counters\n\n")
            f.write("| Counter | Top | Runner-up | Runner-up - Top |\n| --- | --- | --- | --- |\n")
            for key in GPU_COMPARISON_COUNTER_KEYS:
                f.write(f"| {GPU_DYNAMIC_COUNTER_LABELS[key]} | {top.get(key) or 'n/a'} | {runner_up.get(key) or 'n/a'} | {counter_delta(runner_up, top, key)} |\n")
            f.write("\n### Static ISA Counters\n\n")
            f.write("| Counter | Top | Runner-up | Runner-up - Top |\n| --- | --- | --- | --- |\n")
            for key in GPU_STATIC_COUNTER_KEYS:
                f.write(f"| {GPU_STATIC_COUNTER_LABELS[key]} | {top.get(key) or 'n/a'} | {runner_up.get(key) or 'n/a'} | {counter_delta(runner_up, top, key)} |\n")
        f.write("\n## Artifacts\n\n")
        f.write(f"- Sweep CSV: `{csv_path}`\n")
        f.write(f"- Sweep report: `{sweep_md_path}`\n")
    return decision_path


def gpu_static_gate_status(row: dict[str, str]) -> str:
    vgprs = row_int(row, "vgprs")
    no_scratch = row_int(row, "scratch_markers") in (None, 0)
    no_vgpr_spills = row_int(row, "vgpr_spills") in (None, 0)
    no_sgpr_spills = row_int(row, "sgpr_spills") in (None, 0)
    no_spills = row_int(row, "spills") in (None, 0) and no_vgpr_spills and no_sgpr_spills
    vgpr_ok = vgprs is not None and vgprs < 256
    return "PASS" if row.get("status") == "PASS" and no_scratch and no_spills and vgpr_ok else "FAIL"


def write_gpu_stability_report(
    ctx: RunContext,
    rows: Sequence[dict[str, str]],
    repetitions: int,
    sweep_profile: str,
    rocblas_tensile_tops: float | None = None,
) -> tuple[Path, Path]:
    csv_path = ctx.out_dir / "gpu_stability_report.csv"
    md_path = ctx.out_dir / "gpu_stability_report.md"
    baseline = next((row for row in rows if is_accepted_gpu_best(row)), None)
    baseline_tops = row_float(baseline, "median_tops") if baseline else None
    fieldnames = (
        "sweep_profile",
        "variant",
        "group_m",
        "status",
        "static_gate",
        "repetitions",
        "tops_by_rep",
        "mean_ms_by_rep",
        "median_tops",
        "min_tops",
        "max_tops",
        "cv_tops_pct",
        "accepted_best_delta_pct",
        "rocblas_tensile_pct",
        "consistency_min_gate",
        "consistency_target_gate",
        "vgprs",
        "sgprs",
        "vgpr_spills",
        "sgpr_spills",
        "spills",
        "scratch_markers",
        "artifacts",
        "run_logs",
    )
    out_rows: list[dict[str, str]] = []
    for row in rows:
        median_tops = row_float(row, "median_tops")
        accepted_delta = ""
        if baseline_tops is not None and baseline_tops > 0.0 and median_tops is not None:
            accepted_delta = f"{((median_tops - baseline_tops) / baseline_tops) * 100.0:.3f}"
        rocblas_pct = ""
        if rocblas_tensile_tops is not None and rocblas_tensile_tops > 0.0 and median_tops is not None:
            rocblas_pct = f"{(median_tops / rocblas_tensile_tops) * 100.0:.3f}"
        reps = row_int(row, "repetitions") or 0
        pass_status = row.get("status") == "PASS"
        min_gate = bool(pass_status and reps == repetitions and median_tops is not None and median_tops >= GPU_INT8_GEMM_CONSISTENCY_MIN_TOPS)
        target_gate = bool(pass_status and reps == repetitions and median_tops is not None and median_tops >= GPU_INT8_GEMM_CONSISTENCY_TARGET_TOPS)
        out_rows.append({
            "sweep_profile": row.get("sweep_profile", sweep_profile),
            "variant": row.get("variant", ""),
            "group_m": row.get("group_m", ""),
            "status": row.get("status", ""),
            "static_gate": gpu_static_gate_status(row),
            "repetitions": row.get("repetitions", ""),
            "tops_by_rep": row.get("tops_by_rep", ""),
            "mean_ms_by_rep": row.get("mean_ms_by_rep", ""),
            "median_tops": row.get("median_tops", ""),
            "min_tops": row.get("min_tops", ""),
            "max_tops": row.get("max_tops", ""),
            "cv_tops_pct": row.get("cv_tops_pct", ""),
            "accepted_best_delta_pct": accepted_delta,
            "rocblas_tensile_pct": rocblas_pct,
            "consistency_min_gate": "yes" if min_gate else "no",
            "consistency_target_gate": "yes" if target_gate else "no",
            "vgprs": row.get("vgprs", ""),
            "sgprs": row.get("sgprs", ""),
            "vgpr_spills": row.get("vgpr_spills", ""),
            "sgpr_spills": row.get("sgpr_spills", ""),
            "spills": row.get("spills", ""),
            "scratch_markers": row.get("scratch_markers", ""),
            "artifacts": row.get("artifacts", ""),
            "run_logs": row.get("run_logs", ""),
        })
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    ranked = sorted(out_rows, key=lambda row: (-(row_float(row, "median_tops") or 0.0), row.get("variant", "")))
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# GPU INT8 GEMM Stability Report\n\n")
        f.write("## Gates\n\n| Field | Value |\n| --- | --- |\n")
        f.write(f"| Sweep profile | `{sweep_profile}` |\n")
        f.write(f"| Required repetitions | `{repetitions}` |\n")
        f.write(f"| Accepted best | `{GPU_INT8_GEMM_ACCEPTED_BEST_VARIANT} group_m={GPU_INT8_GEMM_ACCEPTED_BEST_GROUP_SIZE}` |\n")
        f.write(f"| Minimum consistency TOPS | `{GPU_INT8_GEMM_CONSISTENCY_MIN_TOPS:.2f}` |\n")
        f.write(f"| Preferred consistency TOPS | `{GPU_INT8_GEMM_CONSISTENCY_TARGET_TOPS:.2f}` |\n")
        f.write(f"| rocBLAS/Tensile median TOPS | `{rocblas_tensile_tops:.6f}` |\n" if rocblas_tensile_tops is not None else "| rocBLAS/Tensile median TOPS | `n/a` |\n")
        f.write("\n## Candidates\n\n")
        f.write("| Rank | Variant | Group M | Status | Static Gate | Reps | TOPS by Rep | Median | Min | Max | CV % | Accepted Delta % | rocBLAS/Tensile % | >=8.35 | >=8.50 | VGPRs | SGPRs | VGPR Spills | SGPR Spills | Scratch | Artifacts |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for rank, row in enumerate(ranked, start=1):
            f.write(
                f"| {rank} | `{row['variant']}` | {row['group_m']} | {row['status']} | {row['static_gate']} | {row['repetitions'] or '0'} | "
                f"`{row['tops_by_rep'] or 'n/a'}` | {row['median_tops'] or 'n/a'} | {row['min_tops'] or 'n/a'} | {row['max_tops'] or 'n/a'} | "
                f"{row['cv_tops_pct'] or 'n/a'} | {row['accepted_best_delta_pct'] or 'n/a'} | {row['rocblas_tensile_pct'] or 'n/a'} | "
                f"{row['consistency_min_gate']} | {row['consistency_target_gate']} | {row['vgprs'] or 'n/a'} | {row['sgprs'] or 'n/a'} | "
                f"{row['vgpr_spills'] or 'n/a'} | {row['sgpr_spills'] or 'n/a'} | {row['scratch_markers'] or 'n/a'} | `{row['artifacts'] or 'n/a'}` |\n"
            )
        f.write(f"\nCSV: `{csv_path}`\n")
    return csv_path, md_path


def run_gpu_variant_sweep(
    ctx: RunContext,
    sweep_profile: str,
    variants: Sequence[str],
    group_sizes: Sequence[int],
    repetitions: int,
    default_threshold_pct: float,
) -> BackendResult:
    rows: list[dict[str, str]] = []
    results: list[BackendResult] = []
    candidates = gpu_sweep_candidates(ctx, sweep_profile, variants, group_sizes)
    for variant, group_size in candidates:
        prefix = sanitize_prefix(f"{variant}_g{group_size}" if gpu_variant_config(variant).grouped_blocks else variant)
        variant_ctx = RunContext(
            ctx.repo,
            ctx.out_dir / "gpu_sweep" / prefix,
            ctx.build_root / "gpu_sweep" / prefix,
            ctx.logs_dir / "gpu_sweep" / prefix,
            ctx.warmups,
            ctx.iterations,
            ctx.gpu_arch,
            variant,
            group_size,
            False,
            ctx.cpu_threads,
            ctx.npu_runtime_loop_tiling,
        )
        for path in (variant_ctx.out_dir, variant_ctx.build_root, variant_ctx.logs_dir):
            path.mkdir(parents=True, exist_ok=True)
        result = gpu_backend(variant_ctx)
        runtime_failures = 0
        run_logs: list[Path] = []
        if ctx.run_enabled and result.status == "PASS":
            final_mlir = result.artifacts_dir / "gpu_int8_gemm.final.mlir"
            for rep in range(1, repetitions + 1):
                ok, log = run_gpu_final_mlir(variant_ctx, result, final_mlir, f"run_rep{rep}")
                if ok:
                    run_logs.append(log)
                else:
                    runtime_failures += 1
            if run_logs:
                apply_gpu_repetition_summary(
                    variant_ctx,
                    result,
                    summarize_gpu_repetition_metrics(run_logs),
                    variant,
                    group_size,
                    run_logs,
                )
            if runtime_failures:
                result.status = "WARN"
                result.runtime = f"{result.runtime}; {runtime_failures} repetition failure(s)"
                result.perf_notes = f"{result.perf_notes}; runtime_failures={runtime_failures}"
        results.append(result)
        evidence = evidence_map(result)
        metrics = summarize_gpu_repetition_metrics(run_logs) if run_logs else {}
        series = gpu_repetition_series(run_logs) if run_logs else {"tops_by_rep": "", "mean_ms_by_rep": ""}
        row = {
            "sweep_profile": sweep_profile,
            "variant": variant,
            "group_m": str(group_size),
            "status": result.status,
            "repetitions": str(int(metrics.get("repetitions", 0))) if metrics else "0",
            "median_mean_ms": fmt_float(metrics.get("median_mean_ms")),
            "min_mean_ms": fmt_float(metrics.get("min_mean_ms")),
            "mean_mean_ms": fmt_float(metrics.get("mean_mean_ms")),
            "stddev_mean_ms": fmt_float(metrics.get("stddev_mean_ms")),
            "cv_mean_ms_pct": fmt_float(metrics.get("cv_mean_ms_pct")),
            "best_kernel_min_ms": fmt_float(metrics.get("best_kernel_min_ms")),
            "median_tops": fmt_float(metrics.get("median_tops")),
            "mean_tops": fmt_float(metrics.get("mean_tops")),
            "stddev_tops": fmt_float(metrics.get("stddev_tops")),
            "cv_tops_pct": fmt_float(metrics.get("cv_tops_pct")),
            "min_tops": fmt_float(metrics.get("min_tops")),
            "max_tops": fmt_float(metrics.get("max_tops")),
            "timing_domain": result.perf_domain,
            "count": result.perf_count,
            "latency": result.perf_latency,
            "throughput": result.perf_throughput,
            "runtime": result.runtime,
            "artifacts": str(result.artifacts_dir),
            "run_logs": ";".join(str(log) for log in run_logs),
            "disassemble_log": str(result.logs.get("disassemble", "")),
            "tops_by_rep": series["tops_by_rep"],
            "mean_ms_by_rep": series["mean_ms_by_rep"],
        }
        for key in (*GPU_STATIC_COUNTER_KEYS, *GPU_DYNAMIC_COUNTER_KEYS):
            row[key] = evidence.get(key, "")
        rows.append(row)

    annotate_gpu_sweep_candidate_gates(rows)

    csv_path = ctx.out_dir / "gpu_variant_sweep.csv"
    md_path = ctx.out_dir / "gpu_variant_sweep.md"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["variant", "status"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    stability_csv_path, stability_md_path = write_gpu_stability_report(ctx, rows, repetitions, sweep_profile)

    retained_ranked = [
        result
        for result, row in zip(results, rows)
        if is_retained_gpu_candidate(row) and result.status == "PASS" and result.perf_tops is not None
    ]
    ranked = retained_ranked or [result for result in results if result.status == "PASS" and result.perf_tops is not None]
    if ranked:
        best = max(ranked, key=lambda result: result.perf_tops or 0.0)
        best_evidence = evidence_map(best)
        best_variant = best_evidence.get("variant", ctx.gpu_int8_gemm_variant)
        best_group = best_evidence.get("group_m", str(ctx.gpu_int8_gemm_group_size))
        best_label = f"{best_variant} group_m={best_group}"
        if retained_ranked:
            best_label += " (retained)"
    else:
        retained_passing = [
            result
            for result, row in zip(results, rows)
            if is_retained_gpu_candidate(row) and result.status == "PASS"
        ]
        passing = retained_passing or [result for result in results if result.status == "PASS"]
        best = passing[0] if passing else (results[0] if results else backend_result(ctx, "gpu"))
        best_label = "n/a (runtime disabled)" if not ctx.run_enabled else evidence_map(best).get("variant", ctx.gpu_int8_gemm_variant)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# GPU INT8 GEMM Variant Sweep\n\n")
        f.write(f"Best variant: `{best_label}`\n\n")
        f.write("| Variant | Group M | Status | Reps | Median TOPS | Mean TOPS | CV TOPS % | Candidate delta % | Keep | Dyn WMMA/wave | Dyn barriers | WMMA/barrier | Dyn waitcnt est | WMMA/waitcnt | VGPRs | Spills | Static waitcnt | Artifacts |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for row in rows:
            f.write(
                f"| `{row['variant']}` | {row['group_m']} | {row['status']} | {row['repetitions']} | "
                f"{row['median_tops'] or 'n/a'} | {row['mean_tops'] or 'n/a'} | {row['cv_tops_pct'] or 'n/a'} | "
                f"{row['candidate_improvement_pct'] or 'n/a'} | {row['keep_candidate'] or 'n/a'} | "
                f"{row['dynamic_wmma_per_wave'] or 'n/a'} | {row['dynamic_barriers'] or 'n/a'} | {row['wmma_per_barrier'] or 'n/a'} | "
                f"{row['dynamic_waitcnt_estimate'] or 'n/a'} | {row['wmma_per_waitcnt'] or 'n/a'} | "
                f"{row['vgprs'] or 'n/a'} | {row['spills'] or 'n/a'} | {row['waitcnt'] or 'n/a'} | `{row['artifacts']}` |\n"
            )
        f.write(f"\nCSV: `{csv_path}`\n")

    decision_path = write_gpu_default_decision(ctx, rows, repetitions, default_threshold_pct, sweep_profile, csv_path, md_path) if ctx.run_enabled else None
    best.runtime = f"{best.runtime}; sweep csv {csv_path}; sweep report {md_path}; stability csv {stability_csv_path}; stability report {stability_md_path}"
    if decision_path:
        best.runtime = f"{best.runtime}; default decision {decision_path}"
    best_row = next((row for result, row in zip(results, rows) if result is best), None)
    if best_row is not None:
        for key in ("candidate_improvement_pct", "keep_candidate"):
            if best_row.get(key):
                best.evidence = f"{best.evidence}, {key}={best_row[key]}"
    best.perf_notes = f"best_variant={best_label}; sweep_profile={sweep_profile}; sweep_candidates={len(rows)}; {best.perf_notes}"
    return best


def prepend_env_path(env: dict[str, str], key: str, path: Path) -> None:
    existing = env.get(key, "")
    entries = [str(path)]
    if existing:
        entries.append(existing)
    env[key] = os.pathsep.join(entries)


def rocm_root() -> Path:
    return Path(os.environ.get("ROCM_PATH", "/opt/rocm"))


def hipcc_path(rocm: Path) -> str | None:
    if os.environ.get("HIPCC"):
        return os.environ["HIPCC"]
    candidate = rocm / "bin" / "hipcc"
    if candidate.exists():
        return str(candidate)
    return shutil.which("hipcc")


def rocm_tool(rocm: Path, name: str) -> str | None:
    for candidate in (rocm / "bin" / name, rocm / "lib" / "llvm" / "bin" / name):
        if candidate.exists():
            return str(candidate)
    return shutil.which(name)


def gpu_provider_env(rocm: Path) -> dict[str, str]:
    env = os.environ.copy()
    lib = rocm / "lib"
    if lib.exists():
        prepend_env_path(env, "LD_LIBRARY_PATH", lib)
    return env


def provider_summary_line(log: Path, provider: str) -> dict[str, str]:
    for line in reversed(read_text(log).splitlines()):
        if f"provider={provider}" in line and "status=" in line:
            return dict(re.findall(r"([A-Za-z0-9_.-]+)=([^\s]+)", line))
    return {}


def provider_repetition_series(log: Path, provider: str) -> dict[str, str]:
    tops: list[float] = []
    mean_ms: list[float] = []
    for line in read_text(log).splitlines():
        if f"provider={provider}" not in line or "repetition=" not in line:
            continue
        fields = dict(re.findall(r"([A-Za-z0-9_.-]+)=([^\s]+)", line))
        if (value := parse_float(fields.get("tops", ""))) is not None:
            tops.append(value)
        if (value := parse_float(fields.get("mean_ms", ""))) is not None:
            mean_ms.append(value)
    return {
        "tops_by_rep": format_metric_series(tops),
        "mean_ms_by_rep": format_metric_series(mean_ms),
        "min_tops": fmt_float(min(tops) if tops else None),
    }


def provider_symbol_body(isa: str, symbol: str) -> str:
    lines = isa.splitlines()
    start = None
    for index, line in enumerate(lines):
        if symbol in line and re.search(r"^[0-9a-fA-F]+\s+<.*>:", line.strip()):
            start = index
            break
    if start is None:
        return ""
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if re.search(r"^[0-9a-fA-F]+\s+<.*>:", lines[index].strip()):
            end = index
            break
    return "\n".join(lines[start:end])


def provider_static_counters(provider: str, disassembly: dict[str, str]) -> dict[str, str]:
    if provider == "hip_wmma":
        body = provider_symbol_body(disassembly.get("isa", ""), "hipWmmaKernel")
    elif provider == "rocwmma":
        body = provider_symbol_body(disassembly.get("isa", ""), "rocwmmaKernel")
    elif provider == "air_tuned":
        body = provider_symbol_body(disassembly.get("isa", ""), "airTuned128x128Kernel")
    else:
        body = ""
    if not body:
        return {}
    return {
        "wmma": str(count_regex(body, r"\bv_wmma_i32_16x16x16_iu8\b")),
        "global_load_b128": str(count_regex(body, r"\bglobal_load_b128\b")),
        "global_load_u8": str(count_regex(body, r"\bglobal_load(?:_d16(?:_hi)?)?_u8\b")),
        "ds_read_b128": str(count_regex(body, r"\bds_(?:read|load)_b128\b")),
        "ds_swizzle": str(count_regex(body, r"\bds_swizzle_b32\b")),
        "global_store_b32": str(count_regex(body, r"\bglobal_store_b32\b")),
        "barriers": str(count_regex(body, r"\bs_barrier\b")),
        "waitcnt": str(count_regex(body, r"\bs_waitcnt\b")),
        "scratch_markers": str(count_regex(body, r"scratch")),
        "spills": "0" if "scratch" not in body else "unknown",
    }


def generic_gpu_static_counters(isa_text: str, metadata_text: str = "") -> dict[str, str]:
    scratch_markers = count_regex(isa_text, r"\bscratch\b") + count_regex(metadata_text, r"uses_flat_scratch:\s*1|uses_flat_scratch\s+1")
    vgpr_spill_matches = re.findall(r"\.vgpr_spill_count:\s*([0-9]+)|vgpr_spill_count\s*=\s*([0-9]+)", metadata_text)
    sgpr_spill_matches = re.findall(r"\.sgpr_spill_count:\s*([0-9]+)|sgpr_spill_count\s*=\s*([0-9]+)", metadata_text)
    vgpr_spills = next((value for match in reversed(vgpr_spill_matches) for value in match if value), "")
    sgpr_spills = next((value for match in reversed(sgpr_spill_matches) for value in match if value), "")
    spill_markers = count_regex(metadata_text, r"spill_count:\s*[1-9]|_spill_count:\s*[1-9]|spill_count = [1-9]")
    if vgpr_spills and sgpr_spills:
        spill_markers = int(vgpr_spills) + int(sgpr_spills)
    return {
        "wmma": str(count_regex(isa_text, r"\bv_wmma_i32_16x16x16_iu8\b")),
        "global_load_b128": str(count_regex(isa_text, r"\b(?:global|buffer)_load_b128\b")),
        "global_load_u8": str(count_regex(isa_text, r"\b(?:global|buffer)_load(?:_d16(?:_hi)?)?_u8\b")),
        "ds_read_b128": str(count_regex(isa_text, r"\bds_(?:read|load)_b128\b")),
        "ds_swizzle": str(count_regex(isa_text, r"\bds_swizzle_b32\b")),
        "global_store_b32": str(count_regex(isa_text, r"\b(?:global|buffer)_store_b32\b")),
        "barriers": str(count_regex(isa_text, r"\bs_barrier\b")),
        "waitcnt": str(count_regex(isa_text, r"\bs_waitcnt\b")),
        "scratch_markers": str(scratch_markers),
        "vgpr_spills": vgpr_spills,
        "sgpr_spills": sgpr_spills,
        "spills": str(spill_markers),
    }


def first_existing(paths: Sequence[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def first_glob(root: Path, patterns: Sequence[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    return None


def rocmlir_tool(bin_dir: Path | None, name: str) -> str | None:
    if bin_dir:
        candidate = bin_dir / name
        if candidate.exists():
            return str(candidate)
    return shutil.which(name)


def rocmlir_reference_paths(artifacts_dir: Path | None, provider_artifacts_dir: Path) -> dict[str, Path | None]:
    roots = [path for path in (artifacts_dir, provider_artifacts_dir) if path and path.exists() and any(path.iterdir())]
    result: dict[str, Path | None] = {"root": artifacts_dir if artifacts_dir and artifacts_dir.exists() and any(artifacts_dir.iterdir()) else None}
    if not roots:
        return result
    search_root = roots[0]
    result["root"] = search_root
    result["isa"] = first_existing(
        [
            search_root / "rocmlir_reference.isa.s",
            search_root / "rocmlir.isa.s",
            search_root / "kernel.isa.s",
        ]
    ) or first_glob(search_root, ("*.isa.s", "*.s"))
    result["hsaco"] = first_existing(
        [
            search_root / "rocmlir_reference.hsaco",
            search_root / "rocmlir.hsaco",
            search_root / "kernel.hsaco",
        ]
    ) or first_glob(search_root, ("*.hsaco", "*.co"))
    result["readobj"] = first_existing(
        [
            search_root / "rocmlir_reference.readobj.txt",
            search_root / "rocmlir.readobj.txt",
            search_root / "kernel.readobj.txt",
        ]
    ) or first_glob(search_root, ("*.readobj.txt", "*readobj*.txt", "*metadata*.txt"))
    result["mlir"] = first_existing(
        [
            search_root / "rocmlir_reference.mlir",
            search_root / "rocmlir.mlir",
            search_root / "kernel.mlir",
        ]
    ) or first_glob(search_root, ("*.mlir",))
    result["profile"] = first_glob(search_root, ("*kernel_stats.csv", "*profile*.csv", "*rocprof*.csv"))
    return result


def materialize_rocmlir_reference_artifacts(
    ctx: RunContext,
    result: BackendResult,
    paths: dict[str, Path | None],
    rocm: Path,
    target_chip: str,
) -> tuple[Path | None, Path | None, list[str]]:
    notes: list[str] = []
    isa_path = paths.get("isa")
    readobj_path = paths.get("readobj")
    hsaco = paths.get("hsaco")
    if not hsaco or not hsaco.exists():
        return isa_path, readobj_path, notes
    if not isa_path or not isa_path.exists():
        llvm_objdump = rocm_tool(rocm, "llvm-objdump")
        if llvm_objdump:
            isa_path = result.artifacts_dir / "rocmlir_reference.isa.s"
            ok, _ = run_capture(isa_path, [llvm_objdump, "-d", f"--mcpu={target_chip}", hsaco], env=gpu_provider_env(rocm))
            if ok:
                notes.append(f"disassembled_hsaco={hsaco}")
            else:
                notes.append(f"rocMLIR HSACO disassembly failed for {hsaco}")
        else:
            notes.append("llvm-objdump unavailable; rocMLIR HSACO was not disassembled")
    if not readobj_path or not readobj_path.exists():
        llvm_readobj = rocm_tool(rocm, "llvm-readobj")
        if llvm_readobj:
            readobj_path = result.artifacts_dir / "rocmlir_reference.readobj.txt"
            ok, _ = run_capture(readobj_path, [llvm_readobj, "--file-headers", "--notes", "--sections", "--symbols", hsaco], env=gpu_provider_env(rocm))
            if ok:
                notes.append(f"readobj_hsaco={hsaco}")
            else:
                notes.append(f"rocMLIR HSACO readobj failed for {hsaco}")
        else:
            notes.append("llvm-readobj unavailable; rocMLIR metadata was not extracted")
    return isa_path, readobj_path, notes


def rocmlir_reference_row(
    ctx: RunContext,
    bin_dir: Path | None,
    artifacts_dir: Path | None,
    target_chip: str,
) -> dict[str, str]:
    result = backend_result(ctx, "gpu_rocmlir_reference", True)
    rocm = rocm_root()
    gen = rocmlir_tool(bin_dir, "rocmlir-gen")
    driver = rocmlir_tool(bin_dir, "rocmlir-driver")
    paths = rocmlir_reference_paths(artifacts_dir, result.artifacts_dir)
    root = paths.get("root")
    if not root:
        notes = "rocMLIR artifacts not found; pass --rocmlir-artifacts-dir with ISA, HSACO, readobj, or MLIR artifacts"
        if bin_dir:
            tool_state = f"rocmlir-gen={'yes' if gen else 'no'}, rocmlir-driver={'yes' if driver else 'no'}"
            notes = f"{notes}; --rocmlir-bin-dir={bin_dir}; {tool_state}; generation is intentionally not used as a provider path"
        return provider_empty_row(GPU_ROCMLIR_REFERENCE_PROVIDER, GPU_ROCMLIR_REFERENCE_SOURCE, "SKIP", "no", notes)
    isa_path, readobj_path, notes = materialize_rocmlir_reference_artifacts(ctx, result, paths, rocm, target_chip)
    row = provider_empty_row(GPU_ROCMLIR_REFERENCE_PROVIDER, GPU_ROCMLIR_REFERENCE_SOURCE, "PASS", "yes", "")
    row["validation"] = "n/a"
    row["artifacts"] = str(root)
    if isa_path:
        row["disassemble_log"] = str(isa_path)
    if readobj_path:
        row["profile_log"] = str(readobj_path)
    if paths.get("profile"):
        row["profile_log"] = ";".join(item for item in (row.get("profile_log", ""), str(paths["profile"])) if item)
    if paths.get("mlir"):
        notes.append(f"mlir={paths['mlir']}")
    if paths.get("hsaco"):
        notes.append(f"hsaco={paths['hsaco']}")
    if bin_dir:
        notes.append(f"rocmlir_bin_dir={bin_dir}")
    notes.append(f"target_chip={target_chip}")
    isa_text = read_text(isa_path) if isa_path else ""
    readobj_text = read_text(readobj_path) if readobj_path else ""
    if isa_text:
        row.update(generic_gpu_static_counters(isa_text, readobj_text))
        if row.get("wmma") in {"", "0"}:
            row["status"] = "WARN"
            notes.append("rocMLIR ISA did not contain v_wmma_i32_16x16x16_iu8")
    else:
        row["status"] = "WARN"
        notes.append("rocMLIR ISA not found; row contains metadata/artifact paths only")
    row["notes"] = "; ".join(notes)
    provider_target_fields(row)
    return row


def write_gpu_rocmlir_reference_report(ctx: RunContext, row: dict[str, str]) -> tuple[Path, Path]:
    csv_path = ctx.out_dir / "gpu_rocmlir_reference.csv"
    md_path = ctx.out_dir / "gpu_rocmlir_reference.md"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(GPU_PROVIDER_BASELINE_FIELDNAMES))
        writer.writeheader()
        writer.writerow(row)
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# GPU INT8 GEMM rocMLIR Reference\n\n")
        f.write("This row is static compiler-reference evidence. It is not a runtime provider and it is not an MLIR-AIR fallback path.\n\n")
        f.write("| Provider | Source | Status | Available | WMMA | b128 Loads | LDS Reads | Scratch | Artifacts | Notes |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        f.write(
            f"| `{row['provider']}` | {row['source']} | {row['status']} | {row['available']} | "
            f"{row['wmma'] or 'n/a'} | {row['global_load_b128'] or 'n/a'} | {row['ds_read_b128'] or 'n/a'} | "
            f"{row['scratch_markers'] or 'n/a'} | `{row['artifacts'] or 'n/a'}` | {row['notes'] or 'n/a'} |\n"
        )
        f.write(f"\nCSV: `{csv_path}`\n")
    return csv_path, md_path


def provider_metric(row: dict[str, str], key: str) -> float | None:
    return parse_float(row.get(key, ""))


def provider_target_fields(row: dict[str, str]) -> None:
    median_tops = provider_metric(row, "median_tops")
    median_ms = provider_metric(row, "median_mean_ms")
    row["target_tops_2x"] = f"{GPU_PROVIDER_2X_TARGET_TOPS:.6f}"
    row["ideal_bytes"] = str(GEMM_INT8_IDEAL_BYTES)
    row["operational_intensity_ops_per_byte"] = f"{GEMM_INT8_OPS / GEMM_INT8_IDEAL_BYTES:.6f}"
    if median_tops is None:
        row["target_pct_2x"] = ""
        row["meets_2x"] = "no"
        row["ideal_bandwidth_gbs"] = ""
        return
    row["target_pct_2x"] = f"{(median_tops / GPU_PROVIDER_2X_TARGET_TOPS) * 100.0:.3f}"
    row["meets_2x"] = "yes" if median_tops >= GPU_PROVIDER_2X_TARGET_TOPS else "no"
    if median_ms is not None and median_ms > 0.0:
        row["ideal_bandwidth_gbs"] = f"{GEMM_INT8_IDEAL_BYTES / (median_ms * 1.0e-3) / 1.0e9:.6f}"
    else:
        row["ideal_bandwidth_gbs"] = ""


def provider_empty_row(provider: str, source: str, status: str, available: str, notes: str) -> dict[str, str]:
    row = {key: "" for key in GPU_PROVIDER_BASELINE_FIELDNAMES}
    row.update({"provider": provider, "source": source, "status": status, "available": available, "notes": notes})
    provider_target_fields(row)
    return row


def gpu_result_validation(result: BackendResult) -> str:
    for log in result.logs.values():
        text = read_text(log)
        if "INT8 Output Mismatches" in text or "mismatch" in text.lower():
            return "FAIL"
        if "INT8 Output Matched" in text:
            return "PASS"
    return "" if result.perf_tops is None else "PASS"


def gpu_result_to_provider_row(result: BackendResult) -> dict[str, str]:
    evidence = evidence_map(result)
    variant = evidence.get("variant", DEFAULT_GPU_INT8_GEMM_VARIANT)
    group = evidence.get("group_m", str(DEFAULT_GPU_INT8_GEMM_GROUP_SIZE))
    row = provider_empty_row(f"mlir_air:{variant}:g{group}", "mlir-air", result.status, "yes", result.perf_notes)
    row["validation"] = gpu_result_validation(result)
    run_logs = [path for stem, path in sorted(result.logs.items()) if stem.startswith("run")]
    row["run_log"] = ";".join(str(path) for path in run_logs)
    row["disassemble_log"] = str(result.logs.get("disassemble", ""))
    row["artifacts"] = str(result.artifacts_dir)
    if result.perf_tops is not None:
        row["median_tops"] = f"{result.perf_tops:.6f}"
    if run_logs:
        row.update(gpu_repetition_series(run_logs))
        metrics = summarize_gpu_repetition_metrics(run_logs)
        for key in ("min_tops", "max_tops", "mean_tops", "stddev_tops", "cv_tops_pct", "min_mean_ms", "mean_mean_ms", "stddev_mean_ms", "cv_mean_ms_pct", "best_kernel_min_ms"):
            row[key] = fmt_float(metrics.get(key))
    median_ms = re.search(r"median mean ([0-9.]+) ms", result.perf_latency)
    if median_ms:
        row["median_mean_ms"] = median_ms.group(1)
    reps = re.match(r"([0-9]+)x", result.perf_count)
    if reps:
        row["repetitions"] = reps.group(1)
    for key in (*GPU_STATIC_COUNTER_KEYS, *GPU_DYNAMIC_COUNTER_KEYS):
        if key in evidence and key in row:
            row[key] = evidence[key]
    for key in ("candidate_improvement_pct", "keep_candidate"):
        if key in evidence and key in row:
            row[key] = evidence[key]
    provider_target_fields(row)
    return row


def build_gpu_provider_binary(ctx: RunContext, result: BackendResult, rocm: Path) -> tuple[bool, Path, Path | None]:
    source = ctx.repo / "benchmarks" / "gemm_int8" / "providers" / "hip_int8_gemm_baseline.cpp"
    binary = result.build_dir / GPU_PROVIDER_EXECUTABLE
    hipcc = hipcc_path(rocm)
    log = log_path(ctx, result, "provider_build")
    if not hipcc:
        write_text(log, "ERROR: hipcc not found\n")
        return False, log, None
    argv = [
        hipcc,
        f"--rocm-path={rocm}",
        "-O3",
        f"--offload-arch={ctx.gpu_arch}",
        "-std=c++17",
        "-isystem",
        rocm / "include",
        source,
        "-L",
        rocm / "lib",
        f"-Wl,-rpath,{rocm / 'lib'}",
        "-lrocblas",
        "-o",
        binary,
    ]
    ok, log = run_capture(log, argv, env=gpu_provider_env(rocm))
    return ok and binary.exists(), log, binary if binary.exists() else None


def collect_gpu_provider_disassembly(ctx: RunContext, result: BackendResult, binary: Path, rocm: Path) -> dict[str, str]:
    roc_obj_ls = rocm_tool(rocm, "roc-obj-ls")
    roc_obj_extract = rocm_tool(rocm, "roc-obj-extract")
    llvm_objdump = rocm_tool(rocm, "llvm-objdump")
    llvm_readobj = rocm_tool(rocm, "llvm-readobj")
    if not roc_obj_ls or not roc_obj_extract or not llvm_objdump:
        return {"notes": "ROCm code-object extraction tools unavailable"}
    list_log = log_path(ctx, result, "provider_code_objects")
    ok, _ = run_capture(list_log, [roc_obj_ls, binary], env=gpu_provider_env(rocm))
    if not ok:
        return {"list_log": str(list_log), "notes": "roc-obj-ls failed"}
    uris = []
    for line in read_text(list_log).splitlines():
        if "hipv4" in line and f"--{ctx.gpu_arch}" in line:
            uris.append(line.split()[-1])
    extract_dir = result.artifacts_dir / "code_objects"
    extract_dir.mkdir(parents=True, exist_ok=True)
    for index, uri in enumerate(uris):
        run_logged(ctx, result, f"provider_extract_{index}", [roc_obj_extract, "-o", extract_dir, uri], env=gpu_provider_env(rocm))
    isa_parts: list[str] = []
    disasm_paths: list[str] = []
    readobj_paths: list[str] = []
    for index, code_object in enumerate(sorted(extract_dir.glob("*.co"))):
        disasm = result.artifacts_dir / f"provider_{index}.isa.s"
        ok, _ = run_capture(disasm, [llvm_objdump, "-d", f"--mcpu={ctx.gpu_arch}", code_object], env=gpu_provider_env(rocm))
        if ok:
            isa_parts.append(read_text(disasm))
            disasm_paths.append(str(disasm))
        if llvm_readobj:
            readobj = result.artifacts_dir / f"provider_{index}.readobj.txt"
            ok, _ = run_capture(readobj, [llvm_readobj, "--file-headers", "--notes", "--sections", "--symbols", code_object], env=gpu_provider_env(rocm))
            if ok:
                readobj_paths.append(str(readobj))
    return {"isa": "\n".join(isa_parts), "disassemble_log": ";".join(disasm_paths), "readobj_log": ";".join(readobj_paths), "list_log": str(list_log)}


def run_gpu_provider_profile(ctx: RunContext, result: BackendResult, binary: Path, provider: str, rocm: Path, validation_samples: int) -> str:
    rocprof = rocm_tool(rocm, "rocprofv3")
    if not rocprof:
        return ""
    profile_dir = result.artifacts_dir / "profiles" / provider
    profile_dir.mkdir(parents=True, exist_ok=True)
    log = log_path(ctx, result, f"provider_profile_{sanitize_prefix(provider)}")
    argv = [
        rocprof,
        "--kernel-trace",
        "--stats",
        "--summary",
        "--output-format",
        "csv",
        "--output-directory",
        profile_dir,
        "--output-file",
        provider,
        "--",
        binary,
        "--provider",
        provider,
        "--warmups",
        ctx.warmups,
        "--iterations",
        ctx.iterations,
        "--repetitions",
        1,
        "--validation-samples",
        validation_samples,
    ]
    run_capture(log, argv, env=gpu_provider_env(rocm))
    return str(log)


def run_gpu_provider_binary(ctx: RunContext, result: BackendResult, binary: Path, provider: str, rocm: Path, repetitions: int, validation_samples: int, profile: bool, disassembly: dict[str, str], build_log: Path) -> dict[str, str]:
    log = log_path(ctx, result, f"provider_run_{sanitize_prefix(provider)}")
    argv = [
        binary,
        "--provider",
        provider,
        "--warmups",
        ctx.warmups,
        "--iterations",
        ctx.iterations,
        "--repetitions",
        repetitions,
        "--validation-samples",
        validation_samples,
    ]
    ok, _ = run_capture(log, argv, env=gpu_provider_env(rocm))
    summary = provider_summary_line(log, provider)
    if not summary:
        row = provider_empty_row(provider, "external", "WARN" if ok else "FAIL", "yes", f"summary line missing; see {log}")
    else:
        source = "air-owned" if provider == "air_tuned" else "external"
        row = provider_empty_row(provider, source, summary.get("status", "PASS" if ok else "FAIL"), "yes", "")
        for key in (
            "validation",
            "mismatches",
            "repetitions",
            "median_mean_ms",
            "min_mean_ms",
            "mean_mean_ms",
            "stddev_mean_ms",
            "cv_mean_ms_pct",
            "best_kernel_min_ms",
            "median_tops",
            "mean_tops",
            "stddev_tops",
            "cv_tops_pct",
            "max_tops",
        ):
            row[key] = summary.get(key, "")
        row.update(provider_repetition_series(log, provider))
    row["build_log"] = str(build_log)
    row["run_log"] = str(log)
    row["artifacts"] = str(result.artifacts_dir)
    row["disassemble_log"] = disassembly.get("disassemble_log", "")
    row.update(provider_static_counters(provider, disassembly))
    if provider == "rocblas_tensile":
        row["notes"] = "rocBLAS GEMM_EX uses installed Tensile libraries; harness binary does not contain the library kernel ISA"
    elif provider == "air_tuned":
        row["notes"] = "AIR-owned fixed-contract raw WMMA kernel: 128x128 macro tile, 64x64 wave tile, four waves, grouped-M launch, host-packed B"
    elif not row.get("notes"):
        row["notes"] = disassembly.get("notes", "")
    if profile and row["status"] == "PASS":
        row["profile_log"] = run_gpu_provider_profile(ctx, result, binary, provider, rocm, validation_samples)
    provider_target_fields(row)
    return row


def annotate_gpu_provider_air_tuned_gates(rows: Sequence[dict[str, str]]) -> None:
    air_tuned = next((row for row in rows if row.get("provider") == "air_tuned"), None)
    tuned_tops = provider_metric(air_tuned, "median_tops") if air_tuned else None
    rocblas = next((row for row in rows if row.get("provider") == "rocblas_tensile"), None)
    rocblas_tops = provider_metric(rocblas, "median_tops") if rocblas else None
    for row in rows:
        row.setdefault("mlir_air_pct_of_air_tuned", "")
        row.setdefault("passes_air_tuned_95pct", "")
        row.setdefault("mlir_air_pct_of_rocblas_tensile", "")
        row.setdefault("candidate_improvement_pct", "")
        row.setdefault("keep_candidate", "")
        if row.get("source") != "mlir-air":
            continue
        mlir_tops = provider_metric(row, "median_tops")
        if mlir_tops is not None and tuned_tops is not None and tuned_tops > 0.0:
            pct = (mlir_tops / tuned_tops) * 100.0
            row["mlir_air_pct_of_air_tuned"] = f"{pct:.3f}"
            row["passes_air_tuned_95pct"] = "yes" if pct >= GPU_AIR_TUNED_ACCEPTANCE_PCT else "no"
        if mlir_tops is not None and rocblas_tops is not None and rocblas_tops > 0.0:
            row["mlir_air_pct_of_rocblas_tensile"] = f"{(mlir_tops / rocblas_tops) * 100.0:.3f}"


def write_gpu_provider_baseline_report(ctx: RunContext, rows: Sequence[dict[str, str]], repetitions: int) -> tuple[Path, Path]:
    annotate_gpu_provider_air_tuned_gates(rows)
    csv_path = ctx.out_dir / "gpu_provider_baselines.csv"
    md_path = ctx.out_dir / "gpu_provider_baselines.md"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(GPU_PROVIDER_BASELINE_FIELDNAMES))
        writer.writeheader()
        writer.writerows(rows)
    ranked = sorted(
        [row for row in rows if row.get("status") == "PASS" and row.get("validation") in {"", "PASS"} and provider_metric(row, "median_tops") is not None],
        key=lambda row: -(provider_metric(row, "median_tops") or 0.0),
    )
    top = ranked[0] if ranked else None
    runner_up = ranked[1] if len(ranked) > 1 else None
    gap_pct = median_tops_gap_pct(top, runner_up) if top and runner_up else None
    cv_ceiling = max(provider_metric(top, "cv_tops_pct") or 0.0, provider_metric(runner_up, "cv_tops_pct") or 0.0) if top and runner_up else 0.0
    reaches_2x = bool(top and (provider_metric(top, "median_tops") or 0.0) >= GPU_PROVIDER_2X_TARGET_TOPS)
    stable_gap = bool(gap_pct is not None and gap_pct >= 3.0 and gap_pct >= cv_ceiling)
    air_ranked = [row for row in ranked if row.get("source") == "air-owned"]
    best_air = air_ranked[0] if air_ranked else None
    rocblas = next((row for row in rows if row.get("provider") == "rocblas_tensile" and provider_metric(row, "median_tops") is not None), None)
    air_vs_rocblas_pct = None
    if best_air and rocblas and provider_metric(rocblas, "median_tops"):
        air_vs_rocblas_pct = ((provider_metric(best_air, "median_tops") or 0.0) / (provider_metric(rocblas, "median_tops") or 1.0)) * 100.0
    mlir_air = next((row for row in rows if row.get("source") == "mlir-air"), None)
    mlir_air_vs_rocblas_pct = None
    if mlir_air and rocblas and provider_metric(rocblas, "median_tops"):
        mlir_air_vs_rocblas_pct = ((provider_metric(mlir_air, "median_tops") or 0.0) / (provider_metric(rocblas, "median_tops") or 1.0)) * 100.0
    air_reaches_2x = bool(best_air and (provider_metric(best_air, "median_tops") or 0.0) >= GPU_PROVIDER_2X_TARGET_TOPS)
    air_reaches_rocblas_parity = bool(air_vs_rocblas_pct is not None and air_vs_rocblas_pct >= GPU_PROVIDER_ROCBLAS_PARITY_PCT)
    mlir_air_reaches_rocblas_parity = bool(mlir_air_vs_rocblas_pct is not None and mlir_air_vs_rocblas_pct >= GPU_PROVIDER_ROCBLAS_PARITY_PCT)
    if best_air and top and top.get("provider") == best_air.get("provider") and air_reaches_2x and air_reaches_rocblas_parity and stable_gap:
        recommendation = "AIR_TUNED_PROVIDER_CANDIDATE"
    elif air_reaches_2x and not air_reaches_rocblas_parity:
        recommendation = "AIR_TUNED_2X_BUT_ROCBLAS_STILL_LEADS"
    elif reaches_2x and stable_gap:
        recommendation = "EXTERNAL_PROVIDER_CANDIDATE"
    else:
        recommendation = "NO_STABLE_2X_PROVIDER"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# GPU INT8 GEMM Provider Baselines\n\n")
        f.write(f"Recommendation: `{recommendation}`\n\n")
        f.write("## Gates\n\n| Field | Value |\n| --- | --- |\n")
        f.write(f"| Required repetitions | `{repetitions}` |\n")
        f.write(f"| 2x target TOPS | `{GPU_PROVIDER_2X_TARGET_TOPS:.6f}` |\n")
        f.write(f"| Top provider | `{top['provider'] if top else 'n/a'}` |\n")
        f.write(f"| Runner-up | `{runner_up['provider'] if runner_up else 'n/a'}` |\n")
        f.write(f"| Top-vs-runner-up gap | `{gap_pct:.3f}%` |\n" if gap_pct is not None else "| Top-vs-runner-up gap | `n/a` |\n")
        f.write(f"| CV ceiling | `{cv_ceiling:.3f}%` |\n")
        f.write(f"| Stable >=3% gap | `{'yes' if stable_gap else 'no'}` |\n")
        f.write(f"| Top reaches 2x | `{'yes' if reaches_2x else 'no'}` |\n")
        f.write(f"| Best AIR-owned provider | `{best_air['provider'] if best_air else 'n/a'}` |\n")
        f.write(f"| Best AIR-owned reaches 2x | `{'yes' if air_reaches_2x else 'no'}` |\n")
        f.write(f"| AIR-owned / rocBLAS TOPS | `{air_vs_rocblas_pct:.3f}%` |\n" if air_vs_rocblas_pct is not None else "| AIR-owned / rocBLAS TOPS | `n/a` |\n")
        f.write(f"| AIR-owned reaches {GPU_PROVIDER_ROCBLAS_PARITY_PCT:.1f}% rocBLAS | `{'yes' if air_reaches_rocblas_parity else 'no'}` |\n")
        f.write(f"| MLIR-AIR / rocBLAS TOPS | `{mlir_air_vs_rocblas_pct:.3f}%` |\n" if mlir_air_vs_rocblas_pct is not None else "| MLIR-AIR / rocBLAS TOPS | `n/a` |\n")
        f.write(f"| MLIR-AIR reaches {GPU_PROVIDER_ROCBLAS_PARITY_PCT:.1f}% rocBLAS | `{'yes' if mlir_air_reaches_rocblas_parity else 'no'}` |\n")
        mlir_air_pct = mlir_air.get("mlir_air_pct_of_air_tuned", "") if mlir_air else ""
        mlir_air_gate = mlir_air.get("passes_air_tuned_95pct", "") if mlir_air else ""
        f.write("| MLIR-AIR / air_tuned TOPS | `{}` |\n".format((mlir_air_pct + "%") if mlir_air_pct else "n/a"))
        f.write("| MLIR-AIR reaches {:.1f}% air_tuned | `{}` |\n".format(GPU_AIR_TUNED_ACCEPTANCE_PCT, "yes" if mlir_air_gate == "yes" else "no"))
        f.write("\n## Results\n\n")
        f.write("| Provider | Source | Status | Validation | Reps | TOPS by Rep | Median ms | Median TOPS | 2x Target % | rocBLAS % | WMMA | Scratch | Notes |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for row in rows:
            rocblas_pct = row.get("mlir_air_pct_of_rocblas_tensile", "") if row.get("source") == "mlir-air" else ""
            f.write(
                f"| `{row['provider']}` | {row['source']} | {row['status']} | {row['validation'] or 'n/a'} | {row['repetitions'] or '0'} | "
                f"`{row['tops_by_rep'] or 'n/a'}` | {row['median_mean_ms'] or 'n/a'} | {row['median_tops'] or 'n/a'} | {row['target_pct_2x'] or 'n/a'} | "
                f"{rocblas_pct or 'n/a'} | {row['wmma'] or 'n/a'} | {row['scratch_markers'] or 'n/a'} | {row['notes'] or 'n/a'} |\n"
            )
        f.write("\n## Roofline Inputs\n\n")
        f.write(f"Ideal bytes per GEMM: `{GEMM_INT8_IDEAL_BYTES}`\n\n")
        f.write(f"Ideal operational intensity: `{GEMM_INT8_OPS / GEMM_INT8_IDEAL_BYTES:.6f}` ops/byte\n\n")
        f.write("| Provider | Median TOPS | Ideal GB/s At Median | Bottleneck Classification |\n")
        f.write("| --- | --- | --- | --- |\n")
        for row in rows:
            classification = "unclassified"
            if row.get("provider") == "hip_wmma" and provider_metric(row, "global_load_u8"):
                classification = "B operand global-gather pressure"
            elif row.get("provider") == "air_tuned" and row.get("meets_2x") == "yes":
                classification = "AIR-owned candidate passes 2x throughput gate; compare against rocBLAS before promotion"
            elif row.get("provider") == "air_tuned":
                classification = "AIR-owned candidate below 2x throughput gate"
            elif row.get("meets_2x") == "yes":
                classification = "passes 2x throughput gate; use profiler counters before promotion"
            elif provider_metric(row, "median_tops") is not None:
                classification = "below 2x throughput gate"
            f.write(f"| `{row['provider']}` | {row['median_tops'] or 'n/a'} | {row['ideal_bandwidth_gbs'] or 'n/a'} | {classification} |\n")
        f.write(f"\nCSV: `{csv_path}`\n")
    return csv_path, md_path


def run_gpu_provider_baselines(
    ctx: RunContext,
    gpu_result: BackendResult,
    repetitions: int,
    validation_samples: int,
    profile: bool,
    rocmlir_row: dict[str, str] | None = None,
) -> tuple[Path, Path]:
    result = backend_result(ctx, "gpu_provider_baselines", True)
    rocm = rocm_root()
    rows: list[dict[str, str]] = [gpu_result_to_provider_row(gpu_result)]
    ok, build_log, binary = build_gpu_provider_binary(ctx, result, rocm)
    if ok and binary:
        disassembly = collect_gpu_provider_disassembly(ctx, result, binary, rocm)
        for provider in ("hip_wmma", "rocwmma", "air_tuned", "rocblas_tensile"):
            rows.append(run_gpu_provider_binary(ctx, result, binary, provider, rocm, repetitions, validation_samples, profile, disassembly, build_log))
    else:
        for provider in ("hip_wmma", "rocwmma", "air_tuned", "rocblas_tensile"):
            row = provider_empty_row(provider, "external", "WARN", "unknown", f"provider build failed; see {build_log}")
            row["build_log"] = str(build_log)
            rows.append(row)
    if rocmlir_row is not None:
        rows.append(rocmlir_row)
    ck_available = (rocm / "include" / "ck_tile").exists() or (rocm / "include" / "ck").exists()
    ck_notes = "CK/CK-Tile headers found; fixed-contract CK driver is not wired into this harness yet" if ck_available else "CK/CK-Tile headers not found"
    rows.append(provider_empty_row("ck_tile", "external", "SKIP", "yes" if ck_available else "no", ck_notes))
    provider_paths = write_gpu_provider_baseline_report(ctx, rows, repetitions)
    sweep_csv = ctx.out_dir / "gpu_variant_sweep.csv"
    if sweep_csv.exists():
        with sweep_csv.open("r", encoding="utf-8", newline="") as f:
            sweep_rows = list(csv.DictReader(f))
        if sweep_rows:
            rocblas = next((row for row in rows if row.get("provider") == "rocblas_tensile"), None)
            rocblas_tops = provider_metric(rocblas, "median_tops") if rocblas else None
            write_gpu_stability_report(ctx, sweep_rows, repetitions, sweep_rows[0].get("sweep_profile", "from gpu_variant_sweep.csv"), rocblas_tops)
    return provider_paths

def find_npu_elves(build_dir: Path) -> list[Path]:
    return sorted(build_dir.rglob("bare_matmul*_core_*.elf")) or sorted(build_dir.rglob("*.elf"))


def npu_env(ctx: RunContext) -> dict[str, str]:
    env = os.environ.copy()
    if not env.get("PYTHON") and (candidate := ctx.repo / "sandbox" / "bin" / "python3").exists():
        env["PYTHON"] = str(candidate)
    for candidate in sorted((ctx.repo / "sandbox" / "lib").glob("python*/site-packages/mlir_aie")):
        if (candidate / "runtime_lib" / "x86_64" / "test_lib" / "include" / "cxxopts.hpp").exists():
            env.setdefault("AIEOPT_DIR", str(candidate))
            break
    bin_paths = []
    for candidate in (ctx.repo / "install-xrt" / "bin", ctx.repo / "install" / "bin", ctx.repo / "build-xrt" / "bin", ctx.repo / "build" / "bin", ctx.repo / "sandbox" / "bin"):
        if (candidate / "aircc").exists() or (candidate / "aiecc").exists() or (candidate / "aiecc.py").exists():
            bin_paths.append(str(candidate))
    if bin_paths:
        env["PATH"] = os.pathsep.join([*bin_paths, env.get("PATH", "")]).rstrip(os.pathsep)
    python_paths = []
    for candidate in (
        ctx.repo / "install-xrt" / "python",
        ctx.repo / "install" / "python",
        ctx.repo / "build-xrt" / "python",
        ctx.repo / "build" / "python",
        ctx.repo / "python",
    ):
        if (candidate / "air" / "backend" / "xrt.py").exists():
            python_paths.append(str(candidate))
    if python_paths:
        env["PYTHONPATH"] = os.pathsep.join([*python_paths, env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    for path in [env.get("PEANO_INSTALL_DIR", ""), *(str(path) for path in sorted((ctx.repo / "sandbox/lib").glob("python*/site-packages/llvm-aie")))]:
        if path and (Path(path) / "bin" / "llc").exists():
            env["PEANO_INSTALL_DIR"] = path
            break
    return env


def npu_backend(ctx: RunContext) -> BackendResult:
    result = backend_result(ctx, "npu", True)
    env = npu_env(ctx)
    source_dir = ctx.repo / "test" / "xrt" / "46_triton_matmul_ver4_strix_8x4_i8_i8_i32"
    build_stamp = result.build_dir / "gemm_int8_build_key.txt"
    build_key = f"runtime_loop_tiling={ctx.npu_runtime_loop_tiling}\n"
    reused = (result.build_dir / "air.xclbin").exists() and (result.build_dir / "air.insts.bin").exists() and find_npu_elves(result.build_dir) and read_text(build_stamp) == build_key
    if reused:
        compile_note = f"reused build_dir={result.build_dir}"
        write_text(log_path(ctx, result, "build"), f"Reusing existing NPU artifacts in {result.build_dir}\n")
    else:
        compile_note = "fresh compile"
        ok, log = run_logged(ctx, result, "build", ["make", "-C", source_dir, f"BUILD_DIR={result.build_dir}", "AIE_TARGET=aie2p", f"M={M}", f"K={K}", f"N={N}", f"RUNTIME_LOOP_TILING={ctx.npu_runtime_loop_tiling}", "compile-xclbin"], env=env)
        if not ok:
            result.status, result.evidence = "WARN", f"NPU compile-xclbin failed; see {log}"
            return result
        write_text(build_stamp, build_key)
    elves = find_npu_elves(result.build_dir)
    if not elves:
        result.status, result.evidence = "WARN", f"NPU build produced no per-core ELF files under {result.build_dir}"
        return result
    disasm_failures = 0
    for elf in elves:
        prefix = sanitize_prefix(str(elf.relative_to(result.build_dir).with_suffix("")))
        ok, _ = run_logged(ctx, result, f"disassemble_{prefix}", [ctx.disassemble, "npu", "--kind", "elf", "--mcpu", "aie2p", "--triple", "aie2p-none-unknown-elf", "--output-dir", result.artifacts_dir, "--prefix", prefix, elf], env=env)
        disasm_failures += 0 if ok else 1
    txn_note = "transaction stream not generated"
    insts = result.build_dir / "air.insts.bin"
    if insts.exists():
        ok, log = run_logged(ctx, result, "disassemble_air_insts", [ctx.disassemble, "npu", "--kind", "txn", "--output-dir", result.artifacts_dir, "--prefix", "npu_air_insts", insts], env=env)
        txn_note = "transaction stream disassembled" if ok else f"transaction stream disassembly failed; see {log}"
    if ctx.run_enabled:
        exe = result.build_dir / "test.exe"
        if not exe.exists():
            ok, log = run_logged(ctx, result, "build_test_exe", ["make", "-C", source_dir, f"BUILD_DIR={result.build_dir}", "AIE_TARGET=aie2p", "build-test-exe"], env=env)
            if not ok or not exe.exists():
                note_run_failure(result, log, "build-test-exe")
        if exe.exists():
            ok, log = run_logged(ctx, result, "profile", ["./test.exe", "-x", "air.xclbin", "-k", "MLIR_AIE", "-i", "air.insts.bin", "-M", M, "-K", K, "-N", N, "-v", "0", "--warmups", ctx.warmups, "--iterations", ctx.iterations, "--b-layout", "row"], cwd=result.build_dir, env=env)
            result.runtime = f"ran; see {log}" if ok else result.runtime
            if not ok:
                note_run_failure(result, log, "profile")
    combined = "\n".join(read_text(path) for path in sorted(result.artifacts_dir.glob("*.disasm.s")))
    vmac = count_regex(combined, r"\bvmac\b")
    vloads = count_regex(combined, r"\bvld[ab]?\b|\bvlda\b|\bvldb\b")
    vstores = count_regex(combined, r"\bvst\b")
    result.evidence = f"{compile_note}, core_elves={len(elves)}, disasm_failures={disasm_failures}, vmac={vmac}, vloads={vloads}, vstores={vstores}, {txn_note}"
    result.status = "PASS" if disasm_failures == 0 and vmac > 0 else "WARN"
    if ctx.run_enabled and "failed" in result.runtime and result.status == "PASS":
        result.status = "WARN"
    if ctx.run_enabled and (run_log := result.logs.get("profile")) and run_log.exists():
        parse_host_perf(ctx, result, run_log, "host run.wait")
        result.perf_notes = f"runtime_loop_tiling={ctx.npu_runtime_loop_tiling}; b_layout={last_kv_value(run_log, 'b_layout') or 'row'}; warmups={last_kv_value(run_log, 'warmups') or ctx.warmups}; validation={last_kv_value(run_log, 'validation') or 'unknown'}; excludes output BO sync; timing wraps run.wait"
    return result


def selected_backends(name: str) -> list[str]:
    return ["cpu", "gpu", "npu"] if name == "all" else [name]


def strict_failed(results: dict[str, BackendResult], selected: set[str], run_enabled: bool) -> bool:
    for name in selected:
        result = results[name]
        if result.status not in {"PASS", "SKIP"}:
            return True
        if run_enabled and result.target_tops is not None:
            if result.perf_tops is None or result.perf_tops < result.target_tops:
                return True
    return False


def write_report(report: Path, ctx: RunContext, args: argparse.Namespace, results: dict[str, BackendResult]) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", encoding="utf-8") as f:
        f.write("# GEMM int8 Benchmark Report\n\n")
        f.write(f"Artifacts: `{ctx.out_dir}`\n\n")
        f.write("## Run Controls\n\n| Field | Value |\n| --- | --- |\n")
        for key, value in (("Selected backend", args.backend), ("Execute kernels", args.run), ("Strict mode", args.strict), ("GPU sweep variants", args.gpu_sweep_variants), ("GPU sweep profile", args.gpu_sweep_profile), ("GPU sweep repetitions", args.gpu_sweep_repetitions), ("GPU sweep group sizes", ",".join(str(size) for size in args.gpu_sweep_group_sizes)), ("GPU default threshold pct", args.gpu_default_threshold_pct), ("GPU provider baselines", args.gpu_provider_baselines), ("GPU provider profile", args.gpu_provider_profile), ("GPU provider validation samples", args.gpu_provider_validation_samples), ("GPU rocMLIR reference", args.gpu_rocmlir_reference), ("rocMLIR bin dir", args.rocmlir_bin_dir or "n/a"), ("rocMLIR artifacts dir", args.rocmlir_artifacts_dir or "n/a"), ("rocMLIR target chip", args.rocmlir_target_chip or args.gpu_arch), ("Warmups", args.warmups), ("Iterations", args.iterations), ("CPU threads", args.cpu_threads), ("NPU runtime loop tiling", args.npu_runtime_loop_tiling), ("Build root", ctx.build_root), ("GPU arch", args.gpu_arch), ("GPU int8 GEMM variant", args.gpu_int8_gemm_variant), ("GPU int8 GEMM group size", args.gpu_int8_gemm_group_size), ("Shape", f"M=N=K={M}, int8 x int8 -> int32")):
            f.write(f"| {key} | `{value}` |\n")
        f.write("\n## ISA Verdicts\n\n| Backend | Status | Evidence | Runtime |\n| --- | --- | --- | --- |\n")
        for name in ("cpu", "gpu", "npu"):
            result = results[name]
            f.write(f"| {name.upper()} | {result.status} | {result.evidence} | {result.runtime} |\n")
        f.write("\n## Performance\n\n| Backend | Timing domain | Count | Latency | Throughput | Target TOPS | % of Target | Notes |\n| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for name in ("cpu", "gpu", "npu"):
            result = results[name]
            target = f"{result.target_tops:.3f}" if result.target_tops is not None else "n/a"
            f.write(f"| {name.upper()} | {result.perf_domain} | {result.perf_count} | {result.perf_latency} | {result.perf_throughput} | {target} | {result.target_pct} | {result.perf_notes} |\n")
        if args.gpu_sweep_variants:
            stability_csv = ctx.out_dir / "gpu_stability_report.csv"
            stability_md = ctx.out_dir / "gpu_stability_report.md"
            f.write("\n## GPU Stability Report\n\n")
            f.write(f"- CSV: `{stability_csv}`\n")
            f.write(f"- Report: `{stability_md}`\n")
        if args.gpu_provider_baselines:
            provider_csv = ctx.out_dir / "gpu_provider_baselines.csv"
            provider_md = ctx.out_dir / "gpu_provider_baselines.md"
            f.write("\n## GPU Provider Baselines\n\n")
            f.write(f"- CSV: `{provider_csv}`\n")
            f.write(f"- Report: `{provider_md}`\n")
        if args.gpu_rocmlir_reference:
            rocmlir_csv = ctx.out_dir / "gpu_rocmlir_reference.csv"
            rocmlir_md = ctx.out_dir / "gpu_rocmlir_reference.md"
            f.write("\n## GPU rocMLIR Reference\n\n")
            f.write(f"- CSV: `{rocmlir_csv}`\n")
            f.write(f"- Report: `{rocmlir_md}`\n")
        f.write("\n## Logs\n\n")
        for name in ("cpu", "gpu", "npu"):
            result = results[name]
            entries = ", ".join(f"{stem} `{path}`" for stem, path in sorted(result.logs.items()))
            f.write(f"- {name.upper()}: {entries or 'n/a'}\n")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["all", "cpu", "gpu", "npu"], default="all", help="backend to process (default: all)")
    parser.add_argument("--out-dir", type=Path, required=True, help="report/artifact root")
    parser.add_argument("--build-dir", type=Path, default=None, help="shared build root; backend subdirectories are created below it")
    parser.add_argument("--run", action="store_true", help="execute selected kernels")
    parser.add_argument("--strict", action="store_true", help="exit non-zero when any selected backend is not PASS")
    parser.add_argument("--gpu-arch", default=os.environ.get("AIR_GPU_CHIP", "gfx1150"), help="AMDGPU chip for GPU lowering (default: AIR_GPU_CHIP or gfx1150)")
    parser.add_argument("--gpu-int8-gemm-variant", choices=GPU_INT8_GEMM_VARIANTS, default=os.environ.get("AIR_INT8_GEMM_VARIANT", DEFAULT_GPU_INT8_GEMM_VARIANT), help="GPU INT8 GEMM lowering variant (default: %(default)s)")
    parser.add_argument("--gpu-int8-gemm-group-size", type=gpu_group_size, default=gpu_group_size(os.environ.get("AIR_INT8_GEMM_GROUP_SIZE", str(DEFAULT_GPU_INT8_GEMM_GROUP_SIZE))), choices=GPU_INT8_GEMM_GROUP_SIZES, help="GPU INT8 GEMM grouped M size for grouped variants (default: %(default)s)")
    parser.add_argument("--gpu-sweep-variants", action="store_true", help="run the fixed GPU INT8 GEMM variant sweep and write CSV/Markdown evidence")
    parser.add_argument("--gpu-sweep-profile", choices=GPU_INT8_GEMM_SWEEP_PROFILES, default=DEFAULT_GPU_INT8_GEMM_SWEEP_PROFILE, help="GPU INT8 GEMM sweep profile (default: full)")
    parser.add_argument("--gpu-sweep-group-sizes", type=gpu_group_sizes, default=DEFAULT_GPU_INT8_GEMM_SWEEP_GROUP_SIZES, help="comma-separated grouped M sizes for grouped swizzle sweep rows in full profile (default: 2,4,8)")
    parser.add_argument("--gpu-sweep-repetitions", type=positive_int, default=DEFAULT_GPU_INT8_GEMM_SWEEP_REPETITIONS, help="runtime repetitions per GPU sweep candidate (default: 3)")
    parser.add_argument("--gpu-default-threshold-pct", type=nonnegative_float, default=DEFAULT_GPU_INT8_GEMM_DEFAULT_THRESHOLD_PCT, help="minimum top-vs-runner-up median TOPS gap required to recommend promoting the GPU benchmark default (default: 3.0)")
    parser.add_argument("--gpu-provider-baselines", action="store_true", help="run fixed-shape external GPU provider baselines and write CSV/Markdown evidence")
    parser.add_argument("--gpu-provider-profile", action="store_true", help="run rocprofv3 kernel trace/stat capture for GPU provider baselines")
    parser.add_argument("--gpu-provider-validation-samples", type=positive_int, default=DEFAULT_GPU_PROVIDER_VALIDATION_SAMPLES, help="sampled output checks per GPU provider run (default: 256)")
    parser.add_argument("--gpu-rocmlir-reference", action="store_true", help="ingest optional rocMLIR static reference artifacts for the fixed GPU INT8 GEMM contract")
    parser.add_argument("--rocmlir-bin-dir", type=Path, default=None, help="optional directory containing rocMLIR tools; used only for tool availability notes")
    parser.add_argument("--rocmlir-artifacts-dir", type=Path, default=None, help="directory containing rocMLIR reference ISA, HSACO, readobj, MLIR, or profile artifacts")
    parser.add_argument("--rocmlir-target-chip", default=None, help="AMDGPU chip used to disassemble rocMLIR HSACO artifacts (default: --gpu-arch)")
    parser.add_argument("--cpu-threads", type=positive_int, default=12, help="CPU worker threads passed to the CPU benchmark (default: 12)")
    parser.add_argument("--npu-runtime-loop-tiling", default="2,4", metavar="M,N", help="AIR runtime loop tiling sizes for NPU compile (default: 2,4)")
    parser.add_argument("--warmups", type=nonnegative_int, default=10, help="warmup iterations for every backend (default: 10)")
    parser.add_argument("--iterations", type=positive_int, default=20, help="timed iterations for every backend (default: 20)")
    args = parser.parse_args(argv)
    tiling_values = args.npu_runtime_loop_tiling.split(",")
    if len(tiling_values) != 2:
        parser.error("--npu-runtime-loop-tiling must contain two positive integers")
    try:
        parsed_tiling = [positive_int(value) for value in tiling_values]
    except argparse.ArgumentTypeError as exc:
        parser.error(f"--npu-runtime-loop-tiling: {exc}")
    args.npu_runtime_loop_tiling = f"{parsed_tiling[0]},{parsed_tiling[1]}"
    if args.gpu_sweep_variants and args.backend not in {"all", "gpu"}:
        parser.error("--gpu-sweep-variants requires --backend all or --backend gpu")
    if args.gpu_sweep_profile != DEFAULT_GPU_INT8_GEMM_SWEEP_PROFILE and not args.gpu_sweep_variants:
        parser.error("--gpu-sweep-profile requires --gpu-sweep-variants")
    if args.gpu_provider_baselines and args.backend not in {"all", "gpu"}:
        parser.error("--gpu-provider-baselines requires --backend all or --backend gpu")
    if args.gpu_rocmlir_reference and args.backend not in {"all", "gpu"}:
        parser.error("--gpu-rocmlir-reference requires --backend all or --backend gpu")
    if (args.rocmlir_bin_dir or args.rocmlir_artifacts_dir or args.rocmlir_target_chip) and not args.gpu_rocmlir_reference:
        parser.error("--rocmlir-bin-dir, --rocmlir-artifacts-dir, and --rocmlir-target-chip require --gpu-rocmlir-reference")
    if args.gpu_provider_baselines and not args.run:
        parser.error("--gpu-provider-baselines requires --run")
    if args.gpu_provider_profile and not args.gpu_provider_baselines:
        parser.error("--gpu-provider-profile requires --gpu-provider-baselines")
    return args


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    out_dir = args.out_dir.resolve()
    ctx = RunContext(Path(__file__).resolve().parents[2], out_dir, args.build_dir.resolve() if args.build_dir else out_dir / "build", out_dir / "logs", args.warmups, args.iterations, args.gpu_arch, args.gpu_int8_gemm_variant, args.gpu_int8_gemm_group_size, args.run, args.cpu_threads, args.npu_runtime_loop_tiling)
    for path in (ctx.out_dir, ctx.build_root, ctx.logs_dir):
        path.mkdir(parents=True, exist_ok=True)
    selected = {"gpu"} if args.gpu_sweep_variants else set(selected_backends(args.backend))
    if args.gpu_provider_baselines:
        selected.add("gpu")
    if args.gpu_rocmlir_reference:
        selected.add("gpu")
    runners = {"cpu": cpu_backend, "gpu": gpu_backend, "npu": npu_backend}
    results = {name: backend_result(ctx, name) for name in ("cpu", "gpu", "npu")}
    if args.gpu_sweep_variants:
        results["gpu"] = run_gpu_variant_sweep(ctx, args.gpu_sweep_profile, GPU_INT8_GEMM_SWEEP_VARIANTS, args.gpu_sweep_group_sizes, args.gpu_sweep_repetitions, args.gpu_default_threshold_pct)
    else:
        for name in ("cpu", "gpu", "npu"):
            if name in selected:
                results[name] = runners[name](ctx)
    rocmlir_row = None
    if args.gpu_rocmlir_reference:
        rocmlir_row = rocmlir_reference_row(ctx, args.rocmlir_bin_dir, args.rocmlir_artifacts_dir, args.rocmlir_target_chip or args.gpu_arch)
        write_gpu_rocmlir_reference_report(ctx, rocmlir_row)
    if args.gpu_provider_baselines:
        run_gpu_provider_baselines(ctx, results["gpu"], args.gpu_sweep_repetitions, args.gpu_provider_validation_samples, args.gpu_provider_profile, rocmlir_row)
    report = out_dir / "gemm_int8_report.md"
    write_report(report, ctx, args, results)
    print(f"Report: {report}")
    return 1 if args.strict and strict_failed(results, selected, args.run) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
