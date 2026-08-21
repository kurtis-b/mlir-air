# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""LM Head GEMV multi-launch — 8-partition GEMV in one ELF for decode.

Partitions the large vocab projection into 8 GEMVs of M=16384, K=2048 each,
stitched as 8 air.launch ops in one ELF. Single-token decode version of the
prefill LM Head (which uses GEMM with M=seq_len).

17 func args: 1 shared input (1D) + 8 weights (2D) + 8 outputs (1D).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "matrix_vector_multiplication",
        "bf16",
    ),
)

from shared.infra.stitching import (
    stitch_elf,
    KernelSlice,
    FuncArg,
)

_EXTERN_FUNCS = {"@matvec_vectorized_bf16_bf16", "@linalg_fill_bf16"}


def build_lm_head_gemv_module(
    emb_dim=2048,
    n_partitions=8,
    n_part=16384,
    tile_m=8,
    m_input=4,
    herd_m=8,
    parts=None,
):
    """Build multi-launch LM Head GEMV: n_partitions GEMV launches in one func.

    Each partition: GEMV with M=n_part (output dim), K=emb_dim (input dim).
    All partitions share the same input vector.

    parts `[2026-08-21]`: an explicit list of partition row counts (mixed sizes
    allowed -- stitching takes launches of different shapes). Overrides
    n_partitions / n_part. Full partitions at the BD-repeat cap plus one tail
    on the tile grid avoid both padding the vocab to a whole partition and
    paying a ~107 us launch boundary per 8192 rows: for Qwen3-0.6B
    9 x 16384 + 4480 at m_input 8 measured 9.35 -> 8.25 ms against 19 x 8192
    (doc 56 H0's planner finding, devq 476).

    Returns:
        Module with func @lm_head_gemv and (1 + 2*n_partitions) memref args:
            %arg0: x (emb_dim,) — shared input vector (1D)
            %arg(1+2*p): weight_p (n_part, emb_dim) — partition weight (2D)
            %arg(2+2*p): output_p (n_part,) — partition output (1D)
    """
    from matvec import build_module as build_gemv
    from ml_dtypes import bfloat16

    if parts is None:
        parts = [n_part] * n_partitions
    parts = list(parts)
    n_partitions = len(parts)
    print(
        f"  Building {n_partitions}-partition LM Head GEMV (M_parts={sorted(set(parts))}, K={emb_dim})..."
    )
    gemv_irs = {}
    for rows in sorted(set(parts)):
        gemv_irs[rows] = str(
            build_gemv(rows, emb_dim, tile_m, m_input, herd_m, bfloat16, bfloat16)
        )

    # Stitch one GEMV per partition, each onto its own (weight, output) arg
    # pair, all sharing arg0 (input vector).
    # GEMV func has 3 args: {0: weight (MxK), 1: input (K,), 2: output (M,)}.
    # Combined: arg0=shared_input, arg(1+2p)=weight_p, arg(2+2p)=output_p.
    # Per-partition mapping: {0: 1+2*p, 1: 0, 2: 2+2*p}.
    base_args = [FuncArg("%arg0", f"memref<{emb_dim}xbf16>")]
    for p, rows in enumerate(parts):
        base_args.append(FuncArg(f"%arg{1+2*p}", f"memref<{rows}x{emb_dim}xbf16>"))
        base_args.append(FuncArg(f"%arg{2+2*p}", f"memref<{rows}xbf16>"))

    slices = [
        KernelSlice(
            gemv_irs[rows],
            f"p{p}",
            {0: 1 + 2 * p, 1: 0, 2: 2 + 2 * p},
            extern_syms=_EXTERN_FUNCS,
        )
        for p, rows in enumerate(parts)
    ]

    module = stitch_elf("lm_head_gemv", base_args, slices)
    print(
        f"  Module: {len(str(module).splitlines())} lines, "
        f"{1+2*n_partitions} args, {n_partitions} launches, parsed OK"
    )
    return module
