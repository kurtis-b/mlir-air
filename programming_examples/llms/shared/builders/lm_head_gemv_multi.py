# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""LM Head GEMV multi-launch — one GEMV launch per partition, in one ELF.

Partitions the large vocab projection into GEMVs of M=rows, K=emb_dim,
stitched as one air.launch each in a single ELF. Single-token decode version
of the prefill LM Head (which uses GEMM with M=seq_len).

Partitions default to equal sizes (`n_partitions` x `n_part`, historically
8 x 16384 at K=2048, giving 17 func args). `parts` takes an explicit list
instead, so a vocabulary that is not a whole multiple of the partition size
can end in a shorter tail rather than being padded up to one -- see the
argument's own documentation.

Func args either way: 1 shared input (1D) + one weight (2D) and one output
(1D) per partition.
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
    """Build multi-launch LM Head GEMV: one GEMV launch per partition, one func.

    Each partition: GEMV with M=its row count (output dim), K=emb_dim (input
    dim). All partitions share the same input vector.

    parts: an explicit list of partition row counts, mixed sizes allowed --
        stitching takes launches of different shapes. Overrides
        n_partitions / n_part, which stay as the equal-sized shorthand.

        Mixed sizes exist to avoid paying for both of the alternatives: padding
        the vocabulary up to a whole number of equal partitions, and paying a
        launch boundary per partition when the partitions are small. Full
        partitions at the BD-repeat cap plus one tail on the tile grid does
        neither. The tag measured Qwen3-0.6B's head at 9 x 16384 + 4480,
        m_input 8, as 9.35 -> 8.25 ms against 19 x 8192 (doc 56 H0's planner
        finding, devq 476). **That number is the research branch's, taken on
        its own tree; it has not been re-measured on main, and this builder
        makes no performance claim of its own.**

    Returns:
        Module with func @lm_head_gemv and (1 + 2*len(parts)) memref args:
            %arg0: x (emb_dim,) — shared input vector (1D)
            %arg(1+2*p): weight_p (parts[p], emb_dim) — partition weight (2D)
            %arg(2+2*p): output_p (parts[p],) — partition output (1D)
    """
    from matvec import build_module as build_gemv
    from ml_dtypes import bfloat16

    if parts is None:
        parts = [n_part] * n_partitions
    parts = list(parts)
    n_partitions = len(parts)

    # Check the row counts here rather than letting build_gemv's assert fire:
    # with mixed sizes the caller needs to know WHICH partition is illegal, and
    # the tail partition is exactly the one that tends to be.
    band = tile_m * herd_m
    for p, rows in enumerate(parts):
        if rows % band:
            raise ValueError(
                f"partition {p} of {rows} rows is not a multiple of "
                f"tile_m ({tile_m}) * herd_m ({herd_m}) = {band}; a partition "
                "must sit on the tile grid"
            )

    print(
        f"  Building {n_partitions}-partition LM Head GEMV "
        f"(M_parts={sorted(set(parts))}, K={emb_dim})..."
    )
    # One build per distinct row count, not one per partition: the equal-sized
    # case stays a single build, and the mixed case pays only for the shapes it
    # actually uses.
    gemv_irs = {
        rows: str(
            build_gemv(rows, emb_dim, tile_m, m_input, herd_m, bfloat16, bfloat16)
        )
        for rows in sorted(set(parts))
    }

    # Stitch one GEMV per partition, each onto its own (weight, output)
    # arg pair, all sharing arg0 (input vector).
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
