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
    herd_rows=1,
):
    """Build multi-launch LM Head GEMV: n_partitions GEMV launches in one func.

    Each partition: GEMV with M=n_part (output dim), K=emb_dim (input dim).
    All partitions share the same input vector.

    herd_rows `[2026-08-26]` (queue item 28): how many of the device's four CORE
    ROWS each partition's herd occupies. **Default 1 -- the IR is then
    byte-identical to the pre-parameter builder at every shipped shape** (item 27
    proved the same for `matvec.py` itself). An int applies to every partition; a
    sequence gives one row count per partition, which the mixed heads need
    because a partition is only legal at `herd_rows` R when
    `rows % (tile_m * herd_m * R) == 0` -- Qwen3-1.7B's 4480-row tail divides by
    128 but not by 256, so it caps at 2 while the 16384-row partitions take 4.

    **A caller that sets `herd_rows > 1` MUST compile with
    `use_lock_race_condition_fix`** or the device hangs with
    `ERT_CMD_STATE_TIMEOUT` (item 27 section 6.1). Use
    `backend_presets.with_herd_rows()` to derive the flag from the same number;
    `KernelCache.compile_and_cache` fails closed if it is missing.

    Rows are worth taking only above a byte threshold -- item 27 section 5.2
    measured 9.06 MB (2 rows) and 20.82 MB (4 rows) for bf16, against +49.7 /
    +166.7 us of added fixed cost per launch. A 16384 x 1024 partition is
    33.55 MB and a 16384 x 2048 one is 67.11 MB, which is why the LM head is a
    call site for this and the QKV GEMVs are not.

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
    if hasattr(herd_rows, "__iter__"):
        rows_per_part = [int(r) for r in herd_rows]
        if len(rows_per_part) != n_partitions:
            raise ValueError(
                f"herd_rows has {len(rows_per_part)} entries for {n_partitions} "
                "partitions; give one int for all of them or one per partition"
            )
    else:
        rows_per_part = [int(herd_rows)] * n_partitions
    for rows, hr in zip(parts, rows_per_part):
        band = tile_m * herd_m * hr
        if rows % band:
            raise ValueError(
                f"partition of {rows} rows cannot use herd_rows={hr}: "
                f"{rows} % (tile_m {tile_m} * herd_m {herd_m} * herd_rows {hr} "
                f"= {band}) != 0. Lower this partition's herd_rows -- the "
                "argument takes a per-partition sequence for exactly this."
            )
    print(
        f"  Building {n_partitions}-partition LM Head GEMV (M_parts={sorted(set(parts))}, "
        f"K={emb_dim}, herd_rows={sorted(set(rows_per_part))})..."
    )
    # Keyed on (rows, herd_rows): two partitions of the same height at different
    # row counts are DIFFERENT modules, and keying on rows alone would silently
    # give the second one the first's geometry.
    gemv_irs = {}
    for key in sorted(set(zip(parts, rows_per_part))):
        rows, hr = key
        gemv_irs[key] = str(
            build_gemv(
                rows,
                emb_dim,
                tile_m,
                m_input,
                herd_m,
                bfloat16,
                bfloat16,
                herd_rows=hr,
            )
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
            gemv_irs[(rows, rows_per_part[p])],
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
