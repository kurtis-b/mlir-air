# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fused QKV projection: one GEMM ``A[M,K] @ B[K,3K]``, C split three ways.

CONTRACT
    ``build_qkv_proj_module(seq_len, emb_dim, ...)`` returns an ``air.ir.Module``
    with one function ``qkv_proj`` and six memref arguments:

        %arg0  x        [seq_len, emb_dim]      bf16   input activations
        %arg1  w_qkv    [emb_dim, 3*emb_dim]    bf16   the fused QKV weight
        %arg2  c_f32    [seq_len, 3*emb_dim]    f32    GEMM scratch (see below)
        %arg3  q        [seq_len, emb_dim]      bf16   output
        %arg4  k        [seq_len, emb_dim]      bf16   output
        %arg5  v        [seq_len, emb_dim]      bf16   output

    Four ``air.launch`` operations: the GEMM over the full ``3*emb_dim`` width,
    then one split-cast launch per projection. Q, K and V land in three separate
    DDR buffers with no host-side slice of C anywhere.

WHY THE SPLIT RIDES THE CAST
    The registry's high-precision method for these shapes is ``fused-cast``: the
    external GEMM accumulates in f32 into a scratch buffer and a *separate*
    launch casts it down to bf16. That cast has to read every element of C
    exactly once regardless, so the three-way split is folded into it -- each
    split-cast launch reads one ``emb_dim``-wide column band of the f32 scratch
    and writes its own bf16 output. The split therefore costs nothing beyond the
    cast that was already going to happen, and it happens in the DMA descriptors
    of the runtime sequence rather than in host code.

    Consequence: this builder REQUIRES a method with an f32 scratch. A shape
    whose registry entry selects ``drain`` (in-GEMM cast, one launch, no
    scratch) raises instead of silently falling back, because splitting a
    bf16 C would mean three extra full-tensor copies and a different numeric
    path from the one the registry measured.

TILES COME FROM THE REGISTRY, NEVER FROM A CONSTANT
    ``resolve_gemm_spec(m, k, n)`` is the only source of tile sizes, method and
    herd. It RAISES on an unmeasured shape, deliberately -- hand-copied tile
    configs previously caused drift bugs (``registry_lookup.py`` docstring).
    ``gemm_spec_fn`` is the one escape hatch, the same one
    ``rms_qkv_qknorm_rope_multi.py`` already ships for ``qwen3_4b``; an injected
    spec is recorded in the results artifact so the guess is visible rather than
    silent. Do not add a fallback that guesses tiles.

FOOTGUNS
    - ``%arg2`` is an INPUT slot to ``XRTRunner.run_test``, not an output. The
      arg order above is chosen so every buffer the host fills (including the
      scratch) precedes every buffer it reads back, which is the layout
      ``run_test`` assumes when it appends output placeholders after ``inputs``.
      Reordering the signature so the scratch trails the outputs silently
      compares the f32 scratch against a bf16 reference.
    - The split-cast herd is ``herd_x x 1``. Two shim DMAs per tile is already
      the whole per-column budget once the herd spans all 8 columns; ``herd_y >
      1`` fails to place with "out of channels".
    - ``mm_m64.o`` (or whichever object the registry method names) must exist in
      the working directory when aiecc links. ``opcheck.py`` compiles it there.
    - The reference below computes in FP32 and rounds ONCE, matching the
      device's f32-accumulate-then-single-epilogue-cast. It is not iron's bf16
      oracle, which agrees with a bf16 device partly by being wrong in the same
      direction.
"""

import os
import sys

import numpy as np
from ml_dtypes import bfloat16

_PROJ_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
for _p in (_PROJ_ROOT, os.path.join(_PROJ_ROOT, "llms")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from air.ir import *  # noqa: E402,F403
from air.dialects.affine import apply as affine_apply  # noqa: E402
from air.dialects.air import *  # noqa: E402,F403
from air.dialects import arith  # noqa: E402
from air.dialects.memref import AllocOp, DeallocOp, subview  # noqa: E402
from air.dialects.memref import collapse_shape as memref_collapse_shape  # noqa: E402
from air.dialects.vector import transfer_read, transfer_write  # noqa: E402
from air.dialects.func import FuncOp  # noqa: E402
from air.dialects.scf import for_, yield_  # noqa: E402
from air.backend.xrt_runner import type_mapper  # noqa: E402

from shared.infra.stitching import FuncArg, KernelSlice, stitch_elf  # noqa: E402

from builders.gemm_spec import resolve_gemm_spec, spec_herd  # noqa: E402

range_ = for_

# The three projections, in the column order the fused weight packs them.
QKV_NAMES = ("q", "k", "v")


def qkv_gemm_spec(seq_len, emb_dim, gemm_spec_fn=None):
    """Resolve the fused-QKV GEMM's method and tiles, registry-first.

    Returns ``(spec, source)`` where ``source`` is ``"registry"`` or
    ``"injected"``. Split out from the builder so ``opcheck.py`` can record the
    resolved spec in the results artifact without building the module twice.
    """
    n_total = 3 * emb_dim
    if gemm_spec_fn is not None:
        return gemm_spec_fn(seq_len, emb_dim, n_total), "injected"
    return resolve_gemm_spec(seq_len, emb_dim, n_total, "bf16", "high"), "registry"


@module_builder
def _build_split_cast_module(
    rows,
    cols_total,
    cols_slice,
    col_offset,
    herd_x=8,
    vector_size=16,
):
    """One column band of an f32 C, cast to bf16 into its own bf16 buffer.

    ``in[r, col_offset : col_offset + cols_slice] -> out[r, :]``, f32 to bf16,
    for every row. This is the piece that makes the QKV split free: it is the
    cast the fused-cast method owes anyway, with a column offset added to its
    read.

    Both L3 arguments stay 2-D at the function boundary and are collapsed to 1-D
    inside the launch, so the DMAs are plain 1-D descriptors: a single row's
    column band is ``cols_slice`` CONTIGUOUS f32 at flat offset
    ``r * cols_total + col_offset``. Walking a row at a time is what keeps it
    that way -- a multi-row tile would need a strided 2-D descriptor for the
    same bytes and buys nothing here, since one row is already a 4 KB transfer.

    L1 holds one row in and one row out (``cols_slice * 6`` bytes), which leaves
    room for the ping-pong pair the AIR pipeliner introduces.
    """
    if rows % herd_x:
        raise ValueError(f"rows ({rows}) must be divisible by herd_x ({herd_x})")
    if cols_slice % vector_size:
        raise ValueError(
            f"cols_slice ({cols_slice}) must be divisible by vector_size "
            f"({vector_size}); the vectorized cast has no scalar tail"
        )
    if col_offset + cols_slice > cols_total:
        raise ValueError(
            f"band [{col_offset}, {col_offset + cols_slice}) does not fit in "
            f"cols_total ({cols_total})"
        )

    rows_per_tile = rows // herd_x
    f32_ty = type_mapper(np.float32)
    bf16_ty = type_mapper(bfloat16)

    l3_in_2d = MemRefType.get([rows, cols_total], f32_ty)
    l3_in_1d = MemRefType.get([rows * cols_total], f32_ty)
    l3_out_2d = MemRefType.get([rows, cols_slice], bf16_ty)
    l3_out_1d = MemRefType.get([rows * cols_slice], bf16_ty)

    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_f32 = MemRefType.get([cols_slice], f32_ty, memory_space=l1_space)
    l1_bf16 = MemRefType.get([cols_slice], bf16_ty, memory_space=l1_space)
    vec_f32 = VectorType.get([vector_size], f32_ty)
    vec_bf16 = VectorType.get([vector_size], bf16_ty)
    identity_map = AffineMapAttr.get(AffineMap.get_identity(1))

    def _row_offset_map(row_pitch, extra):
        # ((tx * rows_per_tile) + loop_iv) * row_pitch + extra
        # s0 = loop_iv (row within this tile), s1 = tx
        row = AffineExpr.get_add(
            AffineSymbolExpr.get(0),
            AffineExpr.get_mul(
                AffineSymbolExpr.get(1), AffineConstantExpr.get(rows_per_tile)
            ),
        )
        flat = AffineExpr.get_mul(row, AffineConstantExpr.get(row_pitch))
        if extra:
            flat = AffineExpr.get_add(flat, AffineConstantExpr.get(extra))
        return AffineMap.get(0, 2, [flat])

    in_map = _row_offset_map(cols_total, col_offset)
    out_map = _row_offset_map(cols_slice, 0)

    @FuncOp.from_py_func(l3_in_2d, l3_out_2d)
    def qkv_split_cast(arg0, arg1):

        @launch(operands=[arg0, arg1])
        def split_launch(l_in, l_out):
            in_flat = memref_collapse_shape(l3_in_1d, l_in, [[0, 1]])
            out_flat = memref_collapse_shape(l3_out_1d, l_out, [[0, 1]])

            @segment(name="split_seg", operands=[in_flat, out_flat])
            def split_seg(s_in, s_out):

                @herd(name="split_herd", sizes=[herd_x, 1], operands=[s_in, s_out])
                def split_body(_tx, _ty, _sx, _sy, h_in, h_out):
                    li = AllocOp(l1_f32, [], [])
                    lo = AllocOp(l1_bf16, [], [])
                    c0 = arith.ConstantOp.create_index(0)
                    cst0 = arith.ConstantOp(f32_ty, 0.0)

                    for loop_iv in range_(0, rows_per_tile, 1):
                        in_off = affine_apply(in_map, [loop_iv, _tx])
                        out_off = affine_apply(out_map, [loop_iv, _tx])
                        dma_memcpy_nd(
                            li,
                            h_in,
                            src_offsets=[in_off],
                            src_sizes=[cols_slice],
                            src_strides=[1],
                        )
                        for j in range_(0, cols_slice, vector_size):
                            sub_i = subview(li.result, [j], [vector_size], [1])
                            sub_o = subview(lo.result, [j], [vector_size], [1])
                            v = transfer_read(
                                vec_f32, sub_i, [c0], identity_map, cst0, [True]
                            )
                            transfer_write(
                                None,
                                arith.TruncFOp(vec_bf16, v),
                                sub_o,
                                [c0],
                                identity_map,
                                [True],
                            )
                            yield_([])
                        dma_memcpy_nd(
                            h_out,
                            lo,
                            dst_offsets=[out_off],
                            dst_sizes=[cols_slice],
                            dst_strides=[1],
                        )
                        yield_([])

                    DeallocOp(li)
                    DeallocOp(lo)


def build_qkv_proj_module(
    seq_len,
    emb_dim,
    herd_m=None,
    herd_n=None,
    split_herd_x=8,
    gemm_spec_fn=None,
):
    """Build the four-launch fused QKV projection.

    Args:
        seq_len, emb_dim: ``x`` is ``[seq_len, emb_dim]`` and the fused weight
            is ``[emb_dim, 3*emb_dim]``. The triple ``(seq_len, emb_dim,
            3*emb_dim)`` must be in the GEMM registry, or ``gemm_spec_fn`` must
            supply a spec for it.
        herd_m, herd_n: GEMM herd. ``None`` (the normal setting) uses the herd
            the resolved row's tiles were measured at, which is 8x4 for every
            shape long enough to hold it and smaller at the short end of the
            sequence ladder. A value here is an explicit override and
            invalidates the tiling without invalidating the lookup.
        split_herd_x: AIE columns for each split-cast launch. Must divide
            ``seq_len``; must stay at ``herd_y == 1`` (see the module footguns).
        gemm_spec_fn: ``(m, k, n) -> spec`` escape hatch for a shape the
            registry has not measured. Leave ``None`` in normal use.

    Returns:
        air.ir.Module with one function ``qkv_proj``.
    """
    n_total = 3 * emb_dim
    spec, source = qkv_gemm_spec(seq_len, emb_dim, gemm_spec_fn)
    if not spec["needs_f32_scratch"]:
        raise ValueError(
            f"qkv_proj at {seq_len}x{emb_dim}x{n_total} resolves to method "
            f"{spec['method']!r}, which casts inside the GEMM and exposes no f32 "
            f"scratch. The three-way C split rides the separate cast launch, so "
            f"this builder needs a scratch-bearing method (fused-cast). Measure "
            f"and register a fused-cast entry for this shape rather than "
            f"widening this check."
        )

    from matrix_multiplication.bf16_in_bf16_out.run import build_module as build_gemm

    herd_m, herd_n = spec_herd(spec, herd_m, herd_n)
    print(
        f"  [1/4] QKV GEMM {seq_len}x{emb_dim}x{n_total} "
        f"({spec['method']}, {source}, tile_m={spec['tile_m']} "
        f"tile_k_l2={spec['tile_k_l2']} tile_k_l1={spec['tile_k_l1']} "
        f"tile_n={spec['tile_n']} herd={herd_m}x{herd_n})..."
    )
    gemm_ir = str(
        build_gemm(
            seq_len,
            emb_dim,
            n_total,
            spec["tile_m"],
            spec["tile_k_l2"],
            spec["tile_k_l1"],
            spec["tile_n"],
            herd_m,
            herd_n,
            bfloat16,
            np.float32,
            arch="aie2p",
            emit_external_call=True,
            sym_suffix=spec["sym_suffix"],
            link_with_name=spec["obj"],
        )
    )

    sfx = spec["sym_suffix"]
    gemm_externs = {
        "@matmul_bf16",
        "@op_has_no_registered_library_name" + sfx,
        "@zero_f32_mn" + sfx,
    }

    slices = [KernelSlice(gemm_ir, "g", {0: 0, 1: 1, 2: 2}, extern_syms=gemm_externs)]
    for i, name in enumerate(QKV_NAMES):
        print(f"  [{i + 2}/4] split-cast {name} (columns {i * emb_dim}..)...")
        split_ir = str(
            _build_split_cast_module(
                seq_len, n_total, emb_dim, i * emb_dim, herd_x=split_herd_x
            )
        )
        # operand 0 = the f32 scratch (%arg2), operand 1 = this projection's out.
        slices.append(KernelSlice(split_ir, f"c{name}", {0: 2, 1: 3 + i}))

    base_args = [
        FuncArg("%arg0", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg1", f"memref<{emb_dim}x{n_total}xbf16>"),
        FuncArg("%arg2", f"memref<{seq_len}x{n_total}xf32>"),
    ] + [
        FuncArg(f"%arg{3 + i}", f"memref<{seq_len}x{emb_dim}xbf16>")
        for i in range(len(QKV_NAMES))
    ]

    module = stitch_elf(
        "qkv_proj",
        base_args,
        slices,
        debug_dump_path="/tmp/debug_qkv_proj.mlir",
    )
    print(f"  Module: {len(str(module).splitlines())} lines, parsed OK")
    return module


def qkv_proj_reference(x, w_qkv):
    """FP32 reference for the fused QKV projection, rounded once per output.

    Matches the device exactly in structure: accumulate the whole ``K`` reduction
    in f32, then round to the output dtype a single time. Slicing after the
    rounding (rather than rounding three separate f32 slices) is the same value
    either way and keeps the correspondence to the single physical C obvious.

    Returns ``(q, k, v)``, each ``[seq_len, emb_dim]`` in ``x``'s dtype.
    """
    emb_dim = x.shape[1]
    c = (x.astype(np.float32) @ w_qkv.astype(np.float32)).astype(x.dtype)
    return tuple(c[:, i * emb_dim : (i + 1) * emb_dim] for i in range(len(QKV_NAMES)))
