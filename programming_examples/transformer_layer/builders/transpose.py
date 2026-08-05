# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""2-D bf16 matrix transpose over the external ``transpose_bf16`` tile kernel.

CONTRACT
    ``build_transpose_module(rows, cols, ...)`` returns an ``air.ir.Module``
    whose single function takes a ``[rows, cols]`` bf16 memref and writes its
    transpose into a ``[cols, rows]`` bf16 memref. Pure data movement: every
    output element is bit-identical to its input element, so the FP32 reference
    below is exact and the whole tolerance budget is unused.

    The port of iron's ``transpose`` operator (``iron/operators/transpose``),
    re-expressed per convention 1 as a plain builder. iron's ``runlist`` mode
    dispatches it as ``k_transpose`` to feed an on-device ``q @ k^T``; in this
    port that GEMM stays on the host (no ``K = 64`` registry row exists or can
    be swept -- 08c), so the operator is validated standalone here and the
    ``runlist`` mode's README records why it is not on that mode's dataflow.

WHY THE MOVEMENT IS CONTIGUOUS AND THE REORDERING IS IN A KERNEL
    A DMA-stride transpose is not available for this dtype: the innermost DMA
    stride must be 1 for sub-32-bit types, so a bf16 transpose cannot be
    expressed as a strided descriptor the way a 32-bit one can
    (``data_transfer_transpose/dma/`` does exactly that for i32). This builder
    is the ``data_transfer_transpose/dma_bf16/`` pattern -- contiguous tile in,
    scalar-transposed in L1, contiguous tile out -- tiled and spread over an
    ``herd_x x 1`` herd, which the single-tile example is not.

FOOTGUNS
    - ``DIM_M`` / ``DIM_N`` are -D flags on the kernel compile, not runtime
      arguments, and they are the L1 TILE shape. The object name carries them
      (``transpose_m64n96.o``), so two tile shapes cannot overwrite each
      other's object -- the same ``(name, baked flag)`` trap E1 fixed for the
      GEMM micro-kernels.
    - Each herd column owns a ``cols // herd_x``-wide column stripe of the
      input, which becomes a row stripe of the output. ``cols`` must divide by
      ``herd_x``; a remainder is rejected here rather than silently dropped.
    - The output block write is 2-D (``[tile_cols, tile_rows]`` at stride
      ``[rows, 1]``): contiguous only along its innermost dimension. That is a
      legal descriptor (inner stride 1) but a wide-and-short one, so tall
      ``tile_rows`` amortize the per-row descriptor cost.
    - ``transpose_m<M>n<N>.o`` must exist in the working directory when aiecc
      links. ``compile_transpose_kernel`` builds it there; run from a scratch
      directory.
"""

import os

import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects.func import FuncOp, CallOp
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import type_mapper

range_ = for_

#: The tile kernel source this example owns (copied from
#: data_transfer_transpose/dma_bf16/ -- provenance in its header).
KERNEL_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "kernels",
    "transpose.cc",
)


def transpose_kernel_obj(tile_rows, tile_cols):
    """The object name for one tile shape. The shape is IN the name on purpose."""
    return f"transpose_m{tile_rows}n{tile_cols}.o"


def compile_transpose_kernel(tile_rows, tile_cols):
    """Build the tile kernel's object into the CWD, where aiecc looks."""
    import shared.infra.external_kernels as ek

    ek._compile_kernel(
        KERNEL_SRC,
        transpose_kernel_obj(tile_rows, tile_cols),
        extra_flags=[f"-DDIM_M={tile_rows}", f"-DDIM_N={tile_cols}"],
    )


@module_builder
def build_transpose_module(rows, cols, np_dtype=bfloat16, herd_x=8, tile_rows=64):
    """Build the tiled transpose ``out[c, r] = in[r, c]`` over an ``herd_x x 1`` herd.

    Args:
        rows, cols: the input's L3 shape; the output is ``[cols, rows]``.
            ``cols`` must divide by ``herd_x`` and ``rows`` by ``tile_rows``.
        np_dtype: element type; bf16, matching the kernel's 16-bit element copy.
        herd_x: AIE columns. Two shim DMAs per tile (in, out).
        tile_rows: input rows per kernel call. The L1 tile is
            ``tile_rows x (cols // herd_x)`` twice over (in and out), and
            ``2 * 2 * tile_rows * (cols // herd_x) * 2`` bytes must fit L1 with
            ping-pong.

    Returns:
        air.ir.Module with one function ``transpose_2d(x, y)``.
    """
    if cols % herd_x:
        raise ValueError(f"cols ({cols}) must be divisible by herd_x ({herd_x})")
    tile_cols = cols // herd_x
    if rows % tile_rows:
        raise ValueError(f"rows ({rows}) must be divisible by tile_rows ({tile_rows})")

    xrt_dtype = type_mapper(np_dtype)
    l3_in_ty = MemRefType.get([rows, cols], xrt_dtype)
    l3_out_ty = MemRefType.get([cols, rows], xrt_dtype)

    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_in_ty = MemRefType.get([tile_rows, tile_cols], xrt_dtype, memory_space=l1_space)
    l1_out_ty = MemRefType.get([tile_cols, tile_rows], xrt_dtype, memory_space=l1_space)

    kernel_obj = transpose_kernel_obj(tile_rows, tile_cols)
    tp_func = FuncOp(
        "transpose_bf16",
        ([l1_in_ty, l1_out_ty], []),
        visibility="private",
    )
    tp_func.attributes["link_with"] = StringAttr.get(kernel_obj)
    tp_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    # This tile's column-stripe start: tx * tile_cols.
    col_map = AffineMap.get(
        0,
        1,
        [
            AffineExpr.get_mul(
                AffineSymbolExpr.get(0), AffineConstantExpr.get(tile_cols)
            )
        ],
    )

    @FuncOp.from_py_func(l3_in_ty, l3_out_ty)
    def transpose_2d(arg0, arg1):

        @launch(operands=[arg0, arg1])
        def tp_launch(l_in, l_out):

            @segment(name="tp_seg", operands=[l_in, l_out])
            def tp_seg(s_in, s_out):

                @herd(name="tp_herd", sizes=[herd_x, 1], operands=[s_in, s_out])
                def tp_body(_tx, _ty, _sx, _sy, h_in, h_out):
                    l1_in = AllocOp(l1_in_ty, [], [])
                    l1_out = AllocOp(l1_out_ty, [], [])
                    col = affine_apply(col_map, [_tx])

                    for row in range_(0, rows, tile_rows):
                        dma_memcpy_nd(
                            l1_in,
                            h_in,
                            src_offsets=[row, col],
                            src_sizes=[tile_rows, tile_cols],
                            src_strides=[cols, 1],
                        )
                        CallOp(tp_func, [l1_in, l1_out])
                        dma_memcpy_nd(
                            h_out,
                            l1_out,
                            dst_offsets=[col, row],
                            dst_sizes=[tile_cols, tile_rows],
                            dst_strides=[rows, 1],
                        )
                        yield_([])

                    DeallocOp(l1_in)
                    DeallocOp(l1_out)

                tp_body.attributes["link_with"] = StringAttr.get(kernel_obj)


def transpose_reference(x):
    """The exact transpose, contiguous, in the input dtype.

    No arithmetic happens on either side, so this is the one operator in the
    example whose device output must be BIT-identical to its reference; the
    shared ``rtol``/``atol`` budget is real for every other operator and pure
    slack here.
    """
    return np.ascontiguousarray(np.asarray(x).T)
