# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Multi-row LayerNorm over the external ``layer_norm_rows`` kernel.

CONTRACT
    ``build_layer_norm_module(rows, cols, ...)`` returns an ``air.ir.Module``
    whose single function takes two ``[rows, cols]`` bf16 memrefs and computes
    the unweighted LayerNorm of each row:

        y[r, :] = (x[r, :] - mean(x[r, :])) * rsqrt(var(x[r, :]) + 1e-5)

    Rows are split across an ``herd_x x 1`` AIE grid; each tile normalizes
    ``rows_per_call`` consecutive rows per kernel invocation, which is the
    whole reason ``layer_norm_rows`` exists -- ``programming_examples/
    layer_norm/layer_norm.py`` does one row per launch by direct codegen and
    that is not the shape a transformer block wants.

WHY THIS IS NOT THE layer_norm/ EXAMPLE
    That example accumulates its statistics in bf16 and gates at
    ``rtol=5e-2, atol=5e-1``, which is not a tolerance the registry would
    accept. This builder links the ported ``layer_norm.cc``, whose
    sum-of-squares accumulates in f32, and gates at the registry's
    ``rtol=1.6e-2``.

FOOTGUNS
    - The kernel computes variance ONE-PASS as ``E[x^2] - E[x]^2``. The
      reference below is the numerically stable TWO-PASS form in f32. The two
      are algebraically equal and numerically are not: on a row whose mean is
      large next to its spread the one-pass form cancels catastrophically and
      the two will disagree by far more than bf16 rounding. Do not use one as
      the oracle for the other on such data; the shapes checked here are
      zero-mean activations, where the cancellation does not bite.
    - ``cols`` must be a multiple of 16 (``LN_VEC_LEN``). There is no scalar
      tail -- a non-multiple silently drops the remainder instead of failing.
    - Rows must be contiguous and exactly ``cols`` apart. A padded tile has to
      pass the padded stride as ``cols``, which then normalizes the padding
      into the statistics too.
    - ``layer_norm.o`` must exist in the working directory when aiecc links.
      ``opcheck.py`` compiles it there; a builder used elsewhere must call
      ``external_kernels.compile_layer_norm()`` first.
"""

import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.func import FuncOp, CallOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import type_mapper

range_ = for_

# Must match kEpsilon in programming_examples/layer_norm/layer_norm.cc.
EPS = 1e-5

# The vector width layer_norm.o is compiled with (LN_VEC_LEN). cols must be a
# multiple of it or the kernel drops the remainder without saying so.
LN_VEC_LEN = 16

KERNEL_OBJ = "layer_norm.o"


@module_builder
def build_layer_norm_module(rows, cols, np_dtype=bfloat16, herd_x=8, rows_per_call=8):
    """Build multi-row LayerNorm over an ``herd_x x 1`` herd.

    Args:
        rows, cols: L3 tensor shape. ``rows`` must divide by
            ``herd_x * rows_per_call``; ``cols`` by ``LN_VEC_LEN``.
        np_dtype: element type; bf16, matching the kernel's C linkage.
        herd_x: AIE columns. Two shim DMAs per tile (in, out), so the
            channel budget is comfortable at the full 8.
        rows_per_call: rows resident in L1 per ``layer_norm_rows`` call. Larger
            amortizes the call and DMA setup; ``2 * rows_per_call * cols * 2``
            bytes must fit L1 with ping-pong.

    Returns:
        air.ir.Module with one function ``layer_norm_multi_row(x, y)``.
    """
    if cols % LN_VEC_LEN:
        raise ValueError(
            f"cols ({cols}) must be a multiple of LN_VEC_LEN ({LN_VEC_LEN}); "
            "the kernel has no scalar tail and would drop the remainder"
        )
    if rows % (herd_x * rows_per_call):
        raise ValueError(
            f"rows ({rows}) must be divisible by herd_x*rows_per_call "
            f"({herd_x * rows_per_call})"
        )
    rows_per_tile = rows // herd_x

    xrt_dtype = type_mapper(np_dtype)
    l3_ty = MemRefType.get([rows, cols], xrt_dtype)

    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_tile_ty = MemRefType.get([rows_per_call, cols], xrt_dtype, memory_space=l1_space)

    ln_func = FuncOp(
        "layer_norm_rows",
        ([l1_tile_ty, l1_tile_ty, T.i32(), T.i32()], []),
        visibility="private",
    )
    ln_func.attributes["link_with"] = StringAttr.get(KERNEL_OBJ)
    ln_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    # first row of this tile's block = loop_iv + tx * rows_per_tile
    row_map = AffineMap.get(
        0,
        2,
        [
            AffineExpr.get_add(
                AffineSymbolExpr.get(0),
                AffineExpr.get_mul(
                    AffineSymbolExpr.get(1), AffineConstantExpr.get(rows_per_tile)
                ),
            )
        ],
    )

    @FuncOp.from_py_func(l3_ty, l3_ty)
    def layer_norm_multi_row(arg0, arg1):

        @launch(operands=[arg0, arg1])
        def ln_launch(l_in, l_out):

            @segment(name="ln_seg", operands=[l_in, l_out])
            def ln_seg(s_in, s_out):

                @herd(name="ln_herd", sizes=[herd_x, 1], operands=[s_in, s_out])
                def ln_body(_tx, _ty, _sx, _sy, h_in, h_out):
                    l1_in = AllocOp(l1_tile_ty, [], [])
                    l1_out = AllocOp(l1_tile_ty, [], [])
                    cols_i32 = ConstantOp(T.i32(), cols)
                    nrows_i32 = ConstantOp(T.i32(), rows_per_call)

                    for loop_iv in range_(0, rows_per_tile, rows_per_call):
                        row = affine_apply(row_map, [loop_iv, _tx])
                        dma_memcpy_nd(
                            l1_in,
                            h_in,
                            src_offsets=[row, 0],
                            src_sizes=[rows_per_call, cols],
                            src_strides=[cols, 1],
                        )
                        CallOp(ln_func, [l1_in, l1_out, cols_i32, nrows_i32])
                        dma_memcpy_nd(
                            h_out,
                            l1_out,
                            dst_offsets=[row, 0],
                            dst_sizes=[rows_per_call, cols],
                            dst_strides=[cols, 1],
                        )
                        yield_([])

                    DeallocOp(l1_in)
                    DeallocOp(l1_out)

                ln_body.attributes["link_with"] = StringAttr.get(KERNEL_OBJ)


def layer_norm_reference(x, eps=EPS):
    """FP32 two-pass LayerNorm reference, rounded once to the input dtype.

    Two-pass -- ``mean((x - mean)^2)`` -- deliberately, not the kernel's
    one-pass ``E[x^2] - E[x]^2``. The one-pass form is what the device does and
    what makes it fast; reproducing it here would hide exactly the error it can
    introduce. See the module footgun note.
    """
    x_f32 = x.astype(np.float32)
    mean = x_f32.mean(axis=-1, keepdims=True)
    var = ((x_f32 - mean) ** 2).mean(axis=-1, keepdims=True)
    return ((x_f32 - mean) / np.sqrt(var + eps)).astype(x.dtype)
