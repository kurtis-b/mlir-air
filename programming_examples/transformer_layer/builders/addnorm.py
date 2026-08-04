# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AddNorm: weighted LayerNorm plus a residual, over ``fused_add_layer_norm_2outs``.

CONTRACT
    ``build_addnorm_module(rows, cols, ...)`` returns an ``air.ir.Module``
    whose single function takes ``x[rows, cols]``, ``residual[rows, cols]``,
    ``weight[cols]`` and ``out[rows, cols]``, all bf16, and computes per row

        out[r, :] = LayerNorm(x[r, :]) * weight + residual[r, :]

    The residual is added AFTER the normalization -- this is the post-add form.
    The statistics come from ``x`` alone; ``residual`` never enters them.

THE WEIGHT IS A RUNTIME ARGUMENT
    iron's ``addnorm`` bakes its weights into the MLIR through ``np.load()`` at
    generation time and hashes them into the artifact name, so changing a
    weight forces a full recompile. That is not reproduced here: ``weight`` is
    a plain memref argument, uploaded like any other buffer, and one compiled
    ELF serves every weight vector of that shape.

TWO OUTPUTS, ONE DRAINED
    The kernel writes its result to two buffers because the fused encoder block
    feeds it to both the FFN and the next residual. Only ``output1`` is drained
    to L3 here: a second L3 output would need a second shim S2MM channel per
    column for a byte-identical copy. ``output2`` stays in L1 and is discarded.

ONE KERNEL CALL PER TILE -- THIS IS A CORRECTNESS CONSTRAINT, NOT A TUNING KNOB
    ``rows`` must equal ``herd_x * rows_per_call``, so every tile issues exactly
    one call and the herd's loop runs a single trip. Two or more trips
    MISCOMPILE: measured on NPU2 at ``[8, 64]``, ``herd_x=1``, the one-trip form
    is exact (0 of 512 elements outside tolerance) and the two-trip form is
    garbage (491 of 512), with the same 491 whether the weight is fetched
    inside the loop or hoisted out of it, whether ``output2`` is drained to L3
    or discarded, and with ping-pong disabled or either lock-race-condition fix
    enabled. The distinguishing feature is THREE distinct L3->L1 streams per
    tile (x, residual, weight) against the two shim MM2S channels a column has;
    the two-stream builders next door -- ``layer_norm.py`` here and
    ``_build_add_2d_to_2d`` in ``llms/shared/`` -- loop correctly for as many
    trips as you like. The builder raises rather than emitting the broken form,
    because the failure looks numerical (partly-right values) rather than
    structural and would otherwise be read as a tolerance problem.

    The practical consequence is a row cap: ``rows <= herd_x * (L1 budget)``,
    which at ``cols=512`` is 64 rows over the full 8-column herd. A larger
    activation needs the weight staged through L2, or the residual folded into
    the same L3 buffer as ``x`` so that one strided DMA fetches both.

FOOTGUNS
    - Variance is ONE-PASS (``E[x^2] - E[x]^2``) inside the kernel; the
      reference below is the stable TWO-PASS f32 form. They are algebraically
      equal and numerically are not -- see ``layer_norm.py``'s note. The kernel
      clamps a negative variance to zero because ``aie::invsqrt`` of a negative
      operand returns NaN.
    - The row sum accumulates in a bf16 vector (only the sum of squares is
      f32), so the mean carries more error than the FP32 reference's. It shows
      up as a small uniform shift of a row, not as an outlier.
    - ``cols`` must be a multiple of 32 (the kernel's ``N``). A non-multiple is
      silently truncated by ``vector_chunks = cols / N``, not diagnosed.
    - ``encoder.o`` is compiled here with ONLY the addnorm half
      (``build_ffn=False``). The FFN half also defines ``ffn_gelu_bf16`` and
      ``ffn_eltwise_add_bf16_vector``, which collide with ``addnorm_ffn.o``;
      building the halves separately is what keeps one kernel object per ELF.
    - ``encoder.o`` must exist in the working directory when aiecc links.
      ``opcheck.py`` compiles it there.
    - L1 is 64 KiB per core, and the four tile buffers (x, residual, output1,
      output2) plus the weight and the 1 KiB stack have to fit. aiecc reports
      an overflow against the ``aie.tile``, not against the builder, so
      ``_l1_bytes`` below checks it up front instead.
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

# Must match `epsilon` in fused_add_layer_norm_2
# (transformer_layer/kernels/encoder_layer_norm.cc).
EPS = 1e-5

# The kernel's vector width (fused_add_layer_norm_2<bfloat16, 32>). cols must
# be a multiple of it.
ADDNORM_VEC_LEN = 32

KERNEL_OBJ = "encoder.o"

# AIE2P core-local memory, and the stack aircc reserves inside it.
L1_BYTES = 64 * 1024
L1_STACK_BYTES = 1024


def _l1_bytes(rows_per_call, cols, itemsize):
    """L1 a single tile needs: four activation tiles plus the weight vector."""
    return 4 * rows_per_call * cols * itemsize + cols * itemsize + L1_STACK_BYTES


@module_builder
def build_addnorm_module(rows, cols, np_dtype=bfloat16, herd_x=8, rows_per_call=None):
    """Build weighted-LayerNorm-plus-residual over an ``herd_x x 1`` herd.

    Args:
        rows, cols: L3 activation shape. ``cols`` must be a multiple of
            ``ADDNORM_VEC_LEN``; ``rows`` must equal ``herd_x * rows_per_call``
            -- see the one-call-per-tile constraint in the module docstring.
        np_dtype: element type; bf16, matching the kernel's C linkage.
        herd_x: AIE columns. Each tile drives three L3->L1 streams (x,
            residual, weight) and one L1->L3 stream.
        rows_per_call: rows resident in L1 per kernel invocation. Defaults to
            ``rows // herd_x``, the only value that keeps the herd loop to one
            trip.

    Returns:
        air.ir.Module with one function ``addnorm(x, residual, weight, out)``.
    """
    if cols % ADDNORM_VEC_LEN:
        raise ValueError(
            f"cols ({cols}) must be a multiple of {ADDNORM_VEC_LEN}; the kernel "
            "truncates the remainder instead of failing"
        )
    if rows % herd_x:
        raise ValueError(f"rows ({rows}) must be divisible by herd_x ({herd_x})")
    rows_per_tile = rows // herd_x
    if rows_per_call is None:
        rows_per_call = rows_per_tile
    if rows_per_call != rows_per_tile:
        raise ValueError(
            f"rows_per_call must be rows // herd_x ({rows_per_tile}), got "
            f"{rows_per_call}. More than one call per tile miscompiles on "
            "three-input-stream herds -- see the module docstring; this is a "
            "correctness constraint, not a performance one."
        )

    need = _l1_bytes(rows_per_call, cols, np.dtype(np_dtype).itemsize)
    if need > L1_BYTES:
        raise ValueError(
            f"[{rows}, {cols}] over an {herd_x}-column herd needs {need} bytes "
            f"of L1 per tile, over the {L1_BYTES}-byte budget. Raise herd_x or "
            "lower rows; aiecc would otherwise report this against the "
            "aie.tile, far from the cause."
        )

    xrt_dtype = type_mapper(np_dtype)
    l3_act_ty = MemRefType.get([rows, cols], xrt_dtype)
    l3_w_ty = MemRefType.get([cols], xrt_dtype)

    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_tile_ty = MemRefType.get([rows_per_call, cols], xrt_dtype, memory_space=l1_space)
    l1_w_ty = MemRefType.get([cols], xrt_dtype, memory_space=l1_space)

    addnorm_func = FuncOp(
        "fused_add_layer_norm_2outs",
        (
            [l1_tile_ty, l1_tile_ty, l1_w_ty, l1_tile_ty, l1_tile_ty, T.i32(), T.i32()],
            [],
        ),
        visibility="private",
    )
    addnorm_func.attributes["link_with"] = StringAttr.get(KERNEL_OBJ)
    addnorm_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

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

    @FuncOp.from_py_func(l3_act_ty, l3_act_ty, l3_w_ty, l3_act_ty)
    def addnorm(arg0, arg1, arg2, arg3):

        @launch(operands=[arg0, arg1, arg2, arg3])
        def addnorm_launch(l_x, l_res, l_w, l_out):

            @segment(name="addnorm_seg", operands=[l_x, l_res, l_w, l_out])
            def addnorm_seg(s_x, s_res, s_w, s_out):

                @herd(
                    name="addnorm_herd",
                    sizes=[herd_x, 1],
                    operands=[s_x, s_res, s_w, s_out],
                )
                def addnorm_body(_tx, _ty, _sx, _sy, h_x, h_res, h_w, h_out):
                    l1_x = AllocOp(l1_tile_ty, [], [])
                    l1_res = AllocOp(l1_tile_ty, [], [])
                    l1_w = AllocOp(l1_w_ty, [], [])
                    l1_out1 = AllocOp(l1_tile_ty, [], [])
                    l1_out2 = AllocOp(l1_tile_ty, [], [])
                    cols_i32 = ConstantOp(T.i32(), cols)
                    nrows_i32 = ConstantOp(T.i32(), rows_per_call)

                    for loop_iv in range_(0, rows_per_tile, rows_per_call):
                        row = affine_apply(row_map, [loop_iv, _tx])
                        # Re-fetched every iteration even though the weight
                        # vector never changes. Hoisting it out of the loop is
                        # the obvious optimization and it silently corrupts the
                        # result: aircc ping-pongs the DMA-fed L1 buffers, and
                        # a buffer filled once outside the loop is read from
                        # the wrong half on later iterations. The symptom is a
                        # partly-correct output -- some columns exact, others
                        # far off -- with the FIRST iteration wrong too, which
                        # reads like a numerical problem rather than a
                        # scheduling one. Measured: cols=64, rows=8,
                        # rows_per_call=4 goes from 0 mismatches (one
                        # iteration) to 497/512 (two).
                        dma_memcpy_nd(
                            l1_w,
                            h_w,
                            src_offsets=[0],
                            src_sizes=[cols],
                            src_strides=[1],
                        )
                        dma_memcpy_nd(
                            l1_x,
                            h_x,
                            src_offsets=[row, 0],
                            src_sizes=[rows_per_call, cols],
                            src_strides=[cols, 1],
                        )
                        dma_memcpy_nd(
                            l1_res,
                            h_res,
                            src_offsets=[row, 0],
                            src_sizes=[rows_per_call, cols],
                            src_strides=[cols, 1],
                        )
                        CallOp(
                            addnorm_func,
                            [
                                l1_x,
                                l1_res,
                                l1_w,
                                l1_out1,
                                l1_out2,
                                cols_i32,
                                nrows_i32,
                            ],
                        )
                        dma_memcpy_nd(
                            h_out,
                            l1_out1,
                            dst_offsets=[row, 0],
                            dst_sizes=[rows_per_call, cols],
                            dst_strides=[cols, 1],
                        )
                        yield_([])

                    DeallocOp(l1_x)
                    DeallocOp(l1_res)
                    DeallocOp(l1_w)
                    DeallocOp(l1_out1)
                    DeallocOp(l1_out2)

                addnorm_body.attributes["link_with"] = StringAttr.get(KERNEL_OBJ)


def addnorm_reference(x, residual, weight, eps=EPS):
    """FP32 reference for ``LayerNorm(x) * weight + residual``.

    Two-pass variance in f32, one rounding to bf16 at the end. Not the kernel's
    one-pass form and not its bf16 intermediate roundings -- the point of the
    check is to measure that gap, not to reproduce it.
    """
    x_f32 = x.astype(np.float32)
    mean = x_f32.mean(axis=-1, keepdims=True)
    var = ((x_f32 - mean) ** 2).mean(axis=-1, keepdims=True)
    normed = (x_f32 - mean) / np.sqrt(var + eps)
    out = normed * weight.astype(np.float32) + residual.astype(np.float32)
    return out.astype(x.dtype)
