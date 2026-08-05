# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""2-D element-wise multiply, the last of the fine-grained decomposition's pieces.

CONTRACT
    ``build_elementwise_mul_module(rows, cols, ...)`` returns an
    ``air.ir.Module`` whose single function takes three ``[rows, cols]`` bf16
    memrefs and computes ``out = a * b`` element-wise. All three arguments stay
    2-D at the L3 boundary; the collapse to 1-D happens inside the launch, so a
    downstream launch reads the output without an ``expand_shape``.

    The port of iron's ``elementwise_mul`` operator
    (``iron/operators/elementwise_mul``), which had no counterpart anywhere in
    ``programming_examples/`` -- the one operator in the plan built from
    nothing. Shape and structure follow ``builders/elementwise_add.py``
    (``_build_add_2d_to_2d``'s streaming form: each tile walks its contiguous
    chunk a row-sized tile at a time); the operation follows
    ``primitives/vector_examples/vector_mul`` (an ``arith.mulf`` on L1 vector
    subviews, direct codegen, no external kernel object to link).

    iron's operator multiplies by a compile-time SCALAR broadcast
    (``attn_scale``, its only use). This one takes a full second tensor
    instead: the ``runlist`` mode needs a per-column gamma
    (``normed * ln_weight``), and a materialized broadcast operand covers both
    the scalar and the row-vector case without a second kernel -- the same
    trade iron's ``causal_mask`` makes when it materializes its triangular
    mask as a static input.

THE f32 LEGALIZATION CONSTRAINT, CHECKED BEFORE THIS WAS DESIGNED
    ``weighted_rms_norm.py`` records that the AIE vector unit does not legalize
    f32 vector element-wise multiply -- its epilogue runs in bf16 vectors for
    exactly that reason. This multiply is therefore bf16 end to end:
    ``arith.mulf`` on ``vector<16xbf16>``, the form weighted_rms_norm already
    ships. Keep it bf16; "improving" the compute type to f32 moves the failure
    to aiecc's legalizer.

FOOTGUNS
    - The device multiply is a single bf16 rounding of the exact product (two
      8-bit mantissas fit f32's 24 exactly), so the reference below computes
      the f32 product and rounds ONCE. Multiplying in bf16 in the reference
      would make the check agree with the device for the wrong reason.
    - ``rows`` must divide by ``herd_x * herd_y`` -- each tile takes a
      contiguous ``rows * cols / tiles`` chunk and walks it a row at a time, so
      a remainder is silently unprocessed rather than diagnosed.
    - ``herd_y`` must be 1. Three independent shim DMA streams per tile (a in,
      b in, out) is the same channel budget as ``elementwise_add``, and
      ``herd_y > 1`` exhausts it -- aircc fails to place with "out of
      channels".
    - A fault injected into ``a`` moves the output by ``delta * b`` at that
      element, so a negative control needs ``b`` bounded away from zero there.
      ``prepare_elementwise_mul`` draws ``b`` from ``uniform(0.5, 1.5)`` -- the
      gamma-shaped operand the ``runlist`` mode actually feeds -- for exactly
      that reason; a standard-normal ``b`` puts the control inside the
      tolerance band with a few percent probability per run.
"""

import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects import arith
from air.dialects.memref import AllocOp, DeallocOp, subview
from air.dialects.memref import collapse_shape as memref_collapse_shape
from air.dialects.vector import transfer_read, transfer_write
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import type_mapper

range_ = for_


@module_builder
def build_elementwise_mul_module(
    rows, cols, np_dtype=bfloat16, vector_size=16, herd_x=8, herd_y=1
):
    """Build the 2-D element-wise multiply ``out = a * b`` over an ``herd_x`` herd.

    Args:
        rows, cols: the L3 tensor shape. ``rows`` must divide by
            ``herd_x * herd_y`` and ``cols`` by ``vector_size``.
        np_dtype: element type; bf16 in every transformer-block use, and the
            compute type too -- see the module docstring on why not f32.
        vector_size: AIE vector lane count for the bf16 element-wise multiply.
        herd_x, herd_y: AIE tile grid. ``herd_y`` must be 1.

    Returns:
        air.ir.Module with one function ``eltwise_mul_2d(a, b, out)``.
    """
    if herd_y != 1:
        raise ValueError(
            f"herd_y must be 1 (three shim DMAs per tile); got herd_y={herd_y}"
        )
    tiles = herd_x * herd_y
    if rows % tiles:
        raise ValueError(f"rows ({rows}) must be divisible by herd_x*herd_y ({tiles})")
    if cols % vector_size:
        raise ValueError(f"cols ({cols}) must be divisible by vector_size")

    xrt_dtype = type_mapper(np_dtype)
    n = rows * cols
    l3_2d_ty = MemRefType.get([rows, cols], xrt_dtype)
    l3_1d_ty = MemRefType.get([n], xrt_dtype)
    chunk_size = n // tiles
    tile_n = cols
    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_ty = MemRefType.get([tile_n], xrt_dtype, memory_space=l1_space)
    vec_ty = VectorType.get([vector_size], xrt_dtype)
    identity_map = AffineMapAttr.get(AffineMap.get_identity(1))

    @FuncOp.from_py_func(l3_2d_ty, l3_2d_ty, l3_2d_ty)
    def eltwise_mul_2d(arg0_2d, arg1_2d, arg2_2d):
        @launch(operands=[arg0_2d, arg1_2d, arg2_2d])
        def mul_launch(l_a, l_b, l_out):
            a_flat = memref_collapse_shape(l3_1d_ty, l_a, [[0, 1]])
            b_flat = memref_collapse_shape(l3_1d_ty, l_b, [[0, 1]])
            out_flat = memref_collapse_shape(l3_1d_ty, l_out, [[0, 1]])

            @segment(name="mul_seg", operands=[a_flat, b_flat, out_flat])
            def mul_seg(s_a, s_b, s_out):
                offset_map = AffineMap.get(
                    0,
                    3,
                    [
                        AffineExpr.get_add(
                            AffineSymbolExpr.get(0),
                            AffineExpr.get_mul(
                                AffineExpr.get_add(
                                    AffineExpr.get_mul(
                                        AffineSymbolExpr.get(1),
                                        AffineConstantExpr.get(herd_y),
                                    ),
                                    AffineSymbolExpr.get(2),
                                ),
                                AffineConstantExpr.get(chunk_size),
                            ),
                        )
                    ],
                )

                @herd(
                    name="mul_herd",
                    sizes=[herd_x, herd_y],
                    operands=[s_a, s_b, s_out],
                )
                def mul_body(_tx, _ty, _sx, _sy, h_a, h_b, h_out):
                    l1_a = AllocOp(l1_ty, [], [])
                    l1_b = AllocOp(l1_ty, [], [])
                    l1_out = AllocOp(l1_ty, [], [])
                    c0 = arith.ConstantOp.create_index(0)
                    cst0 = arith.ConstantOp(xrt_dtype, 0.0)

                    for loop_iv in range_(0, chunk_size, tile_n):
                        offset = affine_apply(offset_map, [loop_iv, _tx, _ty])
                        dma_memcpy_nd(
                            l1_a,
                            h_a,
                            src_offsets=[offset],
                            src_sizes=[tile_n],
                            src_strides=[1],
                        )
                        dma_memcpy_nd(
                            l1_b,
                            h_b,
                            src_offsets=[offset],
                            src_sizes=[tile_n],
                            src_strides=[1],
                        )
                        for j in range_(0, tile_n, vector_size):
                            sub_a = subview(l1_a.result, [j], [vector_size], [1])
                            sub_b = subview(l1_b.result, [j], [vector_size], [1])
                            sub_out = subview(l1_out.result, [j], [vector_size], [1])
                            v_a = transfer_read(
                                vec_ty, sub_a, [c0], identity_map, cst0, [True]
                            )
                            v_b = transfer_read(
                                vec_ty, sub_b, [c0], identity_map, cst0, [True]
                            )
                            v_prod = arith.mulf(v_a, v_b)
                            transfer_write(
                                None, v_prod, sub_out, [c0], identity_map, [True]
                            )
                            yield_([])
                        dma_memcpy_nd(
                            h_out,
                            l1_out,
                            dst_offsets=[offset],
                            dst_sizes=[tile_n],
                            dst_strides=[1],
                        )
                        yield_([])
                    DeallocOp(l1_a)
                    DeallocOp(l1_b)
                    DeallocOp(l1_out)


def elementwise_mul_reference(a, b):
    """FP32 reference for ``out = a * b``, rounded once to the input dtype.

    The exact product of two bf16 values is representable in f32 (8-bit
    mantissas), so this is bit-for-bit what an IEEE bf16 multiply returns and
    the check's tolerance budget does no work here -- like
    ``elementwise_add_reference``, whose measured ``atol_required`` is exactly
    zero.
    """
    return (a.astype(np.float32) * b.astype(np.float32)).astype(a.dtype)


def broadcast_row_weight(weight, rows):
    """Materialize a ``[cols]`` per-column weight as the ``[rows, cols]`` operand.

    The ``runlist`` mode's gamma multiply: ``build_elementwise_mul_module``
    takes two full tensors (see the module docstring for why there is no
    broadcast form), so the row vector is tiled on the host and declared a
    content-keyed static buffer by the caller -- uploaded once, like iron's
    materialized causal mask.
    """
    weight = np.asarray(weight)
    return np.ascontiguousarray(np.broadcast_to(weight, (rows, weight.shape[-1])))
