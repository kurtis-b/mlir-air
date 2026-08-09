# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Row-wise softmax over the external streaming-softmax kernel family.

CONTRACT
    ``build_softmax_module(rows, cols, ...)`` returns an ``air.ir.Module``
    whose single function takes two ``[rows, cols]`` bf16 memrefs and computes
    the softmax of each row:

        y[r, :] = exp(x[r, :] - max(x[r, :])) / sum(exp(x[r, :] - max(x[r, :])))

    Rows are split across an ``herd_x x 1`` AIE grid; each tile softmaxes
    ``rows_per_call`` consecutive rows per kernel invocation, the same shape
    ``builders/layer_norm.py`` uses and for the same reason.

WHY THE STREAMING FAMILY AND NOT ``softmax_bf16``
    ``programming_examples/softmax/softmax.cc`` exports two things. The
    single-shot ``softmax_bf16`` is the one the softmax example dispatches, and
    it is NOT usable here for two independent reasons: it does not subtract a
    row max before exponentiating, which is a real overflow risk on attention
    scores rather than on the example's ``randn`` gate input, and it hardcodes
    ``zero_vectorized<bfloat16, 1, 256, 16>``, which fixes the row width.

    Behind ``-DSOFTMAX_STREAMING`` the same file exports the online-softmax
    recurrence -- ``init_softmax_scale_buffer``, ``partial_softmax_rows_bf16``,
    ``normalize_softmax_rows_bf16``, ``copy_softmax_scale_bf16``. That family
    DOES subtract a running row max (``partial_softmax_alias_bf16`` computes it
    and rebases the exponentials on it), and ``partial_softmax_rows_bf16``
    takes ``row_width`` and ``num_rows`` as runtime ``int32``, so it is
    shape-generic. Its degenerate one-block call sequence -- init, one partial
    over the whole row, normalize -- IS a plain row-wise softmax, and that is
    what this builder emits.

    `[2026-08-09]` Doc 26 §5 attributes the missing row-max to the streaming
    family. It is the single-shot kernel that lacks it; the streaming one is
    numerically the safe choice, which is the opposite of what that note
    implies.

THE SCALE ARGUMENT, WHICH IS NOT WHAT IT LOOKS LIKE
    ``partial_softmax_rows_bf16`` passes ``SM_LOG2E`` (1.4453125, a
    bf16-representable log2(e)) as the ``scale`` of the row it exponentiates.
    That is the BASE CONVERSION for an ``exp2``-based ``exp``, not a free
    parameter -- the kernel computes ``exp2(x * log2(e) - m)``, which is
    ``exp(x - m/log2(e))``. For a plain softmax ``SM_LOG2E`` is therefore
    exactly the right value and there is nothing to change.

    It matters for ATTENTION, which wants ``softmax(QK^T / sqrt(head_dim))``.
    Doc 26 §5 records that as "the entry point hardcodes SM_LOG2E as its scale,
    so 1/sqrt(head_dim) has to be applied upstream or the entry point
    extended". True of the entry point, and the plumbing one level down already
    exists: ``partial_softmax_alias_bf16`` takes ``scale`` as an argument, so
    folding attention's scale in is ``scale = SM_LOG2E / sqrt(head_dim)`` and a
    parameter on the wrapper, not a kernel rewrite. Until something needs it,
    the caller scales upstream -- which is what ``pattern/offload`` does on the
    host today.

FOOTGUNS
    - ``cols`` must be a multiple of 64 (``SM_VEC_LEN``). There is no scalar
      tail: ``elem_iters = row_width / SM_VEC_LEN`` truncates and the
      remainder of every row is silently left unexponentiated and
      unnormalized.
    - The scale buffer is FOUR bands of ``num_rows`` bf16, not one: running
      max, current max, running denominator, and the O-rescale factor. Sizing
      it ``num_rows`` corrupts the three bands past the first, which presents
      as wrong denominators rather than as a fault.
    - THREE L1 buffers, each with exactly ONE role: ``l1_in`` is only ever a
      DMA destination, ``l1_exp`` only ever kernel scratch, ``l1_out`` only
      ever a DMA source. That is not style. The first version of this builder
      saved an allocation by normalizing back into ``l1_in`` -- dead by then,
      and legal as far as ``normalize``'s ``__restrict`` is concerned -- and
      the design returned THE INPUT UNCHANGED from hardware at all three
      shapes (``abs_err_max`` equal to the input's own max, 4.9 / 5.3 / 4.7).
      A buffer that is both a DMA destination and a kernel output does not
      read back the value the kernel wrote. Give every L1 buffer one role,
      exactly as ``builders/layer_norm.py`` does.
    - The scale buffer must be re-initialized per call. It is stateful across
      ``partial`` calls by design (that is what makes the family streaming),
      so a loop that inits once and then walks row blocks computes one softmax
      smeared over every block it visited.
    - ``softmax_streaming.o`` must exist in the working directory when aiecc
      links. ``opcheck.py`` compiles it there; a builder used elsewhere must
      call ``external_kernels.compile_softmax_streaming()`` first.
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

# The vector width softmax_streaming.o is compiled with (SM_VEC_LEN). cols must
# be a multiple of it or every row keeps an unprocessed tail.
SM_VEC_LEN = 64

# Bands in the scale buffer: running max, current max, running denominator,
# O-rescale factor. All four are indexed as `scale_buffer[band * num_rows + r]`.
SM_SCALE_BANDS = 4

KERNEL_OBJ = "softmax_streaming.o"


@module_builder
def build_softmax_module(rows, cols, np_dtype=bfloat16, herd_x=8, rows_per_call=8):
    """Build row-wise softmax over an ``herd_x x 1`` herd.

    Args:
        rows, cols: L3 tensor shape. ``rows`` must divide by
            ``herd_x * rows_per_call``; ``cols`` by ``SM_VEC_LEN``.
        np_dtype: element type; bf16, matching the kernel's C linkage.
        herd_x: AIE columns. Two shim DMAs per tile (in, out), so this stays
            inside the two-stream-per-column budget at the full 8.
        rows_per_call: rows resident in L1 per call. ``3 * rows_per_call *
            cols * 2`` bytes for the three row buffers, plus a negligible
            ``4 * rows_per_call * 2`` for the scale buffer, must fit L1 with
            ping-pong. The row WIDTH is what bounds this: 8 fits at cols 768,
            2 at cols 4096.

    Returns:
        air.ir.Module with one function ``softmax_rows(x, y)``.
    """
    if cols % SM_VEC_LEN:
        raise ValueError(
            f"cols ({cols}) must be a multiple of SM_VEC_LEN ({SM_VEC_LEN}); "
            "the kernel has no scalar tail and would leave the remainder of "
            "every row unexponentiated"
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
    l1_scale_ty = MemRefType.get(
        [SM_SCALE_BANDS * rows_per_call], xrt_dtype, memory_space=l1_space
    )

    def _extern(name, arg_types):
        func = FuncOp(name, (arg_types, []), visibility="private")
        func.attributes["link_with"] = StringAttr.get(KERNEL_OBJ)
        func.attributes["llvm.emit_c_interface"] = UnitAttr.get()
        return func

    # Signatures exactly as softmax.cc declares them -- note that `partial`
    # takes (in, out, scale) and `normalize` takes (in, scale, out). The two
    # orders differ, and swapping them produces a run that completes and
    # normalizes by whatever the exponentials happened to leave in band 2.
    init_func = _extern("init_softmax_scale_buffer", [l1_scale_ty, T.i32()])
    partial_func = _extern(
        "partial_softmax_rows_bf16",
        [l1_tile_ty, l1_tile_ty, l1_scale_ty, T.i32(), T.i32()],
    )
    normalize_func = _extern(
        "normalize_softmax_rows_bf16",
        [l1_tile_ty, l1_scale_ty, l1_tile_ty, T.i32(), T.i32()],
    )

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
    def softmax_rows(arg0, arg1):

        @launch(operands=[arg0, arg1])
        def sm_launch(l_in, l_out):

            @segment(name="sm_seg", operands=[l_in, l_out])
            def sm_seg(s_in, s_out):

                @herd(name="sm_herd", sizes=[herd_x, 1], operands=[s_in, s_out])
                def sm_body(_tx, _ty, _sx, _sy, h_in, h_out):
                    l1_in = AllocOp(l1_tile_ty, [], [])
                    l1_exp = AllocOp(l1_tile_ty, [], [])
                    l1_out = AllocOp(l1_tile_ty, [], [])
                    l1_scale = AllocOp(l1_scale_ty, [], [])
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
                        # Per row block, not once per tile: the scale buffer
                        # carries the running max and denominator ACROSS
                        # partial calls, so a stale one would fold the previous
                        # block's statistics into this block's softmax.
                        CallOp(init_func, [l1_scale, nrows_i32])
                        CallOp(
                            partial_func,
                            [l1_in, l1_exp, l1_scale, cols_i32, nrows_i32],
                        )
                        # Into its own buffer, never back into l1_in: see the
                        # one-role-per-buffer footgun.
                        CallOp(
                            normalize_func,
                            [l1_exp, l1_scale, l1_out, cols_i32, nrows_i32],
                        )
                        dma_memcpy_nd(
                            h_out,
                            l1_out,
                            dst_offsets=[row, 0],
                            dst_sizes=[rows_per_call, cols],
                            dst_strides=[cols, 1],
                        )
                        yield_([])

                    DeallocOp(l1_in)
                    DeallocOp(l1_exp)
                    DeallocOp(l1_out)
                    DeallocOp(l1_scale)

                sm_body.attributes["link_with"] = StringAttr.get(KERNEL_OBJ)


def softmax_reference(x):
    """FP32 max-subtracted row softmax, rounded once to the input dtype.

    Written in the numerically standard form -- subtract the row max, exponentiate,
    divide by the sum -- in f32. The device computes ``exp2(x * log2(e) - m)``
    over bf16 vectors instead; expressing that here would reproduce the
    kernel's own arithmetic and compare it against itself.
    """
    x_f32 = x.astype(np.float32)
    shifted = x_f32 - x_f32.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / exp.sum(axis=-1, keepdims=True)).astype(x.dtype)
