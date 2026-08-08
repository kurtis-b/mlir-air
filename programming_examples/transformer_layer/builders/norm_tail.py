# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The norm tail as a three-herd pipeline: add -> norm -> scale, resident in L1.

CONTRACT
    ``build_norm_tail_module(rows, cols, ...)`` returns an ``air.ir.Module``
    whose single function takes ``packed[rows, 2, cols]``, ``gamma[cols]`` and
    ``out[rows, cols]``, all bf16, and computes per row

        out[r, :] = LayerNorm(packed[r, 0, :] + packed[r, 1, :]) * gamma

    -- the PRE-ADD weighted norm, the same function as
    ``build_addnorm_module(pre_add=True)`` and checked against the same oracle
    (``norm_tail_reference`` below delegates to ``addnorm_pre_add_reference``).
    What differs is the dataflow: three herds in one segment, one stage each,

        stage_add    packed x|residual, ONE strided L3 fetch -> AtoB (L1->L1)
        stage_norm   AtoB -> layer_norm_rows -> BtoC (L1->L1)
        stage_scale  BtoC * gamma (one L3 fetch)  -> L3 out

    so the sum and the normalized tensor never leave the array. The decomposed
    tail this replaces (``pattern/fused``'s ln stages) stages both through L3.

    Callers pack with ``norm_tail_device_inputs(x, residual)``; see below for
    why the packing is load-bearing and not a convenience.

WHY x AND residual ARE PACKED IN ONE BUFFER, AND WHY PER-ROW PLANE PAIRS
    A column has two shim MM2S channels and the budget is PER COLUMN across the
    whole segment. Three stacked 8-wide herds put one tile of every stage into
    each column, so streaming x, residual and gamma separately needs three
    L3-facing streams per column -- and AIR then packet-multiplexes all of them
    onto one queue, the path that silently miscompiles past one trip on more
    than one column (see ``builders/addnorm.py``'s multi-trip section). Packing
    x and residual into ONE buffer fetched by ONE DMA leaves two streams per
    column (packed in, gamma in; the output is S2MM and not in this budget)
    and the lowered module carries ZERO packet-typed channels.

    The LAYOUT is ``[rows, 2, cols]`` -- row r's x then row r's residual, back
    to back -- and not the plane-major ``[2, rows, cols]`` this phase's spec
    proposed, because plane-major CANNOT BE PROGRAMMED at the block's shape.
    Its band fetch is a 3-D DMA whose plane stride is ``rows * cols``, and a
    shim ``aie.dma_bd`` stride is capped at 2^20:

        error: 'aie.dma_bd' op Stride 2 exceeds the [1:1048576] range.
        (strides = [3072, 3145728, 768, 1] at 4096 x 768)

    -- measured through full aiecc; the spec's probe stopped at air-opt, where
    both layouts lower. At 128 rows plane-major compiles (stride 98304) and it
    was measured numerically exact on hardware, so the failure is purely the
    BD range, not the arithmetic. ``[rows, 2, cols]`` makes a band of
    ``rows_per_call`` rows fully CONTIGUOUS in L3 (max stride ``2*cols``), at
    any row count.

    `[2026-08-08]` PLANE-MAJOR IS BACK AS AN OPTION, for a reason the original
    choice did not weigh: WHO WRITES THE PACKED BUFFER. Row-interleaving is
    free when the host packs, and the wrong shape entirely when the producer is
    a device kernel -- which is the case in ``pattern/fused``, the mode this
    pipeline exists to fix. There ``x``'s partner is ``attn_out``, written by a
    GEMM in a previous ELF, and the second point's is ``ffn_out``. Interleaving
    those means teaching the SHARED GEMM builder (used by every LLM model and
    the whole kernel registry) to write strided, and packing them on the host
    means a round trip through the host -- the exact sync ``fused`` exists to
    remove.

    Plane-major has neither problem, because plane 0 is a contiguous
    ``[rows, cols]`` block AT OFFSET 0: an ordinary contiguous producer writes
    it with no change at all, and the host fills plane 1. It is bounded by
    ``rows * cols <= SHIM_BD_STRIDE_MAX``, so at ``cols=768`` it serves
    ``rows <= 1365`` -- the 64…1024 rungs of the sequence ladder and not 2048
    and up. That is a real limit and it is stated rather than worked around:
    the measurement it buys is what should decide whether the strided-GEMM work
    is worth doing for the longer shapes.

    The cost of row-pairing is that x is no longer plane-contiguous inside the
    L1 tile -- and that costs nothing HERE, because the only consumer of the
    packed layout is stage_add's direct codegen (see the next section), which
    addresses row r's x at ``[r*2*cols]`` and its residual at
    ``[r*2*cols + cols]`` and writes the SUM contiguous. The one C kernel in
    the pipeline (``layer_norm_rows``) only ever sees that contiguous sum.

WHY THE ADD IS DIRECT CODEGEN AND NOT AN EXTERNAL KERNEL -- MEASURED
    The obvious stage_add body is a call to the existing C add
    (``ffn_eltwise_add_bf16_vector``) on two subviews of the packed L1 tile.
    The plane-1 subview's type is
    ``memref<rpc x cols x bf16, strided<[cols, 1], offset: rpc*cols>, 2>``,
    which does not cast to the identity layout a callee normally declares --
    H7's offset-subview wall, one level down. Declaring the STRIDED types on
    the callee gets that module through ``air-dma-to-channel`` (which is where
    this phase's spec measured it lowering cleanly), and then DIES in aircc:
    ``air-to-aie`` NORMALIZES every external callee signature to the identity
    layout -- deliberately, "external kernels use C ABI (raw pointers), so MLIR
    layout metadata is irrelevant" (mlir/lib/Conversion/AIRToAIEPass.cpp, the
    normalizedInputs block) -- and then ``memref.cast`` refuses
    strided-with-offset -> identity:

        error: cannot cast operand type 'memref<8x768xbf16, strided<[768, 1],
        offset: 6144>, 2 : i32>' to normalized function type
        'memref<8x768xbf16, 2 : i32>'

    So no strided operand can reach an external kernel today, and the spec's
    open question -- whether the lowered call passes a base pointer that
    includes the subview offset -- is MOOT: the call never compiles. Fixing
    that is an ``mlir/`` change, which this phase must not make.

    The way through that changes neither C nor compiler: the add is the
    ELEMENTWISE_ADD BUILDER'S OWN STAGE BODY -- direct ``arith.addf`` codegen
    on ``vector<16xbf16>``, the exact loop ``_build_add_2d_to_2d`` ships -- run
    on the packed tile held FLAT: row r's x vector j lives at
    ``[r*2*cols + j]`` and its residual at ``[r*2*cols + cols + j]``, both
    plain 1-D subviews of the kind ``builders/elementwise_mul.py`` already
    runs on cores. No callee, no strided type, no base-pointer question left
    to answer on hardware; the plane arithmetic is compiled MLIR, checked by
    the asymmetric-input discipline below.

NO PLACEMENT, NO BUFFER DEPTH -- DELIBERATELY
    This builder writes no tile coordinate and no channel depth anywhere.
    ``air-place-herds`` places the three herds (measured: it does, at
    ``herd_x=8``, 24 tiles of NPU2's 32) and the ping-pong labelling chooses
    buffer depth. A placement attribute here would falsify the phase's claim
    that the compiler derives the pipeline.

FOOTGUNS
    - ``cols`` must be a multiple of 16 (``NORM_TAIL_VEC_LEN``): the add and
      scale vector loops and ``layer_norm_rows`` all step by 16 with no scalar
      tail, so a non-multiple silently drops the remainder.
    - Every L1 tile here is FLAT (``[2*rpc*cols]`` packed, ``[rpc*cols]``
      inter-stage), and ``layer_norm_rows`` is declared with flat operands.
      The C kernel reads linearly from a base pointer, so this is ABI-neutral;
      what it buys is that no subview in the design is rank-reducing or
      strided -- the class the compiler cannot lower (see above).
    - ``layer_norm.o`` must exist in the working directory when aiecc links
      (stage_norm's ``layer_norm_rows``; the other two stages link nothing).
      ``compile_norm_tail_kernels()`` builds it; ``opcheck.py`` calls it.
    - The packing helper's plane ORDER is x then residual. The add is
      symmetric so a swap is invisible here, but callers that later feed the
      raw sum forward must not rely on plane order they did not check.
    - ``layer_norm_rows`` keeps its statistics in f32 and computes variance
      TWO-PASS (``E[(x - mean)^2]``), so rows at a large common offset
      normalize correctly. The one-pass ``E[x^2] - E[x]^2`` form with a bf16
      row sum that this pipeline FIRST shipped loses such a row's variance
      entirely (cancels below zero, clamps, normalizes by ``1/sqrt(eps)``) --
      do not reintroduce it; the ``128x768_offset`` opcheck row exists to
      catch exactly that. What still rounds in bf16 on an offset row is
      stage_add's SUM: the reference adds in f32, so a nonzero residual at a
      large common sum costs half an ulp OF THE SUM per element, amplified
      by ``1/sigma``.
    - Against the fused addnorm kernel this pipeline carries ONE extra bf16
      rounding: the normalized tensor is materialized in bf16 between
      stage_norm and stage_scale, where ``fused_add_layer_norm_1outs`` folds
      the gamma multiply into the same pass. That is the honest cost of the
      decomposition and it is small (the fused kernel's own vectors are bf16
      too); it does not stage anything through L3.
    - Do not add a third L3-facing stream per column (a second input, a debug
      drain). It re-enters the packet path, which is correct at one trip and
      silently wrong past it. The driver's structural check counts
      ``npu_dma_packet`` in the lowered IR and requires zero.
"""

import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects import arith
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp, subview
from air.dialects.vector import transfer_read, transfer_write
from air.dialects.func import FuncOp, CallOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import type_mapper

from builders.addnorm import addnorm_pre_add_reference

range_ = for_

# Every stage steps its vectors by 16 with no scalar tail: the add and scale
# loops below, and layer_norm_rows (LN_VEC_LEN).
NORM_TAIL_VEC_LEN = 16

# Must match kEpsilon in programming_examples/layer_norm/layer_norm.cc.
EPS = 1e-5

# stage_norm: the multi-row LayerNorm layer_norm.py already links. The only
# external object in the design; stage_add and stage_scale are direct codegen.
NORM_KERNEL_OBJ = "layer_norm.o"
NORM_SYMBOL = "layer_norm_rows"

# AIE2P core-local memory, and the stack aircc reserves inside it.
L1_BYTES = 64 * 1024
L1_STACK_BYTES = 1024

# Inter-stage channels. Sized [herd_x, 1] so column i of each herd talks only
# to column i of the next -- the channel bundle is indexed by the herd's tx.
CHANNEL_A2B = "norm_tail_a2b"
CHANNEL_B2C = "norm_tail_b2c"


def _stage_l1_bytes(rows_per_call, cols, itemsize):
    """L1 the worst-case stage (stage_add) allocates per tile.

    stage_add holds the flat packed tile (two planes) plus the flat sum tile,
    and aircc PING-PONGS BOTH -- the DMA-fed packed tile and the channel-put
    sum tile alike. Measured at ``rows_per_call=8``, ``cols=768``: four
    buffers of 24+24+12+12 KiB against the 64 KiB tile, refused by aiecc's
    allocator ("allocated buffers exceeded available memory"). The factor of
    two below is that measurement, which is why 8 is not the default.
    """
    return 2 * 3 * rows_per_call * cols * itemsize + L1_STACK_BYTES


def compile_norm_tail_kernels():
    """Build layer_norm.o into the CWD, where aiecc's link_with looks."""
    import shared.infra.external_kernels as ek

    ek.compile_layer_norm()


SHIM_BD_STRIDE_MAX = 1 << 20


@module_builder
def build_norm_tail_module(
    rows,
    cols,
    np_dtype=bfloat16,
    herd_x=8,
    rows_per_call=4,
    plane_major=False,
    mirror_out=False,
):
    """Build the three-herd norm-tail pipeline.

    Args:
        rows, cols: activation shape. ``rows`` must divide by
            ``herd_x * rows_per_call``; ``cols`` by ``NORM_TAIL_VEC_LEN``.
        np_dtype: element type; bf16, matching layer_norm_rows's C linkage.
        herd_x: columns per herd. Three herds of ``[herd_x, 1]`` stack one tile
            of every stage into each column; the L3 budget (two shim MM2S per
            column) is exactly met by the packed fetch and the gamma fetch.
        rows_per_call: rows resident per tile per trip; each tile runs
            ``rows // herd_x // rows_per_call`` trips. 4 is the largest value
            that fits at ``cols=768`` WITH aircc's ping-pong doubling of both
            of stage_add's tiles -- 8 was measured to overflow the 64 KiB
            tile; see ``_stage_l1_bytes``.

    Returns:
        air.ir.Module with one function ``norm_tail(packed, gamma, out)``.
    """
    if cols % NORM_TAIL_VEC_LEN:
        raise ValueError(
            f"cols ({cols}) must be a multiple of {NORM_TAIL_VEC_LEN}; every "
            "stage's vector loop drops the remainder instead of failing"
        )
    if rows % (herd_x * rows_per_call):
        raise ValueError(
            f"rows ({rows}) must be divisible by herd_x*rows_per_call "
            f"({herd_x * rows_per_call}): the trip loop advances in "
            "rows_per_call-row bands per column"
        )
    if plane_major and rows * cols > SHIM_BD_STRIDE_MAX:
        raise ValueError(
            f"plane_major packing needs a plane stride of rows*cols "
            f"({rows}*{cols} = {rows * cols}), over the shim aie.dma_bd cap of "
            f"{SHIM_BD_STRIDE_MAX}. aircc reports this as \"'aie.dma_bd' op "
            f'Stride 2 exceeds the [1:{SHIM_BD_STRIDE_MAX}] range", against a '
            "tile, far from the cause. Use the default row-interleaved packing "
            "-- whose strides are at most 2*cols at any row count -- and note "
            "that it requires the PRODUCER to write interleaved, which is why "
            "plane_major exists at all: at rows*cols within the cap, plane 0 is "
            "a contiguous [rows, cols] block at offset 0, so an ordinary "
            "contiguous producer writes it with no change."
        )
    itemsize = np.dtype(np_dtype).itemsize
    need = _stage_l1_bytes(rows_per_call, cols, itemsize)
    if need > L1_BYTES:
        raise ValueError(
            f"rows_per_call={rows_per_call} at cols={cols} needs {need} bytes "
            f"of L1 in stage_add, over the {L1_BYTES}-byte budget; lower "
            "rows_per_call. aiecc would report this against the aie.tile, far "
            "from the cause."
        )
    rows_per_tile = rows // herd_x
    tile_elems = rows_per_call * cols

    xrt_dtype = type_mapper(np_dtype)
    l3_pk_ty = MemRefType.get(
        [2, rows, cols] if plane_major else [rows, 2, cols], xrt_dtype
    )
    l3_g_ty = MemRefType.get([cols], xrt_dtype)
    l3_out_ty = MemRefType.get([rows, cols], xrt_dtype)
    # The mirrored output, when the result must ALSO land in a plane of a
    # downstream packed buffer -- see MIRRORED OUTPUT in the module docstring.
    l3_out2_ty = MemRefType.get([2, rows, cols], xrt_dtype)

    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    # Flat: rows_per_call pairs of (x row, residual row), 2*cols apart. Flat
    # because a plane SUBVIEW of a multi-dim tile cannot reach a callee (see
    # the module docstring) and 1-D subviews are the form cores already lower.
    l1_pk_ty = MemRefType.get([2 * tile_elems], xrt_dtype, memory_space=l1_space)
    l1_flat_ty = MemRefType.get([tile_elems], xrt_dtype, memory_space=l1_space)
    l1_g_ty = MemRefType.get([cols], xrt_dtype, memory_space=l1_space)

    ln_func = FuncOp(
        NORM_SYMBOL,
        ([l1_flat_ty, l1_flat_ty, T.i32(), T.i32()], []),
        visibility="private",
    )
    ln_func.attributes["link_with"] = StringAttr.get(NORM_KERNEL_OBJ)
    ln_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    # first row of this tile's band = loop_iv + tx * rows_per_tile
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
    # Where row r's x vector j and its residual sit inside the flat L1 tile,
    # which mirrors whatever the DMA wrote:
    #   row-interleaved  x at r*2*cols + j,  residual one row-width further
    #   plane-major      x at r*cols + j,    residual a whole plane further
    _row_stride = cols if plane_major else 2 * cols
    _residual_gap = tile_elems if plane_major else cols
    pk_x_map = AffineMap.get(
        0,
        2,
        [
            AffineExpr.get_add(
                AffineExpr.get_mul(
                    AffineSymbolExpr.get(0), AffineConstantExpr.get(_row_stride)
                ),
                AffineSymbolExpr.get(1),
            )
        ],
    )
    pk_r_map = AffineMap.get(
        0,
        2,
        [
            AffineExpr.get_add(
                AffineExpr.get_add(
                    AffineExpr.get_mul(
                        AffineSymbolExpr.get(0), AffineConstantExpr.get(_row_stride)
                    ),
                    AffineSymbolExpr.get(1),
                ),
                AffineConstantExpr.get(_residual_gap),
            )
        ],
    )
    # flat element offset of (row r, vector j) inside a tile = r * cols + j
    flat_map = AffineMap.get(
        0,
        2,
        [
            AffineExpr.get_add(
                AffineExpr.get_mul(
                    AffineSymbolExpr.get(0), AffineConstantExpr.get(cols)
                ),
                AffineSymbolExpr.get(1),
            )
        ],
    )

    vec_ty = VectorType.get([NORM_TAIL_VEC_LEN], xrt_dtype)
    identity_map = AffineMapAttr.get(AffineMap.get_identity(1))

    Channel(CHANNEL_A2B, size=[herd_x, 1])
    Channel(CHANNEL_B2C, size=[herd_x, 1])

    _sig = [l3_pk_ty, l3_g_ty, l3_out_ty] + ([l3_out2_ty] if mirror_out else [])

    @FuncOp.from_py_func(*_sig)
    def norm_tail(*args):
        _ops = list(args)

        @launch(operands=_ops)
        def nt_launch(*l_args):
            _l = list(l_args)

            @segment(name="norm_tail_seg", operands=_l)
            def nt_seg(*s_args):
                s_pk, s_g, s_out = s_args[0], s_args[1], s_args[2]
                s_out2 = s_args[3] if mirror_out else None

                @herd(name="stage_add", sizes=[herd_x, 1], operands=[s_pk])
                def stage_add(_tx, _ty, _sx, _sy, h_pk):
                    c0 = ConstantOp.create_index(0)
                    l1_pk = AllocOp(l1_pk_ty, [], [])
                    l1_sum = AllocOp(l1_flat_ty, [], [])
                    cst0 = arith.ConstantOp(xrt_dtype, 0.0)

                    for loop_iv in range_(0, rows_per_tile, rows_per_call):
                        row = affine_apply(row_map, [loop_iv, _tx])
                        # Both operands in ONE DMA either way -- one L3-facing
                        # stream is what leaves room for gamma inside the
                        # column's two shim MM2S channels.
                        if plane_major:
                            # [2, rows, cols]: the band is one contiguous run
                            # per plane, planes rows*cols apart. That stride is
                            # the whole constraint; see the layout precondition.
                            dma_memcpy_nd(
                                l1_pk,
                                h_pk,
                                src_offsets=[0, row, 0],
                                src_sizes=[2, rows_per_call, cols],
                                src_strides=[rows * cols, cols, 1],
                            )
                        else:
                            # [rows, 2, cols]: a band of rows_per_call
                            # (x row, residual row) pairs is CONTIGUOUS, so
                            # every stride here is at most 2*cols -- inside the
                            # shim BD's 2^20 range at any row count.
                            dma_memcpy_nd(
                                l1_pk,
                                h_pk,
                                src_offsets=[row, 0, 0],
                                src_sizes=[rows_per_call, 2, cols],
                                src_strides=[2 * cols, cols, 1],
                            )
                        # x + residual: elementwise_add's stage body (bf16
                        # vector addf). Row r's x vector j is at
                        # [r*2*cols + j], its residual at [r*2*cols+cols+j];
                        # the sum lands CONTIGUOUS at [r*cols + j], the tile
                        # every later stage sees.
                        for r_iv in range_(0, rows_per_call):
                            for j_iv in range_(0, cols, NORM_TAIL_VEC_LEN):
                                x_off = affine_apply(pk_x_map, [r_iv, j_iv])
                                r_off = affine_apply(pk_r_map, [r_iv, j_iv])
                                s_off = affine_apply(flat_map, [r_iv, j_iv])
                                sub_x = subview(
                                    l1_pk.result, [x_off], [NORM_TAIL_VEC_LEN], [1]
                                )
                                sub_r = subview(
                                    l1_pk.result, [r_off], [NORM_TAIL_VEC_LEN], [1]
                                )
                                sub_out = subview(
                                    l1_sum.result, [s_off], [NORM_TAIL_VEC_LEN], [1]
                                )
                                v_x = transfer_read(
                                    vec_ty, sub_x, [c0], identity_map, cst0, [True]
                                )
                                v_r = transfer_read(
                                    vec_ty, sub_r, [c0], identity_map, cst0, [True]
                                )
                                v_sum = arith.addf(v_x, v_r)
                                transfer_write(
                                    None, v_sum, sub_out, [c0], identity_map, [True]
                                )
                                yield_([])
                            yield_([])
                        ChannelPut(CHANNEL_A2B, l1_sum, indices=[_tx, c0])
                        yield_([])

                    DeallocOp(l1_pk)
                    DeallocOp(l1_sum)

                @herd(name="stage_norm", sizes=[herd_x, 1])
                def stage_norm(_tx, _ty, _sx, _sy):
                    c0 = ConstantOp.create_index(0)
                    l1_in = AllocOp(l1_flat_ty, [], [])
                    l1_norm = AllocOp(l1_flat_ty, [], [])
                    cols_i32 = ConstantOp(T.i32(), cols)
                    nrows_i32 = ConstantOp(T.i32(), rows_per_call)

                    for _ in range_(0, rows_per_tile, rows_per_call):
                        ChannelGet(CHANNEL_A2B, l1_in, indices=[_tx, c0])
                        CallOp(ln_func, [l1_in, l1_norm, cols_i32, nrows_i32])
                        ChannelPut(CHANNEL_B2C, l1_norm, indices=[_tx, c0])
                        yield_([])

                    DeallocOp(l1_in)
                    DeallocOp(l1_norm)

                stage_norm.attributes["link_with"] = StringAttr.get(NORM_KERNEL_OBJ)

                _scale_ops = [s_g, s_out] + ([s_out2] if mirror_out else [])

                @herd(name="stage_scale", sizes=[herd_x, 1], operands=_scale_ops)
                def stage_scale(_tx, _ty, _sx, _sy, *h_args):
                    h_g, h_out = h_args[0], h_args[1]
                    h_out2 = h_args[2] if mirror_out else None
                    c0 = ConstantOp.create_index(0)
                    l1_in = AllocOp(l1_flat_ty, [], [])
                    l1_g = AllocOp(l1_g_ty, [], [])
                    l1_out = AllocOp(l1_flat_ty, [], [])
                    cst0 = arith.ConstantOp(xrt_dtype, 0.0)

                    # gamma: ONE L3 fetch per tile, before the trip loop. This
                    # is the column's second (and last) shim MM2S stream.
                    dma_memcpy_nd(
                        l1_g,
                        h_g,
                        src_offsets=[0],
                        src_sizes=[cols],
                        src_strides=[1],
                    )

                    for loop_iv in range_(0, rows_per_tile, rows_per_call):
                        row = affine_apply(row_map, [loop_iv, _tx])
                        ChannelGet(CHANNEL_B2C, l1_in, indices=[_tx, c0])
                        # normed * gamma, gamma broadcast over the tile's rows:
                        # elementwise_mul's stage body (bf16 vector mulf -- the
                        # AIE vector unit does not legalize f32 multiplies).
                        for r_iv in range_(0, rows_per_call):
                            for j_iv in range_(0, cols, NORM_TAIL_VEC_LEN):
                                flat = affine_apply(flat_map, [r_iv, j_iv])
                                sub_in = subview(
                                    l1_in.result, [flat], [NORM_TAIL_VEC_LEN], [1]
                                )
                                sub_g = subview(
                                    l1_g.result, [j_iv], [NORM_TAIL_VEC_LEN], [1]
                                )
                                sub_out = subview(
                                    l1_out.result, [flat], [NORM_TAIL_VEC_LEN], [1]
                                )
                                v_in = transfer_read(
                                    vec_ty, sub_in, [c0], identity_map, cst0, [True]
                                )
                                v_g = transfer_read(
                                    vec_ty, sub_g, [c0], identity_map, cst0, [True]
                                )
                                v_out = arith.mulf(v_in, v_g)
                                transfer_write(
                                    None, v_out, sub_out, [c0], identity_map, [True]
                                )
                                yield_([])
                            yield_([])
                        dma_memcpy_nd(
                            h_out,
                            l1_out,
                            dst_offsets=[row, 0],
                            dst_sizes=[rows_per_call, cols],
                            dst_strides=[cols, 1],
                        )
                        if mirror_out:
                            # Same band, into plane 1 of the downstream packed
                            # buffer. A second S2MM stream per column; the
                            # inbound budget this file protects is MM2S and is
                            # untouched.
                            dma_memcpy_nd(
                                h_out2,
                                l1_out,
                                dst_offsets=[1, row, 0],
                                dst_sizes=[1, rows_per_call, cols],
                                dst_strides=[rows * cols, cols, 1],
                            )
                        yield_([])

                    DeallocOp(l1_in)
                    DeallocOp(l1_g)
                    DeallocOp(l1_out)


def norm_tail_device_inputs(x, residual, plane_major=False):
    """Pack x and residual for the builder's layout. MUST match the module's.

    Default is per-row pairs, ``np.stack(axis=1) -> [rows, 2, cols]``: row r's x
    then row r's residual, back to back -- the layout whose band fetch stays
    inside the shim BD stride range at ANY row count. Keeping the packing beside
    the builder is what stops a caller getting the layout right today and wrong
    next time; the two stacks look interchangeable and are not.

    ``plane_major=True`` gives ``np.stack(axis=0) -> [2, rows, cols]``, which
    only compiles while ``rows*cols`` is inside the shim BD stride cap. Its
    reason to exist is not the packing: plane 0 is then a contiguous
    ``[rows, cols]`` block at offset 0, so a device producer that writes an
    ordinary contiguous tensor writes plane 0 with NO change to it, and the host
    fills plane 1. That is what lets this pipeline consume a GEMM's output
    without teaching the shared GEMM builder to write interleaved.
    """
    if x.shape != residual.shape:
        raise ValueError(f"x {x.shape} and residual {residual.shape} must match")
    axis = 0 if plane_major else 1
    return [np.ascontiguousarray(np.stack([x, residual], axis=axis))]


def norm_tail_reference(x, residual, gamma):
    """FP32 reference: ``LayerNorm(x + residual) * gamma``, rounded once.

    The pipeline computes the PRE-ADD weighted norm -- the same function
    ``build_addnorm_module(pre_add=True)`` computes in one kernel -- so its
    oracle is that operator's, imported rather than re-derived: two references
    for one function is two chances for them to drift apart.
    """
    return addnorm_pre_add_reference(x, residual, gamma, eps=EPS)
