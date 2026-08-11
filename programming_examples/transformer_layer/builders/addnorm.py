# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AddNorm: weighted LayerNorm and a residual, in either order.

CONTRACT
    ``build_addnorm_module(rows, cols, ...)`` returns an ``air.ir.Module``
    whose single function takes ``x[rows, cols]``, ``residual[rows, cols]``,
    ``weight[cols]`` and ``out[rows, cols]``, all bf16, and computes per row

        pre_add=False (default)   out[r, :] = LayerNorm(x[r, :]) * weight
                                             + residual[r, :]
        pre_add=True              out[r, :] = LayerNorm(x[r, :] + residual[r, :])
                                             * weight

    The signature is identical either way; only the arithmetic differs. Each
    form has its OWN reference below -- ``addnorm_reference`` and
    ``addnorm_pre_add_reference`` -- rather than one function with an ordering
    flag, because a branch inside a reference is exactly as easy to read
    backwards as the kernel it is supposed to check.

    Call ``compile_addnorm_kernel(pre_add=...)`` before building: the two forms
    link DIFFERENT objects (see below), and getting that wrong does not fail to
    link, it computes the other function.

WHICH FORM A TRANSFORMER BLOCK WANTS
    ``encoder_bert`` is post-norm, and both of its normalization points are
    ``LayerNorm(attn_out + input) * weight`` -- the PRE-ADD form. The post-add
    form is what iron's ``addnorm`` operator carried and what Phase C validated;
    the two are different functions and a block wired to the wrong one produces
    a plausible activation that is wrong by one residual add.

WHY pre_add IS NOT ONE OBJECT WITH A FLAG
    The post-add form comes from ``encoder.cc`` (``encoder.o``), which has no
    pre-add path at all. The pre-add form lives in a different translation unit,
    ``addnorm_ffn.cc``, behind ``-DADDNORM_PRE_ADD``, and exports its one-output
    entry point as ``fused_add_layer_norm_1outs``. So selecting the variant
    selects the object AND the symbol AND the L1 budget (one output buffer
    rather than two), which is why it is a builder keyword rather than a flag
    flip. ``compile_kernels.py::check_pre_add_variants_differ`` asserts at build
    time that ``-DADDNORM_PRE_ADD`` still reaches the templates.

    The pre-add object built here is ADDNORM-ONLY and named
    ``addnorm_pre_add.o``, distinct from the full 11-symbol
    ``addnorm_ffn_pre_add.o`` the compile gate checks. Linking the full object
    would drag every FFN matmul microkernel into a core that never calls one.

WHY THE ONE-OUTPUT ENTRY POINT IS ENOUGH FOR PRE-ADD
    ``fused_add_layer_norm_2outs`` under ``-DADDNORM_PRE_ADD`` additionally
    exports the raw ``x + residual`` sum through ``output2``, for a block whose
    next residual stream is that sum. ``encoder_bert``'s is not: its second
    residual is ``hidden``, the NORMALIZED output of the first norm. So the
    one-output form carries everything the block needs, and it costs one fewer
    L1 tile buffer than the post-add path.

THE WEIGHT IS A RUNTIME ARGUMENT
    iron's ``addnorm`` bakes its weights into the MLIR through ``np.load()`` at
    generation time and hashes them into the artifact name, so changing a
    weight forces a full recompile. That is not reproduced here: ``weight`` is
    a plain memref argument, uploaded like any other buffer, and one compiled
    ELF serves every weight vector of that shape.

TWO OUTPUTS, ONE DRAINED (POST-ADD ONLY)
    ``encoder.o``'s entry point writes its result to two buffers because the
    fused encoder block feeds it to both the FFN and the next residual. Only
    ``output1`` is drained to L3 here: a second L3 output would need a second
    shim S2MM channel per column for a byte-identical copy. ``output2`` stays in
    L1 and is discarded. The pre-add path calls the one-output entry point
    instead and allocates no ``output2`` at all.

MULTI-TRIP IS SAFE ONLY ON A ONE-COLUMN HERD -- MEASURED, PHASE J1
    This builder once raised unless ``rows == herd_x * rows_per_call``, because
    two or more trips of the herd's row loop MISCOMPILED: measured on NPU2 at
    ``[8, 64]``, ``herd_x=1``, one trip was exact and two trips were garbage
    (491 of 512 elements out). The guard's docstring blamed the three L3->L1
    streams per tile (x, residual, weight) against a column's two shim MM2S
    channels. That was the SYMPTOM. The cause was the shim feed order under
    packet multiplexing: ``air-dma-to-channel`` hoists each L3-side DMA into
    its own launch-scope put loop, the packet-multiplexed shim MM2S queue
    serializes those loops whole channel after whole channel, and the consuming
    tile's BD ring is built from the herd's per-iteration get order, so it
    expects the streams interleaved. At one trip the two orders coincide; at
    two or more, every packet after the first iteration lands in the wrong
    buffer. Ping-pong was never involved: ``--omit-ping-pong-transform=all``
    reproduced the identical corruption, and this loop is never labeled anyway
    (its callee is unannotated, so the rotation proof declines it).

    ``air-fuse-packet-put-loops`` (commit ``bfb647d9``) fixes exactly the
    ``herd_x=1`` form: it fuses SIBLING ``scf.for`` put loops that share a
    block into one loop doing the puts in program order, and the driver's
    fixture (``agents/scripts/port-loop/fixtures/addnorm_multitrip.py``) pins
    that two-trip loop on hardware at zero mismatches. Phase J1 measured the
    same at ``cols=768``, ``herd_x=1``, ``rows_per_call=8``.

    IT DOES NOT GENERALIZE, and the guard below encodes what J1 measured:
    - ``herd_x >= 2`` STILL MISCOMPILES at any trip count above one. On a
      multi-column herd, ``air-dma-to-channel`` wraps each per-tile put loop
      in ``scf.parallel`` (and the broadcast weight put loop sits apart from
      them), so the fusion pass's sibling-``scf.for`` pattern never matches --
      its output IR is byte-identical to its input -- and the feed-order bug
      above is back. Measured: 2 trips, ``cols=64``, ``herd_x=8``,
      ``rows_per_call=4`` -> 4070/4096 mismatches (4039/4096 with the weight
      DMA hoisted, so hoisting is not the fix either); the J1 target shape,
      64 trips at ``[4096, 768]``, ``herd_x=8`` -> compiles and returns
      3,130,958/3,145,728 mismatches.
    - ``herd_x == 1`` is numerically exact but BD-CAPPED: each fused put
      lowers to its own simultaneously-active ``aiex.dma_configure_task``, and
      a shim tile has 16 BDs. Measured: 2 trips at ``cols=768`` exact; 8 trips
      fail to COMPILE ("Too many simultaneously active buffer descriptors on
      tile (0,0)"). The failure is loud, so multi-trip at ``herd_x=1`` is
      permitted -- it either works or refuses to build.
    - Staging the weight through L2 to free the shim (the route this
      docstring used to recommend) is ALSO blocked today: at ``herd_x=8``
      placement fails ("no ShimNOCTile has sufficient DMA capacity" for the
      weight put -- x and residual already fill every column's two MM2S
      channels), and at ``herd_x=4``, with the weight broadcast from L2 or
      replicated per column, ``air-to-aie`` emits conflicting stream-switch
      routes ("'aie.connect' op ... targets same dst as another connect op"
      on the first core tile).

    So the practical row cap STANDS: one kernel call per tile on the herd
    widths the block needs, ``rows <= herd_x * (L1 budget)``, and the layer's
    normalization points stay row-blocked (``builders/block.py``). Lifting it
    for real needs compiler work -- packet put-loop fusion through
    ``scf.parallel``, or loop-shaped packet BD programs -- not a builder
    change; every builder-side route was measured above.

    Two constraints hold in every form and are checked below:
    - ``rows_per_call`` must DIVIDE ``rows // herd_x``. The loop advances in
      ``rows_per_call``-row bands, so a non-divisor would run the final trip
      past the tile's band rather than failing.
    - The L1 budget caps ``rows_per_call`` (13 at ``cols=768`` pre-add);
      ``addnorm_max_rows`` exposes the arithmetic.

FOOTGUNS
    - Statistics are f32 and the variance is TWO-PASS (mean first, then
      ``E[(x - mean)^2]``) in BOTH kernel variants, following
      ``layer_norm/layer_norm.cc``'s discipline -- doc 23 item 2's addnorm
      half. The one-pass ``E[x^2] - E[x]^2`` form both shipped with (bf16 row
      sum) loses an offset row's variance ENTIRELY: measured collapse between
      ``|mean|/sigma`` 2 and 4 (``probe_addnorm_variance_cliff.py`` under
      ``agents/probes/``), where the variance clamps at zero and the row
      normalizes by ``1/sqrt(eps)`` ~ 316. The ``*_offset``
      opcheck rows pin the regime. The kernels still clamp a negative
      variance before ``aie::invsqrt`` as a NaN guard, though it is
      non-negative by construction now.
    - ``cols`` must be a multiple of 32 (the kernel's ``N``). A non-multiple is
      silently truncated by ``vector_chunks = cols / N``, not diagnosed.
    - Both objects are compiled with ONLY the addnorm half
      (``build_ffn=False``). The FFN half also defines ``ffn_gelu_bf16`` and
      ``ffn_eltwise_add_bf16_vector``, which collide across the two sources;
      building the halves separately is what keeps one kernel object per ELF.
    - The object must exist in the working directory when aiecc links.
      ``compile_addnorm_kernel`` puts it there; ``opcheck.py`` calls it.
    - L1 is 64 KiB per core, and the tile buffers (x, residual, output, and
      ``output2`` in the post-add form) plus the weight and the 1 KiB stack have
      to fit. aiecc reports an overflow against the ``aie.tile``, not against
      the builder, so ``_l1_bytes`` below checks it up front instead.
      ``addnorm_max_rows`` exposes the same arithmetic so a caller can derive a
      legal row count rather than guessing one. It counts ALLOCATIONS, not
      aircc's ping-pong copies of the DMA-fed buffers, so it is an upper bound
      and not a target: a shape at the cap may still fail to place.
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

# encoder.cc's addnorm half, two outputs, residual added AFTER the norm.
POST_ADD_KERNEL_OBJ = "encoder.o"
POST_ADD_SYMBOL = "fused_add_layer_norm_2outs"

# addnorm_ffn.cc's addnorm half built with -DADDNORM_PRE_ADD, one output,
# residual folded in BEFORE the statistics. Deliberately not the compile gate's
# `addnorm_ffn_pre_add.o`, which is the full 11-symbol build; see the module
# docstring.
PRE_ADD_KERNEL_OBJ = "addnorm_pre_add.o"
PRE_ADD_SYMBOL = "fused_add_layer_norm_1outs"

# AIE2P core-local memory, and the stack aircc reserves inside it.
L1_BYTES = 64 * 1024
L1_STACK_BYTES = 1024


def _tile_buffers(pre_add):
    """L1 activation tiles the selected entry point needs.

    Post-add allocates x, residual, output1 and output2 -- the kernel writes
    both outputs whether or not the second is drained. Pre-add's one-output
    entry point needs no output2.
    """
    return 3 if pre_add else 4


def _l1_bytes(rows_per_call, cols, itemsize, pre_add=False):
    """L1 a single tile needs: its activation tiles plus the weight vector."""
    return (
        _tile_buffers(pre_add) * rows_per_call * cols * itemsize
        + cols * itemsize
        + L1_STACK_BYTES
    )


def addnorm_max_rows(cols, np_dtype=bfloat16, herd_x=8, pre_add=False):
    """Largest one-trip ``rows`` at ``cols``: ``herd_x`` times the largest
    ``rows_per_call`` the L1 budget fits.

    On a multi-column herd -- where one trip is the only legal form, see the
    module docstring -- this IS the row cap, and it is what ``block.norm_rows``
    derives the layer's band size from. At ``herd_x == 1`` it bounds
    ``rows_per_call`` instead. Provided so either number is derived for a new
    width instead of carried over from one that happened to fit at a narrower
    one.

    UPPER BOUND, NOT A TARGET. This counts allocations only; aircc ping-pongs
    the DMA-fed buffers, so a shape sitting at the cap can still fail to place.
    """
    itemsize = np.dtype(np_dtype).itemsize
    budget = L1_BYTES - L1_STACK_BYTES - cols * itemsize
    rows_per_call = budget // (_tile_buffers(pre_add) * cols * itemsize)
    return herd_x * rows_per_call


def addnorm_kernel_object(pre_add=False):
    """Name of the object the selected variant links against."""
    return PRE_ADD_KERNEL_OBJ if pre_add else POST_ADD_KERNEL_OBJ


def compile_addnorm_kernel(pre_add=False):
    """Build the selected variant's object into the CWD, where aiecc looks.

    Both are addnorm-half-only builds. Which SOURCE is compiled is part of the
    variant, not just which flag: ``encoder.cc`` has no pre-add path.
    """
    import shared.infra.external_kernels as ek

    if pre_add:
        ek.compile_addnorm_ffn(
            build_ffn=False,
            build_addnorm=True,
            pre_add=True,
            out_name=PRE_ADD_KERNEL_OBJ,
        )
    else:
        ek.compile_encoder(
            build_ffn=False, build_addnorm=True, out_name=POST_ADD_KERNEL_OBJ
        )


@module_builder
def build_addnorm_module(
    rows, cols, np_dtype=bfloat16, herd_x=8, rows_per_call=None, pre_add=False
):
    """Build weighted LayerNorm and a residual over an ``herd_x x 1`` herd.

    Args:
        rows, cols: L3 activation shape. ``cols`` must be a multiple of
            ``ADDNORM_VEC_LEN``; ``rows`` must be divisible by ``herd_x``, and
            ``rows // herd_x`` by ``rows_per_call`` -- see the module
            docstring's multi-trip section for both constraints.
        np_dtype: element type; bf16, matching the kernel's C linkage.
        herd_x: AIE columns. Each tile drives three L3->L1 streams (x,
            residual, weight) and one L1->L3 stream.
        rows_per_call: rows resident in L1 per kernel invocation; the herd's
            row loop runs ``rows // herd_x // rows_per_call`` trips. Defaults
            to ``rows // herd_x`` -- one trip, which is the ONLY legal value
            when ``herd_x > 1`` (see the module docstring: multi-trip
            miscompiles on multi-column herds). At ``herd_x == 1`` the L1
            budget caps it (``addnorm_max_rows(cols, ...) // herd_x``, 13 at
            ``cols=768`` pre-add) and the shim BD cap limits the trip count
            loudly at compile time.
        pre_add: select ``LayerNorm(x + residual) * weight`` instead of the
            default ``LayerNorm(x) * weight + residual``. Different object,
            different symbol, different L1 budget -- see the module docstring,
            and check against ``addnorm_pre_add_reference``, not
            ``addnorm_reference``.

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
    if rows_per_tile % rows_per_call:
        raise ValueError(
            f"rows_per_call ({rows_per_call}) must divide rows // herd_x "
            f"({rows_per_tile}): the row loop advances in rows_per_call-row "
            "bands, and a non-divisor would run the final trip past the "
            "tile's band instead of failing."
        )
    if rows_per_call != rows_per_tile and herd_x > 1:
        raise ValueError(
            f"rows_per_call must be rows // herd_x ({rows_per_tile}) on a "
            f"{herd_x}-column herd, got {rows_per_call}. More than one trip "
            "of the row loop MISCOMPILES on multi-column herds: "
            "air-fuse-packet-put-loops cannot fuse the scf.parallel-wrapped "
            "per-tile put loops, so the packet-multiplexed shim feed order is "
            "wrong from the second iteration on (measured J1: 4070/4096 "
            "mismatches at 2 trips, cols=64, herd_x=8). See the module "
            "docstring; the failure looks numerical, not structural. "
            "Multi-trip is permitted only at herd_x=1, where the fusion "
            "fires and any BD overflow refuses to compile."
        )

    need = _l1_bytes(rows_per_call, cols, np.dtype(np_dtype).itemsize, pre_add)
    if need > L1_BYTES:
        raise ValueError(
            f"rows_per_call={rows_per_call} at cols={cols} needs {need} bytes "
            f"of L1 per tile, over the {L1_BYTES}-byte budget. Lower "
            f"rows_per_call -- at most "
            f"{addnorm_max_rows(cols, np_dtype, herd_x, pre_add) // herd_x} at "
            "this width; aiecc would otherwise report this against the "
            "aie.tile, far from the cause."
        )

    kernel_obj = addnorm_kernel_object(pre_add)
    xrt_dtype = type_mapper(np_dtype)
    l3_act_ty = MemRefType.get([rows, cols], xrt_dtype)
    l3_w_ty = MemRefType.get([cols], xrt_dtype)

    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_tile_ty = MemRefType.get([rows_per_call, cols], xrt_dtype, memory_space=l1_space)
    l1_w_ty = MemRefType.get([cols], xrt_dtype, memory_space=l1_space)

    # The two entry points differ by one memref: post-add takes output1 AND
    # output2, pre-add takes one output. Everything else about the signature,
    # including the trailing (cols, rows) pair, is shared.
    kernel_symbol = PRE_ADD_SYMBOL if pre_add else POST_ADD_SYMBOL
    out_tile_tys = [l1_tile_ty] if pre_add else [l1_tile_ty, l1_tile_ty]
    addnorm_func = FuncOp(
        kernel_symbol,
        (
            [l1_tile_ty, l1_tile_ty, l1_w_ty] + out_tile_tys + [T.i32(), T.i32()],
            [],
        ),
        visibility="private",
    )
    addnorm_func.attributes["link_with"] = StringAttr.get(kernel_obj)
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
                    # Post-add only. The kernel writes output2 unconditionally
                    # in that form, so the buffer has to exist even though
                    # nothing drains it; the pre-add entry point has no such
                    # argument. See _tile_buffers, which sizes L1 to match.
                    l1_out2 = None if pre_add else AllocOp(l1_tile_ty, [], [])
                    cols_i32 = ConstantOp(T.i32(), cols)
                    nrows_i32 = ConstantOp(T.i32(), rows_per_call)

                    for loop_iv in range_(0, rows_per_tile, rows_per_call):
                        row = affine_apply(row_map, [loop_iv, _tx])
                        # Re-fetched every iteration even though the weight
                        # vector never changes. Hoisting it out of the loop
                        # was long documented here as a silent corruption;
                        # that was re-measured during Phase H and is FALSE
                        # since air-fuse-packet-put-loops: the driver fixture's
                        # `hoisted` variant is exact, because this loop's
                        # callee is unannotated so the ping-pong rotation
                        # declines it either way. The refill stays because it
                        # is the exact configuration the fixture's `inside`
                        # variant pins on hardware, and 1.5 KiB per trip is
                        # noise next to the row DMAs.
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
                        out_operands = [l1_out1] if pre_add else [l1_out1, l1_out2]
                        CallOp(
                            addnorm_func,
                            [l1_x, l1_res, l1_w] + out_operands + [cols_i32, nrows_i32],
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
                    if l1_out2 is not None:
                        DeallocOp(l1_out2)

                addnorm_body.attributes["link_with"] = StringAttr.get(kernel_obj)


def _layer_norm_f32(x_f32, eps):
    """Two-pass LayerNorm of an f32 array's rows. No weight, no residual.

    Shared by the two references below so that the ONLY difference between them
    is which tensor is normalized and where the residual lands -- the ordering
    is then visible at the call site rather than buried in a branch.
    """
    mean = x_f32.mean(axis=-1, keepdims=True)
    var = ((x_f32 - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x_f32 - mean) / np.sqrt(var + eps)


def addnorm_reference(x, residual, weight, eps=EPS):
    """FP32 reference for POST-ADD: ``LayerNorm(x) * weight + residual``.

    Normalize, THEN add. The statistics never see ``residual``. This is the
    oracle for ``build_addnorm_module(..., pre_add=False)`` and for nothing
    else -- see ``addnorm_pre_add_reference`` for the other ordering.

    Two-pass variance in f32, one rounding to bf16 at the end. The kernel now
    keeps the same two-pass f32 statistics, but this oracle still reproduces
    none of its bf16 intermediate roundings (the squared deviations, the
    normalized value, the weight multiply, the trailing add) -- the point of
    the check is to measure that gap, not to reproduce it.
    """
    normed = _layer_norm_f32(x.astype(np.float32), eps)
    out = normed * weight.astype(np.float32) + residual.astype(np.float32)
    return out.astype(x.dtype)


def addnorm_pre_add_reference(x, residual, weight, eps=EPS):
    """FP32 reference for PRE-ADD: ``LayerNorm(x + residual) * weight``.

    Add, THEN normalize. The statistics are those of the SUM, and the residual
    is not added again afterwards. This is the oracle for
    ``build_addnorm_module(..., pre_add=True)``, which is the form
    ``encoder_bert`` uses at both of its normalization points.

    Same f32 discipline as the post-add reference, and one extra gap it does
    not reproduce: the kernel forms ``x + residual`` in bf16 before taking the
    statistics, and this sums in f32. Rounding the sum here would make the
    check agree with the device for the wrong reason.
    """
    summed = x.astype(np.float32) + residual.astype(np.float32)
    out = _layer_norm_f32(summed, eps) * weight.astype(np.float32)
    return out.astype(x.dtype)
