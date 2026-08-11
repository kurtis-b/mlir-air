# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The resident FFN interior (doc 31, increment R1): up @ GeLU @ down in ONE segment.

CONTRACT
    ``build_ffn_resident_module(seq_len, ffn_dim, emb_dim, ...)`` returns an
    ``air.ir.Module`` with one function ``ffn_resident(hidden, w_up_packed,
    w_down_packed, y)`` computing, for one ``TILE_M``-row band,

        y = gelu(hidden @ w_up) @ w_down          all operands bf16

    with the ``[seq_len, ffn_dim]`` interior (``ffn_up`` and ``ffn_gelu`` in
    the packaged tail's vocabulary) never materialized anywhere: not in DRAM,
    not whole in L2 -- it exists only as ``[TILE_M, tile_k]`` chunks in flight
    between three herds of one segment. DRAM is crossed by exactly ``hidden``
    in, ``w_up``/``w_down`` in, and ``y`` out -- the crossings doc 31a's floor
    says a resident interior must keep, and no other.

    ``hidden`` arrives ROW-MAJOR -- the layout a norm tail emits -- NOT
    pre-tiled like ``ffn_accum``'s A. The blocked-microtile retile the GEMM
    kernels need happens on the shim's L3 READ pattern (the same 4-D idiom
    J7b's C fetch already rides), so the layout seam between a row-major
    producer and a microtiled consumer costs no core cycles and no memtile
    pattern. That is R1's settlement of doc 31 seam 1 *for an L3-resident
    producer*; the R2 (on-chip norm tail) case is measured separately.

THE DATAFLOW (doc 31 seam 3), and why each hop is where it is

        shim ->(A: hidden k'-slice, retiled; B: w_up group slice)-> memtile
        memtile ->[ffn_res_up_feed]-> UP herd      (herd_x cores, one
            [TILE_M, group_n] H group each per sweep, C L1-resident across
            the k' loop -- J7b's in-place kernel + zero, WITHOUT the hoist:
            with the output on-chip there is no C DMA to hoist)
        UP ->[ffn_res_h]-> GELU herd               (core->core, L1->L1: the
            group leaves in [TILE_M, tile_k] chunks in EXACTLY the order the
            down herd's K loop consumes them)
        GELU ->[ffn_res_g]-> memtile               (one chunk staged whole,
            never read at an offset)
        memtile ->[ffn_res_down_feed]-> DOWN herd  (A|B per K step, J7b's
            feed verbatim; the C accumulator pair on ``y`` rides the shim
            and air-hoist-dma-in-accum-pattern lifts it, unchanged)

    - THE UP HERD REUSES THE DOWN HERD'S KERNEL OBJECT, and that is a
      constraint, not a convenience. ``ffn_accum_mm.o`` bakes DIM_M/K/N in as
      -D flags and exports fixed symbol names; a second tile shape would be a
      second object exporting the SAME ``ffn_matmul_bf16_bf16_up_proj``, and
      two private FuncOps cannot share a symbol in one module. So the up
      stage's output group width IS the down stage's tile_n
      (``emb_dim // herd_x``), its K step IS ``tile_k``, and both herds call
      the one 64x32x192 object. This is what fixes ``group_n`` and makes
      ``ffn_dim % (herd_x * group_n) == 0`` a precondition.
    - THE HAND-OFF FANS THROUGH A MEMTILE BY PORT ARITHMETIC, not by choice.
      Every down core consumes EVERY H chunk (each owns a full-K column strip
      of y), so the chunk stream must be broadcast herd_x ways -- but a down
      core has two S2MM channels, both spoken for (A|B feed + hoisted C
      fetch), and an air channel has one physical source. The only node that
      can serialize herd_x producers into herd_x identical consumer feeds is
      a memtile. The up->GeLU edge stays core->core L1->L1; the GeLU->down
      edge is L1->L2->L1 with DRAM untouched.
    - CHUNK ORDER IS THE ORDER SEAM'S CURRENCY. Group ``g = s * herd_x + c``
      (sweep ``s``, up/GeLU column ``c``) spans down-K steps
      ``g * chunks_per_group + jj``. The down feed drains GeLU column c's
      sub-channel for ``chunks_per_group`` chunks before moving to c+1, so
      chunks arrive in strictly increasing K order with every channel index a
      compile-time constant -- no IV-indexed sub-channel anywhere (H5).

L3-SIDE OFFSETS ONLY (the J7b frozen-BD rule, obeyed three ways)
    - ``hidden`` is re-read from L3 once per sweep (``sweeps`` times per
      band) instead of being staged whole in L2 and read at a per-k' offset
      -- the staged form is the miscompile
      (``agents/probes/probe_ffn_accum_bd_offset.py``). The re-read is real
      DMA traffic (sweeps x 96 KiB against 96 KiB) and it is the price of
      correctness under static BDs; doc 31a's crossing floor counts logical
      crossings and is untouched.
    - Both weight refills advance on launch-argument offsets computed from
      loop IVs -- the runtime sequence materializes those per task.
    - Every L2/L1-side offset in the design is a compile-time literal: the
      up core's six chunk puts are Python-unrolled with literal offsets, the
      staged chunk and B slices are put whole or at ``tx``-literal offsets.

NO PLACEMENT, NO BUFFER DEPTHS, NO HAND RING
    No tile coordinate, no channel depth, no memory-space choice for the
    down accumulator: ``air-place-herds`` places the three herds and the
    hoist pass forms the down ring exactly as in ``builders/ffn_accum.py``.

FOOTGUNS
    - ``seq_len`` must equal ``TILE_M`` (one 64-row band). Doc 31 makes every
      increment band-serial: the band loop advances on LAUNCH ARGUMENTS
      (L3-side), one dispatch per band, and never on a staged tail of a
      previous band. A multi-band module would need herd_y > 1 through the
      same memtile fan-out, which is unmeasured -- refuse, do not guess.
    - ``compile_ffn_resident_kernels()`` must have run in the CWD before
      aiecc links. TWO objects: ``ffn_accum_mm.o`` at THIS module's tiles
      (both GEMM herds) and ``encoder_ffn.o`` (GeLU herd). They both define
      ``ffn_gelu_bf16``; they may coexist here ONLY because no core links
      both -- do not merge the herds' link_with.
    - ``ffn_gelu_bf16``'s operands are ``__restrict``: the GeLU stage MUST
      keep separate in/out tiles. An in-place call is UB, not a wrong answer
      you can measure.
    - The up C zero is UNGUARDED, before the k' loop, once per group; the
      down C zero is the J7b guarded first-K-step call. Swapping those
      disciplines ships y_old into the result (down) or re-zeroes a partial
      sum (up).
    - The A and B gets on each feed are strictly A then B, every step, both
      feeds. A swapped get order is a hang, not a numerical bug (the
      transfers have different sizes).
    - ``y`` need not enter zeroed (guarded down zero), but it IS read by the
      hoisted pre-loop fetch: nothing may alias it.
    - The feed loops are emitted AFTER the herds. The structure check
      brace-matches herds by name (this module has three; "first scf.for of
      the dump" would find the up herd's sweep loop, not the down K loop).
"""

import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects.arith import CmpIOp, CmpIPredicate, ConstantOp
from air.dialects.func import CallOp, FuncOp
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.scf import IfOp, for_, yield_
from air.backend.xrt_runner import type_mapper

from builders.ffn import ffn_reference
from builders.ffn_accum import (
    FFN_ACCUM_HERD_X,
    FFN_ACCUM_KERNEL_OBJ,
    FFN_ACCUM_MM_SYMBOL,
    FFN_ACCUM_TILE_K,
    FFN_ACCUM_ZERO_SYMBOL,
    MAX_L1_TILE_K,
    MAX_PLACEABLE_HERD_X,
    MICRO,
    TILE_M,
    compile_ffn_accum_kernel,
    ffn_accum_pack_w,
)
from builders.gelu import GELU_SYMBOL, KERNEL_OBJ as GELU_KERNEL_OBJ

range_ = for_

# The four channels, all sized [herd_x, 1]: sub-channel i pairs column i of
# one stage with column i of the next (or with the memtile relay).
CHANNEL_UP_FEED = "ffn_res_up_feed"
CHANNEL_H = "ffn_res_h"
CHANNEL_G = "ffn_res_g"
CHANNEL_DOWN_FEED = "ffn_res_down_feed"

# A memtile's memory, the ceiling on the staged slices (all of which hold ONE
# step; nothing here stages a whole operand -- see the module docstring).
L2_BYTES = 512 * 1024

# AIE2P core-local memory and the stack aircc reserves inside it, as
# builders/norm_tail.py records them.
L1_BYTES = 64 * 1024
L1_STACK_BYTES = 1024


def compile_ffn_resident_kernels(tile_k=FFN_ACCUM_TILE_K, tile_n=None, emb_dim=768):
    """Build both kernel objects into the CWD, where aiecc's link_with looks.

    ``ffn_accum_mm.o`` (both GEMM herds -- the -DDIM_* set must match the
    module's ``tile_k`` and ``emb_dim // herd_x``, see the object-identity
    footgun) and ``encoder_ffn.o`` (the GeLU herd; encoder.cc's FFN half
    only, exactly as ``builders/gelu.py`` links it).
    """
    import shared.infra.external_kernels as ek

    if tile_n is None:
        tile_n = emb_dim // FFN_ACCUM_HERD_X
    compile_ffn_accum_kernel(tile_k=tile_k, tile_n=tile_n)
    ek.compile_encoder(build_ffn=True, build_addnorm=False, out_name=GELU_KERNEL_OBJ)


def _mul_add_map(factor, offset=0):
    """AffineMap for ``s0 * factor + offset`` (lazily built; needs a context)."""
    return AffineMap.get(
        0,
        1,
        [
            AffineExpr.get_add(
                AffineExpr.get_mul(
                    AffineSymbolExpr.get(0), AffineConstantExpr.get(factor)
                ),
                AffineConstantExpr.get(offset),
            )
        ],
    )


def _two_iv_map(factor0, factor1, offset=0):
    """AffineMap for ``s0 * factor0 + s1 * factor1 + offset``."""
    return AffineMap.get(
        0,
        2,
        [
            AffineExpr.get_add(
                AffineExpr.get_add(
                    AffineExpr.get_mul(
                        AffineSymbolExpr.get(0), AffineConstantExpr.get(factor0)
                    ),
                    AffineExpr.get_mul(
                        AffineSymbolExpr.get(1), AffineConstantExpr.get(factor1)
                    ),
                ),
                AffineConstantExpr.get(offset),
            )
        ],
    )


@module_builder
def build_ffn_resident_module(
    seq_len=TILE_M,
    ffn_dim=3072,
    emb_dim=768,
    np_dtype=bfloat16,
    herd_x=FFN_ACCUM_HERD_X,
    tile_k=FFN_ACCUM_TILE_K,
):
    """Build the one-segment resident FFN interior for one TILE_M-row band.

    Args:
        seq_len: must equal ``TILE_M`` (64) -- one band; see FOOTGUNS.
        ffn_dim, emb_dim: ``y[seq, emb] = gelu(hidden[seq, emb] @
            w_up[emb, ffn]) @ w_down[ffn, emb]``. ``emb_dim`` must split over
            ``herd_x`` into the kernel object's tile_n (``group_n``), and
            ``ffn_dim`` must divide by ``herd_x * group_n`` (whole sweeps).
        np_dtype: element type; bf16, matching the kernels' C linkage.
        herd_x: columns per herd (three herds of ``[herd_x, 1]``). Capped by
            J7b's measured placement wall for the down herd's shim pair.
        tile_k: K advance per kernel call, and the chunk width of the
            resident hand-off. Capped by J7b's measured L1 fit.

    Returns:
        air.ir.Module with one function
        ``ffn_resident(hidden, w_up_packed, w_down_packed, y)``.
    """
    if seq_len != TILE_M:
        raise ValueError(
            f"seq_len ({seq_len}) must equal TILE_M ({TILE_M}): R1 is one band; "
            "iterate bands on launch arguments (L3-side), never inside the module"
        )
    if herd_x > MAX_PLACEABLE_HERD_X:
        raise ValueError(
            f"herd_x {herd_x} is above the {MAX_PLACEABLE_HERD_X} that places "
            "(J7b's measured aie-place-tiles wall on the accumulator pair's "
            "shim slots; see builders/ffn_accum.py)"
        )
    if tile_k > MAX_L1_TILE_K:
        raise ValueError(
            f"tile_k {tile_k} is above the {MAX_L1_TILE_K} that fits L1 "
            "(J7b's measured ping-pong worst case; see builders/ffn_accum.py)"
        )
    if emb_dim % herd_x:
        raise ValueError(f"emb_dim ({emb_dim}) must divide by herd_x ({herd_x})")
    group_n = emb_dim // herd_x
    if group_n % 16 or group_n % MICRO:
        raise ValueError(
            f"group_n ({group_n}) must be a multiple of 16 (kernel static_assert)"
        )
    if group_n % tile_k:
        raise ValueError(
            f"group_n ({group_n}) must divide by tile_k ({tile_k}): the H group "
            "leaves the up herd in tile_k-wide chunks"
        )
    if ffn_dim % (herd_x * group_n):
        raise ValueError(
            f"ffn_dim ({ffn_dim}) must divide by herd_x*group_n "
            f"({herd_x * group_n}): the up herd advances in whole sweeps"
        )
    if emb_dim % tile_k or tile_k % MICRO:
        raise ValueError(
            f"emb_dim ({emb_dim}) must divide by tile_k ({tile_k}), itself a "
            f"multiple of {MICRO}: the up k' loop steps by it"
        )
    if seq_len % MICRO:
        raise ValueError(f"seq_len ({seq_len}) must divide by {MICRO}")

    itemsize = np.dtype(np_dtype).itemsize
    sweeps = ffn_dim // (herd_x * group_n)
    chunks_per_group = group_n // tile_k
    k_steps_up = emb_dim // tile_k
    chunk_elems = TILE_M * tile_k  # one H chunk, [TILE_M, tile_k] blocked
    up_b_block = tile_k * group_n  # one w_up group slice per k' step
    down_chunk = tile_k * emb_dim  # one w_down K-step chunk (all columns)
    down_b_block = tile_k * group_n  # one core's slice of it

    # The chunk put's geometry inside the blocked [TILE_M, group_n] C group:
    # the row-major microtile grid holds chunk jj as (TILE_M // MICRO) runs of
    # (tile_k * MICRO) elements, (group_n * MICRO) apart, at linear offset
    # jj * tile_k * MICRO. All literals.
    chunk_run = tile_k * MICRO
    chunk_run_stride = group_n * MICRO
    chunk_runs = TILE_M // MICRO

    # L2: every staged buffer holds ONE step. Doubled for possible ping-pong,
    # as ffn_accum's guard does.
    l2_need = 2 * itemsize * (chunk_elems + herd_x * up_b_block + chunk_elems + down_chunk)
    if l2_need > L2_BYTES:
        raise ValueError(
            f"staged slices need {l2_need} B ping-ponged, over the "
            f"{L2_BYTES}-byte memtile"
        )
    # L1, up core, WITHOUT ping-pong of the group accumulator (A and B tiles
    # doubled; the C group single). Whether aircc also rotates the group
    # buffer across the sweep loop is measured by the structure probe, not
    # assumed here -- if it does, this guard understates and aiecc's
    # allocator refuses loudly.
    l1_up = itemsize * (
        2 * (TILE_M * tile_k) + 2 * (tile_k * group_n) + TILE_M * group_n
    ) + L1_STACK_BYTES
    if l1_up > L1_BYTES:
        raise ValueError(
            f"the up core needs {l1_up} B of L1 (pp A + pp B + resident C), "
            f"over the {L1_BYTES}-byte tile; lower tile_k or herd_x"
        )

    xrt_dtype = type_mapper(np_dtype)
    l3_hidden_ty = MemRefType.get([seq_len, emb_dim], xrt_dtype)
    l3_wup_ty = MemRefType.get([emb_dim * ffn_dim], xrt_dtype)
    l3_wdown_ty = MemRefType.get([ffn_dim * emb_dim], xrt_dtype)
    l3_y_ty = MemRefType.get([seq_len, emb_dim], xrt_dtype)

    l2_space = IntegerAttr.get(T.i32(), MemorySpace.L2)
    l2_a_up_ty = MemRefType.get([chunk_elems], xrt_dtype, memory_space=l2_space)
    l2_b_up_ty = MemRefType.get(
        [herd_x * up_b_block], xrt_dtype, memory_space=l2_space
    )
    l2_h_ty = MemRefType.get([chunk_elems], xrt_dtype, memory_space=l2_space)
    l2_b_down_ty = MemRefType.get([down_chunk], xrt_dtype, memory_space=l2_space)

    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_a_ty = MemRefType.get([TILE_M, tile_k], xrt_dtype, memory_space=l1_space)
    l1_b_ty = MemRefType.get([tile_k, group_n], xrt_dtype, memory_space=l1_space)
    l1_c_ty = MemRefType.get([TILE_M, group_n], xrt_dtype, memory_space=l1_space)
    l1_chunk_ty = MemRefType.get([chunk_elems], xrt_dtype, memory_space=l1_space)

    mm_func = FuncOp(
        FFN_ACCUM_MM_SYMBOL, ([l1_a_ty, l1_b_ty, l1_c_ty], []), visibility="private"
    )
    mm_func.attributes["link_with"] = StringAttr.get(FFN_ACCUM_KERNEL_OBJ)
    mm_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    zero_func = FuncOp(FFN_ACCUM_ZERO_SYMBOL, ([l1_c_ty], []), visibility="private")
    zero_func.attributes["link_with"] = StringAttr.get(FFN_ACCUM_KERNEL_OBJ)
    zero_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    gelu_func = FuncOp(
        GELU_SYMBOL, ([l1_chunk_ty, l1_chunk_ty, T.i32()], []), visibility="private"
    )
    gelu_func.attributes["link_with"] = StringAttr.get(GELU_KERNEL_OBJ)
    gelu_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    Channel(CHANNEL_UP_FEED, size=[herd_x, 1])
    Channel(CHANNEL_H, size=[herd_x, 1])
    Channel(CHANNEL_G, size=[herd_x, 1])
    Channel(CHANNEL_DOWN_FEED, size=[herd_x, 1])

    @FuncOp.from_py_func(l3_hidden_ty, l3_wup_ty, l3_wdown_ty, l3_y_ty)
    def ffn_resident(arg0, arg1, arg2, arg3):

        @launch(operands=[arg0, arg1, arg2, arg3])
        def fr_launch(l_hidden, l_wup, l_wdown, l_y):

            @segment(
                name="ffn_resident_seg", operands=[l_hidden, l_wup, l_wdown, l_y]
            )
            def fr_seg(s_hidden, s_wup, s_wdown, s_y):
                l2_a_up = AllocOp(l2_a_up_ty, [], [])
                l2_b_up = AllocOp(l2_b_up_ty, [], [])
                l2_h = AllocOp(l2_h_ty, [], [])
                l2_b_down = AllocOp(l2_b_down_ty, [], [])

                @herd(name="ffn_res_up", sizes=[herd_x, 1])
                def up_herd(tx, ty, _sx, _sy):
                    for _s in range_(0, sweeps):
                        # The group accumulator: L1-resident across k' BY
                        # CONSTRUCTION -- there is no C DMA here at all, so
                        # nothing needs the hoist pass. What survives of J7b
                        # is the in-place kernel and the zero discipline.
                        l1_c = AllocOp(l1_c_ty, [], [])
                        CallOp(zero_func, [l1_c])
                        for _k in range_(0, k_steps_up):
                            l1_a = AllocOp(l1_a_ty, [], [])
                            l1_b = AllocOp(l1_b_ty, [], [])
                            ChannelGet(CHANNEL_UP_FEED, l1_a, indices=[tx, ty])
                            ChannelGet(CHANNEL_UP_FEED, l1_b, indices=[tx, ty])
                            CallOp(mm_func, [l1_a, l1_b, l1_c])
                            DeallocOp(l1_a)
                            DeallocOp(l1_b)
                            yield_([])
                        # Hand the finished group to GeLU in down-K order.
                        # Literal offsets, Python-unrolled: no L1-side offset
                        # ever depends on an induction variable.
                        for jj in range(chunks_per_group):
                            ChannelPut(
                                CHANNEL_H,
                                l1_c,
                                offsets=[jj, 0, 0],
                                sizes=[1, chunk_runs, chunk_run],
                                strides=[chunk_run, chunk_run_stride, 1],
                                indices=[tx, ty],
                            )
                        DeallocOp(l1_c)
                        yield_([])

                up_herd.attributes["link_with"] = StringAttr.get(
                    FFN_ACCUM_KERNEL_OBJ
                )

                @herd(name="ffn_res_gelu", sizes=[herd_x, 1])
                def gelu_herd(tx, ty, _sx, _sy):
                    size_i32 = ConstantOp(T.i32(), chunk_elems)
                    l1_in = AllocOp(l1_chunk_ty, [], [])
                    l1_out = AllocOp(l1_chunk_ty, [], [])
                    for _t in range_(0, sweeps * chunks_per_group):
                        ChannelGet(CHANNEL_H, l1_in, indices=[tx, ty])
                        # Separate in/out tiles: the kernel's operands are
                        # __restrict (see FOOTGUNS). Blocked order is
                        # irrelevant to an elementwise map.
                        CallOp(gelu_func, [l1_in, l1_out, size_i32])
                        ChannelPut(CHANNEL_G, l1_out, indices=[tx, ty])
                        yield_([])
                    DeallocOp(l1_in)
                    DeallocOp(l1_out)

                gelu_herd.attributes["link_with"] = StringAttr.get(GELU_KERNEL_OBJ)

                @herd(name="ffn_res_down", sizes=[herd_x, 1], operands=[s_y])
                def down_herd(tx, ty, _sx, _sy, h_y):
                    # J7b's accumulator herd, verbatim but for the feed's
                    # name: naive per-K-step C round trip on y, in-loop
                    # allocs, guarded first-step zero. Do not "optimize" any
                    # of it by hand; air-hoist-dma-in-accum-pattern lifts the
                    # C pair (the structure check reads the 4 -> 2).
                    n_micro = affine_apply(_mul_add_map(group_n // MICRO), [tx])
                    m_micro = affine_apply(_mul_add_map(TILE_M // MICRO), [ty])
                    k_first = ConstantOp.create_index(0)
                    for _k in range_(0, ffn_dim, tile_k):
                        l1_a = AllocOp(l1_a_ty, [], [])
                        l1_b = AllocOp(l1_b_ty, [], [])
                        l1_c = AllocOp(l1_c_ty, [], [])
                        ChannelGet(CHANNEL_DOWN_FEED, l1_a, indices=[tx, ty])
                        ChannelGet(CHANNEL_DOWN_FEED, l1_b, indices=[tx, ty])
                        dma_memcpy_nd(
                            l1_c,
                            h_y,
                            src_offsets=[m_micro, n_micro, 0, 0],
                            src_sizes=[
                                TILE_M // MICRO,
                                group_n // MICRO,
                                MICRO,
                                MICRO,
                            ],
                            src_strides=[MICRO * emb_dim, MICRO, emb_dim, 1],
                        )
                        is_first = CmpIOp(CmpIPredicate.eq, _k, k_first)
                        if_first = IfOp(is_first.result)
                        with InsertionPoint(if_first.then_block):
                            CallOp(zero_func, [l1_c])
                            yield_([])
                        CallOp(mm_func, [l1_a, l1_b, l1_c])
                        dma_memcpy_nd(
                            h_y,
                            l1_c,
                            dst_offsets=[m_micro, n_micro, 0, 0],
                            dst_sizes=[
                                TILE_M // MICRO,
                                group_n // MICRO,
                                MICRO,
                                MICRO,
                            ],
                            dst_strides=[MICRO * emb_dim, MICRO, emb_dim, 1],
                        )
                        DeallocOp(l1_a)
                        DeallocOp(l1_b)
                        DeallocOp(l1_c)
                        yield_([])

                down_herd.attributes["link_with"] = StringAttr.get(
                    FFN_ACCUM_KERNEL_OBJ
                )

                # ---- UP FEED (emitted after the herds; see FOOTGUNS) ----
                # Per (sweep, k'): refill the hidden k'-slice -- ROW-MAJOR in
                # L3, blocked in L2, retiled by the shim's 4-D read pattern
                # (seam 1) -- and the w_up slice for all herd_x groups, then
                # send each core its pair. Both refills advance on L3.
                a_col_map = _mul_add_map(tile_k // MICRO)
                wup_map = _two_iv_map(
                    k_steps_up * herd_x * up_b_block, herd_x * up_b_block
                )
                for s in range_(0, sweeps):
                    for k in range_(0, k_steps_up):
                        a_col = affine_apply(a_col_map, [k])
                        dma_memcpy_nd(
                            l2_a_up,
                            s_hidden,
                            src_offsets=[0, a_col, 0, 0],
                            src_sizes=[
                                TILE_M // MICRO,
                                tile_k // MICRO,
                                MICRO,
                                MICRO,
                            ],
                            src_strides=[MICRO * emb_dim, MICRO, emb_dim, 1],
                        )
                        w_off = affine_apply(wup_map, [s, k])
                        dma_memcpy_nd(
                            l2_b_up,
                            s_wup,
                            src_offsets=[w_off],
                            src_sizes=[herd_x * up_b_block],
                            src_strides=[1],
                        )
                        for c in range(herd_x):
                            ChannelPut(CHANNEL_UP_FEED, l2_a_up, indices=[c, 0])
                            ChannelPut(
                                CHANNEL_UP_FEED,
                                l2_b_up,
                                offsets=[c * up_b_block],
                                sizes=[up_b_block],
                                strides=[1],
                                indices=[c, 0],
                            )
                        yield_([])
                    yield_([])

                # ---- DOWN FEED ----
                # Per sweep, drain GeLU column c for chunks_per_group chunks,
                # in group order: K step j = (s*herd_x + c)*chunks_per_group
                # + jj, strictly increasing. Only c is Python-unrolled -- it
                # indexes a sub-channel, and channel indices are compile-time
                # (H5); jj is a REAL loop. That split is measured, not style:
                # air-dma-to-channel converts each TEXTUAL dma_memcpy_nd
                # instance at segment scope into its own auto channel, and a
                # fully unrolled (c, jj) body left 24 copies of the w_down
                # refill = 24 channels, which put air-isolate-async-dma-
                # loop-nests into a >25-minute state on a 692-line module
                # (measured 2026-08-11; the stitched tail's 15-16 channels
                # compile in ~4 s). One textual instance per channel op is
                # the rule this loop shape exists to keep.
                #
                # ORDERING across the four per-c loops (the down cores'
                # A|B interleave requires j order on every sub-channel) is
                # carried by the SHARED staging buffers: c+1's first get
                # writes l2_h (WAR on c's last A put), and every put reads
                # what a chained refill wrote, so air-dependency serializes
                # the chain c0 < c1 < c2 < c3 without a hand token. Giving
                # each c its own staging buffer would delete exactly that
                # serialization -- do not "parallelize" it.
                for s in range_(0, sweeps):
                    for c in range(herd_x):
                        wd_map = _two_iv_map(
                            herd_x * chunks_per_group * down_chunk,
                            down_chunk,
                            c * chunks_per_group * down_chunk,
                        )
                        for jj in range_(0, chunks_per_group):
                            ChannelGet(CHANNEL_G, l2_h, indices=[c, 0])
                            wd_off = affine_apply(wd_map, [s, jj])
                            dma_memcpy_nd(
                                l2_b_down,
                                s_wdown,
                                src_offsets=[wd_off],
                                src_sizes=[down_chunk],
                                src_strides=[1],
                            )
                            for tx in range(herd_x):
                                ChannelPut(
                                    CHANNEL_DOWN_FEED, l2_h, indices=[tx, 0]
                                )
                                ChannelPut(
                                    CHANNEL_DOWN_FEED,
                                    l2_b_down,
                                    offsets=[tx * down_b_block],
                                    sizes=[down_b_block],
                                    strides=[1],
                                    indices=[tx, 0],
                                )
                            yield_([])
                    yield_([])

                DeallocOp(l2_a_up)
                DeallocOp(l2_b_up)
                DeallocOp(l2_h)
                DeallocOp(l2_b_down)


def ffn_resident_pack_w_up(w_up, herd_x=FFN_ACCUM_HERD_X, tile_k=FFN_ACCUM_TILE_K):
    """Pre-tile ``w_up``: flat, (sweep, k'-step, column)-major, blocked.

    Element ``w_up[k, n]`` lands in the ``[tile_k, group_n]`` block for
    ``(s, k', c)`` where ``n`` falls in group ``g = s * herd_x + c`` and ``k``
    in k'-step ``k // tile_k``; within a block, the kernels' blocked layout (a
    row-major grid of row-major 8x8 microtiles). One (s, k') refill -- all
    herd_x columns' slices -- is then ONE contiguous L3 run, and each column's
    feed put a contiguous slice of it at a tx-literal offset.
    """
    emb_dim, ffn_dim = w_up.shape
    group_n = emb_dim // herd_x
    sweeps = ffn_dim // (herd_x * group_n)
    return np.ascontiguousarray(
        w_up.reshape(
            emb_dim // tile_k,
            tile_k // MICRO,
            MICRO,
            sweeps,
            herd_x,
            group_n // MICRO,
            MICRO,
        )
        .transpose(3, 0, 4, 1, 5, 2, 6)
        .reshape(-1)
    )


def ffn_resident_device_inputs(
    hidden, w_up, w_down, herd_x=FFN_ACCUM_HERD_X, tile_k=FFN_ACCUM_TILE_K
):
    """Every argument the host fills, in signature order (``y`` excluded).

    ``hidden`` stays ROW-MAJOR (the device retiles it on the shim read --
    seam 1; handing it pre-tiled here computes garbage silently). The weights
    are pre-tiled: ``w_up`` by this module's packer, ``w_down`` by
    ``ffn_accum_pack_w`` -- the down feed reads J7b's exact layout. The
    packing parameters MUST match the ``build_ffn_resident_module`` call.
    """
    return [
        np.ascontiguousarray(hidden),
        ffn_resident_pack_w_up(w_up, herd_x=herd_x, tile_k=tile_k),
        ffn_accum_pack_w(w_down, herd_x=herd_x, tile_k=tile_k),
    ]


def ffn_resident_reference(hidden, w_up, w_down):
    """FP32 reference: ``gelu_tanh(hidden @ w_up) @ w_down``, rounded once.

    The same function ``builders/ffn.py`` computes, imported rather than
    re-derived (two references for one function is two chances to drift).
    FP32 end to end: the device's bf16 staging -- the up ring's per-k'
    rounding, GeLU's bf16 intermediates, the down ring's per-K rounding -- is
    measured as error, not reproduced (the rule every builder here follows).
    """
    return ffn_reference(hidden, w_up, w_down)
