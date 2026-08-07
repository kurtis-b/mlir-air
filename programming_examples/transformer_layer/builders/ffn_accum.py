# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The FFN down-projection as a compiler-formed accumulator ring.

CONTRACT
    ``build_ffn_accum_module(seq_len, ffn_dim, emb_dim, ...)`` returns an
    ``air.ir.Module`` with one function ``ffn_accum(a_packed, w_packed, y)``
    computing

        y = a @ w        a: [seq_len, ffn_dim]  w: [ffn_dim, emb_dim]  bf16

    -- the FFN's down-projection GEMM, written as the NAIVE K loop: per K
    step fetch C from DDR, call the in-place accumulating kernel, store C
    back to DDR -- the first iteration zeroing the fetched tile in L1 before
    it accumulates (a guarded ``ffn_zero_bf16_up_proj`` call), so ``y``'s
    initial DDR contents never reach the result and a reused, un-cleared
    output BO is safe. The builder deliberately ships that DDR round trip
    per K step; ``air-hoist-dma-in-accum-pattern`` (second in the aircc
    pipeline, unconditional) recognizes the mirrored C pair and lifts it out
    of the loop, so the accumulator that actually runs is L1-resident across
    K. THE RING IS THE COMPILER'S, NOT THIS FILE'S: nothing here declares a
    channel, a memory space, or a placement for the accumulator. If a change
    stops the pass matching, the module still computes the right numbers --
    at one DDR round trip per K step -- which is why the suite carries a
    structural check (``ffn_accum_structure.py``) and not just numerics.

    ``a_packed`` and ``w_packed`` are pre-tiled by the packing helpers below;
    callers build the input list with ``ffn_accum_device_inputs(a, w)``.
    Handing the device plain row-major operands produces silently wrong
    numbers, not a crash.

THE TWO CONDITIONS THE HOIST NEEDS (both measured, phase J7b)
    - IN-PLACE kernel, not the two-buffer one. ``ffn_matmul_bf16_bf16_up_proj``
      (A, B, C) reads C, accumulates, writes C: one memref on both sides of
      the round trip, which is what ``areSymmetricDmaOps`` requires. The
      four-operand ``ffn_matmul_with_acc_bf16_bf16_down_proj`` (A, B, pAcc, C)
      splits the pair across two ``__restrict`` memrefs and NEVER matches, at
      any alloc site. Yes, this file computes the DOWN-projection with the
      kernel named ``up_proj``: the entry points are named for the FFN stage
      they were ported for, but the in-place variant only exists under the
      up_proj name and is shape-generic through -DDIM_M/K/N.
    - L1 buffers allocated INSIDE the K loop. Counter-intuitive: you write
      "allocate per iteration" to get "resident across iterations", because
      ``isIncomingDmaOp``/``isOutgoingDmaOp`` require the DMA to be tied to an
      in-loop ``memref.alloc``/``dealloc``, and the pass hoists the alloc and
      the mirrored DMA pair together. Allocate at herd scope -- the natural
      way -- and nothing hoists (measured: 4 -> 4).

WHY A AND B SHARE ONE MEMTILE FEED -- THREE WALLS, ALL MEASURED HERE
    The naive form gives every core three inbound streams: A, B and the
    accumulator fetch. That number hits two independent ceilings (and the
    fix for them hits a third, below):

    1. SHIM (the J7a column rule). ``air-dma-to-channel`` estimates
       per-column shim MM2S pressure as one slot per L3->L1 input channel
       declaration (broadcasts count ceil(count/span)) against the 2 shim
       MM2S ports per column, and at 3 it packet-multiplexes EVERY input --
       measured at herd_x 4 and 8 alike (A auto-broadcasts and still counts
       1). The packet path is the one J1 lost a night to and, post-H9,
       exhausts the 16 shim BDs at this trip count.
    2. CORE. Fixing the shim wall by staging ONE operand through L2 leaves
       three inbound streams AT THE CORE -- feed, direct fetch, hoisted C
       fetch -- and an AIE2P core tile has TWO S2MM DMA channels. aiecc
       refuses the third circuit route: ``'aie.connect' op ... targets same
       dst as another connect op`` on the core tile. Phase J7b's original
       aircc probe never saw this because its three streams had been
       silently packet-upgraded, and packet flows share ports.

    J7a's cure -- both operands in one fetch, one L1 tile -- is unavailable
    verbatim: the kernel is an external callee with separate A and B
    pointers, and an offset subview of a packed L1 tile cannot reach an
    external callee (air-to-aie normalizes signatures; see
    builders/norm_tail.py). What fits both ceilings is packing the two
    CO-INDEXED operands onto one CHANNEL rather than into one buffer: per K
    step the memtile sends A's k-slice and then B's k-slice down the same
    channel, and the core issues two gets into two separate L1 buffers. One
    core S2MM port for A+B, one for the hoisted C fetch: exactly the budget.
    Both operands stage through the memtile to make that possible (a channel
    has one physical source), and BOTH stage the same way: a small buffer
    holding ONE K step, refilled from L3 each step. The L3->L2 copies have no
    herd-side endpoint, so they are allocated globally across shim columns:
    per-column shim usage is the C fetch plus at most one global copy. Input
    pressure 2, output pressure 1 (the accumulator store), zero packet-typed
    channels -- measured.

    3. STATIC BD OFFSETS -- the wall that decides how A is staged.
       The obvious economy is to stage A WHOLE in the memtile (it is the
       small operand, and one contiguous L3 copy serves every K step) and
       put each step's slice from an advancing offset. That construction
       COMPILES, places, keeps zero packet-typed channels, and hoists 4 -> 2
       exactly like this one -- and it is WRONG past two K steps.

       Measured: the memtile MM2S program is a static ``aie.dma_bd`` chain,
       and at 2 trips the K loop fully unrolls so each put carries its own
       literal offset (0, then 2048) and the chain IS the whole computation.
       At 4 trips and beyond the loop no longer unrolls, the chain is a
       4-BD cycle covering two steps, and both A BDs read the L2 buffer at
       offset 0 -- the per-iteration offset is simply dropped. The core then
       stalls, the accumulator store never fires, and ``y`` is returned
       byte-identical to what the host supplied (measured: seed y with 1.0
       and 4096/4096 elements come back 1.0). At 4 columns x 96 steps it
       stops returning at all: ERT_CMD_STATE_TIMEOUT.

       This is doc 16's H5 -- compile-time-only channel/BD indexing -- with a
       worse failure mode than H5 records: H5 says such a loop fully unrolls
       and exhausts locks, which at least fails loudly at compile time. Here
       it declines to unroll and silently emits a chain that repeats a stale
       offset forever. Not ping-pong: ``omit_pingpong="all"`` reproduces it
       identically.

       So NEITHER operand is read at a per-iteration L2 offset. Both advance
       on the L3 side, where the shim streams contiguously and no BD needs a
       moving offset -- which is why B always worked and A did not. Reproduce
       with ``agents/probes/probe_ffn_accum_bd_offset.py``.

    THE ACCUMULATOR PATH IS UNTOUCHED BY ALL OF THIS. The C fetch and store
    stay the naive in-loop ``dma_memcpy_nd`` pair on the DDR buffer; only
    the pass moves them.

WHY THE OPERANDS ARRIVE PRE-TILED
    The kernels consume blocked operands -- row-major grids of row-major 8x8
    microtiles (r=s=t=8; kernels/encoder.cc). Retiling in flight would put a
    4-D pattern on every memtile feed put; pre-tiling on the host
    (``ffn_accum_pack_a`` / ``_pack_w``) makes every A/B transfer in the
    design CONTIGUOUS 1-D -- which is also what lets each operand's L3->L2
    refill be a plain contiguous run per K step (herd_y > 1 gathers herd_y of
    them with one stride). C is NOT
    pre-tiled: y stays a plain row-major [seq, emb] tensor, and the C
    fetch/store retile it with the standard 4-D shim idiom (sizes
    [rows/8, cols/8, 8, 8] against strides [8*ld, 8, ld, 1]), because the
    hoisted pair rides shim DMAs which handle 4-D.

WHY THE ZERO IS A GUARDED CALL INSIDE THE K LOOP, NOT A PRE-LOOP STORE --
MEASURED
    The kernel's contract requires C zeroed or holding a valid partial sum.
    The phase spec's construction zeroes ``y`` in DDR before the loop (each
    core running the zero kernel on a scratch tile and STORING it to its
    slice of ``y``). That store is a SECOND shim S2MM stream per column,
    next to the accumulator store -- and ``aie-place-tiles`` refuses it:
    ``no ShimNOCTile has sufficient DMA capacity for 0 input/2 output
    channels near centroid column 2``. The J7a column budget, on the output
    side this time.

    What fits the budget is zeroing the L1 TILE, not the DDR buffer: a
    ``ffn_zero_bf16_up_proj`` call inside the K loop, guarded by ``k == 0``,
    between the accumulator fetch and the accumulate call. No DMA is
    involved, so the column budget never sees it, and the module is correct
    in BOTH forms it can take -- naive, iteration 0 fetches whatever ``y``
    holds and zeroes over it; hoisted, the single pre-loop fetch's stale
    read is overwritten before the first accumulate. Either way ``y``'s
    initial contents never reach the result, so a raw XRT caller reusing an
    un-cleared BO gets ``a @ w`` and not ``y_old + a @ w`` (review round 2's
    finding: ``XRTRunner.run_test`` zero-fills its output placeholders, so
    the numeric arm could never see a reliance on ``y`` entering zeroed).

    The guard is ``scf.if``, not ``affine.if``: the hoist pattern matches
    ``scf.for`` only, and an ``scf.for`` induction variable is not a valid
    affine operand. The pass scans the loop's DIRECT dma ops and does not
    look inside the conditional, so the ring forms exactly as before
    (measured: still 4 -> 2, zero packet-typed channels).

FOOTGUNS
    - The L1/L2 memref types are identity-layout and the DMAs fill them in
      blocked microtile order. MLIR never indexes into them (the kernel gets
      a base pointer via the C ABI), so the declared shape is an extent, not
      a layout claim. Do not "fix" the types to strided forms.
    - ``compile_ffn_accum_kernel`` must have run in the CWD before aiecc
      links: the object bakes DIM_M/DIM_K/DIM_N in as -D flags, so it is a
      DIFFERENT object from encoder_ffn.o (tile_n 192 vs 64 at the defaults)
      and is built to its own name. Linking encoder_ffn.o instead silently
      computes with the wrong microkernel shape -- and so does compiling the
      object at a tile_n other than the module's ``emb_dim // herd_x``,
      which is why the helper's default is DERIVED from the module's default
      shape rather than stated as its own number.
    - ``herd_x * herd_y`` is capped at 6: every core's feed is its own
      memtile MM2S channel and a memtile has six. This -- not the AIE array
      -- is what stops herd_x=8 here.
    - The guarded zero sits AFTER the accumulator fetch, on purpose. Put it
      before the fetch and the fetch overwrites the zeros with ``y``'s stale
      contents -- on every first-K-step in the naive form, once in the
      hoisted form, which is enough to ship ``y_old + a @ w``. The body
      order fetch-zero-accumulate is what makes both forms correct.
    - The feed loop is emitted AFTER the herd in the segment body, on
      purpose: the driver's structural check counts data movement inside the
      FIRST ``scf.for`` of the dump, which must be the herd's K loop, not the
      feed loop. The order changes nothing semantically (async dataflow).
    - Per feed sub-channel the put/get order is A then B, every K step. The
      two transfers have different sizes (A: TILE_M*tile_k, B:
      tile_k*tile_n), so a swapped get order is not a subtle numerical bug,
      it is a hang -- worth knowing before staring at a timeout.
    - Accumulation is bf16 in C between K steps (f32 only inside one call's
      MMUL), so the running sum rounds to bf16 ``ffn_dim / tile_k`` times.
      That is the honest cost of the in-place ring at this kernel's ABI, it
      is what the measured atol in opcheck_specs.py is sized against, and
      the FP32 reference below deliberately reproduces none of it.
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

range_ = for_

# The aie::mmul microtile edge (r = s = t = 8). Every tile dimension below is
# a multiple of it, and the packing helpers block by it.
MICRO = 8

# DIM_M is baked into the kernel object; the builder derives the herd's row
# count from it. 64 is the shape every encoder.cc port was validated at.
TILE_M = 64

# The production geometry: herd_x column-parallel cores, K advanced tile_k per
# call. Set by measured walls, not a performance sweep -- aie-place-tiles
# refuses the accumulator pair's shim slots past 4 columns, and tile_k 64 puts
# the worst-case L1 just over the 64 KiB tile (see build_ffn_accum_module's
# docstring) -- which is why these live here and not in kernel_registry: the
# registry records swept best-measured tile configs and raises on unmeasured
# shapes, and no such sweep exists for this wall-bounded design. Import these
# rather than hand-copying the numbers; opcheck_prepare and
# ffn_accum_structure both do.
FFN_ACCUM_HERD_X = 4
FFN_ACCUM_TILE_K = 32

# The catalogue's one claimed width (opcheck_specs 64x3072x768), which is also
# what the driver's no-argument objective-check build gets. It exists so that
# compile_ffn_accum_kernel's default tile_n is DERIVED from the same numbers
# build_ffn_accum_module defaults to (emb_dim / herd_x): the object bakes its
# tiles in as -D flags, so two no-argument calls that disagree link a
# microkernel whose blocked ABI does not match the module's transfers --
# corrupt output, no link error (review round 2).
FFN_ACCUM_EMB_DIM = 768

# One feed sub-channel per core, each a memtile MM2S port; a memtile has 6.
MAX_FEED_CHANNELS = 6

# The two MEASURED ceilings, named separately from the production geometry above
# because "what we ship" and "what works" are different claims that happen to
# coincide here. Both were documented in prose while the builder still accepted
# them, so the knowledge did not reach a caller until aiecc or aie-place-tiles
# failed minutes later; build_ffn_accum_module now refuses them with the
# measurement in the message.
MAX_PLACEABLE_HERD_X = 4
MAX_L1_TILE_K = 32

# A memtile's memory, the ceiling on the staged operands. Both stages hold ONE
# K step now, so this is roomy -- it was the binding constraint when A was
# staged whole (480 KB of 512 KB at the spec shape).
L2_BYTES = 512 * 1024

# The in-place accumulating entry point -- the ABI string aircc binds the
# CallOp to. See the module docstring for why the down-projection dispatches
# the `up_proj`-named kernel.
FFN_ACCUM_MM_SYMBOL = "ffn_matmul_bf16_bf16_up_proj"

# The device-side zero for the accumulator tile: called on the K loop's FIRST
# iteration only, between the accumulator fetch and the accumulate call, so
# y's initial DDR contents never reach the result. Same object, same -DDIM_*
# set as the matmul. See the module docstring for why the zero is this guarded
# in-L1 call and not the spec's pre-loop store.
FFN_ACCUM_ZERO_SYMBOL = "ffn_zero_bf16_up_proj"

# Distinct from encoder_ffn.o: same translation unit, different -DDIM_* set.
FFN_ACCUM_KERNEL_OBJ = "ffn_accum_mm.o"

# The hand-declared L2->L1 A|B feed. The accumulator ring itself declares no
# channel anywhere -- that is the phase's point.
CHANNEL_FEED = "ffn_accum_feed"


def compile_ffn_accum_kernel(
    tile_k=FFN_ACCUM_TILE_K, tile_n=FFN_ACCUM_EMB_DIM // FFN_ACCUM_HERD_X
):
    """Build ffn_accum_mm.o (encoder.cc's FFN half at THIS builder's tiles)
    into the CWD, where aiecc's link_with looks.

    The tile_n default is derived from the module's own default shape (emb_dim
    768 over herd_x columns), NOT chosen independently: the two no-argument
    calls must describe the same kernel object. Callers at a real shape pass
    ``emb_dim // herd_x`` explicitly, as opcheck_prepare does.
    """
    import shared.infra.external_kernels as ek

    ek.compile_encoder(
        tile_m=TILE_M,
        tile_k=tile_k,
        tile_n=tile_n,
        build_ffn=True,
        build_addnorm=False,
        out_name=FFN_ACCUM_KERNEL_OBJ,
    )


def ffn_accum_pack_a(a, tile_k=FFN_ACCUM_TILE_K):
    """Pre-tile ``a``: flat, (m-tile, K-step)-major, blocked microtiles.

    Element ``a[m, k]`` lands in block ``(m // TILE_M, k // tile_k)``; within
    a block, the [TILE_M, tile_k] slice is the kernels' blocked layout -- a
    row-major grid of row-major 8x8 microtiles. Every device-side A transfer
    is then contiguous (see the module docstring).
    """
    seq_len, ffn_dim = a.shape
    return np.ascontiguousarray(
        a.reshape(
            seq_len // TILE_M,
            TILE_M // MICRO,
            MICRO,
            ffn_dim // tile_k,
            tile_k // MICRO,
            MICRO,
        )
        .transpose(0, 3, 1, 4, 2, 5)
        .reshape(-1)
    )


def ffn_accum_pack_w(w, herd_x=FFN_ACCUM_HERD_X, tile_k=FFN_ACCUM_TILE_K):
    """Pre-tile ``w``: flat, (K-step, n-tile)-major, blocked microtiles.

    Element ``w[k, n]`` lands in block ``(k // tile_k, n // tile_n)`` with
    ``tile_n = emb_dim // herd_x``; within a block, the [tile_k, tile_n]
    slice is the kernels' blocked layout. One K step's chunk -- all herd_x
    column slices -- is then one contiguous L3->L2 copy, and each column's
    slice one contiguous feed put.
    """
    ffn_dim, emb_dim = w.shape
    tile_n = emb_dim // herd_x
    return np.ascontiguousarray(
        w.reshape(
            ffn_dim // tile_k,
            tile_k // MICRO,
            MICRO,
            herd_x,
            tile_n // MICRO,
            MICRO,
        )
        .transpose(0, 3, 1, 4, 2, 5)
        .reshape(-1)
    )


def ffn_accum_device_inputs(a, w, herd_x=FFN_ACCUM_HERD_X, tile_k=FFN_ACCUM_TILE_K):
    """Every argument the host fills, in signature order (``y`` excluded).

    The packing parameters MUST match the ``build_ffn_accum_module`` call;
    keeping this beside the builder is what stops a caller from getting the
    layout right today and wrong the next time a tile size changes.
    """
    return [ffn_accum_pack_a(a, tile_k=tile_k), ffn_accum_pack_w(w, herd_x, tile_k)]


def _mul_add_map(factor, offset=0):
    """AffineMap for ``s0 * factor + offset``.

    Built lazily: AffineMap construction needs the ambient MLIR context that
    ``@module_builder`` provides, so these cannot be module-level constants.
    """
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


def _floordiv_map(divisor):
    """AffineMap for ``s0 floordiv divisor``."""
    return AffineMap.get(
        0,
        1,
        [
            AffineExpr.get_floor_div(
                AffineSymbolExpr.get(0), AffineConstantExpr.get(divisor)
            )
        ],
    )


@module_builder
def build_ffn_accum_module(
    seq_len=64,
    ffn_dim=3072,
    emb_dim=FFN_ACCUM_EMB_DIM,
    np_dtype=bfloat16,
    herd_x=FFN_ACCUM_HERD_X,
    tile_k=FFN_ACCUM_TILE_K,
):
    """Build the down-projection accumulator module.

    The shape arguments default to the operator's ONE claimed catalogue shape
    (``64x3072x768``) because the driver's objective check calls this builder
    with no arguments (``phases.sh``). That is out of step with every sibling
    here -- ``build_norm_tail_module``/``build_layer_norm_module`` take their
    shape positionally, and J7a's own objective check passes one
    (``build_norm_tail_module(4096, 768, herd_x=8)``). The defaults exist so a
    harness inconsistency cannot fail this phase for a reason that has nothing
    to do with accumulator rings; the check itself should be corrected to pass
    the shape, which is an operator edit BETWEEN runs (the driver's scripts are
    fingerprinted, and no phase's allowlist covers them -- deliberately).
    Callers with a real shape should keep passing one: ``opcheck_prepare`` does.

    Args:
        seq_len, ffn_dim, emb_dim: GEMM shape, ``y[seq, emb] = a[seq, ffn] @
            w[ffn, emb]``. ``seq_len`` must divide by ``TILE_M`` (64);
            ``emb_dim`` by ``herd_x`` into a tile_n that is a multiple of 16
            (the kernel's static_assert). Each core owns exactly one output
            tile, so ``herd_x * (seq_len / TILE_M)`` cores are used -- at
            most ``MAX_FEED_CHANNELS`` (the memtile MM2S ports feeding A|B).
        np_dtype: element type; bf16, matching the kernel's C linkage.
        herd_x: columns, one output-tile column each. The memtile-port cap
            admits 6, but ``aie-place-tiles`` refuses the accumulator pair's
            shim slots at 6 columns ("no ShimNOCTile has sufficient DMA
            capacity for 1 input/1 output channels near centroid column 3",
            with all 16 slots demonstrably free) -- a windowed-search
            artifact of that binary pass, measured here. 4 places.
        tile_k: K advance per kernel call, multiple of 8. 32 keeps the
            worst-case L1 (ping-ponged A and B tiles plus the resident C)
            comfortably under the 64 KiB tile; 64 measures just over it.

    Returns:
        air.ir.Module with one function ``ffn_accum(a_packed, w_packed, y)``.
    """
    tile_n = emb_dim // herd_x
    herd_y = seq_len // TILE_M
    if seq_len % TILE_M or herd_y < 1:
        raise ValueError(
            f"seq_len ({seq_len}) must be a positive multiple of {TILE_M}: "
            "each core owns exactly one output tile"
        )
    if herd_x * herd_y > MAX_FEED_CHANNELS:
        raise ValueError(
            f"herd {herd_x}x{herd_y} needs {herd_x * herd_y} feed memtile "
            f"MM2S ports; a memtile has {MAX_FEED_CHANNELS}"
        )
    if emb_dim % herd_x or tile_n % 16:
        raise ValueError(
            f"emb_dim ({emb_dim}) must split across herd_x ({herd_x}) into a "
            f"tile_n multiple of 16; got tile_n={tile_n}"
        )
    if ffn_dim % tile_k or tile_k % MICRO:
        raise ValueError(
            f"ffn_dim ({ffn_dim}) must divide by tile_k ({tile_k}), itself a "
            f"multiple of {MICRO}: the K loop and the feed slices step by it"
        )
    # Two settings this file DOCUMENTS as measured-not-to-work were still
    # accepted, so the knowledge lived only in prose and the failure arrived as
    # an aiecc or placement error several minutes later. Refuse them here, and
    # say what was measured -- if a later toolchain lifts either, deleting the
    # check is a deliberate act with a reason attached.
    if herd_x > MAX_PLACEABLE_HERD_X:
        raise ValueError(
            f"herd_x {herd_x} is above the {MAX_PLACEABLE_HERD_X} that places. "
            "Measured at 6 columns: aie-place-tiles refuses the accumulator "
            "pair's shim slots -- \"no ShimNOCTile has sufficient DMA capacity "
            'for 1 input/1 output channels near centroid column 3" -- with all '
            "16 slots demonstrably free, a windowed-search artifact of that "
            "pass. 5 was never measured; it is refused as unproven, not as known bad."
        )
    if tile_k > MAX_L1_TILE_K:
        raise ValueError(
            f"tile_k {tile_k} is above the {MAX_L1_TILE_K} that fits L1. Measured "
            "at 64: the ping-ponged A and B tiles plus the resident accumulator "
            "land just over the 64 KiB tile. Intermediate values are refused as "
            "unproven rather than known bad."
        )
    itemsize = np.dtype(np_dtype).itemsize
    a_elems = seq_len * ffn_dim
    chunk_elems = tile_k * emb_dim  # one K step of packed w
    a_block_elems = TILE_M * tile_k  # one (m-tile, K-step) A feed slice
    b_block_elems = tile_k * tile_n  # one (K-step, n-tile) B feed slice
    # One K step of A for every core row -- NOT the whole tensor. See the
    # module docstring: staging A whole and reading it at a per-iteration
    # offset is the shape that miscompiles past two trips.
    a_stage_elems = herd_y * a_block_elems
    w_elems = ffn_dim * emb_dim
    if (2 * a_stage_elems + 2 * chunk_elems) * itemsize > L2_BYTES:
        raise ValueError(
            f"the A stage ({a_stage_elems * itemsize} B) and W chunk "
            f"({chunk_elems * itemsize} B), each possibly ping-ponged, exceed "
            f"the {L2_BYTES}-byte memtile"
        )

    xrt_dtype = type_mapper(np_dtype)
    l3_a_ty = MemRefType.get([a_elems], xrt_dtype)
    l3_w_ty = MemRefType.get([w_elems], xrt_dtype)
    l3_y_ty = MemRefType.get([seq_len, emb_dim], xrt_dtype)

    l2_space = IntegerAttr.get(T.i32(), MemorySpace.L2)
    l2_a_ty = MemRefType.get([a_stage_elems], xrt_dtype, memory_space=l2_space)
    l2_b_ty = MemRefType.get([chunk_elems], xrt_dtype, memory_space=l2_space)

    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_a_ty = MemRefType.get([TILE_M, tile_k], xrt_dtype, memory_space=l1_space)
    l1_b_ty = MemRefType.get([tile_k, tile_n], xrt_dtype, memory_space=l1_space)
    l1_c_ty = MemRefType.get([TILE_M, tile_n], xrt_dtype, memory_space=l1_space)

    mm_func = FuncOp(
        FFN_ACCUM_MM_SYMBOL, ([l1_a_ty, l1_b_ty, l1_c_ty], []), visibility="private"
    )
    mm_func.attributes["link_with"] = StringAttr.get(FFN_ACCUM_KERNEL_OBJ)
    mm_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    zero_func = FuncOp(FFN_ACCUM_ZERO_SYMBOL, ([l1_c_ty], []), visibility="private")
    zero_func.attributes["link_with"] = StringAttr.get(FFN_ACCUM_KERNEL_OBJ)
    zero_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    Channel(CHANNEL_FEED, size=[herd_x, herd_y])

    @FuncOp.from_py_func(l3_a_ty, l3_w_ty, l3_y_ty)
    def ffn_accum(arg0, arg1, arg2):

        @launch(operands=[arg0, arg1, arg2])
        def fa_launch(l_a, l_w, l_y):

            @segment(name="ffn_accum_seg", operands=[l_a, l_w, l_y])
            def fa_seg(s_a, s_w, s_y):
                # A and B stage IDENTICALLY: a small buffer refilled from L3
                # once per K step, put to the cores at a STATIC offset. Neither
                # is read at a per-iteration offset -- see the module docstring
                # for the miscompile that shape produces past two trips.
                # Neither copy has a herd-side endpoint, so both are allocated
                # globally across shim columns: no per-column MM2S pressure.
                l2_a = AllocOp(l2_a_ty, [], [])
                l2_b = AllocOp(l2_b_ty, [], [])

                @herd(
                    name="ffn_accum_herd",
                    sizes=[herd_x, herd_y],
                    operands=[s_y],
                )
                def fa_herd(tx, ty, _sx, _sy, h_y):
                    # Lazily built: affine maps need the ambient context, and
                    # the factors bake in the runtime-constant tile geometry.
                    n_micro = affine_apply(_mul_add_map(tile_n // MICRO), [tx])
                    m_micro = affine_apply(_mul_add_map(TILE_M // MICRO), [ty])
                    k_first = ConstantOp.create_index(0)

                    # The naive accumulator loop. Every buffer is allocated
                    # INSIDE the loop and the C round trip is written per K
                    # step -- the exact shape air-hoist-dma-in-accum-pattern
                    # lifts. Do not "optimize" either by hand; see the
                    # module docstring.
                    for _k in range_(0, ffn_dim, tile_k):
                        l1_a = AllocOp(l1_a_ty, [], [])
                        l1_b = AllocOp(l1_b_ty, [], [])
                        l1_c = AllocOp(l1_c_ty, [], [])
                        # A then B off this core's feed, pre-tiled upstream.
                        ChannelGet(CHANNEL_FEED, l1_a, indices=[tx, ty])
                        ChannelGet(CHANNEL_FEED, l1_b, indices=[tx, ty])
                        # The accumulator fetch, mirrored by the store below.
                        dma_memcpy_nd(
                            l1_c,
                            h_y,
                            src_offsets=[m_micro, n_micro, 0, 0],
                            src_sizes=[TILE_M // MICRO, tile_n // MICRO, MICRO, MICRO],
                            src_strides=[MICRO * emb_dim, MICRO, emb_dim, 1],
                        )
                        # First iteration only: zero the fetched tile IN L1,
                        # so y's initial DDR contents never reach the result
                        # -- in the naive form AND the hoisted one (the
                        # hoisted fetch's stale read is overwritten here
                        # before the first accumulate). scf.if, not
                        # affine.if: the hoist pattern matches scf.for only,
                        # whose IV is not a valid affine operand. AFTER the
                        # fetch, on purpose; see the FOOTGUNS.
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
                            dst_sizes=[TILE_M // MICRO, tile_n // MICRO, MICRO, MICRO],
                            dst_strides=[MICRO * emb_dim, MICRO, emb_dim, 1],
                        )
                        DeallocOp(l1_a)
                        DeallocOp(l1_b)
                        DeallocOp(l1_c)
                        yield_([])

                fa_herd.attributes["link_with"] = StringAttr.get(FFN_ACCUM_KERNEL_OBJ)

                # The feed: per K step, refill the B chunk, then send each
                # core its A slice and its B slice down its sub-channel --
                # every transfer contiguous by construction of the packing.
                # Emitted AFTER the herd so the herd's K loop is the first
                # scf.for in the IR (the structural check reads it); the
                # dataflow is unchanged by textual order.
                a_chunk_map = _mul_add_map(TILE_M)  # k -> k*TILE_M elements
                w_chunk_map = _mul_add_map(emb_dim)  # k -> k*emb_dim elements
                for k in range_(0, ffn_dim, tile_k):
                    # This K step's A block for every core row. Row m_tile's
                    # blocks are contiguous in k (the packing lays each row
                    # band out K-major), so herd_y=1 is one contiguous run and
                    # the general case is one strided gather.
                    a_off = affine_apply(a_chunk_map, [k])
                    if herd_y == 1:
                        dma_memcpy_nd(
                            l2_a,
                            s_a,
                            src_offsets=[a_off],
                            src_sizes=[a_block_elems],
                            src_strides=[1],
                        )
                    else:
                        dma_memcpy_nd(
                            l2_a,
                            s_a,
                            src_offsets=[a_off],
                            src_sizes=[herd_y, a_block_elems],
                            src_strides=[ffn_dim * TILE_M, 1],
                        )
                    w_off = affine_apply(w_chunk_map, [k])
                    dma_memcpy_nd(
                        l2_b,
                        s_w,
                        src_offsets=[w_off],
                        src_sizes=[chunk_elems],
                        src_strides=[1],
                    )
                    for tx_const in range(herd_x):
                        for ty_const in range(herd_y):
                            ChannelPut(
                                CHANNEL_FEED,
                                l2_a,
                                offsets=[ty_const * a_block_elems],
                                sizes=[a_block_elems],
                                strides=[1],
                                indices=[tx_const, ty_const],
                            )
                            ChannelPut(
                                CHANNEL_FEED,
                                l2_b,
                                offsets=[tx_const * b_block_elems],
                                sizes=[b_block_elems],
                                strides=[1],
                                indices=[tx_const, ty_const],
                            )
                    yield_([])

                DeallocOp(l2_a)
                DeallocOp(l2_b)


def ffn_accum_reference(a, w):
    """FP32 reference for ``a @ w``, rounded once at the end.

    Takes the UNPACKED operands -- packing is device layout, not arithmetic.
    The device rounds the running sum to bf16 once per K step (the in-place
    kernel's C is bf16); this does not, deliberately, so that staging cost is
    measured rather than defined away -- the same rule every builder here
    follows.
    """
    return (a.astype(np.float32) @ w.astype(np.float32)).astype(a.dtype)
