# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The transformer tail as ONE segment: AddNorm -> FFN (partial-sum staging) -> AddNorm.

CONTRACT
    ``build_tail_pipeline_module(seq_len, emb_dim, ffn_dim, tile_m, tile_k,
    tile_n, down_proj_depth, n_b, ...)`` returns an ``air.ir.Module`` with one
    function ``tail_pipeline(x, residual, gamma1, w_up_packed, w_down_packed,
    gamma2, y)``, all bf16, computing per row

        h1 = gamma1 * LayerNorm(x) + residual          (AN1, post-add form)
        c  = gelu_tanh(h1 @ w_up) @ w_down              (FFN)
        y  = gamma2 * LayerNorm(c) + h1                 (AN2, post-add form)

    -- the encoder block's add-norm / FFN / add-norm tail, with the SECOND
    residual being the FIRST add-norm's output (there is no ``residual2``
    argument and no beta: that is what the AN kernel actually takes, and
    what the reference design carries). ``x``, ``residual`` and ``y`` are
    row-major ``[seq_len, emb_dim]``; the two weights are pre-tiled by
    ``tail_pipeline_pack_weights``; callers build the input list with
    ``tail_pipeline_device_inputs``.

PROVENANCE
    A port of iron's ``addnorm_ffn`` pipeline: branch ``dev-addnorm-ffn``,
    commit ``5cebcd7``, file ``operators/addnorm_ffn/design_old.py`` (read
    with ``git -C /home/cj/iron show dev-addnorm-ffn:operators/addnorm_ffn/
    design_old.py``). That design is a Program of Workers joined by
    ObjectFifos; per replica two AN1 cores, ``nB`` up cores, ``nB`` down
    cores and two AN2 cores, with the down cores' PRIVATE partial
    accumulators staged in the memory tile as a ring of depth
    ``down_proj_depth`` (its ``curr_acc_c`` / ``new_acc_c`` fifos) and the
    ``nB`` partials reduced by an L1->L1 chain before the AN2 cores. This
    file keeps that dataflow and its names; what changes is what AIR can
    express, recorded hop by hop below. Only ``n_a = 1`` (one replica) is
    built; ``n_a > 1`` raises ``NotImplementedError``.

THE DATAFLOW (one air.launch, one air.segment, four herds)

    shim  x, residual, gamma1        -> memtile (L2 stage, refilled per band)
    memtile ->[tp_an1_feed]->  AN1 herd [2,1]   core c holds rows
        [c*rpc, (c+1)*rpc) of the tile_m band (rpc = tile_m / 2) and runs
        fused_add_layer_norm_2outs on them
    AN1 ->[tp_an1_a]-> memtile     l2_a[c], ONE writer each, refilled once
        per SWEEP (AN1 re-runs its LN per sweep) and drained k_steps times
        per fill by the A feed
    AN1 ->[tp_an1_out]-> AN2       core->core, the rows as AN2's residual,
        once per band (the one L1->L1 hop besides H)
    memtile ->[tp_a_feed]-> UP herd [n_b,1]     per k' step, a 4-D memtile
        put RETILES the row-major rows into the kernels' blocked
        [tile_m, tile_k] operand (two puts, one per AN1 core, landing in
        the two halves of one L1 tile), BROADCAST to every up core
    memtile ->[tp_wup_feed]-> UP herd          w_up group slices, L3->L2
        refill per (sweep, k'), tx-literal L2 offsets
    UP ->[tp_h]-> DOWN herd [n_b,1]            core->core, L1->L1: the
        finished [tile_m, tile_n] H tile, one per sweep
    memtile ->[tp_down_feed_t]-> DOWN herd t   ONE herd [1,1], one channel
        set and one memtile (its L2 group) PER COLUMN t; the feed carries,
        per (sweep, block d): column t's w_down chunk into l1_b, then
        accumulator block d (its ``curr_acc_c``) into l1_acc_in; after the
        sweeps, per d: a CHUNK-sized transfer into l1_b -- core t+1's
        reduced block in its prefix (``buffer_to_reduce``, from the relay
        buffer l2_red[t]) or, for the chain's last column, a ZERO chunk --
        then the final block d into l1_acc_in: every get the same
        whole-buffer BD, a strict (b, acc) alternation from the first
        transfer to the last (see FOOTGUNS)
    DOWN t ->[tp_acc_store_t]-> memtile        accumulator block d back to
        its ring buffer (``new_acc_c``), one L2 buffer PER BLOCK per column
    DOWN k ->[tp_reduce_{k-1}]-> memtile -> DOWN k-1   the reduced block
        (the add's output for a column with a neighbour; ``final +
        gelu(H) @ 0`` through the accumulate kernel for the last column),
        relayed through L2 (iron's ``buffer_to_reduce``)
    DOWN 0 ->[tp_down_out]-> memtile           the final reduced block d,
        from l1_red (a kernel's output; never a get destination)
    memtile ->[tp_an2_feed]-> AN2 herd [2,1]   gamma2; then per d a 4-D
        memtile put UN-TILES half c of the blocked block into column block
        d of the core's row-major input (the residual arrives core->core)
    AN2 -> shim                                y rows, herd-direct store

    ITERATION. With ONE sweep the band loop (``seq_len / tile_m`` trips)
    is inside the module as in iron (``ln_iters_per_core``). With several
    sweeps the module is ONE band and the host iterates bands over launch
    arguments (``tail_pipeline_rung.py --band-serial``): the down feed
    then lowers to two repeat-counted DMA tasks per band, which AIE cannot
    loop (FOOTGUNS). Every L3 offset advances on the band and sweep IVs,
    and every L2/L1-side offset in the design is a compile-time literal
    (the block index ``d`` and the k' index are Python-unrolled wherever
    they address a staged buffer). That is the frozen-BD rule of
    builders/ffn_accum.py and builders/ffn_resident.py, obeyed by
    construction; see FOOTGUNS.

PORT ARITHMETIC -- why the hops are where they are
    An AIE2P core tile has TWO S2MM and TWO MM2S DMA channels, and every
    AIR channel endpoint on a core is one of them (AIR routes core->core
    over the stream switch; it does not use neighbour shared memory the
    way iron's placed ObjectFifos do). iron's down core has FOUR inbound
    fifos (H, w_down, curr_acc_c, buffer_to_reduce) -- legal there because
    adjacent-tile fifos are lock-guarded shared buffers, not DMAs. Here
    they cannot all be streams, so:
    - w_down, the accumulator block and the reduction partial share ONE
      memtile-sourced channel per down core (the A|B-on-one-channel idiom
      of builders/ffn_accum.py: sequential puts from several L2 buffers on
      one sub-channel, matching gets on the core). L2 buffers referenced
      by one channel are bucketed onto one memtile by air-to-aie, which is
      what lets one port source all of them.
    - the reduction therefore relays through L2 (core k -> l2_red[k-1] ->
      core k-1's feed) instead of L1->L1; H keeps the core->core edge (the
      one iron hop that survives verbatim), and AN1 -> AN2's residual is
      the other core->core edge (AN2's second S2MM port is free).
    - the AN1 core's x, residual and gamma1 would be three inbound streams;
      all three stage through L2 and arrive on ONE channel. The LN kernel
      needs x and residual as two base pointers, so the norm_tail packing
      trick (one packed L1 tile) is unavailable: an offset subview cannot
      reach an external callee (builders/norm_tail.py).
    - the AN2 core's residual, down blocks and gamma2 likewise share one
      channel.
    - AN1 -> up is a memtile broadcast, not the L1->L1 broadcast the spec
      preferred, because the kernels consume BLOCKED microtile operands
      and the row-major -> blocked retile of a [rpc, tile_k] slice is a
      4-D access pattern: a core-tile DMA BD carries at most 3 dimensions
      (mlir-aie's DMABDOp verifier), a memtile BD 4. With rpc == 8 a 3-D
      form exists (iron's m=16 rung); at iron's baseline tile_m=64 it does
      not. Same reason for the blocked -> row-major un-tile into AN2.
    - core->core ``aie.flow`` count is therefore exactly ``n_b`` (the
      up->down edges). The structure check pins that number.

    Memtile groups: the L2 buffers bucket (by shared channel) into the AN1
    input group {x, residual, gamma1}, the A rows {l2_a}, the w_up stage,
    the AN group {out blocks, gamma2} and ONE DOWN GROUP PER COLUMN {its
    w_down stage, its ring, its relay or zero buffer} -- measured: 4 + n_b
    memtiles in the routed design, one per group, which is what lets a
    column's feed hold its two tasks (a memtile channel pair owns 24 BDs;
    see FOOTGUNS). ``n_b`` is capped at NPU2's eight memtiles less four;
    at the layer's width n_b 2 is measured and n_b 4 is refused by
    aie-place-tiles (FOOTGUNS). The A broadcast is ONE memtile MM2S port
    fanning to every up core (measured: one ``aie.flow`` source, n_b
    destinations).

    The bisection instrument: ``stop_after`` in ``STOP_STAGES`` cuts the
    module after the AN1, up or down herd and routes that herd's output to
    ``y`` (see the builder's docstring); ``tail_pipeline_stage_reference``
    is each cut's oracle and ``tail_pipeline_rung.py --stop-after`` runs
    it. It is what located the 2026-08-22 deadlock and the two wrong
    answers behind it, one hop at a time (devq 513-516).

WHAT THE HOST SEES
    ``tail_pipeline_pack_weights(w_up, w_down, ...)`` pre-tiles both
    weights into the flat (sweep, step, column)-major blocked layouts the
    feeds read contiguously; ``tail_pipeline_device_inputs`` builds the
    argument list; ``tail_pipeline_reference`` is the FP32 oracle (one
    rounding at the end, the rule every builder here follows);
    ``compile_tail_pipeline_kernels`` builds the TWO objects the herds
    link: encoder.cc's FFN half at THIS module's tiles and its AddNorm
    half, as separate objects because each exports the other's colliding
    symbols and no core links both (the ffn_resident precedent).

FOOTGUNS
    - ``emb_dim`` must be a multiple of 32 for the NUMBERS to be right:
      ``fused_add_layer_norm_2outs`` steps 32 lanes (``vector_chunks =
      cols / 32``) and then rewinds its row pointer by ``cols``, so at
      emb 48 every pass after the first starts 16 elements BEFORE the row
      and the output is unrelated to any LayerNorm -- measured: the AN1 cut
      at iron's baseline reads corr 0.11 against the true LN AND against a
      "first 32 lanes only" model (devq 516). iron's baseline has emb 48;
      it compiles and routes identically and runs to completion, so it is
      a hang/no-hang datum, never a number. The builder REFUSES it unless
      ``allow_an_lane_truncation=True`` is passed -- the structure check
      passes it, explicitly, for the baseline; nothing numeric may. The
      numeric witness of the baseline GEOMETRY (one band, one sweep, depth
      1, one column) is emb 32 with tile_k 32.
    - THE ONE-TOKEN RULE (the 2026-08-22 deadlock, devq 512/513). An L1
      buffer that is the destination of channel gets AND the source of a
      channel put gets its put BD's lock counts from air-to-aie's
      buffer-level ``getLockValuePair`` -- ``ceil(#gets / #puts)`` tokens
      per fire -- while the core releases ONE token per put: with two gets
      and one put (the first port's tail forwarded ``l1_acc_in``) the put
      BD waited for 2 and the core gave 1, every dispatch, both shapes
      (``aie.use_lock(%lock, AcquireGreaterEqual, %c2_i32)`` on the down
      core's MM2S against a single ``Release, %c1_i32``). And a buffer put
      on TWO channels gets two lock pairs with the second's acquire hoisted
      to its own block start, so the second put's producing write races
      the first channel's DMA read. Hence: every L1 buffer is a channel
      destination OR a channel source, never both; every source feeds one
      channel; a value that must be forwarded is first written by a kernel
      into its own source buffer (the tail's ``l1_red``). The structure
      check pins every core-tile BD lock at one token.
    - THE FEED'S ROUND-ROBIN RULE (the wrong answer under the deadlock,
      devq 514/515). Several gets on one core S2MM channel share ONE lock
      pair whose init is the number of distinct destination buffers, so
      the DMA may run that many transfers ahead of the core; air-to-aie
      also emits one BD per get OP and cycles them in program order. Both
      are only right when the channel's transfers are a strict round-robin
      over its buffers, each op firing once per cycle. Two things broke
      that here and each was a measured wrong answer: the block loop as an
      scf.for (four ops with equal trip counts became one 4-BD cycle while
      the band fired them as two bursts: a w_down chunk landed on a
      block-sized BD, corr 0.31); and the tail's partial in a third buffer
      (init 3 let the second w_down chunk overwrite the first under the
      matmul: y[d0] = C0[d1] + C1[d0], exactly). Hence: d is
      Python-unrolled on the core side too, and the feed is a strict
      (l1_b, l1_acc_in) alternation -- the partial lands in l1_b's prefix,
      the add's first operand is declared at l1_b's type (aiecc passes
      bare pointers), and the chain's last core, which has nothing to fold
      in, has no tail. The structure check pins the feed lock's init at 2.
    - BD-chain order is program order ONLY when every op on a channel has
      the same trip count inside the band. At ``sweeps > 1`` the sweep
      pair and the tail pair differ, air-to-aie emits two terminated tasks
      (``repeat_count`` bands*sweeps-1, then bands-1) and every band's
      sweeps precede every band's tail (measured hermetically), and AIE
      DMA tasks do not loop; the geometry REFUSES ``sweeps > 1 and bands
      > 1``. The route that lands: ONE band per module, the host
      iterating bands over launch arguments (band-serial; the rung's
      ``--band-serial``) -- with one band the two-task program is exactly
      right, and aircc emits a ``*_reset`` PDI reloaded after every run so
      the next dispatch re-arms. Measured: the layer's width, 512 rows as
      32 bands, 3/3 at n_b 1 and 2 (devq 520).
    - THE PER-FILL ENDPOINT COLLAPSE. A memtile buffer filled by ONE op
      and drained by identical reader endpoints gets ONE token per distinct
      endpoint per fill (``getLockValuePair``): the first port's single
      LN-rows buffer, read k_steps times per sweep plus once as the
      residual, released k_steps + 1 tokens per band and deadlocked on the
      second sweep (``S2MM A65x3`` against five reads, read hermetically).
      Hence l2_a[c] is refilled every sweep by AN1 (which re-runs its
      8-row LN per sweep) and the residual goes core->core. And a buffer
      whose two copies were bucketed onto DIFFERENT memtiles had its AN2
      put silently dropped by air-to-aie (the chain's rows 8-15 sentinel,
      devq 519): no channel may source one core port from two memtiles.
    - STANDALONE IDENTICAL L3->L2 FETCHES VANISH. Two Python-unrolled,
      dependency-free fetches of one zero chunk were deduplicated by
      air-fuse-channels and the survivor erased by air-opt-shim-dma-bds:
      the runtime sequence carried no task for that channel and every
      multi-sweep shape hung (devq 519). A refill of the w_down STAGE for
      the zero is fused into the sweep loop with its bound bumped instead,
      and the op-count lock inference then hands the stage three free
      tokens per fill for two reads. The form that survives: a zero SLAB of
      depth chunks in L3, fetched into the column's own L2 buffer by an
      scf.for over d with an IV offset (loop-carried tokens; one writer op,
      depth identical readers = one token per fill, which is right).
    - A memtile loop holding only channel ops is unrolled into a BD chain
      by air-opt-memtile-dma-bds (``AIRUnrollScfForIntoBDChain``, trip
      <= 16): the refill's affine.apply keeps the per-column sweep loop a
      loop, so it lowers to ONE repeat-counted task -- a pure put loop at
      sweeps 4 was already 20 feed BDs.
    - A memtile DMA channel pair owns 24 BDs (aiecc: "'aie.dma_bd' op
      Allocator exhausted available BD IDs (maximum 24 available for
      channel 0)"): a column's feed is ``4*depth`` (sweep task + tail task)
      plus its refill, so ``depth <= 5``; at emb 768 that is tile_k 192
      (depth 4), fewer and larger blocks than iron's k 96/128 rows (depth 8
      and 6, which read 33 and 25). L1 still holds the k-192 down core at
      62464 B.
    - aie-place-tiles at the width: n_b 4 (ten shim MM2S streams) is
      refused with "no ShimNOCTile has sufficient DMA capacity for 1
      input/0 output channels near centroid column 6"; n_b 2 is measured.
    - (Historical) a lone core's zero partials, stored by the core itself,
      had to precede its ring primes: [ring, zero, ring] on the store
      channel is read by air-to-aie's repeating-prefix detection as a
      [ring, zero] cycle -- exact on one band, wrong from the second. The
      zero now comes from L3's slab.
    - (Historical) a single ``scf.index_switch`` per band carried a core's
      whole tail; two in sequence made the second depend on the first's
      token, which ``air-dependency-canonicalize`` refuses ("'scf.index_switch'
      op unknown op type producing async token"). The per-column herds
      have no switch.
    - Every L2/L1 offset is a literal. The block index d and the k' index
      are Python-unrolled at segment scope because they address staged
      buffers; the band and sweep loops are real because they only move
      L3 offsets. Turning d into an scf.for with an IV-dependent L2 offset
      is the frozen-BD miscompile (ffn_accum's wall 3), silent past two
      trips.
    - ONE writer per L2 buffer, everywhere. The LN output is two buffers
      (one per AN1 core) and the accumulator ring is ``down_proj_depth``
      buffers per core, not one buffer written at offsets: a single-slot
      memtile buffer with several S2MM writers is wall 7
      (builders/ffn_resident.py); and a D-deep ring as ONE alloc is one
      lock pair, which deadlocks the core-side priming at depth >= 3 (the
      core's zero buffer cannot be reused until the memtile accepts the
      previous block, which waits on a feed read the core has not issued
      yet). Per-block buffers give the ring iron's per-slot locks.
    - The ring is PRIMED by the down core (zero -> put, once per block per
      band) exactly as iron's core does, so every read of a ring buffer
      follows a write and the accumulate loop is uniform
      (fetch -> matmul_with_acc -> store) from the first sweep. Do not
      "optimize" the first sweep to the init kernel: it removes the ring
      from the structure at sweeps == 1, which is both shapes the check
      pins.
    - The feed put order per (sweep, d) is w_down THEN the accumulator
      block, and per tail step d the partial THEN the final block. The
      transfers differ in size, so a swapped get order is a hang.
    - ``ffn_gelu_bf16``'s operands are ``__restrict``: GeLU keeps separate
      in/out tiles (iron runs it in place; its kernel is not this one).
    - Both gammas are refilled and re-sent per band so every L2 buffer's
      write:read ratio is constant per trip; a buffer written once outside
      the band loop and read inside it starves on the second band.
    - The objects must exist in the CWD when aiecc links:
      ``compile_tail_pipeline_kernels(tile_m, tile_k, tile_n)``. The FFN
      object bakes DIM_M/K/N; one object serves both GEMM herds because
      iron's geometry makes the down projection's output tile width equal
      ``tile_k`` (``K == k * down_proj_depth``), so up (M,K,N) and down
      (M,N,K) are the same -D set.
    - Shim MM2S demand is SIX streams (x, residual, gamma1, w_up, w_down,
      gamma2), all L2-staged and so allocated across shim columns by AIR;
      the structure check counts the per-column result (measured: five
      columns, worst 2). A seventh L3-facing input would be the packet path.
    - L1 is ONE copy per buffer. Every L1 buffer is allocated at herd
      scope, outside every loop, and aircc allocates exactly one
      ``aie.buffer`` per alloc (measured on both pinned shapes: the AN
      cores carry 5, the up cores 3, the down cores 6).
      ``tail_pipeline_l1_bytes`` counts that, not ffn_resident's doubling
      rule, and the structure check pins the per-core buffer set. Moving an
      alloc inside a loop invites the ping-pong rotation and the baseline
      down core (52 KiB single) no longer fits.
    - TWO AIR WALLS, both measured on this module and both routed around:
      (1) ``air-dependency`` refused a ``memref.dealloc`` whose buffer's
      last use sat inside an ``scf.index_switch`` region -- the first
      port's tail role switch -- with "operand #0 does not dominate this
      use" (the dealloc is handed a token defined in the child region);
      the per-column herds have no switch and deallocate everything. (2) ``air-split-l2-memref`` (armed
      once a segment holds more than 4 tiles) asserts -- an unengaged
      ``std::optional`` in ``getTargetMemrefAllocs``, from
      ``getOffsetDimFromMemrefDim`` on a whole-buffer put -- on the first
      port's LN-rows buffer, read by two channels of different access rank
      (the 4-D retile and the 1-D residual relay). Every L2 alloc carries
      ``air.no_split``, the pass's documented opt-out; the split is a
      memtile-spreading optimization, not part of the design.
"""

import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects.arith import ConstantOp
from air.dialects.func import CallOp, FuncOp
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import type_mapper

from builders.addnorm import EPS, addnorm_reference
from builders.ffn_accum import MICRO
from builders.gelu import gelu_tanh_reference

range_ = for_

# encoder.cc's FFN half at THIS module's tiles (both GEMM herds) and its
# AddNorm half (both AN herds). Two objects: each exports symbols the other
# also defines, and no core links both.
FFN_KERNEL_OBJ = "tail_pipeline_ffn.o"
AN_KERNEL_OBJ = "tail_pipeline_an.o"

# The entry points, named literally (the structure check greps for them).
AN_SYMBOL = "fused_add_layer_norm_2outs"
UP_ZERO_SYMBOL = "ffn_zero_bf16_up_proj"
UP_MM_SYMBOL = "ffn_matmul_bf16_bf16_up_proj"
DOWN_ZERO_SYMBOL = "ffn_zero_bf16_down_proj"
DOWN_MM_SYMBOL = "ffn_matmul_with_acc_bf16_bf16_down_proj"
GELU_SYMBOL = "ffn_gelu_bf16"
ADD_SYMBOL = "ffn_eltwise_add_bf16_vector"

# Channel names (see THE DATAFLOW).
CHANNEL_AN1_FEED = "tp_an1_feed"
CHANNEL_AN1_OUT = "tp_an1_out"
CHANNEL_AN1_A = "tp_an1_a"
CHANNEL_A_FEED = "tp_a_feed"
CHANNEL_WUP_FEED = "tp_wup_feed"
CHANNEL_H = "tp_h"
# Per down column t (one herd, one L2 group each; see PORT ARITHMETIC):
# ``tp_down_feed_t``, ``tp_acc_store_t``, and ``tp_reduce_t`` (core t+1's
# reduced block to core t). These are the PREFIXES; ``down_channel`` names.
CHANNEL_DOWN_FEED = "tp_down_feed"
CHANNEL_ACC_STORE = "tp_acc_store"
CHANNEL_REDUCE = "tp_reduce"
CHANNEL_DOWN_OUT = "tp_down_out"


def down_channel(prefix, t):
    """The per-column channel name for ``prefix`` and down column ``t``."""
    return f"{prefix}_{t}"


def down_herd_name(t):
    """Down column ``t``'s herd name (``HERD_DOWN`` is the prefix)."""
    return f"{HERD_DOWN}_{t}"
CHANNEL_AN2_FEED = "tp_an2_feed"

HERD_AN1 = "tp_an1"
HERD_UP = "tp_up"
HERD_DOWN = "tp_down"
HERD_AN2 = "tp_an2"
SEGMENT_NAME = "tail_pipeline_seg"

# iron's ln_cores_per_nA: the band is split in two row halves.
AN_CORES = 2

# The AN kernel's vector width (fused_add_layer_norm_2<bfloat16, 32>); GeLU
# and the eltwise add step 16.
AN_VEC_LEN = 32
FFN_VEC_LEN = 16

# AIE2P core-local memory and the stack aircc reserves inside it.
L1_BYTES = 64 * 1024
L1_STACK_BYTES = 1024
L2_BYTES = 512 * 1024
SHIM_BD_STRIDE_MAX = 1 << 20

# NPU2 has eight memtiles; every down column stages on one of its own (its
# feed's two tasks and its ring fill one channel pair's 24 BDs at depth 4),
# and the AN1 inputs, the A rows, the w_up stage and the AN group take four.
NPU2_MEMTILES = 8
MAX_N_B = NPU2_MEMTILES - 4

# The bisection cuts ``build_tail_pipeline_module(stop_after=...)`` accepts,
# in pipeline order; ``None`` is the whole pipeline.
STOP_STAGES = ("an1", "up", "down", None)


def compile_tail_pipeline_kernels(tile_m=64, tile_k=48, tile_n=96):
    """Build both objects into the CWD, where aiecc's link_with looks.

    The -DDIM_* set MUST match the module's tiles: the FFN object is a
    different kernel at every tile shape and links silently at any of them.
    """
    import shared.infra.external_kernels as ek

    ek.compile_encoder(
        tile_m=tile_m,
        tile_k=tile_k,
        tile_n=tile_n,
        build_ffn=True,
        build_addnorm=False,
        out_name=FFN_KERNEL_OBJ,
    )
    ek.compile_encoder(
        tile_m=tile_m,
        tile_k=tile_k,
        tile_n=tile_n,
        build_ffn=False,
        build_addnorm=True,
        out_name=AN_KERNEL_OBJ,
    )


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


def _linear_map(factors, offset=0):
    """AffineMap for ``sum_i s_i * factors[i] + offset`` over symbols."""
    expr = AffineConstantExpr.get(offset)
    for i, f in enumerate(factors):
        expr = AffineExpr.get_add(
            expr, AffineExpr.get_mul(AffineSymbolExpr.get(i), AffineConstantExpr.get(f))
        )
    return AffineMap.get(0, len(factors), [expr])


def _two_iv_map(factor0, factor1, offset=0):
    """AffineMap for ``s0 * factor0 + s1 * factor1 + offset``."""
    return _linear_map([factor0, factor1], offset)


def _three_iv_map(factor0, factor1, factor2, offset=0):
    """AffineMap for ``s0 * factor0 + s1 * factor1 + s2 * factor2 + offset``."""
    return _linear_map([factor0, factor1, factor2], offset)


def tail_pipeline_geometry(
    seq_len, emb_dim, ffn_dim, tile_m, tile_k, tile_n, down_proj_depth, n_b, n_a=1
):
    """Check the geometry rules and return the derived counts.

    Raises ValueError with the rule that failed. Shared by the builder, the
    packers and the reference so that one set of rules exists.
    """
    if n_a != 1:
        raise NotImplementedError(
            f"n_a={n_a}: only one replica is built. iron replicates the whole "
            "pipeline per nA tile with its own A/R shim streams; a second replica "
            "here would double the shim MM2S demand (6 -> 12 streams) before any "
            "placement question, and is unmeasured."
        )
    if n_b < 1 or n_b > MAX_N_B:
        raise ValueError(
            f"n_b ({n_b}) must be in [1, {MAX_N_B}]: every down column stages on "
            f"a memtile of its own and the other four groups take four of NPU2's "
            f"{NPU2_MEMTILES}"
        )
    if tile_m % MICRO:
        raise ValueError(f"tile_m ({tile_m}) must be a multiple of {MICRO} (MICRO)")
    if tile_m % 16:
        raise ValueError(
            f"tile_m ({tile_m}) must be a multiple of 16: encoder.cc's up-proj "
            "kernel expands 2x in m over 8x8x8 mmuls (DIM_M % 16 static_assert)"
        )
    if tile_m % AN_CORES or (tile_m // AN_CORES) % MICRO:
        raise ValueError(
            f"tile_m ({tile_m}) must split into {AN_CORES} row halves that are "
            f"multiples of {MICRO}: each AN1 core's rows are retiled into whole "
            "microtile rows by the memtile"
        )
    if tile_k % MICRO:
        raise ValueError(f"tile_k ({tile_k}) must be a multiple of {MICRO} (DIM_K % 8)")
    if tile_n % 16:
        raise ValueError(
            f"tile_n ({tile_n}) must be a multiple of 16 (DIM_N % 16 static_assert)"
        )
    if emb_dim % tile_k:
        raise ValueError(f"emb_dim ({emb_dim}) must divide by tile_k ({tile_k})")
    if emb_dim != tile_k * down_proj_depth:
        raise ValueError(
            f"emb_dim ({emb_dim}) must equal tile_k * down_proj_depth "
            f"({tile_k} * {down_proj_depth} = {tile_k * down_proj_depth}): the "
            "accumulator blocks tile the [tile_m, emb_dim] output exactly (iron's "
            "K == k * down_proj_depth)"
        )
    if ffn_dim % (n_b * tile_n):
        raise ValueError(
            f"ffn_dim ({ffn_dim}) must divide by n_b * tile_n ({n_b * tile_n}): "
            "the up herd advances in whole sweeps"
        )
    if seq_len % tile_m:
        raise ValueError(
            f"seq_len ({seq_len}) must divide by tile_m ({tile_m}): the band loop "
            "advances in whole tiles"
        )
    if emb_dim % FFN_VEC_LEN:
        raise ValueError(
            f"emb_dim ({emb_dim}) must be a multiple of {FFN_VEC_LEN}: the eltwise "
            "add and GeLU step 16 lanes and drop the remainder"
        )
    if (tile_m * tile_n) % FFN_VEC_LEN or (tile_m * tile_k) % FFN_VEC_LEN:
        raise ValueError("H and accumulator tiles must be multiples of 16 elements")
    sweeps = ffn_dim // (n_b * tile_n)
    bands = seq_len // tile_m
    if sweeps > 1 and bands > 1:
        raise ValueError(
            f"sweeps ({sweeps}) > 1 with bands ({bands}) > 1: a down core's feed "
            "channel then lowers to two terminated DMA tasks (the sweep pair "
            f"repeated bands*sweeps times, then the tail pair repeated bands "
            "times), which orders every band's sweeps before any band's tail -- "
            "measured hermetically on the routed dump (2026-08-22), and AIE DMA "
            "tasks do not loop. With several sweeps the module is ONE band "
            "(seq_len == tile_m) and the host iterates bands over launch "
            "arguments -- builders/ffn_resident.py's band-serial rule; "
            "tail_pipeline_rung.py --band-serial does it. See FOOTGUNS."
        )
    return {
        "bands": seq_len // tile_m,
        "rpc": tile_m // AN_CORES,
        "sweeps": ffn_dim // (n_b * tile_n),
        "k_steps": emb_dim // tile_k,
        "depth": down_proj_depth,
    }


def tail_pipeline_l1_bytes(emb_dim, tile_m, tile_k, tile_n, n_b, np_dtype=bfloat16):
    """Per-herd L1 need, every buffer listed, ONE copy each.

    Every L1 buffer here is allocated at herd scope, outside every loop, and
    aircc allocates exactly one ``aie.buffer`` per alloc for that shape --
    measured on the routed dump (the structure check counts the buffers per
    core tile against this plan, so a toolchain that starts rotating them
    fails there rather than in the numbers). Returns
    ``{herd: (bytes, [(name, elems), ...])}``.
    """
    it = np.dtype(np_dtype).itemsize
    rpc = tile_m // AN_CORES
    rows = rpc * emb_dim
    an_plan = [
        ("l1_in", rows),
        ("l1_res", rows),
        ("l1_gamma", emb_dim),
        ("l1_out1", rows),
        ("l1_out2", rows),
    ]
    plans = {
        HERD_AN1: an_plan,
        HERD_UP: [
            ("l1_a", tile_m * tile_k),
            ("l1_b", tile_k * tile_n),
            ("l1_c", tile_m * tile_n),
        ],
        HERD_DOWN: [
            ("l1_h", tile_m * tile_n),
            ("l1_g", tile_m * tile_n),
            ("l1_b", tile_n * tile_k),
            ("l1_acc_in", tile_m * tile_k),
            ("l1_acc_out", tile_m * tile_k),
            # The block a core's tail forwards (the partial it folds in lands
            # in l1_b's prefix). The chain's last core has no tail and the
            # compiler drops its unreferenced l1_red: see FOOTGUNS, the
            # one-token rule and the feed's round-robin rule.
            ("l1_red", tile_m * tile_k),
        ],
        HERD_AN2: an_plan,
    }
    return {
        h: (sum(e for _, e in plan) * it + L1_STACK_BYTES, plan)
        for h, plan in plans.items()
    }


@module_builder
def build_tail_pipeline_module(
    seq_len,
    emb_dim,
    ffn_dim,
    tile_m,
    tile_k,
    tile_n,
    down_proj_depth,
    n_b,
    n_a=1,
    np_dtype=bfloat16,
    allow_an_lane_truncation=False,
    stop_after=None,
):
    """Build the one-segment tail pipeline.

    Args:
        seq_len, emb_dim, ffn_dim: ``x, residual, y: [seq_len, emb_dim]``,
            ``w_up: [emb_dim, ffn_dim]``, ``w_down: [ffn_dim, emb_dim]``.
        tile_m: rows per band (iron's ``m``); multiple of 16, halved across
            the two AN cores into multiples of 8.
        tile_k: the up projection's K step AND the down projection's output
            block width (iron's ``k``); ``emb_dim == tile_k * down_proj_depth``.
        tile_n: the H column tile (iron's ``n``); multiple of 16.
        down_proj_depth: accumulator blocks per down core (iron's ring depth).
        n_b: up/down column pairs (iron's ``nB_tiles_distributed``), at most
            ``MAX_N_B``.
        n_a: replicas; only 1 is built.
        np_dtype: bf16, matching every kernel's C linkage.
        allow_an_lane_truncation: admit an ``emb_dim`` that is not a multiple
            of 32. The module then compiles and routes exactly as otherwise
            but the AN kernel normalizes the first ``32 * (emb_dim // 32)``
            columns only (see FOOTGUNS). For structural checks only.
        stop_after: the bisection knob -- ``None`` (the whole pipeline) or
            one of ``STOP_STAGES``: the module keeps every stage up to and
            including that herd and routes THAT herd's output to ``y``;
            later herds, their channels, feeds and L2 buffers are not
            emitted. ``"an1"``: ``y`` is the ``[seq_len, emb_dim]`` h1 rows,
            stored by the AN1 cores herd-direct. ``"up"``: ``y`` is the flat
            ``[seq_len * ffn_dim]`` run of H tiles in (band, sweep, column,
            blocked [tile_m, tile_n]) order, stored by the up cores
            herd-direct (the up->down core edge is replaced by a shim
            store). ``"down"``: ``y`` is the flat ``[seq_len * emb_dim]`` run
            of reduced C blocks in (band, d, blocked [tile_m, tile_k])
            order -- the down herd's protocol is UNCHANGED (its final block
            still leaves on ``tp_down_out`` into ``l2_out[d]``); the segment
            stores each block from L2 to ``y`` in the AN2 feed's place.
            ``tail_pipeline_stage_reference`` computes the matching oracle.
            The function signature is the same for every cut, so one input
            list serves them all.

    Returns:
        air.ir.Module with one function
        ``tail_pipeline(x, residual, gamma1, w_up_packed, w_down_packed,
        gamma2, y)``.
    """
    if stop_after not in STOP_STAGES:
        raise ValueError(f"stop_after must be one of {STOP_STAGES}, not {stop_after!r}")
    emit_up = stop_after != "an1"
    emit_down = stop_after not in ("an1", "up")
    emit_an2 = stop_after is None
    geo = tail_pipeline_geometry(
        seq_len, emb_dim, ffn_dim, tile_m, tile_k, tile_n, down_proj_depth, n_b, n_a
    )
    if emb_dim % AN_VEC_LEN and not allow_an_lane_truncation:
        raise ValueError(
            f"emb_dim ({emb_dim}) is not a multiple of {AN_VEC_LEN}: "
            f"{AN_SYMBOL} steps {AN_VEC_LEN} lanes and silently normalizes only "
            f"the first {AN_VEC_LEN * (emb_dim // AN_VEC_LEN)} columns of each "
            "row. The module would compile and route identically, so pass "
            "allow_an_lane_truncation=True for a STRUCTURAL build only; never "
            "for numbers."
        )
    bands, rpc, sweeps, k_steps, depth = (
        geo["bands"], geo["rpc"], geo["sweeps"], geo["k_steps"], geo["depth"]
    )
    itemsize = np.dtype(np_dtype).itemsize

    # L1, every herd, every buffer, one copy each (measured; see FOOTGUNS).
    for herd_name, (need, plan) in tail_pipeline_l1_bytes(
        emb_dim, tile_m, tile_k, tile_n, n_b, np_dtype
    ).items():
        if need >= L1_BYTES:
            listing = ", ".join(f"{n} {e}" for n, e in plan)
            raise ValueError(
                f"herd {herd_name} needs {need} B of L1 ({listing}; elements, "
                f"plus the {L1_STACK_BYTES}-byte stack), not under the "
                f"{L1_BYTES}-byte tile; lower tile_m, tile_n or tile_k. aiecc "
                "would report this against the aie.tile, far from the cause."
            )
    # L2: every staged buffer, possibly ping-ponged, per memtile GROUP (the
    # buffers bucket by shared channel onto four memtiles; see PORT
    # ARITHMETIC) -- the sum would refuse the layer's width, which fits
    # group by group.
    wup_stage = n_b * tile_k * tile_n
    chunk = tile_n * tile_k  # one w_down chunk, and one relay buffer
    wdown_stage = depth * chunk  # one column's sweep
    blk = tile_m * tile_k
    l2_groups = {
        "AN1 inputs": emb_dim + 2 * tile_m * emb_dim,
        "w_up stage": wup_stage,
        # per down column: its w_down sweep, its ring, its relay or zero chunk
        "down column": wdown_stage + depth * blk + chunk,
        "AN group": emb_dim + AN_CORES * rpc * emb_dim + depth * blk,
    }
    for group, elems in l2_groups.items():
        if 2 * elems * itemsize > L2_BYTES:
            raise ValueError(
                f"the {group}'s staged buffers need {2 * elems * itemsize} B "
                f"ping-ponged, over the {L2_BYTES}-byte memtile"
            )
    # Shim BD strides: the band fetches/stores are [tile_m, emb] row-major
    # (stride emb_dim); the weight refills are contiguous; the gammas 1-D.
    for name, stride in (("band row", emb_dim), ("w_up refill", 1), ("w_down refill", 1)):
        if stride >= SHIM_BD_STRIDE_MAX:
            raise ValueError(
                f"{name} stride {stride} is over the shim aie.dma_bd cap of "
                f"{SHIM_BD_STRIDE_MAX}"
            )

    xrt_dtype = type_mapper(np_dtype)
    l3_act_ty = MemRefType.get([seq_len, emb_dim], xrt_dtype)
    l3_g_ty = MemRefType.get([emb_dim], xrt_dtype)
    l3_wup_ty = MemRefType.get([emb_dim * ffn_dim], xrt_dtype)
    # w_down carries one trailing ZERO slab of depth chunks (see the packer).
    l3_wdown_ty = MemRefType.get([(n_b * sweeps + 1) * wdown_stage], xrt_dtype)
    # y's shape follows the cut (see tail_pipeline_stage_shape).
    l3_y_ty = MemRefType.get(
        list(tail_pipeline_stage_shape(seq_len, emb_dim, ffn_dim, stop_after)), xrt_dtype
    )

    l2_space = IntegerAttr.get(T.i32(), MemorySpace.L2)
    l2_g_ty = MemRefType.get([emb_dim], xrt_dtype, memory_space=l2_space)
    l2_band_ty = MemRefType.get([tile_m * emb_dim], xrt_dtype, memory_space=l2_space)
    l2_a_ty = MemRefType.get([rpc * emb_dim], xrt_dtype, memory_space=l2_space)
    l2_wup_ty = MemRefType.get([wup_stage], xrt_dtype, memory_space=l2_space)
    l2_wdown_ty = MemRefType.get([wdown_stage], xrt_dtype, memory_space=l2_space)
    l2_chunk_ty = MemRefType.get([chunk], xrt_dtype, memory_space=l2_space)
    l2_blk_ty = MemRefType.get([blk], xrt_dtype, memory_space=l2_space)

    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_rows_ty = MemRefType.get([rpc, emb_dim], xrt_dtype, memory_space=l1_space)
    l1_g_ty = MemRefType.get([emb_dim], xrt_dtype, memory_space=l1_space)
    l1_a_ty = MemRefType.get([tile_m, tile_k], xrt_dtype, memory_space=l1_space)
    l1_bup_ty = MemRefType.get([tile_k, tile_n], xrt_dtype, memory_space=l1_space)
    l1_h_ty = MemRefType.get([tile_m, tile_n], xrt_dtype, memory_space=l1_space)
    l1_bdown_ty = MemRefType.get([tile_n, tile_k], xrt_dtype, memory_space=l1_space)
    l1_acc_ty = MemRefType.get([tile_m, tile_k], xrt_dtype, memory_space=l1_space)

    def _extern(symbol, arg_tys, obj):
        f = FuncOp(symbol, (arg_tys, []), visibility="private")
        f.attributes["link_with"] = StringAttr.get(obj)
        f.attributes["llvm.emit_c_interface"] = UnitAttr.get()
        return f

    an_func = _extern(
        AN_SYMBOL,
        [l1_rows_ty, l1_rows_ty, l1_g_ty, l1_rows_ty, l1_rows_ty, T.i32(), T.i32()],
        AN_KERNEL_OBJ,
    )
    up_zero_func = _extern(UP_ZERO_SYMBOL, [l1_h_ty], FFN_KERNEL_OBJ)
    up_mm_func = _extern(UP_MM_SYMBOL, [l1_a_ty, l1_bup_ty, l1_h_ty], FFN_KERNEL_OBJ)
    down_zero_func = _extern(DOWN_ZERO_SYMBOL, [l1_acc_ty], FFN_KERNEL_OBJ)
    down_mm_func = _extern(
        DOWN_MM_SYMBOL, [l1_h_ty, l1_bdown_ty, l1_acc_ty, l1_acc_ty], FFN_KERNEL_OBJ
    )
    gelu_func = _extern(GELU_SYMBOL, [l1_h_ty, l1_h_ty, T.i32()], FFN_KERNEL_OBJ)
    # The add's first operand is the partial, which lands in the w_down
    # slot (l1_b, a [tile_m, tile_k] prefix of it): declared at l1_b's type.
    # aiecc passes external kernels bare pointers, and the kernel reads
    # ``size`` elements from it. See FOOTGUNS, the feed's round-robin rule.
    add_func = _extern(
        ADD_SYMBOL, [l1_bdown_ty, l1_acc_ty, l1_acc_ty, T.i32()], FFN_KERNEL_OBJ
    )

    # Channels: only those the cut uses are declared.
    Channel(CHANNEL_AN1_FEED, size=[AN_CORES, 1])
    if emit_up:
        Channel(CHANNEL_AN1_A, size=[AN_CORES, 1])
    if emit_an2:
        Channel(CHANNEL_AN1_OUT, size=[AN_CORES, 1])
    if emit_up:
        if n_b > 1:
            Channel(CHANNEL_A_FEED, size=[1, 1], broadcast_shape=[n_b, 1])
        else:
            Channel(CHANNEL_A_FEED, size=[1, 1])
        Channel(CHANNEL_WUP_FEED, size=[n_b, 1])
    if emit_down:
        Channel(CHANNEL_H, size=[n_b, 1])
        for t in range(n_b):
            Channel(down_channel(CHANNEL_DOWN_FEED, t), size=[1, 1])
            Channel(down_channel(CHANNEL_ACC_STORE, t), size=[1, 1])
        for t in range(n_b - 1):
            # Core t+1's reduced block to core t, through l2_red[t].
            Channel(down_channel(CHANNEL_REDUCE, t), size=[1, 1])
        Channel(CHANNEL_DOWN_OUT, size=[1, 1])
    if emit_an2:
        Channel(CHANNEL_AN2_FEED, size=[AN_CORES, 1])

    # The 4-D retile of AN1 core c's row-major [rpc, emb] rows into the
    # blocked [rpc, tile_k] half-tile for k' step kb: stream order (mi, ki,
    # ri, si) reads row mi*8+ri, column kb*tile_k + ki*8 + si.
    retile_sizes = [rpc // MICRO, tile_k // MICRO, MICRO, MICRO]
    retile_strides = [MICRO * emb_dim, MICRO, emb_dim, 1]
    # The 4-D un-tile of half c of a blocked [tile_m, tile_k] block into
    # row-major order: stream order (mi, ri, ki, si) reads microtile
    # (c*rpc/8 + mi, ki), row ri, column si.
    untile_sizes = [rpc // MICRO, MICRO, tile_k // MICRO, MICRO]
    untile_strides = [tile_k * MICRO, MICRO, MICRO * MICRO, 1]

    # The herd-direct row store AN1 (at the "an1" cut) and AN2 share.
    row_map = _two_iv_map(tile_m, rpc)
    # The "up" cut's flat H store: (band, sweep, tx) -> tile offset.
    h_tile = tile_m * tile_n
    h_off_map = _three_iv_map(sweeps * n_b * h_tile, n_b * h_tile, h_tile)
    # The "down" cut's flat block store: band -> band offset; d is literal.
    c_off_maps = [_mul_add_map(depth * blk, d * blk) for d in range(depth)]

    @FuncOp.from_py_func(
        l3_act_ty, l3_act_ty, l3_g_ty, l3_wup_ty, l3_wdown_ty, l3_g_ty, l3_y_ty
    )
    def tail_pipeline(arg0, arg1, arg2, arg3, arg4, arg5, arg6):

        @launch(operands=[arg0, arg1, arg2, arg3, arg4, arg5, arg6])
        def tp_launch(l_x, l_res, l_g1, l_wup, l_wdown, l_g2, l_y):

            @segment(
                name=SEGMENT_NAME,
                operands=[l_x, l_res, l_g1, l_wup, l_wdown, l_g2, l_y],
            )
            def tp_seg(s_x, s_res, s_g1, s_wup, s_wdown, s_g2, s_y):
                def _l2(ty):
                    # Every staged buffer opts out of air-split-l2-memref.
                    # That pass tiles an L2 buffer across memtiles when one
                    # side has several channels, and it asserts (an
                    # unengaged std::optional in getTargetMemrefAllocs) on
                    # the LN-rows buffer, read by two channels with different
                    # access ranks; the split is an optimization, not part
                    # of the design, so it is declined everywhere.
                    a = AllocOp(ty, [], [])
                    a.attributes["air.no_split"] = UnitAttr.get()
                    return a

                l2_g1 = _l2(l2_g_ty)
                l2_x = _l2(l2_band_ty)
                l2_r = _l2(l2_band_ty)
                # AN1's rows for the A feed: l2_a[c], refilled every sweep
                # and drained k_steps times per fill. AN2's residual copy
                # goes AN1 core c -> AN2 core c directly (tp_an1_out is
                # core->core): one buffer for both roles would be filled
                # once and drained sweeps*k_steps + 1 times, which
                # air-to-aie's per-fill endpoint collapse caps at k_steps + 1
                # tokens, and a separate L2 copy per core was bucketed onto a
                # memtile of its own with its AN2 put silently dropped
                # (FOOTGUNS).
                l2_a = [_l2(l2_a_ty) for _ in range(AN_CORES)] if emit_up else []
                l2_wup = _l2(l2_wup_ty) if emit_up else None
                # Per down column t: its w_down sweep stage, its ring (one
                # buffer per block: one lock pair each), and -- for every
                # column but the last -- the relay buffer core t+1's reduced
                # block lands in, CHUNK-sized so the tail's transfer is the
                # same BD as a w_down chunk (see FOOTGUNS, the feed's
                # round-robin rule).
                l2_wdown = [_l2(l2_wdown_ty) for _ in range(n_b)] if emit_down else []
                l2_ring = (
                    [[_l2(l2_blk_ty) for _ in range(depth)] for _ in range(n_b)]
                    if emit_down
                    else []
                )
                l2_red = [_l2(l2_chunk_ty) for _ in range(n_b - 1)] if emit_down else []
                # The chain's last column's tail b-slot: a ZERO chunk from
                # L3, its own L2 buffer (a refill of the stage itself is fused
                # into the sweep loop by air-fuse-channels; see the memtile).
                l2_bzero = _l2(l2_chunk_ty) if emit_down else None
                l2_out = [_l2(l2_blk_ty) for _ in range(depth)] if emit_down else []
                l2_g2 = _l2(l2_g_ty) if emit_an2 else None

                # ---------------- AN1 herd ----------------
                def an1_body(tx, ty, h_y):
                    l1_x = AllocOp(l1_rows_ty, [], [])
                    l1_r = AllocOp(l1_rows_ty, [], [])
                    l1_g = AllocOp(l1_g_ty, [], [])
                    l1_o1 = AllocOp(l1_rows_ty, [], [])
                    l1_o2 = AllocOp(l1_rows_ty, [], [])
                    cols_i32 = ConstantOp(T.i32(), emb_dim)
                    rows_i32 = ConstantOp(T.i32(), rpc)
                    for band in range_(0, bands):
                        ChannelGet(CHANNEL_AN1_FEED, l1_g, indices=[tx, ty])
                        ChannelGet(CHANNEL_AN1_FEED, l1_x, indices=[tx, ty])
                        ChannelGet(CHANNEL_AN1_FEED, l1_r, indices=[tx, ty])
                        if h_y is None:
                            # The kernel writes its two identical outputs;
                            # each leaves on ONE channel (the one-token
                            # rule): o2 to the A feed once per SWEEP, o1 to
                            # AN2's residual once per band. The LN is
                            # re-run inside the sweep loop so every put's
                            # producing write sits under its own acquire
                            # (a write outside the loop would race the
                            # previous band's last send); it costs
                            # sweeps x an 8-row LN, nothing next to the
                            # down herd's sweep.
                            for _s in range_(0, sweeps):
                                CallOp(
                                    an_func,
                                    [l1_x, l1_r, l1_g, l1_o1, l1_o2, cols_i32, rows_i32],
                                )
                                ChannelPut(CHANNEL_AN1_A, l1_o2, indices=[tx, ty])
                                yield_([])
                            if emit_an2:
                                ChannelPut(CHANNEL_AN1_OUT, l1_o1, indices=[tx, ty])
                        else:
                            CallOp(
                                an_func,
                                [l1_x, l1_r, l1_g, l1_o1, l1_o2, cols_i32, rows_i32],
                            )
                            # The "an1" cut: h1 rows straight to y.
                            row = affine_apply(row_map, [band, tx])
                            dma_memcpy_nd(
                                h_y,
                                l1_o1,
                                dst_offsets=[row, 0],
                                dst_sizes=[rpc, emb_dim],
                                dst_strides=[emb_dim, 1],
                            )
                        yield_([])
                    DeallocOp(l1_x)
                    DeallocOp(l1_r)
                    DeallocOp(l1_g)
                    DeallocOp(l1_o1)
                    DeallocOp(l1_o2)

                if emit_up:

                    @herd(name=HERD_AN1, sizes=[AN_CORES, 1])
                    def an1_herd(tx, ty, _sx, _sy):
                        an1_body(tx, ty, None)

                else:

                    @herd(name=HERD_AN1, sizes=[AN_CORES, 1], operands=[s_y])
                    def an1_herd(tx, ty, _sx, _sy, h_y):
                        an1_body(tx, ty, h_y)

                an1_herd.attributes["link_with"] = StringAttr.get(AN_KERNEL_OBJ)

                # ---------------- UP herd ----------------
                def up_body(tx, ty, h_y):
                    l1_a = AllocOp(l1_a_ty, [], [])
                    l1_b = AllocOp(l1_bup_ty, [], [])
                    l1_c = AllocOp(l1_h_ty, [], [])
                    for band in range_(0, bands):
                        for s in range_(0, sweeps):
                            CallOp(up_zero_func, [l1_c])
                            for _k in range_(0, k_steps):
                                # The two AN1 halves land in the two halves
                                # of the blocked A tile: literal offsets.
                                for c in range(AN_CORES):
                                    ChannelGet(
                                        CHANNEL_A_FEED,
                                        l1_a,
                                        offsets=[c * rpc, 0],
                                        sizes=[rpc, tile_k],
                                        strides=[tile_k, 1],
                                        indices=[tx, ty],
                                    )
                                ChannelGet(CHANNEL_WUP_FEED, l1_b, indices=[tx, ty])
                                CallOp(up_mm_func, [l1_a, l1_b, l1_c])
                                yield_([])
                            if h_y is None:
                                ChannelPut(CHANNEL_H, l1_c, indices=[tx, ty])
                            else:
                                # The "up" cut: the H tile straight to y, in
                                # (band, sweep, column) order.
                                off = affine_apply(h_off_map, [band, s, tx])
                                dma_memcpy_nd(
                                    h_y,
                                    l1_c,
                                    dst_offsets=[off],
                                    dst_sizes=[h_tile],
                                    dst_strides=[1],
                                )
                            yield_([])
                        yield_([])
                    DeallocOp(l1_a)
                    DeallocOp(l1_b)
                    DeallocOp(l1_c)

                if emit_down:

                    @herd(name=HERD_UP, sizes=[n_b, 1])
                    def up_herd(tx, ty, _sx, _sy):
                        up_body(tx, ty, None)

                elif emit_up:

                    @herd(name=HERD_UP, sizes=[n_b, 1], operands=[s_y])
                    def up_herd(tx, ty, _sx, _sy, h_y):
                        up_body(tx, ty, h_y)

                if emit_up:
                    up_herd.attributes["link_with"] = StringAttr.get(FFN_KERNEL_OBJ)

                # ---------------- DOWN herds, one per column ----------------
                def down_body(t, tx, ty):
                    """Down column t: its own herd, channels and L2 group."""
                    feed = down_channel(CHANNEL_DOWN_FEED, t)
                    store = down_channel(CHANNEL_ACC_STORE, t)
                    out_channel = CHANNEL_DOWN_OUT if t == 0 else down_channel(CHANNEL_REDUCE, t - 1)
                    last = t == n_b - 1
                    l1_h = AllocOp(l1_h_ty, [], [])
                    l1_g = AllocOp(l1_h_ty, [], [])
                    l1_b = AllocOp(l1_bdown_ty, [], [])
                    l1_acc_in = AllocOp(l1_acc_ty, [], [])
                    l1_acc_out = AllocOp(l1_acc_ty, [], [])
                    # The block this core forwards (its tail's output). Each
                    # L1 buffer is either a channel destination or a channel
                    # source, never both, and every source feeds ONE channel
                    # (the one-token rule; FOOTGUNS).
                    l1_red = AllocOp(l1_acc_ty, [], [])
                    h_elems_i32 = ConstantOp(T.i32(), tile_m * tile_n)
                    blk_i32 = ConstantOp(T.i32(), blk)

                    # The feed is a strict (l1_b, l1_acc_in) alternation from
                    # the first transfer of a band to the last, and EVERY get
                    # is the same whole-buffer BD as the sweep's: air-to-aie
                    # then folds the band's feed into one two-BD cycle,
                    # whatever depth and sweeps (the d-unrolled chain of the
                    # first port crossed the core's 16 BDs at depth 6:
                    # "'aie.mem' op has more than 16 blocks").
                    for _band in range_(0, bands):
                        # Prime the ring, block by block, as iron's core does.
                        for _d in range(depth):
                            CallOp(down_zero_func, [l1_acc_out])
                            ChannelPut(store, l1_acc_out, indices=[tx, ty])
                        for _s in range_(0, sweeps):
                            ChannelGet(CHANNEL_H, l1_h, indices=[t, 0])
                            CallOp(gelu_func, [l1_h, l1_g, h_elems_i32])
                            for _d in range(depth):
                                # w_down chunk THEN the accumulator block.
                                ChannelGet(feed, l1_b, indices=[tx, ty])
                                ChannelGet(feed, l1_acc_in, indices=[tx, ty])
                                CallOp(down_mm_func, [l1_g, l1_b, l1_acc_in, l1_acc_out])
                                ChannelPut(store, l1_acc_out, indices=[tx, ty])
                            yield_([])
                        # The reduction tail, per block d: into the b-slot
                        # the ZERO w_down chunk (the chain's last core: its
                        # final block leaves through the accumulate kernel,
                        # final + gelu(H) @ 0) or core t+1's reduced block
                        # (every other core: the eltwise add); into the
                        # acc-slot the final block; the result out.
                        for _d in range(depth):
                            ChannelGet(feed, l1_b, indices=[tx, ty])
                            ChannelGet(feed, l1_acc_in, indices=[tx, ty])
                            if last:
                                CallOp(down_mm_func, [l1_g, l1_b, l1_acc_in, l1_red])
                            else:
                                CallOp(add_func, [l1_b, l1_acc_in, l1_red, blk_i32])
                            ChannelPut(out_channel, l1_red, indices=[0, 0])
                        yield_([])
                    for buf in (l1_h, l1_g, l1_b, l1_acc_in, l1_acc_out, l1_red):
                        DeallocOp(buf)

                if emit_down:
                    for t in range(n_b):

                        @herd(name=down_herd_name(t), sizes=[1, 1])
                        def down_herd(tx, ty, _sx, _sy, t=t):
                            down_body(t, tx, ty)

                        down_herd.attributes["link_with"] = StringAttr.get(FFN_KERNEL_OBJ)

                # ---------------- AN2 herd ----------------
                def an2_body(tx, ty, h_y):
                    l1_in = AllocOp(l1_rows_ty, [], [])
                    l1_r = AllocOp(l1_rows_ty, [], [])
                    l1_g = AllocOp(l1_g_ty, [], [])
                    l1_o1 = AllocOp(l1_rows_ty, [], [])
                    l1_o2 = AllocOp(l1_rows_ty, [], [])
                    cols_i32 = ConstantOp(T.i32(), emb_dim)
                    rows_i32 = ConstantOp(T.i32(), rpc)
                    for band in range_(0, bands):
                        ChannelGet(CHANNEL_AN2_FEED, l1_g, indices=[tx, ty])
                        # The residual: AN1 core c's rows, core -> core.
                        ChannelGet(CHANNEL_AN1_OUT, l1_r, indices=[tx, ty])
                        for d in range(depth):
                            # Block d arrives row-major (the memtile un-tiled
                            # it) into column block d: literal offsets.
                            ChannelGet(
                                CHANNEL_AN2_FEED,
                                l1_in,
                                offsets=[0, d * tile_k],
                                sizes=[rpc, tile_k],
                                strides=[emb_dim, 1],
                                indices=[tx, ty],
                            )
                        CallOp(
                            an_func,
                            [l1_in, l1_r, l1_g, l1_o1, l1_o2, cols_i32, rows_i32],
                        )
                        row = affine_apply(row_map, [band, tx])
                        dma_memcpy_nd(
                            h_y,
                            l1_o1,
                            dst_offsets=[row, 0],
                            dst_sizes=[rpc, emb_dim],
                            dst_strides=[emb_dim, 1],
                        )
                        yield_([])
                    DeallocOp(l1_in)
                    DeallocOp(l1_r)
                    DeallocOp(l1_g)
                    DeallocOp(l1_o1)
                    DeallocOp(l1_o2)

                if emit_an2:

                    @herd(name=HERD_AN2, sizes=[AN_CORES, 1], operands=[s_y])
                    def an2_herd(tx, ty, _sx, _sy, h_y):
                        an2_body(tx, ty, h_y)

                    an2_herd.attributes["link_with"] = StringAttr.get(AN_KERNEL_OBJ)

                # ---------------- the memtile programs ----------------
                # Emitted after the herds (the structure check brace-matches
                # herds by name). Every L2-side offset below is a literal.
                band_row_map = _mul_add_map(tile_m)
                wup_off_map = _two_iv_map(k_steps * wup_stage, wup_stage)
                # Column t's sweep s is slab (t * sweeps + s); the zero slab
                # trails every column's, its chunk d at d * chunk.
                wdown_off_maps = [
                    _mul_add_map(wdown_stage, t * sweeps * wdown_stage) for t in range(n_b)
                ]
                zero_off_map = _mul_add_map(chunk, n_b * sweeps * wdown_stage)

                for band in range_(0, bands):
                    band_row = affine_apply(band_row_map, [band])
                    # AN1 inputs: gamma1, x band, residual band.
                    dma_memcpy_nd(l2_g1, s_g1)
                    dma_memcpy_nd(
                        l2_x,
                        s_x,
                        src_offsets=[band_row, 0],
                        src_sizes=[tile_m, emb_dim],
                        src_strides=[emb_dim, 1],
                    )
                    dma_memcpy_nd(
                        l2_r,
                        s_res,
                        src_offsets=[band_row, 0],
                        src_sizes=[tile_m, emb_dim],
                        src_strides=[emb_dim, 1],
                    )
                    for c in range(AN_CORES):
                        ChannelPut(CHANNEL_AN1_FEED, l2_g1, indices=[c, 0])
                        for src in (l2_x, l2_r):
                            ChannelPut(
                                CHANNEL_AN1_FEED,
                                src,
                                offsets=[c * rpc * emb_dim],
                                sizes=[rpc * emb_dim],
                                strides=[1],
                                indices=[c, 0],
                            )
                    if emit_up:
                        # The A feed: per sweep the rows land in l2_a[c];
                        # per k' the two halves, retiled, broadcast to every
                        # up core. k' is Python-unrolled: its offset
                        # addresses a staged buffer.
                        for _s in range_(0, sweeps):
                            for c in range(AN_CORES):
                                ChannelGet(CHANNEL_AN1_A, l2_a[c], indices=[c, 0])
                            for kb in range(k_steps):
                                for c in range(AN_CORES):
                                    ChannelPut(
                                        CHANNEL_A_FEED,
                                        l2_a[c],
                                        offsets=[0, kb * (tile_k // MICRO), 0, 0],
                                        sizes=retile_sizes,
                                        strides=retile_strides,
                                    )
                            yield_([])

                        # w_up: refill per (sweep, k'), L3 offsets only;
                        # each up core's slice at a tx-literal offset.
                        for s in range_(0, sweeps):
                            for k in range_(0, k_steps):
                                w_off = affine_apply(wup_off_map, [s, k])
                                dma_memcpy_nd(
                                    l2_wup,
                                    s_wup,
                                    src_offsets=[w_off],
                                    src_sizes=[wup_stage],
                                    src_strides=[1],
                                )
                                for t in range(n_b):
                                    ChannelPut(
                                        CHANNEL_WUP_FEED,
                                        l2_wup,
                                        offsets=[t * tile_k * tile_n],
                                        sizes=[tile_k * tile_n],
                                        strides=[1],
                                        indices=[t, 0],
                                    )
                                yield_([])
                            yield_([])

                    if emit_down:
                        for t in range(n_b):
                            feed = down_channel(CHANNEL_DOWN_FEED, t)
                            store = down_channel(CHANNEL_ACC_STORE, t)
                            last = t == n_b - 1
                            # The ring priming stores.
                            for d in range(depth):
                                ChannelGet(store, l2_ring[t][d], indices=[0, 0])
                            # Per sweep: refill column t's w_down chunks from
                            # its slab (L3 offset on the sweep IV), then per
                            # block d: the chunk, ring block d; and take the
                            # store back.
                            # Per sweep: refill column t's w_down chunks from
                            # its slab (L3 offset on the sweep IV), then per
                            # block d: the chunk, ring block d; and take the
                            # store back. The refill's affine.apply keeps
                            # this loop a LOOP: a memtile loop of channel ops
                            # alone is unrolled into a BD chain by
                            # air-opt-memtile-dma-bds (trip <= 16), which at
                            # sweeps 4 was already 20 feed BDs; as a loop it
                            # lowers to one repeat-counted task.
                            for s_iv in range_(0, sweeps):
                                wd_off = affine_apply(wdown_off_maps[t], [s_iv])
                                dma_memcpy_nd(
                                    l2_wdown[t],
                                    s_wdown,
                                    src_offsets=[wd_off],
                                    src_sizes=[wdown_stage],
                                    src_strides=[1],
                                )
                                for d in range(depth):
                                    ChannelPut(
                                        feed,
                                        l2_wdown[t],
                                        offsets=[d * chunk],
                                        sizes=[chunk],
                                        strides=[1],
                                        indices=[0, 0],
                                    )
                                    ChannelPut(feed, l2_ring[t][d], indices=[0, 0])
                                    ChannelGet(store, l2_ring[t][d], indices=[0, 0])
                                yield_([])
                            # The tail. The last column's b-slot takes a ZERO
                            # chunk per block, fetched from L3's zero slab into
                            # its own L2 buffer by an scf.for over d with an
                            # IV offset: one writer op with depth identical
                            # reader endpoints is one token per fill, which
                            # is right (each fill is read once). Two
                            # Python-unrolled fetches of ONE zero chunk were
                            # IDENTICAL dependency-free puts: air-fuse-channels
                            # deduped them and air-opt-shim-dma-bds erased the
                            # survivor -- the runtime sequence then had no
                            # task for that channel and the down herd hung
                            # (devq 519, sentinel 1.0 at every multi-sweep
                            # shape). A refill of the STAGE itself for the
                            # zero is fused into the sweep loop instead. Every
                            # other column takes core t+1's reduced block into
                            # its relay buffer and feeds that, chunk-sized, in
                            # the chunk's place.
                            if last:
                                for d_iv in range_(0, depth):
                                    z_off = affine_apply(zero_off_map, [d_iv])
                                    dma_memcpy_nd(
                                        l2_bzero,
                                        s_wdown,
                                        src_offsets=[z_off],
                                        src_sizes=[chunk],
                                        src_strides=[1],
                                    )
                                    yield_([])
                            for d in range(depth):
                                if last:
                                    ChannelPut(feed, l2_bzero, indices=[0, 0])
                                else:
                                    ChannelGet(
                                        down_channel(CHANNEL_REDUCE, t),
                                        l2_red[t],
                                        offsets=[0],
                                        sizes=[blk],
                                        strides=[1],
                                        indices=[0, 0],
                                    )
                                    ChannelPut(feed, l2_red[t], indices=[0, 0])
                                ChannelPut(feed, l2_ring[t][d], indices=[0, 0])
                        for d in range(depth):
                            ChannelGet(CHANNEL_DOWN_OUT, l2_out[d], indices=[0, 0])
                            if not emit_an2:
                                c_off = affine_apply(c_off_maps[d], [band])
                                dma_memcpy_nd(
                                    s_y,
                                    l2_out[d],
                                    dst_offsets=[c_off],
                                    dst_sizes=[blk],
                                    dst_strides=[1],
                                )

                    if emit_an2:
                        # AN2 inputs: gamma2, the residual rows (AN1's
                        # output), then every block un-tiled into its column
                        # block.
                        dma_memcpy_nd(l2_g2, s_g2)
                        for c in range(AN_CORES):
                            ChannelPut(CHANNEL_AN2_FEED, l2_g2, indices=[c, 0])
                            for d in range(depth):
                                ChannelPut(
                                    CHANNEL_AN2_FEED,
                                    l2_out[d],
                                    offsets=[c * (rpc // MICRO), 0, 0, 0],
                                    sizes=untile_sizes,
                                    strides=untile_strides,
                                    indices=[c, 0],
                                )
                    yield_([])

                for buf in (
                    [l2_g1, l2_g2, l2_x, l2_r, l2_wup]
                    + l2_a
                    + l2_wdown
                    + [b for ring in l2_ring for b in ring]
                    + l2_red
                    + ([l2_bzero] if l2_bzero is not None else [])
                    + l2_out
                ):
                    if buf is not None:
                        DeallocOp(buf)


def tail_pipeline_pack_w_up(w_up, tile_k, tile_n, n_b):
    """Pre-tile ``w_up``: flat, (sweep, k'-step, column)-major, blocked.

    Element ``w_up[k, n]`` lands in the ``[tile_k, tile_n]`` block for
    ``(s, k', t)`` where ``n`` falls in group ``g = s * n_b + t`` and ``k`` in
    k'-step ``k // tile_k``; within a block the kernels' blocked layout (a
    row-major grid of row-major 8x8 microtiles). One (s, k') refill -- all
    ``n_b`` columns' slices -- is one contiguous L3 run.
    """
    emb_dim, ffn_dim = w_up.shape
    sweeps = ffn_dim // (n_b * tile_n)
    return np.ascontiguousarray(
        w_up.reshape(
            emb_dim // tile_k, tile_k // MICRO, MICRO, sweeps, n_b, tile_n // MICRO, MICRO
        )
        .transpose(3, 0, 4, 1, 5, 2, 6)
        .reshape(-1)
    )


def tail_pipeline_pack_w_down(w_down, tile_k, tile_n, n_b, down_proj_depth):
    """Pre-tile ``w_down``: flat, (column, sweep, block)-major, blocked, plus
    one trailing ZERO slab.

    Element ``w_down[n, k]`` lands in the ``[tile_n, tile_k]`` block for
    ``(t, s, d)`` where ``n`` falls in group ``g = s * n_b + t`` and ``k`` in
    block ``d = k // tile_k``. Column t's sweep s is one contiguous L3 slab
    (``depth`` chunks) at slab index ``t * sweeps + s``; one ZERO slab of
    ``depth`` chunks follows the last -- the chain's last column fetches its
    chunks for its tail (``final + gelu(H) @ 0``), the uniform tail that
    keeps every feed a two-BD cycle (builder FOOTGUNS).
    """
    ffn_dim, emb_dim = w_down.shape
    sweeps = ffn_dim // (n_b * tile_n)
    blocked = (
        w_down.reshape(
            sweeps, n_b, tile_n // MICRO, MICRO, down_proj_depth, tile_k // MICRO, MICRO
        )
        .transpose(1, 0, 4, 2, 5, 3, 6)  # (t, s, d, ni, ki, ri, ci)
        .reshape(-1)
    )
    return np.ascontiguousarray(
        np.concatenate(
            [blocked, np.zeros(down_proj_depth * tile_n * tile_k, dtype=w_down.dtype)]
        )
    )


def tail_pipeline_pack_weights(w_up, w_down, tile_k, tile_n, n_b, down_proj_depth):
    """Both packed weights, in signature order."""
    return (
        tail_pipeline_pack_w_up(w_up, tile_k, tile_n, n_b),
        tail_pipeline_pack_w_down(w_down, tile_k, tile_n, n_b, down_proj_depth),
    )


def tail_pipeline_device_inputs(
    x, residual, gamma1, w_up, w_down, gamma2, tile_k, tile_n, n_b, down_proj_depth
):
    """Every argument the host fills, in signature order (``y`` excluded).

    ``x`` and ``residual`` stay row-major; the weights are pre-tiled. The
    packing parameters MUST match the ``build_tail_pipeline_module`` call.
    """
    if x.shape != residual.shape:
        raise ValueError(f"x {x.shape} and residual {residual.shape} must match")
    w_up_p, w_down_p = tail_pipeline_pack_weights(
        w_up, w_down, tile_k, tile_n, n_b, down_proj_depth
    )
    return [
        np.ascontiguousarray(x),
        np.ascontiguousarray(residual),
        np.ascontiguousarray(gamma1),
        w_up_p,
        w_down_p,
        np.ascontiguousarray(gamma2),
    ]


def tail_pipeline_reference(x, residual, gamma1, w_up, w_down, gamma2):
    """FP32 reference, rounded once at the end.

    ``y = gamma2 * LN(gelu_tanh(h1 @ w_up) @ w_down) + h1`` with
    ``h1 = gamma1 * LN(x) + residual`` -- both add-norms the POST-ADD form
    (statistics over the input alone, the residual added after), which is
    what ``fused_add_layer_norm_2outs`` computes; ``addnorm_reference`` is
    its oracle and is reused rather than re-derived. FP32 end to end: the
    device's bf16 staging at every hop (h1, H, the per-sweep ring rounding,
    the reduction) is measured as error, not reproduced.
    """
    f32 = np.float32
    h1 = addnorm_reference(x.astype(f32), residual.astype(f32), gamma1.astype(f32), eps=EPS)
    h = gelu_tanh_reference(h1.astype(f32) @ w_up.astype(f32))
    c = h.astype(f32) @ w_down.astype(f32)
    y = addnorm_reference(c, h1.astype(f32), gamma2.astype(f32), eps=EPS)
    return y.astype(x.dtype)


def tail_pipeline_stage_shape(seq_len, emb_dim, ffn_dim, stop_after=None):
    """``y``'s shape for a cut: what ``build_tail_pipeline_module`` declares."""
    if stop_after not in STOP_STAGES:
        raise ValueError(f"stop_after must be one of {STOP_STAGES}, not {stop_after!r}")
    if stop_after == "up":
        return (seq_len * ffn_dim,)
    if stop_after == "down":
        return (seq_len * emb_dim,)
    return (seq_len, emb_dim)


def tail_pipeline_stage_reference(
    x, residual, gamma1, w_up, w_down, gamma2, tile_m, tile_k, tile_n, n_b,
    down_proj_depth, stop_after=None,
):
    """The oracle for a cut, in the layout the device writes ``y`` in.

    ``"an1"``: h1 rows ``[seq_len, emb_dim]``. ``"up"``: the flat run of
    H = h1 @ w_up tiles (BEFORE GeLU: that is what leaves the up core) in
    (band, sweep, column, blocked [tile_m, tile_n]) order. ``"down"``: the
    flat run of C = gelu(H) @ w_down blocks in (band, d, blocked [tile_m,
    tile_k]) order. ``None``: ``tail_pipeline_reference``. FP32 throughout,
    one rounding at the end, as the full reference.
    """
    if stop_after is None:
        return tail_pipeline_reference(x, residual, gamma1, w_up, w_down, gamma2)
    f32 = np.float32
    seq_len, emb_dim = x.shape
    ffn_dim = w_up.shape[1]
    bands = seq_len // tile_m
    h1 = addnorm_reference(x.astype(f32), residual.astype(f32), gamma1.astype(f32), eps=EPS)
    if stop_after == "an1":
        return h1.astype(x.dtype)
    h = h1.astype(f32) @ w_up.astype(f32)
    if stop_after == "up":
        sweeps = ffn_dim // (n_b * tile_n)
        # (band, mi, ri, s, t, ni, si) -> (band, s, t, mi, ni, ri, si)
        packed = (
            h.reshape(bands, tile_m // MICRO, MICRO, sweeps, n_b, tile_n // MICRO, MICRO)
            .transpose(0, 3, 4, 1, 5, 2, 6)
            .reshape(-1)
        )
        return np.ascontiguousarray(packed).astype(x.dtype)
    c = gelu_tanh_reference(h).astype(f32) @ w_down.astype(f32)
    # (band, mi, ri, d, ki, si) -> (band, d, mi, ki, ri, si)
    packed = (
        c.reshape(bands, tile_m // MICRO, MICRO, down_proj_depth, tile_k // MICRO, MICRO)
        .transpose(0, 3, 1, 4, 2, 5)
        .reshape(-1)
    )
    return np.ascontiguousarray(packed).astype(x.dtype)
