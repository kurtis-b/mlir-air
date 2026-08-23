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
    AN1 ->[tp_an1_out]-> memtile   l2_ln[c], ONE writer each: the LN rows,
        row-major, which serve BOTH as the up herd's A operand and as the
        AN2 herd's residual
    memtile ->[tp_a_feed]-> UP herd [n_b,1]     per k' step, a 4-D memtile
        put RETILES the row-major rows into the kernels' blocked
        [tile_m, tile_k] operand (two puts, one per AN1 core, landing in
        the two halves of one L1 tile), BROADCAST to every up core
    memtile ->[tp_wup_feed]-> UP herd          w_up group slices, L3->L2
        refill per (sweep, k'), tx-literal L2 offsets
    UP ->[tp_h]-> DOWN herd [n_b,1]            core->core, L1->L1: the
        finished [tile_m, tile_n] H tile, one per sweep
    memtile ->[tp_down_feed]-> DOWN herd       ONE channel per down core
        carrying, per (sweep, block d): the w_down chunk, then accumulator
        block d (its ``curr_acc_c``); after the sweeps, per d: the final
        block d, then the neighbour's partial (``buffer_to_reduce``)
    DOWN ->[tp_acc_store]-> memtile            accumulator block d back to
        its ring buffer (``new_acc_c``), one L2 buffer PER BLOCK per core
    DOWN ->[tp_reduce]-> memtile ->(feed)-> DOWN   core tx+1's reduced
        partial to core tx, relayed through L2 (see PORT ARITHMETIC)
    DOWN core 0 ->[tp_down_out]-> memtile      the final reduced block d
    memtile ->[tp_an2_feed]-> AN2 herd [2,1]   gamma2; h1 rows (residual,
        from l2_ln[c]); then per d a 4-D memtile put UN-TILES half c of
        the blocked block into column block d of the core's row-major
        input
    AN2 -> shim                                y rows, herd-direct store

    ITERATION. The band loop (``seq_len / tile_m`` trips) is INSIDE the
    module as in iron (``ln_iters_per_core``); every L3 offset advances on
    the band and sweep IVs, and every L2/L1-side offset in the design is a
    compile-time literal (the block index ``d`` and the k' index are
    Python-unrolled wherever they address a staged buffer). That is the
    frozen-BD rule of builders/ffn_accum.py and builders/ffn_resident.py,
    obeyed by construction; see FOOTGUNS.

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
    - the reduction therefore relays through L2 (core tx+1 -> l2_red[tx]
      -> core tx's feed) instead of L1->L1, and H keeps the core->core
      edge (the one iron hop that survives verbatim).
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

    Memtile ports: the L2 buffers bucket (by shared channel) into the down
    group {w_down stage, rings, relays}, the AN group {l2_ln, out blocks,
    gamma2}, the AN1 input group {x, residual, gamma1} and the w_up stage
    -- measured: four memtiles in the routed design at both pinned shapes,
    one per group. The down group's S2MM demand is ``2*n_b`` (stores +
    relays + one shim refill), within a memtile's six for ``n_b <= 3``;
    ``n_b`` is capped there rather than guessed past it. The A broadcast is
    ONE memtile MM2S port fanning to every up core (measured: one
    ``aie.flow`` source, n_b destinations).

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
      ``fused_add_layer_norm_2outs`` steps 32 lanes and silently truncates
      a row (``vector_chunks = cols / 32``). iron's baseline has emb 48,
      which compiles and routes identically but normalizes the first 32
      columns only. The builder REFUSES it unless
      ``allow_an_lane_truncation=True`` is passed -- the structure check
      passes it, explicitly, for the baseline; nothing numeric may.
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
      block, and per tail step d the final block THEN the partial. The
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
      cores carry 5, the up cores 3, the down cores 6 -- the chain's last
      core 5, its unreferenced partial dropped after role specialization).
      ``tail_pipeline_l1_bytes`` counts that, not ffn_resident's doubling
      rule, and the structure check pins the per-core buffer set. Moving an
      alloc inside a loop invites the ping-pong rotation and the baseline
      down core (52 KiB single) no longer fits.
    - TWO AIR WALLS, both measured on this module and both routed around:
      (1) ``air-dependency`` refuses a ``memref.dealloc`` whose buffer's
      last use sits inside an ``scf.index_switch`` region -- the tail's
      role switch -- with "operand #0 does not dominate this use" (the
      dealloc is handed a token defined in the child region). The three
      tail buffers are therefore NOT deallocated when ``n_b > 1``; nothing
      downstream needs the dealloc. (2) ``air-split-l2-memref`` (armed
      once a segment holds more than 4 tiles) asserts -- an unengaged
      ``std::optional`` in ``getTargetMemrefAllocs``, from
      ``getOffsetDimFromMemrefDim`` on a whole-buffer put -- on ``l2_ln``,
      which is read by two channels of different access rank (the 4-D
      retile and the 1-D residual relay). Every L2 alloc carries
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
from air.dialects.scf import for_, index_switch, yield_
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
CHANNEL_A_FEED = "tp_a_feed"
CHANNEL_WUP_FEED = "tp_wup_feed"
CHANNEL_H = "tp_h"
CHANNEL_DOWN_FEED = "tp_down_feed"
CHANNEL_ACC_STORE = "tp_acc_store"
CHANNEL_REDUCE = "tp_reduce"
CHANNEL_DOWN_OUT = "tp_down_out"
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

# A memtile has six S2MM ports; the down group's demand is 2*n_b.
MEMTILE_DMA_PORTS = 6
MAX_N_B = MEMTILE_DMA_PORTS // 2


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
            f"n_b ({n_b}) must be in [1, {MAX_N_B}]: the down group's memtile "
            f"S2MM demand is 2*n_b (n_b ring stores, n_b-1 relays, one w_down "
            f"refill) against {MEMTILE_DMA_PORTS} ports"
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
        ]
        + ([("l1_part", tile_m * tile_k)] if n_b > 1 else []),
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

    Returns:
        air.ir.Module with one function
        ``tail_pipeline(x, residual, gamma1, w_up_packed, w_down_packed,
        gamma2, y)``.
    """
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
    # L2: every staged buffer, possibly ping-ponged.
    wup_stage = n_b * tile_k * tile_n
    wdown_stage = depth * n_b * tile_n * tile_k
    blk = tile_m * tile_k
    l2_elems = (
        2 * emb_dim  # gamma1, gamma2
        + 2 * tile_m * emb_dim  # x, residual bands
        + AN_CORES * rpc * emb_dim  # l2_ln
        + wup_stage
        + wdown_stage
        + n_b * depth * blk  # rings
        + (n_b - 1) * blk  # relays
        + depth * blk  # out blocks
    )
    if 2 * l2_elems * itemsize > L2_BYTES:
        raise ValueError(
            f"staged buffers need {2 * l2_elems * itemsize} B ping-ponged, over "
            f"the {L2_BYTES}-byte memtile"
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
    l3_wdown_ty = MemRefType.get([ffn_dim * emb_dim], xrt_dtype)

    l2_space = IntegerAttr.get(T.i32(), MemorySpace.L2)
    l2_g_ty = MemRefType.get([emb_dim], xrt_dtype, memory_space=l2_space)
    l2_band_ty = MemRefType.get([tile_m * emb_dim], xrt_dtype, memory_space=l2_space)
    l2_ln_ty = MemRefType.get([rpc * emb_dim], xrt_dtype, memory_space=l2_space)
    l2_wup_ty = MemRefType.get([wup_stage], xrt_dtype, memory_space=l2_space)
    l2_wdown_ty = MemRefType.get([wdown_stage], xrt_dtype, memory_space=l2_space)
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
    add_func = _extern(
        ADD_SYMBOL, [l1_acc_ty, l1_acc_ty, l1_acc_ty, T.i32()], FFN_KERNEL_OBJ
    )

    Channel(CHANNEL_AN1_FEED, size=[AN_CORES, 1])
    Channel(CHANNEL_AN1_OUT, size=[AN_CORES, 1])
    if n_b > 1:
        Channel(CHANNEL_A_FEED, size=[1, 1], broadcast_shape=[n_b, 1])
    else:
        Channel(CHANNEL_A_FEED, size=[1, 1])
    Channel(CHANNEL_WUP_FEED, size=[n_b, 1])
    Channel(CHANNEL_H, size=[n_b, 1])
    Channel(CHANNEL_DOWN_FEED, size=[n_b, 1])
    Channel(CHANNEL_ACC_STORE, size=[n_b, 1])
    if n_b > 1:
        Channel(CHANNEL_REDUCE, size=[n_b - 1, 1])
    Channel(CHANNEL_DOWN_OUT, size=[1, 1])
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

    @FuncOp.from_py_func(
        l3_act_ty, l3_act_ty, l3_g_ty, l3_wup_ty, l3_wdown_ty, l3_g_ty, l3_act_ty
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
                    # l2_ln, which is read by two channels with different
                    # access ranks; the split is an optimization, not part
                    # of the design, so it is declined everywhere.
                    a = AllocOp(ty, [], [])
                    a.attributes["air.no_split"] = UnitAttr.get()
                    return a

                l2_g1 = _l2(l2_g_ty)
                l2_g2 = _l2(l2_g_ty)
                l2_x = _l2(l2_band_ty)
                l2_r = _l2(l2_band_ty)
                l2_ln = [_l2(l2_ln_ty) for _ in range(AN_CORES)]
                l2_wup = _l2(l2_wup_ty)
                l2_wdown = _l2(l2_wdown_ty)
                l2_ring = [[_l2(l2_blk_ty) for _ in range(depth)] for _ in range(n_b)]
                l2_red = [_l2(l2_blk_ty) for _ in range(n_b - 1)]
                l2_out = [_l2(l2_blk_ty) for _ in range(depth)]

                # ---------------- AN1 herd ----------------
                @herd(name=HERD_AN1, sizes=[AN_CORES, 1])
                def an1_herd(tx, ty, _sx, _sy):
                    l1_x = AllocOp(l1_rows_ty, [], [])
                    l1_r = AllocOp(l1_rows_ty, [], [])
                    l1_g = AllocOp(l1_g_ty, [], [])
                    l1_o1 = AllocOp(l1_rows_ty, [], [])
                    l1_o2 = AllocOp(l1_rows_ty, [], [])
                    cols_i32 = ConstantOp(T.i32(), emb_dim)
                    rows_i32 = ConstantOp(T.i32(), rpc)
                    for _band in range_(0, bands):
                        ChannelGet(CHANNEL_AN1_FEED, l1_g, indices=[tx, ty])
                        ChannelGet(CHANNEL_AN1_FEED, l1_x, indices=[tx, ty])
                        ChannelGet(CHANNEL_AN1_FEED, l1_r, indices=[tx, ty])
                        CallOp(
                            an_func,
                            [l1_x, l1_r, l1_g, l1_o1, l1_o2, cols_i32, rows_i32],
                        )
                        ChannelPut(CHANNEL_AN1_OUT, l1_o1, indices=[tx, ty])
                        yield_([])
                    DeallocOp(l1_x)
                    DeallocOp(l1_r)
                    DeallocOp(l1_g)
                    DeallocOp(l1_o1)
                    DeallocOp(l1_o2)

                an1_herd.attributes["link_with"] = StringAttr.get(AN_KERNEL_OBJ)

                # ---------------- UP herd ----------------
                @herd(name=HERD_UP, sizes=[n_b, 1])
                def up_herd(tx, ty, _sx, _sy):
                    l1_a = AllocOp(l1_a_ty, [], [])
                    l1_b = AllocOp(l1_bup_ty, [], [])
                    l1_c = AllocOp(l1_h_ty, [], [])
                    for _band in range_(0, bands):
                        for _s in range_(0, sweeps):
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
                            ChannelPut(CHANNEL_H, l1_c, indices=[tx, ty])
                            yield_([])
                        yield_([])
                    DeallocOp(l1_a)
                    DeallocOp(l1_b)
                    DeallocOp(l1_c)

                up_herd.attributes["link_with"] = StringAttr.get(FFN_KERNEL_OBJ)

                # ---------------- DOWN herd ----------------
                @herd(name=HERD_DOWN, sizes=[n_b, 1])
                def down_herd(tx, ty, _sx, _sy):
                    l1_h = AllocOp(l1_h_ty, [], [])
                    l1_g = AllocOp(l1_h_ty, [], [])
                    l1_b = AllocOp(l1_bdown_ty, [], [])
                    l1_acc_in = AllocOp(l1_acc_ty, [], [])
                    l1_acc_out = AllocOp(l1_acc_ty, [], [])
                    # The neighbour's partial: only a chain has one.
                    l1_part = AllocOp(l1_acc_ty, [], []) if n_b > 1 else None
                    h_elems_i32 = ConstantOp(T.i32(), tile_m * tile_n)
                    blk_i32 = ConstantOp(T.i32(), blk)

                    def _tail_step(has_partial, out_channel, out_index):
                        """One block of the reduction tail for one role."""
                        ChannelGet(CHANNEL_DOWN_FEED, l1_acc_in, indices=[tx, ty])
                        if has_partial:
                            ChannelGet(CHANNEL_DOWN_FEED, l1_part, indices=[tx, ty])
                            CallOp(add_func, [l1_part, l1_acc_in, l1_acc_out, blk_i32])
                            ChannelPut(out_channel, l1_acc_out, indices=[out_index, 0])
                        else:
                            ChannelPut(out_channel, l1_acc_in, indices=[out_index, 0])

                    def _role(k):
                        """Core k's tail: k == 0 drains to AN2, k > 0 to core k-1;
                        every core but the last folds in core k+1's partial."""
                        has_partial = k < n_b - 1
                        if k == 0:
                            _tail_step(has_partial, CHANNEL_DOWN_OUT, 0)
                        else:
                            _tail_step(has_partial, CHANNEL_REDUCE, k - 1)

                    for _band in range_(0, bands):
                        # Prime the ring, block by block, as iron's core does.
                        for _d in range_(0, depth):
                            CallOp(down_zero_func, [l1_acc_out])
                            ChannelPut(CHANNEL_ACC_STORE, l1_acc_out, indices=[tx, ty])
                            yield_([])
                        for _s in range_(0, sweeps):
                            ChannelGet(CHANNEL_H, l1_h, indices=[tx, ty])
                            CallOp(gelu_func, [l1_h, l1_g, h_elems_i32])
                            for _d in range_(0, depth):
                                # w_down chunk THEN the accumulator block.
                                ChannelGet(CHANNEL_DOWN_FEED, l1_b, indices=[tx, ty])
                                ChannelGet(
                                    CHANNEL_DOWN_FEED, l1_acc_in, indices=[tx, ty]
                                )
                                CallOp(down_mm_func, [l1_g, l1_b, l1_acc_in, l1_acc_out])
                                ChannelPut(
                                    CHANNEL_ACC_STORE, l1_acc_out, indices=[tx, ty]
                                )
                                yield_([])
                            yield_([])
                        # The reduction tail: final block d, plus the
                        # neighbour's partial where there is one.
                        for _d in range_(0, depth):
                            if n_b == 1:
                                _role(0)
                            else:
                                index_switch(
                                    [],
                                    tx,
                                    list(range(n_b - 1)),
                                    case_body_builder=lambda op, k, cv: (
                                        _role(k),
                                        yield_([]),
                                    )[-1],
                                    default_body_builder=lambda op: (
                                        _role(n_b - 1),
                                        yield_([]),
                                    )[-1],
                                )
                            yield_([])
                        yield_([])
                    DeallocOp(l1_h)
                    DeallocOp(l1_g)
                    DeallocOp(l1_b)
                    if n_b == 1:
                        # With a role switch in the tail these buffers' last
                        # uses sit inside scf.index_switch regions, and
                        # air-dependency then hands their deallocs a token
                        # defined in a child region ("operand #0 does not
                        # dominate this use"). Leaving them un-deallocated is
                        # the measured way through; see FOOTGUNS.
                        DeallocOp(l1_acc_in)
                        DeallocOp(l1_acc_out)

                down_herd.attributes["link_with"] = StringAttr.get(FFN_KERNEL_OBJ)

                # ---------------- AN2 herd ----------------
                row_map = _two_iv_map(tile_m, rpc)

                @herd(name=HERD_AN2, sizes=[AN_CORES, 1], operands=[s_y])
                def an2_herd(tx, ty, _sx, _sy, h_y):
                    l1_in = AllocOp(l1_rows_ty, [], [])
                    l1_r = AllocOp(l1_rows_ty, [], [])
                    l1_g = AllocOp(l1_g_ty, [], [])
                    l1_o1 = AllocOp(l1_rows_ty, [], [])
                    l1_o2 = AllocOp(l1_rows_ty, [], [])
                    cols_i32 = ConstantOp(T.i32(), emb_dim)
                    rows_i32 = ConstantOp(T.i32(), rpc)
                    for band in range_(0, bands):
                        ChannelGet(CHANNEL_AN2_FEED, l1_g, indices=[tx, ty])
                        ChannelGet(CHANNEL_AN2_FEED, l1_r, indices=[tx, ty])
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

                an2_herd.attributes["link_with"] = StringAttr.get(AN_KERNEL_OBJ)

                # ---------------- the memtile programs ----------------
                # Emitted after the herds (the structure check brace-matches
                # herds by name). Every L2-side offset below is a literal.
                band_row_map = _mul_add_map(tile_m)
                wup_off_map = _two_iv_map(k_steps * wup_stage, wup_stage)
                wdown_off_map = _mul_add_map(wdown_stage)

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
                    # AN1 output rows, one buffer per core (one writer each).
                    for c in range(AN_CORES):
                        ChannelGet(CHANNEL_AN1_OUT, l2_ln[c], indices=[c, 0])

                    # The A feed: per (sweep, k') the two halves, retiled,
                    # broadcast to every up core. k' is Python-unrolled:
                    # its offset addresses a staged buffer.
                    for _s in range_(0, sweeps):
                        for kb in range(k_steps):
                            for c in range(AN_CORES):
                                ChannelPut(
                                    CHANNEL_A_FEED,
                                    l2_ln[c],
                                    offsets=[0, kb * (tile_k // MICRO), 0, 0],
                                    sizes=retile_sizes,
                                    strides=retile_strides,
                                )
                        yield_([])

                    # w_up: refill per (sweep, k'), L3 offsets only; each up
                    # core's slice at a tx-literal offset.
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

                    # The ring priming stores.
                    for t in range(n_b):
                        for d in range(depth):
                            ChannelGet(CHANNEL_ACC_STORE, l2_ring[t][d], indices=[t, 0])

                    # The down feed: per sweep refill the w_down chunks for
                    # every (block, column), then per block d and core t:
                    # w_down slice, ring block d; and take the store back.
                    for s in range_(0, sweeps):
                        wd_off = affine_apply(wdown_off_map, [s])
                        dma_memcpy_nd(
                            l2_wdown,
                            s_wdown,
                            src_offsets=[wd_off],
                            src_sizes=[wdown_stage],
                            src_strides=[1],
                        )
                        for d in range(depth):
                            for t in range(n_b):
                                ChannelPut(
                                    CHANNEL_DOWN_FEED,
                                    l2_wdown,
                                    offsets=[(d * n_b + t) * tile_n * tile_k],
                                    sizes=[tile_n * tile_k],
                                    strides=[1],
                                    indices=[t, 0],
                                )
                                ChannelPut(
                                    CHANNEL_DOWN_FEED, l2_ring[t][d], indices=[t, 0]
                                )
                                ChannelGet(
                                    CHANNEL_ACC_STORE, l2_ring[t][d], indices=[t, 0]
                                )
                        yield_([])

                    # The reduction tail, block by block: the final ring
                    # block to every core; core t+1's partial relayed to
                    # core t; core 0's result to the out block.
                    for d in range(depth):
                        for t in range(n_b):
                            ChannelPut(CHANNEL_DOWN_FEED, l2_ring[t][d], indices=[t, 0])
                        for t in reversed(range(n_b - 1)):
                            ChannelGet(CHANNEL_REDUCE, l2_red[t], indices=[t, 0])
                            ChannelPut(CHANNEL_DOWN_FEED, l2_red[t], indices=[t, 0])
                        ChannelGet(CHANNEL_DOWN_OUT, l2_out[d], indices=[0, 0])

                    # AN2 inputs: gamma2, the residual rows (AN1's output),
                    # then every block un-tiled into its column block.
                    dma_memcpy_nd(l2_g2, s_g2)
                    for c in range(AN_CORES):
                        ChannelPut(CHANNEL_AN2_FEED, l2_g2, indices=[c, 0])
                        ChannelPut(CHANNEL_AN2_FEED, l2_ln[c], indices=[c, 0])
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
                    [l2_g1, l2_g2, l2_x, l2_r, l2_wup, l2_wdown]
                    + l2_ln
                    + [b for ring in l2_ring for b in ring]
                    + l2_red
                    + l2_out
                ):
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
    """Pre-tile ``w_down``: flat, (sweep, block, column)-major, blocked.

    Element ``w_down[n, k]`` lands in the ``[tile_n, tile_k]`` block for
    ``(s, d, t)`` where ``n`` falls in group ``g = s * n_b + t`` and ``k`` in
    block ``d = k // tile_k``. One sweep's refill -- every (block, column)
    slice -- is one contiguous L3 run, and each feed put a contiguous slice
    of it at a literal offset.
    """
    ffn_dim, emb_dim = w_down.shape
    sweeps = ffn_dim // (n_b * tile_n)
    return np.ascontiguousarray(
        w_down.reshape(
            sweeps, n_b, tile_n // MICRO, MICRO, down_proj_depth, tile_k // MICRO, MICRO
        )
        .transpose(0, 4, 1, 2, 5, 3, 6)
        .reshape(-1)
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
