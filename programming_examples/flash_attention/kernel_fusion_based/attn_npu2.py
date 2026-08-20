# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Flash attention with memtile-relayed dataflow — selective Q capture.

All data (Q, K, V) routes through memtile for L3→L2→L1 transfer.
Per-stage QKIn/QK2L1 and VIn/V2L1 channels handle the relay.
Q tiles are selectively captured: each tile receives all NQ Q sends
but only copies the one matching its tx. Cascade merge follows the
cascade-after pattern.

Multi-head support via 3D channels with segment unroll:
  - num_heads_per_unroll=2 heads are processed per segment unroll
  - Segment sizes=[num_heads_per_unroll, 1], each segment instance handles
    one head index
  - 3D channels have head dimension as first index
  - Cascade channels remain 2D (shared within each segment instance)

Supports multi-head (MHA), grouped-query (GQA), and causal masking.

Default design parameters:
  lk=512, lkp=64, lq=512, lqp=256, dk=64, dv=64
  num_q_tiles=4, num_cascade_stages=4, num_heads=2
  Shared-buffer mode (lkp == dk).

DMA channel strategy (2 S2MM + 2 MM2S per compute tile):
  S2MM 0: QK channel (Q selective capture, then K chunks)
  S2MM 1: V (per-stage via memtile)
  MM2S 0: Cascade or output
  MM2S 1: Cascade

Channel layout:
  QKIn_s/QK2L1_s: per-stage memtile relay with horizontal broadcast
  VIn_s/V2L1_s: per-stage memtile relay with horizontal broadcast
  cascade_gp/cascade_up/cascade_sp: 2D cascade channels (per-segment)
  Gp2L2/GpOut: output from ty=0 tiles
"""

import argparse
from math import sqrt

import numpy as np

import air
from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects.air import channel
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, CollapseShapeOp, DeallocOp, load, store
from air.dialects.func import FuncOp, CallOp
from air.dialects.scf import for_ as scf_range, yield_
from air.dialects import scf, affine, arith


@module_builder
def build_module(
    lk=512,
    lkp=64,
    lq=512,
    lqp=256,
    dk=64,
    dv=64,
    num_q_tiles=4,
    num_cascade_stages=4,
    num_heads=2,
    num_kv_heads=None,
    causal=False,
    num_heads_per_unroll=2,
    window=0,
):
    """Build flash attention module with selective Q capture pattern.

    Args:
        lk: Total K/V sequence length (default: 512)
        lkp: K/V chunk size per tile (default: 64)
        lq: Total Q sequence length (default: 512)
        lqp: Q chunk size per launch iteration (default: 256)
        dk: Key dimension (default: 64)
        dv: Value dimension (default: 64)
        num_q_tiles: Number of tiles to partition Q chunk into (default: 4)
        num_cascade_stages: Number of cascade pipeline stages (default: 4)
        num_heads: Number of attention heads (default: 2)
        num_kv_heads: Number of key/value heads for grouped-query attention
            (GQA). If None, defaults to num_heads (standard MHA).
        causal: Whether to enable causal (autoregressive) masking.
        num_heads_per_unroll: Heads processed per segment instance (default: 2).
            Acts as the physical-column multiplier — physical columns =
            num_heads_per_unroll * num_q_tiles (must be <= 8 on NPU2). Requires
            num_heads % num_heads_per_unroll == 0.
        window: Sliding-window length W (default 0 = full causal). W > 0 keeps
            q_abs - W < k_abs <= q_abs.

            **This kwarg emits no MLIR.** The window is a compile-time constant
            of the KERNEL (``-DWINDOW_LEN=W`` on attn_npu2.cc), because the
            launch signature attention_bf16(q,k,v,gp) has no host scalar path.
            It is accepted here so the builder can (a) reject window/tiling
            combinations the kernel cannot express and (b) let the harness band
            its reference. The caller is responsible for compiling
            attn_npu2.o with a MATCHING -DWINDOW_LEN — the canonical Makefile
            drives both from one WINDOW variable.

            A mismatch is not silent: the reference and the device then
            implement different masks and the element-wise check FAILS. It
            cannot produce a false PASS in either direction.
    """
    # Validate
    assert lq % lqp == 0, f"lq ({lq}) must be divisible by lqp ({lqp})"
    assert (
        lqp % num_q_tiles == 0
    ), f"lqp ({lqp}) must be divisible by num_q_tiles ({num_q_tiles})"
    assert lk % lkp == 0, f"lk ({lk}) must be divisible by lkp ({lkp})"
    assert lk % (lkp * num_cascade_stages) == 0, (
        f"lk ({lk}) must be divisible by lkp * num_cascade_stages "
        f"({lkp * num_cascade_stages})"
    )
    dk_tile = lkp
    assert dk % dk_tile == 0, f"dk ({dk}) must be divisible by dk_tile/lkp ({dk_tile})"
    dk_chunks = dk // dk_tile
    dv_tile = lkp
    assert dv % dv_tile == 0, f"dv ({dv}) must be divisible by dv_tile/lkp ({dv_tile})"
    dv_chunks = dv // dv_tile
    if causal:
        assert lq == lk, f"Causal masking requires lq == lk, got lq={lq}, lk={lk}"
        assert lqp // num_q_tiles == lkp, (
            f"Causal masking requires tile_size_q == lkp, got "
            f"tile_size_q={lqp // num_q_tiles}, lkp={lkp}"
        )
    assert window >= 0, f"window must be >= 0, got {window}"
    if window > 0:
        # Fail CLOSED: a window without causal masking would emit no mask call
        # at all while the reference bands, which is a confusing failure rather
        # than a refusal.
        assert causal, "window > 0 requires causal=True (the band is a two-sided causal mask)"
        assert window % lkp == 0, (
            f"window ({window}) must be a multiple of lkp ({lkp}). The kernel's "
            f"partial-select path is per 8-column block within one lkp chunk; a "
            f"window that is not chunk-aligned is expressible but has never been "
            f"validated, so it is refused rather than run"
        )

    # Multi-head / GQA parameters
    if num_kv_heads is None:
        num_kv_heads = num_heads
    assert num_kv_heads > 0, f"num_kv_heads must be positive, got {num_kv_heads}"
    assert num_heads % num_kv_heads == 0, (
        f"num_heads ({num_heads}) must be divisible by "
        f"num_kv_heads ({num_kv_heads})"
    )
    gqa_group_size = num_heads // num_kv_heads

    assert num_heads % num_heads_per_unroll == 0, (
        f"num_heads ({num_heads}) must be divisible by "
        f"num_heads_per_unroll ({num_heads_per_unroll})"
    )
    num_head_groups = num_heads // num_heads_per_unroll

    bf16 = Type.parse("bf16")
    i32 = IntegerType.get_signless(32)
    index_type = IndexType.get()

    M = 8  # mmul_m = mmul_k = mmul_n

    # Derived parameters
    num_lq_iters = lq // lqp
    tile_size_q = lqp // num_q_tiles
    num_chunks = lk // lkp
    chunks_per_stage = num_chunks // num_cascade_stages
    lk_per_stage = lkp * chunks_per_stage

    NQ = num_q_tiles
    NS = num_cascade_stages

    # Memory spaces
    l1_space = IntegerAttr.get(i32, 2)
    l2_space = IntegerAttr.get(i32, 1)

    # L1 MemRefTypes (Q and K use dk_tile, not full dk)
    q_l1_t = MemRefType.get([tile_size_q, dk_tile], bf16, memory_space=l1_space)
    k_l1_t = MemRefType.get([lkp, dk_tile], bf16, memory_space=l1_space)
    v_l1_t = MemRefType.get([lkp, dv_tile], bf16, memory_space=l1_space)
    g_l1_2d = MemRefType.get([tile_size_q, lkp], bf16, memory_space=l1_space)
    g_l1_1d = MemRefType.get([tile_size_q * lkp], bf16, memory_space=l1_space)
    gp_l1_t = MemRefType.get([tile_size_q, dv_tile], bf16, memory_space=l1_space)
    up_l1_t = MemRefType.get([tile_size_q, 1], bf16, memory_space=l1_space)

    # L2 MemRefTypes (QK relay uses dk_tile)
    qk_l2_t = MemRefType.get([lkp, dk_tile], bf16, memory_space=l2_space)
    v_l2_t = MemRefType.get([lkp, dv_tile], bf16, memory_space=l2_space)
    gp_l2_t = MemRefType.get([lqp, dv_tile], bf16, memory_space=l2_space)

    # L3 MemRefTypes (3D with head dimension)
    q_l3_t = MemRefType.get([num_heads, lq, dk], bf16)
    k_l3_t = MemRefType.get([num_kv_heads, lk, dk], bf16)
    # V and output L3 use transposed layout for contiguous dv_tile access:
    # [heads * dv_chunks, seq, dv_tile] instead of [heads, seq, dv]
    v_l3_t = MemRefType.get([num_kv_heads * dv_chunks, lk, dv_tile], bf16)
    gp_l3_t = MemRefType.get([num_heads * dv_chunks, lq, dv_tile], bf16)

    # External function declarations
    def external_func(name, inputs, outputs=None, link_with=None, visibility="private"):
        if outputs is None:
            outputs = []
        func_type = FunctionType.get(inputs, outputs)
        func = FuncOp(name=name, type=func_type, visibility=visibility)
        func.attributes["llvm.emit_c_interface"] = UnitAttr.get()
        if link_with:
            func.attributes["link_with"] = StringAttr.get(link_with)
        return func

    external_func("zero_fill_g_bf16", [g_l1_1d], link_with="attn_npu2.o")
    external_func("zero_fill_gp_bf16", [gp_l1_t], link_with="attn_npu2.o")
    external_func("zero_fill_sp_bf16", [up_l1_t], link_with="attn_npu2.o")
    external_func("neg_inf_fill_up_bf16", [up_l1_t], link_with="attn_npu2.o")
    external_func(
        "matmul_a_b_bf16",
        [q_l1_t, k_l1_t, g_l1_1d],
        link_with="attn_npu2.o",
    )
    external_func(
        "matmul_g_b_bf16",
        [g_l1_1d, v_l1_t, gp_l1_t],
        link_with="attn_npu2.o",
    )
    external_func(
        "fused_softmax",
        [g_l1_1d, up_l1_t, up_l1_t, up_l1_t],
        link_with="attn_npu2.o",
    )
    external_func("maximum_up_u_bf16", [up_l1_t, up_l1_t], link_with="attn_npu2.o")
    external_func(
        "exp_up_minus_u",
        [up_l1_t, up_l1_t, up_l1_t],
        link_with="attn_npu2.o",
    )
    external_func("mul_r_gp", [up_l1_t, gp_l1_t], link_with="attn_npu2.o")
    external_func(
        "accum_sp_r_s",
        [up_l1_t, up_l1_t, up_l1_t],
        link_with="attn_npu2.o",
    )
    external_func(
        "vector_copy_32elems", [i32, up_l1_t, up_l1_t], link_with="attn_npu2.o"
    )
    external_func("copy_tile", [k_l1_t, q_l1_t], link_with="attn_npu2.o")
    external_func("div_gp_sp", [up_l1_t, gp_l1_t], link_with="attn_npu2.o")
    external_func("add_gp_g", [gp_l1_t, gp_l1_t], link_with="attn_npu2.o")
    if causal:
        external_func("apply_causal_mask", [g_l1_2d, i32, i32], link_with="attn_npu2.o")

    # ----------------------------------------------------------------
    # Channel declarations (3D with head dimension for multi-head)
    # ----------------------------------------------------------------

    # QK: per-stage through memtile (3D with head dimension)
    # L3→memtile via QKIn_s, memtile→L1 via QK2L1_s with broadcast
    for s in range(NS):
        Channel(
            f"QK2L1_{s}",
            size=[num_heads_per_unroll, 1, 1],
            broadcast_shape=[num_heads_per_unroll, 1, NQ],
        )
        Channel(f"QKIn_{s}", size=[num_heads_per_unroll])

    # V: per-stage through memtile (3D with head dimension)
    for s in range(NS):
        Channel(
            f"V2L1_{s}",
            size=[num_heads_per_unroll, 1, 1],
            broadcast_shape=[num_heads_per_unroll, 1, NQ],
        )
        Channel(f"VIn_{s}", size=[num_heads_per_unroll])

    # Cascade: 2D per-segment (shared within each segment instance)
    channel("cascade_gp", size=[NQ, NS - 1], channel_type="npu_cascade")
    channel("cascade_up", size=[NQ, NS - 1], channel_type="npu_cascade")
    channel("cascade_sp", size=[NQ, NS - 1], channel_type="npu_cascade")

    # Output: L1-to-L2 gather, then L2-to-L3
    Channel("Gp2L2", size=[NQ, 1])
    Channel("GpOut", size=[num_heads_per_unroll])

    # ----------------------------------------------------------------
    # Main attention function
    # ----------------------------------------------------------------
    @FuncOp.from_py_func(q_l3_t, k_l3_t, v_l3_t, gp_l3_t)
    def attention_bf16(q_in, k_in, v_in, gp_out):
        c1 = ConstantOp(index_type, 1)
        c_lq_iters = ConstantOp(index_type, num_lq_iters)
        c_num_head_groups = ConstantOp(index_type, num_head_groups)

        if dv_chunks > 1:
            c_dv_chunks = ConstantOp(index_type, dv_chunks)
            launch_sizes = [c_lq_iters, c_num_head_groups, c_dv_chunks]
        else:
            launch_sizes = [c_lq_iters, c_num_head_groups]

        @launch(
            operands=[q_in, k_in, v_in, gp_out],
            sizes=launch_sizes,
        )
        def launch_body(*launch_args):
            if dv_chunks > 1:
                lx, ly, lz, lsx, lsy, lsz, q, k, v, gp = launch_args
            else:
                lx, ly, lsx, lsy, q, k, v, gp = launch_args
                lz = ConstantOp(index_type, 0)

            # Compute Q offset from launch iteration index
            affine_map_q_launch = AffineMap.get(
                0,
                1,
                [
                    AffineExpr.get_mul(
                        AffineSymbolExpr.get(0),
                        AffineConstantExpr.get(lqp * dk),
                    )
                ],
            )
            q_launch_off = affine_apply(affine_map_q_launch, [lx])

            # Output launch offset (transposed layout uses dv_tile, not dv)
            affine_map_out_launch = AffineMap.get(
                0,
                1,
                [
                    AffineExpr.get_mul(
                        AffineSymbolExpr.get(0),
                        AffineConstantExpr.get(lqp * dv_tile),
                    )
                ],
            )
            out_launch_off = affine_apply(affine_map_out_launch, [lx])

            # Compute head base from head group index (ly)
            # head_base = ly * num_heads_per_unroll
            affine_map_head_base = AffineMap.get(
                0,
                1,
                [
                    AffineExpr.get_mul(
                        AffineSymbolExpr.get(0),
                        AffineConstantExpr.get(num_heads_per_unroll),
                    )
                ],
            )
            head_base = affine_apply(affine_map_head_base, [ly])

            # Offset maps for one head's worth of Q/K/V/output data
            affine_map_head_q = AffineMap.get(
                0,
                1,
                [
                    AffineExpr.get_mul(
                        AffineSymbolExpr.get(0),
                        AffineConstantExpr.get(lq * dk),
                    )
                ],
            )
            affine_map_head_k = AffineMap.get(
                0,
                1,
                [
                    AffineExpr.get_mul(
                        AffineSymbolExpr.get(0),
                        AffineConstantExpr.get(lk * dk),
                    )
                ],
            )
            # V/output head offsets use transposed layout:
            # head_v_off = (kv_head * dv_chunks + lz) * lk * dv_tile
            # head_out_off = (head * dv_chunks + lz) * lq * dv_tile
            affine_map_head_v_dv = AffineMap.get(
                0,
                2,
                [
                    AffineExpr.get_mul(
                        AffineExpr.get_add(
                            AffineExpr.get_mul(
                                AffineSymbolExpr.get(0),
                                AffineConstantExpr.get(dv_chunks),
                            ),
                            AffineSymbolExpr.get(1),
                        ),
                        AffineConstantExpr.get(lk * dv_tile),
                    )
                ],
            )
            affine_map_head_out_dv = AffineMap.get(
                0,
                2,
                [
                    AffineExpr.get_mul(
                        AffineExpr.get_add(
                            AffineExpr.get_mul(
                                AffineSymbolExpr.get(0),
                                AffineConstantExpr.get(dv_chunks),
                            ),
                            AffineSymbolExpr.get(1),
                        ),
                        AffineConstantExpr.get(lq * dv_tile),
                    )
                ],
            )

            # s0 + s1
            affine_map_add = AffineMap.get(
                0,
                2,
                [
                    AffineExpr.get_add(
                        AffineSymbolExpr.get(0),
                        AffineSymbolExpr.get(1),
                    )
                ],
            )

            # head_1 = head_base + 1
            affine_map_plus1 = AffineMap.get(
                0,
                1,
                [
                    AffineExpr.get_add(
                        AffineSymbolExpr.get(0),
                        AffineConstantExpr.get(1),
                    )
                ],
            )

            # ----------------------------------------------------------
            # For each head in the unroll group, send Q/K/V and get output
            # ----------------------------------------------------------
            # GQA: compute KV head index from Q head index
            # kv_head = q_head // gqa_group_size
            if gqa_group_size > 1:
                affine_map_kv_head = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_floor_div(
                            AffineSymbolExpr.get(0),
                            AffineConstantExpr.get(gqa_group_size),
                        )
                    ],
                )

            for head_local in range(num_heads_per_unroll):
                if head_local == 0:
                    head_idx = head_base
                else:
                    head_idx = affine_apply(affine_map_plus1, [head_base])

                # KV head index: same as Q head for MHA, floor-div for GQA
                if gqa_group_size == 1:
                    kv_head_idx = head_idx
                else:
                    kv_head_idx = affine_apply(
                        affine_map_kv_head,
                        [head_idx],
                    )

                head_q_off = affine_apply(affine_map_head_q, [head_idx])
                head_k_off = affine_apply(affine_map_head_k, [kv_head_idx])
                head_v_off = affine_apply(affine_map_head_v_dv, [kv_head_idx, lz])
                head_out_off = affine_apply(affine_map_head_out_dv, [head_idx, lz])

                head_offset_idx = ConstantOp(index_type, head_local)

                # Combined Q offset = head_q_off + q_launch_off
                q_combined = affine_apply(affine_map_add, [head_q_off, q_launch_off])
                # Combined output offset (uses dv stride, not dk)
                out_combined = affine_apply(
                    affine_map_add, [head_out_off, out_launch_off]
                )

                # Q puts: 4D transfer with dk_chunks tiling
                # Sends Q_tile0_dk0, Q_tile0_dk1, ..., Q_tileN_dk(C-1)
                # Each sub-transfer is [tile_size_q, dk_tile]
                for stage in range(NS):
                    ChannelPut(
                        f"QKIn_{stage}",
                        q,
                        indices=[head_offset_idx],
                        offsets=[0, q_combined],
                        sizes=[NQ, dk_chunks, tile_size_q, dk_tile],
                        strides=[tile_size_q * dk, dk_tile, dk, 1],
                    )

                # K puts: 4D transfer with dk_chunks tiling
                # Sends K_chunk0_dk0, K_chunk0_dk1, ..., K_chunkN_dk(C-1)
                # Each sub-transfer is [lkp, dk_tile]
                for stage in range(NS):
                    k_stage_off_val = stage * lk_per_stage * dk
                    k_combined = affine_apply(
                        affine_map_add,
                        [head_k_off, ConstantOp(index_type, k_stage_off_val)],
                    )
                    ChannelPut(
                        f"QKIn_{stage}",
                        k,
                        indices=[head_offset_idx],
                        offsets=[0, k_combined],
                        sizes=[chunks_per_stage, dk_chunks, lkp, dk_tile],
                        strides=[lkp * dk, dk_tile, dk, 1],
                    )

                # V puts: contiguous dv_tile slice (transposed L3 layout)
                for stage in range(NS):
                    v_stage_off_val = stage * lk_per_stage * dv_tile
                    v_combined = affine_apply(
                        affine_map_add,
                        [head_v_off, ConstantOp(index_type, v_stage_off_val)],
                    )
                    ChannelPut(
                        f"VIn_{stage}",
                        v,
                        indices=[head_offset_idx],
                        offsets=[0, 0, v_combined],
                        sizes=[chunks_per_stage, lkp, dv_tile],
                        strides=[lkp * dv_tile, dv_tile, 1],
                    )

            # ----------------------------------------------------------
            # Segment: unrolled over heads
            # ----------------------------------------------------------
            c_num_heads_unroll = ConstantOp(index_type, num_heads_per_unroll)
            c1_seg = ConstantOp(index_type, 1)

            @segment(
                name="attn_seg",
                operands=[],
                sizes=[c_num_heads_unroll, c1_seg],
            )
            def segment_body(seg_x, seg_y, seg_sx, seg_sy):
                # L2 allocations for QK and V (per-stage) and output
                qk_l2_bufs = [AllocOp(qk_l2_t, [], []) for _ in range(NS)]
                v_l2_bufs = [AllocOp(v_l2_t, [], []) for _ in range(NS)]
                gp_l2 = AllocOp(gp_l2_t, [], [])

                # L1 allocations passed to herd
                q_saved_bufs = [AllocOp(q_l1_t, [], []) for _ in range(dk_chunks)]
                qk_buf = AllocOp(k_l1_t, [], [])
                v_l1 = AllocOp(v_l1_t, [], [])
                g_l1 = AllocOp(g_l1_2d, [], [])
                gp_l1 = AllocOp(gp_l1_t, [], [])
                up_l1 = AllocOp(up_l1_t, [], [])
                sp_l1 = AllocOp(up_l1_t, [], [])
                if causal:
                    # Counter layout: [0]=q_block, [1]=boot_flag, [2]=head_local,
                    # [3]=dv_iter (counts dv_chunk iterations, for dv_chunks>1 guard)
                    #
                    # BOOT CONTRACT: this alloc lowers to an AIE buffer with NO
                    # initial value, so the boot-flag test ([1] == 0) is only
                    # meaningful because partition initialization zeroes tile
                    # data memory before the first execution -- a property of
                    # the runtime stack, measured (every cold dispatch of every
                    # causal artifact boots correctly), not guaranteed by this
                    # source. The counter-increment wrap below restores the
                    # boot state at the END of every complete execution, so
                    # after the first boot the design no longer depends on
                    # anything re-zeroing this memory. Plumbing a real
                    # initial_value through air-to-aie would close the
                    # remaining cold-start assumption and is a compiler
                    # feature, not a builder change.
                    ctr_size = 4 if dv_chunks > 1 else 3
                    ctr_t = MemRefType.get([ctr_size], i32, memory_space=l1_space)
                    causal_ctr = AllocOp(ctr_t, [], [])

                c_nq = ConstantOp(index_type, NQ)
                c_ns = ConstantOp(index_type, NS)
                c0_seg = ConstantOp(index_type, 0)
                c_chunks_s = ConstantOp(index_type, chunks_per_stage)

                # QK streaming: L3-to-L2-to-L1 per stage
                # Q: NQ * dk_chunks transfers, then K: chunks_per_stage * dk_chunks
                # L2 buffer is [lkp, dk_tile], all transfers are [dk_tile, dk_tile]
                c_nq_dk = ConstantOp(index_type, NQ * dk_chunks)
                c_chunks_dk = ConstantOp(index_type, chunks_per_stage * dk_chunks)
                for stage in range(NS):
                    for qt_iter in scf_range(0, c_nq_dk, 1):
                        ChannelGet(
                            f"QKIn_{stage}",
                            qk_l2_bufs[stage].result,
                            indices=[seg_x],
                        )
                        ChannelPut(
                            f"QK2L1_{stage}",
                            qk_l2_bufs[stage].result,
                            indices=[seg_x, c0_seg, c0_seg],
                            offsets=[0, 0, 0, 0],
                            sizes=[dk_tile // M, lkp // M, M, M],
                            strides=[M, dk_tile * M, dk_tile, 1],
                        )
                        yield_([])
                    for chunk_iter in scf_range(0, c_chunks_dk, 1):
                        ChannelGet(
                            f"QKIn_{stage}",
                            qk_l2_bufs[stage].result,
                            indices=[seg_x],
                        )
                        ChannelPut(
                            f"QK2L1_{stage}",
                            qk_l2_bufs[stage].result,
                            indices=[seg_x, c0_seg, c0_seg],
                            offsets=[0, 0, 0, 0],
                            sizes=[dk_tile // M, lkp // M, M, M],
                            strides=[M, dk_tile * M, dk_tile, 1],
                        )
                        yield_([])

                # V streaming: L3-to-L2-to-L1 per stage
                for stage in range(NS):
                    for chunk_iter in scf_range(0, c_chunks_s, 1):
                        ChannelGet(
                            f"VIn_{stage}",
                            v_l2_bufs[stage].result,
                            indices=[seg_x],
                        )
                        ChannelPut(
                            f"V2L1_{stage}",
                            v_l2_bufs[stage].result,
                            indices=[
                                seg_x,
                                c0_seg,
                                c0_seg,
                            ],  # [head, stage_dim=0, col_dim=0]
                            offsets=[0, 0, 0, 0],
                            sizes=[dv_tile // M, lkp // M, M, M],
                            strides=[M, dv_tile * M, dv_tile, 1],
                        )
                        yield_([])

                # ----------------------------------------------------------
                # Herd: [NQ, NS] — pass seg_x as operand
                # ----------------------------------------------------------
                herd_operands = q_saved_bufs + [
                    qk_buf,
                    v_l1,
                    g_l1,
                    gp_l1,
                    up_l1,
                    sp_l1,
                    seg_x,
                ]
                if causal:
                    herd_operands.append(causal_ctr)

                @herd(
                    name="herd_0",
                    sizes=[c_nq, c_ns],
                    operands=herd_operands,
                    link_with="attn_npu2.o",
                )
                def herd_body(tx, ty, hsx, hsy, *all_args):
                    # Unpack: dk_chunks Q buffers, then qk, v, g, gp, up, sp, seg_x, [causal_ctr]
                    q_bufs = list(all_args[:dk_chunks])
                    qk = all_args[dk_chunks]
                    v = all_args[dk_chunks + 1]
                    g = all_args[dk_chunks + 2]
                    gp = all_args[dk_chunks + 3]
                    up_buf = all_args[dk_chunks + 4]
                    sp_buf = all_args[dk_chunks + 5]
                    h_seg_x = all_args[dk_chunks + 6]
                    counter_buf = all_args[dk_chunks + 7] if causal else None
                    # Precompute affine sets for per-stage V dispatch
                    s0 = AffineSymbolExpr.get(0)
                    s1 = AffineSymbolExpr.get(1)
                    c_ns_m1 = AffineConstantExpr.get(NS - 1)
                    stage_sets = []
                    for s in range(NS):
                        cs = AffineConstantExpr.get(s)
                        stage_sets.append(
                            IntegerSet.get(
                                0,
                                2,
                                [s0, s1 - cs],
                                [False, True],
                            )
                        )

                    # === INIT PHASE (FIRST — before any channel ops) ===
                    CallOp([], "zero_fill_gp_bf16", [gp])
                    CallOp([], "zero_fill_sp_bf16", [sp_buf])
                    CallOp([], "neg_inf_fill_up_bf16", [up_buf])

                    # === CAUSAL COUNTER INIT ===
                    if causal:
                        c0_ctr = ConstantOp(index_type, 0)
                        c1_ctr = ConstantOp(index_type, 1)
                        c2_ctr = ConstantOp(index_type, 2)
                        c3_ctr = ConstantOp(index_type, 3) if dv_chunks > 1 else None
                        boot_flag = load(counter_buf, [c1_ctr])
                        is_first = arith.CmpIOp(
                            arith.CmpIPredicate.eq,
                            boot_flag,
                            ConstantOp(i32, 0),
                        )
                        if_first = scf.IfOp(is_first)
                        with InsertionPoint(if_first.then_block):
                            store(ConstantOp(i32, 0), counter_buf, [c0_ctr])
                            store(ConstantOp(i32, 1), counter_buf, [c1_ctr])
                            store(ConstantOp(i32, 0), counter_buf, [c2_ctr])
                            if dv_chunks > 1:
                                store(ConstantOp(i32, 0), counter_buf, [c3_ctr])
                            scf.YieldOp([])

                    # === Q SELECTIVE CAPTURE ===
                    # Receive all NQ Q tiles × dk_chunks dk slices, but only
                    # copy the one matching this tile's tx index.
                    # Stage-gated get from per-stage QK2L1_s channels.
                    for qt in range(NQ):
                        for dk_c in range(dk_chunks):
                            for s in range(NS):
                                if_qk_q = affine.AffineIfOp(
                                    stage_sets[s],
                                    cond_operands=[tx, ty],
                                )
                                with InsertionPoint(if_qk_q.then_block):
                                    ChannelGet(
                                        f"QK2L1_{s}",
                                        qk,
                                        indices=[h_seg_x, ty, tx],
                                    )
                                    affine.AffineYieldOp([])
                            cmp = arith.CmpIOp(
                                arith.CmpIPredicate.eq,
                                arith.IndexCastOp(i32, tx),
                                arith.ConstantOp(i32, qt),
                            )
                            if_cap = scf.IfOp(cmp)
                            with InsertionPoint(if_cap.then_block):
                                CallOp([], "copy_tile", [qk, q_bufs[dk_c]])
                                scf.YieldOp([])

                    # === K CHUNK LOOP ===
                    c_chunks_h = ConstantOp(index_type, chunks_per_stage)
                    for chunk_iter in scf_range(0, c_chunks_h, 1):
                        # 1. Zero fill G (FIRST — once per K seq chunk)
                        g1d = CollapseShapeOp(g_l1_1d, g, [[0, 1]])
                        CallOp([], "zero_fill_g_bf16", [g1d])

                        # 2. dk_chunks loop: K get + matmul (accumulate G)
                        for dk_c in range(dk_chunks):
                            for s in range(NS):
                                if_qk_k = affine.AffineIfOp(
                                    stage_sets[s],
                                    cond_operands=[tx, ty],
                                )
                                with InsertionPoint(if_qk_k.then_block):
                                    ChannelGet(
                                        f"QK2L1_{s}",
                                        qk,
                                        indices=[h_seg_x, ty, tx],
                                    )
                                    affine.AffineYieldOp([])
                            # Matmul Q_dk_slice @ K_dk_slice^T → G (accumulate)
                            CallOp([], "matmul_a_b_bf16", [q_bufs[dk_c], qk, g1d])

                        # 3. V get via affine.if per stage (AFTER dk_chunks)
                        #    — 3D index with head dim
                        for s in range(NS):
                            if_v = affine.AffineIfOp(
                                stage_sets[s],
                                cond_operands=[tx, ty],
                            )
                            with InsertionPoint(if_v.then_block):
                                ChannelGet(
                                    f"V2L1_{s}",
                                    v,
                                    indices=[h_seg_x, ty, tx],
                                )
                                affine.AffineYieldOp([])

                        # 4b. Apply causal mask (after matmul, before softmax)
                        if causal:
                            c_cps_i32 = ConstantOp(i32, chunks_per_stage)
                            ty_i32 = arith.IndexCastOp(i32, ty).result
                            chunk_i32 = arith.IndexCastOp(
                                i32,
                                chunk_iter,
                            ).result
                            kv_base = arith.MulIOp(ty_i32, c_cps_i32)
                            kv_block = arith.AddIOp(
                                kv_base.result,
                                chunk_i32,
                            )
                            q_base = load(counter_buf, [c0_ctr])
                            tx_i32 = arith.IndexCastOp(i32, tx).result
                            q_block = arith.AddIOp(q_base, tx_i32)
                            CallOp(
                                [],
                                "apply_causal_mask",
                                [g, q_block.result, kv_block.result],
                            )

                        # 5. Softmax + accumulate
                        s_tmp = AllocOp(up_l1_t, [], [])
                        r_tmp = AllocOp(up_l1_t, [], [])
                        CallOp(
                            [],
                            "fused_softmax",
                            [g1d, up_buf, s_tmp.result, r_tmp.result],
                        )
                        CallOp([], "mul_r_gp", [r_tmp.result, gp])
                        CallOp([], "matmul_g_b_bf16", [g1d, v, gp])
                        c0_i32 = ConstantOp(i32, 0)
                        CallOp(
                            [],
                            "accum_sp_r_s",
                            [sp_buf, r_tmp.result, s_tmp.result],
                        )
                        CallOp(
                            [],
                            "vector_copy_32elems",
                            [c0_i32, s_tmp.result, sp_buf],
                        )
                        DeallocOp(s_tmp)
                        DeallocOp(r_tmp)
                        yield_([])

                    # === CASCADE MERGE (last/middle/first) ===
                    # Exactly matching step_test.py ordering.
                    set_first_stage = IntegerSet.get(
                        0, 2, [s0, s1 - c_ns_m1], [False, True]
                    )
                    set_middle_stage = IntegerSet.get(
                        0,
                        2,
                        [
                            AffineExpr.get_add(s1, AffineConstantExpr.get(-1)),
                            AffineExpr.get_add(
                                AffineConstantExpr.get(NS - 2),
                                AffineExpr.get_mul(s1, AffineConstantExpr.get(-1)),
                            ),
                            s0,
                            AffineExpr.get_add(
                                AffineConstantExpr.get(NQ - 1),
                                AffineExpr.get_mul(s0, AffineConstantExpr.get(-1)),
                            ),
                        ],
                        [False, False, False, False],
                    )
                    c1_h = ConstantOp(index_type, 1)

                    # Last stage (ty == NS-1): send cascade down
                    if_last = affine.AffineIfOp(
                        set_first_stage,
                        cond_operands=[tx, ty],
                        has_else=True,
                    )
                    with InsertionPoint(if_last.then_block):
                        subi_l = arith.SubIOp(ty, c1_h)
                        ChannelPut("cascade_gp", gp, indices=[tx, subi_l])
                        ChannelPut("cascade_up", up_buf, indices=[tx, subi_l])
                        ChannelPut("cascade_sp", sp_buf, indices=[tx, subi_l])
                        affine.AffineYieldOp([])

                    with InsertionPoint(if_last.else_block):
                        # Middle stages: 1 <= ty <= NS-2
                        if_mid = affine.AffineIfOp(
                            set_middle_stage,
                            cond_operands=[tx, ty],
                            has_else=True,
                        )
                        with InsertionPoint(if_mid.then_block):
                            gp_c = AllocOp(gp_l1_t, [], [])
                            up_c = AllocOp(up_l1_t, [], [])
                            sp_c = AllocOp(up_l1_t, [], [])
                            ChannelGet(
                                "cascade_gp",
                                gp_c.result,
                                indices=[tx, ty],
                            )
                            ChannelGet(
                                "cascade_up",
                                up_c.result,
                                indices=[tx, ty],
                            )
                            ChannelGet(
                                "cascade_sp",
                                sp_c.result,
                                indices=[tx, ty],
                            )
                            up_s = AllocOp(up_l1_t, [], [])
                            c0m = ConstantOp(i32, 0)
                            CallOp(
                                [],
                                "vector_copy_32elems",
                                [c0m, up_buf, up_s.result],
                            )
                            CallOp(
                                [],
                                "maximum_up_u_bf16",
                                [up_c.result, up_buf],
                            )
                            rc = AllocOp(up_l1_t, [], [])
                            CallOp(
                                [],
                                "exp_up_minus_u",
                                [up_c.result, up_buf, rc.result],
                            )
                            rl = AllocOp(up_l1_t, [], [])
                            CallOp(
                                [],
                                "exp_up_minus_u",
                                [up_s.result, up_buf, rl.result],
                            )
                            CallOp([], "mul_r_gp", [rc.result, gp_c.result])
                            CallOp([], "mul_r_gp", [rl.result, gp])
                            CallOp([], "add_gp_g", [gp, gp_c.result])
                            st = AllocOp(up_l1_t, [], [])
                            CallOp([], "zero_fill_sp_bf16", [st.result])
                            CallOp(
                                [],
                                "accum_sp_r_s",
                                [sp_c.result, rc.result, st.result],
                            )
                            CallOp(
                                [],
                                "accum_sp_r_s",
                                [sp_buf, rl.result, st.result],
                            )
                            CallOp(
                                [],
                                "vector_copy_32elems",
                                [c0m, st.result, sp_c.result],
                            )
                            subi_m = arith.SubIOp(ty, c1_h)
                            ChannelPut(
                                "cascade_gp",
                                gp_c.result,
                                indices=[tx, subi_m],
                            )
                            ChannelPut(
                                "cascade_up",
                                up_buf,
                                indices=[tx, subi_m],
                            )
                            ChannelPut(
                                "cascade_sp",
                                sp_c.result,
                                indices=[tx, subi_m],
                            )
                            DeallocOp(gp_c)
                            DeallocOp(up_c)
                            DeallocOp(sp_c)
                            DeallocOp(up_s)
                            DeallocOp(rc)
                            DeallocOp(rl)
                            DeallocOp(st)
                            affine.AffineYieldOp([])

                        with InsertionPoint(if_mid.else_block):
                            # First stage (ty == 0): cascade in, merge,
                            # div, output
                            gp_c2 = AllocOp(gp_l1_t, [], [])
                            up_c2 = AllocOp(up_l1_t, [], [])
                            sp_c2 = AllocOp(up_l1_t, [], [])
                            ChannelGet(
                                "cascade_gp",
                                gp_c2.result,
                                indices=[tx, ty],
                            )
                            ChannelGet(
                                "cascade_up",
                                up_c2.result,
                                indices=[tx, ty],
                            )
                            ChannelGet(
                                "cascade_sp",
                                sp_c2.result,
                                indices=[tx, ty],
                            )
                            up_s2 = AllocOp(up_l1_t, [], [])
                            c0f = ConstantOp(i32, 0)
                            CallOp(
                                [],
                                "vector_copy_32elems",
                                [c0f, up_buf, up_s2.result],
                            )
                            CallOp(
                                [],
                                "maximum_up_u_bf16",
                                [up_c2.result, up_buf],
                            )
                            rc2 = AllocOp(up_l1_t, [], [])
                            CallOp(
                                [],
                                "exp_up_minus_u",
                                [up_c2.result, up_buf, rc2.result],
                            )
                            rl2 = AllocOp(up_l1_t, [], [])
                            CallOp(
                                [],
                                "exp_up_minus_u",
                                [up_s2.result, up_buf, rl2.result],
                            )
                            CallOp(
                                [],
                                "mul_r_gp",
                                [rc2.result, gp_c2.result],
                            )
                            CallOp([], "mul_r_gp", [rl2.result, gp])
                            CallOp([], "add_gp_g", [gp, gp_c2.result])
                            st2 = AllocOp(up_l1_t, [], [])
                            CallOp([], "zero_fill_sp_bf16", [st2.result])
                            CallOp(
                                [],
                                "accum_sp_r_s",
                                [sp_c2.result, rc2.result, st2.result],
                            )
                            CallOp(
                                [],
                                "accum_sp_r_s",
                                [sp_buf, rl2.result, st2.result],
                            )
                            CallOp(
                                [],
                                "vector_copy_32elems",
                                [c0f, st2.result, sp_c2.result],
                            )
                            CallOp(
                                [],
                                "div_gp_sp",
                                [sp_c2.result, gp_c2.result],
                            )
                            c0_out = ConstantOp(index_type, 0)
                            ChannelPut(
                                "Gp2L2",
                                gp_c2.result,
                                indices=[tx, c0_out],
                                offsets=[0, 0, 0, 0],
                                sizes=[
                                    tile_size_q // M,
                                    M,
                                    dv_tile // M,
                                    M,
                                ],
                                strides=[
                                    M * M,
                                    M,
                                    tile_size_q * M,
                                    1,
                                ],
                            )
                            DeallocOp(gp_c2)
                            DeallocOp(up_c2)
                            DeallocOp(sp_c2)
                            DeallocOp(up_s2)
                            DeallocOp(rc2)
                            DeallocOp(rl2)
                            DeallocOp(st2)
                            affine.AffineYieldOp([])
                        affine.AffineYieldOp([])

                    # === CAUSAL COUNTER INCREMENT ===
                    # Only increment on the last dv_chunk iteration to avoid
                    # double-counting when dv_chunks > 1. Uses counter_buf[3]
                    # as a dv_iter counter that tracks position within the
                    # dv_chunks cycle.
                    if causal:

                        def _emit_counter_increment():
                            head_cur = load(counter_buf, [c2_ctr])
                            c1_i32_inc = ConstantOp(i32, 1)
                            head_next = arith.AddIOp(head_cur, c1_i32_inc)
                            total_hg = ConstantOp(i32, num_head_groups)
                            wrapped = arith.CmpIOp(
                                arith.CmpIPredicate.sge,
                                head_next.result,
                                total_hg,
                            )
                            if_wrap = scf.IfOp(wrapped)
                            with InsertionPoint(if_wrap.then_block):
                                q_cur = load(counter_buf, [c0_ctr])
                                c_nq_i32 = ConstantOp(i32, NQ)
                                q_next = arith.AddIOp(q_cur, c_nq_i32)
                                # The counter lives in UNINITIALIZED L1 that
                                # persists across host dispatches, and the boot
                                # flag only fires on zeroed memory -- so the
                                # q_block base must return to its boot value by
                                # the END of every complete execution, or the
                                # second dispatch runs with q_base past every
                                # kv block and apply_causal_mask never fires
                                # (measured: unmasked attention from dispatch 2
                                # on, with q/k/v all clean). Within one
                                # execution q_next never exceeds the total, so
                                # the wrap changes no in-flight value.
                                c_total_q = ConstantOp(i32, num_lq_iters * NQ)
                                q_wrapped = arith.RemSIOp(q_next.result, c_total_q)
                                store(q_wrapped.result, counter_buf, [c0_ctr])
                                store(ConstantOp(i32, 0), counter_buf, [c2_ctr])
                                scf.YieldOp([])
                            not_wrapped = arith.CmpIOp(
                                arith.CmpIPredicate.slt,
                                head_next.result,
                                total_hg,
                            )
                            if_no_wrap = scf.IfOp(not_wrapped)
                            with InsertionPoint(if_no_wrap.then_block):
                                store(head_next.result, counter_buf, [c2_ctr])
                                scf.YieldOp([])

                        if dv_chunks > 1:
                            # Use counter_buf[3] as dv_iter counter
                            dv_iter_cur = load(counter_buf, [c3_ctr])
                            c_dv_last_i32 = ConstantOp(i32, dv_chunks - 1)
                            is_last_dv = arith.CmpIOp(
                                arith.CmpIPredicate.sge,
                                dv_iter_cur,
                                c_dv_last_i32,
                            )
                            if_last_dv = scf.IfOp(is_last_dv)
                            with InsertionPoint(if_last_dv.then_block):
                                _emit_counter_increment()
                                # Reset dv_iter counter
                                store(
                                    ConstantOp(i32, 0),
                                    counter_buf,
                                    [c3_ctr],
                                )
                                scf.YieldOp([])
                            # If not last dv_chunk, just increment dv_iter
                            not_last_dv = arith.CmpIOp(
                                arith.CmpIPredicate.slt,
                                dv_iter_cur,
                                c_dv_last_i32,
                            )
                            if_not_last = scf.IfOp(not_last_dv)
                            with InsertionPoint(if_not_last.then_block):
                                c1_i32_dv = ConstantOp(i32, 1)
                                dv_next = arith.AddIOp(dv_iter_cur, c1_i32_dv)
                                store(
                                    dv_next.result,
                                    counter_buf,
                                    [c3_ctr],
                                )
                                scf.YieldOp([])
                        else:
                            _emit_counter_increment()

                # Output gather from ty=0 tiles
                affine_map_col = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0),
                            AffineConstantExpr.get(tile_size_q),
                        )
                    ],
                )
                par_out = scf.ForallOp(lower_bounds=[0], upper_bounds=[NQ], steps=[1])
                with InsertionPoint(par_out.body):
                    apply_off = affine_apply(
                        affine_map_col,
                        [par_out.induction_variables[0]],
                    )
                    ChannelGet(
                        "Gp2L2",
                        gp_l2.result,
                        indices=[par_out.induction_variables[0], 0],
                        offsets=[apply_off, 0],
                        sizes=[tile_size_q, dv_tile],
                        strides=[dv_tile, 1],
                    )
                    scf.InParallelOp()

                # Output: L2-to-L3
                ChannelPut("GpOut", gp_l2.result, indices=[seg_x])

                # Deallocs for segment-level buffers
                for q_buf in q_saved_bufs:
                    DeallocOp(q_buf)
                DeallocOp(qk_buf)
                DeallocOp(v_l1)
                DeallocOp(g_l1)
                DeallocOp(gp_l1)
                DeallocOp(up_l1)
                DeallocOp(sp_l1)
                for stage in range(NS):
                    DeallocOp(v_l2_bufs[stage])
                for stage in range(NS):
                    DeallocOp(qk_l2_bufs[stage])
                DeallocOp(gp_l2)
                if causal:
                    DeallocOp(causal_ctr)

            # Output gets: one per head in the unroll group, placed after the
            # producing @segment so source order encodes producer→consumer.
            for head_local in range(num_heads_per_unroll):
                if head_local == 0:
                    head_idx = head_base
                else:
                    head_idx = affine_apply(affine_map_plus1, [head_base])
                head_out_off = affine_apply(affine_map_head_out_dv, [head_idx, lz])
                head_offset_idx = ConstantOp(index_type, head_local)
                out_combined = affine_apply(
                    affine_map_add, [head_out_off, out_launch_off]
                )
                ChannelGet(
                    "GpOut",
                    gp,
                    indices=[head_offset_idx],
                    offsets=[out_combined],
                    sizes=[lqp * dv_tile],
                    strides=[1],
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="attn_npu2.py",
        description="Flash attention with memtile-relayed L3-to-L1 Q/K/V — "
        "selective Q capture",
    )
    parser.add_argument(
        "-p",
        "--print-module-only",
        action="store_true",
        help="Print MLIR module and exit",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--lk",
        type=int,
        default=512,
        help="Total K/V sequence length (default: 512)",
    )
    parser.add_argument(
        "--lq",
        type=int,
        default=512,
        help="Total Q sequence length (default: 512)",
    )
    parser.add_argument(
        "--lqp",
        type=int,
        default=256,
        help="Q chunk size per launch iteration (default: 256)",
    )
    parser.add_argument(
        "--lkp",
        type=int,
        default=64,
        help="K/V chunk size per tile (default: 64)",
    )
    parser.add_argument(
        "--dk",
        type=int,
        default=64,
        help="Key dimension (default: 64). Must be divisible by lkp.",
    )
    parser.add_argument(
        "--dv",
        type=int,
        default=64,
        help="Value dimension (default: 64). Must be divisible by lkp.",
    )
    parser.add_argument(
        "--num-cascade-stages",
        type=int,
        default=4,
        help="Number of cascade pipeline stages (default: 4)",
    )
    parser.add_argument(
        "--num-q-tiles",
        type=int,
        default=4,
        dest="num_q_tiles",
        help="Number of tiles to partition the Q chunk into (default: 4). "
        "Under causal masking, lqp / num_q_tiles must equal lkp.",
    )
    parser.add_argument(
        "--num-heads-per-unroll",
        type=int,
        default=2,
        dest="num_heads_per_unroll",
        help="Heads processed per segment instance (default: 2). "
        "Physical columns = num_heads_per_unroll * num_q_tiles.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=2,
        help="Number of attention heads (default: 2)",
    )
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        default=None,
        help="Number of KV heads (default: num_heads for MHA, " "< num_heads for GQA)",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="compile-and-run",
        choices=["compile-only", "compile-and-run"],
        help="Compilation mode (default: compile-and-run)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="elf",
        choices=["xclbin", "elf"],
        help="Output format (default: elf)",
    )
    parser.add_argument(
        "--causal",
        action="store_true",
        help="Enable causal masking (autoregressive attention)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=0,
        help="Sliding-window length W (default 0 = full causal). W > 0 keeps "
        "q_abs - W < k_abs <= q_abs. MUST match the -DWINDOW_LEN the kernel "
        "object was compiled with; the canonical Makefile drives both from "
        "WINDOW=. A mismatch fails the correctness check, it cannot pass.",
    )
    parser.add_argument(
        "--expect-failure",
        action="store_true",
        dest="expect_failure",
        help="Negative control for the windowed gate. Inverts the verdict: exit "
        "0 if and only if the comparison RAN and REJECTED the run. Intended "
        "use is a banded reference (--window W) against a kernel object built "
        "WITHOUT the band (EXTRA_KERNEL_FLAGS=-DWINDOW_LEN=0), which must be "
        "rejected -- otherwise the windowed gate proves nothing.",
    )
    parser.add_argument(
        "--perf-iters",
        type=int,
        default=0,
        dest="perf_iters",
        help="If >0, time the kernel over this many iters (after 10 warmup) and "
        "print Latency + GFLOPs in addition to the correctness check",
    )
    args = parser.parse_args()

    if args.perf_iters < 0:
        parser.error("--perf-iters must be >= 0")
    if args.num_q_tiles < 1:
        parser.error("--num-q-tiles must be >= 1")
    if args.num_heads_per_unroll < 1:
        parser.error("--num-heads-per-unroll must be >= 1")
    if args.window < 0:
        parser.error("--window must be >= 0")
    if args.window > 0 and not args.causal:
        parser.error("--window requires --causal (the band is a two-sided causal mask)")
    # --expect-failure must FAIL CLOSED. A flag that is parsed and then never
    # branched on is how an off-queue hardware dispatch happened here before:
    # refuse every combination in which the inversion would silently do nothing
    # rather than accept it and report a verdict nobody computed.
    if args.expect_failure:
        if args.window <= 0:
            parser.error(
                "--expect-failure requires --window > 0: the control's whole "
                "content is that a BANDED reference rejects an unwindowed kernel"
            )
        if args.compile_mode != "compile-and-run":
            parser.error(
                f"--expect-failure requires --compile-mode compile-and-run, got "
                f"'{args.compile_mode}': there is no comparison to invert otherwise"
            )
        if args.print_module_only:
            parser.error("--expect-failure is meaningless with -p/--print-module-only")

    lk = args.lk
    lkp = args.lkp
    lq = args.lq
    lqp = args.lqp
    dk = args.dk
    dv = args.dv
    num_cascade_stages = args.num_cascade_stages
    num_q_tiles = args.num_q_tiles
    num_heads_per_unroll = args.num_heads_per_unroll
    num_heads = args.num_heads
    num_kv_heads = args.num_kv_heads if args.num_kv_heads is not None else num_heads
    causal = args.causal
    window = args.window
    gqa_group_size = num_heads // num_kv_heads

    mlir_module = build_module(
        lk=lk,
        lkp=lkp,
        lq=lq,
        lqp=lqp,
        dk=dk,
        dv=dv,
        num_q_tiles=num_q_tiles,
        num_cascade_stages=num_cascade_stages,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        causal=causal,
        num_heads_per_unroll=num_heads_per_unroll,
        window=window,
    )

    if args.print_module_only:
        print(mlir_module)
        exit(0)

    from air.backend.xrt_runner import XRTRunner
    from air.backend.xrt import XRTBackend
    from ml_dtypes import bfloat16

    INPUT_DATATYPE = OUTPUT_DATATYPE = bfloat16
    rng = np.random.default_rng(42)
    # Use N(0,1) (matching the GPU SDPA test standard — PyTorch uses randn) so
    # the correctness check sees a realistic signed distribution rather than an
    # all-positive one.
    input_q = rng.standard_normal((num_heads, lq, dk)).astype(INPUT_DATATYPE)
    input_k = rng.standard_normal((num_kv_heads, lk, dk)).astype(INPUT_DATATYPE)
    input_v_orig = rng.standard_normal((num_kv_heads, lk, dv)).astype(INPUT_DATATYPE)
    # Transpose V to [num_kv_heads * dv_chunks, lk, dv_tile] for contiguous access
    dv_chunks_host = dv // lkp
    input_v = (
        input_v_orig.reshape(num_kv_heads, lk, dv_chunks_host, lkp)
        .transpose(0, 2, 1, 3)
        .reshape(num_kv_heads * dv_chunks_host, lk, lkp)
        .copy()
    )

    inv_sqrt_dk = 1.0 / sqrt(dk)
    sdpa_output = np.zeros((num_heads, lq, dv), dtype=OUTPUT_DATATYPE)
    for h in range(num_heads):
        kv_h = h // gqa_group_size
        Qf = input_q[h].astype(np.float32)
        Kf = input_k[kv_h].astype(np.float32)
        Vf = input_v_orig[kv_h].astype(np.float32)
        scores = Qf @ Kf.T * inv_sqrt_dk
        if causal:
            # Upper triangle: k_abs > q_abs.
            mask = np.triu(np.ones(scores.shape, dtype=bool), k=1)
            if window > 0:
                # Lower band edge: k_abs <= q_abs - window, i.e. row-col >= window.
                # The diagonal is always kept, so no row is ever wholly masked.
                mask |= np.tril(np.ones(scores.shape, dtype=bool), k=-window)
            scores = np.where(mask, -1e9, scores)
        mx = np.max(scores, axis=-1, keepdims=True)
        P = np.exp(scores - mx)
        P = P / np.sum(P, axis=-1, keepdims=True)
        sdpa_output[h] = (P @ Vf).astype(OUTPUT_DATATYPE)

    # Transpose expected output to match transposed L3 layout
    sdpa_output_transposed = (
        sdpa_output.reshape(num_heads, lq, dv_chunks_host, lkp)
        .transpose(0, 2, 1, 3)
        .reshape(num_heads * dv_chunks_host, lq, lkp)
        .copy()
    )

    tiling = [1, 1, 1] if dv_chunks_host > 1 else [1, 1]
    # FLOPs for attention: Q@K^T scales with dk, P@V scales with dv (each is
    # 2*num_heads*lq*lk*<dim>), so total = 2*num_heads*lq*lk*(dk+dv). Causal
    # masking roughly halves the effective work.
    perf_flops = 2.0 * num_heads * lq * lk * (dk + dv)
    if causal:
        perf_flops *= 0.5
    # NOTE (windowing): perf_flops is a CONVENTION, not executed work. The KV
    # chunk loop bound is a static constant (see the c_chunks_h loop above), so
    # a windowed run multiplies exactly as many elements as an unwindowed one
    # and then masks more of them. Deliberately NOT scaled by the band ratio:
    # doing so would report a speedup that does not exist. Any GFLOP/s quoted
    # for window > 0 is therefore already inflated by the causal 0.5 above, and
    # must not be inflated again.
    backend_opts = dict(
        omit_while_true_loop=False,
        omit_pingpong="all",
        verbose=args.verbose,
        runtime_loop_tiling_sizes=tiling,
        output_format=args.output_format,
        instance_name="attention_bf16",
        target_device="npu2",
        report_precision=True,
        n_perf_iters=args.perf_iters,
        perf_flops=(perf_flops if args.perf_iters > 0 else None),
    )

    RTOL, ATOL = 1.6e-2, 1e-1

    class _RecordingRunner(XRTRunner):
        """XRTRunner that copies out the check's statistics as it passes.

        Same idiom as transformer_layer/opcheck.py:151. The VERDICT is
        unchanged -- ``_check_outputs`` records and then returns
        ``super()._check_outputs(...)``, so ``n_mismatch`` agrees with the
        verdict by construction rather than by assertion.

        Two things this buys that the stock runner does not print:

        - ``atol_required`` = ``max(|a - e| - rtol * |e|)``, the smallest atol
          the run would have passed at. It is what makes an atol choice
          checkable, and this datapath needs it: ``1e-1`` is a hard ceiling and
          FlashAttention's honest error already gets within a factor of two of
          it. ``abs_err max`` alone overstates the danger, because under a
          causal (and far more so a banded) mask the largest absolute error
          lands on a large-magnitude element that ``rtol`` already covers --
          early rows attend to a handful of keys, so |y| runs large.
        - ``n_mismatch``, which is what lets the negative control below assert
          that the TOLERANCE CHECK is what rejected the run.
        """

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.stats = None

        def _check_outputs(
            self,
            actual_outputs,
            expected_outputs,
            rtol=1e-3,
            atol=1e-8,
            max_mismatch_percentage=0,
            min_correlation=None,
        ):
            n_elements = n_mismatch = 0
            abs_err_sum = ref_abs_sum = 0.0
            abs_err_max = atol_required = 0.0
            for actual, expected in zip(actual_outputs, expected_outputs):
                a = np.reshape(actual, expected.shape).astype(np.float64)
                e = np.asarray(expected).astype(np.float64)
                abs_err = np.abs(a - e)
                n_elements += e.size
                n_mismatch += int(
                    np.count_nonzero(~np.isclose(a, e, rtol=rtol, atol=atol))
                )
                abs_err_sum += float(abs_err.sum())
                ref_abs_sum += float(np.abs(e).sum())
                abs_err_max = max(abs_err_max, float(abs_err.max()))
                atol_required = max(
                    atol_required, float((abs_err - rtol * np.abs(e)).max())
                )
            self.stats = {
                "mean_rel_L1": abs_err_sum / (ref_abs_sum + 1e-30),
                "abs_err_max": abs_err_max,
                "atol_required": max(atol_required, 0.0),
                "n_elements": n_elements,
                "n_mismatch": n_mismatch,
            }
            return super()._check_outputs(
                actual_outputs=actual_outputs,
                expected_outputs=expected_outputs,
                rtol=rtol,
                atol=atol,
                max_mismatch_percentage=max_mismatch_percentage,
                min_correlation=min_correlation,
            )

    def _negative_control_verdict(rc, stats):
        """Exit status for --expect-failure. 0 only on EVIDENCE the banded
        reference did the rejecting.

        "The process exited non-zero" is not that evidence: a missing
        PEANO_INSTALL_DIR, a kernel that fails to link and an NPU that never
        comes up all exit non-zero without ever comparing anything, and a
        caller that simply inverted the exit status would report a passing
        negative control for every one of them. So this reads the completed
        comparison's own statistics, exactly as
        transformer_layer/opcheck.py:414 does for fault injection.
        """
        problems = []
        if stats is None:
            problems.append(
                "no comparison statistics were recorded -- the run never reached "
                "the output check, so nothing was rejected"
            )
        else:
            if stats["n_elements"] <= 0:
                problems.append("the comparison saw 0 elements")
            if rc == 0:
                problems.append(
                    f"the run PASSED against a W={window} banded reference, so the "
                    f"gate cannot tell a windowed kernel from an unwindowed one"
                )
            elif stats["n_mismatch"] <= 0:
                problems.append(
                    f"the run FAILED with n_mismatch={stats['n_mismatch']}, so the "
                    f"tolerance check is not what rejected it"
                )
        if problems:
            print("NEGATIVE CONTROL: FAIL")
            for p in problems:
                print(f"  {p}")
            return 1
        pct = 100.0 * stats["n_mismatch"] / stats["n_elements"]
        print(
            f"NEGATIVE CONTROL: PASS (banded W={window} reference rejected the "
            f"unwindowed kernel on {stats['n_mismatch']}/{stats['n_elements']} "
            f"elements = {pct:.1f}%, as required)"
        )
        return 0

    if args.compile_mode == "compile-and-run":
        if window > 0:
            print(
                f"[window] reference band W={window}: keeping "
                f"q_abs - {window} < k_abs <= q_abs. The KERNEL OBJECT must have "
                f"been compiled -DWINDOW_LEN={window} to match; if it was not, "
                f"this check FAILS (it has no silent-pass mode)."
            )
        runner = _RecordingRunner(**backend_opts)
        rc = runner.run_test(
            mlir_module,
            inputs=[input_q, input_k, input_v],
            expected_outputs=[sdpa_output_transposed],
            rtol=RTOL,
            atol=ATOL,
        )
        st = runner.stats
        if st is not None:
            margin = ATOL / st["atol_required"] if st["atol_required"] > 0 else float("inf")
            print(
                f"[precision] atol_required={st['atol_required']:.3e} "
                f"| atol={ATOL:.1e} | margin={margin:.2f}x under the 1e-1 ceiling "
                f"| mean_rel_L1={st['mean_rel_L1']:.3e} "
                f"| mismatches={st['n_mismatch']}/{st['n_elements']}"
            )
        if args.expect_failure:
            exit(_negative_control_verdict(rc, st))
        exit(rc)
    elif args.compile_mode == "compile-only":
        backend = XRTBackend(**backend_opts)
        module_function = backend.compile(mlir_module)
        print("Compilation complete.")
