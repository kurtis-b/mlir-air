# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared AIR generator for FlowQKV and FlowKV attention kernels."""

from __future__ import annotations

from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects import arith, linalg
from air.dialects.arith import ConstantOp
from air.dialects.func import FuncOp, CallOp
from air.dialects.memref import AllocOp, DeallocOp, subview
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner, type_mapper


def _affine_mul(factor):
    return AffineMap.get(
        0,
        1,
        [AffineExpr.get_mul(AffineSymbolExpr.get(0), AffineConstantExpr.get(factor))],
    )


def _affine_linear_tile(herd_cols):
    return AffineMap.get(
        0,
        3,
        [
            AffineExpr.get_add(
                AffineSymbolExpr.get(0),
                AffineExpr.get_add(
                    AffineExpr.get_mul(
                        AffineSymbolExpr.get(1), AffineConstantExpr.get(herd_cols)
                    ),
                    AffineSymbolExpr.get(2),
                ),
            )
        ],
    )


def _affine_linear_local(herd_cols):
    return AffineMap.get(
        0,
        2,
        [
            AffineExpr.get_add(
                AffineExpr.get_mul(
                    AffineSymbolExpr.get(0), AffineConstantExpr.get(herd_cols)
                ),
                AffineSymbolExpr.get(1),
            )
        ],
    )


def _affine_kv_group(herd_cols, heads_per_kv, kv_groups):
    linear = AffineExpr.get_add(
        AffineExpr.get_mul(AffineSymbolExpr.get(0), AffineConstantExpr.get(herd_cols)),
        AffineSymbolExpr.get(1),
    )
    rem = AffineExpr.get_mod(
        linear, AffineConstantExpr.get(kv_groups * heads_per_kv)
    )
    return AffineMap.get(
        0,
        2,
        [AffineExpr.get_floor_div(rem, AffineConstantExpr.get(heads_per_kv))],
    )


def _affine_query_base(q_chunk, tiles_per_query_chunk, base_query=0):
    q_slot = AffineExpr.get_floor_div(
        AffineSymbolExpr.get(0), AffineConstantExpr.get(tiles_per_query_chunk)
    )
    query_base = AffineExpr.get_add(
        AffineConstantExpr.get(base_query),
        AffineExpr.get_mul(q_slot, AffineConstantExpr.get(q_chunk)),
    )
    return AffineMap.get(0, 1, [query_base])


def _i32_constant(value):
    i32_type = T.i32()
    return ConstantOp(i32_type, IntegerAttr.get(i32_type, value))


@module_builder
def build_flow_module(
    q_chunk,
    kv_len,
    head_dim,
    kernel_func,
    object_file,
    public_name,
    groups=1,
    herd_rows=1,
    herd_cols=1,
    output_mode="direct",
    l2_gather_layout="rowcol",
    dynamic_query_base=False,
    query_base=0,
    tiles_per_query_chunk=None,
    l2_gather_via_channel=False,
):
    if output_mode not in ("direct", "l2-gather"):
        raise ValueError(f"unsupported output mode: {output_mode}")
    if l2_gather_layout not in ("rowcol", "linear"):
        raise ValueError(f"unsupported L2 gather layout: {l2_gather_layout}")
    use_l2_gather = output_mode == "l2-gather"
    if l2_gather_via_channel and not use_l2_gather:
        raise ValueError("channel-staged L2 gather requires output mode l2-gather")
    use_rowcol_l2_gather = use_l2_gather and l2_gather_layout == "rowcol"
    use_l2_gather_via_channel = use_l2_gather and l2_gather_via_channel
    if dynamic_query_base and not tiles_per_query_chunk:
        raise ValueError("dynamic query-base mode needs tiles_per_query_chunk")

    bf16_type = type_mapper(bfloat16)
    herd_tiles = herd_rows * herd_cols
    qkv_rows = q_chunk + 2 * kv_len

    qkv_l3_ty = MemRefType.get(
        [groups // herd_cols, herd_cols, qkv_rows, head_dim], bf16_type
    )
    out_l3_ty = MemRefType.get([groups, q_chunk, head_dim], bf16_type)

    l2_space = IntegerAttr.get(T.i32(), MemorySpace.L2)
    qkv_l2_ty = MemRefType.get(
        [herd_rows, herd_cols, qkv_rows, head_dim],
        bf16_type,
        memory_space=l2_space,
    )
    if use_l2_gather:
        if use_rowcol_l2_gather:
            out_l2_ty = MemRefType.get(
                [herd_rows, herd_cols, q_chunk, head_dim],
                bf16_type,
                memory_space=l2_space,
            )
        else:
            out_l2_ty = MemRefType.get(
                [herd_tiles, q_chunk, head_dim], bf16_type, memory_space=l2_space
            )

    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    qkv_l1_ty = MemRefType.get([qkv_rows, head_dim], bf16_type, memory_space=l1_space)
    q_l1_ty = MemRefType.get([q_chunk, head_dim], bf16_type, memory_space=l1_space)
    kv_l1_ty = MemRefType.get([kv_len, head_dim], bf16_type, memory_space=l1_space)
    out_l1_ty = MemRefType.get([q_chunk, head_dim], bf16_type, memory_space=l1_space)

    attn_args = [q_l1_ty, kv_l1_ty, kv_l1_ty, out_l1_ty]
    if dynamic_query_base:
        attn_args = [T.i32(), q_l1_ty, kv_l1_ty, kv_l1_ty, out_l1_ty]
    attn_func = FuncOp(
        kernel_func,
        (attn_args, []),
        visibility="private",
    )
    attn_func.attributes["link_with"] = StringAttr.get(object_file)
    attn_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    launch_offset_map = _affine_mul(herd_tiles)
    launch_row_offset_map = _affine_mul(herd_rows)
    group_map = _affine_linear_tile(herd_cols)
    local_map = _affine_linear_local(herd_cols)
    query_base_map = None
    if dynamic_query_base:
        query_base_map = _affine_query_base(q_chunk, tiles_per_query_chunk, query_base)
    out_channel = None
    if use_l2_gather_via_channel:
        out_channel = f"{public_name}_l2_gather_out"
        Channel(out_channel, size=[herd_rows, herd_cols])

    @FuncOp.from_py_func(qkv_l3_ty, out_l3_ty)
    def flow_attention(arg_qkv, arg_out):
        @launch(
            operands=[arg_qkv, arg_out],
            sizes=[groups // herd_tiles, 1],
        )
        def launch_body(lx, _ly, _lsx, _lsy, lqkv, lo):
            launch_base = affine_apply(launch_offset_map, [lx])
            launch_row_base = affine_apply(launch_row_offset_map, [lx])

            @segment(
                name=f"{public_name}_seg", operands=[launch_base, launch_row_base, lqkv, lo]
            )
            def segment_body(base, row_base, sqkv, so):
                l2_qkv = AllocOp(qkv_l2_ty, [], [])
                if use_l2_gather:
                    l2_out = AllocOp(out_l2_ty, [], [])

                dma_memcpy_nd(
                    l2_qkv,
                    sqkv,
                    src_offsets=[row_base, 0, 0, 0],
                    src_sizes=[herd_rows, herd_cols, qkv_rows, head_dim],
                    src_strides=[
                        herd_cols * qkv_rows * head_dim,
                        qkv_rows * head_dim,
                        head_dim,
                        1,
                    ],
                )

                @herd(
                    name=f"{public_name}_herd",
                    sizes=[herd_rows, herd_cols],
                    operands=[base, l2_qkv, l2_out if use_l2_gather else so],
                    link_with=object_file,
                )
                def herd_body(_tx, _ty, _sx, _sy, bbase, hqkv, ho):
                    group_idx = affine_apply(group_map, [bbase, _tx, _ty])
                    local_idx = affine_apply(local_map, [_tx, _ty])

                    l1_qkv = AllocOp(qkv_l1_ty, [], [])
                    l1_q = AllocOp(q_l1_ty, [], [])
                    l1_k = AllocOp(kv_l1_ty, [], [])
                    l1_v = AllocOp(kv_l1_ty, [], [])
                    l1_out = AllocOp(out_l1_ty, [], [])

                    dma_memcpy_nd(
                        l1_qkv,
                        hqkv,
                        src_offsets=[_tx, _ty, 0, 0],
                        src_sizes=[1, 1, qkv_rows, head_dim],
                        src_strides=[
                            herd_cols * qkv_rows * head_dim,
                            qkv_rows * head_dim,
                            head_dim,
                            1,
                        ],
                    )

                    q_src = subview(
                        l1_qkv.result,
                        [0, 0],
                        [q_chunk, head_dim],
                        [1, 1],
                    )
                    k_src = subview(
                        l1_qkv.result,
                        [q_chunk, 0],
                        [kv_len, head_dim],
                        [1, 1],
                    )
                    v_src = subview(
                        l1_qkv.result,
                        [q_chunk + kv_len, 0],
                        [kv_len, head_dim],
                        [1, 1],
                    )
                    linalg.copy(q_src, outs=[l1_q])
                    linalg.copy(k_src, outs=[l1_k])
                    linalg.copy(v_src, outs=[l1_v])

                    if dynamic_query_base:
                        qbase_idx = affine_apply(query_base_map, [group_idx])
                        qbase_i32 = arith.IndexCastOp(T.i32(), qbase_idx).result
                        CallOp(attn_func, [qbase_i32, l1_q, l1_k, l1_v, l1_out])
                    else:
                        CallOp(attn_func, [l1_q, l1_k, l1_v, l1_out])
                    if use_l2_gather:
                        if use_l2_gather_via_channel:
                            ChannelPut(out_channel, l1_out, indices=[_tx, _ty])
                        elif use_rowcol_l2_gather:
                            dma_memcpy_nd(
                                ho,
                                l1_out,
                                dst_offsets=[_tx, _ty, 0, 0],
                                dst_sizes=[1, 1, q_chunk, head_dim],
                                dst_strides=[
                                    herd_cols * q_chunk * head_dim,
                                    q_chunk * head_dim,
                                    head_dim,
                                    1,
                                ],
                            )
                        else:
                            dma_memcpy_nd(
                                ho,
                                l1_out,
                                dst_offsets=[local_idx, 0, 0],
                                dst_sizes=[1, q_chunk, head_dim],
                                dst_strides=[q_chunk * head_dim, head_dim, 1],
                            )
                    else:
                        dma_memcpy_nd(
                            ho,
                            l1_out,
                            dst_offsets=[group_idx, 0, 0],
                            dst_sizes=[1, q_chunk, head_dim],
                            dst_strides=[q_chunk * head_dim, head_dim, 1],
                        )

                    DeallocOp(l1_qkv)
                    DeallocOp(l1_q)
                    DeallocOp(l1_k)
                    DeallocOp(l1_v)
                    DeallocOp(l1_out)

                if use_l2_gather:
                    if use_l2_gather_via_channel:
                        for row in range(herd_rows):
                            for col in range(herd_cols):
                                if use_rowcol_l2_gather:
                                    ChannelGet(
                                        out_channel,
                                        l2_out,
                                        offsets=[row, col, 0, 0],
                                        sizes=[1, 1, q_chunk, head_dim],
                                        strides=[
                                            herd_cols * q_chunk * head_dim,
                                            q_chunk * head_dim,
                                            head_dim,
                                            1,
                                        ],
                                        indices=[row, col],
                                    )
                                else:
                                    local_idx_static = row * herd_cols + col
                                    ChannelGet(
                                        out_channel,
                                        l2_out,
                                        offsets=[local_idx_static, 0, 0],
                                        sizes=[1, q_chunk, head_dim],
                                        strides=[q_chunk * head_dim, head_dim, 1],
                                        indices=[row, col],
                                    )
                    if use_rowcol_l2_gather:
                        dma_memcpy_nd(
                            so,
                            l2_out,
                            dst_offsets=[base, 0, 0],
                            dst_sizes=[herd_tiles, q_chunk, head_dim],
                            dst_strides=[q_chunk * head_dim, head_dim, 1],
                            src_offsets=[0, 0, 0, 0],
                            src_sizes=[herd_rows, herd_cols, q_chunk, head_dim],
                            src_strides=[
                                herd_cols * q_chunk * head_dim,
                                q_chunk * head_dim,
                                head_dim,
                                1,
                            ],
                        )
                    else:
                        dma_memcpy_nd(
                            so,
                            l2_out,
                            dst_offsets=[base, 0, 0],
                            dst_sizes=[herd_tiles, q_chunk, head_dim],
                            dst_strides=[q_chunk * head_dim, head_dim, 1],
                        )
                    DeallocOp(l2_out)

                DeallocOp(l2_qkv)


@module_builder
def build_flow_paper_module(
    q_chunk,
    kv_len,
    head_dim,
    kernel_func,
    object_file,
    public_name,
    kv_groups=4,
    heads_per_kv=2,
    herd_rows=8,
    herd_cols=4,
    output_mode="l2-gather",
    dynamic_query_base=False,
    query_base=0,
):
    if output_mode not in ("direct", "l2-gather"):
        raise ValueError(f"unsupported output mode: {output_mode}")
    if kv_groups * heads_per_kv > herd_rows * herd_cols:
        raise ValueError("kv-groups*heads-per-kv must fit in the physical herd")
    use_l2_gather = output_mode == "l2-gather"
    tiles_per_query_chunk = kv_groups * heads_per_kv

    bf16_type = type_mapper(bfloat16)
    q_l3_ty = MemRefType.get([herd_rows, herd_cols, q_chunk, head_dim], bf16_type)
    kv_l3_ty = MemRefType.get([kv_groups, kv_len, head_dim], bf16_type)
    out_l3_ty = MemRefType.get([herd_rows, herd_cols, q_chunk, head_dim], bf16_type)

    l2_space = IntegerAttr.get(T.i32(), MemorySpace.L2)
    q_l2_ty = MemRefType.get(
        [herd_rows, herd_cols, q_chunk, head_dim],
        bf16_type,
        memory_space=l2_space,
    )
    kv_l2_ty = MemRefType.get(
        [kv_groups, kv_len, head_dim], bf16_type, memory_space=l2_space
    )
    if use_l2_gather:
        out_l2_ty = MemRefType.get(
            [herd_rows, herd_cols, q_chunk, head_dim],
            bf16_type,
            memory_space=l2_space,
        )

    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    q_l1_ty = MemRefType.get([q_chunk, head_dim], bf16_type, memory_space=l1_space)
    kv_l1_ty = MemRefType.get([kv_len, head_dim], bf16_type, memory_space=l1_space)
    out_l1_ty = MemRefType.get([q_chunk, head_dim], bf16_type, memory_space=l1_space)

    attn_args = [q_l1_ty, kv_l1_ty, kv_l1_ty, out_l1_ty]
    if dynamic_query_base:
        attn_args = [T.i32(), q_l1_ty, kv_l1_ty, kv_l1_ty, out_l1_ty]
    attn_func = FuncOp(
        kernel_func,
        (attn_args, []),
        visibility="private",
    )
    attn_func.attributes["link_with"] = StringAttr.get(object_file)
    attn_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    kv_group_map = _affine_kv_group(herd_cols, heads_per_kv, kv_groups)
    local_map = _affine_linear_local(herd_cols)
    query_base_map = None
    if dynamic_query_base:
        query_base_map = _affine_query_base(q_chunk, tiles_per_query_chunk, query_base)

    @FuncOp.from_py_func(q_l3_ty, kv_l3_ty, kv_l3_ty, out_l3_ty)
    def flow_attention_paper(arg_q, arg_k, arg_v, arg_out):
        @launch(operands=[arg_q, arg_k, arg_v, arg_out], sizes=[1, 1])
        def launch_body(_lx, _ly, _lsx, _lsy, lq, lk, lv, lo):
            @segment(name=f"{public_name}_paper_seg", operands=[lq, lk, lv, lo])
            def segment_body(sq, sk, sv, so):
                l2_q = AllocOp(q_l2_ty, [], [])
                l2_k = AllocOp(kv_l2_ty, [], [])
                l2_v = AllocOp(kv_l2_ty, [], [])
                if use_l2_gather:
                    l2_out = AllocOp(out_l2_ty, [], [])

                dma_memcpy_nd(l2_q, sq)
                dma_memcpy_nd(l2_k, sk)
                dma_memcpy_nd(l2_v, sv)

                @herd(
                    name=f"{public_name}_paper_herd",
                    sizes=[herd_rows, herd_cols],
                    operands=[l2_q, l2_k, l2_v, l2_out if use_l2_gather else so],
                    link_with=object_file,
                )
                def herd_body(_tx, _ty, _sx, _sy, hq, hk, hv, ho):
                    kv_group = affine_apply(kv_group_map, [_tx, _ty])

                    l1_q = AllocOp(q_l1_ty, [], [])
                    l1_k = AllocOp(kv_l1_ty, [], [])
                    l1_v = AllocOp(kv_l1_ty, [], [])
                    l1_out = AllocOp(out_l1_ty, [], [])

                    dma_memcpy_nd(
                        l1_q,
                        hq,
                        src_offsets=[_tx, _ty, 0, 0],
                        src_sizes=[1, 1, q_chunk, head_dim],
                        src_strides=[
                            herd_cols * q_chunk * head_dim,
                            q_chunk * head_dim,
                            head_dim,
                            1,
                        ],
                    )
                    dma_memcpy_nd(
                        l1_k,
                        hk,
                        src_offsets=[kv_group, 0, 0],
                        src_sizes=[1, kv_len, head_dim],
                        src_strides=[kv_len * head_dim, head_dim, 1],
                    )
                    dma_memcpy_nd(
                        l1_v,
                        hv,
                        src_offsets=[kv_group, 0, 0],
                        src_sizes=[1, kv_len, head_dim],
                        src_strides=[kv_len * head_dim, head_dim, 1],
                    )
                    if dynamic_query_base:
                        local_idx = affine_apply(local_map, [_tx, _ty])
                        qbase_idx = affine_apply(query_base_map, [local_idx])
                        qbase_i32 = arith.IndexCastOp(T.i32(), qbase_idx).result
                        CallOp(attn_func, [qbase_i32, l1_q, l1_k, l1_v, l1_out])
                    else:
                        CallOp(attn_func, [l1_q, l1_k, l1_v, l1_out])

                    if use_l2_gather:
                        dma_memcpy_nd(
                            ho,
                            l1_out,
                            dst_offsets=[_tx, _ty, 0, 0],
                            dst_sizes=[1, 1, q_chunk, head_dim],
                            dst_strides=[
                                herd_cols * q_chunk * head_dim,
                                q_chunk * head_dim,
                                head_dim,
                                1,
                            ],
                        )
                    else:
                        dma_memcpy_nd(
                            ho,
                            l1_out,
                            dst_offsets=[_tx, _ty, 0, 0],
                            dst_sizes=[1, 1, q_chunk, head_dim],
                            dst_strides=[
                                herd_cols * q_chunk * head_dim,
                                q_chunk * head_dim,
                                head_dim,
                                1,
                            ],
                        )

                    DeallocOp(l1_q)
                    DeallocOp(l1_k)
                    DeallocOp(l1_v)
                    DeallocOp(l1_out)

                if use_l2_gather:
                    dma_memcpy_nd(so, l2_out)
                    DeallocOp(l2_out)

                DeallocOp(l2_q)
                DeallocOp(l2_k)
                DeallocOp(l2_v)


@module_builder
def build_flowqkv_pipeline_module(
    q_chunk,
    kv_len,
    head_dim,
    score_kernel_func,
    apply_kernel_func,
    object_file,
    public_name,
    kv_groups=4,
    herd_rows=8,
    herd_cols=4,
    output_mode="l2-gather",
    query_base=0,
    kv_chunk=32,
):
    if output_mode not in ("direct", "l2-gather"):
        raise ValueError(f"unsupported output mode: {output_mode}")
    if herd_rows % 2 != 0:
        raise ValueError("FlowQKV pipeline mode expects an even number of CT rows")
    if herd_cols != kv_groups:
        raise ValueError("FlowQKV pipeline mode expects one KV group per CT column")
    if query_base != 0:
        raise ValueError("FlowQKV pipeline mode currently expects query-base 0")
    if kv_chunk <= 0 or kv_len % kv_chunk != 0:
        raise ValueError("FlowQKV pipeline mode expects kv_len divisible by positive kv_chunk")
    use_l2_gather = output_mode == "l2-gather"
    kv_chunks = kv_len // kv_chunk

    pipe_rows = herd_rows // 2
    bf16_type = type_mapper(bfloat16)
    q_l3_ty = MemRefType.get([pipe_rows, herd_cols, q_chunk, head_dim], bf16_type)
    kv_l3_ty = MemRefType.get([kv_groups, kv_len, head_dim], bf16_type)
    out_l3_ty = MemRefType.get([pipe_rows, herd_cols, q_chunk, head_dim], bf16_type)

    l2_space = IntegerAttr.get(T.i32(), MemorySpace.L2)
    q_l2_ty = MemRefType.get(
        [pipe_rows, herd_cols, q_chunk, head_dim],
        bf16_type,
        memory_space=l2_space,
    )
    kv_l2_ty = MemRefType.get(
        [kv_groups, kv_len, head_dim], bf16_type, memory_space=l2_space
    )
    if use_l2_gather:
        out_l2_ty = MemRefType.get(
            [pipe_rows, herd_cols, q_chunk, head_dim],
            bf16_type,
            memory_space=l2_space,
        )

    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    q_l1_ty = MemRefType.get([q_chunk, head_dim], bf16_type, memory_space=l1_space)
    k_l1_ty = MemRefType.get([kv_len, head_dim], bf16_type, memory_space=l1_space)
    v_chunk_l1_ty = MemRefType.get(
        [kv_chunk, head_dim], bf16_type, memory_space=l1_space
    )
    attn_l1_ty = MemRefType.get([q_chunk, kv_chunk], bf16_type, memory_space=l1_space)
    out_l1_ty = MemRefType.get([q_chunk, head_dim], bf16_type, memory_space=l1_space)

    score_func = FuncOp(
        score_kernel_func,
        ([T.i32(), T.i32(), q_l1_ty, k_l1_ty, attn_l1_ty], []),
        visibility="private",
    )
    score_func.attributes["link_with"] = StringAttr.get(object_file)
    score_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    apply_func = FuncOp(
        apply_kernel_func,
        ([T.i32(), attn_l1_ty, v_chunk_l1_ty, out_l1_ty], []),
        visibility="private",
    )
    apply_func.attributes["link_with"] = StringAttr.get(object_file)
    apply_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    attn_chan = f"{public_name}_pipe_attn"
    Channel(attn_chan, size=[pipe_rows, kv_groups])
    query_base_map = _affine_mul(q_chunk)

    @FuncOp.from_py_func(q_l3_ty, kv_l3_ty, kv_l3_ty, out_l3_ty)
    def flowqkv_pipeline(arg_q, arg_k, arg_v, arg_out):
        @launch(operands=[arg_q, arg_k, arg_v, arg_out], sizes=[1, 1])
        def launch_body(_lx, _ly, _lsx, _lsy, lq, lk, lv, lo):
            @segment(name=f"{public_name}_pipeline_seg", operands=[lq, lk, lv, lo])
            def segment_body(sq, sk, sv, so):
                l2_q = AllocOp(q_l2_ty, [], [])
                l2_k = AllocOp(kv_l2_ty, [], [])
                l2_v = AllocOp(kv_l2_ty, [], [])
                if use_l2_gather:
                    l2_out = AllocOp(out_l2_ty, [], [])

                dma_memcpy_nd(l2_q, sq)
                dma_memcpy_nd(l2_k, sk)
                dma_memcpy_nd(l2_v, sv)

                @herd(
                    name=f"{public_name}_score_herd",
                    sizes=[pipe_rows, kv_groups],
                    operands=[l2_q, l2_k],
                    link_with=object_file,
                )
                def score_body(_tx, _ty, _sx, _sy, hq, hk):
                    l1_q = AllocOp(q_l1_ty, [], [])
                    l1_k = AllocOp(k_l1_ty, [], [])
                    l1_attn = [AllocOp(attn_l1_ty, [], []) for _ in range(kv_chunks)]

                    dma_memcpy_nd(
                        l1_q,
                        hq,
                        src_offsets=[_tx, _ty, 0, 0],
                        src_sizes=[1, 1, q_chunk, head_dim],
                        src_strides=[
                            herd_cols * q_chunk * head_dim,
                            q_chunk * head_dim,
                            head_dim,
                            1,
                        ],
                    )
                    dma_memcpy_nd(
                        l1_k,
                        hk,
                        src_offsets=[_ty, 0, 0],
                        src_sizes=[1, kv_len, head_dim],
                        src_strides=[kv_len * head_dim, head_dim, 1],
                    )
                    qbase_idx = affine_apply(query_base_map, [_tx])
                    qbase_i32 = arith.IndexCastOp(T.i32(), qbase_idx).result
                    for chunk in range(kv_chunks):
                        chunk_i32 = _i32_constant(chunk * kv_chunk)
                        CallOp(
                            score_func,
                            [qbase_i32, chunk_i32, l1_q, l1_k, l1_attn[chunk]],
                        )
                        ChannelPut(attn_chan, l1_attn[chunk], indices=[_tx, _ty])

                    DeallocOp(l1_q)
                    DeallocOp(l1_k)
                    for chunk in range(kv_chunks):
                        DeallocOp(l1_attn[chunk])

                @herd(
                    name=f"{public_name}_apply_herd",
                    sizes=[pipe_rows, kv_groups],
                    operands=[l2_v, l2_out if use_l2_gather else so],
                    link_with=object_file,
                )
                def apply_body(_tx, _ty, _sx, _sy, hv, ho):
                    l1_attn = [AllocOp(attn_l1_ty, [], []) for _ in range(kv_chunks)]
                    l1_v_full = AllocOp(k_l1_ty, [], [])
                    l1_v_chunk = AllocOp(v_chunk_l1_ty, [], [])
                    l1_out = AllocOp(out_l1_ty, [], [])

                    dma_memcpy_nd(
                        l1_v_full,
                        hv,
                        src_offsets=[_ty, 0, 0],
                        src_sizes=[1, kv_len, head_dim],
                        src_strides=[kv_len * head_dim, head_dim, 1],
                    )
                    for chunk in range(kv_chunks):
                        chunk_offset = chunk * kv_chunk
                        chunk_i32 = _i32_constant(chunk_offset)
                        ChannelGet(attn_chan, l1_attn[chunk], indices=[_tx, _ty])
                        l1_v_src = subview(
                            l1_v_full.result,
                            [chunk_offset, 0],
                            [kv_chunk, head_dim],
                            [1, 1],
                        )
                        linalg.copy(l1_v_src, outs=[l1_v_chunk])
                        CallOp(
                            apply_func,
                            [chunk_i32, l1_attn[chunk], l1_v_chunk, l1_out],
                        )
                    if use_l2_gather:
                        dma_memcpy_nd(
                            ho,
                            l1_out,
                            dst_offsets=[_tx, _ty, 0, 0],
                            dst_sizes=[1, 1, q_chunk, head_dim],
                            dst_strides=[
                                herd_cols * q_chunk * head_dim,
                                q_chunk * head_dim,
                                head_dim,
                                1,
                            ],
                        )
                    else:
                        dma_memcpy_nd(
                            ho,
                            l1_out,
                            dst_offsets=[_tx, _ty, 0, 0],
                            dst_sizes=[1, 1, q_chunk, head_dim],
                            dst_strides=[
                                herd_cols * q_chunk * head_dim,
                                q_chunk * head_dim,
                                head_dim,
                                1,
                            ],
                        )

                    for chunk in range(kv_chunks):
                        DeallocOp(l1_attn[chunk])
                    DeallocOp(l1_v_full)
                    DeallocOp(l1_v_chunk)
                    DeallocOp(l1_out)

                if use_l2_gather:
                    dma_memcpy_nd(so, l2_out)
                    DeallocOp(l2_out)

                DeallocOp(l2_q)
                DeallocOp(l2_k)
                DeallocOp(l2_v)

@module_builder
def build_flowkv_pipeline_module(
    kv_len,
    head_dim,
    score_kernel_func,
    apply_kernel_func,
    object_file,
    public_name,
    kv_groups=4,
    herd_rows=2,
    herd_cols=4,
    output_mode="direct",
    kv_chunk=32,
):
    if output_mode not in ("direct", "l2-gather"):
        raise ValueError(f"unsupported output mode: {output_mode}")
    if herd_rows != 2:
        raise ValueError("FlowKV pipeline mode expects exactly two CT rows")
    if herd_cols != kv_groups:
        raise ValueError("FlowKV pipeline mode expects one pipeline pair per KV group column")
    if kv_chunk <= 0 or kv_len % kv_chunk != 0:
        raise ValueError("FlowKV pipeline mode expects kv_len divisible by positive kv_chunk")
    kv_chunks = kv_len // kv_chunk

    bf16_type = type_mapper(bfloat16)
    q_l3_ty = MemRefType.get([kv_groups, 1, head_dim], bf16_type)
    kv_l3_ty = MemRefType.get([kv_groups, kv_len, head_dim], bf16_type)
    out_l3_ty = MemRefType.get([kv_groups, 1, head_dim], bf16_type)

    l2_space = IntegerAttr.get(T.i32(), MemorySpace.L2)
    q_l2_ty = MemRefType.get([1, head_dim], bf16_type, memory_space=l2_space)
    kv_l2_ty = MemRefType.get([kv_len, head_dim], bf16_type, memory_space=l2_space)
    out_l2_ty = MemRefType.get([1, head_dim], bf16_type, memory_space=l2_space)

    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    q_l1_ty = MemRefType.get([1, head_dim], bf16_type, memory_space=l1_space)
    kv_l1_ty = MemRefType.get([kv_len, head_dim], bf16_type, memory_space=l1_space)
    attn_l1_ty = MemRefType.get([kv_chunk], bf16_type, memory_space=l1_space)
    out_l1_ty = MemRefType.get([1, head_dim], bf16_type, memory_space=l1_space)

    score_func = FuncOp(
        score_kernel_func,
        ([T.i32(), q_l1_ty, kv_l1_ty, attn_l1_ty], []),
        visibility="private",
    )
    score_func.attributes["link_with"] = StringAttr.get(object_file)
    score_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    apply_func = FuncOp(
        apply_kernel_func,
        ([T.i32(), attn_l1_ty, kv_l1_ty, out_l1_ty], []),
        visibility="private",
    )
    apply_func.attributes["link_with"] = StringAttr.get(object_file)
    apply_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    q_chan = f"{public_name}_pipe_q"
    k_chan = f"{public_name}_pipe_k"
    v_chan = f"{public_name}_pipe_v"
    attn_chan = f"{public_name}_pipe_attn"
    out_chan = f"{public_name}_pipe_out"
    Channel(q_chan, size=[1, kv_groups])
    Channel(k_chan, size=[1, kv_groups])
    Channel(v_chan, size=[1, kv_groups])
    Channel(attn_chan, size=[1, kv_groups])
    Channel(out_chan, size=[1, kv_groups])

    @FuncOp.from_py_func(q_l3_ty, kv_l3_ty, kv_l3_ty, out_l3_ty)
    def flowkv_pipeline(arg_q, arg_k, arg_v, arg_out):
        @launch(operands=[arg_q, arg_k, arg_v, arg_out], sizes=[1, 1])
        def launch_body(_lx, _ly, _lsx, _lsy, lq, lk, lv, lo):
            @segment(name=f"{public_name}_pipeline_seg", operands=[lq, lk, lv, lo])
            def segment_body(sq, sk, sv, so):
                l2_q = [AllocOp(q_l2_ty, [], []) for _ in range(kv_groups)]
                l2_k = [AllocOp(kv_l2_ty, [], []) for _ in range(kv_groups)]
                l2_v = [AllocOp(kv_l2_ty, [], []) for _ in range(kv_groups)]
                l2_out = [AllocOp(out_l2_ty, [], []) for _ in range(kv_groups)]

                for group in range(kv_groups):
                    dma_memcpy_nd(
                        l2_q[group],
                        sq,
                        src_offsets=[group, 0, 0],
                        src_sizes=[1, 1, head_dim],
                        src_strides=[head_dim, head_dim, 1],
                    )
                    dma_memcpy_nd(
                        l2_k[group],
                        sk,
                        src_offsets=[group, 0, 0],
                        src_sizes=[1, kv_len, head_dim],
                        src_strides=[kv_len * head_dim, head_dim, 1],
                    )
                    dma_memcpy_nd(
                        l2_v[group],
                        sv,
                        src_offsets=[group, 0, 0],
                        src_sizes=[1, kv_len, head_dim],
                        src_strides=[kv_len * head_dim, head_dim, 1],
                    )
                    ChannelPut(q_chan, l2_q[group], indices=[0, group])
                    ChannelPut(k_chan, l2_k[group], indices=[0, group])
                    ChannelPut(v_chan, l2_v[group], indices=[0, group])

                @herd(
                    name=f"{public_name}_score_herd",
                    sizes=[1, kv_groups],
                    link_with=object_file,
                )
                def score_body(_tx, _ty, _sx, _sy):
                    l1_q = AllocOp(q_l1_ty, [], [])
                    l1_k = AllocOp(kv_l1_ty, [], [])
                    l1_attn = [AllocOp(attn_l1_ty, [], []) for _ in range(kv_chunks)]

                    ChannelGet(q_chan, l1_q, indices=[_tx, _ty])
                    ChannelGet(k_chan, l1_k, indices=[_tx, _ty])
                    for chunk in range(kv_chunks):
                        chunk_i32 = _i32_constant(chunk * kv_chunk)
                        CallOp(score_func, [chunk_i32, l1_q, l1_k, l1_attn[chunk]])
                        ChannelPut(attn_chan, l1_attn[chunk], indices=[_tx, _ty])

                    DeallocOp(l1_q)
                    DeallocOp(l1_k)
                    for chunk in range(kv_chunks):
                        DeallocOp(l1_attn[chunk])

                @herd(
                    name=f"{public_name}_apply_herd",
                    sizes=[1, kv_groups],
                    link_with=object_file,
                )
                def apply_body(_tx, _ty, _sx, _sy):
                    l1_attn = [AllocOp(attn_l1_ty, [], []) for _ in range(kv_chunks)]
                    l1_v = AllocOp(kv_l1_ty, [], [])
                    l1_out = AllocOp(out_l1_ty, [], [])
                    zero = ConstantOp(bf16_type, 0.0)
                    linalg.fill(zero, outs=[l1_out])

                    ChannelGet(v_chan, l1_v, indices=[_tx, _ty])
                    for chunk in range(kv_chunks):
                        chunk_i32 = _i32_constant(chunk * kv_chunk)
                        ChannelGet(attn_chan, l1_attn[chunk], indices=[_tx, _ty])
                        CallOp(apply_func, [chunk_i32, l1_attn[chunk], l1_v, l1_out])
                    ChannelPut(out_chan, l1_out, indices=[_tx, _ty])

                    for chunk in range(kv_chunks):
                        DeallocOp(l1_attn[chunk])
                    DeallocOp(l1_v)
                    DeallocOp(l1_out)

                for group in range(kv_groups):
                    ChannelGet(out_chan, l2_out[group], indices=[0, group])
                    dma_memcpy_nd(
                        so,
                        l2_out[group],
                        dst_offsets=[group, 0, 0],
                        dst_sizes=[1, 1, head_dim],
                        dst_strides=[head_dim, head_dim, 1],
                    )

                for group in range(kv_groups):
                    DeallocOp(l2_q[group])
                    DeallocOp(l2_k[group])
                    DeallocOp(l2_v[group])
                    DeallocOp(l2_out[group])


def run_or_compile(
    module,
    inputs,
    expected,
    *,
    compile_mode,
    output_format,
    verbose,
    instance_name,
    rtol=2e-1,
    atol=1e-1,
    debug_ir=False,
):
    backend_opts = dict(
        verbose=verbose,
        omit_pingpong=True,
        output_format=output_format,
        instance_name=instance_name,
        target_device="npu2",
        runtime_loop_tiling_sizes=[1, 1],
        debug_ir=debug_ir,
    )
    if compile_mode == "compile-and-run":
        runner = XRTRunner(**backend_opts)
        raise SystemExit(
            runner.run_test(
                module,
                inputs=inputs,
                expected_outputs=[expected],
                rtol=rtol,
                atol=atol,
                max_mismatch_percentage=5.0,
            )
        )

    backend = XRTBackend(**backend_opts)
    backend.compile(module)
    backend.unload()
