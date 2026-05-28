# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared AIR generator for FlowQKV and FlowKV attention kernels."""

from __future__ import annotations

from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects import linalg
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
    stage_output=False,
):
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
    if stage_output:
        out_l2_ty = MemRefType.get(
            [herd_tiles, q_chunk, head_dim], bf16_type, memory_space=l2_space
        )

    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    qkv_l1_ty = MemRefType.get([qkv_rows, head_dim], bf16_type, memory_space=l1_space)
    q_l1_ty = MemRefType.get([q_chunk, head_dim], bf16_type, memory_space=l1_space)
    kv_l1_ty = MemRefType.get([kv_len, head_dim], bf16_type, memory_space=l1_space)
    out_l1_ty = MemRefType.get([q_chunk, head_dim], bf16_type, memory_space=l1_space)

    attn_func = FuncOp(
        kernel_func,
        ([q_l1_ty, kv_l1_ty, kv_l1_ty, out_l1_ty], []),
        visibility="private",
    )
    attn_func.attributes["link_with"] = StringAttr.get(object_file)
    attn_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    launch_offset_map = _affine_mul(herd_tiles)
    launch_row_offset_map = _affine_mul(herd_rows)
    group_map = _affine_linear_tile(herd_cols)
    local_map = _affine_linear_local(herd_cols)

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
                if stage_output:
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
                    operands=[base, l2_qkv, l2_out if stage_output else so],
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

                    CallOp(attn_func, [l1_q, l1_k, l1_v, l1_out])
                    if stage_output:
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

                if stage_output:
                    dma_memcpy_nd(
                        so,
                        l2_out,
                        dst_offsets=[base, 0, 0],
                        dst_sizes=[herd_tiles, q_chunk, head_dim],
                        dst_strides=[q_chunk * head_dim, head_dim, 1],
                    )
                    DeallocOp(l2_out)

                DeallocOp(l2_qkv)


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
):
    backend_opts = dict(
        verbose=verbose,
        omit_pingpong=True,
        output_format=output_format,
        instance_name=instance_name,
        target_device="npu2",
        runtime_loop_tiling_sizes=[1, 1],
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
