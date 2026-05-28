# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared AIR generator for FlowQKV and FlowKV attention kernels."""

from __future__ import annotations

from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.air import *
from air.dialects.func import FuncOp, CallOp
from air.dialects.memref import AllocOp, DeallocOp
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner, type_mapper


@module_builder
def build_flow_module(q_chunk, kv_len, head_dim, kernel_func, object_file, public_name):
    bf16_type = type_mapper(bfloat16)

    q_l3_ty = MemRefType.get([q_chunk, head_dim], bf16_type)
    kv_l3_ty = MemRefType.get([kv_len, head_dim], bf16_type)
    out_l3_ty = MemRefType.get([q_chunk, head_dim], bf16_type)

    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
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

    @FuncOp.from_py_func(q_l3_ty, kv_l3_ty, kv_l3_ty, out_l3_ty)
    def flow_attention(arg_q, arg_k, arg_v, arg_out):
        @launch(operands=[arg_q, arg_k, arg_v, arg_out])
        def launch_body(lq, lk, lv, lo):
            @segment(name=f"{public_name}_seg", operands=[lq, lk, lv, lo])
            def segment_body(sq, sk, sv, so):
                @herd(
                    name=f"{public_name}_herd",
                    sizes=[1, 1],
                    operands=[sq, sk, sv, so],
                    link_with=object_file,
                )
                def herd_body(_tx, _ty, _sx, _sy, hq, hk, hv, ho):
                    l1_q = AllocOp(q_l1_ty, [], [])
                    l1_k = AllocOp(kv_l1_ty, [], [])
                    l1_v = AllocOp(kv_l1_ty, [], [])
                    l1_out = AllocOp(out_l1_ty, [], [])

                    dma_memcpy_nd(l1_q, hq)
                    dma_memcpy_nd(l1_k, hk)
                    dma_memcpy_nd(l1_v, hv)
                    CallOp(attn_func, [l1_q, l1_k, l1_v, l1_out])
                    dma_memcpy_nd(ho, l1_out)

                    DeallocOp(l1_q)
                    DeallocOp(l1_k)
                    DeallocOp(l1_v)
                    DeallocOp(l1_out)


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
