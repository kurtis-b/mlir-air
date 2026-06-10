#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gemma3 projection-output view bridge plus Q/K RMSNorm.

FusedDQP writes decode projection outputs as contiguous row-block layouts:

* Q: `memref<32x32xbf16>` == 1024 contiguous values == `4x256` heads.
* K: `memref<8x32xbf16>` == 256 contiguous values == `1x256` head.

This wrapper performs the required zero-copy memref view inside the launch and
then applies weighted RMSNorm per 256-wide head. It is the bridge needed before
stitching Q/K norm and RoPE after the full-column-block FusedDQP projections.
"""

from __future__ import annotations

import argparse

import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects import arith, math as math_dialect
from air.dialects.air import *
from air.dialects.memref import (
    AllocOp,
    DeallocOp,
    collapse_shape as memref_collapse_shape,
    expand_shape as memref_expand_shape,
    subview,
)
from air.dialects.scf import for_, yield_
from air.dialects.vector import (
    BroadcastOp,
    reduction as vector_reduction,
    transfer_read,
    transfer_write,
)
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner, type_mapper

range_ = for_

EPS = 1e-5


def projection_qk_norm_reference(x: np.ndarray, weight: np.ndarray, *, head_dim: int = 256) -> np.ndarray:
    flat = np.asarray(x).reshape(-1)
    if flat.size % head_dim != 0:
        raise ValueError("projection output size must be divisible by head_dim")
    weight_flat = np.asarray(weight).reshape(-1)
    if weight_flat.size != head_dim:
        raise ValueError("Q/K norm weight must match head_dim")
    heads = flat.size // head_dim
    view = flat.reshape(heads, head_dim).astype(np.float32)
    wf = weight_flat.astype(np.float32)
    rms = np.sqrt(np.mean(view * view, axis=-1, keepdims=True) + EPS)
    return ((view / rms) * wf).astype(bfloat16)


@module_builder
def build_module(
    input_rows: int,
    input_cols: int,
    norm_rows: int,
    head_dim: int = 256,
    np_dtype=bfloat16,
    vector_size: int = 16,
):
    total = input_rows * input_cols
    if total != norm_rows * head_dim:
        raise ValueError("input layout and normalized head layout must have equal element counts")
    if head_dim % vector_size != 0:
        raise ValueError("head_dim must be divisible by vector_size")

    xrt_dtype = type_mapper(np_dtype)
    vec_ty = VectorType.get([vector_size], xrt_dtype)
    identity_map = AffineMapAttr.get(AffineMap.get_identity(1))

    l3_src_ty = MemRefType.get([input_rows, input_cols], xrt_dtype)
    l3_flat_ty = MemRefType.get([total], xrt_dtype)
    l3_norm_ty = MemRefType.get([norm_rows, head_dim], xrt_dtype)
    l3_weight_ty = MemRefType.get([head_dim], xrt_dtype)

    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_row_ty = MemRefType.get([head_dim], xrt_dtype, memory_space=l1_space)
    l1_vec_ty = MemRefType.get([vector_size], xrt_dtype, memory_space=l1_space)

    @FuncOp.from_py_func(l3_src_ty, l3_weight_ty, l3_norm_ty)
    def gemma3_projection_qk_norm(arg_src, arg_weight, arg_out):
        @launch(operands=[arg_src, arg_weight, arg_out])
        def norm_launch(l_src, l_weight, l_out):
            flat_src = memref_collapse_shape(l3_flat_ty, l_src, [[0, 1]])
            norm_src = memref_expand_shape(l3_norm_ty, flat_src, [[0, 1]], [], [norm_rows, head_dim])

            @segment(name="gemma3_projection_qk_norm_seg", operands=[norm_src, l_weight, l_out])
            def norm_segment(s_src, s_weight, s_out):
                @herd(
                    name="gemma3_projection_qk_norm_herd",
                    sizes=[1, 1],
                    operands=[s_src, s_weight, s_out],
                )
                def norm_body(_tx, _ty, _sx, _sy, h_src, h_weight, h_out):
                    l1_row = AllocOp(l1_row_ty, [], [])
                    l1_out = AllocOp(l1_row_ty, [], [])
                    l1_weight = AllocOp(l1_row_ty, [], [])
                    l1_acc = AllocOp(l1_vec_ty, [], [])

                    c0 = arith.ConstantOp.create_index(0)
                    cst0 = arith.ConstantOp(xrt_dtype, 0.0)
                    n_f = arith.ConstantOp(xrt_dtype, float(head_dim))
                    eps_f = arith.ConstantOp(xrt_dtype, EPS)
                    v_zero = BroadcastOp(vec_ty, cst0)

                    dma_memcpy_nd(l1_weight, h_weight)

                    for row in range_(norm_rows):
                        dma_memcpy_nd(
                            l1_row,
                            h_src,
                            src_offsets=[row, 0],
                            src_sizes=[1, head_dim],
                            src_strides=[head_dim, 1],
                        )

                        transfer_write(None, v_zero, l1_acc, [c0], identity_map, [True])
                        for j in range_(0, head_dim, vector_size):
                            sub_row = subview(l1_row.result, [j], [vector_size], [1])
                            sub_tmp = subview(l1_out.result, [j], [vector_size], [1])
                            v_x = transfer_read(vec_ty, sub_row, [c0], identity_map, cst0, [True])
                            v_sq = arith.mulf(v_x, v_x)
                            transfer_write(None, v_sq, sub_tmp, [c0], identity_map, [True])
                            v_sq_rd = transfer_read(vec_ty, sub_tmp, [c0], identity_map, cst0, [True])
                            v_acc = transfer_read(vec_ty, l1_acc, [c0], identity_map, cst0, [True])
                            v_sum = arith.addf(v_acc, v_sq_rd)
                            transfer_write(None, v_sum, l1_acc, [c0], identity_map, [True])
                            yield_([])

                        v_final = transfer_read(vec_ty, l1_acc, [c0], identity_map, cst0, [True])
                        total_sum = vector_reduction(xrt_dtype, "add", v_final)
                        rms = arith.divf(total_sum, n_f)
                        rms_eps = arith.addf(rms, eps_f)
                        rms_eps_f32 = arith.extf(F32Type.get(), rms_eps)
                        rstd_f32 = math_dialect.rsqrt(rms_eps_f32)
                        rstd = arith.truncf(xrt_dtype, rstd_f32)
                        v_rstd = BroadcastOp(vec_ty, rstd)

                        for j in range_(0, head_dim, vector_size):
                            sub_row = subview(l1_row.result, [j], [vector_size], [1])
                            sub_w = subview(l1_weight.result, [j], [vector_size], [1])
                            sub_out = subview(l1_out.result, [j], [vector_size], [1])
                            v_x = transfer_read(vec_ty, sub_row, [c0], identity_map, cst0, [True])
                            v_w = transfer_read(vec_ty, sub_w, [c0], identity_map, cst0, [True])
                            v_normed = arith.mulf(v_x, v_rstd)
                            v_weighted = arith.mulf(v_normed, v_w)
                            transfer_write(None, v_weighted, sub_out, [c0], identity_map, [True])
                            yield_([])

                        dma_memcpy_nd(
                            h_out,
                            l1_out,
                            dst_offsets=[row, 0],
                            dst_sizes=[1, head_dim],
                            dst_strides=[head_dim, 1],
                        )
                        yield_([])

                    DeallocOp(l1_row)
                    DeallocOp(l1_out)
                    DeallocOp(l1_weight)
                    DeallocOp(l1_acc)


def _self_test() -> None:
    rng = np.random.default_rng(19)
    q = rng.uniform(-0.7, 0.7, size=(32, 32)).astype(bfloat16)
    k = rng.uniform(-0.7, 0.7, size=(8, 32)).astype(bfloat16)
    w = rng.uniform(0.8, 1.2, size=(256,)).astype(bfloat16)
    q_out = projection_qk_norm_reference(q, w)
    k_out = projection_qk_norm_reference(k, w)
    if q_out.shape != (4, 256) or k_out.shape != (1, 256):
        raise AssertionError("unexpected Q/K norm output shapes")
    print("gemma3_projection_qk_norm self-test status=PASS q_in=32x32 q_out=4x256 k_in=8x32 k_out=1x256")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma3 projection view bridge plus Q/K RMSNorm")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--input-rows", type=int, default=32)
    parser.add_argument("--input-cols", type=int, default=32)
    parser.add_argument("--norm-rows", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--print-module-only", action="store_true")
    parser.add_argument("--parse-only", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--output-format", choices=["xclbin", "elf"], default="elf")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return

    module = build_module(args.input_rows, args.input_cols, args.norm_rows, args.head_dim)
    if args.print_module_only:
        print(module)
        return
    if args.parse_only:
        print(
            f"gemma3_projection_qk_norm status=PARSE_PASS input={args.input_rows}x{args.input_cols} "
            f"output={args.norm_rows}x{args.head_dim}"
        )
        return
    if args.compile_only:
        backend = XRTBackend(
            verbose=False,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="gemma3_projection_qk_norm",
            target_device="npu2",
            runtime_loop_tiling_sizes=[4, 4],
        )
        artifact = backend.compile(module, output_binary_name="gemma3_projection_qk_norm")
        backend.unload()
        print(
            f"gemma3_projection_qk_norm status=COMPILE_PASS output={artifact.output_binary} "
            f"format={args.output_format}"
        )
        return

    rng = np.random.default_rng(23)
    x = rng.uniform(-0.7, 0.7, size=(args.input_rows, args.input_cols)).astype(bfloat16)
    w = rng.uniform(0.8, 1.2, size=(args.head_dim,)).astype(bfloat16)
    expected = projection_qk_norm_reference(x, w, head_dim=args.head_dim)
    runner = XRTRunner(
        verbose=False,
        omit_while_true_loop=False,
        output_format=args.output_format,
        instance_name="gemma3_projection_qk_norm",
        target_device="npu2",
        runtime_loop_tiling_sizes=[4, 4],
    )
    raise SystemExit(
        runner.run_test(
            module,
            inputs=[x, w],
            expected_outputs=[expected],
            rtol=1e-2,
            atol=1e-2,
        )
    )


if __name__ == "__main__":
    main()
