#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gemma3 padded RMSNorm activation bridge.

The standard weighted RMSNorm example produces `memref<1x1152xbf16>` for the
Gemma3 1B decode hidden vector. Paper-style FusedDQP consumes the same values as
five 256-wide activation blocks, `memref<5x256xbf16>`, with zero padding in the
tail. This bridge keeps the RMSNorm math on the NPU and writes the padded
activation layout directly, removing host-side activation packing from the
stitched decode ingress path.
"""

from __future__ import annotations

import argparse

import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects import arith, linalg, math as math_dialect
from air.dialects.air import *
from air.dialects.memref import AllocOp, DeallocOp, subview
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


def padded_rms_norm_reference(
    x: np.ndarray,
    weight: np.ndarray,
    *,
    output_rows: int = 5,
    output_cols: int = 256,
    eps: float = EPS,
) -> np.ndarray:
    x_2d = np.asarray(x).reshape(1, -1)
    weight_flat = np.asarray(weight).reshape(-1)
    if x_2d.shape[1] != weight_flat.size:
        raise ValueError("hidden size and RMSNorm weight size must match")
    total = output_rows * output_cols
    if total < x_2d.shape[1]:
        raise ValueError("padded output is smaller than hidden input")
    xf = x_2d.astype(np.float32)
    wf = weight_flat.astype(np.float32)
    rms = np.sqrt(np.mean(xf * xf, axis=-1, keepdims=True) + eps)
    normed = ((xf / rms) * wf).astype(bfloat16).reshape(-1)
    out = np.zeros((total,), dtype=bfloat16)
    out[: normed.size] = normed
    return out.reshape(output_rows, output_cols)


@module_builder
def build_module(
    hidden_size: int = 1152,
    output_rows: int = 5,
    output_cols: int = 256,
    np_dtype=bfloat16,
    vector_size: int = 16,
):
    if hidden_size % vector_size != 0:
        raise ValueError("hidden_size must be divisible by vector_size")
    if output_rows * output_cols < hidden_size:
        raise ValueError("padded output must fit hidden_size")

    xrt_dtype = type_mapper(np_dtype)
    vec_ty = VectorType.get([vector_size], xrt_dtype)
    identity_map = AffineMapAttr.get(AffineMap.get_identity(1))

    l3_in_ty = MemRefType.get([1, hidden_size], xrt_dtype)
    l3_weight_ty = MemRefType.get([hidden_size], xrt_dtype)
    l3_out_ty = MemRefType.get([output_rows, output_cols], xrt_dtype)

    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_row_ty = MemRefType.get([hidden_size], xrt_dtype, memory_space=l1_space)
    l1_vec_ty = MemRefType.get([vector_size], xrt_dtype, memory_space=l1_space)
    l1_pad_ty = MemRefType.get([output_cols], xrt_dtype, memory_space=l1_space)

    @FuncOp.from_py_func(l3_in_ty, l3_weight_ty, l3_out_ty)
    def gemma3_padded_rms_norm(arg_x, arg_weight, arg_out):
        @launch(operands=[arg_x, arg_weight, arg_out])
        def rms_launch(l_x, l_weight, l_out):
            @segment(name="gemma3_padded_rms_norm_seg", operands=[l_x, l_weight, l_out])
            def rms_segment(s_x, s_weight, s_out):
                @herd(
                    name="gemma3_padded_rms_norm_herd",
                    sizes=[1, 1],
                    operands=[s_x, s_weight, s_out],
                )
                def rms_body(_tx, _ty, _sx, _sy, h_x, h_weight, h_out):
                    l1_row = AllocOp(l1_row_ty, [], [])
                    l1_out = AllocOp(l1_row_ty, [], [])
                    l1_weight = AllocOp(l1_row_ty, [], [])
                    l1_acc = AllocOp(l1_vec_ty, [], [])
                    l1_zero_row = AllocOp(l1_pad_ty, [], [])

                    c0 = arith.ConstantOp.create_index(0)
                    cst0 = arith.ConstantOp(xrt_dtype, 0.0)
                    n_f = arith.ConstantOp(xrt_dtype, float(hidden_size))
                    eps_f = arith.ConstantOp(xrt_dtype, EPS)
                    v_zero = BroadcastOp(vec_ty, cst0)

                    linalg.fill(cst0, outs=[l1_zero_row])
                    for out_row in range(output_rows):
                        dma_memcpy_nd(
                            h_out,
                            l1_zero_row,
                            dst_offsets=[out_row, 0],
                            dst_sizes=[1, output_cols],
                            dst_strides=[output_cols, 1],
                        )

                    dma_memcpy_nd(
                        l1_row,
                        h_x,
                        src_offsets=[0, 0],
                        src_sizes=[1, hidden_size],
                        src_strides=[hidden_size, 1],
                    )
                    dma_memcpy_nd(l1_weight, h_weight)

                    transfer_write(None, v_zero, l1_acc, [c0], identity_map, [True])
                    for j in range_(0, hidden_size, vector_size):
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
                    for j in range_(0, hidden_size, vector_size):
                        sub_row = subview(l1_row.result, [j], [vector_size], [1])
                        sub_w = subview(l1_weight.result, [j], [vector_size], [1])
                        sub_out = subview(l1_out.result, [j], [vector_size], [1])
                        v_x = transfer_read(vec_ty, sub_row, [c0], identity_map, cst0, [True])
                        v_w = transfer_read(vec_ty, sub_w, [c0], identity_map, cst0, [True])
                        v_normed = arith.mulf(v_x, v_rstd)
                        v_weighted = arith.mulf(v_normed, v_w)
                        transfer_write(None, v_weighted, sub_out, [c0], identity_map, [True])
                        yield_([])

                    for out_row in range(output_rows):
                        offset = out_row * output_cols
                        if offset >= hidden_size:
                            continue
                        copy_size = min(output_cols, hidden_size - offset)
                        src = subview(l1_out.result, [offset], [copy_size], [1])
                        dma_memcpy_nd(
                            h_out,
                            src,
                            dst_offsets=[out_row, 0],
                            dst_sizes=[1, copy_size],
                            dst_strides=[output_cols, 1],
                        )

                    DeallocOp(l1_row)
                    DeallocOp(l1_out)
                    DeallocOp(l1_weight)
                    DeallocOp(l1_acc)
                    DeallocOp(l1_zero_row)


def _self_test() -> None:
    rng = np.random.default_rng(7)
    x = rng.uniform(-0.5, 0.5, size=(1, 1152)).astype(bfloat16)
    w = rng.uniform(0.8, 1.2, size=(1152,)).astype(bfloat16)
    out = padded_rms_norm_reference(x, w)
    if out.shape != (5, 256):
        raise AssertionError("unexpected padded output shape")
    if np.any(out.reshape(-1)[1152:].astype(np.float32) != 0.0):
        raise AssertionError("padded tail is not zero")
    print("gemma3_padded_rms_norm self-test status=PASS input=1x1152 output=5x256 padded_tail=128")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma3 padded RMSNorm activation bridge")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--print-module-only", action="store_true")
    parser.add_argument("--parse-only", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--output-format", choices=["xclbin", "elf"], default="elf")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return

    module = build_module()
    if args.print_module_only:
        print(module)
        return
    if args.parse_only:
        print("gemma3_padded_rms_norm status=PARSE_PASS input=1x1152 output=5x256")
        return
    if args.compile_only:
        backend = XRTBackend(
            verbose=False,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="gemma3_padded_rms_norm",
            target_device="npu2",
            runtime_loop_tiling_sizes=[4, 4],
        )
        artifact = backend.compile(module, output_binary_name="gemma3_padded_rms_norm")
        backend.unload()
        print(
            f"gemma3_padded_rms_norm status=COMPILE_PASS output={artifact.output_binary} "
            f"format={args.output_format}"
        )
        return

    rng = np.random.default_rng(8)
    x = rng.uniform(-0.5, 0.5, size=(1, 1152)).astype(bfloat16)
    w = rng.uniform(0.8, 1.2, size=(1152,)).astype(bfloat16)
    expected = padded_rms_norm_reference(x, w)
    runner = XRTRunner(
        verbose=False,
        omit_while_true_loop=False,
        output_format=args.output_format,
        instance_name="gemma3_padded_rms_norm",
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
