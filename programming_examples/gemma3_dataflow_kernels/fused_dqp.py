# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fused Q4NX dequantization and projection example."""

import argparse

import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects.func import FuncOp, CallOp
from air.dialects.memref import AllocOp, DeallocOp
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner, type_mapper

from common import random_q4nx_blocks, fused_dqp_blocks_reference


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


@module_builder
def build_module(
    rows,
    cols,
    kernel_name="fused_dqp_block",
    object_file="fused_dqp.o",
    row_blocks=1,
    herd_rows=1,
    herd_cols=1,
):
    bf16_type = type_mapper(bfloat16)
    i8_type = IntegerType.get_signless(8)
    packed_elems = rows * cols // 2
    herd_tiles = herd_rows * herd_cols

    l3_w_ty = MemRefType.get([row_blocks * packed_elems], i8_type)
    l3_param_ty = MemRefType.get([row_blocks * cols], bf16_type)
    l3_act_ty = MemRefType.get([cols], bf16_type)
    l3_out_ty = MemRefType.get([row_blocks, rows], bf16_type)

    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_w_ty = MemRefType.get([packed_elems], i8_type, memory_space=l1_space)
    l1_param_ty = MemRefType.get([cols], bf16_type, memory_space=l1_space)
    l1_act_ty = MemRefType.get([cols], bf16_type, memory_space=l1_space)
    l1_out_ty = MemRefType.get([rows], bf16_type, memory_space=l1_space)

    dqp_func = FuncOp(
        kernel_name,
        ([l1_w_ty, l1_param_ty, l1_param_ty, l1_act_ty, l1_out_ty], []),
        visibility="private",
    )
    dqp_func.attributes["link_with"] = StringAttr.get(object_file)
    dqp_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    block_map = _affine_linear_tile(herd_cols)
    launch_offset_map = _affine_mul(herd_tiles)
    packed_offset_map = _affine_mul(packed_elems)
    param_offset_map = _affine_mul(cols)

    @FuncOp.from_py_func(l3_w_ty, l3_param_ty, l3_param_ty, l3_act_ty, l3_out_ty)
    def fused_dqp(arg_w, arg_scale, arg_min, arg_act, arg_out):
        @launch(
            operands=[arg_w, arg_scale, arg_min, arg_act, arg_out],
            sizes=[row_blocks // herd_tiles, 1],
        )
        def launch_body(lx, _ly, _lsx, _lsy, lw, ls, lm, la, lo):
            launch_base = affine_apply(launch_offset_map, [lx])

            @segment(name="fused_dqp_seg", operands=[launch_base, lw, ls, lm, la, lo])
            def segment_body(base, sw, ss, sm, sa, so):
                @herd(
                    name="fused_dqp_herd",
                    sizes=[herd_rows, herd_cols],
                    operands=[base, sw, ss, sm, sa, so],
                    link_with=object_file,
                )
                def herd_body(_tx, _ty, _sx, _sy, bbase, hw, hs, hm, ha, ho):
                    block_idx = affine_apply(block_map, [bbase, _tx, _ty])
                    packed_offset = affine_apply(packed_offset_map, [block_idx])
                    param_offset = affine_apply(param_offset_map, [block_idx])

                    l1_w = AllocOp(l1_w_ty, [], [])
                    l1_s = AllocOp(l1_param_ty, [], [])
                    l1_m = AllocOp(l1_param_ty, [], [])
                    l1_a = AllocOp(l1_act_ty, [], [])
                    l1_out = AllocOp(l1_out_ty, [], [])

                    dma_memcpy_nd(
                        l1_w,
                        hw,
                        src_offsets=[packed_offset],
                        src_sizes=[packed_elems],
                        src_strides=[1],
                    )
                    dma_memcpy_nd(
                        l1_s,
                        hs,
                        src_offsets=[param_offset],
                        src_sizes=[cols],
                        src_strides=[1],
                    )
                    dma_memcpy_nd(
                        l1_m,
                        hm,
                        src_offsets=[param_offset],
                        src_sizes=[cols],
                        src_strides=[1],
                    )
                    dma_memcpy_nd(l1_a, ha)
                    CallOp(dqp_func, [l1_w, l1_s, l1_m, l1_a, l1_out])
                    dma_memcpy_nd(
                        ho,
                        l1_out,
                        dst_offsets=[block_idx, 0],
                        dst_sizes=[1, rows],
                        dst_strides=[rows, 1],
                    )

                    DeallocOp(l1_w)
                    DeallocOp(l1_s)
                    DeallocOp(l1_m)
                    DeallocOp(l1_a)
                    DeallocOp(l1_out)


def main():
    parser = argparse.ArgumentParser(description="FusedDQP block projection")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--cols", type=int, default=256)
    parser.add_argument("--row-blocks", type=int, default=1)
    parser.add_argument("--herd-rows", type=int, default=1)
    parser.add_argument("--herd-cols", type=int, default=1)
    parser.add_argument("--kernel-name", default="fused_dqp_block")
    parser.add_argument("--object-file", default="fused_dqp.o")
    parser.add_argument(
        "--compile-mode",
        choices=["compile-only", "compile-and-run"],
        default="compile-and-run",
    )
    parser.add_argument("--output-format", choices=["xclbin", "elf"], default="xclbin")
    args = parser.parse_args()

    if args.rows * args.cols % 2 != 0:
        parser.error("rows*cols must be even for int4 packing")
    herd_tiles = args.herd_rows * args.herd_cols
    if args.row_blocks % herd_tiles != 0:
        parser.error("row-blocks must be divisible by herd-rows*herd-cols")

    module = build_module(
        args.rows,
        args.cols,
        args.kernel_name,
        args.object_file,
        args.row_blocks,
        args.herd_rows,
        args.herd_cols,
    )
    if args.print_module_only:
        print(module)
        return

    rng = np.random.default_rng(2)
    packed3, scale3, min3 = random_q4nx_blocks(
        args.row_blocks, 1, args.rows, args.cols, seed=2
    )
    packed = packed3[:, 0, :]
    scale = scale3[:, 0, :]
    min_offset = min3[:, 0, :]
    activation = rng.uniform(-0.5, 0.5, size=(args.cols,)).astype(bfloat16)
    expected = fused_dqp_blocks_reference(
        packed, scale, min_offset, activation, args.rows, args.cols
    )

    backend_opts = dict(
        verbose=args.verbose,
        omit_pingpong=True,
        output_format=args.output_format,
        instance_name="fused_dqp",
        target_device="npu2",
        runtime_loop_tiling_sizes=[1, 1],
    )
    if args.compile_mode == "compile-and-run":
        runner = XRTRunner(**backend_opts)
        raise SystemExit(
            runner.run_test(
                module,
                inputs=[
                    packed.reshape(-1),
                    scale.reshape(-1),
                    min_offset.reshape(-1),
                    activation,
                ],
                expected_outputs=[expected],
                rtol=1e-1,
                atol=7e-2,
            )
        )

    backend = XRTBackend(**backend_opts)
    backend.compile(module)
    backend.unload()


if __name__ == "__main__":
    main()
