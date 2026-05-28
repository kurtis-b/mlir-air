# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fused Q4NX dequantization and projection example."""

import argparse

import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects import linalg
from air.dialects.func import FuncOp, CallOp
from air.dialects.memref import AllocOp, DeallocOp, subview
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner, type_mapper

from common import (
    SUPPORTED_HERD_SHAPES,
    parse_herd_shape,
    random_q4nx_blocks,
    fused_dqp_blocks_reference,
)


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
def build_module(
    rows,
    cols,
    kernel_name="fused_dqp_block",
    object_file="fused_dqp.o",
    row_blocks=1,
    herd_rows=1,
    herd_cols=1,
    stage_output=False,
):
    bf16_type = type_mapper(bfloat16)
    i8_type = IntegerType.get_signless(8)
    packed_elems = rows * cols // 2
    param_elems = 3 * cols
    herd_tiles = herd_rows * herd_cols

    l3_w_ty = MemRefType.get([row_blocks // herd_cols, herd_cols, packed_elems], i8_type)
    l3_param_ty = MemRefType.get(
        [row_blocks // herd_cols, herd_cols, param_elems], bf16_type
    )
    l3_out_ty = MemRefType.get([row_blocks, rows], bf16_type)

    l2_space = IntegerAttr.get(T.i32(), MemorySpace.L2)
    l2_w_ty = MemRefType.get(
        [herd_rows, herd_cols, packed_elems], i8_type, memory_space=l2_space
    )
    l2_param_ty = MemRefType.get(
        [herd_rows, herd_cols, param_elems], bf16_type, memory_space=l2_space
    )
    if stage_output:
        l2_out_ty = MemRefType.get(
            [herd_tiles, rows], bf16_type, memory_space=l2_space
        )

    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_w_ty = MemRefType.get([packed_elems], i8_type, memory_space=l1_space)
    l1_param_pack_ty = MemRefType.get([param_elems], bf16_type, memory_space=l1_space)
    l1_param_ty = MemRefType.get([cols], bf16_type, memory_space=l1_space)
    l1_out_ty = MemRefType.get([rows], bf16_type, memory_space=l1_space)

    dqp_func = FuncOp(
        kernel_name,
        ([l1_w_ty, l1_param_ty, l1_param_ty, l1_param_ty, l1_out_ty], []),
        visibility="private",
    )
    dqp_func.attributes["link_with"] = StringAttr.get(object_file)
    dqp_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    block_map = _affine_linear_tile(herd_cols)
    local_map = _affine_linear_local(herd_cols)
    launch_offset_map = _affine_mul(herd_tiles)
    launch_row_offset_map = _affine_mul(herd_rows)

    @FuncOp.from_py_func(l3_w_ty, l3_param_ty, l3_out_ty)
    def fused_dqp(arg_w, arg_param, arg_out):
        @launch(
            operands=[arg_w, arg_param, arg_out],
            sizes=[row_blocks // herd_tiles, 1],
        )
        def launch_body(lx, _ly, _lsx, _lsy, lw, lp, lo):
            launch_base = affine_apply(launch_offset_map, [lx])
            launch_row_base = affine_apply(launch_row_offset_map, [lx])

            @segment(
                name="fused_dqp_seg", operands=[launch_base, launch_row_base, lw, lp, lo]
            )
            def segment_body(base, row_base, sw, sp, so):
                l2_w = AllocOp(l2_w_ty, [], [])
                l2_p = AllocOp(l2_param_ty, [], [])
                if stage_output:
                    l2_out = AllocOp(l2_out_ty, [], [])
                dma_memcpy_nd(
                    l2_w,
                    sw,
                    src_offsets=[row_base, 0, 0],
                    src_sizes=[herd_rows, herd_cols, packed_elems],
                    src_strides=[herd_cols * packed_elems, packed_elems, 1],
                )
                dma_memcpy_nd(
                    l2_p,
                    sp,
                    src_offsets=[row_base, 0, 0],
                    src_sizes=[herd_rows, herd_cols, param_elems],
                    src_strides=[herd_cols * param_elems, param_elems, 1],
                )

                @herd(
                    name="fused_dqp_herd",
                    sizes=[herd_rows, herd_cols],
                    operands=[base, l2_w, l2_p, l2_out if stage_output else so],
                    link_with=object_file,
                )
                def herd_body(_tx, _ty, _sx, _sy, bbase, hw, hp, h_out):
                    block_idx = affine_apply(block_map, [bbase, _tx, _ty])
                    local_idx = affine_apply(local_map, [_tx, _ty])

                    l1_w = AllocOp(l1_w_ty, [], [])
                    l1_p = AllocOp(l1_param_pack_ty, [], [])
                    l1_s = AllocOp(l1_param_ty, [], [])
                    l1_m = AllocOp(l1_param_ty, [], [])
                    l1_a = AllocOp(l1_param_ty, [], [])
                    l1_out = AllocOp(l1_out_ty, [], [])

                    dma_memcpy_nd(
                        l1_w,
                        hw,
                        src_offsets=[_tx, _ty, 0],
                        src_sizes=[1, 1, packed_elems],
                        src_strides=[herd_cols * packed_elems, packed_elems, 1],
                    )
                    dma_memcpy_nd(
                        l1_p,
                        hp,
                        src_offsets=[_tx, _ty, 0],
                        src_sizes=[1, 1, param_elems],
                        src_strides=[herd_cols * param_elems, param_elems, 1],
                    )

                    l1_s_src = subview(l1_p.result, [0], [cols], [1])
                    l1_m_src = subview(l1_p.result, [cols], [cols], [1])
                    l1_a_src = subview(l1_p.result, [2 * cols], [cols], [1])
                    linalg.copy(l1_s_src, outs=[l1_s])
                    linalg.copy(l1_m_src, outs=[l1_m])
                    linalg.copy(l1_a_src, outs=[l1_a])

                    CallOp(dqp_func, [l1_w, l1_s, l1_m, l1_a, l1_out])
                    if stage_output:
                        dma_memcpy_nd(
                            h_out,
                            l1_out,
                            dst_offsets=[local_idx, 0],
                            dst_sizes=[1, rows],
                            dst_strides=[rows, 1],
                        )
                    else:
                        dma_memcpy_nd(
                            h_out,
                            l1_out,
                            dst_offsets=[block_idx, 0],
                            dst_sizes=[1, rows],
                            dst_strides=[rows, 1],
                        )

                    DeallocOp(l1_w)
                    DeallocOp(l1_p)
                    DeallocOp(l1_s)
                    DeallocOp(l1_m)
                    DeallocOp(l1_a)
                    DeallocOp(l1_out)

                if stage_output:
                    dma_memcpy_nd(
                        so,
                        l2_out,
                        dst_offsets=[base, 0],
                        dst_sizes=[herd_tiles, rows],
                        dst_strides=[rows, 1],
                    )
                    DeallocOp(l2_out)

                DeallocOp(l2_w)
                DeallocOp(l2_p)


def main():
    parser = argparse.ArgumentParser(description="FusedDQP block projection")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--cols", type=int, default=256)
    parser.add_argument("--herd-shape", choices=SUPPORTED_HERD_SHAPES, default="2x4")
    parser.add_argument("--row-blocks", type=int, default=None)
    parser.add_argument("--herd-rows", type=int, default=None)
    parser.add_argument("--herd-cols", type=int, default=None)
    parser.add_argument("--kernel-name", default="fused_dqp_block")
    parser.add_argument("--object-file", default="fused_dqp.o")
    parser.add_argument(
        "--compile-mode",
        choices=["compile-only", "compile-and-run"],
        default="compile-and-run",
    )
    parser.add_argument("--output-format", choices=["xclbin", "elf"], default="xclbin")
    args = parser.parse_args()

    shape_rows, shape_cols = parse_herd_shape(args.herd_shape)
    args.herd_rows = args.herd_rows if args.herd_rows is not None else shape_rows
    args.herd_cols = args.herd_cols if args.herd_cols is not None else shape_cols
    herd_tiles = args.herd_rows * args.herd_cols
    args.row_blocks = args.row_blocks if args.row_blocks is not None else herd_tiles

    if args.rows * args.cols % 2 != 0:
        parser.error("rows*cols must be even for int4 packing")
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
        stage_output=herd_tiles > 16,
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
    params = np.empty((args.row_blocks, 3 * args.cols), dtype=bfloat16)
    params[:, : args.cols] = scale
    params[:, args.cols : 2 * args.cols] = min_offset
    params[:, 2 * args.cols :] = activation[None, :]

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
                    packed.reshape(args.row_blocks // args.herd_cols, args.herd_cols, -1),
                    params.reshape(args.row_blocks // args.herd_cols, args.herd_cols, -1),
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
