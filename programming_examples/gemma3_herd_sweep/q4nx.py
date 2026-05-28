# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Q4NX block dequantization example."""

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
    q4nx_dequant_blocks_reference,
)


def _affine_mul(factor):
    return AffineMap.get(
        0,
        1,
        [AffineExpr.get_mul(AffineSymbolExpr.get(0), AffineConstantExpr.get(factor))],
    )


def _affine_grid_index(tile_extent):
    return AffineMap.get(
        0,
        2,
        [
            AffineExpr.get_add(
                AffineExpr.get_mul(
                    AffineSymbolExpr.get(0), AffineConstantExpr.get(tile_extent)
                ),
                AffineSymbolExpr.get(1),
            )
        ],
    )


def _affine_linear_block(col_blocks):
    return AffineMap.get(
        0,
        2,
        [
            AffineExpr.get_add(
                AffineExpr.get_mul(
                    AffineSymbolExpr.get(0), AffineConstantExpr.get(col_blocks)
                ),
                AffineSymbolExpr.get(1),
            )
        ],
    )


@module_builder
def build_module(
    rows,
    cols,
    kernel_name="q4nx_dequant_block",
    object_file="q4nx.o",
    row_blocks=1,
    col_blocks=1,
    herd_rows=1,
    herd_cols=1,
):
    bf16_type = type_mapper(bfloat16)
    i8_type = IntegerType.get_signless(8)
    packed_elems = rows * cols // 2
    param_elems = 2 * cols

    l3_w_ty = MemRefType.get([row_blocks, col_blocks, packed_elems], i8_type)
    l3_param_ty = MemRefType.get([row_blocks, col_blocks, param_elems], bf16_type)
    l3_out_ty = MemRefType.get([row_blocks * rows, col_blocks * cols], bf16_type)

    l2_space = IntegerAttr.get(T.i32(), MemorySpace.L2)
    l2_w_ty = MemRefType.get(
        [herd_rows, herd_cols, packed_elems], i8_type, memory_space=l2_space
    )
    l2_param_ty = MemRefType.get(
        [herd_rows, herd_cols, param_elems], bf16_type, memory_space=l2_space
    )
    l2_out_ty = MemRefType.get(
        [herd_rows, herd_cols, rows, cols], bf16_type, memory_space=l2_space
    )

    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_w_ty = MemRefType.get([packed_elems], i8_type, memory_space=l1_space)
    l1_param_pair_ty = MemRefType.get([param_elems], bf16_type, memory_space=l1_space)
    l1_param_ty = MemRefType.get([cols], bf16_type, memory_space=l1_space)
    l1_out_ty = MemRefType.get([rows, cols], bf16_type, memory_space=l1_space)

    dequant_func = FuncOp(
        kernel_name,
        ([l1_w_ty, l1_param_ty, l1_param_ty, l1_out_ty], []),
        visibility="private",
    )
    dequant_func.attributes["link_with"] = StringAttr.get(object_file)
    dequant_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    launch_row_map = _affine_mul(herd_rows)
    launch_col_map = _affine_mul(herd_cols)
    row_offset_map = _affine_mul(rows)
    col_offset_map = _affine_mul(cols)

    @FuncOp.from_py_func(l3_w_ty, l3_param_ty, l3_out_ty)
    def q4nx_dequant(arg_w, arg_param, arg_out):
        @launch(
            operands=[arg_w, arg_param, arg_out],
            sizes=[row_blocks // herd_rows, col_blocks // herd_cols],
        )
        def launch_body(lx, ly, _lsx, _lsy, lw, lp, lo):
            @segment(name="q4nx_seg", operands=[lx, ly, lw, lp, lo])
            def segment_body(sx, sy, sw, sp, so):
                row_base = affine_apply(launch_row_map, [sx])
                col_base = affine_apply(launch_col_map, [sy])
                row_out_offset = affine_apply(row_offset_map, [row_base])
                col_out_offset = affine_apply(col_offset_map, [col_base])

                l2_w = AllocOp(l2_w_ty, [], [])
                l2_p = AllocOp(l2_param_ty, [], [])
                l2_out = AllocOp(l2_out_ty, [], [])

                dma_memcpy_nd(
                    l2_w,
                    sw,
                    src_offsets=[row_base, col_base, 0],
                    src_sizes=[herd_rows, herd_cols, packed_elems],
                    src_strides=[col_blocks * packed_elems, packed_elems, 1],
                )
                dma_memcpy_nd(
                    l2_p,
                    sp,
                    src_offsets=[row_base, col_base, 0],
                    src_sizes=[herd_rows, herd_cols, param_elems],
                    src_strides=[col_blocks * param_elems, param_elems, 1],
                )

                @herd(
                    name="q4nx_herd",
                    sizes=[herd_rows, herd_cols],
                    operands=[l2_w, l2_p, l2_out],
                    link_with=object_file,
                )
                def herd_body(_tx, _ty, _sx, _sy, hw, hp, ho):
                    l1_w = AllocOp(l1_w_ty, [], [])
                    l1_p = AllocOp(l1_param_pair_ty, [], [])
                    l1_s = AllocOp(l1_param_ty, [], [])
                    l1_m = AllocOp(l1_param_ty, [], [])
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
                    linalg.copy(l1_s_src, outs=[l1_s])
                    linalg.copy(l1_m_src, outs=[l1_m])
                    CallOp(dequant_func, [l1_w, l1_s, l1_m, l1_out])
                    dma_memcpy_nd(
                        ho,
                        l1_out,
                        dst_offsets=[_tx, _ty, 0, 0],
                        dst_sizes=[1, 1, rows, cols],
                        dst_strides=[
                            herd_cols * rows * cols,
                            rows * cols,
                            cols,
                            1,
                        ],
                        src_offsets=[0, 0],
                        src_sizes=[rows, cols],
                        src_strides=[cols, 1],
                    )

                    DeallocOp(l1_w)
                    DeallocOp(l1_p)
                    DeallocOp(l1_s)
                    DeallocOp(l1_m)
                    DeallocOp(l1_out)

                dma_memcpy_nd(
                    so,
                    l2_out,
                    dst_offsets=[row_out_offset, col_out_offset],
                    dst_sizes=[herd_rows * rows, herd_cols * cols],
                    dst_strides=[col_blocks * cols, 1],
                    src_offsets=[0, 0, 0, 0],
                    src_sizes=[herd_rows, rows, herd_cols, cols],
                    src_strides=[
                        herd_cols * rows * cols,
                        cols,
                        rows * cols,
                        1,
                    ],
                )

                DeallocOp(l2_w)
                DeallocOp(l2_p)
                DeallocOp(l2_out)


def main():
    parser = argparse.ArgumentParser(description="Q4NX block dequantization")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--cols", type=int, default=256)
    parser.add_argument("--herd-shape", choices=SUPPORTED_HERD_SHAPES, default="2x4")
    parser.add_argument("--row-blocks", type=int, default=None)
    parser.add_argument("--col-blocks", type=int, default=None)
    parser.add_argument("--herd-rows", type=int, default=None)
    parser.add_argument("--herd-cols", type=int, default=None)
    parser.add_argument("--kernel-name", default="q4nx_dequant_block")
    parser.add_argument("--object-file", default="q4nx.o")
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
    args.row_blocks = args.row_blocks if args.row_blocks is not None else args.herd_rows
    args.col_blocks = args.col_blocks if args.col_blocks is not None else args.herd_cols

    if args.rows * args.cols % 2 != 0:
        parser.error("rows*cols must be even for int4 packing")
    if args.row_blocks % args.herd_rows != 0:
        parser.error("row-blocks must be divisible by herd-rows")
    if args.col_blocks % args.herd_cols != 0:
        parser.error("col-blocks must be divisible by herd-cols")

    module = build_module(
        args.rows,
        args.cols,
        args.kernel_name,
        args.object_file,
        args.row_blocks,
        args.col_blocks,
        args.herd_rows,
        args.herd_cols,
    )
    if args.print_module_only:
        print(module)
        return

    packed, scale, min_offset = random_q4nx_blocks(
        args.row_blocks, args.col_blocks, args.rows, args.cols, seed=1
    )
    expected = q4nx_dequant_blocks_reference(
        packed, scale, min_offset, args.rows, args.cols
    )
    params = np.empty(
        (args.row_blocks, args.col_blocks, 2 * args.cols), dtype=bfloat16
    )
    params[:, :, : args.cols] = scale
    params[:, :, args.cols :] = min_offset

    backend_opts = dict(
        verbose=args.verbose,
        omit_pingpong=True,
        output_format=args.output_format,
        instance_name="q4nx_dequant",
        target_device="npu2",
        runtime_loop_tiling_sizes=[1, 1],
    )
    if args.compile_mode == "compile-and-run":
        runner = XRTRunner(**backend_opts)
        raise SystemExit(
            runner.run_test(
                module,
                inputs=[packed, params],
                expected_outputs=[expected],
                rtol=1e-1,
                atol=5e-2,
            )
        )

    backend = XRTBackend(**backend_opts)
    backend.compile(module)
    backend.unload()


if __name__ == "__main__":
    main()
