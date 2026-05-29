# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fused Q4NX dequantization and projection example."""

import argparse

import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects.arith import ConstantOp
from air.dialects import linalg
from air.dialects.func import FuncOp, CallOp
from air.dialects.memref import AllocOp, DeallocOp, subview, view
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner, type_mapper

from common import (
    OUTPUT_MODES,
    SCHEDULE_MODES,
    SUPPORTED_HERD_SHAPES,
    parse_herd_shape,
    random_q4nx_blocks,
    fused_dqp_blocks_reference,
    fused_dqp_paper_reference,
    resolve_output_mode,
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


def _index_constant(value):
    index_type = IndexType.get()
    return ConstantOp(index_type, IntegerAttr.get(index_type, value))


def _pack_l3_inputs(packed: np.ndarray, params: np.ndarray) -> np.ndarray:
    packed_i8 = np.ascontiguousarray(packed, dtype=np.int8)
    params_i8 = np.ascontiguousarray(params).view(np.int8).reshape(
        params.shape[:-1] + (params.shape[-1] * params.dtype.itemsize,)
    )
    block_bytes = packed_i8.shape[-1] + params_i8.shape[-1]
    packed_l3 = np.empty(packed_i8.shape[:-1] + (block_bytes,), dtype=np.int8)
    packed_l3[..., : packed_i8.shape[-1]] = packed_i8
    packed_l3[..., packed_i8.shape[-1] :] = params_i8
    return packed_l3


@module_builder
def build_module(
    rows,
    cols,
    kernel_name="fused_dqp_block",
    object_file="fused_dqp.o",
    row_blocks=1,
    herd_rows=1,
    herd_cols=1,
    output_mode="direct",
):
    if output_mode not in ("direct", "l2-gather"):
        raise ValueError(f"unsupported output mode: {output_mode}")
    use_l2_gather = output_mode == "l2-gather"

    bf16_type = type_mapper(bfloat16)
    i8_type = IntegerType.get_signless(8)
    packed_elems = rows * cols // 2
    param_elems = 3 * cols
    herd_tiles = herd_rows * herd_cols
    block_bytes = packed_elems + param_elems * np.dtype(bfloat16).itemsize

    l3_pack_ty = MemRefType.get(
        [row_blocks // herd_cols, herd_cols, block_bytes], i8_type
    )
    l3_out_ty = MemRefType.get([row_blocks, rows], bf16_type)

    l2_space = IntegerAttr.get(T.i32(), MemorySpace.L2)
    l2_pack_ty = MemRefType.get(
        [herd_rows, herd_cols, block_bytes], i8_type, memory_space=l2_space
    )
    if use_l2_gather:
        l2_out_ty = MemRefType.get(
            [herd_rows, herd_cols, rows], bf16_type, memory_space=l2_space
        )

    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_pack_ty = MemRefType.get([block_bytes], i8_type, memory_space=l1_space)
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
    launch_offset_map = _affine_mul(herd_tiles)
    launch_row_offset_map = _affine_mul(herd_rows)

    @FuncOp.from_py_func(l3_pack_ty, l3_out_ty)
    def fused_dqp(arg_pack, arg_out):
        @launch(
            operands=[arg_pack, arg_out],
            sizes=[row_blocks // herd_tiles, 1],
        )
        def launch_body(lx, _ly, _lsx, _lsy, lpack, lo):
            launch_base = affine_apply(launch_offset_map, [lx])
            launch_row_base = affine_apply(launch_row_offset_map, [lx])

            @segment(
                name="fused_dqp_seg", operands=[launch_base, launch_row_base, lpack, lo]
            )
            def segment_body(base, row_base, spack, so):
                l2_pack = AllocOp(l2_pack_ty, [], [])
                if use_l2_gather:
                    l2_out = AllocOp(l2_out_ty, [], [])
                dma_memcpy_nd(
                    l2_pack,
                    spack,
                    src_offsets=[row_base, 0, 0],
                    src_sizes=[herd_rows, herd_cols, block_bytes],
                    src_strides=[herd_cols * block_bytes, block_bytes, 1],
                )

                @herd(
                    name="fused_dqp_herd",
                    sizes=[herd_rows, herd_cols],
                    operands=[base, l2_pack, l2_out if use_l2_gather else so],
                    link_with=object_file,
                )
                def herd_body(_tx, _ty, _sx, _sy, bbase, hpack, h_out):
                    block_idx = affine_apply(block_map, [bbase, _tx, _ty])

                    l1_pack = AllocOp(l1_pack_ty, [], [])
                    l1_w = AllocOp(l1_w_ty, [], [])
                    l1_s = AllocOp(l1_param_ty, [], [])
                    l1_m = AllocOp(l1_param_ty, [], [])
                    l1_a = AllocOp(l1_param_ty, [], [])
                    l1_out = AllocOp(l1_out_ty, [], [])

                    dma_memcpy_nd(
                        l1_pack,
                        hpack,
                        src_offsets=[_tx, _ty, 0],
                        src_sizes=[1, 1, block_bytes],
                        src_strides=[herd_cols * block_bytes, block_bytes, 1],
                    )

                    l1_w_src = subview(l1_pack.result, [0], [packed_elems], [1])
                    l1_p_view = view(
                        l1_param_pack_ty,
                        l1_pack.result,
                        _index_constant(packed_elems),
                        [],
                    )
                    l1_s_src = subview(l1_p_view, [0], [cols], [1])
                    l1_m_src = subview(l1_p_view, [cols], [cols], [1])
                    l1_a_src = subview(l1_p_view, [2 * cols], [cols], [1])
                    linalg.copy(l1_w_src, outs=[l1_w])
                    linalg.copy(l1_s_src, outs=[l1_s])
                    linalg.copy(l1_m_src, outs=[l1_m])
                    linalg.copy(l1_a_src, outs=[l1_a])

                    CallOp(dqp_func, [l1_w, l1_s, l1_m, l1_a, l1_out])
                    if use_l2_gather:
                        dma_memcpy_nd(
                            h_out,
                            l1_out,
                            dst_offsets=[_tx, _ty, 0],
                            dst_sizes=[1, 1, rows],
                            dst_strides=[herd_cols * rows, rows, 1],
                        )
                    else:
                        dma_memcpy_nd(
                            h_out,
                            l1_out,
                            dst_offsets=[block_idx, 0],
                            dst_sizes=[1, rows],
                            dst_strides=[rows, 1],
                        )

                    DeallocOp(l1_pack)
                    DeallocOp(l1_w)
                    DeallocOp(l1_s)
                    DeallocOp(l1_m)
                    DeallocOp(l1_a)
                    DeallocOp(l1_out)

                if use_l2_gather:
                    dma_memcpy_nd(
                        so,
                        l2_out,
                        dst_offsets=[base, 0],
                        dst_sizes=[herd_tiles, rows],
                        dst_strides=[rows, 1],
                        src_offsets=[0, 0, 0],
                        src_sizes=[herd_rows, herd_cols, rows],
                        src_strides=[herd_cols * rows, rows, 1],
                    )
                    DeallocOp(l2_out)

                DeallocOp(l2_pack)


@module_builder
def build_paper_module(
    rows,
    cols,
    kernel_name="fused_dqp_accum_block",
    object_file="fused_dqp.o",
    row_blocks=1,
    col_blocks=1,
    herd_rows=4,
    herd_cols=4,
    output_mode="l2-gather",
):
    if output_mode not in ("direct", "l2-gather"):
        raise ValueError(f"unsupported output mode: {output_mode}")
    use_l2_gather = output_mode == "l2-gather"

    bf16_type = type_mapper(bfloat16)
    i8_type = IntegerType.get_signless(8)
    packed_elems = rows * cols // 2
    param_elems = 2 * cols
    herd_tiles = herd_rows * herd_cols
    block_bytes = packed_elems + param_elems * np.dtype(bfloat16).itemsize

    l3_pack_ty = MemRefType.get(
        [row_blocks // herd_cols, herd_cols, col_blocks, block_bytes], i8_type
    )
    l3_act_ty = MemRefType.get([col_blocks, cols], bf16_type)
    l3_out_ty = MemRefType.get([row_blocks, rows], bf16_type)

    l2_space = IntegerAttr.get(T.i32(), MemorySpace.L2)
    l2_pack_ty = MemRefType.get(
        [herd_rows, herd_cols, col_blocks, block_bytes],
        i8_type,
        memory_space=l2_space,
    )
    l2_act_ty = MemRefType.get(
        [col_blocks, cols], bf16_type, memory_space=l2_space
    )
    if use_l2_gather:
        l2_out_ty = MemRefType.get(
            [herd_rows, herd_cols, rows], bf16_type, memory_space=l2_space
        )

    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_pack_all_ty = MemRefType.get(
        [col_blocks * block_bytes], i8_type, memory_space=l1_space
    )
    l1_act_all_ty = MemRefType.get(
        [col_blocks * cols], bf16_type, memory_space=l1_space
    )
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
    launch_offset_map = _affine_mul(herd_tiles)
    launch_row_offset_map = _affine_mul(herd_rows)

    @FuncOp.from_py_func(l3_pack_ty, l3_act_ty, l3_out_ty)
    def fused_dqp_paper(arg_pack, arg_act, arg_out):
        @launch(
            operands=[arg_pack, arg_act, arg_out],
            sizes=[row_blocks // herd_tiles, 1],
        )
        def launch_body(lx, _ly, _lsx, _lsy, lpack, lact, lo):
            launch_base = affine_apply(launch_offset_map, [lx])
            launch_row_base = affine_apply(launch_row_offset_map, [lx])

            @segment(
                name="fused_dqp_paper_seg",
                operands=[launch_base, launch_row_base, lpack, lact, lo],
            )
            def segment_body(base, row_base, spack, sact, so):
                l2_pack = AllocOp(l2_pack_ty, [], [])
                l2_act = AllocOp(l2_act_ty, [], [])
                if use_l2_gather:
                    l2_out = AllocOp(l2_out_ty, [], [])

                dma_memcpy_nd(
                    l2_pack,
                    spack,
                    src_offsets=[row_base, 0, 0, 0],
                    src_sizes=[herd_rows, herd_cols, col_blocks, block_bytes],
                    src_strides=[
                        herd_cols * col_blocks * block_bytes,
                        col_blocks * block_bytes,
                        block_bytes,
                        1,
                    ],
                )
                dma_memcpy_nd(l2_act, sact)

                @herd(
                    name="fused_dqp_paper_herd",
                    sizes=[herd_rows, herd_cols],
                    operands=[base, l2_pack, l2_act, l2_out if use_l2_gather else so],
                    link_with=object_file,
                )
                def herd_body(_tx, _ty, _sx, _sy, bbase, hpack, hact, h_out):
                    block_idx = affine_apply(block_map, [bbase, _tx, _ty])

                    l1_pack_all = AllocOp(l1_pack_all_ty, [], [])
                    l1_act_all = AllocOp(l1_act_all_ty, [], [])
                    l1_w = AllocOp(l1_w_ty, [], [])
                    l1_s = AllocOp(l1_param_ty, [], [])
                    l1_m = AllocOp(l1_param_ty, [], [])
                    l1_a = AllocOp(l1_param_ty, [], [])
                    l1_out = AllocOp(l1_out_ty, [], [])
                    zero = ConstantOp(bf16_type, 0.0)
                    linalg.fill(zero, outs=[l1_out])

                    dma_memcpy_nd(
                        l1_pack_all,
                        hpack,
                        src_offsets=[_tx, _ty, 0, 0],
                        src_sizes=[1, 1, col_blocks, block_bytes],
                        src_strides=[
                            herd_cols * col_blocks * block_bytes,
                            col_blocks * block_bytes,
                            block_bytes,
                            1,
                        ],
                    )
                    dma_memcpy_nd(
                        l1_act_all,
                        hact,
                        src_offsets=[0, 0],
                        src_sizes=[col_blocks, cols],
                        src_strides=[cols, 1],
                    )

                    for cb in range(col_blocks):
                        block_offset = cb * block_bytes
                        act_offset = cb * cols
                        l1_w_src = subview(
                            l1_pack_all.result, [block_offset], [packed_elems], [1]
                        )
                        l1_p_view = view(
                            l1_param_pack_ty,
                            l1_pack_all.result,
                            _index_constant(block_offset + packed_elems),
                            [],
                        )
                        l1_s_src = subview(l1_p_view, [0], [cols], [1])
                        l1_m_src = subview(l1_p_view, [cols], [cols], [1])
                        l1_a_src = subview(
                            l1_act_all.result, [act_offset], [cols], [1]
                        )
                        linalg.copy(l1_w_src, outs=[l1_w])
                        linalg.copy(l1_s_src, outs=[l1_s])
                        linalg.copy(l1_m_src, outs=[l1_m])
                        linalg.copy(l1_a_src, outs=[l1_a])
                        CallOp(dqp_func, [l1_w, l1_s, l1_m, l1_a, l1_out])

                    if use_l2_gather:
                        dma_memcpy_nd(
                            h_out,
                            l1_out,
                            dst_offsets=[_tx, _ty, 0],
                            dst_sizes=[1, 1, rows],
                            dst_strides=[herd_cols * rows, rows, 1],
                        )
                    else:
                        dma_memcpy_nd(
                            h_out,
                            l1_out,
                            dst_offsets=[block_idx, 0],
                            dst_sizes=[1, rows],
                            dst_strides=[rows, 1],
                        )

                    DeallocOp(l1_pack_all)
                    DeallocOp(l1_act_all)
                    DeallocOp(l1_w)
                    DeallocOp(l1_s)
                    DeallocOp(l1_m)
                    DeallocOp(l1_a)
                    DeallocOp(l1_out)

                if use_l2_gather:
                    dma_memcpy_nd(
                        so,
                        l2_out,
                        dst_offsets=[base, 0],
                        dst_sizes=[herd_tiles, rows],
                        dst_strides=[rows, 1],
                        src_offsets=[0, 0, 0],
                        src_sizes=[herd_rows, herd_cols, rows],
                        src_strides=[herd_cols * rows, rows, 1],
                    )
                    DeallocOp(l2_out)

                DeallocOp(l2_pack)
                DeallocOp(l2_act)


def _paper_kernel_name(kernel_name: str) -> str:
    if kernel_name == "fused_dqp_block":
        return "fused_dqp_accum_block"
    if kernel_name == "fused_dqp_block_opt":
        return "fused_dqp_accum_block_opt"
    return kernel_name


def main():
    parser = argparse.ArgumentParser(description="FusedDQP block projection")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--cols", type=int, default=256)
    parser.add_argument("--herd-shape", choices=SUPPORTED_HERD_SHAPES, default="2x4")
    parser.add_argument("--row-blocks", type=int, default=None)
    parser.add_argument("--col-blocks", type=int, default=1)
    parser.add_argument("--herd-rows", type=int, default=None)
    parser.add_argument("--herd-cols", type=int, default=None)
    parser.add_argument("--schedule-mode", choices=SCHEDULE_MODES, default="smoke")
    parser.add_argument("--kernel-name", default="fused_dqp_block")
    parser.add_argument("--object-file", default="fused_dqp.o")
    parser.add_argument(
        "--compile-mode",
        choices=["compile-only", "compile-and-run"],
        default="compile-and-run",
    )
    parser.add_argument("--output-format", choices=["xclbin", "elf"], default="xclbin")
    parser.add_argument("--output-mode", choices=OUTPUT_MODES, default="auto")
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
    if args.col_blocks < 1:
        parser.error("col-blocks must be positive")

    try:
        output_mode = resolve_output_mode(
            args.output_mode, args.herd_rows, args.herd_cols, "fused_dqp"
        )
    except ValueError as exc:
        parser.error(str(exc))

    kernel_name = _paper_kernel_name(args.kernel_name) if args.schedule_mode == "paper" else args.kernel_name
    if args.schedule_mode == "paper":
        module = build_paper_module(
            args.rows,
            args.cols,
            kernel_name,
            args.object_file,
            args.row_blocks,
            args.col_blocks,
            args.herd_rows,
            args.herd_cols,
            output_mode,
        )
    else:
        module = build_module(
            args.rows,
            args.cols,
            kernel_name,
            args.object_file,
            args.row_blocks,
            args.herd_rows,
            args.herd_cols,
            output_mode,
        )
    if args.print_module_only:
        print(module)
        return

    rng = np.random.default_rng(2)
    if args.schedule_mode == "paper":
        packed, scale, min_offset = random_q4nx_blocks(
            args.row_blocks, args.col_blocks, args.rows, args.cols, seed=2
        )
        activation = rng.uniform(
            -0.5, 0.5, size=(args.col_blocks, args.cols)
        ).astype(bfloat16)
        expected = fused_dqp_paper_reference(
            packed, scale, min_offset, activation, args.rows, args.cols
        )
        params = np.empty(
            (args.row_blocks, args.col_blocks, 2 * args.cols), dtype=bfloat16
        )
        params[..., : args.cols] = scale
        params[..., args.cols :] = min_offset
        packed_l3 = _pack_l3_inputs(packed, params)
        inputs = [
            packed_l3.reshape(
                args.row_blocks // args.herd_cols,
                args.herd_cols,
                args.col_blocks,
                -1,
            ),
            activation,
        ]
    else:
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
        packed_l3 = _pack_l3_inputs(packed, params)
        inputs = [
            packed_l3.reshape(args.row_blocks // args.herd_cols, args.herd_cols, -1)
        ]

    instance_name = "fused_dqp_paper" if args.schedule_mode == "paper" else "fused_dqp"
    backend_opts = dict(
        verbose=args.verbose,
        omit_pingpong=True,
        output_format=args.output_format,
        instance_name=instance_name,
        target_device="npu2",
        runtime_loop_tiling_sizes=[1, 1],
        use_lock_race_condition_fix=True,
    )
    if args.compile_mode == "compile-and-run":
        runner = XRTRunner(**backend_opts)
        raise SystemExit(
            runner.run_test(
                module,
                inputs=inputs,
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
