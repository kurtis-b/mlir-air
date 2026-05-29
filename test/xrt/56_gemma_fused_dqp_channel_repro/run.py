# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Minimal two-herd channel reproducer from Gemma FusedDQP pipeline mode.

The lit test uses compile-only mode. ``--compile-mode compile-and-run`` is a
hardware diagnostic that currently reproduces ERT_CMD_STATE_TIMEOUT.
"""

import argparse

import numpy as np
from ml_dtypes import bfloat16

from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects.arith import ConstantOp
from air.dialects.func import FuncOp
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.scf import for_ as range_, yield_
from air.ir import *
from air.dialects import linalg


def _affine_mul(factor):
    return AffineMap.get(
        0,
        1,
        [AffineExpr.get_mul(AffineSymbolExpr.get(0), AffineConstantExpr.get(factor))],
    )


@module_builder
def build_module(
    rows=32,
    cols=32,
    col_blocks=2,
    row_blocks=32,
    row_chunk=16,
    col_chunk=8,
    herd_rows=8,
    herd_cols=4,
):
    pipe_rows = herd_rows // 2
    pipe_tiles = pipe_rows * herd_cols
    bf16_type = type_mapper(bfloat16)
    i8_type = IntegerType.get_signless(8)
    packed_elems = rows * cols // 2
    param_elems = 2 * cols
    block_bytes = packed_elems + param_elems * np.dtype(bfloat16).itemsize

    l3_pack_ty = MemRefType.get(
        [row_blocks // herd_cols, herd_cols, col_blocks, block_bytes], i8_type
    )
    l3_act_ty = MemRefType.get([col_blocks, cols], bf16_type)
    l3_out_ty = MemRefType.get([row_blocks, rows], bf16_type)

    l2_space = IntegerAttr.get(T.i32(), MemorySpace.L2)
    l2_pack_ty = MemRefType.get(
        [pipe_rows, herd_cols, col_blocks, block_bytes],
        i8_type,
        memory_space=l2_space,
    )
    l2_act_ty = MemRefType.get([col_blocks, cols], bf16_type, memory_space=l2_space)
    l2_out_ty = MemRefType.get(
        [pipe_rows, herd_cols, rows], bf16_type, memory_space=l2_space
    )

    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_pack_ty = MemRefType.get(
        [col_blocks * block_bytes], i8_type, memory_space=l1_space
    )
    l1_deq_ty = MemRefType.get(
        [row_chunk, col_chunk], bf16_type, memory_space=l1_space
    )
    l1_act_ty = MemRefType.get([col_blocks * cols], bf16_type, memory_space=l1_space)
    l1_out_ty = MemRefType.get([rows], bf16_type, memory_space=l1_space)

    launch_tile_map = _affine_mul(pipe_tiles)
    launch_row_map = _affine_mul(pipe_rows)
    deq_chan = "fused_dqp_pipe_deq"
    Channel(deq_chan, size=[pipe_rows, herd_cols])

    @FuncOp.from_py_func(l3_pack_ty, l3_act_ty, l3_out_ty)
    def fused_dqp_channel_repro(arg_pack, arg_act, arg_out):
        @launch(
            operands=[arg_pack, arg_act, arg_out],
            sizes=[row_blocks // pipe_tiles, 1],
        )
        def launch_body(lx, _ly, _sx, _sy, lpack, lact, lo):
            launch_base = affine_apply(launch_tile_map, [lx])
            launch_row_base = affine_apply(launch_row_map, [lx])

            @segment(
                name="fused_dqp_channel_repro_seg",
                operands=[launch_base, launch_row_base, lpack, lact, lo],
            )
            def segment_body(base, row_base, spack, sact, so):
                l2_pack = AllocOp(l2_pack_ty, [], [])
                l2_act = AllocOp(l2_act_ty, [], [])
                l2_out = AllocOp(l2_out_ty, [], [])

                dma_memcpy_nd(
                    l2_pack,
                    spack,
                    src_offsets=[row_base, 0, 0, 0],
                    src_sizes=[pipe_rows, herd_cols, col_blocks, block_bytes],
                    src_strides=[
                        herd_cols * col_blocks * block_bytes,
                        col_blocks * block_bytes,
                        block_bytes,
                        1,
                    ],
                )
                dma_memcpy_nd(l2_act, sact)

                @herd(
                    name="fused_dqp_dequant_herd",
                    sizes=[pipe_rows, herd_cols],
                    operands=[l2_pack],
                )
                def dequant_body(tx, ty, _sx, _sy, hpack):
                    l1_pack = AllocOp(l1_pack_ty, [], [])
                    l1_deq = AllocOp(l1_deq_ty, [], [])

                    dma_memcpy_nd(
                        l1_pack,
                        hpack,
                        src_offsets=[tx, ty, 0, 0],
                        src_sizes=[1, 1, col_blocks, block_bytes],
                        src_strides=[
                            herd_cols * col_blocks * block_bytes,
                            col_blocks * block_bytes,
                            block_bytes,
                            1,
                        ],
                    )
                    zero = ConstantOp(bf16_type, 0.0)
                    linalg.fill(zero, outs=[l1_deq])

                    for _cb in range_(0, col_blocks, 1):
                        for _row in range_(0, rows, row_chunk):
                            for _col in range_(0, cols, col_chunk):
                                ChannelPut(deq_chan, l1_deq, indices=[tx, ty])
                                yield_([])
                            yield_([])
                        yield_([])

                    DeallocOp(l1_pack)
                    DeallocOp(l1_deq)

                @herd(
                    name="fused_dqp_project_herd",
                    sizes=[pipe_rows, herd_cols],
                    operands=[l2_act, l2_out],
                )
                def project_body(tx, ty, _sx, _sy, hact, hout):
                    l1_act = AllocOp(l1_act_ty, [], [])
                    l1_deq = AllocOp(l1_deq_ty, [], [])
                    l1_out = AllocOp(l1_out_ty, [], [])
                    zero = ConstantOp(bf16_type, 0.0)
                    linalg.fill(zero, outs=[l1_out])

                    dma_memcpy_nd(
                        l1_act,
                        hact,
                        src_offsets=[0, 0],
                        src_sizes=[col_blocks, cols],
                        src_strides=[cols, 1],
                    )

                    for _cb in range_(0, col_blocks, 1):
                        for _row in range_(0, rows, row_chunk):
                            for _col in range_(0, cols, col_chunk):
                                ChannelGet(deq_chan, l1_deq, indices=[tx, ty])
                                yield_([])
                            yield_([])
                        yield_([])

                    dma_memcpy_nd(
                        hout,
                        l1_out,
                        dst_offsets=[tx, ty, 0],
                        dst_sizes=[1, 1, rows],
                        dst_strides=[herd_cols * rows, rows, 1],
                    )

                    DeallocOp(l1_act)
                    DeallocOp(l1_deq)
                    DeallocOp(l1_out)

                dma_memcpy_nd(
                    so,
                    l2_out,
                    dst_offsets=[base, 0],
                    dst_sizes=[pipe_tiles, rows],
                    dst_strides=[rows, 1],
                    src_offsets=[0, 0, 0],
                    src_sizes=[pipe_rows, herd_cols, rows],
                    src_strides=[herd_cols * rows, rows, 1],
                )
                DeallocOp(l2_out)
                DeallocOp(l2_act)
                DeallocOp(l2_pack)


def compile_module(module, args):
    backend = XRTBackend(
        verbose=args.verbose,
        omit_pingpong=True,
        output_format=args.output_format,
        instance_name="fused_dqp_channel_repro",
        target_device="npu2",
        runtime_loop_tiling_sizes=[1, 1],
        use_lock_race_condition_fix=True,
        debug_ir=args.debug_ir,
    )
    backend.compile(module)
    backend.unload()


def run_module(module, args):
    input_pack = np.zeros((8, 4, args.col_blocks, args.block_bytes), dtype=np.int8)
    input_act = np.zeros((args.col_blocks, args.cols), dtype=bfloat16)
    expected = np.zeros((32, args.rows), dtype=bfloat16)

    runner = XRTRunner(
        verbose=args.verbose,
        omit_pingpong=True,
        output_format=args.output_format,
        instance_name="fused_dqp_channel_repro",
        target_device="npu2",
        runtime_loop_tiling_sizes=[1, 1],
        use_lock_race_condition_fix=True,
        debug_ir=args.debug_ir,
    )
    return runner.run_test(module, inputs=[input_pack, input_act], expected_outputs=[expected])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--debug-ir", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=["compile-only", "compile-and-run"],
        default="compile-only",
    )
    parser.add_argument("--output-format", choices=["xclbin", "elf"], default="elf")
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--cols", type=int, default=32)
    parser.add_argument("--col-blocks", type=int, default=2)
    args = parser.parse_args()
    args.block_bytes = args.rows * args.cols // 2 + 2 * args.cols * np.dtype(bfloat16).itemsize

    module = build_module(rows=args.rows, cols=args.cols, col_blocks=args.col_blocks)
    if args.print_module_only:
        print(module)
        return
    if args.compile_mode == "compile-and-run":
        raise SystemExit(run_module(module, args))
    compile_module(module, args)


if __name__ == "__main__":
    main()
