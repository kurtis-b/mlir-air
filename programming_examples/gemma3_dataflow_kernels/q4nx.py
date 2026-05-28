# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Q4NX block dequantization example."""

import argparse

import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.air import *
from air.dialects.func import FuncOp, CallOp
from air.dialects.memref import AllocOp, DeallocOp
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner, type_mapper

from common import random_q4nx_block, q4nx_dequant_reference


@module_builder
def build_module(rows, cols):
    bf16_type = type_mapper(bfloat16)
    i8_type = IntegerType.get_signless(8)
    packed_elems = rows * cols // 2

    l3_w_ty = MemRefType.get([packed_elems], i8_type)
    l3_param_ty = MemRefType.get([cols], bf16_type)
    l3_out_ty = MemRefType.get([rows, cols], bf16_type)

    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_w_ty = MemRefType.get([packed_elems], i8_type, memory_space=l1_space)
    l1_param_ty = MemRefType.get([cols], bf16_type, memory_space=l1_space)
    l1_out_ty = MemRefType.get([rows, cols], bf16_type, memory_space=l1_space)

    dequant_func = FuncOp(
        "q4nx_dequant_block",
        ([l1_w_ty, l1_param_ty, l1_param_ty, l1_out_ty], []),
        visibility="private",
    )
    dequant_func.attributes["link_with"] = StringAttr.get("q4nx.o")
    dequant_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    @FuncOp.from_py_func(l3_w_ty, l3_param_ty, l3_param_ty, l3_out_ty)
    def q4nx_dequant(arg_w, arg_scale, arg_min, arg_out):
        @launch(operands=[arg_w, arg_scale, arg_min, arg_out])
        def launch_body(lw, ls, lm, lo):
            @segment(name="q4nx_seg", operands=[lw, ls, lm, lo])
            def segment_body(sw, ss, sm, so):
                @herd(
                    name="q4nx_herd",
                    sizes=[1, 1],
                    operands=[sw, ss, sm, so],
                    link_with="q4nx.o",
                )
                def herd_body(_tx, _ty, _sx, _sy, hw, hs, hm, ho):
                    l1_w = AllocOp(l1_w_ty, [], [])
                    l1_s = AllocOp(l1_param_ty, [], [])
                    l1_m = AllocOp(l1_param_ty, [], [])
                    l1_out = AllocOp(l1_out_ty, [], [])

                    dma_memcpy_nd(l1_w, hw)
                    dma_memcpy_nd(l1_s, hs)
                    dma_memcpy_nd(l1_m, hm)
                    CallOp(dequant_func, [l1_w, l1_s, l1_m, l1_out])
                    dma_memcpy_nd(ho, l1_out)

                    DeallocOp(l1_w)
                    DeallocOp(l1_s)
                    DeallocOp(l1_m)
                    DeallocOp(l1_out)


def main():
    parser = argparse.ArgumentParser(description="Q4NX block dequantization")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--cols", type=int, default=256)
    parser.add_argument(
        "--compile-mode",
        choices=["compile-only", "compile-and-run"],
        default="compile-and-run",
    )
    parser.add_argument("--output-format", choices=["xclbin", "elf"], default="xclbin")
    args = parser.parse_args()

    if args.rows * args.cols % 2 != 0:
        parser.error("rows*cols must be even for int4 packing")

    module = build_module(args.rows, args.cols)
    if args.print_module_only:
        print(module)
        return

    _, packed, scale, min_offset = random_q4nx_block(args.rows, args.cols, seed=1)
    expected = q4nx_dequant_reference(packed, scale, min_offset, args.rows, args.cols)

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
                inputs=[packed, scale, min_offset],
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
