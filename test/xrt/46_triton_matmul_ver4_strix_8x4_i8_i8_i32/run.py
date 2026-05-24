# run.py -*- Python -*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner
from air.compiler.util import run_transform
from air.ir import *
import air.passmanager


DEFAULT_M = 1024
DEFAULT_K = 1024
DEFAULT_N = 1024
DEFAULT_TILE_M = 512
DEFAULT_TILE_N = 256
SOTA_TILE_M = 576
SOTA_TILE_N = 1152


def positive_int(value: str) -> int:
    parsed = int(value, 10)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value, 10)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def parse_runtime_loop_tiling(value: str) -> list[int]:
    try:
        parsed = [positive_int(item.strip()) for item in value.split(",")]
    except argparse.ArgumentTypeError:
        raise
    if len(parsed) != 2:
        raise argparse.ArgumentTypeError(
            "runtime loop tiling must contain two positive integers"
        )
    return parsed


def validate_shape(m: int, k: int, n: int, tile_m: int, tile_n: int) -> None:
    errors = []
    if m % tile_m:
        errors.append(f"M must be a multiple of tile M ({tile_m})")
    if n % tile_n:
        errors.append(f"N must be a multiple of tile N ({tile_n})")
    if k % 8:
        errors.append("K must be a multiple of 8")
    if tile_m % 8:
        errors.append("tile M must be a multiple of 8")
    if tile_n % 8:
        errors.append("tile N must be a multiple of 8")
    if errors:
        raise ValueError("; ".join(errors))


def render_transform_variant(transform_text: str, variant: str) -> str:
    if variant == "default":
        return transform_text
    if variant != "sota-int8":
        raise ValueError(f"unknown transform variant: {variant}")

    replacements = [
        (
            "tile_using_for %copy1 tile_sizes [0, 64]",
            "tile_using_for %copy1 tile_sizes [0, 72]",
        ),
        (
            "tile_using_for %copy2 tile_sizes [64]",
            "tile_using_for %copy2 tile_sizes [72]",
        ),
        (
            "tile_using_for %packed_c tile_sizes [0, 0, 8]",
            "tile_using_for %packed_c tile_sizes [0, 0, 9]",
        ),
        (
            "tile_using_forall %matmul_1 tile_sizes [8, 8, 0]",
            "tile_using_forall %matmul_1 tile_sizes [18, 18, 0]",
        ),
        (
            "tile_using_forall %interchanged_fill_op tile_sizes [8, 8]",
            "tile_using_forall %interchanged_fill_op tile_sizes [18, 18]",
        ),
        (
            "tile_using_forall %unpack_op tile_sizes [64, 64]",
            "tile_using_forall %unpack_op tile_sizes [144, 144]",
        ),
    ]
    rendered = transform_text
    for old, new in replacements:
        if rendered.count(old) != 1:
            raise ValueError(f"expected one transform fragment for {old!r}")
        rendered = rendered.replace(old, new, 1)
    return rendered


def build_matmul_ir(
    m: int,
    k: int,
    n: int,
    tile_m: int,
    tile_n: int,
    output_type: str,
    b_layout: str,
) -> str:
    out_mlir = "i32" if output_type == "int32" else "i8"
    zero = "%c0_i32 = arith.constant 0 : i32"
    zero_value = "%c0_i32"
    if output_type == "int8":
        zero = "%c0_i8 = arith.constant 0 : i8"
        zero_value = "%c0_i8"

    if b_layout == "row":
        b_offset_code = ""
        b_offset_value = "%n_offset"
        b_strides = f"[{n}, 1]"
        b_tensor_type = f"tensor<{k}x{tile_n}xi8>"
        matmul_op = "linalg.matmul"
        b_tensor_code = f"""    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [{b_offset_value}], sizes: [{k}, {tile_n}], strides: {b_strides} : memref<*xi8> to memref<{k}x{tile_n}xi8, strided<{b_strides}, offset: ?>>
    %alloc_1 = memref.alloc() : memref<{k}x{tile_n}xi8>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<{k}x{tile_n}xi8, strided<{b_strides}, offset: ?>> to memref<{k}x{tile_n}xi8>
    %b_tensor = bufferization.to_tensor %alloc_1 restrict writable : memref<{k}x{tile_n}xi8> to tensor<{k}x{tile_n}xi8>"""
    else:
        # Physical B is stored as padded column-major. Logical B[k,n] is at
        # byte offset (n * K + k) * 4, which preserves column-major traversal
        # while satisfying the NPU DMA 32-bit address-generation granularity.
        b_offset_code = ""
        b_offset_value = "%n_offset"
        b_strides = f"[4, {k * 4}]"
        b_tensor_type = f"tensor<{k}x{tile_n}xi8>"
        matmul_op = "linalg.matmul"
        b_tensor_code = f"""    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [{b_offset_value}], sizes: [{k}, {tile_n}], strides: {b_strides} : memref<*xi8> to memref<{k}x{tile_n}xi8, strided<{b_strides}, offset: ?>>
    %alloc_1 = memref.alloc() : memref<{k}x{tile_n}xi8>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<{k}x{tile_n}xi8, strided<{b_strides}, offset: ?>> to memref<{k}x{tile_n}xi8>
    %b_tensor = bufferization.to_tensor %alloc_1 restrict writable : memref<{k}x{tile_n}xi8> to tensor<{k}x{tile_n}xi8>"""

    return f"""// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

module {{
  func.func @bare_matmul(%arg0: memref<*xi8> {{tt.divisibility = 16 : i32}}, %arg1: memref<*xi8> {{tt.divisibility = 16 : i32}}, %arg2: memref<*x{out_mlir}> {{tt.divisibility = 16 : i32}}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {{
    {zero}
    %cK = arith.constant {k} : index
    %cN = arith.constant {n} : index
    %c4 = arith.constant 4 : index
    %cTileN_i32 = arith.constant {tile_n} : i32
    %cTileM_i32 = arith.constant {tile_m} : i32
    %m_tile_i32 = arith.muli %arg6, %cTileM_i32 : i32
    %m_offset = arith.index_cast %m_tile_i32 : i32 to index
    %n_tile_i32 = arith.muli %arg7, %cTileN_i32 : i32
    %n_offset = arith.index_cast %n_tile_i32 : i32 to index
    %a_offset = arith.muli %m_offset, %cK : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%a_offset], sizes: [{tile_m}, {k}], strides: [{k}, 1] : memref<*xi8> to memref<{tile_m}x{k}xi8, strided<[{k}, 1], offset: ?>>
    %alloc = memref.alloc() : memref<{tile_m}x{k}xi8>
    memref.copy %reinterpret_cast, %alloc : memref<{tile_m}x{k}xi8, strided<[{k}, 1], offset: ?>> to memref<{tile_m}x{k}xi8>
    %a_tensor = bufferization.to_tensor %alloc restrict writable : memref<{tile_m}x{k}xi8> to tensor<{tile_m}x{k}xi8>
    {b_offset_code}
{b_tensor_code}
    %empty = tensor.empty() : tensor<{tile_m}x{tile_n}x{out_mlir}>
    %filled = linalg.fill ins({zero_value} : {out_mlir}) outs(%empty : tensor<{tile_m}x{tile_n}x{out_mlir}>) -> tensor<{tile_m}x{tile_n}x{out_mlir}>
    %matmul = {matmul_op} ins(%a_tensor, %b_tensor : tensor<{tile_m}x{k}xi8>, {b_tensor_type}) outs(%filled : tensor<{tile_m}x{tile_n}x{out_mlir}>) -> tensor<{tile_m}x{tile_n}x{out_mlir}>
    %c_row_offset = arith.muli %m_offset, %cN : index
    %c_offset = arith.addi %c_row_offset, %n_offset : index
    %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [%c_offset], sizes: [{tile_m}, {tile_n}], strides: [{n}, 1] : memref<*x{out_mlir}> to memref<{tile_m}x{tile_n}x{out_mlir}, strided<[{n}, 1], offset: ?>>
    bufferization.materialize_in_destination %matmul in writable %reinterpret_cast_2 : (tensor<{tile_m}x{tile_n}x{out_mlir}>, memref<{tile_m}x{tile_n}x{out_mlir}, strided<[{n}, 1], offset: ?>>) -> ()
    return
  }}
}}
"""


def write_compile_config(args: argparse.Namespace, generated_ir: str | None) -> None:
    if not args.artifact_dir:
        return
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "m": args.m,
        "k": args.k,
        "n": args.n,
        "tile_m": args.tile_m,
        "tile_n": args.tile_n,
        "output_type": args.output_type,
        "b_layout": args.b_layout,
        "runtime_loop_tiling_sizes": args.runtime_loop_tiling_sizes,
        "transform_script": args.transform_script,
        "transform_variant": args.transform_variant,
        "input_ir": args.input_ir or "generated",
        "output_format": args.output_format,
        "target_device": args.target_device,
        "trace_size": args.trace_size,
        "trace_offset": args.trace_offset,
    }
    (artifact_dir / "compile_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if generated_ir is not None:
        (artifact_dir / "generated_input.mlir").write_text(
            generated_ir, encoding="utf-8"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the Strix int8 matmul example",
    )
    parser.add_argument("--input-ir", default=None, help="Optional input IR file path")
    parser.add_argument(
        "--transform-script",
        default="transform.mlir",
        help="Transform script path",
    )
    parser.add_argument("-M", "--m", type=positive_int, default=DEFAULT_M)
    parser.add_argument("-K", "--k", type=positive_int, default=DEFAULT_K)
    parser.add_argument("-N", "--n", type=positive_int, default=DEFAULT_N)
    parser.add_argument("--tile-m", type=positive_int, default=DEFAULT_TILE_M)
    parser.add_argument("--tile-n", type=positive_int, default=DEFAULT_TILE_N)
    parser.add_argument(
        "--output-type",
        choices=["int32", "int8"],
        default="int32",
        help="Output element type generated for C (default: int32)",
    )
    parser.add_argument(
        "--b-layout",
        choices=["row", "column"],
        default="row",
        help="Host/device layout expected for B (default: row)",
    )
    parser.add_argument(
        "--transform-variant",
        choices=["default", "sota-int8"],
        default="default",
        help="Optional transform rewrite applied after loading --transform-script",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Only compile without running validation",
    )
    parser.add_argument(
        "--output-format",
        choices=["elf", "xclbin"],
        default="xclbin",
        help="Output format: xclbin (default) or elf",
    )
    parser.add_argument(
        "--debug-ir",
        default=None,
        metavar="OUTPUT_FILE",
        help="Print the transformed IR to the specified file and exit",
    )
    parser.add_argument(
        "--runtime-loop-tiling-sizes",
        default="2,4",
        metavar="M,N",
        help="Comma-separated AIR runtime loop tiling sizes (default: 2,4)",
    )
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="Directory for compile metadata and generated input IR",
    )
    parser.add_argument(
        "--trace-size",
        type=nonnegative_int,
        default=0,
        help="Trace buffer size in bytes (0 disables trace plumbing)",
    )
    parser.add_argument(
        "--trace-offset",
        type=nonnegative_int,
        default=0,
        help="Trace buffer offset in bytes",
    )
    parser.add_argument(
        "--trace-file",
        default="trace_data.txt",
        help="Trace output file used by XRTRunner validation mode",
    )
    parser.add_argument(
        "--target-device",
        default="npu2",
        help="XRTBackend target device (default: npu2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic RNG seed for validation inputs",
    )
    args = parser.parse_args()
    args.runtime_loop_tiling_sizes = parse_runtime_loop_tiling(
        args.runtime_loop_tiling_sizes
    )
    validate_shape(args.m, args.k, args.n, args.tile_m, args.tile_n)
    return args


args = parse_args()

with air.ir.Context() as ctx, Location.unknown():
    if args.input_ir:
        air_tiled_ir_string = Path(args.input_ir).read_text(encoding="utf-8")
        generated_ir = None
    else:
        air_tiled_ir_string = build_matmul_ir(
            args.m,
            args.k,
            args.n,
            args.tile_m,
            args.tile_n,
            args.output_type,
            args.b_layout,
        )
        generated_ir = air_tiled_ir_string
    write_compile_config(args, generated_ir)

    air_module = Module.parse(air_tiled_ir_string)

    pipeline = (
        "builtin.module(air-override-memref-memory-space{scope=func memory-space=1})"
    )
    pm = air.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)

    transform_ir_string = render_transform_variant(
        Path(args.transform_script).read_text(encoding="utf-8"),
        args.transform_variant,
    )
    if args.artifact_dir:
        artifact_dir = Path(args.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "effective_transform.mlir").write_text(
            transform_ir_string, encoding="utf-8"
        )
    transform_ir = Module.parse(transform_ir_string)
    run_transform(transform_ir, air_module)

    if args.debug_ir:
        output_file = Path(args.debug_ir)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(str(air_module), encoding="utf-8")
        print(f"Transformed IR written to {output_file}")
        raise SystemExit(0)

    input_size = (args.m, args.n, args.k)
    tile_size = (args.tile_m, args.tile_n, args.k)
    launch_size = tuple(i // t for i, t in zip(input_size, tile_size))

    pipeline = (
        "builtin.module("
        + ",".join(
            [
                f"func.func(air-wrap-func-with-parallel{{loop-bounds={launch_size[0]},{launch_size[1]},{launch_size[2]}}})",
                "air-par-to-launch{depth=0 has-air-segment=true}",
                "canonicalize",
                "cse",
                "air-copy-to-dma",
            ]
        )
        + ")"
    )
    pm = air.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)

    if args.artifact_dir:
        artifact_dir = Path(args.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "air_module.final.mlir").write_text(
            str(air_module), encoding="utf-8"
        )

    output_ext = "elf" if args.output_format == "elf" else "xclbin"
    backend_kwargs = dict(
        omit_while_true_loop=False,
        output_format=args.output_format,
        instance_name="bare_matmul",
        runtime_loop_tiling_sizes=args.runtime_loop_tiling_sizes,
        trace_offset=args.trace_offset,
        trace_size=args.trace_size,
        target_device=args.target_device,
    )

    if args.compile_only:
        print(f"Compile-only mode: generating {output_ext} binary...")
        backend = XRTBackend(**backend_kwargs)
        backend.compile(air_module)
        backend.unload()
        print("Compilation complete. Generated files:")
        print(f"  - air.{output_ext}")
        if args.output_format == "xclbin":
            print("  - air.insts.bin")
        print(f"shape={args.m}x{args.k}x{args.n}")
        print(f"tile_shape={args.tile_m}x{args.tile_n}x{args.k}")
        print(f"output_type={args.output_type}")
        print(f"b_layout={args.b_layout}")
        print(f"transform_variant={args.transform_variant}")
        raise SystemExit(0)

    input_type = np.int8
    output_type = np.int32 if args.output_type == "int32" else np.int8
    rng = np.random.default_rng(args.seed)
    A = rng.integers(low=0, high=8, size=(args.m, args.k), dtype=input_type)
    B = rng.integers(low=0, high=8, size=(args.k, args.n), dtype=input_type)
    if args.b_layout == "row":
        B_device = B
    else:
        B_device = np.zeros((args.n, args.k, 4), dtype=input_type)
        B_device[:, :, 0] = B.T
        B_device = B_device.reshape(-1)

    C_i32 = np.matmul(A.astype(np.int32), B.astype(np.int32))
    C = C_i32.astype(output_type)

    runner = XRTRunner(
        **backend_kwargs,
        trace_file=args.trace_file,
    )
    raise SystemExit(
        runner.run_test(
            air_module,
            inputs=[A, B_device],
            expected_outputs=[C],
        )
    )
