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
POWER2_TILE_M = 512
POWER2_TILE_N = 1024
EXTERNAL_MMUL_FUNCTION = "matmul_i8_i8_i8_acc32_strix"
DEFAULT_EXTERNAL_K_PACKS = 9
DEFAULT_EXTERNAL_BLOCK_M = 3
DEFAULT_EXTERNAL_BLOCK_N = 2
DEFAULT_EXTERNAL_CORE_M_PACKS = 18
DEFAULT_EXTERNAL_ACTIVE_M_PACKS = 18
DEFAULT_EXTERNAL_CORE_N_PACKS = 18
DEFAULT_EXTERNAL_SCHEDULE = "software-pipeline"
DEFAULT_EXTERNAL_KERNEL_STYLE = "hand-scheduled"
EXTERNAL_KERNEL_STYLES = ("hand-scheduled",)


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


def render_sota_int8_transform(
    transform_text: str,
    k_packs: int = 9,
    m_packs: int = 18,
    n_packs: int = 18,
) -> str:
    k_elements = k_packs * 8
    m_elements = m_packs * 8
    n_elements = n_packs * 8
    replacements = [
        (
            "tile_using_for %copy1 tile_sizes [0, 64]",
            f"tile_using_for %copy1 tile_sizes [0, {k_elements}]",
        ),
        (
            "tile_using_for %copy2 tile_sizes [64]",
            f"tile_using_for %copy2 tile_sizes [{k_elements}]",
        ),
        (
            "tile_using_for %packed_c tile_sizes [0, 0, 8]",
            f"tile_using_for %packed_c tile_sizes [0, 0, {k_packs}]",
        ),
        (
            "tile_using_forall %matmul_1 tile_sizes [8, 8, 0]",
            f"tile_using_forall %matmul_1 tile_sizes [{m_packs}, {n_packs}, 0]",
        ),
        (
            "tile_using_forall %interchanged_fill_op tile_sizes [8, 8]",
            f"tile_using_forall %interchanged_fill_op tile_sizes [{m_packs}, {n_packs}]",
        ),
        (
            "tile_using_forall %unpack_op tile_sizes [64, 64]",
            f"tile_using_forall %unpack_op tile_sizes [{m_elements}, {n_elements}]",
        ),
        (
            "%herd1 = transform.air.par_to_herd %parallel1 :",
            "%herd1 = transform.air.par_to_herd %parallel1 {first_dim = 1} :",
        ),
        (
            "%herd2 = transform.air.par_to_herd %parallel2 :",
            "%herd2 = transform.air.par_to_herd %parallel2 {first_dim = 1} :",
        ),
        (
            "%herd3 = transform.air.par_to_herd %parallel3 :",
            "%herd3 = transform.air.par_to_herd %parallel3 {first_dim = 1} :",
        ),
    ]
    rendered = transform_text
    for old, new in replacements:
        if rendered.count(old) != 1:
            raise ValueError(f"expected one transform fragment for {old!r}")
        rendered = rendered.replace(old, new, 1)
    return rendered


def render_external_mmul_transform(
    transform_text: str,
    k_packs: int,
    m_packs: int,
    n_packs: int,
    fuse_l3_l2: bool = True,
) -> str:
    rendered = render_sota_int8_transform(
        transform_text, k_packs, m_packs=m_packs, n_packs=n_packs
    )
    phase8_marker = (
        "    //==========================================================================\n"
        "    // PHASE 8:"
    )
    phase9_marker = (
        "    //==========================================================================\n"
        "    // PHASE 9:"
    )
    split_marker = phase9_marker if fuse_l3_l2 else phase8_marker
    if rendered.count(split_marker) != 1:
        raise ValueError(f"expected one transform marker for {split_marker!r}")
    prefix = rendered.split(split_marker, 1)[0]
    return (
        prefix
        + f"""    //==========================================================================
    // PHASE 9: ROUTE COMPUTE TO EXTERNAL AIE2P MMUL KERNEL
    // Purpose: Keep the existing memory layout and herd mapping while replacing
    // the scalar/vector.contract compute body with a linked Peano mmul kernel.
    //==========================================================================

        %generic_fill = transform.structured.match ops{{["linalg.generic"]}} attributes{{init_fill}} in %arg1 : (!transform.any_op) -> !transform.any_op
        %inner_most_fills, %vec_fill_loops:2 =
          transform.structured.tile_using_for %generic_fill tile_sizes [1, 1]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

        %matmul_external = transform.structured.match ops{{["linalg.generic"]}} attributes{{matmul_compute}} in %arg1 : (!transform.any_op) -> !transform.any_op
        %mmul_name = transform.param.constant "{EXTERNAL_MMUL_FUNCTION}" -> !transform.any_param
        transform.annotate %matmul_external "library_call" = %mmul_name : !transform.any_op, !transform.any_param

    //==========================================================================
    // PHASE 10: CONVERT TO AIE HERDS AND VECTORIZE NON-COMPUTE HERDS
    // Purpose: Map parallel work to the 8x4 logical AIE split. The compute herd
    // stays as a library call so aircc lowers it to the linked AIE2P object.
    //==========================================================================

        %forall1 = transform.structured.match ops{{["scf.forall"]}} attributes{{prologue_forall}} in %arg1 : (!transform.any_op) -> !transform.any_op
        %forall2 = transform.structured.match ops{{["scf.forall"]}} attributes{{compute_forall}} in %arg1 : (!transform.any_op) -> !transform.any_op
        %forall3 = transform.structured.match ops{{["scf.forall"]}} attributes{{epilogue_forall}} in %arg1 : (!transform.any_op) -> !transform.any_op
        %parallel1 = transform.loop.forall_to_parallel %forall1  : (!transform.any_op) -> !transform.any_op
        %herd1 = transform.air.par_to_herd %parallel1 {{first_dim = 1}} : (!transform.any_op) -> !transform.any_op
        transform.annotate %herd1 "prologue_herd" : !transform.any_op
        %parallel2 = transform.loop.forall_to_parallel %forall2  : (!transform.any_op) -> !transform.any_op
        %herd2 = transform.air.par_to_herd %parallel2 {{first_dim = 1}} : (!transform.any_op) -> !transform.any_op
        transform.annotate %herd2 "compute_herd" : !transform.any_op
        %parallel3 = transform.loop.forall_to_parallel %forall3  : (!transform.any_op) -> !transform.any_op
        %herd3 = transform.air.par_to_herd %parallel3 {{first_dim = 1}} : (!transform.any_op) -> !transform.any_op
        transform.annotate %herd3 "epilogue_herd" : !transform.any_op

        %vectorized_herd1 = transform.air.herd_vectorize %herd1 : (!transform.any_op) -> !transform.any_op
        %vectorized_herd3 = transform.air.herd_vectorize %herd3 : (!transform.any_op) -> !transform.any_op

        %func7 = transform.structured.match ops{{["func.func"]}} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func7 {{
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
            transform.apply_patterns.memref.fold_memref_alias_ops
        }} : !transform.any_op
        %func_fold_1 = transform.structured.match ops{{["func.func"]}} in %arg1 : (!transform.any_op) -> !transform.any_op
        %func_folded_1 = transform.air.fold_unit_extent_dims %func_fold_1 : (!transform.any_op) -> !transform.any_op

    transform.yield
  }}
}}
"""
    )

def render_transform_variant(
    transform_text: str,
    variant: str,
    kernel_impl: str,
    external_k_packs: int,
    problem_k: int,
    external_active_m_packs: int,
    external_core_n_packs: int,
) -> str:
    if kernel_impl == "external-mmul":
        if variant != "sota-int8":
            raise ValueError("external-mmul requires --transform-variant=sota-int8")
        fuse_l3_l2 = problem_k > external_k_packs * 8
        return render_external_mmul_transform(
            transform_text,
            external_k_packs,
            external_active_m_packs,
            external_core_n_packs,
            fuse_l3_l2=fuse_l3_l2,
        )
    if variant == "default":
        return transform_text
    if variant == "sota-int8":
        return render_sota_int8_transform(transform_text)
    raise ValueError(f"unknown transform variant: {variant}")


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
        # Physical B is packed by N tile: [N/tile_N, K, tile_N + 4]. The
        # padded pitch prevents AIR-to-AIE channel lowering from collapsing the
        # selected B tile into a default-contiguous view and dropping the
        # N-tile base offset.
        b_pitch = tile_n + 4
        b_tile_span = k * b_pitch
        b_offset_code = f"""%n_tile_index = arith.index_cast %arg7 : i32 to index
    %cBTileSpan = arith.constant {b_tile_span} : index
    %b_offset = arith.muli %n_tile_index, %cBTileSpan : index"""
        b_offset_value = "%b_offset"
        b_strides = f"[{b_pitch}, 1]"
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
    %c_full = memref.reinterpret_cast %arg2 to offset: [0], sizes: [{m}, {n}], strides: [{n}, 1] : memref<*x{out_mlir}> to memref<{m}x{n}x{out_mlir}, strided<[{n}, 1]>>
    %c_tile = memref.subview %c_full[%m_offset, %n_offset] [{tile_m}, {tile_n}] [1, 1] : memref<{m}x{n}x{out_mlir}, strided<[{n}, 1]>> to memref<{tile_m}x{tile_n}x{out_mlir}, strided<[{n}, 1], offset: ?>>
    bufferization.materialize_in_destination %matmul in writable %c_tile : (tensor<{tile_m}x{tile_n}x{out_mlir}>, memref<{tile_m}x{tile_n}x{out_mlir}, strided<[{n}, 1], offset: ?>>) -> ()
    return
  }}
}}
"""


def copy_aircc_lowered_ir(artifact_dir: Path | None) -> None:
    if artifact_dir is None:
        return
    debug_dir = Path("air_project") / "debug_ir"
    candidates = sorted(debug_dir.glob("*_after_air-verify-hierarchy-locality.mlir"))
    if not candidates:
        return
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "air_module.lowered.mlir").write_text(
        candidates[-1].read_text(encoding="utf-8"), encoding="utf-8"
    )


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
        "kernel_impl": args.kernel_impl,
        "external_kernel_object": args.external_kernel_object,
        "external_schedule": args.external_schedule,
        "external_kernel_style": args.external_kernel_style,
        "external_k_packs": args.external_k_packs,
        "external_block_m": args.external_block_m,
        "external_block_n": args.external_block_n,
        "external_core_m_packs": args.external_core_m_packs,
        "external_active_m_packs": args.external_active_m_packs,
        "external_core_n_packs": args.external_core_n_packs,
        "external_c_stride_m_packs": args.external_c_stride_m_packs,
        "aircc_debug_ir": args.aircc_debug_ir,
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
        "--kernel-impl",
        choices=["vectorized", "external-mmul"],
        default="vectorized",
        help="Compute implementation selected after tiling (default: vectorized)",
    )
    parser.add_argument(
        "--external-kernel-object",
        default=None,
        help="Object file linked when --kernel-impl=external-mmul",
    )
    parser.add_argument(
        "--external-schedule",
        choices=["software-pipeline"],
        default=DEFAULT_EXTERNAL_SCHEDULE,
        help="Peano schedule annotation mode for --kernel-impl=external-mmul",
    )
    parser.add_argument(
        "--external-kernel-style",
        choices=EXTERNAL_KERNEL_STYLES,
        default=DEFAULT_EXTERNAL_KERNEL_STYLE,
        help="External kernel source style selected at compile time",
    )
    parser.add_argument(
        "--external-k-packs",
        type=positive_int,
        default=DEFAULT_EXTERNAL_K_PACKS,
        help="Packed K tiles consumed by the external mmul kernel (default: 9)",
    )
    parser.add_argument(
        "--external-block-m",
        type=positive_int,
        default=DEFAULT_EXTERNAL_BLOCK_M,
        help="External mmul register-block packs along M (default: 3)",
    )
    parser.add_argument(
        "--external-block-n",
        type=positive_int,
        default=DEFAULT_EXTERNAL_BLOCK_N,
        help="External mmul register-block packs along N (default: 2)",
    )
    parser.add_argument(
        "--external-core-m-packs",
        type=positive_int,
        default=DEFAULT_EXTERNAL_CORE_M_PACKS,
        help="Full per-core C M packs used for the external kernel C stride",
    )
    parser.add_argument(
        "--external-active-m-packs",
        type=positive_int,
        default=DEFAULT_EXTERNAL_ACTIVE_M_PACKS,
        help="Active M packs consumed by each external kernel call",
    )
    parser.add_argument(
        "--external-core-n-packs",
        type=positive_int,
        default=DEFAULT_EXTERNAL_CORE_N_PACKS,
        help="Full per-core C/B N packs consumed by the external kernel",
    )
    parser.add_argument(
        "--external-c-stride-m-packs",
        type=positive_int,
        default=DEFAULT_EXTERNAL_CORE_M_PACKS,
        help="C stride in M packs used by the external kernel",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Only compile without running validation",
    )
    parser.add_argument(
        "--aircc-debug-ir",
        action="store_true",
        help="Forward --debug-ir to aircc and keep pass-by-pass IR under the build directory",
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
    if args.kernel_impl == "external-mmul":
        errors = []
        if args.transform_variant != "sota-int8":
            errors.append("external-mmul requires --transform-variant=sota-int8")
        if args.output_type != "int8":
            errors.append("external-mmul currently supports --output-type=int8 only")
        if args.b_layout != "column":
            errors.append("external-mmul requires --b-layout=column")
        if args.external_schedule != DEFAULT_EXTERNAL_SCHEDULE:
            errors.append(
                f"external-mmul requires external_schedule={DEFAULT_EXTERNAL_SCHEDULE}"
            )
        if args.external_kernel_style != DEFAULT_EXTERNAL_KERNEL_STYLE:
            errors.append(
                f"external-mmul requires external_kernel_style={DEFAULT_EXTERNAL_KERNEL_STYLE}"
            )
        if args.external_c_stride_m_packs != args.external_core_m_packs:
            errors.append("external-mmul requires C stride to match core M packs")
        expected_tile_m = args.external_active_m_packs * 8 * 4
        expected_tile_n = args.external_core_n_packs * 8 * 8
        if args.tile_m != expected_tile_m or args.tile_n != expected_tile_n:
            errors.append(
                "external-mmul tile shape must match the 8x4 herd core tile: "
                f"expected {expected_tile_m}x{expected_tile_n}"
            )
        k_residency = args.external_k_packs * 8
        supported_external_configs = {
            (9, 3, 2, 18, 18, 18, 18): f"legacy SOTA {SOTA_TILE_M}x{SOTA_TILE_N}",
            (8, 2, 2, 16, 16, 16, 16): f"power-of-two {POWER2_TILE_M}x{POWER2_TILE_N}",
        }
        actual_external_config = (
            args.external_k_packs,
            args.external_block_m,
            args.external_block_n,
            args.external_core_m_packs,
            args.external_active_m_packs,
            args.external_core_n_packs,
            args.external_c_stride_m_packs,
        )
        if actual_external_config not in supported_external_configs:
            supported = ", ".join(supported_external_configs.values())
            errors.append(f"unsupported external-mmul profile; supported profiles: {supported}")
        if args.k % k_residency:
            errors.append(f"external-mmul requires K to be a multiple of {k_residency}")
        if not args.external_kernel_object:
            errors.append("external-mmul requires --external-kernel-object")
        if errors:
            raise ValueError("; ".join(errors))
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
        args.kernel_impl,
        args.external_k_packs,
        args.k,
        args.external_active_m_packs,
        args.external_core_n_packs,
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
        debug_ir=args.aircc_debug_ir,
    )
    if args.kernel_impl == "external-mmul":
        backend_kwargs["lower_linalg_to_func"] = args.external_kernel_object
    if args.compile_only:
        print(f"Compile-only mode: generating {output_ext} binary...")
        backend = XRTBackend(**backend_kwargs)
        backend.compile(air_module)
        copy_aircc_lowered_ir(Path(args.artifact_dir) if args.artifact_dir else None)
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
        print(f"kernel_impl={args.kernel_impl}")
        if args.external_kernel_object:
            print(f"external_kernel_object={args.external_kernel_object}")
        print(f"external_schedule={args.external_schedule}")
        print(f"external_kernel_style={args.external_kernel_style}")
        print(f"external_k_packs={args.external_k_packs}")
        print(f"external_block={args.external_block_m}x{args.external_block_n}")
        print(f"external_core_m_packs={args.external_core_m_packs}")
        print(f"external_active_m_packs={args.external_active_m_packs}")
        print(f"external_core_n_packs={args.external_core_n_packs}")
        print(f"external_c_stride_m_packs={args.external_c_stride_m_packs}")
        print(f"aircc_debug_ir={args.aircc_debug_ir}")
        raise SystemExit(0)

    input_type = np.int8
    output_type = np.int32 if args.output_type == "int32" else np.int8
    rng = np.random.default_rng(args.seed)
    A = rng.integers(low=0, high=8, size=(args.m, args.k), dtype=input_type)
    B = rng.integers(low=0, high=8, size=(args.k, args.n), dtype=input_type)
    if args.b_layout == "row":
        B_device = B
    else:
        b_tiles = args.n // args.tile_n
        b_pitch = args.tile_n + 4
        B_device = np.zeros((b_tiles, args.k, b_pitch), dtype=input_type)
        for tile in range(b_tiles):
            start = tile * args.tile_n
            B_device[tile, :, : args.tile_n] = B[:, start : start + args.tile_n]
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
