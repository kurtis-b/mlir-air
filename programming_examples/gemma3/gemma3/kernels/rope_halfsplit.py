# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gemma3 half-split RoPE AIR wrapper.

Wraps the Llama32 half-split RoPE AIE2P microkernel for Gemma3 head dimensions.
The model loop keeps RoPE as a host fallback until model-stage launch evidence is
recorded, but this standalone wrapper provides compile and hardware smoke
coverage for the Gemma tensor contract.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from gemma3.paths import REPO_ROOT
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects.arith import ConstantOp
from air.dialects.func import FuncOp, CallOp
from air.dialects.memref import AllocOp, DeallocOp, collapse_shape as memref_collapse_shape
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend

range_ = for_


def rope_halfsplit_reference(x: np.ndarray, lut: np.ndarray) -> np.ndarray:
    xf = np.asarray(x, dtype=np.float32)
    lf = np.asarray(lut, dtype=np.float32).reshape(xf.shape)
    if xf.shape[-1] != lf.shape[-1]:
        raise ValueError("RoPE LUT head_dim must match input head_dim")
    half = xf.shape[-1] // 2
    cos_vals = lf[..., :half]
    sin_vals = lf[..., half:]
    x1 = xf[..., :half]
    x2 = xf[..., half:]
    out = np.empty_like(xf)
    out[..., :half] = x1 * cos_vals - x2 * sin_vals
    out[..., half:] = x1 * sin_vals + x2 * cos_vals
    return out.astype(bfloat16)


def _repo_root() -> Path:
    return REPO_ROOT


def _aie_api_include() -> Path:
    candidates = []
    mlir_aie = os.environ.get("MLIR_AIE_INSTALL_DIR")
    if mlir_aie:
        base = Path(mlir_aie)
        candidates.extend([
            base / "include",
            base / "lib/python3.12/site-packages/mlir_aie/include",
        ])
    candidates.append(_repo_root() / "sandbox/lib/python3.12/site-packages/mlir_aie/include")
    for candidate in candidates:
        if (candidate / "aie_api").is_dir():
            return candidate
    raise RuntimeError("could not locate aie_api include directory")


def compile_rope_kernel(object_file: Path) -> None:
    peano = os.environ.get("PEANO_INSTALL_DIR")
    if not peano:
        raise RuntimeError("PEANO_INSTALL_DIR is required to compile rope_halfsplit.cc")
    clangxx = Path(peano) / "bin/clang++"
    if not clangxx.exists():
        raise RuntimeError(f"missing Peano clang++: {clangxx}")
    src = _repo_root() / "programming_examples/llama32_1b/kernel_builder/rope_halfsplit.cc"
    object_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(clangxx),
        "-O2",
        "-std=c++20",
        "--target=aie2p-none-unknown-elf",
        "-Wno-parentheses",
        "-Wno-attributes",
        "-Wno-macro-redefined",
        "-Wno-empty-body",
        "-DNDEBUG",
        "-I",
        str(_aie_api_include()),
        "-c",
        str(src),
        "-o",
        str(object_file),
    ]
    subprocess.run(cmd, check=True)


@module_builder
def build_module(rows, head_dim, np_dtype, herd_x=4, object_file="rope.o"):
    xrt_dtype = type_mapper(np_dtype)
    total = rows * head_dim
    herd_y = 1
    total_tiles = herd_x * herd_y
    assert head_dim % 16 == 0
    assert rows % total_tiles == 0

    l3_2d_ty = MemRefType.get([rows, head_dim], xrt_dtype)
    l3_1d_ty = MemRefType.get([total], xrt_dtype)
    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_row_ty = MemRefType.get([head_dim], xrt_dtype, memory_space=l1_space)

    rope_func = FuncOp(
        "rope", ([l1_row_ty, l1_row_ty, l1_row_ty, T.i32()], []), visibility="private"
    )
    rope_func.attributes["link_with"] = StringAttr.get(object_file)
    rope_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    rows_per_tile = rows // total_tiles
    row_offset_map = AffineMap.get(
        0,
        3,
        [
            AffineExpr.get_mul(
                AffineExpr.get_add(
                    AffineSymbolExpr.get(0),
                    AffineExpr.get_mul(
                        AffineExpr.get_add(
                            AffineExpr.get_mul(AffineSymbolExpr.get(1), AffineConstantExpr.get(herd_y)),
                            AffineSymbolExpr.get(2),
                        ),
                        AffineConstantExpr.get(rows_per_tile),
                    ),
                ),
                AffineConstantExpr.get(head_dim),
            )
        ],
    )

    @FuncOp.from_py_func(l3_2d_ty, l3_1d_ty, l3_2d_ty)
    def gemma3_rope_halfsplit(arg_x, arg_lut, arg_out):
        @launch(operands=[arg_x, arg_lut, arg_out])
        def rope_launch(l_x_2d, l_lut, l_out_2d):
            x_flat = memref_collapse_shape(l3_1d_ty, l_x_2d, [[0, 1]])
            out_flat = memref_collapse_shape(l3_1d_ty, l_out_2d, [[0, 1]])

            @segment(name="rope_seg", operands=[x_flat, l_lut, out_flat])
            def rope_seg(s_x, s_lut, s_out):
                @herd(name="rope_herd", sizes=[herd_x, herd_y], operands=[s_x, s_lut, s_out])
                def rope_body(_tx, _ty, _sx, _sy, h_x, h_lut, h_out):
                    l1_x = AllocOp(l1_row_ty, [], [])
                    l1_lut = AllocOp(l1_row_ty, [], [])
                    l1_out = AllocOp(l1_row_ty, [], [])
                    dim_i32 = ConstantOp(T.i32(), head_dim)

                    for local_row in range_(rows_per_tile):
                        row_offset = affine_apply(row_offset_map, [local_row, _tx, _ty])
                        dma_memcpy_nd(l1_x, h_x, src_offsets=[row_offset], src_sizes=[head_dim], src_strides=[1])
                        dma_memcpy_nd(l1_lut, h_lut, src_offsets=[row_offset], src_sizes=[head_dim], src_strides=[1])
                        CallOp(rope_func, [l1_x, l1_lut, l1_out, dim_i32])
                        dma_memcpy_nd(h_out, l1_out, dst_offsets=[row_offset], dst_sizes=[head_dim], dst_strides=[1])
                        yield_([])

                    DeallocOp(l1_x)
                    DeallocOp(l1_lut)
                    DeallocOp(l1_out)
                rope_body.attributes["link_with"] = StringAttr.get(object_file)


def _self_test() -> None:
    x = np.linspace(-1.0, 1.0, 4 * 256, dtype=np.float32).reshape(4, 256).astype(bfloat16)
    half = 128
    lut = np.tile(
        np.concatenate([
            np.ones(half, dtype=np.float32),
            np.zeros(half, dtype=np.float32),
        ]),
        (4, 1),
    ).astype(bfloat16)
    out = rope_halfsplit_reference(x, lut)
    if not np.array_equal(out, x):
        raise AssertionError("identity RoPE LUT should preserve x")
    print(f"rope_halfsplit_reference_checksum={float(np.sum(out.astype(np.float32))):.6f}")
    print("GEMMA3_ROPE_HALFSPLIT_REFERENCE_SELF_TEST: PASS")


def _write_result_json(args, *, status: str, returncode: int) -> None:
    if args.result_json is None:
        return
    result = {
        "schema_version": 1,
        "kernel": "gemma3_rope_halfsplit",
        "status": status,
        "returncode": int(returncode),
        "rows": int(args.rows),
        "head_dim": int(args.head_dim),
        "herd_x": int(args.herd_x),
        "compile_mode": args.compile_mode,
        "output_format": args.output_format,
        "object_file": args.object_file,
        "tolerance": {"rtol": 0.05, "atol": 0.05},
    }
    args.result_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma3 half-split RoPE wrapper")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--herd-x", type=int, default=4)
    parser.add_argument("--object-file", default="rope.o")
    parser.add_argument("--print-module-only", action="store_true")
    parser.add_argument("--skip-kernel-compile", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=["compile-only", "compile-and-run"],
        default="compile-only",
    )
    parser.add_argument("--output-format", choices=["xclbin", "elf"], default="elf")
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return 0

    object_path = Path(args.object_file)
    if not args.skip_kernel_compile:
        compile_rope_kernel(object_path)
    mlir_module = build_module(args.rows, args.head_dim, bfloat16, args.herd_x, object_path.name)
    if args.print_module_only:
        print(mlir_module)
        return 0

    rng = np.random.default_rng(17)
    x = rng.uniform(-2.0, 2.0, size=(args.rows, args.head_dim)).astype(bfloat16)
    half = args.head_dim // 2
    angles = rng.uniform(-1.0, 1.0, size=(args.rows, half)).astype(np.float32)
    lut = np.concatenate([np.cos(angles), np.sin(angles)], axis=1).astype(bfloat16)

    if args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            runtime_loop_tiling_sizes=[4, 4],
        )
        backend.compile(mlir_module)
        backend.unload()
        _write_result_json(args, status="COMPILE_ONLY_PASS", returncode=0)
        print("GEMMA3_ROPE_HALFSPLIT_COMPILE_ONLY: PASS")
        return 0

    expected = rope_halfsplit_reference(x, lut)
    sampled_indices = np.vstack([
        rng.integers(0, args.rows, 100),
        rng.integers(0, args.head_dim, 100),
    ])
    sampled_data = {
        "shape": (args.rows, args.head_dim),
        "indices": sampled_indices,
        "values": expected[sampled_indices[0], sampled_indices[1]],
    }
    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format=args.output_format,
        instance_name="gemma3_rope_halfsplit",
        runtime_loop_tiling_sizes=[4, 4],
    )
    returncode = runner.run_test(
        mlir_module,
        inputs=[x, lut.reshape(-1)],
        stochastic_expected_outputs=[sampled_data],
        rtol=5e-2,
        atol=5e-2,
    )
    _write_result_json(
        args,
        status="HARDWARE_SMOKE_PASS" if returncode == 0 else "HARDWARE_SMOKE_FAIL",
        returncode=returncode,
    )
    if returncode == 0:
        print("GEMMA3_ROPE_HALFSPLIT_HARDWARE_SMOKE: PASS")
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
