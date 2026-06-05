# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gemma3 GeGLU elementwise activation kernel.

Computes GELU(gate) * up with the tanh approximation used by the synthetic
Gemma3 CPU reference. This is a standalone nonlinear promotion candidate; the
Gemma model loop keeps a host fallback until compile and hardware evidence are
recorded for the target shape.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects import arith, math as math_dialect
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp, subview
from air.dialects.vector import BroadcastOp, transfer_read, transfer_write
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend

range_ = for_

SQRT_2_OVER_PI = 0.7978845608
GELU_BETA = 0.044715


def geglu_reference(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    gate_f = np.asarray(gate, dtype=np.float32)
    up_f = np.asarray(up, dtype=np.float32)
    inner = SQRT_2_OVER_PI * (gate_f + GELU_BETA * gate_f * gate_f * gate_f)
    gelu = 0.5 * gate_f * (1.0 + np.tanh(inner))
    return (gelu * up_f).astype(bfloat16)


@module_builder
def build_module(n, tile_n, np_dtype_in, vector_size=16):
    xrt_dtype_in = type_mapper(np_dtype_in)
    num_tiles = 2
    assert n % (tile_n * num_tiles) == 0
    assert tile_n % vector_size == 0
    index_type = IndexType.get()

    l3_ty = MemRefType.get([n], xrt_dtype_in)
    l1_ty = MemRefType.get(
        shape=[tile_n],
        element_type=xrt_dtype_in,
        memory_space=IntegerAttr.get(T.i32(), MemorySpace.L1),
    )

    vec_ty = VectorType.get([vector_size], xrt_dtype_in)
    identity_map = AffineMapAttr.get(AffineMap.get_identity(1))

    @FuncOp.from_py_func(l3_ty, l3_ty, l3_ty)
    def gemma3_geglu(arg_gate, arg_up, arg_out):

        @herd(name="herd_0", sizes=[1, num_tiles], operands=[arg_gate, arg_up, arg_out])
        def herd_body(_tx, _ty, _sx, _sy, l3_gate, l3_up, l3_out):
            for offset_base in range_(0, n, tile_n * num_tiles):
                l1_gate = AllocOp(l1_ty, [], [])
                l1_up = AllocOp(l1_ty, [], [])
                l1_out = AllocOp(l1_ty, [], [])

                offset_map = AffineMap.get(
                    0,
                    2,
                    [
                        AffineExpr.get_add(
                            AffineSymbolExpr.get(0),
                            AffineExpr.get_mul(
                                AffineSymbolExpr.get(1),
                                AffineConstantExpr.get(tile_n),
                            ),
                        )
                    ],
                )
                offset = affine_apply(offset_map, [offset_base, _ty])

                dma_memcpy_nd(
                    l1_gate,
                    l3_gate,
                    src_offsets=[offset],
                    src_sizes=[tile_n],
                    src_strides=[1],
                )
                dma_memcpy_nd(
                    l1_up,
                    l3_up,
                    src_offsets=[offset],
                    src_sizes=[tile_n],
                    src_strides=[1],
                )

                c0 = ConstantOp(index_type, 0)
                c_vec = ConstantOp(index_type, vector_size)
                c_tile = ConstantOp(index_type, tile_n)
                cst0 = arith.ConstantOp(xrt_dtype_in, 0.0)
                half = arith.ConstantOp(xrt_dtype_in, 0.5)
                one = arith.ConstantOp(xrt_dtype_in, 1.0)
                beta = arith.ConstantOp(xrt_dtype_in, GELU_BETA)
                scale = arith.ConstantOp(xrt_dtype_in, SQRT_2_OVER_PI)
                v_half = BroadcastOp(vec_ty, half)
                v_one = BroadcastOp(vec_ty, one)
                v_beta = BroadcastOp(vec_ty, beta)
                v_scale = BroadcastOp(vec_ty, scale)

                for j in range_(c0, c_tile, c_vec):
                    sub_gate = subview(l1_gate.result, [j], [vector_size], [1])
                    sub_up = subview(l1_up.result, [j], [vector_size], [1])
                    sub_out = subview(l1_out.result, [j], [vector_size], [1])

                    v_gate = transfer_read(
                        vec_ty, sub_gate, [c0], identity_map, cst0, [True]
                    )
                    v_up = transfer_read(
                        vec_ty, sub_up, [c0], identity_map, cst0, [True]
                    )
                    v_gate2 = arith.mulf(v_gate, v_gate)
                    v_gate3 = arith.mulf(v_gate, v_gate2)
                    v_beta_gate3 = arith.mulf(v_gate3, v_beta.result)
                    v_inner = arith.addf(v_gate, v_beta_gate3)
                    v_scaled = arith.mulf(v_inner, v_scale.result)
                    v_tanh = math_dialect.tanh(v_scaled)
                    v_one_plus = arith.addf(v_tanh, v_one.result)
                    v_half_gate = arith.mulf(v_gate, v_half.result)
                    v_gelu = arith.mulf(v_half_gate, v_one_plus)
                    v_geglu = arith.mulf(v_gelu, v_up)
                    transfer_write(None, v_geglu, sub_out, [c0], identity_map, [True])
                    yield_([])

                dma_memcpy_nd(
                    l3_out,
                    l1_out,
                    dst_offsets=[offset],
                    dst_sizes=[tile_n],
                    dst_strides=[1],
                )
                DeallocOp(l1_gate)
                DeallocOp(l1_up)
                DeallocOp(l1_out)
                yield_([])


def _self_test() -> None:
    gate = np.linspace(-3.0, 3.0, 64, dtype=np.float32).astype(bfloat16)
    up = np.linspace(0.25, 1.25, 64, dtype=np.float32).astype(bfloat16)
    out = geglu_reference(gate, up)
    assert out.shape == (64,)
    assert np.isfinite(out.astype(np.float32)).all()
    print(f"geglu_reference_checksum={float(np.sum(out.astype(np.float32))):.6f}")
    print("GEMMA3_GEGLU_REFERENCE_SELF_TEST: PASS")


def _write_result_json(args, *, status: str, returncode: int) -> None:
    if args.result_json is None:
        return
    result = {
        "schema_version": 1,
        "kernel": "gemma3_geglu",
        "status": status,
        "returncode": int(returncode),
        "n": int(args.n),
        "tile_n": int(args.tile_n),
        "vector_size": int(args.vector_size),
        "compile_mode": args.compile_mode,
        "output_format": args.output_format,
        "tolerance": {"rtol": 0.1, "atol": 0.05},
    }
    args.result_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma3 GeGLU nonlinear kernel")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--tile-n", type=int, default=512)
    parser.add_argument("--vector-size", type=int, default=16)
    parser.add_argument("--print-module-only", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=["compile-only", "compile-and-run"],
        default="compile-only",
    )
    parser.add_argument(
        "--output-format",
        choices=["xclbin", "elf"],
        default="elf",
    )
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return 0

    mlir_module = build_module(args.n, args.tile_n, bfloat16, args.vector_size)
    if args.print_module_only:
        print(mlir_module)
        return 0

    rng = np.random.default_rng(7)
    gate = rng.uniform(-4.0, 4.0, args.n).astype(bfloat16)
    up = rng.uniform(-2.0, 2.0, args.n).astype(bfloat16)

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
        print("GEMMA3_GEGLU_COMPILE_ONLY: PASS")
        return 0

    sampled_indices = np.vstack([rng.integers(0, args.n, 100)])
    expected = geglu_reference(gate, up)
    sampled_data = {
        "shape": (args.n,),
        "indices": sampled_indices,
        "values": expected[sampled_indices[0]],
    }
    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format=args.output_format,
        instance_name="gemma3_geglu",
        runtime_loop_tiling_sizes=[4, 4],
    )
    returncode = runner.run_test(
        mlir_module,
        inputs=[gate, up],
        stochastic_expected_outputs=[sampled_data],
        rtol=1e-1,
        atol=5e-2,
    )
    _write_result_json(
        args,
        status="HARDWARE_SMOKE_PASS" if returncode == 0 else "HARDWARE_SMOKE_FAIL",
        returncode=returncode,
    )
    if returncode == 0:
        print("GEMMA3_GEGLU_HARDWARE_SMOKE: PASS")
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
