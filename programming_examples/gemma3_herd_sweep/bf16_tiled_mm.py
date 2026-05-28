# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gemma-style wrapper around the existing AIE2P tiled bf16 MM example."""

import argparse
from pathlib import Path
import sys

import numpy as np
from ml_dtypes import bfloat16

from common import SUPPORTED_HERD_SHAPES, parse_herd_shape

THIS_DIR = Path(__file__).resolve().parent
BF16_MM_DIR = THIS_DIR.parent / "matrix_multiplication" / "bf16"
sys.path.insert(0, str(BF16_MM_DIR))

from run import build_module  # noqa: E402
from air.backend.xrt import XRTBackend  # noqa: E402
from air.backend.xrt_runner import XRTRunner  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Gemma-style bf16 tiled MM")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--tile-m", type=int, default=32)
    parser.add_argument("--tile-k-l2", type=int, default=64)
    parser.add_argument("--tile-k-l1", type=int, default=32)
    parser.add_argument("--tile-n", type=int, default=32)
    parser.add_argument("--herd-shape", choices=SUPPORTED_HERD_SHAPES, default="2x4")
    parser.add_argument("--herd-m", type=int, default=None)
    parser.add_argument("--herd-n", type=int, default=None)
    parser.add_argument(
        "--compile-mode",
        choices=["compile-only", "compile-and-run"],
        default="compile-and-run",
    )
    parser.add_argument("--output-format", choices=["xclbin", "elf"], default="xclbin")
    args = parser.parse_args()

    shape_rows, shape_cols = parse_herd_shape(args.herd_shape)
    args.herd_m = args.herd_m if args.herd_m is not None else shape_rows
    args.herd_n = args.herd_n if args.herd_n is not None else shape_cols

    module = build_module(
        args.m,
        args.k,
        args.n,
        args.tile_m,
        args.tile_k_l2,
        args.tile_k_l1,
        args.tile_n,
        args.herd_m,
        args.herd_n,
        bfloat16,
        bfloat16,
        arch="aie2p",
        direct_codegen=False,
    )
    if args.print_module_only:
        print(module)
        return

    rng = np.random.default_rng(5)
    a = rng.uniform(-0.5, 0.5, (args.m, args.k)).astype(bfloat16)
    b = rng.uniform(-0.5, 0.5, (args.k, args.n)).astype(bfloat16)
    expected = (a.astype(np.float32) @ b.astype(np.float32)).astype(bfloat16)

    backend_opts = dict(
        verbose=args.verbose,
        output_format=args.output_format,
        instance_name="matmul_bf16",
        target_device="npu2",
        omit_while_true_loop=False,
        runtime_loop_tiling_sizes=[2, 2],
        stack_size=2048,
        lower_linalg_to_func="mm.o",
    )
    if args.compile_mode == "compile-and-run":
        runner = XRTRunner(**backend_opts)
        raise SystemExit(
            runner.run_test(
                module,
                inputs=[a, b],
                expected_outputs=[expected],
                rtol=1e-1,
                atol=8e-2,
                max_mismatch_percentage=5.0,
            )
        )

    backend = XRTBackend(**backend_opts)
    backend.compile(module)
    backend.unload()


if __name__ == "__main__":
    main()
