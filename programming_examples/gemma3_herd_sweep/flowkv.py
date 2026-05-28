# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FlowKV decode attention: FlowQKV with Q chunk size 1."""

import argparse

import numpy as np
from ml_dtypes import bfloat16

from common import SUPPORTED_HERD_SHAPES, parse_herd_shape, attention_reference
from flow_common import build_flow_module, run_or_compile


def main():
    parser = argparse.ArgumentParser(description="FlowKV decode attention")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--kv-len", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--herd-shape", choices=SUPPORTED_HERD_SHAPES, default="2x4")
    parser.add_argument("--groups", type=int, default=None)
    parser.add_argument("--herd-rows", type=int, default=None)
    parser.add_argument("--herd-cols", type=int, default=None)
    parser.add_argument("--query-base", type=int, default=31)
    parser.add_argument("--window-len", type=int, default=0)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--kernel-name", default="flowkv_decode_bf16")
    parser.add_argument("--object-file", default="flowkv.o")
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
    args.groups = args.groups if args.groups is not None else herd_tiles
    if args.groups % herd_tiles != 0:
        parser.error("groups must be divisible by herd-rows*herd-cols")

    module = build_flow_module(
        1,
        args.kv_len,
        args.head_dim,
        args.kernel_name,
        args.object_file,
        "flowkv",
        args.groups,
        args.herd_rows,
        args.herd_cols,
        stage_output=herd_tiles > 16,
    )
    if args.print_module_only:
        print(module)
        return

    rng = np.random.default_rng(4)
    val_range = 0.35
    q = rng.uniform(-val_range, val_range, (args.groups, 1, args.head_dim)).astype(
        bfloat16
    )
    k = rng.uniform(
        -val_range, val_range, (args.groups, args.kv_len, args.head_dim)
    ).astype(bfloat16)
    v = rng.uniform(
        -val_range, val_range, (args.groups, args.kv_len, args.head_dim)
    ).astype(bfloat16)
    expected = np.stack(
        [
            attention_reference(
                q[g],
                k[g],
                v[g],
                query_base=args.query_base,
                causal=args.causal,
                window_len=args.window_len,
            )
            for g in range(args.groups)
        ]
    ).astype(bfloat16)

    qkv = np.empty(
        (args.groups, 1 + 2 * args.kv_len, args.head_dim), dtype=bfloat16
    )
    qkv[:, : 1, :] = q
    qkv[:, 1 : 1 + args.kv_len, :] = k
    qkv[:, 1 + args.kv_len :, :] = v

    run_or_compile(
        module,
        [qkv.reshape(args.groups // args.herd_cols, args.herd_cols, qkv.shape[-2], qkv.shape[-1])],
        expected,
        compile_mode=args.compile_mode,
        output_format=args.output_format,
        verbose=args.verbose,
        instance_name="flow_attention",
    )


if __name__ == "__main__":
    main()
