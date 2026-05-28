# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FlowKV decode attention: FlowQKV with Q chunk size 1."""

import argparse

import numpy as np
from ml_dtypes import bfloat16

from common import attention_reference
from flow_common import build_flow_module, run_or_compile


def main():
    parser = argparse.ArgumentParser(description="FlowKV decode attention")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--kv-len", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--herd-rows", type=int, default=1)
    parser.add_argument("--herd-cols", type=int, default=1)
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

    herd_tiles = args.herd_rows * args.herd_cols
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

    run_or_compile(
        module,
        [q, k, v],
        expected,
        compile_mode=args.compile_mode,
        output_format=args.output_format,
        verbose=args.verbose,
        instance_name="flow_attention",
    )


if __name__ == "__main__":
    main()
