# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FlowQKV-style chunked prefill attention over one KV group."""

import argparse

import numpy as np
from ml_dtypes import bfloat16

from common import attention_reference
from flow_common import build_flow_module, run_or_compile


def main():
    parser = argparse.ArgumentParser(description="FlowQKV chunked attention")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--q-chunk", type=int, default=4)
    parser.add_argument("--kv-len", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--query-base", type=int, default=0)
    parser.add_argument("--window-len", type=int, default=0)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=["compile-only", "compile-and-run"],
        default="compile-and-run",
    )
    parser.add_argument("--output-format", choices=["xclbin", "elf"], default="xclbin")
    args = parser.parse_args()

    module = build_flow_module(
        args.q_chunk,
        args.kv_len,
        args.head_dim,
        "flowqkv_chunk_bf16",
        "flowqkv.o",
        "flowqkv",
    )
    if args.print_module_only:
        print(module)
        return

    rng = np.random.default_rng(3)
    val_range = 0.35
    q = rng.uniform(-val_range, val_range, (args.q_chunk, args.head_dim)).astype(
        bfloat16
    )
    k = rng.uniform(-val_range, val_range, (args.kv_len, args.head_dim)).astype(
        bfloat16
    )
    v = rng.uniform(-val_range, val_range, (args.kv_len, args.head_dim)).astype(
        bfloat16
    )
    expected = attention_reference(
        q,
        k,
        v,
        query_base=args.query_base,
        causal=args.causal,
        window_len=args.window_len,
    )

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
