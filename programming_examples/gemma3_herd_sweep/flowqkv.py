# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FlowQKV-style chunked prefill attention over KV groups."""

import argparse

import numpy as np
from ml_dtypes import bfloat16

from common import (
    FLOW_VARIANTS,
    FLOW_KV_STAGING_MODES,
    OUTPUT_MODES,
    SCHEDULE_MODES,
    SUPPORTED_HERD_SHAPES,
    parse_herd_shape,
    attention_reference,
    resolve_output_mode,
    tiled_attention_reference,
)
from flow_common import (
    build_flow_module,
    build_flow_paper_module,
    build_flowqkv_pipeline_module,
    run_or_compile,
)


def _paper_causal(args):
    return args.causal or args.variant == "causal"


def _paper_window_len(args):
    if args.variant != "swa":
        return args.window_len
    return args.window_len if args.window_len > 0 else args.kv_len


def _qbase_kernel_name(kernel_name: str) -> str:
    if kernel_name == "flowqkv_chunk_bf16_opt":
        return "flowqkv_chunk_qbase_bf16_opt"
    if kernel_name.endswith("_opt"):
        return f"{kernel_name[:-4]}_qbase_opt"
    return f"{kernel_name}_qbase"


def main():
    parser = argparse.ArgumentParser(description="FlowQKV chunked attention")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--q-chunk", type=int, default=4)
    parser.add_argument("--kv-len", type=int, default=32)
    parser.add_argument("--kv-chunk", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--herd-shape", choices=SUPPORTED_HERD_SHAPES, default="2x4")
    parser.add_argument("--groups", type=int, default=None)
    parser.add_argument("--kv-groups", type=int, default=4)
    parser.add_argument("--heads-per-kv", type=int, default=2)
    parser.add_argument("--herd-rows", type=int, default=None)
    parser.add_argument("--herd-cols", type=int, default=None)
    parser.add_argument("--query-base", type=int, default=0)
    parser.add_argument("--window-len", type=int, default=0)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--variant", choices=FLOW_VARIANTS, default="causal")
    parser.add_argument("--schedule-mode", choices=SCHEDULE_MODES, default="smoke")
    parser.add_argument("--kv-staging", choices=FLOW_KV_STAGING_MODES, default="replicated")
    parser.add_argument("--kernel-name", default="flowqkv_chunk_bf16")
    parser.add_argument("--object-file", default="flowqkv.o")
    parser.add_argument(
        "--compile-mode",
        choices=["compile-only", "compile-and-run"],
        default="compile-and-run",
    )
    parser.add_argument("--output-format", choices=["xclbin", "elf"], default="xclbin")
    parser.add_argument("--output-mode", choices=OUTPUT_MODES, default="auto")
    args = parser.parse_args()

    shape_rows, shape_cols = parse_herd_shape(args.herd_shape)
    args.herd_rows = args.herd_rows if args.herd_rows is not None else shape_rows
    args.herd_cols = args.herd_cols if args.herd_cols is not None else shape_cols
    herd_tiles = args.herd_rows * args.herd_cols

    if args.kv_chunk <= 0 or args.kv_len % args.kv_chunk != 0:
        parser.error("kv-len must be divisible by a positive kv-chunk")

    try:
        output_mode = resolve_output_mode(
            args.output_mode, args.herd_rows, args.herd_cols, "flowqkv"
        )
    except ValueError as exc:
        parser.error(str(exc))

    instance_name = "flow_attention"
    if args.schedule_mode == "paper":
        tiles_per_query_chunk = args.kv_groups * args.heads_per_kv
        if herd_tiles % tiles_per_query_chunk != 0:
            parser.error("herd CT count must be divisible by kv-groups*heads-per-kv")
        qbase_kernel_name = _qbase_kernel_name(args.kernel_name)
        if args.kv_staging == "pipeline":
            if args.herd_rows % 2 != 0:
                parser.error("pipeline FlowQKV paper mode expects an even CT row count")
            if args.herd_cols != args.kv_groups:
                parser.error("pipeline FlowQKV paper mode expects herd-cols == kv-groups")
            module = build_flowqkv_pipeline_module(
                args.q_chunk,
                args.kv_len,
                args.head_dim,
                "flowqkv_scores_bf16_opt",
                "flowqkv_apply_bf16_opt",
                args.object_file,
                "flowqkv",
                args.kv_groups,
                args.herd_rows,
                args.herd_cols,
                output_mode,
                args.query_base,
            )
            instance_name = "flowqkv_pipeline"
        elif args.kv_staging == "shared":
            module = build_flow_paper_module(
                args.q_chunk,
                args.kv_len,
                args.head_dim,
                qbase_kernel_name,
                args.object_file,
                "flowqkv",
                args.kv_groups,
                args.heads_per_kv,
                args.herd_rows,
                args.herd_cols,
                output_mode,
                dynamic_query_base=True,
                query_base=args.query_base,
            )
            instance_name = "flow_attention_paper"
        else:
            module = build_flow_module(
                args.q_chunk,
                args.kv_len,
                args.head_dim,
                qbase_kernel_name,
                args.object_file,
                "flowqkv",
                herd_tiles,
                args.herd_rows,
                args.herd_cols,
                output_mode,
                l2_gather_layout="rowcol",
                dynamic_query_base=True,
                query_base=args.query_base,
                tiles_per_query_chunk=tiles_per_query_chunk,
            )
        if args.print_module_only:
            print(module)
            return

        rng = np.random.default_rng(3)
        val_range = 0.35
        k = rng.uniform(
            -val_range,
            val_range,
            (args.kv_groups, args.kv_len, args.head_dim),
        ).astype(bfloat16)
        v = rng.uniform(
            -val_range,
            val_range,
            (args.kv_groups, args.kv_len, args.head_dim),
        ).astype(bfloat16)
        if args.kv_staging == "pipeline":
            pipe_rows = args.herd_rows // 2
            q = rng.uniform(
                -val_range,
                val_range,
                (pipe_rows, args.herd_cols, args.q_chunk, args.head_dim),
            ).astype(bfloat16)
            expected = np.empty_like(q)
            for row in range(pipe_rows):
                tile_query_base = args.query_base + row * args.q_chunk
                for group in range(args.kv_groups):
                    expected[row, group] = attention_reference(
                        q[row, group],
                        k[group],
                        v[group],
                        query_base=tile_query_base,
                        causal=_paper_causal(args),
                        window_len=_paper_window_len(args),
                    )
            inputs = [q, k, v]
        else:
            q = rng.uniform(
                -val_range,
                val_range,
                (args.herd_rows, args.herd_cols, args.q_chunk, args.head_dim),
            ).astype(bfloat16)
            expected_tiles = tiled_attention_reference(
                q,
                k,
                v,
                kv_groups=args.kv_groups,
                heads_per_kv=args.heads_per_kv,
                query_base=args.query_base,
                causal=_paper_causal(args),
                window_len=_paper_window_len(args),
            ).astype(bfloat16)
            if args.kv_staging == "shared":
                expected = expected_tiles
                inputs = [q, k, v]
            else:
                expected = expected_tiles.reshape(herd_tiles, args.q_chunk, args.head_dim)
                q_flat = q.reshape(herd_tiles, args.q_chunk, args.head_dim)
                qkv = np.empty(
                    (herd_tiles, args.q_chunk + 2 * args.kv_len, args.head_dim),
                    dtype=bfloat16,
                )
                for tile in range(herd_tiles):
                    kv_group = (tile % tiles_per_query_chunk) // args.heads_per_kv
                    qkv[tile, : args.q_chunk, :] = q_flat[tile]
                    qkv[tile, args.q_chunk : args.q_chunk + args.kv_len, :] = k[
                        kv_group
                    ]
                    qkv[tile, args.q_chunk + args.kv_len :, :] = v[kv_group]
                inputs = [
                    qkv.reshape(
                        args.herd_rows,
                        args.herd_cols,
                        qkv.shape[-2],
                        qkv.shape[-1],
                    )
                ]
    else:
        args.groups = args.groups if args.groups is not None else herd_tiles
        if args.groups % herd_tiles != 0:
            parser.error("groups must be divisible by herd-rows*herd-cols")

        module = build_flow_module(
            args.q_chunk,
            args.kv_len,
            args.head_dim,
            args.kernel_name,
            args.object_file,
            "flowqkv",
            args.groups,
            args.herd_rows,
            args.herd_cols,
            output_mode,
            l2_gather_layout="rowcol",
        )
        if args.print_module_only:
            print(module)
            return

        rng = np.random.default_rng(3)
        val_range = 0.35
        q = rng.uniform(
            -val_range, val_range, (args.groups, args.q_chunk, args.head_dim)
        ).astype(bfloat16)
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
            (args.groups, args.q_chunk + 2 * args.kv_len, args.head_dim),
            dtype=bfloat16,
        )
        qkv[:, : args.q_chunk, :] = q
        qkv[:, args.q_chunk : args.q_chunk + args.kv_len, :] = k
        qkv[:, args.q_chunk + args.kv_len :, :] = v
        inputs = [
            qkv.reshape(
                args.groups // args.herd_cols,
                args.herd_cols,
                qkv.shape[-2],
                qkv.shape[-1],
            )
        ]

    run_or_compile(
        module,
        inputs,
        expected,
        compile_mode=args.compile_mode,
        output_format=args.output_format,
        verbose=args.verbose,
        instance_name=instance_name,
    )


if __name__ == "__main__":
    main()
