# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FlowKV decode attention: FlowQKV with Q chunk size 1."""

import argparse

import numpy as np
from ml_dtypes import bfloat16

from common import (
    ALL_OUTPUT_MODES,
    FLOW_VARIANTS,
    FLOW_KV_STAGING_MODES,
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
    build_flowkv_pipeline_module,
    run_or_compile,
)


def _paper_causal(args):
    return args.causal or args.variant == "causal"


def _paper_window_len(args):
    if args.variant != "swa":
        return args.window_len
    return args.window_len if args.window_len > 0 else args.kv_len


def main():
    parser = argparse.ArgumentParser(description="FlowKV decode attention")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--debug-ir", action="store_true")
    parser.add_argument("--kv-len", type=int, default=32)
    parser.add_argument("--kv-chunk", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--herd-shape", choices=SUPPORTED_HERD_SHAPES, default="2x4")
    parser.add_argument("--groups", type=int, default=None)
    parser.add_argument("--kv-groups", type=int, default=4)
    parser.add_argument("--heads-per-kv", type=int, default=2)
    parser.add_argument("--herd-rows", type=int, default=None)
    parser.add_argument("--herd-cols", type=int, default=None)
    parser.add_argument("--query-base", type=int, default=31)
    parser.add_argument("--window-len", type=int, default=0)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--variant", choices=FLOW_VARIANTS, default="causal")
    parser.add_argument("--schedule-mode", choices=SCHEDULE_MODES, default="smoke")
    parser.add_argument("--kv-staging", choices=FLOW_KV_STAGING_MODES, default="replicated")
    parser.add_argument("--kernel-name", default="flowkv_decode_bf16")
    parser.add_argument("--object-file", default="flowkv.o")
    parser.add_argument(
        "--compile-mode",
        choices=["compile-only", "compile-and-run"],
        default="compile-and-run",
    )
    parser.add_argument("--output-format", choices=["xclbin", "elf"], default="xclbin")
    parser.add_argument("--output-mode", default="auto")
    parser.add_argument(
        "--allow-unsupported-output-mode",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    shape_rows, shape_cols = parse_herd_shape(args.herd_shape)
    args.herd_rows = args.herd_rows if args.herd_rows is not None else shape_rows
    args.herd_cols = args.herd_cols if args.herd_cols is not None else shape_cols
    herd_tiles = args.herd_rows * args.herd_cols

    if args.kv_chunk <= 0 or args.kv_len % args.kv_chunk != 0:
        parser.error("kv-len must be divisible by a positive kv-chunk")

    if args.allow_unsupported_output_mode:
        if args.output_mode not in ALL_OUTPUT_MODES:
            parser.error(f"output mode must be one of: {', '.join(ALL_OUTPUT_MODES)}")
        if args.output_mode == "auto":
            parser.error("diagnostic output-mode bypass requires an explicit mode")
        if args.output_mode not in ("direct", "l2-gather"):
            parser.error(
                "diagnostic output-mode bypass supports only direct or l2-gather"
            )
        output_mode = args.output_mode
    else:
        try:
            output_mode = resolve_output_mode(
                args.output_mode, args.herd_rows, args.herd_cols, "flowkv"
            )
        except ValueError as exc:
            parser.error(str(exc))

    instance_name = "flow_attention"
    small_l2_gather = output_mode == "l2-gather" and args.herd_rows < 8
    if args.schedule_mode == "paper":
        tiles_per_query_chunk = args.kv_groups * args.heads_per_kv
        if herd_tiles % tiles_per_query_chunk != 0:
            parser.error("herd CT count must be divisible by kv-groups*heads-per-kv")
        if args.kv_staging == "pipeline":
            if args.herd_rows != 2:
                parser.error("pipeline FlowKV paper mode expects HERD_SHAPE=2x<kv-groups>")
            if args.herd_cols != args.kv_groups:
                parser.error("pipeline FlowKV paper mode expects herd-cols == kv-groups")
            if args.heads_per_kv != 2:
                parser.error("pipeline FlowKV paper mode uses heads-per-kv=2 as the two CT stages")
            module = build_flowkv_pipeline_module(
                args.kv_len,
                args.head_dim,
                "flowkv_scores_chunk_bf16_opt",
                "flowkv_apply_chunk_bf16_opt",
                args.object_file,
                "flowkv",
                args.kv_groups,
                args.herd_rows,
                args.herd_cols,
                output_mode,
                args.kv_chunk,
            )
            instance_name = "flowkv_pipeline"
        elif args.kv_staging == "shared":
            module = build_flow_paper_module(
                1,
                args.kv_len,
                args.head_dim,
                args.kernel_name,
                args.object_file,
                "flowkv",
                args.kv_groups,
                args.heads_per_kv,
                args.herd_rows,
                args.herd_cols,
                output_mode,
            )
            instance_name = "flow_attention_paper"
        else:
            module = build_flow_module(
                1,
                args.kv_len,
                args.head_dim,
                args.kernel_name,
                args.object_file,
                "flowkv",
                herd_tiles,
                args.herd_rows,
                args.herd_cols,
                output_mode,
                l2_gather_layout="rowcol" if small_l2_gather else "linear",
                l2_gather_via_channel=small_l2_gather,
            )
        if args.print_module_only:
            print(module)
            return

        rng = np.random.default_rng(4)
        val_range = 0.35
        k = rng.uniform(
            -val_range, val_range, (args.kv_groups, args.kv_len, args.head_dim)
        ).astype(bfloat16)
        v = rng.uniform(
            -val_range, val_range, (args.kv_groups, args.kv_len, args.head_dim)
        ).astype(bfloat16)

        if args.kv_staging == "pipeline":
            q = rng.uniform(
                -val_range, val_range, (args.kv_groups, 1, args.head_dim)
            ).astype(bfloat16)
            expected = np.stack(
                [
                    attention_reference(
                        q[g],
                        k[g],
                        v[g],
                        query_base=args.query_base,
                        causal=_paper_causal(args),
                        window_len=_paper_window_len(args),
                    )
                    for g in range(args.kv_groups)
                ]
            ).astype(bfloat16)
            inputs = [q, k, v]
        else:
            q = rng.uniform(
                -val_range,
                val_range,
                (args.herd_rows, args.herd_cols, 1, args.head_dim),
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
                expected = expected_tiles.reshape(herd_tiles, 1, args.head_dim)
                q_flat = q.reshape(herd_tiles, 1, args.head_dim)
                qkv = np.empty(
                    (herd_tiles, 1 + 2 * args.kv_len, args.head_dim),
                    dtype=bfloat16,
                )
                for tile in range(herd_tiles):
                    kv_group = (tile % tiles_per_query_chunk) // args.heads_per_kv
                    qkv[tile, :1, :] = q_flat[tile]
                    qkv[tile, 1 : 1 + args.kv_len, :] = k[kv_group]
                    qkv[tile, 1 + args.kv_len :, :] = v[kv_group]
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
            1,
            args.kv_len,
            args.head_dim,
            args.kernel_name,
            args.object_file,
            "flowkv",
            args.groups,
            args.herd_rows,
            args.herd_cols,
            output_mode,
            l2_gather_layout="rowcol" if small_l2_gather else "linear",
            l2_gather_via_channel=small_l2_gather,
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

        qkv = np.empty((args.groups, 1 + 2 * args.kv_len, args.head_dim), dtype=bfloat16)
        qkv[:, :1, :] = q
        qkv[:, 1 : 1 + args.kv_len, :] = k
        qkv[:, 1 + args.kv_len :, :] = v
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
        debug_ir=args.debug_ir,
    )


if __name__ == "__main__":
    main()
