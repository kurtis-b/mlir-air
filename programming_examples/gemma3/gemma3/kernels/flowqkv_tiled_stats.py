# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tiled FlowQKV attention-stat diagnostic for Gemma3 decode attention.

The production FlowQKV wrapper stages the full KV range in L1, so Gemma3's
1k decode cache over-allocates tile memory at `HEAD_DIM=256`. This diagnostic
keeps only one KV tile in L1 and emits enough softmax state for an exact
tile-stat merge outside the kernel.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import filelock

import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects.func import CallOp, FuncOp
from air.dialects.memref import AllocOp, DeallocOp
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import type_mapper

from gemma3.core.common import attention_reference
from gemma3.kernels.flow_common import _affine_linear_tile, _affine_mul


def _fast_exp_approx(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(x, dtype=np.float32), -20.0, 20.0)
    bits = (12102203.0 * clipped + 1064866805.0).astype(np.uint32)
    return bits.view(np.float32)


def tiled_stats_reference(
    q: np.ndarray,
    k_tiles: np.ndarray,
    v_tiles: np.ndarray,
) -> np.ndarray:
    q_f = np.asarray(q, dtype=np.float32)
    k_f = np.asarray(k_tiles, dtype=np.float32)
    v_f = np.asarray(v_tiles, dtype=np.float32)
    tile_count, q_chunk, head_dim = q_f.shape
    stats = np.zeros((tile_count, q_chunk, head_dim + 2), dtype=np.float32)
    inv_sqrt_d = np.float32(1.0 / np.sqrt(np.float32(head_dim)))

    for tile in range(tile_count):
        for qi in range(q_chunk):
            scores = (k_f[tile] @ q_f[tile, qi]) * inv_sqrt_d
            max_score = np.max(scores).astype(np.float32)
            weights = _fast_exp_approx(scores - max_score)
            stats[tile, qi, 0] = max_score
            stats[tile, qi, 1] = np.sum(weights, dtype=np.float32)
            stats[tile, qi, 2:] = weights.astype(np.float32) @ v_f[tile]
    return stats


def merge_tiled_stats(stats: np.ndarray) -> np.ndarray:
    stats_f = np.asarray(stats, dtype=np.float32)
    max_scores = stats_f[:, :, 0]
    denoms = stats_f[:, :, 1]
    numerators = stats_f[:, :, 2:]
    global_max = np.max(max_scores, axis=0)
    scale = np.exp(max_scores - global_max[None, :]).astype(np.float32)
    global_denoms = np.sum(denoms * scale, axis=0)
    global_nums = np.sum(numerators * scale[:, :, None], axis=0)
    return (global_nums / global_denoms[:, None]).astype(bfloat16)


@module_builder
def build_tiled_stats_module(
    q_chunk,
    kv_tile,
    head_dim,
    kernel_func,
    object_file,
    tile_count,
    herd_rows=1,
    herd_cols=1,
    output_mode="direct",
):
    if output_mode not in ("direct", "l2-gather"):
        raise ValueError(f"unsupported output mode: {output_mode}")
    use_l2_gather = output_mode == "l2-gather"

    bf16_type = type_mapper(bfloat16)
    f32_type = type_mapper(np.float32)
    herd_tiles = herd_rows * herd_cols
    stats_width = head_dim + 2

    q_l3_ty = MemRefType.get([tile_count, q_chunk, head_dim], bf16_type)
    kv_l3_ty = MemRefType.get([tile_count, kv_tile, head_dim], bf16_type)
    stats_l3_ty = MemRefType.get([tile_count, q_chunk, stats_width], f32_type)

    l2_space = IntegerAttr.get(T.i32(), MemorySpace.L2)
    if use_l2_gather:
        stats_l2_ty = MemRefType.get(
            [herd_rows, herd_cols, q_chunk, stats_width],
            f32_type,
            memory_space=l2_space,
        )

    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    q_l1_ty = MemRefType.get([q_chunk, head_dim], bf16_type, memory_space=l1_space)
    kv_l1_ty = MemRefType.get([kv_tile, head_dim], bf16_type, memory_space=l1_space)
    stats_l1_ty = MemRefType.get(
        [q_chunk, stats_width], f32_type, memory_space=l1_space
    )

    stats_func = FuncOp(
        kernel_func,
        ([q_l1_ty, kv_l1_ty, kv_l1_ty, stats_l1_ty], []),
        visibility="private",
    )
    stats_func.attributes["link_with"] = StringAttr.get(object_file)
    stats_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    launch_offset_map = _affine_mul(herd_tiles)
    chunk_map = _affine_linear_tile(herd_cols)

    @FuncOp.from_py_func(q_l3_ty, kv_l3_ty, kv_l3_ty, stats_l3_ty)
    def flowqkv_tiled_stats(arg_q, arg_k, arg_v, arg_stats):
        @launch(
            operands=[arg_q, arg_k, arg_v, arg_stats],
            sizes=[tile_count // herd_tiles, 1],
        )
        def launch_body(lx, _ly, _lsx, _lsy, lq, lk, lv, lstats):
            launch_base = affine_apply(launch_offset_map, [lx])

            @segment(
                name="flowqkv_tiled_stats_seg",
                operands=[launch_base, lq, lk, lv, lstats],
            )
            def segment_body(base, sq, sk, sv, sstats):
                if use_l2_gather:
                    l2_stats = AllocOp(stats_l2_ty, [], [])

                @herd(
                    name="flowqkv_tiled_stats_herd",
                    sizes=[herd_rows, herd_cols],
                    operands=[base, sq, sk, sv, l2_stats if use_l2_gather else sstats],
                    link_with=object_file,
                )
                def herd_body(_tx, _ty, _sx, _sy, bbase, hq, hk, hv, hstats):
                    tile_idx = affine_apply(chunk_map, [bbase, _tx, _ty])

                    l1_q = AllocOp(q_l1_ty, [], [])
                    l1_k = AllocOp(kv_l1_ty, [], [])
                    l1_v = AllocOp(kv_l1_ty, [], [])
                    l1_stats = AllocOp(stats_l1_ty, [], [])

                    dma_memcpy_nd(
                        l1_q,
                        hq,
                        src_offsets=[tile_idx, 0, 0],
                        src_sizes=[1, q_chunk, head_dim],
                        src_strides=[q_chunk * head_dim, head_dim, 1],
                    )
                    dma_memcpy_nd(
                        l1_k,
                        hk,
                        src_offsets=[tile_idx, 0, 0],
                        src_sizes=[1, kv_tile, head_dim],
                        src_strides=[kv_tile * head_dim, head_dim, 1],
                    )
                    dma_memcpy_nd(
                        l1_v,
                        hv,
                        src_offsets=[tile_idx, 0, 0],
                        src_sizes=[1, kv_tile, head_dim],
                        src_strides=[kv_tile * head_dim, head_dim, 1],
                    )
                    CallOp(stats_func, [l1_q, l1_k, l1_v, l1_stats])
                    if use_l2_gather:
                        dma_memcpy_nd(
                            hstats,
                            l1_stats,
                            dst_offsets=[_tx, _ty, 0, 0],
                            dst_sizes=[1, 1, q_chunk, stats_width],
                            dst_strides=[
                                herd_cols * q_chunk * stats_width,
                                q_chunk * stats_width,
                                stats_width,
                                1,
                            ],
                        )
                    else:
                        dma_memcpy_nd(
                            hstats,
                            l1_stats,
                            dst_offsets=[tile_idx, 0, 0],
                            dst_sizes=[1, q_chunk, stats_width],
                            dst_strides=[q_chunk * stats_width, stats_width, 1],
                        )

                    DeallocOp(l1_q)
                    DeallocOp(l1_k)
                    DeallocOp(l1_v)
                    DeallocOp(l1_stats)

                if use_l2_gather:
                    dma_memcpy_nd(
                        sstats,
                        l2_stats,
                        dst_offsets=[base, 0, 0],
                        dst_sizes=[herd_tiles, q_chunk, stats_width],
                        dst_strides=[q_chunk * stats_width, stats_width, 1],
                        src_offsets=[0, 0, 0, 0],
                        src_sizes=[herd_rows, herd_cols, q_chunk, stats_width],
                        src_strides=[
                            herd_cols * q_chunk * stats_width,
                            q_chunk * stats_width,
                            stats_width,
                            1,
                        ],
                    )
                    DeallocOp(l2_stats)


def _inputs(args):
    if args.kv_len % args.kv_tile != 0:
        raise ValueError("kv-len must be divisible by kv-tile")
    tile_count = args.kv_len // args.kv_tile
    rng = np.random.default_rng(29)
    val_range = 0.35
    q_single = rng.uniform(
        -val_range, val_range, (args.q_chunk, args.head_dim)
    ).astype(bfloat16)
    k_full = rng.uniform(
        -val_range, val_range, (args.kv_len, args.head_dim)
    ).astype(bfloat16)
    v_full = rng.uniform(
        -val_range, val_range, (args.kv_len, args.head_dim)
    ).astype(bfloat16)
    q_tiles = np.broadcast_to(
        q_single, (tile_count, args.q_chunk, args.head_dim)
    ).copy()
    k_tiles = k_full.reshape(tile_count, args.kv_tile, args.head_dim).copy()
    v_tiles = v_full.reshape(tile_count, args.kv_tile, args.head_dim).copy()
    return q_single, k_full, v_full, q_tiles, k_tiles, v_tiles


def _combined_metrics(q_single, k_full, v_full, stats):
    merged = merge_tiled_stats(stats)
    expected = attention_reference(q_single, k_full, v_full).astype(bfloat16)
    merged_f = merged.astype(np.float32).reshape(-1)
    expected_f = expected.astype(np.float32).reshape(-1)
    if np.std(merged_f) == 0.0 or np.std(expected_f) == 0.0:
        corr = 1.0 if np.allclose(merged_f, expected_f) else 0.0
    else:
        corr = float(np.corrcoef(merged_f, expected_f)[0, 1])
    return {
        "combined_attention_correlation_vs_exact": corr,
        "combined_attention_max_abs_error_vs_exact": float(
            np.max(np.abs(merged_f - expected_f))
        ),
        "combined_attention_mean_abs_error_vs_exact": float(
            np.mean(np.abs(merged_f - expected_f))
        ),
    }


def _write_result_json(args, *, status: str, returncode: int, metrics: dict | None):
    if args.result_json is None:
        return
    tile_count = args.kv_len // args.kv_tile
    result = {
        "schema_version": 1,
        "kernel": "flowqkv_tiled_stats",
        "status": status,
        "returncode": int(returncode),
        "q_chunk": int(args.q_chunk),
        "kv_len": int(args.kv_len),
        "kv_tile": int(args.kv_tile),
        "tile_count": int(tile_count),
        "host_batch_tiles": int(args.host_batch_tiles),
        "host_batch_count": (
            int(tile_count // args.host_batch_tiles)
            if args.host_batch_tiles and args.host_batch_tiles > 0
            else 1
        ),
        "head_dim": int(args.head_dim),
        "herd_rows": int(args.herd_rows),
        "herd_cols": int(args.herd_cols),
        "output_mode": args.output_mode,
        "compile_mode": args.compile_mode,
        "output_format": args.output_format,
        "tolerance": {"rtol": 0.1, "atol": 0.1, "max_mismatch_percentage": 1.0},
        "paper_gap_status": "diagnostic-tiled-1k-kv-attention-stat-path",
    }
    if metrics:
        result.update(metrics)
    args.result_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def _self_test() -> None:
    class Args:
        q_chunk = 4
        kv_len = 64
        kv_tile = 16
        head_dim = 32

    q_single, k_full, v_full, q_tiles, k_tiles, v_tiles = _inputs(Args())
    stats = tiled_stats_reference(q_tiles, k_tiles, v_tiles)
    metrics = _combined_metrics(q_single, k_full, v_full, stats)
    if metrics["combined_attention_correlation_vs_exact"] < 0.99:
        raise AssertionError(metrics)
    print(
        "flowqkv_tiled_stats_reference "
        f"corr={metrics['combined_attention_correlation_vs_exact']:.6f} "
        f"max_abs={metrics['combined_attention_max_abs_error_vs_exact']:.6f}"
    )
    print("GEMMA3_FLOWQKV_TILED_STATS_REFERENCE_SELF_TEST: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="FlowQKV tiled attention-stat diagnostic"
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--q-chunk", type=int, default=4)
    parser.add_argument("--kv-len", type=int, default=1024)
    parser.add_argument("--kv-tile", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--herd-rows", type=int, default=1)
    parser.add_argument("--herd-cols", type=int, default=1)
    parser.add_argument(
        "--host-batch-tiles",
        type=int,
        default=0,
        help="compile this many KV tiles per NPU invocation and loop on the host",
    )
    parser.add_argument(
        "--output-mode", choices=["direct", "l2-gather"], default="direct"
    )
    parser.add_argument("--kernel-name", default="flowqkv_tile_stats_bf16")
    parser.add_argument("--object-file", default="flowqkv_tiled_stats.o")
    parser.add_argument(
        "--compile-mode",
        choices=["compile-only", "compile-and-run"],
        default="compile-only",
    )
    parser.add_argument("--output-format", choices=["xclbin", "elf"], default="elf")
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--print-module-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return 0
    if args.kv_len % args.kv_tile != 0:
        parser.error("kv-len must be divisible by kv-tile")
    tile_count = args.kv_len // args.kv_tile
    module_tile_count = (
        args.host_batch_tiles if args.host_batch_tiles > 0 else tile_count
    )
    if module_tile_count <= 0:
        parser.error("host-batch-tiles must be positive when provided")
    if tile_count % module_tile_count != 0:
        parser.error("(kv-len / kv-tile) must be divisible by host-batch-tiles")
    herd_tiles = args.herd_rows * args.herd_cols
    if module_tile_count % herd_tiles != 0:
        parser.error("module tile count must be divisible by herd-rows*herd-cols")

    module = build_tiled_stats_module(
        args.q_chunk,
        args.kv_tile,
        args.head_dim,
        args.kernel_name,
        args.object_file,
        module_tile_count,
        args.herd_rows,
        args.herd_cols,
        args.output_mode,
    )
    if args.print_module_only:
        print(module)
        return 0

    q_single, k_full, v_full, q_tiles, k_tiles, v_tiles = _inputs(args)
    expected_stats = tiled_stats_reference(q_tiles, k_tiles, v_tiles)
    metrics = _combined_metrics(q_single, k_full, v_full, expected_stats)

    backend_opts = dict(
        verbose=args.verbose,
        omit_pingpong=True,
        output_format=args.output_format,
        instance_name="flowqkv_tiled_stats",
        target_device="npu2",
        runtime_loop_tiling_sizes=[1, 1],
    )
    if args.compile_mode == "compile-only":
        backend = XRTBackend(**backend_opts)
        backend.compile(module)
        backend.unload()
        _write_result_json(
            args, status="COMPILE_ONLY_PASS", returncode=0, metrics=metrics
        )
        print("GEMMA3_FLOWQKV_TILED_STATS_COMPILE_ONLY: PASS")
        return 0

    backend = XRTBackend(**backend_opts)
    compiled_module = backend.compile(module)
    actual_stats = np.zeros_like(expected_stats)
    try:
        with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
            module_function = backend.load(compiled_module)
            for batch_start in range(0, tile_count, module_tile_count):
                batch_end = batch_start + module_tile_count
                batch_expected = expected_stats[batch_start:batch_end]
                batch_output = np.zeros(batch_expected.shape, dtype=np.float32)
                expanded = [
                    q_tiles[batch_start:batch_end],
                    k_tiles[batch_start:batch_end],
                    v_tiles[batch_start:batch_end],
                    batch_output,
                ]
                outputs = module_function(*expanded)
                actual_stats[batch_start:batch_end] = np.asarray(outputs[3]).reshape(
                    batch_expected.shape
                )
    finally:
        backend.unload()

    close_mask = np.isclose(actual_stats, expected_stats, rtol=1e-1, atol=1e-1)
    mismatches = int(np.size(close_mask) - np.count_nonzero(close_mask))
    mismatch_pct = mismatches / close_mask.size * 100.0 if close_mask.size else 0.0
    corr = float(
        np.corrcoef(actual_stats.reshape(-1), expected_stats.reshape(-1))[0, 1]
    )
    actual_metrics = _combined_metrics(q_single, k_full, v_full, actual_stats)
    actual_metrics.update(
        {
            "stats_output_correlation": corr,
            "stats_output_mismatch_percentage": mismatch_pct,
            "stats_output_max_abs_error": float(
                np.max(
                    np.abs(
                        actual_stats.astype(np.float32)
                        - expected_stats.astype(np.float32)
                    )
                )
            ),
        }
    )
    returncode = 0 if mismatch_pct <= 1.0 and corr >= 0.99 else -1
    status = "HARDWARE_SMOKE_PASS" if returncode == 0 else "HARDWARE_SMOKE_FAIL"
    _write_result_json(
        args, status=status, returncode=returncode, metrics=actual_metrics
    )
    if returncode == 0:
        print(f"Output 0 correlation: {corr:.6f} (threshold: 0.99)")
        print("PASS!")
        print("GEMMA3_FLOWQKV_TILED_STATS_HARDWARE_SMOKE: PASS")
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
