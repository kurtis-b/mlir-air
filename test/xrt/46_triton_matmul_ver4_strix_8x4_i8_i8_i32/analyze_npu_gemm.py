#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.

"""Write a static report for the Strix/XDNA2 int8 GEMM candidate."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

DEFAULT_TILE_M = 512
DEFAULT_TILE_N = 256
DEFAULT_ACCEPTANCE_TOPS = 36.15
PUBLISHED_SOTA_TOPS = 38.05
EXTERNAL_MMUL_FUNCTION = "matmul_i8_i8_i8_acc32_strix"
DEFAULT_EXTERNAL_K_PACKS = 9
DEFAULT_EXTERNAL_BLOCK_M = 2
DEFAULT_EXTERNAL_BLOCK_N = 2


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


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def read_text(path: Path | None) -> str:
    if not path:
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def render_transform_variant(
    transform_text: str, variant: str, k_packs: int = 9
) -> str:
    if variant == "default":
        return transform_text
    if variant != "sota-int8":
        raise ValueError(f"unknown transform variant: {variant}")

    k_elements = k_packs * 8
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
            "tile_using_forall %matmul_1 tile_sizes [18, 18, 0]",
        ),
        (
            "tile_using_forall %interchanged_fill_op tile_sizes [8, 8]",
            "tile_using_forall %interchanged_fill_op tile_sizes [18, 18]",
        ),
        (
            "tile_using_forall %unpack_op tile_sizes [64, 64]",
            "tile_using_forall %unpack_op tile_sizes [144, 144]",
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


def first_int(text: str, pattern: str, default: int) -> int:
    match = re.search(pattern, text, flags=re.MULTILINE)
    return int(match.group(1)) if match else default


def first_int_tuple(
    text: str, pattern: str, default: tuple[int, ...]
) -> tuple[int, ...]:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        return default
    return tuple(int(group) for group in match.groups())


def parse_transform(transform_text: str) -> dict[str, Any]:
    pack_m, pack_n, pack_k = first_int_tuple(
        transform_text,
        r"pack %matmul packed_sizes = \[(\d+),\s*(\d+),\s*(\d+)\]",
        (8, 8, 8),
    )
    copy_a_k = first_int(
        transform_text,
        r"tile_using_for %copy1 tile_sizes \[0,\s*(\d+)\]",
        64,
    )
    copy_b_k = first_int(
        transform_text,
        r"tile_using_for %copy2 tile_sizes \[(\d+)\]",
        64,
    )
    k_reduction_outer = first_int(
        transform_text,
        r"tile_using_for %packed_c tile_sizes \[0,\s*0,\s*(\d+)\]",
        8,
    )
    forall_m, forall_n = first_int_tuple(
        transform_text,
        r"tile_using_forall %matmul_1 tile_sizes \[(\d+),\s*(\d+),\s*0\]",
        (8, 8),
    )
    vec_m, vec_n, vec_k = first_int_tuple(
        transform_text,
        r"tile_using_for %generic2 tile_sizes \[(\d+),\s*(\d+),\s*(\d+),\s*0,\s*0,\s*0\]",
        (2, 2, 1),
    )
    epilogue_m, epilogue_n = first_int_tuple(
        transform_text,
        r"tile_using_forall %unpack_op tile_sizes \[(\d+),\s*(\d+)\]",
        (64, 64),
    )
    return {
        "pack_sizes": [pack_m, pack_n, pack_k],
        "copy_a_k_tile": copy_a_k,
        "copy_b_k_tile": copy_b_k,
        "k_reduction_outer_tile": k_reduction_outer,
        "k_reduction_elements_per_step": k_reduction_outer * pack_k,
        "forall_tile_packs": [forall_m, forall_n],
        "core_tile_elements": [forall_m * pack_m, forall_n * pack_n],
        "vector_tile": [vec_m, vec_n, vec_k],
        "epilogue_tile_elements": [epilogue_m, epilogue_n],
        "has_l2_fuse": "transform.loop.fuse_sibling" in transform_text,
        "has_l1_allocs": "memory_space = 2" in transform_text,
        "has_l2_result_alloc": "memory_space = 1" in transform_text,
        "has_herd_vectorize": "transform.air.herd_vectorize" in transform_text,
    }


def bytes_fmt(value: int) -> str:
    if value >= 1024 * 1024:
        return f"{value / (1024 * 1024):.3f} MiB"
    if value >= 1024:
        return f"{value / 1024:.3f} KiB"
    return f"{value} B"


def make_report(args: argparse.Namespace) -> dict[str, Any]:
    transform_text = render_transform_variant(
        read_text(args.transform_script),
        args.transform_variant,
        args.external_k_packs if args.kernel_impl == "external-mmul" else 9,
    )
    final_ir = read_text(args.transformed_ir)
    transform = parse_transform(transform_text)
    out_bytes = 4 if args.output_type == "int32" else 1
    launch_grid = [args.m // args.tile_m, args.n // args.tile_n]
    ops = 2 * args.m * args.k * args.n
    ideal_bytes = args.m * args.k + args.k * args.n + args.m * args.n * out_bytes
    launch_l2 = {
        "a_bytes": args.tile_m * args.k,
        "b_bytes": args.k * args.tile_n,
        "c_bytes": args.tile_m * args.tile_n * out_bytes,
    }
    core_m, core_n = transform["core_tile_elements"]
    k_step = transform["k_reduction_elements_per_step"]
    a_step_bytes = core_m * k_step
    b_step_bytes = k_step * core_n
    c_bytes = core_m * core_n * out_bytes
    estimated_stack_bytes = 1024
    l1_budget_bytes = 64 * 1024
    single_buffered_working_set_bytes = a_step_bytes + b_step_bytes + c_bytes
    pingpong_working_set_bytes = 2 * (a_step_bytes + b_step_bytes) + c_bytes
    partial_pingpong_a_working_set_bytes = 2 * a_step_bytes + b_step_bytes + c_bytes
    partial_pingpong_b_working_set_bytes = a_step_bytes + 2 * b_step_bytes + c_bytes
    k_pack_bytes_a = core_m * transform["pack_sizes"][2]
    k_pack_bytes_b = core_n * transform["pack_sizes"][2]
    l1_available_for_ab = l1_budget_bytes - estimated_stack_bytes - c_bytes
    max_full_pingpong_k_packs = max(
        0, l1_available_for_ab // (2 * (k_pack_bytes_a + k_pack_bytes_b))
    )
    max_partial_pingpong_k_packs = max(
        0,
        l1_available_for_ab
        // min(
            2 * k_pack_bytes_a + k_pack_bytes_b, k_pack_bytes_a + 2 * k_pack_bytes_b
        ),
    )
    core_l1 = {
        "a_step_bytes": a_step_bytes,
        "b_step_bytes": b_step_bytes,
        "c_bytes": c_bytes,
        "double_buffered_a_b_step_bytes": 2 * (a_step_bytes + b_step_bytes),
        "single_buffered_working_set_bytes": single_buffered_working_set_bytes,
        "pingpong_working_set_bytes": pingpong_working_set_bytes,
        "partial_pingpong_a_working_set_bytes": partial_pingpong_a_working_set_bytes,
        "partial_pingpong_b_working_set_bytes": partial_pingpong_b_working_set_bytes,
        "estimated_stack_bytes": estimated_stack_bytes,
        "l1_budget_bytes": l1_budget_bytes,
        "max_full_pingpong_k_packs": max_full_pingpong_k_packs,
        "max_partial_pingpong_k_packs": max_partial_pingpong_k_packs,
        "fits_single_buffer_estimate": (
            single_buffered_working_set_bytes + estimated_stack_bytes <= l1_budget_bytes
        ),
        "fits_pingpong_estimate": (
            pingpong_working_set_bytes + estimated_stack_bytes <= l1_budget_bytes
        ),
        "fits_partial_pingpong_estimate": (
            min(
                partial_pingpong_a_working_set_bytes,
                partial_pingpong_b_working_set_bytes,
            )
            + estimated_stack_bytes
            <= l1_budget_bytes
        ),
    }
    packed_m = args.tile_m // transform["pack_sizes"][0]
    packed_n = args.tile_n // transform["pack_sizes"][1]
    logical_cores = [
        packed_m // transform["forall_tile_packs"][0],
        packed_n // transform["forall_tile_packs"][1],
    ]
    transform_pingpong_candidate = bool(
        transform["has_l2_fuse"]
        and transform["has_l1_allocs"]
        and transform["has_l2_result_alloc"]
    )
    partial_l1_pingpong_requested = args.omit_ping_pong in (
        "L1-partial-a",
        "L1-partial-b",
    )
    l1_pingpong_requested = args.omit_ping_pong not in ("L1", "all")
    pingpong_eligible = bool(
        transform_pingpong_candidate
        and l1_pingpong_requested
        and (
            core_l1["fits_pingpong_estimate"]
            or (
                partial_l1_pingpong_requested
                and core_l1["fits_partial_pingpong_estimate"]
            )
        )
    )
    if not transform_pingpong_candidate:
        pingpong_reason = "missing loop fusion or expected L1/L2 allocation markers in transform script"
    elif not l1_pingpong_requested:
        pingpong_reason = "L1 ping-pong transform is explicitly omitted"
    elif partial_l1_pingpong_requested and core_l1["fits_partial_pingpong_estimate"]:
        pingpong_reason = (
            "partial L1 ping-pong is requested and the estimated single-stream "
            "double-buffered working set fits the 64 KiB core-memory budget"
        )
    elif not core_l1["fits_pingpong_estimate"]:
        pingpong_reason = (
            "estimated L1 working set exceeds the 64 KiB core-memory budget with "
            "fully ping-ponged A/B buffers"
        )
    else:
        pingpong_reason = (
            "L3 copy loops are fused with the K-reduction loop, L1/L2 allocations "
            "are present, and the estimated L1 working set fits"
        )
    b_dma = (
        f"B source is row-major KxN; tile view strides [{args.n},1], contiguous along N"
        if args.b_layout == "row"
        else f"B source is packed by N tile [N/tile_N,K,tile_N+4]; logical tile strides [{args.tile_n + 4},1]"
    )
    dependency_nodes = [
        "L3_A_copy",
        "L3_B_copy",
        "L2_C_fill",
        "compute_herd",
        "L2_C_unpack",
        "L3_C_store",
    ]
    if "air.dma_memcpy_nd" in final_ir:
        dma_count = len(re.findall(r"\bair\.dma_memcpy_nd\b", final_ir))
    else:
        dma_count = "unknown_until_debug_ir"
    return {
        "shape": {"m": args.m, "k": args.k, "n": args.n},
        "tile_shape": {"m": args.tile_m, "n": args.tile_n, "k": args.k},
        "target": "Ryzen AI HX 370 / NPU Strix / XDNA2 / aie2p",
        "published_sota_tops": PUBLISHED_SOTA_TOPS,
        "acceptance_tops": args.acceptance_tops,
        "output_type": args.output_type,
        "b_layout": args.b_layout,
        "runtime_loop_tiling": args.runtime_loop_tiling,
        "transform_variant": args.transform_variant,
        "kernel_impl": args.kernel_impl,
        "external_kernel_function": (
            EXTERNAL_MMUL_FUNCTION if args.kernel_impl == "external-mmul" else None
        ),
        "external_k_packs": args.external_k_packs,
        "external_block": [args.external_block_m, args.external_block_n],
        "external_block_accumulators": (
            args.external_block_m * args.external_block_n
            if args.kernel_impl == "external-mmul"
            else None
        ),
        "external_block_spill_risk": (
            args.kernel_impl == "external-mmul"
            and args.external_block_m * args.external_block_n > 4
        ),
        "omit_ping_pong": args.omit_ping_pong,
        "requires_partial_l1_pingpong": (
            args.kernel_impl == "external-mmul"
            and not core_l1["fits_pingpong_estimate"]
            and core_l1["fits_partial_pingpong_estimate"]
        ),
        "trace_mode": "packet" if args.trace_size else "off",
        "trace_size": args.trace_size,
        "transform": transform,
        "launch_tile": [args.tile_m, args.tile_n, args.k],
        "launch_grid": launch_grid,
        "logical_core_split_m_by_n": logical_cores,
        "physical_mapping_note": "logical 8x4 M/N split maps onto the 4x8 physical Strix array by placement; verify final AIE placement in debug IR",
        "l2_per_launch_bytes": launch_l2,
        "l1_per_core_bytes": core_l1,
        "ideal_bytes": ideal_bytes,
        "ops": ops,
        "arithmetic_intensity_ops_per_byte": ops / ideal_bytes,
        "dma_contiguity": {
            "A": f"A tile strides [{args.k},1], contiguous along K",
            "B": b_dma,
            "C": f"C tile strides [{args.n},1], contiguous along N",
            "byte_granularity": "all non-innermost i8 DMA strides are expected to be multiples of 4 bytes",
        },
        "dependency_graph": {
            "nodes": dependency_nodes,
            "shape": (
                "L3 copies fused into K-reduction compute loop, followed by output unpack/store"
                if transform["has_l2_fuse"]
                else "unfused L3 copies followed by compute and output store"
            ),
            "dma_memcpy_nd_count": dma_count,
        },
        "pingpong_eligibility": {
            "eligible": pingpong_eligible,
            "reason": pingpong_reason,
        },
        "sota_design_checks": {
            "output_stationary_c": True,
            "single_c_buffer": True,
            "double_buffered_a_b_candidate": pingpong_eligible,
            "b_column_major": args.b_layout == "column",
            "spatial_split_4x8_or_8x4": logical_cores in ([8, 4], [4, 8]),
            "k_reduction_in_time": transform["k_reduction_elements_per_step"] < args.k,
            "bd_overlap_candidate": pingpong_eligible
            and args.runtime_loop_tiling != "1,1",
            "dense_aie2p_mmul_candidate": args.kernel_impl == "external-mmul",
        },
    }


def write_markdown(path: Path, report: dict[str, Any], json_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    l2 = report["l2_per_launch_bytes"]
    l1 = report["l1_per_core_bytes"]
    transform = report["transform"]
    checks = report["sota_design_checks"]
    with path.open("w", encoding="utf-8") as f:
        f.write("# XDNA2 INT8 GEMM Static Report\n\n")
        f.write("## Candidate\n\n")
        f.write("| Field | Value |\n| --- | --- |\n")
        f.write(f"| Target | `{report['target']}` |\n")
        f.write(
            f"| Shape | `M={report['shape']['m']} K={report['shape']['k']} N={report['shape']['n']}` |\n"
        )
        f.write(
            f"| Tile shape | `M={report['tile_shape']['m']} K={report['tile_shape']['k']} N={report['tile_shape']['n']}` |\n"
        )
        f.write(f"| Output type | `{report['output_type']}` |\n")
        f.write(f"| B layout | `{report['b_layout']}` |\n")
        f.write(f"| Runtime loop tiling | `{report['runtime_loop_tiling']}` |\n")
        f.write(f"| Transform variant | `{report['transform_variant']}` |\n")
        f.write(f"| Omit ping-pong | `{report['omit_ping_pong']}` |\n")
        f.write(
            f"| Requires partial L1 ping-pong | `{'yes' if report['requires_partial_l1_pingpong'] else 'no'}` |\n"
        )
        f.write(f"| Kernel impl | `{report['kernel_impl']}` |\n")
        if report["external_kernel_function"]:
            f.write(f"| External kernel | `{report['external_kernel_function']}` |\n")
            f.write(f"| External K packs | `{report['external_k_packs']}` |\n")
            f.write(f"| External block | `{report['external_block']}` |\n")
            f.write(
                f"| External accumulators | `{report['external_block_accumulators']}` |\n"
            )
            f.write(
                f"| External spill risk | `{'yes' if report['external_block_spill_risk'] else 'no'}` |\n"
            )
        f.write(f"| Trace mode | `{report['trace_mode']}` |\n")
        f.write(f"| Acceptance TOPS | `{report['acceptance_tops']:.2f}` |\n")
        f.write(f"| Published SOTA TOPS | `{report['published_sota_tops']:.2f}` |\n")
        f.write("\n## Tiling And Placement\n\n")
        f.write("| Field | Value |\n| --- | --- |\n")
        f.write(f"| Launch tile | `{report['launch_tile']}` |\n")
        f.write(f"| Launch grid | `{report['launch_grid']}` |\n")
        f.write(f"| Pack sizes | `{transform['pack_sizes']}` |\n")
        f.write(
            f"| K reduction step | `{transform['k_reduction_elements_per_step']}` elements |\n"
        )
        f.write(f"| Core tile | `{transform['core_tile_elements']}` elements |\n")
        f.write(
            f"| Logical core split M/N | `{report['logical_core_split_m_by_n']}` |\n"
        )
        f.write(f"| Placement note | {report['physical_mapping_note']} |\n")
        f.write("\n## Memory And DMA\n\n")
        f.write("| Field | Value |\n| --- | --- |\n")
        f.write(f"| L2 A per launch | `{bytes_fmt(l2['a_bytes'])}` |\n")
        f.write(f"| L2 B per launch | `{bytes_fmt(l2['b_bytes'])}` |\n")
        f.write(f"| L2 C per launch | `{bytes_fmt(l2['c_bytes'])}` |\n")
        f.write(f"| L1 A per core K step | `{bytes_fmt(l1['a_step_bytes'])}` |\n")
        f.write(f"| L1 B per core K step | `{bytes_fmt(l1['b_step_bytes'])}` |\n")
        f.write(f"| L1 C per core | `{bytes_fmt(l1['c_bytes'])}` |\n")
        f.write(
            f"| L1 double-buffer A/B step | `{bytes_fmt(l1['double_buffered_a_b_step_bytes'])}` |\n"
        )
        f.write(
            f"| L1 single-buffer working set | `{bytes_fmt(l1['single_buffered_working_set_bytes'])}` |\n"
        )
        f.write(
            f"| L1 ping-pong working set | `{bytes_fmt(l1['pingpong_working_set_bytes'])}` |\n"
        )
        f.write(
            f"| L1 partial ping-pong A working set | `{bytes_fmt(l1['partial_pingpong_a_working_set_bytes'])}` |\n"
        )
        f.write(
            f"| L1 partial ping-pong B working set | `{bytes_fmt(l1['partial_pingpong_b_working_set_bytes'])}` |\n"
        )
        f.write(
            f"| L1 estimated stack | `{bytes_fmt(l1['estimated_stack_bytes'])}` |\n"
        )
        f.write(f"| L1 budget | `{bytes_fmt(l1['l1_budget_bytes'])}` |\n")
        f.write(
            f"| Max full ping-pong K packs | `{l1['max_full_pingpong_k_packs']}` |\n"
        )
        f.write(
            f"| Max partial ping-pong K packs | `{l1['max_partial_pingpong_k_packs']}` |\n"
        )
        f.write(
            f"| L1 fits single-buffer estimate | `{'yes' if l1['fits_single_buffer_estimate'] else 'no'}` |\n"
        )
        f.write(
            f"| L1 fits ping-pong estimate | `{'yes' if l1['fits_pingpong_estimate'] else 'no'}` |\n"
        )
        f.write(
            f"| L1 fits partial ping-pong estimate | `{'yes' if l1['fits_partial_pingpong_estimate'] else 'no'}` |\n"
        )
        f.write(f"| A DMA | {report['dma_contiguity']['A']} |\n")
        f.write(f"| B DMA | {report['dma_contiguity']['B']} |\n")
        f.write(f"| C DMA | {report['dma_contiguity']['C']} |\n")
        f.write(
            f"| DMA byte granularity | {report['dma_contiguity']['byte_granularity']} |\n"
        )
        f.write("\n## Dependency And Intensity\n\n")
        f.write("| Field | Value |\n| --- | --- |\n")
        f.write(f"| Dependency shape | {report['dependency_graph']['shape']} |\n")
        f.write(
            f"| DMA op count in transformed IR | `{report['dependency_graph']['dma_memcpy_nd_count']}` |\n"
        )
        f.write(
            f"| Ping-pong eligible | `{'yes' if report['pingpong_eligibility']['eligible'] else 'no'}` |\n"
        )
        f.write(f"| Ping-pong reason | {report['pingpong_eligibility']['reason']} |\n")
        f.write(f"| Ideal bytes | `{bytes_fmt(report['ideal_bytes'])}` |\n")
        f.write(
            f"| Arithmetic intensity | `{report['arithmetic_intensity_ops_per_byte']:.3f}` ops/byte |\n"
        )
        f.write("\n## SOTA Design Checks\n\n")
        f.write("| Check | Status |\n| --- | --- |\n")
        for key, value in checks.items():
            f.write(f"| `{key}` | `{'yes' if value else 'no'}` |\n")
        f.write(f"\nJSON: `{json_path}`\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-M", "--m", type=positive_int, default=1024)
    parser.add_argument("-K", "--k", type=positive_int, default=1024)
    parser.add_argument("-N", "--n", type=positive_int, default=1024)
    parser.add_argument("--tile-m", type=positive_int, default=DEFAULT_TILE_M)
    parser.add_argument("--tile-n", type=positive_int, default=DEFAULT_TILE_N)
    parser.add_argument("--output-type", choices=["int32", "int8"], default="int32")
    parser.add_argument("--b-layout", choices=["row", "column"], default="row")
    parser.add_argument("--runtime-loop-tiling", default="2,4")
    parser.add_argument("--trace-size", type=nonnegative_int, default=0)
    parser.add_argument(
        "--transform-variant", choices=["default", "sota-int8"], default="default"
    )
    parser.add_argument(
        "--kernel-impl", choices=["vectorized", "external-mmul"], default="vectorized"
    )
    parser.add_argument(
        "--external-k-packs",
        type=positive_int,
        default=DEFAULT_EXTERNAL_K_PACKS,
    )
    parser.add_argument(
        "--external-block-m", type=positive_int, default=DEFAULT_EXTERNAL_BLOCK_M
    )
    parser.add_argument(
        "--external-block-n", type=positive_int, default=DEFAULT_EXTERNAL_BLOCK_N
    )
    parser.add_argument(
        "--omit-ping-pong",
        choices=["L1", "L2", "all", "L1-partial-a", "L1-partial-b"],
        default=None,
    )
    parser.add_argument(
        "--acceptance-tops", type=positive_float, default=DEFAULT_ACCEPTANCE_TOPS
    )
    parser.add_argument("--transform-script", type=Path, required=True)
    parser.add_argument("--transformed-ir", type=Path, default=None)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="npu_static")
    args = parser.parse_args()
    if args.m % args.tile_m or args.n % args.tile_n:
        parser.error(
            f"M and N must be multiples of tile sizes {args.tile_m} and {args.tile_n}"
        )
    if args.k % 8:
        parser.error("K must be a multiple of 8")
    return args


def main() -> int:
    args = parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    report = make_report(args)
    json_path = args.artifact_dir / f"{args.prefix}.static.json"
    md_path = args.artifact_dir / f"{args.prefix}.static.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_markdown(md_path, report, json_path)
    print(f"static_report={md_path}")
    print(f"static_report_json={json_path}")
    print(
        f"arithmetic_intensity_ops_per_byte={report['arithmetic_intensity_ops_per_byte']:.6f}"
    )
    print(
        f"pingpong_eligible={'yes' if report['pingpong_eligibility']['eligible'] else 'no'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
