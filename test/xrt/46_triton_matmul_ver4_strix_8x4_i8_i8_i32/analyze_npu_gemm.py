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
DEFAULT_EXTERNAL_CORE_M_PACKS = 18
DEFAULT_EXTERNAL_ACTIVE_M_PACKS = 18
DEFAULT_EXTERNAL_CORE_N_PACKS = 18
ATB_EXTERNAL_K_PACKS = 18
ATB_EXTERNAL_ACTIVE_M_PACKS = 6
DEFAULT_ATB_K_CHUNK_ELEMENTS = 864
ATB_V2_MAX_A_L2_CHUNK_ELEMENTS = DEFAULT_ATB_K_CHUNK_ELEMENTS
ATB_TRANSFORM_VARIANTS = ("sota-int8-atb", "sota-int8-atb-v2")
DEFAULT_EXTERNAL_SCHEDULE = "software-pipeline"
DEFAULT_EXTERNAL_KERNEL_STYLE = "peano-mmul"
EXTERNAL_KERNEL_STYLES = (
    "peano-mmul",
    "hand-scheduled",
    "native-mmul",
    "native-mmul-atb-ref",
    "asm-microkernel",
)
ATB_V2_EXTERNAL_KERNEL_STYLES = ("native-mmul", "native-mmul-atb-ref")


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


def is_atb_variant(variant: str) -> bool:
    return variant in ATB_TRANSFORM_VARIANTS


def is_atb_v2(variant: str) -> bool:
    return variant == "sota-int8-atb-v2"


def uses_full_m_external_k_chunking(args: argparse.Namespace) -> bool:
    return (
        args.kernel_impl == "external-mmul"
        and args.transform_variant == "sota-int8"
        and args.external_k_packs > DEFAULT_EXTERNAL_K_PACKS
    )


def choose_atb_k_chunk_elements(
    problem_k: int,
    requested: int,
    k_step: int,
    max_chunk: int | None = None,
) -> int:
    if requested <= 0:
        raise ValueError("ATB K chunk size must be positive")
    if problem_k % k_step:
        raise ValueError(f"ATB K={problem_k} must be a multiple of {k_step}")
    limit = min(problem_k, requested)
    if max_chunk is not None:
        limit = min(limit, max_chunk)
    limit -= limit % k_step
    if limit < k_step:
        limit = k_step
    for candidate in range(limit, k_step - 1, -k_step):
        if problem_k % candidate == 0:
            return candidate
    return k_step


def render_transform_variant(
    transform_text: str,
    variant: str,
    k_packs: int = 9,
    fuse_l3_l2: bool = True,
    active_m_packs: int = DEFAULT_EXTERNAL_ACTIVE_M_PACKS,
) -> str:
    if variant == "default":
        return transform_text
    if variant not in ("sota-int8",) + ATB_TRANSFORM_VARIANTS:
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
    if is_atb_variant(variant):
        active_fragment = """        %tiled_matmul_1, %inner_forall =
          transform.structured.tile_using_forall %matmul_1 tile_sizes [18, 18, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        transform.annotate %inner_forall "compute_forall" : !transform.any_op
        transform.annotate %tiled_matmul_1 "matmul_compute" : !transform.any_op
"""
        active_replacement = f"""        %tiled_matmul_1, %inner_forall =
          transform.structured.tile_using_forall %matmul_1 tile_sizes [18, 18, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        transform.annotate %inner_forall "compute_forall" : !transform.any_op
        %active_m_matmul, %active_m_loop =
          transform.structured.tile_using_for %tiled_matmul_1 tile_sizes [{active_m_packs}, 0, 0, 0, 0, 0]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        transform.annotate %active_m_loop "atb_active_m_loop" : !transform.any_op
        transform.annotate %active_m_matmul "matmul_compute" : !transform.any_op
"""
        if rendered.count(active_fragment) != 1:
            raise ValueError("expected one ATB active-M transform fragment")
        rendered = rendered.replace(active_fragment, active_replacement, 1)
    if not fuse_l3_l2:
        phase8_marker = (
            "    //==========================================================================\n"
            "    // PHASE 8:"
        )
        if rendered.count(phase8_marker) != 1:
            raise ValueError("expected one PHASE 8 marker in transform script")
        rendered = rendered.split(phase8_marker, 1)[0]
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
    active_m_packs = first_int(
        transform_text,
        r"tile_using_for %tiled_matmul_1 tile_sizes \[(\d+),\s*0,\s*0,\s*0,\s*0,\s*0\]",
        forall_m,
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
        "active_m_packs": active_m_packs,
        "active_m_elements": active_m_packs * pack_m,
        "has_atb_active_m_loop": "atb_active_m_loop" in transform_text,
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


def external_variant_role(args: argparse.Namespace) -> str | None:
    if args.kernel_impl != "external-mmul":
        return None
    block = (args.external_block_m, args.external_block_n)
    if args.transform_variant == "sota-int8-atb":
        return "baseline-atb-active-c-scratch-native-2x2"
    if args.transform_variant == "sota-int8-atb-v2":
        if args.external_kernel_style == "native-mmul-atb-ref":
            return "candidate-atb-v2-ref-cadence-native-2x2"
        return "production-atb-v2-full-c-native-2x2"
    if args.external_kernel_style == "hand-scheduled":
        return "diagnostic-hand-scheduled"
    if args.external_kernel_style == "native-mmul":
        if block == (2, 2):
            return "production-spill-free-native-unrolled"
        if block == (3, 2):
            return "blocked-native-over-dm-register-budget"
        return "diagnostic-native-microkernel"
    if args.external_kernel_style == "native-mmul-atb-ref":
        if block == (2, 2):
            return "candidate-ref-cadence-native-2x2"
        return "invalid-ref-cadence-shape"
    if args.external_kernel_style == "asm-microkernel":
        if block == (2, 2):
            return "diagnostic-asm-spill-free-baseline"
        if block == (3, 2):
            return "blocked-asm-over-dm-register-budget"
        return "diagnostic-asm-microkernel"
    if block == (2, 2):
        return "fallback-baseline"
    if block == (3, 2):
        return "production-candidate"
    if block == (2, 3):
        return "diagnostic-control"
    if block == (4, 2):
        return "diagnostic-main-plus-tail"
    return "diagnostic"


def make_report(args: argparse.Namespace) -> dict[str, Any]:
    transform_text = render_transform_variant(
        read_text(args.transform_script),
        args.transform_variant,
        args.external_k_packs if args.kernel_impl == "external-mmul" else 9,
        fuse_l3_l2=(
            args.kernel_impl != "external-mmul" or args.k > args.external_k_packs * 8
        ),
        active_m_packs=args.external_active_m_packs,
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
    active_m = (
        transform["active_m_elements"]
        if is_atb_variant(args.transform_variant)
        else core_m
    )
    a_step_bytes = active_m * k_step
    b_step_bytes = k_step * core_n
    c_bytes = core_m * core_n * out_bytes
    estimated_stack_bytes = 1024
    l1_budget_bytes = 64 * 1024
    single_buffered_working_set_bytes = a_step_bytes + b_step_bytes + c_bytes
    pingpong_working_set_bytes = 2 * (a_step_bytes + b_step_bytes) + c_bytes
    partial_pingpong_a_working_set_bytes = 2 * a_step_bytes + b_step_bytes + c_bytes
    partial_pingpong_b_working_set_bytes = a_step_bytes + 2 * b_step_bytes + c_bytes
    k_pack_bytes_a = active_m * transform["pack_sizes"][2]
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
    atb_ratio = (
        args.external_core_m_packs // args.external_active_m_packs
        if is_atb_variant(args.transform_variant)
        else None
    )
    atb_active_c_scratch_bytes = (
        active_m * core_n * out_bytes
        if args.transform_variant == "sota-int8-atb"
        else 0
    )
    atb_l1_footprint_bytes = (
        partial_pingpong_a_working_set_bytes + atb_active_c_scratch_bytes
    )
    atb_k_blocks_per_launch = 0
    if is_atb_variant(args.transform_variant):
        if is_atb_v2(args.transform_variant):
            atb_k_blocks_per_launch = math.ceil(
                args.k / args.effective_atb_k_chunk_elements
            )
        else:
            atb_k_blocks_per_launch = math.ceil(
                args.k / (args.external_k_packs * transform["pack_sizes"][2])
            )
    atb_memtile_bd_limit = 24
    atb_a_l2_to_l1_bd_fits = atb_k_blocks_per_launch <= atb_memtile_bd_limit
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
        "external_schedule": args.external_schedule,
        "external_kernel_style": args.external_kernel_style,
        "external_variant_role": external_variant_role(args),
        "external_k_pack_status": (
            "atb-k18"
            if is_atb_variant(args.transform_variant)
            else (
                "full-m-k-chunked"
                if uses_full_m_external_k_chunking(args)
                else (
                    "acceptance"
                    if args.external_k_packs == DEFAULT_EXTERNAL_K_PACKS
                    else "diagnostic-k-residency"
                )
            )
        ),
        "external_k_packs": args.external_k_packs,
        "external_k_chunk_elements": args.effective_atb_k_chunk_elements or None,
        "external_l3_k_chunks": (
            math.ceil(args.k / args.effective_atb_k_chunk_elements)
            if args.effective_atb_k_chunk_elements
            else None
        ),
        "external_block": [args.external_block_m, args.external_block_n],
        "external_core_m_packs": args.external_core_m_packs,
        "external_active_m_packs": args.external_active_m_packs,
        "external_core_n_packs": args.external_core_n_packs,
        "external_c_stride_m_packs": args.external_c_stride_m_packs,
        "atb_k_chunk_elements": args.effective_atb_k_chunk_elements,
        "atb": (
            {
                "rho": atb_ratio,
                "active_m_packs": args.external_active_m_packs,
                "core_m_packs": args.external_core_m_packs,
                "core_n_packs": args.external_core_n_packs,
                "k_packs": args.external_k_packs,
                "k_chunk_elements": args.effective_atb_k_chunk_elements,
                "c_stride_m_packs": args.external_c_stride_m_packs,
                "b_reuse_factor": atb_ratio,
                "l1_footprint_bytes": atb_l1_footprint_bytes,
                "active_c_scratch_bytes": atb_active_c_scratch_bytes,
                "accumulator_register_budget": {
                    "needed": args.external_block_m * args.external_block_n,
                    "available_dm_acc2048": 5,
                },
                "a_l2_to_l1_bd_budget": {
                    "estimated_bds_per_channel": atb_k_blocks_per_launch,
                    "memtile_bd_limit_per_channel": atb_memtile_bd_limit,
                    "fits": atb_a_l2_to_l1_bd_fits,
                    "max_k_elements_without_overflow": (
                        atb_memtile_bd_limit
                        * (
                            args.effective_atb_k_chunk_elements
                            if is_atb_v2(args.transform_variant)
                            else args.external_k_packs * transform["pack_sizes"][2]
                        )
                    ),
                    "note": (
                        "ATB v2 chunks L2 A/B residency before AIR lowering; "
                        "the baseline ATB path still estimates one A L2-to-L1 "
                        "BD per 144-element K block."
                    ),
                },
            }
            if is_atb_variant(args.transform_variant)
            else None
        ),
        "external_block_accumulators": (
            args.external_block_m * args.external_block_n
            if args.kernel_impl == "external-mmul"
            else None
        ),
        "external_block_spill_risk": (
            args.kernel_impl == "external-mmul"
            and args.external_block_m * args.external_block_n > 4
        ),
        "external_asm_gate_required": args.kernel_impl == "external-mmul",
        "external_main_tail": (
            args.kernel_impl == "external-mmul"
            and [args.external_block_m, args.external_block_n] == [4, 2]
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
            "atb_active_m_loop": transform["has_atb_active_m_loop"],
            "bd_overlap_candidate": pingpong_eligible
            and args.runtime_loop_tiling != "1,1",
            "dense_aie2p_mmul_candidate": args.kernel_impl == "external-mmul",
            "atb_a_l2_to_l1_bd_budget": (
                not is_atb_variant(args.transform_variant) or atb_a_l2_to_l1_bd_fits
            ),
        },
    }


def verify_atb_final_ir(
    final_ir: str, args: argparse.Namespace, lowered_ir: str = ""
) -> dict[str, Any] | None:
    if not is_atb_variant(args.transform_variant):
        return None
    if not final_ir.strip():
        return {
            "available": False,
            "passed": None,
            "issues": [
                "final transformed IR was not available for ATB loop-placement checks"
            ],
        }
    issues: list[str] = []
    expected_a = (
        f"memref<{args.external_k_packs}x{args.external_active_m_packs}x8x8xi8, 2>"
    )
    expected_b = (
        f"memref<{args.external_core_n_packs}x{args.external_k_packs}x8x8xi8, 2>"
    )
    expected_c = (
        f"memref<{args.external_core_n_packs * 8}x" f"{(args.tile_m // 8)}x8x8xi8, 2>"
    )
    active_c_scratch = False
    for outs_match in re.finditer(
        rf"linalg\.generic .* outs\((?P<view>%[\w]+) : memref<18x{args.external_active_m_packs}x8x8xi8, 2>\)",
        final_ir,
    ):
        view = outs_match.group("view")
        if (
            f"{view} = memref.alloc() : memref<18x{args.external_active_m_packs}x8x8xi8, 2>"
            in final_ir
        ):
            active_c_scratch = True
            break
    b_alloc_pos = final_ir.find(f"memref.alloc() : {expected_b}")
    a_alloc_pos = final_ir.find(f"memref.alloc() : {expected_a}")
    c_alloc_pos = final_ir.find(f"memref.alloc() : {expected_c}")
    library_pos = final_ir.find(f"func.call @{EXTERNAL_MMUL_FUNCTION}")
    if library_pos < 0:
        library_pos = final_ir.find(f'library_call = "{EXTERNAL_MMUL_FUNCTION}"')
    if b_alloc_pos >= 0 and a_alloc_pos >= 0:
        active_loop_start = final_ir.rfind("scf.for", b_alloc_pos, a_alloc_pos)
    else:
        active_loop_start = -1
    active_loop_end = final_ir.find(
        "atb_active_m_loop", a_alloc_pos if a_alloc_pos >= 0 else 0
    )

    if active_loop_start < 0 or active_loop_end < 0:
        issues.append("expected an active-M loop over the core M tile")
    if a_alloc_pos < 0:
        issues.append(f"expected active A allocation {expected_a}")
    if b_alloc_pos < 0:
        issues.append(f"expected reusable B allocation {expected_b}")
    if c_alloc_pos < 0:
        issues.append(f"expected full C allocation {expected_c}")
    if library_pos < 0:
        issues.append(f"expected external library call {EXTERNAL_MMUL_FUNCTION}")
    if is_atb_v2(args.transform_variant) and active_c_scratch:
        issues.append(
            "ATB v2 must not route the external call through active-C scratch"
        )
    if is_atb_v2(args.transform_variant):
        if args.external_c_stride_m_packs != DEFAULT_EXTERNAL_CORE_M_PACKS:
            issues.append("ATB v2 requires C stride of 18 M packs")
        bad_active_reinterpret = (
            f"sizes: [18, {args.external_active_m_packs}, 8, 8], "
            "strides: [384, 64, 8, 1]"
        )
        if bad_active_reinterpret in final_ir:
            issues.append(
                "ATB v2 active C must not use a standalone active-tile reinterpret"
            )
        has_full_c_call = (
            f"func.call @{EXTERNAL_MMUL_FUNCTION}" in final_ir
            and "memref<18x18x8x8xi8, 2>) -> ()" in final_ir
        )
        has_full_c_stride = (
            "strides: [1152, 64, 8, 1]" in final_ir
            or "strided<[1152, 64, 8, 1]" in final_ir
        )
        if (
            "memref<18x18x8x8xi8" not in final_ir
            or not has_full_c_stride
            or not has_full_c_call
        ):
            issues.append(
                "expected ATB v2 external call to use a full-C tile with full-tile stride"
            )
    if b_alloc_pos >= 0 and active_loop_start >= 0 and b_alloc_pos > active_loop_start:
        issues.append("expected B allocation before the active-M loop for reuse")
    if a_alloc_pos >= 0:
        if not (active_loop_start <= a_alloc_pos <= active_loop_end):
            issues.append("expected active A allocation inside the active-M loop")
    if c_alloc_pos >= 0 and library_pos >= 0 and c_alloc_pos > library_pos:
        issues.append("expected C allocation before the external reduction call")

    elem_step = args.external_k_packs * 8
    pack_step = args.external_k_packs
    if (
        is_atb_v2(args.transform_variant)
        and args.k > args.effective_atb_k_chunk_elements
    ):
        chunk = args.effective_atb_k_chunk_elements
        if f"step %c{chunk}" not in final_ir:
            issues.append(f"expected ATB v2 outer K chunk loop with step {chunk}")
        if f"memref<{args.tile_m}x{chunk}xi8, 1 : i32>" not in final_ir:
            issues.append("expected ATB v2 chunk-sized A L2 allocation")
        if f"memref<{chunk}x{args.tile_n}xi8, 1 : i32>" not in final_ir:
            issues.append("expected ATB v2 chunk-sized B L2 allocation")
    if args.k > elem_step:
        has_k_step = (
            f"step %c{elem_step}" in final_ir or f"step %c{pack_step}" in final_ir
        )
        if not has_k_step:
            issues.append(
                f"expected K reduction loop with step {elem_step} elements/"
                f"{pack_step} packs"
            )
        b_l2_shape = (
            f"memref<{args.external_core_n_packs * 8}x"
            f"{args.external_k_packs * 8}xi8, 1"
        )
        channel_ir = lowered_ir or final_ir
        for line in channel_ir.splitlines():
            if "air.channel.put" not in line or b_l2_shape not in line:
                continue
            bracket_groups = re.findall(r"\[([^\]]*)\]", line)
            if len(bracket_groups) < 2:
                continue
            size_group = bracket_groups[-2]
            stride_group = bracket_groups[-1]
            first_size = re.match(r"\s*%c(-?\d+)(?:_\d+)?\b", size_group)
            first_stride = re.match(r"\s*%c(-?\d+)(?:_\d+)?\b", stride_group)
            if not first_size or not first_stride:
                continue
            if int(first_size.group(1)) > 1 and int(first_stride.group(1)) == 0:
                issues.append(
                    "expected B L2-to-L1 transfers to stay per-K-block; "
                    "found zero-stride repeated transfer across K blocks"
                )
                break
    return {
        "available": True,
        "lowered_ir_available": bool(lowered_ir.strip()),
        "passed": not issues,
        "issues": issues,
    }


def external_static_contract(
    args: argparse.Namespace, report: dict[str, Any]
) -> dict[str, Any] | None:
    if args.kernel_impl != "external-mmul":
        return None
    transform = report["transform"]
    issues: list[str] = []
    if args.output_type != "int8":
        issues.append("external-mmul requires INT8 output")
    if args.b_layout != "column":
        issues.append("external-mmul requires column-major packed B")
    if [args.tile_m, args.tile_n] != [576, 1152]:
        issues.append("external-mmul requires launch tile 576x1152")
    if transform["pack_sizes"] != [8, 8, 8]:
        issues.append(f"expected pack sizes [8, 8, 8], got {transform['pack_sizes']}")
    if transform["core_tile_elements"] != [144, 144]:
        issues.append(
            f"expected per-core tile [144, 144], got {transform['core_tile_elements']}"
        )
    expected_k_step = args.external_k_packs * transform["pack_sizes"][2]
    if transform["k_reduction_elements_per_step"] != expected_k_step:
        issues.append(
            f"expected K reduction step {expected_k_step} elements, got "
            f"{transform['k_reduction_elements_per_step']}"
        )
    if transform["epilogue_tile_elements"] != [144, 144]:
        issues.append(
            "expected INT8 unpack/epilogue tile [144, 144], got "
            f"{transform['epilogue_tile_elements']}"
        )
    if args.external_block_m not in (2, 3, 4) or args.external_block_n not in (2, 3):
        issues.append("external block shape must use M=2|3|4 and N=2|3")
    if [args.external_block_m, args.external_block_n] == [4, 3]:
        issues.append("external block shape 4x3 is not supported by the 18-pack M tile")
    if args.external_active_m_packs > args.external_core_m_packs:
        issues.append("active M packs must fit within core M packs")
    if args.external_core_m_packs % args.external_active_m_packs:
        issues.append("active M packs must divide core M packs")
    if args.external_active_m_packs % args.external_block_m:
        issues.append("active M packs must be divisible by external block M")
    if args.external_core_n_packs % args.external_block_n:
        issues.append("core N packs must be divisible by external block N")
    if args.external_kernel_style == "native-mmul-atb-ref" and not is_atb_v2(
        args.transform_variant
    ):
        issues.append("native-mmul-atb-ref is only valid with ATB v2")
    if is_atb_variant(args.transform_variant):
        if args.transform_variant == "sota-int8-atb":
            if args.external_kernel_style != "native-mmul":
                issues.append("baseline ATB requires native-mmul external kernel style")
        elif args.external_kernel_style not in ATB_V2_EXTERNAL_KERNEL_STYLES:
            issues.append(
                "ATB v2 requires native-mmul or native-mmul-atb-ref external kernel style"
            )
        if [args.external_block_m, args.external_block_n] != [2, 2]:
            issues.append("ATB requires a spill-free 2x2 native block")
        if args.external_k_packs != ATB_EXTERNAL_K_PACKS:
            issues.append("ATB requires K_PACKS=18")
        if args.external_core_m_packs != DEFAULT_EXTERNAL_CORE_M_PACKS:
            issues.append("ATB requires core M packs = 18")
        if args.external_active_m_packs != ATB_EXTERNAL_ACTIVE_M_PACKS:
            issues.append("ATB requires active M packs = 6")
        if args.external_core_n_packs != DEFAULT_EXTERNAL_CORE_N_PACKS:
            issues.append("ATB requires core N packs = 18")
        if args.omit_ping_pong != "L1-partial-a":
            issues.append("ATB requires omit-ping-pong=L1-partial-a")
        if (
            args.transform_variant == "sota-int8-atb"
            and args.external_c_stride_m_packs != ATB_EXTERNAL_ACTIVE_M_PACKS
        ):
            issues.append("baseline ATB requires C stride = 6 active M packs")
        if (
            is_atb_v2(args.transform_variant)
            and args.external_c_stride_m_packs != DEFAULT_EXTERNAL_CORE_M_PACKS
        ):
            issues.append("ATB v2 requires C stride = 18 full core M packs")
        if (
            is_atb_v2(args.transform_variant)
            and args.performance_mode
            and args.runtime_loop_tiling == "1,1"
        ):
            issues.append(
                "ATB v2 performance mode must not use RUNTIME_LOOP_TILING=1,1"
            )
        k_step = args.external_k_packs * transform["pack_sizes"][2]
        if args.k % k_step:
            issues.append(f"ATB requires K to be a multiple of {k_step}")
        atb = report.get("atb") or {}
        bd_budget = atb.get("a_l2_to_l1_bd_budget") or {}
        if (
            is_atb_v2(args.transform_variant)
            and bd_budget
            and not bd_budget.get("fits", True)
        ):
            issues.append(
                "ATB v2 exceeds current A L2-to-L1 memtile BD budget: "
                f"{bd_budget['estimated_bds_per_channel']} K-block BDs are "
                f"needed on one channel, but only "
                f"{bd_budget['memtile_bd_limit_per_channel']} are available; "
                "split or rework A DMA lowering before compiling this shape"
            )
        if not transform["has_atb_active_m_loop"]:
            issues.append("ATB transform must contain an active-M loop")
    if report["logical_core_split_m_by_n"] not in ([4, 8], [8, 4]):
        issues.append(
            "expected 4x8 or 8x4 logical core split, got "
            f"{report['logical_core_split_m_by_n']}"
        )
    if not transform["has_l1_allocs"]:
        issues.append("expected L1 allocations for A/B/C core residency")
    if not transform["has_l2_result_alloc"]:
        issues.append("expected L2 result allocation before L3 store")
    atb_final = report.get("atb_final_ir_verification")
    if atb_final and atb_final.get("available") and not atb_final.get("passed"):
        issues.extend(atb_final["issues"])
    return {"passed": not issues, "issues": issues}


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
            f.write(f"| External schedule | `{report['external_schedule']}` |\n")
            f.write(
                f"| External kernel style | `{report['external_kernel_style']}` |\n"
            )
            f.write(
                f"| External variant role | `{report['external_variant_role']}` |\n"
            )
            f.write(f"| External K packs | `{report['external_k_packs']}` |\n")
            f.write(
                f"| External K-pack status | `{report['external_k_pack_status']}` |\n"
            )
            if report["external_k_chunk_elements"]:
                f.write(
                    f"| External K chunk elements | `{report['external_k_chunk_elements']}` |\n"
                )
                f.write(
                    f"| External L3 K chunks | `{report['external_l3_k_chunks']}` |\n"
                )
            f.write(f"| External block | `{report['external_block']}` |\n")
            f.write(
                f"| External core M packs | `{report['external_core_m_packs']}` |\n"
            )
            f.write(
                f"| External active M packs | `{report['external_active_m_packs']}` |\n"
            )
            f.write(
                f"| External core N packs | `{report['external_core_n_packs']}` |\n"
            )
            if report["atb"]:
                f.write(f"| ATB rho | `{report['atb']['rho']}` |\n")
                f.write(
                    f"| ATB B reuse factor | `{report['atb']['b_reuse_factor']}` |\n"
                )
                f.write(
                    f"| ATB L1 footprint | `{bytes_fmt(report['atb']['l1_footprint_bytes'])}` |\n"
                )
                f.write(
                    f"| ATB active C scratch | `{bytes_fmt(report['atb']['active_c_scratch_bytes'])}` |\n"
                )
                f.write(
                    "| ATB accumulator budget | "
                    f"`{report['atb']['accumulator_register_budget']['needed']}/"
                    f"{report['atb']['accumulator_register_budget']['available_dm_acc2048']}` |\n"
                )
                bd_budget = report["atb"]["a_l2_to_l1_bd_budget"]
                f.write(
                    "| ATB A L2-to-L1 BD budget | "
                    f"`{bd_budget['estimated_bds_per_channel']}/"
                    f"{bd_budget['memtile_bd_limit_per_channel']}` |\n"
                )
                f.write(
                    "| ATB A BD budget fits | "
                    f"`{'yes' if bd_budget['fits'] else 'no'}` |\n"
                )
                f.write(
                    "| ATB max K before A BD overflow | "
                    f"`{bd_budget['max_k_elements_without_overflow']}` |\n"
                )
            f.write(
                f"| External accumulators | `{report['external_block_accumulators']}` |\n"
            )
            f.write(
                f"| External spill risk | `{'yes' if report['external_block_spill_risk'] else 'no'}` |\n"
            )
            f.write(
                f"| External main/tail split | `{'yes' if report['external_main_tail'] else 'no'}` |\n"
            )
            f.write(
                f"| External asm gate required | `{'yes' if report['external_asm_gate_required'] else 'no'}` |\n"
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
        atb_final = report.get("atb_final_ir_verification")
        if atb_final:
            f.write("\n## ATB Final IR Verification\n\n")
            status = (
                "NOT RUN"
                if not atb_final["available"]
                else ("PASS" if atb_final["passed"] else "FAIL")
            )
            f.write(f"Status: `{status}`\n\n")
            for issue in atb_final["issues"]:
                f.write(f"- {issue}\n")
            if not atb_final["issues"]:
                f.write("- none\n")
        contract = report.get("external_static_contract")
        if contract:
            f.write("\n## External Static Contract\n\n")
            f.write(f"Status: `{'PASS' if contract['passed'] else 'FAIL'}`\n\n")
            if contract["issues"]:
                for issue in contract["issues"]:
                    f.write(f"- {issue}\n")
            else:
                f.write("- none\n")
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
        "--transform-variant",
        choices=["default", "sota-int8", "sota-int8-atb", "sota-int8-atb-v2"],
        default="default",
    )
    parser.add_argument(
        "--kernel-impl", choices=["vectorized", "external-mmul"], default="vectorized"
    )
    parser.add_argument(
        "--external-schedule",
        choices=["baseline", "flat", "manual-unroll", "software-pipeline"],
        default=DEFAULT_EXTERNAL_SCHEDULE,
    )
    parser.add_argument(
        "--external-kernel-style",
        choices=EXTERNAL_KERNEL_STYLES,
        default=DEFAULT_EXTERNAL_KERNEL_STYLE,
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
        "--external-core-m-packs",
        type=positive_int,
        default=DEFAULT_EXTERNAL_CORE_M_PACKS,
    )
    parser.add_argument(
        "--external-active-m-packs",
        type=positive_int,
        default=DEFAULT_EXTERNAL_ACTIVE_M_PACKS,
    )
    parser.add_argument(
        "--external-core-n-packs",
        type=positive_int,
        default=DEFAULT_EXTERNAL_CORE_N_PACKS,
    )
    parser.add_argument(
        "--external-c-stride-m-packs",
        type=positive_int,
        default=DEFAULT_EXTERNAL_CORE_M_PACKS,
    )
    parser.add_argument(
        "--atb-k-chunk-elements",
        type=positive_int,
        default=DEFAULT_ATB_K_CHUNK_ELEMENTS,
    )
    parser.add_argument("--performance-mode", action="store_true")
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
    parser.add_argument("--lowered-ir", type=Path, default=None)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="npu_static")
    args = parser.parse_args()
    if args.m % args.tile_m or args.n % args.tile_n:
        parser.error(
            f"M and N must be multiples of tile sizes {args.tile_m} and {args.tile_n}"
        )
    if args.k % 8:
        parser.error("K must be a multiple of 8")
    args.effective_atb_k_chunk_elements = 0
    if is_atb_v2(args.transform_variant):
        args.effective_atb_k_chunk_elements = choose_atb_k_chunk_elements(
            args.k,
            args.atb_k_chunk_elements,
            args.external_k_packs * 8,
            max_chunk=ATB_V2_MAX_A_L2_CHUNK_ELEMENTS,
        )
    elif uses_full_m_external_k_chunking(args):
        args.effective_atb_k_chunk_elements = choose_atb_k_chunk_elements(
            args.k,
            args.atb_k_chunk_elements,
            args.external_k_packs * 8,
            max_chunk=DEFAULT_ATB_K_CHUNK_ELEMENTS,
        )
    elif is_atb_variant(args.transform_variant):
        args.effective_atb_k_chunk_elements = args.external_k_packs * 8
    return args


def main() -> int:
    args = parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    report = make_report(args)
    report["atb_final_ir_verification"] = verify_atb_final_ir(
        read_text(args.transformed_ir), args, read_text(args.lowered_ir)
    )
    report["external_static_contract"] = external_static_contract(args, report)
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
    contract = report.get("external_static_contract")
    if contract:
        print(f"external_static_contract={'PASS' if contract['passed'] else 'FAIL'}")
        if not contract["passed"]:
            for issue in contract["issues"]:
                print(f"external_static_contract_issue={issue}")
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
