# run.py -*- Python -*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner
from air.compiler.util import run_transform
from air.ir import *
import air.passmanager

DEFAULT_M = 1024
DEFAULT_K = 1024
DEFAULT_N = 1024
DEFAULT_TILE_M = 512
DEFAULT_TILE_N = 256
SOTA_TILE_M = 576
SOTA_TILE_N = 1152
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


def parse_runtime_loop_tiling(value: str) -> list[int]:
    try:
        parsed = [positive_int(item.strip()) for item in value.split(",")]
    except argparse.ArgumentTypeError:
        raise
    if len(parsed) != 2:
        raise argparse.ArgumentTypeError(
            "runtime loop tiling must contain two positive integers"
        )
    return parsed


def validate_shape(m: int, k: int, n: int, tile_m: int, tile_n: int) -> None:
    errors = []
    if m % tile_m:
        errors.append(f"M must be a multiple of tile M ({tile_m})")
    if n % tile_n:
        errors.append(f"N must be a multiple of tile N ({tile_n})")
    if k % 8:
        errors.append("K must be a multiple of 8")
    if tile_m % 8:
        errors.append("tile M must be a multiple of 8")
    if tile_n % 8:
        errors.append("tile N must be a multiple of 8")
    if errors:
        raise ValueError("; ".join(errors))


def render_sota_int8_transform(transform_text: str, k_packs: int = 9) -> str:
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


def render_sota_int8_atb_transform(
    transform_text: str, k_packs: int, active_m_packs: int
) -> str:
    if active_m_packs <= 0 or 18 % active_m_packs:
        raise ValueError("ATB active M packs must be a positive divisor of 18")
    rendered = render_sota_int8_transform(transform_text, k_packs)
    old = """        %tiled_matmul_1, %inner_forall =
          transform.structured.tile_using_forall %matmul_1 tile_sizes [18, 18, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        transform.annotate %inner_forall "compute_forall" : !transform.any_op
        transform.annotate %tiled_matmul_1 "matmul_compute" : !transform.any_op

    // Step 12: Fuse pack operations into the inner parallel loop.
    // Purpose: Ensures each core has its own data packing for independent execution.
        %fused_lhs_l1_pack2, %6 = transform.structured.fuse_into_containing_op %fused_lhs_l1_pack into %inner_forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        %fused_rhs_l1_pack2, %7 = transform.structured.fuse_into_containing_op %fused_rhs_l1_pack into %inner_forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
"""
    new = f"""        %tiled_matmul_1, %inner_forall =
          transform.structured.tile_using_forall %matmul_1 tile_sizes [18, 18, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        transform.annotate %inner_forall "compute_forall" : !transform.any_op

    // Step 12: Fuse pack operations into the inner parallel loop.
    // Purpose: First make core-local A/B producers, then split compute into active-M bands.
        %fused_lhs_l1_pack2, %6 = transform.structured.fuse_into_containing_op %fused_lhs_l1_pack into %inner_forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        %fused_rhs_l1_pack2, %7 = transform.structured.fuse_into_containing_op %fused_rhs_l1_pack into %inner_forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        %active_m_matmul, %active_m_loop =
          transform.structured.tile_using_for %tiled_matmul_1 tile_sizes [{active_m_packs}, 0, 0, 0, 0, 0]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        transform.annotate %active_m_loop "atb_active_m_loop" : !transform.any_op
        transform.annotate %active_m_matmul "matmul_compute" : !transform.any_op
"""
    if rendered.count(old) != 1:
        raise ValueError("expected one ATB Phase 5 transform fragment")
    return rendered.replace(old, new, 1)


def render_external_mmul_transform(
    transform_text: str,
    k_packs: int,
    fuse_l3_l2: bool = True,
    active_m_packs: int | None = None,
) -> str:
    if active_m_packs is None:
        rendered = render_sota_int8_transform(transform_text, k_packs)
    else:
        rendered = render_sota_int8_atb_transform(
            transform_text, k_packs, active_m_packs
        )
    phase8_marker = (
        "    //==========================================================================\n"
        "    // PHASE 8:"
    )
    phase9_marker = (
        "    //==========================================================================\n"
        "    // PHASE 9:"
    )
    split_marker = phase9_marker if fuse_l3_l2 else phase8_marker
    if rendered.count(split_marker) != 1:
        raise ValueError(f"expected one transform marker for {split_marker!r}")
    prefix = rendered.split(split_marker, 1)[0]
    return (
        prefix
        + f"""    //==========================================================================
    // PHASE 9: ROUTE COMPUTE TO EXTERNAL AIE2P MMUL KERNEL
    // Purpose: Keep the existing memory layout and herd mapping while replacing
    // the scalar/vector.contract compute body with a linked Peano mmul kernel.
    //==========================================================================

        %generic_fill = transform.structured.match ops{{["linalg.generic"]}} attributes{{init_fill}} in %arg1 : (!transform.any_op) -> !transform.any_op
        %inner_most_fills, %vec_fill_loops:2 =
          transform.structured.tile_using_for %generic_fill tile_sizes [1, 1]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

        %matmul_external = transform.structured.match ops{{["linalg.generic"]}} attributes{{matmul_compute}} in %arg1 : (!transform.any_op) -> !transform.any_op
        %mmul_name = transform.param.constant "{EXTERNAL_MMUL_FUNCTION}" -> !transform.any_param
        transform.annotate %matmul_external "library_call" = %mmul_name : !transform.any_op, !transform.any_param

    //==========================================================================
    // PHASE 10: CONVERT TO AIE HERDS AND VECTORIZE NON-COMPUTE HERDS
    // Purpose: Map parallel work to the 8x4 logical AIE split. The compute herd
    // stays as a library call so aircc lowers it to the linked AIE2P object.
    //==========================================================================

        %forall1 = transform.structured.match ops{{["scf.forall"]}} attributes{{prologue_forall}} in %arg1 : (!transform.any_op) -> !transform.any_op
        %forall2 = transform.structured.match ops{{["scf.forall"]}} attributes{{compute_forall}} in %arg1 : (!transform.any_op) -> !transform.any_op
        %forall3 = transform.structured.match ops{{["scf.forall"]}} attributes{{epilogue_forall}} in %arg1 : (!transform.any_op) -> !transform.any_op
        %parallel1 = transform.loop.forall_to_parallel %forall1  : (!transform.any_op) -> !transform.any_op
        %herd1 = transform.air.par_to_herd %parallel1 {{first_dim = 1}} : (!transform.any_op) -> !transform.any_op
        transform.annotate %herd1 "prologue_herd" : !transform.any_op
        %parallel2 = transform.loop.forall_to_parallel %forall2  : (!transform.any_op) -> !transform.any_op
        %herd2 = transform.air.par_to_herd %parallel2 {{first_dim = 1}} : (!transform.any_op) -> !transform.any_op
        transform.annotate %herd2 "compute_herd" : !transform.any_op
        %parallel3 = transform.loop.forall_to_parallel %forall3  : (!transform.any_op) -> !transform.any_op
        %herd3 = transform.air.par_to_herd %parallel3 {{first_dim = 1}} : (!transform.any_op) -> !transform.any_op
        transform.annotate %herd3 "epilogue_herd" : !transform.any_op

        %vectorized_herd1 = transform.air.herd_vectorize %herd1 : (!transform.any_op) -> !transform.any_op
        %vectorized_herd3 = transform.air.herd_vectorize %herd3 : (!transform.any_op) -> !transform.any_op

        %func7 = transform.structured.match ops{{["func.func"]}} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func7 {{
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
            transform.apply_patterns.memref.fold_memref_alias_ops
        }} : !transform.any_op
        %func_fold_1 = transform.structured.match ops{{["func.func"]}} in %arg1 : (!transform.any_op) -> !transform.any_op
        %func_folded_1 = transform.air.fold_unit_extent_dims %func_fold_1 : (!transform.any_op) -> !transform.any_op

    transform.yield
  }}
}}
"""
    )


_ATB_ACTIVE_A_PATTERN = re.compile(
    r"""(?P<base_line>^(?P<base_indent>\s*)(?P<base>%[\w]+) = affine\.apply[^\n]*\n)"""
    r"""(?P<src_line>^(?P=base_indent)(?P<src_view>%[\w]+) = memref\.subview (?P<src_mem>%[\w]+)\[(?P=base), 0\] \[144, 144\] \[1, 1\] : (?P<src_mem_type>memref<[^\n]+?>) to (?P<src_view_type>memref<144x144xi8,[^\n]+?>)\n)"""
    r"""(?P<alloc_line>^(?P=base_indent)(?P<full_alloc>%[\w]+) = memref\.alloc\(\) : memref<18x18x8x8xi8, 2>\n)"""
    r"""(?P<pack_line>^(?P=base_indent)linalg\.pack (?P=src_view) .*? -> memref<18x18x8x8xi8, 2>\n)"""
    r"""(?P<between>.*?)"""
    r"""(?P<loop_line>^(?P=base_indent)scf\.for (?P<iv>%[\w]+) = [^\n]* step %c6 \{\n)"""
    r"""(?P<old_subview_line>^(?P<body_indent>\s*)(?P<active_a>%[\w]+) = memref\.subview (?P=full_alloc)\[0, (?P=iv), 0, 0\] \[18, 6, 8, 8\] [^\n]*\n)"""
    r"""(?P<body>.*?)"""
    r"""(?P<loop_close>^(?P=base_indent)\} \{atb_active_m_loop\}\n)"""
    r"""^(?P=base_indent)memref\.dealloc (?P=full_alloc) : memref<18x18x8x8xi8, 2>\n""",
    re.MULTILINE | re.DOTALL,
)


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


def _indent_block(block: str, extra: str) -> str:
    return "".join(
        (extra + line if line.strip() else line) for line in block.splitlines(True)
    )


def _dedent_block(block: str, prefix: str) -> str:
    return "".join(
        (line.removeprefix(prefix) if line.strip() else line)
        for line in block.splitlines(True)
    )


def _replace_alloc_memref_type(
    module_text: str, alloc: str, old_type: str, new_type: str
) -> str:
    module_text = module_text.replace(
        f"{alloc} = memref.alloc() : {old_type}",
        f"{alloc} = memref.alloc() : {new_type}",
    )
    module_text = module_text.replace(f": {old_type} to", f": {new_type} to")
    return module_text


def rewrite_atb_k_chunk_buffers(
    module_text: str,
    tile_m: int,
    tile_n: int,
    problem_k: int,
    k_step: int,
    chunk_elements: int,
) -> str:
    if chunk_elements >= problem_k:
        return module_text

    a_alloc_match = re.search(
        rf"(?P<alloc>%[\w]+) = memref\.alloc\(\) : memref<{tile_m}x{problem_k}xi8, 1 : i32>",
        module_text,
    )
    b_alloc_match = re.search(
        rf"(?P<alloc>%[\w]+) = memref\.alloc\(\) : memref<{problem_k}x{tile_n}xi8, 1 : i32>",
        module_text,
    )
    if not a_alloc_match or not b_alloc_match:
        raise ValueError("expected full-K A/B L2 allocations before ATB chunk rewrite")
    a_alloc = a_alloc_match.group("alloc")
    b_alloc = b_alloc_match.group("alloc")

    old_a_type = f"memref<{tile_m}x{problem_k}xi8, 1 : i32>"
    new_a_type = f"memref<{tile_m}x{chunk_elements}xi8, 1 : i32>"
    old_b_type = f"memref<{problem_k}x{tile_n}xi8, 1 : i32>"
    new_b_type = f"memref<{chunk_elements}x{tile_n}xi8, 1 : i32>"
    module_text = _replace_alloc_memref_type(
        module_text, a_alloc, old_a_type, new_a_type
    )
    module_text = _replace_alloc_memref_type(
        module_text, b_alloc, old_b_type, new_b_type
    )
    module_text = module_text.replace(
        f"memref<{tile_m}x{k_step}xi8, strided<[{problem_k}, 1], offset: ?>, 1 : i32>",
        f"memref<{tile_m}x{k_step}xi8, strided<[{chunk_elements}, 1], offset: ?>, 1 : i32>",
    )
    module_text = module_text.replace(
        f"memref<144x{k_step}xi8, strided<[{problem_k}, 1], offset: ?>, 1 : i32>",
        f"memref<144x{k_step}xi8, strided<[{chunk_elements}, 1], offset: ?>, 1 : i32>",
    )
    module_text = module_text.replace(
        f"memref<48x{k_step}xi8, strided<[{problem_k}, 1], offset: ?>, 1 : i32>",
        f"memref<48x{k_step}xi8, strided<[{chunk_elements}, 1], offset: ?>, 1 : i32>",
    )

    loop_re = re.compile(
        rf"^(?P<indent>\s*)scf\.for (?P<iv>%[\w]+) = %c0 to %c{problem_k} step %c{k_step} \{{\n",
        re.MULTILINE,
    )
    loop_match = loop_re.search(module_text)
    if not loop_match:
        raise ValueError("expected top-level ATB K-step loop before chunk rewrite")

    start = loop_match.start()
    body_start = loop_match.end()
    depth = 1
    pos = body_start
    while depth and pos < len(module_text):
        ch = module_text[pos]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        pos += 1
    if depth:
        raise ValueError("could not find end of top-level ATB K-step loop")
    close_end = pos
    if close_end < len(module_text) and module_text[close_end] == "\n":
        close_end += 1
    loop_text = module_text[loop_match.end() : pos - 1]

    copy_re = re.compile(
        rf"(?P<copy>\s*%[\w]+ = memref\.subview %reinterpret_cast\[0, {re.escape(loop_match.group('iv'))}\] \[{tile_m}, {k_step}\].*?linalg\.copy .*?\n"
        rf"\s*%[\w]+ = memref\.subview %reinterpret_cast_0\[{re.escape(loop_match.group('iv'))}, 0\] \[{k_step}, {tile_n}\].*?linalg\.copy .*?\n)",
        re.DOTALL,
    )
    copy_match = copy_re.match(loop_text)
    if not copy_match:
        raise ValueError("expected A/B L3-to-L2 copy prelude in ATB K loop")

    copy_prelude = copy_match.group("copy")
    inner_body = loop_text[copy_match.end() :]
    iv = loop_match.group("iv")
    chunk_iv = f"{iv}_chunk"
    inner_iv = f"{iv}_inner"
    local_iv = inner_iv
    indent = loop_match.group("indent")
    body_indent = indent + "  "

    copy_prelude = copy_prelude.replace(
        f"[0, {iv}] [{tile_m}, {k_step}]",
        f"[0, {chunk_iv}] [{tile_m}, {chunk_elements}]",
    )
    copy_prelude = copy_prelude.replace(
        f"[{iv}, 0] [{k_step}, {tile_n}]",
        f"[{chunk_iv}, 0] [{chunk_elements}, {tile_n}]",
    )
    copy_prelude = copy_prelude.replace(
        f"{a_alloc}[0, {iv}] [{tile_m}, {k_step}]",
        f"{a_alloc}[0, 0] [{tile_m}, {chunk_elements}]",
    )
    copy_prelude = copy_prelude.replace(
        f"{b_alloc}[{iv}, 0] [{k_step}, {tile_n}]",
        f"{b_alloc}[0, 0] [{chunk_elements}, {tile_n}]",
    )
    copy_prelude = copy_prelude.replace(
        f"{a_alloc}[0, {chunk_iv}] [{tile_m}, {chunk_elements}]",
        f"{a_alloc}[0, 0] [{tile_m}, {chunk_elements}]",
    )
    copy_prelude = copy_prelude.replace(
        f"{b_alloc}[{chunk_iv}, 0] [{chunk_elements}, {tile_n}]",
        f"{b_alloc}[0, 0] [{chunk_elements}, {tile_n}]",
    )
    copy_prelude = copy_prelude.replace(
        f"{tile_m}x{k_step}xi8", f"{tile_m}x{chunk_elements}xi8"
    )
    copy_prelude = copy_prelude.replace(
        f"{k_step}x{tile_n}xi8", f"{chunk_elements}x{tile_n}xi8"
    )
    copy_prelude = copy_prelude.replace(
        f"memref<{tile_m}x{chunk_elements}xi8, strided<[{problem_k}, 1], offset: ?>, 1 : i32>",
        f"memref<{tile_m}x{chunk_elements}xi8, strided<[{chunk_elements}, 1]>, 1 : i32>",
    )
    copy_prelude = copy_prelude.replace(
        f"memref<{tile_m}x{chunk_elements}xi8, strided<[{chunk_elements}, 1], offset: ?>, 1 : i32>",
        f"memref<{tile_m}x{chunk_elements}xi8, strided<[{chunk_elements}, 1]>, 1 : i32>",
    )
    copy_prelude = copy_prelude.replace(
        f"memref<{chunk_elements}x{tile_n}xi8, strided<[{tile_n}, 1], offset: ?>, 1 : i32>",
        f"memref<{chunk_elements}x{tile_n}xi8, strided<[{tile_n}, 1]>, 1 : i32>",
    )

    inner_body = inner_body.replace(f"{a_alloc}[0, {iv}]", f"{a_alloc}[0, {local_iv}]")
    inner_body = inner_body.replace(f"{b_alloc}[{iv}, 0]", f"{b_alloc}[{local_iv}, 0]")

    chunk_const = (
        f"{indent}%c{chunk_elements} = arith.constant {chunk_elements} : index\n"
    )
    if (
        f"%c{chunk_elements} = arith.constant {chunk_elements} : index"
        in module_text[:start]
    ):
        chunk_const = ""
    replacement = (
        chunk_const
        + f"{indent}scf.for {chunk_iv} = %c0 to %c{problem_k} step %c{chunk_elements} {{\n"
        + _indent_block(copy_prelude, "  ")
        + f"{body_indent}scf.for {local_iv} = %c0 to %c{chunk_elements} step %c{k_step} {{\n"
        + _indent_block(inner_body, "  ")
        + f"{body_indent}}}\n"
        + f"{indent}}}\n"
    )
    return module_text[:start] + replacement + module_text[close_end:]


def _find_matching_brace(text: str, open_pos: int) -> int:
    depth = 0
    pos = open_pos
    while pos < len(text):
        ch = text[pos]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return pos + 1
        pos += 1
    raise ValueError("could not find matching brace")


def rewrite_atb_v2_chunk_l2_dma_offsets(
    module_text: str,
    tile_m: int,
    tile_n: int,
    k_step: int,
    chunk_elements: int,
) -> str:
    if chunk_elements <= k_step:
        return module_text

    a_full_type = f"memref<{tile_m}x{chunk_elements}xi8, 1 : i32>"
    a_view_type = (
        f"memref<{tile_m}x{k_step}xi8, "
        f"strided<[{chunk_elements}, 1], offset: ?>, 1 : i32>"
    )
    b_full_type = f"memref<{chunk_elements}x{tile_n}xi8, 1 : i32>"
    b_view_type = (
        f"memref<{k_step}x{tile_n}xi8, " f"strided<[{tile_n}, 1], offset: ?>, 1 : i32>"
    )
    subviews_re = re.compile(
        rf"^(?P<indent>\s*)(?P<a_view>%[\w]+) = memref\.subview "
        rf"(?P<a_alloc>%[\w]+)\[0, (?P<iv>%[\w]+)\] "
        rf"\[{tile_m}, {k_step}\] \[1, 1\] : "
        rf"{re.escape(a_full_type)} to {re.escape(a_view_type)}\n"
        rf"(?P=indent)(?P<b_view>%[\w]+) = memref\.subview "
        rf"(?P<b_alloc>%[\w]+)\[(?P=iv), 0\] "
        rf"\[{k_step}, {tile_n}\] \[1, 1\] : "
        rf"{re.escape(b_full_type)} to {re.escape(b_view_type)}\n",
        re.MULTILINE,
    )

    out = []
    cursor = 0
    rewrite_count = 0
    for match in subviews_re.finditer(module_text):
        out.append(module_text[cursor : match.start()])
        herd_start = module_text.find(f"{match.group('indent')}air.herd", match.end())
        if herd_start < 0:
            raise ValueError("expected ATB v2 compute herd after chunk L2 subviews")
        herd_line_end = module_text.find("\n", herd_start)
        if herd_line_end < 0:
            raise ValueError("expected ATB v2 compute herd line")
        herd_open = module_text.rfind("{", herd_start, herd_line_end)
        if herd_open < 0:
            raise ValueError("expected ATB v2 compute herd body")
        herd_end = _find_matching_brace(module_text, herd_open)
        if herd_end < len(module_text) and module_text[herd_end] == "\n":
            herd_end += 1
        herd_text = module_text[herd_start:herd_end]

        a_view = match.group("a_view")
        b_view = match.group("b_view")
        a_alloc = match.group("a_alloc")
        b_alloc = match.group("b_alloc")
        iv = match.group("iv")
        args_match = re.search(
            rf"args\((?P<a_arg>%[\w]+)={re.escape(a_view)}, "
            rf"(?P<b_arg>%[\w]+)={re.escape(b_view)},",
            herd_text,
        )
        if not args_match:
            raise ValueError("expected ATB v2 compute herd A/B subview arguments")
        a_arg = args_match.group("a_arg")
        b_arg = args_match.group("b_arg")
        k_arg = f"%{iv[1:]}_l2_offset"

        herd_text = herd_text.replace(f"{a_arg}={a_view}", f"{a_arg}={a_alloc}", 1)
        herd_text = herd_text.replace(f"{b_arg}={b_view}", f"{b_arg}={b_alloc}", 1)
        header_end = herd_text.find("\n")
        if header_end < 0:
            raise ValueError("expected single-line ATB v2 compute herd header")
        header = herd_text[:header_end]
        header, header_count = re.subn(
            r"args\((?P<args>[^)]*)\) : (?P<types>.*) attributes",
            rf"args(\g<args>, {k_arg}={iv}) : \g<types>, index attributes",
            header,
            count=1,
        )
        if header_count != 1:
            raise ValueError("expected ATB v2 compute herd argument list")
        herd_text = header + herd_text[header_end:]
        herd_text = herd_text.replace(a_view_type, a_full_type)
        herd_text = herd_text.replace(b_view_type, b_full_type)
        k_pack_arg = f"{k_arg}_pack"
        b_dma_match = re.search(
            rf"^(?P<indent>\s*)air\.dma_memcpy_nd .*{re.escape(b_arg)}\[",
            herd_text,
            re.MULTILINE,
        )
        if not b_dma_match:
            raise ValueError("expected ATB v2 B L2-to-L1 DMA")
        herd_text = (
            herd_text[: b_dma_match.start()]
            + f"{b_dma_match.group('indent')}{k_pack_arg} = "
            + f"affine.apply affine_map<(d0) -> (d0 floordiv 8)>({k_arg})\n"
            + herd_text[b_dma_match.start() :]
        )
        herd_text, b_count = re.subn(
            rf"{re.escape(b_arg)}\[(%c0[\w]*), (%c0[\w]*), (%c0[\w]*), (%[\w]+)\]",
            rf"{b_arg}[\1, {k_pack_arg}, \3, \4]",
            herd_text,
            count=1,
        )
        herd_text, a_count = re.subn(
            rf"{re.escape(a_arg)}\[(%c0[\w]*), (%c0[\w]*), (%[\w]+), (%c0[\w]*)\]",
            rf"{a_arg}[{k_pack_arg}, \2, \3, \4]",
            herd_text,
            count=1,
        )
        if a_count != 1 or b_count != 1:
            raise ValueError(
                f"expected one ATB v2 A and B L2-to-L1 DMA source each, found A={a_count} B={b_count}"
            )

        out.append(herd_text)
        cursor = herd_end
        rewrite_count += 1

    if rewrite_count == 0:
        raise ValueError("expected ATB v2 chunk L2 subviews to rewrite")
    out.append(module_text[cursor:])
    return "".join(out)


def rewrite_atb_v2_compute_k_loop_inside_herd(
    module_text: str,
    k_step: int,
    chunk_elements: int,
) -> str:
    if chunk_elements <= k_step:
        return module_text

    marker = "attributes {compute_herd}"
    marker_pos = module_text.find(marker)
    if marker_pos < 0:
        raise ValueError("expected one ATB v2 compute herd")
    if module_text.find(marker, marker_pos + 1) >= 0:
        raise ValueError("expected exactly one ATB v2 compute herd")

    herd_start = module_text.rfind("\n", 0, marker_pos) + 1
    herd_indent = re.match(r"\s*", module_text[herd_start:]).group(0)
    loop_start = module_text.rfind("\n", 0, herd_start - 1) + 1
    loop_header_end = module_text.find("\n", loop_start)
    loop_header = module_text[loop_start:loop_header_end]
    loop_re = re.compile(
        rf"^(?P<indent>\s*)scf\.for (?P<iv>%[\w]+) = %c0(?:_\w+)? "
        rf"to %c{chunk_elements}(?:_\w+)? step %c{k_step}(?:_\w+)? \{{$"
    )
    loop_match = loop_re.match(loop_header)
    if not loop_match:
        raise ValueError("expected ATB v2 chunk-internal K loop around compute herd")
    loop_indent = loop_match.group("indent")
    loop_iv = loop_match.group("iv")
    if herd_indent != loop_indent + "  ":
        raise ValueError("expected compute herd to be the only op in the K loop")

    loop_open = module_text.rfind("{", loop_start, loop_header_end)
    loop_end = _find_matching_brace(module_text, loop_open)
    if loop_end < len(module_text) and module_text[loop_end] == "\n":
        loop_end += 1
    herd_line_end = module_text.find("\n", herd_start)
    herd_open = module_text.rfind("{", herd_start, herd_line_end)
    herd_end = _find_matching_brace(module_text, herd_open)
    if herd_end < len(module_text) and module_text[herd_end] == "\n":
        herd_end += 1
    if module_text[herd_end:loop_end].strip(" \n") != "}":
        raise ValueError("expected compute herd to be the only op in the K loop")

    herd_text = _dedent_block(module_text[herd_start:herd_end], "  ")
    header_end = herd_text.find("\n")
    if header_end < 0:
        raise ValueError("expected single-line compute herd header")
    header = herd_text[:header_end]
    header_re = re.compile(
        rf"args\((?P<args>.*), (?P<k_arg>%[\w]+)={re.escape(loop_iv)}\) : "
        rf"(?P<types>.*), index attributes"
    )
    header_match = header_re.search(header)
    if not header_match:
        raise ValueError("expected compute herd K-offset scalar argument")
    k_arg = header_match.group("k_arg")
    header = header_re.sub(
        r"args(\g<args>) : \g<types> attributes",
        header,
        count=1,
    )

    body = herd_text[header_end + 1 :]
    body_close = body.rfind(f"\n{loop_indent}}}")
    if body_close < 0:
        raise ValueError("expected compute herd closing brace")
    body_content = body[:body_close]
    body_indent = loop_indent + "  "
    leading = []
    rest = []
    in_leading_constants = True
    for line in body_content.splitlines(True):
        if in_leading_constants and re.match(
            rf"^{re.escape(body_indent)}%[\w]+ = arith\.constant ", line
        ):
            leading.append(line)
            continue
        in_leading_constants = False
        rest.append(line)
    if not rest:
        raise ValueError("expected compute herd body after constants")
    if not any(f"arith.constant {chunk_elements} : index" in line for line in leading):
        leading.insert(
            0,
            f"{body_indent}%c{chunk_elements}_chunk = arith.constant {chunk_elements} : index\n",
        )
    if not any(f"arith.constant {k_step} : index" in line for line in leading):
        leading.append(
            f"{body_indent}%c{k_step}_kstep = arith.constant {k_step} : index\n"
        )

    def _find_const(name: str, value: int) -> str:
        pattern = re.compile(r"(%[\w]+) = arith\.constant " + str(value) + r" : index")
        for line in leading:
            match = pattern.search(line)
            if match:
                return match.group(1)
        raise ValueError(f"expected {name} constant {value} in compute herd")

    step_const = _find_const("step", k_step)
    upper_const = _find_const("upper", chunk_elements)
    zero_const = _find_const("zero", 0)
    rest_text = "".join(rest).replace(k_arg, loop_iv)
    wrapped = (
        "".join(leading)
        + f"{body_indent}scf.for {loop_iv} = {zero_const} to {upper_const} step {step_const} {{\n"
        + _indent_block(rest_text, "  ")
        + f"{body_indent}}}\n"
    )
    new_herd_text = header + "\n" + wrapped + body[body_close:]
    return module_text[:loop_start] + new_herd_text + module_text[loop_end:]


def _make_active_a_pack(
    match: re.Match[str], active_m_packs: int
) -> tuple[str, str, str]:
    active_a = match.group("active_a")
    stem = active_a[1:]
    active_m_elements = active_m_packs * 8
    source_active_type = match.group("src_view_type").replace(
        "144x144xi8", f"{active_m_elements}x144xi8", 1
    )
    body_indent = match.group("body_indent")
    source_mem = match.group("src_mem")
    source_mem_type = match.group("src_mem_type")
    base = match.group("base")
    iv = match.group("iv")
    active_a_type = f"memref<18x{active_m_packs}x8x8xi8, 2>"
    active_pack = (
        f"{body_indent}%{stem}_m_base = affine.apply "
        f"affine_map<(d0, d1) -> (d0 + d1 * 8)>({base}, {iv})\n"
        f"{body_indent}%{stem}_src = memref.subview {source_mem}[%{stem}_m_base, 0] "
        f"[{active_m_elements}, 144] [1, 1] : {source_mem_type} to {source_active_type}\n"
        f"{body_indent}{active_a} = memref.alloc() : {active_a_type}\n"
        f"{body_indent}linalg.pack %{stem}_src outer_dims_perm = [1, 0] "
        f"inner_dims_pos = [0, 1] inner_tiles = [8, 8] into {active_a} : "
        f"{source_active_type} -> {active_a_type}\n"
    )
    return active_a, active_a_type, active_pack


def _rewrite_active_a_input_type(body: str, active_a: str, active_m_packs: int) -> str:
    return re.sub(
        rf"ins\({re.escape(active_a)}, (?P<b>%[\w]+) : "
        rf"memref<18x{active_m_packs}x8x8xi8, strided<\[[^\]]+\], offset: \?>, 2>,",
        rf"ins({active_a}, \g<b> : memref<18x{active_m_packs}x8x8xi8, 2>,",
        body,
    )


def rewrite_atb_active_a_buffers(module_text: str, active_m_packs: int) -> str:
    if active_m_packs != ATB_EXTERNAL_ACTIVE_M_PACKS:
        raise ValueError("ATB active-A rewrite currently expects 6 active M packs")

    def rewrite_match(match: re.Match[str]) -> str:
        active_a, active_a_type, active_pack = _make_active_a_pack(
            match, active_m_packs
        )
        body_indent = match.group("body_indent")
        body = match.group("body")
        active_c = f"%{active_a[1:]}_c"
        c_line_match = re.search(
            rf"^(?P<i>\s*)(?P<c_view>%[\w]+) = memref\.subview .* "
            rf"to (?P<c_type>memref<18x{active_m_packs}x8x8xi8, "
            rf"strided<\[[^\]]+\], offset: \?>, 2>)\n",
            body,
            re.MULTILINE,
        )
        if not c_line_match:
            raise ValueError("expected one ATB active-C subview")
        indent = c_line_match.group("i")
        c_view = c_line_match.group("c_view")
        c_type = c_line_match.group("c_type")
        c_stem = active_c[1:]
        active_c_type = f"memref<18x{active_m_packs}x8x8xi8, 2>"
        c_insert = (
            f"{indent}{active_c} = memref.alloc() : {active_c_type}\n"
            f"{indent}%{c_stem}_c0 = arith.constant 0 : index\n"
            f"{indent}%{c_stem}_c1 = arith.constant 1 : index\n"
            f"{indent}%{c_stem}_c6 = arith.constant 6 : index\n"
            f"{indent}%{c_stem}_c8 = arith.constant 8 : index\n"
            f"{indent}%{c_stem}_c18 = arith.constant 18 : index\n"
            f"{indent}scf.for %{c_stem}_li0 = %{c_stem}_c0 to %{c_stem}_c18 step %{c_stem}_c1 {{\n"
            f"{indent}  scf.for %{c_stem}_li1 = %{c_stem}_c0 to %{c_stem}_c6 step %{c_stem}_c1 {{\n"
            f"{indent}    scf.for %{c_stem}_li2 = %{c_stem}_c0 to %{c_stem}_c8 step %{c_stem}_c1 {{\n"
            f"{indent}      scf.for %{c_stem}_li3 = %{c_stem}_c0 to %{c_stem}_c8 step %{c_stem}_c1 {{\n"
            f"{indent}        %{c_stem}_lv = memref.load {c_view}[%{c_stem}_li0, %{c_stem}_li1, %{c_stem}_li2, %{c_stem}_li3] : {c_type}\n"
            f"{indent}        memref.store %{c_stem}_lv, {active_c}[%{c_stem}_li0, %{c_stem}_li1, %{c_stem}_li2, %{c_stem}_li3] : {active_c_type}\n"
            f"{indent}      }}\n"
            f"{indent}    }}\n"
            f"{indent}  }}\n"
            f"{indent}}}\n"
        )
        body = body[: c_line_match.end()] + c_insert + body[c_line_match.end() :]
        old_outs = f"outs({c_view} : {c_type})"
        new_outs = f"outs({active_c} : {active_c_type})"
        outs_pos = body.find(old_outs)
        if outs_pos < 0:
            raise ValueError("expected ATB active-C linalg outs operand")
        body = body[:outs_pos] + new_outs + body[outs_pos + len(old_outs) :]
        generic_end_marker = f"\n{indent}}}\n"
        generic_end = body.find(generic_end_marker, outs_pos)
        if generic_end < 0:
            raise ValueError("expected ATB linalg.generic closing brace")
        generic_end += len(generic_end_marker)
        c_copy_back = (
            f"{indent}scf.for %{c_stem}_so0 = %{c_stem}_c0 to %{c_stem}_c18 step %{c_stem}_c1 {{\n"
            f"{indent}  scf.for %{c_stem}_so1 = %{c_stem}_c0 to %{c_stem}_c6 step %{c_stem}_c1 {{\n"
            f"{indent}    scf.for %{c_stem}_so2 = %{c_stem}_c0 to %{c_stem}_c8 step %{c_stem}_c1 {{\n"
            f"{indent}      scf.for %{c_stem}_so3 = %{c_stem}_c0 to %{c_stem}_c8 step %{c_stem}_c1 {{\n"
            f"{indent}        %{c_stem}_sv = memref.load {active_c}[%{c_stem}_so0, %{c_stem}_so1, %{c_stem}_so2, %{c_stem}_so3] : {active_c_type}\n"
            f"{indent}        memref.store %{c_stem}_sv, {c_view}[%{c_stem}_so0, %{c_stem}_so1, %{c_stem}_so2, %{c_stem}_so3] : {c_type}\n"
            f"{indent}      }}\n"
            f"{indent}    }}\n"
            f"{indent}  }}\n"
            f"{indent}}}\n"
            f"{indent}memref.dealloc {active_c} : {active_c_type}\n"
        )
        body = body[:generic_end] + c_copy_back + body[generic_end:]
        body = _rewrite_active_a_input_type(body, active_a, active_m_packs)
        return (
            match.group("base_line")
            + match.group("between")
            + match.group("loop_line")
            + active_pack
            + body
            + f"{body_indent}memref.dealloc {active_a} : {active_a_type}\n"
            + match.group("loop_close")
        )

    rewritten, rewrite_count = _ATB_ACTIVE_A_PATTERN.subn(rewrite_match, module_text)
    if rewrite_count != 1:
        raise ValueError(
            f"expected one ATB active-A rewrite opportunity, found {rewrite_count}"
        )
    return rewritten


def rewrite_atb_v2_active_a_buffers(
    module_text: str,
    active_m_packs: int,
    external_kernel_object: str | None,
    use_static_c_offset: bool,
    external_c_stride_m_packs: int,
) -> str:
    if active_m_packs != ATB_EXTERNAL_ACTIVE_M_PACKS:
        raise ValueError("ATB v2 active-A rewrite currently expects 6 active M packs")

    a_l2_alloc_re = re.compile(
        rf"(?P<prefix>%[\w]+ = memref\.alloc\(\)) : "
        rf"(?P<type>memref<{SOTA_TILE_M}x\d+xi8, 1 : i32>)"
    )
    module_text, a_l2_marked = a_l2_alloc_re.subn(
        r"\g<prefix> {air.shrinkage = false} : \g<type>",
        module_text,
        count=1,
    )
    if a_l2_marked != 1:
        raise ValueError("expected one ATB v2 A L2 allocation to mark no-shrink")

    def rewrite_match(match: re.Match[str]) -> str:
        active_a, active_a_type, active_pack = _make_active_a_pack(
            match, active_m_packs
        )
        body_indent = match.group("body_indent")
        body = match.group("body")
        c_line_match = re.search(
            rf"^(?P<i>\s*)(?P<c_view>%[\w]+) = memref\.subview (?P<c_mem>%[\w]+)"
            rf"\[(?P<n_off>%[\w]+), (?P<m_off>%[\w]+), 0, 0\] "
            rf"\[18, {active_m_packs}, 8, 8\] \[1, 1, 1, 1\] : "
            rf"(?P<c_mem_type>memref<[^\n]+?>) to (?P<c_type>memref<18x{active_m_packs}x8x8xi8, "
            rf"strided<\[[^\]]+\], offset: \?>, 2>)\n",
            body,
            re.MULTILINE,
        )
        if not c_line_match:
            raise ValueError("expected one ATB v2 active-C subview")
        iv = match.group("iv")
        n_off = c_line_match.group("n_off")
        m_off = c_line_match.group("m_off")
        n_def = re.search(
            rf"^\s*{re.escape(n_off)} = affine\.apply [^\n]*\[?(?P<coord>%[\w]+)\]?\n",
            body,
            re.MULTILINE,
        )
        m_def = re.search(
            rf"^\s*{re.escape(m_off)} = affine\.apply [^\n]*[\[(]{re.escape(iv)}, (?P<coord>%[\w]+)[\])]\n",
            body,
            re.MULTILINE,
        )
        if not n_def or not m_def:
            raise ValueError("expected ATB v2 C offset affine definitions")
        n_coord = n_def.group("coord")
        m_coord = m_def.group("coord")
        c_mem = c_line_match.group("c_mem")
        c_mem_type = c_line_match.group("c_mem_type")
        c_view = c_line_match.group("c_view")
        c_type = c_line_match.group("c_type")
        full_c = f"%{active_a[1:]}_c_full"
        local_c = f"%{active_a[1:]}_c_local"
        full_c_n = f"%{active_a[1:]}_c_n"
        full_c_m = f"%{active_a[1:]}_c_m"
        full_c_type = c_type.replace(f"18x{active_m_packs}x8x8xi8", "18x18x8x8xi8", 1)
        local_c_type = "memref<18x18x8x8xi8, 2>"
        full_c_n_stride = DEFAULT_EXTERNAL_CORE_M_PACKS * 8 * 8
        full_c_insert = (
            f"{body_indent}{full_c_n} = affine.apply affine_map<(d0) -> (d0 * 18)>({n_coord})\n"
            f"{body_indent}{full_c_m} = affine.apply affine_map<(d0) -> (d0 * 18)>({m_coord})\n"
            f"{body_indent}{full_c} = memref.subview {c_mem}[{full_c_n}, {full_c_m}, 0, 0] "
            f"[18, 18, 8, 8] [1, 1, 1, 1] : {c_mem_type} to {full_c_type}\n"
            f"{body_indent}{local_c} = memref.reinterpret_cast {full_c} to "
            f"offset: [0], sizes: [18, 18, 8, 8], "
            f"strides: [{full_c_n_stride}, 64, 8, 1] : {full_c_type} to {local_c_type}\n"
        )
        if use_static_c_offset:
            c_routing_insert = full_c_insert
            body = body[: c_line_match.start()] + body[c_line_match.end() :]
        else:
            local_c_base = f"%{active_a[1:]}_c_base"
            local_c_base_type = c_type.replace(
                f"18x{active_m_packs}x8x8xi8", "18x1x8x8xi8", 1
            )
            explicit_c_insert = (
                f"{body_indent}{local_c_base} = memref.subview {c_mem}"
                f"[{n_off}, {m_off}, 0, 0] [18, 1, 8, 8] [1, 1, 1, 1] : "
                f"{c_mem_type} to {local_c_base_type}\n"
                f"{body_indent}{local_c} = memref.reinterpret_cast {local_c_base} to "
                f"offset: [0], sizes: [18, 18, 8, 8], "
                f"strides: [{full_c_n_stride}, 64, 8, 1] : "
                f"{local_c_base_type} to {local_c_type}\n"
            )
            c_routing_insert = ""
            body = (
                body[: c_line_match.start()]
                + explicit_c_insert
                + body[c_line_match.end() :]
            )
        body = _rewrite_active_a_input_type(body, active_a, active_m_packs)
        old_outs = f"outs({c_view} : {c_type})"
        outs_pos = body.find(old_outs)
        if outs_pos < 0:
            raise ValueError("expected ATB v2 active-C linalg outs operand")
        generic_start = body.rfind("\n", 0, outs_pos) + 1
        generic_indent = re.match(r"\s*", body[generic_start:]).group(0)
        generic_end_marker = f"\n{generic_indent}}}\n"
        generic_end = body.find(generic_end_marker, outs_pos)
        if generic_end < 0:
            raise ValueError("expected ATB v2 linalg.generic closing brace")
        generic_end += len(generic_end_marker)
        generic_text = body[generic_start:generic_end]
        ins_match = re.search(
            rf"ins\({re.escape(active_a)}, (?P<b>%[\w]+) : "
            rf"{re.escape(active_a_type)}, (?P<b_type>memref<18x18x8x8xi8, 2>)\)",
            generic_text,
        )
        if not ins_match:
            raise ValueError("expected ATB v2 linalg.generic A/B operands")
        b_view = ins_match.group("b")
        b_type = ins_match.group("b_type")
        call = (
            f"{generic_indent}func.call @{EXTERNAL_MMUL_FUNCTION}({active_a}, {b_view}, {local_c}) "
            f": ({active_a_type}, {b_type}, {local_c_type}) -> ()\n"
        )
        body = body[:generic_start] + call + body[generic_end:]
        return (
            match.group("base_line")
            + match.group("between")
            + c_routing_insert
            + match.group("loop_line")
            + active_pack
            + body
            + f"{body_indent}memref.dealloc {active_a} : {active_a_type}\n"
            + match.group("loop_close")
        )

    rewritten, rewrite_count = _ATB_ACTIVE_A_PATTERN.subn(rewrite_match, module_text)
    if rewrite_count != 1:
        raise ValueError(
            f"expected one ATB v2 active-A rewrite opportunity, found {rewrite_count}"
        )
    if (
        f"func.call @{EXTERNAL_MMUL_FUNCTION}" not in rewritten
        or "memref<18x18x8x8xi8" not in rewritten
    ):
        raise ValueError("ATB v2 rewrite did not route the external call to full C")
    link_attr = (
        f"link_with = {json.dumps(str(external_kernel_object))}, "
        if external_kernel_object
        else ""
    )
    declaration = (
        f"  func.func private @{EXTERNAL_MMUL_FUNCTION}("
        f"memref<18x{active_m_packs}x8x8xi8, 2>, "
        f"memref<18x18x8x8xi8, 2>, memref<18x18x8x8xi8, 2>) "
        f"attributes {{{link_attr}llvm.emit_c_interface}}\n"
    )
    if f"func.func private @{EXTERNAL_MMUL_FUNCTION}" not in rewritten:
        rewritten = rewritten.replace("module {\n", "module {\n" + declaration, 1)
    return rewritten


def render_transform_variant(
    transform_text: str,
    variant: str,
    kernel_impl: str,
    external_k_packs: int,
    problem_k: int,
    external_active_m_packs: int,
) -> str:
    if kernel_impl == "external-mmul":
        if variant not in ("sota-int8",) + ATB_TRANSFORM_VARIANTS:
            raise ValueError(
                "external-mmul requires --transform-variant=sota-int8, "
                "sota-int8-atb, or sota-int8-atb-v2"
            )
        fuse_l3_l2 = problem_k > external_k_packs * 8
        active_m_packs = external_active_m_packs if is_atb_variant(variant) else None
        return render_external_mmul_transform(
            transform_text,
            external_k_packs,
            fuse_l3_l2=fuse_l3_l2,
            active_m_packs=active_m_packs,
        )
    if variant == "default":
        return transform_text
    if variant == "sota-int8":
        return render_sota_int8_transform(transform_text)
    if is_atb_variant(variant):
        raise ValueError(f"{variant} requires --kernel-impl=external-mmul")
    raise ValueError(f"unknown transform variant: {variant}")


def build_matmul_ir(
    m: int,
    k: int,
    n: int,
    tile_m: int,
    tile_n: int,
    output_type: str,
    b_layout: str,
) -> str:
    out_mlir = "i32" if output_type == "int32" else "i8"
    zero = "%c0_i32 = arith.constant 0 : i32"
    zero_value = "%c0_i32"
    if output_type == "int8":
        zero = "%c0_i8 = arith.constant 0 : i8"
        zero_value = "%c0_i8"

    if b_layout == "row":
        b_offset_code = ""
        b_offset_value = "%n_offset"
        b_strides = f"[{n}, 1]"
        b_tensor_type = f"tensor<{k}x{tile_n}xi8>"
        matmul_op = "linalg.matmul"
        b_tensor_code = f"""    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [{b_offset_value}], sizes: [{k}, {tile_n}], strides: {b_strides} : memref<*xi8> to memref<{k}x{tile_n}xi8, strided<{b_strides}, offset: ?>>
    %alloc_1 = memref.alloc() : memref<{k}x{tile_n}xi8>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<{k}x{tile_n}xi8, strided<{b_strides}, offset: ?>> to memref<{k}x{tile_n}xi8>
    %b_tensor = bufferization.to_tensor %alloc_1 restrict writable : memref<{k}x{tile_n}xi8> to tensor<{k}x{tile_n}xi8>"""
    else:
        # Physical B is packed by N tile: [N/tile_N, K, tile_N + 4]. The
        # padded pitch prevents AIR-to-AIE channel lowering from collapsing the
        # selected B tile into a default-contiguous view and dropping the
        # N-tile base offset.
        b_pitch = tile_n + 4
        b_tile_span = k * b_pitch
        b_offset_code = f"""%n_tile_index = arith.index_cast %arg7 : i32 to index
    %cBTileSpan = arith.constant {b_tile_span} : index
    %b_offset = arith.muli %n_tile_index, %cBTileSpan : index"""
        b_offset_value = "%b_offset"
        b_strides = f"[{b_pitch}, 1]"
        b_tensor_type = f"tensor<{k}x{tile_n}xi8>"
        matmul_op = "linalg.matmul"
        b_tensor_code = f"""    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [{b_offset_value}], sizes: [{k}, {tile_n}], strides: {b_strides} : memref<*xi8> to memref<{k}x{tile_n}xi8, strided<{b_strides}, offset: ?>>
    %alloc_1 = memref.alloc() : memref<{k}x{tile_n}xi8>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<{k}x{tile_n}xi8, strided<{b_strides}, offset: ?>> to memref<{k}x{tile_n}xi8>
    %b_tensor = bufferization.to_tensor %alloc_1 restrict writable : memref<{k}x{tile_n}xi8> to tensor<{k}x{tile_n}xi8>"""

    return f"""// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

module {{
  func.func @bare_matmul(%arg0: memref<*xi8> {{tt.divisibility = 16 : i32}}, %arg1: memref<*xi8> {{tt.divisibility = 16 : i32}}, %arg2: memref<*x{out_mlir}> {{tt.divisibility = 16 : i32}}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {{
    {zero}
    %cK = arith.constant {k} : index
    %cN = arith.constant {n} : index
    %cTileN_i32 = arith.constant {tile_n} : i32
    %cTileM_i32 = arith.constant {tile_m} : i32
    %m_tile_i32 = arith.muli %arg6, %cTileM_i32 : i32
    %m_offset = arith.index_cast %m_tile_i32 : i32 to index
    %n_tile_i32 = arith.muli %arg7, %cTileN_i32 : i32
    %n_offset = arith.index_cast %n_tile_i32 : i32 to index
    %a_offset = arith.muli %m_offset, %cK : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%a_offset], sizes: [{tile_m}, {k}], strides: [{k}, 1] : memref<*xi8> to memref<{tile_m}x{k}xi8, strided<[{k}, 1], offset: ?>>
    %alloc = memref.alloc() : memref<{tile_m}x{k}xi8>
    memref.copy %reinterpret_cast, %alloc : memref<{tile_m}x{k}xi8, strided<[{k}, 1], offset: ?>> to memref<{tile_m}x{k}xi8>
    %a_tensor = bufferization.to_tensor %alloc restrict writable : memref<{tile_m}x{k}xi8> to tensor<{tile_m}x{k}xi8>
    {b_offset_code}
{b_tensor_code}
    %empty = tensor.empty() : tensor<{tile_m}x{tile_n}x{out_mlir}>
    %filled = linalg.fill ins({zero_value} : {out_mlir}) outs(%empty : tensor<{tile_m}x{tile_n}x{out_mlir}>) -> tensor<{tile_m}x{tile_n}x{out_mlir}>
    %matmul = {matmul_op} ins(%a_tensor, %b_tensor : tensor<{tile_m}x{k}xi8>, {b_tensor_type}) outs(%filled : tensor<{tile_m}x{tile_n}x{out_mlir}>) -> tensor<{tile_m}x{tile_n}x{out_mlir}>
    %c_row_offset = arith.muli %m_offset, %cN : index
    %c_offset = arith.addi %c_row_offset, %n_offset : index
    %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [%c_offset], sizes: [{tile_m}, {tile_n}], strides: [{n}, 1] : memref<*x{out_mlir}> to memref<{tile_m}x{tile_n}x{out_mlir}, strided<[{n}, 1], offset: ?>>
    bufferization.materialize_in_destination %matmul in writable %reinterpret_cast_2 : (tensor<{tile_m}x{tile_n}x{out_mlir}>, memref<{tile_m}x{tile_n}x{out_mlir}, strided<[{n}, 1], offset: ?>>) -> ()
    return
  }}
}}
"""


def copy_aircc_lowered_ir(artifact_dir: Path | None) -> None:
    if artifact_dir is None:
        return
    debug_dir = Path("air_project") / "debug_ir"
    candidates = sorted(debug_dir.glob("*_after_air-verify-hierarchy-locality.mlir"))
    if not candidates:
        return
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "air_module.lowered.mlir").write_text(
        candidates[-1].read_text(encoding="utf-8"), encoding="utf-8"
    )


def write_compile_config(args: argparse.Namespace, generated_ir: str | None) -> None:
    if not args.artifact_dir:
        return
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "m": args.m,
        "k": args.k,
        "n": args.n,
        "tile_m": args.tile_m,
        "tile_n": args.tile_n,
        "output_type": args.output_type,
        "b_layout": args.b_layout,
        "runtime_loop_tiling_sizes": args.runtime_loop_tiling_sizes,
        "transform_script": args.transform_script,
        "transform_variant": args.transform_variant,
        "kernel_impl": args.kernel_impl,
        "external_kernel_object": args.external_kernel_object,
        "external_schedule": args.external_schedule,
        "external_kernel_style": args.external_kernel_style,
        "external_k_packs": args.external_k_packs,
        "external_block_m": args.external_block_m,
        "external_block_n": args.external_block_n,
        "external_core_m_packs": args.external_core_m_packs,
        "external_active_m_packs": args.external_active_m_packs,
        "external_core_n_packs": args.external_core_n_packs,
        "external_c_stride_m_packs": args.external_c_stride_m_packs,
        "external_atb_c_offset": args.external_atb_c_offset,
        "atb_k_chunk_elements": args.atb_k_chunk_elements,
        "effective_k_chunk_elements": args.effective_atb_k_chunk_elements,
        "effective_atb_k_chunk_elements": args.effective_atb_k_chunk_elements,
        "omit_ping_pong": args.omit_ping_pong,
        "aircc_debug_ir": args.aircc_debug_ir,
        "input_ir": args.input_ir or "generated",
        "output_format": args.output_format,
        "target_device": args.target_device,
        "trace_size": args.trace_size,
        "trace_offset": args.trace_offset,
    }
    (artifact_dir / "compile_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if generated_ir is not None:
        (artifact_dir / "generated_input.mlir").write_text(
            generated_ir, encoding="utf-8"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the Strix int8 matmul example",
    )
    parser.add_argument("--input-ir", default=None, help="Optional input IR file path")
    parser.add_argument(
        "--transform-script",
        default="transform.mlir",
        help="Transform script path",
    )
    parser.add_argument("-M", "--m", type=positive_int, default=DEFAULT_M)
    parser.add_argument("-K", "--k", type=positive_int, default=DEFAULT_K)
    parser.add_argument("-N", "--n", type=positive_int, default=DEFAULT_N)
    parser.add_argument("--tile-m", type=positive_int, default=DEFAULT_TILE_M)
    parser.add_argument("--tile-n", type=positive_int, default=DEFAULT_TILE_N)
    parser.add_argument(
        "--output-type",
        choices=["int32", "int8"],
        default="int32",
        help="Output element type generated for C (default: int32)",
    )
    parser.add_argument(
        "--b-layout",
        choices=["row", "column"],
        default="row",
        help="Host/device layout expected for B (default: row)",
    )
    parser.add_argument(
        "--transform-variant",
        choices=["default", "sota-int8", "sota-int8-atb", "sota-int8-atb-v2"],
        default="default",
        help="Optional transform rewrite applied after loading --transform-script",
    )
    parser.add_argument(
        "--kernel-impl",
        choices=["vectorized", "external-mmul"],
        default="vectorized",
        help="Compute implementation selected after tiling (default: vectorized)",
    )
    parser.add_argument(
        "--external-kernel-object",
        default=None,
        help="Object file linked when --kernel-impl=external-mmul",
    )
    parser.add_argument(
        "--external-schedule",
        choices=["baseline", "flat", "manual-unroll", "software-pipeline"],
        default=DEFAULT_EXTERNAL_SCHEDULE,
        help="Peano schedule annotation mode for --kernel-impl=external-mmul",
    )
    parser.add_argument(
        "--external-kernel-style",
        choices=EXTERNAL_KERNEL_STYLES,
        default=DEFAULT_EXTERNAL_KERNEL_STYLE,
        help="External kernel source style selected at compile time",
    )
    parser.add_argument(
        "--external-k-packs",
        type=positive_int,
        default=DEFAULT_EXTERNAL_K_PACKS,
        help="Packed K tiles consumed by the external mmul kernel (default: 9)",
    )
    parser.add_argument(
        "--external-block-m",
        type=positive_int,
        default=DEFAULT_EXTERNAL_BLOCK_M,
        help="External mmul register-block packs along M (default: 2)",
    )
    parser.add_argument(
        "--external-block-n",
        type=positive_int,
        default=DEFAULT_EXTERNAL_BLOCK_N,
        help="External mmul register-block packs along N (default: 2)",
    )
    parser.add_argument(
        "--external-core-m-packs",
        type=positive_int,
        default=DEFAULT_EXTERNAL_CORE_M_PACKS,
        help="Full per-core C M packs used for the external kernel C stride",
    )
    parser.add_argument(
        "--external-active-m-packs",
        type=positive_int,
        default=DEFAULT_EXTERNAL_ACTIVE_M_PACKS,
        help="Active M packs consumed by each external kernel call",
    )
    parser.add_argument(
        "--external-core-n-packs",
        type=positive_int,
        default=DEFAULT_EXTERNAL_CORE_N_PACKS,
        help="Full per-core C/B N packs consumed by the external kernel",
    )
    parser.add_argument(
        "--external-c-stride-m-packs",
        type=positive_int,
        default=DEFAULT_EXTERNAL_CORE_M_PACKS,
        help="C stride in M packs used by the external kernel",
    )
    parser.add_argument(
        "--atb-k-chunk-elements",
        type=positive_int,
        default=DEFAULT_ATB_K_CHUNK_ELEMENTS,
        help="Maximum K elements per ATB v2 L2 chunk before lowering",
    )
    parser.add_argument(
        "--external-atb-c-offset",
        type=int,
        choices=[0, 1],
        default=0,
        help="Use the external kernel static ATB active-M C offset counter",
    )
    parser.add_argument(
        "--omit-ping-pong",
        choices=["L1", "L2", "all", "L1-partial-a", "L1-partial-b"],
        default=None,
        help="Forward --omit-ping-pong-transform to aircc for residency experiments",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Only compile without running validation",
    )
    parser.add_argument(
        "--aircc-debug-ir",
        action="store_true",
        help="Forward --debug-ir to aircc and keep pass-by-pass IR under the build directory",
    )
    parser.add_argument(
        "--output-format",
        choices=["elf", "xclbin"],
        default="xclbin",
        help="Output format: xclbin (default) or elf",
    )
    parser.add_argument(
        "--debug-ir",
        default=None,
        metavar="OUTPUT_FILE",
        help="Print the transformed IR to the specified file and exit",
    )
    parser.add_argument(
        "--runtime-loop-tiling-sizes",
        default="2,4",
        metavar="M,N",
        help="Comma-separated AIR runtime loop tiling sizes (default: 2,4)",
    )
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="Directory for compile metadata and generated input IR",
    )
    parser.add_argument(
        "--trace-size",
        type=nonnegative_int,
        default=0,
        help="Trace buffer size in bytes (0 disables trace plumbing)",
    )
    parser.add_argument(
        "--trace-offset",
        type=nonnegative_int,
        default=0,
        help="Trace buffer offset in bytes",
    )
    parser.add_argument(
        "--trace-file",
        default="trace_data.txt",
        help="Trace output file used by XRTRunner validation mode",
    )
    parser.add_argument(
        "--target-device",
        default="npu2",
        help="XRTBackend target device (default: npu2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic RNG seed for validation inputs",
    )
    args = parser.parse_args()
    args.runtime_loop_tiling_sizes = parse_runtime_loop_tiling(
        args.runtime_loop_tiling_sizes
    )
    validate_shape(args.m, args.k, args.n, args.tile_m, args.tile_n)
    if args.kernel_impl == "external-mmul":
        errors = []
        if args.transform_variant not in ("sota-int8",) + ATB_TRANSFORM_VARIANTS:
            errors.append(
                "external-mmul requires --transform-variant=sota-int8, "
                "sota-int8-atb, or sota-int8-atb-v2"
            )
        if args.output_type != "int8":
            errors.append("external-mmul currently supports --output-type=int8 only")
        if args.b_layout != "column":
            errors.append("external-mmul requires --b-layout=column")
        if args.tile_m != SOTA_TILE_M or args.tile_n != SOTA_TILE_N:
            errors.append(
                f"external-mmul requires tile shape {SOTA_TILE_M}x{SOTA_TILE_N}"
            )
        k_residency = args.external_k_packs * 8
        block_shape = (args.external_block_m, args.external_block_n)
        if args.external_block_m not in (2, 3, 4) or args.external_block_n not in (
            2,
            3,
        ):
            errors.append(
                "external-mmul supports EXTERNAL_BLOCK_M=2|3|4 and EXTERNAL_BLOCK_N=2|3"
            )
        if block_shape == (4, 3):
            errors.append(
                "external-mmul does not support EXTERNAL_BLOCK_M=4 with EXTERNAL_BLOCK_N=3"
            )
        if args.external_active_m_packs > args.external_core_m_packs:
            errors.append("EXTERNAL_ACTIVE_M_PACKS must be <= EXTERNAL_CORE_M_PACKS")
        if args.external_core_m_packs % args.external_active_m_packs:
            errors.append("EXTERNAL_ACTIVE_M_PACKS must divide EXTERNAL_CORE_M_PACKS")
        if args.external_active_m_packs % args.external_block_m:
            errors.append(
                "EXTERNAL_ACTIVE_M_PACKS must be divisible by EXTERNAL_BLOCK_M"
            )
        if args.external_core_n_packs % args.external_block_n:
            errors.append("EXTERNAL_CORE_N_PACKS must be divisible by EXTERNAL_BLOCK_N")
        if args.external_c_stride_m_packs < args.external_active_m_packs:
            errors.append("EXTERNAL_C_STRIDE_M_PACKS must cover active M packs")
        if args.external_k_packs > 18:
            errors.append("external-mmul supports at most 18 packed K tiles")
        if (
            args.external_k_packs > DEFAULT_EXTERNAL_K_PACKS
            and args.omit_ping_pong
            not in (
                "L1",
                "all",
                "L1-partial-a",
                "L1-partial-b",
            )
        ):
            errors.append(
                "external-mmul K residency above 9 packs exceeds L1 with "
                "fully ping-ponged A/B buffers; pass --omit-ping-pong L1, "
                "all, L1-partial-a, or L1-partial-b"
            )
        if args.external_kernel_style == "native-mmul-atb-ref" and not is_atb_v2(
            args.transform_variant
        ):
            errors.append(
                "EXTERNAL_KERNEL_STYLE=native-mmul-atb-ref is only valid with sota-int8-atb-v2"
            )
        if (
            args.external_kernel_style == "native-mmul-atb-ref"
            and args.external_atb_c_offset != 1
        ):
            errors.append(
                "EXTERNAL_KERNEL_STYLE=native-mmul-atb-ref requires --external-atb-c-offset=1"
            )
        if is_atb_variant(args.transform_variant):
            if args.transform_variant == "sota-int8-atb":
                if args.external_kernel_style != "native-mmul":
                    errors.append(
                        "sota-int8-atb requires EXTERNAL_KERNEL_STYLE=native-mmul"
                    )
            elif args.external_kernel_style not in ATB_V2_EXTERNAL_KERNEL_STYLES:
                errors.append(
                    "sota-int8-atb-v2 requires EXTERNAL_KERNEL_STYLE=native-mmul "
                    "or native-mmul-atb-ref"
                )
            if block_shape != (2, 2):
                errors.append(
                    f"{args.transform_variant} requires EXTERNAL_BLOCK_M=2 and EXTERNAL_BLOCK_N=2"
                )
            if args.external_k_packs != ATB_EXTERNAL_K_PACKS:
                errors.append(f"{args.transform_variant} requires EXTERNAL_K_PACKS=18")
            if args.external_core_m_packs != DEFAULT_EXTERNAL_CORE_M_PACKS:
                errors.append(
                    f"{args.transform_variant} requires EXTERNAL_CORE_M_PACKS=18"
                )
            if args.external_active_m_packs != ATB_EXTERNAL_ACTIVE_M_PACKS:
                errors.append(
                    f"{args.transform_variant} requires EXTERNAL_ACTIVE_M_PACKS=6"
                )
            if args.external_core_n_packs != DEFAULT_EXTERNAL_CORE_N_PACKS:
                errors.append(
                    f"{args.transform_variant} requires EXTERNAL_CORE_N_PACKS=18"
                )
            if args.omit_ping_pong != "L1-partial-a":
                errors.append(
                    f"{args.transform_variant} requires --omit-ping-pong L1-partial-a"
                )
            if (
                args.transform_variant == "sota-int8-atb"
                and args.external_c_stride_m_packs != ATB_EXTERNAL_ACTIVE_M_PACKS
            ):
                errors.append(
                    "sota-int8-atb keeps baseline EXTERNAL_C_STRIDE_M_PACKS=6"
                )
            if (
                is_atb_v2(args.transform_variant)
                and args.external_c_stride_m_packs != DEFAULT_EXTERNAL_CORE_M_PACKS
            ):
                errors.append("sota-int8-atb-v2 requires EXTERNAL_C_STRIDE_M_PACKS=18")
        if args.k % k_residency:
            errors.append(f"external-mmul requires K to be a multiple of {k_residency}")
        if not args.external_kernel_object:
            errors.append("external-mmul requires --external-kernel-object")
        if errors:
            raise ValueError("; ".join(errors))
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
    return args


args = parse_args()

with air.ir.Context() as ctx, Location.unknown():
    if args.input_ir:
        air_tiled_ir_string = Path(args.input_ir).read_text(encoding="utf-8")
        generated_ir = None
    else:
        air_tiled_ir_string = build_matmul_ir(
            args.m,
            args.k,
            args.n,
            args.tile_m,
            args.tile_n,
            args.output_type,
            args.b_layout,
        )
        generated_ir = air_tiled_ir_string
    write_compile_config(args, generated_ir)

    air_module = Module.parse(air_tiled_ir_string)

    pipeline = (
        "builtin.module(air-override-memref-memory-space{scope=func memory-space=1})"
    )
    pm = air.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)

    transform_ir_string = render_transform_variant(
        Path(args.transform_script).read_text(encoding="utf-8"),
        args.transform_variant,
        args.kernel_impl,
        args.external_k_packs,
        args.k,
        args.external_active_m_packs,
    )
    if args.artifact_dir:
        artifact_dir = Path(args.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "effective_transform.mlir").write_text(
            transform_ir_string, encoding="utf-8"
        )
    transform_ir = Module.parse(transform_ir_string)
    run_transform(transform_ir, air_module)
    if uses_full_m_external_k_chunking(args) and args.effective_atb_k_chunk_elements:
        air_module = Module.parse(
            rewrite_atb_k_chunk_buffers(
                str(air_module),
                args.tile_m,
                args.tile_n,
                args.k,
                args.external_k_packs * 8,
                args.effective_atb_k_chunk_elements,
            )
        )
    elif args.transform_variant == "sota-int8-atb":
        air_module = Module.parse(
            rewrite_atb_active_a_buffers(str(air_module), args.external_active_m_packs)
        )
    elif is_atb_v2(args.transform_variant):
        atb_module_text = rewrite_atb_k_chunk_buffers(
            str(air_module),
            args.tile_m,
            args.tile_n,
            args.k,
            args.external_k_packs * 8,
            args.effective_atb_k_chunk_elements,
        )
        atb_module_text = rewrite_atb_v2_active_a_buffers(
            atb_module_text,
            args.external_active_m_packs,
            args.external_kernel_object,
            bool(args.external_atb_c_offset),
            args.external_c_stride_m_packs,
        )
        air_module = Module.parse(atb_module_text)

    if args.debug_ir:
        output_file = Path(args.debug_ir)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(str(air_module), encoding="utf-8")
        print(f"Transformed IR written to {output_file}")
        raise SystemExit(0)

    input_size = (args.m, args.n, args.k)
    tile_size = (args.tile_m, args.tile_n, args.k)
    launch_size = tuple(i // t for i, t in zip(input_size, tile_size))

    pipeline = (
        "builtin.module("
        + ",".join(
            [
                f"func.func(air-wrap-func-with-parallel{{loop-bounds={launch_size[0]},{launch_size[1]},{launch_size[2]}}})",
                "air-par-to-launch{depth=0 has-air-segment=true}",
                "canonicalize",
                "cse",
                "air-copy-to-dma",
            ]
        )
        + ")"
    )
    pm = air.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)

    if (
        is_atb_v2(args.transform_variant)
        and args.k > args.effective_atb_k_chunk_elements
    ):
        atb_module_text = rewrite_atb_v2_chunk_l2_dma_offsets(
            str(air_module),
            args.tile_m,
            args.tile_n,
            args.external_k_packs * 8,
            args.effective_atb_k_chunk_elements,
        )
        atb_module_text = rewrite_atb_v2_compute_k_loop_inside_herd(
            atb_module_text,
            args.external_k_packs * 8,
            args.effective_atb_k_chunk_elements,
        )
        air_module = Module.parse(atb_module_text)

    if args.artifact_dir:
        artifact_dir = Path(args.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "air_module.final.mlir").write_text(
            str(air_module), encoding="utf-8"
        )

    output_ext = "elf" if args.output_format == "elf" else "xclbin"
    backend_kwargs = dict(
        omit_while_true_loop=False,
        output_format=args.output_format,
        instance_name="bare_matmul",
        runtime_loop_tiling_sizes=args.runtime_loop_tiling_sizes,
        trace_offset=args.trace_offset,
        trace_size=args.trace_size,
        target_device=args.target_device,
        debug_ir=args.aircc_debug_ir,
    )
    if args.kernel_impl == "external-mmul":
        backend_kwargs["lower_linalg_to_func"] = args.external_kernel_object
    if args.omit_ping_pong:
        backend_kwargs["omit_pingpong"] = args.omit_ping_pong

    if args.compile_only:
        print(f"Compile-only mode: generating {output_ext} binary...")
        backend = XRTBackend(**backend_kwargs)
        backend.compile(air_module)
        copy_aircc_lowered_ir(Path(args.artifact_dir) if args.artifact_dir else None)
        backend.unload()
        print("Compilation complete. Generated files:")
        print(f"  - air.{output_ext}")
        if args.output_format == "xclbin":
            print("  - air.insts.bin")
        print(f"shape={args.m}x{args.k}x{args.n}")
        print(f"tile_shape={args.tile_m}x{args.tile_n}x{args.k}")
        print(f"output_type={args.output_type}")
        print(f"b_layout={args.b_layout}")
        print(f"transform_variant={args.transform_variant}")
        print(f"kernel_impl={args.kernel_impl}")
        if args.external_kernel_object:
            print(f"external_kernel_object={args.external_kernel_object}")
        print(f"external_schedule={args.external_schedule}")
        print(f"external_kernel_style={args.external_kernel_style}")
        print(f"external_k_packs={args.external_k_packs}")
        print(f"external_block={args.external_block_m}x{args.external_block_n}")
        print(f"external_core_m_packs={args.external_core_m_packs}")
        print(f"external_active_m_packs={args.external_active_m_packs}")
        print(f"external_core_n_packs={args.external_core_n_packs}")
        print(f"external_c_stride_m_packs={args.external_c_stride_m_packs}")
        print(
            f"atb_k_chunk_elements={args.effective_atb_k_chunk_elements or args.atb_k_chunk_elements}"
        )
        print(f"omit_ping_pong={args.omit_ping_pong}")
        print(f"aircc_debug_ir={args.aircc_debug_ir}")
        raise SystemExit(0)

    input_type = np.int8
    output_type = np.int32 if args.output_type == "int32" else np.int8
    rng = np.random.default_rng(args.seed)
    A = rng.integers(low=0, high=8, size=(args.m, args.k), dtype=input_type)
    B = rng.integers(low=0, high=8, size=(args.k, args.n), dtype=input_type)
    if args.b_layout == "row":
        B_device = B
    else:
        b_tiles = args.n // args.tile_n
        b_pitch = args.tile_n + 4
        B_device = np.zeros((b_tiles, args.k, b_pitch), dtype=input_type)
        for tile in range(b_tiles):
            start = tile * args.tile_n
            B_device[tile, :, : args.tile_n] = B[:, start : start + args.tile_n]
        B_device = B_device.reshape(-1)

    C_i32 = np.matmul(A.astype(np.int32), B.astype(np.int32))
    C = C_i32.astype(output_type)

    runner = XRTRunner(
        **backend_kwargs,
        trace_file=args.trace_file,
    )
    raise SystemExit(
        runner.run_test(
            air_module,
            inputs=[A, B_device],
            expected_outputs=[C],
        )
    )
