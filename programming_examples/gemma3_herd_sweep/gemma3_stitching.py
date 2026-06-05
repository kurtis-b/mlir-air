#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gemma3 text-based MLIR stitching helpers.

This is the Gemma-side adaptation of the Llama32 multi-launch stitching pattern.
It keeps the implementation text-based because the MLIR Python bindings do not
provide a stable operation-moving API for assembling independent generated
modules into one function.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


GEMMA3_EXTERN_FUNCS = frozenset(
    {
        "@fused_dqp_accum_block_opt",
        "@fused_dqp_accum_block",
        "@rope",
        "@gemma3_geglu",
        "@gemma3_residual_add",
        "@flowqkv_chunk_bf16",
    }
)


@dataclass(frozen=True)
class StitchSpec:
    mlir_text: str
    prefix: str
    arg_map: dict[int, int]
    extern_funcs: frozenset[str] = GEMMA3_EXTERN_FUNCS
    wrap_bare_herds: bool = False


def _extract_between_func_and_return(mlir_text: str) -> str:
    """Extract the public func body between its signature and trailing return."""
    lines = mlir_text.split("\n")
    body_start = None
    for i, line in enumerate(lines):
        if "func.func @" in line and "private" not in line:
            body_start = i + 1
    if body_start is None:
        raise ValueError("MLIR module does not contain a public func.func")

    return_re = re.compile(r"^\s*return(\s|$|//|loc\()")
    body_end = None
    for i in range(len(lines) - 1, body_start, -1):
        if return_re.match(lines[i]):
            body_end = i
            break
    if body_end is None:
        raise ValueError("MLIR public func does not contain a trailing return")
    return "\n".join(lines[body_start:body_end])


def _extract_affine_maps(mlir_text: str) -> list[str]:
    return [line for line in mlir_text.split("\n") if line.startswith("#map")]


def _extract_private_funcs(mlir_text: str) -> list[str]:
    return [line for line in mlir_text.split("\n") if "func.func private" in line]


def _rename_all_with_externs(text: str, prefix: str, extern_funcs: frozenset[str]) -> str:
    """Rename SSA values, affine maps, and non-external symbols with a prefix."""
    for name in sorted(set(re.findall(r"#map\d*", text)), key=len, reverse=True):
        text = re.sub(re.escape(name) + r"(?!\w)", f"#{prefix}_{name[1:]}", text)

    for name in sorted(set(re.findall(r"%[a-zA-Z_]\w*", text)), key=len, reverse=True):
        text = re.sub(re.escape(name) + r"(?!\w)", f"%{prefix}_{name[1:]}", text)

    for name in sorted(
        set(re.findall(r"%\d+", text)), key=lambda value: int(value[1:]), reverse=True
    ):
        text = re.sub(re.escape(name) + r"(?!\d)", f"%{prefix}_n{name[1:]}", text)

    for name in sorted(set(re.findall(r"@[\w]+", text)), key=len, reverse=True):
        if name not in extern_funcs:
            text = text.replace(name, f"@{prefix}_{name[1:]}")

    return text


def _fix_launch_func_args(text: str, prefix: str, arg_map: dict[int, int]) -> str:
    """Remap public func args in air.launch args() clauses after prefixing."""
    for original_index, combined_index in arg_map.items():
        old_ref = f"%{prefix}_arg{original_index}"
        new_ref = f"%arg{combined_index}"
        text = text.replace(f"={old_ref},", f"={new_ref},")
        text = text.replace(f"={old_ref})", f"={new_ref})")
    return text


def _wrap_ir_in_launch(mlir_text: str) -> str:
    """Wrap a public func containing bare herds in launch+segment scaffolding."""
    lines = mlir_text.split("\n")
    func_line_idx = None
    for i, line in enumerate(lines):
        if "func.func @" in line and "private" not in line:
            func_line_idx = i
            break
    if func_line_idx is None:
        return mlir_text

    body_text = "\n".join(lines[func_line_idx + 1 :])
    if "air.launch" in body_text:
        return mlir_text

    func_line = lines[func_line_idx]
    func_name_match = re.search(r"func\.func @(\w+)", func_line)
    func_name = func_name_match.group(1) if func_name_match else "wrapped"
    sig_match = re.search(r"func\.func @\w+\(([^)]*)\)", func_line)
    if not sig_match:
        return mlir_text

    func_args: list[tuple[str, str]] = []
    for arg in sig_match.group(1).split(","):
        arg = arg.strip()
        if not arg:
            continue
        name, typ = arg.split(":", 1)
        func_args.append((name.strip(), typ.strip()))

    body_start = func_line_idx + 1
    body_end = None
    for i in range(len(lines) - 1, body_start, -1):
        if lines[i].strip() == "return":
            body_end = i
            break
    if body_end is None:
        return mlir_text

    body_text = "\n".join(lines[body_start:body_end])
    existing_args = [int(match) for match in re.findall(r"%arg(\d+)", body_text)]
    max_existing = max(existing_args) if existing_args else len(func_args) - 1
    launch_arg_start = max_existing + 1
    segment_arg_start = launch_arg_start + len(func_args)

    launch_args = ", ".join(
        f"%arg{launch_arg_start + i}={name}" for i, (name, _typ) in enumerate(func_args)
    )
    launch_types = ", ".join(typ for _name, typ in func_args)
    segment_args = ", ".join(
        f"%arg{segment_arg_start + i}=%arg{launch_arg_start + i}"
        for i in range(len(func_args))
    )

    for i in range(len(func_args) - 1, -1, -1):
        old_name = func_args[i][0]
        body_text = re.sub(
            re.escape(old_name) + r"(?!\w)", f"%arg{segment_arg_start + i}", body_text
        )

    new_lines = lines[:body_start]
    new_lines.append(f"    air.launch () in () args({launch_args}) : {launch_types} {{")
    new_lines.append(
        f"      air.segment @{func_name}_seg args({segment_args}) : {launch_types} {{"
    )
    for line in body_text.split("\n"):
        new_lines.append("    " + line)
    new_lines.append("      }")
    new_lines.append("    }")
    new_lines.extend(lines[body_end:])
    return "\n".join(new_lines)


def _private_symbol_name(private_decl: str) -> str:
    match = re.search(r"@(\w+)", private_decl)
    return match.group(1) if match else private_decl


def stitch_module_text(
    *,
    function_name: str,
    arg_types: tuple[str, ...],
    specs: tuple[StitchSpec, ...],
) -> str:
    """Build one MLIR module containing one public multi-launch function."""
    bodies: list[str] = []
    maps_all: list[str] = []
    private_decls: list[str] = []
    private_names: set[str] = set()

    for spec in specs:
        mlir_text = _wrap_ir_in_launch(spec.mlir_text) if spec.wrap_bare_herds else spec.mlir_text
        body = _extract_between_func_and_return(mlir_text)
        maps = _extract_affine_maps(mlir_text)
        body = _rename_all_with_externs(body, spec.prefix, spec.extern_funcs)
        maps = [
            _rename_all_with_externs(affine_map, spec.prefix, spec.extern_funcs)
            for affine_map in maps
        ]
        body = _fix_launch_func_args(body, spec.prefix, spec.arg_map)
        bodies.append(body)
        maps_all.extend(maps)

        for private_decl in _extract_private_funcs(mlir_text):
            renamed = _rename_all_with_externs(private_decl.strip(), spec.prefix, spec.extern_funcs)
            symbol = _private_symbol_name(renamed)
            if symbol not in private_names:
                private_names.add(symbol)
                private_decls.append(renamed)

    args = ",\n    ".join(
        f"%arg{index}: {arg_type}" for index, arg_type in enumerate(arg_types)
    )
    maps_text = "\n".join(maps_all)
    private_text = "\n  ".join(private_decls)
    body_text = "\n".join(bodies)
    prefix = f"{maps_text}\n" if maps_text else ""
    private_block = f"  {private_text}\n" if private_text else ""
    return f"""{prefix}module {{
{private_block}  func.func @{function_name}(
    {args}
  ) {{
{body_text}
    return
  }}
}}
"""


def _self_test() -> None:
    a_ir = """module {
  func.func @a(%arg0: memref<4xbf16>, %arg1: memref<4xbf16>) {
    air.launch () in () args(%arg2=%arg0, %arg3=%arg1) : memref<4xbf16>, memref<4xbf16> {
    }
    return
  }
}
"""
    b_ir = """module {
  func.func private @fused_dqp_accum_block_opt(memref<4xbf16>) attributes {link_with = "fused_dqp.o", llvm.emit_c_interface}
  func.func @b(%arg0: memref<4xbf16>, %arg1: memref<4xbf16>) {
    air.launch () in () args(%arg2=%arg0, %arg3=%arg1) : memref<4xbf16>, memref<4xbf16> {
      call @fused_dqp_accum_block_opt(%arg2) : (memref<4xbf16>) -> ()
    }
    return
  }
}
"""
    combined = stitch_module_text(
        function_name="gemma3_stitch_self_test",
        arg_types=("memref<4xbf16>", "memref<4xbf16>", "memref<4xbf16>"),
        specs=(
            StitchSpec(a_ir, "a", {0: 0, 1: 1}),
            StitchSpec(b_ir, "b", {0: 1, 1: 2}),
        ),
    )
    if "%a_arg0" in combined or "%b_arg0" in combined:
        raise AssertionError("stitched launch args were not remapped")
    if combined.count("air.launch") != 2:
        raise AssertionError("expected two launches")
    if combined.count("func.func private @fused_dqp_accum_block_opt") != 1:
        raise AssertionError("expected one deduped FusedDQP private declaration")
    print("gemma3_stitching self-test status=PASS launches=2 args=3")


if __name__ == "__main__":
    _self_test()
