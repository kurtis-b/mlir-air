#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.

"""Sweep NPU int8 GEMM transform and runtime-loop variants."""

from __future__ import annotations

import argparse
import itertools
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

M = N = K = 1024
RUNTIME_TILINGS = ("1,1", "1,2", "2,4", "4,4", "4,8", "8,4")


@dataclass(frozen=True)
class Variant:
    name: str
    runtime_tiling: str
    k_reduction_tile: int = 8
    l3_l2_copy_tile: int = 64
    vector_tile: tuple[int, int, int] = (2, 2, 1)


def positive_int(value: str) -> int:
    parsed = int(value, 10)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"expected exactly one occurrence of {old!r}")
    return text.replace(old, new, 1)


def render_transform(base: str, variant: Variant) -> str:
    text = base
    text = replace_once(
        text,
        "transform.structured.tile_using_for %copy1 tile_sizes [0, 64]",
        f"transform.structured.tile_using_for %copy1 tile_sizes [0, {variant.l3_l2_copy_tile}]",
    )
    text = replace_once(
        text,
        "transform.structured.tile_using_for %copy2 tile_sizes [64]",
        f"transform.structured.tile_using_for %copy2 tile_sizes [{variant.l3_l2_copy_tile}]",
    )
    text = replace_once(
        text,
        "transform.structured.tile_using_for %packed_c tile_sizes [0, 0, 8]",
        f"transform.structured.tile_using_for %packed_c tile_sizes [0, 0, {variant.k_reduction_tile}]",
    )
    vector_tile = ", ".join(str(v) for v in (*variant.vector_tile, 0, 0, 0))
    text = replace_once(
        text,
        "transform.structured.tile_using_for %generic2 tile_sizes [2, 2, 1, 0, 0, 0]",
        f"transform.structured.tile_using_for %generic2 tile_sizes [{vector_tile}]",
    )
    return text


def last_kv_value(text: str, key: str) -> str:
    matches = re.findall(rf"\b{re.escape(key)}=([^\s]+)", text)
    return matches[-1] if matches else ""


def run_capture(log: Path, argv: Sequence[object], *, cwd: Path | None = None) -> bool:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as output:
        cd = f"cd {cwd} && " if cwd else ""
        output.write(f"+ {cd}{' '.join(shlex.quote(str(arg)) for arg in argv)}\n")
        output.flush()
        completed = subprocess.run(
            [str(arg) for arg in argv],
            cwd=str(cwd) if cwd else None,
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        return completed.returncode == 0


def runtime_variants() -> list[Variant]:
    return [
        Variant(name=f"rtl_{sanitize(runtime)}", runtime_tiling=runtime)
        for runtime in RUNTIME_TILINGS
    ]


def transform_variants(best_runtime: str) -> list[Variant]:
    variants: list[Variant] = []
    for k_tile, copy_tile, vec_tile in itertools.product(
        (8, 16), (64, 128), ((2, 2, 1), (4, 2, 1))
    ):
        name = (
            f"rtl_{sanitize(best_runtime)}_k{k_tile}_copy{copy_tile}_"
            f"vec{vec_tile[0]}x{vec_tile[1]}x{vec_tile[2]}"
        )
        variants.append(
            Variant(
                name=name,
                runtime_tiling=best_runtime,
                k_reduction_tile=k_tile,
                l3_l2_copy_tile=copy_tile,
                vector_tile=vec_tile,
            )
        )
    return variants


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=repo / "test" / "xrt" / "46_triton_matmul_ver4_strix_8x4_i8_i8_i32",
    )
    parser.add_argument(
        "--phase", choices=["runtime", "transform", "all"], default="runtime"
    )
    parser.add_argument(
        "--best-runtime-loop-tiling",
        default="2,4",
        metavar="M,N",
        help="runtime tiling used for transform phase (default: 2,4)",
    )
    parser.add_argument(
        "--run", action="store_true", help="profile each compiled variant"
    )
    parser.add_argument("--warmups", type=positive_int, default=20)
    parser.add_argument("--iterations", type=positive_int, default=100)
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    out_dir = args.out_dir.resolve()
    transforms_dir = out_dir / "transforms"
    logs_dir = out_dir / "logs"
    builds_dir = out_dir / "build"
    for path in (transforms_dir, logs_dir, builds_dir):
        path.mkdir(parents=True, exist_ok=True)

    base_transform_path = args.source_dir / "transform_aie2p.mlir"
    base_transform = base_transform_path.read_text(encoding="utf-8")

    variants: list[Variant] = []
    if args.phase in {"runtime", "all"}:
        variants.extend(runtime_variants())
    if args.phase in {"transform", "all"}:
        variants.extend(transform_variants(args.best_runtime_loop_tiling))

    rows = []
    for variant in variants:
        transform_path = transforms_dir / f"{variant.name}.mlir"
        transform_path.write_text(
            render_transform(base_transform, variant), encoding="utf-8"
        )
        build_dir = builds_dir / variant.name
        target = "profile" if args.run else "compile-xclbin"
        command = [
            "make",
            "-C",
            args.source_dir,
            f"BUILD_DIR={build_dir}",
            "AIE_TARGET=aie2p",
            f"M={M}",
            f"K={K}",
            f"N={N}",
            f"TRANSFORM_SCRIPT_PATH={transform_path}",
            f"RUNTIME_LOOP_TILING={variant.runtime_tiling}",
            f"WARMUPS={args.warmups}",
            f"ITERATIONS={args.iterations}",
            target,
        ]
        log = logs_dir / f"{variant.name}.log"
        ok = run_capture(log, command)
        text = log.read_text(encoding="utf-8", errors="replace")
        rows.append(
            {
                "name": variant.name,
                "status": "PASS" if ok else "FAIL",
                "runtime_loop_tiling": variant.runtime_tiling,
                "k_reduction_tile": str(variant.k_reduction_tile),
                "l3_l2_copy_tile": str(variant.l3_l2_copy_tile),
                "vector_tile": "x".join(str(v) for v in variant.vector_tile),
                "avg_us": last_kv_value(text, "avg_us") or "n/a",
                "gops": last_kv_value(text, "gops") or "n/a",
                "validation": last_kv_value(text, "validation") or "n/a",
                "log": str(log),
            }
        )

    report = out_dir / "npu_sweep_report.md"
    with report.open("w", encoding="utf-8") as output:
        output.write("# NPU int8 GEMM Sweep\n\n")
        output.write(
            "| Variant | Status | Runtime Tiling | K Tile | Copy Tile | Vector Tile | Avg us | GOPS | Validation | Log |\n"
        )
        output.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for row in rows:
            output.write(
                f"| {row['name']} | {row['status']} | {row['runtime_loop_tiling']} | "
                f"{row['k_reduction_tile']} | {row['l3_l2_copy_tile']} | {row['vector_tile']} | "
                f"{row['avg_us']} | {row['gops']} | {row['validation']} | `{row['log']}` |\n"
            )
    print(f"Report: {report}")
    return 1 if any(row["status"] != "PASS" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
