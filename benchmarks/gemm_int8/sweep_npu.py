#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.

"""Sweep NPU int8 GEMM transform/runtime variants with static evidence."""

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

DEFAULT_M = DEFAULT_N = DEFAULT_K = 1024
DEFAULT_TILE_M = 512
DEFAULT_TILE_N = 256
SOTA_M = 4032
SOTA_K = 4320
SOTA_N = 4608
SOTA_TILE_M = 576
SOTA_TILE_N = 1152
DEFAULT_ACCEPTANCE_TOPS = 36.15
RUNTIME_TILINGS = ("1,1", "1,2", "2,4", "4,4", "4,8", "8,4")


@dataclass(frozen=True)
class Variant:
    name: str
    runtime_tiling: str
    k_reduction_tile: int = 8
    l3_l2_copy_tile: int = 64
    vector_tile: tuple[int, int, int] = (2, 2, 1)
    m: int | None = None
    k: int | None = None
    n: int | None = None
    tile_m: int | None = None
    tile_n: int | None = None
    transform_variant: str = "default"
    output_type: str | None = None
    b_layout: str | None = None
    notes: str = ""


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
        name = f"rtl_{sanitize(best_runtime)}_k{k_tile}_copy{copy_tile}_vec{vec_tile[0]}x{vec_tile[1]}x{vec_tile[2]}"
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


def sota_variants() -> list[Variant]:
    return [
        Variant(
            name="sota_int8_col_rtl_2x4",
            runtime_tiling="2,4",
            m=SOTA_M,
            k=SOTA_K,
            n=SOTA_N,
            tile_m=SOTA_TILE_M,
            tile_n=SOTA_TILE_N,
            transform_variant="sota-int8",
            output_type="int8",
            b_layout="column",
            notes="INT8-output acceptance row: output-stationary C, column-major B, 4x8 SOTA split",
        ),
        Variant(
            name="sota_int8_col_rtl_4x8",
            runtime_tiling="4,8",
            m=SOTA_M,
            k=SOTA_K,
            n=SOTA_N,
            tile_m=SOTA_TILE_M,
            tile_n=SOTA_TILE_N,
            transform_variant="sota-int8",
            output_type="int8",
            b_layout="column",
            notes="stress runtime-loop tiling for BD overlap",
        ),
    ]


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
        "--phase", choices=["runtime", "transform", "sota", "all"], default="runtime"
    )
    parser.add_argument(
        "--best-runtime-loop-tiling",
        default="2,4",
        metavar="M,N",
        help="runtime tiling used for transform phase (default: 2,4)",
    )
    parser.add_argument("-M", "--m", type=positive_int, default=DEFAULT_M)
    parser.add_argument("-K", "--k", type=positive_int, default=DEFAULT_K)
    parser.add_argument("-N", "--n", type=positive_int, default=DEFAULT_N)
    parser.add_argument("--tile-m", type=positive_int, default=DEFAULT_TILE_M)
    parser.add_argument("--tile-n", type=positive_int, default=DEFAULT_TILE_N)
    parser.add_argument("--output-type", choices=["int32", "int8"], default="int32")
    parser.add_argument("--b-layout", choices=["row", "column"], default="row")
    parser.add_argument(
        "--acceptance-tops", type=positive_float, default=DEFAULT_ACCEPTANCE_TOPS
    )
    parser.add_argument("--trace-size", type=nonnegative_int, default=0)
    parser.add_argument("--trace-offset", type=nonnegative_int, default=0)
    parser.add_argument(
        "--validation", choices=["none", "samples", "full"], default="samples"
    )
    parser.add_argument("--validation-samples", type=positive_int, default=64)
    parser.add_argument(
        "--run", action="store_true", help="profile each compiled variant"
    )
    parser.add_argument("--warmups", type=positive_int, default=20)
    parser.add_argument("--iterations", type=positive_int, default=100)
    args = parser.parse_args(argv)
    if args.m % args.tile_m or args.n % args.tile_n:
        parser.error("M and N must be multiples of --tile-m and --tile-n")
    if args.tile_m % 8 or args.tile_n % 8:
        parser.error("tile sizes must be multiples of 8")
    if args.k % 8:
        parser.error("K must be a multiple of 8")
    return args


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
    if args.phase in {"sota", "all"}:
        variants.extend(sota_variants())

    rows = []
    for variant in variants:
        transform_path = transforms_dir / f"{variant.name}.mlir"
        transform_path.write_text(
            render_transform(base_transform, variant), encoding="utf-8"
        )
        build_dir = builds_dir / variant.name
        artifact_dir = build_dir / "artifacts"
        output_type = variant.output_type or args.output_type
        b_layout = variant.b_layout or args.b_layout
        m = variant.m or args.m
        k = variant.k or args.k
        n = variant.n or args.n
        tile_m = variant.tile_m or args.tile_m
        tile_n = variant.tile_n or args.tile_n
        target = "profile" if args.run else "compile-xclbin"
        command = [
            "make",
            "-C",
            args.source_dir,
            f"BUILD_DIR={build_dir}",
            f"ARTIFACT_DIR={artifact_dir}",
            "AIE_TARGET=aie2p",
            f"M={m}",
            f"K={k}",
            f"N={n}",
            f"TILE_M={tile_m}",
            f"TILE_N={tile_n}",
            f"OUTPUT_TYPE={output_type}",
            f"B_LAYOUT={b_layout}",
            f"VALIDATION={args.validation}",
            f"VALIDATION_SAMPLES={args.validation_samples}",
            f"TRANSFORM_SCRIPT_PATH={transform_path}",
            f"TRANSFORM_VARIANT={variant.transform_variant}",
            f"RUNTIME_LOOP_TILING={variant.runtime_tiling}",
            f"TRACE_SIZE={args.trace_size}",
            f"TRACE_OFFSET={args.trace_offset}",
            f"ACCEPTANCE_TOPS={args.acceptance_tops}",
            f"WARMUPS={args.warmups}",
            f"ITERATIONS={args.iterations}",
            target,
        ]
        log = logs_dir / f"{variant.name}.log"
        ok = run_capture(log, command)
        text = log.read_text(encoding="utf-8", errors="replace")
        avg_tops = last_kv_value(text, "avg_tops") or "n/a"
        rows.append(
            {
                "name": variant.name,
                "status": "PASS" if ok else "FAIL",
                "shape": f"{m}x{k}x{n}",
                "tile_shape": f"{tile_m}x{tile_n}",
                "runtime_loop_tiling": variant.runtime_tiling,
                "transform_variant": variant.transform_variant,
                "k_reduction_tile": str(variant.k_reduction_tile),
                "l3_l2_copy_tile": str(variant.l3_l2_copy_tile),
                "vector_tile": "x".join(str(v) for v in variant.vector_tile),
                "output_type": output_type,
                "b_layout": b_layout,
                "avg_us": last_kv_value(text, "avg_us") or "n/a",
                "gops": last_kv_value(text, "gops") or "n/a",
                "avg_tops": avg_tops,
                "meets_acceptance": last_kv_value(text, "meets_acceptance") or "n/a",
                "validation": last_kv_value(text, "validation") or "n/a",
                "static_report": str(artifact_dir / "npu_gemm.static.md"),
                "log": str(log),
                "notes": variant.notes,
            }
        )

    report = out_dir / "npu_sweep_report.md"
    with report.open("w", encoding="utf-8") as output:
        output.write("# NPU int8 GEMM Sweep\n\n")
        output.write(f"Acceptance target: `{args.acceptance_tops:.2f} TOPS`\n\n")
        output.write(
            "| Variant | Status | Shape | Tile | Runtime Tiling | Transform | K Tile | Copy Tile | Vector Tile | Output | B Layout | Avg us | GOPS | TOPS | Accept | Validation | Static Report | Log | Notes |\n"
        )
        output.write(
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
        )
        for row in rows:
            output.write(
                f"| {row['name']} | {row['status']} | {row['shape']} | {row['tile_shape']} | {row['runtime_loop_tiling']} | {row['transform_variant']} | "
                f"{row['k_reduction_tile']} | {row['l3_l2_copy_tile']} | {row['vector_tile']} | "
                f"{row['output_type']} | {row['b_layout']} | {row['avg_us']} | {row['gops']} | {row['avg_tops']} | "
                f"{row['meets_acceptance']} | {row['validation']} | `{row['static_report']}` | `{row['log']}` | {row['notes']} |\n"
            )
    print(f"Report: {report}")
    return 1 if any(row["status"] != "PASS" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
