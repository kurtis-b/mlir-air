#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compile or run Gemma3 herd-sweep kernels across supported herd shapes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess

from common import (
    OUTPUT_MODES,
    SUPPORTED_HERD_SHAPES,
    is_output_mode_supported,
    parse_herd_shape,
)

KERNEL_TARGETS = (
    "run-q4nx",
    "run-mm",
    "run-fused-dqp",
    "run-flowqkv",
    "run-flowkv",
)

OUTPUT_MODE_ENV = {
    "run-q4nx": ("q4nx", "Q4NX_OUTPUT_MODE"),
    "run-fused-dqp": ("fused_dqp", "FUSED_DQP_OUTPUT_MODE"),
    "run-flowqkv": ("flowqkv", "FLOWQKV_OUTPUT_MODE"),
    "run-flowkv": ("flowkv", "FLOWKV_OUTPUT_MODE"),
}


def run_make(cmd: list[str]) -> None:
    cmd_str = " ".join(cmd)
    print(f"==> {cmd_str}", flush=True)
    subprocess.run(cmd, check=True, env=os.environ.copy())


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Gemma3 herd-shape sweep")
    parser.add_argument(
        "--compile-mode",
        default="compile-only",
        choices=["compile-only", "compile-and-run"],
    )
    parser.add_argument("--output-format", default="elf", choices=["elf", "xclbin"])
    parser.add_argument("--shape", action="append", choices=SUPPORTED_HERD_SHAPES)
    parser.add_argument("--kernel", action="append", choices=KERNEL_TARGETS)
    parser.add_argument("--output-mode", action="append", choices=OUTPUT_MODES)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    shapes = tuple(args.shape) if args.shape else SUPPORTED_HERD_SHAPES
    targets = tuple(args.kernel) if args.kernel else KERNEL_TARGETS
    output_modes = tuple(dict.fromkeys(args.output_mode or ("auto",)))

    for shape in shapes:
        herd_rows, herd_cols = parse_herd_shape(shape)
        for target in targets:
            if target == "run-mm":
                run_make(
                    [
                        "make",
                        "-C",
                        str(root),
                        target,
                        f"HERD_SHAPE={shape}",
                        f"COMPILE_MODE={args.compile_mode}",
                        f"OUTPUT_FORMAT={args.output_format}",
                    ],
                )
                continue

            kernel, env_name = OUTPUT_MODE_ENV[target]
            for output_mode in output_modes:
                if not is_output_mode_supported(
                    output_mode, herd_rows, herd_cols, kernel
                ):
                    print(
                        f"==> skip {target} HERD_SHAPE={shape} "
                        f"OUTPUT_MODE={output_mode} (unsupported)",
                        flush=True,
                    )
                    continue

                run_make(
                    [
                        "make",
                        "-C",
                        str(root),
                        target,
                        f"HERD_SHAPE={shape}",
                        f"COMPILE_MODE={args.compile_mode}",
                        f"OUTPUT_FORMAT={args.output_format}",
                        f"{env_name}={output_mode}",
                    ],
                )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode)
