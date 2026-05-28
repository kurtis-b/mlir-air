#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compile or run Gemma3 herd-sweep kernels across supported herd shapes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from common import SUPPORTED_HERD_SHAPES

KERNEL_TARGETS = (
    "run-q4nx",
    "run-mm",
    "run-fused-dqp",
    "run-flowqkv",
    "run-flowkv",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Gemma3 herd-shape sweep")
    parser.add_argument("--compile-mode", default="compile-only", choices=["compile-only", "compile-and-run"])
    parser.add_argument("--output-format", default="elf", choices=["elf", "xclbin"])
    parser.add_argument("--shape", action="append", choices=SUPPORTED_HERD_SHAPES)
    parser.add_argument("--kernel", action="append", choices=KERNEL_TARGETS)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    shapes = tuple(args.shape) if args.shape else SUPPORTED_HERD_SHAPES
    targets = tuple(args.kernel) if args.kernel else KERNEL_TARGETS

    for shape in shapes:
        for target in targets:
            cmd = [
                "make",
                "-C",
                str(root),
                target,
                f"HERD_SHAPE={shape}",
                f"COMPILE_MODE={args.compile_mode}",
                f"OUTPUT_FORMAT={args.output_format}",
            ]
            print(f"==> {' '.join(cmd)}", flush=True)
            subprocess.run(cmd, check=True, env=os.environ.copy())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode)
