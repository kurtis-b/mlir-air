#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Synthetic Gemma3 model-loop entrypoint.

This is the first Llama32-style control-plane shell for Gemma3: compile-only
writes a kernel manifest, run-only loads it, and --verify uses the CPU reference
path until real NPU artifact execution is implemented.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gemma3_config import describe_kernel_sequence, synthetic_text_config
from gemma3_model_loop import Gemma3SyntheticSession
from gemma3_runtime import format_manifest, prepare_runtime


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma3 synthetic model-loop inference")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--compile-only", action="store_true")
    mode.add_argument("--run-only", action="store_true")
    mode.add_argument("--print-sequence", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path("gemma3_kernel_cache"))
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--include-stages", action="store_true")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--local-window-len", type=int, default=None)
    parser.add_argument("--prefill-chunks", type=int, default=2)
    parser.add_argument("--decode-tokens", type=int, default=2)
    args = parser.parse_args()

    config_kwargs = {"n_layers": args.layers}
    if args.local_window_len is not None:
        config_kwargs["local_window_len"] = args.local_window_len
    config = synthetic_text_config(**config_kwargs)

    if args.print_sequence:
        print(describe_kernel_sequence(config))
        return 0

    if args.compile_only:
        manifest = prepare_runtime(config, cache_dir=args.cache_dir, compile_only=True)
        print(format_manifest(manifest))
        print(f"GEMMA3_RUNTIME_COMPILE_ONLY: wrote {args.cache_dir}")
        return 0

    manifest = prepare_runtime(config, cache_dir=args.cache_dir, run_only=True)
    print(format_manifest(manifest))
    print("GEMMA3_RUNTIME_RUN_ONLY: manifest loaded")
    if args.verify:
        session = Gemma3SyntheticSession(config=config, manifest=manifest, profile=args.profile)
        report = session.run(
            prefill_chunks=args.prefill_chunks,
            decode_tokens=args.decode_tokens,
            include_stages=args.include_stages,
        )
        print(report.format(profile=args.profile, include_stages=args.include_stages))
        print("GEMMA3_SYNTHETIC_VERIFY: PASS")
    else:
        print("GEMMA3_NPU_EXECUTION: not implemented; use --verify for CPU reference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
