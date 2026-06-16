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
import sys
from pathlib import Path

from gemma3.core.config import describe_kernel_sequence, synthetic_text_config
from gemma3.model.model_loop import Gemma3SyntheticSession
from gemma3.evidence.paper_compare import compare_results, load_targets
from gemma3.evidence.results import build_paper_result, format_result, write_result_json
from gemma3.core.runtime import format_manifest, prepare_runtime as prepare_synthetic_runtime


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma3 synthetic model-loop inference")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--compile-only", action="store_true")
    mode.add_argument("--run-only", action="store_true")
    mode.add_argument("--print-sequence", action="store_true")
    mode.add_argument("--paper-benchmark", action="store_true")
    mode.add_argument("--prepare-runtime", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path("gemma3_kernel_cache"))
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--include-stages", action="store_true")
    parser.add_argument("--include-ownership", action="store_true")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--local-window-len", type=int, default=None)
    parser.add_argument("--prefill-chunks", type=int, default=2)
    parser.add_argument("--decode-tokens", type=int, default=2)
    parser.add_argument("--model-variant", choices=["gemma3-1b", "gemma3-4b", "gemma3-4b-vision"], default="gemma3-1b")
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--backend", choices=["cpu", "igpu", "npu"], default="npu")
    parser.add_argument("--prompt-len", type=int, default=1024)
    parser.add_argument("--decode-context", type=int)
    parser.add_argument("--metric", choices=["prefill_ttft_seconds", "decode_tps", "vision_ttft_seconds"])
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--timed-iters", type=int, default=10)
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--power-sample", action="store_true")
    parser.add_argument("--compare-paper", action="store_true")
    parser.add_argument("--trace-size", type=int)
    parser.add_argument("--debug-ir", action="store_true")
    parser.add_argument("--skip-host-fallback-measurement", action="store_true")
    parser.add_argument("--fallback-timed-iters", type=int, default=3)
    parser.add_argument("--quantized-weights", choices=["required", "off"], default="required")
    parser.add_argument("--quantized-weights-dir", type=Path)
    parser.add_argument("--force-quantized-weights", action="store_true")
    parser.add_argument("--runtime-cache-dir", type=Path)
    parser.add_argument("--prefill-evidence-json", type=Path)
    args = parser.parse_args()

    if args.prepare_runtime:
        from gemma3.npu.inference_runtime import prepare_runtime as prepare_npu_runtime

        session = prepare_npu_runtime(
            model_variant=args.model_variant,
            prompt_len=args.prompt_len,
            decode_context=args.decode_context or args.prompt_len,
            weights_dir=args.weights_dir,
            tokenizer=args.tokenizer,
            quantized_weights=args.quantized_weights,
            quantized_weights_dir=args.quantized_weights_dir,
            force_quantized_weights=args.force_quantized_weights,
            runtime_cache_dir=args.runtime_cache_dir,
            prefill_evidence_path=args.prefill_evidence_json,
        )
        print(session.setup.format(include_ownership=args.include_ownership))
        if args.result_json:
            write_result_json(session.setup.to_json_dict(), args.result_json)
            print(f"GEMMA3_NPU_RUNTIME_JSON: {args.result_json}")
        print(
            "GEMMA3_NPU_RUNTIME_PREPARE: ready"
            if session.setup.ready_for_entrypoints
            else "GEMMA3_NPU_RUNTIME_PREPARE: blocked"
        )
        return 0

    if args.paper_benchmark:
        result = build_paper_result(
            model_variant=args.model_variant,
            backend=args.backend,
            weights_dir=args.weights_dir,
            tokenizer=args.tokenizer,
            prompt_len=args.prompt_len,
            decode_tokens=args.decode_tokens,
            metric=args.metric,
            warmup_iters=args.warmup_iters,
            timed_iters=args.timed_iters,
            artifact_format="elf",
            compile_time_included=False,
            command=["python", "-m", "gemma3.model.inference", *sys.argv[1:]],
            power_sample=args.power_sample,
            trace_size=args.trace_size,
            debug_ir=args.debug_ir,
            measure_host_fallbacks=not args.skip_host_fallback_measurement,
            fallback_timed_iters=args.fallback_timed_iters,
            quantized_weights=args.quantized_weights,
            quantized_weights_dir=args.quantized_weights_dir,
            force_quantized_weights=args.force_quantized_weights,
        )
        print(format_result(result))
        if result["classification"] == "MISSING_REAL_ARTIFACTS":
            print("GEMMA3_PAPER_BENCHMARK_BLOCKED: missing_real_artifacts")
        if result["classification"] == "REAL_MODEL_EXECUTION_NOT_IMPLEMENTED":
            print("GEMMA3_PAPER_BENCHMARK_BLOCKED: real_model_execution_not_implemented")
        if args.result_json:
            write_result_json(result, args.result_json)
            print(f"GEMMA3_RESULT_JSON: {args.result_json}")
        if args.compare_paper:
            for comparison in compare_results(load_targets(), result):
                print(comparison.format())
        return 0

    config_kwargs = {"n_layers": args.layers}
    if args.local_window_len is not None:
        config_kwargs["local_window_len"] = args.local_window_len
    config = synthetic_text_config(**config_kwargs)

    if args.print_sequence:
        print(describe_kernel_sequence(config))
        return 0

    if args.compile_only:
        manifest = prepare_synthetic_runtime(config, cache_dir=args.cache_dir, compile_only=True)
        print(format_manifest(manifest))
        print(f"GEMMA3_RUNTIME_COMPILE_ONLY: wrote {args.cache_dir}")
        return 0

    manifest = prepare_synthetic_runtime(config, cache_dir=args.cache_dir, run_only=True)
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
