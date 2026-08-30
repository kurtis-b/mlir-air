# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SmolLM2-1.7B int4 (GGUF q4_0) Inference on MLIR-AIR (NPU2) — thin entry.

The shipped shape mirrors `llama32_1b_int4`: **bf16 NPU prefill on the
dequantized q4_0 weights + int4 NPU decode on the packed BOs**, so the whole
model IS the quantized model (the prefill dequant is taken from the q4_0
payloads, never the original bf16 — see `smollm2_1_7b_int4_weights`).

All heavy machinery (Session, prepare_runtime, run_npu_prefill,
run_npu_decode_step, generate, run_once, REPL) is reused verbatim from
`llama32_1b_int4_inference` — it is config-driven, and SmolLM2's full MHA is
its `group_size = 1` case. This module only supplies:

  1. SmolLM2's config / RoPE / model IDs and the GGUF q4_0 weights loader
     (the checkpoint path resolves through `resolve_gguf_path`: `--gguf`,
     else `$SMOLLM2_GGUF`, else the hub file via the HF cache).
  2. Kernel compilation at SmolLM2 shapes: bf16 prefill via the shared
     registry-driven `llama32_1b_prefill.compile_all_kernels` (MHA-safe, the
     same route the bf16 smollm2 example takes), int4 decode via
     `smollm2_1_7b_int4_decode.compile_decode_kernels` (gs=32).

Usage:
  python3 smollm2_1_7b_int4_inference.py --compile-only
  python3 smollm2_1_7b_int4_inference.py --run-only --n-tokens 100
  python3 smollm2_1_7b_int4_inference.py --run-only --interactive
"""

import os
import sys

from ml_dtypes import bfloat16

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_LLMS_DIR = os.path.dirname(_THIS_DIR)
_LLAMA_BF16 = os.path.join(_LLMS_DIR, "llama32_1b")
_LLAMA_INT4 = os.path.join(_LLMS_DIR, "llama32_1b_int4")
_SMOLLM2_BF16 = os.path.join(_LLMS_DIR, "smollm2_1_7b")
for _p in (_LLMS_DIR, _SMOLLM2_BF16, _LLAMA_BF16, _LLAMA_INT4, _THIS_DIR):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

# --- (1) SmolLM2 config / weights / RoPE / model IDs. ---
from smollm2_1_7b_weights import LlamaConfig, generate_rope_lut  # noqa: E402
from smollm2_1_7b_int4_weights import (  # noqa: E402
    HF_MODEL,
    load_weights_gguf_q4_0,
    resolve_gguf_path,
)
from shared.infra.cache import KernelCache, Profiler  # noqa: E402

# --- (2) kernel compilation entries. ---
from llama32_1b_prefill import compile_all_kernels  # noqa: E402
from smollm2_1_7b_int4_decode import compile_decode_kernels  # noqa: E402

# Reuse the int4 reference's Session machinery + run loops verbatim, including
# its `_multi_launch_dir` context that swaps which `multi_launch_builder`
# package a compile resolves (bf16 prefill vs int4 decode namesakes).
from llama32_1b_int4_inference import (  # noqa: E402
    Session,
    _multi_launch_dir,
    _print_one_shot_output,
    prepare_runtime,
    repl_loop,
    run_once,
)

SEQ_LEN = 2048


def build_session(args) -> Session:
    """One-time setup, mirroring `llama32_1b_int4_inference.build_session`
    with SmolLM2 config, gs=32 decode kernels, and the GGUF loader."""
    config = LlamaConfig()
    seq_len = SEQ_LEN

    prefill_cache = KernelCache(
        "prefill_kernel_cache",
        verbose=args.verbose,
        profiler=Profiler(enabled=args.profile),
    )
    decode_cache = KernelCache(
        "decode_kernel_cache",
        verbose=args.verbose,
        profiler=Profiler(enabled=args.profile),
    )

    if not args.run_only:
        print("Compiling bf16 prefill kernels (SmolLM2 shapes)...")
        with _multi_launch_dir(_LLAMA_BF16):
            compile_all_kernels(prefill_cache, config, seq_len, cpu_attn=args.cpu_attn)
        print("\nCompiling int4 decode kernels (gs=32)...")
        with _multi_launch_dir(_LLAMA_INT4):
            compile_decode_kernels(decode_cache, config)

    if args.compile_only:
        # Stable end-of-compile marker for CI (mirrors the siblings' lit CHECK).
        print("\nCompilation passed.")
        sys.exit(0)

    if args.run_only:
        prefill_cache.load_manifest()
        decode_cache.load_manifest()

    gguf = resolve_gguf_path(args.gguf)
    print(f"\nLoading GGUF q4_0 weights ({gguf})...")
    weights = load_weights_gguf_q4_0(gguf, HF_MODEL, config=config)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)

    rope_lut_bf16 = generate_rope_lut(
        config=config,
        seq_len=seq_len + args.n_tokens,
    ).astype(bfloat16)

    prepare_runtime(
        prefill_cache, decode_cache, weights, config, seq_len, rope_lut_bf16
    )

    return Session(
        config=config,
        seq_len=seq_len,
        weights=weights,
        tokenizer=tokenizer,
        prefill_cache=prefill_cache,
        decode_cache=decode_cache,
        rope_lut_bf16=rope_lut_bf16,
        model_path=gguf,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SmolLM2-1.7B int4 (GGUF q4_0) Inference (NPU)"
    )
    parser.add_argument(
        "--gguf",
        type=str,
        default=None,
        help="the Q4_0 GGUF checkpoint (default: $SMOLLM2_GGUF, else the "
        "bartowski hub file through the HF cache)",
    )
    parser.add_argument("--prompt", type=str, default="What is the capital of France?")
    parser.add_argument("--n-tokens", type=int, default=100)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--run-only", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--cpu-attn", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.interactive:
        if args.compile_only:
            parser.error("--interactive cannot be combined with --compile-only")
        if not args.run_only:
            parser.error("--interactive requires --run-only")
        if args.profile:
            print(
                "WARNING: --profile is ignored in --interactive mode.",
                file=sys.stderr,
            )
            args.profile = False

    session = build_session(args)

    if args.interactive:
        repl_loop(session, args)
    else:
        # The int4 reference's run_once has no `profile` kwarg (profiling is
        # a KernelCache/Profiler property set at build_session time).
        generated, _prompt_len = run_once(
            session,
            args.prompt,
            n_tokens=args.n_tokens,
            cpu_attn=args.cpu_attn,
        )
        _print_one_shot_output(session, args.prompt, generated)
        if args.profile:
            print("\n=== Prefill profile ===")
            session.prefill_cache.profiler.report()
            print("\n=== Decode profile ===")
            session.decode_cache.profiler.report()
