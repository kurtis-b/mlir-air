# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared CLI driver steps for the bf16 inference examples.

`build_session` was one implementation copied into six `<model>_inference.py`
files (346 lines; the only textual difference was black's wrapping of one call).
It is kept here once, with the per-model handles passed in — the injection
contract recorded for goal 6c: the config and Session classes, the model-id
map, the two compile callables, and this model's weights/rope/runtime helpers.

Note for anyone gating a change here: `build_session` is reached from the
inference CLI's `main`, NOT from `verify_adapter.py`, so `make verify` passes
regardless of what this function does. The signal is `make run`, whose greedy
(argmax) decode is deterministic — see `agents/.state/6c-gate/`.
"""

import sys

from ml_dtypes import bfloat16

from shared.infra.cache import KernelCache, Profiler


def build_session(
    args,
    *,
    config_cls,
    session_cls,
    model_choices,
    load_weights,
    generate_rope_lut,
    compile_all_kernels,
    compile_decode_kernels,
    prepare_runtime,
    seq_len=2048,
):
    """Build the inference Session: compile or load kernels, then load weights.

    Every argument after `args` is that model's own handle; nothing here is
    model-specific. `seq_len` is a parameter rather than a constant only
    because it is a property of the model, not of this procedure — all six
    callers pass the same 2048 today.
    """
    config = config_cls()

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
        print("Compiling prefill kernels...")
        compile_all_kernels(
            prefill_cache, config, seq_len, verbose=args.verbose, cpu_attn=args.cpu_attn
        )
        print("\nCompiling decode kernels...")
        compile_decode_kernels(decode_cache, config, verbose=args.verbose)

    if args.compile_only:
        print("\nCompilation passed.")
        sys.exit(0)

    if args.run_only:
        prefill_cache.load_manifest()
        decode_cache.load_manifest()

    model_id = model_choices.get(args.model, args.model)
    print(f"\nLoading weights ({model_id})...")
    weights = load_weights(model_id, config=config)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    rope_lut_bf16 = generate_rope_lut(
        config=config, seq_len=seq_len + args.n_tokens
    ).astype(bfloat16)

    prepare_runtime(
        prefill_cache, decode_cache, weights, config, seq_len, rope_lut_bf16
    )

    return session_cls(
        config=config,
        seq_len=seq_len,
        weights=weights,
        tokenizer=tokenizer,
        prefill_cache=prefill_cache,
        decode_cache=decode_cache,
        rope_lut_bf16=rope_lut_bf16,
        model_variant=args.model,
    )
