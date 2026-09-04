# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared CLI driver steps for the bf16 inference examples.

`build_session` was one implementation copied into six `<model>_inference.py`
files (346 lines; the only textual difference was black's wrapping of one call).
It is kept here once, with the per-model handles passed in — the injection
contract recorded for goal 6c: the config and Session classes, the model-id
map, the two compile callables, and this model's weights/rope/runtime helpers.

Gating a change here — the recipe, because the obvious gate does not work.
`build_session` is reached from each model's CLI `main`, NOT from
`verify_adapter.py`, so `make verify` passes whatever this function does. The
signal is `make run N_TOKENS=16`, whose greedy (argmax) decode is deterministic:
capture its output before and after with timings and throughput stripped, and
the generated token ids must be unchanged.

Measured on 2026-09-04, when the six copies were consolidated into this file:

    qwen3_0_6b   before == after   first token 25   rc=0
    qwen3_1_7b   before == after                    rc=0
    qwen25_0_5b  before == after                    rc=0

captured under Turbo, baseline taken TWICE first (identical, so the comparison
is a real signal rather than a coincidence) and once after the change. The other
three callers — qwen3_4b, qwen25_3b, qwen25_1_5b — cannot `make run` on the
development machine (empty kernel-cache manifests, pre-existing and unrelated);
their guard is the wiring test in `test_driver.py`, not a device run.
"""

import sys
import time

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


def tokenize_prompt(session, prompt_text):
    """Encode a prompt, applying the chat template for the instruct variant.

    Carried verbatim from the seven copies. llama32_3b's copy differed only in
    having no type annotations; the body was identical.
    """
    if session.model_variant == "instruct":
        messages = [{"role": "user", "content": prompt_text}]
        chat_text = session.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return session.tokenizer.encode(chat_text)
    return session.tokenizer.encode(prompt_text)


def run_once(
    session,
    prompt_text,
    *,
    generate,
    n_tokens,
    profile=False,
    cpu_attn=True,
    on_token=None,
):
    """One prompt through prefill+decode. `generate` is the model's own."""
    ttft_start = time.perf_counter()
    with session.prefill_cache.profiler.time_cpu("tokenize"):
        tokens = tokenize_prompt(session, prompt_text)
    prompt_len_actual = len(tokens)
    with session.prefill_cache.profiler.time_cpu("eos_pad"):
        if len(tokens) < session.seq_len:
            tokens = tokens + [session.tokenizer.eos_token_id] * (
                session.seq_len - len(tokens)
            )
    generated = generate(
        tokens,
        session.weights,
        session.config,
        session.prefill_cache,
        session.decode_cache,
        session.rope_lut_bf16,
        tokenizer=session.tokenizer,
        n_tokens=n_tokens,
        profile=profile,
        cpu_attn=cpu_attn,
        on_token=on_token,
        ttft_start=ttft_start,
    )
    return generated, prompt_len_actual


def repl_loop(session, args, *, generate):
    """Interactive prompt loop. Ctrl-D, Ctrl-C or /quit leaves it."""
    print("\nInteractive mode — Ctrl-D or /quit to exit.\n")

    def _cb(_tid, delta):
        sys.stdout.write(delta)
        sys.stdout.flush()

    while True:
        try:
            prompt = input("Prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not prompt:
            continue
        if prompt in ("/quit", "/exit"):
            return
        sys.stdout.write("\nResponse: ")
        sys.stdout.flush()
        try:
            run_once(
                session,
                prompt,
                generate=generate,
                n_tokens=args.n_tokens,
                profile=False,
                cpu_attn=args.cpu_attn,
                on_token=_cb,
            )
        except KeyboardInterrupt:
            print("\n[interrupted]")
            continue
        print("\n")
