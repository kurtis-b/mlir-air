# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared CLI driver steps for the bf16 inference examples.

Each function here was one implementation copied into every model's
`<model>_inference.py`; it lives here once, with the genuinely per-model handles
passed in — goal 6c's injection contract.

    build_session      6 copies, 346 lines   config/Session classes, model-id map,
                                             two compile callables, weights/rope/
                                             runtime helpers
    run_once           7 copies, 264 lines   this model's `generate`
    repl_loop          7 copies, 314 lines   this model's `generate`
    tokenize_prompt    7 copies,  72 lines   nothing (fully generic)
    generate           6 copies, 570 lines   this model's run_npu_prefill and
                                             run_npu_decode_step, plus `label`
    StreamState        6 copies,  18 lines   nothing
    delta_text         6 copies,  30 lines   nothing
    run_npu_prefill    6 copies, 474 lines   this model's transformer block,
                                             rms_norm, LM-head and eps
    run_lm_head        6 copies, 126 lines   this model's vocabulary split
                                             (_LM_N_PARTITIONS/_LM_N_PART) and
                                             backend selector
    free_original_...  5 copies, 100 lines   nothing (qwen25_3b never had it)

`generate`'s six copies hashed as two bodies, but the two differ by exactly one
line — the banner string — so `label` is a parameter and the rest is shared.
`run_npu_prefill` hashed as THREE bodies and was described here as "genuinely
per-model"; that was wrong, and reading the diff is what showed it: the three
differ only in a docstring, in the local name given to `inter["v"]`, and in
which `run_transformer_block_*` they call. One handle, not three implementations.

`run_npu_decode_step` really does stay per-model, and for a reason worth keeping
here: qwen3_0_6b/1_7b's `run_decode_block` returns a bare `x`, while the other
four return `(out, inter)`. That is an API difference, not a spelling, so
sharing it needs the callee normalised first — a separate change.

Gating a change here — and WHICH gate depends on the function.

`build_session`, `run_once`, `repl_loop`, `tokenize_prompt` and `generate` hang
off each model's CLI `main` and are NOT reached from `verify_adapter.py`, so
`make verify` passes whatever they do. Their signal is `make run N_TOKENS=16`,
whose greedy (argmax) decode is deterministic: capture its output before and
after with timings and throughput stripped; the generated token ids must be
unchanged. Baseline twice before trusting it.

`run_npu_prefill` is different and must be gated differently: every model's
`verify_adapter.py` imports it and `verify/runners/bf16_npu_runner.py` calls it
from `prefill()`. So `make verify` DOES reach it — and is the stronger check
there, because the top-k token-set gate sees changes in non-top-1 logits that an
argmax comparison cannot. Run both for this function.

Measured on 2026-09-04, per consolidation, all under Turbo, all rc=0:

    build_session  (devq 913/914 baseline, 915 after)   3/3 models identical
    REPL trio      (devq 916 baseline,     917 after)   3/3 models identical
    generate path  (devq 918 baseline,     919 after)   3/3 models identical
    run_npu_prefill(devq 920 baseline,     921 after)   3/3 models identical
                   (devq 922 before,       923 after)   `make verify` PASS both
                                                        sides, topk_passed 2 /
                                                        topk_failed 0, on
                                                        qwen3_0_6b AND
                                                        qwen25_0_5b — one per
                                                        config family, so both
                                                        injected transformer
                                                        blocks are covered
    lm_head + free-  (devq 924 baseline,     926 after)   3/3 models identical
    weights helpers  (devq 925 before,       927 after)   `make verify` PASS both
                                                        sides on both families

Each baseline reproduced its predecessor byte-for-byte (913 == 914 == 916 == 918
== 920),
so an unchanged result is a real signal and not a fixture that never varies. The
`generate` banner is printed output, so that run also checks the `label` wiring
directly: 919 shows `Qwen3 Inference:` for the qwen3 models and `Qwen2.5
Inference:` for qwen25_0_5b.

The three models are qwen3_0_6b, qwen3_1_7b and qwen25_0_5b — the only ones that
`make run` on the development machine; qwen3_4b, qwen25_3b, qwen25_1_5b and
llama32_3b have empty kernel-cache manifests (pre-existing, unrelated), so their
guard is the wiring test in `test_driver.py`, not a device run. Note also that
`make run` drives `run_once` and `tokenize_prompt` but NOT `repl_loop`, which is
interactive: the REPL's guard is its host tests, which script stdin.
"""

import sys
import time

import numpy as np
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


class StreamState:
    """Cursor into the decoded text, so streaming emits deltas not repeats."""

    def __init__(self) -> None:
        self.printed_len: int = 0


def delta_text(tokenizer, ids, state):
    """The text `ids` adds beyond what has already been streamed."""
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    delta = decoded[state.printed_len :]
    state.printed_len = len(decoded)
    return delta


def generate(
    prompt_tokens,
    weights,
    config,
    prefill_cache,
    decode_cache,
    rope_lut_bf16,
    tokenizer,
    n_tokens=10,
    profile=False,
    cpu_attn=True,
    on_token=None,
    ttft_start=None,
    *,
    run_npu_prefill,
    run_npu_decode_step,
    label,
):
    """NPU prefill then greedy decode. `label` only names the model in the banner.

    `run_npu_prefill` and `run_npu_decode_step` are this model's own — they are
    genuinely per-model (3 and 2 distinct implementations across the six), which
    is why they are injected rather than shared.
    """
    seq_len = len(prompt_tokens)
    max_seq = seq_len + n_tokens
    streaming = on_token is not None
    if ttft_start is None:
        ttft_start = time.perf_counter()

    if not streaming:
        print(f"\n{'='*60}")
        print(f"{label} Inference: prompt_len={seq_len}, n_tokens={n_tokens}")
        print(f"{'='*60}\n")

    prefill_token, _logits, k_cache, v_cache, prompt_len = run_npu_prefill(
        prompt_tokens,
        weights,
        config,
        prefill_cache,
        decode_cache,
        rope_lut_bf16,
        max_seq,
        tokenizer=tokenizer,
        cpu_attn=cpu_attn,
        profile=profile,
        quiet=True,
    )

    ttft = time.perf_counter() - ttft_start
    if not streaming:
        print(f"Time to first token (TTFT): {ttft:.2f}s. First token: {prefill_token}")

    generated_tokens = [prefill_token]
    current_pos = prompt_len
    x_decode = weights.embed_table[prefill_token].astype(bfloat16)

    stream_state = StreamState() if streaming else None
    if streaming:
        on_token(prefill_token, delta_text(tokenizer, generated_tokens, stream_state))

    if not streaming:
        print(f"\nDecoding {n_tokens} tokens...")
    t_dec = time.time()

    eos_ids = {tokenizer.eos_token_id}
    eot = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(eot, int) and eot >= 0:
        eos_ids.add(eot)

    for _ in range(n_tokens):
        next_token, _ = run_npu_decode_step(
            x_decode,
            weights,
            config,
            decode_cache,
            rope_lut_bf16,
            k_cache,
            v_cache,
            current_pos,
        )
        generated_tokens.append(next_token)
        current_pos += 1
        with decode_cache.profiler.time_cpu("embed_lookup"):
            x_decode = weights.embed_table[next_token].astype(bfloat16)
        if streaming:
            on_token(next_token, delta_text(tokenizer, generated_tokens, stream_state))
        if next_token in eos_ids:
            break

    t_decode = time.time() - t_dec
    n_gen = len(generated_tokens) - 1
    if not streaming and n_gen > 0:
        print(
            f"\nGenerated {n_gen} tokens in {t_decode:.2f}s ({n_gen / t_decode:.2f} tok/s)"
        )

    if prefill_cache.profiler.enabled:
        print(f"\n{'='*60}\nPREFILL detail")
        prefill_cache.profiler.report()
    if decode_cache.profiler.enabled:
        print(f"\n{'='*60}\nDECODE detail")
        decode_cache.profiler.report()

    return generated_tokens


def run_npu_prefill(
    token_ids,
    weights,
    config,
    prefill_cache,
    decode_cache,
    rope_lut_bf16,
    max_seq,
    tokenizer,
    cpu_attn=True,
    profile=False,
    quiet=False,
    *,
    run_transformer_block,
    rms_norm,
    run_lm_head,
    eps,
):
    """NPU prefill over every layer, then the KV cache and the first token.

    Returns (prefill_token, logits_row, k_cache, v_cache, prompt_len).

    The caches hold whatever this model's transformer block produced: `k_roped`
    after the block's RoPE, and `inter["v"]` — which is the bias-added
    projection for Qwen2.5 and the raw projection for Qwen3. That difference
    lives in `run_transformer_block`, not here, which is why the six copies of
    this function differed only in the handle they called and in the local name
    they gave that value.

    `eps` is passed explicitly, never defaulted: it is per-model config (1e-6
    across all six callers today) and a shared default is exactly how a wrong
    epsilon reaches a verify path unnoticed.
    """
    seq_len = len(token_ids)
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    vocab_size = weights.lm_head.shape[0]

    k_cache = np.zeros((config.n_layers, n_kv_heads, max_seq, head_dim), dtype=bfloat16)
    v_cache = np.zeros((config.n_layers, n_kv_heads, max_seq, head_dim), dtype=bfloat16)

    with prefill_cache.profiler.time_cpu("embed_lookup"):
        x_bf16 = weights.embed_table[token_ids].astype(np.float32).astype(bfloat16)

    if not quiet:
        print(f"Running NPU prefill ({config.n_layers} layers, seq_len={seq_len})...")
    t_start = time.time()

    for layer_idx in range(config.n_layers):
        t0 = prefill_cache.profiler.start_layer()
        x_bf16, inter = run_transformer_block(
            x_bf16,
            weights.layers[layer_idx],
            rope_lut_bf16,
            config,
            prefill_cache,
            layer_idx=layer_idx,
            cpu_attn=cpu_attn,
            verbose=profile,
        )
        with prefill_cache.profiler.time_cpu("kv_cache_extract"):
            k_roped = inter["k_roped"]
            v_proj = inter["v"]
            k_cache[layer_idx, :, :seq_len, :] = (
                k_roped.astype(bfloat16)
                .reshape(seq_len, n_kv_heads, head_dim)
                .transpose(1, 0, 2)
            )
            v_cache[layer_idx, :, :seq_len, :] = (
                v_proj.astype(bfloat16)
                .reshape(seq_len, n_kv_heads, head_dim)
                .transpose(1, 0, 2)
            )
        prefill_cache.profiler.end_layer(layer_idx, t0)

    # Final RMSNorm on the prediction-position row + NPU LM-head.
    prompt_len = len([t for t in token_ids if t != tokenizer.eos_token_id])
    pred_pos = prompt_len - 1
    with prefill_cache.profiler.time_cpu("final_rms_norm"):
        last_hidden = np.asarray(x_bf16, dtype=np.float32)[pred_pos : pred_pos + 1]
        last_normed = (
            rms_norm(last_hidden, weights.final_norm, eps=eps)
            .flatten()
            .astype(bfloat16)
        )

    logits_row = run_lm_head(decode_cache, weights, last_normed, vocab_size)
    prefill_token = int(np.argmax(logits_row))

    t_prefill = time.time() - t_start
    if not quiet:
        print(f"NPU prefill done in {t_prefill:.2f}s. First token: {prefill_token}")
    return prefill_token, logits_row, k_cache, v_cache, prompt_len


def free_original_weight_numpy(weights, config):
    """Collapse host numpy originals to zero-stride broadcasts after prefill
    preload. Weights are resident in the prefill BOs and passed as static
    inputs, so only their dtype/shape metadata is read afterward."""
    import gc

    z = np.zeros((), dtype=bfloat16)
    for layer_idx in range(config.n_layers):
        lw = weights.layers[layer_idx]
        for attr in ("wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down"):
            a = getattr(lw, attr, None)
            if a is not None and getattr(a, "size", 0) > 1:
                setattr(lw, attr, np.broadcast_to(z, a.shape))
    gc.collect()


def run_lm_head(
    decode_cache,
    weights,
    x_normed_bf16,
    vocab_size,
    *,
    lm_gemv_backend,
    n_partitions,
    n_part,
    parts=None,
    kernel_name="lm_head_gemv",
):
    """The partitioned NPU LM-head GEMV.

    `n_partitions`/`n_part` are this model's vocabulary split (`_LM_N_PARTITIONS`
    and `_LM_N_PART`) and `lm_gemv_backend` its backend selector — all three live
    in `<model>_decode`, which is why they are injected rather than imported.

    `kernel_name` is the cache artifact key. It defaults to the historical
    `lm_head_gemv`; a model whose host ABI has changed versions the key (as
    `_RMS_QKV_KERNEL` does) so a stale cache cannot bind one ABI to the
    other's ELF.

    `parts` overrides the equal split with an explicit list of partition row
    counts, matching `build_lm_head_gemv_module`'s argument of the same name. A
    vocabulary that is not a whole multiple of the partition size can then end
    in a shorter tail instead of being padded up to one. Omit it and the
    behaviour is exactly the equal split, which is what every caller but
    Qwen3-0.6B uses.
    """
    if parts is None:
        parts = [n_part] * n_partitions
    parts = list(parts)
    n = len(parts)

    lm_inputs = [x_normed_bf16.flatten().astype(bfloat16)]
    out_idx = []
    for p, rows in enumerate(parts):
        lm_inputs.append(weights._lm_weight_parts_gemv[p])
        lm_inputs.append(np.zeros(rows, dtype=bfloat16))
        out_idx.append(2 + 2 * p)
    res = decode_cache.load_and_run(
        kernel_name,
        lm_gemv_backend(),
        *lm_inputs,
        output_indices=out_idx,
        static_input_indices={1 + 2 * p for p in range(n)},
        intermediate_indices={2 + 2 * p for p in range(n)},
    )
    logits = np.zeros(vocab_size, dtype=np.float32)
    # Running offset rather than p * n_part: with unequal partitions the p-th
    # one no longer starts at a multiple of anything.
    n_start = 0
    for p, rows in enumerate(parts):
        n_end = min(n_start + rows, vocab_size)
        if n_end > n_start:
            logits[n_start:n_end] = res[2 + 2 * p][: n_end - n_start].astype(np.float32)
        n_start += rows
    return logits
