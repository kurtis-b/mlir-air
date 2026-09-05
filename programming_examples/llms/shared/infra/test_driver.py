# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only tests for the shared CLI driver (shared/infra/driver.py).

`build_session` is reached only from each model's `main`, never from
`verify_adapter.py`, so `make verify` cannot see a regression here -- and the
`make run` gate needs the NPU. These tests pin the wiring on the host: which
handle is called with what, that `--run-only` loads manifests instead of
compiling, that `--compile-only` exits 0, and that the rope LUT still covers
the tokens the run will generate.

The last test is the cross-model one: each shim must pass ITS OWN modules'
handles. A copy-paste that left `qwen3_0_6b`'s compile function in another
model's shim would run and produce plausible garbage; it fails here instead.

No NPU, no air, no test framework -- see `test_decode_qkv2.py`.

    python shared/infra/test_driver.py
"""

import ast
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ml_dtypes import bfloat16

_LLMS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_LLMS))

from shared.infra import driver  # noqa: E402

MODELS = [
    "qwen3_0_6b",
    "qwen3_1_7b",
    "qwen3_4b",
    "qwen25_0_5b",
    "qwen25_3b",
    "qwen25_1_5b",
]


class _Cache:
    """Stand-in for KernelCache: records its name and manifest loads."""

    def __init__(self, name, verbose=False, profiler=None):
        self.name = name
        self.verbose = verbose
        self.profiler = profiler
        self.manifest_loads = 0

    def load_manifest(self):
        self.manifest_loads += 1


def _args(**over):
    base = dict(
        verbose=False,
        profile=False,
        run_only=False,
        compile_only=False,
        cpu_attn=False,
        model="0.6B",
        n_tokens=16,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _harness(**over):
    """Fake handles that record every call the driver makes."""
    log = []
    cfg = object()

    def rope(config=None, seq_len=None):
        log.append(("rope", config, seq_len))
        return np.zeros((4, 4), dtype=np.float32)

    h = dict(
        config_cls=lambda: cfg,
        session_cls=lambda **kw: ("session", kw),
        model_choices={"0.6B": "Qwen/Qwen3-0.6B"},
        load_weights=lambda mid, config=None: log.append(("weights", mid, config))
        or "W",
        generate_rope_lut=rope,
        compile_all_kernels=lambda *a, **k: log.append(("prefill", a, k)),
        compile_decode_kernels=lambda *a, **k: log.append(("decode", a, k)),
        prepare_runtime=lambda *a: log.append(("runtime", a)),
    )
    h.update(over)
    return h, log, cfg


def _run(args, handles):
    """Call the driver with KernelCache/Profiler/transformers stubbed out."""
    real_cache, real_prof = driver.KernelCache, driver.Profiler
    caches = []

    def make_cache(*a, **k):
        c = _Cache(*a, **k)
        caches.append(c)
        return c

    driver.KernelCache = make_cache
    driver.Profiler = lambda enabled=False: ("profiler", enabled)
    tok = SimpleNamespace(
        from_pretrained=lambda mid: ("tokenizer", mid),
    )
    sys.modules["transformers"] = SimpleNamespace(AutoTokenizer=tok)
    try:
        return driver.build_session(args, **handles), caches
    finally:
        driver.KernelCache, driver.Profiler = real_cache, real_prof
        sys.modules.pop("transformers", None)


def test_compiles_then_builds_the_session():
    handles, log, cfg = _harness()
    (kind, kw), caches = _run(_args(), handles)
    assert kind == "session", kind
    assert [c.name for c in caches] == [
        "prefill_kernel_cache",
        "decode_kernel_cache",
    ], [c.name for c in caches]
    steps = [e[0] for e in log]
    assert steps == ["prefill", "decode", "weights", "rope", "runtime"], steps
    # the compile calls get this model's cache, config and seq_len
    assert log[0][1] == (caches[0], cfg, 2048), log[0]
    assert log[0][2] == {"verbose": False, "cpu_attn": False}, log[0]
    assert log[1][1] == (caches[1], cfg), log[1]
    # the model id is resolved through the map, not passed through raw
    assert log[2][1] == "Qwen/Qwen3-0.6B", log[2]
    assert kw["weights"] == "W" and kw["model_variant"] == "0.6B", kw
    assert kw["tokenizer"] == ("tokenizer", "Qwen/Qwen3-0.6B"), kw
    assert kw["prefill_cache"] is caches[0] and kw["decode_cache"] is caches[1]
    assert kw["seq_len"] == 2048 and kw["config"] is cfg


def test_run_only_loads_manifests_instead_of_compiling():
    handles, log, _ = _harness()
    _, caches = _run(_args(run_only=True), handles)
    assert [e[0] for e in log] == ["weights", "rope", "runtime"], log
    assert [c.manifest_loads for c in caches] == [1, 1]


def test_compile_only_exits_zero():
    handles, log, _ = _harness()
    try:
        _run(_args(compile_only=True), handles)
    except SystemExit as e:
        assert e.code == 0, e.code
        assert [e[0] for e in log] == ["prefill", "decode"], log
        return
    raise AssertionError("--compile-only must exit, not return a Session")


def test_rope_lut_covers_the_generated_tokens_in_bf16():
    handles, log, cfg = _harness()
    (_, kw), _ = _run(_args(n_tokens=32), handles)
    rope = [e for e in log if e[0] == "rope"][0]
    assert rope[1] is cfg and rope[2] == 2048 + 32, rope
    assert kw["rope_lut_bf16"].dtype == bfloat16, kw["rope_lut_bf16"].dtype


def test_unknown_model_falls_through_to_the_raw_id():
    handles, log, _ = _harness()
    _run(_args(model="some/other-repo"), handles)
    assert [e for e in log if e[0] == "weights"][0][1] == "some/other-repo"


# Each injected handle, the name the shim must bind it to, and the module that
# name must come from ("" = defined in the inference file itself). Checking the
# module alone is vacuous -- a file only imports its own modules, so every
# mis-wire inside one shim passes that test. The NAME is what discriminates.
_WIRING = {
    "config_cls": ("LlamaConfig", "{m}_weights"),
    "session_cls": ("Session", ""),
    "model_choices": ("MODEL_CHOICES", ""),
    "load_weights": ("load_weights", "{m}_weights"),
    "generate_rope_lut": ("generate_rope_lut", "{m}_weights"),
    "compile_all_kernels": ("compile_all_kernels", "{m}_prefill"),
    "compile_decode_kernels": ("compile_decode_kernels", "{m}_decode"),
    "prepare_runtime": ("prepare_runtime", ""),
}


def test_each_shim_wires_the_right_handle_from_the_right_module():
    """Wiring check: `compile_all_kernels=generate_rope_lut` must fail here."""
    for m in MODELS:
        path = _LLMS / m / f"{m}_inference.py"
        tree = ast.parse(path.read_text())
        fn = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "build_session"
        ]
        assert len(fn) == 1, m
        call = fn[0].body[-1].value
        assert isinstance(call, ast.Call), m
        assert {k.arg for k in call.keywords} == set(_WIRING), (
            m,
            sorted(k.arg for k in call.keywords),
        )
        origin = {}
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                for a in n.names:
                    origin[a.asname or a.name] = n.module or ""
        for k in call.keywords:
            want_name, want_mod = _WIRING[k.arg]
            got = getattr(k.value, "id", None)
            assert got == want_name, (m, k.arg, "bound to", got)
            assert origin.get(got, "") == want_mod.format(m=m), (
                m,
                k.arg,
                origin.get(got, ""),
            )


def _main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS {name}")
        except Exception:
            failed.append(name)
            print(f"  FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


# --- the REPL trio (slice 2) ----------------------------------------------
# `make run` reaches run_once and tokenize_prompt, so those have a device gate.
# `repl_loop` is interactive and no device gate can reach it: these are its only
# guard, which is why they cover /quit, EOF, blank input and Ctrl-C separately.

REPL_MODELS = [
    "llama32_3b",
    "qwen25_0_5b",
    "qwen25_1_5b",
    "qwen25_3b",
    "qwen3_0_6b",
    "qwen3_1_7b",
    "qwen3_4b",
]


class _Prof:
    def time_cpu(self, _name):
        import contextlib

        return contextlib.nullcontext()


def _session(variant="base", seq_len=8, eos=99):
    tok = SimpleNamespace(
        encode=lambda t: [1, 2, 3],
        apply_chat_template=lambda m, tokenize, add_generation_prompt: "CHAT:"
        + m[0]["content"],
        eos_token_id=eos,
    )
    return SimpleNamespace(
        model_variant=variant,
        tokenizer=tok,
        seq_len=seq_len,
        weights="W",
        config="C",
        prefill_cache=SimpleNamespace(profiler=_Prof()),
        decode_cache="DC",
        rope_lut_bf16="ROPE",
    )


def test_tokenize_prompt_applies_the_chat_template_only_for_instruct():
    base = _session()
    assert driver.tokenize_prompt(base, "hi") == [1, 2, 3]
    chat = _session(variant="instruct")
    seen = {}

    def _encode(text):
        seen["text"] = text
        return [7]

    chat.tokenizer.encode = _encode
    assert driver.tokenize_prompt(chat, "hi") == [7]
    assert seen["text"] == "CHAT:hi", seen


def test_run_once_pads_to_seq_len_and_passes_the_models_generate():
    got = {}

    def fake_generate(tokens, w, c, pc, dc, rope, **kw):
        got["tokens"] = tokens
        got["kw"] = kw
        return ["tok"]

    s = _session(seq_len=6, eos=99)
    out, plen = driver.run_once(s, "hi", generate=fake_generate, n_tokens=4)
    assert out == ["tok"] and plen == 3, (out, plen)
    # padded to seq_len with eos, and the UNPADDED length is what is reported
    assert got["tokens"] == [1, 2, 3, 99, 99, 99], got["tokens"]
    assert got["kw"]["n_tokens"] == 4 and got["kw"]["ttft_start"] > 0


def test_run_once_does_not_pad_a_prompt_already_at_seq_len():
    got = {}
    s = _session(seq_len=3)
    driver.run_once(
        s,
        "hi",
        generate=lambda t, *a, **k: got.setdefault("tokens", t),
        n_tokens=1,
    )
    assert got["tokens"] == [1, 2, 3], got["tokens"]


def _repl(inputs, generate):
    """Drive repl_loop with a scripted stdin."""
    import builtins

    it = iter(inputs)

    def fake_input(_prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError

    real = builtins.input
    builtins.input = fake_input
    try:
        driver.repl_loop(
            _session(), SimpleNamespace(n_tokens=4, cpu_attn=False), generate=generate
        )
    finally:
        builtins.input = real


def test_repl_loop_runs_a_prompt_then_leaves_on_quit():
    calls = []
    _repl(["hello", "/quit", "never reached"], lambda *a, **k: calls.append(k))
    assert len(calls) == 1, calls
    assert calls[0]["on_token"] is not None and calls[0]["n_tokens"] == 4


def test_repl_loop_skips_blank_input_and_leaves_on_eof():
    calls = []
    _repl(["", "   "], lambda *a, **k: calls.append(k))  # then EOF
    assert calls == [], calls


def test_repl_loop_survives_ctrl_c_during_generation():
    calls = []

    def boom(*a, **k):
        calls.append(k)
        if len(calls) == 1:
            raise KeyboardInterrupt
        return []

    try:
        _repl(["one", "two", "/quit"], boom)
    except KeyboardInterrupt:
        # Caught here on purpose: if repl_loop stops swallowing this, the escape
        # would otherwise kill the whole harness with rc=130 and report no test.
        raise AssertionError("Ctrl-C during generation escaped repl_loop")
    assert len(calls) == 2, "an interrupted generation must not end the session"


def test_each_shim_binds_its_own_generate():
    """`generate` is model-local; a shim must pass its own, not import another's."""
    for m in REPL_MODELS:
        tree = ast.parse((_LLMS / m / f"{m}_inference.py").read_text())
        for name in ("run_once", "repl_loop"):
            fn = [
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == name
            ]
            assert len(fn) == 1, (m, name)
            call = [n for n in ast.walk(fn[0]) if isinstance(n, ast.Call)]
            assert call, (m, name)
            kw = {k.arg: getattr(k.value, "id", None) for k in call[0].keywords}
            assert kw.get("generate") == "generate", (m, name, kw)
        # and `generate` must be defined in this file, not imported from elsewhere
        local = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        assert "generate" in local, m


# --- the generate path (slice 3) ------------------------------------------
# `make run` DOES reach this one, and its banner is printed output, so the
# device comparison checks the label wiring directly. These cover the parts a
# 16-token run does not exercise: EOS stop, streaming deltas, and the per-model
# handles.

GEN_MODELS = {
    "qwen25_0_5b": "Qwen2.5",
    "qwen25_1_5b": "Qwen2.5",
    "qwen25_3b": "Qwen2.5",
    "qwen3_0_6b": "Qwen3",
    "qwen3_1_7b": "Qwen3",
    "qwen3_4b": "Qwen3",
}


def _gen_env(decode_tokens, eos=99, eot=-1):
    """Fakes for generate(): prefill returns token 5, decode replays a script."""
    seen = {"decode_positions": []}

    def prefill(tokens, w, c, pc, dc, rope, max_seq, **kw):
        seen["prefill"] = dict(n=len(tokens), max_seq=max_seq, kw=kw)
        return 5, None, "K", "V", 3

    it = iter(decode_tokens)

    def step(x, w, c, dc, rope, k, v, pos):
        seen["decode_positions"].append(pos)
        return next(it), None

    tok = SimpleNamespace(
        eos_token_id=eos,
        convert_tokens_to_ids=lambda s: eot,
        decode=lambda ids, skip_special_tokens=True: "".join(f"<{i}>" for i in ids),
    )
    weights = SimpleNamespace(embed_table={i: np.zeros(2) for i in range(200)})
    caches = SimpleNamespace(profiler=_Prof())
    caches.profiler.enabled = False
    return seen, tok, weights, caches, prefill, step


def _run_generate(decode_tokens, *, label="Qwen3", on_token=None, n_tokens=3, eos=99):
    seen, tok, weights, caches, prefill, step = _gen_env(decode_tokens, eos=eos)
    out = driver.generate(
        [1, 2, 3],
        weights,
        "CFG",
        caches,
        caches,
        "ROPE",
        tok,
        n_tokens=n_tokens,
        on_token=on_token,
        run_npu_prefill=prefill,
        run_npu_decode_step=step,
        label=label,
    )
    return out, seen


def test_generate_prints_the_models_label_in_the_banner(capsys=None):
    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _run_generate([10, 11, 12], label="Qwen2.5")
    text = buf.getvalue()
    assert "Qwen2.5 Inference: prompt_len=3, n_tokens=3" in text, text[:200]
    assert "Qwen3 Inference" not in text


def test_generate_returns_prefill_token_then_the_decoded_ones():
    out, seen = _run_generate([10, 11, 12])
    assert out == [5, 10, 11, 12], out
    assert seen["prefill"]["max_seq"] == 6, seen["prefill"]
    # decode walks forward from the prompt length the prefill reported
    assert seen["decode_positions"] == [3, 4, 5], seen["decode_positions"]


def test_generate_stops_at_eos_instead_of_running_the_full_budget():
    out, seen = _run_generate([10, 99, 12], n_tokens=3, eos=99)
    assert out == [5, 10, 99], out
    assert len(seen["decode_positions"]) == 2, seen["decode_positions"]


def test_streaming_emits_deltas_not_the_whole_string_each_time():
    deltas = []
    out, _ = _run_generate([10, 11], n_tokens=2, on_token=lambda t, d: deltas.append(d))
    # one callback for the prefill token, then one per decoded token
    assert deltas == ["<5>", "<10>", "<11>"], deltas
    assert out == [5, 10, 11]


def test_delta_text_advances_the_cursor():
    state = driver.StreamState()
    tok = SimpleNamespace(
        decode=lambda ids, skip_special_tokens=True: "abc"[: len(ids)]
    )
    assert driver.delta_text(tok, [1], state) == "a"
    assert driver.delta_text(tok, [1, 2], state) == "b"
    assert state.printed_len == 2


def test_each_shim_binds_its_own_npu_steps_and_the_right_label():
    """The label is printed output, so a wrong one is a visible regression."""
    for m, label in GEN_MODELS.items():
        tree = ast.parse((_LLMS / m / f"{m}_inference.py").read_text())
        fn = [
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "generate"
        ]
        assert len(fn) == 1, m
        call = [n for n in ast.walk(fn[0]) if isinstance(n, ast.Call)][0]
        kw = {k.arg: k.value for k in call.keywords}
        # `arg=None` is the `**kw` forward; without it the shim would silently
        # drop the caller's n_tokens/on_token/ttft_start.
        assert None in kw, (m, "shim must forward **kw")
        assert set(kw) - {None} == {
            "run_npu_prefill",
            "run_npu_decode_step",
            "label",
        }, (m, sorted(str(k) for k in kw))
        assert getattr(kw["run_npu_prefill"], "id", None) == "run_npu_prefill", m
        assert (
            getattr(kw["run_npu_decode_step"], "id", None) == "run_npu_decode_step"
        ), m
        assert getattr(kw["label"], "value", None) == label, (m, ast.dump(kw["label"]))
        local = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        assert {"run_npu_prefill", "run_npu_decode_step"} <= local, m


# --- the prefill path (slice 4) -------------------------------------------

PREFILL_BLOCK = {
    "qwen25_0_5b": "run_transformer_block_qwen25",
    "qwen25_1_5b": "run_transformer_block_qwen25",
    "qwen25_3b": "run_transformer_block_qwen25",
    "qwen3_0_6b": "run_transformer_block_qwen3",
    "qwen3_1_7b": "run_transformer_block_qwen3",
    "qwen3_4b": "run_transformer_block_qwen3",
}


class _PrefProf(_Prof):
    def start_layer(self):
        return 0

    def end_layer(self, idx, t0):
        pass


def _prefill_env(n_layers=2, seq=4, eos=99):
    cfg = SimpleNamespace(n_layers=n_layers, emb_dim=8, n_kv_heads=2, head_dim=3)
    weights = SimpleNamespace(
        embed_table=np.zeros((200, 8), dtype=np.float32),
        layers=[f"L{i}" for i in range(n_layers)],
        final_norm="FN",
        lm_head=np.zeros((7, 8)),
    )
    seen = {"layers": [], "rms": []}

    def block(
        x, layer_w, rope, config, cache, layer_idx=None, cpu_attn=None, verbose=None
    ):
        seen["layers"].append((layer_w, layer_idx, cpu_attn))
        inter = {
            "k_roped": np.full((seq, 2, 3), layer_idx + 1, dtype=np.float32).reshape(
                seq, 6
            ),
            "v": np.full((seq, 2, 3), 100 + layer_idx, dtype=np.float32).reshape(
                seq, 6
            ),
        }
        return x, inter

    def rms(h, w, eps=None):
        seen["rms"].append((w, eps))
        return np.ones((1, 8), dtype=np.float32)

    def lm_head(dc, w, normed, vocab):
        seen["lm"] = (vocab, normed.dtype)
        out = np.zeros(vocab)
        out[3] = 1.0
        return out

    cache = SimpleNamespace(profiler=_PrefProf())
    tok = SimpleNamespace(eos_token_id=eos)
    return cfg, weights, cache, tok, block, rms, lm_head, seen


def _run_prefill(eps=1e-6, seq=4, eos=99, pad=0):
    tokens = [1, 2, 3, 4][:seq] + [eos] * pad
    # the block sees the PADDED length, as it does in the real driver
    cfg, weights, cache, tok, block, rms, lm_head, seen = _prefill_env(
        seq=len(tokens), eos=eos
    )
    out = driver.run_npu_prefill(
        tokens,
        weights,
        cfg,
        cache,
        cache,
        "ROPE",
        len(tokens) + 4,
        tok,
        quiet=True,
        run_transformer_block=block,
        rms_norm=rms,
        run_lm_head=lm_head,
        eps=eps,
    )
    return out, seen


def test_prefill_returns_argmax_token_and_the_filled_caches():
    (token, logits, k_cache, v_cache, prompt_len), seen = _run_prefill()
    assert token == 3, token
    assert prompt_len == 4, prompt_len
    assert k_cache.shape == (2, 2, 8, 3) and k_cache.dtype == bfloat16, k_cache.shape
    # layer 0 wrote 1s into k and 100s into v; layer 1 wrote 2s and 101s
    assert float(k_cache[0, 0, 0, 0]) == 1.0 and float(k_cache[1, 0, 0, 0]) == 2.0
    assert float(v_cache[0, 0, 0, 0]) == 100.0 and float(v_cache[1, 0, 0, 0]) == 101.0
    assert [i for _, i, _ in seen["layers"]] == [0, 1], seen["layers"]
    assert [w for w, _, _ in seen["layers"]] == ["L0", "L1"], seen["layers"]


def test_prefill_passes_the_models_eps_through_verbatim():
    _, seen = _run_prefill(eps=1e-5)
    assert seen["rms"] == [("FN", 1e-5)], seen["rms"]


def test_prompt_len_ignores_eos_padding():
    (_, _, _, _, prompt_len), _ = _run_prefill(seq=4, pad=6)
    assert prompt_len == 4, prompt_len


def test_each_shim_binds_its_own_transformer_block_and_eps():
    for m, block in PREFILL_BLOCK.items():
        tree = ast.parse((_LLMS / m / f"{m}_inference.py").read_text())
        fn = [
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "run_npu_prefill"
        ]
        assert len(fn) == 1, m
        call = [n for n in ast.walk(fn[0]) if isinstance(n, ast.Call)][0]
        kw = {k.arg: k.value for k in call.keywords}
        assert None in kw, (m, "shim must forward **kw")
        assert set(kw) - {None} == {
            "run_transformer_block",
            "rms_norm",
            "run_lm_head",
            "eps",
        }, (m, sorted(str(k) for k in kw))
        assert getattr(kw["run_transformer_block"], "id", None) == block, (m, block)
        assert getattr(kw["eps"], "id", None) == "EPS", m
        assert getattr(kw["run_lm_head"], "id", None) == "_run_lm_head", m


# --- the two remaining shared helpers (slice 5) ---------------------------

LM_MODELS = [
    "qwen25_0_5b",
    "qwen25_1_5b",
    "qwen25_3b",
    "qwen3_0_6b",
    "qwen3_1_7b",
    "qwen3_4b",
]


def test_free_original_weight_numpy_keeps_shape_and_drops_storage():
    layer = SimpleNamespace(
        wq=np.ones((4, 4), dtype=bfloat16),
        wk=np.ones((2, 4), dtype=bfloat16),
        w_down=np.ones((4, 8), dtype=bfloat16),
        w_up=None,  # absent attributes must be tolerated
        # size-1 and left alone by the `size > 1` guard. It must be one of the
        # attributes the function actually iterates, and shape (1,) not 0-d: a
        # 0-d array broadcasts to identical strides, so neither an unlisted
        # attribute nor a 0-d one can tell whether the guard is there.
        w_gate=np.ones((1,), dtype=bfloat16),
    )
    gate_before = layer.w_gate
    weights = SimpleNamespace(layers=[layer])
    driver.free_original_weight_numpy(weights, SimpleNamespace(n_layers=1))
    for attr, shape in (("wq", (4, 4)), ("wk", (2, 4)), ("w_down", (4, 8))):
        a = getattr(layer, attr)
        assert a.shape == shape, (attr, a.shape)
        assert a.strides == (0, 0), (attr, a.strides)  # zero-stride broadcast
    assert layer.w_up is None
    # the size-1 array must be the SAME object, not a broadcast of it
    assert layer.w_gate is gate_before, "size-1 arrays must be left alone"
    assert layer.w_gate.strides == gate_before.strides


def test_run_lm_head_assembles_logits_from_every_partition():
    seen = {}

    class _DC:
        def load_and_run(self, name, backend, *inputs, **kw):
            seen["name"] = name
            seen["backend"] = backend
            seen["kw"] = kw
            seen["n_inputs"] = len(inputs)
            # two partitions -> results at indices 2 and 4
            return {2: np.array([1.0, 2.0]), 4: np.array([3.0, 4.0])}

    weights = SimpleNamespace(_lm_weight_parts_gemv=["W0", "W1"])
    out = driver.run_lm_head(
        _DC(),
        weights,
        np.ones((1, 3), dtype=bfloat16),
        4,
        lm_gemv_backend=lambda: "BACKEND",
        n_partitions=2,
        n_part=2,
    )
    assert list(out) == [1.0, 2.0, 3.0, 4.0], out
    assert out.dtype == np.float32
    assert seen["name"] == "lm_head_gemv" and seen["backend"] == "BACKEND"
    assert seen["n_inputs"] == 5  # x + (weight, zeros) per partition
    assert seen["kw"]["static_input_indices"] == {1, 3}
    assert seen["kw"]["intermediate_indices"] == {2, 4}


def test_run_lm_head_truncates_the_last_partition_to_vocab_size():
    class _DC:
        def load_and_run(self, name, backend, *inputs, **kw):
            return {2: np.array([1.0, 2.0]), 4: np.array([3.0, 9.0])}

    out = driver.run_lm_head(
        _DC(),
        SimpleNamespace(_lm_weight_parts_gemv=["W0", "W1"]),
        np.ones((1, 3), dtype=bfloat16),
        3,  # vocab 3, so the last partition contributes one element, not two
        lm_gemv_backend=lambda: "B",
        n_partitions=2,
        n_part=2,
    )
    assert list(out) == [1.0, 2.0, 3.0], out


def test_each_shim_binds_its_own_lm_head_partitioning():
    for m in LM_MODELS:
        tree = ast.parse((_LLMS / m / f"{m}_inference.py").read_text())
        fn = [
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_run_lm_head"
        ]
        assert len(fn) == 1, m
        call = [n for n in ast.walk(fn[0]) if isinstance(n, ast.Call)][0]
        kw = {k.arg: getattr(k.value, "id", None) for k in call.keywords}
        assert kw == {
            "lm_gemv_backend": "_lm_gemv_backend",
            "n_partitions": "_LM_N_PARTITIONS",
            "n_part": "_LM_N_PART",
        }, (m, kw)
        origin = {}
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                for a in n.names:
                    origin[a.asname or a.name] = n.module or ""
        # the partitioning must come from THIS model's decode module
        for handle in ("_lm_gemv_backend", "_LM_N_PARTITIONS", "_LM_N_PART"):
            assert origin.get(handle) == f"{m}_decode", (m, handle, origin.get(handle))


if __name__ == "__main__":
    sys.exit(_main())
