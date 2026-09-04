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


if __name__ == "__main__":
    sys.exit(_main())
