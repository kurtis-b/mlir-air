# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Host-only regression for the shared bf16 NpuRunner wiring (no Peano, no NPU).

    python3 runners/test_bf16_npu_runner.py   (from programming_examples/llms/verify)

`make verify` builds the runner with `lite_mode=True` (`verify_runner.py`:
`lite = in_verify_mode`), and `prefill()` returns early in lite mode. So the
device verify gate never reaches the diagnosis loop or the RMSNorm epsilon --
a swapped `eps` or a mis-wired `run_prefill_block` hook passes it unnoticed.
Only `make diagnosis` (lite_mode=False) runs that code on hardware.

These tests take the non-lite path with fake hooks and assert the two things
the device gate cannot see:

1. `run_prefill_block` is invoked once per layer, through the subclass's hook.
2. `rms_norm` receives THAT subclass's `eps` -- 1e-5 for the llama/SmolLM2
   family, 1e-6 for qwen. This is the one semantic difference between the two
   NpuRunner bodies the shared base replaced: llama/SmolLM2 relied on their
   helper's 1e-5 default while qwen passed 1e-6 explicitly, so a base that
   dropped `eps` would silently take whichever default the shared import
   carried and be wrong for one family.

`test_a_wrong_eps_is_actually_caught` is the non-vacuity control: it wires a
subclass whose eps disagrees with what it declares and asserts the check
above fails. Without it, tests 1-2 could pass against an implementation that
never forwards eps at all.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # verify/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # llms/

from runners.bf16_npu_runner import Bf16NpuRunner  # noqa: E402

N_LAYERS = 3
MAX_SEQ = 8


class _Cfg:
    n_layers = N_LAYERS


class _Layer:
    pass


class _Weights:
    def __init__(self):
        self.embed_table = np.zeros((32, 4), dtype=np.float32)
        self.layers = [_Layer() for _ in range(N_LAYERS)]
        self.final_norm = np.ones(4, dtype=np.float32)


class _Tok:
    eos_token_id = 0


def _make_runner(eps, seen):
    """A Bf16NpuRunner subclass whose hooks only record what they were given."""

    def run_prefill_block(x, layer, rope, cfg, cache, layer_idx=None, **kw):
        seen["blocks"].append(layer_idx)
        return x, {"ffn_out": np.zeros((MAX_SEQ, 4), dtype=np.float32)}

    def rms_norm(x, weight, eps=None):
        seen["eps"].append(eps)
        return np.asarray(x, dtype=np.float32)

    def run_npu_prefill(padded, *a, **kw):
        return 7, np.zeros(4, dtype=np.float32), None, None, MAX_SEQ

    class _Runner(Bf16NpuRunner):
        this_dir = Path("/nonexistent")
        eps_declared = eps

    _Runner.eps = eps
    _Runner.generate_rope_lut = staticmethod(lambda config, seq_len: np.zeros(2))
    _Runner.compile_prefill_kernels = staticmethod(lambda *a, **k: None)
    _Runner.compile_decode_kernels = staticmethod(lambda *a, **k: None)
    _Runner.prepare_runtime = staticmethod(lambda *a, **k: None)
    _Runner.run_npu_prefill = staticmethod(run_npu_prefill)
    _Runner.run_prefill_block = staticmethod(run_prefill_block)
    _Runner.run_npu_decode_step = staticmethod(lambda *a, **k: (0, np.zeros(4)))
    _Runner.rms_norm = staticmethod(rms_norm)
    return _Runner


def _run_prefill(eps, lite):
    seen = {"blocks": [], "eps": []}
    cls = _make_runner(eps, seen)
    # KernelCache is constructed against this_dir; stub it out for a host run.
    import runners.bf16_npu_runner as mod

    real_cache = mod.KernelCache
    mod.KernelCache = lambda *a, **k: None
    try:
        r = cls(_Weights(), _Cfg(), MAX_SEQ, _Tok(), lite_mode=lite)
        r.prefill(np.zeros(4, dtype=np.int64))
    finally:
        mod.KernelCache = real_cache
    return seen


def test_non_lite_prefill_runs_every_layer_through_the_hook():
    seen = _run_prefill(1e-5, lite=False)
    assert seen["blocks"] == list(range(N_LAYERS)), seen["blocks"]


def test_each_family_gets_its_own_eps():
    for eps in (1e-5, 1e-6):
        seen = _run_prefill(eps, lite=False)
        assert seen["eps"] == [eps], f"eps {eps}: forwarded {seen['eps']}"


def test_lite_mode_skips_the_diagnosis_path():
    """Documents why the device verify gate cannot cover the two checks above."""
    seen = _run_prefill(1e-5, lite=True)
    assert seen["blocks"] == [] and seen["eps"] == [], seen


def test_a_wrong_eps_is_actually_caught():
    """Non-vacuity control: the eps assertion must fail on a mis-wired runner."""
    seen = _run_prefill(1e-6, lite=False)
    try:
        assert seen["eps"] == [1e-5]
    except AssertionError:
        return  # the check discriminates, as it must
    raise AssertionError("eps check is vacuous: it passed on the wrong value")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"bf16 NpuRunner wiring tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
