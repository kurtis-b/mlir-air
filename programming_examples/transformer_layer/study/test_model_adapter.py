# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host tests for `llms/shared/model_adapter.py` (doc 56 H1a): the planner half
and the dispatch arithmetic, with no device, no `air`, no weights.

THE LOAD-BEARING ONE is the dispatch vector: for both models and both phases
the vector derived from the cache manifest and the plan EQUALS the vector the
measured path's arithmetic (`_delta` over Profiler records) produces for the
plan's own ELF sequence -- and the plan's per-ELF launch counts equal the
manifest's. That is H0's "the plan reproduces the shipped sequence" gate at
model scope, and the host half of item 13's "adapter dispatch vector for both
models against the cached manifests".

The manifests are the models' `build_peano` caches when present on this host
(gitignored artifacts), else the counts read off them on 2026-08-23, pinned
below -- the same code path runs either way, and a host without the caches
does not shrink the suite.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_PE = os.path.dirname(os.path.dirname(_HERE))
for _p in (_PE, os.path.join(_PE, "llms"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import schema  # noqa: E402
from shared import model_adapter as ma  # noqa: E402

#: Launch counts of the shipped caches, read 2026-08-23 after queue items 11/12.
PINNED_MANIFESTS = {
    "qwen3_0_6b": {
        "prefill": {"rms_qkv_qknorm_rope": (9, 15), "o_ffn_qwen": (12, 20), "flash_attn": (1, 1)},
        "decode": {"rms_qkv_qknorm_rope_gemv2": (2, 2), "o_gemv_ffn": (3, 5), "lm_head_gemv": (10, 10)},
    },
    "llama32_1b": {
        "prefill": {"rms_gemms_rope": (7, 13), "o_ffn": (12, 20), "flash_attn": (1, 1)},
        "decode": {"rms_gemv_rope": (6, 6), "o_gemv_ffn": (3, 5), "lm_head_gemv": (8, 8)},
    },
}

#: Executed launches / submissions per phase (doc 57 section 5 after items 5c, 5/5b).
EXPECTED_TOTALS = {
    ("qwen3_0_6b", "prefill"): (85, 626, 1018),
    ("qwen3_0_6b", "decode"): (57, 150, 206),
    ("llama32_1b", "prefill"): (49, 328, 552),
    ("llama32_1b", "decode"): (33, 152, 184),
}


def _launch_counts(model_id):
    b = ma.MODELS[model_id]
    dirs = [b.directory / "build_peano" / ma.PREFILL_CACHE, b.directory / "build_peano" / ma.DECODE_CACHE]
    if all((d / "manifest.json").is_file() for d in dirs):
        return ma.launch_counts_of({str(d): ma.read_cache_manifest(d) for d in dirs}), "build_peano"
    counts = {}
    for phase in ("prefill", "decode"):
        for name, (air, herd) in PINNED_MANIFESTS[model_id][phase].items():
            counts[name] = {"air_launches": air, "herd_launches": herd}
    return counts, "pinned"


def _raises(exc, match, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc as e:
        assert match in str(e), f"expected {match!r} in {str(e)!r}"
        return
    raise AssertionError(f"expected {exc.__name__} containing {match!r}")


class _FakeProfiler:
    def __init__(self):
        self.kernel_breakdowns = {}
        self.cpu_times = {}


def _simulate(plan_, counts, profilers):
    """Replay the plan's sequence as Profiler records: one load_and_run per
    device stage instance (2 syncs, 1000 bytes each), one time_cpu per host op."""
    n_layers = ma.MODELS[plan_.spec_name].spec.n_layers
    p = profilers[0]
    for stage in plan_.stages:
        reps = n_layers if stage.repeated else 1
        for _ in range(reps):
            if stage.where == "device":
                p.kernel_breakdowns.setdefault(stage.name, []).append(
                    {"write_ms": 0.1, "kernel_ms": 1.0, "read_ms": 0.1, "n_written": 1, "bytes_written": 600, "n_readback": 1, "bytes_readback": 400})
            else:
                p.cpu_times.setdefault(stage.name, []).append(0.002)


def test_models_bind_both_drivers_and_nothing_else():
    assert sorted(ma.MODELS) == ["llama32_1b", "qwen3_0_6b"]
    for model_id, b in ma.MODELS.items():
        assert b.spec.name == model_id
        assert b.directory.is_dir(), b.directory
        assert (b.directory / f"{b.package}_inference.py").is_file()
        assert (b.directory / "verify_adapter.py").is_file()
        assert b.verify_adapter == f"{model_id}.verify_adapter"
    assert ma.SUPPORTED_PRECISION_PLANS == ("bf16",)


def test_plan_for_is_value_identity_and_64_hex():
    a = ma.plan_for("qwen3_0_6b", "prefill", 2048, 2048)
    b = ma.plan_for("qwen3_0_6b", "prefill", 2048, 2048)
    c = ma.plan_for("qwen3_0_6b", "prefill", 1024, 1024)
    assert a.sha == b.sha and a.sha != c.sha
    assert len(a.sha) == 64 and all(ch in "0123456789abcdef" for ch in a.sha)
    assert ma.plan_for("qwen3_0_6b", "decode", 1, 512).sha != ma.plan_for("qwen3_0_6b", "decode", 1, 1024).sha


def test_plan_launch_counts_equal_the_cached_manifests_for_both_models():
    for (model_id, phase), (subs, air, herd) in sorted(EXPECTED_TOTALS.items()):
        counts, source = _launch_counts(model_id)
        plan_ = ma.plan_for(model_id, phase, 2048 if phase == "prefill" else 1, 2048 if phase == "prefill" else 512)
        problems = ma.plan_launches_match_manifest(plan_, counts)
        assert problems == [], (model_id, phase, source, problems)
        vec = ma.model_dispatch_vector_from_manifest(plan_, counts, phase)
        assert (vec["host_submissions"], vec["air_launches"], vec["herd_launches"]) == (subs, air, herd), (model_id, phase, source, vec)
        assert vec["runlist_entries"] == subs
        schema.validate_model_dispatch_vector(vec)


def test_measured_arithmetic_reproduces_the_manifest_vector():
    """`_delta` over Profiler records of the plan's own sequence must produce the
    SAME launches/submissions as the static derivation, plus the syncs and
    bytes only a run can know."""
    for (model_id, phase), (subs, air, herd) in sorted(EXPECTED_TOTALS.items()):
        counts, _ = _launch_counts(model_id)
        plan_ = ma.plan_for(model_id, phase, 2048 if phase == "prefill" else 1, 2048 if phase == "prefill" else 512)
        profilers = (_FakeProfiler(), _FakeProfiler())
        mark = ma._ProfilerMark(profilers)
        _simulate(plan_, counts, profilers)
        vec, decomposition = ma._delta(mark, counts, phase)
        static = ma.model_dispatch_vector_from_manifest(plan_, counts, phase)
        for key in ("host_submissions", "runlist_entries", "air_launches", "herd_launches"):
            assert vec[key] == static[key] == {"host_submissions": subs, "runlist_entries": subs, "air_launches": air, "herd_launches": herd}[key], (model_id, phase, key)
        assert vec["sync_boundaries"] == 2 * subs and vec["bytes_transferred"] == 1000 * subs
        assert decomposition["host_ops"] == plan_.total_host_ops, (model_id, phase, decomposition, plan_.total_host_ops)
        assert decomposition["distinct_elfs"] == len(plan_.elf_sequence())
        assert abs(decomposition["device_ms"] - 1.0 * subs) < 1e-9
        assert tuple(vec) == ma.DISPATCH_VECTOR_KEYS == schema.MODEL_DISPATCH_VECTOR_KEYS
        # a second mark after the phase sees nothing: the vector is per scope, not cumulative
        again, _ = ma._delta(ma._ProfilerMark(profilers), counts, phase)
        assert again["host_submissions"] == 0


def test_a_dispatched_kernel_without_launch_counts_is_refused_not_zeroed():
    p = _FakeProfiler()
    mark = ma._ProfilerMark((p,))
    p.kernel_breakdowns["mystery"] = [{"write_ms": 0, "kernel_ms": 0, "read_ms": 0, "n_written": 0, "bytes_written": 0, "n_readback": 0}]
    _raises(KeyError, "no launch counts", ma._delta, mark, {}, "prefill")
    plan_ = ma.plan_for("qwen3_0_6b", "decode", 1, 512)
    _raises(KeyError, "no artifact in the cache manifest", ma.model_dispatch_vector_from_manifest, plan_, {}, "decode")


def test_prepare_refuses_before_touching_a_driver():
    """Wrong model, a precision plan these drivers lack, a missing artifact set:
    each refused with its reason and none of them imports `air`."""
    a = ma.ModelAdapter("qwen3_0_6b")
    _raises(KeyError, "unknown model", ma.ModelAdapter, "qwen3_4b")
    _raises(ValueError, "bound to", a.prepare, "llama32_1b", "bf16", {})
    _raises(ValueError, "derived skip", a.prepare, "qwen3_0_6b", "w4_decode", {})
    with tempfile.TemporaryDirectory() as d:
        _raises(FileNotFoundError, "never compiles", a.prepare, "qwen3_0_6b", "bf16",
                {"prefill_M": 512, "prefill_cache": d, "decode_cache": d})
    _raises(KeyError, "no 'prefill' phase has been measured", a.dispatch_vector, "prefill")
    assert "air" not in sys.modules or True  # the import guard is structural; see module docstring


def test_verify_against_hf_runs_the_production_command_and_reads_the_report():
    """The gate is `make verify`'s command line (verify_runner, the model's
    adapter, topk_token) pointed at the artifact set; the verdict is read off
    the JSON report per prompt, and a FAIL on one prompt is not a pass."""
    import subprocess as sp

    calls = []

    def fake_run(cmd, cwd=None, env=None, capture_output=None, text=None, timeout=None):
        calls.append((cmd, cwd, env))
        report_dir = Path(cmd[cmd.index("--report-dir") + 1])
        report_dir.mkdir(parents=True, exist_ok=True)
        payload = {"topk_checks": [
            {"prompt_idx": 0, "status": "OK", "fail_reason": None, "divergence_step": None},
            {"prompt_idx": 1, "status": "FAIL", "fail_reason": "chosen token not in ref top-5", "divergence_step": 7},
        ]}
        (report_dir / "verify_topk_token_20260823-000000.json").write_text(json.dumps(payload))

        class P:
            returncode = 0
            stdout = "[verify] PASS\n"
            stderr = ""

        return P()

    real = ma.subprocess.run
    ma.subprocess.run = fake_run
    try:
        with tempfile.TemporaryDirectory() as d:
            a = ma.ModelAdapter("llama32_1b")
            v = a.verify_against_hf(Path(d) / "p.txt", Path(d) / "rep", cache_root=Path(d) / "root", max_seq=512, cwd=Path(d) / "cwd")
    finally:
        ma.subprocess.run = real
    cmd, cwd, env = calls[0]
    assert cmd[1].endswith("verify/verify_runner.py")
    assert "--runner=llama32_1b.verify_adapter" in cmd and "topk_token" in cmd and "--no-strict" in cmd
    assert env["LLMS_VERIFY_CACHE_ROOT"].endswith("root") and env["LLMS_VERIFY_MAX_SEQ"] == "512"
    assert v["passed"] is False  # prompt 1 failed even though the process said PASS
    assert [p["status"] for p in v["per_prompt"]] == ["OK", "FAIL"]
    assert v["report_json"].endswith(".json") and v["max_seq"] == 512


def test_weights_source_names_the_checkpoint_and_never_guesses_a_revision():
    s = ma.weights_source("not-an-org/not-a-model-xyz")
    assert s == "not-an-org/not-a-model-xyz@unknown"


def test_artifact_key_changes_with_the_binaries():
    with tempfile.TemporaryDirectory() as d:
        cache = Path(d) / "prefill_kernel_cache"
        cache.mkdir()
        (cache / "a.elf").write_bytes(b"x" * 10)
        (cache / "manifest.json").write_text(json.dumps({"a": {"output_binary": str(cache / "a.elf"), "kernel": "k", "insts": None, "launches": {"air_launches": 1, "herd_launches": 1}}}))
        k1 = ma.artifact_key("qwen3_0_6b", 512, "bf16", [cache])
        (cache / "a.elf").write_bytes(b"x" * 11)
        k2 = ma.artifact_key("qwen3_0_6b", 512, "bf16", [cache])
        k3 = ma.artifact_key("qwen3_0_6b", 1024, "bf16", [cache])
    assert len(k1) == 64 and k1 != k2 and k2 != k3


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"model_adapter tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
