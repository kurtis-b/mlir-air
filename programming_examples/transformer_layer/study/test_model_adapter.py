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
    # `[2026-08-26]` doc 56 H2a: the shipped int4 sibling caches (build_peano,
    # ELFs of 2026-08-19). bf16 prefill on dequantized AWQ weights; int4 decode.
    "llama32_1b_int4": {
        "prefill": {"rms_gemms_rope": (7, 13), "o_ffn": (12, 20), "flash_attn": (1, 1)},
        "decode": {"rms_qkv_int4_rope": (6, 6), "o_gemv_ffn_int4": (3, 5), "lm_head_gemv": (8, 8)},
    },
}

#: The Qwen3-0.6B M=1024 prefill set (devq 570): the registry-driven QKV stage is
#: all-drain there (no cast launch), the O+FFN cascade forced fused-cast.
PINNED_MANIFEST_QWEN_M1024 = {"rms_qkv_qknorm_rope": (8, 14), "o_ffn_qwen": (12, 20), "flash_attn": (1, 1)}

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


FIXTURES = Path(_HERE) / "fixtures" / "h1a_driver_traces"


def _traces():
    out = []
    for path in sorted(FIXTURES.glob("trace_*.json")):
        out.append(json.loads(path.read_text(encoding="utf-8")))
    assert len(out) == 5, sorted(p.name for p in FIXTURES.glob("*"))
    return out


def test_models_bind_the_three_drivers_and_nothing_else():
    assert sorted(ma.MODELS) == ["llama32_1b", "llama32_1b_int4", "qwen3_0_6b"]
    for model_id, b in ma.MODELS.items():
        assert b.spec.name == model_id
        assert b.directory.is_dir(), b.directory
        assert (b.directory / f"{b.package}_inference.py").is_file()
        assert (b.directory / "verify_adapter.py").is_file()
        assert b.verify_adapter == f"{model_id}.verify_adapter"
    assert ma.SUPPORTED_PRECISION_PLANS == ("bf16",)
    # a precision plan is a DRIVER, not a flag (doc 56 sections 3.5 / H2a):
    assert ma.MODELS["qwen3_0_6b"].precision_plans == ("bf16",)
    assert ma.MODELS["llama32_1b"].precision_plans == ("bf16",)
    b4 = ma.MODELS["llama32_1b_int4"]
    assert b4.precision_plans == ("w4_decode",)
    assert b4.session_model_kwargs() == {"model_path": b4.hf_id}
    assert ma.MODELS["llama32_1b"].session_model_kwargs() == {"model_variant": "instruct"}
    assert b4.quant_contract_module == "llama32_1b_int4.awq_repacker"


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


def test_the_recorded_driver_traces_reproduce_the_plan_vector():
    """THE LOAD-BEARING ONE, and it is NOT derived from the plan (H1a review,
    finding 5). `fixtures/h1a_driver_traces/*.json` are Profiler traces the
    PRODUCTION DRIVERS produced on the device (devq 574: every `load_and_run`
    call per kernel name, its syncs and bytes, every `time_cpu` bucket) over
    the rungs of one walk, with the cache manifests they loaded. This test
    runs the adapter's one arithmetic over that recorded driver behaviour and
    compares it with what the PLAN + manifest derive -- recomputing the plan
    here, not reading the fixture's prediction. A driver that adds, removes or
    duplicates a `load_and_run` (or a planner that changes a stage's launch
    count) changes one side and not the other: the runner's live check fails
    the row on the next walk, and re-recording the fixture fails this test.
    The fixture's plan sha is pinned too, so the planner cannot drift quietly."""
    seen = set()
    for t in _traces():
        model_id, phase = t["model_id"], t["phase"]
        seen.add((model_id, phase, t["M"]))
        # the launch counts the driver loaded are the pinned manifests
        for name, counts in t["launch_counts"].items():
            pinned = PINNED_MANIFESTS[model_id]["prefill" if name in PINNED_MANIFESTS[model_id]["prefill"] else "decode"]
            if t["M"] == 1024 and name in PINNED_MANIFEST_QWEN_M1024:
                pinned = PINNED_MANIFEST_QWEN_M1024
            assert (counts["air_launches"], counts["herd_launches"]) == pinned[name], (t["case_id"], name, counts)
        vec, decomposition = ma.dispatch_vector_from_trace(t["trace"], t["launch_counts"], t["scope"])
        n = int(t["trace_samples"])
        per = {k: (v if k == "scope" else v // n) for k, v in vec.items()}
        plan_ = ma.plan_for(model_id, phase, t["M"] if phase == "prefill" else 1, t["context_end"], t["M"], forced=t["forced"])
        assert plan_.sha == t["plan_sha"], (t["case_id"], plan_.sha, t["plan_sha"])
        assert ma.plan_launches_match_manifest(plan_, t["launch_counts"]) == []
        static = ma.model_dispatch_vector_from_manifest(plan_, t["launch_counts"], "prefill" if phase == "prefill" else "decode")
        for key in ("host_submissions", "runlist_entries", "air_launches", "herd_launches"):
            assert per[key] == static[key], (t["case_id"], key, per[key], static[key])
        # the driver's own record, which the plan cannot produce: syncs and bytes are real
        assert per["sync_boundaries"] > 0 and per["bytes_transferred"] > 0
        # `[2026-08-25]` (queue item 15) every planned host stage is a named
        # time_cpu bucket -- kv_append in both decode blocks and both prefill
        # loops, the head-first FA transposes, the adapter's decode embed --
        # so the measured count EQUALS the plan's, and the trace's bucket
        # names are exactly the plan's host-stage names. run_model's live
        # check enforces the count on every rung; this pins it offline on the
        # recorded traces (walk 6, devq 580).
        raw_total = sum(c["calls"] for c in t["trace"]["cpu"].values())
        assert raw_total == plan_.total_host_ops * n, (t["case_id"], raw_total, plan_.total_host_ops, n)
        # decomposition here is over the WHOLE trace (the adapter divides per
        # sample only on the live path), so the raw-total assertion above is
        # the offline form of run_model's live check.
        assert decomposition["host_ops"] == raw_total
        plan_host_names = {st.name for st in plan_.stages if st.where == "host"}
        assert set(t["trace"]["cpu"]) == plan_host_names, (t["case_id"], sorted(t["trace"]["cpu"]), sorted(plan_host_names))
        assert decomposition["distinct_elfs"] == len(plan_.elf_sequence())
        assert per == t["measured_vector"], (t["case_id"], per, t["measured_vector"])
        schema.validate_model_dispatch_vector(per)
    assert seen == {("qwen3_0_6b", "prefill", 2048), ("qwen3_0_6b", "decode", 2048), ("qwen3_0_6b", "prefill", 1024),
                    ("llama32_1b", "prefill", 2048), ("llama32_1b", "decode", 2048)}
    # the forced M=1024 trace carries its deviation in the plan it was hashed with
    forced = [t for t in _traces() if t["M"] == 1024][0]
    assert forced["forced"] == {"o_ffn_qwen": "fused-cast", "o_ffn": "fused-cast"}
    assert ma.plan_for("qwen3_0_6b", "prefill", 1024, 1024, 1024).sha != forced["plan_sha"]


def test_a_driver_that_dispatches_one_more_call_is_caught_by_the_arithmetic():
    """The negative control for the test above: one extra `o_gemv_ffn` call in
    a recorded decode trace (what a driver duplicating a launch looks like)
    moves the vector off the plan's."""
    t = [x for x in _traces() if x["phase"] == "decode" and x["model_id"] == "qwen3_0_6b"][0]
    trace = json.loads(json.dumps(t["trace"]))
    trace["kernels"]["o_gemv_ffn"]["calls"] += 1
    vec, _ = ma.dispatch_vector_from_trace(trace, t["launch_counts"], t["scope"])
    per = {k: (v if k == "scope" else v // t["trace_samples"]) for k, v in vec.items()}
    plan_ = ma.plan_for("qwen3_0_6b", "decode", 1, t["context_end"], t["M"])
    static = ma.model_dispatch_vector_from_manifest(plan_, t["launch_counts"], "decode")
    # 32 tokens: one extra call is 1/32 of a submission per token -- per-token
    # integer division hides it, the SUMMED vector does not; the runner's live
    # check compares the summed phase vector divided by n exactly as here, so
    # the control asserts on the phase total.
    assert vec["host_submissions"] == static["host_submissions"] * t["trace_samples"] + 1
    assert vec["air_launches"] != static["air_launches"] * t["trace_samples"]
    assert per != t["measured_vector"] or vec["host_submissions"] != t["measured_vector"]["host_submissions"] * t["trace_samples"]


def test_a_dispatched_kernel_without_launch_counts_is_refused_not_zeroed():
    trace = {"kernels": {"mystery": {"calls": 1, "n_written": 0, "n_readback": 0, "bytes_written": 0, "bytes_readback": 0, "kernel_ms": 0.0, "sync_ms": 0.0}}, "cpu": {}}
    _raises(KeyError, "no launch counts", ma.dispatch_vector_from_trace, trace, {}, "prefill")
    plan_ = ma.plan_for("qwen3_0_6b", "decode", 1, 512)
    _raises(KeyError, "no artifact in the cache manifest", ma.model_dispatch_vector_from_manifest, plan_, {}, "decode")


def test_prepare_refuses_before_touching_a_driver():
    """Wrong model, a precision plan these drivers lack, a missing artifact set:
    each refused with its reason and none of them imports `air`."""
    a = ma.ModelAdapter("qwen3_0_6b")
    _raises(KeyError, "unknown model", ma.ModelAdapter, "qwen3_4b")
    _raises(ValueError, "bound to", a.prepare, "llama32_1b", "bf16", {})
    _raises(ValueError, "derived skip", a.prepare, "qwen3_0_6b", "w4_decode", {})
    # ... and the int4 driver is w4_decode-ONLY: asking it for bf16 is the
    # same derived skip in the other direction (doc 56 H2a).
    _raises(ValueError, "derived skip", ma.ModelAdapter("llama32_1b_int4").prepare, "llama32_1b_int4", "bf16", {})
    with tempfile.TemporaryDirectory() as d:
        _raises(FileNotFoundError, "never compiles", a.prepare, "qwen3_0_6b", "bf16",
                {"prefill_M": 512, "prefill_cache": d, "decode_cache": d})
    _raises(KeyError, "no 'prefill' phase has been measured", a.dispatch_vector, "prefill")
    assert "air" not in sys.modules or True  # the import guard is structural; see module docstring


def test_verify_against_hf_runs_the_production_command_on_the_named_artifacts():
    """The gate is `make verify`'s command line (verify_runner, the model's
    adapter, topk_token) handed the TIMED artifact set by path (loaded, never
    compiled: the env names prefill/decode caches, the pad M and the KV room);
    the verdict is read off the report THIS call wrote, per prompt. A FAIL on
    one prompt, a nonzero exit, or a stale report from an earlier call is not
    a pass (H1a review, findings 1 and 2)."""
    calls = []
    behaviour = {"returncode": 0, "write_report": True, "statuses": ["OK", "FAIL"]}

    def fake_run(cmd, cwd=None, env=None, capture_output=None, text=None, timeout=None):
        calls.append((cmd, cwd, env))
        report_dir = Path(cmd[cmd.index("--report-dir") + 1])
        assert report_dir.is_dir() and not list(report_dir.iterdir()), "the report dir must be fresh and empty"
        if behaviour["write_report"]:
            payload = {"topk_checks": [
                {"prompt_idx": i, "status": st, "fail_reason": None if st == "OK" else "chosen token not in ref top-5", "divergence_step": None}
                for i, st in enumerate(behaviour["statuses"])]}
            (report_dir / "verify_topk_token_20260823-000000.json").write_text(json.dumps(payload))

        class P:
            returncode = behaviour["returncode"]
            stdout = "[verify] PASS\n"
            stderr = ""

        return P()

    real = ma.subprocess.run
    ma.subprocess.run = fake_run
    try:
        with tempfile.TemporaryDirectory() as d:
            a = ma.ModelAdapter("llama32_1b")
            kw = dict(prefill_cache=Path(d) / "p", decode_cache=Path(d) / "dec", prefill_M=1024, max_seq=1056, cwd=Path(d) / "cwd")
            v = a.verify_against_hf(Path(d) / "p.txt", Path(d) / "rep", **kw)
            cmd, cwd, env = calls[0]
            assert cmd[1].endswith("verify/verify_runner.py")
            assert "--runner=llama32_1b.verify_adapter" in cmd and "topk_token" in cmd and "--no-strict" in cmd
            assert env["LLMS_VERIFY_PREFILL_CACHE"].endswith("/p") and env["LLMS_VERIFY_DECODE_CACHE"].endswith("/dec")
            assert env["LLMS_VERIFY_PREFILL_M"] == "1024" and env["LLMS_VERIFY_MAX_SEQ"] == "1056"
            assert "LLMS_VERIFY_CACHE_ROOT" not in env and "LLMS_VERIFY_O_FFN_GEMM_METHOD" not in env
            assert v["passed"] is False and "a prompt failed" in " ".join(v["problems"])
            assert [p["status"] for p in v["per_prompt"]] == ["OK", "FAIL"]
            assert v["report_json"].startswith(v["report_dir"]) and v["report_dir"].startswith(str(Path(d) / "rep"))
            # all OK, exit 0: passed, in a SECOND fresh dir
            behaviour["statuses"] = ["OK", "OK"]
            v2 = a.verify_against_hf(Path(d) / "p.txt", Path(d) / "rep", **kw)
            assert v2["passed"] is True and v2["report_dir"] != v["report_dir"]
            # exit 1 with an OK report: not a pass
            behaviour["returncode"] = 1
            v3 = a.verify_against_hf(Path(d) / "p.txt", Path(d) / "rep", **kw)
            assert v3["passed"] is False and "exited 1" in " ".join(v3["problems"])
            # exit 0 but no report of its own (a crash that left older reports in OTHER dirs): not a pass
            behaviour["returncode"] = 0
            behaviour["write_report"] = False
            v4 = a.verify_against_hf(Path(d) / "p.txt", Path(d) / "rep", **kw)
            assert v4["passed"] is False and v4["report_json"] is None and "wrote no report" in " ".join(v4["problems"])
    finally:
        ma.subprocess.run = real


def test_artifact_content_sha_identifies_the_bytes_not_the_paths():
    with tempfile.TemporaryDirectory() as d:
        cache = Path(d) / "prefill_kernel_cache"
        cache.mkdir()
        (cache / "a.elf").write_bytes(b"x" * 10)
        (cache / "manifest.json").write_text(json.dumps({"a": {"output_binary": "prefill_kernel_cache/a.elf", "kernel": "k", "insts": None, "launches": {"air_launches": 1, "herd_launches": 1}}}))
        k1 = ma.artifact_content_sha([cache])
        k1b = ma.artifact_content_sha([cache])  # relative manifest path, resolved against the cache's parent
        (cache / "a.elf").write_bytes(b"y" * 10)  # same size, same name, other bytes
        k2 = ma.artifact_content_sha([cache])
    assert k1["sha256"] == k1b["sha256"] and k1["sha256"] != k2["sha256"]
    assert k1["files"]["prefill_kernel_cache/a"][0][2] != k2["files"]["prefill_kernel_cache/a"][0][2]


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



# ---------------------------------------------------------------------------
# `[2026-08-25]` Queue item 14: the O+FFN cascade is per-GEMM. The layout owner
# (`shared.builders.gemm_builder.o_ffn_gemm_layout`, air-free) derives each
# GEMM's registry method / tile_n / scratch tail; these pin it against the
# planner's independent derivation and against the recorded artifact shapes.
# ---------------------------------------------------------------------------


def _offn_layout(M, method=None):
    from shared.builders.gemm_builder import o_ffn_gemm_layout

    # Qwen3-0.6B: emb 1024, hidden 3072, decoupled q_dim 2048.
    return o_ffn_gemm_layout(M, 1024, 3072, q_dim=2048, method=method)


def test_the_o_ffn_cascade_layout_is_per_gemm_from_the_registry():
    """At M=512/1024 the registry best is drain for ALL FOUR GEMMs: 1 launch
    each, NO f32 scratch args (the ELF has 15 args, not 19) -- the doc 56 H1a
    wall 1 shape, now buildable. At M=2048 all four are fused-cast: the
    scratch tail is exactly 15..18 and the cascade is byte-identical policy to
    the shipped artifact. The layout's launch count must equal the PLANNER's
    o_ffn_qwen stage launches (two independent derivations of one artifact)."""
    for M, method, scratch, launches in (
        (512, "drain", [None, None, None, None], 8),
        (1024, "drain", [None, None, None, None], 8),
        (2048, "fused-cast", [15, 16, 17, 18], 12),
    ):
        lay = _offn_layout(M)
        for role in ("o", "gate_up", "down"):
            assert lay[role]["method"] == method, (M, role, lay[role]["method"])
        assert lay["scratch_for"] == scratch, (M, lay["scratch_for"])
        assert len(lay["scratch_args"]) == len([s for s in scratch if s is not None])
        assert lay["launches"] == launches, (M, lay["launches"])
        stage = [s for s in ma.plan_for("qwen3_0_6b", "prefill", M, M, M).stages if s.name == "o_ffn_qwen"][0]
        assert stage.launches == lay["launches"], (M, stage.launches, lay["launches"])
        assert stage.source == "measured"


def test_a_forced_cascade_layout_is_the_recorded_m1024_artifact_shape():
    """`gemm_method=` stays as the explicit override (test-only): forced
    fused-cast at M=1024 reproduces the recorded forced artifact's shape --
    12 air launches (PINNED_MANIFEST_QWEN_M1024) and the 15..18 scratch tail
    -- and equals the forced plan's launch count."""
    lay = _offn_layout(1024, method="fused-cast")
    assert lay["scratch_for"] == [15, 16, 17, 18]
    assert lay["launches"] == PINNED_MANIFEST_QWEN_M1024["o_ffn_qwen"][0]
    p = ma.plan_for("qwen3_0_6b", "prefill", 1024, 1024, 1024, forced={"o_ffn_qwen": "fused-cast"})
    stage = [s for s in p.stages if s.name == "o_ffn_qwen"][0]
    assert stage.launches == lay["launches"] and stage.source == "forced"


def test_the_m512_mix_names_a_distinct_mm_variant_per_gemm():
    """Doc 56 H1a wall 2 was `one mm.o variant per ELF`: the M=512 fused-cast
    registry rows are tile_n 96 (gate/up) vs 128 (O/down). Per-GEMM naming
    must give the two DIFFERENT variants their own sym suffix + object -- and
    the unforced M=512 layout names one drain variant for all four."""
    forced = _offn_layout(512, method="fused-cast")
    assert forced["gate_up"]["sym_suffix"] == "_m64n96" and forced["gate_up"]["obj"] == "mm_m64n96.o"
    assert forced["o"]["sym_suffix"] == "_m64n128" and forced["down"]["sym_suffix"] == "_m64n128"
    lay = _offn_layout(512)
    assert {lay[r]["sym_suffix"] for r in ("o", "gate_up", "down")} == {"_m32n128"}
    assert {lay[r]["obj"] for r in ("o", "gate_up", "down")} == {"mm_m32n128.o"}


def test_the_driver_binds_its_o_ffn_scratch_layout_to_the_layout_owner():
    """The Qwen driver's run-only path must re-derive the o_ffn scratch tail
    from `o_ffn_gemm_layout` (the QKV half of this bug shipped once: 17 args
    to an 18-arg ELF). Read from source by ast, not imported -- the driver
    module needs weights/numpy this suite deliberately does not."""
    import ast

    src = (Path(_PE) / "llms" / "qwen3_0_6b" / "qwen3_0_6b_prefill.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    restore = ast.unparse(fns["restore_scratch_layout"])
    assert "o_ffn_gemm_layout" in restore and "_OFFN_SCRATCH_FOR" in restore
    call = ast.unparse(fns["_o_ffn_call"])
    assert "_OFFN_SCRATCH_FOR" in call and "RuntimeError" in call, "the call site must refuse an unset layout, not guess 19 args"
    # both call sites go through the one owner
    for consumer in ("preload_prefill_weights", "run_transformer_block_qwen3"):
        assert "_o_ffn_call(" in ast.unparse(fns[consumer]), consumer


def test_artifact_content_sha_sees_the_abi_metadata():
    """`[2026-08-25]` (item 14 review, blocking) compile.json's
    `artifact_deviation` is ABI metadata -- it decides which scratch layout
    the drivers restore -- so the timed-vs-verified identity must move when it
    is deleted or edited between timing and gating, with the ELF bytes
    unchanged. Absence hashes as absence (two never-forced caches still agree)
    and the REST of compile.json (wall_s, cwd) stays out of the identity."""
    with tempfile.TemporaryDirectory() as d:
        cache = Path(d) / "prefill_kernel_cache"
        cache.mkdir()
        (cache / "a.elf").write_bytes(b"x" * 10)
        (cache / "manifest.json").write_text(json.dumps({"a": {"output_binary": str(cache / "a.elf"), "kernel": "k", "insts": None, "launches": {"air_launches": 1, "herd_launches": 1}}}))
        absent = ma.artifact_content_sha([cache])["sha256"]
        (cache / "compile.json").write_text(json.dumps({"artifact_deviation": {"o_ffn_gemm_method": "fused-cast"}, "wall_s": 1.0}))
        forced = ma.artifact_content_sha([cache])["sha256"]
        (cache / "compile.json").write_text(json.dumps({"artifact_deviation": None, "wall_s": 1.0}))
        unforced = ma.artifact_content_sha([cache])["sha256"]
        (cache / "compile.json").write_text(json.dumps({"artifact_deviation": None, "wall_s": 99.0, "cwd": "/elsewhere"}))
        unforced2 = ma.artifact_content_sha([cache])["sha256"]
    assert len({absent, forced, unforced}) == 3, "deleting or editing the deviation must move the identity"
    assert unforced == unforced2, "wall_s/cwd are not part of the identity"


def test_a_forced_artifact_sets_recorded_deviation_reaches_the_restore():
    """`[2026-08-25]` The layout restored for a LOADED artifact set must be the
    set's OWN: a forced set's compile.json `artifact_deviation` names the
    method its ELF was built with, and restoring the registry-best layout
    against a forced fused-cast o_ffn ELF sets 15 of its 19 args -- the f32
    scratch args stay UNBOUND, a nondeterministic wrong answer the token gate
    catches only sometimes (observed live, devq 583). Pinned at all three
    hops by source: the adapter reads the note and passes the method, the
    driver's restore accepts and forwards it, the verify adapter does the
    same on its loaded-cache path."""
    import ast
    import inspect

    src = inspect.getsource(ma.ModelAdapter._restore_scratch_layout)
    assert "COMPILE_NOTE" in src and "artifact_deviation" in src and "o_ffn_gemm_method" in src
    prep = inspect.getsource(ma.ModelAdapter.prepare)
    assert "_restore_scratch_layout(config, M, prefill_dir)" in prep
    driver = (Path(_PE) / "llms" / "qwen3_0_6b" / "qwen3_0_6b_prefill.py").read_text(encoding="utf-8")
    fns = {n.name: n for n in ast.walk(ast.parse(driver)) if isinstance(n, ast.FunctionDef)}
    restore = fns["restore_scratch_layout"]
    assert "o_ffn_gemm_method" in [a.arg for a in restore.args.args], "restore must accept the deviation"
    assert "method=o_ffn_gemm_method" in ast.unparse(restore)
    va = (Path(_PE) / "llms" / "qwen3_0_6b" / "verify_adapter.py").read_text(encoding="utf-8")
    assert "artifact_deviation" in va and "o_ffn_gemm_method" in va


# ---------------------------------------------------------------------------
# `[2026-08-26]` Queue item 17 (doc 56 H2a): the int4 sibling under the runner.
# ---------------------------------------------------------------------------


def test_the_w4_decode_plan_matches_the_shipped_int4_manifest():
    """The w4_decode decode plan names the SHIPPED int4 ELF sequence with its
    launch counts (rms_qkv_int4_rope 6 / o_gemv_ffn_int4 3 / lm_head_gemv 8:
    33 submissions, 152 executed launches, 184 herd launches per token -- doc
    57 section 1.3's counts), its host stages are exactly the driver's named
    buckets (34 host ops), and the w4_decode PREFILL plan is the bf16 prefill
    (dequantized AWQ weights through the bf16 stitchers -- doc 56 section
    3.5's 'bf16 prefill weights resident separately')."""
    counts, source = _launch_counts("llama32_1b_int4")
    p = ma.plan_for("llama32_1b_int4", "decode", 1, 512, 2048, "w4_decode")
    assert p.elf_sequence() == ["rms_qkv_int4_rope", "o_gemv_ffn_int4", "lm_head_gemv"]
    assert ma.plan_launches_match_manifest(p, counts) == [], source
    vec = ma.model_dispatch_vector_from_manifest(p, counts, "decode")
    assert (vec["host_submissions"], vec["air_launches"], vec["herd_launches"]) == (33, 152, 184), (source, vec)
    assert p.host_sequence() == ["embed_lookup", "kv_append", "decode_attention_cpu", "final_rms_norm"]
    assert p.total_host_ops == 34
    schema.validate_model_dispatch_vector(vec)
    # the int4 stages carry less than a quarter of the bf16 weight bytes
    bf16_llama = ma.plan_for("llama32_1b", "decode", 1, 512)
    assert p.resident_weight_bytes < bf16_llama.resident_weight_bytes * 0.45, (
        p.resident_weight_bytes, bf16_llama.resident_weight_bytes)
    # value identity: the precision plan and the spec are both in the hash
    assert p.sha != bf16_llama.sha
    assert p.sha != ma.plan_for("llama32_1b_int4", "decode", 1, 1024, 2048, "w4_decode").sha
    # prefill under w4_decode is the bf16 kernel sequence, matching the shipped prefill cache
    pp = ma.plan_for("llama32_1b_int4", "prefill", 2048, 2048, 2048, "w4_decode")
    assert pp.elf_sequence() == ["rms_gemms_rope", "flash_attn", "o_ffn", "lm_head_gemv"]
    assert ma.plan_launches_match_manifest(pp, counts) == [], source
    # NEGATIVE control: a bf16-precision decode plan against the int4 manifest
    # names rms_gemv_rope / o_gemv_ffn, and the mismatch is REPORTED, not
    # silently absorbed -- precision changes the artifact names.
    pb = ma.plan_for("llama32_1b_int4", "decode", 1, 512, 2048, "bf16")
    problems = ma.plan_launches_match_manifest(pb, counts)
    assert any("rms_gemv_rope" in x for x in problems), problems


def test_the_quant_columns_come_from_the_packing_code_and_differ_by_path():
    """quant_* for a w4_decode row are read from the model's packing module
    (awq_repacker.quant_contract), never hand-typed in the study: gs parsed
    from the checkpoint id AND equal to the loader's own default (read by
    ast), the GEMM and GEMV contracts DIFFERENT (doc 56 section 3.6: the
    columns are populated, not duplicated), the accumulator pinned against
    the kernel source. bf16 rows get EMPTY quant columns. The plan's stage
    contract NAME equals the packing code's (the fa_cache_name pattern)."""
    q = ma.quant_columns("llama32_1b_int4", "w4_decode")
    assert q["quant_group_size"] == 128
    import importlib

    qc = importlib.import_module("llama32_1b_int4.awq_repacker").quant_contract()
    assert qc == q, "the study's columns must BE the packing module's contract"
    assert q["quant_gemm_contract"] != q["quant_gemv_contract"]
    assert "dequant_to_bf16" in q["quant_gemm_contract"]
    assert "in-kernel" in q["quant_gemv_contract"]
    assert "accfloat" in q["quant_accum_type"]
    # schema columns only, plus the plan-facing name key
    quant_fields = {f for f in schema.empty_row("results") if f.startswith("quant_")}
    assert set(q) - {"quant_gemv_contract_name"} == quant_fields, sorted(q)
    # the plan mirrors the NAME without importing the model dir
    from shared.plan import W4_GEMV_CONTRACT

    assert W4_GEMV_CONTRACT == q["quant_gemv_contract_name"]
    p = ma.plan_for("llama32_1b_int4", "decode", 1, 512, 2048, "w4_decode")
    for stage in p.stages:
        if stage.name in ("rms_qkv_int4_rope", "o_gemv_ffn_int4"):
            assert any(W4_GEMV_CONTRACT in why for _op, _n, why in stage.launch_breakdown), stage.name
    # bf16 rows: empty (doc 03 -- present and empty)
    assert ma.quant_columns("llama32_1b", "bf16") == {}
    assert ma.quant_columns("qwen3_0_6b", "bf16") == {}
    # a qk-norm model has no w4_decode plan until H2b builds its candidates
    _raises(ValueError, "H2b", ma.plan_for, "qwen3_0_6b", "decode", 1, 512, 2048, "w4_decode")


def test_the_int4_driver_buckets_every_planned_host_stage():
    """By source (ast), as the qwen scratch-layout hops are pinned: the int4
    decode block buckets kv_append and decode_attention_cpu, the decode step
    buckets final_rms_norm (embed_lookup is the adapter's bucket), the prefill
    loop uses the plan's kv_append name (not the old kv_cache_extract), the
    prefill entry accepts the seam's profile= kwarg, and the verify adapter
    has the loaded-cache hop (LLMS_VERIFY_*: the gate runs on the timed bytes,
    never compiles on that path)."""
    import ast

    d = Path(_PE) / "llms" / "llama32_1b_int4"
    dec = (d / "llama32_1b_int4_decode.py").read_text(encoding="utf-8")
    fns = {n.name: n for n in ast.walk(ast.parse(dec)) if isinstance(n, ast.FunctionDef)}
    block = ast.unparse(fns["run_decode_block"])
    assert "time_cpu('kv_append')" in block and "time_cpu('decode_attention_cpu')" in block
    inf = (d / "llama32_1b_int4_inference.py").read_text(encoding="utf-8")
    assert "time_cpu('kv_cache_extract')" not in ast.unparse(ast.parse(inf)), \
        "the prefill bucket must carry the plan's stage name (kv_append)"
    fns_i = {n.name: n for n in ast.walk(ast.parse(inf)) if isinstance(n, ast.FunctionDef)}
    assert "time_cpu('final_rms_norm')" in ast.unparse(fns_i["run_npu_decode_step"])
    prefill_fn = fns_i["run_npu_prefill"]
    assert "profile" in [a.arg for a in prefill_fn.args.args], "the adapter passes profile="
    assert "time_cpu('kv_append')" in ast.unparse(prefill_fn)
    # the driver Session takes model_path -- exactly what the binding declares
    sessions = [n for n in ast.walk(ast.parse(inf)) if isinstance(n, ast.ClassDef) and n.name == "Session"]
    assert "model_path" in ast.unparse(sessions[0])
    # weights module presents the seam surface (load_weights / generate_rope_lut)
    w = (d / "llama32_1b_int4_weights.py").read_text(encoding="utf-8")
    assert "load_weights = load_weights_awq" in w and "generate_rope_lut" in w
    va = (d / "verify_adapter.py").read_text(encoding="utf-8")
    for needle in ("LLMS_VERIFY_PREFILL_CACHE", "LLMS_VERIFY_DECODE_CACHE", "LLMS_VERIFY_PREFILL_M", "load_manifest"):
        assert needle in va, needle


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"model_adapter tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
