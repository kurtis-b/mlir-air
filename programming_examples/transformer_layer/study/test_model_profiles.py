# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host tests for `model_profiles.py` (doc 56 H1a): the model-smoke walk's shape,
its applicability rule, its resume identity and its curve labels."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import model_profiles as mp  # noqa: E402
import resume  # noqa: E402
import schema  # noqa: E402


def _compiled_all(prof):
    return {(m, M): {"prefill_cache": f"/c/{m}/M{M}/p", "decode_cache": f"/c/{m}/d"} for m in prof.models for M in prof.prefill_Ms[m]}


def test_model_smoke_is_the_h1a_row_of_doc_56():
    prof = mp.profile("model-smoke")
    assert prof.models == ("qwen3_0_6b", "llama32_1b")
    assert prof.prefill_Ms == {"qwen3_0_6b": (512, 1024, 2048), "llama32_1b": (2048,)}
    assert prof.decode_ctxs == (512, 1024, 2048)
    assert prof.decode_n_tokens == mp.GATE_N_TOKENS == 32
    ids = [r.case_id for r in prof.rungs()]
    assert len(ids) == len(set(ids)) == 10
    assert prof.expected_files() == ["model_qwen3_0_6b.csv", "model_llama32_1b.csv"]


def test_unknown_profile_is_refused():
    try:
        mp.profile("model-everything")
    except ValueError as e:
        assert "unknown model profile" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_prefill_rungs_are_labelled_kernel_scaling_and_never_ubatch():
    """Doc 56 section 3.4: prompt length == M, no chunking, and the label says
    so on every row of the curve."""
    for r in mp.profile("model-smoke").rungs():
        if r.phase == "prefill":
            assert r.curve == mp.KERNEL_SCALING and "kernel-scaling" in r.label and "no chunking" in r.label
            assert r.prompt_tokens == r.context_end == r.M == r.seq == r.ubatch_tokens
            assert "ubatch" not in r.label
        else:
            assert r.curve == mp.DECODE_CONTEXT and r.seq == 1 == r.ubatch_tokens
            assert r.prompt_tokens == r.context_end - r.n_tokens == r.context_start
            assert r.M == mp.SHIPPED_PREFILL_M


def test_the_gate_runs_the_timed_prompt_at_its_own_length():
    """H1a review finding 3: a prefill rung times M valid tokens, so its gate
    prompt is M tokens -- never M-32 -- and the 32 generation slots come from
    `gate_max_seq` (= M + 32, the KV capacity), not from a shorter prompt."""
    for r in mp.profile("model-smoke").rungs():
        assert r.gate_prompt_tokens == r.prompt_tokens, r.case_id
        if r.phase == "prefill":
            assert r.gate_prompt_tokens == r.M
        assert r.gate_max_seq == r.M + mp.GATE_N_TOKENS
        assert r.gate_prompt_tokens + mp.GATE_N_TOKENS <= r.gate_max_seq


def test_resume_identity_is_the_v3_row_key():
    """`rung_key(mode, seq, extra)` must equal `row_key` of the row the rung
    writes -- three decode rungs at seq 1 must be three keys."""
    prof = mp.profile("model-smoke")
    keys = set()
    for r in prof.rungs():
        row = schema.empty_row()
        row.update(execution_mode=schema.EXECUTION_MODE_CSV[r.mode], seq_len=r.seq, measurement_scope="model",
                   model_id=r.model_id, phase=r.phase, ubatch_tokens=r.ubatch_tokens,
                   context_end_tokens=r.context_end, precision_plan_id=r.precision_plan)
        assert resume.row_key(row) == resume.rung_key(r.mode, r.seq, r.extra), r.case_id
        keys.add(resume.row_key(row))
    assert len(keys) == 10


def test_bind_turns_a_missing_artifact_set_into_a_skip_with_the_reason():
    prof = mp.profile("model-smoke")
    unbound = prof.rungs()
    assert all(r.skip_reason and "no compiled prefill artifact set" in r.skip_reason for r in unbound)
    compiled = _compiled_all(prof)
    del compiled[("qwen3_0_6b", 512)]
    del compiled[("qwen3_0_6b", 1024)]
    bound = prof.bind(compiled, {("qwen3_0_6b", 512): "registry lacks 512x1024x2048 (devq 567)"})
    skips = {r.case_id: r.skip_reason for r in bound.rungs() if r.skip_reason}
    assert set(skips) == {"qwen3_0_6b/prefill/M512/ctx512/bf16", "qwen3_0_6b/prefill/M1024/ctx1024/bf16"}
    assert "devq 567" in skips["qwen3_0_6b/prefill/M512/ctx512/bf16"]
    assert "run_model.py compile" in skips["qwen3_0_6b/prefill/M1024/ctx1024/bf16"]
    assert bound.expected_rows() == {
        "model_qwen3_0_6b.csv": {"rows": 6, "measured": 4, "skipped": 2},
        "model_llama32_1b.csv": {"rows": 4, "measured": 4, "skipped": 0},
    }
    assert bound.artifact_sets() == [("qwen3_0_6b", 2048), ("llama32_1b", 2048)]
    full = prof.bind(_compiled_all(prof))
    assert full.artifact_sets() == [("qwen3_0_6b", 512), ("qwen3_0_6b", 1024), ("qwen3_0_6b", 2048), ("llama32_1b", 2048)]
    assert all(r.skip_reason is None for r in full.rungs())
    assert full.summary()["expected_rows"]["model_qwen3_0_6b.csv"]["skipped"] == 0


def test_discover_compiled_needs_a_manifest_not_a_directory():
    with tempfile.TemporaryDirectory() as d:
        llms = Path(d) / "llms"
        for m in ("qwen3_0_6b", "llama32_1b"):
            (llms / m / "build_peano" / "prefill_kernel_cache").mkdir(parents=True)
            (llms / m / "build_peano" / "decode_kernel_cache").mkdir(parents=True)
        (llms / "qwen3_0_6b" / "build_peano" / "prefill_kernel_cache" / "manifest.json").write_text("{}")
        (llms / "qwen3_0_6b" / "build_peano" / "decode_kernel_cache" / "manifest.json").write_text("{}")
        root = Path(d) / "compiled"
        (root / "qwen3_0_6b" / "M512" / "prefill_kernel_cache").mkdir(parents=True)
        (root / "qwen3_0_6b" / "M1024" / "prefill_kernel_cache").mkdir(parents=True)
        (root / "qwen3_0_6b" / "M1024" / "prefill_kernel_cache" / "manifest.json").write_text("{}")
        compiled, notes = mp.discover_compiled(("qwen3_0_6b", "llama32_1b"), root, llms_dir=llms)
    assert set(compiled) == {("qwen3_0_6b", 2048), ("qwen3_0_6b", 1024)}
    assert ("llama32_1b", 2048) in notes and ("qwen3_0_6b", 512) in notes
    assert "compile did not finish" in notes[("qwen3_0_6b", 512)]
    assert compiled[("qwen3_0_6b", 1024)]["decode_cache"].endswith("build_peano/decode_kernel_cache")


def test_summary_is_json_and_names_every_rung():
    prof = mp.profile("model-smoke").bind(_compiled_all(mp.profile("model-smoke")))
    text = json.dumps(prof.summary())
    for r in prof.rungs():
        assert r.case_id in text


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"model_profiles tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
