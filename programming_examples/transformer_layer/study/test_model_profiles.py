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
    return {(m, M, prof.precision_plan): {"prefill_cache": f"/c/{m}/M{M}/p", "decode_cache": f"/c/{m}/d"}
            for m in prof.models for M in prof.prefill_Ms[m]}


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
    assert all(r.skip_reason and "no compiled artifact set" in r.skip_reason for r in unbound)
    compiled = _compiled_all(prof)
    del compiled[("qwen3_0_6b", 512, "bf16")]
    del compiled[("qwen3_0_6b", 1024, "bf16")]
    bound = prof.bind(compiled, {("qwen3_0_6b", 512, "bf16"): "registry lacks 512x1024x2048 (devq 567)"})
    skips = {r.case_id: r.skip_reason for r in bound.rungs() if r.skip_reason}
    assert set(skips) == {"qwen3_0_6b/prefill/M512/ctx512/bf16", "qwen3_0_6b/prefill/M1024/ctx1024/bf16"}
    assert "devq 567" in skips["qwen3_0_6b/prefill/M512/ctx512/bf16"]
    assert "run_model.py compile" in skips["qwen3_0_6b/prefill/M1024/ctx1024/bf16"]
    assert bound.expected_rows() == {
        "model_qwen3_0_6b.csv": {"rows": 6, "measured": 4, "skipped": 2},
        "model_llama32_1b.csv": {"rows": 4, "measured": 4, "skipped": 0},
    }
    assert bound.artifact_sets() == [("qwen3_0_6b", 2048, "bf16"), ("llama32_1b", 2048, "bf16")]
    full = prof.bind(_compiled_all(prof))
    assert full.artifact_sets() == [("qwen3_0_6b", 512, "bf16"), ("qwen3_0_6b", 1024, "bf16"),
                                    ("qwen3_0_6b", 2048, "bf16"), ("llama32_1b", 2048, "bf16")]
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
    assert set(compiled) == {("qwen3_0_6b", 2048, "bf16"), ("qwen3_0_6b", 1024, "bf16")}
    assert ("llama32_1b", 2048, "bf16") in notes and ("qwen3_0_6b", 512, "bf16") in notes
    assert "compile did not finish" in notes[("qwen3_0_6b", 512, "bf16")]
    assert compiled[("qwen3_0_6b", 1024, "bf16")]["decode_cache"].endswith("build_peano/decode_kernel_cache")


def test_summary_is_json_and_names_every_rung():
    prof = mp.profile("model-smoke").bind(_compiled_all(mp.profile("model-smoke")))
    text = json.dumps(prof.summary())
    for r in prof.rungs():
        assert r.case_id in text


def test_w4_decode_is_the_h2a_row_of_doc_56():
    """`[2026-08-26]` Queue item 17: three decode rungs of the EXISTING int4
    driver at ctx 512/1024/2048 on the shipped M=2048 set, precision_plan_id
    w4_decode on every row, decode-context labels, one CSV; the shipped
    build_peano caches are the artifact set (discover_compiled's convention),
    and an unbound profile is three skips, not a shrunken walk."""
    prof = mp.profile("w4-decode")
    assert prof.models == ("llama32_1b_int4",)
    assert prof.prefill_Ms == {"llama32_1b_int4": ()}
    assert prof.decode_ctxs == (512, 1024, 2048)
    assert prof.precision_plan == "w4_decode"
    rungs = prof.rungs()
    assert len(rungs) == 3 and all(r.phase == "decode" and r.curve == mp.DECODE_CONTEXT for r in rungs)
    assert [r.case_id for r in rungs] == [
        f"llama32_1b_int4/decode/M2048/ctx{c}/w4_decode" for c in (512, 1024, 2048)]
    assert all(r.precision_plan == "w4_decode" and r.M == mp.SHIPPED_PREFILL_M for r in rungs)
    assert all(r.skip_reason and "no compiled artifact set" in r.skip_reason for r in rungs)
    assert prof.expected_files() == ["model_llama32_1b_int4.csv"]
    bound = prof.bind({("llama32_1b_int4", 2048, "w4_decode"): {"prefill_cache": "/c/p", "decode_cache": "/c/d"}})
    assert all(r.skip_reason is None for r in bound.rungs())
    assert bound.artifact_sets() == [("llama32_1b_int4", 2048, "w4_decode")]
    assert bound.expected_rows() == {"model_llama32_1b_int4.csv": {"rows": 3, "measured": 3, "skipped": 0}}
    # resume identity: three decode rungs at seq 1 are three keys, distinct
    # from the bf16 llama rungs at the same contexts by the model AND plan.
    keys = {resume.rung_key(r.mode, r.seq, r.extra) for r in bound.rungs()}
    assert len(keys) == 3
    bf16 = {resume.rung_key(r.mode, r.seq, r.extra) for r in mp.profile("model-smoke").rungs() if r.phase == "decode"}
    assert not keys & bf16


def test_w4_default_qwen_carries_both_precisions_in_one_walk():
    """`[2026-08-26]` Queue item 24 (doc 56 H2b, the default flip): the standing
    decode numbers re-taken with BOTH precisions in one walk. `decode_points` is
    the decode mirror of item 20's `prefill_points`, and it exists for the same
    reason: an A/B whose two arms are two sessions measures session drift as
    much as it measures the thing. Six rungs, three plans-per-context pairs, two
    artifact sets, distinct resume keys, and item 18's profile left untouched so
    its walks stay reproducible."""
    prof = mp.profile("w4-default-qwen")
    assert prof.models == ("qwen3_0_6b",)
    assert prof.precision_plan == "w4_decode", "the DEFAULT is the profile's plan"
    assert prof.precision_plans_used() == ("w4_decode", "bf16")
    rungs = prof.rungs()
    assert [r.case_id for r in rungs] == (
        [f"qwen3_0_6b/decode/M2048/ctx{c}/w4_decode" for c in (512, 1024, 2048)]
        + [f"qwen3_0_6b/decode/M2048/ctx{c}/bf16" for c in (512, 1024, 2048)])
    assert all(r.phase == "decode" and r.n_tokens == 32 for r in rungs)
    # two artifact sets, one per plan -- and each rung skips on ITS OWN plan's set
    bound = prof.bind({("qwen3_0_6b", 2048, "w4_decode"): {"prefill_cache": "/c/p", "decode_cache": "/c/w4/d"}})
    got = {(r.precision_plan, r.skip_reason is None) for r in bound.rungs()}
    assert got == {("w4_decode", True), ("bf16", False)}
    assert bound.expected_rows()["model_qwen3_0_6b.csv"] == {"rows": 6, "measured": 3, "skipped": 3}
    both = prof.bind({("qwen3_0_6b", 2048, "w4_decode"): {"prefill_cache": "/c/p", "decode_cache": "/c/w4/d"},
                      ("qwen3_0_6b", 2048, "bf16"): {"prefill_cache": "/c/p", "decode_cache": "/c/bf16/d"}})
    assert both.artifact_sets() == [("qwen3_0_6b", 2048, "w4_decode"), ("qwen3_0_6b", 2048, "bf16")]
    keys = {resume.rung_key(r.mode, r.seq, r.extra) for r in both.rungs()}
    assert len(keys) == 6, "a plan-blind resume key would collapse the A/B to 3 rows"
    # item 18's profile is untouched
    assert mp.profile("w4-decode-qwen").decode_points == ()
    assert [r.case_id for r in mp.profile("w4-decode-qwen").rungs()] == [
        f"qwen3_0_6b/decode/M2048/ctx{c}/w4_decode" for c in (512, 1024, 2048)]
    assert "decode_points" in prof.summary()


def test_w4_decode_qwen_is_the_h2b_row_of_doc_56():
    """`[2026-08-26]` Queue item 18 (doc 56 H2b): three qwen decode rungs under
    w4_decode. The DECODE artifact set is plan-selected: qwen's shipped
    build_peano decode implements its binding's FIRST plan (bf16), so the w4
    walk needs `<root>/qwen3_0_6b/w4_decode/decode_kernel_cache` from
    `run_model.py compile-decode` -- a missing set is a derived skip NAMING
    that command, never a silent fall-through to the shipped bf16 bytes. The
    llama w4 profile keeps binding the shipped caches (its shipped plan IS
    w4_decode)."""
    prof = mp.profile("w4-decode-qwen")
    assert prof.models == ("qwen3_0_6b",)
    assert prof.decode_ctxs == (512, 1024, 2048) and prof.precision_plan == "w4_decode"
    rungs = prof.rungs()
    assert [r.case_id for r in rungs] == [
        f"qwen3_0_6b/decode/M2048/ctx{c}/w4_decode" for c in (512, 1024, 2048)]
    # discovery: shipped prefill + PLAN-suffixed decode set, or a skip that says how
    with tempfile.TemporaryDirectory() as d:
        llms = Path(d) / "llms"
        (llms / "qwen3_0_6b" / "build_peano" / "prefill_kernel_cache").mkdir(parents=True)
        (llms / "qwen3_0_6b" / "build_peano" / "decode_kernel_cache").mkdir(parents=True)
        for c in ("prefill_kernel_cache", "decode_kernel_cache"):
            (llms / "qwen3_0_6b" / "build_peano" / c / "manifest.json").write_text("{}")
        root = Path(d) / "compiled"
        # no w4 decode set yet: NOT compiled, and the note names compile-decode
        compiled, notes = mp.discover_compiled(("qwen3_0_6b",), root, llms_dir=llms, precision_plan="w4_decode")
        assert compiled == {}
        assert "compile-decode" in notes[("qwen3_0_6b", mp.SHIPPED_PREFILL_M, "w4_decode")]
        assert "w4_decode" in notes[("qwen3_0_6b", mp.SHIPPED_PREFILL_M, "w4_decode")]
        # the compiled w4 decode set binds with the SHIPPED prefill cache
        w4d = root / "qwen3_0_6b" / "w4_decode" / "decode_kernel_cache"
        w4d.mkdir(parents=True)
        (w4d / "manifest.json").write_text("{}")
        compiled, notes = mp.discover_compiled(("qwen3_0_6b",), root, llms_dir=llms, precision_plan="w4_decode")
        assert compiled[("qwen3_0_6b", mp.SHIPPED_PREFILL_M, "w4_decode")]["decode_cache"] == str(w4d)
        assert compiled[("qwen3_0_6b", mp.SHIPPED_PREFILL_M, "w4_decode")]["prefill_cache"].endswith("build_peano/prefill_kernel_cache")
        # bf16 discovery on the same tree is untouched: the shipped decode cache
        compiled_b, _ = mp.discover_compiled(("qwen3_0_6b",), root, llms_dir=llms, precision_plan="bf16")
        assert compiled_b[("qwen3_0_6b", mp.SHIPPED_PREFILL_M, "bf16")]["decode_cache"].endswith("build_peano/decode_kernel_cache")
    # resume identity: distinct from the bf16 qwen decode rungs by the plan column
    bound = prof.bind({("qwen3_0_6b", 2048, "w4_decode"): {"prefill_cache": "/c/p", "decode_cache": "/c/w4/d"}})
    keys = {resume.rung_key(r.mode, r.seq, r.extra) for r in bound.rungs()}
    bf16 = {resume.rung_key(r.mode, r.seq, r.extra) for r in mp.profile("model-smoke").rungs()
            if r.phase == "decode" and r.model_id == "qwen3_0_6b"}
    assert len(keys) == 3 and not keys & bf16


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"model_profiles tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())


#: `[2026-08-28]` Queue item 21, review finding 1. The counts H5's S1 clause
#: fired on, and the counts P2/P3 are scored against. They are DERIVED from the
#: planner below, never read from a table -- the first version of this test
#: inspected only the hard-coded profile, so a planner change that moved the
#: selection left the same 15 rungs passing and the gate could not tell.
H5_PLANNER_COUNTS = {
    # (phase, ubatch, context_end, precision) -> (air_launches, host_submissions)
    ("qwen3_0_6b", "prefill", 512, 512, "bf16"): (479, 85),
    ("qwen3_0_6b", "prefill", 1024, 1024, "bf16"): (479, 85),
    ("qwen3_0_6b", "prefill", 2048, 2048, "bf16"): (619, 85),
    ("qwen3_0_6b", "prefill", 512, 1024, "bf16"): (955, 169),   # 2 chunks
    ("qwen3_0_6b", "decode", 1, 2048, "bf16"): (143, 57),
    ("qwen3_0_6b", "decode", 1, 2048, "w4_decode"): (143, 57),
    ("llama32_1b", "prefill", 2048, 2048, "bf16"): (328, 49),
    ("llama32_1b", "decode", 1, 2048, "bf16"): (152, 33),
}

#: What S1 fired ON: the launch counts the newest passing rows recorded BEFORE
#: item 28's LM head went 10 launches to 3. If the planner ever agrees with these
#: again, S1 no longer fires and H5's Qwen-prefill selection is stale.
H5_PRE_ITEM28_LAUNCHES = {
    ("qwen3_0_6b", "prefill", 512, 512, "bf16"): 486,
    ("qwen3_0_6b", "prefill", 1024, 1024, "bf16"): 486,
    ("qwen3_0_6b", "prefill", 2048, 2048, "bf16"): 626,
    ("qwen3_0_6b", "prefill", 512, 1024, "bf16"): 962,
}


def _planner_counts(model_id, phase, ubatch, context_end, precision):
    """The planner's own numbers, through `shared.plan` (pure Python, no `air`,
    no torch -- checked: it pulls in no heavy module, which is why a host test
    may call it)."""
    import sys, os
    _pe = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for _p in (_pe, os.path.join(_pe, "llms")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    from shared.plan.graph import decoder_graph, QWEN3_0_6B, LLAMA32_1B
    from shared.plan.plan import plan as _plan, plan_ubatch_prefill as _ub
    from shared.plan.placement import Workload
    spec = {"qwen3_0_6b": QWEN3_0_6B, "llama32_1b": LLAMA32_1B}[model_id]
    g = decoder_graph(spec)
    if phase == "decode":
        p = _plan(g, Workload("decode", 1, context_end, 2048, precision))
    elif context_end > ubatch:
        p = _ub(g, context_end, ubatch, ctx=ubatch, precision_plan=precision)
    else:
        p = _plan(g, Workload("prefill", ubatch, context_end, ubatch, precision))
    return p.total_launches, p.total_submissions


def test_h5_selection_still_fires_where_the_profile_says_it_does():
    """`[2026-08-28]` Queue item 21, review finding 1 -- the gate that makes the
    `h5-cells` profile a claim about the WORLD rather than a transcript of one
    afternoon.

    H5's S1 clause selected the four Qwen prefill cells because item 28's LM head
    (10 launches to 3) moved the planner's count away from what the newest passing
    rows recorded. Two things must therefore stay true, and both are computed from
    `shared.plan` rather than copied from a table:

      1. the planner's counts are what H5 measured against (P2 and P3 are scored
         on exactly these numbers), and
      2. they still DIFFER from the pre-item-28 counts S1 fired on.

    If a later change moves the planner back -- or moves it anywhere else -- this
    goes red and the selection has to be re-derived, which is precisely what the
    first version of this test could not detect."""
    for key, (want_L, want_subs) in H5_PLANNER_COUNTS.items():
        got_L, got_subs = _planner_counts(*key)
        assert (got_L, got_subs) == (want_L, want_subs), (
            f"{key}: planner now says {got_L} launches / {got_subs} submissions, "
            f"H5 measured and scored against {want_L}/{want_subs}; re-derive the selection")
    for key, pre in H5_PRE_ITEM28_LAUNCHES.items():
        got_L, _ = _planner_counts(*key)
        assert got_L != pre, (
            f"{key}: the planner agrees with the pre-item-28 count {pre} again, so S1 "
            f"no longer fires here and H5's Qwen-prefill selection is stale")


def test_h5_cells_is_planner_selected_and_carries_a_derived_skip_on_purpose():
    """`[2026-08-27]` Queue item 21 (doc 56 H5). The spec's own words are
    "planner-selected cells ... plus negative controls", and the revision at doc
    56 line 859 changed H5 AWAY from a Cartesian matrix -- so the test that
    matters is that this profile is NOT one. Over the axes it spans (2 models x
    2 phases x 4 prompt lengths x 3 ubatches x 3 contexts x 2 precisions) a
    Cartesian product would be far larger than 15 rungs, and the specific cells
    absent are the ones whose planner account has not moved.

    `[2026-08-28]` The three Qwen bf16 DECODE rungs are in this profile as the
    CONTROL ARM of the w4 A/B, not because the rule selected them -- the
    re-derived selection REJECTS them (plan, bytes and timing contract all
    stand). That is stated here so the profile cannot be read as claiming they
    were selected; item 24 established that a precision A/B whose two arms are
    two sessions measures session drift as much as it measures the precision.

    The `w_bfp16_prefill` Qwen rung is carried DELIBERATELY as a derived skip:
    H5's gate is "every row `passed` or a derived skip", and while that clause is
    satisfied vacuously when no skip occurs, a walk with no skip in it cannot
    DEMONSTRATE that its skips are derived rather than asserted. Its reason must
    name the command that would make it measurable."""
    prof = mp.profile("h5-cells")
    assert prof.models == ("qwen3_0_6b", "llama32_1b"), "two models: the model axis needs more than one"
    assert prof.precision_plans_used() == ("bf16", "w_bfp16_prefill", "w4_decode")
    rungs = prof.rungs()
    assert len(rungs) == 15
    # NOT a Cartesian matrix: llama gets one prefill M and no w4 arm; qwen gets
    # no ctx-512/1024 ubatch points and no bf16-vs-w4 crossing on prefill.
    axes = (len(prof.models) * 2 * 4 * 3 * 3 * 2)
    assert len(rungs) < axes / 4, f"{len(rungs)} rungs must be far short of the {axes}-cell product"
    assert [r.case_id for r in rungs if r.model_id == "llama32_1b"] == [
        "llama32_1b/prefill/M2048/ctx2048/bf16"] + [
        f"llama32_1b/decode/M2048/ctx{c}/bf16" for c in (512, 1024, 2048)]
    # the three curves are LABELLED, never inferred
    by_curve = {}
    for r in rungs:
        by_curve.setdefault(r.curve, []).append(r.case_id)
    assert len(by_curve[mp.UBATCH]) == 1 and by_curve[mp.UBATCH] == ["qwen3_0_6b/prefill/M512/ctx1024/bf16"]
    assert len(by_curve[mp.KERNEL_SCALING]) == 5 and len(by_curve[mp.DECODE_CONTEXT]) == 9

    # every measured rung must be a cell the planner can actually account for
    for r in rungs:
        if r.model_id not in ("qwen3_0_6b", "llama32_1b"):
            continue
        key = (r.model_id, r.phase, r.ubatch_tokens, r.context_end, r.precision_plan)
        if r.precision_plan == "w_bfp16_prefill":
            continue  # the deliberate skip; the planner refuses it structurally
        probe = (r.model_id, r.phase, r.ubatch_tokens if r.phase == "prefill" else 1,
                 r.context_end if r.phase == "prefill" else 2048, r.precision_plan)
        assert probe in H5_PLANNER_COUNTS, f"{key}: no planner count pinned for a measured rung"

    # No two rungs may share a resume key: the ubatch point at (1024, 1024)
    # would be the SAME execution as the M=1024 kernel-scaling rung and the same
    # key, so it is deliberately not a separate cell.
    keys = {resume.rung_key(r.mode, r.seq, r.extra) for r in rungs}
    assert len(keys) == 15, "a collapsed key would silently drop a cell"

    # bound to everything the walk actually has, the bfp16 rung is the ONLY skip
    have = {("qwen3_0_6b", M, "bf16"): {"prefill_cache": f"/c/q{M}", "decode_cache": "/c/qd"} for M in (512, 1024, 2048)}
    have[("qwen3_0_6b", 2048, "w4_decode")] = {"prefill_cache": "/c/q2048", "decode_cache": "/c/w4d"}
    have[("llama32_1b", 2048, "bf16")] = {"prefill_cache": "/c/lp", "decode_cache": "/c/ld"}
    bound = prof.bind(have)
    skipped = [r for r in bound.rungs() if r.skip_reason]
    assert [r.case_id for r in skipped] == ["qwen3_0_6b/prefill/M2048/ctx2048/w_bfp16_prefill"]
    assert "w_bfp16_prefill" in skipped[0].skip_reason
    assert bound.expected_rows() == {
        "model_qwen3_0_6b.csv": {"rows": 11, "measured": 10, "skipped": 1},
        "model_llama32_1b.csv": {"rows": 4, "measured": 4, "skipped": 0},
    }


def test_h5_cold_is_a_control_not_a_cell_and_cannot_move_the_standing_number():
    """`[2026-08-27]` Queue item 21, the cold/warm control. `plan()` has no
    cold/warm term -- its cost model is steady state -- so the planner cannot
    select on that axis and this is a control, not a planner-selected cell.

    It is its own profile for a reason the row key makes unavoidable: the key is
    (model, phase, ubatch, context_end, precision) and does NOT carry the
    warm-up count, so a warmup-0 rung inside `h5-cells` would collide with the
    standing ctx-2048 rung. Folding it in the other way -- setting
    `decode_warmup = 0` on `h5-cells` itself -- would put the cold token inside
    the standing 32-token mean, i.e. would move the standing number by the act
    of measuring the control. Both failures are asserted here."""
    cold, cells = mp.profile("h5-cold"), mp.profile("h5-cells")
    assert cold.decode_warmup == 0, "the cold rung's first TIMED token is the first dispatch"
    assert cells.decode_warmup == 1, "the standing profile keeps its warm-up token"
    assert len(cold.rungs()) == 1
    (r,) = cold.rungs()
    assert (r.model_id, r.phase, r.context_end, r.precision_plan) == ("qwen3_0_6b", "decode", 2048, "bf16")
    assert r.n_tokens == mp.GATE_N_TOKENS, "the cold rung samples the same 32 tokens as the standing one"
    # the collision the separate profile avoids, demonstrated rather than asserted
    standing = [x for x in cells.rungs()
                if x.model_id == "qwen3_0_6b" and x.phase == "decode"
                and x.context_end == 2048 and x.precision_plan == "bf16"]
    assert len(standing) == 1
    assert resume.rung_key(r.mode, r.seq, r.extra) == resume.rung_key(
        standing[0].mode, standing[0].seq, standing[0].extra), (
        "same key by construction -- which is exactly why the control needs its own root")
