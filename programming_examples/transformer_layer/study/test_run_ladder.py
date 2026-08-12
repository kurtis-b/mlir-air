# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only checks on the sequence-ladder shape override.

    python3 study/test_run_ladder.py

No device and no compile: these cover ``run_mode._shape_for``, which is the one
place a ladder can go wrong quietly. The load-bearing one is
``test_override_does_not_mutate_the_spec``. ``SPECS`` rows are module-level
dicts, so overriding ``seq_len`` in place would rewrite the catalogue, and every
later rung in the same process would inherit the first rung's length -- a ladder
that reports four lengths and measured one, with four plausible latencies to
prove it. That failure is invisible in the output CSV, which is why it is
pinned here rather than left to review.
"""

import ast
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import results_io  # noqa: E402
import run_ladder  # noqa: E402
import run_mode  # noqa: E402

#: The operator whose catalogue row ``_spec`` stands in for.
_SPEC_OPERATOR = "coarse"


def _spec(seq=4096, emb=768):
    """A SPECS-shaped row. Deliberately not imported: opcheck_specs pulls in the
    builders, which need ml_dtypes and a toolchain, and these checks need
    neither.

    Standing in for a real row is only safe while the stand-in has the same
    shape as the row -- so
    ``test_the_stand_in_row_still_matches_the_catalogue`` re-derives that from
    ``opcheck_specs.py``'s source by ast, the idiom ``test_profiles.py`` uses
    for the same reason. Without it this fixture agrees with the catalogue
    until the catalogue moves and then agrees with history."""
    return {
        "operator": _SPEC_OPERATOR,
        "shape_key": f"{seq}x{emb}_encoder_bert",
        "shape": {"seq_len": seq, "emb_dim": emb, "ffn_dim": 4 * emb},
    }


def _catalogue_rows():
    """``[(operator, shape_key, {shape key: value})]`` parsed from the source.

    Text, not an import: see ``_spec``. Only constant-valued shape entries are
    returned, which is every one the catalogue writes today.
    """
    source = open(os.path.join(_EXAMPLE, "opcheck_specs.py"), encoding="utf-8").read()
    rows = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Dict):
            continue
        entries = {
            k.value: v
            for k, v in zip(node.keys, node.values)
            if isinstance(k, ast.Constant)
        }
        operator = entries.get("operator")
        shape_key = entries.get("shape_key")
        shape = entries.get("shape")
        if not (
            isinstance(operator, ast.Constant)
            and isinstance(shape_key, ast.Constant)
            and isinstance(shape, ast.Dict)
        ):
            continue
        rows.append(
            (
                operator.value,
                shape_key.value,
                {
                    k.value: v.value
                    for k, v in zip(shape.keys, shape.values)
                    if isinstance(k, ast.Constant) and isinstance(v, ast.Constant)
                },
            )
        )
    return rows


def test_the_stand_in_row_still_matches_the_catalogue():
    """``_spec`` is a hand-written copy of a real SPECS row. Pin it to the real
    one, or these checks go on testing a row shape nobody ships."""
    rows = _catalogue_rows()
    assert rows, "no SPECS rows were parsed; the ast walk broke, not the catalogue"
    mine = [r for r in rows if r[0] == _SPEC_OPERATOR]
    assert mine, (
        f"opcheck_specs.py no longer has a {_SPEC_OPERATOR!r} row, so _spec() "
        f"stands in for nothing. Operators present: {sorted({r[0] for r in rows})}"
    )
    _, real_key, real_shape = mine[0]
    fixture = _spec()

    # 1. The keys _shape_for reads must all be present on a real row.
    assert set(fixture) <= set(("operator", "shape_key", "shape"))
    assert set(fixture["shape"]) <= set(real_shape), (
        f"_spec() supplies shape keys {sorted(set(fixture['shape']) - set(real_shape))} "
        f"that the real {_SPEC_OPERATOR} row does not have {sorted(real_shape)}"
    )

    # 2. The key format the fixture asserts against is the catalogue's format.
    seq, emb = real_shape["seq_len"], real_shape["emb_dim"]
    assert real_key == f"{seq}x{emb}_encoder_bert", (
        f"the catalogue's shape_key format moved to {real_key!r}; every "
        "expected key in this module is written in the old one"
    )

    # 3. The ffn_dim == 4 * emb_dim relation the fixture bakes in.
    assert real_shape["ffn_dim"] == 4 * real_shape["emb_dim"], (
        f"the real row has ffn_dim {real_shape['ffn_dim']} against emb_dim "
        f"{real_shape['emb_dim']}; _spec() hardcodes 4x"
    )

    # 4. The default width the expected keys below are written at.
    assert emb == 768, (
        f"the {_SPEC_OPERATOR} row is now emb_dim {emb}; every expected "
        "shape_key in this module says 768"
    )


def test_no_override_returns_the_specs_own_shape():
    spec = _spec()
    shape, key, variant = run_mode._shape_for(spec, None)
    assert shape == spec["shape"]
    assert key == spec["shape_key"], "the catalogue's own key must survive verbatim"


def test_override_sets_the_length_and_derives_a_key():
    shape, key, variant = run_mode._shape_for(_spec(), 512)
    assert shape["seq_len"] == 512
    assert key == "512x768_encoder_bert"
    assert shape["emb_dim"] == 768, "only the length moves"
    assert shape["ffn_dim"] == 3072


def test_override_does_not_mutate_the_spec():
    """The bug this module exists for: a ladder reporting N lengths, measuring 1."""
    spec = _spec()
    run_mode._shape_for(spec, 512)
    assert spec["shape"]["seq_len"] == 4096, "the catalogue row was rewritten"
    assert spec["shape_key"] == "4096x768_encoder_bert"


def test_successive_rungs_are_independent():
    spec = _spec()
    keys = [run_mode._shape_for(spec, s)[1] for s in (512, 1024, 2048)]
    lens = [run_mode._shape_for(spec, s)[0]["seq_len"] for s in (512, 1024, 2048)]
    assert keys == [
        "512x768_encoder_bert",
        "1024x768_encoder_bert",
        "2048x768_encoder_bert",
    ]
    assert lens == [512, 1024, 2048]


def test_a_family_overrides_the_width_and_the_key_says_so():
    """`[2026-08-12]` The coverage-sweep unblock. `_shape_for` had varied only
    `seq_len`, which is the whole reason five of six declared families were
    called unreachable -- the registry rows for hidden 512 and 1024 had existed
    since 2026-08-07."""
    shape, key, variant = run_mode._shape_for(_spec(), 1024, "tinybert_512")
    assert (shape["emb_dim"], shape["ffn_dim"], shape["num_heads"]) == (512, 2048, 8)
    assert shape["head_dim"] == 64, "derived from hidden//heads, never carried"
    assert key == "1024x512_encoder_bert"
    assert variant == "encoder_bert"

    shape, key, _ = run_mode._shape_for(_spec(), 2048, "baseline_1024")
    assert (shape["emb_dim"], shape["ffn_dim"], shape["num_heads"]) == (1024, 4096, 16)
    assert key == "2048x1024_encoder_bert"


def test_a_family_override_does_not_mutate_the_spec_either():
    """`test_override_does_not_mutate_the_spec`'s bug, on the second axis: a
    width leaking into the module-level SPECS row would make every later rung in
    the process a different family than its row claims."""
    spec = _spec()
    run_mode._shape_for(spec, 512, "tinybert_512")
    assert spec["shape"]["emb_dim"] == 768
    assert spec["shape"]["ffn_dim"] == 3072


def test_a_decoder_family_is_refused_rather_than_run_as_an_encoder():
    """The worst outcome available here is a valid-looking bidirectional
    measurement stamped `decoder_gpt2`, because nothing downstream could detect
    it. So the variant is carried into the row AND the run is refused."""
    _, key, variant = run_mode._shape_for(_spec(), 512, "gpt2_small_768")
    assert variant == "decoder_gpt2"
    assert key == "512x768_decoder_gpt2"
    assert variant in run_mode.UNBUILDABLE_VARIANTS

    # And `run` must actually branch on it. Audited from the source rather than
    # called: `run` imports opcheck_specs, which pulls in the builders and needs
    # a toolchain this suite deliberately does not have. A refusal that is
    # declared and never consulted is the item 19 defect shape.
    source = open(os.path.join(_HERE, "run_mode.py"), encoding="utf-8").read()
    assert "if variant in UNBUILDABLE_VARIANTS:" in source, (
        "run_mode.run no longer refuses an unbuildable variant; a decoder "
        "family would be measured as an encoder and stamped `decoder_gpt2`"
    )
    # ...before anything is prepared, or the refusal costs a compile.
    assert source.index("if variant in UNBUILDABLE_VARIANTS:") < source.index(
        'prepared = spec["prepare"](shape)'
    )


def test_key_falls_back_when_the_shape_names_hidden_size():
    """Some rows carry hidden_size rather than emb_dim; the key must not say '?'."""
    spec = {
        "operator": "coarse",
        "shape_key": "4096x768_encoder_bert",
        "shape": {"seq_len": 4096, "hidden_size": 768},
    }
    _, key, _variant = run_mode._shape_for(spec, 1024)
    assert key == "1024x768_encoder_bert"


def _fake_rung(seen):
    """Stands in for the one function that dispatches, so no device is needed."""

    def fake(mode, seq, study_id, warmup, samples, rps, scratch, family=None):
        seen.append((mode, seq))
        row = run_ladder.schema.empty_row("results")
        row["execution_mode"] = run_ladder.schema.EXECUTION_MODE_CSV[mode]
        row["study_id"] = study_id
        row["seq_len"] = seq
        row["study_case_label"] = f"{mode} {seq}"
        row["run_status"] = "passed"
        row["failure_message"] = ""
        return row

    return fake


def _walk_with_stub(seen, **kwargs):
    original, run_ladder._rung = run_ladder._rung, _fake_rung(seen)
    try:
        return run_ladder.walk(**kwargs)
    finally:
        run_ladder._rung = original


def test_a_skipped_rung_is_written_and_never_run():
    """The skipped rung must not reach `_rung` -- the only thing that dispatches.

    Asserted on the CALL LIST rather than only on the row, because a runner that
    ran the rung and then relabelled the row would produce identical output and
    cost the whole compile.
    """
    seen = []
    with tempfile.TemporaryDirectory() as d:
        rows = _walk_with_stub(
            seen,
            modes=["fused"],
            seqs=[512, 2048],
            out_dir=d,
            study_id="test",
            warmup=0,
            samples=1,
            runs_per_sample=1,
            skip_reason=lambda m, s: "out of range" if s > 1024 else None,
        )
        assert [r["run_status"] for r in rows] == ["passed", "skipped"]
        back = results_io.read_rows(os.path.join(d, "fused.csv"))
        assert [r["run_status"] for r in back] == ["passed", "skipped"]
        assert back[1]["failure_message"] == "skipped: out of range"
        assert [r["seq_len"] for r in back] == ["512", "2048"]
    assert seen == [("fused", 512)], "the 2048 rung must not have been dispatched"


def test_a_skipped_rung_is_not_a_failed_run():
    """`main`'s exit code: skipped rungs are not regressions."""
    seen = []
    with tempfile.TemporaryDirectory() as d:
        rows = _walk_with_stub(
            seen,
            modes=["fused"],
            seqs=[2048],
            out_dir=d,
            study_id="test",
            warmup=0,
            samples=1,
            runs_per_sample=1,
            skip_reason=lambda m, s: "out of range",
        )
    passed = [r for r in rows if r["run_status"] == "passed"]
    skipped = [r for r in rows if r["run_status"] == "skipped"]
    assert len(passed) + len(skipped) == len(rows) and not passed
    assert seen == []


def test_no_skip_callback_is_the_existing_behaviour():
    """Every existing caller passes nothing and must be unaffected."""
    seen = []
    with tempfile.TemporaryDirectory() as d:
        rows = _walk_with_stub(
            seen,
            modes=["coarse"],
            seqs=[512, 1024],
            out_dir=d,
            study_id="test",
            warmup=0,
            samples=1,
            runs_per_sample=1,
        )
    assert seen == [("coarse", 512), ("coarse", 1024)]
    assert [r["run_status"] for r in rows] == ["passed", "passed"]


def test_a_reused_rung_is_written_verbatim_and_never_run():
    """`test_a_skipped_rung_is_written_and_never_run`'s argument, for reuse.

    Asserted on the CALL LIST, because a walk that re-ran the rung and wrote a
    row that happened to look similar would cost the whole compile and be
    invisible in the output. And asserted BYTE FOR BYTE, because
    `resume.fidelity_problems` compares digests afterwards: a row rebuilt rather
    than copied would fail that audit as if the walker had cheated.
    """
    seen = []
    carried = run_ladder.schema.empty_row("results")
    carried["execution_mode"] = "hybrid"
    carried["study_id"] = "an-earlier-session"
    carried["seq_len"] = 512
    carried["study_case_label"] = "coarse 512"
    carried["avg_latency_ms"] = 3.25
    carried["run_status"] = "passed"
    carried["failure_message"] = ""

    with tempfile.TemporaryDirectory() as d:
        rows = _walk_with_stub(
            seen,
            modes=["coarse"],
            seqs=[512, 1024],
            out_dir=d,
            study_id="today",
            warmup=0,
            samples=1,
            runs_per_sample=1,
            reuse={("coarse", 512): carried},
        )
        back = results_io.read_rows(os.path.join(d, "coarse.csv"))
    assert seen == [("coarse", 1024)], "the carried rung must not have dispatched"
    assert rows[0]["study_id"] == "an-earlier-session"
    assert back[0]["avg_latency_ms"] == "3.25"
    assert back[0]["study_id"] == "an-earlier-session"


def test_a_skip_beats_a_reuse():
    """A rung the profile now refuses must not be resurrected from a row
    measured back when it did not. `resume.plan` excludes those too; checking
    here as well means the two cannot disagree in the resurrecting direction."""
    seen = []
    carried = run_ladder.schema.empty_row("results")
    carried["execution_mode"] = "fused_elf"
    carried["study_id"] = "before-the-bound-moved"
    carried["seq_len"] = 2048
    carried["study_case_label"] = "fused 2048"
    carried["run_status"] = "passed"
    carried["failure_message"] = ""

    with tempfile.TemporaryDirectory() as d:
        rows = _walk_with_stub(
            seen,
            modes=["fused"],
            seqs=[2048],
            out_dir=d,
            study_id="today",
            warmup=0,
            samples=1,
            runs_per_sample=1,
            skip_reason=lambda m, s: "the bound moved",
            reuse={("fused", 2048): carried},
        )
    assert seen == []
    assert rows[0]["run_status"] == "skipped"
    assert rows[0]["study_id"] == "today"


def test_the_per_rung_hook_sees_every_rung_with_its_source():
    """The ledger is written from this, so a source it never reports is a rung
    that lands on disk attributed to nobody."""
    events = []
    carried = run_ladder.schema.empty_row("results")
    carried["execution_mode"] = "fused_elf"
    carried["study_id"] = "earlier"
    carried["seq_len"] = 512
    carried["study_case_label"] = "fused 512"
    carried["run_status"] = "passed"
    carried["failure_message"] = ""

    with tempfile.TemporaryDirectory() as d:
        _walk_with_stub(
            [],
            modes=["fused"],
            seqs=[512, 1024, 2048],
            out_dir=d,
            study_id="test",
            warmup=0,
            samples=1,
            runs_per_sample=1,
            skip_reason=lambda m, s: "out of range" if s > 1024 else None,
            reuse={("fused", 512): carried},
            on_rung=lambda m, s, row, source: events.append((m, s, source)),
        )
    assert events == [
        ("fused", 512, "reused"),
        ("fused", 1024, "measured"),
        ("fused", 2048, "skipped"),
    ]
    assert set(e[2] for e in events) <= set(run_ladder.schema.RUNG_SOURCES)


def test_neither_new_callback_changes_the_existing_behaviour():
    """Every caller predating resume passes neither and must be unaffected."""
    seen = []
    with tempfile.TemporaryDirectory() as d:
        rows = _walk_with_stub(
            seen,
            modes=["coarse"],
            seqs=[512, 1024],
            out_dir=d,
            study_id="test",
            warmup=0,
            samples=1,
            runs_per_sample=1,
        )
    assert seen == [("coarse", 512), ("coarse", 1024)]
    assert [r["study_id"] for r in rows] == ["test", "test"]


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"ladder tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
