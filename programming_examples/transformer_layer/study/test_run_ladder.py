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

import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import results_io  # noqa: E402
import run_ladder  # noqa: E402
import run_mode  # noqa: E402


def _spec(seq=4096, emb=768):
    """A SPECS-shaped row. Deliberately not imported: opcheck_specs pulls in the
    builders, which need ml_dtypes and a toolchain, and these checks need
    neither."""
    return {
        "operator": "coarse",
        "shape_key": f"{seq}x{emb}_encoder_bert",
        "shape": {"seq_len": seq, "emb_dim": emb, "ffn_dim": 4 * emb},
    }


def test_no_override_returns_the_specs_own_shape():
    spec = _spec()
    shape, key = run_mode._shape_for(spec, None)
    assert shape == spec["shape"]
    assert key == spec["shape_key"], "the catalogue's own key must survive verbatim"


def test_override_sets_the_length_and_derives_a_key():
    shape, key = run_mode._shape_for(_spec(), 512)
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


def test_key_falls_back_when_the_shape_names_hidden_size():
    """Some rows carry hidden_size rather than emb_dim; the key must not say '?'."""
    spec = {
        "operator": "coarse",
        "shape_key": "4096x768_encoder_bert",
        "shape": {"seq_len": 4096, "hidden_size": 768},
    }
    _, key = run_mode._shape_for(spec, 1024)
    assert key == "1024x768_encoder_bert"


def _fake_rung(seen):
    """Stands in for the one function that dispatches, so no device is needed."""

    def fake(mode, seq, study_id, warmup, samples, rps, scratch):
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


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"ladder tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
