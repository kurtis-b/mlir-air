# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only checks on the suite profiles.

    python3 study/test_profiles.py

Two of these are claims about OTHER files rather than about this module, and
they are re-derived from those files' source text rather than trusted -- the
idiom ``test_attention_path.py`` established here for the same reason:

  ``test_reachable_family_is_the_one_the_runner_can_build`` parses
  ``opcheck_specs.py`` with ``ast`` and asserts every whole-layer SPECS row is
  ``emb_dim 768``. If a row at 512 or 1024 ever lands, this fails and
  ``UNREACHABLE_FAMILIES`` has to be revisited -- which is the point. It imports
  nothing from that module: ``opcheck_specs`` pulls in the builders, which need
  ml_dtypes and a toolchain, and this suite needs neither.

  ``test_workload_variant_is_hardcoded`` reads ``run_mode.py`` for the literal
  assignment that makes the three decoder families unreachable at any width.

The third load-bearing one is ``test_fused_bound_is_applied_in_both_directions``:
a skip rule with no case that skips and no case that does not is a rule nobody
has shown discriminates.
"""

import ast
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import cases  # noqa: E402
import profiles  # noqa: E402
import run_ladder  # noqa: E402
import schema  # noqa: E402


def _files_walk_writes(prof):
    """The CSVs a real ``run_ladder.walk`` of ``prof`` puts on disk.

    ``_rung`` -- the only function that dispatches -- is stubbed, so this walks
    the real writer with no device and no compile. One sequence length is
    enough: ``walk`` opens one file per mode regardless of ladder depth, and a
    full walk would only make the check slower.
    """
    original = run_ladder._rung

    def _stub(mode, seq, study_id, warmup, samples, rps, scratch):
        row = schema.empty_row("results")
        row["execution_mode"] = schema.EXECUTION_MODE_CSV[mode]
        row["study_id"] = study_id
        row["seq_len"] = seq
        row["study_case_label"] = f"{mode} {seq}"
        row["run_status"] = "passed"
        row["failure_message"] = ""
        return row

    run_ladder._rung = _stub
    try:
        with tempfile.TemporaryDirectory() as out_dir:
            run_ladder.walk(
                modes=list(prof.modes),
                seqs=list(prof.seqs)[:1],
                out_dir=out_dir,
                study_id="test-expected-files",
                warmup=0,
                samples=1,
                runs_per_sample=1,
                skip_reason=profiles.skip_reason,
            )
            return sorted(os.listdir(out_dir))
    finally:
        run_ladder._rung = original

#: The whole-layer modes a profile can name. Operator rows for single kernels
#: (``ffn``, ``softmax``, ...) are not profile rungs and are not checked here.
_LAYER_OPERATORS = frozenset(profiles.PROFILE_MODES) | {
    "block",
    "coarse_c2",
    "coarse_c3",
}


def _specs_emb_dims() -> dict[str, set]:
    """``operator -> {emb_dim seen}``, parsed from opcheck_specs.py's source."""
    source = open(os.path.join(_EXAMPLE, "opcheck_specs.py"), encoding="utf-8").read()
    found: dict[str, set] = {}
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Dict):
            continue
        entries = {
            k.value: v
            for k, v in zip(node.keys, node.values)
            if isinstance(k, ast.Constant)
        }
        operator = entries.get("operator")
        shape = entries.get("shape")
        if not isinstance(operator, ast.Constant) or not isinstance(shape, ast.Dict):
            continue
        widths = {
            v.value
            for k, v in zip(shape.keys, shape.values)
            if isinstance(k, ast.Constant)
            and k.value in ("emb_dim", "hidden_size")
            and isinstance(v, ast.Constant)
        }
        found.setdefault(operator.value, set()).update(widths)
    return found


def test_reachable_family_is_the_one_the_runner_can_build():
    """M5 as a test: one of six families is reachable, and the source says so."""
    widths = _specs_emb_dims()
    layer_widths = {op: w for op, w in widths.items() if op in _LAYER_OPERATORS and w}
    assert layer_widths, "no whole-layer SPECS rows were parsed; the ast walk broke"
    every = set().union(*layer_widths.values())
    assert every == {768}, (
        f"whole-layer SPECS rows now span widths {sorted(every)}, not just 768. "
        "profiles.UNREACHABLE_FAMILIES claims 512 and 1024 are out of reach; "
        "revisit it rather than deleting this assert."
    )
    spec = cases.FAMILY_SPECS[profiles.REACHABLE_FAMILY]
    assert spec.hidden_size == 768


def test_workload_variant_is_hardcoded():
    """The decoder half of M5. Read from run_mode.py, not asserted about it."""
    source = open(os.path.join(_HERE, "run_mode.py"), encoding="utf-8").read()
    assert 'row["workload_variant"] = "encoder_bert"' in source, (
        "run_mode no longer hardcodes the workload variant; the gpt2_* entries "
        "in profiles.UNREACHABLE_FAMILIES need re-deriving"
    )


def test_unreachable_families_cover_every_declared_family():
    named = set(profiles.UNREACHABLE_FAMILIES) | {profiles.REACHABLE_FAMILY}
    assert named == set(cases.FAMILY_IDS), (
        "a family was added to the case matrix without a reachability verdict; "
        "an undeclared family silently disappears from every run report"
    )
    assert profiles.REACHABLE_FAMILY not in profiles.UNREACHABLE_FAMILIES


def test_fused_bound_is_applied_in_both_directions():
    low, high = profiles.FUSED_SEQ_RANGE
    assert profiles.skip_reason("fused", low) is None
    assert profiles.skip_reason("fused", high) is None
    assert profiles.skip_reason("fused", low // 2)
    assert profiles.skip_reason("fused", high * 2)
    # and no other mode is bounded, at any declared ladder point
    for mode in profiles.PROFILE_MODES:
        if mode == "fused":
            continue
        for seq in cases.SEQUENCE_LADDER:
            assert profiles.skip_reason(mode, seq) is None, (
                f"{mode} at {seq} was declared inapplicable; only artifact-backed "
                "bounds belong in skip_reason -- a rung that MIGHT fail must run"
            )


def test_skip_reason_names_the_source_of_the_bound():
    reason = profiles.skip_reason("fused", 4096)
    assert "fused.py" in reason and "256..1024" in reason


def test_expected_rows_are_derived_and_balance():
    for name, prof in profiles.PROFILES.items():
        expectation = prof.expected_rows()
        assert set(expectation) == set(prof.expected_files()), name
        total = 0
        for rel, counts in expectation.items():
            assert counts["rows"] == counts["measured"] + counts["skipped"], rel
            assert counts["rows"] == len(prof.seqs), rel
            total += counts["rows"]
        assert total == len(prof.rungs()), name


def test_expected_files_are_one_per_mode_and_match_run_ladder():
    """The expectation must match ``run_ladder.walk``'s OUTPUT, so take the
    output rather than a typed list of it.

    A transcribed ``["coarse.csv", ...]`` states the claim and does not check
    it: it agrees with ``walk`` until ``walk``'s naming or mode order moves,
    and then it agrees with the day it was written -- while the manifest that
    consumes ``expected_files()`` starts looking for files no run produces and
    reports the tree incomplete for a reason nothing here would catch.
    """
    prof = profiles.profile("ladder")
    written = _files_walk_writes(prof)
    assert sorted(prof.expected_files()) == written, (
        f"the profile expects {sorted(prof.expected_files())} and "
        f"run_ladder.walk wrote {written}"
    )
    # One CSV per mode, in the profile's own mode order -- the manifest reads
    # this list positionally nowhere, but the gate's contract says mode order.
    assert prof.expected_files() == [f"{mode}.csv" for mode in prof.modes]
    assert len(written) == len(set(written)) == len(prof.modes)


def test_every_profile_expects_exactly_what_a_walk_of_it_writes():
    """Not just ``ladder``: the smoke and full profiles too, since the manifest
    gates each of them on ``expected_files()``."""
    for name in sorted(profiles.PROFILES):
        prof = profiles.profile(name)
        assert sorted(prof.expected_files()) == _files_walk_writes(prof), name


def test_the_ladder_profile_skips_exactly_the_two_out_of_range_fused_rungs():
    counts = profiles.profile("ladder").expected_rows()
    assert counts["fused.csv"] == {"rows": 4, "measured": 2, "skipped": 2}
    for rel in ("coarse.csv", "offload.csv", "runlist.csv"):
        assert counts[rel] == {"rows": 4, "measured": 4, "skipped": 0}


def test_the_smoke_profile_skips_nothing():
    counts = profiles.profile("smoke").expected_rows()
    assert all(c["skipped"] == 0 for c in counts.values())
    assert sum(c["rows"] for c in counts.values()) == 4


def test_the_full_profile_walks_the_declared_ladder():
    prof = profiles.profile("full")
    assert prof.seqs == cases.SEQUENCE_LADDER
    counts = prof.expected_rows()
    # 64 and 128 are below fused's floor, 2048..16384 above its ceiling.
    assert counts["fused.csv"] == {"rows": 9, "measured": 3, "skipped": 6}
    assert sum(c["rows"] for c in counts.values()) == 36


def test_modes_are_validated_through_the_schema():
    for prof in profiles.PROFILES.values():
        for mode in prof.modes:
            assert cases.canonical_execution_mode(mode) in schema.EXECUTION_MODES
    try:
        profiles.Profile("bad", "d", ("nonesuch",), (512,))
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown mode must not build a profile")


def test_an_unknown_family_is_refused():
    try:
        profiles.Profile("bad", "d", ("coarse",), (512,), family="nonesuch")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown family must not build a profile")


def test_unknown_profile_raises_rather_than_guessing():
    try:
        profiles.profile("nightly")
    except ValueError as exc:
        assert "known are" in str(exc)
    else:
        raise AssertionError("an unknown profile name must raise")


def test_summary_records_what_was_not_walked():
    summary = profiles.profile("ladder").summary()
    assert set(summary["families_not_walked"]) == set(profiles.UNREACHABLE_FAMILIES)
    assert len(summary["skipped_rungs"]) == 2
    assert summary["rung_count"] == 16


def test_cli_lists_and_plans():
    assert profiles.main(["--list"]) == 0
    assert profiles.main(["--profile", "ladder"]) == 0
    assert profiles.main(["--profile", "ladder", "--expect"]) == 0


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"profile tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
