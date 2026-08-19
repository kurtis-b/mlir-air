# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only checks on the suite profiles.

    python3 study/test_profiles.py

Several of these are claims about OTHER files rather than about this module, and
they are re-derived from those files rather than trusted -- the idiom
``test_attention_path.py`` established here for the same reason.

`[2026-08-12]` AND ONE OF THEM RE-DERIVED THE WRONG FILE, WHICH COST A PHASE ITS
SCOPING. ``test_reachable_family_is_the_one_the_runner_can_build`` parsed
``opcheck_specs.py`` and asserted every whole-layer SPECS row was ``emb_dim
768``, concluding that hidden 512 and 1024 were out of reach and that widening
the matrix needed a Phase-C-sized sweep (doc 34 M5, sized against C4's 504 + 66
minutes). The assertion was true and the conclusion was false: the widths that
decide reachability are the REGISTRY's, and the baseline_512 and baseline_1024
sweeps landed on 2026-08-07. A test that re-derives a claim from a source is
only as good as its choice of source.

  ``test_reachability_is_re_derived_from_the_registry_not_asserted`` now reads
  ``kernel_registry/details/GEMM_bf16_in_bf16_out.json`` -- every projection
  triple a reachable family implies must be present, no unreachable family may
  cite a COVERAGE reason, and every declared method gap must still be a gap.
  Pure JSON: ``opcheck_specs`` pulls in the builders, which need ml_dtypes and
  a toolchain, and this suite needs neither.

  ``test_every_layer_specs_row_is_still_written_at_one_width`` keeps the ast
  walk for the narrower thing the SPECS width really decides -- ``_spec_for``
  returns the first ``operator`` match, so a second row at another width would
  be unreachable code.

  ``test_a_decoder_variant_is_refused_by_name_not_omitted`` reads
  ``run_mode.py`` for the refusal that keeps a decoder family from being
  measured as an encoder under a causal name.

The load-bearing bound check is ``test_fused_bound_is_applied_in_both
_directions``: a skip rule with no case that skips and no case that does not is
a rule nobody has shown discriminates.
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

    def _stub(mode, seq, study_id, warmup, samples, rps, scratch, family=None):
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
                skip_reason=prof.skip_rule(),
                family=prof.family,
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


def test_every_layer_specs_row_is_still_written_at_one_width():
    """The fact the width override RESTS on, kept as a check rather than dropped.

    `[2026-08-12]` This assertion used to BE the reachability claim -- "every
    whole-layer SPECS row is emb_dim 768, so 512 and 1024 are out of reach". It
    was a correct reading of the wrong file: the widths that matter are the
    REGISTRY's, and those landed 2026-08-07. What the SPECS width still decides
    is something narrower and worth keeping: ``_shape_for`` overrides a row's
    width from the family, so a SECOND row for the same operator at a different
    width would be dead code that ``_spec_for`` (which returns the first
    ``operator`` match) could never select. If one lands, this fails, and the
    right fix is to make ``_spec_for`` key on width -- not to delete the assert.
    """
    widths = _specs_emb_dims()
    layer_widths = {op: w for op, w in widths.items() if op in _LAYER_OPERATORS and w}
    assert layer_widths, "no whole-layer SPECS rows were parsed; the ast walk broke"
    for operator, seen in sorted(layer_widths.items()):
        assert len(seen) == 1, (
            f"opcheck_specs has {operator!r} rows at widths {sorted(seen)}. "
            "run_mode._spec_for returns the FIRST operator match, so every row "
            "after the first is unreachable; make it key on width."
        )


def test_reachability_is_re_derived_from_the_registry_not_asserted():
    """THE CLAIM THIS MODULE EXISTS FOR, and the one that was wrong for a phase.

    ``UNREACHABLE_FAMILIES`` used to say ``tinybert_512`` and ``baseline_1024``
    "need registry coverage at hidden 512/1024", and doc 34 M5 sized the fix
    against C4's 504 + 66 minutes on the strength of it. The coverage had been
    in the tree since 2026-08-07 and nothing had read the file.

    So this reads the file. Every projection triple the case matrix implies for
    a family claimed REACHABLE must be in the registry; a family claimed
    unreachable for a COVERAGE reason must actually be missing rows. Pure JSON,
    no builder import, so the suite stays hermetic and under a second.
    """
    import json

    registry = os.path.join(
        os.path.dirname(_EXAMPLE), "kernel_registry", "details",
        "GEMM_bf16_in_bf16_out.json",
    )
    with open(registry, encoding="utf-8") as handle:
        rows = json.load(handle)["shapes"]
    have = {(r["M"], r["K"], r["N"]) for r in rows}
    assert len(have) > 100, f"only {len(have)} registry rows parsed; the reader broke"

    # (K multiple, N multiple) of hidden, per projection role -- taken from the
    # table that also records WHICH METHOD each role's consumer pins, rather
    # than retyped here. cases.py's own FLOP derivation names the same four.
    roles = tuple(profiles.PINNED_PROJECTION_METHODS)
    for family in profiles.REACHABLE_FAMILIES:
        h = cases.FAMILY_SPECS[family].hidden_size
        missing = [
            (seq, h * km, h * nm)
            for km, nm in roles
            for seq in cases.SEQUENCE_LADDER
            if (seq, h * km, h * nm) not in have
        ]
        assert not missing, (
            f"{family} is declared reachable and the registry is missing "
            f"{len(missing)} of its projection triples, e.g. {missing[:3]}. "
            "A profile that walks it would fail at gemm_config, not at the device."
        )

    # ...and the TRIPLE being present is the weaker half. `[2026-08-12]` The one
    # entry KNOWN_REGISTRY_GAPS ever held was a triple-vs-method confusion in the
    # other direction, so the check that a reachable family is really reachable
    # is the PINNED METHOD being present, not the row existing.
    by_method = {(r["M"], r["K"], r["N"]): set(r.get("methods", {})) for r in rows}
    for family in profiles.REACHABLE_FAMILIES:
        h = cases.FAMILY_SPECS[family].hidden_size
        for (km, nm), (method, pinned_by) in profiles.PINNED_PROJECTION_METHODS.items():
            for seq in cases.SEQUENCE_LADDER:
                triple = (seq, h * km, h * nm)
                assert method in by_method[triple], (
                    f"{family} is declared reachable but {triple} has no "
                    f"{method!r} row (it has {sorted(by_method[triple])}). "
                    f"{pinned_by} pins that method and raises without it, so "
                    f"this rung fails at resolution rather than at the device."
                )

    # ...and the converse, so this cannot pass by declaring nothing reachable.
    for family, reason in profiles.UNREACHABLE_FAMILIES.items():
        assert "registry" not in reason and "coverage" not in reason, (
            f"{family} is refused for a COVERAGE reason ({reason!r}). Every "
            "encoder width in the matrix has full registry coverage; a coverage "
            "reason here is the stale claim this test exists to prevent."
        )


def test_a_declared_registry_gap_must_be_one_some_mode_actually_demands():
    """A missing method is not a gap unless something ASKS for it.

    `[2026-08-12]` THIS TEST USED TO ASSERT ONLY THE CONVERSE, AND THE ONE ENTRY
    IT GUARDED WAS INERT. It checked that each declared gap's method was still
    absent from the registry -- true, and not the question. ``2048x1024x3072``
    really has no ``drain`` row, and the entry concluded that ``offload`` and
    ``runlist`` therefore fail at seq 2048 on ``baseline_1024``. Neither mode
    resolves a ``3h`` shape at all: both chains are ``(seq, h, h)``,
    ``(seq, h, 4h)``, ``(seq, 4h, h)``. The only consumer of ``(seq, h, 3h)`` is
    ``qkv_proj``, which pins ``fused-cast``, which is present.

    So the old assertion would have gone on passing forever while the entry it
    protected described a failure that does not happen -- and doc 50 cited that
    entry as the reason ``baseline_1024`` was 35/36 and went unwalked.

    Both directions now:

    - a declared gap's method must still be MISSING (the old check, kept -- a
      warning about a hole somebody swept is worse than no warning), and
    - the (triple, method) must be one ``PINNED_PROJECTION_METHODS`` says a
      consumer actually pins. A gap nothing demands is not a gap, and filing one
      costs a walk that never happens.

    The negative control is the deleted entry itself: feeding
    ``{"triple": (2048, 1024, 3072), "method": "drain", ...}`` back in fails the
    second clause, because no role at ``N = 3h`` pins ``drain``.
    """
    import json

    registry = os.path.join(
        os.path.dirname(_EXAMPLE), "kernel_registry", "details",
        "GEMM_bf16_in_bf16_out.json",
    )
    with open(registry, encoding="utf-8") as handle:
        rows = json.load(handle)["shapes"]
    by_method = {(r["M"], r["K"], r["N"]): set(r.get("methods", {})) for r in rows}

    for gap in profiles.KNOWN_REGISTRY_GAPS:
        triple = tuple(gap["triple"])
        assert triple in by_method, f"{triple} is not in the registry at all"
        assert gap["method"] not in by_method[triple], (
            f"{gap['family']}'s recorded gap is closed: {triple} now has a "
            f"{gap['method']!r} row. Delete the KNOWN_REGISTRY_GAPS entry -- a "
            "warning about a hole somebody swept is worse than no warning."
        )
        assert gap["family"] in profiles.REACHABLE_FAMILIES

        # ...and the half that was missing. Which role is this triple, at this
        # family's width, and does that role's consumer pin this method?
        hidden = cases.FAMILY_SPECS[gap["family"]].hidden_size
        seq, k, n = triple
        role = (k // hidden, n // hidden)
        assert k % hidden == 0 and n % hidden == 0 and role in (
            profiles.PINNED_PROJECTION_METHODS
        ), (
            f"{gap['family']}'s gap names {triple}, which is not a projection "
            f"shape of a family whose hidden size is {hidden} "
            f"(roles are {sorted(profiles.PINNED_PROJECTION_METHODS)}). A gap "
            "against a shape no mode resolves is not a gap."
        )
        pinned, pinned_by = profiles.PINNED_PROJECTION_METHODS[role]
        assert gap["method"] == pinned, (
            f"{gap['family']}'s gap says {gap['method']!r} is missing at "
            f"{triple}, but the consumer of that role pins {pinned!r} "
            f"({pinned_by}). Nothing asks for {gap['method']!r} there, so the "
            "absent row costs no rung -- this is the inert entry that made "
            "baseline_1024 read as 35/36 and go unwalked."
        )


def test_the_pinned_methods_are_re_derived_from_the_modules_that_pin_them():
    """``PINNED_PROJECTION_METHODS`` is a claim about two OTHER files.

    Stated in ``profiles.py`` so the reachability check can use it without
    importing the builders (which need ml_dtypes and a toolchain this suite
    deliberately does not have), and therefore re-derived here by ``ast`` from
    the two modules that really carry the pins. A table copied out of a source
    file agrees with the day it was copied; this fails instead.
    """
    qkv = os.path.join(_EXAMPLE, "builders", "qkv_proj.py")
    source = open(qkv, encoding="utf-8").read()
    scratch = None
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "SCRATCH_METHOD" for t in node.targets
        ):
            scratch = node.value.value
    assert scratch, "qkv_proj.py no longer assigns SCRATCH_METHOD at module scope"
    assert profiles.PINNED_PROJECTION_METHODS[(1, 3)][0] == scratch, (
        f"qkv_proj pins {scratch!r} and the table says "
        f"{profiles.PINNED_PROJECTION_METHODS[(1, 3)][0]!r}"
    )

    # offload's shared chain re-resolves a fused-cast winner to one named
    # method. Take the name from the call rather than from the comment above it.
    offload = os.path.join(_EXAMPLE, "pattern", "offload", "offload.py")
    source = open(offload, encoding="utf-8").read()
    pinned = {
        kw.value.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "resolve_gemm_spec"
        for kw in node.keywords
        if kw.arg == "method" and isinstance(kw.value, ast.Constant)
    }
    assert len(pinned) == 1, (
        f"offload.py pins {sorted(pinned)} through resolve_gemm_spec; the table "
        "assumes exactly one re-resolution method for its whole chain"
    )
    chain_method = pinned.pop()
    for role in ((1, 1), (1, 4), (4, 1)):
        assert profiles.PINNED_PROJECTION_METHODS[role][0] == chain_method, (
            f"offload re-resolves to {chain_method!r} and the table says "
            f"{profiles.PINNED_PROJECTION_METHODS[role][0]!r} for role {role}"
        )
    # ...and the qkv role is NOT the chain's, which is the confusion that filed
    # the inert gap: offload's drain pin does not reach the 3h shape.
    assert profiles.PINNED_PROJECTION_METHODS[(1, 3)][0] != chain_method, (
        "the 3h role is pinned to offload's chain method. offload and runlist "
        "resolve (h,h), (h,4h) and (4h,h) only -- if that has really changed, "
        "the gap record deleted on 2026-08-12 needs revisiting"
    )


def test_a_blocked_variant_mode_pair_surfaces_as_a_skip_not_a_run():
    """The decoder half, per (variant, mode) since `[2026-08-19]`.

    Every pairing run_mode still refuses by name must surface through
    ``skip_reason`` -- so a profile SKIPS it and never starts a child process
    that would fail -- and the skip's text must carry run_mode's own reason
    rather than a restatement that would rot. And every reachable family must
    have at least one profile mode that is NOT blocked, or admitting it walks
    nothing."""
    import run_mode

    for family in profiles.REACHABLE_FAMILIES:
        variant = cases.FAMILY_SPECS[family].workload_variant
        blockers = run_mode.UNBUILDABLE_VARIANTS.get(variant, {})
        for mode in profiles.PROFILE_MODES:
            reason = profiles.skip_reason(mode, 512, family)
            if mode in blockers:
                assert reason is not None and blockers[mode] in reason, (
                    f"({variant}, {mode}) is refused by run_mode but "
                    f"skip_reason returns {reason!r}; a profile would start a "
                    "child process that can only fail"
                )
        unblocked = [m for m in profiles.PROFILE_MODES if m not in blockers]
        assert unblocked, (
            f"{family} is listed reachable but every profile mode is refused"
        )
    # Ordering: the variant clause must win over the fused packing bound --
    # probed at 4096, where BOTH would apply. A fused decoder rung reported
    # as "outside 256..1024" would read as a width limit on a wall that has
    # nothing to do with width (re-execution unmasks at EVERY length).
    reason = profiles.skip_reason("fused", 4096, "gpt2_small_768")
    assert reason is not None and "UNMASKED" in reason
    assert "plane-major packing" not in reason


def test_unreachable_families_cover_every_declared_family():
    named = set(profiles.UNREACHABLE_FAMILIES) | set(profiles.REACHABLE_FAMILIES)
    assert named == set(cases.FAMILY_IDS), (
        "a family was added to the case matrix without a reachability verdict; "
        "an undeclared family silently disappears from every run report"
    )
    assert profiles.REACHABLE_FAMILY not in profiles.UNREACHABLE_FAMILIES
    # Disjoint -- one verdict per family, never both.
    assert not set(profiles.UNREACHABLE_FAMILIES) & set(profiles.REACHABLE_FAMILIES)
    # And DERIVED, not tabulated: any family with at least one unblocked
    # profile mode must be REACHABLE. Leaving a buildable family in the
    # unreachable table misreports reachability in every run report -- the
    # `[2026-08-19]` contract is that reachability is per (family, mode), so
    # family-level unreachability is only for a family NO mode can build.
    import run_mode

    for fid in cases.FAMILY_IDS:
        variant = cases.FAMILY_SPECS[fid].workload_variant
        blockers = run_mode.UNBUILDABLE_VARIANTS.get(variant, {})
        if any(m not in blockers for m in profiles.PROFILE_MODES):
            assert fid in profiles.REACHABLE_FAMILIES, (
                f"{fid} has a buildable profile mode but is not reachable"
            )


def test_fused_bound_is_applied_in_both_directions():
    low, high = profiles.fused_seq_range()
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


def test_the_fused_bound_moves_with_the_width_rather_than_being_tabulated():
    """`[2026-08-12]` The cap is on ``rows*cols`` and cols IS the width, so a
    tabulated bound is right at one width and silently wrong at the others --
    and being wrong in the SKIPPING direction is the dangerous one: a skipped
    rung is not a failure, so the walk would report complete having never
    attempted a length the mode supports."""
    assert profiles.fused_seq_range("baseline_768") == (256, 1024), (
        "the derived bound must reproduce fused.py's own stated 256..1024 at "
        "the width it was stated for, or the derivation is not the same rule"
    )
    assert profiles.fused_seq_range("tinybert_512") == (256, 2048)
    assert profiles.fused_seq_range("baseline_1024") == (256, 1024)
    # ...and the rule discriminates at the moved bound, not only at 768's.
    assert profiles.skip_reason("fused", 2048, "tinybert_512") is None
    assert profiles.skip_reason("fused", 2048, "baseline_768")
    assert profiles.skip_reason("fused", 4096, "tinybert_512")


def test_retargeting_a_profile_retargets_its_gate_with_no_number_edited():
    """The profile table's whole discipline, on the axis resume made reachable."""
    ladder = profiles.profile("ladder")
    at768 = ladder.expected_rows()["fused.csv"]
    at512 = ladder.retarget("tinybert_512").expected_rows()["fused.csv"]
    assert (at768["measured"], at768["skipped"]) == (2, 2)
    assert (at512["measured"], at512["skipped"]) == (3, 1)
    assert at512["rows"] == at768["rows"] == len(ladder.seqs)


def test_retargeting_to_a_decoder_family_skips_fused_with_the_wall_reason():
    """`[2026-08-19]` Three modes run the decoder; fused's stitch is
    single-dispatch-correct but re-execution unmasks its attention (devq
    382-384), so a decoder retarget derives fused fully SKIPPED -- with the
    measured wall as the reason, not the retired wiring-gap text -- and
    everything else measured, with no number edited anywhere."""
    smoke = profiles.profile("smoke").retarget("gpt2_small_768")
    counts = smoke.expected_rows()
    assert counts["fused.csv"] == {"rows": 1, "measured": 0, "skipped": 1}
    for rel in ("coarse.csv", "offload.csv", "runlist.csv"):
        assert counts[rel] == {"rows": 1, "measured": 1, "skipped": 0}, rel
    ladder = profiles.profile("ladder").retarget("gpt2_small_768")
    lcounts = ladder.expected_rows()
    assert lcounts["fused.csv"] == {"rows": 4, "measured": 0, "skipped": 4}
    for rel in ("coarse.csv", "offload.csv", "runlist.csv"):
        assert lcounts[rel] == {"rows": 4, "measured": 4, "skipped": 0}, rel
    reason = smoke.rungs()[-1].skip_reason
    assert reason is not None and "UNMASKED" in reason


def test_skip_reason_names_the_source_of_the_bound():
    reason = profiles.skip_reason("fused", 4096)
    assert "plane-major packing" in reason and "256..1024" in reason
    assert str(profiles.FUSED_PLANE_STRIDE_CAP) in reason


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
    # Every declared family except the one walked, with a reason each -- the
    # unbuildable ones by name, the merely-not-targeted ones as such. A family
    # this profile could have walked and did not is still not walked, and
    # collapsing the two would make the report claim more coverage than it has.
    assert set(summary["families_not_walked"]) == set(cases.FAMILY_IDS) - {
        profiles.REACHABLE_FAMILY
    }
    for family, why in summary["families_not_walked"].items():
        if family in profiles.UNREACHABLE_FAMILIES:
            assert why == profiles.UNREACHABLE_FAMILIES[family]
        else:
            assert why == "not walked by this profile"
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
