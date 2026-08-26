# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only checks on the run-to-run comparator.

    python3 study/test_compare_roots.py

Every test builds two small results roots on disk and asserts the VERDICT, not
just the arithmetic: the module's job is to decide, and a tiering rule that
computes the right percentage and then fails the wrong way is still wrong.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import compare_roots  # noqa: E402
import results_io  # noqa: E402
import schema  # noqa: E402


def _row(mode="hybrid", seq=1024, latency=100.0, **overrides):
    row = schema.empty_row("results")
    row.update(
        {
            "study_id": "s",
            "study_case_id": "baseline_768",
            "study_case_label": "BERT-Base",
            "workload_variant": "encoder_bert",
            "backend": "xrt",
            "execution_mode": mode,
            "attention_path": "device",
            "seq_len": seq,
            "hidden_size": 768,
            "batch_size": 1,
            "dtype": "bf16",
            "warmup_runs": 1,
            "runs_per_sample": 1,
            "latency_sample_count": 3,
            "avg_latency_ms": latency,
            "min_latency_ms": latency * 0.9,
            "max_latency_ms": latency * 1.1,
            "host_submissions_per_layer": 4,
            "run_status": "passed",
            "failure_message": "",
        }
    )
    row.update(overrides)
    return row


def _roots(baseline_rows, candidate_rows, *, name="coarse.csv", manifests=None):
    """Two results roots on disk. Returns a TemporaryDirectory to keep alive."""
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    (root / "a").mkdir()
    (root / "b").mkdir()
    results_io.write_rows(root / "a" / name, baseline_rows)
    results_io.write_rows(root / "b" / name, candidate_rows)
    if manifests:
        for side, payload in zip(("a", "b"), manifests):
            (root / side / "manifest.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
    return tmp, root / "a", root / "b"


def _named_roots(filename, payload):
    """Two roots whose manifest carries ``filename``. Returns (tmp, a, b)."""
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    for side in ("a", "b"):
        (root / side).mkdir()
        results_io.write_rows(root / side / "coarse.csv", [_row()])
        (root / side / filename).write_text(json.dumps(payload), encoding="utf-8")
    return tmp, root / "a", root / "b"


def _turbo_manifest():
    block = schema.empty_conditions()
    block["npu_power_mode"] = "turbo"
    block["npu_power_mode_source"] = "observed"
    return {schema.CONDITIONS_KEY: block}


def test_the_manifest_the_RUNNER_writes_is_the_one_this_module_reads():
    """`[2026-08-14]` The name mismatch that made items 15 and 16 inert.

    ``run_profile.py`` writes ``results_manifest.json``; this module looked only
    for ``manifest.json``. A comparison of two real Phase G walks therefore
    reported `npu_power_mode: unknown (absent)` and said the mode was "NOT
    recoverable from its files", while a manifest recording `turbo` (observed)
    sat in the same directory under the other name.

    No test caught it because every fixture in this file wrote whatever the
    module read -- which is why this one takes the filename as a parameter and
    asserts BOTH names work.
    """
    for filename in ("results_manifest.json", "manifest.json"):
        tmp, a, b = _named_roots(filename, _turbo_manifest())
        try:
            payload, why = compare_roots.load_manifest(a)
            assert payload is not None, f"{filename} unreadable: {why}"
            assert payload[schema.CONDITIONS_KEY]["npu_power_mode"] == "turbo"
        finally:
            tmp.cleanup()


def test_the_runner_and_this_module_agree_on_the_name_BY_CONSTRUCTION():
    """The pin that stops the two drifting apart again.

    Asserting both names load is not enough: a future runner could rename its
    output a third time and this module would go quiet again. So the runner's
    own constant must appear in the list this module searches.
    """
    import run_profile

    assert run_profile.MANIFEST_NAME in compare_roots.MANIFEST_NAMES, (
        f"run_profile writes {run_profile.MANIFEST_NAME!r} and compare_roots "
        f"searches {compare_roots.MANIFEST_NAMES!r} -- the reader would go "
        "silently unconditioned on every root the runner produces"
    )


def test_a_root_with_a_DIFFERENT_name_is_still_flagged():
    """The control: the fallback must not be a wildcard."""
    tmp, a, _b = _named_roots("some_other_manifest.json", _turbo_manifest())
    try:
        payload, why = compare_roots.load_manifest(a)
        assert payload is None
        assert "results_manifest.json or manifest.json" in why
    finally:
        tmp.cleanup()


def test_relative_percent_handles_a_zero_baseline_without_dividing():
    assert compare_roots.relative_percent(0.0, 0.0) is None
    assert compare_roots.relative_percent(0.0, 1.0) == float("inf")
    assert compare_roots.relative_percent(100.0, 110.0) == 10.0
    assert compare_roots.relative_percent(100.0, 90.0) == 10.0


def test_signed_percent_keeps_the_direction_relative_percent_discards():
    assert compare_roots.signed_percent(100.0, 90.0) == -10.0
    assert compare_roots.signed_percent(100.0, 110.0) == 10.0
    assert compare_roots.signed_percent(0.0, 5.0) is None


def test_percentile_90_is_defined_for_a_short_list():
    assert compare_roots.percentile_90([]) == 0.0
    assert compare_roots.percentile_90([7.0]) == 7.0
    assert compare_roots.percentile_90([1.0, 2.0, 3.0]) == 3.0


def test_number_never_raises_on_junk():
    for junk in (None, "", "None", "nan", "NaN", "not-a-number", []):
        assert compare_roots._number(junk) is None
    assert compare_roots._number("2.5") == 2.5


def test_identical_roots_pass():
    tmp, a, b = _roots([_row()], [_row()])
    with tmp:
        report = compare_roots.compare_roots(a, b, ["coarse.csv"])
    assert report.failures == 0, report.render()
    assert "VERDICT: OK" in report.render()


def test_naming_no_csvs_is_a_failure_not_a_pass():
    """The same rule as smoke_gate: a check of nothing must not be cheerful."""
    tmp, a, b = _roots([_row()], [_row()])
    with tmp:
        report = compare_roots.compare_roots(a, b, [])
    assert report.failures == 1
    assert "proved nothing" in report.render()


def test_a_small_latency_drift_passes_for_a_quiet_mode():
    tmp, a, b = _roots([_row(latency=100.0)], [_row(latency=103.0)])
    with tmp:
        report = compare_roots.compare_roots(a, b, ["coarse.csv"])
    assert report.failures == 0, report.render()


def test_a_large_latency_drift_fails_for_a_quiet_mode():
    tmp, a, b = _roots([_row(latency=100.0)], [_row(latency=130.0)])
    with tmp:
        report = compare_roots.compare_roots(a, b, ["coarse.csv"])
    assert report.failures >= 1
    assert "avg_latency_ms" in report.render()


def test_the_same_drift_passes_for_offload_and_fails_for_runlist():
    """The per-mode band, verified as a DIFFERENCE rather than in one direction."""
    tmp, a, b = _roots(
        [_row(mode="offload", latency=100.0)],
        [_row(mode="offload", latency=125.0)],
        name="offload.csv",
    )
    with tmp:
        loose = compare_roots.compare_roots(a, b, ["offload.csv"])
    tmp, a, b = _roots(
        [_row(mode="runlist", latency=100.0)],
        [_row(mode="runlist", latency=125.0)],
        name="runlist.csv",
    )
    with tmp:
        tight = compare_roots.compare_roots(a, b, ["runlist.csv"])
    assert loose.failures == 0, loose.render()
    assert tight.failures >= 1, tight.render()


def test_min_and_max_latency_are_reported_and_never_gate():
    """A single unlucky sample must not turn a healthy run red."""
    tmp, a, b = _roots(
        [_row(latency=100.0)],
        [_row(latency=100.0, min_latency_ms=10.0, max_latency_ms=900.0)],
    )
    with tmp:
        report = compare_roots.compare_roots(a, b, ["coarse.csv"])
    text = report.render()
    assert report.failures == 0, text
    assert "min_latency_ms" in text
    assert "[info]" in text


def test_a_moved_structural_count_is_a_failure_not_drift():
    """A mode whose submission count changed is a different mode."""
    tmp, a, b = _roots([_row()], [_row(host_submissions_per_layer=6)])
    with tmp:
        report = compare_roots.compare_roots(a, b, ["coarse.csv"])
    assert report.failures >= 1
    assert "host_submissions_per_layer" in report.render()


def test_a_row_present_on_one_side_only_fails():
    tmp, a, b = _roots([_row(seq=512), _row(seq=1024)], [_row(seq=1024)])
    with tmp:
        report = compare_roots.compare_roots(a, b, ["coarse.csv"])
    assert report.failures >= 1
    assert "only in baseline" in report.render()


def test_a_csv_absent_from_the_baseline_fails_rather_than_skipping():
    """iron skips this case; a file a run stopped producing is the thing to catch."""
    tmp, a, b = _roots([_row()], [_row()])
    with tmp:
        report = compare_roots.compare_roots(a, b, ["nope.csv"])
    assert report.failures >= 1
    assert "absent from the baseline" in report.render()


def test_a_status_flip_to_failed_is_an_identifier_mismatch():
    tmp, a, b = _roots([_row()], [_row(run_status="failed")])
    with tmp:
        report = compare_roots.compare_roots(a, b, ["coarse.csv"])
    assert report.failures >= 1
    assert "run_status" in report.render()


def test_an_unreadable_csv_fails_rather_than_being_skipped():
    tmp, a, b = _roots([_row()], [_row()])
    with tmp:
        (b / "coarse.csv").write_text("not,a,schema\n1,2,3\n", encoding="utf-8")
        report = compare_roots.compare_roots(a, b, ["coarse.csv"])
    assert report.failures >= 1
    assert "unreadable as the current schema" in report.render()


def test_provenance_differences_are_notes_not_failures():
    manifests = (
        {"complete": True, "git": {"sha": "aaa", "dirty": False}, "toolchain": {}},
        {"complete": True, "git": {"sha": "bbb", "dirty": False}, "toolchain": {}},
    )
    tmp, a, b = _roots([_row()], [_row()], manifests=manifests)
    with tmp:
        report = compare_roots.compare_roots(a, b, ["coarse.csv"])
    text = report.render()
    assert report.failures == 0, text
    assert "NOTE  git.sha: aaa -> bbb" in text


def test_an_incomplete_candidate_manifest_fails():
    manifests = (
        {"complete": True, "git": {}, "toolchain": {}},
        {"complete": False, "git": {}, "toolchain": {}},
    )
    tmp, a, b = _roots([_row()], [_row()], manifests=manifests)
    with tmp:
        report = compare_roots.compare_roots(a, b, ["coarse.csv"])
    assert report.failures >= 1
    assert "complete=false" in report.render()


def test_there_is_no_intended_rename_exception_table():
    """Convention 7 settled the naming once; nothing needs excepting."""
    assert not hasattr(compare_roots, "RENAMED_VALUES")
    assert "pattern_label" not in compare_roots.IDENTIFIER_FIELDS


def test_key_fields_gained_the_model_scope_and_layer_rows_key_as_before():
    """Doc 56 section 3.6 `[2026-08-23]`: two model rows that differ only in
    context_end_tokens are two measurements; a layer row's key is unchanged
    in content (the six new members are empty strings)."""
    assert compare_roots.KEY_FIELDS[:3] == ("study_case_id", "execution_mode", "seq_len")
    assert compare_roots.KEY_FIELDS[3:] == (
        "measurement_scope", "model_id", "phase", "ubatch_tokens",
        "context_end_tokens", "precision_plan_id")
    layer = compare_roots.row_key(_row())
    assert layer == ("baseline_768", "hybrid", "1024", "None", "None", "None", "None", "None", "None"), layer
    # through a CSV the empty cells read back as None and key identically
    tmp, a, _b = _roots([_row()], [_row()])
    with tmp:
        assert compare_roots.row_key(results_io.read_rows(a / "coarse.csv")[0]) == layer
    a = compare_roots.row_key(_model_row(512))
    b = compare_roots.row_key(_model_row(1024))
    assert a != b and a[:3] == b[:3]


def _model_row(ctx, tps=13.0, **over):
    fields = dict(
        study_case_id="qwen3_0_6b/decode", host_submissions_per_layer=None,
        measurement_scope="model", model_id="qwen3_0_6b", phase="decode",
        logical_token_count=32, ubatch_tokens=1, context_start_tokens=ctx - 32,
        context_end_tokens=ctx, measured_token_count=32, tokens_per_second=tps,
        precision_plan_id="bf16", plan_hash="b" * 64, host_ops=58,
        model_dispatch_vector_json='{"scope": "decode_token", "host_submissions": 57, '
        '"runlist_entries": 57, "air_launches": 150, "herd_launches": 206, '
        '"sync_boundaries": 200, "bytes_transferred": 100}')
    fields.update(over)
    return _row("hybrid", 1, 1000.0 / tps, **fields)


def test_tokens_per_second_is_gated_with_the_mode_tolerance():
    """A 20% tok/s drift on a `hybrid` model row fails (band 5/15); the same
    drift at 4% warns at most. The plan hash is an identifier: a moved hash
    is a different planned sequence, not drift."""
    tmp, a, b = _roots([_model_row(512, 13.0), _model_row(1024, 12.0)],
                       [_model_row(512, 10.4), _model_row(1024, 9.6)], name="model.csv")
    with tmp:
        report = compare_roots.compare_roots(a, b, ["model.csv"])
    assert report.failures >= 1
    assert any("tokens_per_second" in line and "exceeds the fail" in line for line in report.lines), report.lines
    tmp, a, b = _roots([_model_row(512, 13.0)], [_model_row(512, 13.3)], name="model.csv")
    with tmp:
        report = compare_roots.compare_roots(a, b, ["model.csv"])
    assert report.failures == 0, report.render()
    tmp, a, b = _roots([_model_row(512, 13.0)], [_model_row(512, 13.0, plan_hash="c" * 64)], name="model.csv")
    with tmp:
        report = compare_roots.compare_roots(a, b, ["model.csv"])
    assert report.failures == 1 and any("plan_hash" in line for line in report.lines)


def test_two_v2_roots_still_compare_after_the_v3_bump():
    """The roots this tool is pointed at were written at schema v2; the bump
    must not take them out of it. A v2 root against a v3 root fails on the
    version identifier rather than being compared column for column."""
    import csv

    v2_names = [n for n in schema.RESULTS_FIELDNAMES if n not in schema.MODEL_SCOPE_FIELDNAMES]

    def write_v2(path, latency):
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=v2_names)
            w.writeheader()
            row = {n: "" for n in v2_names}
            row.update(schema_version=2, study_case_id="baseline_768", execution_mode="hybrid",
                       seq_len=1024, run_status="passed", avg_latency_ms=latency)
            w.writerow(row)

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for side, latency in (("a", 100.0), ("b", 101.0)):
            (root / side).mkdir()
            write_v2(root / side / "coarse.csv", latency)
        report = compare_roots.compare_roots(root / "a", root / "b", ["coarse.csv"])
        assert report.failures == 0, report.render()
        v3 = schema.empty_row()
        v3.update(study_case_id="baseline_768", execution_mode="hybrid", seq_len=1024,
                  run_status="passed", avg_latency_ms=100.0)
        results_io.write_rows(root / "b" / "coarse.csv", [v3])
        report = compare_roots.compare_roots(root / "a", root / "b", ["coarse.csv"])
        assert report.failures == 1, report.render()
        assert any("schema_version: baseline='2' candidate='3'" in line for line in report.lines)


def test_every_gating_field_is_also_reported():
    reported = set(compare_roots.LATENCY_FIELDS) | set(compare_roots.POWER_FIELDS)
    assert set(compare_roots.GATING_FIELDS) <= reported


def test_every_compared_field_is_a_real_schema_column():
    names = {f.name for f in schema.fields_for("results")}
    for field in (
        compare_roots.IDENTIFIER_FIELDS
        + compare_roots.LATENCY_FIELDS
        + compare_roots.POWER_FIELDS
        + compare_roots.KEY_FIELDS
    ):
        assert field in names, field


def test_every_csv_execution_mode_has_a_tolerance_band():
    """A mode with no band silently gets the default; that must be a choice."""
    for mode in schema.EXECUTION_MODES:
        assert mode in compare_roots.LATENCY_TOLERANCE, mode


# ---------------------------------------------------------------------------
# The measurement-condition guard `[2026-08-12]`, queue item 15.
#
# Every clause below is asserted as a DIFFERENCE -- the same two roots, the same
# drift, with only the recorded power mode changed -- because a guard that
# refuses everything and a guard that refuses nothing both pass a one-sided
# test. The drift used is 1900%, the size a Turbo-vs-Default pair actually
# produces on this host, so the "before" behaviour under test is the real one:
# a red latency verdict printed for a power-mode change.
# ---------------------------------------------------------------------------

_SPLICE_DRIFT = 2000.0  # 100 ms at Turbo against ~2 s at Default, README trap 0


def _manifest(mode=None, source="observed", *, block=True):
    """A manifest payload; ``block=False`` for one predating the conditions block."""
    payload = {"complete": True, "git": {}, "toolchain": {}}
    if block:
        conditions = schema.empty_conditions()
        if mode is not None:
            conditions["npu_power_mode"] = mode
            conditions["npu_power_mode_source"] = source
        payload[schema.CONDITIONS_KEY] = conditions
    return payload


def _spliced_roots(baseline_mode, candidate_mode, **kw):
    """Two roots whose latencies differ by a pmode-sized amount."""
    return _roots(
        [_row(latency=100.0)],
        [_row(latency=100.0 * (1 + _SPLICE_DRIFT / 100.0))],
        manifests=(_manifest(baseline_mode, **kw), _manifest(candidate_mode, **kw)),
    )


def test_a_pmode_mismatch_is_refused_rather_than_read_as_a_regression():
    """THE defect: a power-mode change alone printed VERDICT: PROBLEM."""
    tmp, a, b = _spliced_roots("turbo", "default")
    with tmp:
        report = compare_roots.compare_roots(a, b, ["coarse.csv"])
    text = report.render()
    assert report.refusals, text
    assert "COMPARISON REFUSED" in text
    assert "different NPU power modes" in text
    assert "baseline=turbo, candidate=default" in text
    assert "VERDICT: PROBLEM" in text


def test_the_same_drift_at_the_same_pmode_is_a_latency_failure():
    """The other side of the difference: the drift itself is unchanged."""
    tmp, a, b = _spliced_roots("turbo", "turbo")
    with tmp:
        report = compare_roots.compare_roots(a, b, ["coarse.csv"])
    text = report.render()
    assert not report.refusals, text
    assert "avg_latency_ms" in text
    assert "exceeds the fail threshold" in text
    assert "[GATE]" in text
    assert "VERDICT: PROBLEM" in text


def test_a_refused_comparison_withdraws_the_latency_VERDICT_not_the_numbers():
    """Reported as SPLICED: no FAIL on the latency, and no silence either."""
    tmp, a, b = _spliced_roots("turbo", "default")
    with tmp:
        refused = compare_roots.compare_roots(a, b, ["coarse.csv"])
    tmp, a, b = _spliced_roots("turbo", "turbo")
    with tmp:
        gated = compare_roots.compare_roots(a, b, ["coarse.csv"])
    assert "[SPLICED]" in refused.render()
    assert "[GATE]" not in refused.render()
    assert "exceeds the fail threshold" not in refused.render()
    assert "exceeds the fail threshold" in gated.render()
    # The drift is still measured and printed on both sides.
    for report in (refused, gated):
        assert "avg_latency_ms" in report.render()
    # And the refusal is the ONLY failure: the numbers did not vote.
    assert refused.failures == 1, refused.render()


def test_matching_pmodes_say_so_and_gate_normally():
    tmp, a, b = _roots(
        [_row(latency=100.0)],
        [_row(latency=101.0)],
        manifests=(_manifest("turbo"), _manifest("turbo")),
    )
    with tmp:
        report = compare_roots.compare_roots(a, b, ["coarse.csv"])
    text = report.render()
    assert report.failures == 0, text
    assert "both roots were measured at `turbo`" in text
    assert "VERDICT: OK" in text


def test_an_unrecorded_pmode_flags_and_KEEPS_gating():
    """A recorded root cannot be stamped after the fact, so refusing is a dead end."""
    tmp, a, b = _roots(
        [_row(latency=100.0)],
        [_row(latency=101.0)],
        manifests=(_manifest(block=False), _manifest("turbo")),
    )
    with tmp:
        report = compare_roots.compare_roots(a, b, ["coarse.csv"])
    text = report.render()
    assert report.failures == 0, text
    assert report.warnings >= 1
    assert "does not record the NPU power mode" in text
    assert "[GATE]" in text  # still gating: an unknown must not disarm the tool
    assert "VERDICT: OK" in text


def test_an_unknown_pmode_still_fails_on_a_real_regression():
    """The reason unknown flags rather than refusing: the tool stays useful."""
    tmp, a, b = _roots(
        [_row(latency=100.0)],
        [_row(latency=130.0)],
        manifests=(_manifest(block=False), _manifest(block=False)),
    )
    with tmp:
        report = compare_roots.compare_roots(a, b, ["coarse.csv"])
    assert report.failures >= 1, report.render()
    assert "exceeds the fail threshold" in report.render()


def test_a_root_with_no_manifest_at_all_flags_rather_than_skipping_silently():
    """The recorded ladder walks are exactly this shape -- no manifest either side."""
    tmp, a, b = _roots([_row(latency=100.0)], [_row(latency=101.0)])
    with tmp:
        report = compare_roots.compare_roots(a, b, ["coarse.csv"])
    text = report.render()
    assert report.failures == 0, text
    assert report.warnings >= 2  # one per side
    assert "no results_manifest.json or manifest.json in the root" in text
    assert "=== measurement condition ===" in text


def test_the_condition_section_runs_even_when_the_provenance_diff_skips():
    """A guard behind the manifest early-out would never run on real evidence."""
    tmp, a, b = _roots([_row()], [_row()])
    with tmp:
        text = compare_roots.compare_roots(a, b, ["coarse.csv"]).render()
    assert "SKIP (missing on one side)" in text  # the provenance diff skipped
    assert "npu_power_mode: baseline=unknown" in text  # the guard did not


def test_a_refused_comparison_still_reports_the_pmode_independent_half():
    """Bytes, counts and identifiers are pmode-independent (trap 0) and still hold."""
    tmp, a, b = _roots(
        [_row()],
        [_row(host_submissions_per_layer=6)],
        manifests=(_manifest("turbo"), _manifest("default")),
    )
    with tmp:
        report = compare_roots.compare_roots(a, b, ["coarse.csv"])
    text = report.render()
    assert "host_submissions_per_layer" in text  # the structural change is still caught
    assert "COMPARISON REFUSED" in text
    assert report.failures >= 2  # the refusal AND the structural mismatch


def test_an_unreadable_manifest_is_a_failure_and_an_unknown_condition():
    tmp, a, b = _roots(
        [_row()], [_row()], manifests=(_manifest("turbo"), _manifest("turbo"))
    )
    with tmp:
        (b / "manifest.json").write_text("{not json", encoding="utf-8")
        report = compare_roots.compare_roots(a, b, ["coarse.csv"])
    text = report.render()
    assert "could not be read" in text
    assert "does not record the NPU power mode" in text  # and it flags, not matches
    assert report.failures >= 1


def test_there_is_no_way_to_defeat_the_pmode_guard():
    """pmode_guard's rule: nothing added for a guard may become its bypass.

    Asserted through the CLI rather than by grepping, so the docstring may go
    on saying which flag deliberately does not exist.
    """
    tmp, a, b = _spliced_roots("turbo", "default")
    with tmp:
        for bypass in ("--allow-pmode-splice", "--ignore-pmode", "--force"):
            try:
                # argparse writes its usage to stderr on a rejection; swallow it
                # so a passing suite stays quiet.
                with open(os.devnull, "w") as null, contextlib.redirect_stderr(null):
                    compare_roots.main(
                        ["--baseline", str(a), "--candidate", str(b), bypass]
                    )
            except SystemExit as e:
                assert e.code == 2, f"{bypass} was accepted"
                continue
            raise AssertionError(f"{bypass} did not raise; the guard has a bypass")
        # And no environment variable is consulted anywhere in the module.
        assert "environ" not in Path(compare_roots.__file__).read_text(encoding="utf-8")
        # The refusal stands on the honest invocation.
        assert (
            compare_roots.main(
                ["--baseline", str(a), "--candidate", str(b), "--csv", "coarse.csv"]
            )
            == 1
        )


def test_the_guard_reads_the_RECORDED_condition_and_never_the_live_one():
    """The whole reason item 13's fix does not transfer.

    `require_turbo()` reads the mode NOW and says nothing about a run recorded
    last week. A live query creeping in here would make the guard pass or fail
    on the state of the machine running the comparison, which is not a fact
    about either root -- so the guard reads the manifest through the schema and
    subprocesses nothing.
    """
    source = Path(compare_roots.__file__).read_text(encoding="utf-8")
    assert "schema.conditions_from_manifest" in source
    # Asserted on the module's bindings rather than on its text, so the
    # docstring may keep explaining require_turbo without tripping this.
    for live in ("subprocess", "shutil", "npu_power_mode", "require_turbo"):
        assert not hasattr(compare_roots, live), (
            f"compare_roots binds {live}: that is a LIVE query, and this module "
            "compares two runs recorded earlier"
        )


# ---------------------------------------------------------------------------
# The toolchain guard `[2026-08-12]` -- queue item 16.
#
# The defect these close: `compare_manifests` diffed a `toolchain` block nothing
# wrote, so its inner loop iterated an empty key and the toolchain half of every
# comparison compared NOTHING. The tests therefore come in two kinds -- that the
# block is now actually read, and that reading it FLAGS without withdrawing the
# gate, which is where it deliberately differs from the pmode guard.
# ---------------------------------------------------------------------------

_TOOLCHAIN_A = {
    "xrt_version": "2.21.0+4eb1f4392a01",
    "mlir_aie_version": "1.4.0",
    "peano_version": "21.0.0.2026080401+512badad",
    "air_resolution": "build-xrt",
    "toolchain_source": "probed_at_manifest_build",
    "toolchain_detail": "probed 4/4 field(s) at manifest build",
}


def _tc(**overrides):
    return dict(_TOOLCHAIN_A, **overrides)


#: A writable timing block naming the CURRENT contract. `[2026-08-26]`, item 19
#: review finding 5 -- see `compare_roots.compare_timing`.
_TIMING_A = {
    "kernel_ms_contract": schema.KERNEL_MS_CONTRACT_NOW,
    "xrt_run_cache": "on",
    "timing_source": "probed_at_manifest_build",
    "timing_detail": "fixture",
}


def _tm(**overrides):
    return dict(_TIMING_A, **overrides)


def _timing_manifest(timing, mode="turbo"):
    """A manifest with a matched pmode and toolchain, so only the timing
    contract varies."""
    payload = _manifest(mode)
    payload["toolchain"] = _tc()
    if timing is None:
        payload.pop(schema.TIMING_KEY, None)  # a manifest predating the block
    else:
        payload[schema.TIMING_KEY] = timing
    return payload


def _timing_roots(left, right, *, drift=0.0):
    return _roots(
        [_row(latency=100.0)],
        [_row(latency=100.0 * (1 + drift / 100.0))],
        manifests=(_timing_manifest(left), _timing_manifest(right)),
    )


def _tool_manifest(toolchain, mode="turbo"):
    """A manifest with a matched pmode AND a matched timing contract, so only
    the toolchain axis varies. `[2026-08-26]` the timing block is part of the
    fixture for the same reason the pmode is: `compare_timing` warns when a root
    does not say what its `kernel_ms` includes, and a toolchain test that
    inherited that warning would be asserting two axes at once."""
    payload = _manifest(mode)
    payload[schema.TIMING_KEY] = _tm()
    if toolchain is None:
        payload.pop("toolchain", None)  # a manifest predating the block
    else:
        payload["toolchain"] = toolchain
    return payload


def _tool_roots(left, right, *, drift=0.0):
    return _roots(
        [_row(latency=100.0)],
        [_row(latency=100.0 * (1 + drift / 100.0))],
        manifests=(_tool_manifest(left), _tool_manifest(right)),
    )


def test_the_toolchain_block_is_actually_read_now():
    """The regression test for the original defect.

    Before item 16 this section did not exist and the diff loop it depends on
    iterated `{}`. A mismatch had to produce no output at all.
    """
    tmp, a, b = _tool_roots(_tc(), _tc(xrt_version="2.20.0+aaaaaaaaaaaa"))
    with tmp:
        text = compare_roots.compare_roots(a, b, ["coarse.csv"]).render()
    assert "=== build condition (toolchain) ===" in text, text
    assert "2.20.0+aaaaaaaaaaaa" in text, text


def test_a_toolchain_mismatch_flags_and_does_NOT_refuse():
    """The decision item 16 had to take, asserted as behaviour.

    A pmode mismatch refuses; this one warns and keeps going. With no drift to
    gate on, the verdict must stay OK -- a flag that failed the run would be a
    refusal wearing a warning's name.
    """
    tmp, a, b = _tool_roots(_tc(), _tc(xrt_version="2.20.0+aaaaaaaaaaaa"))
    with tmp:
        report = compare_roots.compare_roots(a, b, ["coarse.csv"])
    text = report.render()
    assert not report.refusals, text
    assert report.warnings >= 1, text
    assert "different xrt_version" in text, text
    assert "VERDICT: OK" in text, text


def test_a_toolchain_mismatch_names_the_field_that_moved():
    """ "the toolchain differs" is not actionable; naming the layer is."""
    tmp, a, b = _tool_roots(_tc(), _tc(peano_version="20.0.0.2026010101+beef"))
    with tmp:
        text = compare_roots.compare_roots(a, b, ["coarse.csv"]).render()
    assert "different peano_version" in text, text
    assert "21.0.0.2026080401+512badad" in text and "20.0.0" in text, text


def test_THE_SAME_DRIFT_fails_identically_at_matched_and_mismatched_toolchains():
    """THE anti-swallow check, and the reason `may_gate` is not touched.

    Item 15's key assertion was that an identical drift at the same condition
    still fails. The equivalent here is stronger, because this guard is not
    allowed to withdraw anything at all: the FAIL text must be IDENTICAL on both
    sides, so the only thing a toolchain mismatch can add is a warning.
    """
    matched, a1, b1 = _tool_roots(_tc(), _tc(), drift=40.0)
    with matched:
        left = compare_roots.compare_roots(a1, b1, ["coarse.csv"])
        left_fails = [l for l in left.render().splitlines() if "FAIL " in l]
    mismatched, a2, b2 = _tool_roots(
        _tc(), _tc(xrt_version="2.20.0+aaaaaaaaaaaa"), drift=40.0
    )
    with mismatched:
        right = compare_roots.compare_roots(a2, b2, ["coarse.csv"])
        right_fails = [l for l in right.render().splitlines() if "FAIL " in l]

    assert left_fails == right_fails, (left_fails, right_fails)
    assert left_fails, "the 40% drift must fail at all -- otherwise this proves nothing"
    assert left.failures == right.failures == len(left_fails)
    assert right.warnings > left.warnings, (left.warnings, right.warnings)


def test_a_toolchain_mismatch_never_splices_the_gate():
    """`[SPLICED]` belongs to the pmode refusal alone; this guard must not reach it."""
    tmp, a, b = _tool_roots(_tc(), _tc(air_resolution="install-xrt"), drift=40.0)
    with tmp:
        text = compare_roots.compare_roots(a, b, ["coarse.csv"]).render()
    assert "[GATE]" in text, text
    assert "[SPLICED]" not in text, text


def test_matching_toolchains_say_so_and_add_no_warning():
    tmp, a, b = _tool_roots(_tc(), _tc())
    with tmp:
        report = compare_roots.compare_roots(a, b, ["coarse.csv"])
    text = report.render()
    assert "both roots were built against the same toolchain" in text, text
    assert report.warnings == 0, text
    assert "VERDICT: OK" in text


def test_a_provenance_only_difference_is_not_a_toolchain_difference():
    """How the values were obtained is not what they are."""
    tmp, a, b = _tool_roots(
        _tc(toolchain_source="observed", toolchain_detail="one way"),
        _tc(toolchain_detail="another way"),
    )
    with tmp:
        report = compare_roots.compare_roots(a, b, ["coarse.csv"])
    text = report.render()
    assert "both roots were built against the same toolchain" in text, text
    assert report.warnings == 0, text


def test_a_manifest_predating_the_block_flags_on_both_sides():
    """Every root recorded before today, and it must not crash or match."""
    tmp, a, b = _tool_roots(None, None)
    with tmp:
        report = compare_roots.compare_roots(a, b, ["coarse.csv"])
    text = report.render()
    assert text.count("does not record xrt_version") == 2, text
    assert "must not be stamped after the" in text, text
    assert "both roots were built against the same toolchain" not in text, text
    assert not report.refusals, text
    assert "VERDICT: OK" in text, text


def test_an_absent_toolchain_still_fails_on_a_real_regression():
    """The corpus keeps gating -- refusing there would make the tool useless."""
    tmp, a, b = _tool_roots(None, None, drift=40.0)
    with tmp:
        report = compare_roots.compare_roots(a, b, ["coarse.csv"])
    text = report.render()
    assert "exceeds the fail threshold" in text, text
    assert "VERDICT: PROBLEM" in text, text


def test_one_side_recorded_and_one_side_not_is_flagged_not_compared():
    """A half-known axis is unknown, not agreement and not a difference."""
    tmp, a, b = _tool_roots(_tc(), None)
    with tmp:
        report = compare_roots.compare_roots(a, b, ["coarse.csv"])
    text = report.render()
    assert text.count("does not record xrt_version") == 1, text
    assert "different xrt_version" not in text, text
    assert report.warnings >= 1, text


def test_a_root_with_no_manifest_at_all_is_flagged_on_the_toolchain_axis_too():
    """The recorded ladder walks carry no manifest; the guard must still run."""
    tmp, a, b = _roots([_row(latency=100.0)], [_row(latency=100.0)])
    with tmp:
        report = compare_roots.compare_roots(a, b, ["coarse.csv"])
    text = report.render()
    assert "=== build condition (toolchain) ===" in text, text
    assert "no results_manifest.json or manifest.json in the root" in text, text
    assert report.warnings >= 2, text


def test_the_pmode_refusal_still_wins_when_both_conditions_moved():
    """The two guards compose; the stronger verdict is not softened by the flag."""
    tmp, a, b = _roots(
        [_row(latency=100.0)],
        [_row(latency=100.0 * (1 + _SPLICE_DRIFT / 100.0))],
        manifests=(
            _tool_manifest(_tc(), mode="turbo"),
            _tool_manifest(_tc(xrt_version="2.20.0+aaaaaaaaaaaa"), mode="default"),
        ),
    )
    with tmp:
        report = compare_roots.compare_roots(a, b, ["coarse.csv"])
    text = report.render()
    assert report.refusals, text
    assert "COMPARISON REFUSED" in text and "[SPLICED]" in text, text
    assert "different xrt_version" in text, text
    assert "VERDICT: PROBLEM" in text


def test_every_toolchain_identity_field_is_reported_and_compared():
    """A field declared but never printed would be a fifth silent half-check."""
    for name in schema.TOOLCHAIN_IDENTITY_FIELDNAMES:
        tmp, a, b = _tool_roots(_tc(), _tc(**{name: "moved-value"}))
        with tmp:
            text = compare_roots.compare_roots(a, b, ["coarse.csv"]).render()
        assert f"  {name}: baseline=" in text, (name, text)
        assert f"different {name}" in text, (name, text)



# ---------------------------------------------------------------------------
# `[2026-08-26]` The TIMING-CONTRACT guard -- queue item 19 review, finding 5.
# `device_ms` is the sum of `kernel_ms`, and until this block existed
# `load_and_run` timed the host-side xrt.run construction inside it (38-57 us
# per call, measured). A build that skipped that work moved `device_ms` by
# 30-50 us/call with the device unchanged, and compare_roots called the two
# roots the same metric. These pin the guard's three outcomes and the split
# between the field that REFUSES and the field that only warns.
# ---------------------------------------------------------------------------


def test_two_roots_under_different_kernel_ms_contracts_are_refused():
    """The core of finding 5: not two measurements of one quantity."""
    tmp, a, b = _timing_roots(
        _tm(kernel_ms_contract="bind_and_start_wait"),
        _tm(kernel_ms_contract="start_wait_only"),
    )
    with tmp:
        report = compare_roots.compare_roots(a, b, ["coarse.csv"])
    text = report.render()
    assert "COMPARISON REFUSED" in text, text
    assert "bind_and_start_wait" in text and "start_wait_only" in text, text
    assert report.refusals, text


def test_a_refused_contract_withdraws_gating_but_keeps_the_unit_free_half():
    """A refusal must not silently pass a real latency regression as OK, and it
    must not throw away the comparisons that do not depend on the unit."""
    tmp, a, b = _timing_roots(
        _tm(kernel_ms_contract="bind_and_start_wait"),
        _tm(kernel_ms_contract="start_wait_only"),
        drift=400.0,
    )
    with tmp:
        report = compare_roots.compare_roots(a, b, ["coarse.csv"])
    text = report.render()
    assert "[SPLICED]" in text, text
    assert "=== timing contract" in text, text


def test_matching_contracts_say_so_and_add_no_warning():
    tmp, a, b = _timing_roots(_tm(), _tm())
    with tmp:
        report = compare_roots.compare_roots(a, b, ["coarse.csv"])
    text = report.render()
    assert "means the same thing on both sides" in text, text
    assert report.warnings == 0, text
    assert not report.refusals, text


def test_a_manifest_predating_the_timing_block_flags_and_keeps_gating():
    """Every root recorded before 2026-08-26. It must not crash, must not read
    as agreement, and must not refuse -- refusing would make the whole recorded
    corpus uncomparable, which is `compare_conditions`' own argument."""
    tmp, a, b = _timing_roots(None, None, drift=400.0)
    with tmp:
        report = compare_roots.compare_roots(a, b, ["coarse.csv"])
    text = report.render()
    assert text.count("does not record which terms its `kernel_ms` includes") == 2, text
    assert "must not be stamped after the fact" in text, text
    assert "means the same thing on both sides" not in text, text
    assert not report.refusals, text
    # still gating: a real regression at an unknown contract must still fail
    assert "[SPLICED]" not in text, text
    assert report.failures > 0, text


def test_a_run_cache_difference_warns_but_does_not_refuse():
    """Under `start_wait_only` the cache cannot move `device_ms` -- that is what
    the split bought -- but it moves the token wall, so it is a warning that
    names which numbers it touches, not a refusal."""
    tmp, a, b = _timing_roots(_tm(xrt_run_cache="off"), _tm(xrt_run_cache="on"))
    with tmp:
        report = compare_roots.compare_roots(a, b, ["coarse.csv"])
    text = report.render()
    assert "different `xrt_run_cache` states" in text, text
    assert "does NOT move `device_ms`" in text, text
    assert not report.refusals, text
    assert report.warnings >= 1, text


def test_there_is_no_way_to_defeat_the_timing_guard():
    """The pmode guard's rule, transferred: nothing added for a guard may become
    its bypass. No flag, no environment variable, no argument."""
    import inspect

    src = inspect.getsource(compare_roots)
    for bypass in ("--allow-contract-splice", "--ignore-timing", "--any-contract"):
        assert bypass not in src, bypass
    sig = inspect.signature(compare_roots.compare_timing)
    assert list(sig.parameters) == ["report", "baseline", "candidate"], sig


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"compare-roots tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
