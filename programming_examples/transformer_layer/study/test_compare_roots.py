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
    assert "no manifest.json in the root" in text
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
    tmp, a, b = _roots([_row()], [_row()], manifests=(_manifest("turbo"), _manifest("turbo")))
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
        assert "environ" not in Path(compare_roots.__file__).read_text(
            encoding="utf-8"
        )
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


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"compare-roots tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
