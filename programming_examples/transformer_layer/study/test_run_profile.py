# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only checks on the profile runner.

    python3 study/test_run_profile.py

No device: ``run_profile.run`` takes an injectable ``walker``, so every step
after the walk -- the skip accounting, the gate, the manifest's row counts, the
run report -- is exercised against synthetic rows. The walker is the only thing
that needs hardware, and it is ``run_ladder.walk``, which has its own tests.

The load-bearing ones are the two directions of the count gate:
``test_a_short_walk_is_not_complete`` (the M3 defect: a CSV holding one rung of
four used to report complete) and
``test_an_inapplicable_rung_recorded_as_failed_is_caught`` (the M6 defect: an
inapplicable rung recorded as ``failed`` is indistinguishable from a
regression -- unless something counts).
"""

import json
import os
import sys
import tempfile
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import profiles  # noqa: E402
import results_io  # noqa: E402
import run_profile  # noqa: E402
import schema  # noqa: E402


def _row(mode, seq, status, message=""):
    row = schema.empty_row("results")
    row["execution_mode"] = schema.EXECUTION_MODE_CSV.get(mode, mode)
    row["study_id"] = "test"
    row["study_case_id"] = f"{seq}x768_encoder_bert"
    row["study_case_label"] = f"{mode} seq {seq}"
    row["seq_len"] = seq
    row["run_status"] = status
    row["failure_message"] = message
    return row


def _fake_walker(status_for):
    """A walker writing one CSV per mode, honouring the profile's skip rule."""

    def walk(modes, seqs, out_dir, study_id, warmup, samples, rps, skip_reason=None):
        every = []
        for mode in modes:
            rows = []
            for seq in seqs:
                reason = skip_reason(mode, seq) if skip_reason else None
                if reason:
                    rows.append(_row(mode, seq, "skipped", f"skipped: {reason}"))
                else:
                    rows.append(_row(mode, seq, status_for(mode, seq)))
            results_io.write_rows(os.path.join(out_dir, f"{mode}.csv"), rows)
            every.extend(rows)
        return every

    return walk


def _run(profile_name, walker, directory):
    return run_profile.run(
        profiles.profile(profile_name),
        directory,
        study_id="test",
        warmup=0,
        samples=1,
        runs_per_sample=1,
        power_backend="none",
        walker=walker,
        repo=directory,
    )


def test_a_clean_ladder_walk_is_complete():
    with tempfile.TemporaryDirectory() as d:
        report = _run("ladder", _fake_walker(lambda m, s: "passed"), d)
        assert report["complete"] is True, report["incomplete_reasons"]
        assert report["rungs_by_status"] == {"passed": 14, "skipped": 2}
        built = json.loads((Path(d) / run_profile.MANIFEST_NAME).read_text())
        assert built["complete"] is True
        assert built["row_counts_checked"] is True


def test_a_short_walk_is_not_complete():
    """M3: a CSV that should hold four rungs and holds one used to be complete."""

    def walk(modes, seqs, out_dir, study_id, w, s, r, skip_reason=None):
        rows = [_row("coarse", 512, "passed")]
        for mode in modes:
            results_io.write_rows(os.path.join(out_dir, f"{mode}.csv"), rows)
        return rows * len(modes)

    with tempfile.TemporaryDirectory() as d:
        report = _run("ladder", walk, d)
        assert report["complete"] is False
        reasons = " ".join(report["incomplete_reasons"])
        assert "expected 4 row(s), found 1" in reasons
        # and the smoke gate alone would have passed it: every CSV has a passed row
        assert "none with run_status=passed" not in reasons


def test_an_inapplicable_rung_recorded_as_failed_is_caught():
    """M6: without an emitted `skipped`, this is indistinguishable from a break."""

    def walk(modes, seqs, out_dir, study_id, w, s, r, skip_reason=None):
        every = []
        for mode in modes:
            rows = [
                _row(
                    mode,
                    seq,
                    (
                        "passed"
                        if not (skip_reason and skip_reason(mode, seq))
                        else "failed"
                    ),
                    (
                        ""
                        if not (skip_reason and skip_reason(mode, seq))
                        else "ValueError: plane stride over the shim cap"
                    ),
                )
                for seq in seqs
            ]
            results_io.write_rows(os.path.join(out_dir, f"{mode}.csv"), rows)
            every.extend(rows)
        return every

    with tempfile.TemporaryDirectory() as d:
        report = _run("ladder", walk, d)
        assert report["complete"] is False
        reasons = " ".join(report["incomplete_reasons"])
        assert "expected 2 skipped row(s), found 0" in reasons
        assert "run_status=skipped" in reasons


def test_a_regressed_rung_fails_the_measured_clause():
    walk = _fake_walker(
        lambda m, s: "failed" if (m, s) == ("coarse", 2048) else "passed"
    )
    with tempfile.TemporaryDirectory() as d:
        report = _run("ladder", walk, d)
        assert report["complete"] is False
        assert "expected 4 passed row(s), found 3" in " ".join(
            report["incomplete_reasons"]
        )


def test_the_run_report_records_the_plan_and_what_was_not_walked():
    with tempfile.TemporaryDirectory() as d:
        _run("smoke", _fake_walker(lambda m, s: "passed"), d)
        report = json.loads((Path(d) / run_profile.RUN_REPORT_NAME).read_text())
        assert report["profile"]["name"] == "smoke"
        assert set(report["profile"]["families_not_walked"]) == set(
            profiles.UNREACHABLE_FAMILIES
        )
        assert report["profile"]["rung_count"] == 4
        assert len(report["rungs"]) == 4
        assert "power_over_whole_walk" in report
        assert report["wall_clock_sec"] >= 0


def test_power_is_recorded_at_run_level_and_never_in_a_row():
    with tempfile.TemporaryDirectory() as d:
        _run("smoke", _fake_walker(lambda m, s: "passed"), d)
        block = json.loads((Path(d) / run_profile.RUN_REPORT_NAME).read_text())[
            "power_over_whole_walk"
        ]
        assert "power_backend" in block
        rows = results_io.read_rows(Path(d) / "coarse.csv")
        assert (
            rows[0]["avg_power_w"] is None
        ), "a whole-walk SoC figure must not be written into a per-rung row"


def test_gate_only_re_verifies_a_tree_without_walking_it():
    with tempfile.TemporaryDirectory() as d:
        _run("smoke", _fake_walker(lambda m, s: "passed"), d)
        (Path(d) / run_profile.MANIFEST_NAME).unlink()
        assert (
            run_profile.main(["--profile", "smoke", "--out-dir", d, "--gate-only"]) == 0
        )
        assert (Path(d) / run_profile.MANIFEST_NAME).is_file()


def test_dry_run_touches_no_device_and_no_tree():
    with tempfile.TemporaryDirectory() as d:
        assert run_profile.main(["--profile", "full", "--out-dir", d, "--dry-run"]) == 0
        assert not list(Path(d).iterdir()), "a dry run must write nothing"


def test_unknown_profile_exits_two_rather_than_walking():
    with tempfile.TemporaryDirectory() as d:
        assert run_profile.main(["--profile", "nightly", "--out-dir", d]) == 2


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"run_profile tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
