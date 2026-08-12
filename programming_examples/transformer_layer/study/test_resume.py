# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only checks on resume: the plan, the ledger, and the audit.

    python3 study/test_resume.py

No device. ``run_profile.run`` takes an injectable walker, so a whole two-session
resume -- walk, interrupt, resume, audit -- runs against synthetic rows.

EVERY CHECK HERE IS DRIVEN BY A DELIBERATELY BROKEN INPUT, and that is the point
rather than thoroughness for its own sake. G0's two closed defects were both
checks that COULD NOT FAIL: a `run_status="skipped"` in the schema since v1 with
nothing emitting it, and a manifest that validated files while a CSV holding 1
of 9 rungs reported ``complete: True``. A resume adds four more chances to build
one, because its natural implementation is bookkeeping -- and bookkeeping agrees
with itself whatever the walk did. So each audit clause below is paired with the
input that makes it fire:

    clause                       broken input
    ---------------------------  --------------------------------------------
    reuse fidelity               a walker that IGNORES `reuse` and re-measures
    ledger over-claim            a hand-added rung entry for a rung never run
    row moved under the ledger   a CSV row edited between two sessions
    unattributed row             a walker that writes rows and calls no hook
    corrupt ledger               `walk_sessions.json` holding `not json`
    pmode splice                 two sessions stamped turbo and default
    unreadable CSV               a schema-v1 header in the results root
"""

import csv
import json
import os
import sys
import tempfile
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import profiles  # noqa: E402
import resume  # noqa: E402
import results_io  # noqa: E402
import run_profile  # noqa: E402
import schema  # noqa: E402
from test_run_profile import _fake_walker, _row  # noqa: E402


def _run(directory, walker, *, profile="ladder", resume_=False, mode="turbo"):
    return run_profile.run(
        profiles.profile(profile),
        directory,
        study_id="test",
        warmup=0,
        samples=1,
        runs_per_sample=1,
        power_backend="none",
        walker=walker,
        repo=directory,
        resume=resume_,
        npu_power_mode=mode,
    )


def _ledger(directory):
    return json.loads((Path(directory) / resume.LEDGER_NAME).read_text())


def _all_pass(nonce=""):
    return _fake_walker(lambda m, s: "passed", nonce=nonce)


# ---------------------------------------------------------------------------
# The digest -- everything else rests on it round-tripping through a CSV.
# ---------------------------------------------------------------------------


def test_a_row_hashes_the_same_before_and_after_a_csv_round_trip():
    """If this fails, every reuse looks like a re-measurement and resume is off.

    A row built in memory carries floats and ints; the same row read back from a
    CSV carries strings. The digest hashes the SERIALISATION so the two agree.
    """
    row = _row("coarse", 512, "passed")
    row["avg_latency_ms"] = 12.5
    row["seq_len"] = 512
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "coarse.csv"
        results_io.write_rows(path, [row])
        back = results_io.read_rows(path)[0]
    assert resume.row_digest(row) == resume.row_digest(back)


def test_a_changed_measurement_changes_the_digest():
    """The control for the test above: it must not hash everything the same."""
    a = _row("coarse", 512, "passed")
    b = dict(a)
    b["avg_latency_ms"] = 12.5
    assert resume.row_digest(a) != resume.row_digest(b)


def test_row_key_agrees_across_the_file_boundary():
    """A string seq_len and an int seq_len must be the same rung, or the reuse
    lookup misses every time and the resume silently redoes everything."""
    row = _row("coarse", 512, "passed")
    assert resume.row_key(row) == ("hybrid", 512)
    assert resume.row_key({**row, "seq_len": "512"}) == ("hybrid", 512)


# ---------------------------------------------------------------------------
# The plan.
# ---------------------------------------------------------------------------


def test_a_passed_rung_is_carried_forward_and_a_failed_one_is_not():
    """THE REUSE POLICY. `failed` is re-run on purpose -- a retained failure is
    a claim about code that may no longer be there."""
    with tempfile.TemporaryDirectory() as d:
        _run(d, _fake_walker(lambda m, s: "failed" if s == 4096 else "passed"))
        prior = resume.scan(d, profiles.profile("ladder").expected_files())
        plan = resume.plan(profiles.profile("ladder"), prior)
    assert ("coarse", 512) in plan.reuse
    assert ("coarse", 4096) in plan.remeasure
    assert "re-run rather than carried forward" in plan.reasons[("coarse", 4096)]


def test_a_skipped_rung_is_re_derived_every_session_never_reused():
    """A skip is the PROFILE's current claim, not a recorded observation. Freeze
    it and a packing bound that moves leaves the old rule in force."""
    with tempfile.TemporaryDirectory() as d:
        _run(d, _all_pass())
        prior = resume.scan(d, profiles.profile("ladder").expected_files())
        plan = resume.plan(profiles.profile("ladder"), prior)
    assert ("fused", 2048) in plan.skipped
    assert ("fused", 2048) not in plan.reuse
    assert ("fused", 2048) not in plan.remeasure


def test_planning_without_resume_walks_everything():
    with tempfile.TemporaryDirectory() as d:
        _run(d, _all_pass())
        prior = resume.scan(d, profiles.profile("ladder").expected_files())
        plan = resume.plan(profiles.profile("ladder"), prior, enabled=False)
    assert not plan.reuse
    assert len(plan.remeasure) == 14


# ---------------------------------------------------------------------------
# The ledger.
# ---------------------------------------------------------------------------


def test_a_resumed_walk_reuses_the_passed_rungs_and_walks_the_rest():
    with tempfile.TemporaryDirectory() as d:
        _run(d, _fake_walker(lambda m, s: "failed" if s == 4096 else "passed"))
        report = _run(d, _all_pass(nonce="second"), resume_=True)
    assert report["complete"] is True, report["incomplete_reasons"]
    # ladder is 16 rungs: 2 structurally skipped (`fused` at 2048 and 4096),
    # leaving 14 attempted. The first walk failed every 4096 rung that ran --
    # coarse, offload and runlist, since fused's 4096 was skipped -- so 11
    # passed rows are carried forward and 3 failures are re-walked.
    assert report["rungs_by_source"] == {"measured": 3, "reused": 11, "skipped": 2}
    assert report["session_count"] == 2
    assert report["resume_defects"] == []


def test_the_ledger_is_flushed_per_rung_so_a_killed_session_keeps_its_work():
    """The whole feature is for a machine that dies mid-walk. Batching the
    ledger at the end would lose exactly the session it exists to recover."""
    seen = []

    def walk(modes, seqs, out_dir, study_id, w, s, r, skip_reason=None, reuse=None,
             on_rung=None, family=None):
        rows = [_row(modes[0], seqs[0], "passed")]
        on_rung(modes[0], seqs[0], rows[0], "measured")
        # ...and observe the file as it stands mid-walk, before any close.
        seen.append(len(_ledger(out_dir)["sessions"][0]["rungs"]))
        results_io.write_rows(os.path.join(out_dir, f"{modes[0]}.csv"), rows)
        for mode in modes[1:]:
            results_io.write_rows(os.path.join(out_dir, f"{mode}.csv"), rows)
        return rows

    with tempfile.TemporaryDirectory() as d:
        _run(d, walk, profile="smoke")
    assert seen == [1], "the ledger was not on disk before the session closed"


def test_a_session_that_never_ended_is_relabelled_interrupted_by_the_next():
    """A killed process writes nothing. The NEXT session is the only party in a
    position to observe that its predecessor never finished."""

    def dies(modes, seqs, out_dir, study_id, w, s, r, skip_reason=None, reuse=None,
             on_rung=None, family=None):
        rows = [_row(modes[0], seqs[0], "passed")]
        on_rung(modes[0], seqs[0], rows[0], "measured")
        results_io.write_rows(os.path.join(out_dir, f"{modes[0]}.csv"), rows)
        raise KeyboardInterrupt("the lid closed")

    with tempfile.TemporaryDirectory() as d:
        try:
            _run(d, dies, profile="smoke")
        except KeyboardInterrupt:
            pass
        # The `finally` closed it, because THIS process did observe the end.
        assert _ledger(d)["sessions"][0]["status"] == "complete"
        # Now the same shape with a session left genuinely open, as a SIGKILL
        # would: reopen the ledger and mark it running again.
        payload = _ledger(d)
        payload["sessions"][0]["status"] = "running"
        payload["sessions"][0]["ended_utc"] = None
        (Path(d) / resume.LEDGER_NAME).write_text(json.dumps(payload))
        _run(d, _all_pass(), profile="smoke", resume_=True)
        states = [s["status"] for s in _ledger(d)["sessions"]]
    assert states == ["interrupted", "complete"], states


# ---------------------------------------------------------------------------
# THE AUDIT. One deliberately broken input each.
# ---------------------------------------------------------------------------


def test_a_walker_that_ignores_reuse_is_caught_as_a_resume_defect():
    """BROKEN INPUT: a walker that re-measures a rung the plan carried forward.

    THE load-bearing check. A resume that silently redoes work makes every
    downstream diff vacuous, and no amount of bookkeeping can see it -- the
    plan, the ledger and the report would all agree it was reused.
    """
    with tempfile.TemporaryDirectory() as d:
        _run(d, _all_pass())
        report = _run(
            d,
            _fake_walker(lambda m, s: "passed", honour_reuse=False, nonce="redone"),
            resume_=True,
        )
    assert report["complete"] is False
    assert report["resume_defects"], "the walker re-ran 14 carried rungs unseen"
    joined = " ".join(report["resume_defects"])
    assert "planned as REUSED and its row changed" in joined
    # ...and it reaches the manifest, not just the run report.
    assert any("planned as REUSED" in r for r in report["incomplete_reasons"])


def test_a_ledger_claiming_a_rung_the_csvs_do_not_hold_is_caught():
    """BROKEN INPUT: a rung entry hand-added for a rung nothing ever ran."""
    with tempfile.TemporaryDirectory() as d:
        _run(d, _all_pass(), profile="smoke")
        payload = _ledger(d)
        payload["sessions"][0]["rungs"].append(
            {
                "execution_mode": "hybrid",
                "seq_len": 999,
                "source": "measured",
                "run_status": "passed",
                "row_digest": "deadbeefdeadbeef",
            }
        )
        (Path(d) / resume.LEDGER_NAME).write_text(json.dumps(payload))
        prof = profiles.profile("smoke")
        block = resume.walk_block(
            resume.Ledger.load(d), resume.scan(d, prof.expected_files()), profile=prof
        )
    assert any(
        "claims rung hybrid seq 999 and no such row" in p
        for p in block["attribution_problems"]
    ), block["attribution_problems"]


def test_a_row_edited_behind_the_ledger_is_caught_by_its_digest():
    """BROKEN INPUT: a CSV row's latency rewritten between two sessions.

    This is the check that makes attribution EVIDENCE rather than a claim. The
    row count is still right, the file is still readable, and the ledger still
    names the session -- only the hash disagrees.
    """
    with tempfile.TemporaryDirectory() as d:
        _run(d, _all_pass(), profile="smoke")
        path = Path(d) / "coarse.csv"
        rows = results_io.read_rows(path)
        rows[0]["avg_latency_ms"] = 0.001  # a flattering number, from nowhere
        results_io.write_rows(path, rows)
        prof = profiles.profile("smoke")
        block = resume.walk_block(
            resume.Ledger.load(d), resume.scan(d, prof.expected_files()), profile=prof
        )
    assert any(
        "was re-measured or edited behind the ledger" in p
        for p in block["attribution_problems"]
    ), block["attribution_problems"]


def test_a_row_no_session_claims_is_unattributed_and_incomplete():
    """BROKEN INPUT: a walker that writes rows and calls the hook for none.

    A row nobody measured has unknown provenance. The row counts cannot see it
    -- the count is right -- which is exactly why the ledger exists.
    """

    def silent(modes, seqs, out_dir, study_id, w, s, r, skip_reason=None, reuse=None,
               on_rung=None, family=None):
        every = []
        for mode in modes:
            rows = [_row(mode, seqs[0], "passed")]
            results_io.write_rows(os.path.join(out_dir, f"{mode}.csv"), rows)
            every.extend(rows)
        return every

    with tempfile.TemporaryDirectory() as d:
        report = _run(d, silent, profile="smoke")
    assert report["complete"] is False
    assert any("belong to no session" in r for r in report["incomplete_reasons"])


def test_a_corrupt_ledger_is_a_problem_and_not_a_fresh_start():
    """BROKEN INPUT: `walk_sessions.json` holding text that is not JSON.

    Starting over would erase the provenance of every row already on disk and
    let the resumed walk report clean.
    """
    with tempfile.TemporaryDirectory() as d:
        _run(d, _all_pass(), profile="smoke")
        (Path(d) / resume.LEDGER_NAME).write_text("not json {")
        ledger = resume.Ledger.load(d)
        assert ledger.unreadable is not None
        assert ledger.sessions == []
        prof = profiles.profile("smoke")
        block = resume.walk_block(ledger, resume.scan(d, prof.expected_files()), profile=prof)
    assert any("cannot be read" in p for p in block["attribution_problems"])


def test_a_ledger_whose_session_record_is_malformed_is_also_refused():
    """BROKEN INPUT: a session record with a key the schema does not declare.

    A hand-assembled ledger with a typo'd key must fail loudly rather than be
    read back as a session that recorded nothing.
    """
    with tempfile.TemporaryDirectory() as d:
        _run(d, _all_pass(), profile="smoke")
        payload = _ledger(d)
        payload["sessions"][0]["npu_powr_mode"] = "turbo"
        (Path(d) / resume.LEDGER_NAME).write_text(json.dumps(payload))
        assert resume.Ledger.load(d).unreadable is not None


def test_two_sessions_at_different_power_modes_are_refused():
    """BROKEN INPUT: a second session stamped `default`.

    `compare_roots` refuses a pmode mismatch BETWEEN two roots. Inside one CSV
    it is strictly worse: there is not even a root boundary to warn a reader
    that two populations were joined. ~15-20x on this host, README trap 0.
    """
    with tempfile.TemporaryDirectory() as d:
        _run(d, _fake_walker(lambda m, s: "failed" if s == 4096 else "passed"))
        report = _run(d, _all_pass(nonce="x"), resume_=True, mode="default")
    assert "npu_power_mode" in report["condition_splices"]
    assert report["complete"] is False
    assert any(
        "more than one NPU power mode" in r for r in report["incomplete_reasons"]
    )


def test_a_toolchain_or_sha_splice_is_flagged_and_not_refused():
    """The other half of the split: refusing here would make resume unusable,
    because resuming after a commit is the ordinary case."""
    sessions = [
        {
            **{name: None for name in schema.SESSION_FIELDNAMES},
            "session_id": "s001",
            "status": "complete",
            "npu_power_mode": "turbo",
            "git_sha": "aaaa",
            "toolchain_fingerprint": "x|y|z|build-xrt",
            "rungs": [
                {
                    "execution_mode": "hybrid",
                    "seq_len": 512,
                    "source": "measured",
                    "run_status": "passed",
                    "row_digest": "0",
                }
            ],
        },
        {
            **{name: None for name in schema.SESSION_FIELDNAMES},
            "session_id": "s002",
            "status": "complete",
            "npu_power_mode": "turbo",
            "git_sha": "bbbb",
            "toolchain_fingerprint": "x|y|z|install-xrt",
            "rungs": [
                {
                    "execution_mode": "hybrid",
                    "seq_len": 1024,
                    "source": "measured",
                    "run_status": "passed",
                    "row_digest": "0",
                }
            ],
        },
    ]
    splices = resume.condition_splices(sessions)
    assert splices == ["toolchain", "git_sha"], splices
    block = resume.walk_block(resume.Ledger(Path("/nonexistent"), sessions), resume.PriorWalk())
    # Flagged in the prose, and NOT among the problems.
    assert "FLAGGED" in block["walk_detail"]
    assert not any("toolchain" in p for p in block["attribution_problems"])


def test_a_pure_reuse_session_cannot_create_a_phantom_splice():
    """A session that measured nothing describes no measurement condition, so
    its pmode must not be compared against one that did."""
    sessions = [
        {
            **{name: None for name in schema.SESSION_FIELDNAMES},
            "session_id": "s001",
            "status": "complete",
            "npu_power_mode": "turbo",
            "rungs": [
                {
                    "execution_mode": "hybrid",
                    "seq_len": 512,
                    "source": "measured",
                    "run_status": "passed",
                    "row_digest": "0",
                }
            ],
        },
        {
            **{name: None for name in schema.SESSION_FIELDNAMES},
            "session_id": "s002",
            "status": "complete",
            "npu_power_mode": "default",
            "rungs": [
                {
                    "execution_mode": "hybrid",
                    "seq_len": 512,
                    "source": "reused",
                    "run_status": "passed",
                    "row_digest": "0",
                }
            ],
        },
    ]
    assert resume.condition_splices(sessions) == []


def test_an_unreadable_csv_is_reported_rather_than_walked_over():
    """BROKEN INPUT: a schema-v1 header sitting in the results root.

    Absent means "re-measure it"; unreadable means "something is here this
    reader does not understand", and silently re-measuring over a recorded walk
    to make a gate green is the failure that distinction exists to prevent.
    """
    with tempfile.TemporaryDirectory() as d:
        with open(Path(d) / "coarse.csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["study_id", "execution_mode", "schema_version"])
            writer.writerow(["old", "hybrid", "1"])
        prior = resume.scan(d, ["coarse.csv"])
        assert "coarse.csv" in prior.unreadable
        assert not prior.rows
        block = resume.walk_block(resume.Ledger(Path(d) / resume.LEDGER_NAME, []), prior)
    assert any(
        "cannot be read as the current schema" in p
        for p in block["attribution_problems"]
    )


# ---------------------------------------------------------------------------
# The guarantee the brief names: a resume cannot report a complete walk that is
# not one.
# ---------------------------------------------------------------------------


def test_a_resume_cannot_turn_a_short_walk_complete():
    """The row-count clauses read the CSVs and know nothing about sessions, so
    resuming changes the cost of a walk and never its verdict."""
    with tempfile.TemporaryDirectory() as d:
        _run(d, _fake_walker(lambda m, s: "failed" if s == 4096 else "passed"))
        report = _run(
            d,
            _fake_walker(lambda m, s: "failed" if s == 4096 else "passed", nonce="2"),
            resume_=True,
        )
    assert report["rungs_by_source"]["reused"] == 11
    assert report["complete"] is False
    assert any(
        "expected 4 passed row(s), found 3" in r for r in report["incomplete_reasons"]
    )


def test_a_resumed_manifest_records_that_it_was_resumed():
    with tempfile.TemporaryDirectory() as d:
        _run(d, _all_pass(), profile="smoke")
        built = json.loads(
            (Path(d) / run_profile.MANIFEST_NAME).read_text()
        )
        assert built[schema.WALK_KEY]["walk_source"] == "single_session"
        assert built["walk_attribution_checked"] is True
        _run(d, _all_pass(nonce="2"), profile="smoke", resume_=True)
        built = json.loads((Path(d) / run_profile.MANIFEST_NAME).read_text())
    assert built[schema.WALK_KEY]["walk_source"] == "resumed"
    assert built[schema.WALK_KEY]["session_count"] == 2
    assert built[schema.WALK_KEY]["rungs_reused"] == 4


def test_a_manifest_built_without_a_ledger_checks_nothing_and_stays_readable():
    """Back-compatibility, and the M3 distinction one level up: "attributed" and
    "nobody looked" must not read the same."""
    import manifest as manifest_mod

    with tempfile.TemporaryDirectory() as d:
        results_io.write_rows(Path(d) / "coarse.csv", [_row("coarse", 512, "passed")])
        built = manifest_mod.build_manifest(d, ["coarse.csv"], repo=d)
    assert built["walk_attribution_checked"] is False
    assert built["complete"] is True
    # Three distinguishable states, and collapsing any two is the M3 shape:
    #   unknown  a block is present and nobody supplied a ledger.
    #   absent   the manifest predates the block entirely.
    #   resumed/single_session  somebody looked, and this is what they found.
    assert schema.walk_from_manifest(built)["walk_source"] == schema.UNKNOWN_CONDITION
    assert schema.walk_from_manifest({})["walk_source"] == "absent"
    assert "NOT assumed" in built[schema.WALK_KEY]["walk_detail"]


def test_a_populated_root_is_refused_without_resume():
    """Both directions of the same guard: half-overwriting a recorded root, and
    resuming one by accident."""
    with tempfile.TemporaryDirectory() as d:
        _run(d, _all_pass(), profile="smoke")
        assert run_profile.main(["--profile", "smoke", "--out-dir", d]) == 2


def test_the_walk_block_never_writes_the_reader_only_absent():
    with tempfile.TemporaryDirectory() as d:
        _run(d, _all_pass(), profile="smoke")
        built = json.loads((Path(d) / run_profile.MANIFEST_NAME).read_text())
    schema.validate_walk(built[schema.WALK_KEY])  # raises on `absent`


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"resume tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
