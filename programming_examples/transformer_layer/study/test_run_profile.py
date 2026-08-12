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


# ---------------------------------------------------------------------------
# The device preflight `[2026-08-12]` -- queue item 19.
#
# Hermetic: every case drives `device_preflight` against a STUB broker script, so
# the real /tmp/mlir-air-npu.lock is never opened and no device is touched. The
# real broker's own behaviour is gated by agents/scripts/devq-selftest.sh TEST 7;
# what these pin is that this script READS the answer correctly -- which is the
# half that failed in item 19, where a flag was parsed and never branched on.
# ---------------------------------------------------------------------------


def _stub_devq(tmp, exit_code, stderr=""):
    """A fake devq.sh whose `preflight` exits with a chosen code."""
    p = Path(tmp) / "devq.sh"
    p.write_text(
        "#!/bin/sh\n"
        f'[ "$1" = preflight ] || exit 9\n'
        + (f'echo "{stderr}" >&2\n' if stderr else "")
        + f"exit {exit_code}\n"
    )
    p.chmod(0o755)
    return p


def test_a_walk_records_the_toolchain_it_measured_against():
    """Queue item 16's payoff: the block must reach a REAL manifest.

    A declared block that only the CLI ever writes would leave every profile
    artifact `unknown`, which is a record that records nothing.
    """
    with tempfile.TemporaryDirectory() as d:
        _run("smoke", _fake_walker(lambda m, s: "passed"), d)
        built = json.loads((Path(d) / run_profile.MANIFEST_NAME).read_text())
        assert schema.TOOLCHAIN_KEY in built, sorted(built)
        block = built[schema.TOOLCHAIN_KEY]
        assert set(block) == set(schema.TOOLCHAIN_FIELDNAMES), sorted(block)
        assert block["toolchain_source"] != "absent"


def test_gate_only_does_NOT_stamp_todays_toolchain_on_an_old_tree():
    """The other half, and the one that would be a data corruption.

    Re-gating a recorded tree must not claim it was built against whatever this
    host has now -- "never stamp a condition you did not observe".
    """
    with tempfile.TemporaryDirectory() as d:
        _run("smoke", _fake_walker(lambda m, s: "passed"), d)
        (Path(d) / run_profile.MANIFEST_NAME).unlink()
        assert (
            run_profile.main(["--profile", "smoke", "--out-dir", d, "--gate-only"]) == 0
        )
        block = json.loads((Path(d) / run_profile.MANIFEST_NAME).read_text())[
            schema.TOOLCHAIN_KEY
        ]
        for name in schema.TOOLCHAIN_IDENTITY_FIELDNAMES:
            assert block[name] == schema.UNKNOWN_CONDITION, name
        assert "NOT assumed" in block["toolchain_detail"]


def test_preflight_passes_when_the_device_is_free():
    with tempfile.TemporaryDirectory() as d:
        assert run_profile.device_preflight(_stub_devq(d, 0)) is True


def test_preflight_REFUSES_when_another_job_holds_the_device():
    """The item 19 scenario: a dispatch beside devq job 252's regression."""
    with tempfile.TemporaryDirectory() as d:
        stub = _stub_devq(d, 3, "devq: preflight REFUSED -- the lock is held")
        assert run_profile.device_preflight(stub) is False


def test_an_indeterminate_answer_warns_rather_than_refusing():
    """Refusing on an unknown would let a broken broker block all measurement.

    compare_roots' refuse-known / flag-unknown split, applied to a device.
    """
    with tempfile.TemporaryDirectory() as d:
        assert run_profile.device_preflight(_stub_devq(d, 7)) is None


def test_a_missing_broker_is_indeterminate_and_never_raises():
    with tempfile.TemporaryDirectory() as d:
        assert run_profile.device_preflight(Path(d) / "absent.sh") is None


def test_the_three_states_are_distinguishable():
    """A bool would collapse "busy" into "could not tell"; they need opposite
    responses, and only `is False` may stop a run."""
    with tempfile.TemporaryDirectory() as d:
        free = run_profile.device_preflight(_stub_devq(d, 0))
        held = run_profile.device_preflight(_stub_devq(d, 3))
        unsure = run_profile.device_preflight(_stub_devq(d, 7))
    assert (free, held, unsure) == (True, False, None)
    assert len({str(free), str(held), str(unsure)}) == 3
    # Only the definite refusal stops the run.
    assert [x is False for x in (free, held, unsure)] == [False, True, False]


def _declared_options(path):
    """Every option string this module passes to argparse. AST, not grep.

    Grep would match the docstring sentence that says the escape hatch does not
    exist, which is the opposite of the thing being asserted.
    """
    import ast

    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    out = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    out.add(arg.value)
    return out


def test_there_is_no_flag_to_skip_the_device_preflight():
    """pmode_guard's rule: nothing added for a guard may become its defeat."""
    declared = _declared_options(Path(_HERE, "run_profile.py"))
    assert declared, "no options parsed -- the AST walk is broken, not the guard"
    for escape in (
        "--ignore-device-lock",
        "--no-preflight",
        "--force-dispatch",
        "--skip-preflight",
        "--allow-contention",
    ):
        assert escape not in declared, escape


def test_every_flag_this_module_parses_is_also_branched_on():
    """Item 19's defect shape, made unrepeatable in this module.

    `--compile-mode` was accepted, advertised in `--help` and changed nothing, so
    an operator who asked for the safe branch got the dispatching one. A flag
    whose dest is never read is that defect, whatever the flag is called.
    """
    import ast

    path = Path(_HERE, "run_profile.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    dests, read = {}, set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            explicit = [
                kw.value.value
                for kw in node.keywords
                if kw.arg == "dest" and isinstance(kw.value, ast.Constant)
            ]
            longs = [
                a.value
                for a in node.args
                if isinstance(a, ast.Constant) and str(a.value).startswith("--")
            ]
            if explicit:
                dests[explicit[0]] = node.lineno
            elif longs:
                dests[longs[0].lstrip("-").replace("-", "_")] = node.lineno
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            read.add(node.attr)
    dead = {d: line for d, line in dests.items() if d not in read}
    assert not dead, f"flags parsed and never branched on: {dead}"


def test_the_preflight_is_actually_called_before_dispatching():
    """The item 19 defect itself: a guard that is present and never branched on.

    An audit of `main` -- if the call is ever deleted or its result dropped, the
    preflight becomes exactly the dead `--compile-mode` this item is about.
    """
    source = Path(_HERE, "run_profile.py").read_text(encoding="utf-8")
    assert "device_preflight() is False" in source, (
        "run_profile.main no longer branches on device_preflight; a guard whose "
        "result is discarded is the item 19 defect reintroduced"
    )
    # And it must be consulted BEFORE the walk takes the output lock.
    assert source.index("device_preflight() is False") < source.index(
        "with run_lock.hold("
    )


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"run_profile tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
