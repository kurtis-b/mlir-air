# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only checks on the results manifest.

    python3 study/test_manifest.py

The load-bearing one is ``test_all_failed_tree_is_not_complete``: iron's manifest
calls a tree complete when the files exist, so a run whose every measurement
failed reports complete. Phase G's gate is "a full profile run completes with a
complete results_manifest.json", so that definition would make the gate unable
to fail -- the same shape as the smoke test doc 09 records.

`[2026-08-12]` The condition tests below never touch the device. The probe is
exercised by putting a fake ``xrt-smi`` on PATH, which runs the REAL parse path
end to end and leaves no test-only branch in the production one -- the same
technique, for the same reason, as ``port-loop/pmode_guard.py``'s selftest. The
suite's contract is "no hardware, no MLIR, no toolchain, well under a second",
and a manifest test that queried a real NPU would quietly break it.

Its sibling is ``test_one_row_where_nine_were_expected_is_not_complete``, which
pins the SECOND version of that hole: "at least one passed row" is still a file
rule wearing a row's clothes, so a truncated walk reported complete until the
row counts landed. The pair is deliberate -- the first shows the file rule is
not enough, the second shows the passed-row rule is not either.
"""

import json
import os
import stat
import sys
import tempfile
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import manifest  # noqa: E402
import results_io  # noqa: E402
import schema  # noqa: E402

_PLATFORM_REPORT = """\
  Platform
    Name                   : RyzenAI-npu4
    Power Mode             : %s
"""


def _stub_path(tmp, text):
    """A PATH directory holding a fake ``xrt-smi``. ``text=None`` -> empty dir."""
    d = Path(tempfile.mkdtemp(dir=tmp, prefix="stub-"))
    if text is None:
        return d
    p = d / "xrt-smi"
    # `echo` is a shell builtin, so the stub needs nothing else on PATH.
    body = "#!/bin/sh\n" + "".join(f"echo {line!r}\n" for line in text.splitlines())
    p.write_text(body.replace("'", '"'))
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return d


def _with_path(d, fn):
    old = os.environ.get("PATH", "")
    os.environ["PATH"] = str(d)
    try:
        return fn()
    finally:
        os.environ["PATH"] = old


def _row(status, msg=None):
    row = schema.empty_row("results")
    row["execution_mode"] = "hybrid"
    row["run_status"] = status
    row["failure_message"] = msg
    return row


def test_complete_tree_is_complete():
    with tempfile.TemporaryDirectory() as d:
        results_io.write_rows(Path(d) / "m.csv", [_row("passed")])
        m = manifest.build_manifest(d, ["m.csv"])
        assert m["complete"] is True
        assert m["incomplete_reasons"] == []
        assert m["missing_files"] == []


def test_all_failed_tree_is_not_complete():
    """iron would call this complete: the file exists. It measured nothing."""
    with tempfile.TemporaryDirectory() as d:
        results_io.write_rows(
            Path(d) / "m.csv", [_row("failed", "ERT_CMD_STATE_TIMEOUT")]
        )
        m = manifest.build_manifest(d, ["m.csv"])
        assert m["complete"] is False
        assert m["missing_files"] == []  # the file IS present
        assert any("none with run_status=passed" in r for r in m["incomplete_reasons"])
        assert any("ERT_CMD_STATE_TIMEOUT" in r for r in m["incomplete_reasons"])


def test_missing_file_is_reported_twice_over():
    with tempfile.TemporaryDirectory() as d:
        m = manifest.build_manifest(d, ["gone.csv"])
        assert m["complete"] is False
        assert m["missing_files"] == ["gone.csv"]
        assert m["incomplete_reasons"]


def test_empty_expectation_is_not_complete():
    """Describing an empty tree as a finished run is the failure to avoid."""
    with tempfile.TemporaryDirectory() as d:
        m = manifest.build_manifest(d, [])
        assert m["complete"] is False


def test_file_records_carry_size_and_mtime():
    with tempfile.TemporaryDirectory() as d:
        results_io.write_rows(Path(d) / "m.csv", [_row("passed")])
        rec = manifest.build_manifest(d, ["m.csv"])["expected_files"][0]
        assert rec["exists"] and rec["bytes"] > 0 and rec["mtime_utc"]


def test_provenance_is_present_and_best_effort():
    with tempfile.TemporaryDirectory() as d:
        m = manifest.build_manifest(d, [], repo=d)  # not a git repo
        assert "git" in m and "system" in m
        assert m["git"]["sha"] is None  # best effort, not a crash
        assert m["system"]["python"]


def test_write_manifest_round_trips_as_sorted_json():
    with tempfile.TemporaryDirectory() as d:
        results_io.write_rows(Path(d) / "m.csv", [_row("passed")])
        m = manifest.build_manifest(d, ["m.csv"])
        out = manifest.write_manifest(Path(d) / "manifest.json", m)
        back = json.loads(out.read_text())
        assert back["complete"] is True
        keys = list(back)
        assert keys == sorted(keys), "keys must be sorted so runs diff cleanly"


def test_cli_exit_codes():
    with tempfile.TemporaryDirectory() as d:
        results_io.write_rows(Path(d) / "m.csv", [_row("passed")])
        out = str(Path(d) / "m.json")
        # --no-probe keeps this test off the device; the probe path has its own
        # tests below, against a stub.
        args = ["--no-probe", "--expect"]
        assert manifest.main([d, *args, "m.csv", "-o", out]) == 0
        assert manifest.main([d, *args, "nope.csv", "-o", out]) == 1


# ---------------------------------------------------------------------------
# The measurement condition (doc 34 M4). See the module docstring on the stub.
# ---------------------------------------------------------------------------


def test_a_manifest_records_the_measurement_condition():
    with tempfile.TemporaryDirectory() as d:
        results_io.write_rows(Path(d) / "m.csv", [_row("passed")])
        block = manifest.observe_conditions("Turbo")
        m = manifest.build_manifest(d, ["m.csv"], conditions=block)
        assert m["conditions"]["npu_power_mode"] == "turbo"  # lowercased
        assert m["conditions"]["npu_power_mode_source"] == "observed"
        assert m["conditions"]["observed_at_utc"]


def test_no_supplied_condition_records_unknown_with_a_reason():
    """Never a silent blank, and never a guess: `unknown` plus why."""
    with tempfile.TemporaryDirectory() as d:
        results_io.write_rows(Path(d) / "m.csv", [_row("passed")])
        block = manifest.build_manifest(d, ["m.csv"])["conditions"]
        assert block["npu_power_mode"] == schema.UNKNOWN_CONDITION
        assert block["npu_power_mode_source"] == schema.UNKNOWN_CONDITION
        assert "no measurement condition was supplied" in block["npu_power_mode_detail"]


def test_build_manifest_does_not_query_the_device_behind_the_caller():
    """The default path must be usable with no xrt-smi anywhere near it.

    Asserted by running it on a PATH with nothing on it at all: a probe would
    still degrade to unknown, but it would also have SHELLED OUT, which is what
    keeps this suite hermetic and under a second.
    """
    with tempfile.TemporaryDirectory() as d:
        results_io.write_rows(Path(d) / "m.csv", [_row("passed")])
        empty = _stub_path(d, None)
        block = _with_path(
            empty, lambda: manifest.build_manifest(d, ["m.csv"])["conditions"]
        )
        assert block["npu_power_mode_source"] == schema.UNKNOWN_CONDITION
        assert "no measurement condition was supplied" in block["npu_power_mode_detail"]


def test_an_unknown_condition_does_not_make_a_tree_incomplete():
    """Two different verdicts: `complete` is about MEASURING, not about comparability."""
    with tempfile.TemporaryDirectory() as d:
        results_io.write_rows(Path(d) / "m.csv", [_row("passed")])
        m = manifest.build_manifest(d, ["m.csv"])
        assert m["conditions"]["npu_power_mode"] == schema.UNKNOWN_CONDITION
        assert m["complete"] is True


def test_a_malformed_condition_block_is_refused_at_build_time():
    """A typo'd key must fail here, not read back later as a silent unknown."""
    with tempfile.TemporaryDirectory() as d:
        results_io.write_rows(Path(d) / "m.csv", [_row("passed")])
        bad = schema.empty_conditions()
        bad["npu_powermode"] = "turbo"  # typo
        try:
            manifest.build_manifest(d, ["m.csv"], conditions=bad)
        except ValueError as e:
            assert "not in the schema" in str(e)
            return
        raise AssertionError("a malformed conditions block was accepted")


def test_observe_conditions_probes_and_labels_the_weaker_source():
    """A probe at manifest-build time is not an observation of the measurement."""
    with tempfile.TemporaryDirectory() as d:
        stub = _stub_path(d, _PLATFORM_REPORT % "Turbo")
        block = _with_path(stub, manifest.observe_conditions)
        assert block["npu_power_mode"] == "turbo"
        assert block["npu_power_mode_source"] == "probed_at_manifest_build"


def test_observe_conditions_reads_a_non_turbo_mode_rather_than_refusing():
    """This module RECORDS the condition; refusing on it is compare_roots' job."""
    with tempfile.TemporaryDirectory() as d:
        stub = _stub_path(d, _PLATFORM_REPORT % "Default")
        block = _with_path(stub, manifest.observe_conditions)
        assert block["npu_power_mode"] == "default"
        assert block["npu_power_mode_source"] == "probed_at_manifest_build"


def test_observe_conditions_degrades_when_xrt_smi_is_absent():
    with tempfile.TemporaryDirectory() as d:
        block = _with_path(_stub_path(d, None), manifest.observe_conditions)
        assert block["npu_power_mode"] == schema.UNKNOWN_CONDITION
        assert "xrt-smi" in block["npu_power_mode_detail"]


def test_observe_conditions_degrades_when_the_report_has_no_power_mode_line():
    with tempfile.TemporaryDirectory() as d:
        stub = _stub_path(d, "  Platform\n    Name  : RyzenAI-npu4\n")
        block = _with_path(stub, manifest.observe_conditions)
        assert block["npu_power_mode"] == schema.UNKNOWN_CONDITION
        assert block["npu_power_mode_detail"]


def test_cli_stamps_an_observed_mode_and_can_refuse_to_probe():
    with tempfile.TemporaryDirectory() as d:
        results_io.write_rows(Path(d) / "m.csv", [_row("passed")])
        out = Path(d) / "m.json"
        assert (
            manifest.main(
                [d, "--expect", "m.csv", "-o", str(out), "--npu-power-mode", "turbo"]
            )
            == 0
        )
        assert json.loads(out.read_text())["conditions"] == {
            "npu_power_mode": "turbo",
            "npu_power_mode_source": "observed",
            "npu_power_mode_detail": "supplied by the caller that measured",
            "observed_at_utc": json.loads(out.read_text())["conditions"][
                "observed_at_utc"
            ],
        }
        assert manifest.main([d, "--expect", "m.csv", "-o", str(out), "--no-probe"]) == 0
        block = json.loads(out.read_text())["conditions"]
        assert block["npu_power_mode"] == schema.UNKNOWN_CONDITION


# ---------------------------------------------------------------------------
# The toolchain block `[2026-08-12]` -- queue item 16.
#
# HERMETIC, like the conditions tests beside them: `observe_toolchain` is
# exercised against a scratch XRT version.json and a scratch sys.path, never
# against this host's real toolchain. A test that asserted "xrt_version ==
# 2.21.0" would be asserting the machine, and would go red on the next upgrade
# -- which is the event the field exists to RECORD, not to fail on.
# ---------------------------------------------------------------------------


def _manifest_toolchain(tmp, **kwargs):
    results_io.write_rows(Path(tmp) / "m.csv", [_row("passed")])
    return manifest.build_manifest(tmp, ["m.csv"], **kwargs)


def test_a_manifest_records_the_toolchain_under_the_diffed_key():
    """The point of the whole item: the key must be the one the diff reads."""
    with tempfile.TemporaryDirectory() as d:
        block = schema.empty_toolchain()
        block["xrt_version"] = "2.21.0+4eb1f4392a01"
        block["toolchain_source"] = "probed_at_manifest_build"
        man = _manifest_toolchain(d, toolchain=block)
        assert "toolchain" in man, sorted(man)
        assert man["toolchain"]["xrt_version"] == "2.21.0+4eb1f4392a01"


def test_no_supplied_toolchain_records_unknown_with_a_reason():
    """It is not probed behind the caller's back, and it is not assumed."""
    with tempfile.TemporaryDirectory() as d:
        block = _manifest_toolchain(d)["toolchain"]
        for name in schema.TOOLCHAIN_IDENTITY_FIELDNAMES:
            assert block[name] == schema.UNKNOWN_CONDITION, name
        assert "NOT assumed" in block["toolchain_detail"]
        assert "19-39%" in block["toolchain_detail"]


def test_a_malformed_toolchain_block_is_refused_at_build_time():
    """A typo'd key must fail here, not be written and read back as unknown."""
    with tempfile.TemporaryDirectory() as d:
        block = schema.empty_toolchain()
        block["xrt_verison"] = block.pop("xrt_version")
        try:
            _manifest_toolchain(d, toolchain=block)
        except ValueError as exc:
            # A typo trips both clauses (a key missing AND one invented); the
            # assertion is that it is refused and that the message names the
            # field, not which clause happens to fire first.
            assert "xrt_version" in str(exc), exc
        else:
            raise AssertionError("a typo'd toolchain key was accepted")


def test_an_unknown_toolchain_does_not_make_a_tree_incomplete():
    """Same split as the pmode: measuring everything asked for is `complete`.

    Whether two such trees may be COMPARED is compare_roots' question.
    """
    with tempfile.TemporaryDirectory() as d:
        assert _manifest_toolchain(d)["complete"] is True


def test_the_toolchain_block_did_not_change_the_schema_version_on_disk():
    """Pinned on the written artifact, not just on the constant. The manifest
    records whatever `schema.SCHEMA_VERSION` is -- 2 when this block landed,
    3 since the model scope (`[2026-08-23]`); the block itself never moved it."""
    with tempfile.TemporaryDirectory() as d:
        assert _manifest_toolchain(d)["schema_version"] == schema.SCHEMA_VERSION


def test_observe_toolchain_reads_versions_and_labels_the_source():
    with tempfile.TemporaryDirectory() as d:
        version_json = Path(d) / "version.json"
        version_json.write_text(
            json.dumps({"BUILD_VERSION": "2.21.0", "VERSION_HASH": "4eb1f4392a01beef"})
        )
        old = manifest.XRT_VERSION_JSON
        manifest.XRT_VERSION_JSON = version_json
        try:
            block = manifest.observe_toolchain()
        finally:
            manifest.XRT_VERSION_JSON = old
        # Truncated to 12 so the diff line stays readable.
        assert block["xrt_version"] == "2.21.0+4eb1f4392a01", block["xrt_version"]
        assert block["toolchain_source"] == "probed_at_manifest_build"
        schema.validate_toolchain(block)


def test_observe_toolchain_degrades_field_by_field_with_reasons():
    """A missing layer must not take the readable ones down with it."""
    with tempfile.TemporaryDirectory() as d:
        old = manifest.XRT_VERSION_JSON
        manifest.XRT_VERSION_JSON = Path(d) / "nope.json"
        try:
            block = manifest.observe_toolchain()
        finally:
            manifest.XRT_VERSION_JSON = old
        assert block["xrt_version"] == schema.UNKNOWN_CONDITION
        assert "does not exist" in block["toolchain_detail"]
        assert "xrt_version:" in block["toolchain_detail"]
        schema.validate_toolchain(block)


def test_observe_toolchain_survives_a_malformed_version_file():
    """Best-effort like _git: a manifest with imperfect provenance still beats none."""
    with tempfile.TemporaryDirectory() as d:
        bad = Path(d) / "version.json"
        bad.write_text("{not json at all")
        old = manifest.XRT_VERSION_JSON
        manifest.XRT_VERSION_JSON = bad
        try:
            block = manifest.observe_toolchain()
        finally:
            manifest.XRT_VERSION_JSON = old
        assert block["xrt_version"] == schema.UNKNOWN_CONDITION
        schema.validate_toolchain(block)


def test_observe_toolchain_never_writes_the_reader_only_absent():
    """`absent` means "older than the field" and a fresh probe is never that."""
    with tempfile.TemporaryDirectory() as d:
        old = manifest.XRT_VERSION_JSON
        manifest.XRT_VERSION_JSON = Path(d) / "nope.json"
        try:
            block = manifest.observe_toolchain()
        finally:
            manifest.XRT_VERSION_JSON = old
        assert block["toolchain_source"] != "absent"


def test_cli_probes_the_toolchain_and_no_probe_suppresses_it():
    with tempfile.TemporaryDirectory() as d:
        results_io.write_rows(Path(d) / "m.csv", [_row("passed")])
        out = Path(d) / "m.json"
        version_json = Path(d) / "version.json"
        version_json.write_text(json.dumps({"BUILD_VERSION": "0.0.0-test"}))
        old = manifest.XRT_VERSION_JSON
        manifest.XRT_VERSION_JSON = version_json
        try:
            assert manifest.main([d, "--expect", "m.csv", "-o", str(out)]) == 0
        finally:
            manifest.XRT_VERSION_JSON = old
        block = json.loads(out.read_text())["toolchain"]
        assert set(block) == set(schema.TOOLCHAIN_FIELDNAMES), sorted(block)
        assert block["xrt_version"] == "0.0.0-test", block

        assert manifest.main([d, "--expect", "m.csv", "-o", str(out), "--no-probe"]) == 0
        block = json.loads(out.read_text())["toolchain"]
        for name in schema.TOOLCHAIN_IDENTITY_FIELDNAMES:
            assert block[name] == schema.UNKNOWN_CONDITION, name


def _rungs(passed=0, failed=0, skipped=0):
    return (
        [_row("passed") for _ in range(passed)]
        + [_row("failed", "ERT_CMD_STATE_TIMEOUT") for _ in range(failed)]
        + [
            _row("skipped", "skipped: outside the mode's supported range")
            for _ in range(skipped)
        ]
    )


def test_one_row_where_nine_were_expected_is_not_complete():
    """M3, both directions in one test: the count is what closes it."""
    with tempfile.TemporaryDirectory() as d:
        results_io.write_rows(Path(d) / "m.csv", _rungs(passed=1))

        # Without counts this is the old verdict, and it is wrong about the run.
        assert manifest.build_manifest(d, ["m.csv"])["complete"] is True

        m = manifest.build_manifest(
            d, ["m.csv"], expected_rows={"m.csv": {"rows": 9, "measured": 9}}
        )
        assert m["complete"] is False
        assert m["row_counts_checked"] is True
        assert any("expected 9 row(s), found 1" in r for r in m["incomplete_reasons"])


def test_matching_counts_are_complete_and_recorded():
    with tempfile.TemporaryDirectory() as d:
        results_io.write_rows(Path(d) / "m.csv", _rungs(passed=3, skipped=2))
        m = manifest.build_manifest(
            d,
            ["m.csv"],
            expected_rows={"m.csv": {"rows": 5, "measured": 3, "skipped": 2}},
        )
        assert m["complete"] is True, m["incomplete_reasons"]
        record = m["expected_files"][0]
        assert record["observed_rows"]["total"] == 5
        assert record["observed_rows"]["skipped"] == 2
        assert record["expected_rows"]["measured"] == 3


def test_an_inapplicable_rung_recorded_as_failed_is_reported():
    """M6: the clause that needs `skipped` to be emitted to mean anything."""
    with tempfile.TemporaryDirectory() as d:
        results_io.write_rows(Path(d) / "m.csv", _rungs(passed=3, failed=2))
        m = manifest.build_manifest(
            d,
            ["m.csv"],
            expected_rows={"m.csv": {"rows": 5, "measured": 3, "skipped": 2}},
        )
        assert m["complete"] is False
        joined = " ".join(m["incomplete_reasons"])
        assert "expected 2 skipped row(s), found 0" in joined
        assert "indistinguishable from a regression" in joined


def test_a_regression_fails_the_measured_clause_not_the_row_clause():
    with tempfile.TemporaryDirectory() as d:
        results_io.write_rows(Path(d) / "m.csv", _rungs(passed=2, failed=1))
        m = manifest.build_manifest(
            d, ["m.csv"], expected_rows={"m.csv": {"rows": 3, "measured": 3}}
        )
        assert m["complete"] is False
        joined = " ".join(m["incomplete_reasons"])
        assert "expected 3 passed row(s), found 2" in joined
        assert "row(s), found" in joined and "expected 3 row(s)" not in joined


def test_counts_for_a_file_nobody_expected_are_a_failure():
    with tempfile.TemporaryDirectory() as d:
        results_io.write_rows(Path(d) / "m.csv", _rungs(passed=1))
        m = manifest.build_manifest(
            d, ["m.csv"], expected_rows={"m.csv": {"rows": 1}, "ghost.csv": {"rows": 4}}
        )
        assert m["complete"] is False
        assert any("not in the expected list" in r for r in m["incomplete_reasons"])


def test_a_missing_file_is_not_double_reported_by_the_count_clause():
    with tempfile.TemporaryDirectory() as d:
        m = manifest.build_manifest(
            d, ["gone.csv"], expected_rows={"gone.csv": {"rows": 4}}
        )
        assert m["complete"] is False
        assert len(m["incomplete_reasons"]) == 1, m["incomplete_reasons"]
        assert "MISSING" in m["incomplete_reasons"][0]
        assert m["expected_files"][0]["observed_rows"] is None


def test_no_expected_rows_leaves_the_verdict_and_the_keys_alone():
    with tempfile.TemporaryDirectory() as d:
        results_io.write_rows(Path(d) / "m.csv", _rungs(passed=1))
        m = manifest.build_manifest(d, ["m.csv"])
        assert m["complete"] is True
        assert m["row_counts_checked"] is False
        assert m["expected_files"][0]["expected_rows"] is None
        # observed counts are still recorded: describing what is there costs
        # nothing and is what makes a manifest worth diffing.
        assert m["expected_files"][0]["observed_rows"]["total"] == 1



# `[2026-08-26]` The TIMING-CONTRACT block -- queue item 19 review, finding 5.


def test_observe_timing_stamps_this_builds_contract_and_the_cache_state():
    import os

    saved = os.environ.get("LLMS_CACHE_XRT_RUNS")
    try:
        os.environ.pop("LLMS_CACHE_XRT_RUNS", None)
        block = manifest.observe_timing()
        schema.validate_timing(block)
        assert block["kernel_ms_contract"] == schema.KERNEL_MS_CONTRACT_NOW
        assert block["xrt_run_cache"] == "on"          # KernelCache's default
        assert block["timing_source"] == "probed_at_manifest_build"
        os.environ["LLMS_CACHE_XRT_RUNS"] = "0"
        assert manifest.observe_timing()["xrt_run_cache"] == "off"
        os.environ["LLMS_CACHE_XRT_RUNS"] = "1"
        assert manifest.observe_timing()["xrt_run_cache"] == "on"
    finally:
        if saved is None:
            os.environ.pop("LLMS_CACHE_XRT_RUNS", None)
        else:
            os.environ["LLMS_CACHE_XRT_RUNS"] = saved


def test_a_manifest_without_a_supplied_timing_block_records_unknown_with_a_reason():
    """The toolchain block's rule: NOT assumed. Which terms a recorded
    `kernel_ms` includes is a property of the build that MEASURED, not of the
    build that writes the manifest."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "coarse.csv").write_text("x")
        payload = manifest.build_manifest(root, ["coarse.csv"])
        block = payload[schema.TIMING_KEY]
        assert block["kernel_ms_contract"] == schema.UNKNOWN_CONDITION
        assert "NOT assumed" in block["timing_detail"]
        assert schema.timing_from_manifest(payload)["timing_source"] == \
            schema.UNKNOWN_CONDITION


def test_a_supplied_timing_block_is_validated_and_written_under_its_key():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "coarse.csv").write_text("x")
        good = manifest.observe_timing()
        payload = manifest.build_manifest(root, ["coarse.csv"], timing=good)
        assert payload[schema.TIMING_KEY]["kernel_ms_contract"] == \
            schema.KERNEL_MS_CONTRACT_NOW
        bad = dict(good)
        bad.pop("xrt_run_cache")
        try:
            manifest.build_manifest(root, ["coarse.csv"], timing=bad)
        except ValueError as exc:
            assert "missing keys" in str(exc), exc
        else:
            raise AssertionError("a typo'd timing block must fail at build time")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"manifest tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
