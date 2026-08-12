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


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"manifest tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
