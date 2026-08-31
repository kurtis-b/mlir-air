# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Host tests for the split-phase max_seq binding (review of #34, P1): the
capture phase records the shape the NPU ran in the gate file; the compare
phase binds its report to that record and refuses a mismatched environment.

    python3 verify/test_gate_max_seq.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from verify_runner import _gate_max_seq  # noqa: E402


def test_match_binds_captured_value():
    got, bound = _gate_max_seq({"_max_seq": 1056, "0": {}}, 1056)
    assert (got, bound) == (1056, True)


def test_mismatch_is_refused():
    try:
        _gate_max_seq({"_max_seq": 1056}, 2048)
    except SystemExit as e:
        assert "max_seq=1056" in str(e) and "2048" in str(e)
    else:
        raise AssertionError("a mismatched compare environment was accepted")


def test_absent_record_reports_unverified():
    got, bound = _gate_max_seq({"0": {}}, 2048)
    assert (got, bound) == (2048, False)
    got, bound = _gate_max_seq(None, 2048)  # legacy 'both' path
    assert (got, bound) == (2048, False)


def test_capture_writes_the_record():
    """The write site exists and stores max_seq beside the per-prompt keys
    (source-level: the capture path needs a device to run)."""
    with open(os.path.join(_HERE, "verify_runner.py")) as f:
        src = f.read()
    assert 'captured["_max_seq"] = int(max_seq)' in src
    i = src.index('captured["_max_seq"]')
    assert "json.dump(captured, f)" in src[i:], "record written after dump?"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"gate max_seq tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
