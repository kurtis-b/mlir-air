# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only checks on the per-output study lock.

    python3 study/test_run_lock.py

The contention tests use a real subprocess rather than a second handle in this
process, because ``flock(2)`` is per open file description and a same-process
re-open would demonstrate nothing about two runners.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import run_lock  # noqa: E402

_CHILD = textwrap.dedent("""
    import sys
    sys.path.insert(0, {here!r})
    import run_lock
    try:
        with run_lock.hold({lock!r}, study="child"):
            print("ACQUIRED")
    except run_lock.StudyAlreadyRunning as e:
        print("REFUSED", e)
        sys.exit(3)
    """)


def _child(lock_path):
    return subprocess.run(
        [sys.executable, "-c", _CHILD.format(here=_HERE, lock=str(lock_path))],
        capture_output=True,
        text=True,
    )


def test_lock_path_appends_rather_than_replaces_the_suffix():
    """`a.csv` and `a.json` must not collide on one lock."""
    assert run_lock.lock_path_for("results/coarse.csv").name == "coarse.csv.lock"
    assert run_lock.lock_path_for("results/coarse.json").name == "coarse.json.lock"


def test_lock_path_stays_beside_the_output():
    path = run_lock.lock_path_for("/a/b/results/coarse.csv")
    assert path.parent == Path("/a/b/results")


def test_holding_creates_the_parent_directory():
    with tempfile.TemporaryDirectory() as d:
        lock = Path(d) / "nested" / "deeper" / "out.csv.lock"
        with run_lock.hold(lock, study="s"):
            assert lock.exists()


def test_the_holder_is_recorded_while_held():
    with tempfile.TemporaryDirectory() as d:
        lock = Path(d) / "out.csv.lock"
        with run_lock.hold(lock, study="sequence ladder"):
            text = lock.read_text(encoding="utf-8")
        assert f"pid={os.getpid()}" in text
        assert "study=sequence ladder" in text


def test_the_record_is_cleared_on_release():
    """An empty lock file means released, not held by an unknown process."""
    with tempfile.TemporaryDirectory() as d:
        lock = Path(d) / "out.csv.lock"
        with run_lock.hold(lock, study="s"):
            pass
        assert lock.read_text(encoding="utf-8").strip() == ""


def test_a_second_process_is_refused_while_held():
    with tempfile.TemporaryDirectory() as d:
        lock = Path(d) / "out.csv.lock"
        with run_lock.hold(lock, study="parent study"):
            result = _child(lock)
        assert result.returncode == 3, result.stdout + result.stderr
        assert "REFUSED" in result.stdout


def test_the_refusal_names_the_holder():
    """Fail-fast is only useful if it says who to go and look at."""
    with tempfile.TemporaryDirectory() as d:
        lock = Path(d) / "out.csv.lock"
        with run_lock.hold(lock, study="parent study"):
            result = _child(lock)
        assert f"pid={os.getpid()}" in result.stdout
        assert "study=parent study" in result.stdout


def test_the_refusal_says_it_is_not_the_device_queue():
    """The one thing a reader must not conclude from this lock."""
    with tempfile.TemporaryDirectory() as d:
        lock = Path(d) / "out.csv.lock"
        with run_lock.hold(lock, study="s"):
            result = _child(lock)
        assert "not the device queue" in result.stdout


def test_a_second_process_acquires_after_release():
    with tempfile.TemporaryDirectory() as d:
        lock = Path(d) / "out.csv.lock"
        with run_lock.hold(lock, study="s"):
            pass
        result = _child(lock)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "ACQUIRED" in result.stdout


def test_the_lock_is_released_when_the_block_raises():
    with tempfile.TemporaryDirectory() as d:
        lock = Path(d) / "out.csv.lock"
        try:
            with run_lock.hold(lock, study="s"):
                raise ValueError("boom")
        except ValueError:
            pass
        result = _child(lock)
        assert result.returncode == 0, result.stdout + result.stderr


def test_different_outputs_do_not_serialize_against_each_other():
    """Per-mode ladder rungs write different CSVs and must both proceed."""
    with tempfile.TemporaryDirectory() as d:
        first = run_lock.lock_path_for(Path(d) / "coarse.csv")
        second = run_lock.lock_path_for(Path(d) / "runlist.csv")
        with run_lock.hold(first, study="a"):
            result = _child(second)
        assert result.returncode == 0, result.stdout + result.stderr


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"run-lock tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
