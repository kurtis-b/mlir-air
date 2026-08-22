"""Both directions of ``sweep.registry_sweep.require_turbo`` -- the ONE Turbo-pmode rule --
through the REAL ``npu_power_mode`` parser, with ``xrt-smi`` stubbed on PATH.

The retired port-loop ``pmode_guard.py`` selftest (tag ``pre-cleanup-20260821``) drove the
parser the same way: a fake ``xrt-smi`` first on PATH printing a chosen report, or no
``xrt-smi`` at all. Monkeypatching ``npu_power_mode`` would leave a parser regression that
returns "turbo" for a missing or unparseable report undetected, so nothing here stubs Python.
"""
from __future__ import annotations

import os
import stat
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sweep import registry_sweep as rs  # noqa: E402


def _with_xrt_smi(text, fn):
    """Run ``fn`` with PATH = one temp dir holding a fake ``xrt-smi`` that prints ``text``
    (``text=None``: the dir is empty, so ``xrt-smi`` is not on PATH at all)."""
    d = tempfile.mkdtemp(prefix="xrt-smi-stub-")
    if text is not None:
        p = os.path.join(d, "xrt-smi")
        with open(p, "w") as f:
            f.write("#!/bin/sh\n/bin/cat <<'REPORT'\n" + text + "\nREPORT\n")
        os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR)
    saved = os.environ.get("PATH")
    os.environ["PATH"] = d
    try:
        return fn()
    finally:
        os.environ["PATH"] = saved


def _refused(text):
    try:
        _with_xrt_smi(text, rs.require_turbo)
    except rs.TurboNotEnforced as exc:
        return str(exc)
    raise AssertionError(f"require_turbo passed on report {text!r}")


def test_a_turbo_report_passes_through_the_real_parser():
    assert _with_xrt_smi("Platform\n  Power Mode             : Turbo \n", rs.require_turbo) is None
    assert _with_xrt_smi("  Power Mode : Turbo\n", rs.npu_power_mode) == (
        "turbo", "xrt-smi examine -r platform")


def test_a_default_report_refuses_and_names_the_mode():
    assert "`default`, not turbo" in _refused("  Power Mode             : Default \n")


def test_no_xrt_smi_on_path_refuses_as_undetermined():
    msg = _refused(None)
    assert "could not determine" in msg and "xrt-smi is not on PATH" in msg, msg


def test_a_report_without_a_power_mode_line_refuses_as_undetermined():
    msg = _refused("Platform\n  Name : NPU\n")
    assert "could not determine" in msg and "no 'Power Mode' line" in msg, msg


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"require_turbo tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
