# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Host regression for --compile-mode compile-only (no device, no aircc).

    python3 test_compile_only.py

compile-only must COMPILE AND EXIT: XRTBackend.compile is called and
XRTRunner.run_test is never reached. The default mode arm proves the harness
can see a dispatch, so the first arm cannot pass vacuously. Both the backend
and the runner are stubbed -- nothing builds, nothing dispatches.
"""

import os
import runpy
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_MMIO = os.path.join(_HERE, "mmio.py")


class _Dispatched(Exception):
    pass


def _run(argv, calls):
    import air.backend.xrt as xrt_mod
    import air.backend.xrt_runner as runner_mod

    class FakeBackend:
        def __init__(self, **kw):
            calls.append("backend")

        def compile(self, module, **kw):
            calls.append("compile")

        def unload(self):
            calls.append("unload")

    def fake_run_test(self, *a, **kw):
        calls.append("run_test")
        raise _Dispatched()

    old_backend = xrt_mod.XRTBackend
    old_run_test = runner_mod.XRTRunner.run_test
    old_argv = sys.argv
    xrt_mod.XRTBackend = FakeBackend
    runner_mod.XRTRunner.run_test = fake_run_test
    sys.argv = ["mmio.py"] + argv
    try:
        runpy.run_path(_MMIO, run_name="__main__")
        return None
    except SystemExit as e:
        return e.code
    except _Dispatched:
        return "dispatched"
    finally:
        xrt_mod.XRTBackend = old_backend
        runner_mod.XRTRunner.run_test = old_run_test
        sys.argv = old_argv


def main():
    calls = []
    rc = _run(["--compile-mode", "compile-only", "--output-format", "elf"], calls)
    assert rc == 0, f"compile-only exited {rc!r}"
    assert "compile" in calls, f"compile-only never compiled: {calls}"
    assert (
        "run_test" not in calls
    ), f"compile-only reached run_test -- the fail-open dispatch is back: {calls}"
    print("PASS  compile-only compiles and exits without run_test")
    calls = []
    rc = _run([], calls)  # default mode: the dispatch path must be reachable
    assert (
        rc == "dispatched" and "run_test" in calls
    ), f"the harness cannot see a dispatch (vacuous): rc={rc!r} calls={calls}"
    print("PASS  default mode reaches run_test (the harness is not vacuous)")
    print("mmio compile-only tests: 2/2 passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
