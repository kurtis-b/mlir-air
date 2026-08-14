# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Discover and run every host-only test module in this directory.

    python3 study/run_host_tests.py

CONTRACT
    Globs ``test_*.py`` beside this file, imports each, runs every ``test_*``
    function it defines, and prints one line per module plus a named grand
    total. Exit status is zero only if every test in every module passed.

WHY DISCOVERY RATHER THAN A LIST
    Convention rule 11: iron's ``pytest.ini`` set ``python_files = test.py``, so
    its own ``study/test_*.py`` files were silently never collected, and that hid
    real bugs. A hand-maintained list of modules here would reintroduce exactly
    that -- a new test module that nobody adds to the list is a test nobody runs,
    and it looks identical to a passing suite. So the glob is the point.

    The grand total is then printed as an exact count and matched exactly by
    ``run_study_host_tests.lit``, following the precedent
    ``run_block_cache_tests.lit`` sets. Discovery alone cannot catch a test that
    stops being *defined*; the pinned count can. The two together mean a suite
    that quietly shrinks fails the gate instead of passing it.

NO FRAMEWORK
    These modules are collectible by pytest unchanged -- ``test_*`` functions,
    plain ``assert`` -- but nothing here depends on it.

    `[2026-08-14]` pytest IS now installed (queue item 11(b) installed it beside
    matplotlib/pandas/seaborn in an exclusive window), so the original reason --
    "it is not installed" -- no longer holds. The choice does. Installing into
    the venv the gates run from is the hazard doc 09 records, and a suite that
    only runs under a framework is a suite that stops running the moment that
    framework is inconvenient to have present. This runner needs a bare
    interpreter and stays that way.

    One module does need a package: ``test_plots.py`` imports ``plots.py``,
    which imports matplotlib. That is a dependency ``study/requirements.txt``
    has declared since Phase F, and its absence surfaces as the import FAILURE
    below rather than as a silent skip.

FOOTGUNS
    - **Each module is imported, so its ``main()`` does not run**; the functions
      are collected directly. That is deliberate, but it means a module doing
      real work at import time would do it here too. Keep module level cheap.
    - **A module that fails to import is a failure, not a skip.** An ImportError
      from a missing dependency would otherwise silently subtract a module's
      whole test count, which the pinned total would then report as a mismatch
      with no cause attached. It is reported as the import error it is.
    - Modules run in sorted order in one process, so a test that mutates
      module-level state can affect a later module. None does today; if one must,
      it belongs in its own module rather than relying on ordering.
"""

from __future__ import annotations

import glob
import importlib.util
import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _load(path: str):
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_all(directory: str = _HERE) -> tuple[int, int, int, list[str]]:
    """Returns (passed, total, module_count, failure lines)."""
    paths = sorted(glob.glob(os.path.join(directory, "test_*.py")))
    passed = total = 0
    failures: list[str] = []

    for path in paths:
        name = os.path.basename(path)
        try:
            module = _load(path)
        except Exception as e:
            # Not a skip: see the docstring.
            failures.append(f"{name}: import failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            total += 1
            print(f"  FAIL  {name}: import failed", flush=True)
            continue

        tests = [v for k, v in sorted(vars(module).items()) if k.startswith("test_")]
        module_passed = 0
        for test in tests:
            total += 1
            try:
                test()
                module_passed += 1
                passed += 1
            except Exception as e:
                failures.append(f"{name}::{test.__name__}: {type(e).__name__}: {e}")
                traceback.print_exc()
        print(f"  {module_passed}/{len(tests)}  {name}", flush=True)

    return passed, total, len(paths), failures


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    directory = argv[0] if argv else _HERE

    passed, total, modules, failures = run_all(directory)
    for line in failures:
        print(f"FAILURE {line}")
    # Named exactly, and matched exactly by the lit wrapper.
    print(f"study host tests: {passed}/{total} passed in {modules} modules")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
