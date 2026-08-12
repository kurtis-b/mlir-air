# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run a host test script and assert the pins its `.lit` carries.

    python3 lit_pin.py --lit <file.lit> [--check-prefix P] -- <command> [args...]

WHY THIS EXISTS
    `[2026-08-12]` Every host arm in this example had TWO entry points: a
    ``make`` target and a lit. The lit pipes the script through ``FileCheck``
    and pins an exact count (``blocked attention tests: 5/5 passed``); the
    ``make`` target ran the same script bare. So the pinned counts -- the only
    thing that catches a test which stops being DEFINED, since a globbing
    runner reports a smaller suite passing -- were unenforced whenever anyone
    ran ``make``, and a developer running ``make blocked-attention-tests`` saw
    a pass that asserted nothing beyond "the script exited 0".

    That is this batch's shape exactly: a mechanism that reads as present and
    does nothing, producing a green result rather than a wrong one. This module
    closes it by making the ``make`` target assert what the lit asserts.

WHY NOT JUST CALL FileCheck
    ``FileCheck`` is not part of this project's toolchain. It is not in
    ``build-xrt/bin`` or ``install-xrt/bin``, not on ``PATH``, and the lit
    suites reach it only through ``config.llvm_tools_dir``, an absolute path
    baked into ``build-xrt/programming_examples/lit.site.cfg.py`` pointing at a
    separate MLIR install. A Makefile cannot derive that path, and these
    targets are documented "No NPU required" and run under a bare ``python3``
    -- so requiring FileCheck would make them fail on any host without a
    second LLVM build. That trades a check nobody runs for a target nobody can
    run.

THE PIN LIVES IN THE LIT, AND IS READ FROM IT
    The expected count is NOT duplicated here. It is read out of the ``.lit``
    that already carries it, so ``make`` and ``lit`` assert the same literal by
    construction and updating one updates both. Copying the number into the
    Makefile would be the transcription defect this repository has spent a day
    removing.

WHAT IT MODELS, AND WHAT IT REFUSES
    These host lits use plain literal ``CHECK:`` lines: an ordered sequence of
    substring matches. That is all this implements. Anything else -- a
    ``{{regex}}``, a ``[[variable]]``, or any directive suffix (``-NEXT``,
    ``-NOT``, ``-SAME``, ``-DAG``, ``-COUNT``, ``-LABEL``, ``-EMPTY``) -- is
    REFUSED with a message, never approximated. A checker that quietly
    mis-modelled a directive would pass where FileCheck fails, which is the
    same defect one layer down.

    Directive scanning follows lit's own rule and this repository's warning
    about it: the WHOLE file is scanned for ``<PREFIX>:``, so writing the
    prefix in prose creates a directive.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

#: FileCheck syntax this module does not implement. Present => refuse.
_UNSUPPORTED_SUFFIXES = (
    "NEXT",
    "NOT",
    "SAME",
    "DAG",
    "LABEL",
    "EMPTY",
    "COUNT",
)


def directives(lit_text: str, prefix: str) -> list[str]:
    """The literal patterns ``prefix`` pins, in file order.

    Scans the whole file, as lit does: the text after ``<prefix>:`` on each
    line that carries it, stripped.
    """
    out = []
    for line in lit_text.splitlines():
        idx = line.find(prefix + ":")
        if idx < 0:
            continue
        # Not a directive if the prefix is part of a longer word/suffix form;
        # those are handled by the refusal below, which sees the raw text.
        out.append(line[idx + len(prefix) + 1 :].strip())
    return out


def refuse_unsupported(lit_text: str, prefix: str) -> None:
    for suffix in _UNSUPPORTED_SUFFIXES:
        token = f"{prefix}-{suffix}:"
        if token in lit_text:
            raise SystemExit(
                f"lit_pin: {token} is a FileCheck directive this module does "
                "not implement, and approximating it would make `make` and "
                "`lit` disagree about what passes. Run this arm through lit, "
                "or extend lit_pin deliberately."
            )


def check(output: str, patterns: list[str], lit_path: str, prefix: str) -> None:
    """Ordered substring match, FileCheck's ``CHECK:`` semantics."""
    for pattern in patterns:
        if "{{" in pattern or "[[" in pattern:
            raise SystemExit(
                f"lit_pin: {prefix} pin {pattern!r} in {os.path.basename(lit_path)} "
                "uses FileCheck regex/variable syntax, which this module does "
                "not implement. Run this arm through lit, or extend lit_pin."
            )

    remaining = output
    for pattern in patterns:
        idx = remaining.find(pattern)
        if idx < 0:
            raise SystemExit(
                f"\nlit_pin: FAILED -- {os.path.basename(lit_path)} pins\n"
                f"    {prefix}: {pattern}\n"
                "and the run above did not produce it (searching forward from "
                "the previous pin).\n"
                "This is the pinned expectation, not a flaky one: if the suite "
                "legitimately changed size, update the .lit -- that edit is the "
                "point, not friction to route around.\n"
            )
        remaining = remaining[idx + len(pattern) :]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lit", required=True, help="the .lit carrying the pins")
    ap.add_argument("--check-prefix", default="CHECK")
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("lit_pin: no command given after --")

    with open(args.lit, encoding="utf-8") as fh:
        lit_text = fh.read()

    refuse_unsupported(lit_text, args.check_prefix)
    patterns = directives(lit_text, args.check_prefix)
    if not patterns:
        # A prefix that matches nothing would assert nothing and exit 0 -- the
        # "a lit suite that selects nothing exits 0" failure, verbatim.
        raise SystemExit(
            f"lit_pin: {args.lit} carries no {args.check_prefix}: directives, so "
            "this would assert nothing and pass. Check --check-prefix."
        )

    proc = subprocess.run(command, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    sys.stdout.flush()

    if proc.returncode != 0:
        raise SystemExit(
            f"lit_pin: the command exited {proc.returncode}; not checking pins."
        )

    check(proc.stdout, patterns, args.lit, args.check_prefix)
    print(
        f"[lit-pin] {len(patterns)} {args.check_prefix} pin(s) from "
        f"{os.path.basename(args.lit)} matched"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
