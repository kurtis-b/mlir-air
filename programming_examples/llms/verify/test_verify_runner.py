# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only audit of the shared verify driver's command line.

    python3 verify/test_verify_runner.py

WHY THIS FILE EXISTS
    `[2026-08-12]` ``verify_runner.py`` parsed ``--seed`` and never read it. A
    verification invoked with ``--seed N`` was therefore **not seeded by N**,
    and every reproducibility claim resting on that flag was empty -- while the
    run passed, printed PASS, and looked seeded. This is the failure shape the
    2026-08-12 queue is about: a mechanism that reads as present and does
    nothing, producing a green result rather than a wrong one.

    The flag was removed rather than wired, because nothing this gate reaches
    consumes randomness (``verify_runner``'s docstring records the measurement
    behind that). This module is what keeps the next one from landing: **every
    flag the driver parses must be read somewhere in the same module.**

WHY AN AST WALK RATHER THAN A RUN
    The driver's ``main()`` builds NPU runners and multi-GB HF references, so
    it cannot be exercised without a device and a checkpoint. Its ARGUMENT
    TABLE, though, is a static property of the source, and a dead flag is
    exactly a static property: a ``dest`` no expression reads. So this parses
    and never imports -- no numpy, no torch, no transformers, no XRT -- which
    is what lets it gate beside the compile-only tests instead of behind a
    device.

    This is ``agents/scripts``' dead-flag audit (queue item 19) narrowed to the
    one file all ten shipped models pass through, and kept where it can run on
    every change rather than only when someone re-runs a tree-wide sweep.

SCOPE, STATED
    A flag read only to be forwarded verbatim to a subprocess would count as
    read here, and should: ``--gate-phase auto`` re-invokes this driver, so
    passthrough IS a use. What this cannot see is a flag that is read and then
    has no effect. That is a different defect and needs a behavioural test.
"""

from __future__ import annotations

import ast
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_RUNNER = os.path.join(_HERE, "verify_runner.py")

#: argparse fills these in itself; they are never read by the module.
_ARGPARSE_INTERNAL = {"help"}


def _tree():
    return ast.parse(open(_RUNNER, encoding="utf-8").read(), _RUNNER)


def _dest_for(call: ast.Call) -> str | None:
    """The attribute name ``add_argument`` will store this option under."""
    for kw in call.keywords:
        if kw.arg == "dest" and isinstance(kw.value, ast.Constant):
            return kw.value.value
    longs, shorts = [], []
    for arg in call.args:
        if not isinstance(arg, ast.Constant) or not isinstance(arg.value, str):
            continue
        if arg.value.startswith("--"):
            longs.append(arg.value)
        elif arg.value.startswith("-"):
            shorts.append(arg.value)
        else:
            return arg.value.replace("-", "_")  # positional
    name = (longs or shorts or [None])[0]
    return name.lstrip("-").replace("-", "_") if name else None


def parsed_flags() -> dict[str, str]:
    """``dest -> the option string`` for every ``add_argument`` in the driver."""
    out = {}
    for node in ast.walk(_tree()):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            dest = _dest_for(node)
            if dest and dest not in _ARGPARSE_INTERNAL:
                first = next(
                    (
                        a.value
                        for a in node.args
                        if isinstance(a, ast.Constant) and isinstance(a.value, str)
                    ),
                    dest,
                )
                out[dest] = first
    return out


def read_names() -> set[str]:
    """Every attribute read off the parsed namespace, plus wholesale uses.

    ``args.x`` is the ordinary form. ``vars(args)`` / ``**vars(args)`` would
    make every flag reachable, so it is detected and reported rather than
    silently defeating the audit.
    """
    names, wholesale = set(), False
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "args":
                names.add(node.attr)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "vars" and any(
                isinstance(a, ast.Name) and a.id == "args" for a in node.args
            ):
                wholesale = True
    if wholesale:
        names.add("*")
    return names


def test_the_driver_parses_flags_at_all():
    """Guard the audit itself: an ast walk that finds nothing would pass every
    check below by vacuity, which is this batch's whole subject."""
    flags = parsed_flags()
    assert len(flags) >= 10, (
        f"only {len(flags)} flags parsed from verify_runner.py -- the ast walk "
        "has broken, not the driver. Every check in this module is vacuous "
        "until this passes."
    )
    for expected in ("runner", "prompts", "gate_phase", "gate_file"):
        assert expected in flags, f"{expected!r} not found; the walk is wrong"


def test_no_flag_is_parsed_and_never_read():
    """THE ITEM. A flag the driver accepts and never reads is a promise it does
    not keep -- and the caller cannot tell, because the run still passes."""
    flags = parsed_flags()
    read = read_names()
    if "*" in read:
        raise AssertionError(
            "verify_runner reads its namespace wholesale (vars(args)), which "
            "makes this audit unable to see a dead flag. Read flags by name."
        )
    dead = sorted(
        f"{option} (args.{dest})" for dest, option in flags.items() if dest not in read
    )
    assert not dead, (
        "verify_runner.py parses these and never reads them, so asking for "
        f"them changes nothing: {dead}. Either wire each to the code that "
        "needs it, or delete it -- do not leave it parsed and ignored."
    )


def test_the_audit_can_actually_fail():
    """The negative control, run on every invocation.

    A dead-flag audit that cannot detect a dead flag is the same defect it
    exists to catch. This injects one into a copy of the real source text and
    asserts the same walk finds it -- so the check is exercised against a
    positive case every time, not only on the day something breaks.
    """
    source = open(_RUNNER, encoding="utf-8").read()
    injected = source.replace(
        "    args = p.parse_args()",
        '    p.add_argument("--never-read", type=int, default=42)\n'
        "    args = p.parse_args()",
        1,
    )
    assert injected != source, (
        "STALE: the anchor 'args = p.parse_args()' is no longer in "
        "verify_runner.py, so this control injected nothing and proved "
        "nothing. Re-anchor it rather than deleting it."
    )
    tree = ast.parse(injected)
    dests = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            d = _dest_for(node)
            if d:
                dests.add(d)
    assert "never_read" in dests, "the walk did not see the injected flag"
    read = {
        n.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute)
        and isinstance(n.value, ast.Name)
        and n.value.id == "args"
    }
    assert "never_read" not in read, "the injected flag was somehow read"


def test_the_seed_flag_is_gone_and_stays_gone():
    """The specific one. Re-adding ``--seed`` without a consumer would be caught
    by the audit above; this names it so the reason is not lost to a diff."""
    flags = parsed_flags()
    assert "seed" not in flags, (
        "--seed is back. Nothing on the verify path consumes randomness "
        "(greedy decode, checkpoint weights, deterministic tokenization), so a "
        "seed here is either dead again or the path has changed -- if it has, "
        "wire the flag to what needs it and say so in verify_runner's "
        "docstring section 'WHY THERE IS NO --seed'."
    )


def test_the_gate_decodes_greedily_which_is_why_no_seed_is_needed():
    """The claim the removal rests on, checked against the source rather than
    remembered: the driver takes ``top1_token``, never a sampled token."""
    source = open(_RUNNER, encoding="utf-8").read()
    assert "top1_token" in source
    for sampler in ("multinomial", "temperature", "top_p", "do_sample"):
        assert sampler not in source, (
            f"verify_runner now mentions {sampler!r}. If the gate has gained a "
            "sampled decode it is no longer deterministic, and it needs a seed "
            "-- a real one, wired to the sampler."
        )


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"verify runner tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
