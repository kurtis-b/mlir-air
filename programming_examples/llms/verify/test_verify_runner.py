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


def _tree(source: str | None = None) -> ast.AST:
    """The driver's AST, or the AST of ``source`` when a control injects one.
    Every helper below takes the tree it audits, so a control exercises the
    SAME code the production check runs -- never a private re-walk."""
    if source is None:
        source = open(_RUNNER, encoding="utf-8").read()
    return ast.parse(source, _RUNNER)


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


def parsed_flags(tree: ast.AST) -> dict[str, str]:
    """``dest -> the option string`` for every ``add_argument`` in ``tree``."""
    out = {}
    for node in ast.walk(tree):
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


def read_names(tree: ast.AST) -> set[str]:
    """Every attribute LOADED off the parsed namespace, plus wholesale uses.

    ``args.x`` in a Load context is the ordinary form. A Store
    (``args.x = ...``) is not a read: a driver that overwrites a parsed value
    and never looks at what the caller passed has ignored the flag exactly as
    if it had never read it. ``vars(args)`` / ``**vars(args)`` would make
    every flag reachable, so it is detected and reported rather than silently
    defeating the audit.
    """
    names, wholesale = set(), False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "args"
            and isinstance(node.ctx, ast.Load)
        ):
            names.add(node.attr)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "vars" and any(
                isinstance(a, ast.Name) and a.id == "args" for a in node.args
            ):
                wholesale = True
    if wholesale:
        names.add("*")
    return names


def dead_flags(tree: ast.AST) -> list[str]:
    """The flags ``tree`` parses and never loads -- THE predicate. The
    production check and every control below call this one function."""
    flags = parsed_flags(tree)
    read = read_names(tree)
    if "*" in read:
        raise AssertionError(
            "verify_runner reads its namespace wholesale (vars(args)), which "
            "makes this audit unable to see a dead flag. Read flags by name."
        )
    return sorted(
        f"{option} (args.{dest})" for dest, option in flags.items() if dest not in read
    )


#: Names that would mean a sampled, seed-needing decode if they were called
#: on the token-selection path.
_SAMPLER_NAMES = {"multinomial", "sample", "pick_token", "choice", "random"}


def greedy_violations(tree: ast.AST) -> tuple[int, list[str]]:
    """Inspect ``_generate_with_topk``: every expression that becomes a chosen
    token (``chosen = [...]``, ``chosen.append(...)``, ``next_tok = ...``) must
    be a ``.top1_token`` load, and nothing on that path may call a sampler.

    Returns ``(selection sites seen, violations)``; the caller asserts the site
    count so a renamed function cannot make the check vacuous.
    """
    fn = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_generate_with_topk"
        ),
        None,
    )
    if fn is None:
        return 0, ["_generate_with_topk is missing"]

    def is_top1_load(e: ast.AST) -> bool:
        return (
            isinstance(e, ast.Attribute)
            and e.attr == "top1_token"
            and isinstance(e.ctx, ast.Load)
        )

    sites, bad = 0, []
    for node in ast.walk(fn):
        selected = []
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in ("chosen", "next_tok")
            for t in node.targets
        ):
            v = node.value
            selected = list(v.elts) if isinstance(v, ast.List) else [v]
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "append"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "chosen"
        ):
            selected = list(node.args)
        for e in selected:
            sites += 1
            if not is_top1_load(e):
                bad.append(ast.unparse(e))
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if name in _SAMPLER_NAMES:
                bad.append(ast.unparse(node))
    return sites, bad


def _inject_after_parse(source: str, lines: str) -> str:
    """A copy of the driver source with ``lines`` inserted around the
    ``parse_args`` anchor; the control asserts the anchor was found."""
    injected = source.replace("    args = p.parse_args()", lines, 1)
    assert injected != source, (
        "STALE: the anchor 'args = p.parse_args()' is no longer in "
        "verify_runner.py, so this control injected nothing and proved "
        "nothing. Re-anchor it rather than deleting it."
    )
    return injected


def test_a_stored_flag_does_not_count_as_read():
    """Negative control for the Load rule: a flag the driver parses and then
    OVERWRITES (``args.x = "fixed"``) has ignored the caller's value, and the
    real predicate must still report it dead."""
    source = open(_RUNNER, encoding="utf-8").read()
    injected = _inject_after_parse(
        source,
        '    p.add_argument("--never-read", type=int, default=42)\n'
        "    args = p.parse_args()\n"
        '    args.never_read = "fixed"\n',
    )
    tree = _tree(injected)
    assert "never_read" in parsed_flags(tree), "the walk did not see the flag"
    assert "never_read" not in read_names(tree), (
        "a Store of args.never_read was counted as a read; the audit would "
        "pass a flag whose supplied value is thrown away"
    )
    assert "--never-read (args.never_read)" in dead_flags(tree)


def test_the_driver_parses_flags_at_all():
    """Guard the audit itself: an ast walk that finds nothing would pass every
    check below by vacuity, which is this batch's whole subject."""
    flags = parsed_flags(_tree())
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
    dead = dead_flags(_tree())
    assert not dead, (
        "verify_runner.py parses these and never reads them, so asking for "
        f"them changes nothing: {dead}. Either wire each to the code that "
        "needs it, or delete it -- do not leave it parsed and ignored."
    )


def test_the_audit_can_actually_fail():
    """The negative control, run on every invocation, THROUGH THE REAL
    PREDICATE: a dead flag injected into a copy of the driver source must come
    back from ``dead_flags`` -- the same function the production check calls,
    so a regression in ``parsed_flags``/``read_names`` turns this red too."""
    source = open(_RUNNER, encoding="utf-8").read()
    injected = _inject_after_parse(
        source,
        '    p.add_argument("--never-read", type=int, default=42)\n'
        "    args = p.parse_args()",
    )
    dead = dead_flags(_tree(injected))
    assert (
        "--never-read (args.never_read)" in dead
    ), f"the real predicate did not report the injected dead flag: {dead}"


def test_the_seed_flag_is_gone_and_stays_gone():
    """The specific one. Re-adding ``--seed`` without a consumer would be caught
    by the audit above; this names it so the reason is not lost to a diff."""
    flags = parsed_flags(_tree())
    assert "seed" not in flags, (
        "--seed is back. Nothing on the verify path consumes randomness "
        "(greedy decode, checkpoint weights, deterministic tokenization), so a "
        "seed here is either dead again or the path has changed -- if it has, "
        "wire the flag to what needs it and say so in verify_runner's "
        "docstring section 'WHY THERE IS NO --seed'."
    )


def test_the_gate_decodes_greedily_which_is_why_no_seed_is_needed():
    """The claim the removal rests on, checked against the EXECUTABLE token
    selection rather than the source text: every chosen token in
    ``_generate_with_topk`` is a ``.top1_token`` load and no sampler is called
    on that path. Comments and docstrings cannot satisfy this."""
    sites, bad = greedy_violations(_tree())
    assert sites >= 3, (
        f"only {sites} token-selection sites found in _generate_with_topk; "
        "the walk no longer sees the decode loop, so this check is vacuous"
    )
    assert not bad, (
        f"_generate_with_topk selects tokens by something other than "
        f"top1_token: {bad}. If the gate has gained a sampled decode it is no "
        "longer deterministic, and it needs a seed -- a real one, wired to "
        "the sampler."
    )


def test_the_greedy_check_can_actually_fail():
    """Mutation control for the check above: replace the decode-step token
    read with an unlisted sampler and the same predicate must object."""
    source = open(_RUNNER, encoding="utf-8").read()
    mutated = source.replace(
        "chosen.append(ds.top1_token)", "chosen.append(runner.pick_token(ds))", 1
    )
    assert (
        mutated != source
    ), "STALE: 'chosen.append(ds.top1_token)' anchor is gone; re-anchor"
    sites, bad = greedy_violations(_tree(mutated))
    assert (
        sites >= 3 and bad
    ), f"the greedy check did not object to a sampled token: sites={sites} bad={bad}"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"verify runner tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
