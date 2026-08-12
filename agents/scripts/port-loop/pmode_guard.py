#!/usr/bin/env python3
"""The NPU power-mode precondition for the gates that assert on latency.

WHY THIS EXISTS
    The `xrt-smi` power mode is a MEASUREMENT CONDITION and it silently resets.
    It is non-persistent: every reboot and every `amdxdna` reload puts it back to
    `Default`, and at `Default` a dispatch-bound measurement on this host runs
    roughly 15-20x slow (the study's verdict rung reads ~2.5-2.7 s against 156 ms
    at Turbo -- README trap 0, doc 32). That cost a day of measurement once
    already, diagnosed as a "machine anomaly" before the boot history gave it
    away.

    Every latency gate therefore fails at `Default`, and -- this is the part that
    matters -- it fails LOOKING LIKE A COMPILER REGRESSION. `gate-h.sh` is the one
    place in this harness that halts an unattended run, so trap 0's failure mode
    reaching it is worse than a slow measurement: it is a wrong diagnosis handed
    to whoever reads the halt.

WHAT IT DOES NOT DO
    It does not SET the power mode. That needs root and is the operator's action:

        sudo xrt-smi configure --device 0000:64:00.1 --pmode turbo

    verified with `xrt-smi examine -r platform`. This module only refuses to let a
    latency assertion run against a condition that invalidates it.

NO SECOND PARSER
    Detection is `sweep/registry_sweep.py`'s `npu_power_mode()` / `require_turbo()`
    -- the same implementation `study/run_mode.py` and `study/component_groups.py`
    already fail closed on, imported rather than re-derived. gate-h.sh makes the
    same call about `extract_perf.py` in its leg 4 header ("Do not add a second
    parser here") and for the same reason: two parsers of one device's output
    disagree eventually, and the disagreement shows up as a gate verdict.

FAIL CLOSED, INCLUDING "COULD NOT DETERMINE"
    Undetectable is a refusal, not a pass. If `xrt-smi` is missing or its output
    cannot be parsed, this refuses. `require_turbo` already made that choice and
    the reasoning is iron's: it shipped a smoke test that reported 21/21 passed on
    a machine where every measurement had failed, and "cannot check, so green" is
    that same shape.

THE FLOOR IS ALSO CONDITIONED
    A recorded throughput floor is a number measured at some power mode, and
    comparing it against a run at a different one is not a comparison. The floor
    file did not record the mode until `[2026-08-12]`, so entries seeded before
    then carry `unknown` -- see `--baseline` below for why that is a loud flag
    rather than a refusal, and why it must not be "fixed" by guessing.

FOOTGUN
    Nothing here may become a way to DEFEAT the guard. There is deliberately no
    `PL_PMODE=turbo` override: the selftest stubs the device query by putting a
    fake `xrt-smi` on PATH, which exercises the real parse path end to end and
    leaves no bypass in the production path.

ENTRY POINTS
    python3 agents/scripts/port-loop/pmode_guard.py require [--baseline F] [--model M]...
    python3 agents/scripts/port-loop/pmode_guard.py observe    # bare mode, or `unknown`
    python3 agents/scripts/port-loop/pmode_guard.py selftest   # both directions, no hardware
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent.parent
_EXAMPLE = _ROOT / "programming_examples" / "transformer_layer"

# _EXAMPLE ahead of anything else so `sweep` is this example's package. Matches the
# path dance run_mode.py and registry_sweep.py both do, and for the same reason: the
# Ryzen setup scripts leave a trailing colon on PYTHONPATH, so the cwd is already on
# sys.path and a guarded insert can be a silent no-op against the wrong tree.
_p = str(_EXAMPLE)
if _p in sys.path:
    sys.path.remove(_p)
sys.path.insert(0, _p)

#: This host's NPU, per README trap 0. `xrt-smi configure` wants the BDF explicitly.
DEVICE_BDF = "0000:64:00.1"

FIX = f"sudo xrt-smi configure --device {DEVICE_BDF} --pmode turbo"
VERIFY = "xrt-smi examine -r platform"

#: What a floor entry records when it was seeded before the field existed. NOT a
#: synonym for "probably turbo": the 2026-08-06 floor falls inside an uninterrupted
#: Turbo window, which is an inference from boot history and not an observation, and
#: writing an inference into a data field is how the prose rule got there in the
#: first place (doc 34 M4). Recorded as unknown, flagged loudly, never guessed.
UNKNOWN = "unknown"

BASELINE_KEY = "npu_power_mode"


class PmodeRefusal(RuntimeError):
    """The latency assertion's precondition does not hold."""


def _detect():
    """(mode, detail) from the single shared implementation. Fail closed on import."""
    try:
        from sweep.registry_sweep import npu_power_mode
    except Exception as exc:  # pragma: no cover - harness breakage, not a task
        raise PmodeRefusal(
            f"could not import the shared power-mode check from "
            f"{_EXAMPLE}/sweep/registry_sweep.py ({type(exc).__name__}: {exc}). "
            f"That is a harness bug. It refuses rather than skipping, because a "
            f"latency gate that cannot check its own precondition must not run."
        ) from exc
    return npu_power_mode()


def observed_mode():
    """The device's power mode, lowercased, or None when undeterminable."""
    mode, _detail = _detect()
    return mode


def check(baseline_path=None, models=(), where=""):
    """Refuse unless the latency precondition holds. Returns (mode, [flags]).

    Clause 1 -- the run is at turbo. Anything else, including undeterminable,
                refuses.
    Clause 2 -- every named floor entry that RECORDS a mode records the same one.
                A floor measured at `Default` cannot gate a run at Turbo, and the
                reverse is the case that actually bites.
    Clause 3 -- a floor entry with no recorded mode is FLAGGED, not refused. The
                shipped floor predates the field; refusing on it would make the
                gate unpassable until someone re-seeds, and "re-seed to make the
                gate pass" is the exact move the driver-owned floor file exists to
                prevent.
    """
    site = f" [{where}]" if where else ""
    mode, detail = _detect()

    if mode is None:
        raise PmodeRefusal(
            f"could not determine the NPU power mode{site} ({detail}). A latency "
            f"assertion that cannot be shown to be at turbo must not run: at "
            f"`Default` this host measures ~15-20x slow and the gate fails looking "
            f"like a compiler regression (README trap 0). Set it with `{FIX}` "
            f"(needs the operator) and verify with `{VERIFY}`."
        )
    if mode != "turbo":
        raise PmodeRefusal(
            f"NPU power mode is `{mode}`, not turbo{site} ({detail}). The pmode is "
            f"non-persistent and resets on every reboot and every amdxdna reload; "
            f"at `Default` this host measures ~15-20x slow, so this gate would fail "
            f"on the power mode while reading as a compiler regression (README "
            f"trap 0). Set it with `{FIX}` (needs the operator, does not persist) "
            f"and verify with `{VERIFY}`."
        )

    flags = []
    if baseline_path:
        flags = _check_floor(baseline_path, models, mode)
    return mode, flags


def _check_floor(baseline_path, models, mode):
    """Clauses 2 and 3 against the recorded floor. Returns the flag lines."""
    try:
        with open(baseline_path) as f:
            base = json.load(f)
    except Exception as exc:
        raise PmodeRefusal(
            f"could not read the throughput floor at {baseline_path} "
            f"({type(exc).__name__}: {exc}); refusing rather than comparing "
            f"against a floor whose measurement condition cannot be read."
        ) from exc

    entries = base.get("models", {})
    flags = []
    for m in models:
        entry = entries.get(m)
        if entry is None:
            # Not this check's business: leg 4 already reports unseeded models and
            # fails closed on an empty intersection.
            continue
        recorded = entry.get(BASELINE_KEY)
        if recorded is None or str(recorded).strip().lower() in ("", UNKNOWN):
            flags.append(
                f"{m}: floor recorded at an UNKNOWN power mode "
                f"(seeded {entry.get('recorded_utc', '?')} at "
                f"{entry.get('recorded_at_sha', '?')}, before the field existed). "
                f"The comparison is unconditioned on the axis that costs 15-20x. "
                f"Re-seed it -- as an operator, between phases, on a known-good "
                f"build -- to condition it; do NOT edit the number to make a gate pass."
            )
            continue
        recorded = str(recorded).strip().lower()
        if recorded != mode:
            raise PmodeRefusal(
                f"the recorded floor for {m} was measured at power mode "
                f"`{recorded}` and this run is at `{mode}`. tok/s on this host "
                f"differs by ~15-20x across that boundary, so this is not a "
                f"comparison and a verdict from it means nothing either way. "
                f"Never splice a comparison across a pmode change (README trap 0). "
                f"Re-seed the floor at `{mode}` as an operator, between phases."
            )
    return flags


# ----------------------------------------------------------------------------------
# selftest: every clause in both directions, with the device query stubbed on PATH.
#
# A guard nobody has seen refuse is not evidence -- the same rule phase_e_selftest.py
# states as "a clause with no failing case is a clause nobody has shown discriminates".
# The stub is a real executable named xrt-smi on a PATH containing only it, so the
# production parse path runs unmodified; there is no test-only branch in `check`.
# ----------------------------------------------------------------------------------

_PLATFORM_REPORT = """\
  Platform
    Name                   : RyzenAI-npu4
    Power Mode             : %s
"""


def _stub_xrt_smi(tmp, text):
    """A PATH directory holding a fake xrt-smi that prints `text`. `text=None` -> empty dir."""
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
        # shutil.which caches nothing, but registry_sweep may already be imported;
        # the lookup happens per call, so this is enough.
        assert shutil is not None
        return fn()
    finally:
        os.environ["PATH"] = old


def _baseline(tmp, entry):
    p = Path(tempfile.mkdtemp(dir=tmp)) / "floor.json"
    models = {} if entry is None else {"llama32_1b": entry}
    p.write_text(json.dumps({"models": models, "floor_fraction": 0.85}))
    return str(p)


def selftest():
    print("pmode_guard selftest: every clause in both directions, no hardware\n")
    tmp = tempfile.mkdtemp(prefix="pmode-selftest-")
    seeded = {"decode_tokens_per_sec": 11.1, "recorded_utc": "x", "recorded_at_sha": "y"}

    turbo = _stub_xrt_smi(tmp, _PLATFORM_REPORT % "Turbo")
    default = _stub_xrt_smi(tmp, _PLATFORM_REPORT % "Default")
    silent = _stub_xrt_smi(tmp, "  Platform\n    Name  : RyzenAI-npu4\n")
    absent = _stub_xrt_smi(tmp, None)

    floor_turbo = _baseline(tmp, dict(seeded, npu_power_mode="turbo"))
    floor_default = _baseline(tmp, dict(seeded, npu_power_mode="default"))
    floor_unknown = _baseline(tmp, dict(seeded, npu_power_mode="unknown"))
    floor_nofield = _baseline(tmp, dict(seeded))
    floor_absent = _baseline(tmp, None)

    M = ["llama32_1b"]
    # (name, path-dir, kwargs, expect_refusal, expect_flag)
    cases = [
        ("Turbo, no floor named", turbo, {}, False, False),
        ("Default refuses", default, {}, True, False),
        ("no xrt-smi on PATH refuses", absent, {}, True, False),
        ("no Power Mode line refuses", silent, {}, True, False),
        ("Turbo vs a Turbo-recorded floor", turbo,
         {"baseline_path": floor_turbo, "models": M}, False, False),
        ("Turbo vs a Default-recorded floor refuses", turbo,
         {"baseline_path": floor_default, "models": M}, True, False),
        ("Turbo vs an unknown-pmode floor flags", turbo,
         {"baseline_path": floor_unknown, "models": M}, False, True),
        ("Turbo vs a floor with no pmode field flags", turbo,
         {"baseline_path": floor_nofield, "models": M}, False, True),
        ("an unseeded model is leg 4's business, not this check's", turbo,
         {"baseline_path": floor_absent, "models": M}, False, False),
        ("an unreadable floor refuses", turbo,
         {"baseline_path": os.path.join(tmp, "nope.json"), "models": M}, True, False),
        ("Default outranks a matching floor", default,
         {"baseline_path": floor_default, "models": M}, True, False),
    ]

    failures = 0
    for name, d, kwargs, want_refusal, want_flag in cases:
        got_refusal, got_flag, msg = False, False, ""
        try:
            _mode, flags = _with_path(d, lambda: check(**kwargs))
            got_flag = bool(flags)
            msg = flags[0][:60] if flags else "proceeded"
        except PmodeRefusal as exc:
            got_refusal, msg = True, str(exc)[:60]

        ok = (got_refusal == want_refusal) and (got_flag == want_flag)
        want = "REFUSE" if want_refusal else ("FLAG" if want_flag else "proceed")
        print(f"  [{'ok' if ok else 'WRONG':>5}] expected to {want:<7} : {name}")
        if not ok:
            print(f"          got refusal={got_refusal} flag={got_flag}: {msg}")
            failures += 1

    shutil.rmtree(tmp, ignore_errors=True)
    print()
    if failures:
        print(f"selftest: {failures} of {len(cases)} clause(s) did not behave as specified")
        return 1
    print(f"selftest: {len(cases)}/{len(cases)} clauses behave in both directions")
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("require", help="refuse unless the latency precondition holds")
    r.add_argument("--baseline", default=None, help="throughput-baseline.json to condition against")
    r.add_argument("--model", action="append", default=[], help="floor entry to check (repeatable)")
    r.add_argument("--where", default="", help="label for the message, e.g. 'gate-h leg 4'")

    sub.add_parser("observe", help="print the observed mode, or `unknown`; always exits 0")
    sub.add_parser("selftest", help="every clause in both directions, no hardware")

    args = ap.parse_args(argv)

    if args.cmd == "selftest":
        return selftest()

    if args.cmd == "observe":
        try:
            print(observed_mode() or UNKNOWN)
        except PmodeRefusal:
            print(UNKNOWN)
        return 0

    try:
        mode, flags = check(args.baseline, args.model, args.where)
    except PmodeRefusal as exc:
        print(f"PMODE GUARD: REFUSED -- {exc}")
        return 1
    for line in flags:
        print(f"PMODE GUARD: WARNING -- {line}")
    print(f"PMODE GUARD: NPU power mode is `{mode}` -- the measurement condition holds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
