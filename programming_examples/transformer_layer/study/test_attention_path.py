# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check ``run_mode.ATTENTION_PATH_BY_MODE`` against what the modes actually import.

    python3 study/test_attention_path.py

That map is a claim about four files this module does not own, recorded into every
results row. If a mode moves its attention to the device and the map is not
updated, every row it writes afterwards carries the wrong value for the one
covariate that decides how its latency curve reads -- and nothing would notice,
because the field is free text as far as the CSV is concerned.

So the map is re-derived here from the pattern modules' own imports: a mode that
imports ``blocked_attention`` from ``pattern/blocked_attention.py`` runs attention
on the host. Parsing is ``ast`` only -- no toolchain, no device, and no importing
of the pattern modules themselves, which would need the builders.

FOOTGUN
    ``round_bf16`` also lives in ``pattern/blocked_attention.py`` and is a plain
    rounding helper. ``fused`` imports it and runs attention on the device, so a
    check written against the *module* name rather than the *symbol* would call
    ``fused`` a host-attention mode. The name-level check is the one that is
    wrong here, which is why the symbol list is what this compares.
"""

import ast
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import run_mode  # noqa: E402

_PATTERN = os.path.join(os.path.dirname(_HERE), "pattern")
_HOST_ATTENTION_SYMBOL = "blocked_attention"


def _imported_symbols(path: str) -> set[str]:
    tree = ast.parse(open(path, encoding="utf-8").read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith(
            "blocked_attention"
        ):
            names.update(alias.name for alias in node.names)
    return names


def _module_for(mode: str) -> str:
    path = os.path.join(_PATTERN, mode, f"{mode}.py")
    assert os.path.isfile(path), f"no pattern module for mode {mode!r} at {path}"
    return path


def test_every_mode_in_the_map_has_a_pattern_module():
    for mode in run_mode.ATTENTION_PATH_BY_MODE:
        _module_for(mode)


def test_the_map_matches_what_each_mode_imports():
    for mode, declared in sorted(run_mode.ATTENTION_PATH_BY_MODE.items()):
        symbols = _imported_symbols(_module_for(mode))
        derived = "host_torch" if _HOST_ATTENTION_SYMBOL in symbols else "device"
        assert declared == derived, (
            f"{mode} is declared {declared!r} but imports {sorted(symbols)} from "
            f"pattern/blocked_attention.py, which derives {derived!r}. Either the "
            f"mode moved its attention or the map is stale."
        )


def test_the_symbol_not_the_module_is_what_decides():
    """fused imports round_bf16 from that module and is still device attention."""
    symbols = _imported_symbols(_module_for("fused"))
    assert symbols, "fused does import from pattern/blocked_attention.py"
    assert _HOST_ATTENTION_SYMBOL not in symbols
    assert run_mode.ATTENTION_PATH_BY_MODE["fused"] == "device"


def test_attention_placement_is_no_longer_a_covariate():
    """`[2026-08-09]` All four modes are on the device. This is the END STATE.

    This assertion used to read ``values == {"device", "host_torch"}``, with the
    rationale that "the ladder's whole point is comparing the two placements; if
    this ever holds only one, the confound is gone or the map broke". The
    confound IS gone -- deliberately. `offload` moved on 2026-08-08 and
    `runlist` on 2026-08-09, both because the corrected taxonomy puts their
    attention on the device, and no mode is left on the host side.

    Inverting it rather than deleting it keeps a live claim, and the claim is
    now the one that matters: **the first sequence ladder's headline cannot be
    reproduced.** That result read its slope split off this covariate --
    host-attention modes at 1.23-1.27 against device-attention modes at
    1.03-1.17 -- and there is no longer a host-attention mode to produce one
    side of it. A future run that shows separated slopes is measuring something
    else.

    If a mode ever moves back to the host, this fails and the reader is sent
    here rather than to a stale comparison.
    """
    values = set(run_mode.ATTENTION_PATH_BY_MODE.values())
    assert values == {"device"}, (
        f"expected every mode on the device, got {sorted(values)}. If a mode "
        "moved back to host attention, the sequence ladder gains a covariate it "
        "has not had since 2026-08-09 and every cross-mode latency comparison "
        "recorded in between needs re-reading, not just this map updating"
    )


def test_the_declared_values_are_in_the_schema_domain():
    import schema

    for mode, path in run_mode.ATTENTION_PATH_BY_MODE.items():
        assert path in schema.ATTENTION_PATHS, f"{mode}: {path!r} is not a valid value"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"attention-path tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
