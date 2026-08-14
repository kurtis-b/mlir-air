# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only checks on the LayerNorm row derivation, and on the width wall it closes.

    python3 builders/test_layer_norm_rows.py

WHAT THIS IS ABOUT
    `block.norm_rows` is derived per width and says so: "the cap moves with
    ``emb_dim`` -- 104 rows at 768, 80 at 1024 -- and a row count that happened
    to fit at one width is a placement failure at the next".
    ``build_layer_norm_module``'s ``rows_per_call`` was the other half of the
    same sum and was NOT derived -- it defaulted to 8, which with ``herd_x = 8``
    silently requires ``64 | rows``.

    At ``emb_dim 1024`` ``norm_rows`` derives 32, so `runlist` refused to build
    at EVERY sequence length: **0 of 9 ladder points against 9 of 9 at 512 and
    768** (doc 50 section 7). Each core owns 4 of those 32 rows, so 4 was legal
    all along and 8 was never the only choice.

THE CLAIM THAT MATTERS IS THE ONE ABOUT WHAT DID *NOT* CHANGE
    Doc 50 left this unfixed because "every plausible repair changes what a
    `runlist` row means" -- a fair worry, since a mode's row is only comparable
    across widths if the dispatch structure is the same. The derivation is
    bounded ABOVE by the historical default, so wherever 8 was legal it still
    picks 8 and the emitted IR is byte-identical. That is asserted here at
    concrete shapes rather than argued, which is what makes the repair safe to
    land beside gated designs.
"""

import hashlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from builders.block import norm_rows  # noqa: E402
from builders.layer_norm import (  # noqa: E402
    DEFAULT_ROWS_PER_CALL,
    L1_BYTES,
    build_layer_norm_module,
    derive_rows_per_call,
)

#: The ladder, and the widths `cases.py` declares.
LADDER = (512, 1024, 2048, 4096)
WIDTHS = (512, 768, 1024)


def _sha(module):
    return hashlib.sha256(str(module).encode()).hexdigest()


def test_the_derivation_never_exceeds_the_historical_default():
    """The ceiling is what makes this safe: it may only ever go DOWN from 8.

    A derivation free to go UP could change a design that already gates, and
    the whole argument for landing this beside `runlist` and `coarse` is that
    it cannot.
    """
    for emb in WIDTHS:
        for seq in LADDER:
            rows = norm_rows(seq, emb)
            assert derive_rows_per_call(rows, emb) <= DEFAULT_ROWS_PER_CALL


def test_the_derivation_returns_8_wherever_8_was_legal():
    """The other half of the ceiling property: it does not go down needlessly."""
    checked = 0
    for emb in WIDTHS:
        for seq in LADDER:
            rows = norm_rows(seq, emb)
            if rows % (8 * DEFAULT_ROWS_PER_CALL):
                continue
            assert derive_rows_per_call(rows, emb) == DEFAULT_ROWS_PER_CALL
            checked += 1
    assert checked, "no shape exercised the 8-is-legal branch; test is vacuous"


def test_the_emitted_IR_is_byte_identical_wherever_8_was_legal():
    """The claim doc 50 was right to want: no shipped design moves.

    Byte equality over the built module, not a structural argument about it --
    [23 section 5] records a structural check missing the very claim its phase
    rested on.
    """
    checked = 0
    for rows, cols in ((64, 768), (512, 768), (1024, 768), (128, 1024), (512, 1024)):
        if rows % (8 * DEFAULT_ROWS_PER_CALL):
            continue
        old = _sha(build_layer_norm_module(rows, cols, rows_per_call=8))
        new = _sha(build_layer_norm_module(rows, cols))
        assert old == new, f"rows={rows} cols={cols} moved: {old} != {new}"
        checked += 1
    assert checked >= 4, f"only {checked} shapes compared; test is too weak"


def test_a_derived_value_below_8_really_does_change_the_IR():
    """The discrimination control for the test above.

    If the module were insensitive to `rows_per_call` the byte-identity check
    would pass no matter what the derivation returned, and would prove nothing.
    """
    a = _sha(build_layer_norm_module(64, 768, rows_per_call=8))
    b = _sha(build_layer_norm_module(64, 768, rows_per_call=4))
    assert a != b, "rows_per_call does not affect the IR; byte-identity is vacuous"


def test_the_width_wall_at_emb_1024_is_gone():
    """0 of 9 ladder points -> every point builds. Doc 50 section 7's wall."""
    for seq in LADDER:
        rows = norm_rows(seq, 1024)
        assert rows == 32, f"seq {seq}: norm_rows moved to {rows}, expected 32"
        module = build_layer_norm_module(rows, 1024)
        assert "layer_norm_multi_row" in str(module)


def test_the_wall_was_REAL_and_the_fix_is_the_derivation_not_a_loosened_check():
    """An explicit rows_per_call=8 must still raise at 32 rows.

    This is what separates "derive the parameter" from "delete the constraint".
    The divisibility rule is a real property of the herd walk; if this stopped
    raising, the module would emit a herd whose cores step past their own rows.
    """
    try:
        build_layer_norm_module(32, 1024, rows_per_call=8)
    except ValueError as exc:
        assert "divisible" in str(exc)
        return
    raise AssertionError("rows=32 with an explicit rows_per_call=8 must still raise")


def test_the_L1_budget_bounds_the_derivation():
    """Divisibility alone is not legality -- the ping-ponged tile must fit.

    At cols 4096 a rows_per_call of 8 needs 2*8*4096*2 = 128 KiB against a
    64 KiB L1, so the derivation must come down even though 8 divides cleanly.
    """
    rows, cols = 512, 4096
    assert rows % (8 * DEFAULT_ROWS_PER_CALL) == 0, "fixture must divide by 64"
    got = derive_rows_per_call(rows, cols)
    assert 2 * got * cols * 2 <= L1_BYTES
    assert got < DEFAULT_ROWS_PER_CALL, f"budget ignored: derived {got}"


def test_an_impossible_shape_is_refused_by_message():
    """No legal value must raise rather than return a shape that miscompiles."""
    try:
        derive_rows_per_call(8, 1 << 20)
    except ValueError as exc:
        assert "does not fit" in str(exc)
        return
    raise AssertionError("a shape with no legal rows_per_call must raise")


def test_every_declared_width_resolves_a_buildable_norm_shape():
    """Phase G reachability, as a property rather than a walk.

    This is the assertion doc 50's walk had to spend a devq job to discover.
    It is host-only and takes milliseconds, so a future width that cannot norm
    fails here instead of 0-of-9 rungs into a ladder.
    """
    for emb in WIDTHS:
        for seq in LADDER:
            rows = norm_rows(seq, emb)
            n = derive_rows_per_call(rows, emb)
            assert rows % (8 * n) == 0, f"emb {emb} seq {seq}: rows {rows}, n {n}"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"layer-norm row tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
