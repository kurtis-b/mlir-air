# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The device softmax's row-width wall, and the derivation that names it.

    python3 builders/test_softmax_rows.py

Host-only: builds ``air.ir.Module``s and hashes them, never reaches a device.
Same shape as ``test_layer_norm_rows.py`` because it is the same class of
defect found the same way: a ``rows_per_call`` that was a CONSTANT sized at
one width and silently illegal at the next. `runlist` pinned the softmax's
at 2 (48 KiB of L1 at cols 4096); the first ``full`` profile (devq 427) took
it to 8192, where 2 rows are 96 KiB on a 64 KiB tile, and aircc failed with an
EMPTY error body -- twice, 88 s and 468 s in. The repair derives the value
bounded above by the historical constant (so nothing at <= 4096 moves, pinned
here as byte equality with its discrimination control), gives 8192 the one row
that fits, and turns 16384 -- where even one row is 96 KiB -- into a named
ValueError at config time.
"""

import hashlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, os.path.join(os.path.dirname(_ROOT), "llms"), _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from builders.softmax import (  # noqa: E402
    L1_BYTES,
    build_softmax_module,
    derive_rows_per_call,
    softmax_l1_bytes,
)

#: `runlist`'s historical constant, and the lengths the full profile walks.
RUNLIST_CEILING = 2
LADDER = (512, 1024, 2048, 4096, 8192, 16384)


def _sha(module):
    return hashlib.sha256(str(module).encode()).hexdigest()


def _raises(match, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError as e:
        assert match in str(e), f"expected {match!r} in {str(e)!r}"
        return
    raise AssertionError(f"expected ValueError containing {match!r}")


def test_the_derivation_never_exceeds_the_ceiling():
    for seq in LADDER[:-1]:
        assert derive_rows_per_call(seq, seq, ceiling=RUNLIST_CEILING) <= RUNLIST_CEILING


def test_the_derivation_returns_the_ceiling_wherever_it_was_legal():
    """Every length the mode has ever gated at keeps its value."""
    for seq in (512, 1024, 2048, 4096):
        assert derive_rows_per_call(seq, seq, ceiling=RUNLIST_CEILING) == 2, seq
    # And the standalone softmax gate's three shapes keep theirs.
    assert derive_rows_per_call(512, 512, ceiling=8) == 8
    assert derive_rows_per_call(4096, 768, ceiling=8) == 8
    assert derive_rows_per_call(64, 4096, ceiling=2) == 2


def test_the_emitted_IR_is_byte_identical_wherever_2_was_legal():
    """No gated `runlist` softmax moves: byte equality, not an argument."""
    checked = 0
    for seq in (512, 1024, 2048, 4096):
        old = _sha(build_softmax_module(seq, seq, rows_per_call=RUNLIST_CEILING))
        new = _sha(
            build_softmax_module(
                seq, seq, rows_per_call=derive_rows_per_call(seq, seq, ceiling=RUNLIST_CEILING)
            )
        )
        assert old == new, f"seq={seq} moved: {old} != {new}"
        checked += 1
    assert checked == 4


def test_a_derived_value_below_2_really_does_change_the_IR():
    """Discrimination control: the module must be sensitive to rows_per_call,
    or the byte-identity clause above is vacuous."""
    a = _sha(build_softmax_module(4096, 4096, rows_per_call=2))
    b = _sha(build_softmax_module(4096, 4096, rows_per_call=1))
    assert a != b


def test_8192_derives_one_row_and_builds():
    assert derive_rows_per_call(8192, 8192, ceiling=RUNLIST_CEILING) == 1
    assert softmax_l1_bytes(1, 8192) <= L1_BYTES < softmax_l1_bytes(2, 8192)
    build_softmax_module(8192, 8192, rows_per_call=1)  # must not raise


def test_the_wall_was_REAL_and_the_fix_is_the_derivation_not_a_loosened_check():
    """An EXPLICIT rows_per_call of 2 at 8192 still refuses -- now by name, at
    build time, instead of inside aircc with the message lost."""
    _raises("over the 65536-byte tile", build_softmax_module, 8192, 8192, rows_per_call=2)


def test_16384_is_refused_by_name_with_the_bytes():
    """Even one 16384-wide row is three 32 KiB tiles: this design's wall."""
    _raises("even rows_per_call=1 needs 98312 B", derive_rows_per_call, 16384, 16384, ceiling=2)
    _raises("over the 65536-byte tile", build_softmax_module, 16384, 16384, rows_per_call=1)


def test_the_L1_count_matches_the_allocations():
    """softmax_l1_bytes counts three row tiles + the scale band in bf16 --
    the four AllocOps in build_softmax_module and nothing else."""
    assert softmax_l1_bytes(2, 4096) == (3 * 2 * 4096 + 4 * 2) * 2 == 49168


def test_runlist_config_carries_the_derived_value_and_refuses_16384():
    from pattern.runlist.runlist import SOFTMAX_ROWS_PER_CALL, runlist_config

    assert SOFTMAX_ROWS_PER_CALL == RUNLIST_CEILING
    assert runlist_config(4096, 768, 3072, 12, 64)["softmax_rows_per_call"] == 2
    assert runlist_config(8192, 768, 3072, 12, 64)["softmax_rows_per_call"] == 1
    _raises("even rows_per_call=1 needs", runlist_config, 16384, 768, 3072, 12, 64)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"softmax row tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
