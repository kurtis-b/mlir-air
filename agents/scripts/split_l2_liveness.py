#!/usr/bin/env python3
"""Report producer/consumer counts for the L2 buffers in post-pass AIR IR.

Written for the `air-split-l2-memref` multi-row-herd investigation (goal 4d):
after the split, a buffer that is filled by an `air.channel.get` and then only
deallocated has had its contents dropped, which is exactly how the S2MM defect
behind the LM-head port blocker shows up. The check is general, though -- any
split that loses a buffer looks the same.

Usage:
    air-opt <module.mlir> --air-split-l2-memref=... | split_l2_liveness.py

Exits 1 if any L2 buffer is written but never drained, so it can be used as a
gate rather than read by eye.
"""

import re
import sys

# `%results_12 = air.execute -> (memref<4xbf16, 1 : i32>)`: an L2 allocation.
# Memory space 1 is L2; L1 (space 2) buffers are the herd's own and not at
# issue here.
ALLOC = re.compile(
    r"(%results(?:_\d+)?) = air\.execute -> \(memref<([0-9x]+)x(\w+), 1 : i32>\)"
)


def liveness(ir):
    """Yield (name, shape, dtype, n_gets, n_puts) for every L2 buffer."""
    for m in ALLOC.finditer(ir):
        name, shape, dtype = m.group(1), m.group(2), m.group(3)
        # Match uses as a transfer's memref operand -- `<name>[` -- so that a
        # mention inside a type or a dealloc is not counted as data movement.
        use = re.escape(name) + r"\["
        gets = len(re.findall(r"air\.channel\.get[^\n]*" + use, ir))
        puts = len(re.findall(r"air\.channel\.put[^\n]*" + use, ir))
        yield name, shape, dtype, gets, puts


def main():
    ir = sys.stdin.read()
    rows = list(liveness(ir))
    if not rows:
        print("no L2 (memory space 1) allocations found", file=sys.stderr)
        return 1

    dropped, unfilled = [], []
    print(f"{'buffer':<16}{'shape':<20}{'gets':>6}{'puts':>6}  note")
    for name, shape, dtype, gets, puts in rows:
        note = ""
        # Filled but never drained: whatever was written into it is lost.
        if gets and not puts:
            note = "<-- NEVER DRAINED"
            dropped.append(name)
        # Drained but never filled: whatever is read out of it is undefined.
        # Only meaningful alongside siblings of the same shape that ARE filled,
        # so it is reported but does not by itself set the exit status.
        elif puts and not gets:
            note = "<-- never filled"
            unfilled.append(name)
        print(f"{name:<16}{shape + 'x' + dtype:<20}{gets:>6}{puts:>6}  {note}")

    if unfilled:
        print(
            f"\n{len(unfilled)} buffer(s) drained but never filled: "
            f"{', '.join(unfilled)}",
            file=sys.stderr,
        )
    if dropped:
        print(
            f"{len(dropped)} buffer(s) written but never drained: "
            f"{', '.join(dropped)}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
