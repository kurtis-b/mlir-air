# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Does the PLANE-MAJOR norm tail survive the same structural rules as the packed one?

    python3 agents/probes/probe_norm_tail_plane_major.py [--rows 1024] [--cols 768]

WHY THIS EXISTS
    Wiring `norm_tail` into `pattern/fused` needs the packed buffer's plane 0 to
    be written by a DEVICE producer -- `attn_out` from a GEMM in a previous ELF,
    and `ffn_out` from the FFN. The shipped row-interleaved layout cannot accept
    that without teaching the SHARED GEMM builder to write strided, and packing
    on the host reintroduces the round trip `fused` exists to remove.

    Plane-major has neither problem: plane 0 is a contiguous [rows, cols] block
    at offset 0, so an ordinary contiguous producer writes it unchanged. What is
    not established is whether it still satisfies J7a's column rule -- the layout
    was rejected on the shim BD stride cap at 4096 rows and never carried through
    the structural checks below the cap.

WHAT IT MEASURES
    Exactly `norm_tail_structure.check_shape`'s five checks, reused rather than
    restated, with the builder swapped for its plane-major form: zero packet
    channels in every dump, three compiler-placed herds, 2*herd_x core->core
    flows, at most 2 shim-inbound per column, and liveness. The column budget is
    the one that decides this: plane-major still fetches both operands in ONE
    DMA, so the count should be unchanged -- but "should" is what the probe is
    for, and the 3-D band fetch is a different BD program than the 1-D one.

    A PASS here does not license 2048 and up; the builder refuses those on the
    stride cap by precondition. It licenses the 64..1024 rungs.
"""

import argparse
import os
import sys

_TL = (
    "/home/cj/mlir-air/.claude/worktrees/phase-f/programming_examples/transformer_layer"
)
if _TL not in sys.path:
    sys.path.insert(0, _TL)
os.chdir(_TL)  # aircc writes its debug dumps relative to cwd

import builders.norm_tail as nt  # noqa: E402
import norm_tail_structure as nts  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1024)
    ap.add_argument("--cols", type=int, default=768)
    args = ap.parse_args()

    original = nt.build_norm_tail_module
    # check_shape calls build_norm_tail_module(rows, cols); swap in the layout
    # under test so all five checks run against it verbatim.
    nts.build_norm_tail_module = lambda rows, cols: original(
        rows, cols, plane_major=True
    )

    key = f"{args.rows}x{args.cols}_plane_major"
    print(f"[probe] {key}: running norm_tail_structure's five checks", flush=True)
    problems = nts.check_shape(key, args.rows, args.cols)

    for p in problems:
        print(f"[probe] {p}")
    if problems:
        print(f"[probe] FAIL ({len(problems)} problem(s)) -- plane-major does NOT")
        print("[probe] satisfy J7a's column rule; fused needs the strided-GEMM path")
        return 1
    print("[probe] PASS -- plane-major satisfies every rule the packed layout does,")
    print(f"[probe] so fused can consume a contiguous producer at rows <= {args.rows}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
