# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The J7b structural check: the ring forms, and forms off the packet path.

CONTRACT
    Builds ``build_ffn_accum_module`` at EVERY shape the catalogue claims for
    ``ffn_accum`` (read from ``opcheck_specs.SPECS``) and checks, host-only:

    1. THE HOIST. Lowered through ``air-dependency`` alone, the herd's K loop
       carries 4 data-movement ops (A get, B get, C fetch, C store); with
       ``air-hoist-dma-in-accum-pattern`` appended -- the aircc pipeline's
       second pass -- it must carry 2. That 4 -> 2 IS the phase: a loop that
       keeps its C round trip computes identical numbers one DDR trip per K
       step slower, invisible to every numerical arm.
    2. ZERO PACKET-TYPED CHANNELS after the prefix ending at
       ``air-dma-to-channel``, J7a's column rule. Three per-core input
       streams packet-multiplex EVERY input (measured; see the builder), and
       the packet path is silently wrong past one trip pre-H9 and BD-starved
       post-H9.

    Liveness, so neither count can pass vacuously: the lowered IR must hold
    no residual ``air.dma_memcpy_nd`` and must still declare the A|B feed
    channel by name.

    The counting mirrors ``agents/probes/probe_accum_hoist.py`` (first
    ``scf.for`` in the module), which is also how the driver reads aircc's
    ``--debug-ir`` dump; the builder emits the feed loop after the herd so
    that first loop IS the K loop.
"""

import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJ_ROOT = _HERE.parent  # programming_examples/
for _p in (str(_PROJ_ROOT), str(_PROJ_ROOT / "llms"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import air.ir  # noqa: E402
import air.passmanager  # noqa: E402

from builders.ffn_accum import CHANNEL_FEED, build_ffn_accum_module  # noqa: E402
from opcheck_specs import SPECS  # noqa: E402

# tools/aircc/aircc.cpp::buildOptimizationPipeline prefixes. The hoist pass
# runs second in the real pipeline; air-dma-to-channel decides packet typing.
_DEP = "builtin.module(air-dependency)"
_HOIST = "builtin.module(air-dependency,air-hoist-dma-in-accum-pattern)"
_CHANNEL = (
    "builtin.module("
    + ",".join(
        [
            "air-dependency",
            "air-hoist-dma-in-accum-pattern",
            "air-broadcast-detection",
            "air-specialize-dma-broadcast",
            "air-dma-to-channel",
            "canonicalize",
            "cse",
        ]
    )
    + ")"
)


def _lower(src, pipeline):
    context = air.ir.Context()
    module = air.ir.Module.parse(src, context=context)
    air.passmanager.PassManager.parse(pipeline, context=context).run(module.operation)
    return str(module)


def _in_first_loop(text):
    """Data-movement ops inside the first scf.for -- probe_accum_hoist's count."""
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if "scf.for" in l), None)
    if start is None:
        return None
    depth, body = 0, []
    for line in lines[start:]:
        body.append(line)
        depth += line.count("{") - line.count("}")
        if depth <= 0 and len(body) > 1:
            break
    return len(
        re.findall(r"air\.dma_memcpy_nd|air\.channel\.(put|get)", "\n".join(body))
    )


def check_shape(shape_key, seq_len, ffn_dim, emb_dim):
    """One shape's verdict. Returns the list of problems, empty on pass."""
    src = str(build_ffn_accum_module(seq_len, ffn_dim, emb_dim))
    before = _in_first_loop(_lower(src, _DEP))
    after = _in_first_loop(_lower(src, _HOIST))
    lowered = _lower(src, _CHANNEL)
    n_packet = lowered.count("npu_dma_packet")
    n_dma = lowered.count("air.dma_memcpy_nd")
    label = f"ffn_accum [{shape_key}]"

    problems = []
    if before != 4 or after != 2:
        problems.append(
            f"{label}: K-loop data movement {before} -> {after}, need 4 -> 2 "
            "-- the accumulator round trip did not hoist; the ring is not "
            "forming and every numerical check will still pass"
        )
    if n_dma:
        problems.append(
            f"{label}: {n_dma} air.dma_memcpy_nd left after air-dma-to-channel "
            "-- the lowering did not run, so the packet count proves nothing"
        )
    if f"@{CHANNEL_FEED}" not in lowered:
        problems.append(
            f"{label}: feed channel @{CHANNEL_FEED} missing from the lowered "
            "IR -- this is not the A|B-fed accumulator design"
        )
    if n_packet:
        problems.append(
            f"{label}: {n_packet} packet-typed channels -- a third L3-facing "
            "stream is back in the per-column budget; see builders/ffn_accum.py"
        )

    verdict = "FAIL" if problems else "PASS"
    print(
        f"[ffn-accum-structure] {label}: {verdict} (K-loop {before} -> {after}, "
        f"{n_packet} packet-typed channels)"
    )
    return problems


def main():
    shapes = [s for s in SPECS if s["operator"] == "ffn_accum"]
    if not shapes:
        print("[ffn-accum-structure] FAIL: no ffn_accum shapes in SPECS")
        return 1

    problems = []
    for spec in shapes:
        shape = spec["shape"]
        problems += check_shape(
            spec["shape_key"], shape["seq_len"], shape["ffn_dim"], shape["emb_dim"]
        )

    if problems:
        print("[ffn-accum-structure] FAIL")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print("[ffn-accum-structure] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
