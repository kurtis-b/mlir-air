# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The J7a structural check: zero packet-typed channels in the lowered pipeline.

CONTRACT
    Builds ``build_norm_tail_module`` at EVERY shape the catalogue claims for
    ``norm_tail`` (read from ``opcheck_specs.SPECS``, so a new claimed shape
    joins this check without being remembered), lowers each through the aircc
    pipeline prefix that ends at ``air-dma-to-channel`` -- the pass that
    decides packet typing -- and exits 0 if and only if every lowered module
    carries ZERO ``npu_dma_packet`` channels. No NPU, no Peano: the check is
    compile-time and structural.

WHY THIS IS A GATE AND NOT A NOTE
    A column has two shim MM2S channels and the budget is per column across
    the whole segment. The pipeline meets it exactly (packed x|residual in,
    gamma in), so every channel is circuit-routed. A later edit that adds a
    third L3-facing stream -- a second input, a debug drain -- pushes the
    per-column pressure over the limit and ``air-dma-to-channel`` silently
    upgrades EVERY input channel to the packet queue, which is correct at one
    trip and wrong past it (the defect J1 lost a night to). The numerical
    check cannot see that: it passes at one trip. Counting ``npu_dma_packet``
    in the lowered IR is the one cheap place the regression is visible.

WHY THE CHECK IS NOT ONLY "count == 0"
    Zero is also what a module that never lowered would score, so the count
    alone can pass vacuously. Two liveness assertions close that hole: the
    lowered IR must contain no residual ``air.dma_memcpy_nd`` (the pass
    actually converted the L3-facing DMAs into channels), and the two declared
    L1->L1 channels must still be present by name (the module is the pipeline,
    not a stub).

FOOTGUNS
    - The pipeline below mirrors ``buildOptimizationPipeline`` in
      ``tools/aircc/aircc.cpp`` UP TO ``air-dma-to-channel``, including the
      broadcast passes aircc runs by default. Trimming it to just the one pass
      would measure a module aircc never produces.
    - Importing ``opcheck_specs`` pulls in every preparer (torch included);
      that is the price of reading the claimed shapes from the one catalogue
      instead of copying them here to drift.
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJ_ROOT = _HERE.parent  # programming_examples/
for _p in (str(_PROJ_ROOT), str(_PROJ_ROOT / "llms"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import air.ir  # noqa: E402
import air.passmanager  # noqa: E402

from builders.norm_tail import (  # noqa: E402
    CHANNEL_A2B,
    CHANNEL_B2C,
    build_norm_tail_module,
)
from opcheck_specs import SPECS  # noqa: E402

# tools/aircc/aircc.cpp::buildOptimizationPipeline, up to and including the
# pass that decides packet typing. air-dma-to-channel converts the herd-side
# L3-facing DMAs into channels and upgrades them to channel_type
# "npu_dma_packet" when the per-column shim pressure exceeds the limit.
_PIPELINE = (
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


def lowered_norm_tail_ir(rows, cols):
    """Build the pipeline at (rows, cols) and lower it to the channel form."""
    built = build_norm_tail_module(rows, cols)
    context = air.ir.Context()
    module = air.ir.Module.parse(str(built), context=context)
    pm = air.passmanager.PassManager.parse(_PIPELINE, context=context)
    pm.run(module.operation)
    return str(module)


def check_shape(shape_key, rows, cols):
    """One shape's verdict. Returns the list of problems, empty on pass."""
    ir = lowered_norm_tail_ir(rows, cols)
    n_packet = ir.count("npu_dma_packet")
    n_dma = ir.count("air.dma_memcpy_nd")
    n_channels = ir.count("air.channel ")
    label = f"norm_tail [{shape_key}]"

    problems = []
    if n_dma:
        problems.append(
            f"{label}: {n_dma} air.dma_memcpy_nd left after air-dma-to-channel "
            "-- the lowering did not run, so the packet count proves nothing"
        )
    for name in (CHANNEL_A2B, CHANNEL_B2C):
        if f"@{name}" not in ir:
            problems.append(
                f"{label}: declared L1->L1 channel @{name} missing from the "
                "lowered IR -- this is not the three-herd pipeline"
            )
    if n_packet:
        problems.append(
            f"{label}: {n_packet} packet-typed channels -- a third L3-facing "
            "stream has entered the per-column budget; see builders/norm_tail.py"
        )

    verdict = "FAIL" if problems else "PASS"
    print(
        f"[norm-tail-structure] {label}: {verdict} ({n_packet} packet-typed "
        f"channels over {n_channels} channel declarations)"
    )
    return problems


def main():
    shapes = [s for s in SPECS if s["operator"] == "norm_tail"]
    if not shapes:
        print("[norm-tail-structure] FAIL: no norm_tail shapes in SPECS")
        return 1

    problems = []
    for spec in shapes:
        problems += check_shape(
            spec["shape_key"], spec["shape"]["rows"], spec["shape"]["cols"]
        )

    if problems:
        print("[norm-tail-structure] FAIL")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print("[norm-tail-structure] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
