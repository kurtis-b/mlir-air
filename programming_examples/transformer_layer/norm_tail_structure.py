# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The J7a structural check: the pipeline routes as three herds on L1->L1 edges.

CONTRACT
    Compiles ``build_norm_tail_module`` at EVERY shape the catalogue claims for
    ``norm_tail`` (read from ``opcheck_specs.SPECS``, so a new claimed shape
    joins this check without being remembered) through
    ``XRTBackend(debug_ir=True)`` -- the production aircc binary -- and reads
    the per-pass dumps aircc writes to ``air_project/debug_ir/``.

    ``[2026-08-07]`` This replaces a hand-built ``air-opt`` pass list that
    stopped at ``air-dma-to-channel``. That was sound for what it claimed --
    packet typing IS decided there -- but it could not prove final routing or
    live L1-backed endpoints, which is doc 23's open item 5, and the same
    weakness the J7b review found in ``ffn_accum_structure.py``. A pass list
    answers "what does this pass do", not "what does the toolchain do", and
    the two have diverged on this port before (doc 22, "at which altitude").

    STILL NO NPU AND NO PEANO, despite reading the ROUTED design. aiecc writes
    every MLIR pass dump before it compiles core ELFs, so the compile failing
    for want of a kernel object costs nothing here: all 59 dumps land, the last
    of them fully routed (measured -- 40 ``aie.flow``, 32 ``aie.tile``, without
    ``PEANO_INSTALL_DIR`` set). The check therefore keeps the cheap, hermetic
    contract the old one had while asserting far more. Building the kernels
    first would ALSO work and is what the numeric arm does, but it would put a
    Peano dependency on an arm that does not need one.

    Five checks:

    1. ZERO PACKET-TYPED CHANNELS IN EVERY DUMP, J7a's column rule. Three
       L3-facing streams per column packet-multiplex EVERY input (measured;
       see builders/norm_tail.py), and the packet path is silently wrong past
       one trip pre-H9 and BD-starved post-H9. Checking every dump, not just
       the one after ``air-dma-to-channel``, catches a later pass introducing
       one.
    2. THREE HERDS, COMPILER-PLACED. The routed device must carry 3 * herd_x
       core tiles spread over exactly THREE distinct rows, herd_x per row.
       This is the phase's claim that ``air-place-herds`` placed a three-stage
       pipeline; the builder declares no tile coordinate anywhere, so a
       regression to one herd, or to a placement that collapses two stages
       onto a row, fails here.
    3. THE STAGE EDGES ARE L1->L1, in the final routed IR. Exactly
       2 * herd_x ``aie.flow`` ops must run core tile -> core tile: the
       declared ``norm_tail_a2b`` and ``norm_tail_b2c`` edges, one per column
       each. THIS IS WHAT THE OLD CHECK COULD NOT DO. Keeping the
       intermediates off L3 is the entire point of the phase -- it is why
       ``fused``'s decomposed tail measures worse -- and an edge that
       silently round-trips through L3 or a memtile would still show zero
       packet-typed channels and still pass every numeric arm.
    4. THE COLUMN BUDGET, MEASURED ON THE ROUTED DESIGN. Shim-facing flows
       must be at most 2 inbound per column. The old check inferred this from
       packet typing, which is the compiler's REACTION to exceeding it; this
       counts the thing the rule is actually about. Measured: 16 shim->core
       over 8 columns, exactly 2 each -- the packed x|residual fetch and
       gamma.
    5. LIVENESS, so no count passes vacuously. The dump after
       ``air-dma-to-channel`` must exist, hold no residual
       ``air.dma_memcpy_nd``, and still declare both L1->L1 channels by name;
       the final dump must hold no ``air.channel`` at all (everything lowered)
       and at least one ``aie.flow``.

FOOTGUNS
    - The compile is EXPECTED to fail, at the core-ELF step, for want of the
      stage kernel objects. That is not tolerated blindly: check 5 requires the
      final dump to exist and to carry routing, so a compile that died EARLIER
      -- before the passes these checks read -- fails rather than passing with
      nothing measured.
    - Tile rows are the placement, not a naming convention: row 0 is shim,
      row 1 is memtile, rows >= 2 are cores. Do not assert on tile SSA names.
    - Verifying this script standalone is NOT verifying its gate. It is run
      through a lit arm that FileChecks the verdict line, so changing what that
      line says breaks the suite even when every check passes -- which is
      exactly what happened when these checks were added `[2026-08-07]`.
"""

import glob
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJ_ROOT = _HERE.parent  # programming_examples/
for _p in (str(_PROJ_ROOT), str(_PROJ_ROOT / "llms"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from air.backend.xrt import XRTBackend  # noqa: E402

from builders.norm_tail import (  # noqa: E402
    CHANNEL_A2B,
    CHANNEL_B2C,
    build_norm_tail_module,
)
from opcheck_specs import SPECS  # noqa: E402

# The pipeline is three stages wide; the builder names them but places nothing.
N_STAGES = 3

# build_norm_tail_module's herd width. The catalogue's shapes all run it.
HERD_X = 8

_CHANNEL_DUMP_SUFFIX = "_after_air-dma-to-channel.mlir"

_TILE_RE = re.compile(r"%(\S+) = aie\.tile\((\d+),\s*(\d+)\)")
_FLOW_RE = re.compile(
    r"aie\.flow\(%(\S+),\s*\w+\s*:\s*\d+,\s*%(\S+),\s*\w+\s*:\s*\d+\)"
)


def _aircc_debug_dumps(module):
    """The production pipeline's per-pass dumps for ``module``.

    No kernel objects are built, so the compile dies at the core-ELF step --
    after the MLIR pipeline has run and dumped everything, including the routed
    form. Returns the ordered ``(name, text)`` list plus the compile error, and
    lets the caller decide by which dumps exist.
    """
    prev_cwd = os.getcwd()
    error = None
    with tempfile.TemporaryDirectory(prefix="norm-tail-structure-") as work:
        os.chdir(work)
        try:
            backend = XRTBackend(
                omit_while_true_loop=False,
                output_format="xclbin",
                instance_name="norm_tail",
                target_device="npu2",
                debug_ir=True,
            )
            try:
                backend.compile(module)
            except Exception as exc:
                error = f"{type(exc).__name__}: {str(exc)[-200:]}"
            finally:
                try:
                    backend.unload()
                except Exception:
                    pass
            dumps = [
                (os.path.basename(p), Path(p).read_text())
                for p in sorted(glob.glob("air_project/debug_ir/pass_*.mlir"))
            ]
        finally:
            os.chdir(prev_cwd)
    return dumps, error


def _tile_rows(text):
    """SSA name -> tile row. Row 0 is shim, 1 is memtile, >= 2 is a core."""
    return {m[0]: int(m[2]) for m in _TILE_RE.findall(text)}


def _tile_cols(text):
    return {m[0]: int(m[1]) for m in _TILE_RE.findall(text)}


def check_shape(shape_key, rows, cols):
    """One shape's verdict. Returns the list of problems, empty on pass."""
    label = f"norm_tail [{shape_key}]"
    module = build_norm_tail_module(rows, cols)
    dumps, compile_error = _aircc_debug_dumps(module)

    if not dumps:
        return [
            f"{label}: aircc wrote no debug dumps -- nothing structural is "
            f"measured ({compile_error or 'compile reported success'})"
        ]

    problems = []

    # 1. No packet-typed channel anywhere in the pipeline.
    packet_dumps = [n for n, text in dumps if "npu_dma_packet" in text]
    if packet_dumps:
        problems.append(
            f"{label}: packet-typed channels appear in {len(packet_dumps)} dump(s), "
            f"first {packet_dumps[0]} -- a third L3-facing stream has entered the "
            "per-column budget; see builders/norm_tail.py"
        )

    # 5a. Liveness at the channel-conversion altitude.
    chan = next((t for n, t in dumps if n.endswith(_CHANNEL_DUMP_SUFFIX)), None)
    if chan is None:
        problems.append(
            f"{label}: no air-dma-to-channel dump -- the compile stopped before "
            f"the pass that decides packet typing ({compile_error or 'no error'})"
        )
    else:
        if "air.dma_memcpy_nd" in chan:
            n_dma = chan.count("air.dma_memcpy_nd")
            problems.append(
                f"{label}: {n_dma} air.dma_memcpy_nd left after air-dma-to-channel "
                "-- the lowering did not run, so the packet count proves nothing"
            )
        for name in (CHANNEL_A2B, CHANNEL_B2C):
            if f"@{name}" not in chan:
                problems.append(
                    f"{label}: declared L1->L1 channel @{name} missing at "
                    "air-dma-to-channel -- this is not the three-herd pipeline"
                )

    # 5b. Liveness at the end: everything lowered, and routing exists.
    final_name, final = dumps[-1]
    if "air.channel" in final:
        problems.append(
            f"{label}: air.channel still present in {final_name} -- the design did "
            "not lower to AIE routing, so the flow checks below prove nothing"
        )
    flows = _FLOW_RE.findall(final)
    if not flows:
        problems.append(
            f"{label}: no aie.flow in {final_name} -- nothing was routed "
            f"({compile_error or 'compile reported success'})"
        )
        return problems

    rows_by_tile = _tile_rows(final)
    cols_by_tile = _tile_cols(final)

    # 2. Three herds, compiler-placed, herd_x wide each.
    core_rows = Counter(r for r in rows_by_tile.values() if r >= 2)
    if len(core_rows) != N_STAGES or any(n != HERD_X for n in core_rows.values()):
        problems.append(
            f"{label}: expected {N_STAGES} herd rows of {HERD_X} core tiles, got "
            f"{dict(sorted(core_rows.items()))} -- the three-stage placement "
            "air-place-herds derived is not what shipped"
        )

    def kind(tile):
        row = rows_by_tile.get(tile)
        if row is None:
            return None
        return "shim" if row == 0 else "memtile" if row == 1 else "core"

    edges = Counter((kind(s), kind(d)) for s, d in flows)

    # 3. The stage edges stay on chip, core tile to core tile.
    expected_l1 = 2 * HERD_X
    n_core_core = edges.get(("core", "core"), 0)
    if n_core_core != expected_l1:
        problems.append(
            f"{label}: {n_core_core} core->core flows, expected {expected_l1} "
            f"({CHANNEL_A2B} and {CHANNEL_B2C}, one per column each) -- a stage "
            "edge is not L1->L1, which is the whole point of the pipeline and is "
            "invisible to every numeric arm"
        )

    # 4. The column budget, on the routed design rather than via packet typing.
    inbound = Counter(
        cols_by_tile.get(d)
        for s, d in flows
        if kind(s) == "shim" and kind(d) == "core" and d in cols_by_tile
    )
    over = {c: n for c, n in inbound.items() if n > 2}
    if over:
        problems.append(
            f"{label}: columns {sorted(over)} take more than 2 shim-facing inbound "
            f"flows ({over}) -- the per-column budget J7a's packing exists to meet"
        )

    verdict = "FAIL" if problems else "PASS"
    print(
        f"[norm-tail-structure] {label}: {verdict} "
        f"({len(core_rows)} herd rows x {HERD_X}, {n_core_core} L1->L1 flows, "
        f"max {max(inbound.values(), default=0)} shim inbound per column, "
        f"0 packet-typed channels)"
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
        print()
        for p in problems:
            print(f"[norm-tail-structure] {p}")
        print("[norm-tail-structure] FAIL")
        return 1
    print("[norm-tail-structure] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
