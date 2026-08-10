#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Structural probe: the closest RESIDENT tail the existing builders admit.

WHAT THIS COMPOSES, AND WHAT IT DELIBERATELY DOES NOT CLAIM
    ``fused``'s tail today is PACKAGED (doc 03's vocabulary): ln1 / FFN / ln2
    stitched into one artifact whose operator boundaries all round-trip DRAM
    through L3 func args. The resident tail replaces those boundaries with
    L1->L1 hand-offs. The two space-multiplexed pieces the tree has are J7a's
    norm-tail pipeline (``builders/norm_tail.py``, three herds on L1->L1
    channels) and J7b's accumulator-ring FFN down-projection
    (``builders/ffn_accum.py``, compiler-formed ring). This probe stitches

        nt1 (J7a, mirror_out)  +  ffn_accum (J7b)  +  nt2 (J7a)

    into ONE module with ``stitch_elf`` -- the same mechanism ``pattern/fused``
    uses -- and compiles it through ``XRTBackend(debug_ir=True)`` (the
    production aircc binary) to pass-dump altitude. No NPU, no Peano kernel
    objects: aiecc writes every MLIR pass dump BEFORE compiling core ELFs
    (doc 23 §5), so the compile failing at the core-ELF step costs nothing.

    THIS IS NOT YET A RESIDENT TAIL, and the probe prints its two seams
    rather than hiding them:
      - ``hidden`` (nt1's output, row-major [rows, cols]) and ``a_packed``
        (ffn_accum's A operand, blocked microtiles) are SEPARATE func args:
        the retile between them is unresolved engineering, so the nt1->FFN
        hand-off still crosses L3 here.
      - ``y`` (ffn_accum's output) and ``packed2`` plane 0 (nt2's input) are
        likewise separate args, at different seq (see next item).
      - ``build_ffn_accum_module`` only places at seq 64 (herd_x*herd_y <= 6
        memtile feed ports; MAX_PLACEABLE_HERD_X 4), so the FFN slice is ONE
        64-row band. A full-seq resident tail iterates seq/64 bands: 16 at
        1024, 64 at 4096.
    What the probe DOES establish is the structural envelope the resident
    tail's phase spec (doc 31) needs: whether the toolchain routes the
    composition at all, what air-place-herds does with seven herds over three
    segments, whether both norm-tail pipelines keep their 2*herd_x core->core
    flows when stitched beside the ring, whether any column exceeds the
    2-stream shim budget, whether anything goes packet-typed, and how many
    air.channel symbols the composition carries against air-fuse-channels'
    measured >1200 s wall at 90 channels (doc 03 / doc 26 lane C).

CHECKS (assertions; everything else is reported)
    1. ZERO packet-typed channels in every dump (the J7a/J7b column rule).
    2. Liveness at air-dma-to-channel: dump exists, no residual
       air.dma_memcpy_nd, and all five composed channels -- both prefixed
       copies of norm_tail_a2b/norm_tail_b2c plus the prefixed ffn_accum_feed
       -- survive the stitch by name.
    3. Liveness at the end: no air.channel left, at least one aie.flow.
    4. CORE->CORE FLOW COUNT == 2 * 2 * herd_x (both norm-tail pipelines'
       stage edges, one per column each; ffn_accum contributes none -- its
       feed is memtile->core). An edge silently rerouted through L3 or a
       memtile would still pass every numeric arm; this is the residency
       check.
    5. COLUMN BUDGET: at most 2 shim-facing inbound flows per column, counted
       per aie.device on the routed design.

SHAPES
    1024x768 (fused's gated shape; norm tails plane_major, as fused ships) and
    4096x768 (above the plane-major stride cap, so norm tails row-interleaved
    -- the packing that needs a strided producer, doc 28). Run one shape per
    invocation::

        python3 agents/probes/probe_fused_resident_tail.py --rows 1024
        python3 agents/probes/probe_fused_resident_tail.py --rows 4096

FOOTGUNS
    - The compile is EXPECTED to fail at the core-ELF step (no kernel
      objects in the scratch dir). Check 3 keeps that honest: a compile that
      died before routing fails the probe rather than passing vacuously.
    - aircc writes outputs relative to cwd; the probe chdirs into a scratch
      tempdir per shape.
    - ``--no-mirror`` drops nt1's mirror write (packed2 plane 1). At 4096 the
      mirror's L3-side plane stride (rows*cols = 3,145,728) may exceed the
      shim BD range the plane-major INPUT is known to hit; if the mirrored
      compose stops before routing, re-run with --no-mirror and record both.
"""

import argparse
import glob
import os
import re
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_PE = _REPO / "programming_examples"
# transformer_layer LAST so it ends up FIRST: its builders/ must shadow
# llms/shared/builders, not the other way around (the known trap).
for _p in (str(_PE), str(_PE / "llms"), str(_PE / "transformer_layer")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from air.backend.xrt import XRTBackend  # noqa: E402

from builders.norm_tail import (  # noqa: E402
    CHANNEL_A2B,
    CHANNEL_B2C,
    build_norm_tail_module,
)
from builders.ffn_accum import (  # noqa: E402
    CHANNEL_FEED,
    FFN_ACCUM_HERD_X,
    FFN_ACCUM_TILE_K,
    TILE_M,
    build_ffn_accum_module,
)
from shared.infra.stitching import FuncArg, KernelSlice, stitch_elf  # noqa: E402

COLS = 768
FFN_DIM = 3072
HERD_X = 8  # norm-tail herd width; the catalogue's shapes all run it
# ffn_accum's one placeable geometry: seq 64 (herd 4x1). See module docstring.
FFN_SEQ = TILE_M

_TILE_RE = re.compile(r"%(\S+) = aie\.tile\((\d+),\s*(\d+)\)")
_FLOW_RE = re.compile(
    r"aie\.flow\(%(\S+),\s*\w+\s*:\s*\d+,\s*%(\S+),\s*\w+\s*:\s*\d+\)"
)
_CHANNEL_DUMP_SUFFIX = "_after_air-dma-to-channel.mlir"


def _private_syms(ir_text):
    """Every ``func.func private`` symbol -- the slice's extern set (as
    ``pattern/fused/fused.py`` reads it off the text)."""
    return {
        "@" + m for m in re.findall(r"func\.func private @([A-Za-z0-9_]+)", ir_text)
    }


def build_composition(rows, mirror, herd_x=HERD_X):
    """Stitch nt1 + ffn_accum + nt2 into one module; return (module, layout).

    Above the plane-major cap (rows*cols > 2^20) the mirror is DISABLED even
    when requested, and that is a measured wall, not a convenience: the mirror
    write's layout is plane-major ``[2, rows, cols]`` (norm_tail's
    ``l3_out2_ty``) while the only input packing that builds at those shapes
    is row-interleaved ``[rows, 2, cols]`` -- so ln1's residual hand-off into
    ln2's packed input cannot be composed from these builders at 2048+. The
    same strided-producer wall doc 28 records for the front, one seam later.
    """
    plane_major = rows * COLS <= (1 << 20)
    if mirror and not plane_major:
        print(
            f"[resident-tail-probe] NOTE: mirror disabled at {rows}x{COLS} -- "
            "the mirror writes plane-major [2, rows, cols] but row-interleaved "
            "[rows, 2, cols] is the only packing that builds here (shim BD "
            "stride cap), so the mirrored compose is ill-typed by construction"
        )
        mirror = False
    nt1_ir = str(
        build_norm_tail_module(
            rows, COLS, herd_x=herd_x, plane_major=plane_major, mirror_out=mirror
        )
    )
    nt2_ir = str(
        build_norm_tail_module(rows, COLS, herd_x=herd_x, plane_major=plane_major)
    )
    ffn_ir = str(build_ffn_accum_module(FFN_SEQ, FFN_DIM, COLS))

    act = f"memref<{rows}x{COLS}xbf16>"
    pk = (
        f"memref<2x{rows}x{COLS}xbf16>"
        if plane_major
        else f"memref<{rows}x2x{COLS}xbf16>"
    )
    args, idx = [], {}

    def add(name, ty):
        idx[name] = len(args)
        args.append(FuncArg(f"%arg{len(args)}", ty))

    add("packed1", pk)
    add("gamma1", f"memref<{COLS}xbf16>")
    add("hidden", act)
    # nt2's input, in the same packing as packed1. In the plane-major regime
    # this coincides with the mirror's [2, rows, cols] layout, which is the
    # only reason nt1 can write it; in the row-interleaved regime it is
    # host-filled (see build_composition's mirror note).
    add("packed2", pk)
    # THE SEAM, left visible: ffn_accum's operands are pre-tiled blocked
    # microtiles at seq 64; hidden -> a_packed and y -> packed2 plane 0 are
    # the two unresolved hand-offs a real resident tail must close.
    add("a_packed", f"memref<{FFN_SEQ * FFN_DIM}xbf16>")
    add("w_packed", f"memref<{FFN_DIM * COLS}xbf16>")
    add("y", f"memref<{FFN_SEQ}x{COLS}xbf16>")
    add("gamma2", f"memref<{COLS}xbf16>")
    add("output", act)

    nt1_map = {0: idx["packed1"], 1: idx["gamma1"], 2: idx["hidden"]}
    if mirror:
        nt1_map[3] = idx["packed2"]
    slices = [
        KernelSlice(nt1_ir, "n1", nt1_map, extern_syms=_private_syms(nt1_ir)),
        KernelSlice(
            ffn_ir,
            "ff",
            {0: idx["a_packed"], 1: idx["w_packed"], 2: idx["y"]},
            extern_syms=_private_syms(ffn_ir),
        ),
        KernelSlice(
            nt2_ir,
            "n2",
            {0: idx["packed2"], 1: idx["gamma2"], 2: idx["output"]},
            extern_syms=_private_syms(nt2_ir),
        ),
    ]
    module = stitch_elf(
        "resident_tail",
        args,
        slices,
        debug_dump_path="/tmp/debug_resident_tail.mlir",
    )
    return module, plane_major, mirror


def _aircc_debug_dumps(module):
    """Compile through the production aircc with debug_ir; return dumps."""
    prev_cwd = os.getcwd()
    error = None
    with tempfile.TemporaryDirectory(prefix="resident-tail-probe-") as work:
        os.chdir(work)
        try:
            backend = XRTBackend(
                omit_while_true_loop=False,
                output_format="elf",
                instance_name="resident_tail",
                runtime_loop_tiling_sizes=[2, 2],
                target_device="npu2",
                debug_ir=True,
            )
            try:
                backend.compile(module)
            except Exception as exc:
                error = f"{type(exc).__name__}: {str(exc)[-300:]}"
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


def _device_blocks(text):
    """Each ``aie.device`` block in a dump as ``(name, body)``, in order.

    One block per segment (each ``air.launch`` becomes its own array
    configuration under the multi-launch ELF), plus one anonymous control
    device with no tiles.
    """
    blocks = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if "aie.device" in lines[i]:
            m = re.search(r"aie\.device\(\w+\)\s*(@\S+)?", lines[i])
            name = (m.group(1) if m and m.group(1) else "<anonymous>")
            depth, body = 0, []
            while i < len(lines):
                body.append(lines[i])
                depth += lines[i].count("{") - lines[i].count("}")
                i += 1
                if depth <= 0 and len(body) > 1:
                    break
            blocks.append((name, "\n".join(body)))
        else:
            i += 1
    return blocks or ([("<module>", text)] if "aie.tile" in text else [])


def check_shape(rows, mirror, herd_x=HERD_X):
    module, plane_major, mirror = build_composition(rows, mirror, herd_x=herd_x)
    label = f"resident_tail [{rows}x{COLS}{'' if mirror else ' no-mirror'}]"
    n_launch = str(module).count("air.launch")
    print(
        f"[resident-tail-probe] {label}: composed 3 slices "
        f"({n_launch} air.launch, norm tails "
        f"{'plane-major' if plane_major else 'row-interleaved'}, FFN band "
        f"{FFN_SEQ}x{FFN_DIM}x{COLS} at herd_x={FFN_ACCUM_HERD_X} "
        f"tile_k={FFN_ACCUM_TILE_K}); full-seq residency needs "
        f"{rows // FFN_SEQ} such bands"
    )
    t0 = time.monotonic()
    dumps, compile_error = _aircc_debug_dumps(module)
    elapsed = time.monotonic() - t0

    problems = []
    if not dumps:
        return [
            f"{label}: aircc wrote no debug dumps -- nothing structural is "
            f"measured ({compile_error or 'compile reported success'})"
        ]

    # 1. Zero packet-typed channels anywhere.
    packet_dumps = [n for n, t in dumps if "npu_dma_packet" in t]
    if packet_dumps:
        problems.append(
            f"{label}: packet-typed channels in {len(packet_dumps)} dump(s), "
            f"first {packet_dumps[0]} -- the composition exceeds a per-column "
            "budget somewhere"
        )

    # 2. Liveness + channel census at air-dma-to-channel.
    chan = next((t for n, t in dumps if n.endswith(_CHANNEL_DUMP_SUFFIX)), None)
    n_channels = 0
    if chan is None:
        problems.append(
            f"{label}: no air-dma-to-channel dump -- the compile stopped "
            f"before channel lowering ({compile_error or 'no error'})"
        )
    else:
        if "air.dma_memcpy_nd" in chan:
            problems.append(
                f"{label}: {chan.count('air.dma_memcpy_nd')} air.dma_memcpy_nd "
                "left after air-dma-to-channel -- lowering incomplete"
            )
        n_channels = len(re.findall(r"air\.channel @", chan))
        expect = [
            f"@{p}_{c}"
            for p in ("n1", "n2")
            for c in (CHANNEL_A2B, CHANNEL_B2C)
        ] + [f"@ff_{CHANNEL_FEED}"]
        missing = [c for c in expect if c not in chan]
        if missing:
            problems.append(
                f"{label}: composed channel(s) {missing} missing at "
                "air-dma-to-channel -- the stitch did not preserve the "
                "declared pipelines"
            )

    # The air-fuse-channels census: how many channel symbols enter and leave
    # the pass that did not finish 1200 s on the 90-channel whole-layer stitch
    # (doc 26 lane C). The probe's own wall-clock bounds the pass from above.
    fuse_idx = next(
        (
            i
            for i, (n, _) in enumerate(dumps)
            if n.endswith("_after_air-fuse-channels.mlir")
        ),
        None,
    )
    if fuse_idx is not None and fuse_idx > 0:
        pre = len(re.findall(r"air\.channel @", dumps[fuse_idx - 1][1]))
        post = len(re.findall(r"air\.channel @", dumps[fuse_idx][1]))
        print(
            f"[resident-tail-probe]   air-fuse-channels: {pre} -> {post} "
            f"channel symbols (whole compile {elapsed:.1f}s, vs >1200s at 90 "
            "channels on the whole-layer stitch)"
        )

    # 3-5. The routed design.
    final_name, final = dumps[-1]
    if "air.channel" in final:
        problems.append(
            f"{label}: air.channel still present in {final_name} -- not "
            f"lowered to AIE routing ({compile_error or 'no error'})"
        )
    devices = _device_blocks(final)
    total_core_core = 0
    placements = []
    max_inbound = 0
    if not any(_FLOW_RE.search(d) for _, d in devices):
        problems.append(
            f"{label}: no aie.flow in {final_name} -- nothing was routed "
            f"({compile_error or 'compile reported success'})"
        )
    else:
        for dev_name, dev in devices:
            tiles = {m[0]: (int(m[1]), int(m[2])) for m in _TILE_RE.findall(dev)}
            rows_by = {k: v[1] for k, v in tiles.items()}
            cols_by = {k: v[0] for k, v in tiles.items()}

            def kind(t):
                r = rows_by.get(t)
                if r is None:
                    return None
                return "shim" if r == 0 else "memtile" if r == 1 else "core"

            flows = _FLOW_RE.findall(dev)
            edges = Counter((kind(s), kind(d)) for s, d in flows)
            core_core = edges.get(("core", "core"), 0)
            total_core_core += core_core
            inbound = Counter(
                cols_by.get(d)
                for s, d in flows
                if kind(s) == "shim" and kind(d) == "core"
            )
            over = {c: n for c, n in inbound.items() if n > 2}
            max_inbound = max(max_inbound, max(inbound.values(), default=0))
            if over:
                problems.append(
                    f"{label}: device {dev_name}: columns {sorted(over)} take "
                    f"more than 2 shim-facing inbound flows ({over}) -- the "
                    "per-column budget the packing rules exist to meet"
                )
            core_rows = Counter(r for r in rows_by.values() if r >= 2)
            placements.append(
                f"device {dev_name}: core rows {dict(sorted(core_rows.items()))}, "
                f"flows {dict(edges)}, shim inbound/col "
                f"{dict(sorted(inbound.items()))}"
            )
        expected_l1 = 2 * 2 * HERD_X
        if total_core_core != expected_l1:
            problems.append(
                f"{label}: {total_core_core} core->core flows across "
                f"{len(devices)} aie.device, expected {expected_l1} (both "
                "norm-tail pipelines' stage edges, one per column each) -- a "
                "stage edge is not L1->L1"
            )

    verdict = "FAIL" if problems else "PASS"
    print(
        f"[resident-tail-probe] {label}: {verdict} "
        f"({len(dumps)} dumps in {elapsed:.1f}s, {n_channels} air.channel "
        f"symbols, {len(devices)} aie.device, {total_core_core} core->core "
        f"flows, max {max_inbound} shim inbound per column, "
        f"{len(packet_dumps)} packet dumps; compile: "
        f"{compile_error or 'succeeded'})"
    )
    for p in placements:
        print(f"[resident-tail-probe]   {p}")
    print(
        "[resident-tail-probe]   seams left visible: hidden->a_packed "
        "(retile, L3) and y->packed2 plane 0 (band writeback, L3); the FFN "
        f"slice is one {FFN_SEQ}-row band of {rows}"
    )
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1024, choices=(1024, 4096))
    ap.add_argument(
        "--no-mirror",
        action="store_true",
        help="drop nt1's mirror write into packed2 plane 1",
    )
    ap.add_argument(
        "--herd-x",
        type=int,
        default=HERD_X,
        help="norm-tail herd width for the BUILD only; the checks stay pinned "
        "to the 8-wide contract, so any other value is the negative control "
        "(air-place-herds collapses a 4-wide pipeline and the core->core and "
        "placement checks must fire -- verified failing at --herd-x 4)",
    )
    ns = ap.parse_args()

    problems = check_shape(ns.rows, not ns.no_mirror, herd_x=ns.herd_x)
    if problems:
        print()
        for p in problems:
            print(f"[resident-tail-probe] {p}")
        print("[resident-tail-probe] FAIL")
        return 1
    print("[resident-tail-probe] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
