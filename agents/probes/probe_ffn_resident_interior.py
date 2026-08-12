#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Structural probe: doc 31 R1, the one-segment resident FFN interior.

WHAT THIS MEASURES, AND WHY EACH QUESTION IS OPEN
    ``builders/ffn_resident.py`` composes up-projection, GeLU and the J7b
    down-projection ring as three herds of ONE segment, hand-offs on
    channels, for one 64-row band. ``probe_fused_resident_tail.py``
    established that a *stitched* composition lowers to one aie.device per
    launch -- residency needs a one-segment builder, and this probe is that
    builder's first compile at pass-dump altitude (hermetic: no NPU, no
    kernel objects; aiecc writes every MLIR dump before core ELFs, doc 23
    section 5). The open questions it settles, none of which an air-opt pass
    list can answer:

    1. ONE DEVICE. The resident claim itself: one air.launch must lower to
       exactly one tile-bearing aie.device (plus the anonymous control
       device). The stitched probe measured three.
    2. THE SHIM RETILE (seam 1, L3 case). ``hidden`` is row-major; the up
       feed's refill reads it through the 4-D blocked pattern on the L3 side.
       J7b's C fetch proves the idiom per K step at [64, 192]; this is the
       same idiom at [64, 32] per k' step, plus a second 1-D refill on the
       same loop.
    3. THE FAN-THROUGH-MEMTILE HAND-OFF (seam 3). Segment-scope ChannelGet
       (GeLU -> L2 chunk) has no precedent in this tree -- every landed
       builder only PUTS from segment scope. If air-dma-to-channel or the
       memtile DMA allocation refuses, that is the wall, stated exactly.
    4. MEMTILE PORT PRESSURE. Both feeds' buffers plus the relay staging sum
       to 8 MM2S + 8 S2MM if everything lands on one memtile (a memtile has
       6/6); the design is legal only if the allocator spreads L2 buffers
       across columns' memtiles. Nothing in the tree measures that today.
    5. THE UP ACCUMULATOR'S PING-PONG FATE. The group C is written by kernel
       calls and read by channel puts across a sweep loop -- the exact shape
       aircc's labelling ping-pongs in J7a's stages. Doubled, the up core
       needs 81 KiB of 64; single, 57 KiB (J7b's proven budget). The probe
       counts the [64x192] allocs in the up herd in the last AIR-dialect
       dump, so the answer is measured, not assumed.
    6. THE DOWN RING STILL FORMS INSIDE THE COMPOSITION: K-loop data
       movement 4 -> 2 across air-hoist-dma-in-accum-pattern, scoped to
       @ffn_res_down (this module has three herds; "first loop in the dump"
       would find the up herd's sweep loop).
    7. THE FUSE-CHANNELS CENSUS: 4 channel symbols against the >1200 s wall
       at 90 (doc 26 lane C); the probe's wall-clock bounds the pass.

CHECKS (assertions; everything else is reported)
    A. Zero packet-typed channels in every dump.
    B. air-dma-to-channel liveness: dump exists, no residual
       air.dma_memcpy_nd, all four composed channels present by name.
    C. Final-dump liveness: no air.channel left, aie.flow present.
    D. Exactly one tile-bearing aie.device.
    E. core->core flows == herd_x (the up->GeLU edges, one per column);
       every other hand-off is memtile-mediated BY the port arithmetic in
       the builder's docstring, so more core->core is as wrong as fewer.
    F. Per-column shim MM2S DEMAND <= 2 on the routed design (doc 23's
       rule: a column has two shim MM2S channels and the budget is per
       column across the whole segment). `[2026-08-12]` WIDENED with the
       gate's, queue item 10: it counted ``shim -> core`` circuit flows
       only and read 1 for a column at 2, because an L2-staged refill is a
       ``shim -> memtile`` flow on the same port (doc 31b 3.6), and because
       AIR answers an over-budget column by PACKET-multiplexing the streams
       onto one queue rather than refusing -- so a circuit-only count reads
       0 exactly when the clause is needed. Measured on R1: 7 of 16 ports,
       worst column 2 (was 4 of 16, worst 1). The gate twin
       (``programming_examples/transformer_layer/ffn_resident_structure.py``)
       carries the same census AND its negative control; this probe is the
       exploratory half.
    G. Down-herd K loop 4 -> 2 across the hoist pass, with the accumulate
       call and guarded zero inside.

RUN
    python3 agents/probes/probe_ffn_resident_interior.py
    (~seconds; add --keep-dumps DIR to copy the debug_ir tree out)
"""

import argparse
import glob
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

# aircc runs as a child; a pass that stops terminating (measured: 24 auto
# channels put air-isolate-async-dma-loop-nests into a >25-minute state)
# would otherwise wedge the build queue. On alarm, kill the aircc child so
# the backend's subprocess call returns and the dumps written so far are
# still read -- the partial census is the diagnostic.
COMPILE_TIMEOUT_S = 600


class _CompileTimeout(Exception):
    pass


def _on_alarm(_sig, _frame):
    subprocess.run(["pkill", "-P", str(os.getpid()), "aircc"], check=False)
    raise _CompileTimeout(f"compile exceeded {COMPILE_TIMEOUT_S}s")


_REPO = Path(__file__).resolve().parents[2]
_PE = _REPO / "programming_examples"
# transformer_layer LAST so it ends up FIRST: its builders/ must shadow
# llms/shared/builders, not the other way around (the known trap).
for _p in (str(_PE), str(_PE / "llms"), str(_PE / "transformer_layer")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from air.backend.xrt import XRTBackend  # noqa: E402

from builders.ffn_resident import (  # noqa: E402
    CHANNEL_DOWN_FEED,
    CHANNEL_G,
    CHANNEL_H,
    CHANNEL_UP_FEED,
    build_ffn_resident_module,
)
from builders.ffn_accum import (  # noqa: E402
    FFN_ACCUM_HERD_X,
    FFN_ACCUM_MM_SYMBOL,
    FFN_ACCUM_TILE_K,
    FFN_ACCUM_ZERO_SYMBOL,
    TILE_M,
)

FFN_DIM = 3072
EMB_DIM = 768

# AIETargetModel.h: NPU2TargetModel::columns() = 8, and a shim NOC tile
# drives two MM2S DMA channels -- doc 23's per-column budget.
NPU2_COLUMNS = 8
SHIM_MM2S_PER_COLUMN = 2
NPU2_SHIM_MM2S_PORTS = NPU2_COLUMNS * SHIM_MM2S_PER_COLUMN

_TILE_RE = re.compile(r"%(\S+) = aie\.tile\((\d+),\s*(\d+)\)")
_FLOW_RE = re.compile(
    r"aie\.flow\(%(\S+),\s*(\w+)\s*:\s*(\d+),\s*%(\S+),\s*(\w+)\s*:\s*(\d+)\)"
)
# ONE aie.packet_source per aie.packet_flow (a flow may broadcast to many
# dests), so counting sources counts flows and needs no brace matching.
_PACKET_SOURCE_RE = re.compile(r"aie\.packet_source<%(\S+),\s*(\w+)\s*:\s*(\d+)>")
_MOVEMENT = re.compile(r"air\.dma_memcpy_nd|air\.channel\.(?:put|get)")
_CHANNEL_DUMP_SUFFIX = "_after_air-dma-to-channel.mlir"
_HOIST_DUMP_SUFFIX = "_after_air-hoist-dma-in-accum-pattern.mlir"
_DOWN_HERD_MARKER = "air.herd @ffn_res_down"
_UP_HERD_MARKER = "air.herd @ffn_res_up"


def _aircc_debug_dumps(module, keep_dumps=None):
    prev_cwd = os.getcwd()
    error = None
    with tempfile.TemporaryDirectory(prefix="ffn-resident-probe-") as work:
        os.chdir(work)
        try:
            backend = XRTBackend(
                omit_while_true_loop=False,
                output_format="elf",
                instance_name="ffn_resident",
                runtime_loop_tiling_sizes=[2, 2],
                target_device="npu2",
                debug_ir=True,
            )
            signal.signal(signal.SIGALRM, _on_alarm)
            signal.alarm(COMPILE_TIMEOUT_S)
            try:
                backend.compile(module)
            except Exception as exc:
                error = f"{type(exc).__name__}: {str(exc)[-400:]}"
            finally:
                signal.alarm(0)
                try:
                    backend.unload()
                except Exception:
                    pass
            dumps = [
                (os.path.basename(p), Path(p).read_text())
                for p in sorted(glob.glob("air_project/debug_ir/pass_*.mlir"))
            ]
            if keep_dumps and os.path.isdir("air_project/debug_ir"):
                shutil.copytree("air_project/debug_ir", keep_dumps, dirs_exist_ok=True)
        finally:
            os.chdir(prev_cwd)
    return dumps, error


def _braced_block(text, marker):
    if text is None:
        return None
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if marker in line), None)
    if start is None:
        return None
    depth, body = 0, []
    for line in lines[start:]:
        body.append(line)
        depth += line.count("{") - line.count("}")
        if depth <= 0 and len(body) > 1:
            break
    return "\n".join(body)


def _first_loop(text):
    return _braced_block(text, "scf.for")


def _device_blocks(text):
    blocks = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if "aie.device" in lines[i]:
            m = re.search(r"aie\.device\(\w+\)\s*(@\S+)?", lines[i])
            name = m.group(1) if m and m.group(1) else "<anonymous>"
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


def _shim_mm2s_census(dev):
    """Per-column shim MM2S demand -- the same census the gate twin runs.

    Kept literally in step with
    ``programming_examples/transformer_layer/ffn_resident_structure.py``'s
    ``_shim_mm2s_census``; that one is the gate and carries the negative
    control. Returns ``(demand, detail)`` with ``detail[col] =
    (circuit_ports, packet_streams, to_core, to_memtile, s2mm)``.

    A circuit ``aie.flow`` out of a shim tile takes one MM2S channel per
    distinct source (bundle, channel) whatever it is wired to -- core (a
    herd-direct fetch) or memtile (an L2-staged refill); several flows off
    one source port are one broadcast and count once. A packet flow counts
    once per stream, because packet-multiplexing IS the over-budget
    behaviour and counting surviving ports there reads 1 (or 0) for a
    column carrying k.
    """
    tiles = {m[0]: (int(m[1]), int(m[2])) for m in _TILE_RE.findall(dev)}

    def kind(t):
        rc = tiles.get(t)
        if rc is None:
            return None
        return "shim" if rc[1] == 0 else "memtile" if rc[1] == 1 else "core"

    def col(t):
        rc = tiles.get(t)
        return None if rc is None else rc[0]

    circuit_ports = {}
    to_core, to_memtile, s2mm = Counter(), Counter(), Counter()
    for s, sb, sp, d, db, _dp in _FLOW_RE.findall(dev):
        if kind(s) == "shim" and sb == "DMA":
            circuit_ports.setdefault(col(s), set()).add((sb, int(sp)))
            if kind(d) == "core":
                to_core[col(s)] += 1
            elif kind(d) == "memtile":
                to_memtile[col(s)] += 1
        if kind(d) == "shim" and db == "DMA":
            s2mm[col(d)] += 1

    packet = Counter()
    for t, b, _p in _PACKET_SOURCE_RE.findall(dev):
        if kind(t) == "shim" and b == "DMA":
            packet[col(t)] += 1

    cols = set(circuit_ports) | set(packet) | set(s2mm)
    demand, detail = {}, {}
    for c in sorted(x for x in cols if x is not None):
        nc = len(circuit_ports.get(c, ()))
        np_ = packet.get(c, 0)
        demand[c] = nc + np_
        detail[c] = (nc, np_, to_core.get(c, 0), to_memtile.get(c, 0),
                     s2mm.get(c, 0))
    return demand, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-dumps", default=None)
    ns = ap.parse_args()

    herd_x = FFN_ACCUM_HERD_X
    module = build_ffn_resident_module(TILE_M, FFN_DIM, EMB_DIM, herd_x=herd_x)
    src = str(module)
    n_launch = src.count("air.launch")
    print(
        f"[ffn-resident-probe] composed one segment, {n_launch} air.launch, "
        f"{len(src.splitlines())} lines; band {TILE_M}x{FFN_DIM}x{EMB_DIM}, "
        f"herd_x={herd_x}, tile_k={FFN_ACCUM_TILE_K}"
    )
    problems = []
    if n_launch != 1:
        problems.append(f"{n_launch} air.launch in the built module, need 1")

    t0 = time.monotonic()
    dumps, compile_error = _aircc_debug_dumps(module, keep_dumps=ns.keep_dumps)
    elapsed = time.monotonic() - t0
    if not dumps:
        print(
            f"[ffn-resident-probe] FAIL: no debug dumps "
            f"({compile_error or 'compile reported success'})"
        )
        return 1

    # A. packet census
    packet_dumps = [n for n, t in dumps if "npu_dma_packet" in t]
    if packet_dumps:
        problems.append(
            f"packet-typed channels in {len(packet_dumps)} dump(s), first "
            f"{packet_dumps[0]} -- a per-column budget is exceeded"
        )

    # B. channel-lowering liveness
    chan = next((t for n, t in dumps if n.endswith(_CHANNEL_DUMP_SUFFIX)), None)
    n_channels = 0
    if chan is None:
        problems.append(
            "no air-dma-to-channel dump -- the compile stopped before channel "
            f"lowering ({compile_error or 'no error'})"
        )
    else:
        if "air.dma_memcpy_nd" in chan:
            problems.append(
                f"{chan.count('air.dma_memcpy_nd')} air.dma_memcpy_nd left "
                "after air-dma-to-channel -- lowering incomplete"
            )
        n_channels = len(re.findall(r"air\.channel @", chan))
        missing = [
            c
            for c in (CHANNEL_UP_FEED, CHANNEL_H, CHANNEL_G, CHANNEL_DOWN_FEED)
            if f"@{c}" not in chan
        ]
        if missing:
            problems.append(
                f"channel(s) {missing} missing at air-dma-to-channel -- the "
                "declared pipeline did not survive lowering"
            )

    # 7. fuse-channels census
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
            f"[ffn-resident-probe]   air-fuse-channels: {pre} -> {post} "
            f"channel symbols (whole compile {elapsed:.1f}s, vs >1200s at 90 "
            "on the whole-layer stitch)"
        )

    # G. the down ring inside the composition
    names = [n for n, _ in dumps]
    hoist_idx = next(
        (i for i, n in enumerate(names) if n.endswith(_HOIST_DUMP_SUFFIX)), None
    )
    if not hoist_idx:
        problems.append(
            "no air-hoist-dma-in-accum-pattern dump -- the down ring's "
            "formation is unmeasured"
        )
    else:
        down_before = _first_loop(
            _braced_block(dumps[hoist_idx - 1][1], _DOWN_HERD_MARKER)
        )
        down_after = _first_loop(_braced_block(dumps[hoist_idx][1], _DOWN_HERD_MARKER))
        before = len(_MOVEMENT.findall(down_before or ""))
        after = len(_MOVEMENT.findall(down_after or ""))
        mm_in = f"call @{FFN_ACCUM_MM_SYMBOL}(" in (down_after or "")
        zero_in = f"call @{FFN_ACCUM_ZERO_SYMBOL}(" in (down_after or "")
        print(
            f"[ffn-resident-probe]   down K loop movement {before} -> {after} "
            f"across the hoist (mm call inside: {mm_in}, guarded zero: {zero_in})"
        )
        if before != 4 or after != 2 or not mm_in or not zero_in:
            problems.append(
                f"down-herd K loop {before} -> {after} across the hoist "
                "(need 4 -> 2 with the accumulate call and guarded zero "
                "inside) -- the ring is not forming inside the composition"
            )

    # 5. the up accumulator's ping-pong fate, measured in the last AIR dump
    group_n = EMB_DIM // herd_x
    last_air = next((t for _, t in reversed(dumps) if _UP_HERD_MARKER in t), None)
    if last_air is not None:
        up_body = _braced_block(last_air, _UP_HERD_MARKER)
        n_group_allocs = len(
            re.findall(
                rf"memref\.alloc\(\)[^\n]*memref<{TILE_M}x{group_n}xbf16, 2",
                up_body or "",
            )
        )
        print(
            f"[ffn-resident-probe]   up herd carries {n_group_allocs} "
            f"[{TILE_M}x{group_n}] group alloc(s) in the last AIR dump "
            "(1 = single accumulator, 2 = ping-ponged: +24 KiB per core)"
        )
    # The authoritative L1 answer is the placed design: per up-core buffer
    # set in the final dump (measured 2026-08-11: C+A+B, all single -- the
    # labelling ping-pongs NOTHING in this composition, so the up core sits
    # at 40 KiB of 64 and the open question is latency overlap, not fit).
    final_bufs = Counter(re.findall(r"aie\.buffer\((%tile_\d+_[23])\)", dumps[-1][1]))
    if final_bufs:
        by_count = Counter(final_bufs.values())
        print(
            f"[ffn-resident-probe]   placed L1 buffers per core tile: "
            f"{dict(sorted(by_count.items()))} (tiles x buffer-count)"
        )

    # C-F. the routed design
    final_name, final = dumps[-1]
    if "air.channel" in final:
        problems.append(
            f"air.channel still present in {final_name} -- not lowered to AIE "
            f"routing ({compile_error or 'no error'})"
        )
    devices = _device_blocks(final)
    tiled = [(n, d) for n, d in devices if _TILE_RE.search(d)]
    if len(tiled) != 1:
        problems.append(
            f"{len(tiled)} tile-bearing aie.device in {final_name}, need "
            "exactly 1 -- the one-segment claim itself"
        )
    total_core_core = 0
    mm2s_total = 0
    mm2s_worst = 0
    if not any(_FLOW_RE.search(d) for _, d in devices):
        problems.append(
            f"no aie.flow in {final_name} -- nothing was routed "
            f"({compile_error or 'compile reported success'})"
        )
    for dev_name, dev in devices:
        tiles = {m[0]: (int(m[1]), int(m[2])) for m in _TILE_RE.findall(dev)}
        if not tiles:
            continue
        rows_by = {k: v[1] for k, v in tiles.items()}
        cols_by = {k: v[0] for k, v in tiles.items()}

        def kind(t):
            r = rows_by.get(t)
            if r is None:
                return None
            return "shim" if r == 0 else "memtile" if r == 1 else "core"

        flows = _FLOW_RE.findall(dev)
        edges = Counter(
            (kind(s), kind(d)) for s, _sb, _sp, d, _db, _dp in flows
        )
        core_core = edges.get(("core", "core"), 0)
        total_core_core += core_core
        demand, detail = _shim_mm2s_census(dev)
        mm2s_total += sum(demand.values())
        mm2s_worst = max([mm2s_worst] + list(demand.values()))
        over = {c: n for c, n in demand.items() if n > SHIM_MM2S_PER_COLUMN}
        if over:
            problems.append(
                f"device {dev_name}: columns {sorted(over)} demand more than "
                f"{SHIM_MM2S_PER_COLUMN} shim MM2S ({over}) -- doc 23's "
                "per-column budget; AIR packet-multiplexes past it rather "
                "than refusing"
            )
        core_rows = Counter(r for r in rows_by.values() if r >= 2)
        memtile_cols = sorted({cols_by[t] for t, r in rows_by.items() if r == 1})
        census_str = ", ".join(
            "col {}: {} (core {}, memtile {}, packet {})".format(
                c, demand[c], detail[c][2], detail[c][3], detail[c][1]
            )
            for c in sorted(demand)
        )
        print(
            f"[ffn-resident-probe]   device {dev_name}: core rows "
            f"{dict(sorted(core_rows.items()))}, memtile cols {memtile_cols}, "
            f"flows {dict(edges)}, shim MM2S [{census_str}]"
        )
    if tiled and total_core_core != herd_x:
        problems.append(
            f"{total_core_core} core->core flows, expected exactly {herd_x} "
            "(the up->GeLU edges; every other hand-off is memtile-mediated by "
            "the port arithmetic) -- an edge moved"
        )

    verdict = "FAIL" if problems else "PASS"
    print(
        f"[ffn-resident-probe] {verdict} ({len(dumps)} dumps in {elapsed:.1f}s, "
        f"{n_channels} air.channel symbols, {len(devices)} aie.device "
        f"({len(tiled)} with tiles), {total_core_core} core->core flows, shim "
        f"MM2S {mm2s_total}/{NPU2_SHIM_MM2S_PORTS} worst column {mm2s_worst}, "
        f"{len(packet_dumps)} packet dumps; "
        f"compile: {compile_error or 'succeeded'})"
    )
    if problems:
        print()
        for p in problems:
            print(f"[ffn-resident-probe] {p}")
        print("[ffn-resident-probe] FAIL")
        return 1
    print("[ffn-resident-probe] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
