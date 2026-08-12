#!/usr/bin/env python3
"""Find L2/L1 buffers that MORE THAN ONE DMA channel writes, with no ordering
between the writers -- the shape of queue item 21 (wall 7).

WHY THIS EXISTS.  Item 23 was a lock *placement* defect; item 18 was a lock
*count* defect.  This is a third, independent class the other two instruments
are blind to by construction:

    N independent S2MM channels fill ONE buffer, each gated by the SAME
    counting semaphore pair with the SAME acquire/release counts.

Lock counts are conserved (item 18's audit passes), and every acquire dominates
its own write (item 23's audit passes) -- yet the *order in which the N writers
fill the buffer is undefined*, because a counting semaphore has no identity.
Whichever channel's acquire is granted first writes first.  Downstream that is
not a hang and not a truncation: it is a **permutation** of an intended stream,
which reads as a plausible-looking wrong answer.

R1's resident FFN hits it at ``herd_x >= 2``: the down feed's ``l2_h`` staging
buffer is filled by one S2MM channel PER GeLU CORE (``ChannelGet(CHANNEL_G,
l2_h, indices=[c, 0])``, one textual op per ``c``), while the builder's own
comment relies on those gets being *serialized* by the WAR chain on the shared
buffer.  At ``herd_x == 1`` there is exactly one writer and the reliance is
vacuously satisfied, which is why the whole ``herd_x=1`` ladder is clean.

WHAT IT REPORTS.  For every ``aie.buffer`` in an ``aie.air.mlir``: the set of
DMA channels that write it (S2MM BDs) and read it (MM2S BDs), the acquire and
release lock symbols and counts on each side, and the number of distinct BD
slots (the ping-pong depth).  A buffer with >1 writer channel is flagged
MULTI-WRITER; if it also has only one slot it is flagged as a RACE, because
then the writers contend for the same storage rather than for a queue.

    python3 agents/probes/probe_aie_buffer_writer_race.py path/to/aie.air.mlir
    python3 agents/probes/probe_aie_buffer_writer_race.py --refuse-race a.mlir

``--refuse-race`` exits non-zero when a single-slot multi-writer buffer exists,
so it can gate a build.  It is only meaningful with a negative control: run it
on a ``herd_x=1`` module (must be clean) and a ``herd_x>=2`` one (must flag).

WHAT IT IS NOT.  It does not prove a race is REACHABLE -- two writers can be
serialized by a data dependency the DMA program does not encode (e.g. an
upstream token).  It reports the *absence of an ordering mechanism in the DMA
program*, which is a necessary condition, and names the buffer so the argument
can be made or refuted on the spot.
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

# `%memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {` / `aie.mem(%tile_0_2) {`
_RE_DMA_OP = re.compile(r"aie\.(memtile_dma|mem|shim_dma)\((%[\w]+)\)")
# `%0 = aie.dma_start(MM2S, 0, ^bb1, ^bb6)`
_RE_DMA_START = re.compile(r"aie\.dma_start\((MM2S|S2MM),\s*(\d+)")
# `aie.dma_bd(%buf18 : memref<2048xbf16, 1 : i32> offset = 0 len = 2048)`
_RE_DMA_BD = re.compile(
    r"aie\.dma_bd\((%[\w]+)\s*:\s*memref<([^>]*?),\s*(\d+)\s*:\s*i32>"
    r"(?:\s*offset\s*=\s*(\d+))?(?:\s*len\s*=\s*(\d+))?"
)
# `aie.use_lock(%lock_0_1_31, AcquireGreaterEqual, %c1_i32)`
_RE_USE_LOCK = re.compile(
    r"aie\.use_lock\((%[\w]+),\s*(AcquireGreaterEqual|Acquire|Release)"
    r"(?:,\s*(%[\w]+|\d+))?"
)
# `%c2_i32 = arith.constant 2 : i32`
_RE_CONST = re.compile(r"(%[\w]+)\s*=\s*arith\.constant\s+(-?\d+)\s*:")
# `%buf21 = aie.buffer(%mem_tile_3_1) {sym_name = "buf21"} : memref<2048xbf16, 1 : i32>`
_RE_BUFFER = re.compile(
    r"(%[\w]+)\s*=\s*aie\.buffer\((%[\w]+)\).*?memref<([^>]*?),\s*(\d+)\s*:\s*i32>"
)
_RE_FLOW = re.compile(
    r"aie\.flow\((%[\w]+),\s*(\w+)\s*:\s*(\d+),\s*(%[\w]+),\s*(\w+)\s*:\s*(\d+)\)"
)

MEMSPACE = {0: "L3", 1: "L2", 2: "L1"}


class BD:
    __slots__ = ("buf", "offset", "length", "acq", "acqn", "rel", "reln")

    def __init__(self, buf, offset, length):
        self.buf, self.offset, self.length = buf, offset, length
        self.acq = self.rel = None
        self.acqn = self.reln = None


def parse(text):
    consts = {m.group(1): int(m.group(2)) for m in _RE_CONST.finditer(text)}

    buffers = {}
    for m in _RE_BUFFER.finditer(text):
        buffers[m.group(1)] = {
            "tile": m.group(2),
            "type": m.group(3),
            "space": int(m.group(4)),
        }

    flows = []
    for m in _RE_FLOW.finditer(text):
        flows.append((m.group(1), m.group(2), int(m.group(3)),
                      m.group(4), m.group(5), int(m.group(6))))

    # Walk lines, tracking the enclosing DMA op and the active dma_start.
    cur_tile = None
    cur_dir = None
    cur_chan = None
    # (tile, dir, chan) -> [BD, ...]
    chans = defaultdict(list)
    pending_acq = None

    for line in text.splitlines():
        m = _RE_DMA_OP.search(line)
        if m:
            cur_tile = m.group(2)
            cur_dir = cur_chan = None
            pending_acq = None
            continue
        m = _RE_DMA_START.search(line)
        if m:
            cur_dir, cur_chan = m.group(1), int(m.group(2))
            pending_acq = None
            continue
        if cur_tile is None or cur_dir is None:
            continue
        m = _RE_USE_LOCK.search(line)
        if m:
            sym, action, cnt = m.group(1), m.group(2), m.group(3)
            n = consts.get(cnt, None)
            if n is None and cnt is not None and cnt.isdigit():
                n = int(cnt)
            if action.startswith("Acquire"):
                pending_acq = (sym, n)
            else:
                bds = chans[(cur_tile, cur_dir, cur_chan)]
                if bds and bds[-1].rel is None:
                    bds[-1].rel, bds[-1].reln = sym, n
            continue
        m = _RE_DMA_BD.search(line)
        if m:
            bd = BD(m.group(1), int(m.group(4) or 0), int(m.group(5) or 0))
            if pending_acq:
                bd.acq, bd.acqn = pending_acq
                pending_acq = None
            chans[(cur_tile, cur_dir, cur_chan)].append(bd)
            continue

    return buffers, chans, flows


def analyze(path, refuse_race=False, only_space=None, quiet=False):
    text = Path(path).read_text()
    buffers, chans, flows = parse(text)

    # buffer -> {"S2MM": {(tile,chan): [bd...]}, "MM2S": {...}}
    per_buf = defaultdict(lambda: {"S2MM": defaultdict(list), "MM2S": defaultdict(list)})
    for (tile, d, chan), bds in chans.items():
        for bd in bds:
            per_buf[bd.buf][d][(tile, chan)].append(bd)

    # flow source lookup: which tile feeds (dst_tile, dst_chan) on S2MM
    src_of = {}
    for st, sb, sc, dt, db, dc in flows:
        src_of[(dt, dc)] = (st, sc)

    races, multi = [], []
    print(f"=== {path} ===")
    hdr = (f"{'buffer':<8} {'tile':<14} {'sp':<3} {'slots':>5} "
           f"{'writers':>7} {'readers':>7}  detail")
    print(hdr)
    for buf in sorted(per_buf, key=lambda b: (buffers.get(b, {}).get("space", 9), b)):
        info = buffers.get(buf, {})
        space = info.get("space")
        if only_space is not None and space != only_space:
            continue
        w = per_buf[buf]["S2MM"]
        r = per_buf[buf]["MM2S"]
        # ping-pong depth: how many DISTINCT buffers cycle on a writer channel
        slots = set()
        for (tile, chan), bds in w.items():
            for other in chans[(tile, "S2MM", chan)]:
                slots.add(other.buf)
        nslots = len(slots) if slots else 1
        flag = ""
        if len(w) > 1:
            multi.append(buf)
            flag = "  MULTI-WRITER"
            if nslots <= 1:
                races.append(buf)
                flag = "  <== RACE: single slot, no writer ordering"
        print(f"{buf:<8} {info.get('tile','?'):<14} {MEMSPACE.get(space,'?'):<3} "
              f"{nslots:>5} {len(w):>7} {len(r):>7}{flag}")
        if flag and not quiet:
            for (tile, chan), bds in sorted(w.items()):
                s = src_of.get((tile, chan))
                bd = bds[0]
                print(f"         writer S2MM {chan} on {tile} "
                      f"<- {s[0] if s else '?'}  acq {bd.acq}>={bd.acqn} "
                      f"rel {bd.rel}x{bd.reln}")
            for (tile, chan), bds in sorted(r.items()):
                bd = bds[0]
                print(f"         reader MM2S {chan} on {tile} "
                      f"acq {bd.acq}>={bd.acqn} rel {bd.rel}x{bd.reln}")

    print(f"[verdict] {len(multi)} multi-writer buffer(s), {len(races)} of them "
          f"single-slot (unordered writer race): {races if races else 'none'}")
    if refuse_race and races:
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mlir", nargs="+", help="aie.air.mlir dump(s)")
    ap.add_argument("--refuse-race", action="store_true",
                    help="exit 1 if any single-slot multi-writer buffer exists")
    ap.add_argument("--space", type=int, default=None,
                    help="restrict to a memory space (1 = L2, 2 = L1)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    rc = 0
    for p in a.mlir:
        rc |= analyze(p, a.refuse_race, a.space, a.quiet)
        print()
    return rc


if __name__ == "__main__":
    sys.exit(main())
