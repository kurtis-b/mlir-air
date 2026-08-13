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

``--check-order`` (added for queue item 22) closes exactly that gap.  The
structural report above is a *shape* test: >1 writer channel on a 1-slot
buffer.  A shape test cannot tell an unordered counting semaphore from a
correctly chained one -- both have N writer channels on one slot -- so it can
neither certify a fix nor refute one.  ``--check-order`` therefore SIMULATES
the emitted lock protocol: it builds a Petri net whose places are the tile's
``aie.lock``s (with their declared ``init``) and whose transitions are the
``aie.dma_bd``s (each with its single acquire and single release, advancing a
per-channel program counter along the ``aie.next_bd`` chain), then explores the
reachable state space and asks, for each write to the buffer:

    is the identity of the k-th writer the SAME on every interleaving?

Verdicts: ``ORDERED`` (one possible writer per position -- the DMA program
alone forces the order), ``RACE`` (some position admits >1 writer), or
``DEADLOCK`` (a reachable state with no enabled transition).  Streams are
modelled as always-ready, which is the conservative reading of the question the
probe asks: *does the DMA program by itself force the order*.

``--refuse-unordered`` gates on that simulation.  ``--refuse-race`` is left
exactly as it was -- structural -- so the numbers already recorded against it
stay comparable.

WHERE ``--check-order`` IS VALID, AND WHERE IT IS NOT.  `[2026-08-12]`, queue
rows 28/30.  The net models the DMA program and NOTHING ELSE.  On a memtile that
is the whole protocol: measured on R1, an ``aie.memtile_dma``'s locks are
touched by 0 core ops.  On an L1 tile it is NOT -- ``aie.core`` acquires and
releases *the same locks* the ``aie.mem`` BDs do (measured on R1: 4 of 4 and 6
of 6 locks shared on every compute tile).  A net missing one participant loses
releases, so it reports DEADLOCK/OVERWRITE where the real protocol is fine:
false positives, in the direction that manufactures hazards.  Such buffers are
therefore reported ``UNMODELLED`` rather than given a hazard verdict, and
``--refuse-unordered`` does not gate on them.  Reading an L1 verdict as a hazard
is how a latent-race claim gets attached to a tile the instrument cannot see.
"""

import argparse
import re
import sys
from collections import defaultdict, deque
from pathlib import Path

# `%memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {` / `aie.mem(%tile_0_2) {`
_RE_DMA_OP = re.compile(r"aie\.(memtile_dma|mem|shim_dma)\((%[\w]+)\)")
# `%0 = aie.dma_start(MM2S, 0, ^bb1, ^bb6)`
_RE_DMA_START = re.compile(r"aie\.dma_start\((MM2S|S2MM),\s*(\d+)")
# `aie.dma_bd(%buf18 : memref<2048xbf16, 1 : i32> offset = 0 len = 2048)`.
# The `: i32` on the memory space is present in placed (aircc) IR and absent in
# air-opt's pre-placement output, so it is optional here -- otherwise the probe
# silently reports "no buffers" on a lit-level module.
_RE_DMA_BD = re.compile(
    r"aie\.dma_bd\((%[\w]+)\s*:\s*memref<([^>]*?),\s*(\d+)\s*(?::\s*i32\s*)?>"
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
    r"(%[\w]+)\s*=\s*aie\.buffer\((%[\w]+)\).*?memref<([^>]*?),\s*(\d+)\s*(?::\s*i32\s*)?>"
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


# --------------------------------------------------------------------------
# --check-order: decide whether the emitted lock protocol FORCES the writer
# order, by exploring the reachable state space of the tile's DMA program.
# --------------------------------------------------------------------------

# `%lock_3_1_23 = aie.lock(%mem_tile_3_1, 1) {init = 2 : i32}`
_RE_LOCK_DECL = re.compile(
    r"(%[\w]+)\s*=\s*aie\.lock\((%[\w]+),\s*\d+\)"
    r"(?:\s*\{[^}]*init\s*=\s*(-?\d+))?"
)
_RE_BLOCK_LABEL = re.compile(r"^\s*(\^bb\d+):")
_RE_NEXT_BD = re.compile(r"aie\.next_bd\s+(\^bb\d+)")
_RE_DMA_START_FULL = re.compile(
    r"aie\.dma_start\((MM2S|S2MM),\s*(\d+),\s*(\^bb\d+),\s*(\^bb\d+)\)"
)


class _Bd:
    """One aie.dma_bd with its single acquire, single release and successor.

    Single acquire + single release is not a simplification: `generateDmaBd`
    emits exactly one of each, because an AIE2 BD descriptor carries exactly
    one acquire-lock field and one release-lock field.  That is the hardware
    budget the ordering argument in doc 52 is spent against.
    """

    __slots__ = ("label", "buf", "acq", "acqn", "rel", "reln", "nxt")

    def __init__(self, label):
        self.label = label
        self.buf = None
        self.acq = self.rel = None
        self.acqn = self.reln = None
        self.nxt = None


def parse_tile_programs(text):
    """{tile: {"locks": {sym: init}, "chans": [(dir, chan, [Bd, ...])]}}."""
    consts = {m.group(1): int(m.group(2)) for m in _RE_CONST.finditer(text)}

    lock_init = {}
    lock_tile = {}
    for m in _RE_LOCK_DECL.finditer(text):
        lock_init[m.group(1)] = int(m.group(3)) if m.group(3) is not None else 0
        lock_tile[m.group(1)] = m.group(2)

    tiles = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = _RE_DMA_OP.search(lines[i])
        if not m:
            i += 1
            continue
        tile = m.group(2)
        # Consume the region by brace depth.
        depth = 0
        body = []
        while i < len(lines):
            body.append(lines[i])
            depth += lines[i].count("{") - lines[i].count("}")
            i += 1
            if depth <= 0 and len(body) > 1:
                break

        blocks = {}
        starts = []
        cur = _Bd("^entry")
        blocks["^entry"] = cur
        pending_acq = None
        for line in body:
            lm = _RE_BLOCK_LABEL.match(line)
            if lm:
                cur = _Bd(lm.group(1))
                blocks[lm.group(1)] = cur
                pending_acq = None
                continue
            sm = _RE_DMA_START_FULL.search(line)
            if sm:
                starts.append((sm.group(1), int(sm.group(2)), sm.group(3)))
                continue
            um = _RE_USE_LOCK.search(line)
            if um:
                sym, action, cnt = um.group(1), um.group(2), um.group(3)
                n = consts.get(cnt)
                if n is None and cnt is not None and cnt.isdigit():
                    n = int(cnt)
                if n is None:
                    n = 1
                if action.startswith("Acquire"):
                    pending_acq = (sym, n)
                else:
                    cur.rel, cur.reln = sym, n
                continue
            bm = _RE_DMA_BD.search(line)
            if bm:
                cur.buf = bm.group(1)
                if pending_acq:
                    cur.acq, cur.acqn = pending_acq
                    pending_acq = None
                continue
            nm = _RE_NEXT_BD.search(line)
            if nm:
                cur.nxt = nm.group(1)

        chans = []
        for d, ch, first in starts:
            chain, seen, cursor = [], set(), first
            while cursor and cursor not in seen and cursor in blocks:
                seen.add(cursor)
                bd = blocks[cursor]
                if bd.buf is None:  # e.g. the shared ^bb with only aie.end
                    break
                chain.append(bd)
                cursor = bd.nxt
            if chain:
                chans.append((d, ch, chain))

        locks = {s: v for s, v in lock_init.items() if lock_tile.get(s) == tile}
        tiles[tile] = {"locks": locks, "chans": chans}
    return tiles


ORDER_UNKNOWN = "UNKNOWN"
# The DMA program is not the whole protocol on this tile: an `aie.core` region
# holds some of the same locks, and the net does not model it. Distinct from
# UNKNOWN, which means the simulation ran and could not decide.
ORDER_UNMODELLED = "UNMODELLED"

_RE_CORE_OP = re.compile(r"aie\.core\((%[\w]+)\)")


def parse_core_locks(text):
    """{tile: {lock sym, ...}} for every ``aie.core`` region.

    A lock in here is a participant the Petri net does NOT have a transition
    for, so any buffer whose tile program shares one is outside what
    ``check_write_order`` can decide.
    """
    out = defaultdict(set)
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = _RE_CORE_OP.search(lines[i])
        if not m:
            i += 1
            continue
        tile = m.group(1)
        depth, body = 0, []
        while i < len(lines):
            body.append(lines[i])
            depth += lines[i].count("{") - lines[i].count("}")
            i += 1
            if depth <= 0 and len(body) > 1:
                break
        for line in body:
            lm = _RE_USE_LOCK.search(line)
            if lm:
                out[tile].add(lm.group(1))
    return out


def check_write_order(prog, buf, depth=8, max_states=400000, slots=1):
    """Decide, for one buffer, whether the emitted lock protocol is sound.

    Two independent properties are checked over the reachable state space, and
    a single-slot buffer needs BOTH -- doc 52's two hazards:

      writer order   is the k-th writer of `buf` the same channel on every
                     interleaving?  Otherwise ``RACE``.
      read binding   on a single-slot buffer, is every fill consumed by every
                     reader channel exactly once before the next fill lands?
                     A fill destroyed early is ``OVERWRITE``; a reader taking
                     the same fill twice (and so missing the next one -- the
                     "reader takes a peer's token" hazard) is ``DOUBLE-READ``.

    The order property alone is NOT sufficient, which is the whole reason this
    checks both: a writers-then-readers daisy chain gives every writer a
    distinct predecessor -- perfectly ORDERED -- while still letting writer i+1
    overwrite the slot before any reader has touched writer i's fill.

    Returns (verdict, per_position_writer_sets, note).  Streams are modelled as
    always-ready: the question is whether the DMA PROGRAM ALONE forces the
    order, so anything a stream might happen to serialise is deliberately not
    credited.
    """
    chans = prog["chans"]
    lock_names = sorted(prog["locks"])
    lidx = {n: i for i, n in enumerate(lock_names)}
    init = tuple(prog["locks"][n] for n in lock_names)

    writers = [
        ci for ci, (d, _, chain) in enumerate(chans)
        if d == "S2MM" and any(bd.buf == buf for bd in chain)
    ]
    readers = [
        ci for ci, (d, _, chain) in enumerate(chans)
        if d == "MM2S" and any(bd.buf == buf for bd in chain)
    ]
    if not writers:
        return ORDER_UNKNOWN, [], "no S2MM writer of this buffer on the tile"

    rbit = {ci: 1 << i for i, ci in enumerate(readers)}
    full = (1 << len(readers)) - 1

    possible = [set() for _ in range(depth)]
    start = (init, tuple(0 for _ in chans), 0, 0)
    seen = {start}
    queue = deque([start])
    deadlock = None
    overwrite = None
    double_read = None
    truncated = False

    while queue:
        if len(seen) > max_states:
            truncated = True
            break
        marking, pcs, k, mask = queue.popleft()
        fired_any = False
        for ci, (d, _, chain) in enumerate(chans):
            bd = chain[pcs[ci] % len(chain)]
            if bd.acq is not None:
                ai = lidx.get(bd.acq)
                if ai is None or marking[ai] < bd.acqn:
                    continue
            fired_any = True
            nm = list(marking)
            if bd.acq is not None:
                nm[lidx[bd.acq]] -= bd.acqn
            if bd.rel is not None and bd.rel in lidx:
                nm[lidx[bd.rel]] += bd.reln
            npcs = list(pcs)
            npcs[ci] = (pcs[ci] + 1) % len(chain)
            nk, nmask = k, mask
            if bd.buf == buf and d == "S2MM":
                if k >= depth:
                    continue  # enough evidence; stop growing this branch
                possible[k].add(ci)
                # Read binding: on a single-slot buffer the PREVIOUS fill must
                # be fully consumed before this one lands.  With `slots` > 1
                # the writer legitimately runs ahead, so the check does not
                # apply -- ping-pong is a different (and sound) mechanism.
                if slots <= 1 and k > 0 and readers and mask != full:
                    if overwrite is None:
                        missing = [
                            f"{chans[c][0]} {chans[c][1]}"
                            for c in readers if not (mask & rbit[c])
                        ]
                        overwrite = (k, missing)
                nk, nmask = k + 1, 0
            elif bd.buf == buf and d == "MM2S":
                if slots <= 1 and k > 0 and (mask & rbit[ci]):
                    if double_read is None:
                        double_read = (k, f"{chans[ci][0]} {chans[ci][1]}")
                nmask = mask | rbit[ci]
            st = (tuple(nm), tuple(npcs), nk, nmask)
            if st not in seen:
                seen.add(st)
                queue.append(st)
        if not fired_any and k < depth and deadlock is None:
            deadlock = (marking, pcs, k)

    positions = [s for s in possible if s]
    if deadlock is not None:
        return (
            "DEADLOCK",
            positions,
            f"reachable state with no enabled BD after {deadlock[2]} write(s)",
        )
    if not positions:
        return ORDER_UNKNOWN, positions, "no write of this buffer was reachable"
    if any(len(s) > 1 for s in positions):
        bad = next(i for i, s in enumerate(positions) if len(s) > 1)
        names = ", ".join(
            f"{chans[ci][0]} {chans[ci][1]}" for ci in sorted(positions[bad])
        )
        return (
            "RACE",
            positions,
            f"write #{bad} can be performed by any of: {names}",
        )
    if overwrite is not None:
        k, missing = overwrite
        return (
            "OVERWRITE",
            positions,
            f"fill #{k - 1} can be overwritten before {', '.join(missing)} read it",
        )
    if double_read is not None:
        k, who = double_read
        return (
            "DOUBLE-READ",
            positions,
            f"{who} can consume fill #{k - 1} twice, so it misses fill #{k}",
        )
    if truncated:
        return ORDER_UNKNOWN, positions, "state-space bound hit before a verdict"
    return (
        "ORDERED",
        positions,
        f"{len(positions)} write position(s), each with exactly one writer, "
        f"every fill consumed by all {len(readers)} reader(s) before the next",
    )


def order_verdict(prog, tile_core_locks, buf, depth=8, slots=1):
    """The order verdict for one buffer, INCLUDING the validity guard.

    Factored out so `--self-test` exercises the same decision `analyze` makes:
    a guard that only the real path runs is a guard with no calibration.
    """
    shared = sorted(set(tile_core_locks or ()) & set(prog["locks"]))
    if shared:
        return (
            ORDER_UNMODELLED,
            f"aie.core holds {len(shared)} of this tile's DMA locks "
            f"({', '.join(shared[:3])}{', ...' if len(shared) > 3 else ''}); "
            "the net models the DMA program only, so a verdict here would be "
            "a false positive",
        )
    verdict, _, note = check_write_order(prog, buf, depth=depth, slots=slots)
    return verdict, note


def analyze(path, refuse_race=False, only_space=None, quiet=False,
            check_order=False, refuse_unordered=False, order_depth=8):
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

    want_order = check_order or refuse_unordered
    tile_progs = parse_tile_programs(text) if want_order else {}
    core_locks = parse_core_locks(text) if want_order else {}
    order_verdicts = {}

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

        # Semantic verdict: simulate the emitted lock protocol on the owning
        # tile and ask whether the k-th write is always by the same channel,
        # and whether every fill is consumed before the next lands.  Run for
        # every buffer that is both written and read -- the read-binding
        # hazard does not need a second writer.
        if want_order and w and r:
            owner = info.get("tile") or next(iter(w))[0]
            prog = tile_progs.get(owner)
            if prog is None:
                order_verdicts[buf] = (ORDER_UNKNOWN, "no DMA program parsed for tile")
            else:
                # The net has a transition per BD and none for the core. If the
                # core holds any of this tile's DMA locks, its releases are
                # missing and every hazard verdict here is a false positive.
                order_verdicts[buf] = order_verdict(
                    prog, core_locks.get(owner), buf,
                    depth=order_depth, slots=nslots)
            v, note = order_verdicts[buf]
            if v != "ORDERED" or not quiet:
                print(f"         [order] {v}: {note}")

    print(f"[verdict] {len(multi)} multi-writer buffer(s), {len(races)} of them "
          f"single-slot (unordered writer race): {races if races else 'none'}")
    if want_order:
        outside = sorted(
            b for b, (v, _) in order_verdicts.items() if v == ORDER_UNMODELLED
        )
        unordered = sorted(
            b for b, (v, _) in order_verdicts.items()
            if v not in ("ORDERED", ORDER_UNMODELLED)
        )
        print(f"[order] {len(order_verdicts) - len(outside)} written+read "
              f"buffer(s) simulated, {len(unordered)} not provably sound: "
              f"{unordered if unordered else 'none'}")
        print(f"[order] {len(outside)} buffer(s) OUTSIDE the model (core shares "
              f"the tile's DMA locks; not a hazard verdict): "
              f"{outside if outside else 'none'}")
    rc = 0
    if refuse_race and races:
        rc = 1
    # UNMODELLED is not a finding: gating on it would fail every module that has
    # an L1 buffer, which is every module.
    if refuse_unordered and any(
        v not in ("ORDERED", ORDER_UNMODELLED) for v, _ in order_verdicts.values()
    ):
        rc = 1
    return rc


# Synthetic memtile programs for --self-test. Each is a complete, parseable
# aie.memtile_dma over one L2 buffer, differing ONLY in the lock wiring, so the
# three verdicts are attributable to the protocol and nothing else.
_SELFTEST_HEAD = """
  %lk0 = aie.lock(%mem_tile_0_1, 0) {init = %I0 : i32}
  %lk1 = aie.lock(%mem_tile_0_1, 1) {init = %I1 : i32}
  %lk2 = aie.lock(%mem_tile_0_1, 2) {init = %I2 : i32}
  %lk3 = aie.lock(%mem_tile_0_1, 3) {init = %I3 : i32}
  %b = aie.buffer(%mem_tile_0_1) {sym_name = "b"} : memref<8xbf16, 1 : i32>
  %m = aie.memtile_dma(%mem_tile_0_1) {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
  ^bb1:
    aie.use_lock(%WA0, AcquireGreaterEqual, %WN0)
    aie.dma_bd(%b : memref<8xbf16, 1 : i32> offset = 0 len = 8)
    aie.use_lock(%WR0, Release, %WM0)
    aie.next_bd ^bb1
  ^bb2:
    %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
  ^bb3:
    aie.use_lock(%WA1, AcquireGreaterEqual, %WN1)
    aie.dma_bd(%b : memref<8xbf16, 1 : i32> offset = 0 len = 8)
    aie.use_lock(%WR1, Release, %WM1)
    aie.next_bd ^bb3
  ^bb4:
    %2 = aie.dma_start(MM2S, 0, ^bb5, ^bb6)
  ^bb5:
    aie.use_lock(%RA0, AcquireGreaterEqual, %RN0)
    aie.dma_bd(%b : memref<8xbf16, 1 : i32> offset = 0 len = 8)
    aie.use_lock(%RR0, Release, %RM0)
    aie.next_bd ^bb5
  ^bb6:
    %3 = aie.dma_start(MM2S, 1, ^bb7, ^bb8)
  ^bb7:
    aie.use_lock(%RA1, AcquireGreaterEqual, %RN1)
    aie.dma_bd(%b : memref<8xbf16, 1 : i32> offset = 0 len = 8)
    aie.use_lock(%RR1, Release, %RM1)
    aie.next_bd ^bb7
  ^bb8:
    aie.end
  }
"""

# Legacy counted template: both writers on the same cap/done pair, both readers
# likewise. This is what wall 7 actually emits.
_SELFTEST_LEGACY = {
    "I0": 2, "I1": 0, "I2": 0, "I3": 0,
    "WA0": "%lk0", "WN0": "%c2_i32", "WR0": "%lk1", "WM0": "%c2_i32",
    "WA1": "%lk0", "WN1": "%c2_i32", "WR1": "%lk1", "WM1": "%c2_i32",
    "RA0": "%lk1", "RN0": "%c1_i32", "RR0": "%lk0", "RM0": "%c1_i32",
    "RA1": "%lk1", "RN1": "%c1_i32", "RR1": "%lk0", "RM1": "%c1_i32",
}
# Two chains back to back (the mimo-chain-lock arm): cap -> W0 -> W1 -> R0 ->
# R1 -> cap. Orders the writers; strands the readers behind BOTH writers.
_SELFTEST_TWOCHAIN = {
    "I0": 1, "I1": 0, "I2": 0, "I3": 0,
    "WA0": "%lk0", "WN0": "%c1_i32", "WR0": "%lk1", "WM0": "%c1_i32",
    "WA1": "%lk1", "WN1": "%c1_i32", "WR1": "%lk2", "WM1": "%c1_i32",
    "RA0": "%lk2", "RN0": "%c1_i32", "RR0": "%lk3", "RM0": "%c1_i32",
    "RA1": "%lk3", "RN1": "%c1_i32", "RR1": "%lk0", "RM1": "%c1_i32",
}
# Sound reference: ONE writer, readers chained behind it (v2's fan-out shape),
# with the second writer channel parked on a lock nobody ever releases so the
# buffer still has two writer channels structurally. Writer order is forced and
# every fill is consumed before the next -- the verdict must be ORDERED.
_SELFTEST_FANOUT = {
    "I0": 1, "I1": 0, "I2": 0, "I3": 0,
    "WA0": "%lk0", "WN0": "%c1_i32", "WR0": "%lk1", "WM0": "%c1_i32",
    "WA1": "%lk3", "WN1": "%c2_i32", "WR1": "%lk3", "WM1": "%c1_i32",
    "RA0": "%lk1", "RN0": "%c1_i32", "RR0": "%lk2", "RM0": "%c1_i32",
    "RA1": "%lk2", "RN1": "%c1_i32", "RR1": "%lk0", "RM1": "%c1_i32",
}


# An aie.core region that touches ONE of the tile's DMA locks. Appended to the
# FANOUT case, whose verdict without it is ORDERED -- so the guard is calibrated
# in both directions on the same protocol: with no core the verdict stands, and
# adding a single shared lock must withdraw it.
_SELFTEST_CORE = """
  %c = aie.core(%mem_tile_0_1) {
    aie.use_lock(%lk1, AcquireGreaterEqual, %c1_i32)
    aie.use_lock(%lk2, Release, %c1_i32)
    aie.end
  }
"""


def _selftest_module(cfg, with_core=False):
    text = _SELFTEST_HEAD + (_SELFTEST_CORE if with_core else "")
    for k, v in cfg.items():
        text = text.replace(f"%{k}", str(v) if not str(v).startswith("%") else v)
    return text


def self_test():
    """The three verdicts must be attributable to the lock wiring alone.

    Same buffer, same channels, same BD chains in all three -- only the lock
    identities and counts differ. If any of these stops discriminating, the
    instrument is reporting its own shape rather than the protocol.
    """
    cases = [
        ("legacy counted (wall 7 as shipped)", _SELFTEST_LEGACY, False, "RACE"),
        ("two chains (mimo-chain-lock arm)", _SELFTEST_TWOCHAIN, False, "OVERWRITE"),
        ("single writer + reader chain", _SELFTEST_FANOUT, False, "ORDERED"),
        # Same protocol as the line above, plus an aie.core holding one of the
        # tile's locks. The verdict must be WITHDRAWN, not kept and not turned
        # into a hazard -- the net has no transition for the core.
        ("+ a core sharing one lock", _SELFTEST_FANOUT, True, ORDER_UNMODELLED),
        # And the guard must not fire on a protocol it does not apply to: the
        # racing legacy template still reads RACE when no core is present.
        ("legacy counted, core absent (guard must not fire)",
         _SELFTEST_LEGACY, False, "RACE"),
    ]
    ok = True
    for name, cfg, with_core, want in cases:
        text = _selftest_module(cfg, with_core=with_core)
        progs = parse_tile_programs(text)
        prog = progs.get("%mem_tile_0_1")
        if prog is None:
            print(f"[self-test] FAIL {name}: synthetic module did not parse")
            ok = False
            continue
        core = parse_core_locks(text).get("%mem_tile_0_1")
        if with_core and not core:
            print(f"[self-test] FAIL {name}: the core region did not parse, so "
                  "this case would pass for the wrong reason")
            ok = False
            continue
        got, note = order_verdict(prog, core, "%b", depth=6, slots=1)
        mark = "ok  " if got == want else "FAIL"
        if got != want:
            ok = False
        print(f"[self-test] {mark} {name}: want {want}, got {got} ({note})")
    print(f"[self-test] {'PASS' if ok else 'FAILED'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mlir", nargs="*", help="aie.air.mlir dump(s)")
    ap.add_argument("--self-test", action="store_true",
                    help="run the order checker against three synthetic "
                         "memtile programs that differ only in lock wiring")
    ap.add_argument("--refuse-race", action="store_true",
                    help="exit 1 if any single-slot multi-writer buffer exists "
                         "(STRUCTURAL -- unchanged, so recorded numbers compare)")
    ap.add_argument("--check-order", action="store_true",
                    help="simulate the emitted lock protocol and report whether "
                         "the writer order is forced (ORDERED/RACE/DEADLOCK)")
    ap.add_argument("--refuse-unordered", action="store_true",
                    help="exit 1 unless every multi-writer buffer is provably "
                         "ORDERED by the simulation")
    ap.add_argument("--order-depth", type=int, default=8,
                    help="how many write positions to decide (default 8)")
    ap.add_argument("--space", type=int, default=None,
                    help="restrict to a memory space (1 = L2, 2 = L1)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.mlir:
        ap.error("give at least one aie.air.mlir, or --self-test")
    rc = 0
    for p in a.mlir:
        rc |= analyze(p, a.refuse_race, a.space, a.quiet,
                      check_order=a.check_order,
                      refuse_unordered=a.refuse_unordered,
                      order_depth=a.order_depth)
        print()
    return rc


if __name__ == "__main__":
    sys.exit(main())
