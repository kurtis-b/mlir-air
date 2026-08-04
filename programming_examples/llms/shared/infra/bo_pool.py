# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Buffer-object liveness pooling and dirty-bit synchronization for `KernelCache`.

A dispatch sequence names the same logical tensor at several steps. Allocating one
`xrt.bo` per (kernel, argument) wastes DDR proportional to the sequence length, and
syncing every buffer in both directions on every step wastes wall time proportional
to the bytes. This module fixes both: `plan_pool` computes live ranges over the
sequence and assigns non-overlapping same-shaped buffers to one pool slot, and
`BoPool` tracks a dirty bit per BO so only written buffers go host->device and only
declared outputs come back.

The rules this implements are written down in
`docs/plans/transformer-layer-execution-studies/05b-phase-b-buffer-rules.md`.
Read that before changing anything here; the rule identifiers (C2, D5, L3, ...) in
the comments below refer to its sections.

Footguns, in the order they will bite you:

- **A pooled BO is bigger than the buffer in it.** Slots are 4 KiB-rounded bins (O3).
  Build every readback view with `count=<logical elements>`, never `bo.size()`.
  Reading the bin size returns whatever the previous occupant left behind.

- **Returned arrays are zero-copy views into pool memory** (H1). The next step that
  reuses that slot overwrites them. Declare a buffer `host_output=True` if the host
  must read it after the sequence — that pins it live to the end (L3) so nothing
  takes its slot. If you need it past the *sequence*, copy it (H3).

- **Under the xclbin ABI a slot is keyed by (kernel, argument index)** (C2), because
  `kernel.group_id(i + 3)` decides the memory bank. Only ELF-ABI buffers
  (`xrt.ext.bo`, no group id) may be pooled across kernels. A size-only allocator
  silently produces cross-bank aliasing here.

- **Reassigning a slot marks the new occupant dirty** (D5). The bytes in it belong to
  the previous buffer. Skipping that sync is how pooling produces stale reads that
  look like a numerics bug in the kernel.

- **Static buffers are content-keyed and never pooled with transient ones** (S1, C6).
  Two operators passing identical weight bytes share one BO, written once. Writing
  through a static slot would corrupt every other operator sharing that content, so
  a static buffer may not be a step's output (A3).

- **A buffer a step writes is not necessarily one the device produces** (D7). An
  in-place buffer is read *and* written, so its bytes come from the host and it is
  live from before the sequence — `start = -1`, not the index of the step that
  writes it. Treating it as produced both skips its upload and lets an
  earlier-dying buffer take its slot. `host_supplied_names` draws that line, and
  states the one case the declaration cannot decide.

- **Content keying is subordinate to the bank rule** (S5). Under the xclbin ABI two
  identical weights sitting at different `(kernel, argument index)` positions get
  *different* BOs, because the position is what picks the memory group. Deduping
  them would hand the second kernel a buffer banked for the first.

- **A pool is reused by sequence identity, never by object identity** (O5).
  `PoolPlan.signature` is that identity. `id(plan)` is not: every dispatch builds a
  fresh plan, so keying on it both defeats reuse and lets CPython hand a recycled
  id's pool to an unrelated plan.
"""

import hashlib
from dataclasses import dataclass, field

PAGE_BYTES = 4096


def _bin_size(nbytes):
    """4 KiB-rounded pool bin for a logical size (rule C3)."""
    return -(-int(nbytes) // PAGE_BYTES) * PAGE_BYTES


@dataclass(frozen=True)
class BufferSpec:
    """One logical tensor in a dispatch sequence.

    Attributes:
        name: Stable identity across the sequence. Two steps naming the same
            `name` mean the same tensor (L5/A2); two different names must never
            resolve to one BO within a step (A1).
        nbytes: Logical size. The pool slot is `_bin_size(nbytes)` and may be
            larger; readback must use the logical size (O3).
        static: Weight-like. Goes to the content-keyed pool, written once,
            never re-synced (S1, S2), never a step output (A3).
        host_output: The host reads this after the sequence, so it stays live to
            the end and keeps its slot to itself (L3, H2).
        content_key: Set for static buffers; `sha256:<hex>` of the bytes.
    """

    name: str
    nbytes: int
    static: bool = False
    host_output: bool = False
    content_key: str = None


@dataclass(frozen=True)
class DispatchStep:
    """One kernel invocation in a dispatch sequence.

    `args` is positional and matches the kernel's argument order. `writes` holds
    the *indices into args* the kernel writes; those buffers get this step as
    their producer (L1) and are marked device-dirty afterwards (D4).
    """

    kernel: str
    args: tuple
    writes: tuple = ()

    def written_names(self):
        return tuple(self.args[i] for i in self.writes)


@dataclass
class PoolPlan:
    """Result of `plan_pool`: which slot backs which buffer, and why.

    `slot_of` maps buffer name -> slot key. `live` maps buffer name ->
    (start, end) for inspection and tests. `bins` maps slot key -> allocation
    size in bytes. `static_of` maps static buffer name -> content key.

    `slot_positions` maps slot key -> the `(kernel, arg_index)` positions its
    buffers are bound at, in first-use order. Under the xclbin ABI that is what
    picks the memory group, so the allocator reads it rather than guessing from
    the first use (C2).

    `signature` is the plan's *value* identity: two plans built from the same
    steps, specs and ABI compare equal. `KernelCache` keys its pools on it so a
    repeated sequence reuses its BOs and its already-synced static weights
    (O5). Never key a pool on `id(plan)`; see the module docstring.
    """

    slot_of: dict = field(default_factory=dict)
    live: dict = field(default_factory=dict)
    bins: dict = field(default_factory=dict)
    static_of: dict = field(default_factory=dict)
    slot_positions: dict = field(default_factory=dict)
    signature: tuple = ()

    def n_slots(self):
        return len(self.bins)

    def footprint_bytes(self):
        return sum(self.bins.values())


def host_supplied_names(steps):
    """Buffers whose bytes come from the host rather than from a step (rule D7).

    A buffer is host-supplied when some step reads it before any step has
    written it: its first appearance at an argument position *not* in that
    step's `writes` is at or before the first step that writes it. That covers a
    plain input, which no step writes at all, and it covers the in-place buffer
    A2 asks for — one identity at a read position and a write position of the
    same step, where the read sees whatever the host left there.

    The case the declaration cannot decide: a buffer appearing **only** at
    written positions. `writes` records that the kernel writes an argument;
    nothing says whether it reads it first, so a single-position
    read-modify-write is indistinguishable from a plain output, and the plain
    output is much the commoner of the two. Those are reported as produced. A
    caller updating a buffer in place at a single argument position declares it
    either A2's way — pass the identity at a read position as well — or by
    naming it in `run_sequence`'s `host_writes`.
    """
    first_read = {}
    first_write = {}
    for idx, step in enumerate(steps):
        for i, name in enumerate(step.args):
            table = first_write if i in step.writes else first_read
            table.setdefault(name, idx)
    return {
        name
        for name, read_idx in first_read.items()
        if name not in first_write or read_idx <= first_write[name]
    }


def _host_supplied_set(steps, specs, host_supplied):
    """Canonical host-supplied set for a plan: derived when not given.

    Restricted to the non-static buffers the steps actually name, because those
    are the only ones it can move: statics are outside the liveness analysis
    (S3) and a name no step names has no slot. Canonicalizing here keeps
    `plan_signature` from separating two identical plans over an entry that
    could not have changed the assignment.
    """
    if host_supplied is None:
        host_supplied = host_supplied_names(steps)
    named = {n for s in steps for n in s.args}
    return frozenset(n for n in host_supplied if n in named and not specs[n].static)


def compute_live_ranges(steps, specs, host_supplied=None):
    """Live range per non-static buffer, as (start, end) step indices.

    start = -1 for a host-supplied buffer — one whose bytes the host writes
    before the sequence, so it is live from before it begins — and otherwise the
    index of the first step that writes it, its producer (L1). Note that being
    written by a step does not make a buffer produced: an in-place buffer is
    both read and written, and starting it at its writing step would let an
    earlier-dying buffer take a slot whose host bytes are still needed. Which
    buffers those are is `host_supplied_names`' decision (D7); pass
    `host_supplied` to override it, as `run_sequence` does when the caller
    declares `host_writes` by hand.

    end   = index of the last step naming it as any argument — L2 — raised to
    len(steps) for declared host outputs so the host read cannot race a slot
    reuse — L3.
    """
    supplied = _host_supplied_set(steps, specs, host_supplied)
    first_write = {}
    last_use = {}
    for idx, step in enumerate(steps):
        for name in step.args:
            last_use[name] = idx
        for name in step.written_names():
            first_write.setdefault(name, idx)

    live = {}
    for name, use_idx in last_use.items():
        spec = specs[name]
        if spec.static:
            continue
        start = -1 if name in supplied else first_write.get(name, -1)
        end = len(steps) if spec.host_output else use_idx
        live[name] = (start, end)
    return live


def _overlaps(a, b):
    """Rule L4."""
    return not (a[1] < b[0] or b[1] < a[0])


def _positions(steps):
    """Every `(kernel, arg_index)` each buffer is bound at, in first-use order.

    Under the xclbin ABI this is the memory group (C2); under the ELF ABI it is
    only bookkeeping. A buffer with more than one entry here cannot be pooled at
    a single position, and — if it is static — cannot be content-deduped with a
    buffer at a different one (S5).
    """
    out = {}
    for step in steps:
        for i, name in enumerate(step.args):
            seen = out.setdefault(name, [])
            if (step.kernel, i) not in seen:
                seen.append((step.kernel, i))
    return {name: tuple(v) for name, v in out.items()}


def plan_signature(steps, specs, elf_abi, host_supplied=None):
    """Value identity of the plan `plan_pool` would build from these inputs (O5).

    Two sequences with this signature produce the same slot assignment, the same
    bin sizes and the same bank choices, so one `BoPool` may serve both — which
    is what makes a repeated sequence cheap: its BOs are already allocated and
    its static weights already resident.

    Everything the assignment depends on is in here. `content_key` is included
    because it keys the static pool: a caller that changes a weight's bytes and
    its key gets a fresh pool rather than a silently stale BO (S2). The
    host-supplied set is included because it sets each buffer's live-range start
    (D7/L1): two sequences that differ only in which buffers the host writes
    have different slot assignments and must not share a pool.
    """
    names = sorted({a for s in steps for a in s.args})
    return (
        "elf" if elf_abi else "xclbin",
        tuple((s.kernel, tuple(s.args), tuple(s.writes)) for s in steps),
        tuple(
            (
                n,
                specs[n].nbytes,
                specs[n].static,
                specs[n].host_output,
                specs[n].content_key,
            )
            for n in names
        ),
        tuple(sorted(_host_supplied_set(steps, specs, host_supplied))),
    )


def plan_pool(steps, specs, elf_abi, host_supplied=None):
    """Assign pool slots to the sequence's buffers.

    Args:
        steps: ordered list of `DispatchStep`.
        specs: dict name -> `BufferSpec`, covering every name the steps use.
        elf_abi: True for the ELF ABI (`xrt.ext.bo`, no group id, so a slot may
            back different kernels), False for the xclbin ABI (slot keyed by
            (kernel, arg index) because `group_id` picks the bank) — rule C2.
        host_supplied: buffers the host writes before the sequence, which are
            live from before it (D7/L1). Defaults to what
            `host_supplied_names` derives from the steps; pass the caller's own
            set when it declares one, or an in-place buffer the derivation
            cannot see will lose its slot to an earlier-dying buffer.

    Returns:
        `PoolPlan`.

    Raises:
        KeyError: a step names a buffer with no spec.
        ValueError: a static buffer is written by a step (A3), or a step names
            the same distinct buffer under two identities in a way that cannot
            be honoured.
    """
    for step in steps:
        for name in step.args:
            if name not in specs:
                raise KeyError(
                    f"step {step.kernel!r} names buffer {name!r} with no BufferSpec"
                )
        for name in step.written_names():
            if specs[name].static:
                raise ValueError(
                    f"step {step.kernel!r} writes static buffer {name!r}; "
                    "static buffers are content-keyed and shared (rule A3)"
                )

    supplied = _host_supplied_set(steps, specs, host_supplied)
    plan = PoolPlan(signature=plan_signature(steps, specs, elf_abi, supplied))
    positions = _positions(steps)

    # Under the xclbin ABI a buffer is only poolable at one (kernel, arg index),
    # so a buffer used at two positions is not poolable at all (C2).
    unpoolable = set()
    if not elf_abi:
        unpoolable = {n for n, pos in positions.items() if len(pos) > 1}
    position_of = {n: pos[0] for n, pos in positions.items()}

    # Static buffers: content-keyed, outside the liveness analysis (S1, S3).
    #
    # Content keying is subordinate to the bank rule (S5). Under the xclbin ABI
    # the memory group comes from the (kernel, arg index) the buffer is bound
    # at, so two identical weights at different positions must NOT collapse onto
    # one BO: `_alloc` would bank it for whichever position it saw first and the
    # other kernel would read it through the wrong group. Putting the position
    # in the slot key is what keeps dedup inside one bank. A static buffer bound
    # at several positions has to be one BO whatever its content, so it is
    # pinned to its own slot and the group agreement is checked at allocation.
    for name, spec in specs.items():
        # A spec no step names has nothing to bind to: no position, hence no
        # bank, hence no slot. `specs` is allowed to carry more than the
        # sequence uses, and those entries are simply not part of the plan.
        if not spec.static or name not in positions:
            continue
        key = spec.content_key or f"anon:{name}"
        binsz = _bin_size(spec.nbytes)
        pos = positions[name]
        if elf_abi:
            slot = ("static", key, binsz)
        elif len(pos) > 1:
            slot = ("static-pinned", name, binsz)
        else:
            slot = ("static", key, pos[0], binsz)
        plan.slot_of[name] = slot
        plan.static_of[name] = key
        plan.bins.setdefault(slot, binsz)

    plan.live = compute_live_ranges(steps, specs, supplied)

    # Buffers appearing together in one step conflict regardless of live-range
    # arithmetic, so two distinct tensors never land on one BO (A1).
    same_step = {}
    for step in steps:
        names = [n for n in dict.fromkeys(step.args) if n in plan.live]
        for name in names:
            same_step.setdefault(name, set()).update(n for n in names if n != name)

    # Deterministic order: by first-write, then by name (O4).
    order = sorted(plan.live, key=lambda n: (plan.live[n][0], n))
    occupied = {}  # slot key -> list of buffer names already in it
    for name in order:
        spec = specs[name]
        binsz = _bin_size(spec.nbytes)
        family = ("elf", binsz) if elf_abi else ("xclbin", position_of[name], binsz)

        if name in unpoolable:
            slot = ("pinned", name, binsz)
            plan.slot_of[name] = slot
            plan.bins[slot] = binsz
            continue

        chosen = None
        idx = 0
        while True:
            slot = family + (idx,)
            residents = occupied.get(slot, [])
            clash = any(
                other in same_step.get(name, ())
                or _overlaps(plan.live[name], plan.live[other])
                for other in residents
            )
            if not clash:
                chosen = slot
                break
            idx += 1
        occupied.setdefault(chosen, []).append(name)
        plan.slot_of[name] = chosen
        plan.bins[chosen] = binsz

    # Every position each slot is bound at, for the allocator's bank check (C2).
    # A slot with more than one position under the xclbin ABI is a buffer the
    # sequence forces to be shared; the allocator confirms the groups agree
    # rather than silently taking the first.
    for name, slot in plan.slot_of.items():
        for pos in positions.get(name, ()):
            bound = plan.slot_positions.setdefault(slot, [])
            if pos not in bound:
                bound.append(pos)
    plan.slot_positions = {k: tuple(v) for k, v in plan.slot_positions.items()}

    return plan


class BoPool:
    """Owns the `xrt.bo` objects for a plan and tracks their dirty bits.

    One pool per `KernelCache` (O1), bound to that cache's device (O2). `alloc`
    is injected so the pool has no XRT import and stays unit-testable off
    hardware: it is called as `alloc(nbytes, slot)` and must return an object
    exposing `map()`, `sync(direction)` and `size()`.
    """

    def __init__(self, alloc, sync_to_device, sync_from_device):
        self._alloc = alloc
        self._to_device = sync_to_device
        self._from_device = sync_from_device
        self._slots = {}  # slot key -> bo
        self._occupant = {}  # slot key -> buffer name currently in it
        self._dirty = {}  # buffer name -> needs host->device sync
        # Static slots already synced (S2). Keyed by *slot*, not by content key:
        # under the xclbin ABI one content key can span several slots, one per
        # memory group (S5), and each of those BOs needs its own write.
        self._static_written = set()

    def bo_for(self, name, plan):
        """The BO backing `name`, allocating its slot on first use.

        Reassigning a slot to a different buffer marks the new occupant dirty
        (D5): the bytes in the slot belong to the previous occupant.
        """
        slot = plan.slot_of[name]
        bo = self._slots.get(slot)
        if bo is None:
            bo = self._alloc(plan.bins[slot], slot)
            self._slots[slot] = bo
        if self._occupant.get(slot) != name:
            self._occupant[slot] = name
            if slot[0] != "static":
                self._dirty[name] = True
        return bo

    def mark_written_by_host(self, name):
        """Rule D1."""
        self._dirty[name] = True

    def mark_written_by_device(self, name):
        """Rule D4. The device owns the bytes now, so a host->device sync would
        destroy them; clear the dirty bit rather than leaving it set."""
        self._dirty[name] = False

    def is_dirty(self, name):
        return self._dirty.get(name, False)

    def is_static_resident(self, name, plan):
        """True once this static buffer's BO holds its bytes (S2).

        Callers skip the host-side copy as well as the sync: with pools reused
        across dispatches (O5) a static weight is written on the first sequence
        and must not be re-copied on every one after it.
        """
        return plan.slot_of[name] in self._static_written

    def sync_to_device_if_needed(self, name, plan, specs):
        """Rule D2, plus S2 for static buffers. Returns True if a sync ran."""
        spec = specs[name]
        if spec.static:
            slot = plan.slot_of[name]
            if slot in self._static_written:
                return False
            self._to_device(self.bo_for(name, plan))
            self._static_written.add(slot)
            return True
        if not self._dirty.get(name, False):
            return False
        self._to_device(self.bo_for(name, plan))
        self._dirty[name] = False
        return True

    def sync_from_device(self, name, plan):
        """Rule D3. Only ever called for declared outputs."""
        self._from_device(self.bo_for(name, plan))
        self._dirty[name] = False

    def stats(self):
        return {
            "slots": len(self._slots),
            "bytes": sum(bo.size() for bo in self._slots.values()),
        }


def content_key(buf):
    """Content key for a static buffer (rule S1).

    `buf` is a numpy array, `bytes`, a mapped BO — anything exposing either the
    buffer protocol or `tobytes()`. Hashing reads the whole buffer, which is why
    it is opt-in per buffer rather than automatic (S4).

    The key is over **bytes**, never over the dtype or shape: the BO holds bytes,
    so two arrays that would upload identically must key identically. A caller
    that needs two same-byte tensors kept apart gives them distinct names, not a
    distinct dtype.

    `memoryview` refuses ml_dtypes' `bfloat16` outright — it is a numpy extension
    dtype whose format code ('E') has no buffer-protocol meaning — and bf16 weight
    tensors are precisely what this is called on, so that fallback is the common
    path here rather than an edge case.
    """
    try:
        mv = memoryview(buf)
    except (TypeError, ValueError):
        if not hasattr(buf, "tobytes"):
            raise
        mv = memoryview(buf.tobytes())
    if not mv.c_contiguous:
        mv = memoryview(bytes(mv))
    return "sha256:" + hashlib.sha256(mv.cast("B")).hexdigest()
