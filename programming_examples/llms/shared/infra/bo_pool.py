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
    """

    slot_of: dict = field(default_factory=dict)
    live: dict = field(default_factory=dict)
    bins: dict = field(default_factory=dict)
    static_of: dict = field(default_factory=dict)

    def n_slots(self):
        return len(self.bins)

    def footprint_bytes(self):
        return sum(self.bins.values())


def compute_live_ranges(steps, specs):
    """Live range per non-static buffer, as (start, end) step indices.

    start = index of the first step that writes the buffer, or -1 when no step
    writes it (a host-supplied input, live from before the sequence) — L1.
    end   = index of the last step naming it as any argument — L2 — raised to
    len(steps) for declared host outputs so the host read cannot race a slot
    reuse — L3.
    """
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
        start = first_write.get(name, -1)
        end = len(steps) if spec.host_output else use_idx
        live[name] = (start, end)
    return live


def _overlaps(a, b):
    """Rule L4."""
    return not (a[1] < b[0] or b[1] < a[0])


def plan_pool(steps, specs, elf_abi):
    """Assign pool slots to the sequence's buffers.

    Args:
        steps: ordered list of `DispatchStep`.
        specs: dict name -> `BufferSpec`, covering every name the steps use.
        elf_abi: True for the ELF ABI (`xrt.ext.bo`, no group id, so a slot may
            back different kernels), False for the xclbin ABI (slot keyed by
            (kernel, arg index) because `group_id` picks the bank) — rule C2.

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

    plan = PoolPlan()

    # Static buffers: content-keyed, outside the liveness analysis (S1, S3).
    for name, spec in specs.items():
        if not spec.static:
            continue
        key = spec.content_key or f"anon:{name}"
        slot = ("static", key, _bin_size(spec.nbytes))
        plan.slot_of[name] = slot
        plan.static_of[name] = key
        plan.bins.setdefault(slot, _bin_size(spec.nbytes))

    plan.live = compute_live_ranges(steps, specs)

    # Under the xclbin ABI a buffer is only poolable at one (kernel, arg index),
    # so a buffer used at two positions is not poolable at all (C2).
    position_of = {}
    unpoolable = set()
    if not elf_abi:
        for step in steps:
            for i, name in enumerate(step.args):
                pos = (step.kernel, i)
                if position_of.setdefault(name, pos) != pos:
                    unpoolable.add(name)

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
        self._static_written = set()  # content keys already synced (S2)

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

    def sync_to_device_if_needed(self, name, plan, specs):
        """Rule D2, plus S2 for static buffers. Returns True if a sync ran."""
        spec = specs[name]
        if spec.static:
            key = plan.static_of[name]
            if key in self._static_written:
                return False
            self._to_device(self.bo_for(name, plan))
            self._static_written.add(key)
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

    `buf` is anything `memoryview` accepts — a numpy array's buffer, bytes, or a
    mapped BO. Hashing reads the whole buffer, which is why it is opt-in per
    buffer rather than automatic (S4).
    """
    mv = memoryview(buf)
    if not mv.c_contiguous:
        mv = memoryview(bytes(mv))
    return "sha256:" + hashlib.sha256(mv.cast("B")).hexdigest()
