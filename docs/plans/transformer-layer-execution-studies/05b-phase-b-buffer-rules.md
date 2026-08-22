# 05b — Phase B: buffer ownership, synchronization, bank and aliasing rules

[05-phase-b-runtime-seam.md](01-original-plan-superseded.md) work item 2: *"Define buffer ownership,
host/device synchronization rules, bank compatibility and aliasing rules before implementing."*

This is that definition. `shared/infra/bo_pool.py` implements it and its module docstring points
back here. A rule that is not in this list is not enforced.

Throughout, **step** means one entry of a dispatch sequence (one kernel invocation), indexed from
0, and **buffer** means one logical tensor with a stable identity across the sequence.

---

## 1. Ownership

**O1.** Every `xrt.bo` is owned by exactly one `BoPool`, which is owned by exactly one
`KernelCache`. Nothing else allocates BOs on that cache's device.

**O2.** A pool is bound to one XRT device. A BO never crosses devices. `KernelCache` holds one
device, so this is structural, not checked per call.

**O3.** A pool slot's allocation size is its 4 KiB-rounded bin, which is ≥ every logical buffer
assigned to it. **The logical size is carried separately.** Readback constructs its numpy view
with `count=<logical elements>`; it never uses `bo.size()`. Getting this wrong reads adjacent
pool bytes and is the single easiest way to produce plausible garbage.

**O4.** Pool slots are handed out deterministically, lowest free index first, so a given dispatch
sequence always produces the same assignment. Reproducibility matters more here than packing
quality: a measurement that changes because the allocator changed its mind is not a measurement.

**O5 — a pool is reused by sequence identity, never by object identity.** `PoolPlan.signature`
is the value identity of (steps, buffer specs, ABI); `KernelCache` keys its pools on it. Every
dispatch builds a fresh `PoolPlan`, so keying on `id(plan)` would allocate a new pool and
re-upload every static weight on each call — and CPython recycles the freed plan's id, so it
could instead hand an old pool to an unrelated sequence whose slots are sized and banked for
buffers that are no longer there. Reuse is the point: the second and later runs of a sequence
skip both the allocation and the weight upload.

---

## 2. Compatibility — when two buffers may share a slot

Two buffers may share a pool slot **only if all six hold**:

**C1 — same allocator class.** ELF-ABI buffers are `xrt.ext.bo(device, size)`. xclbin-ABI buffers
are `xrt.bo(device, size, host_only, kernel.group_id(i + 3))`. The two are never interchangeable
and are binned separately.

**C2 — same memory group / bank.** Under the xclbin ABI the group id is a function of *(kernel,
argument index)*. Therefore an xclbin-ABI slot is keyed by `(kernel_name, arg_index)` and can
only be reused at that same position. Cross-kernel pooling is legal **only** on the ELF ABI,
where `xrt.ext.bo` carries no group id. This is the rule that a naive size-bin allocator breaks.

C2 binds the static pool as well as the transient one — see S5. Where a sequence *forces* one BO
across several positions (a buffer one kernel writes and another reads), the plan records every
position the slot is bound at and the allocator confirms they resolve to the same group, raising
if they do not rather than banking for whichever it saw first.

**C3 — same size bin.** Bins are `ceil(size / 4096) * 4096`. Buffers in different bins never
share, even when one would fit in the other — a larger slot holding a smaller buffer makes O3's
logical-size bookkeeping load-bearing in more places than is worth it.

**C4 — disjoint live ranges.** See §3.

**C5 — same context reachability.** A slot may back steps in different artifacts' contexts only
under the ELF ABI, where `xrt.ext.bo` is device-scoped rather than context-scoped. Every BO in a
pool is allocated against one device wrapper (O2), not against whichever artifact's backend
happened to name the buffer first. `runlist_gate.py` legs A and C are what keep this rule honest:
leg A shares one BO set across three artifacts' kernels in one runlist, leg C does it through the
pool. If either ever fails, C5 tightens to "same context" and cross-artifact pooling is off.

**C6 — neither is static.** Static buffers live in their own pool (§5) and are never shared with
transient buffers, in either direction.

---

## 3. Live ranges

**L1.** For buffer `b`, `start(b) = -1` if `b` is host-supplied (D7), so it is live from before
the sequence begins; otherwise `start(b)` is the index of the first step that writes `b`, its
producer. Being written by a step is not what decides this — an in-place buffer is written *and*
host-supplied, and starting it at its writing step would let a buffer that dies earlier take the
slot its host bytes are sitting in.

**L2.** `end(b)` is the index of the last step that names `b` as any argument.

**L3.** If `b` is a declared host output, `end(b) = len(steps)`. It stays live past the last
kernel so that the host read is not racing a slot reuse. This is the rule that makes the
zero-copy return views in `cache.py` safe; see §6.

**L4.** Buffers `b` and `c` conflict iff their ranges overlap:
`not (end(b) < start(c) or end(c) < start(b))`. Conflicting buffers never share a slot.

**L5.** A buffer that a step both reads and writes (an in-place argument, or the same buffer
passed at two argument positions) contributes to that step exactly as any other appearance:
`end` is at least that step, and `start` is `-1` because its incoming bytes come from the host
(L1/D7) rather than from a producer. No special case is needed in the allocator, but the aliasing
must be *declared* — see §4 and D7.

---

## 4. Aliasing

**A1.** Two distinct logical buffers passed to the same step must never resolve to the same BO.
The allocator enforces this by treating every buffer appearing in a step as mutually conflicting
for that step, regardless of live-range arithmetic.

**A2.** Passing one logical buffer at two argument positions of the same step is legal and means
in-place. It is expressed by using the same buffer identity twice, never by two identities that
happen to land on one slot.

**A3.** A step's output buffer must not be a static buffer. Writing through a content-keyed slot
would corrupt every other operator sharing that content.

---

## 5. Static (weight) buffers

**S1.** A buffer declared static is placed in a content-keyed pool: the key is
`(sha256(bytes), nbytes)`. Two operators passing identical weight bytes get one BO.

**S2.** A static BO is written and synced host→device exactly once, on first use, and is never
re-synced — including across dispatches, since pools outlive a sequence (O5). "Once" is counted
per **BO**, not per content key: S5 can split one content key across several BOs, and each of
those still has to be written. Its dirty bit is cleared at that point and never set again.

**S5 — content keying is subordinate to the bank rule.** Under the xclbin ABI the static slot key
carries the `(kernel, arg_index)` position as well as the content key, so two identical weights
bound at different positions get different BOs. Deduping them would give the pair one BO, banked
from whichever position the allocator saw first, and the other kernel would read its weights
through the wrong memory group — no error, wrong numbers. Under the ELF ABI there is no group id
and S1's dedup applies unrestricted. A static buffer bound at several positions gets a pinned
slot of its own and is checked for group agreement at allocation (C2).

**S3.** Static BOs are pinned for the lifetime of the `KernelCache`. They are not part of the
liveness analysis and never enter the transient pool (C6).

**S4.** Content keying reads the whole buffer to hash it. That cost is paid once per distinct
weight, at setup, and is why hashing is opt-in per buffer rather than automatic.

---

## 6. Host/device synchronization — the dirty-bit discipline

Ported in concept from `iron/common/aie_base.py`. Without it, latency numbers are not comparable
to iron's, because iron's pre-fix behaviour — syncing every BO in both directions on every run —
is exactly what the port must not reproduce.

**D1.** Every BO carries `dirty_to_device`. A host write sets it.

**D2.** Before a step, an argument syncs host→device **only if** its dirty bit is set. After the
sync the bit clears.

**D3.** After a submission, **only declared outputs** sync device→host. Not every argument, not
the last argument by convention.

**D4.** A device write invalidates the host view: after a step, each of its output buffers is
marked device-dirty, and the host may not read it before a D3 sync.

**D5 — the rule that makes pooling safe.** When a slot is reassigned from buffer `b` to buffer
`c`, `c` starts dirty. The slot's contents belong to `b` and are meaningless for `c`; skipping
that sync is how a pooling allocator produces stale reads.

**D6.** Instruction BOs (xclbin ABI only) sync once per BO identity, tracked by `id()`, never per
call.

**D7 — which buffers the host supplies.** A buffer is host-supplied iff some step reads it before
any step has written it: its first appearance at an argument position *not* in that step's
`writes` is at or before the first step that writes it. Those are the buffers D1 uploads and L1
starts at `-1`. A buffer only the device writes gets no upload: it has no host bytes, and pushing
its slot to the device before its producer runs would send the previous occupant's bytes to a
kernel that is about to overwrite them.

The classification is exact for a plain input and for A2's in-place form — one identity at a read
position and a write position of the same step. It **cannot** decide a buffer that appears only at
a written position: `writes` says the kernel writes that argument and nothing says whether it
reads it first, so a single-position read-modify-write is indistinguishable from a plain output,
which is much the commoner case. Those count as produced, and a caller doing such an update
declares it — either A2's way, or by naming the buffer in `run_sequence`'s `host_writes`, which
uploads it and pins it live from before the sequence. Deriving it wrong in either direction is
silent: too few uploads means the kernel reads the pool's previous occupant, too many means the
dirty-bit discipline stops reducing `sync_boundaries` and the latency numbers stop being
comparable to iron's.

---

## 7. Host-view lifetime — the footgun

`KernelCache.load_and_run` returns **zero-copy numpy views into BO memory**. `cache.py` already
documents this for `shared_nonstatic`; pooling generalizes the hazard.

**H1.** A view returned from a pooled slot is invalidated by the next step that reuses that slot.
Not "may be stale" — the bytes are overwritten.

**H2.** A caller that must keep an output past the next dispatch declares it a **host output**.
L3 then keeps it live to the end of the sequence, so no other buffer can take its slot.

**H3.** A caller that keeps an output past the end of the *sequence* must copy it. The pool is
reused by the next sequence.

**H4.** Per-layer diagnosis capture, `make diagnosis`, and anything that accumulates activations
across layers are exactly the callers H2/H3 exist for. This is the same caveat as
`shared_nonstatic`, and it is why that flag's contract note is preserved verbatim rather than
being deleted when the allocator subsumed it.

---

## 8. What the allocator subsumes

`KernelCache` grew three hand-rolled special cases. Each is a degenerate case of the above:

| Existing flag | Expressed as |
|---|---|
| `static_input_indices` | buffers marked static (§5) |
| `intermediate_indices` | buffers a producer inside the sequence writes and no step reads first (D7), so `start > -1` and no host write (D1 never sets their dirty bit) |
| `shared_nonstatic` | one pool, all non-static buffers eligible for slot sharing, with L3 pinning declared outputs |

The flags stay on `load_and_run` as the single-step API, and keep their current semantics and
their contract notes. `run_sequence` is the multi-step API and derives all three from the
declared sequence instead of from caller-supplied index sets.

---

## 9. Error attribution

**E1.** `pyxrt.runlist` exposes only `add` / `execute` / `wait`, so a failed submission reports
one aggregate state. On failure the batch polls every `xrt.run.state()` and raises naming the
**first** non-`ERT_CMD_STATE_COMPLETED` entry, with its index, kernel name and artifact.

**E2.** A runlist whose entries would span artifact configurations **under the xclbin ABI** is
never submitted. It raises `RunlistSplitError` at build time. Rationale: 05a §4 measured that such
a runlist *executes* and returns wrong numbers, with no exception and no timeout, because the
array configuration comes from the xclbin behind the context rather than from the run. A
build-time refusal is the only safe behaviour.

E2 does not apply to the ELF ABI. There every artifact is loaded into its own `hw_context` from
its own full ELF, so each entry brings its own configuration; 05a §5 measures that a runlist
spanning three separately-compiled ELFs is bit-identical to sequential dispatch in every ordering.
Aggregating those is the *point* of the seam, and `runlist_gate.py` leg A2 exists to keep the
xclbin refusal in place while leg A exercises the ELF path — the two are one edit apart and only
one of them is safe.

---

## 10. Timing

**T1.** One timing scope covers the whole submission: from just before `runlist.execute()` to
just after `runlist.wait()`. Buffer sync is timed separately and is not inside it.

**T2.** A split sequence (§9 E2) times each submission separately and reports the sum, together
with the true submission count. It never reports a split sequence as a single submission — that
would make `runlist` and `offload` indistinguishable, which is the whole point of the dispatch
vector in [03-measurement-model.md](03-measurement-model.md).
