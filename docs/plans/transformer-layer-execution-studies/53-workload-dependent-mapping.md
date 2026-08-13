# 53 — Workload-dependent mapping for `fused`, and what the arithmetic already decides

`[2026-08-13]` R1's mapping is not chosen. It is **derived from the shape**, by two identities in
`builders/ffn_resident.py` that nothing states in one place. This document writes them down, and
then uses them to settle four questions that were being argued from intuition: whether staging
buys anything against `maxq`, whether a wider model needs a different mapping or has none, which
axis a two-workload split should cut on, and in what order the two fixes must land.

**Everything here is HOST-ONLY ARITHMETIC over constants read out of the sources named.** Nothing in
this document was compiled and nothing was dispatched. That is a deliberate limit, not an omission:
every quantity below is a closed form over `emb_dim`, `ffn_dim`, `tile_k`, `herd_x` and `TILE_M`,
so it is re-derivable in five lines and falsifiable by one compile. Where a number *is* measured, it
is measured elsewhere and cited — none of it is re-measured here.

**One claim in this document is a refutation of a proposal made in the same session**, and it is kept
rather than deleted, per this directory's convention: §2.2's unrolled-literal-offset staging was
proposed as the route around `maxq`, and §2.3 refuses it at 96 BD blocks against a cap of 48.

---

## The constants, and where each comes from

| Quantity | Definition | Source |
|---|---|---|
| `group_n` | `emb_dim // herd_x` | `builders/ffn_resident.py:312` |
| `sweeps` | `ffn_dim // (herd_x * group_n)` | `:336` |
| `chunks_per_group` | `group_n // tile_k` | `:337` |
| `k_steps_up` | `emb_dim // tile_k` | `:338` |
| `chunk_elems` | `TILE_M * tile_k` | `:339` |
| `up_b_block` | `tile_k * group_n` | `:340` |
| up-core L1 | `2*(2*(TILE_M*tile_k) + 2*(tile_k*group_n) + TILE_M*group_n) + 1024` | `:376-379` |
| `L1_BYTES` / `L2_BYTES` | 65,536 / 524,288 | `study/mapping_space.py:174,177` |
| `MAX_PLACEABLE_HERD_X` | 4 — a **measured** `aie-place-tiles` wall | `builders/ffn_accum.py`, restated `mapping_space.py:186` |
| `MAX_L1_TILE_K` | 32 — a **measured** L1 wall | same |
| 48-block memtile cap | the observed refusal `'aie.memtile_dma' op has more than 48 blocks` | [31b](31b-r2-order-seam.md)'s `l2_staged` control |

`maxq` is read off the refill loop nest at `:560-574`: `for s in sweeps: for k in k_steps_up:` one
`dma_memcpy_nd` each, all onto one shim channel with no await — so

```
maxq = sweeps * k_steps_up
```

which agrees with [52 §10.6](52-wall-7-race.md)'s measured `maxq == down_K` on all 21 compiled rungs.
§3 below shows *why* they agree, and when they stop.

---

## 1. The static legality predicate cannot see this class of change, and that is structural

`study/mapping_space.py` was the obvious instrument to price a staging change with. It cannot, and
the reason is worth recording because it will recur.

`r1_interior_demand()` already carries `shim_global=3` — `hidden`, `w_up` and `w_down` are **all**
modelled as L2-staged. The model counts **streams, not tasks**. So the current builder, which issues
`sweeps × k_steps_up` shim starts for `hidden`, and a staged variant that issues one, present a
**byte-identical `Demand`**:

```
Demand(herds=((4,0),(4,0),(4,1)), shim_global=3, shim_s2mm=4,
       memtiles=((4,2),(4,5)), l1_bytes=0, l2_bytes=0, ...)
```

Its fields are cores, shim MM2S slots (pinned and floating), shim S2MM, memtile ports, L1/L2 bytes,
core DMA ports and core→core edges. **Neither BD blocks nor queue occupancy is among them.**

This is the host-side twin of what [52 §11](52-wall-7-race.md) found in the compiler — *"the hang is
channel task-queue occupancy, not descriptor-pool exhaustion, and nothing was counting it."* Nothing
on the host counts it either. Doc 48's predicate answers *can this be routed*; it does not answer
*will this run*, and `maxq` lives in the gap.

Not filed as a defect. The predicate is placement-invariant by construction ([48](48-static-legality-and-space-size.md)
§where the line falls), and queue occupancy is a property of the emitted runtime sequence, not of a
declaration. Recorded so the next person does not spend a session pointing it at this question.

---

## 2. Staging `hidden`: the bytes fit, and the obvious form is refused

`hidden` is re-read from L3 once per `(sweep, k')` — `sweeps × k_steps_up` separate
`dma_memcpy_nd`s into the same L2 buffer, which **is** `maxq`. The builder's docstring gives the
reason (`:69-74`): staging it whole in L2 and reading at a per-k' offset is the frozen-BD miscompile.

Note what `agents/probes/probe_ffn_accum_bd_offset.py` actually says about that miscompile, because
it is narrower than the docstring implies: *"At 2 trips the K loop fully unrolls, so each put carries
its own literal offset and the 4-BD chain IS the whole computation. At 4 and beyond the loop stops
unrolling ... and the advancing offset is simply gone."* The defect is the **induction variable**
(`get1DOffset`'s unchecked `getConstantIntValue`), which H10 made refuse by message. **Literal
offsets are correct.**

### 2.1 The bytes, at the gate shape (`768×3072`, `tile_k 32`, `herd_x 4`)

| | now | staged whole |
|---|---|---|
| `l2_a_up` | 4,096 B | 98,304 B |
| `l2_b_up` | 49,152 B | 49,152 B |
| memtile total | 53,248 B | **147,456 B** of 524,288 — **28.1%** |

Fits, with room. The byte question is not the binding one.

### 2.2 The proposal: Python-unrolled literal offsets

Emit `k_steps_up` separate gets, each at a compile-time literal offset into the staged buffer —
the pattern the builder already uses for the up core's six chunk puts (`:80-82`). The class of
defect that would have made this silently wrong — N gets at different offsets measured as one, so
the buffer shrinks under its own readers — is **queue item 9, fixed 2026-08-12**, which is also
why [31b](31b-r2-order-seam.md)'s avoidance of literal-offset L1 bands no longer binds.

### 2.3 REFUSED — 96 BD blocks against a cap of 48

Each distinct-offset get is one memtile MM2S BD, and the A feed fans to one sub-channel per up core:

```
k_steps_up (24)  ×  herd_x (4)  =  96 blocks   vs the 48-block aie.memtile_dma cap
```

Over by 2×, at the gate shape. This is the same cap that refused 31b's `l2_staged` control and
re-parked wall 7's per-column fix.

**The assumption this rests on, stated so it can be falsified by one compile**: that AIR emits one
BD per (k' slice × sub-channel) with no folding across the sweep loop. If it folds, the count drops
and the refusal is wrong. Nobody has compiled it.

### 2.3a `[2026-08-13]` COMPILED. The refusal stands; the arithmetic around it was wrong twice

`agents/probes/probe_ffn_resident_interior.py --keep-dumps` at the gate shape (devq **338**,
build-class, 59 dumps in 2.1 s, hermetic — the ELF link fails on absent kernel objects by design).
Counted off `pass_058_after_cse.mlir`, top-level blocks per `aie.memtile_dma` region:

| memtile | MM2S | S2MM | `aie.dma_bd` | top-level blocks | of cap 48 |
|---|---|---|---|---|---|
| `mem_tile_1_1` (up feed) | 4 ch × 4 BD | 2 ch × 2 BD | 20 | **26** | 22 spare |
| `mem_tile_3_1` (down feed) | 4 ch × 2 BD | 5 ch × 1 BD | 13 | **22** | 26 spare |

Two corrections, both against things §2.3 assumed rather than read:

1. **R1 spreads across two memtiles** (cols 1 and 3), so "the 48-block cap" is per region and the
   up feed only ever contended with its own 26. §2.3 wrote as if there were one.
2. **The estimate of 96 was low, and the corrected figure still refuses.** The up feed's 4 MM2S
   sub-channels carry **4 BDs each** today — A and B, each ping-ponged — not 1. Replacing A's 2
   ping-ponged BDs with `k_steps_up = 24` distinct-offset ones gives 4 × (24 + 2) = **104 BDs, ~114
   top-level blocks against 48.** Refused by a wider margin than §2.3 claimed.

**The finding that matters is neither of those.** The current design already issues
`sweeps × k_steps_up = 96` shim tasks for `hidden` against **2 ping-ponged A BDs** on the memtile
side. So the memtile chain is *already* a repeating cycle and already maximally folded — **the 96 is
purely shim-side**, and no memtile-side count was ever the obstacle. §2.3 refuted a construction
that was solving the wrong side of the hop.

**This reframes §2.4 rather than confirming it.** The contiguous re-stream is not a question about
memtile block counts at all. It is two dimensionality questions, and both are 4-D budget questions
on an AIE2 BD:

- can the **shim** read the whole band in ONE task — `[TILE_M/MICRO, emb/MICRO, MICRO, MICRO]` is
  4 dims, where the current per-k' read is `[TILE_M/MICRO, tile_k/MICRO, MICRO, MICRO]`, so this
  looks like a re-parameterization rather than a new dimension; and
- can the **memtile→core** BD walk 24 slices of the staged buffer inside 4 dims.

Both are *first-pass* dimensional arithmetic, done on paper, not compiled. They are the right
questions; §2.4's framing (block counts) was the wrong one. Note the adjacent precedent for
pessimism: [52 §11](52-wall-7-race.md) killed the fold it wanted because six identical copies
needed *an outer stride-0 fifth dimension over four irreducible ones* — the same feed, from the
other side.

### 2.4 The variant that survives the count, and is OPEN

Stage `hidden` whole in **k'-major blocked layout** — one shim read, retiled by the same 4-D shim
pattern the design already rides (seam 1) — and feed each up core with **one contiguous BD per
sub-channel per sweep** rather than 24 literal-offset gets. That is 4 blocks, not 96, and it is the
idiom this compiler already emits for `@air_channel_0`/`@air_channel_1` — which
[52 §10.8](52-wall-7-race.md) names as what any plan reaching the gate has to make the `hidden`
refill do.

It requires successive k' slices to be contiguous in the staged buffer, which is a **builder** layout
choice, not a compiler capability. **Unverified.** Its bytes and its block count are checked above;
nothing else is. The check is compile-only, no device: build through
`XRTBackend.compile(debug_ir=True)` and count `aie.dma_bd` blocks per `aie.memtile_dma`.

### 2.4a `[2026-08-13]` COMPILED. Q1 lands, Q2 is REFUSED BY MESSAGE, and that is the answer

`agents/probes/probe_r1_staged_hidden.py`, two arms at the gate shape, hermetic, devq **339/340**
(build-class). `builders/ffn_resident.py` gains `stage_hidden=False`, mirroring the existing
`shared_h_staging` experiment flag; **the default path is byte-identical** (module sha256
`2582c733e19f26ba` against the committed builder, checked by building both), so no gated design moves.

**Q1 — the whole band in one 4-D shim read — builds.** As §2.3a predicted, it is a
re-parameterization of the per-k' read rather than a new dimension.

**Q2 — the per-k' drain out of the staged buffer — is REFUSED, verbatim:**

```
'air.channel.put' op channel @channel_23: BD offset is not a compile-time constant:
an aie.dma_bd is a static descriptor whose offset cannot advance per loop iteration,
so this transfer cannot be lowered to a tile-side (L1/L2) BD.
Stage the operand per iteration from L3 instead
```

That is doc 23 §2's rule and H10's diagnostic doing exactly their job — refusing rather than
emitting a chain that repeats a stale offset. **The refusal is the result**, and it converts §2.4
from open to conditionally closed: a staged `hidden` cannot be drained by a moving L2 offset, so the
contiguous re-stream **requires its own A-only channel**. A and B share `CHANNEL_UP_FEED` today and
the stream is FIFO, so an A-only contiguous put would desynchronize the core's alternating gets.

**And the next obstacle on that route is priced rather than guessed**: splitting A and B into two
channels wants **8 memtile MM2S ports against the 6 a memtile has** (§2.3a measured the up feed at
4 MM2S on `mem_tile_1_1`). Whether the allocator spreads them across memtiles or packet-multiplexes
is the question that route opens, and doc 23's counting rule says a port census reads **0** exactly
when it multiplexes.

### 2.4b The instrument was wrong first, and the number it printed was confident

The first run of this probe reported `maxq = 25` for the control. It is **15**. The configure op
opens with a brace and the regex required a paren, so *every* task resolved to one anonymous bucket
and the maximum over that bucket was reported as a per-channel maximum. Nothing in the output looked
wrong.

Fixed, and the probe now **raises** if any task resolves to no channel — a per-channel count that
silently collapses to one bucket is doc 51's defect class exactly, and this is the second time in
two iterations that the instrument, not the design, was the thing at fault. Recorded because
`maxq` is the quantity the whole `down_K` story rests on.

---

## 3. The identity that couples `maxq` to `group_n`, and when it breaks

`maxq == down_K` is not a coincidence and it is not a law. It is a consequence of the shared kernel
object.

```
maxq   = sweeps * k_steps_up = [ffn/(herd_x*group_n)] * [emb/tile_k]
down_K = ffn/tile_k

maxq == down_K   <=>   emb/(herd_x*group_n) == 1   <=>   group_n == emb/herd_x
```

And `group_n == emb/herd_x` is exactly what `builders/ffn_resident.py:44-54` forces: both GEMM herds
link one `ffn_accum_mm.o` whose tile shape is baked in as `-D` flags, and two private FuncOps cannot
export the same symbol in one module, so *"the up stage's output group width IS the down stage's
`tile_n`"*.

**So any change that decouples `group_n` from `emb/herd_x` decouples `maxq` from `down_K` — and it
decouples upward**, since `maxq` scales as `1/group_n`. That is the whole content of §5.

~~**`[2026-08-13]` CAUTION — the closed form above is NOT what the gate shape measures.** …the
compiler already folds this refill substantially at the gate shape…~~ **RETRACTED the same day, and
the error was mine rather than the formula's** — see §3.1. The caution conflated **task starts**
with **outstanding** starts. The closed form predicts the former and is exactly right.

### 3.1 `[2026-08-13]` The formula holds; the hazard quantity is a different number, and something is already pacing it

Walked the runtime sequence in program order, tracking starts and awaits per channel (devq 340's
control arm, `pass_056_after_airrt-to-npu.mlir`):

| channel | operand | total starts | peak outstanding |
|---|---|---|---|
| `air_channel_2` | `%arg0` = `hidden` | **96** | **15** |
| `air_channel_3` | `%arg1` = `w_up` | 1 | 1 |
| `air_channel_4` | `%arg2` = `w_down` | 1 | 1 |
| `air_channel_0_*`, `air_channel_1_*` | C fetch / `y` out | 1 each | 1 |

**Attribution is now settled** and §3's earlier "unresolved" is closed: `air_channel_2` reads
`%arg0 : memref<64x768xbf16>` with `sizes = [8, 4, 8, 8] strides = [6144, 8, 768, 1]` — the per-k'
blocked retile, on `%shim_noc_tile_1_0 MM2S 0`. It is the `hidden` refill, and it carries
`repeat_count = 7`, which is doc 52 §11's finding at this shape.

**Total starts are 96, exactly `sweeps × k_steps_up`.** §3's closed form is correct. What it does
not predict — and never claimed to — is the *outstanding* count, which peaks at **15** because
awaits interleave after the first 25 events (10 single-start channels + 15 of `air_channel_2`).
Doc 52 §10.6's `maxq == down_K` held on rungs 2–12 because at those sizes *every* start precedes
the first await; at the gate shape it cannot, and the two quantities separate.

**What sets the 15 is item 6b's pacing, and this is read off the source, not guessed.**
`AIRRtToNpuPass.cpp:3340` computes `depth = (capacity - fixed) / candidates.size()`, where
`capacity = tm.getNumBDs(tile)` — 16 for a shim tile. With one un-recyclable descriptor fixed and
one paceable feed, `depth = (16 − 1) / 1 = 15`. So `paceShimFeedForBdReuse` is **already bounding
this refill**, and it reaches R1 today.

**And it is pacing the wrong budget — which is exactly what doc 52 §11 concluded, now visible in a
number.** 6b's depth is derived from the **descriptor pool**; the hazard is **channel task-queue
occupancy**, whose measured band is PASS at 2/3/4, FAIL at 5, TIMEOUT at 6+. A BD-pool-optimal
depth of 15 sits four rungs above the last passing value. The two budgets are different resources
and only one of them is being counted.

**This gives queue row 28(b) a quantified prediction it did not have.** Its shipped step
`boundIdenticalShimPutRuns` paces to `depth = 2` — a queue-appropriate depth — and is inert only
because it walks `func::FuncOp` while R1 presents `aie.runtime_sequence` (doc 52 §12). Note the
tell: `paceShimFeedForBdReuse`'s *other* caller reaches R1 fine, so the fix is to give 28(b)'s
caller the traversal the working one already uses. **Predicted effect at the gate shape: peak
outstanding 15 → 2, from above the TIMEOUT threshold to inside the measured PASS band.** Recorded
here, before the fix, so it can be checked against what happens rather than fitted afterwards.

---

## 4. Model width: at `emb >= 1024` there is no legal mapping, not merely a different one

The up core's resident C group is `[TILE_M, group_n]`, and `group_n = emb/herd_x`, so **narrowing the
herd widens the accumulator**. With `herd_x <= 4` (`MAX_PLACEABLE_HERD_X`, measured) and
`tile_k <= 32` (`MAX_L1_TILE_K`, measured):

| emb | ffn | legal `herd_x` | `down_K` |
|---|---|---|---|
| 512 | 2048 | **[4]** | 64 |
| 768 | 3072 | **[4]** | 96 |
| 1024 | 4096 | **NONE** | 128 |
| 1536 | 8960 | **NONE** | 240 |
| 2048 | 8192 | **NONE** | 256 |

At 512 and 768, `herd_x = 4` is not chosen — it is the only value that fits. At 1024 and above the
current axis ranges contain **no legal point at all**, and the next family in `study/cases.py`
(`baseline_1024`) is already there. Both escapes are pinned:

| escape at emb 1024 | up-core L1 | cost |
|---|---|---|
| `herd_x 8`, `tile_k 32` | 41,984 — fits | crosses `MAX_PLACEABLE_HERD_X = 4` |
| `herd_x 4`, `tile_k 16` | 54,272 — fits | `down_K` 128 → **256** |
| `herd_x 4`, `tile_k 8` | 44,032 — fits | `down_K` **512** |

`tile_k` is a shared knob: what it buys in L1 headroom at a wider model it spends into `down_K`.
This is the same trade §5 finds on `group_n`, and both run through the up herd's resident C.

**This is the measured form of doc 44's finding 3 and doc 48's result.** 48 enumerated 15,347 legal
structures against iron's 7-entry hand-authored table and concluded *iron's hardcoded placements do
not port*; §4 is that conclusion at one axis, with the arithmetic that produces it.

---

## 5. The two-workload split: cut on `emb`, and do it second

Splitting the interior into two logically independent workloads dissolves the `-D` identity of §3,
which is the constraint that makes §4's table read NONE. It is the right instinct. Two things about
it are not obvious.

### 5.1 Cut on `emb` (output columns), not on `ffn`

The down GEMM reduces over `ffn`. An **ffn-split** therefore produces two *partial sums* of `y` that
must be added — a reduction network, extra traffic, and structurally the N-writers-onto-one-buffer
shape that **is wall 7** ([52 §§1-7](52-wall-7-race.md)). An **emb-split** produces two **disjoint**
halves of `y`: no reduction, each half reading its own columns of `w_down`, total weight traffic
unchanged. Only `H` is re-read.

The two are not symmetric and the cheap-looking one is the hazard.

### 5.2 It doubles `maxq`, by §3, and that sequences the work

| case | `group_n` | up-core L1 | `sweeps` | **`maxq`** | `down_K` |
|---|---|---|---|---|---|
| emb 768 (gate, today) | 192 | 58,368 FIT | 4 | 96 | 96 |
| emb 1024, no split | 256 | 74,752 **OVER** | 4 | 128 | 128 |
| emb 1024, **split 2** | 128 | 41,984 FIT | 8 | **256** | 128 |
| emb 1024, split 4 | 64 | 25,600 FIT | 16 | **512** | 128 |

The split buys legality and pays in shim-queue occupancy, against a wall that already trips at 6
outstanding starts ([52 §10.6](52-wall-7-race.md)).

**So the order is forced: fix the `hidden` refill first — 28(b)'s pacing, or §2.4's contiguous
re-stream — and split second.** In the other order the design does not run at any width, and the
failure would present as the split being wrong rather than as the refill being unfixed.

### 5.3 `H` becomes the thing to keep resident, and it is tight

Both halves of an emb-split need the **whole** interior, so `H = [TILE_M, ffn]` wants to be staged
rather than recomputed:

| ffn | `H` staged | of one 512 KiB memtile |
|---|---|---|
| 3072 | 393,216 B | 75.0% |
| 4096 | 524,288 B | **100.0%** |

At the very width the split exists to unlock, `H` fills a memtile on its own, leaving nothing for the
`w_up`/`w_down` staging that shares it — so it would have to spread across memtiles, which changes
the port arithmetic the design's fan-out rests on (`builders/ffn_resident.py:54-61`). **That is the
question to answer before committing to the split**, and it is answerable statically.

### 5.4 `[2026-08-13]` SPECIFIED — and the split's price is R1's own central property

Two things settle it, both host-only arithmetic over the builder's allocation types.

**First: the split is forced by the DOWN herd, which is why decoupling the kernel objects is not an
alternative to it.** Both herds allocate the same three L1 buffers — `l1_a[TILE_M, tile_k]`,
`l1_b[tile_k, group_n]`, `l1_c[TILE_M, group_n]` (`:400-403`, `:509-511`) — but the builder guards
only `l1_up` (`:376-379`). So the down core is over L1 at emb 1024 for the same reason the up core is:

| | group_n | L1/core | |
|---|---|---|---|
| emb 768, `herd_x` 4 | 192 | 58,368 | FIT |
| emb 1024, `herd_x` 4 | 256 | 74,752 | **OVER** (cap 65,536) |
| emb 1024, split 2 | 128 | 41,984 | FIT |
| emb 1024, split 4 | 64 | 25,600 | FIT |

Breaking the `-D` identity (§3) frees the **up** stage's group width, since its output is the `ffn`
axis and is tied to the down stage's `tile_n` only by the shared object. It does **not** free the
down stage: `group_n = emb/herd_x` is the definition of partitioning `y`'s columns across the herd,
so the only way to narrow it is to narrow `y` — which is the emb-split. **Two objects and the split
are not alternatives; the split is the necessary half.**

**Second, and this is the price: both halves need ALL of `H`.** `y[:, j] = Σ_k H[:, k]·w_down[k, j]`,
so every output column depends on the whole interior. That collides with R1's contract head-on — the
`[seq_len, ffn_dim]` interior is specified to exist *"not in DRAM, not whole in L2 — only as
`[TILE_M, tile_k]` chunks in flight"* (`:14-18`). An emb-split cannot preserve that clause:

| | what it costs | at the gate shape |
|---|---|---|
| **(a) recompute `H` per half** | up projection runs twice: 2× up compute **and** a second full `w_up` fetch per band | **+37.7 MB @512, +75.5 MB @1024** of DRAM, on top of §6's already-negative total |
| **(b) materialize `H` in L2** | breaks *"not whole in L2"* — but **not** the DRAM clause, so 31a's crossing count is untouched | 393,216 B at ffn 3072 (75% of a memtile); **524,288 B at ffn 4096 = 100%**, so it must spread |

**(b) is the right form and (a) should not be built.** (a) pays in the currency §6 just showed is
the binding one; (b) pays in L2 capacity and in a contract clause that was a design *simplification*
rather than a result. Note what (b) does not cost: nothing crosses DRAM that did not before, so
every crossing 31a credits residency with removing stays removed.

**The one remaining question is narrow and static.** At ffn 4096 `H` is exactly one memtile, so (b)
needs it across at least two — and the fan-through-memtile hand-off is derived from port arithmetic
on a *single* memtile node (`:54-61`: "the only node that can serialize `herd_x` producers into
`herd_x` identical consumer feeds is a memtile"). Whether that hand-off survives `H` on two memtiles
is answerable by doc 48's predicate with no compile, and it is the last thing before the split is
buildable.

**Ordering, with §6 folded in.** §5.2 already required the `hidden` refill fix before the split. §6
adds that **weight retention across bands belongs before both** — it is the larger term, it is
orthogonal to the split, and (a)-vs-(b) only matters once per-band weight traffic is not already
dominating.

---

## 6. The band-serial weight term, deliberately uncosted

The design is band-serial — one dispatch per `TILE_M`-row band, the band loop advancing on launch
arguments (`builders/ffn_resident.py` FOOTGUNS) — so `w_up` and `w_down` cross DRAM **once per
band**. That is the reuse term scaling with **sequence length** rather than with width, and §5's
split leaves it exactly where it is.

~~**No number is given here on purpose.**~~ **`[2026-08-13]` COSTED, inside 31a's accounting — and
the answer is that the band-serial weight term is larger than everything residency removes.**

**The per-band quantity is measured, not assumed.** In the emitted runtime sequence (devq 338/340,
§2.4a's control arm) the weight feeds are `%arg1 : memref<2359296xbf16> offset = 0 len = 2359296`
and the same for `%arg2` — each band dispatch fetches **the whole `w_up` and the whole `w_down`**,
9,437,184 B. That is structural rather than a defect: one band of 64 rows needs every element of
both weights, so band-serial execution cannot fetch less.

**31a's lens counts each static weight once per layer execution**, and its tail floor (rows 11, 12,
16, 21, 24, 25) counts both FFN weights exactly once. R1 is band-serial — `seq_len` must equal
`TILE_M`, one dispatch per 64-row band advancing on launch arguments — so it fetches them `S/64`
times:

| | @512 (8 bands) | @1024 (16 bands) |
|---|---|---|
| weights once (31a tail floor) | 9,437,184 | 9,437,184 |
| weights band-serial | **75,497,472** (8×) | **150,994,944** (16×) |
| tail total: floor → band-serial | 11,799,552 → **77,859,840** (6.6×) | 14,158,848 → **155,716,608** (11.0×) |
| whole packaged layer, 31a's derived floor | 51,121,152 | 88,083,456 |
| so R1's tail alone is | **1.52× the packaged LAYER** | **1.77× the packaged LAYER** |

**And it is net-negative against what residency buys.** 31a's per-scope split puts tail-internal
removable crossings at 17,301,504 @512 and 34,603,008 @1024 — linear at **33,792 B/row**, identical
at both lengths. Against an extra weight cost of `(S/64 − 1) × 9,437,184`:

| | @512 | @1024 |
|---|---|---|
| residency removes | 17,301,504 | 34,603,008 |
| band-serial weights add | 66,060,288 | 141,557,760 |
| **net** | **−48,758,784** | **−106,954,752** |

Setting the two equal gives the crossover: `33,792·S = 9,437,184·(S/64 − 1)` → **S = 83 rows, or
1.30 bands.** Above roughly one band, the composition costs more DRAM traffic than it saves.

**What this does and does not say.** It does *not* falsify resident composition — every crossing
31a says residency removes is still removed. It falsifies **band-serial** residency, and the term
is orthogonal to walls 7 and 28: neither `herd_x` nor `down_K` appears in it. So a session that
cleared both walls would arrive at a design that still moves 1.77× the packaged layer at 1024.

**The lever is weight retention across bands**, which is a different piece of work from either wall
— keeping `w_up`/`w_down` resident in L2 across a multi-band launch, or making the band loop
internal so one dispatch covers all bands. Doc 31's FOOTGUNS make band-serial a deliberate
simplification of the increment ("iterate bands on launch arguments, never inside the module"), so
this is a scoping consequence rather than a defect — but it is the first quantity that makes the
simplification expensive rather than merely temporary.

**Caveats, stated plainly.** This is arithmetic on 31a's DRAM-crossing lens — a logical-traffic
floor, not hardware. The per-band descriptor is read off an emitted artifact; the band count is
arithmetic. **No resident-tail byte figure has ever been measured on hardware**, and this is not
one.

---

## 7. What is open, in order

1. **Verify §2.4** — compile-only, no device, one build-class devq job. It decides whether a
   builder-side route to `maxq` exists at all; if it does not, [52 §12](52-wall-7-race.md)'s
   traversal fix is the only route. Cheapest item here and it gates the rest.
2. **§5's split is unspecified past the axis choice.** Open: where `H` lives (§5.3), whether the up
   stage runs once with `H` reused or twice, and what a second kernel object costs once the `-D`
   identity is broken on purpose.
3. **Cost the band term against 31a** (§6).
4. ~~**Nothing here composes into a selector.**~~ **`[2026-08-13]` The two instruments do not compose,
   and the reason is structural — see §8. Row 31's premise is wrong as written.**

---

## 8 `[2026-08-13]` The selector cannot be assembled from the parts row 31 names, and the gap is a decision

Row 31 describes the selector as composing three existing things: doc 48's static predicate to
enumerate the legal set, doc 47's ERT to price it, and the GEMM registry's shape-keyed cache. Read
against the sources, **the first two live on opposite sides of the compile**:

| | input | needs a compile? |
|---|---|---|
| `mapping_space.legality(mapping)` | a **declaration** — "no compile, no device, no dump: it reads the declaration, not a routed design" | **no** |
| `balance.demand_matrix` / `back_solve` / `balance_ports` / `bottleneck` | `parse_transfers(text)` + `parse_allocations(text)` off `routed_design_path(project_dir)` — a **routed artifact** | **yes** |

Doc 47's "affordable to run over a search space" is true *per artifact*; it is not a claim that the
space can be priced without compiling. There is no declaration-side entry point in `balance.py` —
every stage downstream of `demand_matrix` takes parsed transfers.

**So pricing the legal space means compiling it, and the census sizes that three ways** (numbers
from `study/mapping_space.py`'s own run, at the observed 2.1 s hermetic compile of devq 338):

| granularity | points | compile cost at 2.1 s |
|---|---|---|
| every legal point | 3,721,772 | **~90 days** — out |
| legal (structure, seam vector) | 15,347 | **~9 hours** of build-class jobs |
| legal structure | 428 | **~15 minutes** |

**Every route requires a modelled step, and that is the decision.** Compiling per structure or per
(structure, seam) prices a *representative* and attributes it to the points beneath — an
approximation over the axes not compiled. Pricing declaration-side needs a `Demand → cost` bridge
that does not exist. Either way something is modelled.

**Why that is not mine to choose.** Doc 47's central property is **1,208 measured / 5 counted / 0
modelled**, and it deliberately declined to import iron's cross-toolchain bandwidth figure so that
ports report `unpriced` rather than ranking against a constant wearing a measurement's label. A
selector introduces the first modelled numbers into that instrument. Which of the three routes to
take — and whether to keep the modelled costs in a separate table so the 0 stays 0 — is a call
about what the instrument is *for*.

**What is established here**: the composition row 31 describes does not exist as a composition, the
three granularities and their costs are as tabled, and the cheapest route is affordable. **What is
not**: which approximation is acceptable. The loop stops here rather than picking one.

**A note on what `fused` can and cannot do dynamically.** `group_n` is baked into the kernel object
as `-D` flags and the herd structure is baked into the xclbin, so for `fused` a workload-dependent
mapping is **build-time selection plus a shape-keyed artifact cache**, never runtime dispatch.
`offload` is the mode that takes its matmul loop bounds from a **runtime parameter**
([03](03-measurement-model.md) §The taxonomy). That asymmetry is taxonomic rather than incidental,
and it is most of what distinguishes the two modes.

**And a caution on how to select.** Do not copy iron's hardware search: it searches balance on
hardware in a **1% band** ([38](38-iron-encoder-pipeline-reference.md)), while
[47](47-balance-instrument.md) measured 259 repeats of an identical priced action spread to **42.2%
worst / 1.6% median** — a 1% band does not separate a fifth of them from themselves. Doc 48's finding
is that **enumeration beats search because the predicate is static**, which is the route this points
at.
