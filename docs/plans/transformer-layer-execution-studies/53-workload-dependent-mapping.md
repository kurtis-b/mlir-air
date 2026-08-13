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

**`[2026-08-13]` CAUTION — the closed form above is NOT what the gate shape measures.** Read off the
`airrt-to-npu` dump (devq 340, §2.4a's control arm), outstanding starts before the first await are
**15** on one channel and **1** on every other, against the **96** that `sweeps × k_steps_up`
predicts. So the compiler already folds this refill substantially at the gate shape, and doc 52
§10.6's `maxq == down_K` — measured on rungs 2 through 12 — **does not extend to it**.

Three things follow, and only the first two are established:

1. **The formula is an upper bound at this shape, not the value.** Every use of it in this document
   (§4's table, §5.2's 256) is a count of *puts in the loop nest*, not of *task starts in the
   emitted sequence*, and the two diverge once folding is in play. Treat those columns as the shape
   of the trade, not as predicted `maxq`.
2. **The conclusion that the gate shape is over the wall survives** — §10.6's separation is PASS at
   2/3/4, FAIL at 5, TIMEOUT at 6+, and 15 is over. What changes is the margin.
3. **Unresolved, and it is the next thing to settle**: which channel carries the 15 is *not*
   verified here — doc 52 §10.6 names `@air_channel_2` as the `hidden` refill at ITS shapes, and the
   symbol numbering at the gate shape has not been attributed. Do not cite the 15 as "the hidden
   refill's `maxq`" until it is.

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

---

## 6. The band-serial weight term, deliberately uncosted

The design is band-serial — one dispatch per `TILE_M`-row band, the band loop advancing on launch
arguments (`builders/ffn_resident.py` FOOTGUNS) — so `w_up` and `w_down` cross DRAM **once per
band**. That is the reuse term scaling with **sequence length** rather than with width, and §5's
split leaves it exactly where it is.

**No number is given here on purpose.** [31a](31a-resident-byte-floor.md) derives the resident byte
floor with its own accounting of what counts as a crossing, and a per-band weight figure quoted
outside that accounting is the shape of claim this directory has twice had to retract. Costing it
against 31a is an open item, not a result.

---

## 7. What is open, in order

1. **Verify §2.4** — compile-only, no device, one build-class devq job. It decides whether a
   builder-side route to `maxq` exists at all; if it does not, [52 §12](52-wall-7-race.md)'s
   traversal fix is the only route. Cheapest item here and it gates the rest.
2. **§5's split is unspecified past the axis choice.** Open: where `H` lives (§5.3), whether the up
   stage runs once with `H` reused or twice, and what a second kernel object costs once the `-D`
   identity is broken on purpose.
3. **Cost the band term against 31a** (§6).
4. **Nothing here composes into a selector.** §4 says the mapping is derived from the shape; doc 48's
   predicate enumerates the legal set for a shape with no compiler and no device; doc 47's ERT prices
   it; the GEMM registry is the precedent for a shape-keyed cache. No queue row claims the composition
   — see the README's queue row **31**.

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
