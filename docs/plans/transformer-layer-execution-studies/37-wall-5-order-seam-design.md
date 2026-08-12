# Item 6c — wall 5 scoping: the shim issue order and R1's consumers

`[2026-08-12]` Design/scoping output for queue item **6c**, the last wall in front of R1's device
gate and therefore in front of `fused`'s definitional gap. Branch
`exper/transformer-layer-execution-studies`, tip `b777517b`. **Nothing in the repository was
changed. No device job was run, nothing was rebuilt, no compile was invoked.**

Every claim below is exactly one of:

- **MEASURED** — an existing recorded artifact (devq job, doc 31/31b table), cited by job number;
- **SOURCE** — read out of the compiler or the builder in this tree, with file and line;
- **INFERENCE** — reasoning from SOURCE + MEASURED, with no run behind it. Marked inline, every
  time. Several of the load-bearing conclusions here are inference and are labelled as such.

**Carried markings.** [31b](../../../home/cj/mlir-air/docs/plans/transformer-layer-execution-studies/31b-r2-order-seam.md)
marks everything dump-derived PROVISIONAL pending 6b; 6b has since landed, so 31b's constants are
still owed a re-derivation and nothing here relies on one. `fused`'s SPECS atol stays **PROVISIONAL**.
No latency claim is made anywhere in this document (NPU pmode is `Default`, and it would be
irrelevant regardless — nothing here is a timing result).

---

## 0. The shape, in numbers, so the rest is checkable

SOURCE — `programming_examples/transformer_layer/builders/ffn_resident.py` +
`builders/ffn_accum.py` (`MICRO=8`, `TILE_M=64`, `FFN_ACCUM_HERD_X=4`, `FFN_ACCUM_TILE_K=32`), at
the gated spec row `ffn_resident [64x3072x768]`:

| | |
|---|---|
| `seq_len` = `TILE_M` | 64 (one band) |
| `emb_dim` | 768 |
| `ffn_dim` | 3072 |
| `herd_x` | 4 |
| `tile_k` | 32 |
| `group_n` = `emb_dim // herd_x` | 192 |
| `sweeps` = `ffn_dim // (herd_x·group_n)` | **4** |
| `k_steps_up` = `emb_dim // tile_k` | **24** |
| `chunks_per_group` = `group_n // tile_k` | **6** |

Three L3-side feeds, all emitted at segment scope as `dma_memcpy_nd`:

| feed | source loop nest | transfer count | L3 offset map |
|---|---|---|---|
| `hidden` → `l2_a_up` | `for s(4) { for k(24) }` | **96** | `src_offsets=[0, 4k, 0, 0]`, 4-D retile `sizes [8,4,8,8]` `strides [6144,8,768,1]`; **`s` does not appear** — the sweep re-read |
| `w_up` → `l2_b_up` | same nest (same body) | **96** | contiguous `24576·(24s + k)`, length 24576 |
| `w_down` → `l2_b_down` | `for s(4) { for c=0..3 (PYTHON-unrolled) { for jj(6) } }` | **96** | `589824·s + 147456·c + 24576·jj`, length 24576 |

Note two facts that the rest of the document turns on:

1. `w_up`'s 96 transfers **tile its whole packed array contiguously and in index order**
   (`24576·(24s+k)`, k inner) — they are perfectly coalescible into one wide BD.
2. `w_down`'s 96 transfers **also** tile its whole array contiguously in *source* order
   (`(s,c,jj)` ⇒ `589824s + 147456c + 24576jj` is the identity tiling) — but only if `c` stays
   between `s` and `jj`.
3. `hidden`'s is the seam-1 retile and is **not** coalescible: MEASURED and closed in
   [31 §Wall 4](../../../home/cj/mlir-air/docs/plans/transformer-layer-execution-studies/31-fused-resident-tail.md)
   — four hardware BD dimensions all in use, no mergeable adjacent pair, the `k` loop would be a
   fifth and the `s` loop a sixth (a stride-0 repeat still needs a spare dimension).

---

## 1. The deadlock, restated precisely enough to be falsified

### 1.1 Where the order is actually made — SOURCE, and it confirms doc 19 step 1

`DmaToChannelPass::runOnOperation` (`mlir/lib/Transform/AIRDmaToChannel.cpp:1543-1554`) walks the
external channel ops into `externalChannelOps` and then does:

```cpp
for (auto getput : externalChannelOps) {
  getput->setAttr("loop-carried-dep", StringAttr::get(context, "external"));
  RewritePatternSet hoistChannelPatterns(context);
  hoistChannelPatterns.add<AIRHoistExternalAIRChannelPattern<air::HerdOp>,
                           AIRHoistExternalAIRChannelPattern<air::SegmentOp>>(context);
  (void)applyPatternsGreedily(module, std::move(hoistChannelPatterns));
}
```

**One external channel op is marked at a time**, and the pattern
(`AIRHoistExternalAIRChannelPattern::matchAndRewrite`, `:781-1010`) then clones the *entire
enclosing block* — loop nest and all — for that op alone. So N external ops in one body produce N
sibling launch-scope loop nests, concatenated in walk order.

This is worth stating carefully because it is a smaller defect than the doc's wording suggests: the
**pattern already collects `externalGetPuts` as a vector** and would hoist them together into one
cloned nest preserving their relative order. The fragmentation is entirely in the driver loop above
it. (SOURCE.)

Applied to R1 that yields launch-scope order:

```
[ hidden : for s,k  -> 96 puts on @air_channel_2 ]
[ w_up   : for s,k  -> 96 puts on @air_channel_3 ]
[ w_down : c=0, for s,jj -> 24 ] [ c=1 -> 24 ] [ c=2 -> 24 ] [ c=3 -> 24 ]   (@air_channel_4 after fusion)
```

which is exactly the MEASURED grouping in devq 236 and exactly the MEASURED `w_down` offset
sequence `0, 24576 … 122880, then 589824 …` (= `c=0` across all `s`, then `c=1`, …).

Downstream, nothing restores it and two passes actively re-impose it:
`air-isolate-async-dma-loop-nests{scope=launch}` runs **twice** in `buildOptimizationPipeline`
(`tools/aircc/aircc.cpp:845, 874`) and a third time inside `air-opt-shim-dma-bds`
(`AIRDependencyScheduleOpt.cpp:8540`). Its splitting relation
(`channelEndpointsResourceDep`, `:4611`) requires **the same channel name**, so two different
channels are never held together in one loop. (SOURCE.)

### 1.2 The two serializers, and the falsifiable statement

The runtime sequence is one sequential microcontroller program. Two independent things can block it
inside a run of same-channel tasks:

- **BD pool + 6b's pacing.** 16 BDs on a shim tile; the frees are clustered at the terminal
  `wait_all`, so a 96-task run needs 96 live BDs — MEASURED, 97 vs 16 on tile (1,0), doc 31
  §Wall 4. `boundShimBdLiveness` now paces the run with `dma_await_task(t[i-depth])` **before
  task i's configure** — a blocking await.
- **Task-queue backpressure.** `dma_start_task` pushes onto a finite per-channel shim queue; the
  push stalls when it is full. (SOURCE-adjacent: the mechanism is the reason `synthesizeDouble-
  BufferedAwaits` exists at all; the exact queue depth is not asserted here.)

**The falsifiable statement (INFERENCE):**

> A run of more than `depth(L2 pad)` consecutive tasks on channel X, issued before the first task
> of a channel Y that X's consumer needs in order to retire X's first chunk, deadlocks the
> sequencer. R1's `l2_a_up` holds one chunk (two after ping-pong labelling — MEASURED, doc 31
> §Wall 5), and an up core's memtile BD chain alternates A,B,A,B, so `hidden` chunk 0 cannot
> retire without `w_up` block 0. With `hidden` ×96 emitted first, the sequencer blocks at
> latest at `hidden` task ~2–15 and never reaches `w_up`.

Falsified by: any `npu.air.mlir` runtime sequence in which the paced `hidden` run does **not**
precede `w_up`'s first start and the module still times out. That is exactly what §5's experiment
reads.

### 1.3 Does the "needs round-major interleave" inference hold? — **Not as stated. I do not believe it.**

The doc's inference is:

> no ordering of *whole channel runs* can satisfy R1 — every linear channel-major order starves
> some consumer, because all three feeds are coupled through one compute pipeline.

That argument is valid **only if every feed is a multi-task run**. It is not. Doc 31 §Wall 4's own
table is the counter-evidence, MEASURED:

| | tile (1,0) live BDs |
|---|---|
| `@air_channel_2` (`hidden`) | **96** |
| `@air_channel_3` (`w_up`) | **1** |

`w_up` is **one** BD. It folded. `air-opt-shim-dma-bds`'s
`applyAIRSpecializeChannelWrapAndStridePattern` collapses a per-channel put loop whose offsets are
affine in the IV into a single wide wrap/stride BD — and §0 shows `w_up`'s 96 offsets are the
identity tiling of its array, so it folds to one contiguous transfer. **A single wide BD is not a
run.** It is configured, started, and then streamed by the hardware under memtile backpressure —
the sequencer never blocks in it. This is precisely the model `coalesce_shim_dma` is built on:
"the receiving memtile ring drains the wider stream via backpressure exactly as it drained the
fragments" (`AIRRtToNpuPass.cpp:1652-1654`, SOURCE).

Consequently (**INFERENCE**): the whole-channel-run order

```
[ w_up : 1 wide streaming BD ] [ w_down : k wide streaming BDs ] [ hidden : 96 tasks, paced ]
```

has no starvation argument against it. Both co-operands are already streaming into their L2 pads
before the first `hidden` task is configured; each `hidden` task then retires as the up cores
consume, and the pacing await returns.

So the doc's inference is **not established, and I believe it is false as written**. What *is*
true, and what I would substitute for it:

> **The unfoldable feed must be issued last.** No ordering works in which a feed that survives as a
> multi-task run precedes a co-operand its consumer needs. Round-major interleave is *a* way to
> satisfy that; it is not the only way, and it is not the cheapest.

Two important qualifications, both marked:

- **INFERENCE.** There is no green run behind either version. The experiment in §5 is what
  settles it, and it settles it hermetically.
- The doc's `[ch2 ×96][ch3 ×96][ch4 ×96]` figure is from **devq 236, the marker-on build**, where
  folding is disabled by construction (see §1.4). It is not evidence about the unmarked build's
  fold state, and the README's item-6c row conflates the two. Doc 31 §Wall 5's own prose is
  actually precise: only `hidden` carries "×96" in the unmarked bullet.

### 1.4 A trap in the marker that the scoping must not walk into — SOURCE

`AIROptimizeShimDMABDs::applyAIRL3DmaFoldingPatterns` (`AIRDependencyScheduleOpt.cpp:8499-8534`)
implements `air.preserve_shim_dma_order` by **skipping the whole launch region**: neither the
isolate pattern nor the wrap/stride specialization runs. So the marker does not only "prevent
regrouping" — it also **turns off folding for all three feeds**, converting `w_up` from 1 task to
96 and `w_down` likewise. That is why devq 236 read `[96][96][96]`, and it means

> **the marker as it stands makes the problem strictly worse unless a correct round-major order
> already exists in the IR.**

Which is exactly what devq 236 MEASURED. The marker is all-or-nothing per launch; there is no
per-feed folding opt-out today (`air.shim_feed_no_pace` is a *pacing* opt-out, not a folding one —
`AIRDialect.h:52-60`, SOURCE).

### 1.5 The second defect is not secondary — it is an independent deadlock

The `w_down` c-major order (MEASURED, doc 31 §Wall 5) is usually described as a separate blemish.
It is not: **INFERENCE**, it deadlocks on its own, with the up feed perfectly ordered.

The down herd accumulates over K, so K order per se is free — what is *not* free is that the A
operand of down-K-step `j` is GeLU column `c`'s chunk, and the up herd produces **sweep-major**:
sweep `s` is computed by all four up cores at once, giving groups `g = 4s + c` for `c = 0..3`
together. A `c`-major consumption order asks for column 0's chunks for all four sweeps before
touching column 1 — but columns 1..3 have already produced sweep 0's six chunks each, they fit two
per `l2_h` slot, and then GeLU cores 1..3 block, then up cores 1..3 block on their chunk puts, then
the up herd cannot advance to sweep 1, so column 0's sweep-1 chunks never exist. Deadlock, with no
shim ordering involved at all.

So the required down-feed order is the builder's source order `(s, c, jj)` — or any order that is
sweep-major on the outside. **c-major is not a schedule R1 can be re-tuned onto.**

**Consequence for the scoping: fixing the inter-channel order alone cannot green the gate.** Both
defects sit on the same root cause (§1.1's per-op hoist), but they are not one bug.

---

## 2. The candidate routes

Six, ordered by blast radius. (a) and (b) are the two the doc names; (c)–(f) are found in this
tree's source and are new to this scoping.

### Route A — change how `air-dma-to-channel` hoists (the doc's route (a))

**Mechanism.** Group `externalChannelOps` by `(hierarchy op, target level)` and mark a whole group
"external" before applying the pattern once, so `AIRHoistExternalAIRChannelPattern` clones the body
**once** with all the group's ops in it. The pattern needs no change — it already takes a vector and
`cloneOpsInBlock` preserves order. Roughly a 20-line driver change.

**What it fixes.** Both defects at once, for free: `hidden` and `w_up` come out interleaved in one
`for s,k` nest, and `w_down`'s four textual instances come out in one `for s { c0..c3 } ` body in
source order.

**Blast radius — the reason not to do it.** Everything downstream is written against the *exact*
shape this emits, in writing:

- `air-fuse-packet-put-loops` — "A candidate is **the exact shape air-dma-to-channel emits**"
  (`AIRDependencyScheduleOpt.cpp:4716`), and its wrapper-sequentialization eligibility is
  "deliberately the exact wrapper shape air-dma-to-channel emits, nothing wider" (`:4875-4879`).
  Its entire reason to exist is to undo this hoist; changing the hoist makes it a no-op on some
  designs and mis-shaped on others.
- `air-isolate-async-dma-loop-nests` would immediately re-split the newly-fused nests (its relation
  keys on channel name), so route A **does not survive its own pipeline** without route C's
  preservation anyway.
- `air-fuse-channels`' sibling-nest merge (the 6a fix) is defined over sibling nests that would no
  longer be siblings.
- `air-split-l2-memref`, `AIRToAIEPass.cpp:8181`'s `scf.parallel` unroll, and the ping-pong
  labeller all read this shape.

**Risk of silent miscompile: high.** The 6a experience is the precedent to quote — the old fusion
pass's *green* outputs were wrong, and a change at this altitude re-shapes every design in the
tree. 17 lit tests target `air-dma-to-channel` directly; `check-air-mlir` is 492; the
transformer-layer suite is 31 recipes; ten shipped models resolve the same path.

**Effort.** Small to write, weeks to trust. Not proportionate to one blocked increment.

### Route B — a runtime-sequence scheduling step + an IR notion of "coupled feeds" (the doc's route (b))

**Mechanism.** In `airrt-to-npu`, after everything is unrolled flat, re-order the
`aiex.dma_configure_task_for` / `dma_start_task` pairs of a declared feed group into a round-major
(or ratio-major) interleave.

**This is much less new than the doc assumes.** `AIRRtToNpuPass.cpp` already contains a
runtime-sequence reordering framework driven by opt-in IR markers, with the same anti-deadlock
rationale:

- `air.runtime_hoist` (`:2232-2280`) moves marked configure+start pairs to the global front of the
  sequence "otherwise the control program can block on a later input's dma_await — whose consumer
  is stalled in a feedback loop waiting on the compute that the hoisted feed drives — BEFORE it
  ever issues the hoisted feed";
- `air.await_appends` / `air.append_barrier` move awaits;
- `air.launch_wave` carries a per-wave index used for wave-keyed placement;
- `synthesizeDoubleBufferedAwaits` (`:2847`) already groups marked MM2S tasks **per channel in
  program order** — the exact data structure an interleave step needs.

So route B is "one more sibling in a family of five", not new infrastructure. The genuinely new
part is the **coupling + rate** notion: R1 does not want a 1:1:1 interleave (that starves too —
`w_down`'s block `j` is not consumed until up sweep `s` completes, so a uniform round-robin fills
`l2_b_down` at up-k-step 2 and blocks). It wants *per sweep: 24 up pairs, then 24 down blocks* —
a ratio, not a rotation.

**Blast radius.** Zero for unmarked designs, by construction (this is the strongest property any
route here has). **But it must interlock with 6b**: after an interleave, `hidden`'s frees are still
clustered, peak live BDs on tile (1,0) is still 97, so `boundShimBdLiveness` fires, paces — and
**sinks the run**, destroying the interleave (`paceShimFeedForBdReuse`, `:3344-3392`, moves every
task of the run before the anchor). Any route-B design must either mark the feeds
`preserve_shim_dma_order` (which excludes them from `boundShimBdLiveness` — `:3271-3275`, SOURCE)
and accept §1.4's folding loss, or teach `boundShimBdLiveness` to pace in place.

**Effort.** Days for the step; the hard part is specifying "coupled at rate R" so it is not
R1-specific. Medium risk, low regression risk.

### Route C — generalize `air-fuse-packet-put-loops` beyond packet channels

**Mechanism.** The pass that already exists to restore per-iteration interleave, un-gated from
`channel_type == "npu_dma_packet"` and gated instead on an opt-in coupling marker. It sits at
exactly the right pipeline slot (`aircc.cpp:878-881`: "Must run AFTER the last
air-isolate-async-dma-loop-nests (which would re-split the fused loop into per-channel loops)"),
and the one pass that would otherwise re-split it afterwards — `air-opt-shim-dma-bds` — is the one
that already honours `air.preserve_shim_dma_order`.

**What it needs beyond deleting the type check** (SOURCE, all three are real):

1. **Loop-nest depth.** `isCandidate` matches a single-level `scf.for` with one put. R1's feeds are
   two- and three-level nests. Either the matcher generalizes to nests with identical bound
   vectors, or a flattening runs first.
2. **A resource remap so the token chain survives canonicalization.** The packet version relies on
   `remapPacketStreamResources` (`AIRDialect.cpp:497-517`) mapping every packet channel symbol to
   one sentinel so `CanonicalizeAsyncOpDeps` does not prune cross-channel edges. Without an
   analogous remap keyed on the coupling marker, the interleaved chain is stripped and "the shim
   feed degenerates to per-channel grouping" — the comment says so verbatim.
3. **Bounds mismatch.** `hidden`/`w_up` share `for s,k`; `w_down` is `for s { c { jj } }`. The
   same-bounds grouping cannot merge the up nest with the down nest. Route C alone therefore fixes
   §1.1's *intra-nest* order (hidden↔w_up, and the four `w_down` siblings) but not the
   up-nest↔down-nest relationship.

**Blast radius.** Zero for unmarked designs. Marked designs: none exist in-tree today — grep for
`preserve_shim_dma_order` across `programming_examples/` and `python/` returns only a docstring in
`python/air/backend/xrt.py:102`. That is the cheapest possible safety argument: the change is inert
for all 492 compiler tests, 31 suite recipes and 10 shipped models.

**Effort.** Days. ~300 lines + lit tests, on a pass that already has the correctness argument
written down.

### Route D — strengthen 6b's sink instead of building an interleave

**Mechanism.** `paceShimFeedForBdReuse` already sinks the paced run "past the feeds it must not
out-order", but its anchor is *the first pre-existing `DMAAwaitTaskOp` after the run*
(`:3363-3378`). Change/tighten the anchor so a paced (i.e. unfoldable) run is placed after **every**
other feed's start in the sequence, not merely before the first blocking op.

**Why it is a candidate at all.** Per §1.3, if the co-operands fold to streaming BDs, this *is* the
whole inter-channel fix. And it may already be happening — the sink's eligibility conditions all
appear satisfied for R1's `hidden` run (96 ≥ 3 tasks, MM2S, `start` present, no `issue_token`, no
preserve/coalesced/nopace attrs, non-packet, exactly one `DMAFreeTaskOp` release each, all releases
after the run's last start). **Whether it fired in devq 235 is unknown and is the single most
valuable unknown in this document.** §5 reads it.

**Blast radius.** Confined to modules already over the BD budget — R1 is still the only one
(MEASURED, doc 31: "R1 is the only module that triggers the recycling").

**Effort.** Hours, *if* §5 says the sink is the gap.

**Risk.** Low but not nil: the sink is a scheduling change and its "cannot violate a dependence"
argument (tokens joined at the terminal wait_all) has to be re-checked for a stronger anchor.

### Route E — builder-side, no compiler change at all

Two independent edits to `builders/ffn_resident.py`, each targeting one of §1's two defects.

**E1 — take the `w_down` refill out of the `c` unroll.** The refill
(`ffn_resident.py:561-567`) is a plain segment-scope `dma_memcpy_nd`; **it carries no channel
index**. Only the `ChannelGet(CHANNEL_G, l2_h, indices=[c,0])` beside it and the
`ChannelPut(CHANNEL_DOWN_FEED, …, indices=[tx,0])` below it do, and those are what H5 forces to be
literal. So the refill can live in its own nest `for s { for c(real scf.for) { for jj } }` — **one
textual instance**, one hoisted nest, `(s,c,jj)` order preserved, and (§0 fact 2) the whole
`w_down` array then folds to a single contiguous streaming BD.

*Hazards, stated.* (i) It deletes the shared-`l2_b_down` WAR chain the builder's own comment
(`:543-550`) relies on to serialize `c0 < c1 < c2 < c3` — but that chain exists to order the *puts*,
and the refill/put decoupling is what folding does anyway; `air-enforce-channel-fifo-order` exists
for the residue. (ii) The builder records a MEASURED disaster in the opposite direction — a fully
unrolled `(c,jj)` body gave 24 channels and put `air-isolate-async-dma-loop-nests` into a >25-minute
state. E1 moves *away* from that (fewer textual instances), so it should be safe, but the channel
census must be re-counted.

**E2 — emit the unfoldable feed last.** Because §1.1's hoist preserves *walk order* between nests,
the launch-scope nest order is the source order of the `dma_memcpy_nd` calls. Splitting the `w_up`
refill into its own `for s,k` nest placed **before** E1's `w_down` nest, and leaving `hidden`'s
refill last, yields `[w_up][w_down][hidden]` with no compiler change whatsoever.

**Blast radius: zero.** Nothing outside R1 moves. R1 is `UNSUPPORTED`/parked today, so it cannot
regress anything.

**Effort.** Hours. Verifiable hermetically (§5) before any device time is spent.

**Risk of silently miscompiling something green: none** — it touches one parked builder.

**The honest limitation.** E is a *design* fix, not a *compiler* fix. It leaves §1.1's defect in the
compiler for the next design to trip over, and doc 31's own rule is to report such defects rather
than route around them silently. It should therefore land **with** the defect written up (items 8/9
are the precedent) and, ideally, with route C or D as the durable follow-up.

### Route F — make the offending feed foldable, or its pad deeper (both CLOSED)

For completeness, because they are the obvious first thoughts and both are already closed by
measurement:

- **Stage the whole `hidden` band in L2 and read it at a per-k' offset.** Closed twice: it is the
  frozen-BD construction doc 23's L3-side-offset rule forbids (H10 now refuses it loudly rather
  than miscompiling), and 31b §3.3 CONTROL 1/2 MEASURED the two obvious spellings — a
  literal-offset L1 band is a silent miscompile, and the L2-staged band refuses at
  `'aie.memtile_dma' op has more than 48 blocks`.
- **Deepen `l2_a_up` so the 96-task run can drain ahead.** Needs ≥96 slots against a 48-block
  memtile limit — same MEASURED refusal.
- **Fold `hidden` into a wide BD.** MEASURED closed, doc 31 §Wall 4: all four BD dimensions in use,
  no mergeable adjacent pair.

---

## 3. Does the second defect need fixing too?

**Yes, and it does not fall out of the primary fix.** §1.5 argues (INFERENCE) that `w_down`'s
c-major delivery deadlocks R1 on its own, with the inter-channel order perfect: a c-major
consumption order pins three of the four GeLU columns behind depth-2 `l2_h` pads and stops the up
herd from ever reaching sweep 1.

Which routes cover it:

| route | covers defect 1 (inter-channel) | covers defect 2 (`w_down` c-major) |
|---|---|---|
| A — change the hoist | yes | yes |
| B — runtime-sequence interleave | yes | yes, if the step re-sorts within a channel too |
| C — generalized put-loop fusion | partly (intra-nest only) | **yes** — the four `w_down` siblings share bounds `(s,jj)` and are exactly the pass's target shape |
| D — stronger sink | yes | **no** |
| E — builder | E2: yes | E1: yes |

So **D alone is insufficient**, and that is the sharpest structural fact in this scoping. Any plan
that is only about "interleave the three feeds" is incomplete.

One caution on route C's coverage: fusing the four `w_down` sibling nests round-major gives
`(s, jj, c)` order, not the source's `(s, c, jj)`. That is still sweep-major, and since both the A
(`l2_h` get) and B (`w_down`) sides sit in the same fused body their pairing is preserved and the
down herd's accumulation is order-free — so it should be admissible (**INFERENCE**), but it is a
different order from the builder's and must be re-derived, not assumed.

---

## 4. Recommendation

**Do not start with a compiler change. Run §5's one-compile diagnostic first, then take route E,
and hold route C as the durable compiler follow-up. Reject route A.**

Reasoning, in order of weight:

1. **The premise the two named routes rest on is probably false.** The doc's "no whole-channel-run
   ordering works" inference silently assumes all three feeds are runs of tasks. Doc 31 §Wall 4's
   own table records `w_up` at **1** BD — it folded. If the co-operands stream, the required
   property is only "the unfoldable feed goes last", which is a whole-channel-run ordering. Both
   named routes are sized for a problem that may not be the problem.
2. **There are two defects, not one, and the second is builder-reachable.** §1.5. Fixing it needs
   no compiler at all (route E1), because the `w_down` refill carries no channel index and is not
   what H5 constrains.
3. **Blast radius asymmetry.** Route E touches one parked builder. Route A touches every design in
   the tree, against passes whose comments say in writing that they are keyed to the current shape.
   6a's lesson — the old pass's *green* outputs were themselves wrong — is the reason to be
   unwilling to re-shape this altitude for one increment.
4. **R2 shrinks the prize for an expensive fix.** 31b's R2 deletes the `hidden` crossings entirely
   (rows 13/15 of 31a) by feeding the FFN from the on-chip norm tail. If that lands, the only
   L3-side feeds left in the resident tail are `w_up` and `w_down` — **both foldable**. So the
   heavy machinery would be paid for a configuration that R2 removes. R1 still has to gate first,
   which is why it needs *a* fix — just not necessarily an expensive one. (INFERENCE, from 31b §1.)
5. **Route C is the right durable fix if one is needed**, because it is the only compiler route that
   is provably inert for everything currently green (no in-tree design sets the marker), it sits at
   the one pipeline slot that survives, and the pass it generalizes already carries the correctness
   argument in its own comments.

### What would falsify the recommendation

- **§5 shows the paced `hidden` run already sunk behind `w_up`/`w_down`, `w_down` already s-major,
  and the module still times out.** Then §1.3's inference is dead, doc 31's stands, and route B or
  C is required. This is the cleanest falsifier and §5 is designed around it.
- **§5 shows `w_up` and `w_down` at ~96 tasks each in the *unmarked* build.** Then the "streaming
  co-operand" premise fails, whole-channel ordering genuinely cannot work, and route C (plus a
  per-feed folding opt-out, §1.4) becomes the recommendation.
- **E1 blows up the channel census or the compile time** (the builder's own MEASURED >25-minute
  `air-isolate-async-dma-loop-nests` state). Then defect 2 has to be fixed in the compiler after
  all — route C.
- **The sink turns out not to fire** (eligibility rejected for a reason §2-D did not anticipate).
  Then route D's anchor change is cheap and should be tried before E2.

### What the recommendation explicitly does NOT claim

- No claim that E makes the gate pass. E removes two identified deadlocks; there may be a third
  wall behind them, and this phase's history is four walls deep already.
- No claim about any latency or byte figure. None has been measured for the resident tail, and
  `fused`'s SPECS atol stays **PROVISIONAL**.
- The `(s, jj, c)` admissibility note in §3 and the whole of §1.3/§1.5 are INFERENCE.

---

## 5. The first verification step — one hermetic compile, no device

**Read the runtime sequence the current compiler actually emits for R1's numeric arm.** Everything
in §1–§4 hinges on three facts that no recorded artifact pins, and one compile pins all three.

Why it is cheap: aiecc writes `air_project/npu.air.mlir` (the post-`airrt-to-npu` module, containing
the `aiex.runtime_sequence`) before the device is ever touched — devq 235's whole job, compile
included, ran in **13 s**. It needs the two Peano kernel objects and the bare-shell toolchain env
(`tlenv.sh`, per the standing memory note), and **no NPU**. Stop before `run_test`'s dispatch.

> Note it must NOT be run while `build-xrt` is being relinked — 31b records the tree moving under a
> session four times in one day, and a structural literal without the `air-opt` mtime beside it
> cannot be placed. Record `stat build-xrt/bin/air-opt` with the result.

**The census to take from `npu.air.mlir`, in order:**

| # | question | how to read it | what it discriminates |
|---|---|---|---|
| 1 | Per-channel task counts | count `aiex.dma_configure_task_for @air_channel_N` per symbol | If `w_up` ≈ 1 and `w_down` ≈ 4 → §1.3's premise holds, routes D/E live. If all ≈ 96 → §1.3 dead, route C. |
| 2 | Emitted order | the symbol sequence of the configure ops | If `[ch3][ch4][ch2×96]` → **6b's sink already fired**, inter-channel order is fixed, and the residual blocker is defect 2 alone → route E1 is the whole fix. If `[ch2×96][ch3][ch4]` → the sink did not fire → route D first. |
| 3 | Pacing state | `issue_token` on `@air_channel_2`'s configures + `dma_await_task` before each | Confirms 6b is active on this build and did not silently decline. |
| 4 | `w_down`'s K order | the `src_offsets` literal on each `@air_channel_4` configure | Reproduces the MEASURED `0, 24576 … 122880, 589824 …`; confirms defect 2 survives folding. |
| 5 | Peak live BDs per tile | configure/free positions per `(col,row)` | Sanity: 6b's own invariant, and the number route D would have to preserve. |

**The second arm, still hermetic, still no device — run only if 1–4 support route E:** apply E1+E2
to a *copy* of `builders/ffn_resident.py` under the scratchpad, recompile, and re-take the same
census. Green means: `@air_channel_4` count small and offsets monotone in `(s,c,jj)`; `@air_channel_2`
last; `@air_channel_3` before it. That is a complete structural verdict on route E **before** one
second of device time, and it is the same discipline 31b used (design arm + controls verified
failing).

Only then re-arm `run_npu2_ffn_resident_peano.lit` and spend the gate run.

---

## 6. Two smaller things this scoping found, for the queue

Neither is item 6c; both are one-line-ish and both are latent traps.

1. **`air.preserve_shim_dma_order` is a folding switch as well as an ordering switch**
   (`AIRDependencyScheduleOpt.cpp:8499-8534`, SOURCE). Its own documentation in `Passes.td` and in
   `AIRDialect.h` describes it as an ordering opt-out; the implementation skips the launch region
   entirely, so it also multiplies a marked design's shim task count by whatever the fold was
   buying. Any future user of the marker will pay that without being told. Worth a comment at
   minimum, a per-feed opt-out at best.
2. **`AIRHoistExternalAIRChannelPattern` is already N-ary; only its driver is 1-ary**
   (`AIRDmaToChannel.cpp:1543-1554`). Whatever is eventually decided about route A, the fact that
   the fragmentation is a driver loop rather than a pattern limitation belongs in doc 19's step 1,
   which currently reads as though the pattern cannot do better.

---

## `[2026-08-12]` Verified at merge — the folding claim holds in source, verbatim

§2's load-bearing claim is that `air.preserve_shim_dma_order` is a **folding** switch as well as an
ordering one, and therefore that devq 236's `[96][96][96]` says nothing about the *unmarked* build's
fold state. Re-checked at `AIRDependencyScheduleOpt.cpp:8499-8534` before this document was
accepted. The source comment states it outright:

> Opt-out: a launch carrying `air.preserve_shim_dma_order` keeps its air.channel.put/get program
> order untouched (**no per-channel BD regrouping/folding**).

and the per-op opt-out path confirms the scope: a feed marked `air.shim_feed_no_pace`

> still benefits from the launch's no-fold guarantee (**the early return below skips per-channel BD
> folding for the whole launch region**).

So the marker-on measurement and the marker-off build are **not the same experiment**, and the
inference the README's queue row 6c drew from `[96][96][96]` — that no ordering of whole channel
runs can satisfy R1 — is **not established by that artifact**. It may still be true; it is not
measured. Doc 31's prose is careful here; the README row conflated the two, and is corrected.

**This is why §5's census is worth one compile before any route is chosen.** The single most
valuable unknown it settles is census row 2 — whether 6b's sink already fired — because that
determines whether the residual blocker is D2 alone (route E, hours, one parked builder) or both
deadlocks (route D first). No recorded artifact pins it.

Two smaller findings this raised, both worth carrying:

- **The preserve marker's folding behaviour is undocumented outside the source comment.** It reads
  as a pure ordering marker from its name and from every doc that cites it. The next user will be
  surprised the same way. It belongs in [23](23-rules-and-open-items.md) beside the other design
  rules.
- **Doc 19 step 1 should record that `air-dma-to-channel`'s fragmentation is a driver loop, not a
  pattern limitation** (`AIRDmaToChannel.cpp:1543-1554`; the pattern is already N-ary). That
  distinction is exactly what makes route A's blast radius large and route C's small, and it is not
  currently written anywhere.

---

## `[2026-08-12]` §5's census was TAKEN — route E confirmed, and the recorded order is refuted

Three arms, no device touched, `air_project/npu.air.mlir` read directly.

**Provenance, recorded first as §5 requires.** `build-xrt/bin/air-opt` 2026-08-11 13:28:03.455570624,
unchanged before *and* after every arm; `build-xrt/bin/aircc` sha256 `5cb08407…`, the same binary the
re-pinned item-10 STRUCT literal cites. Baseline compile 1.3 s, module unmarked
(`preserve_shim_dma_order` count = 0), so this is the **unmarked** build the design argued had never
been measured.

| row | measured | what it settles |
|---|---|---|
| 1 — task counts | `hidden` **96**, `w_up` **1** (one BD, `offset 0 len 2359296 sizes [2359296] strides [1]`), `w_down` **13** | §1.3's premise **holds**; "all ≈ 96" refuted; **route C's trigger is not met** |
| 2 — emitted order | **`[w_up][w_down][hidden ×96]`** | **6b's sink ALREADY FIRED.** Inter-channel order is fixed, **route D is not needed**, residual blocker is defect 2 alone |
| 3 — pacing | all 96 `hidden` configures carry `{air.bd_recycled, issue_token, repeat_count 7}`; awaits at **uniform depth 15**, 96 tokens each consumed once | 6b active, did not silently decline |
| 4 — `w_down` K order | `0, 147456, 737280, 1327104, 1916928, 294912, …` — c=0 folded across all four sweeps, then c=1 s=0..3, c=2, c=3 | **defect 2 survives folding, in a stronger form**: the whole of column 0 streams before column 1 begins |
| 5 — live BDs | (1,0) 97 by configure→free, all others ≤2 — **but the metric is misleading**: `hidden` emits **zero** `dma_free_task`, since 6b recycles BD *ids* via `air.bd_recycled` + awaits | direct evidence is that it compiled clean through the allocator; *inference*: ≤15 outstanding + `w_up`'s permanent 1 = 16, exactly the budget |

**Verdict: route E, and E1 alone is the whole remaining structural fix.** Rows 1 + 2 kill the premise
route C was reserved for and show route D's target already achieved.

**Second arm, E1 applied to a COPY** (shipped builder untouched): `@air_channel_4` collapses to **1**
task — the entire `w_down` array as one contiguous BD, monotone in `(s, c, jj)` by construction, so
**defect 2 is gone**. Order `[w_up][w_down][hidden ×96]`, pacing unchanged. **No blow-up**: channel
symbols **12 → 9** (down, not up), compile 1.4 s against 1.3 s, and the builder's measured
>25-minute `air-isolate-async-dma-loop-nests` hazard did not materialize. Structure identical to
baseline on every clause. **E1+E2 is structurally identical to E1, so E2 is inert by measurement** —
the sink already produces its target order.

### The discrepancy, and it is not resolved

Doc 31 §Wall 5 and the README's queue row both record the order as **`@air_channel_2 ×96, then
w_up, then w_down`**. This census measures the **opposite**. The timing is the likely explanation and
is recorded rather than assumed: devq 235/236 ran at **13:06:15 / 13:08:52**, `AIRRtToNpuPass.cpp`
was edited at **13:27:46** and relinked at **13:28:03**, and the only committed change to an
order-producing pass in that window is `ea3b98ce` — 6b's own fix. **So devq 235's binary is not this
binary.** Whether "the sink was not yet in 235's build" or "the recorded order was carried from an
earlier dump" is correct **is not established**: both scratchpads are gone and the old dump cannot be
re-read. Either way, **the order sentence in doc 31 §Wall 5 and in the queue row describes a
superseded binary** and must not be cited against the current one.

### What this census does NOT establish

- **Nothing numerical, nothing on hardware.** E1 and E1+E2 are structural verdicts only.
- **E1's correctness is UNVERIFIED, and this is its real risk.** E1 decouples the `w_down` refill from
  the `l2_b_down` puts, deleting the shared-buffer WAR chain the builder's own comment (`:543-550`)
  relies on. The token graph becomes cross-nest. That it *schedules* correctly is exactly what the
  emulation tests and a device run would settle, and neither has been run.
- **Why the baseline folds `w_down` to 13** (one 4-sweep mega-BD for c=0 plus 12 singles) rather than
  4 or 96 — mechanism unexplained.
- **Whether the same-day compiler merges move any of this.** The census binary predates `ba3916f8`
  (item 9, `air-shrink-memref-sizes-by-access`) and `971bab2a` (item 8, `air-split-l2-memref`). **Item
  8's pass is squarely on R1's L3-offset path**, so the census should be re-taken against the
  integrated build before E1 is committed to.
- **No claim that route E greens the gate.** §4's caveat about a possible third wall stands.
