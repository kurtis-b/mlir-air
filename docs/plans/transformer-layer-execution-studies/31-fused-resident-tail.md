# 31 — The resident fused tail: draft phase spec

`[2026-08-10]` DRAFT — scoping output, not a scheduled phase. The phase letter is deliberately
unassigned: J7c (the `mha_out_proj` pipeline, [21](21-phase-j7a-norm-tail-pipeline.md)) and J8
([23](23-rules-and-open-items.md)'s downstream list) are already spoken for, and this document
should not collide with either. Everything below is either **measured** (with the artifact named)
or **marked as a prediction**; the supporting arithmetic is
[31a](31a-resident-byte-floor.md) and the structural evidence is
`agents/probes/probe_fused_resident_tail.py`.

## What "resident" means here, in doc 03's vocabulary

[03 §Composition](03-measurement-model.md) splits what "fused" used to conflate:

- **packaged** — `stitch_elf`: sub-kernel IRs spliced into one artifact; one configuration per
  segment, hand-offs still crossing DRAM through L3 func args. `fused`'s tail today.
- **resident** — operators on the array **simultaneously**, hand-offs L1→L1. What `fused` is *by
  definition* and has never been.

The role-style axis names the same move: the packaged tail is **time-multiplexed** (zero
core→core flows between operators), a resident tail is **space-multiplexed** (≥ stage-count
core→core flows). The discriminator is countable in the routed IR, which is what makes this phase
gateable structurally and not just numerically.

**Scope: the tail only.** [28](28-coarse-blend-space.md) derived that `fused` and `coarse` build
their front from the same two modules and differ in the tail alone, so the tail is where
residency work lives. The front (qkv_proj ELF, mha_out_proj ELF) is untouched — see Non-goals.

A resident tail is: **ln1 → FFN up-projection → GeLU → FFN down-projection → ln2 as concurrent
stages in ONE segment, hand-offs on L1→L1 channels, with DRAM crossed only by the tail's input
(`packed1`), its output, and its weights (`gamma1`, `w_up`, `w_down`, `gamma2`).**

## The byte floor it buys

From [31a](31a-resident-byte-floor.md), derived to the byte and reconciled exactly against
[27](27-common-ladder-result.md)'s measured 42,467,328 B @1024 and
[26 §C](26-mode-rebuild-feasibility.md)'s 84.0 MiB crossing table:

| DRAM-crossing lens | @512 | @1024 |
|---|---|---|
| packaged `fused`, total crossings | 51,121,152 (48.75 MiB) | 88,083,456 (84.00 MiB) |
| **tail-resident** (this phase) | **33,819,648 (−33.8 %)** | **53,480,448 (−39.3 %)** |
| whole-layer resident floor (not this phase) | 15,731,712 (−69.2 %) | 17,304,576 (−80.4 %) |

The tail-internal crossings this phase deletes are 16.5 MiB @512 / 33.0 MiB @1024 — `hidden`
(write, mirror-write, two reads), `ffn_up` (write+read), `ffn_gelu` (write+read), `ffn_out`
(write+read) — **48.9 % of everything residency can ever remove** at 1024. The remaining 51.1 %
is front-internal (`qkv_f32`, `q/k/v`, `attn_context`, `attn_out`) and out of scope.

**The instrument warning, restated as a design input:** `bytes_transferred` counts host syncs
and will not move for any of this — every removed crossing is device-side. Worse, the mode's
recorded number will *drop* anyway because intermediate boundaries stop existing as readable L3
args — today 9 of `fused`'s 13 steady-state syncs are verification readbacks of boundaries a
resident tail no longer exposes. The gate must therefore treat the vector change as an ABI
consequence to pin, not as the residency evidence; the residency evidence is structural (below)
plus latency and precision as proxies, exactly as J7a's gate reasoned.

**Precision is part of the payoff, and it is the measured part.** Every deleted crossing is a
bf16 round-trip; J7a measured the same deletion on the norm tail as `mean_rel_L1` 1.806e-2 →
measurably better than the decomposed form, and `fused` today sits at 1.784e-2 with the least
headroom of the four modes (1.27× under the 1e-1 ceiling, [23 §3](23-rules-and-open-items.md)).
Prediction, marked as such: a resident tail improves `fused`'s margin; the gate makes that a
clause rather than a hope.

## What exists to compose, and what the probe measured

The tree has exactly two space-multiplexed pieces, both standalone-gated, neither dispatched by
any mode's SPECS row (doc 03): J7a's three-herd norm-tail pipeline (`builders/norm_tail.py`) and
J7b's accumulator-ring FFN down-projection (`builders/ffn_accum.py`). `fused`'s stitched tail
already uses the norm-tail pipeline for both normalization points; the FFN between them is the
packaged three-launch `builders/ffn.py`.

`agents/probes/probe_fused_resident_tail.py` stitched nt1 (mirror) + ffn_accum + nt2 into one
module and compiled it through the production aircc to pass-dump altitude (hermetic, no NPU, no
kernel objects — [23 §5](23-rules-and-open-items.md)'s altitude lesson). Measured, at 1024×768
and 4096×768, ~4 s per shape, 59 dumps each:

- **Each launch becomes its own `aie.device`.** The composition lowers to three configurations
  (`@n1_norm_tail_seg`, `@ff_ffn_accum_seg`, `@n2_norm_tail_seg`) plus an anonymous control
  device. Stitch-level composition is time-multiplexed at segment granularity; **residency holds
  only within a segment**, so the resident tail is by construction a ONE-segment design, which no
  existing builder emits. This is the phase's central build item.
- **The pieces survive stitching intact.** 32 core→core flows (2 × herd_x per norm-tail
  instance, exactly as their standalone gates count), zero packet-typed channels in every dump,
  and the column budget met: both nt devices at exactly 2 shim-inbound per column over 8
  columns, the ring device at 1 per column over 4, with its full J7b signature visible
  (4 memtile→core feed flows, 2 shim→memtile stage refills, hoisted C fetch/store as 4+4
  shim↔core).
- **`air-fuse-channels` is not this scope's wall.** The composition carries 15–16 `air.channel`
  symbols and the whole compile takes ~4 s, against the measured >1200 s at 90 channels on the
  whole-layer stitch (doc 26 lane C). Prediction, marked: a one-segment resident tail carries
  more channels than the stitched form (the inter-operator hand-offs become channels) but
  remains an order of magnitude under 90 — the probe's census is the number to re-check when the
  one-segment builder exists.
- **Two composition walls, measured rather than assumed** (both printed by the probe):
  1. `build_ffn_accum_module` only builds a **64-row band** — `herd_x·herd_y ≤ 6` memtile feed
     ports with `MAX_PLACEABLE_HERD_X = 4` caps seq at `TILE_M` = 64. A full-seq tail iterates
     16 bands @1024, 64 @4096.
  2. At 4096 the mirrored compose is **ill-typed by construction**: the mirror writes
     plane-major `[2, rows, cols]` while row-interleaved `[rows, 2, cols]` is the only packing
     that builds above the shim-BD stride cap. The resident tail inherits `fused`'s 256..1024
     domain (doc 28's plane-major bound), and this is a second, independent reason 2048+ needs
     the strided-producer work first.

## The two whole-layer walls, and how a tail scope relates to each

Doc 03 records `fused`'s one-xclbin blocker as two measured walls. Stated per wall:

1. **`runtime_loop_tiling_sizes` `[1,1]`/`[2,2]` hardware conflict.** FlashAttention requires
   `[1,1]`; the wide GEMMs are built at `[2,2]`; one ELF is one aircc invocation; `[2,2]` over
   `mha_out_proj` @4096 compiles and then hangs, `ERT_CMD_STATE_TIMEOUT` 3/3
   (`agents/probes/probe_backend_preset_hardware.py`). **A tail-only scope avoids this wall
   entirely**: the tail's single aircc invocation contains no FlashAttention — attention keeps
   its own ELF with its own settings, exactly as `fused` ships today. Measured corroboration:
   the stitched tail already compiles at `[2,2]`, and the probe compiled the composition at
   `[2,2]` at both shapes.
2. **`air-fuse-channels` O(N²), >1200 s at 90 channels.** The wall belongs to the *whole-layer*
   stitch (90 channels at seq 256). The tail-scope composition measures 15–16 channels and ~4 s
   (probe, above). **A tail-only scope stays under this wall by an order of magnitude today**;
   the one-segment rebuild will raise the count and must re-measure it — the probe's
   fuse-channels census exists so that number cannot drift silently.

So the tail scope *avoids* wall 1 structurally and *stays measurably clear of* wall 2 — which is
the argument that residency work can proceed without the compiler phase doc 26 sized for
`air-fuse-channels`, and without touching `mlir/` at all.

## The capacity math

From [31a](31a-resident-byte-floor.md), all figures verified against code and doc 26 §C: the
chip is 2 MiB L1 (32 × 64 KiB) + 4 MiB L2 (8 memtiles × 512 KiB) = **6 MiB, not flat**. The S×F
intermediate is 3 MiB @512, **6 MiB @1024**, 24 MiB @4096 — whole-tensor residency of the FFN
interior is arithmetically out of reach at 1024+, which is doc 03's capacity bound and it is
real.

**The resident tail does not contradict that bound; it routes around it by never materializing
the tensor.** In a space-multiplexed stream the S×F intermediate exists only as tiles in flight:
per-stage L1 footprints are seq-independent (37,888 B per norm-tail stage_add tile at
rows_per_call 4; 57,344 B worst-case in the ring's cores at tile_k 32 — both measured fits, both
with their measured overflow points one notch up). The capacity question for this phase is not
"does S×F fit" but "do the STAGES fit beside each other", which is a column/tile budget question
(next section), not a bytes question.

## The three seams — the phase's actual engineering

The probe leaves both hand-offs into and out of the FFN as visible L3 args, because the existing
builders cannot close them. Closing them **within the standing rules** is the phase. Named,
with what makes each hard:

1. **Layout seam (ln1 → up-projection).** The norm tail emits row-major rows; every FFN GEMM
   kernel consumes blocked 8×8 microtiles, pre-tiled on the host precisely so that no in-flight
   retile puts a 4-D pattern on a memtile feed (`builders/ffn_accum.py`, WHY THE OPERANDS
   ARRIVE PRE-TILED). A resident hand-off must retile on-chip: candidate designs are a
   retile stage in cores (vector shuffles, no DMA pattern cost) or teaching the norm tail's
   stage_scale to emit microtile order directly. **Unmeasured either way — this is the first
   thing the phase must probe.**
2. **Order seam (column-striped vs band-serial).** The norm-tail herds partition rows BY COLUMN
   (column c owns rows `[c·rows_per_tile, (c+1)·rows_per_tile)`), while the ring consumes
   64-row bands fed through one memtile. Production order and consumption order disagree, so a
   direct channel cannot connect them without either re-mapping the norm tail's row→tile
   assignment to band order (builder change, admissible) or staging bands in L2 — and a band
   loop reading L2 at per-iteration offsets is EXACTLY the frozen-BD miscompile
   ([23 §Never read a staged buffer at a per-iteration offset](23-rules-and-open-items.md)).
   Any staged variant must advance on the L3 side, which re-introduces the crossing it exists
   to delete. **Prediction: the row→band re-mapping is the only shape that obeys the rules.**
3. **Dataflow seam (up → GeLU → down).** J7b's ring covers the down-projection only, and its
   compiler-formed C hoist exists *because* C lives in L3; with the output on-chip, C is
   L1-resident trivially and what survives of J7b is the in-place kernel + first-iteration zero
   mechanics, not the hoist. The up→down connection has a fortunate structure, stated as a
   design direction and NOT yet measured: if the up-projection iterates its output columns
   outermost, it produces H's column blocks in exactly the order the down-projection's K loop
   consumes them — a direct channel hand-off with static offsets everywhere, GeLU applied
   elementwise on tiles in flight (its kernel half already exists in `encoder_ffn.o`; doc 26
   also prices GeLU-as-epilogue as the alternative). Note `fused`'s current down-projection
   already accumulates in L2 (`tile_k_l2=512`, doc 26 §C) — the ring's win here is the
   *boundary* deletion, not the C round trip, which is not on this path.

Column-budget arithmetic for the one-segment design, **prediction, to be verified by the probe
re-run against the one-segment builder**: nt1 needs 2 L3-facing inbound per column (packed1 +
gamma1) at herd_x 8 — the budget exactly full on all 8 columns before the FFN's `w_up`/`w_down`
staging streams or nt2's gamma2 place anywhere. So the one-segment tail CANNOT ship all three
pieces at their standalone widths; the design space is narrower herds in disjoint columns
(placement currently stacks, doc 23's per-column-across-stacked-herds lesson), packing the
gammas into an existing stream, or dropping nt1's separate gamma fetch by folding the scale.
This arithmetic is why the phase is specified band-serial FIRST (increment R1 below): a 64-row
band engages 4 columns for the ring and leaves 4 for norm stages, inside budget, at the cost of
band-loop serialization whose price is unmeasured.

## What to build

Incremental, each with its own gate arm, none touching `mlir/`:

- **R1 — the resident FFN interior.** One segment: up-projection stage(s) + GeLU stage + ring
  down-projection, channels between them, per 64-row band; `hidden` in and `output`-band out
  still L3. Deletes the `ffn_up` and `ffn_gelu` crossings (24.0 of the 33.0 MiB @1024). This is
  where the layout seam (1) and dataflow seam (3) are settled at the smallest size.
- **R2 — attach the norm tails.** nt1 and nt2 join the segment; the order seam (2) is settled
  here. Deletes the `hidden` family and `ffn_out` crossings (the remaining 9.0 MiB @1024).
- **R3 — the mode wiring.** `pattern/fused` entry 3 becomes the resident tail artifact; the
  dispatch vector's new totals are pinned; the ladder re-walks BOTH lengths twice
  ([27](27-common-ladder-result.md)'s two-walk rule; a partial re-run is forbidden by
  [23 §One process per device measurement](23-rules-and-open-items.md)).

`build_ffn_accum_module`'s 64-row cap makes every increment band-serial; lifting the cap
(herd_y through more memtile ports, or row-band multiplexing per core) is its own decision with
its own placement walls (`MAX_PLACEABLE_HERD_X`, measured) and is not assumed anywhere above.

### `[2026-08-11]` R1 status: built, structurally green at pass-dump altitude

`builders/ffn_resident.py` + `agents/probes/probe_ffn_resident_interior.py` (hermetic, ~1 s), the
structure promoted to `ffn_resident_structure.py` as a suite arm. Measured, one 64×3072×768 band:

- **One tile-bearing `aie.device`** — the resident claim the stitched probe measured as three.
  12 channel symbols, `air-fuse-channels` 12 → 12, whole compile 1.0 s.
- **Seam 3 resolved as predicted, with one correction.** The up-projection produces H's column
  blocks in exactly the down ring's K order (groups of `group_n` = 192, chunked at `tile_k` = 32),
  GeLU on tiles in flight as its own herd. The correction: the GeLU→down hand-off **fans through a
  memtile by port arithmetic** — every down core consumes every chunk, a down core's two S2MM
  ports are spoken for (A|B feed + hoisted C fetch), and a channel has one physical source — so
  gate arm (b)'s derived core→core constant for R1 is `herd_x` (the up→GeLU edges), not
  3×herd_x. The interior still crosses DRAM nowhere; the L2 hop is the broadcast's price.
- **Seam 1 (L3 case) costs nothing**: `hidden` arrives row-major and the shim's 4-D read pattern
  (J7b's C-fetch idiom) retiles it during the per-k' refill. The R2 (on-chip producer) case
  remains open — a memtile MM2S cannot walk a per-iteration offset, so the same trick does NOT
  transfer to a staged band (doc 23's frozen-BD rule); the order seam owns that problem.
- **The kernel object is a composition constraint.** `-D`-baked symbols cannot coexist twice in
  one module, so the up stage's group width IS the down stage's `tile_n` (`emb/herd_x`) and both
  GEMM herds link the one 64×32×192 object. `ffn_dim % (herd_x · group_n) == 0` is now a
  precondition (holds at 3072/768/4: 4 sweeps of 4 groups).
- **The predicted ping-pong overflow does not happen**: the labelling ping-pongs NOTHING in this
  composition (up core C+A+B single = 40 KiB of 64; J7b standalone had A/B doubled). Fit is
  settled; the un-overlapped feed is a latency question for the ladder to price, not a gate
  matter.
- **A compile-time wall found and routed around** (in-builder comment + probe): every TEXTUAL
  segment-scope `dma_memcpy_nd` instance becomes its own auto channel under `air-dma-to-channel`,
  and 24 unrolled copies of the w_down refill left `air-isolate-async-dma-loop-nests`
  non-terminating (>25 min, 99.9 % CPU, on a 692-line module). The feed shape that compiles:
  real loops everywhere except the sub-channel index, which H5 forces to a literal. Cross-loop
  put ORDER on a shared sub-channel is carried by the shared staging buffers' inferred
  dependencies — the reason the relay keeps ONE `l2_h`/`l2_b_down` pair.
- **Ordering verified host-side to exactness** (session scratch, f64 emulation of every DMA
  pattern and channel op: packing, shim retile, chunk extraction, global K order — max error
  5.5e-12 vs the plain composition), so a device-run failure isolates to hardware behavior, not
  addressing arithmetic.

Numeric arm (`check-ffn-resident` + fault twin + lit recipe) registered at the catalogue's
64×3072×768 with the `ffn` scaling; atol provisional until the first gated hardware run records
its `mean_rel_L1`/`atol_required` in the row.

### `[2026-08-11]` R1's gate is BLOCKED by a compiler crash — the wall this scope did not predict

**`air-fuse-channels` segfaults on the R1 module, nondeterministically under aircc** — the same
binary (`install-xrt`, 2026-08-07) on the same 284-line module compiled clean twice (59 dumps,
the structural PASS above) and then crashed twice (stops after pass_017; SEGV in
`xilinx::air::isAsyncOp(mlir::Operation*)` ← `AIRFuseChannels::runOnFunction`,
`AIRDependencyScheduleOpt.cpp`). On the round-tripped pre-fuse dump the crash is deterministic:
`air-opt --air-fuse-channels pass_017_after_cse.mlir` → SEGV 10/10.

**Minimal shape, measured** (`agents/probes/probe_fuse_channels_sibling_nests.py`): a segment
whose body carries N sibling SAME-BOUNDS `scf.for` nests, each with one textual
`dma_memcpy_nd` refill (→ one auto channel each under `air-dma-to-channel`) and named-channel
puts. **N=2 fuses cleanly (5/5); N=3 crashes (5/5)** — the third mutually-mergeable channel is
what makes the pairwise O(N²) candidate loop revisit ops an earlier merge of the same set
already erased, consistent with a use-after-free in the NFL merge path (`runOnFunction` collects
`nfl_erased_ops`, calls `wrapRegionsWithForLoops`, then queries `air::isAsyncOp(e)` on them) and
with the ASLR-coin-toss behavior under aircc. The R1 down feed presents exactly herd_x=4 such
nests — forced there by H5 (the sub-channel index must be a literal, so the c dimension is
textually unrolled) — so the module is a production witness, not an exotic corner.

**Consequence:** every R1 gate arm (structure and numerics both compile through aircc) is a coin
toss until the pass is fixed; the wiring is landed verified-failing, and per this doc's rules the
defect is REPORTED with its minimal shape, not fixed in-phase (the H9/H10 route). Candidate
builder-side dodges, both unverified: staggered per-nest bounds (c·6 … (c+1)·6, which moves the
pair into the LB/UB unpeel path instead — different code, same stale-pointer risk), or any
restructuring that leaves ≤2 mutually-mergeable channels per candidate set. Note the OTHER
compile-time wall this phase measured (24 sibling instances → `air-isolate-async-dma-loop-nests`
non-terminating) bounds the same design space from above: the feed's legal shapes sit between
"few enough textual instances for isolate-loop-nests" and "≤2 mutually-mergeable channels for
fuse-channels", and today only the left constraint has a green point.

## Gate

The transformer-layer suite, allowlist `^programming_examples/transformer_layer/`, plus a
driver-owned objective check with four arms:

1. **Structural arm** — the probe's checks, promoted and extended, at every claimed shape,
   through `XRTBackend(debug_ir=True)` at pass-dump altitude (hermetic, ~seconds): **(a)** the
   tail is ONE segment / one `aie.device` (the resident claim itself — the probe measures three
   today, so this clause is verified-failing from birth); **(b)** core→core flow count equals
   the composed stage-edge count (32 for the two norm tails alone today; the constant moves
   with the design and must be derived, not copied); **(c)** ≤ 2 shim-facing inbound flows per
   column; **(d)** zero packet-typed channels in every dump; **(e)** liveness at both ends so
   no count passes vacuously (dma-to-channel dump clean of `air.dma_memcpy_nd`, channels
   present by name; final dump routed, no `air.channel` left). An edge that silently
   round-trips through L3 or a memtile passes every numeric arm — (b) is what catches it
   (J7a's lesson, verbatim).
2. **Numeric arm** — full-output `np.isclose` against the FP32 golden reference at the
   registry/mode tolerances, zero mismatches: the surviving boundaries at `BLOCK_STAGE_ATOL`,
   the layer output under the hard 1e-1 ceiling, and — the phase's payoff clause —
   `mean_rel_L1` at or below `fused`'s current 1.784e-2 with `atol_required` margin NOT worse
   than the current 1.27×. If residency does not improve the mode with the least headroom, the
   intermediates are not actually staying on chip.
3. **Fault-injection negative control** — the mode's measured target (`ln1_weight`, index 3,
   `fused.py`), which inside a resident tail scales one column of `hidden` and must cascade
   through both residual paths to a detected failure; the fault run's dispatch totals must
   EQUAL the clean run's (the standing anti-conditional rule). Doc 26 §5's lesson applies
   verbatim here: an injection target that does not discriminate for a normalization is a
   measured risk, so the target's detection margin is re-measured at each claimed shape, not
   assumed from the packaged mode.
4. **Vector pinning** — the new dispatch vector (submissions still 1; entries, air launches,
   sync count and bytes all change) pinned to literals in the lit recipe, both halves, with the
   old totals recorded beside them so the diff is legible. `bytes_transferred` will fall partly
   for verification-visibility reasons; the recipe must not present that fall as the residency
   result ([31a](31a-resident-byte-floor.md)'s instrument warning).

Run the lit arm, not the scripts standalone — verifying a checker standalone is not verifying
its gate ([23 §5](23-rules-and-open-items.md), the lesson that cost two suite runs).

## Design rules this phase must obey

All standing, all measured elsewhere, restated because every one of them has already decided an
outcome in this tree:

- **Two or fewer L3-facing MM2S streams per column, ACROSS the whole segment** — stacked herds
  add per column ([23](23-rules-and-open-items.md)'s first rule). Pack co-indexed L3 operands
  into one fetch; a third stream re-enters the packet path (silently wrong past one trip
  pre-H9, BD-starved post-H9).
- **L3-side offsets only.** Never read a staged L2/L1 buffer at a per-iteration offset — the
  frozen-BD miscompile presents as timeout or byte-identical output with no compile-time
  signal ([23](23-rules-and-open-items.md), `probe_ffn_accum_bd_offset.py`). H10
  ([24](24-phase-h10-non-constant-bd-offsets.md)) will make the compiler refuse; until it
  lands, the rule is the only protection.
- **One role per L1 buffer.** A buffer that is both a DMA destination and a kernel output does
  not read back what the kernel wrote — measured on the softmax build, returned-input-unchanged
  at all three shapes (doc 26 §5).
- **No hand placement, no hand buffer depths, no hand-built ring.** `air-place-herds` places
  and ping-pong labelling chooses depth; a tile coordinate in a builder falsifies the phase's
  claim (J7a), and if a pattern the compiler should form does not form, that finding outranks a
  workaround (J7b).
- **No widened tolerance** — the layer sits at the hard 1e-1 ceiling with `fused` at 1.27×.
- **Do not touch `mlir/`.** A compiler defect exposed here is reported with its minimal shape
  (that is how H9 and H10 came to exist), not fixed in-phase.
- **L3-side offsets for the band loop specifically**: the band iteration must advance on launch
  arguments, never on an L2-staged tail of a previous band.

## Non-goals

- **Whole-layer residency.** Blocked by the `[1,1]`/`[2,2]` conflict and the fuse-channels wall
  (both measured, §above), and capacity-bounded for whole tensors regardless. Nothing here
  moves attention or qkv into the tail's artifact.
- **Front changes.** `qkv_proj` and `mha_out_proj` ELFs, their settings, and the front→tail
  `packed1` boundary (including the x double-upload and the `qkv_f32` scratch ABI) are
  untouched. Doc 26 already prices the front's own wins (GeLU epilogue aside, `qkv_f32`
  deletion is explicitly do-not-attempt).
- **Sequence lengths above 1024.** The plane-major bound on the device-written front→tail
  boundary and the mirror's layout wall (probe, measured) keep the mode's domain at 256..1024;
  2048+ residency belongs to the strided-producer work doc 28 names, after `coarse`'s
  territory question is settled there.
- **No registry writes, no new kernel objects beyond what `encoder_ffn.o` /
  `ffn_accum_mm.o` / `layer_norm.o` already provide** — if a stage needs a genuinely new
  kernel, that is a scope change to be surfaced, not absorbed.

## Premise status

Measured before this spec was written (`probe_fused_resident_tail.py`, 2026-08-10):

1. The composed pieces survive stitching with their structures intact — 32 core→core flows,
   column budget met, zero packet channels, at 1024 and 4096. ✅
2. Tail-scope channel count is an order of magnitude under the fuse-channels wall (15–16 vs 90;
   ~4 s vs >1200 s). ✅
3. Segments are separate configurations — residency requires a one-segment builder that does
   not exist. ✅ (measured as the three-device lowering)
4. The FFN slice builds only as a 64-row band; the mirror hand-off is unbuildable at 4096. ✅
5. The three seams close within the rules — **not measured.** R1's first probe. The order-seam
   re-mapping and the up→down order match are predictions and are labelled as such above.
6. The one-segment column budget works out at some width — **not measured**; arithmetic above
   says the standalone widths do NOT fit together, so a narrower composition is required, and
   its latency price is unknown.
