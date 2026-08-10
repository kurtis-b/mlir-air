# 23 — What the 2026-08-06/07 run established: standing rules and open items


Four phases ran overnight. Two compiler defects were found and fixed, one was found and left with
its shape recorded, and the first piece of iron's dataflow form landed. This document holds what
came out of it that is **not** already a phase spec: rules that should govern later work, and open
items nobody has claimed.

## The rule that explains more than anything else found

**A column has two shim MM2S channels, and the budget is per COLUMN across the whole segment.**
Exceed it and AIR packet-multiplexes onto one queue — and until H9 the packet queue silently
misdelivered every trip after the first on more than one column.

Measured through `air-dma-to-channel`:

| builder | L3→L1 streams | packet-typed channels |
|---|---|---|
| `addnorm` (x, residual, weight) | 3 | **3 — multiplexed** |
| `elementwise_add` (a, b) | 2 | 0 |
| `layer_norm` (in) | 1 | 0 |

Reproduce with `python3 agents/probes/probe_packet_streams.py`; the probes and what each one
establishes are indexed in [`agents/probes/README.md`](../../../agents/probes/README.md).

This one fact explains a chain of things that looked unrelated:

- why **`fused`'s decomposed tail already ran 64 trips on 8 columns correctly** while the fused
  `addnorm` could not — each decomposed launch stays within two streams;
- why `addnorm` needed its one-trip guard at all;
- why J1's L2-staged-weight attempt failed with *"no ShimNOCTile has sufficient DMA capacity"* —
  8 columns × 2 = 16, already full before the third stream;
- why **J7a works**: packing x and residual into one strided fetch puts each column at two.

**The design rule for everything downstream (J7b, J7c, J8): keep every column's L3-facing streams
at two or fewer; put the rest on L1→L1 channels, and pack co-indexed L3 operands into one fetch.**
A stage wanting a third L3 input per column is back on the packet path.

Note the budget is per column *across stacked herds*: three 8-wide herds put one tile of each into
every column, so their demands add. Getting this wrong per-herd instead of per-column is the first
mistake this run made.

## Never read a staged buffer at a per-iteration offset — advance on the L3 side

`[2026-08-07]` **`aie.dma_bd` offsets are static, and nothing rewrites a BD chain into a form
that can walk.** Stage a buffer in L2 and put slice `k` of it from an offset that advances with
the induction variable, and:

| trips | what the memtile MM2S chain does |
|---|---|
| 2 | the K loop fully unrolls; each BD carries its own literal offset — **correct** |
| ≥ 4 | the chain becomes a cycle covering two steps with **every offset frozen at 0** |

Past the unroll limit the core stalls, the design's output DMA never fires, and the output
buffer is returned **byte-identical to what the host wrote** — seed it with 1.0 and every
element comes back 1.0. At J7b's spec shape (4 columns × 96 steps) it stops returning at all:
`ERT_CMD_STATE_TIMEOUT`. Not ping-pong; `omit_pingpong="all"` reproduces it identically.

**The rule: advance on the L3 side, never on the L2 read.** Stream each operand from L3 into a
small per-step staging buffer and put it from a *static* offset. The shim streams contiguously,
so no BD needs a moving offset. This is why J7b's B operand always worked and its A operand did
not — B was staged per step from the start, A was staged whole "because it is the small operand".

**`[2026-08-07]` And that is not a heuristic — it is the mechanism.** L3-side transfers are
programmed by the runtime sequence, which materializes an offset per task; tile-side (L2/L1) BDs
are static and cannot. The failing module shows all three cases side by side: an IV-dependent
offset on a **launch argument (L3)** is fine, the same thing on a `memref<..., 1 : i32>` (**L2**)
is silently wrong, and a whole-buffer put with no offset is fine. Survey: no other design in
`programming_examples/` takes a non-literal offset on an L2/L1 operand.

Reproduce with `python3 agents/probes/probe_ffn_accum_bd_offset.py`; the docstring says how to
point it at the pre-fix builder (`e6cdd138`) to see the frozen chain.

**This is H5, and H5 understates it.** Doc 16 records H5 as "channel indices are compile-time, so
a 64-band loop fully unrolls" and exhausts locks — a loud compile-time failure. The same root
limitation also produces this: a loop that declines to unroll and silently emits a chain
repeating a stale offset. H5's entry should be widened to say so before anyone scopes it.

## Match a probe's altitude to its claim

`air-opt` with a hand-built pass list answers *"does this pass fire"*. It does **not** answer
*"does this compile"*. The two diverge wherever a later pass rewrites what was measured.

This cost a spec claim: a strided-callee construction that lowered cleanly through `air-opt` never
compiles under `aircc`, because `air-to-aie` normalizes external callee signatures to identity
layout afterwards. J7a's session found it by compiling for real and routed around it.

**Use `aircc` — or `XRTBackend.compile(debug_ir=True)` — whenever the claim concerns anything
downstream of `air-to-aie`.** `air-opt` remains right for pass-level questions, which it answered
correctly several times in the same run. J7b's accumulator-ring claim was re-checked at `aircc`
altitude for this reason and holds (`pass_006`, 4 → 2 data-movement ops in the K loop).

## One process per device measurement — a loop over shapes is not a loop

`[2026-08-08]` A sequence-ladder runner that called the measurement function in a loop produced a
result that looked like a finding and was an artefact of its own structure. `coarse` and `offload`
walked 512/1024/2048/4096. `runlist` passed 512 and 1024, then failed 2048 **and** 4096 — two
different ELFs, both reporting "Failed to load ELF kernel for XRT ... contains a kernel symbol
matching the provided name" for a symbol that `llvm-nm` shows is present. Then `fused` failed its
**first** rung, 512, which had run clean an hour before.

The pattern is not shape and not mode; it is **process state**. `runlist` loads about ten kernels
per rung against `coarse`'s four, and `fused` came thirteenth, so pressure on XRT kernel and context
handles is what the failing rungs have in common.

**It is not a monotonic ceiling, and the record should not read as one.** `fused` failed 512,
**passed 1024** (66.3 ms), then failed 2048 and 4096. A handle count that only rises cannot do
that, so release is happening too — nondeterministically, on whatever schedule the previous rung's
objects are collected. The rung that fails is therefore the unlucky one rather than every rung past
a threshold, which is worse for diagnosis: a re-run moves the failures around and each individual
failure keeps looking like a property of its own shape.

The claim this rule rests on is narrower than a mechanism: **the same mode and shape behaves
differently depending on what ran in the process before it.** That is enough to invalidate
in-process laddering, and it is established by `fused` at 512 and by `runlist` at 4096, each of
which passed alone. The discriminating test is the isolated re-run; if a rung still fails with one
process per rung, that failure is real and belongs to its shape.

Written up in process, that reads *"runlist cannot run at 2048"* — a false limit, attributed to a
mode, in the exact voice a study uses for a real result. Nothing about the message suggests the
harness. The tell was that the same mode and shape had passed alone minutes earlier, which is only
visible if a single-shape path exists to compare against.

**So every device measurement runs in a child process that exits before the next begins**, and the
ladder invokes the single-shape CLI rather than importing it — which also makes a ladder row and a
single-shape row identical in provenance. This is what iron's `end_to_end/modes.py` subprocess
isolation is for; it reads like defensive scaffolding until a run walks far enough to need it, and
it was on the list of iron scaffolding to consider dropping.

**Corollary — measurement conditions are part of the measurement.** The same `coarse` at 4096 read
488 ms as a later rung of a shared process and 731 ms alone in a fresh one earlier the same
morning, a 1.5× spread on identical code. Host CPU load is the other half of it: compilation sits
outside the clock, but host-side dispatch does not, so anything CPU-heavy running alongside inflates
whichever rung it overlaps. Re-measure a whole comparison under one set of conditions rather than
assembling it from runs taken under several.

## Silence is the wrong default for `air-fuse-packet-put-loops`

H1s settled "skip and warn, never refuse to compile" for `air-label-scf-for-to-ping-pong`, and that
was right **because declining leaves a correct single-buffered loop** — only the optimization is
lost.

`air-fuse-packet-put-loops` has the opposite property: declining leaves the per-channel put loops
whole-channel-after-whole-channel against a consumer ring built for per-iteration interleave, which
past one trip is **silently wrong data**. The untransformed program is the broken one.

~~The pass currently contains **zero** diagnostics on any decline path.~~ Proposal, ~~unclaimed~~
**DONE `[2026-08-10]`**:

> When it leaves two or more same-bounds packet put loops unfused **and the trip count exceeds
> one**, warn — naming the loop, the channels and the trip count.

The trip count is free (`isCandidate` already requires static bounds and stores them as the
grouping key). Not unconditional: at one trip the orders coincide and the unfused form is correct,
so an unconditional diagnostic would fire on most shipped designs. Not an error either, because the
pass cannot establish that the group's channels share a queue before placement —
`aie.shim_dma_allocation` does not exist until `air-to-aie`.

**Implemented as specified**, as a post-transform scan in the pass (`warnUnfusedGroups`,
`AIRDependencyScheduleOpt.cpp`): whatever candidate shape remains after the transform was declined,
so the scan re-groups remaining same-bounds candidates per block — covering the dominance decline,
sealed group splits, *and* candidates inside a wrapper `scf.parallel` the expansion declined,
while excluding herd/segment bodies the pass does not own. The warning names the channels and trip
count and notes each sibling loop; one-trip declines and different-bounds pairs are verified
silent. `fuse_packet_put_loops_decline_warns.mlir`, four cases under `-verify-diagnostics`, both
lit tests green.

## Open items nobody has claimed

**1. ~~`layer_norm`'s two-pass variance has no throughput measurement.~~ MEASURED `[2026-08-07]`:
it costs ~13%.** Both kernels compiled from source and loaded once, 50 timed invocations each with
compilation outside the timed region:

| shape | one-pass bf16 | two-pass f32 | cost (min) | accuracy |
|---|---|---|---|---|
| 4096×768 | 4.835 ms | 5.474 ms | **+13.2%** | 1.969e-3 → 7.117e-5 (27.7×) |
| 512×512 | 0.406 ms | 0.461 ms | **+13.5%** | 2.009e-3 → 8.082e-5 (24.9×) |

Compare minimums, not medians: the one-pass runs carried more host jitter (p90 6.447 ms against the
two-pass 5.828 ms at 4096×768), which flatters it on the median to +9.2%. The min-to-min figures
agree to 0.3 points across a 12× shape range, so ~13% is the kernel cost and the rest is dispatch.

**The measurement carries its own provenance check**, and it passed: the reconstructed one-pass
kernel reproduced `mean_rel_L1` 1.969e-3 at 4096×768 — the exact figure `opcheck_specs.py` records
for the run that sized that row's `atol` — and 2.009e-3 → 8.082e-5 at 512×512, matching this
document's own recorded ≈2.0e-3 → 8.1e-5. Without that, a "before" build that was not actually the
old kernel would produce a plausible and meaningless number.

Reproduce with `agents/probes/probe_layer_norm_twopass_cost.py`. **~13% for ~26× is the trade Phase F
should quote**; nothing here argues for reverting it.

**2. Large-mean activations, measured `[2026-08-07]`: `layer_norm` is fine, the fused `addnorm`
falls off a cliff — and the cliff is not reached by this workload.** The regime that exposed the
one-pass defect is covered for `norm_tail` by the `128x768_offset` row; `layer_norm` and `addnorm`
had no equivalent. Run against the same regime `prepare_norm_tail` uses (x ~ 8.0 + 0.25·randn,
residual identically zero so the bf16 sum is exact and the row isolates the statistics):

| operator | control | offset regime |
|---|---|---|
| `layer_norm` | 0/49152 | **0/49152**, `mean_rel_L1` 1.1e-4 |
| `addnorm` (pre-add, 64×768) | 0/49152 | **43058/49152**, `mean_rel_L1` 33.1 |
| `addnorm` (post-add, 64×512) | 0/32768 | **28170/32768**, `mean_rel_L1` 22.2 |

J7a's f32 two-pass fix covers `layer_norm` completely. Both `addnorm` variants share the one-pass
path and lose the row's variance entirely: `E[x²]` and `E[x]²` become the same bf16 number, variance
clamps, and the row normalizes by `1/sqrt(eps)` ≈ 316.

**Where the cliff is, measured rather than derived.** Sweeping only the input against one binary:

```
|mean|/sigma      2 -> clean (0/49152)      4 -> COLLAPSED (1386/49152)      8 -> 24061      32 -> 43058
```

So the boundary is between 2 and 4. **A mantissa-only derivation gives 16 and is wrong by ~4×** —
it counts the final cancellation but not the 768-term row sums, which lose mantissa in bf16 first.
Worth remembering before trusting a similar hand-derived bound elsewhere.

**Not reachable here.** Per-row `|mean|/sigma` at both tensors `encoder_bert`'s addnorms normalize,
taken host-only from the FP32 golden reference: median 0.025, **max 0.115**, zero rows over 4. That
is a ~35× margin, so the recorded `block` / `coarse` / `runlist` / `fused` figures are **not** in
question. The defect is latent, and unpinned by any test. (Measured for `encoder_bert` only;
`decoder_gpt2` is pre-norm and no mode dispatches it — that is J5.)

**What to do, ~~unclaimed~~ half done.** ~~Adding the offset row to `layer_norm` pins its boundary
and passes today.~~ **`[2026-08-10]` The `layer_norm` half is DONE**: `128x768_offset` is a
catalogue row (`opcheck_specs.py`), pinned by `run_npu2_layer_norm_peano.lit`, and its first
hardware run measured `mean_rel_L1` 9.819e-5 with `atol_required` 0.0 — rtol alone covers every
element, within 1.4× of the zero-mean row, so the offset costs the two-pass kernel nothing
measurable. A revert to one-pass statistics now fails a suite recipe rather than a probe nobody
runs. Adding one to `addnorm` would fail, so it needs the kernel moved to two-pass f32 first —
which by analogy with item 1 costs ~13% on that kernel and would shift every provenance figure that
flows through it (`block` and the three modes), probably improving them. That is a phase, not a
patch, and the trade should be decided knowing the cliff is real but ~35× away — and it is still
unclaimed.

Reproduce with `agents/probes/probe_addnorm_variance_cliff.py`.

**3. ~~`fused` and `runlist` provenance figures are stale.~~ REFRESHED `[2026-08-07]`** from the
J7b gate run — the driver's own, so the provenance is a gate and not a hand-run:

| mode | was | now | `atol_required` | margin |
|---|---|---|---|---|
| `fused` | 1.806e-2 | **1.784e-2** | 7.572e-2 → **7.896e-2** | 1.32× → **1.27×** |
| `runlist` | 1.755e-2 | **1.732e-2** | 7.011e-2 → **7.077e-2** | 1.43× → **1.41×** |
| `block` | 1.688e-2 | 1.688e-2 | 7.398e-2 | 1.35× |

The prediction held for the mean, and `block` is correctly unchanged — it dispatches the fused
`addnorm`, which kept one-pass statistics, not `build_layer_norm_module`. **But both margins
tightened while the means improved**, which the item did not anticipate: `mean_rel_L1` is an
average and `atol_required` is a worst element, so they move independently. `fused` now sits 1.27×
under the hard ceiling, the least headroom of the four modes. Updated in `opcheck_specs.py`, the
example README, `pattern/fused/README.md`, `pattern/runlist/README.md` and
[09](09-phase-f-study-harness.md).

**4. J1 is blocked on shim BD exhaustion**, not on correctness any more: `herd_x=8` multi-trip
refuses at six trips (column 0 carries weight + x + residual, three packet tasks per trip, 18 > 16)
against a 64-trip target. The candidate fix is loop-shaped packet BD programs on the shim rather
than one `aiex.dma_configure_task` per iteration. **Not on the goal path** — J7a reaches the same
dispatch collapse without the packet queue.

**5a. `phases.sh`'s J7b objective check calls its builder with no shape.** `[2026-08-07]`
`phase_j7b_objective_check` does `build_ffn_accum_module()`, while J7a's sibling passes one
(`build_norm_tail_module(4096, 768, herd_x=8)`) and no builder here defaults its shape. Written
before the builder existed, it would have raised `TypeError` at the objective-check step —
failing the phase, after three review rounds and the gate, for a reason with nothing to do with
accumulator rings. Worked around by defaulting `build_ffn_accum_module`'s shape to the operator's
one claimed catalogue row. **The check should be corrected to pass the shape, and the default then
dropped.** That is an operator edit *between* runs: the driver's scripts are fingerprinted and no
phase's allowlist covers them, so a phase cannot fix its own checker — which is the design working,
not a gap. It does mean a bug in a checker costs a whole run unless someone runs the check by hand
first, which is worth doing for any newly-written objective check.

**5b. The static-BD-offset defect, LOCATED `[2026-08-07]` — it is an unchecked `std::optional`
dereference, and it is specced as [24](24-phase-h10-non-constant-bd-offsets.md).** J7b routed
around it (advance on L3, never on the L2 read), but the compiler still accepts the losing
construction silently. Root cause, in
`mlir/lib/Conversion/AIRToAIESchedulingUtils.cpp`:

```cpp
auto offset = mlir::getConstantIntValue(memcpy_offsets[i]);   // nullopt when NOT constant
one_d_offset += *offset;                                      // dereferenced unchecked
```

`get1DOffset` (206, 210) and the BD-dim-layout construction (462–464) both do this; **the same
file checks correctly at 527 and 945**, so the idiom is known and inconsistently applied. The only
caller is `AIRToAIEPass.cpp:6527`, and the line directly above it checks. Dereferencing a
disengaged optional is UB; what was observed is a silent `0`, which is how every BD in the cycle
ends up addressing the same block.

Confirmed against the real IR — the pre-`air-to-aie` dump carries
`air.channel.put ... (%arg4[%7] ...)` with `%7 = affine.apply #map()[%arg6]` over the K loop's IV,
while the accumulator fetch beside it is all literals, which is exactly why the C BDs were right
and the A BD was not.

So this is not "add a diagnostic to a pass that cannot know" — the pass has the information and
throws it away. Doc [24](24-phase-h10-non-constant-bd-offsets.md) specs the fix: return an optional,
refuse with a message that says what to do instead. **Refuse, not skip** — unlike ping-pong
labelling there is no correct fallback, since a BD cannot express a per-iteration offset. The
dynamic-index lowering that would make it expressible is H5 and is much larger.

**5. ~~`norm_tail_structure.py` checks at `air-dma-to-channel` altitude.~~ CLOSED `[2026-08-07]`.**
It now compiles through `XRTBackend(debug_ir=True)` — the production aircc binary — and asserts on
the routed design. The same weakness J7b's round-1 review found in `ffn_accum_structure.py`; this
applies the pattern that review settled on. Two of the five checks are new and could not be made at
the old altitude:

- **The stage edges are L1→L1**, measured on the final IR: exactly `2 × herd_x` = **16 `aie.flow`
  ops running core tile → core tile**. Keeping the intermediates off L3 *is* the phase, and an edge
  that silently round-tripped through L3 or a memtile would still show zero packet-typed channels
  and still pass every numeric arm.
- **The column budget, counted directly** — at most 2 shim-facing inbound flows per column
  (measured: 16 over 8 columns, exactly 2 each). The old check inferred this from packet typing,
  which is the compiler's *reaction* to exceeding the budget; this counts the thing the rule is
  about.

Plus: three herd rows of `herd_x` cores (the placement `air-place-herds` derived, which the builder
declares nowhere), zero packet-typed channels in **every** dump rather than one, and liveness at
both ends so no count passes vacuously.

**Verified in the failing direction**, which is the point of it: routing the same pipeline 4 columns
wide is rejected — `air-place-herds` collapses it to `{row 2: 8, row 3: 4}` and only 8 core→core
flows survive, so both new checks fire.

**Still no NPU and no Peano, and all three shapes take 3 s.** Reading the routed design sounds like
it needs a full build; it does not. **aiecc writes every MLIR pass dump before it compiles core
ELFs**, so the compile failing for want of a kernel object costs nothing — all 59 dumps land, the
last fully routed. Worth knowing generally: a structural check can sit at the very bottom of the
pipeline and stay hermetic. (The first version *did* build the kernels, which added a Peano
dependency to an arm whose Makefile target and lit RUN line deliberately pass none — it failed the
suite, not the script.)

**And a lesson that cost two suite runs: verifying such a script standalone is not verifying its
gate.** These checks pass on their own while the lit arm FileChecks the verdict line, so making the
verdict richer broke the suite with every check green. Run the lit arm, not the script.

## Struck from the plan

**H4 — `air.disable_ping_pong` works.** Measured: set on a loop that is otherwise labeled, `unroll`
and `hoist_alloc` both go to zero and the buffers are not rotated, and the attribute survives every
pass to the labeler. The original "setting it changed nothing" was taken on a shape whose callee is
unannotated and which is therefore never labeled at all — it disabled something that was not
happening.
