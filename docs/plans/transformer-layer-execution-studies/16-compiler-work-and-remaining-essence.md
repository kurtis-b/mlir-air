# 16 — The compiler work, and what "the essence" still needs

`[2026-08-06]` Phase E landed four execution strategies that agree numerically and separate on the
dispatch vector. This document is what stands between that and the study the port set out to be:
*an analysis of ways to execute LLMs on the Ryzen NPU and their tradeoffs.*

Two tranches. **H** is compiler work in `mlir/` — it has the largest blast radius in the plan, since
every shipped model compiles through it. **J** is the study itself. They are separated because their
gates are different.

**`[2026-08-06]` J is no longer blocked on H.** The first draft of this document said three of J's
items were, on the strength of a root-cause analysis that Phase H then disproved by measurement. The
real blocker was a different defect, it is fixed, and the fix is verified on hardware. Every J item
can start today; H's remaining work is worth doing on its own merits, not as a prerequisite.

Everything below was measured or read on 2026-08-05/06. Nothing here is speculative.

## The root cause — corrected `[2026-08-06]`, and the guard is now liftable

> **This section replaces an earlier one that was wrong.** It blamed ping-pong buffering and a
> missing dependency edge, and Phase H's first attempt disproved it by measurement. The wrong
> version is not preserved here; [17](17-phase-h-compiler-hardening.md) records how it fell.

`builders/addnorm.py` forbids more than one trip of its row loop. That single rule turns 4096 rows
into **64 host dispatches**, which is 128 of `coarse`'s 131 runlist entries — the largest structural
gap between this port and iron's `hybrid` (5 entries).

**The cause was the shim feed order under packet multiplexing, not ping-pong.**

1. `air-dma-to-channel` hoists each L3-side DMA into its **own** launch-scope loop, so a herd that
   fills N buffers per iteration produces N sibling per-channel put loops.
2. When those channels are packet-multiplexed onto one shim MM2S queue (`channel_type =
   "npu_dma_packet"`), the queue serializes in task order — **whole channel after whole channel**.
3. The consuming tile's BD ring is built from the herd's per-iteration get order, so it expects the
   streams **interleaved per iteration**. At one trip the two orders coincide. At two or more, every
   packet after the first iteration lands in the wrong buffer.

Measured at `cols=64, rows=8, rows_per_call=4`: 481 of 512 elements wrong.

**How ping-pong was ruled out:** compiling the same shape with `--omit-ping-pong-transform=all`
reproduces the **identical** 481/512 corruption. A cause you can disable without changing the result
is not the cause. The `air.channel` rewrite that "confirmed" the old hypothesis was passing for a
different reason — channel form does not produce the sibling put-loop grouping in the first place.

**Fixed by `air-fuse-packet-put-loops`** (commit `bfb647d9`), which fuses sibling per-channel put
loops that share a block, share static bounds and all target packet-typed channels into one loop
performing the puts in program order, plus modelling packet-typed channels as one shared stream
resource in `CanonicalizeAsyncOpDeps` so the token chain survives pruning.

**The guard is liftable, and this is measured, not inferred.** The driver-owned fixture's
`--variant inside` — a legitimate two-trip loop at exactly the shape `addnorm.py` measured the
miscompile at — now runs on hardware with **zero mismatches**. That is the evidence J1 was waiting
for, and it means J1 is no longer blocked on anything.

### The second defect, real but not that one

`checkOpOperandReadOrWrite` (`mlir/lib/Util/Util.cpp`) classified a memref use via memory effects,
`ChannelPutOp`, `ChannelGetOp` or linalg and returned `'u'` otherwise — so **an external kernel
`func.call`, which registers no memory effects, was invisible to dependency construction**. Unknown
uses were silently omitted, and an empty producer or consumer set became
`air::WaitAllOp::create(..., SmallVector<Value>{})`: a dependency-free placeholder rather than a
rejection.

This did not cause the addnorm corruption. It is still worth fixing and **H2 fixed it**, because it
is the blocker under every dataflow analysis over external-kernel programs — including the
promotion pass in H8 below. A callee carrying `llvm.emit_c_interface` now classifies its memref
operands from `llvm.readonly` / `llvm.writeonly` argument attributes; an unannotated operand stays
unknown and the compiler never guesses a direction.

### What "bail out" turned out to mean

H1 was specified as "hard-fails compilation with a diagnostic", citing upstream `memref::multiBuffer`
as prior art. **That reading was wrong and the error is mine.** `memref::multiBuffer` returns
`failure()`, which means *decline to transform and leave the code alone* — not *abort the build*.
IREE (`skipOverrideAnalysis=false`), Triton's precondition list and TVM's `ICHECK` all bail out of
the transformation. Gate leg 4 caught the consequence: three shipped models (`llama32_1b_int4`,
`qwen3_0_6b`, `qwen3_1_7b`) failing to build on programs that were always correct.

The corrected rule: when the pass cannot prove the rotation safe it **skips** — leaves the loop
single-buffered, warns naming the loop, and compilation proceeds. Compilation aborts only for IR
that is genuinely malformed. The dependency-free `WaitAllOp` placeholder stays forbidden: skipping
means not transforming, not transforming with an empty edge set. That was always the real defect.

## Tranche H — compiler

Gate for every H item: mlir-air's own lit suite, the transformer-layer suite, **and** `make verify`
over the ten shipped models. `gate-e1.sh` is the model; H needs a build step in front of it —
`gate-h.sh` has it, as four legs.

### Landed `[2026-08-06]`

Committed on the branch, green through gate legs 1–3 (build + install, `check-air-mlir` at
486/500 with 7 pre-existing UNSUPPORTED and 7 pre-existing XFAIL, transformer-layer suite 24/24).
Phase H is still halted — see "In flight" below — but these are done and should not be re-derived:

| # | Item | State |
|---|---|---|
| — | **`air-fuse-packet-put-loops`** and packet-typed channels as one shared stream resource. The actual fix for the two-trip miscompile. | landed `bfb647d9` |
| H2 | **The classifier sees external kernel calls.** `llvm.emit_c_interface` callees classify memref operands from `llvm.readonly` / `llvm.writeonly`; unannotated operands stay `'u'` and no direction is guessed. | landed |
| H3 | **`AIRDialect::verifyOperationAttribute`** (`hasOperationAttrVerify = 1`), validating each `air.*` attribute's type and the op type it may sit on, as `GPUDialect` does. Starts with `air.disable_ping_pong` and `air.shim_dma_tile_sizes`. | landed `3428238b` |
| — | 522 lines of new compiler test coverage. | landed |

### In flight — H1, three bounded items

The spec was corrected mid-phase (see above): **skip and warn, do not abort.** What remains is
mechanical and has no open questions.

| # | Item | Why | Size |
|---|---|---|---|
| H1a | **Implement refuse → skip.** When the rotation cannot be proven safe for a buffer it privatizes, leave the loop single-buffered and emit a warning naming the loop. Refuse to *compile* only for genuinely malformed IR. | Three shipped models fail to build under the refusal spec. Correctness is preserved by skipping; only the optimization is lost. | small |
| H1b | **Re-specify the `hoisted` fixture clause.** With refusal gone it can no longer discriminate by demanding an error. It must assert: compiles, numerically correct, **and was not ping-pong transformed**. | Without the third clause it is a second copy of `inside` and proves nothing. | small |
| H1c | **Add the `CHECK-NOT: unroll` lit test** over the labeled IR, which is the natural home for H1b's third clause — asserting non-transformation from the Python runner is not straightforward. | The fixture proves numerics; the lit test proves the pass declined. | small |

Then re-run `gate-h.sh` to clear leg 4 (`make verify` over the ten shipped models), which is the
only leg that has never passed.

**`[2026-08-06]` H1b and H1c are done, and the work now runs as phase `H1s`, not as a resume.**
See [18](18-phase-h1s-skip-not-refuse.md). The driver made its own edits between phases: the
fixture's docstring is rewritten around the packet feed order and it now carries **four** variants
rather than two, `phases.sh:37`'s falsified description is fixed, the objective check asserts
compiles + exact + the per-buffer labeling decision, `guard_gate_files()` fingerprints
`mlir/test/**/*.mlir`, and `gate-h.sh` has a fifth leg comparing decode throughput to a recorded
floor. **H1a — the refuse → skip edit itself — is what remains**, plus one in-tree lit test.

### Not started

| # | Item | Why | Size |
|---|---|---|---|
| ~~H4~~ | ~~**Resolve the `air.disable_ping_pong` discrepancy.**~~ **`[2026-08-06]` STRUCK — measured working.** Set on a loop that IS otherwise labeled, `unroll` and `hoist_alloc` both go to zero and the buffers are not rotated; the attribute is present in every `--debug-ir` dump through the labeling pass, so the four hand-patched propagation sites are doing their job. The original "setting it changed nothing" was taken on a shape whose callee is unannotated and is therefore never labeled at all — it disabled something that was not happening, the same confound that made `--omit-ping-pong-transform=all` look exculpatory. Nothing to fix; do not promote it to an inherent attribute. | — | — |
| H5 | **Dynamic channel indices.** Split `air.channel`'s `indices` into a static dimension (selects flow/tile, must stay compile-time) and a dynamic one resolved by a runtime counter modulo depth. | `air.channel` indices are compile-time today, so a 64-band loop fully unrolls. **`[2026-08-06]` That unroll does not merely cost compile time — it exhausts the hardware.** The block configuration (4096x768, 8 cores, 64 bands) failed with `error: 'aie.lock' op lock assigned invalid id (maximum is 15)` in `air-to-aie`: 64 bands x 4 channels is far past the 16 locks a tile has. So the channel workaround is correct at 2 trips (measured, zero mismatches) but **cannot reach the block's band count at all**, which makes this item a prerequisite for that path rather than an optimization of it. **mlir-aie already solved this one layer down**: `-aie-objectFifo-stateful-transform`'s `dynamic-objFifos` uses a per-core counter plus `scf.index_switch`, and it is now the default, with static LCM unrolling as the legacy fallback. **`[2026-08-07]` The failure mode above is the mild one; J7b measured the other.** When the loop does NOT fully unroll, a channel put whose offset depends on the induction variable does not fail loudly — the BD chain cycles with that offset **frozen at its first value**, because `aie.dma_bd` offsets are static and nothing rewrites a chain into a form that can walk. The consumer stalls, the design's output DMA never fires, and the output buffer is returned byte-identical to what the host wrote; at scale it is `ERT_CMD_STATE_TIMEOUT` with no compile-time signal at all. So this item is not only an optimization blocker — it is a silent-miscompile source, and a **diagnostic** (the pass knows both that the offset is IV-dependent and that the loop will not unroll) is worth landing even if the dynamic-index work never is. See [23](23-rules-and-open-items.md) and `agents/probes/probe_ffn_accum_bd_offset.py`. | medium |
| H6 | **Per-region `omit_pingpong`.** `[2026-08-06]` **Re-scoped: try zero compiler change first.** The tiling half already has a per-launch override — `air.shim_dma_tile_sizes`, consumed at `AIRDependencyScheduleOpt.cpp:7868`, with the CLI merely outranking it — so a builder that OMITS `runtime_loop_tiling_sizes` and annotates each `air.launch` gets per-region tiling today. The ping-pong half works at loop granularity (see the struck H4). What is genuinely missing is only herd/segment granularity: `AIRDialect.cpp:71` restricts `air.disable_ping_pong` to `scf.for`/`scf.parallel`. **Step 1 is a measurement**, not C++: build the mixed FlashAttention + 4096-row-GEMM ELF with no tiling kwarg, per-launch attributes, and the opt-out on the attention herd's loops. | Still the gate on a single-ELF attention column pipeline (J7c): FlashAttention needs `omit_pingpong="all"` + `[1,1]` or it does not place, the GEMMs need `[2,2]`, and one ELF is one aircc invocation. | small |
| H7 | **Re-localize the offset-subview blocker.** `[2026-08-06]` **No longer unknown — two walls, and they are the same boundary from both sides.** (1) mlir-aie's `traceSubviewToBlockArgument` (`lib/Dialect/AIEX/Utils/AIEUtils.cpp:19`) already accumulates a nonzero byte offset and the caller adds it, but bails unless the subview is **rank-1 → rank-1**. (2) Upstream of it, `memref.cast` cannot erase a nonzero offset back to the identity layout a launch signature declares. The row-0 trick in `o_gemv_ffn_multi.py:142` works only because an offset-0 subview+cast is a no-op that folds away before either wall applies. **A third route needs neither change**: pass the band index as a launch operand and let the existing `dma_memcpy_nd(..., src_offsets=[row, 0], ...)` address it — which is what every banded builder here already does. Run the one small compile named in the note before scoping. | Blocks reusing a row-banded operator inside a fused module, and is doc 09's named fix for `fused`'s streamed norm tail. | bounded |
| H8 | **Automatic on-chip staging between pipeline stages** — see [the survey below](#what-air-automates-today-and-what-it-does-not). A pass that finds a memref written by exactly one hierarchy op, read by exactly one, with no host aliasing and dead afterwards, replaces it with a channel and demotes its memory space (L2, falling back to L3 on capacity). Plus launch fusion, so adjacent stages are co-resident to begin with. | Today memory space is an **input** to the AIR pipeline, never an output: no pass looks at a DDR buffer and decides it should not be there. This is the one piece of iron's dataflow that AIR cannot currently derive. | large, **and needs H2** — the analysis is unsound without the external-call classifier |

Prefer **inherent** over discardable for anything that must reach the backend. Erase on consume, as
the ping-pong labels already do. Do not attempt blind attribute propagation — upstream declined an
automatic mechanism twice; detect drops instead, as LLVM's `WarnMissedTransformationsPass` does.

### Harness gaps this tranche exposed

Not compiler work, but Phase H halted on all three and the next compiler phase will hit them again.

- **`guard_gate_files()` does not fingerprint `mlir/test/**/*.mlir`.** It covers `.lit` files, not
  `.mlir` inputs. Three separate weakened-gate halts in Phase H were lit tests edited to accommodate
  new behaviour, and the tamper check could see **none** of them — only the Codex `weakened_gates`
  layer caught them, three times running. Widen the set before the next `mlir/` phase.
- **`gate-h.sh` has no throughput leg.** Dropping ping-pong regressed a shipped model 12.4 → 7.8
  tok/s (recorded in `llms/shared/infra/backend_presets.py`), and nothing in the four legs would
  notice. A correctness-only gate cannot catch H1a skipping more than it should.
- **Leg 4 is where the expensive surprises live.** Both of Phase H's substantive spec errors
  surfaced there, an hour into ten `make verify` runs, after legs 1–3 were green. Run
  `check-air-mlir` yourself first — it takes seconds — but do not treat it as predictive: it stayed
  green through both.

## Tranche J — the study's essence

| # | Item | Why | Blocked on |
|---|---|---|---|
| J1 | **Collapse the norm dispatches.** Lift `builders/addnorm.py`'s one-trip guard and re-measure `coarse`. Expect 131 entries → ~5. | The single biggest structural divergence from iron. | **`[2026-08-06]` nothing — unblocked.** The blocker was never H1/H2; it was the packet feed order, fixed in `bfb647d9`. The fixture's `--variant inside` now runs two trips on hardware with zero mismatches at the exact shape the guard was written for. This is the highest-value item in the tranche and it is ready. |
| J2 | **Attention on device for `offload` and `runlist`.** `attn_scores` (4096×64×4096) already **passes on hardware** with hand-chosen tiles, zero mismatches — the registry was never a buildability constraint. `attn_output` (4096×4096×64) timed out on the one configuration tried, out of 828 legal ones; search the rest. | Today two modes run attention on the host and two on the device, so a mode-versus-mode comparison varies attention placement *and* dispatch boundary. Attention dominates the layer, so the confound is not small. | — |
| J3 | **Walk the `baseline_768` sequence ladder** for all four modes. E1 unblocked it; nothing has used it. | A tradeoff analysis at a single shape has no curves and therefore no crossover — which is the result the study exists to produce. | — |
| J4 | ~~**Replace distinguishability clause 3.**~~ **`[2026-08-08]` DONE**, `db2b1b53`. Clause 3 asserts `herd_launches` (404 vs 146 measured), not `runlist_entries` (391 vs 131, true by construction). **It took three coordinated edits, not one:** the selftest's violating fixture had runlist `herd=160` against coarse's 146, so flipping the clause alone would have left the fixture non-violating and the new clause silently unverified — the same shape as H10's objective-check flaw. The fixture drops to `herd=140` and keeps `entries=150` deliberately high, so it can only violate the *new* clause. Selftest 30/30 both directions; all four clauses then verified against real four-mode data ([09](09-phase-f-study-harness.md)). | A gate that cannot fail measures nothing. | ~~J1 re-measures both~~ — J1 will re-measure and the clause must be re-checked, but the clause no longer depends on J1 to be meaningful. |
| J5 | **Wire `decoder_gpt2`.** The oracle implements and tests it; no mode dispatches it. | It is the causal, LLM-shaped workload. Encoder-only makes this a study of a BERT layer, not of executing LLMs. | — |
| J6 | **Phase F cost metrics.** Latency, power, resource usage, GFLOPs — the seven studies in [09](09-phase-f-study-harness.md). | The dispatch vector is a proxy for cost, not cost. Until this lands there are no tradeoffs, only structure. | J2, J3 |

### J7 — pipelined `mha_out_proj` and `ffn` with on-chip partial-sum staging

> **`[2026-08-07]` J7a is built and gated** — see [21](21-phase-j7a-norm-tail-pipeline.md). The
> norm tail is three herds joined by L1→L1 channels with placement and buffer depth derived by the
> compiler, measuring `mean_rel_L1` 3.620e-3 against the 1.688e-2 target this section set. **J7b is
> staged** ([22](22-phase-j7b-accumulator-ring.md)), scoped to the FFN down-projection.
>
> Read [23](23-rules-and-open-items.md) before designing J7c: the per-column shim stream budget is
> what decides whether a stage is buildable, and the accumulator paragraph below names the one
> kernel whose ring the compiler cannot derive.

**`[2026-08-06]` Missing from the first draft of this document.** It is neither compiler work nor
covered by J1–J6, and it is the largest remaining structural difference between this port and iron.

**What iron does.** Its FFN down-projection and o-projection accumulate through an **L2
memtile-resident ring**: `matmul_with_acc_*` reads `of_o_acc_in` and writes `of_o_acc_out`, and the
memtile forwards the result back as the next call's accumulation input. Nothing in that reduction
tree leaves the chip. Its `mha_out_proj` is a four-stage **spatial pipeline down a column** — QK on
row 2, softmax on row 3, PV on row 4, o-projection on row 5 — with the FlashAttention O accumulator
held **resident in L1** across every KV block, rescaled in place by α and written to L2 only once
fully normalized.

**What this port does instead.** `drain` keeps an f32 accumulator in L1 with an in-GEMM drain-herd
cast — close to iron's spirit and numerically better. But `fused-cast` stages partials in a
**full-size f32 scratch in L3** (`qkv_f32`, `ffn_up_f32`, `ffn_out_f32` are whole `[seq, ffn]`
launch arguments), which is numerically better than iron and worse on movement: it round-trips a
full tensor through DDR where iron never leaves the memtile. And there is no pipeline anywhere —
the array is used as a data-parallel grid throughout.

**The pieces are already here and unused.**

- Three accumulate-into-C kernels are ported, compiled and exported —
  `matmul_with_acc_vectorized_2x2_mmul`, `matmul_with_acc_vectorized_1x4_mmul`,
  `matmul_with_acc_bf16_bf16_down_proj` — and `grep matmul_with_acc` across every builder and mode
  returns **nothing**. They have never been dispatched.
- `programming_examples/bottleneck/` is a working multi-herd spatial pipeline: heterogeneous named
  herds inside one segment, joined by `Channel("L1ToL1_...", broadcast_shape=...)` with
  `ChannelPut`/`ChannelGet` between stages, plus `L3ToL2_*`/`L2ToL1_*` channels doing the job of
  iron's `.split()`/`.forward()`/`.join()`.
- `channel_examples/worker_to_self/` is the feedback ring — a channel from a herd back to itself,
  with explicit `MemorySpace.L1` and `MemorySpace.L2` — which is the shape iron's accumulator ring
  needs.

So this is **unbuilt, not blocked**. The transformer-layer port reached for AIR's data-parallel grid
throughout and never used its dataflow constructs, even though both are demonstrated in-tree.

**Why it is worth doing, in order of confidence.**

1. **It recovers `fused`'s lost precision.** That mode measures `mean_rel_L1` 1.806e-2 against the
   block's 1.688e-2, and the cause is named in its own README: the tail decomposes `addnorm` into
   `elementwise_add` → `layer_norm` → `elementwise_mul` and **stages bf16 through L3 between them**.
   A two- or three-stage herd pipeline with L1→L1 channels keeps those intermediates resident — which
   is exactly iron's `AIEAddAndNorm`, a two-worker pipeline. Precision recovered without giving up
   the fusion.
2. **It is the honest comparison.** Today a `coarse`-versus-`hybrid` latency number measures
   persistent-worker streaming against launch-grid dispatch at least as much as the execution
   boundary. Building the pipelined form makes the two comparable on their own terms.
3. **Dispatch count.** A pipeline is one launch where the decomposition is several.

**Sequencing.** Independent of H and of J1–J6, and it does not need the `addnorm` one-trip rule
lifted. It is builder work in `transformer_layer/builders/` plus a mode variant, gated exactly as
every operator here is: full-output `np.isclose` against the FP32 oracle at `rtol` 1.6e-2, zero
mismatches, with a fault-injection control. Start with the **norm tail** — it is the smallest piece,
it has a measured precision target to beat (1.806e-2 → 1.688e-2), and it proves the L1→L1 channel
path on this example before the harder `mha_out_proj` pipeline.

### What AIR automates today, and what it does not

`[2026-08-06]` Read before designing J7. The question is whether these mappings must be written by
hand the way iron writes them. The answer splits, and the split decides how much of J7 is work.

**The accumulator half is already automatic.** `air-hoist-dma-in-accum-pattern` runs
**unconditionally, second in the pipeline** (`tools/aircc/aircc.cpp:837`). It matches an incoming and
an outgoing DMA on the same memref with mirrored offsets/sizes/strides
(`AIRDependencyScheduleOpt.cpp:322`), both loop-invariant, and hoists both out of the loop —
leaving the buffer L1-resident across the whole reduction. That is iron's `of_o_acc_in` /
`of_o_acc_out` ring, derived rather than declared.

Critically it is **purely syntactic on the DMA ops** and never asks what the kernel does, so an
opaque external `func.call` between the fetch and the store does not block it. **J7 does not need to
hand-build the accumulator ring.**

> ### `[2026-08-06]` But under two conditions this paragraph originally left out
>
> It used to end "write a KV-block loop that fetches C, calls `matmul_with_acc_vectorized_2x2_mmul`,
> stores C — and the round-trip collapses on its own." The conclusion is right; **both specifics were
> wrong**, and either one alone silently costs you the ring. Measured with
> `air-opt --air-dependency,--air-hoist-dma-in-accum-pattern`, counting data-movement ops left inside
> the K loop (A and B fetches must stay; the accumulator's two should lift):
>
> | accumulator kernel | L1 buffers allocated | before → after | ring? |
> |---|---|---|---|
> | in-place — one `C`, read-add-write | outside the loop (herd scope) | 4 → 4 | **no** |
> | in-place — one `C`, read-add-write | **inside the loop** | **4 → 2** | **yes** |
> | two-buffer — `pAcc` in, `C` out | outside the loop | 4 → 4 | no |
> | two-buffer — `pAcc` in, `C` out | inside the loop | 4 → 4 | no |
>
> **Condition 1 — it must be the in-place kernel.** `areSymmetricDmaOps` requires the incoming DMA's
> destination and the outgoing DMA's source to be the *same* memref. `ffn_matmul_bf16_bf16_up_proj(A,
> B, C)` reads C, adds, writes C — one memref, matches.
> `ffn_matmul_with_acc_bf16_bf16_down_proj(A, B, pAcc, C)` uses two, both `__restrict` so they may not
> alias, and **never matches at any alloc site**. That is the kernel this paragraph used to name, and
> it is a faithful match for iron's `of_o_acc_in`/`of_o_acc_out` pair — which is precisely why it
> needs a hand-built ring and the in-place form does not. `encoder_matmul.cc:26` flags the
> distinction as a footgun in its own words.
>
> **Condition 2 — the accumulator must be allocated *inside* the loop.** Counter-intuitive: you write
> "allocate per iteration" to get "resident across iterations". `isIncomingDmaOp` requires the DMA to
> depend on both the loop's first iter-arg **and** an `air.execute` holding a `memref.alloc`;
> `isOutgoingDmaOp` requires its token's users to include both an `air.wait_all` **and** an
> `air.execute` holding a `memref.dealloc`. Allocate at herd scope — the natural way — and neither
> predicate holds. The pass hoists the alloc and the mirrored DMA pair *together*.
>
> **Consequence for J7b's gate:** assert from the `--debug-ir` dump that the C DMAs left the K loop.
> The numbers are identical whether the ring formed or not, so a DDR round-trip per K step is
> invisible to `np.isclose` — and it is the entire thing J7b exists to remove. Two of the four cells
> above would have shipped as working code.

**The inter-stage half is not automatic, and cannot be today.** Memory space is an input to the AIR
pipeline. There is no producer-consumer forwarding pass and no launch fusion —
`air-fuse-parallel-launch` is about `scf.parallel` around a herd (`AIRMiscPasses.cpp:900`), not about
merging two `air.launch`s. `air-override-memref-memory-space` is flagged experimental and only
rewrites allocs *inside* a region; `qkv_f32` and friends are launch arguments, and `block.py:550`
lists them in `host_writes` — host-visible ABI the compiler is not permitted to touch.

So the DDR round-trip is not a missed optimization. **Our builders declared it**, and the mode
boundary (one op per launch in `offload` / `runlist` / `coarse`) forces it structurally. `fused` is
the only mode that need not, and it still does.

**What is less manual than iron either way.** Declaring the edge is the whole of what J7 writes; the
compiler decides the rest, where iron makes you spell each one out:

| decision | iron | AIR |
|---|---|---|
| tile placement | `placement=Tile(col,row)` | `air-place-herds` |
| buffer depth | `depth=2` | `air-label-scf-for-to-ping-pong` + `air-ping-pong-transform` |
| memtile assignment and sharding | by hand | `air-split-l2-memref`, capped by the real shim budget (`aircc.cpp:869`) |
| DMA BDs, wrap-and-stride | by hand | `air-opt-shim-dma-bds`, `air-opt-memtile-dma-bds` |
| broadcast fan-out | `of.cons(n)` | `air-broadcast-detection` + `air-specialize-dma-broadcast` |
| sharing one physical channel across flows | no equivalent | `air-fuse-channels{aggressive-mode=L1,L2,L3}` |

**Consequence for sequencing.** Do J7 by hand first, and do not wait on H8. Declaring three channels
is a builder edit; the promotion pass is large, needs H2 underneath it to be sound at all, and wants
a hand-written reference dataflow to validate against — which is precisely what J7 produces.

## The finish line

Not "it feels done". [13](13-verification-and-acceptance.md) already defines it:

- **Phase F gate** — `unattended_reboot execution-smoke-test` yields **≥1 `run_status=passed` row per
  measurement CSV**, and reports the first `failure_message` verbatim. A broken environment still
  writes complete, well-formed CSVs full of failed rows; iron shipped a smoke test that reported
  21/21 passed on a machine where every measurement had failed.
- **Phase G gate** — a full profile run with a complete `results_manifest.json`.

When those pass over the four modes across the ladder, with `attention_path` and the norm-staging
difference recorded beside every row, the port captures the study's essence: it can say what each
way of executing a transformer layer on this NPU costs, and why.
