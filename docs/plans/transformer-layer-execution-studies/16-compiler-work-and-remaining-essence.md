# 16 — The compiler work, and what "the essence" still needs

`[2026-08-06]` Phase E landed four execution strategies that agree numerically and separate on the
dispatch vector. This document is what stands between that and the study the port set out to be:
*an analysis of ways to execute LLMs on the Ryzen NPU and their tradeoffs.*

Two tranches. **H** is compiler work in `mlir/` — it has the largest blast radius in the plan, since
every shipped model compiles through it. **J** is the study itself. They are separated because their
gates are different, not because they are independent: three of J's items are blocked on H.

Everything below was measured or read on 2026-08-05/06. Nothing here is speculative.

## The root cause, established

`builders/addnorm.py` forbids more than one trip of its row loop. That single rule turns 4096 rows
into **64 host dispatches**, which is 128 of `coarse`'s 131 runlist entries — the largest structural
gap between this port and iron's `hybrid` (5 entries). The builder's docstring blamed ping-pong
buffering. The real cause is narrower and worse:

1. `air-dma-to-channel` splits an L3→L1 `dma_memcpy_nd` into an internal `air.channel.get` (writes
   L1) and an external `air.channel.put` marked `hoist`. **The channel bundle is indexed by spatial
   IDs only** — herd and `scf.parallel` — so the temporal `scf.for` IV is not part of channel
   identity (`AIRDmaToChannel.cpp`).
2. `air-label-scf-for-to-ping-pong` marks the loop `unroll = 2` and every candidate alloc
   `hoist_alloc`, having checked only that no alloc is filled by more than one non-exclusive
   `channel.get` per iteration. It does **not** require exactly one producer, a producer for *both*
   duplicated halves, a recognized consumer, or order-preserving correspondence with the hoisted
   external endpoint (`AIRDependencyScheduleOpt.cpp::isPingPongCandidate`).
3. `checkOpOperandReadOrWrite` (`mlir/lib/Util/Util.cpp`) classifies a use via memory effects,
   `ChannelPutOp`, `ChannelGetOp` or linalg, and returns `'u'` otherwise. **An external kernel
   `func.call` has no registered memory effects, so the compute step is invisible.**
4. Unknown uses are silently omitted from dependency construction, and an empty producer or consumer
   set becomes `air::WaitAllOp::create(..., SmallVector<Value>{})` — a dependency-free placeholder
   rather than a rejection.

Net: the ping/pong halves get **no reuse edge** protecting a buffer until the kernel has finished
reading it. One trip is safe because nothing is reused; two trips corrupt. Measured at
`cols=64, rows=8, rows_per_call=4`: 481–497 of 512 elements wrong.

Two independent confirmations. Rewriting the same computation with `air.channel` passes at two trips
with zero mismatches and a one-trip control — because `ChannelGet`/`ChannelPut` *are* in the
classifier. And hoisting the weight DMA out of the loop corrupts, which the builder already
documents, for the same missing-edge reason.

**The channel form is not the fix, though.** At the block's own configuration it fails in
`air-to-aie` with `'aie.lock' op lock assigned invalid id (maximum is 15)` — the 64-band unroll
needs far more locks than a tile has. It is a diagnostic that isolates the cause, not a route to
one dispatch. Fixing the dependency analysis (H1 + H2) is what makes the DMA path work multi-trip;
H5 is what would make the channel path viable at scale.

**Every comparable compiler refuses to transform when it cannot establish the invariant.** Upstream
`memref::multiBuffer` returns `failure()` unless it can point at a user that provably clobbers the
whole buffer each iteration — and its `overrideBuffer()` only recognizes `memref.copy`, so a custom
DMA op would not qualify. IREE calls it with `skipOverrideAnalysis=false` and never forces it.
Triton gates entry on an explicit precondition list. TVM `ICHECK`-aborts. Silently emitting wrong
numbers is the outlier behaviour here, and fixing *that* is worth more than the optimization.

## Tranche H — compiler

Gate for every H item: mlir-air's own lit suite, the transformer-layer suite, **and** `make verify`
over the ten shipped models. `gate-e1.sh` is the model; H needs a build step in front of it.

| # | Item | Why | Size |
|---|---|---|---|
| H1 | **Bail out instead of miscompiling.** Reject ping-pong candidacy unless every alloc has a recognized producer for *both* halves and at least one recognized consumer, and no relevant use classifies `'u'`. Emit a diagnostic naming the loop. | Converts a silent 481/512 wrong answer into a compile error. It is what `builders/addnorm.py`'s Python guard is standing in for. | small |
| H2 | **Teach the classifier about external kernel calls.** A `func.call` whose callee carries `llvm.emit_c_interface` should classify its memref operands from the callee's argument attributes rather than falling through to `'u'`. | This is the actual missing edge. With it, H1's bail-out stops firing on the legitimate case and multi-trip becomes correct. | small–medium |
| H3 | **`AIRDialect::verifyOperationAttribute`** (`hasOperationAttrVerify = 1`). Validate every `air.*` discardable attribute and the op type it may sit on, as `GPUDialect` does. | Absent entirely today. Runs on every op after every pass under `-verify-each`, so a misplaced or mistyped attribute is caught at the pass that broke it. | small |
| H4 | **Resolve the `air.disable_ping_pong` discrepancy.** `isPingPongCandidate` checks it, yet setting it on the row loop changed nothing — both arms produced byte-identical 481/512. Determine whether it is dropped, attached to the wrong op after rewrites, or read too late. If dropped, promote it from discardable to an **inherent** ODS attribute. | A documented opt-out that does not work is worse than none. Discardable attributes may legitimately be dropped by any pass that does not know them; PR #1664 already hand-patched four such sites. | small |
| H5 | **Dynamic channel indices.** Split `air.channel`'s `indices` into a static dimension (selects flow/tile, must stay compile-time) and a dynamic one resolved by a runtime counter modulo depth. | `air.channel` indices are compile-time today, so a 64-band loop fully unrolls. **`[2026-08-06]` That unroll does not merely cost compile time — it exhausts the hardware.** The block configuration (4096x768, 8 cores, 64 bands) failed with `error: 'aie.lock' op lock assigned invalid id (maximum is 15)` in `air-to-aie`: 64 bands x 4 channels is far past the 16 locks a tile has. So the channel workaround is correct at 2 trips (measured, zero mismatches) but **cannot reach the block's band count at all**, which makes this item a prerequisite for that path rather than an optimization of it. **mlir-aie already solved this one layer down**: `-aie-objectFifo-stateful-transform`'s `dynamic-objFifos` uses a per-core counter plus `scf.index_switch`, and it is now the default, with static LCM unrolling as the legacy fallback. | medium |
| H6 | **Per-region `omit_pingpong`.** Teach `air-label-scf-for-to-ping-pong` to read a per-herd/per-segment attribute layered over its module-wide option. | FlashAttention needs `omit_pingpong="all"` + `runtime_loop_tiling_sizes=[1,1]`; the 4096-row GEMMs need `[2,2]`. One ELF is one aircc invocation, so `fused` is 3 ELFs instead of 1. It also costs codegen *inside* existing ELFs — one K=8192 launch forces every sibling in `o_ffn` to give up ping-pong. **The transform pass needs no change**: the flow is already label→transform and already does `removeAttr` on consume. Prototype with `transform.apply_registered_pass` first — AIR has `air-transform` and `AIRTransformOps.td` today, so this needs no C++ to validate. | medium |
| H7 | **Re-localize the offset-subview blocker.** A `memref.subview` at a nonzero offset cannot reach a launch argument. **`air.launch` is not the blocker** — its ODS is `Variadic<AnyType>`. The two `isIdentity()` gates found so far are narrow (cascade channels, `#air.symmetric_heap`), so the real wall is elsewhere, likely the Python builder's signature. Find it before scoping. | Blocks reusing a row-banded operator inside a fused module, and iron-style banded addressing with the offset baked into the instruction stream. | unknown |

Prefer **inherent** over discardable for anything that must reach the backend. Erase on consume, as
the ping-pong labels already do. Do not attempt blind attribute propagation — upstream declined an
automatic mechanism twice; detect drops instead, as LLVM's `WarnMissedTransformationsPass` does.

## Tranche J — the study's essence

| # | Item | Why | Blocked on |
|---|---|---|---|
| J1 | **Collapse the norm dispatches.** With H1+H2, lift `builders/addnorm.py`'s one-trip guard and re-measure `coarse`. Expect 131 entries → ~5. | The single biggest structural divergence from iron. | H1, H2 |
| J2 | **Attention on device for `offload` and `runlist`.** `attn_scores` (4096×64×4096) already **passes on hardware** with hand-chosen tiles, zero mismatches — the registry was never a buildability constraint. `attn_output` (4096×4096×64) timed out on the one configuration tried, out of 828 legal ones; search the rest. | Today two modes run attention on the host and two on the device, so a mode-versus-mode comparison varies attention placement *and* dispatch boundary. Attention dominates the layer, so the confound is not small. | — |
| J3 | **Walk the `baseline_768` sequence ladder** for all four modes. E1 unblocked it; nothing has used it. | A tradeoff analysis at a single shape has no curves and therefore no crossover — which is the result the study exists to produce. | — |
| J4 | **Replace distinguishability clause 3.** `runlist entries > coarse entries` is now true by construction. Use `herd_launches` (404 vs 146), which counts executed work rather than dispatch packaging and which neither mode fixes by construction. | A gate that cannot fail measures nothing. | J1 re-measures both |
| J5 | **Wire `decoder_gpt2`.** The oracle implements and tests it; no mode dispatches it. | It is the causal, LLM-shaped workload. Encoder-only makes this a study of a BERT layer, not of executing LLMs. | — |
| J6 | **Phase F cost metrics.** Latency, power, resource usage, GFLOPs — the seven studies in [09](09-phase-f-study-harness.md). | The dispatch vector is a proxy for cost, not cost. Until this lands there are no tradeoffs, only structure. | J2, J3 |

### J7 — pipelined `mha_out_proj` and `ffn` with on-chip partial-sum staging

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
