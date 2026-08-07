# 24 — Phase H10: a non-constant BD offset is dereferenced, not diagnosed

`air-to-aie` lowers a channel put's offset into a **static** `aie.dma_bd` offset. When that offset
is not a compile-time constant — an induction variable, which is how anyone walks a staged buffer —
the lowering dereferences a disengaged `std::optional` and carries on. No error, no warning. The
design compiles, places, routes, and hangs on hardware.

J7b lost its implement session to this. The rest of that session's work was correct.

## The defect, located

Two unchecked dereferences in `mlir/lib/Conversion/AIRToAIESchedulingUtils.cpp`:

```cpp
// air::get1DOffset, lines ~199-215
auto offset = mlir::getConstantIntValue(memcpy_offsets[i]);
if ((unsigned)i == memcpy_offsets.size() - 1)
  one_d_offset += *offset;                       // <-- no has_value()
else {
  auto stride_i = mlir::getConstantIntValue(memcpy_strides[i]);
  one_d_offset += (*offset) * (*stride_i);       // <-- nor here
}
```

```cpp
// ~line 460, building AIE::BDDimLayoutAttr from the wraps and strides
auto stepsize = mlir::getConstantIntValue(stepsizeVal);
auto wrap = mlir::getConstantIntValue(wrapVal);
auto tuple = AIE::BDDimLayoutAttr::get(ctx, *wrap, *stepsize);   // <-- same class
```

`mlir::getConstantIntValue` returns `std::nullopt` **exactly** when the value is not a compile-time
constant. Dereferencing that is undefined behaviour; what was observed is a silent `0`.

**The same file gets it right twice**, so the idiom is known and applied inconsistently:

| site | handling |
|---|---|
| `get1DOffset` (206, 210) | `*offset`, `*stride_i` — unchecked |
| BD dim layout (462–464) | `*wrap`, `*stepsize` — unchecked |
| line 527 | `auto c = getConstantIntValue(v); if (!c) return std::nullopt;` ✅ |
| line 945 | `if (!ca \|\| !cb \|\| *ca != *cb) return false;` ✅ |

The only caller of `get1DOffset` is `AIRToAIEPass.cpp:6527`, computing the BD offset. The line
immediately above it — `if (auto const_highest_stride = getConstantIntValue(strides[0]))` — checks.

## The evidence, end to end

Measured on J7b's pre-fix builder at 4 K steps (`agents/probes/probe_ffn_accum_bd_offset.py`):

1. **The IR really does carry a non-constant offset.** In the dump immediately before `air-to-aie`:

   ```mlir
   %5 = scf.for %arg6 = %c0 to %c128 step %c32 iter_args(...) {
     %7 = affine.apply #map()[%arg6]
     %8 = air.channel.put ... (%arg4[%7] [2048] [1]) : (memref<8192xbf16>)
   ```

   Contrast the accumulator fetch on the next line, `%arg5[0, 0, 0, 0]` — all literals, which is
   why the C BDs were correct and the A BD was not.

2. **The offset comes out frozen.** The memtile MM2S chain, at `herd_x=1`, block 2048:

   ```
   ksteps=2   [0, 0, 2048, 0]    the loop fully unrolls; each put has a literal offset
   ksteps=4   [0, 0,    0, 0]    it does not; every A BD reads 0
   ```

3. **The hardware consequence.** The consumer stalls, the design's output DMA never fires, and the
   output buffer is returned **byte-identical to what the host wrote** — seed it with 1.0 and
   4096/4096 elements come back 1.0. At 4 columns × 96 steps it stops returning at all:
   `ERT_CMD_STATE_TIMEOUT`. Not ping-pong; `omit_pingpong="all"` is identical.

4. **Nothing warns, at any stage.** The structural checks are green: 4 → 2 hoist, zero packet-typed
   channels, full compile.

## The scope — and a correction this spec got wrong

> **`[2026-08-07]` CORRECTED. The paragraph below claimed the refusal cannot reach L3 transfers
> because `get1DOffset`'s caller takes an `AIE::TileLike tile`. That inference was wrong, and it
> cost the implement session a round.** `TileLike` is an *interface*, which shim tiles implement
> too, and `generateDmaBdProgram` is instantiated **twice**:
>
> ```
> generateDmaBdProgram<air::TileDMAAllocator, AIE::BufferOp,         AIE::MemOp>      // core/memtile
> generateDmaBdProgram<air::ShimDMAAllocator, AIE::ExternalBufferOp, AIE::ShimDMAOp>  // SHIM
> ```
>
> Both reach `generateDmaBd` → `get1DOffset`, so a refusal placed there fires on **shim** BDs as
> well, where a moving offset is legitimate. Scope the refusal to the `TileDMAAllocator`
> instantiation.
>
> **That is a real correction, but it is NOT why the three existing lit tests fail.** Measured:
> `async_gemm_to_locks_aie2.mlir` and the two `async_gemm_w_pingpong_*` contain **four L2
> (`memref<64x64xi32, 1>`) channel ops whose offsets are `scf.for` induction variables, in the
> input as written** — before any pass runs. They encode the losing construction on the tile side,
> so the refusal fires on them **correctly**.
>
> Which makes them category (a) from the allowlist note: *designs relying on the undefined
> behaviour, a finding to report rather than tests to edit*. Their CHECK lines assert AIE
> structure — tiles, locks, flows — and never a `dma_bd` offset, so a frozen offset was invisible
> to them. **They have been passing over a construction whose BD offsets are silently zero.**
> Deciding what those tests should assert instead is the human call the allowlist exists to force.
>
> Two wrong diagnoses to not repeat, both corrected by measurement:
> - The implement session reported that the tests skip `-air-specialize-channel-wrap-and-stride`,
>   which "the production aircc pipeline runs". **That pass is not in aircc's pipeline at all** —
>   it is `air-dependency, air-hoist-dma-in-accum-pattern, [broadcast], air-dma-to-channel, …,
>   air-to-aie`, and no dump for it appears in J7b's real compile. Do not normalize the tests on
>   that basis.
> - This document first claimed the tests were all-L3 and the refusal was simply too broad. That
>   came from grepping for `memref<…, 1 : i32>`; these tests use the short form
>   `memref<…, 1>`, so the L2 operands did not match. A false negative from a pattern, not from
>   the IR.

`get1DOffset`'s callers sit inside `generateDmaBd`, reached from **both** the tile-side and the
shim-side BD programs (see the correction above). L3-side transfers are programmed by the runtime
sequence (`AIRRtToNpuPass`), which materializes offsets per task and can express one that advances.

The IR bears this out exactly. In J7b's failing module, three channel puts carry an offset:

| put | operand | offset | outcome |
|---|---|---|---|
| W chunk refill | `%arg4` — launch arg, **no memory space (L3)** | `%7`, IV-dependent | **fine** |
| A feed | `%results` — `memref<8192xbf16, 1 : i32>`, **L2** | `%13`, IV-dependent | **silently wrong** |
| B feed | `%results_4` — `memref<2048xbf16, 1 : i32>`, L2 | none (whole buffer) | fine |

So the rule the J7b builder arrived at empirically — *advance on the L3 side, never on the L2 read*
— is exactly the mechanism: **an IV-dependent offset is materializable on an L3 operand and
inexpressible on an L2/L1 one.**

**A survey of every `ChannelPut`/`ChannelGet` with a non-literal offset in `programming_examples/`
found no other case on an L2/L1 operand.** The nearest miss is
`attention_decode/attn_decode_npu2.py:347`, which puts `offsets=[tx_i, mm_iter, 0, ...]` with
`mm_iter` a genuine `scf.for` IV over 6 trips — but from `l3_b_data`, an **L3** buffer, so it goes
through the runtime sequence and is safe. `herd_dataflow/run.py` uses an `affine.apply` offset
inside an `scf.forall`, which AIR specializes per index.

That is a static survey, not a build of all ten shipped models, so premise 4 below still stands —
but it means the expected outcome of leg 5 is "nothing breaks", and a leg-5 failure is a signal to
investigate rather than to relax the refusal.

## Why this is refuse, not skip

H1s settled *skip and warn, never refuse* for `air-label-scf-for-to-ping-pong`, and that was right
**because declining leaves a correct single-buffered loop** — only the optimization is lost.

This is the opposite, and the same shape as `air-fuse-packet-put-loops` (doc 23): **there is no
correct fallback.** A BD cannot express a per-iteration offset. Continuing emits a chain that
addresses the wrong memory forever. The untransformed program is the broken one, so the only honest
outcomes are to refuse, or to lower into a form that can walk (H5's dynamic indexing, which
mlir-aie already solved one layer down with `dynamic-objFifos`: a per-core counter plus
`scf.index_switch`).

**Scope this phase to refusing well.** The dynamic-index lowering is H5 and is much larger.

## What to build

1. **`get1DOffset` returns `std::optional<int64_t>`** (or takes a failure out-param), returning
   `nullopt` when any offset or stride it needs is non-constant. Do not paper over it with a
   default of 0 — that is the current behaviour spelled out loud.
2. **The caller emits a diagnostic and fails**, naming the channel, the offending operand and the
   loop, e.g.: *"channel put offset is not compile-time constant (`affine.apply` of a loop
   induction variable); an `aie.dma_bd` offset is static and cannot advance per iteration. Stage
   the operand per iteration from L3 instead, or see H5."* The message must say what to do — this
   defect's whole cost was that the failure gave no hint.
3. **The same treatment for the BD dim layout site** (462–464). A non-constant size or stride is
   the same class and is currently the same UB.
4. **Lit coverage in `mlir/test/`**: a module with an IV-dependent channel-put offset must produce
   the diagnostic. Keep an existing constant-offset test untouched as the passing control.

## Gate

`gate-h.sh`'s five legs — build + install, `check-air-mlir`, the transformer-layer suite, decode
throughput against the recorded floor, and `make verify` over the ten shipped models.

**Leg 5 is the one that matters here** and the one to expect surprises from: this change turns a
silently-miscompiling construction into a compile error, so **any shipped design that currently
relies on it stops building**. That is the point, but it must be looked at rather than assumed —
if a shipped model does hit it, that model has been computing on a frozen BD offset and the finding
is much larger than this phase.

The objective check is wired (`phase_h10_objective_check`) and has **three** clauses, because one
alone would be satisfied by refusing everything:

1. **The phase.** An IV-dependent channel-put offset on an **L2** operand is refused, by a message
   naming a non-constant/static offset — not by the generic core-ELF link failure every one of
   these compiles ends with.
2. **Not unconditional.** The same shape with a **constant** offset still gets past the MLIR
   pipeline.
3. **Correctly scoped.** An IV-dependent offset on an **L3** operand still compiles. This is the
   clause that keeps the fix from being worse than the defect: every shipped design that walks a
   buffer does it this way, and a refusal that caught them would break the fleet to fix a bug none
   of them have.

**Verified failing before the phase starts**, which is the discipline H9's `multicolumn` clause
exists for. On today's compiler, clause 1 fails — the construction compiles clean through the MLIR
pipeline and dies only at the link, with nothing said about the offset — while 2 and 3 pass
trivially because nothing is refused yet. They become load-bearing the moment the diagnostic lands.

## What this phase must not do

- **Do not implement dynamic BD indexing.** That is H5. Refusing correctly is this phase.
- **Do not widen a tolerance or touch a gate file** outside `mlir/test/`, and there keep the inputs
  and update only the CHECK lines — the rule `lib-guard.sh` records.
- **Do not "fix" the two checked sites (527, 945) to match the unchecked ones.** They are the
  correct ones.

## Premise status

Everything above is measured except one thing, and it is the first thing to check:

1. Two unchecked dereferences, with two correct siblings in the same file — **read, confirmed.** ✅
2. The IR carries a non-constant offset at that point — **confirmed** from the pre-`air-to-aie`
   dump. ✅
3. The BD offset comes out frozen and the design hangs — **measured on hardware.** ✅
4. ~~The refusal fires only on tile-side BDs, so an IV-dependent **L3** offset is untouched —
   read from the caller and confirmed against the IR.~~ **FALSE, corrected `[2026-08-07]`.** This
   was inferred from the caller's `AIE::TileLike` parameter without checking its instantiations;
   `generateDmaBdProgram` is instantiated for `ShimDMAAllocator` as well, so a refusal in
   `generateDmaBd` reaches shim BDs. **Scope it to the `TileDMAAllocator` instantiation.** The IR
   table above is still correct — it is the *conclusion drawn from it* that was wrong. ❌
5. **That no shipped model relies on the losing construction — NOT established by a build.** A
   static survey of every non-literal channel offset in `programming_examples/` found none on an
   L2/L1 operand, so the expectation is that nothing breaks; leg 5 is what actually answers it. If
   a model does break, stop and report — it has been computing on a frozen BD offset, and the
   finding is much larger than this phase.

   **`[2026-08-07]` The compiler's OWN tests do rely on it**: three `mlir/test/Conversion/AIRToAIE`
   tests carry four L2 channel ops with induction-variable offsets. That is a stronger version of
   the same warning — the construction was reachable enough to be written into the regression
   suite — and it means leg 2 (`check-air-mlir`) answers this question before leg 5 does.
