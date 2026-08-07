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

## The scope, which is narrower than it first looks — and why nothing shipped breaks

`get1DOffset`'s only caller sits inside `generateDmaBd(..., AIE::TileLike tile, ...)`: the
**tile-side** (core and memtile) BD path. L3-side transfers are programmed by the runtime sequence
(`AIRRtToNpuPass`), which materializes offsets per task and can express one that advances.

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

An objective check with two clauses:

1. The diagnostic fires on the minimal shape (an IV-dependent channel-put offset), by message.
2. **The pre-fix J7b construction is rejected.** Build `e6cdd138`'s `ffn_accum` builder at 4 K
   steps and require a compile failure naming the offset. This is the phase's real claim: the exact
   program that hung now refuses.

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
4. The refusal fires only on tile-side BDs, so an IV-dependent **L3** offset is untouched —
   **read from the caller and confirmed against the IR** (the table above). ✅
5. **That no shipped model relies on the losing construction — NOT established by a build.** A
   static survey of every non-literal channel offset in `programming_examples/` found none on an
   L2/L1 operand, so the expectation is that nothing breaks; leg 5 is what actually answers it. If
   a model does break, stop and report — it has been computing on a frozen BD offset, and the
   finding is much larger than this phase.
