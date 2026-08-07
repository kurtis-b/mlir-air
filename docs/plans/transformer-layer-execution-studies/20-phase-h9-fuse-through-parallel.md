# 20 — Phase H9: fuse packet put loops through `scf.parallel`

The defect that blocks J1, and the reason Phase H's fix looked complete when it was not.

Read [19](19-phase-j1-collapse-norm-dispatches.md) for what J1 measured, and the "Phase J1
findings" section of `programming_examples/transformer_layer/README.md` for the walk table.

## What is wrong

`air-fuse-packet-put-loops` was Phase H's fix for the two-trip `addnorm` miscompile: sibling
per-channel put loops feeding one packet-multiplexed shim queue get fused into a single loop that
issues the puts in program order, restoring the per-iteration interleave the consuming tile's BD
ring expects.

It works. On **one column**. `AIRFusePacketPutLoops::runOnOperation`
(`mlir/lib/Transform/AIRDependencyScheduleOpt.cpp:4840`) collects blocks like this:

```cpp
module.walk([&](air::LaunchOp l) {
  for (auto &b : l.getBody())
    blocks.push_back(&b);
});
for (auto *b : blocks)
  runOnBlock(b, builder);
```

Only `air.launch`'s **immediate** body blocks. At `herd_x ≥ 2`, `air-dma-to-channel` wraps the
per-tile put loops in an `scf.parallel`, so they live in the parallel's body block — which is never
visited. The pass's output IR is byte-identical to its input, and the shim feed-order corruption
returns from the second trip on.

**Measured on NPU2**, all at two trips:

| shape | result |
|---|---|
| `herd_x=1`, cols 64 and 768 | exact |
| **`herd_x=8`, cols 64** | **4070 / 4096 wrong** |
| **`herd_x=8`, cols 768** | **97726 / 98304 wrong** |
| J1's target — 64 trips, 4096×768, `herd_x=8` | compiles, 3,130,958 / 3,145,728 wrong |

Silent every time. Nothing refuses; the numbers are simply wrong.

**Why nobody caught it.** Every clause in the driver's fixture ran at `herd_x=1` — the width the
original miscompile happened to be measured at. Four green variants coexisted with a live silent
miscompile one column wider for an entire phase. The fixture now has a fifth, `multicolumn`, which
**fails today** (verified: 3747+ mismatches) and must pass when you are done.

## What to do

Make the pass see put-loop groups nested inside region-holding ops within the launch — at minimum
`scf.parallel`, which is what `air-dma-to-channel` actually emits. `runOnBlock`'s grouping, its
`touchesPacketStream` sealing and its dominance checks are written against a single block and
should be reusable per nested block; the mechanical part is choosing which blocks to walk.

**The substance is the correctness argument, and it is yours to make, not to assume.**
`runOnBlock` reasons about **one shared shim queue's task order**. Inside an `scf.parallel` over
columns, each iteration is a different tile. You must establish:

- that fusing *within* one parallel iteration restores the interleave that iteration's consumer
  ring was built from; and
- that doing so reorders nothing *across* iterations that the shared stream depends on.

If you cannot establish the second, **say so and leave the pass alone.** A second silent
miscompile is far worse than a missing optimization, and this defect class has now cost two
phases. `work_not_completed` with the reason is a good outcome; a guess is not.

Consider also whether the honest fix is upstream: running the fusion **before** the per-tile
specialization that introduces the `scf.parallel` wrapper would sidestep the nested-block question
entirely. Check what the pipeline order permits (`tools/aircc/aircc.cpp`,
`buildOptimizationPipeline`) — the pass currently sits after the last
`air-isolate-async-dma-loop-nests` deliberately, because that pass would otherwise re-split the
fused loop. Whichever route you take, write down why the other was rejected.

## What this phase must NOT do

- **Do not touch `builders/addnorm.py`.** Its guard is now at the measured boundary (multi-trip
  only at `herd_x=1`) and widening it is J1's job, after this lands.
- **Do not widen the fixture's tolerance** or narrow its shape. It is driver-owned, fingerprinted,
  and in no allowlist.
- **Do not disable ping-pong** anywhere. It is not the mechanism — `--omit-ping-pong-transform=all`
  reproduces the original corruption identically — and it is measurably expensive.

## Gate

```
flock -x -w 1800 /tmp/mlir-air-npu.lock  agents/scripts/port-loop/gate-h.sh
```

Five legs: build + install, `check-air-mlir`, the transformer-layer suite on hardware, decode
throughput against the recorded floor, then `make verify` over the ten shipped models. Budget ~2 h;
leg 5 is the expensive one and it is where the previous compiler phase's spec errors surfaced.

Run `check-air-mlir` yourself first — seconds — but do not treat it as predictive. Its baseline is
488 passed, 7 UNSUPPORTED, 7 XFAIL, 0 failures.

Then the driver's fixture, now **five** clauses. The four single-column ones must stay green; they
pin that the corrected pass still declines what it should and still transforms what it may. The
fifth is this phase:

| variant | columns | must be exact | must be labeled |
|---|---|---|---|
| `inside` | 1 | yes | no |
| `hoisted` | 1 | yes | no |
| `annotated` | 1 | yes | yes, weight rotated |
| `annotated_hoisted` | 1 | yes | yes, weight NOT rotated |
| **`multicolumn`** | **8** | **yes — 3747+ wrong today** | no |

**Your allowlist is empty.** This phase's subject is the compiler; it is not expected to touch any
`.lit`, Makefile, CMakeLists, registry JSON, verify module, or any `mlir/test/**/*.mlir` beyond
whatever new coverage you add — and `mlir/test` is fingerprinted, so a change there halts the run
unless it is allowlisted. If you need a new lit test, say so in `work_not_completed` and the
operator will widen the allowlist deliberately between phases.

## What lands after this

J1 re-runs: lift `addnorm`'s guard to multi-column, drop `block.py`'s row banding, and re-measure
`coarse` against the `runlist_entries ≤ 10` clause already in place.

**Expect the next wall.** Shim 16-BD exhaustion refused 8 trips even at `herd_x=1`, and J1's target
is 64. Fixing the fusion may make multi-column *correct* without making the target shape
*buildable*. If your work exposes that boundary, record the shape that shows it — that is the next
compiler phase, and the candidate fix is emitting loop-shaped packet BD programs on the shim rather
than one `aiex.dma_configure_task` per iteration.
