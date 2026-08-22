# 16 — Compiler changes on this branch

Consolidated 2026-08-22 from docs 16 (compiler work and the essence), 17 (Phase H), 18 (H1s),
19 (J1), 20 (H9), 21 (J7a), 22 (J7b), 24 (H10), 48 (static legality) and
PREDICTION-FUSED-REEXEC; their full text is at git tag `pre-cleanup-20260821`. This is the record of
every change made to `mlir/` on `exper/transformer-layer-execution-studies` — the defect, the
mechanism, what the fix touched, the lit or gate that proves it, and the measured numbers with their
provenance — plus the two builder-side items (J7a, J7b) whose claim is that the compiler derives
what iron places by hand, and doc 48's host-only legality model. The compiler code stays as-is;
code comments citing "doc 16 … doc 24, doc 48" by number resolve to the headed sections below.

**Gates, as of 2026-08-21.** The port-loop harness (`gate-h.sh`, `phases.sh`,
`guard_gate_files()`, `throughput-baseline.json`, `agents/probes/*`) was retired on 2026-08-21 and
lives only at `pre-cleanup-20260821`. Every "run `agents/probes/...`" or "run `gate-h.sh`"
instruction in the source docs is stale. The live gates are the lits: `check-air-mlir`
(build-xrt/mlir/test, the compiler subset), `ninja -C build-xrt
check-programming-examples-transformer-layer` (the hardware suite), and `make verify` over the
shipped models. The H9 hardware fixture moved from `agents/scripts/port-loop/fixtures/` to
`programming_examples/transformer_layer/addnorm_multitrip.py`, run by
`run_npu2_addnorm_multitrip_peano.lit` (`make check-addnorm-multitrip`), five variants, all exact.

## 0. Index

| § | change | where | commit(s) | proof |
|---|---|---|---|---|
| 1 | `air-fuse-packet-put-loops` — the two-trip `addnorm` miscompile | `AIRDependencyScheduleOpt.cpp`, `CanonicalizeAsyncOpDeps` | `bfb647d9`; warn-on-decline `1b15a1b0` | `fuse_packet_put_loops.mlir`, `fuse_packet_put_loops_decline_warns.mlir`; `run_npu2_addnorm_multitrip_peano.lit` |
| 2 | H2 — classifier sees external kernel calls | `mlir/lib/Util/Util.cpp` `checkOpOperandReadOrWrite` | `7659a503` | `label_ping_pong_external_call_proof.mlir`, `ping_pong_shared_resident_ring*_annotated.mlir` |
| 3 | H3 — `AIRDialect::verifyOperationAttribute` | `AIRDialect.cpp` | `3428238b` | `check-air-mlir` |
| 4 | H1 → H1s — ping-pong safety proof skips, never refuses | `AIRDependencyScheduleOpt.cpp` `provePingPongSafety` | `1514e553`, `cb7be1ab`, `610fadc2`, `5a380615` | `label_ping_pong_loop_invariant_not_rotated.mlir`, `label_ping_pong_alias_escape_proof.mlir` |
| 5 | H9 — packet put loops fused through `scf.parallel` | `AIRFusePacketPutLoops::runOnOperation` | `82715ccc`, `1499d7a9`, `9a7d8c26`, `0842f946` | `multicolumn` variant of the H9 lit |
| 6 | J1 — closed, superseded by J7a | `builders/addnorm.py` guard refined | `52b57c8f`, `ef5e1cf1` | — |
| 7 | H10 — non-constant tile-side BD offset refused | `AIRToAIESchedulingUtils.cpp` `get1DOffset`, `AIRToAIEPass.cpp` | `495e8991`, lit `d57831f4` | `non_constant_bd_offset.mlir` |
| 8 | items 8, 9, wall 6 — three silent-miscompile classes | `air-split-l2-memref`, `air-shrink-memref-sizes-by-access`, `getLockValuePair` | `971bab2a`, `ba3916f8`, `ed9a565d` | `air_split_l2_memref_multi_symbol_offset.mlir`, `shrink_memref_multi_get_band.mlir`, `memtile_lock_count_per_fill.mlir` |
| 9 | item 23 lock placement; item 29 MIMO refusal | `AIRToAIEPass.cpp` `allocateCoreLocksPerMemcpyOp`; `isChainLockCandidate` | `92b05de9`; `a3f3f41e` | `air_channel_to_locks_shared_buffer_producer.mlir`; `memtile_chain_lock_v2_mimo.mlir` |
| 10 | H8 — `air-fuse-pipeline-launches` (first cut, opt-in) | `AIRFusePipelineLaunches.cpp`, `AIRDialect.h` | — | `run_pipeline_fusion_tests.lit`, `mlir/test/Transform/AIRFusePipelineLaunches/` |
| 11 | J7a — norm-tail pipeline, placement and depth derived | `builders/norm_tail.py` | — | `run_npu2_norm_tail_peano.lit` |
| 12 | J7b — accumulator ring formed by `air-hoist-dma-in-accum-pattern` | `builders/ffn_accum.py` | — | `run_npu2_ffn_accum_peano.lit` |
| 13 | fused-decoder re-execution — causal Q-block counter wrapped | `flash_attention/kernel_fusion_based/attn_npu2*.py` | `03402cc1` | `run_npu2_fused_decoder_reexec_peano.lit` |
| 14 | static legality and the mapping-space census | `study/mapping_space.py` | — | `run_mapping_space_tests.lit`, `run_study_host_tests.lit` |

`check-air-mlir` along the way: 486/500 (H, 7 UNSUPPORTED + 7 XFAIL) → 488 (H1s) → 489/489 (H10)
→ 492 → 497 (items 8, 9, wall 6, H8) → 498 (item 23) → 499 (item 29) → 500 (end of 2026-08-12)
→ 505/0 (2026-08-20, README status board).

## 1. `air-fuse-packet-put-loops` — the two-trip `addnorm` miscompile `[2026-08-06]`

**Defect.** `builders/addnorm.py` forbade more than one trip of its row loop because two trips
corrupted the output: at `cols=64, rows=8, rows_per_call=4`, **481 of 512** elements wrong (the
per-trip range 481–497/512 over runs). That rule row-blocked 4096 rows into 64 host dispatches per
normalization point — 128 of `coarse`'s 131 runlist entries, against iron `hybrid`'s 5.

**Mechanism — the shim feed order under packet multiplexing, not ping-pong.**

1. `air-dma-to-channel` hoists each L3-side DMA into its own launch-scope loop, so a herd filling
   N buffers per iteration produces N sibling per-channel put loops.
2. Packet-multiplexed onto one shim MM2S queue (`channel_type = "npu_dma_packet"`), the queue
   serializes in task order — whole channel after whole channel.
3. The consuming tile's BD ring is built from the herd's per-iteration get order and expects the
   streams interleaved per iteration. At one trip the orders coincide; at two or more every packet
   after the first iteration lands in the wrong buffer.

Ping-pong was ruled out by measurement: `--omit-ping-pong-transform=all` reproduces the identical
481/512 corruption. The `air.channel` rewrite that had "confirmed" the ping-pong hypothesis passed
for a different reason — channel form never produces the sibling put-loop grouping.

**Retracted.** Docs 16 and 17 first blamed ping-pong buffering plus a missing dependency edge from
an unclassified `func.call` (the `'u'` classification and the empty `WaitAllOp` placeholder, §2).
Phase H attempt 1 disproved that on hardware, 2026-08-06; the wrong version is not preserved.

**Fix** (`bfb647d9`): a new pass `air-fuse-packet-put-loops` fuses sibling per-channel put loops
that share a block, share static bounds and all target packet-typed channels into one loop issuing
the puts in program order, plus modelling packet-typed channels as one shared stream resource in
`CanonicalizeAsyncOpDeps` so the token chain survives pruning. It sits after the last
`air-isolate-async-dma-loop-nests` deliberately — that pass would re-split a fused loop.

**Proof.** The fixture's `--variant inside` — a legitimate two-trip loop at exactly the shape the
miscompile was measured at — runs on hardware with **zero mismatches** (recorded in
`agents/.state/port-loop/phase-H/implement.report.json` at the tag, not in `gate.log`, which never
got past leg 4). Lit: `fuse_packet_put_loops.mlir`. Hardware: `run_npu2_addnorm_multitrip_peano.lit`
(`inside`, `hoisted`, `annotated`, `annotated_hoisted`, `multicolumn`).

**Silence on decline was wrong, fixed `[2026-08-10]`** (`1b15a1b0`). Unlike ping-pong, declining
here leaves the broken program: the unfused loops feed whole-channel-after-whole-channel against a
ring built for per-iteration interleave. `warnUnfusedGroups` (a post-transform scan in
`AIRDependencyScheduleOpt.cpp`) warns when two or more same-bounds packet put loops remain unfused
and the trip count exceeds one, naming the loop, the channels and the trip count; one-trip declines
and different-bounds pairs are verified silent. Not an error, because the pass cannot establish a
shared queue before `air-to-aie` creates `aie.shim_dma_allocation`.
`fuse_packet_put_loops_decline_warns.mlir`, four cases under `-verify-diagnostics`. Rule and
rationale: [23 §Silence is the wrong default](23-rules-and-open-items.md).

**What this fix does not reach** — one column only. See §5 (H9) and §6 (J1).

## 2. H2 — the classifier sees external kernel calls `[2026-08-06]`

**Gap.** `checkOpOperandReadOrWrite` (`mlir/lib/Util/Util.cpp`) classified a memref use via memory
effects, `ChannelPutOp`, `ChannelGetOp` or linalg and returned `'u'` otherwise, so an external
kernel `func.call` — which registers no memory effects — was invisible to dependency construction.
Unknown uses were silently omitted, and an empty producer or consumer set became
`air::WaitAllOp::create(..., SmallVector<Value>{})`: a dependency-free placeholder rather than a
rejection. This did **not** cause the addnorm corruption (§1), but it is the blocker under every
dataflow analysis over external-kernel programs, including H8 (§10).

**Fix** (`7659a503`, with H1). A callee carrying `llvm.emit_c_interface` classifies its memref
operands from `llvm.readonly` / `llvm.writeonly` argument attributes; an unannotated operand stays
`'u'` and the compiler never guesses a direction. Keep the rule the narrowest that covers
`llvm.emit_c_interface` callees: the classifier is shared well beyond ping-pong, and widening what
counts as a read or write changes dependency graphs everywhere — which is why `check-air-mlir` and
the model leg exist.

**Proof.** `label_ping_pong_external_call_proof.mlir`; the `ping_pong_shared_resident_ring*`
tests kept byte-identical inputs (covering the unannotated path) and gained `_annotated` siblings.
Part of the 522 lines of new compiler test coverage Phase H landed.

## 3. H3 — `AIRDialect::verifyOperationAttribute` `[2026-08-06]`

`grep -rn verifyOperationAttribute mlir/` returned nothing before this: no `air.*` discardable
attribute was validated anywhere. `hasOperationAttrVerify = 1` on the dialect and the hook
implemented (`3428238b`), validating each `air.*` attribute's type **and the op type it may sit
on**, as `GPUDialect::verifyOperationAttribute` does for `gpu.container_module`. Starts with
`air.disable_ping_pong` (restricted to `scf.for` / `scf.parallel`, `AIRDialect.cpp:71`) and
`air.shim_dma_tile_sizes`; H8 later added `air.pipeline_group` / `air.pipeline_stage` /
`air.staging` (§10). Rule: prefer inherent over discardable for anything that must reach the
backend; erase on consume, as the ping-pong labels do; do not attempt blind attribute propagation —
upstream declined an automatic mechanism twice — detect drops instead, as LLVM's
`WarnMissedTransformationsPass` does.

## 4. H1 → H1s — the ping-pong safety proof skips, never refuses `[2026-08-06]`

**Gap.** `air-label-scf-for-to-ping-pong` marks a loop `unroll = 2` and its allocs `hoist_alloc`;
`air-ping-pong-transform` duplicates the buffers and rebuilds the dependency graph. Eligibility
(`isPingPongCandidate`) checked only that no alloc is filled by more than one non-exclusive
`channel.get` per iteration; it did not require a producer for both halves, a recognized consumer,
or that every use be understood, and an empty set became the placeholder of §2.

**The rule, as settled.** When the rotation cannot be proven safe for a buffer it privatizes, the
pass **skips**: leaves the loop single-buffered, emits a *warning* naming the loop (pointing at
`air.disable_ping_pong` and `--omit-ping-pong-transform`), and compilation proceeds. Compilation
aborts only for genuinely malformed IR. The dependency-free `WaitAllOp` placeholder stays
forbidden — skipping means not transforming, not transforming with an empty edge set. Refuse only
what the rotation actually privatizes (a buffer in the `hoist_alloc` set being duplicated); a call
touching any other memref, including one defined outside the loop, is not this transform's
business. Write-only and alloc-only buffers are vacuously safe.

**How it got there** (Phase H, halted at `confirm/3`, 29 of 60 invocations; then H1s, 109 min):

- Attempt 1 specified "hard-fail with a diagnostic", citing upstream `memref::multiBuffer`. That
  reading was wrong: `multiBuffer` returns `failure()` meaning *decline to transform*; IREE
  (`skipOverrideAnalysis=false`), Triton's precondition list and TVM's `ICHECK` all bail out of the
  transformation, not the build. Gate leg 3 failed **8 of 24** (`run_npu2_qkv_proj_peano`,
  `run_npu2_ffn_peano`, `run_npu2_block_peano`, `run_npu2_coarse_peano`, `run_npu2_offload_peano`,
  `run_npu2_runlist_peano`, `run_npu2_fused_peano`, `run_npu2_runlist_gate`) on designs that work.
- Attempt 4: leg 3 24/24; leg 4 (`make verify` × 10) failed **three shipped models**
  (`llama32_1b_int4`, `qwen3_0_6b`, `qwen3_1_7b`) — a buffer filled before the loop and read
  inside it through an external call, refused. Seven passed.
- Attempt 5 (`1514e553`, "refuse only what the rotation actually privatizes") narrowed the verdict
  past everything real. Measured 2026-08-06 on the `hoisted` fixture: it compiles, is exact
  (`XRTRunner: PASS!`), and the labeler never labels the loop.
- H1s (`cb7be1ab` the verdict, `610fadc2` the two existing tests, `5a380615` the new test): `Refuse`
  → `Skip` in `provePingPongSafety`; the driver's `anyRefusal` pre-scan and `signalPassFailure()`
  removed, enumerator deleted. **H1s removed a latent hazard, not a live one**: `make verify` on
  `llama32_1b_int4` against the pre-change compiler passed with fresh `aircc` compiles, so leg 5's
  10/10 held at the phase base too.

**The labeling decisions, measured 2026-08-06** from aircc's `--debug-ir` dump, all at
`cols=64, rows=8, rows_per_call=4`:

| callee | weight DMA | compiles | labeled (`unroll`) | `hoist_alloc` set |
|---|---|---|---|---|
| unannotated | in loop | yes | **no** | — |
| unannotated | hoisted | yes | **no** | — |
| annotated | in loop | yes | **yes** | 4 — the three tiles **and** the weight |
| annotated | hoisted | yes | **yes** | 3 tiles; the weight is **excluded** |

So the `hoisted` shape was never a hazard (the rotation already excludes a buffer filled before the
loop — why the three models are correct); an unannotated external call is never guessed at (the
two-trip `inside` loop is correct because of §1, not ping-pong — it is never ping-ponged); and the
transform still fires where provable, per buffer. Those four rows are the fixture's four
single-column variants (`must compile` all yes; `annotated` labeled with weight rotated;
`annotated_hoisted` labeled with weight NOT rotated), all of which already passed on the pre-change
build — the check pins the per-buffer labeling decision against side effects of simplifying the
predicate, it is not a test the phase turned green.

**H1s outcome**, three Codex rounds `verdict=pass blocking=0 weakened=0`:

| leg | result |
|---|---|
| 1 build + install | pass |
| 2 `check-air-mlir` | 488 passed, 7 UNSUPPORTED, 7 XFAIL, 0 failures |
| 3 transformer-layer suite on hardware | pass |
| 4 decode throughput | `llama32_1b` **11.01 tok/s** against a **9.43** floor — pass; `llama32_1b_int4` `NOT GATED` |
| 5 ten shipped models | 10/10 |

**Tests.** `label_ping_pong_alias_escape_proof.mlir:80` and
`label_ping_pong_external_call_proof.mlir:522, :606` changed from `expected-error` to the warning
(inputs byte-identical); `label_ping_pong_loop_invariant_not_rotated.mlir` asserts over an
annotated callee with a buffer filled before the loop that the loop **is** labeled and the
loop-invariant buffer is **not** in `hoist_alloc` (`CHECK-NOT` on `hoist_alloc` against that
buffer's type) — the clause that keeps the suite discriminating against a pass narrowed until it
never fires. `label_ping_pong_loops.mlir` and `label_ping_pong_multifill_alloc.mlir` were reverted
to their phase-base inputs after attempt 2 edited them to dodge the refusal.

**The rule for existing tests** (three weakened-gate halts relearned it): when a change alters the
outcome for an existing test's input, keep the input byte-identical and update only the CHECK lines
to the new intended outcome; if the transformed path needs coverage, add a **new** case. Annotating
an input to preserve the old outcome deletes the evidence that behaviour changed.

**H4 struck `[2026-08-06]`** — `air.disable_ping_pong` measured working: set on a loop that IS
otherwise labeled, `unroll` and `hoist_alloc` both go to zero, the attribute present in every
`--debug-ir` dump through the labeling pass (the four hand-patched propagation sites of PR #1664 do
their job). The earlier "setting it changed nothing" (byte-identical 481/512 both arms) was taken on
a shape whose callee is unannotated and therefore never labeled — the same confound that made
`--omit-ping-pong-transform=all` look exculpatory. Do not promote it to an inherent attribute.
Covered by `label_ping_pong_disable_opt_out.mlir`, `hoist_preserves_disable_ping_pong_attr.mlir`.

**Throughput is the failure mode correctness gates cannot see.** Dropping ping-pong regressed a
shipped model **12.4 → 7.8 tok/s** (`llms/shared/infra/backend_presets.py`); the 9.43 floor existed
for that. Do not disable ping-pong globally or narrow the predicate until it stops firing.

## 5. H9 — fuse packet put loops through `scf.parallel` `[2026-08-07]`

**Defect.** `AIRFusePacketPutLoops::runOnOperation` (`AIRDependencyScheduleOpt.cpp:4840` at the
time) walked only `air.launch`'s immediate body blocks. At `herd_x ≥ 2`, `air-dma-to-channel`
wraps each per-tile put loop in an `scf.parallel`, so the pass's output IR was byte-identical to
its input (`--debug-ir` pass 026 vs 027) and the §1 corruption returned from the second trip on.
Silent every time. Every clause of the Phase H fixture had run at `herd_x=1`, the width the original
miscompile was measured at.

**Measured on NPU2**, two trips unless stated:

| shape | result |
|---|---|
| `herd_x=1`, cols 64 and 768 | exact |
| **`herd_x=8`, cols 64** | **4070 / 4096 wrong** |
| `herd_x=8`, cols 64, weight DMA hoisted | 4039 / 4096 wrong |
| **`herd_x=8`, cols 768** | **97,726 / 98,304 wrong** |
| J1's target — 64 trips, 4096×768, `herd_x=8` | compiles, **3,130,958 / 3,145,728** wrong |

The full walk, including the L2-staged-weight arms, is in
`programming_examples/transformer_layer/README.md` §"Phase J1 findings".

**Fix** (`82715ccc`, reviews `1499d7a9`, `9a7d8c26`, `0842f946`). The first framing ("choose which
blocks to walk") was measured false: `air-dma-to-channel` emits **one `scf.parallel` wrapper per
hoisted put loop**, so every nested block holds a single loop and the groups that matter *span* the
wrappers. What landed: **sequentialize** each eligible wrapper into per-iteration clones in
ascending order — the order `airrt-to-npu` unrolls launch-scope parallels anyway — then run the
existing single-block grouping. Eligibility, narrowed by three review rounds each finding a real
defect: the `scf.reduce` combiner must be exactly the wait-all join `air-dma-to-channel` emits,
verified whether or not the result has users (a combiner with memory effects such as
`memref.atomic_rmw` declines the wrapper); live result tokens are expanded rather than declined,
because declining would leave the miscompile in place for any launch whose per-channel parallels
feed a later async op. Upstream placement (before the per-tile specialization) was rejected with a
recorded reason: the put loops reach their final shape only after the last
`air-isolate-async-dma-loop-nests`.

**Proof.** The fixture's fifth variant, `multicolumn` (`herd_x=8`, two trips per column), went
from **3747+ / 4096 wrong** (verified failing before the fix) to **exact**; the four single-column
variants stayed green. Gate 184 min: all five legs, 10/10 models. Today the fixture is
`programming_examples/transformer_layer/addnorm_multitrip.py` under
`run_npu2_addnorm_multitrip_peano.lit`, its clauses exact rather than a tolerance.

**The wall it exposed** — shim BD exhaustion: at `herd_x=8` column 0 carries weight + x + residual,
three packet tasks per trip, and each put in the fused loop lowers to its own active
`aiex.dma_configure_task` against a shim tile's 16 BDs. **Six trips refuse** (6 × 3 = 18 > 16);
five compile; at `herd_x=1`, 8 trips refuse. The candidate fix — loop-shaped packet BD programs on
the shim rather than one `dma_configure_task` per iteration — is unclaimed.

## 6. J1 — collapse the norm dispatches: CLOSED, superseded by J7a `[2026-08-21]`

J1 set out to lift `builders/addnorm.py`'s guard and collapse the 64 dispatches per normalization
point into one launch (131 → ~5 runlist entries). The arithmetic at `cols=768, herd_x=8`, pre-add:
`addnorm_max_rows(768, herd_x=8, pre_add=True)` = 104 rows per launch; L1 62,464 of 65,536 bytes;
`block.py` bands at 64 rows → 64 dispatches; lifted, one launch with `rows_per_tile = 512`, and
because `rows_per_call` must divide 512 under the L1 cap of 13, the largest legal value is 8 —
**64 trips per tile**, a 32× extrapolation from the two trips anyone had measured.

**What it found** (`[2026-08-07]`, stopped by the operator at `fix/1`): wall 1, the §5 multi-column
miscompile (4070/4096 at `herd_x=8`, 2 trips; 3,130,958/3,145,728 at the 64-trip target) — fixed by
H9; wall 2, shim BD exhaustion at six trips against a 64-trip target. J1 moved from
*compiles-silently-wrong* to *refuses-loudly*. The guard was refined to the measured boundary
(multi-trip only at `herd_x=1`; `52b57c8f`, `ef5e1cf1`) rather than lifted, and `coarse` stays at
131 entries. The `air.channel` workaround is not a route: correct at 2 trips but 64 bands × 4
channels exceeds a tile's 16 locks (`'aie.lock' op lock assigned invalid id (maximum is 15)` in
`air-to-aie`) — that is H5 (§15).

**Closed `[2026-08-21]`, per the operator**: the same dispatch collapse is reached by J7a (§11),
which packs x and residual into one strided fetch and never enters the packet path. J1's
distinguishability work (J4) is recorded in §16.

## 7. H10 — a non-constant tile-side BD offset is refused, not frozen `[2026-08-07/08]`

**Defect.** `air-to-aie` lowers a channel put's offset into a static `aie.dma_bd` offset. Two
unchecked dereferences in `mlir/lib/Conversion/AIRToAIESchedulingUtils.cpp` — `get1DOffset`
(`*offset`, `*stride_i`, lines ~199–215) and the `AIE::BDDimLayoutAttr` build (`*wrap`,
`*stepsize`, ~460) — dereferenced the `std::nullopt` that `mlir::getConstantIntValue` returns for a
non-constant value; observed as a silent `0`. The same file checks correctly at lines 527 and 945,
and the only caller of `get1DOffset` (`AIRToAIEPass.cpp:6527`) checks the line above it.

**Evidence** (J7b's pre-fix builder at 4 K steps): the dump before `air-to-aie` carries
`air.channel.put ... (%arg4[%7] ...)` with `%7 = affine.apply #map()[%arg6]` on the loop IV; the
memtile MM2S chain at `herd_x=1`, block 2048, reads `[0, 0, 2048, 0]` at `ksteps=2` (fully
unrolled, literal offsets) and `[0, 0, 0, 0]` at `ksteps=4`; on hardware the consumer stalls, the
output DMA never fires and the output buffer comes back byte-identical to what the host wrote
(seed 1.0 → 4096/4096 elements 1.0); at 4 columns × 96 steps, `ERT_CMD_STATE_TIMEOUT`. Not
ping-pong (`omit_pingpong="all"` identical). Every structural check stayed green (4 → 2 hoist, zero
packet-typed channels, full compile). Why the J7b builder's rule holds — *advance on the L3 side,
never on the L2 read*: L3 transfers are programmed by the runtime sequence (`AIRRtToNpuPass`),
which materializes offsets per task; an IV-dependent offset is inexpressible on an L2/L1 operand.
In J7b's failing module the W refill (`%arg4`, L3, IV-dependent) was fine, the A feed (`%results`,
`memref<8192xbf16, 1 : i32>`, IV-dependent) silently wrong, the B feed (whole buffer) fine.
[23 §Never read a staged buffer at a per-iteration offset](23-rules-and-open-items.md).

**Why refuse, not skip.** H1s's skip was right because declining leaves a correct single-buffered
loop. Here there is no correct fallback: a BD cannot express a per-iteration offset, so the only
honest outcomes are a refusal or H5's dynamic-index lowering (mlir-aie's `dynamic-objFifos`: a
per-core counter plus `scf.index_switch`). This phase refuses.

**Fix** (`495e8991`; lit `d57831f4`). `get1DOffset` returns `std::optional<int64_t>`, `nullopt` when
any offset or stride it needs is non-constant; the caller emits a diagnostic naming the channel,
operand and loop and saying what to do ("stage the operand per iteration from L3, or see H5"); the
BD dim layout site gets the same treatment. **Scope: both tile-side allocators —
`TileDMAAllocator` and `MemTileDMAAllocator` — exempting only `ShimDMAAllocator`.**
`generateDmaBdProgram` is instantiated three times (`AIRToAIEPass.cpp` 7512 core, 7616 shim, 7664
memtile) and all reach `generateDmaBd` → `get1DOffset`; an earlier revision said "scope to
`TileDMAAllocator`" from a truncated grep, which would have exempted the memtile path J7b's frozen
chain was on (caught by review round 3).

**The compiler's own tests relied on the freeze.** Re-measured (round 3) on the pre-phase compiler
(`bb017619`) with every `aie.memtile_dma` BD attributed to its L2 buffer:

| test / buffer | feed's offsets | measured MM2S BDs | verdict |
|---|---|---|---|
| `async_gemm_to_locks_aie2` C | both herd-derived, no IV | {0, 32, 2048, 2080} | correct |
| … A | column is the k-loop IV | **{0, 0, 2048, 2048}** | frozen at k=0 |
| … B | row is the k-loop IV | **{0, 32, 0, 32}** | frozen at k=0 |
| `async_gemm_w_pingpong_to_locks_aie2` A ping/pong `64x128xi32` | correct {0,32,64,96} ∪ {4096,4128,4160,4192} | **{0, 4096}** | frozen |
| … B ping/pong `128x64xi32` | correct {0,2048,4096,6144} ∪ {32,2080,4128,6176} | **{0, 32}** | frozen |
| … C `64x64xi32` | — | {0, 32, 2048, 2080} | correct |

(`async_gemm_w_pingpong_to_locks_npu.mlir` carries the same segment code.) Their CHECK lines pinned
no BD offset, so they were structurally green over data movement re-reading block 0 forever. The
§DECIDED revision that concluded the opposite (retracted the same day) had credited the C feed's
offsets to the A feed's put. Mechanism: in the cloned device region the IV operand is already
`arith.constant 0` (`cloneL2AndL3MemcpysToDeviceOp`'s zero-substitution), and nothing in those RUN
lines unrolls the loop — `AIRUnrollScfForIntoBDChain` lives in
`applyAIRSpecializeChannelWrapAndStridePattern`, reached only via
`-air-specialize-channel-wrap-and-stride`, `-air-opt-shim-dma-bds`, `-air-opt-memtile-dma-bds`,
none of which is in aircc's pipeline either. The three tests were given `-air-opt-memtile-dma-bds`
in their RUN lines (`93a0f5ce`, "Give three AIRToAIE tests the pass their construction needs"; their
headers say the pass is REQUIRED because the feed loops carry the k-loop IV in an L2 put's offset,
which only that pass folds into BD wrap/stride); the red state was the designed halt,
not a regression, and neither unrolling nor IV-walk specialization was added to `air-to-aie` (H5's
territory).

**Gate** (`[2026-08-08]`): `H GATE: PASS`, all five legs — `check-air-mlir` 489/489, hardware suite,
decode **11.44 tok/s** vs the 9.43 floor (above the 11.10 baseline; an earlier 10.66 was
contention), 10/10 models — and the objective check with its four clauses: an IV-dependent L2 offset
refused by message; the same builder at 2 trips still compiling; a constant offset compiling; an
L3-side moving offset compiling. The first clause's probe had to be J7b's real construction
(`e6cdd138`'s `ffn_accum` at 4 K steps, A-feed slices interleaved across four cores): a synthetic 1-D
probe whose four slices tile the buffer is coalesced by `air-opt-memtile-dma-bds` into one
whole-buffer BD (`offset = 0 len = 8192`) and correctly not refused — and that probe had "failed
before the fix" only because nothing was refused then. **Check that a failing clause fails for the
reason intended.** The tamper check then halted on five gate files changed in the phase window
(`phases.sh` operator repair, the three `async_gemm*` reworks, the GEMM registry JSON from the
`baseline_512`/`baseline_1024` sweeps) — all legitimate; the fix is sound, the phase's baseline
(`bb017619`) is not clean, and it was not re-fingerprinted. Lit: `non_constant_bd_offset.mlir`. A
static survey found no other `ChannelPut`/`ChannelGet` with a non-literal offset on an L2/L1 operand
in `programming_examples/` (`attention_decode/attn_decode_npu2.py:347` walks an L3 buffer;
`herd_dataflow/run.py`'s `affine.apply` is inside an `scf.forall`, specialized per index).

## 8. Three silent-miscompile classes fixed — items 8, 9, wall 6 `[2026-08-12]`

All three were checks or computations that produced a *green* wrong answer. Each has a regression
lit **verified failing** pre-fix; `check-air-mlir` **492 → 497 / 0** across the three plus H8; the
transformer-layer suite 31/1/0 and the ten models 10/10 unchanged — none moves a shipped design.
Full rows: README queue items 8, 9, 18; [31](31-resident-tail-r1-record.md) for the R1 context.

**Item 8 — `air-split-l2-memref` sized a replacement affine map from one symbol** (`971bab2a`).
The pass lifted the expression out of the offset's existing `affine.apply` (N symbols) and rebuilt
the replacement as `AffineMap::get(0, 1, add)`, so a two-level nest over an L3 operand tripped
`willBeValidAffineMap` and SIGABRTed (exit 134, 5/5) — the two-symbol maps [23 §The rule](23-rules-and-open-items.md)'s
L3-side rule produces routinely. All three sites now delegate to one `composeAffineMap` helper;
instrumentation showed the reproducer hits the third (`composeAffineExprFromSizes`), and the corpus
exercises all three (18 / 6 / 128). Two more hardcodes of the same class found while fixing:
`replace(...)` calls with literal `(0,1)`/`(1,0)` result counts, and the `air.execute`-wrapped path
binding operands via `getUsedValuesDefinedAbove` with no ordering guarantee — harmless at one
symbol, **wrong addresses at two**. Verified per case against the preserved unpatched binary (134
each, 0 patched); 493/0 against a 492 baseline; the suite with only `air-opt` swapped for the
unfixed binary gives 480/13, exactly one status change. Lit
`air_split_l2_memref_multi_symbol_offset.mlir`.

**Item 9 — `air-shrink-memref-sizes-by-access` measured extent from sizes and strides alone**
(`ba3916f8`). Offsets were used only to test emptiness, so N gets of equal volume at different
offsets measured as one: `memref<12288xbf16, 2>` shrank to `<3072>` with the gets left at
3072/6144/9216, reading past the end at `EXIT=0` with nothing on stderr. Fix: a real extent
(`offset + size` per dimension), so a band whose gets reach 6144 shrinks to 6144; a diagnostic
only for offsets that cannot be classified; a latent OOB read on the empty offsets vector fixed
alongside. Lit `shrink_memref_multi_get_band.mlir` (4 cases), failing pre-fix at three CHECK lines
and under `-verify-diagnostics`; 481/12 vs unpatched 480/12 in the isolated tree (the 12 are an
absent `aircc`; 480 + 12 = 492). Consequences recorded: five allocs in `loop_fusion.mlir`'s output
stop shrinking (same defect on channel ops, pinned by no CHECK; zero offset-bearing `ChannelGet` in
`programming_examples/`), and `probe_r2_order_seam.py --arm row_band` fails by design.

**Wall 6 / item 18 — `getLockValuePair` sized a memtile semaphore from static user ops**
(`ed9a565d`; `AIRToAIESchedulingUtils.cpp:510-556`, the `ceil(read_counter/write_counter)` at
`:550`). R1's `l2_b_down` presents 1 writer / 16 reader ops (4 sub-channels × 4-way unroll) →
`ceil(16/1) = 16`, **while the same pass emits 4 MM2S BDs for those 16 ops**. Three siblings with
the identical four-consumer shape get 4; the pre-E1 builder yields 16 too (refuting "E1 caused it").
Fix collapses reader ops sharing channel symbol + constant indices + constant access region,
scoped to `write_counter == 1` (multi-writer buffers time-multiplex; `l2_h` would go 4 → 1 and
starve three consumers). Lit `memtile_lock_count_per_fill.mlir` with a multi-writer control
identical under both binaries; 495/0 vs 494/0; `l2_b_down` 4/4, whole-module lock audit conserving
on both memtiles and all eight core tiles (devq 259).

## 9. `air-to-aie` lock placement (item 23) and the MIMO refusal (item 29) `[2026-08-12]`

Both found inside the R1 resident-tail work; the R1 record is
[31](31-resident-tail-r1-record.md). Recorded here because they are compiler changes.

**Item 23 — core lock placement** (`92b05de9`, `AIRToAIEPass.cpp` `allocateCoreLocksPerMemcpyOp`).
In `sharedStagingBuffer` mode the acquire was placed at each put, pacing put *i+1* against put *i*
but leaving the core's **own** writes unguarded, so a core that produces a buffer once and ships it
in N>1 slices has its next round's memset overtake the last chunk's in-flight BD. Lock *counts* are
conserved, which is why wall 6's audit could not see it. Reproduced byte-deterministically (devq
278: 20 dispatches, 6/6 rungs at exactly 1 distinct `y` sha256); minimal failing config
`emb 64 / ffn 64, herd_x 1`, trigger `chunks_per_group > 1` (`emb 32 / ffn 128`, 4 sweeps, is
correct); a rate model fit on two rungs predicted **0.810** survival at the unmeasured
`emb 96 / ffn 96`, devq 294 measured **0.81**. Fix: hoist the acquire to the earliest op touching
the buffer since the previous DMA on it — the original placement for a pure relay
(`air_channel_to_locks_shared_buffer.mlir`, #1515, unaffected). Lit
`air_channel_to_locks_shared_buffer_producer.mlir` verified failing pre-fix; `check-air-mlir`
497 → 498/0; suite 32/1/0; models 10/10 (devq 305); hardware ladder **21/21** (devq 300), the rungs
that already passed byte-identical and exactly the four broken ones moving. Two null runs (devq
298, 299) reported the pre-fix answer from a green log because `air.tools.resolve_tool` prefers the
bundled `install-xrt/bin/aircc` over PATH — `build-xrt/python` **plus** `build-xrt/bin` are both
required, the runner now refuses unless the resolved `aircc` sha matches the built one, and an
ELF-hash diff is not a valid discriminator (the ELF is not byte-reproducible).

**Item 29 — wall 7's compiler fix does not exist; v2 refuses MIMO by name** (`a3f3f41e`). The
recommended move (extend `isChainLockCandidate` to MIMO, two chains in `getOrCreateChainLockSet`)
was built and measured: writers ORDERED, readers **OVERWRITE**. An AIE2 BD carries one acquire and
one release field, so a writer's release either orders writers or binds readers, never both; for 2
writers / 2 readers / 1 slot the constraints force `a₀ > 2ρ` and `a₀ < 2ρ`. Any correct scheme
needs `P = herd_x × chunks_per_group` reader phases — 24 at the gate → 48 blocks per MM2S channel
against a 48-block cap — so compiler and builder fixes hit the same wall ("the compiler fix scales"
retracted). Ships: under `use-lock-race-condition-fix-v2` a MIMO memtile buffer **errors** naming
the buffer, counts and reason (the silent fall-through is why v2 A/B'd byte-identical five times —
never reached), plus `--check-order`, a Petri-net check deciding writer-order and read-binding
separately. Default path untouched: R1 D2's `aie.air.mlir` sha256 `5439c51d…` from both compilers.
`check-air-mlir` 498 → 499/0, delta exactly `memtile_chain_lock_v2_mimo.mlir`, verified failing
first. The rest of doc 52 (the row 28(a) order-preserving BD program `9ed6f267`, the unbuilt pacing
step, `--check-order`'s two fixes) is in [31 (52 §§8–13)](31-resident-tail-r1-record.md).

## 10. H8 — `air-fuse-pipeline-launches`, first cut `[2026-08-12]`

Doc 16 sized H8 (automatic on-chip staging between pipeline stages: a memref written by one
hierarchy op, read by one, no host aliasing, dead afterwards → a channel with demoted memory
space, plus launch fusion) as *large, and needing H2*. H2 had landed and J7a/J7b supplied the
hand-written reference dataflow. Scoped **declarative rather than derived**: builders emit
`air.pipeline_group` / `air.pipeline_stage` / `air.staging` (each verified for type and host op
via §3), and `air-fuse-pipeline-launches` (`mlir/lib/Transform/AIRFusePipelineLaunches.cpp`)
splices attributed launches into one segment — opt-in via `air-opt`, deliberately **not** in
aircc's pipeline, so no shipped model's compile moves. Gate: **byte-identical** reproduction of the
hand-written `norm_tail.py` at all four buildable shape/variant combinations (chosen over a
structural list because [23 item 5](23-rules-and-open-items.md) records a structural check missing
its phase's claim); the two arrangements share stage bodies, the default IR is byte-identical to the
pre-change builder's, perturbing the stage sort fails all four clauses; 9 negative cases + 6 positive
controls, refusing rather than skipping. `air.staging` is checked (`AIRDialect.h:140`): the only
thing in the toolchain that can catch a lost accumulator ring (§12), since both losing constructions
compile and return correct numbers. `check-air-mlir` 497/0, host suite 357/357;
`run_pipeline_fusion_tests.lit`, `make check-pipeline-fusion`. **It cannot express R1**: a
segment-scope `scf.for` that both gets stage 1's output and puts stage 2's input, plus an alloc
shared across two stages' feed nests — neither survives `IsolatedFromAbove` per-stage segments.
`air-fuse-parallel-launch` is unrelated (it is about `scf.parallel` around a herd,
`AIRMiscPasses.cpp:900`); `air-override-memref-memory-space` is experimental and rewrites only
allocs inside a region — launch arguments such as `qkv_f32` in `block.py:550`'s `host_writes` are
host-visible ABI the compiler may not touch.

## 11. J7a — the norm-tail pipeline `[2026-08-07]`

Builder work (`builders/norm_tail.py`, 87 min), recorded here because its claim is about the
compiler: three herds in one segment joined by L1→L1 channels — `stage_add` (x|residual packed, one
strided L3 fetch) → `AtoB` → `stage_norm` → `BtoC` → `stage_scale` (+ gamma, one L3 fetch) → L3 —
with **no placement and no buffer depth declared**; `air-place-herds` places (24 of NPU2's 32
tiles, three herd rows of 8) and ping-pong labeling chooses depth. Modelled on
`programming_examples/bottleneck/` and `channel_examples/worker_to_self/`.

**The constraint that decided it**: a column has two shim MM2S channels and the budget is per column
across the whole segment, stacked herds adding. Measured at 4096×768, `herd_x=8`, 64 trips per tile:
x, residual, gamma each streamed → **3 packet-typed channels** (the §1/§5 path); x|residual packed,
gamma second → **0**. Doc 21 SPECIFIED packing as planes `[2, rows, cols]` (3-D DMA,
`strides=[rows*cols, cols, 1]`) and a callee declaring strided operand types; **the builder
measured otherwise** (`builders/norm_tail.py`): plane-major's band fetch has plane stride
`rows*cols`, and a shim `aie.dma_bd` stride is capped at 2^20 — `'aie.dma_bd' op Stride 2 exceeds
the [1:1048576] range` (strides `[3072, 3145728, 768, 1]` at 4096×768), through full aiecc where
the spec's probe had stopped at `air-opt`. The shipped layout is row-interleaved `[rows, 2, cols]`
(a `rows_per_call` band is contiguous in L3 at any row count); plane-major compiles at 128 rows
(stride 98,304) and was measured numerically exact there, so `[2026-08-08]` it is back as the
`plane_major` opt-in for a device-written plane 0, bounded by `rows*cols <= 2^20` → `rows <= 1365`
at `cols=768` (the 64…1024 rungs, not 2048 and up). Doc 21's "L1 offset honoured" claim was
measured with deliberately asymmetric x and residual at the 128-row shape. It is
also why J1's L2-staged weight failed: 8 × 2 = 16 shim MM2S already full before a third stream.
J7a is unaffected by §5's BD wall because BD exhaustion counts *packet* tasks and the packed form
has none; the existing streamed builders already run 64 trips at `herd_x=8` on that path.

**Result.** `mean_rel_L1` **3.620e-3** at 4096×768 (3.590e-3 at 128×768) against the block's
1.688e-2 bound and `fused`'s decomposed tail at 1.806e-2 — 4.7× under the bound; `layer_norm`
itself improved ~25× as a side effect. Gate `run_npu2_norm_tail_peano.lit`: `check-norm-tail-structure`
(through `XRTBackend(debug_ir=True)` on the routed design: 3 herd rows × 8, **16** core-tile→core-tile
`aie.flow`, max 2 shim inbound per column, 0 packet-typed channels — verified failing against a
4-wide placement; the original clause, counting `"npu_dma_packet"` at `air-dma-to-channel`, could
not see an edge that round-tripped through L3), `check-norm-tail` (np.isclose at rtol 1.6e-2, zero
mismatches, 128×768 / 4096×768 / 128×768_offset), `check-norm-tail-fault` (input 0 at (127, 0, 0)).
The standing rule: **two or fewer L3-facing streams per column; everything else on L1→L1 channels;
pack co-indexed L3 operands into one fetch** ([23](23-rules-and-open-items.md)). Do not hand-place
a herd or hand-set a depth; do not widen the `1e-1` layer tolerance.

## 12. J7b — the accumulator ring, formed by `air-hoist-dma-in-accum-pattern` `[2026-08-07]`

**What AIR automates.** `air-hoist-dma-in-accum-pattern` runs unconditionally, second in aircc's
pipeline (`tools/aircc/aircc.cpp:837`), matching an incoming and an outgoing DMA on the same memref
with mirrored offsets/sizes/strides (`AIRDependencyScheduleOpt.cpp:322`), both loop-invariant, and
hoisting both — iron's `of_o_acc_in`/`of_o_acc_out` ring, derived. It is purely syntactic on the
DMA ops, so an opaque `func.call` does not block it. **Under two conditions doc 16 first omitted**,
measured with `air-opt --air-dependency,--air-hoist-dma-in-accum-pattern` and re-checked at aircc
altitude (`XRTBackend(debug_ir=True)`, `pass_006_after_air-hoist-dma-in-accum-pattern.mlir`, same
4 → 2), counting data-movement ops left in the K loop:

| accumulator kernel | L1 buffers allocated | before → after | ring? |
|---|---|---|---|
| in-place — one `C`, read-add-write | outside the loop (herd scope) | 4 → 4 | no |
| in-place — one `C`, read-add-write | **inside the loop** | **4 → 2** | **yes** |
| two-buffer — `pAcc` in, `C` out | outside the loop | 4 → 4 | no |
| two-buffer — `pAcc` in, `C` out | inside the loop | 4 → 4 | no |

Condition 1: the in-place kernel — `areSymmetricDmaOps` requires the same memref;
`ffn_matmul_bf16_bf16_up_proj(A, B, C)` matches, `ffn_matmul_with_acc_bf16_bf16_down_proj(A, B,
pAcc, C)` (two `__restrict` memrefs, the one doc 16 first named) never does (`encoder_matmul.cc:26`
flags it). Condition 2: allocate the accumulator **inside** the loop — `isIncomingDmaOp` needs the
DMA to depend on the first iter-arg and an `air.execute` holding `memref.alloc`, `isOutgoingDmaOp`
needs an `air.wait_all` and a `memref.dealloc` user; the pass hoists alloc and DMA pair together.
Match a probe's altitude to its claim ([23](23-rules-and-open-items.md)): a J7a workaround that
lowered cleanly through `air-opt` never compiled under `aircc`, because `air-to-aie` normalizes
callee signatures afterwards.

**Built** (`builders/ffn_accum.py`, `build_ffn_accum_module`, FFN down-projection only; 58 min):
zero C once (`ffn_zero_bf16_*`), then per K step fetch C, call, store C, and let the pass collapse
it. `check-ffn-accum` 0 mismatches / 49152, `mean_rel_L1` 1.417e-2, `atol_required` 1.383e-3 against
atol 5e-3; `check-ffn-accum-structure` K loop **4 → 2**, zero packet-typed channels;
`check-ffn-accum-fault` input 0 at (0,). Gate `run_npu2_ffn_accum_peano.lit`. **Clause 3 is the
phase**: the numbers are identical whether the ring formed or not. Three walls decided the shape: the
shim column budget, a core S2MM ceiling, and the per-iteration L2 read offset the compiler silently
froze (§7), which cost the implement session — A and B ride one memtile feed channel per core and
the C pair rides the shim. Do not hand-build the ring; do not declare a memory space for the
accumulator; the o-projection is the same construction and unbuilt.

**Where the split falls** (doc 16's survey): the accumulator half is automatic; the inter-stage half
is not and cannot be today — memory space is an input to the pipeline, never an output, so
`fused-cast`'s full-size f32 scratch in L3 (`qkv_f32`, `ffn_up_f32`, `ffn_out_f32`) is declared by
the builders, not missed by the compiler. What is less manual than iron either way:

| decision | iron | AIR |
|---|---|---|
| tile placement | `placement=Tile(col,row)` | `air-place-herds` |
| buffer depth | `depth=2` | `air-label-scf-for-to-ping-pong` + `air-ping-pong-transform` |
| memtile assignment and sharding | by hand | `air-split-l2-memref`, capped by the real shim budget (`aircc.cpp:869`) |
| DMA BDs, wrap-and-stride | by hand | `air-opt-shim-dma-bds`, `air-opt-memtile-dma-bds` |
| broadcast fan-out | `of.cons(n)` | `air-broadcast-detection` + `air-specialize-dma-broadcast` |
| sharing one physical channel across flows | no equivalent | `air-fuse-channels{aggressive-mode=L1,L2,L3}` |

The three accumulate-into-C kernels (`matmul_with_acc_vectorized_2x2_mmul`, `_1x4_mmul`,
`matmul_with_acc_bf16_bf16_down_proj`) had never been dispatched before J7b.

## 13. The fused decoder's re-execution wall — prediction and outcome `[2026-08-19/20]`

**Prediction** (PREDICTION-FUSED-REEXEC, written before dispatch, 2026-08-19). The wall — dispatch 1
of the fused decoder stitch 12/12 clean, dispatches 2..n UNMASKED attention with clean q/k/v (devq
382–384) — is the causal mha's Q-block counter. `attn_npu2.py` keeps per-core causal state in an
**uninitialized** L1 buffer `causal_ctr` ([0] q_block base, [1] boot flag, [2] head_local, [3]
dv_iter); the boot flag fires only on zeroed memory, head_local and dv_iter wrap, q_block only
advances (+NQ per head-group wrap). A complete execution ends with `q_base = num_lq_iters × NQ`,
past every kv block; L1 persists across dispatches, so dispatch 2 loads boot=1 and
`kv_block_idx > q_block_idx` is never true — `apply_causal_mask` never fills. This explains the
controls: `coarse` re-executes causally because its flow re-initializes the partition per dispatch;
evicting/reloading the fused hw_context did not heal it (devq 384) because the reload rewrites only
CDO-initialized state. Fix under test: `_emit_counter_increment` wraps the advance,
`q_wrapped = remsi(q_cur + NQ, num_lq_iters × NQ)` — identity within one execution, boot state
restored at its end. Four predictions (gpt2_small 512×768, 3 dispatches): (1) the causal module's
MLIR gains exactly the remsi wrap (constant 8 = 2 × 4), the encoder module byte-identical (cache
hit); (2) dispatch 1 12/12 under `DECODER_STAGE_ATOL`; (3) dispatches 2 and 3 `attn_context` corr
vs the causal reference ~1 (baseline 0.9994 vs the unmasked reference), all 12 clean; (4) dispatch 2
== dispatch 3 bytes. Falsifiers: d2/d3 still unmasked → not the only state; d1 regressing → revert;
d2 clean but d3 not → bound off by one execution.

**Outcome** (README status board, `03402cc1`): **all four clauses met** — devq 413, three dispatches
12/12 each, corr **0.9995 causal / 0.438 unmasked** on every one, d2 == d3 bytes. The
`14209f71` framing of the wall as composition-level device state in the H10 frozen-BD family is
**retracted**. Gate `run_npu2_fused_decoder_reexec_peano.lit` (`make check-fused-decoder-reexec`,
`study/fused_reexec_gate.py`): two dispatches of one prepared stitch, both 12/12 plus a
causal-correlation clause on the second; its falsifier ran (devq 426) — HEAD's builder restored
byte-exact under the checked-in gate → dispatch 1 clean, dispatch 2 `FAIL` with **67,687**
`attn_context` mismatches and corr **0.4382**, exit 2. Legs: decoder smoke walk 4/4 with `fused`
measured for the first time (devq 414); suite **35/1/0** (devq 422, the +1 is this lit);
`check-air-mlir` 505/0 (devq 412); models 8/11 = the standing leg (devq 416 + 425; the three ≥3B
models deferred, [15](15-environment-notes.md)'s oomd note). `attn_npu1.py` carries the same
counter and is patched for parity, **unverified** (no NPU1 here). The boot contract (zeroed L1 at
partition init) is a runtime-stack property, measured not source-guaranteed, documented at both
alloc sites; plumbing a real `initial_value` through air-to-aie is the named follow-up. Artifacts
`results/decoder-gpt2-first-walk/` (jobs 412–414, 416, 422, 425, 426). The decoder families' numbers
are in [54](54-first-full-profile-and-decoder-families.md).

## 14. Static legality, and how big the mapping space is `[2026-08-12]`

Queue item 26. **Host-only**: no device dispatched, no compiler run. Every number from
`programming_examples/transformer_layer/study/mapping_space.py`, pinned by
`run_mapping_space_tests.lit`; `python3 study/mapping_space.py` (~1 min), `--axes` for the axis table
with each bound's source; `python3 study/run_host_tests.py` 393/393 including the 36 predicate
checks, pinned by `run_study_host_tests.lit` (393/20 modules). Both in the PR-safe allowlist (10 → 11).

| | points | |
|---|---:|---|
| raw axis product | **293,601,280** | every axis at its full range |
| before legality | **115,343,360** | after the divisibility the builders `raise` on (Timeloop's first tier, [44 §bibliography](44-mapping-frameworks-synthesis.md)) |
| after legality | **3,721,772** | a 31× cut, **96.77%** removed |
| priced, not refused | **2,181,680** | **59%** of the legal space is over the per-column shim budget under the placement the tools produce, and stays in |

The item's hypothesis — that our space, like iron's, collapses to a hand-authorable table (iron's
legality leaves **seven** legal `(parallel_heads, parallel_ffn)` tails, a two-axis slice of its
space, [25 (38 §2.3)](25-mode-rebuilds-and-results.md)) — is falsified; the conclusion it wanted
holds because the predicate is **static**. Against iron on a comparable slice (axes differ: iron
heads × FFN branches, ours FFN width × sequence lanes — `gemm_herd_x` ↔ `parallel_ffn`,
`parallel_bands` ↔ `parallel_seq` — and no `parallel_heads` analogue): spatial replication 7 vs
**21**; herd widths (`gemm_herd_x`, `norm_herd_x`) 16; whole structural sub-space (forms × fold ×
widths × bands) **428**; structures × scope-per-seam **15,347** against iron's hardcoded
`TOPOLOGY_PLACEMENTS` — so placement must be derived (`air-place-herds`), not tabulated.

**Refused (no placement can route it) vs priced.** The line is placement invariance. Refused: a herd
with > 2 herd-direct L3 operands (item 10's control, MEASURED 12 `aie.packet_flow`, 3 per column);
segment shim demand > 16 slots (J1: 8 × 2 = 16 full before the third stream); cores > 32 or widths
not packing into 4 rows × 8 columns ([31 (31b §3.5)](31-resident-tail-r1-record.md): nine `[4,1]`
herds refuse with *"row index (6) must be less than the number of rows in the device (6)"*, eight
place); L1 over 64 KiB (`norm_tail._stage_l1_bytes`, `rows_per_call` 8, `cols` 768); memtile over
6 MM2S / 6 S2MM (`ffn_accum.MAX_FEED_CHANNELS`); `Para` at a dependent seam (TileFlow §4.1). Priced:
the per-column demand under doc 23's stacking rule, charged `min(1, budget/demand)`. R1 shows why
that cannot be a cliff: a placement exists with worst column 1 and the shipped allocator produced
2, both legal. **59% priced is the finding that matters**: had the shim budget been a hard filter
(the proposal [44](44-mapping-frameworks-synthesis.md) corrects), 2,181,680 routable designs and
every instance of the failure mode the study exists to see would have been deleted — AIR does not
refuse an over-subscribed column, it packet-multiplexes silently (item 10: zero inbound `aie.flow`,
12 `aie.packet_flow` on a design 50% over).

**Controls, run on every invocation inside the gate.** NEGATIVE: item 10's over-budget design
(one `[4,1]` herd, 3 herd-direct L3 operands) → REFUSED, `12 of 16 slots, 3 per column, predicted
[3,3,3,3,0,0,0,0]`, matching aircc 2026-08-12 (0 inbound `aie.flow`, 12 `aie.packet_flow`).
POSITIVE: R1's shipped interior → ADMITTED, `shim MM2S 7/16, 4 shim→core + 3 shim→memtile`,
reproducing [31 (31b §3.6)](31-resident-tail-r1-record.md)'s measured 7 of 16, 4 + 3 with no compiler;
R1's memtiles `[(4, 2), (4, 5)]` (up feed 4/6 MM2S, 2/6 S2MM; down 4/6, 5/6); herd inventory 9 →
REFUSED, 8/7/5 place; J7a's column budget `[2,2,2,2,2,2,2,2]`. Tamper-verified with two injected
defects: clamping the shim refusal to its own budget is caught by the negative control (`3 per
column -> ADMITTED`, census 3,721,772 → 4,071,136); inverting TileFlow's combinator (`max` for `Σ`)
passes the negative control and is caught by the positive one (R1 counts `2/16 (0 shim→core + 2
shim→memtile)` vs 7/16; J7a predicted `[1,1,…]`). A `max`-instead-of-Σ combinator is invisible to
any single-herd control — doc 44's charge against MAESTRO. Two defects the controls caught in the
module itself: `core_s2mm=min(CORE_S2MM, …)` (a demand clamped to its own budget, twice; now raw,
`test_no_demand_is_clamped_to_its_budget`), and a GEMM L1 byte formula derived from prose
(`tile_k` 64 is 91 KiB, not "just over" 64; the first version invented a wall refusing
`gemm_herd_x` 1 and 2; the wall is `MAX_L1_TILE_K`, already the axis bound). The first draft's
census read 0 legal points and `main()` refused itself.

**Axes** (the R2 resident tail `nt1 → up → gelu → down → nt2`): `nt1_form`/`nt2_form` fused, j7a;
`gemm_fold` split, folded; `gemm_herd_x` 1–4 (`ffn_accum.MAX_PLACEABLE_HERD_X`, MEASURED:
`aie-place-tiles` refuses the accumulator pair's shim slots at 6 columns); `norm_herd_x` 1–8;
`tile_k` 8,16,24,32 (`MICRO` = 8, `MAX_L1_TILE_K` = 32); `rows_per_call` divisors of 64
(`TILE_M`); `parallel_bands` 1–8 — **nothing in this tree bounds it** (the shipped builder refuses
above 1, `ffn_resident` requires `seq_len == TILE_M`; the census names it every run and the lit
pins the line); routing × 5 operands herd-direct / L2-staged (`packed1`, `gamma1`, `w_up`,
`w_down`, `gamma2`); scope per seam Seq, Shar, Para, Pipe ([44](44-mapping-frameworks-synthesis.md)'s
2×2 on [03](03-measurement-model.md)'s words). Excluded: R1's column partitioning (a
different module), attention, the norm tails' internal seams.

**Does enumeration beat search? Yes, because the predicate is static** — tens of microseconds a
point; the census walks ~10⁵ (structure, seam vector) pairs and multiplies in tiling/routing
sub-counts (`test_the_factorisation_matches_brute_force`). The binding constraint is not search but
`generate_fusion_plans` — a builder that can emit an arbitrary point of the 15,347-structure
sub-space; we have `ffn_resident.py` at one point. No MCTS/GA before a parameterised builder exists.
By-products: **the shim budget, not the array, caps band parallelism** — four lanes of the leanest
co-resident tail want 20 of 16 shim MM2S slots
(`test_the_band_axis_is_capped_by_the_shim_budget_not_by_cores`); a `Seq` seam lifts it. **Eight-wide
norm herds and the FFN cannot share one segment's shim budget** — `norm_herd_x` 8 survives in R2
only through a `Seq` seam (`test_the_axis_values_legality_eliminates_outright_are_the_pinned_ones`).
The model knows no bytes or cycles, no column, and whether the shim clauses are jointly sufficient
(both necessary, none of the herd multisets produces a joint counterexample).

## 15. Compiler items not started

- **H5 — dynamic channel indices.** Split `air.channel` `indices` into a static dimension (flow/tile)
  and a dynamic one resolved by a runtime counter modulo depth. Today a 64-band loop fully unrolls
  and exhausts the hardware (16 locks per tile, §6), and where the loop does not unroll an
  IV-dependent offset is now refused by §7 rather than frozen. mlir-aie solved it one layer down
  (`-aie-objectFifo-stateful-transform` `dynamic-objFifos`, the default, with static LCM unrolling
  as fallback). Medium.
- **H6 — per-region `omit_pingpong`.** Re-scoped to zero compiler change first: the tiling half has
  `air.shim_dma_tile_sizes` per launch (`AIRDependencyScheduleOpt.cpp:7868`, CLI outranks), the
  ping-pong half works per loop (H4 struck); only herd/segment granularity is missing
  (`AIRDialect.cpp:71`). Step 1 is a measurement: the mixed FlashAttention + 4096-row GEMM ELF with
  no tiling kwarg, per-launch attributes, the opt-out on the attention loops. FlashAttention needs
  `omit_pingpong="all"` + `[1,1]` or it does not place; the GEMMs need `[2,2]`. Gates J7c.
- **H7 — the offset-subview wall**, two sides of one boundary: mlir-aie's
  `traceSubviewToBlockArgument` (`lib/Dialect/AIEX/Utils/AIEUtils.cpp:19`) bails unless rank-1 →
  rank-1; `memref.cast` cannot erase a nonzero offset back to an identity layout. The row-0 trick in
  `o_gemv_ffn_multi.py:142` works only because an offset-0 subview+cast folds away. Third route: pass
  the band index as a launch operand and let `dma_memcpy_nd(..., src_offsets=[row, 0], ...)` address
  it — what every banded builder does. J7a took the strided-callee route (§11). Bounded.

## 16. Tranche J as doc 16 left it, and the finish line

- **J2** — `attn_scores` (4096×64×4096) passes on hardware with hand-chosen tiles, zero mismatches;
  `attn_output` (4096×4096×64) timed out on the one configuration tried, of 828 legal. Attention
  placement is the confound in any mode-versus-mode comparison.
- **J3** — DONE 2026-08-08: two 16-rung walks, `fused` leads at 512 and 2048, `coarse` at 4096,
  slopes split on attention placement (device 1.03–1.17, host 1.23–1.27); 1024 indistinguishable.
  [25](25-mode-rebuilds-and-results.md).
- **J4** — DONE 2026-08-08, `db2b1b53`: distinguishability clause 3 asserts `herd_launches` (404 vs
  146) not `runlist_entries` (391 vs 131, true by construction). Three coordinated edits: the
  selftest's violating fixture had `herd=160` against coarse's 146, so it drops to `herd=140` and
  keeps `entries=150` high, violating only the new clause; selftest 30/30 both directions
  ([01](01-original-plan-superseded.md)).
- **J5** — wire `decoder_gpt2`: done — all four modes run the decoder since the fused decoder's
  re-execution fix (§13, 2026-08-20), and `gpt2_512` / `gpt2_medium_1024` walked on 2026-08-20
  ([54 §4](54-first-full-profile-and-decoder-families.md); doc 16 itself still listed J5 as not started).
- **J6** — power needs a decision before code: iron's backend calls `sudo -n turbostat` and
  passwordless sudo is unavailable; `turbostat --no-msr` emits no samples; RAPL `package-0` is the
  CPU package and `amdgpu`'s `PPT` is the SoC rail, both root-free; no sensor measures the NPU, so a
  cross-mode power comparison largely measures host work.
- **Finish line**: Phase F gate — `execution-smoke-test` yields ≥1 `run_status=passed` row per CSV
  and reports the first `failure_message` verbatim (iron's smoke test reported 21/21 passed on a
  machine where every measurement had failed); Phase G gate — a full profile with a complete
  `results_manifest.json`, **passed 2026-08-20** (36 rungs, 21 measured + 15 derived skips, walked
  twice, [54](54-first-full-profile-and-decoder-families.md)).

## 17. Rules the compiler phases left standing

- **Every H item's gate** is the compiler lit subset, the transformer-layer suite on hardware, and
  `make verify` over the shipped models — after a build **and install** (the examples resolve
  `aircc` from `install-xrt`; a build without an install tests the previous compiler). `check-air-mlir`
  is seconds and not predictive: it stayed green through both of Phase H's spec errors, which
  surfaced an hour into the ten-model leg. The decode-throughput leg (9.43 tok/s floor on
  `llama32_1b`) exists because every other leg is correctness-only.
- **Refuse vs skip**: skip and warn when the untransformed program is correct (§4); refuse when
  there is no correct fallback (§1's decline warning, §7, §9's MIMO).
- **Tests**: keep inputs byte-identical, change CHECK lines; new cases for new paths (§4). A
  regression lit is verified failing before the fix, **for the reason intended** (§7).
- **Do not** widen a tolerance, disable ping-pong globally, annotate shipped models' callees, add
  `air.disable_ping_pong` to builders, or re-fingerprint a tamper baseline to clear a halt.
- **A phase window is not just its own work**: unrelated operator edits to gate files inside it
  (registry sweeps, checker repairs) make the tamper check unable to distinguish them (§7).
- Running a hardware fixture from a fingerprinted directory leaked `air.mlir`, `air.elf`,
  `addnorm_pre_add.o`, `air_project/`; the fixture `chdir`s into a temp directory it owns, and a
  catch-all `.gitignore` there was itself a weakened gate (hides the leak from the guard) — removed.
- Resume a halted driver phase with `resume-at`, never `resume` (the latter redoes the implement
  step and empties the review diff) — historical, the harness is retired.
