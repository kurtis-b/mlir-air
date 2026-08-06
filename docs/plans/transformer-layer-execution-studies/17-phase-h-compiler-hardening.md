# 17 — Phase H: Compiler hardening (H1–H3)

The first phase in this plan to change `mlir/` — the AIR compiler itself. Read
[16](16-compiler-work-and-remaining-essence.md) first; it holds the root cause and the evidence.

Three items. They are one phase because H1 without H2 turns a live miscompile into a build failure
for a program that ought to work, and H2 without H1 fixes one unsound case while leaving the pass
willing to transform others it cannot prove.

## The defect, precisely

`air-label-scf-for-to-ping-pong` marks a loop `unroll = 2` and its allocs `hoist_alloc`, then
`air-ping-pong-transform` duplicates the buffers and rebuilds the dependency graph. Eligibility
checks only that no alloc is filled by more than one non-exclusive `channel.get` per iteration
(`AIRDependencyScheduleOpt.cpp::isPingPongCandidate`). It does **not** require a producer for both
duplicated halves, a recognized consumer, or that every use is understood.

The gap is in the classifier. `checkOpOperandReadOrWrite` (`mlir/lib/Util/Util.cpp`) resolves a
memref use through memory effects, `ChannelPutOp`, `ChannelGetOp` or linalg, and returns `'u'`
otherwise:

```cpp
if (mlir::hasEffect<mlir::MemoryEffects::Write>(owner, op_operand.get())) return 'w';
if (mlir::hasEffect<mlir::MemoryEffects::Read>(owner, op_operand.get()))  return 'r';
// ... ChannelPutOp / ChannelGetOp / linalg ...
else return 'u';
```

**An external kernel `func.call` registers no memory effects, so the compute step is invisible.**
Unknown uses are silently omitted from dependency construction, and when a producer or consumer set
comes out empty it becomes a placeholder with no operands rather than a rejection:

```cpp
if (!yield_operands[i])
  yield_operands[i] = air::WaitAllOp::create(..., SmallVector<Value>{}).getAsyncToken();
```

So the ping/pong halves get no reuse edge protecting a buffer until the kernel has finished reading
it. One trip is safe because nothing is reused; two trips corrupt 481–497 of 512 elements.

## Work items

### H1 — refuse what cannot be proven safe

Strengthen `isPingPongCandidate` to bail unless, for every alloc it would duplicate:

- every memref use is classified (**no relevant use returns `'u'`**);
- there is a producer that definitely executes on each logical iteration, for **both** halves;
- there is at least one recognized consumer;
- producer and consumer front/back tokens are non-null.

Emit a diagnostic naming the loop and the reason. **Do not emit an empty `WaitAllOp` placeholder
when a set is empty — reject instead.** That placeholder is the mechanism by which a missing edge
becomes silent wrong data.

This is the item that matters most even if H2 slips. Every comparable compiler bails rather than
guessing: upstream `memref::multiBuffer` returns `failure()` unless it can point at a user that
provably clobbers the buffer (and its `overrideBuffer()` recognizes only `memref.copy`, so a custom
DMA would not qualify); IREE calls it with `skipOverrideAnalysis=false`; Triton gates entry on an
explicit precondition list; TVM `ICHECK`-aborts. Silently emitting wrong numbers is the outlier.

### H2 — teach the classifier about external kernel calls

A `func.call` whose callee carries `llvm.emit_c_interface` should classify its memref operands from
the callee's signature/argument attributes rather than falling through to `'u'`. Without this, H1's
bail-out fires on the *legitimate* multi-trip loop and the one-trip rule stands.

Be conservative about what counts as a write. If read-versus-write cannot be established for an
operand, that operand is `'u'` and H1 rejects — which is the correct outcome, not a regression.

### H3 — `AIRDialect::verifyOperationAttribute`

Set `hasOperationAttrVerify = 1` on the dialect and implement the hook. Validate each `air.*`
discardable attribute's type **and the op type it is allowed to sit on**, the way
`GPUDialect::verifyOperationAttribute` does for `gpu.container_module`.

`grep -rn verifyOperationAttribute mlir/` currently returns nothing, so no `air.*` attribute is
validated anywhere. With the hook, the verifier hands every dialect-prefixed attribute to it on
every op after every pass, so a misplaced or mistyped attribute is caught at the pass that broke it.
Start with `air.disable_ping_pong` and `air.shim_dma_tile_sizes`, and move the existing inline
`emitOpError` checks for the latter out of `AIRDependencyScheduleOpt.cpp` so they run even when the
consuming pass does not.

## What this phase must not do

- **Do not touch `programming_examples/transformer_layer/builders/addnorm.py`.** Lifting its
  one-trip guard is J1 and it belongs to a later phase, after this one proves the compiler is right.
  A session that "fixes" addnorm here has moved the evidence instead of the defect.
- **Do not disable ping-pong globally** to make a test pass. That is measurable: dropping ping-pong
  regressed a shipped model 12.4 → 7.8 tok/s, recorded in `llms/shared/infra/backend_presets.py`.
- **Do not widen a tolerance anywhere.** No numeric in this phase's gate is yours to move.

## Gate

```
flock -x -w 1800 /tmp/mlir-air-npu.lock  agents/scripts/port-loop/gate-h.sh
```

Four legs: rebuild **and install** the compiler; `check-air-mlir`; the transformer-layer suite on
hardware; then `make verify` over the ten shipped models. The install is not optional — the examples
resolve `aircc` from `install-xrt`, so a build without an install tests the previous compiler and
proves nothing.

The driver then runs two hardware checks you cannot influence, from a fixture it owns
(`agents/scripts/port-loop/fixtures/addnorm_multitrip.py`, fingerprinted and in no allowlist):

- **`--variant inside`** — a legitimate two-trip loop, every L1 buffer refilled each iteration. It
  must compile and produce **zero mismatches**. This is H2's proof.
- **`--variant hoisted`** — the weight DMA lifted out of the loop, so the buffer carries data across
  iterations and rotating it is genuinely unsound. The compiler must **refuse it with a
  diagnostic**. Producing any answer, right or wrong, fails the phase. This is H1's proof, and it is
  what stops "disable ping-pong everywhere" from passing.

Both fixtures run at `cols=64, rows=8, rows_per_call=4` — the exact point
`builders/addnorm.py` measured the miscompile at.

## `[2026-08-06]` Attempt 1: the root cause was different, and the refusal is too strong

Two things came out of the first attempt. Both are recorded here rather than left in a session log,
because the second is what the next attempt has to fix.

**The root cause in [16](16-compiler-work-and-remaining-essence.md) and above is wrong.** Measured
on hardware: compiling the fixture's `inside` variant with `--omit-ping-pong-transform=all`
produces the **identical 481/512 corruption**. Ping-pong is therefore not the cause. The actual
defect is launch-side per-channel put-loop grouping against the tile ring's per-iteration order
under packet multiplexing, fixed by a new `air-fuse-packet-put-loops` pass plus modelling
packet-typed channels as one shared stream resource so the token chain survives pruning. H1 and H2
as specified were **necessary for the `hoisted` refusal but insufficient for `inside` correctness**.
The session reported this rather than quietly editing the plan, which is the behaviour asked for.

**And H1's refusal is far too strong.** Gate leg 3 failed with **8 of 24 tests failing** — every
GEMM-bearing operator and all four execution modes:

```
run_npu2_qkv_proj_peano   run_npu2_ffn_peano      run_npu2_block_peano    run_npu2_coarse_peano
run_npu2_offload_peano    run_npu2_runlist_peano  run_npu2_fused_peano    run_npu2_runlist_gate
```

each on the same diagnostic:

> `'scf.for' op is a ping-pong candidate that cannot be proven safe to transform: 'func.call' may
> access a memref that is not privatized by this loop's ping-pong rotation (defined outside the loop
> or not refilled per iteration), and no callee argument attribute establishes the access as a read
> or a write.`

These are designs that **work today**. Codex review round 1 raised exactly this ("the proof
hard-fails on unclassifiable uses of unrelated memrefs") and the fix rounds did not close it.
`check-air-mlir` stayed green, so AIR's own tests do not cover the case — only the real examples do.

**What the next attempt must change.** The refusal has to be scoped to the memrefs the rotation
actually endangers:

- Refuse only when an unclassifiable (`'u'`) or may-read-write (`'b'`) use lands on **a memref this
  loop's rotation privatizes** — one of the duplicated allocs. A call touching any *other* memref,
  including one defined outside the loop, is none of this transform's business and must not block it.
- A `'b'` use on a rotated buffer is only unsafe if the buffer is *read* without a per-iteration
  producer. Write-only and alloc-only buffers stay vacuously safe, as attempt 1 already had it.
- Keep the diagnostic's shape — naming the loop and pointing at `air.disable_ping_pong` and
  `--omit-ping-pong-transform` is good. It is the trigger that is wrong, not the message.

The whole point of H1 is to stop *silent wrong data*. Refusing to compile programs that are already
correct trades one failure mode for a worse one, and the ten-model leg — which never ran — would
have been the more expensive place to discover it.

**Do not** satisfy this by weakening the fixture, by adding `air.disable_ping_pong` to the example
builders, or by narrowing the refusal until it never fires. `--variant hoisted` must still be
refused; that clause is what proves the transform still discriminates.

## `[2026-08-06]` Attempt 2: halted on weakened gates, and an artifact leak in the harness

**Codex round 2 reported two weakened gates, and it is right.** The diff edited two of the
compiler's own lit tests to add consumers that dodge the new refusal:

- `mlir/test/.../label_ping_pong_loops.mlir` — at the phase base this test required ping-pong
  labeling of bare alloc/dealloc loops. `channel.get`/`channel.put` consumers were added
  specifically to avoid the no-consumer refusal, which removes the coverage proving those
  previously-accepted safe loops still compile.
- `mlir/test/.../label_ping_pong_multifill_alloc.mlir` — the single-fill and mutually-exclusive
  broadcast cases previously ended in deallocation with no read consumer. Drain `channel.put` ops
  were added, so neither case detects the new rejection of safe fill-only buffers any more.

**That is the wrong direction.** Attempt 1's own report already stated the correct rule —
"alloc-only and write-only buffers are treated as vacuously safe" — and these tests are exactly the
evidence for it. If the refusal fires on them, the refusal is wrong; the tests are not. Revert both
test edits and make the predicate not fire on a buffer that is never *read* without a per-iteration
producer. A test edited to stop detecting a regression is the one change this harness halts on
unconditionally, and it will halt again.

Note these files are `mlir/test/**/*.mlir`, which `guard_gate_files()` does **not** fingerprint — it
covers `.lit` files, not `.mlir`. So the tamper check would not have caught this; only the Codex
`weakened_gates` layer did. That is a real gap in the fingerprint set, and widening it to the
compiler's own lit inputs is worth doing before the next compiler phase.

**Separately, and my fault rather than the session's:** running this phase's fixture from
`agents/scripts/port-loop/fixtures/` leaked `air.mlir`, `air.elf`, `addnorm_pre_add.o` and
`air_project/` into the driver's own directory, and `commit_step`'s `git add -A` committed two of
them. That directory is fingerprinted and covered by no allowlist, so the tamper check would have
halted the phase for artifacts the phase did not intend to create.
[15](15-environment-notes.md) predicted precisely this and it was still missed. The fixture now
`chdir`s into a temp directory it owns — a caller-supplied CWD is not a fix, because the leak
depends on where the caller happened to stand.

**And the `.gitignore` I added alongside it was itself a weakened gate**, which Codex caught on the
next round. A leaked artifact in this directory gets tracked by `commit_step`'s `git add -A`, and
`guard_gate_files()` then flags it as an unauthorized addition to a fingerprinted path with an empty
allowlist — a halt. That is *detection*. A catch-all `*` rule hides the leak from git and therefore
from the guard, trading a loud failure for silence, which is the exact anti-pattern this harness
exists to prevent. The `chdir` already prevents the leak; the guard must stay able to catch any
future one from code that does not go through this fixture. The `.gitignore` is removed.

## `[2026-08-06]` The rule for existing tests, after three weakened-gate halts

Every halt in this phase so far has been a lit test edited to accommodate the new behaviour, and
each was flagged correctly. The three `ping_pong_shared_resident_ring*.mlir` tests are the clearest
case, and I initially judged them benign — wrongly. Adding `llvm.emit_c_interface` and
`llvm.readonly` to their `@acc` declaration is not "making the test realistic": at the phase base
those tests covered the **unannotated callee** path, which is precisely the path H2 changes. After
annotation they no longer exercise it. That is removed coverage, whatever the intent.

**The rule for this phase, and it is not negotiable by a session:**

When H1 or H2 changes the outcome for an existing test's input, **keep the input exactly as it is
and update the CHECK lines to assert the new, intended outcome.** For an unannotated external call
that is now single-buffered rather than ping-ponged, the test should assert the Skip. If the
transformed path also needs coverage, add a **new** case with the annotated callee — do not convert
the old one.

Annotating the input to preserve the old outcome deletes the evidence that the behaviour changed at
all, which is the one thing this phase most needs recorded. Three halts have now been spent
relearning it.

Note again that `guard_gate_files()` fingerprints `.lit` files and not `mlir/test/**/*.mlir`, so the
tamper check cannot see any of this. Only the Codex `weakened_gates` layer can, which is why it has
been the halting layer three times running, and why widening the fingerprint set is on the list.

## `[2026-08-06]` Attempt 4: leg 3 fixed, and leg 4 found the real tension

**Leg 3 now passes 24/24** — the over-refusal that failed 8 of 24 on attempt 1 is gone, and all
three `weakened_gates` halts are resolved (the reverted tests stayed reverted, and the CHECK-line
rule above held). Reviews 1–3 all cleared, round 3 clean with no fix needed.

**Leg 4 then ran for the first time and caught three regressions**: `llama32_1b_int4`,
`qwen3_0_6b`, `qwen3_1_7b`. Seven models pass. All three fail the same way — `aircc` refusing to
compile:

> `'scf.for' op is a ping-pong candidate that cannot be proven safe to transform: 'func.call' may
> access a memref that this loop's ping-pong rotation does not privatize and whose data carries
> across iterations (it is filled before the loop, within the loop's own scope)`

**That is the same shape `--variant hoisted` asserts must be refused** — a buffer filled before the
loop and read inside it through an external call. And these three models produce correct output
today. So the fixture's premise and the shipped models disagree, and one of them has to give.

**The resolution, for the next attempt.** The hazard is not "data carries across iterations". It is
"a buffer *the rotation duplicates* is not refilled for both halves". If the rotation leaves a
loop-invariant buffer alone — one physical buffer, read by every iteration — there is no hazard and
refusing is wrong. So:

- Refuse **only** when the unprovable buffer is one this loop's ping-pong rotation actually
  privatizes (i.e. it is in the `hoist_alloc` set being duplicated). A loop-carried buffer the
  rotation does not touch is not this transform's problem.
- Verify the `hoisted` fixture still refuses under that rule. If its `l1_w` *is* rotated, it will,
  and both the fixture and the three models are satisfied. **If it is not rotated, then the fixture
  is asserting the wrong thing** — say so in `work_not_completed` rather than bending the rule to
  keep it green. `builders/addnorm.py` documents that hoisting its weight DMA corrupts, so there is
  a real hazard somewhere in that shape; the question is whether refusal is the right instrument.

Do not resolve this by annotating the three models' callees, by adding `air.disable_ping_pong` to
their builders, or by dropping the `hoisted` clause. The first two hide the question, the third
removes the only thing proving the transform still discriminates.

## `[2026-08-06]` RESOLVED: H1's spec was wrong — "decline to transform", not "refuse to compile"

The measurement settles it, and against me.

Run today against the current build: the `hoisted` fixture **compiles and produces numerically
correct output** (`XRTRunner: PASS!`), and `air-label-scf-for-to-ping-pong` does not label its loop
at all — no `unroll`, no `hoist_alloc` on any of its four allocs. The loop is *Skipped* to
single-buffered, and single-buffered is correct.

So the hoisted shape is safe precisely because the rotation leaves the buffer alone. That is also
why `llama32_1b_int4`, `qwen3_0_6b` and `qwen3_1_7b` are correct today. **The fixture was asserting
the wrong thing**, exactly as this document allowed for.

**The root error is in H1's specification, which is mine.** It says "hard-fails compilation with a
diagnostic". But the prior art it cites does no such thing: upstream `memref::multiBuffer` returns
`failure()`, which means *decline to transform* and leave the code alone — not *abort the build*.
IREE, Triton and TVM all bail out of the transformation, not out of compilation. I conflated
"refuse to transform" with "refuse to compile", and gate leg 4 caught the consequence: three shipped
models failing to build on programs that were always correct.

**The corrected rule:**

- When the pass cannot prove the rotation safe, it **skips** — leaves the loop single-buffered,
  emits a *warning* naming the loop, and compilation proceeds. Correctness is preserved; only the
  optimization is lost.
- Compilation aborts only for IR that is genuinely malformed, which is not this case.
- The dependency-free `WaitAllOp` placeholder is still forbidden: skipping means not transforming at
  all, not transforming with an empty edge set. That was always the real defect.

**What still needs proving, and how the fixture must change.** With refusal gone, `--variant
hoisted` can no longer discriminate by demanding an error. It should assert that the program
compiles, is numerically correct, **and was not ping-pong transformed** — the third clause is what
keeps it a discriminating test rather than a second copy of `inside`. Asserting non-transformation
from the Python runner is not straightforward; a lit test over the labeled IR (`CHECK-NOT: unroll`)
is the natural home for it, alongside the two the phase already added.

## What landed, and is worth keeping regardless

## What did land, and is worth keeping regardless

Committed on the branch, gated by four review rounds and three of the four gate legs:

- The **real root cause**, found by measurement and contradicting this document's original claim:
  the two-trip corruption is launch-side per-channel put-loop grouping against the tile ring's
  per-iteration order under packet multiplexing, **not** ping-pong. Compiling with
  `--omit-ping-pong-transform=all` reproduces the identical 481/512 corruption.
- `air-fuse-packet-put-loops`, plus packet-typed channels modelled as one shared stream resource.
- H2's external-call classifier and H3's `AIRDialect::verifyOperationAttribute`, both green through
  `check-air-mlir` (480+ passing).
- 522 lines of new compiler test coverage.
- Gate legs 1–3 green: build, install, `check-air-mlir`, and the transformer-layer suite at **24/24**.

## Risks

- **The gate is the widest in the plan.** A compiler regression surfaces as ten `make verify` runs
  failing an hour in. Run `check-air-mlir` yourself before you think you are done; it is seconds.
- **The classifier is shared.** `checkOpOperandReadOrWrite` is used well beyond ping-pong. Widening
  what counts as a read or write changes dependency graphs everywhere, which is exactly why leg 2
  and leg 4 exist. Prefer the narrowest rule that covers `llvm.emit_c_interface` callees.
- **`air.disable_ping_pong` may not work today.** Setting it on the row loop and rebuilding produced
  byte-identical wrong output — both arms 481/512. It is a discardable attribute, and those may be
  dropped by any pass that does not know them; PR #1664 already hand-patched four such sites. If H3
  surfaces the reason, say so; if it turns out to need promoting to an inherent ODS attribute, that
  is H4 and it is not this phase's job.
