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
