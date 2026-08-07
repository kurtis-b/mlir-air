# 18 — Phase H1s: the safety proof declines to *transform*, never to compile

Read [17](17-phase-h-compiler-hardening.md) first, at least its banner and
§"[RESOLVED: H1's spec was wrong](17-phase-h-compiler-hardening.md#2026-08-06-resolved-h1s-spec-was-wrong--decline-to-transform-not-refuse-to-compile)".
This phase is the correction that document arrived at, run as a fresh phase rather than a resume:
the spec changed mid-run, and the driver's fixture and objective check changed with it, so a
resume would have carried a fingerprint baseline taken against a specification that no longer
holds.

**Do not re-derive what Phase H already landed.** `air-fuse-packet-put-loops` (the actual fix for
the two-trip miscompile), H2's external-call classifier, H3's `AIRDialect::verifyOperationAttribute`
and 522 lines of new compiler tests are all committed and green through the first three gate legs.
This phase changes one verdict and the tests that assert it.

## The rule

When the ping-pong rotation cannot be proven safe for a buffer it privatizes, the pass **skips**:
leaves the loop single-buffered, emits a *warning* naming the loop, and compilation proceeds.
Compilation aborts only for IR that is genuinely malformed, which this never is.

The dependency-free `WaitAllOp` placeholder stays forbidden. Skipping means not transforming at
all — not transforming with an empty edge set. That was always the real defect.

## What measurement already established, on 2026-08-06

Do not spend session time rediscovering any of this. All four rows were compiled against the
currently installed build, and the labeled IR read from aircc's `--debug-ir` dump:

| callee | weight DMA | compiles | labeled (`unroll`) | `hoist_alloc` set |
|---|---|---|---|---|
| unannotated | in loop | yes | **no** | — |
| unannotated | hoisted | yes | **no** | — |
| annotated | in loop | yes | **yes** | 4 — the three tiles **and** the weight |
| annotated | hoisted | yes | **yes** | 3 tiles; the weight is **excluded** |

Three things follow, and they are the shape of this phase:

- **The `hoisted` shape was never a hazard.** The rotation already excludes a buffer filled before
  the loop, which is exactly why it is correct, and why `llama32_1b_int4`, `qwen3_0_6b` and
  `qwen3_1_7b` are correct. The old fixture clause demanding a refusal was asserting the wrong
  thing.
- **An unannotated external call is never guessed at.** Both unannotated rows are left
  single-buffered. So the two-trip `inside` loop is correct because of `air-fuse-packet-put-loops`,
  **not** because of ping-pong — it never gets ping-ponged at all.
- **The transform still fires where it is provable**, and privatizes per buffer rather than per
  loop. That is the property the new fixture pins, and the one a narrowed predicate would silently
  lose.

## Work items

### H1a — `Refuse` becomes `Skip`

`mlir/lib/Transform/AIRDependencyScheduleOpt.cpp`.

- `provePingPongSafety` returns `PingPongSafety::Refuse` in exactly one place (the buffer that is
  read with no producer provably refilling it every iteration). It becomes `Skip`.
- The pass driver's pre-scan drops `anyRefusal` and its `signalPassFailure()`, and the `Refuse`
  arm of the `switch` goes with it. Prefer deleting the enumerator over leaving it unreachable: a
  verdict nothing can return is a trap for the next reader.
- **Keep the diagnostic's shape.** Naming the loop and pointing at `air.disable_ping_pong` and
  `--omit-ping-pong-transform` is right; it is the severity that is wrong. The skip arm's message
  already reads correctly — reuse it rather than writing a third variant.
- `ConstructPingPongDependencyPattern` re-proves safety before mutating and already `emitWarning`s
  and returns `failure()` on a non-`Safe` verdict. Check it needs no change beyond the enumerator.

Note `air.disable_ping_pong` is checked by `isPingPongCandidate` and is believed not to work
(setting it produced byte-identical output). That is **H4 and not this phase** — but this phase's
warning tells users to reach for it, so if you find the reason in passing, record it in
`work_not_completed` rather than fixing it here.

### H1b — the two existing tests assert the new severity

Three `expected-error` lines assert the old one:

- `mlir/test/Transform/AIRDependencyScheduleOpt/label_ping_pong_alias_escape_proof.mlir:80`
- `mlir/test/Transform/AIRDependencyScheduleOpt/label_ping_pong_external_call_proof.mlir:522`
- `mlir/test/Transform/AIRDependencyScheduleOpt/label_ping_pong_external_call_proof.mlir:606`

**Keep every test input byte-identical and change only the CHECK lines.** This is not style. Three
halts in the previous run were lit inputs edited to accommodate new behaviour — consumers added to
dodge a refusal, a callee annotated so it stopped exercising the unannotated path H2 changes — and
each was removed coverage whatever the intent. If the transformed path also needs coverage, add a
**new** case; do not convert an old one.

These two files, and one more named below, are the only gate files this phase may touch.
`guard_gate_files()` now fingerprints `mlir/test/**/*.mlir`, which it did not during the previous
run, so anything else halts.

### H1c — a new test for the property the old one only claimed

`mlir/test/Transform/AIRDependencyScheduleOpt/label_ping_pong_loop_invariant_not_rotated.mlir`
(the name is in the allowlist; use it).

Over a loop with an **annotated** callee and a buffer filled *before* the loop, assert that the
loop **is** labeled and that the loop-invariant buffer is **not** in the `hoist_alloc` set. A
`CHECK-NOT` on `hoist_alloc` against that buffer's type is the direct form.

This is the clause that keeps the suite discriminating. Asserting only "does not crash" or only
"is not labeled" is satisfied by a pass that has been narrowed until it never fires — which passes
every correctness check ever written and costs throughput silently.

## What this phase must not do

- **Do not touch `programming_examples/transformer_layer/builders/addnorm.py`.** Lifting its
  one-trip guard is J1 and belongs to a later phase. A session that "fixes" addnorm here has moved
  the evidence instead of the defect.
- **Do not disable ping-pong globally**, or narrow the predicate until it stops firing. Both are
  measurable: gate leg 4 compares decode throughput against a recorded floor, and the fixture's
  `annotated` clause demands the transform actually happen.
- **Do not annotate the shipped models' callees** or add `air.disable_ping_pong` to their builders
  to make leg 5 pass. That hides the question.
- **Do not widen a tolerance anywhere.** No numeric in this phase's gate is yours to move.

## Gate

```
flock -x -w 1800 /tmp/mlir-air-npu.lock  agents/scripts/port-loop/gate-h.sh
```

Five legs now — build + install, `check-air-mlir`, the transformer-layer suite on hardware,
**decode throughput against a recorded floor**, then `make verify` over the ten shipped models.

Leg 4 is new. It exists because every other leg is correctness-only and this phase's characteristic
failure is invisible to all of them: a predicate that declines to transform more loops than it
should holds every numeric exactly and costs throughput. Dropping ping-pong regressed a shipped
model 12.4 → 7.8 tok/s (`llms/shared/infra/backend_presets.py`). The floor lives in
`agents/scripts/port-loop/throughput-baseline.json`, which is fingerprinted and in no allowlist.

Leg 5 is where both of the previous run's substantive spec errors surfaced, an hour in, after legs
1–3 were green. Run `check-air-mlir` yourself first — it takes seconds — but do not treat it as
predictive: it stayed green through both.

Then the driver runs four opposed clauses from a fixture it owns
(`agents/scripts/port-loop/fixtures/addnorm_multitrip.py`, fingerprinted, in no allowlist), all at
`cols=64, rows=8, rows_per_call=4` — two trips of the row loop, the exact point
`builders/addnorm.py` measured the miscompile at:

| variant | must compile | must be exact | must be labeled | weight in rotation set |
|---|---|---|---|---|
| `inside` | yes | yes | no | — |
| `hoisted` | yes | yes | no | — |
| `annotated` | yes | yes | **yes** | **yes** |
| `annotated_hoisted` | yes | yes | **yes** | **no** |

**Every variant must compile.** That is the whole correction: a refusal reintroduced here fails all
four rows at once. `annotated` and `annotated_hoisted` are the opposed pair — one demands the
transform fire, the other demands it leave one specific buffer alone — and no blunt change
satisfies both.

**Know what this check is and is not.** `[2026-08-06]` All four clauses were run on hardware
against the build that exists *before* your change, and all four **already pass**. So the objective
check is not a failing test this phase turns green, and passing it is not on its own evidence that
the phase did anything. Its job is narrower and still worth having: H1a's edit sits inside the
predicate that decides *which* loops get labeled, and the risk of "simplify while removing the
verdict" is that the labeling changes as a side effect. These four rows pin the labeling decision
per buffer, so that side effect cannot pass silently. The evidence that the phase achieved
something is gate leg 5 and the answer to the open question below.

## `[2026-08-06]` RESOLVED, and the phase PASSED

**The open question is answered: `Refuse` was already unreachable on real input.** The session ran
`make verify` on `llama32_1b_int4` against the *pre-change* installed (attempt-5) compiler and it
passed, with genuine fresh `aircc` compiles of every prefill and decode kernel — no cache hits.
Attempt 5 (`1514e553`, "refuse only what the rotation actually privatizes") had already narrowed
the verdict past everything real; the leg-5 failure recorded in [17](17-phase-h-compiler-hardening.md)
is attempt **four**'s behaviour.

**So this phase removed a LATENT hazard, not a live one**, and that is the claim the record should
carry. It is still worth having done — a verdict that aborts the build has no place in a transform
whose prior art (`memref::multiBuffer`, IREE, Triton, TVM) all declines the transform instead — and
the two new tests pin the labeling decision per buffer, which nothing did before. But leg 5 passed
10/10 at the phase base as well, and saying otherwise would overstate it.

**Outcome**, 109 minutes, 4 agent invocations, three Codex rounds all `verdict=pass blocking=0
weakened=0`:

| leg | result |
|---|---|
| 1 build + install | pass |
| 2 `check-air-mlir` | pass — 488 passed, 7 UNSUPPORTED, 7 XFAIL, 0 failures |
| 3 transformer-layer suite on hardware | pass |
| 4 decode throughput | `llama32_1b` 11.01 tok/s against a 9.43 floor — pass; `llama32_1b_int4` `NOT GATED` |
| 5 ten shipped models | 10/10 pass |

Objective check green on all four fixture variants. Commits: `cb7be1ab` (the verdict),
`610fadc2` (the two existing tests), `5a380615` (the new loop-invariant test).
