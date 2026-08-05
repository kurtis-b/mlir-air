# Transformer-Layer Execution Studies — Port Plan

Porting the transformer-layer execution-strategy studies from the AMD IRON repository
(`iron`, commit `1e014c1`) into MLIR-AIR, then building two follow-on capabilities on the
resulting measurement harness: SOTA model coverage via sliding-window attention, and
quantized inference.

These documents are the working plan. They live in the repository rather than the published
docs site (`mkdocs.yml` excludes `plans/`), because they describe work in progress rather than
how to use MLIR-AIR.

## Read in this order

| Doc | What it covers |
|---|---|
| [00-context-and-goals.md](00-context-and-goals.md) | Why this port, what is being ported, success criteria |
| [01-port-inventory.md](01-port-inventory.md) | Per-artifact triage: port / adapt / rewrite / drop |
| [02-porting-conventions.md](02-porting-conventions.md) | **How iron code is refactored into MLIR-AIR house style.** Reviewable checklist |
| [03-measurement-model.md](03-measurement-model.md) | The execution-boundary taxonomy, the dispatch vector, CSV schema v1 |
| [04-phase-a-kernels.md](04-phase-a-kernels.md) | AIE2P device kernels |
| [05-phase-b-runtime-seam.md](05-phase-b-runtime-seam.md) | Runlist aggregation + BO liveness pooling |
| [06-phase-c-operators.md](06-phase-c-operators.md) | The six new operators as AIR builders — **overview**; the four sub-phase specs are [06a](06a-phase-c1-gate-and-small-operators.md) · [06b](06b-phase-c2-qkv-proj-and-ffn.md) · [06c](06c-phase-c3-mha-out-proj.md) · [06d](06d-phase-c4-coverage-sweep.md) |
| [07-phase-d-block-integration.md](07-phase-d-block-integration.md) | Single-block integration gate |
| [08-phase-e-execution-strategies.md](08-phase-e-execution-strategies.md) | The four execution strategies |
| [09-phase-f-study-harness.md](09-phase-f-study-harness.md) | The seven measurement studies |
| [10-phase-g-unattended-runner-and-ci.md](10-phase-g-unattended-runner-and-ci.md) | Unattended suite runner, CI wiring |
| [11-goal-sota-sliding-window.md](11-goal-sota-sliding-window.md) | Goal 1 — sliding-window / local-global attention |
| [12-goal-quantized-inference.md](12-goal-quantized-inference.md) | Goal 2 — quantized inference |
| [13-verification-and-acceptance.md](13-verification-and-acceptance.md) | Every gate, in one place |
| [14-the-port-loop-harness.md](14-the-port-loop-harness.md) | The automated driver: how it works, how to run a phase, what it learned the hard way |
| [15-environment-notes.md](15-environment-notes.md) | Toolchain state and the setup traps that silently hollow out hardware gates |

## Status board

Update the status column as phases land. A phase is `done` only when its gate passes — see
[13-verification-and-acceptance.md](13-verification-and-acceptance.md).

| Phase | Gate | Status |
|---|---|---|
| A — AIE2P kernels | Every kernel compiles to `.o` with Peano; compile-only lit passes | **done** 2026-08-04 (18 min) |
| B — runtime seam | Multi-ELF runlist on hardware: numerically identical to sequential, lower latency | **done** 2026-08-04 (362 min) |
| C1 — gate mechanism + small operators | `opcheck.py` and its fault-injection negative control; `causal_mask`, `addnorm`, `layer_norm`, `elementwise_add` pass on hardware | **done** 2026-08-04 (61 min) |
| C2 — `qkv_proj`, `ffn` | Both pass full-output `np.isclose` at registry tolerance vs an FP32 reference | **done** 2026-08-04 (45 min) |
| C3 — `mha_out_proj` | Passes at the registry's FlashAttention tolerance, causal and non-causal | **done** 2026-08-04 (68 min) |
| C4 — coverage sweep | The 36 `baseline_768` shapes resolve through `gemm_config()`; registry rows written; ten shipped models still pass `make verify` | **done** 2026-08-04 (504 min + 66 min re-run) |
| D — block integration | One full transformer layer matches the torch reference on hardware | not started — **next** |
| E — execution strategies | All four modes agree with the reference; dispatch vectors differ as predicted | not started |
| F — study harness | `execution-smoke-test` yields ≥1 `run_status=passed` row per measurement CSV | not started |
| G — unattended runner + CI | Full profile run completes with a complete `results_manifest.json` | not started |
| Goal 1 — sliding window | `make verify` passes with window-crossing prompts | not started |
| Goal 2 — quantization | Second quantized model passes a gate that exercises the quantized path | not started |

Phases A and B were executed by the automated driver — see
[14-the-port-loop-harness.md](14-the-port-loop-harness.md). Both passed their gate, objective
check and tamper check. All ten shipped LLM deployments still pass `make verify` after Phase B's
changes to `llms/shared/infra/cache.py`.

`[2026-08-04]` With one correction: **Phase B's driver-run gate never touched the NPU.** Its
`phase_gate_cmd` was `ninja check-programming-examples-transformer-layer`, and that suite held only
a compile-only test and a host-only test — 2 tests, 16 seconds, per
`agents/.state/port-loop/phase-B/gate.log`. The hardware runlist result recorded in
[05a](05a-phase-b-runlist-spike-result.md) was produced by `make runlist-gate`, which the session
ran and self-reported. `run_npu2_runlist_gate.lit` now puts that gate in the suite; it has been
re-run and all four legs pass, so the claim stands — but it stood on a self-report until then.

Phase C ran as C1–C4 on 2026-08-04, 21 of 40 invocations, ~12 hours wall clock. All four passed
gate, objective and tamper checks. C4 halted once on a driver bug rather than on its own work —
the objective check demanded a registry mtime no honest run could produce — recorded in
[14](14-the-port-loop-harness.md). The registry grew from 33 to 69 bf16-out GEMM shapes with every
pre-existing row byte-identical, and all ten shipped LLM deployments still pass `make verify`.

**One loose end for Phase D.** The three review rounds are the whole review budget, so a finding
raised in round 3 is fixed by round 3's fix session and then *nothing re-reviews it*. C4's round-3
review raised two blocking findings — the `64x768x2304` QKV shape lacked the `fused-cast` row its
builder pins, and the resolution gate checked only each row's winner rather than the method a
builder actually requires. Both were fixed (that shape now carries all three methods, and the
sweep's fused-cast configuration for it replaced one returning zeros for two of nine cast
sub-tiles), and both fixes were verified by hand afterwards — but by the loop's structure, not by
a fourth Codex round.

## Picking this up in a new session

Read [00-context-and-goals.md](00-context-and-goals.md) and
[02-porting-conventions.md](02-porting-conventions.md) first — the conventions document is a hard
requirement, not advice, and ported code is rewritten to MLIR-AIR style rather than transplanted.

Then, before touching anything:

- [15-environment-notes.md](15-environment-notes.md) — the toolchain was four layers stale on
  2026-08-03 and had to be upgraded end to end. Two CMake flags are lost on any clean rebuild and
  silently hollow out every hardware gate if missing. Read this before running a gate.
- [05a-phase-b-runlist-spike-result.md](05a-phase-b-runlist-spike-result.md) — the plan's
  load-bearing assumption, answered. **Do not act on §"The resolution" in
  [05-phase-b-runtime-seam.md](05-phase-b-runtime-seam.md); the mechanism it proposes is wrong.**
- [14-the-port-loop-harness.md](14-the-port-loop-harness.md) — how the automated driver works and
  how to run the next phase through it.

**The next phase is C (operators), split into four sub-phases C1–C4.** Start from
[06-phase-c-operators.md](06-phase-c-operators.md), which is the overview; each sub-phase has its
own specification ([06a](06a-phase-c1-gate-and-small-operators.md),
[06b](06b-phase-c2-qkv-proj-and-ffn.md), [06c](06c-phase-c3-mha-out-proj.md),
[06d](06d-phase-c4-coverage-sweep.md)) because the driver injects one document as a session's
entire task list.

Two decisions taken on 2026-08-04 that amend what those documents used to say:

- **The reference oracles are re-expressed, not ported verbatim.** iron computes them in bf16 at
  `rtol=4e-2` with a 0.5% element mismatch budget; this port uses an FP32 reference, the registry's
  `rtol`/`atol`, and zero mismatches. Details and the two further traps (erf vs tanh GeLU, the
  MHA oracle's precision switch at `seq_len 16384`) are in
  [06 §The numerics standard](06-phase-c-operators.md#the-numerics-standard--do-not-port-irons).
- **Shape coverage is a sweep, not a redesign.** The case matrix needs 108 distinct
  projection-GEMM shapes, not the "several hundred" previously estimated — 5 are registered and
  103 are missing. C4 builds the sweep tool and registers the 36 `baseline_768` shapes Phase D
  needs; the other two families are a later machine-time run.

## Load-bearing questions already answered

| Question | Answer | Where |
|---|---|---|
| Can separately-compiled ELFs share one runlist? | Yes — N ELFs, N `hw_context`s, one runlist. Bit-identical to sequential, 1.02–1.15× faster. **Not** by sharing one context; XRT rejects that three ways. | [05a](05a-phase-b-runlist-spike-result.md) |
| How many concurrent `hw_context`s does NPU2 grant? | 32 (33 fails with `DRM_IOCTL_AMDXDNA_CREATE_HWCTX err=-2`). Phase E's `runlist` mode wants 29 — fits, with three to spare. Caveats on the margin recorded. | [08 §Risks](08-phase-e-execution-strategies.md) |

## Provenance

The source is `iron` commit `1e014c1` "Add transformer-layer execution-strategy studies"
(145 files, ~58.6k insertions), validated there by an 888/888-job suite run. This plan was
reviewed by Codex before approval; findings that materially changed it are marked `[Codex]`
in the phase documents.
