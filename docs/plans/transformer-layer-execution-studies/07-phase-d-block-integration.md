# 07 — Phase D: Single-Block Integration Gate

A short phase with one purpose: prove that one complete transformer layer works through the real
runtime path before four execution strategies are built on top of it.

**This document is the overview. It is not any session's specification.** Phase D runs as two
sub-phases, each with its own spec, gate and objective check:

| Sub-phase | Spec | Covers |
|---|---|---|
| D1 | [07a](07a-phase-d1-operators-at-baseline-768.md) | Every operator validated at the `baseline_768` widths, plus the pre-add `addnorm` variant the block needs and nothing has run |
| D2 | [07b](07b-phase-d2-block-integration.md) | The FP32 golden model, the `encoder_bert` assembly, the per-boundary comparison, and the block gate |

`[2026-08-05]` It is split for the reason Phase C was, recorded in
[14](14-the-port-loop-harness.md): the driver caps an implement session at three hours, and the old
single-phase form asked one session for six work items spanning hardware bring-up on six operators
*and* novel multi-launch integration. Splitting also means a D2 failure does not re-run D1's
hardware time.

## Why this phase exists

`[Codex]` Phase C's per-operator `np.isclose` checks are necessary but not sufficient. They do not
exercise:

- AIR launch argument maps
- layout transitions between operators
- external-kernel linking across a multi-operator sequence
- BO reuse and synchronization under the Phase B allocator
- complete multi-launch layer assembly

Every one of those is a documented source of silent corruption in this repository. Without a
block-level gate, the first place they would surface is inside a four-way comparison, where
attributing a discrepancy to a mode versus to the integration is far harder.

This mirrors the repository's own deployment discipline: `phase-2-single-block-validation` exists as
a distinct gate between per-kernel validation and full-model assembly for exactly this reason.

## What Phase C left you

`[2026-08-04]` All of this exists and is gated on real hardware. Do not rebuild any of it; the
example's own `programming_examples/transformer_layer/README.md` is the authoritative file-by-file
inventory.

| Piece | Where |
|---|---|
| Operator builders | `transformer_layer/builders/` — `elementwise_add` (with the `causal_mask=` keyword), `layer_norm`, `addnorm`, `qkv_proj`, `gelu`, `ffn`, `mha_attention`, `o_proj`, `mha_out_proj`, plus `gemm_spec.py` |
| Numerical check | `transformer_layer/opcheck.py` — the CLI, the results artifact, and the fault-injection negative control |
| Registry sweep | `transformer_layer/sweep/` — `registry_sweep.py`, `registry_writer.py`, and their host-only tests |
| Per-operator gates | one `run_npu2_<op>_peano.lit` each, plus `run_npu2_fault_control_peano.lit` |
| Runtime seam | `runlist_gate.py` — `KernelCache.run_sequence` over a multi-artifact runlist, with `leg_c_run_sequence` as the working template |

**Reuse `opcheck.py`. Do not write a second checker.** It already owns the contract the driver's
objective check depends on. Phase D adds shapes and a `block` entry to it rather than standing up a
parallel mechanism.

## The three things that decide how the phase starts

Read these before planning anything; each is developed in the sub-phase spec that owns it.

**The family is forced, and so is the sequence length.** `registry_lookup.gemm_config()` raises on
an unmeasured shape, so a family is usable only if its projection GEMMs are registered. After C4
that is `baseline_768` alone (36 of 36, against 2 and 3 for `tinybert_512` and `baseline_1024`).
Within it, `seq = 4096` is the only point where `build_ffn_module` builds, because the up- and
down-projections collide on `f32_to_bf16_mn_<suffix>` everywhere else. Details and the table in
[07a](07a-phase-d1-operators-at-baseline-768.md#the-family-and-why-the-sequence-length-is-forced).

**The operators are not all validated at that family.** Phase C validated each at whatever width
was cheapest to bring up; only `qkv_proj` has a point at `hidden = 768`. The GEMM *tiles* for the
other shapes are registered, which is a different claim from the operator computing the right answer
at that width. That is what D1 is.

**One of them is the wrong operator.** The validated `addnorm` computes
`LayerNorm(x) * weight + residual`; `encoder_bert` needs `LayerNorm(x + residual) * weight`. The
kernel supports both behind `-DADDNORM_PRE_ADD` and Phase A already compiles the pre-add object, but
no builder exposes it and it has never been dispatched. See
[07a §The pre-add gap](07a-phase-d1-operators-at-baseline-768.md#the-pre-add-gap-the-validated-addnorm-is-not-the-one-the-block-needs).

## The golden model needs the same correction Phase C made

`pattern/reference.py` (172 lines, pure torch) is the correctness anchor for this phase *and* all of
Phase E. An earlier version of this document said to port it **verbatim**. Do not — it builds every
tensor in bf16, and chained over eight GEMMs that is worse than it was per-operator. Port the
structure; compute in FP32 from bf16-rounded inputs. The full treatment, including the RNG draw
order that must survive the re-expression, is in
[07b](07b-phase-d2-block-integration.md#the-golden-model-port-the-structure-not-the-numerics).

## Gate

One full transformer layer matches the torch golden model end-to-end on real hardware, via a
`run_npu2_block_peano.lit` in the existing `check-programming-examples-transformer-layer` suite.

If the element-wise comparison fails, the per-boundary intermediates identify which stage diverged;
do not proceed to Phase E on a layer that only approximately matches.

## Risks

- This phase has no new device code, so a failure here means something in Phase B or C is wrong in
  a way its own gate did not catch. Budget time for iterating back into those phases rather than
  treating this as a formality.
- D1 is real hardware time on six operator points before the block runs at all. It is still cheaper
  than debugging a chained layer whose stages were never individually checked at that width.
