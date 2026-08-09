# 25 — The first study result: the `baseline_768` sequence ladder

> ## `[retracted 2026-08-08, and now unreproducible as of 2026-08-09]`
>
> **Do not cite the crossover or the slopes.** Two separate things invalidate this document, and the
> second one is permanent:
>
> 1. **It ranks four implementations, not the four modes.** The taxonomy was corrected on
>    2026-08-08 ([03 §The taxonomy](03-measurement-model.md)) and none of the four things measured
>    here matched its corrected definition.
> 2. **Its explanation cannot be re-tested.** The finding was that the slopes split on ATTENTION
>    PLACEMENT — host-attention modes at 1.23–1.27 against device-attention modes at 1.03–1.17 —
>    rather than on dispatch structure. As of 2026-08-09 **all four modes run attention on the
>    device**, so there is no host-attention mode left to produce one side of that split. A rerun
>    showing separated slopes is measuring something else and needs a new explanation.
>
> **What survives:** the measurement itself. 16 rungs, walked twice on hardware, every rung
> validated, and the 1024 ordering correctly recorded as indistinguishable because it did not
> survive the second walk. Read it as a record of what four *implementations* did on 2026-08-08, and
> as the reason `attention_path` became a recorded per-row covariate at all.
>
> The one clean cross-mode number the study now has is DRAM traffic at 4096 — `runlist` 190,513,152
> bytes against `offload` 970,457,088 — which differs in the taxonomy's own variable and nothing
> else. See [03](03-measurement-model.md).

`[2026-08-08]` Four execution modes across four sequence lengths on NPU2 hardware, 16 of 16 rungs
passing. This is J3, and it is the first output of this project that is a *result* rather than a
capability: every earlier measurement established that something works or how it is structured.

**What makes it a result is the crossover.** Doc [16](16-compiler-work-and-remaining-essence.md)
states the reason J3 exists: "a tradeoff analysis at a single shape has no curves and therefore no
crossover — which is the result the study exists to produce." There is one, and it is not the pair
anyone would have picked.

## How it was measured

`study/run_ladder.py`, one **child process per rung** (see the rule in
[23](23-rules-and-open-items.md) — this is not optional and the first attempt without it produced
five false failures), `--samples 3 --warmup 1`, `baseline_768` encoder shape
(`emb 768`, `ffn 3072`, `12 heads × 64`), bf16, under `/tmp/mlir-air-npu.lock`, on an otherwise
quiet host. Compilation is outside the clock. Every rung validates against the FP32 golden
reference with zero mismatches; a rung that failed validation would appear as `FAILED` and none did.

**It was measured twice**, as two independent 16-rung walks, because a single walk cannot tell a
finding from run-to-run spread. Artifacts (gitignored):
`results/j3_ladder_iso2/` is authoritative — it is the run whose rows carry `attention_path`, added
after the first walk showed the field was never populated — and `results/j3_ladder_iso/` is the
independent replicate. Each holds one CSV per mode and one schema-v1 row per rung, plus `report.md`
from `study/ladder_report.py`.

## Latency, both walks

| mode | 512 | 1024 | 2048 | 4096 | slope | attention |
|---|---|---|---|---|---|---|
| `fused` | **46.7 / 45.0** | 97.7 / 99.0 | **197.6 / 195.6** | 524.9 / 536.2 | 1.15 / 1.17 | device |
| `coarse` | 53.0 / 48.6 | 106.7 / 98.6 | 204.5 / 214.6 | **465.1 / 455.2** | **1.03 / 1.08** | device |
| `offload` | 57.2 / 56.0 | 117.3 / 124.3 | 274.4 / 273.9 | 782.1 / 813.6 | 1.26 / 1.27 | host |
| `runlist` | 59.8 / 61.7 | 130.3 / 136.9 | 303.8 / 292.8 | 811.2 / 819.3 | 1.25 / 1.23 | host |

Run-to-run spread is **0.2 % to 9.0 %**, worst on `coarse`. Any claim below has to clear that bar,
and one of the claims in the first draft of this document did not — see the 1024 column.

The **structural columns are bit-identical across both walks** for all four modes — every
submission, herd-launch, sync-boundary and byte count. Counts are deterministic; only durations
move. That is also why the distinguishability verification rests on the counts.

Slope is the least-squares fit of `log(latency)` against `log(seq_len)` over the four rungs. Read it
as "closer to linear" or "closer to quadratic", not as a model: four points over one decade cannot
separate `n²` from `n log n`, and a fixed per-launch cost pulls every slope toward 1 at short
lengths.

## The two findings

**1. `fused` leads below 4096 and `coarse` leads at 4096.** Both walks agree at the three lengths
that carry the claim:

| | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|
| walk 1 | `fused` by 13.4 % | `fused` by 9.2 % | `fused` by 3.5 % | `coarse` by 12.9 % |
| walk 2 | `fused` by 7.9 % | **`coarse` by 0.4 %** | `fused` by 9.7 % | `coarse` by 17.8 % |

**At 1024 the two walks disagree**, and walk 2's margin there is 0.4 % — an order of magnitude
inside the spread. So the honest statement is that `fused` leads at 512 and 2048, `coarse` leads at
4096, and **at 1024 the two modes are indistinguishable at this sample count.** The first draft of
this document claimed `fused` won 512, 1024 and 2048, which is what one walk showed and what a
second walk refuted.

The crossover itself survives that: it sits between 2048 and 4096, and it is the largest and most
reproducible effect in the table (12.9 % and 17.8 %, same direction).

`fused` is the single-submission extreme — one host submission, 23 herd launches and 12 sync
boundaries at *every* length — so it pays the least synchronization and leads wherever
synchronization dominates. `coarse` pays 4 submissions and a sync count that grows with length
(59 → 396), and still overtakes `fused` at 4096. **A design chosen on the 4096 measurement alone
picks `coarse`; chosen at 2048 it picks `fused`.** That is the tradeoff this study exists to expose,
and no single shape can show it.

**2. The slopes split exactly on attention placement, not on dispatch structure.** Across both
walks the device-attention modes fit **1.03–1.17** and the host-attention modes **1.23–1.27**, with
no overlap and the same ordering each time. That grouping
crosses the dispatch taxonomy completely — `coarse` and `fused` sit at opposite ends of it (4
submissions versus 1), as do `offload` and `runlist` (6 versus 5, and 19 herd launches versus 404).
So at this shape range, *where attention runs* predicts how a mode scales and *how the layer is
dispatched* does not.

That is a measured argument for J2 being on the critical path for the study's conclusions rather
than a tidiness item: while attention placement varies across modes, it is the dominant term in the
comparison, and every latency ranking here is partly a ranking of attention placement.

## What this result does not support

- **It is not a claim about NPU efficiency.** `offload` and `runlist` run attention in host torch,
  and the clock covers host-side dispatch, so a share of their latency — plausibly most of their
  extra slope — is CPU work. Separating that needs the `Profiler` instrumentation described in
  [09](09-phase-f-study-harness.md); `pattern/` records no timing today.
- **Three samples is not a distribution**, and two walks are not an error bar. They are enough to
  reject one claim (the 1024 ordering) and to establish that spread is under ~9 %, which is what the
  surviving claims are stated against. A difference smaller than that needs more samples, not more
  interpretation.
- **`fused` is not "the fastest mode".** It leads below the crossover at this shape, on this device,
  with attention on device, at bf16 — and not detectably at 1024.
- **No power or energy claim is available at all.** No sensor on this platform measures the NPU; see
  [09](09-phase-f-study-harness.md).

## Structure, which is what the gate reads

Counts, not durations, so these are unaffected by host load — and they are what the four
distinguishability clauses in `phase_e_checks.py` assert.

| mode | subs | herd 512 → 4096 | sync 512 → 4096 |
|---|---|---|---|
| `coarse` | 4 | 33 → 146 | 59 → 396 |
| `offload` | 6 | 18 → 19 | 18 → 19 |
| `runlist` | 5 | 67 → 404 | 58 → 395 |
| `fused` | 1 | 23 → 24 | 12 → 13 |

`offload` and `fused` hold their launch and sync counts nearly flat across an 8× sequence range,
while `coarse` and `runlist` scale theirs with length. Both flat modes reach that flatness for
opposite reasons — `offload` by moving work to the host, `fused` by fusing it on the device — which
is why a dispatch-structure column alone cannot tell them apart and `attention_path` has to be
recorded beside it.

## Reproducing it

```
cd programming_examples/transformer_layer
flock -x -w 7200 /tmp/mlir-air-npu.lock python3 study/run_ladder.py \
    --modes coarse,offload,runlist,fused --seqs 512,1024,2048,4096 \
    --out-dir results/j3_ladder_iso2 --study-id j3-ladder --samples 3
python3 study/ladder_report.py results/j3_ladder_iso2 --md results/j3_ladder_iso2/report.md
```

Warm ELF caches make a full 16-rung walk about 90 seconds of device time; cold, budget ~45 minutes,
almost all of it compilation. **Nothing CPU-heavy may run alongside it** — that inflated an earlier
table by 1.55×.
