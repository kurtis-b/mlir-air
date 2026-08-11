# 27 — The first four-mode comparison at one sequence length

> **`[2026-08-10]` Two of this document's results are superseded — read
> [32](32-cost-decomposed-ladder.md) beside it.** The warm DRAM ordering it
> measured (`fused` < `coarse` < `runlist` < `offload`) described the
> wholesale-eviction implementation: the targeted pool eviction landed
> 2026-08-10 moved `runlist` below `fused` and `coarse` by exactly its
> static-weight set, so the warm ordering is now
> `runlist` < `fused` < `coarse` < `offload` (byte-identical across two
> walks). And its latencies were measured on a machine whose `hw_context`
> load cost has since risen ~30× — they remain the healthy-machine record,
> but no comparison against post-2026-08-09 numbers is valid until [32]'s
> verdict rung reads healthy again. The byte totals for `fused`, `coarse`
> and `offload`, the two-walk discipline, and the cold-vs-warm conventions
> all stand.

`[2026-08-09]` The four modes had never been measured at the same sequence
length. `fused` sat at 1024 and the other three at 4096, so every cross-mode
table assembled from the catalogue compared two lengths — the first of the three
traps [README §Where things stand](README.md) lists, and the largest single
thing it says is outstanding.

This is that comparison. **Two rungs, 512 and 1024, all four modes, walked
twice.**

## What the walks are

Two independent walks under one set of conditions, each
`study/run_ladder.py --modes coarse,offload,runlist,fused --seqs 512,1024`
with `--warmup 2 --samples 5`, one process per rung, submitted
`devq.sh --class measure` so no build ran beside a timed region. 8/8 rungs
passed in both. Artifacts: `results/common-ladder-w1/`, `results/common-ladder-w2/`
(gitignored; the CSVs are schema v1).

Every ELF was compiled BEFORE the walks, as separate `--class build` jobs, so
no aircc ran inside a device reservation. Compilation is outside `run_mode.py`'s
clock either way; this only keeps a CPU-bound compile out of the measurement's
host.

## The result that survives both walks

**DRAM traffic — the corrected taxonomy's own variable.** These are counts, not
timings: **byte-identical between the two walks**, which is what a dispatch
count should be.

| mode | bytes @ 512 | bytes @ 1024 |
|---|---|---|
| `fused` | 21,233,664 | 42,467,328 |
| `coarse` | 22,020,096 | 44,040,192 |
| `runlist` | 34,799,616 | 55,246,848 |
| `offload` | 44,040,192 | 99,090,432 |

**The ordering is exactly what [03 §The taxonomy](03-measurement-model.md)
predicts**, at both lengths, with no confound left over: `fused` eliminates
inter-operator DRAM traffic and moves the least; `offload` puts every non-linear
operator on the host and moves the most; `runlist` and `coarse` sit between them.
This is the first time that prediction has been checked against all four modes
under one set of conditions.

**`offload` against `runlist` is 1.79× at 1024, where at 4096 it is 5.1×.** Not
a contradiction — the mode's extra traffic is the softmax round trip, and a
score matrix is O(seq²) while the linear operators are O(seq). Quartering the
sequence length cuts the attention term 16× and the rest 4×, so the ratio has to
shrink. The 4096 decomposition in the README (33× on the attention component)
and this 1.79× total are the same effect measured at two points on its curve.

**Latency, and it is more equivocal than the byte table.** Comparing minimums as
well as averages, because [23 §1](23-rules-and-open-items.md) settled that host
jitter flatters medians here:

| mode | avg @512 (w1/w2) | min @512 | avg @1024 (w1/w2) | min @1024 |
|---|---|---|---|---|
| `fused` | 44.7 / 43.7 | 43.5 / 42.8 | 93.7 / 92.9 | 89.5 / 90.0 |
| `coarse` | 47.6 / 47.3 | 45.3 / 45.7 | 103.0 / 109.4 | 101.5 / 106.0 |
| `runlist` | 89.9 / 89.9 | 82.6 / 84.9 | 161.7 / 159.6 | 160.4 / 157.9 |
| `offload` | 84.3 / **111.4** | 78.2 / 79.9 | 184.0 / **226.4** | 147.9 / **184.9** |

What survives both walks on both statistics:

1. **`fused` is fastest at both lengths.** 1 submission, 13 sync boundaries.
2. **`coarse` is second at both lengths.**
3. **`runlist` and `offload` are both far slower, and INDISTINGUISHABLE from each
   other.** On averages `runlist` wins at 1024 in both walks; on minimums that
   flips between walks. Neither statistic settles the pair at either length.

## The crossover that did not survive, and why this document exists

Walk 1 produced a clean crossover, and `ladder_report.py` reported it:

> offload leads runlist at seq 512 (84.3 vs 89.9 ms) and trails it at seq 1024
> (184.0 vs 161.7 ms)

Walk 2, same code, same conditions, reports **`none: the ranking is the same at
every rung`**. The crossover was `offload`'s run-to-run noise, not a curve.

**Had one walk been run, this document would have published a crossover.** It is
the same failure J3 recorded — [25](25-first-study-result-sequence-ladder.md)'s
1024 ordering also failed to survive a second walk — and the same failure the
README's own lesson names twice. The two-walk rule is what caught it, and
nothing else would have: walk 1 is internally consistent, 8/8 passed, and the
crossover is exactly the shape a real result takes.

## `offload`'s noise, measured fresh

`offload` is not merely noisy, it is noisy enough to invert a ranking. Intra-walk
spread, `(max-min)/min` over five samples:

| mode | @512 w1 / w2 | @1024 w1 / w2 |
|---|---|---|
| `fused` | 7.8% / 6.0% | 8.7% / 7.3% |
| `coarse` | 10.4% / 7.7% | 2.7% / 5.9% |
| `runlist` | 31.5% / 20.9% | 1.8% / 4.6% |
| `offload` | 20.3% / **120.7%** | **61.6% / 59.8%** |

This **corroborates [03 §Run-to-run comparison](03-measurement-model.md)**, which
gives `offload` a 20% median / 35% p90 band against 5% / 15% for the others and
says it drifts roughly ten times as much. That table was inherited from iron;
this is an independent measurement on this port agreeing with it. `offload`'s
walk-to-walk average moved 32% at 512 and 23% at 1024 — at or past even its own
widened band.

**A likely mechanism, and it is now half-tested.** `offload` is the only mode
that unloads and reloads its `hw_context` on every dispatch
(`pattern/offload/offload.py:465`, `_evict_context`), 30 times per layer. Context
teardown and setup is host- and driver-side work, which is precisely the part of
the clock that host conditions perturb — so this mode alone carrying this
variance is what that mechanism predicts.

`[2026-08-09]` What has since been established is that **the 30 reloads are
removable**, which makes the hypothesis testable rather than merely plausible.
`agents/probes/probe_context_reuse.py` puts the corruption `_evict_context`
exists for in exactly one cell of a 2×2: `elf`+`[2,2]` diverges from its own
first run by 3.8141e-01, while `elf`+`[1,1]`, `xclbin`+`[2,2]` and
`xclbin`+`[1,1]` are all bit-identical over four runs. The docstring attributes
it to "these runtime-tiled GEMM ELFs" as a class; it is the ABI and the tiling
**together**.

**The measurement that would settle it** is this same ladder re-walked with
`offload` holding one context — if the variance collapses toward the other three
modes, the eviction was the cause. That is a rung of the N-streams work rather
than a separate experiment, and it should be taken before anyone attributes
`offload`'s noise to the mode's partition.

> **`[2026-08-09]` TAKEN. The variance collapses — but the hypothesis is
> SUPPORTED, not confirmed.** Four walks, `{ELF, shared} × {w1, w2}`,
> interleaved, with `runlist` inside each as a same-conditions control. At 512
> the intra-walk spread goes **316.9% / 134.1% to 17.6% / 14.0%**, in both
> walks, while the control stayed in band.
>
> **The intervention is not single-variable, which this paragraph asked for and
> the measurement could not deliver.** Switching to the shared xclbin stops the
> per-dispatch reconfiguration *and* changes the ABI from ELF to xclbin. The
> control rules out environmental drift, not the ABI. Eviction is still the
> leading candidate; isolating it needs a third arm — the xclbin ABI with
> eviction forced back on — and that knob does not exist yet.
> [29](29-offload-n-streams.md) has the table and the experiment.
>
> Two qualifications this document's own numbers need. **The 1024 rung did not
> reproduce**: the 61.6% / 59.8% in the table above read 9.0% / 10.5% on the
> same ELF path today, so the baseline is unstable day to day and the effect
> *size* should not be quoted from one measurement. And the shared path costs
> **~20% on best-case latency at 512** (97.5–99.5 ms against 78.9–82.0), so it
> is a trade rather than a free win. The minimums here *do* reproduce this
> document's — 82.0 / 78.9 against its 78.2 / 79.9 — which is what says the two
> measurements are of the same thing. Full table: [29](29-offload-n-streams.md).

## What this does and does not settle

**Settles trap 1.** There is now a cross-mode comparison at one set of lengths.
Any table built from the SPECS rows still spans two lengths, because those rows
are unchanged — `fused` at 1024, the rest at 4096 — so build cross-mode tables
from a ladder run, not from the catalogue.

**Does not settle the `coarse` row's meaning.** `coarse` here is the
**uncorrected D2 block**, not a corrected blend: it is second-fastest and
near-`fused` on bytes, and that is a fact about the D2 block, not about the
`coarse` point of the taxonomy. When the corrected `coarse` lands, this whole
table is re-walked — not just its row. [23 §One process per device
measurement](23-rules-and-open-items.md) forbids assembling a comparison from
runs taken under several conditions, and that applies to a partial re-run.

**Does not produce a scaling exponent.** Two rungs. `ladder_report.py` declines
to fit below three passing rungs and is right to.

## Why the ladder is 512 and 1024, measured rather than chosen

| rung | `coarse` | `offload` | `runlist` | `fused` |
|---|---|---|---|---|
| 256 | builds | **fails** | **fails** | builds |
| 512 | builds | builds | builds | builds |
| 1024 | builds | builds | builds | builds |
| 2048+ | builds | builds | builds | **out of range** |

**Above 1024** is `fused`'s bound, and it is arithmetic: `plane_major` packing
needs a plane stride of `rows*cols` against the shim `aie.dma_bd` cap of
1,048,576, so it caps at 1365 rows. 1024×768 = 786,432 fits; 2048×768 =
1,572,864 does not.

**256 fails** for `offload` and `runlist` — the two modes with device attention
GEMMs, and they fail identically because `runlist` imports the tiles from
`offload.ATTENTION_GEMM_TILES` rather than copying them. `attn_scores` at 256 is
`256x64x256`, and the module builder asserts
`n % (tile_n * herd_n) == 0` → `256 % (128 * 4) = 256 % 512 ≠ 0`,
`AssertionError: (256, 128, 4)`. The tiles are doc 26 Spike B's, measured at
4096, and they are legal at 512 and 1024 because both divide by 512.

**This is a tile constraint, not a hardware one** — the same distinction doc 26
draws about the K=64/N=64 rows. A legal 256 tile exists (`tile_n=64, herd_n=4`
divides 256), but substituting an unmeasured tile into a comparison run would be
the "recorded claim with no artifact" failure the README warns about. Retuning
and validating a 256 tile is a separate, honest piece of work, and it is what a
third rung — and therefore a scaling exponent — costs.

## A methodological note worth more than the rung it cost

The feasibility check for these lengths was run first at the wrong altitude. It
resolved every mode's **config** at 256/512/1024 and reported all four resolving
at all three. Config resolution answers *"do the specs resolve"*; the 256 failure
is raised by the module builder's own assert, one level below that, and only a
build surfaces it.

[23 §Match a probe's altitude to its claim](23-rules-and-open-items.md) says
exactly this about `air-opt` versus `aircc`. It cost ten minutes here rather than
a spec claim, because the build ran immediately afterwards — but the general form
recurred, and the guard is the same: **let the thing that will run in anger be
the thing that answers the feasibility question.**

## Numbers that moved from their 4096 records

Three structural counts differ at 1024 and a reader carrying the 4096 figures
across will be wrong:

- **`coarse` bands its norms 16 ways per normalization point at 1024, not 64**
  (the L1 cap is 104 rows at this width). `fused.py:58-61` explains `coarse`'s
  402 sync boundaries as 64 bands per point; here `coarse` crosses **107** at
  1024 and **59** at 512.
- **`runlist` is 139 entries over 17 runlists at 1024**, against 427 over 17 at
  4096. The runlist count is invariant and the entry count is not, which is the
  mode's granularity claim behaving as designed.
- **`offload` is 30 dispatches at every length** — 24 attention + 6 projections.
  The head count sets it, not the sequence length.
