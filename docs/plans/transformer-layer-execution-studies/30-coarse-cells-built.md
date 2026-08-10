# 30 — `coarse`'s two interior cells, built and gated

`[2026-08-09]` [28](28-coarse-blend-space.md) scoped `coarse`'s blend space and stopped there: two
axes, six cells, four of them already owned by an existing mode, and the two interior ones — **C2**
(`block` front, decomposed tail) and **C3** (`runlist` front, banded tail) — specified but not
built. This document records building them.

It is a **landing and a structural result**, not the mode's decision. The measurement that selects
`coarse`'s cell is the ladder in §The ladder below.

## What landed

| | |
|---|---|
| `pattern/coarse/cells.py` | The blend space as data (all six cells, four naming their owner), the two-half config, the composed dispatch, and the shared preparer |
| `pattern/coarse_c2/`, `pattern/coarse_c3/` | One thin preparer and one ELF cache each, on `pattern/coarse/coarse.py`'s model |
| `coarse_cells_structure.py` | The host-only structural arm: what each cell WILL dispatch, derived from the configs |
| `run_npu2_coarse_c2_peano.lit`, `..._c3_peano.lit` | Clean + negative control + the structure arm, per cell |
| `opcheck_specs.py` | Two catalogue rows at `4096x768_encoder_bert`, `atol` at the `1e-1` ceiling |

**`coarse` itself did not move.** C1 = `(block, banded)` is still `pattern/coarse/coarse.py` over
`builders/block.py`, and `cells.py` *refuses* to build it — a second implementation would measure
something D2 never validated. The same refusal covers `fused` and `runlist`, which are two more
cells of the same space.

**The composition calls; it does not copy.** `builders/block.py`'s `_sequence_a` / `_sequence_norm`
/ `_sequence_ffn` were already module-level. `pattern/runlist/runlist.py`'s six dispatch regions were
nested closures inside `prepare_runlist` and are now module-level on the same `(cache, cfg, ...)`
convention — the mechanical step [28 §What each new cell costs](28-coarse-blend-space.md) asked for,
and the *only* extraction taken.

> **The `fused.py` extraction the README's item 1 also asked for was NOT done, deliberately.** No
> cell at seq ≥ 2048 uses a stitched tail, because no stitched tail *builds* there — that is the
> premise of the whole phase. Extracting `prepare_fused`'s dispatch closure would churn a gated mode
> for a composition nothing calls. `fused` is untouched and its recipe stayed green as an unmodified
> control.

## The extraction is inert, and that is measured

`run_npu2_runlist_peano.lit` reproduces its pinned totals **byte for byte** on both the clean and
the fault-injected half:

```
submissions 17 entries 427 air 50 herd 488 sync 451 bytes 190513152
```

That equality is the no-behaviour-change proof; nothing weaker substitutes for it, because the
extraction touched the one file the mode's whole dispatch lives in.

## Both cells passed their first hardware run

`4096x768_encoder_bert`, 10/10 stage boundaries clean, negative controls failing as required, summed
dispatch totals equal between the clean and injected runs.

| cell | front | tail | submissions | entries | air | herd | sync | bytes |
|---|---|---|---|---|---|---|---|---|
| C1 `coarse` | block | banded | 4 | 131 | 12 | 146 | 402 | 202,902,528 |
| **C3** | runlist | banded | 17 | **169** | 46 | 232 | 451 | 190,319,616 |
| **C2** | block | decomposed | 4 | **389** | 16 | 402 | 402 | 203,096,064 |
| C6 `runlist` | runlist | decomposed | 17 | 427 | 50 | 488 | 451 | 190,513,152 |

**The entry counts were predicted before the cells ran**, host-side, and hardware confirmed both.
`coarse_cells_structure.py` composes each half's contribution independently — front `block` 1/2,
front `runlist` `2+heads` / `4+3·heads`, tail banded 3 / `1+2·blocks`, tail decomposed 3 /
`3+6·blocks` — and **two of the four combinations were already pinned by shipped gates**, so the
model reproduces `coarse`'s 4/131 and `runlist`'s 17/427 from the same arithmetic that predicts
the interior cells. A model that recovers both endpoints is a model rather than a guess.

The ordinal claim the pair owns holds: **131 < 169 < 389 < 427**. Each cell refines exactly one half
of C1 and neither refines both, so each must land strictly between the incumbent and the fully
decomposed mode.

### A free provenance check: the four vectors are ADDITIVE

Nobody designed this check; the arithmetic offered it. On **every one of the six columns**,
`C1 + C6` equals `C2 + C3`:

| column | C1 + C6 | C2 + C3 |
|---|---|---|
| submissions | 21 | 21 |
| entries | 558 | 558 |
| air launches | 62 | 62 |
| herd launches | 634 | 634 |
| sync boundaries | 853 | 853 |
| bytes | **393,415,680** | **393,415,680** |

The two diagonals of a 2×2 factorial sum identically exactly when each cell's cost is its front's
plus its tail's, with no interaction term. A cell that had quietly forked a half, or composed
something other than one mode's front with the other's tail, would not sum — so this says the
composition is what it claims **independently of anyone's account of it**. It is the same shape of
evidence as `runlist`'s byte-identical non-attention total after its rebuild.

Solving the system gives the halves directly, and two of the numbers are worth reading rather than
only checking:

- **The `runlist` front moves 12,582,912 bytes FEWER than the `block` front** — exactly
  `2 × [4096, 768]` bf16, to the byte.
- **The decomposed tail costs 193,536 bytes more than the banded one**, on a layer that moves
  ~190 MB. The tail axis is nearly free in DRAM traffic; the front axis is where the bytes are.

## The error tracks the TAIL, not the front

All four corners at `atol = 1e-1`, over 3,145,728 elements, zero mismatches:

| cell | tail | `mean_rel_L1` | `atol_required` | margin |
|---|---|---|---|---|
| C3 | banded | **1.654e-2** | 7.266e-2 | 1.38× |
| C1 `coarse` | banded | 1.688e-2 | 7.398e-2 | 1.35× |
| C6 `runlist` | decomposed | 1.746e-2 | 6.981e-2 | 1.43× |
| **C2** | decomposed | **1.784e-2** | 7.896e-2 | **1.27×** |

Sorted by `mean_rel_L1`, the two banded-tail cells come first and the two decomposed-tail cells
second, regardless of front. A decomposed tail stages bf16 between the add, the LayerNorm and the
gamma multiply where the fused `addnorm` keeps all three in one kernel — the same effect `fused`'s
catalogue row records about its own tail. **No tolerance was widened for either cell.**

C2's 1.27× is the least headroom any whole-layer row has carried. It is a *recorded* fact, not a
problem to be solved by moving the ceiling: if a future change pushes it past `1e-1`, that is a
defect report.

## The ladder — four corners, two lengths, walked twice

`--warmup 2 --samples 5`, one process per rung, every ELF pre-built by a build-class warm pass so no
aircc ran inside a device reservation. 8/8 rungs passed in each walk.

| cell | front | tail | 2048 avg w1 / w2 | 2048 min w1 / w2 | 4096 avg w1 / w2 | 4096 min w1 / w2 |
|---|---|---|---|---|---|---|
| **C1** `coarse` | block | banded | **215.6 / 216.2** | **204.0 / 200.6** | **479.0 / 459.8** | **455.2 / 444.7** |
| C2 | block | decomposed | 238.4 / 220.5 | 225.6 / 206.6 | 490.6 / 509.8 | 485.3 / 488.1 |
| C3 | runlist | banded | 309.0 / 309.9 | 304.5 / 296.8 | 748.2 / 764.8 | 724.1 / 745.4 |
| C6 `runlist` | runlist | decomposed | 333.3 / 322.0 | 327.5 / 311.8 | 805.6 / 858.6 | 779.0 / 805.9 |

**The ordering C1 < C2 < C3 < C6 survives both walks, on averages AND on minimums, at both
lengths.** That is the thing a single walk cannot establish, and it is why this was walked twice.

### The two axes do not resolve equally, and the difference matters

**The FRONT axis is unambiguous and large.** A block front runs ~1.5–1.6× faster than a runlist
front at both lengths, with no overlap between the two groups on any statistic in either walk. That
separation is far outside the intra-walk spread.

**The TAIL axis is clean only at 4096.** There, C1's slowest minimum (455.2) sits below C2's fastest
(485.3) and the averages do not overlap either, so banded < decomposed is real. At **2048 it is
not resolved**: the C1/C2 average gap is 10.6% in walk 1 and 2.0% in walk 2, against intra-walk
spreads of 15–18%, and C2's own average moved −7.5% between walks. The minimums keep C1 ahead in
both walks, but the honest reading at 2048 is **C1 ≤ C2, not cleanly separated**. Do not quote a
2048 tail effect.

**A band to state rather than assume.** Intra-walk spread `(max−min)/min` here runs 1.8–20.3%,
wider than the 2–10% [27](27-common-ladder-result.md) recorded for the non-`offload` modes at
512/1024. These are longer rungs; nothing suggests a new mechanism, but the two bands are not
interchangeable and a future comparison should not treat 27's as this one's baseline.

### DRAM traffic, and why these bytes are not the gate's bytes

Byte totals were **identical across the two walks** for all four cells. They are *not* the numbers
in the gate table above, and the difference is a property of the measurement rather than a
discrepancy: `run_mode` records the vector of the **last of five samples**, after two warmups, so
static weights are already resident; the gate records a **cold first dispatch**.

| cell | cold (gate) | warm (ladder) | drop |
|---|---|---|---|
| C1 | 202,902,528 | 188,743,680 | 14,158,848 |
| C2 | 203,096,064 | 188,743,680 | 14,352,384 |
| C3 | 190,319,616 | 190,319,616 | **0** |
| C6 | 190,513,152 | 190,513,152 | **0** |

Every number here is derivable to the byte, which is what says the instrumentation is measuring
what it claims:

- **The cold C2−C1 gap, 193,536, is exactly the gamma broadcast**:
  `2 × (64 × 768 × 2) − 2 × (768 × 2)`. The banded tail's `addnorm` takes the `[emb]` weight; the
  decomposed tail's `elementwise_mul` takes a materialized `[norm_rows, emb]` band. That is the
  *whole* cold cost of decomposing the tail.
- **C1's drop, 14,158,848, is exactly its static-weight set**: `w_qkv + w_o + w_up + w_down` at
  4,718,592 each plus the two `[emb]` gammas at 3,072. C2's drop is the same plus its larger
  gammas. So a warm run skips precisely the static uploads and nothing else.
- **The runlist-front cells drop ZERO**, and there is a mechanism in the code:
  `evict_attention_contexts` clears `cache._pools` — *all* pools, not only the two attention
  artifacts' — once per head, so the content-keyed static-weight pool never survives a dispatch and
  every weight is re-uploaded every time.

**That last one is a real cost and it is NOT the reason those cells are slower.** It is ~14 MB
against ~190 MB, ~7% of traffic, while the latency gap is ~60%. Recorded because it is a
removable inefficiency somebody will want, not because it explains the ranking.

> **`[2026-08-10]` REMOVED — the eviction is targeted now.** `evict_attention_contexts` drops
> only the pools whose sequences involve the two attention artifacts
> (`KernelCache.evict_pools_for`, reading the kernel names out of the plan signature via
> `signature_kernels`), and the content-keyed static-weight pools survive. The measured footgun
> it exists for is context state, not BO state — pooled ELF-ABI BOs are `xrt.ext.bo`,
> device-level, and survive a context unload — so nothing safety-shaped was riding on the
> wholesale clear. Measured after the fix: a warm `runlist` run reads `sync 443
> bytes 176160768` against the cold 451 / 190,513,152 — a drop of exactly **14,352,384** bytes,
> the same static set C2's drop decomposes into, across 8 skipped uploads. **The cold totals
> did not move**, because within one cold layer run the per-head evictions only ever destroyed
> pools that run never reused — so every pinned gate literal stands unchanged. `check-runlist`,
> its fault twin and `check-coarse-c3` re-ran green, dispatch host tests 32/32 (one new,
> verified in the failing direction first).

## What `coarse` is, with provenance

**`coarse` = C1 = (block front, banded tail).** That is what it already was — and the point of this
phase is that it is now *chosen* rather than inherited from D2 having been built at 4096. The
selection is recorded in the mode's artifact (`blend_cell`, `blend_front`, `blend_tail`,
`blend_selected_by`) instead of being implicit in which file the mode happens to call.

**The taxonomy does not collapse, and that was the live risk.** [28](28-coarse-blend-space.md)
warned that a space containing its own endpoints would re-derive one of them: on
[27](27-common-ladder-result.md)'s evidence, measured at 1024, every cell is dominated by one that
already has a mode name and the winner would have been `fused`. Measured at 2048 and 4096 — where
no stitched tail builds, which is the whole reason those are the lengths — **the winner is an
interior cell.** `coarse` is distinct from `runlist` (C6, slowest of the four) and distinct from
`fused` (unbuildable here). Four points, four things.

**What the result does not say.** It does not say a block front is better in general; it says that
at these two lengths, on this workload, the front axis dominates the tail axis by roughly an order
of magnitude in effect size. And it does not rank `coarse` against `fused`, which cannot run at
either length — that comparison belongs at ≤1024 and is [27](27-common-ladder-result.md)'s.

## Two things this run found that are not about `coarse`

**1. The `runlist` catalogue row was stale, and its tolerance figures with it.** The row still
described "host torch attention through the SAME blocked implementation offload uses" and "5
submissions over 391 entries" — the mode as it stood *before* the corrected-taxonomy rebuild
(`52b93e1a`). The lit recipe moved with the rebuild and pins 17/427; the catalogue comment did not,
so it asserted a structure the code had stopped having. Its recorded `mean_rel_L1` 1.732e-2 /
`atol_required` 7.077e-2 were pre-rebuild numbers, which is why a fresh run reads 1.746e-2 /
6.981e-2. **That difference is not a regression — it is a different computation**, since both
attention matmuls and the softmax moved to the device. Corrected in place, with the old figures kept
and marked non-comparable.

The general lesson is doc 16's, one layer over: when a mode is rebuilt, the *gate* gets updated
because it fails otherwise. Nothing fails when a catalogue comment goes stale.

**2. `run_npu2_runlist_gate.lit`'s latency clause has no margin, and it is now INTERMITTENT.**
Its leg A passes only if `agg_ms < seq_ms` — a strict inequality with no tolerance. In one 30-test
suite run (28 tests before these two cells) it failed at `sequential 25.191 ms / runlist 25.277 ms`,
**saved -87 µs, 0.9966×**, while its three bit-identical checks all passed.

**Three runs of the same code disagree, which is the whole finding:** red in that suite run, green
on the isolated re-run (31 s), green again in a second full 30-test suite on the committed tree
(**30/30**, 541 s). The discriminating test from [23 §One process per device
measurement](23-rules-and-open-items.md) is the isolated re-run, and it passes — so this is
contention rather than a regression, and it is *occasional* contention rather than a new steady
state. Do not read a single red here as "the cells broke the seam gate"; do not read a single green
as "it is fine" either.

**The criterion was NOT widened.** Doc 05a measured the real effect at 1.02–1.15×, so the clause is
sound and its margin is the problem, not its claim. Recorded here rather than papered over; whoever
picks it up should decide between a stated margin, a serialized recipe, or leaving it and knowing
that a full-suite red on this one clause is a scheduling artefact until an isolated re-run says
otherwise.

> **`[2026-08-10]` Decided, and it is none of the three options as offered: the verdict now
> compares interleaved MINIMUMS** (`agg_min < seq_min`, both legs of `runlist_gate.py`), with
> medians and the win count still reported beside them. The reasoning is doc
> [23 §1](23-rules-and-open-items.md)'s own convention: host contention only ever *inflates* a
> sample, so the minimum over interleaved samples converges on the mechanism's floor while the
> median carries whatever ran beside the suite — which is exactly what the 0.9966× red was.
> The inequality stays strict; nothing was widened. Validated on an isolated run: leg A
> minimums 24.786 vs 24.921 ms (saved 135 µs, 1.0054×), leg B 23.263 vs 23.607 ms (saved
> 344 µs, 1.0148×), `PHASE B GATE: PASS` on all four legs. A contended suite can still in
> principle inflate every one of 15 interleaved samples, but it has to do so in *both* arms'
> quiet windows at once — the failure now needs sustained, not incidental, contention.

**The suite's standing state on the committed tree is 30/30** (`b795deb1`, 541 s).
