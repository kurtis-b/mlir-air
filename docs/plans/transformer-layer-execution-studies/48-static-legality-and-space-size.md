# 48 — Static legality, and how big the mapping space actually is

`[2026-08-12]` Queue item 26. **Host-only. No device was dispatched for anything in this document, and
no compiler was run either** — that is the claim being made, not an economy.

Every number below is produced by `programming_examples/transformer_layer/study/mapping_space.py` and
pinned exactly by `run_mapping_space_tests.lit`. Reproduce with
`python3 study/mapping_space.py` (about a minute) or `--axes` for the axis table with each bound's
source.

---

## The measurement

| | points | |
|---|---:|---|
| raw axis product | **293,601,280** | every axis at its full declared range |
| **before legality** | **115,343,360** | after the divisibility the builders `raise` on — Timeloop's first tier ([46](46-research-tileflow.md): *"divisors constructed-in, capacity rejected"*) |
| **after legality** | **3,721,772** | **a 31× cut, 96.77% removed** |
| of which **priced, not refused** | **2,181,680** | **59% of the legal space** is over the per-column shim budget under the placement the tools produce, and stays in |

**The queue row's hypothesis is falsified.** Item 26 was filed on the expectation that our space might
be, like iron's, *"small enough that placements were hand-authored tables"* — doc
[38](38-iron-encoder-pipeline-reference.md) measured iron's legality prune leaving only **seven** legal
`(parallel_heads, parallel_ffn)` tails. Ours does not collapse like that. Legality removes 96.77% and
leaves 3.7 million points.

**But the conclusion the row wanted still holds, for a different reason** — see *Does enumeration beat
search* below. The reason is not that the space is small; it is that the predicate is **static**.

### Against iron, on a comparable slice — with the caveat stated first

**Seven is not the size of iron's space.** It is a **two-axis slice** of it: `LOW_HEAD_TAILS` keyed on
`(parallel_heads, parallel_ffn)`, doc [38 §2.3](38-iron-encoder-pipeline-reference.md). So the honest
comparison is slice against slice, not slice against total.

**And the axes are not the same axes.** iron's spatial-replication axes are heads × FFN branches; ours
are FFN width × sequence lanes (`gemm_herd_x` ↔ `parallel_ffn`, `parallel_bands` ↔ `parallel_seq`).
**We have no `parallel_heads` analogue at all** — the resident tail has no attention in it — so this
compares the *size of each system's spatial-replication slice*, not a point-for-point correspondence.
Read it as an order-of-magnitude check, which is all it can be.

| slice | iron | ours |
|---|---:|---:|
| spatial replication — iron `(parallel_heads, parallel_ffn)`, ours `(gemm_herd_x, parallel_bands)` | **7** | **21** |
| herd widths (`gemm_herd_x`, `norm_herd_x`) | — | 16 |
| whole structural sub-space (forms × fold × widths × bands) | — | **428** |
| structures × scope-per-seam | (hardcoded registry) | **15,347** |

**Same order on the replication slice — three times as many — and three orders more structures.** That
last row is the one that closes a door: iron's `TOPOLOGY_PLACEMENTS` is a hand-authored `(col,row)` for every
core of every supported topology, and hand-authoring 15,347 of those is not open to us. Whatever we
build has to *derive* placement, which is what `air-place-herds` already does and what doc 31's
"NO PLACEMENT, NO BUFFER DEPTHS" discipline has been banking on.

### The finding that matters most

**59% of the legal space is priced.** Doc [44](44-mapping-frameworks-synthesis.md) corrects an earlier
proposal to make the per-column shim budget a hard legality filter. Had that proposal stood, this
predicate would have deleted **2,181,680 routable designs** — the majority of the space — and with them
every instance of the failure mode the study is trying to see. AIR does not refuse an over-subscribed
column; it packet-multiplexes, silently (doc [23](23-rules-and-open-items.md), and queue item 10
measured it: zero inbound `aie.flow` and 12 `aie.packet_flow` on a design 50% over). A filter would
have made the space look cleaner and the model blind.

That is the strongest argument for doc 44's slope that this study has produced, and it is a
measurement rather than an argument.

---

## What is refused, and what is priced

The line is drawn at **placement invariance**, and that is the one design decision in the module worth
arguing about.

The shim budget is per column; which column a stream lands in is the allocator's choice. So most of
the budget is *not statically decidable*. What is decidable is its placement-invariant part:

**REFUSED — no placement can route this**

| clause | why it is a cliff | evidence |
|---|---|---|
| a herd with > 2 herd-direct L3 operands | every column that herd occupies carries all of them, under every placement | queue item 10's control, MEASURED: 12 `aie.packet_flow`, 3 per column |
| segment shim demand > 16 slots | there are 16 | doc 23: J1 failed here — *"8 columns × 2 = 16, already full before the third stream"* |
| cores > 32, or herd widths that do not pack into 4 rows × 8 columns | `aie.tile` refuses | [31b §3.5](31b-r2-order-seam.md), MEASURED: nine herds of `[4,1]` refuse with *"row index (6) must be less than the number of rows in the device (6)"*; eight place |
| L1 over 64 KiB | aiecc's allocator refuses | `norm_tail._stage_l1_bytes`, MEASURED at `rows_per_call` 8, `cols` 768 |
| memtile over 6 MM2S / 6 S2MM | there is no port to give | `ffn_accum.MAX_FEED_CHANNELS` |
| `Para` at a dependent seam | the stages would read garbage | TileFlow §4.1: *"Para … is only applicable to tiles without data dependency"* |

**PRICED — routes, and is degraded**

The per-column demand under the placement the tools are *known* to produce — doc 23's stacking rule,
*"three 8-wide herds put one tile of each into every column, so their demands add"* — charged
`min(1, budget/demand)` per [46 §1](46-research-tileflow.md).

R1 is the live example of why this cannot be a cliff: a placement exists with worst column **1**, and
the shipped allocator produced worst column **2** ([31b §3.6](31b-r2-order-seam.md), MEASURED). Both
legal. A predicate that refused on the difference between those two would refuse R1, which ships.

---

## The controls, and why a static predicate needed them at all

Item 10's lesson is that a census counting surviving *ports* reads **0** on an over-budget column,
because AIR converts the design to packet flows — blind exactly where it is needed. A static predicate
has a different shape: it reads the declaration, so it sees the third stream directly. **"Different
shape" is not evidence**, so both directions are demonstrated on every invocation, inside the gate.

| control | result | against |
|---|---|---|
| **NEGATIVE** — item 10's own over-budget design, one herd `[4,1]` with 3 herd-direct L3 operands | **REFUSED**, `12 of 16 slots, 3 per column, predicted [3,3,3,3,0,0,0,0]` | MEASURED through aircc 2026-08-12: 0 inbound `aie.flow`, 12 `aie.packet_flow` |
| **POSITIVE** — R1's shipped interior | **ADMITTED**, `shim MM2S 7/16, 4 shim→core + 3 shim→memtile` | [31b §3.6](31b-r2-order-seam.md) MEASURED **7 of 16, 4 + 3** — reproduced with no compiler |
| R1's memtiles | `[(4, 2), (4, 5)]` | §3.6 MEASURED the up feed 4/6 MM2S with 2/6 S2MM, the down feed 4/6 with 5/6 |
| herd inventory | 9 → REFUSED, 8/7/5 → place | [31b §3.5](31b-r2-order-seam.md), MEASURED |
| J7a's column budget | `[2,2,2,2,2,2,2,2]` | doc 23: *"exactly met by the packed fetch and the gamma fetch"* |

**The positive control is the load-bearing one.** Reproducing §3.6's routed-design census —
column for column, plus both memtile rows — from a declaration with no `air-opt`, no aircc and no
dump is the whole claim of this item. `ffn_resident_structure.py` needs a compile to count that; this
needs none.

### Tamper-verified, and it takes both controls

Two defects were injected into a scratch copy and the gate re-run. **Neither control catches both**,
which is why both run.

| tamper | negative control | positive control | census |
|---|---|---|---|
| clamp the placement-invariant shim refusal to its own budget — item 10's failure mode, injected | **CAUGHT**: `3 per column -> ADMITTED`, *"it cannot fail and is not gating anything"* | passes | `space after legality` moves **3,721,772 → 4,071,136**; FileCheck red |
| invert TileFlow's combinator — `max` where a space scope must be **Σ** | **passes** (a one-herd control has `max` = `Σ`) | **CAUGHT**: R1 counts `2/16 (0 shim→core + 2 shim→memtile)` against the measured 7/16, 4+3 | J7a control also caught: predicted `[1,1,…]` where doc 23 says exactly 2 |

The second is the one worth dwelling on. **A `max`-instead-of-Σ combinator is invisible to any
single-herd control** — it is precisely doc [44](44-mapping-frameworks-synthesis.md)'s charge against
MAESTRO's per-level budget, and only a control with *several herds in one segment* and a measured
number to hit can see it. That is R1.

### Two defects the controls caught in this module itself

Both were mine, both were caught by clauses written to make the predicate falsifiable, and both are
worth recording because they are the same defect in new places.

1. **`core_s2mm=min(CORE_S2MM, …)`** — a demand clamped to its own budget, in two places. That is
   item 10's failure mode exactly: the clause could never fire, so every design passed it. Demands are
   now reported raw and compared by the predicate, and `test_no_demand_is_clamped_to_its_budget`
   asserts the demand, not the verdict — a clamp is invisible in a verdict that is right for another
   reason.
2. **A GEMM L1 byte formula derived from prose.** `build_ffn_accum_module`'s docstring says
   *"32 keeps the worst-case L1 comfortably under the 64 KiB tile; 64 measures just over it"*. Ping-ponged
   A+B plus a resident C at `tile_k` 64 comes to **91 KiB**, not "just over" 64 — so the prose does not
   determine the formula, and the version I first wrote **invented a wall**, refusing `gemm_herd_x`
   1 and 2 which `builders/ffn_accum` accepts. In a module whose output is a space size, an invented
   wall is an invented cut in the direction that flatters the headline. The tree states that wall as
   the constant `MAX_L1_TILE_K`, which is already the `tile_k` axis bound, so the GEMM groups now
   contribute no L1 byte demand and the module says why at the point where the formula would go.

The first draft's census read **0 legal points** and `main()` refused itself. That refusal is the
reason both defects were found before this document existed rather than after.

---

## The space: axes, ranges, and where every bound comes from

The object is the **R2 resident tail** — `nt1 → up → gelu → down → nt2`, doc
[31b §5](31b-r2-order-seam.md)'s stage groups. Run `python3 study/mapping_space.py --axes` for this
table live.

| axis | values | bounded by |
|---|---|---|
| `nt1_form`, `nt2_form` | fused, j7a | [31b §5](31b-r2-order-seam.md)'s herd inventory: `NORM_TAIL_STAGE_SPEC`'s three stages, or the one-herd form on the addnorm pre-add kernel |
| `gemm_fold` | split, folded | [31b §5](31b-r2-order-seam.md) row 2 — the fold is free at the object level, `ffn_accum_mm.o` already exports `ffn_gelu_bf16` (§7.2) |
| `gemm_herd_x` | 1–4 | `ffn_accum.MAX_PLACEABLE_HERD_X`, **MEASURED**: `aie-place-tiles` refuses the accumulator pair's shim slots at 6 columns |
| `norm_herd_x` | 1–8 | `NPU2TargetModel::columns()` = 8; `norm_tail` defaults to 8 and records that `air-place-herds` places it |
| `tile_k` | 8,16,24,32 | `ffn_accum.MICRO` = 8, `MAX_L1_TILE_K` = 32, **MEASURED** |
| `rows_per_call` | divisors of 64 | `ffn_accum.TILE_M` — the band a dispatch carries. Its L1 ceiling is a *capacity rejection*, not a range bound |
| `parallel_bands` | 1–8 | **NOTHING IN THIS TREE BOUNDS IT** — see below |
| routing × 5 operands | herd-direct, L2-staged | [31b §4](31b-r2-order-seam.md)'s table verbatim (`packed1`, `gamma1`, `w_up`, `w_down`, `gamma2`) |
| scope per seam | Seq, Shar, Para, Pipe | [46 §6](46-research-tileflow.md)'s 2×2, mapped onto doc [03](03-measurement-model.md)'s words |

### The axis I could not bound from an artifact, stated plainly

**`parallel_bands`.** A band lane needs at least one column, so 8 is an upper bound — but no
measurement in this tree bounds it, and the shipped builder refuses anything above 1
(`ffn_resident` requires `seq_len == TILE_M`). Doc 31: a multi-band module *"would need `herd_y` > 1
through the same memtile fan-out, which is unmeasured — refuse, do not guess."* The census names it on
every run (`axes no artifact bounds: ['parallel_bands']`) and the lit recipe pins that line, so the
loosest bound in the space cannot be dropped quietly.

It is also the only axis carrying doc 46's fourth scope — `Para`, *"unnamed, and we have it"* — since
every seam of the tail chain is dependent.

### Deliberately excluded, with the reason

- **R1's column partitioning vs R2's row partitioning.** [31b §6.1](31b-r2-order-seam.md): *"That is
  not a parameter, it is a different module."* A different module is a different space.
- **Anything about attention.** iron's `parallel_heads` has no analogue here; the resident tail has no
  heads in it. Note this makes our space *smaller* than iron's on an axis iron has and we do not.
- **The norm tails' internal seams.** [31b §5](31b-r2-order-seam.md): the stages R2 is about are the
  tail's *operator* boundaries, *"not the internals of a normalization"*.

---

## Does enumeration beat search?

**Yes — but not because the space is small.** It is 3.7 million points, which is not small.

It is because **the predicate is static**. No compile, no device, no dump — one `legality()` call is
tens of microseconds of arithmetic on a declaration, where the compiled census it reproduces costs an
aircc run per point.

The census itself takes **about a minute**, and the honest description of that minute is that it does
*not* visit 115 million points one at a time: it walks ~10⁵ (structure, seam vector) pairs and
multiplies in the tiling and routing sub-counts, a factorisation valid because the predicate reads
those axis groups through disjoint clauses and asserted against brute force over whole sub-spaces
(`test_the_factorisation_matches_brute_force`). A point-by-point walk of all 3.7 million legal points
is roughly two orders more work — still minutes, not hours, and trivially parallel.

A search exists to avoid evaluating points. At tens of microseconds a point, there is nothing here
worth avoiding.

The corrected shape of the argument:

- **The structural sub-space is 15,347** (structures × scope-per-seam). Each needs a builder that can
  emit it. That is three orders above the seven iron hand-authored, so **iron's hardcoded-placement
  approach does not port** — placement must be derived, not tabulated.
- **The remaining multiplicity is tiling and routing**, which are cheap sweeps once a shape is built,
  and which the predicate treats independently of the structure (asserted against brute force over
  whole sub-spaces, `test_the_factorisation_matches_brute_force`).
- **So the binding constraint is not search, it is `generate_fusion_plans`** — a builder that can emit
  an arbitrary point of the structural sub-space. We do not have one; we have `ffn_resident.py` at one
  point of it.

**What this reorders.** Item 25's balance instrument stays first, unchanged and now better motivated:
with legality static and the space enumerable, the objective is the only thing left that can be wrong,
and doc 44's finding that 480,000 mappings within 5% of peak vary **19×** in energy says an unsound
objective converges confidently on the wrong point. What this item removes from the plan is any
argument for building a *search* — MCTS, GA, or otherwise — before a parameterised builder exists.
Doc [46 §7](46-research-tileflow.md)'s verdict on TileFlow's mapper (*"the MCTS is not the hard part"*)
is confirmed from our own side.

---

## Two by-products worth their own lines

**The shim budget, not the array, is what caps band parallelism of a co-resident tail.** Four lanes of
the leanest possible co-resident tail fit in cores and do not fit in DMA: the tail reads five L3
operands however they are routed, so four concurrent lanes want 20 of NPU2's 16 shim MM2S slots. Not
what *"8 columns, 32 cores"* suggests, and pinned in
`test_the_band_axis_is_capped_by_the_shim_budget_not_by_cores`. Splitting the tail with a `Seq` seam
lifts it, because segments take the array in turns — which is the packaged/resident trade appearing as
a *legality* result rather than a performance one.

**Eight-wide norm herds and the FFN cannot share one segment's shim budget.** J7a alone *"exactly
meets"* the budget at width 8 (doc 23), so in the R2 tail `norm_herd_x` 8 survives **only through a
`Seq` seam** — width 8 in R2 means packaging it. This is [31b §4](31b-r2-order-seam.md)'s anxiety
turned into a decidable statement, and it is why that section stages both gammas.

`test_the_axis_values_legality_eliminates_outright_are_the_pinned_ones` pins the whole set of axis
values legality eliminates, so either of these changing has to be noticed.

---

## What this model does not know

- **Bytes and cycles.** It is a cardinality model. It answers *can this be routed* and *how many ports
  does it want*, never *how fast*. That is item 25.
- **Which column.** It answers *does a legal placement exist* and *is the placement the tools are known
  to produce within budget* — not what `air-place-herds` will do.
- **Whether the shim clauses are jointly sufficient.** Both refusals are *necessary* conditions and
  both are placement-invariant; a joint packing counterexample — widths and demands that satisfy each
  condition separately and admit no simultaneous assignment — would be a refinement, not a
  contradiction. None appears among the herd multisets this space produces.

## Reproduce

```
cd programming_examples/transformer_layer
python3 study/mapping_space.py          # the census, ~1 min, host-only
python3 study/mapping_space.py --axes   # every axis with the artifact that bounds it
python3 study/run_host_tests.py         # 393/393, includes the 36 predicate checks
```

Gated by `run_mapping_space_tests.lit` (the census, every number above pinned) and
`run_study_host_tests.lit` (the predicate checks, count pinned at 393/20 modules). Both are in the
PR-safe allowlist, which moves 10 → 11.
