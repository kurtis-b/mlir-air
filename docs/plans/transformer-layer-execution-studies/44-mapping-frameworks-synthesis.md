# 44 — Five mapping frameworks, side by side, and what we should take

`[2026-08-12]` Synthesis of docs [39](44-mapping-frameworks-synthesis.md) (LLMCompass),
[40](44-mapping-frameworks-synthesis.md) (Accelergy), [41](44-mapping-frameworks-synthesis.md) (Timeloop),
[42](44-mapping-frameworks-synthesis.md) (SCALE-Sim) and [43](44-mapping-frameworks-synthesis.md) (MAESTRO). Each was
researched against primary sources — papers, docs and **the actual repositories** — and three of the
five agents found the shipped code disagreeing with the published paper. Read the individual docs for
the evidence; this one is the comparison and the recommendation.

The question that prompted it: **our mapping space is large, particularly once pipelining and
parallelising are mixed**, and we have neither a balance instrument nor a search.

## The comparison

| | data space | mapping space | legality | search | cost model | multi-op |
|---|---|---|---|---|---|---|
| **Timeloop** | einsum: dims + data-space projections | **4 sub-spaces**, Cartesian, one uint128 ID: index-factorisation · permutation · spatial · bypass | **two-tier**: structure *constructed-in*, resources *checked and rejected* with per-level fail reasons | hybrid random + linear-pruned, typed feedback kills whole slabs, victory-condition termination | analytical bottleneck: `max` over per-resource isolated cycles | **no** (LoopTree *models* fused chains, ships no mapper) |
| **MAESTRO** | tensor **dimensions**, not loop vars | 3 directives — `SpatialMap` · `TemporalMap` · `Cluster` — order alone encodes stationarity | **three-tier**: structural fatal; capacity *and* rate **warn with demand beside budget**, and the rate shortfall also lands in runtime | none — it is a cost model (GAMMA/Marvel search on top) | analytical reuse + `required_BW = traffic / compute_delay` | **no** — a transformer is 288 independent `Layer` blocks |
| **SCALE-Sim** | flat CSV, one row per layer, **no edges** | **none** — mapping is *derived* from `{dataflow preset} × {array dims}` | none — everything folds or pads | **none** (one grep hit: a PyPI classifier) | cycle-level simulation over a `[cycle × port]` demand matrix | **no** — parallel branches serialised in file order |
| **Accelergy** | n/a — not a mapper | n/a | n/a | n/a | `Σ count × energy_per_action`; **no time, no contention, no stalls** | n/a |
| **LLMCompass** | transformer template, prefill + decode, KV cache | **13 scalars**; **no field names a core** | capacity-only | brute-force argmin | analytical, wave-based | **no** — sums 12 operator latencies, no `max()` anywhere |

## The finding that matters most

**None of the five can express our central axis.**

> **`[2026-08-12]` CORRECTED by [45](44-mapping-frameworks-synthesis.md), and the correction matters.** This section
> originally read "…and the state of the art ships no fused mapper at all." That is false. **TileFlow
> ships a fused mapper** — a tree of `Tile` / `Scope` / `Op` nodes over **four** scopes
> (`Sequential` / `Sharing` / `Parallel` / `Pipeline`), with FLAT encoded as its test data.
> **`[2026-08-12]` two details in this box were wrong and are corrected by [46](44-mapping-frameworks-synthesis.md)**:
> there is **no genetic algorithm** anywhere in the repo (structure is exhaustively enumerated then
> *uniformly randomly sampled*; MCTS applies only to tile factors), and the machine is **not "an AIE
> column in all but name"** — no memtile in the validated config, no channel cardinality, no DMA
> descriptor, and the vector unit is absent from every architecture YAML, though on AIE the vector
> unit *is* the core. Its RTL validation is the best in this survey and still narrower than it
> sounds: a two-GEMM chain **with no softmax** (the dataset directory is named `No_Softmax`) on a
> single array through a **`Sequential`** scope — **`Pipeline` and `Parallel` are validated against
> nothing**. It was simply not among the five surveyed. The claim that survives is the narrower one: none
> of *these five* can express fusion, and each says so in its own words. **Read TileFlow before
> FLAT.**

Every one of them is single-operator. Not by oversight — each says so in its own words. Timeloop:
*"We leave exploration of cross-layer reuse to future work."* LLMCompass: *"We do not explore
operator fusion in this paper."* FLAT (ASPLOS 2023), surveying the field including MAESTRO: *"none of
them offer support for cross-layer performance (and reuse) modeling, assuming layer-by-layer
execution."*

The closest anyone gets is **LoopTree** (Timeloop v4, in-tree): it *models* fused chains at under 4%
error, including FLAT attention — but ships **no fused mapspace, no fused search and no fused
constraints**. You hand it the mapping. **`[2026-08-12]` correction** ([46](44-mapping-frameworks-synthesis.md)):
LoopTree is **polyhedral (ISL)**, not a node tree, and its sequential-vs-pipeline distinction is a
single **global** flag — so it and TileFlow are different ideas rather than the same one twice. They
are complementary: take TileFlow's tree for *composition*, LoopTree's per-intermediate retention for
*residency*.

**So the thing we are trying to do is not a solved problem we failed to look up.** That is worth
knowing before treating our four walls as evidence of incompetence: the walls are real and the
tooling that would have predicted them does not exist in any of these five.

Two corollaries. Nobody validates a *fused* cost model against silicon, so a mapper we build would
be validating on ground nobody else has. And our willingness to *measure* — every figure in this
study is a real latency or a counted byte — is a genuine asset, because four of these five are
analytical models whose validation is thinner than their abstracts imply (doc 40 on Accelergy's
"95%" being post-layout simulation rather than silicon; doc 39 on LLMCompass's 4.1% being a ratio of
two 12-term sums; doc 42 on SCALE-Sim's scale-out study being unreproducible from released code).

## The correction we owe ourselves

Earlier in this session, reasoning from LLMCompass's wave-deduplication idea, this project proposed
making the per-column MM2S budget a **legality predicate** — exclude any mapping demanding more than
2 per column.

**Timeloop's considered answer is the opposite, and it is right.** Bandwidth is deliberately *not* a
legality condition there: over-subscription becomes `slowdown = min(slowdown, bw/demand)`, never a
rejection. **Capacity is the cliff; bandwidth is a slope.**

That is correct for us for exactly the same reason. Exceeding our per-column budget **does not break
correctness** — AIR packet-multiplexes onto one queue and the design runs slower. A legal-but-degraded
point must be *modelled as degraded*, not filtered out. Worse, filtering would have hidden the precise
failure mode we are trying to see: silent multiplexing would vanish from the search rather than show
up as cost.

MAESTRO offers the refinement: it **warns with demand printed beside budget *and* charges the
shortfall into runtime**. Both channels, not either. That is the shape to copy — the overflow is
visible as a diagnostic *and* priced in the objective.

## The instrument, which all five point at from different directions

Four of the five converge on the same missing piece, and between them they specify it. Given traffic
counts derived from a tiling:

1. **A `[cycle × port]` demand matrix per column** — rows cycles, columns the two MM2S channels
   (SCALE-Sim). Max concurrent demand per column *is* the budget check.
2. **Back-solve the required bandwidth statically** — assume stall-free execution and compute the
   bandwidth that assumption required, one `ceil(elems/cycles)` (SCALE-Sim's `InterfaceBandwidth:
   CALC`). **This needs no simulator and no hardware run**, which is what makes a search affordable.
3. **Price overflow as a slope**, `slowdown = min(1, 2/demand)`, and **print demand beside budget**
   (Timeloop + MAESTRO).
4. **Latency = `max` over per-resource isolated cycles, and the argmax NAMES the offending
   resource** (Timeloop's bottleneck model, ~100 lines). That is precisely the
   "which stage is the bottleneck" answer iron approximated by hand with truncated binaries —
   and doc [38](25-mode-rebuilds-and-results.md) found iron's version had two defects
   inflating its headline gap.
5. **Persist it as a `(component, action, arguments) → cost` table** and make every candidate a dot
   product against it (Accelergy's ERT). **Ours holds *measured* nanoseconds and bytes**, which makes
   our version stronger than Accelergy's rather than weaker. **Actions must carry arguments**: a
   `dma_transfer` is a function of `(n_words, n_dims, stride)`, not a scalar — given our BD-stride
   walls, a counter reporting "number of DMA transfers" has already destroyed the information we need.

**The objective matters more than the search**, and Timeloop has the number: **480,000 mappings within
5% of peak performance vary 19× in energy**, and 6,582 mappings tie on minimum DRAM accesses while
varying 11×. Ties on one metric hide enormous variation on another. Building a search against an
unsound objective converges efficiently on the wrong point — which is exactly what iron's instrument
defect did.

## How to make the space small, in order of leverage

1. **Decouple the subproblems so sizes ADD instead of MULTIPLY.** Marvel optimises the off-chip
   subspace first, then constructs the on-chip space *from* that optimum: **9.4×10¹⁸ → 1.5×10⁸ +
   5.9×10⁷, a factor of 10¹⁰**. Our cut has the same shape, and it gives the per-column budget a
   natural home as an outer-subproblem constraint decided once. This is the single largest reduction
   available and it is arithmetic, not heuristics.

   > **`[2026-08-12]` EXEMPT ONE JOINT — and it is ours** ([45](44-mapping-frameworks-synthesis.md)). FLAT §5.3 argues
   > that **fusion granularity and intra-stage tiling are not separable**: the granularity choice
   > changes what the inner tiling is optimising against. Decoupling *there* would optimise each half
   > against a stale model of the other. That axis is a small enum, so **let it multiply** and decouple
   > everything else. A reduction move applied at the wrong joint is worse than none, because it
   > converges confidently.
   >
   > **`[2026-08-12]` SUPERSEDED by [46](44-mapping-frameworks-synthesis.md) — replace the exemption, do not keep
   > it.** TileFlow §7.3 shows FLAT's `{M, B, H, R}` granularity enum is an **artifact of FLAT
   > refusing to tile the column dimension**, and dissolving it wins **82× L1**. The joint is not
   > inseparable; the enum was. What replaces the exemption is a sharper warning from its Table 7:
   > under **fixed** tiling, granularity looked **18× important**, and under **searched** tiling
   > **three of four granularities tie exactly**. **Every four-mode comparison in this study is at a
   > fixed tiling**, so before any packaged-vs-resident-vs-interleaved claim is published it must
   > survive a tile-size search **on both sides**.
2. **Infer tiles instead of enumerating them.** LoopTree specifies only the *final* stage's tiling and
   derives every upstream stage's by walking dependences; then **one small enumerable choice per
   intermediate** — the last partitioned rank its retained tile spans — decides resident vs refetched
   vs recomputed. That collapses `tile_m/n/k`, `emb_tile`, `seq_tile`, `kv_seq_tile` across stages
   into one spec plus a handful of choices, and it is our residency-within-a-segment constraint in
   searchable form.
3. **Construct structural constraints into the space rather than filtering after.** Timeloop's
   first tier: a sampled mapping obeys them by construction. Keep resource checks as rejection with
   *typed* failure reasons — its mapper uses those to kill whole slabs of the space, not just the
   failing point.
4. **Derive legality, search balance.** iron's split (doc 38), and after its legality prune the space
   was small enough that placements were **hand-authored tables with only seven legal
   `(parallel_heads, parallel_ffn)` tails**. Worth measuring for our own constraint set before
   building any search — the space may be large before legality and small after.

## What does not transfer

- **Timeloop's spatial axis** replicates *one* einsum's nest homogeneously across the mesh. It cannot
  express distinct stages running distinct code on distinct cores — our central axis. There is also
  no shared or global resource budget anywhere (network classes carry no bandwidth field at all), so
  a per-column, per-segment channel budget is unstateable as such.
- **MAESTRO's fixed grammar** — one ordered map list per level, uniform clusters only, no dependence
  model — is *why* it is single-operator. And its budget is a **byte-rate**; ours is a **channel
  count**. A straight port catches a per-column byte-rate overrun and still misses a 2-MM2S
  violation. That cardinality resource has to be added deliberately.
- **SCALE-Sim's service model** — global-lockstep stalling that freezes the whole array, and
  non-contending ports — is exactly wrong for 32 independently scheduled cores sharing two channels
  per column. Borrow its demand *accounting*, not its service.
- **LLMCompass's hardware model** has no channel concept, no per-column resource and no core
  adjacency; the paper concedes machines "where inter-core communication mechanism plays a key role"
  are out of scope. Its `template_to_system` also hardcodes A100 overhead constants for **every**
  user-described architecture.
- **Accelergy's plugins**: CACTI hard-asserts 22–180 nm and reaches a 4 nm-class node via a literal
  `read_energy *= scale**0.5` extrapolation ~5.5× out of range; plugin "accuracy" is a self-declared
  constant typed into source. Take the ERT *pattern*, not the tool.

## Read next

- **TileFlow** — **read this first.** It is the fused mapper the section above wrongly said did not
  exist: GA + MCTS over `sequential`/`pipeline`/`parallel` scopes, RTL-validated at 5.4% latency on a
  machine that is an AIE column in all but name.
- **FuseMax** (MICRO'24) — worked from FLAT's private code and found confirmed bugs, conceptual errors
  in part of its search space, and an unmodelled softmax worth **6.7× iso-area**. Read it beside FLAT,
  not after.
- **FLAT** (ASPLOS 2023) — still worth reading for the *reasoning*, with [45](44-mapping-frameworks-synthesis.md)'s
  caveats attached. It searches
  tiles × order × **fusion granularity** across a *fused attention chain*. It is the closest published
  work to what we are trying to do.
- **LoopTree** — the fused *model*, in-tree in Timeloop v4, for the tile-inference and retention
  encoding above.
- **Union** (PACT'21) — the **MLIR-native** precedent for decoupling problem and mapping abstractions
  from pluggable mappers and cost models. Directly relevant given we are already in MLIR.
- **Ruby** (Timeloop mapspace template) — relaxes perfect factorisation, worth adopting from day one
  given our padded tiles.
- **Eyexam** (Eyeriss v2 appendix) — pencil-and-paper, no software, but it is the member of this
  family aimed at our actual question: where performance is lost, step by step, as a roofline bound is
  progressively tightened.

## Provenance note

Three of the five agents found the shipped code contradicting the published paper: Timeloop's paper
names three mapspace sub-spaces where the repo has four and claims far weaker search than it ships;
Accelergy's ART is absent from the ICCAD'19 paper entirely; SCALE-Sim's v3 paper contradicts its own
Table I on multi-core support, with the code agreeing with the prose. Two agents also corrected
themselves mid-investigation (MAESTRO's bandwidth check exists at warning tier; the wall-6 style
"initially concluded X, then found Y"). **Cite these documents, not the abstracts.**

---

## `[2026-08-12]` What FLAT changed — the plan as it now stands

[45](44-mapping-frameworks-synthesis.md) settled the two questions that could have reordered everything above, and
one of its answers is a genuine redirection.

**Fusing by interleaving is not the same thing as fusing by residency, and we had one word for both.**
FLAT achieves its win by **re-ordering loops on one array**, not by placing stages on distinct cores.
Its arXiv v1 §5.2(2) — a passage cut from the published version — compares interleaved against
spatially pipelined and gives four reasons for choosing interleaving, of which one is directly ours:
**a prefetch spread across two stage durations halves peak off-chip demand**. LoopTree independently
classifies FLAT's parallelism as `sequential`.

So **add `interleaved` as a third composition state** beside doc 03's *packaged* and *resident*. We
currently have one word for two very different DMA profiles, and R1 spent four walls pursuing the
harder of the two without anyone having established that it was the one worth having.

**FLAT's reasoning does not transfer from attention to the FFN interior, and the authors say so in
four places.** The mechanism needs a *quadratic* intermediate **and** operands with zero algorithmic
reuse, so that shrinking the tile is free. R1's FFN interior has neither — its 64-row band **divides
weight reuse**, which is exactly the cost FLAT §5.3 prices. `f(FC, FC)` was considered and rejected.

**But that is not an argument to retarget R1, because the answer depends on the rung.** FLAT's own
end-to-end numbers give the crossover: the L/A chain is **12% of the layer at N=512, 49% at 4096,
79% at 16384**. At `baseline_768` the FFN interior is the right target; at the top of our sequence
ladder attention is. **Make the sequence length an input to which increment gets built**, rather than
picking one and defending it.

**R1's wall 5 and FLAT §5 are the same object.** Doc 31 needs round-major shim issue order and got
channel-major. FLAT's shared outer loop **makes channel-major unreachable by construction** — the
ordering constraint we have been trying to impose downstream is a property of the loop structure
upstream. **Before implementing queue item 6c as a re-interleaving step, check whether the new
`air-fuse-pipeline-launches` spec can carry that shape**; if it can, 6c may be a builder change rather
than a scheduling pass.

**One caveat on FLAT itself, and it is severe.** Its cost model was validated *"within 1% to
MAESTRO"* — analytical against analytical, single-layer only, so **the fused path is validated against
nothing**. FuseMax (MICRO'24), working from the authors' private code, reports confirmed bugs,
"larger conceptual errors" in part of the search space, and that the codebase does not model the
softmax cost at all — worth **6.7× iso-area**. No public code exists; the docs page has said "Code
Available — Coming soon" since June 2023. And the published abstract's headline speedups are **stale
v1 numbers that appear in no table of the paper**. Take FLAT's *reasoning*; do not take its numbers.


---

## Bibliography of the retired research docs (39–43, 45, 46)

`[2026-08-22]` Docs 39, 40, 41, 42, 43, 45 and 46 are consolidated into this section; their full text
(every repo line cited, every paper-vs-artifact audit) is at git tag `pre-cleanup-20260821`. Each entry
gives what the framework is, the finding this study drew from it that the sections above do not already
carry, and what was done with it. The five instrument parts named in §The instrument were built as
queue item 25 and live in [31](31-resident-tail-r1-record.md) (the balance-instrument record, formerly
47: 1,213 ERT entries, 1,208 measured on device; first finding `addnorm` column 0 MM2S demand 3 against
budget 2, priced `slowdown 0.667`); the static legality derivation is queue item 26 in
[16](16-compiler-changes.md) (formerly 48: 115,343,360 → 3,721,772 legal points, a 31× / 96.77% cut,
of which 2,181,680 = 59% are over the shim budget and **priced, not refused**).

**39 — LLMCompass** (Zhang, Ning, Prabhakar, Wentzlaff, ISCA 2024; repo 7,878 lines of Python, read
nearly whole). A transformer-template cost model built to search *hardware* with the mapping held cheap
— the dual of our question. It sums 12 independent operator latencies per block (`"We do not explore
operator fusion"`, §VI-2), `Mapping` is 13 scalars with no field naming a core, data movement is two
scalar bandwidths, and `template_to_system` hardcodes A100's fitted overheads (matmul 21 µs, softmax
12 µs, layernorm 45 µs, gelu 45 µs) for *every* described target — a trap for any NPU config. Audit of
the artifact: the headline 4.1% inference error is one ratio of two 12-term sums whose components carry
9.0–14.9% error; no TPU v3 data exists in the AE (the TPU is configured with infinite DRAM bandwidth and
capacity); the all-reduce "measurements" are transcribed values (8.7% of prefill, 4.7% of decode); the
MI210 was validated with its clock pinned to 1400 MHz (the cousin of our pmode trap); the 26,400-round /
15–16 min figure is the heuristic mapper, not `exhaustive`. The 10.9% per-operator A100/MI210 figure is
the one to trust. Taken: wave batching with cross-wave operand deduplication (`matmul.py:1197-1274`),
which turns a loop order into a *count of distinct transfers per wave* — the currency of the MM2S
budget; and a measured per-operator-class overhead constant fitted at input size 1, the shape
[57 §1](57-inference-path-optimizations-from-hexagon.md) later filled with a measured launch boundary.
The count-as-legality-predicate reading of it is what §The correction we owe ourselves retracted.

**40 — Accelergy** (MIT, ICCAD 2019). Not a mapper, simulator or performance model: a per-action
energy/area accumulator `E = Σ count(action, args) × ERT[component][action][args]`, with counts supplied
by someone else (Timeloop). Findings: the CACTI plug-in supports 22–90 nm and reaches a 4 nm-class node
by `read_energy *= scale**0.5` with `scale = 4/22 ≈ 0.18` (energy ×0.43, area ×0.18), 5.5× outside its
range; plug-in "accuracy" is a self-declared constant (`percent_accuracy_0_to_100 = 80`, Aladdin 70);
the "95% on Eyeriss" is post-layout simulation at 65 nm on N = 1 design and 1 workload (AlexNet), with
leaf energies calibrated from the same flow (Fig. 7: Accelergy 95%, Aladdin 88%, fixed-cost 78%) — no
silicon anywhere. Verdict: **do not adopt**; build the action-count layer, because that artifact *is* the
balance instrument and the energy multiplication is incidental. Taken: the ERT indirection with
actions carrying arguments (instrument part 5, `balance_ert.Ert`), holding measured nanoseconds and
counted bytes rather than modelled picojoules.

**41 — Timeloop** (Parashar et al., ISPASS 2019; `NVlabs/timeloop` @ `32370826`, 2025-06-09; with
Sparseloop, Ruby, LoopTree, Orojenesis, Union). Findings beyond the table above: the unconstrained
mapspace for a 7D CNN on 4 levels is `(7!)⁴ × (2⁴)³ ≈ 2.6 × 10¹⁸` before index factorisation, and
`Uber::Init` hard-fails past 2¹²⁸; the hybrid mapper terminates on `victory_condition` (default 500) or
`timeout` (1000 consecutive invalid); validation is mean 95% cycle accuracy (78–99%) and energy within
8% over 107 DeepBench workloads; bandwidth is deliberately not legality — `buffer.cpp:2475-2599` turns
over-subscription into `slowdown = min(slowdown, bw/demand)`; LoopTree (v4, in-tree) models fused
chains under 4% error including FLAT attention but ships no fused mapspace, search or constraints.
Taken: "capacity is the cliff, bandwidth is a slope" (§The correction); instrument parts 3 and 4
(`balance.balance_ports`, `balance.bottleneck`); the constructed-in / rejected two-tier legality that
[16](16-compiler-changes.md)'s derivation applies; the 480,000-within-5%-vary-19× argument that the
objective matters more than the search, which is why item 25 was built before any search.

**42 — SCALE-Sim** (Samajdar et al., ISPASS 2020; v3 on arXiv). A single-systolic-array performance
model whose whole space is `{os, ws, is} × array dims × 3 SRAM sizes × DRAM bandwidth` — nine INI
scalars, no search, no legality, layer-at-a-time with branches serialised in file order. Paper-vs-repo
ledger: scale-out and DSE are paper-only (v1's `run_sweep()` is dead code); v3's `multi-core/` directory
is absent from `main` (`9f98c43`) and `3.1`; v3 Table II's Sc/T columns are swapped against
`systolic_compute_{ws,is}.py`; `main` crashes on its own `tpuv4.cfg` (`NoSectionError: 'layout'`) and
in both bandwidth modes after patching; only tag `v2.0.2` with a hand-fixed topology runs. RTL
validation covers compute cycles only (OS, 4×4–90×90, full utilisation); v3 reports OS 30.1% fewer
cycles than WS once DRAM stalls count, reversing the compute-only ranking — so v2's unvalidated stall
numbers were load-bearing. Taken: the `[cycle × port]` demand trace and `InterfaceBandwidth: CALC`
(`read_buffer_estimate_bw.py:96-152`) — back-solve the bandwidth a stall-free run required, no
simulator, no hardware — as instrument parts 1 and 2 (`balance.demand_matrix`, `balance.back_solve`),
with the recorded deviation that our rows are ASAP async-dependence levels, not cycles. Its
global-lockstep stall service was left behind.

**43 — MAESTRO** (Kwon, Chatarasi, Pellauer, Parashar, Sarkar, Krishna, MICRO-52 2019; arXiv
1805.02566; Georgia Tech). Data-centric directives `SpatialMap` / `TemporalMap` / `Cluster`; three-tier
legality with the `NotEnough*Buffer` aborts commented out in shipped code, so capacity and rate only
`[WARNING:…]` with demand printed beside budget while the rate shortfall is also charged into runtime;
validated within 3.9% mean error vs MAERI RTL and Eyeriss. No mapping search of its own (the
480M-explored / 2.5M-valid figure is a *hardware* sweep); GAMMA (O(10²⁴) per layer), Marvel
(9.4 × 10¹⁸ → ~2.1 × 10⁸ by sequential off-chip/on-chip decoupling), ConfuciuX and DNNFuser (fusion
mapspaces 64¹⁸ = O(10³²) for ResNet18, O(10⁹⁰) for ResNet50) are each thin wrappers shelling out to the
binary per candidate — the lesson being *build the fast deterministic evaluator first*. FLAT and
DNNFuser both had to write their own cost models because MAESTRO cannot express fusion (FLAT↔MAESTRO
"within 1%" holds only single-layer). Taken: warn-with-demand-beside-budget *and* charge (instrument
part 3, `PortBalance.warning`); Marvel's decoupling as §How to make the space small item 1. Not taken:
its byte-rate budget, which cannot see a 2-MM2S cardinality violation — the gap TileFlow later closed.

**45 — FLAT** (Kao, Subramanian, Agrawal, Yazdanbakhsh, Krishna, ASPLOS 2023; arXiv 2107.06419; no
public code — "Coming soon" since June 2023). Its amendments to this doc are in §What FLAT changed; the
numbers behind them: geomean 1.75× (Edge) / 1.65× (Cloud) speedup and 44% / 55% energy over an
exhaustively-searched intra-operator baseline (v7 Table 5, Fig. 14), 2.8× / 3.07× at N = 64K, off-chip
bandwidth to reach 0.95 utilisation −82% cloud / −71% edge; **Table 4 at N = 512: 1.02× at 20 MB and
2 GB buffer, 1.7× (L/A) and 1.1× (end-to-end) at 200 KB** — fusion buys 2% at conventional length with a
real buffer; batch 64 throughout; Edge = 32×32 PEs, 1 TB/s on-chip, 50 GB/s off-chip, Cloud = 256×256,
8 TB/s, 400 GB/s. The abstract's 1.94× / 1.76× are v1 numbers appearing in no table of the published
paper. FuseMax's audit: the private code models the softmax on 2³⁰ 1D PEs and omits its data transfers;
charging it gives FuseMax 6.7× iso-area on attention (79% of energy), 5.3× end-to-end; four analytical
models agreeing to 1–4% (FLAT↔MAESTRO 1%, LoopTree↔FLAT 3.4%, FuseMax↔FLAT <1%) on a number wrong by
6.7× — *agreement between analytical models is not evidence*. FLAT rejected `f(FC, FC)` in four places;
the FFN interior has neither property its win needs (quadratic intermediate; zero-reuse operands), and
R1's 64-row band divides weight reuse across the three coupled feeds `hidden ×96`, `w_up`, `w_down`
([31](31-resident-tail-r1-record.md)) — with the qualifications that "FCs are compute-bound" is a
batch-64 statement, that the lost reuse is DRAM refetch which memtile residency could remove, and that
our reason for fusing is the 24.0 of 33.0 MiB @1024 crossing of a linear intermediate. Also adopted:
`proportion_spilled` (0 → 0.0078 → 0.50 in FuseMax's CSV) as a first-class instrument output, and the
rule that every resource in the instrument's `max` carries a measured budget or is named as excluded.

**46 — TileFlow** (Zheng, Chen, Gao, Jia, Sun, Wang, Liang, MICRO 2023; `pku-liang/TileFlow` +
`KnowingNothing/Domino`; DOI 10.5281/zenodo.8350955). Its corrections to this doc are in the boxes
above; the numbers: 5,103–20,412 dataflow trees per workload; tiling search 50 rounds × 200 choices at
~12 s/round (3.2–6.4 min, ~60 ms per fused-mapping evaluation); the 3D search visits < 50 rounds × 20
dataflows = 1,000 trees, 5–20% of the structure space (1–2 days single-threaded, < 1 hour on 56
processes). RTL validation: 131 mappings of a two-GEMM chain with **no softmax**, on **one** 16×16 core
(384 KB, 25.6 GB/s, TSMC 22 nm, 7.84 mm², 400 MHz, Verilator), **5.4% per-point MAPE on cycles,
unfitted** — the best-supported claim in the survey — while the 6.1% energy figure is against a table
synthesised in `validation.py` and the 48.8% "graph-based" strawman is three lines in the same script;
vs Timeloop, 1,152 single-operator mappings at R² 0.999. Table 7c: FLAT-HGran 4.10 MB L1 / 32.77 MB L2
against TileFlow 0.05 MB / 20.48 MB (82× less L1) at 16.78 vs 14.68 × 10⁶ cycles (14% slower).
Taken: the scope-typed resource combinator (`checker.cpp:486-535`, ~30 lines: `max` over `Seq`/`Shar`
children, Σ over `Para`/`Pipe`), reimplemented as a per-column vector priced as a slope — the mechanism
behind [16](16-compiler-changes.md)'s legality derivation; `SlowDown` per level with the bandwidth sweep
as a spec generator; the 2×2 composition ontology `Seq`/`Shar`/`Pipe`/`Para` = packaged / resident /
interleaved / *(unnamed, the `f(heads ‖ ffn)` tails)* replacing the three-state list above; free
variables by naming (`factors: B=? H=?`) and the ERT as an input file; max-of-rollouts and `−log₁₀`
reward if a search is ever built. Rejected: `Pipeline` as a bare `max` with no fill or drain (ours must
be `T·max(stage) + (S−1)·max(stage) + handoff`, COMET concurring), its small-tile data movement (an
upper bound, since it "assumes data replacement happens for every outer iteration"), and TileFlow as a
tool — an unmaintained 2023 Timeloop fork that `exit(1)`s on an illegal mapping, usable at most as a
batch oracle for data-movement volume. Read-next additions: COMET (arXiv 2509.00599 — collectives as
nodes, ramp-up/ramp-down modelled, spatially distributed clusters), Chimera, AccelForge (pushed
2026-08-10, not investigated).

**What the two fused-dataflow docs left open, and how it closed.** 45 and 46 together asked whether
*resident* composition is the right target at our sequence lengths at all — FLAT's crossover puts
attention at 12% of the layer at N = 512 and 79% at 16K, TileFlow's Table 7 shows composition claims at
fixed tiling can be artifacts of the tiling, and FLAT itself reaches its win by interleaving on one
array rather than by co-residency. That made continuing R1 an operator decision rather than a default.
**It is settled `[2026-08-21]`**: R1 is reframed, not closed — the array's workload is modelled as
**supertiles** of per-core tiles composed as regions, one supertile after another as separate executions
in the runtime sequence, each inside the working box (`herd_x = 1`, `down_K ≤ 6`) so wall 7's shared-L2
multi-writer buffer is designed out; each supertile produces a finished output block (`down_K = 96` per
execution, meeting the unresolved `down_K ≥ 7` wedge), and accumulating down partials across executions is admissible only if it measures faster —
R1's first increment is that two-form comparison on hardware. Recorded in
[31](31-resident-tail-r1-record.md).
