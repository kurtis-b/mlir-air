# 44 — Five mapping frameworks, side by side, and what we should take

`[2026-08-12]` Synthesis of docs [39](39-research-llmcompass.md) (LLMCompass),
[40](40-research-accelergy.md) (Accelergy), [41](41-research-timeloop.md) (Timeloop),
[42](42-research-scalesim.md) (SCALE-Sim) and [43](43-research-maestro.md) (MAESTRO). Each was
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

> **`[2026-08-12]` CORRECTED by [45](45-research-flat.md), and the correction matters.** This section
> originally read "…and the state of the art ships no fused mapper at all." That is false. **TileFlow
> ships a fused mapper** — a tree of `Tile` / `Scope` / `Op` nodes over **four** scopes
> (`Sequential` / `Sharing` / `Parallel` / `Pipeline`), with FLAT encoded as its test data.
> **`[2026-08-12]` two details in this box were wrong and are corrected by [46](46-research-tileflow.md)**:
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
constraints**. You hand it the mapping. **`[2026-08-12]` correction** ([46](46-research-tileflow.md)):
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
   and doc [38](38-iron-encoder-pipeline-reference.md) found iron's version had two defects
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

   > **`[2026-08-12]` EXEMPT ONE JOINT — and it is ours** ([45](45-research-flat.md)). FLAT §5.3 argues
   > that **fusion granularity and intra-stage tiling are not separable**: the granularity choice
   > changes what the inner tiling is optimising against. Decoupling *there* would optimise each half
   > against a stale model of the other. That axis is a small enum, so **let it multiply** and decouple
   > everything else. A reduction move applied at the wrong joint is worse than none, because it
   > converges confidently.
   >
   > **`[2026-08-12]` SUPERSEDED by [46](46-research-tileflow.md) — replace the exemption, do not keep
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
- **FLAT** (ASPLOS 2023) — still worth reading for the *reasoning*, with [45](45-research-flat.md)'s
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

[45](45-research-flat.md) settled the two questions that could have reordered everything above, and
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
