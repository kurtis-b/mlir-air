# 45 — FLAT (ASPLOS 2023), researched against primary sources

`[2026-08-12]` Research doc in the series of [39](39-research-llmcompass.md)–[43](43-research-maestro.md),
synthesised in [44](44-mapping-frameworks-synthesis.md). Doc 44's "Read next" names FLAT as *"the paper
to read instead of MAESTRO for our problem"*. This is that read.

**Subject.** *FLAT: An Optimized Dataflow for Mitigating Attention Bottlenecks* — Sheng-Chun (Felix)
Kao (Georgia Tech), Suvinay Subramanian (Google), Gaurav Agrawal (Microsoft; work done at Google),
Amir Yazdanbakhsh (Google Research, Brain), Tushar Krishna (Georgia Tech). ASPLOS 2023,
[DOI 10.1145/3575693.3575747](https://dl.acm.org/doi/10.1145/3575693.3575747), pp. 556–568,
arXiv [2107.06419](https://arxiv.org/abs/2107.06419).

---

## 0. Provenance first: what exists, and what does not

### 0.1 Versions — and the camera-ready check

arXiv 2107.06419 has **seven versions**: v1 2021-07-13, v2 2021-08-21, v3 2021-08-26, v4 2021-12-03,
v5 2022-04-18, v6 2022-04-19, v7 2022-09-24 ([arXiv abs page](https://arxiv.org/abs/2107.06419)).
v7's cover line reads *"Appears in the Proceedings of the International Conference on Architectural
Support for Programming Languages and Operating Systems, 2023"* — the authors' own statement that v7
is the ASPLOS version.

**I could not read the ACM camera-ready body.** Semantic Scholar records it as
`openAccessPdf … "status": "GOLD", "license": "CCBY"` pointing at `dl.acm.org/doi/pdf/…`, but
dl.acm.org returns 403/Cloudflare to every automated fetch I tried (direct, browser UA, r.jina.ai).
What I *can* verify is the **abstract**: the ACM/Semantic Scholar abstract
([S2 API, DOI:10.1145/3575693.3575747](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1145/3575693.3575747))
is word-for-word arXiv v7's abstract. Everything below is cited to arXiv v7 unless marked otherwise;
where v1 differs materially I say so, because **v1 → v7 is a substantive rewrite, not a polish**.

**v1 vs v7, the differences that matter to us:**

| | arXiv v1 (2021-07) | arXiv v7 / ASPLOS (2022-09) |
|---|---|---|
| Accelerator name | **ATTACC** ("FLAT-compatible accelerators") | dropped; FLAT is the dataflow only |
| Tiling hierarchy | **L1-tile / L2-tile / L3-tile + FLAT-tile**, with per-tensor FLAT-tile enable/disable (v1 §3.1, §5.1) | collapsed to "intra-operator tiling sizes" + FLAT-tile |
| Interleaved-vs-pipelined | **explicit §5.2(2) with four named reasons** for choosing interleaved over spatially pipelined | **removed**; only the mechanism survives ("run one after another (interleaved)") |
| Cost-model validation claim | "backward compatible to MAESTRO (which in turn is **RTL-validated**)" (v1 line 1434) | **the words "RTL" and "validated" do not appear anywhere in v7** (verified by grep) |
| Headline numbers | body: 2.52×/1.94× edge, 2.60×/1.76× cloud; 60%/49% edge energy, 69%/42% cloud | body: **1.75× edge, 1.65× cloud; 44% edge, 55% cloud energy** |

The v1 interleaved-vs-pipelined section is the single most useful passage in the entire paper for us,
and it is **not in the published version**. Read v1 for it.

### 0.2 The abstract's headline numbers are stale v1 numbers that no table in the published paper supports

This is a concrete, checkable defect of the same class doc 44 catalogues for three of the other five
frameworks.

- **Abstract (v5, v6, v7, and the ACM camera-ready):** *"FLAT delivers 1.94× (1.76×) speedup and 49%
  and (42%) of energy savings compared to the state-of-the-art Edge (Cloud) accelerators with no
  customized dataflow optimization."*
- **Body of the same document (v7 §1, and again §8.1 line 7019–7022):** *"Compared to a range of
  state-of-the-art dataflow optimizers, FLAT delivers **1.75× and 1.65×** speedup and **44% and 55%**
  energy savings for recent Edge and Cloud accelerators."* Figures 14a/14b are labelled
  "Geomean Speedup = 1.75×" and "Geomean Speedup = 1.65×".
- I grepped v7 for `1.94`, `1.76`, `1.75`, `1.65`: **1.94 and 1.76 occur exactly once each, both in
  the abstract, and nowhere in any table, figure or body sentence.**
- Their origin is v1 §6, verbatim: *"ATTACC achieves 2.52x and 1.94x speedup in edge cases and 2.60x
  and 1.76x speedup in cloud cases, while reducing the energy by 60% and 49% in edge cases and 69%
  and 42% in cloud cases."* So the abstract quotes **the second (weaker) of two baselines in each
  platform, from a version whose evaluation was subsequently redone**.
- The abstract also attributes them to accelerators *"with no customized dataflow optimization"* —
  i.e. the Naïve baseline. But v7's Naïve comparison is Table 4, whose numbers are 1.5×–3.3×
  depending on buffer size, not 1.94×/1.76×.

**Consequence for us: cite 1.75×/1.65× over an exhaustively-searched intra-operator baseline, and
never the abstract.** The abstract number is unsourced in its own paper.

### 0.3 Code: there is effectively none

- The official documentation page,
  [github.com/flat-attention/web](https://github.com/flat-attention/web) (org created for the paper),
  `index.md`, verbatim: **`# Code Available` / `Coming soon`**. Repo has 9 commits, all on
  2023-06-14, and has never been touched since (verified via the commits API).
- The only public artifact by an author is
  [felix0901/flat_prototype](https://github.com/felix0901/flat_prototype) — created and last pushed
  2022-06-03, **3 KB repo, 2 stars, no licence, one file**: `flat_prototype.ipynb`, a 16 KB Colab
  notebook. I read all 13 cells. It contains ~60 lines of JAX: `matmul3(A,B,C) = matmul → softmax →
  matmul` under `@jax.jit`, plus `fused_L_softmax_A(...)` which is a triple loop
  `for bt in range(0,batch,B): for ht in range(0,head,H): for mt in range(0,seq_q,R)` calling it, and
  an unfused `L_softmax_A` that materialises the whole `[B,H,N,N]` intermediate. The two sweeps
  (head=12, hidden=768, seq=256, batch 1…128; then batch=1, seq 1…128) reproduce **Tables 8 and 9**
  of the paper exactly. **This is the GPU compatibility experiment, not the simulator, not the
  cost model, not the map-space exploration framework.**
- No FLAT repository exists in `maestro-project` (8 repos, checked) or `Accelergy-Project`
  (33 repos, checked). MAESTRO's tree contains `Transformer_Complete.m` / `Transformer_Layers.m` and
  nothing FLAT-related.
- **The simulator does exist — privately.** FuseMax (MICRO 2024) §VI-A: *"Though we started with the
  FLAT authors' original code, we found and corrected a number of bugs. Through private
  correspondence with the FLAT authors, we verified the bugs were indeed bugs."*
  ([arXiv 2406.10491](https://arxiv.org/abs/2406.10491)).

**So FLAT is the one framework in this survey where "what the tool does" cannot be checked directly
at all — there is no tool.** What follows is therefore: the paper's claims, plus two independent
reimplementations by third parties who *did* have the code (FuseMax, LoopTree) and one who
reimplemented the dataflow from the paper (TileFlow). Those three are where the audit comes from.

---

## 1. The core intuition, as the authors state it

**The quantity traded is operational intensity**, defined in v7 §2.1 Eq. (1) as
`I = #operations / #memory accesses`, with off-chip traffic in the denominator, and the roofline
ridge point as the threshold.

The argument runs in three steps, all in §3:

**(a) The two attention GEMMs are structurally memory-bound and batching cannot fix it.** FLAT splits
attention's operators into *activation-weight* (Q, K, V, O — GEMMs against learned weights) and
*activation-activation* (L = QKᵀ, A = SV — GEMMs between two activations). For activation-weight,
`I = O(BND² / (BND + D² + BND))`, so raising `B` raises intensity — the standard batching move. For
activation-activation, ops are `O(BN²D)` and accesses are `O(BND) + O(BND) + O(BN²)`, giving
`I = O(BN²D / (2BND + BHN²))` — **`B` cancels**. §3.1's closing sentence: *"low operational intensity
of the individual L/A operators makes them fundamentally memory-bound, and **any dataflow/mapping
exploration at the individual operator level cannot further improve performance**."* The Fig. 2
roofline caption puts it bluntly: *"L/A operator is seriously memory-bounded. Packing larger batch
size does not help increase its performance."*

**(b) The naive dataflow loses because it materialises the score matrix to DRAM.** v1's abstract has
the crispest statement: *"the high BW requirement of attention layers on typical inference
accelerators is actually from moving the inter-operator (intermediate) tensor back and forth between
memory, and the problem exaggerates when dealing with O(N²) attention matrix."* Fig. 7(a)'s caption:
*"Conventionally, entire L tensor gets materialized back in off-chip to pass data and avoid data
dependency b/w Op."* Fig. 4 sizes it: for BERT/TrXL/XLM the L→A intermediate is 10–12 MB at N=512,
136–144 MB at N=2K, **8.2–8.3 GB at N=16K**.

**(c) The fused dataflow changes the *footprint growth rate*, and intensity follows.** Abstract:
*"transforming the memory footprint quadratic growth to merely a linear one."* §4.1: after fusing L
and A, *"the effective operational intensity (of the fused operator) is higher"* — Fig. 5 plots the
ridge point at **130 FLOPs/Byte** on TPU-v3 and shows `L/A` below it and `F(L,A)` above it.

**Note carefully what "linear" means.** Table 1 gives buffer requirement by granularity:

| Granularity | Buffer requirement |
|---|---|
| M (batched multi-head) | `O(8BDN + BHN²)` |
| B (batch) | `O(8DN + HN²)` |
| H (head) | `O(8Nd + N²)` |
| R (row) | `O(4Rd + 4Nd + RN)` |

§5.3 derives R-Gran: *"L operator consumes (Rd+Nd)×2 size of the on-chip buffer (2 to account for
double buffering), and A consumes (Nd+Rd)×2. RN for buffering the intermediate tensor (FLAT-tile)."*
**The `4Nd` and `RN` terms are the whole key/value tensors and a full score row: `N` is never tiled.**
FLAT has **no online/streaming softmax** — it is a 3-pass cascade. Residency is `O(N)`, not `O(1)`.
FuseMax states this as the design's central flaw: *"FLAT requires that the entire vector over which
the softmax is performed be buffered on chip… When the vector/sequence length grows beyond allowable
buffer capacity, FLAT is forced to spill."* FLAT's arXiv v1 predates FlashAttention by ten months, so
this is chronology, not oversight — but it is a live limitation of the published design.

### Effect size, in their numbers

| Comparison | Number | Source |
|---|---|---|
| FLAT-Opt vs FLEX-Opt, geomean over 5 models, **Edge** | **1.75×** speedup, **44%** energy | v7 §8.1, Table 5, Fig. 14a |
| FLAT-Opt vs FLEX-Opt, geomean, **Cloud** | **1.65×** speedup, **55%** energy | v7 §8.1, Table 5, Fig. 14b |
| At N=64K, Edge | 2.8× vs FLEX-Opt | v7 §8.1 |
| At N=64K, Cloud | 3.07× vs FLEX-Opt | v7 §8.1 |
| Off-chip BW requirement to hit 0.95 utilisation | **−82%** cloud, **−71%** edge | v7 §8.1, Fig. 15 |
| **At N=512, FLAT over FLEX** | **1.02×** at 20 MB and 2 GB buffer; **1.7×** (L/A) and **1.1×** (end-to-end) at 200 KB | **v7 Table 4** |
| At N=512, FLAT over Naïve | 1.5×–3.3× depending on buffer | v7 Table 4 |

**Table 4 is the number nobody quotes and the one that matters most to us.** At conventional sequence
length with a real buffer, fusion buys **2%**. Everything FLAT sells is bought at long sequence or at
a starved buffer. Table 2's platforms: Edge = 32×32 PEs, 1 TB/s on-chip, 50 GB/s off-chip;
Cloud = 256×256 PEs, 8 TB/s on-chip, 400 GB/s off-chip. Batch size is **64 throughout** (v1 §6.1:
*"We run all the models with batch size of 64"*).

---

## 2. Fusion granularity as an axis — how it is parameterised

**This is the part doc 44 hoped for, and it is genuinely there.**

### The knob

FLAT introduces the **FLAT-tile**: the inter-operator tile, distinct from intra-operator tiles.
Fig. 6 gives the loop nest exactly:

```
for b_o = [0, B, Bx):          #  \
  for h_o = [0, H, Hx):        #   >  OUTER LOOP — shared by L and A
    for n_o = [0, N, Rx):      #  /
       L_t = inner_loop_L(Q_t, K_t)    #  \
       L_t = Softmax(L_t)              #   >  INNER LOOPS — one per operator, run back to back
       A_t = inner_loop_A(L_t, V_t)    #  /
```

§5 states the rule: *"we divide the loop nests into two groups: 'outer-loop' and 'inner-loop'… The
outer-loops are shared across L and A. The inner-loops are unique for each operator. After fusion,
the fused operator has two inner-loops, which we run one after another (interleaved), and iterate
through the shared outer-loop."* And: *"we use L and A for illustration… but **the principles are
applicable to any set of consecutive tensor operators**."*

**The granularity knob is the triple `(Bx, Hx, Rx)`**, and the named levels are which of the three is
less than full:

| Level | Meaning | Floor set by |
|---|---|---|
| **M-Gran** | whole batched multi-head intermediate on chip (the naive "keep it all resident") | — |
| **B-Gran** | micro-batch `Bx < B` | — |
| **H-Gran** | `Hx < H` heads | — |
| **R-Gran** | `Rx < N` rows | **row granularity is the hard floor** |

The floor is *derived from a dependence, not chosen*. §5.2: *"The Softmax reduction is along the key
dimension… The minimum Softmax execution requires an `[1, N]` input array, which in turn requires a
query of `[1, D]` and a key of `[D, N]`… This forms our basic tiling unit (finest granularity) —
row-granularity, which respects the data dependency introduced by the Softmax while keeping minimum
number of elements to pass between L and A. **FLAT restricts the tile sizes to operate in multiples
of this row-granularity.**"*

That is the pattern worth stealing wholesale: **the finest legal fusion tile is a theorem about the
chain's reduction, and the search then ranges over integer multiples of it.**

### How it interacts with tiling and loop order: **jointly, and the paper says why**

§5.3, explicitly: *"reducing the number of rows at the outer-loop could also decrease the achievable
performance at the inner-loop, e.g., not enough dimension size to fully utilize PE array. Thus,
**FLAT co-explores inter-operator (optimizing the outer-loop) and intra-operator dataflow (optimizing
the inner-loop)** to mitigate these potential sources of inefficiencies."*

Two named coupling mechanisms:

1. **Outer granularity destroys inner reuse.** *"even for L/A fusion, using fewer rows means the same
   key vectors need to be fetched multiple times across the interleaved cross-operator outer loops."*
2. **Outer granularity starves the PE array.** Not enough rows to fill the systolic dimension.

And then the **load-bearing asymmetry** — the reason the whole scheme works for L/A and only for L/A
(§5.3):

> *"for f(FC, FC) and f(CONV, CONV), when decreasing the batch size (i.e., micro-batching), we
> directly reduce the number of times a weight can be reused. The weights need to be re-fetched again
> and again for each micro-batch… **In contrast, L and A are activation-activation operations. Each
> new activation of L needs to compute with a new activation of A, i.e., there are no reuse
> opportunities at the algorithmic level. Decreasing the tiling granularity (M-Gran to B-Gran to
> H-Gran), does not preclude any reuse opportunity, since there are no reuse opportunities at the
> algorithmic level.** Thus, the finer M-Gran, B-Gran, H-Gran in FLAT are well-suited for f(L, A)."*

**Fusion granularity is free exactly when the operands have no algorithmic reuse to lose.** Hold that
sentence; §6 below turns on it.

---

## 3. The search

Fig. 8 ("Map space exploration framework") specifies the space exhaustively:

| Axis | Levels |
|---|---|
| **Granularity** | `B, M, H, R` |
| **Tiles** | `Bx, Hx, Rx`, **and intra-operator tiling sizes** |
| **Compute order** | `W / I / O`-stationary |
| (v1 only) NoC | tree, systolic, crossbar |
| (v1 only) extra tiling levels | `L1-tile`, `L2-tile`, `L3-tile`; FLAT-tile enable/disable per tensor |

Inputs: HW config (PEs, buffers, NoCs, array shape), workload, objective. Outputs: mapping (dataflow
+ tile sizes), runtime, energy.

**Search strategy: exhaustive, and nothing else.** §6.1: *"In this work, we use **exhaustive search**
to find the optimal design point uniformly across all the dataflow optimizations."* v1 §6.1 repeats
it: *"We use exhaustive search to find the optimum point under the user-specified objective, e.g.,
best run time."* There is no pruning, no feedback, no typed rejection, no sampling. Table 3 defines
the three points compared: **Naïve** = intra-operator weight-stationary with fixed tile size;
**FLEX-Opt** = exhaustive search over intra-operator dataflow only; **FLAT-Opt** = exhaustive search
over intra- *and* inter-operator dataflow.

**No space-size and no search-time number is published.** I grepped v7 and v1 for `design space`,
`search space`, `exhaustive`, `search time`, `map space`, `candidate` — every hit is prose. Fig. 13
plots the space as a utilisation-vs-footprint scatter and the accompanying text is qualitative
(*"there are abundance of parameters that can be tuned under different optimization objectives"*).
**This is a weaker search story than Timeloop's or MAESTRO's downstream tools, not a stronger one.**
FLAT's contribution is the *axis*, not the search over it.

**Objective:** compute utilisation, `Util = Runtime_ideal / Runtime_dataflow`, or energy. §7 makes
the doc-44 point independently: *"while in this work, we focus on maximizing the compute utilization,
one may choose other objectives such as maximizing utilization normalized to memory footprint size,
leading to points in the top-left corner, or the least memory footprint size, leading to points in
the left-most region."*

**The most concrete rendering of FLAT's mapping space is not in FLAT's paper — it is in FuseMax's
artifact.** `workspace/outputs/pregenerated/results/flat_validation.csv` in
[FPSG-UIUC/micro24-fusemax-artifact](https://github.com/FPSG-UIUC/micro24-fusemax-artifact) has one
row per (platform, model, seq_len) with the chosen mapping spelled out:

```
… Q_loc, K_loc, QK_loc, A_loc, V_loc, AV_loc,   # ∈ {onchip, offchip}  — per-tensor placement
  QK_stationarity, AV_stationarity,             # ∈ {output, weight}
  fusion_granularity,                           # ∈ {head, row}
  P1,                                           # row tile ∈ {256, 512, 1024}
  proportion_buffered, proportion_spilled, …
```

Reading the BERT-cloud rows shows the tiles × granularity coupling *as data*: head-granularity wins
up to 4K; at 8K it switches to row-granularity with `P1=1024`, then `512` at 32K, then `256` at 64K;
`proportion_spilled` goes 0 → 0.0078 → **0.50 at 128K**; and at 256K+ the search abandons fusion
altogether (`QK_loc`/`A_loc` flip to `offchip`, stationarity flips `output`→`weight`).

---

## 4. Interleaving vs co-residency — the decisive answer

**FLAT does *not* require the stages to be co-resident. It is loop re-ordering on a single compute
resource. And the authors considered co-residency explicitly and rejected it.**

arXiv **v1 §5.2(2)**, in full — this passage is not in the published version:

> *"(2) **Interleaved Execution of fused-operator.** The fused-operator can be executed either in an
> interleaved manner or in a pipelined manner. **In interleaved execution (aka temporal pipelined),
> all PEs compute the FLAT-tile of L and feed it back to the PE array (after a softmax) to now
> compute A**, followed by the next FLAT-tile of L, and so on. **In (spatially) pipelined execution,
> half the PEs can compute L and feed the output to the rest of the half running A.** We found
> interleaving with double buffering to be a better implementation choice. **First**, interleaved
> execution requires minimal changes to the controller fetching tiles from SG while pipelining
> requires splitting the PE array, adding area overhead. **Second**, the pipelined array incurs fill
> and drain latencies. **Third**, the pipelined array becomes inefficient during the execution of
> non-fused operators. **Fourth**, interleaved execution ameliorates the off-chip memory bandwidth
> requirements of double buffering. When executing fused L-A operator, the warm-up buffer of stage-L
> can be fetched across the duration of two stages: when the active buffer of stage-L and the active
> buffer of stage-A are being worked on. The pipelined execution (and baseline), however, can only
> leverage one stage, viz., when that operator's active buffer is being worked on."*

Corroboration from three independent directions:

- **v1 §Related Work:** *"Some recent work also considers cross-operator dataflow, targeting CNNs and
  leveraging **pipeline** execution. This work targets attention-based models and leverages
  **interleaved** execution, while respecting data dependencies."* The distinction is deliberate and
  the authors drew it themselves.
- **LoopTree** (IEEE TCAS-AI 2024, [arXiv 2409.13625](https://arxiv.org/abs/2409.13625)) §V-C-5
  classifies FLAT independently: *"FLAT is an accelerator for transformers that partitions B, H, and
  M and **processes tiles sequentially** with a B, H, M schedule."* Its Table V "Parallelism" column
  reads **`s`** (sequential) for FLAT, against `p` (parallel/pipelined) for Fused-layer CNN, ISAAC
  and PipeLayer.
- **The hardware requirement is trivially small** (v7 §5.3): *"FLAT requires minimal HW support:
  (1) controller to recognize the proposed fine-grained dataflow and (2) on-chip buffer to be
  software-addressable to support tiling."*

**One caveat that matters enormously to us.** FLAT is not *purely* single-resource: Fig. 8 shows a
**Special Function Unit** (a separate 1D array) alongside the 2D PE array, and softmax runs there
concurrently with the GEMMs. §6.1: *"we allot **sufficient FLOPs to the Special Function Unit** in
order to eradicate the expected compute bottlenecks, uniformly across all the dataflow variants."*
FuseMax quantifies "sufficient": **2³⁰ 1D PEs** (§5.1 below). So FLAT's design *does* rely on one
co-resident heterogeneous partner stage — it just assumes that partner is free.

**Bottom line for our decision:** FLAT's argument is that you get the DRAM-traffic win from
**re-ordering one stage's loops so the chain is consumed incrementally**, and that spatially
splitting the array to do it is a net loss. It does not argue for segment co-residency; it argues
against the analogous thing.

---

## 5. The cost model and its validation — be skeptical

### 5.1 What the model is

**Analytical**, hand-built, *"following similar methodology as prior work [MAESTRO, Timeloop]"*
(§6.1). It models: a PE array with configurable per-PE bandwidth to a global buffer; W/I/O-stationary
intra-operator dataflow; systolic / tree / crossbar distribution-reduction NoCs; per-PE local
scratchpads for input/weight/psum/output; a global on-chip buffer holding intra- and inter-operator
tiles; **data spilling** when live footprint exceeds capacity; **tile fill/drain overhead** (*"cold
start and tailing effect"*); and on-/off-chip memory as **shared, limited-bandwidth resources**
(*"if the access rate to a shared memory resource exceeds a pre-defined bandwidth, the data accesses
are throttled. This overhead manifests as longer runtime"*). Energy via **Accelergy** on activity
counts. Note §6.1's own scoping: *"FLAT neither alters the total number of computations nor the total
number of accesses to the on-chip global buffer. Instead, it optimizes the number of off-chip memory
accesses."*

### 5.2 What it was validated against — read this exactly

> v7 §6.1: *"To ensure the integrity and correctness of our framework, **we compared the simulation
> results from our framework under single-layer modeling to MAESTRO. The performance metrics are
> within 1% difference to MAESTRO's results.**"*

Three things follow, and all three are load-bearing:

1. **It is an analytical model checked against another analytical model** — not silicon, not RTL, not
   a measured accelerator.
2. **It is checked only in the single-layer, unfused configuration** — i.e. exactly the configuration
   that is *not* the paper's contribution. **The fused path is validated against nothing.**
3. **The transitive claim was quietly retired.** v1 said the model *"is backward compatible to
   MAESTRO (which in turn is **RTL-validated**)"*. v7 contains no occurrence of "RTL" or "validated"
   at all (grep). Whether that was a length cut or a retraction, I cannot tell; note that doc
   [43](43-research-maestro.md) records MAESTRO's own validation as 3.9% mean error vs MAERI RTL and
   Eyeriss's *reported* runtimes — so the chain, even taken at face value, is
   FLAT-unfused ≈(1%) MAESTRO ≈(3.9%) MAERI RTL, with the fused contribution hanging off the end
   unattached.

**The real measurement in the paper is the GPU experiment** (Tables 6–9): a Tesla T4 with 16 GB
running BERT-Edge and the TrEMBL protein dataset, showing FLAT reaching 64K tokens where the baseline
OOMs at 4K, and 1.5×/32×-sequence/8×-batch headlines on the poster. This is a genuine measurement —
and I identified its source code exactly: the `flat_prototype.ipynb` notebook (§0.3). But it measures
**a JAX/XLA prototype on a GPU**, and validates nothing about the accelerator model that produces
every headline number.

### 5.3 The audit: FuseMax found the code contradicting the paper

FuseMax (MICRO 2024, Nayak/Andrulis et al., MIT + Berkeley + UIUC;
[arXiv 2406.10491](https://arxiv.org/abs/2406.10491) §VI-A, "FLAT Baseline"):

> *"Though we started with the FLAT authors' original code, **we found and corrected a number of
> bugs**. Through private correspondence with the FLAT authors, **we verified the bugs were indeed
> bugs**. We also discovered **a couple of larger conceptual errors, which the authors told us to
> avoid by restricting FLAT to only search through configurations without these issues**.
> …
> **However, the FLAT codebase does not model the cost to perform the softmax. Specifically, their
> model ignores the cost of the data transfers required for the softmax (between any levels of the
> memory hierarchy) and uses 2³⁰ 1D PEs for compute.** When comparing FuseMax and FLAT in this work,
> we augment our Timeloop model to model softmax correctly per the 3-pass cascade implicitly assumed
> by FLAT **using only 256 1D PEs**."*

And, on energy (FuseMax footnote 6): *"FLAT reports larger energy savings over the unfused baseline
because **it only reports energy associated with DRAM traffic during the tensor products**."*

Set that against v7 §6.1's sentence *"Finally, we account for softmax operation runtime in all the
evaluations."* The two are not reconcilable on the data-movement half. The compute half is
semi-disclosed — "sufficient FLOPs to the SFU" *is* how you would describe 2³⁰ PEs — but "sufficient
FLOPs" reads as a modelling simplification and 2³⁰ PEs against a 256×256 main array is an
**infinity**.

**Effect size of the correction: 6.7× iso-area speedup for FuseMax over FLAT on attention (79% of the
energy), 5.3× end-to-end (83% of the energy).** FuseMax's diagnosis of *why*: *"While FLAT's design
does make attention compute bound, it becomes compute bottlenecked in the 1D array (the softmax),
causing severe under utilization of the 2D array."* Their Fig. 7 shows FLAT's 2D-array utilisation
collapsing. **Once you charge for the softmax, FLAT's win is a fraction of what it reports.**

This belongs in doc 44's provenance note as a **fourth** framework whose shipped code contradicts its
paper — and it is the worst case of the four, because the discrepancy is in the mechanism the paper
is *about*.

### 5.4 Third-party agreement figures, and what they actually mean

| Claim | What it compares | Silicon? |
|---|---|---|
| FLAT "within 1% of MAESTRO" | analytical vs analytical, **single-layer only** | no |
| LoopTree "differ by at most 3.4%" vs FLAT | **LoopTree's model vs FLAT's simulator** (fused) | no |
| FuseMax Timeloop model reproduces corrected FLAT "< 1%" | model vs **corrected** code | no |
| **TileFlow: 5.4% mean latency error, 6.1% energy, over 131 self-attention fusion mappings** | model vs **synthesised RTL, Verilator cycle counts** | **yes (RTL)** |

Doc 44's line *"LoopTree models fused chains … including FLAT attention, at under 4% error"* is
correct as stated but should be read precisely: the 4% is **worst case across five validated
designs**, the FLAT row is **3.4%**, and the reference is **FLAT's simulator, not hardware**.
LoopTree even names the gap: *"The small difference stems from aspects in the FLAT simulator that
LoopTree does not model (e.g., latency from loading weights and systolic array startup)."*

**Correction owed to doc 44.** Doc 44 says *"nobody validates a fused cost model against silicon"*.
**TileFlow does — against RTL.** [pku-liang/TileFlow](https://github.com/pku-liang/TileFlow),
MICRO'23 §7: a 4-core accelerator, each core with a 16×16 matmul array **and a 16×3 vector array**,
384 KB per core, 25.6 GB/s DRAM, 16-bit words, Chisel→Verilog, Cadence Genus/Innovus, TSMC 22 nm,
7.84 mm², 400 MHz, simulated in Verilator; 131 hand-written assembly self-attention fusion mappings
compared against TileFlow's predictions → **5.4% mean absolute latency error, 6.1% energy**. That
architecture — multiple cores, each with a matrix unit *and* a vector unit, a per-core L1, one shared
DRAM port — is by a distance the closest published model to an AIE column.

**Second correction owed to doc 44.** Doc 44 says *"the state of the art ships no fused mapper at
all."* **TileFlow ships one.** MIT-licensed C++, 71 stars, last pushed 2024-04-12, with
`src/mapper/mapper.cpp` + `src/mapper/checker.cpp`, a search combining **genetic algorithm and Monte
Carlo Tree Search** over a **3D space of {compute ordering, resource binding, loop tiling}**, and a
tile-centric tree notation whose `Scope` node carries **`sequential` / `pipeline` / `parallel`**
types. It even ships FLAT as test data: `tests/cases/04-test-attention/map/map-flat.yaml`, which
encodes FLAT as a `Scope: sequential` over two `Tile` subtrees (`permutation: LMK # input stationary`
for GEMM1, `permutation: LMN # output stationary` for GEMM2) — **FLAT's interleaving rendered as a
`sequential` scope, exactly as §4 concluded.** TileFlow reports 1.85× over FLAT-HGran and 2.30× over
FLAT-RGran on self-attention, its own dataflow being *"to pipeline all the three"* operators.

---

## 6. **Attention vs FFN — the direct answer**

**FLAT's reasoning does *not* transfer to the FFN interior. FLAT considered fusing FF1→FF2 and
explicitly rejected it, in four separate places, and its published evaluation runs the FFN unfused.**

1. **§4.1, "Why not fuse other operators":** *"(2) **Fusing two FCs (f(FC, FC)) can achieve higher
   operational intensity; however since the operator is already compute-bound, there is not much
   value in leveraging fusion** (and the additional complexity). (3) We often need finer-granularity
   dataflow schemes to fit fused operator tensors on-chip; however **fusing two activation-weight
   computation (f(FC, FC)) can trade-off (weight) reuse opportunity and may reduce actual achievable
   performance**."*
2. **§5.3 (quoted in full in §2 above):** micro-batching an FC-FC fusion *"directly reduce[s] the
   number of times a weight can be reused. The weights need to be re-fetched again and again for each
   micro-batch. **This effect is exacerbated when considering finer granularities such as H-Gran for
   the weight-activation K/Q/V/O operators.**"*
3. **v1 §4.5:** *"the intermediate tensors between K-L, A-O, etc., **do not exhibit the quadratic
   growth problem**. Therefore, the benefit of fusing other operator become limited and can otherwise
   deteriorate the performance owing to larger memory footprint."*
4. **§8.1, what they actually ran:** *"in FLAT-Opt-Edge, **both K/Q/V/O and FF1/FF2 are treated as
   non-fused operators**, and hence the map space for them are the same as the one in FLEX-Opt-Edge."*

Fig. 5 does plot `F(FC2, FC2)` — that bar *is* FF1⊗FF2 fusion (the caption: *"FC1 and FC2 indicate
operator K/Q/V/O and FF1/FF2, respectively"*). Its intensity rises. It is already above the
130 FLOPs/Byte ridge without fusing. **That is the whole argument: you cannot buy performance with
operational intensity you already have.**

The mechanism needs two properties, and **the FFN interior has neither**:

| Property FLAT's win depends on | Attention L→S→A | FFN up→GeLU→down |
|---|---|---|
| Intermediate grows **quadratically** in the tiled dimension | `[N × N]` per head — yes | `[rows × 4D]` — **linear in rows** |
| Operands have **zero algorithmic reuse**, so shrinking the fusion tile costs nothing | activation⊗activation — yes | activation⊗**weight** — **shrinking the band divides weight reuse** |

**Our R1 is the second column.** A 64-row band means `w_up` and `w_down` are re-fed per band — and
doc [31](31-fused-resident-tail.md) confirms this is literally what the runtime sequence does: the
three coupled L3 feeds are `hidden ×96`, `w_up`, `w_down`. FLAT §5.3 is a closed-form prediction that
this trade loses, and it is the paper's stated reason for not doing what R1 does.

**Three qualifications that pull back the other way, and I want them on the record because they are
the difference between "FLAT says we were wrong" and "FLAT says we are answering a different
question".**

- **(a) FLAT's "the FCs are already compute-bound" is a batch-64 statement.** v1 §6.1: *"We run all
  the models with batch size of 64."* Fig. 2's legend plots FC at `B=1` (light) and `B=128` (dark)
  as *separate points*, and the body says *"FC operators scatter across both memory and compute bound
  region; however, with the increase in batch size, their operation intensity increases and can
  become compute-bound."* **[Marked as inference:]** at batch 1 — inference, one sequence — the FFN
  GEMMs sit in the memory-bound region too, and FLAT's premise for excluding them does not hold at
  our operating point. FLAT never evaluates batch 1 on an accelerator.
- **(b) The reuse FLAT says you lose is *DRAM refetch* reuse.** If the weights can be made resident
  across bands rather than re-fed, the cost FLAT prices disappears. FLAT has no mechanism for that
  because its "on-chip buffer" is one undifferentiated global scratchpad; we have memtiles.
- **(c) FLAT is silent on our actual reason for fusing.** We fuse to eliminate a *DRAM crossing* of
  a linear intermediate (24.0 of 33.0 MiB @1024, doc 31). FLAT's framework prices that correctly —
  it just concludes the intensity gain is not where the money is at batch 64 and N≤512.

### And the number that decides *ordering*, not *whether*

v7 §7, on end-to-end sensitivity:

> *"**for the sequence length below 512, both Block-level and Model-level (i.e., End-to-End)
> performance is dominated by FC/GEMM operators. Therefore, the gains from FLEX-Opt and FLAT-Opt are
> immaterial.** The significant gains from our approach emerge when the sequence length increases
> beyond 512 to 4K, 16K, and to 64K. Under these long-sequence lengths, **the runtime contribution of
> L and A operators grows from 12% to 49%, 79%, and 94%**, respectively."*

| Sequence length | L/A share of end-to-end runtime |
|---|---|
| 512 | **12%** |
| 4K | 49% |
| 16K | 79% |
| 64K | 94% |

**At our `baseline_768` rung, FLAT's own data says attention is ~12–15% of the layer and the FFN is
where the time is.** At the top of our 64…16384 ladder, it says attention is ~79%. **The answer to
"is attention the right first resident target?" is therefore rung-dependent, and FLAT supplies the
crossover: it is between 512 and 4K.** That is a far more useful answer than yes or no.

---

## 7. What transfers to a 32-core AIE array, and what assumes a monolith

### Transfers

1. **The granularity ladder as a named, ordered, small knob with a floor derived from a dependence.**
   `M ⊃ B ⊃ H ⊃ R`, floor = the reduction's minimum unit. Our analogues are exact: for the FFN
   interior the floor is the down-projection's K-reduction unit; for attention it is the softmax row.
   This is a **four-valued enum**, not a factorisation lattice, and that is why FLAT's exhaustive
   search is affordable.
2. **Table 1's closed-form buffer requirement per granularity level.** A capacity predicate
   parameterised by the granularity choice, computable without a run. Our L1/L2 capacity check has
   this exact shape.
3. **The joint-search argument, with its stated mechanism.** Outer granularity is what destroys inner
   reuse and PE fill; therefore inter- and intra-operator dataflow must be co-explored. This is a
   *real* coupling and it is precisely the "pipelining × tiling" coupling we are struggling with.
4. **v1 §5.2(2) reason (iv) — the prefetch-spreading argument — is a per-column MM2S argument in
   disguise.** Under interleaving, a stage's warm-up buffer can be fetched across **two** stage
   durations instead of one, so **peak** channel demand roughly halves at unchanged total bytes. That
   is directly a statement about our budget: the same traffic, scheduled interleaved rather than
   stage-parallel, has lower peak concurrency per column.
5. **`Util = Runtime_ideal / Runtime_dataflow` as the objective**, and the Fig. 13 observation that
   different objectives land in different corners of the same space — consistent with doc 44's
   "the objective matters more than the search".
6. **The 12% / 49% / 79% / 94% table**, as a free prior on where to spend residency effort per rung.

### Does not transfer

1. **One homogeneous PE array + one global scratchpad + a free SFU.** FLAT has no per-column
   resource, no channel *count*, no notion of distinct code on distinct cores. Its only shared
   resource is a byte-rate on a bus, throttled when exceeded — the same gap doc 44 records for
   MAESTRO, and for the same reason.
2. **FLAT explicitly rejects spatial partitioning, and two of its four reasons are inapplicable to
   us.** Reasons (i) "splitting the PE array adds area overhead" and (iii) "the pipelined array is
   inefficient during non-fused operators" both presuppose a *re-taskable homogeneous* array. Our
   herd/segment structure is statically partitioned at compile time either way; there is no unsplit
   array to preserve. Reason (ii) fill/drain is real and we pay it. Reason (iv) is the one that
   survives and it is the good one.
3. **Batch 64 throughout.** Every roofline claim is a throughput claim. Our inference workload is not
   that.
4. **No online softmax → residency is `O(N)` in the key dimension.** `4Nd + RN` at R-Gran means a
   FLAT-style attention segment on our array holds the whole K and V for a head plus a score row. At
   our memtile capacity that caps the sequence length at which attention residency is even
   *expressible*, and the fix (FlashAttention-2's 1-pass cascade, as adopted by FuseMax) is a
   **different algorithm, not a different mapping** — it will not fall out of any search over FLAT's
   space.
5. **The published performance numbers themselves**, per §5.3 — softmax data movement unmodelled,
   softmax compute given 2³⁰ PEs, energy counting DRAM traffic during tensor products only.

---

## Comparable summary

- **Data-space representation.** Named tensor *dimensions* of the attention layer specifically —
  `B` (batch), `H` (heads), `N` (sequence), `D` (hidden), `d = D/H` — with operators pre-classified
  into two fixed classes: **activation-weight** (Q, K, V, O, FF1, FF2) and **activation-activation**
  (L, A). There is no general einsum, no index-projection notation and no tensor-shape DSL: the graph
  of Fig. 1 is hard-wired. Operators are additionally typed as one-to-one (element-wise) or
  many-to-many (tensor-wise), and the paper's whole framing is that fusing many-to-many with
  many-to-many is the unsolved case.

- **Mapping-space representation.** Two nested levels, explicitly split. **Inter-operator**: a
  **FLAT-tile** `(Bx, Hx, Rx)` in a loop nest whose outer loops are *shared* by both operators and
  whose two inner loops are per-operator and run back to back; the tile is named by a four-valued
  **granularity** enum `{M, B, H, R}` = which of `(Bx, Hx, Rx)` is less than full, with `R` the floor
  derived from the softmax reduction. **Intra-operator**: conventional tile sizes plus a
  `W/I/O`-stationary compute order (v1 also carried `L1/L2/L3`-tile levels, a NoC-topology choice,
  and per-tensor FLAT-tile enable/disable). Reconstructed from FuseMax's artifact, the searched
  vector is `{granularity, row-tile, per-operator stationarity, per-tensor on/off-chip placement}`.

- **Legality model.** Effectively one hard rule and one soft one. **Hard, constructed-in**: tile sizes
  must be integer multiples of row-granularity — the softmax dependence is enforced by the shape of
  the space, never checked and rejected. **Soft, priced**: capacity is *not* a cliff — exceeding the
  global buffer triggers modelled **spilling** (*"while the live memory footprint is larger than the
  SG buffer, we model the data to be partially fetched on-chip and partially fetched off-chip"*), and
  bandwidth over-subscription is **throttled into runtime**, never rejected. This is Timeloop's
  bandwidth-as-a-slope position extended to capacity as well. There is no typed failure reason, no
  per-level diagnostic, and no user-visible budget-vs-demand print.

- **Search strategy.** **Exhaustive enumeration, and nothing else** — no pruning, no feedback, no
  sampling, no termination condition. Affordable only because the space is tiny by construction: a
  4-valued granularity enum × a handful of tile factors × 3 stationarities. **No published space size
  and no published search time**; the "design space" figure (Fig. 13) is a qualitative scatter.

- **Cost model.** Analytical, hand-built after MAESTRO/Timeloop. PE array with configurable per-PE
  bandwidth; systolic/tree/crossbar NoCs; per-PE and global scratchpads; modelled spilling; modelled
  fill/drain (cold start and tail); on- and off-chip memory as shared limited-bandwidth resources
  that throttle when over-subscribed; Accelergy for energy from activity counts. **Validation:
  "within 1% difference to MAESTRO's results" — analytical vs analytical, and *single-layer only*, so
  the fused path is validated against nothing.** v1's transitive "MAESTRO … is RTL-validated" claim
  does not survive into v7. Independently audited by FuseMax against the (private) code: bugs
  confirmed by the authors, "larger conceptual errors" in part of the search space that the authors
  advised avoiding, **softmax data movement not modelled at all, softmax compute given 2³⁰ PEs**, and
  energy counting only DRAM traffic during the tensor products — worth **6.7× iso-area** once
  corrected.

- **Multi-op support.** **Yes — and it is the only one of the six with a fused *mapper*, however
  small.** Fusion granularity is a first-class searched axis, jointly with tiles and order. But the
  support is narrow in three ways: exactly **two** operators (§4.1 rejects `f(L,A,O)` and `f(K,L,A)`
  outright), exactly **that** pair (K/Q/V/O and FF1/FF2 are run unfused in every reported result),
  and the cross-stage schedule is **temporal interleaving on one array**, with spatial pipelining
  explicitly considered and rejected. Third parties have since gone further on all three counts:
  **LoopTree** models fused chains with `Pipeline`/`Sequential` nodes but ships no mapspace;
  **TileFlow** ships a real fused mapper (GA + MCTS over compute-ordering × resource-binding ×
  loop-tiling, `Scope: sequential | pipeline | parallel`) validated to **5.4% against RTL**;
  **FuseMax** replaces the algorithm rather than the mapping.

- **Single most transferable idea.** **Derive the finest legal fusion tile as a theorem about the
  chain's reduction, name the coarser tiles as a short ordered enum above it, and give each level a
  closed-form buffer requirement — then search granularity *jointly* with tiles and order, because
  coarsening the shared outer loop is precisely what starves the inner one.** For the FFN interior
  that reads: the floor is the down-projection's K-reduction unit; the ladder above it is band size;
  the buffer formula is arithmetic we already have; and the joint part is the coupling between band
  size, weight refetch and per-column feed concurrency that R1 hit head-on. Runner-up, and closer to
  immediately actionable: **v1 §5.2(2) reason (iv)** — interleaving lets a prefetch be spread across
  two consumer stages instead of one, cutting **peak** channel demand at unchanged total bytes. That
  is a statement about our 2-MM2S-per-column budget written in someone else's vocabulary.

- **Single biggest mismatch with our target.** **FLAT's entire win is powered by two properties of
  the attention score matrix that the FFN interior does not have** — a quadratic intermediate, and
  activation⊗activation operands with zero algorithmic reuse, so shrinking the fusion tile is free.
  In the FFN the intermediate is linear in the band and shrinking the band divides weight reuse;
  §5.3 says so explicitly and §4.1 declines the fusion on exactly that ground. Secondarily, and
  structurally: FLAT's machine is one homogeneous PE array plus one global scratchpad plus a
  *modelled-as-free* softmax unit, with no per-column resource, no channel-count cardinality, and no
  way to say that two stages run distinct code on distinct cores — so the axis our hard problem lives
  on is not merely unsearched, it is unstateable, and the design's own reasoning argues against
  putting it there.

---

## What this changes for our plan

Doc 44's plan: **(A)** build a balance instrument — per-column demand matrix → static bandwidth
back-solve → overflow priced as a slope → latency as `max` over per-resource cycles whose argmax
names the bottleneck → persisted as a measured cost table; **(B)** shrink the space by decoupling
subproblems so sizes add rather than multiply.

### FLAT **confirms**, and sharpens

- **Overflow as a slope, not a cliff — extend it to capacity too.** FLAT prices *capacity* overflow
  as modelled spilling rather than as illegality (*"while the live memory footprint is larger than
  the SG buffer, we model the data to be partially fetched on-chip and partially fetched off-chip"*).
  Doc 44 concluded "capacity is the cliff; bandwidth is a slope" from Timeloop. FLAT is a second,
  independent design that makes **both** slopes, and its `proportion_spilled` column (FuseMax's CSV:
  0 → 0.0078 → 0.50) is exactly the diagnostic MAESTRO-style budget-beside-demand printing would
  give us. **Recommend adopting the spill fraction as a first-class output of the instrument**, since
  a segment that *nearly* fits and one that half-spills are the two cases we most need to tell apart.
- **The argmax-names-the-bottleneck design is validated by FLAT's failure to have it.** FuseMax's
  6.7× is entirely a bottleneck-attribution story: FLAT's model could not see that its 1D array was
  the bound because it gave that array 2³⁰ PEs. An instrument whose `max` ranges over a resource you
  have modelled as infinite reports the wrong argmax with perfect confidence. **Concrete rule: every
  resource in our `max` must have a *measured* budget, and any resource deliberately excluded must be
  named in the output**, not silently omitted.
- **The measured cost table is the right bet.** Four analytical models now agree with each other to
  1–4% about FLAT's fused performance (FLAT↔MAESTRO 1%, LoopTree↔FLAT 3.4%, FuseMax↔FLAT <1%) and
  the number they agree on is wrong by 6.7× against a corrected accounting. **Agreement between
  analytical models is not evidence.** This is doc 44's "compare distributions, not numbers" at the
  level of whole frameworks.

### FLAT **contradicts**

- **Lever (B)'s "decouple so sizes add" is wrong at exactly one joint — and it is our joint.** Marvel's
  10¹⁰ reduction decouples off-chip from on-chip *sequentially*: optimise the outer subproblem, then
  construct the inner space from that optimum. **FLAT's §5.3 is a direct argument that this specific
  cut is unsound when the outer choice is fusion granularity**, because coarsening/refining the
  shared outer loop is *precisely* what changes inner-loop reuse and PE fill. Marvel's cut is legal
  because off-chip tiling and on-chip tiling are (near-)separable; **fusion granularity and
  intra-stage tiling are not**. Recommendation: **keep the decoupling for the off-chip/on-chip cut,
  and explicitly exempt the fusion-granularity axis from it** — co-explore band size with the
  per-stage tiling, and pay the multiplication there. It is a 4-to-6-valued axis; multiplying by it
  is affordable, and FLAT's whole search is affordable *because* that axis is small.
- **"Attention rather than the FFN interior" — the answer is *not* the one the question implies.**
  FLAT does not argue we picked the wrong sub-chain in general. It argues something more specific
  and more useful: **at N ≤ 512, L/A is 12% of the layer and fusion gains are "immaterial"; the
  crossover to attention-dominance is between 512 and 4K; by 16K attention is 79%.** So:
  - At `baseline_768`, R1's FFN interior **is** the right target and FLAT's own numbers say so.
  - At the top of our 64…16384 ladder, it is **not** — attention is ~79% of the layer and an
    unfused L→S→A there is the dominant cost.
  - **Recommendation: do not retarget R1. Add a second resident increment, attention-side, and make
    the ladder rung an explicit input to which increment is worth building** — that is a strictly
    better outcome than either "keep going" or "switch", and it is the first time in this study we
    have had a published crossover point to aim at.
- **Our "packaged vs resident" vocabulary is missing a third state, and FLAT lives in it.** We have
  *packaged* (stages spliced, hand-offs cross L3) and *resident* (hand-offs stay on chip within one
  segment). FLAT is **neither**: one compute resource, one shared outer loop, two inner loops run
  back to back, intermediate never leaving the buffer. Call it **interleaved**. TileFlow already has
  the vocabulary — `Scope: sequential | pipeline | parallel` — and its FLAT test case is a
  `sequential` scope. **Recommendation: adopt the three-way distinction before designing any mapping
  notation**, because *sequential-fused* and *pipelined-fused* have different DMA profiles, different
  fill/drain costs and different per-column budgets, and we currently have one word for both.

### FLAT **reorders**

- **R1's failure is an ordering problem, and FLAT's mechanism is an ordering mechanism. Read doc 31
  §Wall 5 next to FLAT §5.** Doc 31's diagnosis: the three coupled L3 feeds are issued **channel-major**
  (`hidden ×96`, then `w_up`, then `w_down`), the consumer needs **round-major**, and *"no ordering of
  whole channel runs can satisfy R1 — every linear channel-major order starves some consumer, because
  all three feeds are coupled through one compute pipeline."* FLAT's entire contribution is a
  **shared outer loop that forces round-major issue by construction** — the tile index is the outer
  loop variable, and every operand of every stage is indexed by it. Queue item 6c is currently scoped
  as "restore round-major order in `air-dma-to-channel`". **FLAT reframes it: round-major is not a
  repair to apply after hoisting, it is the definition of a fused loop nest.** If our declarative
  pipeline-fusion pass emitted a shared outer loop over FLAT-tiles with per-stage inner loops beneath
  it, channel-major grouping would be unreachable rather than repaired. **Recommendation: before
  implementing 6c as a re-interleaving pass, spend an hour checking whether the newly-landed fusion
  pass can carry the shared-outer-loop shape — a construction that makes the bug unstateable beats a
  scheduling step that undoes it, and doc 44's own tier-1 principle ("construct structural
  constraints into the space rather than filtering after") says the same thing.**
- **Promote TileFlow above FLAT in the read-next list.** It is the closest published artifact to our
  target on four axes at once: it is a **shipped fused mapper with a search**; its legality checker is
  separate from its cost model; its notation has an explicit `pipeline` vs `sequential` scope; and it
  is the **only** fused model in this family validated against RTL (5.4% latency, 6.1% energy) — on a
  4-core machine where **each core has a matrix array *and* a vector array with its own L1**, which is
  an AIE column in all but name. Doc 44's two claims — "no fused mapper" and "nobody validates a fused
  cost model against silicon" — should both be amended, and TileFlow inserted between LoopTree and
  Union.
- **Add a fourth entry to doc 44's provenance note.** Three of five surveyed frameworks had shipped
  code contradicting the paper. **FLAT makes four, and is the worst**: it ships no code at all
  publicly (docs page: "Code Available — Coming soon", untouched since June 2023), its abstract's
  headline speedup and energy numbers are stale v1 figures that appear in no table of the published
  paper, and the private code was found by a third party to omit the cost of the very operation the
  fusion exists to accommodate.
- **One thing to *stop* worrying about.** Doc 44 framed our four walls as "the tooling that would have
  predicted them does not exist". After this read the framing should be sharper and less consoling:
  **FLAT is the closest thing to that tooling, and its own model would not have predicted a single one
  of our four walls** — it has no BD, no channel count, no shim, no per-column budget, and it models
  the non-GEMM stage as free. It *would* have predicted the weight-refetch cost of R1's 64-row band,
  in closed form, from §5.3. That is one wall out of five, and it is the one we did not look for.

### Read next, revised

1. **TileFlow** (MICRO'23, [pku-liang/TileFlow](https://github.com/pku-liang/TileFlow)) — shipped
   fused mapper + checker, RTL-validated, `sequential`/`pipeline`/`parallel` scopes, multi-core with
   per-core matrix+vector arrays. **Now the top of the list.**
2. **FuseMax** (MICRO'24, [FPSG-UIUC/micro24-fusemax-artifact](https://github.com/FPSG-UIUC/micro24-fusemax-artifact))
   — the audit of FLAT, the 1-pass cascade, and a *pipelined/interleaved binding across heterogeneous
   units* that is much closer to our herd structure than FLAT's. Its `flat_validation.csv` is the
   cleanest published statement of what FLAT's mapping space actually contains.
3. **LoopTree** — still the right source for tile inference and retain/recompute encoding.
4. **AccelForge** ([Accelergy-Project/accelforge](https://github.com/Accelergy-Project/accelforge)) —
   the Timeloop/Accelergy line's active successor, MIT-licensed, **pushed 2026-08-10 (two days ago)**.
   Not investigated here; flagged because doc 44's Timeloop findings may already be stale.
5. **Union**, **Ruby**, **Eyexam** — unchanged from doc 44.
