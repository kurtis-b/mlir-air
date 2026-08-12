# 46 — TileFlow (MICRO 2023), researched against primary sources

`[2026-08-12]` Research doc in the series of [39](39-research-llmcompass.md)–[43](43-research-maestro.md)
and [45](45-research-flat.md), synthesised in [44](44-mapping-frameworks-synthesis.md). Doc 44's
`[2026-08-12]` correction names TileFlow as **the fused mapper the survey wrongly said did not exist**,
and says *"Read TileFlow before FLAT."* This is that read.

**Subject.** *TileFlow: A Framework for Modeling Fusion Dataflow via Tree-based Analysis* — Size Zheng,
Siyuan Chen, Siyuan Gao, Liancheng Jia, Guangyu Sun, Runsheng Wang, Yun Liang (Peking University).
MICRO '23, Toronto, 28 Oct – 1 Nov 2023, pp. 1271–1288,
[DOI 10.1145/3613424.3623792](https://dl.acm.org/doi/10.1145/3613424.3623792).
Author PDF: <https://gulang2019.github.io/files/tileflow-micro23.pdf>.
Code: <https://github.com/pku-liang/TileFlow> (MIT), plus a **second required repository**
<https://github.com/KnowingNothing/Domino> (Apache-2.0). Artifact DOI
[10.5281/zenodo.8350955](https://doi.org/10.5281/zenodo.8350955).

**The question this doc answers is different from the other seven.** For Timeloop, MAESTRO, SCALE-Sim,
Accelergy, LLMCompass and FLAT the question was "what ideas can we borrow". For TileFlow it is
**"can we use this, and if not, exactly what do we take"**. The answer is at the bottom; the evidence is
in between.

---

## 0. Provenance first

### 0.1 There is no arXiv preprint

Unlike FLAT (seven arXiv versions, headline numbers stale by v7 — doc 45 §0), TileFlow has **no arXiv
posting at all**. Searches for a preprint return only the ACM DL entry, the authors' pages
([sizezheng.github.io/publications](https://sizezheng.github.io/publications/),
[gulang2019.github.io](https://gulang2019.github.io/)) and the GitHub repo. The author-hosted PDF at
`gulang2019.github.io/files/tileflow-micro23.pdf` is byte-for-byte a camera-ready ACM PDF: it carries
the ACM ISBN 979-8-4007-0329-4, the page range 1271–1288, and the footer
*"Received April 2023; revised August 2023; accepted September 2023"*. **So the paper/paper discrepancy
class that bit three of the previous seven does not apply here.** Everything below is cited to that
camera-ready by section number.

### 0.2 What I did with the code

Cloned `pku-liang/TileFlow` (`--depth 50`, 1.7 MB, HEAD `d72d278`, last push 2024-04-12) and
`KnowingNothing/Domino` (`--depth 1`, 8.6 MB). **Both cloned cleanly and the validation data is
present** — this is not a FLAT situation (doc 45 §0: *"Code Available — Coming soon"* since June 2023,
nothing ever shipped).

Repo health, via the GitHub API: **71 stars, 11 forks, 0 issues ever filed (open or closed), 0 open
PRs, last push 2024-04-12, not archived.** Zero issues in three years is ambiguous — either it works
for everyone or nobody has run it. Given what follows in §7, I lean toward the latter.

**I could not build it here.** The build needs `scons`, `libyaml-cpp-dev`, `libconfig++-dev`,
`libncurses-dev`, `libgpm-dev` (README install line); this machine has `g++`, `cmake` and boost but
none of the other four, and installing them needs `sudo apt`. Per the task's "modify nothing"
constraint I did not install them. **Everything I assert about the code below is read from source, not
from a run**, and is cited by file and line.

Two build traps found by inspection, both real:

- `.gitmodules` pins the Timeloop submodule to **`git@github.com:gulang2019/timeloop.git`** — an SSH
  URL. The README's `git clone --recursive git@github.com:pku-liang/TileFlow.git` therefore fails for
  anyone without a GitHub SSH key. Fixable with `url.https://.insteadOf`, but it is an undocumented
  step-zero.
- That submodule is a **personal fork of `NVlabs/timeloop`, last pushed 2023-04-18** (GitHub API). So
  TileFlow rides a spring-2023 Timeloop — squarely **pre-v4, therefore pre-LoopTree**. The two fused
  models in the world are built on the same infrastructure a year apart and do not share a line.

### 0.3 The artifact is split across two repositories, and the split matters

Artifact appendix §A.3: *"We provide two software repositories for evaluation. The first is TileFlow
main repository, and the second is Domino compiler. … We develop Domino to provide a Python interface
for TileFlow so that we can do experiments easily. **It is not a critical part of TileFlow** because
TileFlow also has a programming interface using configuration files (.yaml format)."*

That parenthetical is misleading, and it is the single most important provenance fact in this doc.
**The `pku-liang/TileFlow` binary searches tile sizes only.** The search over *fusion structure* —
which operator fuses into which, at which memory level, under which scope — lives entirely in Domino
(`python/domino/analysis/fusion.py`, `python/domino/program_ir/ir_builder.py`). §A.5 confirms it: the
validation experiment (Fig. 8) runs from the TileFlow repo; the **dataflow comparison experiment
(Figs. 9–11) runs from `KnowingNothing/Domino/testing/tileflow/test/experiments/`**.

If you take only the repo named in the paper's contribution list, you get a fused *evaluator* with a
tile-size tuner. You do not get the fused mapper.

---

## 1. The tree representation

### 1.1 The central claim, in the authors' words

§1: *"Our insight is that fusion dataflow **is not a perfect polyhedron, but is a tree structure**. So
the modeling analysis should be designed for the tree structure as well."*

§2.3 gives the argument: *"these models fail to model the performance of fusion dataflows because, for
a fusion dataflow, the iteration space is not a perfect polyhedron. Fusing one operator (Op₁) into
another operator (Op₂) is to insert the iteration space of Op₁ into the iteration space Op₂. So after
fusion, the iteration space of the fused workload is not perfectly nested."*

And §4.1 makes the correspondence exact: *"the tree structure can naturally capture the insertion of
one polyhedron into another polyhedron, which corresponds to the fusion of operators. … When an
operator Op₁ is fused into another operator Op₂, Op₁'s iteration space (a polyhedron) is inserted into
the iteration space of Op₂, which is modeled as **inserting a node (for Op₁) as the child to another
node (for Op₂)** in the tree structure."*

This is a genuinely clean idea and it is the paper's real contribution. Imperfect nesting is exactly
what defeats a single-polyhedron formulation; a tree of perfect nests is the minimal generalisation
that survives it. **Every leaf is a perfect loop nest, so all the polyhedral machinery still applies
locally** (§5.1.1, §5.2: *"can be calculated by polyhedron analysis [31, 38, 45]"*), and the tree only
has to supply the composition rules between nests.

### 1.2 Node types, exactly

Three node kinds. From `docs/frontend-syntax.md` (repo) and `include/tileflow/mapping/mapping.hpp`:

| node-type | attributes | meaning |
|---|---|---|
| **`Tile`** | `type: Temporal\|Spatial`, `factors`, `permutation`, `target`, `split`, `bypass`, `profile`, `tag`, `multicast` | a loop nest over exactly **one** child; `target` names the storage level it is bound to |
| **`Scope`** | `type: Sequential\|Sharing\|Parallel\|Pipeline` | composition of **two or more** children; carries no loops |
| **`Op`** | `name` | a leaf: one einsum from the problem spec |

The four `Scope` types are Table 1's **inter-tile primitives**; `Temporal`/`Spatial` on a `Tile` are
Table 1's **intra-tile primitives** (`Tp`/`Sp`). Table 1's own glosses:

- **`Seq`** — *"tiles each occupies all the hardware resources in turns"*
- **`Shar`** — *"tiles share the hardware memory and execute in turns"*
- **`Para`** — *"tiles spatially use different compute and memory units"*
- **`Pipe`** — *"tiles are dependent and execute in a pipeline manner"*

**These four are not four independent things.** Reading the composition rules out of the source, they
are the four points of a 2×2 product — {compute shared in *time* vs partitioned in *space*} ×
{memory footprint *exclusive* vs *additive*}:

| | latency (`src/model/topology.cpp:56-77`, `src/loop-analysis/dm-calculator.cpp:167-169`) | #PE (§5.2) | footprint (§5.2) | working set (`dm-calculator.cpp:160-173`) |
|---|---|---|---|---|
| `Seq` | **Σ** children | max | **max** | chained (evicted between tiles) |
| `Shar` | **Σ** children | max | **Σ** | unioned |
| `Para` | **max** children | **Σ** | **Σ** | unioned |
| `Pipe` | **max** children | **Σ** | **Σ** | chained |

**`Para` and `Pipe` differ in exactly one line of the whole codebase** — `dm-calculator.cpp:160`,
whether the working set is chained between children (Pipe, like Seq) or unioned (Para). They are
identical in latency, PE count and footprint. Note this; §4.4 returns to it, because it is the biggest
single problem with using TileFlow for what we are doing.

### 1.3 What an edge means, and how tiling attaches

An edge is **containment in the iteration space**: child *c* of tile *T* is executed once per iteration
of *T*'s loop nest. §4.2 gives the notation, Eq. (1):

```
T_n = {l¹_n, l²_n, ...} (T¹_{n-1}, T²_{n-1}, ...)
```

*"where {l¹ₙ, l²ₙ, …} is a loop nest over a list of sub-tiles (T¹ₙ₋₁, T²ₙ₋₁, …), forming a tree
structure with this recursive definition."*

So **tiling does not attach to a node — tiling *is* the node.** A `Tile` node's `factors` are the trip
counts of its loops and its `target` is the storage level those loops' data is staged in. There is no
separate "tiling" object. §4.2: *"For loop tiling, the loops in each tile correspond to the tiling
results and thus express the tiling choices naturally."*

Depth in the tree is therefore memory-hierarchy depth, enforced structurally. `SanityChecker`
(`src/mapper/checker.cpp:449-484`) asserts:

- storage levels descend monotonically down the tree, never skip and never reverse
  (`"skip or reverse storage level mapping"`, line 458);
- a `Tile` has **exactly one** child (line 461) — branching happens only at `Scope` nodes;
- a `Spatial` tile's child **must** be a `Temporal` tile (lines 466-471);
- a `Scope` has **more than one** child (line 477);
- at the leaf, all storage levels must have been consumed (line 482).

Plus one rewrite rather than a check: `SpatialScopeSwapper` (`checker.cpp:76-122`) asserts *"A spatial
tile's child cannot be a parallel/pipeline scope"* and, where a Scope sits under a Spatial tile, it
**hoists the Scope above the Spatial tile and replicates the Spatial tile into each branch**. This is
a normalisation pass, not a rejection — the space is made canonical rather than filtered.

### 1.4 A real example from the repo

`tests/cases/13-test-attention/map/tileflow.yaml` — the paper's own "TileFlow dataflow" for attention.
Trimmed to structure, with `?` marking a free variable the mapper will solve for:

```yaml
mapping:
  node-type: Tile           # L2 temporal — the outer loop over the whole fused group
  type: Temporal
  factors: B=? H=? M=? L=?
  permutation: LMHB
  target: L2
  subtree:
  - node-type: Tile         # L2 spatial — spread B/H/M across cores
    type: Spatial
    factors: B=? H=? M=?
    permutation: MHB
    target: L2
    split: 1
    subtree:
    - node-type: Tile       # L1 temporal — the shared inner loop of BOTH stages
      type: Temporal
      factors: M=? L=? B=? H=?
      permutation: HBLM
      target: L1
      subtree:
      - node-type: Scope    # <-- the fusion seam
        type: Sequential
        subtree:
        - node-type: Tile   # stage 1: S = Q x K, reduction dim A
          type: Temporal
          factors: A=?
          target: L1
          bypass: [C]       # C (the intermediate) is NOT written back to L1's parent
          profile: False
          subtree: [ ... Spatial(M,L) -> Temporal(M=1,A=1,L=1)@L0 -> Op(ProduceC) ]
        - node-type: Tile   # stage 2: O = C x V, reduction dim N
          type: Temporal
          factors: N=?
          target: L1
          bypass: [C]
          profile: False
          subtree: [ ... Spatial(M,L) -> Temporal(M=1,L=1,N=1)@L0 -> Op(ProduceO) ]
```

Read the structure: the `L=?` loop appears **above** the Scope, in the shared L1 temporal tile. That is
the fusion. The two stages' own reduction loops (`A` for `QKᵀ`, `N` for `·V`) sit **below** the Scope,
private to each stage. `bypass: [C]` says the intermediate is not spilled. This is FLAT's shared-outer-
loop shape (doc 45 §4, doc 44's `interleaved`) written declaratively — and it is written as **the same
kind of object** as the non-fused version, which is exactly what our `air.pipeline_group` /
`air.pipeline_stage` /`air.staging` triple is trying to be.

§4.1 states the one *semantic* rule that makes this legal: *"when fusing two operators (fuse Op₁ into
Op₂), **only the reduction loops of Op₂ are allowed in the parent tile** … Otherwise, if the reduction
loops of Op₁ appear in parent tile (as outer loops), Op₂ can't start execution until Op₁ has finished.
As a result, the fusion pipeline is inefficient."* **This is doc 31's wall 5 stated as a rule about
loop placement.** Doc 45 §5 already found that FLAT's shared outer loop makes channel-major
unreachable by construction; TileFlow states the general form: *put the consumer's reduction loops
outside, never the producer's.* We can check that mechanically on a `pipeline_group` today.

For comparison, the same repo's `map/flat.yaml` differs only in **where the Scope sits** — above the
L2 spatial tile rather than below the L1 temporal one — and in which loops are shared. That is the
whole encoding of "FLAT vs TileFlow dataflow". The representation is doing real work.

### 1.5 Compared to LoopTree — different ideas, not the same one

Doc 44 lists LoopTree (Timeloop v4, in-tree) as *"models fused chains with `Pipeline`/`Sequential`
nodes"*. **Checked against the LoopTree paper, that characterisation needs correcting.**

LoopTree — Gilbert, Wu, Emer, Sze, [arXiv 2409.13625](https://arxiv.org/abs/2409.13625) /
IEEE TCAS-AI 2024 — is **not a node tree**. Its representation is **polyhedral**: operation tiles and
data accesses are *"constrained by equalities and inequalities containing affine expressions"*, and
analysis is set/relation algebra (ISL), not tree traversal. Its mapping is a **taxonomy of discrete
choices**: partitioned ranks, tile shape, tile processing schedule, retain-recompute (per intermediate
fmap: *"the last rank partitioned to form the retained tile"*), retain-refetch, parallelism, intra-layer
mapping. And critically, **its `sequential`/`pipeline` choice is a single global attribute of the fused
group** — *"We can arrange these tiles to be processed sequentially or in a pipeline"* — not a per-node
scope.

So the two are **different ideas that arrived at overlapping capability**:

| | TileFlow | LoopTree |
|---|---|---|
| representation | tree of perfect nests + 4 scope types | polyhedral sets/relations + choice taxonomy |
| composition | per-`Scope`-node, 4 values, nestable | one global sequential-vs-pipeline flag |
| retention | implicit, via `bypass` + where the Scope sits | **explicit**, per-tensor, per-intermediate-fmap |
| recomputation | LoopTree's Table I: **"Limited"** (*"Only a subset of recomputation choices are supported"*) | "All" |
| search | yes (see §3) | **none** — model only |
| validation | RTL, 5.4% (see §5) | 5 prior architectures, worst case 4%, incl. FLAT at 3.4% |

LoopTree's Table I says of TileFlow: partitioned ranks **"Any"**, recomputation **"Limited"**,
per-intermediate-fmap recomputation **"No"**, per-tensor retention **"No"**. And in prose: *"Among prior
work, **only TileFlow supports an extensive set of partitioned rank choices**."* That is a real
compliment from the competing camp and it is the one axis on which TileFlow is unambiguously ahead.

**The synthesis for us: take TileFlow's tree for *composition*, take LoopTree's per-intermediate
retention encoding for *residency*.** They are complementary and neither subsumes the other. Doc 44
item 2 already wanted LoopTree's retention encoding; nothing here changes that, and TileFlow's tree
does not supply it.

---

## 2. The mapping space

### 2.1 The 3D framing

§4.1: *"We characterize the design space of fusion dataflow as a 3D space composed of three dimensions:
**compute ordering, resource binding, and loop tiling**."*

- **Compute ordering** = the shape of the tree (which op is a child of which, at which level).
- **Resource binding** = the `Scope` type at each branch, plus `Temporal`/`Spatial` on each loop.
- **Loop tiling** = which loops to tile (granularity) and the tile factors.

§2.3 positions this against SET (a resource-allocation-tree predecessor): *"SET's design space is
limited because it uses **DNN layers as the scheduling unit** and only allows pipelining among
mini-batches. As a result, SET's design space is between 2D and 3D. By contrast, TileFlow can express
the full 3D space via the tile-centric notation. **Each layer is split (at any possible dimension, not
limited to mini-batches) into tiles and the scheduling unit is the tile.**"*

**That last sentence is the one that matters for us.** Our composition axis is stages-on-distinct-cores
at tile granularity, which doc 44 "What does not transfer" says Timeloop's spatial axis cannot express
(*"replicates one einsum's nest homogeneously across the mesh"*). TileFlow's `Para`/`Pipe` scope is
precisely the missing construct: **distinct stages, distinct code, distinct cores, at tile
granularity.** No other framework in this survey has it.

### 2.2 Constructed-in vs filtered-after — the two-tier answer, and it matches Timeloop's

Doc 44's reference point is Timeloop's two-tier answer: structure constructed-in, resources rejected
with typed reasons. **TileFlow lands in the same place, by a different mechanism.**

`Checker` (`include/tileflow/mapper/checker.hpp:93-133`) emits exactly **three** constraint classes
(`src/mapper/checker.cpp:317-333`):

1. **`LOOPCOUNT`** — Π(tile factors for dim *d*) == extent(*d*). Perfect factorisation.
   (`ShapeConstraintParser`, `checker.cpp:10-74`.)
2. **`MEM`** — Σ over live tensors of footprint ≤ buffer size, evaluated at each profiled `Temporal`
   tile and each `Sharing` scope. (`MemoryConstraintParser`, `checker.cpp:335-442`.)
3. **`SPATIAL`** — core usage as an **(x, y) pair** ≤ (fanoutX, fanoutY) of that storage level.
   (`ResourceConstraintParser`, `checker.cpp:486-535`.)

**Tier 1 (constructed-in): `LOOPCOUNT`.** In `SymbolTable::init` and `fix_and_update`
(`src/mapper/expr.cpp:269-376`) the loop-count constraint is not *checked* — it **generates the
candidate set**. `get_candidates(lc)` returns the divisors of the remaining loop count; fixing one
factor calls `intersect(entry.candidates_, candidates)` to shrink every co-constrained variable's
candidate set, and when a set collapses to a singleton the variable is fixed outright. `docs/mcts.md`
states it explicitly: *"For loop count constraints, use it to give concrete candidate values easily."*
**A sampled point is perfectly-factorising by construction.** This is Timeloop's index-factorisation
sub-space, arrived at independently.

*Note the same limitation Timeloop has and Ruby fixes: perfect factorisation only. Doc 44's "Read
next" already flags Ruby for our padded tiles; TileFlow does not help there.*

**Tier 1 also: tree structure.** `SanityChecker` + `SpatialScopeSwapper` (§1.3) make the structural
rules either normalisations or asserts that fire once per tree, not per tile-size point. And the
Domino-side plan generator (§3.1) only emits dominator-legal attach points, so no dependence-violating
tree is ever constructed.

**Tier 2 (rejected): `MEM` and `SPATIAL`.** `docs/mcts.md`: *"For other two type of constraints, use
them as **0/1 pruning condition**."* In code, `State::init_factors` (`src/mapper/mapper.cpp:277-307`)
walks a variable's candidate list **from the largest downward, erasing candidates that violate MEM or
SPATIAL, and stopping at the first feasible one** — a monotonicity assumption (bigger factor ⇒ more
memory) that holds for footprint but is asserted nowhere. Anything still infeasible after propagation
sets `failed_`, and `get_next_var()` returns `ERROR_OUT` (`expr.cpp:378-393`), which the environment
scores with a `punish_` reward rather than excluding.

**Where TileFlow is weaker than Timeloop: the failure is a bare bool.** `SymbolTable::failed_` carries
no reason, no level, no class. Doc 44 item 3 wants Timeloop's *typed* failure reasons *"to kill whole
slabs of the space, not just the failing point"*. TileFlow kills the point (plus the monotone tail).
The information to do better is right there — `Constraint` already carries `type_`, `msg` and
`short_msg` naming the storage level — and is simply not fed back into the search.

### 2.3 What is *not* in the space

- **Loop permutation.** `docs/mcts.md` lists the search variables as *"tiling factors; scope type
  (**not realized yet**); permutation (**not realized**)"*. Permutations are written by hand in the
  YAML (`permutation: LMHB`) and never varied. Timeloop searches this as a first-class sub-space;
  TileFlow does not.
- **Bypass.** `bypass: [C]` is hand-written per node. Timeloop's bypass sub-space has no analogue.
- **Scope type, inside TileFlow.** Also "not realized". It *is* searched — by Domino (§3.1).
- **Anything about the architecture.** Arch is fixed input; this is a mapper, not a co-designer.

---

## 3. The search, and how the combinatorics are tamed

### 3.1 What the paper says vs what ships

§6: *"The mapper uses a combination of **genetic algorithm** and Monte Carlo Tree Search (MCTS) in
exploration. … **The genetic algorithm can generate a population of analysis trees** from a set of
randomly sampled initial choices **through crossover and mutation** of the encoded choices. … The
**top-K analysis trees are reserved to produce the next population**."*

**There is no genetic algorithm in either repository's TileFlow path.** I grepped both for
`genetic|crossover|mutat|population`:

- `pku-liang/TileFlow`: **zero hits** outside the word "permutation". The only `Algorithm` subclass is
  `MCTS` (`include/tileflow/mapper/mapper.hpp:185-201`), and the config's `alg: [random, mcts]`
  (`docs/frontend-syntax.md`) maps `random` to `MCTS(timeout, random=true)` — the *same* class with
  random action selection (`src/mapper/mapper.cpp:229-232`).
- `KnowingNothing/Domino`: the only GA is
  `test/python/dac-2023-soc-mapping-v1/evolution_mapper.py` — **a different paper's** SoC mapper, not
  reachable from `testing/tileflow/`.

What actually runs in the TileFlow path is a **two-level sampler**:

1. **Fusion structure.** `IRBuilder.define_fuse` (`Domino/python/domino/program_ir/ir_builder.py:164`)
   calls `generate_fusion_plans(final_tensor, min(2*levels, 7))`
   (`Domino/python/domino/analysis/fusion.py:422-543`), which **exhaustively enumerates** every legal
   plan by DFS over the producer-consumer graph, constrained by dominator relations. Per intermediate
   tensor the choice is `(attach-target tensor, attach level, scope)`; the scope enumeration is
   literally `for scope in ["Sequential", "Sharing", "Pipeline"]` (line 516) — **`Parallel` is not
   enumerated**, consistent with §4.1's *"Para … is only applicable to tiles without data
   dependency"*. Attach levels are restricted to even indices (`if l % 2: continue`, line 514), i.e.
   temporal levels only. The result is dropped into a `CategoricalSpace` sampled by
   `CategoricalRandomPolicy` — **uniform random over the enumerated list** (`policy.py:112-136`).
2. **Tile factors.** Either (a) TileFlow's C++ MCTS, or (b) Domino's
   `AnnealingMutateOneDim` policy (`policy.py:32-110`) — *"`--define_tiling_space` uses **the GA search**
   for tiling factors. If this is not specified, it will use MCTS"*
   (`Domino/testing/tileflow/test/experiments/README.md`, "Notes"). Read the code: it picks a start
   point from history with probability `exp(value − best)`, **mutates exactly one dimension**, and
   round-robins which dimension via a counter. **That is simulated annealing / coordinate descent.
   There is no population and no crossover.**

**Verdict on the discrepancy.** It is of the same class doc 44 found in three of five: the shipped
search is *weaker and simpler* than the paper describes. It does not invalidate the results (a
one-dimension-at-a-time annealer over an exhaustively enumerated structure list is a perfectly
reasonable searcher), but **"GA + MCTS" should not be repeated as a description of this tool**, and
doc 44's correction paragraph should be amended. This is the fourth "shipped code disagrees with the
paper" finding in the series.

### 3.2 The MCTS, precisely

`src/mapper/mapper.cpp:172-307`, `include/tileflow/mapper/mapper.hpp:53-201`:

- **State** = the symbol table: for each tile-factor variable, fixed-with-value or unfixed-with-
  candidate-set.
- **Action** = fix one variable to one candidate. **Variable choice is a heuristic, not a search
  decision**: `get_next_var()` (`expr.cpp:378-393`) returns the unfixed variable with the **fewest
  remaining candidates** — a most-constrained-variable-first ordering straight out of CSP. Value choice
  is UCB (`select_action`, `mapper.cpp:240-261`, `C = 1`).
- **Transition** = `fix_and_update`, which re-propagates the loop-count constraints and re-runs
  `fail_check`.
- **Objective** = a **scalar**, `CYCLE` or `ENERGY`, selected by config; `reward = −log₁₀(value)`
  (`mapper.cpp:135`). Not EDP, not a vector. (Domino's driver layer adds `1e9/EDP` and per-level
  `Utilization_L{0..3}` as selectable metrics — `testing/tileflow/python/tileflow/tuning.py:110-150`.)
- **Rollout** = `n_rollout = 100` random completions, scored by **max**, not mean (`mapper.cpp:192-207`).
  Max-of-rollouts is the right choice for design search (you want the best attainable, not the average)
  and is worth copying.
- **Termination**, three ways (`mapper.cpp:217-238`): `n_iteration = 10000` hard-coded;
  `timeout_` seconds (Mapper default 600, MCTS default 120, config `tileflow-mapper.timeout`); or
  `n_unexplored == 0` → *"MCTS: early exit for exhausting the search space"*. **There is no
  victory-condition termination** of Timeloop's kind (doc 44).
- **Top-K** kept (`topk` config, `Env::insert`), but of *tile-factor tables within one tree*, not of
  trees. The paper's "top-K analysis trees reserved to produce the next population" has no
  implementation.

### 3.3 Space size and search time — the published numbers

All from §7.2, and they are unusually forthcoming (contrast FLAT, doc 45: *"no published space size and
no published search time"*):

| quantity | value |
|---|---|
| distinct **dataflows** (tree structures) per workload | **5,103 – 20,412** |
| tiling space per dataflow | *"each dataflow design further has its own tiling factor space"* (unquantified) |
| tiling search, to converge | **50 rounds × 200 tiling choices**, ~12 s/round → **3.2 – 6.4 min** |
| 3D search, to converge | **< 50 rounds × 20 dataflows/round** |
| 3D search, full | **1–2 days** single-threaded; **< 1 hour** on 56 processes |
| all AE experiments | ~**3 hours** on 112 cores (`Domino/testing/tileflow/test/experiments/README.md`) |
| machine | Intel Xeon Gold 6348 @ 2.6 GHz, single thread for the timed runs |

**Do the arithmetic the paper does not.** 50 rounds × 20 dataflows = **1,000 tree evaluations out of
5,103–20,412 enumerated** — the 3D search visits **5–20% of the structure space**, each visit paying a
full tiling search. That is the honest characterisation: *exhaustive enumeration of structures,
uniform-random sampling of ~10% of them, a real search inside each.*

### 3.4 How the combinatorics are tamed — the actual answer

Four mechanisms, in descending order of leverage:

1. **The structure space is small because it is defined over *attach points*, not over schedules.**
   `generate_fusion_plans` enumerates, per intermediate tensor, `(where, which level, which scope)`.
   For a 3-op chain over 3–4 memory levels that is thousands, not billions. **The fusion decision is
   deliberately made a small enum and everything else is pushed into tiling.** This is precisely doc
   44's reduction move #1 (decouple so sizes add) applied at the joint doc 45 says is *not*
   separable — see §8, because this is the one place where TileFlow and FLAT directly contradict.
2. **Constraint propagation shrinks the tiling space before search touches it.** Divisor-set candidate
   generation + `intersect` on every fix + singleton collapse. This is why an MCTS over ~10 variables
   with ~10 candidates each is tractable at all.
3. **Most-constrained-variable-first ordering.** Free, and it makes the MCTS tree shallow where it
   matters.
4. **Analytical evaluation, no simulation.** One evaluation is a tree traversal computing set
   differences at *time-step boundaries only* — §5.1.2: *"In implementation, there is no need to fully
   unroll all the time steps. Thanks to the regularity of DNN workloads, we only need to consider time
   step boundaries."* ~12 s / 200 candidates ⇒ **~60 ms per full fused-mapping evaluation**. That is
   the number that makes any search affordable, and it is the same argument doc 44 item 2 makes for
   static bandwidth back-solve.

---

## 4. Legality and resource modelling

### 4.1 The three resources, and which are cliffs

| resource | where | cliff or slope | notes |
|---|---|---|---|
| **buffer capacity** | `MEM` constraint | **cliff** — hard reject | Σ live footprints ≤ size, per node |
| **spatial fanout (core count)** | `SPATIAL` constraint | **cliff** — hard reject | (x,y) pair ≤ (fanoutX, fanoutY) |
| **bandwidth** | *not a constraint at all* | **slope** | enters only the latency formula |

**Bandwidth is deliberately not a legality condition** — there is no bandwidth constraint class in
`Checker`. It enters only through §5.3's latency:

```
Lat(T_n) = max{ DM_load/BW_n ,  Σ or max over children ,  DM_store/BW_n }
```

and through Timeloop's `BufferLevel::Evaluate`, which TileFlow calls with **`break_on_failure = false`
everywhere** (`src/loop-analysis/dm-calculator.cpp:25`, `src/model/topology.cpp:93`). Timeloop's
internal slowdown therefore prices over-subscription instead of rejecting it.

**This is exactly doc 44's adopted position — capacity is the cliff, bandwidth is the slope — reached
independently by a fused mapper.** Two frameworks converging on it from opposite directions is about
as strong a confirmation as this literature offers.

### 4.2 The bottleneck instrument already exists, and it names the level

`dm-calculator.cpp:209` computes, per node, per storage level:

```cpp
double slow_down = storage_level->Cycles() / (0.0 + ret.cycle_);
...
data_movement["SlowDown"] = std::max(data_movement["SlowDown"], slow_down);
```

and `docs/tileflow-metrics.md` documents it: *"`SlowDown` >= 1: the slowdown of this level compared to
the child level. **> 1 slowdown indicates this level is bottleneck compared to the child levels.**"*
It is emitted per level into the output CSV alongside `CapUtil`, `SpatialUtil`, and per-tensor
`Read`/`Update`/`Fill`. A real sample from the repo
(`tests/cases/07-test-fusion-attention/result/attention-fused.csv`):

```
Cycle,2490368
L2::SlowDown,1.06645
L2::CapUtil,8.64e-06
L2::SpatialUtil,1
L2::Read::Q,393216
L2::Read::K,786432
...
```

§7.5 uses it as a design instrument, sweeping L1 bandwidth 1→1200 GB/s in 1 GB/s steps and reading off
*"the suitable L1 bandwidth is the minimal value that makes L1 slow-down as 1"* (Fig. 14: 96 GB/s for
Fused-Layer and ISOS; 1080 GB/s for TileFlow's dataflow on CC1, 720 on CC2).

**This is doc 44 item 4 — "latency = max over per-resource isolated cycles and the argmax NAMES the
offending resource" — shipped, working, and demonstrated on a fused chain.** It is also the answer to
"which stage is the bottleneck" that doc 38 found iron approximating by hand with truncated binaries.
The mechanism is ~20 lines. It is the single most directly transferable thing in the repository.

The reported utilisations are also nicely normalised: `Mapper::report_csv`
(`src/mapper/mapper.cpp:88-101`) prints, for every MEM constraint, `used/limit`, and for every SPATIAL
constraint, `(usage.x·usage.y)/(limit.x·limit.y)`. **Demand printed beside budget** — MAESTRO's
refinement (doc 44), also shipped.

### 4.3 Can it express a per-column, whole-segment channel-count budget?

**Structurally yes; semantically no. This is the most important answer in the doc, so here it is in
detail.**

The `SPATIAL` constraint is a **cardinality** resource, not a byte-rate, and — decisively — it is
**accumulated across a fused group with a scope-dependent combinator**
(`src/mapper/checker.cpp:506-519`):

```cpp
void ResourceConstraintParser::visitScope(const ScopeNode* node) {
    std::vector<std::shared_ptr<ResourceExpr>> exprs;
    for (auto child: node->get_children()) { child->accept(this); exprs.push_back(core_usage_); }
    auto scope_type = node->get_scope_type();
    core_usage_ = (scope_type == ScopeNode::Sequential ||
                   scope_type == ScopeNode::Sharing)? Op::max(exprs) : Op::sum(exprs);
    add_constraint(node);
}
```

and at a `Spatial` tile (`checker.cpp:486-504`) `core_usage_` is the (x,y) pair of products of spatial
loop extents. `add_constraint` emits `core_usage_ <= pair(fanoutX[level], fanoutY[level])`.

**Read that again: for a `Pipe` or `Para` scope, the resource demands of all stages are *summed* and
checked against one budget.** That is a *cardinality resource shared across a whole fused group* — the
exact shape doc 44 says MAESTRO could not do (*"its budget is a byte-rate; ours is a channel count. A
straight port catches a per-column byte-rate overrun and still misses a 2-MM2S violation"*). **TileFlow
has the cardinality mechanism MAESTRO lacks.** It is ~30 lines: a visitor, a pair-typed expression, a
max/sum combinator keyed on scope, and a per-level budget.

**Now the three reasons it is not our budget as it stands:**

1. **The counted quantity is hard-wired.** `core_usage_` is always *the product of spatial loop
   extents*. There is no way to say "this node also consumes 1 shim MM2S channel". To count DMA
   channels you would add a second `ResourceExpr` accumulator with its own per-node demand — a
   mechanical change to a 50-line file, but a change to TileFlow's source, not to its input.
2. **The budget is per storage *level*, not per *column*.** `fanoutX_map` / `fanoutY_map` come from
   Timeloop's `ArchProperties::FanoutX()` (`src/mapping/parser.cpp:325-326`), which is a single (x,y)
   for the whole level, applied uniformly to every instance. Doc 44 already flagged this for Timeloop:
   *"there is no shared or global resource budget anywhere … so a per-column, per-segment channel
   budget is unstateable as such."* TileFlow inherits it verbatim. **Our budget is per-column and
   asymmetric across columns; TileFlow's is one number applied everywhere.**
3. **It is a cliff, and ours must be a slope.** `TILEFLOW_ASSERT` is
   `if(!(cond)) { std::cerr << …; exit(1); }` (`include/tileflow/common.hpp:7`). A violated SPATIAL
   constraint on a user-supplied mapping **terminates the process**. Inside the MCTS it is softer
   (`failed_` → `ERROR_OUT` → `punish_` reward), but it is still exclusion, not pricing. Doc 44's
   settled position — over-subscription is legal-but-degraded and must be *modelled as degraded* —
   requires the opposite. **Bandwidth is a slope here; cardinality is a cliff. We need cardinality to
   be a slope too, because AIR packet-multiplexes rather than failing.**

**So the transfer is: adopt the mechanism (pair-typed cardinality accumulated up the tree with
max-for-time / sum-for-space), reject the policy (hard reject), and generalise the budget (per-column
vector, not one scalar per level).** Point 3 is the interesting one — TileFlow gets the cliff/slope
split right for bandwidth and wrong (for our machine) for cardinality, and the reason is that on their
machine exceeding core count is genuinely impossible whereas on ours exceeding channel count is merely
slow. **That distinction is a property of our hardware, not of the modelling, and no upstream framework
will make it for us.**

### 4.4 Inter-core communication — the real gap

Two mechanisms, both weak for our purposes.

**(a) Transfers between stages go through the least common ancestor.** §5.1.2: *"If level X memory
can't move data to level Y memory directly (**which is common in DNN accelerators [36]**), then we need
to move the data through their least common ancestor tile (Tile 0). Otherwise, we can directly move
data between Tile 1 and Tile 2 and record the data movement volume between the memory of level X and
level Y."* Implemented in `Checker::add_access_pattern` (`checker.cpp:205-290`), which walks
producer and consumer paths to the root and finds the common node. This is correct and it is the right
abstraction — **an AIE core-to-memtile-to-core hop is exactly an LCA transfer** — but which case
applies is decided by the *architecture topology*, not by anything the mapping can choose. There is no
"route this through the memtile vs through DRAM" decision variable.

**(b) Timeloop link transfers.** Inherited and enabled unconditionally
(`src/loop-analysis/nest-analysis.cpp:904`: `const int enable_link_transfer = true;`). These are
nearest-neighbour spatial transfers of the *same* tensor between PEs at one level — a systolic
mechanism. It is a rough analogue of AIE's shared-memory neighbour access, but it is **derived from
spatial deltas, not schedulable**, and it does not model AIE cascade.

**And the load-bearing omission: `Pipeline` has no fill or drain.** From §1.2's table, `Pipe` latency
is `max` over children, full stop. There is no pipeline depth, no ramp-up, no ramp-down, no rate-
matching penalty between unequal stages. For a fused group of `S` stages executed `T` times under a
parent temporal loop, the true latency is ≈ `T·max(stage) + (S−1)·max(stage)`; TileFlow reports
`T·max(stage)`. **The error is (S−1)/T, which is small for the paper's workloads and large for ours** —
our stage counts are 2–4 and our per-stage trip counts at `baseline_768` are small.

This is not my inference alone. **COMET** (Negi, Singhal et al.,
[arXiv 2509.00599](https://arxiv.org/abs/2509.00599), 2025), a direct successor, states it: prior
frameworks including TileFlow *"do not account for data staging inefficiencies — such as the data
transfer time from a parent memory to a child memory when the compute has not started (**ramp-up
phase**)"*, and reports that *"COMET produces higher latency estimates"* than TileFlow specifically
because it *"includes ramp-up and ramp-down effects"* and *"explicitly model[s] operation-level
dependencies"*.

COMET adds a second charge: *"**TileFlow does not account for intermediate tensor reuse when operations
are fused at certain levels** — a limitation acknowledged in Section 7.1 of the TileFlow paper."* That
acknowledgement is real; §7.1 reads: *"TileFlow tends to **over-estimate data movement volume** … for
them because **it assumes data replacement happens for every outer iteration**. But in real accelerator,
small tiles may not cause data replacement."* Small tiles are our regime.

**COMET's third contribution is the one that names our machine**: it models *"explicit collective
operations"* and *"deployment across spatially distributed compute clusters"* with *"latency and energy
cost models accounting for both GEMM and non-GEMM operation dependencies"*. A per-column resource on a
distributed cluster is closer to COMET's frame than TileFlow's. **COMET should go on the read-next
list.**

---

## 5. The RTL validation — what it actually is

### 5.1 What the paper claims

§7.1. Two validations:

- **Against Timeloop**: single-operator matmul, **1,152 enumerated mappings**, R² = 0.999 for cycles,
  0.1% mean absolute error for energy. *"we use a single operator workload because Timeloop doesn't
  support multi-operators or fusion."* Honest and narrow.
- **Against RTL**: *"For comparison with real accelerator, **we use self-attention [10]**. We program
  highly optimized **fusion kernels** for our accelerator in assembly and enumerate **131 different
  mappings** (by changing tiling factors and shapes). … The average error of TileFlow in absolute value
  is **5.4%**, while the average error of graph-based method is 48.8%. … [energy] average error in
  absolute value is **6.1%**."*

The accelerator: *"We implement the accelerator in Chisel to generate Verilog RTL. The accelerator has
**four cores**. Each core has **two PE arrays: one for matrix multiplication (16 × 16) and the other for
vector computation (16 × 3)**. The on-chip buffer size is **384 KB per core**. The DRAM bandwidth is
25.6 GB/s. The word width is 16 bits. The RTL is then synthesized using Cadence Genus … our accelerator
area is 7.84 mm² under TSMC 22 nm … frequency is 400 MHz. … we use **Verilator (version 4.0)** to
simulate the RTL and binary."*

### 5.2 What the artifact contains

`AE/validation/accelerator/`. Everything needed is present: `data/data.pkl` (**131 tuples** of
`(shape, cycle)`), `data/io_data.csv` (**131 rows** of `mem_to_buf, buf_to_mem, buf_to_reg, reg_to_buf`
counts), `arch/arch.yaml`, `prob/prob.yaml`, `map/map.yaml`, `map/map-gemm.yaml`, `validation.py`,
`readme.md`. `readme.md`: *"`data/`: **RTL simulation result** of a systolic based hardware."*

I loaded `data.pkl`. It is a list of 131 12-tuples with real cycle counts, e.g.
`((512, 512, 64, 16, 512, 16, 512, 64, 512, 16, 64, 16), 542974)`. Shapes vary M ∈ {512, 1024},
K ∈ {48, 64}, L = 512, N ∈ {48, 64}, micro-tiles ∈ {16, 32, …, 256}. **This is genuine data and it is
genuinely present.** Set against FLAT (doc 45: fused path validated against nothing, no code ever
released) and Accelergy (doc 40: "95%" is post-layout simulation) and LLMCompass (doc 39: 4.1% is a
ratio of two 12-term sums), **TileFlow's validation is the best in this survey of eight.** Say that
plainly before the criticisms.

Now the criticisms, all four verifiable from the artifact.

### 5.3 Finding 1 — there is no softmax in the validated workload

**`AE/validation/accelerator/prob/prob.yaml` defines exactly two operators**:

```yaml
ops:
- name: GEMM1     #  C = A x B     dimensions [M,L,K]
- name: GEMM2     #  E = C x D     dimensions [M,L,N]
```

No softmax. No `max`, `sub`, `exp`, `sum`, `div`. The workload the RTL was compared against is a
**back-to-back matmul chain**. The shapes are attention-shaped (M = 512 seq, K = 64 head-dim, L = 512
seq, N = 64) — it is `QKᵀ` then `·V` — but **the softmax between them is absent from the model.**

The authors say so themselves. `tests/cases/00-validation/02-attention/topk-analysis.py:31`:

```python
filename = 'data/No_Softmax/data.pkl'
```

**The dataset directory is named `No_Softmax`.** That is not my reading of the code; that is the
authors' own filename.

**This is the same hole FuseMax found in FLAT** (doc 45: *"the codebase does not model the softmax cost
at all — worth 6.7× iso-area"*), in a different framework, found a different way. §7.2 shows TileFlow's
*model* can express softmax — *"we need to expand it into five small operators (max, sub, exp, sum,
div)"* — so this is not an expressiveness gap. **It is a validation gap: the exploration results (§7.2,
7.3) include softmax; the RTL validation (§7.1) does not.** The 5.4% number therefore certifies the
two-GEMM fused chain and says nothing about the accuracy of the five-op softmax expansion, which is
where the vector-unit modelling and the non-GEMM dependency structure live — and which is exactly what
COMET says TileFlow gets wrong (§4.4).

*Inference, marked as such: since `validation.py` computes `flops = N·K·L + M·N·L` (two GEMMs only) and
the mean absolute error is 5.4%, either the RTL kernel also omitted softmax or softmax was free. Given
the directory name I believe the former, but I cannot prove it from the artifact.*

### 5.4 Finding 2 — the validated architecture is one core, not four

`AE/validation/accelerator/arch/arch.yaml`:

```yaml
- name: MainMemory   class: DRAM     read_bandwidth: 4
  - name: Cache      class: SRAM     read_bandwidth: 32, block_size: 16384, depth: 8
    - name: RegFile[0..255]  class: regfile  meshX: 16, meshY: 16
    - name: mac[0..255]      class: intmac   meshX: 16, meshY: 16
```

**256 MACs in a 16×16 mesh, one shared SRAM, one DRAM.** That is *one* of §7.1's four cores. There is
no second core, no vector array (16×3), no core-to-core structure. The mapping confirms it:
`factors: M=SX K=SY` with `SX = SY = 16` (`validation.py`).

So **the 5.4% figure validates a single 16×16 array running a fused 2-GEMM chain** — not the four-core
machine, not the vector unit, and not any `Para`/`Pipe` scope across cores. Checking `map/map.yaml`:
the fusion seam is `node-type: Scope, type: Sequential`. **The validated scope is `Seq`.** `Pipe` and
`Para` — the two scopes that carry our entire question — appear nowhere in the RTL-validated path.

*That is the finding that most changes the answer for us. The mechanism we would be adopting TileFlow
for is the one part of it that was never checked against silicon.*

Two smaller mismatches, noted for completeness: Table 4's `Edge` row (32×32 PEs, 4 cores, **L1: 4 MB**,
DRAM 60 GB/s) matches neither §7.1's RTL machine (384 KB/core, 25.6 GB/s) nor the repo's
`tests/cases/13-test-attention/arch/edge.yaml` (L1 = 64 KB × 4) nor Domino's `get_edge_small()`
(L1 = 2000 KB × 4, `testing/tileflow/python/tileflow/accelerator.py:8-27`). And the artifact's DRAM
`read_bandwidth: 4` (words/cycle) at 400 MHz × 16-bit is 3.2 GB/s against a stated 25.6 GB/s — an 8×
gap I **could not reconcile** and flag as unresolved rather than assert.

### 5.5 Finding 3 — the 6.1% energy figure is not against measured energy

`AE/validation/accelerator/validation.py`. The "real" energy — the *target* of the comparison — is
**synthesised in the validation script** from RTL-counted movement events:

```python
reg_energy = 249.643;  cache_energy = 13.8392;  memory_energy = 200
energy_estimation_table = {
    'mem_to_buf': memory_energy/16 + cache_energy/16,
    'buf_to_mem': memory_energy/16 + cache_energy/16,
    'buf_to_reg': cache_energy/16 + reg_energy/(256*16),
    'reg_to_buf': cache_energy/16 + reg_energy/(256*16),
    'flops':      1 + reg_energy/256*4,
}
ret['energy'] = Σ_k energy_estimation_table[k] * ret[k]   # ret[k] from io_data.csv (RTL counts)
ret['energy'] *= 1.21                                      # <-- undocumented
```

Then `analyze(ret, 'Energy', 'energy')` compares TileFlow's own modelled `Energy` against that.

So: **TileFlow's analytical energy vs. a hand-written analytical energy applied to RTL-measured access
counts, with a 1.21 multiplier on the reference side that appears in no paper text and carries no
comment.** What is really being validated is *data-movement counts* — which is a fine and useful thing
to validate — but "6.1% energy error against real hardware" over-claims it. This is doc 40's Accelergy
finding in a new costume: an energy number whose reference is itself a model.

(The script also computes `ret['estimated_energy']` from TileFlow's counts × a second energy table,
and then never uses it. Dead code.)

### 5.6 Finding 4 — the 48.8% strawman is three lines in the same script

The "graph-based method [72]" that scores 48.8% is not a third-party tool. It is constructed in
`validation.py`:

```python
first_gemm_cycle  = run_gemm(M, L, K, MO, LO, KO, SX, SY)   # TileFlow, op 1 alone
second_gemm_cycle = run_gemm(M, N, L, MO, NO, LO, SX, SY)   # TileFlow, op 2 alone
ret['graph-based'] = first_gemm_cycle + second_gemm_cycle - (M*L)/4
```

Sum of two independently-modelled GEMMs minus `mid_tensor/4`. **That is the authors' own reconstruction
of the class of prior work, not a run of anyone's released code.** The 9× accuracy claim is therefore
a claim about a strawman the authors built. TileFlow's own 5.4% is unaffected — see next — but the
comparison is not evidence about published tools.

### 5.7 What survives, and it is not nothing

`analyze()` computes `np.mean(np.abs((pred − target)/target))` — a genuine **per-point MAPE**, not a
correlation, not a ratio of sums, and — checking the code path — **not regression-rescaled**
(`LinearRegression` is fitted and its score printed, but `df[metric]` is only overwritten when
`len(metrics) > 1`, and both calls pass a single string). So:

> **TileFlow's cycle model predicts RTL cycle counts for a fused two-GEMM chain on a single 16×16
> array, over 131 real mappings, to 5.4% mean absolute error, unfitted.**

That statement is true, it is reproducible from the artifact, and it is a better-supported claim than
anything in the other seven documents. **The over-claims are in the framing — "self-attention", "four
cores", "energy" — not in the arithmetic.**

---

## 6. The target machine vs an AIE column

Doc 44's correction calls the evaluation machine *"an AIE column in all but name."* Having read both
the paper and the arch YAMLs, **that is more true of the paper's prose than of the modelled machine,
and the differences are load-bearing.**

### 6.1 What actually matches

| | §7.1 RTL machine | Domino `get_edge_small()` | NPU2 (AIE2P) column |
|---|---|---|---|
| compute cores | 4 | 4 (`Buffer` instance=4) | 4 rows |
| per-core matrix engine | 16×16 | 16×16 (1024 MACs / 4) | AIE vector unit, 2D MAC |
| per-core vector engine | **16×3, separate array** | **absent** | same VLIW core, not a separate array |
| per-core local memory | 384 KB | 2 MB "L1" | 64 KB L1 data memory |
| shared on-chip level | none (DRAM above) | none in Edge; Cloud has 40 MB L2 per 4 clusters | **512 KB memtile per column** |
| off-chip | 25.6 GB/s DRAM | "L2" = DRAM | shim → DDR |
| word width | 16 bit | 16 bit | bf16/int8 |

The **tree shape** — DRAM → shared SRAM → per-core SRAM → registers → MACs, with per-level instance
counts — is exactly Timeloop's arch model and it maps onto NPU2 without strain: `L3=DDR`,
`L2=memtile ×8`, `L1=core data mem ×32`, `L0=registers`. **Expressing NPU2's *memory hierarchy* in
TileFlow's arch YAML is a half-hour job.** Domino's `get_cloud_small()` already builds a 4-cluster ×
16-sub-core tree with a real shared L2, which is structurally our 8 columns × 4 rows with a memtile.

### 6.2 What does not match, in descending order of severity

1. **No channel cardinality.** A Timeloop storage level has `read_bandwidth` / `write_bandwidth`
   (words/cycle) and `instances`. **It has no notion of "this level is reached through exactly 2 DMA
   channels."** Our hardest constraint — *two shim MM2S per column, budgeted across the whole
   segment* — has no field to live in. The `SPATIAL` cardinality mechanism (§4.3) is the closest
   thing and it counts cores, not channels.
2. **No silent-degradation model.** Exceeding fanout is fatal (`exit(1)`). Exceeding our channel
   budget is *slow*. There is no construct for "over-subscribe and pay".
3. **Segment/residency scope has no representation.** Our residency holds only within one segment;
   a `Scope` node in TileFlow has no lifetime attribute. The closest analogue is which storage level
   the intermediate's tile is `target`ed at plus `bypass`, and LoopTree's per-intermediate retention
   encoding is strictly better here (§1.5).
4. **Homogeneous fanout.** `fanoutX_map` is one (x,y) per level. Our columns are not interchangeable
   (shim placement, memtile pressure, and which column owns which stage all differ).
5. **No explicit DMA.** No buffer descriptors, no strides, no dimension count. Doc 44 item 5 insists
   *"a `dma_transfer` is a function of `(n_words, n_dims, stride)`"* precisely because of our
   BD-stride walls (doc [24](24-phase-h10-non-constant-bd-offsets.md), and the
   `air-to-aie` walls memo). TileFlow counts words. **It cannot see either of our aircc-only walls.**
6. **No fill/drain** (§4.4). Our pipeline groups are short and shallow; this error is ours, not theirs.
7. **The vector unit is a separate array in the paper and absent from every arch YAML I found.** On
   AIE the vector unit *is* the core. Softmax on our machine competes with matmul for the same issue
   slots; on their machine it does not. Combined with §5.3, **the softmax cost model is unvalidated and
   its hardware assumption is wrong for us.**

**Net.** The machine is a four-core cluster with per-core scratchpads over DRAM. That is a *plausible
abstraction* of one AIE column and the arch model would accept a description of NPU2. It is **not**
"an AIE column in all but name": it lacks the memtile in the validated config, the channel cardinality
everywhere, the DMA descriptor entirely, and the vector unit in every YAML. Doc 44's phrasing should be
softened to: *"a four-core cluster with per-core matrix arrays and a per-core scratchpad — the closest
published evaluation target in this survey to one AIE column, and still missing the memtile, the DMA
channel budget and the descriptor."*

---

## 7. Usability — could we drive it from our AIR builders?

### 7.1 The interfaces that exist

`tileflow arch/*.yaml prob/*.yaml map/*.yaml [config.yaml]` — order-independent, all YAML, output a CSV
of `metric,value`. Four optional sections: `check` (toggle `mem`/`loopcount`/`spatial` checks),
`tileflow-mapper` (`alg`, `timeout`, `topk`), `macro` (named integer constants usable in both `factors`
and `problem.instance`), `output`, `verbose`. **Replacing any tile factor with a bare string makes it a
free variable** (`docs/frontend-syntax.md`: *"user can simply replace the number in the specification
for tile factors with arbitrary string"*). That is a genuinely good interface for a mapper.

And `main.cpp:75-88` accepts an **Accelergy ERT and ART** in the input, *"replacing internal energy
model"*. **That is the seam for doc 44 item 5** — our measured `(component, action, arguments) → cost`
table can be handed to TileFlow as an ERT without touching its source.

### 7.2 What it would actually take

**The problem spec is a Timeloop einsum, and ours is not.** `prob.yaml` requires, per op, `dimensions`,
`data-spaces` with explicit index `projection`s, `read-write` flags, and `ins`/`out`. Our builders emit
`memref`s, `air.launch`/`air.segment`/`air.herd` and now `air.pipeline_group`. **Emitting einsum
projections from an AIR module is a real translation, not a serialisation** — and for anything with a
non-affine or data-dependent access it is impossible. For our matmuls, norms and elementwise ops it is
mechanical. For softmax it is the five-op expansion the paper does by hand.

Concretely, driving TileFlow from AIR needs:

| # | work | size | risk |
|---|---|---|---|
| 1 | NPU2 arch YAML (DDR → memtile ×8 → core L1 ×32 → regs) | hours | low |
| 2 | AIR → TileFlow `prob.yaml` einsum emitter | ~1 week | **medium** — softmax expansion, non-affine ops |
| 3 | AIR `pipeline_group`/`stage` → TileFlow mapping tree emitter | days | low — the shapes correspond |
| 4 | TileFlow mapping tree → AIR builder parameters (the *return* path) | ~1 week | **high** — this is a code generator |
| 5 | source patch: per-column channel cardinality as a 2nd `ResourceExpr` | days | medium |
| 6 | source patch: cardinality over-subscription priced as a slope, not `exit(1)` | days | medium |
| 7 | source patch: `Pipe` fill/drain | days | medium |
| 8 | build the whole thing (submodule SSH URL, four apt packages, static scons build of a 2023 Timeloop fork) | hours–days | **medium** — unmaintained since 2024-04 |
| 9 | wire our measured cost table in as an ERT | days | low |

**And the killer, which is operational rather than architectural: `TILEFLOW_ASSERT` calls `exit(1)`.**
Any illegal candidate kills the process. Domino works around this by forking a subprocess per candidate
with a 100-second timeout and catching non-zero exit as `{"status_ok": False}`
(`testing/tileflow/python/tileflow/tuning.py:88-113`). Our search would have to do the same. At ~60 ms
of useful work per evaluation, process-per-candidate is 10–50× overhead — survivable, but it means
TileFlow is a *batch oracle*, not a library.

### 7.3 The straight answer on usability

**Not usable directly.** Items 5, 6 and 7 are patches to a C++ codebase whose upstream is an
unmaintained personal fork of a three-year-old Timeloop, with zero issues filed and no commits since
April 2024. Item 4 — turning a TileFlow tree back into AIR builder parameters — is the same amount of
work whether we host the search in TileFlow or in our own code. And items 5–7 are precisely the parts
that encode what is *different* about our machine, so we would be maintaining a fork whose delta is the
entire reason we adopted it.

**The half that *is* usable, at low cost, is the evaluator as an oracle.** Steps 1, 2, 8 and 9 with
none of 4–7 gives us: a second opinion on data-movement volume for a fused group, in bytes, per
memory level, cross-checkable against our measured DMA counters. That is a **falsification instrument
for our own cost model** and it is worth having for exactly that. It is not worth having as a mapper.

---

## 8. Where TileFlow and FLAT directly contradict, and who is right

Doc 44's reduction move #1 is "decouple the subproblems so sizes add instead of multiply," with doc 45's
exemption: **FLAT §5.3 says fusion granularity and intra-stage tiling are not separable, so let that
one joint multiply.**

**TileFlow decouples exactly that joint.** Structure (fusion granularity, attach level, scope) is
enumerated and sampled *outside*; tiling is searched *inside*, conditioned on the sampled structure.
Sizes add: `|structures| + |tiles per structure|`, evaluated `|sampled structures| × |tiling search|`.

Who is right? **Both, and the resolution is in TileFlow's own §7.3, which reads as a direct rebuttal of
FLAT:**

> *"Compared to FLAT, TileFlow achieves similar or better performance at a lower cost of on-chip
> memory. The reason is that TileFlow **tiles all the dimensions** of all the three operators … **This
> tiling strategy is not explored by FLAT because FLAT does not tile column dimension of S, L, and A.**
> FLAT requires at least one full row of intermediate data or output data to be staged in on-chip
> buffer. Each row of K and L (the length is 1024) has to be placed in L1 buffer, while TileFlow tiles
> the column dimension … the searched tiling factor for column dimension is 64, so only a sub-row of
> size 64 will be placed in L1 buffer. As a result, TileFlow can reduce the L1 usage, **leading to an
> order of magnitude lower L1 usage**."

Table 7c: with a memory limit, FLAT-MGran and FLAT-BGran are **OOM**; FLAT-HGran needs 4.10 MB L1 and
32.77 MB L2; **TileFlow needs 0.05 MB L1 and 20.48 MB L2** — 82× less L1 — at 16.78 vs 14.68 ×10⁶
cycles (14% *slower*, note). And §7.5's summary: *"finer tiling granularity is suitable for
memory-limited scenarios. Different granularities can achieve similar performance when on-chip memory
resource is abundant. **Tiling exploration can always improve performance compared to fixed tiling
factors.**"*

**The reconciliation.** FLAT's non-separability argument is about *granularity as a four-valued enum*
`{M,B,H,R}` — an enum defined in terms of *which named dimensions are tiled at all*. TileFlow abolishes
the enum: **every dimension is tileable, so "granularity" ceases to be a separate axis and becomes a
region of the tiling space.** With granularity dissolved into tiling, there is no joint left to
separate. FLAT is right that you cannot fix `{M,B,H,R}` first and then tile; TileFlow is right that you
should not have had `{M,B,H,R}` as an axis in the first place.

**And Table 7's second row is the harder lesson.** Comparing part (a) — fixed tiling — to part (b) —
searched tiling: FLAT-BGran goes from 151.00 to **8.39** ×10⁶ cycles, an **18× improvement from tiling
search alone, on a dataflow structure that was unchanged.** Under a fixed tiling, granularity looked
like it mattered enormously (335 vs 151 vs 67 vs 18.87); with tiling searched, three of the four
granularities **tie exactly at 8.39**. *The apparent importance of the structure axis was an artifact of
under-searching the tiling axis.* That is a direct warning to us: **any conclusion we draw about
composition (packaged vs resident vs interleaved) from hand-picked tile sizes may be an artifact of the
tile sizes.** Doc 44 already has the Timeloop version of this warning (480,000 mappings within 5% of
peak varying 19× in energy); this is the fused version and it is sharper.

**Amendment to doc 44's plan, therefore:** the fusion-granularity/intra-stage-tiling joint stays
non-separable *as FLAT parameterises it*, and dissolves *as TileFlow parameterises it*. **Prefer
TileFlow's parameterisation.** Do not carry a granularity enum; carry a per-dimension tileability flag
and let the tiling search find the granularity. That is a genuine reordering of doc 44 §"How to make
the space small" item 1's exemption.

---

## 9. Two more things worth stealing

**Max-of-rollouts, not mean.** `MCTS::rollout` (`mapper.cpp:192-207`) does 100 random completions and
back-propagates the **maximum**, with the averaging line commented out. For design search that is
correct — you want the best reachable point, not the expected one — and it is a one-line difference
that most textbook MCTS gets wrong for this application.

**`−log₁₀(objective)` as reward, with an adaptive punishment floor.** `mapper.cpp:135-147`:
`reward = −log₁₀(value)`, and infeasible states get `punish_`, which self-adjusts:
`if (punish_ + 2 > reward.reward) punish_ = reward.reward − 2;` — the punishment tracks two decades
below the best reward seen. Log-scaling matters because cycle counts across a fused mapping space span
5+ orders of magnitude (Table 7: 7.67 to 335.54 ×10⁶ on one workload) and a linear reward makes UCB
useless. **We will hit the same dynamic range.**

---

## Comparable summary

- **Data-space representation.** Timeloop's einsum, **extended to a list of operators with explicit
  dataflow edges**. `problem: {io: {ins, outs}, dimensions, instance, ops: [...]}` where each op is a
  Timeloop problem shape (`dimensions`, `data-spaces` with index `projection`s, `read-write`) plus two
  new fields, **`ins` and `out`**, naming the tensors it consumes and produces. Dimensions are shared
  in one global namespace across ops, which is what lets a loop be *the same loop* for two operators.
  `Checker::get_active_tensors` walks the op list to build a producer→consumer map and locates each
  tensor's live scope at the producer/consumer **least common ancestor** in the mapping tree. The
  einsum-with-edges is the minimum extension that makes fusion statable, and it is the right one.
  Non-affine and data-dependent access is out of scope; softmax must be hand-expanded into five
  element-wise ops (§7.2).

- **Mapping-space representation.** A **tree**, and the tree *is* the mapping — there is no separate
  schedule object. Three node types: **`Tile`** (a perfect loop nest, `Temporal` or `Spatial`, with
  `factors`/`permutation`/`target` storage level/`bypass`/`split`, exactly one child), **`Scope`**
  (composition, ≥2 children, one of **`Sequential` / `Sharing` / `Parallel` / `Pipeline`**, no loops),
  **`Op`** (leaf einsum). An edge is containment in the iteration space. Depth is memory-hierarchy
  depth, enforced monotonically. The four scopes are a 2×2 of {compute shared in time vs partitioned
  in space} × {footprint exclusive vs additive}; `Parallel` and `Pipeline` differ in **one line** of
  the codebase. The **3D space** is (compute ordering = tree shape) × (resource binding = scope types
  + Temporal/Spatial) × (loop tiling = which loops, which factors). Loop permutation and bypass are
  **not** in the searched space (`docs/mcts.md`: *"not realized"*); they are hand-written.

- **Legality model.** **Two-tier, and it lands on Timeloop's answer independently.** *Tier 1,
  constructed-in*: perfect factorisation is not checked but **generates the candidate sets** (divisors
  of the residual loop count, intersected across co-constrained variables on every fix, singletons
  collapsed); tree structural rules are asserted once per tree or normalised away by a rewrite
  (`SpatialScopeSwapper` hoists a Scope above a Spatial tile and replicates the tile per branch); the
  Domino-side plan generator only emits dominator-legal attach points. *Tier 2, rejected*: **`MEM`**
  (Σ live footprints ≤ buffer size, per node, combinator max-for-`Seq` / sum-otherwise) and
  **`SPATIAL`** (an **(x, y) cardinality pair** ≤ per-level fanout, combinator **max for `Seq`/`Shar`,
  Σ for `Para`/`Pipe`** — a genuine cardinality budget shared across a fused group). **Bandwidth is not
  a constraint at all** — it is priced into latency, and `break_on_failure = false` everywhere, so
  Timeloop's per-level failures never abort. **Capacity and core-count are cliffs; bandwidth is a
  slope.** Failure carries **no typed reason** into the search — one bool, `failed_` — so a violated
  point is killed but not its slab. `TILEFLOW_ASSERT` is `exit(1)`, so a user-supplied illegal mapping
  terminates the binary.

- **Search strategy.** **Paper says "genetic algorithm + MCTS". No genetic algorithm exists in either
  repository's TileFlow path** (grepped both for `genetic|crossover|mutat|population`; the only GA is a
  different paper's SoC mapper in Domino). What ships is two levels: (1) **fusion structure —
  exhaustively enumerated** (`generate_fusion_plans`, DFS over the producer-consumer graph with
  dominator constraints; per intermediate tensor a choice of attach-target × attach-level × scope ∈
  {`Sequential`,`Sharing`,`Pipeline`} — `Parallel` is never enumerated) and then **uniformly randomly
  sampled**; (2) **tile factors — MCTS** (UCB, C = 1, most-constrained-variable-first ordering,
  `n_rollout = 100` scored by **max not mean**, reward = `−log₁₀(objective)` with an adaptive
  punishment floor, termination on 10000 iterations / timeout / space exhausted) **or** a
  **one-dimension-at-a-time annealer** (`AnnealingMutateOneDim`, restart sampled from history with
  `exp(v − best)`) which the artifact README calls "the GA". Objective is a **scalar**, cycles or
  energy. Published sizes: **5,103–20,412 structures**; converges in **<50 rounds × 20 structures**
  (≈5–20% of the structure space), **3.2–6.4 min** for tiling alone, **1–2 days** single-threaded for
  the full 3D space, **<1 h** on 56 processes. **~60 ms per fused-mapping evaluation.**

- **Cost model.** Analytical, tree-traversal, **built on a spring-2023 fork of NVlabs/Timeloop** whose
  `BufferLevel`/network models are reused wholesale. Data movement is computed as **set differences of
  data slices between adjacent time steps**, evaluated only at time-step boundaries (no unrolling);
  inter-tile transfers route through the **least common ancestor** when the levels have no direct path.
  Latency per §5.3: `Lat(Tₙ) = max{DMload/BW, {Σ or max over children}, DMstore/BW}`, load/compute/store
  assumed fully double-buffered. **`Pipeline` is `max` over stages with no fill, no drain, no rate
  matching.** Energy via Timeloop/Accelergy from access counts; an **Accelergy ERT/ART can be supplied
  in the input to replace the internal energy model**. Per-level `SlowDown`, `CapUtil`, `SpatialUtil`
  and per-tensor `Read`/`Update`/`Fill` are emitted to CSV, and **`SlowDown > 1` names the bottleneck
  level** — used in §7.5 to back-solve required bandwidth by sweeping it until slowdown hits 1.
  **Validation:** against Timeloop, single-op matmul, 1152 mappings, R² = 0.999 / 0.1% energy; against
  **RTL** (Chisel → Verilog → Verilator 4.0, Genus/Innovus, TSMC 22 nm, 7.84 mm², 400 MHz), **131 real
  mappings, 5.4% mean absolute cycle error, unfitted** — the strongest validation in this survey of
  eight. But: the validated workload is a **two-GEMM chain with no softmax** (the authors' own dataset
  directory is named **`No_Softmax`**), on a **single 16×16 array** rather than the four-core machine,
  through a **`Sequential`** scope — so **`Pipeline` and `Parallel` are validated against nothing**;
  the "6.1% energy" reference is itself an analytical model over RTL access counts with an
  undocumented **×1.21**; and the 48.8% "graph-based" strawman is three lines in the same script.

- **Multi-op support.** **Yes — the only one in this survey with a fused *mapper*, not just a fused
  model.** Fusion is the primitive, not an add-on: the tree encodes it, the constraint system
  accumulates resources across it, the search enumerates it, and the paper compares five published
  self-attention dataflows (Layerwise, Uni-pipe, FLAT-HGran, FLAT-RGran, Chimera) and four conv-chain
  dataflows (Layerwise, Fused-Layer, ISOS, TileFlow) **written in its own notation**, in-repo, as test
  data. Headline: **1.85× over FLAT-HGran for self-attention, 1.28× over Fused-Layer for conv chains**;
  and against FLAT under a memory limit, **82× less L1** at 14% more cycles (Table 7c).

- **Single most transferable idea.** **The scope-typed resource combinator: a cardinality demand
  accumulated up the mapping tree, `max` where stages share hardware in time and `Σ` where they occupy
  it in space, checked once at the scope against one budget.** Thirty lines
  (`ResourceConstraintParser`, `checker.cpp:486-535`). It is the mechanism doc 44 says MAESTRO cannot
  express — a *cardinality* resource, not a byte-rate — and it is exactly the shape of "sum the MM2S
  channels demanded by every stage in this `pipeline_group` and compare against 2 per column." Runner-up
  and nearly as valuable: the **per-level `SlowDown` metric that names the bottleneck**, plus its use
  as a *bandwidth back-solve* (sweep BW until slowdown = 1) — doc 44 item 4 and item 2, shipped and
  demonstrated on a fused chain.

- **Single biggest mismatch with our target.** **`Pipeline` is modelled as `max` over stage latencies
  with no fill, no drain, and no rate matching — and it was never validated against silicon.** Our
  fused groups are 2–4 stages with small per-stage trip counts, so the unmodelled ramp is O((S−1)/T)
  of the answer; the mechanism we would adopt TileFlow *for* is the one part of it the RTL never
  touched (§5.4), and COMET independently reports TileFlow under-predicting latency for exactly this
  reason. Close second: **no channel cardinality anywhere** — a storage level has bandwidth and
  instances, never "reached through 2 DMA channels" — and where a cardinality budget does exist
  (`SPATIAL`) it is one uniform (x,y) per *level*, not per *column*, and it is a hard `exit(1)` rather
  than the slope our silent packet-multiplexing requires.

---

## What this changes for our plan

Doc 44 as amended by doc 45 says: build a balance instrument (per-column demand matrix → static
bandwidth back-solve → overflow priced as a slope → latency as max over per-resource cycles whose
argmax names the bottleneck → persisted as a *measured* cost table); shrink the space by decoupling so
sizes add, **except** at the fusion-granularity/intra-stage-tiling joint; and treat `interleaved` as a
third composition state. Here is what TileFlow does to each.

### 1. **Adopt: the scope-typed resource combinator. This is the plan's missing piece and it is 30 lines.**

Doc 44's "What does not transfer" says a per-column, per-segment channel budget is *"unstateable as
such"* in Timeloop, and that MAESTRO's byte-rate *"has to be [made a cardinality] deliberately."*
**TileFlow shows the deliberate version and it is small.** Reimplement it against our IR, not their
C++:

```
demand(node) :=
  leaf (air.pipeline_stage)  -> the per-column channel vector that stage requires
  Scope(Seq | Shar, kids)    -> elementwise max over kids     # stages share the column in time
  Scope(Para | Pipe, kids)   -> elementwise sum over kids     # stages occupy the column together
  Tile(loops, kid)           -> demand(kid), scaled by spatial extents
check at the group:  demand ≤ budget, per column
```

Two deliberate deviations from theirs, both forced by our machine: the demand is a **vector over
columns**, not one (x,y) scalar pair; and the comparison **prices the excess** rather than rejecting
it, `slowdown = min(1, budget/demand)` per column, because AIR packet-multiplexes. Doc 44 already
settled the pricing question; TileFlow supplies the accumulation TileFlow's own policy then gets wrong
for us.

**Do this first.** It is the smallest piece of the balance instrument, it is the one we had no design
for, and it is testable today against a `pipeline_group` without any measurement.

### 2. **Adopt: `SlowDown` per resource, with the bandwidth back-solve as its calibration ritual.**

Doc 44 item 4 wanted "latency = max over per-resource cycles whose argmax names the bottleneck." §7.5
adds the move we did not have: **sweep the resource until slowdown hits 1, and report that value as the
requirement.** *"the suitable L1 bandwidth is the minimal value that makes L1 slow-down as 1"*
(Fig. 14). That converts our instrument from a diagnostic into a **spec generator**: for a given fused
group, print "this group needs N channels per column and M GB/s of memtile bandwidth to stop being
DMA-bound." That is a far more useful artifact than a latency number, and doc 38's iron comparison
would have been settled in one line by it.

### 3. **Reorder: dissolve the granularity enum instead of exempting it.**

Doc 44 item 1's exemption — *"fusion granularity and intra-stage tiling are not separable, so let that
axis multiply"* — should be **replaced, not kept**. TileFlow §7.3 shows FLAT's `{M,B,H,R}` enum is an
artifact of FLAT refusing to tile the column dimension, and that abolishing the enum (every dimension
tileable) both removes the joint and wins 82× L1. **Carry a per-dimension tileability flag, not a
granularity enum.** Then doc 44 item 1's decoupling applies everywhere with no exemption, and the
Marvel-style "sizes add" reduction is available at full strength.

**And carry the warning that comes with it.** Table 7(a) vs 7(b): under fixed tiling, granularity
appeared to matter 18×; under searched tiling, three of four granularities **tie exactly**. Our
composition ladder (packaged / resident / interleaved) is measured at hand-chosen tile sizes.
**Before we publish any claim that composition state X beats state Y, we must show the comparison
survives a tile-size search on both sides** — otherwise we are measuring our tile choices. This is the
fused-domain version of doc 44's Timeloop warning and it is aimed straight at doc 25, 27 and 31's
ladders.

### 4. **Reimplement, don't port: `Pipeline` needs fill and drain, and TileFlow's does not have them.**

TileFlow's `Pipe` is `max` over stages — steady state only. Our groups are short. **Our pipeline
latency model must be `T·max(stage) + (S−1)·max(stage) + handoff`, and the fill term must be visible in
the output**, because at `baseline_768` it may be the whole difference between packaged and
interleaved. COMET confirms this is a real defect in TileFlow, not a conservative choice. **Do not
inherit `max`.**

Related and equally ours: TileFlow §7.1 admits it *"assumes data replacement happens for every outer
iteration"* and therefore **over-estimates data movement for small tiles**. Small tiles are our regime.
Any TileFlow-derived data-movement figure we compute for our shapes is an **upper bound**, and should be
labelled as one.

### 5. **Adopt the interface pattern: free variables by naming, ERT by injection.**

Two interface decisions worth copying verbatim into our own instrument:

- **A tile factor is either a number or a name.** `factors: B=? H=? M=?` — the same document is a
  concrete mapping *and* a search template, distinguished only by whether the fields are bound. Our
  `air.pipeline_group` attributes should work the same way, so one artifact serves the reproduction
  gate and the search.
- **The cost table is an input, not a build-time constant.** `main.cpp` accepts an Accelergy ERT/ART in
  the input YAML and *"replac[es] internal energy model."* Doc 44 item 5 wants our measured
  nanoseconds persisted as a `(component, action, arguments) → cost` table; **make it an input file the
  instrument reads**, so a re-measurement is a data change, not a code change. And keep doc 44's
  insistence that actions carry arguments — TileFlow's own ERT keys are argument-free
  (`random_read`, `random_fill`), which is why it cannot see a BD-stride wall.

### 6. **`interleaved` is not a third state — it is a `Scope` type, and there are four of them.**

Doc 44's `[2026-08-12]` amendment adds `interleaved` beside *packaged* and *resident*. TileFlow's Table 1
says the axis is bigger than three and gives the right factorisation: **{compute shared in time vs
partitioned in space} × {memory exclusive vs additive}**. Mapped onto doc 03's vocabulary:

| TileFlow | our word | meaning on NPU2 |
|---|---|---|
| `Seq` | **packaged** | stages take the herd in turns, buffers freed between |
| `Shar` | **resident** | stages take the herd in turns, buffers stay live |
| `Pipe` | **interleaved** | stages on distinct cores, dependent, chained working set |
| `Para` | *(unnamed, and we have it)* | independent stages on distinct cores — our `f(heads ‖ ffn)` tails |

**We have four composition states and three words.** `Para` is the one we have been building (doc 20's
fuse-through-parallel, doc 22's accumulator ring) without a name distinct from `Pipe`. Adopt the 2×2 —
it is a better ontology than a list, it makes the resource combinator fall out (§1 above: max vs sum is
*exactly* the compute-axis of the 2×2), and it tells us which cell we have never tested.

### 7. **Ignore: the search, the GA, and the idea of driving TileFlow from our builders.**

The GA does not exist. The MCTS searches tile factors only, over a space our constraint propagation
would shrink the same way. The structure "search" is enumerate-then-sample-uniformly, which we can
write in a day once `generate_fusion_plans`'s shape is understood (per intermediate: attach point ×
level × scope, dominator-constrained). And the round trip — AIR → einsum → TileFlow → tree → AIR
builder parameters — costs more than the search it would host, on an unmaintained 2023 Timeloop fork
that `exit(1)`s on an illegal mapping.

**Verdict, by component:**

| component | verdict |
|---|---|
| the tree representation (`Tile`/`Scope`/`Op`, 4 scope types) | **adopt the design** — it is the right IR shape, and ours already resembles it |
| scope-typed resource combinator (max/sum by scope) | **reimplement** — 30 lines, vectorised over columns, priced as a slope |
| `SlowDown` bottleneck naming + bandwidth back-solve | **reimplement** — doc 44 items 2 and 4, now with a calibration ritual |
| two-tier legality (divisors constructed-in, capacity rejected) | **adopt the design** — confirms Timeloop; add the typed reasons neither has |
| the "only the consumer's reduction loops may be outer" rule (§4.1) | **adopt** — it is doc 31's wall 5 as a checkable property of a `pipeline_group` |
| free-variable-by-naming, ERT-as-input | **adopt the interface pattern** |
| max-of-rollouts, `−log₁₀` reward, most-constrained-variable-first | **adopt** if we ever build a search |
| the 2×2 composition ontology | **adopt** — replaces doc 44's three-state list |
| `Pipeline` cost model (`max`, no fill/drain) | **ignore** — actively wrong for us |
| the mapper (GA + MCTS) | **ignore** — the GA does not exist; the MCTS is not the hard part |
| TileFlow as a tool in our loop | **ignore as a mapper; optionally adopt as a batch oracle** for cross-checking data-movement volumes (steps 1, 2, 8, 9 of §7.2 only) |
| the RTL numbers as evidence about fused pipelines | **ignore** — `Seq` scope, one core, no softmax |

**One sentence.** TileFlow is the first framework in this survey that can *state* our problem, and it
states it well enough that we should copy its vocabulary and two of its mechanisms outright — but the
one construct we need it for, `Pipeline` across cores, is modelled as a bare `max` and was never put in
front of silicon, so **we take the representation and the combinator, and we still have to build the
instrument ourselves.**

### Read next (revised)

- **COMET** (arXiv [2509.00599](https://arxiv.org/abs/2509.00599), 2025) — **new to the list and it
  should be near the top.** Explicit **collective operations** as first-class nodes, **ramp-up/ramp-down
  modelled**, per-tensor-per-level loop nests, and it targets *"spatially distributed compute
  clusters"*. It names and fixes the two TileFlow defects that matter most to us. Ours is a distributed
  cluster with explicit data movement; this is the closest published frame.
- **LoopTree** (arXiv [2409.13625](https://arxiv.org/abs/2409.13625) / TCAS-AI 2024) — still wanted, for
  the **per-intermediate-fmap retention** encoding TileFlow's Table-I row says it lacks. Doc 44's
  description of LoopTree as a `Pipeline`/`Sequential` node tree should be corrected: it is polyhedral,
  and its parallelism choice is one global flag.
- **Chimera** (cited by TileFlow as a fused conv/attention dataflow, and in-repo as
  `chimera_self_attention.py`) — the third published fused attention dataflow after FLAT and
  Fused-Layer, and it beats FLAT-RGran in TileFlow's own Fig. 10.
- Unchanged from doc 44: **FuseMax**, **Union**, **Ruby**, **Eyexam**.
