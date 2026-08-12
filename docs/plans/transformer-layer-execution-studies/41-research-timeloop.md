# Timeloop — research notes for the NPU2 mapping-space study

**Scope.** Timeloop (Parashar et al., ISPASS 2019) plus the descendants that matter to us: Sparseloop
(v2), Ruby (v3), LoopTree / fused-mapping (v4, in-tree), Orojenesis, Accelergy, and Union.

**Primary sources used.**

- Paper: *Timeloop: A Systematic Approach to DNN Accelerator Evaluation*, ISPASS 2019 —
  <https://people.csail.mit.edu/anurag_m/papers/2019.timeloop.ispass.pdf> (also linked from the repo
  README as <https://parashar.org/ispass19.pdf>). Cited below as **[ISPASS19 §x]**.
- Source: `NVlabs/timeloop` @ `32370826` (2025-06-09), cloned to
  `scratchpad/research-timeloop-private/timeloop/`. Cited as **[repo path:line]**.
  Repo browse URL: <https://github.com/NVlabs/timeloop>.
- Docs: <https://timeloop.csail.mit.edu/> (v4 input/output formats, Orojenesis).
- LoopTree: <https://arxiv.org/abs/2409.13625> / <https://arxiv.org/html/2409.13625v4> (IEEE TCAS-AI 2024).
- Ruby ISPASS 2022: <https://ieeexplore.ieee.org/document/9804679>.
- Sparseloop MICRO 2022: <https://arxiv.org/abs/2205.05826>.
- Union PACT 2021: <https://arxiv.org/abs/2109.07419>.
- Orojenesis / "Mind the Gap" ISCA 2024: <https://timeloop.csail.mit.edu/orojenesis>.

Anything I derived rather than read is marked **[inference]**.

---

## 1. Data space — how the problem is represented

A **problem** is one Einstein-summation ("einsum") over a rectangular iteration space, plus affine
projections from that iteration space to each tensor.

Three parts [<https://timeloop.csail.mit.edu/v4/input-formats/problem>]:

1. `name` — string.
2. `dimensions` — list of single-character rank names. These *are* the iteration space; the loop nest
   has exactly one loop per dimension per tiling level.
3. `data_spaces` — each with a `name`, a `projection` (list-of-lists of dimension variables), and
   `read_write: True` for outputs.

Real example, verbatim from `problem-shapes/gemm_ABZ.yaml`:

```yaml
shape:
  name: "gemm_ABZ"
  dimensions: [ M, N, K ]
  data_spaces:
    - name: A
      projection:
        - [ [M] ]
        - [ [K] ]
    - name: B
      projection:
        - [ [N] ]
        - [ [K] ]
    - name: Z
      projection:
        - [ [M] ]
        - [ [N] ]
      read_write: True
```

Each top-level list entry is one tensor rank; the inner lists are a **sum-of-products** over
iteration variables and named `coefficients`, which is how strided/dilated convolution is expressed
(`problem-shapes/cnn_layer.yaml`):

```yaml
  dimensions: [ R, S, P, Q, C, K, N ]
  coefficients:
    - name: Wstride
      default: 1
    - name: Hdilation
      default: 1
  data_spaces:
    - name: Inputs
      projection:
        - [ [N] ]
        - [ [C] ]
        - [ [R, Wdilation], [P, Wstride] ] # SOP form: R*Wdilation + P*Wstride
        - [ [S, Hdilation], [Q, Hstride] ]
    - name: Outputs
      projection: [ [ [N] ], [ [K] ], [ [Q] ], [ [P] ] ]
      read_write: True
```

The paper's framing [ISPASS19 §V-A]: "Each iteration of the loop body is a MAC operation, which we
refer to as a *point* in the **operation space** of a workload… Given a set of points in the
operation space, Timeloop can determine the set of operands and results for those points by
*projecting* the 7D operation points into the 4D dataspace dimensions."

**Dependences are not represented.** Loop iterations are asserted to be commutative and freely
reorderable [ISPASS19 §V-A: "body iterations that may be freely re-ordered"]. There is no dependence
graph — the projections are the entire semantics. Instance sizes go in a separate `instance:` block
(`R: 3, S: 3, P: 48, …`, see `configs/mapper/sample.yaml`).

Multi-einsum problems exist only in the v4 `FusedWorkload` class
(`include/workload/fused-workload.hpp`), which adds `EinsumId`/`DataSpaceId`/`DimensionId`, a
producer/consumer index (`WriterEinsum(dspace)`, `ReaderEinsums(dspace)`), and stores projections as
`isl::multi_aff` / `isl::map` instead of the SOP lists. See §6.

---

## 2. Mapping space — what a mapping actually is

### 2.1 The mapping object

A mapping is a **single flat loop nest over all problem dimensions, cut into `tiling levels` by
storage-tiling boundaries**, where every loop is tagged temporal or spatial (SpaceX/SpaceY), plus a
per-level per-dataspace **bypass mask**.

[ISPASS19 §V-C]: "To construct the mapping, the 1D convolution loop nest is split into a number of
sections (called *tiling levels*) equal to the number of storage hierarchy levels, plus the number
of levels with spatial fanouts. Each tiling level has a loop corresponding to each dimension in the
original workload (though the bound may be 1). The product of all the loop bounds belonging to a
dimension must be equal to the final (optionally padded) value of the dimension, e.g.,
`P = P0*P1*P2*P3`."

Concretely in code, `Mapping` (`include/mapping/mapping.hpp`) carries `loop_nest`,
`complete_loop_nest`, `datatype_bypass_nest`, `fanoutX_map`/`fanoutY_map`, and the loop-nest
modifiers `skew_descriptors`, `no_link_transfer`, `no_multicast`, `no_temporal_reuse`,
`rmw_first_update`, `no_coalesce` [`src/mapspaces/uber.cpp:616-625`].

Emitted mappings are printed as an indented loop nest with per-level tensor residency
[<https://timeloop.csail.mit.edu/v4/output-formats/mapping>]:

```
RegisterFile [ Weights:1 (1) Inputs:16 (16) Outputs:16 (16) ]
|   for C in [0:2)
|     for C in [0:16)     (Spatial-X)
```

A mapping can also be lowered to a code generator: `global_best_.mapping.PrintTenssella(...)`
[`src/applications/mapper/mapper.cpp:641`].

### 2.2 The mapspace decomposition — get this exactly right

This is the part worth copying. The mapspace is a **Cartesian product of four independent
sub-spaces**, and a mapping is addressed by a **single 128-bit integer** decomposed into one
coordinate per sub-space [`include/mapspaces/mapspace-base.hpp:44-60`]:

```cpp
enum class Dimension
{
  IndexFactorization,  // Factorization of loop bounds across storage levels
  LoopPermutation,     // Permutation of loop nests in each storage level
  Spatial,             // Position of the transition point between horizontal and vertical
                       //   spatial tilings
  DatatypeBypass,      // Optionally bypass a storage level for a datatype
  Num
};
typedef CartesianCounter<int(Dimension::Num)> ID;
```

- **IndexFactorization** — for every problem dimension, the ordered tuple of cofactors, one per
  tiling level, whose product is the dimension size. Perfect factorization by default; the `ruby`
  mapspace template permits remainders (§7 of this doc).
- **LoopPermutation** — for each tiling level, the order of that level's loops.
- **Spatial** — for each *spatial* tiling level, a single integer: the index in that level's
  permutation at which loops stop being SpaceX and start being SpaceY
  [`src/mapspaces/uber.cpp:867-888`]. Space is 2-D only.
- **DatatypeBypass** — per dataspace, a bitmask over storage levels: `1` = keep, `0` = bypass, `X` =
  let the mapper try both [`src/mapspaces/uber.cpp:310-313`].

**Note a paper/repo discrepancy.** [ISPASS19 §V-E] names only *three* sub-spaces —
"the *IndexFactorization* sub-space… the *LoopPermutation* sub-space… and the *LevelBypass*
sub-space." The shipped code has **four**; `Spatial` is a first-class mapspace axis and
`LevelBypass` is spelled `DatatypeBypass`. Anyone reproducing the paper's mapspace-size arithmetic
must add the spatial-split factor.

The four sub-spaces are constructed independently at init and a mapping is *materialised* in four
staged passes [`src/mapspaces/uber.cpp:517-628`]: Stage 0 `InitSubnests` → Stage 1 `PermuteSubnests`
→ Stage 2 `AssignIndexFactors` → Stage 4 `ConstructDatatypeBypassNest` → Stage 3
`AssignSpatialTilingDirections` (3 and 4 are deliberately swapped because the spatial pass needs the
bypass nest).

### 2.3 A real mapping-constraints YAML

The mapspace is *narrowed* by constraints. Verbatim from
<https://github.com/NVlabs/timeloop/blob/master/configs/mapper/sample.yaml>:

```yaml
mapspace:
  constraints:
  - target: AccumulationBuffer
    type: datatype
    keep:   [ Outputs ]
    bypass: [ Weights, Inputs ]
  - target: AccumulationBuffer
    type: spatial
    factors: P1 Q1 R1 S1 C64 K1 N1
    permutation: KQRSPNC
  - target: WeightInputBuffer
    type: spatial
    factors: P1 Q1 R1 S1 C1 K16 N1
    permutation: KCQRSPN
  - target: Registers
    type: temporal
    factors: R1 S1 C1 K1 N1
    permutation: PQRSCKN
  - target: WeightInputBuffer
    type: temporal
    factors: P1 Q1 K1        # partial: unspecified dims stay free
  - target: Registers
    type: utilization
    min: 0.01
```

The full constraint-type list, read off the parser [`src/mapping/constraints.cpp:604-908`]:

| `type` | fields | effect |
|---|---|---|
| `temporal` | `factors` (`X4`, `X>=2`, `X<=8`, `X-1` = "exhaust"), `permutation` (prefix and/or suffix), `no_reuse`/`no_temporal_reuse`, `rmw_first_update` | fixes/bounds IF cofactors and LP order at a temporal level |
| `spatial` | same plus `split` (the SpaceX→SpaceY changeover), `no_link_transfer`, `no_multicast_no_reduction` | same at a spatial level, and pins the `Spatial` coordinate |
| `datatype` / `bypass` / `bypassing` / `dataspace` | `keep: [...]`, `bypass: [...]`, `no_coalesce` | pins DatatypeBypass bits |
| `utilization` / `parallelism` | `min` | minimum cumulative spatial-fanout utilization |
| `max_overbooked_proportion` | scalar | sparse/stochastic occupancy slack |
| `skew` | `modulo`, `terms: [{constant, variable, bound}]` | affine spatial skew (Ruby) |

Also relevant to us: **`Ruby` vs `Uber` mapspace templates** are selected by
`mapspace: {template: "ruby"}` [`src/mapspaces/mapspace-factory.cpp:37-47`;
<https://timeloop.csail.mit.edu/v4/input-formats/mapspace>]. Docs: `"ruby"` permits imperfect
factorizations; anything else restricts to perfect factorizations only.

---

## 3. Legality — and this is the answer to our real question

Timeloop splits legality into **two tiers with different mechanisms**, and the split is explicit in
the paper [ISPASS19 §V-E]:

> "User-specified constraints are accommodated into the mapspace, shrinking the sizes of the
> underlying sub-spaces. A mapping sampled from the mapspace is therefore **guaranteed to obey those
> constraints**. Hardware resource constraints, e.g., whether a set of tensor tiles at a level fit
> into the size of memory at that level, **are checked once a mapping is sampled** from the
> mapspace, and the mapping is **rejected** if the constraints cannot be met."

So: **structural** legality (dataflow, permutation, which dim goes where, X/Y split) is
*constructed into* the enumeration and is inexpressible-if-illegal. **Resource** legality (capacity,
fanout, instance count) is *checked and rejected after the fact*.

And even the constructed part is not dense. The mapper source says so bluntly
[`src/applications/mapper/mapper-thread.cpp:540-543`]:

```cpp
// Stage 1: Construct a mapping from the mapping ID. This step can fail
//          because the space of *legal* mappings isn't dense (unfortunately),
//          so a mapping ID may point to an illegal mapping.
```

### 3.1 The three-stage rejection pipeline

Each candidate goes through three gates of increasing cost, each returning a typed status
[`src/applications/mapper/mapper-thread.cpp:538-638`]:

| Stage | Call | Fail class | Checks |
|---|---|---|---|
| 1 | `mapspace_->ConstructMapping(id, &mapping)` | `FailClass::Fanout` | mapped fanoutX/fanoutY vs hardware fanout; `min-parallelism` |
| 2 | `engine.PreEvaluationCheck(...)` | `FailClass::Capacity` | fast working-set-size vs buffer capacity, before full tile analysis |
| 3 | `engine.Evaluate(...)` | `FailClass::Capacity` | full tile analysis: capacity, metadata capacity, utilized instances, meshX/meshY expansion, confidence thresholds |

Statuses are per-topology-level and carry a human-readable reason, e.g.
[`src/mapspaces/uber.cpp:894-906`]:

```cpp
if (filter_spatial_fanout_ && x_expansion > arch_props_.FanoutX(storage_level_id))
{
  success = false;
  fail_reason << "mapped fanoutX " << x_expansion << " exceeds hardware fanoutX "
              << arch_props_.FanoutX(storage_level_id);
}
```

and [`src/model/buffer.cpp:2040-2060`]:

```cpp
else if (stats_.utilized_instances.Max() > specs_.instances.Get())
  fail_reason << "mapped instances " << ... << " exceeds available hardware instances " << ...;
else if (stats_.utilized_x_expansion.Max() > specs_.meshX.Get()) ...
else if (stats_.utilized_y_expansion.Max() > specs_.meshY.Get()) ...
```

With `mapper.diagnostics: True`, every rejection is bucketed by `(FailClass, level)` and aggregated
into a histogram across threads [`src/applications/mapper/mapper.cpp:470-500`], and when the search
finds nothing the tool prints a targeted remediation script
[`src/applications/mapper/mapper.cpp:645-660`]: observe the per-thread termination message, which
tells you "the number of mappings that failed because of a spatial fanout violation and the number
that failed because of a buffer capacity violation"; then find the offending constraint; then relax
`victory_condition` / `timeout` / `search_size`.

### 3.2 The one that matters most to us: bandwidth is NOT a legality condition

This is the direct analogue of our silently-packet-multiplexed MM2S channel, and Timeloop's answer
is deliberate. Comment in `src/model/buffer.cpp:2062-2064`:

```cpp
// Bandwidth constraints cannot be checked/inherited at this point
// because the calculation is a little more involved. We will do
// this later in the ComputePerformance() function.
```

And `ComputePerformance` [`src/model/buffer.cpp:2475-2599`] computes, per dataspace, the
unconstrained demand `total_accesses / compute_cycles`, sums it, and turns over-subscription into a
multiplicative **slowdown**, never a rejection:

```cpp
stats_.slowdown = 1.0;
if (specs_.read_bandwidth.IsSpecified()
    && specs_.read_bandwidth.Get() < total_unconstrained_read_bandwidth)
  stats_.slowdown = std::min(stats_.slowdown,
                             specs_.read_bandwidth.Get() / total_unconstrained_read_bandwidth);
// ... same for write_bandwidth and shared_bandwidth
```

**So Timeloop models exactly our failure mode — an over-subscribed port degrades rather than
errors — and it treats that as a modelling problem, not a legality problem.** Capacity is a cliff
(reject); bandwidth is a slope (slow down). [inference] The design rule this implies for us:
a per-column *channel count* is structural and belongs in the constructed-in tier; per-column
*bandwidth* is a resource and belongs in the cost model as a slowdown so illegal-looking points get
correctly-bad numbers instead of a surprise.

---

## 4. Search — taming the explosion

### 4.1 How big is it

[ISPASS19 §V-E], verbatim: "The Cartesian product of these sub-spaces gives us an *unconstrained*
mapspace, which can be quite large due to combinatorial explosion, e.g., for a 7D CNN on a
4-tiling-level architecture the size is **(7!)⁴ × (2⁴)³ ×** size of the Cartesian product of the
co-factor sets for each of the 7 loop bounds. Although there are ways to prune this space, e.g.,
permutations do not matter for the innermost tiling level, and for factors that are 1, the space is
still large."

`(7!)⁴ × (2⁴)³ ≈ 2.6 × 10¹⁸` before the index-factorization term **[inference — my arithmetic]**.

The code corroborates the scale: `Uber::Init` prints each sub-space size and hard-fails if the
product overflows 128 bits [`src/mapspaces/uber.cpp:100-119`]:

```cpp
std::cerr << "ERROR: overflow detected: mapspace size appears to be "
          << "greater than 2^128. Please add some mapspace constraints." << std::endl;
```

The other published number is a **quality**-of-mapspace number, not a size, and it is the paper's
motivating figure [ISPASS19 §II, Fig. 1]: for VGG_conv3_2 on a 1024-MAC NVDLA-like architecture,
**480k mappings all within 5% of peak performance vary nearly 19× in energy efficiency**; only one
is energy-optimal and 9 others are within 1%. Separately, **6,582 mappings tie on minimum DRAM
accesses yet vary 11× in energy efficiency**. That is the argument for searching rather than
hand-picking.

### 4.2 The strategies

Five, chosen by `mapper: {algorithm: ...}`, **default `hybrid`**
[`src/search/search-factory.cpp:44-71`]:

| name | behaviour |
|---|---|
| `exhaustive` | odometer over the 4-D mapspace ID |
| `linear_pruned` | odometer + feedback-driven sub-cube skipping (below) |
| `random` | uniform random 128-bit IDs |
| `random_pruned` | random over IndexFactorization, linear + pruned over the rest |
| `hybrid` (default) | **random over IndexFactorization, linear-pruned over LoopPermutation/Spatial/DatatypeBypass**, with a `visited_` revisit filter |

The search interface is a two-way channel, not a generator
[`include/search/search.hpp:35-50`]:

```cpp
enum class Status { Success, MappingConstructionFailure, EvalFailure };
virtual bool Next(mapspace::ID& mapping_id) = 0;
virtual void Report(Status status, double cost = 0) = 0;
```

### 4.3 The two pruning mechanisms — the genuinely clever bit

**(a) Structural pruning per index-factorization (`InitPruned`).** The odometer's slowest axis is
IndexFactorization. Every time it advances, the *other* sub-spaces are rebuilt for that specific
factorization [`src/mapspaces/uber.cpp:409-459`]: any dimension whose cofactor at a level is `1`
gives a trivial loop, so its position in that level's permutation is irrelevant and it is removed
from the permutation space; the count of such unit factors also shrinks the spatial-split space.
That is a free, large reduction, recomputed per IF point.

**(b) Feedback pruning from evaluation outcome (`Report`).** Dimension order, innermost-first, is
`DatatypeBypass ← Spatial ← LoopPermutation ← IndexFactorization`
[`include/search/linear-pruned.hpp:66-73`]. The inference rules, verbatim from
`src/search/linear-pruned.cpp:153-190`:

```cpp
else if (status == Status::MappingConstructionFailure)
{
  // Accelerate search by invalidating bad spaces.
  // ConstructMapping failure =>
  //   Combination of (IF, LP, S) is bad.
  //   Skip all DBs.
  skip_datatype_bypass = true;
}
else if (status == Status::EvalFailure)
{
  // PreEval/Eval failure (capacity) =>
  //   Combination of (IF, DB) is bad.
  //   If all DBs cause Eval failure for an IF, then that IF is bad,
  //   no need to look at other LP, S combinations.
  eval_fail_count_++;
}
...
if (eval_fail_count_ == mapspace_->Size(mapspace::Dimension::DatatypeBypass))
{
  // All DBs failed eval for this combination of IF*LP*S. This means
  // this IF is bad. Skip to the next IF by fast-forwarding to the end of
  // this IF.
  iterator_[Spatial]         = mapspace_->Size(Spatial) - 1;
  iterator_[LoopPermutation] = mapspace_->Size(LoopPermutation) - 1;
}
```

A fanout failure kills one whole bypass axis; a total capacity failure kills the entire
permutation × spatial slab for that factorization. **This is the "greedy prune derives legality"
pattern**: legality is not solved, it is *learned from rejections and projected back onto whole
sub-cubes of the index space.*

### 4.4 Termination — four conditions, all in the thread loop

[`src/applications/mapper/mapper-thread.cpp:390-445`]:

| condition | meaning | default |
|---|---|---|
| `gTerminate` | SIGINT | — |
| `search_size` | stop after N **valid** mappings (divided across threads) | `0` = unlimited [`mapper.cpp:191-196`] |
| **`victory_condition`** | stop after N consecutive mappings **since the last improvement to the best** | `500` in code [`mapper.cpp:203-206`]; `100` in `configs/mapper/sample.yaml` |
| `timeout` | stop after N consecutive **invalid** mappings (fanout + capacity) since the last valid one | `1000` |
| `search_->Next()` returns false | mapspace exhausted | — |

Also: the mapspace is **split across threads along the IndexFactorization axis**
(`Uber::Split`, `src/mapspaces/uber.cpp:464-500`), threads periodically sync a global best
(`sync_interval_`), and there is a `penalize_consecutive_bypass_fails_` knob so that a failure whose
*only* delta from the previous candidate was in the bypass axis does not count against `timeout`
[`mapper-thread.cpp:512-531, 591`].

Note the honest paper/repo gap. [ISPASS19 §V-E] says only: "we currently employ either an exhaustive
linear search (for small mapspaces) or a random sampling based heuristic (for large mapspaces). More
sophisticated search heuristics are planned for future work." The five-algorithm menu, the pruning
rules, and the victory condition are all **repo-only**; do not cite the 2019 paper for them.

---

## 5. Cost model

**Analytical, not simulated** [ISPASS19 §IV]: "This evaluation does not rely on a cycle-accurate
simulator; instead, Timeloop exploits the fact that computation and data-movement patterns in DNN
computations are largely deterministic, enabling it to compute throughput and access counts
analytically."

Three stages [ISPASS19 §VI, Fig. 2]:

1. **Tile analysis (§VI-A).** Track a `point set` per tiling level per dataspace. Between
   consecutive iterations `i` and `i+1` of a loop, compute the set-difference — the **delta**
   (Fig. 7). For a temporal loop, an empty delta = perfect reuse (stationarity); a non-empty delta =
   the incremental data that must be transferred. For a `parallel_for` loop, a delta at the same
   timestep = **multicast** opportunity; an empty delta between adjacent instances at consecutive
   timesteps = **forwarding** (systolic). Two optimisations make this cheap: only the first, second
   and last iterations of each loop are computed and the rest extrapolated algebraically; and each
   tile shape is an axis-aligned hyper-rectangle.
2. **Microarchitecture model (§VI-B).** Turn tile-access counts into per-component action counts —
   multiplier accesses, buffer reads/fills/updates, network ingresses, multicast signatures, hops,
   spatial reduction, address-generator invocations.
3. **Technology model (§VI-C).** Energy per action; built-in TSMC 16nm memory/arithmetic/wire models
   in the 2019 paper.

**Latency is a max over per-resource isolated cycles** [ISPASS19 §VI-D], verbatim:

> "Performance is estimated by calculating the number of cycles it would take for each hardware
> component to complete the workload in isolation. For multipliers, required cycles are equal to the
> number of MACs in the workload divided by the number of multipliers. **For communication
> interfaces, required cycles are equal to the amount of data flowing through that interface divided
> by its bandwidth.** Buffers, networks and arithmetic units are modeled as operating in a pipeline.
> Thus, **the overall latency is the maximum of isolated execution cycles across all buffers,
> networks and arithmetic units** in the hardware. This model, which assumes negligible pipeline
> stalls, is reasonable for architectures that use double-buffering or more sophisticated techniques
> like *buffets*."

In code that is literally `total_cycles = std::max(total_cycles, storage_level->Cycles())` over all
storage levels seeded with `compute_cycles` [`src/model/topology.cpp:1428-1478`], and
`utilization = ArithmeticLevel()->IdealCycles() / cycles` [`topology.cpp:1616`].

**Validation** [ISPASS19 §VII-C]: against an in-house NVDLA-derived RTL model over 107 DeepBench
workloads — cycle accuracy 78%–99%, **mean 95%**; energy projections **within 8%** of baseline for
all 107. The six 78–88% outliers were traced to address-generator/layout effects causing real
pipeline stalls the model assumes away.

**Objective.** `optimization_metrics: [energy, delay]` in `configs/mapper/sample.yaml`; the paper
uses energy-delay product "though any of the statistics provided by the model can be trivially used"
[ISPASS19 §V-E].

**Accelergy handoff.** Timeloop does not call Accelergy in-process; it consumes two generated tables
[`src/model/topology.cpp:49-180`, `include/model/topology.hpp:174-175`]:

- **ERT** (Energy Reference Table) → `Topology::Specs::ParseAccelergyERT` → per-level
  `UpdateOpEnergyViaERT(...)`.
- **ART** (Area Reference Table) → `ParseAccelergyART` → `UpdateAreaViaART(component_area)`.

The contract is a name map from Timeloop action names to a *priority list* of Accelergy action names
[`include/model/topology.hpp:51-80`]:

```cpp
static std::map<std::string, std::vector<std::string>> arithmeticOperationMappings
  = {{"random_compute", {"mac_random", "mult_random", "mac", "mult", "compute"}}, ...};
static std::map<std::string, std::vector<std::string>> storageOperationMappings
  = {{"random_read", {"random_read", "read"}},
     {"random_fill", {"random_fill", "random_write", "fill", "write"}}, ...};
```

So Timeloop owns *counts*; Accelergy owns *joules-per-count and mm²*. Clean separation, and the
interchange is a flat table keyed by (component, action).

---

## 6. Multi-operator — the decisive question

### 6.1 Base Timeloop: fundamentally single-einsum

[ISPASS19 §V-A], verbatim and unambiguous:

> "Timeloop's workload format is similar to the form of a single DNN layer. To evaluate a complete
> network, one can invoke Timeloop sequentially on each layer and accumulate the results. Each layer
> has tremendous reuse opportunities, and **we leave exploration of cross-layer reuse to future
> work**."

The mapper application takes one `problem`, builds one `Workload`, one mapspace, one loop nest. There
is no fusion, no producer-consumer residency, no pipeline across operators.

### 6.2 LoopTree / fused-mapping: the extension, and its critical limitation

v4 of the repo ships the LoopTree machinery: `include/mapping/fused-mapping.hpp`,
`src/mapping/fused-mapping/`, `include/workload/fused-workload.hpp`,
`src/loop-analysis/mapping-to-isl/`, `src/applications/looptree-model/`.

**A fused mapping is a tree, not a nest.** Node variant
[`include/mapping/fused-mapping.hpp:135-137`]:

```cpp
using MappingNodeTypes
    = std::variant<Root, For, ParFor, Storage, Compute, Pipeline, Sequential>;
```

with fields:

- `For { iterator_name, op_dim, begin, end, tile_size, child }`
- `ParFor { … , int spatial, … }`
- `Storage { BufferId buffer, DataSpaceId dspace, bool exploits_reuse, child }`
- `Compute { EinsumId kernel, BufferId compute, optional<isl::pw_multi_aff> tiling_spec, optional<double> parallelism }`
- `Pipeline { vector<NodeID> children }`, `Sequential { vector<NodeID> children }`

The YAML surface [`src/mapping/fused-mapping/parser.cpp:46-56, 110-320`] is
`mapping: {type: fused, nodes: [...]}` where each node has `type:` in
`{temporal, spatial, storage, compute, pipeline, sequential}`; `temporal`/`spatial` take
`rank`, `iterator_name`, and `factor` **or** `tile_shape` (`spatial` also takes `spatial:` = which
spatial axis); `storage` takes `target` and `dspace: [...]` plus `exploits_reuse`; `compute` takes
`einsum`, `target`, `parallelism`; `pipeline`/`sequential` take `branches: [[...], [...]]`. The last
node in any list must be a branch or a leaf, or the parser throws.

`Pipeline` branches are tagged `PipelineSpatial()` in the ISL analysis
[`src/loop-analysis/mapping-to-isl/fused-mapping-to-isl.cpp:674-676`] — i.e. **concurrent spatial
occupancy of different einsums**. `Sequential` is tagged `Sequential()` and drives
`BufRightAboveSequential` to find where an intermediate must be buffered
[`fused-mapping-to-isl.cpp:503-522`]. This is structurally our "distinct pipeline stages on distinct
cores with a hand-off buffer between them".

**LoopTree's mapping-space ideas** (from <https://arxiv.org/html/2409.13625v4>, §III–IV):

- *Inter-layer choices* (§III, Table IV): which ranks of the **last** einsum to partition; the tile
  shape; a **tile processing schedule** = permutation of partitioned ranks; a per-intermediate
  **retain-recompute** choice; a per-tensor **retain-refetch** choice; and **parallelism**:
  sequential or pipelined across layers.
- *Tile-shape inference* (§IV-A): only the last layer's tiling is specified; every earlier layer's
  tile shape is **derived** by walking data dependences backwards, subtracting retained data, and
  recursing. Done with polyhedral sets via ISL, so storage is proportional to tiling steps, not
  network size.
- *The retention encoding* (§III-D) is the compact bit: for each tensor you name **the last
  partitioned rank spanned by the retained tile** (one of the partitioned ranks, or none). That
  single choice determines what stays on-chip, what is refetched, and what must be recomputed —
  the paper unifies recomputation and refetch: "if we specify a processing schedule and retention
  choice such that an operation needs to access an intermediate fmap activation that is not
  retained… we have to recompute."
- Cost model: same three-stage shape, ISL sets instead of hyper-rectangles, Accelergy for energy,
  and a pipeline latency formula rather than a pure max. Validated to **<4% error** against DepFin,
  Fused-layer CNN, ISAAC, PipeLayer and **FLAT (transformer self-attention)** (§V).
- Results worth knowing (§VI): partitioned-rank/schedule choice moves required buffer capacity by
  **up to 10×**; per-tensor retention choices cut capacity by **up to 9×** vs uniform retention; and
  tiled fusion only beats layer-by-layer above a buffer-capacity threshold.

**The limitation, and it is severe for us.** In `NVlabs/timeloop` LoopTree is a **model, not a
mapper**. `LooptreeModel` parses the mapping straight out of the config
[`src/applications/looptree-model/model.cpp:29-33`]:

```cpp
workload_ = problem::ParseFusedWorkload(rootNode.lookup("problem"));
mapping_  = mapping::ParseMapping(rootNode.lookup("mapping"), workload_);
```

There is **no `FusedMapSpace`, no fused search, and no fused constraint system** — I grepped for any
`FusedMapping` use in a mapper/search/mapspace context and found none. The paper agrees by omission:
it "does not explicitly describe automated legality checking or search space pruning algorithms";
its case studies hand-search selected axes while holding others fixed (§VI, Table IX).

So: **Timeloop does single-einsum mapping-space search; LoopTree does multi-einsum modelling with a
human in the mapping seat.** Nobody in this lineage ships a fused *mapper*.

### 6.3 The other descendants, briefly

- **Sparseloop** (Timeloop v2, MICRO 2022, <https://arxiv.org/abs/2205.05826>): adds a taxonomy of
  three *sparse acceleration features* — representation format, gating, skipping — and stochastic
  density models. >2000× faster than cycle-level simulation, 0.1–8% average error. Orthogonal to us.
- **Ruby** (Timeloop v3, ISPASS 2022, Horeni et al., <https://ieeexplore.ieee.org/document/9804679>):
  admits **imperfect factorization** (remainders) into the IndexFactorization sub-space, plus spatial
  skews and flattened mappings. Reported EDP improvement up to 50% (avg 20%) for ResNet-50 on an
  Eyeriss-like array. Selected via `mapspace: {template: "ruby"}`. **Directly relevant** — our tile
  sizes rarely divide our shapes evenly.
- **Orojenesis** ("Mind the Gap", ISCA 2024, <https://timeloop.csail.mit.edu/orojenesis>, and
  `orojenesis/` in-tree): computes, as a function of on-chip buffer capacity, a **lower bound on
  DRAM traffic that no mapping can beat** — "ski-slope" curves — **including mappings that fuse a
  sequence of tensor operations to exploit producer-consumer reuse**. In the mapper this shows up as
  `log_orojenesis_mappings_` / `PrintOrojenesis`, which emits the best mapping *per index
  factorization* rather than one global best [`mapper-thread.cpp:448-471`]. Useful to us as a way to
  answer "is my hand-written builder near the achievable floor?" without a search.
- **Union** (PACT 2021, <https://arxiv.org/abs/2109.07419>) — worth reading because it is **MLIR**:
  TensorFlow/COMET → TOSA → Linalg → Affine, with "cost-model dependent conformability passes" that
  check whether an op is expressible in a given cost model, then a `Union problem` + `Union mapping`
  abstraction under which mappers (exhaustive, Timeloop's random sampling, Marvel's decoupled) and
  cost models (Timeloop, MAESTRO) are interchangeable (§III–IV). Its Table I classifies Timeloop as
  operation-abstraction *nested loops*, mapping abstraction *memory-target loop-centric*, hardware
  abstraction *hierarchical*.

---

## 7. What transfers to an AIE2P 8×4 array — honestly

### Transfers well

1. **The analytic bottleneck cost model as a balance instrument.** [ISPASS19 §VI-D] latency = max
   over per-resource isolated cycles, and for any communication interface, cycles = traffic ÷
   bandwidth. We can build this in a day for our target: enumerate resources = {each core's compute,
   each shim MM2S/S2MM channel, each memtile port, DRAM}, compute per-resource traffic from the
   tiling, and report `max` plus the argmax. We said we have "no balance instrument"; this *is* the
   balance instrument, and it is ~100 lines given traffic counts. It also directly names which knob
   to turn, which a scalar cycle count does not.
2. **The capacity-cliff / bandwidth-slope split.** Timeloop rejects on capacity and *slows down* on
   bandwidth [`buffer.cpp:2565-2599`]. Our packet-multiplexing is a bandwidth-slope phenomenon.
   [inference] Rather than trying to make >2 MM2S per column inexpressible, model it: per-column
   demand `d` channels against budget 2 gives `slowdown = min(1, 2/d)` on that segment. Then the
   illegal-looking point produces a correctly-pessimistic prediction instead of a silent surprise —
   which is what we actually want from a search, since the compiler will accept it anyway.
   What *should* be inexpressible is the structural part: which dataspaces a column's mapping may
   name at all. That belongs in the constructed-in constraint tier.
3. **The Cartesian-counter mapspace + `Report(status)` feedback pruning.** A mapping = an integer
   over a product of independent sub-spaces; the searcher gets typed failure feedback and uses it to
   fast-forward whole sub-cubes [`linear-pruned.cpp:153-190`]. Our three axes (tiling / pipelining /
   parallelizing) are exactly such a product. The inference rules generalise: a *placement* failure
   invalidates the innermost axis for a fixed prefix; an all-inner-values *resource* failure
   invalidates the whole outer point. This gives a searcher that prunes hard with no solver.
4. **Unit-factor permutation pruning** (`InitPruned`, `uber.cpp:409-459`): if a dimension's tile
   factor at a level is 1, its position in that level's order is meaningless — drop it from the
   permutation space and shrink the spatial-split space accordingly. Trivial to implement, large
   constant-factor win.
5. **Typed rejections with reasons + a fail histogram + remediation text**
   [`mapper-thread.cpp:566-635`, `mapper.cpp:470-500, 645-660`]. Our legality is implicit in
   hand-written builders. The cheapest upgrade is not a solver but a `check(mapping) ->
   vector<Status{success, class, level, reason}>` that our builders call, plus a counter. It converts
   "the build failed" into "37 candidates died on per-column MM2S at column 3".
6. **Victory condition as the termination shape.** "Stop after N candidates with no improvement to
   the best" [`mapper-thread.cpp:413-421`] is the right stopping rule for *our* setting, where we
   search balance on real hardware and each evaluation is expensive. Pair it with the second
   Timeloop counter — N *consecutive invalid* points — which is the signal that our constraints are
   over-tight rather than that we have converged.
7. **LoopTree's "specify the last einsum's tiling, infer the rest by dependence"** (§IV-A) plus its
   **retention encoding** (§III-D: name the last partitioned rank the retained tile spans). For a
   transformer layer this collapses per-stage `tile_m/n/k`, `emb_tile`, `seq_tile`, `kv_seq_tile`
   into one tile spec on the final stage plus one small retention choice per intermediate — and that
   retention choice is precisely "does this intermediate stay resident in this segment or get
   re-fetched/re-computed". That is our on-chip-residency-within-one-segment constraint in a form a
   searcher can enumerate.
8. **The Pipeline/Sequential mapping-tree node types** [`fused-mapping.hpp:135`]. If we build a
   mapping IR, this is the shape to copy: a tree where `Pipeline{branches}` means concurrent spatial
   occupancy by different kernels and `Sequential{branches}` forces an intermediate to be buffered
   between them. It expresses our space; we would supply the mapspace and search it lacks.
9. **Union's decoupling, in MLIR** — a `problem` abstraction and a `mapping` abstraction with
   pluggable mappers and cost models, and "conformability passes" that check up front whether an op
   is even expressible in a given cost model. Good architectural template for mlir-air.

### Does not transfer

1. **Homogeneous spatial replication.** A `parallel_for` distributes *one einsum's* operation space
   across instances, and `AssignSpatialTilingDirections_Level_Expand` "assumes that spatial tiling
   will expand the instances for *each* datatype exactly by the tiling parameters"
   [`uber.cpp:854-857`; ISPASS19 §V-C]. Every instance runs the same nest. **Our pipelining axis —
   distinct stages running distinct code on distinct cores — is not expressible in base Timeloop's
   mapspace at all.** Only LoopTree's `Pipeline` node reaches it, and that has no search.
2. **No shared or global resource budgets.** Bandwidth lives on a storage *level*
   (`read_bandwidth` / `write_bandwidth` / `shared_bandwidth`, `num_ports`, `num_banks`,
   `buffer.cpp:389-445`) and is applied per instance. The network classes carry **no bandwidth field
   whatsoever** — only `word_bits`, energy and `fill_latency`/`drain_latency`
   (`network-simple-multicast.cpp`, `network-reduction-tree.cpp`, `network-legacy.cpp`). There is no
   way to express "these levels share one 2-channel budget, per column, across the whole segment".
   Our nastiest constraint is the one the framework cannot state.
3. **The uniform-mesh fanout derivation.** Fanout is `inner_meshX / outer_meshX` between adjacent
   storage levels, and it must divide evenly — `assert(inner_meshX % outer_meshX == 0)`
   [`arch-properties.cpp:60-91`]. Our shim/memtile/core structure fits loosely, but a segment
   occupying 3 of 8 columns is not "fanout 3 from a meshX-8 level" without fudging the arch spec.
4. **The no-stall pipeline assumption.** "the overall latency is the maximum of isolated execution
   cycles… assumes negligible pipeline stalls… reasonable for architectures that use
   double-buffering" [ISPASS19 §VI-D]. Fill/drain is a *constant* per level
   (`network_fill_latency` / `network_drain_latency`, `buffer.cpp:138-158`), not a queueing model.
   Our L1→L1 hand-offs with finite BD counts and ping-pong depth are exactly the case this abstracts
   away — and it is exactly where the paper's own six 78–88% accuracy outliers came from
   [ISPASS19 §VII-C].
5. **Perfect factorization by default.** Base `uber` enumerates only exact cofactorings; padded and
   remainder tiles need the `ruby` template. Worth adopting Ruby's relaxation from day one rather
   than inheriting the restriction.
6. **Energy/EDP as the headline objective**, backed by a TSMC-16nm technology model and Accelergy
   ERT/ART. We care about cycles. The metric plumbing is configurable
   (`optimization_metrics`), but the whole validation story and half the machinery is energy.
7. **Single-segment scope.** Timeloop evaluates one workload against one static architecture. It has
   no notion of a mapping being valid only *within* a segment, of segment boundaries flushing
   residency, or of a multi-segment program. That is our structure and it has no analogue.

---

## Comparable summary

- **Data-space representation.** One einsum: a list of named iteration dimensions plus, per tensor,
  an affine sum-of-products **projection** from the iteration space to the tensor's ranks, with
  named coefficients for stride/dilation and `read_write: True` marking outputs
  (`problem-shapes/gemm_ABZ.yaml`; [ISPASS19 §V-A]). No dependence graph — iterations are asserted
  freely reorderable. The v4 `FusedWorkload` adds multiple einsums with ISL-based projections and a
  producer/consumer index.

- **Mapping-space representation.** A mapping is one flat loop nest over all problem dims, cut into
  tiling levels by storage boundaries, every loop tagged temporal/SpaceX/SpaceY, plus a per-level
  per-dataspace bypass mask. The mapspace is a Cartesian product of **four** sub-spaces addressed by
  one uint128 ID: `IndexFactorization × LoopPermutation × Spatial × DatatypeBypass`
  (`mapspace-base.hpp:44-60`) — note the ISPASS19 §V-E text names only three and omits `Spatial`.
  LoopTree replaces the nest with a **mapping tree** of `{Root, For, ParFor, Storage, Compute,
  Pipeline, Sequential}` (`fused-mapping.hpp:135`).

- **Legality model.** Two tiers. Architecture-imposed *structure* is **constructed into** the
  enumeration via mapspace constraints — "a mapping sampled from the mapspace is therefore
  guaranteed to obey those constraints" [ISPASS19 §V-E]. Hardware *resources* are **checked and
  rejected** afterwards, in three staged gates (construction/fanout → pre-eval capacity → full
  eval), each returning `{success, fail_reason}` per level. The constructed space is admittedly not
  dense: "the space of *legal* mappings isn't dense (unfortunately), so a mapping ID may point to an
  illegal mapping" [`mapper-thread.cpp:541`]. **Bandwidth is deliberately not a legality condition
  at all** — over-subscription becomes a multiplicative slowdown [`buffer.cpp:2565-2599`].

- **Search strategy.** Five algorithms — `exhaustive`, `linear_pruned`, `random`, `random_pruned`,
  `hybrid` (default: random over IndexFactorization, linear-pruned over the rest) — over the uint128
  ID, parallelised by splitting the IndexFactorization axis across threads. Tamed by (a) per-IF
  structural re-pruning that drops unit-factor loops from the permutation space, and (b) typed
  failure feedback that fast-forwards whole sub-cubes. Four termination conditions: `search_size`
  (N valid), **`victory_condition`** (N since last improvement, default 500), `timeout` (N
  consecutive invalid, default 1000), mapspace exhausted. The 2019 paper describes only
  exhaustive-linear and random sampling; everything else is repo-only.

- **Cost model.** Analytic, no simulation. Point-set/delta tile analysis (deltas reveal reuse,
  multicast and systolic forwarding) → per-component action counts → energy from a technology model
  or from Accelergy's **ERT/ART** tables (Timeloop owns counts, Accelergy owns J-per-action and
  area). Latency = **max over per-resource isolated cycles**, each interface's cycles = traffic ÷
  bandwidth, assuming no pipeline stalls. Validated at mean 95% cycle accuracy (78–99% range) and
  energy within 8% across 107 DeepBench workloads [ISPASS19 §VII-C].

- **Multi-op support.** Base Timeloop: **none** — one einsum per invocation; "we leave exploration of
  cross-layer reuse to future work" [ISPASS19 §V-A]. LoopTree (v4, in-tree) **models** fused chains
  with explicit `Pipeline`/`Sequential` nodes, infers all earlier layers' tiles from the last layer's
  tiling by dependence, and encodes residency-vs-refetch-vs-recompute as one "last retained rank" per
  tensor — validated <4% error including transformer attention (FLAT). But it ships as a **model
  only**: no fused mapspace, no fused search, no fused constraints. Orojenesis gives fusion-aware
  *lower bounds* on DRAM traffic without search.

- **Single most transferable idea.** The **max-over-per-resource-isolated-cycles bottleneck model**,
  where every communication interface contributes `traffic ÷ bandwidth`. It converts a mapping into a
  per-resource balance report with an argmax that names the offending resource — the balance
  instrument we lack — and its natural extension is to treat per-column MM2S over-subscription as a
  `min(1, 2/demand)` slowdown so silently-multiplexed points get correctly-bad numbers instead of
  looking free. (Runner-up: the typed-rejection + `Report(status)` sub-cube pruning loop.)

- **Single biggest mismatch with our target.** Timeloop's spatial axis **replicates one einsum's nest
  homogeneously across a mesh**; it cannot express distinct pipeline stages running distinct code on
  distinct cores, which is our central axis. Compounding it: no shared/global resource budget exists
  anywhere in the model (networks have no bandwidth field at all), so our per-column, per-segment
  two-MM2S budget is unstateable. LoopTree's `Pipeline` node reaches the first half of that gap —
  but brings no mapspace and no search.
