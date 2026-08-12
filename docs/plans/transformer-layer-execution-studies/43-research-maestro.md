# MAESTRO — research notes

**Target of comparison:** AMD NPU2 (AIE2P), 8 columns × 4 rows = 32 cores, shim tiles (DRAM), memtiles (L2); mapping axes = tiling / pipelining / parallelizing; hard constraint = 2 shim MM2S channels per column, budgeted per-column across a whole segment, silently packet-multiplexed when exceeded.

## Sources used

Primary, all fetched:

- **Paper (extended arXiv version of the MICRO-52 paper)** — Kwon, Chatarasi, Pellauer, Parashar, Sarkar, Krishna, *"Understanding Reuse, Performance, and Hardware Cost of DNN Dataflows: A Data-Centric Approach Using MAESTRO"*, arXiv:1805.02566 — <https://arxiv.org/abs/1805.02566> (PDF: <https://arxiv.org/pdf/1805.02566>). MICRO-52 version: <https://doi.org/10.1145/3352460.3358252>, pp. 754–768. Section numbers below refer to the arXiv PDF. I diffed the section structure against the Georgia Tech–hosted MICRO-52 PDF (<https://bpb-us-e1.wpmucdn.com/sites.gatech.edu/dist/c/332/files/2019/11/maestro_micro2019.pdf?bid=332>) and they are identical (§3.1, §3.2, §4.2, §4.3 and the top-level headings land within a few lines of each other), so the references hold for either version.
- **IEEE Micro Top Picks 2020 restatement** — Kwon et al., *"MAESTRO: A Data-Centric Approach to Understand Reuse, Performance, and Hardware Cost of DNN Mappings"*, IEEE Micro 40(3):20–29 (cited from the repo README).
- **Source repo** — <https://github.com/maestro-project/maestro>. Specific files cited inline by path.
- **Docs** — <https://maestro.ece.gatech.edu/docs/build/html/index.html> (pages: `examples/running_maestro.html`, `layer_supported.html`, `hw_supported.html`).

Secondary (the search layer and the fusion gap), used in §5–§6:

- **GAMMA** — Kao & Krishna, ICCAD 2020. **Paper is closed-access** (Unpaywall: `oa_status: closed`, no repository copy); details come from the authors' MICRO 2020 tutorial deck <https://maestro.ece.gatech.edu/docs/build/html/_downloads/0e743fdb154d76f274328250ab70b9ec/7_GAMMA.pdf> and the released code <https://github.com/maestro-project/gamma>.
- **Marvel** — Chatarasi et al., TACO 19(1):6 2021 — <https://arxiv.org/pdf/2002.07752>, free published version <https://par.nsf.gov/servlets/purl/10348661>.
- **ConfuciuX** — Kao, Jeong, Krishna, MICRO 2020 — <https://arxiv.org/pdf/2009.02010>.
- **DiGamma** — <https://arxiv.org/pdf/2201.11220>. **DNNFuser** — <https://arxiv.org/pdf/2201.11218>. **FLAT** (ASPLOS 2023) — <https://arxiv.org/pdf/2107.06419>. **MAGMA** — <https://arxiv.org/abs/2104.13997>. **AIrchitect v2** — <https://arxiv.org/pdf/2501.09954>.

Throughout I mark **[paper]** for a claim made in the paper and **[code]** for something I verified in the released source, and I flag where they disagree. Inferences are marked **[inference]**.

---

## 1. The data-centric representation

### 1.1 The formal definitions, verbatim

From §3.1 [paper], the dataflow is defined to consist of "(1) the schedule of DNN computations (e.g., choice of loop transformations) across time for exploiting a wide range of reuse, and (2) the mapping of the DNN computations across PEs for parallelism", built from four components:

> **(1) SpatialMap(size, offset) α** specifies a distribution of dimension α (e.g., R, X) of a data structure across PEs, where `size` refers to the number of indices mapped in the dimension α to each PE, and `offset` describes the shift in the starting indices of α across consecutive PEs.
>
> **(2) TemporalMap(size, offset) α** specifies a distribution of dimension α of a data structure across time steps in a PE, and also the mapped chunk of dimension indices is the same across PEs in a time step. The `size` refers to the number of indices mapped in the dimension α to each PE, and `offset` describes the shift in the starting indices of α across consecutive time steps in a PE.
>
> **(3) Data Movement Order:** The sequence of spatial and temporal maps in the dataflow specification dictate the order of data movement, i.e., the change of the data mappings to PEs across time.
>
> **(4) Cluster(size)** — introduced in §3.2.

Three details that matter and are easy to miss:

- `α` is a **dimension of a data structure**, not a loop variable. `SpatialMap(2,2) X'` maps the *output tensor's* `X'` dimension. This is the whole point of the name "data-centric".
- **`offset < size` produces deliberate overlap** across consecutive PEs — this is how halo/sliding-window (convolutional) reuse is expressed: "if offset value is smaller than size value, then there will be an overlap of indices across consecutive PEs, and this is useful in describing mappings on input activation dimensions X and Y because their iteration space is skewed" (§3.2) [paper].
- **Folding is implicit**: "If the number of PEs is not sufficient to cover all indices of the dimension mapped, then the mapping is folded over time across the same set of PEs" (§3.2) [paper]. A `SpatialMap` over a dimension bigger than the array is legal and silently becomes time-multiplexed.

### 1.2 The concrete surface syntax [code]

The full token set of the DSL, from `cost-model/include/dataflow-specification-language/DFSL_syntax_tokens.hpp`:

```
Constant | Network | Layer | Type | Stride | Dimensions | Dataflow
  Type ∈ { CONV, DSCONV, TRCONV, NGCONV, FC, POOL, LSTM, GEMM, RESIDUAL_IDENTITY }
  Dimensions (CONV) ∈ { N, G, K, C, R, S, Y, X, Y', X' }
  TemporalMap | SpatialMap | Cluster | Sz
  Cluster type ∈ { L (logical), P (physical) }
```

`Sz(R)` resolves to the **declared size of dimension R** for this layer — a symbolic "the whole of it", so a dataflow can be written once and reused across layers of different shapes. Verified in `DFSL_parser.hpp` [code], `Sz(...)` is accepted in *three* positions — map size, map offset, and **cluster size** — which is why the paper's Table 3 can write `Cluster(Sz(R))` to make the cluster width track the filter height. Combined with `Constant` declarations (see the GEMM example below), this is how the shipped mappings stay shape-independent.

A **real** mapping, verbatim from `data/mapping/Resnet50_kcp_ws.m` (the NVDLA-like K–C-partitioned weight-stationary dataflow):

```
Network Resnet50 {
	Layer CONV1 {
		Type: CONV
		Stride { X: 2, Y: 2 }
		Dimensions { K: 64, C: 3, R: 7, S: 7, Y:224, X:224 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(64,64) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			Cluster(64, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	...
}
```

Read that as: **above** the `Cluster(64,P)` line, `K` is spread across *clusters*; **below** it, `C` is spread across the 64 PEs *inside* each cluster. Note `TemporalMap(Sz(R),1) Y` — size = full filter height, offset = 1: that is exactly the sliding window, written in one line.

The GEMM form is much closer to what we care about. Verbatim, `data/mapping/GEMM_Example.m`:

```
// BLAS 3 - Dense Matrix-Dense Matrix multiplication
// Constants are in GEMM convention; (MxK matrix) x (KxN matrix) = (MxN matrix)
Constant SzM 100;
Constant SzN 100;
Constant SzK 100;

// Optimize for throughput
Constant MTileSz 1;
Constant NTileSz 1;
Constant KTileSz 100;

Network BLAS3 {
	Layer BLAS {
		Type: GEMM
		Dimensions { K: SzK, M: SzM, N: SzN }
		Dataflow {
			TemporalMap(NTileSz,NTileSz) N;
			SpatialMap(MTileSz,MTileSz) M;
			TemporalMap(KTileSz,KTileSz) K;
			Cluster(KTileSz, P);
			TemporalMap(NTileSz,NTileSz) N;
			TemporalMap(MTileSz,MTileSz) M;
			SpatialMap(1,1) K;
		}
	}
}
```

Here `K` is the **contraction** dimension (GEMM convention, per the file's own comment), so this is: M spread across clusters, K spread within a cluster (a spatially-reduced inner product), N walked temporally. The commented-out block in the same file shows the same GEMM re-expressed as a `CONV` — useful evidence that the `Dimensions`/`Dataflow` split is the same machinery either way.

And the three-level version showing what `Cluster` really means, from Fig. 4(c) [paper] (a 1D convolution, `O[x'] += W[s]*I[x'+s]`, on 3 PEs):

```
—— Map Target: On-Chip Global Buffer ——
TemporalMap (6, 6) X'
TemporalMap (6, 6) S
Cluster (NumPEs=3)          // We have one global buffer
—— Map Target: PE L1 buffer ——
SpatialMap (2, 2) X'
TemporalMap (3, 3) S
Cluster (1)                 // Each PE includes one L1 buffer
—— Map Target: PE L0 buffer (Reg) ——
TemporalMap (1, 1) X'
TemporalMap (1, 1) S
```

The paper's own point about this listing (Fig. 4 caption): the gray-boxed directives are **omittable** — they are either inferred, or they do not affect reuse over PEs. The abbreviated form of the same dataflow is just:

```
SpatialMap (2, 2) X'
TemporalMap (3, 3) S
```

The defaulting rule is concrete [code] — `UnrollMissingDirectives` in `DFA_cluster-analysis.hpp`: for every dimension of an input tensor that has no directive at a given cluster level, MAESTRO inserts

```cpp
auto dummy_directive = std::make_shared<DFA::directive::TemporalMap>(dim_sz, dim_sz, var);
```

i.e. **an omitted dimension defaults to `TemporalMap(full_size, full_size)`** — "the whole dimension, held, not partitioned". That is why the shipped mappings can be short, and it matches the paper's Fig. 6 annotation "Omittable (Automatically inferred if not specified)". **[inference]** Worth stealing as an ergonomic default: unmentioned axis ⇒ resident in full, which is both the safe reading and the one that makes short mappings mean something definite.

### 1.3 What data-centric buys, and what it costs — my read

This is the part worth internalising. The equivalent loop-nest for that same 1D conv is (§2.5, Fig. 4(b)) [paper]:

```
for (x'2=0; x'2 < 2; x'2++)
 par_for (s2=0; s2 < 1; s2++)
  par_for (x'1=0; x'1 < 3; x'1++)
   for (s1=0; s1 < 2; s1++)
    for (x'0=0; x'0 <2; x'0++)
     for (s0=0; s0 < 6/2/1; s0++)
      s  = 2*3*s2 + 3*s1 + s0
      x' = 3*2*x'2 + 2*x'1 + x'0
      O[x'] += W[s] * I[x'+s]
```

Both descriptions denote the same execution. The difference is **what is a first-class term and what has to be derived**.

**What the data-centric form makes easy:**

1. **Data placement is stated, not solved for.** In the loop nest, "which PE holds which slice of `W` at time t" is a *consequence* of loop order + tile factors + the index expressions `s = 2*3*s2 + 3*s1 + s0`. You recover it by composing affine maps. In the directive form it *is* the notation: `SpatialMap(2,2) X'` says directly, in terms of the tensor, who holds what. §2.5 [paper]: "Since data movement is explicit in the data-centric representation, our analytical model becomes simpler and relatively faster."
2. **Spatial reuse becomes a syntactic property.** This is the sharpest win. Because a `TemporalMap` guarantees "the mapped chunk of dimension indices is the same across PEs in a time step" (§3.1), *every* `TemporalMap` is by definition a multicast opportunity, and every `SpatialMap` on an output dimension is by definition a reduction opportunity. You read reuse off the directive kind. The paper is explicit that prior compiler reuse analyses "did not consider spatial reuse ... that leverages multicasting and reduction support of accelerators" (§2.5) — spatial reuse is not spatial *locality*, it is reuse over wires, and a sequential loop nest has no term for it.
3. **Stationarity is directive order, and nothing else.** §3.2 [paper]: swapping `SpatialMap(1,1) X'` / `TemporalMap(1,1) S` (Fig. 5A) to the reverse order (Fig. 5B) turns an output-stationary dataflow into a weight-stationary one. The paper draws the moral bluntly: "This indicates why the informal dataflow name should not be taken as a complete and precise specification of its behavior." A 2-element enumeration replaces an argument.
4. **It is short.** Eyeriss v2's dataflow "is described in a 22-dimensional loop nest" (§2.5); the equivalent directive list is ~6–12 lines, and roughly half are omittable.
5. **Non-affine index math stops being a blocker.** §2.5 [paper] notes polyhedral reuse analysis struggles with "array subscripts involving non-affine expressions or complex subscripts, such as modulus operations which are common in strided convolutions". The directive form sidesteps it because it never analyses subscripts — it declares tensor slices.

**What the loop-nest form makes easy and this form makes hard:**

1. **Arbitrary composition and dependence.** A loop nest can express anything you can write; the directive list is a *fixed grammar of one ordered sequence of maps per cluster level*. If your execution is not "one operator, hierarchically tiled, all PEs doing the same thing", you cannot write it down. This is the reason MAESTRO is single-operator (§6 below).
2. **Correctness/dependence reasoning.** A loop nest exposes iteration order and thus dependences; polyhedral tools can prove a transform legal. Directives assert a data distribution and carry no dependence model at all — legality is a small set of structural well-formedness rules (§4 below), not a proof.
3. **Anything with per-PE heterogeneity.** The docs state the limit outright: "MAESTRO supports any level of hierarchies but only supports **uniform clusters**" (<https://maestro.ece.gatech.edu/docs/build/html/hw_supported.html>). Loop nests with guards can express skew and specialisation; directives cannot.
4. **It is not the input to a code generator.** The directive form is an analysis IR. It abstracts away exactly the addressing math a DMA descriptor needs.

**[inference]** The relevance to us: our problem statement is "dataflow between stages", which sounds like it should suit a data-centric IR. It partly does — the *tiling* axis maps almost directly onto `SpatialMap`/`TemporalMap` sizes and offsets, and `offset < size` is the right way to talk about `kv_seq_tile` overlap. But MAESTRO's data-centrism is about *where a tensor slice lives across a homogeneous PE array*, not about *which stage of a pipeline a core is running*. Our pipelining axis has no counterpart in the notation. See §6 and §7.

---

## 2. Reuse analysis and communication cost

### 2.1 How reuse is derived

**Structurally, from directive kind and order.** The Reuse Analysis (RA) engine "identifies the amount of temporal and spatial reuse across adjacent time steps, which is the data iteration corresponding to the inner-most non-temporally/spatially unrolled mapping directive" (§4.1) [paper]. So reuse is a *delta between two adjacent time steps*, not a classical reuse distance.

The mechanism is a lookup keyed on **(spatially mapped dimension, innermost temporally mapped dimension)**. Table 1 [paper] enumerates it for CONV: e.g. if `K` (output channels) is spatially mapped, the input feature map is *not coupled* to `K`, so all PEs receive the same input tile ⇒ full spatial multicast; and if the innermost temporal map is `C`, then outputs accumulate across time ⇒ temporal reduction. The generalisation is the **dimension-coupling table** built by the Tensor Analysis engine (§4.1): each tensor declares which dimensions it is coupled to, and "MAESTRO allows users to specify tensors with arbitrary dimension coupling, and such coupling relationship is input to the rest of engines, which provides generality."

That is the entire reuse rule, in one sentence: **a tensor is spatially reused iff the spatially-mapped dimension is not one of its coupled dimensions; it is temporally reused iff the innermost temporally-mapped dimension is not one of its coupled dimensions.** Multicast if it is an input tensor, reduction if it is the output tensor (Table 2) [paper].

I verified this is literally how the code works [code]: `cost-model/include/cost-analysis/CA_reuse-analysis.hpp` (1,722 lines) walks each tensor's coupled-dimension list and gates traffic on the conjunction

```cpp
bool is_coupled_dim = (std::find(coupled_dims->begin(), coupled_dims->end(), dim) != coupled_dims->end());
...
if(is_coupled_dim && is_changing_dim) { ... }
```

— i.e. a tensor needs fresh data only along dimensions it is *coupled to* that are *changing* between adjacent time steps. `GetSpatialIngressTraffic` / `GetSpatialEgressTraffic` are built directly on that test.

**Note what this is not.** There is no reuse-distance histogram and no cache model. There is no dependence analysis. It is a coupling test plus tile arithmetic — which is exactly why it runs fast enough to be a fitness function.

### 2.2 What it computes — the cost model, verbatim

§4.2/4.3 and Fig. 8 [paper]. The core loop is over **data iteration cases** — "the cross product of all the possible data iteration cases (Init, Steady, and Edge) of each data dimension" — with each case weighted by its occurrence count:

```
iteration_cases = ExtractDataIterationCases(tensor_tbl, cluster_info_tbl);
for each iter_case in iteration_cases
  num_case_occurrences  = GetNumCaseOccurrences(...)
  num_psums             = GetNumPSums(...)
  cluster_ingress_traffic = ...   // new input data fetched from the buffer one level up
  cluster_egress_traffic  = ...   // output data committed to the buffer one level up

  //// Core cost analysis ////
  stats.upstream_buffer_read[t]   += cluster_ingress_traffic[t];
  stats.downstream_buffer_write[t]+= cluster_ingress_traffic[t];
  stats.upstream_buffer_write[t]  += cluster_egress_traffic[t];
  stats.downstream_buffer_read[t] += num_psums;
  stats.upstream_buffer_size_req[t]   = 2*Max(stats.upstream_buffer_size_req[t],
                                              cluster_ingress_traffic[t],
                                              cluster_egress_traffic[t]);
  stats.downstream_buffer_size_req[t] = 2*Max(stats.downstream_buffer_size_req[t],
                                              num_psums, cluster_egress_traffic[t]);

  //// Core performance analysis ////
  ingress_delay = GetDelay(cluster_ingress_traffic, hw_model);
  egress_delay  = GetDelay(cluster_output_traffic,  hw_model);
  compute_delay = GetComputeDelay(num_psums, hw_model);
  compute_delay += GetPSumFwdDelay(...);
  /* Considers double-buffering; treats the initialization case as an exception */
  if IsFullInit(iter_case) then
      outstanding_delay = ingress_delay + compute_delay + egress_delay;
  else
      outstanding_delay = Max(ingress_delay, egress_delay, compute_delay);
  end
  stats.run_time += outstanding_delay * num_case_occurrences;
  stats.num_macs += num_psums * num_active_clusters * num_case_occurrences;
end
```

So: **buffer requirement = 2 × worst-case per-iteration traffic** (the `2×` is double-buffering), and **runtime = Σ over iteration cases of max(ingress, egress, compute) × occurrences**. The whole model is a per-level, per-iteration-case roofline. It is not a simulator; there is no queueing and no contention between concurrent flows.

I verified the released code matches (`cost-model/include/cost-analysis/CA_cost-analysis-engine.hpp`) [code]:

```cpp
long ingress_comm_delay = noc->GetOutStandingDelay(ingress_spatial_traffic);
long egress_comm_delay  = noc->GetOutStandingDelay(egress_spatial_traffic);
outstanding_delay = (do_double_buffering)
    ? std::max(egress_comm_delay, std::max(computation_delay, ingress_comm_delay))
    : ingress_comm_delay + computation_delay + egress_comm_delay;
```

### 2.3 How communication cost is obtained — this is the part that maps onto our DMA budget

Two separate things, and it is important to keep them apart.

**(a) Communication *delay* — the "pipe model".** §4.2 [paper]: "MAESTRO relies on its analytical network-on-chip (NoC) model based on a pipe model ... The pipe model utilizes two parameters, the pipe width (bandwidth) and length (average delay)". The actual implementation, verbatim from `cost-model/include/abstract-hardware-model/AHW_noc-model.hpp` [code]:

```cpp
long GetOutStandingDelay(long data_amount) {
    long num_sends;
    if(data_amount % bandwidth_ != 0) { num_sends = data_amount / bandwidth_ + 1; }
    else                              { num_sends = data_amount / bandwidth_;     }
    long avg_zero_load_delay = num_average_hops_ * latency_per_hops_;
    delay = avg_zero_load_delay      // Head delay
          + (num_sends-1);           // Pipeline delay
    return delay;
}
```

That is the entire NoC model: `hops × latency_per_hop + ceil(traffic / BW) − 1`. The paper is candid about its scope: "For more complicated NoC architectures, users should select bisection bandwidth and average latency considering uniform communication to all the PEs from a global buffer ... Assuming that the user has access to the NoC implementation information, **the NoC model is precise when the NoC is a bus or a crossbar**" (§4.2) [paper]. There is one NoC object *per cluster level*, so bandwidth is specified per level of the hierarchy (`ErrorCode::MissingNoCForCluster` fires if a level has none) [code].

**(b) Communication *requirement* — the balance instrument.** Separately from delay, MAESTRO reports the bandwidth the mapping *demands*, per level [code, `CA_cost-analysis-engine.hpp`]:

```cpp
peak_noc_bw_req = std::max(peak_noc_bw_req,
                           std::max(ingress_spatial_traffic, egress_spatial_traffic)/computation_delay);
avg_noc_bw_req += (num_case_occurrences * std::max(ingress_spatial_traffic, egress_spatial_traffic))/computation_delay;
...
off_chip_bw_req = std::max(off_chip_bw_req,
                           results->GetBufferSizeReq(Upstream, Output)/computation_delay);
off_chip_bw_req = std::max(off_chip_bw_req,
                           (results->GetBufferSizeReq(Upstream, Input)
                          + results->GetBufferSizeReq(Upstream, Weight))/computation_delay);
```

i.e. **required bandwidth = bytes moved per iteration ÷ compute time of that iteration** — a pure arithmetic-intensity inverse, computed per level, reported as peak and average. The code also attributes the bottleneck per iteration case (`else if(outstanding_delay == ingress_comm_delay) { ... }` marks ingress as the limiter) [code]. §5.2 [paper] uses these numbers directly: "although an accelerator has sufficient number of PEs to exploit the maximum degree of parallelism a dataflow allows, if the NoC does not provide sufficient bandwidth, the accelerator suffers a communication bottleneck in the NoC."

**And it diffs that requirement against the provisioned budget and says so.** This is the single most relevant thing in the whole tool for us. `API_user-interface-v2.hpp` [code]:

```cpp
bool pass=true;
std::cout << "BW Analysis:"<<std::endl;
if( max_noc_bw_req > configuration_->noc_bw_->at(0)){
  std::cout << "[WARNING:BW] Per-layer NoC BW requirement [" << max_noc_bw_req
            << "] is larger than the given NoC BW [" << configuration_->noc_bw_->at(0) << "]"<< std::endl;
  pass=false;
}
if( max_offchip_bw_req > configuration_->offchip_bw_){
  std::cout << "[WARNING:BW] Per-layer OffChip BW requirement [" << max_offchip_bw_req
            << "] is larger than the given OffChip BW [" << configuration_->offchip_bw_ << "]"<< std::endl;
  pass=false;
}
if(pass==true){ std::cout << "[PASS]"<<std::endl; }
```

where `max_noc_bw_req` is the max of `GetPeakBWReq()` over all layers. So a mapping that oversubscribes the interconnect gets a named warning printing **both the demand and the provision**, plus a failed `[PASS]` — *and* the shortfall is separately folded into runtime via `GetOutStandingDelay`.

**[inference] The load-bearing observation for us:** MAESTRO handles "the resource is oversubscribed" **both ways at once** — as a reported requirement-vs-budget diff, *and* as a cost. It does not need the oversubscription to be a hard error to make it visible. Given that our compiler silently packet-multiplexes, the reported-diff half is precisely the instrument we are missing, and it costs nothing but arithmetic we already have the inputs for.

### 2.4 The full report, as actually printed [code]

Worth listing, because it *is* the balance instrument and it is short (`API_user-interface-v2.hpp`):

```
[Performance Analysis]     Runtime: N cycles ; Throughput: N MACs/cycle ; Num MACs: N
[Buffer Access Analysis]   per tensor: L2 size requirement, L1 size requirement,
                                       L2 buffer write/read, L1 buffer write/read,
                                       Data reuse factor
                           Overall data reuse factor = total_l1_read / total_l1_write
                           Input/Weight multicasting factor = l2_to_l1_wr_count / l2_rd_count
                           num_active_pes ; uppermost & innermost cluster size
Buffer Analysis:           [WARNING:Buffer] ... | [PASS]
                           [Model-wise Buffer Summary] total L1 / L2 usage
BW Analysis:               [WARNING:BW] ... | [PASS]
[Energy Analysis]          activity counts × Cacti-derived per-access energy
```

Two of these deserve a second look for our purposes: **"Input/Weight multicasting factor"** = `l2_to_l1_write_count / l2_read_count` is a direct, dimensionless measure of how much spatial fan-out the mapping achieves (how many L1s each L2 read feeds) — for us, how much one memtile read is shared across the 4 cores in a column. And **"Data reuse factor"** = `L1 read / L1 write` measures temporal reuse in L1. Both are cheap ratios of counters the cost model already accumulates.

**Important mismatch, stated honestly:** our constraint is a *channel count* (2 MM2S per column), not a byte-rate. MAESTRO has no notion of a fixed number of physical channels — three logical flows through a column are just more bytes down one pipe. So a MAESTRO-style model, ported naively, **would not catch our specific violation either**. It would need an added per-column "distinct concurrent flows" counter, which is a different resource dimension from bandwidth. See §7.

---

## 3. Cluster and hierarchy

### 3.1 Semantics

§3.2 [paper], verbatim:

> The cluster directive logically groups multiple PEs or nested sub-clusters (when a dataflow has multiple cluster directives) of `size` parameter. ... **All the mapping directives specified above a Cluster directive perform the mapping across logical clusters** created by the Cluster directive. **All the mapping directives specified below a Cluster directive perform the mapping across PEs or lower level logical clusters inside** a logical cluster created by the Cluster directive. That is, all the mapping directives above a Cluster directive see logical clusters while those below the Cluster directive see inside of each logical cluster.

The stated motivation is that without it, "data mappings related to a map in the outer position get updated after a full exploration of a map in the inner position", which forbids simultaneously parallelising two dimensions. `Cluster` is what lets you have **one `SpatialMap` per level** and thus N-dimensional spatial partitioning. The paper cites Eyeriss (spatial R and Y) and NVDLA (spatial K and C) as the motivating cases, and notes clusters can also stand in for "coarse-grained PEs ... such as SIMD units and matrix tensor accelerators like GPU Tensor Cores".

**Sizing [code].** In `cost-model/include/dataflow-analysis/DFA_cluster-analysis.hpp`, `AnalyzeClusterStructure()` multiplies all declared `Cluster` sizes into `uppermost_cluster_unit_size`, and then the top level gets

```cpp
if(is_top_cluster) { current_cluster_size = num_unit_clusters / uppermost_cluster_unit_size; }
```

So the top-level cluster count is **derived**: `num_pes / Π(declared cluster sizes)`. An innermost `Cluster` of the last declared size is appended automatically.

**The `P`/`L` argument appears to be inert in the released code.** The parser builds `ClusterType::Physical` or `ClusterType::Logical` from the token (`DFSL_parser.hpp`), and `DFA_directives.hpp` stores it and exposes `GetAllocType()` — but the *only* branch on the value anywhere is inside `Cluster::ToString()`, which merely prints `P` or `L` back out:

```cpp
virtual std::string ToString() {
    std::string type_str;
    if(type_ == ClusterType::Logical) { type_str = DFSL::dataflow_cluster_type_logical_; }
    else                              { type_str = DFSL::dataflow_cluster_type_physical_; }
    return "Cluster(" + std::to_string(size_) + "," + type_str + ")";
}
```

I found **no caller of `GetAllocType()`** in any header I fetched, and I fetched the whole parse → cluster-analysis → reuse-analysis → cost-analysis path (`DFSL_parser`, `DFA_cluster-analysis`, `DFA_cluster-unit`, `DFA_cluster-table`, `DFA_directive-table`, `DFA_iteration-analysis`, `CA_reuse-analysis`, `CA_cost-analysis-engine`, `CA_iterations`, `API_user-interface-v2`, `API_configuration`, `BASE_maestro-class`). **[inference, scoped to those files]** `Cluster(64,P)` and `Cluster(64,L)` therefore compute the same cost. All shipped `.m` mappings use `P`, and the auto-appended innermost cluster is hardcoded `Physical`. Treat the physical/logical distinction as documentation, not semantics.

**Per level, each cluster gets its own NoC and its own buffer pair.** The cost engine tracks `BufferType::Upstream` / `BufferType::Downstream` size requirements and access counts at every cluster level (`CA_cost-analysis-engine.hpp`) [code], and §4.4 [paper] describes the recursion: "Multi-cluster cases can be split into single-cluster cases with the data dimension size set as the mapping size of the corresponding mapping directive in the upper cluster. **The outstanding delay of a cluster level becomes the computation delay of the next cluster level above.**" Base case is the innermost cluster whose sub-clusters are real PEs. "the number of PE cluster levels are typically two or three."

### 3.2 Could it express "8 columns × 4 rows with a shared L2 memtile per column"?

**Yes, structurally — that is exactly the shape it is built for.** [inference, but a well-supported one]

With `num_pes: 32` and a single `Cluster(4, P)` in the dataflow, the top level becomes `32/4 = 8` clusters of 4 PEs. The mapping then reads:

```
<directives above Cluster>   // partition across the 8 columns
Cluster(4, P);
<directives below Cluster>   // partition across the 4 rows within a column
```

and the model's levels line up with our hardware as:

| MAESTRO level | Our hardware |
|---|---|
| top-level `Upstream` buffer | DRAM / shim side |
| top-level NoC (bandwidth, hops) | shim → memtile |
| cluster-level `Upstream` buffer | **memtile (L2), one per column** |
| inner NoC | memtile → core |
| innermost `Downstream` buffer | core L1 |
| PE | AIE core |

`offchip_bw_cstr` in the HW file is the DRAM-side budget, and each cluster level carries its own bandwidth — so a per-column feed limit has a natural slot. A minimal HW file is just five lines (`data/hw/accelerator_1.m`) [code]:

```
num_pes: 256
l1_size_cstr: 100
l2_size_cstr: 3000
noc_bw_cstr: 1000
offchip_bw_cstr: 50
```

**Two caveats.**
1. "MAESTRO supports any level of hierarchies but only supports **uniform clusters**" — the docs say explicitly it cannot model "irregular PE distributions across rows or columns" (<https://maestro.ece.gatech.edu/docs/build/html/hw_supported.html>). Our 8×4 is uniform, so this is fine *for a single operator*.
2. The killer: this hierarchy describes **one operator spread over all 32 cores**. There is no way to say "columns 0–1 run stage A, columns 2–7 run stage B". The cluster tree is a broadcast/reduction tree over a single tensor computation. See §6.

---

## 4. Legality

Worth separating the paper's story from the shipped tool's, because they differ.

### 4.1 What the code actually enforces [code]

`cost-model/include/tools/TL_error-handler.hpp` is the complete legality vocabulary. Verbatim messages:

**Structural (fatal — `TerminateProgram()`):**
- `NoSpatialMap` — "Cluster level: N, **No spatial map in a cluster**"
- `MultiParallelismInSingleCluster` — "**Found too many spatial maps within a single cluster.** Cluster level: N"
- `IllegalClusterConstruction` — "**Specified cluster does not cover entire number of PEs**"
- `InvalidCluster` — "Cluster level N **contains directives other than temporal and spatial map**"
- `InvalidClusterLevel` — "Cluster level N does not exist"
- `MissingNoCForCluster` — "NoC is not defined at cluster level N"
- `InvalidDirective` / `InvalidDimension` / `MissingDimension` / `NotSupportedLayerType`
- `DuplicatedDimDefinition`, `DoubleDimDefinition` ("Both input- and output-centric dimension definition is used")

**Tiling/edge-case well-formedness (fatal):**
- `NotEnoughSpDim` — "Dimension α **is not sufficient for conv windows**"
- `EdgeOnSpatialMap` — "**Dataflow cannot have edge on spatial map.** Please check the mapping size of spatial map at cluster level N"
- `IllegalTemporalEdgeSp`, `InvalidTemporalEdgeSz`

The invariant that matters: **exactly one `SpatialMap` per cluster level**, and **the declared cluster sizes must exactly divide the PE count**. Those two are checked at analysis time and are fatal. Everything about *which* dimension you map and at what size is free.

**Capacity (NOT fatal in the released code):** `NotEnoughL1Buffer` and `NotEnoughL2Buffer` exist as error codes, with messages "The required L1 buffer size N is larger than your L1 size. **Reduce the L1 tile size by reducing mapping sizes.**" — but in `cost-model/include/user-api/API_user-interface-v2.hpp` **both call sites are commented out** and replaced by a warning:

```cpp
//            if(layer_wise_total_l1_size > configuration_->l1_size_) {
//                  error_handler_->PrintErrorMsg(TL::ErrorCode::NotEnoughL1Buffer, ...);
//                  error_handler_->TerminateProgram();
//            }
...
          if(min_l1_size_req > configuration_->l1_size_){
            std::cout << "[WARNING:Buffer] Per-layer L1 size requirement [" << min_l1_size_req
                      << "] is larger than the given L1 size [" << configuration_->l1_size_ << "]"<< std::endl;
            pass= false;
          }
```

So capacity overflow prints `[WARNING:Buffer]` and a failed `[PASS]`, and the run continues and reports numbers. **[inference]** This is almost certainly deliberate: for a mapper doing millions of evaluations you want a scored infeasibility, not a process abort.

**Rate (also NOT fatal, but definitely checked):** the same warning tier covers interconnect and DRAM bandwidth — `[WARNING:BW] Per-layer NoC BW requirement [X] is larger than the given NoC BW [Y]` and the equivalent for `OffChip BW`, both feeding the same `[PASS]` flag (code quoted in §2.3). So `noc_bw_cstr` and `offchip_bw_cstr` are *both* compared against computed demand *and* folded into runtime.

So the tiering is:

| Tier | What | Mechanism |
|---|---|---|
| **Fatal** | structural well-formedness (one `SpatialMap`/level, clusters divide `num_pes`, no edge on spatial map, known dims/types) | `PrintErrorMsg` + `TerminateProgram()` |
| **Warning + failed `[PASS]`** | capacity (L1/L2) and rate (NoC BW, off-chip BW) | `[WARNING:Buffer]` / `[WARNING:BW]`, printing demand *and* provision |
| **Absorbed into cost** | the performance consequence of the same rate shortfall | `GetOutStandingDelay` → longer runtime |

### 4.2 Checked, constructed-into, or searched around?

All three, at different layers:

- **Constructed-into**: the one-`SpatialMap`-per-`Cluster`-level rule is a property of the *grammar*. You cannot write two parallel dimensions at one level; you must introduce a `Cluster`. The notation makes a class of illegality unwriteable.
- **Checked**: structural rules and PE-count divisibility, fatal, at analysis time.
- **Searched around / scored**: buffer capacity (a warning + a `pass` flag) and bandwidth (folded into runtime). The DSE tool in §5.2 [paper] treats *hardware* infeasibility (area/power) as a pruning predicate, not an error.

**[inference] The transferable structure here is the split**: illegality that the *notation* can prevent is prevented; illegality that is a *resource inequality* is turned into a reported requirement compared against a provisioned budget. Our per-column MM2S budget is squarely the second kind.

---

## 5. Search

**Confirmed: MAESTRO is an analytical cost model, not a mapper.** The repo's own one-line description is "An analytical cost model evaluating DNN mappings (dataflows and tiling)" (<https://github.com/maestro-project/maestro>). §7 [paper], Discussion and Future Work: "In the future, we plan to leverage MAESTRO to implement **a dataflow auto-tuner to find an optimal dataflow** on the specified DNN model and hardware configuration." That is, in the MICRO 2019 paper, mapping search is explicitly future work. The mapping is an *input file* you write by hand or generate. `tools/frontend/modelfile_to_mapping.py` is 37 lines: it takes `--dataflow` from a fixed menu (`'dataflow choices: ykp_os, kcp_ws, xp_ws, rs'`), reads the corresponding template from `./dataflow/<name>.m`, and stamps it onto every layer of the model [code]. That is templating, not search.

**Careful with the headline DSE number.** The abstract's "searches across **480M** designs to identify **2.5M valid** designs" (§1) is a **hardware** design-space exploration, not a mapping-space one. §5.2 [paper]: the DSE tool "searches four hardware parameters (the number of PEs, L1 buffer size, L2 buffer size, and NoC bandwidth) optimized for either energy efficiency, throughput, or energy-delay-product **within given hardware area and power constraints**", sweeping ranges with a given granularity and skipping subtrees "by checking the minimum area and power of all the possible design points from inner loops". "Valid" there means *meets the area/power budget*, not *is a legal mapping*. Reported rate: 3.3K–0.46M designs/sec, avg 0.17M/s, four DSE runs finishing within 24 min on an i7-8700K. The dataflow is held **fixed** in each sweep (they run one DSE per dataflow, KC-P and YR-P).

**Model fidelity, for calibration:** §4.5 [paper] validates runtime against MAERI RTL simulation (64 PEs, VGG16) and Eyeriss's reported numbers (168 PEs, AlexNet), reporting "**within 3.9% absolute error**" on average. Note this is a validation against two accelerators the same group had access to, on CNNs.

### 5.1 The searchers built on top (all Georgia Tech Synergy Lab)

**GAMMA** (Kao & Krishna, ICCAD 2020) — a domain-specific GA over the mapping space with MAESTRO as the fitness function. *The ICCAD paper is closed-access (Unpaywall: `"oa_status": "closed"`, no repository copy), so the details below come from the authors' own MICRO 2020 tutorial deck and from the released code — labelled accordingly.*

The genome is a **7-tuple per parallelism level**, concatenated across levels (tutorial deck p.5–6; <https://maestro.ece.gatech.edu/docs/build/html/_downloads/0e743fdb154d76f274328250ab70b9ec/7_GAMMA.pdf>):

```
[P, K]  [C,20] [R,3] [S,3] [X,15] [K,64] [Y,10]
 ^par     ^--- 6 (dimension, tile-size) pairs; ORDER = loop order ---^
```

Note the encoding trick: **loop order is positional**, not a separate gene. Confirmed in `src/GAMMA/gamma.py:113-151` [code]: `create_genome()` builds `[[sp, sp_sz]] + [df[i] for i in idx]` with `idx = np.random.permutation(len(df))`, and everything is strided by 7 (`len(ind)//7` = number of levels).

**It emits MAESTRO directives literally and shells out.** `write_maestro()` writes a `.m` file with `SpatialMap(sz,sz) D;` / `TemporalMap(sz,sz) D;` / `Cluster(sz,P);`, then `Popen`s the MAESTRO binary and parses the result CSV; `build.py` git-clones `maestro-project/maestro` at pinned commit `e1d8efd8e5`. **One MAESTRO process per candidate** [code].

Genetic operators — the deck advertises "3 additional genetic operators" beyond crossover/mutation/select, and the code matches one-to-one:

| Deck name | Code (`gamma.py:731-770`) | Acts on |
|---|---|---|
| Crossover | `crossover_tile` | tile sizes |
| Mutation | `mutate_tile`, `mutate_par` | tiles; parallelism dim |
| Reorder | `swap_order` | loop order |
| Growing | `born_cluster` | **adds** a 7-gene level |
| Aging | `kill_cluster` | **drops** a level |

`Growing`/`Aging` are the notable ones: the genome is **variable-length**, so the *number of hierarchy levels* is itself searched.

**Invalid-candidate handling is all three mechanisms at once** [code] — directly relevant to our legality question:
1. **Constrained encoding**: tile sizes sampled within loop bounds; optional `--use_factor` restricts them to *divisors* of the bound (no ragged tiles).
2. **Repair**: `correctify_tile_dependency()` runs every generation and *clamps* each inner level's tile to its enclosing cluster's — `d_sz = min(last_cluster[d], d_sz)`.
3. **Death penalty**: what repair can't fix (L1/L2 overflow, MAESTRO erroring) returns `(None, None)` from `oberserve_maestro`; `evaluate()` sets reward to `-Inf` and shrinks the parent pool (`self.num_parents = min(self.num_parents, len(population) - count_non_valid)`). If *all* candidates die, the population is reinitialised.

**Mapspace size: O(10²⁴)** per layer. Not from GAMMA's own (paywalled) text, but cited to GAMMA by its own authors twice — DiGamma §I "as large as O(10²⁴) **[4]**" where [4] is GAMMA (<https://arxiv.org/pdf/2201.11220>), and DNNFuser §2 "search space as large as O(10²⁴) **[15]**" (<https://arxiv.org/pdf/2201.11218>). Related: HW space O(10¹²), HW×mapping O(10³⁶) (DiGamma §II-C). The deck also quantifies *impact*: VGG16 layer 2 on 168 PEs ranges 8.18×10⁶ → 1.85×10⁹ cycles across mappings — "up to 4 orders of magnitude difference by different mappings".

**Marvel** (Chatarasi et al., TACO 19(1):6, 2021; arXiv:2002.07752) — the mapspace-size source. Free published version: <https://par.nsf.gov/servlets/purl/10348661>. Min/avg/max over CONV2D ops of AlexNet, VGG16, ResNet50, MobileNetV2 (Table 3 / Fig. 10):

| Variant | Min | **Avg** | Max |
|---|---|---|---|
| Original search space | 2.7×10¹⁷ | **9.4×10¹⁸** | 1.8×10¹⁹ |
| Off-chip subspace, decoupled | 7.3×10⁸ | 3.6×10¹¹ | 1.3×10¹² |
| On-chip subspace, decoupled | 2.9×10⁷ | 2.4×10¹⁰ | 1.4×10¹¹ |
| Off-chip, decoupled **+ pruned** | 9.9×10⁵ | **1.5×10⁸** | 6.3×10⁸ |
| On-chip, decoupled **+ pruned** | 3.8×10⁵ | **5.9×10⁷** | 2.4×10⁸ |

Headline: 9.4×10¹⁸ → 1.5×10⁸ **+** 5.9×10⁷ ≈ 2.1×10⁸, "a reduction factor of ten billion" (O(10¹⁰)).

**The `+` is the entire mechanism.** Because the two subspaces are optimised *sequentially* — off-chip first, then the on-chip space constructed from the off-chip optimum — their costs **add** instead of multiplying. That is the whole decoupling argument in one symbol, and it is the single most portable idea in the mapper literature. Split rationale: "off-chip data movement between DRAM and accelerator is 2-3 orders of magnitude more compared to the on-chip data movement".

Marvel uses **two different cost models**: the classical distinct-block locality model for the off-chip subspace (solved semi-analytically by minimising a data-movement function and ordering loops by its *partial derivatives* — not enumerated), and MAESTRO's cost model for the on-chip subspace (which *is* enumerated). Pruning: PE-utilisation bound (0.1), exact-factor tile sizes (no prologues/epilogues), L1 feasibility. Marvel is honest that pruning is lossy — its figures colour-code strategies that "preserve optimal mappings" green vs those that "may prune optimal" red. **No public code**; paper-only.

*Two data-hygiene flags:* the conclusion sentence prints "5.9 × 10⁸" in both arXiv and TACO, but the table says 5.9×10⁷ and only 10⁷ reproduces the stated ≈2.1×10⁸ — **cite 5.9×10⁷**, the conclusion has a typo that survived camera-ready. Also 3.6×10¹¹ × 2.4×10¹⁰ = 8.6×10²¹ ≠ 9.4×10¹⁸, so the decoupled rows don't multiply back to the "original" row; don't build a claim on that product.

**ConfuciuX** (Kao, Jeong, Krishna, MICRO 2020; arXiv:2009.02010) — confirmed on all counts: it searches **hardware resource assignment** (`(PEs, Buffers)` per layer) *given* a dataflow, i.e. GAMMA turned inside out; MAESTRO is the RL environment (Fig. 3, "HW Perf. Estimator (MAESTRO)"); REINFORCE for global search + GA for local fine-tuning, 4.7–24× faster convergence than BO/GA/annealing. Design space **O(10⁷²)** for 128 PEs / 128 buffers / 52-layer MobileNetV2. Invalid handling is a *scaled* penalty, deliberately not a constant: "we accumulate all the rewards experiences in this episode, and use negative of the accumulated value as a penalty".

**The rest of the `maestro-project` org:**

| Repo | What it searches |
|---|---|
| `maestro` | nothing — the cost model itself |
| `gamma` | mapping (tiles/order/parallel dim/clustering), HW fixed |
| `digamma` | a stub — "the implementation is integrated in GAMMA"; the `--num_pe -1 --area_budget` mode that unlocks `mutate_pe`, searching HW **and** mapping jointly in one genome under an area budget |
| `gamma-timeloop` | same GA, **Timeloop swapped in** for MAESTRO |
| `confuciux` | HW resources per layer, RL+GA |
| `magma` | **multi-tenant job scheduling** (HPCA 2022) — jobs from several DNNs onto multiple sub-accelerators. Job→core assignment, *not* fusion |
| `frame` | roofline analytical model (incl. Transformer); a cost model, not a searcher |
| `AIrchitect-v2` | DATE 2025 — *predicts* the answer in constant time instead of searching; trained on data generated by running ConfuciuX; per-layer inputs |

### 5.2 The division of labour, and why it matters to us

**[inference]** The reusable lesson, independent of which searcher you like: a *fast, deterministic, analytical* evaluator is the enabling asset — MAESTRO does no simulation, just tile arithmetic, which is what makes GA / RL / exhaustive sweep viable on top of it at all. Every tool above is a thin search wrapper around a subprocess call to that evaluator. We currently have neither the evaluator nor the search. **The ordering the whole ecosystem implies is: build the evaluator first.** It is independently valuable as an instrument (§2.3) even if we never attach a search to it.

**[inference] What this layering says for us.** The division of labour is the reusable part, independent of the specific searchers: a *fast, deterministic, analytical* evaluator (MAESTRO: no simulation, pure tile arithmetic, ~ms per layer) is what makes any search on top viable at all — GA, RL, or plain sweep. Our situation is the mirror image: we have neither. Building the evaluator first is the ordering the whole ecosystem implies, and it is independently useful as an instrument even if we never attach a search to it.

---

## 6. Multi-operator — the decisive question

**MAESTRO is single-layer. Plainly. There is no fusion, no cross-layer pipelining, and no on-chip residency across an operator boundary.**

Evidence, strongest first:

1. **The DSL has no construct for it [code].** The complete token list in `DFSL_syntax_tokens.hpp` (reproduced in §1.2) contains `Network`, `Layer`, `Type`, `Stride`, `Dimensions`, `Dataflow`, `TemporalMap`, `SpatialMap`, `Cluster`, `Sz`, `L`, `P` — and nothing expressing a dependence, a producer/consumer edge, a fusion group, or a pipeline stage. `RESIDUAL_IDENTITY` is a *layer type* (an elementwise add), not a fusion construct. A `Network` is a flat list of `Layer`s, and **each `Layer` carries its own independent `Dataflow` block**.

2. **The driver loops over layers independently [code].** `API_user-interface-v2.hpp`:
   ```cpp
   for(auto layer : *(configuration_->network_)) {
     auto layer_results = AnalyzeCostAllClusters(layer_id, ...);
     ...
     ret->push_back(layer_results);
     layer_id++;
   }
   ```
   Results are then aggregated arithmetically — `model_wise_total_l1_size += layer_wise_total_l1_size`, and the capacity check uses the max over layers. Nothing couples layer *i*'s output placement to layer *i+1*'s input placement.

3. **The docs list only per-layer operator types**, with no fusion/pipelining entry: CONV2D, DWCONV, TRCONV, FC, GEMM (<https://maestro.ece.gatech.edu/docs/build/html/layer_supported.html>).

4. **The repo's own transformer example proves it [code].** `data/mapping/Transformer_Layers.m` writes a transformer as a flat sequence of independent layers — `MH_FC_DimReduce_VKQ_0`, `SD_MatMul_QK_00`, `SD_MatMul_V_00`, `MH_FC_DimRecast_0`, … — each with its own full `Dataflow` block:
   ```
   Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
       Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN
       Stride { X: 1, Y: 1 }
       Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
       Dataflow {
           SpatialMap(2,1) K;
           TemporalMap(1,1) C;
           TemporalMap(Sz(R),Sz(R)) R;
           ...
       }
   }
   ```
   Q·K^T and the subsequent ·V are *separate, independently mapped layers*. The attention chain's intermediate is not modelled as staying on chip.

   The companion file `data/mapping/Transformer_Complete.m` settles it: **5,995 lines, 288 independent `Layer` blocks**, in which even the *per-head* attention matmuls are separate layers (`SD_MatMul_QK_00`, `SD_MatMul_V_00`, `SD_MatMul_QK_01`, `SD_MatMul_V_01`, … per encoder block, then `MH_FC_DimRecast_0`, `FF_A_0`, `FF_B_0`, and the whole pattern repeated for block 1, 2, …). I grepped it for any fusion / dependency / producer construct: **none**. This is precisely the workload we care about, and MAESTRO models it as a bag of 288 unrelated GEMMs whose costs are summed.

5. **§4.4 [paper]** scopes generality as "all the operations represented as the loop nest with **two input tensors and one output tensor**" — a single-operator einsum.

The only concession is §4.1's note that the analysis omits "edge case handling, **multiple layers**, and multiple cluster levels" from the *paper's* pseudocode for space — i.e. multiple layers are handled by *repetition*, which is what the code shows.

### 6.1 The authors say so themselves

This is not only my inference from the code. The same group states it in print, about their whole tool class:

> "There are several popular open-sourced DNN accelerator modeling frameworks [18, 50, 59, 64, 74, 101]. However, **none of them offer support for cross-layer performance (and reuse) modeling, assuming layer-by-layer execution.**"
> — FLAT, ASPLOS 2023, §6.1 (<https://arxiv.org/pdf/2107.06419>), where reference [50] is MAESTRO.

And **Marvel's notation explicitly excludes fusion** — it is not an omission but a documented non-goal: "there can be the implementation of certain operators such as **fused convolutions**, where each PE requires executing the non-uniform computation. Hence, **such operators are discarded and are non-conformable to the MDC notation**" (conformability rule R1), restated in its limitations section: the representation "**cannot support** ... fusion of multiple convolution layers". Marvel sums per-operator runtimes to report a model number.

Every mapper in the org inherits the limit. GAMMA's driver is a flat per-layer loop that fully resets state each iteration (`for dimension in model_defs: env.reset_dimension(...)`), and its `write_maestro` hardcodes `dimensions = [self.dimension]` — **exactly one layer per MAESTRO file** [code].

**One near miss worth knowing about:** ConfuciuX has a **Layer Pipelined (LP)** mode where "the entire model is mapped and run in a pipelined manner, with the compute and memory partitioned across all layers" (Fig. 2) — the agent emits `2N` actions for an `N`-layer model under one global area/power budget, so layers *are* coupled through the resource constraint. But this is **resource partitioning for a pipeline, not a fused mapping**: each layer is still costed by a separate MAESTRO call, and MAESTRO still cannot model one stage's output staying on-chip for the next. Do not mistake it for fusion. **[inference]** It is nonetheless the closest structural analogue to our "budget is per-column across the whole segment" constraint — a global resource budget split across pipeline stages — and the one piece of the MAESTRO ecosystem whose *problem shape* matches ours.

### 6.2 The GT fusion work exists — but beside MAESTRO, not on it

Two tools, and the first is close to our actual problem:

- **FLAT** (Kao, Subramanian, Agrawal, Yazdanbakhsh, Krishna, **ASPLOS 2023**, <https://arxiv.org/abs/2107.06419>) — fuses **tensor-tensor (many-to-many)** operators, specifically the *logit → softmax → attend* chain, turning attention's memory footprint from quadratic to linear. It has a real **map-space exploration across fused operators** (Fig. 8): tile sizes × compute order (W/I/O-stationary) × **fusion granularity** `{B, M, H, R}`, by exhaustive search. This is the closest published thing to "map a chain where one stage's output feeds the next on-chip", and it is on attention specifically.
- **DNNFuser** (Kao, Huang, Krishna, <https://arxiv.org/abs/2201.11218>) — a GPT-style Transformer that infers a whole **layer-fusion strategy** (per-layer micro-batching) in one forward pass, 66–127× faster than search. Fusion mapspace sizes: **64¹⁸ = O(10³²)** for ResNet18, **O(10⁹⁰)** for ResNet50/MobileNetV2/MnasNet. Its framing confirms the gap: "mappers for inter-layer map-space (aka layer-fusion map-space) have been rarely discussed" — "no prior mappers" targeted it.

**The decisive detail: both had to write their own cost models, because MAESTRO cannot express fusion.** DNNFuser §4.5: "We built an analytical cost model to model the effect of layer fusion ... The built cost model is **validated against** MAESTRO." FLAT §6.1: "we compared the simulation results from our framework **under single-layer modeling** to MAESTRO. The performance metrics are within 1% difference" — note the qualifier; agreement is claimed only in the degenerate single-layer case. Neither tool is in the `maestro-project` org.

**[inference] Consequence for us:** our hard problem — distinct pipeline stages on distinct cores with L1→L1 hand-offs, and residency that holds only within a segment — is entirely outside MAESTRO's model. MAESTRO's spatial axis is "all PEs cooperate on one operator"; our pipelining axis is "different cores run different operators". You could use a MAESTRO-style model to cost *one stage* of our pipeline, but it will not tell you anything about stage balance, hand-off cost, or segment-level residency, because it has no term for any of them. **If we want the multi-stage part, FLAT is the paper to read next, not MAESTRO** — and the ecosystem's own precedent (two separate teams building fusion cost models *beside* MAESTRO and calibrating single-layer against it) is the realistic template: reuse MAESTRO's per-stage arithmetic, add our own inter-stage terms.

---

## 7. What transfers, and what doesn't

### Transfers well

1. **Required-resource reporting, diffed against provision, as the balance instrument.** The single most directly usable thing, and MAESTRO ships it in ~30 lines. Compute `required_BW = bytes_moved_per_iteration / compute_time_of_that_iteration` **per hierarchy level** (peak and average) and `required_buffer = 2 × worst-case per-iteration traffic`; then print demand *and* budget together and set a `[PASS]` flag: `[WARNING:BW] Per-layer NoC BW requirement [X] is larger than the given NoC BW [Y]`. No simulator, no measurement, no hardware run. For us the levels are DRAM→shim, shim→memtile (**per column**), memtile→core. That single line, instantiated per column, is the instrument we do not have.

2. **The `max(ingress, compute, egress)` steady-state model with explicit init/steady/edge cases, plus bottleneck attribution.** MAESTRO records *which* of the three was the max for each iteration case. That is a per-stage roofline, and it is what tells you whether a stage is DMA-bound or compute-bound — the precondition for balancing a pipeline. The init/steady/edge decomposition (`ExtractDataIterationCases`, weighted by `num_case_occurrences`) is a clean way to handle our prologue/epilogue and ragged last tiles without simulating.

3. **Making a class of illegality unwriteable in the notation.** MAESTRO cannot express two parallel dimensions at one level; you must introduce a `Cluster`, and cluster sizes must exactly divide the PE count (`IllegalClusterConstruction`). **[inference]** The analogue for us: a mapping notation in which *the per-column channel assignment is part of the syntax* — so that a segment declares its per-column feeds and the count is a well-formedness property checked at construction, rather than a downstream consequence discovered by inspecting generated BDs. This is the structural answer to "illegal points look legal".

4. **`SpatialMap(size, offset)` with `offset < size` for halo/overlap.** A compact, exact way to write our overlapping `kv_seq_tile` / sliding windows, including the derived buffer requirement, without hand-deriving index math.

5. **Directive order == stationarity.** Reordering two maps flips which tensor is stationary (Fig. 5A vs 5B). For our `tile_m/n/k` and `emb_tile` choices this is a very cheap enumeration — the space of orders at one level is small, and each order has a directly computable reuse consequence.

6. **The coupling-table reuse rule.** "A tensor is reused across a mapped dimension iff that dimension is not one of its coupled dimensions." For transformer GEMMs this is a two-line test that immediately tells you which operand can be broadcast down a column and which must be reduced across rows.

7. **Marvel's sequential decoupling — the best idea in the mapper layer.** Split the space by cost asymmetry, optimise the expensive half first, then construct the cheap half's space *from* that optimum, so the sizes **add instead of multiply**: 9.4×10¹⁸ → 1.5×10⁸ + 5.9×10⁷ ≈ 2.1×10⁸, a factor of 10¹⁰. **[inference]** Our natural cut is the same shape and for the same reason (off-chip movement dominates): fix the DRAM↔memtile traffic pattern first — which is where the per-column MM2S budget actually binds — then explore core-level tiling within that. It also gives our per-column budget a natural home: it is a constraint on the *outer* subproblem, decided once, rather than something re-checked at every inner point.

8. **GAMMA's three-layer treatment of infeasibility.** Prevent what the encoding can prevent (tile sizes restricted to divisors — no ragged tiles); **repair** what is mechanically fixable (clamp inner tiles to the enclosing cluster); **penalise** only the residue. Note the ordering: rejection is the last resort, not the first. If we ever put a search on top, this is the pattern — and the `Growing`/`Aging` operators show how to make the *number of hierarchy levels* itself searchable with a variable-length genome.

### Does not transfer

1. **Single-operator scope — the big one.** No fusion, no inter-stage pipelining, no cross-operator residency (§6). Our central axis has no representation. Anything we build in this style covers one stage at a time; stage balancing is *our* problem to model.

2. **Channel *count* vs byte-rate.** MAESTRO's NoC is a byte-rate pipe: `hops×latency + ceil(traffic/BW) − 1`, with a bandwidth budget it does check. But there is no concept of a **bounded number of concurrent physical flows**. Three logical feeds into a column appear as more bytes over the same pipe, not as a third channel. So a straight port would catch a per-column *byte-rate* overrun and **still miss our actual 2-MM2S violation**. The fix is small but must be deliberate: carry a *cardinality* resource — distinct concurrent MM2S flows per column, budgeted across the whole segment, since our budget is per-column-per-segment rather than per-operator — alongside the byte-rate one, and give it its own `[WARNING]` line. That dimension is genuinely absent upstream; do not expect the design to hand it to us.

3. **Uniform-cluster assumption.** "only supports uniform clusters", cannot model "irregular PE distributions across rows or columns" (hw_supported docs). Fine for one operator on 8×4; **wrong** the moment we assign different columns to different pipeline stages, which is the thing we most want to reason about.

4. **The bus/crossbar accuracy caveat.** The paper states the NoC model "is precise when the NoC is a bus or a crossbar" and otherwise asks the user to hand-pick a bisection bandwidth and average hop count (§4.2). Our shim/memtile/stream-switch fabric with fixed per-column channels is neither. Treat any absolute latency from a model of this shape as unvalidated; the *ratios* and the *requirement* numbers are the useful part.

5. **No contention or queueing model.** Delay is computed independently per level and per iteration case, then max'ed. Two stages competing for the same memtile port is not representable.

6. **The energy model** (activity counts × Cacti at 28 nm) is irrelevant to us; the paper itself notes it is swappable (§4.3).

7. **Validation scope.** 3.9% mean error was measured on MAERI and Eyeriss running VGG16/AlexNet — dense CNNs on PE-array-with-NoC accelerators. No published validation on a spatial NPU of our kind, and none on transformers.

---

## Comparable summary

- **Data-space representation.** Tensors are declared by their coupled dimensions (CONV: `N,G,K,C,R,S,Y,X,Y',X'`; GEMM: `M,N,K`); a *dimension of a tensor* — not a loop variable — is the object that gets mapped. Dimension coupling per tensor is user-specifiable, which is what makes the reuse rule generic.

- **Mapping-space representation.** An ordered list of `SpatialMap(size,offset) α` / `TemporalMap(size,offset) α` directives, segmented by `Cluster(size, P|L)` into hierarchy levels — above a `Cluster` you map across clusters, below it you map within one. Order encodes stationarity; `size` encodes tiling; `offset<size` encodes halo; exactly one `SpatialMap` per level encodes the parallel axis.

- **Legality model.** Three-tier. **Fatal**: structural well-formedness, checked at analysis time (exactly one `SpatialMap` per cluster level, cluster sizes must exactly divide `num_pes`, no edge on a spatial map). **Warning + failed `[PASS]`**: capacity (`[WARNING:Buffer]`) and rate (`[WARNING:BW]`), each printing computed demand alongside provisioned budget — the `NotEnoughL1Buffer`/`NotEnoughL2Buffer` *aborts* are commented out in the released code, so these never halt a run. **Absorbed into cost**: the same rate shortfall also shows up as extra runtime. Nothing is scored as illegal that the notation could have prevented, and nothing that is a resource inequality is left unreported.

- **Search strategy.** None in MAESTRO itself — it is a cost model, and the paper lists a "dataflow auto-tuner" as future work (§7). The shipped DSE sweeps *hardware* parameters with area/power pruning — that is the 480M-explored / 2.5M-valid figure, where "valid" means within area+power, **not** a legal mapping. Mapping search lives in later tools, each a thin wrapper shelling out to the MAESTRO binary once per candidate: GAMMA (variable-length GA; mapspace O(10²⁴)/layer), Marvel (sequential off-chip/on-chip decoupling, 9.4×10¹⁸ → ~2.1×10⁸), ConfuciuX (REINFORCE+GA over *hardware* resources, O(10⁷²)), DiGamma (HW+mapping jointly).

- **Cost model.** Analytical, per cluster level, recursive up the hierarchy ("the outstanding delay of a cluster level becomes the computation delay of the next level above"). Per iteration case (init/steady/edge, occurrence-weighted): runtime `= max(ingress, egress, compute)` under double buffering; buffer requirement `= 2 × worst-case per-iteration traffic`; NoC delay `= hops×latency_per_hop + ceil(traffic/BW) − 1`; required bandwidth `= traffic / compute_delay`. Validated within 3.9% mean error vs MAERI RTL and Eyeriss's reported runtimes on VGG16/AlexNet.

- **Multi-op support.** **None.** A `Network` is a flat list of `Layer`s, each with its own independent `Dataflow`; the driver analyses each in isolation and aggregates arithmetically. No dependence/fusion/pipeline-stage token in the DSL. `Transformer_Complete.m` is 288 independent `Layer` blocks. The authors say it themselves — FLAT §6.1: of the open-source modelling frameworks including MAESTRO, "none of them offer support for cross-layer performance (and reuse) modeling, assuming layer-by-layer execution" — and Marvel's notation *explicitly discards* fused operators. GT's own fusion tools (FLAT, ASPLOS 2023, which fuses the attention chain; DNNFuser) had to **write separate cost models**, validating against MAESTRO only in the single-layer case.

- **Single most transferable idea.** Compute the resource **requirement** per hierarchy level — bandwidth as `traffic / compute_delay`, buffer as `2 × peak per-iteration traffic` — and **report it next to the provisioned budget with a PASS flag**, exactly as MAESTRO's `[WARNING:BW] Per-layer NoC BW requirement [X] is larger than the given NoC BW [Y]` does, with per-iteration-case bottleneck attribution from `max(ingress, compute, egress)`. It is pure arithmetic over tiling parameters we already have, it needs no hardware run, and it turns our silently-degrading per-column DMA budget into a number. Runner-up: make the per-column feed assignment *syntactic*, so the budget is a well-formedness property at construction the way `Cluster` sizes must divide `num_pes` — illegality the notation can prevent, prevented.

- **Single biggest mismatch with our target.** MAESTRO's entire model is *one operator spread uniformly over a homogeneous PE array*; ours is *distinct pipeline stages on distinct cores with L1→L1 hand-offs and segment-scoped residency*. There is no representation of a producer/consumer edge, of heterogeneous column assignment, or of a channel-*count* resource — so neither the cost model nor the notation can express the axis our hard problem actually lives on. (Sharper for near-term use: its NoC is a byte-rate pipe with no notion of a bounded number of concurrent flows, so it would flag a per-column byte-rate overrun but still miss a 2-MM2S-per-column violation.)

