# SCALE-Sim — research notes for the mapping-tool survey

Target of comparison: AMD NPU2 / AIE2P spatial array, 8 columns × 4 rows = 32 cores, shim
tiles (DRAM interface), memtiles (L2). Our mapping axes: tiling, pipelining (distinct
stages on distinct cores, L1→L1), parallelizing (replicating the chain across a data
dimension). Hard constraint: two shim MM2S channels per column, budgeted per-column across
the whole segment; overflow silently packet-multiplexes.

**Headline.** SCALE-Sim is not a mapping tool. It is a *single-systolic-array performance
model* whose entire design space is `{dataflow ∈ {os, ws, is}} × {array rows × cols} ×
{3 SRAM sizes} × {DRAM bandwidth}` — nine-ish scalars in an INI file. There is no search,
no mapping IR, no legality checker, and no notion of multiple cores in the released code.
What it *does* have, and what nobody else in this survey has in the same form, is a
cycle-resolved SRAM/DRAM address trace and a bandwidth-driven stall model that answers
"at B words/cycle, how many cycles does this array spend stalled?" — and its inverse,
"what B would make this stall-free?". That inverse mode is the part worth stealing.

---

## 0. Provenance — which artifact is which

This matters because the survey question says "v2" and the repo named `scale-sim-v2` now
serves v3 code.

| Artifact | What it is |
|---|---|
| Samajdar et al., *SCALE-Sim: Systolic CNN Accelerator Simulator*, arXiv **1811.02883** (v1 Oct 2018, v2 Feb 2019) | The original tool paper. [abs](https://arxiv.org/abs/1811.02883) |
| Samajdar, Joseph, Zhu, Whatmough, Mattina, Krishna, *A Systematic Methodology for Characterizing Scalability of DNN Accelerators using SCALE-Sim*, **ISPASS 2020**, pp. 58–68 | The scale-up/scale-out study. [author PDF](https://bpb-us-e1.wpmucdn.com/sites.gatech.edu/dist/c/332/files/2020/03/scalesim_ispass2020.pdf) · [IEEE 9238602](https://ieeexplore.ieee.org/document/9238602) |
| [`ARM-software/SCALE-Sim`](https://github.com/ARM-software/SCALE-Sim) | v1 code. README line 1: "THIS PROJECT IS SUPSERSEDED BY SCALE-SIM-V2". Last functional commit 2019-02-26. |
| [`scalesim-project/scale-sim-v2`](https://github.com/scalesim-project/scale-sim-v2) | **Redirects to `scalesim-project/SCALE-Sim`.** `main` today (`9f98c43`, 2025-12-17) is *v3* code. Tag `v2.0.2` (2024-01-17) is the last true v2. |
| Raj et al., *SCALE-Sim v3*, **ISPASS 2025**, arXiv **2504.15377** | Adds sparsity, Ramulator, Accelergy, SRAM layout, "multi-core". [pdf](https://arxiv.org/pdf/2504.15377) |

Everything below cites either the ISPASS'20 paper by section, or a repo path with the ref
I read it at. I cloned both repos and ran the tool; results marked **[ran it]** are mine.

---

## 1. Data space — the topology CSV

Two formats, selected by a CLI switch. Both are flat, one line per layer, no graph.

### 1a. Conv format (default)

Header verbatim from `topologies/conv_nets/alexnet.csv` @ `main`:

```
Layer name, IFMAP Height, IFMAP Width, Filter Height, Filter Width, Channels, Num Filter, Strides,
Conv1     ,224         ,224        ,11           ,11          ,3       ,96        ,4      ,
Conv2     ,27         ,27        ,5            ,5           ,96      ,256       ,1      ,
```

Eight fields **and a mandatory trailing comma** — the parser does
`elems = row.split(',')[:-1]`, so a row without the trailing comma loses its last field
(`scalesim/topology_utils.py:139`). Matches Table II of the ISPASS'20 paper ("SCALE-Sim
Topology file description").

Some shipped files (`topologies/llama/llama3b.csv`) add a ninth `Batch Size` column. It is
*not* a batch dimension in the model — at `main` that column is consumed by the sparsity-ratio
parser (`topology_utils.py:145-150`); at `v2.0.2` it was simply carried and ignored.
Treat batch as "not expressible" and fold it into M yourself.

What is **not** expressible: padding, dilation, groups, bias, activation functions,
elementwise ops, softmax, layernorm, any non-GEMM op, dtype/precision, and — critically —
**any edge between layers**. There is no producer/consumer relation anywhere in the format.
Depthwise conv is the one special case, triggered by the substring `DP` in the layer name,
which expands to `Channels` separate single-channel layers (`topology_utils.py:153-157`).

### 1b. GEMM / mnk format (`-i gemm`)

Header verbatim from `topologies/GEMM_mnk/gpt2.csv` @ `main`:

```
Layer,M,N,K,
QKT,1024,1024,64,
QKTV,1024,64,1024,
Linear1,1024,4800,1600,
Linear2,1024,1600,1600,
PW-FF-L1,1024,3072,1600,
PW-FF-L2,1024,1600,3072,
```

So: **yes, it expresses a transformer workload directly**, and the repo ships GPT-2, ViT-S/L,
GNMT, NCF and a partial Transformer as mnk files (`topologies/GEMM_mnk/`). Documented at
[readthedocs topology.html](https://scale-sim-project.readthedocs.io/en/latest/topology.html).

The mnk format is pure sugar. It is rewritten to the conv format at load
(`scalesim/topology_utils.py:110` @ `main`):

```python
# Entries: layer name, Ifmap h, ifmap w, filter h, filter w, num_ch, num_filt, stride h, stride w, ...
entries = [layer_name, m, k, 1, k, 1, n, 1, 1, sparsity_ratio[0], sparsity_ratio[1]]
```

i.e. `M×K` GEMM becomes an IFMAP of height M, width K with a 1×K filter, 1 channel,
N filters, stride 1. Everything downstream is the conv path. **Inference:** this means a
GEMM is always im2col'd into a `[M × K]` operand matrix and a `[K × N]` filter matrix and
nothing about GEMM-specific blocking is available to the tool that isn't available to a conv.

Caveat found by running it: at tag `v2.0.2` the shipped `topologies/GEMM_mnk/gpt2.csv` has
**no** trailing comma and crashes the v2.0.2 parser (`AssertionError: There should be at
least 4 entries per row`). **[ran it]** It was fixed on `main`. Don't trust the shipped
workload files without opening them.

---

## 2. Mapping space — this is the narrow one, and it is narrower than "narrow"

**There is no mapping search. There is no mapping *representation*.** The mapping is a pure
function of (dataflow preset, array dims, layer dims), computed in four lines.

### 2a. The whole space

Every knob, read off `scalesim/scale_config.py:109-137` and a shipped config
(`configs/scale.cfg` @ `v2.0.2`):

```ini
[general]
run_name = scale_example_run_32x32_os

[architecture_presets]
ArrayHeight:    32           # R
ArrayWidth:     32           # C
IfmapSramSzkB:   64
FilterSramSzkB:  64
OfmapSramSzkB:   64
IfmapOffset:    0            # address-space base, cosmetic
FilterOffset:   10000000
OfmapOffset:    20000000
Bandwidth : 10               # words/cycle, DRAM<->SRAM
Dataflow : os                # 'os' | 'ws' | 'is'   <- the entire dataflow space
MemoryBanks:   1

[run_presets]
InterfaceBandwidth: CALC     # CALC | USER
```

v3 adds `[layout]` (bank count/ports/custom layout), `[sparsity]`, `TimeLinearModel`,
`UseRamulatorTrace`. None of them widen the *mapping* space; they refine the memory model.

`self.valid_df_list = ['os', 'ws', 'is']` (`scale_config.py:50`). That's it. The ISPASS'20
paper is explicit that this is deliberate: "Although many different dataflows exist for
spatial arrays, we only consider *true systolic* dataflows that only use local
communication" (§II-A, Dataflow).

### 2b. What the dataflow actually determines

The dataflow picks which of (M, N, K) lands on rows, columns, and time. ISPASS'20 Table III
gives it in conv terms; combining that with the mnk→conv rewrite above, and verified against
`systolic_compute_{os,ws,is}.py`:

| Dataflow | S_R (array rows) | S_C (array cols) | T (temporal) | source |
|---|---|---|---|---|
| Output stationary | M | N | K | `systolic_compute_os.py:85-87` |
| Weight stationary | K | N | M | `systolic_compute_ws.py:102-103` |
| Input stationary | K | M | N | `systolic_compute_is.py:86-87` |

*(Note: Table II of the v3 paper prints WS as (K, M, N) and IS as (K, N, M) — the Sc/T
columns look swapped relative to the code. The code above is authoritative; I read it.)*

### 2c. The "tiling" — one line, no choice

```python
self.row_fold = math.ceil(self.Sr / self.arr_row)
self.col_fold = math.ceil(self.Sc / self.arr_col)
```
(`systolic_compute_os.py:91-92`, identically in `ws`/`is`.)

That is the complete tiling model. Fold order is fixed (column-fold outer, row-fold inner,
contiguous slices, zero-padded at the boundary — `create_ifmap_prefetch_mat`, `os:116-136`).
There is no choice of tile size independent of the array size, no loop permutation, no
choice of which operand to hold resident beyond the three presets, no L2/scratchpad tiling
distinct from the array fold. The ISPASS'20 paper names this: "we term this practice as
*folding*... Folds can be generated by slicing the compute along the S_R and S_C
dimensions" (§III-B2), giving eq. (2): `F_R = ⌈S_R/R⌉, F_C = ⌈S_C/C⌉`.

### 2d. Is there a search?

**In the released code: no.** `grep -rniE "design.space|exhaustive|sweep|search" --include=*.py`
over `main` returns exactly one hit, in `setup.py`'s PyPI classifier list ("Intended
Audience :: Science/Research"). There is no DSE driver, no cost-model optimizer, no
autotuner. v1 had a vestigial one: `configs/*.cfg` were parsed as `min,max` ranges and
`scale.py` has a `run_sweep()` stub, but `__init__` is called with `sweep=False` at both
call sites (`v1/scale.py:197,204`) and the run path uses only the `_min` values.

**In the ISPASS'20 paper: yes, but not in the simulator and not in the repo.** §III builds a
*separate analytical model*, deliberately weaker than the simulator: "*Unlike SCALE-Sim, the
analytical model does not consider cycle by cycle accesses and bandwidth demands due to
limited memory sizes.* Instead, it captures the first-order execution time, and thus helps
prune the search space" (§III intro). That model is eq. (4):

```
τ_scaleup = (2R + C + T − 2) · ⌈S_R/R⌉ · ⌈S_C/C⌉
```

and §IV-B does an exhaustive enumeration over candidates `a_k = (S'_C, S'_R, R, C)`,
selecting `A = argmin_{a_k} Σ_{w_l} T_r(w_l, a_k)`, justified as "As the number of candidates
is limited, exhaustive search is feasible to find the optima."

**The analytical model and the search are not in the open-source repo.** I grepped `main` for
it (`2 ?\* ?R`, `scaleup`, `analytical`) — zero hits. So the paper's "search" is a spreadsheet-grade
closed form the authors ran offline; the tool you can download simulates one point.

**Verdict for our purposes: the mapping space is `{3 dataflows} × {R, C} × {3 buffer sizes}
× {BW}`, exhaustively enumerable by a shell loop, with no legality notion at all.** Any
(R, C) is "legal"; the model just folds. That's a clean negative result — SCALE-Sim is the
floor of the mapping-expressiveness axis in this survey.

---

## 3. What it models that the others don't — the memory/stall model

This is the real content, and it is better than the mapping space would suggest.

### 3a. Architecture: analytical compute, replayed through a memory system

ISPASS'20 §II-C is honest about the trick: "the simulator assumes that the accelerator is
always compute bound and the PEs are always used to the maximum possible utilization — as
dictated by the dataflow... SCALE-Sim generates cycle accurate read addresses for elements
required to be fed on the top and left edges of the array *such that the PE array never
stalls*."

Concretely (`single_layer_sim.py:186-295`):
1. Build **operand address matrices** — `ifmap [ofmap_px × window]`, `filter [window × num_filt]`,
   `ofmap [ofmap_px × num_filt]`, holding *addresses*, not values (`operand_matrix.py:47-49`).
2. Fold them per §2c into **demand matrices**: one row per array cycle, one column per array
   row (ifmap) or column (filter/ofmap). `-1` means "no request this cycle".
3. Replay the demand matrices through a **double-buffered scratchpad** and let the memory
   system insert stalls.

Step 3 is where the interesting behaviour lives.

### 3b. The stall model — global lockstep on max-of-three

`memory/double_buffered_scratchpad_mem.py:254-280`:

```python
for i in range(ofmap_lines):
    cycle_arr = np.zeros((1,1)) + i + self.stall_cycles
    ifmap_cycle_out  = self.ifmap_buf.service_reads(ifmap_demand_line,  cycle_arr)
    ifmap_stalls  = ifmap_cycle_out[0]  - cycle_arr[0] - ifmap_hit_latency
    filter_cycle_out = self.filter_buf.service_reads(filter_demand_line, cycle_arr)
    filter_stalls = filter_cycle_out[0] - cycle_arr[0] - filter_hit_latency
    ofmap_cycle_out  = self.ofmap_buf.service_writes(ofmap_demand_line, cycle_arr)
    ofmap_stalls  = ofmap_cycle_out[0]  - cycle_arr[0]
    self.stall_cycles += int(max(ifmap_stalls[0], filter_stalls[0], ofmap_stalls[0]))
```

Read that carefully, because it is the model's whole personality:

- Three **independent** double-buffered SRAMs (ifmap / filter / ofmap), each with its own
  size and its own backing bandwidth. They do **not** contend with each other for a shared
  port. There is no arbiter.
- Stalls compose as **max**, not sum, and they stall the **entire array** globally — a
  filter miss freezes the ifmap stream too. The old code path spells it out:
  "*The entire array stops when there is a stall*" (`:400`).
- The stall is a scalar added to a monotonically advancing cycle counter. There is no
  queueing, no reordering, no per-channel backpressure, no partial progress.

A miss is a *capacity* miss: `service_reads` checks the active buffer, and on a miss calls
`new_prefetch()` in a `while` loop until the address is resident, charging
`last_prefetch_cycle − (cycle + offset)` if the prefetch hadn't landed
(`memory/read_buffer.py:346-370`). Prefetch cost is `⌈prefetch_buf_size / backing_buf_bw⌉`
cycles (`read_buffer.py:476`), i.e. **bandwidth is modelled purely as words/cycle of issue
rate.** Buffers split 50/50 active/prefetch by default (`single_layer_sim.py:239`).

### 3c. How faithful is the DRAM model? In v2, not very — and that's fine

At `v2.0.2`, `memory/read_port.py` is, in full:

```python
# Dummy memory like interface to service the requests of the last level memory
class read_port:
    def service_reads(self, incoming_requests_arr_np, incoming_cycles_arr):
        out_cycles_arr = incoming_cycles_arr + self.latency   # self.latency == 1
        return out_cycles_arr
```

So v2's "DRAM" is: fixed 1-cycle latency, and a bandwidth cap enforced by the *shape* of the
prefetch request block. No banks, no rows, no refresh, no read/write turnaround, no queueing.
ISPASS'20 §II-B is upfront: SCALE-Sim "allows for modeling the main memory behavior by
generating accurate read and write bandwidths at the interface, which can then be fed into
a DRAM simulator e.g., DRAM-Sim2." **The trace is the product; the DRAM model is somebody
else's job.** v3 closes this by piping the trace to Ramulator and adding SRAM bank-conflict
modelling (`read_buffer.py:304-334` @ `main`; v3 paper §III and Limitation 3).

### 3d. Two modes, and the second one is the interesting one

- **`InterfaceBandwidth: USER`** — you give `Bandwidth: B`; the tool reports stall cycles.
- **`InterfaceBandwidth: CALC`** — a different buffer class is instantiated
  (`memory/read_buffer_estimate_bw.py`), whose `service_reads` begins:
  `outcycles = incoming_cycles_arr + self.hit_latency  # In estimate mode, operation is stall
  free. Therefore its always a hit` (`:96-97`). It then back-solves the prefetch bandwidth
  that would have been required: `self.prefetch_bandwidth = math.ceil(elems_to_prefetch /
  cycles_needed)` (`:152`) and reports it.

**CALC mode is an inverse solver: "what interface bandwidth does this mapping demand for
stall-free execution?"** That is exactly the shape of question our per-column MM2S budget
poses, and it is the single most transferable idea in the tool.

### 3e. Outputs — exactly what you get

Four summary CSVs plus six per-layer traces
([README](https://github.com/scalesim-project/scale-sim-v2), `simulator.py:173-215`):

- `COMPUTE_REPORT.csv` — `LayerID, Total Cycles, Stall Cycles, Overall Util %, Mapping Efficiency %, Compute Util %`
  (`main` prepends a `Total Cycles (incl. prefetch)` column).
- `BANDWIDTH_REPORT.csv` — avg IFMAP/FILTER/OFMAP bandwidth at **both** SRAM and DRAM, words/cycle.
- `DETAILED_ACCESS_REPORT.csv` — per operand, per level: start cycle, stop cycle, access count.
- `TIME_REPORT.csv` (v3 only) — cycles → µs via a fitted `TPUv4/v5e/v6e` linear model.
- `layerN/{IFMAP,FILTER,OFMAP}_{SRAM,DRAM}_TRACE.csv`.

Trace format, **[ran it]** at `v2.0.2`, 32×32 WS, `Bandwidth: 10`:

```
IFMAP_SRAM_TRACE.csv   1,-1,-1,...            <- cycle, then 32 address columns (one per array row); -1 = idle
FILTER_DRAM_TRACE.csv  -205.0,10000000.0,10000032.0,...   <- cycle, then exactly 10 address columns = Bandwidth
```

Note the DRAM trace has **exactly `Bandwidth` address columns per row**. The bandwidth budget
is literally the width of the trace. Negative cycles are the pre-roll prefetch.

Sample run **[ran it]**, `v2.0.2`, 32×32 WS, 64 kB each buffer, `Bandwidth: 10`, `USER`,
two GEMMs written as mnk:

```
LayerID, Total Cycles, Stall Cycles, Overall Util %, Mapping Efficiency %, Compute Util %,
0, 1775,    0,       57.69, 100.0, 50.59      # M=128 N=128  K=64
1, 3984321, 3756994,  3.29, 100.0, 50.59      # M=128 N=2048 K=512
```

94 % of layer 1 is stall. Also worth reading: layer 1's SRAM IFMAP reads are 4 194 304 =
`M·K · col_fold` (65 536 × 64) — the input matrix is re-read **once per column fold**,
because the tool has no L2 and no reuse across folds. That inflation is visible in the
report, which is genuinely useful; the fact that it can't be avoided by remapping is the
limitation.

**Practical cost.** The demand matrix has one row per array cycle, materialized in NumPy and
walked in a Python loop (`double_buffered_scratchpad_mem.py:254`). The toy layer above
(M=128, N=2048, K=512) already produces ~4 M rows. **[ran it]** The real GPT-2 mnk topology
(six GEMMs, largest `1024×4800×1600`) on a 32×32 array had not finished after >10 minutes of
wall time and I killed it. Budget minutes-to-hours per configuration for transformer-scale
GEMMs — which is itself a reason the ISPASS'20 authors built the separate closed-form model
for their sweeps (§III intro: "running simulation for all possible data points in a large
search space is expensive and sometimes unnecessary").

### 3f. Validation

ISPASS'20 §II-D, Fig. 4: compared against an RTL implementation of a systolic array, OS
dataflow, arrays 4×4 → 90×90, **under full utilization**. "the cycle counts obtained by both
the methods are in good agreement." Note the scope: compute cycles only, one dataflow, no
memory pressure. The stall model is **not** validated against RTL in that paper. v3 §I
reports the memory model changes the answer materially — "when factoring in DRAM stalls,
OS dataflow exhibits 30.1 % lower execution cycles compared to WS", reversing the compute-only
ranking — which is simultaneously an argument for modelling stalls and a warning that
v2's stall numbers were load-bearing and unvalidated.

---

## 4. Scale-out / multi-array — the part closest to our replication axis, and it isn't in the tool

This is the biggest paper-vs-repo gap in SCALE-Sim.

### What the ISPASS'20 paper does

§III-C: `P = P_R × P_C` independent systolic arrays, each `R × C`. The workload is split by
**partitioning the output matrix**:

```
S'_R = ⌈S_R / P_R⌉ ,  S'_C = ⌈S_C / P_C⌉                                  (5)
τ_scaleout = (2R + C + T − 2) · ⌈S'_R/R⌉ · ⌈S'_C/C⌉                       (6)
```

with the load-balance assumption stated outright: "Since the individual partitions execute
in parallel, the total runtime of the scaled-out system is simply the runtime of the slowest
cluster." Every partition gets a full copy of the temporal dimension T — the split is over
the two *spatial* dims only, so **there is no cross-partition reduction and no partial-sum
traffic** in the model. §II claims "SCALE-Sim can model both scale-up (one partition) and
scale-out (multiple partition) instances."

The findings are the interesting export:
- Partitioning is essentially always faster at equal MAC count; for 65 536 MACs the best
  monolithic config is **50×** slower than the best partitioned one (§IV, Fig. 10).
- The cost is **loss of spatial reuse**: splitting the array cuts the row/column broadcast
  distance, so SRAM reads, data replication and DRAM bandwidth all rise (§IV-A, "Cost of
  scaling out").
- Hence a **runtime-vs-bandwidth knee**: runtime falls monotonically with partition count
  while DRAM BW rises, and "the sweet spot lies at the intersection of runtime and bandwidth
  curves" (§IV-A, Fig. 11). At 2^18 MACs the knee sits near ~10 KB/cycle of DRAM bandwidth
  for both a ResNet-50 conv layer and Transformer TF0.
- Energy has its own minimum that moves right (more partitions) as MAC count grows (Fig. 12).

### What the code does

**Nothing.** `grep -rniE "partition|scale.out|p_r|p_c" --include=*.py` over `v1` returns only
false positives (`ifmap_read`, `scale_out` as an output directory name). Over `v2` `main`:
zero. The scale-out numbers in the paper come from the **analytical model**, which was never
released, plus per-partition invocations of the simulator with the SRAM budget divided by
the partition count ("a total of 512KB of SRAM is allocated... This memory is evenly
distributed amongst the partitions in case of scaling out", §IV-A). **Inference:** to
reproduce Fig. 11 you'd shrink `S_R`/`S_C` and the buffer sizes by hand and run the
single-array simulator once per partition. Nothing in the tool does that for you.

The v3 paper confirms this in as many words: "**SCALE-Sim v2 does not have support to model
such chips**, in part because when it was proposed most accelerators just had single systolic
arrays" (Limitation 1). *(Its own Table I row for SCALE-Sim v2 says cores = "many",
partitioning = "Spatial" — that row describes the ISPASS'20 paper, not the v2 code. Two
statements in the same paper, and the code agrees with the prose, not the table.)*

### What v3 claims, and what shipped

v3 §III adds **spatio-temporal partitioning**: partition along T as well as S_R/S_C, giving
two more runtime forms alongside the v2 spatial one (eqs. 2–3):

```
spatial          : (2R + C + T − 2)      · ⌈(S_r/P_r)/R⌉ · ⌈(S_c/P_c)/C⌉
spatio-temporal 1: (2R + C + ⌈T/P_c⌉ − 2)· ⌈(S_r/P_r)/R⌉ · ⌈S_c/C⌉
spatio-temporal 2: (2R + C + ⌈T/P_r⌉ − 2)· ⌈S_r/R⌉      · ⌈(S_c/P_c)/C⌉
```

plus a **hierarchical L2 shared between cores** explicitly to kill the input/weight
duplication that spatial partitioning creates (§III-B, Fig. 4), plus heterogeneous cores
(systolic + SIMD) and non-uniform partitioning. Fig. 3 plots compute-cycles vs memory-footprint
Pareto fronts over 27 GEMMs × array sizes {8,16,32} × cores {16,32,64}.

That is genuinely close to our replication axis with memtiles. **But: the `multi-core/`
directory the v3 README points at (`refer to the multi-core/README.md`) is not present on
`main` (`9f98c43`) and not on the `3.1` branch.** I checked both. So as of this reading the
multi-core feature is documented and published but not in the tree I could clone.

---

## 5. Multi-operator — strictly layer-at-a-time. Plainly.

No fusion, no pipelining, no cross-layer anything. Three independent confirmations:

1. **The data format has no edges.** A topology CSV is a list of independent layer shapes
   (§1). Nothing names a producer.
2. **The driver loop is a `for` over independent objects**, each constructing its own memory
   system from scratch, with results summed (`simulator.py:76-104`). The comment on line 97
   is `# TODO: This is parallelizable` — the layers are independent by construction, so
   buffer state at the end of layer N has no effect on layer N+1. There is no L1→L1 handoff
   to model because there is no state carried between layers.
3. **The paper says so.** §II-E: "SCALE-Sim parses the topology file one line at a time and
   simulates the execution of the layer. This is a natural approach for traditional neural
   networks which are primarily composed of a single path. However, modern DNNs often contain
   'cells' that are composed of multiple convolution layers in parallel. **SCALE-Sim
   serializes the execution of such layers in the same order in which they are listed in the
   topology file.**"

So even *parallel* branches are serialized — never mind fused. The paper's related-work
section flags this as a known gap, citing Tangram for "custom dataflow for inter-layer
pipelining" (§V) without adopting it. There is a `dev-fusion` branch on the remote
(`22f88b17`) — unmerged, undocumented, not in any release; I did not evaluate it.

**Consequence for us: SCALE-Sim cannot represent our pipelining axis at all.** Not "models it
crudely" — the concept has no encoding in the input format.

---

## 6. Paper vs. repo — the honest ledger

| Claim | Where | Status |
|---|---|---|
| Cycle-accurate compute model, RTL-validated | ISPASS'20 §II-D | **True**, for compute cycles, OS dataflow, full utilization. |
| Models scale-up *and* scale-out | ISPASS'20 §II | **Paper only.** No partition code in v1, v2, or v3 `main`. |
| DSE / optimal-config search | ISPASS'20 §IV-B; v1 README "Accelerator Design Space Exploration" | **Paper only.** v1's `run_sweep()` is dead code; v2/v3 have none. |
| Multi-core support | v3 README feature 5; v3 §III | **Documented, `multi-core/` dir absent** from `main` and `3.1`. |
| Fusion / inter-layer pipelining | — | Never claimed. Unmerged `dev-fusion` branch only. |
| Detailed DRAM | v2 README | v2's DRAM is a 1-cycle dummy port; real DRAM (Ramulator) is v3. |
| WS/IS spatio-temporal mapping table | v3 Table II | Sc/T columns appear swapped vs. `systolic_compute_{ws,is}.py`. Trust the code. |

Reproducibility **[ran it]**, all against clones taken 2026-08-12:

- `main` (`9f98c43`): the **shipped** `configs/tpuv4.cfg` fails immediately —
  `configparser.NoSectionError: 'layout'` (`scale_config.py:126` now requires a `[layout]`
  section that the shipped example configs don't have).
- `main` with a `[layout]` section added: GEMM mode crashes in **both** bandwidth modes —
  `TypeError: only 0-dimensional arrays can be converted to Python scalars` at
  `double_buffered_scratchpad_mem.py:307` (CALC) and `read_buffer.py:423` (USER).
- `v2.0.2`: the shipped `topologies/GEMM_mnk/gpt2.csv` crashes the parser (missing trailing
  comma).
- `v2.0.2` with a hand-fixed mnk file: **works**, produces the reports and traces quoted in §3e.

If you want to run SCALE-Sim, use tag `v2.0.2` and write your own topology files.

---

## 7. What transfers to a per-column-DMA-budgeted spatial NPU

### Worth stealing

1. **CALC mode as an inverse solver.** Don't ask "does this mapping fit the budget"; ask
   "what per-column MM2S rate would this mapping need to run stall-free?" and compare against
   2. That converts our silent packet-multiplexing failure into a *number with headroom*,
   computable statically from the mapping, before any hardware run. This is the one idea I'd
   port verbatim. It needs no simulator — just the demand schedule and a division
   (`read_buffer_estimate_bw.py:152` is literally `ceil(elems / cycles)`).
2. **The demand-matrix representation.** `[cycle × port]` of addresses, `-1` for idle. That
   is exactly the shape of a per-column shim-channel occupancy schedule: rows are cycles,
   columns are the two MM2S channels, entries are BD IDs. Building that artefact gives us the
   balance instrument we don't have — max over cycles of concurrent channel demand per column
   *is* the budget check. It is a data structure, not a simulator, and it's cheap.
3. **Bandwidth as trace width.** SCALE-Sim encodes "B words/cycle" as "B address columns per
   trace row". A per-column budget of 2 MM2S is "2 columns". Overflow is then a *shape*
   violation you can assert on, which is precisely the error our hardware declines to raise.
4. **The runtime-vs-bandwidth knee framing** (ISPASS'20 §IV-A, Fig. 11). Their finding —
   replication monotonically improves runtime while monotonically worsening interface
   bandwidth, so the design point is the crossing — transfers directly to our parallelizing
   axis, where replicating a chain across more columns multiplies aggregate shim demand.
   Expect a knee; look for it; report it as a curve, not a point. Their loss-of-spatial-reuse
   explanation (partitioning shortens broadcast paths, so each partition re-fetches) has an
   exact analogue: replicated chains re-fetch the same weights per column unless a memtile
   broadcasts, which is the argument for our L2.
5. **Reporting `Mapping Efficiency %` separately from `Overall Util %`.** Mapping efficiency
   = fraction of the array a fold actually covers (padding loss); overall util folds in
   stalls. Two numbers, two different fixes. In my run both layers had 100 % mapping
   efficiency and 57 % vs 3 % overall util — the split immediately says "this is a memory
   problem, not a tiling problem". Cheap and diagnostic.
6. **v3's shared-L2-to-kill-duplication argument** (v3 §III-B) is the memtile argument, stated
   for a partitioned array. Worth reading if we formalize why memtiles matter for replication.

### Does not transfer

1. **Global lockstep stalls.** `stall_cycles += max(ifmap, filter, ofmap)` freezes the whole
   array. Our 32 cores are independently scheduled with per-core DMA queues and L1→L1
   handoffs; a stalled consumer backpressures its producer locally and the rest of the array
   keeps running. A max-of-operands global scalar would systematically overestimate our
   stalls. Any borrowed stall accounting has to become per-core with backpressure.
2. **No contention model.** SCALE-Sim's three buffers have independent, non-competing backing
   ports. Our whole problem is that *N cores share 2 channels per column* — the arbitration
   SCALE-Sim doesn't have is exactly the thing we need to model. So: borrow the *demand
   accounting*, do not borrow the *service model*.
3. **Systolic store-and-forward assumptions.** The `2R + C + T − 2` skew/drain term is
   neighbour-to-neighbour register forwarding across a mesh. AIE cores are VLIW processors
   with explicit local memory and DMA; there is no array-wide skew fill. The formula's *shape*
   (pipeline fill + steady state + drain, times fold count) survives as an intuition; the
   coefficients don't.
4. **Layer-at-a-time.** Structurally incompatible with our pipelining axis. Nothing to borrow.
5. **im2col-everything.** Forcing GEMMs through a conv operand matrix is fine for a systolic
   array with one blocking scheme, and useless where the interesting decision *is* the
   blocking scheme.
6. **The dataflow preset triple.** {os, ws, is} is a taxonomy of which operand sits still in a
   PE. On AIE the analogous question — which tensor stays resident in L1 across iterations — is
   per-core, per-stage, and can differ between pipeline stages of the same chain. Three global
   presets cannot express that.

### Net

Take the **inverse bandwidth question** and the **`[cycle × port]` demand matrix**; leave the
mapping model, the stall arithmetic, and everything about systolic skew. SCALE-Sim answers
"given a mapping, what does memory cost?" with more resolution than the rest of the survey,
and answers "what mapping?" not at all.

---

## Comparable summary

**Data-space representation.** Flat CSV, one line per layer, no edges: either conv
(`Layer name, IFMAP H, IFMAP W, Filter H, Filter W, Channels, Num Filter, Strides,`) or GEMM
(`Layer, M, N, K,`), the latter rewritten into the former at load
(`topology_utils.py:110`). Transformers are expressible as a list of GEMMs — the repo ships
GPT-2 and ViT — but batch, dtype, non-GEMM ops, and all inter-layer structure are not.

**Mapping-space representation.** There isn't one. The mapping is a derived quantity:
`{dataflow ∈ os|ws|is} × {ArrayHeight, ArrayWidth}` determines `S_R, S_C, T` (Table III) and
then `row_fold = ⌈S_R/R⌉, col_fold = ⌈S_C/C⌉` with fixed order and no other freedom. Nine
INI scalars is the entire configurable space.

**Legality model.** None. Every (R, C, dataflow) combination is accepted; oversized dimensions
simply fold, undersized ones zero-pad (and are charged as `Mapping Efficiency < 100 %`).
Buffers never "overflow" — they just miss and stall. No constraint is ever violated because
no constraint is ever stated.

**Search strategy.** None in the code (`grep` for search/DSE over `main`: one hit, a PyPI
classifier). The ISPASS'20 paper does exhaustive enumeration over `(S'_C, S'_R, R, C)` under a
*separate, weaker, unreleased* closed-form model (eqs. 4 and 6) that explicitly excludes
memory effects — so the search never sees the thing the simulator is good at.

**Cost model.** Analytical compute schedule (RTL-validated for OS at full utilization, §II-D)
replayed through three independent double-buffered SRAMs; stalls charged as
`max(ifmap, filter, ofmap)` freezing the whole array; DRAM is a bandwidth cap on prefetch
issue rate plus a 1-cycle dummy latency in v2 (Ramulator in v3). Outputs cycles, stall cycles,
utilization, mapping efficiency, per-operand per-level access counts and average/max
bandwidths, plus full `[cycle × port]` address traces for all six operand/level pairs. The
`CALC` mode inverts it: assume no stalls, report the bandwidth that assumption requires.

**Multi-op support.** None. Layer-at-a-time by construction — independent simulator objects,
summed cycles (`simulator.py:76-104`), and the paper states parallel branches are *serialized*
in file order (§II-E). No fusion, no inter-layer pipelining, no residency across layers.
An unmerged `dev-fusion` branch exists.

**Single most transferable idea.** `InterfaceBandwidth: CALC` — treat the bandwidth budget as
the *unknown* and back-solve the rate a candidate mapping demands for stall-free execution
(`read_buffer_estimate_bw.py:96-152`). Applied per column, that turns our silent
packet-multiplexing overflow into a computable headroom number, statically, with no hardware
run and no simulator — paired with SCALE-Sim's `[cycle × port]` demand matrix as the data
structure to compute it over.

**Single biggest mismatch with our target.** It models one monolithic array running one layer
at a time with non-contending memory ports. Two of our three mapping axes — pipelining across
cores, and replication with shared per-column DMA — have no encoding in its input format and
no arbiter in its cost model; and its one global stall scalar is precisely the wrong shape for
a 32-core array where contention is local to a column.
