# LLMCompass — research notes

**Subject.** Hengrui Zhang, August Ning, Rohan Baskar Prabhakar, David Wentzlaff (Princeton),
*"LLMCompass: Enabling Efficient Hardware Design for Large Language Model Inference"*, ISCA 2024.

**Primary sources used.**
- Paper PDF: <https://parallel.princeton.edu/papers/isca24_llmcompass.pdf> (text extracted locally; section
  numbers below are the paper's own, line numbers refer to my `pdftotext -layout` extraction in
  `scratchpad/research-llmcompass-private/paper.txt`).
- Repo: <https://github.com/PrincetonUniversity/LLMCompass>, cloned at depth 1 to
  `scratchpad/research-llmcompass-private/LLMCompass` (main branch; AE branch is `ISCA_AE`).
  The whole codebase is **7,878 lines of Python** — I read essentially all of the load-bearing parts.
- DOI/venue records: <https://dl.acm.org/doi/10.1109/ISCA59077.2024.00082>,
  <https://ieeexplore.ieee.org/document/10609604/>. Zenodo artifact DOI badge in the repo README:
  10.5281/zenodo.10892431.

**Headline for our purposes, stated up front.** LLMCompass costs each operator in a transformer layer
**independently and adds the results**. There is no overlap term between operators anywhere in the code, and
the paper says so explicitly: *"We do not explore operator fusion in this paper"* (Sec. VI-2, paper.txt:899).
Its `Mapping` object has **no field that names a core**. Our pipelining axis — distinct stages resident on
distinct cores handing off L1→L1 — is therefore not merely unimplemented, it is **structurally
inexpressible**. The thing our `fused` mode is defined by is the thing this tool cannot represent.

---

## 1. Data space — how an LLM is described

**Not an operator graph, not a parsed model file. A hand-written Python class per model shape.**

`software_model/transformer.py` contains exactly two usable workload classes:

| class | stage | line |
|---|---|---|
| `TransformerBlockInitComputationTP` | prefill ("initial computation") | :20 |
| `TransformerBlockAutoRegressionTP` | decode ("auto regression") | :355 |

Constructor parameters are only `(d_model, n_heads, device_count, data_type)` — see `docs/run.md` §Step 2.
Note what is *absent*: `n_layers`. The class that would have held it, `LLMInitComputationTP`
(transformer.py:712), is a **stub whose body is `pass`**. Every test in `ae/` instantiates exactly one
transformer block; whole-model figures are per-block latency × layer count applied outside the model
(e.g. Fig. 13 caption: *"running 48 GPT-3 layers (half of GPT-3)"*, paper.txt:892). *Inference:* inter-layer
effects (weight streaming, cross-layer reuse) are outside the model by construction.

**The "computational graph" is a shape-propagating Python trace, not a traversable data structure.**
`__call__` runs straight-line Python that applies operator objects to `Tensor` objects, and `Tensor` carries
only shape + dtype. Each operator caches its own shapes as side effects. There is no edge list, no IR, no
graph object. `Matmul.ComputationalGraph` (matmul.py:206) is literally just `(M, N, K, data_type)`.

The template is **fixed and GPT-3-shaped** (prefill body, transformer.py:60–112): Q/K/V proj →
reshape/transpose → QKᵀ → softmax → AV → transpose/reshape → Wo → LayerNorm → all-reduce → W1 (4·d) →
GeLU → W2 → LayerNorm → all-reduce. That is **post-LN, GeLU, MHA, 4× FFN**. No RoPE, no RMSNorm, no
SwiGLU, no GQA. Available operators are only `Matmul`, `BatchedMatmul`, `Softmax`, `LayerNorm`, `GeLU`,
`AllReduceMultiPCB`, plus zero-cost `Reshape`/`Transpose`/`Concat` (`docs/run.md` §"Build Your Own LLM").
Retargeting to a modern model means writing new Python operator classes, each with its own hand-written
`compile_and_simulate`.

**Prefill vs decode** is the two-class split above. Decode differs by (a) M = 1 — activations are
`[b, 1, d]` — and (b) explicit KV cache tensors, transformer.py:409–430:

```python
K_cache = Tensor([b, h // dev_cnt, d_h, s], self.data_type)
V_cache = Tensor([b, h // dev_cnt, s, d_h], self.data_type)
...
K_T = self.K_concat(K_cache, k_T, 3)   # [b, h/dev, d_h, s+1]
V_T = self.V_concat(V_cache, v_T, 2)   # [b, h/dev, s+1, d_h]
```

**How the KV cache is actually modelled: as a shape, and as a capacity number. Not as traffic.**
This is important and easy to miss. `Concat`, `Reshape` and `Transpose` are constructed in `__init__` and
called in `__call__`, but **`compile_and_simulate` never calls them** — compare the decode cost function
(transformer.py:551–640), which sums exactly twelve terms: 3×qkv, Q·K, A·V, Wo, W1, W2, softmax,
2×layernorm, gelu, 2×all-reduce. The KV append, the transposes and the reshapes contribute **zero cycles**.
The cache is felt only through (i) the operand shapes of `Q_mul_K` / `A_mul_V`, which grow with `s`, and
(ii) `self.memory_requirement` (transformer.py:458–467), a pure capacity accounting used to check the model
fits in DRAM. `Operator.Concat` does populate `load_count`/`store_count` (operators.py:78–80), but nothing
ever reads them.

*Consequence for us:* **layout changes are free in this cost model.** On a GPU that is roughly defensible
(fused into kernels by cuBLAS/PyTorch). On an AIE array a transpose is a real DMA with real stride limits.
A model that prices them at zero systematically favours exactly the layouts that cost us most — and it also
means LLMCompass would **under-report the benefit of fusion** even if fusion were added, because the
data movement fusion eliminates is already free in its baseline.

---

## 2. Mapping space — the actual data structure

`Matmul.Mapping` (matmul.py:219–260) is the whole mapping. Thirteen fields:

```python
class Mapping:
    def __init__(self,
        l2_tile_M, l2_tile_N, l2_tile_K,     # tile shape at the global buffer
        is_l2_double_buffering,              # bool
        l1_tile_M, l1_tile_N, l1_tile_K,     # tile shape at each core's local buffer
        l2_loop_order,                       # one of mkn mnk nkm nmk knm kmn
        l1_loop_order,                       # one of mkn mnk nkm nmk knm kmn
        l0_M_tiling_factor,                  # split across lanes within a core
        l0_N_tiling_factor,
        l0_K_tiling_factor,
        dataflow="os"):                      # only "os" is ever used
```

Three levels: **L2** (global buffer / shared SRAM) → **L1** (per-core local buffer) → **L0** (lanes =
`sublane_count` = `systolic_array_count` inside one core). `l0_*_tiling_factor` must multiply to
≤ `systolic_array_count` (matmul.py:1332). Loop order is a genuine per-level knob — six permutations at
each of L2 and L1 (`generate_tile_loops`, matmul.py:173–204).

**How work reaches cores — the "wave" model.** This is the only parallelisation mechanism, and it is
implicit. `simulate_l2_tile_compute_cycle_count` (matmul.py:1176–1291) walks the L1 tile loop in
`l1_loop_order`, accumulating tiles into `active_l1_tile_list` until `len(...) == core_count`, then costs
that batch as one wave:

- wave compute = **max** over the tiles in the wave (matmul.py:1224);
- wave read = sum of operand tiles that are *newly* needed, computed by boolean-masking this wave's operand
  footprint against the previous wave's (`current_batch_Read_M_K * (~previous_batch_Read_M_K)`,
  matmul.py:1231–1249) — this is the "memory access merging" of Fig. 4;
- waves are software-pipelined: `max(current_read, previous_compute) + previous_write` (matmul.py:1268).

So **all cores always execute the same operator, in lockstep waves, each on its own output tile.** There is
no core index, no spatial coordinate, no per-core assignment field. The `l1_loop_order` is the *only*
control over which tiles land in the same wave.

**Parallelism axes actually expressible:**
1. Tiling at three levels (the 9 tile-extent fields).
2. Reduction splitting: `l0_K_tiling_factor > 1` adds an intra-core reduction on the vector unit
   (matmul.py:1348); `k > 0` in the L1 loop adds an inter-core partial-sum reduction through L2
   (matmul.py:1218–1223, and `K_reduction_cycle_count` at :987).
3. **Tensor parallelism across devices** — `device_count` in the transformer classes splits `d_model`
   (transformer.py:28–33) and inserts two all-reduces per layer (paper Sec. II, paper.txt:196).
4. Loop order at each level.

**Our pipelining axis: absent, and structurally so.** There is no field assigning an *operator* to a core;
the only assignment is tile→core and it is implicit round-robin by loop order. Two operators can never be
resident simultaneously. The paper's own scope confirms it: pipeline parallelism is mentioned only as a
device-level, peer-to-peer thing (*"we don't model more communication primitives as LLM inference only
requires all-reduce for tensor parallelism and peer-to-peer for pipeline parallelism"*, paper.txt:335–337),
and `communication_primitives.py` implements only `AllReduceMultiPCB`.

Worth noting for calibration: they **do** know how to write an overlap term — double buffering at both L2
and L1 is exactly `max(read, prev_compute) + prev_write`. They simply never apply it across an operator
boundary. That is a design choice about scope, not an oversight, and it is the choice that makes the tool
unusable for our central question.

---

## 3. Hardware description — and whether an AIE array fits

JSON template → `read_architecture_template` / `template_to_system` (dse.py:22–107) → `System`.
Canonical example: `configs/template.json` and `configs/GA100.json`, documented in `docs/run.md`.

```
System           = Device + InterConnectModule                       (hardware_model/system.py)
Device           = ComputeModule + IOModule + MemoryModule           (device.py)
ComputeModule    = Core × core_count, clock_freq, l2_size,
                   l2_bandwidth_per_cycle, Overhead                  (compute_module.py:118)
Core             = VectorUnit + SystolicArray + systolic_array_count
                   + SRAM_size                                       (compute_module.py:59)
IOModule         = bandwidth (scalar B/s) + latency                  (io_module.py)  ← the entire DRAM iface
MemoryModule     = capacity                                          (memory_module.py)
InterConnectModule = device_count, topology ∈ {RING, FC}, LinkModule,
                   link_count_per_device                             (interconnect.py:35)
```

`hardware_model/arch_template.py` is a **four-line empty stub** — dead code, ignore it.

### Could it express NPU2 / AIE2P (8 cols × 4 rows = 32 cores, shim + memtiles)?

**What maps, loosely:**
- 32 compute cores → `core_count: 32`. L1 64 KB → `SRAM_KB: 64`. Memtile total → `global_buffer_MB`.
- The AIE vector unit has a plausible home in `VectorUnit(total_vector_flops_per_cycle, word_size,
  flops_per_exp, vector_width, vector_count)`.
- DRAM bandwidth → `io_module.bandwidth`.

**What does not map — and these are exactly our constraints:**

1. **There is no channel concept anywhere in the codebase.** DRAM is one scalar `io_module.bandwidth`;
   L2 is one scalar `l2_bandwidth_per_cycle` (B/cycle). Every transfer cost is
   `bytes / bandwidth` (e.g. `simulate_l2_tile_io_cycle_count`, matmul.py:1009–1020). **"Two shim MM2S
   channels per column, budgeted per-column across the segment" cannot be written down**, and neither can
   the failure mode — exceeding a channel budget silently packet-multiplexes, which is a *degradation
   event*, and a bandwidth divisor has no way to represent an event.
2. **L2 is one monolithic shared pool, not 8 per-column memtiles.** No partitioning, no per-column capacity,
   no affinity between a column's memtile and its own 4 cores. `l2_size` is a single integer checked once
   against the working set (matmul.py:319–333).
3. **No core adjacency and no L1↔L1 path.** Cores communicate only by writing to and reading from L2. AIE's
   cascade and neighbour-memory sharing have no representation, so the hand-off that *defines* our
   pipelined mode has no cost model even if you hand-wrote a fused operator.
4. **No shim tile, no segment, no residency scope.** Nothing expresses "on-chip residency holds only within
   one segment."
5. **Interconnect is off-chip only** (NVLink/TPU-link between devices). The paper concedes the general
   point in Sec. VI-1 (paper.txt:886–894): *"LLMCompass does not incorporate network modeling, and therefore
   cannot accurately model Cerebras wafer-scale processors, which have 850K cores and are more like a
   distributed system where inter-core communication mechanism plays a key role."* **That caveat is about
   us.** A 32-core AIE array with explicit programmed inter-core DMA is a small member of exactly the class
   the authors name as out of scope.
6. **The core model is a systolic array.** AIE cores are VLIW vector processors. You would have to fake a
   `SystolicArray(height, width, mac_per_cycle, ...)` of matching peak MACs; then the small-tile path calls
   **SCALE-Sim**, an output-stationary systolic-array simulator (matmul.py:1440), which would be modelling
   a machine we do not have. The large-tile path (matmul.py:1369–1402) short-circuits to
   `M·N·K / H / W / mac_per_clock / 0.98…0.99` — i.e. peak FLOPs with a fudge factor.
7. **`template_to_system` hardcodes A100 constants for every user-described architecture.**
   dse.py:68 passes `overhead_dict["A100"]` unconditionally, so any target you describe silently inherits
   A100's fitted per-kernel overheads — `matmul 21 µs, softmax 12 µs, layernorm 45 µs, gelu 45 µs`
   (compute_module.py:111–115). dse.py:41 likewise hardcodes `flops_per_exp = 35`. For our decode path,
   where a whole layer is hundreds of µs, a fabricated 45 µs LayerNorm constant would dominate the answer.
   *This is a genuine trap:* the tool will happily produce numbers for an NPU config and they will be
   contaminated by GPU-fitted constants with no warning.

**Verdict:** the hardware model is usable for us only as a *source of ideas*, not as a tool. Items 1–4 are
not gaps you patch with a config file; they require adding a resource/locality concept the framework does
not have.

---

## 4. Cost model and its validation

### Method
**Analytical + deterministic tile-loop walk + a memoised cycle-level micro-kernel simulator.** Not a
whole-chip cycle simulator; there is no contention model, no queueing, no DMA engine model.

- `Matmul.compile_and_simulate` enumerates mappings, calls `simulate(graph, mapping, device)` on each, keeps
  the argmin, returns `min_cycle_count / clock_freq` (matmul.py:733–740).
- `simulate` builds an `L2TileSimulator` per distinct tile shape (including the seven ragged-remainder
  cases, matmul.py:809–894), walks the L2 loop, and accumulates
  `max(current_read, previous_compute) + previous_write` when double-buffered, or the plain sum when not
  (matmul.py:941–953).
- Each `L2TileSimulator` recurses into an `L1TileSimulator` grid and the wave model of §2.
- The innermost GEMM cycle count comes from **SCALE-Sim** [Samajdar et al.], memoised into
  `systolic_array_model/look_up_table_{H}_{W}.csv` keyed by `(M,N,K,H,W,dataflow)`, with an analytic
  short-circuit for large well-shaped tiles (matmul.py:1356–1477). Cache misses shell out to SCALE-Sim and
  append the result to the CSV.
- Per-operator latency = simulated cycles / clock **+ a fixed `Overhead` constant** fitted per operator class
  per machine by *"running the operator with an input of size 1"* (paper Sec. III-C, paper.txt:314).
- A `roofline_model` path exists alongside, for the paper's roofline comparison bars.
- Communication uses a LogGP-style link model, `T = L + O + n̂/B` with flit/payload inflation
  (paper Eqs. 1–2, paper.txt:319–331), plus ring all-reduce.
- Area/cost model (Sec. III-D) is separate: transistor-count-based die area, validated to
  **5.1% (GA100) and 8.1% (Aldebaran)** error (paper.txt:523).

### The claimed numbers (paper Sec. III-C, paper.txt:316–323 — verbatim)

> *"for Matmul, Softmax, LayerNorm, GELU, and all-reduce, LLMCompass achieves an average error rate of
> **9.0%, 12.0%, 13.8%, 5.0%, and 14.9%** respectively. For LLM inference, LLMCompass achieves an average
> error rate of **0.69%** and **7.5%** for prefill and decoding respectively. On average, LLMCompass achieves
> a **10.9%** error rate for different operators at various input sizes and a **4.1%** error rate across the
> prefill and decoding stages."*

Platforms (paper.txt:297–315): (1) 4× **NVIDIA A100 SXM4 80 GB** fully connected by NVLink, CUDA 11.7 /
PyTorch 2.0, FP16, `torch.compile` on for LayerNorm and GELU; (2) a Google Cloud **TPU v3** node with 8 cores
in a 2D torus, JAX 0.4.18, matmul in BF16 and everything else FP32; (3) one **AMD MI210**, ROCm 5.4.2,
FP16 matmul / FP32 others, **clock pinned to 1400 MHz "to avoid frequency fluctuation"** (footnote 3).
All-reduce benchmarked with `nccl-tests`. Workload for Fig. 5i–l: one GPT-3 layer, `d_model=12288`,
`n_heads=96`, batch 8, sequence 2048, 4-way TP on A100 / 8-way on TPU.

Additional claim, Sec. VI-1 (paper.txt:880–885): *"we asked our collaborators to validate LLMCompass on an
NVIDIA RTX A6000 without changing any code, and LLMCompass achieves within 2.5% error rate for LLM inference
workloads."* **There is no artifact for this anywhere in the repo** — no A6000 config, no A6000 data, no
script. Treat as an unsupported assertion.

### Skeptical audit — repo vs. paper

I checked the released artifact against the claims. Four findings, in descending order of importance.

**(a) The headline LLM-inference errors (0.69% / 7.5% → "4.1%") are total-vs-total, and the components are
free to cancel.** `ae/figure5/ijkl/plot_transformer.py:71` computes the reported ratio as

```python
print(f"gpu prefilling: {value_sim/value_gt}, {value_roofline/value_gt}")
```

where `value_gt` and `value_sim` are each the **sum of the 12 stacked components** (accumulated at :52–:63).
By the authors' own numbers the components carry 9–15% error individually. So "4.1% error for LLM inference"
is **one ratio of two sums per stage per platform** — a far weaker statement than "4.1% per operator", and
substantially weaker than the 10.9% per-operator figure it sits next to in the abstract. The 0.69% prefill
number in particular is a single data point in which six matmul terms, three normalisation terms, a GeLU and
two all-reduces happened to cancel. *This is the "compare distributions, not numbers" failure mode, in a
published artifact.*

**(b) There is no TPU v3 real-hardware data in the artifact, and the "Real TPUv3" bars are commented out.**
`find` over every `real_hardware/` directory returns only A100 and MI210 CSVs — no `*TPU*` file exists. In
`plot_transformer.py` the real-TPU bars are commented out for prefill (lines 84–88) and the real-TPU decode
read is commented out at line 167:

```python
# values_tpu = read_csv("real_hardware/transformerAR_TPUv3.csv")
```

Yet paper Fig. 5j and 5l are labelled "Real Google TPUv3". **The released AE cannot reproduce any TPU
validation.** Compounding this: the TPU device is configured with `IOModule(float("inf"), 1e-6)` and
`MemoryModule(float('inf'))` — **infinite DRAM bandwidth and infinite capacity** (io_module.py:8,
memory_module.py:3). Any TPU error figure is therefore produced by a compute-only model with no memory
system at all. I would not cite the TPU validation.

**(c) The all-reduce components of the "real hardware" transformer measurements are pasted in, not
measured by the measurement script.** `TransformerBlock*.run_on_gpu()` sets `allreduce_total_latency = 0`
and never measures all-reduce (transformer.py:333 and :690). Yet the ground-truth files end with values:

```
ae/figure5/ijkl/real_hardware/transformer_A100.csv    → ... 0.0028909, 0.0028909
ae/figure5/ijkl/real_hardware/transformerAR_A100.csv  → ... 26.04e-06, 26.04e-06
```

Both pairs are in a formatting style (short decimal / scientific notation) unmistakably different from the
17-significant-digit `time.time()` values in the ten lines above them. These are **8.7% of the prefill total
and 4.7% of the decode total** (I summed the files). *Inference, clearly marked:* per Sec. III-C these
presumably come from `nccl-tests`, which is a legitimate source — but the consequence is that the "Real
A100" column of the flagship validation figure is **part-measured, part-transcribed into a file the
measurement script cannot regenerate**, with no provenance recorded. There is likewise **no real-hardware
all-reduce CSV at all** (`ae/figure5/h/` has only `test_allreduce.py`), so the claimed 14.9% all-reduce
error is not reproducible from the artifact either.

**(d) Only a single transformer block is ever simulated.** `LLMInitComputationTP` is `pass`
(transformer.py:712). Whole-model numbers scale one block by the layer count outside the model.

**What does hold up.** The matmul, softmax, layernorm and GeLU ground-truth CSVs for **A100 and MI210 are
present and complete** (`ae/figure5/{ab,cf,de,g}/real_hardware/`), with real measured sweeps — e.g.
`matmul_A100.csv` records `M, N, K, latency_ms, throughput_Tflops` across an M sweep at N=K=12288. The
operator-level validation on the two GPUs is genuinely backed by artifacts, and the die-area validation has
`ae/figure6/real_hardware/die_area.csv`. The 10.9% operator figure is the trustworthy one; **the 4.1%
inference figure is the one to discount.**

Also worth flagging as a methodological cousin of a lesson we already learned: the MI210 was validated with
its clock **pinned to 1400 MHz to avoid frequency fluctuation**. Defensible, but it means the model is
calibrated against a machine held in a non-default power state — the same class of hazard as our NPU pmode
Turbo-vs-Default trap.

---

## 5. Search

Two distinct things are called "search".

**(i) The mapper's parameter search** — over `Mapping`, for **one operator on one device**. Brute-force
enumeration with an argmin on cycle count; no gradient, no learning, no pruning beyond capacity feasibility.
Selected by a `compile_mode` string (matmul.py:275–732), of which there are five:

| mode | what it enumerates |
|---|---|
| `exhaustive` | l2_M/N/K as powers of two from 2⁵ up; l1_M/N/K powers of two ≤ the l2 extent; × 6 l2 loop orders × 6 l1 loop orders × `find_permutations(systolic_array_count)` |
| `heuristic-GPU` | fixed l2_M list × 3 l2_N ratios × 4 K-tiling factors × 4 l1_M × 6 l1 K-factors × l0 permutations; loop orders **pinned to `knm`** |
| `heuristic-our-throughput` | as above, l0 pinned to (2,2,1), l1_N = l1_M, l1_K/l2_K derived to fill the buffer |
| `heuristic-TPU`, `heuristic-TPU-new` | l2 tile = whole matrix; sweep l1_M/l1_N only; l0 pinned to (1,2,1) / (1,1,1) |

**Legality is capacity-only.** Two checks: L2 working set `l2_M·l2_N + l2_N·l2_K + l2_M·l2_K ≤ l2_size/word`
(halved when double-buffering, matmul.py:314–333), and L1 working set ≤ `SRAM_size/word/2`
(matmul.py:340–348). Infeasible points are `continue`d; the L1 simulator re-`assert`s the fit
(matmul.py:1322). Double-buffering is not a free knob — it is **derived** from whether the working set fits
in half the buffer. There is no bandwidth legality, no resource-count legality, and no notion of a mapping
that is legal-but-degraded.

**(ii) Design-space exploration** over hardware JSON — `design_space_exploration/dse.py`, sweeping core
count, core design, buffer sizes, memory type and capacity (Fig. 13). Also brute-force.

**Published sizes and times:**
- Abstract (paper.txt:27–30): *"simulating a 4-NVIDIA A100 GPU node running GPT-3 175B inference can be done
  within 16 minutes on commodity hardware, including **26,400 rounds of the mapper's parameter search**."*
  Sec. I repeats *"only 15–16 minutes"* (paper.txt:106–108).
  *Inference:* `ae/figure5/ijkl/run.sh` invokes `compile_and_simulate(..., "heuristic-GPU")`, so **26,400 is
  the pruned mode's count, not `exhaustive`'s.** The headline speed number is for the heuristic mapper.
- Fig. 13 caption (paper.txt:892–894): *"It took **84 minutes** to collect all the data points on one Intel
  Xeon Gold 6242R CPU @ 3.10GHz"* — sweeping compute system design, buffer size, memory type and capacity,
  at 48 GPT-3 layers, input 1024 / output 1024, 4-way TP.
- Full AE runtime from the README: ~100 min (Fig. 5) + 20 + 40 + 30 + 45 + 5 + ~240 min (Fig. 12).

No mapspace **cardinality** is published — only round counts for a chosen mode. Sec. III-B3 offers the one
structural remark: Softmax/LayerNorm/GeLU have fewer dimensions than matmul, so *"the mapper search space is
much smaller"* (paper.txt:286).

---

## 6. Multi-operator / fusion — the decisive question

**It sums independent operators. Plainly and without qualification.**

Both transformer classes compute (prefill at transformer.py:194–284, decode at :551–640):

```python
matmul_total_latency = qkv + q_mul_k + a_mul_v + h_matmul0 + h1_matmul1 + h2_matmul2
normlization_total_latency = softmax_latency + layernorm_latency * 2
self.latency = matmul_total_latency + normlization_total_latency + gelu_latency + allreduce_total_latency
```

where each term is an **independent** `op.compile_and_simulate(device, compile_mode)` plus a constant
`Overhead`. `grep "max(" software_model/transformer.py` returns nothing but unrelated lines: **there is no
overlap term between any two operators anywhere in the workload model.**

Every operator is mapped and simulated in isolation, starting and ending in the memory hierarchy. Producer→
consumer residency does not exist: the Wo output is written out and re-read by LayerNorm, and neither the
write nor the re-read is attributed to a fusion opportunity. Each operator independently re-runs the mapper
from scratch on the same device, as if it were the only thing on the chip.

The paper states the scope decision outright (Sec. VI-2, paper.txt:895–902):

> *"LLMCompass can be extended to a variety of optimization techniques. To support operator fusion like
> FlashAttention, users can implement a simulated fused operator based on the simulation code for its
> individual operators. **We do not explore operator fusion in this paper** as many of them are specific to
> NVIDIA GPUs and we are not sure whether they can be applied to other hardware platform such as Google
> TPUs."*

**So, plainly: LLMCompass cannot express the thing our `fused` mode is defined by.** Three independent
reasons, any one of which is sufficient:

1. The composition rule for operators is **addition**.
2. `Mapping` has **no field naming a core**; cores are an anonymous homogeneous pool that all run the same
   operator each wave.
3. There is **no L1↔L1 path** in the hardware model, so even a hand-written fused operator would have no
   cost model for the hand-off — its only inter-core medium is the shared L2 pool.

The escape hatch the paper offers — "write your own fused operator class" — means writing the interaction
model yourself. The framework contributes nothing to the part we care about. And per §1, the transposes and
KV-concat a fusion would eliminate are already priced at zero, so the framework's baseline is biased
*against* showing fusion benefit even after you do that work.

**One nuance in fairness:** the *intra*-operator model does have real pipelining — double buffering with
`max(read, prev_compute) + prev_write` at both L2 and L1. The machinery for an overlap term exists. It is
applied only inside an operator, never across one.

---

## 7. What transfers

### Worth stealing

1. **Wave batching with cross-wave operand deduplication — the single most transferable idea.**
   (matmul.py:1197–1274.) Fill the core group with tiles in `l1_loop_order`; cost the wave as
   `max(new-operand read, previous wave's compute) + previous wave's write`, where "new operand" is derived
   by boolean-masking this wave's operand footprint against the previous wave's. It is a cheap, honest way
   to price reuse across a group of cores without simulating a NoC.
   **Why it fits us specifically:** it converts a loop order into *a count of distinct operand tiles that
   must cross a level per wave* — and a count of transfers is precisely the currency our per-column MM2S
   budget is denominated in. Map a column's 4 cores to a wave and the mask directly yields how many distinct
   L2→L1 transfers that wave needs. The adaptation we would have to make is the interesting part: turn that
   count into a **legality/degradation predicate** (count ≤ 2 per column per segment, else packet-multiplex)
   rather than a bandwidth divisor. That is a small change to their formulation and it is exactly the
   instrument we lack.

2. **A measured per-operator-class overhead constant.**
   `Overhead(matmul, softmax, layernorm, gelu)` (compute_module.py:103–115), fitted per machine by running
   each operator at input size 1. Crude, but the right *shape* for our XRT dispatch overhead (~50–200 µs),
   and it is measured rather than assumed. We have the hardware to fit our own table, and having it would
   immediately tell us whether a given fusion pays for itself — one measured constant per kernel class.
   (Note the anti-pattern to avoid: `template_to_system` reuses the A100 table for every target, dse.py:68.)

3. **Loop order as a first-class, per-level mapping knob.** Six permutations, enumerated separately at L2 and
   L1. Small, finite, and it changes how many distinct operand tiles cross each level — which under our
   channel budget is a legality question, not just a performance one.

4. **Memoised micro-kernel lookup keyed by shape, with an analytic short-circuit.**
   `look_up_table_{H}_{W}.csv` keyed by `(M,N,K,H,W,dataflow)`, populated on demand, with a closed-form
   fast path when the tile is large and well-shaped (matmul.py:1356–1402). Our analogue writes itself:
   **measure each (kernel, tile shape) once on real hardware, cache it, cost mappings by table lookup.**
   Given that every figure we have is measured, this is the structure that lets isolated measurements
   compose into a balance instrument without needing a simulator.

5. **Capacity-legality as a cheap `continue`, and double-buffering as a *derived* consequence of capacity
   rather than a free knob.** The right discipline for our L1 (64 KB) and memtile budgets.

### Does not transfer

1. **Data movement as two scalars** (`io_module.bandwidth`, `l2_bandwidth_per_cycle`). Our binding
   constraint is a per-column *count of channels* whose violation is a silent degradation, not an aggregate
   B/s. Nothing here can be bent into that without inventing a resource concept the framework lacks.
2. **Monolithic shared L2.** Our L2 is 8 per-column memtiles with affinity to their own 4 cores.
3. **Anonymous lockstep core pool.** Our pipelining axis *is* core identity.
4. **Additive operator composition.** Kills fused mode outright.
5. **Free transposes / reshapes / KV concat.** Biases every conclusion toward layouts that cost us DMAs, and
   suppresses the measured benefit of fusion.
6. **Systolic-array + SCALE-Sim core model.** Wrong machine for a VLIW vector core; the small-tile path is
   actively misleading and the large-tile path degenerates to peak FLOPs × 0.98.
7. **Device-to-device-only interconnect**, plus the authors' own Sec. VI-1 concession that machines "where
   inter-core communication mechanism plays a key role" are out of scope.
8. **No segment / residency-scope analogue.** Nothing expresses that on-chip data survives only within a
   bounded region of the schedule.

### On the "closest workload" framing — is the closeness real?

**Partly real, superficial exactly where it matters to us.** Real: the workload genuinely *is* transformer
inference; the prefill/decode split is a first-class structural distinction, not a batch-size parameter; and
KV-cache growth genuinely drives the QKᵀ / AV operand shapes.

Superficial where it counts: (a) it is a **fixed GPT-3-shaped template** — post-LN, GeLU, MHA, 4×d FFN —
not our model family, and extending it means writing operator classes with hand-written cost functions;
(b) the **KV cache is a shape and a capacity number, not modelled traffic**; (c) the **decode path is M=1
GEMMs costed independently and summed** — precisely the regime where per-dispatch overhead and fusion
dominate on our hardware, and precisely where an additive model is weakest.

The tool's closeness to us is at the level of *which operators appear*, not *how they interact*. Our mapping
problem lives entirely in the interaction.

**One framing point that explains everything above.** LLMCompass is built to answer the **dual** of our
question. It holds the mapping problem cheap so it can search *hardware*: "should I halve compute capability,
swap HBM for DDR?" — its headline results are 3.41× performance/cost over an A100 and a design at 95.3% of
GA100 performance in 57.9% of the area. The mapper exists only so that a candidate hardware design is not
unfairly handicapped by a bad mapping. **We have fixed hardware and want to search mappings.** That is why
the mapping space is thin, why legality is capacity-only, and why operator composition is addition — none of
those were oversights, they were the right economies for the question being asked. It also means we should
expect to take *mechanisms* from this work (the wave/dedup cost kernel, the measured overhead table, the
memoised shape→cycles cache) and none of its *framing*.

---

## Comparable summary

- **Data-space representation** — Hand-written Python classes per model shape (`TransformerBlockInitComputationTP` for prefill, `TransformerBlockAutoRegressionTP` for decode), parameterised only by `(d_model, n_heads, device_count, data_type)`; the "graph" is a shape-propagating trace over `Tensor` shape-holders, not a traversable IR. Fixed GPT-3 template (post-LN, GeLU, MHA); KV cache is a `Tensor` shape plus a capacity number — its append, and all transposes/reshapes, cost **zero cycles**.
- **Mapping-space representation** — `Matmul.Mapping`: 13 scalar fields — L2 tile M/N/K, L1 tile M/N/K, L0 tiling factors M/N/K (across lanes), two independent loop orders from 6 permutations each, a double-buffer bool, and a dataflow string. Cores receive tiles implicitly in `core_count`-sized lockstep waves; **no field names a core, and there is no operator→core assignment.** Plus device-level tensor parallelism (`device_count`) with all-reduce.
- **Legality model** — Capacity-only, checked twice: L2 working set ≤ `l2_size/word` (halved when double-buffering, which is *derived* not chosen), L1 working set ≤ `SRAM_size/word/2`, and `l0_M·l0_N·l0_K ≤ systolic_array_count`. Infeasible points are skipped. No bandwidth, resource-count, or channel legality; no concept of legal-but-silently-degraded.
- **Search strategy** — Brute-force enumeration with argmin on cycle count, per operator, per device; five `compile_mode` presets from `exhaustive` down to per-vendor heuristics that pin loop orders and L0 factors. Outer hardware DSE is also brute-force. Published: 26,400 mapper rounds and 15–16 min for a 4-A100 GPT-3-175B layer (**heuristic** mode, not exhaustive); 84 min for the whole Fig. 13 sweep on one Xeon Gold 6242R. No mapspace cardinality published.
- **Cost model** — Analytical + deterministic tile-loop walk, with the innermost GEMM from SCALE-Sim memoised into shape-keyed CSV lookup tables (and a closed-form peak-FLOPs×0.98 short-circuit for large tiles); double buffering as `max(read, prev_compute) + prev_write` at L2 and L1; per-operator fitted additive `Overhead` constants; LogGP link model + ring all-reduce for inter-device. **Validation: trust the 10.9% operator figure on A100/MI210 (real CSVs are in the artifact); discount the 4.1% inference figure — it is a ratio of two 12-term sums where 9–15% component errors cancel, two of its twelve "real" components are transcribed values the measurement script cannot produce, and the TPU half of the validation has no data in the artifact at all and was run with infinite DRAM bandwidth and capacity.**
- **Multi-op support** — **None.** Operator latencies are summed with no overlap term anywhere; paper states "We do not explore operator fusion in this paper" (Sec. VI-2). Intra-operator pipelining exists (double buffering) but is never applied across an operator boundary. Pipelined residency across cores is inexpressible for three independent reasons: additive composition, no core-naming field in `Mapping`, and no L1↔L1 path in the hardware model.
- **Single most transferable idea** — The **wave cost kernel with cross-wave operand deduplication** (matmul.py:1197–1274): fill a core group with tiles in loop order, cost it as `max(newly-needed operand reads, previous wave's compute) + previous wave's write`, where "newly needed" is a boolean-mask difference against the previous wave. It turns a loop order into a **count of distinct transfers crossing a memory level per wave** — the exact currency of our per-column MM2S budget. Adapt it by making that count a legality/degradation predicate (≤ 2 per column per segment) rather than a bandwidth divisor. Runner-up: the measured per-operator-class `Overhead` constant, fitted at input size 1 — the right shape for our XRT dispatch cost.
- **Single biggest mismatch with our target** — All data movement is priced as `bytes / scalar_bandwidth` against a monolithic shared L2 and a single DRAM bandwidth number. **There is no channel, no per-column resource, no core adjacency, and no L1↔L1 path anywhere in the codebase** — so neither our hard constraint (two shim MM2S channels per column, budgeted per-segment, silently packet-multiplexing on violation) nor our central mapping axis (pipelined multi-stage residency) has any representation. The paper concedes the general case itself in Sec. VI-1: machines "where inter-core communication mechanism plays a key role" are out of scope, and a 32-core AIE array with programmed inter-core DMA is squarely in that class.
