# iron `encoder_pipeline` — extraction reference for the mlir-air execution-mode study

Extracted 2026-08-12 from `/home/cj/iron`, branch `extend_enc_pipeline` (tip `64a1f29`, 27 commits ahead of `devel`, unmerged). **Read-only; nothing was built, run, or modified in either repo.**

## Citation convention

- `iron:<path>:<line>` — file present on `extend_enc_pipeline`. Read via `git show extend_enc_pipeline:<path>`. All paths are relative to `/home/cj/iron`.
- `iron@<rev>:<path>:<line>` — file **deleted** from the branch, recovered from history. Every such citation was verified by `git show <rev>:<path>`.
- **[MEASURED]** — a figure that appears in an iron artifact.
- **[INFERENCE]** — my reasoning, not stated in iron.
- **[UNSOURCED]** — a claim iron's own docs make that has no artifact behind it, or that iron's code contradicts.

---

## 0. Provenance — read this before using any number

The brief's framing ("iron's `extend_enc_pipeline` has a working fused BERT encoder layer, and the balance numbers are in `encoder_pipeline_archive/current_status.md`") is right in substance but conflates **three different operator generations**. Getting this wrong will misroute the port, so it is the first section.

### 0.1 Three generations, not one

| Gen | Package | Status on `extend_enc_pipeline` | Knobs | LN1→FFN staging |
|---|---|---|---|---|
| **G1** | `operators/encoder_pipeline/` (pre-split, shared) | **deleted** (`b5538a7`, `a1327c9`, 2026-03-13) | `parallel_heads`, `parallel_ffn`(=`nB_tiles_distributed`), `proj_acc_depth`, `o_proj_acc_group_size` | runtime switch `ln1_staging_design ∈ {"memtile","ddr"}` |
| **G2** | `encoder_pipeline_ddr/` + `encoder_pipeline_memtile/` (mode split) | **`_ddr` deleted; `_memtile` survives** | same as G1 | frozen per package |
| **G3** | `iron/operators/encoder_pipeline/` (new, placement-driven) | **live, benchmarked** | G1's knobs **plus `parallel_seq`, `seq_tile`, `kv_seq_tile`, `emb_tile`, `ffn_tile`** | **DDR only — no switch** |

Evidence:
- G1→G2 split plan and its completion: `iron:iron/operators/encoder_pipeline_archive/mode_split_plan.md:1-5, 272-296`.
- G2's `_memtile` hard-codes the mode and deletes the argument: `iron:iron/operators/encoder_pipeline_memtile/op.py:69-70` — `del ln1_staging_design; self.ln1_staging_design = "memtile"`. Its design accepts `ln1_stage_mode` only to discard it: `iron:iron/operators/encoder_pipeline_memtile/design.py:153-154` — `del ln1_stage_mode; stage_ln1_to_ddr = False`.
- `encoder_pipeline_ddr/` **does not exist on this branch** (`git ls-tree -r --name-only extend_enc_pipeline | grep encoder_pipeline_ddr` → empty), although `encoder_pipeline_archive/README.md:6-8` still names it as a "live implementation". That README is stale.
- G3 is a *new* file, not a rename: `git log` for `iron/operators/encoder_pipeline` on this branch starts at `70d2e29` (2026-03-17), after the G1 deletion. G3's `design.py` is 6764 lines vs G1's 4962.

**Consequence:** `encoder_pipeline_archive/current_status.md` — the doc the brief points at — documents **G1**. Its case IDs, its `lnstage_memtile`/`lnstage_ddr` selections, its four topology constraints and its `full`-vs-stage numbers are all G1 measurements. G3, the thing that is actually benchmarked today, has a different (larger) knob set, a different test-ID scheme (`iron:iron/operators/encoder_pipeline/test.py:887`), and **no staging switch at all**.

### 0.2 Correction to the brief

> "`encoder_pipeline` exists on **no other branch**"

Not literally true. `encoder_pipeline/design.py` also exists on `dev-mha-an-combine-bufs` (4512 lines), `thesis_design_patterns` (4459), `update_bert_clean` (4459), `origin/dev-encoder-pipeilne` (954) and `origin/update_bert_rebased` (3651) — all older G1-lineage snapshots.

**The load-bearing half of the claim holds and is confirmed:** it is on **neither `devel` nor `final_exec_strats`** (`git ls-tree -r --name-only final_exec_strats | grep -i 'operators/encoder'` → empty). The mlir-air study was ported from a tree that genuinely never contained this operator.

### 0.3 Artifact availability — what you can and cannot re-derive

| Thing | Where | Re-derivable? |
|---|---|---|
| G3 design, placements, topology registry, tests | live on branch | yes |
| G2 memtile design + its DDR dead-code branch | live on branch | yes |
| **G2/G1 DDR *mode implementation*** (`design_ln1_ddr.py`, 482 lines) | **deleted**; `iron@b5538a7^:operators/encoder_pipeline/design_ln1_ddr.py` | recoverable from git only |
| **The stage-profile instrument** (`profile_debug_modes.py`, 265 lines) | **deleted**; `iron@b5538a7^:operators/encoder_pipeline/profile_debug_modes.py` | recoverable from git only |
| **G1 `test.py` with `stage_profile`** (370 lines) | **deleted**; `iron@b5538a7^:operators/encoder_pipeline/test.py` | recoverable from git only |
| **Machine-readable measurement artifacts** (CSV/JSON of the reported latencies) | **none exist anywhere in the repo** | **no** |

That last row is the important one. `git log --all --diff-filter=A -- '*encoder_pipeline*results*.csv'` returns nothing; the branch contains no `.csv` or results `.json` under `iron/operators/`. The profiler had an `--output-json` flag but it defaults to `None` (`iron@b5538a7^:.../profile_debug_modes.py:176-181`), and the pytest CSV reporter never scraped the stage-profile tests because those tests carry no `@pytest.mark.metrics` decorator (contrast `iron@b5538a7^:.../test.py:318-321` with `:337-339`).

**Every latency figure in `current_status.md`, `design_optimization_plan.md`, `debug_findings_2026-03-04.md` and `ffn_latency_optimization_log_2026-03-08.md` is prose transcribed by hand from console stdout.** That is precisely the mlir-air "claim without an artifact" failure mode. Treat all of them as single-run console readings unless otherwise noted.

Forensic support for "hand-transcribed" (this is inference, but tight): the differences are exact (`16488.12 − 11739.34 = 4748.78`; `12226.31 − 8984.99 = 3241.32`; `4349 − 3514 = 835`), the two-decimal precision matches the profiler's `f"latency_us={...:.2f}"` (`iron@b5538a7^:.../profile_debug_modes.py:224`) rather than pytest's one-decimal `f"Latency (us): {...:.1f}"` (`:332`), and no code anywhere computes the subtraction. **[INFERENCE]**

---

## 1. What the fused design actually is

### 1.1 Scope and shape

`MHA + AddNorm1 + FFN + AddNorm2` for one BERT encoder layer, bf16, `d=64` only (`iron:iron/operators/encoder_pipeline/README.md:16`). LN1 and LN2 are genuinely two-pass layer norms because they need full-row statistics (`iron:iron/operators/encoder_pipeline_archive/current_status.md:6-7`) — every LN input FIFO is fed twice.

### 1.2 Stage decomposition (G3, non-sequence-parallel path)

Eight compute stages. Each is a separate AIE core with its own kernel:

| # | Stage | Core fn | Kernels |
|---|---|---|---|
| 0 | QKV projection *(optional, `staged_hidden_states` mode only)* | `iron:.../design.py:5263,5297,5330` | `matmul_init_bf16_bf16_{q,k,v}_proj`, `eltwise_add_*` bias (`:1017-1057`) |
| 1 | QKᵀ score | `batched_matmul_qk` `:1545` | `zero_bf16`, `matmul_bf16_bf16_wrapper` (`:985,1011`) |
| 2 | Online (flash-style) softmax | `softmax` `:1562` | `partial_softmax`, `init_scale_buffer` (`:993,1008`) |
| 3 | PV / context + rescale | `batched_matmul_pv` `:1601` | `matmul_PV`, `rescale_o` (`:1058,1071`) |
| 4 | O-proj + cross-head accumulation | `matmul_o_proj` `:1679`, grouped variants `:1778-1869` | `matmul_with_acc_bf16_bf16_o_proj` (`:1077`) |
| 5 | **AddNorm1** (two-pass LN) | `core_fn_ln1_from_replayed_inputs` `:1914`, fused `:1956` | `ln_calc_sum_sumsq` / `ln_add_calc_sum_sumsq`, `fused_add_layer_norm_1outs_fp32weights` (`:1088-1103`) |
| 6 | FFN up **+ GELU fused into the same core** | `core_fn_ffn_up_proj` `:1995`, GELU at `:2012` | `ffn_matmul_*_up_proj`, `ffn_gelu_bf16` (`:1114-1134`) |
| 7 | FFN down + cross-branch reduction chain | `core_fn_ffn_down_proj` `:2014`, grouped `:2118-2160` | `ffn_matmul_with_acc_bf16_bf16_down_proj` (`:1129`) |
| 8 | **AddNorm2** (two-pass LN) | `core_fn_add_norm2_from_replayed_inputs` `:2217` | same LN kernel set |

There is **no separate activation core** — GELU is fused into the FFN-up worker. Worth noting against R1, which builds GeLU as its own herd.

### 1.3 Where hand-offs happen — the answer is "almost everywhere on-chip, except one"

**Core→core, direct L1 / AIE stream (no memtile, no DDR):**

QK→softmax (`memA`, `:1324`), softmax→PV (`memP` `:1334`, running-scale `scaleOF` `:1344`), PV→O-proj (`outOProj` `:1348`), **O-proj head *i* → head *i+1* partial-sum cascade** (`outOPart` `:1358`), O-proj(last)→LN1 (`outOProjInput` `:1405`), FFN-up→FFN-down per branch (`ffnUpOut` `:1527`), **FFN-down branch *b* → *b+1* reduction chain** (`ffnDownReduce` `:1527-1533`), FFN-down(last)→LN2 (`ffnDownOut` `:1534`).

The tile assignment makes these physically adjacent: `iron:.../design.py:533-536` puts `qk` on row 2, `softmax` row 3, `pv` row 4, `o_proj` row 5 of the same column, one column per parallel head.

**Memtile (L2), via `.forward()` / `.split()` / `.join()` at `row=1`:**

shim→per-head Q/K/V fan-out (`:1183-1208`), shim→per-head W_O fan-out (`:1313-1321`), **O-proj accumulation loop** core→memtile→same core (`:1352, 1390-1404`), **FFN-down accumulation loop** (`:1497-1526`), B_Up/B_Down weight staging (`:1453-1526`), LN2→output shim (`:1536-1543`).

**DDR / L3, inside the fused block:**

Exactly one interior data hand-off crosses DRAM: **LN1's output**. Per Q-block iteration (`for tap_idx in q_block_schedule:` `:6449`):

```
:6683  rt.drain(ln1StageOut, OR, tap=branch_stage_tap, placement=Tile(col=ln1_stage_shim_col, row=0), wait=True)
:6691  rt.finish_task_group(tg_ln1_drain)
:6694  rt.fill(inLNFromDDR.prod(), OR, tap=branch_refill_tap, ...)      # → FFN-up
:6702  rt.fill(ffnRFromDDR.prod(), OR, tap=residual_stage_tap, ...)     # → AddNorm2 residual, pass 1
:6710  rt.fill(ffnRFromDDR.prod(), OR, tap=residual_stage_tap, ...)     # → AddNorm2 residual, pass 2
```

Four shim transactions per Q-block for one intermediate. `outLNBroadcast` (`:1418`) has exactly one consumer, `ln1StageOut = outLNBroadcast.cons(depth=1)` (`:1424`), and its only use is the drain at `:6683`. The FFN-up consumers `memOutLN[b]` (`:1433-1436`) are fed only from `inLNFromDDR` (`:1425`). **There is no on-chip edge from the LN1 core to the FFN-up cores in G3.** The sequence-parallel path does the same thing with lane-joined tensors (`:5067-5080` drain, `:5200-5213` refill, `:5215-5241` residual refills ×2).

The scratch lives in a reserved tail of the `OR` DDR buffer: `or_tensor_shape = (or_rows_before_ln1_stage + ln1_dram_stage_rows, embed_sz)` (`:896`) with `ln1_dram_stage_rows = seq_len if sequence_parallel else proj_acc_depth * seq_tile` (`:474-476`).

### 1.4 How many configurations it lowers to — **one**

- One device, one program: `iron:.../design.py:621-623` — `Program(NPU2(), rt).resolve_program(SequentialPlacer())`. Two `Runtime()` calls exist (`:4676` seq-par, `:6304` non-seq) but they are mutually exclusive branches; exactly one executes.
- One kernel, one runlist entry, **five buffer objects**: `iron:iron/operators/encoder_pipeline/op.py:1313-1318` (`add_kernel("encoder_pipeline", ...)`) and `:1364-1376` (`add_to_runlist("encoder_pipeline", "W_O", "QKV", "OR", "B_Up", "B_Down")`, or `("W_ATTN","X","OR","B_Up","B_Down")` in staged mode). Output aliases the residual buffer: `:1363` `self.buffer_aliases["O"] = "OR"`.
- Workers are `rt.start(...)`ed once up front (`:6353-6367`) with infinite core loops; the runtime sequence then issues `rt.fill`/`rt.drain` descriptors grouped into `rt.task_group()`s. This is **one persistent dataflow graph driven by a host-side DMA schedule — not repeated reconfiguration.**

### 1.5 Does iron face mlir-air's "residency holds only within a segment" constraint?

**It does not have the constraint, because it never creates a second segment.** The whole layer is one IRON `Program` on one `NPU2()` device with one runlist entry. There is no analogue of "each launch becomes its own device" because there is one launch.

That is the structural difference, and it is worth being blunt about it: **iron did not solve the multi-segment residency problem — it sidestepped it by never partitioning the layer into multiple launches in the first place.** The 8-stage chain is co-resident by construction; the only question left was how to *balance* it, which is why the operator's owner says balancing was the hard part.

Two secondary observations:

1. **[MEASURED]** Iron's own benchmark methodology names three NPU paths — `encoder_pipeline` (fused), `gemm_only` (six GEMMs offloaded, `4 + 2*num_heads` dispatches per layer, non-GEMM work on host), and `operator_runlist` (stitched runlist, "the least fused current NPU path") — `iron:iron/applications/bert/BENCHMARKING_METHODOLOGY.md:12-16, 137-181`. This is close to the mlir-air study's mode taxonomy, and the fused path is the one that wins.
2. **[INFERENCE]** Iron's `fused` is genuinely *resident* in the sense the mlir-air study means, with exactly one deliberate exception (LN1). mlir-air's shipped `fused` is *packaged*. The gap between the two is not one of ambition but of lowering: IRON lets you write one program that owns the whole array; AIR's launch/segment model does not.

---

## 2. The balancing machinery

### 2.1 The knobs

`iron:iron/operators/encoder_pipeline/topology.py:53-74` defines the complete space as a frozen dataclass:

| Knob | Meaning | Effect on cores |
|---|---|---|
| `seq_tile` | Q-block rows per iteration | none |
| `kv_seq_tile` | K/V block rows | none |
| `emb_tile` | embedding/output tile width | none (drives L1/L2 residency) |
| `ffn_tile` | FFN intermediate tile | none |
| `parallel_seq` (`ps`) | sequence lanes | **× whole 8-stage lane** |
| `parallel_heads` (`ph`) | head-parallel MHA columns | **+4 cores per head** |
| `parallel_ffn` / `nB_tiles_distributed` (`pffn`) | FFN branches | **+2 cores per branch** |
| `proj_acc_depth` (`pacc`) | embedding accumulation depth | none; sets accumulation FIFO depth |
| `o_proj_acc_group_size` (`opg`) | how many heads share one O-proj accumulator | none; swaps core fn, collapses `ph` memtile accumulators into 1 |
| `ffn_down_acc_group_size` | same for FFN-down branches | none |

Plus six *layout variant* overrides, exposed for sweeping: `weight_forward_depth`, `o_proj_fifo_depth`, `ffn_replay_fifo_depth`, `use_fused_replayed_addnorm`, `use_transport_groups`, `use_unified_qr_split` (`topology.py:16-25`).

### 2.2 The constraints

Hard, enforced in code:

1. `emb_tile * proj_acc_depth == embed_sz` — `iron:.../design.py:361-365`, `iron:.../encoder_pipeline_memtile/design.py:121-125`.
2. `ffn_intermediate_size % (emb_tile * parallel_ffn) == 0` (evenly-partitioned FFN only) — `iron:.../encoder_pipeline_memtile/cases.py:62-64,115-116`.
3. `o_proj_acc_group_size <= parallel_heads` and `parallel_heads % opg == 0` — `cases.py:117-118`.
4. `nB_tiles_distributed <= ffn_col_groups` where `ffn_col_groups = ffn_intermediate_size // emb_tile` — `encoder_pipeline_memtile/design.py:234-238`.
5. `seq_len % seq_tile == 0` and `(seq_len // seq_tile) % parallel_seq == 0` — `iron:iron/operators/encoder_pipeline/README.md:76-78`.

Resource budgets, enforced at design time (`iron:.../encoder_pipeline_memtile/design.py:530-560`), and documented at `iron:.../encoder_pipeline_archive/docs/archive/resource_utilization_2026-03-04.md:3-15`:

```
required_compute_tiles = parallel_heads*4 + 3 + 2*effective_ffn_branches  <= 32
shim_output_stream_total = 5 + estimated_b_weight_streams + ln1_ddr_streams <= 16
        # 5 fixed = Q/K/V/W_O/R ;  ln1_ddr_streams = 0 (memtile) or 1 (ddr)
```
plus memtile DMA `6 in / 6 out` per memtile, compute-tile DMA `2 in / 2 out`, memtile BD budget `48`.

Topology-legality also includes a **routing** constraint the mlir-air study has no analogue for: the FFN-down reduction chain must be a chain of N/E/S neighbours ending at a tile that is itself an N/E/S neighbour of LN2, or the design falls back to a "chain minus one bypass core" with exactly two LN2 inputs (`encoder_pipeline_memtile/design.py:296-357`). Westward reduction edges are rejected outright (`:256-264`).

### 2.3 How a balanced configuration is chosen — **three mechanisms, and none of them is derivation**

**(a) Legality: greedy auto-prune (G1/G2 only).** The memtile-lineage design wraps its whole layout resolution in `while True:` with a `prune_or_fail(...)` that *drops one FFN branch and retries* whenever a budget is exceeded (`encoder_pipeline_memtile/design.py:489-565`). This is a real, if crude, constraint search: it converges to the largest branch count that fits compute tiles, shim outputs, memtile channels and the reduction-routing rule simultaneously.

**(b) Placement: hand-authored tables (G3).** G3 replaced (a) with a **hardcoded registry**. `TOPOLOGY_PLACEMENTS` (`iron:iron/operators/encoder_pipeline/placements.py:155`) maps a 13-tuple topology key to an explicit `(col,row)` for every compute core, plus explicit memtile and shim column assignments. `LOW_HEAD_TAILS` (`:86-152`) holds the non-seq tails, keyed by `(parallel_heads, parallel_ffn)` — **only 7 combinations exist**: (1,1) (1,2) (1,4) (2,1) (2,2) (2,4) (4,1). Unsupported combinations simply do not exist. `SUPPORTED_ENCODER_PIPELINE_TOPOLOGIES` (`:1538`) is the enabled subset. `iron:.../README.md:8` states this plainly: *"The implementation is placement-driven. Supported runtime/topology combinations are the hardcoded keys in placements.py."*

**(c) Performance: exhaustive measured autotune with a per-shape cache.** This is the answer to "how is a balanced configuration chosen".

`iron:iron/applications/bert/npu_inference.py`:
- `--topology-policy {fixed, cache, autotune}`, default `cache`; `--topology-cache npu_topology_cache_latest.json`; `--candidate-topologies`; `--autotune-warmup-runs`; `--autotune-runs` (`:169-204`).
- `autotune_topology(...)` enumerates every registered topology matching the shape, **runs each one on hardware**, records `avg_latency_ms`, and prints `compute_tiles` and `utilization` per candidate.
- Between candidates it calls `cooldown_before_benchmark(...)` — sleep and/or wait until a temperature threshold (`iron:iron/applications/bert/benchmark_common.py`, `cooldown_before_benchmark`).
- Selection is **not pure argmin**:

```python
best_latency_ms = min(c["avg_latency_ms"] for c in candidate_results)
latency_band = [c for c in candidate_results if c["avg_latency_ms"] <= best_latency_ms * 1.01]
return max(latency_band, key=lambda c: (c["topology"].compute_tile_count,
                                        int(c["topology"].family_id == preferred_family_id),
                                        c["topology"].topology_id))
```
(`npu_inference.py::select_autotune_topology`) — a **1% latency band, then prefer the topology that uses more compute tiles**, then prefer the current family, then a deterministic ID tie-break. That is a deliberate hedge against measurement noise plus a bias toward the more scalable point.

The winner is cached per shape (`topology_cache_key` at `iron:.../topology.py:508-522`, keyed on all shape+tiling fields) and `find_cached_topology` (`:525-542`) only accepts a cached entry whose signature is still in the supported set.

**So: legality is derived (partly), placement is hand-tuned, and balance is searched empirically on hardware.** No cost model, no analytic balance solver.

### 2.4 The result the search actually converged to — and it is not a balanced pipeline

**[MEASURED]** `iron:iron/operators/encoder_pipeline/PERFORMANCE_IMPROVEMENT_PLAN.md:24-27`:

> The current unattended benchmark winners are still the sequence-parallel topologies:
> - `seq32_kv64__ps2_ph2_pffn2` at `seq64`
> - `seq32_kv64__ps4_ph1_pffn1` at `seq128+`

Look at what `4ps` is (`iron:iron/operators/encoder_pipeline/placements.py:1214-1257`):

```
lane 0: qk(0,2) softmax(0,3) pv(0,4) o_proj(0,5) ln1(1,5) ffn_up(1,2) ffn_down(1,3) ln2(1,4)
lane 1: ... cols {2,3}
lane 2: ... cols {4,5}
lane 3: ... cols {6,7}
```

Four lanes × 8 cores = **32 cores = the entire NPU2 array**, each lane hosting a complete, private copy of the whole 8-stage encoder chain over a different slice of the sequence. K/V/W_O ingress is shared (`shared_ingress_cols`, `:1193`), and lanes are grouped into `transport_groups` of two (`:1258-1310`) purely to fit the shim column budget.

**[INFERENCE] The empirical winner is not a balanced pipeline — it is four replicas of an unbalanced one.** Pipelining survives *within* a lane (across Q-blocks), but across lanes the design is pure data parallelism over sequence, which is legal because LN, residual add, and FFN are all row-wise and only K/V are shared. This is corroborated by `PERFORMANCE_IMPROVEMENT_PLAN.md:104-106`: *"The current mainline winner already uses essentially all compute tiles ... The remaining mainline wins are therefore more likely to come from lower data movement and synchronization cost than from more raw parallelism."*

This is arguably the single most important finding for the mlir-air study: **iron spent months on pipeline balance and the thing that won was replicate-the-whole-pipeline-per-sequence-lane.** If a resident FFN interior is achievable in AIR, the next question is not "how do I balance the three herds" but "how many independent copies of the whole thing fit".

---

## 3. The balance metric — full vs isolated-stage gap

This is the most portable artifact in the extraction. Here is exactly what it is.

### 3.1 The reported numbers

**[MEASURED]** `iron:.../encoder_pipeline_archive/current_status.md:38-53`, duplicated with ratios at `docs/archive/design_optimization_plan.md:20-39` and with provenance at `docs/archive/debug_findings_2026-03-04.md:16-28`:

| Case | `full` (µs) | max stage-only | gap | `full/max` |
|---|---|---|---|---|
| memtile control `512seq/96e/4ph/4pffn/8pacc/4opg` | 16488.12 | `addnorm1` 11739.34 | 4748.78 | 1.405 |
| "fast DDR" `512seq/**128e**/4ph/**6pffn**/**6pacc**/**2opg**` | 12226.31 | `addnorm1` 8984.99 | 3241.32 | 1.361 |
| high-`pacc` memtile `64seq/64q/48e/6ph/1pffn/16pacc/2opg` | ~4349 | `addnorm1` 3514 | ~835 | 1.238 |

Full stage table for the high-`pacc` case (`design_optimization_plan.md:103-113`): `self_attn 1637`, `mha_input 1568`, `residual 1572`, `ffn_up 1590`, `ffn_down 1387`, `addnorm2 1063`, `mha 1068`, `addnorm1 3514`, `full 4349`.

A later, fuller table for the memtile control after LN micro-optimizations (`ffn_latency_optimization_log_2026-03-08.md:48-59`): `full ≈16065`, `mha 7442`, `ffn_up 7340`, `ffn_down 6647`, `addnorm2 7170`, `addnorm1 12207`, `addnorm1_stats 12797`, `addnorm1_post 7740`.

### 3.2 How it is measured

**Timing.** Host-side wall clock around the XRT dispatch only. `iron:iron/common/aie_base.py:263-272`:

```python
start = time.perf_counter()
result = xrt_kernel(3, insts_bo, insts_len, *bos).wait()
stop = time.perf_counter()
...
return stop - start
```
(and the runlist variant at `:222-225`). Host→device and device→host BO syncs are **outside** the timed window; the device's own shim DMA of DDR is inside it.

Aggregation: `iron:iron/common/test_utils.py` `run_test(...)` — inputs written **once**, `warmup_iters` untimed runs, then `elapsed_total += operator.run_runlist()` over `timed_iters` and `elapsed = elapsed_total / timed_iters`. **Arithmetic mean**, not median or min. (Note: the working tree of `/home/cj/iron` is checked out to `devel`, whose `test_utils.py` adds an optional `return_timing_details` with min/max; the `extend_enc_pipeline` version does not have it. No encoder call site uses it on either branch.)

Counts: `warmup=3, timed=20` for both the profiler and the G1 stage-profile test (`iron@b5538a7^:.../test.py:22-23,38-39`). The branch-tip G3 test raised these to `warmup=10, timed=100` (`iron:iron/operators/encoder_pipeline/test.py:1186-1187`).

**A second, outer averaging layer exists on the pytest path only.** `iron:conftest.py::pytest_generate_tests` re-parametrizes every test `--iterations` times (**default 5**, `:29-34`), and a `CSVReporter` regex-scrapes the printed latency and emits `mean/median/min/max/stddev` per test into `tests_latest.csv` (`:78-95, 100-126`). So a pytest latency is mean-of-5-means-of-20. A `profile_debug_modes.py` latency is a single mean of 20.

**Isolated stages.** *Not tracing.* One **separately compiled binary per stage**. The selector is an integer `debug` kwarg folded into the artifact hash and the artifact stem (`iron@b5538a7^:.../op.py:225-227`: `f"d{self.debug}_m{self.mha_debug}_f{stage_tag}_n1{...}_n2{...}_{ln1_staging_design[:2]}_{digest}"`), and the AIE C kernel is itself rebuilt with `-DDEBUG={self.mha_debug}` (`:365,369`). Every stage measurement therefore requires a clean rebuild — the docs say so (`ffn_latency_optimization_log_2026-03-08.md:26`, `--clean-build` at `profile_debug_modes.py:171-175`).

Mode table (`iron:iron/operators/encoder_pipeline_memtile/debug_modes.py:6-53`, tuple = `(mha_debug, ffn_stage_only, an1_mode, an2_mode)`):

| `debug` | profile name | tuple | what runs |
|---|---|---|---|
| −1 | `full` | `(0, None, -1, -1)` | everything |
| 0 | `self_attn` | `(1, None, 0, 0)` | MHA `-DDEBUG=1` variant; both AddNorms pass input through |
| 1 | `mha_input` | `(-1, None, 0, 0)` | MHA compute off; AddNorms emit the LN-input operand |
| 2 | `residual` | `(-1, None, 1, 1)` | MHA compute off; AddNorms emit the residual operand |
| 3 | `ffn_up` | `(-1, 0, 0, 0)` | only FFN-up matmul + GELU |
| 4 | `ffn_down` | `(-1, 1, 0, 0)` | only FFN-down |
| 5 | `addnorm2` | `(-1, 2, 0, 0)` | only the LN2 tail |
| 6 | `mha` | `(0, 3, 0, 1)` | full MHA; LN1 math bypassed; FFN/LN2 compute off |
| 7 | **`addnorm1`** | `(0, 4, -1, 1)` | **full MHA *and* full AddNorm1**; FFN/LN2 compute off |
| 8 | `addnorm1_stats` | `(0, 5, -1, 1)` | LN1 statistics/normalize only |
| 9 | `addnorm1_post` | `(0, 6, -1, 1)` | LN1 mul/add only |

The design comment states the intent exactly (`encoder_pipeline_memtile/design.py:2635-2636` region): *"if this stage is not active, keep FIFO traffic/replay shape but bypass heavy LN statistics/math"* — **the dataflow graph is preserved; only arithmetic is neutralized.** That is the whole trick, and it is what makes the numbers comparable.

**The `max` is computed in code; the subtraction is not.** `iron@b5538a7^:.../profile_debug_modes.py:29-48` attaches a `contributes_to_bottleneck` flag to each mode — **`True` only for `ffn_up, ffn_down, addnorm2, mha, addnorm1, addnorm1_stats, addnorm1_post`**; `full`, `self_attn`, `mha_input`, `residual` are excluded. `_find_bottleneck` (`:119-132`) returns the argmax over that set and the CLI prints `bottleneck_stage: <name> (<us>)` (`:238`). **Nothing subtracts `full`.** The gap was typed into markdown by hand.

### 3.3 Two defects in the instrument you must fix before porting

**Defect 1 — `addnorm1` is not an isolated stage; it is a *prefix*.** `debug=7` keeps `mha_debug=0`, i.e. the whole MHA front-end computes. So the reported "max isolated stage `addnorm1`" already contains MHA. In the high-`pacc` table, `addnorm1 = 3514` and `mha = 1068`, so LN1's own marginal cost is ~2.4 ms, not 3.5. Every mode with `mha_debug ∈ {0,1}` shares this; only `ffn_up/ffn_down/addnorm2` have `mha_debug = -1`. **The stage latencies are neither additive nor disjoint.** **[INFERENCE]** Because the max is taken over a mixed set of prefixes and true single-stage variants, `full − max` is a *conservative under-estimate* of exposed serialization when the max happens to be a prefix — which it does in all three reported cases.

**Defect 2 — weight DDR traffic is elided in exactly the mode that wins the max.** `iron:iron/operators/encoder_pipeline_memtile/design.py:1910-1911`:

```python
need_bup_weights   = ffn_stage_only in (None, 0)
need_bdown_weights = ffn_stage_only in (None, 1)
```
with the runtime scheduler mirroring it at `:4288-4289`. `addnorm1` is `ffn_stage_only = 4`, so **both are False**: `B_Up` (768×3072) and `B_Down` (3072×768) are never fetched. At bf16 that is ~9.4 MB of DDR reads per invocation that the `full` build performs and the `addnorm1` build does not. **[INFERENCE]** A meaningful fraction of the reported 3.2–4.7 ms "exposed gap" is therefore FFN weight traffic, not pipeline imbalance. The doc's interpretation — *"still expose meaningful non-overlapped work above the stage bottleneck"* (`current_status.md:56-58`) — **over-reads its own instrument.** This is a good candidate for a dated retraction if the study cites it.

**A third, smaller oddity, flagged for honesty:** `addnorm1_stats` (12797) measures *slower* than `addnorm1` (12207) in the `2026-03-08` table, which should be impossible if stats ⊂ addnorm1. The doc notes the packetized stats path "does not reduce isolated `addnorm1_stats` time" (`ffn_latency_optimization_log_2026-03-08.md:65`) but does not remark on the ordering. And the same file's summary line — *"FFN-up remains the dominant stage in typical memtile profiling"* (`:20`) — **is contradicted by its own table 33 lines later** and by every other doc, all of which say `addnorm1`. **[UNSOURCED]**

### 3.4 Minimum viable port of the instrument

1. Compile **N+1 variants** of the same design, keyed on a stage enum folded into the artifact identity.
2. Variant *k*: full input staging, full output store, full FIFO/DMA graph; compute enabled only for stage *k*, all other stages bypass to a pass-through operand.
3. **Decide deliberately whether to elide weight DMAs for inactive stages.** iron elides them; that is what breaks the metric. Recommend: *do not* elide, so every variant issues identical DDR traffic and the gap is purely serialization.
4. Make each variant's max a *true* single stage (set the MHA-equivalent to bypass in the LN-only variants) so stages are comparable.
5. Time with the same host stopwatch discipline, inputs written once, mean over ≥20 after ≥3 warm-ups, **and record the full distribution** (mlir-air's own "compare distributions, not numbers" note applies directly here).
6. Emit `full`, `max_stage`, `gap`, `full/max_stage` **to a JSON/CSV artifact, always** — the original's `--output-json` defaulted to off and that is why none of iron's numbers has a file behind it.

---

## 4. `memtile` vs `ddr` staging — does "DDR beats on-chip" hold?

### 4.1 What the two modes are

- `memtile`: LN1 output is broadcast on-chip. `iron:iron/operators/encoder_pipeline_memtile/hooks.py:6-8` — `ln1_dram_stage_rows(...) → 0`; `:31-53` `build_ln1_to_ffn_up_path` returns one `outLNBroadcast` ObjectFifo with `effective_ffn_branches` consumers and empty `ln1OutStageToDDR` / `ln1InFromDDR`.
- `ddr`: LN1 output drains to DDR once from **one** source branch and is refilled and broadcast on-chip. `iron@b5538a7^:.../design_ln1_ddr.py:8-9` — `ln1_dram_stage_rows → profile_replay_groups * seq_tile`; `:197-243` builds `outLNBroadcast` → drain, plus `inLNFromDDR` → N consumers; `:97-119` logs *"LN1 DDR single-stream staging: source_branch=0 stage_col=%d (broadcast to %d FFN branches)"*.

### 4.2 The comparison is **not** like-for-like. Four independent confounds.

**Confound 1 — different topology.** The two cases differ in `emb_tile` (96 vs 128), `parallel_ffn` (4 vs 6), `proj_acc_depth` (8 vs 6) and `o_proj_acc_group_size` (4 vs 2). This is visible in the case names and stated explicitly at `debug_findings_2026-03-04.md:19-24`.

**Confound 2 — the like-for-like measurement exists as a *stated intent* and was never reported.** `debug_findings_2026-03-04.md:13-15` says both controls at the *same* topology `512seq/96e/4ph/4pffn/8pacc/4opg` were revalidated in both modes — and then reports the latency for **memtile only**. The DDR number at the control topology is **missing from every artifact in the repo.** Likewise the memtile number at the "fast DDR" topology is missing, even though that topology is in the memtile test matrix (`encoder_pipeline_memtile/cases.py:42-45`, `(512,64,12,3072,32,64,128,4,6,6)` with `opg` defaulting to 2 via `:52-59`). **Two runs would settle this and neither was recorded.**

**Confound 3 — different runtime schedule, not just different staging.** The DDR mode ships a *software-pipelined* runtime that the memtile mode does not have. `iron@b5538a7^:.../design_ln1_ddr.py:373-461` (`schedule_runtime_tap`) defers `tg_ln1_refill_and_weights` and only awaits it at the *start of the next* Q-block (`:401-406`), overlapping the DDR round-trip with the next block's compute. The memtile mode uses the default hook, which finishes the task group inline (`encoder_pipeline_memtile/design.py:200-211`). The modes also differ in `should_prefill_ffn_weights` (**DDR: False** at `design_ln1_ddr.py:365-366`; **memtile: True**, the default at `encoder_pipeline_memtile/design.py:190-194`) and in `adjust_wait_ffn_weight_fill` (DDR passes through, `:357-362`; memtile forces `True` when broadcasting to >1 branch, `design.py:181-189`).

**Confound 4 — different resource envelope, therefore different reachable topologies.** DDR costs `+1` shim output stream (`resource_utilization_2026-03-04.md:13-15`) but frees the memtile output channels that the on-chip broadcast consumes. The consequence is recorded: the 16-head/4096-ffn/128-embtile family is marked **`ddr` only** (`resource_utilization_2026-03-04.md:45`), and the test matrix hard-codes the same exclusion (`encoder_pipeline_memtile/cases.py:215-228`, `_DDR_ONLY_REGULAR_CASES`).

### 4.3 What the artifacts *do* support

**Supported:** In one uncontrolled pairing at 512 seq, a DDR-staged run of a *wider* topology (128e/6pffn) measured 12226.31 µs against an on-chip-staged run of a *narrower* topology (96e/4pffn) at 16488.12 µs.

**Not supported:** that staging an intermediate through DRAM is faster than keeping it on-chip at equal topology and equal schedule. No artifact in the repo tests that.

**[INFERENCE], and the more likely mechanism:** DDR staging is a **resource-relief move, not a bandwidth move**. Pushing one intermediate out through a single shim stream removes the on-chip broadcast fan-out from the tail memtiles, which is exactly the pressure point the docs name (`current_status.md:71-75`, `optimization_plan.md:66-76`). That relief lets a *wider, more parallel* topology become placeable, and the extra parallelism is what pays. The observed win is consistent with "DDR unlocked 6 FFN branches at `emb_tile=128`", not with "DRAM is cheaper than L2".

### 4.4 The strongest evidence available — and it is *revealed preference*, not an A/B

The live, benchmarked G3 operator **stages LN1 through DDR unconditionally, on both paths**, and dropped the on-chip option entirely: non-seq at `iron:.../design.py:6683-6717`, seq-par at `:5067-5080` and `:5200-5241`; `ln1_staging_design` does not appear anywhere in G3 (`design.py`, `op.py`, `placements.py`, `topology.py`); `encoder_pipeline_ddr/` was the survivor of the split in the code lineage while `encoder_pipeline_memtile/` was retained only as a focused experiment package (`encoder_pipeline_memtile/README.md:1-3` — *"Focused memtile-mode entrypoints for encoder pipeline experimentation"*); and `mode_split_plan.md:315-328` directs the BERT application to `AIEEncoderPipelineDDR` *"because the shared operator defaulted to DDR staging"* (confirmed by `encoder_pipeline_memtile/op.py:170`, `raw = ln1_staging_design if ... else "ddr"`).

**[INFERENCE]** After running both modes across a 40+ case matrix for months, the operator's owner shipped DDR staging as the only path. That is a strong signal. It is *not* a controlled comparison, and it should be reported as what it is.

### 4.5 What this means for `fused`'s founding premise — stated carefully

The honest reading is narrower than "DRAM traffic doesn't matter", and more interesting:

**Eliminating a DRAM crossing is not free; it is paid for in on-chip channel and memtile-BD budget, and that budget is the binding constraint on how much parallelism the layer can carry.** When the on-chip route costs enough tail-memtile fan-out to force a narrower topology, routing one intermediate through DRAM can be the better trade. Iron's fused design keeps **eight of nine** interior hand-offs on-chip (§1.3) and pays DRAM for exactly one — the one whose consumer count is largest (broadcast to every FFN branch, plus two residual replays for the two-pass LN2). **[INFERENCE]** The rule that generalizes is *not* "DRAM is fine" but **"the crossings worth eliminating are the point-to-point ones; a high-fanout broadcast may be cheaper through DRAM"**.

---

## 5. Failure modes, and how they map onto R1's walls

### 5.1 iron's named pressure points

**[MEASURED]** `current_status.md:69-76`:
1. accumulation-core L1 pressure on **O-proj accumulation cores** (dominant resident buffer family `memOW*_cons`, `design_optimization_plan.md:54-67`)
2. accumulation-core L1 pressure on **FFN-down stage/root cores** (`memBDown*_cons`, `:69-81`)
3. **tail memtile output-channel pressure** on the aggressive `64q/48e/16pacc` families (`:83-95`)
4. **final `memLN2` shim-drain pressure** in the `64embtile/12pacc` experiments

Plus, from `debug_findings_2026-03-04.md`:
5. runtime timeout `ERT_CMD_STATE_TIMEOUT` on `4pheads_6pffn_8pacc` after compile constraints were removed (`:54`)
6. a **split-weight TAP packing order** bug: the split B-weight tap order did not match `ObjectFifo.split(...)` consumer ordering; mismatches went `32702 → 7` when fixed (`:38-41`)
7. a **grouped FFN-down accumulation ordering** bug: grouping broke the forward-FIFO `curr_acc → new_acc` order (`:43-46`)

### 5.2 The shim-drain wall — same family as R1's wall 4, and *strictly harder*

**[MEASURED]** `iron:.../encoder_pipeline_archive/docs/repros/shim_drain_bd_repro/README.md`. Two frozen repros, both `aiecc` failures in resource allocation:

```
'aie.dma_bd' op Allocator exhausted available BD IDs (maximum 24 available for channel 0)   # 1ph/1pffn/12pacc  (:41)
'aie.dma_bd' op Allocator exhausted available BD IDs (maximum 24 available for channel 3)   # 4ph/4pffn/12pacc/4opg (:86)
```
The exhausted object is the **final LN2 output drain**, `aie.objectfifo @memLN2` with its runtime-generated `aiex.dma_configure_task_for @memLN2` tasks (`:44-56`). Frozen MLIR at all three stages (source, `row_store_lowered`, `aiecc_failure`) is checked in.

**Two findings that transfer directly:**

**(a) The limit is BD *count per channel*, not descriptor shape.** `:136-143`: *"simplifying `memLN2` from the expanded vectorized `dimensionsToStream` form to a direct row-major drain did not change the allocator boundary ... the remaining pressure is the runtime output tap count on the final `32x64` LN2 output stream, not the `dimensionsToStream` form."* **[INFERENCE]** If mlir-air's R1 wall 4 was addressed by reshaping descriptors, that fix is on the wrong axis; the driver is *how many* runtime taps the output stream is split into.

**(b) The output drain is the last thing to break.** In iron, every other BD pressure point was eventually placed around; the tail drain was not. `:54-56`: *"the remaining exposed limit is the final output drain path."*

**Mapping to R1's wall 4 (shim BD exhaustion): same family, same allocator, and iron never solved it** — it retreated to a narrower topology envelope. `resource_utilization_2026-03-04.md:40-45` lists the "narrowed stable envelope"; the `12pacc` family is simply excluded. **[INFERENCE]** If R1 is fighting shim BD exhaustion at the aggressive rung, iron's evidence says the productive response is to shrink the tap count of the output stream or shrink the topology — not to keep tuning the descriptor form.

### 5.3 The wait-relaxation failure — **is it R1's race? Probably yes, but iron does not have the evidence to say so, and its correctness gate could not have detected it.**

The claim, in full. `current_status.md:64-67`:

> a simple runtime relaxation of per-head `K/V/W_O` waits was rejected because it broke correctness, so overlap work now needs to focus on safer structural changes rather than blindly removing waits

Corroborated at `optimization_plan.md:36-38`: *"keep the current DDR-only residual-fill wait relaxation; avoid naive wait removal on `K/V/W_O` head-block fills; that path was tested and broke correctness on the memtile control case"*, and listed as covered-and-rejected at `PERFORMANCE_IMPROVEMENT_PLAN.md:165-166` (*"naive tail `wait=True/False` relaxations"*, *"simple fill-order shuffles for `K/V/W_O`"*).

**The mechanism is structurally a data race.** The `wait` in question is the per-fill flag on `rt.fill(fifo.prod(), BUF, tap=..., placement=..., task_group=tg, wait=True)` — the runtime's completion wait on a shim DMA before the dependent consumer proceeds. The K/V/W_O head-block fills are at `iron:.../design.py:6540-6618` (Q, per-`k_tap` K, per-`v_tap` V, then `inOW` for W_O, all inside one `tg_head` task group finished at `:6618`); note W_O already carries a *conditional* wait, `wait=not use_staged_hidden_state_kv_cache` (`:6617`), i.e. targeted relaxation is in the tree while blanket relaxation is not. Removing it lets a core's consumer read an ObjectFifo buffer the shim DMA has not finished writing. That is a read-before-write hazard whose manifestation is timing-dependent — **the same class as R1's wall 7.** **[INFERENCE]**

**But here are three reasons not to declare them the same failure:**

1. **iron never characterizes it as intermittent.** No iron doc contains the words flaky, intermittent, nondeterministic, sporadic, or alternating (grep over all encoder docs). It is described as *"broke correctness"* on a specific control case — reported as reproducible, unlike R1's alternating PASS/TIMEOUT. **[MEASURED]** (absence)
2. **iron's correctness gate is a mismatch *budget*, not equality.** `iron:iron/operators/encoder_pipeline_memtile/cases.py:18-20,155-161`: `REL_TOL=4e-2`, `ABS_TOL=1.5e-1`, `ERROR_THRESHOLD=0.005`, and `max_errors = int(seq_len * d * heads * 0.005)` — for 512seq/64d/12heads that is **1966 mismatching elements allowed out of 393216 (0.5%)**. The comparator is `nearly_equal` with `norm = |a|+|b|` (`iron:iron/common/test_utils.py:11-29`), i.e. `rel_tol=0.04` is roughly 8% of a single operand. G3 **loosened this tenfold** to `ERROR_THRESHOLD=0.05` (`iron:iron/operators/encoder_pipeline/test.py:38-40`). Under such a gate, a race that produces slightly-stale-but-plausible values in a small fraction of a tile would pass silently. **The recorded pass criterion for the accepted DDR wait relaxation was literally "without changing mismatch counts"** (`current_status.md:60-62`) — i.e. it was accepted on the basis that a soft budget did not move.
3. **iron's only recorded nondeterminism is a timeout, not a wrong answer.** `ERT_CMD_STATE_TIMEOUT` appears at `debug_findings_2026-03-04.md:54` and `PERFORMANCE_IMPROVEMENT_PLAN.md:209,255,182`. The plan's non-goals include *"retry-based masking of runtime instability"* and *"Do not rely on retries for performance work. Timeouts must be treated as real failures."* (`:120-122, 507-510, 539`) — **[INFERENCE]** language that only gets written if retries were masking something, i.e. iron did see nondeterministic hangs. But no artifact ties a hang to the wait relaxation.

**Verdict:** the *mechanism* (dropped completion wait on a shim fill → consumer reads unwritten buffer) is the same hazard class as R1's wall 7, and iron independently found that this class of relaxation is not safely removable "blindly". That is genuine, useful corroboration. But it is **not** the same observation as R1's four-distinct-wrong-answers-plus-alternating-timeout, and iron's gate is too loose to have detected R1-class corruption even if present. Do not claim "iron hit the same race"; claim **"iron independently found that removing per-fill completion waits on the head-block ingress breaks correctness, and concluded overlap must come from structural change instead"**.

### 5.4 What iron did instead — the safe overlap pattern

Since blind wait removal failed, iron's *retained* overlap mechanism is **task-group deferral**: keep `wait=True` on every fill, but do not `finish_task_group` until the next iteration needs it (`iron@b5538a7^:.../design_ln1_ddr.py:401-406, 448-461`; carried into G3 at `iron:.../design.py:6743-6744, 6746-6758` via `pending_ln1_refill_tg` / `pending_output_tap_idx`). This preserves the ordering guarantee while moving the *wait point* one iteration later. **[INFERENCE] This is the single most directly portable idea in §5** — it is exactly the "safer structural change" the doc asks for, and it is a schedule transformation, not a synchronization removal.

Also retained: `wait=effective_ffn_branches > 1` on the B_Up/B_Down fills (`iron:.../design.py:6728, 6739`) — a *targeted*, condition-guarded relaxation rather than a blanket one.

---

## 6. Transferability — what is design, what is existence proof, what is neither

### 6.1 Transfers as **design** (portable ideas, restatable in AIR/MLIR terms)

| Idea | iron artifact | Why it ports |
|---|---|---|
| **One program, one dispatch, whole layer, N buffer objects** | `op.py:1313-1376`; `design.py:621-623` | Pure architecture. The mlir-air equivalent is "one segment must own all three herds"; iron confirms the shape is right and that the whole encoder fits in one config. |
| **Stage-truncated binaries as a balance instrument** | `debug_modes.py:23-53`; `profile_debug_modes.py:36-48,119-132` | Compiler-agnostic. Needs only a per-stage compute-bypass switch folded into the artifact identity. **Fix the two defects in §3.3 before using it.** |
| **Task-group deferral instead of wait removal** | `design_ln1_ddr.py:401-461`; `design.py:6743-6758` | A schedule transformation on the host-side descriptor program. AIR has task/dependency structure that can express "await the previous iteration's fill at the top of this one". |
| **1% latency band + prefer-more-tiles tie-break in autotune** | `npu_inference.py::select_autotune_topology` | Directly addresses mlir-air's "compare distributions, not numbers" lesson. Costs nothing to adopt. |
| **Thermal cooldown gate between benchmark candidates** | `benchmark_common.py::cooldown_before_benchmark` | Directly relevant given mlir-air's NPU-pmode incident. |
| **Per-shape topology cache with a signature that invalidates on any shape/tiling change** | `topology.py:508-542` | Prevents stale-winner reuse across shape changes. |
| **Route the highest-fanout intermediate through DRAM, keep point-to-point on-chip** | §1.3 + §4.5 | A design heuristic, honestly caveated. |
| **Replicate the whole pipeline per sequence lane rather than balancing one copy** | `placements.py:1214-1310` | The empirical winner. Applies to any row-wise-decomposable block. |
| **Neighbour-only reduction chains with an explicit legality check** | `encoder_pipeline_memtile/design.py:296-357` | The FFN-down accumulator ring in R1 is the same object; iron's rule (chain of N/E/S neighbours terminating adjacent to the consumer, no westward edges) is a placement invariant worth stating in AIR. |

### 6.2 Transfers only as **existence proof**

- **A fully resident `MHA + AddNorm1 + FFN + AddNorm2` runs correctly on NPU2 in one configuration.** That is the headline: the thing R1 is trying to prove is possible *is* possible on this silicon. But the passing evidence is `100 passed, 40 skipped` / `115 passed, 40 skipped` at `ERROR_THRESHOLD=0.005` (§5.3.2) and the numbers behind it have no file (§0.3).
- **Both LN1 staging modes were made to work**, so neither is a hard blocker.
- **The high-`pacc` case reaches `full/max_stage = 1.238`** — i.e. a fused layer can get within ~24% of its bottleneck stage. Useful as a target, not as a method.
- **`aiecc`'s BD allocator is a real ceiling that a mature design also hit and did not defeat** (§5.2).

### 6.3 Does **not** transfer

- **`placements.py` in its entirety.** 1540 lines of hardcoded `(col,row)` tuples for an 8×4 NPU2 grid under IRON's `Tile`/`ObjectFifo` placement model. There is no AIR construct these map onto. **A reader must not mistake an IRON placement for an AIR builder change.**
- **All the `choose_*_mem_tile_col` / `adjust_*_cols` hooks** (`hooks.py`, `design_ln1_ddr.py:12-56, 245-309`). These are hand-fitted `if parallel_heads >= 6 and proj_acc_depth >= 6: return 4` rules for one specific device and one specific mlir-aie wheel. They are a record of *where* pressure appears, not a transferable algorithm.
- **The exact latency figures.** Different toolchain, different lowering, and pinned to one wheel: `iron:iron/operators/encoder_pipeline/README.md:36-44` — *"`encoder_pipeline` should be used with the specific `mlir_aie` wheel currently validated in `ironenv`: `mlir_aie==0.0.1.2026031811+71fb44f147` ... Different wheels can change DMA-BD allocation outcomes, compile viability of some topologies, runtime behavior."* This is iron's own version of mlir-air's toolchain-layer-staleness note.
- **The mismatch-budget correctness gate.** `0.5%` (G2) / `5%` (G3) of output elements allowed to be arbitrarily wrong is far looser than the mlir-air study's standard and would mask exactly the class of defect R1 is chasing. **Do not port this. Cite it as a caveat on iron's own PASS results.**
- **`nearly_equal`'s `norm = |a| + |b|`** — a nonstandard comparator roughly 2× looser than `np.isclose` at the same `rel_tol`.

---

## 7. Practical appendix

### 7.1 Recovering the deleted files (read-only)

```bash
cd /home/cj/iron
git show b5538a7^:operators/encoder_pipeline/profile_debug_modes.py   # 265 lines — the balance instrument
git show b5538a7^:operators/encoder_pipeline/test.py                  # 370 lines — stage_profile harness
git show b5538a7^:operators/encoder_pipeline/design_ln1_ddr.py        # 482 lines — the DDR staging mode
git show b5538a7^:operators/encoder_pipeline/design_ln1_memtile.py    # 102 lines
git show b5538a7^:operators/encoder_pipeline/op.py                    # 595 lines — artifact-stem/debug plumbing
git show b5538a7^:operators/encoder_pipeline/reference.py             # 202 lines — debug-mode-aware golden ref
git show 0fe8e8c:operators/encoder_pipeline/README.md                 # lines 111,126-127: the 100/40 and 115/40 counts
```
`b5538a7` and `a1327c9` ("Archive and remove shared encoder_pipeline package") are reachable only from `origin/update_bert_rebased`. `0fe8e8c` = "Use double-buffered memtile row-store for encoder LN1 replay".

Working copies of all extracted files are in this session's `scratchpad/iron-ref-private/` (prefix `DELETED__` for recovered ones); line numbers in this document match those copies exactly.

### 7.2 Reconciling `100 passed, 40 skipped` / `115 passed, 40 skipped`

Fully accounted for, which also dates the claim. Source: `iron@0fe8e8c:operators/encoder_pipeline/README.md:126-127`.

- 23 base cases (`iron@b5538a7^:.../test.py:52-77,163`), of which the **last 3 are DDR-only** (`:166-168`, comment at `:73`) → **20 memtile params, 23 ddr params**.
- 8 stage-profile modes × 1 hardcoded case × 2 designs = 16 params, split 8/8 by the `lnstage_` ID prefix (`:291-311`), **all skipped** by a source-level flag: `ENABLE_STAGE_PROFILE_TESTS = False` (`:26`) → `pytest.skip(...)` (`:348-349`).
- `conftest.py --iterations` default **5** multiplies everything (`iron:conftest.py:29-34, 169-175`).

⇒ `-k lnstage_memtile`: 20×5 = **100 passed**, 8×5 = **40 skipped**. `-k lnstage_ddr`: 23×5 = **115 passed**, 8×5 = **40 skipped**. Exact match.

**Consequence: those counts predate the high-`pacc` comparison-case expansion.** The matrix on the branch today is 42 cases (23 base + 19 evenly-partitioned high-`pacc` variants), of which 36 are memtile-eligible (`iron:iron/operators/encoder_pipeline_memtile/cases.py:208-228`). **`current_status.md:23-25` presents these counts as "Current validated state"; they describe a matrix ~40% smaller than the one that exists. [UNSOURCED] as a current claim.**

### 7.3 Known-broken cases the docs admit

`iron:iron/operators/encoder_pipeline_memtile/README.md:20-31` — after the row-store path was removed, two numerical regressions and one compile regression remain in the FIFO-only path:
- `encoder_64seq_..._96embtile_1pheads_1pffn_8pacc_1opg`
- `encoder_64seq_..._96embtile_4pheads_4pffn_8pacc_4opg` ← **this is the memtile control case, i.e. the topology that produced the 16488.12 µs headline number**
- `encoder_64seq_..._48embtile_6pheads_2pffn_16pacc_2opg` (compile-time memtile DMA pressure)

**[INFERENCE]** The `4pheads_4pffn_8pacc_4opg` control is listed as a live numerical regression in the memtile package on this branch. The 16488.12 µs figure was taken before the row-store removal (G1 era). Treat the memtile control number as historical, not as a currently reproducible measurement.

### 7.4 One-line file map

| File | Lines | What it is |
|---|---|---|
| `iron/operators/encoder_pipeline/design.py` | 6764 | G3 worker graph, ObjectFifos, runtime schedule. Two disjoint branches: seq-par `:2400-5261`, non-seq `:5263-6760` |
| `iron/operators/encoder_pipeline/op.py` | 1829 | operator surface, artifact naming/hashing, XRT buffers + runlist |
| `iron/operators/encoder_pipeline/placements.py` | 1540 | hardcoded `(col,row)` registry; `LOW_HEAD_TAILS:86-152`, `TOPOLOGY_PLACEMENTS:155`, `SUPPORTED_...:1538` |
| `iron/operators/encoder_pipeline/topology.py` | 833 | knob dataclass, keys/IDs, cache signature, variant expansion. **No search logic** |
| `iron/operators/encoder_pipeline/test.py` | 1330 | G3 pytest matrix; `warmup=10, timed=100`; `ERROR_THRESHOLD=0.05` |
| `iron/operators/encoder_pipeline_memtile/design.py` | 4674 | G2 memtile design; contains the auto-prune legality loop `:489-565` and the debug/stage gating |
| `iron/operators/encoder_pipeline_memtile/hooks.py` | 97 | the on-chip-broadcast policy surface |
| `iron/operators/encoder_pipeline_memtile/debug_modes.py` | 65 | **the stage-mode table** |
| `iron/operators/encoder_pipeline_memtile/cases.py` | 241 | G2 test matrix; `ERROR_THRESHOLD=0.005`; `warmup=3, timed=20` |
| `iron/applications/bert/npu_inference.py` | — | `--topology-policy`, `autotune_topology`, `select_autotune_topology` |
| `iron/applications/bert/BENCHMARKING_METHODOLOGY.md` | — | the three-NPU-path taxonomy and measurement contract |
| `iron/common/aie_base.py` | — | `run_runlist` / `run_kernel_once`, the `perf_counter` stopwatch |
| `iron/common/test_utils.py` | — | `run_test`, `nearly_equal`, `verify_buffer` |
| `conftest.py` | — | `--iterations` (default 5), `CSVReporter` mean/median/min/max/stddev → `tests_latest.csv` |
