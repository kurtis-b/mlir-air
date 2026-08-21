# 55 — What llama.cpp's Hexagon backend does at the system level, and what of it transfers to XDNA2

`[2026-08-20]` Three questions from the operator, answered in order: what the iron adapter is
for (§1); whether this framework's four modes optimize inference per workload (§2 — **no**,
with the evidence); and what Qualcomm's Hexagon NPU and llama.cpp's `ggml-hexagon` backend do
that this framework should learn from (§3–§6). The last was researched two ways — Codex with
live web access ([55a](55a-codex-hexagon-research-report.md), verbatim) and a direct reading of
the upstream source at `ggml-org/llama.cpp@6503355df0eb` (master, 2026-08-20). A second Codex
pass then audited this document against source and the repository and corrected it in the
places marked `[per Codex review]` (its report: [56 §7](56-full-model-mixed-precision-study-plan.md)).
Where a number below is arithmetic rather than a measurement it says so.

## 0. Provenance, and two corrections to the Codex report

Codex prompt: research Hexagon architecture (§1 of its report), `ggml-hexagon` (§2), the
earlier QNN/Genie paths (§3), and system-level lessons mapped to XDNA2 (§4). It had web access.
Source files read directly for this document, with line numbers at `6503355df0eb`:
`ggml/src/ggml-hexagon/ggml-hexagon.cpp` (options 60–90; op batch `fit_op` 1325; enqueue /
flush 1596–1632; HMX eligibility ~2300; matmul planner 2380–2560; `try_fuse_node` 3562;
`graph_compute` 3655–3720; `graph_optimize_reorder` 3733), `htp/main.c` (VTCM acquire
231–296; power 424–470), `htp/htp-ctx.h`, `htp/htp-ops.h` (106–117, 161), `htp/htp_iface.idl`,
`htp/matmul-ops.h` (16–27), `htp/matmul-ops.c` (1360–1545), and
`docs/backend/snapdragon/README.md`.

Two places where 55a is corrected here:

1. **Its §4 opening quotes this study's dispatch vectors from doc 08** (`6/5/4/1` submissions,
   `6/391/131/3` runlist entries, `19/403/402/19` sync boundaries). Those are Phase E numbers
   from before the 2026-08-09 mode corrections. The standing vectors are [54 §1](54-first-full-profile-and-decoder-families.md):
   `offload` 30 submissions / 90 / 90, `fused` 1 / 23 / 13, `coarse` 4 submissions, `runlist`
   17 submissions with `herd_launches` growing 151 → 873 over 512 → 8192.
2. **Its §4 "per-workload tuning" verdict says the registry is "the stronger foundation"** and
   recommends extending its keys. §2 below shows the registry is a once-measured exact-shape
   table with `KeyError` on a miss and that the one attempt at an analytical selector closed as
   a negative finding; the transferable Hexagon lesson is narrower and more specific than
   "add keys" (§5, lesson 4).

## 1. What the iron adapter is for

`study/iron_adapter.py` is the mechanism behind success criterion 3 of
[00](00-context-and-goals.md) ("comparable to iron's result trees through an explicit
adapter"), and it is **deliberately a shape-only join**. It reads exactly one file of an iron
results tree — `<root>/end_to_end/results_all_power.csv` — translates iron's family id
(`baseline_768`) into this port's `study_case_id` (`512x768_encoder_bert`) through
`cases.FAMILY_SPECS`, refusing an unknown family or a row whose own width contradicts it, and
then `validate_port` checks that for every `(case, mode, seq)` point both sides measured, the
**seven shape fields agree**. It copies 20 "safe" columns (identity, shape, dtype, bias,
weights source, the selected-config JSON) and refuses four classes by name with the reason
recorded in `_INCOMPARABLE`: **latency** (iron's `timed_total_sec` is built two different ways,
neither this schema's region), the **power block** (a filtered mean over a dynamic sample
window), **dispatch counts** (one iron scalar against this study's six-field vector), and
**`run_status`** (iron's gate is `rel 0.1 / abs 0.5` with a 5% mismatch budget; this port's is
zero mismatches at registry tolerance). `execution_mode` crosses unchanged — iron's `hybrid` is
this port's `coarse` by convention 7 — and `fused` has no iron counterpart.

What it guarantees: that this port measures the layers iron measured (four roots validated,
0 disagreements, [54 §5](54-first-full-profile-and-decoder-families.md)). What it refuses to
say: anything about the 2.7–9× latency gap the join puts side by side — a test pins that
`validate_port` passes with 0.001 ms against 9000 ms and never prints the word "latency". It
is the instrument that makes "same layer" a checked claim so that, when the latency gap is
attributed (work-queue item 1), the comparison is over identical shapes by construction.

## 2. Do the four modes optimize per workload? No — and here is exactly what is and is not chosen

**Verdict: the modes are operator-selected study conditions. There is no dispatcher that maps a
workload (model, width, sequence length, context length, quantization, batch) to a mode, and
nothing inside a mode is optimized per workload at run time.** The evidence, per axis:

| Axis | What chooses it today | Kind |
|---|---|---|
| Execution mode | `run_mode.py --mode` (default `coarse`); `run_ladder.py --modes` walks all four; `profiles.py` `PROFILE_MODES` is the constant 4-tuple and the three profiles differ only in `seqs`; the Makefile has one target per mode | **manual** |
| Workload → mode logic | `profiles.skip_reason(mode, seq, family)` — structural refusal only (`fused` packing cap, FA `parallel_seq` floor, attention-GEMM tile multiple, softmax L1 width, `UNBUILDABLE_VARIANTS`), each derived from the refusing builder's source by `ast` | **refusal, not selection** |
| GEMM tiles / method / herd | `gemm_config(M,K,N,…)` = exact-shape lookup in a once-measured sweep; a missing shape **raises `KeyError`** ("run a sweep") — no fallback, no interpolation | **table, measured once** |
| Attention `parallel_seq` | caller default 256, validated for divisibility, never derived | **constant** |
| Softmax / layer-norm `rows_per_call` | `derive_rows_per_call`: largest legal value **at or below** the historical constant (divides the rows, tile fits 64 KiB) | **legality derivation, downward only** |
| Decoder tolerance | `decoder_stage_atol(hidden)`: base table + a measured per-width override (`1024 → 6e-1`) | **table, measured** |
| Quantization | study is bf16 only (`run_mode.py` stamps `dtype="bf16"`; the seven `quant_*` schema columns are deliberately empty); int4 exists only in `llms/llama32_1b_int4`, prefill only, 8× slower than bf16 per its README | **none in the study** |
| Model in `llms/` | kernel sequence and multi-launch merge hand-forked per model (`head_dim` 64 vs 128 picks seq-first vs head-first FA; hidden alignment picks 3-ELF vs split-FFN form); tiles from the registry per shape | **manual per model** |
| Context length | `build_session` pins `seq_len = 2048`, compiles prefill at that one length, and **pads the prompt with EOS up to it**; KV cache is fixed-size | **fixed + padding** |
| Batch / prefill-vs-decode | `llms/` has a GEMM prefill path and a GEMV decode path, chosen by the driver, not by M | **manual** |
| A selector | work-queue row 31's `study/analytical_cost.py` prices FFN-tail mappings on the declaration; its `rank()` is restricted to one dispatch structure (the recorded counter-example: byte order `runlist < fused < coarse < offload` against latency order `fused < coarse < runlist < offload`), and the 2026-08-19 tier row records it **closed as a negative finding** — at fixed seam scope every axis separating two buildable designs prices as a tie | **does not exist; the attempt closed negative** |

So the honest description is: a measurement harness that walks a fixed matrix of (mode × length
× family) and refuses, by derived reason, the cells it cannot build; plus a deployment path that
is configured by hand per model over a measured tile table. That is the right shape for a
*study* — the whole point of [03](03-measurement-model.md)'s axis is to hold the mode fixed and
measure — but it is not a per-workload optimizer, and §5 says what the nearest useful one would
look like.

## 3. The Hexagon NPU, in the terms that matter here

Facts with a primary source; generation-specific values flagged where Qualcomm does not publish
them (55a §1 has the citations).

- **Three engines, one shared address space.** A 4-wide VLIW scalar core with SMT hardware
  threads (product-specific; `ggml-hexagon` observes 4 HVX contexts on v75 and 8 on v81, caps at
  `HTP_MAX_NTHREADS 10`); **HVX**, 1024-bit vector units, several contexts; **HMX**, a 32×32-tile
  matrix engine (`HTP_MM_HMX_TILE_N_ROWS/COLS 32`, fp16 accumulate in llama.cpp's use), v73+.
  Peak HMX throughput and clock are **not published**; platform "TOPS" (45 on X Elite) is not an
  HMX number.
- **VTCM**: software-managed on-chip SRAM, **8 MiB** on v69/v73/v75/v81 (compiler defaults and
  `ggml-hexagon` logs; v79 inferred), 4 MiB on v68. Not a cache — the application acquires it
  through `HAP_compute_res_*` (`htp/main.c:261–296`, single page, HMX flagged in the same
  request, a release callback so another stack can borrow it) and decides what lives there.
  HMX and scatter/gather operate **only on VTCM**, never on DDR. This is the analog of one flat
  XDNA2 L2 — except XDNA2's 4 MiB of L2 is **eight 512 KiB memtiles, one per column**, and its
  64 KiB L1s are per tile.
- **DDR**: shared with the CPU/GPU through distinct mappings and cache domains; buffers are
  `rpcmem` (ION / DMA-BUF) allocations `fastrpc_mmap`'d into the DSP process once
  (`ggml-hexagon.cpp:329, 359–376`); ownership transitions need explicit clean/invalidate
  (dirty ranges, `htp-ctx.h HTP_MAX_DIRTY_RANGES 16`). Theoretical peaks ~67–85 GB/s on a
  64-bit mobile bus (8 Gen 2 → 8 Elite, inferred from published LPDDR5X rates), 136 GB/s on X
  Elite's 128-bit bus. For comparison the HX 370 here is LPDDR5X-7500 on 128 bits, ~120 GB/s
  theoretical — but what matters is what the NPU's DMA path can ingest, see §5.
- **Control plane**: FastRPC — an IDL-generated stub/skeleton pair; the whole `ggml-hexagon`
  interface is **seven calls** (`htp_iface.idl`: `start/stop/mmap/munmap/profiler/etm/hwinfo`).
  Work never goes through FastRPC; it goes through a **`dspqueue`** shared-memory queue created
  at `start`. One process domain has a VA mapping limit (~2–3.5 GiB: `HTP_MMAP_MAX_VMEM` 2 GiB,
  `HTP_OP_MAX_VMEM_DEFAULT` 3.2 GB, probed at init by allocating until failure), which is the
  entire reason for `GGML_HEXAGON_NDEV`.
- **Power**: the DSP-side service votes at start (`htp/main.c:424–470`): compute client class,
  **DCVS disabled**, core and bus corners pinned `MAX`, sleep disabled, HVX powered up, and
  since 2026-04 the HMX clock bumped to its max corner (PR #22334). That is the Hexagon form
  of this study's Turbo rule — except they do it **programmatically from inside the session**,
  and this machine needs `sudo xrt-smi configure --pmode turbo` and a runner that refuses.

## 4. How `ggml-hexagon` actually runs a token — the mechanisms, from source

Merged 2025-10-22 (PR #16547, HVX only, Q4_0/Q8_0/MXFP4/F32), then in order: op-queue and
dispatch rework (#16820), HMX matmul (#20693, 2026-03-19), DMA pipelining and flash attention
on HMX (#20118, #22347), op fusion (#23835, 2026-05-28). The current shape:

1. **One persistent program, generic kernels.** The DSP side is one shared object with every
   op as a C function over HVX/HMX intrinsics (`htp-ctx.h` at this sha lists 26: matmul, matmul_id,
   fused `matmul_qkv` / `matmul_ffn`, binary, unary, rms/l2 norm, softmax, rope, flash_attn,
   get/set_rows, cpy, argsort, ssm_conv, gated_delta_net, …). Kernels loop over shapes; nothing
   is compiled per shape. **This is the structural opposite of XDNA2**, where every kernel is a
   per-shape placed-and-routed artifact and the array has no resident runtime.
2. **Ops are batched, not dispatched.** `graph_compute` walks the ggml graph once, fuses
   (`try_fuse_node`: the three QKV matmuls sharing an input → `MUL_MAT_QKV`, gate/up →
   `MUL_MAT_FFN`, `MUL_MAT+ADD`, `RMS_NORM+MUL` — the quantized QKV/FFN and matmul+add
   fusions **guarded by "VTCM needed ≤ budget"**; `RMS_NORM+MUL` is not `[per Codex review]`),
   precomputes every op's kernel parameters, and **caches the whole plan by graph uid** so the
   next token with the same shape skips planning. Then `enqueue_op` packs descriptors into a
   batch — bounded by **`opt_opbatch` (1024) ops, `HTP_OP_MAX_BUFS` (16) distinct buffers,
   `min(n_ops·(1+MAX_INPUTS), HTP_OP_MAX_TENSORS)` tensor descriptors, and the session's vmem**
   (`fit_op`; it does not test VTCM — the per-op planners do) — and `flush_batch` does **one
   `dspqueue_write` per batch**, with up to 16 batches in flight. A 16-layer decoder at a few
   hundred ops *can* be one or two writes per token; buffer and descriptor reuse decide it
   `[per Codex review]`. The DSP walks the
   batch and posts one response (with per-op cycle counters and PMU events if profiling).
3. **Weights are resident and repacked once.** Model tensors live in `rpcmem` buffers mapped
   into the DSP VA at load; quantized weights (Q4_0/Q4_1/Q8_0/IQ4_NL/MXFP4) are repacked in
   `buffer_set_tensor` into padded 32×32 tiles (`HTP_MM_WEIGHT_TILE_SIZE_Q4_0 576` bytes per
   tile, scales separated from quants) so DMA bursts and HMX tiles line up; F16/F32 weights are
   copied unchanged. "Resident" means mapped shared DDR — VTCM holds only the tiles in flight.
   Nothing is re-uploaded per token. The KV cache can live in HTP buffers but its dtype is a
   run-time choice: the official example uses **Q8_0 K/V**, while `FLASH_ATTN_EXT` requires
   **F16 K/V**, so that configuration implies a CPU attention split `[per Codex review]`.
4. **VTCM is planned analytically per op, on the host, by capacity.** The matmul planner
   (`precompute_matmul_params`, 2380–2560) is a **legality cascade with one greedy knob**: HMX
   if `M > 4` rows (`HTP_MM_HMX_MIN_NROWS 4`) and K%32, N%32, contiguous; else HVX tiled; else
   HVX flat; else CPU (`GGML_HEXAGON_MM_SELECT` orders the cascade). For each candidate it
   builds the VTCM layout (weight-tile ring × prefetch depth, the quantized activation block,
   the output slab, per thread) and takes the **largest prefetch depth 16 → 2 that fits the
   budget**. The HMX branch goes one step further (`htp/matmul-ops.h:712` `solve_2d_params` →
   `compute_chunks`): it picks the `(m_chunk, n_chunk)` that **minimizes a two-term reload
   count** — `mblocks × N × 3 + nblocks × M × 2`, weight-dequant reloads against
   activation-convert reloads, the 3 : 2 being fixed constants — under the VTCM budget, tie-breaking
   to the largest chunk, halving the activation-thread count until it fits, and turning on
   double-buffering only at `M > 32` (`htp_mm_hmx_pipeline`). So: no *latency* model, no
   autotuning, no per-model table — a traffic model over shape + dtype + capacity + thread
   count, deterministic, cached. `[corrected 2026-08-20, second pass]` This is also exactly
   why prefill throughput rises with `ubatch`: each weight tile is loaded once per `m_chunk`
   rows, so a 1024-row ubatch amortizes the weight stream ~2× better than 512 until VTCM caps
   `m_chunk` and HMX compute becomes the bound. The reference scripts
   (`scripts/snapdragon/adb/run-*.sh`) run `--ubatch-size 1024 -fa on -ngl 99`, with
   `OB=` → `GGML_HEXAGON_OPBATCH`.
5. **On the DSP: split by output rows across threads, stream weights through a DMA ring,
   quantize activations once.** `matmul-ops.c:1367` divides weight rows across `n_threads`
   (rounded to 32 for repacked types); each thread owns a `dma_queue` that prefetches its next
   weight tiles DDR → VTCM while HVX/HMX consumes the current one (`dma_queue_push` /
   `dma_queue_pop` pairs throughout 340–800). On the **HVX quant path** f32 activations are
   quantized to Q8_0 (Q8_1 for Q4_1 weights) **into VTCM once per op** and shared by all
   threads — and `graph_optimize_reorder` stacks matmuls with the same input adjacent so the
   quantized block is reused across Q, K, V and gate, up. The **HMX path does not use Q8
   activations**: weight tiles are dequantized to fp16 in VTCM, f32 activations are converted
   to fp16 tiles, the HMX accumulates in its fp16 path, and the output tile is converted back to
   the f32 destination `[per Codex review]`.
6. **Prefill vs decode is a threshold, not a mode.** `M > 4` → HMX GEMM; `M ≤ 4` (decode) →
   HVX GEMV over int8 activations × int4 weights — a matmul-family choice, not a backend-wide
   mode; FA makes its own HMX/HVX decision and at `M = 1`, `DK ≤ 128` runs on HVX. Decode
   attention is on the device **when** `FLASH_ATTN_EXT`'s predicates pass (F16 K/V, supported
   shapes/layouts, VTCM), in which case it rides inside the same batch and costs no extra round
   trip; a Q8_0 KV cache or an unsupported neighbour makes it a CPU split `[per Codex review]`.
7. **Partial offload is a predicate, not a refusal.** `supports_op` (plus the
   `GGML_HEXAGON_OPFILTER` regex) returns false per (op, type, shape) and ggml's scheduler
   runs that node on the CPU, at the cost of a split boundary (a v73 report with unsupported
   Q4_K tensors showed hundreds of splits and no useful NPU work — 55a §2).
8. **Numbers** (all Q4_0 unless stated; 55a §2 has sources). Galaxy S25+ / 8 Elite (v79),
   Llama-3.2-1B: **pp128 169 t/s, tg64 51.5 t/s** (PR #16547, 2025-10). OnePlus 15 / v81,
   Qwen3-4B: pp512 **176 (HMX) vs 44 (HVX) vs 69 (CPU)**, tg128 **14.3 vs 13.3 vs 18.2** — the
   matrix unit gives prefill 2.5× over the CPU and decode **loses to the CPU** (PR #20693,
   2026-03). Galaxy S26+ / v81 after the HMX-FA rewrite: Llama-3.2-1B **4028 t/s prefill,
   54 t/s decode** (PR #25085; it names phones, not Hexagon revisions, and reports no
   `-ub` — the repository's run scripts use `--ubatch-size 1024`). The decode numbers barely
   move across two generations of silicon and three rewrites, which is **consistent with a
   weight-stream bound — a hypothesis from sensitivity, not a controlled result**
   `[per Codex review]`. Issue #18139 (closed "not
   planned") measures ~0.5 TOPS achieved against 45 TOPS peak on 8 Elite.
9. **What was rejected, and why it matters here.** The QNN-based `ggml-qnn` (PR #12326,
   closed) built one QNN graph per op and also tried direct cDSP kernels; its author reported
   poor QNN performance and preferred the direct route, without publishing a cost decomposition
   `[per Codex review]`. Qualcomm's own Genie/QAIRT compiles the whole
   model AOT into shared-weight context binaries with **separate AR-1 (decode) and AR-128
   (prefill) variants**. `ggml-hexagon` chose the middle: persistent generic kernels + batched
   descriptors. **This framework's four modes are points on the same spectrum**: `offload`
   (30 submissions/layer) is the unbatched per-op end, `runlist` (17) is batched dispatch,
   `fused` (1) is the Genie end — a compiled, shape-specialized graph with the packing bound
   that comes with static placement.

## 5. What transfers, what does not, and the twist on each

`[corrected 2026-08-20, second pass]` The first draft of this section compared against the
2025-10 HVX-only prefill figure (169 t/s) and concluded XDNA2 prefill was "an order of
magnitude ahead". **That was stale.** With HMX and the 2026 flash-attention rework
(PR #24954 / #25085, merged 2026-06/07, ~740–770-token prompts; the PRs do not report `-ub`,
the repository's run scripts use `--ubatch-size 1024` `[per Codex review]`):

| | Llama-3.2-1B, this repo (`llms/`, bf16, seq 2048, NPU2, June 2026 — pmode unrecorded) | Llama-3.2-1B, `ggml-hexagon` (Q4_0, HMX + FA, 2026-07) |
|---|---|---|
| prefill | TTFT 1.21 s for 2048 **padded** tokens ⇒ **~1,690 tok/s** incl. padded work | **4,028 tok/s** on S26+; 3B: 1,819. gemma-4-E2B 1,488 (S25+) / 1,837 (S26+); Qwen3.5-2B **985 (S25+) / 1,301 (S26+)** — phones, not Hexagon revisions `[per Codex review]` |
| decode | **12.2 tok/s**, 2 ELF dispatches/layer + LM head = **33 dispatches/token**, decode attention on the **CPU** | **54.2 tok/s** (v81) / 51.5 (v79, 2025-10), 1–2 queue writes/token, attention on the NPU |
| weight bytes/token | 1.24 B params × 2 B = **2.47 GB** | ~**0.73 GB** (Q4_0 incl. scales) |
| implied weight-stream rate | 2.47 GB × 12.2 /s ≈ **30 GB/s** (arithmetic) | 0.73 GB × 54 /s ≈ **40 GB/s** (arithmetic) |

Qwen3-0.6B is below every published model; the operator's figure of **≥ 3,300 tok/s prefill at
ubatch 512–1024 with HMX** sits on the published 1B → 0.6B scaling and is taken as the target
to beat, not a number measured here. So the framing is: **Hexagon prefill is ~2.4× ahead on a
like-sized model (and that is before subtracting the padded work here); decode is ~4× ahead on
4× fewer bytes at the same weight-stream rate.** Both halves of the token have a lesson; the
decode row still orders them because it is the larger ratio and the cheaper fix.

**Lesson 1 — the decode lever is weight width — and the first measurement of it here says the
lever alone gives 1.5×, not 4×. Transfers directly; the gap is the finding.** Hexagon never
streams bf16 weights; on the HVX decode path every GEMV is int4 weights × int8 activations.
Here, `llms/llama32_1b_int4` has the AWQ int4 **prefill** GEMM (8× slower than bf16 for the two
structural reasons its README records — `herd_m=2` at K=8192 from the L2 budget, `tile_n=16`
from a Peano immediate range) **and, contrary to its README and `llms/README.md` ("decode
follow-up"), an int4 decode path that exists and runs**: `llama32_1b_int4_inference.py` does
bf16 NPU prefill then int4 NPU decode through `rms_qkv_int4_rope` / `o_gemv_ffn_int4`, gated by
`make verify`, and its Makefile header records **~56 ms/token = 17.8 tok/s** against the bf16
path's 12.2 — **1.46×** (checkpoints and conditions differ; not controlled) `[per Codex
review]`. The first draft of this lesson predicted ~3–4× from weight bytes alone (2.47 → ~0.7
GB at a fixed ingest rate); that prediction is **refuted as already-evidenced**, and the
distance between 1.46× and ~4× is now the measurement to make: at 56 ms/token, ~0.7 GB of
weights at the ~30 GB/s implied rate is ~23 ms, so roughly 30 ms of every int4 token is
dequant, the 33 dispatches, CPU attention and host glue — which is Lesson 2's territory. The
study's decomposition (`device_ms / sync_ms / host_cpu_ms`) on that path is the first thing to
run. This is also [Goal 2's](00-context-and-goals.md) missing step 5 by another route, and the
`quant_*` schema columns, with their separate GEMM/GEMV contracts, were designed for exactly
this row.

**Lesson 2 — one dispatch per token, and attention comes back on-device with it. Transfers
with a twist; second.** Hexagon's whole-token batching is why decode attention is free there.
Here `llms/README.md` states the inverse: decode attention runs on the CPU "because NPU launch
overhead > compute", and the M=1 glue (residual add, FFN RMSNorm, SwiGLU) is on the host for
the same reason at ~0.13 ms/layer. Per-dispatch cost here is ~50–200 µs; 33 dispatches/token
is 2–7 ms of an 82 ms token — **not** the bf16 bottleneck (Lesson 1 is), but it becomes the
bottleneck the moment Lesson 1 lands and the token drops toward 20 ms. The twist is what a
"batch" is on XDNA2: a `dspqueue` batch is interpreted by a resident program with a dynamic
allocator; the XRT analog is a **runlist of runs within one hardware context**, and this
study's `runlist` mode already measured the trap — 24 context loads per dispatch when
heterogeneous attention artifacts are chained (12 heads × 2); doc [32](32-cost-decomposed-ladder.md)
measured a context load at ~3.7 ms on `offload`'s path at Turbo (82 ms at `Default`), and
whether `runlist`'s 24 cost the same each is an inference, not a measurement. So the translation is not "use runlists" but "**one
kernel image per token, dispatched N_layers times with different BO arguments**" — the
`llms/` 2-ELF decode form is already the same two kernels per layer with different weights,
so a 16-layer token is 32 runs of 2 kernels + 1 LM-head in one submit and one wait — and
**one submit, one hardware context, and one kernel image are three distinct targets**:
`dispatch.plan_submissions` aggregates ELF-ABI runs from artifacts that each hold their own
context into one runlist already; sharing a context needs a shared configuration
`[per Codex review]`. Then decode attention and the M=1 glue go back on the array because the marginal
dispatch is gone, exactly as Hexagon's `FLASH_ATTN_EXT` rides in the batch. The gate shape
exists (`run_npu2_fused_decoder_reexec_peano.lit` is a two-dispatch re-execution gate; the
unwrapped-Q-counter defect it caught is precisely the class of bug a many-dispatch-per-token
design exposes).

**Lesson 3 — power is voted from inside the session. Transfers directly; small.** Their
`start()` pins DCVS, corners and sleep; this study's 22× context-creation penalty at `Default`
([32](32-cost-decomposed-ladder.md)) is the same physics. The runner already refuses off
Turbo; the remaining gap is that the `llms/` profile numbers (June 2026) predate the pmode
finding and carry **no recorded pmode** — they should be re-walked once under recorded Turbo
before any of the arithmetic above is cited outside this document.

**Lesson 4 — "per-workload optimization" on Hexagon is a capacity-constrained legality cascade
with one greedy knob, not a cost model. Transfers with a twist, and it is the answer to §2.**
This is the lesson Codex's §4 under-states. `ggml-hexagon` chooses nothing by predicted
latency; it chooses the first kernel family whose analytically-built scratch layout fits, and
the deepest prefetch that fits. That is **the same move this study already made three times
in one week** — `derive_rows_per_call` for layer norm, then softmax, then
`decoder_stage_atol(hidden)` — each time replacing a constant sized at one width with a
derivation from capacity. Hexagon simply does it for every op, every shape, at graph-build
time, and never raises `KeyError`. The translation:

- `gemm_config()`'s miss should **derive** (largest legal `(tile_m, tile_n, tile_k)` under the
  L1 / memtile / BD / stride bounds the builders already assert) with the measured registry as
  the override table it already is. [54 §6](54-first-full-profile-and-decoder-families.md)'s
  three open walls — sub-256 `parallel_seq`, sub-512 attention tiles, `runlist` @16384's
  softmax width — are all "constant sized at one width", and the `ast` pins already turn a
  lifted bound into a measured rung automatically.
- The one **performance** decision Hexagon makes per workload is a hard threshold, `M > 4`
  ⇒ matrix unit. The study has no decode rung at all (§2: no KV cache, no M=1 path); the
  cheapest honest "per-workload" axis to add is `M ∈ {1, 4, 32, seq}` as a row dimension, not
  a selector.
- The selector that doc 53 / row 31 tried to build — rank modes by a declaration-side cost —
  closed negative, and Hexagon's history agrees: nobody there ranks `coarse` against `fused`;
  the "mode" (batched-descriptor) is fixed by design and only kernels vary. The defensible
  claim for this framework is the same: **pick the dispatch structure once from the
  measurement (doc 32's 16/16 ordering), and make every other parameter a capacity
  derivation.**

**Lesson 5 — partial offload as a predicate with CPU fallback. Transfers directly; cheap.**
`supports_op` + `OPFILTER` is the mechanism that lets a model with one unsupported op still
run (badly), and the split count is their diagnostic for "badly". The study's `skip_reason`
is the refusal half of the same predicate; `llms/` hand-codes the fallback half (decode
attention, the head-first transpose). Making it one predicate — `(op, shape, dtype) → device
| host | refuse`, with a split count in the dispatch vector — is what turns "all four modes
run the decoder" into "and this is what each mode leaves on the host".

**Lesson 6 — on-chip memory as explicit scratch with a planner. Transfers in principle; the
allocator does not.** VTCM is one flat 8 MiB with a runtime allocator; XDNA2 is 8 × 512 KiB
memtiles + 32 × 64 KiB L1s behind static placement and routing. The principle (tile to
capacity, double-buffer DMA against compute, fuse only while the live set fits) is already
how `fused` and the multi-launch ELFs are built; the twist is that Hexagon's "fits" check is a
function call at graph time and ours is an aircc compile — which is why [54 §3](54-first-full-profile-and-decoder-families.md)
insisted the bounds be lifted into pure functions the host suite can call. Lesson 4 is the
same point from the other side.

**Lesson 7 — `NDEV` / multiple sessions. Does not transfer.** It exists only to escape a
~2–3.5 GiB per-process VA mapping limit; XRT BOs have no such cliff here, and splitting layers
across logical devices multiplies nothing physical. The one adjacent idea worth a note is
ggml's experimental backend-agnostic tensor parallelism (PR #19378), which on XDNA2 would mean
column-partitioned hardware contexts — a research item, not a lesson.

**Lesson 8 — activations quantized on-device, once, and shared. Transfers with a twist; part
of Goal 2.** On the HVX quant path, f32 → Q8_0 into VTCM per op, reused across Q/K/V and
gate/up by reordering the graph; on the HMX path the analogous step is f32 → fp16 tiles with
weights dequantized to fp16 `[per Codex review]`. The template to copy is therefore the
**separation of storage type, scratchpad compute type, accumulator type and model-visible
output type** — not Hexagon's particular fp16/Q8 choices; AIE2P's natural floating path is
bf16, and its native narrow block format is `bfp16ebs8`, which is the closer analog of HMX's
fp16 tile path. The shipped int4 kernel here dequantizes weights to bf16 and MACs in bf16
(`d·(q−8)`); an int8×int8 path would need the AIE2P integer MAC route and is a kernel study.
The cheaper half — compute the shared input once for the three projections — is already what
`qkv_proj` as one `(seq, h, 3h)` GEMM does.

**Lesson 9 — on-device profiling returned with the batch.** Per-op cycles and eight PMU
events come back in the batch response; `GGML_HEXAGON_PROFILE=2`. The study's schema v2 has
the host-side decomposition (`device_ms / sync / host_cpu`, the six-field dispatch vector) and
no on-array counters; mlir-aie's trace path is the analog. Optional.

## 6. What this changes in the work queue

Nothing here displaces item 1 (the iron gap), which is an attribution task with the adapter
already in place. It **re-ranks item 5 (Goal 1's model path) below a decode item that did not
exist on the list**, and gives item 4 (lift a derived skip into a rung) its principled form:

1. *(unchanged)* Attribute the iron latency gap.
2. **New — decompose the int4 decode token that already exists** (Lesson 1, corrected): the
   `llama32_1b_int4` decode path measures 17.8 tok/s against a ~4× weight-bytes ceiling; run it
   under the study's `device / sync / host` decomposition and dispatch vector to attribute the
   other ~30 ms/token before any kernel work. Then the `w4_decode` plan in [56](56-full-model-mixed-precision-study-plan.md).
3. **New — one-context, one-submit decode token** (Lesson 2), then move decode attention and
   the M=1 glue back on-device; gate with the two-dispatch re-execution lit shape. Measure
   dispatches/token 33 → 1 and the host residue.
4. *(re-shaped)* `gemm_config()` derive-on-miss under the builders' bounds (Lesson 4), which
   is how the three doc-54 walls close together rather than one kernel at a time.
5. *(new, small)* Re-walk the `llms/` profile table once under recorded Turbo (Lesson 3)
   before citing it; add `M` as a study axis (Lesson 4) so decode has a rung.
6. *(unchanged)* The big-three model leg; the doc rewrite; R1.

Not proposed, with reasons: `NDEV`-style sessions (no VA cliff here); a latency-model mode
selector (closed negative, and Hexagon does not have one either). `[corrected 2026-08-20,
second pass]` Prefill IS in scope — the first draft struck it on a stale number; the
ubatch-chunked prefill (compile at `M = ubatch`, loop chunks, pad only the last) is the prefill
half of Lesson 2 and is carried into [56](56-full-model-mixed-precision-study-plan.md).
The cross-chip comparison stays not-like-for-like and the adapter's refusal logic applies.
