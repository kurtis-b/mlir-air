# 57 — Optimizing the `llms/` inference path: the `ggml-hexagon` mechanisms, translated, with the measurements that rank them

`[2026-08-20]` The operator asked, while away, for the inference-path optimizations that
follow from [55](55-hexagon-llama-cpp-lessons-for-xdna2.md)'s reading of llama.cpp's Hexagon
backend — mechanism to mechanism, methodology to methodology — worked through with Codex. This
document does that in the order the evidence demands: §1 measures where a decode token and a
prefill layer actually spend their time today (new runs, Turbo verified, devq 436–446); §2
names the one structural fact those numbers expose; §3 is the translation table, each row a
Hexagon mechanism → the mechanism here → the optimization → a prediction → the experiment that
gates it; §4 is the predicted budget after each step; §5 the experiments in the order to run
them; §6 what does not transfer; §7 Codex's review. Numbers marked *arithmetic* are derived
from measured ones and say so. Every job log, the probe scripts and the probe's JSON are in
`programming_examples/transformer_layer/results/hexagon-opt-20260820/` (gitignored, local).

## 1. Where the time goes today

### 1.1 Qwen3-0.6B bf16, decode — devq 436 (`make profile N_TOKENS=32`, Turbo observed before and after, commit `debf9be2`)

32 generated tokens, short prompt (context ≤ ~60). Per token, from the profiler's own buckets:

| Component | Per call | Calls / token | Per token | Bytes streamed | Effective rate |
|---|---|---|---|---|---|
| `o_gemv_ffn` (O + residual + RMSNorm + gate/up + SwiGLU + down) | 1.53 ms NPU run (BO write 0.01) | 28 | **42.8 ms** | 23.1 MB | **15 GB/s** |
| `rms_qkv_qknorm_rope_gemv` (RMSNorm + Q/K/V + QK-norm + RoPE) | 1.03 ms | 28 | **28.8 ms** | 8.4 MB | **8 GB/s** |
| `lm_head_gemv` (19 partitions × 8192 rows) | 9.68 ms | 1 | **9.7 ms** | 311 MB logical / **319 MB padded** | **33 GB/s** |
| `decode_attention_cpu` | 0.12 ms | 28 | 3.4 ms | — | — |
| embed / final norm / BO writes / reads | — | — | ~1 ms | — | — |
| **Total** (profiler: 2738.6 ms NPU + 106.9 ms CPU over 32 tokens) | | | **~89 ms ⇒ 11.2 tok/s** | **1.19 GB** | **13 GB/s overall** |

Three things the table says on its own:

- **The machine streams weights at 32 GB/s when the launch is large** (`lm_head`: 19
  launches of 16 MB, 0.51 ms each). **The per-layer decode ELFs reach 8–15 GB/s.** If the
  per-layer work ran at the `lm_head` rate it would take 28 × (31.5 MB / 32 GB/s) = **27.6 ms**
  instead of 71.6 — *arithmetic*.
- **Decode for this model is not bandwidth-bound.** 1.19 GB/token at 89 ms is 13 GB/s overall;
  the study's single-shim-port figure is 5.3 GB/s (`analytical_cost.py:49`) and eight columns
  are in use by the GEMV builders (`n_cores=8`, `_STAGE2_HERD_COLS=8`).
- **CPU attention is negligible at short context** (3.4 ms/token) — and will not stay so:
  `decode_attention_cpu` (`qwen3_0_6b_decode.py:285-316`) converts the *whole* cached slice
  `k_cache[:, :seq_len, :].astype(float32)` (and `v`) per layer per token — at context 2048
  that is 2 × 8 × 2048 × 128 × 4 B = **16.8 MB of conversion per layer, ~470 MB per token** —
  and loops over 16 heads in Python. **Measured (devq 444, Turbo, 16 tokens each):** 0.90 ms
  per layer at ~1,000 tokens of context and **1.93 ms at ~1,900** — **25 and 54 ms per token**,
  linear in context — while the NPU part stayed at ~88 ms; the token grows from 89 ms to
  ~145 ms (**6.9 tok/s**) at ~1,900 context. The wall is real and it is host-side.

The Llama-3.2-1B bf16 path (June table: 12.2 tok/s, 2.47 GB/token) implies ~30 GB/s overall
because its weight matrices are bigger per launch (emb 2048, hidden 8192) and amortize the
same fixed costs better — which is the first hint of §2.

### 1.2 Qwen3-0.6B bf16, prefill at `seq_len = 2048` (same run)

| Kernel | NPU run | BO write | Per layer | Share | Rate |
|---|---|---|---|---|---|
| `flash_attn` (head-first, hd 128, causal) | 21.23 ms | 1.41 ms (**24 MB** host-transposed Q/K/V) | 22.7 ms | **51 %** | 34 GFLOP dense / 17 causal-effective ⇒ **0.8 TFLOPS effective** |
| `o_ffn_qwen` | 12.71 ms | 0.69 ms | 13.5 ms | 30 % | 47 GFLOP ⇒ **3.7 TFLOPS** |
| `rms_qkv_qknorm_rope` | 7.72 ms | 0.30 ms | 8.2 ms | 18 % | 17 GFLOP ⇒ 2.2 TFLOPS |
| Σ kernels | | | **44.8 ms / layer**, 1,256 ms / 28 layers | | 2048 tokens ⇒ 1,630 tok/s kernel-only |
| layer-time incl. host transposes and KV extract | | | 51.8 ms / layer (1,451 ms) | | |

And the whole prefill is spent on **2,048 padded tokens regardless of the prompt**
(`qwen3_0_6b_inference.py:645-649`): a 60-token prompt pays the full 1.45 s.

### 1.3 Llama-3.2-1B int4, decode — devq 440 (caches compiled in devq 437; Turbo observed)

The int4 driver builds a `Profiler` but never prints it, so this ran through a wrapper that
calls `report()` (the wrapper forced CPU *prefill* attention by mistake — its TTFT is not a
number; its decode path is the production one). 32 tokens after a 57-token prompt,
**66 ms/token = 15.3 tok/s** (devq 438, the plain driver: 64 ms, 15.7).

| Component | Per call | Calls / token | Per token | Bytes | Rate |
|---|---|---|---|---|---|
| `lm_head_gemv` — **still bf16** (tied embeddings, 128256 × 2048 × 2 B) | 14.85 ms | 1 | **14.9 ms (23 %)** | 525 MB logical / **537 MB padded** | **36 GB/s** |
| `o_gemv_ffn_int4` (3 launches: O+add, gate/up cascade, down+add) | 2.04 ms | 16 | **32.6 ms** | ~28 MB int4 + scales | **14 GB/s** |
| `rms_qkv_int4_rope` (6 launches) | 0.77 ms | 16 | **12.3 ms** | ~3.2 MB | **4 GB/s** |
| `decode_attention_cpu` | 0.24 ms | 16 | 3.8 ms | — | — |
| **Total** | | | **~65 ms** | **~1.0 GB** (0.5 of it the bf16 head) | 15 GB/s overall |

Read against the bf16 path's 12.2 tok/s (82 ms): int4 removed ~1.45 GB of the 2.47 GB per
token and bought 16 ms, because (i) the bf16 LM head is now the single largest item, (ii) the
int4 GEMVs stream at 4–14 GB/s — *slower per byte* than the bf16 ones, dequant and the
per-launch fixed cost dominating at these sizes — and (iii) the token still carries
16 × (6 + 3) + 8 = **152 launch boundaries**. Every one of those is a §3 row (O4, O3, O1).

### 1.4 The per-launch cost, isolated — devq 445 (`probe_pdi_cost.py`, Turbo observed)

Identical LM-head work — the same 155,648 × 1024 bf16 weights, the same input, 319 MB
streamed — built two ways with `build_lm_head_gemv_module`: **19 launches × 8192 rows** and
**38 launches × 4096 rows**. 3 warm-ups, then 20 timed calls each, interleaved:

| Variant | p50 | min | avg | max |
|---|---|---|---|---|
| 19 × 8192 | **9.834 ms** | 9.639 | 9.852 | 10.164 |
| 38 × 4096 | **11.901 ms** | 11.853 | 11.978 | 12.468 |

**Δp50 = 2.067 ms for 19 extra boundaries ⇒ 109 µs per in-ELF launch boundary**, at
32.4 GB/s for the 19-launch form (which matches the production `lm_head` line in §1.1 to 2 %).
Compile time is its own cost of launch count: 54 s for 19 launches, **592 s for 38**.

Reproduced (devq 446): p50 9.96 vs 12.05 ms, **110 µs per boundary**, 32.0 GB/s.

**What the probe does and does not isolate** `[per Codex review]`: bytes, arithmetic, kernel
source and weights are identical, but halving `n_part` also halves each launch's internal
iteration count (`launch_size = m / tile_m / herd_m`, `matvec.py:108`: 128 → 64) and moves the
broadcast-DMA repeat geometry from the 255 limit to ~127. The 2.07 ms therefore contains the
reconfiguration **plus** whatever the changed per-launch DMA schedule costs; BD-count equality
is not established. "109 µs per boundary" is the right order and the right direction — the
per-layer gaps in §1.1 say the same thing independently — but as an *isolated* boundary cost
it is **unverified**. The isolating experiments: 38 correct 4096-row segments run either as 38
devices or as **19 devices each performing two segments** (same descriptors and repeat
geometry, only the configuration count differs — not yet run); and, for the repeat edge,
19 × 8192 at `m_input = 4` (repeat 255) against `m_input = 8` (~127), same bytes and launch
count — **run (devq 449)**:

| Variant | p50 | Rate |
|---|---|---|
| 19 × 8192, `m_input 4` (production, repeat 255) | 9.96 ms | 32.0 GB/s |
| 19 × 8192, `m_input 8` (repeat ~127) | **9.12 ms** | 35.0 GB/s |
| 38 × 4096, `m_input 4` (repeat ~127) | 12.23 ms | 26.1 GB/s |

The shorter repeat geometry is **faster** per byte, not slower — so the 38-launch form enjoys
it too, and the confound makes the isolated boundary cost **larger** than the naive delta, not
smaller: between (12.23 − 9.96) / 19 = **120 µs** and (12.23 − 9.12) / 19 = **164 µs**
depending on which single-launch geometry the 4096-row form matches. Every conclusion drawn
from "~110 µs" below holds with that band; the 19-devices-×-2-segments probe would pin it.
Two by-products: **`m_input = 8` is a free ~0.85 ms/token on the production LM head** if it
verifies (§5 item 4); and the per-launch geometry is itself an O3 knob.

**A defect found on the way, not chased** (devq 446/447). The probe's correctness check
passed the 38 × 4096 form (max 2.5e-3 of output scale, bit-identical across runs) and failed
the 19 × 8192 form: **2,455–2,800 of its 155,648 outputs wrong, all in partition 0, rows 64
onward (the second 64-row tile of the first launch), and non-deterministic run to run (max
diff 3.4)** — but **only when the ELF re-executes immediately after itself**; run once after a
different ELF it is exact ("p19 after p38: bad 0"). The production Qwen LM head is this exact
ELF (`_LM_N_PART = 8192`, pinned at the BD repeat-count limit `n_part/32 − 1 = 255`) and its
top-5 gate passes because decode never runs `lm_head` back-to-back — an `o_gemv_ffn` always
precedes it. Any design that does run it back-to-back (batched logits, a per-token runlist
that places it adjacent to itself, re-scoring) will hit this. It is the same family as the
fused decoder's re-execution wall ([PREDICTION-FUSED-REEXEC](PREDICTION-FUSED-REEXEC.md):
state left in the partition by one execution corrupting the next), with the repeat-count edge
as one suspect — though the pattern (non-deterministic, partition 0 only, from row 64, healed
by any intervening configuration) fits **stale repeat / buffer / configuration state** better
than a last-tile defect `[per Codex review]` — and devq 448 **rules the repeat limit out**:
19 × 8192 at `m_input = 8` (repeat ~127) is still non-deterministic back-to-back (partition 0,
14–107 bad rows at 48, 96, 288, … instead of ~2,800 from row 64), while 38 × 4096 at the same
repeat is exact. The defect follows the **launch size (128 iterations per launch)**, not the
repeat count. Production's gate cannot see it: only the token
set gates, the full-logit comparison is informational (`verify/report.py:54`,
`comparators.py:184`). The settling experiment: the same 19-launch ELF at `n_part = 4096`-sized partitions but 8192
rows via two BDs, re-executed back-to-back; and the existing two-dispatch re-execution gate
shape applied to `lm_head_gemv`. The timing comparison is unaffected — both forms move the
same bytes through the same kernel, the timed loop alternated the two ELFs (the correct
case), and the per-boundary figure is corroborated independently by §1.1's per-layer gaps
(`rms_qkv`: 1.03 ms measured − 0.26 ms at 32 GB/s = 0.77 ms over 8 boundaries ≈ 0.1 ms each).

## 2. The structural fact: every `air.launch` boundary is a partition reconfiguration

[29](29-offload-n-streams.md) records the multi-launch mechanism: a module with N
`air.launch` ops lowers to **N `aie.device` ops plus a `main` device** whose runtime sequence
issues a configure/run pair per launch (`mlir/lib/Conversion/AIRRtToNpuPass.cpp:1443`). On the
ELF path the per-launch images travel as `.pdi.N` / `.ctrltext.N` sections and the **ELF
loader resolves the reconfiguration** — a raw in-stream `load_pdi` faults NPU2 firmware
([29 §The hardware verdict](29-offload-n-streams.md)) — so the precise statement is that **each
launch selects and configures a distinct device image** `[per Codex review]`. Either way the
array has no resident program: a "launch" costs a reconfiguration, not a descriptor.

Count them per decode token for Qwen3-0.6B: `rms_qkv_qknorm_rope_gemv` is 8 launches
(`rms_qkv_qknorm_rope_multi.py:670`: RMSNorm, Q, K, V, QK-norm ×2, RoPE ×2), `o_gemv_ffn` is 3
(`o_gemv_ffn_multi.py`: `matvec_2tile_add`, `matvec_swiglu_rms` cascade, `matvec_2tile_add`),
`lm_head` 19 — **28 × 11 + 19 = 327 launch boundaries per token**, against 57 `xrt.run`s and
one logical token. **Measured: 109 µs per boundary, with §1.4's caveat that the probe did not hold the per-launch DMA geometry constant** — the order of magnitude is corroborated by the per-layer gaps. Applied to the 308 per-layer
boundaries that is **33.6 ms of the 89 ms token — 38 %** — before a single weight byte moves.
It also explains the June int4 result: Llama-1B int4 decode went 12.2 → 17.8 tok/s (1.46×,
15.3 re-measured today) on a ~4× narrower weight stream because its token still carries
152 boundaries (16.6 ms) and a 525 MB bf16 LM head (14.9 ms).

This is the exact inverse of the Hexagon design. `ggml-hexagon`'s DSP runs **one persistent
program**; an "op" in a batch is a descriptor the program interprets, and 1024 of them cost one
`dspqueue_write` and zero reconfiguration. The lesson [55](55-hexagon-llama-cpp-lessons-for-xdna2.md)
called "op batching" lands here one level lower than `xrt.run`: **it is the launch count, not
the submission count, that has to fall.**

A second, smaller structural fact from the same place: `_LM_N_PART = 8192` is pinned by a BD
**repeat-count limit** (`qwen3_0_6b_decode.py:85`: `n_part/32 − 1 = 255`) — one launch cannot
stream more than 8192 rows of this GEMV, which is why the LM head is 19 launches at all.
Hexagon's DMA descriptors have no such cap; the XDNA analog is a BD chain or an outer loop in
the runtime sequence rather than one BD's repeat count.

## 3. The translation table

Each row: the Hexagon mechanism (55 §4, corrected) → what this path does today → the
optimization → predicted effect on the Qwen3-0.6B token of §1.1 (*arithmetic* unless marked)
→ the gate.

| # | Hexagon mechanism | Here, today | Optimization | Predicted effect | Gate / experiment |
|---|---|---|---|---|---|
| **O1** | **No reconfiguration between ops**: one persistent program; an op is a descriptor | 327 PDI loads / token (§2) | **Cut launch boundaries per layer 11 → ≤ 3**: (a) Q, K, V as **one** GEMV launch over the concatenated `[wq; wk; wv]` (N = 4096) — Hexagon's `MUL_MAT_QKV`; (b) the M = 1 vector ops (RMSNorm, QK-norm, RoPE, SwiGLU glue) as **prologue/epilogue inside the GEMV core program** rather than their own launches — Hexagon's `RMS_NORM+MUL` / `MUL_MAT+ADD` fusions (at M = 1 these touch 1–4 K elements; a launch for each is all overhead); (c) `lm_head` partitions under **one** `aie.device` as a single-device BD chain (an "outer loop in the runtime sequence" does not work: the lowering deliberately resets at launch end, `AIRRtToNpuPass.cpp:1037` `[per Codex review]`). (a) is builder work; **(b) and (c) need new core kernels** — stitching only concatenates launch bodies (`stitching.py:383`); `matvec_swiglu_rms.py` shows fused epilogues are possible, not that they are mechanical | **with O3, jointly** −25 … −35 ms / token (§4) | the isolating probe of §1.4 first; then `rms_qkv` 8 → 1–2 launches as the first kernel change, gated by the model's `make verify` and the profiler's `rms_qkv` line |
| **O2** | **Op batching**: ≤ 1024 ops per `dspqueue_write`, ≤ 16 in flight | 57 `xrt.run` + wait per token, one per ELF | `dispatch.run_sequence(require_single_submission=True)` over adjacent runs: layer L's `o_gemv_ffn` → layer L+1's `rms_qkv` have no host op between them ⇒ **57 → 30 submissions** today (RMS₀; 27 `(O_L, RMS_{L+1})` pairs; O₂₇; LM head — the final RMSNorm is a CPU barrier before the head, `inference.py:431` `[per Codex review]`), lower once attention and the final norm are on device (O6) | 0 … −6 ms / token **until measured** (host submit/wait per run, ~50–200 µs) | exists; a driver flag; dispatch vector shows `host_submissions` |
| **O3** | **All HVX threads stream, each with its own DMA queue** (`n_threads` rows × per-thread `dma_queue`) | per-layer ELFs at 8–15 GB/s vs 32 GB/s for `lm_head`; `matvec_2tile_add` uses **one core per column** (`n_cores=8`, `herd [8,1]`) | After O1, re-measure each stage's streaming rate; where below the large-GEMV reference: both shim input channels per column, 2–4 rows per column splitting the output rows — **new mappings, not the existing `lm_head` geometry** `[per Codex review]` | **not additive with O1**: the joint O1+O3 saving is `71.6 ms − (measured post-fusion time)`, bounded below by the 27.6 ms weight-stream floor plus vector/dequant work. 32 GB/s is an *achieved reference* for large eight-column GEMVs, not a ceiling (8 × 5.336 = 42.7 GB/s of shim ports before contention) | per-stage rate from the profiler's `NPU Run` column vs bytes |
| **O4** | **The output matrix is quantized too** (Q4_0/Q8_0 `output.weight`; repacked once) | `lm_head` bf16, 311 MB / token = 9.7 ms at 32 GB/s | int4 (q4_0 gs 32 symmetric or AWQ) LM-head GEMV — `matvec_int4_packed.py` exists, `symmetric=True`; one resident packed copy beside the bf16 embedding rows | **−6 … −7 ms / token** (311 → ~90 MB at the same rate) | verify top-5 token set (the head is where int4 error shows first); `bytes_transferred` |
| **O5** | **All weights narrow, resident, repacked once** | bf16 per-layer weights, 881 MB / token | `w4_decode` for Qwen3-0.6B — the Llama int4 QKV builder lacks Qwen's per-head QK-norm, so this is a builder change, not reuse `[per Codex review]`; **after O1**, because the Llama int4 data says weight width alone buys 1.46× while boundaries dominate, and the int4 GEMVs measured at 4–14 GB/s cannot be assumed to reach the bf16 reference rate | **no budget credit today**; 881 → ~250 MB is the byte ceiling, the rate is unmeasured | prediction doc first ([56 §4 H2b](56-full-model-mixed-precision-study-plan.md)); verify; `quant_*` columns populated |
| **O6** | **Attention rides in the batch** (`FLASH_ATTN_EXT` at M = 1, KV in mapped DDR, when F16 K/V) | CPU attention; whole KV slice converted bf16→f32 per layer per token; Python loop over heads; KV written on host | (i) now: keep the host KV cache in f32 and vectorize over GQA groups — removes the per-token conversion; (ii) then: the `attn_decode_npu2` kernel (device KV; its `pos` is **compile-time** today — `pos_host`, `attn_decode_npu2.py:470` — so first make it a run-time argument) generalized to hd 128 / GQA 2, inside the per-token runlist | (i) removes the copies and the Python loop, but **cannot hold attention at ~3 ms**: at 2,048 context the host must still read ~470 MB of K/V per token (16.8 MB × 28) — a constant-factor win over today's 54 ms, to measure `[per Codex review]`; (ii) is the real fix and needs a kernel/interface change (`pos` at run time) — **no budget credit today** | long-prompt profile (ctx 1024, 2048) before/after |
| **O7** | **Per-token state stays resident**; position is an op parameter | RoPE LUT for `pos` rebuilt with `np.tile` and uploaded **per layer** (2 BO writes × 28 = 56 / token, identical across layers); a dead 6 MB `np.zeros((hidden, emb))` allocated per layer per token (`_run_o_gemv_ffn`) | share one LUT BO across layers (`shared_nonstatic` pool) or resident full table + `pos` as a run-time arg (no kernel here takes one yet); preallocate the dead args once | small at bf16 (BO write 0.01 ms/call); ~1 ms/token of host allocation; matters once O1–O5 shrink the token | profiler `BO Write` and CPU buckets |
| **O8** | **`ubatch`: the physical chunk is the compute shape**; a 60-token prompt is one 64/128-row ubatch | every prompt padded to 2048 (`inference.py:645`) | compile prefill at `M ∈ {128, 256, 512, 1024}` and run ⌈L / M⌉ chunks ([56 §3.4](56-full-model-mixed-precision-study-plan.md)); single-chunk prompts need no new kernel | **TTFT for a 60-token prompt 1.45 s → ~0.1 s**; at 1024 tokens ~0.7 s | [56](56-full-model-mixed-precision-study-plan.md) H1a/H1b gates |
| **O9** | **HMX flash attention** (2026 rework: +40 % prefill) | FA is 51 % of the prefill layer at 0.8 TFLOPS causal-effective vs 3.7 for the FFN GEMM; 24 MB of host-transposed Q/K/V uploaded per layer | (i) causal block skipping in the head-first kernel (half the tiles are masked); (ii) **hypothesis, undemonstrated**: move the seq↔head transpose into the DMA by addressing bf16 pairs as 32-bit elements — the repository establishes only that the sub-32-bit innermost stride must be 1 (`data_transfer_transpose/dma_bf16/transpose_bf16.py:9`); legality of the reinterpretation, alignment and the on-device handoff are unproven `[per Codex review]` | (i) up to −10 ms / layer (−280 ms / prefill) — causal masking is applied after a dense K matmul today (`attn_npu2.py:734`), so the skip is real; (ii) −1.4 ms upload + ~5 ms/layer of host transposes **if** it works | FA lit at 2048 hd 128; verify |
| **O10** | **Plans cached by graph uid; kernel params precomputed on the host** | artifact cache keyed by name (`cache.py:358`) | plan-hash keying ([56 §3.3](56-full-model-mixed-precision-study-plan.md)) | correctness of every experiment above (no silent shape collisions) | [56](56-full-model-mixed-precision-study-plan.md) H1a |
| **O11** | **Power voted at session start** | Turbo required by the study runner; `make profile` prints nothing about pmode | print observed pmode in the profile header; refuse off-Turbo in `llms/` drivers as the study does | measurement validity | one-line change |
| **O12** | **Per-op cycle counters and PMU events returned with the batch** | host-side three-segment timing per `xrt.run` | optional: per-launch timestamps from the runtime sequence (trace) — after O1 there are few enough launches to read by eye | — | — |

## 4. The decode budget — a defensible band, not a waterfall `[rewritten per Codex review]`

The first draft added O1, O3, O5 and O6 as independent savings down to ~18 ms/token. They are
not independent (removing boundaries also removes part of the "rate gap" charged to O3) and
two of them have no measured rate behind them. The defensible statement, on §1's measurements:

| Item | Credit | Condition |
|---|---|---|
| baseline (devq 436) | **89 ms / token** (11.2 tok/s) | short context; 145 ms at ~1,900 context (devq 444) |
| O1 + O3 **jointly** | **−25 … −35 ms** | the joint saving is `71.6 − (post-fusion measured)`, floored by the 27.6 ms weight-stream time plus vector/dequant work; credited only after the isolating probe (§1.4) |
| O4 (int4 LM head) | **−6 … −7 ms** | after the accuracy gate |
| O2 + O7 | **0 … −6 ms** | until measured |
| O5, O6 | **0** today | O5's int4 rate and O6's device attention are unmeasured |
| **Conditional band** | **41 – 58 ms / token (17 – 24 tok/s)** | |

The Hexagon-style end state — one submission, a few launches, narrow resident weights,
attention in the batch, at a number near llama.cpp's 54 tok/s for Llama-1B on v81 — remains the
*direction*; as a figure it is a stretch hypothesis, not arithmetic, until O5 and O6 have
rates. Moving from the band to that figure is what [56](56-full-model-mixed-precision-study-plan.md)'s
H2/H3 phases measure.

## 5. Experiments, in order

1. ~~**Per-boundary cost**~~ **DONE** (devq 445, §1.4): **109 µs per boundary**.
2. ~~**int4 decode decomposition**~~ **DONE** (devq 440, §1.3): int4 GEMVs at 4–14 GB/s, the
   bf16 LM head 23 % of the token, 152 boundaries.
3. ~~**Long-context decode profile**~~ **DONE** (devq 444, §1.1): 25 / 54 ms per token of CPU
   attention at ~1,000 / ~1,900 context.
3b. ~~**Repeat-geometry isolation**~~ **DONE** (devq 448/449, §1.4): `m_input 8` is 8.5 %
   faster at the same launch count; the defect is not the repeat limit. **Still to run**: the
   19-devices-×-2-segments probe and the `lm_head` back-to-back re-execution gate.
3c. **`m_input = 8` on the production LM head**: `make verify` for Qwen3-0.6B; predicted
   −0.85 ms/token.
4. **O2 prototype** behind a driver flag: `run_sequence` over the (L `o_gemv_ffn`, L+1
   `rms_qkv`) pairs; dispatch vector `host_submissions 57 → 29`; `make verify`.
5. **O1 first cut**: `rms_qkv_qknorm_rope_gemv` as one GEMV launch over `[wq; wk; wv]` with
   QK-norm + RoPE as an epilogue; predicted `rms_qkv` line 1.03 → ~0.4 ms (8.4 MB at 32 GB/s
   + 1 boundary); `make verify`.
6. **O4**: int4 LM head; predicted 9.7 → ~3 ms.
7. Then O3 / O5 / O6(ii) / O8 / O9 per [56](56-full-model-mixed-precision-study-plan.md)'s phases.

## 6. What does not transfer, restated for the inference path

`NDEV` sessions (no VA cliff); the `M > 4` HMX threshold (an HMX/HVX fact); Q8 activation
quantization as the first precision step (Hexagon's HMX path does not use it either, and the
decode GEMVs here are overhead-bound, not MAC-bound); a latency-model mode selector. And one
thing that transfers only with a warning: Hexagon's decode numbers are the *ceiling* this
table walks toward, measured on a different memory system; the comparison that matters is
each row's own before/after under recorded Turbo.

## 7. Codex review

Report: [57a](57a-codex-review-of-inference-optimizations.md), verbatim. Verdict as delivered:
"major revision — the launch-count diagnosis is directionally persuasive, but 109 µs is not an
isolated PDI-boundary measurement, and the 18–22 ms endpoint rests on overlapping or
unsupported assumptions." Applied, each marked above: the ELF reconfiguration mechanism
restated (device images via `.pdi.N` sections, not a raw `load_pdi`); the probe's confound
(per-launch iteration count and repeat geometry change with `n_part`) and the two isolating
experiments; the defect re-read as stale state rather than a last-tile bug, with the note that
the token gate cannot see it; launch counts confirmed; padded LM-head bytes (318.8 / 536.9 MB)
for rates; 32 GB/s demoted from ceiling to achieved reference; O1(b) and O1(c) re-costed as
new kernels and O1(c)'s mechanism corrected; O2 = 30 not 29; O3 made non-additive with O1; O5
re-costed (QK-norm, unmeasured int4 rate); O6(i)'s 3 ms claim withdrawn (the host must read
~470 MB/token at 2,048 context regardless); O9(ii) marked undemonstrated; O10–O12 moved out of
the performance budget; §4 replaced by the 41–58 ms conditional band.

Codex's list of Hexagon mechanisms this document under-uses — the capacity/live-set fusion
planner, one-time shared activation preparation reused across stacked matmuls, and explicit
host-fallback split accounting — is [56](56-full-model-mixed-precision-study-plan.md)'s H0
by another name, and is why O1's fusion decisions should be made by that planner rather than
by hand once it exists.
