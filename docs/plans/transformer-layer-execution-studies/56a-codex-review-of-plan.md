# 56a — Codex review: verification of doc 55 and critique of the doc 56 plan

`[2026-08-20]` Produced by Codex (session `01a0207e-304b-7940-9c32-74f17020b5a1`) on the prompt recorded in [56 §7](56-full-model-mixed-precision-study-plan.md), reproduced **verbatim**. Its labels: **SOURCE** / **INFERENCE** / **UNVERIFIABLE**. Every correction it made to [55](55-hexagon-llama-cpp-lessons-for-xdna2.md) is applied there and marked `[per Codex review]`; every change it made to the plan is in 56 §7.

---

# Research and design review

Review basis: repository branch as provided; upstream `ggml-org/llama.cpp` `master` inspected on 2026-08-20. Labels mean:

- **SOURCE** — directly verified in the cited repository file, upstream source, or PR.
- **INFERENCE** — reasoned conclusion from cited evidence.
- **UNVERIFIABLE** — the requested evidence was not published or could not be established from the cited material.

## 1. Part A

### A1 — Is Qwen3-0.6B Q4_0 at ≥3300 prefill tok/s plausible?

#### Published evidence

- **SOURCE** — PR #25085 was posted 2026-06-27 and merged 2026-07-01. Its post-rework results were:

  - Gemma-4-E2B Q4_0, 741 prompt tokens: S25+ **1488.15 tok/s**, S26+ **1837.38 tok/s**.
  - Qwen3.5-2B Q4_0, 742 prompt tokens: S25+ **984.92 tok/s**, S26+ **1300.80 tok/s**.
  - Llama-3.2-1B Q4_0, 766 prompt tokens: S26+ **4027.72 tok/s** prefill and **54.22 tok/s** decode. [PR #25085 benchmark block](https://github.com/ggml-org/llama.cpp/pull/25085)

- **SOURCE** — therefore, “Qwen3.5-2B is ~1300 tok/s on S25+/v79” is wrong: **1300.80 is the S26+ result; S25+ is 984.92**. The PR names phone models, not Hexagon architecture revisions. Current documentation lists both v79 and v81 libraries, but does not itself map those two benchmark phones to those revisions. [developer.md:3–21](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/developer.md#L3-L21)

- **UNVERIFIABLE** — PR #25085 does not report `-ub`, `-b`, `GGML_HEXAGON_OPBATCH`, or `OPQUEUE` for those measurements. The current Windows launcher does specify `--ubatch-size 1024`, but there is no evidence in the PR that this exact script/configuration produced the published table. [run-cli.ps1:43–48](https://github.com/ggml-org/llama.cpp/blob/master/scripts/snapdragon/windows/run-cli.ps1#L43-L48)

- **UNVERIFIABLE** — I found no published controlled `pp512`/`pp1024` sweep against `n_ubatch`, for either Qwen3-0.6B or another current HMX model. The available PR tables vary model and device, not ubatch while holding the workload fixed.

#### What the controls actually mean

- **SOURCE** — llama.cpp defaults are logical batch `n_batch=2048` and physical batch `n_ubatch=512`. [`common.h`:403–407](https://github.com/ggml-org/llama.cpp/blob/master/common/common.h#L403-L407)

- **SOURCE** — `-b` bounds the logical prompt batch; `-ub` is the physical compute chunk. `process_ubatch()` builds or reuses one graph and calls graph compute once; the caller repeats that process until the logical batch is exhausted. [`llama-context.cpp`:1216–1272](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-context.cpp#L1216-L1272), [`llama-context.cpp`:1643–1664](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-context.cpp#L1643-L1664), [`llama-context.cpp`:1796–1798](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-context.cpp#L1796-L1798)

- **INFERENCE** — for a single long sequence, each full physical chunk therefore presents `M=n_ubatch` to its projection matmuls; increasing `-b` without increasing `-ub` does not make those matmuls use `M=n_batch`.

- **SOURCE** — one Hexagon `graph_compute` queues all supported graph nodes and then flushes/waits. Each ubatch thus produces one graph compute and one or more `dspqueue` batches, depending on `fit_op`; separate ubatches do not remain pending across that final graph flush. [`ggml-hexagon.cpp`:3336–3400](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L3336-L3400)

- **SOURCE** — `GGML_HEXAGON_OPBATCH` defaults to 1024 **operations**, while `OPQUEUE=16` is the maximum pending batch depth. They are not token batch sizes. `fit_op` limits:

  - operation count;
  - distinct tensor descriptors;
  - distinct backing buffers;
  - summed mapped-buffer virtual-address space.

  It does **not** test kernel VTCM scratch, DMA rings, or HMX tiles; those are tested by the per-op planners. [`ggml-hexagon.cpp`:57–80](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L57-L80), [`ggml-hexagon.cpp`:1041–1095](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L1041-L1095), [`ggml-hexagon.cpp`:1194–1224](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L1194-L1224)

- **SOURCE** — `NHVX` controls selected worker/HVX counts. `NHMX` is effectively an enable/disable switch: nonzero exposes all hardware-reported HMX units; it is not a requested HMX count. [`ggml-hexagon.cpp`:1593–1611](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L1593-L1611)

- **SOURCE** — HMX matmul eligibility requires:

  - HMX enabled and selected;
  - `M>4`;
  - `K%32==0`;
  - padded `N%32==0`;
  - supported weight type and compatible contiguity/batching.

  For repacked quantized weights, `N` is padded to 32, so original `N%32==0` is not a universal requirement; it matters directly for unrepacked F16/F32 weights. [`ggml-hexagon.cpp`:1990–2031](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L1990-L2031), [`ggml-hexagon.cpp`:2278–2314](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L2278-L2314)

- **SOURCE** — HMX double-buffering begins at `M>32`; HMX FlashAttention is selected separately, requires F16 K/V and `DK,DV` multiples of 64, and deliberately falls back to HVX for fewer than five query rows when `DK<=128`. [`matmul-ops.h`:247–262](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/htp/matmul-ops.h#L247-L262), [`ggml-hexagon.cpp`:1752–1784](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L1752-L1784)

- **SOURCE** — graph caching removes host-side fusion and parameter planning on a repeated graph UID. It is a single cached graph slot per session, not a general multi-shape plan map. [`ggml-hexagon.cpp`:260–263](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L260-L263), [`ggml-hexagon.cpp`:3341–3388](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L3341-L3388)

#### How throughput scales with ubatch

- **SOURCE** — the HMX planner searches `m_chunk` and `n_chunk` under the VTCM layout, minimizing a block-reload cost and breaking ties toward larger tiles. The layout accounts for weight tiles, activation tiles, F32 conversion scratch, output tiles, dequantization scratch, scales, and optional pipeline buffers. [`matmul-ops.h`:102–150](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/htp/matmul-ops.h#L102-L150), [`matmul-ops.h`:273–305](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/htp/matmul-ops.h#L273-L305)

- **INFERENCE** — for a fixed prompt length `T` and ubatch `U`, a matmul’s approximate repeated weight-pass count is:

  `ceil(T/U) × ceil(U/m_chunk)`.

  While `U <= m_chunk`, increasing `U` reduces graph submissions and repeated weight loading nearly proportionally. Once `U > m_chunk`, the second term grows and the product approaches `T/m_chunk`; further gains flatten because VTCM has fixed the largest reusable row tile. Compute occupancy, HMX 32-row tiling, DMA overlap, and attention then dominate.

- **SOURCE** — FlashAttention saturation is independent of the matmul breakpoint: it chooses `Br/Bc`, thread count, and pipelining from query length, accumulated KV length, head dimensions, and VTCM. [`ggml-hexagon.cpp`:1837–1878](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L1837-L1878)

- **INFERENCE** — there is therefore no universal “512 saturates” or “1024 saturates” rule. The crossover varies by projection `(K,N)`, quant type, device VTCM, thread count, and the attention context length. A controlled curve is required.

#### Plausibility

- **INFERENCE** — on the operator-assumed v81/S26+ mapping, **3300 tok/s is plausible as a target**, because a larger Llama-3.2-1B achieved 4027.72 tok/s. It is not a safe lower bound: Qwen graph structure, QK norm, operator coverage, fallback splits, prompt length, and ubatch configuration differ.

- **INFERENCE** — on the assumed v79/S25+ mapping, the claim is materially weaker. The same PR’s S25+ values are 1488.15 for Gemma-4-E2B and 984.92 for Qwen3.5-2B. Inverse parameter scaling could put a 0.6B model above 3300, but dispatch, attention, and bandwidth do not scale linearly with parameter count.

**Verdict — UNVERIFIABLE.** `≥3300 tok/s` is a reasonable v81 optimization target, not a published or defensible “should be at least” result. It is especially unproven for v79. The supplied S25+/Qwen3.5 comparator is wrong, and no controlled ggml-hexagon ubatch curve was found.

---

### A2 — Correct current prefill comparison

- **SOURCE** — this repository defines TTFT as tokenize + EOS padding + prefill + LM head at `seq_len=2048`. Its June 2026 table reports:

  - Llama-3.2-1B bf16: **1.21 s TTFT**, **12.2 tok/s decode**.
  - Qwen3-0.6B bf16: **1.52 s TTFT**, **11.7 tok/s decode**.
  - Power mode is not recorded in that table. [llms/README.md:33–60](/home/cj/mlir-air/programming_examples/llms/README.md:33)

- **SOURCE/ARITHMETIC** — padded-work rates are:

  - Llama: `2048 / 1.21 = 1692.6 padded tok/s`.
  - Qwen: `2048 / 1.52 = 1347.4 padded tok/s`.

- **SOURCE/ARITHMETIC** — the current Hexagon Llama headline is `4027.72 / 1692.6 = 2.38×` faster by those two headline rates, not 10×. Decode is `54.22 / 12.2 = 4.44×`. [PR #25085](https://github.com/ggml-org/llama.cpp/pull/25085)

- **INFERENCE** — even the 2.38× figure is not controlled:

  - XDNA2 uses bf16; Hexagon uses Q4_0.
  - XDNA2’s denominator is 2048 padded tokens; Hexagon evaluated 766 prompt tokens.
  - XDNA2 TTFT includes tokenize, padding, and LM head; llama.cpp’s reported prompt-eval timer is narrower.
  - XDNA2 power mode is unknown.

- **INFERENCE** — there is no same-model current Hexagon result for Qwen3-0.6B. If it reached 3300 tok/s, that would be `3300 / 1347.4 = 2.45×` the repository’s padded-work rate, but this is a target ratio, not a comparison.

- **SOURCE** — the current working doc already acknowledges that its first 169 tok/s comparison was stale, but then incorrectly says all new PR measurements used ubatch 1024 and misassigns Qwen3.5’s 1300 result to S25+. [doc 55:204–223](/home/cj/mlir-air/docs/plans/transformer-layer-execution-studies/55-hexagon-llama-cpp-lessons-for-xdna2.md:204)

**Verdict — STALE.** Replace “prefill here is already ~10× theirs” with: “Current Hexagon HMX Llama-1B reports about **2.4×** this repository’s padded-work headline rate and **4.4×** its decode rate, but the comparison is not controlled.”

---

### A3 — Audit of doc 55 §4 mechanisms, items 1–9

#### 1. Persistent generic kernels

- **SOURCE** — upstream ships CPU `libggml-hexagon` plus architecture-specific persistent HTP shared libraries selected at runtime. [developer.md:3–21](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/developer.md#L3-L21)

- **INFERENCE** — the structural contrast with XDNA2 is valid: the HTP library interprets runtime shape descriptors, whereas this repository’s kernel objects bake tile dimensions and its fused mapping is build-time selection plus a shape-keyed artifact cache. [gemm_builder.py:11–44](/home/cj/mlir-air/programming_examples/llms/shared/builders/gemm_builder.py:11), [53-workload-dependent-mapping.md:576–581](/home/cj/mlir-air/docs/plans/transformer-layer-execution-studies/53-workload-dependent-mapping.md:576)

- **UNVERIFIABLE** — “26 ops” is not a durable architectural fact and should not be retained as a headline count.

**Item verdict — CONFIRMED in mechanism; the exact op count is brittle.**

#### 2. Fusion, parameter precompute, graph caching, op batching

- **SOURCE** — MUL_MAT and FLASH_ATTN parameters are now computed on the host and retained with the cached graph. PR #25085 explicitly says this for FA; current master does it for MUL_MAT, FA, and unary nodes. [PR #25085 overview](https://github.com/ggml-org/llama.cpp/pull/25085), [`ggml-hexagon.cpp`:3350–3388](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L3350-L3388)

- **SOURCE** — “every fusion is guarded by VTCM” is wrong:

  - quantized, non-HMX QKV and gate/up FFN fusions have VTCM guards;
  - MUL_MAT+ADD has eligibility and VTCM guards;
  - RMS_NORM+MUL does not perform an explicit VTCM-budget rejection at this fusion site.

  [`ggml-hexagon.cpp`:3198–3215](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L3198-L3215), [`ggml-hexagon.cpp`:3252–3329](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L3252-L3329)

- **SOURCE** — the document’s “16 buffers” is not established by `fit_op`; the code uses `HTP_OP_MAX_BUFS`. Its hardcoded 8192-tensor phrasing is also too specific: tensor capacity is `min(n_ops+n_ops*MAX_INPUTS, HTP_OP_MAX_TENSORS)`. [`ggml-hexagon.cpp`:1075–1093](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L1075-L1093)

- **INFERENCE** — “a 16-layer decoder is one or two writes” is possible but not guaranteed. Mapped-buffer count/size and tensor-descriptor reuse decide it.

**Item verdict — PARTLY STALE and overstated.**

#### 3. Resident/repacked weights and KV

- **SOURCE** — repacking occurs in `buffer_set_tensor`, not `init_tensor`. Q4_0, Q4_1, Q8_0, IQ4_NL, and MXFP4 are stored in the 32×32 tiled layout; other types are copied unchanged. [`ggml-hexagon.cpp`:840–881](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L840-L881)

- **SOURCE** — the stored quantized-weight representation is now tiled. The “flat” HVX variants refer to the dynamically quantized activation layout/kernel path, not an alternate unrepacked weight allocation; supported quant matmuls require the weight buffer to be a repack buffer. [`ggml-hexagon.cpp`:2506–2527](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L2506-L2527), [`ggml-hexagon.cpp`:2529–2560](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L2529-L2560)

- **SOURCE** — “resident” means mapped shared DDR, not permanently resident VTCM. VTCM is temporary storage for activation quantization and DMA-fetched weight chunks. [developer.md:23–33](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/developer.md#L23-L33)

- **SOURCE** — KV buffers can be allocated on HTP, but their dtype is configurable; the official example uses Q8_0 K/V. [developer.md:67–80](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/developer.md#L67-L80)

**Item verdict — PARTLY CONFIRMED; `init_tensor`, “VTCM resident,” and fixed KV dtype are wrong.**

#### 4. Analytical capacity planner

- **SOURCE** — the legality cascade is real: HMX first when eligible, then HVX. HMX searches a capacity-feasible `(m_chunk,n_chunk)`; HVX separately tries the deepest feasible prefetch/tiled activation layout and falls back to flat activation layout. [`ggml-hexagon.cpp`:2035–2098](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L2035-L2098), [`ggml-hexagon.cpp`:2483–2527](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L2483-L2527)

- **SOURCE** — the “largest prefetch 16→2 that fits” description applies to the HVX planner. HMX uses a different chunk solver and optional double buffering.

- **INFERENCE** — describing this as a deterministic traffic/capacity planner rather than an autotuner is accurate. Describing a universal ubatch saturation point is not.

**Item verdict — CONFIRMED after separating the HMX and HVX planners.**

#### 5. DMA, threading, activation quantization, graph reorder

- **SOURCE** — HVX quant matmuls dynamically convert F32 activations to Q8_0 or Q8_1 in VTCM; Q4_1 selects Q8_1 and the other quantized weight formats select Q8_0. [`matmul-ops.c`:779–816](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/htp/matmul-ops.c#L779-L816), [`matmul-ops.c`:1377–1427](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/htp/matmul-ops.c#L1377-L1427)

- **SOURCE** — HMX does **not** use this Q8 activation path. Quantized weights are dequantized to FP16 tiles, while F32 activations are converted to FP16 tiles. [`matmul-ops.c`:1742–1779](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/htp/matmul-ops.c#L1742-L1779), [`matmul-ops.c`:1881–1917](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/htp/matmul-ops.c#L1881-L1917)

- **SOURCE** — graph reorder intentionally places same-input matmuls near one another to reuse dynamically prepared activations, but current source contains a warning that the reorder may violate dependencies in some cases. [`ggml-hexagon.cpp`:3412–3453](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L3412-L3453)

**Item verdict — WRONG as written for HMX; correct for the HVX quant path.**

#### 6. Prefill/decode threshold and attention

- **SOURCE** — `M>4` versus `M<=4` is a matmul-family choice, not a backend-wide mode. FlashAttention makes its own HMX/HVX decision; at M=1 and `DK<=128`, HMX attention is rejected in favor of HVX. [`ggml-hexagon.cpp`:1752–1784](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L1752-L1784)

- **SOURCE** — the backend can run `FLASH_ATTN_EXT` for a single query row, but only when its dtype, shape, layout, and VTCM predicates pass. It requires F16 K/V. [`ggml-hexagon.cpp`:1897–1937](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L1897-L1937)

- **INFERENCE** — “decode attention is on the NPU” is therefore conditional, not universal. It is true for a fully supported F16-KV graph assigned to HTP; a Q8 KV cache or unsupported adjacent node can create a fallback split.

**Item verdict — PARTLY CONFIRMED; the blanket statement is wrong.**

#### 7. Predicate-based partial offload

- **SOURCE** — operation support is explicitly predicate-based. MUL_MAT, FA, norms, softmax, layouts, and dtypes each have refusal conditions; unsupported nodes are left for another scheduler backend. [`ggml-hexagon.cpp`:2529–2593](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L2529-L2593), [`ggml-hexagon.cpp`:2695–2716](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L2695-L2716)

- **INFERENCE** — the transferable lesson is incomplete unless the planner records split count and transfer cost. ggml’s fallback preserves functionality; it does not make the split free.

**Item verdict — CONFIRMED.**

#### 8. Benchmark interpretation

- **SOURCE** — the old 169 tok/s HVX number is stale for current HMX comparison. PR #25085’s current Llama result is 4027.72 prefill/54.22 decode on S26+. [PR #25085](https://github.com/ggml-org/llama.cpp/pull/25085)

- **INFERENCE** — “decode barely moved, therefore weight-stream-bound” is plausible but not proven by comparing different devices, dates, prompts, and software revisions. It should be presented as a hypothesis supported by sensitivity, not a direct source conclusion.

**Item verdict — STALE for prefill; decode diagnosis is an inference.**

#### 9. QNN history

- **SOURCE** — PR #12326 was closed, included a per-op QNN approach and a direct cDSP approach, and explicitly described whole-cgraph QNN mapping as a separate approach beyond that PR’s scope. [PR #12326:195–200, 254–267](https://github.com/ggml-org/llama.cpp/pull/12326)

- **SOURCE** — the PR author reported poor QNN performance and attributed preference for direct cDSP to experiments and Qualcomm guidance, but it did not establish the precise causal decomposition “finalize + FastRPC + binding exceeded compute for all non-matmuls.” [PR #12326:419–466](https://github.com/ggml-org/llama.cpp/pull/12326)

- **INFERENCE** — the spectrum analogy to this study is useful, but QNN/Genie should not be presented as an experimentally rejected equivalent of the current XDNA2 modes.

**Item verdict — OVERSTATED and partly UNVERIFIABLE.**

**Overall A3 verdict — WRONG as written.** The broad architecture is recognizable, but items 2, 3, 5, 6, 8, and 9 contain material inaccuracies or overgeneralizations.

---

### A4 — Exact mixed-precision runtime contract

#### Weights

- **SOURCE** — Q4_0, Q4_1, Q8_0, IQ4_NL, and MXFP4 weights are repacked once into padded 32×32 tiled buffers in shared DDR. F16/F32 weights are HMX-supported but are not placed in that quant repack buffer format. [`ggml-hexagon.cpp`:189–196](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L189-L196), [`ggml-hexagon.cpp`:455–496](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L455-L496), [`ggml-hexagon.cpp`:840–881](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L840-L881)

#### Matmul activations and output

- **SOURCE** — the external MUL_MAT contract accepts F32 or F16 activation input and requires F32 destination. [`ggml-hexagon.cpp`:2529–2539](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L2529-L2539)

- **SOURCE** — on the HVX quant path:

  - activation arrives as F32 in mapped DDR;
  - it is dynamically quantized into VTCM;
  - Q4_1 weights use Q8_1 activations;
  - Q4_0, Q8_0, IQ4_NL, and MXFP4 use Q8_0 activations;
  - tiled or flat refers to the activation’s VTCM layout.

- **SOURCE** — on the HMX path:

  - quantized weight chunks are DMA’d and dequantized to FP16 tiles in VTCM;
  - F32 activation chunks are converted to FP16 tiles;
  - the HMX instructions use the F16 accumulator/store path;
  - the FP16 output tile is converted to the F32 global destination.

  [`matmul-ops.c`:1742–1779](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/htp/matmul-ops.c#L1742-L1779), [`hmx-mm-kernels-tiled.h`:632–673](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/htp/hmx-mm-kernels-tiled.h#L632-L673), [`hmx-mm-kernels-tiled.h`:681–706](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/htp/hmx-mm-kernels-tiled.h#L681-L706)

#### Norms, activations, softmax

- **SOURCE** — ordinary unary/norm and activation nodes require F32 input/output. Ordinary softmax requires F32 logits/output; its optional mask may be F32 or F16. [`ggml-hexagon.cpp`:2695–2716](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L2695-L2716), [`ggml-hexagon.cpp`:2740–2804](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L2740-L2804)

#### Attention and KV

- **SOURCE** — `FLASH_ATTN_EXT` accepts Q as F16 or F32, requires K/V as F16, accepts an F16 mask and optional F32 sinks, and writes F16 or F32. [`ggml-hexagon.cpp`:1897–1920](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L1897-L1920)

- **SOURCE** — PR #25085 moved FA parameter computation host-side and changed softmax accumulation to FP32. [PR #25085 overview](https://github.com/ggml-org/llama.cpp/pull/25085)

- **SOURCE** — KV cache is **not intrinsically F16**. It is a configurable model/runtime choice; upstream demonstrates HTP-resident Q8_0 K/V. That format cannot directly satisfy this backend’s current `FLASH_ATTN_EXT` F16-K/V predicate. [developer.md:67–80](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/developer.md#L67-L80)

#### Transferable template

- **INFERENCE** — the accurate template is not simply “int4 weights × int8 activations everywhere.” It is:

  - compact repacked weights resident in device-visible DDR;
  - an op-specific preparation path in scratchpad;
  - HVX quant matmul: F32→Q8 activation quantization;
  - HMX matmul: quant-weight→FP16 and F32-activation→FP16 tiling;
  - F32 model-visible residual/norm/softmax boundaries;
  - an explicitly chosen KV dtype whose attention kernel supports it.

- **INFERENCE** — XDNA2 should copy the separation of **storage type**, **scratchpad compute type**, **accumulator type**, and **model-visible output type**, not blindly copy Hexagon’s FP16/Q8 choices; AIE2P’s natural floating-point path here is bf16.

**Verdict — WRONG as a blanket “int4×int8” description; the corrected per-path contract above is CONFIRMED.**

## 2. Part B architecture proposal

### B1 — Analytical inference planner

#### Architectural boundary

- **SOURCE** — this repository already has:

  - shape-parameterized multi-launch builders; [builders/__init__.py:4–10](/home/cj/mlir-air/programming_examples/llms/shared/builders/__init__.py:4)
  - compiled artifact, XRT-context, instruction BO, and transient BO caching; [cache.py:316–352](/home/cj/mlir-air/programming_examples/llms/shared/infra/cache.py:316)
  - shared-xclbin/context attachment; [cache.py:462–480](/home/cj/mlir-air/programming_examples/llms/shared/infra/cache.py:462), [cache.py:567–637](/home/cj/mlir-air/programming_examples/llms/shared/infra/cache.py:567)
  - resident static-weight BO reuse; [cache.py:656–675](/home/cj/mlir-air/programming_examples/llms/shared/infra/cache.py:656)
  - runlist planning and dispatch-vector accounting. [dispatch.py:219–304](/home/cj/mlir-air/programming_examples/llms/shared/infra/dispatch.py:219), [dispatch.py:374–412](/home/cj/mlir-air/programming_examples/llms/shared/infra/dispatch.py:374)

- **INFERENCE** — do not build a second artifact/context/BO runtime. Add a new pure planner above these facilities.

#### Proposed modules

1. **`ModelGraph`**

   - **INFERENCE** — a small typed DAG, not a full ggml clone.
   - Tensor fields: ID, symbolic/concrete shape, logical dtype, packed storage dtype, compute dtype, accumulator dtype, layout, lifetime, mutability, and storage class (`weight`, `activation`, `kv_state`, `scratch`).
   - Node fields: semantic op, inputs/outputs, attributes, phase predicate, repeated-layer index.
   - Represent the transformer as a repeated block template plus embedding, final norm, LM head, and recurrent KV state.

2. **`DeviceCaps`**

   - **SOURCE** — XDNA2 capacity is spatial: 8 columns × 4 core rows, two shim channels per direction per column, six memtile channels per direction, two core DMA channels per direction, 64 KiB L1 per core, and 512 KiB memtile storage per column. [mapping_space.py:140–189](/home/cj/mlir-air/programming_examples/transformer_layer/study/mapping_space.py:140)

   - **INFERENCE** — model L2 as eight separate 512 KiB resources, not one fungible 4 MiB pool. Include placement, route, channel, BD, stride, object-code, and context constraints.

3. **`KernelCandidate` providers**

   Each provider implements pure:

   - `supports(op, shape, dtype_plan, caps)`;
   - `resources(...) -> spatial demand`;
   - `estimated_cost(...) -> bytes, launches, calibrated cycles`;
   - `artifact_key(...)`;
   - `lower(...) -> existing builder call`.

   Candidate families include GEMM/GEMV variants, attention, norm/glue, packed quant, and host fallback.

4. **Legality/cost pipeline**

   - Normalize shapes and layouts.
   - Enumerate per-op candidates.
   - Enumerate local fusion candidates.
   - Check L1/L2 liveness, BD/channel demand, placement cardinality, routing invariants, stride legality, and external-object compatibility.
   - Select device/host segments and charge transfers.
   - Run global buffer-liveness/residency planning.
   - Form artifacts and dispatch groups.
   - Emit a fully resolved plan plus rejection/provenance messages.

- **SOURCE** — the existing study already distinguishes static “refusal” from routable-but-degraded “price” and warns that per-column placement cannot be reduced to a scalar budget. [mapping_space.py:9–99](/home/cj/mlir-air/programming_examples/transformer_layer/study/mapping_space.py:9)

- **INFERENCE** — fusion legality must therefore be a spatial resource composition, not merely “sum live bytes ≤ L1/L2.” Sequentially reused resources combine with `max`; simultaneously resident stages combine with sum; per-column route/channel conflicts remain explicit.

#### Registry policy

- **SOURCE** — `gemm_config` is exact-shape lookup and deliberately raises `KeyError` rather than guessing an unmeasured tile. [registry_lookup.py:67–120](/home/cj/mlir-air/programming_examples/kernel_registry/registry_lookup.py:67)

- **INFERENCE** — keep the registry as measured overrides/calibration, not as the planner itself.

- **INFERENCE** — “derived on miss” should initially mean:

  1. derive legal candidates with pure builder constraints;
  2. mark them `analytical_unmeasured`;
  3. require an explicit policy to compile/use them;
  4. never write them back as “best” until swept and verified.

#### Plan and cache key

- **INFERENCE** — emit:

  - selected kernel family and exact artifact key per node;
  - fusion groups;
  - placement/resource ledger;
  - static/resident BO specifications;
  - KV ownership/layout;
  - ordered dispatch steps and submission groups;
  - device/host split count and bytes;
  - all rejected alternatives and reasons;
  - prediction source: measured registry, analytical, or forced override.

- **INFERENCE** — cache plans by model/checkpoint, phase, query-M bucket, KV-context bucket, dtype plan, caps, planner version, registry revision, and toolchain identity. Keep this separate from the existing compiled-artifact cache.

#### Fallback policy

- **INFERENCE** — support three outcomes:

  - `device`;
  - `host`, with explicit split/transfer charge;
  - `refuse`.

  Study configurations should default to `refuse_on_unplanned_split`; production may allow host fallback.

#### Planner phases

- **P0 — pure legality extraction.** Lift builder/profile constraints into pure candidate providers; reproduce current derived skips and exact registered configurations. Gate with host-only tests.

- **P1 — two-model bf16 graphs.** Add Qwen3-0.6B and Llama-3.2-1B adapters; emit plans matching their current hand-built boundaries. Gate by manifest diff and compile-only construction.

- **P2 — fusion/buffer/dispatch planning.** Lower selected groups into existing builders, BO pools, shared xclbins/ELFs, and runlists. Gate each change with production `make verify`.

- **P3 — mixed precision.** Add packed-storage, dequant/activation-quant preparation, accumulator/output contracts, and KV precision candidates. Calibrate costs from measurements.

- **P4 — broader model coverage.** Generalize to the remaining standing deployments only after the two-model planner is stable.

### B2 — Full-model mixed-precision study

#### Preserve the layer study

- **SOURCE** — the current study’s contract is a single transformer layer and four execution modes; its success criteria concern layer output, dispatch-vector separation, manifests, and explicit comparison adapters. [00-context-and-goals.md:3–7](/home/cj/mlir-air/docs/plans/transformer-layer-execution-studies/00-context-and-goals.md:3), [00-context-and-goals.md:96–109](/home/cj/mlir-air/docs/plans/transformer-layer-execution-studies/00-context-and-goals.md:96)

- **INFERENCE** — add a `model` measurement scope; do not redefine the four layer modes as precision plans. Execution boundary and numerical precision are orthogonal axes.

#### Full-model runner contract

- **INFERENCE** — `run_model.py` should call a narrow production adapter:

  - `prepare(model, precision_plan, compiled_shapes)`;
  - `prefill(token_ids, ubatch_policy, state)`;
  - `decode(state, n_tokens)`;
  - `dispatch_vector(scope)`;
  - `verify_against_hf(...)`.

- **SOURCE** — the current runners are imperative, fixed-shape drivers rather than such an interface. Qwen prefill allocates a host bf16 KV cache, runs all layers over one `seq_len`, and extracts every layer’s K/V afterward. [qwen3_0_6b_inference.py:335–413](/home/cj/mlir-air/programming_examples/llms/qwen3_0_6b/qwen3_0_6b_inference.py:335)

- **SOURCE** — today’s Qwen driver pads every short prompt to the session’s compiled length. [qwen3_0_6b_inference.py:637–648](/home/cj/mlir-air/programming_examples/llms/qwen3_0_6b/qwen3_0_6b_inference.py:637)

- **INFERENCE** — ubatch chunking is therefore not “remove padding and loop.” Correct incremental prefill needs:

  - chunk-outer/layer-inner scheduling;
  - per-layer KV append;
  - each chunk attending to all earlier chunks;
  - position-correct RoPE and mask;
  - rectangular `Lq × Lk` attention artifacts;
  - verification that padded tail rows cannot affect valid rows.

#### Study phases

- **S0 — model-schema baseline.** Run current fixed-2048 bf16 Qwen and Llama through the new model adapter. Preserve manifests, resume, failure rows, power-mode provenance, dispatch vector, and HF verification. The existing layer runner already keeps setup outside timing and writes failures as complete rows. [run_mode.py:4–27](/home/cj/mlir-air/programming_examples/transformer_layer/study/run_mode.py:4), [run_mode.py:281–369](/home/cj/mlir-air/programming_examples/transformer_layer/study/run_mode.py:281)

- **S1 — valid ubatch prefill.** Implement incremental causal prefill for one model. Hold logical prompt and token IDs fixed while varying only physical ubatch. Report both TTFT and prefill-only tok/s.

- **S2 — decode curve.** Measure a warmed token window at controlled starting contexts, e.g. 128/512/1024/2048. Record per-token totals, per-layer normalization, splits, KV dtype/layout, and whether attention/glue is host or device.

- **S3 — mixed precision.** First ingest and measure the already-present Llama int4-decode production path; then add a second model and on-device activation preparation. Use HF token-set inclusion as the production gate and per-layer/logit diagnostics to localize drift.

- **S4 — bounded matrix.** Expand only selected `(model, phase, ubatch/context, precision_plan)` cells. Require a prediction before each kernel/design experiment.

#### Measurement discipline

- **SOURCE** — schema v2 already contains separate quant packing, group, scale, zero-point, accumulator, GEMM, and GEMV fields. [schema.py:257–274](/home/cj/mlir-air/programming_examples/transformer_layer/study/schema.py:257)

- **SOURCE** — the ladder already provides one subprocess per rung, derived skips, incremental row persistence, and reuse/resume. [run_ladder.py:4–37](/home/cj/mlir-air/programming_examples/transformer_layer/study/run_ladder.py:4), [run_ladder.py:263–320](/home/cj/mlir-air/programming_examples/transformer_layer/study/run_ladder.py:263)

- **SOURCE** — manifests already record repository/platform/toolchain coverage and fail on incomplete coverage. [03-measurement-model.md:437–441](/home/cj/mlir-air/docs/plans/transformer-layer-execution-studies/03-measurement-model.md:437)

- **INFERENCE** — retain those mechanics, but key resume/comparison on model, checkpoint, phase, logical token count, ubatch/context bucket, and precision plan—not only `(case, mode, seq_len)`.

## 3. H0–H4 critique

### H0

**Correct**

- **INFERENCE** — a small typed model graph, host-only legality planner, registry overrides, resource-driven fusion, buffer residency, dispatch grouping, and explicit fallback are the right components.
- **SOURCE** — one compiled shape may be reused across all identical layers with different weight BO keys; current Qwen prepares weights once and runs 28 layers through repeated artifacts. [Qwen ARCHITECTURE.md:65–80](/home/cj/mlir-air/programming_examples/llms/qwen3_0_6b/ARCHITECTURE.md:65)

**Wrong or missing**

- **SOURCE** — “the 8 standing models” is stale. The top-level table currently contains ten deployments including the int4 entry, with nine performance rows. [llms/README.md:8–21](/home/cj/mlir-air/programming_examples/llms/README.md:8)

- **SOURCE** — derived-on-miss is not current registry policy; current lookup intentionally refuses unmeasured shapes. [registry_lookup.py:87–120](/home/cj/mlir-air/programming_examples/kernel_registry/registry_lookup.py:87)

- **INFERENCE** — “GEMM versus GEMV by M” must be candidate/cost driven. Hexagon’s `M>4` threshold is specific to HMX/HVX, not an XDNA2 architectural law.

- **INFERENCE** — “live set fits L1/L2 + BD budget” omits per-column L2, shim/memtile/core channel topology, object-code compatibility, placement, routing, and stride constraints.

- **SOURCE** — a runlist submission is not the same as one hardware context. With ELF artifacts, current dispatch may aggregate runs whose artifacts each have their own contexts; one shared context requires a shared xclbin/configuration. [dispatch.py:219–267](/home/cj/mlir-air/programming_examples/llms/shared/infra/dispatch.py:219), [cache.py:567–637](/home/cj/mlir-air/programming_examples/llms/shared/infra/cache.py:567)

- **INFERENCE** — “weights resident” must mean DDR BO resident/reused, not L1/L2 resident.

**Over-ambitious**

- **INFERENCE** — reproducing every model, every skip, automatic tiling, fusion, dispatch, and fallback in H0 is too broad. Start with bf16 Qwen/Llama, exact registry entries, and manifest-equivalent current plans. Host-only tests cannot validate routing success or performance ranking.

### H1

**Correct**

- **INFERENCE** — full-model results, manifests, resume, comparison, phase identity, ubatch, tok/s, split counts, and HF gates belong here.

**Wrong or missing**

- **SOURCE** — the quant columns are already present; H1 should populate them, not introduce a second quant schema. [schema.py:50–54](/home/cj/mlir-air/programming_examples/transformer_layer/study/schema.py:50), [schema.py:257–274](/home/cj/mlir-air/programming_examples/transformer_layer/study/schema.py:257)

- **INFERENCE** — “runner over llms drivers” needs an adapter seam first; the current drivers hardcode padding, cache extraction, and phase structure.

- **SOURCE** — current Qwen FlashAttention compilation is square: it passes the same `seq_len` as `lk` and `lq`. [fa_headfirst.py:53–94](/home/cj/mlir-air/programming_examples/llms/shared/infra/fa_headfirst.py:53)

- **INFERENCE** — ubatch-chunked prefill therefore requires incremental KV and rectangular attention, not just multiple invocations of today’s driver.

**Over-ambitious**

- **INFERENCE** — split H1 into H1a fixed-shape model measurement and H1b semantically correct ubatching. Otherwise schema/runtime/attention failures become inseparable.

### H2

**Correct**

- **SOURCE** — the int4 prefill wall is accurately documented: `herd_m=2` at K=8192 and `tile_n=16` due to the Peano immediate range; the measured int4 path is about 8× slower. [llama32_1b_int4/README.md:13–30](/home/cj/mlir-air/programming_examples/llms/llama32_1b_int4/README.md:13)

- **INFERENCE** — a written prediction before measurement is the correct gate.

**Wrong or stale**

- **SOURCE** — “int4 GEMV decode not done” is stale. Current source explicitly implements bf16 NPU prefill followed by NPU int4 decode, and the shared verify adapter calls it the production path. [llama32_1b_int4_inference.py:4–20](/home/cj/mlir-air/programming_examples/llms/llama32_1b_int4/llama32_1b_int4_inference.py:4), [verify_adapter.py:183–204](/home/cj/mlir-air/programming_examples/llms/llama32_1b_int4/verify_adapter.py:183)

- **SOURCE** — its Makefile wires full prefill+decode verification and reports roughly **17.8 tok/s** int4 decode. [llama32_1b_int4/Makefile:4–20](/home/cj/mlir-air/programming_examples/llms/llama32_1b_int4/Makefile:4), [llama32_1b_int4/Makefile:157–176](/home/cj/mlir-air/programming_examples/llms/llama32_1b_int4/Makefile:157)

- **INFERENCE** — compared with the bf16 table’s 12.2 tok/s, the published local headline is about 1.46×, not 3–4×. This is not a controlled comparison—checkpoint and conditions differ—but it refutes treating 3–4× as already evidenced.

- **INFERENCE** — an ideal 4-bit payload is 4× narrower than bf16, not “halved”; scales, zero points, alignment, dequantization, and non-weight work reduce the end-to-end gain. The int4 README’s “halved footprint” sentence is itself inaccurate. [llama32_1b_int4/README.md:32–33](/home/cj/mlir-air/programming_examples/llms/llama32_1b_int4/README.md:32)

**Over-ambitious**

- **INFERENCE** — auditing existing int4 decode, repairing int4 prefill GEMM, and adding on-device activation quantization are three separate projects. Make them H2a/H2b/H2c.

### H3

**Correct**

- **SOURCE** — current bf16 deployments leave decode attention and M=1 glue on CPU because their individual dispatch cost exceeds their compute. [llms/README.md:62–69](/home/cj/mlir-air/programming_examples/llms/README.md:62)

- **INFERENCE** — after dispatch amortization and narrower weights, bringing this work back on-device is a valid objective.

**Wrong or missing**

- **SOURCE** — current Qwen decode is two NPU ELFs per layer plus one LM-head invocation, with host KV write and CPU attention between them. [Qwen ARCHITECTURE.md:33–42](/home/cj/mlir-air/programming_examples/llms/qwen3_0_6b/ARCHITECTURE.md:33)

- **INFERENCE** — a runlist can reduce host submissions without creating one image or one context. “One submit,” “one context,” and “one kernel image” are distinct targets.

- **INFERENCE** — on-device attention requires a device-owned KV layout, update semantics, context-length parameterization, and a compatible shared array configuration. It cannot be added as isolated M=1 glue.

**Over-ambitious**

- **INFERENCE** — stage it:

  1. re-execute one decode projection artifact across layers;
  2. aggregate the existing two projections per layer;
  3. aggregate all layers;
  4. move KV update/attention;
  5. add glue and LM head.

### H4

**Correct**

- **INFERENCE** — Qwen3-0.6B and Llama-3.2-1B are the right first pair: one has QK norm/head_dim 128, the other is the reference Llama/head_dim 64 path.

**Wrong or missing**

- **INFERENCE** — the matrix needs prompt length, physical ubatch, starting/ending KV context, power mode, cold/warm state, fallback policy, and verification contract—not just model × ubatch × precision.

- **SOURCE** — power mode is a gating condition: the study measured approximately 3.7 ms/context load at Turbo versus about 82 ms at Default on one rung. [32-cost-decomposed-ladder.md:53–64](/home/cj/mlir-air/docs/plans/transformer-layer-execution-studies/32-cost-decomposed-ladder.md:53)

**Over-ambitious**

- **INFERENCE** — do not launch a Cartesian “full matrix.” Use planner-selected cells plus negative controls; otherwise artifact count grows with model × query-M × KV-length × precision × placement.

## 4. Direct answers

### 1. New planner or generalize `skip_reason`/`derive_rows_per_call`/`gemm_config`?

**Answer: a new planner module, reusing those functions as leaf predicates.**

- **SOURCE** — `skip_reason` currently derives study-mode legality from the one-layer profile matrix; it is not a model graph, buffer, placement, or dispatch planner. [profiles.py:310–369](/home/cj/mlir-air/programming_examples/transformer_layer/study/profiles.py:310)

- **SOURCE** — `gemm_config` is measured exact-shape selection, not analytical synthesis. [registry_lookup.py:67–120](/home/cj/mlir-air/programming_examples/kernel_registry/registry_lookup.py:67)

- **SOURCE** — the present analytical selector explicitly cannot rank alternatives with different dispatch structures. [analytical_cost.py:394–413](/home/cj/mlir-air/programming_examples/transformer_layer/study/analytical_cost.py:394)

- **INFERENCE** — put pure constraints in reusable leaf modules; let `profiles.skip_reason` and the new planner call them. Do not turn `profiles.py` into a model compiler.

### 2. What is the correct ggml “graph” analogue?

**Answer: a small semantic op graph feeding existing builders, not direct builder calls and not a general-purpose tensor IR.**

- **INFERENCE** — builders are lowering endpoints. Driving them directly loses tensor lifetime, layout, precision, KV state, fusion alternatives, and host/device boundary information.

- **INFERENCE** — the graph should express only supported decoder primitives and repeated block structure. It should remain small enough that every node has an explicit candidate provider and verification contract.

- **INFERENCE** — lower the selected subgraphs into the existing shape-specific multi-launch builders and infrastructure; do not recreate MLIR or ggml inside Python.

### 3. How should ubatch be realized under per-shape compilation?

**Answer: compile a geometric set of query-M buckets plus attention context buckets. Do not port Hexagon’s M=4 threshold.**

- **Recommended steady set — INFERENCE:** `M={1,128,256,512,1024}`.

  - `1`: mandatory decode.
  - `128/256/512/1024`: geometric prefill curve and practical prompt chunking.
  - Add `64` only as a small-tail bucket if padding to 128 proves too costly.
  - Do not compile `M=4` unless speculative decoding or multiple simultaneous decode tokens create an actual workload. It has no special XDNA2 meaning.

- **INFERENCE** — long prompts use repeated largest-supported chunks plus a padded smaller bucket or an explicitly compiled tail. Padding must be masked and excluded from throughput’s logical-token numerator.

- **SOURCE** — Qwen’s current short-M coverage is incomplete. Its builder requires Q `(M,1024,2048)`, O `(M,2048,1024)`, gate/up `(M,1024,3072)`, and down `(M,3072,1024)`. [qwen3_0_6b_prefill.py:51–78](/home/cj/mlir-air/programming_examples/llms/qwen3_0_6b/qwen3_0_6b_prefill.py:51), [qwen3_0_6b_prefill.py:139–144](/home/cj/mlir-air/programming_examples/llms/qwen3_0_6b/qwen3_0_6b_prefill.py:139)

- **SOURCE** — direct registry audit shows short-M gate/up entries, but Q/O/down appear for this model’s dimensions only at M=2048 in the present JSON. [GEMM registry:328–432](/home/cj/mlir-air/programming_examples/kernel_registry/details/GEMM_bf16_in_bf16_out.json:328), [short-M gate entries:4441–4600](/home/cj/mlir-air/programming_examples/kernel_registry/details/GEMM_bf16_in_bf16_out.json:4441)

- **INFERENCE** — attention’s artifact key must include both `Lq=M` and accumulated `Lk`, not just M. For fixed 1024 tokens:

  - ubatch 1024 needs `(Lq,Lk)=(1024,1024)`;
  - ubatch 512 needs `(512,512)` and `(512,1024)`;
  - ubatch 256 adds `(256,256/512/768/1024)`.

  This triangular context grid is the principal artifact-growth problem.

### 4. Minimal schema v3 delta

**Answer: keep layer rows unchanged and add an additive model scope. Do not duplicate quant fields.**

- **SOURCE** — `seq_len` currently means physical projection M, and dispatch fields explicitly mean “per layer.” [schema.py:114–124](/home/cj/mlir-air/programming_examples/transformer_layer/study/schema.py:114), [schema.py:199–219](/home/cj/mlir-air/programming_examples/transformer_layer/study/schema.py:199)

- **INFERENCE** — minimal new fields:

  - `measurement_scope`: `layer|model`
  - `model_id`
  - `phase`: `prefill|decode`
  - `logical_token_count`
  - `ubatch_tokens`
  - `context_start_tokens`
  - `context_end_tokens`
  - `measured_token_count`
  - `tokens_per_second`
  - `precision_plan_id`
  - `model_dispatch_vector_json`: strict `{scope, host_submissions, runlist_entries, air_launches, herd_launches, sync_boundaries, bytes_transferred}`

- **INFERENCE** — reuse:

  - `seq_len` for physical M (`ubatch_tokens` for prefill, 1 for decode);
  - `weights_source` for checkpoint and immutable revision;
  - existing timing, power, quant, outcome, selected-config, provenance, and failure fields.

- **INFERENCE** — leave the old `*_per_layer` dispatch columns null in model rows rather than silently redefining them. Store whole-phase/per-token totals in the strictly validated JSON field.

### 5. First measurable milestone

**Answer: a controlled two-point Qwen3-0.6B bf16 prefill curve at ubatch 512 and 1024 over the same exact 1024-token prompt, followed by the normal decode verification gate.**

Required work:

- **INFERENCE** — add six measured GEMM registry rows: Q, O, and down at M=512 and M=1024. Gate/up and K/V dimensions are already represented.

- **INFERENCE** — extend head-first FA from square `(seq,seq)` to:

  - `(512,512)`;
  - `(512,1024)`;
  - `(1024,1024)`.

- **INFERENCE** — schedule chunks correctly with per-layer KV append and positional masking.

- **INFERENCE** — for each point record:

  - `1024 / prefill_elapsed` tok/s;
  - TTFT;
  - per-chunk timing;
  - dispatch vector and host/device splits;
  - artifact/compile counts;
  - observed Turbo power mode;
  - final logits/top-k against HF;
  - the normal 32-token production token-set verification after prefill.

- **INFERENCE** — this is the smallest experiment that directly tests the operator’s 512-versus-1024 hypothesis without conflating ubatch with prompt length. A sweep where logical prompt length equals M would be useful kernel scaling data, but it would not be an ubatch curve.

### What I would cut

- **INFERENCE** — cut M=4 until there is a multi-token decode workload.
- **INFERENCE** — cut all-model H0 parity; use Qwen0.6 and Llama1B.
- **INFERENCE** — cut automatic unmeasured “derived-on-miss” execution from the first planner.
- **INFERENCE** — cut int4 prefill repair from the initial mixed-precision milestone; first audit and measure the int4 decode path that already exists.
- **INFERENCE** — cut on-device activation quantization until packed-weight decode and its accuracy/performance contract are stable.
- **INFERENCE** — cut “one image/context/submit per token” as a single H3 deliverable; stage submission aggregation, shared configuration, KV residency, and on-device attention separately.
- **INFERENCE** — cut the full Cartesian matrix. Expand only after the two-point Qwen ubatch result validates the runner, schema, rectangular attention, and verify gate.

