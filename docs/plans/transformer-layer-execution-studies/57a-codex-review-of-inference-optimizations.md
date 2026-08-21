# 57a — Codex adversarial review of doc 57

`[2026-08-20]` Produced by Codex (session `01a020c2-6b65-7ae2-afa6-d09177cf35d6`) on the prompt recorded in [57 §7](57-inference-path-optimizations-from-hexagon.md), reproduced **verbatim**. Every point is applied in 57 and marked `[per Codex review]`.

---

# Adversarial review

**Verdict: major revision.** The launch-count diagnosis is directionally persuasive, but **109 µs is not an isolated PDI-boundary measurement**, and the 18–22 ms endpoint rests on overlapping or unsupported assumptions.

## 1. Central mechanism and probe

- **[SOURCE]** The sentence “**every `air.launch` boundary is a partition reconfiguration**” is qualitatively right for these multi-launch ELFs: lowering creates one device per launch, and the main device emits `configure`/`run` pairs ([AIRRtToNpuPass.cpp:1443](/home/cj/mlir-air/mlir/lib/Conversion/AIRRtToNpuPass.cpp:1443)). However, “the runtime sequence inlines … `aiex.npu.load_pdi` between them” is literally stale for ELF. Doc 29 later records that raw in-stream `load_pdi` faults NPU2 firmware; ELF instead carries `.pdi.N`/`.ctrltext.N` sections and the ELF loader resolves reconfiguration ([29:380](/home/cj/mlir-air/docs/plans/transformer-layer-execution-studies/29-offload-n-streams.md:380), [29:394](/home/cj/mlir-air/docs/plans/transformer-layer-execution-studies/29-offload-n-streams.md:394)). Correct wording: **each launch selects/configures a distinct device image**, not “executes one raw `load_pdi` instruction.”

- **[SOURCE]** The probe’s claim that “**only the launch count differs**” is false. Total weights, arithmetic and kernel source are constant, but changing `n_part` 8192→4096 changes each launch’s internal grid from 128 to 64 iterations because `launch_size = m/tile_m/herd_m` ([matvec.py:108](/home/cj/mlir-air/programming_examples/matrix_vector_multiplication/bf16/matvec.py:108)). It also moves the broadcast DMA repeat geometry from the 255 limit to approximately 127. Thus the 2.07 ms delta includes reconfiguration **plus altered DMA scheduling/repeat state**; BD-count equality is unestablished. The timing numbers themselves are reproduced ([job 445:2](/home/cj/mlir-air/agents/.state/devq/jobs/job-000445.log:2)), but attribution of all 109 µs to a boundary is **[UNVERIFIABLE]**.

- **[INFERENCE]** The failed 8192 arm makes the attribution weaker, not “unaffected.” The corruption is nondeterministic and concentrated in partition 0 from row 64 onward ([job 446:2](/home/cj/mlir-air/agents/.state/devq/jobs/job-000446.log:2)). The observed pattern better supports stale repeat/BO/configuration state than a simple “last tile” defect, although the repeat-count edge remains plausible.

- **[SOURCE]** Production’s top-5 gate can miss such errors: only token-set failure is gating; full-logit diagnosis is informational ([report.py:54](/home/cj/mlir-air/programming_examples/llms/verify/report.py:54), [comparators.py:184](/home/cj/mlir-air/programming_examples/llms/verify/comparators.py:184)). Production also never invokes the LM-head ELF back-to-back, masking the observed state transition.

- **[INFERENCE]** Settling experiment: keep **19×8192, identical bytes and launch count**, compare `tile_m=8,m_input=4` (repeat 255) against `m_input=8` (roughly 127), plus 8128/8064 near-limit sweeps and a single-partition back-to-back test. Use sentinel/random weights, repeated and freshly loaded runs, and full-output error by partition. Separately measure 38 correct 4096 segments as either 38 devices or 19 devices each performing two segments; that holds descriptor/repeat geometry constant while changing only configuration count.

## 2. Counts, bytes and the “32 GB/s ceiling”

- **[SOURCE]** Launch counts are correct: Qwen 8 ([rms builder:664](/home/cj/mlir-air/programming_examples/llms/shared/builders/rms_qkv_qknorm_rope_multi.py:664)), 3 ([O/FFN builder:94](/home/cj/mlir-air/programming_examples/llms/shared/builders/o_gemv_ffn_multi.py:94)), and 19 ([LM builder:47](/home/cj/mlir-air/programming_examples/llms/shared/builders/lm_head_gemv_multi.py:47)); int4 is 6 ([int4 RMS:361](/home/cj/mlir-air/programming_examples/llms/llama32_1b_int4/multi_launch_builder/rms_qkv_int4_rope_multi.py:361)), 3 ([int4 O/FFN:4](/home/cj/mlir-air/programming_examples/llms/llama32_1b_int4/multi_launch_builder/o_gemv_ffn_int4_multi.py:4)), and 8 ([int4 driver:76](/home/cj/mlir-air/programming_examples/llms/llama32_1b_int4/llama32_1b_int4_inference.py:76)).

- **[SOURCE]** Weight arithmetic checks: Qwen QKV = 8.389 MB/layer, O+FFN = 23.069 MB/layer; Llama QKV = 12.583 MB, O+FFN = 109.052 MB, from the recorded dimensions ([Qwen config:33](/home/cj/mlir-air/programming_examples/llms/qwen3_0_6b/qwen3_0_6b_weights.py:33), [Llama config:39](/home/cj/mlir-air/programming_examples/llms/llama32_1b/llama32_1b_weights.py:39)). But the LM-head tables mix logical and physical bytes: Qwen is 311.165 MB logical versus **318.767 MB padded**; Llama 525.337 versus **536.871 MB padded**. Effective hardware rates should use the latter.

- **[INFERENCE]** “32 GB/s” is a useful achieved reference for large eight-column GEMVs, not a demonstrated ceiling. The only independently measured resource rate is 5.336 GB/s for one shim port ([analytical_cost.py:46](/home/cj/mlir-air/programming_examples/transformer_layer/study/analytical_cost.py:46)); multiplying by eight gives 42.7 GB/s before contention. Small/fused and especially int4 GEMVs cannot simply inherit 32 GB/s.

## 3. Translation-table audit

- **O1 — [INFERENCE]:** QKV concatenation is feasible. Prologue/epilogue fusion requires a **new core kernel**, not mechanical builder stitching; stitching only extracts and concatenates launch bodies ([stitching.py:383](/home/cj/mlir-air/programming_examples/llms/shared/infra/stitching.py:383)). An existing fused RMS/SwiGLU kernel proves feasibility, not generality ([matvec_swiglu_rms.py:4](/home/cj/mlir-air/programming_examples/decode_ffn_swiglu/matvec_swiglu_rms.py:4)). O1(c)’s “outer loop in the runtime sequence” is wrong for repeat-count ELF launches: the pass deliberately inserts a reload/reset at launch end ([AIRRtToNpuPass.cpp:1037](/home/cj/mlir-air/mlir/lib/Conversion/AIRRtToNpuPass.cpp:1037)). It needs a single-device BD-chain or reset-free design.

- **O2 — [SOURCE]:** “57→29 submissions” should be **30**: RMS0; 27 `(O_L,RMS_L+1)` pairs; O27; LM-head. Final RMS remains a CPU barrier before LM-head ([inference.py:431](/home/cj/mlir-air/programming_examples/llms/qwen3_0_6b/qwen3_0_6b_inference.py:431)), so O6 alone cannot reach one submission.

- **O3 — [UNVERIFIABLE]:** The current rates are accurate, but “both shim channels” and “2–4 rows per column” are new mappings, not existing LM-head geometry. The −15 ms overlaps O1 unless applied only to the post-O1 measured residual.

- **O4 — [SOURCE]:** Implementable and arithmetic is sound: symmetric packed int4 exists ([matvec_int4_packed.py:17](/home/cj/mlir-air/programming_examples/matrix_vector_multiplication/int4_awq/matvec_int4_packed.py:17)). Accuracy remains the gate.

- **O5 — [INFERENCE]:** Existing Llama int4 QKV lacks Qwen’s Q/K norm, so this is not simple reuse. Assuming the measured 4–14 GB/s int4 path suddenly reaches 32 GB/s is unsupported.

- **O6 — [SOURCE]:** Today’s conversion/loop diagnosis is exact ([decode.py:302](/home/cj/mlir-air/programming_examples/llms/qwen3_0_6b/qwen3_0_6b_decode.py:302)). But “keeps attention at ~3 ms/token to 2048” requires reading about 470 MB/token at **157 GB/s**, before attention math. Reject. Runtime `pos` also requires a kernel/interface change ([attn_decode_npu2.py:470](/home/cj/mlir-air/programming_examples/attention_decode/attn_decode_npu2.py:470)).

- **O7/O8 — [SOURCE]:** Both “today” columns are accurate ([decode.py:347](/home/cj/mlir-air/programming_examples/llms/qwen3_0_6b/qwen3_0_6b_decode.py:347), [inference.py:645](/home/cj/mlir-air/programming_examples/llms/qwen3_0_6b/qwen3_0_6b_inference.py:645)). O7 is implementable but unmeasured; O8’s 0.1 s is a scaling target, not established latency.

- **O9 — [SOURCE]:** Causal masking occurs after dense K matmul, so skipping is real ([attn_npu2.py:734](/home/cj/mlir-air/programming_examples/flash_attention/kernel_fusion_based/attn_npu2.py:734)). **[UNVERIFIABLE]** The bf16-pair DMA trick has no repository demonstration. The source establishes only that sub-32-bit innermost stride must be one ([transpose_bf16.py:9](/home/cj/mlir-air/programming_examples/data_transfer_transpose/dma_bf16/transpose_bf16.py:9)); pairing also requires legal reinterpretation, alignment, layouts, and on-device BO handoff.

- **O10/O11/O12 — [SOURCE]:** Accurate and implementable, but plan hashing is correctness, pmode reporting is measurement hygiene, and trace is diagnostics—not decode-speed budget.

## 4. Defensible budget

**[INFERENCE]** Do not add O1 and O3 independently. Define their combined saving as `71.6 ms − measured_post_fusion_time`, bounded by the optimistic 27.6 ms weight-stream floor plus vector/dequant work. A conditional planning band is:

- baseline: 89 ms;
- O1+O3 together: −25…−35 ms only after the corrected isolation;
- O4: −6…−7 ms after accuracy;
- O2+O7: 0…−6 ms until measured.

That yields **41–58 ms/token conditionally**. O5 and O6 receive zero budget credit today. The proposed **18–22 ms is a stretch hypothesis**, not an arithmetic budget.

## 5. Missing and cuts

**[SOURCE]** Missing Hexagon mechanisms are the capacity/live-set fusion planner and one-time shared activation preparation/reuse ([55:160](/home/cj/mlir-air/docs/plans/transformer-layer-execution-studies/55-hexagon-llama-cpp-lessons-for-xdna2.md:160), [55:179](/home/cj/mlir-air/docs/plans/transformer-layer-execution-studies/55-hexagon-llama-cpp-lessons-for-xdna2.md:179)), plus explicit host-fallback/split accounting ([55:323](/home/cj/mlir-air/docs/plans/transformer-layer-execution-studies/55-hexagon-llama-cpp-lessons-for-xdna2.md:323)). Cut O6(i)’s 3 ms promise, O9(ii)’s asserted legality, and O1(c) until a reset-safe design exists; move O10–O12 outside the performance budget.

