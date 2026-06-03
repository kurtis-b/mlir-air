# Gemma3 Model Loop Implementation

This document is the implementation roadmap for turning the current
Gemma3-style MLIR-AIR/NPU2 kernels into an iterative model loop. It is scoped to
source-level MLIR-AIR examples in this repository, not to a binary
FastFlowLM reproduction or a validated end-to-end Gemma3 runtime.

Primary sources:

- Paper: ["Mapping Gemma3 onto an Edge Dataflow Architecture",
  arXiv:2602.06063v2](https://arxiv.org/abs/2602.06063), last revised
  2026-02-24.
- Current source-level implementation:
  [`programming_examples/gemma3_herd_sweep`](.).
- Historical first-pass sketches:
  [`programming_examples/gemma3_dataflow_kernels`](../gemma3_dataflow_kernels).
- Model-inference architecture reference:
  [`programming_examples/llama32_1b`](../llama32_1b).

## Status And Non-Claims

The current directory contains kernel-level mappings for the paper's key data
movement and compute patterns:

- Q4NX int4 block dequantization.
- BF16 tiled matrix multiplication.
- FusedDQP fused dequantization/projection for decode-style MVM.
- FlowQKV chunked prefill attention.
- FlowKV decode attention as the `Q_CHUNK=1` attention specialization.

These kernels are readable, sweepable AIR mappings. They are not yet a model
runtime. Do not report this directory as:

- an end-to-end Gemma3 deployment,
- a FastFlowLM reproduction,
- a performance reproduction of the arXiv paper,
- a complete text-plus-vision VLM implementation,
- or a validated model-accuracy result.

The immediate goal is a host-driven loop that invokes validated kernels in the
order a Gemma3 text model needs, records the shape and buffer contracts, and
then replaces host fallbacks with NPU kernels only after isolated compile and
hardware evidence is clean.

## Implemented Synthetic Loop Status

The first implementable text-only loop is now implemented as source-level
Gemma3 examples. It remains a synthetic correctness and control-plane artifact,
not a real Gemma3 deployment or hardware performance claim.

Implemented files:

- `gemma3_config.py`: text config, layer metadata, public output-mode defaults,
  and explicit nonlinear fallback steps.
- `gemma3_weights.py`: deterministic synthetic Q4NX/BF16-compatible weights.
- `gemma3_reference.py`: CPU references for projections, attention, nonlinear
  operations, residuals, logits, embeddings, and KV-cache updates.
- `gemma3_runtime.py`: manifest/cache preparation with compile-only and
  run-only validation.
- `gemma3_prefill.py`: synthetic text prefill orchestration with per-stage
  checksums, cache metadata, and local/global KV sweep reporting.
- `gemma3_decode.py`: one-token decode orchestration with FlowKV-compatible
  `Q_CHUNK=1` metadata and cache growth checks.
- `gemma3_nonlinears.py`: nonlinear reuse registry and CPU contract checks.
- `../gemma3_dataflow_kernels/geglu.py`: Gemma-specific GeGLU standalone AIR
  kernel candidate with reference and compile-only lit coverage.
- `gemma3_model_loop.py`: unified synthetic session for multiple prompt chunks
  and decode tokens, compact logs, optional stage logs, and failure context.
- `gemma3_scaling.py`: public-mode scaling policy checks for `2x4`, `4x4`, and
  `8x4`, including classified unsupported-mode diagnostics.
- `gemma3_vision.py`: disabled text-only contract plus a synthetic non-causal
  vision prefill contract that produces visual context tokens without claiming
  NPU validation.
- `gemma3_inference.py`: Llama32-style entrypoint with compile-only, run-only,
  verify, profile, layer-count, prompt-chunk, decode-token, local-window, and
  stage-log controls.
- `gemma3_real_execution.py`: CPU/HF real-artifact smoke path for the 1B,
  4B text, and 4B synthetic-image model paths, proving local weights,
  tokenizer, and processor can execute without AIR imports while making no
  paper timing claim.
- `gemma3_npu_preflight.py`: real-shape NPU preflight planner that derives
  projection padding, Q4NX block counts, attention metadata, and the remaining
  NPU execution blocker from local artifacts.

Focused lit coverage:

- `run_model_loop_reference.lit`
- `run_model_loop_runtime.lit`
- `run_model_loop_prefill_decode.lit`
- `run_model_loop_nonlinears.lit`
- `run_model_loop_session.lit`
- `run_model_loop_scaling.lit`
- `run_model_loop_vision.lit`
- `run_model_loop_real_execution.lit`
- `run_model_loop_npu_preflight.lit`
- `../gemma3_dataflow_kernels/run_geglu_compile_only.lit`

Current phase status:

| Phase | Status | Evidence |
| --- | --- | --- |
| Phase 0 | Complete | README link, support matrix, known unsupported-mode classes, and this status section are present. |
| Phase 1 | Complete | Config, weights, and CPU references are implemented without AIR imports. |
| Phase 2 | Complete | Manifest preparation supports compile-only/run-only and refuses missing or mismatched artifacts. |
| Phase 3 | Complete | Two-chunk synthetic prefill records Q4NX/BF16/FlowQKV stages and distinct local/global KV sweeps. |
| Phase 4 | Complete | Repeated one-token decode grows cache lengths and records FlowKV-compatible stage metadata. |
| Phase 5 | Complete for standalone promotion readiness | RMSNorm/QK-Norm reuse, RoPE half-split source reuse, GeGLU kernel candidate, and fallback policy are documented and tested. |
| Phase 6 | Complete | The synthetic transformer layer skeleton includes attention, output projection, residual, MLP, and deterministic stage checksums. |
| Phase 7 | Complete | Multi-layer, multi-chunk, multi-token text loop is implemented in `Gemma3SyntheticSession`. |
| Phase 8 | Policy complete, no performance claims | Scaling manifests and unsupported-mode diagnostics are checked; timing and hardware numbers are intentionally absent. |
| Phase 9 | Disabled contract complete | Vision can be disabled entirely and text-only checksums/cache lengths are unchanged. |

Latest focused verification commands used for this status:

```bash
source /home/cj/iron/ironenv/bin/activate && sandbox/bin/lit -v --filter=model_loop build-xrt/programming_examples
source /home/cj/iron/ironenv/bin/activate && sandbox/bin/lit -v --filter=geglu_compile_only build-xrt/programming_examples
```

No hardware run, NPU power-mode change, clean rebuild, or `/home/cj/mlir-aie`
edit is part of this synthetic model-loop completion.

## Paper Reproduction Target

The next roadmap goal is to turn the current synthetic, source-level Gemma3
model-loop scaffold into an iterative reproduction of the paper's implementation
and results. Until every item in this section has local evidence, do not claim
that this repository matches the paper.

### Source-of-truth policy

Track both paper result sources because they currently disagree:

- arXiv PDF/HTML v2 result tables and body text are the primary numeric target:
  <https://arxiv.org/pdf/2602.06063> and
  <https://arxiv.org/html/2602.06063v2>.
- arXiv abstract and public FastFlowLM pages remain secondary headline sources:
  <https://arxiv.org/abs/2602.06063> and
  <https://fastflowlm.com/docs/benchmarks/gemma3_results/>.
- If PDF/table values and abstract/site values conflict, record both values in
  the result ledger, compare against the PDF/table value first, and add a note
  explaining the mismatch.
- Treat a local result as `PAPER_MATCH` only when it is within 20% of the paper
  value for the same model, sequence length, backend, metric, and power mode.
- Treat a local result outside 20% as `EXPLAINED_DEVIATION` only when the root
  cause is documented with a concrete artifact: different hardware, runtime,
  power mode, unsupported route, compile/runtime failure, fallback path, or
  measurement-method mismatch.

Similarity formulas:

```text
latency_delta_pct = abs(local_seconds - paper_seconds) / paper_seconds * 100
throughput_delta_pct = abs(local_tps - paper_tps) / paper_tps * 100
speedup_delta_pct = abs(local_speedup - paper_speedup) / paper_speedup * 100
power_delta_pct = abs(local_watts - paper_watts) / paper_watts * 100
```

Only compare timings after correctness passes. Do not use lit wall time, compile
time, manifest generation time, or synthetic checksum time as model latency.

### Paper headline conflict ledger

| Source | NPU vs iGPU prefill | NPU vs iGPU decode | NPU vs CPU prefill | NPU vs CPU decode | Power efficiency |
| --- | ---: | ---: | ---: | ---: | --- |
| PDF/HTML v2 body | up to 7.5x | up to 5.9x | up to 23.7x | up to 2.7x | up to 96.7x vs iGPU, 157.7x vs CPU |
| arXiv abstract / secondary pages | up to 5.2x | up to 4.8x | up to 33.5x | up to 2.2x | up to 67.2x vs iGPU, 222.9x vs CPU |

Acceptance for the repo should target the PDF/HTML v2 tables. The abstract/site
numbers are still tracked so that any future paper-version or FastFlowLM-version
reconciliation is explicit.

### Paper result targets to reproduce

Prefill TTFT targets are seconds. Decode targets are tokens per second.

| Metric | Model | Backend | 1k | 2k | 4k | 8k | 16k | 32k | 64k | 128k |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Prefill TTFT | 1B | NPU | 0.95 | 1.47 | 2.46 | 4.42 | 8.45 | 17.19 | n/a | n/a |
| Prefill TTFT | 1B | iGPU | 0.51 | 1.14 | 2.61 | 6.64 | 21.3 | 95.1 | n/a | n/a |
| Prefill TTFT | 1B | CPU | 4.06 | 8.06 | 17.2 | 36.2 | 75.5 | 165 | n/a | n/a |
| Prefill TTFT | 4B | NPU | 1.81 | 2.81 | 4.79 | 8.37 | 16.17 | 33.5 | n/a | n/a |
| Prefill TTFT | 4B | iGPU | 2.05 | 4.26 | 9.54 | 23.8 | 71.5 | 265 | n/a | n/a |
| Prefill TTFT | 4B | CPU | 20.3 | 42.4 | 85.4 | 176 | 766 | 832 | n/a | n/a |
| Decode TPS | 1B | NPU | 41.1 | 40.5 | 39.5 | 37.3 | 33.6 | 27.9 | OOC | OOC |
| Decode TPS | 1B | iGPU | 38.0 | 53.9 | 42.3 | 33.5 | 25.0 | 13.6 | n/a | n/a |
| Decode TPS | 1B | CPU | 41.9 | 40.8 | 41.7 | 40.2 | 38.1 | 33.8 | n/a | n/a |
| Decode TPS | 4B | NPU | 18.2 | 18.0 | 17.8 | 17.3 | 16.3 | 14.8 | 13.2 | 11.2 |
| Decode TPS | 4B | iGPU | 18.6 | 17.4 | 15.3 | 12.6 | 9.2 | 5.9 | 3.3 | 1.9 |
| Decode TPS | 4B | CPU | 14.6 | 13.5 | 13.9 | 13.0 | 11.4 | 10.8 | 7.5 | 4.1 |

Vision target from the paper body:

| Metric | Model | Backend | Value |
| --- | --- | --- | ---: |
| Vision TTFT | 4B vision tower | NPU | 2.6 sec |
| Vision TTFT | 4B vision tower | iGPU | 7.45 sec |
| Vision TTFT | 4B vision tower | CPU | 38.55 sec |
| Vision speedup | 4B vision tower | NPU vs iGPU | 2.9x |
| Vision speedup | 4B vision tower | NPU vs CPU | 14.8x |

Average power targets from the paper's cross-hardware power table:

| Category | Model | CPU W | GPU W | NPU W | Total W |
| --- | --- | ---: | ---: | ---: | ---: |
| NPU decoding | 1B | 2.8 | 0 | 1.8 | 4.6 |
| NPU decoding | 4B | 2.9 | 0 | 1.6 | 4.5 |
| NPU prefill | 1B | 0.0 | 3.1 | 1.2 | 4.3 |
| NPU prefill | 4B | 0.0 | 3.4 | 1.1 | 4.5 |
| iGPU decoding | 1B | 31 | 22 | 0 | 53 |
| iGPU decoding | 4B | 31 | 23 | 0 | 54 |
| iGPU prefill | 1B | 33 | 24 | 0 | 57 |
| iGPU prefill | 4B | 33 | 25 | 0 | 58 |
| CPU decoding | 1B | 29 | 0 | 0 | 29 |
| CPU decoding | 4B | 26 | 0 | 0 | 26 |
| CPU prefill | 1B | 24 | 0 | 0 | 24 |
| CPU prefill | 4B | 30 | 0 | 0 | 30 |

Power-efficiency targets from the paper body:

| Mode | Model | NPU vs iGPU TPS/W | NPU vs CPU TPS/W |
| --- | --- | --- | --- |
| Prefill | 1B | 6.9x-69.7x | 22.9x-50.7x |
| Prefill | 4B | 13.9x-96.7x | 70.6x-157.7x |
| Decode | 1B | 12.4x-24.2x | 6.2x-5.2x as written in PDF/HTML |
| Decode | 4B | 11.9x-71.1x | 7.2x-15.8x |

The `6.2x-5.2x` CPU decode range is recorded exactly as the paper text presents
it. Future implementation should verify whether this is a descending range, a
typographical error, or a derived value from the figure.

### Current gap to paper parity

| Area | Current state | Paper-parity requirement |
| --- | --- | --- |
| Model weights | Synthetic Q4NX/BF16-compatible weights only | Real Gemma3 1B and 4B weights, quantized/packed in the same Q4NX contract used by the NPU kernels |
| Tokenizer and prompts | Synthetic token IDs | Real tokenizer, deterministic prompts, and sequence lengths 1k-32k for 1B and 1k-128k decode for 4B |
| Text runtime | Host-driven synthetic loop with CPU references and manifests | End-to-end NPU execution for all validated model substeps, with host fallbacks removed or measured separately |
| Vision runtime | Disabled contract plus synthetic CPU non-causal vision prefill and visual context token handoff contract | 4B vision prefill path with non-causal attention validated on NPU and timed against paper |
| Nonlinear operations | CPU fallbacks plus standalone GeGLU candidate; host fallback microbenchmarks are recorded in result JSON | RMSNorm, QK-Norm, RoPE, residual add, activation, logits, and sampling either validated on NPU or explicitly accounted for in timing |
| Baselines | No comparable CPU/iGPU model path | CPU and iGPU runs for the exact same model variant, prompt lengths, output count, tokenizer, and measurement window |
| Timing | Synthetic profile/event logs only | TTFT and decode TPS with warmup, timed iterations, and compile/setup excluded |
| Power | No local power logs | CPU/GPU/NPU/total watt readings aligned with benchmark windows and TPS/W calculations |
| Accuracy | Synthetic checksums only | Model-output agreement against CPU reference and paper-compatible prompts, plus tolerance policy for quantized outputs |
| Result comparison | No result JSON ledger | Machine-readable paper/local comparison with percent deltas and pass/deviation labels |

## Iterative Paper-Match Implementation Loop

Each iteration should do exactly one paper-parity increment:

1. Identify the next missing result cell or capability.
2. Add the smallest implementation needed for that cell.
3. Run compile-only checks before hardware.
4. Run correctness before timing.
5. Run timing before power/TPS-W.
6. Save the command, environment, result JSON, and logs.
7. Update this document's evidence ledger.
8. Commit that iteration before moving to the next cell.

Do not broaden public support matrices, sweep modes, or paper claims until the
same mode passes compile, correctness, hardware, and paper-comparison checks.

### Phase A: paper metric extraction and ledger generation

Goal: create an explicit, versioned local representation of every paper result
cell before implementing more runtime code.

Implementation requirements:

- Add a paper-result data file under the Gemma example tree, preferably JSON or
  CSV, containing every TTFT, decode TPS, vision TTFT, power, speedup, and TPS/W
  target listed above.
- Include source fields for `pdf_v2`, `html_v2`, `abstract`, and `fastflowlm`
  where values differ.
- Add a parser/checker that validates the result file has all required model,
  backend, metric, and sequence-length cells.
- Add a comparison utility that reads local benchmark JSON and emits
  `PAPER_MATCH`, `EXPLAINED_DEVIATION`, `LOCAL_FAIL`, or `MISSING_LOCAL_RESULT`.

Acceptance:

- The result-target file is complete for all tables above.
- The checker reports the abstract/PDF headline conflict without failing.
- A synthetic local result fixture can be compared and classified within or
  outside the 20% threshold.

Implemented evidence:

- `paper_targets.json` records 141 paper target cells and 6 headline conflicts.
- `gemma3_paper_compare.py --validate` validates target completeness.
- `gemma3_paper_compare.py --self-test` exercises `PAPER_MATCH`,
  `EXPLAINED_DEVIATION`, and out-of-capacity comparisons.
- `run_model_loop_paper_compare.lit` covers validation and fixture comparison.

### Phase B: hardware and environment parity capture

Goal: make every local result reproducible and comparable.

Implementation requirements:

- Capture branch, commit, dirty worktree, build directory, MLIR-AIR install,
  MLIR-AIE install, Peano install, LLVM install, XRT version, `xrt-smi examine`,
  NPU power mode, CPU model, GPU/iGPU model, memory size, kernel artifact type,
  and Python environment.
- Record whether the run used `xclbin` or `elf`, whether trace was enabled, and
  whether compile/setup time was excluded.
- Refuse paper-comparison output when required environment fields are missing.

Acceptance:

- A no-hardware dry run can produce a complete environment JSON with missing
  hardware fields marked explicitly.
- A hardware run records XRT and NPU state before and after execution.
- Timeout or packet-route diagnostic runs require a following
  `xrt-smi examine -r all` record.

Implemented evidence:

- `gemma3_environment.py` captures git, Python, install paths, tool versions,
  CPU/memory, PCI hints, XRT availability, NPU power-mode hints, and runtime
  timing metadata.
- `gemma3_paper_compare.py --environment` rejects paper comparison when
  required environment fields are missing unless explicitly overridden.
- `run_model_loop_environment.lit` covers self-test and summary capture.
- Current no-hardware capture recovers repo-local tool paths and reports a
  paper-comparable software environment; hardware fields remain explicit rather
  than silently accepted when hardware is not required.

### Phase C: real Gemma3 model artifacts

Goal: replace synthetic token IDs and weights with reproducible Gemma3 1B and
4B artifacts.

Implementation requirements:

- Add model metadata for Gemma3 1B text, Gemma3 4B text, and Gemma3 4B vision.
- Add tokenizer loading and deterministic prompt generation for 1k, 2k, 4k,
  8k, 16k, 32k, 64k, and 128k contexts where supported.
- Add real safetensor loading or an explicit import path for already-converted
  weights.
- Define the exact Q4NX packing contract for every projected matrix:
  row/column order, group size, scale/min storage, low-nibble order, transpose
  policy, and BF16 conversion points.
- Preserve a CPU reference path for every model variant before NPU promotion.

Acceptance:

- CPU-only real-model load succeeds for 1B and 4B without importing AIR modules.
- Tokenizer-backed prompt generation and round-trip checks pass for the local
  real tokenizer at paper sequence lengths.
- Q4NX pack/dequant round-trip error is recorded per projection family.

Implemented evidence and blocker:

- `gemma3_artifacts.py` defines paper-model metadata for `gemma3-1b`,
  `gemma3-4b`, and `gemma3-4b-vision`, including paper prompt/decode lengths.
- Deterministic prompt-ID fixtures validate 1k-128k sequence-length plumbing
  without requiring AIR imports.
- `Q4NXPackingContract` records the block size, low-nibble order, BF16
  scale/min metadata, matrix order, and dequant formula.
- `run_model_loop_artifacts.lit` covers metadata, prompt-length generation,
  discovery, strict-load blocking diagnostics, and the additional processor-file
  requirement for vision artifacts.
- `requirements.txt` records the approved Python packages for artifact loading,
  tokenizer/processor setup, Hugging Face snapshot access, and safetensor shape
  inspection. These dependencies were installed in the active `ironenv` during
  blocker-fix work.
- `gemma3_artifacts.py` records official source repositories, manifest
  validation, optional snapshot download support, safetensor shape inspection,
  and Q4NX quantize/dequantize round-trip error for projection-family samples
  when real weights are available.
- Real Gemma3 artifact loading has been validated against the local default
  model root `/home/cj/models` for `gemma3-1b`, `gemma3-4b`, and
  `gemma3-4b-vision`; the remaining blocker is NPU model execution, not
  artifact availability. The artifact checker also supports `GEMMA3_MODEL_ROOT`
  and default per-variant directories for reproducible local discovery.

### Phase D: standalone kernel parity

Goal: prove each paper kernel role independently before using it in end-to-end
model timing.

Implementation requirements:

- Keep Q4NX, BF16 MM, FusedDQP, FlowQKV, and FlowKV as separately testable
  artifacts.
- Scale from compact smoke shapes to paper shapes, then to full paper sequence
  lengths.
- For each kernel, record correctness versus CPU reference, compile time,
  runtime latency, output mode, herd shape, schedule mode, KV staging, and
  unsupported-mode classification.
- Keep 8x4 direct S2MM over-allocation unsupported and use `l2-gather` for
  full-physical public model paths.
- Keep `packet-direct` diagnostic-only until packet S2MM receive behavior has
  three consecutive correct hardware runs per affected kernel.

Acceptance:

- Every kernel used by model timing has passing compile and hardware validation
  for its exact paper-mode shape.
- FlowKV small-shape `l2-gather` and FusedDQP `pipeline` remain excluded from
  production model routes until their timeout/channel issues are fixed.
- Kernel-level performance is logged but not used as end-to-end paper parity.

Implemented evidence and blocker:

- `gemma3_kernel_parity.py` defines the standalone production-candidate matrix
  for Q4NX, BF16 MM, FusedDQP, FlowQKV, and FlowKV, including paper-layout
  targets where the current sweep already exposes them.
- The matrix records the compile-only and hardware command contract for each
  role, herd shape, output mode, schedule mode, production/diagnostic class,
  and current snapshot status.
- `run_model_loop_kernel_parity.lit` validates that 8x4 public routes use
  `l2-gather`, that paper-layout FlowQKV/FusedDQP/FlowKV targets remain visible,
  and that diagnostic-only routes keep their explicit failure classes.
- Diagnostic exclusions are classified as hardware resource limit
  (`8x4 direct`), packet S2MM backend limitation (`packet-direct`), or
  channel/runtime scheduling bug (FlowKV small `l2-gather`).
- Fresh paper-shape hardware validation and kernel latency capture are still
  blocked by the same real-artifact and environment-comparability gaps recorded
  in Phases B and C; snapshot status must not be used as new paper evidence.

### Phase E: nonlinear and vector-kernel promotion

Goal: remove or account for host fallbacks so local timing can be compared to
paper end-to-end timing.

Implementation requirements:

- Reuse `weighted_rms_norm` for RMSNorm and QK-Norm when layout matches.
- Reuse or wrap Llama32 half-split RoPE for Gemma head dimensions.
- Use the Gemma GeGLU kernel candidate for activation only after standalone
  compile and hardware validation.
- Add residual add, elementwise multiply, logits/LM head, and sampling kernels
  only when they reduce measured model-loop overhead and preserve tensor
  contracts.
- Record every remaining host fallback in result JSON, including whether it is
  included in timing.

Acceptance:

- No paper-match timing can be labeled `PAPER_MATCH` while an unmeasured host
  fallback contributes to the timed window.
- Each promoted nonlinear has CPU reference, compile-only lit, hardware
  validation, and tolerance data.

Implemented evidence and blocker:

- `gemma3_nonlinears.py` now records CPU reference, tensor contract,
  compile-lit availability, hardware-validation status, tolerance policy, and
  timed-window status for RMSNorm, QK-Norm, RoPE, GeGLU, and residual add.
- The registry keeps nonlinear/vector stages as host fallbacks until
  Gemma-specific model wiring uses validated kernels. GeGLU now has standalone
  ELF hardware-smoke validation, but it is not promoted into model timing yet.
- `gemma3_results.py` now records fallback entries with backend, elapsed-ms,
  timed-iteration count, measurement source, tensor contract, hardware status,
  and `npu_promoted=false` for CPU-reference fallbacks.
- `gemma3_paper_compare.py` rejects `PAPER_MATCH` for timed paper metrics when
  local result JSON declares an unmeasured host fallback that contributes to the
  timed window; measured host fallback records are accepted as accounted timing
  but do not claim NPU promotion.
- `run_model_loop_nonlinears.lit`, `run_model_loop_results.lit`, and
  `run_model_loop_paper_compare.lit` cover the nonlinear metadata, measured
  result records, and paper-match fallback gate.
- Remaining work for full Phase E completion is Gemma model wiring for the
  validated GeGLU path plus RMSNorm, QK-Norm, RoPE, residual, logits, and
  sampling promotion or measured timing treatment in end-to-end execution.

### Phase F: end-to-end 1B text reproduction

Goal: reproduce Gemma3 1B prefill and decode tables.

Implementation requirements:

- Add a real 1B text session with tokenizer, real weights, real KV cache,
  static weight BO preloading, and per-layer artifact reuse.
- Implement prefill TTFT for 1k, 2k, 4k, 8k, 16k, and 32k prompts.
- Implement decode TPS at 1k, 2k, 4k, 8k, 16k, and 32k context lengths.
- Run CPU, iGPU, and NPU backends with the same prompt/token settings.
- Compare NPU, iGPU, CPU, and derived speedups against the paper table.

Acceptance:

- Correctness passes for all 1B paper sequence lengths.
- Every NPU TTFT/TPS cell is within 20% of the PDF/HTML v2 target or has an
  explained deviation.
- CPU/iGPU baselines are present before speedup claims are emitted.

Blocked evidence:

- `gemma3_reproduction_blockers.py` reports Phase F as `BLOCKED` while real
  Gemma3 1B safetensors/tokenizer artifacts are missing. The prior
  unmeasured-nonlinear fallback blocker is retired by measured CPU-reference
  fallback records, but those records are not NPU promotion evidence.
- Dependency-light CPU/HF smoke paths now validate local 1B text, 4B text,
  and 4B synthetic-image weights/tokenizer/processor execution without AIR
  imports. `gemma3_npu_preflight.py` records
  real projection padding and Q4NX block counts needed for NPU wiring. No
  CPU/iGPU/NPU paper baseline or speedup claim is emitted until
  benchmark-length execution and NPU model execution are implemented.

### Phase G: end-to-end 4B text reproduction

Goal: reproduce Gemma3 4B text prefill and decode tables.

Implementation requirements:

- Add a real 4B text session with the 5-local/1-global layer pattern and the
  paper's full supported decode range.
- Implement 4B prefill TTFT for 1k-32k prompts.
- Implement 4B decode TPS for 1k, 2k, 4k, 8k, 16k, 32k, 64k, and 128k context
  lengths.
- Track out-of-capacity behavior separately from failures.
- Use `l2-gather` full-physical routes where direct shim output is illegal.

Acceptance:

- Correctness passes for all 4B text sequence lengths that fit local memory and
  host memory limits.
- NPU, CPU, and iGPU local results are compared against Tables 2 and 4.
- Any 64k/128k deviation includes explicit KV-cache, memory, or schedule data.

Blocked evidence:

- `gemma3_reproduction_blockers.py` reports Phase G as `BLOCKED` while real
  Gemma3 4B safetensors/tokenizer artifacts are missing. Measured host fallback
  records account for timing metadata but do not replace nonlinear NPU
  validation.
- 64k/128k decode cells remain target-ledger entries only; no local paper claim
  is emitted without real KV-cache, memory, and schedule evidence.

### Phase H: 4B vision path reproduction

Goal: replace the disabled vision contract with the paper's vision-tower
inference result.

Implementation requirements:

- Add vision input preprocessing and image-token contract.
- Implement non-causal full attention for the vision path.
- Produce visual context tokens that seed the text prefill path.
- Measure vision TTFT separately from text-only prefill.
- Run NPU, iGPU, and CPU vision baselines.

Acceptance:

- Vision can still be disabled with unchanged text-only results.
- 4B vision TTFT is compared against 2.6 sec NPU, 7.45 sec iGPU, and 38.55 sec
  CPU targets.
- NPU vision speedups are compared against 2.9x over iGPU and 14.8x over CPU.

Blocked evidence:

- `gemma3_reproduction_blockers.py` reports Phase H as `BLOCKED` while real
  Gemma3 4B vision artifacts and processor files are missing and the synthetic
  vision contract is not NPU-validated. Measured host fallback records account
  for text nonlinear timing metadata but do not validate vision hardware.
- Existing text-only synthetic and blocked-result tests keep vision optional;
  the enabled vision smoke is a CPU-reference contract for non-causal attention
  and visual context token shape only. No vision TTFT or speedup claim is
  emitted without real vision execution.

### Phase I: benchmark harness and result JSON

Goal: make all paper comparisons one-command reproducible.

Future CLI shape:

```bash
python3 gemma3_inference.py --paper-benchmark \
  --model-variant gemma3-1b \
  --backend npu \
  --weights-dir <path> \
  --tokenizer <path> \
  --prompt-len 32768 \
  --decode-tokens 128 \
  --result-json results/gemma3_1b_npu_32k.json
```

Expected CLI additions:

- `--model-variant {gemma3-1b,gemma3-4b,gemma3-4b-vision}`
- `--weights-dir`
- `--tokenizer`
- `--backend {cpu,igpu,npu}`
- `--prompt-len`
- `--decode-tokens`
- `--paper-benchmark`
- `--warmup-iters`
- `--timed-iters`
- `--result-json`
- `--power-sample`
- `--compare-paper`
- `--trace-size`
- `--debug-ir`

Result JSON schema:

```json
{
  "schema_version": 1,
  "paper_source": "arxiv_pdf_v2",
  "paper_table": "Table 4",
  "model_variant": "gemma3-4b",
  "backend": "npu",
  "metric": "decode_tps",
  "sequence_length": 131072,
  "decode_tokens": 128,
  "local_value": 11.2,
  "paper_value": 11.2,
  "unit": "tokens_per_second",
  "delta_pct": 0.0,
  "classification": "PAPER_MATCH",
  "correctness": "PASS",
  "host_fallbacks": [],
  "command": "...",
  "git_commit": "...",
  "dirty_worktree": false,
  "xrt_version": "...",
  "npu_power_mode": "...",
  "artifact_format": "elf",
  "warmup_iters": 3,
  "timed_iters": 10,
  "compile_time_included": false,
  "power_watts": {
    "cpu": 2.9,
    "gpu": 0.0,
    "npu": 1.6,
    "total": 4.5
  },
  "notes": []
}
```

Acceptance:

- A paper benchmark run writes one result JSON per result cell.
- A comparison command generates Markdown and CSV summaries.
- Missing or failed cells are visible and cannot be silently skipped.

Implemented evidence and blocker:

- `gemma3_inference.py --paper-benchmark` accepts the paper CLI shape for model
  variant, backend, weights, tokenizer, prompt length, decode tokens, warmup,
  timed iterations, result JSON, paper comparison, power sampling, trace size,
  and debug-IR intent.
- `gemma3_results.py` writes one result JSON cell for the requested paper target
  and records command, git/environment metadata, artifact inventory, nonlinear
  host fallbacks, null power fields, and explicit blocked classification.
- `gemma3_paper_compare.py --compare` accepts either a single result cell or a
  wrapper with `results`, and can emit Markdown and CSV summaries.
- `run_model_loop_results.lit` covers blocked real-artifact result generation,
  paper comparison, Markdown/CSV summary emission, and JSON schema essentials.
- The harness is implemented, but real `PAPER_MATCH` cells remain blocked until
  Phases F-H can run with real artifacts, validated kernels, and comparable
  hardware/power telemetry.

### Phase J: power and TPS/W reproduction

Goal: reproduce the paper's average power table and TPS/W improvement ranges.

Implementation requirements:

- Define a power-sampling backend before collecting numbers: XRT telemetry,
  platform telemetry, RAPL, or another reproducible source.
- Align power sampling windows with timed inference windows.
- Record CPU/GPU/NPU/total watts separately when telemetry supports it.
- Compute TPS/W using the same timed throughput used for the paper comparison.
- Never combine timing from one run with power from another run unless both run
  IDs are recorded and the method is documented.

Acceptance:

- Local average watts are compared against the power table.
- TPS/W improvement ranges are computed for every model/backend/mode with local
  CPU/iGPU/NPU values.
- Any unavailable rail or telemetry source is reported as `MISSING_POWER_FIELD`,
  not zero.

Implemented evidence and blocker:

- `gemma3_power.py` defines the power telemetry contract, including CPU/GPU/NPU
  and total rails, timed-window alignment metadata, run IDs, missing-field
  classification, and TPS/W helper calculations.
- `gemma3_results.py` includes power watts, per-rail power status,
  sampling-backend metadata, and timed-window alignment in every paper result
  JSON; unavailable rails are `null` with `MISSING_POWER_FIELD`, never zero.
- `run_model_loop_power.lit` covers the missing-telemetry contract and verifies
  that JSON output keeps watts null while statuses classify missing rails.
- Full power-table comparison and TPS/W reproduction remain blocked until a
  real timed inference run and an approved telemetry backend are available.

### Phase K: final paper-parity report

Goal: produce a report that can answer whether the implementation and results
match the paper.

Implementation requirements:

- Generate a Markdown report with one row per paper result cell.
- Include paper value, local value, delta, classification, command, and log path.
- Include a separate section for abstract/PDF/site conflicts.
- Include a separate section for unsupported modes and why they are not paper
  parity blockers or are still blockers.
- Include accuracy/correctness summary before performance summary.

Acceptance:

- The final report can state one of:
  - `MATCHES_PAPER`: all required cells are within 20% and correctness passes.
  - `MATCHES_WITH_EXPLAINED_DEVIATIONS`: all cells are present, but some exceed
    20% with accepted explanations.
  - `DOES_NOT_MATCH_PAPER`: one or more required cells are missing, incorrect,
    unsupported, or unexplained.
- The report is backed by committed source changes and saved result artifacts.

Implemented evidence and blocker:

- `gemma3_report.py` renders one Markdown row per paper target cell, including
  paper value, local value, delta, classification, correctness, command, log
  path, and note.
- The report puts correctness summary before performance summary, then lists all
  result cells, headline source conflicts, and unsupported/diagnostic mode
  classifications from the kernel parity matrix.
- The report status is `MATCHES_PAPER`, `MATCHES_WITH_EXPLAINED_DEVIATIONS`, or
  `DOES_NOT_MATCH_PAPER`; with current blocked/missing real results it correctly
  reports `DOES_NOT_MATCH_PAPER`.
- `run_model_loop_report.lit` covers self-test, blocked-result input, Markdown
  output, required sections, and final status.
- The report generator is complete, but a positive paper-parity status remains
  blocked until all required paper cells have real local results.

### Artifact and result directory policy

Use source-controlled files for plans, target tables, scripts, and small
fixtures. Keep large or machine-specific artifacts out of source unless they are
explicitly reviewed.

Recommended paths:

```text
programming_examples/gemma3_herd_sweep/paper_targets.json
programming_examples/gemma3_herd_sweep/gemma3_paper_compare.py
programming_examples/gemma3_herd_sweep/results/README.md
programming_examples/gemma3_herd_sweep/results/*.json
programming_examples/gemma3_herd_sweep/results/*.md
programming_examples/gemma3_herd_sweep/results/*.csv
programming_examples/gemma3_herd_sweep/power_logs/*.json
programming_examples/gemma3_herd_sweep/kernel_cache/
```

Do not commit xclbins, ELFs, large safetensors, tokenizer caches, trace dumps,
or generated debug IR unless the file is intentionally added as a compact test
fixture.

### Paper-match guardrails

- Do not claim FastFlowLM parity until real weights, tokenizer, CPU/iGPU/NPU
  baselines, correctness, timing, and power are all present.
- Do not claim a paper speedup from NPU-only data; speedup requires local CPU or
  iGPU baseline data from the same benchmark harness.
- Do not claim TPS/W without local power logs.
- Do not hide host fallbacks in timed windows.
- Do not expose diagnostic modes in sweeps just to fill result cells.
- Do not edit `/home/cj/mlir-aie`, reboot, change NPU power mode, or clean
  rebuild without explicit approval.
- Do not treat compile-only success as runtime support.
- Do not treat synthetic checksums as model accuracy.
- Always separate compile failure, validation failure, runtime timeout,
  unsupported mode, and paper deviation.

## Paper Concepts To Preserve

The implementation should preserve the following architectural ideas from the
paper while adapting them to this repo's MLIR-AIR programming examples.

### Target device model

- Target NPU: AMD Ryzen AI NPU2 / XDNA2 / AIE2P.
- Physical array: 32 compute tiles as 8 columns by 4 rows.
- Memory hierarchy:
  - L3: host/main memory buffers visible to XRT.
  - L2 / MT: memory tiles used for staging, fanout, and gather.
  - L1 / CT local memory: per-compute-tile scratch and kernel inputs.
  - Shim tiles: main-memory DMA endpoints with limited MM2S/S2MM channels.
- Data movement should be explicit. Prefer L3-to-L2-to-L1 staging for production
  full-physical paths when direct shim DMA would over-allocate resources.

### Model execution phases

- Prefill consumes a prompt chunk, runs the model once over those tokens, and
  appends K/V results to the cache.
- Decode is autoregressive. It runs one token at a time and repeatedly attends
  over the growing K/V cache.
- Text Gemma3 uses local sliding-window attention layers and global full
  attention layers. The paper describes a 5-local-layer then 1-global-layer
  pattern for the 4B text stack, starting with a local layer.
- The vision tower is optional for this repo's first model-loop milestone.
  Treat it as a later prefill-like path using non-causal attention, not as a
  blocker for text loop integration.

### Operation classes

The paper groups Gemma3 work into three implementation classes:

- Projection/MM work:
  - Prefill projections are matrix multiplications.
  - Decode projections are matrix-vector multiplications.
  - Projection weights are stored compactly and dequantized before or during
    projection.
- Attention work:
  - FlowQKV handles prefill attention by chunking query and KV cache work,
    preserving online-softmax state across KV chunks.
  - FlowKV handles decode attention with `Q_CHUNK=1`, splitting score/softmax
    and value-apply work into a pipeline when the route is supported.
  - SWA variants limit the KV sweep to a window; full attention sweeps all
    available context; vision attention is full non-causal.
- Nonlinear/vector work:
  - RMSNorm, residual adds, RoPE, QK-Norm, GeLU, elementwise multiply, logits,
    and sampling are required for a full model.
  - This repo should keep these on host or in diagnostic kernels until isolated
    NPU implementations exist and are validated.

## Current Kernel Inventory

| Paper role | Local target | Kernel source | Loop phase | Current use |
| --- | --- | --- | --- | --- |
| Q4NX dequantization | `run-q4nx` | `q4nx.py`, `q4nx_opt.cc` | Prefill weight preparation | Validated standalone kernel. |
| BF16 projection/MM | `run-mm` | `bf16_tiled_mm.py`, `../matrix_multiplication/bf16/mm_aie2p.cc` | Prefill projections and dense fallback | Validated standalone GEMM wrapper. |
| FusedDQP | `run-fused-dqp`, `run-fused-dqp-paper` | `fused_dqp.py`, `fused_dqp_opt.cc` | Decode projections/MVM | Smoke and paper targets pass; pipeline target remains diagnostic-only. |
| FlowQKV | `run-flowqkv`, `run-flowqkv-paper` | `flowqkv.py`, `flow_attention_opt.cc` | Prefill attention | Smoke and paper targets pass with supported modes. |
| FlowKV | `run-flowkv`, `run-flowkv-paper` | `flowkv.py`, `flow_attention_opt.cc` | Decode attention | Direct small routes and 8x4 gather route pass; small gather remains diagnostic-only. |

Current public output modes are:

- `auto`
- `direct`
- `l2-gather`

`packet-direct` is diagnostic-only. It must not appear in public sweeps or model
loop defaults.

Supported public mode matrix:

| Kernel | 2x4 modes | 4x4 modes | 8x4 modes |
| --- | --- | --- | --- |
| Q4NX | direct, l2-gather | direct, l2-gather | l2-gather |
| FusedDQP | direct, l2-gather | direct, l2-gather | l2-gather |
| FlowQKV | direct, l2-gather | direct, l2-gather | l2-gather |
| FlowKV | direct | direct | l2-gather |

Unsupported modes must stay explicit:

- 8x4 `direct` output for Q4NX, FusedDQP, FlowQKV, and FlowKV exceeds physical
  shim S2MM resources. Use `l2-gather`.
- Q4NX/FusedDQP `packet-direct` has backend/runtime packet S2MM limitations on
  hardware even when lowering is compile-legal. Keep hidden.
- FlowKV 2x4/4x4 `l2-gather` is a channel/runtime scheduling issue, not a
  supported route.
- FusedDQP `pipeline` currently times out on hardware. Treat it as a
  channel/runtime scheduling reproducer, not a production route.

## Llama32 Inference Reference Pattern

Use `programming_examples/llama32_1b` as the model-inference architecture
reference for Gemma3. It is not a source for Gemma-specific model semantics; it
is the control-plane and runtime organization to copy once the Gemma kernels are
ready to be invoked as a model.

| Llama32 file or directory | Pattern Gemma should reuse |
| --- | --- |
| `llama32_1b_inference.py` | Unified session object, `prepare_runtime`, compile-only/run-only split, synthetic path, prompt/token boundary, prefill followed by decode. |
| `llama32_1b_prefill.py` | Per-layer prefill orchestration, cached kernel compilation, intermediate verification, and multi-launch fused kernel calls. |
| `llama32_1b_decode.py` | Per-token decode loop, KV-cache update, per-layer BO keys, static weight inputs, and decode attention fallback structure. |
| `llama32_1b_reference.py` | CPU F32 reference functions, per-stage intermediate names, tolerances, and shape validation before NPU promotion. |
| `llama32_1b_weights.py` | Config dataclass, per-layer weight container, HuggingFace/synthetic weight split, and host-side transpose/layout preparation. |
| `kernel_builder/` | `KernelCache`, backend presets, external kernel compilation, artifact manifests, and text-based MLIR stitching conventions. |
| `multi_launch_builder/` | Multi-launch ELF pattern for packing several layer substeps into one XRT call while keeping intermediate buffers explicit. |

Gemma implementation should first copy the Llama32 split into Gemma-specific
files, then substitute Gemma kernels and shapes. The minimum future file split
should be:

- `gemma3_config.py`: Gemma text/vision config, local/global attention pattern,
  Q4NX block metadata, and public output-mode defaults.
- `gemma3_weights.py`: synthetic weights first, then real Gemma safetensor
  loading and Q4NX packing/import compatibility.
- `gemma3_reference.py`: CPU references for projections, attention, nonlinear
  operations, residuals, logits, and KV-cache updates.
- `gemma3_prefill.py`: text prefill orchestration using Q4NX, BF16 MM, FlowQKV,
  and validated nonlinear kernels or host fallbacks.
- `gemma3_decode.py`: one-token decode using FusedDQP, FlowKV, KV-cache updates,
  and validated nonlinear kernels or host fallbacks.
- `gemma3_inference.py`: unified session, compile-only/run-only flags, runtime
  preparation, synthetic verification, optional real-tokenizer path, and profile
  reporting.

## Gemma3 Inference Architecture Modeled After Llama32

The first Gemma3 inference loop should preserve these Llama32 runtime patterns:

- Build or load kernel artifacts once through a cache, then run by artifact name.
- Pre-load static weights into BOs before timed inference.
- Use per-layer BO keys so weights and intermediates do not alias across layers.
- Mark static weight inputs and intermediate output buffers explicitly, following
  Llama32 `static_input_indices` and `intermediate_indices`.
- Keep synthetic weights and deterministic token IDs as the first correctness
  target. Real tokenizer and real weights are a later compatibility target.
- Keep compile-only, run-only, verify, and profile modes separate.
- Prefer multi-launch ELFs only after the individual Gemma substeps have passing
  CPU-reference and hardware evidence.

Gemma-specific differences must remain explicit:

- Projection weights are Q4NX/int4 where the Gemma paper kernels require them.
- Prefill projection uses Q4NX/BF16 MM paths; decode projection uses FusedDQP
  when supported.
- Attention uses FlowQKV for prefill and FlowKV for decode, with local SWA and
  global full-attention metadata per layer.
- QK-Norm is a Gemma-specific nonlinear requirement and should not be assumed to
  match Llama without a CPU-reference contract.
- Vision prefill is optional later scope and must not block the text-only loop.
- Public Gemma kernel modes remain constrained by the support matrix above.

## Nonlinear Operation Policy

Nonlinear operations needed by Gemma3 must follow this order:

1. Reuse an existing implementation under `programming_examples` when the math,
   layout, dtype, and shape constraints match.
2. Reuse Llama32 builder patterns when the fused-runtime structure is useful but
   the operation still needs a Gemma wrapper.
3. Implement a Gemma-specific kernel under `programming_examples/gemma3_dataflow_kernels`
   when no existing implementation is shape- or semantics-compatible.

| Operation | Preferred source | Gemma rule |
| --- | --- | --- |
| RMSNorm | `programming_examples/weighted_rms_norm`, plus Llama CPU reference | Reuse for attention/FFN/final norm if Gemma row layout matches; otherwise add a Gemma wrapper around the same math. |
| RoPE | `llama32_1b/kernel_builder/rope_halfsplit.cc`, `programming_examples/rope_sincos` | Use half-split LUT only if Gemma uses the same convention; otherwise add `gemma3_dataflow_kernels/gemma_rope.py/.cc`. |
| QK-Norm | weighted RMSNorm pattern | Implement as per-head Q/K normalization with CPU reference first; add `gemma3_dataflow_kernels/qk_norm.py/.cc` if existing RMSNorm cannot express the layout. |
| MLP activation | `programming_examples/gelu`, `programming_examples/ffn_swiglu`, Llama `silu_and_mul` | Confirm Gemma activation from config/paper evidence before choosing GELU/GeGLU/SwiGLU; do not copy Llama SwiGLU by assumption. |
| Elementwise multiply | `programming_examples/ffn_swiglu`, Llama `silu_and_mul` | Reuse if activation semantics match; otherwise add a standalone Gemma elementwise kernel. |
| Residual add | Llama multi-launch builder patterns | Keep host-side initially; promote through a small Gemma vector kernel or fused builder once shape checks are stable. |
| Softmax | FlowQKV/FlowKV attention kernels | Keep inside attention kernels unless a separate model-level softmax is required. |
| LM head/logits | Llama LM-head GEMV multi-launch pattern | Reuse only after Gemma vocab size, embedding dim, partitioning, and weight layout are documented. |
| Sampling | Host-side | Keep deterministic or host-side until logits correctness is validated. |

Every new nonlinear kernel under `gemma3_dataflow_kernels` needs:

- a CPU reference in `gemma3_reference.py`,
- a compact synthetic test vector,
- a compile-only lit test before model-loop dependency,
- hardware validation before being labeled supported,
- and a documented fallback path in the model loop.

## Target Model Loop

The first end-to-end artifact should be host-driven. It should own model-layer
metadata, allocate deterministic buffers, invoke kernels in model order, and
compare against CPU references. Later iterations can fuse launches or introduce
an AIR-level scheduler only after the host loop proves the data contracts.

### Top-level loop

```text
load_model_metadata()
load_or_generate_quantized_weights()
allocate_activation_buffers()
allocate_kv_cache()

if image_inputs:
  run_vision_prefill_or_host_fallback()
  append_visual_context_tokens()

for prompt_chunk in prompt_chunks:
  run_text_prefill(prompt_chunk)

while not done:
  token = sample_or_use_test_token()
  logits = run_text_decode(token)
  done = update_token_stream(logits)
```

For early validation, sampling should be deterministic or bypassed entirely.
Use synthetic inputs and fixed test tokens before attempting real tokenizer and
logits integration.

### Prefill layer flow

Each prefill layer should eventually execute:

```text
input_state
  -> RMSNorm or host fallback
  -> Q/K/V projection path
     - Q4NX dequantization where needed
     - BF16 tiled MM for projection
  -> RoPE and QK-Norm or host fallback
  -> append K/V to KV cache
  -> FlowQKV attention
     - causal full attention for global text layers
     - causal SWA for local text layers
     - non-causal full attention for vision path
  -> output projection
  -> residual update
  -> MLP path
     - gate/up/down projections
     - GeLU or host fallback
     - elementwise multiply or host fallback
  -> output_state
```

The first prefill milestone should not require every arrow to be on the NPU.
Mark each stage as `npu`, `host-fallback`, or `missing` in the implementation
metadata.

### Decode layer flow

Each decode layer should eventually execute:

```text
single_token_state
  -> RMSNorm or host fallback
  -> FusedDQP Q/K/V projection path
  -> RoPE and QK-Norm or host fallback
  -> append one K/V entry to KV cache
  -> FlowKV attention over current cache length
     - full cache for global layers
     - SWA window for local layers
  -> FusedDQP or BF16 output projection
  -> residual update
  -> FusedDQP or BF16 MLP projections
  -> nonlinear host fallback or NPU vector kernels
  -> output_state
```

For decode, `Q_CHUNK` is fixed to `1`. `FLOWKV_QUERY_BASE` should track the
current cache position. For local layers, `WINDOW_LEN` should restrict the
attention sweep once SWA is enabled.

### Layer pattern

Represent layer behavior explicitly:

```text
layer_index
attention_kind = local_swa or global_full
window_len = 1024 for local_swa, 0 for global_full
uses_gqa = true for text attention
kv_groups = PAPER_KV_GROUPS for paper-style attention experiments
heads_per_kv = PAPER_HEADS_PER_KV for paper-style attention experiments
```

Do not encode local/global behavior as ad hoc command-line conditionals spread
through the loop. Keep it in a layer metadata table so validation can enumerate
expected layer behavior.

## Shape And Buffer Contracts

Use compact defaults first:

- `OUTPUT_FORMAT=elf`
- `COMPILE_MODE=compile-only` before hardware
- `HERD_SHAPE=2x4` before `4x4`, before `8x4`
- `Q_CHUNK=4`
- `KV_LEN=32`
- `KV_CHUNK=32` for smoke mode
- `PAPER_KV_CHUNK=16` for paper-mode attention
- `HEAD_DIM=64`
- `WINDOW_LEN=0` for full attention smoke tests
- `PAPER_KV_GROUPS=4`
- `PAPER_HEADS_PER_KV=2`
- `PAPER_FUSED_DQP_COL_BLOCKS=2`

### Weight buffers

Q4NX weight blocks:

- Logical block shape in this repo: `Q4NX_ROWS=32`, `Q4NX_COLS=256`.
- Packed int4 weights use low nibble first, as documented in the historical
  kernel sketches.
- Each column has BF16 scale and BF16 minimum offset metadata.
- Dequantization formula used by the source kernels:

```text
w_bf16[row, col] = scale[col] * q4[row, col] + min[col]
```

Prefill projection path:

- Dequantize Q4NX blocks into BF16 tiles.
- Feed BF16 tiles to `bf16_tiled_mm.py`.
- Keep separate validation for dequant output and MM output before combining
  them in a layer loop.

Decode projection path:

- Use FusedDQP to avoid materializing full dequantized weights before MVM.
- Broadcast the activation vector to participating CTs.
- Each CT fetches its assigned Q4NX-aligned weight block and accumulates its
  output row block.

### Activation and residual buffers

Maintain two model-state buffers per layer boundary until in-place behavior is
validated:

- `layer_input_bf16`
- `layer_output_bf16`

Keep residual inputs alive across attention and MLP subpaths. Host fallbacks for
RMSNorm, RoPE, QK-Norm, GeLU, residual add, and logits should use these same
buffers rather than creating incompatible layouts.

### KV cache buffers

Track K and V separately:

```text
kv_cache[layer][kv_group][token_range][head_dim]
```

Minimum metadata:

- `layer_index`
- `kv_group`
- `head_dim`
- `cache_len`
- `window_len`
- `token_base`
- `tokens_valid`
- `layout_version`

For prefill:

- Append K/V for the whole prompt chunk.
- FlowQKV reads the current query chunk and all relevant previous K/V chunks.
- Causal mode must mask future tokens inside the current chunk.

For decode:

- Append a single K/V entry per layer for the current token.
- FlowKV reads one query and sweeps chunks of K/V up to the current cache
  position.
- SWA mode should clamp the chunk sweep to the most recent `WINDOW_LEN` tokens.

### Output routing

Use `l2-gather` as the default route for 8x4 public model-loop work. Direct
output may be used for 2x4 and 4x4 only where the support matrix allows it.

The model loop must never silently rewrite unsupported output modes. It should
fail early with the same classification used by the kernel drivers:

- hardware resource limit,
- packet S2MM backend limitation,
- AIE routing bug,
- channel/runtime scheduling bug.

## Iterative Implementation Phases

### Phase 0: Baseline evidence and document control

Artifacts:

- This document, including the Llama32 reference mapping.
- README link to this document.
- A support matrix copied from current Gemma kernel evidence.
- A known-failures table with unsupported-mode classifications.

Acceptance:

- Every local path named in this document exists or is clearly labeled future
  work.
- No clean rebuild, reboot, NPU power-mode change, or `/home/cj/mlir-aie` edit.
- `git diff --check` passes for documentation edits.

### Phase 1: Gemma config, weights, and CPU reference

Model this phase after `llama32_1b_weights.py` and `llama32_1b_reference.py`.

Milestones:

- Add a Gemma config dataclass for text-only synthetic inference first.
- Add per-layer metadata for `local_swa`, `global_full`, and future
  `vision_nca` layers.
- Add synthetic weights that include Q4NX block metadata and BF16 fallback
  weights.
- Add CPU references for Q4NX, BF16 MM, FusedDQP, FlowQKV, FlowKV, RMSNorm,
  RoPE, QK-Norm, activation, residual add, KV-cache update, and logits shape.
- Keep real safetensor loading and tokenizer integration disabled until the
  synthetic path is stable.

Acceptance:

- CPU reference tests run without importing `air` modules.
- A one-layer local/global synthetic config prints the exact kernel sequence and
  fallback status.
- Every tensor has a named shape contract before it is passed to an NPU kernel.

### Phase 2: Kernel cache and runtime preparation

Model this phase after Llama32 `KernelCache`, backend presets, external-kernel
compilation, and per-layer BO preloading.

Milestones:

- Add Gemma cache names for Q4NX, BF16 MM, FusedDQP, FlowQKV, FlowKV, and each
  promoted nonlinear kernel.
- Separate `compile-only`, `run-only`, `verify`, and `profile` flags.
- Pre-load static weights and mark static/intermediate buffers explicitly.
- Use per-layer BO keys for all model-loop kernels.
- Keep multi-launch stitching as a later optimization; start with isolated
  Gemma kernels and explicit host sequencing.

Acceptance:

- Compile-only mode can prepare the intended artifact list without running
  hardware.
- Run-only mode refuses to run if required artifacts are missing.
- The runtime preparation log records cache key, layer, kernel, output mode,
  schedule mode, and static input set.

### Phase 3: Synthetic text prefill

Model this phase after `llama32_1b_prefill.py`, substituting Gemma kernels and
layer metadata.

Milestones:

- Implement one synthetic text prefill layer with host nonlinear fallbacks.
- Use Q4NX dequantization and BF16 MM for projection preparation where needed.
- Use FlowQKV for attention with full causal and SWA variants.
- Append projected K/V tensors to the Gemma KV-cache layout.
- Compare each intermediate against the CPU reference.

Acceptance:

- One-layer prefill passes CPU-only reference checks.
- Compile-only passes before any hardware run.
- Hardware uses only public supported output modes.
- SWA and full-attention layers produce different documented KV sweep ranges.

### Phase 4: Synthetic decode

Model this phase after `llama32_1b_decode.py`, substituting FusedDQP and FlowKV.

Milestones:

- Implement one-token decode over an existing synthetic KV cache.
- Use FusedDQP for decode projections where supported.
- Apply RoPE and QK-Norm through host fallback or validated nonlinear kernels.
- Append one K/V entry per layer and update cache metadata.
- Use FlowKV with `Q_CHUNK=1` and public supported output modes only.

Acceptance:

- Decode step 0 after prefill reads the expected cache length.
- Repeated decode steps grow global-layer cache length monotonically.
- Local layers clamp cache reads to `WINDOW_LEN`.
- FlowKV small-shape `l2-gather` remains diagnostic-only.

### Phase 5: Promote nonlinear fallbacks

Promote nonlinear operations only after the operation policy table is satisfied.

Milestones:

- Reuse `weighted_rms_norm` for RMSNorm where layout-compatible.
- Reuse or wrap RoPE only after the Gemma rotation convention is documented.
- Add Gemma QK-Norm under `gemma3_dataflow_kernels` if no existing wrapper fits.
- Select GELU/GeGLU/SwiGLU from Gemma evidence, then reuse `gelu`, `ffn_swiglu`,
  or add a Gemma-specific activation kernel.
- Add residual/add/multiply vector kernels only when they reduce host fallback
  cost without changing tensor contracts.

Acceptance:

- Each promoted nonlinear has CPU reference, compile-only lit coverage, and a
  fallback path.
- No nonlinear is fused into the model loop before standalone validation.
- The document records which existing `programming_examples` source was reused
  or why a Gemma-specific implementation was required.

### Phase 6: Full transformer layer skeleton

Join attention, output projection, residual, MLP, and layer output into a
single synthetic transformer layer.

Milestones:

- Define exact buffer handoff between attention output and output projection.
- Define gate/up/down projection order for the MLP path.
- Use Llama32-style intermediate names and per-stage checksum logging.
- Keep host fallbacks explicit until each nonlinear has NPU evidence.

Acceptance:

- One synthetic layer runs prefill and decode with deterministic outputs.
- A two-layer local/global sequence runs without KV-cache aliasing.
- Layer output buffers can be reused across tokens without stale data.

### Phase 7: Multi-layer text model loop

Scale the layer skeleton to the Gemma text-layer pattern.

Milestones:

- Generate layer metadata programmatically.
- Support multiple prompt chunks.
- Support multiple decode tokens.
- Add failure recovery that reports layer, token, kernel, mode, and fallback.
- Keep tokenizer, real weights, and sampling optional.

Acceptance:

- Synthetic multi-layer loop runs with documented fallbacks.
- Logs are compact enough to compare across runs.
- Compile-only validation is available without hardware.

### Phase 8: Paper-style scaling and performance evidence

Only after compact correctness passes:

- Scale `HERD_SHAPE` from 2x4 to 4x4 to 8x4.
- Use `l2-gather` for full-physical 8x4 production routes.
- Enable paper targets:
  - `run-fused-dqp-paper`
  - `run-flowqkv-paper`
  - `run-flowkv-paper`
- Consider Llama32-style multi-launch ELF fusion only after the unfused sequence
  is correct.
- Add trace/timing collection only after correctness is stable.

Acceptance:

- Performance claims are separated from correctness claims.
- Every reported number has command, hardware state, XRT version, shape, and
  output-mode metadata.
- Timeout runs are followed by `xrt-smi examine -r all`.

### Phase 9: Vision path

Treat the vision tower as later scope.

Milestones:

- Define image-token input contract.
- Use non-causal FlowQKV-style attention for vision attention experiments.
- Produce visual context tokens that can seed text prefill.
- Keep the text-only loop as the regression baseline.

Acceptance:

- Vision path can be disabled entirely.
- Text-only results are unchanged when vision support is compiled out or unused.

## Verification Commands

Documentation and path checks:

```bash
git diff --check -- programming_examples/gemma3_herd_sweep/GEMMA3_MODEL_LOOP_IMPLEMENTATION.md
rg -n "Llama32 Inference Reference Pattern|Nonlinear Operation Policy" programming_examples/gemma3_herd_sweep/GEMMA3_MODEL_LOOP_IMPLEMENTATION.md
test -d programming_examples/llama32_1b
test -d programming_examples/gemma3_dataflow_kernels
test -d programming_examples/weighted_rms_norm
test -d programming_examples/rope_sincos
test -d programming_examples/gelu
test -d programming_examples/ffn_swiglu
```

Existing nonlinear candidates to evaluate before adding Gemma-specific kernels:

```bash
make -C programming_examples/weighted_rms_norm run OUTPUT_FORMAT=elf
make -C programming_examples/rope_sincos run AIE_TARGET=aie2p OUTPUT_FORMAT=elf
make -C programming_examples/gelu run OUTPUT_FORMAT=elf
make -C programming_examples/ffn_swiglu/decode run OUTPUT_FORMAT=elf
```

These standalone targets are reuse candidates, not Gemma dependencies yet. Add
focused compile-only lit coverage for any nonlinear before the Gemma model loop
depends on it.

Compiler tools:

```bash
ninja -C build-xrt air-opt aircc
```

Compile sweep:

```bash
make -C programming_examples/gemma3_herd_sweep sweep COMPILE_MODE=compile-only OUTPUT_FORMAT=elf
```

Gemma lit coverage:

```bash
sandbox/bin/lit -v --filter=gemma3_herd_sweep build-xrt/programming_examples
```

Targeted compile-only examples:

```bash
make -C programming_examples/gemma3_herd_sweep run-q4nx HERD_SHAPE=2x4 COMPILE_MODE=compile-only OUTPUT_FORMAT=elf
make -C programming_examples/gemma3_herd_sweep run-mm HERD_SHAPE=4x4 COMPILE_MODE=compile-only OUTPUT_FORMAT=elf
make -C programming_examples/gemma3_herd_sweep run-fused-dqp HERD_SHAPE=8x4 COMPILE_MODE=compile-only OUTPUT_FORMAT=elf
make -C programming_examples/gemma3_herd_sweep run-flowqkv HERD_SHAPE=4x4 COMPILE_MODE=compile-only OUTPUT_FORMAT=elf
make -C programming_examples/gemma3_herd_sweep run-flowkv HERD_SHAPE=2x4 COMPILE_MODE=compile-only OUTPUT_FORMAT=elf
```

Paper-mode compile-only examples:

```bash
make -C programming_examples/gemma3_herd_sweep run-fused-dqp-paper COMPILE_MODE=compile-only OUTPUT_FORMAT=elf
make -C programming_examples/gemma3_herd_sweep run-flowqkv-paper COMPILE_MODE=compile-only OUTPUT_FORMAT=elf
make -C programming_examples/gemma3_herd_sweep run-flowkv-paper COMPILE_MODE=compile-only OUTPUT_FORMAT=elf
```

Hardware validation uses the same targets with:

```bash
COMPILE_MODE=compile-and-run
```

Before hardware runs:

- Source the MLIR-AIR environment.
- Source XRT setup after the MLIR-AIR environment so `pyxrt` remains on
  `PYTHONPATH`.
- Do not wrap targets in an outer `/tmp/npu.lock`; `XRTRunner` already owns
  that lock.
- Use `DEBUG_IR=1` only when retaining lowering artifacts is useful.

After any timeout or packet-route diagnostic run:

```bash
xrt-smi examine -r all
```

## Evidence Log Template

Every model-loop milestone should record:

```text
date:
git_commit:
dirty_worktree_summary:
build_dir:
MLIR_AIR_INSTALL_DIR:
MLIR_AIE_INSTALL_DIR:
PEANO_INSTALL_DIR:
XRT_version:
target:
command:
compile_mode:
output_format:
herd_shape:
kernel:
output_mode:
schedule_mode:
kv_staging:
q_chunk:
kv_len:
kv_chunk:
head_dim:
window_len:
result: PASS | COMPILE_FAIL | VALIDATION_FAIL | TIMEOUT | UNSUPPORTED
failure_class:
reference_tolerance:
output_checksum:
debug_ir_path:
notes:
```

Failure classes:

- `hardware-resource-limit`
- `packet-s2mm-backend-limitation`
- `aie-routing-bug`
- `channel-runtime-scheduling-bug`
- `reference-mismatch`
- `shape-contract-error`
- `host-fallback-missing`

## Guardrails

- Do not expose a kernel mode to sweeps until compile and hardware evidence is
  clean.
- Do not silently rewrite unsupported modes in the model loop.
- Do not edit `/home/cj/mlir-aie` without explicit approval.
- Do not clean rebuild, delete build directories, reboot, change NPU power mode,
  or revert user changes without explicit approval.
- Prefer incremental `ninja`, focused lit tests, and one programming example
  while iterating.
- Keep generated XRT, xclbin, ELF, and debug IR artifacts out of source commits
  unless they are intentionally added as tests.
- Treat compile failures and runtime failures separately. A clean compile does
  not imply hardware support.
- Treat paper performance numbers as external reference points only. Local
  performance claims require local logs and reproducible commands.

## Open Implementation Risks

- The current source tree does not yet provide a complete Gemma model-runtime
  abstraction for layers, token loops, tokenizer/logits/sampling, or model
  weight loading; `llama32_1b` is the reference structure to adapt.
- Nonlinear operations are not yet integrated as validated NPU kernels in this
  Gemma loop, and Gemma semantics must be confirmed before copying Llama
  activation or RoPE behavior.
- Real Gemma3 weight import, safetensor mapping, and Q4NX packing need a
  separate compatibility document before accuracy claims.
- FlowKV small-shape `l2-gather` and FusedDQP `pipeline` are useful diagnostics
  but must not be used as public model-loop defaults.
- Packet S2MM routing may require backend work outside this repository. Stop and
  ask before editing MLIR-AIE.
- Full text-plus-vision Gemma3 integration should not block text-only loop
  validation.

## First Implementable Checklist

1. Keep this document linked from the Gemma herd-sweep README.
2. Add the Gemma config/weights/reference split modeled after `llama32_1b`.
3. Add a shape/layer manifest that can describe one synthetic local layer and
   one synthetic global layer.
4. Add CPU references for Q4NX, BF16 MM, FusedDQP, FlowQKV, FlowKV, RMSNorm,
   RoPE, QK-Norm, the selected MLP activation, residual add, KV-cache update,
   and logits shape.
5. Add a KernelCache/runtime-preparation layer modeled after Llama32, including
   compile-only and run-only modes.
6. Build a host-driven synthetic prefill loop over validated Gemma kernels and
   explicit nonlinear fallbacks.
7. Build a one-token decode loop over a synthetic KV cache.
8. Promote nonlinear fallbacks only by reusing existing `programming_examples`
   implementations or adding validated kernels under `gemma3_dataflow_kernels`.
9. Scale to 4x4 and then 8x4 using public supported output modes.
10. Only then collect timing, multi-launch fusion, real-weight, tokenizer, and
    paper-style scaling evidence.
