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
- `gemma3_real_execution.py`: CPU/HF real-artifact smoke and small benchmark
  paths for the 1B, 4B text, and 4B synthetic-image model paths, proving local
  weights, tokenizer, and processor can execute without AIR imports while
  making no paper timing claim.
- `gemma3_npu_preflight.py`: real-shape NPU preflight planner that derives
  projection padding, Q4NX block counts, attention metadata, and the remaining
  NPU execution blocker from local artifacts.
- `gemma3_npu_wiring.py`: real-shape per-layer execution wiring manifest that
  records prefill/decode stage roles, NPU kernel candidates, host fallbacks,
  attention windows, and the remaining launch blockers without claiming
  execution.
- `gemma3_weight_plan.py`: text-stack static projection-weight planning for real
  safetensors, including Q4NX padded block counts and packed weight/scale/min
  byte estimates for future BO preloading.
- `gemma3_bo_plan.py`: shape-only activation, benchmark-cell per-layer
  KV-cache, intermediate, logits, and static-weight BO planning for future XRT
  allocation and binding, with a monolithic KV strategy retained for diagnostic
  reproduction.
- `gemma3_xrt_runner.py`: capped and full-plan `pyxrt` BO allocation/preload
  smoke runner that exercises real XRT allocation without claiming kernel
  execution or paper-shape runtime.
- `gemma3_static_preload.py`: real safetensor-to-Q4NX serialization and XRT BO
  preload smoke for selected projection tensors, plus full-model evidence
  recognition when every planned projection tensor is serialized and written.
- `gemma3_buffer_binding.py`: runtime buffer-binding manifest that assigns
  persistent BO keys, virtual intermediate keys, static-weight families, and
  layer-specific mutable KV-cache buffers per model stage.
- `gemma3_argument_binding.py`: deterministic positional kernel argument-layout
  validation for every NPU candidate stage, including persistent BOs, static
  BOs, mutable KV buffers, virtual intermediates, shapes, dtypes, directions,
  and missing-storage diagnostics.
- `gemma3_launch_probe.py`: diagnostic first-kernel launch probe for the
  promoted Gemma3 1B pre-attention RMSNorm stage; hardware is touched only with
  `--run-hardware`, while lit covers parsing and evidence formatting.
- `gemma3_substep_probe.py`: diagnostic Gemma3 1B decode RMSNorm-to-`q_proj`
  substep probe. It launches real layer-0 RMSNorm and five FusedDQP q-projection
  col-blocks with real weights and runner-owned pyxrt BOs, then validates the
  accumulated q vector against CPU references.
- `gemma3_qkv_substep_probe.py`: diagnostic Gemma3 1B decode RMSNorm-to-Q/K/V
  substep probe. It launches real layer-0 RMSNorm plus real q/k/v FusedDQP
  col-block loops and validates the accumulated Q/K/V vectors against CPU
  references.
- `gemma3_full_layer_probe.py`: diagnostic Gemma3 1B staged decode layer
  probe with a selectable `--layer-index`. It launches pre-attention RMSNorm,
  Q/K RMSNorm, post-attention RMSNorm, pre/post-feedforward RMSNorm, and all
  seven projection families on the NPU through the weighted RMSNorm and FusedDQP
  wrappers; layer 0 also launches RoPE, single-token FlowQKV attention,
  GeGLU, and both residual adds through Gemma standalone wrappers. No staged
  layer-0 operation remains as a host-reference stage in this diagnostic. RMSNorm uses a preselected BF16
  norm-vector argument until the two-argument RMSNorm ABI grows static-BO
  offset/sub-BO plumbing.
- `gemma3_decode_loop_probe.py`: diagnostic Gemma3 1B decode loop probe.
  It preloads and repacks real layer weights before timing, warms one layer to
  compile/load reusable ELF runners and allocate runner-owned BOs, then measures
  decode tokens while recording both post-warmup loop wall time and segmented
  NPU `run.start()/wait2()` time. The default `--ingress-mode staged` path
  keeps the older separate RMSNorm/QKV/QK-Norm/RoPE launches for historical
  comparison. The explicit `--ingress-mode stitched` path now replaces that
  ingress group with `gemma3_decode_ingress_rms_qkv_qknorm_rope` and preloads
  one aliased stitched-ingress BO set per layer before timing. The route now
  passes both one-layer and 26-layer hardware diagnostics in
  `results/gemma3_1b_decode_loop_stitched_ingress_L1_probe.json` and
  `results/gemma3_1b_decode_loop_stitched_ingress_probe.json`. The explicit
  `--attention-o-mode stitched` path also integrates the standalone
  `attention -> O projection` slice into the 26-layer loop, with evidence in
  `results/gemma3_1b_decode_loop_stitched_ingress_attention_o_probe.json`.
  Explicit `--attention-mode tiled-stats-1k` exercises host-batched 1k tiled-stat
  attention with host-side softmax-stat reduction for blocker isolation.
- `gemma3_stitching.py`: Gemma-side adaptation of the Llama32 text-based
  multi-launch MLIR stitching pattern. It extracts launch bodies, prefixes SSA
  values/maps/symbols, deduplicates preserved Peano external declarations, and
  remaps launch operands into one public function.
- `gemma3_padded_rms_norm.py`: earlier Gemma3 decode activation bridge that
  computes weighted RMSNorm and writes `5x256` directly. It remains
  hardware-free compile/self-test coverage; the stitched ingress production
  path now uses the proven weighted RMSNorm kernel with BO aliasing instead.
- `gemma3_projection_qk_norm.py`: projection-output view bridge plus Q/K
  RMSNorm. It treats FusedDQP's contiguous `32x32` Q output as `4x256` heads
  and `8x32` K output as `1x256`, using zero-copy memref collapse/expand inside
  the launch before applying weighted RMSNorm.
- `gemma3_stitched_decode.py`: active stitched-ELF decode track for real Gemma3
  text inference. The current integrated loop slices are the full decode ingress
  `gemma3_decode_ingress_rms_qkv_qknorm_rope`, the post-ingress
  `gemma3_decode_attention_o_projection` slice, and the post-attention residual
  `gemma3_decode_post_attention_residual` slice. The ingress is an eight-launch
  stitched ELF covering `RMSNorm -> Q/K/V projections -> Q/K Norm -> RoPE`; it
  aliases the RMSNorm output and padded activation view to the same zero-tailed
  BO, avoiding a separate pad-copy kernel and removing host activation packing,
  host col-block accumulation, and the projection-output layout bridge from the
  timed contract for that slice. Hardware correctness against real layer-0
  HF/reference tensors is recorded in
  `results/gemma3_1b_stitched_decode_ingress_probe.json`. The attention/O slice
  stitches single-token FlowQKV attention into O projection with an aliased
  `1x4x256`/`4x256` attention-output BO; standalone evidence is recorded in
  `results/gemma3_1b_stitched_attention_o_probe.json` and 26-layer loop evidence
  is recorded in
  `results/gemma3_1b_decode_loop_stitched_ingress_attention_o_probe.json`. The
  post-attention residual slice stitches post-attention RMSNorm into the
  attention residual add with an aliased `1x1152`/`1152` norm-output BO;
  standalone hardware evidence is recorded in
  `results/gemma3_1b_stitched_post_attention_residual_probe.json`, and 26-layer
  loop evidence is recorded in
  `results/gemma3_1b_decode_loop_stitched_ingress_attention_o_post_attention_probe.json`.
- `gemma3_model_runner.py`: launch-order manifest that composes BO planning,
  static-preload planning, buffer bindings, argument layouts, and per-layer
  kernel/fallback wiring without claiming kernel execution.

## Stitched ELF Inference Track

Stitched ELF real text inference is now the main implementation track. The
older first-kernel, full-layer, decode-loop, and tiled-stat probes remain
validation/support artifacts only; they should not drive new paper-parity timing
claims unless the same behavior has been promoted into stitched inference.

The target decode ingress ELF is:

```text
RMSNorm -> Q/K/V projections -> Q/K Norm -> RoPE
```

The current implemented stitched slice is the full decode ingress through RoPE:

```text
layer_input:1x1152 + input_norm_weight:1152
  -> weighted RMSNorm -> rms_out:1x1152
  -> rms_out/padded_activation shared zero-tailed BO alias -> activation_padded:5x256
  -> q_proj full-col-block FusedDQP l2-gather -> q:32x32
  -> k_proj full-col-block FusedDQP l2-gather -> k:8x32
  -> v_proj full-col-block FusedDQP l2-gather -> v:8x32
  -> Q/K projection-output view + RMSNorm -> q_norm:4x256, k_norm:1x256
  -> Q/K half-split RoPE -> q_rope:4x256, k_rope:1x256
```

This slice is intentionally shaped around the paper-style FusedDQP builder with
`col_blocks=5`, not the older per-column-block diagnostic loop. It uses explicit
`l2-gather` projection output and a runtime BO alias for the RMSNorm output and
padded activation view. It removes host activation packing, host col-block
accumulation, and host projection-output layout conversion from the ingress
path. Current local evidence is parse, compile-only ELF, and one real layer-0
hardware run of `gemma3_decode_ingress_rms_qkv_qknorm_rope` with 18 public BO
arguments and eight stitched `air.launch` regions. The clean-provenance result
records `dirty_worktree=false`, one 0.009281 s diagnostic `run.start()/wait2()`
window, and correlations at or above 0.999957 for input RMSNorm, padded
activation, Q/K/V projection, Q/K norm, and Q/K RoPE outputs. This is still not
a TTFT/TPS or paper-parity result.

Remaining stitched decode work:

- Tune or restructure the stitched ingress route because the 26-layer diagnostic
  reduces launch count but does not yet improve kernel-only TPS over the staged
  baseline.
- Tune the integrated stitched ingress plus attention/O/post-attention route
  because it improves loop-wall timing but still trails the staged baseline on
  kernel-only TPS.
- Stitch the remaining FFN tail: pre-feedforward RMSNorm, gate/up projections,
  GeGLU, down projection, post-feedforward RMSNorm, and final residual.
- Wire real prefill-produced KV cache before collecting paper-comparison
  TTFT/TPS/power numbers.

Timing policy for stitched inference:

- Compile, ELF load, BO creation, static weight preload, and kernel argument
  binding validation stay outside the timed region.
- Timed prefill TTFT covers only the prefill inference execution path.
- Timed decode TPS covers only post-prefill decode token execution.
- Any host-side bridge, reduction, logits, or sampling that remains in a result
  must be called out explicitly and cannot be used as a paper-parity claim.

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
- `run_model_loop_npu_wiring.lit`
- `run_model_loop_weight_plan.lit`
- `run_model_loop_bo_plan.lit`
- `run_model_loop_xrt_runner.lit`
- `run_model_loop_static_preload.lit`
- `run_model_loop_model_runner.lit`
- `run_model_loop_buffer_binding.lit`
- `run_model_loop_argument_binding.lit`
- `run_model_loop_launch_probe.lit`
- `run_model_loop_substep_probe.lit`
- `run_model_loop_qkv_substep_probe.lit`
- `run_model_loop_full_layer_probe.lit`
- `run_model_loop_decode_loop_probe.lit`
- `run_model_loop_stitching.lit`
- `../gemma3_dataflow_kernels/run_geglu_compile_only.lit`

Current phase status:

| Phase | Status | Evidence |
| --- | --- | --- |
| Phase 0 | Complete | README link, support matrix, known unsupported-mode classes, and this status section are present. |
| Phase 1 | Complete | Config, weights, and CPU references are implemented without AIR imports. |
| Phase 2 | Complete | Manifest preparation supports compile-only/run-only and refuses missing or mismatched artifacts. |
| Phase 3 | Complete | Two-chunk synthetic prefill records Q4NX/BF16/FlowQKV stages and distinct local/global KV sweeps. |
| Phase 4 | Complete | Repeated one-token decode grows cache lengths and records FlowKV-compatible stage metadata. |
| Phase 5 | Complete for standalone promotion readiness | RMSNorm/QK-Norm reuse, RoPE half-split AIR wrapper reuse, GeGLU, residual-add, and fallback policy are documented and tested. |
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
| Model weights | Synthetic weights plus local real Gemma3 artifact discovery, full static projection preload evidence, and norm-weight preload evidence | Real Gemma3 1B and 4B weights, quantized/packed in the same Q4NX contract used by the NPU kernels |
| Tokenizer and prompts | Synthetic token IDs plus real tokenizer execution for initial 1B CPU/iGPU 1k baseline cells | Real tokenizer, deterministic prompts, and sequence lengths 1k-32k for 1B and 1k-128k decode for 4B |
| Text runtime | Host-driven synthetic loop with CPU references, manifests, real-shape preflight, and per-layer NPU wiring metadata | End-to-end NPU execution for all validated model substeps, with host fallbacks removed or measured separately |
| Vision runtime | Disabled contract plus synthetic CPU non-causal vision prefill and visual context token handoff contract | 4B vision prefill path with non-causal attention validated on NPU and timed against paper |
| Nonlinear operations | RMSNorm, QK-Norm, RoPE, residual add, and GeGLU have standalone NPU evidence and layer-0 staged 1B launch evidence with `host_fallbacks=[]`; host fallback microbenchmarks remain recorded in result JSON | Full-loop/paper-shape 1B coverage, 4B/vision coverage, logits, and sampling either validated on NPU or explicitly accounted for in timing |
| Baselines | Initial 1B 1k CPU/HF and iGPU/HF ROCm cells are measured; iGPU prefill matches the paper 1k iGPU target, CPU and decode cells are explained deviations | CPU and iGPU runs for every paper model variant, prompt length, output count, tokenizer, and measurement window |
| Timing | Initial 1B 1k CPU/iGPU TTFT and decode TPS exclude load/setup; NPU has staged layer and decode-loop diagnostics but no official paper-cell local value yet | TTFT and decode TPS with warmup, timed iterations, correctness, and compile/setup excluded |
| Power | CPU cells use direct RAPL package energy; iGPU cells use ROCm SMI GPU rail; staged NPU diagnostics record pseudo-NPU RAPL package deltas, but official NPU paper-cell power is still pending | CPU/GPU/NPU/total watt readings aligned with benchmark windows and TPS/W calculations |
| Accuracy | Synthetic checksums only | Model-output agreement against CPU reference and paper-compatible prompts, plus tolerance policy for quantized outputs |
| Result comparison | Paper target ledger, initial 1B 1k result bundle, and Markdown/CSV summaries exist | Machine-readable paper/local comparison for every paper cell with percent deltas and pass/deviation labels |

## Iterative Paper-Match Implementation Loop

Each iteration should do exactly one paper-parity increment and should finish
with a committed evidence update before the next increment starts.

1. Start from a clean git worktree, or explicitly record unrelated dirty files
   before doing any work.
2. Run `gemma3_reproduction_blockers.py` and select exactly one blocker or one
   missing paper cell.
3. Add the smallest implementation needed for that blocker or cell.
4. Run compile-only checks before hardware.
5. Run correctness before timing.
6. Run timing before power/TPS/W.
7. Save the command, environment, result JSON, comparison summary, and relevant
   logs.
8. Update this document's evidence ledger and the small result-artifact README.
9. Commit that iteration before moving to the next blocker or paper cell.

Do not broaden public support matrices, sweep modes, output-mode exposure, or
paper claims until the same mode passes compile, correctness, hardware, and
paper-comparison checks.

### Current next-loop priorities

The next implementation loops should stay on 1B 1k NPU text before expanding to
4B, vision, or longer context lengths.

| Priority | Target | Done when |
| ---: | --- | --- |
| 1 | Complete: resolve `model-kernel-argument-binding-not-validated` | `gemma3_argument_binding.py --self-test` validates 56 fixture NPU candidate layouts with 172 positional args and no missing storage; the real 1B 1k/32k-context plan validates 728 NPU candidate layouts with 2,236 positional args and zero argument-binding blockers. |
| 2 | Partially complete: resolve `model-kernel-launch-not-wired` | The first promoted Gemma3 1B pre-attention RMSNorm shape (`1024x1152`, ELF) launches on the NPU with correlation 0.999983 using the validated first-stage positional layout (`layer_input`, `static_norm_weights`, `prefill_L0_pre_attention_norm`), a full contiguous static norm payload whose layer-0 vector is at byte offset 0, and runner-owned pyxrt BO allocation/binding. The decode RMSNorm-to-`q_proj` substep passes with RMSNorm correlation 0.999991, q-projection correlation 1.000000, and dense original-weight correlation 0.994609. The decode RMSNorm-to-Q/K/V substep passes with Q/K/V projection correlations all 1.000000 and dense original-weight correlations 0.994609/0.995959/0.995720. The staged decode layer-0 probe now launches pre-attention RMSNorm, Q/K RMSNorm, post-attention RMSNorm, pre/post-feedforward RMSNorm, q/k/v/o/gate/up/down projection families, RoPE, single-token FlowQKV attention, GeGLU/MLP activation, and both residual adds on the NPU; no operation remains as a host-reference stage in that layer-0 probe. Layer 1 exposed and fixed the missing static-norm offset/sub-BO path by using a preselected BF16 norm-vector argument (`model.layers.1.input_layernorm.weight` at byte offset 10240 in the contiguous norm BO, passed as a 2304-byte argument). The staged 26-layer decode-loop diagnostic now measures one post-warmup token across all 26 real layers, preloads packed projection inputs into 1,456 runner-owned BO sets before timing, and uses no-allocation static metadata placeholders in the timed projection path. It now launches QK-Norm, RoPE, single-token FlowQKV attention, post/pre/post RMSNorm, GeGLU, and both residual adds on the NPU for every layer and records `host_fallbacks=[]`. The stitched-ingress decode-loop mode now passes both one real layer and all 26 layers on hardware from clean trees: the one-layer artifact preloads one aliased stitched ingress BO set plus 41 remaining staged projection BO sets, validates RMSNorm/QKV/final-output correlations at or above 0.999967, and records 49 NPU launch windows totaling 0.146627 s; the 26-layer artifact preloads 26 stitched ingress BO sets plus 1,066 remaining staged projection BO sets, validates every layer with no blockers, and records 1,274 NPU launch windows totaling 3.791450 s. A standalone `attention -> O projection` stitched slice also passes hardware with attention/O correlations 0.999999/0.999991 and a 0.007962 s single-run window, and the same slice is now integrated into the 26-layer decode-loop diagnostic: it reduces the measured run to 1,170 launch windows totaling 3.763859 s and improves loop-wall TPS to 0.194062, while still trailing the staged baseline on kernel-only TPS. A standalone `post-attention RMSNorm -> residual add` stitched slice now passes hardware with correlations 0.999989/0.999955 and a 0.000291 s single-run window, and the same slice is now integrated into the 26-layer decode-loop diagnostic: it reduces the measured run to 1,144 launch windows totaling 3.706862 s and improves loop-wall TPS to 0.205379, while kernel-only TPS is 0.269770. The 26-layer loop also runs an untimed all-layer reference pass before the measured loop, so the measured loop excludes CPU reference/correlation checks. It remains diagnostic because the default paper-result bundle still uses the single-token attention path, logits/sampling are not wired, and the production contiguous static-weight BO route is not complete. An explicit `tiled-stats-1k` decode-loop mode integrates the 1k tiled-stat attention diagnostic across all 26 layers: it passes hardware with a synthetic prefill-shaped KV cache, 16 host-batched attention launches per layer, and host-side softmax-stat reduction. That result narrows the attention blocker to prefill-produced KV-cache construction and NPU-side reduction rather than first loop integration. For 1B, committed evidence now narrows the real-artifact blocker to tuning the stitched ingress plus attention/O/post-attention route, stitching the FFN layer tail, paper-shaped prefill/decode loop integration, and timed paper-cell measurement. |
| 3 | Complete for 1B staged layer: retire `nonlinear-model-stage-promotion-incomplete` from real 1B blocker reports | Standalone NPU evidence covers RMSNorm, QK-Norm, RoPE, GeGLU, and residual add, and the layer-0 staged full-layer probe now records `host_fallbacks=[]`. `gemma3_npu_wiring.py`, `gemma3_model_runner.py`, and `gemma3_reproduction_blockers.py` now report narrowed 1B blockers: `prefill-1k-npu-not-wired`, `prefill-produced-kv-cache-not-wired`, `npu-attention-reduction-not-wired`, `logits-sampling-not-wired`, and `production-contiguous-static-weight-bo-not-used-by-fused-dqp-route`. 4B text and 4B vision keep `nonlinear-model-stage-promotion-incomplete` until equivalent composed evidence exists. Remaining 1B paper-cell work is production prefill, real KV-cache handoff, NPU-side attention-stat reduction, logits/sampling treatment, production static-BO routing, and timed prefill/decode loop execution. |
| 4 | Re-run 1B 1k NPU paper cells | Prefill and decode result JSONs contain real local NPU TTFT/TPS or a narrower, artifact-backed failure classification. |
| 5 | Partially complete: capture pseudo-NPU power | Direct RAPL is readable when the run is launched under `sg power`. The refreshed layer-0 staged full-layer diagnostic records segmented package-energy deltas over only NPU `run.start()/wait2()` windows: 0.154274 s across 68 kernel launches after adding staged single-token FlowQKV attention on top of Q/K and post/pre/post RMSNorm launches, RoPE, GeGLU, and residual adds, 19.027 W segmented package power, and 4.598 W pseudo-NPU package-delta from a 14.429 W quiescent sample. The layer-1 diagnostic records 0.142578 s across 57 launch windows, 17.420 W segmented package power, and 5.918 W pseudo-NPU package-delta from an 11.502 W quiescent sample. The staged 26-layer decode-loop diagnostic records a post-warmup full-loop RAPL window of 16.790 W package power and 7.494 W pseudo-NPU package-delta while measuring 0.184746 diagnostic loop-wall TPS. The clean-provenance `tiled-stats-1k` decode-loop diagnostic records 17.577 W package power and 0.473 W pseudo-NPU package-delta while measuring 0.027921 diagnostic loop-wall TPS and 0.030363 kernel-only diagnostic TPS across 2,158 NPU launch windows with a synthetic prefill-shaped KV cache. Official paper-cell pseudo-NPU power remains blocked until paper-shaped prefill/decode execution exists. |
| 6 | Expand cautiously | Only after 1B 1k NPU correctness, timing, and pseudo-power evidence is clean should the loop expand to more 1B lengths, 4B text, or vision. |

### Blocker-resolution decision tree

Use the first failing artifact to classify the loop result. Do not continue to
timing or power after a compile, launch, or correctness failure.

| Failure point | Required evidence | Classification to use |
| --- | --- | --- |
| AIR/NPU compile fails | Exact `aircc` command, `-v`, `--debug-ir` directory, last good IR, and whether the failure is AIR transform, AIR-to-AIE, Peano, xclbin/ELF packaging, or host build | compiler/lowering blocker |
| XRT artifact load fails | xclbin/ELF/insts paths, XRT error text, target device, artifact format, and XRT version | runtime artifact-load blocker |
| Kernel launch fails | Kernel name, launch order, BO binding manifest, argument layout, XRT error text, and layer/stage/token | launch/binding blocker |
| Correctness fails | Layer/stage/token, tensor shape, expected/reference checksum or max error, observed checksum or max error, quantization route, output mode, and tolerance | correctness blocker |
| Timeout or hang | Command, logs, last emitted stage, output mode, artifact format, and `xrt-smi examine -r all` after the timeout | channel/runtime scheduling blocker |
| Performance gap after correctness passes | Result JSON, comparison summary, warmup/timed iterations, runtime path, output mode, trace setting, and likely root cause | `EXPLAINED_DEVIATION` only if evidence is concrete |

### NPU 1B 1k first-measurement runbook

The first real NPU measurement should be built in staged proof points. Do not
skip directly from manifests to full-model timing.

1. Confirm prerequisites: full paper-shape BO allocation, full static projection
   preload, norm preload, buffer-binding manifest, and model-runner manifest are
   all present for `gemma3-1b`.
2. Generate or validate concrete argument-layout records for the 1B prefill
   and decode NPU candidates. This is now implemented for storage, direction,
   shape, dtype, layer index, and KV-buffer identity; static-weight sub-offsets
   still need kernel-specific ABI plumbing during launch wiring.
3. Launch one real kernel with real BO bindings and compare its output against
   the existing CPU reference for that stage. Save a small JSON/log artifact
   even if it fails.
4. Extend from one kernel to one substep sequence, preserving intermediate
   correctness checks and per-stage logs. Decode RMSNorm-to-`q_proj` and
   decode RMSNorm-to-Q/K/V are now validated as staged substep probes; full
   layer execution remains before the qkv stage can be used in a timed loop.
5. Extend from one substep sequence to one full transformer layer with host
   fallbacks still explicit.
6. Extend from one layer to full 1B 1k prefill. Time only after correctness
   passes.
7. Build the 1k KV cache outside the timed decode window, then time 16
   token-by-token NPU decode steps for the initial decode TPS cell.
8. Refresh `gemma3_1b_npu_prefill_1k_blocked_initial.json` and
   `gemma3_1b_npu_decode_1k_blocked_initial.json` into measured NPU result cells
   only when they contain real local execution evidence.
9. Rebuild `gemma3_1b_initial_1k_results.json`,
   `gemma3_1b_initial_1k_summary.md`, and
   `gemma3_1b_initial_1k_summary.csv` after any NPU result-cell change.

### Measurement-window and telemetry policy

Result JSONs must describe exactly what is timed.

- Prefill TTFT excludes compile, model load, tokenizer work, input construction,
  device placement, BO creation, BO preload, xclbin/ELF load, and kernel
  argument binding.
- Decode TPS excludes prefill and KV-cache construction; it times only the
  token-by-token decode loop for the requested number of decode tokens.
- CPU power uses direct RAPL package energy and maps package watts to CPU and
  total for CPU-only baseline cells.
- iGPU power uses ROCm SMI socket graphics package power and leaves CPU/total
  rails `MISSING_POWER_FIELD` unless a separate package/CPU rail sampler is
  explicitly added.
- NPU power uses pseudo-NPU package delta: direct RAPL package watts during the
  timed NPU window minus direct RAPL quiescent package watts sampled immediately
  before the run.
- Never combine timing from one run with power from another unless both run IDs
  are recorded and the method is documented in the result JSON.

### Clean-provenance and commit policy

- Commit implementation code before final measurement runs when feasible.
- If final measurement output would make the worktree dirty before later cells
  run, write final JSONs to `/tmp` from a clean tree, verify
  `dirty_worktree: false`, then copy them into `results/`.
- Every committed result bundle should have clean provenance where feasible.
- Each loop should end with one focused commit. Do not mix unrelated user
  changes, broad refactors, or stale generated artifacts into the same commit.
- If an iteration discovers a blocker instead of fixing it, commit the narrower
  blocker evidence and update this document with the exact next smallest step.

### Evidence-update checklist

For each iteration, update only the sections whose evidence changed:

- Phase F/G/H blocker text when blocker status changes.
- Phase I when result schema, result bundles, comparison summaries, or report
  behavior changes.
- Phase J when power telemetry behavior or availability changes.
- `results/README.md` when a small committed result artifact is added or
  refreshed.
- `gemma3_1b_initial_1k_results.json` and its Markdown/CSV summaries when any
  initial 1B 1k CPU/iGPU/NPU result cell changes.
- `gemma3_paper_compare.py --compare` output should be regenerated for changed
  result bundles.

### Do not do yet

- Do not expand to 4B or vision timing before 1B 1k NPU launch, correctness,
  timing, and pseudo-power evidence exists.
- Do not expose diagnostic output modes as public supported modes without
  compile and hardware evidence.
- Do not silently rewrite unsupported output modes; keep early diagnostics.
- Do not use 8x4 direct output as a production path unless new hardware evidence
  disproves the current shim S2MM resource-limit classification.
- Do not perform a clean rebuild, reboot, NPU power-mode change, or
  `/home/cj/mlir-aie` edit without explicit approval.

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
- Real text projection-weight static BO planning records every Q4NX projection tensor
  shape, padded block count, and packed weight/scale/min byte estimate.

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
- `gemma3_weight_plan.py` scans real text-stack safetensor metadata without materializing
  full tensors and emits the static projection BO plan needed before model
  runner weight preloading.

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
- `flowqkv_tiled_stats.py` and `flow_attention_stats.cc` add a diagnostic
  tiled decode-attention path for the Gemma3 1B 1k shape. The direct full-cache
  FlowQKV wrapper still fails compile-only because it stages two
  `1024x256xbf16` KV tensors in one tile L1. The new stats wrapper keeps a
  `32x256xbf16` KV tile in L1, emits per-tile softmax stats, and merges those
  stats outside the kernel. Compact Strix/XRT evidence is saved in
  `results/gemma3_flowqkv_tiled_stats_1k_smoke.json`: `kv_len=1024`,
  `kv_tile=32`, `head_dim=256`, 16 two-tile host batches, stats correlation
  1.000000, 0.0% stats mismatches, and merged attention correlation 0.999958
  versus exact CPU attention. This narrows the attention blocker to production
  reduction/model-loop integration. It does not yet make a paper-cell claim: the
  full 8x4 direct stats route hits the known shim S2MM resource limit, and the
  attempted full-herd L2-gather route hits an AIE routing packet-id-0 blocker.

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
- The registry keeps nonlinear/vector stages conservative until
  Gemma-specific model wiring uses validated kernels. GeGLU/MLP activation now
  enters the model wiring as an NPU launch candidate using its 1B-sized standalone
  ELF hardware-smoke evidence, and layer-0 staged model launch validates that
  route; full paper-cell timing still waits for end-to-end loop wiring. RMSNorm and QK-Norm now enter the wiring as
  `weighted_rms_norm` NPU candidates where standalone evidence matches the
  shape contract: 1B RMSNorm rows use the M=8/N=1152 smoke, 4B text/vision
  RMSNorm rows use the M=8/N=2560 smoke, and QK-Norm uses the flattened
  per-head M=32/N=256 smoke. Residual add now has a Gemma-specific BF16 AIR
  wrapper with compile-only lit coverage and a Strix/XRT ELF hardware smoke
  for n=1152/tile_n=288, so attention and MLP residual stages enter wiring as
  NPU candidates. RoPE now has a Gemma half-split AIR wrapper with compile-only
  lit coverage and a Strix/XRT ELF hardware smoke for rows=4/head_dim=256, so
  RoPE stages also enter wiring as NPU candidates.
- `gemma3_results.py` now records fallback entries with backend, elapsed-ms,
  timed-iteration count, measurement source, tensor contract, hardware status,
  and `npu_promoted=false` for CPU-reference fallbacks.
- `gemma3_paper_compare.py` rejects `PAPER_MATCH` for timed paper metrics when
  local result JSON declares an unmeasured host fallback that contributes to the
  timed window; measured host fallback records are accepted as accounted timing
  but do not claim NPU promotion.
- `run_model_loop_nonlinears.lit`, `run_model_loop_npu_wiring.lit`,
  `run_model_loop_model_runner.lit`, `run_model_loop_results.lit`, and
  `run_model_loop_paper_compare.lit` cover the nonlinear metadata, residual-add
  NPU candidate wiring, measured result records, and paper-match fallback gate.
- `programming_examples/gemma3_dataflow_kernels/residual_add.py` provides the
  standalone BF16 residual-add AIR wrapper.
  `programming_examples/gemma3_dataflow_kernels/run_residual_add_compile_only.lit`
  covers ELF compile-only validation, and
  `results/gemma3_residual_add_smoke.json` records the n=1152/tile_n=288
  hardware smoke pass.
- `gemma3_norm_weight_plan.py` now records the BF16 vector static-input
  contract needed before RMSNorm and QK-Norm can be promoted: 1B has 156 norm
  tensors totaling 266,240 bytes, while 4B text and the 4B vision text stack
  each have 204 norm tensors totaling 731,136 bytes. The compact metadata is
  saved in `results/gemma3_norm_weight_plan_evidence.json`.
- `gemma3_norm_preload.py` now serializes those BF16 vectors and validates full
  contiguous XRT BO preload for all three local variants: 266,240 bytes for 1B
  and 731,136 bytes each for 4B text and 4B vision text-stack. The compact
  XRT evidence is saved in `results/gemma3_norm_preload_evidence.json`.
- `gemma3_npu_wiring.py` and `gemma3_buffer_binding.py` now match the local
  Transformers `Gemma3DecoderLayer` norm order: `post_attention_layernorm`
  runs before the attention residual, and `pre_feedforward_layernorm` plus
  `post_feedforward_layernorm` wrap the MLP before the MLP residual.
- `gemma3_bo_plan.py` now includes a dedicated `static_norm_weights` BO when
  norm weights are planned, and `gemma3_buffer_binding.py` maps RMSNorm/QK-Norm
  families to that BO while keeping projection families on
  `static_projection_weights`. `gemma3_npu_wiring.py` promotes matching norm
  stages to `weighted_rms_norm/standalone-elf-smoke` model candidates, including
  the local Strix/XRT M=8/N=2560 standalone smoke with 0.999983 output
  correlation for 4B RMSNorm. This is a model-runner argument and launch-intent
  contract; it is not yet a validated
  model-timed norm-kernel launch.
- For the real 1B staged decode layer, composed model launch validation for
  promoted RMSNorm/QK-Norm, RoPE, GeGLU, and residual-add paths is present and
  records `host_fallbacks=[]`; blocker reports consume that evidence and no
  longer list `nonlinear-model-stage-promotion-incomplete` for the real 1B
  plan. Remaining Phase E work is equivalent full-loop/paper-shape coverage,
  equivalent 4B/vision coverage, plus logits/sampling promotion or measured
  timing treatment in end-to-end execution.

### Phase F: end-to-end 1B text reproduction

Goal: reproduce Gemma3 1B prefill and decode tables.

Implementation requirements:

- Add a real 1B text session with tokenizer, real weights, real KV cache,
  full static weight BO preload validation, and per-layer artifact reuse.
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

- `gemma3_reproduction_blockers.py` reports Phase F as `BLOCKED` because
  local 1B artifacts are available but full 1B prefill/decode loop wiring and
  fresh paper-shape hardware reruns are not complete. The prior 1B nonlinear
  model-stage blocker is retired for the staged full-layer evidence because the
  current layer-0 diagnostic records `host_fallbacks=[]`.
  Kernel argument-layout validation is complete for the real 1B 1k/32k-context
  plan: 728 NPU candidate layouts and 2,236 positional arguments validate with
  no missing storage, shape, dtype, direction, or KV-buffer identity blockers.
  First-kernel launch evidence is present for the promoted Gemma3 1B
  pre-attention RMSNorm shape: `gemma3_1b_first_kernel_launch_probe.json`
  records a local Strix/XRT ELF launch at shape 1024x1152 using the validated
  first-stage positional layout (`layer_input`, `static_norm_weights`,
  `prefill_L0_pre_attention_norm`). The worker passes the full contiguous
  `static_norm_weights` payload as argument 1, with the actual layer-0
  `input_layernorm.weight` vector at byte offset 0, allocates/binds the three
  pyxrt BOs directly, and validates output correlation 0.999983. Decode
  RMSNorm-to-`q_proj` substep evidence is also present:
  `gemma3_1b_decode_rmsnorm_qproj_substep_probe.json` records a real layer-0
  RMSNorm launch followed by five real FusedDQP q-projection col-block launches
  with host accumulation. It validates RMSNorm correlation 0.999991, accumulated
  q-projection correlation 1.000000 against the quantized FusedDQP reference,
  and dense original-weight q-projection correlation 0.994609. Full Q/K/V
  staged substep evidence is present in
  `gemma3_1b_decode_rmsnorm_qkv_substep_probe.json`: the same real RMSNorm
  output feeds real q/k/v FusedDQP col-block loops, validating Q/K/V projection
  correlations of 1.000000/1.000000/1.000000 and dense original-weight
  correlations of 0.994609/0.995959/0.995720. Full staged decode layer-0
  evidence is present in `gemma3_1b_decode_full_layer_probe.json`: pre-attention
  RMSNorm, Q/K RMSNorm, post-attention RMSNorm, pre/post-feedforward RMSNorm,
  and q/k/v/o/gate/up/down projection families launch on the NPU through split
  weighted RMSNorm and FusedDQP wrappers. RoPE launches through the Gemma
  half-split wrapper for Q and K at identity position 0, both residual adds
  launch through the Gemma residual-add wrapper, all projection correlations are
  1.000000 against the quantized staged references, Q/K/post-attention/pre-FF/
  post-FF norm correlations are 0.999988/0.999990/0.999985/0.999891/0.999979,
  RoPE Q/K correlations are 1.000000/1.000000, residual correlations are
  0.999956/0.999955, dense original-weight projection correlations are
  0.994609/0.995959/0.995720/0.997553/0.996694/0.996806/0.997571,
  single-token FlowQKV attention correlation is 0.999998, and the final
  layer-output correlation is 0.999953. Layer-1 evidence is present in
  `gemma3_1b_decode_full_layer_L1_probe.json`; it uses the selected-vector
  norm argument for `model.layers.1.input_layernorm.weight` at contiguous norm
  BO offset 10240 and validates RMSNorm correlation 0.999991, all seven
  projection correlations at 1.000000, and final layer-output correlation
  1.000000. No staged layer-0 operation remains as a host-reference step, although the
  model-runner manifest still needs full paper-cell loop wiring and full-context
  KV-cache attention validation. The refreshed layer-0 diagnostic also records
  segmented NPU kernel timing and RAPL after adding staged single-token FlowQKV
  attention: 68 `run.start()/wait2()` launch windows total 0.154274 s for one
  staged layer in reused-ELF mode, corresponding to 6.481976 staged layer
  passes/s and a clearly non-paper-comparable 26-layer kernel-only extrapolation
  of 0.249307 decode TPS versus the paper's 41.1 TPS 1B/1k NPU target. The
  segmented package average is 19.027 W; the pseudo-NPU delta is 4.598 W over a
  14.429 W quiescent package sample. The layer-1
  diagnostic records 0.142578 s, 7.013715 staged layer passes/s, a 0.269758
  kernel-only extrapolated decode TPS, 17.420 W segmented package power, and a
  5.918 W pseudo-NPU package-delta over an 11.502 W quiescent sample. A new
  26-layer staged decode-loop diagnostic is present in
  `gemma3_1b_decode_loop_probe.json`: it measures one post-warmup staged decode
  token across all 26 real Gemma3 1B layers with reusable ELF runners. The probe
  preloads packed projection inputs into 1,456 runner-owned BO sets before
  timing and uses no-allocation static metadata placeholders in the timed path.
  The refreshed loop now launches QK-Norm, RoPE, single-token FlowQKV
  attention, post/pre/post RMSNorm, GeGLU, and both residual adds on the NPU
  for every layer, so `host_fallbacks=[]`. It also runs a 6.902014 s untimed
  all-layer reference pass before the measured loop, so the loop wall timing
  excludes CPU reference/correlation checks. The measured loop wall window is
  5.412830 s, or 0.184746 diagnostic TPS, and includes dynamic BO writes plus
  output sync/readback. The summed NPU launch windows total 3.640356 s across
  1,768 launches, or 0.274698 kernel-only diagnostic TPS. Against the paper's
  41.1 TPS 1B/1k NPU decode target, the post-warmup loop-wall diagnostic is
  99.550% low and the kernel-only diagnostic is 99.332% low. Full-window direct
  RAPL reports 16.790 W package power and a 7.494 W pseudo-NPU package-delta
  from a 9.295 W quiescent sample; the segmented kernel-only pseudo-NPU delta
  is unusable in this run because its quiescent sample was taken while
  preparation was already busy. These split routes are staged correctness and
  diagnostic timing probes because the current full 5-col-block paper module
  over-allocates tile memory for this shape, and because production 1k KV
  attention, logits/sampling, and the production contiguous static-weight BO
  route remain open. A standalone host-batched tiled-stat FlowQKV diagnostic
  validates the 1k x 256 attention shape on hardware. The new
  `gemma3_1b_decode_loop_tiled_stats_probe.json` result integrates that path
  across all 26 layers with `attention_mode=tiled-stats-1k`: it passes with
  `host_fallbacks=[]`, a synthetic repeated current-token KV cache, 16
  host-batched attention launches per layer, host-side softmax-stat reduction,
  a 37.129594 s untimed reference pass, a 35.815148 s measured loop window
  (0.027921 diagnostic TPS), 32.934816 s summed NPU launch time across 2,158
  launches (0.030363 kernel-only diagnostic TPS), 17.577 W full-window package
  power, and a 0.473 W pseudo-NPU package delta. The pseudo-NPU delta is low
  because the quiescent RAPL package sample was close to the timed-window
  package average in this run. This narrows the attention
  work to prefill-produced KV-cache construction and NPU-side stat reduction. The
  real 1B blocker report now narrows the launch
  blocker to the narrower set `prefill-1k-npu-not-wired`,
  `prefill-produced-kv-cache-not-wired`, `npu-attention-reduction-not-wired`,
  `logits-sampling-not-wired`, and
  `production-contiguous-static-weight-bo-not-used-by-fused-dqp-route`;
  repeated full-model loop wiring and timed TTFT/TPS remain blocked. Full paper-shape BO allocation validation is
  complete for 1B, 4B text, and the 4B vision text stack under the
  benchmark-cell KV allocation plan. Full contiguous static-weight BO preload
  validation is complete for 1B, 4B text, and the 4B vision text stack. The
  prior unmeasured-nonlinear fallback blocker is retired by measured CPU-reference
  fallback records, but those records are not NPU promotion evidence. Residual add and RoPE now have standalone NPU evidence and are represented
  as NPU candidates in the wiring manifest; the remaining nonlinear promotion
  blocker is composed model-stage launch integration plus any logits/sampling
  work needed for a paper cell.
- Dependency-light Torch/HF smoke paths now validate local 1B text, 4B text,
  and 4B synthetic-image weights/tokenizer/processor execution without AIR
  imports. Initial local Gemma3 1B 1k CPU/iGPU paper-cell measurements are
  saved under `results/` and use `warmup_iters=1`, `timed_iters=3`, prompt
  length 1024, and 16 decode tokens. The timed region excludes model load,
  tokenizer work, input construction, device placement, compile, BO creation,
  BO preload, xclbin/ELF load, and kernel argument setup. These are local
  Torch/HF baseline measurements on the Strix host, not NPU paper-parity
  claims.

  | Backend | Metric | Local | Paper target | Delta | Classification | Timed power |
  | --- | --- | ---: | ---: | ---: | --- | --- |
  | CPU/HF | Prefill TTFT 1k | 1.430773033 s | 4.06 s | 64.76% faster | `EXPLAINED_DEVIATION` | 45.643 W package/total by direct RAPL |
  | CPU/HF | Decode 1k | 12.400321286 TPS | 41.9 TPS | 70.40% slower | `EXPLAINED_DEVIATION` | 45.727 W package/total by direct RAPL |
  | iGPU/HF ROCm | Prefill TTFT 1k | 0.527177805 s | 0.51 s | 3.37% slower | `PAPER_MATCH` | 37.273 W ROCm SMI GPU rail |
  | iGPU/HF ROCm | Decode 1k | 13.738045814 TPS | 38.0 TPS | 63.85% slower | `EXPLAINED_DEVIATION` | 42.871 W ROCm SMI GPU rail |
  | NPU | Prefill TTFT 1k | blocked | 0.95 s | n/a | `REAL_MODEL_EXECUTION_NOT_IMPLEMENTED` | official pseudo-NPU paper-cell power pending; staged diagnostic payload attached |
  | NPU | Decode 1k | blocked | 41.1 TPS | n/a | `REAL_MODEL_EXECUTION_NOT_IMPLEMENTED` | official pseudo-NPU paper-cell power pending; staged diagnostic payload attached |

  iGPU runs set `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`; without that
  setting, local prefill was materially slower and did not match the iGPU 1k
  paper cell. The decode helper constructs the 1k KV cache before the timed
  window and times only 16 token-by-token decode steps. The matching 1k NPU
  result records remain blocked JSON cells with
  `REAL_MODEL_EXECUTION_NOT_IMPLEMENTED` because full 1B prefill/decode loop
  wiring and fresh paper-shape hardware reruns remain incomplete. They now
  include `npu_staged_diagnostic` payloads pointing at the staged layer-0
  kernel-only timing and segmented RAPL evidence, but
  `local_value` remains null for official paper comparison.
  `gemma3_npu_preflight.py` records
  real projection padding and Q4NX block counts needed for NPU wiring.
  `gemma3_npu_wiring.py` maps each real-shape text layer into prefill/decode
  stage roles, NPU kernel candidates, host fallbacks, local/global attention
  windows, and remaining launch blockers; `gemma3_argument_binding.py` validates
  the corresponding positional argument layouts; GeGLU/MLP activation
  is now represented as a model NPU candidate backed by standalone hardware
  smoke evidence, and the norm/residual ordering follows the local HF Gemma3
  decoder layer.
  `gemma3_weight_plan.py`
  records real text-stack projection static BO byte estimates for future
  preloading.
  `gemma3_bo_plan.py` records the activation, KV-cache, and intermediate BO
  shape/byte contract. `gemma3_norm_preload.py` validates full contiguous XRT
  preload for the RMSNorm/QK-Norm BF16 vectors needed by future nonlinear
  promotion. `gemma3_xrt_runner.py` provides capped real-XRT BO
  allocation/preload smoke coverage; a local 1B smoke run allocated 5,303,808
  bytes and saved `/tmp/gemma3_1b_xrt_bo_smoke.json`. `gemma3_static_preload.py`
  serializes real projection tensors into the Q4NX packed/scale/min byte stream
  and can write selected tensors into XRT BOs. A full local 1B contiguous
  static-preload XRT smoke wrote all 182 planned text projection tensors into
  one static BO, totaling 468,049,920 bytes, and saved
  `results/gemma3_static_preload_evidence.json`. A full local 4B contiguous
  static-preload XRT smoke wrote all 238 planned text projection tensors into
  one static BO, totaling 2,005,401,600 bytes, and updated the same evidence
  ledger; a full local 4B-vision text-stack contiguous static-preload XRT
  smoke wrote all 238 planned text projection tensors into one static BO, totaling
  2,005,401,600 bytes, and updated the same evidence ledger. A full local 1B benchmark-cell paper-shape BO allocation smoke allocated all
  69 planned BOs for prompt 32k/decode 32k, totaling 1,998,196,224 bytes, and
  saved `results/gemma3_bo_allocation_evidence.json`. Full local 4B text and
  4B-vision text-stack benchmark-cell allocation smokes each allocated all 85
  planned BOs for prompt 32k/decode 128k, totaling 7,261,614,080 bytes. The
  largest BO is the 2,005,401,600-byte static projection-weight BO, with a
  separate 731,136-byte `static_norm_weights` BO; the largest K/V BO is a
  268,435,456-byte global-attention layer slice. The same evidence
  ledger preserves earlier monolithic-KV failures where 4B text and vision
  requested 22,708,504,576 bytes and failed at the first 9,126,805,504-byte
  `kv_cache_k` BO after allocating 4,454,893,568 bytes. Those old records are
  strategy-failure evidence, not current BO allocation blockers. `gemma3_real_execution.py` now has
  a Torch/HF warmup/timed-iteration benchmark path for CPU and ROCm iGPU local
  baseline runs. No NPU paper baseline or NPU speedup claim is emitted until
  benchmark-length NPU model execution is implemented.


### Phase G: end-to-end 4B text reproduction

Goal: reproduce Gemma3 4B text prefill and decode tables.

Implementation requirements:

- Add a real 4B text session with the 5-local/1-global layer pattern and the
  paper's full supported decode range.
- Implement 4B prefill TTFT for 1k-32k prompts.
- Implement 4B decode TPS for 1k, 2k, 4k, 8k, 16k, 32k, 64k, and 128k context
  lengths.
- Track out-of-capacity behavior separately from failures.
- Use the implemented benchmark-cell-specific KV allocation plan for 64k/128k
  decode: split K/V by layer; allocate full-context KV only for global layers;
  and clamp local SWA layers to the sliding window where the kernel contract
  allows it. Keep the old monolithic all-layer K/V plan only as a diagnostic
  reproducer.
- Use `l2-gather` full-physical routes where direct shim output is illegal.

Acceptance:

- Correctness passes for all 4B text sequence lengths that fit the
  benchmark-cell-specific memory plan.
- 64k/128k allocation avoids the old single huge KV BO mmap failure and has
  dry-run plus XRT evidence showing per-layer K/V allocation totals.
- NPU, CPU, and iGPU local results are compared against Tables 2 and 4.
- Any 64k/128k deviation includes explicit KV-cache, memory, or schedule data.

Blocked evidence:

- `gemma3_reproduction_blockers.py` reports Phase G as `BLOCKED` because
  local 4B artifacts are available but model-kernel launch, nonlinear
  model-stage promotion, and fresh paper-shape hardware reruns are not complete.
  Full paper-shape BO allocation validation is now
  complete for 4B text under the benchmark-cell KV plan: prompt 32k/decode 128k
  allocates 85 BOs totaling 7,261,614,080 bytes on local Strix/XRT, including
  the dedicated static norm-weight BO. The old
  monolithic KV plan requested 22,708,504,576 bytes, including one
  9,126,805,504-byte `kv_cache_k` BO and one 9,126,805,504-byte `kv_cache_v`
  BO, and stopped at `kv_cache_k` with `xrt-bo-allocation-failed` after
  allocating 4,454,893,568 bytes. Treat that preserved record as a failure of
  the simultaneous all-layer KV BO strategy, not proof that Strix cannot run
  the benchmark. The next implementation path is kernel launch, correctness,
  schedule, and timing validation using the per-layer KV BO contract. Measured
  host fallback
  records account for timing metadata but do not replace nonlinear NPU
  validation.
- `gemma3_npu_wiring.py` emits the 4B text per-layer NPU candidate and host
  fallback plan from local artifacts, including the 5-local/1-global attention
  pattern, but no 64k/128k local paper claim is emitted without kernel-launch
  evidence, correctness, schedule data, and timing data using the revised
  KV-cache allocation contract.

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

- `gemma3_reproduction_blockers.py` reports Phase H as `BLOCKED` because
  local 4B vision artifacts and processor files are available but the model-kernel
  launch, nonlinear model-stage promotion, fresh paper-shape hardware reruns,
  and the vision NPU path are not complete.
  Paper-shape text-stack BO allocation is complete under the same
  benchmark-cell KV plan as 4B text: prompt 32k/decode 128k allocates 85 BOs
  totaling 7,261,614,080 bytes on local Strix/XRT, including the dedicated
  static norm-weight BO. The evidence ledger also
  preserves the older monolithic-KV strategy failure at the first
  9,126,805,504-byte `kv_cache_k` BO after allocating 4,454,893,568 bytes.
  Measured host fallback records account for text
  nonlinear timing metadata but do not validate vision hardware.
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
  and records command, git/environment metadata, artifact inventory, execution
  wiring blockers when real NPU artifacts are present, nonlinear host fallbacks,
  null power fields, and explicit blocked classification.
- `gemma3_xrt_runner.py` can save capped and full-plan BO allocation/preload
  smoke JSON so future paper-result records can distinguish BO allocation
  limits from kernel launch or validation failures. `gemma3-1b`, `gemma3-4b`,
  and `gemma3-4b-vision` now have committed full paper-shape BO allocation
  evidence under the benchmark-cell KV plan. The same ledger preserves
  monolithic all-layer KV allocation-failure evidence for 4B text and vision,
  showing failure at the first 9,126,805,504-byte KV-cache BO on local
  Strix/XRT. Future result records must include `kv_strategy` and distinguish
  allocation strategy failures from kernel launch, correctness, or timing
  failures.
- `gemma3_static_preload.py` can save real Q4NX static-weight preload smoke JSON
  so future model-runner failures can distinguish serialization/preload from
  kernel binding and execution failures. `gemma3-1b`, `gemma3-4b`, and the
  `gemma3-4b-vision` text stack now have committed full contiguous-XRT preload
  evidence with no static-preload blockers.
- `gemma3_buffer_binding.py` records that runtime BO and virtual-intermediate
  binding is planned with no missing BO keys in the self-test fixture.
- `gemma3_argument_binding.py` records deterministic positional argument layouts
  for NPU candidates. The self-test validates 56 fixture candidate layouts with
  172 positional arguments; the real 1B 1k/32k-context text plan validates 728
  candidate layouts with 2,236 positional arguments and zero binding blockers.
- `gemma3_model_runner.py` records launch-order state in result JSON so blocked
  paper cells distinguish BO planning, static preload planning, kernel launch,
  host fallback, runtime buffer-binding state, and argument-layout status.
- `gemma3_launch_probe.py --run-hardware` records first-kernel launch evidence
  for the promoted Gemma3 1B pre-attention RMSNorm stage. The committed Strix
  result launches the 1024x1152 ELF probe, validates the three-argument
  model-runner layout, passes the full contiguous static norm payload as the
  static argument, allocates/binds the three pyxrt BOs directly, and validates
  output correlation 0.999983 against the standalone CPU reference.
- `gemma3_substep_probe.py --run-hardware` records the next decode substep
  evidence: real layer-0 RMSNorm plus real `q_proj` through a five-col-block
  FusedDQP loop with host accumulation. The committed Strix result validates
  RMSNorm correlation 0.999991, q-projection correlation 1.000000 against the
  quantized FusedDQP reference, and dense original-weight correlation 0.994609.
- `gemma3_qkv_substep_probe.py --run-hardware` records full decode Q/K/V
  projection substep evidence using the same split FusedDQP route for q/k/v.
  The committed Strix result validates RMSNorm correlation 0.999991, Q/K/V
  projection correlations of 1.000000/1.000000/1.000000, and dense original-
  weight correlations of 0.994609/0.995959/0.995720.
- `gemma3_full_layer_probe.py --run-hardware --power-sample` records staged
  decode layer evidence using real Gemma3 1B weights. The committed layer-0
  Strix result validates Q/K/post-attention/pre-FF/post-FF norm correlations of
  0.999988/0.999990/0.999985/0.999891/0.999979, all seven NPU projection-family
  correlations at 1.000000 against quantized staged references, dense
  original-weight correlations of
  0.994609/0.995959/0.995720/0.997553/0.996694/0.996806/0.997571,
  single-token FlowQKV attention correlation at 0.999998, GeGLU NPU activation
  correlation at 0.999992, RoPE correlations at 1.000000/1.000000, residual
  correlations at 0.999952/0.999953, and final layer-output correlation
  0.999953. The committed layer-1 result validates the
  same staged projection route after switching RMSNorm to a selected-vector norm
  argument for the layer-1 norm at static norm BO offset 10240. The layer-0 JSON
  records diagnostic compile/load/run work, which is intentionally not a
  TTFT/TPS timing window. It also records a segmented kernel-only timing window
  that excludes compile, ELF load, BO allocation, BO writes/preload, argument
  binding, output sync/readback, and host fallback compute: 68 NPU launches
  total 0.154274 s, or 6.481976 staged layer passes/s. A 26-layer kernel-only
  extrapolation is 0.249307 decode TPS, about 164.9x below the paper's 41.1 TPS
  1B/1k NPU decode target; this is an extrapolation, not a measured full-model
  TPS. Direct RAPL under `sg power` reports 19.027 W segmented package power,
  14.429 W quiescent package power, and a 4.598 W pseudo-NPU package-delta.
  Layer 1 records
  0.142578 s across 57 launches, or 7.013715 staged layer passes/s, with a
  0.269758 kernel-only extrapolated decode TPS and a 5.918 W pseudo-NPU
  package-delta. `gemma3_npu_wiring.py` and
  `gemma3_model_runner.py` consume the first-kernel, q-only, Q/K/V, and
  full-layer evidence to report
  `full-1b-loop-not-wired,paper-shape-hardware-rerun-required` for the real 1B
  plan instead of the stale first-kernel, substep-sequence, full-layer, or nonlinear
  model-stage blocker.
  This is not a repeated model-runner loop, TTFT/TPS timing, pseudo-NPU paper
  power, or a paper cell.
- `gemma3_paper_compare.py --compare` accepts either a single result cell or a
  wrapper with `results`, and can emit Markdown and CSV summaries. The initial
  1B 1k CPU/iGPU measured cells plus NPU blocked cells are bundled in
  `results/gemma3_1b_initial_1k_results.json`; the NPU blocked cells include
  the staged diagnostic payload, `host_fallbacks=[]` for the staged layer-0
  diagnostic, and narrowed 1B blockers while keeping official local TTFT/TPS
  null. The
  generated Markdown and CSV summaries are saved beside the bundle.
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
- Initial 1k CPU/HF and iGPU/HF paper-cell result JSONs request timed-window
  power sampling. CPU cells use direct RAPL sysfs package-energy deltas from
  `/sys/class/powercap/intel-rapl:0/energy_uj`; the refreshed CPU prefill cell
  reports 45.643 W package/total versus the paper CPU-prefill 1B power target
  of 24 W, and the refreshed CPU decode cell reports 45.727 W package/total
  versus the paper CPU-decode 1B target of 29 W. iGPU cells use ROCm SMI socket
  graphics package power; the refreshed iGPU prefill cell reports 37.273 W on
  the GPU rail versus the paper iGPU-prefill 1B GPU rail target of 24 W, and
  the refreshed iGPU decode cell reports 42.871 W on the GPU rail versus the
  paper iGPU-decode 1B GPU rail target of 22 W. iGPU CPU and total rails remain
  `MISSING_POWER_FIELD` because this iteration did not combine ROCm SMI with a
  separate package/CPU rail measurement. `xrt-smi examine -r all` still reports
  `Estimated Power: N/A`, so pseudo-NPU package-delta power remains pending a
  real timed NPU run.
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
| Residual add | `run_residual_add_compile_only.lit` | `residual_add.py` | Attention and MLP residual paths | ELF compile-only and n=1152/tile_n=288 hardware smoke pass; layer-0 staged full-layer launches pass for attention and MLP residuals. |
| RoPE | `run_rope_halfsplit_compile_only.lit` | `rope_halfsplit.py`, `../llama32_1b/kernel_builder/rope_halfsplit.cc` | Q/K rotary embedding | ELF compile-only and rows=4/head_dim=256 hardware smoke pass; layer-0 staged full-layer launch passes for Q/K identity-position RoPE. |

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
| RoPE | `gemma3_dataflow_kernels/rope_halfsplit.py`, `llama32_1b/kernel_builder/rope_halfsplit.cc` | Use the validated half-split LUT wrapper for head_dim=256; keep `rope_sincos` as a semantics reference only because its even/odd layout differs. |
| QK-Norm | weighted RMSNorm pattern | Implement as per-head Q/K normalization with CPU reference first; add `gemma3_dataflow_kernels/qk_norm.py/.cc` if existing RMSNorm cannot express the layout. |
| MLP activation | `programming_examples/gelu`, `programming_examples/ffn_swiglu`, Llama `silu_and_mul` | Confirm Gemma activation from config/paper evidence before choosing GELU/GeGLU/SwiGLU; do not copy Llama SwiGLU by assumption. |
| Elementwise multiply | `programming_examples/ffn_swiglu`, Llama `silu_and_mul` | Reuse if activation semantics match; otherwise add a standalone Gemma elementwise kernel. |
| Residual add | `gemma3_dataflow_kernels/residual_add.py` | Use the validated standalone BF16 vector add wrapper until a fused model-stage route is measured. |
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
allocate_kv_cache(benchmark_cell_per_layer_strategy)

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

Benchmark-cell allocation policy:

- Build the KV allocation plan from the exact prompt length and decode context
  being benchmarked; do not use one global maximum allocation for all cells.
- Preallocate projection weights, norm weights, and reusable work buffers once
  per benchmark cell.
- Split K/V storage by layer and by page/chunk where needed to avoid single
  multi-GB BO mmap failures.
- Local SWA layers should allocate/read only the legal sliding window when the
  kernel contract does not require older tokens.
- Global layers should allocate/read the full current context for that benchmark
  cell.
- Kernel argument binding must name the exact layer/page KV slices consumed or
  produced by each stage.

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
- Use `gemma3_norm_weight_plan.py` and `gemma3_norm_preload.py` to bind the
  preloaded norm-weight BOs into the validated weighted RMSNorm wrapper for
  Gemma RMSNorm and QK-Norm in the model loop.
- Keep the promoted GeGLU/MLP activation path as a model NPU candidate and
  validate launch ABI against the generated argument layout before counting it in timed model results.
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
- Preserve Gemma3's post-attention, pre-feedforward, and post-feedforward norm
  placement from the HF decoder layer.
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
test -f programming_examples/gemma3_dataflow_kernels/rope_halfsplit.py
test -d programming_examples/gelu
test -d programming_examples/ffn_swiglu
```

Existing nonlinear candidates to evaluate before adding Gemma-specific kernels:

```bash
make -C programming_examples/weighted_rms_norm run OUTPUT_FORMAT=elf
python3 programming_examples/gemma3_dataflow_kernels/rope_halfsplit.py --compile-mode compile-only --output-format elf
make -C programming_examples/rope_sincos run AIE_TARGET=aie2p OUTPUT_FORMAT=elf
make -C programming_examples/gelu run OUTPUT_FORMAT=elf
make -C programming_examples/ffn_swiglu/decode run OUTPUT_FORMAT=elf
```

These standalone targets are reuse candidates; `rope_halfsplit.py` and
`residual_add.py` are Gemma-specific standalone candidates with focused
compile-only lit coverage. Add focused compile-only lit coverage for any new
nonlinear before the Gemma model loop depends on it.

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
