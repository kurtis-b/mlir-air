# Gemma3 Paper Result Artifacts

This directory is reserved for small Gemma3 paper-comparison result artifacts:
JSON result cells, Markdown summaries, and CSV summaries. Large model weights,
tokenizer caches, xclbins, ELFs, trace dumps, and debug IR should stay out of
source control unless they are reviewed as compact fixtures.

## Initial 1k CPU/iGPU/NPU Paper-Cell Evidence

The initial 1B 1k baseline cells use prompt length 1024, one warmup iteration,
three timed iterations, and 16 decode tokens. The timed region excludes model
load, tokenizer work, input construction, device placement, compile, BO
creation/preload, xclbin/ELF load, and kernel argument setup.

| File | Backend | Metric | Local | Paper | Classification | Power |
| --- | --- | --- | ---: | ---: | --- | --- |
| `gemma3_1b_cpu_prefill_1k_initial.json` | CPU/HF | Prefill TTFT | 1.430773033 s | 4.06 s | `EXPLAINED_DEVIATION` | 45.643 W RAPL package/total |
| `gemma3_1b_cpu_decode_1k_initial.json` | CPU/HF | Decode TPS | 12.400321286 | 41.9 | `EXPLAINED_DEVIATION` | 45.727 W RAPL package/total |
| `gemma3_1b_igpu_prefill_1k_initial.json` | iGPU/HF ROCm | Prefill TTFT | 0.527177805 s | 0.51 s | `PAPER_MATCH` | 37.273 W ROCm SMI GPU rail |
| `gemma3_1b_igpu_decode_1k_initial.json` | iGPU/HF ROCm | Decode TPS | 13.738045814 | 38.0 | `EXPLAINED_DEVIATION` | 42.871 W ROCm SMI GPU rail |
| `gemma3_1b_npu_prefill_1k_blocked_initial.json` | NPU | Prefill TTFT | blocked | 0.95 s | `REAL_MODEL_EXECUTION_NOT_IMPLEMENTED` | pseudo-NPU RAPL delta pending |
| `gemma3_1b_npu_decode_1k_blocked_initial.json` | NPU | Decode TPS | blocked | 41.1 | `REAL_MODEL_EXECUTION_NOT_IMPLEMENTED` | pseudo-NPU RAPL delta pending |

The iGPU cells set `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` and use ROCm SMI
for timed-window GPU rail sampling. CPU cells use direct RAPL sysfs package
energy through the `power` group. iGPU CPU/total rails remain
`MISSING_POWER_FIELD`; NPU timing and pseudo-NPU power remain blocked by full
Q/K/V substep wiring, nonlinear model-stage promotion, and fresh paper-shape
hardware reruns. Kernel argument-layout validation is complete for the real 1B
1k/32k context model-runner plan: 572 NPU candidate layouts, 1,924 positional
arguments, and zero binding blockers.

`gemma3_1b_initial_1k_results.json` bundles these six cells, and
`gemma3_1b_initial_1k_summary.md` / `gemma3_1b_initial_1k_summary.csv` contain
the generated paper-target comparison summary.


## First Kernel Launch Probe Evidence

- `gemma3_1b_first_kernel_launch_probe.json`: compact Strix/XRT evidence that
  the promoted Gemma3 1B pre-attention RMSNorm shape (`1024x1152`) launches as
  an ELF on the NPU with the validated first-stage positional layout
  (`layer_input`, `static_norm_weights`, `prefill_L0_pre_attention_norm`). The
  worker passes the full contiguous `static_norm_weights` payload as argument 1,
  allocates/binds the three pyxrt BOs directly, and uses the actual layer-0
  `input_layernorm.weight` vector at byte offset 0. It validates with output
  correlation 0.999983 against the standalone CPU reference. This is first-stage
  launch evidence only; it is not a substep sequence, full model-runner launch,
  TTFT/TPS timing, pseudo-NPU power, or paper-parity result.


## Decode Substep Probe Evidence

- `gemma3_1b_decode_rmsnorm_qproj_substep_probe.json`: compact Strix/XRT
  evidence for a real Gemma3 1B decode substep. It launches layer-0 RMSNorm
  with the full contiguous `static_norm_weights` payload, then launches five
  FusedDQP q-projection col blocks with real `q_proj.weight` and accumulates the
  partial outputs on the host. It validates RMSNorm correlation 0.999991,
  accumulated q-projection correlation 1.000000 against the quantized FusedDQP
  reference, and dense original-weight correlation 0.994609. This is staged
  correctness evidence only; it is not full QKV, a full layer, TTFT/TPS timing,
  pseudo-NPU power, or paper-parity evidence.


## Static Preload Evidence

- `gemma3_static_preload_evidence.json`: compact Strix/XRT evidence that
  `gemma3-1b`, `gemma3-4b`, and the `gemma3-4b-vision` text stack
  serialized and wrote all planned text projection tensors into one contiguous
  XRT BO per model variant. This is static-weight preload evidence only; it is not a model
  kernel launch, correctness, timing, or paper-parity result.


## BO Allocation Evidence

- `gemma3_bo_allocation_evidence.json`: compact Strix/XRT evidence for full
  paper-shape BO allocation. The current benchmark-cell entries validate
  `gemma3-1b` at 32k prompt/32k decode context with 69 BOs totaling
  1,998,196,224 bytes, and `gemma3-4b` plus `gemma3-4b-vision` at 32k
  prompt/128k decode context with 85 BOs totaling 7,261,614,080 bytes. The
  ledger also preserves earlier monolithic-KV failures where 4B text and vision
  hit the local XRT host-memory allocation limit at the first
  9,126,805,504-byte KV-cache BO after allocating 4,454,893,568 bytes. This is
  allocation evidence only; it is not a kernel launch, correctness, timing, or
  paper-parity result.


## Norm Weight Plan Evidence

- `gemma3_norm_weight_plan_evidence.json`: compact safetensor-metadata evidence
  for the BF16 norm vectors needed by RMSNorm and QK-Norm promotion. It records
  tensor counts and byte totals only; it is not XRT preload, kernel launch,
  correctness, timing, or paper-parity evidence.


## Norm Preload Evidence

- `gemma3_norm_preload_evidence.json`: compact Strix/XRT evidence that the
  RMSNorm/QK-Norm BF16 vectors for `gemma3-1b`, `gemma3-4b`, and the
  `gemma3-4b-vision` text stack were serialized and written into one contiguous
  XRT BO per variant. This is norm-weight preload evidence only; it is not
  kernel launch, correctness, timing, or paper-parity evidence.
