# Gemma3 Paper Result Artifacts

This directory is reserved for small Gemma3 paper-comparison result artifacts:
JSON result cells, Markdown summaries, and CSV summaries. Large model weights,
tokenizer caches, xclbins, ELFs, trace dumps, and debug IR should stay out of
source control unless they are reviewed as compact fixtures.

## Initial 1k CPU/NPU Paper-Cell Evidence

- `gemma3_1b_cpu_prefill_1k_initial.json`: real local Gemma3 1B CPU/HF 1k
  prefill TTFT measurement. Local runtime-only TTFT is 1.727778663 s versus the
  paper CPU target of 4.06 s, classified as `EXPLAINED_DEVIATION`.
- `gemma3_1b_cpu_decode_1k_initial.json`: real local Gemma3 1B CPU/HF 1k
  decode-only TPS measurement. The helper builds the KV cache before the timed
  section and times 16 token steps. Local TPS is 14.280128371 versus the paper
  CPU target of 41.9, classified as `EXPLAINED_DEVIATION`.
- `gemma3_1b_npu_prefill_1k_blocked_initial.json` and
  `gemma3_1b_npu_decode_1k_blocked_initial.json`: matching NPU paper cells that
  record `REAL_MODEL_EXECUTION_NOT_IMPLEMENTED` rather than timing data because
  model-kernel launch, kernel argument binding, nonlinear model-stage
  promotion, and paper-shape hardware reruns remain incomplete.
- Power sampling was requested for these cells, but all CPU/GPU/NPU/total rails
  remain `MISSING_POWER_FIELD`: XRT reports `Estimated Power: N/A`; ROCm SMI can sample future iGPU
  timed-window power; CPU package watts and pseudo-NPU package-delta watts need
  `turbostat_pkgwatt` or working raw `turbostat` PkgWatt support. Current raw
  `turbostat` is blocked by missing `linux-tools-6.14.0-1020-oem`.


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
  kernel argument binding, launch, correctness, timing, or paper-parity
  evidence.
