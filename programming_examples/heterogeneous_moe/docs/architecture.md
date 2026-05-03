# Heterogeneous MoE Architecture Guide

This document describes the branch-local architecture of `programming_examples/heterogeneous_moe`: how data moves through the runtime, how backends are selected, where artifacts are generated, and how to read the results.

## Runtime Dataflow

The runtime has four logical stages:

1. `router`: computes two-expert logits from the input tokens.
2. CPU top-k selection: converts logits to `top1` or `top2` route weights.
3. `expert0` and `expert1`: run MLP expert work in parallel through a two-worker thread pool.
4. `aggregation`: packs expert outputs and combines them with the route weights.

The important v1 constraint is that top-k selection is CPU-side. The configurable router backend computes logits, but logits are transferred to CPU before route weights and routed expert inputs are produced.

For a single chunk, the flow is:

```text
inputs
  -> router backend
  -> logits on CPU
  -> CPU top-k and routed inputs
  -> expert0 backend and expert1 backend in parallel
  -> CPU pack of expert outputs
  -> aggregation backend
  -> output on CPU
  -> untimed validation and reporting
```

For model-preset workloads where `routed_tokens` is larger than `model.batch_tokens`, the runtime executes fixed-size chunks and merges outputs, traces, and validation metrics. The `workload.kernel_chunk_tokens` field records the fixed kernel chunk size.

## Backend Execution Model

CPU stages use the NumPy reference math in `reference.py`.

GPU stages use AIR-to-ROCDL compiled shared libraries loaded through Python `ctypes`. Router and aggregation compile as single kernels. Expert execution uses split `expert_hidden` and `expert_output` kernels with encoded bf16 buffers and cached weight descriptors.

NPU stages use `air.backend.xrt` and XRT compile artifacts. Router and aggregation can compile as direct NPU kernels. Expert execution first tries full split `expert_hidden` plus `expert_output` kernels. If that compile path fails, the compiler falls back to tiled split expert kernels and host-side accumulation across tiles.

The harness stores bf16 device buffers as raw `uint16` bit patterns at GPU/NPU boundaries. Use the helpers in `numerics.py` for encoding, decoding, and quantization behavior.

## Transfer Semantics

The transfer manager records edge events between stage backends. Its model is intentionally host-visible:

- `host`: always performs a NumPy host-staged copy.
- `peer`: permits same-backend and CPU-facing edges only; unsupported direct edges raise an error.
- `auto`: aliases or contiguous-copies supported edges in the NumPy host-array model and host-stages unsupported edges.

Supported peer edges are `cpu<->cpu`, `cpu<->gpu`, `gpu<->cpu`, `cpu<->npu`, `npu<->cpu`, `gpu<->gpu`, and `npu<->npu`. Direct `gpu<->npu` peer transfer is not implemented. Result files therefore report `transfer_summary.model = numpy_host_array_transfer_model` and `device_resident_buffers = false`.

## LLM-Linear Roadmap Harness

`llm_linear/` is the newer GEMM/GEMV benchmark used by the Ryzen heterogeneous
execution roadmap. It keeps the MoE harness archived as a reference and models:

- prefill GEMM: `X[M,K] @ Wp[K,H] -> P[M,H]`
- decode GEMV: `P[M-1,:] @ Wd[H,N] -> Y[N]`
- CPU, GPU, NPU, host-staged mixed, and hardware-gated direct mixed placements

The direct mixed cases use `transfer_mode=direct` and the
`DeviceResidentTensor` result contract. They must not claim success unless a
native bridge records an audited GPU/NPU edge with zero NumPy host
materializations. Result summaries report the direct contract as
`no_host_copies`, the selected mechanism, its direct class, the structured probe
report, `zero_host_copy=true`, and `device_resident_buffers` as a capability
flag rather than the definition of direct. The checked-in bridge builds a C++
XRT/HIP path around HIP-owned VMem allocations exported as POSIX fds and
imported into XRT as `xrt::bo` views; if the platform probe cannot validate an
audited zero-host-copy path, direct cases fail before host fallback.

Milestone 2 is accepted as of May 3, 2026 with:
`source /opt/xilinx/xrt/setup.sh && ../../sandbox/bin/python run_llm_linear_milestone2.py`.
The accepted output root is `llm_linear/artifacts/benchmarks/milestone2_e2e`.
That hardware gate covers `medium_m8_k512_h512_n256` in both direct directions
plus matching host-mixed baselines.

Milestone 3 is accepted as of May 3, 2026 with:
`source /opt/xilinx/xrt/setup.sh && ../../sandbox/bin/python run_llm_linear_milestone3.py`.
The accepted output root is
`llm_linear/artifacts/benchmarks/milestone3_int4_hw`. That hardware gate covers
the full `tiny_ci`, `medium`, and `llm_like` LLM-linear suites for `gpu_only`,
`npu_only`, host-mixed GPU/NPU splits, and direct mixed GPU/NPU splits with
signed int4 decode weights. Accelerator fused decode is fail-closed to signed
`int4`, `quant_axis=0`, `H % block_size == 0`, and `N % 8 == 0`. CPU decode
continues to support signed `int4` and unsigned `uint4` packed storage.

Quantized decode result JSON/CSV/report fields include the requested storage,
block size, quant axis, kernel key, hardware-fused eligibility, packed
shape/bytes, scale shape/bytes, zero-point bytes when present, and the NPU
decode tile width. CPU fused-dequant details still separate dequant time,
linear time, packed bytes read, scale bytes read, and zero-point bytes read.

## Manifest Schema

`default_manifest.json` is the portable base manifest. Its top-level sections are:

| Section | Meaning |
| --- | --- |
| `model` | `batch_tokens`, `hidden_size`, `ffn_size`, and `dtype`. The default is `4x16x32 bf16`. |
| `paths` | Checked-in AIR source root, generated AIR source root, and artifact root. |
| `runtime` | Default `router_mode`, `transfer_mode`, and `stage_backends`. |
| `compiler` | Backend target names such as `npu_device` and `gpu_arch`. |
| `benchmark` | Default warmup and timed iteration counts. |
| `inputs`, `weights` | Deterministic seeds and scale factors. |
| `workload` | Routing profile and optional suite metadata. |
| `artifacts` | Compiled artifact paths, normally populated only in sidecar manifests. |

Keep compiled paths out of `default_manifest.json`. `compile_kernels.py` and workload suites should write sidecar manifests under ignored artifact roots.

## Benchmark Matrix Schema

`default_benchmark_matrix.json` contains a `cases` array. Each case provides:

- `name`
- `router_mode`
- `router_backend`
- `expert0_backend`
- `expert1_backend`
- `aggregation_backend`
- `transfer_mode`

`run_matrix.py --case-filter` selects exact case names. Any case containing an NPU backend is skipped unless `--allow-npu` is set.

## Workload Suites

`run_workload_suite.py` builds generated manifests from `default_manifest.json` and `default_benchmark_matrix.json`.

`shape_sweep` uses five shape tiers:

| Name | Shape | Scale |
| --- | --- | --- |
| `small` | `4x16x32` | `0.5` |
| `smallplus` | `4x24x48` | `0.375` |
| `medium` | `8x32x64` | `0.25` |
| `midlarge` | `8x40x80` | `0.1875` |
| `large` | `8x48x96` | `0.125` |

`routing_sweep` runs the default shape across `balanced`, `expert0_hot`, `expert1_hot`, and `alternating` routing profiles. It uses a reduced case list focused on CPU, GPU, NPU, and mixed expert placements.

`model_presets` records model-anchored metadata for:

- `LiquidAI/LFM2-8B-A1B`
- `ibm-granite/granite-4.0-h-tiny`
- `google/gemma-4-26B-A4B-it`
- `Qwen/Qwen3.6-35B-A3B`

Preset fields include `model_id`, `model_class`, `batch_tokens`, `hidden_size`, `ffn_size`, `scale`, `weight_storage`, `compute_dtype`, `num_experts`, `active_experts`, and optional `shared_expert_ffn_size`.

Context lengths are converted to routed-token workloads with:

```text
ceil(context_length * active_experts / num_experts)
```

Those logical routed tokens are executed in chunks of `model.batch_tokens`.

## Generated Artifacts

Checked-in AIR goldens live under `air/`. Generated AIR and compiled outputs live under manifest-controlled artifact roots. Common locations are:

- `artifacts/generated_air/`
- `artifacts/compiled_manifest.json`
- `artifacts/gpu/*.so`
- `artifacts/gpu/*.mlir`
- `artifacts/gpu/*.ll`
- `artifacts/npu/*.xclbin`
- `artifacts/npu/*.insts.bin`
- `artifacts/workloads/<suite>/<workload>/air_sources/`
- `artifacts/workloads/<suite>/<workload>/compiled/`

GPU compilation runs `air-opt -air-to-rocdl -air-gpu-outlining -air-gpu-host-staging`, lowers with the LLVM tools under `LLVM_INSTALL_DIR`, translates to LLVM IR, strips generated module destructors, and links a shared library with MLIR ROCm runtime libraries and HIP.

## Result Schema

Benchmark results use `schema_version = edge-study-v1`. Important fields include:

| Field | Meaning |
| --- | --- |
| `metadata` | Command line, manifest hash, and run metadata. |
| `case_name`, `router_mode`, `stage_backends`, `transfer_mode` | Logical placement and routing configuration. |
| `workload` | Shape, routing profile, scales, context length, routed tokens, and chunk count. |
| `measurement` | Requested mode, effective warmup, validation/setup timing flags, and cold/warm runs. |
| `latency_ms`, `latencies_ms` | Timed latency summaries. These exclude setup and validation. |
| `timing_breakdown_ms`, `phase_timings_ms` | Host-side timing breakdown, including setup, input generation, warmup, timed loop, and validation. |
| `correctness`, `stage_metrics`, `torch_validation` | Final output and optional PyTorch validation status. |
| `trace_summary`, `device_events` | Host-visible stage and event summaries from the untimed validation run. |
| `transfer_events`, `transfer_summary` | Transfer accounting in the NumPy host-array model. |
| `quantized_decode` | Decode quantization metadata, hardware-fused flag, packed/scales byte counts, and CPU fused-dequant timing detail when available. |
| `npu_development` | NPU buffer layout, encoded dtype summaries, artifact/source paths, and `executed`. |
| `execution_truth` | Booleans for NPU execution and timing exclusions. |
| `limitations` | Reader-facing caveats for the run. |

The most important interpretation details are:

- `latency_ms` measures `runtime.run(..., validate=False, capture_details=False)`.
- Validation runs once after timing and is not included in latency.
- Compile/load setup is not included in latency.
- `npu_development.executed` and `execution_truth.npu_executed` are the truth flags for NPU execution.
- `quantized_decode.hardware_fused` is the truth flag for accelerator int4 decode; CPU-only int4/uint4 runs can still have `quantized_decode.enabled=true`.
- Transfer bytes and elapsed times are host-array model events, not device DMA measurements.

## Output Layouts

`bench.py` writes only the files requested by output flags.

`run_matrix.py` writes:

- `<output-dir>/<case>/results.json`
- `<output-dir>/<case>/stage_metrics.json`
- `<output-dir>/<case>/trace_summary.json`
- `<output-dir>/<case>/transfer_summary.json`
- `<output-dir>/<case>/device_events.json`
- `<output-dir>/<case>/npu_development.json`
- `<output-dir>/traces/<case>.json`
- `<output-dir>/summary.json`
- `<output-dir>/summary.csv`

`run_workload_suite.py` writes:

- `<output-dir>/<suite>/<workload>/compiled_manifest.json`
- `<output-dir>/<suite>/<workload>/summary.json`
- `<output-dir>/<suite>/<workload>/cases/<case>.json`
- `<output-dir>/summary.json`
- `<output-dir>/summary.csv`
- `<output-dir>/report.md`

`edge_study.py` writes:

- `<output-dir>/suite/`
- `<output-dir>/edge_efficiency_summary.json`
- `<output-dir>/edge_efficiency_report.md`

## Correctness Gates

`--require-correctness` fails a command when final output validation is outside dtype-aware tolerances. Intermediate stage metrics are still emitted for debugging, but final output correctness is the pass/fail gate for benchmark runs.

`--require-torch` fails if PyTorch validation is unavailable or fails. Use it when validating numerics with the sandbox dependency installed; omit it for infrastructure-only smoke runs.

## Known Limitations

- Router top-k selection is CPU-side.
- Direct iGPU-to-NPU peer transfer is hardware-gated by the LLM-linear native
  bridge probe.
- Transfer accounting is host-visible and does not prove device-resident overlap.
- LLM-linear direct cases are fail-closed unless the native GPU/NPU bridge probe
  succeeds and hardware verification records an audited direct edge.
- The harness models MoE routing and expert compute, not a full transformer.
- Bias terms are omitted to keep the current runtime ABI small.
- Quantized model presets execute bf16 math after dequantized weight loading.
- LLM-linear int4/uint4 support covers decode GEMV, not prefill GEMM.
- Accelerator fused int4 decode supports only signed `int4`, `quant_axis=0`,
  `H % block_size == 0`, and `N % 8 == 0`.
- Shared expert metadata is recorded, but shared-expert execution is not modeled.
