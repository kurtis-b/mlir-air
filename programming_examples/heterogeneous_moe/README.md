# Heterogeneous MoE

This example implements a fixed-shape, two-expert MoE runtime on top of the MLIR-AIR repo. It is designed as a v1 research harness rather than a polished product surface:

- Router math, expert 0, expert 1, and aggregation are independently assignable to `cpu`, `npu`, or `gpu`.
- `top1` and `top2` routing are supported.
- Synthetic workload profiles currently include `balanced`, `expert0_hot`, `expert1_hot`, and `alternating`.
- Top-k selection remains on CPU in v1. The configurable router stage computes logits.
- Transfer mode can be `host`, `peer`, or `auto`.

The default kernel shape is intentionally small so the generated kernels stay explicit and easy to inspect:

- Tokens: `4`
- Hidden size: `16`
- FFN size: `32`
- Datatype: `bf16`

## Status

- CPU path: verified for both `top1` and `top2`; the benchmark reports `max_abs_error = 0.0` against the NumPy reference.
- NPU path: verified on `npu2` for `top1`, `top2`, and mixed NPU+GPU runs under the default benchmark configuration. The default manifest now uses half-scale randomized inputs and weights so dense `bf16` accumulation stays in a realistic validation regime.
- GPU path: verified for both `top1` and `top2`; the benchmark reports `max_abs_error = 0.0` on `gfx1150` after lowering the checked-in AIR kernels through `air-to-rocdl`.
- Mixed NPU+GPU path: verified with CPU router and aggregation plus split expert execution across NPU and GPU.

## Claims

- What this example proves today: the same checked-in AIR kernels can be compiled for both Ryzen XDNA NPU and Ryzen iGPU backends, and the runtime can route stages independently across CPU, GPU, and NPU backends.
- What this example does not prove yet: cost-model scheduling, NPU performance leadership, or larger-model scaling beyond this fixed two-expert harness.

## Files

- `default_manifest.json`: default runtime and compiler configuration.
- `kernels.py`: emits the canonical AIR sources under `air/` that feed both the NPU and GPU compile flows.
- `compile_kernels.py`: compiles the AIR sources for the requested backends, then writes a sidecar manifest with artifact paths.
- `bench.py`: runs one benchmark configuration and can write structured JSON, CSV, trace summaries, stage metrics, and NPU development reports.
- `smoke_tests.py`: runs first-class golden, CPU, GPU, mixed GPU, and hardware-gated NPU smoke lanes with final-output correctness gating.
- `run_matrix.py`: runs a benchmark matrix and writes per-case JSON, aggregate CSV/JSON, and traces.
- `run_workload_suite.py`: runs expanded shape, routing-profile, and model-anchored sweeps, compiling per-shape artifacts as needed and writing suite summaries.
- `edge_study.py`: runs a canonical edge-efficiency suite and summarizes whether mixed CPU/iGPU/NPU placements beat single-backend baselines after transfer and launch overhead.
- `report.py`: renders a markdown report from a matrix run.
- `setup_sandbox.sh`: creates a repo-local `sandbox` venv and installs NumPy plus CPU PyTorch for validation.
- `reference.py`: NumPy reference implementation and optional PyTorch validation.
- `air/`: tiny checked-in AIR goldens for default shapes; generated AIR variants are ignored.
- `artifacts/`: ignored runtime/compiler output root for compiled manifests, libraries, benchmark results, reports, and traces.

## Sandbox Setup

```bash
cd <mlir-air-repo>/programming_examples/heterogeneous_moe
./setup_sandbox.sh
source ../../sandbox/bin/activate
```

## CPU Smoke Test

```bash
cd <mlir-air-repo>/programming_examples/heterogeneous_moe
python3 bench.py --iterations 1 --warmup 0 \
  --router-backend cpu \
  --expert0-backend cpu \
  --expert1-backend cpu \
  --aggregation-backend cpu \
  --router-mode top2 \
  --require-correctness \
  --require-torch
```

CI-safe MoE smoke entrypoint:

```bash
cd <mlir-air-repo>/programming_examples/heterogeneous_moe
python3 smoke_tests.py --lane ci
```

## Compile Kernels

The checked-in compile inputs are the tiny default goldens under `air/`. Those AIR kernels drive both the NPU flow and the iGPU `air-to-rocdl` flow. Generated AIR variants and compiled outputs belong under ignored artifact/build roots such as `artifacts/`, `air_gpu/`, `air_probe*/`, and `air_project/`.

```bash
cd <mlir-air-repo>/programming_examples/heterogeneous_moe
source ../../sandbox/bin/activate
export LLVM_INSTALL_DIR=<path-to-llvm-amdgpu-install>
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
python3 compile_kernels.py --backends gpu
```

That command writes `artifacts/compiled_manifest.json` and leaves the checked-in `default_manifest.json` portable.

## Example Configurations

CPU-only:

```bash
python3 bench.py --router-backend cpu --expert0-backend cpu --expert1-backend cpu --aggregation-backend cpu --require-correctness --require-torch
```

GPU-only:

```bash
export LLVM_INSTALL_DIR=<path-to-llvm-amdgpu-install>
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
python3 bench.py --manifest artifacts/compiled_manifest.json \
  --router-backend gpu \
  --expert0-backend gpu \
  --expert1-backend gpu \
  --aggregation-backend gpu \
  --router-mode top2 \
  --require-correctness \
  --require-torch
```

CPU/GPU benchmark matrix plus markdown report:

```bash
export LLVM_INSTALL_DIR=<path-to-llvm-amdgpu-install>
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
python3 run_matrix.py --manifest artifacts/compiled_manifest.json --require-correctness --require-torch
python3 report.py --summary artifacts/benchmarks/latest/summary.json
```

GPU and mixed CPU/GPU smoke lanes:

```bash
export LLVM_INSTALL_DIR=<path-to-llvm-amdgpu-install>
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
python3 smoke_tests.py --lane gpu-all
```

Expanded workload suites:

```bash
export LLVM_INSTALL_DIR=<path-to-llvm-amdgpu-install>
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export AIRCC_PATH=<path-to-aircc>
export AIR_OPT_PATH=<path-to-air-opt>
python3 run_workload_suite.py --allow-npu --require-correctness --iterations 1 --warmup 1 \
  --output-dir artifacts/benchmarks/workload_suites/latest
```

Canonical edge-efficiency study:

```bash
export LLVM_INSTALL_DIR=<path-to-llvm-amdgpu-install>
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export AIRCC_PATH=<path-to-aircc>
export AIR_OPT_PATH=<path-to-air-opt>
python3 edge_study.py --profile routing --measurement-mode both --allow-npu \
  --require-correctness \
  --iterations 3 --warmup 1 \
  --output-dir artifacts/benchmarks/edge_study/latest
```

That command writes the raw workload-suite outputs under `suite/`, plus `edge_efficiency_summary.json` and `edge_efficiency_report.md`. Use `--profile full` for the full shape, routing, and model-preset sweep.

Modern MoE model presets:

```bash
export LLVM_INSTALL_DIR=<path-to-llvm-amdgpu-install>
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export AIRCC_PATH=<path-to-aircc>
export AIR_OPT_PATH=<path-to-air-opt>
python3 run_workload_suite.py --suite model_presets --allow-npu --require-torch \
  --require-correctness \
  --iterations 1 --warmup 0 \
  --output-dir artifacts/benchmarks/workload_suites/model_presets_latest
```

Context-length smoke run for one preset:

```bash
export LLVM_INSTALL_DIR=<path-to-llvm-amdgpu-install>
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export AIR_OPT_PATH=<path-to-air-opt>
python3 run_workload_suite.py --suite model_presets \
  --workload-filter qwen36_35b_a3b_qbf16 \
  --case-filter cpu_top2 gpu_top2 \
  --require-correctness --require-torch --iterations 1 --warmup 0 \
  --output-dir artifacts/benchmarks/workload_suites/qwen_context_sweep
```

Stable shape sweep with the expanded five-tier ladder:

```bash
export LLVM_INSTALL_DIR=<path-to-llvm-amdgpu-install>
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export AIRCC_PATH=<path-to-aircc>
export AIR_OPT_PATH=<path-to-air-opt>
python3 run_workload_suite.py --suite shape_sweep --allow-npu --require-torch \
  --require-correctness \
  --iterations 3 --warmup 1 \
  --output-dir artifacts/benchmarks/workload_suites/apr21_shape_sweep_stable
```

NPU development report without NPU execution:

```bash
python3 bench.py \
  --router-backend cpu \
  --expert0-backend cpu \
  --expert1-backend cpu \
  --aggregation-backend cpu \
  --router-mode top2 \
  --npu-dev-report-out artifacts/benchmarks/latest/npu_development.json
```

## Notes

- The bias terms are omitted in v1 to keep the AIR kernel interface within the current Python XRT backend argument limit.
- `peer` transfer mode currently models copy elision on CPU-facing edges and same-backend edges in the NumPy host-array runtime. Direct `npu <-> gpu` peer transfer is intentionally reported as unsupported in `peer` mode and falls back to host staging only in `auto`.
- Final-output validation uses dtype-aware tolerances in the recorded stage metrics and can be required with `--require-correctness`.
- PyTorch validation compares actual stage outputs against a PyTorch eager reference and can be required with `--require-torch`.
- The example defaults to `bf16` because that matches the Ryzen NPU data path. The harness does not require `ml_dtypes`; it marshals `bf16` buffers as raw `uint16` bit patterns when talking to the device runtime.
- The default manifest uses `inputs.scale = 0.5` and `weights.scale = 0.5`. Unit-scale random tensors exaggerated dense `bf16` accumulation drift on the NPU and were not a good correctness target for this small demo.
- `run_workload_suite.py` now applies a shape-aware validation scale ladder for the expanded shape sweep: `4x16x32 -> 0.5`, `4x24x48 -> 0.375`, `8x32x64 -> 0.25`, `8x40x80 -> 0.1875`, and `8x48x96 -> 0.125`.
- The model-preset sweep uses four model-anchored expert shapes with a decode-style routed-token bucket of `4`: `LiquidAI/LFM2-8B-A1B`, `ibm-granite/granite-4.0-h-tiny`, `google/gemma-4-26B-A4B-it`, and `Qwen/Qwen3.6-35B-A3B`. The Gemma and Qwen presets are labeled `quantized` in the metadata, but the kernels still execute `bf16` math after dequantized weight loading, matching the intended llama.cpp-style comparison point.
- The model-preset sweep now also accepts representative context lengths `64`, `128`, `256`, `512`, `1024`, and `2048`. Because this harness measures MoE expert compute rather than full attention, each context length is converted to an average per-expert routed-token workload `ceil(context_length * active_experts / num_experts)`, then executed in fixed kernel chunks of `model.batch_tokens`.
- The model-preset suites now run on CPU, GPU, and NPU. The NPU expert path uses cached tiled `expert_hidden` plus `expert_output` AIR kernels and host-side accumulation, so model-sized experts no longer depend on the old monolithic `aircc` lowering path.
- The current expert fast path is a split AIR implementation: `expert_hidden` and `expert_output` are compiled separately. On the GPU path those split kernels now expose tile-level parallelism through AIR herd dimensions, and on the NPU path the runtime first tries the same full split kernels before falling back to the older tiled split when the full kernel does not fit the device.
- The benchmark timing path now excludes NumPy/PyTorch reference generation. `bench.py`, `run_matrix.py`, and `run_workload_suite.py` measure `runtime.run(..., validate=False, capture_details=False)` inside timed loops and then run one untimed validation pass to emit correctness, transfer accounting, and trace data.
- Result JSON uses schema `edge-study-v1` and includes run metadata, cold/warm measurement mode, phase timings, p50/p95 latency, transfer events, host/device event summaries, correctness, NPU execution truth flags, and explicit study limitations.
- The iGPU path is compiled into per-kernel shared libraries under `artifacts/gpu/` and invoked through `_mlir_ciface_*_host` entrypoints.
- The GPU compile step uses `air-opt -air-to-rocdl -air-gpu-outlining -air-gpu-host-staging` before lowering with the local LLVM/MLIR toolchain from `LLVM_INSTALL_DIR`.
- The shared-library link step strips generated module dtors before linking so the Python `ctypes` path exits cleanly.
- The benchmark output now includes host-visible trace summaries, per-stage error metrics, and an NPU host-side buffer-layout report to support future hardware debugging.
- The default benchmark matrix includes both CPU/GPU and NPU cases. `run_matrix.py` still requires `--allow-npu` before it will touch hardware-backed NPU cases.
- PyTorch validation for the large workload suites now gates on final output correctness. Intermediate tensors are still recorded in the JSON results for debugging, but they are no longer the pass/fail condition for the model-sized benchmark runs.
- The AIR-to-ROCDL crash that used to affect these kernels is covered by `mlir/test/Conversion/AIRToROCDL/air_to_rocdl_launch_ids.mlir`.
