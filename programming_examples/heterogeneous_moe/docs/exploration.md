# Heterogeneous MoE Exploration Guide

This guide is the recommended path for getting from a fresh checkout to useful heterogeneous MoE results. It assumes you know the MLIR-AIR repository layout, but not this example's scripts, artifacts, or hardware gates.

## Backend Setup Matrix

| Backend | Required setup | First useful command |
| --- | --- | --- |
| CPU | Repo-local Python sandbox with NumPy and optional CPU PyTorch. | `python3 smoke_tests.py --lane ci` |
| GPU | CPU setup plus local AMDGPU LLVM tools, ROCm runtime, and `air-opt`. Set `LLVM_INSTALL_DIR`, `ROCM_PATH`, and `AIR_OPT_PATH`, or put the tools on `PATH`. | `python3 smoke_tests.py --lane gpu-all` |
| NPU | CPU setup plus `AIRCC_PATH`, XRT environment, visible Ryzen AI NPU, and explicit `--allow-npu`. | `python3 smoke_tests.py --lane npu --allow-npu` |

Common setup:

```bash
cd <mlir-air-repo>/programming_examples/heterogeneous_moe
./setup_sandbox.sh
source ../../sandbox/bin/activate
```

GPU setup:

```bash
export LLVM_INSTALL_DIR=<path-to-llvm-amdgpu-install>
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export AIR_OPT_PATH=<path-to-air-opt>
```

NPU setup:

```bash
export AIRCC_PATH=<path-to-aircc>
source /opt/xilinx/xrt/setup.sh
xrt-smi examine
```

## Recommended Path

Start with the CI-safe lane. It checks golden AIR generation and CPU `top1`/`top2` behavior without compiling hardware artifacts.

```bash
python3 smoke_tests.py --lane ci
```

Run the deterministic Python coverage gate when changing harness logic:

```bash
python3 run_coverage.py
```

The same coverage gate is available from a configured CMake build:

```bash
ninja -C build check-heterogeneous-moe-coverage
```

Move to GPU only after the toolchain variables are set:

```bash
python3 compile_kernels.py --backends gpu
python3 smoke_tests.py --lane gpu-all
```

Move to NPU only after XRT is sourced and the device is visible:

```bash
python3 smoke_tests.py --lane npu --allow-npu
```

## Compile Flow

`compile_kernels.py` reads `default_manifest.json` by default, generates AIR under the manifest's generated source root, compiles the requested non-CPU backends, and writes a sidecar manifest:

```bash
python3 compile_kernels.py --backends gpu --manifest-out artifacts/compiled_manifest.json
```

The sidecar manifest is the one to pass to benchmark commands that need precompiled artifacts:

```bash
python3 bench.py --manifest artifacts/compiled_manifest.json \
  --router-backend gpu \
  --expert0-backend gpu \
  --expert1-backend gpu \
  --aggregation-backend gpu \
  --router-mode top2 \
  --iterations 1 \
  --warmup 0 \
  --require-correctness
```

NPU compilation is not guarded by `--allow-npu` in `compile_kernels.py`; only request it when the NPU toolchain is intentionally configured:

```bash
python3 compile_kernels.py --backends npu --manifest-out artifacts/compiled_manifest.json
```

## One-Off Benchmarks

`bench.py` runs one placement. The default manifest is CPU-only, but all placements can be overridden on the command line.

```bash
python3 bench.py --iterations 1 --warmup 0 \
  --router-backend cpu \
  --expert0-backend cpu \
  --expert1-backend cpu \
  --aggregation-backend cpu \
  --router-mode top2 \
  --require-correctness
```

Useful flags:

| Flag | Meaning |
| --- | --- |
| `--prepare` | Compile/load selected non-CPU executors, then exit. |
| `--measurement-mode cold|warm|both` | Choose cold start, warm steady-state, or both. Validation is still untimed. |
| `--transfer-mode host|peer|auto` | Select the transfer model for stage edges. |
| `--require-correctness` | Fail if final output validation exceeds dtype tolerances. |
| `--require-torch` | Fail if PyTorch validation is unavailable or fails. |
| `--results-out`, `--csv-out` | Write one-case structured summaries. |
| `--trace-out`, `--trace-summary-out` | Write Chrome trace JSON and summary data. |
| `--stage-metrics-out` | Write per-stage correctness metrics. |
| `--transfer-summary-out`, `--device-events-out` | Write transfer and host/device event summaries. |
| `--npu-dev-report-out` | Write the host-visible NPU buffer and artifact report. |

## Matrices And Workload Suites

Run a small matrix first. `--case-filter` matches exact case names from `default_benchmark_matrix.json`.

```bash
python3 run_matrix.py \
  --case-filter cpu_top2 gpu_top2 \
  --iterations 1 \
  --warmup 0 \
  --require-correctness
```

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

Run workload suites after the small matrix is stable. `--suite` accepts `shape_sweep`, `routing_sweep`, and `model_presets`. `--workload-filter` matches substrings in generated workload names, while `--case-filter` still matches exact case names.

```bash
python3 run_workload_suite.py --suite model_presets \
  --workload-filter qwen36_35b_a3b_qbf16 \
  --case-filter cpu_top2 gpu_top2 \
  --iterations 1 \
  --warmup 0 \
  --require-correctness \
  --output-dir artifacts/benchmarks/workload_suites/qwen_context_sweep
```

Workload suites write one directory per workload under `<output-dir>/<suite>/<workload>/`, each with a `compiled_manifest.json`, `summary.json`, and `cases/<case>.json`. The suite root also gets `summary.json`, `summary.csv`, and `report.md`.

## Edge Study

`edge_study.py` wraps `run_workload_suite.py` and then writes an edge-efficiency summary and markdown report. Profiles are:

| Profile | Suites |
| --- | --- |
| `smoke` | `shape_sweep` |
| `routing` | `shape_sweep`, `routing_sweep` |
| `model` | `model_presets` |
| `full` | `shape_sweep`, `routing_sweep`, `model_presets` |

Start with a filtered smoke study:

```bash
python3 edge_study.py --profile smoke \
  --case-filter cpu_top2 gpu_top2 \
  --iterations 1 \
  --warmup 0 \
  --require-correctness \
  --output-dir artifacts/benchmarks/edge_study/smoke
```

The edge-study root contains `suite/`, `edge_efficiency_summary.json`, and `edge_efficiency_report.md`.

## NPU Gates

NPU cases in `run_matrix.py`, `run_workload_suite.py`, `edge_study.py`, and `smoke_tests.py` are skipped unless `--allow-npu` is passed. The result truth flag is `npu_development.executed`, also mirrored in `execution_truth.npu_executed`; do not infer NPU execution from the case name alone.

Example NPU workload command:

```bash
python3 run_workload_suite.py --suite shape_sweep \
  --allow-npu \
  --require-correctness \
  --iterations 1 \
  --warmup 1 \
  --output-dir artifacts/benchmarks/workload_suites/npu_shape_sweep
```

## LLM-Linear Milestone 2 Direct Acceptance

The LLM-linear direct GPU/NPU handoff hardware gate is accepted as of May 3,
2026. Run it from `programming_examples/heterogeneous_moe` with:

```bash
source /opt/xilinx/xrt/setup.sh && ../../sandbox/bin/python run_llm_linear_milestone2.py
```

The wrapper writes accepted evidence under
`llm_linear/artifacts/benchmarks/milestone2_e2e`, builds the native direct
bridge, runs `tiny_g2n_direct`, `tiny_n2g_direct`, `medium_g2n_direct`,
`medium_n2g_direct`, and `medium_host_mixed`, and rejects logs containing the
known XRT host-copy fallback markers. The accepted medium workload is
`medium_m8_k512_h512_n256`.

## LLM-Linear Milestone 4 Crossover Acceptance

The final LLM-linear crossover hardware gate is accepted as of May 4, 2026. Run
it from `programming_examples/heterogeneous_moe` with:

```bash
source /opt/xilinx/xrt/setup.sh && ../../sandbox/bin/python run_llm_linear_milestone4.py
```

The wrapper writes accepted evidence under
`llm_linear/artifacts/benchmarks/milestone4_crossover`, including per-storage
BF16 and signed-int4 suite outputs plus the top-level `report.md`. It covers
`tiny_ci`, `medium`, and `llm_like` across CPU-only, GPU-only, NPU-only,
host-staged mixed, and audited direct mixed cases. The final report classifies
both BF16 and int4 as 13 wins, 32 losses, and 3 inconclusive direct/baseline
comparisons: direct mixed execution helps some larger accelerator baselines, but
the study does not show a universal crossover.

## Artifact Hygiene

Keep `default_manifest.json` portable. It should not contain machine-local compiled paths. Generated sources, shared libraries, NPU binaries, benchmark trees, coverage reports, and sidecar manifests belong under ignored roots such as:

- `artifacts/generated_air/`
- `artifacts/compiled_manifest.json`
- `artifacts/gpu/`
- `artifacts/npu/`
- `artifacts/workloads/`
- `artifacts/benchmarks/`
- `artifacts/coverage/`

If a result looks stale, remove the specific artifact subdirectory for that run rather than editing the default manifest.

## Troubleshooting

`Required tool 'air-opt' is not on PATH`: set `AIR_OPT_PATH=<path-to-air-opt>` or put `air-opt` on `PATH`.

`LLVM_INSTALL_DIR is unset`: set it to the LLVM/MLIR install used for AMDGPU lowering. The GPU flow expects `mlir-opt`, `mlir-translate`, `clang`, and MLIR ROCm runtime libraries under that install.

`libamdhip64.so` or HIP runtime load failure: set `ROCM_PATH` to the ROCm install root and confirm the library exists under `$ROCM_PATH/lib`.

GPU shared library cannot find `_mlir_ciface_*`: recompile the GPU artifacts for the current AIR sources and manifest shape, then rerun with the new sidecar manifest.

`Peer transfer is not supported for edge npu->gpu` or `gpu->npu`: use `--transfer-mode auto` or `host`. Direct iGPU-to-NPU peer transfer is intentionally unsupported in this harness.

NPU cases are listed as skipped: pass `--allow-npu` only after XRT is sourced and `xrt-smi examine` sees the device.

`npu_development.executed` is `false`: the run did not execute an NPU stage, even if an NPU report file was written.

PyTorch validation fails because PyTorch is unavailable: omit `--require-torch` for infrastructure smoke, or rerun `./setup_sandbox.sh` to install the CPU validation dependency.
