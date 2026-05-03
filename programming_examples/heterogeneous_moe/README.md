# Heterogeneous MoE

This example is a fixed-shape, two-expert Mixture-of-Experts research harness for the MLIR-AIR repository. It exists to exercise one AIR kernel family across CPU, Ryzen iGPU, and Ryzen XDNA NPU execution paths while keeping the runtime small enough to inspect.

This harness is now a prototype/reference being closed out. Future work on efficient Ryzen heterogeneous execution should move to the [LLM-linear roadmap](docs/ryzen_heterogeneous_execution_todo.md), which focuses on GEMM/GEMV, direct GPU/NPU handoff, and low-bit linear inference patterns rather than expanding this MoE harness.

## What It Proves

- Router math, expert 0, expert 1, and aggregation can be assigned independently to `cpu`, `gpu`, or `npu`.
- The same generated AIR sources feed the NPU compile path and the iGPU `air-to-rocdl` path.
- `top1` and `top2` routing work through a shared validation and benchmark harness.
- Mixed placements can be measured with structured JSON, CSV, trace, transfer, and correctness outputs.

## What It Does Not Prove

- It is not a production MoE runtime or scheduler.
- It does not model full transformer attention, KV cache, tokenization, or service overhead.
- Transfer events are a NumPy host-array model, not a true device-resident DMA timeline.
- NPU results are opt-in hardware runs; use the smoke lanes to validate the current machine before treating local numbers as meaningful.

## Start Here

```bash
cd <mlir-air-repo>/programming_examples/heterogeneous_moe
./setup_sandbox.sh
source ../../sandbox/bin/activate
python3 smoke_tests.py --lane ci
```

Recommended exploration order:

1. Run the CI-safe CPU lane above.
2. Run the deterministic Python coverage gate: `python3 run_coverage.py`.
3. If ROCm and the local AMDGPU LLVM tools are configured, compile and run `gpu-all`.
4. Run a filtered matrix or workload suite before attempting broad sweeps.
5. Only run NPU lanes after sourcing XRT and passing `--allow-npu`.

## Command Chooser

| Goal | Command |
| --- | --- |
| CPU smoke lane | `python3 smoke_tests.py --lane ci` |
| Deterministic Python coverage | `python3 run_coverage.py` |
| Compile GPU artifacts | `python3 compile_kernels.py --backends gpu` |
| GPU and mixed GPU smoke lanes | `python3 smoke_tests.py --lane gpu-all` |
| One benchmark case | `python3 bench.py --iterations 1 --warmup 0 --router-backend cpu --expert0-backend cpu --expert1-backend cpu --aggregation-backend cpu --router-mode top2 --require-correctness` |
| CPU/GPU matrix subset | `python3 run_matrix.py --case-filter cpu_top2 gpu_top2 --iterations 1 --warmup 0 --require-correctness` |
| One model-preset workload | `python3 run_workload_suite.py --suite model_presets --workload-filter qwen36_35b_a3b_qbf16 --case-filter cpu_top2 gpu_top2 --iterations 1 --warmup 0 --require-correctness` |
| Edge-efficiency smoke study | `python3 edge_study.py --profile smoke --case-filter cpu_top2 gpu_top2 --iterations 1 --warmup 0 --require-correctness` |
| NPU smoke lane | `python3 smoke_tests.py --lane npu --allow-npu` |
| LLM-linear CPU smoke | `../../sandbox/bin/python run_llm_linear_suite.py --suite tiny_ci --case-filter cpu_only --iterations 1 --warmup 0 --require-correctness` |
| LLM-linear int4 decode CPU smoke | `../../sandbox/bin/python run_llm_linear_suite.py --suite tiny_ci --case-filter cpu_only --decode-weight-storage int4 --iterations 1 --warmup 0 --require-correctness` |
| LLM-linear Milestone 2 direct acceptance | `source /opt/xilinx/xrt/setup.sh && ../../sandbox/bin/python run_llm_linear_milestone2.py` |
| LLM-linear Milestone 3 int4 hardware acceptance | `source /opt/xilinx/xrt/setup.sh && ../../sandbox/bin/python run_llm_linear_milestone3.py` |

GPU commands need `LLVM_INSTALL_DIR`, `ROCM_PATH`, and `AIR_OPT_PATH` or matching tools on `PATH`. NPU commands need `AIRCC_PATH`, XRT setup, visible hardware, and explicit `--allow-npu`. The Milestone 2 and Milestone 3 wrappers seed the common local tool paths when they exist and reject direct runs that emit the XRT host-copy fallback warning. See the [exploration guide](docs/exploration.md) for the setup matrix and troubleshooting flow.

LLM-linear Milestone 2 direct GPU/NPU handoff is accepted as of May 3, 2026 for
the hardware gate. The accepted command is:
`source /opt/xilinx/xrt/setup.sh && ../../sandbox/bin/python run_llm_linear_milestone2.py`.
It writes evidence under `llm_linear/artifacts/benchmarks/milestone2_e2e` and
covers the `medium_m8_k512_h512_n256` workload in both direct directions plus
matching host-mixed baselines.

LLM-linear Milestone 3 fused int4 decode is accepted as of May 3, 2026 for the
hardware gate. The accepted command is:
`source /opt/xilinx/xrt/setup.sh && ../../sandbox/bin/python run_llm_linear_milestone3.py`.
It writes evidence under `llm_linear/artifacts/benchmarks/milestone3_int4_hw`
and covers `tiny_ci`, `medium`, and `llm_like` workloads across CPU/GPU/NPU,
host-mixed, and direct mixed cases with signed int4 decode weights. Accelerator
decode remains intentionally narrow: signed int4, `quant_axis=0`,
`H % block_size == 0`, and `N % 8 == 0`.

## Key Files

- `default_manifest.json`: portable default model, runtime, compiler, benchmark, input, weight, and artifact schema.
- `default_benchmark_matrix.json`: named CPU/GPU/NPU placement cases.
- `kernels.py`: emits router, expert, aggregation, and split expert AIR sources.
- `compile_kernels.py`: compiles selected GPU/NPU artifacts and writes a sidecar manifest.
- `bench.py`: runs one case and can emit JSON, CSV, trace, transfer, stage metric, device event, and NPU development files.
- `smoke_tests.py`: first-run validation lanes, with NPU gated by `--allow-npu`.
- `run_matrix.py`: matrix runner that writes per-case outputs plus aggregate JSON/CSV.
- `run_workload_suite.py`: shape, routing, and model-preset workload suites.
- `edge_study.py`: canonical edge-efficiency wrapper over workload suites.
- `llm_linear/`: LLM-linear GEMM/GEMV benchmark for the Ryzen heterogeneous roadmap, including host-staged baselines, hardware-gated native direct handoff plumbing, int4/uint4 decode-weight packing, and crossover report plumbing.

## Deeper Docs

- [Exploration guide](docs/exploration.md): setup by backend, first runs, command recipes, hardware gates, artifact hygiene, and troubleshooting.
- [Architecture guide](docs/architecture.md): dataflow, backend execution model, transfer semantics, manifest and result schemas, output interpretation, and limitations.
- [Ryzen heterogeneous execution roadmap](docs/ryzen_heterogeneous_execution_todo.md): close-out evaluation for this MoE harness and ordered next steps for an MLIR-AIR-first LLM-linear benchmark.

Keep `default_manifest.json` portable. Compiled paths belong in sidecar manifests under ignored artifact roots such as `artifacts/compiled_manifest.json` or workload-suite output directories.
