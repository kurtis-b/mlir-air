# Agent Operational Overlay

Use this directory after reading `AGENTS.md`. The files here do not replace `docs/`; they tell agents how to choose the right canonical documentation, preserve local state, and verify narrowly.

## Routing

| Task | Canonical docs | Agent overlay |
| --- | --- | --- |
| Ryzen, AIE, NPU, `aiecc`, Peano, or XRT setup | `docs/buildingRyzenLin.md`, `docs/aircc.md` | `agents/ryzen-aie.md`, `agents/environment.md` |
| GPU lowering, ROCDL, HIP/OpenCL runtime, or `aircc --target gpu` | `docs/buildingGPU.md`, `docs/aircc.md` | `agents/gpu-rocdl.md`, `agents/environment.md` |
| Unit tests, lit tests, or build validation | `docs/testing.md`, `docs/building.md` | `agents/build-test.md` |
| AIR operations, hierarchy, dependency, async, or backend mapping | `docs/AIRComputeModel.md`, `docs/AIRAsyncConcurrency.md` | `agents/compiler-development.md` |
| GEMM/NPU pipeline work | `docs/GEMMCaseStudy.md`, `docs/buildingRyzenLin.md` | `agents/ryzen-aie.md`, `agents/compiler-development.md` |
| Runtime, runner, trace, or benchmark interpretation | `docs/AIRRunner.md`, `docs/trace.md` | `agents/benchmarking.md` |
| Documentation-only changes | Relevant file in `docs/` | Keep `agents/` changes limited to operational behavior |

## Default Flow

1. Identify the task profile.
2. Run `bash agents/scripts/doctor.sh env` or inspect the same state manually.
3. Read the canonical doc for the profile.
4. Source the matching environment if the build exists but tools are missing from `PATH`.
5. Rebuild incrementally with targeted `ninja` commands.
6. Verify with the narrowest command that exercises the changed behavior.

## Helper Commands

```bash
bash agents/scripts/doctor.sh env
bash agents/scripts/doctor.sh setup-plan ryzen
bash agents/scripts/doctor.sh setup-plan gpu
bash agents/scripts/doctor.sh npu
bash agents/scripts/doctor.sh gpu
```

The doctor script writes fingerprints under `agents/.state/`, which is ignored. Use those files to compare dependency paths, git heads, dirty state, build directories, CMake options, XRT, Peano, LLVM, and ROCm state between sessions.

Optional bootstrap helpers:

```bash
bash agents/scripts/bootstrap-ryzen-source.sh
bash agents/scripts/bootstrap-gpu.sh
```

These helpers are incremental entry points. They do not delete build directories.
