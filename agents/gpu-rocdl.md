# GPU/ROCDL Overlay

Canonical setup is `docs/buildingGPU.md`; `docs/aircc.md` covers common compiler driver options. Use this overlay for agent behavior around GPU-only MLIR-AIR builds.

## State Checks

Check:

- `LLVM_INSTALL_DIR` or `llvm/install` with `lib/cmake/mlir/MLIRConfig.cmake`.
- `MLIR_AIR_INSTALL_DIR` or `install-gpu`.
- `air-opt`, `aircc`, `mlir-opt`, `mlir-runner`, and ROCm tools such as `hipcc` when present.
- `build-gpu/CMakeCache.txt` for `AIR_ENABLE_AIE=OFF`, `AIR_ENABLE_GPU=ON`, `LLVM_DIR`, and `MLIR_DIR`.
- ROCm runtime availability separately from AIR compile availability.

## Incremental Rebuild Rules

- If only shell state is missing, source `utils/env_setup_gpu.sh`.
- If MLIR-AIR changed and `build-gpu/build.ninja` exists, run a targeted `ninja -C build-gpu <target>` or `ninja -C build-gpu install`.
- If LLVM changed, rebuild LLVM first, then rebuild the GPU AIR profile.
- If ROCm changed, rerun GPU runtime validation. Rebuild AIR only if CMake linkage or LLVM configuration depends on the changed ROCm installation.

Do not delete `llvm`, `build-gpu`, `install-gpu`, or other build products without explicit user direction.

## Smoke Policy

Prefer compile-only GPU lowering before `mlir-runner` execution:

```bash
bash agents/scripts/doctor.sh gpu
```

The smoke uses `aircc --target gpu` and writes outputs under `/tmp`. Runtime execution with `mlir-runner` requires ROCm runtime libraries and should be a separate verification step.
