# Ryzen/AIE/NPU Overlay

Canonical setup is `docs/buildingRyzenLin.md`; `docs/aircc.md` covers the driver. Use this overlay for agent behavior around incremental builds and smoke tests.

## State Checks

Check:

- `MLIR_AIR_INSTALL_DIR`, `MLIR_AIE_INSTALL_DIR`, `PEANO_INSTALL_DIR`, and `LLVM_INSTALL_DIR`.
- `air-opt`, `aircc`, `aie-opt`, `aiecc`, and `clang++` under `PEANO_INSTALL_DIR`.
- XRT only when hardware execution, xclbin generation, or XRT tests are required.
- `build/CMakeCache.txt` for `AIR_ENABLE_AIE`, `AIE_DIR`, `PEANO_INSTALL_DIR`, and XRT options.

## Incremental Rebuild Rules

- If only shell state is missing, source `utils/env_setup.sh` and, for hardware, XRT setup.
- If MLIR-AIR changed and `build/build.ninja` exists, run a targeted `ninja -C build <target>` or `ninja -C build install`.
- If LLVM wheel contents changed, rebuild the LLVM wheel unpack or rerun the source build script, then rebuild AIR.
- If MLIR-AIE changed, rebuild or reinstall MLIR-AIE, then rebuild AIR.
- If Peano or XRT configuration changed, reconfigure the Ryzen build before rebuilding.

Do not delete `build`, `install`, `my_install`, or other build products without explicit user direction.

## Smoke Policy

Prefer compile-only checks before hardware runs:

```bash
bash agents/scripts/doctor.sh npu
```

This runs a small Peano-backed NPU compile under `/tmp` when the environment is complete. Treat missing hardware as separate from compiler failures; hardware is only required for XRT execution or xclbin workflows.
