# Environment And State Policy

`docs/` explains how to build and run MLIR-AIR. This file explains how agents should inspect and preserve local state before acting.

## State Check

Before setup, build, test, or benchmark work, record:

- Current branch, `HEAD`, and dirty worktree status.
- Whether untracked files are source files, generated artifacts, or local build trees.
- Active Python environment, `PATH`, `PYTHONPATH`, and `LD_LIBRARY_PATH` clues.
- `MLIR_AIR_INSTALL_DIR`, `MLIR_AIE_INSTALL_DIR`, `PEANO_INSTALL_DIR`, `LLVM_INSTALL_DIR`, `XILINX_XRT`, `ROCM_PATH`, and matching tool paths.
- Existing build/install directories and their `CMakeCache.txt` options.
- Dependency checkout heads for local LLVM, MLIR-AIE, llvm-aie, or other adjacent repos when present.

Run `bash agents/scripts/doctor.sh env` when possible. It records a fingerprint under `agents/.state/`.

## Sourced-Shell Recovery

If tools are missing from `PATH` but build/install directories exist, source the setup script instead of rebuilding:

```bash
source utils/env_setup.sh install <mlir-aie-install> <llvm-aie-install> my_install/mlir
source utils/env_setup_gpu.sh install-gpu llvm/install
source /opt/xilinx/xrt/setup.sh
```

Use `python3 -m pip show mlir_aie` and `python3 -m pip show llvm-aie` to locate wheel installs when using the Ryzen source build.

## Generated Artifacts

Keep generated state out of commits. Use:

- `agents/.state/` for fingerprints and agent-local notes.
- `/tmp/` for smoke-test build products.
- Existing ignored build directories for normal CMake/Ninja outputs.

Do not add build trees, install trees, benchmark output, `__pycache__/`, or local logs unless the user explicitly asks.

## Fingerprint Policy

Fingerprints should include the profile, dependency paths, git heads, dirty status, build/install directories, CMake options, `PEANO_INSTALL_DIR`, XRT path, LLVM install path, and ROCm runtime detection.

Use fingerprints to decide whether a shell needs sourcing, an AIR target needs a targeted rebuild, or a dependency changed. A fingerprint mismatch is not permission for a clean rebuild.
