#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

LLVM_DIR="${LLVM_DIR:-llvm}"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-build}"
LLVM_INSTALL_NAME="${LLVM_INSTALL_NAME:-install}"
AIR_GPU_BUILD_DIR="${AIR_GPU_BUILD_DIR:-build-gpu}"
AIR_GPU_INSTALL_DIR="${AIR_GPU_INSTALL_DIR:-install-gpu}"
LLVM_INSTALL_DIR="${LLVM_INSTALL_DIR:-${LLVM_DIR}/${LLVM_INSTALL_NAME}}"

if [ -f "utils/setup_python_packages.sh" ]; then
  # shellcheck source=/dev/null
  source utils/setup_python_packages.sh
fi

if [ ! -d "$LLVM_DIR/llvm" ]; then
  if [ "$LLVM_DIR" = "llvm" ]; then
    echo "LLVM checkout not found; cloning with utils/clone-llvm.sh."
    ./utils/clone-llvm.sh
  else
    echo "ERROR: custom LLVM_DIR does not contain an llvm checkout: ${LLVM_DIR}" >&2
    exit 1
  fi
fi

if [ -f "${LLVM_INSTALL_DIR}/lib/cmake/mlir/MLIRConfig.cmake" ]; then
  echo "Using existing LLVM install: ${LLVM_INSTALL_DIR}"
else
  echo "Building LLVM incrementally; existing directories are preserved."
  ./utils/build-llvm-local.sh "$LLVM_DIR" "$LLVM_BUILD_DIR" "$LLVM_INSTALL_NAME"
fi

if [ -f "${AIR_GPU_BUILD_DIR}/build.ninja" ]; then
  echo "Existing GPU AIR build found: ${AIR_GPU_BUILD_DIR}"
  echo "Running incremental install target."
  ninja -C "$AIR_GPU_BUILD_DIR" install
else
  echo "Configuring GPU AIR build without deleting existing artifacts."
  ./utils/build-mlir-air-gpu.sh "$LLVM_INSTALL_DIR" "$AIR_GPU_BUILD_DIR" "$AIR_GPU_INSTALL_DIR"
fi

echo
echo "To use this build in the current shell:"
echo "  source utils/env_setup_gpu.sh ${AIR_GPU_INSTALL_DIR} ${LLVM_INSTALL_DIR}"
echo "Then run: bash agents/scripts/doctor.sh gpu"
