#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BUILD_DIR="${AIR_BUILD_DIR:-build}"
INSTALL_DIR="${AIR_INSTALL_DIR:-install}"
XRT_DIR="${XRT_DIR:-}"

if [ -f "utils/setup_python_packages.sh" ]; then
  # shellcheck source=/dev/null
  source utils/setup_python_packages.sh
fi

if [ -f "${BUILD_DIR}/build.ninja" ]; then
  echo "Existing Ryzen/AIE build found: ${BUILD_DIR}"
  echo "Running incremental install target."
  ninja -C "$BUILD_DIR" install
else
  args=()
  if [ -n "$XRT_DIR" ]; then
    args+=(--xrt-dir "$XRT_DIR")
  fi
  echo "Configuring Ryzen/AIE source build without deleting existing artifacts."
  ./utils/build-mlir-air-using-wheels.sh "${args[@]}" "$BUILD_DIR" "$INSTALL_DIR"
fi

echo
echo "To use this build in the current shell, source the environment described in docs/buildingRyzenLin.md."
echo "Then run: bash agents/scripts/doctor.sh npu"
