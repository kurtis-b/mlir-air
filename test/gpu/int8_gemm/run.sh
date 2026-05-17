#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TMPDIR="${TMPDIR:-/tmp/air_int8_gemm}"
GPU_CHIP="${AIR_GPU_CHIP:-gfx1150}"
export AIRGPU_USE_HIP_MALLOC="${AIRGPU_USE_HIP_MALLOC:-1}"
if [ -z "${AIR_OPT:-}" ] && [ -x "$REPO_DIR/build-gpu/bin/air-opt" ]; then
  export AIR_OPT="$REPO_DIR/build-gpu/bin/air-opt"
fi
if [ -z "${MLIR_OPT:-}" ] && [ -x "$REPO_DIR/llvm/install-amdgpu/bin/mlir-opt" ]; then
  export MLIR_OPT="$REPO_DIR/llvm/install-amdgpu/bin/mlir-opt"
fi
mkdir -p "$TMPDIR"

"$REPO_DIR/utils/isa_inspect/disassemble.sh" gpu --gpu-arch "$GPU_CHIP" \
  --output-dir "$TMPDIR" --prefix int8_gemm \
  --expect v_wmma_i32_16x16x16_iu8 \
  --forbid 'v_wmma_.*16x16x64|v_swmmac|swmmac' \
  "$SCRIPT_DIR/air_sync.mlir"

MLIR_RUNNER="${MLIR_RUNNER:-$(command -v mlir-runner || true)}"
[ -n "$MLIR_RUNNER" ] || MLIR_RUNNER="$REPO_DIR/llvm/install-amdgpu/bin/mlir-runner"
[ -x "$MLIR_RUNNER" ] || { echo "ERROR: mlir-runner not found" >&2; exit 1; }

AIRGPU_LIB="${AIRGPU_LIB:-${MLIR_AIR_INSTALL_DIR:+$MLIR_AIR_INSTALL_DIR/lib/libairgpu.so}}"
[ -n "$AIRGPU_LIB" ] || AIRGPU_LIB="$REPO_DIR/install-gpu/lib/libairgpu.so"
[ -f "$AIRGPU_LIB" ] || { echo "ERROR: libairgpu.so not found" >&2; exit 1; }

"$MLIR_RUNNER" --entry-point-result=void --shared-libs="$AIRGPU_LIB" \
  "$TMPDIR/int8_gemm.final.mlir"
