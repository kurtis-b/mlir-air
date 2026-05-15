#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TMPDIR="${TMPDIR:-/tmp/air_int8_gemm}"
GPU_CHIP="${AIR_GPU_CHIP:-gfx1150}"
export AIRGPU_USE_HIP_MALLOC="${AIRGPU_USE_HIP_MALLOC:-1}"
mkdir -p "$TMPDIR"

"$REPO_DIR/utils/isa_inspect/disassemble.sh" gpu --gpu-arch "$GPU_CHIP" \
  --output-dir "$TMPDIR" --prefix int8_gemm \
  --expect v_wmma_i32_16x16x16_iu8 \
  --forbid 'v_wmma_.*16x16x64|v_swmmac|swmmac' \
  "$SCRIPT_DIR/air_sync.mlir"

if [ -n "${MLIR_AIR_INSTALL_DIR:-}" ]; then
  AIRGPU_LIB="${MLIR_AIR_INSTALL_DIR}/lib/libairgpu.so"
else
  AIRGPU_LIB="$(dirname "$(command -v air-opt)")/../lib/libairgpu.so"
fi

mlir-runner --entry-point-result=void --shared-libs="$AIRGPU_LIB" \
  "$TMPDIR/int8_gemm.final.mlir"
