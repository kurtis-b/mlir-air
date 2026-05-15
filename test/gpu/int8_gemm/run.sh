#!/usr/bin/env bash
#===- run.sh --------------------------------------------------------------===//
#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
#===----------------------------------------------------------------------===//

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMPDIR="${TMPDIR:-/tmp/air_int8_gemm}"
GPU_CHIP="${AIR_GPU_CHIP:-gfx1150}"
export AIRGPU_USE_HIP_MALLOC="${AIRGPU_USE_HIP_MALLOC:-1}"
mkdir -p "$TMPDIR"

air-opt "$SCRIPT_DIR/air_sync.mlir" -air-to-rocdl -o "$TMPDIR/int8_gemm_rocdl.mlir"

air-opt "$TMPDIR/int8_gemm_rocdl.mlir" -air-gpu-outlining -o "$TMPDIR/int8_gemm_outline.mlir"

mlir-opt "--pass-pipeline=builtin.module(func.func(lower-affine,convert-linalg-to-loops,convert-scf-to-cf),gpu-kernel-outlining)" \
  "$TMPDIR/int8_gemm_outline.mlir" -o "$TMPDIR/int8_gemm_outline_llvm.mlir"

mlir-opt "--pass-pipeline=builtin.module(rocdl-attach-target{chip=${GPU_CHIP} O=3},gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl{chipset=${GPU_CHIP} runtime=HIP},reconcile-unrealized-casts),gpu-module-to-binary{format=isa})" \
  "$TMPDIR/int8_gemm_outline_llvm.mlir" -o "$TMPDIR/int8_gemm_isa.mlir"

if ! grep -q "v_wmma_i32_16x16x16_iu8" "$TMPDIR/int8_gemm_isa.mlir"; then
  echo "ERROR: expected v_wmma_i32_16x16x16_iu8 in generated ISA" >&2
  exit 1
fi

if grep -Eq "v_wmma_.*16x16x64|v_swmmac|swmmac" "$TMPDIR/int8_gemm_isa.mlir"; then
  echo "ERROR: generated ISA contains a gfx12-only WMMA form" >&2
  exit 1
fi

mlir-opt "--pass-pipeline=builtin.module(rocdl-attach-target{chip=${GPU_CHIP} O=3},gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl{chipset=${GPU_CHIP} runtime=HIP},reconcile-unrealized-casts),gpu-module-to-binary,func.func(gpu-async-region,convert-scf-to-cf),gpu-to-llvm,convert-to-llvm,reconcile-unrealized-casts)" \
  "$TMPDIR/int8_gemm_outline_llvm.mlir" -o "$TMPDIR/int8_gemm_final.mlir"

if [ -n "${MLIR_AIR_INSTALL_DIR:-}" ]; then
  AIRGPU_LIB="${MLIR_AIR_INSTALL_DIR}/lib/libairgpu.so"
else
  AIRGPU_LIB="$(dirname "$(command -v air-opt)")/../lib/libairgpu.so"
fi

mlir-runner --entry-point-result=void \
  --shared-libs="$AIRGPU_LIB" \
  "$TMPDIR/int8_gemm_final.mlir"
