#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-${SCRIPT_DIR}/llm_linear_rocm_blas_baseline}"
if [[ -z "${ROCM_PATH:-}" ]]; then
  if [[ -d /opt/rocm-7.2.0 ]]; then
    ROCM_PATH=/opt/rocm-7.2.0
  else
    ROCM_PATH=/opt/rocm
  fi
fi

hipcc --rocm-path="${ROCM_PATH}" --hip-path="${ROCM_PATH}" \
  -std=c++17 \
  "${SCRIPT_DIR}/rocm_blas_baseline.cpp" \
  -I"${ROCM_PATH}/include" \
  -L"${ROCM_PATH}/lib" \
  -Wl,-rpath,"${ROCM_PATH}/lib" \
  -D__AMDGCN_WAVEFRONT_SIZE=32 \
  -lrocblas -lhipblas \
  -o "${OUT}"

printf '%s\n' "${OUT}"
