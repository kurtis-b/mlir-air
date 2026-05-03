#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-/tmp/npu_gpu_transfer_probe}"
if [[ -z "${ROCM_PATH:-}" ]]; then
  if [[ -d /opt/rocm-7.2.0 ]]; then
    ROCM_PATH=/opt/rocm-7.2.0
  else
    ROCM_PATH=/opt/rocm
  fi
fi
XILINX_XRT="${XILINX_XRT:-/opt/xilinx/xrt}"

hipcc --rocm-path="${ROCM_PATH}" --hip-path="${ROCM_PATH}" \
  -std=c++17 \
  "${SCRIPT_DIR}/transfer_probe.cpp" \
  -I"${ROCM_PATH}/include" \
  -I"${XILINX_XRT}/include" \
  -L"${ROCM_PATH}/lib" \
  -L"${XILINX_XRT}/lib" \
  -Wl,-rpath,"${ROCM_PATH}/lib" \
  -Wl,-rpath,"${XILINX_XRT}/lib" \
  -lxrt_coreutil -lxrt++ \
  -o "${OUT}"

printf '%s\n' "${OUT}"
