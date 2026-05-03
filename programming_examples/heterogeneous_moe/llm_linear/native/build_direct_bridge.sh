#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-${SCRIPT_DIR}/libllm_linear_direct_bridge.so}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
XILINX_XRT="${XILINX_XRT:-/opt/xilinx/xrt}"

hipcc -std=c++17 -shared -fPIC \
  "${SCRIPT_DIR}/direct_bridge.cpp" \
  -I"${XILINX_XRT}/include" \
  -L"${XILINX_XRT}/lib" \
  -Wl,-rpath,"${XILINX_XRT}/lib" \
  -Wl,-rpath,"${ROCM_PATH}/lib" \
  -lxrt_coreutil -lxrt++ -ldl \
  -o "${OUT}"

printf '%s\n' "${OUT}"
