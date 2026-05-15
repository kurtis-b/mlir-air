#!/usr/bin/env bash
# Source this file from the repository root to activate the local LLVM GPU build.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Usage: source ${BASH_SOURCE[0]}"
  exit 1
fi

_air_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "${_air_repo_root}/sandbox/bin/activate" ]; then
  echo "ERROR: missing Python environment: ${_air_repo_root}/sandbox/bin/activate"
  return 1
fi

if [ ! -d "${_air_repo_root}/install-gpu" ]; then
  echo "ERROR: missing MLIR-AIR GPU install: ${_air_repo_root}/install-gpu"
  return 1
fi

if [ ! -d "${_air_repo_root}/llvm/install-amdgpu" ]; then
  echo "ERROR: missing LLVM AMDGPU install: ${_air_repo_root}/llvm/install-amdgpu"
  return 1
fi

source "${_air_repo_root}/sandbox/bin/activate"

if [ -d /opt/rocm ]; then
  export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
fi
export AIR_GPU_CHIP="${AIR_GPU_CHIP:-gfx1150}"

source "${_air_repo_root}/utils/env_setup_gpu.sh" \
  "${_air_repo_root}/install-gpu" \
  "${_air_repo_root}/llvm/install-amdgpu"

_air_print_tool() {
  _air_tool_path="$(command -v "$1" 2>/dev/null || true)"
  if [ -n "${_air_tool_path}" ]; then
    echo "$1: ${_air_tool_path}"
  else
    echo "$1: <not found>"
  fi
}

echo "AIR_GPU_CHIP: ${AIR_GPU_CHIP}"
if [ -n "${ROCM_PATH:-}" ]; then
  echo "ROCM_PATH: ${ROCM_PATH}"
fi
_air_print_tool air-opt
_air_print_tool aircc
_air_print_tool aie-opt
_air_print_tool mlir-opt

unset _air_repo_root
unset -f _air_print_tool
