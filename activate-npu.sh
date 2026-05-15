#!/usr/bin/env bash
# Source this file from the repository root to activate the wheel/XRT NPU build.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Usage: source ${BASH_SOURCE[0]}"
  exit 1
fi

_air_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "${_air_repo_root}/sandbox/bin/activate" ]; then
  echo "ERROR: missing Python environment: ${_air_repo_root}/sandbox/bin/activate"
  return 1
fi

source "${_air_repo_root}/sandbox/bin/activate"

_mlir_aie_site="$(python3 -m pip show mlir-aie 2>/dev/null | sed -n 's/^Location: //p')"
_llvm_aie_site="$(python3 -m pip show llvm-aie 2>/dev/null | sed -n 's/^Location: //p')"
_mlir_aie_dir="${_mlir_aie_site}/mlir_aie"
_llvm_aie_dir="${_llvm_aie_site}/llvm-aie"

if [ ! -d "${_mlir_aie_dir}" ]; then
  echo "ERROR: missing MLIR-AIE wheel directory: ${_mlir_aie_dir}"
  return 1
fi

if [ ! -d "${_llvm_aie_dir}" ]; then
  echo "ERROR: missing LLVM-AIE wheel directory: ${_llvm_aie_dir}"
  return 1
fi

if [ ! -d "${_air_repo_root}/install-xrt" ]; then
  echo "ERROR: missing MLIR-AIR XRT install: ${_air_repo_root}/install-xrt"
  return 1
fi

if [ ! -d "${_air_repo_root}/my_install/mlir" ]; then
  echo "ERROR: missing MLIR wheel support install: ${_air_repo_root}/my_install/mlir"
  return 1
fi

source "${_air_repo_root}/utils/env_setup.sh" \
  "${_air_repo_root}/install-xrt" \
  "${_mlir_aie_dir}" \
  "${_llvm_aie_dir}" \
  "${_air_repo_root}/my_install/mlir"

if [ ! -f /opt/xilinx/xrt/setup.sh ]; then
  echo "ERROR: missing XRT setup script: /opt/xilinx/xrt/setup.sh"
  return 1
fi

source /opt/xilinx/xrt/setup.sh

_air_print_tool() {
  _air_tool_path="$(command -v "$1" 2>/dev/null || true)"
  if [ -n "${_air_tool_path}" ]; then
    echo "$1: ${_air_tool_path}"
  else
    echo "$1: <not found>"
  fi
}

_air_print_tool air-opt
_air_print_tool aircc
_air_print_tool aie-opt
_air_print_tool mlir-opt

unset _air_repo_root _mlir_aie_site _llvm_aie_site _mlir_aie_dir _llvm_aie_dir
unset -f _air_print_tool
