#!/usr/bin/env bash

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STATE_DIR="${ROOT_DIR}/agents/.state"

usage() {
  cat <<'USAGE'
Usage:
  doctor.sh env
  doctor.sh setup-plan ryzen
  doctor.sh setup-plan gpu
  doctor.sh npu
  doctor.sh gpu

Reports environment state, prints incremental setup plans, or runs compile-only
smoke checks. Generated fingerprints are written under agents/.state/.
USAGE
}

section() {
  printf '\n== %s ==\n' "$1"
}

have() {
  command -v "$1" >/dev/null 2>&1
}

tool_line() {
  local tool="$1"
  if have "$tool"; then
    printf '%-14s %s\n' "$tool:" "$(command -v "$tool")"
  else
    printf '%-14s missing\n' "$tool:"
  fi
}

dir_line() {
  local label="$1"
  local path="$2"
  if [ -e "$path" ]; then
    printf '%-22s %s\n' "$label:" "$(realpath "$path" 2>/dev/null || printf '%s' "$path")"
  else
    printf '%-22s missing (%s)\n' "$label:" "$path"
  fi
}

git_head() {
  local path="$1"
  git -C "$path" rev-parse --short HEAD 2>/dev/null || printf 'n/a'
}

git_dirty() {
  local path="$1"
  local count
  count="$(git -C "$path" status --short 2>/dev/null | wc -l | tr -d ' ')"
  if [ -z "$count" ]; then
    printf 'n/a'
  elif [ "$count" = "0" ]; then
    printf 'clean'
  else
    printf '%s entries' "$count"
  fi
}

pip_location() {
  local package="$1"
  if have python3; then
    python3 -m pip show "$package" 2>/dev/null | awk '/^Location:/ {print $2; exit}'
  fi
}

resolve_peano_dir() {
  if [ -n "${PEANO_INSTALL_DIR:-}" ]; then
    printf '%s' "$PEANO_INSTALL_DIR"
    return
  fi
  local loc
  loc="$(pip_location llvm-aie)"
  if [ -n "$loc" ]; then
    printf '%s/llvm-aie' "$loc"
    return
  fi
  local cache_dir
  for cache_dir in "$ROOT_DIR/build/CMakeCache.txt" "$ROOT_DIR/build-gpu/CMakeCache.txt"; do
    loc="$(cache_value "$cache_dir" PEANO_INSTALL_DIR)"
    if [ -n "$loc" ] && [ "$loc" != "<unset>" ]; then
      printf '%s' "$loc"
      return
    fi
  done
}

resolve_mlir_aie_dir() {
  if [ -n "${MLIR_AIE_INSTALL_DIR:-}" ]; then
    printf '%s' "$MLIR_AIE_INSTALL_DIR"
    return
  fi
  local loc
  loc="$(pip_location mlir_aie)"
  if [ -n "$loc" ]; then
    printf '%s/mlir_aie' "$loc"
    return
  fi
  loc="$(cache_value "$ROOT_DIR/build/CMakeCache.txt" AIE_DIR)"
  if [ -n "$loc" ] && [ "$loc" != "<unset>" ]; then
    printf '%s' "${loc%/lib/cmake/aie}"
  fi
}

resolve_llvm_install_dir() {
  if [ -n "${LLVM_INSTALL_DIR:-}" ]; then
    printf '%s' "$LLVM_INSTALL_DIR"
    return
  fi
  local loc
  for cache in "$ROOT_DIR/build-gpu/CMakeCache.txt" "$ROOT_DIR/build/CMakeCache.txt"; do
    loc="$(cache_value "$cache" LLVM_DIR)"
    if [ -n "$loc" ] && [ "$loc" != "<unset>" ]; then
      printf '%s' "${loc%/lib/cmake/llvm}"
      return
    fi
  done
  printf '%s' "${ROOT_DIR}/llvm/install"
}

resolve_xrt_dir() {
  if [ -n "${XILINX_XRT:-}" ]; then
    printf '%s' "$XILINX_XRT"
  elif [ -d /opt/xilinx/xrt ]; then
    printf '%s' /opt/xilinx/xrt
  fi
}

resolve_rocm_dir() {
  if [ -n "${ROCM_PATH:-}" ]; then
    printf '%s' "$ROCM_PATH"
  elif have hipconfig; then
    hipconfig --rocmpath 2>/dev/null | awk 'NF {print; exit}'
  elif [ -d /opt/rocm ]; then
    printf '%s' /opt/rocm
  elif have hipcc; then
    local hipcc_path
    hipcc_path="$(command -v hipcc)"
    realpath "$(dirname "$hipcc_path")/.." 2>/dev/null || dirname "$hipcc_path"
  fi
}

rocm_runtime_status() {
  local rocm_dir
  rocm_dir="$(resolve_rocm_dir)"
  if [ -z "$rocm_dir" ]; then
    printf 'missing'
    return
  fi
  if [ -f "$rocm_dir/lib/libamdhip64.so" ] || [ -f "$rocm_dir/lib64/libamdhip64.so" ]; then
    printf 'present (%s)' "$rocm_dir"
  else
    printf 'directory found, HIP runtime library not found (%s)' "$rocm_dir"
  fi
}

cache_value() {
  local cache="$1"
  local key="$2"
  if [ -f "$cache" ]; then
    awk -F= -v key="$key" '$1 ~ "^" key ":" {print $2; exit}' "$cache"
  fi
}

cache_summary() {
  local label="$1"
  local cache="$2"
  if [ ! -f "$cache" ]; then
    printf '%s: missing\n' "$label"
    return
  fi
  printf '%s: %s\n' "$label" "$cache"
  for key in AIR_ENABLE_AIE AIR_ENABLE_GPU CMAKE_BUILD_TYPE CMAKE_INSTALL_PREFIX LLVM_DIR MLIR_DIR AIE_DIR PEANO_INSTALL_DIR LLVM_EXTERNAL_LIT XRT_LIB_DIR XRT_BIN_DIR XRT_INCLUDE_DIR; do
    local value
    value="$(cache_value "$cache" "$key")"
    if [ -n "$value" ]; then
      printf '  %s=%s\n' "$key" "$value"
    fi
  done
}


try_source_python_env() {
  if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "${ROOT_DIR}/sandbox/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${ROOT_DIR}/sandbox/bin/activate"
  fi
}

write_fingerprint() {
  local profile="$1"
  mkdir -p "$STATE_DIR"
  {
    printf 'timestamp_utc=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf 'profile=%s\n' "$profile"
    printf 'repo=%s\n' "$ROOT_DIR"
    printf 'mlir_air_head=%s\n' "$(git_head "$ROOT_DIR")"
    printf 'mlir_air_dirty=%s\n' "$(git_dirty "$ROOT_DIR")"
    printf 'branch=%s\n' "$(git -C "$ROOT_DIR" branch --show-current 2>/dev/null || printf 'n/a')"
    printf 'mlir_aie_dir=%s\n' "$(resolve_mlir_aie_dir)"
    printf 'peano_dir=%s\n' "$(resolve_peano_dir)"
    printf 'xrt_dir=%s\n' "$(resolve_xrt_dir)"
    printf 'llvm_install_dir=%s\n' "$(resolve_llvm_install_dir)"
    printf 'rocm_runtime=%s\n' "$(rocm_runtime_status)"
    for dir in "$ROOT_DIR/../mlir-aie" "$ROOT_DIR/../llvm-aie" "$ROOT_DIR/llvm"; do
      if [ -d "$dir/.git" ]; then
        printf 'dependency=%s head=%s dirty=%s\n' "$(realpath "$dir")" "$(git_head "$dir")" "$(git_dirty "$dir")"
      fi
    done
    for dir in build install build-gpu install-gpu llvm/install my_install/mlir; do
      if [ -e "$ROOT_DIR/$dir" ]; then
        printf 'path=%s present\n' "$dir"
      else
        printf 'path=%s missing\n' "$dir"
      fi
    done
    cache_summary cmake_build "$ROOT_DIR/build/CMakeCache.txt"
    cache_summary cmake_build_gpu "$ROOT_DIR/build-gpu/CMakeCache.txt"
  } >"${STATE_DIR}/${profile}.fingerprint"
  printf 'Fingerprint: %s\n' "${STATE_DIR}/${profile}.fingerprint"
}

try_source_ryzen_env() {
  try_source_python_env

  if have aircc && have aiecc && have air-opt; then
    return 0
  fi

  local air_dir="${MLIR_AIR_INSTALL_DIR:-${ROOT_DIR}/install}"
  local aie_dir
  local peano_dir
  local llvm_dir="${LLVM_INSTALL_DIR:-${ROOT_DIR}/my_install/mlir}"
  aie_dir="$(resolve_mlir_aie_dir)"
  peano_dir="$(resolve_peano_dir)"

  if [ -d "$air_dir" ] && [ -d "$aie_dir" ] && [ -d "$peano_dir" ] && [ -d "$llvm_dir" ]; then
    PYTHONPATH="${PYTHONPATH:-}"
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
    # shellcheck source=/dev/null
    source "${ROOT_DIR}/utils/env_setup.sh" "$air_dir" "$aie_dir" "$peano_dir" "$llvm_dir" >/dev/null
  fi
}

try_source_gpu_env() {
  try_source_python_env

  if have aircc && have air-opt && have mlir-opt; then
    return 0
  fi

  local air_dir="${MLIR_AIR_INSTALL_DIR:-${ROOT_DIR}/install-gpu}"
  local llvm_dir
  llvm_dir="$(resolve_llvm_install_dir)"
  if [ -d "$air_dir" ] && [ -d "$llvm_dir" ]; then
    PYTHONPATH="${PYTHONPATH:-}"
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
    # shellcheck source=/dev/null
    source "${ROOT_DIR}/utils/env_setup_gpu.sh" "$air_dir" "$llvm_dir" >/dev/null
  fi
}

env_report() {
  section "Repository"
  printf 'Root:    %s\n' "$ROOT_DIR"
  printf 'Branch:  %s\n' "$(git -C "$ROOT_DIR" branch --show-current 2>/dev/null || printf 'n/a')"
  printf 'HEAD:    %s\n' "$(git_head "$ROOT_DIR")"
  printf 'Dirty:   %s\n' "$(git_dirty "$ROOT_DIR")"

  section "Shell"
  printf 'SHELL:                %s\n' "${SHELL:-unknown}"
  printf 'VIRTUAL_ENV:          %s\n' "${VIRTUAL_ENV:-}"
  printf 'CONDA_PREFIX:         %s\n' "${CONDA_PREFIX:-}"
  printf 'CODEX_SANDBOX:        %s\n' "${CODEX_SANDBOX:-}"
  printf 'MLIR_AIR_INSTALL_DIR: %s\n' "${MLIR_AIR_INSTALL_DIR:-}"
  printf 'MLIR_AIE_INSTALL_DIR: %s\n' "$(resolve_mlir_aie_dir)"
  printf 'PEANO_INSTALL_DIR:    %s\n' "$(resolve_peano_dir)"
  printf 'LLVM_INSTALL_DIR:     %s\n' "$(resolve_llvm_install_dir)"
  printf 'XRT dir:              %s\n' "$(resolve_xrt_dir)"
  printf 'ROCm runtime:         %s\n' "$(rocm_runtime_status)"

  section "Tools"
  printf 'Note: tool paths are reported for the current shell.\n'
  printf '      doctor.sh npu and doctor.sh gpu may source the repo sandbox and AIR setup scripts before running smokes.\n'
  for tool in python3 cmake ninja clang clang++ lld ccache lit air-opt aircc aie-opt aiecc mlir-opt mlir-runner hipcc xrt-smi; do
    tool_line "$tool"
  done

  section "Build Directories"
  for dir in build install build-gpu install-gpu llvm llvm/install my_install my_install/mlir; do
    dir_line "$dir" "$ROOT_DIR/$dir"
  done

  section "CMake Caches"
  cache_summary build "$ROOT_DIR/build/CMakeCache.txt"
  cache_summary build-gpu "$ROOT_DIR/build-gpu/CMakeCache.txt"

  section "Dependency Checkouts"
  for dir in "$ROOT_DIR/../mlir-aie" "$ROOT_DIR/../llvm-aie" "$ROOT_DIR/llvm"; do
    if [ -d "$dir/.git" ]; then
      printf '%s head=%s dirty=%s\n' "$(realpath "$dir")" "$(git_head "$dir")" "$(git_dirty "$dir")"
    else
      printf '%s missing or not a git checkout\n' "$dir"
    fi
  done

  write_fingerprint env
}

add_step() {
  printf ' - %s\n' "$1"
}

setup_plan_ryzen() {
  section "Ryzen/AIE/NPU Setup Plan"
  printf 'Canonical doc: docs/buildingRyzenLin.md\n'
  printf 'Driver doc:    docs/aircc.md\n\n'
  printf 'Required setup gaps:\n'

  local gaps=0
  if ! have python3; then
    add_step "Install or activate Python 3.10+."
    gaps=$((gaps + 1))
  fi
  if ! have ninja; then
    add_step "Install ninja before configuring MLIR-AIR."
    gaps=$((gaps + 1))
  fi
  if ! have cmake; then
    add_step "Install CMake before configuring MLIR-AIR."
    gaps=$((gaps + 1))
  fi

  local peano_dir
  local aie_dir
  peano_dir="$(resolve_peano_dir)"
  aie_dir="$(resolve_mlir_aie_dir)"

  if [ -z "$aie_dir" ] || [ ! -d "$aie_dir" ]; then
    add_step "Install or locate mlir_aie, then source utils/env_setup.sh."
    gaps=$((gaps + 1))
  fi
  if [ -z "$peano_dir" ] || [ ! -x "$peano_dir/bin/clang++" ]; then
    add_step "Install or locate llvm-aie/Peano; PEANO_INSTALL_DIR/bin/clang++ is required."
    gaps=$((gaps + 1))
  fi
  local has_build=0
  if [ ! -f "$ROOT_DIR/build/build.ninja" ]; then
    add_step "Configure a Ryzen/AIE source build with docs/buildingRyzenLin.md or agents/scripts/bootstrap-ryzen-source.sh."
    gaps=$((gaps + 1))
  else
    has_build=1
  fi
  if [ ! -x "$ROOT_DIR/install/bin/air-opt" ] && ! have air-opt; then
    add_step "Build/install AIR or source utils/env_setup.sh so air-opt is on PATH."
    gaps=$((gaps + 1))
  fi
  if ! have aiecc && [ ! -x "$aie_dir/bin/aiecc" ] && [ ! -x "$aie_dir/bin/aiecc.py" ]; then
    add_step "Source the MLIR-AIE environment so aiecc is on PATH."
    gaps=$((gaps + 1))
  fi

  local cache="$ROOT_DIR/build/CMakeCache.txt"
  if [ -f "$cache" ]; then
    local aie_enabled
    aie_enabled="$(cache_value "$cache" AIR_ENABLE_AIE)"
    if [ "$aie_enabled" = "OFF" ]; then
      add_step "Current build has AIR_ENABLE_AIE=OFF; configure a Ryzen/AIE build before NPU work."
      gaps=$((gaps + 1))
    fi
  fi

  local xrt_dir
  xrt_dir="$(resolve_xrt_dir)"
  if [ "$gaps" = "0" ]; then
    printf ' - None detected.\n'
  fi

  printf '\nRecommended next steps:\n'
  local recommendations=0
  if [ "$has_build" = "1" ]; then
    add_step "Existing build found. For MLIR-AIR source changes, prefer: ninja -C build install."
    recommendations=$((recommendations + 1))
  fi
  if [ -z "$xrt_dir" ]; then
    add_step "XRT is not sourced or installed at /opt/xilinx/xrt. This is OK for compile-only checks, but hardware runs need XRT."
    recommendations=$((recommendations + 1))
  fi
  if [ "$gaps" = "0" ]; then
    add_step "Run compile-only NPU smoke: bash agents/scripts/doctor.sh npu"
    recommendations=$((recommendations + 1))
  fi
  if [ "$recommendations" = "0" ]; then
    add_step "Address the required setup gaps above, then rerun this setup plan."
  fi
  write_fingerprint ryzen
}

setup_plan_gpu() {
  section "GPU/ROCDL Setup Plan"
  printf 'Canonical doc: docs/buildingGPU.md\n'
  printf 'Driver doc:    docs/aircc.md\n\n'
  printf 'Required setup gaps:\n'

  local gaps=0
  if ! have python3; then
    add_step "Install or activate Python."
    gaps=$((gaps + 1))
  fi
  if ! have ninja; then
    add_step "Install ninja before configuring MLIR-AIR."
    gaps=$((gaps + 1))
  fi
  if ! have cmake; then
    add_step "Install CMake before configuring MLIR-AIR."
    gaps=$((gaps + 1))
  fi
  local llvm_install_dir
  llvm_install_dir="$(resolve_llvm_install_dir)"
  if [ ! -f "$llvm_install_dir/lib/cmake/mlir/MLIRConfig.cmake" ]; then
    add_step "Build LLVM with utils/build-llvm-local.sh as described in docs/buildingGPU.md."
    gaps=$((gaps + 1))
  fi
  local has_build=0
  if [ ! -f "$ROOT_DIR/build-gpu/build.ninja" ]; then
    add_step "Configure a GPU AIR build with docs/buildingGPU.md or agents/scripts/bootstrap-gpu.sh."
    gaps=$((gaps + 1))
  else
    has_build=1
  fi
  if [ ! -x "$ROOT_DIR/install-gpu/bin/aircc" ] && ! have aircc; then
    add_step "Build/install GPU AIR or source utils/env_setup_gpu.sh so aircc is on PATH."
    gaps=$((gaps + 1))
  fi

  local cache="$ROOT_DIR/build-gpu/CMakeCache.txt"
  if [ -f "$cache" ]; then
    local gpu_enabled
    local aie_enabled
    gpu_enabled="$(cache_value "$cache" AIR_ENABLE_GPU)"
    aie_enabled="$(cache_value "$cache" AIR_ENABLE_AIE)"
    if [ "$gpu_enabled" != "ON" ]; then
      add_step "Current build-gpu cache does not have AIR_ENABLE_GPU=ON."
      gaps=$((gaps + 1))
    fi
    if [ "$aie_enabled" != "OFF" ]; then
      add_step "Current build-gpu cache does not have AIR_ENABLE_AIE=OFF."
      gaps=$((gaps + 1))
    fi
  fi

  if [ "$gaps" = "0" ]; then
    printf ' - None detected.\n'
  fi

  printf '\nRecommended next steps:\n'
  local recommendations=0
  if [ "$has_build" = "1" ]; then
    add_step "Existing GPU build found. For MLIR-AIR source changes, prefer: ninja -C build-gpu install."
    recommendations=$((recommendations + 1))
  fi

  local rocm_status
  rocm_status="$(rocm_runtime_status)"
  case "$rocm_status" in
    missing*)
      add_step "ROCm runtime was not detected. GPU binary generation or runtime checks may fail."
      recommendations=$((recommendations + 1))
      ;;
    *"not found"*)
      add_step "ROCm directory exists, but libamdhip64 was not found. Validate ROCm before runtime checks."
      recommendations=$((recommendations + 1))
      ;;
  esac

  if [ "$gaps" = "0" ]; then
    add_step "Run compile-only GPU smoke: bash agents/scripts/doctor.sh gpu"
    recommendations=$((recommendations + 1))
  fi
  if [ "$recommendations" = "0" ]; then
    add_step "Address the required setup gaps above, then rerun this setup plan."
  fi
  write_fingerprint gpu
}

require_tool() {
  local tool="$1"
  if ! have "$tool"; then
    printf 'ERROR: required tool missing from PATH: %s\n' "$tool" >&2
    return 1
  fi
}

npu_smoke() {
  try_source_ryzen_env

  require_tool python3 || return 1
  require_tool make || return 1
  require_tool aie-opt || return 1

  local peano_dir
  peano_dir="$(resolve_peano_dir)"
  if [ -z "$peano_dir" ] || [ ! -x "$peano_dir/bin/clang++" ]; then
    printf 'ERROR: PEANO_INSTALL_DIR/bin/clang++ is required for the NPU compile-only smoke.\n' >&2
    printf 'Run: bash agents/scripts/doctor.sh setup-plan ryzen\n' >&2
    return 1
  fi

  local tmpdir
  tmpdir="$(mktemp -d /tmp/mlir-air-agent-npu.XXXXXX)"
  local target="${AIE_TARGET:-aie2p}"
  section "NPU Compile-Only Smoke"
  printf 'Build dir: %s\n' "$tmpdir"
  printf 'Target:    %s\n' "$target"

  if PEANO_INSTALL_DIR="$peano_dir" make -C "$ROOT_DIR/programming_examples/matrix_multiplication/i8" run1x1 BUILD_DIR="$tmpdir" COMPILE_MODE=compile-only AIE_TARGET="$target"; then
    printf 'NPU compile-only smoke passed.\n'
    write_fingerprint npu
  else
    printf 'NPU compile-only smoke failed. Artifacts kept at %s\n' "$tmpdir" >&2
    return 1
  fi
}

gpu_smoke() {
  try_source_gpu_env

  require_tool aircc || return 1
  require_tool air-opt || return 1
  require_tool mlir-opt || return 1

  case "$(rocm_runtime_status)" in
    missing*)
      printf 'ERROR: ROCm runtime was not detected; GPU binary generation may not be available.\n' >&2
      printf 'Run: bash agents/scripts/doctor.sh setup-plan gpu\n' >&2
      return 1
      ;;
  esac

  local tmpdir
  tmpdir="$(mktemp -d /tmp/mlir-air-agent-gpu.XXXXXX)"
  local arch="${GPU_ARCH:-gfx942}"
  local input="$ROOT_DIR/test/gpu/simple_test/simple_test.mlir"
  local output="$tmpdir/simple_gpu.mlir"
  section "GPU Compile-Only Smoke"
  printf 'Input:  %s\n' "$input"
  printf 'Output: %s\n' "$output"
  printf 'Arch:   %s\n' "$arch"

  if aircc --target gpu --gpu-arch "$arch" --tmpdir "$tmpdir" -o "$output" "$input"; then
    printf 'GPU compile-only smoke passed.\n'
    write_fingerprint gpu-smoke
  else
    printf 'GPU compile-only smoke failed. Artifacts kept at %s\n' "$tmpdir" >&2
    return 1
  fi
}

main() {
  local command="${1:-}"
  case "$command" in
    env)
      env_report
      ;;
    setup-plan)
      case "${2:-}" in
        ryzen) setup_plan_ryzen ;;
        gpu) setup_plan_gpu ;;
        *) usage; return 2 ;;
      esac
      ;;
    npu)
      npu_smoke
      ;;
    gpu)
      gpu_smoke
      ;;
    -h|--help|help|"")
      usage
      ;;
    *)
      usage
      return 2
      ;;
  esac
}

main "$@"
