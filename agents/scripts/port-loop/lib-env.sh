# shellcheck shell=bash
#
# Environment detection, sourcing, and per-step preflight.
#
# Contract: pl_env_ensure() makes aircc/air-opt/aiecc available exactly once per driver process.
# pl_preflight() asserts the repository is in a fit state to run a step and returns non-zero
# (with a stated reason) rather than letting a step start on a bad footing.
#
# Footguns:
#   - utils/env_setup.sh must be SOURCED (it uses bare `return`) and is NOT idempotent: it
#     re-prepends to PATH/PYTHONPATH/LD_LIBRARY_PATH on every source. We therefore gate on tool
#     presence, never re-source. This is doctor.sh's idiom.
#   - env_setup.sh dereferences $PYTHONPATH and $LD_LIBRARY_PATH unguarded, so under `set -u`
#     they must be pre-seeded. doctor.sh does the same.
#   - PEANO_INSTALL_DIR must be set consistently for the whole run: its presence flips example
#     Makefiles between build_peano/ and build_chess/, and an inconsistent setting silently
#     builds into two trees.
#   - We use install-xrt (not install): only build-xrt carries XRT_LIB_DIR, and Phase B runs on
#     hardware.

pl_have() { command -v "$1" >/dev/null 2>&1; }

pl_env_ensure() {
  if pl_have aircc && pl_have air-opt && pl_have aiecc; then
    return 0
  fi

  export PYTHONPATH="${PYTHONPATH:-}"
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

  if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "${PL_ROOT}/sandbox/bin/activate" ]; then
    # shellcheck disable=SC1091
    . "${PL_ROOT}/sandbox/bin/activate"
  fi

  local aie peano llvm air
  air="${PL_ROOT}/install-xrt"
  aie="${PL_ROOT}/sandbox/lib/python3.12/site-packages/mlir_aie"
  peano="${PL_ROOT}/sandbox/lib/python3.12/site-packages/llvm-aie"
  llvm="${PL_ROOT}/my_install/mlir"

  local d
  for d in "${air}" "${aie}" "${peano}" "${llvm}"; do
    if [ ! -d "${d}" ]; then
      log_error "toolchain directory missing: ${d}"
      return 1
    fi
  done

  # shellcheck disable=SC1091
  . "${PL_ROOT}/utils/env_setup.sh" "${air}" "${aie}" "${peano}" "${llvm}"
  export PEANO_INSTALL_DIR="${peano}"

  if ! pl_have aircc || ! pl_have air-opt; then
    log_error "env_setup.sh sourced but aircc/air-opt still not on PATH"
    return 1
  fi
  return 0
}

pl_env_ensure_xrt() {
  [ -n "${XILINX_XRT:-}" ] && return 0
  if [ -f /opt/xilinx/xrt/setup.sh ]; then
    # shellcheck disable=SC1091
    . /opt/xilinx/xrt/setup.sh >/dev/null 2>&1
  fi
  [ -n "${XILINX_XRT:-}" ] || { log_error "XRT setup.sh did not export XILINX_XRT"; return 1; }
  return 0
}

# Modified TRACKED files only. Untracked directories (llvm/, my_install/, air_project/) are
# local toolchain and scratch state; they are expected and must not block a run.
pl_tracked_dirty() {
  git -C "${PL_ROOT}" status --porcelain --untracked-files=no
}

pl_preflight() {
  local needs_hw="${1:-no}"

  local branch
  branch="$(git -C "${PL_ROOT}" rev-parse --abbrev-ref HEAD)"
  if [ "${branch}" != "${PL_BRANCH}" ]; then
    log_error "on branch '${branch}', expected '${PL_BRANCH}'"
    return 1
  fi

  if [ -n "$(pl_tracked_dirty)" ]; then
    log_error "tracked files are modified; commit or stash before starting a step:"
    pl_tracked_dirty >&2
    return 1
  fi

  pl_env_ensure || return 1

  if [ "${needs_hw}" = "yes" ]; then
    pl_env_ensure_xrt || return 1
    if [ ! -e /dev/accel/accel0 ]; then
      log_error "no NPU at /dev/accel/accel0"
      return 1
    fi
    pl_assert_lit_sees_npu || return 1
  fi

  local free_gb
  free_gb="$(df -BG --output=avail "${PL_ROOT}" | tail -1 | tr -dc '0-9')"
  if [ "${free_gb:-0}" -lt 20 ]; then
    log_error "only ${free_gb}G free on the repo filesystem; need at least 20G"
    return 1
  fi

  return 0
}

# A hardware gate that runs zero hardware tests is worse than a gate that fails: it reports
# success. lit decides whether the NPU exists by running `${config.xrt_bin_dir}/xrt-smi`, and
# that path is baked into the GENERATED lit.site.cfg.py at configure time. It goes stale: on this
# machine it was left pointing at /usr/lib/bin (which does not exist) while xrt-smi lives in
# /usr/bin, so the ryzen_ai_npu2 feature was never set and 100% of NPU tests reported UNSUPPORTED
# while the suite still exited 0.
#
# Refuse to start a hardware phase in that state. Fix by reconfiguring so the site config
# regenerates with a correct XRT path, e.g.
#   cmake -S . -B build-xrt -DXRT_DIR=/opt/xilinx/xrt
# then re-check with: lit -sv --show-unsupported build-xrt/programming_examples/flash_attention
pl_assert_lit_sees_npu() {
  local site="${PL_ROOT}/build-xrt/programming_examples/lit.site.cfg.py"
  if [ ! -f "${site}" ]; then
    log_error "lit site config not found at ${site}"
    return 1
  fi
  local bindir
  bindir="$(sed -n 's/^config\.xrt_bin_dir = "\(.*\)"$/\1/p' "${site}" | head -1)"
  if [ -z "${bindir}" ]; then
    log_error "could not read config.xrt_bin_dir from ${site}"
    return 1
  fi
  if [ ! -x "${bindir}/xrt-smi" ]; then
    log_error "lit will not detect the NPU: config.xrt_bin_dir='${bindir}' has no xrt-smi"
    log_error "  actual xrt-smi: $(command -v xrt-smi || echo 'not on PATH')"
    log_error "  every NPU-gated test would report UNSUPPORTED and the suite would still exit 0"
    log_error "  fix: cmake -S ${PL_ROOT} -B ${PL_ROOT}/build-xrt -DXRT_DIR=/opt/xilinx/xrt"
    return 1
  fi
  log_info "lit NPU detection OK (xrt_bin_dir=${bindir})"
  return 0
}

# The pre-push fence. Installed for the duration of a run so an agent cannot publish work.
pl_install_push_fence() {
  local hook="${PL_ROOT}/.git/hooks/pre-push"
  if [ -f "${hook}" ] && ! grep -q "port-loop fence" "${hook}" 2>/dev/null; then
    log_warn "an existing pre-push hook is present; leaving it alone and not fencing"
    return 0
  fi
  cat > "${hook}" <<'FENCE'
#!/bin/sh
# port-loop fence: an unattended run must not publish. Remove this file to push again.
echo "pre-push blocked: agents/scripts/port-loop.sh is running (or did not clean up)." >&2
echo "Remove .git/hooks/pre-push to push manually." >&2
exit 1
FENCE
  chmod +x "${hook}"
}

pl_remove_push_fence() {
  local hook="${PL_ROOT}/.git/hooks/pre-push"
  if [ -f "${hook}" ] && grep -q "port-loop fence" "${hook}" 2>/dev/null; then
    rm -f "${hook}"
  fi
}
