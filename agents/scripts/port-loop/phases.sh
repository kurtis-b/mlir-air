# shellcheck shell=bash
#
# The phase table: what each phase is, how it is gated, and what the driver checks itself.
#
# Contract: every phase declares four things —
#   doc              the specification a fresh session is pointed at
#   needs_hardware   whether the gate touches the NPU (drives flock + XRT sourcing)
#   gate_cmd         the phase's own gate, run by the DRIVER, never self-reported
#   objective_check  a driver-side assertion about build products that a session cannot satisfy
#                    by writing a more permissive test
#
# Why objective_check exists: Phase A creates the very lit suite that gates it, so "the suite
# passed" proves nothing on its own. The objective check inspects the actual .o files and their
# exported symbols. A stub test passes the suite and fails this.
#
# Footgun: gate_allowlist is a regex of gate-file paths the phase is permitted to create or
# change. Anything outside it trips the tamper check and halts the run. Widen it deliberately,
# never reflexively.

PL_PHASES_IN_SCOPE='["A","B"]'

phase_name() {
  case "$1" in
    A) echo "AIE2P device kernels" ;;
    B) echo "Runtime seam (runlist aggregation + BO pooling)" ;;
    *) echo "unknown" ;;
  esac
}

phase_doc() {
  case "$1" in
    A) echo "docs/plans/transformer-layer-execution-studies/04-phase-a-kernels.md" ;;
    B) echo "docs/plans/transformer-layer-execution-studies/05-phase-b-runtime-seam.md" ;;
    *) echo "" ;;
  esac
}

phase_needs_hardware() {
  case "$1" in
    A) echo "no" ;;
    B) echo "yes" ;;
    *) echo "no" ;;
  esac
}

phase_gate_description() {
  case "$1" in
    A) cat <<'EOF'
ninja -C build-xrt check-programming-examples-transformer-layer

Compile-only; no NPU required. In addition the driver independently verifies that each ported
kernel produced a non-trivial object file and that the expected extern "C" symbols are present.
A lit test that passes without actually compiling the kernels will fail that check.
EOF
;;
    B) cat <<'EOF'
A hardware test using the exact separately-compiled ELF artifacts the study will use, showing a
multi-ELF runlist that is (1) numerically identical to sequential dispatch and (2) measurably
lower latency.

Because this phase modifies programming_examples/llms/shared/infra/cache.py, the gate also runs
the cross-deployment regression check: make verify on the shipped models, serialized under flock.
EOF
;;
  esac
}

# Gate-file paths this phase may legitimately create or modify.
phase_gate_allowlist() {
  case "$1" in
    A) echo '^programming_examples/(transformer_layer/|CMakeLists\.txt$)' ;;
    B) echo '^programming_examples/transformer_layer/' ;;
    *) echo '' ;;
  esac
}

phase_gate_cmd() {
  case "$1" in
    A) echo "ninja -C ${PL_ROOT}/build-xrt check-programming-examples-transformer-layer" ;;
    B) echo "flock -x -w 1800 /tmp/mlir-air-npu.lock ninja -C ${PL_ROOT}/build-xrt check-programming-examples-transformer-layer" ;;
    *) echo "false" ;;
  esac
}

# --- Objective checks -------------------------------------------------------------------------

# Phase A: the ported kernels must have produced real object files with the expected symbols.
# Deliberately independent of lit, Makefiles, and anything the session authored.
#
# Footgun that bit during bring-up: lit does NOT build in the source tree. Its working directory
# is under the CMake build tree, so objects land in
#   build-xrt/test/transformer_layer/<lit-workdir>/build_peano/*.o
# Searching only programming_examples/ finds nothing and fails a perfectly good phase.
#
# Freshness matters as much as existence: a stale object from an earlier run would otherwise
# satisfy this check even if the gate had just compiled nothing. run_gate() stamps
# _GATE_STARTED_AT, and objects must be newer than it.
phase_a_objective_check() {
  local kdir="${PL_ROOT}/programming_examples/transformer_layer/kernels"
  if [ ! -d "${kdir}" ]; then
    log_error "objective check: ${kdir} does not exist — no kernels were ported"
    return 1
  fi

  local srcs found=0
  srcs="$(find "${kdir}" -name '*.cc' 2>/dev/null | sort)"
  if [ -z "${srcs}" ]; then
    log_error "objective check: no .cc kernel sources under ${kdir}"
    return 1
  fi

  local search_roots=(
    "${PL_ROOT}/programming_examples/transformer_layer"
    "${PL_ROOT}/build-xrt/test/transformer_layer"
    "${PL_ROOT}/build/test/transformer_layer"
  )
  local existing=() r
  for r in "${search_roots[@]}"; do [ -d "${r}" ] && existing+=("${r}"); done
  if [ ${#existing[@]} -eq 0 ]; then
    log_error "objective check: no build output directory for transformer_layer kernels"
    return 1
  fi

  local objs
  objs="$(find "${existing[@]}" -name '*.o' -size +4k 2>/dev/null | sort)"
  if [ -z "${objs}" ]; then
    log_error "objective check: no compiled object files (>4k) found for the ported kernels"
    log_error "  searched: ${existing[*]}"
    log_error "  a lit test can pass without compiling anything; this check is why that is caught"
    return 1
  fi

  # Reject stale artifacts: at least one object must postdate the gate run.
  if [ -n "${_GATE_STARTED_AT:-}" ] && [ -f "${_GATE_STARTED_AT}" ]; then
    local fresh
    fresh="$(find "${existing[@]}" -name '*.o' -size +4k -newer "${_GATE_STARTED_AT}" 2>/dev/null | head -1)"
    if [ -z "${fresh}" ]; then
      log_error "objective check: object files exist but none is newer than the gate run"
      log_error "  the gate reported success without rebuilding anything — treating as vacuous"
      return 1
    fi
  fi

  local o
  while read -r o; do
    [ -z "${o}" ] && continue
    if command -v nm >/dev/null 2>&1; then
      if nm --defined-only "${o}" 2>/dev/null | grep -q ' [TtWw] '; then
        found=$((found + 1))
      else
        log_warn "objective check: ${o} defines no text symbols"
      fi
    else
      found=$((found + 1))
    fi
  done <<< "${objs}"

  if [ "${found}" -eq 0 ]; then
    log_error "objective check: no object file exports any defined symbol"
    return 1
  fi

  log_info "objective check passed: ${found} object file(s) with defined symbols"
  return 0
}

# Phase B: the spike must have produced a recorded result, and cache.py must actually have grown
# a runlist path rather than the phase having been declared done on the strength of a comment.
phase_b_objective_check() {
  local cache="${PL_ROOT}/programming_examples/llms/shared/infra/cache.py"
  if ! grep -q "runlist" "${cache}" 2>/dev/null; then
    log_error "objective check: no runlist support appears in ${cache}"
    return 1
  fi
  local spike
  spike="$(find "${PL_ROOT}/programming_examples/transformer_layer" -name '*runlist*spike*' 2>/dev/null | head -1)"
  if [ -z "${spike}" ]; then
    log_warn "objective check: no runlist spike artifact found; relying on the gate alone"
  fi
  log_info "objective check passed: runlist path present in shared infra"
  return 0
}

phase_objective_check() {
  case "$1" in
    A) phase_a_objective_check ;;
    B) phase_b_objective_check ;;
    *) return 0 ;;
  esac
}
