#!/usr/bin/env bash
#
# port-loop: unattended driver for the transformer-layer port plan.
#
# Works through the phases defined in docs/plans/transformer-layer-execution-studies/, running
# each in a FRESH claude -p session, gating each with three sequential Codex review->fix rounds,
# and running every gate itself rather than trusting a session's self-report.
#
# Scope is whatever PL_PHASES_IN_SCOPE in port-loop/phases.sh lists, then halt. Phases A, B and
# C1-C4 are done; the current scope is Phase D's two sub-phases D1 and D2.
#
# Usage:
#   agents/scripts/port-loop.sh help
#   agents/scripts/port-loop.sh status
#   agents/scripts/port-loop.sh dry-run
#   agents/scripts/port-loop.sh start
#   agents/scripts/port-loop.sh resume
#   agents/scripts/port-loop.sh resume-at <phase> <step> [round] [base-sha]
#   agents/scripts/port-loop.sh stop
#   agents/scripts/port-loop.sh run-one <phase> <step> [arg]
#
# run-one steps: preflight | implement | review | gate | objective-check | hardware-check |
#                tamper-check | fingerprint | prompt
# hardware-check takes an optional gate-log path, so the assertion can be exercised against a
# known-good and a known-bad log before a run depends on it. The confirm review has no run-one
# arm: it needs the pre-fix SHA, which only run_phase knows.
#
# Options are environment variables, matching doctor.sh's style:
#   PL_STEP_TIMEOUT     seconds per claude session          (default 10800 = 3h)
#   PL_REVIEW_TIMEOUT   seconds per codex review            (default 3600 = 1h)
#   PL_STEP_BUDGET      USD cap per claude session          (default 25)
#   PL_MAX_TURNS        agentic turns per claude session    (default 300)
#   PL_MAX_INVOCATIONS  total agent invocations for the run (default 40)
#   PL_MAX_HOURS        wall-clock deadline in hours        (default 24)
#   PL_MODEL            claude model                        (default claude-fable-5)
#   PL_CODEX_MODEL      codex model            (default: from ~/.codex/config.toml)
#   PL_CODEX_EFFORT     codex reasoning effort (default: from ~/.codex/config.toml)
#   PL_DRY_RUN          yes|no                              (default no)
#   PL_API_RETRIES      retries on a transient API failure  (default 10)
#   PL_API_RETRY_DELAY  seconds, multiplied by attempt no.  (default 60)
#   PL_API_RETRY_MAX_DELAY  backoff ceiling in seconds      (default 600)
#
# Exit codes: 0 success, 1 operational failure, 2 usage error.
#
# Footgun: this driver must NOT be started with sudo, and must not be run concurrently with
# itself. A run lock enforces the latter.

set -u

PL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PL_LIB="${PL_ROOT}/agents/scripts/port-loop"
PL_STATE_DIR="${PL_ROOT}/agents/.state/port-loop"
PL_BRANCH="${PL_BRANCH:-exper/transformer-layer-execution-studies}"

PL_STEP_TIMEOUT="${PL_STEP_TIMEOUT:-10800}"
PL_REVIEW_TIMEOUT="${PL_REVIEW_TIMEOUT:-3600}"
PL_STEP_BUDGET="${PL_STEP_BUDGET:-25}"
PL_MAX_TURNS="${PL_MAX_TURNS:-300}"
PL_MAX_INVOCATIONS="${PL_MAX_INVOCATIONS:-40}"
PL_MAX_HOURS="${PL_MAX_HOURS:-24}"
PL_MODEL="${PL_MODEL:-claude-fable-5}"
PL_CODEX_MODEL="${PL_CODEX_MODEL:-}"
PL_CODEX_EFFORT="${PL_CODEX_EFFORT:-}"
PL_DRY_RUN="${PL_DRY_RUN:-no}"
PL_REVIEW_ROUNDS=3

log_info()  { printf '[%s] %s\n'   "$(date '+%H:%M:%S')" "$*"; }
log_warn()  { printf '[%s] WARN: %s\n'  "$(date '+%H:%M:%S')" "$*" >&2; }
log_error() { printf '[%s] ERROR: %s\n' "$(date '+%H:%M:%S')" "$*" >&2; }

# shellcheck source=agents/scripts/port-loop/lib-state.sh
. "${PL_LIB}/lib-state.sh"
# shellcheck source=agents/scripts/port-loop/lib-env.sh
. "${PL_LIB}/lib-env.sh"
# shellcheck source=agents/scripts/port-loop/lib-guard.sh
. "${PL_LIB}/lib-guard.sh"
# shellcheck source=agents/scripts/port-loop/lib-claude.sh
. "${PL_LIB}/lib-claude.sh"
# shellcheck source=agents/scripts/port-loop/lib-codex.sh
. "${PL_LIB}/lib-codex.sh"
# shellcheck source=agents/scripts/port-loop/lib-docs.sh
. "${PL_LIB}/lib-docs.sh"
# shellcheck source=agents/scripts/port-loop/phases.sh
. "${PL_LIB}/phases.sh"

usage() {
  # Range ends at the "Exit codes" line. Extending the header without moving this silently
  # truncates --help, which is how PL_MODEL and PL_DRY_RUN once vanished from it.
  sed -n '3,42p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

require_deps() {
  local missing="" t
  for t in jq git flock timeout claude codex ninja; do
    command -v "$t" >/dev/null 2>&1 || missing="${missing} ${t}"
  done
  if [ -n "${missing}" ]; then
    log_error "missing required tools:${missing}"
    return 1
  fi
  return 0
}

# Ordinal of each stage within a phase, used only for resume. implement=0, then review/fix pairs,
# then the gate. A resumed phase skips every stage with a lower ordinal than where it stopped.
stage_ordinal() {
  local step="$1" round="${2:-0}"
  case "${step}" in
    preflight|implement) echo 0 ;;
    review)              echo $(( 1 + (round - 1) * 2 )) ;;
    fix)                 echo $(( 2 + (round - 1) * 2 )) ;;
    # confirm sits after the last fix and before the gate. It shares the gate's ordinal because
    # resuming at either means every review/fix round is behind you; the confirm review itself is
    # skipped on a resume, since _FIX_BASE_SHA cannot be reconstructed after the fact.
    confirm|gate|objective-check|tamper-check) echo 7 ;;
    *)                   echo 0 ;;
  esac
}

phase_dir() { echo "${PL_STATE_DIR}/phase-$1"; }

# Render a prompt template, substituting the __PLACEHOLDER__ tokens.
# Substitutions are passed through the environment rather than interpolated into the script, so
# that gate descriptions and JSON findings containing quotes, newlines or backslashes cannot
# break the renderer.
render_prompt() {
  local template="$1" out="$2" phase="$3" round="${4:-0}" review_json="${5:-}"

  local blocking="[]" nonblocking="[]"
  if [ -n "${review_json}" ] && [ -f "${review_json}" ]; then
    blocking="$(jq -c '.blocking // []' "${review_json}")"
    nonblocking="$(jq -c '.non_blocking // []' "${review_json}")"
  fi

  PL_T_BRANCH="${PL_BRANCH}" \
  PL_T_PHASE_ID="${phase}" \
  PL_T_PHASE_NAME="$(phase_name "${phase}")" \
  PL_T_PHASE_DOC="${_PHASE_DOC}" \
  PL_T_ROUND="${round}" \
  PL_T_START_SHA="${_START_SHA:-HEAD}" \
  PL_T_GATE="$(phase_gate_description "${phase}")" \
  PL_T_BLOCKING="${blocking}" \
  PL_T_NONBLOCKING="${nonblocking}" \
  PL_T_FIX_BASE="${_FIX_BASE_SHA:-}" \
  python3 -c '
import os, pathlib, sys
tpl = pathlib.Path(sys.argv[1]).read_text()
for token, var in (
    ("__BRANCH__",            "PL_T_BRANCH"),
    ("__PHASE_ID__",          "PL_T_PHASE_ID"),
    ("__PHASE_NAME__",        "PL_T_PHASE_NAME"),
    ("__PHASE_DOC__",         "PL_T_PHASE_DOC"),
    ("__ROUND__",             "PL_T_ROUND"),
    ("__PHASE_START_SHA__",   "PL_T_START_SHA"),
    ("__GATE_DESCRIPTION__",  "PL_T_GATE"),
    ("__BLOCKING_JSON__",     "PL_T_BLOCKING"),
    ("__NON_BLOCKING_JSON__", "PL_T_NONBLOCKING"),
    ("__FIX_BASE_SHA__",      "PL_T_FIX_BASE"),
):
    tpl = tpl.replace(token, os.environ.get(var, ""))
if sys.argv[2] == "-":
    sys.stdout.write(tpl)
else:
    pathlib.Path(sys.argv[2]).write_text(tpl)
' "${template}" "${out}"
}

commit_step() {
  local phase="$1" label="$2"
  if [ -z "$(pl_tracked_dirty)" ] && [ -z "$(git -C "${PL_ROOT}" status --porcelain | grep -v '^?? llvm/\|^?? my_install/\|^?? air_project/')" ]; then
    log_info "nothing to commit after ${label}"
    return 0
  fi
  git -C "${PL_ROOT}" add -A -- ':!llvm' ':!my_install' ':!air_project' >/dev/null 2>&1
  if git -C "${PL_ROOT}" diff --cached --quiet; then
    log_info "nothing staged after ${label}"
    return 0
  fi
  PRE_COMMIT_ALLOW_NO_CONFIG=1 git -C "${PL_ROOT}" commit -q -m "port-loop: phase ${phase} ${label}

Automated step from agents/scripts/port-loop.sh. Reviewed by Codex before the phase gate.
" || { log_error "commit failed after ${label}"; return 1; }
  log_info "committed after ${label}: $(git -C "${PL_ROOT}" rev-parse --short HEAD)"
  return 0
}

run_gate() {
  local phase="$1" log="$2"
  local cmd
  cmd="$(phase_gate_cmd "${phase}")"
  log_info "running gate: ${cmd}"
  if [ "${PL_DRY_RUN}" = "yes" ]; then
    log_info "DRY RUN: would run gate"
    return 0
  fi
  # Timestamp reference so the objective check can reject artifacts the gate did not rebuild.
  _GATE_STARTED_AT="$(dirname "${log}")/.gate-started"
  : > "${_GATE_STARTED_AT}"
  if ( cd "${PL_ROOT}" && eval "${cmd}" ) > "${log}" 2>&1; then
    log_info "gate PASSED"
    return 0
  fi
  log_error "gate FAILED; last 30 lines of ${log}:"
  tail -30 "${log}" >&2
  return 1
}

# --- One complete phase -----------------------------------------------------------------------

run_phase() {
  local phase="$1"
  local pdir; pdir="$(phase_dir "${phase}")"
  mkdir -p "${pdir}"

  local needs_hw; needs_hw="$(phase_needs_hardware "${phase}")"
  _PHASE_DOC="$(phase_doc "${phase}")"

  # Per-phase, not per-run: a stale value from the previous phase would point the confirm review
  # at that phase's diff. A mid-phase resume leaves these empty and the confirm review is skipped,
  # which is logged rather than silent.
  _FIX_BASE_SHA=""
  _FIX_BASE_ROUND=""

  log_info "EVENT: phase-start ${phase} — $(phase_name "${phase}")"
  state_set --arg p "${phase}" '.step = "preflight" | .round = 0'
  state_log "phase_start" "$(phase_name "${phase}")"

  local phase_epoch; phase_epoch="$(date +%s)"
  timing_start "${phase}"

  pl_preflight "${needs_hw}" || { state_halt "preflight failed for phase ${phase}"; return 1; }

  # Resume support. Without this a resumed run re-executes `implement`, throwing away work that
  # is already committed and burning a full session to do it. _RESUME_FROM is the ordinal of the
  # first stage still to run; stages before it are skipped.
  local resume_from=0 resuming=no
  if [ "$(state_get '.resume_phase // ""')" = "${phase}" ]; then
    resuming=yes
    resume_from="$(stage_ordinal "$(state_get '.resume_step // "implement"')" "$(state_get '.resume_round // 1')")"
    state_set '.resume_phase = null | .resume_step = null | .resume_round = null'
    log_info "resuming phase ${phase} at stage ordinal ${resume_from}"
  fi

  # A FRESH phase takes its base from HEAD. A RESUMED one keeps the base already recorded, at every
  # ordinal including zero.
  #
  # `[2026-08-05]` The `-eq 0` test alone was not enough, and Phase E1 hit it. stage_ordinal maps
  # both `preflight` and `implement` to 0, so a halt DURING implement -- E1's session exhausted
  # PL_STEP_BUDGET after committing six commits -- is indistinguishable from a phase that never
  # started. Re-deriving the base from HEAD then takes it from a tree that already contains the
  # halted attempt's work, and three things go wrong at once: the three Codex reviews diff an empty
  # range, guard_fingerprint's baseline already includes the phase's own gate-file edits so the
  # tamper check is vacuous, and the implement session re-runs over committed work. That is the
  # same defect as the working-tree fingerprint baseline, reached by a third road.
  if [ "${resume_from}" -eq 0 ] && [ "${resuming}" = "no" ]; then
    _START_SHA="$(git -C "${PL_ROOT}" rev-parse HEAD)"
    state_set --arg s "${_START_SHA}" '.phase_start_sha = $s'
  else
    _START_SHA="$(state_start_sha)"
    [ -n "${_START_SHA}" ] || _START_SHA="$(git -C "${PL_ROOT}" rev-parse HEAD)"
  fi

  # Re-running implement over committed work is wasteful but not unsafe -- the session sees its own
  # commits and can continue. Say so, and name the cheaper route, because the operator is reading
  # this after a halt and `resume` is what the halt message told them to run.
  if [ "${resuming}" = "yes" ] && [ "${resume_from}" -eq 0 ] \
     && [ "$(git -C "${PL_ROOT}" rev-parse HEAD)" != "$(git -C "${PL_ROOT}" rev-parse "${_START_SHA}")" ]; then
    log_warn "resuming at implement, but commits already exist since the phase base."
    log_warn "  The implement session will RE-RUN over them. If its work items are already done,"
    log_warn "  stop and skip to the reviews instead:"
    log_warn "    ./agents/scripts/port-loop.sh resume-at ${phase} review 1 $(git -C "${PL_ROOT}" rev-parse --short "${_START_SHA}")"
  fi
  log_info "phase base commit: $(git -C "${PL_ROOT}" rev-parse --short "${_START_SHA}")"

  # Always re-derive the baseline from the phase's BASE COMMIT, never the working tree and never
  # a cached file. Reusing a stale or working-tree baseline is exactly how a weakened gate
  # becomes the reference it is checked against.
  guard_fingerprint "${pdir}/gates.before" "${_START_SHA}"

  # --- implement ---
  if [ "${resume_from}" -le "$(stage_ordinal implement 0)" ]; then
    state_set '.step = "implement"'
    render_prompt "${PL_LIB}/prompts/implement.md" "${pdir}/implement.prompt" "${phase}"
    if ! pl_claude_run "${pdir}/implement.prompt" "${PL_LIB}/schema/session.json" "${pdir}" "implement"; then
      state_halt "implement session failed in phase ${phase}"
      timing_end "${phase}" "halted-implement" "${phase_epoch}"
      return 1
    fi
    commit_step "${phase}" "implement" || { state_halt "commit failed"; return 1; }
  else
    log_info "skipping implement (already done; resuming later in the phase)"
  fi

  # --- three sequential review -> fix rounds ---
  local round
  for round in $(seq 1 "${PL_REVIEW_ROUNDS}"); do
    if [ "${resume_from}" -gt "$(stage_ordinal review "${round}")" ] \
       && [ "${resume_from}" -gt "$(stage_ordinal fix "${round}")" ]; then
      log_info "skipping review/fix round ${round} (already done)"
      continue
    fi
    state_set --argjson r "${round}" '.step = "review" | .round = $r'
    local rdir="${pdir}/round-${round}"
    mkdir -p "${rdir}"

    render_prompt "${PL_LIB}/prompts/review.md" "${rdir}/review.prompt" "${phase}" "${round}"
    if ! pl_codex_review "${_START_SHA}" "${rdir}/review.prompt" "${rdir}/review.json" "${rdir}/codex.log"; then
      state_halt "codex review round ${round} failed in phase ${phase}"
      timing_end "${phase}" "halted-review${round}" "${phase_epoch}"
      return 1
    fi

    # A weakened gate halts unconditionally, whatever the verdict says.
    if ! guard_check_weakened "${rdir}/review.json"; then
      state_halt "phase ${phase} round ${round}: Codex reported weakened gates"
      timing_end "${phase}" "halted-weakened-gate" "${phase_epoch}"
      return 1
    fi

    local nblock; nblock="$(pl_codex_blocking_count "${rdir}/review.json")"
    state_log "review" "round ${round}: verdict=$(pl_codex_verdict "${rdir}/review.json") blocking=${nblock}"

    if [ "${nblock}" -eq 0 ]; then
      log_info "round ${round}: no blocking findings, fix step is a no-op"
      continue
    fi

    state_set --argjson r "${round}" '.step = "fix" | .round = $r'
    render_prompt "${PL_LIB}/prompts/fix.md" "${rdir}/fix.prompt" "${phase}" "${round}" "${rdir}/review.json"
    # Remember where this fix started. The last one to run is the diff nothing else will look at.
    _FIX_BASE_SHA="$(git -C "${PL_ROOT}" rev-parse HEAD)"
    _FIX_BASE_ROUND="${round}"
    if ! pl_claude_run "${rdir}/fix.prompt" "${PL_LIB}/schema/session.json" "${rdir}" "fix"; then
      state_halt "fix session round ${round} failed in phase ${phase}"
      timing_end "${phase}" "halted-fix${round}" "${phase_epoch}"
      return 1
    fi
    commit_step "${phase}" "review round ${round} fixes" || { state_halt "commit failed"; return 1; }
  done

  # After the final round there must be no blocking findings left.
  local final_json="${pdir}/round-${PL_REVIEW_ROUNDS}/review.json"
  if [ -f "${final_json}" ] && [ "$(pl_codex_blocking_count "${final_json}")" -gt 0 ]; then
    log_warn "blocking findings remained after round ${PL_REVIEW_ROUNDS}; the gate now decides"
  fi

  # --- confirm review: the one diff nothing else reviews -------------------------------------
  #
  # The rounds are review -> fix, so the LAST fix to run is never itself reviewed. C4 hit exactly
  # this: its round-3 review raised two blocking findings, both were fixed, and the only thing that
  # checked those fixes was a human afterwards. This closes that, narrowly -- it reviews the final
  # fix's diff alone and asks only whether the fix is correct and complete.
  #
  # It halts on blocking findings, and does so BEFORE the gate, which is the cheapest place in the
  # phase to stop: no hardware time has been spent yet. The prompt is deliberately scoped to keep
  # that rare -- style and scope observations are non-blocking by instruction.
  # Only the FINAL round's fix is unreviewed. If the last fix ran in round 1 or 2, a later review
  # round already covered that diff and a confirm review would be a wasted codex invocation
  # against PL_MAX_INVOCATIONS.
  if [ -n "${_FIX_BASE_SHA:-}" ] \
     && [ "${_FIX_BASE_ROUND:-0}" = "${PL_REVIEW_ROUNDS}" ] \
     && [ "$(git -C "${PL_ROOT}" rev-parse HEAD)" != "${_FIX_BASE_SHA}" ]; then
    state_set '.step = "confirm"'
    local cdir="${pdir}/confirm"
    mkdir -p "${cdir}"
    log_info "confirm review over round ${_FIX_BASE_ROUND}'s fix (${_FIX_BASE_SHA}..HEAD)"
    render_prompt "${PL_LIB}/prompts/confirm.md" "${cdir}/confirm.prompt" "${phase}" \
                  "${_FIX_BASE_ROUND}" "${pdir}/round-${_FIX_BASE_ROUND}/review.json"
    if ! pl_codex_review "${_FIX_BASE_SHA}" "${cdir}/confirm.prompt" \
                         "${cdir}/review.json" "${cdir}/codex.log"; then
      state_halt "confirm review failed in phase ${phase}"
      timing_end "${phase}" "halted-confirm" "${phase_epoch}"
      return 1
    fi
    if ! guard_check_weakened "${cdir}/review.json"; then
      state_halt "phase ${phase} confirm review: the final fix weakened a gate"
      timing_end "${phase}" "halted-weakened-gate" "${phase_epoch}"
      return 1
    fi
    local cblock; cblock="$(pl_codex_blocking_count "${cdir}/review.json")"
    state_log "confirm" "confirm review of round ${_FIX_BASE_ROUND} fix: verdict=$(pl_codex_verdict "${cdir}/review.json") blocking=${cblock}"
    if [ "${cblock}" -gt 0 ]; then
      log_error "confirm review found ${cblock} blocking finding(s) in the final fix — the diff no"
      log_error "  other round would have seen. Halting before the gate; see ${cdir}/review.json"
      jq -r '.blocking[] | "  \(.file // "?"): \(.title // .summary // .finding // "?")"' \
        "${cdir}/review.json" >&2 2>/dev/null || true
      state_halt "phase ${phase} confirm review found blocking findings in the final fix"
      timing_end "${phase}" "halted-confirm-blocking" "${phase_epoch}"
      return 1
    fi
    log_info "confirm review clean: the final fix holds up"
  elif [ "${resume_from}" -gt 0 ] && [ -z "${_FIX_BASE_SHA:-}" ]; then
    # Do not claim no fix ran. On a mid-phase resume the rounds were skipped, so the pre-fix SHA
    # was never captured and the confirm review cannot be reconstructed -- which is a gap in the
    # resumed run, not a clean bill of health.
    log_warn "resumed mid-phase: the final fix's diff cannot be reconstructed, so the confirm"
    log_warn "  review is SKIPPED. Review round ${PL_REVIEW_ROUNDS}'s fix by hand before trusting this phase."
  else
    log_info "no fix ran in the final round; confirm review not needed"
  fi

  # --- gate ---
  state_set '.step = "gate"'
  if ! run_gate "${phase}" "${pdir}/gate.log"; then
    state_halt "phase ${phase} gate failed"
    timing_end "${phase}" "halted-gate" "${phase_epoch}"
    return 1
  fi

  # A green gate is not the same claim as a gate that ran what it says it runs. Phase B's passed
  # in 16 seconds without touching the NPU; see pl_assert_gate_ran_hardware.
  if ! pl_assert_gate_ran_hardware "${phase}" "${pdir}/gate.log"; then
    state_halt "phase ${phase} passed a gate that never exercised the hardware it claims"
    timing_end "${phase}" "halted-no-hardware" "${phase_epoch}"
    return 1
  fi

  # --- driver-side checks the session cannot influence ---
  state_set '.step = "objective-check"'
  if ! phase_objective_check "${phase}"; then
    state_halt "phase ${phase} passed its gate but failed the driver's objective check"
    timing_end "${phase}" "halted-objective" "${phase_epoch}"
    return 1
  fi

  state_set '.step = "tamper-check"'
  if ! guard_check_tamper "${pdir}/gates.before" "$(phase_gate_allowlist "${phase}")"; then
    state_halt "phase ${phase} modified gate-defining files outside its allowlist"
    timing_end "${phase}" "halted-tamper" "${phase_epoch}"
    return 1
  fi
  if ! guard_check_destructive "${_START_SHA}"; then
    state_halt "phase ${phase} deleted tracked files"
    timing_end "${phase}" "halted-destructive" "${phase_epoch}"
    return 1
  fi

  log_info "EVENT: phase-complete ${phase} — gate, objective and tamper checks all passed"
  state_log "phase_complete" "phase ${phase} passed gate, objective and tamper checks"
  timing_end "${phase}" "passed" "${phase_epoch}"

  # The status board, from what the driver measured. Deliberately after the tamper and destructive
  # checks (so the driver's own edit is never in the diff they inspect) and committed immediately
  # (pl_preflight refuses to start the next phase on a dirty tracked worktree).
  docs_update_status_board "${phase}" "$(( ($(date +%s) - phase_epoch) / 60 ))" "passed"
  docs_commit_status_board "${phase}"
  log_info "=== Phase ${phase} COMPLETE ==="
  return 0
}

# --- Subcommands ------------------------------------------------------------------------------

cmd_start() {
  require_deps || return 1
  if state_exists && [ "$(state_status)" = "running" ]; then
    log_error "a run is already in progress; use 'status' or 'stop'"
    return 1
  fi
  local deadline=$(( $(date +%s) + PL_MAX_HOURS * 3600 ))
  state_init "${PL_PHASES_IN_SCOPE}" "${deadline}"
  cmd_loop
}

# Position the state machine at a specific phase/step without running anything, so an operator
# can resume a phase whose earlier stages were completed by hand.
cmd_resume_at() {
  local phase="${1:-}" step="${2:-}" round="${3:-1}" base="${4:-}"
  [ -n "${phase}" ] && [ -n "${step}" ] || { usage; return 2; }
  state_exists || state_init "${PL_PHASES_IN_SCOPE}" "$(( $(date +%s) + PL_MAX_HOURS * 3600 ))"
  local idx
  idx="$(state_get --arg p "${phase}" '.phases | index($p) // 0' 2>/dev/null || echo 0)"
  [ -n "${base}" ] || base="$(state_start_sha)"
  [ -n "${base}" ] || base="$(git -C "${PL_ROOT}" rev-parse HEAD)"
  state_set --argjson i "${idx}" --arg p "${phase}" --arg s "${step}" \
            --argjson r "${round}" --arg b "${base}" \
    '.phase_index = $i | .resume_phase = $p | .resume_step = $s | .resume_round = $r
     | .phase_start_sha = $b | .status = "ready" | .halt_reason = null'
  log_info "state positioned at phase ${phase}, step ${step}, round ${round}, base $(git -C "${PL_ROOT}" rev-parse --short "${base}")"
  log_info "run 'resume' to continue from here"
}

cmd_resume() {
  require_deps || return 1
  state_exists || { log_error "no state to resume; run 'start' first"; return 1; }
  log_info "resuming from phase $(state_phase) step $(state_step)"
  state_set '.status = "running" | .halt_reason = null'
  cmd_loop
}

cmd_loop() {
  pl_install_push_fence
  trap 'pl_remove_push_fence' EXIT

  state_set '.status = "running"'
  local total idx phase
  total="$(state_get '.phases | length')"

  while :; do
    idx="$(state_get '.phase_index')"
    if [ "${idx}" -ge "${total}" ]; then
      state_done
      log_info "all phases in scope are complete"
      break
    fi
    phase="$(state_phase)"
    if ! run_phase "${phase}"; then
      log_error "run halted during phase ${phase}"
      log_error "read ${STATUS_MD} then resume with: agents/scripts/port-loop.sh resume"
      return 1
    fi
    state_set '.phase_index += 1 | .step = "preflight" | .round = 0'
  done

  local scope_list
  scope_list="$(printf '%s' "${PL_PHASES_IN_SCOPE}" | jq -r 'join(", ")' 2>/dev/null)"
  [ -n "${scope_list}" ] || scope_list="${PL_PHASES_IN_SCOPE}"
  log_info "Scope was ${scope_list}. Review the work before extending scope."
  return 0
}

cmd_status() {
  if ! state_exists; then
    echo "No run has been started. Use: agents/scripts/port-loop.sh start"
    return 0
  fi
  cat "${STATUS_MD}"
}

cmd_stop() {
  state_exists || { log_error "no state to stop"; return 1; }
  state_halt "stopped by operator"
  pl_remove_push_fence
  log_info "stopped; resume with: agents/scripts/port-loop.sh resume"
}

cmd_run_one() {
  local phase="${1:-}" step="${2:-}"
  [ -n "${phase}" ] && [ -n "${step}" ] || { usage; return 2; }
  require_deps || return 1
  state_exists || state_init "${PL_PHASES_IN_SCOPE}" "$(( $(date +%s) + PL_MAX_HOURS * 3600 ))"

  local pdir; pdir="$(phase_dir "${phase}")"
  mkdir -p "${pdir}"
  _PHASE_DOC="$(phase_doc "${phase}")"
  _START_SHA="$(state_start_sha)"
  [ -n "${_START_SHA}" ] || _START_SHA="$(git -C "${PL_ROOT}" rev-parse HEAD)"

  case "${step}" in
    preflight)       pl_preflight "$(phase_needs_hardware "${phase}")" && log_info "preflight OK" ;;
    gate)            run_gate "${phase}" "${pdir}/gate.log" ;;
    objective-check)
      # run_gate sets this in-process; for a standalone run-one invocation, point at the
      # stamp the last gate left behind so freshness can still be proven.
      _GATE_STARTED_AT="${pdir}/.gate-started"
      # An objective check may import from the repository -- Phase E's resolves GEMM specs through
      # shared.builders.gemm_builder to prove the symbol names separate. In a real run pl_preflight
      # has already sourced the venv; standalone it has not, and the import then fails for an
      # environment reason while the check reports a substantive one. Source it here so a manual
      # invocation behaves like the real thing.
      pl_env_ensure || return 1
      # And XRT, for the same reason one step further out. Phase H's objective check DISPATCHES --
      # it runs a fixture on the NPU -- and without this it fails with "XRT runtime (pyxrt) is not
      # available" while the driver reports the substantive failure it was looking for. In a real
      # run pl_preflight sources XRT for any needs_hardware phase; standalone it does not.
      if [ "$(phase_needs_hardware "${phase}")" = "yes" ]; then
        pl_env_ensure_xrt || return 1
      fi
      phase_objective_check "${phase}" ;;
    hardware-check)
      # Standalone so the assertion can be exercised against a gate log in both directions before
      # a run depends on it -- the lesson from the objective check that no honest run could pass.
      pl_assert_gate_ran_hardware "${phase}" "${3:-${pdir}/gate.log}" ;;
    tamper-check)
      guard_fingerprint "${pdir}/gates.before" "${_START_SHA}"
      guard_check_tamper "${pdir}/gates.before" "$(phase_gate_allowlist "${phase}")" ;;
    fingerprint)     guard_fingerprint "${pdir}/gates.before" "${_START_SHA}" && log_info "wrote ${pdir}/gates.before (base ${_START_SHA})" ;;
    implement)
      render_prompt "${PL_LIB}/prompts/implement.md" "${pdir}/implement.prompt" "${phase}"
      pl_claude_run "${pdir}/implement.prompt" "${PL_LIB}/schema/session.json" "${pdir}" "implement" ;;
    review)
      mkdir -p "${pdir}/round-manual"
      render_prompt "${PL_LIB}/prompts/review.md" "${pdir}/round-manual/review.prompt" "${phase}" 1
      pl_codex_review "${_START_SHA}" "${pdir}/round-manual/review.prompt" \
                      "${pdir}/round-manual/review.json" "${pdir}/round-manual/codex.log" ;;
    prompt)
      render_prompt "${PL_LIB}/prompts/implement.md" "-" "${phase}" ;;
    *) log_error "unknown step '${step}'"; return 2 ;;
  esac
}

cmd_dry_run() {
  PL_DRY_RUN=yes
  log_info "DRY RUN — no sessions, reviews, gates or commits will actually execute"
  require_deps || return 1
  # Read the scope rather than hardcoding it: a phase added to phases.sh but missing here is
  # invisible to dry-run, which is precisely the run meant to catch a mis-declared phase.
  local phase
  for phase in $(printf '%s' "${PL_PHASES_IN_SCOPE}" | jq -r '.[]'); do
    echo
    echo "Phase ${phase}: $(phase_name "${phase}")"
    echo "  doc:       $(phase_doc "${phase}")"
    echo "  hardware:  $(phase_needs_hardware "${phase}")"
    echo "  gate:      $(phase_gate_cmd "${phase}")"
    echo "  allowlist: $(phase_gate_allowlist "${phase}")"
    # Every OTHER mis-declared arm is visible above. The gate DESCRIPTION was not, and Phase H
    # proved it matters: six of the seven dispatchers had an H arm, phase_gate_description did
    # not, and dry-run rendered a clean-looking phase. The session then burned an invocation
    # correctly reporting the ERROR default as a blocker -- which is the fail-closed design
    # working, but a minute of dry-run should have caught it first.
    local _desc; _desc="$(phase_gate_description "${phase}" | head -1)"
    if printf '%s' "${_desc}" | grep -q '^ERROR:'; then
      echo "  gate-desc: *** MISSING — phase_gate_description has no ${phase} arm ***"
      echo "             A session would be told its gate is a harness bug and would halt."
    else
      echo "  gate-desc: ${_desc}"
    fi
    local _name; _name="$(phase_name "${phase}")"
    [ "${_name}" = "unknown" ] && echo "  *** phase_name has no ${phase} arm ***"
    [ -n "$(phase_doc "${phase}")" ] || echo "  *** phase_doc has no ${phase} arm ***"
    echo "  steps:     preflight -> implement -> commit"
    local r
    for r in $(seq 1 "${PL_REVIEW_ROUNDS}"); do echo "             -> review${r} -> fix${r} -> commit"; done
    echo "             -> confirm (only if a fix ran)"
    echo "             -> gate$([ "$(phase_needs_hardware "${phase}")" = "yes" ] && echo " -> hardware-check")"
    echo "             -> objective-check -> tamper-check"
  done
  echo
  echo "Caps: ${PL_MAX_INVOCATIONS} invocations, ${PL_MAX_HOURS}h deadline, \$${PL_STEP_BUDGET}/session."
}

main() {
  case "${1:-help}" in
    start)    shift; cmd_start "$@" ;;
    resume)   shift; cmd_resume "$@" ;;
    resume-at) shift; cmd_resume_at "$@" ;;
    stop)     shift; cmd_stop "$@" ;;
    status)   shift; cmd_status "$@" ;;
    dry-run)  shift; cmd_dry_run "$@" ;;
    run-one)  shift; cmd_run_one "$@" ;;
    help|-h|--help|"") usage ;;
    *) log_error "unknown command '$1'"; usage; return 2 ;;
  esac
}

main "$@"
