# shellcheck shell=bash
#
# port-loop state persistence.
#
# Contract: state.json is the single source of truth for where a run is. Every mutation goes
# through state_set/state_advance so STATUS.md is regenerated in lockstep and never drifts from
# the machine state.
#
# Footguns:
#   - state.json is rewritten via a temp file + mv. Never edit it in place; a partial write
#     during a step would strand the run with unparseable state and no way to resume.
#   - STATUS.md is generated output. Editing it by hand accomplishes nothing; it is overwritten
#     on the next step.
#   - `jq` is required. It is checked once at startup rather than per call, because a mid-run
#     failure here loses the run's position.

STATE_JSON="${PL_STATE_DIR}/state.json"
STATUS_MD="${PL_STATE_DIR}/STATUS.md"
TIMING_MD="${PL_STATE_DIR}/phase_timing.md"

state_exists() { [ -f "${STATE_JSON}" ]; }

state_init() {
  local phases="$1" deadline_epoch="$2"
  mkdir -p "${PL_STATE_DIR}"
  jq -n \
    --arg started "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
    --arg branch "${PL_BRANCH}" \
    --argjson phases "${phases}" \
    --argjson deadline "${deadline_epoch}" \
    --argjson maxinv "${PL_MAX_INVOCATIONS}" \
    '{
       version: 1,
       started_utc: $started,
       branch: $branch,
       phases: $phases,
       phase_index: 0,
       step: "preflight",
       round: 0,
       status: "ready",
       halt_reason: null,
       invocations: 0,
       max_invocations: $maxinv,
       deadline_epoch: $deadline,
       phase_start_sha: null,
       history: []
     }' > "${STATE_JSON}.tmp" && mv "${STATE_JSON}.tmp" "${STATE_JSON}"
  state_render
}

state_get() { jq -r "$@" "${STATE_JSON}"; }

# state_set [jq-options...] <jq-assignment-expression>
# All arguments are forwarded to jq verbatim, so callers may pass --arg/--argjson before the
# filter. The filter must be the last argument.
state_set() {
  jq "$@" "${STATE_JSON}" > "${STATE_JSON}.tmp" && mv "${STATE_JSON}.tmp" "${STATE_JSON}"
  state_render
}

state_phase()      { state_get '.phases[.phase_index] // "none"'; }
state_step()       { state_get '.step'; }
state_round()      { state_get '.round'; }
state_status()     { state_get '.status'; }
state_start_sha()  { state_get '.phase_start_sha // ""'; }

state_log() {
  local event="$1" detail="${2:-}"
  state_set --arg e "${event}" --arg d "${detail}" --arg t "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
    '.history += [{ts: $t, phase: (.phases[.phase_index] // "none"), step: .step, round: .round, event: $e, detail: $d}]'
}

state_halt() {
  local reason="$1"
  state_set --arg r "${reason}" '.status = "halted" | .halt_reason = $r'
  state_log "halt" "${reason}"
  log_error "HALTED: ${reason}"
}

state_done() {
  state_set '.status = "done" | .halt_reason = null'
  state_log "done" "all phases in scope complete"
}

# Count an agent invocation and enforce the caps. Returns 1 when a cap is hit.
state_count_invocation() {
  state_set '.invocations += 1'
  local n max now deadline
  n="$(state_get '.invocations')"
  max="$(state_get '.max_invocations')"
  if [ "${n}" -gt "${max}" ]; then
    state_halt "invocation cap reached (${n} > ${max})"
    return 1
  fi
  now="$(date +%s)"
  deadline="$(state_get '.deadline_epoch')"
  if [ "${now}" -gt "${deadline}" ]; then
    state_halt "wall-clock deadline reached"
    return 1
  fi
  return 0
}

state_render() {
  local phase step round status halt inv max deadline started
  phase="$(jq -r '.phases[.phase_index] // "none"' "${STATE_JSON}")"
  step="$(jq -r '.step' "${STATE_JSON}")"
  round="$(jq -r '.round' "${STATE_JSON}")"
  status="$(jq -r '.status' "${STATE_JSON}")"
  halt="$(jq -r '.halt_reason // "-"' "${STATE_JSON}")"
  inv="$(jq -r '.invocations' "${STATE_JSON}")"
  max="$(jq -r '.max_invocations' "${STATE_JSON}")"
  deadline="$(jq -r '.deadline_epoch' "${STATE_JSON}")"
  started="$(jq -r '.started_utc' "${STATE_JSON}")"

  {
    echo "# port-loop status"
    echo
    echo "Generated $(date -u '+%Y-%m-%dT%H:%M:%SZ') — regenerated after every step, do not edit."
    echo
    echo "| Field | Value |"
    echo "|---|---|"
    echo "| Status | **${status}** |"
    echo "| Phase | ${phase} |"
    echo "| Step | ${step} |"
    echo "| Review round | ${round} of 3 |"
    echo "| Halt reason | ${halt} |"
    echo "| Agent invocations | ${inv} / ${max} |"
    echo "| Started | ${started} |"
    echo "| Deadline | $(date -u -d "@${deadline}" '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || echo "${deadline}") |"
    echo
    if [ "${status}" = "halted" ]; then
      echo "## This run halted"
      echo
      echo "\`\`\`"
      echo "${halt}"
      echo "\`\`\`"
      echo
      echo "A halt is the harness working as designed. Read the reason above, then the step log"
      echo "under \`agents/.state/port-loop/phase-*/\`. Resume with:"
      echo
      echo "    agents/scripts/port-loop.sh resume"
      echo
    fi
    echo "## Recent activity"
    echo
    echo "| Time (UTC) | Phase | Step | Event | Detail |"
    echo "|---|---|---|---|---|"
    jq -r '.history[-25:] | reverse | .[] | "| \(.ts) | \(.phase) | \(.step) | \(.event) | \(.detail) |"' \
      "${STATE_JSON}"
  } > "${STATUS_MD}.tmp" && mv "${STATUS_MD}.tmp" "${STATUS_MD}"
}

# Phase timing ledger, schema borrowed from deploy-new-llm's phase_timing.md.
timing_start() {
  local phase="$1"
  mkdir -p "${PL_STATE_DIR}"
  [ -f "${TIMING_MD}" ] || { echo "# port-loop phase timing"; echo; } > "${TIMING_MD}"
  {
    echo "## Phase ${phase}"
    echo
    echo "- start_ts: $(date '+%s : %Y-%m-%d %H:%M:%S %Z')"
  } >> "${TIMING_MD}"
}

timing_end() {
  local phase="$1" outcome="$2" start_epoch="$3"
  local end_epoch wall_min
  end_epoch="$(date +%s)"
  wall_min=$(( (end_epoch - start_epoch) / 60 ))
  {
    echo "- end_ts:   $(date '+%s : %Y-%m-%d %H:%M:%S %Z')"
    echo "- wall_min: **${wall_min}**"
    echo "- outcome:  ${outcome}"
    echo
  } >> "${TIMING_MD}"
}
