# shellcheck shell=bash
#
# claude -p invocation wrapper.
#
# Contract: pl_claude_run() starts a genuinely FRESH session (no --resume, no --continue),
# enforces a timeout and a dollar budget, and returns non-zero unless the session both exited
# cleanly and reported success in its structured result.
#
# Footguns, all verified against claude 2.1.220:
#   - Exit code alone is NOT sufficient. The docs are explicit that in-run failures (e.g. missing
#     auth) are printed as the result on stdout. We therefore also check .is_error and .subtype.
#   - --max-budget-usd, --max-turns and --output-format only work with --print.
#   - --max-turns exists but is hidden from --help in 2.1.220. It is registered and functional.
#   - Variadic flags (--add-dir, --allowedTools) swallow following tokens, so the prompt goes in
#     on stdin rather than as a positional argument.
#   - Do NOT use --bare: it forces ANTHROPIC_API_KEY/apiKeyHelper auth and breaks subscription
#     OAuth login, which is what this machine uses.
#   - --permission-mode has no "default" value in 2.1.220 despite the public docs; omit the flag
#     instead of passing "default".
#   - Session persistence is deliberately left ON so transcripts remain inspectable afterward.

# A transient API failure is not a phase failure.
#
# D1's implement session died three minutes in on `API Error: 529 Overloaded` -- one turn, zero
# cost, no work attempted -- and halted a run scheduled to go unattended for 36 hours. Retrying is
# obviously right there, and obviously wrong for a session that actually ran: re-running one that
# already edited files and committed would duplicate its work against a moving base.
#
# The discriminator is that the session did NOTHING. `total_cost_usd == 0` means no tokens were
# billed, so nothing can have been written. Any session that spent budget produced a real outcome
# -- success or failure -- and is reported as one.
PL_API_RETRIES="${PL_API_RETRIES:-5}"
PL_API_RETRY_DELAY="${PL_API_RETRY_DELAY:-60}"

pl_result_is_transient() {
  local out="$1" cost result
  cost="$(jq -r '.total_cost_usd // 0' "${out}" 2>/dev/null || echo 0)"
  # Never re-run anything that spent budget, whatever the error text says.
  awk -v c="${cost}" 'BEGIN { exit !(c + 0 == 0) }' || return 1
  result="$(jq -r '.result // ""' "${out}" 2>/dev/null || echo "")"
  printf '%s' "${result}" | grep -qiE \
    'API Error: (429|5[0-9][0-9])|overloaded|rate.?limit|temporarily unavailable|service unavailable|connection (error|reset|refused)|upstream'
}

# pl_claude_run <prompt-file> <schema-file> <log-dir> <label>
# Writes: <log-dir>/<label>.json (full result), <label>.report.json (structured output),
#         <label>.stderr
pl_claude_run() {
  local prompt_file="$1" schema_file="$2" log_dir="$3" label="$4"
  mkdir -p "${log_dir}"

  local out="${log_dir}/${label}.json"
  local report="${log_dir}/${label}.report.json"
  local errlog="${log_dir}/${label}.stderr"

  if [ "${PL_DRY_RUN}" = "yes" ]; then
    state_count_invocation || return 1
    log_info "DRY RUN: would run claude -p with prompt ${prompt_file}"
    return 0
  fi

  local attempt=1 rc=0
  while : ; do
    # Counted per attempt, so a retry storm still runs into the invocation cap and the deadline
    # rather than spinning quietly.
    state_count_invocation || return 1

    log_info "claude session '${label}' (timeout ${PL_STEP_TIMEOUT}s, budget \$${PL_STEP_BUDGET}, attempt ${attempt})"

    rc=0
    ( cd "${PL_ROOT}" && timeout --signal=TERM "${PL_STEP_TIMEOUT}" \
        claude -p \
          --permission-mode bypassPermissions \
          --model "${PL_MODEL}" \
          --max-budget-usd "${PL_STEP_BUDGET}" \
          --max-turns "${PL_MAX_TURNS}" \
          --output-format json \
          --json-schema "$(cat "${schema_file}")" \
          --append-system-prompt "$(cat "${PL_LIB}/prompts/guardrails.md")" \
          --add-dir "${PL_ROOT}" \
          < "${prompt_file}" > "${out}" 2> "${errlog}" ) || rc=$?

    if [ "${rc}" -eq 124 ] || [ "${rc}" -eq 143 ]; then
      log_error "claude session '${label}' timed out after ${PL_STEP_TIMEOUT}s"
      return 1
    fi

    if [ ! -s "${out}" ]; then
      log_error "claude session '${label}' produced no output (exit ${rc}); see ${errlog}"
      return 1
    fi

    if [ "${attempt}" -le "${PL_API_RETRIES}" ] \
       && [ "$(jq -r '.is_error // false' "${out}" 2>/dev/null || echo true)" = "true" ] \
       && pl_result_is_transient "${out}"; then
      local delay=$(( PL_API_RETRY_DELAY * attempt ))
      log_warn "transient API failure on '${label}', attempt ${attempt}/${PL_API_RETRIES}:"
      log_warn "  $(jq -r '.result // ""' "${out}" 2>/dev/null | head -1)"
      log_warn "  no budget was spent, so no work was done; retrying in ${delay}s"
      sleep "${delay}"
      attempt=$(( attempt + 1 ))
      continue
    fi
    break
  done

  # Exit code is not authoritative — inspect the result envelope.
  local is_error subtype cost
  is_error="$(jq -r '.is_error // false' "${out}" 2>/dev/null || echo "true")"
  subtype="$(jq -r '.subtype // "unknown"' "${out}" 2>/dev/null || echo "unknown")"
  cost="$(jq -r '.total_cost_usd // 0' "${out}" 2>/dev/null || echo 0)"

  jq '.structured_output // {}' "${out}" > "${report}" 2>/dev/null || echo '{}' > "${report}"

  log_info "session '${label}' finished: subtype=${subtype} is_error=${is_error} cost=\$${cost}"

  if [ "${is_error}" = "true" ] || [ "${subtype}" != "success" ]; then
    log_error "claude session '${label}' reported failure (subtype=${subtype})"
    jq -r '.result // "no result field"' "${out}" 2>/dev/null | head -20 >&2
    return 1
  fi

  # Surface the session's own honest self-report; it informs the operator, it does not gate.
  local not_done blockers
  not_done="$(jq -r '(.work_not_completed // []) | length' "${report}")"
  blockers="$(jq -r '(.blockers // []) | length' "${report}")"
  if [ "${blockers:-0}" -gt 0 ]; then
    log_warn "session reported ${blockers} blocker(s):"
    jq -r '.blockers[]? | "  - \(.)"' "${report}" >&2
  fi
  if [ "${not_done:-0}" -gt 0 ]; then
    log_warn "session reported ${not_done} incomplete item(s):"
    jq -r '.work_not_completed[]? | "  - \(.)"' "${report}" >&2
  fi
  return 0
}
