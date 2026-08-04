# shellcheck shell=bash
#
# codex exec review wrapper.
#
# Contract: pl_codex_review() reviews exactly the phase's diff (--base <phase-start-sha>) and
# writes a schema-validated verdict to a JSON file the driver can act on without parsing prose.
#
# Footguns, all verified against codex-cli 0.146.0:
#   - We use `codex exec`, NOT `codex exec review`. The review subcommand looks like the better
#     fit, but its scope selectors are mutually exclusive with a custom prompt:
#       error: the argument '--base <BRANCH>' cannot be used with '[PROMPT]'
#     Our review instructions are the whole value here, so we scope the diff inside the prompt
#     and let codex run git itself under the read-only sandbox.
#   - Progress streams to STDERR and only the final message to STDOUT. Redirecting 2>&1 produces
#     a multi-megabyte log with the answer buried at the end; keep the streams separate.
#   - Exit code is not reliable (see openai/codex#4721). We assert the -o file exists, is
#     non-empty, and parses as the expected schema.
#   - Model and effort default to whatever ~/.codex/config.toml already selects, because that is
#     the configuration known to work with this account. Passing --ignore-user-config together
#     with an explicit -m is a trap: an arbitrary model name is rejected outright with
#       "The 'gpt-5.3-codex' model is not supported when using Codex with a ChatGPT account."
#     Override deliberately via PL_CODEX_MODEL / PL_CODEX_EFFORT when you know the name is valid.
#   - Review runs read-only by construction; there is no -s/--sandbox flag on the review
#     subcommand.

# pl_codex_review <base-sha> <prompt-file> <out-json> <log-file>
pl_codex_review() {
  local base="$1" prompt_file="$2" out_json="$3" log_file="$4"
  mkdir -p "$(dirname "${out_json}")"

  state_count_invocation || return 1

  log_info "codex review against ${base} (timeout ${PL_REVIEW_TIMEOUT}s)"

  if [ "${PL_DRY_RUN}" = "yes" ]; then
    log_info "DRY RUN: would run codex exec review --base ${base}"
    printf '{"verdict":"pass","summary":"dry run","blocking":[],"non_blocking":[],"weakened_gates":[]}\n' \
      > "${out_json}"
    return 0
  fi

  local extra=()
  [ -n "${PL_CODEX_MODEL}" ]  && extra+=(-m "${PL_CODEX_MODEL}")
  [ -n "${PL_CODEX_EFFORT}" ] && extra+=(-c "model_reasoning_effort=\"${PL_CODEX_EFFORT}\"")

  local rc=0
  ( cd "${PL_ROOT}" && timeout --signal=TERM "${PL_REVIEW_TIMEOUT}" \
      codex exec \
        --sandbox read-only \
        --output-schema "${PL_LIB}/schema/review.json" \
        --output-last-message "${out_json}" \
        --skip-git-repo-check \
        "${extra[@]}" \
        - < "${prompt_file}" > /dev/null 2> "${log_file}" ) || rc=$?

  if [ "${rc}" -eq 124 ] || [ "${rc}" -eq 143 ]; then
    log_error "codex review timed out after ${PL_REVIEW_TIMEOUT}s"
    return 1
  fi

  if [ ! -s "${out_json}" ]; then
    log_error "codex review produced no verdict file (exit ${rc}); see ${log_file}"
    tail -20 "${log_file}" >&2 2>/dev/null || true
    return 1
  fi

  if ! jq -e '.verdict and (.blocking | type == "array") and (.weakened_gates | type == "array")' \
        "${out_json}" >/dev/null 2>&1; then
    log_error "codex review verdict did not match the expected schema; see ${out_json}"
    head -40 "${out_json}" >&2
    return 1
  fi

  local verdict nblock nweak
  verdict="$(jq -r '.verdict' "${out_json}")"
  nblock="$(jq -r '.blocking | length' "${out_json}")"
  nweak="$(jq -r '.weakened_gates | length' "${out_json}")"
  local nlim; nlim="$(jq -r '(.gate_limitations // []) | length' "${out_json}" 2>/dev/null || echo 0)"
  log_info "EVENT: review-done verdict=${verdict} blocking=${nblock} weakened=${nweak} limitations=${nlim}"
  jq -r '"  summary: \(.summary)"' "${out_json}" >&2 2>/dev/null || true
  return 0
}

pl_codex_blocking_count() { jq -r '(.blocking // []) | length' "$1" 2>/dev/null || echo 0; }
pl_codex_verdict()        { jq -r '.verdict // "fail"' "$1" 2>/dev/null || echo "fail"; }
