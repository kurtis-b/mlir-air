# shellcheck shell=bash
#
# Guardrails: gate-file fingerprinting, tamper detection, destructive-change detection.
#
# Contract: the driver never trusts a session's account of its own work. guard_fingerprint()
# records the state of every file that DEFINES a gate; guard_check_tamper() detects changes to
# those files; guard_check_destructive() detects deleted tracked files. Together with the Codex
# `weakened_gates` field these are three independent views of the same risk.
#
# The risk being defended against: an autonomous agent told "make the gate pass" can make the
# gate weaker instead of making the code better. Phase A is the acute case because it CREATES
# the lit suite that gates it, so "the suite passes" is not evidence on its own. That is why
# phases also declare objective checks (see phases.sh) that inspect build products directly
# rather than asking the test framework whether it is happy.
#
# Footgun: a phase legitimately needs to create its own gate files. Phases declare an allowlist
# of path prefixes they may touch; anything outside it halts the run.

# Files whose content defines whether a gate is meaningful.
guard_gate_files() {
  {
    git -C "${PL_ROOT}" ls-files 'programming_examples/**/*.lit'
    git -C "${PL_ROOT}" ls-files 'programming_examples/**/Makefile'
    git -C "${PL_ROOT}" ls-files 'programming_examples/CMakeLists.txt'
    git -C "${PL_ROOT}" ls-files 'programming_examples/kernel_registry/details/*.json'
    git -C "${PL_ROOT}" ls-files 'programming_examples/llms/verify/*.py'
    git -C "${PL_ROOT}" ls-files 'test/**/*.lit'
  } 2>/dev/null | sort -u
}

# guard_fingerprint <out> [ref]
#
# With a git ref, hashes each gate file's content AT THAT COMMIT rather than in the working
# tree. This matters more than it looks: taking the baseline from the working tree means any
# weakening that already landed becomes the baseline and is invisible forever after.
#
# That is not hypothetical — it happened. Resume support introduced a
# `[ -f gates.before ] || guard_fingerprint` guard, and combined with a wiped state directory the
# baseline was captured ten hours after the phase's last commit, with the phase's own gate edits
# already in it. The tamper check for that run was vacuous. Codex caught it; timestamps confirmed
# it. Always pass the phase's base commit.
guard_fingerprint() {
  local out="$1" ref="${2:-}"
  mkdir -p "$(dirname "${out}")"
  local f h
  guard_gate_files | while read -r f; do
    if [ -n "${ref}" ]; then
      h="$(git -C "${PL_ROOT}" show "${ref}:${f}" 2>/dev/null | sha256sum | cut -d' ' -f1)"
      # A file absent at the base commit is new in this phase; record it as such so its later
      # appearance is attributable rather than silently accepted.
      git -C "${PL_ROOT}" cat-file -e "${ref}:${f}" 2>/dev/null || h="ABSENT_AT_BASE"
    else
      [ -f "${PL_ROOT}/${f}" ] || continue
      h="$(sha256sum "${PL_ROOT}/${f}" | cut -d' ' -f1)"
    fi
    printf '%s  %s\n' "${h}" "${f}"
  done > "${out}"
}

# guard_check_tamper <before-file> <allowed-prefix-regex>
# Returns 1 and reports when a gate-defining file outside the allowed prefixes changed.
guard_check_tamper() {
  local before="$1" allowed="${2:-}"
  local after="${before%.before}.after"
  guard_fingerprint "${after}"

  # Compare hash+path pairs; report paths whose hash changed or which appeared/disappeared.
  local changed
  changed="$(comm -3 <(sort "${before}") <(sort "${after}") | awk '{print $2}' | sort -u)"

  [ -z "${changed}" ] && return 0

  local unauthorized="" p
  while read -r p; do
    [ -z "${p}" ] && continue
    if [ -n "${allowed}" ] && printf '%s' "${p}" | grep -Eq "${allowed}"; then
      log_info "gate file changed within this phase's allowlist: ${p}"
    else
      unauthorized="${unauthorized}${p}"$'\n'
    fi
  done <<< "${changed}"

  if [ -n "${unauthorized}" ]; then
    log_error "gate-defining files changed outside this phase's allowlist:"
    printf '%s' "${unauthorized}" >&2
    return 1
  fi
  return 0
}

# Deleted tracked files are never acceptable in this workflow.
guard_check_destructive() {
  local deleted
  deleted="$(git -C "${PL_ROOT}" log --diff-filter=D --name-only --pretty=format: "$1..HEAD" 2>/dev/null | sort -u | sed '/^$/d')"
  if [ -n "${deleted}" ]; then
    log_error "tracked files were deleted during this phase:"
    printf '%s\n' "${deleted}" >&2
    return 1
  fi
  return 0
}

# A review halts the run when the diff WEAKENED a gate. Inherent limits of what a gate can prove
# are recorded and surfaced but never halt: they describe the phase's design, not a regression,
# and no fix inside the phase would clear them. Conflating the two halted three of four early
# runs on observations like "a compile-only gate cannot prove numerical correctness" -- true, and
# exactly why Phases C and D exist.
guard_check_weakened() {
  local review_json="$1"

  local lim
  lim="$(jq -r '(.gate_limitations // []) | length' "${review_json}" 2>/dev/null || echo 0)"
  if [ "${lim:-0}" -gt 0 ]; then
    log_warn "${lim} gate limitation(s) recorded (informational, not halting):"
    jq -r '.gate_limitations[] | "  \(.file): \(.limitation)"' "${review_json}" >&2
  fi

  local n
  n="$(jq -r '(.weakened_gates // []) | length' "${review_json}" 2>/dev/null || echo 0)"
  if [ "${n:-0}" -gt 0 ]; then
    log_error "Codex reported ${n} gate(s) WEAKENED BY THIS DIFF:"
    jq -r '.weakened_gates[] | "  \(.file): \(.what_was_weakened)"' "${review_json}" >&2
    return 1
  fi
  return 0
}
