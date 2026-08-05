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

# A hardware phase must prove its gate actually ran hardware.
#
# WHY THIS EXISTS. Phase B's phase_gate_description named a hardware test -- the multi-ELF runlist
# the whole execution taxonomy rests on. Its phase_gate_cmd ran
# `ninja check-programming-examples-transformer-layer`, and at the time that suite held exactly two
# tests: one compile-only and one host-only. The gate log records 2 tests, 329 excluded, 16
# seconds. The hardware claim had been produced by `make runlist-gate`, which the session ran and
# self-reported, and it stood on that self-report for a day.
#
# None of the three anti-reward-hacking layers caught it, and none of them could have: the
# fingerprint, the objective check and Codex's weakened_gates all police what the DIFF did to the
# gate. Not one of them asks whether the gate exercises what its description claims. This does,
# and it is cheap, because lit reports its own counts.
#
# Three failure modes, two of which have happened here:
#   1. the suite contains no NPU-gated test at all           -> Phase B, above
#   2. it does, and every one reported UNSUPPORTED           -> the XRT_COREUTIL /
#      ENABLE_RUN_XRT_TESTS regression in 15-environment-notes.md, where lit cannot find
#      xrt-smi, marks every NPU test unsupported, and the suite still exits 0
#   3. an NPU test is marked XFAIL. lit counts that as "Expectedly Failed", neither Passed nor
#      Unsupported, and exits 0. A one-line edit inside D1/D2's own allowlist would do it.
#
# THE COUNT MUST HAVE NO SLACK. An earlier version required `passed >= npu_tests` (9 of 13 files),
# which left four tests of headroom: XFAIL one and the arithmetic still passed. lit runs succinct
# here -- there are no per-test PASS lines in the log to correlate against -- so the exact
# correspondence available is the total: this gate runs every .lit under transformer_layer/, and
# they must all pass, so `passed` must reach the tracked file count.
#
# Equivalently and more robustly: PASSED AND EXCLUDED MUST BE THE ONLY NONZERO OUTCOMES. That
# needs no list of lit's category names and stays correct as lit adds them.
#
# lit's summary is ragged, and only the LAST one in the log is this suite's (a script gate may
# wrap several commands):
#     Total Discovered Tests: 340
#       Excluded: 329 (96.76%)
#       Passed  :  11 (3.24%)
pl_assert_gate_ran_hardware() {
  local phase="$1" log="$2"

  [ "$(phase_needs_hardware "${phase}")" = "yes" ] || return 0
  if [ "${PL_DRY_RUN}" = "yes" ]; then
    log_info "DRY RUN: would assert the gate ran hardware"
    return 0
  fi
  if [ ! -f "${log}" ]; then
    log_error "hardware assertion: no gate log at ${log}"
    return 1
  fi

  # Counted from the lit files themselves, never from their names: run_npu2_compile_peano.lit is
  # named npu2 and REQUIRES only peano, and sweep/run_npu2_registry_resolution.lit needs neither.
  # A filename count would have called Phase B's gate a hardware gate.
  local total_lit=0 npu_tests=0 f
  while IFS= read -r f; do
    [ -n "${f}" ] || continue
    total_lit=$(( total_lit + 1 ))
    if grep -qE '^[^A-Za-z]*REQUIRES:.*ryzen_ai_npu2' "${PL_ROOT}/${f}" 2>/dev/null; then
      npu_tests=$(( npu_tests + 1 ))
    fi
  done <<< "$(git -C "${PL_ROOT}" ls-files 'programming_examples/transformer_layer/**/*.lit' \
                                          'programming_examples/transformer_layer/*.lit' 2>/dev/null | sort -u)"

  if [ "${npu_tests}" -eq 0 ]; then
    log_error "hardware assertion: phase ${phase} declares needs_hardware=yes, but the suite"
    log_error "  contains no test REQUIRING ryzen_ai_npu2. Its gate cannot have touched the NPU."
    return 1
  fi

  PL_HW_LOG="${log}" PL_HW_TOTAL="${total_lit}" PL_HW_NPU="${npu_tests}" \
  python3 -c '
import os, re, sys

log   = open(os.environ["PL_HW_LOG"], errors="replace").read().splitlines()
total = int(os.environ["PL_HW_TOTAL"])
npu   = int(os.environ["PL_HW_NPU"])

# The LAST summary in the log is this suite s. A gate that wraps several commands (gate-c4.sh
# runs lit and then ten make-verify runs) can print other counter-shaped lines; anchoring to the
# final "Total Discovered Tests:" keeps stray output from being read as a result category.
starts = [i for i, l in enumerate(log) if l.strip().startswith("Total Discovered Tests")]
if not starts:
    print("hardware assertion: the gate log has no lit summary at all; the suite did not run",
          file=sys.stderr)
    sys.exit(1)

counts = {}
for line in log[starts[-1] + 1:]:
    m = re.match(r"\s*([A-Za-z][A-Za-z ]*?)\s*:\s*(\d+)", line)
    if not m:
        if line.strip():
            break          # summary block ended
        continue
    counts[m.group(1).strip()] = int(m.group(2))

passed = counts.pop("Passed", 0)
counts.pop("Excluded", None)

# Anything else nonzero -- Unsupported, Failed, Expectedly Failed, Unexpectedly Passed, Timed
# Out, whatever lit adds next -- means the suite did not simply run and pass.
bad = {k: v for k, v in counts.items() if v}
if bad:
    for k, v in sorted(bad.items()):
        print("hardware assertion: %d test(s) reported %s" % (v, k), file=sys.stderr)
    if "Unsupported" in bad:
        print("  UNSUPPORTED is usually lit failing to detect the NPU: check that",
              file=sys.stderr)
        print("  build-xrt/programming_examples/lit.site.cfg.py has xrt_bin_dir pointing at a",
              file=sys.stderr)
        print("  real xrt-smi and enable_run_xrt_tests ON -- see 15-environment-notes.md.",
              file=sys.stderr)
    print("  A hardware gate that skips or excuses its hardware tests still exits 0.",
          file=sys.stderr)
    sys.exit(1)

if passed < total:
    print("hardware assertion: %d tracked .lit file(s) under transformer_layer/ but only %d "
          "passed. Every test in the suite must run and pass; the gate did not run what its "
          "description claims." % (total, passed), file=sys.stderr)
    sys.exit(1)

print("  %d test(s) passed, %d of them NPU-gated; no unsupported, xfailed or failed tests"
      % (passed, npu))
' || { log_error "hardware assertion FAILED: the gate did not exercise the hardware it claims"; return 1; }

  log_info "hardware assertion passed: the gate ran the whole suite, ${npu_tests} NPU-gated"
  return 0
}
