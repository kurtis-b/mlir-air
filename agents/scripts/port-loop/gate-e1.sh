#!/usr/bin/env bash
#
# Phase E1's gate: the lit suite, then the cross-deployment regression check.
#
# E1 changes llms/shared/builders/gemm_builder.py -- the symbol suffix and object name minted for
# every external GEMM -- which the ten shipped LLM deployments resolve their tile configs through.
# 13-verification-and-acceptance.md is explicit that any shared-infrastructure change re-runs
# `make verify` on every sibling model, and calls it "the most expensive check in the plan and the
# one most likely to be skipped".
#
# It lives in a script rather than inline in phase_gate_cmd for gate-c4.sh's reasons: it is two
# commands with a failure summary, and Phase B taught the cost of a gate command that does less
# than its gate description claims.
#
# The caller already holds /tmp/mlir-air-npu.lock. Nothing here may take /tmp/npu.lock -- that
# inode belongs to KernelCache and the lit suites, and taking it from a wrapper deadlocks them.
#
# ORDER AND THE SENTINEL, both load-bearing.
#
# pl_assert_gate_ran_hardware() finds the LAST "Total Discovered Tests:" line in this log and then
# reads every following line that matches `<Name> : <count>` as a lit outcome category, stopping at
# the first non-matching non-blank line (lib-guard.sh). Blank lines do not stop it. So anything the
# second leg prints that happens to be counter-shaped -- and `make verify` prints summary lines --
# must be separated from lit's summary by a line that cannot parse as a counter.
#
# gate-c4.sh gets away with this by accident: its "== C4 gate leg 2 ... ==" header is the first
# non-blank non-matching line and ends the block. Relying on the shape of a header is not a
# guarantee, so this script emits an explicit sentinel immediately after the lit leg. Do not remove
# it, and do not move the lit leg after the make-verify leg -- the assertion reads the LAST summary
# in the log, and a lit summary that is not last is not the one it thinks it is reading.

set -uo pipefail

ROOT="${PL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"

MODELS=(
  llama32_1b llama32_1b_int4 llama32_3b smollm2_1_7b qwen25_0_5b
  qwen25_1_5b qwen25_3b qwen3_0_6b qwen3_1_7b qwen3_4b
)

echo "== E1 gate leg 1: transformer-layer lit suite =="
lit_rc=0
ninja -C "${ROOT}/build-xrt" check-programming-examples-transformer-layer || lit_rc=$?

# The sentinel, unconditionally and before any verdict, so it is present whether the suite passed
# or failed and whatever the second leg goes on to print.
echo "-- end of lit summary (E1 gate leg 1) --"

if [ "${lit_rc}" -ne 0 ]; then
  echo "E1 GATE: FAIL -- the transformer-layer lit suite did not pass"
  exit 1
fi

echo
echo "== E1 gate leg 2: cross-deployment regression over ${#MODELS[@]} shipped models =="
echo "   (gemm_builder.py mints every external GEMM's symbol suffix and object name; these ten"
echo "    deployments resolve through it)"
regressions=()
for m in "${MODELS[@]}"; do
  d="${ROOT}/programming_examples/llms/${m}"
  if [ ! -d "${d}" ]; then
    echo "  ${m}: MISSING directory ${d}"
    regressions+=("${m} (missing)")
    continue
  fi
  echo "  --- ${m} ---"
  if ( cd "${d}" && make verify ); then
    echo "  ${m}: pass"
  else
    echo "  ${m}: REGRESSION"
    regressions+=("${m}")
  fi
done

echo
if [ ${#regressions[@]} -gt 0 ]; then
  echo "E1 GATE: FAIL -- ${#regressions[@]} model(s) regressed: ${regressions[*]}"
  exit 1
fi

echo "E1 GATE: PASS -- lit suite green and all ${#MODELS[@]} shipped models still verify"
exit 0
