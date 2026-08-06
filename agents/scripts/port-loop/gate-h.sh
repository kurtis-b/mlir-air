#!/usr/bin/env bash
#
# Phase H's gate: rebuild the compiler, then prove nothing downstream of it broke.
#
# H is the first phase in this plan to change mlir/ -- the AIR compiler itself. Every example and
# all ten shipped LLM deployments compile through it, so this gate is the widest in the plan.
#
# FOUR LEGS, in increasing cost, so a cheap failure stops before an expensive one:
#
#   1. build + INSTALL. The install is not optional and it is the easy thing to get wrong: the
#      examples run against install-xrt (utils/env_setup.sh and lib-env.sh both point there), so a
#      pass edited and merely *built* leaves every downstream leg testing the OLD aircc and the
#      gate passes while proving nothing about the change.
#   2. check-air-mlir -- the compiler's own lit suite. A broken pass shows up here in seconds
#      rather than an hour into the ten-model leg.
#   3. the transformer-layer suite on real hardware.
#   4. make verify over the ten shipped models.
#
# The caller already holds /tmp/mlir-air-npu.lock. Nothing here may take /tmp/npu.lock -- that
# inode belongs to KernelCache and the lit suites, and taking it from a wrapper deadlocks them.
#
# The sentinel after the lit leg exists for the same reason gate-e1.sh's does:
# pl_assert_gate_ran_hardware parses counter-shaped lines following the LAST "Total Discovered
# Tests" until a non-matching non-blank line, so later legs must not be able to leak a "Name: N"
# line into lit's summary block.

set -uo pipefail

ROOT="${PL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
BUILD="${ROOT}/build-xrt"

MODELS=(
  llama32_1b llama32_1b_int4 llama32_3b smollm2_1_7b qwen25_0_5b
  qwen25_1_5b qwen25_3b qwen3_0_6b qwen3_1_7b qwen3_4b
)

echo "== H gate leg 1: rebuild AND INSTALL the compiler =="
echo "   (the examples resolve aircc/air-opt from install-xrt; a build without an install would"
echo "    leave every later leg testing the previous compiler)"
if ! ninja -C "${BUILD}"; then
  echo "H GATE: FAIL -- the compiler did not build"
  exit 1
fi
if ! ninja -C "${BUILD}" install; then
  echo "H GATE: FAIL -- the compiler built but did not install"
  exit 1
fi

echo
echo "== H gate leg 2: the compiler's own lit suite (check-air-mlir) =="
if ! ninja -C "${BUILD}" check-air-mlir; then
  echo "H GATE: FAIL -- AIR's own MLIR lit suite regressed"
  exit 1
fi

echo
echo "== H gate leg 3: transformer-layer suite on hardware =="
lit_rc=0
ninja -C "${BUILD}" check-programming-examples-transformer-layer || lit_rc=$?

# Unconditionally, before any verdict, so it is present whatever the suite did and whatever the
# next leg prints.
echo "-- end of lit summary (H gate leg 3) --"

if [ "${lit_rc}" -ne 0 ]; then
  echo "H GATE: FAIL -- the transformer-layer suite did not pass"
  exit 1
fi

echo
echo "== H gate leg 4: cross-deployment regression over ${#MODELS[@]} shipped models =="
echo "   (a change to mlir/ reaches every one of them through aircc)"
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
  echo "H GATE: FAIL -- ${#regressions[@]} model(s) regressed: ${regressions[*]}"
  exit 1
fi

echo "H GATE: PASS -- compiler builds, installs, its own suite is green, and all"
echo "${#MODELS[@]} shipped models still verify"
exit 0
