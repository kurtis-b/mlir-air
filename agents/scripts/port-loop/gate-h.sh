#!/usr/bin/env bash
#
# Phase H's gate: rebuild the compiler, then prove nothing downstream of it broke.
#
# H is the first phase in this plan to change mlir/ -- the AIR compiler itself. Every example and
# all ten shipped LLM deployments compile through it, so this gate is the widest in the plan.
#
# FIVE LEGS, in increasing cost, so a cheap failure stops before an expensive one:
#
#   1. build + INSTALL. The install is not optional and it is the easy thing to get wrong: the
#      examples run against install-xrt (utils/env_setup.sh and lib-env.sh both point there), so a
#      pass edited and merely *built* leaves every downstream leg testing the OLD aircc and the
#      gate passes while proving nothing about the change.
#   2. check-air-mlir -- the compiler's own lit suite. A broken pass shows up here in seconds
#      rather than an hour into the ten-model leg.
#   3. the transformer-layer suite on real hardware.
#   4. decode throughput against a recorded floor. `[2026-08-06]` NEW -- see the leg's own header.
#      Every other leg is correctness-only, and a compiler change can hold every numeric exactly
#      while destroying performance.
#   5. make verify over the ten shipped models.
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
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODELS=(
  llama32_1b llama32_1b_int4 llama32_3b smollm2_1_7b qwen25_0_5b
  qwen25_1_5b qwen25_3b qwen3_0_6b qwen3_1_7b qwen3_4b
)

# Leg 4's subjects. Two, not ten: this leg costs a full recompile per model, and these are the two
# whose decode path is governed by the backend settings ping-pong actually decides.
# llama32_1b_int4 is the model the 12.4 -> 7.8 tok/s regression was measured on (its decode ELFs
# inherit GEMV_K2048_BACKEND); llama32_1b covers the same path in bf16.
THROUGHPUT_MODELS=(llama32_1b llama32_1b_int4)
THROUGHPUT_BASELINE="${HERE}/throughput-baseline.json"
EXTRACT_PERF="${ROOT}/programming_examples/llms/bench/extract_perf.py"

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

# ------------------------------------------------------------------------------------------------
# Leg 4: decode throughput against a recorded floor.
#
# WHY THIS LEG EXISTS. Every other leg is correctness-only, and a compiler change can hold every
# numeric exactly while destroying performance. Dropping ping-pong regressed a shipped model
# 12.4 -> 7.8 tok/s -- recorded at llms/shared/infra/backend_presets.py:69 -- and not one of the
# four original legs would have noticed. H1's skip-rather-than-refuse rule makes that failure mode
# live rather than hypothetical: a safety predicate that declines to transform MORE loops than it
# should is invisible to correctness and obvious here.
#
# WHY IT RUNS BEFORE THE TEN-MODEL LEG. The same principle as the rest of this file, cheapest
# first: this is minutes against leg 5's hour, and a throughput collapse is a compiler defect worth
# stopping on before spending that hour.
#
# WHY `make compile` AND NOT JUST `make profile`. `make profile` runs the inference driver with
# --run-only, which calls load_manifest() and dispatches whatever ELFs already sit in the model's
# cache directory -- compiled by the compiler this phase just replaced. That is leg 1's install
# trap wearing a different hat: the gate would go green having measured the previous build.
# KernelCache.compile_and_cache() has no fingerprint-based skip, so it genuinely rebuilds.
#
# AND WHERE THAT PREMISE FAILS. `[2026-08-06]` The skip lives one level UP, in the model's own
# driver script, and it is existence-based rather than fingerprint-based:
#
#   llama32_1b_int4_prefill.py:1058
#     def _need(name, kernel_sym=None):
#         elf = Path(args.cache_dir) / f"{name}.elf"
#         if elf.exists():
#             print(f"  using cached {name}.elf (...)")
#
# So for llama32_1b_int4 the second and every later `make compile` skips every kernel already
# built, whatever compiler produced it -- leg 1's install trap a THIRD time. That model also
# builds prefill only from `make compile` (three dtypes); its decode ELFs come from
# `make compile-inference` / `make compile-decode` and live in a different cache dir, which is
# what `make profile` actually dispatches.
#
# llama32_1b -- the only model with a floor today -- has no such skip and does rebuild, so this
# leg is sound as it currently runs. Before adding llama32_1b_int4: clear its cache dirs and use
# `make compile-inference`, then check the leg's log for "using cached". If that string appears,
# the leg measured the previous compiler and proves nothing.
#
# WHY THE FLOOR LIVES IN A DRIVER-OWNED FILE. throughput-baseline.json sits under
# agents/scripts/port-loop/, which guard_gate_files() fingerprints and which no phase's allowlist
# covers. A session cannot lower the floor to meet its result; an operator seeds it between runs.
#
# WHY IT FAILS CLOSED. A model with no recorded baseline FAILS this leg rather than passing it.
# iron shipped a smoke test that reported 21/21 passed on a machine where every measurement had
# failed; "no baseline, so nothing to compare, so green" is that same shape.
#
# The measurement is parsed by programming_examples/llms/bench/extract_perf.py, which already
# handles the three different throughput line formats the inference drivers print. Do not add a
# second parser here.
# ------------------------------------------------------------------------------------------------
echo
echo "== H gate leg 4: decode throughput against the recorded floor =="

if [ ! -f "${THROUGHPUT_BASELINE}" ]; then
  echo "H GATE: FAIL -- no throughput baseline at ${THROUGHPUT_BASELINE}"
  echo "  Seed it once, on a known-good build, with:"
  echo "    agents/scripts/port-loop/seed-throughput-baseline.sh"
  echo "  This leg fails closed rather than passing an ungated measurement."
  exit 1
fi
if [ ! -f "${EXTRACT_PERF}" ]; then
  echo "H GATE: FAIL -- ${EXTRACT_PERF} is missing; this is a harness bug, not a task"
  exit 1
fi

# Run parameters come from the baseline file so the measurement and the floor cannot drift apart:
# tok/s depends on n_tokens and on KV-cache depth, so a comparison against a floor recorded at
# different parameters is not a comparison at all.
tp_n_tokens="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_params"]["n_tokens"])' "${THROUGHPUT_BASELINE}" 2>/dev/null)"
tp_prompt="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_params"]["prompt"])' "${THROUGHPUT_BASELINE}" 2>/dev/null)"
tp_floor_frac="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["floor_fraction"])' "${THROUGHPUT_BASELINE}" 2>/dev/null)"
if [ -z "${tp_n_tokens}" ] || [ -z "${tp_prompt}" ] || [ -z "${tp_floor_frac}" ]; then
  echo "H GATE: FAIL -- ${THROUGHPUT_BASELINE} is missing run_params or floor_fraction"
  exit 1
fi

tp_workdir="$(mktemp -d)"
trap 'rm -rf "${tp_workdir}"' EXIT
tp_regressions=()

# Gate the models that HAVE a recorded floor, and say out loud which declared ones do not.
#
# The alternative -- failing on every declared-but-unseeded model -- produces a gate that cannot
# pass until someone seeds it, and a gate that cannot pass gets worked around. So the intersection
# is computed, the difference is reported as NOT GATED, and adding a model to the gate is exactly
# one action -- seed it -- with no edit to this file.
#
# `[2026-08-06]` An earlier version of this comment said llama32_1b_int4 "cannot be seeded yet,
# because H1's refusal spec stops it compiling on the build that is installed". That was already
# false when it was written: the refusal was narrowed in 1514e553 and the model has compiled ever
# since -- measured pre-H1s with a full `make verify` that recompiled every prefill AND decode
# kernel from scratch. The claim came from doc 17's leg-4 record, which describes attempt FOUR.
# int4 is unseeded because nobody had measured it, not because it could not be measured.
#
# Before seeding it, read the caching note on `make compile` below -- for that model the compile
# step can be vacuous, and a floor recorded from stale ELFs gates nothing.
#
# Still fail-closed where it counts: a model that HAS a floor and stops reporting throughput fails,
# and an empty intersection fails rather than passing a leg that measured nothing.
tp_seeded=()
tp_unseeded=()
for m in "${THROUGHPUT_MODELS[@]}"; do
  if PL_TP_B="${THROUGHPUT_BASELINE}" PL_TP_M="${m}" python3 -c '
import json, os, sys
b = json.load(open(os.environ["PL_TP_B"]))
e = b.get("models", {}).get(os.environ["PL_TP_M"])
sys.exit(0 if e and e.get("decode_tokens_per_sec") is not None else 1)
'; then
    tp_seeded+=("${m}")
  else
    tp_unseeded+=("${m}")
  fi
done

if [ ${#tp_unseeded[@]} -gt 0 ]; then
  echo "  NOT GATED (no recorded floor): ${tp_unseeded[*]}"
  echo "    Seed on a known-good build, between phases:"
  echo "      flock -x -w 1800 /tmp/mlir-air-npu.lock \\"
  echo "        agents/scripts/port-loop/seed-throughput-baseline.sh ${tp_unseeded[*]}"
  echo "    A floor recorded AFTER a change cannot gate that change, so seed before the phase."
fi

if [ ${#tp_seeded[@]} -eq 0 ]; then
  echo "H GATE: FAIL -- no declared throughput model has a recorded floor, so this leg measured"
  echo "  nothing. A leg that passes having measured nothing is worse than no leg."
  exit 1
fi
echo "  gating ${#tp_seeded[@]} of ${#THROUGHPUT_MODELS[@]} declared model(s): ${tp_seeded[*]}"

for m in "${tp_seeded[@]}"; do
  d="${ROOT}/programming_examples/llms/${m}"
  echo "  --- ${m} ---"
  if [ ! -d "${d}" ]; then
    echo "  ${m}: MISSING directory ${d}"
    tp_regressions+=("${m} (missing)")
    continue
  fi

  if ! ( cd "${d}" && make compile ); then
    echo "  ${m}: FAILED to compile against the new compiler"
    tp_regressions+=("${m} (compile)")
    continue
  fi

  tp_log="${tp_workdir}/${m}.profile.log"
  if ! ( cd "${d}" && make profile N_TOKENS="${tp_n_tokens}" PROMPT="${tp_prompt}" ) > "${tp_log}" 2>&1; then
    echo "  ${m}: FAILED to run"
    tail -n 20 "${tp_log}"
    tp_regressions+=("${m} (run)")
    continue
  fi

  # extract_perf.py leaves a metric null rather than guessing; a null here means the driver printed
  # no throughput line at all, which is a broken measurement and not a passing one.
  if ! python3 "${EXTRACT_PERF}" "${tp_log}" --model "${m}" > "${tp_workdir}/${m}.json"; then
    echo "  ${m}: FAILED to parse the profile output"
    tp_regressions+=("${m} (parse)")
    continue
  fi

  verdict="$(PL_TP_BASELINE="${THROUGHPUT_BASELINE}" PL_TP_MEASURED="${tp_workdir}/${m}.json" \
             PL_TP_MODEL="${m}" PL_TP_FLOOR_FRAC="${tp_floor_frac}" python3 -c '
import json, os, sys

base = json.load(open(os.environ["PL_TP_BASELINE"]))
meas = json.load(open(os.environ["PL_TP_MEASURED"]))
model = os.environ["PL_TP_MODEL"]
frac = float(os.environ["PL_TP_FLOOR_FRAC"])

got = meas["metrics"]["decode_tokens_per_sec"]
if got is None:
    print("FAIL no throughput line in the profile output")
    sys.exit(0)

entry = base.get("models", {}).get(model)
if entry is None or entry.get("decode_tokens_per_sec") is None:
    print(f"FAIL no baseline entry for {model} (measured {got:.2f} tok/s)")
    sys.exit(0)

want = float(entry["decode_tokens_per_sec"])
floor = want * frac
status = "PASS" if got >= floor else "FAIL"
print(f"{status} {got:.2f} tok/s vs baseline {want:.2f} (floor {floor:.2f}, {frac:.0%})")
')"
  echo "  ${m}: ${verdict}"
  case "${verdict}" in
    PASS*) ;;
    *) tp_regressions+=("${m}") ;;
  esac
done

if [ ${#tp_regressions[@]} -gt 0 ]; then
  echo
  echo "H GATE: FAIL -- ${#tp_regressions[@]} throughput leg failure(s): ${tp_regressions[*]}"
  echo "  A correctness-clean compiler change that costs this much throughput is still a defect."
  echo "  Do NOT resolve this by lowering the floor: ${THROUGHPUT_BASELINE} is fingerprinted and"
  echo "  covered by no allowlist, so editing it halts the run."
  echo
  echo "  If the message above says 'no baseline entry', that is a HARNESS state, not a compiler"
  echo "  regression: the model is listed in THROUGHPUT_MODELS but has never been measured. Seed"
  echo "  it on a known-good build, between phases, with"
  echo "    flock -x -w 1800 /tmp/mlir-air-npu.lock \\"
  echo "      agents/scripts/port-loop/seed-throughput-baseline.sh <model>"
  echo "  and note that a floor recorded AFTER a change cannot gate that change."
  exit 1
fi

echo
echo "== H gate leg 5: cross-deployment regression over ${#MODELS[@]} shipped models =="
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
