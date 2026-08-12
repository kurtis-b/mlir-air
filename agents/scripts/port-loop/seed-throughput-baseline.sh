#!/usr/bin/env bash
#
# Seed (or re-seed) gate-h.sh leg 4's throughput floor.
#
# OPERATOR TOOL, RUN BETWEEN PHASES -- never during a run, and never by a session. This file and
# the baseline it writes both sit under agents/scripts/port-loop/, which guard_gate_files()
# fingerprints and which no phase's allowlist covers, so any change to either during a phase halts
# it. That is the point: leg 4's floor must not be movable by the thing being gated.
#
# `[2026-08-12]` IT REFUSES TO SEED OFF TURBO, and records the mode it did seed at. The pmode is
# non-persistent and resets to `Default` on every reboot and every amdxdna reload, and at `Default`
# this host measures ~15-20x slow -- so a floor seeded without checking is an unconditioned number
# that looks like a conditioned one, and gate-h leg 4 would then pass anything. See pmode_guard.py.
#
# WHEN TO RE-SEED. Only when the measurement is legitimately supposed to change and you can say why
# in a sentence -- a deliberate performance change that lands, or new hardware. Re-seeding because
# the gate failed is exactly the move this arrangement exists to prevent; a failing leg 4 is a
# finding to report, not a number to update.
#
# WHAT IT MEASURES. Decode throughput (tok/s) for the two models whose decode path is governed by
# the backend settings ping-pong decides, recompiled against the CURRENT installed compiler so the
# floor describes this toolchain rather than the ELFs some earlier one left in the cache.
#
# Run it holding the NPU lock, the same way the gate does:
#   flock -x -w 1800 /tmp/mlir-air-npu.lock agents/scripts/port-loop/seed-throughput-baseline.sh

set -uo pipefail

ROOT="${PL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PL_ROOT="${ROOT}"

OUT="${HERE}/throughput-baseline.json"
EXTRACT_PERF="${ROOT}/programming_examples/llms/bench/extract_perf.py"

# Reuse the driver's own toolchain resolution rather than a second copy of it. env_setup.sh is not
# idempotent and must be sourced, not executed; pl_env_ensure() already knows both and gates on
# tool presence. It calls log_error, which normally comes from the driver, so supply a fallback.
log_error() { echo "seed: $*" >&2; }
# shellcheck disable=SC1091
. "${HERE}/lib-env.sh"
if ! pl_env_ensure; then
  echo "seed: could not put aircc/air-opt/aiecc on PATH -- see 15-environment-notes.md" >&2
  exit 1
fi
# Separate call, and not optional: env_setup.sh gets the COMPILER onto PATH, and /opt/xilinx/xrt's
# setup.sh gets pyxrt onto PYTHONPATH. `make compile` needs only the first, so a seeding run
# missing the second gets all the way through both compiles before dying on
# `ModuleNotFoundError: No module named 'pyxrt'` at the first dispatch.
if ! pl_env_ensure_xrt; then
  echo "seed: XRT is not available, so nothing can be measured on hardware" >&2
  exit 1
fi

# Must match THROUGHPUT_MODELS in gate-h.sh. Kept as a literal in both rather than shared through a
# third file: two short lists that a reader can see are the same beat one indirection they have to
# go and resolve.
#
# Seeding is INCREMENTAL and per-model: pass model names as arguments to record just those, and
# they are merged into whatever the baseline already holds. Each model carries the SHA it was
# measured at, so a floor recorded later is attributable rather than silently equal in standing to
# one that spans the change it is supposed to gate.
#
# `[2026-08-06]` An earlier version of this comment justified that with "H1's refusal spec stops
# llama32_1b_int4, qwen3_0_6b and qwen3_1_7b compiling at all". That was already false when it was
# written -- the refusal was narrowed in 1514e553 and all three have compiled since, measured
# pre-H1s by a full `make verify` that recompiled every prefill and decode kernel. The claim came
# from doc 17's leg-4 record, which describes attempt FOUR, not the installed build. The
# per-model SHA is worth keeping on its own merits; the reason given for it was wrong.
#
# Before seeding llama32_1b_int4 specifically: its prefill script skips any kernel whose .elf
# already exists (llama32_1b_int4_prefill.py:1058), so `make compile` is vacuous on a warm cache
# and would record a floor from the PREVIOUS compiler's ELFs. Clear its cache dirs first and use
# `make compile-inference`, which is also the only target that builds the decode kernels
# `make profile` dispatches.
MODELS=(llama32_1b llama32_1b_int4)
if [ "$#" -gt 0 ]; then
  MODELS=("$@")
fi

# tok/s depends on how many tokens were generated and on KV-cache depth, so the floor is only
# meaningful at fixed parameters. They are recorded IN the baseline and read back by the gate, so
# the two cannot drift apart.
N_TOKENS=32
PROMPT="What is the capital of France?"

# How far below the recorded number the gate tolerates before calling it a regression. 0.85 is
# chosen against the one regression this leg exists to catch: dropping ping-pong cost 12.4 -> 7.8
# tok/s, a 37% fall, which this catches by a wide margin, while leaving room for the few percent of
# run-to-run variation a decode loop on a shared NPU shows. It is not a performance target.
FLOOR_FRACTION=0.85

if [ ! -f "${EXTRACT_PERF}" ]; then
  echo "seed: ${EXTRACT_PERF} is missing" >&2
  exit 1
fi

# `[2026-08-12]` THE POWER MODE IS PART OF WHAT A FLOOR MEANS, so it is checked before the
# measurement and recorded with it. A floor seeded at `Default` is ~15-20x below the same build's
# Turbo number on this host (README trap 0), and gate-h leg 4 would then pass anything -- an
# unconditioned floor is worse than no floor, because it looks like one. Refusing here is the
# cheaper half of the fix; stamping npu_power_mode into each entry is the half that makes the
# comparison checkable afterwards.
PMODE_GUARD="${HERE}/pmode_guard.py"
if [ ! -f "${PMODE_GUARD}" ]; then
  echo "seed: ${PMODE_GUARD} is missing; refusing to record a floor whose power mode is unknown" >&2
  exit 1
fi
if ! python3 "${PMODE_GUARD}" require --where "seeding the throughput floor"; then
  echo "seed: refusing to record a floor measured off Turbo. This is the operator's action and it" >&2
  echo "  does not persist across a reboot or an amdxdna reload." >&2
  exit 1
fi
SEED_PMODE="$(python3 "${PMODE_GUARD}" observe)"

workdir="$(mktemp -d)"
trap 'rm -rf "${workdir}"' EXIT

echo "Seeding throughput baseline at n_tokens=${N_TOKENS}, prompt=\"${PROMPT}\""
echo "Compiler under measurement: $(command -v aircc || echo 'aircc NOT ON PATH')"
echo

for m in "${MODELS[@]}"; do
  d="${ROOT}/programming_examples/llms/${m}"
  echo "--- ${m} ---"
  if [ ! -d "${d}" ]; then
    echo "seed: missing ${d}" >&2
    exit 1
  fi
  if ! ( cd "${d}" && make compile ); then
    echo "seed: ${m} failed to compile; refusing to record a baseline from a broken build" >&2
    exit 1
  fi
  if ! ( cd "${d}" && make profile N_TOKENS="${N_TOKENS}" PROMPT="${PROMPT}" ) \
       > "${workdir}/${m}.profile.log" 2>&1; then
    echo "seed: ${m} failed to run; see ${workdir}/${m}.profile.log" >&2
    tail -n 30 "${workdir}/${m}.profile.log" >&2
    exit 1
  fi
  if ! python3 "${EXTRACT_PERF}" "${workdir}/${m}.profile.log" --model "${m}" \
       > "${workdir}/${m}.json"; then
    echo "seed: could not parse ${m}'s profile output" >&2
    exit 1
  fi
  echo "  $(python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))["metrics"]
tps, ttft, ctx = d["decode_tokens_per_sec"], d["ttft_ms"], d["context_len"]
print(f"{tps} tok/s, ttft {ttft} ms, ctx {ctx}")
' "${workdir}/${m}.json")"
done

PL_SEED_DIR="${workdir}" PL_SEED_OUT="${OUT}" PL_SEED_MODELS="${MODELS[*]}" \
PL_SEED_NTOK="${N_TOKENS}" PL_SEED_PROMPT="${PROMPT}" PL_SEED_FRAC="${FLOOR_FRACTION}" \
PL_SEED_SHA="$(git -C "${ROOT}" rev-parse --short HEAD)" PL_SEED_PMODE="${SEED_PMODE}" \
python3 -c '
import datetime, json, os

out_path = os.environ["PL_SEED_OUT"]
n_tokens = int(os.environ["PL_SEED_NTOK"])
prompt = os.environ["PL_SEED_PROMPT"]
frac = float(os.environ["PL_SEED_FRAC"])
now = (datetime.datetime.now(datetime.timezone.utc)
       .replace(microsecond=0).isoformat().replace("+00:00", "Z"))

out = {}
if os.path.exists(out_path):
    out = json.load(open(out_path))

# The run parameters are what make a floor comparable to a measurement. Changing them invalidates
# every entry recorded under the old ones, so refuse a partial re-seed that would leave two sets of
# parameters in one file claiming equal standing.
old = out.get("run_params")
if old and (old.get("n_tokens") != n_tokens or old.get("prompt") != prompt):
    old_n, old_p = old.get("n_tokens"), old.get("prompt")
    raise SystemExit(
        "seed: this baseline was recorded at "
        f"n_tokens={old_n} prompt={old_p!r}, and you are seeding at "
        f"n_tokens={n_tokens} prompt={prompt!r}. tok/s depends on both, so the existing entries "
        "would no longer be comparable. Delete the file and re-seed every model, or seed at the "
        "recorded parameters."
    )

models = out.setdefault("models", {})
for m in os.environ["PL_SEED_MODELS"].split():
    d = json.load(open(os.path.join(os.environ["PL_SEED_DIR"], m + ".json")))
    tps = d["metrics"]["decode_tokens_per_sec"]
    if tps is None:
        raise SystemExit(f"seed: {m} printed no throughput line; refusing to record a null floor")
    models[m] = {
        "decode_tokens_per_sec": tps,
        "ttft_ms": d["metrics"]["ttft_ms"],
        "context_len": d["metrics"]["context_len"],
        "recorded_utc": now,
        "recorded_at_sha": os.environ["PL_SEED_SHA"],
        # The measurement condition, observed rather than assumed. The guard above
        # already refused anything but turbo; this records WHICH turbo run it was, so
        # gate-h leg 4 can refuse a comparison across a pmode change instead of taking
        # a verdict from one.
        "npu_power_mode": os.environ["PL_SEED_PMODE"],
    }

out["_comment"] = (
    "gate-h.sh leg 4 floor. Driver-owned: fingerprinted by guard_gate_files() and covered by no "
    "phase allowlist, so a session cannot move it. Re-seed with "
    "agents/scripts/port-loop/seed-throughput-baseline.sh, and only when the measurement is "
    "legitimately supposed to change -- never because the gate failed. Each model carries the SHA "
    "it was measured at: a floor recorded after a change cannot gate that change, and the "
    "npu_power_mode it was measured at, because tok/s on this host differs ~15-20x across "
    "Turbo/Default and a floor compared across that boundary is not a comparison."
)
# The note explaining the pre-2026-08-12 `unknown` entries is only true while one exists. Drop it
# once every entry carries an observed mode, so the file cannot end up documenting a state it is no
# longer in -- a stale caveat is read as a live one.
if all(e.get("npu_power_mode", "unknown") != "unknown" for e in models.values()):
    out.pop("_comment_npu_power_mode", None)

out["run_params"] = {"n_tokens": n_tokens, "prompt": prompt}
out["floor_fraction"] = frac
out["last_seeded_utc"] = now

with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
    f.write("\n")
print("wrote " + out_path + " (" + ", ".join(sorted(models)) + ")")
'
