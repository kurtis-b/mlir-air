# shellcheck shell=bash
#
# The phase table: what each phase is, how it is gated, and what the driver checks itself.
#
# Contract: every phase declares four things —
#   doc              the specification a fresh session is pointed at
#   needs_hardware   whether the gate touches the NPU (drives flock + XRT sourcing)
#   gate_cmd         the phase's own gate, run by the DRIVER, never self-reported
#   objective_check  a driver-side assertion about build products that a session cannot satisfy
#                    by writing a more permissive test
#
# Why objective_check exists: Phase A creates the very lit suite that gates it, so "the suite
# passed" proves nothing on its own. The objective check inspects the actual .o files and their
# exported symbols. A stub test passes the suite and fails this.
#
# Footgun: gate_allowlist is a regex of gate-file paths the phase is permitted to create or
# change. Anything outside it trips the tamper check and halts the run. Widen it deliberately,
# never reflexively.

PL_PHASES_IN_SCOPE='["C1","C2","C3","C4"]'

phase_name() {
  case "$1" in
    A) echo "AIE2P device kernels" ;;
    B) echo "Runtime seam (runlist aggregation + BO pooling)" ;;
    C1) echo "Operators: check mechanism + causal_mask/addnorm/layer_norm/elementwise_add" ;;
    C2) echo "Operators: qkv_proj + ffn" ;;
    C3) echo "Operators: mha_out_proj" ;;
    C4) echo "Operators: registry coverage sweep" ;;
    *) echo "unknown" ;;
  esac
}

phase_doc() {
  case "$1" in
    A) echo "docs/plans/transformer-layer-execution-studies/04-phase-a-kernels.md" ;;
    B) echo "docs/plans/transformer-layer-execution-studies/05-phase-b-runtime-seam.md" ;;
    C1) echo "docs/plans/transformer-layer-execution-studies/06a-phase-c1-gate-and-small-operators.md" ;;
    C2) echo "docs/plans/transformer-layer-execution-studies/06b-phase-c2-qkv-proj-and-ffn.md" ;;
    C3) echo "docs/plans/transformer-layer-execution-studies/06c-phase-c3-mha-out-proj.md" ;;
    C4) echo "docs/plans/transformer-layer-execution-studies/06d-phase-c4-coverage-sweep.md" ;;
    *) echo "" ;;
  esac
}

# Phase C is four sub-phases rather than one because Phase B was 3,725 lines and took 362 minutes
# with blocking findings still open after round 3; Phase C's source material is 8,160 lines across
# five rewrites. Each sub-phase points at its OWN document: the implement prompt injects the doc
# as the session's entire task list, so four sessions sharing one doc would each try to do
# everything.

phase_needs_hardware() {
  case "$1" in
    A) echo "no" ;;
    B) echo "yes" ;;
    C1|C2|C3|C4) echo "yes" ;;
    *) echo "no" ;;
  esac
}

phase_gate_description() {
  case "$1" in
    A) cat <<'EOF'
ninja -C build-xrt check-programming-examples-transformer-layer

Compile-only; no NPU required. In addition the driver independently verifies that each ported
kernel produced a non-trivial object file and that the expected extern "C" symbols are present.
A lit test that passes without actually compiling the kernels will fail that check.
EOF
;;
    B) cat <<'EOF'
A hardware test using the exact separately-compiled ELF artifacts the study will use, showing a
multi-ELF runlist that is (1) numerically identical to sequential dispatch and (2) measurably
lower latency.

Because this phase modifies programming_examples/llms/shared/infra/cache.py, the gate also runs
the cross-deployment regression check: make verify on the shipped models, serialized under flock.
EOF
;;
    C1|C2|C3) cat <<'EOF'
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  ninja -C build-xrt check-programming-examples-transformer-layer

Every test in the suite passes, on real hardware. That includes the tests earlier sub-phases added
and the pre-existing compile, seam and runlist-gate tests.

The driver then runs an objective check you cannot influence by changing a test. It reads the
machine-readable results artifact each operator writes, and:

  - requires every results file to be NEWER than the gate's start stamp;
  - re-derives the verdict rather than trusting the file's own `passed` flag: zero np.isclose
    mismatches over the full output, ref_dtype == "float32", rtol == 1.6e-2 exactly, atol <= 1e-1;
  - re-runs each operator with `--fault-inject input` and REQUIRES that run to FAIL.

That last one is the layer a laxer test cannot satisfy. A reference compared against itself, a
tolerance wide enough to swallow anything, or an ignored --fault-inject flag all still pass under
injection, and the sub-phase fails for it.
EOF
;;
    C4) cat <<'EOF'
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  ninja -C build-xrt check-programming-examples-transformer-layer

plus the cross-deployment regression check, because this sub-phase writes rows into
kernel_registry that the ten shipped LLM deployments resolve against: make verify in each
programming_examples/llms/<model>/, serialized under flock.

The driver then checks, independently of anything you write:

  - the 36 baseline_768 shapes resolve through registry_lookup.gemm_config() without raising;
  - every registry shape present at this sub-phase's BASE COMMIT is byte-identical afterwards
    (the sweep is append-only; re-measuring an existing shape into a different winner would change
    shipped-model behaviour without anyone asking);
  - the registry JSON is newer than the gate's start stamp.
EOF
;;
    *) cat <<'EOF'
ERROR: no gate description is declared for this phase in agents/scripts/port-loop/phases.sh.
This is a harness bug, not a task. Stop and report it as a blocker.
EOF
;;
  esac
}

# Gate-file paths this phase may legitimately create or modify.
phase_gate_allowlist() {
  case "$1" in
    A) echo '^programming_examples/(transformer_layer/|CMakeLists\.txt$)' ;;
    B) echo '^programming_examples/transformer_layer/' ;;
    C1|C2|C3) echo '^programming_examples/transformer_layer/' ;;
    # C4 alone may write the registry JSON. guard_gate_files() fingerprints
    # kernel_registry/details/*.json, so the sweep trips the tamper check without this. Note it
    # does NOT fingerprint supported_kernels.md or details/*.md -- those rows are covered by the
    # objective check instead.
    C4) echo '^programming_examples/(transformer_layer/|kernel_registry/details/.*\.json$)' ;;
    *) echo '' ;;
  esac
}

phase_gate_cmd() {
  case "$1" in
    A) echo "ninja -C ${PL_ROOT}/build-xrt check-programming-examples-transformer-layer" ;;
    B) echo "flock -x -w 1800 /tmp/mlir-air-npu.lock ninja -C ${PL_ROOT}/build-xrt check-programming-examples-transformer-layer" ;;
    C1|C2|C3) echo "flock -x -w 1800 /tmp/mlir-air-npu.lock ninja -C ${PL_ROOT}/build-xrt check-programming-examples-transformer-layer" ;;
    C4) echo "flock -x -w 1800 /tmp/mlir-air-npu.lock ${PL_LIB}/gate-c4.sh" ;;
    *) echo "false" ;;
  esac
}

# --- Objective checks -------------------------------------------------------------------------

# Phase A: the ported kernels must have produced real object files with the expected symbols.
# Deliberately independent of lit, Makefiles, and anything the session authored.
#
# Footgun that bit during bring-up: lit does NOT build in the source tree. Its working directory
# is under the CMake build tree, so objects land in
#   build-xrt/test/transformer_layer/<lit-workdir>/build_peano/*.o
# Searching only programming_examples/ finds nothing and fails a perfectly good phase.
#
# Freshness matters as much as existence: a stale object from an earlier run would otherwise
# satisfy this check even if the gate had just compiled nothing. run_gate() stamps
# _GATE_STARTED_AT, and objects must be newer than it.
phase_a_objective_check() {
  local kdir="${PL_ROOT}/programming_examples/transformer_layer/kernels"
  if [ ! -d "${kdir}" ]; then
    log_error "objective check: ${kdir} does not exist — no kernels were ported"
    return 1
  fi

  local search_roots=(
    "${PL_ROOT}/programming_examples/transformer_layer"
    "${PL_ROOT}/build-xrt/test/transformer_layer"
    "${PL_ROOT}/build/test/transformer_layer"
  )
  local existing=() r
  for r in "${search_roots[@]}"; do [ -d "${r}" ] && existing+=("${r}"); done
  if [ ${#existing[@]} -eq 0 ]; then
    log_error "objective check: no build output directory for transformer_layer kernels"
    return 1
  fi


  # The objects under inspection are aie2p ELFs ("unknown arch 0x108" to a host toolchain).
  # Reading them with whatever `nm` happens to be on PATH is a bet on host binutils being
  # lenient about an architecture it does not know; a host nm that rejects the format prints
  # nothing, every symbol reads as missing, and a perfectly good phase fails. Peano ships the
  # llvm-nm that actually targets these objects, so prefer it — the same choice
  # compile_kernels.py::_llvm_nm() already makes.
  local nm_tool=""
  local c
  for c in "${PEANO_INSTALL_DIR:+${PEANO_INSTALL_DIR}/bin/llvm-nm}" \
           "${PL_ROOT}/sandbox/lib/python3.12/site-packages/llvm-aie/bin/llvm-nm"; do
    if [ -n "${c}" ] && [ -x "${c}" ]; then
      nm_tool="${c}"
      break
    fi
  done
  [ -n "${nm_tool}" ] || nm_tool="$(command -v llvm-nm 2>/dev/null || true)"
  [ -n "${nm_tool}" ] || nm_tool="$(command -v nm 2>/dev/null || true)"
  if [ -z "${nm_tool}" ]; then
    log_error "objective check: no llvm-nm or nm available to read the aie2p objects"
    return 1
  fi
  log_info "objective check: reading symbols with ${nm_tool}"

  # Every extern "C" symbol declared in every kernel source must be defined by some object.
  # The expectation is derived from the SOURCES, independently of compile_kernels.py, so a
  # weakened test script cannot narrow what is demanded here.
  PL_KDIR="${kdir}" PL_OBJ_ROOTS="${existing[*]}" PL_GATE_STAMP="${_GATE_STARTED_AT:-}" \
  PL_NM="${nm_tool}" python3 -c '
import os, re, pathlib, subprocess, sys

kdir = pathlib.Path(os.environ["PL_KDIR"])
roots = os.environ["PL_OBJ_ROOTS"].split()

expected = {}
for f in sorted(kdir.glob("*.cc")):
    src = f.read_text()
    names = []
    for m in re.finditer(r"extern\s+\"C\"\s*\{", src):
        i = m.end(); depth = 1; j = i
        while j < len(src) and depth:
            if src[j] == "{": depth += 1
            elif src[j] == "}": depth -= 1
            j += 1
        names += re.findall(r"^\s*(?:void|int|float)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
                            src[i:j-1], re.M)
    if names:
        expected[f.name] = sorted(set(names))

if not expected:
    print("objective check: no extern \"C\" symbols found in kernel sources", file=sys.stderr)
    sys.exit(1)

# Only objects the gate itself rebuilt count. Collecting symbols globally would let a stale
# object from an earlier run supply a symbol whose kernel the current gate never built, while a
# single freshly rebuilt object satisfied a per-run freshness test. Symbols must come from
# artifacts this gate produced, or the check proves nothing.
stamp = os.environ.get("PL_GATE_STAMP") or ""
cutoff = os.path.getmtime(stamp) if stamp and os.path.exists(stamp) else None

objs, stale = [], 0
for r in roots:
    for dp, _, fns in os.walk(r):
        for fn in fns:
            if not fn.endswith(".o"):
                continue
            path = os.path.join(dp, fn)
            if cutoff is not None and os.path.getmtime(path) < cutoff:
                stale += 1
                continue
            objs.append(path)

if cutoff is None:
    print("objective check: no gate timestamp available; cannot prove objects are fresh",
          file=sys.stderr)
    sys.exit(1)
if not objs:
    print("objective check: the gate rebuilt no object files (%d stale ignored)" % stale,
          file=sys.stderr)
    sys.exit(1)
print("  considering %d object(s) rebuilt by this gate (%d stale ignored)" % (len(objs), stale))

nm = os.environ["PL_NM"]

# Fail closed on an unreadable object. A symbol reader that errors out still returns an
# empty stdout, which is indistinguishable from "this object defines nothing" -- and that
# is exactly the shape of a false failure that invites someone to relax the check.
defined = set()
for o in objs:
    try:
        p = subprocess.run([nm, "--defined-only", o], capture_output=True, text=True)
    except OSError as e:
        print("objective check: cannot run %s: %s" % (nm, e), file=sys.stderr); sys.exit(1)
    if p.returncode != 0:
        print("objective check: %s could not read %s (rc=%d): %s"
              % (nm, o, p.returncode, p.stderr.strip()), file=sys.stderr)
        sys.exit(1)
    out = p.stdout
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] in ("T", "t", "W", "w"):
            defined.add(parts[2])

rc = 0
for srcname, syms in expected.items():
    missing = [s for s in syms if s not in defined]
    if missing:
        print("objective check: %s declares %d extern \"C\" symbols; %d missing from every "
              "object this gate rebuilt: %s" % (srcname, len(syms), len(missing), ", ".join(missing[:6])),
              file=sys.stderr)
        rc = 1
    else:
        print("  %s: all %d extern \"C\" symbols present" % (srcname, len(syms)))
sys.exit(rc)
' || { log_error "objective check FAILED: kernel sources declare symbols no object defines"; return 1; }

  log_info "objective check passed: every extern \"C\" symbol in every kernel source is defined"
  return 0
}

# Phase B: the spike must have produced a recorded result, and cache.py must actually have grown
# a runlist path rather than the phase having been declared done on the strength of a comment.
phase_b_objective_check() {
  local cache="${PL_ROOT}/programming_examples/llms/shared/infra/cache.py"
  if ! grep -q "runlist" "${cache}" 2>/dev/null; then
    log_error "objective check: no runlist support appears in ${cache}"
    return 1
  fi
  local spike
  spike="$(find "${PL_ROOT}/programming_examples/transformer_layer" -name '*runlist*spike*' 2>/dev/null | head -1)"
  if [ -z "${spike}" ]; then
    log_warn "objective check: no runlist spike artifact found; relying on the gate alone"
  fi
  log_info "objective check passed: runlist path present in shared infra"
  return 0
}

# --- Phase C -----------------------------------------------------------------------------------
#
# Phase C's sessions author both the builder and its reference, so reading back their own verdict
# proves nothing -- the Phase B check's shape (grep for a string, warn if an artifact is absent) is
# exactly what not to do here.
#
# Three layers, in increasing order of how hard they are to satisfy dishonestly:
#
#   1. freshness   -- results must be newer than the gate stamp, so a stale file does not count.
#   2. re-derived  -- the driver recomputes the verdict from n_mismatch / ref_dtype / rtol / atol
#                     rather than trusting `passed`, and pins rtol to the repository's canonical
#                     bf16 value with atol bounded by the loosest tolerance in kernel_registry.
#   3. NEGATIVE CONTROL -- the driver re-runs each operator with a fault injected into its input
#                     and REQUIRES failure. A vacuous check (reference vs itself, a tolerance wide
#                     enough to swallow anything, an ignored --fault-inject flag) still PASSES
#                     under injection, and that fails the phase. This is the layer that cannot be
#                     satisfied by making the test laxer, and it is the harness's own lesson from
#                     Phase A: test the negative path before trusting a fix.
#
# The contract opcheck.py must satisfy is specified in
# docs/plans/transformer-layer-execution-studies/06a-phase-c1-gate-and-small-operators.md.

# Canonical bf16 rtol, fixed repository-wide (kernel_registry/README.md), and the loosest atol
# anywhere in the registry (FlashAttention's). A session cannot widen either.
PL_C_RTOL="1.6e-2"
PL_C_ATOL_MAX="1e-1"

# phase_c_operator_check <operator> [<operator> ...]
phase_c_operator_check() {
  local opcheck="${PL_ROOT}/programming_examples/transformer_layer/opcheck.py"
  if [ ! -f "${opcheck}" ]; then
    log_error "objective check: ${opcheck} does not exist — the check mechanism was not built"
    return 1
  fi
  if [ -z "${_GATE_STARTED_AT:-}" ] || [ ! -e "${_GATE_STARTED_AT}" ]; then
    log_error "objective check: no gate timestamp; cannot prove results are fresh"
    return 1
  fi

  local listing
  if ! listing="$(cd "${PL_ROOT}/programming_examples/transformer_layer" && python3 "${opcheck}" --list 2>&1)"; then
    log_error "objective check: 'opcheck.py --list' failed: ${listing}"
    return 1
  fi

  # Layers 1 and 2, over every (operator, shape) the session declares.
  PL_C_LISTING="${listing}" \
  PL_C_RESULTS="${PL_ROOT}/programming_examples/transformer_layer/results" \
  PL_C_STAMP="${_GATE_STARTED_AT}" \
  PL_C_EXPECTED="$*" \
  PL_C_RTOL="${PL_C_RTOL}" PL_C_ATOL_MAX="${PL_C_ATOL_MAX}" \
  python3 -c '
import json, os, pathlib, sys

listing = json.loads(os.environ["PL_C_LISTING"])
results = pathlib.Path(os.environ["PL_C_RESULTS"])
cutoff  = os.path.getmtime(os.environ["PL_C_STAMP"])
expected = os.environ["PL_C_EXPECTED"].split()
rtol_req = float(os.environ["PL_C_RTOL"])
atol_max = float(os.environ["PL_C_ATOL_MAX"])

if not listing:
    print("objective check: opcheck.py --list declared no operators", file=sys.stderr)
    sys.exit(1)

declared = {e["operator"] for e in listing}
missing = [o for o in expected if o not in declared]
if missing:
    print("objective check: this sub-phase must land %s; opcheck.py --list declares only %s"
          % (", ".join(missing), ", ".join(sorted(declared))), file=sys.stderr)
    sys.exit(1)

rc = 0
for entry in listing:
    op, key = entry["operator"], entry["shape_key"]
    f = results / ("%s__%s.json" % (op, key))
    if not f.exists():
        print("objective check: %s: no results file %s" % (op, f), file=sys.stderr)
        rc = 1
        continue
    if f.stat().st_mtime < cutoff:
        print("objective check: %s: %s predates the gate; it was not produced by this run"
              % (op, f.name), file=sys.stderr)
        rc = 1
        continue
    try:
        d = json.loads(f.read_text())
    except Exception as e:
        print("objective check: %s: %s is not readable JSON: %s" % (op, f.name, e), file=sys.stderr)
        rc = 1
        continue

    # Re-derive rather than trusting d["passed"].
    problems = []
    if d.get("fault_injected") is not None:
        problems.append("is a fault-injected run (fault_injected=%r)" % d["fault_injected"])
    if d.get("ref_dtype") != "float32":
        problems.append("ref_dtype=%r, not float32" % d.get("ref_dtype"))
    if d.get("n_mismatch") != 0:
        problems.append("n_mismatch=%r, not 0" % d.get("n_mismatch"))
    if not isinstance(d.get("n_elements"), int) or d["n_elements"] <= 0:
        problems.append("n_elements=%r" % d.get("n_elements"))
    try:
        if abs(float(d["rtol"]) - rtol_req) > 1e-12:
            problems.append("rtol=%r, not the canonical %g" % (d["rtol"], rtol_req))
    except (KeyError, TypeError, ValueError):
        problems.append("rtol missing or not a number")
    try:
        if float(d["atol"]) > atol_max:
            problems.append("atol=%r exceeds the registry maximum %g" % (d["atol"], atol_max))
    except (KeyError, TypeError, ValueError):
        problems.append("atol missing or not a number")

    if problems:
        print("objective check: %s [%s]: %s" % (op, key, "; ".join(problems)), file=sys.stderr)
        rc = 1
    else:
        print("  %s [%s]: %d elements, 0 mismatches, rtol=%g atol=%g, fp32 reference"
              % (op, key, d["n_elements"], float(d["rtol"]), float(d["atol"])))
sys.exit(rc)
' || { log_error "objective check FAILED: results do not satisfy the numerics contract"; return 1; }

  # Layer 3: the negative control. One shape per operator is enough -- what is being tested is
  # whether the check discriminates at all, not the shape.
  local op key
  for op in "$@"; do
    key="$(printf '%s' "${listing}" | python3 -c '
import json, sys
op = sys.argv[1]
for e in json.load(sys.stdin):
    if e["operator"] == op:
        print(e["shape_key"]); break
' "${op}")"
    if [ -z "${key}" ]; then
      log_error "objective check: no shape declared for operator ${op}"
      return 1
    fi
    log_info "objective check: negative control ${op} [${key}] — this run MUST fail"
    if ( cd "${PL_ROOT}/programming_examples/transformer_layer" \
         && flock -x -w 1800 /tmp/mlir-air-npu.lock \
              python3 "${opcheck}" --operator "${op}" --shape-key "${key}" \
                                   --fault-inject input ) >/dev/null 2>&1; then
      log_error "objective check FAILED: ${op} PASSED with a fault injected into its input."
      log_error "  The check does not discriminate: it would pass on a broken kernel too."
      return 1
    fi
    log_info "  ${op}: correctly failed under injection"
  done

  log_info "objective check passed: fresh results, re-derived verdicts, negative control fails as it must"
  return 0
}

phase_c1_objective_check() {
  phase_c_operator_check causal_mask addnorm layer_norm elementwise_add
}

phase_c2_objective_check() {
  phase_c_operator_check qkv_proj ffn
}

phase_c3_objective_check() {
  phase_c_operator_check mha_out_proj
}

# C4 is coverage, not numerics: the sweep must have made the shapes resolvable WITHOUT having
# rewritten any shape the shipped deployments already depend on.
phase_c4_objective_check() {
  local json="${PL_ROOT}/programming_examples/kernel_registry/details/GEMM_bf16_in_bf16_out.json"
  if [ ! -f "${json}" ]; then
    log_error "objective check: ${json} does not exist"
    return 1
  fi
  # Proof of work is taken from GIT, not from mtime.
  #
  # This check originally required the registry JSON to be newer than the gate's start stamp, by
  # analogy with Phase A. That analogy is wrong and the requirement was unsatisfiable. Phase A's
  # gate REBUILDS the objects it inspects, so they are necessarily newer than the stamp. C4's
  # sweep runs in the implement session, hours before the gate, and gate-c4.sh deliberately does
  # not re-sweep -- so the JSON is always older than the stamp, and no honest run could pass.
  #
  # The C4 session diagnosed this in review and said so rather than running `touch` to get past
  # it, which is the behaviour the harness is built to elicit. Note that mtime was a weak proof
  # anyway: one `touch` forges it. "The staged shapes were absent at the phase base commit and are
  # present now" is checkable from git, unforgeable by a filesystem timestamp, and is what the
  # requirement was actually reaching for.
  local base="${_START_SHA:-$(state_start_sha)}"
  if [ -z "${base}" ]; then
    log_error "objective check: no phase base commit; cannot prove the sweep was append-only"
    return 1
  fi

  local baseline
  baseline="$(git -C "${PL_ROOT}" show "${base}:programming_examples/kernel_registry/details/GEMM_bf16_in_bf16_out.json" 2>/dev/null)" || {
    log_error "objective check: cannot read the registry JSON at base commit ${base}"
    return 1
  }

  PL_C4_BASELINE="${baseline}" PL_C4_JSON="${json}" PL_ROOT="${PL_ROOT}" \
  python3 -c '
import json, os, sys

sys.path.insert(0, os.path.join(os.environ["PL_ROOT"], "programming_examples"))

before = {(s["M"], s["K"], s["N"]): s for s in json.loads(os.environ["PL_C4_BASELINE"])["shapes"]}
after  = {(s["M"], s["K"], s["N"]): s for s in json.load(open(os.environ["PL_C4_JSON"]))["shapes"]}

rc = 0

# Append-only: every shape the shipped deployments already resolve against must be untouched.
for key, entry in before.items():
    if key not in after:
        print("objective check: shape %dx%dx%d was REMOVED from the registry" % key, file=sys.stderr)
        rc = 1
    elif after[key] != entry:
        print("objective check: shape %dx%dx%d was REWRITTEN; the sweep must be append-only "
              "(the ten shipped models resolve against these rows)" % key, file=sys.stderr)
        rc = 1
if rc:
    sys.exit(rc)
print("  %d pre-existing shape(s) unchanged; %d added" % (len(before), len(after) - len(before)))

LADDER = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
ROLES = {"qkv_proj": (768, 2304), "ffn_up": (768, 3072),
         "ffn_down": (3072, 768), "o_proj": (768, 768)}
STAGED = {(m, k, n) for (k, n) in ROLES.values() for m in LADDER}

# Proof of work, from git: this phase must have ADDED staged shapes. A run that inherited an
# already-swept registry and did nothing would satisfy every other clause here.
gained = STAGED - set(before)
if not gained:
    print("objective check: all %d staged shapes were already registered at the base commit — "
          "this phase swept nothing" % len(STAGED), file=sys.stderr)
    sys.exit(1)
print("  %d of %d staged shapes were absent at the base commit and are present now"
      % (len(gained), len(STAGED)))

# Coverage: the staged baseline_768 family must resolve for real, through the same entry point
# the builders use.
from kernel_registry.registry_lookup import gemm_config

missing = []
for role, (k, n) in ROLES.items():
    for m in LADDER:
        try:
            gemm_config(m, k, n)
        except Exception as e:
            missing.append("%s %dx%dx%d (%s)" % (role, m, k, n, type(e).__name__))
if missing:
    print("objective check: %d of 36 baseline_768 shapes do not resolve: %s"
          % (len(missing), ", ".join(missing[:6])), file=sys.stderr)
    sys.exit(1)
print("  36/36 baseline_768 shapes resolve through gemm_config()")
' || { log_error "objective check FAILED: registry coverage or append-only invariant violated"; return 1; }

  log_info "objective check passed: registry is append-only and the staged family resolves"
  return 0
}

phase_objective_check() {
  case "$1" in
    A) phase_a_objective_check ;;
    B) phase_b_objective_check ;;
    C1) phase_c1_objective_check ;;
    C2) phase_c2_objective_check ;;
    C3) phase_c3_objective_check ;;
    C4) phase_c4_objective_check ;;
    *) return 0 ;;
  esac
}
