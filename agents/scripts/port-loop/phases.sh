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

PL_PHASES_IN_SCOPE='["E1","E2","E3","E4","E5"]'

phase_name() {
  case "$1" in
    A) echo "AIE2P device kernels" ;;
    B) echo "Runtime seam (runlist aggregation + BO pooling)" ;;
    C1) echo "Operators: check mechanism + causal_mask/addnorm/layer_norm/elementwise_add" ;;
    C2) echo "Operators: qkv_proj + ffn" ;;
    C3) echo "Operators: mha_out_proj" ;;
    C4) echo "Operators: registry coverage sweep" ;;
    D1) echo "Block integration: operators at baseline_768" ;;
    D2) echo "Block integration: the encoder_bert layer gate" ;;
    E1) echo "Execution strategies: unblock the sequence ladder" ;;
    E2) echo "Execution strategies: coarse + the shared dispatch instrumentation" ;;
    E3) echo "Execution strategies: offload" ;;
    E4) echo "Execution strategies: runlist" ;;
    E5) echo "Execution strategies: fused + the distinguishability gate" ;;
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
    D1) echo "docs/plans/transformer-layer-execution-studies/07a-phase-d1-operators-at-baseline-768.md" ;;
    D2) echo "docs/plans/transformer-layer-execution-studies/07b-phase-d2-block-integration.md" ;;
    E1) echo "docs/plans/transformer-layer-execution-studies/08a-phase-e1-unblock-the-ladder.md" ;;
    E2) echo "docs/plans/transformer-layer-execution-studies/08b-phase-e2-coarse-and-instrumentation.md" ;;
    E3) echo "docs/plans/transformer-layer-execution-studies/08c-phase-e3-offload.md" ;;
    E4) echo "docs/plans/transformer-layer-execution-studies/08d-phase-e4-runlist.md" ;;
    E5) echo "docs/plans/transformer-layer-execution-studies/08e-phase-e5-fused-and-distinguishability.md" ;;
    *) echo "" ;;
  esac
}

# Phase C is four sub-phases rather than one because Phase B was 3,725 lines and took 362 minutes
# with blocking findings still open after round 3; Phase C's source material is 8,160 lines across
# five rewrites. Each sub-phase points at its OWN document: the implement prompt injects the doc
# as the session's entire task list, so four sessions sharing one doc would each try to do
# everything.
#
# Phase D splits for the same reason and one more: PL_STEP_TIMEOUT caps an implement session at
# three hours, and the single-phase form asked one session for hardware bring-up on six operators
# AND novel multi-launch integration. Splitting also means a D2 failure does not re-run D1's
# hardware time.
#
# Phase E splits five ways, for the same reason and one that is specific to it: E1 changes SHARED
# infrastructure (llms/shared/builders/gemm_builder.py) and therefore carries the ten-model
# cross-deployment regression check, which cost C4's gate hours. Folding that into a sub-phase that
# also builds an execution strategy would re-run it on every failure of either. E5 is last because
# its objective check is the only cross-mode one: the four modes' dispatch vectors either separate
# as the taxonomy predicts or they do not, and that cannot be asked until all four exist.

phase_needs_hardware() {
  case "$1" in
    A) echo "no" ;;
    B) echo "yes" ;;
    C1|C2|C3|C4) echo "yes" ;;
    D1|D2) echo "yes" ;;
    E1|E2|E3|E4|E5) echo "yes" ;;
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
    D1) cat <<'EOF'
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  ninja -C build-xrt check-programming-examples-transformer-layer

Every test in the suite passes, on real hardware. That includes every test Phases B and C added.

The driver then runs the same three-layer objective check Phase C used -- results newer than the
gate stamp, verdicts re-derived from n_mismatch / ref_dtype / rtol / atol rather than read from
your `passed` flag, and every operator re-run with `--fault-inject input` and REQUIRED to fail --
plus one clause specific to this sub-phase:

  - every operator carries a baseline_768 shape. The dimensions are read from the `shape` dict
    each results file records, NOT from the shape_key string, so naming a 512-wide shape
    "4096x768" does not satisfy it. And only a shape `opcheck.py --list` DECLARES counts, whose
    own artifact re-derives the contract above -- a results file for an undeclared shape is
    validated by nothing, because results/ is gitignored and no review or fingerprint sees it.

        layer_norm, elementwise_add   cols == 768                    (rows free)
        addnorm                       cols == 768 AND pre-add        (rows free)
        qkv_proj                      seq_len == 4096, emb_dim == 768
        ffn                           seq_len == 4096, emb_dim == 768, ffn_dim == 3072
        mha_out_proj                  seq_len == 4096, num_heads * head_dim == 768, non-causal

    The three GEMM-backed operators are pinned to seq 4096 because that is where the block runs;
    the row-parallel three are not, because build_addnorm_module derives its legal row count.

  - addnorm needs the PRE-ADD variant specifically, and the driver puts its own fault-injection
    negative control on it. The validated addnorm computes LayerNorm(x) * weight + residual;
    encoder_bert needs LayerNorm(x + residual) * weight, and nothing has ever dispatched that.
    Record `pre_add` in the shape dict the way mha_out_proj records `causal`, or give the
    operator a distinct name containing "pre_add" -- either is accepted.

  Each baseline_768 point that phase C's per-operator control did not already cover gets its own
  `--fault-inject input` run, which must fail.

  causal_mask is deliberately exempt: its shape is seq x seq, so it has no hidden dimension to
  widen, and encoder_bert uses an all-ones attention mask.
EOF
;;
    D2) cat <<'EOF'
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  ninja -C build-xrt check-programming-examples-transformer-layer

Every test in the suite passes on real hardware, including the new run_npu2_block_peano.lit. A new
.lit anywhere under programming_examples/transformer_layer/ joins the suite automatically -- lit
enrolment is path-based (--filter "transformer_layer/") -- so no CMake change is needed.

The driver then re-derives the numerics contract over every declared (operator, shape) including
`block`, runs `--fault-inject input` against `block` and requires it to FAIL, re-checks D1's
baseline_768 coverage so this sub-phase cannot regress it, and adds:

  - at least one block result is at the forced configuration: seq_len 4096, emb_dim 768,
    ffn_dim 3072, num_heads 12, head_dim 64. A smaller bring-up block beside it is fine;
    the gate point must exist.
  - its n_elements equals the full 4096 x 768 layer output, so a comparison over a slice or a
    single tile cannot pass;
  - it carries a `stages` list of at least eight per-boundary comparisons with DISTINCT names,
    each with n_mismatch == 0 and n_elements no smaller than one 4096 x 768 boundary tensor.

That last one is why the per-boundary comparison is a work item and not a nicety. C4 found a GEMM
configuration that returned zeros for two of the nine sub-tiles of each cast worker while still
resolving from the registry and still producing a plausibly-shaped output. A layer that ends in a
LayerNorm can absorb a lot of upstream damage before an end-to-end comparison trips.
EOF
;;
    E1) cat <<'EOF'
flock -x -w 1800 /tmp/mlir-air-npu.lock  agents/scripts/port-loop/gate-e1.sh

Two legs. The transformer-layer lit suite on real hardware, then the cross-deployment regression
check -- `make verify` in each of the ten shipped programming_examples/llms/<model>/ directories --
because this sub-phase changes llms/shared/builders/gemm_builder.py, which all ten resolve through.

The driver then checks, independently of anything you write:

  - THE NAMES SEPARATE. It resolves two same-method, different-tile_n GEMMs through
    gemm_registry_config -- the FFN up-projection (4096x768x3072, drain, tile_n 128) and the
    o-projection (4096x768x768, drain, tile_n 96) -- and requires sym_suffix AND obj to differ.
    Today both mint '_m32' and 'mm_m32.o', which is the collision. If the registry ever puts those
    two shapes on different methods the check FAILS LOUDLY rather than passing vacuously, because
    then it would no longer be testing anything.

  - THE LADDER MOVED. `ffn` must carry a fresh, declared, contract-satisfying result at a
    baseline_768 shape whose seq_len is NOT 4096, read from the recorded `shape` dict rather than
    from the shape_key string. Before this sub-phase build_ffn_module cannot build at any other
    ladder point at all, so this is not something a more permissive test can produce. seq 64 is the
    cheapest such point and its two registry rows both resolve to `drain`, which is exactly the
    collision.

  - THAT POINT GETS ITS OWN FAULT INJECTION. The driver re-runs opcheck.py against that exact
    shape with --fault-inject input and REQUIRES it to fail. The generic per-operator control
    injects only an operator's FIRST declared shape, so without this the one new point here would
    be the only one never injected. That is D1's recorded lesson, repeated deliberately.

  - NOTHING REGRESSED. The full D1 baseline_768 coverage clause and the D2 `block` verdict are
    re-derived from their artifacts, and `block` is re-run under injection. Changing how every
    external GEMM's symbol and object are named is exactly the change that could break them
    quietly.
EOF
;;
    E2|E3|E4|E5) cat <<'EOF'
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  ninja -C build-xrt check-programming-examples-transformer-layer

Every test in the suite passes on real hardware, including the ones earlier sub-phases added. A new
.lit anywhere under programming_examples/transformer_layer/ joins the suite automatically --
enrolment is path-based (--filter "transformer_layer/") -- so no CMake change is needed, and there
is no CMakeLists.txt in the example.

The driver then runs the same three-layer check Phases C and D used over your mode -- results newer
than the gate stamp, verdicts re-derived from n_mismatch / ref_dtype / rtol / atol rather than read
from your `passed` flag, and the mode re-run with --fault-inject input and REQUIRED to fail -- plus
the clauses Phase E adds, all in agents/scripts/port-loop/phase_e_checks.py:

  - FULL-LAYER SCOPE, exactly the standard D2's block was held to. Exactly ONE fresh, declared
    result at the forced configuration (seq_len 4096, emb_dim 768, ffn_dim 3072, 12 heads,
    head_dim 64), n_elements equal to the whole 4096 x 768 layer output, and a `stages` list of at
    least eight per-boundary comparisons with DISTINCT names, each at n_mismatch 0 and no smaller
    than one 4096 x 768 boundary tensor. Two conforming results is an error, not a convenience: the
    driver will not guess which one your dispatch vector describes.

  - THE DISPATCH VECTOR CONTRACT. A non-empty `dispatch_vectors` list of DispatchVector.as_row()
    dicts -- record the shared implementation's output, do not hand-build one. Every value finite
    and non-negative, the five count fields whole numbers, at least one submission per recorded
    vector, and some bytes actually moved. Note runlist_entries_per_submission is a derived MEAN,
    so the driver totals entries as sum(round(mean * submissions)); a value whose product is not a
    whole number of runlist entries is rejected.

  - VECTOR PROVENANCE. results/ is gitignored, so a hand-written dispatch vector is invisible to
    the fingerprint, the tamper check and every review diff. So the driver compares your recorded
    totals against the run IT initiates: the fault-injected artifact's six summed totals must EQUAL
    the clean run's. Injection perturbs one input element after the reference exists and does not
    touch the dispatch path, so on an honest run they are identical -- D2's block clean and fault
    artifacts both total 4 / 131 / 12 / 146 / 402 / 202,902,528. Emit the vectors on the injected
    path too; do not add a "skip instrumentation when injecting" shortcut.

Each sub-phase adds one clause of its own:

  E2  coarse   -- nothing further. It sets the contract the other three are measured against.
  E3  offload  -- it must aggregate NOTHING: summed runlist entries equal to summed host
                  submissions, and at least six of each. Batching them into a runlist would make
                  this mode `coarse`.
  E4  runlist  -- summed runlist entries strictly greater than `coarse`'s. This is the one ordinal
                  claim the mode owns, and the reason coarse had to be measured first. coarse
                  already measures 131, 128 of them one operator's row blocking, so a decomposition
                  that folds normalization back into a fused kernel can land BELOW it. If that
                  happens, report the number; do not inflate the decomposition.
  E5  fused    -- the DISTINGUISHABILITY gate over all four modes, which is what Phase E exists to
                  establish. Four clauses, ordinal over driver-summed totals and never absolute
                  thresholds: no two modes share a vector; offload's host submissions exceed every
                  other mode's and it aggregates nothing; runlist has more entries than coarse; and
                  fused crosses fewer sync boundaries than coarse. Two further predictions -- fused
                  entries below coarse, fused air launches at or above coarse -- are printed with a
                  verdict but do NOT halt, because both depend on how a faithful stitch decomposes.
                  If the gating clauses fail, that is a finding about the measurement model and it
                  halts the run by design. Report the measured table; never tune a mode until an
                  inequality holds.
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
    # D1/D2 add shapes, a pre-add addnorm builder, pattern/reference.py and one new lit test, all
    # inside the example. Deliberately NOT widened to kernel_registry: Phase D consumes the rows
    # C4 measured and must not write new ones, and it is not permitted to touch llms/shared/ --
    # the gemm_builder.py sym_suffix fix that would unlock the rest of the ladder is Phase E's.
    D1|D2) echo '^programming_examples/transformer_layer/' ;;
    # Phase E stays inside the example too, and 14-the-port-loop-harness.md was WRONG to predict
    # this had to widen. guard_gate_files() fingerprints .lit files, example Makefiles,
    # programming_examples/CMakeLists.txt, kernel_registry/details/*.json and llms/verify/*.py.
    # E1's llms/shared/builders/gemm_builder.py is in none of those sets, and its second gate leg
    # RUNS the ten shipped models rather than editing them. Keeping the tight prefix is what stops
    # E1 quietly touching a shipped model's Makefile to make its own regression leg pass.
    #
    # Note also what is now fingerprinted and is in NO allowlist: the driver's own scripts under
    # agents/scripts/port-loop/. A session that edits an objective check or a gate script halts the
    # run, which is the point.
    E1|E2|E3|E4|E5) echo '^programming_examples/transformer_layer/' ;;
    *) echo '' ;;
  esac
}

phase_gate_cmd() {
  case "$1" in
    A) echo "ninja -C ${PL_ROOT}/build-xrt check-programming-examples-transformer-layer" ;;
    B) echo "flock -x -w 1800 /tmp/mlir-air-npu.lock ninja -C ${PL_ROOT}/build-xrt check-programming-examples-transformer-layer" ;;
    C1|C2|C3) echo "flock -x -w 1800 /tmp/mlir-air-npu.lock ninja -C ${PL_ROOT}/build-xrt check-programming-examples-transformer-layer" ;;
    C4) echo "flock -x -w 1800 /tmp/mlir-air-npu.lock ${PL_LIB}/gate-c4.sh" ;;
    D1|D2) echo "flock -x -w 1800 /tmp/mlir-air-npu.lock ninja -C ${PL_ROOT}/build-xrt check-programming-examples-transformer-layer" ;;
    # E1 alone changes shared infrastructure, so E1 alone carries the ten-model leg.
    E1) echo "flock -x -w 1800 /tmp/mlir-air-npu.lock ${PL_LIB}/gate-e1.sh" ;;
    E2|E3|E4|E5) echo "flock -x -w 1800 /tmp/mlir-air-npu.lock ninja -C ${PL_ROOT}/build-xrt check-programming-examples-transformer-layer" ;;
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

# --- Phase D -----------------------------------------------------------------------------------
#
# D reuses Phase C's three layers wholesale -- freshness, re-derived verdict, fault-injection
# negative control -- because opcheck.py is the same mechanism and phase_c_operator_check is
# already parameterized by operator name. What D adds is COVERAGE, and it is a different claim
# from correctness: a Phase C results file proves an operator computed the right answer at SOME
# shape, and the block needs it right at the one shape the block uses.
#
# Everything below reads the recorded `shape` dict, never the shape_key. The key is a string the
# session chooses; naming a 512-wide shape "4096x768" would satisfy a name check and prove
# nothing. The dict is the spec's own named dimensions and it is what opcheck.py writes out.

# The forced configuration. baseline_768 is the only family whose projection GEMMs resolve, and
# seq 4096 is the only point in it where build_ffn_module builds at all -- the up-projection
# (N=3072, tile_n=128) and down-projection (N=768, tile_n=96) collide on f32_to_bf16_mn_<suffix>
# everywhere the registry puts them on the same method. See
# docs/plans/transformer-layer-execution-studies/07a-phase-d1-operators-at-baseline-768.md.
PL_D_HIDDEN=768
PL_D_FFN=3072
PL_D_SEQ=4096
PL_D_HEADS=12
PL_D_HEAD_DIM=64
PL_D_MIN_STAGES=8

# Shared by D1 and D2. D2 re-runs it so that landing the block cannot regress the coverage D1
# established. Writes the "<operator> <shape_key>" pairs that satisfied a clause to $1, so the
# caller can put a negative control on exactly those points.
#
# TWO THINGS THIS DELIBERATELY DOES, both of which an earlier version got wrong.
#
# It does not trust `passed`, and it will not read a results file for a shape that
# `opcheck.py --list` never declared. `results/` is GITIGNORED, so a hand-written JSON there is
# invisible to guard_fingerprint, guard_check_tamper, guard_check_destructive and every Codex diff
# review -- freshness would have been the only barrier, and the session-authored lit recipe runs
# inside the gate window. So every record here must be a DECLARED (operator, shape_key) and must
# independently re-derive the same contract phase_c_operator_check enforces.
#
# It does not ignore seq_len. Requiring only `emb_dim == 768` of qkv_proj was already satisfied by
# the Phase C row on disk today (opcheck_specs.py: seq_len 64, emb_dim 768) -- D1 could have
# cleared that clause having added nothing at all. The block runs at one sequence length and the
# operators have to be right at it, which is the entire purpose of D1.
phase_d_baseline_768_coverage() {
  local pairs_out="${1:-/dev/null}"
  local opcheck="${PL_ROOT}/programming_examples/transformer_layer/opcheck.py"

  if [ ! -f "${opcheck}" ]; then
    log_error "objective check: ${opcheck} does not exist"
    return 1
  fi
  if [ -z "${_GATE_STARTED_AT:-}" ] || [ ! -e "${_GATE_STARTED_AT}" ]; then
    log_error "objective check: no gate timestamp; cannot prove coverage results are fresh"
    return 1
  fi

  local listing
  if ! listing="$(cd "${PL_ROOT}/programming_examples/transformer_layer" \
                    && python3 "${opcheck}" --list 2>&1)"; then
    log_error "objective check: 'opcheck.py --list' failed: ${listing}"
    return 1
  fi

  PL_D_LISTING="${listing}" \
  PL_D_RESULTS="${PL_ROOT}/programming_examples/transformer_layer/results" \
  PL_D_STAMP="${_GATE_STARTED_AT}" \
  PL_D_PAIRS="${pairs_out}" \
  PL_D_INJECTED="${PL_D_INJECTED:-}" \
  PL_D_HIDDEN="${PL_D_HIDDEN}" PL_D_FFN="${PL_D_FFN}" PL_D_SEQ="${PL_D_SEQ}" \
  PL_C_RTOL="${PL_C_RTOL}" PL_C_ATOL_MAX="${PL_C_ATOL_MAX}" \
  python3 -c '
import json, os, pathlib, sys

listing   = json.loads(os.environ["PL_D_LISTING"])
results   = pathlib.Path(os.environ["PL_D_RESULTS"])
cutoff    = os.path.getmtime(os.environ["PL_D_STAMP"])
pairs_out = os.environ["PL_D_PAIRS"]
injected  = set(os.environ.get("PL_D_INJECTED", "").split())
HID       = int(os.environ["PL_D_HIDDEN"])
FFN       = int(os.environ["PL_D_FFN"])
SEQ       = int(os.environ["PL_D_SEQ"])
rtol_req  = float(os.environ["PL_C_RTOL"])
atol_max  = float(os.environ["PL_C_ATOL_MAX"])

if not listing:
    print("objective check: opcheck.py --list declared no operators", file=sys.stderr)
    sys.exit(1)

# Only a declared shape whose own artifact satisfies the numerics contract counts as evidence.
valid = []
for e in listing:
    op, key = e["operator"], e["shape_key"]
    f = results / ("%s__%s.json" % (op, key))
    if not f.exists() or f.stat().st_mtime < cutoff:
        continue
    try:
        d = json.loads(f.read_text())
    except Exception:
        continue
    if d.get("fault_injected") is not None:
        continue
    if d.get("ref_dtype") != "float32":
        continue
    if d.get("n_mismatch") != 0:
        continue
    if not isinstance(d.get("n_elements"), int) or d["n_elements"] <= 0:
        continue
    try:
        if abs(float(d["rtol"]) - rtol_req) > 1e-12:
            continue
        if float(d["atol"]) > atol_max:
            continue
    except (KeyError, TypeError, ValueError):
        continue
    valid.append((op, key, d.get("shape") or {}))

# The first declared shape_key of an operator is the one phase_c_operator_check injects.
first_key = {}
for e in listing:
    first_key.setdefault(e["operator"], e["shape_key"])

# Row counts are left free for the three row-parallel operators: build_addnorm_module requires
# rows == herd_x * rows_per_call, and less fits in L1 at 768 than at 512, so the legal row count
# is derived rather than chosen. The GEMM-backed three are pinned to the block sequence length.
#
# causal_mask is deliberately absent. Its shape is seq x seq, so there is no hidden dimension in
# it to widen, and encoder_bert uses an all-ones attention mask -- the block never calls it.
REQUIRED = [
    ("layer_norm", "cols == %d" % HID, None,
     lambda op, s: s.get("cols") == HID),
    ("elementwise_add", "cols == %d" % HID, None,
     lambda op, s: s.get("cols") == HID),
    ("qkv_proj", "seq_len == %d and emb_dim == %d" % (SEQ, HID), None,
     lambda op, s: s.get("seq_len") == SEQ and s.get("emb_dim") == HID),
    ("ffn", "seq_len == %d, emb_dim == %d, ffn_dim == %d" % (SEQ, HID, FFN), None,
     lambda op, s: s.get("seq_len") == SEQ and s.get("emb_dim") == HID
                   and s.get("ffn_dim") == FFN),
    ("mha_out_proj",
     "seq_len == %d, num_heads * head_dim == %d, causal false" % (SEQ, HID), None,
     lambda op, s: s.get("seq_len") == SEQ
                   and (s.get("num_heads") or 0) * (s.get("head_dim") or 0) == HID
                   and s.get("causal") is False),
    ("addnorm", "PRE-ADD at cols == %d" % HID,
     "encoder_bert normalizes the SUM. The post-add form is a different function and is the only "
     "one Phase C ever ran; record pre_add in the shape dict the way mha_out_proj records causal, "
     "or name the operator distinctly.",
     lambda op, s: s.get("cols") == HID
                   and (s.get("pre_add") is True or "pre_add" in op)),
]

rc = 0
satisfying = []
for name, why, hint, pred in REQUIRED:
    hits = [(op, key) for (op, key, s) in valid if op.startswith(name) and pred(op, s)]
    if hits:
        print("  %s: %s  [%s]" % (name, why, ", ".join("%s %s" % h for h in hits)))
        satisfying.extend(hits)
    else:
        print("objective check: %s has no declared, fresh, contract-satisfying result at the "
              "baseline_768 point (%s)" % (name, why), file=sys.stderr)
        if hint:
            print("  %s" % hint, file=sys.stderr)
        rc = 1

if rc:
    sys.exit(rc)

# Everything phase_c_operator_check did not already inject, since it takes one shape per operator
# NAME and picks the first declared key.
need = sorted({(op, key) for (op, key) in satisfying
               if op not in injected or first_key.get(op) != key})
with open(pairs_out, "w") as fh:
    for op, key in need:
        fh.write("%s %s\n" % (op, key))
sys.exit(0)
' || { log_error "objective check FAILED: the operators are not all validated at baseline_768"; return 1; }

  return 0
}

# The fault-injection layer, over the specific points the coverage clause accepted.
#
# phase_c_operator_check injects one shape per operator NAME and picks the first declared key.
# That is the wrong point here twice over: where the pre-add addnorm is a distinct operator it was
# never passed that name at all, and where it is a flag in the shape dict the first declared
# addnorm key is still the 64x512 POST-add row. Either way the pre-add form -- the one function in
# D1 that has never run on hardware, and the one whose reference is most likely to agree with the
# device by construction -- would never be injected. This closes that.
phase_d_negative_control() {
  local pairs="$1"
  local opcheck="${PL_ROOT}/programming_examples/transformer_layer/opcheck.py"
  local op key n=0

  [ -s "${pairs}" ] || {
    log_info "objective check: no injection points beyond the ones phase_c already covered"
    return 0
  }

  while read -r op key; do
    [ -n "${op}" ] && [ -n "${key}" ] || continue
    n=$(( n + 1 ))
    log_info "objective check: negative control ${op} [${key}] — this run MUST fail"
    if ( cd "${PL_ROOT}/programming_examples/transformer_layer" \
         && flock -x -w 1800 /tmp/mlir-air-npu.lock \
              python3 "${opcheck}" --operator "${op}" --shape-key "${key}" \
                                   --fault-inject input ) >/dev/null 2>&1; then
      log_error "objective check FAILED: ${op} [${key}] PASSED with a fault injected into its input."
      log_error "  The check does not discriminate: it would pass on a broken kernel too."
      return 1
    fi
    log_info "  ${op} [${key}]: correctly failed under injection"
  done < "${pairs}"

  log_info "objective check: ${n} baseline_768 point(s) correctly failed under injection"
  return 0
}

phase_d1_objective_check() {
  local ops="causal_mask addnorm layer_norm elementwise_add qkv_proj ffn mha_out_proj"
  local pairs="${PL_STATE_DIR}/d-negative-control-points.txt"
  : > "${pairs}"

  # shellcheck disable=SC2086
  phase_c_operator_check ${ops} || return 1
  PL_D_INJECTED="${ops}" phase_d_baseline_768_coverage "${pairs}" || return 1
  phase_d_negative_control "${pairs}" || return 1

  log_info "objective check passed: every operator validated at baseline_768, pre-add included"
  return 0
}

# D2 is integration, and what a laxer test could hide here is SCOPE: a comparison over one tile of
# the output, or a block gate with no per-boundary breakdown behind it. Both are read out of the
# artifact rather than taken on trust.
#
# The coverage clause is re-run so that landing the block cannot regress D1, but its negative
# controls are not: D1 already proved those points discriminate, and repeating six hardware
# injections here would cost the gate more than it buys. The block itself IS injected, by
# phase_c_operator_check.
phase_d2_objective_check() {
  phase_c_operator_check block || return 1
  phase_d_baseline_768_coverage || return 1

  PL_D_RESULTS="${PL_ROOT}/programming_examples/transformer_layer/results" \
  PL_D_STAMP="${_GATE_STARTED_AT}" \
  PL_D_SEQ="${PL_D_SEQ}" PL_D_HIDDEN="${PL_D_HIDDEN}" PL_D_FFN="${PL_D_FFN}" \
  PL_D_HEADS="${PL_D_HEADS}" PL_D_HEAD_DIM="${PL_D_HEAD_DIM}" \
  PL_D_MIN_STAGES="${PL_D_MIN_STAGES}" \
  python3 -c '
import json, os, pathlib, sys

results = pathlib.Path(os.environ["PL_D_RESULTS"])
cutoff  = os.path.getmtime(os.environ["PL_D_STAMP"])
SEQ     = int(os.environ["PL_D_SEQ"])
HID     = int(os.environ["PL_D_HIDDEN"])
FFN     = int(os.environ["PL_D_FFN"])
HEADS   = int(os.environ["PL_D_HEADS"])
HDIM    = int(os.environ["PL_D_HEAD_DIM"])
MIN_ST  = int(os.environ["PL_D_MIN_STAGES"])

blocks = [f for f in sorted(results.glob("block__*.json")) if f.stat().st_mtime >= cutoff]
if not blocks:
    print("objective check: no fresh results/block__*.json; the block gate produced no artifact",
          file=sys.stderr)
    sys.exit(1)

# At least ONE block result must be at the forced configuration -- not all of them. Every other
# operator in this example carries several shapes, and nothing forbids a smaller bring-up block
# beside the gate point; such a shape is declared, so phase_c_operator_check already holds it to
# the numerics contract. What must not happen is the gate point being absent.
conforming, rejected = [], []
for f in blocks:
    try:
        d = json.loads(f.read_text())
    except Exception as e:
        rejected.append((f.name, ["not readable JSON: %s" % e]))
        continue

    problems = []
    shape = d.get("shape") or {}

    # The configuration is forced, not chosen. A block quietly run at a smaller sequence or a
    # narrower family would pass every numeric clause and prove nothing about the case matrix.
    for key, want in (("seq_len", SEQ), ("emb_dim", HID), ("ffn_dim", FFN),
                      ("num_heads", HEADS), ("head_dim", HDIM)):
        if shape.get(key) != want:
            problems.append("shape[%r]=%r, not %r" % (key, shape.get(key), want))

    # Full-output coverage. This is what stops a comparison over one tile from passing: the layer
    # output is seq x hidden and nothing smaller is the layer.
    want_elems = SEQ * HID
    if d.get("n_elements") != want_elems:
        problems.append("n_elements=%r, not the full %d x %d = %d layer output"
                        % (d.get("n_elements"), SEQ, HID, want_elems))

    # The per-boundary breakdown. C4 found a GEMM configuration that returned zeros for two of the
    # nine sub-tiles of each cast worker while still resolving from the registry and still
    # producing a plausibly-shaped output; a layer ending in a LayerNorm absorbs a lot of upstream
    # damage before an end-to-end comparison trips. This makes the localization mandatory.
    stages = d.get("stages")
    if not isinstance(stages, list):
        problems.append("no `stages` list: the per-boundary comparison is a work item, not "
                        "optional")
    elif len(stages) < MIN_ST:
        problems.append("only %d stage(s); at least %d operator boundaries must be compared"
                        % (len(stages), MIN_ST))
    else:
        seen_names = set()
        for i, st in enumerate(stages):
            if not isinstance(st, dict):
                problems.append("stage %d is not an object" % i)
                continue
            name = st.get("name")
            # Distinct, non-empty names. Without this the clause is satisfiable by repeating one
            # trivial entry MIN_ST times, which localizes nothing -- and localization is the
            # entire reason the stage list is required.
            if not isinstance(name, str) or not name.strip():
                problems.append("stage %d has no name" % i)
                name = "#%d" % i
            elif name in seen_names:
                problems.append("stage name %r repeats; the boundaries must be distinct" % name)
            else:
                seen_names.add(name)
            # Every boundary in an encoder layer is at least a seq x hidden tensor (the FFN
            # interior is seq x ffn, larger). A stage smaller than that is not a boundary.
            n = st.get("n_elements")
            if not isinstance(n, int) or n < SEQ * HID:
                problems.append("stage %s: n_elements=%r, smaller than one %d x %d boundary "
                                "tensor" % (name, n, SEQ, HID))
            if st.get("n_mismatch") != 0:
                problems.append("stage %s: n_mismatch=%r, not 0" % (name, st.get("n_mismatch")))

    if problems:
        rejected.append((f.name, problems))
    else:
        conforming.append(f.name)
        print("  %s: %d elements over the full %dx%d layer, %d distinct clean stage boundaries"
              % (f.name, d["n_elements"], SEQ, HID, len(stages)))

if not conforming:
    print("objective check: no block result is at the gate configuration "
          "(seq_len %d, emb_dim %d, ffn_dim %d, %d heads) with a full-layer, per-boundary "
          "comparison behind it:" % (SEQ, HID, FFN, HEADS), file=sys.stderr)
    for name, problems in rejected:
        print("  %s: %s" % (name, "; ".join(problems)), file=sys.stderr)
    sys.exit(1)

# A smaller bring-up block beside the gate point is legitimate -- it is a declared shape, so
# phase_c_operator_check already held it to the numerics contract. Say what was skipped rather
# than pass over it silently.
for name, problems in rejected:
    print("  (%s is not the gate configuration, ignored: %s)" % (name, problems[0]))
sys.exit(0)
' || { log_error "objective check FAILED: the block artifact does not prove a full-layer, per-boundary comparison"; return 1; }

  log_info "objective check passed: the encoder_bert layer matches at full scope, stage by stage"
  return 0
}

# --- Phase E -----------------------------------------------------------------------------------
#
# E reuses phase_c_operator_check for every mode -- freshness, re-derived verdict, fault-injection
# negative control -- because a mode is just another opcheck operator to it. What E adds is in
# phase_e_checks.py rather than inline here, for two reasons. It is far more than the forty lines
# the other phases' embedded python runs to, and putting it in a module makes it runnable in BOTH
# DIRECTIONS without hardware:
#
#     python3 agents/scripts/port-loop/phase_e_checks.py selftest
#
# That is the harness's own twice-learned lesson. C4 halted on an objective check no honest run
# could pass because only its failure direction had been tried; Phase B passed a hardware gate that
# never touched the NPU. The selftest builds conforming and violating artifact sets in a temp
# directory and asserts the verdict flips for every clause. Run it after touching either file.
#
# The pass direction is also demonstrated against real data: D2's block artifact pair satisfies the
# full-layer scope, the vector contract and the provenance clause unmodified.

PL_E_CHECKS="${PL_LIB:-}/phase_e_checks.py"

# The example's results directory. PL_E_RESULTS exists so these checks can be aimed at a synthetic
# artifact set; the default is the real path, and a run never sets it.
phase_e_results_dir() {
  echo "${PL_E_RESULTS:-${PL_ROOT}/programming_examples/transformer_layer/results}"
}

phase_e_stamp() {
  local s="${PL_E_STAMP:-${_GATE_STARTED_AT:-}}"
  if [ -z "${s}" ] || [ ! -e "${s}" ]; then
    log_error "objective check: no gate timestamp; cannot prove artifacts are fresh"
    return 1
  fi
  printf '%s' "${s}"
}

# `opcheck.py --list`, to a file the python checks read. Only a DECLARED (operator, shape_key)
# counts as evidence: results/ is gitignored, so an artifact for an undeclared shape is validated
# by nothing at all -- not the fingerprint, not the tamper check, not any review diff.
phase_e_write_listing() {
  local out="$1"
  local opcheck="${PL_ROOT}/programming_examples/transformer_layer/opcheck.py"
  if [ ! -f "${opcheck}" ]; then
    log_error "objective check: ${opcheck} does not exist"
    return 1
  fi
  if ! ( cd "${PL_ROOT}/programming_examples/transformer_layer" \
           && python3 "${opcheck}" --list ) > "${out}" 2>/dev/null; then
    log_error "objective check: 'opcheck.py --list' failed"
    return 1
  fi
  if [ ! -s "${out}" ]; then
    log_error "objective check: 'opcheck.py --list' declared nothing"
    return 1
  fi
  return 0
}

# phase_e_run <subcommand> [extra args...]
phase_e_run() {
  local sub="$1"; shift
  local stamp listing
  stamp="$(phase_e_stamp)" || return 1
  listing="${PL_STATE_DIR}/e-listing.json"
  phase_e_write_listing "${listing}" || return 1
  python3 "${PL_E_CHECKS}" "${sub}" \
    --results "$(phase_e_results_dir)" \
    --stamp "${stamp}" \
    --listing "${listing}" \
    "$@"
}

# phase_e_mode_objective_check <mode> [extra args for the `mode` subcommand]
phase_e_mode_objective_check() {
  local mode="$1"; shift
  phase_c_operator_check "${mode}" || return 1
  phase_e_run mode --operator "${mode}" "$@" || {
    log_error "objective check FAILED: ${mode}'s artifact does not prove a full-layer,"
    log_error "  per-boundary comparison behind a dispatch vector the driver's own fault run agrees with"
    return 1
  }
  log_info "objective check passed: ${mode} matches at full scope with a measured dispatch vector"
  return 0
}

phase_e1_objective_check() {
  # 1. The naming fix itself, resolved through the real registry rather than read off the source.
  # The message deliberately does not assert WHY this failed. The check distinguishes a live
  # collision from an unimportable module from a registry that no longer puts those two shapes on
  # the same method, and says which on its own stderr; restating one of them here would send a
  # reader chasing the wrong thing when it was one of the others.
  if ! python3 "${PL_E_CHECKS}" naming --repo "${PL_ROOT}"; then
    log_error "objective check FAILED: the GEMM naming clause did not pass; see the line above."
    log_error "  Until same-method GEMMs at different tile_n mint distinct symbol suffixes and"
    log_error "  object names, the sequence ladder stays pinned to seq 4096."
    return 1
  fi

  # 2. The ladder actually moved, with its OWN negative control on the new point. Reusing
  #    phase_d_negative_control here is deliberate: it is the mechanism D1 built precisely because
  #    phase_c_operator_check injects only an operator's FIRST declared shape.
  local pairs="${PL_STATE_DIR}/e1-ladder-points.txt"
  : > "${pairs}"
  if ! phase_e_run ladder --pairs-out "${pairs}"; then
    log_error "objective check FAILED: no ffn result at a second point on the sequence ladder"
    return 1
  fi
  phase_d_negative_control "${pairs}" || return 1

  # 3. Nothing regressed. Changing how every external GEMM is named is exactly the change that
  #    could quietly break the operators and the block that already pass.
  phase_c_operator_check block || return 1
  phase_d_baseline_768_coverage || return 1

  log_info "objective check passed: the ladder is unblocked and D1/D2 still hold"
  return 0
}

phase_e2_objective_check() { phase_e_mode_objective_check coarse; }

# offload aggregates nothing -- checkable from its own artifact, without the other three modes.
# Six because attention stays in host torch (08c), so it is six projection GEMMs rather than the
# eight the plan originally predicted.
phase_e3_objective_check() {
  phase_e_mode_objective_check offload --expect-no-aggregation --min-submissions 6
}

phase_e4_objective_check() {
  phase_e_mode_objective_check runlist || return 1
  if ! phase_e_run compare --left runlist --right coarse \
                           --field runlist_entries --relation gt; then
    log_error "objective check FAILED: the fine-grained mode is not finer than the coarse one"
    return 1
  fi
  return 0
}

phase_e5_objective_check() {
  phase_e_mode_objective_check fused || return 1
  if ! phase_e_run distinguish; then
    log_error "objective check FAILED: the four modes' dispatch vectors do not separate."
    log_error "  Per 08 this means the measurement model is not measuring what it claims, and it"
    log_error "  must be resolved BEFORE Phase F consumes these numbers. The table is above."
    return 1
  fi
  log_info "objective check passed: all four modes agree with the oracle AND separate"
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
    D1) phase_d1_objective_check ;;
    D2) phase_d2_objective_check ;;
    E1) phase_e1_objective_check ;;
    E2) phase_e2_objective_check ;;
    E3) phase_e3_objective_check ;;
    E4) phase_e4_objective_check ;;
    E5) phase_e5_objective_check ;;
    *) return 0 ;;
  esac
}
