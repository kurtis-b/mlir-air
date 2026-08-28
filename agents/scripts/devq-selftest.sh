#!/usr/bin/env bash
# devq-selftest.sh -- prove the four scheduling properties of devq.sh with sleep
# jobs and NO device.  Runs against a throwaway DEVQ_DIR and a throwaway device
# lock so it never touches the real queue or the real NPU lock.
#
# Run it from a PLAIN shell.  If DEVQ_JOB_ID is set -- which it is inside any
# queued devq job -- tests 5 and 7 take devq's nesting and bypass paths and fail
# for environmental reasons rather than real ones.  This unsets it rather than
# leaving a confusing red for the next person to debug.
set -uo pipefail
if [ -n "${DEVQ_JOB_ID:-}" ]; then
  printf 'devq-selftest: DEVQ_JOB_ID=%s is set; unsetting it (tests 5 and 7 probe the nesting refusal)\n' \
    "$DEVQ_JOB_ID" >&2
  unset DEVQ_JOB_ID
fi

DEVQ=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/devq.sh
ROOT=$(mktemp -d /tmp/devq-selftest.XXXXXX)
export DEVQ_NPU_LOCK="$ROOT/fake-npu.lock"   # TEST-ONLY override
export DEVQ_POLL=0.05
trap 'rm -rf "$ROOT"' EXIT

PASS=0; FAIL=0
ok()   { PASS=$((PASS+1)); printf '  PASS  %s\n' "$*"; }
bad()  { FAIL=$((FAIL+1)); printf '  FAIL  %s\n' "$*"; }
check(){ if awk "BEGIN{exit !($1)}"; then ok "$2"; else bad "$2  [expr false: $1]"; fi; }
fresh(){ export DEVQ_DIR="$ROOT/$1"; mkdir -p "$DEVQ_DIR"; }
fld()  { "$DEVQ" status --raw | awk -F'\t' -v i="$1" -v c="$2" '$1==i{print $c}'; }
st()   { fld "$1" 3; }; started(){ fld "$1" 6; }; fin(){ fld "$1" 7; }; jpid(){ fld "$1" 8; }

# ---------------------------------------------------------------- 1
echo "TEST 1: two builds submitted together run concurrently"
fresh t1
A=$("$DEVQ" submit --class build --name a -- sleep 2)
B=$("$DEVQ" submit --class build --name b -- sleep 2)
"$DEVQ" wait "$A" >/dev/null; "$DEVQ" wait "$B" >/dev/null
sA=$(started "$A"); fA=$(fin "$A"); sB=$(started "$B"); fB=$(fin "$B")
OV=$(awk "BEGIN{m=($fA<$fB)?$fA:$fB; s=($sA>$sB)?$sA:$sB; printf \"%.2f\", m-s}")
printf '  jobs %s[%s..%s] %s[%s..%s] overlap=%ss\n' "$A" "$sA" "$fA" "$B" "$sB" "$fB" "$OV"
check "$OV > 1.5" "builds $A and $B overlapped by ${OV}s (>1.5s of a 2s job)"
check "\"$(st "$A")\"==\"done\" && \"$(st "$B")\"==\"done\"" "both builds reported done"

# ---------------------------------------------------------------- 2
echo "TEST 2: a measure submitted after two builds waits for BOTH to finish"
fresh t2
A=$("$DEVQ" submit --class build -- sleep 2)
B=$("$DEVQ" submit --class build -- sleep 3)
M=$("$DEVQ" submit --class measure -- sleep 0.3)
"$DEVQ" wait "$M" >/dev/null
fA=$(fin "$A"); fB=$(fin "$B"); sM=$(started "$M")
printf '  build %s finished %s | build %s finished %s | measure %s started %s\n' "$A" "$fA" "$B" "$fB" "$M" "$sM"
check "$sM > $fA" "measure started after build $A finished (+$(awk "BEGIN{printf \"%.2f\", $sM-$fA}")s)"
check "$sM > $fB" "measure started after build $B finished (+$(awk "BEGIN{printf \"%.2f\", $sM-$fB}")s)"
check "$sM - $fB < 1.0" "measure admitted promptly once builds drained"

# ---------------------------------------------------------------- 3
echo "TEST 3: continuous build churn submitted AFTER a queued measure cannot pass it"
echo "        (this is the anti-starvation property the flock -s/-x design lacked)"
fresh t3
L=$("$DEVQ" submit --class build --name longbuild -- sleep 3)
M=$("$DEVQ" submit --class measure --name meas -- sleep 0.5)
CHURN=()
for _ in $(seq 16); do          # ~2.5s of continuous build submissions, all after M
  CHURN+=("$("$DEVQ" submit --class build --name churn -- sleep 0.4)")
  sleep 0.15
done
"$DEVQ" wait "$M" >/dev/null
fL=$(fin "$L"); sM=$(started "$M")
printf '  long build %s finished %s | measure %s started %s | %d churn builds submitted after it\n' \
  "$L" "$fL" "$M" "$sM" "${#CHURN[@]}"
check "${#CHURN[@]} >= 10" "churn actually ran (${#CHURN[@]} later builds submitted while the measure was queued)"
check "$sM > $fL" "measure started only after the in-flight build drained"
check "$sM - $fL < 1.0" "measure was NOT starved by the churn (waited $(awk "BEGIN{printf \"%.2f\", $sM-$fL}")s past drain)"
LATE=0; EARLIEST=; NSTARTED=0
for c in "${CHURN[@]}"; do
  s=$(started "$c")
  if [ -z "$s" ] || [ "$s" = 0 ]; then continue; fi
  NSTARTED=$((NSTARTED+1))
  awk "BEGIN{exit !($s < $sM)}" && LATE=$((LATE+1))
  { [ -z "$EARLIEST" ] || awk "BEGIN{exit !($s < $EARLIEST)}"; } && EARLIEST=$s
done
printf '  earliest churn-build start: %s (%d of %d churn builds had started)\n' \
  "${EARLIEST:-<none started>}" "$NSTARTED" "${#CHURN[@]}"
check "$LATE == 0" "zero of the ${#CHURN[@]} later builds jumped ahead of the measure"
# The measure is a barrier, not a block: the churn must all drain afterwards.
for c in "${CHURN[@]}"; do "$DEVQ" wait "$c" >/dev/null; done
DRAINED=0; for c in "${CHURN[@]}"; do [ "$(st "$c")" = done ] && DRAINED=$((DRAINED+1)); done
check "$DRAINED == ${#CHURN[@]}" "all ${#CHURN[@]} churn builds ran to completion after the measure ($DRAINED done)"

# ---------------------------------------------------------------- 4
echo "TEST 4: a SIGKILLed job is reported failed, not left running"
fresh t4
K=$("$DEVQ" submit --class build -- sleep 60)
for _ in $(seq 100); do [ "$(st "$K")" = running ] && break; sleep 0.05; done
P=$(jpid "$K")
KIDS=$(pgrep -g "$P" 2>/dev/null | grep -v "^$P\$" | tr '\n' ' ')
printf '  job %s state=%s runner_pid=%s descendants=[%s]\n' "$K" "$(st "$K")" "$P" "$KIDS"
check "\"$(st "$K")\"==\"running\"" "job is running before the kill"
kill -9 "$P"; sleep 0.3
S=$(st "$K")
printf '  after SIGKILL of pid %s: state=%s exit=%s\n' "$P" "$S" "$(fld "$K" 4)"
check "\"$S\"==\"failed\"" "status reconciled the SIGKILLed job to '$S' (not 'running')"
LIVE=0; for k in $KIDS; do kill -0 "$k" 2>/dev/null && LIVE=$((LIVE+1)); done
check "$LIVE == 0" "orphaned descendants of the killed runner were reaped ($LIVE still alive)"
N=$("$DEVQ" submit --class measure -- true); "$DEVQ" wait "$N" >/dev/null
check "\"$(st "$N")\"==\"done\"" "a later measure is not blocked forever by the dead job (state=$(st "$N"))"

# ---------------------------------------------------------------- 5
# `run` is what replaces a bare `flock -x LOCK CMD` at a gate, so the two things a
# gate depends on are the two things asserted here: the command's OUTPUT reaches
# stdout (submit alone diverts it to the job log, which would blank a FileCheck
# while still exiting 0) and its EXIT STATUS is the caller's.
echo "TEST 5: run relays the job's output and exits with the job's status"
fresh t5
OUT=$("$DEVQ" run --class build -- bash -c 'echo relayed-one; echo relayed-two' 2>/dev/null); RC=$?
# Newlines are flattened before the comparison: `check` evaluates through awk, and an
# embedded newline in an awk string constant is a parse error, not a failed compare.
FLAT=$(printf '%s' "$OUT" | tr '\n' '|')
printf '  rc=%s output=[%s]\n' "$RC" "$FLAT"
check "$RC == 0" "run exits 0 for a successful job"
check "\"$FLAT\"==\"relayed-one|relayed-two\"" "both output lines reached stdout in order"
"$DEVQ" run --class build -- bash -c 'echo doomed; exit 37' >/dev/null 2>&1
check "$? == 37" "a job exiting 37 makes run exit 37 (a swallowed status would read as PASS)"
"$DEVQ" run --class measure -- true >/dev/null 2>&1
check "$? == 0" "run works for the measure class too, holding the device lock"

# ---------------------------------------------------------------- 6
# Nesting is the one way `run` can hang rather than fail: the inner measure queues
# behind the device lock its own parent runner holds, and reports a lock timeout
# NPU_LOCK_WAIT seconds later with nothing pointing at the nested call.
echo "TEST 6: run refuses to nest inside a running devq job"
fresh t6
ERR=$(DEVQ_JOB_ID=999 "$DEVQ" run --class measure -- echo nope 2>&1 >/dev/null); RC=$?
printf '  rc=%s stderr=[%s]\n' "$RC" "$ERR"
check "$RC == 2" "a nested run exits 2 immediately rather than stalling for NPU_LOCK_WAIT"
case $ERR in *"nesting would deadlock"*) ok "the refusal names the cause";;
              *) bad "refusal message does not name nesting: $ERR";; esac

# ---------------------------------------------------------------- 7
# `[2026-08-12]` queue item 19.  The device lock is advisory: nothing outside this
# broker consults it, which is how a run that meant to be compile-only dispatched
# beside job 252's 65-minute regression.  `preflight` is the askable form of the
# probe build admission already uses, so the properties worth pinning are that it
# REFUSES when held, that it cannot be turned into a way to acquire, and that it
# does not refuse the caller its own queued job.
echo "TEST 7: preflight fails CLOSED on a held device and holds nothing itself"
fresh t7
RC=0; "$DEVQ" preflight >/dev/null 2>&1 || RC=$?
check "$RC == 0" "an idle device passes preflight (rc=$RC)"

# Stand in for an un-migrated job that took the lock directly.
( exec 8>>"$DEVQ_NPU_LOCK"; flock -x 8; sleep 3 ) &
HOLDER=$!
sleep 0.4
ERR=$("$DEVQ" preflight 2>&1 >/dev/null); RC=$?
printf '  rc=%s stderr=[%.120s]\n' "$RC" "$ERR"
check "$RC == 3" "a held device REFUSES with 3, distinct from a command's own failure"
case $ERR in *"run --class measure"*) ok "the refusal names the queued alternative";;
              *) bad "refusal does not say how to dispatch correctly: $ERR";; esac
# Inside a job the runner holds the lock on its own fd; probing there would refuse
# every legitimate measure.  Same nesting trap `run` refuses for, other way round.
RC=0; DEVQ_JOB_ID=999 "$DEVQ" preflight >/dev/null 2>&1 || RC=$?
check "$RC == 0" "inside a devq job preflight passes without probing (rc=$RC)"
kill "$HOLDER" 2>/dev/null; wait "$HOLDER" 2>/dev/null
# It must never acquire: after a passing preflight the lock is still takeable.
"$DEVQ" preflight >/dev/null 2>&1
if ( exec 7>>"$DEVQ_NPU_LOCK" && flock -n -x 7 ) 2>/dev/null; then
  ok "preflight holds nothing -- the lock is still free after it passes"
else
  bad "preflight left the device lock held; it must only ever ask"
fi

# TEST 8 exists because devq records a job's exit code FAITHFULLY -- so a script
# that swallows a leg's status makes the job read done/0 with every leg red. That
# happened in four scripts and burned three device slots (devq 806/810/814: six
# profile legs at rc=2, job done/0).
#
# The first version of this test PASSED ON A BROKEN TEMPLATE -- a review found
# that `leg x -- bash -c 'false | true'` exited 0 and every assertion here was
# still green. So each case below is a concrete input that was measured failing
# BEFORE the corresponding fix, not a restatement of the template's comments.
echo "TEST 8: new-job writes a skeleton that cannot exit 0 on red legs"
JOBSH="$ROOT/gen-job.sh"
JR="$ROOT/jr"
# gen LEGS BODY : regenerate the skeleton with EXPECT_LEGS=LEGS and BODY as its legs
gen() {
  rm -f "$JOBSH"
  "$DEVQ" new-job "$JOBSH" "selftest generated" >/dev/null 2>&1
  sed -i "s/^EXPECT_LEGS=0 /EXPECT_LEGS=$1 /" "$JOBSH"
  python3 - "$JOBSH" "$2" <<'PY'
import io, sys
p, body = sys.argv[1], sys.argv[2]
s = io.open(p).read()
old = "# leg build -- make -C some/dir\n# leg verify -- python3 some/check.py"
assert s.count(old) == 1, "skeleton lost its leg placeholder"
io.open(p, "w").write(s.replace(old, body))
PY
}
runjob() { RC=0; OUT=$(R="$JR" bash "$JOBSH" 2>&1) || RC=$?; }

if [ -x "$JOBSH" ] 2>/dev/null || "$DEVQ" new-job "$JOBSH" "t" >/dev/null 2>&1; then
  [ -x "$JOBSH" ] && ok "new-job writes an executable script" || bad "new-job did not write an executable script"
fi
# the refusal must also leave the existing file BYTE-IDENTICAL, not merely return 2
BEFORE=$(md5sum < "$JOBSH")
RC=0; "$DEVQ" new-job "$JOBSH" >/dev/null 2>&1 || RC=$?
check "$RC == 2" "new-job REFUSES to overwrite an existing job script (rc=$RC)"
[ "$(md5sum < "$JOBSH")" = "$BEFORE" ] && ok "the refused overwrite left the existing script untouched" \
  || bad "new-job modified the file it refused to overwrite"

# a DANGLING symlink is not "absent": cat > would create the target through it
rm -f "$ROOT/dangle" "$ROOT/dangle-target"
ln -s "$ROOT/dangle-target" "$ROOT/dangle"
RC=0; "$DEVQ" new-job "$ROOT/dangle" >/dev/null 2>&1 || RC=$?
if [ "$RC" -ne 0 ] && [ ! -e "$ROOT/dangle-target" ]; then
  ok "new-job refuses a dangling symlink without creating its target"
else
  bad "new-job followed a dangling symlink (rc=$RC, target exists: $([ -e "$ROOT/dangle-target" ] && echo yes || echo no))"
fi
# an unwritable target must not be reported as written
RC=0; "$DEVQ" new-job /proc/devq-selftest-job >/dev/null 2>&1 || RC=$?
check "$RC != 0" "new-job reports failure when it cannot create the file (rc=$RC)"
# a TITLE is not sed replacement syntax: & would expand, | would abort the substitution
rm -f "$ROOT/t-amp.sh"; "$DEVQ" new-job "$ROOT/t-amp.sh" 'A&B|C' >/dev/null 2>&1
case $(sed -n 2p "$ROOT/t-amp.sh") in "# A&B|C") ok "a TITLE containing & and | is written verbatim";;
  *) bad "TITLE mangled: $(sed -n 2p "$ROOT/t-amp.sh")";; esac

# --- the generated script's own accounting -------------------------------------
gen 0 "leg green -- true"
runjob; check "$RC != 0" "an untouched EXPECT_LEGS=0 skeleton is RED, not a clean exit 0 over nothing"

gen oops "leg green -- true"
runjob; check "$RC != 0" "a MALFORMED EXPECT_LEGS is RED rather than failing open (rc=$RC)"
case $OUT in *"not a number"*) ok "the malformed-count refusal names the value";;
              *) bad "a malformed EXPECT_LEGS did not say so: $OUT";; esac

gen 2 "leg green -- true
leg red -- false"
runjob; check "$RC != 0" "one green leg and one RED leg exits non-zero (rc=$RC)"
case $OUT in *"### red rc=1"*) ok "the failing leg is named in the log with its status";;
              *) bad "the log does not name the failing leg: $OUT";; esac

# rc=2 is the status the motivating incident actually had (devq 806/810/814).
# An implementation that only accumulated rc==1 would pass the `false` case above.
gen 1 "leg two -- bash -o pipefail -c 'exit 2'"
runjob; check "$RC == 2" "a leg failing with rc=2 propagates as 2, not just 'nonzero-if-1' (rc=$RC)"

# pipefail is NOT inherited by a child shell; this exact input exited 0 before the fix
gen 1 "leg piped -- bash -c 'false | true'"
runjob; check "$RC != 0" "a pipeline inside a child-shell leg is RED (pipefail handed to the child)"
case $OUT in *"### piped rc=0"*) bad "the child-shell pipeline still reports rc=0";;
              *) ok "the child-shell pipeline leg records a failing status";; esac

# an empty leg does no work and must not count as a passing one
gen 1 "leg empty --"
runjob; check "$RC != 0" "a leg with NO command is a failure, not a free pass (rc=$RC)"

# a leg's status must survive a subshell -- a shell variable would not
gen 2 "leg one -- true & leg two -- true & wait"
runjob; check "$RC == 0" "backgrounded legs still COUNT (status kept in a file, not a variable)"
case $OUT in *"legs run: 2"*) ok "both backgrounded legs reached the tally";;
              *) bad "backgrounded legs were lost from the tally: $OUT";; esac

# an EXIT trap that calls exit would replace the status wholesale
gen 1 "trap 'exit 0' EXIT
leg red -- false"
runjob; check "$RC != 0" "an EXIT trap calling exit cannot turn a red job green (rc=$RC)"

# fewer legs than declared, and MORE legs than declared, are both wrong
gen 2 "leg only -- true"
runjob; check "$RC != 0" "running FEWER legs than declared is RED even when each one passed"
case $OUT in *"LEG COUNT WRONG"*) ok "the leg-count refusal says what it counted";;
              *) bad "a short leg count did not name itself: $OUT";; esac
gen 1 "leg a -- true
leg b -- true"
runjob; check "$RC != 0" "running MORE legs than declared is RED too (an -lt check would miss this)"

# and the both-green case must still pass, or the template is merely failing closed
gen 2 "leg a -- true
leg b -- true"
runjob; check "$RC == 0" "two green legs exit 0 (the template does not fail closed on everything)"

# a toolchain env that did not load means nothing below it measured what it claims
gen 1 "leg a -- true"
RC=0; OUT=$(TLENV=/nonexistent-tlenv R="$JR" bash "$JOBSH" 2>&1) || RC=$?
check "$RC != 0" "an unloadable toolchain env is RED rather than silently unset (rc=$RC)"

echo
printf 'RESULT: %d passed, %d failed\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
