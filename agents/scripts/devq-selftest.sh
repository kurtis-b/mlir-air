#!/usr/bin/env bash
# devq-selftest.sh -- prove the four scheduling properties of devq.sh with sleep
# jobs and NO device.  Runs against a throwaway DEVQ_DIR and a throwaway device
# lock so it never touches the real queue or the real NPU lock.
set -uo pipefail

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

echo
printf 'RESULT: %d passed, %d failed\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
