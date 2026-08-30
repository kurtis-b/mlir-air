#!/usr/bin/env bash
# devq.sh -- FIFO device job broker for the single shared NPU.
#
# WHY NOT A READERS-WRITER LOCK.  The obvious design (builds take `flock -s`,
# measurements take `flock -x`) does not serialise: Linux flock(2) documents no
# FIFO order, no writer preference and no fairness, so churning readers can starve
# a queued writer indefinitely.  This is a real queue instead: jobs get monotonic sequence
# numbers and a measurement at the head is an absolute barrier -- later builds are
# not admitted until it has run.
#
# ONE DEVICE LOCK.  The only lock that gates execution is /tmp/mlir-air-npu.lock,
# taken `flock -x -w 1800` by measure jobs, for compatibility with code that has
# not migrated to the queue.  A second device lock would be actively harmful during
# migration.  The queue's own state mutex is an flock on the state DIRECTORY's file
# descriptor -- no new file, held for microseconds, never across job execution.
#
# STATUS RECONCILIATION, NOT EXIT TRAPS.  SIGKILL cannot be caught, so completion
# records are unreliable.  Every scheduling decision and every `status` first
# reconciles: a 'running' entry whose runner pid is gone is marked failed, and its
# orphaned process group is reaped (killing a `flock` wrapper does NOT release the
# lock -- the forked child still holds the open file description, so we kill the
# descendants, and the runner holds the lock on its own fd so death releases it).
#
# RUN IS THE ONE TO USE FROM A GATE.  `submit` sends the job's output to the job
# log and prints only an id, so substituting it for a bare `flock CMD` silently
# swallows everything the gate printed -- and a gate whose output vanished can
# still exit 0.  `run` submits, relays the log to stdout as it grows, and exits
# with the job's own status, so it is a drop-in for `flock -x LOCK CMD`.
#
# NEVER `tee /dev/stderr` (OR /dev/stdout) INSIDE A JOB.  `[2026-08-12]`, queue
# item 20.  A job's log is held open here `O_APPEND` on ONE fd, but /dev/stderr
# re-opens that same regular file at offset 0 -- so the tee writes over the log
# from the top.  Measured: job 266 lost 10 of its 13 legs this way and STILL
# PRINTED PASS, because the verdict is emitted last and therefore survives the
# overwrite.  That is the worst available failure shape on a project whose first
# rule is no claim without an artifact: the artifact is destroyed while the
# verdict is preserved, and a truncated log is indistinguishable from a short
# clean run unless you count the legs.  Capture to a separate file and `cat` it,
# or just let the job write to stdout.  When READING any job log as evidence,
# count the legs against what the script should have emitted; do not trust a
# trailing verdict on its own.  `devq.sh log` cannot detect this after the fact
# -- the bytes are gone -- which is why the rule is at the writing end.
#
# PREFLIGHT MAKES THE ADVISORY LOCK ASKABLE.  `[2026-08-12]`, queue item 19: a
# `--compile-mode` flag was parsed and never branched on, so an intended
# compile-only run dispatched to the NPU off-queue WHILE job 252 held the device
# lock for the 65-minute ten-model regression.  252 survived only because
# contention pushes a correctness gate toward false FAILURE, not false pass.
# Nothing outside this broker consults the lock, so it stops nothing.
#
# `preflight` is the smallest thing that changes that: a READ-ONLY subcommand a
# dispatching script can call to ask "is the device busy?" and get an exit code.
# It is deliberately NOT a new mechanism -- it exposes `device_idle` below, the
# probe this broker has always used to keep builds off the box, so there is ONE
# probe and ONE lock path.  It never holds the lock across anything (microseconds
# inside a subshell, exactly as build admission does), it never touches
# /tmp/npu.lock (a different inode KernelCache owns -- taking it from a wrapper
# deadlocks the suites), and it can only ever REFUSE: nothing acquires the device
# through preflight, so it cannot become the way to get it without queueing.
#
# Inside a devq job it exits 0 WITHOUT probing.  A measure runner holds the lock
# on its own fd, so probing there would report the caller's own job as the
# contender and refuse every legitimate measurement -- the same nesting trap
# `run` refuses for, seen from the other side.
#
# Usage:
#   devq.sh run    --class build|measure [--name TAG] -- CMD [ARG...]
#   devq.sh submit --class build|measure [--name TAG] [--wait] -- CMD [ARG...]
#   devq.sh preflight [--quiet]        # 0 = device free, 3 = held by another job
#   devq.sh status [--raw]
#   devq.sh wait ID [--timeout SEC]
#   devq.sh log ID
#   devq.sh new-job FILE [TITLE]  # write a job-script skeleton that cannot exit 0 on red legs
set -uo pipefail

SELF=$(readlink -f "${BASH_SOURCE[0]}")
# ONE QUEUE PER HOST, not per worktree: the NPU is one device.  The default lives
# under the MAIN checkout's agents/.state (git's common dir is shared by every
# worktree of a repository); outside a repository it falls back to /tmp.
_devq_common=$(git -C "$(dirname "$SELF")" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)
DEVQ_DIR=${DEVQ_DIR:-${_devq_common:+$(dirname "$_devq_common")/agents/.state/devq}}
DEVQ_DIR=${DEVQ_DIR:-/tmp/mlir-air-devq}
NPU_LOCK=${DEVQ_NPU_LOCK:-/tmp/mlir-air-npu.lock}   # override is TEST-ONLY
NPU_LOCK_WAIT=${DEVQ_NPU_LOCK_WAIT:-1800}
POLL=${DEVQ_POLL:-0.2}
SPAWN_GRACE=${DEVQ_SPAWN_GRACE:-30}
PROBE=${DEVQ_MIGRATION_PROBE:-1}   # builds defer while an un-migrated job holds the device lock
JOBS="$DEVQ_DIR/jobs"

die() { printf 'devq: %s\n' "$*" >&2; exit 2; }
now() { printf '%s' "${EPOCHREALTIME/,/.}"; }
ensure() { mkdir -p "$JOBS" || die "cannot create $JOBS"; }
metaf() { printf '%s/job-%06d.meta' "$JOBS" "$1"; }
logf()  { printf '%s/job-%06d.log'  "$JOBS" "$1"; }

lock_state()   { exec 9<"$DEVQ_DIR" || die "cannot open state dir"; flock -x -w 60 9 || die "state mutex timeout"; }
unlock_state() { flock -u 9 2>/dev/null; exec 9<&- 2>/dev/null; }

read_meta() {  # $1 = meta path -> M_* vars
  M_id= M_class= M_name= M_state= M_submitted= M_started= M_finished= M_exit= M_pid= M_note= M_cmd=
  local k v
  while IFS='=' read -r k v; do
    case $k in
      id) M_id=$v;; class) M_class=$v;; name) M_name=$v;; state) M_state=$v;;
      submitted) M_submitted=$v;; started) M_started=$v;; finished) M_finished=$v;;
      exit) M_exit=$v;; pid) M_pid=$v;; note) M_note=$v;; cmd) M_cmd=$v;;
    esac
  done < "$1"
}

write_meta() {  # $1 = meta path, from M_* vars; atomic rename
  local t="$1.tmp.$$"
  {
    printf 'id=%s\nclass=%s\nname=%s\nstate=%s\n' "$M_id" "$M_class" "$M_name" "$M_state"
    printf 'submitted=%s\nstarted=%s\nfinished=%s\nexit=%s\n' "$M_submitted" "$M_started" "$M_finished" "$M_exit"
    printf 'pid=%s\nnote=%s\ncmd=%s\n' "$M_pid" "$M_note" "$M_cmd"
  } > "$t" && mv -f "$t" "$1"
}

# Liveness that is immune to pid reuse: the pid must exist AND its exec-time
# environment must carry this job's marker.
runner_alive() {  # pid id
  [ -n "$1" ] || return 1
  kill -0 "$1" 2>/dev/null || return 1
  grep -qaz "^DEVQ_JOB_ID=$2\$" "/proc/$1/environ" 2>/dev/null
}

reap_group() {  # pgid id -- kill descendants still carrying this job's marker
  local p
  for p in $(pgrep -g "$1" 2>/dev/null); do
    grep -qaz "^DEVQ_JOB_ID=$2\$" "/proc/$p/environ" 2>/dev/null && kill -KILL "$p" 2>/dev/null
  done
}

reconcile() {  # caller holds the state mutex
  # M_* are declared local so read_meta below cannot clobber the caller's job record.
  local f t M_id M_class M_name M_state M_submitted M_started M_finished M_exit M_pid M_note M_cmd
  t=$(now)
  for f in "$JOBS"/job-*.meta; do
    [ -e "$f" ] || continue
    read_meta "$f"
    case $M_state in
      running)
        runner_alive "$M_pid" "$M_id" && continue
        reap_group "$M_pid" "$M_id"
        M_state=failed; M_exit=137; M_finished=$t; M_note="reconciled: runner pid $M_pid gone"
        write_meta "$f" ;;
      queued)
        if [ -n "$M_pid" ]; then
          runner_alive "$M_pid" "$M_id" && continue
          M_state=failed; M_exit=137; M_finished=$t; M_note="reconciled: runner died before start"
          write_meta "$f"
        elif awk "BEGIN{exit !($t - $M_submitted > $SPAWN_GRACE)}"; then
          M_state=failed; M_exit=127; M_finished=$t; M_note="reconciled: runner never started"
          write_meta "$f"
        fi ;;
    esac
  done
}

# FIFO admission rule (caller holds the mutex, after reconcile):
#   build   may start iff no OLDER job that is a measure is still queued/running
#   measure may start iff NO older job at all is still queued/running
# Consecutive builds at the head therefore run concurrently; a measure drains the
# builds ahead of it; and builds submitted AFTER a measure cannot pass it.
eligible() {  # id class
  local want_id=$1 want_class=$2 f
  local M_id M_class M_name M_state M_submitted M_started M_finished M_exit M_pid M_note M_cmd
  for f in "$JOBS"/job-*.meta; do
    [ -e "$f" ] || continue
    read_meta "$f"
    [ "$M_id" -ge "$want_id" ] && continue
    case $M_state in queued|running) ;; *) continue;; esac
    [ "$want_class" = measure ] && return 1
    [ "$M_class" = measure ] && return 1
  done
  return 0
}

# Non-blocking probe of the compatibility lock.  Held for microseconds inside a
# subshell; used only to keep a build off the box while an un-migrated measurement
# still holds /tmp/mlir-air-npu.lock.  Never blocks and never starves anyone.
device_idle() { ( exec 7>>"$NPU_LOCK" && flock -n -x 7 ) 2>/dev/null; }

cmd_submit() {
  local class= name=- dowait=0
  while [ $# -gt 0 ]; do
    case $1 in
      --class) class=${2:-}; shift 2;;
      --name)  name=${2:-}; shift 2;;
      --wait)  dowait=1; shift;;
      --) shift; break;;
      *) die "submit: unexpected argument '$1' (did you forget --?)";;
    esac
  done
  case $class in build|measure) ;; *) die "submit: --class must be build or measure";; esac
  [ $# -gt 0 ] || die "submit: no command given after --"
  ensure
  local cmdstr id
  cmdstr=$(printf '%q ' "$@")
  lock_state
  id=$(( $(cat "$DEVQ_DIR/seq" 2>/dev/null || echo 0) + 1 ))
  # Every write is checked: an id printed without its record would leave `run`
  # polling a job that does not exist (the script deliberately runs without set -e).
  printf '%s\n' "$id" > "$DEVQ_DIR/seq" || { unlock_state; die "submit: cannot write $DEVQ_DIR/seq"; }
  M_id=$id M_class=$class M_name=$name M_state=queued M_submitted=$(now) \
    M_started= M_finished= M_exit= M_pid= M_note= M_cmd=$cmdstr
  write_meta "$(metaf "$id")" || { unlock_state; die "submit: cannot write $(metaf "$id")"; }
  : >> "$DEVQ_DIR/runner.log" || { unlock_state; die "submit: cannot write $DEVQ_DIR/runner.log"; }
  unlock_state
  # DEVQ_JOB_ID must be in the exec-time environment: /proc/<pid>/environ does not
  # reflect variables exported after exec, and reconciliation reads it.
  DEVQ_JOB_ID=$id setsid "$SELF" __run "$id" >>"$DEVQ_DIR/runner.log" 2>&1 </dev/null &
  disown 2>/dev/null
  printf '%s\n' "$id"
  [ "$dowait" = 1 ] && { cmd_wait "$id"; return $?; }
  return 0
}

cmd_run() {  # internal: the per-job runner
  local id=$1 f rc cls
  f=$(metaf "$id")
  lock_state; read_meta "$f"; M_pid=$$; write_meta "$f"; unlock_state
  while :; do
    lock_state
    reconcile
    read_meta "$f"
    [ "$M_state" != queued ] && { unlock_state; exit 0; }   # reconciled away
    if eligible "$id" "$M_class" && { [ "$M_class" = measure ] || [ "$PROBE" != 1 ] || device_idle; }; then
      M_state=running; M_started=$(now); M_pid=$$; write_meta "$f"
      unlock_state; break
    fi
    unlock_state
    sleep "$POLL"
  done
  cls=$M_class
  if [ "$cls" = measure ]; then
    # The runner holds the device lock on its own fd, so if the runner is SIGKILLed
    # the fd closes and the lock is released; the job's child gets it closed (8>&-)
    # so it can never keep the lock alive behind us.
    : >>"$NPU_LOCK" 2>/dev/null
    exec 8>>"$NPU_LOCK"
    if ! flock -x -w "$NPU_LOCK_WAIT" 8; then
      lock_state; read_meta "$f"
      M_state=failed M_exit=124 M_finished=$(now) M_note="device lock unavailable after ${NPU_LOCK_WAIT}s"
      write_meta "$f"; unlock_state; exit 124
    fi
  fi
  bash -c "$M_cmd" >>"$(logf "$id")" 2>&1 8>&- 9<&- </dev/null
  rc=$?
  [ "$cls" = measure ] && exec 8>&-
  lock_state; read_meta "$f"
  M_finished=$(now); M_exit=$rc
  [ "$rc" = 0 ] && M_state=done || M_state=failed
  write_meta "$f"; unlock_state
  exit "$rc"
}

cmd_status() {
  local raw=0; [ "${1:-}" = "--raw" ] && raw=1
  ensure; lock_state; reconcile; unlock_state
  local f t; t=$(now)
  [ "$raw" = 1 ] || printf '%-5s %-7s %-7s %-4s %-8s %8s  %s\n' ID CLASS STATE EXIT PID ELAPSED CMD
  for f in "$JOBS"/job-*.meta; do
    [ -e "$f" ] || continue
    read_meta "$f"
    if [ "$raw" = 1 ]; then
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$M_id" "$M_class" "$M_state" \
        "${M_exit:--}" "${M_submitted:-0}" "${M_started:-0}" "${M_finished:-0}" \
        "${M_pid:--}" "$M_name" "$M_cmd"
      continue
    fi
    local el=- ref=$t
    [ -n "$M_finished" ] && ref=$M_finished
    [ -n "$M_started" ] && el=$(awk "BEGIN{printf \"%.1fs\", $ref - $M_started}")
    printf '%-5s %-7s %-7s %-4s %-8s %8s  %.60s\n' \
      "$M_id" "$M_class" "$M_state" "${M_exit:--}" "${M_pid:--}" "$el" "$M_cmd"
    [ -n "$M_note" ] && printf '      note: %s\n' "$M_note"
  done
  return 0
}

# relay <logfile> <offset>
#
# Writes the bytes past <offset> to fd 3 (the caller's real stdout) and echoes the new
# offset on stdout, so the caller can do `off=$(relay "$lf" "$off")` without the payload
# and the bookkeeping fighting over one stream.
relay() {
  local lf=$1 off=$2 sz
  [ -e "$lf" ] || { printf '%s' "$off"; return 0; }
  sz=$(wc -c <"$lf" 2>/dev/null); sz=${sz//[^0-9]/}; sz=${sz:-0}
  if [ "$sz" -gt "$off" ]; then
    tail -c "+$((off + 1))" "$lf" | head -c "$((sz - off))" >&3   # never past the snapshot
    off=$sz
  fi
  printf '%s' "$off"
}

cmd_runjob() {
  local class= name=- id lf mf off=0
  while [ $# -gt 0 ]; do
    case $1 in
      --class) class=${2:-}; shift 2;;
      --name)  name=${2:-}; shift 2;;
      --) shift; break;;
      *) die "run: unexpected argument '$1' (did you forget --?)";;
    esac
  done
  case $class in build|measure) ;; *) die "run: --class must be build or measure";; esac
  [ $# -gt 0 ] || die "run: no command given after --"
  # A measure submitted from inside a running job would queue behind the device lock its
  # own parent runner is holding and sit there for NPU_LOCK_WAIT before failing 124 -- a
  # 30-minute stall reported as a lock timeout, a long way from the nested call that
  # caused it.  Refuse immediately instead.
  if [ -n "${DEVQ_JOB_ID:-}" ]; then
    die "run: already inside devq job ${DEVQ_JOB_ID}; nesting would deadlock on the device lock"
  fi
  ensure
  id=$(cmd_submit --class "$class" --name "$name" -- "$@") || return $?
  lf=$(logf "$id"); mf=$(metaf "$id")
  printf 'devq: job %s (%s) submitted\n' "$id" "$class" >&2
  exec 3>&1
  while :; do
    off=$(relay "$lf" "$off")
    lock_state; reconcile; unlock_state
    read_meta "$mf"
    case $M_state in
      done|failed)
        # Relay once more AFTER the terminal state is observed: cmd_run writes the state
        # only after its child has exited, so this pass cannot race the job's last write.
        off=$(relay "$lf" "$off")
        [ -n "$M_note" ] && printf 'devq: job %s %s (%s)\n' "$id" "$M_state" "$M_note" >&2
        return "${M_exit:-1}";;
    esac
    sleep "$POLL"
  done
}

# preflight [--quiet] -- read-only: may a dispatch start right now?
#
#   0  free, or the caller is already inside a devq job (the queue scheduled it)
#   3  the device lock is held by something else
#
# 3 rather than 1 so a caller can tell "busy" from its own command failing.
cmd_preflight() {
  local quiet=0; [ "${1:-}" = "--quiet" ] && quiet=1
  local say=printf; [ "$quiet" = 1 ] && say=:

  if [ -n "${DEVQ_JOB_ID:-}" ]; then
    $say 'devq: preflight OK -- inside devq job %s, already scheduled\n' "$DEVQ_JOB_ID"
    return 0
  fi
  if device_idle; then
    $say 'devq: preflight OK -- %s is free\n' "$NPU_LOCK"
    return 0
  fi

  # Held.  Name the holder if the queue knows it; an un-migrated job that took
  # the lock directly will not be in the queue, and saying so is the useful
  # answer rather than pretending nobody is there.
  local holder= f
  if [ -d "$JOBS" ]; then
    local M_id M_class M_name M_state M_submitted M_started M_finished M_exit M_pid M_note M_cmd
    for f in "$JOBS"/job-*.meta; do
      [ -e "$f" ] || continue
      read_meta "$f"
      [ "$M_state" = running ] && [ "$M_class" = measure ] && \
        holder="job $M_id (${M_name:--}) pid $M_pid: $M_cmd"
    done
  fi
  printf 'devq: preflight REFUSED -- the device lock %s is held%s\n' \
    "$NPU_LOCK" "${holder:+ by $holder}" >&2
  [ -n "$holder" ] || printf 'devq:   no running measure job in the queue owns it; something took the lock outside the broker\n' >&2
  printf 'devq:   dispatching now would run beside that job. Queue instead:\n' >&2
  printf 'devq:     %s run --class measure -- <your command>\n' "$SELF" >&2
  return 3
}

cmd_wait() {
  local id=${1:?wait: need an id} deadline= f
  # printf, not print: awk's default OFMT (%.6g) mangles epoch floats to 1.78e+09.
  [ "${2:-}" = "--timeout" ] && deadline=$(awk "BEGIN{printf \"%.6f\", $(now) + ${3:?}}")
  f=$(metaf "$id"); [ -e "$f" ] || die "wait: no such job $id"
  while :; do
    lock_state; reconcile; unlock_state
    read_meta "$f"
    case $M_state in
      done|failed) [ -n "$M_note" ] && printf 'devq: job %s %s (%s)\n' "$id" "$M_state" "$M_note" >&2
                   return "${M_exit:-1}";;
    esac
    if [ -n "$deadline" ] && awk "BEGIN{exit !($(now) > $deadline)}"; then
      printf 'devq: wait timeout, job %s still %s\n' "$id" "$M_state" >&2; return 125
    fi
    sleep "$POLL"
  done
}

# A job script's exit code is the ONLY thing devq records, and devq records it
# faithfully -- so a script that ends in an unconditional `exit 0`, or loses a
# leg's status to a pipe, makes the job read done/0 with every leg red.  That has
# now happened in four different scripts and burned three device slots on a
# silent failure (devq 806/810/814, all six profile legs rc=2, all reported
# done/0).  This writes the skeleton so the accounting is there before the author
# starts, rather than being remembered per script.

cmd_newjob() {
  local out=${1:?new-job: need a FILE} title=${2:-devq job}
  # -L as well as -e: a DANGLING symlink is not "absent". Without it, `cat >`
  # would create the target through the link and `sed -i` would then replace the
  # link itself with a regular file, silently leaving two files behind.
  { [ -e "$out" ] || [ -L "$out" ]; } && die "new-job: $out exists (refusing to overwrite a job script)"
  mkdir -p "$(dirname "$out")" || die "new-job: cannot create $(dirname "$out")"
  # The title is printf'd, never sed'd in: a TITLE containing & or | would
  # otherwise be taken as sed replacement syntax -- & expanding to the match, |
  # aborting the substitution -- and new-job would still report success.
  {
    printf '#!/bin/bash\n# %s\n' "$title"
    printf 'ROOT=%q   # repository root at generation time\n' "$(git -C "$(dirname "$SELF")" rev-parse --show-toplevel 2>/dev/null || pwd)"
    cat <<'SKEL'
#
# Generated by `devq.sh new-job`. The accounting below is the point of the
# template: devq records THIS SCRIPT'S exit code and nothing else, so a leg whose
# failure does not reach `exit` is a leg that silently passes. That has happened
# in four scripts and burned three device slots (devq 806/810/814: six profile
# legs at rc=2, job reported done/0).
#
# What the template guarantees, and what it CANNOT:
#
#   * every leg runs through `leg`, which records its status to a FILE -- so a
#     leg in a subshell or a background job still counts (a shell variable would
#     not survive the subshell, and the leg would vanish from the tally);
#   * a leg with NO command is a failure, not a free pass;
#   * `bash -c` / `sh -c` legs are re-invoked with `-o pipefail`, because
#     pipefail is NOT inherited by a child shell -- `leg x -- bash -c 'false |
#     true'` would otherwise report rc=0;
#   * EXPECT_LEGS must be a number and must match the legs actually run; a
#     malformed value is RED, not ignored;
#   * the toolchain env and the `cd` are CHECKED -- an env that failed to load is
#     not a job that ran.
#
# It cannot stop you from installing `trap '... exit 0' EXIT`, which replaces
# this script's status wholesale. The footer detects an EXIT trap containing
# `exit` and fails, but do not write one.
#
# Do not `tee` a leg's output into the devq log: the second writer reopens the
# log at offset 0 and shreds it, leaving a trailing PASS over legs that are gone.
# Write legs to files under $R/logs and grep them.
set -uo pipefail

TLENV=${TLENV:-$ROOT/agents/.state/tlenv.sh}   # per-host toolchain env (ignored file); override per job
if [ ! -r "$TLENV" ] || ! source "$TLENV" >/dev/null 2>&1; then
  echo "== FATAL: toolchain env $TLENV did not load; nothing below would be measuring what it claims"
  exit 1
fi
cd "$ROOT" || { echo "== FATAL: cd $ROOT failed"; exit 1; }

R=${R:-$ROOT/agents/.state/devq-results/CHANGEME}
mkdir -p "$R/logs" || { echo "== FATAL: cannot create $R/logs"; exit 1; }
LEDGER="$R/logs/.legs"
: > "$LEDGER" || { echo "== FATAL: cannot write $LEDGER"; exit 1; }

EXPECT_LEGS=0   # <-- set this to the number of legs you intend to run

pmode() { xrt-smi examine -r platform 2>/dev/null | grep -i 'Power Mode' | tr -s ' '; }

# leg NAME -- CMD...   : run one leg, log it, record its status where a subshell
#                        cannot lose it, and never swallow it
leg() {
  local name=$1; shift
  [ "${1:-}" = "--" ] && shift
  if [ $# -eq 0 ]; then
    echo ""; echo "################ leg: $name ################"
    echo "### $name NO COMMAND GIVEN -- an empty leg is a failure, not a passing leg"
    printf '%s\t2\n' "$name" >> "$LEDGER"
    return 2
  fi
  # pipefail does not cross into a child shell; hand it back explicitly
  if { [ "$1" = bash ] || [ "$1" = sh ] || [ "$1" = /bin/bash ] || [ "$1" = /bin/sh ]; } \
     && [ "${2:-}" = "-c" ]; then
    local sh0=$1; shift 2
    set -- "$sh0" -o pipefail -c "$@"
  fi
  local f="$R/logs/$name.txt" rc
  echo ""; echo "################ leg: $name ################"
  "$@" > "$f" 2>&1
  rc=$?
  printf '%s\t%s\n' "$name" "$rc" >> "$LEDGER"
  echo "### $name rc=$rc"
  tail -5 "$f"
  return $rc
}

echo "== $(date -Is) HEAD $(git rev-parse --short HEAD) (uncommitted: $(git status --short | grep -vc '^??') tracked)"
echo "== pmode before: $(pmode)"

# ---------------------------------------------------------------- legs go here
# leg build -- make -C some/dir
# leg verify -- python3 some/check.py
# ------------------------------------------------------------------------------

echo ""
echo "== pmode after: $(pmode)"

RC_ALL=0
NLEG=0
while IFS=$'\t' read -r _lname _lrc; do
  NLEG=$((NLEG + 1))
  [ "$_lrc" -ne 0 ] 2>/dev/null && RC_ALL=$_lrc
done < "$LEDGER"

case $EXPECT_LEGS in
  ''|*[!0-9]*)
    echo "== EXPECT_LEGS is not a number: '$EXPECT_LEGS' -- refusing to guess what this job intended"
    RC_ALL=1 ;;
  0)
    echo "== EXPECT_LEGS IS STILL 0 -- set it, or this job cannot tell 'all passed' from 'nothing ran'"
    RC_ALL=1 ;;
  *)
    [ "$NLEG" -ne "$EXPECT_LEGS" ] && {
      echo "== LEG COUNT WRONG: ran $NLEG, expected $EXPECT_LEGS"
      RC_ALL=1
    } ;;
esac

# Detecting this is not enough: the trap fires on our own `exit` and replaces the
# status wholesale, so a warning alone still exits 0. Clear it. A trap that does
# NOT call exit cannot change the status and is left alone to do its cleanup.
if trap -p EXIT | grep -q 'exit'; then
  echo "== AN EXIT TRAP CALLS exit -- it would REPLACE this script's status. CLEARED, and this job is RED:"
  echo "==   $(trap -p EXIT)"
  echo "== a job that reports its cleanup's status instead of its legs' has been lying; fix the trap."
  trap - EXIT
  RC_ALL=1
fi

echo "== legs run: $NLEG (expect $EXPECT_LEGS)"
echo "== OVERALL RC=$RC_ALL"
exit "$RC_ALL"
SKEL
  } > "$out" || die "new-job: cannot write $out"
  chmod +x "$out" || die "new-job: cannot chmod $out"
  [ -s "$out" ] || die "new-job: wrote an empty $out"
  printf 'devq: wrote %s\n' "$out" >&2
  printf '%s\n' "$out"
}

case ${1:-} in
  run)    shift; cmd_runjob "$@";;
  submit) shift; cmd_submit "$@";;
  preflight) shift; cmd_preflight "$@";;
  status) shift; cmd_status "$@";;
  wait)   shift; cmd_wait "$@";;
  log)    shift; cat "$(logf "${1:?log: need an id}")";;
  new-job) shift; cmd_newjob "$@";;
  __run)  shift; cmd_run "$@";;
  *) sed -n '/^# Usage:/,/^set -uo/p' "$SELF" | sed '$d;s/^# \{0,1\}//'; exit 2;;
esac
