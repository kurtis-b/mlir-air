# shellcheck shell=bash
#
# Status-board maintenance.
#
# Contract: docs_update_status_board() rewrites one phase's row in the plan README's status board
# from what the DRIVER measured -- wall time and outcome -- at the moment the phase completes.
#
# Why the driver owns this and a session cannot: the outcome does not exist while the session is
# running. It finishes, then three review rounds run, then the gate, then the objective check.
# Wall time, pass or fail, and invocation count are all downstream of that, so a session asked to
# update the board could only guess. Phases A through C4 all left the board stale for exactly this
# reason -- it said "not started" for work that was done.
#
# What this deliberately does NOT do: gate on documentation. A check of the form "the diff must
# touch the docs" is satisfied by a one-word edit, so it measures nothing and invites the edit.
# This function writes a fact the driver measured; it never asserts that a session wrote prose.
#
# Footguns:
#   - The caller must run this AFTER the tamper and destructive checks. Editing a tracked file
#     before them puts the driver's own change into the diff those checks inspect.
#   - The caller must COMMIT the result. pl_preflight() refuses to start the next phase on a dirty
#     tracked worktree, so an uncommitted board edit halts the following phase.
#   - A phase with no row in the board is a warning, not an error. The board is documentation;
#     failing a phase that passed every real gate because its row is missing would be absurd.

PL_STATUS_BOARD="docs/plans/transformer-layer-execution-studies/README.md"

# docs_update_status_board <phase-id> <wall-min> <outcome>
# Rewrites the status cell of the row whose first column names <phase-id>.
docs_update_status_board() {
  local phase="$1" wall_min="$2" outcome="${3:-passed}"
  local board="${PL_ROOT}/${PL_STATUS_BOARD}"

  if [ ! -f "${board}" ]; then
    log_warn "status board ${PL_STATUS_BOARD} not found; not updating"
    return 0
  fi

  local status_cell
  case "${outcome}" in
    passed) status_cell="**done** $(date +%Y-%m-%d) (${wall_min} min)" ;;
    *)      status_cell="**${outcome}** $(date +%Y-%m-%d) (${wall_min} min)" ;;
  esac

  PL_BOARD="${board}" PL_PHASE="${phase}" PL_CELL="${status_cell}" python3 -c '
import os, re, sys

board = os.environ["PL_BOARD"]
phase = os.environ["PL_PHASE"]
cell  = os.environ["PL_CELL"]

lines = open(board, encoding="utf-8").read().split("\n")

# A status-board row is "| <phase label> | <gate> | <status> |". Match on the FIRST cell naming
# this phase, so "C1" never matches "C10" and a phase name containing a digit never matches by
# accident. The label is "<id>" or "<id> <dash> <words>"; both em dash and hyphen are accepted
# because the board is hand-maintained prose.
pat = re.compile(r"^\|\s*" + re.escape(phase) + r"\s*(?:[—-]\s*[^|]*)?\|")

hits = [i for i, ln in enumerate(lines) if pat.match(ln)]
if not hits:
    print("no status-board row for phase %s" % phase, file=sys.stderr)
    sys.exit(2)
if len(hits) > 1:
    print("phase %s matches %d status-board rows; refusing to guess" % (phase, len(hits)),
          file=sys.stderr)
    sys.exit(3)

i = hits[0]
cells = lines[i].split("|")
# "| a | b | c |" splits to ["", " a ", " b ", " c ", ""] -- the status cell is the last real one.
if len(cells) < 5:
    print("status-board row for %s has %d columns, expected at least 3"
          % (phase, len(cells) - 2), file=sys.stderr)
    sys.exit(4)
old = cells[-2].strip()
if old == cell:
    print("  status board already reads: %s" % cell)
    sys.exit(0)
cells[-2] = " %s " % cell
lines[i] = "|".join(cells)

open(board, "w", encoding="utf-8").write("\n".join(lines))
print("  status board: %s  %s -> %s" % (phase, old, cell))
' || {
    local rc=$?
    case "${rc}" in
      2) log_warn "no status-board row for phase ${phase}; add one to ${PL_STATUS_BOARD}" ;;
      *) log_warn "could not update the status board for phase ${phase} (rc=${rc})" ;;
    esac
    return 0
  }
  return 0
}

# docs_commit_status_board <phase>
# Commits ONLY the board. A blanket `git add -A` here would sweep up whatever the gate left in the
# working tree and attribute it to the driver.
docs_commit_status_board() {
  local phase="$1"
  local board="${PL_ROOT}/${PL_STATUS_BOARD}"
  [ -f "${board}" ] || return 0

  if git -C "${PL_ROOT}" diff --quiet -- "${PL_STATUS_BOARD}" 2>/dev/null; then
    return 0
  fi
  git -C "${PL_ROOT}" add -- "${PL_STATUS_BOARD}" >/dev/null 2>&1 || return 0
  PRE_COMMIT_ALLOW_NO_CONFIG=1 git -C "${PL_ROOT}" commit -q -m \
    "port-loop: phase ${phase} status board

Written by the driver from what it measured, not by the session -- the outcome
does not exist while the session is running.
" >/dev/null 2>&1 \
    && log_info "  committed the status-board row for ${phase}" \
    || log_warn "  could not commit the status-board row for ${phase}"
  return 0
}
