#!/usr/bin/env bash
# Dependency gate: start work only when every dependency PR is merged (no stacked PRs).
# Usage: scripts/check_pr_deps.sh <pr-number> [<pr-number> ...]
# Prints DEPS-OK and exits 0 iff every argument PR is MERGED; DEPS-BLOCKED and exit 1 otherwise.
# A PR that is OPEN, CLOSED without merge, or not found all block: the dependency never landed.
set -euo pipefail
# Pin gh to the repository behind `origin`: a two-remote clone otherwise resolves PR numbers
# against the remote named `upstream` (mlir-air's pull-only Xilinx fork parent).
if [ -z "${GH_REPO:-}" ] && url="$(git remote get-url origin 2>/dev/null)"; then
  GH_REPO="$(printf '%s' "$url" | sed -E 's#^(https?://[^/]+/|ssh://[^/]+/|[^@]+@[^:]+:)##; s#\.git/?$##')"; export GH_REPO
fi
[ $# -ge 1 ] || { echo "usage: $0 <pr-number> [...]" >&2; exit 2; }
fail=0
for pr in "$@"; do
  if state=$(gh pr view "$pr" --json state --jq .state 2>/dev/null); then
    if [ "$state" = "MERGED" ]; then
      echo "DEP-OK: PR #$pr MERGED"
    else
      echo "DEP-FAIL: PR #$pr state=$state (need MERGED — waiting on human review)"
      fail=1
    fi
  else
    echo "DEP-FAIL: PR #$pr not found"
    fail=1
  fi
done
if [ "$fail" -eq 0 ]; then
  echo "DEPS-OK"
else
  echo "DEPS-BLOCKED: do not start this task."
  exit 1
fi
