#!/usr/bin/env bash
# Hermetic tests for check_pr_deps.sh: a stubbed `gh` on PATH, exit codes 0/1/2.
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
cat > "$WORK/gh" <<'STUB'
#!/usr/bin/env bash
# stub for: gh pr view <n> --json state --jq .state
[ -z "${GH_REPO_LOG:-}" ] || echo "${GH_REPO:-unset}" >> "$GH_REPO_LOG"
pr="$3"
case "$pr" in
  1) echo MERGED;;
  2) echo OPEN;;
  3) echo CLOSED;;
  *) exit 1;;
esac
STUB
chmod +x "$WORK/gh"
export PATH="$WORK:$PATH"

pass=0; fail=0
expect() { # expect <exit-code> [pr ...]
  local want="$1"; shift
  bash "$DIR/check_pr_deps.sh" "$@" >/dev/null 2>&1
  local got=$?
  if [ "$got" = "$want" ]; then pass=$((pass+1)); echo "PASS  [exit $want] deps: $*"
  else fail=$((fail+1)); echo "FAIL  [want $want, got $got] deps: $*"; fi
}

expect 0 1       # merged dependency -> DEPS-OK
expect 0 1 1     # all merged -> DEPS-OK
expect 1 2       # open PR blocks
expect 1 3       # closed-without-merge blocks
expect 1 99      # not-found blocks
expect 1 1 2     # one blocker blocks the set
expect 2         # no arguments -> usage error

echo "--- two-remote clone: gh is pinned to origin, never the upstream remote ---"
REPO="$WORK/two-remote"; git init -q "$REPO"
git -C "$REPO" remote add upstream https://github.com/Xilinx/mlir-air
git -C "$REPO" remote add origin git@github.com:kurtis-b/mlir-air.git
export GH_REPO_LOG="$WORK/gh_repo.log"; : > "$GH_REPO_LOG"
( cd "$REPO" && env -u GH_REPO bash "$DIR/check_pr_deps.sh" 1 >/dev/null 2>&1 )
if grep -qx 'kurtis-b/mlir-air' "$GH_REPO_LOG"; then pass=$((pass+1)); echo "PASS  gh saw GH_REPO=kurtis-b/mlir-air"
else fail=$((fail+1)); echo "FAIL  gh saw GH_REPO=$(cat "$GH_REPO_LOG")"; fi
unset GH_REPO_LOG

echo; echo "== $pass passed, $fail failed =="
exit $((fail > 0))
