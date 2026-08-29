#!/usr/bin/env bash
# Hermetic test matrix for origin-only-guard.mjs (no git state needed: the guard
# reads only the command text). Same harness shape as test_guard.sh.
set -u
HOOK="$(cd "$(dirname "$0")" && pwd)/origin-only-guard.mjs"
pass=0; fail=0
run() { # run <expected: deny|allow|ask> <command...>
  local expect="$1"; shift
  local cmd="$*" json out decision
  json=$(node -e 'process.stdout.write(JSON.stringify({tool_name:"Bash",tool_input:{command:process.argv[1]},cwd:"/x"}))' "$cmd")
  out=$(printf '%s' "$json" | node "$HOOK" 2>/dev/null)
  if [[ -z "$out" ]]; then decision=allow
  elif grep -q '"permissionDecision":"deny"' <<<"$out"; then decision=deny
  elif grep -q '"permissionDecision":"ask"' <<<"$out"; then decision=ask
  else decision="unknown($out)"; fi
  if [[ "$decision" == "$expect" ]]; then pass=$((pass+1)); echo "PASS  [$expect] $cmd"
  else fail=$((fail+1)); echo "FAIL  [expected $expect, got $decision] $cmd"; fi
}

echo "--- U1: no push to upstream ---"
run deny  git push upstream feat/x
run deny  git push -u upstream feat/x
run deny  git push --repo upstream HEAD
run deny  git push --repo=upstream feat/x
run deny  git push https://github.com/Xilinx/mlir-air HEAD:feat/x
run deny  git push git@github.com:Xilinx/mlir-air.git HEAD:feat/x
run deny  "git fetch upstream && git push upstream HEAD"
run deny  'git push "upstream" feat/x'
run allow git push -u origin feat/x
run allow git push origin HEAD
run allow git push https://github.com/kurtis-b/mlir-air.git feat/x
run allow git push
echo "--- U2: the pull-only remote stays neutered ---"
run deny  git remote set-url upstream https://github.com/Xilinx/mlir-air
run deny  git remote set-url --push upstream https://github.com/Xilinx/mlir-air
run deny  git remote add up2 https://github.com/Xilinx/mlir-air
run deny  git remote rename upstream up
run deny  git config remote.upstream.pushurl https://github.com/Xilinx/mlir-air
run deny  git config --unset remote.upstream.pushurl
run allow git remote -v
run allow git remote get-url upstream
run allow git remote show upstream
run allow git config --get remote.upstream.url
run allow git remote add fork2 https://github.com/kurtis-b/mlir-air.git
run allow git fetch upstream
run allow git fetch upstream main
run allow git merge origin/main
echo "--- U3: gh writes addressed to upstream ---"
run deny  gh pr create -R Xilinx/mlir-air --fill
run deny  gh --repo Xilinx/mlir-air pr create --fill
run deny  gh pr create --repo=Xilinx/mlir-air --fill
run deny  gh pr comment 5 -R xilinx/mlir-air -b hi
run deny  gh pr edit 5 -R Xilinx/mlir-air --title x
run deny  gh issue create -R Xilinx/mlir-air --title x
run deny  gh release create v1 -R Xilinx/mlir-air
run deny  gh repo set-default Xilinx/mlir-air
run deny  gh api -X POST repos/Xilinx/mlir-air/pulls -f title=x
run deny  gh api --method PATCH repos/Xilinx/mlir-air/pulls/5
run deny  gh api repos/Xilinx/mlir-air/issues -f title=x
run deny  'gh api graphql -f query="mutation { createPullRequest(input: {repositoryId: \"Xilinx/mlir-air\"}) { clientMutationId } }"'
run allow gh pr view 1959 -R Xilinx/mlir-air
run allow gh pr list -R Xilinx/mlir-air --state open
run allow gh pr checks 1959 --repo Xilinx/mlir-air
run allow gh api repos/Xilinx/mlir-air --jq .permissions
run allow gh api repos/Xilinx/mlir-air/pulls/1959 --jq .state
run allow gh api -X GET repos/Xilinx/mlir-air/commits/main
run allow gh pr create -R kurtis-b/mlir-air --fill
run allow gh pr create --fill
run allow gh pr comment 3 -b hi
run allow gh api -X POST repos/kurtis-b/mlir-air/issues -f title=x
run allow gh repo set-default kurtis-b/mlir-air
echo "--- prose in quoted args must not trip the scan ---"
run allow 'git commit -m "never git push upstream or open PRs on Xilinx/mlir-air"'
run allow 'gh pr comment 3 -b "upstream Xilinx/mlir-air is pull-only; git push upstream is denied"'
echo "--- malformed input degrades to ask ---"
out=$(printf 'not json' | node "$HOOK" 2>/dev/null)
if grep -q '"permissionDecision":"ask"' <<<"$out"; then pass=$((pass+1)); echo "PASS  [ask] malformed JSON"
else fail=$((fail+1)); echo "FAIL  [expected ask] malformed JSON -> $out"; fi

echo; echo "== $pass passed, $fail failed =="
exit $((fail > 0))
