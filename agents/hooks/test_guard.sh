#!/usr/bin/env bash
# Self-contained test matrix for main-branch-guard.mjs. Pipes fake PreToolUse
# JSON through the hook and checks the decision (deny / allow / ask).
set -u
HOOK="$(cd "$(dirname "$0")" && pwd)/main-branch-guard.mjs"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

git init -q -b main "$WORK/on-main"
(cd "$WORK/on-main" && git commit -q --allow-empty -m seed)
git init -q -b main "$WORK/on-feat"
(cd "$WORK/on-feat" && git commit -q --allow-empty -m seed && git switch -qc feat/x)
git init -q -b main "$WORK/on-feat/submod"
(cd "$WORK/on-feat/submod" && git commit -q --allow-empty -m seed)

MAIN="$WORK/on-main" FEAT="$WORK/on-feat"
pass=0; fail=0
run() { # run <expected: deny|allow|ask> <project_dir> <cwd> <command...>
  local expect="$1" proj="$2" cwd="$3"; shift 3
  local cmd="$*" json out decision
  json=$(node -e 'const [cwd,cmd]=process.argv.slice(1);process.stdout.write(JSON.stringify({tool_name:"Bash",tool_input:{command:cmd},cwd}))' "$cwd" "$cmd")
  out=$(printf '%s' "$json" | CLAUDE_PROJECT_DIR="$proj" node "$HOOK" 2>/dev/null)
  if [[ -z "$out" ]]; then decision=allow
  elif grep -q '"permissionDecision":"deny"' <<<"$out"; then decision=deny
  elif grep -q '"permissionDecision":"ask"' <<<"$out"; then decision=ask
  else decision="unknown($out)"; fi
  if [[ "$decision" == "$expect" ]]; then pass=$((pass+1)); echo "PASS  [$expect] $cmd"
  else fail=$((fail+1)); echo "FAIL  [expected $expect, got $decision] $cmd"; fi
}

echo "--- G2: main may never be checked out ---"
run deny  "$FEAT" "$FEAT" git switch main
run deny  "$FEAT" "$FEAT" git checkout main
run deny  "$FEAT" "$FEAT" git switch -
run deny  "$FEAT" "$FEAT" git checkout -
run deny  "$FEAT" "$FEAT" 'git switch "main"'
run deny  "$FEAT" "$FEAT" git checkout main -- somefile
run deny  "$FEAT" "$FEAT" git switch -c main
run deny  "$FEAT" "$FEAT" git checkout -B main origin/main
run deny  "$FEAT" "$FEAT" git switch --create=main
run deny  "$FEAT" "$FEAT" "git fetch origin && git switch main"
run allow "$FEAT" "$FEAT" git switch -c feat/y origin/main
run allow "$FEAT" "$FEAT" git switch --create=feat/z
run allow "$FEAT" "$FEAT" git checkout -b feat/w main
run allow "$FEAT" "$FEAT" git checkout -- file
run allow "$FEAT" "$FEAT" git checkout .
run allow "$FEAT" "$FEAT" git switch feat/other
echo "--- G1 belt: commit-creating on main ---"
run deny  "$MAIN" "$MAIN" git commit -m x
run deny  "$MAIN" "$MAIN" git commit --amend --no-edit
run deny  "$MAIN" "$MAIN" git rebase -i HEAD~3
run deny  "$MAIN" "$MAIN" git cherry-pick abc123
run deny  "$MAIN" "$MAIN" "cd $MAIN && git commit -m x"
run allow "$FEAT" "$FEAT" git commit -m x
run allow "$FEAT" "$FEAT" git rebase origin/main
run allow "$FEAT" "$FEAT" git cherry-pick abc123
echo "--- G3: merge and reset --hard ---"
run deny  "$FEAT" "$FEAT" git merge feat/other
run deny  "$FEAT" "$FEAT" git merge main
run deny  "$MAIN" "$MAIN" git merge origin/main
run allow "$FEAT" "$FEAT" git merge origin/main
run allow "$FEAT" "$FEAT" 'git merge origin/main -m "sync from main"'
run allow "$FEAT" "$FEAT" git merge --continue
run allow "$FEAT" "$FEAT" git merge --abort
run deny  "$FEAT" "$FEAT" git reset --hard
run deny  "$FEAT" "$FEAT" git reset --hard HEAD~1
run deny  "$MAIN" "$MAIN" git reset --soft HEAD~1
run allow "$FEAT" "$FEAT" git reset --soft HEAD~1
run allow "$FEAT" "$FEAT" git reset HEAD~1 -- file
echo "--- G4: push protections ---"
run deny  "$FEAT" "$FEAT" git push -f origin feat/x
run deny  "$FEAT" "$FEAT" git push --force
run deny  "$FEAT" "$FEAT" git push --force-with-lease=feat/x origin feat/x
run deny  "$FEAT" "$FEAT" git push --force-if-includes origin feat/x
run deny  "$FEAT" "$FEAT" git push --mirror origin
run deny  "$FEAT" "$FEAT" git push --all origin
run deny  "$FEAT" "$FEAT" git push --delete origin feat/old
run deny  "$FEAT" "$FEAT" git push -d origin feat/old
run deny  "$FEAT" "$FEAT" git push origin main
run deny  "$FEAT" "$FEAT" git push origin feat/x:main
run deny  "$FEAT" "$FEAT" git push origin HEAD:main
run deny  "$FEAT" "$FEAT" git push origin +main
run deny  "$FEAT" "$FEAT" 'git push origin "main"'
run deny  "$FEAT" "$FEAT" 'git push origin $TARGET'
run deny  "$FEAT" "$FEAT" 'git push origin refs/heads/*'
run deny  "$FEAT" "$FEAT" git push --repo origin main
run deny  "$FEAT" "$FEAT" git push --repo=origin main
run deny  "$MAIN" "$MAIN" git push
run deny  "$MAIN" "$MAIN" git push origin HEAD
run deny  "$FEAT" "$FEAT" git push -uf origin feat/x
run deny  "$FEAT" "$FEAT" git push -fu origin feat/x
run deny  "$FEAT" "$FEAT" git push -qd origin feat/old
run deny  "$FEAT" "$FEAT" git push origin :feat/old
run deny  "$FEAT" "$FEAT" git push origin +:feat/old
run allow "$FEAT" "$FEAT" git push
run allow "$FEAT" "$FEAT" git push -u origin feat/x
run allow "$FEAT" "$FEAT" git push -q origin feat/x
run allow "$FEAT" "$FEAT" git push origin main:backup-of-main
echo "--- G5: gh merge/review are human-only ---"
run deny  "$FEAT" "$FEAT" gh pr merge 30
run deny  "$FEAT" "$FEAT" gh pr merge 30 --squash
run deny  "$FEAT" "$FEAT" gh -R kurtis-b/torch-air pr merge 1
run deny  "$FEAT" "$FEAT" gh pr -R kurtis-b/torch-air merge 1
run deny  "$FEAT" "$FEAT" gh pr review 30 --approve
run deny  "$FEAT" "$FEAT" gh pr review 30 --comment -b hi
run deny  "$FEAT" "$FEAT" gh api -X PUT repos/x/y/pulls/5/merge
run deny  "$FEAT" "$FEAT" 'gh api "repos/x/y/pulls/5/reviews" -f event=APPROVE'
run deny  "$FEAT" "$FEAT" 'PR=5; gh api -X PUT repos/x/y/pulls/$PR/merge'
run deny  "$FEAT" "$FEAT" 'gh api graphql -f query="mutation { mergePullRequest(input: {pullRequestId: \"x\"}) { pullRequest { id } } }"'
run deny  "$FEAT" "$FEAT" 'gh api graphql -f query="mutation { addPullRequestReview(input: {}) { clientMutationId } }"'
run allow "$FEAT" "$FEAT" 'gh api graphql -f query="query { repository(owner: \"x\", name: \"y\") { pullRequest(number: 5) { title } } }"'
run allow "$FEAT" "$FEAT" gh pr comment 30 -b hi
run allow "$FEAT" "$FEAT" gh pr create --fill
run allow "$FEAT" "$FEAT" gh pr view 30
run allow "$FEAT" "$FEAT" gh api repos/x/y/pulls/5
echo "--- G6: main ref edits ---"
run deny  "$FEAT" "$FEAT" git branch -f main HEAD~1
run deny  "$FEAT" "$FEAT" git branch -D main
run deny  "$FEAT" "$FEAT" git branch -m main old-main
run deny  "$FEAT" "$FEAT" git update-ref refs/heads/main abc123
run allow "$FEAT" "$FEAT" git branch
run allow "$FEAT" "$FEAT" git branch -D feat/old
echo "--- scope: other repos are out of scope ---"
run allow "$FEAT" "$FEAT/submod" git commit -m x
run allow "$FEAT" "$FEAT" "cd submod && git checkout main"
run allow "$FEAT" "$FEAT" git -C submod push origin main
echo "--- prose in quoted args must not trip the scan ---"
run allow "$FEAT" "$FEAT" 'git commit -m "never run git push origin main or git merge here"'
run allow "$FEAT" "$FEAT" 'gh pr create --body "the guard denies gh pr merge and gh pr review"'
echo "--- benign lookalikes and reads ---"
run allow "$FEAT" "$FEAT" git log maincommit
run allow "$FEAT" "$FEAT" git commit-tree "HEAD^{tree}"
run allow "$MAIN" "$MAIN" git status
run allow "$MAIN" "$MAIN" git fetch origin
run allow "$FEAT" "$FEAT" git restore --source origin/main -- file
echo "--- malformed input degrades to ask ---"
out=$(printf 'not json' | CLAUDE_PROJECT_DIR="$FEAT" node "$HOOK" 2>/dev/null)
if grep -q '"permissionDecision":"ask"' <<<"$out"; then pass=$((pass+1)); echo "PASS  [ask] malformed JSON"
else fail=$((fail+1)); echo "FAIL  [expected ask] malformed JSON -> $out"; fi

echo; echo "== $pass passed, $fail failed =="
exit $((fail > 0))
