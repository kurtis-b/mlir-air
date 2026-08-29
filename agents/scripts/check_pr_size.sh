#!/usr/bin/env bash
# PR size gate (repo policy, LOOP-QUEUE 2b): <=500 ADDED lines vs the merge-base with
# origin/main, rename-aware. Churn (added+deleted) above the advisory threshold does not
# fail the gate but must be acknowledged in the review.
# Exemptions: submodule pointer bumps and lockfiles always; vendored/generated files only via
# exact-path `PR-Size-Exempt` commit trailers, which are printed here with their added-line counts
# and handed to the landing gate's review as declarations (authored code under one is P1 there).
set -euo pipefail
CAP=${PR_SIZE_CAP:-500}
CHURN_ADVISORY=${PR_CHURN_ADVISORY:-2000}
base=$(git merge-base HEAD origin/main)
# Declared exemptions: any commit in the range may carry `PR-Size-Exempt: <pathspec>...` trailers
# (vendored files, generated files). They are excluded from the count and PRINTED — a silent
# exemption is not a declaration — and the landing gate's review sees the trailer in the diff's
# commits.
exempt_args=()
while IFS= read -r spec; do
  [ -n "$spec" ] || continue
  case "$spec" in *'*'*|*'?'*|*'['*) echo "FAIL: PR-Size-Exempt '$spec' is a pattern; exemptions must be exact paths"; exit 1;; esac
  if ! git diff --name-only "$base"..HEAD -- "$spec" | grep -q .; then
    echo "FAIL: PR-Size-Exempt '$spec' matches nothing in the diff"; exit 1
  fi
  added=$(git diff -M --numstat "$base"..HEAD -- "$spec" | awk '$1!="-"{a+=$1} END{print a+0}')
  exempt_args+=(":(exclude)$spec")
  echo "EXEMPT (declared by trailer): $spec (+$added)"
done < <(git log --format='%(trailers:key=PR-Size-Exempt,valueonly)' "$base"..HEAD | tr ' ' '\n' | sed '/^$/d' | sort -u)
read -r ins del <<<"$(git diff -M --numstat "$base"..HEAD \
  -- ':(exclude).gitmodules' ':(exclude)*.lock' ${exempt_args[@]+"${exempt_args[@]}"} \
  | awk '$1!="-"{i+=$1; d+=$2} END{print i+0, d+0}')"
echo "added=$ins deleted=$del cap=$CAP base=$(git rev-parse --short "$base")"
if [ "$ins" -gt "$CAP" ]; then
  echo "FAIL: $ins added lines exceed the $CAP cap — split the PR or refactor first"
  exit 1
fi
churn=$((ins + del))
if [ "$churn" -gt "$CHURN_ADVISORY" ]; then
  echo "ADVISORY: total churn $churn > $CHURN_ADVISORY — the review must acknowledge it"
fi
echo "PASS"
