#!/usr/bin/env bash
# Open and land pull requests under agents/WORKFLOW.md. This is the ONLY path that merges.
#
#   agents/scripts/pr.sh open  --weakened "none|<what and why>" [--title "<title>"] [--deps "N M"]
#       Push the current branch and create the PR from .github/PULL_REQUEST_TEMPLATE.md.
#       --weakened is mandatory: the declaration of weakened/deleted/skipped checks.
#
#   agents/scripts/pr.sh land  <N>
#       The landing gate: dependencies MERGED, "Weakened checks:" declared, branch contains
#       origin/main, CI green at the PR head, and ONE Codex branch review per PR:
#         - no review yet   -> run it now (posted as a PR comment); PASS -> merge, BLOCK -> exit 1
#         - review was PASS -> merge only the reviewed head
#         - review was BLOCK -> merge the head named by `pr.sh adjudicate` (fixes pushed, CI green)
#       Gate passes -> merge commit (branch kept).
#
#   agents/scripts/pr.sh adjudicate <N> --text "<finding-by-finding resolution>"
#       After fixing a BLOCK: record how each finding was resolved (or rejected, with reason),
#       bound to the current head. `land` then merges that head without a second review.
#
#   agents/scripts/pr.sh status <N>
#       Show the gate inputs without acting.
#
# Environment: gh (authenticated), codex (authenticated). Every gh call is pinned to the repository
# behind git remote `origin` (GH_REPO), whatever other remotes the clone carries — gh would otherwise
# prefer a remote named `upstream`, which here is the pull-only Xilinx/mlir-air.
# CI: the base ruleset's REQUIRED checks decide, and one that has not reported yet is pending.
# While the base has no ruleset, every reported check and every workflow run at the head decide,
# except names matching PR_CI_OPTIONAL (default: the self-hosted Ryzen runners, which never start
# on this fork) — those never count, pending or failed.
set -euo pipefail

ROOT_DIR="${PR_SH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
STATE_DIR="${ROOT_DIR}/agents/.state"
DEPS="${PR_SH_DEPS:-${ROOT_DIR}/agents/scripts/check_pr_deps.sh}"
BASE="main"
CI_TIMEOUT_MIN="${PR_CI_TIMEOUT_MIN:-30}"
CI_OPTIONAL_RE="${PR_CI_OPTIONAL:-Ryzen AI}"
mkdir -p "${STATE_DIR}"
cd "${ROOT_DIR}"

die() { echo "pr.sh: $*" >&2; exit 2; }
note() { echo "pr.sh: $*" >&2; }
need() { command -v "$1" >/dev/null 2>&1 || die "$1 not found on PATH"; }
need gh; need git; need codex; need jq; need python3

# owner/repo behind `origin`, for https, ssh and scp-style URLs. Exported as GH_REPO so gh never
# resolves a PR number against another remote (a bare `gh pr view 3` here once meant Xilinx#3).
origin_repo() { git remote get-url origin | sed -E 's#^(https?://[^/]+/|ssh://[^/]+/|[^@]+@[^:]+:)##; s#\.git/?$##'; }
GH_REPO="$(origin_repo)"; export GH_REPO
[[ "$GH_REPO" == */* ]] || die "cannot derive owner/repo from origin ($(git remote get-url origin 2>/dev/null))"
repo() { gh repo view --json nameWithOwner --jq .nameWithOwner; }

# Retry a command that talks to the network. Transient DNS/TLS/HTTP failures are common enough
# that letting one abort a gate mid-run (under `set -e`) loses the CI wait and can strand a
# review worktree. Retries are quiet and back off; the final attempt's failure propagates.
RETRIES="${PR_NET_RETRIES:-3}"
RETRY_SLEEP="${PR_NET_RETRY_SLEEP:-5}"
retry() { # retry <command> [args...]
  local attempt=1
  while :; do
    if "$@"; then return 0; fi
    if [ "$attempt" -ge "$RETRIES" ]; then
      note "'$*' failed after ${attempt} attempts"
      return 1
    fi
    note "'$*' failed (attempt ${attempt}/${RETRIES}); retrying in $((RETRY_SLEEP * attempt))s"
    sleep "$((RETRY_SLEEP * attempt))"
    attempt=$((attempt + 1))
  done
}
git_fetch() { retry git fetch -q origin "$@"; }
gh_json() { retry gh "$@"; }

# The review runs in a throwaway worktree. Cleanup must happen however the script leaves —
# normal return, `set -e`, or a signal — and must be safe to call twice, so it hangs off a
# single EXIT trap keyed on a global rather than a RETURN trap that outlives its function.
REVIEW_WORKTREE=""
cleanup_review_worktree() {
  [ -n "${REVIEW_WORKTREE:-}" ] || return 0
  local wt="$REVIEW_WORKTREE"
  REVIEW_WORKTREE=""
  git worktree remove -f "$wt" >/dev/null 2>&1 || true
  git worktree prune >/dev/null 2>&1 || true
}
trap cleanup_review_worktree EXIT

ensure_labels() {
  for l in "codex-reviewed:1d76db:The single Codex review for this PR has run" "codex-blocked:e99695:The Codex review blocked; fixes need an adjudication record" "needs-human:b60205:Landing gate cannot proceed without a human" "landed-by-gate:0e8a16:Merged by agents/scripts/pr.sh"; do
    IFS=: read -r name color desc <<<"$l"
    gh label create "$name" --color "$color" --description "$desc" --force >/dev/null 2>&1 || true
  done
}

# Verdict from a review run. Fails closed:
#   ERROR  non-zero codex exit, empty output, or no explicit verdict token at all;
#   BLOCK  any P0/P1/BLOCKING marker, or the token `VERDICT: BLOCK`;
#   PASS   only a completed run (exit 0) carrying the token `VERDICT: PASS` and no blocker.
# "Looks fine" prose without the token is ERROR and gets re-run, never merged.
verdict_of() {
  local text="$1" status="${2:-0}"
  if [ -z "$(printf '%s' "$text" | tr -d '[:space:]')" ]; then echo "ERROR"; return; fi
  if [ "$status" != "0" ]; then echo "ERROR"; return; fi
  if printf '%s' "$text" | grep -qE '\[(P0|P1|BLOCKING)\]'; then echo "BLOCK"; return; fi
  if printf '%s' "$text" | grep -qiE '\bVERDICT:\s*BLOCK\b'; then echo "BLOCK"; return; fi
  if printf '%s' "$text" | grep -qiE '\bVERDICT:\s*PASS\b'; then echo "PASS"; return; fi
  echo "ERROR"
}

# Run the Codex branch review of origin/$BASE...HEAD inside a detached worktree at $sha, with the
# review policy taken from origin/$BASE (falls back to the reviewed tree only while origin/$BASE
# has no checklist yet, i.e. the bootstrap PR). Sets REVIEW_TEXT and REVIEW_STATUS.
REVIEW_TEXT=""; REVIEW_STATUS=0
run_branch_review_at() { # run_branch_review_at <sha> [<weakened-declaration>] [<depends-on>]
  local sha="$1" weakened="${2:-}" depends="${3:-}"
  git_fetch "$BASE"
  local policy
  if ! policy="$(git show "origin/${BASE}:agents/review-checklist.md" 2>/dev/null)"; then
    if policy="$(git show "${sha}:agents/review-checklist.md" 2>/dev/null)"; then
      note "BOOTSTRAP: origin/${BASE} has no review checklist yet; using the reviewed tree's copy"
    else
      policy="$(cat "${ROOT_DIR}/agents/review-checklist.md")"
      note "BOOTSTRAP: neither origin/${BASE} nor ${sha:0:10} has a review checklist; using the working tree's copy"
    fi
  fi
  local worktree; worktree="$(mktemp -d "${STATE_DIR}/land-XXXXXX")"
  git worktree add -q --detach "$worktree" "$sha"
  REVIEW_WORKTREE="$worktree"
  materialize_base_instructions "$worktree" "origin/${BASE}"
  local scope="The change under review is the current branch relative to origin/${BASE}: \`git diff origin/${BASE}...HEAD\` (commits: \`git log --oneline origin/${BASE}..HEAD\`)."
  local exempt; exempt="$(git log --format='%(trailers:key=PR-Size-Exempt,valueonly)' "origin/${BASE}..${sha}" | tr ' ' '\n' | sed '/^$/d' | sort -u \
    | while read -r p; do printf '%s (+%s) ' "$p" "$(git diff -M --numstat "origin/${BASE}...${sha}" -- "$p" | awk '$1!="-"{a+=$1} END{print a+0}')"; done)"
  local decl="PR body declarations to check against the diff:
- Weakened checks: ${weakened:-<missing>}  (every weakened, deleted, skipped or loosened test, assertion, tolerance or timeout in the diff must be named here; an undeclared one is P1; 'none' must be true)
- Depends on: ${depends:-<missing>}
- Size exemptions (PR-Size-Exempt commit trailers, excluded from the 500-added-line count): ${exempt:-none}  (each must be a vendored/generated file, submodule bump or lockfile; authored code under an exemption is P1)"
  set +e
  REVIEW_TEXT="$( cd "$worktree" && { printf '%s\n\n%s\n\n%s\n' "$scope" "$decl" "$policy"; } | codex review - 2>/dev/null )"
  REVIEW_STATUS=$?
  set -e
  cleanup_review_worktree
  printf '%s\n' "$REVIEW_TEXT" > "${STATE_DIR}/review-${sha:0:10}.md"
}

# Short digest of the PR-body declarations a review was run with; a body edit invalidates reuse.
decl_digest() { printf '%s\x1f%s' "${1:-}" "${2:-}" | sha1sum | cut -c1-12; }

# The review record carries the head SHA, the origin/$BASE SHA and the declaration digest it was
# reviewed against.
post_review_comment() { # post_review_comment <N> <sha> <base_sha> <decl_digest> <verdict> <text>
  local n="$1" sha="$2" base_sha="$3" digest="$4" verdict="$5" text="$6"
  local body
  body="$(printf '<!-- codex-review sha=%s base=%s decl=%s verdict=%s -->\n## Codex branch review at `%s` (base `%s`) — verdict **%s**\n\n%s\n' "$sha" "$base_sha" "$digest" "$verdict" "${sha:0:10}" "${base_sha:0:10}" "$verdict" "${text:-"(no output)"}" | head -c 60000)"
  # If the verdict cannot be recorded, the review effectively did not happen: stop rather than
  # merge on an unrecorded review or silently spend the budget again on the next run.
  retry gh pr comment "$n" --body "$body" >/dev/null ||
    die "Codex returned ${verdict} for #${n} at ${sha:0:10} but the record could not be posted; re-run land"
}

gate_login() { gh api user --jq .login; }
base_sha() { git_fetch "$BASE"; git rev-parse "origin/${BASE}"; }

# `land`/`adjudicate` run from origin/$BASE's own copy of pr.sh and check_pr_deps.sh, never the
# PR's: the ruleset has no review requirement, so the gate code IS the review requirement, and a
# PR that edits cmd_land must not be able to land itself through the edit. Bootstrap: while
# origin/$BASE has no pr.sh — the adoption PR itself — the working copy runs and says so.
trusted_exec() { # trusted_exec <argv...>
  [ -z "${PR_SH_TRUSTED:-}" ] || return 0
  git_fetch "$BASE"
  if ! git cat-file -e "origin/${BASE}:agents/scripts/pr.sh" 2>/dev/null; then
    note "BOOTSTRAP: origin/${BASE} has no agents/scripts/pr.sh yet; running the working copy"
    export PR_SH_TRUSTED=bootstrap; return 0
  fi
  local dir; dir="${STATE_DIR}/trusted-$(git rev-parse --short "origin/${BASE}")"
  mkdir -p "$dir"
  git show "origin/${BASE}:agents/scripts/pr.sh" > "$dir/pr.sh"
  if ! git show "origin/${BASE}:agents/scripts/check_pr_deps.sh" > "$dir/check_pr_deps.sh" 2>/dev/null; then cp "$DEPS" "$dir/check_pr_deps.sh"; fi
  chmod +x "$dir/pr.sh" "$dir/check_pr_deps.sh"
  cmp -s "$dir/pr.sh" "${BASH_SOURCE[0]}" || note "gate code differs from origin/${BASE}; running the trusted copy"
  PR_SH_TRUSTED=1 PR_SH_ROOT="$ROOT_DIR" PR_SH_DEPS="$dir/check_pr_deps.sh" exec bash "$dir/pr.sh" "$@"
}

# Codex loads AGENTS.md / AGENTS.override.md (root and nested), CLAUDE.md and .codex/ from the
# tree it runs in. In a
# review worktree those must be the *base's* versions, or a PR could weaken the instructions it
# is reviewed under (its edits to them are still visible in the diff). For the union of matching
# paths on base and head: remove the head's path without following symlinks, then restore the
# base's copy through git (which refuses to write through a symlinked directory).
instruction_paths() { # instruction_paths <worktree> <base-ref> -> newline-separated union
  local wt="$1" base="$2"
  {
    git -C "$wt" ls-files -- 'AGENTS.md' '*/AGENTS.md' 'AGENTS.override.md' '*/AGENTS.override.md' 'CLAUDE.md' '*/CLAUDE.md' '.codex' '.codex/**' 2>/dev/null || true
    git -C "$wt" ls-tree -r --name-only "$base" 2>/dev/null | grep -E '(^|/)(AGENTS|AGENTS\.override|CLAUDE)\.md$|^\.codex(/|$)' || true
  } | sort -u
}
# Replace every symlinked component of <worktree>/<path> (including the leaf) with nothing, so a
# later rm/checkout cannot follow a head-controlled link out of the worktree.
unlink_symlinked_components() { # unlink_symlinked_components <worktree> <path>
  local wt="$1" rel="$2" cur="$wt" part
  local IFS=/
  for part in $rel; do
    [ -n "$part" ] || continue
    cur="${cur}/${part}"
    if [ -L "$cur" ]; then
      rm -f -- "$cur"
      return 0                                   # nothing below a removed link exists any more
    fi
  done
  return 0
}
materialize_base_instructions() { # materialize_base_instructions <worktree> <base-ref>
  local wt="$1" base="$2" f
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    unlink_symlinked_components "$wt" "$f"
    rm -rf -- "${wt:?}/${f}"
    if git -C "$wt" cat-file -e "${base}:${f}" 2>/dev/null; then
      git -C "$wt" checkout -q "$base" -- "$f"     # git refuses to write through a symlinked dir
    fi
  done < <(instruction_paths "$wt" "$base")
  return 0
}

# Verdict recorded on PR <N> by GitHub user <me> for the exact marker prefix (head+base), or "".
# `gh pr view --jq` cannot take jq variables, so the JSON goes through standalone jq.
# Comment bodies on PR <N> written by <login>. Non-zero if GitHub could not be read, so the
# caller fails closed instead of mistaking an unreadable PR for one with no records — that
# distinction is what protects the one-review-per-PR budget.
gate_comments() { # gate_comments <N> <login>
  local n="$1" me="$2" json
  json="$(retry gh pr view "$n" --json comments)" || return 1
  # Only a gate comment's FIRST line is a record: the review text appended after the header is
  # untrusted Codex output and may quote marker-shaped lines for the same SHA.
  printf '%s' "$json" | jq -r --arg me "$me" '.comments[] | select(.author.login == $me) | (.body | split("\n")[0])' 2>/dev/null || return 1
}

# The latest record matching <regex> in <login>'s comments on PR <N>, or "" when there is none.
# Non-zero only when the comments could not be read.
latest_record() { # latest_record <N> <login> <regex>
  local n="$1" me="$2" re="$3" bodies
  bodies="$(gate_comments "$n" "$me")" || return 1
  printf '%s\n' "$bodies" | grep -oE "$re" | tail -1 || true
  return 0
}

# The latest review record on PR <N> by <login>: "sha base decl verdict" or "".
latest_review_record() { # latest_review_record <N> <login>
  local line
  line="$(latest_record "$1" "$2" '<!-- codex-review sha=[0-9a-f]+ base=[0-9a-f]+ decl=[0-9a-f]+ verdict=[A-Z]+ -->')" || return 1
  [ -n "$line" ] || return 0
  printf '%s\n' "$line" | sed -nE 's/.*sha=([0-9a-f]+) base=([0-9a-f]+) decl=([0-9a-f]+) verdict=([A-Z]+).*/\1 \2 \3 \4/p'
  return 0
}

# The latest adjudication record on PR <N> by <login>: "reviewed_sha fixed_sha" or "".
latest_adjudication() { # latest_adjudication <N> <login>
  local line
  line="$(latest_record "$1" "$2" '<!-- codex-adjudication reviewed=[0-9a-f]+ fixed=[0-9a-f]+ decl=[0-9a-f]+ -->')" || return 1
  [ -n "$line" ] || return 0
  printf '%s\n' "$line" | sed -nE 's/.*reviewed=([0-9a-f]+) fixed=([0-9a-f]+) decl=([0-9a-f]+).*/\1 \2 \3/p'
  return 0
}

# Always exits 0 (callers assign its output under `set -e -o pipefail`); empty if absent.
body_field() { # body_field "<body>" "Depends on"  -> value after "Depends on:" (first match)
  local line
  line="$(printf '%s\n' "$1" | grep -iE "^\s*$2:" | head -1 || true)"
  [ -n "$line" ] || return 0
  printf '%s\n' "$line" | sed -E "s/^\s*$2:\s*//I" | tr -d '\r' || true
  return 0
}

cmd_open() {
  local title="" deps="" weakened=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --title) title="$2"; shift 2 ;;
      --deps) deps="$2"; shift 2 ;;
      --weakened) weakened="$2"; shift 2 ;;
      *) die "unknown option $1" ;;
    esac
  done
  [ -n "$weakened" ] || die "--weakened is required: 'none', or name each weakened/deleted/skipped test, assertion, tolerance or timeout and why"
  local branch; branch="$(git rev-parse --abbrev-ref HEAD)"
  [ "$branch" != "$BASE" ] || die "refusing to open a PR from ${BASE}"
  git_fetch "$BASE"
  [ -n "$(git log --oneline "origin/${BASE}..HEAD")" ] || die "no commits relative to origin/${BASE}"
  if [ -n "$deps" ]; then
    # shellcheck disable=SC2086
    "$DEPS" $deps
  fi
  local off_master="[ ]"
  if git merge-base --is-ancestor "origin/${BASE}" HEAD; then off_master="[x]"; fi
  note "pushing ${branch}"
  retry git push -q -u origin "$branch"

  local sha; sha="$(git rev-parse HEAD)"
  [ -n "$title" ] || title="$(git log -1 --format=%s)"
  local summary; summary="$(git log --reverse --format='- %s' "origin/${BASE}..HEAD")"
  local deps_line="none"; [ -n "$deps" ] && deps_line="$(printf '#%s ' $deps | sed 's/ $//')"

  # Only claims the script verified are ticked; the rest stay for the gate / the author.
  local body
  body="$(cat <<BODY
## Summary

${summary}

## Depends on

Depends on: ${deps_line}

## Weakened checks

Weakened checks: ${weakened}

## Evidence

none

## Codex review

One review per PR, run and posted by \`agents/scripts/pr.sh land\`; fixes after a BLOCK are
recorded by \`pr.sh adjudicate\`.

## Checklist

- ${off_master} Branch \`${branch}\` contains \`origin/${BASE}\` (verified by pr.sh open); no commits on ${BASE}
- [ ] \`agents/scripts/check_pr_size.sh\` PASS locally; churn advisory acknowledged if fired
- [ ] Depends on: ${deps_line} (verified MERGED by pr.sh open when listed); weakened checks: ${weakened}
- [ ] Compiler/test files touched -> check-air-mlir lit subset run; hardware behavior touched -> device gates via devq, with evidence links
- [ ] Landed by \`agents/scripts/pr.sh land\` (CI green + Codex branch review without P0/P1)
BODY
)"
  local url; url="$(gh pr create --base "$BASE" --head "$branch" --title "$title" --body "$body")"
  echo "$url"
  local n; n="$(grep -oE '[0-9]+$' <<<"$url")"
  echo "$n" > "${STATE_DIR}/last-pr"

  echo "OPENED #$n (the single Codex review runs at \`pr.sh land $n\`)"
}

# Waits for CI at the PR head. Prints PASS / FAIL / NONE.
# The checks that decide CI at a PR head, as a JSON list. With a ruleset on the base, the
# required set decides — and a required check that has not REPORTED yet is pending: a matrix
# job is registered only when its workflow reaches it, so "everything reported passes" is not
# "CI passed" (PR #3's first run read PASS with 8 of 28 checks registered). Without a ruleset,
# every reported check decides except PR_CI_OPTIONAL matches (which never count, pending or
# failed — WORKFLOW.md: non-required hardware runners do not count at all), and every
# non-optional workflow RUN at the head must have completed.
ci_checks_json() { # ci_checks_json <N> <sha>
  local n="$1" sha="$2" json required
  required="$(gh api "repos/${GH_REPO}/rules/branches/${BASE}" --jq '[.[] | select(.type=="required_status_checks") | .parameters.required_status_checks[].context]' 2>/dev/null || echo '[]')"
  printf '%s' "$required" | python3 -c 'import json,sys; sys.exit(0 if json.load(sys.stdin) else 1)' 2>/dev/null || required='[]'
  if [ "$required" != '[]' ]; then
    set +e
    json="$(gh pr checks "$n" --required --json name,state,bucket 2>/dev/null)"
    set -e
    printf '%s' "$json" | REQUIRED="$required" python3 -c '
import json, os, sys
try: checks = json.load(sys.stdin)
except Exception: checks = []
names = {c.get("name") for c in checks}
missing = [r for r in json.loads(os.environ["REQUIRED"]) if r not in names]
print(json.dumps(checks + [{"name": m, "bucket": "pending", "state": "UNREPORTED"} for m in missing]))'
    return
  fi
  set +e
  # gh pr checks exits 8 while checks are pending (and still prints JSON); never append to it.
  json="$(gh pr checks "$n" --json name,state,bucket 2>/dev/null)"
  set -e
  local runs; runs="$(gh run list --commit "$sha" --json status,workflowName --jq '[.[] | select(.status != "completed") | .workflowName]' 2>/dev/null || echo '[]')"
  printf '%s' "$json" | CI_OPTIONAL_RE="$CI_OPTIONAL_RE" RUNS="$runs" python3 -c '
import json, os, re, sys
try: checks = json.load(sys.stdin)
except Exception: checks = []
opt = re.compile(os.environ["CI_OPTIONAL_RE"])
checks = [c for c in checks if not opt.search(c.get("name", ""))]
try: runs = json.loads(os.environ["RUNS"])
except Exception: runs = []
checks += [{"name": "run:" + w, "bucket": "pending", "state": "IN_PROGRESS"} for w in runs if not opt.search(w)]
print(json.dumps(checks))'
}

# One poll's verdict: NONE (nothing reported yet), FAIL, PENDING or PASS.
ci_decide() { # ci_decide <checks-json>
  printf '%s' "$1" | python3 -c '
import json, sys
try: checks = json.load(sys.stdin)
except Exception: checks = []
if not checks: print("NONE")
elif any(c.get("bucket") in ("fail", "cancel") for c in checks): print("FAIL")
elif any(c.get("bucket") == "pending" for c in checks): print("PENDING")
else: print("PASS")'
}

ci_state() { # ci_state <N> <sha>
  local n="$1" sha="$2" deadline=$(( $(date +%s) + CI_TIMEOUT_MIN * 60 )) verdict
  while :; do
    verdict="$(ci_decide "$(ci_checks_json "$n" "$sha")")"
    case "$verdict" in
      PASS|FAIL) echo "$verdict"; return ;;
      NONE)      if [ "$(date +%s)" -gt "$deadline" ]; then echo "NONE"; return; fi ;;
      *)         if [ "$(date +%s)" -gt "$deadline" ]; then echo "TIMEOUT"; return; fi ;;
    esac
    sleep 20
  done
}

cmd_status() {
  local n="$1"
  gh pr view "$n" --json number,title,state,headRefName,headRefOid,labels,mergeable --jq '{number,title,state,head:.headRefName,sha:.headRefOid,mergeable,labels:[.labels[].name]}'
  gh pr checks "$n" 2>/dev/null || true
}

cmd_land() {
  local n="$1"
  ensure_labels
  local meta; meta="$(gh_json pr view "$n" --json state,headRefName,headRefOid,body,labels,isDraft,baseRefName)"
  local state head sha body labels draft base
  state="$(jq -r .state <<<"$meta")"; head="$(jq -r .headRefName <<<"$meta")"; sha="$(jq -r .headRefOid <<<"$meta")"
  body="$(jq -r .body <<<"$meta")"; labels="$(jq -r '[.labels[].name]|join(",")' <<<"$meta")"
  draft="$(jq -r .isDraft <<<"$meta")"; base="$(jq -r .baseRefName <<<"$meta")"
  [ "$state" = "OPEN" ] || die "PR #$n is $state"
  [ "$draft" = "false" ] || die "PR #$n is a draft"
  [ "$base" = "$BASE" ] || die "PR #$n targets $base, not $BASE"
  if grep -q 'needs-human' <<<"$labels"; then die "PR #$n is labelled needs-human; a human decides"; fi
  note "landing #$n (${head} @ ${sha:0:10})"

  local refusals=()

  # 0. The head must contain the current origin/$BASE: a PR reviewed and CI-tested against an
  #    older base is not evidence about the merge that would actually happen.
  local base; base="$(base_sha)"
  git_fetch "$head"
  if ! git merge-base --is-ancestor "$base" "$sha"; then
    refusals+=("branch is behind origin/${BASE} (${base:0:10}): run \`git merge origin/${BASE}\` on ${head}, push, and land again")
  fi

  # 1. dependencies
  local deps; deps="$(body_field "$body" "Depends on")"
  if [ -z "$deps" ]; then refusals+=("PR body has no 'Depends on:' line");
  elif [ "$deps" != "none" ]; then
    local nums; nums="$(grep -oE '#[0-9]+' <<<"$deps" | tr -d '#' | tr '\n' ' ')"
    # shellcheck disable=SC2086
    if ! "$DEPS" $nums >/dev/null 2>&1; then refusals+=("dependency PR(s) not MERGED: ${deps}"); fi
  fi

  # 2. weakened checks declared
  local weak; weak="$(body_field "$body" "Weakened checks")"
  [ -n "$weak" ] || refusals+=("PR body has no 'Weakened checks:' line (write 'none' or name each)")

  # 3. CI at head
  note "waiting for CI (timeout ${CI_TIMEOUT_MIN} min)"
  local ci; ci="$(ci_state "$n" "$sha")"
  note "CI: ${ci}"
  [ "$ci" = "PASS" ] || refusals+=("CI at ${sha:0:10} is ${ci}")

  # 4. The single Codex review per PR.
  #    - none yet: run it at this head and post it. PASS -> proceed; BLOCK -> refuse (exit 1).
  #    - PASS recorded: only the reviewed head may land (a new head needs... nothing more: the
  #      budget is spent, so a head that differs from the reviewed one needs an adjudication).
  #    - BLOCK recorded: the current head must be the one named by `pr.sh adjudicate`.
  local digest; digest="$(decl_digest "$weak" "$deps")"
  local me; me="$(gate_login)"
  local record; record="$(latest_review_record "$n" "$me")"
  local r_sha r_base r_decl r_verdict
  if [ -z "$record" ]; then
    local attempt verdict=""
    for attempt in 1 2; do
      note "running the PR's single Codex branch review at ${sha:0:10} against ${base:0:10} (attempt ${attempt})"
      run_branch_review_at "$sha" "$weak" "$deps"
      verdict="$(verdict_of "$REVIEW_TEXT" "$REVIEW_STATUS")"
      note "Codex verdict: ${verdict}"
      post_review_comment "$n" "$sha" "$base" "$digest" "$verdict" "$REVIEW_TEXT"
      [ "$verdict" = "ERROR" ] || break
    done
    gh pr edit "$n" --add-label codex-reviewed >/dev/null 2>&1 || true
    if [ "$verdict" = "BLOCK" ]; then
      gh pr edit "$n" --add-label codex-blocked >/dev/null 2>&1 || true
      refusals+=("Codex branch review at ${sha:0:10}: BLOCK — fix or reject each finding, push, then \`pr.sh adjudicate $n --text ...\` and \`pr.sh land $n\`")
    elif [ "$verdict" != "PASS" ]; then
      refusals+=("Codex branch review at ${sha:0:10}: ${verdict} (no usable verdict after 2 attempts)")
    fi
  else
    read -r r_sha r_base r_decl r_verdict <<<"$record"
    note "review budget already spent: ${r_verdict} at ${r_sha:0:10}"
    case "$r_verdict" in
      PASS)
        if [ "$r_sha" = "$sha" ]; then
          # Same head, but the body's declarations may have been edited since the review.
          [ "$r_decl" = "$digest" ] || refusals+=("PR body declarations changed since the review (digest ${r_decl} -> ${digest}): restore them, or push a commit and record the change with \`pr.sh adjudicate $n --text ...\`")
        else
          local adj; adj="$(latest_adjudication "$n" "$me")"
          local a_rev a_fixed a_decl; read -r a_rev a_fixed a_decl <<<"${adj:-x x x}"
          if [ "$a_rev" != "$r_sha" ] || [ "$a_fixed" != "$sha" ] || [ "$a_decl" != "$digest" ]; then
            refusals+=("head ${sha:0:10} or its declarations differ from the reviewed state (review at ${r_sha:0:10}) and no adjudication binds both: run \`pr.sh adjudicate $n --text ...\` at this head")
          fi
        fi
        ;;
      BLOCK)
        local adj; adj="$(latest_adjudication "$n" "$me")"
        local a_rev a_fixed a_decl; read -r a_rev a_fixed a_decl <<<"${adj:-x x x}"
        if [ "$a_rev" != "$r_sha" ] || [ "$a_fixed" != "$sha" ] || [ "$a_decl" != "$digest" ]; then
          refusals+=("review BLOCKed at ${r_sha:0:10}; landing needs fixes pushed and \`pr.sh adjudicate $n --text ...\` recorded at the current head ${sha:0:10} with the current declarations")
        fi
        ;;
      *)
        refusals+=("recorded review verdict ${r_verdict} is not usable; a human must decide")
        gh pr edit "$n" --add-label needs-human >/dev/null 2>&1 || true
        ;;
    esac
  fi

  # Decide
  if [ "${#refusals[@]}" -eq 0 ]; then
    # Re-read head and base: everything above was checked at ($sha, $base), so only that pair
    # may be merged.
    local head_now; head_now="$(gh_json pr view "$n" --json headRefOid --jq .headRefOid)"
    if [ "$head_now" != "$sha" ]; then
      echo "pr.sh: head moved from ${sha:0:10} to ${head_now:0:10} during the gate; run land again" >&2
      return 1
    fi
    local base_now; base_now="$(base_sha)"
    if [ "$base_now" != "$base" ]; then
      echo "pr.sh: origin/${BASE} moved from ${base:0:10} to ${base_now:0:10} during the gate; sync and land again" >&2
      return 1
    fi
    note "gate passed; merging #$n at ${sha:0:10} with a merge commit"
    # The merge itself is deliberately not retried: a retry after a request that actually
    # merged would report a failure for a landed PR. Re-running `land` is the safe recovery —
    # it sees the PR as MERGED and stops.
    gh pr merge "$n" --merge --match-head-commit "$sha" --subject "Merge #${n}: $(gh_json pr view "$n" --json title --jq .title)" >/dev/null
    gh pr edit "$n" --add-label landed-by-gate >/dev/null 2>&1 || true
    git fetch -q origin "$BASE"
    echo "LANDED #$n"
    return 0
  fi

  local msg; msg="$(printf 'Landing gate refused:\n'; printf -- '- %s\n' "${refusals[@]}")"
  gh pr comment "$n" --body "$msg" >/dev/null
  echo "$msg"; echo "REFUSED #$n"; return 1
}

cmd_adjudicate() {
  local n="$1"; shift
  local text=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --text) text="$2"; shift 2 ;;
      *) die "unknown option $1" ;;
    esac
  done
  [ -n "$text" ] || die "--text is required: how each finding was fixed, or rejected with a reason"
  local me; me="$(gate_login)"
  local record; record="$(latest_review_record "$n" "$me")"
  [ -n "$record" ] || die "PR #$n has no Codex review record yet; run \`pr.sh land $n\` first"
  local r_sha r_base r_decl r_verdict; read -r r_sha r_base r_decl r_verdict <<<"$record"
  local sha; sha="$(gh_json pr view "$n" --json headRefOid --jq .headRefOid)"
  [ "$sha" != "$r_sha" ] || die "head ${sha:0:10} is the reviewed head; push the fixes first"
  local body; body="$(gh_json pr view "$n" --json body --jq .body)"
  local weak deps; weak="$(body_field "$body" "Weakened checks")"; deps="$(body_field "$body" "Depends on")"
  [ -n "$weak" ] || die "PR body has no 'Weakened checks:' line"
  local digest; digest="$(decl_digest "$weak" "$deps")"
  [ "$digest" = "$r_decl" ] || note "declarations changed since the review (digest ${r_decl} -> ${digest}); this adjudication binds the current ones"
  retry gh pr comment "$n" --body "$(printf '<!-- codex-adjudication reviewed=%s fixed=%s decl=%s -->\n## Adjudication of the Codex review at `%s`, resolved at `%s`\n\n%s\n' "$r_sha" "$sha" "$digest" "${r_sha:0:10}" "${sha:0:10}" "$text")" >/dev/null ||
    die "could not post the adjudication for #${n}"
  echo "ADJUDICATED #$n: review ${r_sha:0:10} -> fixes at ${sha:0:10}"
}

# Only dispatch when executed, so tests can `source` the functions.
[[ "${BASH_SOURCE[0]}" == "$0" ]] || return 0 2>/dev/null || true
case "${1:-}" in
  open) shift; cmd_open "$@" ;;
  land) [ $# -ge 2 ] || die "usage: pr.sh land <N>"; trusted_exec "$@"; cmd_land "$2" ;;
  adjudicate) [ $# -ge 2 ] || die "usage: pr.sh adjudicate <N> --text ..."; trusted_exec "$@"; n="$2"; shift 2; cmd_adjudicate "$n" "$@" ;;
  status) [ $# -ge 2 ] || die "usage: pr.sh status <N>"; cmd_status "$2" ;;
  -h|--help|"") sed -n '2,24p' "$0" ;;
  *) die "unknown command $1" ;;
esac
