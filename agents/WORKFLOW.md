# Agent Workflow Standard — mlir-air (kurtis-b fork)

Adapted from the shared `agent-standards/WORKFLOW.md` (kurtis-b/agent-standards @ 84484c2, incl.
its PR #4 review amendments) via DAM-RS's landing-gate variant. One deliberate difference from the
shared standard: **pull requests are landed automatically by a script-enforced gate, not by a
human** (operator direction, 2026-08-29). Everything else — branch-only work, no force pushes, the
single Codex review per PR, integrity rules, layered enforcement — carries over. This file is the
single source for the rules below; `AGENTS.md` adds only repo-specific context and points here.

## Roles

- Claude Code is the only coding agent. It implements, tests, commits, opens PRs and lands them
  through `agents/scripts/pr.sh`.
- Codex is the reviewer: advisory, read-only. It never edits, approves or merges. Its verdict is
  one of the two inputs to the landing gate. (Documented exception: the phase-7 independent
  evaluator writes only its `evaluation_report.md` — see `.codex/skills/`.)
- The human sets direction, owns golden data (kernel_registry measurements, verified perf
  numbers), and is the escalation point when the gate refuses a PR twice. The human does not
  review routine PRs.

## Git workflow

1. NEVER work on `main` — never even check it out. Branch:
   `git fetch origin && git switch -c <type>/<slug> origin/main` (type = Conventional Commit
   type: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`). Never branch from another feature
   branch; no stacked PRs.
2. No force-push, no `git reset --hard`, no bulk push, no `git pull`, no deleting or pruning
   remote branches; `git branch <name>` only with `origin/main` as the start point; always push
   with an explicit destination (`git push -u origin <branch>`) — a bare `git push` is allowed
   only once it provably resolves to a non-main branch. `git merge` only as
   `git merge origin/main` to bring a PR branch up to date when it is behind or conflicted
   (resolve, then `git merge --continue`). Never rebase a pushed branch.
3. Stage by explicit path only — never `git add -A`, `git add .`, or `git add -u`: the tree may
   hold the human's uncommitted work.
4. Codex runs **exactly once per PR**, inside `pr.sh land`. Do not run `codex review` during
   implementation.
5. Done = `agents/scripts/pr.sh open --weakened "none|<declaration>"` (pushes the branch,
   creates the PR) and `agents/scripts/pr.sh land <N>` (the gate, which runs the PR's single
   review). One concern per PR; ≤ 500 added lines (`agents/scripts/check_pr_size.sh`; vendored
   files, submodule bumps, lockfiles and declared generated files exempt; total churn above the
   advisory threshold must be acknowledged in the review record).
6. NEVER run `gh pr merge`, `gh pr review`, or the merge/review REST endpoints directly. The only
   merge path is `pr.sh land`, which verifies the gate first. `gh pr comment` is fine for notes.
7. Dependent work waits: if the task depends on PR #N, run
   `agents/scripts/check_pr_deps.sh N [M ...]` first; on DEPS-BLOCKED, land the dependency
   first (or stop and name it to the human if it cannot be landed).
8. This fork is origin-only: `origin` (kurtis-b/mlir-air) is the only push target. `upstream`
   (Xilinx/mlir-air) is PULL-ONLY — never push to it or open PRs against it; upstream syncs into
   `main` are the operator's own action. Enforced in three local layers, none of which may be
   undone by an agent: the `upstream` remote's push URL is `no_push`; `gh repo set-default` is
   `kurtis-b/mlir-air` (gh otherwise prefers the `upstream` remote, so a bare `gh pr` would hit
   Xilinx); and `agents/hooks/origin-only-guard.mjs` + `permissions.deny` refuse upstream pushes,
   re-arming the remote, and `gh` writes addressed to Xilinx/ (reads stay allowed). The one
   GitHub-side hard block — *Leave fork network*, which makes a PR against upstream impossible —
   is irreversible and the operator's call.

## Landing gate (`pr.sh land`)

A PR lands when, at its current head `H` and the current `origin/main` `B`:

1. `H` contains `B` (a stale branch is synced with `git merge origin/main`);
2. CI is green at `H` (`gh pr checks`; the self-hosted Ryzen runners count only when they are
   required checks — offline hardware runners must not wedge a docs-only PR);
3. the PR body's `Weakened checks:` line is present (`none` or names each one) and its
   `Depends on:` PRs are all MERGED;
4. the PR's **single Codex review** is satisfied:
   - **no review yet** — `land` runs it now: a branch review of `B...H` with
     `agents/review-checklist.md` and the repository instruction files (`AGENTS.md`, `CLAUDE.md`,
     `.codex/`) as they stand on `B`, given the PR body's declarations. It is posted as a PR
     comment carrying `(H, B, declaration digest, verdict)` and labels the PR `codex-reviewed`.
     The review must carry the literal token `VERDICT: PASS` (no P0/P1) to pass; an error/empty
     run is retried once, then counts as a refusal. `VERDICT: PASS` → merge; `BLOCK` → refuse;
   - **review was PASS** — only the reviewed head lands; a later head needs an adjudication
     record naming it;
   - **review was BLOCK** — Claude adjudicates every finding (fix it, or reject it with a stated
     reason), pushes ordinary commits, and records the resolution with
     `pr.sh adjudicate <N> --text "..."`, binding the reviewed SHA to the new head. `land` then
     merges that head on CI green **without a second Codex review** — the budget is one review
     per PR, and the adjudication comment is the auditable record.

CI is bounded so a hang fails instead of stalling the gate (`timeout-minutes` in the fork-side
workflow; upstream workflows carry their own bounds). The gate still runs the PR's single review
when CI is red or timed out — the review binds to the reviewed SHA and its findings carry forward
through `pr.sh adjudicate`, so a red CI does not cost a review. Network calls retry with backoff
(`PR_NET_RETRIES`); the review worktree is removed however the review exits; a failure that
survives the retries stops the gate with nothing recorded, so re-running `land` is always safe.

`pr.sh land` re-checks that head and base are still `(H, B)` and merges exactly `H`
(`--match-head-commit`) with a merge commit (branch history kept; branches not deleted). If the
gate cannot proceed (unusable verdict, `needs-human` label), it stops and the human decides.

## Review protocol

- Findings are ranked P0–P3 (P0/P1 block) and falsifiable: `file:line — claim — concrete failing
  scenario — minimal suggested edit`. Style-only commentary is P3 at most. Review the diff, not
  unrelated files. Repo review-brief rules: `agents/codex-review-brief.md`.
- Every review also states, for the diff as a whole, either a smaller-diff or reuse alternative —
  naming the existing file/symbol and its semantic compatibility — or `no smaller alternative`
  with the search scope stated. A suggestion that exceeds the PR-size policy is a plan, marked as
  such, not a finding.
- One review per PR. Reviewer fixes are never auto-applied; Claude adjudicates finding by finding
  in the adjudication comment. The adjudication may include an in-scope P2 whose fix strictly
  reduces total changed lines while preserving intended behavior; every other P2/P3 is rejected
  with a stated reason or logged as an explicit follow-up, never silently dropped.
- A clean review or green CI never waives a measured-evidence precondition: no performance or
  correctness claim ships without an artifact (a log, a lit run, a devq job id), and hardware
  latency claims require Turbo pmode verification first.

## Integrity

- Never weaken, delete, skip, or loosen a test, assertion, tolerance, or timeout without naming
  each one, and why, in the PR body's `Weakened checks:` section (`none` if none). Undeclared
  weakening is a P1 finding.
- Golden references: `programming_examples/kernel_registry/` measured rows and recorded verify
  gates are edited only with a cited measurement; agents do not invent numbers.
- Device (NPU) measurement runs go through the device scheduler discipline (see AGENTS.md);
  latency claims without pmode verification are findings, not facts.

## Documentation

As succinct as possible; every fact lives in exactly one document, everywhere else links to it.
Before writing, find the fact's single home; extend or link, never restate.

## Enforcement

Three layers: these rules (advisory context) → `.claude/hooks/guard.sh` running
`agents/hooks/main-branch-guard.mjs` (vendored verbatim) then `agents/hooks/origin-only-guard.mjs`
(repo rule 8) as a PreToolUse(Bash) hook plus the `permissions.deny` list
in `.claude/settings.json` (agent-side; a denial reason contains the fix — follow it, never
bypass with `sh -c`, aliases, or wrappers) → a GitHub ruleset on `main` (PR required, required
checks on an up-to-date branch, no force-push/deletion; the gate account merges — no human-review
requirement). The hook is a guardrail, not a sandbox; the ruleset is the backstop and binds
everyone. Tests: `bash agents/hooks/test_guard.sh`, `bash agents/hooks/test_origin_only.sh`,
`bash agents/scripts/test_pr.sh`,
`bash agents/scripts/test_check_pr_deps.sh` — hermetic (stubbed `gh`, no network).
