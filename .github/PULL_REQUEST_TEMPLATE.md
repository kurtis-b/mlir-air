## Summary

## Depends on

Depends on: none
<!-- or "Depends on: #N" — listed PRs must be MERGED before this one's work started -->

## Weakened checks

Weakened checks: none
<!-- or name each weakened/deleted/skipped test, assertion, tolerance, or timeout — and why -->

## Evidence

<!-- gate runs with links/paths (lit subset, device suite, verify) — a link or artifact, never prose; or "none" -->

## Codex review

<!-- Left blank at open. The PR's single Codex review is run and posted by
     `agents/scripts/pr.sh land`; on BLOCK, record the finding-by-finding resolution
     with `pr.sh adjudicate <N>`. Never run `codex review` yourself (WORKFLOW.md rule 4). -->

## Checklist

- [ ] Branch `<kind>/<short-slug>` off latest origin/main; no commits on main
- [ ] `agents/scripts/check_pr_size.sh` PASS (added ≤ 500; churn advisory acknowledged if fired)
- [ ] check-air-mlir lit subset green (build-xrt/mlir/test); device suite run if hardware behavior touched
- [ ] `Depends on` PRs merged; weakened checks declared
- [ ] Landed by `agents/scripts/pr.sh land` (CI green + the gate's single Codex review with no P0/P1);
      on BLOCK, every finding fixed or rejected with a reason and recorded via `pr.sh adjudicate`
