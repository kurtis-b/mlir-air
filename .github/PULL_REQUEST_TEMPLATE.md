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

<!-- pre-PR review at <sha>: findings + resolutions (incl. the smaller-diff/reuse statement), or "no findings" -->

## Checklist

- [ ] Branch `<kind>/<short-slug>` off latest origin/main; no commits on main
- [ ] `agents/scripts/check_pr_size.sh` PASS (added ≤ 500; churn advisory acknowledged if fired)
- [ ] check-air-mlir lit subset green (build-xrt/mlir/test); device suite run if hardware behavior touched
- [ ] `Depends on` PRs merged; Codex findings fixed or rejected with reason; weakened checks declared
- [ ] Curt reviews and merges — the merge is the approval
