# Codex review brief template

Every Codex review request from this repo composes on `agent-standards/WORKFLOW.md` "Review
protocol" (findings P0–P3, falsifiable format, the smaller-diff/reuse statement) plus these
repo rules:

- **One structural objective per brief.** Never mix an architectural question with a cleanup or
  an upgrade in one request; split them into separate rounds.
- **Blockers first.** Structure/behavior mixing in a diff is reported only when it creates a
  concrete review or correctness risk, ranked at its normal severity — never ahead of a P0/P1.
- **Safety valve.** Flag any suggested simplification that changes behavior in edge cases,
  compilation output, hardware resource pressure, or latency — this repo's "structural" changes
  can alter generated IR and DMA behavior.
- **Reuse findings are grounded.** Name the existing file/symbol and its semantic compatibility;
  a "none exists" verdict states the search scope.
- **Suggestions are sized.** A proposal that cannot land inside the repo's 500-added-line cap is
  a plan, marked as such (agents/scripts/check_pr_size.sh defines the cap).
