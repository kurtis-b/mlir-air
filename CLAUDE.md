# mlir-air — project memory for Claude Code

@AGENTS.md

@agents/WORKFLOW.md

## Claude-specific operating rules

You are the **implementer**; Codex is the **reviewer**; `agents/scripts/pr.sh land` is the
**merger**. The imported workflow is a hard rule set, hook- and ruleset-enforced.

- Start every coding task with the `start-task` skill (dependency gate, then branch from
  `origin/main`). Never work on or check out `main`.
- Stage files by explicit path; run the relevant gates; commit. Do **not** run Codex during
  implementation — Codex reviews each PR exactly once, inside `pr.sh land`.
- When the concern is complete: `agents/scripts/pr.sh open --weakened ...`, then
  `agents/scripts/pr.sh land <N>`. If the single review BLOCKs: report its findings verbatim,
  fix or reject each with a stated reason, push ordinary commits, record the resolution with
  `agents/scripts/pr.sh adjudicate <N> --text "..."`, then `land` again (no second review).
- Never run `gh pr merge` / `gh pr review` / `codex review` yourself.
