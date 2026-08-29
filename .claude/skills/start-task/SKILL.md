---
name: start-task
description: Begin a unit of work - check dependency PRs are merged, then branch from origin/main.
  Use at the start of any new coding task, or when the user says "start task" or names dependency
  PRs.
---

Run the dependency gate and then the branch step of the "Git workflow" in
`agents/WORKFLOW.md` (steps 7, then 1), in order. Stop immediately on DEPS-BLOCKED.
