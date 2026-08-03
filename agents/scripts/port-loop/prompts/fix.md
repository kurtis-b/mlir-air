You are addressing code-review findings in a fresh session with no prior context. Another session
implemented this phase; Codex reviewed the diff; you are fixing what it found.

Repository: /home/cj/mlir-air
Branch: __BRANCH__
Phase: __PHASE_ID__ — __PHASE_NAME__
Review round: __ROUND__ of 3

## Read these first

1. `AGENTS.md`
2. `__PHASE_DOC__` — the phase specification the code is meant to satisfy.
3. `docs/plans/transformer-layer-execution-studies/02-porting-conventions.md` — binding style.

## The findings

Blocking findings from this round are below as JSON. Fix every one of them.

```json
__BLOCKING_JSON__
```

Non-blocking findings, for context — address them only if the fix is small and safe:

```json
__NON_BLOCKING_JSON__
```

## How to fix

Review the diff yourself before changing anything: `git diff __PHASE_START_SHA__..HEAD`. A finding
can be wrong. If you conclude one is mistaken, say so in `blockers` with your reasoning rather
than making a change you believe is incorrect — the next review round will re-examine it.

**Fix the code, never the test.** If a finding says a gate fails, the fix is in the
implementation. Weakening the gate is the one thing this whole harness exists to prevent, and it
is checked independently.

Your gate, unchanged:

```
__GATE_DESCRIPTION__
```

Return the structured report when done, listing what you fixed in `work_completed` and anything
you deliberately did not fix — with the reason — in `work_not_completed`.
