You are implementing one phase of a committed engineering plan, in a fresh session with no prior
context. Everything you need is in the repository.

Repository: /home/cj/mlir-air
Branch: __BRANCH__
Phase: __PHASE_ID__ — __PHASE_NAME__

## Read these first, in this order

1. `AGENTS.md` — repository conventions. Name your task profile as it asks.
2. `__PHASE_DOC__` — your phase. This is your specification: its work items are your task list
   and its gate is what you must satisfy.
3. `docs/plans/transformer-layer-execution-studies/02-porting-conventions.md` — binding house
   style. Ported code is rewritten to MLIR-AIR conventions, never transplanted.
4. `docs/plans/transformer-layer-execution-studies/01-port-inventory.md` — the per-artifact
   triage table, so you know what is meant to be ported, adapted, rewritten, or dropped.

The source material being ported lives at /home/cj/iron (branch `devel`, commit `1e014c1`). It is
a different repository with a different API; read from it freely, but do not copy files across
unchanged.

## Your gate

```
__GATE_DESCRIPTION__
```

The driver runs this gate itself after you finish, plus independent objective checks you cannot
influence. Do not report success on the basis of your own judgement — the gate decides.

## Environment

Already sourced for you: the `sandbox/` venv, `utils/env_setup.sh` against `install-xrt`, and XRT.
`aircc`, `air-opt`, `aie-opt` and `aiecc` are on PATH. Verify with `command -v aircc` rather than
re-sourcing anything — `utils/env_setup.sh` is not idempotent and re-sourcing corrupts PATH.

## What to do

Work through your phase document's work items. Commit as you go with clear messages; the driver
snapshots around you and a granular history is what makes your work reviewable and revertable.

Document what you build **in the example's own `README.md`**, next to the code, as you go: what
each new module is for, and every footgun you hit that the next reader would otherwise hit too.
Write down the things that cost you time — a flag that must be a `-D` rather than a `#define`, a
kernel whose objects cannot co-link, a reference that is not a valid oracle for the kernel it
looks like it matches. Those are the parts nobody can reconstruct from the diff.

Do **not** update the plan's status board in
`docs/plans/transformer-layer-execution-studies/README.md`. The driver writes that row itself from
what it measured, and it cannot be known while you are running: your wall time and your outcome
are decided by three review rounds, the gate and the objective check, all of which happen after
you finish. If you find a plan document that is now *wrong* — a claim your work falsified, not a
status — say so in `work_not_completed` rather than quietly editing around it.

When you are done — or blocked — return the structured report. Be accurate about
`work_not_completed`; the driver cross-checks it against the gate result, and an honest partial
report is far more useful than an optimistic one.
