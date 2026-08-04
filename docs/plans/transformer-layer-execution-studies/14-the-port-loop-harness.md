# 14 — The port-loop harness

`agents/scripts/port-loop.sh` runs phases of this plan unattended. Each phase gets a **fresh**
`claude -p` session, three sequential Codex review→fix rounds, and then gates run **by the
driver** rather than reported by the session.

It executed Phases A and B on 2026-08-04. This document is how to use it and what it got wrong
first, because most of those mistakes are not obvious and one of them silently disabled a check
the whole design rests on.

## Running it

```bash
./agents/scripts/port-loop.sh help          # usage; options are env vars, doctor.sh style
./agents/scripts/port-loop.sh dry-run       # print every step, execute none
./agents/scripts/port-loop.sh status        # or read agents/.state/port-loop/STATUS.md
./agents/scripts/port-loop.sh start         # begin at the first phase in scope
./agents/scripts/port-loop.sh resume        # continue after a halt
./agents/scripts/port-loop.sh stop
./agents/scripts/port-loop.sh run-one <phase> <step>          # one step, in the foreground
./agents/scripts/port-loop.sh resume-at <phase> <step> [round] [base-sha]
```

Detached, surviving logout and the lid:

```bash
PL_CODEX_EFFORT=medium systemd-inhibit --what=handle-lid-switch:sleep:idle \
  setsid nohup ./agents/scripts/port-loop.sh resume \
  > agents/.state/port-loop/driver.log 2>&1 &
```

Phases in scope are set by `PL_PHASES_IN_SCOPE` in `agents/scripts/port-loop/phases.sh`. Adding
Phase C means adding its doc, gate command, hardware flag, gate allowlist and objective check
there.

### Watching it

`--output-format json` buffers everything until a session ends, so the terminal stays silent for
hours. That is expected, not a hang.

```bash
tail -f agents/.state/port-loop/driver.log                  # driver narration + EVENT: markers
tail -f $(ls -t ~/.claude/projects/-home-cj-mlir-air/*.jsonl | head -1)   # live tool calls
```

To be notified once when a run reaches a terminal state, prefer a bounded wait over an unbounded
tail — an `until` loop on `jq -r .status state.json != "running"`, backgrounded. A persistent
`tail -f | grep` monitor was tried and its notifications did not arrive; the bounded wait worked.

## The step machine

```
preflight → implement → commit
  → [ review₁ → fix₁ → commit ] × 3
  → gate → objective-check → tamper-check → advance | halt
```

Three rounds always run; a clean round's fix step is a no-op. State lives in
`agents/.state/port-loop/state.json` (gitignored), and phases resume mid-phase via
`resume_phase`/`resume_step`/`resume_round`.

## Why three rounds

Phase A round 2 **passed** and round 3 **failed** on byte-identical code — round 2's fix was a
no-op, so nothing changed between them. The rounds are therefore **repeated samples of a
non-deterministic detector**, not iterative refinement, and one passing round means little.

Phase B behaved differently because fixes landed between rounds: blocking counts fell 3 → 2 → 1 →
clear. So the rounds do converge when there is something to converge on.

Both argue for keeping three. At `medium` effort a review is 2–5 minutes and a clean round costs
no fix session, so the marginal cost is small and the variance reduction is real.

## Effort level

`PL_CODEX_EFFORT=medium` is the right default. A `medium` review found a real bug that an
`xhigh` review had missed, in a tenth of the runtime (2.5 min vs 19). Model and effort otherwise
default to `~/.codex/config.toml`, deliberately: passing an explicit `-m` alongside
`--ignore-user-config` gets rejected on ChatGPT accounts.

## Anti-reward-hacking, and how it failed first

The design assumes an autonomous agent told "make the gate pass" may make the *gate* weaker
instead of the code better. Three independent layers defend against that. **On the first real
run, one layer was disabled and another was weaker than documented** — caught only because the
third worked.

| Layer | What it does | How it broke first |
|---|---|---|
| Gate-file fingerprint | Hashes every `.lit`, example `Makefile`, `CMakeLists.txt`, registry JSON and verify module, against a per-phase allowlist | Baseline was taken from the **working tree**, ten hours after the phase's commits, with the phase's own gate edits already in it. The check was vacuous. Now hashed at the phase **base commit** via `git show`, with files absent at base recorded as `ABSENT_AT_BASE`. |
| Objective check | Driver-side assertion about build products the session cannot satisfy by writing a laxer test | Accepted *any one* fresh object over 4 KiB with *any* symbol. Then, once tightened, still collected symbols **globally**, so stale objects supplied symbols for kernels the gate never rebuilt. Now: every `extern "C"` symbol parsed from every kernel **source** must be defined by an object **this gate rebuilt**; stale objects are counted and ignored. |
| Codex `weakened_gates` | Review reports checks the diff made weaker | Conflated "this diff weakened X" with "X is inherently limited", so observations like *a compile-only gate cannot prove numerical correctness* halted the run. Split into `weakened_gates` (halts) and `gate_limitations` (recorded only). |

Two lessons worth carrying:

- **Review the harness alongside the phase.** Including the driver's own commits in the review
  base is what caught the vacuous fingerprint. Narrowing the base to exclude the driver would
  have hidden it.
- **Test the negative path.** Each fix above was only trusted after confirming it *fails* when it
  should: a planted symbol no object defines, and a run where 7 objects were backdated and one
  rebuilt.

## Gates are run by the driver, never self-reported

Sessions return a structured self-report (`work_completed`, `work_not_completed`, `blockers`,
`gate_files_touched`). It is treated as a **claim to be checked**, useful for the audit trail and
for deciding whether to halt early — never as evidence. The gate and objective check decide.

This works. A Phase A fix session reported that a review finding did not reproduce on this host,
tested it, applied the fix anyway as a latent portability issue, and said explicitly it was "not
a live breakage repaired."

## Guardrails

Sessions run under `--permission-mode bypassPermissions`, which is a real exposure — the fence is
a system prompt plus post-hoc `git status` checks, not a sandbox.

- Run-scoped `.git/hooks/pre-push` fence; the driver never pushes
- Snapshot commit before every phase and after every step
- Deleted-tracked-file detection across the phase
- `timeout` on every invocation; `--max-budget-usd` and `--max-turns` per session
- Total invocation cap and a wall-clock deadline in `state.json`
- Hardware steps under `flock -x -w 1800 /tmp/mlir-air-npu.lock`. The driver never takes
  `/tmp/npu.lock` — that inode belongs to `KernelCache` and the lit suites, and taking it
  deadlocks them.

## Measured cost

Phases A and B: **11 of 40 invocations**, 18 min and 362 min wall clock. Phase B's six hours were
dominated by the hardware runlist spike and by ten `make verify` runs across every shipped model —
the cross-deployment regression rule being honoured rather than skipped.

Codex spend was dominated by **four aborted Phase A restarts**, every one halting on a harness
bug rather than on the code under review. The per-phase steady-state cost is three reviews.

The review base is already phase-local: `run_phase` sets it from `HEAD` at phase entry, so
Phase B reviewed 3,725 lines against Phase A's end commit, not the cumulative diff.
