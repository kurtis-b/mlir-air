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

Phases in scope are set by `PL_PHASES_IN_SCOPE` in `agents/scripts/port-loop/phases.sh` — a hard
assignment, not `${VAR:-default}`, so it is edited rather than overridden from the environment.
Adding a phase means adding a `case` arm to each of the seven dispatchers there (name, doc,
hardware flag, gate description, gate allowlist, gate command, objective check).

One of those arms fails **open** if forgotten: `phase_objective_check` defaults to `return 0`, so a
missing one passes vacuously. `phase_gate_description`'s `*)` arm now renders an explicit "this is
a harness bug" block rather than an empty one; `phase_gate_cmd` defaults to `false` and
`phase_gate_allowlist` to the empty string, which makes every changed gate file unauthorized. Those
three fail closed. Check a new phase with `run-one <phase> objective-check` before trusting it.

`[2026-08-05]` Phase D is entered as **two** sub-phases, `D1` and `D2`, for the reason Phase C was
split and one more: `PL_STEP_TIMEOUT` caps an implement session at three hours, and the
single-phase form asked one session for hardware bring-up on six operators *and* novel multi-launch
integration. Their objective checks layer a `baseline_768` coverage clause (D1) and full-layer
scope plus per-boundary stage assertions (D2) on top of `phase_c_operator_check`.

`[2026-08-05]` **Phase E is entered as five sub-phases, `E1`–`E5`**, for the reasons C and D split
and one more: E1 changes shared infrastructure and therefore carries the ten-model regression
check, which cost C4's gate hours. Folding that into a sub-phase that also builds an execution
strategy would re-run it on every failure of either. E5 is last because its objective check is the
only cross-mode one — the four dispatch vectors either separate or they do not, and that cannot be
asked until all four exist.

Of the three things this document said to settle when adding the entry, one was right, one was
wrong, and one grew:

- **"Split it" was right.** Five ways.
- **"The allowlist has to widen past `^programming_examples/transformer_layer/`" was wrong.**
  `guard_gate_files()` fingerprints `.lit` files, example `Makefile`s,
  `programming_examples/CMakeLists.txt`, `kernel_registry/details/*.json` and `llms/verify/*.py`.
  `gemm_builder.py` is in none of those sets, and `gate-e1.sh`'s second leg *runs* the ten shipped
  models rather than editing them. All five sub-phases keep the tight prefix — which is also what
  stops E1 quietly editing a shipped model's `Makefile` to make its own regression leg pass.
- **The objective check's "natural shape" was right and needed one thing more.** It is
  `phase_c_operator_check` per mode plus a driver-side cross-mode assertion, as predicted. What was
  missing is that **the dispatch vectors themselves had no negative control**. `results/` is
  gitignored, so a fabricated `dispatch_vectors` block is invisible to `guard_fingerprint`,
  `guard_check_tamper`, `guard_check_destructive` and every Codex diff — freshness was the only
  barrier, and it does not stop a number that was typed rather than measured. The fix reuses what
  is already there: the driver re-runs every operator under `--fault-inject input`, so the fault
  artifact's summed vector totals must **equal** the clean run's. Injection perturbs one input
  element after the reference exists and does not touch the dispatch path, so on an honest run they
  are identical — D2's block pair both total 4 / 131 / 12 / 146 / 402 / 202,902,528 — and a session
  cannot produce those six numbers without dispatching.

### The E checks are a module, and they run in both directions without hardware

`agents/scripts/port-loop/phase_e_checks.py`, with its fixtures in `phase_e_selftest.py`. Every
other phase embeds its objective check in `phases.sh` and at forty lines that is right; Phase E's
are an order of magnitude larger, and a module can be *tested*:

```bash
python3 agents/scripts/port-loop/phase_e_checks.py selftest    # 27 clauses, no hardware, no repo writes
```

Each case builds a conforming four-mode artifact set in a temp directory, applies exactly one
mutation, and asserts the verdict flips. That is this document's own twice-learned lesson made
routine — C4 halted on a check no honest run could pass because only its failure direction had been
tried, and Phase B passed a hardware gate that ran no hardware. The pass direction is additionally
demonstrated against real data: D2's `block` artifact pair satisfies the full-layer scope, the
vector contract and the provenance clause unmodified.

Writing the selftest immediately repaid itself: the first version shared its fixture vectors by
reference, so one case's mutation leaked into every later case and several clauses were "failing as
expected" for the *previous* case's reason. A green selftest that proves nothing is the same defect
as a green gate that runs nothing.

### The driver now watches itself

`guard_gate_files()` includes `agents/scripts/port-loop.sh` and everything under
`agents/scripts/port-loop/`, and **no phase's allowlist covers them**, so any session edit halts the
run.

This was a gap from Phase A onward. All three anti-reward-hacking layers police what a diff did to
a *gate*; not one watched the thing that runs the gates, while sessions execute under
`--permission-mode bypassPermissions`. And the review would not have caught it either: `run_phase`
sets the review base from `HEAD` at phase entry, so the driver's own commits sit *before* the base
and are outside every phase's review diff — the same blind spot that made D1's round-3 reviewer
raise a finding about fault injection that the driver was already doing.

### A convention violation this phase found and did not fix

`phases.sh` is **1306 lines** against the ~800 cap that [00](00-context-and-goals.md),
[02](02-porting-conventions.md) and [13](13-verification-and-acceptance.md) gate on. It was 1028
before Phase E and no document had recorded it — the same oversight as `opcheck_specs.py` at 1043
and `sweep/registry_sweep.py` at 866, which E1 is splitting.

It is recorded rather than fixed because the obvious two-way split does not help: roughly 1050 of
those lines are objective checks, so a `phases.sh` / `phase-checks.sh` split leaves the second file
over the cap too. The real seam is per-phase-family — the table, then C/D's checks, then E's — and
that is a deliberate refactor, not something to do to the driver in the hour before a
multi-day run. Phase E's own checks are already outside the file, which is why it only grew by
~280 lines rather than ~700.

### A sixth lesson: a coverage clause is not a correctness clause

Writing D's checks reproduced the harness's own recurring mistake in a new place, and an
independent review caught it before the run started. Three defects, all in the first draft:

- **It read `results/*.json` off the filesystem and trusted the `passed` flag.** Six hand-written
  files containing three keys each — no `n_mismatch`, no `rtol`, no `ref_dtype` — passed it. And
  `results/` is **gitignored**, so such a file is invisible to `guard_fingerprint`,
  `guard_check_tamper`, `guard_check_destructive` and every Codex diff: freshness was the only
  barrier, and the session's own lit recipe runs inside the gate window. The fix intersects with
  `opcheck.py --list` and re-derives the contract per record.
- **It ignored `seq_len`.** `qkv_proj`'s clause was `emb_dim == 768`, which the Phase C row already
  on disk satisfies — D1 could have passed it having added nothing at all.
- **It put the negative control on the wrong shape.** `phase_c_operator_check` injects one shape
  per operator *name*, taking the first declared key, which for `addnorm` is the `64x512` post-add
  row. The pre-add variant — the one function in D1 that has never run on hardware, and the one
  whose reference is likeliest to agree with the device by construction — would never have been
  injected at all.

The generalizable form: **the layer that proves an operator is correct and the layer that proves it
was exercised at the right shape are different layers, and re-using the first for the second gets
you neither.** A coverage clause needs its own freshness, its own verdict re-derivation, and its
own negative control.

### A seventh: what a provider outage found that nothing else had

`[2026-08-05]` A sustained `529 Overloaded` incident during D1 halted the run twice. The retries
were the least interesting part; recovering from them exposed two defects in paths that only
execute after something has already gone wrong, which is why nine phases had never reached them.

**`state_halt` did not record where it halted.** With `.resume_phase` null, `run_phase` computes
`resume_from = 0` and re-runs the phase from its implement session — throwing away completed,
committed work. Worse, at `resume_from = 0` it also re-derives `_START_SHA` from the *current*
HEAD, so the review diff and the gate-file fingerprint baseline both come from a tree that already
contains the halted attempt's commits: the reviews would have seen an empty diff and the tamper
check would have been vacuous. That is the same defect as the working-tree fingerprint baseline in
[the table above](#anti-reward-hacking-and-how-it-failed-first), reached by a different road. The
halt message instructs the operator to run `resume`, so that instruction has to be correct.

**Retries were charged against `PL_MAX_INVOCATIONS`.** Nine attempts were spent without a single
token billed — a run exhausting its budget while standing still. The cap bounds agent work and an
attempt that never reached the model did none; the wall-clock deadline is the right bound on
waiting.

The lesson is not about outages. It is that **the recovery paths are the least-tested code in the
harness, and they are the ones that run when you are least able to supervise them.** Both defects
were latent from the first phase and neither could surface until a halt happened somewhere other
than the implement step.

### The confirm review, exercised

`[2026-08-05]` D2 round 3 raised two blocking findings — the block lit gate never invoked
`check-block-fault`, and the golden-model identity tests were enrolled in no lit test at all. The
fix session addressed both, and the confirm review then read that fix diff alone
(`64e946d7..HEAD`) and returned clean, naming each finding and what resolved it. That is C4's exact
situation, and it is now reviewed rather than verified by hand afterwards.

It also stayed quiet when it should: D2's round-2 fix was covered by round 3's review, so no
confirm ran for it. Cost was one codex invocation, about two minutes.

One caveat found the same night. D1's round-3 review raised a blocking finding — that the new
`baseline_768` shapes were never fault-injected — which was **wrong**, because the driver injects
every one of them. The reviewer could not see that: the driver's own commits sat *before* the phase
base, so they were outside the review diff. Re-running the round with those commits inside the diff
cleared it immediately. This is the same lesson as "review the harness alongside the phase", in the
opposite direction: excluding the driver hid a check that *exists*, and cost a fix round to
rediscover.

`cmd_loop` reads `.phases` from `state.json`, not from `PL_PHASES_IN_SCOPE`, and `resume-at`
resolves an unknown phase id to `index // 0` — i.e. **phase 0**, re-running the first phase. After
changing scope, launch with `start`, which re-inits state.

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
  → confirm → gate → hardware-check → objective-check → tamper-check → advance | halt
```

`[2026-08-05]` `run-one <phase> objective-check` now sources the venv via `pl_env_ensure` before
dispatching. In a real run `pl_preflight` has already done it; standalone it had not, so Phase E's
naming clause — which resolves GEMM specs through `shared.builders.gemm_builder` — failed on
`No module named 'ml_dtypes'` while the driver reported it as a live symbol collision. A check that
reports the wrong cause is the failure mode this whole section exists to avoid.

Three rounds always run; a clean round's fix step is a no-op. State lives in
`agents/.state/port-loop/state.json` (gitignored), and phases resume mid-phase via
`resume_phase`/`resume_step`/`resume_round`.

`confirm` and `hardware-check` were added on 2026-08-05; both are described below.

## Why three rounds

Phase A round 2 **passed** and round 3 **failed** on byte-identical code — round 2's fix was a
no-op, so nothing changed between them. The rounds are therefore **repeated samples of a
non-deterministic detector**, not iterative refinement, and one passing round means little.

Phase B behaved differently because fixes landed between rounds: blocking counts fell 3 → 2 → 1 →
clear. So the rounds do converge when there is something to converge on.

Both argue for keeping three. At `medium` effort a review is 2–5 minutes and a clean round costs
no fix session, so the marginal cost is small and the variance reduction is real.

### The fix nothing reviewed, and the `confirm` step

`[2026-08-05]` The loop is review → fix, repeated, which means **the last fix to run is never
reviewed by anything**. The round that requested it has already finished and there is no round
after it. C4 hit this squarely: its round-3 review raised two blocking findings, both were fixed,
and the only thing that ever checked those fixes was a human reading them afterwards.

A fourth round is not the answer — the rounds are repeated samples, so round 4's own fix would be
just as unreviewed, and the regress does not terminate. What terminates is a step that reviews
*only* the final fix's diff and asks only whether that fix is correct: `prompts/confirm.md`, run
over `<sha-before-the-last-fix>..HEAD` through the same `pl_codex_review`. It is skipped entirely
when no fix ran, which is the common case for a clean phase.

It halts on blocking findings, and it runs **before** the gate — the cheapest place in a phase to
stop, since no hardware time has been spent yet. To keep that halt rare the prompt is calibrated
explicitly: style, structure, coverage and anything outside the fix diff are non-blocking by
instruction, and a session's reasoned refusal to apply a finding it believes is wrong counts as a
correct outcome rather than a defect.

Cost is one codex invocation per phase, and only when a fix actually ran.

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

### A fourth failure, found later: the gate that ran no hardware

`[2026-08-04]` Phase B's `phase_gate_description` describes a hardware test. Its `phase_gate_cmd`
was `flock … ninja -C build-xrt check-programming-examples-transformer-layer`, and at the time that
suite contained exactly two tests: a compile-only one (peano, no NPU) and a host-only one. The gate
log records 2 tests, 329 excluded, **16 seconds**. The phase's central hardware claim — the
multi-ELF runlist the whole taxonomy rests on — was produced by `make runlist-gate`, which the
session ran and reported itself.

Nothing caught it. The three anti-reward-hacking layers all police *what the diff did to the
gate*; none asks whether the gate exercises what its description claims. `run_npu2_runlist_gate.lit`
now puts the hardware legs inside the suite, and re-running it confirms all four pass — but the
result stood on a self-report for a day.

The generalizable lesson is narrower than "check the gate": **a phase whose `needs_hardware` is
`yes` should have to prove its gate executed at least one hardware test.** That is a driver-side
assertion of the same kind as the objective check, and it is cheap — lit reports its own
pass/exclude counts.

`[2026-08-05]` Now implemented, as `pl_assert_gate_ran_hardware` in `lib-guard.sh`, called from
`run_phase` immediately after the gate passes. For a `needs_hardware=yes` phase it requires the
suite to contain at least one `.lit` whose `REQUIRES` names `ryzen_ai_npu2` — counted by reading
the files, not their names, because `run_npu2_compile_peano.lit` is named `npu2` and requires only
Peano — and then, from the last lit summary in the gate log, that **`Passed` and `Excluded` are the
only nonzero outcome categories** and that `Passed` reaches the tracked `.lit` file count.

The second clause is stated that way for two reasons. It covers the `XRT_COREUTIL` /
`ENABLE_RUN_XRT_TESTS` regression in [15](15-environment-notes.md), where lit cannot find
`xrt-smi`, marks every NPU test UNSUPPORTED, and the suite still exits 0. And it needs no list of
lit's category names, so it stays correct as lit adds them.

**The count must have no slack.** The first version required `Passed >= npu_tests` — 9 NPU-gated
of 13 files — which left four tests of headroom. Marking one NPU test `XFAIL` is a one-line edit
*inside* D1/D2's own allowlist; lit then counts it as "Expectedly Failed", neither Passed nor
Unsupported, and exits 0, and the assertion passed. Since lit runs succinct here there are no
per-test `PASS:` lines to correlate against, so the exact correspondence available is the total:
this gate runs every `.lit` under `transformer_layer/` and they must all pass.

Tested in both directions before anything depended on it: `phase-B/gate.log` (2 passed) fails, a
synthetic `Unsupported: 9` log fails, an `Expectedly Failed: 1` log fails, a log with no summary at
all fails, a tree with no NPU-gated test fails — which is Phase B's original defect exactly — and a
full `Passed: 13` log passes. `run-one <phase> hardware-check [gate-log]` exists so any of those
can be re-run against any log.

### A fifth: a driver-side check no honest run could pass

`[2026-08-04]` Phase C4 passed its gate — lit suite green, all ten shipped models still verifying —
and then halted on the objective check, which demanded the registry JSON be newer than the gate's
start stamp.

That requirement was unsatisfiable. It was written by analogy with Phase A, where the freshness
test works because the gate itself *rebuilds* the objects it inspects, so they are necessarily
newer than the stamp. C4's sweep runs in the implement session, hours before the gate, and
`gate-c4.sh` deliberately does not re-sweep. The JSON is therefore always older than the stamp.

Two things are worth keeping from it. First, the C4 session diagnosed the contradiction in review
and reported it, explicitly noting it had not touched the file to get past the check — which is
exactly the behaviour the guardrails ask for, and it is what made the bug legible instead of
invisible. Second, mtime was a weak proof anyway: one `touch` forges it. The replacement takes the
proof from **git** — the staged shapes must have been absent at the phase base commit and present
now — which is unforgeable by a filesystem timestamp and is what the requirement was reaching for.

The generalizable lesson pairs with the one above. A driver-side check must be tested in both
directions before a run depends on it: that it *fails* when it should, and that a correct run can
actually *pass* it. Only the first was tested here.

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

| Phase | Invocations | Wall clock |
|---|---|---|
| A | — | 18 min |
| B | — | 362 min |
| A + B together | 11 of 40 | 380 min |
| C1 | — | 61 min |
| C2 | — | 45 min |
| C3 | — | 68 min |
| C4 | — | 504 min, then 66 min to re-run the gate after the objective-check bug |
| C1–C4 together | 10 of 40 | ~12 h |
| D1 | — | 11 min of work, inside a ~2 h window dominated by a provider outage |
| D2 | — | 156 min |
| D1 + D2 together | 21 of 40 | ~4.5 h wall clock, of which ~1 h was the outage |

D1's eleven minutes are real: its implement session did the work in one pass, all three review rounds were clean on the code, and the gate plus the thirteen fault injections ran in four. The wall-clock window around it was three times that, entirely because of the 529 incident below. **Splitting Phase D paid for itself the same way splitting C did** — D2 needed 156 minutes and two rounds of substantive fixes, and none of that re-ran D1's hardware time.

Phase B's six hours were dominated by the hardware runlist spike and by ten `make verify` runs
across every shipped model — the cross-deployment regression rule being honoured rather than
skipped. C4's eight hours were the registry sweep plus the same ten-model check, which its gate
runs directly (`gate-c4.sh`) rather than describing and leaving to the session.

**Splitting a large phase paid for itself.** Phase C's source material was 8,160 lines against
Phase B's 3,725, and it cost the same order of invocations while every sub-phase gate passed first
time. C1–C3 averaged 58 minutes each. The cost is concentrated entirely in whichever sub-phase
owns the hardware sweep.

Codex spend was dominated by **four aborted Phase A restarts**, every one halting on a harness
bug rather than on the code under review. The per-phase steady-state cost is three reviews.

The review base is already phase-local: `run_phase` sets it from `HEAD` at phase entry, so
Phase B reviewed 3,725 lines against Phase A's end commit, not the cumulative diff.
