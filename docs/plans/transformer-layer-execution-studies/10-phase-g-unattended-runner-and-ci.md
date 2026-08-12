# 10 — Phase G: Unattended Runner and CI

Drive the whole suite unattended, across reboots, with resumable checkpointing. Then wire what
can safely run in CI.

> ## `[2026-08-12]` READ THIS BEFORE THE REST OF THIS DOCUMENT
>
> **This spec was written against iron's design and half of it is obsolete.**
> [34](34-phase-g-scoping.md) investigated Phase G and found that roughly half of iron's
> 2,494-line `unattended_reboot.py` already exists here in better form, a quarter should be
> dropped on measured grounds, and what remains is small: a profile table, a runner that walks
> it, and row counts in the manifest.
>
> **G0 is built.** `study/profiles.py` + `study/run_profile.py`, `run_status="skipped"` emitted
> for the first time, and row-count validation in `study/manifest.py` — see §What G0 shipped.
> The CI leg is built too — see §CI wiring, which this document got wrong three ways.
>
> **Four of this document's behaviours are DECIDED AGAINST, not deferred** — see
> §Deliberately dropped. Between them they delete this document's entire passwordless-sudo
> block except one binary, and the reboot-loop failure class that halted iron's queue at job
> 885 of 888.
>
> **The honest blocker is not effort, it is reachability.** `run_mode.py` hardcodes
> `encoder_bert` and every whole-layer SPECS row is `emb_dim 768`, so **one of the six declared
> families is reachable**. A "full profile" over the declared 6×9 matrix needs a
> Phase-C-sized coverage sweep first (C4's precedent: 504 + 66 min of gate time). G0 does not
> attempt it; it scopes the profile to what exists and says so in the manifest and the run
> report, which is this repo's established idiom (`component_groups.py` reports `0/12` rather
> than presenting a mode total under a group label).

## The runner

~~Port `unattended_reboot.py` (2494 lines) and `test_unattended_reboot.py` (1790).~~
**Do not port it.** §1 of [34](34-phase-g-scoping.md) maps every behaviour below onto what
already exists. The device scheduler (`agents/scripts/devq.sh`, 20/20 selftest, 23 `flock` sites
migrated), power-mode enforcement (`require_turbo`, which *refuses* where iron warns), the
results manifest, the completeness gate, the per-output-file lock, power sampling, the case
matrix and row selection are all here and tested. What follows is kept as the record of what
was asked for; the state of each item is annotated.

The job plan is a data table of `(job_id, description, module, argv, privileged_setup)` dicts
built by `build_job_plan()` over families x sequence ladder x block kinds, then families x ladder
x execution modes, then the helper studies, exports, plots and manifest. **Retargeting is
confined to that function.**

Behaviour to preserve:

- **State** — `<results_root>/automation/state.json` holding `current_job_index`, per-job
  status, `reboot_command`, `baseline_temperature_c`, `normal_ttm_pages_limit`, and pending
  reboot actions. Per-job logs under `automation/logs/`.
  `[2026-08-12]` **Partly superseded, partly open.** `devq` keeps per-job logs and reconciles a
  dead job by process liveness, immune to pid reuse; `run_ladder` rewrites each mode's CSV after
  every rung so a killed run keeps what it measured; `run_profile` writes `profile_run.json`
  with the plan and every rung's outcome. `reboot_command`, `baseline_temperature_c` and
  `normal_ttm_pages_limit` are all dropped — see §Deliberately dropped. What is genuinely still
  missing is `current_job_index` **resume**, below.
- **Resume idempotence** — every job is passed `--resume-input <its own output CSV>`, so resume
  is idempotent at both the job and the row level. `resume` re-arms the hook, retries the failed
  job, and continues.
  `[2026-08-12]` **Still open, and it is the next increment after G0.** Copy the verdict split
  the registry sweep already has (`REUSABLE_STATUSES`, `sweep/registry_sweep.py:177`), which
  separates verdicts describing the CANDIDATE from verdicts describing the MACHINE — the former
  are reusable on resume, the latter must be retried. Note doc 14's harder lesson, learned by
  the port-loop driver: a halt must record *where* it halted, or resume redoes the work, empties
  the review diff and makes the tamper check vacuous.
- ~~**The `@reboot` crontab hook**~~ — **DROPPED**, see §Deliberately dropped.
- ~~**Thermal gating** between jobs via `sensors`, falling back to `rocm-smi`.~~ — **DROPPED**.
- **Power-mode enforcement** — `npu_runtime_checks.require_npu_power_mode_turbo()` parses
  `xrt-smi --batch examine -f JSON -r all`, falling back to a text regex.
  `[2026-08-12]` **Done, stricter, and the text regex is the PRIMARY path here rather than the
  fallback.** `sweep/registry_sweep.py`'s `require_turbo()` **refuses** where iron warns and
  continues, and "could not determine" is a refusal too. It is the single implementation
  `study/run_mode.py`, `study/component_groups.py`, `agents/scripts/port-loop/pmode_guard.py`
  and now `study/run_profile.py` all fail closed on — imported, never re-derived, because two
  parsers of one device's output disagree eventually and the disagreement shows up as a verdict.
  `run_profile` takes it once up front so a walk refuses before a results root exists;
  `run_mode` re-takes it per rung (each rung is a fresh process), so a driver reload mid-walk is
  caught at the next rung rather than at the end.

### Two fixes iron made that must carry over

1. **TTM page-limit comparison within 1%, not one page.** The kernel derives that default from
   boot-time available memory, so a reboot onto a different kernel shifted it by 602 pages, the
   clear-TTM step could never be satisfied, and the queue halted at job 885 of 888 to avoid a
   reboot loop. A 1% band still rejects the deliberate 26 GB override.
   `[2026-08-12]` **Not applicable here and re-filed.** There is no TTM step to carry it into —
   `amd-ttm` appears nowhere in this tree. iron's 26 GB override existed for six 16384-token
   **iGPU** jobs, i.e. `host_comparison`, which is unported and needs a ROCm torch wheel that
   conflicts with the pinned CPU-only index. Carry this as a note attached to *that* study if it
   is ever ported, not to Phase G.
2. **Guard the empty-mask column drop** in `plot_selected_component_groups_vs_pattern`. Selecting
   rows with a plain list mask means that when no row matches the execution mode, the empty list
   is read as a *column* selection, dropping every column, and the next lookup raises
   `KeyError: '_pair_key'`. Sparse result trees hit this routinely.
   `[2026-08-12]` **Re-filed against the plot tier.** That module is in the still-blocked
   plotting tier (needs matplotlib/pandas/seaborn, which must not be installed beside a live
   gate — doc 09 item 6). It belongs to queue item 11(b), not to Phase G.

### Convention rule 5

At 2494 lines, this module is three times the repository's norm. Split it along the seams it
already has: job planning, state persistence, crontab hook, thermal gate, TTM transitions,
reboot orchestration, CLI.

## Job counts

`[Codex]` **Do not hard-code iron's 888 / 834 / 21 / 3 counts as acceptance criteria.** Those
describe iron's case matrix; this port's matrix will differ, and a hard-coded number becomes
either a false failure or a rubber stamp.

Generate expected counts from the checked-in suite profile and validate them in the manifest.

For context, iron's profiles were:

| Profile | Jobs | Observed wall clock |
|---|---|---|
| `full` (default) | 888 | 11 h to 2 days |
| `paper` | 834 | ~20 h; helper studies restricted to seq_len 512/2048/8192 |
| `execution-smoke-test` | 21 | minutes |
| `smoke-test` | 3 | seconds; measures nothing |

Wall clock swings by a factor of four depending on how warm `build/` is — compilation, not
measurement, dominates a cold run. Preserve `build/` between runs.

> `[2026-08-12]` **Done, and the factor is understated at rung granularity.** devq job 224 walked
> 8 rungs (4 modes × {512, 1024}) cold in **631 s** and the same 8 warm in **32 s** — a **~20×**
> swing on the same hardware in the same job (per-rung walls 98/102/29/30/55/57/128/132 s cold
> against 5/5/2/3/6/7/2/2 s warm, `agents/.state/devq/jobs/job-000224.log`). Size a window off
> the cold number.
>
> **The counts are generated, as this section asks.** `study/profiles.py` declares three
> profiles as (modes × sequence lengths) over the one reachable family, and `expected_files`,
> `expected_rows`, `measured` and `skipped` are all *computed* from those tables — no count is
> typed out anywhere, so retargeting a profile retargets its gate. `study/manifest.py` validates
> against them. This port's counts:
>
> | Profile | Rungs | Measured | Skipped | Why skipped |
> |---|---|---|---|---|
> | `smoke` | 4 | 4 | 0 | — |
> | `ladder` | 16 | 14 | 2 | `fused` at 2048 and 4096 |
> | `full` | 36 | 30 | 6 | `fused` at 64, 128, 2048, 4096, 8192, 16384 |
>
> `full` is the nine-point ladder over **one** family, not a six-family matrix walk, and does not
> claim to be — the other five are recorded in the manifest's run report with the reason each is
> out of reach.
>
> **`full` is not expected to be green today, deliberately.** It attempts 64, 128, 8192 and
> 16384, which no mode has ever been measured at. Truncating it to the four points that are known
> to work would make it a synonym for `ladder` and would quietly convert "we have not measured
> this" into "this is not in the matrix" — the opposite of what `cases.py` says, which is that
> "which of them a mode can build is a separate question that only a run answers". So a rung that
> CANNOT apply is `skipped`; a rung nobody has tried is RUN, and its refusal message is the
> result. **`smoke` and `ladder` are the profiles to gate on.**

## Host prerequisites

Document these in the example README. The runner shells out to all of them, and a missing tool
fails mid-suite rather than at start unless checked.

| Tool | Package | Used for |
|---|---|---|
| `xrt-smi` | XRT | NPU power mode; probed at `/opt/xilinx/xrt/bin`, `/usr/local/bin`, `/usr/bin` |
| `amd-ttm` | `amd-debug-tools` | TTM page-limit transitions |
| `turbostat` | `linux-tools-$(uname -r)` | Package power for every end-to-end row |
| `sensors` | `lm-sensors` | Thermal gate between jobs |
| `rocm-smi` | ROCm | iGPU power, and the thermal-gate fallback |
| `crontab` | `cron` | The `@reboot` continuation hook |

`turbostat` is tied to the running kernel — `/usr/bin/turbostat` is only a dispatcher, and the
real binary ships per-kernel. Since this suite performs reboots, a kernel change mid-run can
leave every power row failing. Check with `sudo -n turbostat --version`.

Passwordless sudo is required because the runner executes from crontab and cannot answer a
prompt:

```
<user> ALL=(root) NOPASSWD: /usr/bin/xrt-smi, /opt/xilinx/xrt/bin/xrt-smi, \
                            /usr/local/bin/amd-ttm, /usr/bin/turbostat, \
                            /usr/sbin/reboot, /usr/bin/systemctl reboot
```

> `[2026-08-12]` **This table is largely obsolete for this host, and the sudo block collapses to
> ONE binary.** Measured, not assumed (doc 09 §Power):
>
> | Tool | State here |
> |---|---|
> | `xrt-smi` | **needed, and the only one.** Setting the pmode needs root; *reading* it does not, and reading it is all any script here does |
> | `amd-ttm` | unused — appears nowhere in the tree |
> | `turbostat` | **cannot run**: `sudo -n turbostat` fails, a password is required. Unprivileged `turbostat --no-msr` exits 0 and emits **no samples**. Replaced by `study/power.py`'s two root-free sysfs backends, both verified live |
> | `sensors` | dropped with thermal gating |
> | `rocm-smi` | matters only to `host_comparison`, which is unported |
> | `crontab` | dropped with the `@reboot` hook |
>
> So the sudoers line above reduces to `xrt-smi configure`, and even that is the **operator's**
> action between phases — no script here runs `sudo`. `agents/scripts/doctor.sh` already has the
> shape for a preflight and does not yet check pmode.

## NPU serialization

Wrap hardware jobs in the repository-wide convention:

```bash
flock -x -w 1800 /tmp/mlir-air-npu.lock <command>
```

`KernelCache` deliberately uses a *different* inode (`/tmp/npu.lock`) to avoid flock
self-deadlock. **Do not unify them.**

> `[2026-08-12]` **Superseded by `agents/scripts/devq.sh`, which is strictly more than this.**
> The bare `flock` design has no FIFO order, no writer preference and no fairness — measured, a
> writer blocked 3197 ms while a *later* reader acquired in 4 µs. devq is a real queue:
> monotonic sequence numbers, a measure/build barrier (a measurement at the head is absolute —
> later builds are not admitted until it has run), and status reconciliation by process liveness
> that survives SIGKILL. All 23 `flock` sites are migrated; `devq-selftest.sh` is 20/20.
>
> **A Phase G runner therefore needs no device serialization of its own.** `run_profile.py`
> takes no device lock and names no device lock path. `study/run_lock.py` is per OUTPUT FILE and
> is a third, unrelated inode; pointing it at either device inode would deadlock against the
> wrapper that launched the run. The two-inode warning above stands and is enforced in three
> independent places.
>
> **`run`, never `submit`, from a gate.** `submit` diverts output to the job log and returns an
> id, so substituting it blanks the gate's own FileCheck and still exits 0.

## CI wiring

~~Add to `programming_examples/CMakeLists.txt`, following the existing `add_lit_testsuite`
pattern:~~

```cmake
add_lit_testsuite(check-programming-examples-transformer-layer
  "Running transformer-layer execution-study tests (compile-only)"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TEST_DEPENDS}
  ARGS ${AIR_TEST_LIT_ARGS} -j1 --filter "transformer_layer/.*/run_npu2_compile"
)
```

- ~~**Compile-only is PR-gate-safe** — no NPU, no HF token, no secrets.~~
- **The measurement suite stays opt-in.** It runs 11 h to 2 days and needs NPU + iGPU on one
  host with passwordless sudo and reboot rights. That belongs to a dedicated runner invoked by
  `workflow_dispatch`, not to PR CI.
- `-j1` on any hardware suite is mandatory — concurrent NPU work causes OOM and contention.

> ### `[2026-08-12]` This block **cannot be applied as written**, and it was wrong three ways
>
> Measured against the real discovered test list (`lit --show-tests` over
> `build-xrt/programming_examples`, then lit's own filter semantics —
> `opts.filter.search(t.getFullName())` — applied to it):
>
> 1. **The target name already exists.** `check-programming-examples-transformer-layer` has been
>    declared since Phase A; `add_lit_testsuite` with the same name is a duplicate CMake target.
> 2. **The proposed filter matches `0` of 32 tests.** `transformer_layer/.*/run_npu2_compile`
>    requires an intermediate directory. The compile test is
>    `transformer_layer/run_npu2_compile_peano.lit`, at the top level. The `llms/` filters work
>    only because `llms/<model>/` has that level.
> 3. **"Compile-only is PR-gate-safe" is a false description of that target.** It selects all
>    **32** tests, of which **22** carry `REQUIRES: ryzen_ai_npu2`, 1 is Peano-only and 9 are
>    host-only. Wiring it into PR CI on the strength of its old comment would produce exactly the
>    hazard doc 15 opens with: on a runner with no NPU2 all 22 report UNSUPPORTED and lit still
>    **exits 0** — a hardware gate that runs zero hardware tests and reports success.
>
> **And a fourth thing, found while measuring the other three: PR CI already pulls 22 of these
> tests in today, and 21 of them are NPU-gated.** `check-programming-examples-peano` filters on
> the string `peano`, which matches every `run_npu2_*_peano.lit` under `transformer_layer/`, and
> `buildAndTestRyzenAI.yml` runs that target on both `amd8845hs` (NPU1) and `amdhx370` (NPU2).
> So on the 8845hs runner those 21 are silently UNSUPPORTED. That is pre-existing, repository-wide
> and NOT changed here — recorded so the next person does not rediscover it.
>
> **What shipped instead.** The existing target keeps its name (the port-loop driver runs it as
> the local regression gate) and gets a corrected comment saying it needs an NPU. A **second**
> target carries the PR gate:
>
> ```cmake
> add_lit_testsuite(check-programming-examples-transformer-layer-host
>   "Running transformer-layer PR-safe tests (10, no NPU dispatch)"
>   ${CMAKE_CURRENT_BINARY_DIR}
>   DEPENDS ${TEST_DEPENDS}
>   ARGS ${AIR_TEST_LIT_ARGS} --filter "transformer_layer/(run_npu2_compile_peano|...)[.]lit$"
> )
> ```
>
> — an explicit **allowlist** of the 10 genuinely PR-safe tests, verified to select exactly 10
> with 0 NPU-gated. An allowlist rather than a `--filter-out` deliberately: a negative filter
> enrols every future `.lit` automatically, so a new NPU-gated test would silently join the PR
> gate and report UNSUPPORTED. The allowlist fails the other way — a new host-only test is simply
> not gated until someone adds it, which is visible and harmless.
>
> `[.]lit$` and not `\.lit$`: CMake eats a backslash before a non-escapable character, so the
> anchor would silently degrade to `.lit$`. Verified with `cmake -P`.
>
> **The count is asserted in the workflow, and that is the load-bearing half.** A lit suite that
> selects nothing, or reports everything unsupported, exits 0 — so a green step is not evidence
> it ran. `buildAndTestRyzenAI.yml` now requires `Total Discovered Tests: 10`, `Passed: 10`, and
> that Passed is the only nonzero category. Same shape as `lib-guard.sh`'s
> `pl_assert_gate_ran_hardware`, which exists for this exact regression.
>
> **`-j1`, deliberately not applied to either transformer-layer target.** The host target
> dispatches nothing, so parallelism costs nothing. The hardware target has been run at 24
> workers and passed (30/30 in 519.7 s) — but doc 30 records `run_npu2_runlist_gate.lit`'s
> latency clause going intermittent under exactly that contention, so if that target is ever
> used as a measurement-adjacent gate rather than a correctness one it should be `-j1`. Changing
> it now would alter a standing gate's timing for no measured reason, which is the opposite of
> what this study is for.
>
> **Three known-red lit failures are NOT in either target** — `llms/llama32_1b_int4/.../run_o_gemv_ffn_int4_fused_npu2_peano.lit`,
> `conv2d_14x14/run_npu2_makefile_peano.lit`, `matrix_vector_multiplication/bf16/run_npu2_makefile_peano.lit`
> (doc 15, reproducible 2/2, all predating the study). They live under
> `check-programming-examples-peano`, which already has a retry. If a whole-tree sweep is ever
> promoted to a gate, they need an expected-failure allowlist keyed by test path first, with the
> rule that the allowlist shrinking is fine and growing needs a reason.

## Work items

1. ~~Split `unattended_reboot.py` along its seams (rule 5).~~ **Not applicable** — not ported.
2. ~~Retarget `build_job_plan()` to the ported studies.~~ **Done as `study/profiles.py`**, which
   is the same idea at a tenth the size: retargeting is confined to one table.
3. ~~Carry over the TTM 1% comparison and the empty-mask plot guard.~~ **Both re-filed** — see
   §Two fixes iron made.
4. ~~Derive expected job counts from the profile; validate in the manifest.~~ **Done** —
   `profiles.expected_rows()` + `manifest.build_manifest(..., expected_rows=...)`.
5. Write the prerequisites and recovery sections of the example README. **Open.**
6. ~~Register the compile-only lit suite in CMake.~~ **Done, differently** — see §CI wiring.
7. ~~Decide the opt-in workflow for the measurement suite.~~ **Decided: there is no workflow.**
   This host is a laptop and is enrolled as no CI runner; the nightly slot
   (`nightlyPerfBenchmark.yml`, cron `17 4 * * *`) is on `amdryzenai5pro340`, a *different*
   machine, and study results are host- and pmode-specific so they cannot ride it. The
   measurement half of Phase G is a **local operator-invoked command**, not a GitHub workflow:
   `systemd-inhibit --what=handle-lid-switch:sleep:idle agents/scripts/devq.sh run --class
   measure -- python3 study/run_profile.py --profile ... --out-dir ...`.
8. **NEW — resume.** See §The runner. Not started.

## Gate

A full profile run completes, and `results_manifest.json` shows no missing files or rows —
against counts derived from the profile itself, not hard-coded.

> `[2026-08-12]` **The gate is implemented and its "or rows" half was the missing piece.**
> `manifest.py` previously validated FILES: `complete` meant every expected CSV existed and had
> at least one `run_status=passed` row, so **a CSV that should hold nine rungs and held one
> reported `complete: True`**. It now takes `expected_rows` and checks three clauses per file —
> total rows, rows that must have passed, rows that must be `skipped` — every number derived
> from the profile.
>
> Read "full profile" as **the profile a profile names**, not as the declared 6×9 matrix. The
> matrix is not reachable (see the header) and a gate that cannot be satisfied is not a gate.

## What G0 shipped `[2026-08-12]`

Doc 34's recommended first increment, "one profile, one command, one manifest".

| File | What |
|---|---|
| `study/profiles.py` | the profile table: `smoke`/`ladder`/`full`, the reachable family and the five that are not with reasons, the `fused` 256..1024 applicability rule, and `expected_files`/`expected_rows` derived from all of it |
| `study/run_profile.py` | the one command: refuse off-Turbo → take `run_lock` → sample power → walk → `smoke_gate` → `results_manifest.json` + `profile_run.json`. `--dry-run` prints the plan and touches nothing; `--gate-only` re-verifies a recorded tree without re-measuring it |
| `study/manifest.py` | `expected_rows`, the three count clauses, `observed_rows` per file, `row_counts_checked` |
| `study/run_ladder.py` | `walk(..., skip_reason=fn)`: an inapplicable rung is written `run_status="skipped"` and **no child process starts**. A skipped rung is not a failed run, so the exit code no longer counts it as one |
| `programming_examples/CMakeLists.txt` | the corrected comment and the new `-host` target |
| `.github/workflows/buildAndTestRyzenAI.yml` | the PR gate step, with the count assertion |

Two defects doc 34 found are closed by that, and they close in the right order — the second
had to land first or the first's counts would be wrong by construction:

- **`run_status="skipped"` existed in the schema since v1 and nothing emitted it**, so a
  structurally inapplicable rung was recorded identically to a broken one. It is emitted now,
  and the reason rides `failure_message` prefixed `skipped:` (a new column would be a schema
  version bump).
- **The manifest validated files, not rows.** Closed above.

`run_lock.py` and `power.py` had no callers at all. `run_profile.py` is the caller both were
written for. The power block is deliberately **run-level**: it is SoC watts over the whole walk
with compilation included, no sensor on this platform measures the NPU, and compilation
dominates a cold walk ~20× — so it is a condition of the run and is never written into a
results row.

**Not done, and named so nobody assumes otherwise:** resume (item 8), the example README's
prerequisites/recovery sections (item 5), and the reachability sweep that would make `full` mean
the declared matrix.

## Deliberately dropped `[2026-08-12]`

Recorded as *decided against* with the measurement behind each, rather than left open. Doc 34
§4.4 recommends all four; this is that recommendation taken.

| Dropped | Why |
|---|---|
| **The `@reboot` crontab hook** | Installing it on a shared personal machine affects other users, and this document already flags the footgun that running `start` under `sudo` puts it in *root's* crontab. `systemd-inhibit --what=handle-lid-switch:sleep:idle setsid nohup …` — which doc 14 already uses in anger — covers the actual need on this host: surviving lid close, idle and logout (`KillUserProcesses=no`). Dropping it also removes the **reboot-loop failure class that halted iron's queue at job 885 of 888** |
| **TTM page-limit transitions** | `amd-ttm` appears nowhere in this tree. iron's 26 GB override existed for six 16384-token **iGPU** jobs — `host_comparison`, which is unported and needs a ROCm torch wheel conflicting with the pinned CPU-only index. Nothing to transition |
| **Thermal gating** (`sensors`, falling back to `rocm-smi`) | **No artifact anywhere in this tree shows thermal throttling affecting any recorded number.** Adding an ungrounded halt condition to an unattended runner adds a way to stop without adding evidence. If a thermal effect is ever measured, gate on the measurement |
| **`turbostat` power** | **Cannot run here.** `sudo -n turbostat` fails — a password is required — so iron's backend cannot run unattended on this host at all; unprivileged `turbostat --no-msr` exits 0 and emits no samples because `PkgWatt` needs MSR access. `study/power.py` already replaced it, porting iron's *statistics* (modified-Z filter, interquartile fallback, the ≥10/≥6 policy, and the rule that a filter which WIDENS the spread has found structure and is skipped) and swapping the sampling backend for two root-free sysfs readers, both verified live: RAPL `package-0` at 19.96 W over 2.00 s, `amdgpu` PPT at 22.05 W |

Between them these delete the entire passwordless-sudo block above except `xrt-smi configure`,
and one whole class of unattended failure. **They are also why the risk list below is shorter
than it looks** — see §Risks.

## Risks

- **The suite needs a dedicated machine.** NPU + iGPU on one host, passwordless sudo, reboot
  rights, ~2.4 GB per results root, and up to two days of wall clock.
  `[2026-08-12]` **Partly retired.** No sudo and no reboot rights are needed any more. What
  stands: a `full` profile monopolizes the only NPU for its whole duration — devq's `measure`
  class is an absolute barrier — so it is not "a nightly job", it is "nobody else uses this
  machine today". ~2.4 GB per results root on a laptop makes retention part of the design.
- ~~A healthy iron run rebooted exactly twice~~ `[2026-08-12]` **retired with the reboot
  machinery.** No run here reboots.
- ~~Installing a crontab hook on a shared machine~~ `[2026-08-12]` **retired** — no hook.
- `[2026-08-12]` **NEW, and the one that matters: unattended measurement is exactly where this
  project has published wrong claims.** The README lists three, two of them measurement-condition
  failures — a 1.55× inflation from host work running alongside a timed region, and a "5.9%
  improvement" that was three fresh runs compared against one stale number. An unattended runner
  multiplies the opportunity. The mitigation is to put the conditions **in the data** rather than
  in prose, which is the item G0 does *not* close: the manifest still records git and platform
  provenance and no measurement condition.
- `[2026-08-12]` **NEW: the profile is not the matrix.** Six declared families, one reachable.
  Either the profile is scoped to what exists and says so — which is what G0 does — or Phase G
  silently becomes a Phase-C-sized coverage sweep.
- `[2026-08-12]` **NEW: which toolchain tree a run tested.** `install-xrt` and `build-xrt`
  diverge whenever `mlir/` changes: the lit suites test the build tree, the probes and models
  (and a `devq` job that sources `lib-env.sh`) test the install tree. A run must state which.
  Check with `ls -l` on the two binaries, **never `cmp`** — the install step rewrites RUNPATH, so
  the bytes always differ and a `cmp` difference proves nothing.
