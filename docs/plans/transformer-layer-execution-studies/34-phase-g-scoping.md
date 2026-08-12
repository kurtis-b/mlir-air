# Phase G — unattended runner and CI: what it would actually be

Investigation only. Repo `/home/cj/mlir-air`, branch `exper/transformer-layer-execution-studies`,
tip `b777517b`. **Nothing changed, nothing built, no device job run.** `build-xrt` untouched
(two concurrent compiler builds).

Three host-only, read-only commands were run to convert inferences into artifacts; all three are
pure-stdlib Python over already-recorded result trees, no NPU, no XRT, no writes into the repo:

| command | result | why it was run |
|---|---|---|
| `python3 study/smoke_gate.py results/phasef_smoke --expect {coarse,offload,runlist,fused}.csv` | `FAIL (4 problems)`, exit 1 | tests whether the recorded Phase F gate artifact still verifies |
| `python3 study/smoke_gate.py results/postflip-ladder-w1 --expect …` | `PASS (4 CSVs, each with a passed row)`, exit 0 | tests the same machinery on a current tree |
| `python3 study/manifest.py results/postflip-ladder-w1 --expect … -o <scratchpad>/probe-manifest.json` | `complete: True`, exit 0 | ditto for the manifest, written to scratchpad not the repo |

Every figure below cites the file behind it. Inferences are marked **[inference]**.

**No latency claim is made anywhere in this document.** NPU pmode is `Default`; per README trap 0
that puts every latency ~15–20× off. Everything quoted here is a *wall-clock job duration* from a
`devq` meta file or a recorded doc, which is a scheduling fact, not a measurement of the machine.

---

## 0. The one-paragraph answer

Phase G as doc 10 specifies it is a port of iron's 2,494-line `unattended_reboot.py` plus its
1,790-line test. **That framing is now substantially obsolete.** Roughly half of what that module
does already exists here, in better form, spread across `agents/scripts/devq.sh` and the Phase F
study tier; another quarter (TTM page limits, `turbostat` power, thermal gating, the `@reboot`
crontab hook) is either impossible on this host, already replaced, or actively inadvisable on a
personal laptop. What is genuinely missing is small and specific: **a named suite profile, a
runner that walks it, and a manifest that validates row counts against it.** The three environmental
hazards that would actually bite an unattended runner are all *already fail-closed in the study tier*
and *not guarded at all in the two latency-sensitive pass/fail gates a CI leg would run* — which is
the single highest-value item on the whole list and takes an hour.

---

## 1. What already exists that Phase G would build on

### 1.1 The device scheduler — done, and it is not what doc 10 asks for

`agents/scripts/devq.sh` (321 lines) is a FIFO device broker with monotonic sequence numbers and a
build/measure barrier. Doc 10 §NPU serialization only asks for `flock -x -w 1800
/tmp/mlir-air-npu.lock`; devq is strictly more than that and its header explains why the obvious
readers-writer design was refuted by measurement (writer blocked 3197 ms while a *later* reader
acquired in 4 µs — `agents/scripts/devq.sh:4-10`).

- Migration complete: "All 23 `flock` sites in `phases.sh` and `llama32_1b_int4`'s `run-inference`
  are migrated" (`README.md:477-479`).
- `devq-selftest.sh` **20/20** (`README.md:480`); the file itself declares 6 test groups
  (`agents/scripts/devq-selftest.sh:22,34,47,81,103,121`).
- Status reconciliation by process liveness immune to pid reuse (`/proc/<pid>/environ` carries
  `DEVQ_JOB_ID`, `devq.sh:80-92`) — this is the *"a job died and left the queue wedged"* recovery
  that an unattended runner needs, already written and already exercised (248 jobs on disk,
  `agents/.state/devq/jobs/*.meta`).
- `run` vs `submit` footgun already documented and enforced: `submit` diverts output to the job log
  and returns an id, so substituting it at a gate blanks the FileCheck and still exits 0
  (`devq.sh:26-30`, `phases.sh:717-727`).

**Consequence for Phase G:** the runner does not need its own device serialization, its own
liveness reconciliation, or its own orphan reaping. It needs to be a `devq measure` job.

### 1.2 The phase driver — done, and it is a *different* machine from Phase G's

`agents/scripts/port-loop.sh` (645) + `port-loop/phases.sh` (2,260) + six `lib-*.sh` + two Python
check modules (`phase_e_checks.py` 770, `phase_e_selftest.py` 365). It drives *implement → review ×3
→ confirm → gate → hardware-check → objective-check → tamper-check* per phase (doc 14 §The step
machine).

Things it already learned that Phase G's runner would otherwise re-learn from scratch:

| lesson | where | why Phase G needs it |
|---|---|---|
| a halt must record *where* it halted, or resume throws away committed work and makes the tamper check vacuous | doc 14 §A seventh, §The step machine | doc 10's `state.json` resume has the identical shape |
| recovery paths are the least-tested code and run when you are least able to supervise | doc 14 §A seventh | this is Phase G's entire thesis |
| a hardware phase must *prove its gate executed ≥1 hardware test* — `pl_assert_gate_ran_hardware` in `lib-guard.sh` | doc 14 §A fourth | it exists **specifically** to catch the `XRT_COREUTIL`/`ENABLE_RUN_XRT_TESTS` hazard (doc 14:335-339) |
| the count must have no slack — `Passed` must reach the tracked `.lit` count and `Passed`+`Excluded` be the only nonzero categories | doc 14:329-346 | the general shape of "derive expected counts from the profile" |
| retries must not be charged against the invocation budget | doc 14 §A seventh | a suite that exhausts its budget standing still |
| the driver watches itself — `guard_gate_files()` covers `port-loop.sh` and everything under `port-loop/`, and no allowlist covers them | doc 14 §The driver now watches itself | a Phase G runner is a gate-adjacent artifact too |

`pl_assert_gate_ran_hardware` is the single most reusable thing in the harness for CI. It is already
tested in both directions against synthetic logs (doc 14:347-352).

### 1.3 The Phase F study tier — larger than doc 10 anticipates, and partly unwired

17 modules, `study/run_host_tests.py` reports ~~231/231~~ **`[2026-08-12]` 265/265 in 17 modules**
(M4 below) in ~0.4 s, pinned by FileCheck in `run_study_host_tests.lit` and verified in the
shrinking direction (`09-phase-f-study-harness.md:528-538`). Still hermetic: the M4 probe is
exercised against a stub `xrt-smi` on PATH, never the device.

Mapping onto doc 10's stated behaviours:

| doc 10 behaviour | state | evidence |
|---|---|---|
| **manifest** (`results_manifest.json`) | **done and better** — `complete` means *measured*, not *present*, delegating to `smoke_gate.check_results_root` | `study/manifest.py:14-25,148`; iron's `results_manifest.py` deliberately **not** ported (`09:400-405`) |
| **the smoke gate** | **done** — ≥1 `run_status=passed` row per expected CSV, quotes the first `failure_message` verbatim, refuses an empty `--expect` | `study/smoke_gate.py:56-110`; iron's `unattended_smoke_job.py` measurement half superseded (`09:409-412`) |
| **power-mode enforcement** (`require_npu_power_mode_turbo`) | **done and stricter** — iron *warns and continues*; here it **refuses** | `sweep/registry_sweep.py:209-225`; parses `xrt-smi --batch examine -r platform` then `-r all` with a text regex, i.e. doc 10's fallback is the primary path here. iron's `npu_runtime_checks.py` deliberately not ported (`09:405-408`) |
| **per-output-file lock** (`run_lock.py`) | **ported, and has no caller** | `study/run_lock.py`; `grep -rn run_lock` outside the module and its test returns nothing |
| **power sampling** (`end_to_end/power.py`) | **ported, root-free backends verified live, and has no caller** | `study/power.py`; doc 09:414-426. Every `avg_power_w` column in every CSV is empty today |
| **cross-run comparison** (`compare_results_roots.py`) | **done, three of iron's holes closed** | `study/compare_roots.py`; doc 09:428-436 |
| **case matrix** (`end_to_end/cases.py` + `plot_families.py`) | **done as one table** — 6 families × 9-point ladder | `study/cases.py:76-166`. Consumed **only** by `select_rows.py:70` |
| **row selection** (`end_to_end/select.py`) | **done, renamed to avoid stdlib shadowing** | `study/select_rows.py:7-17` |
| **the job walker** | **partial** — `study/run_ladder.py` walks (mode × seq), one process per rung, rewriting each mode CSV after every rung so a killed run keeps what it measured | `study/run_ladder.py:189-201` |
| **resume idempotence** (`--resume-input <own CSV>`) | **absent** | `run_ladder.py:222-231` has no such flag; a re-run redoes every rung |
| **state.json / `current_job_index` / per-job logs** | **absent** | no such file; devq keeps per-job logs but has no notion of a plan |
| **job plan from a profile** | **absent** | no `profiles.py`, no `build_job_plan`, `grep -rn "suite_profile"` outside `docs/plans/` returns nothing |
| **`@reboot` crontab hook, TTM transitions, thermal gate, reboot orchestration** | **absent** | and see §4.4 — three of the four should stay absent |

### 1.4 The gate itself already passes — on a *current* tree

Doc 09 §The gate passes on hardware records `smoke_gate: PASS`, `manifest: complete: True`
(2026-08-08). **That artifact no longer verifies.** `results/phasef_smoke/*.csv` are schema v1;
`study/schema.py:71` is `SCHEMA_VERSION = 2` since 2026-08-10 (`README.md:92`), and
`results_io.read_rows` rejects both a header mismatch and a version mismatch
(`study/results_io.py:71-113`). Re-run today:

```
[smoke-gate] coarse.csv: unreadable as the current schema -- header does not match schema v2
  results (missing=['context_loads','device_ms','host_cpu_ms','kernel_attaches','sync_ms'])
[smoke-gate] FAIL (4 problem(s))                                            exit 1
```

The machinery is fine — the *recorded evidence* is stale. On a current v2 tree it passes:

```
$ python3 study/smoke_gate.py results/postflip-ladder-w1 --expect {coarse,offload,runlist,fused}.csv
[smoke-gate] PASS (4 CSV(s), each with a passed row)                        exit 0
$ python3 study/manifest.py results/postflip-ladder-w1 --expect … -o …
[manifest] complete: True                                                   exit 0
```

Trees carrying schema v2 today: `results/ladder-v2-w{1,2}`, `results/postflip-ladder-w{1,2}`
(8 CSVs). Everything else on disk is v1 (56 CSVs).

**This is the single most encouraging finding for Phase G: its literal gate sentence — "full profile
run completes with a complete `results_manifest.json`" — is satisfiable today by two existing
scripts. What is missing is the word *profile*.**

### 1.5 The CI side — one target exists, and its comment is wrong

`programming_examples/CMakeLists.txt:169-176` already declares
`check-programming-examples-transformer-layer`, filter `transformer_layer/`. Three problems:

1. **Doc 10's work item 6 cannot be applied as written.** It proposes `add_lit_testsuite` with
   *that same target name* — a duplicate CMake target.
2. **Its proposed filter matches nothing here.** `transformer_layer/.*/run_npu2_compile` requires an
   intermediate directory; the compile test is `transformer_layer/run_npu2_compile_peano.lit`, at
   the top level. The `llms/` filters work because `llms/<model>/` has that level.
   **[inference — verified from directory structure, lit not run]**
3. **The existing target's comment is stale and dangerous.** It reads *"Peano object builds plus a
   symbol check, no NPU dispatch and no HF download. Safe as a PR gate on any runner that has
   Peano"* (`CMakeLists.txt:164-168`). Doc 15 §The transformer-layer suite now needs an NPU
   contradicts it: the suite has needed an NPU since C1. Measured today: **32 `.lit` files, 22 with
   `REQUIRES: … ryzen_ai_npu2`, 1 Peano-only, 9 host-only.** Consistent with the recorded suite
   result of 31 pass / 1 unsupported / 0 fail (`README.md:143-145`) — 32 tests. Anyone who wired
   that target into PR CI on the strength of its comment would get **exactly hazard 3.2**: on an
   NPU-less runner all 22 report UNSUPPORTED and lit exits 0.

No workflow references it: `grep -rn transformer .github/workflows/` returns only unrelated
`transformers` pip lines.

**The genuinely PR-safe subset that exists today is 10 tests, not 1** — `run_npu2_compile_peano.lit`
plus the nine with no `REQUIRES`: `run_block_cache_tests`, `run_blocked_attention_tests`,
`run_ffn_resident_emulation_tests`, `run_reference_tests`, `run_seam_tests`,
`run_study_host_tests`, and `sweep/run_sweep_families_tests`, `sweep/run_sweep_writer_tests`
(`sweep/run_npu2_registry_resolution.lit` has no `REQUIRES` line either — worth checking whether it
should).

`nightlyPerfBenchmark.yml` is the precedent for the opt-in half: dedicated runner label
`amdryzenai5pro340`, `schedule` + `workflow_dispatch`, `concurrency: cancel-in-progress: false`,
`timeout-minutes: 300`, artifact upload. Doc 10's item 7 is largely "copy this file".

---

## 2. What is genuinely missing — a list you could work from

Ordered by value per hour, not by doc 10's numbering.

**M1. No pmode guard on the two latency-sensitive pass/fail gates.** `require_turbo()` is wired into
exactly two places: `study/run_mode.py:397` and `study/component_groups.py:357`. It is wired into
**neither** of the gates that assert on latency:
- `agents/scripts/port-loop/gate-h.sh` leg 4 — decode throughput against
  `throughput-baseline.json` (11.1 tok/s × `floor_fraction` 0.85 = **9.435 tok/s floor**, recorded
  `2026-08-06T18:24:05Z` at sha `d72a2ccf`). `grep -n "turbo\|pmode\|power" gate-h.sh` → nothing.
  That floor was recorded inside the uninterrupted-Turbo window 08-03 → 08-10 (`README.md:333-336`)
  and **the baseline file records `recorded_utc`, `recorded_at_sha`, `n_tokens`, `prompt`,
  `context_len` — and not the pmode.**
- `run_npu2_runlist_gate.lit`'s latency clause, which doc 30 records as having **no margin** and
  being intermittent under suite contention (red once, green twice, same code — `README.md:46`).

At `Default` both fail spuriously. An unattended runner halts, and the halt looks like a compiler
regression. *Effort: ~1–2 h. This is the item to do first regardless of whether Phase G is taken.*

> **`[2026-08-12]` DONE — queue item 13 is closed.** `agents/scripts/port-loop/pmode_guard.py`
> imports `require_turbo` rather than re-deriving it; `gate-h.sh` gains a **leg 0** that refuses
> before the build (and re-checks at leg 4, because a driver reload during legs 1–3 resets the mode
> under the running gate) and **leg 3 was exposed too** — the transformer-layer suite contains the
> runlist gate. `runlist_gate.py` refuses with exit 2 before compiling and prints a banner
> `run_npu2_runlist_gate.lit` now matches, so deleting the guard turns the lit red. The floor file
> carries `npu_power_mode`, the seed script refuses to seed off Turbo and stamps the observed mode,
> and leg 4 refuses a floor/run mismatch. **No number moved and no tolerance widened** — the "no
> margin" half of this finding was already answered on 2026-08-10 by comparing minimums (queue item
> 5), so the clause needed a guard, not a margin. Both directions shown with a stub `xrt-smi` on
> PATH; `pmode_guard.py selftest` is 11/11 in both directions. The shipped floor's own pmode is
> **unrecoverable as an observation** and is recorded `unknown` — flagged, not refused. **M4 below
> is NOT closed by this**: the manifest and schema still record no measurement condition.

**M2. No suite profile, and therefore no derivable expected counts.** `cases.py` declares the
matrix (6 families × 9 ladder points = 54 cases) but nothing consumes it except `select_rows.py`.
There is no `profiles.py`, no `build_job_plan` equivalent, and `manifest.py`'s `--expect` list is
supplied by the caller by hand — which is the right design (the same "the CSV list is the CALLER's"
discipline `compare_roots` adopted, doc 09:428) but leaves the derivation unwritten.

**M3. The manifest validates files, not rows.** `build_manifest` records `exists` / `bytes` /
`mtime_utc` per expected file and `complete = not problems`, where `problems` is
`smoke_gate.check_results_root` = *"≥1 passed row per file"* (`manifest.py:130-150`). Doc 10's gate
asks for "no missing files **or rows** — against counts derived from the profile". A CSV that should
hold 9 rungs and holds 1 is `complete: True` today. Verified on the probe manifest: its keys are
`complete, created_at_utc, expected_files, git, incomplete_reasons, missing_files, repo_root,
results_root, schema_version, study_id, system` — **no row count anywhere**. *Effort: ~30 lines +
fixtures in the existing 231-test suite.*

**M4. ~~Neither the manifest nor the schema records the measurement condition.~~ CLOSED
`[2026-08-12]` by queue item 15, which was blocked on it.**

> `schema.CONDITION_FIELDS` declares a `conditions` block — `npu_power_mode`, how it was obtained
> (`observed` / `probed_at_manifest_build` / `unknown`), the provenance verbatim, and when —
> and `manifest.build_manifest` writes it. **A BLOCK, not a `results` column, and that is the whole
> versioning decision**: a column bumps `SCHEMA_VERSION` to 3, and `results_io.read_rows` rejects
> both a header and a version mismatch, so it would have done to the 16 surviving v2 CSVs exactly
> what §1.4 above records v1 → v2 doing to 56 — including `postflip-ladder-w{1,2}`, the roots the
> comparator is actually pointed at. `schema.py`'s own `RESOURCE_FIELDS` precedent covers it:
> "adding a table is not a version bump". `SCHEMA_VERSION` stays **2**, pinned by a test, and the
> block is deliberately absent from `_FIELDS_BY_TABLE` so nothing can emit it as a CSV.
> A manifest predating the block reads back `unknown` with source `absent`, never a crash and never
> a silent match. `compare_roots` then **refuses** a recorded mismatch and **flags** an unknown.
> Verified: 6/6 readable v2 root pairs byte-identical against the pre-change binary, 16/16 v2 CSVs
> still parse, host suite 231 → **265/265 in 17 modules**. §M3's row counts are a separate item and
> untouched. The remaining conditions this section names — `xrt_version`, the LLVM/mlir-aie/Peano
> pin, `install-xrt`-vs-`build-xrt` — are now a **declaration** in that block rather than a design;
> note while adding them that `compare_roots.compare_manifests` has been diffing a `toolchain` key
> `manifest.py` has never written. Original text follows.

**Neither the manifest nor the schema records the measurement condition.**
`grep -n "turbo\|power_mode\|pmode\|xrt_version\|toolchain\|firmware" schema.py manifest.py` → one
hit, and it is `manifest.py`'s docstring *claiming* "the toolchain and git provenance to reproduce
it" (`manifest.py:10`). Actual provenance is git sha/branch/dirty plus `python`/`platform`/`machine`.
No pmode, no XRT version, no LLVM/mlir-aie/Peano pin, no `install-xrt` vs `build-xrt` resolution.
**This is why doc 32's anomaly cost a day and why the README now carries a prose rule ("08-10's
records are `Default`-conditional, pre-08-10's are Turbo-conditional") — the rule lives in prose
because the data cannot carry it.** *Effort: ~1 h; `run_mode.py` already calls `require_turbo()` and
therefore already knows the answer.*

**M5. The runner can reach one of the six declared families.** `study/run_mode.py:170` hardcodes
`row["workload_variant"] = "encoder_bert"`; `_shape_for` varies **only** `seq_len` and builds the
key as `f"{seq_len}x{emb}_encoder_bert"` (`run_mode.py:145-159`); every mode row in
`opcheck_specs.SPECS` is `emb_dim 768` (`opcheck_specs.py:843,871,916,984` etc.). So
`baseline_768` is reachable and `tinybert_512`, `baseline_1024`, `gpt2_512`, `gpt2_small_768`,
`gpt2_medium_1024` are not. **A "full profile" over the declared matrix is not achievable without
new registry coverage at hidden 512/1024 and a decoder variant** — and the last coverage sweep of
that kind (C4) cost **504 min + 66 min** (`README.md:63`). *This is the decision the queue row is
really asking for: widen the matrix, or scope the profile to what exists and say so in the manifest
the way `component_groups.py` says it in its columns.*

**M6. `run_status="skipped"` exists in the vocabulary and nothing emits it.**
`schema.py:640` — `RUN_STATUSES = ("passed", "failed", "skipped")`. `grep -rn '"skipped"' study/*.py`
excluding tests returns only that declaration. So a structurally-inapplicable rung (`fused` above
1024 — bounded 256..1024, `README.md:82`; anything waiting on item 6c) is recorded as `failed`. The
current gate survives it (it only needs one passed row per CSV) but a **count-based** gate — which
is what doc 10 asks for — cannot distinguish "not applicable" from "broke". *Effort: small, but it
must land before M3, or M3's counts are wrong by construction.*

**M7. No resume, at either granularity.** `run_ladder.walk` rewrites each mode CSV after every rung
(`run_ladder.py:198-200`) so a kill keeps what it measured — good — but a re-run redoes everything.
Doc 10's `--resume-input <its own output CSV>` row-level idempotence does not exist. Note the
registry sweep *does* have the concept (`REUSABLE_STATUSES` at `sweep/registry_sweep.py:177`,
splitting verdicts that describe the candidate from verdicts that describe the machine) — that is
the right model to copy, in-tree.

**M8. Nothing orchestrates.** There is no script that runs *walk → components → resource_usage →
smoke_gate → manifest* as one unit. The recorded Phase F manifest was assembled by hand — its
`results_root` is `/tmp/phasef_results` and its `repo_root` is
`/home/cj/mlir-air/.claude/worktrees/phase-f` (`results/phasef_smoke/manifest.json`).

**M9. `run_lock.py` and `power.py` have no callers.** Both written, both tested, both verified —
`power.py`'s two root-free backends verified live on this host (doc 09:414-426). The runner is the
caller they were written for. Wiring them is ~20 lines each.

**M10. Doc 10's own carry-over item 3 is not applicable yet.** The TTM 1% comparison has no TTM step
to carry it into (`amd-ttm` appears nowhere in the tree), and the empty-mask plot guard belongs to
`plot_selected_component_groups_vs_pattern.py`, which is in the *still-blocked* plotting tier
(queue item 11(b), needs matplotlib/pandas/seaborn, which must not be installed while gates run).
Record it against the plot tier, not against Phase G.

**M11. The CI target: rename, re-filter, re-comment.** §1.5. *Effort: ~1 h + one CI round trip.*

**M12. Host-prerequisite check.** Doc 10's table is partly obsolete for this host: `turbostat` is
**unusable** (`sudo -n turbostat` fails, a password is required — doc 09:298-300) and has been
replaced; `amd-ttm` is unused; `rocm-smi` matters only for `host_comparison`, which is unported.
**So the passwordless-sudo block doc 10 specifies collapses to one binary: `xrt-smi configure`.**
`agents/scripts/doctor.sh` already has the shape for a preflight (`doctor.sh:292` walks a tool list)
and does not check pmode.

---

## 3. The environmental hazards CI would have to encode

For each: what the current state is, and what an unattended runner must do about it.

### 3.1 NPU power mode — silently resets, currently `Default`

`sudo xrt-smi configure --device 0000:64:00.1 --pmode turbo`, needs root, does not persist across
reboot or `amdxdna` reload (`README.md:329-342`, trap 0). At `Default` the verdict rung reads
~2.5–2.7 s against 156 ms at Turbo.

**Already fail-closed in the study path, and provably so.** `run_mode.py` calls `require_turbo()`
*before preparing anything* and returns 2 writing no row (`run_mode.py:392-400`). `run_ladder`
runs each rung as a fresh process, so the check is re-taken per rung — a mid-run driver reload is
caught at the next rung, not at the end. A rung whose child exits without a row is *synthesized* as
a failed row carrying the child's last stderr line (`run_ladder.py:158-176`), so a `Default` walk
produces an all-failed CSV → `smoke_gate` FAIL → `manifest complete: False`. That chain is correct
and needs nothing.

**Not guarded at all in:** `gate-h.sh` leg 4, `run_npu2_runlist_gate.lit`'s latency clause (M1), any
lit correctness gate (harmless), and the recorded floor in `throughput-baseline.json` (M4).

**What CI must encode:** (a) verify pmode before any timed leg and *refuse*, not warn; (b) record
the observed pmode in the manifest and in `throughput-baseline.json`; (c) treat "could not
determine" as a refusal — `require_turbo` already does (`registry_sweep.py:213-219`); (d) never
splice a comparison across a pmode change (README trap 0's closing sentence).

### 3.2 The two CMake flags lost on any clean rebuild

`-DXRT_COREUTIL=/opt/xilinx/xrt/lib/libxrt_coreutil.so -DENABLE_RUN_XRT_TESTS=ON`
(`15-environment-notes.md:7-38`). Without them lit cannot find `xrt-smi`, marks every NPU test
UNSUPPORTED, and **the suite exits 0**. They survive a reconfigure, only a wipe loses them.
`utils/build-mlir-air-using-wheels.sh` sets neither — verified: it sets `XRT_LIB_DIR`,
`XRT_BIN_DIR`, `XRT_INCLUDE_DIR`, `ENABLE_RUN_XRT_TESTS=ON` (lines 117-120) but **no
`XRT_COREUTIL`**, and doc 15 is explicit that `FindXRT` overwrites `XRT_DIR` so only `XRT_COREUTIL`
is the lever.

Currently in effect (read-only check of the existing config, no build touched):
```
build-xrt/programming_examples/lit.site.cfg.py:
  config.xrt_bin_dir = "/opt/xilinx/xrt/bin"
  config.enable_run_xrt_tests = lit.util.pythonize_bool("ON")
```

**Two guards already exist and should be reused rather than rewritten:** `port-loop.sh`'s preflight
refuses to start a hardware phase when `config.xrt_bin_dir` holds no executable `xrt-smi`
(doc 15:37-38), and `pl_assert_gate_ran_hardware` requires `Passed` + `Excluded` to be the *only*
nonzero lit categories and `Passed` to reach the tracked `.lit` count — stated that way
**specifically** to cover this regression (doc 14:335-339). The GitHub workflows are also fine
today (`buildAndTestRyzenAI.yml:126-127`, `nightlyPerfBenchmark.yml:141-142` both pass
`-DENABLE_RUN_XRT_TESTS=ON`) — but neither passes `XRT_COREUTIL`, which matters only on a host with
two XRT installs, as this one has.

**What CI must encode:** assert the two config lines *before* running anything, and assert the lit
category invariant *after*. Both assertions already exist in `lib-guard.sh`; they are just not
available to a CI leg.

### 3.3 This machine is a laptop

Lid close suspends; on battery GNOME suspends after 15 min idle; on AC it does not; background runs
survive logout (`KillUserProcesses=no`) — `15-environment-notes.md:141-142`. Doc 14:27-30 has the
incantation actually used:
`systemd-inhibit --what=handle-lid-switch:sleep:idle setsid nohup … &`.
`grep -rn systemd-inhibit` over `agents/` returns **nothing** — it is documented in two plan docs
and implemented in no script.

**Consequence for Phase G:** this host is not a CI runner and is enrolled as none. The measurement
half of Phase G is a *local operator-invoked script*, not a GitHub workflow. **[inference]**

### 3.4 Three known-red pre-existing lit failures

`15-environment-notes.md:158-176`, reproducible 2/2, all outside every standing gate leg, none in a
directory with a 2026-08 commit:
`llms/llama32_1b_int4/multi_launch_builder/run_o_gemv_ffn_int4_fused_npu2_peano.lit` (Chess-only
header under Peano), `conv2d_14x14/run_npu2_makefile_peano.lit` (output mismatch, NPU1 port),
`matrix_vector_multiplication/bf16/run_npu2_makefile_peano.lit` (`std::runtime_error`, abort 134).
Doc 15's own rule: *"a red whole-tree run with exactly these three is the known baseline, not a
regression"*, and *"if a whole-tree sweep is ever promoted to a gate, these three need owners
first"*.

Also relevant if CI ever runs `check-programming-examples-peano`: doc 15:128-130 records **six**
pre-existing NPU1-only failures there (`matrix_multiplication/{bf16,i16,i8}` passing
`-mllvm -aie-disable-fold-imm`, rejected by the installed llvm-aie).

**What CI must encode:** an explicit expected-failure allowlist keyed by test path, checked in the
same commit as any whole-tree promotion, and a rule that the allowlist shrinking is fine and
growing needs a reason. Note `buildAndTestRyzenAI.yml:152` already carries a
`|| ninja check-programming-examples-peano` retry, which is a *different* mitigation (flake retry)
and does not distinguish the two cases.

### 3.5 Two lock inodes that must not be unified

`/tmp/mlir-air-npu.lock` (the agent/human convention, `flock -x -w 1800`) vs `/tmp/npu.lock` (taken
internally by `KernelCache` and the lit suites — taking it from a wrapper deadlocks them),
`15-environment-notes.md:144-150`. This is documented **at 15 separate call sites** in
`programming_examples/transformer_layer/*.lit` and in the `Makefile:191`, `opcheck.py:118`,
`runlist_gate.py:40`, `sweep/registry_sweep.py:58`, `devq.sh:12-17`.

`devq.sh` takes only `/tmp/mlir-air-npu.lock` (`devq.sh:40`) and **refuses to nest** — a `devq run`
inside a running devq job exits 2 immediately with a message naming the cause, rather than stalling
for `NPU_LOCK_WAIT` seconds (`devq-selftest.sh:121-128`, test 6). `study/run_lock.py`'s docstring's
stated job is saying what it is *not*: pointing it at either device inode would deadlock against
the wrapper that launched the run.

**What CI must encode:** nothing new — take the device through `devq run --class measure` and never
name a lock path. The hazard is real but already fenced in three independent places.

### 3.6 Concurrent `hw_context` ceiling is 32

33 fails with `DRM_IOCTL_AMDXDNA_CREATE_HWCTX err=-2` (`15-environment-notes.md:140`). And there is
a *softer*, more dangerous cousin already measured: `run_ladder.py:32-52` records that in-process
looping made `runlist` fail 2048 **and** 4096 on *different* ELFs with "Failed to load ELF kernel …
contains a kernel symbol matching the provided name" when the symbol was present — process-level
resource exhaustion that "reports as *this mode cannot run at this length*, which is false and is
exactly the kind of wrong conclusion a study publishes."

**What CI must encode:** one process per rung (already the design), `-j1` on any hardware lit suite
(doc 10 is right about this; the existing `check-programming-examples-transformer-layer` target does
**not** pass `-j1` — `CMakeLists.txt:174` — while the four `llms` targets all do). Worth deciding
deliberately: the suite has been run at 24 workers and passed (`README.md:237`, 30/30 in 519.7 s),
but doc 30 records the runlist gate's no-margin latency clause going intermittent *under that
contention*. **[inference]** the right answer is `-j1` for the suite when it is used as a
measurement-adjacent gate and unrestricted when it is used as a correctness gate — which argues for
two targets, not one.

### 3.7 Gates leak artifacts into the source tree as a default outcome

`15-environment-notes.md:108-121`. `make runlist-gate` and the sweep run with the *source* directory
as cwd, so `aircc` and `KernelCache` write `.o`, `air.mlir`, `air.elf`, `air.insts.bin`,
`air_project/` and per-mode `*_cache/` there. Eleven were committed by mistake in `bf69ed69`; the
nine `.o` files were the only tracked object files in the repository. It happened *again* with D2's
6.3 MB `block_cache/`. Doc 15's verdict: **"a new artifact directory is the default outcome of
adding a `KernelCache`-backed gate, not an exception."**

Current state confirms it: `ls programming_examples/transformer_layer/` shows 14 `.o` files,
`air.mlir`, `air.elf`, `air.insts.bin`, `air.xclbin`, five `air_shared_*.xclbin`, `air_project/`,
and eight `*_cache/` directories — all untracked and covered by
`programming_examples/transformer_layer/.gitignore` (which names each cache directory
*individually*). Root `.gitignore:56-66` covers `results/` and `results_unattended_*/` (Phase F item
7, deliberately scoped to result trees rather than `*.csv` — doc 09:546-553).

**What CI must encode:** (a) `git status --porcelain` must be clean after a profile run, asserted by
the runner, not by a human; (b) every new cache directory joins `.gitignore` *and* the `clean`
target in the same commit; (c) a results root is ~2.4 GB (doc 09:335) and Phase G would produce one
per run on a laptop — retention policy is part of the design, not an afterthought.

---

## 4. Effort, risk, and the first increment

### 4.1 Effort

| item | scope | estimate |
|---|---|---|
| **M1** pmode guard on `gate-h.sh` leg 4 + runlist gate latency clause; pmode into `throughput-baseline.json` | ~60 lines, 2 files, both already tested in both directions elsewhere | **1–2 h** |
| **M11** CI target rename/re-filter/re-comment + wire the 10 PR-safe tests into a workflow | 1 CMake block, 1 workflow edit | **1 h + one CI round trip** |
| **G0** profile + runner + manifest counts (M2, M3, ~~M4~~ **done**, M6, M8, M9) | ~400–600 lines across `study/profiles.py`, `study/run_profile.py`, edits to `manifest.py`; host-only tests into the existing suite (231 → **265** after M4) | **1 session code + 1 devq measure window** |
| **M7** resume, copying `REUSABLE_STATUSES`' verdict split from the registry sweep | ~150 lines | **half a session** |
| **M5** widen the matrix past `baseline_768` | new registry coverage at hidden 512/1024 + a decoder variant | **unbounded — C4's precedent is 504 + 66 min of gate time alone** |
| doc 10's crontab / TTM / thermal / reboot orchestration | — | **recommend NOT doing — see §4.4** |

### 4.2 Risks

- **HIGH — the profile is not reachable (M5).** "Full profile" over the declared 6×9 matrix is
  currently impossible. Either the profile is scoped to what exists and *says so in the manifest*,
  or Phase G silently becomes a Phase-C-sized coverage sweep. The former is in this repo's
  established idiom (`component_groups.py` reports `0/12` rather than presenting a mode total under
  a group label).
- **HIGH — unattended measurement is exactly where this project has published wrong claims.**
  `README.md:507-523` lists three, two of them measurement-condition failures (a 1.55× inflation
  from host work running alongside; a "5.9% improvement" that was three fresh runs against one
  stale number). An unattended runner multiplies the opportunity. Mitigation is M4: put the
  conditions **in the data**, not in prose. **`[2026-08-12]` DONE for the pmode** — the manifest
  carries it and `compare_roots` refuses a splice across it — so the risk narrows to the conditions
  still unrecorded (`xrt_version`, the toolchain pin), each now a declaration in
  `schema.CONDITION_FIELDS`.
- **MEDIUM — structural holes are not failures (M6).** `fused` above 1024 and anything gated on
  item 6c will populate a count-based manifest with `failed` rows that are not regressions.
- **MEDIUM — 2.4 GB per results root on a laptop**, times however often CI runs.
- **MEDIUM — the install/build tree split.** `install-xrt/bin/air-opt` is 2026-08-07 against
  `build-xrt`'s 2026-08-11 (`15:195-203`), so probes and models resolve a four-day-old compiler
  while lit suites do not. A Phase G runner must state which tree it tested — and check it with
  `ls -l`, **never `cmp`** (RUNPATH rewrite makes bytes always differ).
- **LOW — crontab/reboot on a shared personal machine.** Doc 10 flags it itself
  (`10:147-148`).

### 4.3 The first increment — smallest thing that delivers value

**Do M1 first, on its own, whether or not Phase G is taken.** It is one to two hours, it is
independent of everything else, and it removes the single failure mode most likely to waste an
overnight run: a latency gate failing for a reason that is not a code change. It also makes the
`Default`-pmode state the machine is in *right now* visible to the gates rather than only to the
study tier.

**Then G0, "one profile, one command, one manifest":**

1. `study/profiles.py` — a table of named profiles, each a tuple of `(mode, family, seq)` rungs plus
   the expected-CSV list *derived from it*. Three to start, mirroring iron's shape without its
   counts (doc 10 §Job counts is explicit that iron's 888/834/21/3 must not become acceptance
   criteria): `smoke` = 4 modes × 1 length; `ladder` = 4 modes × {512,1024,2048,4096}; `full` =
   whatever the matrix actually reaches today, with the unreachable families recorded as such.
2. `study/run_profile.py` — takes a profile name and: calls `require_turbo()` up front (free per
   rung already), takes `run_lock` on the results root (**its first caller**), walks via the
   existing `run_ladder.walk`, then runs `smoke_gate` + `manifest` with the expected list derived
   from the profile, writing `results_manifest.json` into the root. Invoked as
   `agents/scripts/devq.sh run --class measure -- python3 study/run_profile.py --profile …`,
   wrapped in `systemd-inhibit`.
3. `manifest.py` gains `expected_rows` per file, an observed-vs-expected row count, and a
   `conditions` block (`npu_power_mode`, XRT version, which toolchain tree resolved). `complete`
   becomes files-present **and** every file measured **and** every count met.
4. Host-only tests for all three into the existing suite; the pinned count moves 231 → N, verified
   in the shrinking direction as `run_study_host_tests.lit` requires.

This delivers doc 10's literal gate sentence at the smallest honest scope, uses only modules that
already exist and are already tested, needs no reboot/crontab/TTM machinery, and is gateable in one
devq measure window. It also finally gives `run_lock.py` and `power.py` the caller they were written
for.

**In parallel and independently, M11** — the CI leg. It is an hour, it is not device work, and it
converts 10 already-green tests from "gate on nothing in CI" to "gate on every PR".

### 4.4 What to deliberately drop from doc 10

Recommend recording these as *decided against* rather than leaving them open:

- **The `@reboot` crontab hook.** Doc 10 itself flags installing it on a shared machine, and the
  footgun it documents (running `start` under `sudo` puts it in root's crontab) is real. A
  `systemd-inhibit` + `setsid nohup` launch, which doc 14 already uses in anger, covers the actual
  need on this host: surviving lid, idle and logout.
- **TTM page-limit transitions.** `amd-ttm` appears nowhere in the tree. iron's 26 GB override
  existed for six 16384-token **iGPU** jobs, i.e. `host_comparison`, which is unported and needs a
  ROCm torch wheel that conflicts with the pinned CPU-only index (doc 09:264-265). Carry doc 10's
  1%-band fix as a note *attached to that study*, not to Phase G.
- **Thermal gating via `sensors`/`rocm-smi`.** No artifact anywhere in this tree shows thermal
  throttling affecting any recorded number. Adding an ungrounded gate to an unattended runner adds a
  halt condition without adding evidence.
- **`turbostat` power.** Cannot run here (`sudo -n` fails, doc 09:298-300) and `study/power.py`
  already replaced it with two verified root-free sysfs backends.

Dropping those four removes doc 10's entire passwordless-sudo block except `xrt-smi configure`, and
removes the reboot-loop failure class that halted iron's queue at job 885 of 888.

---

## 5. What a full profile run costs in wall clock

### 5.1 The artifact-backed components

| what | duration | artifact |
|---|---|---|
| 8 cold rungs (4 modes × {512,1024}) | **631 s ≈ 10.5 min** | `agents/.state/devq/jobs/job-000224.log`, per-rung walls 98/102/29/30/55/57/128/132 s |
| the same 8 rungs warm, immediately after | **32 s** | same log, 5/5/2/3/6/7/2/2 s |
| both walks together (job `postflip-ladder`) | **11.0 min** | `agents/.state/devq/jobs/job-000224.meta` |
| 16-rung walk (4 modes × 512/1024/2048/4096), warm | **~90 s of device time** | `25-first-study-result-sequence-ladder.md:153` |
| the same walk **cold** | **~45 min** | same line |
| full transformer-layer lit suite (32 tests) | **8.1 / 8.3 / 8.5 / 8.7 / 8.9 / 8.6 min** over six recorded runs | devq metas `job-000177/132/135/212/241/248` |
| ten-model `make verify` | **63.2 min** | devq meta `job-000222` |
| suite + ten models in one job | **73.6 / 73.6 / 73.8 min** | devq metas `job-000081/085/075` |
| the registry coverage sweep (C4) | **504 min + 66 min re-run** | `README.md:63` |
| Phase B (runlist spike + ten models) | **362 min** | `README.md:59` |
| the study host suite | **~0.4 s** | `09-phase-f-study-harness.md:538` |

Compilation, not measurement, dominates: 631 s cold against 32 s warm for the identical 8 rungs is
a **~20× swing** on the same hardware in the same job. Doc 10's "wall clock swings by a factor of
four depending on how warm `build/` is" understates it at this granularity.

### 5.2 The estimate **[inference — extrapolation from the above; no full profile has ever run]**

Per-rung cold cost from doc 25's 16-rung figure: 45 min / 16 ≈ **2.8 min/rung** averaged over
512–4096. Job 224's cold rungs at 512/1024 (29–132 s) bracket that consistently.

| profile | rungs | cold | warm |
|---|---|---|---|
| `smoke` (4 modes × 1 length) | 4 | ~5 min | ~15 s |
| `ladder` (4 modes × 4 lengths), walked twice | 32 | ~45 min + ~2 min | ~3 min |
| one family × 9 ladder points × 4 modes, walked twice | 72 | **~1.7–2 h** | ~5 min |
| six families × 9 × 4 (each family a fresh ELF cache) | 216 | **~10 h** | — |
| + the block/registry sweep at C4's precedent | — | **+8.4 h** | — |
| **"full" as doc 10 imagines it** | — | **~18–20 h cold; ~2–4 h with `build/` warm** | |

That brackets iron's recorded 11 h – 2 days for its 888-job `full`, which is the sanity check.

### 5.3 What that bounds

- **A full profile monopolizes the only NPU for its whole duration.** devq's `measure` class is an
  absolute barrier — later builds are not admitted until it has run (`devq.sh:8-10`). So a full
  profile is not "a nightly job"; it is "nobody else uses this machine today".
- **Realistic cadence:** `smoke` per change (minutes); the 32-test lit suite nightly (~9 min);
  `ladder` weekly or on demand (~45 min cold); `full` on explicit `workflow_dispatch`-equivalent
  only, monthly or before a write-up.
- **The nightly slot is already taken and is on a different machine.**
  `nightlyPerfBenchmark.yml` runs at cron `17 4 * * *` on runner `amdryzenai5pro340` (a Krackan
  Point NPU2), not this Strix laptop, with `timeout-minutes: 300`. Transformer-layer study results
  are host- and pmode-specific, so they cannot ride that workflow. **[inference]**
- **Two walks is the standing rule, not an option** — README trap 1: "A single walk would have
  published a crossover that a second walk refuted, which is the J3 failure repeating — so walk
  anything twice." Every estimate above already includes it, and the second walk is cheap (warm).

---

## 6. Bottom line for the decision in queue row 12

Phase G is **not** a 2,494-line port. Its device scheduling, power-mode enforcement, results
manifest, completeness gate, per-file lock, power sampling, case matrix and row selection all exist
and are tested; its crontab/TTM/thermal/turbostat quarter should be dropped on measured grounds;
and what remains is a profile table, a runner that walks it, and row counts in the manifest.

The honest blocker is not effort — it is **M5**: the case matrix the study declares is six times
larger than the one the runner can reach. Phase G should be taken with that scoped down and
recorded, or not taken.

And **M1 is worth doing this week regardless**: two latency gates that a CI leg would run assert on
throughput with no pmode guard, on a machine whose pmode is `Default` right now.

---

# `[2026-08-12]` G1 — resume, doc 10 item 5, and the coverage sweep measured

Phase G's three open items, taken in the order queue row 12 names them. Worktree branch
`worktree-agent-a0bab073cb6414184` off the tip `39a08a8b`; commit `869b8684`. NPU pmode **Turbo**,
verified before any device work (`xrt-smi examine -r platform` → `Power Mode : Turbo`).

**Headline: §M5 was wrong, and it was wrong in the direction that mis-sized a phase.** The registry
coverage two of the five unreachable families were said to need has been in the tree since
2026-08-07. `tinybert_512` walked end to end in **301 s** against §4.1's estimate of **unbounded —
504 + 66 min**. Full costing in [50](50-coverage-sweep-costing.md); the correction is §M5 below.

## M7 — resume: CLOSED

`study/resume.py` (new), `run_ladder.walk(..., reuse=, on_rung=)`, `manifest.build_manifest(walk=)`,
`schema`'s `WALK`/`SESSION`/`SESSION_RUNG` blocks, `run_profile --resume`.

**What it guarantees.** A rung with a `passed` row is not re-run. Every profile rung appears in its
CSV exactly once. The completeness verdict is **unchanged by resuming** — `manifest`'s three
row-count clauses read the CSVs and know nothing about sessions, so a resumed short walk is
incomplete exactly as a fresh one is. A rung the plan declared reused whose final row does not hash
to the prior row's digest is a **`RESUME DEFECT`** and makes the run incomplete. A row on disk that
no session claims is `rungs_unattributed` and a problem. A splice across power modes is refused; one
across a toolchain or a git sha is flagged — `compare_roots`' refuse-known / flag-unknown split,
applied *within* one root instead of between two.

**What it cannot.** It cannot make a spliced walk one measurement — rows from two sessions are two
populations on a laptop whose thermal and load state nobody recorded; the block says so and does not
fix it. It cannot resume mid-rung: the granularity is one `run_mode` child process. It cannot detect
a *distorted* measurement — a rung measured beside another job's dispatches produces a valid row with
a good digest, and contention stays devq's problem. And attribution is keyed by
`(execution_mode, seq_len)` **outside** the CSV, because a `session_id` column would bump
`SCHEMA_VERSION` to 3 and take every surviving v2 root out of every reader (item 15's decision,
unchanged and **not** revisited). The digest is what buys back the confidence key-based attribution
loses.

**The design decision worth arguing.** Doc 10 §Resume idempotence points at the registry sweep's
`REUSABLE_STATUSES`, which reuses `failed_build`/`failed_precision`/`failed_tier`. Here **only
`passed` is reused**. The sweep can reuse a failure because a registry row is keyed by a
`MEASUREMENT_CONTRACT` hash that changes when the meaning does; a results CSV has no such key, and
between two sessions the tree can change — which is *why* the walk was interrupted often enough to
need resume. A retained failure is a claim about code that may no longer be there. A skip is
re-derived every session for the mirror reason: it is the profile's current claim about what a mode
supports, not a recorded observation, so freezing one would leave a superseded rule in force.

**The trap that shaped it.** Bookkeeping agrees with itself whatever the walk did. G0's two closed
defects were both checks that could not fail, and a resume's natural implementation is a third. So
the ledger is *evidence*: written per rung by the walker (not by the plan), carrying a row digest,
and re-hashed against the files afterwards. `profile_run.json`'s `rungs_by_source` is counted off the
ledger for the same reason.

Also closed in passing: **`run_profile`'s gate never passed `conditions=`**, so every profile manifest
recorded `npu_power_mode: unknown` — on a run that had just *refused to start* unless the mode was
turbo. The rule is "never stamp a condition you did not observe"; this was its inverse, observing and
discarding, which is the worse half because the artifact then reads as though nobody could tell.

## Doc 10 work item 5 — SPLIT: the table dropped with evidence, the requirement met by a check

Item 5 asks for "the prerequisites and recovery sections of the example README", because "the runner
shells out to all of them, and a missing tool fails mid-suite rather than at start **unless
checked**."

**The prerequisites TABLE is dropped**, joining the other four. Of its six tools, `amd-ttm`,
`turbostat`, `sensors`, `rocm-smi` and `crontab` are already recorded as dropped in doc 10
§Deliberately dropped with a measurement behind each; the sixth, `xrt-smi`, is only ever *read* here,
and `require_turbo` already refuses when it is missing or unparsable. A table of five tools nothing
invokes is a false claim about what the runner does.

**The requirement is its last four words, and a README paragraph cannot fail.** This project's own
record is of prose rules that were true when written and silently stopped being — README trap 0 lived
in prose until the conditions block moved it into the artifact. So
`run_profile.environment_problems()` refuses at start, and what it refuses is not a binary at all: it
is the two Python modules a bare devq shell lacks. `pyxrt` is not added by `env_setup.sh`, so a job
that sources it and stops **compiles every kernel and then dies at the first dispatch** with a
`ModuleNotFoundError` that reads like a model regression; `ml_dtypes` fails at the first builder
import. The third clause refuses a working directory that is not the example's, because aircc and
`KernelCache` write relative to cwd and only that directory's `.gitignore` covers the debris (doc 15;
eleven artifacts were committed by mistake once).

**The recovery half is written**, because it carries information that did not exist before resume did:
README §"Running a profile: invocation and recovery" — the one command, what each of the four
artifacts answers, how to resume, why failed rungs are re-run, and why a populated root is refused
with no override.

## M5 — CORRECTED. The blocker was not coverage, and the test that guarded it read the wrong file

Full account in [50](50-coverage-sweep-costing.md). In brief:

- `kernel_registry/details/GEMM_bf16_in_bf16_out.json` holds **36 of 36** projection triples at each
  of hidden 512, 768, 1024 — landed 2026-08-07, 69 → 103 → 136 rows. Verified twice: by triple, and
  by **resolution through the owning builder** (which is the stronger check, since `qkv_proj` pins
  `fused-cast` and `offload`/`runlist` re-resolve through `drain`). 36/36 at 512 and 768,
  ~~**35/36 at 1024** — `2048x1024x3072` has no `drain` row, so two modes fail at one ladder point.~~
  **`[2026-08-12]` 36/36 at 1024 too — see the correction below and [50](50-coverage-sweep-costing.md)
  §7.** The probe applied `offload`'s `drain` re-resolution to the **qkv** shape; `offload` resolves
  `(seq, h, h)`, `(seq, h, 4h)` and `(seq, 4h, h)` and never a `3h` shape.
- The blocker was `run_mode._shape_for` overriding `seq_len` and not the width. It is a parameter now.
  **Three families are reachable**; the three decoders are **refused by name**.
- `tinybert_512` end to end: devq **304**, `measure`, Turbo, cold, **301 s**, 4/4 passed, ten clean
  boundaries per mode, `atol_required` 5.2–6.0e-2 against the inherited 1e-1 ceiling (1.66–1.92×
  margin). The resume leg carried all four rungs in **0 s** and the ledger audit found no problems;
  a populated root without `--resume` refused with exit 2. **No latency is quoted as a result** —
  one walk is not a result (README trap 1).
- **The lesson is about this project's strongest habit.** Re-deriving a claim from source is what has
  repeatedly saved this study, and it failed in the one way it can: a re-derivation is only as good
  as its choice of source. `test_profiles.py`'s ast walk over `opcheck_specs.py` was mechanically
  perfect and asserted something still true; the *inference* from it was false. It now reads the
  registry, asserts the converse, and asserts each declared method gap is still open.

~~`profiles.KNOWN_REGISTRY_GAPS` records the `drain` hole as a **failure to be run**, not a skip —
`cases.py`'s rule that pre-declaring a failure is how a matrix stops being a measurement — and a test
asserts it is still a gap, so sweeping it makes the warning fail rather than linger.~~

> **`[2026-08-12]` CORRECTED — THE HOLE WAS NOT A GAP, AND THE TEST COULD NOT HAVE SAID SO.**
> Full account in [50](50-coverage-sweep-costing.md) §7. `2048x1024x3072` really has no `drain`
> row, and the consequence drawn from it was false: **`offload` and `runlist` never resolve a `3h`
> shape**. Both chains are `(seq, h, h)`, `(seq, h, 4h)`, `(seq, 4h, h)`; the only consumer of
> `(seq, h, 3h)` is `qkv_proj`, which pins `fused-cast`, which is present. `baseline_1024` was
> **36/36, not 35/36**, and it is now walked (devq **307**). The guarding test asserted only "the
> method is still missing" — true of every method nobody asks for — so it would have passed forever.
> It now also requires a declared gap to name a `(triple, method)` some consumer **pins**, and the
> deleted entry is its negative control. `KNOWN_REGISTRY_GAPS` is empty.
>
> The shape of the error is this section's own lesson applied one level in: G1 corrected M5 by
> reading the **registry** instead of `opcheck_specs.py`, then modelled `offload`'s `drain` pin
> against the qkv shape `offload` never resolves. *A re-derivation is only as good as its choice of
> source* — and that applies to the fix as much as to the bug.

## The negative controls

Every check added here is demonstrated failing on a deliberately broken input, and the input is named
in `study/test_resume.py`'s docstring beside the clause it drives. The load-bearing one is a walker
that **ignores `reuse` and re-measures**: the plan, the ledger and the run report would all still say
"reused", and only re-hashing the row afterwards catches it.

## Counts

Host suite **357 → 409 in 20 modules**, pinned in `run_study_host_tests.lit` and verified in both
directions (one test renamed out of discovery → 395/395, refused; the module hidden → 372/372 in 19,
refused). `SCHEMA_VERSION` **stays 2** and every recorded v2 CSV still reads.

## Still open after G1

- **`baseline_1024` has not been walked.** It is reachable; two of 36 ladder rungs will fail on the
  one missing `drain` method until somebody sweeps that single shape (~2 min) or accepts the gap.
- **The three decoder families**, unchanged and correctly sized: a D2-class layer-graph integration
  per mode, no sweep. `gpt2_small_768` is cheapest — its width is already the default.
- **Two walks into two roots**, which nothing here has done. Every latency in this section is a
  single walk and is quoted as evidence that a mode *built*, never as a ranking.
- **The `tree_dirt_after_run` check cannot distinguish a leak from an author.** It reports every
  tracked-tree change, so on a dirty working tree it names the operator's edits. Job 304's report
  listed eleven modified files, all mine, and **zero untracked** — which is the half that would have
  shown a leak. Worth narrowing to untracked paths, or worth saying in the field's own description.
