# 09 — Phase F: The Study Harness

The seven measurement studies. This is the largest phase by line count, and the phase where
most of the convention refactoring lands.

> **`[2026-08-11]` There is no unmerged Phase F worktree. The README's row saying otherwise was
> stale.** It read "in progress on `exper/phase-f-study-harness` (a worktree, unmerged)", and
> anyone acting on that would go looking for work that is already in front of them. Verified:
> `exper/phase-f-study-harness` (tip `4775722e`) is a **full ancestor** of
> `exper/transformer-layer-execution-studies`, **0 unmerged commits**, so every Phase F commit to
> date is on the experiment branch. Ignore the `.claude/worktrees/phase-f` checkout entirely and
> work from the experiment branch. The README row is corrected in the same commit as this note.

## The seven studies

| Study | iron run.py | Depends on |
|---|---|---|
| `block` | 1313 | NPU; per-operator candidate sweep. Also becomes the registry sweep tool (Phase C). |
| `end_to_end` | 1048 (+ `modes.py` 2336) | NPU; all four execution strategies |
| `memory_tile_staging` | 558 | `block/results.csv` via `--reference-input` |
| `resource_usage` | 1623 (+ `analysis.py` 299) | Build tree only; **no NPU** |
| `host_comparison` | 1784 | ROCm iGPU + the end-to-end CSV |
| `memcpy_bandwidth` | 586 | NPU |
| `roofline` | 1772 | CSVs only; **no NPU** |

Run `block` before `memory_tile_staging` — the latter reads the former's CSV.

## `[2026-08-05]` Carried in from Phase E: `fused`'s norm tail loses partial-sum staging

Phase E measured all four modes correct, but one of them pays a numerical cost that this phase
should treat as a measurement subject rather than a footnote.

**The GEMMs are staged correctly in every mode.** `matrix_multiplication/bf16_in_bf16_out/run.py:69`
keeps an f32 accumulator across the whole K loop with a single epilogue cast, and
`pattern/fused/fused.py` derives each GEMM's f32 scratch argument from its spec's
`needs_f32_scratch` rather than hardcoding it, so a registry method change adds or drops the scratch
automatically. Measured `q`/`k`/`v` at `mean_rel_L1` 9.7e-3 against the registry's 9.3e-3 standard.

**`fused`'s normalization tail is not staged.** `fused_tail` decomposes the fused `addnorm` into
`elementwise_add` → `layer_norm` → `elementwise_mul` and **rounds to bf16 between each launch**,
where the fused operator keeps those intermediates in higher precision inside one kernel. The cost
stacks across the modes and is measured, not estimated:

```
block   1.688e-2      runlist 1.732e-2      fused 1.784e-2      (mean_rel_L1, whole layer)
```

`fused` lands at `atol_required` 7.896e-2 against the hard `1e-1` ceiling — a 1.27x margin, the
thinnest of the four modes.

`[2026-08-07]` Refreshed from the J7b gate run, after J7a moved `layer_norm_rows` to f32 two-pass
statistics. Previously `runlist` 1.755e-2 and `fused` 1.806e-2 at `atol_required` 7.572e-2 (1.32x).
`block` is unchanged — it dispatches the fused `addnorm`, which deliberately keeps one-pass
statistics, not `build_layer_norm_module`. Note the mean improved while `fused`'s margin tightened,
1.32x → 1.27x: an average and a worst-element statistic move independently, and this mode has the
least headroom either way.

The cause is a layout limitation rather than a shortcut. `build_addnorm_module` caps a launch at 104
rows of 768, and a band at a nonzero row offset cannot be routed into a launch's args clause:
`memref.cast` will not cast an offset subview back to the identity layout the signature declares.
So the stitched module cannot reuse the banded fused operator, and streaming the decomposed
builders is the only form available today.

### And the larger one: the four modes do not all run attention in the same place

| mode | attention | normalization |
|---|---|---|
| `offload` | **host torch**, blocked (`pattern/blocked_attention.py`) | host torch |
| `runlist` | **host torch**, blocked — the same module, shared | decomposed, banded at 64 rows |
| `coarse` | **device** FlashAttention (`mha_out_proj`) | fused `addnorm`, banded |
| `fused` | **device** FlashAttention (`mha_out_proj`) | decomposed, streamed |

Both host-attention choices are forced by the same constraint that reshaped `offload`: the attention
interior needs `4096x64x4096` and `4096x4096x64`, which no registry row holds and no `--family` can
sweep. And `08d` deliberately had `runlist` share `offload`'s blocking so the two would block
attention identically. So neither is a defect.

But it means **a mode-versus-mode comparison varies more than the dispatch boundary it claims to
isolate.** `offload` against `coarse` differs in dispatch granularity *and* in whether attention ran
on the NPU; so does `runlist` against `coarse`. That is the same confound that made E4's first
`runlist` structure uninterpretable — two variables moving at once — and it is load-bearing here
because attention is the dominant cost in the layer.

Three things for this phase:

- **Do not let the case matrix compare modes on latency alone.** A row reporting only time is
  comparing different arithmetic *and* different hardware utilization. The schema already carries
  per-boundary data and `attention_path`; surface both next to any mode-versus-mode number.
- **Consider a fifth measured point**: `coarse` or `fused` with host attention, or the reverse, to
  separate "attention on device" from "dispatch boundary". One extra column in the case matrix buys
  the ability to attribute a difference to the thing the taxonomy names.
- **The norm fix, if it is worth one, is an offset-subview path** for launch arguments, which would
  let a stitched module reuse the fused `addnorm` at a nonzero row offset. That is a compiler change
  in the launch-args legalization, not a study change, so it is scoped here only as a recorded
  dependency — see [08 §Outcome](08-phase-e-execution-strategies.md).

## The port tier that carries over

~19,000 lines have **zero** `iron` imports: `run_lock.py`, `plot_families.py`,
`npu_runtime_checks.py`, `results_manifest.py`, `compare_results_roots.py` (+ its test),
`regenerate_plots.py`, `unattended_smoke_job.py`, `end_to_end/power.py`, every `plot_*.py`,
every `cases.py` and `select.py`, `roofline/run.py` and its test.

`[Codex]` **These port by structure, not by import path alone.** MLIR-AIR has no
`RESULTS_CSV_FIELDNAMES`, no `execution_mode`, no `run_status`; its benchmark pipeline emits
heterogeneous JSON. Copying column names does not define what they mean under AIR's timing and
synchronization model.

So the schema work in [03-measurement-model.md](03-measurement-model.md) is a prerequisite for
this tier, not a detail of it:

- A **versioned MLIR-AIR study schema** with a `schema_version` column and written field
  semantics.
- An **explicit adapter** where byte-level comparison against iron trees is wanted, rather than
  pretending the schemas are identical.
- `fused_elf` as a new `execution_mode` *value*, not a new column.

## The modules that need real retargeting

Only five:

| Module | Lines | Why | `[2026-08-11]` state |
|---|---|---|---|
| `end_to_end/modes.py` | 2336 | The `_build_operator()` dispatch point mapping `execution_mode` → strategy. Shape survives; every `_benchmark_<op>` body does not. | ~~pending~~ **done, under other names** |
| `block/run.py` | 1313 | Per-operator candidate sweep against real operators. | ~~pending~~ **done, under other names** |
| `end_to_end/run_selected_component_aggregates.py` | 1578 | | ~~pending~~ **done** — `study/component_groups.py`, an honest partial |
| `memcpy_bandwidth/run.py` | 586 | ~~Lazy-imports the memcopy operator inside a function; small surface.~~ **The surface is small and the OPERATOR does not exist** | **open** — see below |
| `resource_usage/{run,analysis}.py` | 1623 + 299 | See the open issue below. | ~~pending~~ **done** — `study/resource_usage.py` |

`host_comparison/run.py` (1784) only imports the torch reference and joins on the end-to-end
CSV, so it is nearly free — swap the reference and the join columns.

### `[2026-08-11]` Item 4: four of the five are retargeted, and two of those were already done

Two of the five were retargeted before this item was ever opened, under different names, which
is why nothing was left to do for them. Recorded here so a reader does not go looking:

- **`end_to_end/modes.py` → `study/run_mode.py` + `opcheck_specs.SPECS` + `pattern/{coarse,offload,runlist,fused}/`.**
  The `_build_operator()` switch this row names is `run_mode._spec_for()`, which resolves a mode
  through the SPECS catalogue rather than a dispatch table; convention rule 5's "one module per
  execution mode plus a thin registry" is the `pattern/` package, and none of the four modules is
  over 1,100 lines against `modes.py`'s 2,336.
- **`block/run.py` → `sweep/registry_sweep.py` + `sweep_families.py` / `sweep_measure.py` /
  `sweep_report.py` / `registry_writer.py`.** This row's own note says `block` "also becomes the
  registry sweep tool (Phase C)", and that is what shipped — with a per-candidate subprocess
  timeout iron has no equivalent for, which is why iron's FFN sweep never converged.

The two genuinely new ones:

- **`study/resource_usage.py`** consumes the artifact item 2 pinned and nothing had read. It
  keeps iron's three regexes verbatim — only the FILE moved — and adds `core_to_core_flows`,
  which makes doc 03's AIE-role-style axis measurable per design instead of per hand-written
  gate. **Verified against real artifacts in both directions** (devq job 238, build class, log
  `agents/.state/devq/jobs/job-000238.log`): a fresh norm-tail compile reads **24 cores, 40
  flows, 16 core→core → space-multiplexed**, matching [23 §5](23-rules-and-open-items.md)'s
  independent pin of 2 × `herd_x` at `herd_x = 8` and the 40/24 counts `aircc_artifacts.py`
  recorded when item 2 closed; the `transformer_layer` project reads **0/116 → time-multiplexed**,
  as does every other artifact on disk — exactly what doc 03 predicts, since J7a and J7b are the
  only space-multiplexed designs in the tree. It also fixes one arithmetic defect in iron's
  version: the combined `shim_dma_channels_used` set is keyed on the channel NUMBER, so a tile
  using S2MM 0 and MM2S 0 reads one channel of four when two distinct hardware channels are
  busy. Keyed on `(direction, channel)` here; the per-direction columns are unchanged.
- **`study/component_groups.py`** is deliberately a PARTIAL, and says so in its columns rather
  than in a footnote. See §The component aggregate is honestly partial below.

### `[2026-08-11]` `memcpy_bandwidth` is the one still open, and this row mis-sized it

"Lazy-imports the memcopy operator inside a function; small surface" is true of the RUNNER and
misleading about the work. iron's runner calls `iron.operators.mem_copy.op.AIEMemCopy`, and
**MLIR-AIR has no equivalent operator.** The two nearest examples,
`programming_examples/passthrough/passthrough_{dma,channel}.py`, are both `herd sizes=[1, 1]` —
one tile. So this is a new device design, not a runner port, and it should be scoped as one.

Worse for a straight port, **one of iron's four case axes does not exist as an input here.** Its
matrix is `(size, num_cores, num_channels, bypass)`, and in AIR the shim channel count is not
something a builder asks for — it is what routing produces. Measured: the norm-tail compile above
allocates **17 shim channels over 8 tiles** without anything requesting a number.

So the retarget is a re-shaping rather than a translation, and the pieces for it are already here:

- `num_cores` → herd size, over the `channel_examples/channel_size` shape (a tiled multi-worker
  copy with a per-worker channel pair), which is the multi-core form `passthrough_*` lacks.
- `bypass` → whether the herd body runs a load/store loop over the L1 tile or only stages it.
- `num_channels` → **not an axis. It becomes an observed column**, which
  `study/resource_usage.py` now reports (`shim_dma_channels_used`,
  `shim_{s2mm,mm2s}_channel_utilization`) off the same routed design.

That last point is the reason to do this study at all under AIR: the question stops being "how
does bandwidth vary as I ask for more channels" and becomes "what does the compiler allocate, and
what bandwidth does that reach" — which is a compiler result, not a configuration sweep.

## Open issue: the resource-usage artifact

`[Codex]` `resource_usage/analysis.py` regex-parses `aie.core(%tile_r_c)`,
`aie.buffer(%tile…) : memref<…>` and `aie.*dma_allocation @…(%shim_noc_tile…, S2MM|MM2S, ch)`
out of iron's `input_physical.mlir`, normalizing against `AIE_TILE_LOCAL_MEMORY_BYTES=65536`,
`MEM_TILE_LOCAL_MEMORY_BYTES=524288` and `SHIM_DMA_CHANNELS_PER_DIRECTION=2`.

AIR lowers to the same `aie` dialect, but **"same dialect" does not guarantee a stable
equivalent artifact under `aircc`**. Identify the concrete post-`aircc` artifact and pin its
path before this phase starts. If none is stable, emit one deliberately rather than parsing an
incidental intermediate.

> **`[2026-08-11]` Closed on both halves.** `study/aircc_artifacts.py` pinned the artifact
> (`air_project/aie.air.mlir`) when item 2 closed, and `study/resource_usage.py` now consumes it
> — until today nothing did, so the resolution was recorded and untested end to end. The
> normalization constants carry over unchanged because they are AIE2P properties, and the parse
> is verified on real compiles rather than on fixtures alone (job 238 above). One property of the
> artifact that the port depends on is now exercised rather than asserted: the norm-tail compile
> **failed at the core-ELF link** (no kernel object, `chesslinked_{0}.ll` edge failed) and the
> routed design was still written, which is what makes resource usage readable without Peano.

## Where the convention refactoring lands

Rules 5, 7, 8, 10 and 11 all apply here. Budget for it — "ports by structure" is not "ports
quickly".

- **Rule 5 (module size)** — split `modes.py` (2336) into one module per execution mode plus a
  thin registry.
- **Rule 7 (dual naming)** — collapse `hybrid` / "coarse runlist" / `pattern_label` down to one
  mapping in the schema module.
- **Rule 8 (redundancy)** — the re-export `reference.py` shims are already dropped in Phase E.
- **Rule 10 (measurement plumbing)** — route through `Profiler` (`shared/infra/cache.py`) and
  `llms/bench/extract_perf.py` rather than iron's parallel `@pytest.mark.metrics` stdout
  scraper.
- **Rule 11 (test discovery)** — iron's `pytest.ini` sets `python_files = test.py`, silently
  excluding its own `study/test_*.py` from directory collection. Normalize on standard
  discovery.

## Testing convention

`[Codex]` Programming examples use lit for device gates, and there is no repository-wide pytest
runner or discovery — but pytest is **not** absent: it is already a declared dev dependency of
the Spensor sub-project (`python/spensor/pyproject.toml`). So this is about wiring, not about
breaking a convention.

- Add `pytest` and `pytest-xdist` to `utils/requirements_dev.txt`.
- Scope the suite to `programming_examples/transformer_layer/study/`; do not spread it.
- Define how it is installed, discovered and invoked from CMake/lit — a single `.lit` wrapper so
  `ninja check-...` remains the one entry point.
- Port iron's `conftest.py` `--iterations` / `--csv-output` / `metrics`-marker machinery,
  including its fix for unbracketed node IDs. That bug aborted the whole pytest session with an
  `INTERNALERROR` whenever a test node ID had no bracketed parameter suffix — which is every
  non-parametrized study test, and *any* test under `--iterations 1`, which is how CI runs.

## Environment items iron got wrong

- `matplotlib` (26 imports), `seaborn` (10) and `pandas` (6) are used across the study tier and
  declared in **no** iron requirements file. Pin them in
  `programming_examples/transformer_layer/requirements.txt`.
- `host_comparison` needs a ROCm torch wheel (`torch-2.9.1+rocm7.2.1`, from `repo.radeon.com`)
  that conflicts with the CPU-only index the root requirements pin. Document the force-reinstall.
- `fcntl` and `pwd` make this tier POSIX-only. State that.

### `[2026-08-08]` The latency confound has a cheaper fix than J2, and convention 10 already names it

The recorded plan for the attention-placement confound is J2: move attention onto the device for
`offload` and `runlist` so all four modes place it identically. That is right for the *design*
question. It is not the cheapest fix for the *measurement* question, and the measurement fix is
additive rather than a search over 828 configurations.

`Profiler` (`programming_examples/llms/shared/infra/cache.py`, which convention 10 already requires
this tier to route through) separates `record_kernel` — an `xrt.run()` — from `record_cpu`, whose
docstring names "CPU attention fallback" as the case it exists for, and reports the two in separate
sections. That is exactly the split the confound needs: a mode running attention on the host and a
mode running it on the device become comparable on their NPU time, with the host contribution
reported beside it instead of folded in.

**What it costs:** `pattern/` contains no timing at all today — `grep -rn perf_counter pattern/`
returns nothing — so the whole latency number is the single `perf_counter` that `study/run_mode.py`
wraps around the dispatch. Adding per-stage `record_kernel`/`record_cpu` calls to the four pattern
modules is new instrumentation on an untimed seam, not a rewrite of an existing one, and it lands in
Lane 1 files shared with `main`.

**Until it lands, latency comparisons across modes stay confounded**, and
`study/ladder_report.py`'s output says so in its own docstring rather than relying on a reader
remembering this section.

### `[2026-08-08]` Power: iron's backend cannot run here, and no NPU counter exists

Measured on this host, because J6 lists power among the cost metrics and `end_to_end/power.py`
(409 lines) is in the portable set — so it would have been ported before anyone checked whether it
can take a sample.

| path | result |
|---|---|
| `sudo -n turbostat --show PkgWatt` — **iron's actual invocation** | **unavailable.** `sudo -n` fails: a password is required. Iron's backend cannot run unattended here at all |
| `turbostat --no-msr` unprivileged | installed (2026.02.14), exits 0, and emits **no samples** — `PkgWatt` needs MSR access |
| `/sys/class/powercap/intel-rapl:0/energy_uj` (`package-0`) | **readable unprivileged.** Differencing it gave 19.96 W over 2.00 s. The `intel-rapl:0:0` (`core`) sub-zone is *not* readable |
| `/sys/class/hwmon/hwmon10/power1_average`, label `PPT` (`amdgpu`) | **readable unprivileged**, 22.05 W |
| `amdxdna` driver sysfs (the NPU) | exposes `power_state` only — a PM state, **no energy or power counter** |

Two conclusions, and the second is the load-bearing one.

**A root-free backend exists, and it is not turbostat.** Difference the RAPL `package-0` counter, or
read `amdgpu`'s `PPT`. Either keeps Phase G's unattended runner from needing root, which iron's
design does not. Port `power.py`'s *statistics* — the outlier detection, the percentile summary, the
probe-completeness policy — and replace its sampling backend rather than porting the `sudo` call.

**No sensor on this platform measures the NPU.** RAPL `package-0` is CPU package energy and
`amdgpu`'s `PPT` is the GPU/SoC rail. On an APU the NPU shares that SoC power envelope, so neither
number isolates it. That is not a wiring problem to solve later: it means a power comparison
*between execution modes* partly measures **host CPU work**, and the modes differ in exactly that —
some run attention on the host. So the confound this document already records for latency applies
to power with more force, since the host contribution is not merely a share of the number, it is
most of what the available sensors can see. **Decide what a power study claims before porting the
plumbing for it**; "watts per token on the NPU" is not measurable here, while "SoC watts while
executing this mode" is.

## Housekeeping

MLIR-AIR's `.gitignore` has no rules for result trees. Add:

```
results/
results_unattended_*/
*.csv
!**/removed_cases.csv
!**/*_candidates.json
```

A completed results root is ~2.4 GB.

## Work items

1. ~~Define and implement the versioned study schema~~ **done** — `study/schema.py`.
2. ~~Resolve the resource-usage artifact question~~ **done** — `study/aircc_artifacts.py`.
3. Port the ~19k-line infrastructure tier, applying the convention rules. ~~**Partly blocked, and
   the split is measured**~~ **`[2026-08-11]` The PORTABLE half is done; only the plot/analysis
   tier is left, and it is still blocked on an install that must not happen beside a live gate.**
   See §Item 3's portable half below. `matplotlib`, `pandas` and `seaborn` are absent, and
   installing them into the sandbox venv while a gate or a measurement run is live is the hazard
   item 6 records.
4. ~~Retarget the five device-touching modules.~~ **`[2026-08-11]` Four of five.** Two were
   already retargeted under other names (`run_mode.py` + `pattern/`; `sweep/registry_sweep.py`),
   `resource_usage` and the component aggregate landed today, and `memcpy_bandwidth` is open
   because the operator it needs does not exist — see §The modules that need real retargeting.
5. ~~Wire the pytest suite into CMake/lit~~ **done** `[2026-08-08]` — `run_study_host_tests.lit`
   + `study/run_host_tests.py`, **without pytest**, which is the one departure from this item as
   written. See below.
6. ~~Pin the missing dependencies; document the ROCm wheel conflict~~ **done** —
   `study/requirements.txt`.
7. ~~Update `.gitignore`~~ **done** `[2026-08-08]`, **scoped to result trees, not to `*.csv`**.
   See below.
8. ~~Write the iron-results adapter~~ **done** — `study/iron_adapter.py`.

### `[2026-08-08]` Item 3 is not uniformly blocked: 22,729 of 36,138 lines are portable today

Measured over the 50 modules of iron's `study/` tree, taking the **transitive** closure over
in-tree imports — a module that imports a plotting module is blocked at import time even if it
draws nothing itself:

| | modules | lines |
|---|---|---|
| Needs `matplotlib`/`pandas`/`seaborn` directly | 13 | 9,518 |
| Blocked transitively (the above plus importers) | 18 | 13,409 |
| **Portable with the dependencies as they are** | **32** | **22,729** |

Of the port-tier modules this document names, only three are blocked: `results_manifest.py` (380
— and **already superseded** by `study/manifest.py`, whose completeness rule departs from it
deliberately), `regenerate_plots.py` (271, plotting by definition) and `roofline/run.py` (1,773).
Everything else is clear, including **`end_to_end/power.py` (409), which is what J6's power metric
needs**, and `plot_families.py` (133), whose name is misleading — it holds family definitions, not
drawing code.

So the blocked set is genuinely the plotting and analysis tier, and the sequencing that follows is:
port the portable 22.7k under the convention rules now, defer plotting until the dependency
install can be done between device runs, and treat text-only reporting as the interim output —
`study/ladder_report.py` is that, and it needs nothing that is missing.

### `[2026-08-11]` Item 3's portable half: the named port tier is done, and three of it was superseded

Every module §The port tier that carries over names is now either ported, deliberately
superseded, or blocked on the install. What landed, with the conventions applied rather than the
files transplanted:

| iron | ported to | note |
|---|---|---|
| `run_lock.py` (39) | `study/run_lock.py` | Per OUTPUT FILE. The docstring's job is saying what it is NOT: `devq.sh` schedules the device, and pointing this at either device-lock inode would deadlock against the wrapper that launched the run |
| `plot_families.py` (132) + `end_to_end/cases.py` (772) | `study/cases.py` | ONE table. This document already says `plot_families`' "name is misleading — it holds family definitions, not drawing code", and two files defining the same six families is the drift convention 8 deletes on the way in. Brings `effective_gflops_per_sec`, which the schema has had a column for since v1 and nothing had filled |
| `end_to_end/select.py` (217) | `study/select_rows.py` | **Not `select.py`** — that is a stdlib module and this directory goes on `sys.path`, so the file loses to the stdlib one whenever that is already imported. Found by an `AttributeError` that says nothing about shadowing |
| `end_to_end/power.py` (408) | `study/power.py` | Statistics ported, sampling backend replaced — see below |
| `compare_results_roots.py` (511) | `study/compare_roots.py` | Doc 03's tiering kept; three holes closed — see below |
| `end_to_end/cases.py`'s group tables | `study/component_groups.py` | The item-4 half |

**Three are deliberately NOT ported, and this is the class the document already records for
`results_manifest.py`.** Check for an existing counterpart before porting anything:

- `results_manifest.py` (379) — superseded by `study/manifest.py`, whose completeness rule
  departs deliberately (already recorded).
- `npu_runtime_checks.py` (243) + `test_npu_runtime_checks.py` (141) — superseded by
  `sweep/registry_sweep.py`'s `require_turbo()` / `TurboNotEnforced`, which is **stricter**:
  iron's `warn_if_npu_power_mode_not_turbo` logs a warning and continues, and doc 32's resolved
  anomaly is exactly why that is wrong here — a `Default`-pmode latency is ~15–20× off any
  recorded number, so an undetectable power mode is a refusal and not a warning.
- `unattended_smoke_job.py` (216) — its measurement-gate half is superseded by
  `study/smoke_gate.py`, which closes the two holes recorded in that module's docstring; its
  fixture-plumbing half belongs to Phase G's unattended runner ([10](10-phase-g-unattended-runner-and-ci.md)),
  not here.

**`power.py` is where this document's own 2026-08-08 measurement paid off.** Its instruction was
to port the statistics and replace the sampling backend, and that is what shipped: the modified-Z
filter, the interquartile fallback, the ≥10/≥6 policy and iron's third condition (a filter that
WIDENS the spread has found structure, not outliers, and is skipped) are unchanged; the `sudo -n
turbostat` backend is gone and two root-free sysfs readers replace it, both verified live on this
host — `rapl_package` (differencing `energy_uj`, wrap-safe) and `amdgpu_ppt`. A fixture then found
a limit worth recording: **when more than three quarters of the samples share a value the MAD is 0
AND both quartiles coincide, so the interquartile fallback is 0 too and a lone spike survives into
the filtered mean.** Kept as iron has it — the alternative is inventing a dispersion measure for a
distribution that has none — and pinned in both directions, with the docstring telling a reader to
compare `max_power_w` against `avg_power_w` rather than trusting
`power_outlier_filter_applied`. The finding that no sensor here measures the NPU is restated in
the module, where someone about to quote a number will read it.

**`compare_roots.py` closes three of iron's holes.** The CSV list is the CALLER's — iron bakes its
own, so a file a run stopped producing is silently skipped, which is the same "passed having
measured nothing" shape as its smoke test and is why `smoke_gate` already takes `--expect`. Rows
are read through the schema, so a foreign CSV fails rather than being compared column-by-name
against something it is not (reading an iron tree is `iron_adapter`'s job). And there is no
intended-rename exception table, because convention 7 settled that naming once in the schema
module and an unexpected identifier difference is now unambiguously a failure. The dispatch vector
is compared as an IDENTIFIER rather than as drift: a mode whose submission count moved is a
different mode, not a noisier one.

**Still blocked on the install, enumerated so the next session does not re-derive it:**
`regenerate_plots.py` (270), `roofline/run.py` (1772) + `roofline/test.py` (720), and every
`plot_*.py` — `end_to_end/plot_selected_component_groups_vs_pattern.py` (621),
`end_to_end/plot_dataflow_blocks_vs_pattern.py` (405), `end_to_end/plot_tps_by_pattern.py` (327),
`memory_tile_staging/plot_staging_depth.py` (325), `block/plot_best_latency.py` (323) — plus the
runners that import one transitively. Text-only reporting stands in meanwhile:
`study/ladder_report.py`, and now `component_groups.render`.

### `[2026-08-11]` The component aggregate is honestly partial, and the columns say which part

`study/component_groups.py` cannot do what iron's `run_selected_component_aggregates.py` does,
and the reason is this document's own 2026-08-08 section: `pattern/` "contains no timing at all
today", the per-stage `record_kernel`/`record_cpu` calls are "new instrumentation on an untimed
seam", and they land in Lane 1 files shared with `main`. So what exists to aggregate is:

- **Named host components** — the `Profiler.time_cpu` buckets, which `offload` populates with
  five (`attention_layout`, `softmax`, `ln1`, `gelu`, `ln2`) and the other three modes leave
  empty BY CONSTRUCTION, because they run no host compute.
- **Unnamed device and sync totals** — every mode sums `device_submission_ms` and `host_sync_ms`
  over its dispatch vectors, and those vectors carry no component name.

The module therefore reports, per group, how many components it accounted for against how many
the taxonomy says it has, plus the missing ones BY NAME — a device group reads `0/12` rather than
presenting a mode total under a group label as though the group had been measured. The schema
`component` table carries `is_complete` as a column for exactly that reason. When the per-stage
instrumentation lands, the same taxonomy fills in and nothing else changes.

Its `render` also prints the **unattributed remainder** rather than leaving a reader to notice
the columns are short — doc 03 records that remainder dominating the per-operator modes at 1024
(`runlist` device 44 / sync 6.4 / host 0 against ~1959 total). The host taxonomy is a claim about
other files, so it is re-derived from the pattern sources by a test that reads text and imports
nothing, the way `test_attention_path.py` guards the attention map.

### `[2026-08-08]` Item 5 landed without pytest, and item 7 without `*.csv`

Both are deliberate departures from this document as first written, and both are narrower than
what it asked for.

**Item 5 — no pytest.** This section's plan was to add `pytest`/`pytest-xdist` to
`utils/requirements_dev.txt` and port iron's `conftest.py` machinery. What shipped is a `.lit`
wrapper running plain scripts, which is the idiom `run_block_cache_tests.lit` already established
here for the same reason: pytest is not in the sandbox venv, and installing it is the same class
of hazard as item 6's. The modules are written as `test_*` functions with plain `assert`, so
pytest collects them unchanged if it ever lands — the wiring is deferred, not designed out. The
`--iterations`/`--csv-output`/`metrics`-marker machinery is still unported and still wanted; it
serves *measurement* tests, and the suite today is host logic only, so nothing needs it yet.
**When it is ported, port its unbracketed-node-ID fix with it.**

The wrapper pins the test and module counts exactly (~~`61/61 passed in 6 modules`~~
~~`103/103 passed in 10 modules`~~ **`[2026-08-11]` `231/231 passed in 17 modules`**). Discovery by
glob satisfies convention rule 11, but glob alone cannot notice a test that stops being *defined* —
a deleted test function leaves a smaller suite passing. Verified in all three directions: matches
as-is, a shrunken 60/60 suite fails the `CHECK`, an injected failure exits nonzero.

`[2026-08-11]` The pin moved 103 → 133 → 196 → 229 → 231 over the commits that added
`resource_usage`, `run_lock`, `cases`, `power`, `compare_roots`, `component_groups` and
`select_rows` with their tests. Re-verified in the shrinking direction at the new value: a doctored
`228/228` output fails the `CHECK` with "expected string not found in input". **The suite is host
only and runs in ~0.4 s**, which is what keeps the pin cheap enough to be worth moving by hand.

Two of those 231 exist because writing the test found a bug in the port rather than confirming it:
`iter_cases` was carrying iron's short-circuit, where naming a family makes the variant filter a
no-op and a caller asking for decoders gets an encoder case back — a silently wrong answer that
looks correct. It now composes, and refuses an unknown family instead of returning an empty tuple
that reads the same as "no matches".

**Item 7 — result trees, not the extension.** The rules above proposed a blanket `*.csv`. That
silently ignores any CSV a future test wants to track as a fixture, and the failure mode is a file
that appears to commit and does not. iron's own rule set carries two negations
(`!**/removed_cases.csv`, `!**/*_candidates.json`) clawing specific files back out from under it,
which is evidence they hit exactly this. So what landed is `results/` and
`results_unattended_*/` and no extension rule: a measurement belongs in a result tree, and a CSV
being written outside one is the thing to fix. Checked every tracked file in the repository
against the new rules — zero become ignored.

## Gate

`execution-smoke-test` passes — meaning **at least one row per measurement CSV has
`run_status=passed`**, not merely that the expected files exist and are non-empty.

That distinction is not pedantry: iron shipped a smoke test that checked only file existence,
and it reported 21/21 passed on an environment where every measurement had failed. A broken
environment still writes complete, well-formed CSVs full of failed rows. The gate must also
report the first `failure_message` verbatim, which is usually enough to identify the cause.

### `[2026-08-08]` The gate passes on hardware, over all four modes

Measured through the harness's own `run_mode.py` at `baseline_768` seq 4096, `--warmup 1
--samples 2`, one mode at a time under `/tmp/mlir-air-npu.lock`. **`smoke_gate`: PASS (4 CSVs,
each with a passed row). `manifest`: `complete: True`.** Artifacts in the gitignored
`programming_examples/transformer_layer/results/phasef_smoke/`.

| mode | subs | entries | air | herd | sync | bytes | avg ms |
|---|---|---|---|---|---|---|---|
| `coarse` | 4 | 131 | 12 | 146 | 396 | 188,743,680 | 731.6 |
| `offload` | 6 | 6 | 7 | 19 | 19 | 139,984,896 | 606.5 |
| `runlist` | 5 | 391 | 14 | 404 | 395 | 150,994,944 | 787.7 |
| `fused` | 1 | 3 | 16 | 24 | 13 | 157,286,400 | 537.1 |

**All four distinguishability clauses hold on this data**, including J4's replacement — so the
clause is now verified against a measurement rather than only against a fixture. The two
recorded-but-not-gating predictions (`fused entries < coarse`, `fused air_launches >= coarse`)
also hold.

> **`[2026-08-08]` The latency column above is contaminated and superseded. The structural columns
> stand.** These four runs were taken while host work ran alongside them — builds, a formatter, the
> test suite. Compilation sits outside the clock; host-side dispatch does not. Measured immediately
> afterwards, `coarse` at this same 4096 reads **466.9 ms** and **476.9 ms** in two fresh processes
> on a quiet host, against the **731.6 ms** above: a **1.55× inflation**, larger than any gap this
> table was being read for.
>
> `subs`, `entries`, `air`, `herd`, `sync` and `bytes` are counts, not durations, and are unaffected
> — so the distinguishability verification, which reads only those, stands. The authoritative
> latencies are the sequence ladder's, one process per rung on a quiet host.
>
> The correction is not "I mis-typed a number": every latency in this table was measured under
> conditions the runner's own docstring warns against, by me, while I edited files in the same
> minutes. That is the failure mode, and it is why measurement conditions now appear as a rule in
> [23](23-rules-and-open-items.md) rather than as advice in a docstring.

**Do not read the latency column as a study result** even once re-measured. Attention placement
still varies across these modes (see the confound above), `--samples 2` is not a distribution, and
`fused` is fastest here partly *because* it runs attention on the host. The numbers that mean
something today are the structural ones — the six-field vectors — which is exactly what the gate
checks.

## Risks

- The schema and resource-artifact questions are both prerequisites; starting the bulk port
  before they are settled means reworking the tier.
- `host_comparison` needs an iGPU on the same host as the NPU. A machine without one can run
  every other study but not the full suite.
