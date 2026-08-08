# 09 — Phase F: The Study Harness

The seven measurement studies. This is the largest phase by line count, and the phase where
most of the convention refactoring lands.

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

| Module | Lines | Why |
|---|---|---|
| `end_to_end/modes.py` | 2336 | The `_build_operator()` dispatch point mapping `execution_mode` → strategy. Shape survives; every `_benchmark_<op>` body does not. |
| `block/run.py` | 1313 | Per-operator candidate sweep against real operators. |
| `end_to_end/run_selected_component_aggregates.py` | 1578 | |
| `memcpy_bandwidth/run.py` | 586 | Lazy-imports the memcopy operator inside a function; small surface. |
| `resource_usage/{run,analysis}.py` | 1623 + 299 | See the open issue below. |

`host_comparison/run.py` (1784) only imports the torch reference and joins on the end-to-end
CSV, so it is nearly free — swap the reference and the join columns.

## Open issue: the resource-usage artifact

`[Codex]` `resource_usage/analysis.py` regex-parses `aie.core(%tile_r_c)`,
`aie.buffer(%tile…) : memref<…>` and `aie.*dma_allocation @…(%shim_noc_tile…, S2MM|MM2S, ch)`
out of iron's `input_physical.mlir`, normalizing against `AIE_TILE_LOCAL_MEMORY_BYTES=65536`,
`MEM_TILE_LOCAL_MEMORY_BYTES=524288` and `SHIM_DMA_CHANNELS_PER_DIRECTION=2`.

AIR lowers to the same `aie` dialect, but **"same dialect" does not guarantee a stable
equivalent artifact under `aircc`**. Identify the concrete post-`aircc` artifact and pin its
path before this phase starts. If none is stable, emit one deliberately rather than parsing an
incidental intermediate.

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
3. Port the ~19k-line infrastructure tier, applying the convention rules. **Partly blocked, and
   the split is measured** — see below. `matplotlib`, `pandas` and `seaborn` are absent, and
   installing them into the sandbox venv while a gate or a measurement run is live is the hazard
   item 6 records.
4. Retarget the five device-touching modules.
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

The wrapper pins the test and module counts exactly (`61/61 passed in 6 modules`). Discovery by
glob satisfies convention rule 11, but glob alone cannot notice a test that stops being *defined* —
a deleted test function leaves a smaller suite passing. Verified in all three directions: matches
as-is, a shrunken 60/60 suite fails the `CHECK`, an injected failure exits nonzero.

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
