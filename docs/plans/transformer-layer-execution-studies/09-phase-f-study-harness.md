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

1. Define and implement the versioned study schema (prerequisite).
2. Resolve the resource-usage artifact question (prerequisite).
3. Port the ~19k-line infrastructure tier, applying the convention rules.
4. Retarget the five device-touching modules.
5. Wire the pytest suite into CMake/lit.
6. Pin the missing dependencies; document the ROCm wheel conflict.
7. Update `.gitignore`.
8. Write the iron-results adapter.

## Gate

`execution-smoke-test` passes — meaning **at least one row per measurement CSV has
`run_status=passed`**, not merely that the expected files exist and are non-empty.

That distinction is not pedantry: iron shipped a smoke test that checked only file existence,
and it reported 21/21 passed on an environment where every measurement had failed. A broken
environment still writes complete, well-formed CSVs full of failed rows. The gate must also
report the first `failure_message` verbatim, which is usually enough to identify the cause.

## Risks

- The schema and resource-artifact questions are both prerequisites; starting the bulk port
  before they are settled means reworking the tier.
- `host_comparison` needs an iGPU on the same host as the NPU. A machine without one can run
  every other study but not the full suite.
