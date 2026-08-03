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
