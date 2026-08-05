# 01 — Port Inventory

Per-artifact triage of iron commit `1e014c1`. Four dispositions:

- **PORT** — carries over with import-path and convention changes only
- **ADAPT** — same structure, different plumbing underneath
- **REWRITE** — must be re-expressed against the AIR device API
- **DROP** — MLIR-AIR already covers it, or it should not exist

Every PORT and ADAPT item is still subject to
[02-porting-conventions.md](02-porting-conventions.md); "ports by structure" is not "ports
unchanged".

> **`[2026-08-05]` This is a triage, not a to-do list — much of it is done.** Phases A–D have
> landed: every `aie_kernels/aie2p/` row (A), the BO allocator and dirty-bit sync rows (B), all six
> operator rows (C1–C3), and `pattern/reference.py` (D2). Individual rows are annotated where the
> disposition itself turned out wrong. The status board in [README](README.md) is authoritative for
> what is complete; this file is authoritative for what each artifact *becomes*.
>
> **What is still open, and whose it is.** Phase E owns `offload/op.py`, `runlist/op.py` and
> `hybrid/op.py` (the last largely delivered already — see its row). Note that
> `iron/operators/{transpose,elementwise_mul}/design.py` appear below with no phase assigned, and
> `runlist` needs both: there is no `transpose` or `elementwise_mul` builder or example anywhere in
> `programming_examples/`, so they are **new device work**, not re-expression — the only new device
> work left in this plan. Everything else outstanding is Phase F's study tier.

## Summary

| Disposition | Approx. lines | Where it lands |
|---|---|---|
| PORT | ~19,000 | Phase F (study infrastructure tier) |
| ADAPT | ~6,500 | Phases B *(done)*, F |
| REWRITE | ~9,000 | Phases C *(done)*, E |
| DROP | ~1,500 | — |

## `iron/common/` — runtime layer

| Artifact | Lines | Disposition | Notes |
|---|---|---|---|
| `aie_context.py` BO liveness allocator | ~180 of 246 | ADAPT | The one genuinely valuable idea in this tier. Live ranges over the dispatch sequence, size-binned pooling, content-keyed static-data pool. Lands in `KernelCache`. See Phase B for why it is not a drop-in. |
| `aie_base.py` dirty-bit sync discipline | ~60 of 338 | ADAPT | Only written buffers sync to device; only declared outputs sync back. Without it, latency is not comparable to iron's. |
| `aie_base.py` `AIEOperatorBase` hierarchy | ~280 | DROP | Replaced by `build_*_module()` functions + `KernelCache`. Convention rule 1. |
| `aie_context.py` remainder | ~66 | DROP | Path discovery duplicating `air.tools` / `external_kernels._get_aie_include_dir()`. |
| `aie_device_manager.py` | 88 | DROP | Singleton over `DefaultNPURuntime` with `reset_runtime()` workarounds. `KernelCache` owns lifecycle. |
| `compilation.py` artifact DAG | 712 | DROP | `KernelCache` + native `aircc` cover compile caching and rebuild-on-flag-change. |
| `utils.py` bf16 bit-reinterpret helpers | 52 | PORT | Check first whether `air.backend.xrt_runner.type_mapper` already covers it. |
| `test_utils.py` `run_test()` | 149 | ADAPT | Assumes a 4-method interface (`buffers[name]`, `read_buffer`, `write_buffer`, `run_runlist`). Satisfy that shape over `KernelCache`. |

## `iron/operators/` — the six new operators

All `design.py` files are REWRITE: they are written against `aie.iron` ObjectFifo / Worker /
Runtime and have no counterpart in AIR's launch-herd model. All `reference.py` torch oracles
are PORT. Detail in [06-phase-c-operators.md](06-phase-c-operators.md).

| Operator | design.py | op.py | Disposition | Notes |
|---|---|---|---|---|
| `causal_mask` | — | 86 | DROP as an operator | Pure composition over eltwise-add plus a precomputed static mask. Becomes a builder keyword argument. |
| `qkv_proj` | 561 | 233 | REWRITE | GEMM `A(M,K) @ B(K,3K)` with C split three ways. Closest analogue: `shared/builders/rms_gemms_rope_multi.py` minus RMSNorm/RoPE. |
| `addnorm` | 382 | 213 | REWRITE | Weighted LayerNorm + residual. iron bakes weights into MLIR via `np.load()` at generation time and hashes them into artifact names — pass them as runtime memref args instead. |
| `ffn` | 1096 | 462 | REWRITE | Staged up-proj → fused GeLU → down-proj with memtile accumulation depth. |
| `mha_out_proj` | 1350 | 293 | REWRITE | Largest. Fused attention + output projection with optional causal masking. |
| `dynamic_gemm` | 1009 | 430 | **DROP** `[2026-08-04]` | Runtime M/N tail handling was one of three candidate answers to shape coverage. The C4 registry sweep is the answer taken, so this is not ported. Revisit only if a later phase needs a shape ladder a sweep cannot cover. |

Also modified in the source commit and needing the same treatment where used:
`gemm/design_batched.py` (988, REWRITE), `layer_norm/design_weighted.py` (298, REWRITE),
`gemm/{op,design}.py` deltas, `softmax/design.py` (+240), `transpose/design.py` (+109),
`elementwise_mul/design.py` (+118).

## `iron/applications/transformer_layer/pattern/`

| Artifact | Lines | Disposition | Notes |
|---|---|---|---|
| `pattern/reference.py` | 172 | **DONE (D2)** | Pure torch golden model, `encoder_bert` + `decoder_gpt2`. The correctness anchor for all four modes. Ported by structure, computed in FP32 from bf16-rounded inputs; `WEIGHT_DRAW_ORDER` preserved. Lives at `transformer_layer/pattern/reference.py` with `test_reference.py` pinning its composition. **Phase E imports it; it does not re-port it.** |
| `{offload,runlist,hybrid}/reference.py` | 8 each | DROP | Re-export shims. Import the shared reference. Convention rule 8. |
| `offload/op.py` | 689 | REWRITE | Easiest of the three: host-torch plus 8 single-GEMM dispatches. Port `_blocked_attention` / `_resolve_query_block_size` logic. |
| `runlist/op.py` | 1566 | REWRITE | 29 kernels, 42 runlist entries. Depends on Phase B. |
| `hybrid/op.py` | 709 | REWRITE | 12 runlist entries over 5-6 coarse kernels. Renamed `coarse` — convention rule 7. **`[2026-08-05]` Largely already built**: `transformer_layer/builders/block.py` is a fused-operator sequence over one runlist, which is this mode. Phase E gives it a strategy directory and the shared instrumentation rather than writing it again. Note its measured shape is 4 sequences, and the two normalization points are 64 dispatches each, not one. |
| xclbin incremental-merge mechanic | — | DROP | `aiecc --xclbin-input` / `--xclbin-instance-name` / `--xclbin-kernel-id` chaining. MLIR-AIR binds multiple ELF modules into one `hw_context` instead. |

## `iron/applications/transformer_layer/study/`

Of ~37 tracked modules, only **seven** import anything from `iron`. That split drives the
triage.

### PORT — zero `iron` imports (~19,000 lines)

`run_lock.py` (39), `plot_families.py` (132), `npu_runtime_checks.py` (243),
`results_manifest.py` (379), `compare_results_roots.py` (511) + `test_compare_results_roots.py`
(283), `regenerate_plots.py` (270), `unattended_smoke_job.py` (216), `end_to_end/power.py`
(408), all `plot_*.py` (~1,700), all `cases.py` (752 + 752 + 135), all `select.py` (798),
`run_latency_variation.py`, `run_staging_ablation.py`, `run_correctness_spot_checks.py`,
`run_fairness_repeatability.py` x2, `remeasure_power_only.py` x2,
`export_fixed_winner_candidates.py`, `roofline/run.py` (1772) + `roofline/test.py` (720).

Caveats that apply across this tier:

- The CSV schemas are column-name tuples, but copying names does not define their *meaning*
  under AIR's timing and synchronization model. See
  [03-measurement-model.md](03-measurement-model.md).
- `matplotlib`, `seaborn` and `pandas` are used by ~40 of these modules and declared in **no**
  iron requirements file. Pin them.
- `fcntl` and `pwd` make this tier POSIX-only.
- Convention rules 5, 7, 10 and 11 apply here — this is where most refactoring effort lands.

### ADAPT

| Artifact | Lines | Notes |
|---|---|---|
| `end_to_end/run.py` | 1048 | Imports nothing from `iron`; delegates device work to `modes.py`. Arg parsing, resume-row matching, `mark_best_rows`, lock/CSV writing are all generic. |
| `host_comparison/run.py` | 1784 | Only imports the torch reference; NPU numbers come from the end-to-end CSV. Swap the reference and the join columns. |
| `resource_usage/analysis.py` | 299 | Regex-parses `aie.core` / `aie.buffer` / `aie.*dma_allocation` from iron's `input_physical.mlir`. AIR lowers to the same dialect, but the equivalent post-`aircc` artifact is **unidentified** — open issue, resolve before Phase F. |
| `resource_usage/run.py` | 1623 | Locates build artifacts. |
| `unattended_reboot.py` | 2494 | Job plan is a data table of `(module, argv)` dicts; retargeting is confined to `build_job_plan()`. Convention rule 5 requires splitting this module. |
| `test_unattended_reboot.py` | 1790 | Splits alongside it. |
| `conftest.py` | 172 | `--iterations` / `--csv-output` / `metrics`-marker machinery, plus its unbracketed-nodeid fix. Route measurement through `Profiler` / `extract_perf.py` rather than a second scraper (rule 10). |

### REWRITE

| Artifact | Lines | Notes |
|---|---|---|
| `end_to_end/modes.py` | 2336 | The `_build_operator()` dispatch point mapping `execution_mode` → pattern class. Shape survives; every `_benchmark_<op>` body does not. Split per convention rule 5. |
| `block/run.py` | 1313 | Per-operator candidate sweep. Also becomes the registry sweep tool — see Phase C. |
| `end_to_end/run_selected_component_aggregates.py` | 1578 | |
| `memcpy_bandwidth/run.py` | 586 | Lazy-imports `AIEMemCopy` inside a function; small surface. |

## `aie_kernels/aie2p/` — C++ device kernels

All PORT, with build-recipe adaptation. Detail in [04-phase-a-kernels.md](04-phase-a-kernels.md).

| Artifact | Lines | Notes |
|---|---|---|
| `encoder.cc` | 1061 | New |
| `addnorm_ffn.cc` | 931 | New |
| `addnorm_ffn_addnorm.cc` | 936 | New, near-duplicate of the above — merge behind a `-D` flag (rule 8) |
| `mm.cc` delta | +1463 | `matmul_init_*` and `matmul_with_acc_*` variants. Must target the GEMM source the LLM path actually compiles. |
| `softmax.cc` delta | +68 | Two-pass streaming softmax |
| `layer_norm.cc` delta | +104 | `layer_norm_rows`, `add_layer_norm_rows` |
| `mha.cc` delta | +170 | Causal-mask row helpers |
| `aie_kernel_utils.h` | — | Pragma abstraction shim. Port only if MLIR-AIR has no equivalent. |

Note `.cc` files including other `.cc` files is **not** a deviation — MLIR-AIR does the same
(`mm_aie2p.cc` includes `zero.cc`).

## Test and packaging surface

| Artifact | Disposition | Notes |
|---|---|---|
| `pytest.ini` | ADAPT | `python_files = test.py` silently excludes iron's own `study/test_*.py` from directory collection. Normalize on standard discovery (rule 11). |
| `REUSE.toml` + `reuse lint` CI | DROP | MLIR-AIR does not use REUSE (rule 6). |
| `requirements.txt` (application) | ADAPT | Add the undeclared `matplotlib` / `seaborn` / `pandas`; document the ROCm torch wheel conflict. |
| `.gitignore` result-tree rules | PORT | MLIR-AIR has none; a completed results root is ~2.4 GB. |
