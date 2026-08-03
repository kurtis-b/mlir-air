# 03 — Measurement Model

Define the measurement unit **before** writing any study code. This document fixes the
execution-boundary taxonomy, the metric that distinguishes its points, and the CSV schema that
records it.

## The taxonomy

Four points on the spectrum, from most host-mediated to most fused:

| Mode | CSV key | Who sequences the work | MLIR-AIR mechanism |
|---|---|---|---|
| offload | `offload` | Host, per GEMM | `KernelCache.load_and_run` per GEMM (exists) |
| fine runlist | `runlist` | One XRT runlist over many small kernels | Runlist aggregation — Phase B |
| coarse runlist | `hybrid` | One XRT runlist over few fused kernels | Same, plus coarse builders |
| fused ELF | `fused_elf` | MLIR-level fusion before compilation | `shared/infra/stitching.py::stitch_elf` (exists) |

The first three CSV keys are iron's, kept so results stay diffable against iron result trees
through the adapter described below. `fused_elf` is new — it is MLIR-AIR's existing production
mechanism, and adding it as a measured point is what makes this port additive rather than
duplicative.

Per convention rule 7, code and directories say `coarse`; only the CSV value says `hybrid`, and
that mapping lives in one place in the schema module.

## Why a single dispatch count does not work

`[Codex]` iron records `npu_dispatch_count`. That metric does not exist in MLIR-AIR, and
importing it would not separate these four modes:

- **Three of the four have one host submission.** `runlist`, `hybrid` and `fused_elf` all submit
  once per layer, so a submission count collapses them.
- **They do not count the same thing.** A runlist counts runtime `run` objects. `stitch_elf`
  counts AIR launches fused *before* compilation — it splices MLIR text fragments
  (`shared/infra/stitching.py:318`); it does not fuse arbitrary compiled ELFs.
- **The premise that a fused ELF means one dispatch per layer is false.** `llama32_1b` prefill
  is **3 XRT calls per layer** — `rms_gemms_rope.elf` (6 launches) → `flash_attn.elf`
  (1 launch) → `o_ffn.elf` (8 launches): 3 submissions but 15 AIR launches
  (`programming_examples/llms/llama32_1b/ARCHITECTURE.md`).

One number cannot carry that.

## The dispatch vector

Report a vector, not a scalar. Every measured row carries all six:

| Field | Definition |
|---|---|
| `host_submissions_per_layer` | Count of host→device work submissions (a runlist counts as one) |
| `runlist_entries_per_submission` | Number of `run` objects in the submitted runlist; 1 for a plain kernel call |
| `air_launches_per_elf` | `air.launch` operations in the compiled module |
| `herd_launches` | Herd (kernel) launches executed |
| `sync_boundaries` | Host↔device sync / readback boundaries crossed per layer |
| `bytes_transferred` | Bytes moved across those boundaries per layer |

These are what actually distinguish the modes. `offload` has 8 submissions and 8 sync
boundaries; `runlist` has 1 submission with ~29 entries; `coarse` has 1 submission with ~6
entries; `fused_elf` has few submissions with many AIR launches and near-zero intermediate sync.

Each field must have a written definition in the schema module, tied to the Phase B dispatch
model, and a single implementation that all four modes call. A per-mode reimplementation of
"what counts as a submission" would make the comparison meaningless.

## CSV schema

`[Codex]` MLIR-AIR has no `RESULTS_CSV_FIELDNAMES`, no `execution_mode`, no `run_status`. Its
benchmark pipeline emits heterogeneous JSON (`llms/bench/extract_perf.py`). **Copying iron's
column names does not define what they mean under AIR's timing and synchronization model.**

So: define a **versioned MLIR-AIR study schema**, not a copy.

- Every results file carries a `schema_version` column. Start at 1.
- Every timing, sync, power, resource and transfer field has written semantics in the schema
  module — specifically, what is inside and outside each timed region.
- `fused_elf` is a new `execution_mode` **value**, not a new column.
- The `run_status` / `failure_message` convention carries over unchanged: a failed measurement
  still produces a complete, well-formed row. This matters — see the smoke-test gate below.

### Relationship to iron's schema

Where byte-level comparison against iron result trees is wanted, write an **explicit adapter**
rather than pretending the schemas are identical. The adapter's job is to make
`compare_results_roots.py` able to read both, and to fail loudly on fields whose semantics
differ rather than silently comparing incomparable numbers.

iron's two schemas, for reference:

- `TUNING_CSV_FIELDNAMES` — 24 columns, one row per candidate config per internal operator.
- `RESULTS_CSV_FIELDNAMES` — ~50 columns, one row per resolved full path: case identifiers,
  shape parameters, latency statistics, `compile_setup_time_ms`, `effective_gflops_per_sec`,
  the power block, dispatch counts, `run_status` / `failure_message`, and the selected-config
  JSON blobs.

The power block (`PERSISTED_POWER_RESULT_FIELDS`) carries raw and filtered statistics plus
`power_outlier_filter_applied`; outlier filtering is modified-Z ≥ 3.5, applied only when there
are ≥10 samples and ≥6 retained.

### Quantization fields

`[Codex]` A `dtype` column is not enough to describe a quantized run. When Goal 2 lands, the
schema needs:

- packing scheme (e.g. `two_values_per_byte_low_nibble_first`)
- group size
- scale / zero-point layout
- accumulation type
- separate GEMM and GEMV contracts

Fold these into schema v1 now rather than bolting them on later — the columns can be empty for
bf16 rows.

## Power measurement

Two backends, matching iron:

- **NPU / package** — `turbostat` `PkgWatt`, streamed via `subprocess.Popen`. Tied to the
  running kernel: `/usr/bin/turbostat` is a dispatcher, and the real binary ships in
  `linux-tools-$(uname -r)`. A kernel change silently breaks every power row.
- **iGPU** — `rocm-smi`, one sample per invocation.

Neither exists in MLIR-AIR today; there is no `turbostat`, `powercap`, `energy_uj` or watt
reference anywhere in the tree. This is new capability, not a port of something adjacent.

## Run-to-run comparison

`compare_results_roots.py` keys on `(study_case_id, execution_mode, seq_len)`, requires
identifier columns to match exactly, and gates on median and p90 drift of `avg_latency_ms`,
`effective_gflops_per_sec` and `avg_power_w`. `min_*` / `max_*` are reported but never gate.

Per-mode tolerances, which must be preserved:

| Mode | Median | p90 |
|---|---|---|
| `hybrid`, `runlist` | 5% | 15% |
| `offload` | 20% | 35% |
| power (any) | 15% | 50% |

`offload` gets a wider band because it genuinely is noisier — roughly ten times the run-to-run
drift of the other modes, and an XRT version change alone has moved it 19–39% at
`seq_len >= 4096` while leaving the others within 0.6%. The comparator also reports provenance
differences (git commit, XRT version) that explain drift.

## Provenance

`results_manifest.py` writes a `results_manifest.json` per results root: git commit and dirty
flag, platform, full `xrt-smi examine` output, per-file records, and suite coverage. It exits
non-zero when coverage is incomplete. Keep that behaviour — it is what makes a run citable.
