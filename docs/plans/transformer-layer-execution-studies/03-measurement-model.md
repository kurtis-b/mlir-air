# 03 — Measurement Model

Define the measurement unit **before** writing any study code. This document fixes the
execution-boundary taxonomy, the metric that distinguishes its points, and the CSV schema that
records it.

## The taxonomy

`[2026-08-08]` **Corrected by the study's author.** The table this document used to open with
described the four modes by *who sequences the work*. That is not what the four points isolate, and
it is superseded — it survives only as a record, in §The superseded taxonomy at the end of this
section. This is the table to read.

The spectrum the four points span is **reconfiguration cost against DRAM traffic**:

| Mode | CSV key | What it isolates | The mechanism it must use |
|---|---|---|---|
| `runlist` | `runlist` | **Reconfiguration overhead**, paid in full. Every operator in the encoder layer executed **individually, on the device** | one kernel per operator, dispatched back to back; **nothing on the host** |
| `offload` | `offload` | **Reconfiguration MINIMIZED by dynamic partitioning.** **`[2026-08-08]` One xclbin, N instruction streams — one per GEMM shape — which is what iron implements and what this port now matches.** The array configuration never changes; only the BD program does, so moving between the layer's GEMM shapes costs an instruction swap rather than a reconfiguration. *One* stream with runtime-parameterized loop bounds is the increment beyond parity, deliberately deferred — see below | **every LINEAR operator on the NPU** — the six projections *and* both attention matmuls — through one runtime-parameterized matmul; **every NON-LINEAR operator on the host**: softmax, both LayerNorms, GeLU |
| `coarse` | `hybrid` | **Reconfiguration AND sync overhead minimized together**, by *mixing* `runlist` and `fused` per workload | per-operator choice between an individually dispatched kernel and a fused region |
| `fused` | `fused_elf` | **DRAM movement eliminated.** The whole encoder layer placed on the array, so nothing but the layer input and output crosses DRAM | **one xclbin**, whole layer resident |

`runlist` and `fused` are the two extremes — every operator reconfigured and nothing fused at one
end, one configuration and no intermediate DRAM traffic at the other. `offload` attacks the
reconfiguration axis with a runtime parameter instead of with fusion, and `coarse` is explicitly the
**blend** of the two extremes rather than a point of its own.

### `[2026-08-08]` Why `offload` matches iron rather than parameterizing the bounds

The corrected definition first said *one* instruction stream with the matmul's loop bounds arriving
as a runtime parameter. A feasibility spike established that **the stack cannot do that today**, and
the author's decision is to **match iron — N instruction streams under one xclbin — for now.**

The two are the same idea at different depths: both bake the shape information, and the only
question is whether the BD fields are *literals* or *patchable*. Today they are literals, with no
SSA operand anywhere on the path:

- `AIRRtToNpuPass.cpp:604-617` folds every offset/size/stride to `int64_t`; a still-dynamic entry
  does not fail, it **silently defaults** — offset→0 with a warning, size→1, stride→0.
- `AIEX.td:1095-1131` declares `NpuWriteBdOp`'s `buffer_length`, `d0_size`, `d1_size`, `d2_size`,
  `iteration_size` and every stride as **`I32Attr`**, so there is nothing to bind a value to.
  `DMAConfigureTaskForOp` likewise takes only `I32Attr:$repeat_count`, which is why AIR passes
  `Value()` with the comment "dynamic repeat count unused here".
- There is no loop left to parameterize regardless: `unrollAffineFors`/`unrollSCFFors`
  (`AIRRtToNpuPass.cpp:1850`, `:1977`) run `loopUnrollFull` and hard-fail if they cannot. Measured
  in real output — `air_project/npu.air.mlir:588-748` holds 25 `dma_configure_task_for` ops and
  **zero** `scf.for`.

So runtime bounds means operand forms in **mlir-aie's** dialect plus a lowering change in
**mlir-air** — two repos, its own port-loop phase. Recorded as a future increment, not a
prerequisite: MLIR-AIR reaches parity with iron without it, and the increment is worth doing only
if the study wants to *price* it.

**One cheaper experiment is still open** and would settle it without a dialect change:
`npu.update_from_scratchpad`, a firmware instruction documented as "originally intended to patch
shim buffer descriptor addresses". If it can patch a shim BD *register* on NPU2, the BD stays baked
and the size/stride fields are rewritten per run. Untested for BD registers — the only existing test
patches an L1 `aie.buffer`.

**A consequence to weigh before ever taking the increment:** one stream forces **one tiling recipe**
on all three GEMM shapes, which today resolve to three (`drain/tile_n96`, `drain/tile_n128`,
`fused-cast/tile_m64`). That cost is unmeasured, and it may exceed the reconfiguration cost the
change is meant to remove.

Two things about `offload` that the superseded table had backwards, and that the code does not yet
do: it is the mode with the *least* reconfiguration rather than the most host-mediated one, and its
host/device split is decided by **linearity** — not by which GEMMs happen to resolve in the
registry.

The CSV keys are naming, not taxonomy. The first three are iron's, kept so results stay diffable
against iron result trees through the adapter described below; `fused_elf` is new — it is MLIR-AIR's
existing production mechanism, and adding it as a measured point is what makes this port additive
rather than duplicative. Per convention rule 7, code and directories say `coarse`; only the CSV
value says `hybrid`, and that mapping lives in one place in the schema module.

### What is implemented, and what is left

`[2026-08-09]` **Rewritten.** This section used to size four gaps against the corrected definitions.
Three of the four are now closed and the fourth is untouched, so it lists state rather than gaps.
The superseded version is not kept: it was a to-do list, and a stale to-do list is worse than none.

| Mode | Implemented today | Left |
|---|---|---|
| `runlist` | **every operator individually on the device, nothing on the host.** 427 entries over 17 runlists; per head `attn_scores` → `softmax` → `attn_output`, device-resident inside one submission. Gated 2026-08-09 | nothing for the definition |
| `offload` | **every LINEAR operator on the NPU, every NON-LINEAR one on the host** — six projections plus both attention matmuls, with softmax / both LayerNorms / GeLU on the host. 30 dispatches. Gated 2026-08-08. **`[2026-08-09]` The reconfiguration half landed too** ([29](29-offload-n-streams.md)): five shapes in ONE xclbin, `context_loads 1` against the ELF path's 30, dispatch vector unchanged. So the mode now implements both halves of its definition | **`[2026-08-09]` GATED** — `run_npu2_offload_peano.lit` has a third recipe pinning `context_loads 1 kernel_attaches 4` on the shared path and `context_loads 30 kernel_attaches 0` on the ELF one, verified in the failing direction. Two things remain. **The shared path is bounded to single-launch modules**, so it builds at 1024 and *not* at the mode's own gated 4096, where the down-projection is a two-launch `fused-cast` and the fixed `air.insts.bin` name collides with itself ([29](29-offload-n-streams.md)); a fix is a change to `python/air/backend/xrt.py` and its own phase. **The default is still the ELF path** — switching packaging removes this mode's variance (316.9% → 17.6% at 512), though the switch changes the ABI as well as the reconfiguration and so does not isolate `_evict_context`; and the shared path costs ~20% on best-case latency there, and flipping would invalidate every recorded `offload` number including [27](27-common-ladder-result.md)'s table. Beyond that only the deferred increment: *one* stream with runtime-parameterized loop bounds, still blocked in the stack (§A of [26](26-mode-rebuild-feasibility.md)) |
| `fused` | **three ELFs at seq 1024**, every operator boundary inside the tail still round-tripping through DRAM. Gated 2026-08-08 | one xclbin, blocked twice over — see below — and "no DRAM between operators", which is capacity-bounded rather than engineering-bounded: 6 MiB on chip against one 6 MiB S×F intermediate at 1024 |
| `coarse` | the D2 block: five coarse fused kernels, four submissions | everything. It is defined as a per-workload *blend*, which is a choice per operator between an individual dispatch and a fused region, and nothing in the port expresses such a choice yet |

**`fused`'s one-xclbin blocker, measured rather than asserted.** `[2026-08-09]` FlashAttention
requires `runtime_loop_tiling_sizes=[1,1]` and the wide GEMMs are built at `[2,2]`; one ELF is one
aircc invocation. At `[2,2]` `mha_out_proj` @4096 compiles and then **hangs**
(`ERT_CMD_STATE_TIMEOUT`, 3/3, against 3/3 clean passes at `[1,1]`). Two corrections travel with
that: `omit_pingpong` is **not** part of the conflict, and the two settings produce **identical
lowered IR**, so a compile-only comparison "refutes" the conflict and is wrong to — which is what
[26 §4](26-mode-rebuild-feasibility.md) did before its retraction. The second blocker is
`air-fuse-channels`, which is O(N²) in channels and did not finish in 1200 s on a 90-channel stitch.

**The measurement consequence, restated for the current state.** Every result recorded before the
taxonomy correction — including the sequence ladder in
[25](25-first-study-result-sequence-ladder.md) and its crossover — ranks four implementations that
are not these four modes, and additionally differed in **attention placement**, which is what its
slopes actually split on. That covariate is now gone: all four modes run attention on the device, so
the split cannot be reproduced and `attention_path` is constant across every row a run can produce.

**Two things to check before building any cross-mode table from the catalogue as it stands.** The
modes are **not at one sequence length** — `fused` is at 1024, the other three at 4096 — so a table
assembled from the SPECS rows compares two lengths. And the one clean cross-mode number recorded so
far is DRAM traffic at 4096, and it should be read DECOMPOSED rather than as its headline ratio:
`runlist` 190,513,152 bytes against `offload` 970,457,088 is 5.1× overall, but on the **attention**
component it is 25,165,824 against 830,472,192 — **33×** — which is the part the taxonomy is about.
The total understates it because `runlist` additionally pays ~25 MB more on its banded norm chains,
a confound that opposes the effect rather than producing it. The README carries the full table.

### The superseded taxonomy — "who sequences the work"

> **`[superseded 2026-08-08]` Do not use this table.** It is kept as a record, not as a definition:
> the rest of this directory is ~2,900 lines that still describe the modes this way, and a reader
> tracing that framing needs to be able to find where it came from and see that it is retired.
> Every claim it makes about *what a mode isolates* is wrong. Rewriting the other documents is
> deliberately deferred until the corrected mechanisms exist. The CSV keys in it are still current,
> and are restated in the corrected table above.

Four points on the spectrum, from most host-mediated to most fused:

| Mode | CSV key | Who sequences the work | MLIR-AIR mechanism |
|---|---|---|---|
| offload | `offload` | Host, per GEMM | `KernelCache.load_and_run` per GEMM (exists) |
| fine runlist | `runlist` | One XRT runlist over many small kernels | Runlist aggregation — Phase B |
| coarse runlist | `hybrid` | One XRT runlist over few fused kernels | Same, plus coarse builders |
| fused ELF | `fused_elf` | MLIR-level fusion before compilation | `shared/infra/stitching.py::stitch_elf` (exists) |

Its two load-bearing errors, for anyone reconciling a downstream document against it: it places
`offload` at the host-mediated *end* of the spectrum when the mode is meant to be the one that
minimizes reconfiguration, and it makes `coarse` a point of its own — "one runlist over few fused
kernels" — rather than the per-workload blend of `runlist` and `fused` the author defines.

## `[2026-08-10]` The vocabulary — one term per level, and where the spectrum sits among the axes

Recorded by the study's author. Two kinds of drift had accumulated: "dispatch" had come to name
two different levels of the execution hierarchy — harmless in `offload`, where they coincide at
30/30, and load-bearing in `coarse`, where they are 4 and 131 — and "fused" names both a
mechanism the tree has (`stitch_elf`) and an aspiration it does not (operators resident on the
array), which is why every description of `fused`-the-mode has needed a footnote about DRAM
round-trips. This section fixes the terms and places the two-cost spectrum among the other axes
an execution strategy has.

### The execution hierarchy

One term per level, each tied to the counter that measures it. The runtime half of these
definitions already exists as written text — `DispatchVector`'s docstring
(`llms/shared/infra/dispatch.py:121-153`), whose "equals the number of dispatch steps" for
`runlist_entries` is the pin this table ratifies rather than competes with.

| Term | Definition | Instrument |
|---|---|---|
| **configuration** | one array state; loading one is a **reconfiguration** | `context_loads` / `kernel_attaches` — instrumented and gated for `offload` only ([29](29-offload-n-streams.md)) |
| **artifact** | one compiled, loadable unit — an ELF or an xclbin; one aircc invocation | the SPECS catalogue row |
| **submission** | one host→device handoff, and the host's completion-wait granularity. A runlist counts as one whatever it holds | `host_submissions` |
| **dispatch** | one `xrt.run` — one kernel invocation. ≡ *runlist entry*, and **never** the submission | `runlist_entries` |
| **launch** | one `air.launch` op in a compiled module; counted once per distinct artifact (a compile-time property) | `air_launches` |
| **herd launch** | one `air.herd` executed; accumulates per dispatch | `herd_launches` |
| **data sync** | one `bo.sync()`, either direction, with its bytes | `sync_boundaries`, `bytes_transferred` |

**Control sync and data sync are different events, and the vector already separates them.** A
submission's completion wait is a *control* sync, counted by `host_submissions`; a `bo.sync()`
is a *data* sync, counted by `sync_boundaries`. That distinction — not the raw counts — is what
reconciles `offload` having the fewest counted syncs at 4096 (90, against `runlist`'s 451)
while being the most synchronization-bound mode: each of its 90 sits on the critical path with
host compute between submissions, where `runlist`'s 451 are dominated by dispatches chained
inside 17 submissions.

### Composition — packaged against resident

"Fusion" was doing double duty. Split by mechanism:

| Term | Mechanism | What it collapses |
|---|---|---|
| **packaged** | what the code calls *stitched*: sub-kernel IRs spliced into one multi-launch module before compilation (`stitch_elf`, `shared/infra/stitching.py:318`) — one artifact, one configuration, hand-offs still crossing DRAM through the function's L3 arguments | submissions, dispatches, reconfigurations |
| **resident** | operators on the array **simultaneously**, hand-offs L1→L1 | bytes |

`fused`-the-mode becomes one line instead of a recurring footnote: **packaged today, resident
by definition.** Its 42.5 MB against `coarse`'s 44.0 at 1024 ([27](27-common-ladder-result.md))
is a *packaged* result; the resident result does not exist and is capacity-bounded (§What is
implemented, above).

### AIE role style — named by the multiplexing dimension

"Homogeneous per operator" is true at any instant and false across the run, which made it a
poor name. Name what is being multiplexed instead:

| Term | Supersedes | Meaning | Discriminator |
|---|---|---|---|
| **single-function** | homogeneous | the array only ever runs one kernel family; operator diversity is the host's problem | one kernel family per mode |
| **time-multiplexed** | homogeneous per operator | the array is wholly re-purposed, operator by operator | **zero** core→core `aie.flow` ops |
| **space-multiplexed** | heterogeneous, dataflow | tiles differentiated into concurrent stages, hand-offs L1→L1 | ≥ stage-count core→core flows |

Nothing in the dispatch vector measures this axis; the one instrument the tree has is
`norm_tail_structure.py`'s flow count — exactly 16 core→core at `herd_x = 8`
([23 §5](23-rules-and-open-items.md)). The only space-multiplexed designs in the tree are J7a's
norm-tail pipeline and J7b's accumulator ring, both standalone gates; **no mode's SPECS row
dispatches either.**

### The axes — three knobs, three costs

The author's extension of the axis list: alongside reconfiguration and off-chip traffic sit
NPU-side graph coverage, host↔NPU synchronization, AIE role style and the execution boundary.
They are not six independent dimensions — three are knobs a builder chooses, three are costs
the choices induce:

| Kind | Axis | Instrument |
|---|---|---|
| knob | **graph coverage** — which operators run NPU-side | `study/test_attention_path.py`, which asserts all-device across every mode |
| knob | **execution boundary** — where artifact and submission seams sit | `host_submissions` / `runlist_entries` / `air_launches` / `herd_launches` |
| knob | **AIE role style** | core→core flow count; structural checks only |
| cost | **reconfiguration** | `context_loads` / `kernel_attaches` (`offload`'s gate only) |
| cost | **off-chip traffic** | `bytes_transferred` |
| cost | **synchronization** | `sync_boundaries`; the ms decomposition (`device_ms` / `sync_ms` / `host_cpu_ms`) is measured on every rung and **discarded** — schema v1 has no column for it, README queue item 2 |

The corrected taxonomy is a two-**cost** spectrum with the knobs pinned. Coverage was equalized
across three modes by the 2026-08-08/09 rebuilds — which is what retired `attention_path` — and
`offload`'s partial coverage is definitional *and instrumental*: restricting the NPU to linear
operators is what lets one configuration cover its whole graph. Role style is single-function
or time-multiplexed in every gated row. The execution boundary is the mechanism knob: the
superseded table above ordered the modes by it, and the correction demoted it from *what the
modes isolate* to *what induces the costs*.

### The four modes, restated

| | artifacts | submissions | dispatches | composition | role style |
|---|---|---|---|---|---|
| `runlist` | one ELF per operator instance | 17 | 427 @4096 · 139 @1024 | none — fully decomposed | time-multiplexed |
| `offload` | 5 ELFs, or 1 xclbin on the shared path | 30 | 30 | none on the device — the host stitches the graph | single-function |
| `coarse` (C1) | 5 | 4 | 131 @4096 | packaged front, banded tail | time-multiplexed |
| `fused` | 3 | 1 | 3 | packaged — resident by definition | time-multiplexed — space-multiplexed by definition |

Sources: the README status board and [27](27-common-ladder-result.md) for `runlist` and
`offload`; the re-measured D2 vector (§The dispatch vector, below) for `coarse`;
[26 §6](26-mode-rebuild-feasibility.md)'s repair-run vector for `fused`'s structural counts.
Reconfiguration counts exist for `offload` alone; the ELF-path modes' context loads are
uninstrumented.

**Reading the historical documents against this section:** "stitched" reads as *packaged*;
"homogeneous per operator" as *time-multiplexed*; "heterogeneous" and "dataflow" as
*space-multiplexed*; an unqualified "dispatches" resolves by its number (`offload`'s "30
dispatches" is both levels at once; the banded `addnorm`'s "64 dispatches" are dispatches
inside one submission). The historical documents are deliberately not rewritten — the same
policy as the superseded-taxonomy note above.

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

> **`[2026-08-05]` `offload` is six submissions, not eight.** Two of its eight GEMMs —
> `attn_scores` (`4096x64x4096`) and `attn_output` (`4096x4096x64`) — resolve in no registry, and
> the C4 sweep **cannot be made to produce them**: `sweep_families.py` derives K and N from
> `FAMILY_HIDDEN × ROLE_KN_MULTIPLES` with a minimum hidden of 512, so no family stages a 64 in the
> K or N position. Phase E therefore keeps `offload`'s attention in host torch and dispatches the
> six projection GEMMs, which makes it a hybrid boundary rather than a pure per-GEMM device mode.
> Recorded as `attention_path` in every `offload` artifact. See
> [08 §Build order](08-phase-e-execution-strategies.md) and [08c](08c-phase-e3-offload.md).
>
> The consequence for this section is that **no absolute submission count is load-bearing**. The
> distinguishability gate asks for an ordering — `offload` above every other mode, and aggregating
> nothing — rather than for the number eight.

> **`[2026-08-05]` Those numbers were predictions, and the first one measured is wrong.** Phase D
> built the `coarse` layer and recorded its vector: **4 submissions, 131 runlist entries, 12 AIR
> launches, 146 herd launches, 402 sync boundaries**, not "1 submission with ~6 entries". The
>
> > **`[2026-08-08]` Re-measured through Phase F's runner: 4 / 131 / 12 / 146 / **396** / 188743680.**
> > Five of the six match D2 exactly; `sync_boundaries` is 396 rather than 402. Small, and not
> > investigated — recorded here so the next reader does not treat 402 as still-current or spend
> > time rediscovering the gap. The distinguishability gate is ordinal over these totals, so a
> > six-count drift changes no verdict.
> cause is not the taxonomy — it is that `build_addnorm_module` caps rows per call, so each of the
> two normalization points is 64 dispatches, and 128 of those 131 entries are that one operator's
> row blocking.
>
> This matters more than a corrected constant. [08](08-phase-e-execution-strategies.md)'s
> distinguishability gate says `runlist` should have "many" entries and `coarse` "few"; on these
> numbers `coarse` already has 131 before any fine-grained mode exists, so the predicted separation
> may be between a number and itself. **Decide which fields actually discriminate, and re-derive
> the expected values at `baseline_768`, before building the modes** — 08 §Gate says a failure to
> separate means the measurement model needs revisiting, and that condition has already fired once.
>
> **`[2026-08-05]` Decided, before the modes were built**, and implemented in
> `agents/scripts/port-loop/phase_e_checks.py`: the criterion is ordinal over driver-summed totals,
> never an absolute threshold. Four gating clauses and two recorded-but-not-halting predictions,
> listed in [08 §Gate](08-phase-e-execution-strategies.md). One arithmetic note that belongs here
> rather than there: `as_row()` emits `runlist_entries_per_submission` as a derived **mean**
> (`dispatch.py:166`), so a mode's total entry count is `Σ round(mean × submissions)` and not the
> sum of the means. The two agree for the block only because each of its submissions is 1.

Each field must have a written definition in the schema module, tied to the Phase B dispatch
model, and a single implementation that all four modes call. A per-mode reimplementation of
"what counts as a submission" would make the comparison meaningless.

> **`[2026-08-09]` Under the xclbin ABI a vector read from a COLD dispatch is inflated, and two
> of its fields move.** The first call uploads each artifact's **instruction stream** once
> (`sync_instruction_bos`), so `offload` at 1024 reads `sync 95 bytes 99141520` on the first
> dispatch and `sync 90 bytes 99090432` on every one after — five artifacts, five extra sync
> boundaries, and exactly **51,088 bytes**, which is the total size of the five cached
> `.insts.bin` files. **The ELF ABI does not do this at all**: an ELF embeds its instructions, the
> caller skips the upload, and the ELF path reads `sync 90` on its first dispatch and on every
> one after — measured on both paths at the same length.
>
> **The steady-state figures are what every recorded number in this study means**, because the
> ladder always runs with warmup; anything quoting a vector from a single cold dispatch under the
> xclbin ABI is quoting a different number. Found by a gate that dispatched once and disagreed
> with [27](27-common-ladder-result.md)'s ladder by exactly this delta. Reconfiguration counts
> are unaffected.

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
