# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The versioned MLIR-AIR study schema: field names AND what they mean.

CONTRACT
    Every results file this study writes carries ``schema_version``, and every
    field in it is declared here as a ``Field`` -- a name, what it measures, and
    for anything timed, WHAT SITS INSIDE AND OUTSIDE THE TIMED REGION. Callers
    build rows with ``empty_results_row()`` and check them with
    ``validate_row()``; nothing writes a bare dict of column names.

    This is a rewrite, not a port. iron's schemas are bare name tuples
    (``RESULTS_CSV_FIELDNAMES``, 39 columns; ``TUNING_CSV_FIELDNAMES``, 24), and
    doc 03 is explicit about why copying them is not enough: "MLIR-AIR has no
    RESULTS_CSV_FIELDNAMES, no execution_mode, no run_status. Copying iron's
    column names does not define what they mean under AIR's timing and
    synchronization model." A column called ``avg_latency_ms`` that includes
    compile time in one mode and excludes it in another produces a table whose
    rows cannot be compared, and nothing about the name says so. Hence
    ``Field.timing``, and hence ``test_schema.py`` asserting every timing field
    has one.

WHAT IS DELIBERATELY NOT HERE
    - **No measurement.** This module is the catalogue; the mechanism lives with
      the runners. That is the same seam ``opcheck_specs.py`` / ``opcheck.py``
      already draw, and porting convention 5's module cap is why.
    - **No dispatch counting.** The six dispatch-vector fields are declared here,
      but their single implementation is ``DispatchVector`` in
      ``llms/shared/infra/dispatch.py``, built in Phase B. Doc 03: "A per-mode
      reimplementation of 'what counts as a submission' would make the comparison
      meaningless." Declare the column here, call that class to fill it.

FOOTGUNS
    - **A failed measurement still writes a COMPLETE row.** ``run_status`` and
      ``failure_message`` carry the failure; the numeric fields stay ``None``.
      This is load-bearing for Phase F's gate, which requires at least one row
      per CSV with ``run_status=passed`` -- iron shipped a smoke test that
      checked only that files existed and reported 21/21 on an environment where
      every measurement had failed.
    - **``fused_elf`` is an execution_mode VALUE, not a column** (doc 03). Adding
      a mode must not add a column, or every existing row becomes unreadable.
    - **The CSV value for ``coarse`` is ``hybrid``, and that is not a mistake.**
      Porting convention 7 renames iron's ``hybrid`` module to ``coarse`` in
      code, directories and prose, and keeps ``hybrid`` as the CSV *value* so
      results stay diffable against iron's trees. So ``EXECUTION_MODES`` below
      lists ``hybrid``, not ``coarse`` -- getting this backwards makes the
      schema reject every row the shipped modes actually write, which is how it
      was caught.
    - **The quantization fields are here NOW and empty for bf16 rows** (doc 03).
      Bolting them on later renumbers the schema and invalidates every row
      already written; a ``dtype`` column alone cannot describe a quantized run.
    - **Adding a field is a schema version bump.** ``SCHEMA_VERSION`` is not
      decoration -- the iron adapter and ``compare_results_roots`` branch on it.
"""

from __future__ import annotations

from dataclasses import dataclass

#: v2 `[2026-08-10]`: adds the per-run millisecond decomposition
#: (``device_ms`` / ``sync_ms`` / ``host_cpu_ms``) and the reconfiguration
#: counters (``context_loads`` / ``kernel_attaches``). Both were measured and
#: DISCARDED under v1 -- doc 29 records that ``prepare_offload`` returned all
#: three ms components in its extra dict while ``run_mode`` read none of them,
#: and that the reconfiguration count, the ``offload`` mode's own axis, had
#: "nowhere to put it" in a results row. Every v1 column is unchanged in name,
#: meaning and position; the five new columns are appended AFTER all of them
#: (see the v2 section below), so a v1-shaped reader keeps working column for
#: column and only the version check tells the two apart.
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class Field:
    """One column: its name, its meaning, and its timing boundary if it has one.

    ``timing`` is required for any field whose value is a duration or a rate --
    it states what the clock covers. ``None`` means the field is not timed;
    ``test_schema.py`` enforces that a field whose name looks timed has one.
    """

    name: str
    meaning: str
    timing: str | None = None


# ---------------------------------------------------------------------------
# Identity and shape. One row per resolved full path through the case matrix.
# ---------------------------------------------------------------------------

_IDENTITY = (
    Field("schema_version", "This module's SCHEMA_VERSION at write time."),
    Field("study_id", "The study run this row belongs to; stable across resume."),
    Field("study_case_id", "Case identity within the study; the resume key."),
    Field("study_case_label", "Human-readable case name; never parsed."),
    Field("workload_variant", "encoder_bert or decoder_gpt2."),
    Field("backend", "Which runtime produced the row; xrt for device rows."),
    Field(
        "execution_mode",
        "One of EXECUTION_MODES. The taxonomy point being measured, and the "
        "column the whole study compares along.",
    ),
    Field(
        "attention_path",
        "Where attention ran: device or host_torch. NOT redundant with "
        "execution_mode -- offload keeps attention in host torch because its "
        "two attention GEMMs resolve in no registry, which makes it a hybrid "
        "boundary. A mode-vs-mode latency comparison that ignores this column "
        "is comparing attention placement as much as dispatch boundary.",
    ),
)

_SHAPE = (
    Field("seq_len", "Sequence length (M of every projection GEMM)."),
    Field("hidden_size", "Model width."),
    Field("intermediate_size", "FFN width; 4x hidden throughout the case matrix."),
    Field("num_attention_heads", "Head count."),
    Field("attention_head_size", "hidden_size // num_attention_heads."),
    Field("batch_size", "Always 1 in this study; present so rows stay joinable."),
    Field("dtype", "Activation element type, e.g. bf16. See the QUANT fields."),
    Field("use_bias", "Whether the projections carry a bias term."),
    Field("weights_source", "How weights were drawn; pins the golden reference."),
)

# ---------------------------------------------------------------------------
# Timing. Every field here states its region, because that is the whole point.
# ---------------------------------------------------------------------------

_TIMING = (
    Field(
        "warmup_runs",
        "Iterations executed and discarded before timing starts.",
        timing="Outside every timed region, by construction.",
    ),
    Field(
        "runs_per_sample",
        "Iterations inside one latency sample.",
        timing="Inside; a sample's duration divided by this is one iteration.",
    ),
    Field(
        "measured_inference_count",
        "Total timed iterations = latency_sample_count * runs_per_sample.",
        timing="Inside.",
    ),
    Field(
        "latency_sample_count",
        "Number of samples the statistics below are taken over.",
        timing="n/a -- a count of samples, not a duration.",
    ),
    Field(
        "timed_total_sec",
        "Wall time of all samples summed.",
        timing="Starts after warmup and after every buffer is resident; stops "
        "after the final device sync. EXCLUDES compile, weight upload and "
        "golden-reference construction.",
    ),
    Field(
        "avg_latency_ms",
        "Mean per-inference latency.",
        timing="timed_total_sec / measured_inference_count. Same region.",
    ),
    Field(
        "min_latency_ms",
        "Fastest sample, per inference.",
        timing="Same region. The least noisy estimator; prefer it when "
        "comparing kernels, since the mean carries host jitter.",
    ),
    Field(
        "max_latency_ms",
        "Slowest sample, per inference.",
        timing="Same region.",
    ),
    Field(
        "compile_setup_time_ms",
        "aircc compile plus xclbin load plus BO allocation.",
        timing="OUTSIDE the latency region, and reported separately rather than "
        "amortized into it. A fused mode compiles once and dispatches many "
        "times; folding this in would flatter it for the wrong reason.",
    ),
    Field(
        "host_qkv_precompute_ms",
        "Host-side work a mode does before dispatch, e.g. fusing QKV weights.",
        timing="OUTSIDE the latency region. Non-zero only for modes that "
        "precompute; a mode that does this work per iteration must count it "
        "inside instead, and say so here.",
    ),
    Field(
        "effective_gflops_per_sec",
        "Achieved throughput against the layer's analytic FLOP count.",
        timing="Derived from avg_latency_ms; inherits its region exactly.",
    ),
)

# ---------------------------------------------------------------------------
# The dispatch vector. Six fields, one implementation, doc 03's taxonomy.
# ---------------------------------------------------------------------------

_DISPATCH = (
    Field(
        "host_submissions_per_layer",
        "Host-to-device work submissions per layer. A runlist counts as ONE "
        "however many entries it carries.",
    ),
    Field(
        "runlist_entries_per_submission",
        "run objects in the submitted runlist; 1 for a plain kernel call. "
        "Emitted as a derived MEAN, so a mode's total is "
        "sum(round(mean * submissions)) and NOT the sum of the means -- the two "
        "agree only when every submission is 1.",
    ),
    Field("air_launches_per_elf", "air.launch operations in the compiled module."),
    Field("herd_launches", "Herd launches executed. Counts work, not packaging."),
    Field(
        "sync_boundaries",
        "Host-device sync and readback boundaries crossed per layer.",
    ),
    Field("bytes_transferred", "Bytes moved across those boundaries per layer."),
)

# ---------------------------------------------------------------------------
# Power. Raw and filtered, because the filter is a judgement call.
# ---------------------------------------------------------------------------

_POWER = (
    Field("power_backend", "Which sampler produced the block; None if unmeasured."),
    Field(
        "avg_power_w",
        "Mean power after outlier filtering.",
        timing="Sampled across the latency region only.",
    ),
    Field("min_power_w", "Minimum retained sample."),
    Field("max_power_w", "Maximum retained sample."),
    Field("power_sample_count", "Samples retained after filtering."),
    Field("power_std_w", "Standard deviation of retained samples."),
    Field("raw_avg_power_w", "Mean before filtering."),
    Field("raw_min_power_w", "Minimum before filtering."),
    Field("raw_max_power_w", "Maximum before filtering."),
    Field("raw_power_sample_count", "Samples before filtering."),
    Field("raw_power_std_w", "Standard deviation before filtering."),
    Field("power_outlier_sample_count", "Samples the filter removed."),
    Field(
        "power_outlier_filter_applied",
        "Whether filtering ran at all. Modified-Z >= 3.5, applied ONLY with "
        ">=10 samples and >=6 retained. Both raw and filtered statistics are "
        "persisted so a reader can tell what the filter did rather than "
        "trusting that it was reasonable.",
    ),
    Field(
        "effective_gflops_per_sec_per_watt",
        "Efficiency.",
        timing="Derived from effective_gflops_per_sec and avg_power_w; inherits "
        "the latency region.",
    ),
)

# ---------------------------------------------------------------------------
# Quantization. Empty for bf16 rows, present from v1 (doc 03).
# ---------------------------------------------------------------------------

_QUANT = (
    Field("quant_packing_scheme", "e.g. two_values_per_byte_low_nibble_first."),
    Field("quant_group_size", "Elements sharing one scale."),
    Field("quant_scale_layout", "How scales are laid out relative to the weight."),
    Field("quant_zero_point_layout", "Zero-point layout; None for symmetric."),
    Field("quant_accum_type", "Accumulator element type, which need not be dtype."),
    Field(
        "quant_gemm_contract",
        "The GEMM path's quantization contract. Separate from the GEMV one "
        "because prefill and decode do not have to agree, and a single column "
        "would hide it when they do not.",
    ),
    Field("quant_gemv_contract", "The GEMV (decode) path's contract."),
)

# ---------------------------------------------------------------------------
# Outcome. A failed row is still a complete row.
# ---------------------------------------------------------------------------

_OUTCOME = (
    Field("validation_error_count", "Elements outside tolerance; 0 on a clean run."),
    Field(
        "run_status",
        "One of RUN_STATUSES. A row exists whatever happened, so a CSV full of "
        "failed rows is well-formed -- which is exactly why the Phase F gate "
        "checks for a PASSED row rather than for a non-empty file.",
    ),
    Field("failure_message", "First failure verbatim; empty when passed."),
    Field("process_model", "in_process or subprocess; affects setup cost."),
    Field("npu_dispatch_count", "Dispatches the runtime actually issued."),
    Field("npu_unique_instruction_binary_count", "Distinct instruction binaries."),
    Field("npu_unique_xclbin_count", "Distinct xclbins loaded."),
    Field("selected_config_json", "Resolved tile configuration, as JSON."),
    Field("selected_candidate_ids_json", "Candidate ids the resolution chose."),
    Field("is_best", "Whether this row won its case after selection."),
)

# ---------------------------------------------------------------------------
# v2: the per-run millisecond decomposition, and the reconfiguration counters.
#
# Appended AFTER every v1 column, never interleaved -- ``test_schema.py`` pins
# the v1 prefix -- so anything that read a v1 file by column position reads a
# v2 file the same way and the five new columns are strictly additive.
#
# The decomposition answers the question a bare latency cannot: how much of a
# mode's time is the DEVICE executing, how much is DATA SYNC, and how much is
# the mode's own HOST arithmetic. Doc 03 lists it under the synchronization
# cost as "measured on every rung and discarded"; these columns stop the
# discarding. The three components are disjoint by construction and do NOT sum
# to ``avg_latency_ms`` -- the remainder is unattributed host overhead (Python
# scheduling, layout copies, the per-boundary stage comparisons), and a reader
# treating the gap as noise is misreading it.
# ---------------------------------------------------------------------------

_DECOMPOSITION = (
    Field(
        "device_ms",
        "Mean per-inference milliseconds the device spent executing submitted "
        "runlists: DispatchVector.submission_ms summed over the layer's "
        "submissions for one dispatch, averaged over the same timed "
        "iterations avg_latency_ms averages.",
        timing="INSIDE the latency region. Covers xrt.runlist "
        "execute()+wait() only (dispatch rule T1) -- EXCLUDES the bo.sync() "
        "traffic around it (that is sync_ms), the mode's host compute between "
        "submissions (host_cpu_ms), and the unattributed remainder. The three "
        "components do NOT sum to avg_latency_ms.",
    ),
    Field(
        "sync_ms",
        "Mean per-inference milliseconds in bo.sync() data traffic: host "
        "writes before submission, instruction-BO uploads (xclbin ABI, once "
        "per artifact identity), and host-output readbacks after -- "
        "DispatchVector.sync_ms, summed and averaged exactly as device_ms is.",
        timing="INSIDE the latency region and DISJOINT from device_ms by "
        "construction: the sync clock stops before execute() starts. With "
        "warmup >= 1 the once-per-process instruction upload lands in warmup, "
        "so this is the steady-state figure -- the same convention doc 03 "
        "fixes for the dispatch vector's cold-dispatch inflation.",
    ),
    Field(
        "host_cpu_ms",
        "Mean per-inference milliseconds of the mode's OWN timed host "
        "compute: the Profiler.time_cpu buckets (softmax, both LayerNorms, "
        "GeLU, attention layout), summed across buckets per dispatch and "
        "averaged as device_ms is. A recorded 0.0 is a MEASUREMENT -- the "
        "mode instruments host compute and ran none (fused, coarse) -- and an "
        "empty field means the mode reported no such component at all; never "
        "write a fabricated zero for the latter.",
        timing="INSIDE the latency region; disjoint from device_ms and "
        "sync_ms (a time_cpu block never wraps a dispatch). NOT all host "
        "time: untimed layout, Python overhead and the stage comparisons are "
        "in avg_latency_ms and in none of the three components.",
    ),
)

_RECONFIGURATION = (
    Field(
        "context_loads",
        "Array configurations performed during ONE steady-state layer "
        "dispatch. Every hw_context load counts one -- an xclbin load or an "
        "ELF backend.load(), both counted at the single increment in "
        "KernelCache.ensure_loaded -- and an eviction followed by a reload "
        "counts AGAIN: that is offload-ELF's 30 (a fresh context per GEMM "
        "dispatch, see pattern/offload's _evict_context) and the runlist "
        "front's per-head attention reloads. Taken from the LAST timed "
        "iteration, so with warmup >= 1 a once-per-process load -- the "
        "shared xclbin's single configuration -- lands in warmup and this "
        "column honestly reads 0 there. The cumulative-since-process counts "
        "the offload lit gate pins (context_loads 30 / 1) are "
        "KernelCache.reconfiguration_counts(), a different quantity; on a "
        "single cold dispatch the two coincide.",
    ),
    Field(
        "kernel_attaches",
        "Kernels bound onto an ALREADY-STANDING configuration "
        "(XRTBackend.attach_kernel) during that same dispatch -- the cheap "
        "half of reconfiguration, an instruction-stream bind rather than an "
        "array configuration. Non-zero only under a shared xclbin, and those "
        "attaches happen once per process, so a warmed row reads 0: the "
        "steady-state per-layer figure, same convention as context_loads.",
    ),
)

RESULTS_FIELDS: tuple[Field, ...] = (
    *_IDENTITY,
    *_SHAPE,
    *_TIMING,
    *_DISPATCH,
    *_POWER,
    *_QUANT,
    *_OUTCOME,
    # v2 additions -- LAST, and appended in this order. See the section
    # comment above; test_schema.py pins the v1 prefix and this suffix.
    *_DECOMPOSITION,
    *_RECONFIGURATION,
)

# The per-candidate tuning table: one row per candidate config per operator.
# Narrower on purpose -- it is a search log, not a comparison surface.
TUNING_FIELDS: tuple[Field, ...] = (
    Field("schema_version", "This module's SCHEMA_VERSION at write time."),
    Field("study_id", "The study run this row belongs to."),
    Field("study_case_id", "Case identity within the study."),
    Field("study_case_label", "Human-readable case name."),
    Field("workload_variant", "encoder_bert or decoder_gpt2."),
    Field("execution_mode", "One of EXECUTION_MODES."),
    Field("internal_operator", "Which operator this candidate belongs to."),
    Field("candidate_id", "Candidate identity within the operator's search."),
    Field("seq_len", "Sequence length."),
    Field("hidden_size", "Model width."),
    Field("intermediate_size", "FFN width."),
    Field("num_attention_heads", "Head count."),
    Field("attention_head_size", "hidden_size // num_attention_heads."),
    Field(
        "warmup_runs",
        "Discarded iterations.",
        timing="Outside every timed region.",
    ),
    Field("runs_per_sample", "Iterations per sample.", timing="Inside."),
    Field("latency_sample_count", "Samples taken.", timing="n/a -- a count."),
    Field(
        "avg_latency_ms",
        "Mean per-iteration latency for this candidate.",
        timing="Same region as the results table: post-warmup, buffers "
        "resident, through final sync; compile excluded.",
    ),
    Field("min_latency_ms", "Fastest sample.", timing="Same region."),
    Field("max_latency_ms", "Slowest sample.", timing="Same region."),
    Field(
        "bandwidth_gbps",
        "Achieved bandwidth for movement-bound candidates.",
        timing="Derived from avg_latency_ms; inherits its region.",
    ),
    Field("validation_error_count", "Elements outside tolerance."),
    Field("run_status", "One of RUN_STATUSES."),
    Field("failure_message", "First failure verbatim."),
    Field("operator_config_json", "The candidate's configuration, as JSON."),
    Field("is_operator_best", "Whether this candidate won its operator."),
)

# The per-artifact resource table: one row per compiled design, read off the
# routed AIE artifact `aircc_artifacts.routed_design` pins. A STRUCTURAL table
# -- every column is a property of the compiled design, so a row is reproducible
# from the artifact alone and no column is a wall-clock measurement.
#
# WHY IT IS A THIRD TABLE AND NOT COLUMNS ON `results`. A results row is one
# MEASUREMENT of one mode at one shape; a resource row is one COMPILED ARTIFACT,
# and a mode dispatches several (`offload` five, `coarse` five, `fused` three).
# Folding these columns into `results` would force one artifact's occupancy to
# stand for the mode's, which is the aggregation doc 03 refuses for the dispatch
# vector for the same reason. Adding a table is not a version bump of an
# existing one: `results` and `tuning` keep their field order exactly.
RESOURCE_FIELDS: tuple[Field, ...] = (
    Field("schema_version", "This module's SCHEMA_VERSION at write time."),
    Field("study_id", "The study run this row belongs to."),
    Field("study_case_id", "Case identity within the study."),
    Field("study_case_label", "Human-readable case name."),
    Field(
        "execution_mode",
        "One of EXECUTION_MODES, or None for an artifact that belongs to no "
        "mode -- a standalone operator compile has resource usage and no "
        "taxonomy point.",
    ),
    Field("artifact_name", "The compiled design's name, as the study knows it."),
    Field(
        "routed_design_path",
        "Where the parsed artifact was read from. Recorded because the file is "
        "per-segment-name and a second compile into the same project directory "
        "overwrites it (see aircc_artifacts).",
    ),
    Field("compute_tiles_used", "Distinct tiles carrying an aie.core."),
    Field("aie_tiles_with_buffers", "Distinct compute tiles declaring a buffer."),
    Field("aie_tile_allocated_bytes", "Bytes of L1 buffers over those tiles."),
    Field(
        "aie_tile_memory_utilization",
        "aie_tile_allocated_bytes / (aie_tiles_with_buffers * 65536). A POOLED "
        "ratio over the tiles that hold something, not over the array -- a "
        "design on 4 tiles and one on 32 are both measured against their own "
        "footprint. The min/max/mean/median below are the per-tile spread.",
    ),
    Field("aie_tile_memory_utilization_min", "Least-loaded compute tile."),
    Field("aie_tile_memory_utilization_max", "Most-loaded compute tile."),
    Field("aie_tile_memory_utilization_mean", "Mean over compute tiles."),
    Field("aie_tile_memory_utilization_median", "Median over compute tiles."),
    Field("mem_tiles_with_buffers", "Distinct memory tiles declaring a buffer."),
    Field("mem_tile_allocated_bytes", "Bytes of L2 buffers over those tiles."),
    Field(
        "mem_tile_memory_utilization",
        "Pooled against 524288 bytes per memory tile, same convention as the "
        "compute-tile ratio.",
    ),
    Field("mem_tile_memory_utilization_min", "Least-loaded memory tile."),
    Field("mem_tile_memory_utilization_max", "Most-loaded memory tile."),
    Field("mem_tile_memory_utilization_mean", "Mean over memory tiles."),
    Field("mem_tile_memory_utilization_median", "Median over memory tiles."),
    Field("shim_tiles_with_s2mm", "Shim tiles with at least one S2MM allocation."),
    Field("shim_s2mm_channels_used", "Distinct (shim tile, S2MM channel) pairs."),
    Field(
        "shim_s2mm_channel_utilization",
        "Against 2 channels per direction per shim tile (AIE2P).",
    ),
    Field("shim_tiles_with_mm2s", "Shim tiles with at least one MM2S allocation."),
    Field("shim_mm2s_channels_used", "Distinct (shim tile, MM2S channel) pairs."),
    Field("shim_mm2s_channel_utilization", "Against 2 channels per direction."),
    Field("shim_tiles_with_dma", "Shim tiles with any DMA allocation."),
    Field("shim_dma_channels_used", "Distinct channels over those tiles."),
    Field("shim_dma_channel_utilization", "Against 4 channels per shim tile."),
    Field("mem_tiles_with_dma", "Memory tiles with an explicit DMA allocation."),
    Field("mem_dma_channels_used", "Distinct channels over those tiles."),
    Field("mem_dma_channel_utilization", "Against 6 channels per memory tile."),
    Field(
        "mem_dma_channel_note",
        "Why the three columns above are None, when they are. An absent "
        "allocation is not zero usage -- it means the artifact does not state "
        "it -- so the note is written rather than a 0 invented.",
    ),
    Field("compute_tiles_with_dma", "Compute tiles with an explicit DMA allocation."),
    Field("compute_dma_channels_used", "Distinct channels over those tiles."),
    Field("compute_dma_channel_utilization", "Against 2 channels per compute tile."),
    Field("compute_dma_channel_note", "As mem_dma_channel_note, for compute tiles."),
    Field(
        "core_to_core_flows",
        "aie.flow ops whose source AND destination are compute tiles (row >= 2). "
        "AIR-specific and NOT in iron's table: doc 03 makes this the "
        "discriminator for AIE role style -- zero is time-multiplexed, at least "
        "the stage count is space-multiplexed -- and names "
        "norm_tail_structure.py's flow count as the tree's only instrument for "
        "it. Reading it off the same artifact as the occupancy makes the axis "
        "measurable per design instead of per hand-written gate.",
    ),
    Field("total_flows", "aie.flow ops of every kind, for the ratio's denominator."),
    Field("run_status", "One of RUN_STATUSES."),
    Field("failure_message", "First failure verbatim."),
)

# The per-component-group table: one row per named group of one mode's layer,
# so a whole-layer latency can be read as where the time went rather than as one
# number. `study/component_groups.py` writes it.
#
# WHY `is_complete` IS A COLUMN AND NOT AN ASSERTION. Only some of a mode's
# components are individually timed today -- doc 09's 2026-08-08 section scopes
# the per-stage `record_kernel`/`record_cpu` instrumentation the rest needs and
# notes it lands in Lane 1 pattern files. So a group row says how many of its
# expected components it actually accounted for, and which are missing by name.
# A table that silently reported partial groups as whole ones would understate
# every device group by exactly the amount nobody measured.
COMPONENT_FIELDS: tuple[Field, ...] = (
    Field("schema_version", "This module's SCHEMA_VERSION at write time."),
    Field("study_id", "The study run this row belongs to."),
    Field("study_case_id", "Case identity within the study."),
    Field("study_case_label", "Human-readable case name."),
    Field("workload_variant", "encoder_bert or decoder_gpt2."),
    Field("execution_mode", "One of EXECUTION_MODES."),
    Field("seq_len", "Sequence length."),
    Field("group_label", "The component group this row aggregates."),
    Field(
        "group_kind",
        "host_cpu, device or sync. WHICH INSTRUMENT produced the row: named "
        "Profiler.time_cpu buckets, the dispatch vectors' device submission "
        "time, or their sync time. Not decoration -- a device row is a mode "
        "TOTAL today and a host_cpu row is a named component, and reading the "
        "two as the same granularity is the mistake this column prevents.",
    ),
    Field(
        "avg_latency_ms",
        "Mean milliseconds this group accounted for, per timed iteration.",
        timing="Inside the same region as the results table's avg_latency_ms, "
        "and over the same iterations. The groups are DISJOINT but do NOT sum "
        "to it -- the remainder is unattributed host overhead, which doc 03 "
        "records as dominating the per-operator modes at 1024.",
    ),
    Field("component_count", "Components this group actually accounted for."),
    Field(
        "expected_component_count",
        "Components the taxonomy says the group has. Equal to component_count "
        "only when the group is fully instrumented.",
    ),
    Field(
        "missing_components_json",
        "The expected components with no measurement, as a JSON list. Empty "
        "list when the group is complete.",
    ),
    Field(
        "is_complete",
        "component_count == expected_component_count. False is the normal "
        "state for device groups today; see this table's note.",
    ),
    Field("run_status", "One of RUN_STATUSES."),
    Field("failure_message", "First failure verbatim."),
)

RESULTS_FIELDNAMES: tuple[str, ...] = tuple(f.name for f in RESULTS_FIELDS)
TUNING_FIELDNAMES: tuple[str, ...] = tuple(f.name for f in TUNING_FIELDS)
RESOURCE_FIELDNAMES: tuple[str, ...] = tuple(f.name for f in RESOURCE_FIELDS)
COMPONENT_FIELDNAMES: tuple[str, ...] = tuple(f.name for f in COMPONENT_FIELDS)

DISPATCH_VECTOR_FIELDNAMES: tuple[str, ...] = tuple(f.name for f in _DISPATCH)
POWER_FIELDNAMES: tuple[str, ...] = tuple(f.name for f in _POWER)
QUANT_FIELDNAMES: tuple[str, ...] = tuple(f.name for f in _QUANT)
DECOMPOSITION_FIELDNAMES: tuple[str, ...] = tuple(f.name for f in _DECOMPOSITION)
RECONFIGURATION_FIELDNAMES: tuple[str, ...] = tuple(f.name for f in _RECONFIGURATION)

#: The CSV values ``execution_mode`` may take -- taxonomy points as they appear
#: in a results file. ``fused_elf`` is a VALUE here, never a column.
#:
#: NOTE the direction, which is easy to get backwards: the code name is
#: ``coarse`` and the CSV value is ``hybrid``. Porting convention 7 renames
#: iron's ``hybrid`` module to ``coarse`` "in code, directories and prose" and
#: keeps ``hybrid`` only as the CSV value, so results stay diffable against
#: iron's trees. Verified against the shipped artifacts: the recorded
#: ``coarse`` run carries ``execution_mode='hybrid'``.
EXECUTION_MODES: tuple[str, ...] = ("offload", "runlist", "hybrid", "fused_elf")

#: mode name in this repository -> its ``execution_mode`` CSV value.
#:
#: Convention 7 says to "confine that mapping to one place in the schema
#: module", and this is that module. It is currently DUPLICATED in
#: ``pattern/__init__.py::EXECUTION_MODE_CSV``, which predates this file and
#: which the shipped modes read from; the two agree, and closing the duplication
#: means pointing that one here rather than adding a third. Asserted equal by
#: ``test_schema.py`` so they cannot drift in the meantime.
#: `[2026-08-09]` ``coarse_c2`` and ``coarse_c3`` are CELLS of ``coarse``
#: (28-coarse-blend-space.md) and record ``coarse``'s value: a cell is a point
#: inside the mode, not a fifth taxonomy point, and ``EXECUTION_MODES`` stays
#: four so a cross-mode table cannot silently compare a cell against the modes
#: as though it were one of them.
EXECUTION_MODE_CSV: dict[str, str] = {
    "coarse": "hybrid",
    "coarse_c2": "hybrid",
    "coarse_c3": "hybrid",
    "offload": "offload",
    "runlist": "runlist",
    "fused": "fused_elf",
}

#: Where attention ran. See the ``attention_path`` field on why this is not
#: derivable from ``execution_mode``.
ATTENTION_PATHS: tuple[str, ...] = ("device", "host_torch")

RUN_STATUSES: tuple[str, ...] = ("passed", "failed", "skipped")

_FIELDS_BY_TABLE: dict[str, tuple[Field, ...]] = {
    "results": RESULTS_FIELDS,
    "tuning": TUNING_FIELDS,
    "resource": RESOURCE_FIELDS,
    "component": COMPONENT_FIELDS,
}


def fields_for(table: str) -> tuple[Field, ...]:
    """The Field tuple for a known table. Raises on anything else."""
    try:
        return _FIELDS_BY_TABLE[table]
    except KeyError:
        raise ValueError(
            f"unknown table {table!r}; known tables are " f"{sorted(_FIELDS_BY_TABLE)}"
        ) from None


def empty_row(table: str = "results") -> dict[str, object]:
    """A complete row with every field present and ``None``-valued.

    Rows are built by filling this in, never by assembling a dict of the keys a
    caller happens to remember -- a missing column is how one mode's CSV stops
    joining against another's.  ``schema_version`` is pre-filled because it is
    never the caller's choice.
    """
    row: dict[str, object] = {f.name: None for f in fields_for(table)}
    row["schema_version"] = SCHEMA_VERSION
    return row


def validate_row(row: dict[str, object], table: str = "results") -> None:
    """Raise ``ValueError`` unless ``row`` is writable against this schema.

    Checks the things that silently corrupt a results tree: a missing or extra
    column, a schema version that is not this one, and an unknown value in a
    field whose domain is closed. It deliberately does NOT check that numeric
    fields are populated -- a failed measurement writes a complete row of
    ``None`` and that is valid, and required by the Phase F gate's design.
    """
    expected = {f.name for f in fields_for(table)}
    got = set(row)
    if missing := expected - got:
        raise ValueError(f"{table} row is missing columns: {sorted(missing)}")
    if extra := got - expected:
        raise ValueError(
            f"{table} row has columns not in schema v{SCHEMA_VERSION}: "
            f"{sorted(extra)}. Adding a field is a version bump."
        )
    if row["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"row carries schema_version {row['schema_version']!r}, "
            f"this module is v{SCHEMA_VERSION}; use the adapter to read it"
        )

    for name, domain in (
        ("execution_mode", EXECUTION_MODES),
        ("run_status", RUN_STATUSES),
        ("attention_path", ATTENTION_PATHS),
    ):
        value = row.get(name)
        if value is not None and value not in domain:
            raise ValueError(
                f"{name}={value!r} is not one of {list(domain)}"
                + (
                    f"; {value!r} is a mode's CODE name -- its CSV value is "
                    f"{EXECUTION_MODE_CSV[value]!r} (convention 7 keeps the two "
                    "different so results stay diffable against iron's)"
                    if name == "execution_mode" and value in EXECUTION_MODE_CSV
                    else ""
                )
            )
