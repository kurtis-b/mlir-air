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
#:
#: v3 `[2026-08-23]` (doc 56 section 3.6, H1a): the MODEL scope. Thirteen
#: columns appended AFTER every v2 column -- the same additive rule, the same
#: pinned prefix -- so a row measured over a whole model forward (prefill of a
#: prompt, decode of N tokens) is written to the SAME table as a layer row and
#: distinguished by ``measurement_scope``, not by a second schema. Every
#: recorded v1 and v2 CSV still reads through ``results_io.read_rows_compatible``
#: (the analysis tier's reader since 2026-08-14); the strict ``read_rows`` that
#: every WRITER uses now rejects them, which is the point of the version: a v2
#: row must not be carried into a v3 CSV and look complete.
SCHEMA_VERSION = 3


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

# ---------------------------------------------------------------------------
# v3: the MODEL scope `[2026-08-23]` -- doc 56 section 3.6, H1a.
#
# Appended AFTER every v2 column, never interleaved -- ``test_schema.py`` pins
# the v2 prefix exactly as it pins the v1 one -- so a v2-shaped reader reads a
# v3 file column for column and the thirteen new columns are strictly additive.
#
# WHAT A MODEL ROW IS. One measurement of one PHASE of one model's forward --
# the prefill of a prompt, or the decode of N tokens -- through the production
# drivers (``llms/shared/model_adapter.py``), under the study's discipline: the
# clock is the forward pass only (dispatch to the instant the result is
# CPU-readable; verification outside), the power mode is recorded, and a
# failure is a complete row. The REUSED columns keep their meaning: ``seq_len``
# is the physical M the kernels were compiled for (``ubatch_tokens`` for
# prefill, 1 for decode), ``weights_source`` is the checkpoint and its immutable
# revision, and every timing / power / quant / outcome / provenance field means
# what it means for a layer row. ``execution_mode`` keeps doc 03's meaning: the
# drivers' per-ELF ``load_and_run`` path IS ``hybrid`` (one submission per ELF,
# split at every host op), and a one-runlist-per-token path would be ``runlist``.
#
# THE PER-LAYER DISPATCH COLUMNS STAY NULL IN A MODEL ROW. ``air_launches_per_elf``
# counts launches IN a module and ``host_submissions_per_layer`` is per layer;
# neither survives summation over 28 layers and an LM head without changing
# meaning. A model row's dispatch record is ``model_dispatch_vector_json`` --
# the same seven-key record for the whole phase or per token, where
# ``air_launches`` is the launches EXECUTED in the scope (the boundary count doc
# 57 prices at ~107 us each), not the per-module figure. ``validate_row``
# refuses a model row that fills a per-layer dispatch column, so the two
# definitions cannot be read as one.
#
# THE BLOCK COMMENTS BELOW THAT SAY "SCHEMA_VERSION STAYS 2" record their own
# day's decision (items 15, 16 and resume each declined a bump for a fact that
# was not a per-row quantity). A whole-forward measurement IS a per-row
# quantity with its own columns, which is exactly what those comments said a
# bump was for; the v2 roots they protected read through
# ``results_io.read_rows_compatible``, which ``compare_roots``, ``smoke_gate``
# and ``manifest`` use since this bump.
# ---------------------------------------------------------------------------

_MODEL_SCOPE = (
    Field(
        "measurement_scope",
        "One of MEASUREMENT_SCOPES: `layer` (every row written before v3, and "
        "every layer-study row since) or `model` (a whole-forward row). None "
        "reads as `layer` -- the v1/v2 corpus never said, because it had "
        "nothing else to be.",
    ),
    Field("model_id", "The llms/ deployment directory name, e.g. qwen3_0_6b. None in a layer row."),
    Field("phase", "One of MODEL_PHASES: prefill or decode. None in a layer row."),
    Field(
        "logical_token_count",
        "Tokens the phase processed as the workload sees them: the valid prompt "
        "length for prefill (padding excluded), the tokens generated for decode.",
    ),
    Field(
        "ubatch_tokens",
        "The physical chunk the kernels were compiled for and dispatched at: "
        "M for prefill, 1 for decode. Equal to seq_len in a model row; a "
        "kernel-scaling row has prompt length == ubatch_tokens (no chunking) "
        "and says so in study_case_label (doc 56 section 3.4).",
    ),
    Field("context_start_tokens", "KV positions already held when the phase began: 0 for a fresh prefill."),
    Field("context_end_tokens", "KV positions held when the phase ended: prompt length after prefill, start + tokens generated after decode."),
    Field(
        "measured_token_count",
        "Tokens inside the timed region that the throughput counts: valid "
        "prompt tokens for prefill (padded tail rows are dispatched and "
        "EXCLUDED from the numerator), sampled tokens for decode.",
        timing="Inside. The numerator of tokens_per_second.",
    ),
    Field(
        "tokens_per_second",
        "measured_token_count / timed_total_sec.",
        timing="Derived from the same region as avg_latency_ms: the forward "
        "pass only -- dispatch to the instant logits are CPU-readable -- "
        "summed over the samples. Tokenization, EOS padding, the HF gate and "
        "the per-row verification are OUTSIDE. This is NOT a TTFT.",
    ),
    Field("precision_plan_id", "The doc 56 section 3.5 precision plan the row ran under: bf16 | w4_decode | w_bfp16_prefill | a8."),
    Field(
        "plan_hash",
        "`Plan.sha` (llms/shared/plan) of the workload this row measured: the "
        "artifact cache key, so two rows with one hash ran one planned "
        "sequence. 64 hex characters.",
    ),
    Field("host_ops", "Host-side operations executed inside the timed region (named Profiler.time_cpu buckets, counted per call)."),
    Field(
        "model_dispatch_vector_json",
        "JSON object with EXACTLY the keys MODEL_DISPATCH_VECTOR_KEYS: `scope` "
        "(one of MODEL_DISPATCH_SCOPES) and six non-negative integers. "
        "`host_submissions` counts xrt.run submissions, `runlist_entries` the "
        "run objects in them, `air_launches` and `herd_launches` the launches "
        "EXECUTED in the scope, `sync_boundaries` the bo.sync calls and "
        "`bytes_transferred` their bytes. Validated strictly: a missing key, an "
        "extra key or a negative count is refused at write time.",
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
    # v2 additions -- appended in this order. See the section comment above;
    # test_schema.py pins the v1 prefix and this v2 suffix.
    *_DECOMPOSITION,
    *_RECONFIGURATION,
    # v3 additions -- LAST. test_schema.py pins the v2 prefix and this suffix.
    *_MODEL_SCOPE,
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

# ---------------------------------------------------------------------------
# The MEASUREMENT CONDITION block `[2026-08-12]` (doc 34 M4).
#
# WHAT IT IS FOR. A latency is a number measured under conditions, and until now
# the conditions lived in README prose: "08-10's records are `Default`-
# conditional, pre-08-10's are Turbo-conditional". That rule is in prose because
# the data could not carry it, and it cost a day -- the 2026-08-10 "machine
# anomaly" was the non-persistent `xrt-smi` power mode resetting itself on an
# overnight reboot, and every latency recorded after it is ~15-20x off every
# latency recorded before it with nothing in either file saying so (README trap
# 0, doc 32). This block is that rule moved out of prose and into the artifact.
#
# WHY IT IS A BLOCK ON THE MANIFEST AND NOT A COLUMN ON `results`. Two reasons,
# and the first one is decisive.
#
#   1. A new `results` column is a SCHEMA VERSION BUMP, and a bump makes every
#      recorded CSV unreadable. `results_io.read_rows` rejects a header mismatch
#      AND a version mismatch, so v1 -> v2 on 2026-08-10 took 56 recorded CSVs
#      out of every current reader in one edit (doc 34 §1.4 records the Phase F
#      gate artifact failing for exactly this reason). Doing it again would take
#      the 8 that survived -- `ladder-v2-w{1,2}`, `postflip-ladder-w{1,2}` --
#      which are the roots `compare_roots` is actually pointed at. A guard bought
#      by destroying the artifacts it guards is not a fix.
#   2. It is not a per-row quantity. The pmode is a property of the RUN, holds
#      across every rung of a walk, and is exactly what two runs must agree on
#      before their rows may be compared at all. `RESOURCE_FIELDS` above already
#      states the general form of this rule -- "Adding a table is not a version
#      bump of an existing one: `results` and `tuning` keep their field order
#      exactly" -- and this is the same move: a new declared block, no bump,
#      `SCHEMA_VERSION` stays 2 and every recorded CSV keeps reading.
#
# It is deliberately NOT registered in `_FIELDS_BY_TABLE`. That registry is the
# CSV-table registry -- `results_io` builds a header out of it -- and a JSON
# block that turned up there could be written out as a one-row CSV by anything
# that iterates tables. `test_schema.py` pins `fields_for("conditions")` raising.
#
# DEGRADING TO `unknown` IS THE WHOLE POINT, AND IT IS NOT A SYNONYM FOR TURBO.
# Every root recorded before today has no such block, and an old root CANNOT be
# stamped after the fact: the mode it ran at is not recoverable from the files.
# So a reader gets `unknown` with the reason attached, never a crash and never a
# quiet match -- `conditions_from_manifest` below is the only supported way to
# read it. Writing an inference ("that walk falls inside the uninterrupted Turbo
# window, so call it turbo") into a data field is how the prose rule got there in
# the first place; `agents/scripts/port-loop/pmode_guard.py` made the same call
# for the throughput floor and for the same reason.
#
# THE OTHER HALF OF THE CONDITION IS `TOOLCHAIN_FIELDS`, BELOW `[2026-08-12]`.
# `xrt_version`, the mlir-aie/Peano pin and the `install-xrt`-vs-`build-xrt`
# resolution are the remaining conditions doc 34 M4 names, and doc 03 records an
# XRT version change alone moving `offload` 19-39%. Queue item 16 closed them as
# a SIBLING BLOCK rather than four more fields here; the reason is in that
# block's own comment.
# ---------------------------------------------------------------------------

#: What a condition records when it is not known. See the block comment: this is
#: never to be replaced by a guess, and a reader must treat it as "unconditioned
#: on the axis that costs 15-20x", not as "probably fine".
UNKNOWN_CONDITION = "unknown"

#: How a recorded condition was obtained. The three that may be WRITTEN, plus one
#: the reader synthesises.
#:
#:   observed                 -- the party that ran the measurement observed the
#:                               mode and stamped it. The strongest form, and the
#:                               only one whose clock is the measurement's clock.
#:   probed_at_manifest_build -- this host was queried when the manifest was
#:                               written, which is AFTER the run. Same machine,
#:                               later clock: good evidence, not an observation
#:                               of the measurement, and labelled so a reader can
#:                               tell the difference.
#:   unknown                  -- could not be determined, or was never supplied.
#:   absent                   -- READER-ONLY. The manifest predates this block
#:                               entirely. Never written; produced by
#:                               `conditions_from_manifest` so a guard can say
#:                               "this root is older than the field" rather than
#:                               "this root failed to determine its mode".
CONDITION_SOURCES: tuple[str, ...] = (
    "observed",
    "probed_at_manifest_build",
    "unknown",
    "absent",
)

CONDITION_FIELDS: tuple[Field, ...] = (
    Field(
        "npu_power_mode",
        "The `xrt-smi` NPU power mode this run was measured at, lowercased -- "
        "`turbo`, `default`, or UNKNOWN_CONDITION. THE gating condition: it is "
        "non-persistent (every reboot and every amdxdna reload resets it to "
        "`Default`), it is invisible in every other recorded field, and on this "
        "host a dispatch-bound measurement at `Default` runs ~15-20x slow. Two "
        "runs at different modes are not a comparison; README trap 0's closing "
        "rule is to re-measure the whole comparison after a pmode change and "
        "never to splice across one.",
    ),
    Field(
        "npu_power_mode_source",
        "One of CONDITION_SOURCES: how the mode above was obtained. Load-"
        "bearing, because `observed` and `probed_at_manifest_build` are not the "
        "same claim -- the second was read after the measurement finished and a "
        "driver reload in between would have moved it silently.",
    ),
    Field(
        "npu_power_mode_detail",
        "Provenance verbatim: the `xrt-smi` report the value was parsed out of, "
        "or the reason it could not be. An `unknown` that does not say why is "
        "indistinguishable from an `unknown` nobody tried to fill.",
    ),
    Field(
        "observed_at_utc",
        "When the condition was observed, ISO-8601 UTC. Read against the "
        "measurement CSVs' own mtimes it says whether the observation and the "
        "measurement belong to the same window. None when nothing was observed.",
    ),
)

# ---------------------------------------------------------------------------
# THE TOOLCHAIN BLOCK `[2026-08-12]` -- queue item 16.
#
# WHAT WAS BROKEN. `compare_roots.compare_manifests` has always contained
#
#     for label, block in (("git", "git"), ("toolchain", "toolchain")):
#
# and `manifest.py` has never written a `toolchain` key. `a.get("toolchain") or
# {}` is therefore `{}` on both sides, the inner loop iterates nothing, and the
# toolchain half of every root comparison has compared NOTHING for as long as it
# has existed. That is item 10's defect shape exactly: a check that reads as
# present in the source and is blind precisely where it is needed. The fix is to
# write the block, not to rewrite the reader -- the reader was always right.
#
# WHY A SIBLING BLOCK AND NOT FOUR MORE `CONDITION_FIELDS`. Three reasons.
#
#   1. DECISIVE: `compare_manifests` diffs `manifest["toolchain"]` BY NAME. Put
#      these fields in `conditions` and that loop is still iterating an empty
#      key -- the record gets fixed and the reader stays broken, which is half a
#      fix wearing a whole one's clothes.
#   2. `validate_conditions` rejects unknown keys, so widening CONDITION_FIELDS
#      would change the validation surface of every block item 15 shipped and
#      every test that pins it. A sibling leaves `conditions` byte-identical.
#   3. They answer different questions. The pmode is a property of the RUN and
#      resets under it; the toolchain is a property of the BUILD the run
#      exercised. `compare_roots` treats them differently on purpose (refuse vs
#      flag -- see `compare_toolchain` there), and two verdicts should not be
#      computed from one bag of fields.
#
# THE VERSIONING IS ITEM 15's, UNCHANGED AND FOR ITS REASON. A new `results`
# COLUMN would bump `SCHEMA_VERSION` to 3; `results_io.read_rows` rejects both a
# header and a version mismatch, so the bump that took 56 v1 CSVs out of every
# reader on 08-10 would take the 16 v2 CSVs that survive -- which are the roots
# `compare_roots` is actually pointed at. `RESOURCE_FIELDS` above states the
# governing precedent ("adding a table is not a version bump"). So: a new
# declared block, **`SCHEMA_VERSION` STAYS 2**, and deliberately NOT in
# `_FIELDS_BY_TABLE`, so nothing that iterates CSV tables can write it out as a
# one-row CSV. `test_schema.py` pins both.
#
# `absent` IS NOT `unknown` AND NEITHER IS A MATCH. Every root recorded before
# today has no such block. `toolchain_from_manifest` synthesises `absent` for
# those -- "older than the field" -- while `unknown` means "tried and failed".
# A reader must never treat either as agreement: two roots that both record
# nothing are not two roots that record the same thing.
#
# WHICH FIELDS COMPARE. Only `TOOLCHAIN_IDENTITY_FIELDNAMES` decide whether two
# roots agree. `toolchain_source` and `toolchain_detail` are PROVENANCE -- they
# say how the values were obtained, and a difference there is not a toolchain
# difference. There is deliberately no `observed_at_utc` here (the conditions
# block has one because a pmode can reset between the run and the manifest
# build): the manifest's own `created_at_utc` is the observation time, and a
# per-block timestamp would differ on every pair and print a NOTE on every
# comparison, which is how a diff teaches its reader to skip it.
# ---------------------------------------------------------------------------

#: The toolchain facts that decide whether two roots were built the same way.
#: A difference in ANY of these is a real toolchain difference; see the block
#: comment on why `toolchain_source`/`toolchain_detail` are excluded.
TOOLCHAIN_FIELDS: tuple[Field, ...] = (
    Field(
        "xrt_version",
        "The XRT runtime version this run dispatched through, as "
        "`BUILD_VERSION+VERSION_HASH[:12]` from `/opt/xilinx/xrt/version.json`. "
        "Doc 03 records an XRT version change ALONE moving `offload` 19-39% at "
        "seq_len >= 4096 while leaving the other three modes within 0.6% -- so "
        "this single field can move a latency by more than `offload`'s own 35% "
        "fail band, and is invisible in every recorded results column.",
    ),
    Field(
        "mlir_aie_version",
        "The installed `mlir_aie` wheel version. Layer 2 of the four-layer "
        "toolchain stack: AIR's build requires a floor here (the "
        "`ExpandModeAttr` wall), and the layers mask each other when stale, so "
        "a run's mlir-aie pin is part of what its numbers mean.",
    ),
    Field(
        "peano_version",
        "The installed `llvm-aie` (Peano) wheel version -- the per-core "
        "backend `PEANO_INSTALL_DIR` points at. It compiles the core code whose "
        "execution is being timed, so two roots at different Peanos are two "
        "different binaries measured.",
    ),
    Field(
        "air_resolution",
        "Which AIR tree the measuring interpreter would import: `build-xrt`, "
        "`install-xrt`, another path verbatim, or UNKNOWN_CONDITION. Doc 15's "
        "install-vs-build divergence is exactly this: the lit suites see a "
        "compiler fix as soon as `build-xrt` relinks, while an `install-xrt` "
        "caller sees it only after `ninja -C build-xrt install`. Two roots that "
        "differ here may have measured the same source at different ages.",
    ),
)

#: PROVENANCE, recorded beside the block and NOT compared. Same split, and the
#: same reasoning, as `npu_power_mode_source`/`_detail` in the conditions block.
TOOLCHAIN_PROVENANCE_FIELDS: tuple[Field, ...] = (
    Field(
        "toolchain_source",
        "One of CONDITION_SOURCES -- reused rather than duplicated, because the "
        "distinction it draws (`observed` at measurement time vs "
        "`probed_at_manifest_build` afterwards vs `unknown` vs reader-only "
        "`absent`) is the same distinction here. A second enum saying the same "
        "four things is a second enum to keep in sync.",
    ),
    Field(
        "toolchain_detail",
        "Why a field is missing, or where the values were read from. An "
        "`unknown` that does not say why is indistinguishable from an `unknown` "
        "nobody tried to fill.",
    ),
)

#: The manifest key the toolchain block lives under. This string is not a free
#: choice: `compare_manifests` has diffed `manifest["toolchain"]` since it was
#: written, and matching it is the entire point of the block.
TOOLCHAIN_KEY = "toolchain"

TOOLCHAIN_IDENTITY_FIELDNAMES: tuple[str, ...] = tuple(f.name for f in TOOLCHAIN_FIELDS)
TOOLCHAIN_FIELDNAMES: tuple[str, ...] = TOOLCHAIN_IDENTITY_FIELDNAMES + tuple(
    f.name for f in TOOLCHAIN_PROVENANCE_FIELDS
)

# ---------------------------------------------------------------------------
# THE TIMING-CONTRACT BLOCK `[2026-08-26]` -- queue item 19 review, finding 5.
#
# WHAT WAS BROKEN. `device_ms` is the sum of every dispatch's `kernel_ms`
# (`shared/model_adapter.dispatch_vector_from_trace`), and `KernelCache.
# load_and_run` used to time the construction of the `xrt.run` and the binding
# of its arguments INSIDE that number. Measured, that host work is 38-57 us per
# call (item 19 stage 1, devq 622/623). So the moment the run cache could skip
# it, `device_ms` fell by 30-50 us per call with the device doing exactly the
# same work -- and `compare_roots` would have reported a parent root against a
# post-commit root as a device improvement, under an unchanged schema, with the
# deciding variable (`LLMS_CACHE_XRT_RUNS`) recorded nowhere.
#
# THE FIX IS IN TWO HALVES AND BOTH ARE NEEDED.
#   1. `load_and_run` now times build/bind separately on BOTH paths, so
#      `kernel_ms` is start+wait only and no longer moves with the cache state.
#      That removes the phantom instead of labelling it.
#   2. The block below records WHICH CONTRACT a root's numbers were measured
#      under, because half 1 is itself a change of meaning: every root recorded
#      before it has run construction inside `kernel_ms` and every root after it
#      does not, and nothing in the files said so. `compare_roots.compare_timing`
#      REFUSES a comparison between two roots that name different contracts --
#      the pmode rule, for the pmode's reason: they are not two measurements of
#      one quantity.
#
# WHY A SIBLING BLOCK, AGAIN. The toolchain block's three reasons hold verbatim:
# a `results` column would bump `SCHEMA_VERSION` and take every recorded CSV out
# of every reader; `validate_conditions` rejects unknown keys so widening
# CONDITION_FIELDS changes the validation surface of every block item 15 shipped;
# and this answers a third question -- not "what mode did the device run at" and
# not "what was it built with", but "what does the recorded number MEAN". So:
# a new declared block, **`SCHEMA_VERSION` STAYS 2**, and deliberately NOT in
# `_FIELDS_BY_TABLE`.
#
# `absent` IS NOT A MATCH, for the toolchain block's reason. Every root recorded
# before today has no such block; `timing_from_manifest` synthesises `absent`
# and `compare_timing` FLAGS that (it cannot refuse -- refusing would make every
# recorded root uncomparable, which is the trade `compare_conditions` already
# reasoned through for an unknown pmode).
# ---------------------------------------------------------------------------

#: The contracts `kernel_ms` has been recorded under. Open-ended in the sense
#: that a future change adds a NAME here rather than silently changing what the
#: current name means -- that silent change is the defect this block exists for.
#:
#:   start_wait_only          -- `[2026-08-26]` and after: `kernel_ms` times
#:                               `run.start()` + `run.wait2()` (ELF) or the
#:                               kernel functor + `wait()` (xclbin), and the
#:                               host-side build/bind is reported separately as
#:                               `bind_ms`.
#:   bind_and_start_wait      -- before that: construction and argument binding
#:                               were inside `kernel_ms`, worth 38-57 us/call.
KERNEL_MS_CONTRACTS: tuple[str, ...] = ("start_wait_only", "bind_and_start_wait")

#: What the CURRENT code records. `manifest.observe_timing()` stamps it; a root
#: that names anything else was measured by a different build.
KERNEL_MS_CONTRACT_NOW = "start_wait_only"

TIMING_FIELDS: tuple[Field, ...] = (
    Field(
        "kernel_ms_contract",
        "What a recorded `kernel_ms` -- and therefore `device_ms`, which is its "
        "sum -- INCLUDES. One of KERNEL_MS_CONTRACTS. `bind_and_start_wait` "
        "roots carry 38-57 us/call of host-side `xrt.run` construction inside "
        "the device number; `start_wait_only` roots do not. Two roots that name "
        "different contracts are not two measurements of one quantity, and "
        "`compare_roots` refuses them.",
    ),
    Field(
        "xrt_run_cache",
        "Whether `KernelCache` reused one `xrt.run` per (kernel, bo_key) during "
        "this run -- `on`, `off`, or UNKNOWN_CONDITION, from "
        "`LLMS_CACHE_XRT_RUNS`. Under `start_wait_only` it does not move "
        "`device_ms` (that is the point of the split), but it does move the "
        "TOKEN-level wall by ~2-3 ms on a 57-submission decode token, so a "
        "tok/s comparison across two states is a comparison of two "
        "configurations.",
    ),
)

#: PROVENANCE, recorded beside the block and NOT compared -- the split, and the
#: reasoning, of `toolchain_source`/`toolchain_detail`.
TIMING_PROVENANCE_FIELDS: tuple[Field, ...] = (
    Field(
        "timing_source",
        "One of CONDITION_SOURCES, reused rather than duplicated.",
    ),
    Field(
        "timing_detail",
        "Where the values came from, or why a field is unknown. An `unknown` "
        "that does not say why is indistinguishable from one nobody filled.",
    ),
)

#: The manifest key the timing-contract block lives under.
TIMING_KEY = "timing"

TIMING_IDENTITY_FIELDNAMES: tuple[str, ...] = tuple(f.name for f in TIMING_FIELDS)
TIMING_FIELDNAMES: tuple[str, ...] = TIMING_IDENTITY_FIELDNAMES + tuple(
    f.name for f in TIMING_PROVENANCE_FIELDS
)

RESULTS_FIELDNAMES: tuple[str, ...] = tuple(f.name for f in RESULTS_FIELDS)
TUNING_FIELDNAMES: tuple[str, ...] = tuple(f.name for f in TUNING_FIELDS)
RESOURCE_FIELDNAMES: tuple[str, ...] = tuple(f.name for f in RESOURCE_FIELDS)
COMPONENT_FIELDNAMES: tuple[str, ...] = tuple(f.name for f in COMPONENT_FIELDS)
CONDITION_FIELDNAMES: tuple[str, ...] = tuple(f.name for f in CONDITION_FIELDS)

#: The manifest key the conditions block lives under.
CONDITIONS_KEY = "conditions"

DISPATCH_VECTOR_FIELDNAMES: tuple[str, ...] = tuple(f.name for f in _DISPATCH)
POWER_FIELDNAMES: tuple[str, ...] = tuple(f.name for f in _POWER)
QUANT_FIELDNAMES: tuple[str, ...] = tuple(f.name for f in _QUANT)
DECOMPOSITION_FIELDNAMES: tuple[str, ...] = tuple(f.name for f in _DECOMPOSITION)
RECONFIGURATION_FIELDNAMES: tuple[str, ...] = tuple(f.name for f in _RECONFIGURATION)
MODEL_SCOPE_FIELDNAMES: tuple[str, ...] = tuple(f.name for f in _MODEL_SCOPE)

#: ``measurement_scope`` values. ``layer`` is what every row before v3 was.
MEASUREMENT_SCOPES: tuple[str, ...] = ("layer", "model")
#: ``phase`` values of a model row.
MODEL_PHASES: tuple[str, ...] = ("prefill", "decode")
#: The precision plans doc 56 section 3.5 names. Open for a reason the conditions
#: block states for power modes: refusing a name would refuse to record it.
PRECISION_PLANS: tuple[str, ...] = ("bf16", "w4_decode", "w_bfp16_prefill", "a8")
#: What a model dispatch vector may describe: one whole phase, or one token.
MODEL_DISPATCH_SCOPES: tuple[str, ...] = ("prefill", "decode", "decode_token")
#: The seven keys of ``model_dispatch_vector_json``, in this order. Strict: see
#: the field and ``validate_model_dispatch_vector``.
MODEL_DISPATCH_VECTOR_KEYS: tuple[str, ...] = (
    "scope",
    "host_submissions",
    "runlist_entries",
    "air_launches",
    "herd_launches",
    "sync_boundaries",
    "bytes_transferred",
)

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
#: `host_numpy` `[2026-08-23]`: the drivers' decode attention (schema v3 model
#: rows) runs on the host in numpy, which is neither of the layer study's two.
ATTENTION_PATHS: tuple[str, ...] = ("device", "host_torch", "host_numpy")

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
        ("measurement_scope", MEASUREMENT_SCOPES),
        ("phase", MODEL_PHASES),
    ):
        if name not in row:
            continue  # a table without the column (tuning, resource, ...)
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
    if table == "results":
        _validate_model_scope(row)


def _validate_model_scope(row: dict[str, object]) -> None:
    """The v3 clauses. A layer row (scope None or `layer`) must carry NO model
    column; a model row must carry the scope's own record and NO per-layer
    dispatch column. Either direction silently redefines a column otherwise."""
    scope = row.get("measurement_scope")
    model_columns = [n for n in MODEL_SCOPE_FIELDNAMES if n != "measurement_scope"]
    if scope in (None, "layer"):
        if filled := [n for n in model_columns if row.get(n) is not None]:
            raise ValueError(
                f"a layer row (measurement_scope={scope!r}) carries model-scope "
                f"columns {filled}; set measurement_scope='model' or leave them None"
            )
        return
    # scope == "model"
    if filled := [n for n in DISPATCH_VECTOR_FIELDNAMES if row.get(n) is not None]:
        raise ValueError(
            f"a model row fills per-layer dispatch columns {filled}; they stay "
            "None in model rows (doc 56 section 3.6) -- the whole-phase record "
            "is model_dispatch_vector_json"
        )
    for name in ("model_id", "phase", "precision_plan_id"):
        if row.get(name) is None:
            raise ValueError(f"a model row must name its {name}")
    plan_hash = row.get("plan_hash")
    if plan_hash is not None:
        text = str(plan_hash)
        if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
            raise ValueError(f"plan_hash={plan_hash!r} is not a 64-hex Plan.sha")
    vector = row.get("model_dispatch_vector_json")
    if vector is not None:
        validate_model_dispatch_vector(vector)


def validate_model_dispatch_vector(value: object) -> dict[str, object]:
    """Parse and check a ``model_dispatch_vector_json`` value; returns the dict.

    Strict by design (doc 56 section 3.6): exactly MODEL_DISPATCH_VECTOR_KEYS,
    `scope` in MODEL_DISPATCH_SCOPES, every count a non-negative integer. A
    JSON string (as read from a CSV) or a dict (as built in memory) both work.
    """
    import json as _json

    if isinstance(value, str):
        try:
            parsed = _json.loads(value)
        except ValueError as exc:
            raise ValueError(f"model_dispatch_vector_json is not JSON: {exc}") from None
    else:
        parsed = value
    if not isinstance(parsed, dict):
        raise ValueError("model_dispatch_vector_json must be a JSON object")
    expected = set(MODEL_DISPATCH_VECTOR_KEYS)
    got = set(parsed)
    if missing := expected - got:
        raise ValueError(f"model_dispatch_vector_json is missing keys: {sorted(missing)}")
    if extra := got - expected:
        raise ValueError(
            f"model_dispatch_vector_json has keys not in the schema: {sorted(extra)}"
        )
    if parsed["scope"] not in MODEL_DISPATCH_SCOPES:
        raise ValueError(
            f"model_dispatch_vector_json scope={parsed['scope']!r} is not one of "
            f"{list(MODEL_DISPATCH_SCOPES)}"
        )
    for key in MODEL_DISPATCH_VECTOR_KEYS[1:]:
        count = parsed[key]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(
                f"model_dispatch_vector_json {key}={count!r} is not a non-negative integer"
            )
    return parsed


# ---------------------------------------------------------------------------
# The conditions block: build, read, validate. See the section comment above.
# ---------------------------------------------------------------------------


def empty_conditions() -> dict[str, object]:
    """A complete conditions block that claims nothing.

    Every declared field present, the mode ``unknown``, the source ``unknown``.
    Callers fill it in; nothing assembles a dict of the keys it remembers, for
    the same reason ``empty_row`` exists.
    """
    return {
        "npu_power_mode": UNKNOWN_CONDITION,
        "npu_power_mode_source": UNKNOWN_CONDITION,
        "npu_power_mode_detail": None,
        "observed_at_utc": None,
    }


def normalise_power_mode(value: object) -> str:
    """A recorded mode as a comparable string; anything unusable is ``unknown``.

    ``None``, ``""``, whitespace and the literal ``"unknown"`` all collapse to
    UNKNOWN_CONDITION, so a reader never has to enumerate the ways a field can
    fail to say something. Case is folded because ``xrt-smi`` prints ``Turbo``
    and every comparison in this study is against the lowercased form.
    """
    text = str("" if value is None else value).strip().lower()
    return text or UNKNOWN_CONDITION


def conditions_from_manifest(manifest: object) -> dict[str, object]:
    """The conditions block of a manifest dict, degrading to ``unknown``.

    THE ONLY SUPPORTED READER, and the reason is back-compatibility: every
    results root recorded before `[2026-08-12]` has no such block -- the shipped
    `results/phasef_smoke/manifest.json` is schema v1 and the recorded ladder
    walks carry no manifest at all -- so a reader that indexed the key would
    raise on the whole recorded corpus. It returns a COMPLETE block whatever it
    is handed, with ``npu_power_mode_source`` set to:

      ``absent``  the manifest predates the block, or is not a mapping at all.
                  Distinguished from ``unknown`` on purpose: "older than the
                  field" and "tried and failed" are different things to tell an
                  operator, and only the first is expected of the corpus.
      whatever the manifest recorded, otherwise.

    It never returns a mode that was inferred rather than recorded.
    """
    if not isinstance(manifest, dict):
        block = None
    else:
        block = manifest.get(CONDITIONS_KEY)

    out = empty_conditions()
    if not isinstance(block, dict):
        out["npu_power_mode_source"] = "absent"
        out["npu_power_mode_detail"] = (
            "this manifest has no conditions block -- it was written before "
            "the measurement condition was recorded (doc 34 M4). The mode it "
            "ran at is not recoverable from the files and must not be guessed."
        )
        return out

    for name in CONDITION_FIELDNAMES:
        if name in block:
            out[name] = block[name]
    out["npu_power_mode"] = normalise_power_mode(out["npu_power_mode"])
    out["npu_power_mode_source"] = normalise_power_mode(out["npu_power_mode_source"])
    return out


def validate_conditions(block: dict[str, object]) -> None:
    """Raise ``ValueError`` unless ``block`` is a writable conditions block.

    Same shape of check as ``validate_row``: a missing or extra key, and a
    ``source`` outside CONDITION_SOURCES. The MODE's domain is deliberately open
    -- ``xrt-smi`` names the modes and may name a new one, and a schema that
    rejected an unrecognised mode would refuse to record the very thing a reader
    most needs to see.
    """
    expected = set(CONDITION_FIELDNAMES)
    got = set(block)
    if missing := expected - got:
        raise ValueError(f"conditions block is missing keys: {sorted(missing)}")
    if extra := got - expected:
        raise ValueError(
            f"conditions block has keys not in the schema: {sorted(extra)}. "
            "Adding a condition is a declaration in schema.CONDITION_FIELDS, "
            "not a key invented at the call site."
        )
    source = normalise_power_mode(block.get("npu_power_mode_source"))
    if source not in CONDITION_SOURCES:
        raise ValueError(
            f"npu_power_mode_source={block.get('npu_power_mode_source')!r} is "
            f"not one of {list(CONDITION_SOURCES)}"
        )
    if source == "absent":
        raise ValueError(
            "npu_power_mode_source='absent' is READER-ONLY -- it is what "
            "conditions_from_manifest synthesises for a manifest older than the "
            "block, and writing it would claim a run predates a field it "
            "carries. Use 'unknown' if the mode could not be determined."
        )


# ---------------------------------------------------------------------------
# The toolchain block: build, read, validate. See the section comment above.
# ---------------------------------------------------------------------------


def empty_toolchain() -> dict[str, object]:
    """A complete toolchain block that claims nothing.

    Every declared field present and ``unknown``. Same rule as
    ``empty_conditions``: nothing assembles a dict of the keys it remembers.
    """
    block: dict[str, object] = {
        name: UNKNOWN_CONDITION for name in TOOLCHAIN_IDENTITY_FIELDNAMES
    }
    block["toolchain_source"] = UNKNOWN_CONDITION
    block["toolchain_detail"] = None
    return block


def toolchain_from_manifest(manifest: object) -> dict[str, object]:
    """The toolchain block of a manifest dict, degrading to ``absent``.

    THE ONLY SUPPORTED READER, for ``conditions_from_manifest``'s reason: no
    manifest written before `[2026-08-12]` has this key -- and, because the diff
    that consumes it was written against a key nothing produced, NO manifest ever
    written has it. A reader that indexed the key would raise on the entire
    recorded corpus.

    ``toolchain_source='absent'`` means the manifest predates the block. It is
    NOT ``unknown`` and it is emphatically not a match: see the section comment.
    """
    if not isinstance(manifest, dict):
        block = None
    else:
        block = manifest.get(TOOLCHAIN_KEY)

    out = empty_toolchain()
    if not isinstance(block, dict):
        out["toolchain_source"] = "absent"
        out["toolchain_detail"] = (
            "this manifest has no toolchain block -- it was written before the "
            "toolchain condition was recorded (doc 34 M4, queue item 16). The "
            "toolchain it ran against is not recoverable from the files and "
            "must not be guessed."
        )
        return out

    for name in TOOLCHAIN_FIELDNAMES:
        if name in block:
            out[name] = block[name]
    for name in TOOLCHAIN_IDENTITY_FIELDNAMES:
        out[name] = normalise_power_mode(out[name])
    out["toolchain_source"] = normalise_power_mode(out["toolchain_source"])
    return out


def validate_toolchain(block: dict[str, object]) -> None:
    """Raise ``ValueError`` unless ``block`` is a writable toolchain block.

    Same shape of check as ``validate_conditions``, and the same open domain on
    the VALUES: a wheel may take a version string this module has never seen, and
    a schema that rejected an unrecognised one would refuse to record the very
    thing a reader most needs. Only ``toolchain_source`` has a closed domain.
    """
    expected = set(TOOLCHAIN_FIELDNAMES)
    got = set(block)
    if missing := expected - got:
        raise ValueError(f"toolchain block is missing keys: {sorted(missing)}")
    if extra := got - expected:
        raise ValueError(
            f"toolchain block has keys not in the schema: {sorted(extra)}. "
            "Adding a toolchain fact is a declaration in "
            "schema.TOOLCHAIN_FIELDS, not a key invented at the call site."
        )
    source = normalise_power_mode(block.get("toolchain_source"))
    if source not in CONDITION_SOURCES:
        raise ValueError(
            f"toolchain_source={block.get('toolchain_source')!r} is not one of "
            f"{list(CONDITION_SOURCES)}"
        )
    if source == "absent":
        raise ValueError(
            "toolchain_source='absent' is READER-ONLY -- it is what "
            "toolchain_from_manifest synthesises for a manifest older than the "
            "block, and writing it would claim a run predates a field it "
            "carries. Use 'unknown' if the toolchain could not be determined."
        )


def toolchain_differences(left: dict, right: dict) -> list[tuple[str, str, str]]:
    """``(field, left, right)`` for each IDENTITY field the two disagree on.

    Provenance fields are not compared -- two roots whose values were obtained
    differently are not two roots built differently.

    An ``unknown`` on either side is NOT reported as a difference here: "we do
    not know" is a different finding from "these differ", and the caller reports
    it separately so an operator is never told two roots disagree on evidence
    that does not exist.
    """
    out = []
    for name in TOOLCHAIN_IDENTITY_FIELDNAMES:
        a = normalise_power_mode(left.get(name))
        b = normalise_power_mode(right.get(name))
        if UNKNOWN_CONDITION in (a, b):
            continue
        if a != b:
            out.append((name, a, b))
    return out



def empty_timing() -> dict[str, object]:
    """A complete timing-contract block that claims nothing."""
    block: dict[str, object] = {
        name: UNKNOWN_CONDITION for name in TIMING_IDENTITY_FIELDNAMES
    }
    block["timing_source"] = UNKNOWN_CONDITION
    block["timing_detail"] = None
    return block


def timing_from_manifest(manifest: object) -> dict[str, object]:
    """The timing-contract block of a manifest dict, degrading to ``absent``.

    THE ONLY SUPPORTED READER, for ``toolchain_from_manifest``'s reason: no
    manifest written before `[2026-08-26]` has this key, and every one of them
    was measured under ``bind_and_start_wait`` without saying so. ``absent`` is
    what a reader gets, and it is NOT a match -- see the section comment.
    """
    if not isinstance(manifest, dict):
        block = None
    else:
        block = manifest.get(TIMING_KEY)

    out = empty_timing()
    if not isinstance(block, dict):
        out["timing_source"] = "absent"
        out["timing_detail"] = (
            "this manifest has no timing block -- it was written before the "
            "kernel_ms contract was recorded (queue item 19, finding 5). Its "
            "kernel_ms almost certainly includes the host-side xrt.run "
            "construction (the 'bind_and_start_wait' contract), but that is an "
            "inference from the file's age and must not be stamped into the "
            "data."
        )
        return out

    for name in TIMING_FIELDNAMES:
        if name in block:
            out[name] = block[name]
    for name in TIMING_IDENTITY_FIELDNAMES:
        out[name] = normalise_power_mode(out[name])
    out["timing_source"] = normalise_power_mode(out["timing_source"])
    return out


def validate_timing(block: dict[str, object]) -> None:
    """Raise ``ValueError`` unless ``block`` is a writable timing block.

    ``kernel_ms_contract`` has a CLOSED domain, unlike the toolchain's version
    strings: a contract this module has never heard of is a contract nothing can
    interpret, and recording it would put an uninterpretable value where a
    comparison guard reads. Adding one is a declaration in
    ``KERNEL_MS_CONTRACTS``.
    """
    expected = set(TIMING_FIELDNAMES)
    got = set(block)
    if missing := expected - got:
        raise ValueError(f"timing block is missing keys: {sorted(missing)}")
    if extra := got - expected:
        raise ValueError(
            f"timing block has keys not in the schema: {sorted(extra)}. Adding "
            "a timing fact is a declaration in schema.TIMING_FIELDS, not a key "
            "invented at the call site."
        )
    source = normalise_power_mode(block.get("timing_source"))
    if source not in CONDITION_SOURCES:
        raise ValueError(
            f"timing_source={block.get('timing_source')!r} is not one of "
            f"{list(CONDITION_SOURCES)}"
        )
    if source == "absent":
        raise ValueError(
            "timing_source='absent' is READER-ONLY -- it is what "
            "timing_from_manifest synthesises for a manifest older than the "
            "block, and writing it would claim a run predates a field it "
            "carries. Use 'unknown' if the contract could not be determined."
        )
    contract = normalise_power_mode(block.get("kernel_ms_contract"))
    if contract not in KERNEL_MS_CONTRACTS + (UNKNOWN_CONDITION,):
        raise ValueError(
            f"kernel_ms_contract={block.get('kernel_ms_contract')!r} is not one "
            f"of {list(KERNEL_MS_CONTRACTS)}. A contract name that is not "
            "declared cannot be interpreted by the guard that reads it."
        )
    cache = normalise_power_mode(block.get("xrt_run_cache"))
    if cache not in ("on", "off", UNKNOWN_CONDITION):
        raise ValueError(
            f"xrt_run_cache={block.get('xrt_run_cache')!r} is not 'on', 'off' "
            f"or {UNKNOWN_CONDITION!r}"
        )


def timing_differences(left: dict, right: dict) -> list[tuple[str, str, str]]:
    """``(field, left, right)`` for each IDENTITY field the two disagree on.

    An ``unknown`` on either side is NOT a difference -- "we do not know" is a
    different finding from "these differ", exactly as in
    ``toolchain_differences``, and the caller reports the two separately.
    """
    out = []
    for name in TIMING_IDENTITY_FIELDNAMES:
        a = normalise_power_mode(left.get(name))
        b = normalise_power_mode(right.get(name))
        if UNKNOWN_CONDITION in (a, b):
            continue
        if a != b:
            out.append((name, a, b))
    return out


# ---------------------------------------------------------------------------
# THE WALK BLOCK `[2026-08-12]` -- resume (doc 10 work item 8).
#
# WHAT IT IS FOR. A profile walk is long -- 45 min cold for `ladder`, hours for
# a matrix -- and this host is a laptop that suspends on a lid close. Resume
# makes an interrupted walk restartable without re-measuring the rungs that
# already produced rows. The moment it does, one CSV can hold rows measured in
# two sessions, hours or days apart, possibly across a reboot -- and NOTHING in
# a results row says which session it came from.
#
# That is the same defect the conditions block closed one level up: a fact the
# numbers depend on, living nowhere in the artifact. Two runs at different
# power modes are not a comparison; two HALVES OF ONE CSV measured at different
# toolchains are not one walk either, and until this block existed a resumed
# walk was indistinguishable from a single-session one.
#
# WHY A BLOCK AND NOT A `results` COLUMN -- item 15's decision, unchanged and
# for its reason. A `session_id` column is the obvious design and it is the
# wrong one: a new column bumps `SCHEMA_VERSION` to 3, `results_io.read_rows`
# rejects both a header and a version mismatch, and the bump that took 56 v1
# CSVs out of every reader on 08-10 would take the surviving v2 roots
# `compare_roots` is actually pointed at. So attribution lives OUT of the row,
# keyed by `(execution_mode, seq_len)`, and the cost of that choice is stated
# in `resume.py`: attribution is a claim ABOUT rows rather than a field IN them,
# which is exactly why every attributed rung carries a `row_digest`.
#
# THE DIGEST IS THE LOAD-BEARING PART, and it is what makes this block able to
# fail. Bookkeeping that only counts what the runner says it did cannot catch a
# runner that says the wrong thing -- G0's two closed defects were both checks
# that could not fail. A digest is evidence: a rung the ledger calls `reused`
# whose row no longer hashes to what was recorded was NOT reused, it was
# re-measured or edited, and either way the ledger is lying about the file.
#
# `absent` IS NOT `unknown`, for `toolchain_from_manifest`'s reason. Every root
# recorded before today has no ledger, and a walk block synthesised for one says
# `absent` -- "older than the field" -- and adds NO problems. Back-compatible by
# construction: a manifest built without a walk block is byte-identical to what
# it was, exactly as `expected_rows=None` leaves the counts unchecked.
#
# SCHEMA_VERSION STAYS 2. Not in `_FIELDS_BY_TABLE` either -- it is a JSON block,
# and anything iterating CSV tables must not be able to write it out as a
# one-row CSV. `test_schema.py` pins both.
# ---------------------------------------------------------------------------

#: How a walk's rows came to exist. `single_session` and `resumed` are the two
#: WRITABLE values; `absent` is reader-only, for a root older than the ledger.
WALK_SOURCES: tuple[str, ...] = (
    "single_session",
    "resumed",
    "unknown",
    "absent",
)

#: A session's own state. `interrupted` is not written by the session it
#: describes -- a killed process writes nothing -- it is what the NEXT session
#: relabels an unfinished predecessor as, which is the only party in a position
#: to observe that it never ended.
SESSION_STATUSES: tuple[str, ...] = ("running", "complete", "interrupted")

#: Where one rung's row came from, within a session.
#:
#:   measured  a child process ran and produced it in THIS session.
#:   reused    it was carried forward from an earlier session unchanged.
#:   skipped   the profile's applicability rule refused it; nothing ran.
#:
#: `reused` and `skipped` are deliberately distinct even though neither
#: dispatches: a skip is a claim about what the MODE supports and is re-derived
#: every session, while a reuse is a claim that an EARLIER MEASUREMENT still
#: stands. Collapsing them would make "we did not run this today" mean two
#: different things under one label.
RUNG_SOURCES: tuple[str, ...] = ("measured", "reused", "skipped")

#: One rung's entry in a session's ledger. `row_digest` is the evidence; the
#: rest is identity.
SESSION_RUNG_FIELDS: tuple[Field, ...] = (
    Field("execution_mode", "The row's `execution_mode` CSV value, not the code name."),
    Field("seq_len", "The rung's sequence length. With the mode, the LAYER row key."),
    Field(
        "model_key",
        "`[2026-08-23]` None for a layer rung. For a model rung, the list "
        "`[measurement_scope, model_id, phase, ubatch_tokens, "
        "context_end_tokens, precision_plan_id]` (`resume.MODEL_KEY_FIELDS`) "
        "that, with the mode and seq_len, is the row key: a decode rung at "
        "three contexts is three rows at seq_len 1, and a ledger keyed on the "
        "layer pair alone would attribute all three to one. A ledger written "
        "before this field reads back with None here (`Ledger.load` fills it), "
        "never a guess.",
    ),
    Field("source", "One of RUNG_SOURCES: how this session came by the row."),
    Field("run_status", "The row's `run_status`, copied so the ledger is readable alone."),
    Field(
        "row_digest",
        "Stable hash of every schema field of the row as this session left it. "
        "The check that can fail: a `reused` rung whose on-disk row no longer "
        "hashes to this was re-measured or edited behind the ledger.",
    ),
)

#: One walk session. Everything two sessions must agree on before their rows may
#: sit in one CSV is recorded here, per session, so the disagreement is visible.
SESSION_FIELDS: tuple[Field, ...] = (
    Field("session_id", "Monotonic within a results root: s001, s002, ..."),
    Field("profile", "The profile name this session walked. A session that walked a DIFFERENT profile into the same root is a splice of two plans, not a resume."),
    Field(
        "status",
        "One of SESSION_STATUSES. `interrupted` is the one that carries "
        "information: it is a session that started walking and never recorded "
        "an end, which is what a reboot mid-walk looks like from the outside.",
    ),
    Field("started_utc", "ISO-8601 UTC when the session began walking."),
    Field("ended_utc", "ISO-8601 UTC when it finished, or None if it never did."),
    Field("devq_job_id", "The `DEVQ_JOB_ID` this session ran under, or None off-queue. The log behind the session."),
    Field("git_sha", "HEAD at session start. Two sessions at different shas measured two trees."),
    Field("npu_power_mode", "The mode this session measured at. THE axis a splice must never cross -- README trap 0."),
    Field("toolchain_fingerprint", "The toolchain identity fields joined, so a splice across a rebuild is one string comparison."),
    Field("rungs", "List of SESSION_RUNG_FIELDS records, appended AS EACH RUNG COMPLETES so a killed session's attribution survives it."),
)

WALK_FIELDS: tuple[Field, ...] = (
    Field(
        "walk_source",
        "One of WALK_SOURCES: whether these rows came from one walk or several. "
        "`resumed` is not a defect and is not a warning -- it is the fact a "
        "reader needs before treating a CSV as one population.",
    ),
    Field("session_count", "How many sessions produced this root's rows."),
    Field("sessions", "The ledger: a list of SESSION_FIELDS records, oldest first."),
    Field("rungs_measured", "Rungs a child process actually ran, summed over sessions."),
    Field("rungs_reused", "Rungs carried forward from an earlier session."),
    Field(
        "rungs_unattributed",
        "Rows on disk that no session claims. Never zero by accident: a row "
        "nobody measured is a row whose provenance is unknown, and the whole "
        "point of the ledger is that there are none of those.",
    ),
    Field(
        "condition_splices",
        "Axes on which the measuring sessions disagree -- any of "
        "`npu_power_mode`, `toolchain`, `git_sha`. A pmode splice is a PROBLEM "
        "(compare_roots refuses one between roots; inside one CSV it is worse); "
        "the other two are FLAGGED, because resuming after a commit is normal "
        "and refusing it would make resume unusable.",
    ),
    Field(
        "attribution_problems",
        "Why this walk cannot be described honestly, if it cannot. Merged into "
        "the manifest's `incomplete_reasons`, so a lying ledger makes a run "
        "INCOMPLETE rather than merely annotated.",
    ),
    Field("walk_detail", "Prose provenance, or why a field is empty."),
)

#: The manifest key the walk block lives under.
WALK_KEY = "walk"

WALK_FIELDNAMES: tuple[str, ...] = tuple(f.name for f in WALK_FIELDS)
SESSION_FIELDNAMES: tuple[str, ...] = tuple(f.name for f in SESSION_FIELDS)
SESSION_RUNG_FIELDNAMES: tuple[str, ...] = tuple(f.name for f in SESSION_RUNG_FIELDS)


def empty_walk() -> dict[str, object]:
    """A complete walk block that claims nothing. ``empty_conditions``' rule."""
    return {
        "walk_source": UNKNOWN_CONDITION,
        "session_count": 0,
        "sessions": [],
        "rungs_measured": 0,
        "rungs_reused": 0,
        "rungs_unattributed": 0,
        "condition_splices": [],
        "attribution_problems": [],
        "walk_detail": None,
    }


def walk_from_manifest(manifest: object) -> dict[str, object]:
    """The walk block of a manifest dict, degrading to ``absent``.

    THE ONLY SUPPORTED READER, for ``toolchain_from_manifest``'s reason: no
    manifest written before `[2026-08-12]` carries this key, and a reader that
    indexed it would raise on the entire recorded corpus. ``absent`` means the
    root predates the ledger -- its rows may well have come from one session,
    but nothing recorded that, and a reader must not assume it.
    """
    block = manifest.get(WALK_KEY) if isinstance(manifest, dict) else None

    out = empty_walk()
    if not isinstance(block, dict):
        out["walk_source"] = "absent"
        out["walk_detail"] = (
            "this manifest has no walk block -- it was written before resume "
            "existed. Whether its rows came from one session is not recoverable "
            "from the files and must not be guessed."
        )
        return out
    for name in WALK_FIELDNAMES:
        if name in block:
            out[name] = block[name]
    out["walk_source"] = normalise_power_mode(out["walk_source"])
    return out


def validate_session(record: dict[str, object]) -> None:
    """Raise ``ValueError`` unless ``record`` is a writable session record."""
    expected = set(SESSION_FIELDNAMES)
    got = set(record)
    if missing := expected - got:
        raise ValueError(f"session record is missing keys: {sorted(missing)}")
    if extra := got - expected:
        raise ValueError(
            f"session record has keys not in the schema: {sorted(extra)}. "
            "Adding a session fact is a declaration in schema.SESSION_FIELDS, "
            "not a key invented at the call site."
        )
    if record["status"] not in SESSION_STATUSES:
        raise ValueError(
            f"session status={record['status']!r} is not one of "
            f"{list(SESSION_STATUSES)}"
        )
    rungs = record["rungs"]
    if not isinstance(rungs, list):
        raise ValueError(f"session `rungs` must be a list, got {type(rungs).__name__}")
    for i, rung in enumerate(rungs):
        if not isinstance(rung, dict) or set(rung) != set(SESSION_RUNG_FIELDNAMES):
            raise ValueError(
                f"session rung {i} does not match SESSION_RUNG_FIELDS: "
                f"{sorted(rung) if isinstance(rung, dict) else rung}"
            )
        if rung["source"] not in RUNG_SOURCES:
            raise ValueError(
                f"session rung {i} source={rung['source']!r} is not one of "
                f"{list(RUNG_SOURCES)}"
            )


def validate_walk(block: dict[str, object]) -> None:
    """Raise ``ValueError`` unless ``block`` is a writable walk block.

    Same shape of check as ``validate_conditions``, plus every session record.
    ``absent`` is refused for its reason: it is what ``walk_from_manifest``
    synthesises for a root older than the ledger, and writing it would claim a
    run predates a field it carries.
    """
    expected = set(WALK_FIELDNAMES)
    got = set(block)
    if missing := expected - got:
        raise ValueError(f"walk block is missing keys: {sorted(missing)}")
    if extra := got - expected:
        raise ValueError(
            f"walk block has keys not in the schema: {sorted(extra)}. Adding a "
            "walk fact is a declaration in schema.WALK_FIELDS, not a key "
            "invented at the call site."
        )
    source = normalise_power_mode(block.get("walk_source"))
    if source not in WALK_SOURCES:
        raise ValueError(
            f"walk_source={block.get('walk_source')!r} is not one of "
            f"{list(WALK_SOURCES)}"
        )
    if source == "absent":
        raise ValueError(
            "walk_source='absent' is READER-ONLY -- it is what "
            "walk_from_manifest synthesises for a root older than the ledger, "
            "and writing it would claim a run predates a field it carries."
        )
    for record in block["sessions"]:
        validate_session(record)
