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

SCHEMA_VERSION = 1


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

RESULTS_FIELDS: tuple[Field, ...] = (
    *_IDENTITY,
    *_SHAPE,
    *_TIMING,
    *_DISPATCH,
    *_POWER,
    *_QUANT,
    *_OUTCOME,
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

RESULTS_FIELDNAMES: tuple[str, ...] = tuple(f.name for f in RESULTS_FIELDS)
TUNING_FIELDNAMES: tuple[str, ...] = tuple(f.name for f in TUNING_FIELDS)

DISPATCH_VECTOR_FIELDNAMES: tuple[str, ...] = tuple(f.name for f in _DISPATCH)
POWER_FIELDNAMES: tuple[str, ...] = tuple(f.name for f in _POWER)
QUANT_FIELDNAMES: tuple[str, ...] = tuple(f.name for f in _QUANT)

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
EXECUTION_MODE_CSV: dict[str, str] = {
    "coarse": "hybrid",
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
}


def fields_for(table: str) -> tuple[Field, ...]:
    """The Field tuple for ``results`` or ``tuning``. Raises on anything else."""
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
