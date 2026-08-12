# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only checks that the study schema means what it says.

    python3 study/test_schema.py

No hardware, no MLIR, no toolchain, well under a second. Plain ``test_*``
functions with a ``main()`` runner, matching ``pattern/test_reference.py``:
that runs anywhere, and ordinary pytest discovery still finds it (porting
convention 11). It deliberately does NOT import pytest -- pytest is not
installed in this environment, and Phase F work item 6 (pin the missing
dependencies) is not a prerequisite for testing a pure-Python catalogue.

WHAT THESE ARE FOR
    The schema's value is that every field carries its meaning and every timed
    field carries its region. Prose rots; these make it fail loudly instead.
    The load-bearing one is ``test_timed_fields_declare_their_region``: doc 03's
    whole objection to porting iron's schema verbatim was that a column name
    does not say what is inside the clock, so a timing field with no region is
    precisely the defect this module exists to prevent.

    They also pin two conventions that are easy to undo by accident:
    ``fused_elf`` staying a VALUE rather than becoming a column, and the CSV
    value for ``coarse`` being ``hybrid`` (porting convention 7) -- which this
    module originally had backwards.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import schema  # noqa: E402

# `[2026-08-11]` `resource` joins the two measurement tables: one row per
# COMPILED ARTIFACT (study/resource_usage.py), which is a different unit from a
# measurement and so a third table rather than columns on `results`. The generic
# checks below -- every field documented, no duplicate names, timing declared
# where the name implies a duration -- apply to it unchanged, which is the point
# of driving them off this tuple.
TABLES = ("results", "tuning", "resource")

# Names whose value is a duration or a rate, so ``Field.timing`` is mandatory.
_TIMED_SUFFIXES = ("_ms", "_sec", "_gflops_per_sec", "_gflops_per_sec_per_watt")
# Counts that merely LOOK timed; they carry an explicit "n/a" rather than None.
_ALSO_TIMED = (
    "warmup_runs",
    "runs_per_sample",
    "measured_inference_count",
    "latency_sample_count",
)


def _raises(exc, match, fn, *args, **kwargs):
    """Assert ``fn`` raises ``exc`` whose message contains ``match``."""
    try:
        fn(*args, **kwargs)
    except exc as e:
        assert match in str(e), f"expected {match!r} in {str(e)!r}"
        return
    raise AssertionError(f"expected {exc.__name__} containing {match!r}")


def test_timed_fields_declare_their_region():
    """Anything measuring time must say what the clock covers."""
    for table in TABLES:
        for field in schema.fields_for(table):
            if field.name.endswith(_TIMED_SUFFIXES) or field.name in _ALSO_TIMED:
                assert field.timing, (
                    f"{table}.{field.name} is a timing field with no declared "
                    "region. A column name does not say what is inside the "
                    "clock; that is the defect this schema exists to prevent."
                )


def test_every_field_has_a_meaning():
    for table in TABLES:
        for field in schema.fields_for(table):
            assert field.meaning.strip(), f"{table}.{field.name} has no meaning"


def test_field_names_are_unique():
    for table in TABLES:
        names = [f.name for f in schema.fields_for(table)]
        assert len(names) == len(set(names)), f"duplicate column in {table}"


def test_fused_elf_is_a_value_not_a_column():
    """Adding a mode must not add a column, or old rows stop being readable."""
    assert "fused_elf" in schema.EXECUTION_MODES
    assert not any("fused" in name for name in schema.RESULTS_FIELDNAMES)


def test_csv_value_for_coarse_is_hybrid():
    """Convention 7: code name `coarse`, CSV value `hybrid`, on purpose.

    Getting this backwards makes the schema reject every row the shipped modes
    write -- their recorded artifacts carry execution_mode='hybrid'.
    """
    assert schema.EXECUTION_MODE_CSV["coarse"] == "hybrid"
    assert "hybrid" in schema.EXECUTION_MODES
    assert "coarse" not in schema.EXECUTION_MODES


def test_mode_mapping_agrees_with_the_pattern_package():
    """The mapping is duplicated in pattern/__init__.py; pin them equal.

    Convention 7 wants it in the schema module; that one predates this file and
    the shipped modes read from it. Until the duplication is closed by pointing
    that one here, drift between them is the failure to catch.
    """
    import os, sys

    example = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if example not in sys.path:
        sys.path.insert(0, example)
    from pattern import EXECUTION_MODE_CSV as shipped

    assert shipped == schema.EXECUTION_MODE_CSV, (
        f"mode->CSV mapping drifted: pattern/__init__.py has {shipped}, "
        f"schema.py has {schema.EXECUTION_MODE_CSV}"
    )


def test_quant_fields_present_from_v1():
    """Doc 03: fold them in now, empty for bf16, rather than renumbering later."""
    assert schema.QUANT_FIELDNAMES
    row = schema.empty_row()
    for name in schema.QUANT_FIELDNAMES:
        assert name in row and row[name] is None


def test_dispatch_vector_is_all_six_fields():
    assert len(schema.DISPATCH_VECTOR_FIELDNAMES) == 6
    for name in schema.DISPATCH_VECTOR_FIELDNAMES:
        assert name in schema.RESULTS_FIELDNAMES


# Every v1 results column, in v1 order. Frozen HERE, independently of schema.py,
# so an edit over there that renames, drops or reorders a v1 column fails this
# file rather than silently reshaping every existing results tree. The v2
# additions are appended strictly AFTER this prefix (the section comment in
# schema.py and the test below).
_V1_RESULTS_FIELDNAMES = (
    "schema_version", "study_id", "study_case_id", "study_case_label",
    "workload_variant", "backend", "execution_mode", "attention_path",
    "seq_len", "hidden_size", "intermediate_size", "num_attention_heads",
    "attention_head_size", "batch_size", "dtype", "use_bias", "weights_source",
    "warmup_runs", "runs_per_sample", "measured_inference_count",
    "latency_sample_count", "timed_total_sec", "avg_latency_ms",
    "min_latency_ms", "max_latency_ms", "compile_setup_time_ms",
    "host_qkv_precompute_ms", "effective_gflops_per_sec",
    "host_submissions_per_layer", "runlist_entries_per_submission",
    "air_launches_per_elf", "herd_launches", "sync_boundaries",
    "bytes_transferred", "power_backend", "avg_power_w", "min_power_w",
    "max_power_w", "power_sample_count", "power_std_w", "raw_avg_power_w",
    "raw_min_power_w", "raw_max_power_w", "raw_power_sample_count",
    "raw_power_std_w", "power_outlier_sample_count",
    "power_outlier_filter_applied", "effective_gflops_per_sec_per_watt",
    "quant_packing_scheme", "quant_group_size", "quant_scale_layout",
    "quant_zero_point_layout", "quant_accum_type", "quant_gemm_contract",
    "quant_gemv_contract", "validation_error_count", "run_status",
    "failure_message", "process_model", "npu_dispatch_count",
    "npu_unique_instruction_binary_count", "npu_unique_xclbin_count",
    "selected_config_json", "selected_candidate_ids_json", "is_best",
)


def test_v2_keeps_every_v1_column_first_and_unchanged():
    """The bump is ADDITIVE: v1 names, meanings-by-name and POSITIONS survive.

    Position matters because the CSV header is the field order, and anything
    that read a v1 file by column index must read a v2 file the same way. So
    the v1 columns are pinned as an exact ordered PREFIX, not as a subset.
    """
    prefix = schema.RESULTS_FIELDNAMES[: len(_V1_RESULTS_FIELDNAMES)]
    assert prefix == _V1_RESULTS_FIELDNAMES, (
        "a v1 results column was renamed, dropped or reordered; the v2 bump "
        "is additive-only and new columns go AFTER every v1 one"
    )


def test_v2_appends_the_decomposition_then_the_reconfiguration_columns():
    """The five v2 columns, in order, after every v1 column and nowhere else."""
    assert schema.SCHEMA_VERSION == 2
    assert schema.DECOMPOSITION_FIELDNAMES == ("device_ms", "sync_ms", "host_cpu_ms")
    assert schema.RECONFIGURATION_FIELDNAMES == ("context_loads", "kernel_attaches")
    suffix = schema.DECOMPOSITION_FIELDNAMES + schema.RECONFIGURATION_FIELDNAMES
    assert schema.RESULTS_FIELDNAMES[-len(suffix):] == suffix
    for name in suffix:
        assert name not in _V1_RESULTS_FIELDNAMES


def test_the_decomposition_components_declare_disjoint_regions():
    """The three ms columns are one decomposition, not three clocks.

    Each must say it sits INSIDE the latency region -- the same region
    avg_latency_ms covers -- or a reader cannot know whether the components
    may be compared against the total they decompose.
    """
    by_name = {f.name: f for f in schema.fields_for("results")}
    for name in schema.DECOMPOSITION_FIELDNAMES:
        assert "INSIDE the latency region" in by_name[name].timing, (
            f"{name} does not place itself inside the latency region"
        )


def test_the_reconfiguration_columns_state_the_steady_state_convention():
    """context_loads/kernel_attaches record ONE steady-state dispatch.

    The cumulative counter the offload gate pins is a different quantity, and
    the meaning must say so -- that distinction is what makes offload-ELF's 30
    and a warmed shared-path row's 0 both correct at once.
    """
    by_name = {f.name: f for f in schema.fields_for("results")}
    for name in schema.RECONFIGURATION_FIELDNAMES:
        assert "steady-state" in by_name[name].meaning
    assert "eviction followed by a reload counts AGAIN" in (
        by_name["context_loads"].meaning
    )


def test_empty_row_is_complete_and_versioned():
    row = schema.empty_row()
    assert set(row) == set(schema.RESULTS_FIELDNAMES)
    assert row["schema_version"] == schema.SCHEMA_VERSION


def test_a_failed_measurement_is_still_a_valid_row():
    """The Phase F gate's premise: a failed run writes a complete, valid row."""
    row = schema.empty_row()
    row["execution_mode"] = "hybrid"
    row["run_status"] = "failed"
    row["failure_message"] = "ERT_CMD_STATE_TIMEOUT"
    schema.validate_row(row)  # must not raise despite every metric being None


def test_validate_rejects_a_missing_column():
    row = schema.empty_row()
    del row["avg_latency_ms"]
    _raises(ValueError, "missing columns", schema.validate_row, row)


def test_validate_rejects_an_unknown_column():
    row = schema.empty_row()
    row["latency_ms"] = 1.0
    _raises(ValueError, "not in schema", schema.validate_row, row)


def test_validate_rejects_a_foreign_schema_version():
    row = schema.empty_row()
    row["schema_version"] = schema.SCHEMA_VERSION + 1
    _raises(ValueError, "use the adapter", schema.validate_row, row)


def test_validate_names_the_csv_value_when_given_a_code_name():
    """Writing the CODE name `coarse` is the mistake; say what to write."""
    row = schema.empty_row()
    row["execution_mode"] = "coarse"
    _raises(ValueError, "its CSV value is 'hybrid'", schema.validate_row, row)


def test_validate_rejects_an_unknown_run_status():
    row = schema.empty_row()
    row["run_status"] = "ok"
    _raises(ValueError, "run_status", schema.validate_row, row)


def test_attention_path_is_checked_because_it_is_not_derivable():
    row = schema.empty_row()
    row["attention_path"] = "somewhere"
    _raises(ValueError, "attention_path", schema.validate_row, row)
    row["attention_path"] = "host_torch"
    schema.validate_row(row)


def test_tuning_table_is_independently_valid():
    row = schema.empty_row("tuning")
    row["execution_mode"] = "fused_elf"
    row["run_status"] = "passed"
    schema.validate_row(row, "tuning")
    assert set(row) == set(schema.TUNING_FIELDNAMES)


def test_unknown_table_raises():
    _raises(ValueError, "unknown table", schema.fields_for, "results_v2")


# ---------------------------------------------------------------------------
# The measurement-condition block `[2026-08-12]` (doc 34 M4).
#
# The load-bearing one is test_the_conditions_block_is_not_a_csv_table: the
# whole reason this is a manifest block and not a `results` column is that a
# column would bump SCHEMA_VERSION, and a bump makes every recorded CSV
# unreadable -- v1 -> v2 did exactly that to 56 of them. If someone later
# registers it as a table, results_io will happily build a header out of it and
# the next person to add a condition will bump the version to ship it.
# ---------------------------------------------------------------------------


def test_the_conditions_block_is_not_a_csv_table():
    _raises(ValueError, "unknown table", schema.fields_for, "conditions")
    for name in schema.CONDITION_FIELDNAMES:
        assert name not in schema.RESULTS_FIELDNAMES, name
        assert name not in schema.TUNING_FIELDNAMES, name


def test_recording_the_condition_did_not_bump_the_schema_version():
    """The point of the design: every recorded v2 CSV still reads."""
    assert schema.SCHEMA_VERSION == 2
    prefix = schema.RESULTS_FIELDNAMES[: len(_V1_RESULTS_FIELDNAMES)]
    assert prefix == _V1_RESULTS_FIELDNAMES


def test_every_condition_field_has_a_meaning():
    for field in schema.CONDITION_FIELDS:
        assert field.meaning.strip(), field.name
    names = [f.name for f in schema.CONDITION_FIELDS]
    assert len(names) == len(set(names))


def test_empty_conditions_claims_nothing():
    block = schema.empty_conditions()
    assert set(block) == set(schema.CONDITION_FIELDNAMES)
    assert block["npu_power_mode"] == schema.UNKNOWN_CONDITION
    assert block["npu_power_mode_source"] == schema.UNKNOWN_CONDITION
    schema.validate_conditions(block)


def test_normalise_power_mode_collapses_every_way_of_saying_nothing():
    for nothing in (None, "", "   ", "unknown", "UNKNOWN"):
        assert schema.normalise_power_mode(nothing) == schema.UNKNOWN_CONDITION
    assert schema.normalise_power_mode("Turbo") == "turbo"
    assert schema.normalise_power_mode(" Default ") == "default"


def test_a_manifest_predating_the_block_reads_as_absent_not_as_a_match():
    """Every recorded root is in this state and none of them can be stamped."""
    v1_manifest = {"schema_version": 1, "complete": True, "git": {}}
    for older in (v1_manifest, {}, None, "not a manifest", []):
        block = schema.conditions_from_manifest(older)
        assert set(block) == set(schema.CONDITION_FIELDNAMES)
        assert block["npu_power_mode"] == schema.UNKNOWN_CONDITION
        assert block["npu_power_mode_source"] == "absent"
        assert "not recoverable" in block["npu_power_mode_detail"]


def test_a_recorded_block_is_read_back_normalised():
    block = schema.conditions_from_manifest(
        {"conditions": {"npu_power_mode": "Turbo", "npu_power_mode_source": "observed"}}
    )
    assert block["npu_power_mode"] == "turbo"
    assert block["npu_power_mode_source"] == "observed"
    # A partial block still comes back complete, so no reader indexes a missing key.
    assert set(block) == set(schema.CONDITION_FIELDNAMES)


def test_a_recorded_but_undeterminable_mode_is_unknown_not_absent():
    """`older than the field` and `tried and failed` are different facts."""
    block = schema.conditions_from_manifest(
        {"conditions": schema.empty_conditions()}
    )
    assert block["npu_power_mode"] == schema.UNKNOWN_CONDITION
    assert block["npu_power_mode_source"] == schema.UNKNOWN_CONDITION


def test_validate_conditions_rejects_a_missing_or_invented_key():
    block = schema.empty_conditions()
    del block["observed_at_utc"]
    _raises(ValueError, "missing keys", schema.validate_conditions, block)
    block = schema.empty_conditions()
    block["npu_pmode"] = "turbo"
    _raises(ValueError, "not in the schema", schema.validate_conditions, block)


def test_validate_conditions_rejects_an_unknown_source():
    block = schema.empty_conditions()
    block["npu_power_mode_source"] = "probably"
    _raises(ValueError, "npu_power_mode_source", schema.validate_conditions, block)


def test_absent_is_reader_only_and_cannot_be_written():
    """Writing it would claim a run predates a field it is carrying."""
    block = schema.empty_conditions()
    block["npu_power_mode_source"] = "absent"
    _raises(ValueError, "READER-ONLY", schema.validate_conditions, block)


def test_the_mode_domain_is_open_on_purpose():
    """xrt-smi names the modes; a schema that rejected a new one would hide it."""
    block = schema.empty_conditions()
    block["npu_power_mode"] = "some_future_mode"
    block["npu_power_mode_source"] = "observed"
    schema.validate_conditions(block)  # must not raise


# ---------------------------------------------------------------------------
# The toolchain block `[2026-08-12]` -- queue item 16.
# ---------------------------------------------------------------------------


def test_the_toolchain_key_is_the_one_compare_manifests_diffs():
    """THE load-bearing assertion of item 16.

    ``compare_roots.compare_manifests`` iterates ``manifest["toolchain"]`` by
    that literal name and always has. The defect was that nothing wrote it. If
    this key is ever renamed, the diff silently goes back to iterating an empty
    dict and the toolchain half of every comparison compares nothing again --
    with no test failing anywhere else, because an empty diff is a quiet one.
    """
    assert schema.TOOLCHAIN_KEY == "toolchain", schema.TOOLCHAIN_KEY
    source = open(os.path.join(_HERE, "compare_roots.py"), encoding="utf-8").read()
    assert '("toolchain", "toolchain")' in source, (
        "compare_manifests no longer diffs a block named `toolchain`; "
        "TOOLCHAIN_KEY exists to match it"
    )


def test_the_toolchain_block_did_not_bump_the_schema_version():
    """Item 15's precedent, and the reason the block is not a results column.

    A column bump makes ``results_io.read_rows`` reject every recorded CSV on
    the version check -- it took 56 v1 files out of every reader on 08-10, and
    would take the 16 v2 files that survive, which are the roots compare_roots
    is pointed at. ``RESOURCE_FIELDS`` states the rule: adding a table is not a
    version bump.
    """
    assert schema.SCHEMA_VERSION == 2, schema.SCHEMA_VERSION


def test_the_toolchain_block_is_not_a_csv_table():
    """It must not be reachable by anything that iterates CSV tables.

    Registered in ``_FIELDS_BY_TABLE`` it could be written out as a one-row CSV
    by any table-walking caller. Same pin as the conditions block has.
    """
    _raises(ValueError, "unknown table", schema.fields_for, "toolchain")


def test_every_toolchain_field_is_documented_and_unique():
    names = list(schema.TOOLCHAIN_FIELDNAMES)
    assert len(names) == len(set(names)), names
    for field in schema.TOOLCHAIN_FIELDS + schema.TOOLCHAIN_PROVENANCE_FIELDS:
        assert field.meaning.strip(), field.name


def test_empty_toolchain_is_complete_and_claims_nothing():
    block = schema.empty_toolchain()
    assert set(block) == set(schema.TOOLCHAIN_FIELDNAMES)
    for name in schema.TOOLCHAIN_IDENTITY_FIELDNAMES:
        assert block[name] == schema.UNKNOWN_CONDITION, name
    schema.validate_toolchain(block)  # a claim-nothing block is still writable


def test_a_manifest_older_than_the_block_reads_back_as_absent():
    """Not a crash, and not a quiet match. Every recorded root is in this state.

    In fact EVERY manifest ever written is: the block's reader was in
    compare_roots before its writer was in manifest.py.
    """
    out = schema.toolchain_from_manifest({"study_id": "x"})
    assert out["toolchain_source"] == "absent", out
    assert "must not be guessed" in out["toolchain_detail"]
    for name in schema.TOOLCHAIN_IDENTITY_FIELDNAMES:
        assert out[name] == schema.UNKNOWN_CONDITION, name


def test_a_non_mapping_manifest_reads_back_as_absent_too():
    for junk in (None, [], "toolchain", 7):
        assert schema.toolchain_from_manifest(junk)["toolchain_source"] == "absent"


def test_two_roots_recording_nothing_are_not_two_roots_that_agree():
    """The trap this block exists to avoid, stated as a test.

    Both sides `absent` must produce NO reported difference AND must not be
    readable as a match -- the caller distinguishes them by the source field,
    never by an empty difference list.
    """
    left = schema.toolchain_from_manifest({})
    right = schema.toolchain_from_manifest({})
    assert schema.toolchain_differences(left, right) == []
    assert left["toolchain_source"] == right["toolchain_source"] == "absent"


def test_toolchain_differences_reports_only_identity_fields():
    """Provenance differing is not the toolchain differing."""
    left = schema.empty_toolchain()
    right = schema.empty_toolchain()
    for name in schema.TOOLCHAIN_IDENTITY_FIELDNAMES:
        left[name] = right[name] = "same"
    left["toolchain_source"] = "observed"
    right["toolchain_source"] = "probed_at_manifest_build"
    left["toolchain_detail"] = "read from a file"
    right["toolchain_detail"] = "read from somewhere else"
    assert schema.toolchain_differences(left, right) == []

    right["xrt_version"] = "2.20.0"
    assert schema.toolchain_differences(left, right) == [
        ("xrt_version", "same", "2.20.0")
    ]


def test_an_unknown_on_either_side_is_not_a_difference():
    """"We do not know" and "these differ" are different findings.

    Reporting an unknown as a difference would tell an operator two roots
    disagree on evidence that does not exist.
    """
    left = schema.empty_toolchain()
    right = schema.empty_toolchain()
    left["xrt_version"] = "2.21.0"
    # right's stays `unknown`
    assert schema.toolchain_differences(left, right) == []
    assert schema.toolchain_differences(right, left) == []


def test_validate_toolchain_rejects_a_missing_or_invented_key():
    block = schema.empty_toolchain()
    del block["xrt_version"]
    _raises(ValueError, "missing keys", schema.validate_toolchain, block)

    block = schema.empty_toolchain()
    block["gcc_version"] = "13"
    _raises(ValueError, "not in the schema", schema.validate_toolchain, block)


def test_validate_toolchain_rejects_a_bad_source_and_reader_only_absent():
    block = schema.empty_toolchain()
    block["toolchain_source"] = "probably"
    _raises(ValueError, "toolchain_source", schema.validate_toolchain, block)

    block = schema.empty_toolchain()
    block["toolchain_source"] = "absent"
    _raises(ValueError, "READER-ONLY", schema.validate_toolchain, block)


def test_the_toolchain_value_domains_are_open_on_purpose():
    """A wheel may take a version string this module has never seen."""
    block = schema.empty_toolchain()
    block["xrt_version"] = "9.9.9+deadbeefcafe"
    block["peano_version"] = "some-future-pin"
    block["air_resolution"] = "/an/unrecognised/tree"
    block["toolchain_source"] = "probed_at_manifest_build"
    schema.validate_toolchain(block)  # must not raise


def test_the_toolchain_source_domain_is_shared_with_conditions():
    """One enum, not two saying the same four things."""
    for value in schema.CONDITION_SOURCES:
        block = schema.empty_toolchain()
        block["toolchain_source"] = value
        if value == "absent":
            _raises(ValueError, "READER-ONLY", schema.validate_toolchain, block)
        else:
            schema.validate_toolchain(block)


# ---------------------------------------------------------------------------
# The WALK block `[2026-08-12]` -- resume. Same three pins as the toolchain
# block, because the same mistake would cost the same thing.
# ---------------------------------------------------------------------------


def test_the_walk_block_did_not_bump_the_schema_version():
    """The decisive one. A `session_id` COLUMN is the obvious design for
    attribution and is the wrong one: it bumps SCHEMA_VERSION to 3, and
    `results_io.read_rows` rejects both a header and a version mismatch, so it
    would take every surviving v2 root out of every reader -- including the ones
    `compare_roots` is pointed at. Item 15's decision, unchanged."""
    assert schema.SCHEMA_VERSION == 2
    assert "session_id" not in schema.RESULTS_FIELDNAMES
    assert "walk_session" not in schema.RESULTS_FIELDNAMES


def test_the_walk_block_is_not_a_csv_table():
    """Anything iterating CSV tables must not be able to write it out as one."""
    _raises(ValueError, "unknown table", schema.fields_for, schema.WALK_KEY)
    _raises(ValueError, "unknown table", schema.fields_for, "sessions")


def test_empty_walk_is_complete_and_claims_nothing():
    block = schema.empty_walk()
    assert set(block) == set(schema.WALK_FIELDNAMES)
    assert block["walk_source"] == schema.UNKNOWN_CONDITION
    assert block["sessions"] == []
    assert block["attribution_problems"] == []


def test_a_manifest_older_than_the_walk_block_reads_back_as_absent():
    """`absent` (older than the field) and `unknown` (tried and failed) are
    different things to tell an operator, and neither is a match."""
    assert schema.walk_from_manifest({})["walk_source"] == "absent"
    assert schema.walk_from_manifest(None)["walk_source"] == "absent"
    assert "must not be guessed" in schema.walk_from_manifest({})["walk_detail"]


def test_validate_walk_rejects_a_missing_or_invented_key():
    block = schema.empty_walk()
    block["walk_source"] = "single_session"
    schema.validate_walk(block)

    del block["rungs_reused"]
    _raises(ValueError, "missing keys", schema.validate_walk, block)

    block = schema.empty_walk()
    block["walk_source"] = "single_session"
    block["rungs_reusd"] = 0
    _raises(ValueError, "not in the schema", schema.validate_walk, block)


def test_validate_walk_rejects_a_bad_source_and_reader_only_absent():
    block = schema.empty_walk()
    block["walk_source"] = "partially"
    _raises(ValueError, "walk_source", schema.validate_walk, block)

    block["walk_source"] = "absent"
    _raises(ValueError, "READER-ONLY", schema.validate_walk, block)


def test_validate_session_rejects_a_malformed_rung():
    """A hand-assembled ledger with a typo'd key must fail here rather than be
    read back as a session that recorded nothing."""
    base = {name: None for name in schema.SESSION_FIELDNAMES}
    base["status"] = "complete"
    base["rungs"] = []
    schema.validate_session(base)

    bad = dict(base, status="finished")
    _raises(ValueError, "session status", schema.validate_session, bad)

    bad = dict(base, rungs=[{"execution_mode": "hybrid"}])
    _raises(ValueError, "SESSION_RUNG_FIELDS", schema.validate_session, bad)

    rung = {name: None for name in schema.SESSION_RUNG_FIELDNAMES}
    rung["source"] = "carried"
    _raises(ValueError, "not one of", schema.validate_session, dict(base, rungs=[rung]))

    rung["source"] = "reused"
    schema.validate_session(dict(base, rungs=[rung]))

    bad = dict(base)
    bad["extra"] = 1
    _raises(ValueError, "not in the schema", schema.validate_session, bad)


def test_reused_and_skipped_are_distinct_rung_sources():
    """Neither dispatches, and collapsing them would make "we did not run this
    today" mean two different things under one label: a skip is a claim about
    what the MODE supports, a reuse that an EARLIER MEASUREMENT still stands."""
    assert set(schema.RUNG_SOURCES) == {"measured", "reused", "skipped"}
    assert set(schema.SESSION_STATUSES) == {"running", "complete", "interrupted"}


def test_every_walk_field_is_documented_and_unique():
    for fields in (schema.WALK_FIELDS, schema.SESSION_FIELDS, schema.SESSION_RUNG_FIELDS):
        names = [f.name for f in fields]
        assert len(names) == len(set(names)), names
        for f in fields:
            assert f.meaning and len(f.meaning) > 20, f.name


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"schema tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
