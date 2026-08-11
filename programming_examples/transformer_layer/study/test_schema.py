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


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"schema tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
