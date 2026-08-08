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

TABLES = ("results", "tuning")

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
