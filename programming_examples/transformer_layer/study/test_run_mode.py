# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only checks on the parts of ``run_mode`` that need no hardware.

WHY THIS MODULE EXISTS
    `[2026-08-09]` The packaging counters a mode reports were copied into the
    results row and gated by NOTHING. ``run_npu2_offload_peano.lit`` FileChecks
    console output, and the counters only ever reach the CSV -- so a regression
    that blanked ``npu_unique_xclbin_count``, or that collapsed the ELF path's
    0 and the shared path's 1 to one value, stayed green everywhere. That column
    is what separates the two arms of the variance measurement, so losing it
    silently would make two result sets indistinguishable after the fact.

    The copy is therefore a named function rather than a loop inline in
    ``run()``, which needs an NPU and cannot be exercised here.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_mode  # noqa: E402
import schema  # noqa: E402


def test_counters_a_mode_reports_reach_their_columns():
    row = schema.empty_row("results")
    run_mode._apply_mode_counters(row, {"unique_xclbins": 1, "n_dispatches": 30})
    assert row["npu_unique_xclbin_count"] == 1
    assert row["npu_dispatch_count"] == 30


def test_a_reported_zero_survives():
    """The ELF path loads NO xclbin, so 0 is its honest value -- not a blank.

    A truthiness test here would drop it and make the two packaging paths
    indistinguishable in the CSV, which is the exact failure this column was
    populated to prevent.
    """
    row = schema.empty_row("results")
    run_mode._apply_mode_counters(row, {"unique_xclbins": 0, "n_dispatches": 30})
    assert row["npu_unique_xclbin_count"] == 0, "a reported zero was dropped"


def test_a_mode_reporting_nothing_leaves_the_columns_unset():
    """Absent is None, never a guess derived from some other field."""
    row = schema.empty_row("results")
    run_mode._apply_mode_counters(row, {})
    assert row["npu_unique_xclbin_count"] is None
    assert row["npu_dispatch_count"] is None


def test_the_two_packaging_paths_do_not_collide():
    """The separation the variance measurement's provenance rests on.

    Asserts the VALUES, not merely that they differ. `!=` alone passes when one
    side degrades to ``None`` -- which is exactly what a truthiness bug does to
    the ELF path's 0 -- so it would report a separation that no longer carries
    the meaning the walks depend on.
    """
    elf, shared = schema.empty_row("results"), schema.empty_row("results")
    run_mode._apply_mode_counters(elf, {"unique_xclbins": 0, "n_dispatches": 30})
    run_mode._apply_mode_counters(shared, {"unique_xclbins": 1, "n_dispatches": 30})
    assert elf["npu_unique_xclbin_count"] == 0, "the ELF path must record 0"
    assert shared["npu_unique_xclbin_count"] == 1, "the shared path must record 1"


def test_a_filled_row_still_validates_against_schema_v1():
    """No version bump was taken, so the columns must already be v1 columns."""
    row = schema.empty_row("results")
    row["execution_mode"] = "offload"
    row["run_status"] = "passed"
    run_mode._apply_mode_counters(row, {"unique_xclbins": 1, "n_dispatches": 30})
    schema.validate_row(row, "results")


def test_every_counter_column_is_a_real_schema_column():
    """A typo in _MODE_COUNTERS would otherwise write a column nothing reads."""
    for _key, column in run_mode._MODE_COUNTERS:
        assert column in schema.RESULTS_FIELDNAMES, f"{column} is not in the schema"
