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


def test_a_filled_row_still_validates_against_the_schema():
    """Every counter lands in a declared column of the CURRENT schema (v2)."""
    row = schema.empty_row("results")
    row["execution_mode"] = "offload"
    row["run_status"] = "passed"
    run_mode._apply_mode_counters(
        row,
        {
            "unique_xclbins": 1,
            "n_dispatches": 30,
            "context_loads": 30,
            "kernel_attaches": 0,
        },
    )
    schema.validate_row(row, "results")


def test_every_counter_column_is_a_real_schema_column():
    """A typo in _MODE_COUNTERS would otherwise write a column nothing reads."""
    for _key, column in run_mode._MODE_COUNTERS:
        assert column in schema.RESULTS_FIELDNAMES, f"{column} is not in the schema"


# --- the v2 reconfiguration columns -----------------------------------------


def test_reconfiguration_counters_reach_their_columns():
    """The mode reports one dispatch's delta; the row records it verbatim.

    30/0 is the offload-ELF known truth: a fresh hw_context per GEMM dispatch,
    thirty loads per steady-state layer, zero attaches.
    """
    row = schema.empty_row("results")
    run_mode._apply_mode_counters(row, {"context_loads": 30, "kernel_attaches": 0})
    assert row["context_loads"] == 30
    assert row["kernel_attaches"] == 0, "a reported zero was dropped"


def test_a_standing_context_modes_zero_loads_survive():
    """0/0 is a warmed shared-xclbin (or fused/coarse) row's honest value.

    The one configuration was loaded in warmup and stands; recording None
    there would read as unmeasured and erase the mode's whole claim on this
    axis, and recording anything else would fabricate a reconfiguration.
    """
    row = schema.empty_row("results")
    run_mode._apply_mode_counters(row, {"context_loads": 0, "kernel_attaches": 0})
    assert row["context_loads"] == 0
    assert row["kernel_attaches"] == 0


# --- the v2 millisecond decomposition ---------------------------------------


def test_host_cpu_ms_arrives_as_buckets_and_persists_as_the_total():
    """The mode reports Profiler.time_cpu buckets; the column is their sum."""
    assert run_mode._component_ms({"host_cpu_ms": {"softmax": 2.5, "ln1": 1.5}},
                                  "host_cpu_ms") == 4.0


def test_an_empty_bucket_dict_is_a_measured_zero():
    """{} is `fused`/`coarse` saying 'no host compute ran' -- 0.0, not None."""
    assert run_mode._component_ms({"host_cpu_ms": {}}, "host_cpu_ms") == 0.0


def test_an_unreported_component_is_none_never_zero():
    """A missing key means unmeasured; fabricating 0.0 would claim otherwise."""
    assert run_mode._component_ms({}, "device_ms") is None
    assert run_mode._component_ms({"device_ms": None}, "device_ms") is None


def test_the_decomposition_mean_matches_the_latency_population():
    """Each column is the mean over the SAME iterations avg_latency_ms uses."""
    row = schema.empty_row("results")
    decomposition = {"device_ms": [10.0, 12.0], "sync_ms": [1.0, 3.0],
                     "host_cpu_ms": [0.0, 0.0]}
    run_mode._apply_ms_decomposition(row, decomposition, 2)
    assert row["device_ms"] == 11.0
    assert row["sync_ms"] == 2.0
    assert row["host_cpu_ms"] == 0.0, "a measured zero must persist as 0.0"


def test_a_partial_series_stays_empty():
    """A component missing from SOME iterations has no defensible mean."""
    row = schema.empty_row("results")
    decomposition = {"device_ms": [10.0], "sync_ms": [], "host_cpu_ms": []}
    run_mode._apply_ms_decomposition(row, decomposition, 2)
    assert row["device_ms"] is None
    assert row["sync_ms"] is None
    assert row["host_cpu_ms"] is None


def test_collection_feeds_application_end_to_end():
    """The two halves agree on shape: collect per iteration, then average."""
    row = schema.empty_row("results")
    decomposition = {key: [] for key in run_mode._MS_COMPONENTS}
    for extra in (
        {"device_ms": 10.0, "sync_ms": 1.0, "host_cpu_ms": {"softmax": 2.0}},
        {"device_ms": 14.0, "sync_ms": 3.0, "host_cpu_ms": {"softmax": 4.0}},
    ):
        run_mode._collect_ms_decomposition(decomposition, extra)
    run_mode._apply_ms_decomposition(row, decomposition, 2)
    assert row["device_ms"] == 12.0
    assert row["sync_ms"] == 2.0
    assert row["host_cpu_ms"] == 3.0


def test_every_ms_component_is_a_real_schema_column():
    """The component names double as column names; a drift writes nothing."""
    for key in run_mode._MS_COMPONENTS:
        assert key in schema.RESULTS_FIELDNAMES, f"{key} is not in the schema"
    assert tuple(run_mode._MS_COMPONENTS) == schema.DECOMPOSITION_FIELDNAMES
