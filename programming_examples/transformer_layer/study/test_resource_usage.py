# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only checks on the routed-design resource parse.

    python3 study/test_resource_usage.py [--project DIR]

Without ``--project`` these are unit tests over synthetic artifacts: no NPU, no
MLIR, no aircc. Every number is checked against one computed by hand in the
test, because the module's whole job is arithmetic over a regex match and a
parser that agrees with itself proves nothing.

With ``--project`` pointing at a real ``air_project`` directory it additionally
parses that artifact and prints the row, which is how the parse gets checked
against a fresh toolchain rather than against this file's fixtures.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import aircc_artifacts  # noqa: E402
import resource_usage  # noqa: E402
import results_io  # noqa: E402
import schema  # noqa: E402

# One compute tile at 50% of L1, one memory tile at 25% of L2, one shim tile
# with one channel each way, and no core->core flow. Every expectation below is
# derived from these numbers by hand.
_TIME_MULTIPLEXED = """
    %tile_2_0 = aie.tile(2, 0)
    %mem_tile_2_1 = aie.tile(2, 1)
    %tile_2_2 = aie.tile(2, 2)
    %buf0 = aie.buffer(%tile_2_2) {sym_name = "buf0"} : memref<16384xbf16, 2 : i32>
    %buf1 = aie.buffer(%mem_tile_2_1) {sym_name = "buf1"} : memref<65536xbf16, 1 : i32>
    %core_2_2 = aie.core(%tile_2_2) {
      aie.end
    }
    aie.flow(%tile_2_0, DMA : 0, %mem_tile_2_1, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 0, %tile_2_2, DMA : 0)
    aie.shim_dma_allocation @c0(%shim_noc_tile_2_0, S2MM, 0)
    aie.shim_dma_allocation @c1(%shim_noc_tile_2_0, MM2S, 0)
"""

# Two compute tiles handing off L1 -> L1: one core->core flow, which is doc 03's
# space-multiplexed discriminator.
_SPACE_MULTIPLEXED = """
    %tile_3_0 = aie.tile(3, 0)
    %tile_3_2 = aie.tile(3, 2)
    %tile_3_3 = aie.tile(3, 3)
    %core_3_2 = aie.core(%tile_3_2) {
      aie.end
    }
    %core_3_3 = aie.core(%tile_3_3) {
      aie.end
    }
    aie.flow(%tile_3_0, DMA : 0, %tile_3_2, DMA : 0)
    aie.flow(%tile_3_2, DMA : 0, %tile_3_3, DMA : 0)
"""


def _raises(exc, match, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc as e:
        assert match in str(e), f"expected {match!r} in {str(e)!r}"
        return
    raise AssertionError(f"expected {exc.__name__} containing {match!r}")


def _parse(text):
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / aircc_artifacts.ROUTED_DESIGN_NAME
        path.write_text(text, encoding="utf-8")
        return resource_usage.parse_routed_design(path)


def test_memref_bytes_multiplies_every_dimension():
    """The artifact's memrefs are 6-D; a 1-D-only parser would undercount."""
    assert resource_usage.memref_bytes("1x1x16x4x8x8xf32, 2 : i32") == 4096 * 4
    assert resource_usage.memref_bytes("768xbf16, 2 : i32") == 1536


def test_memref_bytes_returns_none_rather_than_zero_when_unsizeable():
    """None is countable as 'not accounted for'; 0 joins the sum invisibly."""
    assert resource_usage.memref_bytes("?x8xbf16") is None
    assert resource_usage.memref_bytes("8x8xsomedtype") is None
    assert resource_usage.memref_bytes("bf16") is None


def test_unsizeable_buffers_are_counted_not_summed():
    usage = _parse(
        _TIME_MULTIPLEXED
        + '    %bufx = aie.buffer(%tile_2_2) {sym_name = "bufx"} : memref<?xbf16>\n'
    )
    assert usage.unsizeable_buffers == 1
    # Unchanged from the fixture without it: the bad buffer added nothing.
    assert usage.aie_tile_allocated_bytes == 32768


def test_l1_and_l2_are_normalized_against_their_own_capacities():
    usage = _parse(_TIME_MULTIPLEXED)
    assert usage.compute_tiles_used == 1
    assert usage.aie_tiles_with_buffers == 1
    assert usage.aie_tile_allocated_bytes == 32768
    assert usage.aie_tile_memory_utilization == 32768 / 65536
    assert usage.mem_tiles_with_buffers == 1
    assert usage.mem_tile_allocated_bytes == 131072
    assert usage.mem_tile_memory_utilization == 131072 / 524288


def test_utilization_is_pooled_over_used_tiles_not_over_the_array():
    """The documented footgun, pinned: adding an EMPTY core moves nothing."""
    base = _parse(_TIME_MULTIPLEXED)
    with_idle_core = _parse(
        _TIME_MULTIPLEXED
        + "    %tile_2_3 = aie.tile(2, 3)\n"
        + "    %core_2_3 = aie.core(%tile_2_3) {\n      aie.end\n    }\n"
    )
    assert with_idle_core.compute_tiles_used == base.compute_tiles_used + 1
    assert (
        with_idle_core.aie_tile_memory_utilization == base.aie_tile_memory_utilization
    )


def test_per_tile_spread_separates_a_balanced_design_from_a_lopsided_one():
    lopsided = _parse(
        _TIME_MULTIPLEXED
        + "    %tile_2_3 = aie.tile(2, 3)\n"
        + '    %buf2 = aie.buffer(%tile_2_3) {sym_name = "buf2"} : memref<512xbf16, 2 : i32>\n'
    )
    assert lopsided.aie_tile_memory_utilization_min == 1024 / 65536
    assert lopsided.aie_tile_memory_utilization_max == 32768 / 65536
    assert (
        lopsided.aie_tile_memory_utilization_mean
        == ((1024 / 65536) + (32768 / 65536)) / 2
    )
    # The pooled ratio hides that spread entirely, which is why both are columns.
    assert lopsided.aie_tile_memory_utilization == (1024 + 32768) / (2 * 65536)


def test_shim_channels_split_by_direction():
    usage = _parse(_TIME_MULTIPLEXED)
    assert usage.shim_tiles_with_s2mm == 1
    assert usage.shim_s2mm_channels_used == 1
    assert usage.shim_s2mm_channel_utilization == 0.5
    assert usage.shim_tiles_with_mm2s == 1
    assert usage.shim_mm2s_channels_used == 1
    assert usage.shim_tiles_with_dma == 1


def test_the_same_channel_number_both_ways_is_two_channels():
    """The one departure from iron's arithmetic, pinned.

    The fixture allocates S2MM 0 and MM2S 0 on one shim tile. Those are distinct
    hardware channels; iron's combined set is keyed on the number alone and
    reports one of four. This reports two of four, and the per-direction columns
    above are unchanged.
    """
    usage = _parse(_TIME_MULTIPLEXED)
    assert usage.shim_dma_channels_used == 2
    assert usage.shim_dma_channel_utilization == 2 / 4


def test_a_repeated_allocation_of_one_channel_counts_once():
    """Two allocations naming the same (tile, direction, channel) are one."""
    usage = _parse(
        _TIME_MULTIPLEXED
        + "    aie.shim_dma_allocation @c2(%shim_noc_tile_2_0, S2MM, 0)\n"
    )
    assert usage.shim_s2mm_channels_used == 1
    assert usage.shim_dma_channels_used == 2


def test_absent_dma_allocations_are_none_with_a_note_not_zero():
    usage = _parse(_TIME_MULTIPLEXED)
    assert usage.mem_tiles_with_dma is None
    assert usage.mem_dma_channels_used is None
    assert usage.mem_dma_channel_utilization is None
    assert "absent is not zero" in usage.mem_dma_channel_note
    assert usage.compute_dma_channel_utilization is None
    assert "absent is not zero" in usage.compute_dma_channel_note


def test_memory_tile_dma_allocations_are_normalized_against_six_channels():
    usage = _parse(
        _TIME_MULTIPLEXED
        + "    aie.memtile_dma_allocation @m0(%mem_tile_2_1, S2MM, 0)\n"
        + "    aie.memtile_dma_allocation @m1(%mem_tile_2_1, MM2S, 1)\n"
    )
    assert usage.mem_tiles_with_dma == 1
    assert usage.mem_dma_channels_used == 2
    assert usage.mem_dma_channel_utilization == 2 / 6
    assert usage.mem_dma_channel_note is None


def test_core_to_core_flows_separate_the_two_role_styles():
    """Doc 03's discriminator, verified in BOTH directions."""
    time_multiplexed = _parse(_TIME_MULTIPLEXED)
    assert time_multiplexed.total_flows == 2
    assert time_multiplexed.core_to_core_flows == 0
    assert resource_usage.role_style(time_multiplexed) == "time-multiplexed"

    space_multiplexed = _parse(_SPACE_MULTIPLEXED)
    assert space_multiplexed.total_flows == 2
    assert space_multiplexed.core_to_core_flows == 1
    assert resource_usage.role_style(space_multiplexed) == "space-multiplexed"


def test_a_flow_through_an_undeclared_tile_is_not_counted_as_core_to_core():
    """It cannot be classified, so it is not guessed into the on-chip count."""
    core_to_core, total = resource_usage.count_core_to_core_flows(
        "aie.flow(%mystery_a, DMA : 0, %mystery_b, DMA : 0)\n"
    )
    assert (core_to_core, total) == (0, 1)


def test_row_round_trips_through_the_schema():
    usage = _parse(_TIME_MULTIPLEXED)
    row = resource_usage.resource_row(
        usage, artifact_name="fixture", routed_design_path="/nowhere/aie.air.mlir"
    )
    schema.validate_row(row, "resource")
    assert row["run_status"] == "passed"
    assert row["core_to_core_flows"] == 0
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "resource.csv"
        results_io.write_rows(path, [row], table="resource")
        back = results_io.read_rows(path, table="resource")
    assert len(back) == 1
    assert back[0]["artifact_name"] == "fixture"


def test_a_failed_parse_still_writes_a_complete_row():
    """Same rule as a failed measurement: absent rows and failed rows differ."""
    row = resource_usage.resource_row(
        None,
        artifact_name="broken",
        routed_design_path="/nowhere/aie.air.mlir",
        failure_message="RoutedDesignMissing: it is absent",
    )
    schema.validate_row(row, "resource")
    assert row["run_status"] == "failed"
    assert "RoutedDesignMissing" in row["failure_message"]
    assert row["compute_tiles_used"] is None


def test_a_missing_project_produces_a_failed_row_rather_than_raising():
    with tempfile.TemporaryDirectory() as d:
        rows = resource_usage.rows_for_projects([str(Path(d) / "air_project")])
    assert len(rows) == 1
    assert rows[0]["run_status"] == "failed"
    assert "died before lowering completed" in rows[0]["failure_message"]


def test_rows_are_named_by_the_parent_not_by_air_project():
    """Every aircc project directory has the same leaf name."""
    with tempfile.TemporaryDirectory() as d:
        project = Path(d) / "coarse_cache" / "air_project"
        project.mkdir(parents=True)
        (project / aircc_artifacts.ROUTED_DESIGN_NAME).write_text(
            _TIME_MULTIPLEXED, encoding="utf-8"
        )
        rows = resource_usage.rows_for_projects([str(project)])
    assert rows[0]["artifact_name"] == "coarse_cache"
    assert rows[0]["study_case_id"] == "coarse_cache"


def test_the_resource_table_rejects_a_results_row():
    """The tables are separate on purpose; a mixed write must not be silent."""
    _raises(
        ValueError,
        "missing columns",
        schema.validate_row,
        schema.empty_row("results"),
        "resource",
    )


def test_execution_mode_still_takes_only_csv_values():
    row = resource_usage.resource_row(
        None, artifact_name="a", routed_design_path="p", execution_mode="coarse"
    )
    _raises(ValueError, "CODE name", schema.validate_row, row, "resource")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", help="a real air_project directory to parse")
    args = ap.parse_args()

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"resource-usage tests: {len(tests)}/{len(tests)} passed")

    if args.project:
        print(f"\nparsing {args.project}")
        usage = resource_usage.parse_routed_design(
            aircc_artifacts.routed_design(args.project)
        )
        for key, value in sorted(usage.as_columns().items()):
            print(f"  {key:<40} {value}")
        print(f"  {'role_style':<40} {resource_usage.role_style(usage)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
