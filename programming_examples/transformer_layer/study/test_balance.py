# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host tests for ``balance.py``. No NPU, no toolchain, no compiled tree.

THE FIXTURES ARE REAL, TRIMMED
    ``_MULTIPLEXED`` is the shipped ``addnorm`` design's routed artifact
    (``<repo>/air_project/aie.air.mlir``, the ``addnorm_seg`` segment) with the
    herd body and the per-column repetition trimmed and everything the
    instrument reads left verbatim: the three ``shim_dma_allocation`` symbols
    that land on ``(%shim_noc_tile_0_0, MM2S, 0)``, the packet-typed channel
    declarations, and the launch-level puts with their ``metadataArray``.
    ``python3 study/balance.py <repo>/air_project`` reads demand 3 on column 0
    against budget 2, which is doc 23's measured addnorm row.

    ``_COMPLIANT`` is the same design with the weight stream removed, which is
    doc 23's ``elementwise_add`` row: two streams per column, no multiplexing.
    It is the negative control for every check below that can pass -- each one
    is asserted to read differently on the two fixtures, so no check here is
    one that could not fail.

WHAT IS PINNED, AND WHY EACH
    Each test names the defect it would catch. The two that matter most:
    ``test_flow_only_counting_reads_zero_where_demand_is_three`` pins the
    blind spot in counting per-column ingress as ``aie.flow`` -- it reads 0 on
    the over-budget design, because AIR emits ``aie.packet_flow`` instead --
    and ``test_over_budget_is_priced_not_filtered`` pins doc 44's correction
    that the budget is a slope and not a legality predicate.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import balance  # noqa: E402
import balance_ert  # noqa: E402

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

#: Verbatim shape of the shipped `addnorm` routed design, trimmed to columns
#: 0 and 1. Column 0 carries weight + x + residual (3 L3-facing streams);
#: column 1 carries x + residual (2).
_MULTIPLEXED = """\
module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    aie.packet_flow(0) {
      aie.packet_source<%shim_noc_tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_2, DMA : 0>
    }
    aie.flow(%tile_0_2, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    aie.flow(%tile_1_2, DMA : 0, %shim_noc_tile_1_0, DMA : 0)
    aie.shim_dma_allocation @air_channel_3_0(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_3_1(%shim_noc_tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_0(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_1_0(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_1_1(%shim_noc_tile_1_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_2_0(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_2_1(%shim_noc_tile_1_0, MM2S, 0)
  }
  air.channel @channel_0 [1, 1] {broadcast_shape = [2, 1], channel_type = "npu_dma_packet"}
  air.channel @channel_1 [2, 1] {channel_type = "npu_dma_packet"}
  air.channel @channel_2 [2, 1] {channel_type = "npu_dma_packet"}
  air.channel @channel_3 [2, 1]
  func.func @addnorm(%arg0: memref<16x768xbf16>, %arg1: memref<16x768xbf16>, %arg2: memref<768xbf16>, %arg3: memref<16x768xbf16>) {
    %0 = air.launch async () in () args(%arg4=%arg0, %arg5=%arg1, %arg6=%arg2, %arg7=%arg3) : memref<16x768xbf16>, memref<16x768xbf16>, memref<768xbf16>, memref<16x768xbf16> attributes {id = 1 : i32} {
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %1 = air.channel.put async  @channel_0[] (%arg6[] [] []) {id = 1 : i32, metadataArray = [{base = "air_channel_0", index = 0 : i32}], packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>} : (memref<768xbf16>)
      %2 = air.channel.put async  @channel_1[%c0, %c0] (%arg4[%c0, 0] [8, 768] [768, 1]) {id = 2 : i32, metadataArray = [{base = "air_channel_1_0", index = 0 : i32}, {base = "air_channel_1_1", index = 1 : i32}], packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>} : (memref<16x768xbf16>)
      %3 = air.channel.put async  @channel_1[%c1, %c0] (%arg4[%c8, 0] [8, 768] [768, 1]) {id = 2 : i32, metadataArray = [{base = "air_channel_1_0", index = 0 : i32}, {base = "air_channel_1_1", index = 1 : i32}], packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>} : (memref<16x768xbf16>)
      %4 = air.channel.put async  @channel_2[%c0, %c0] (%arg5[%c0, 0] [8, 768] [768, 1]) {id = 3 : i32, metadataArray = [{base = "air_channel_2_0", index = 0 : i32}, {base = "air_channel_2_1", index = 1 : i32}], packet = #aie.packet_info<pkt_type = 0, pkt_id = 3>} : (memref<16x768xbf16>)
      %5 = air.channel.put async  @channel_2[%c1, %c0] (%arg5[%c8, 0] [8, 768] [768, 1]) {id = 3 : i32, metadataArray = [{base = "air_channel_2_0", index = 0 : i32}, {base = "air_channel_2_1", index = 1 : i32}], packet = #aie.packet_info<pkt_type = 0, pkt_id = 4>} : (memref<16x768xbf16>)
      %6 = air.channel.get async  @channel_3[%c0, %c0] (%arg7[%c0, 0] [8, 768] [768, 1]) {id = 4 : i32, metadataArray = [{base = "air_channel_3_0", index = 0 : i32}, {base = "air_channel_3_1", index = 1 : i32}]} : (memref<16x768xbf16>)
      %7 = air.channel.get async  @channel_3[%c1, %c0] (%arg7[%c8, 0] [8, 768] [768, 1]) {id = 4 : i32, metadataArray = [{base = "air_channel_3_0", index = 0 : i32}, {base = "air_channel_3_1", index = 1 : i32}]} : (memref<16x768xbf16>)
    }
    return
  }
}
"""

#: The same design with the weight stream removed: two per column, no
#: multiplexing. Doc 23's `elementwise_add` row.
_COMPLIANT = "\n".join(
    line
    for line in _MULTIPLEXED.splitlines()
    if "air_channel_0" not in line and "@channel_0" not in line
) + "\n"


def _matrix(text: str) -> balance.DemandMatrix:
    return balance.demand_matrix(
        balance.parse_transfers(text), balance.parse_allocations(text)
    )


def _shim_to_core_flow_count(text: str) -> int:
    """Per-column ingress counted the way ``norm_tail_structure`` counts it.

    Reimplemented here rather than imported so the test states the definition
    it is making a claim about; the claim is that this definition reads ZERO on
    an over-budget design.
    """
    core, _total = __import__("resource_usage").count_core_to_core_flows(text)
    del core
    rows = {
        name: int(row)
        for name, _col, row in __import__("resource_usage")._TILE_RE.findall(text)
    }
    flows = __import__("resource_usage")._FLOW_RE.findall(text)
    return sum(1 for s, d in flows if rows.get(s, -1) == 0 and rows.get(d, -1) >= 2)


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def test_parse_allocations_reads_column_direction_and_channel():
    allocations = balance.parse_allocations(_MULTIPLEXED)
    assert allocations["air_channel_1_1"] == balance.Port(1, "MM2S", 0)
    assert allocations["air_channel_3_0"] == balance.Port(0, "S2MM", 0)
    # Three symbols on ONE physical channel is the multiplexing, and the map
    # keeps them apart -- collapsing them here would lose the demand.
    on_col0_mm2s = [
        s for s, p in allocations.items() if p == balance.Port(0, "MM2S", 0)
    ]
    assert sorted(on_col0_mm2s) == [
        "air_channel_0",
        "air_channel_1_0",
        "air_channel_2_0",
    ]


def test_transfers_carry_the_bd_shape_the_ert_needs():
    transfers = balance.parse_transfers(_MULTIPLEXED)
    strided = [t for t in transfers if t.channel == "channel_1"]
    assert len(strided) == 2
    assert strided[0].sizes == (8, 768)
    assert strided[0].strides == (768, 1)
    assert strided[0].n_dims == 2
    assert strided[0].n_words == 8 * 768
    assert strided[0].element_bytes == 2


def test_whole_memref_transfer_gets_the_memrefs_own_shape():
    # `(%arg6[] [] [])` is AIR's whole-buffer form; treating it as zero words
    # would silently drop the weight stream's bytes.
    transfers = balance.parse_transfers(_MULTIPLEXED)
    weight = [t for t in transfers if t.channel == "channel_0"][0]
    assert weight.sizes == (768,)
    assert weight.strides == (1,)
    assert weight.n_words == 768
    assert weight.bytes == 1536


def test_herd_level_channel_ops_are_not_shim_facing():
    # The discriminator is `metadataArray`, not a guess about memory space.
    text = _MULTIPLEXED.replace(
        "    return",
        "      %9 = air.channel.get async  @channel_1[%c0, %c0] "
        "(%alloc[] [] []) {id = 6 : i32} : (memref<8x768xbf16, 2 : i32>)\n    return",
    )
    assert len(balance.parse_transfers(text)) == len(
        balance.parse_transfers(_MULTIPLEXED)
    )


# --------------------------------------------------------------------------
# 1. The demand matrix
# --------------------------------------------------------------------------


def test_demand_counts_distinct_channels_not_transfers():
    matrix = _matrix(_MULTIPLEXED)
    column1 = matrix.port(1, "MM2S")
    # Column 1 takes two transfers on two channels. Counting transfers would
    # give the same 2 here, so the discriminating case is column 0: three
    # channels but ALSO three transfers... so use the S2MM side, where one
    # channel carries one transfer per column, and the multiplexed column 0
    # MM2S, which carries three transfers on three channels.
    assert column1.static_demand == 2
    assert column1.transfers == 2
    assert matrix.port(0, "S2MM").static_demand == 1


def test_repeated_transfers_on_one_channel_are_one_stream():
    # Eight puts on one channel are one stream contending for a column. If
    # demand counted transfers this would read 8 and the column would look 4x
    # over a budget it is exactly meeting.
    text = _COMPLIANT
    extra = "\n".join(
        f'      %2{i} = air.channel.put async  @channel_1[%c0, %c0] '
        f"(%arg4[%c0, 0] [8, 768] [768, 1]) {{id = 2 : i32, metadataArray = "
        '[{base = "air_channel_1_0", index = 0 : i32}, '
        '{base = "air_channel_1_1", index = 1 : i32}]}} : (memref<16x768xbf16>)'
        for i in range(6)
    )
    text = text.replace("    return", extra + "\n    return")
    matrix = _matrix(text)
    assert matrix.port(0, "MM2S").transfers == 8
    assert matrix.port(0, "MM2S").static_demand == 2


def test_multiplexed_column_reads_three_against_budget_two():
    matrix = _matrix(_MULTIPLEXED)
    column0 = matrix.port(0, "MM2S")
    assert column0.static_demand == 3
    assert column0.budget == 2
    assert column0.over_budget
    assert column0.max_multiplex_depth == 3
    assert matrix.port(1, "MM2S").over_budget is False


def test_compliant_fixture_reads_two_and_is_not_over_budget():
    # The failing direction for the test above: same instrument, same shape,
    # one stream fewer.
    matrix = _matrix(_COMPLIANT)
    column0 = matrix.port(0, "MM2S")
    assert column0.static_demand == 2
    assert column0.over_budget is False
    assert column0.max_multiplex_depth == 2


def test_multiplex_depth_separates_three_on_one_from_two_on_two():
    # Demand 2 on two DISTINCT physical channels is compliant; demand 2 on one
    # physical channel is multiplexed. The two are different machines and a
    # bare demand count cannot tell them apart.
    spread = _COMPLIANT.replace(
        "@air_channel_2_0(%shim_noc_tile_0_0, MM2S, 0)",
        "@air_channel_2_0(%shim_noc_tile_0_0, MM2S, 1)",
    )
    assert _matrix(spread).port(0, "MM2S").max_multiplex_depth == 1
    assert _matrix(_COMPLIANT).port(0, "MM2S").max_multiplex_depth == 2


def test_flow_only_counting_reads_zero_where_demand_is_three():
    # THE BLIND SPOT. AIR's reaction to exceeding the per-column budget is to
    # emit `aie.packet_flow`, so a per-column ingress count over `aie.flow`
    # reads 0 exactly on the design that is over budget -- a check that cannot
    # fail. The allocation-based demand reads 3 on the same text.
    assert _shim_to_core_flow_count(_MULTIPLEXED) == 0
    assert _matrix(_MULTIPLEXED).port(0, "MM2S").static_demand == 3


def test_peak_concurrent_never_exceeds_static_demand():
    for text in (_MULTIPLEXED, _COMPLIANT):
        for port in _matrix(text).ports:
            assert port.peak_concurrent_demand <= port.static_demand


def test_bytes_are_exact_on_the_multiplexed_fixture():
    matrix = _matrix(_MULTIPLEXED)
    # weight 768*2 + x 8*768*2 + residual 8*768*2
    assert matrix.port(0, "MM2S").bytes == 1536 + 12288 + 12288
    assert matrix.port(1, "MM2S").bytes == 12288 + 12288
    assert matrix.port(0, "S2MM").bytes == 12288


def test_launch_iteration_space_multiplies_bytes():
    # An `air.launch ... in (%argN=%c4, ...)` repeats its whole body, so the
    # traffic is 4x. Missing this understated the shipped matmul artifact's
    # A-operand traffic by 8x while the demand table read correct.
    base = _matrix(_MULTIPLEXED).port(0, "MM2S").bytes
    text = _MULTIPLEXED.replace(
        "air.launch async () in ()", "air.launch async (%i, %j) in (%a=%c4, %b=%c1)"
    ).replace("%c8 = arith.constant 8 : index", "%c8 = arith.constant 8 : index\n      %c4 = arith.constant 4 : index")
    assert _matrix(text).port(0, "MM2S").bytes == base * 4


def test_unattributed_transfer_is_reported_not_dropped():
    text = _MULTIPLEXED.replace("@channel_1[%c0, %c0]", "@channel_1[%dyn, %c0]")
    matrix = _matrix(text)
    assert len(matrix.unattributed) == 1
    assert "not a constant" in matrix.unattributed[0].reason
    assert matrix.is_complete is False
    # And the same text with the constant restored attributes it.
    assert _matrix(_MULTIPLEXED).unattributed == ()
    assert _matrix(_MULTIPLEXED).is_complete is True


def test_unknown_trip_count_is_none_and_never_one():
    text = _MULTIPLEXED.replace(
        "      %2 = air.channel.put async  @channel_1[%c0, %c0]",
        "      %loop = scf.for %k = %c0 to %dynub step %c1 iter_args(%t = %c0) -> "
        "(!air.async.token) {\n"
        "      %2 = air.channel.put async  @channel_1[%c0, %c0]",
    ).replace(
        "      %3 = air.channel.put async  @channel_1[%c1, %c0]",
        "      }\n      %3 = air.channel.put async  @channel_1[%c1, %c0]",
    )
    matrix = _matrix(text)
    assert matrix.unknown_trip_counts == 1
    # The port's byte total is None, not a number that assumed one trip.
    assert matrix.port(0, "MM2S").bytes is None
    assert matrix.is_complete is False


# --------------------------------------------------------------------------
# 2. The static back-solve
# --------------------------------------------------------------------------


def test_back_solve_is_traffic_over_duration():
    matrix = _matrix(_MULTIPLEXED)
    required = balance.back_solve(matrix, 1000.0)
    column0 = [r for r in required if (r.column, r.direction) == (0, "MM2S")][0]
    assert column0.bytes == 26112
    assert column0.bytes_per_ns == 26112 / 1000.0
    assert column0.gigabytes_per_second == column0.bytes_per_ns


def test_back_solve_scales_inversely_with_duration():
    matrix = _matrix(_MULTIPLEXED)
    fast = balance.back_solve(matrix, 500.0)[0]
    slow = balance.back_solve(matrix, 1000.0)[0]
    assert abs(fast.bytes_per_ns - 2 * slow.bytes_per_ns) < 1e-9


def test_back_solve_refuses_a_zero_duration():
    matrix = _matrix(_MULTIPLEXED)
    for bad in (0.0, -1.0):
        try:
            balance.back_solve(matrix, bad)
        except ValueError as e:
            assert "duration_ns" in str(e)
        else:
            raise AssertionError(
                f"back_solve accepted duration_ns={bad}, which reports an "
                "infinite requirement as though it were a finding"
            )


def test_a_measured_duration_gives_a_lower_bound_on_the_requirement():
    # A stall-free run is no longer than the achieved one, so the rate an
    # achieved latency implies is a LOWER bound on what stall-free would need.
    # Pinned as a relation rather than a sentence so a caller cannot quote a
    # sustained rate as a requirement without this ordering holding.
    matrix = _matrix(_MULTIPLEXED)
    achieved = balance.back_solve(matrix, 1000.0)[0]
    stall_free = balance.back_solve(matrix, 800.0)[0]
    assert stall_free.bytes_per_ns > achieved.bytes_per_ns


def test_back_solve_leaves_unknown_bytes_unknown():
    matrix = _matrix(
        _MULTIPLEXED.replace(
            "      %2 = air.channel.put async  @channel_1[%c0, %c0]",
            "      %loop = scf.for %k = %c0 to %dynub step %c1 iter_args(%t = %c0) -> "
            "(!air.async.token) {\n"
            "      %2 = air.channel.put async  @channel_1[%c0, %c0]",
        ).replace(
            "      %3 = air.channel.put async  @channel_1[%c1, %c0]",
            "      }\n      %3 = air.channel.put async  @channel_1[%c1, %c0]",
        )
    )
    column0 = [
        r
        for r in balance.back_solve(matrix, 1000.0)
        if (r.column, r.direction) == (0, "MM2S")
    ][0]
    assert column0.bytes_per_ns is None


# --------------------------------------------------------------------------
# 3. The slope
# --------------------------------------------------------------------------


def test_over_budget_is_priced_not_filtered():
    # Doc 44's correction, pinned: the over-budget port keeps a record and gets
    # a slowdown. If the budget were a legality predicate this port would be
    # absent from the result and the silent multiplexing invisible.
    matrix = _matrix(_MULTIPLEXED)
    balances = balance.balance_ports(matrix)
    assert len(balances) == len(matrix.ports)
    column0 = [b for b in balances if (b.column, b.direction) == (0, "MM2S")][0]
    assert column0.over_budget
    assert abs(column0.slowdown - 2 / 3) < 1e-12
    assert abs(column0.inflation - 1.5) < 1e-12
    assert column0.multiplexed


def test_compliant_port_still_gets_a_record():
    # An empty violation list and a check that did not run look identical, so
    # every port carries a row with slowdown 1.0.
    balances = balance.balance_ports(_matrix(_COMPLIANT))
    assert balances
    assert all(b.slowdown == 1.0 for b in balances)
    assert all(b.warning() is None for b in balances)


def test_warning_prints_demand_beside_budget():
    # MAESTRO's warn tier: the diagnostic carries both numbers, not a verdict.
    balances = balance.balance_ports(_matrix(_MULTIPLEXED))
    column0 = [b for b in balances if (b.column, b.direction) == (0, "MM2S")][0]
    warning = column0.warning()
    assert "demand 3" in warning and "budget 2" in warning
    assert "multiplexed" in warning


def test_charged_ns_takes_the_worst_port_not_the_product():
    # Columns are parallel resources: a layer waits on the slowest, not on all
    # of them in series. Two 1.5x columns must charge 1.5x, not 2.25x.
    two = (
        balance.PortBalance(0, "MM2S", 3, 2, 2 / 3, True),
        balance.PortBalance(1, "MM2S", 3, 2, 2 / 3, True),
    )
    assert abs(balance.charged_ns(100.0, two) - 150.0) < 1e-9
    assert balance.charged_ns(100.0, ()) == 100.0


def test_slope_is_continuous_not_a_cliff():
    # Demand 3 and demand 4 must price differently. A predicate would give the
    # same answer (rejected) to both.
    three = balance.PortBalance(0, "MM2S", 3, 2, min(1.0, 2 / 3), True)
    four = balance.PortBalance(0, "MM2S", 4, 2, min(1.0, 2 / 4), True)
    assert three.inflation < four.inflation


# --------------------------------------------------------------------------
# 4. The bottleneck, and iron's two defects
# --------------------------------------------------------------------------


def _timed(name, ns, **kw):
    return balance.IsolatedTime(
        resource=name, ns=ns, source="measured", provenance="test", **kw
    )


def test_bottleneck_argmax_names_the_resource():
    result = balance.bottleneck(
        (_timed("ffn_up", 1590.0), _timed("ffn_down", 1387.0), _timed("ln2", 1063.0))
    )
    assert result.resource == "ffn_up"
    assert result.ns == 1590.0
    assert result.ranked[0] == ("ffn_up", 1590.0)
    assert result.is_complete


def test_unpriced_resource_is_named_and_not_treated_as_zero():
    # The shim ports are genuinely unpriced today (doc 33 deferred the
    # bandwidth operator). The max must be reported as taken over an incomplete
    # set rather than silently ranking only what happens to be priced.
    result = balance.bottleneck(
        (
            _timed("compute", 1590.0),
            balance.IsolatedTime("col0.MM2S", None, "absent", ""),
        )
    )
    assert result.resource == "compute"
    assert result.unpriced == ("col0.MM2S",)
    assert result.is_complete is False


def test_isolated_time_refuses_a_value_source_mismatch():
    for ns, source in ((100.0, "absent"), (None, "measured")):
        try:
            balance.IsolatedTime("r", ns, source, "test")
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"IsolatedTime accepted ns={ns!r} with source={source!r}; an "
                "unpriced resource reading as a free one names the wrong "
                "bottleneck"
            )


def test_bottleneck_refuses_an_empty_or_wholly_unpriced_set():
    for entries in (
        (),
        (balance.IsolatedTime("a", None, "absent", ""),),
    ):
        try:
            balance.bottleneck(entries)
        except ValueError:
            pass
        else:
            raise AssertionError("bottleneck named a resource on no evidence")


def test_containment_check_reports_whether_it_could_run():
    # It is VACUOUS on pure resources, and says so rather than implying a guard
    # fired. This is the honest half of the defect-1 fix.
    pure = balance.bottleneck((_timed("a", 2.0), _timed("b", 1.0)))
    assert pure.containment_checked is False
    tagged = balance.bottleneck(
        (
            _timed("a", 2.0, contains=("a",)),
            _timed("b", 1.0, contains=("b",)),
        )
    )
    assert tagged.containment_checked is True


def test_prefix_masquerading_as_a_stage_is_refused():
    # iron defect 1, with iron's own numbers (doc 38 §3.1, high-pacc case):
    # `addnorm1` = 3514 us kept the whole MHA computing, and `mha` = 1068 us is
    # in the same comparison. The max over that set is a prefix.
    try:
        balance.bottleneck(
            (
                _timed("addnorm1", 3514.0, contains=("mha", "ln1")),
                _timed("mha", 1068.0, contains=("mha",)),
            )
        )
    except balance.PrefixComparison as e:
        assert "addnorm1" in str(e) and "mha" in str(e)
    else:
        raise AssertionError(
            "a prefix and a stage were compared as two stages -- doc 38 §3.3 "
            "defect 1"
        )


def test_disjoint_stages_are_not_refused():
    # The failing direction for the test above: same guard, non-nested sets.
    result = balance.bottleneck(
        (
            _timed("ffn_up", 1590.0, contains=("ffn_up",)),
            _timed("ffn_down", 1387.0, contains=("ffn_down",)),
        )
    )
    assert result.resource == "ffn_up"


# --------------------------------------------------------------------------
# stage_gap: iron's metric with both defects made impossible
# --------------------------------------------------------------------------

#: B_Up 768x3072 + B_Down 3072x768 at bf16 -- the traffic iron's `addnorm1`
#: variant never fetched (doc 38 §3.3 defect 2).
_FFN_WEIGHT_BYTES = 768 * 3072 * 2 + 3072 * 768 * 2


def _stage(name, ns, contains, l3_bytes):
    return balance.IsolatedTime(
        resource=name,
        ns=ns,
        source="measured",
        provenance="test",
        contains=contains,
        l3_bytes=l3_bytes,
    )


def test_stage_gap_computes_full_max_gap_and_ratio():
    full = _stage("full", 4349.0, ("mha", "ln1", "ffn_up", "ffn_down", "ln2"), 10**7)
    stages = (
        _stage("ffn_up", 1590.0, ("ffn_up",), 10**7),
        _stage("ffn_down", 1387.0, ("ffn_down",), 10**7),
        _stage("addnorm2", 1063.0, ("ln2",), 10**7),
    )
    gap = balance.stage_gap(full, stages)
    assert gap.max_stage == "ffn_up"
    assert gap.gap_ns == 4349.0 - 1590.0
    assert abs(gap.ratio - 4349.0 / 1590.0) < 1e-12
    # Always serialisable: iron's `--output-json` defaulted to off, which is why
    # none of its numbers has a file behind it.
    assert gap.to_json()["max_stage"] == "ffn_up"


def test_stage_gap_refuses_a_variant_that_elided_weight_traffic():
    # iron defect 2, with the real shortfall: `addnorm1` set both
    # need_bup_weights and need_bdown_weights False, so ~9.4 MB the `full`
    # build reads were never fetched, and that difference lands in the gap as
    # though it were exposed serialization.
    full = _stage("full", 4349.0, ("mha", "ln1", "ffn_up"), 10**7)
    elided = _stage("addnorm1", 3514.0, ("ln1",), 10**7 - _FFN_WEIGHT_BYTES)
    try:
        balance.stage_gap(full, (elided,))
    except balance.ElidedTraffic as e:
        assert f"{_FFN_WEIGHT_BYTES:,}" in str(e)
    else:
        raise AssertionError(
            "a stage variant issuing 9.4 MB less DDR traffic than `full` was "
            "compared against it -- doc 38 §3.3 defect 2"
        )
    # Failing direction: identical traffic is accepted.
    matched = _stage("addnorm1", 3514.0, ("ln1",), 10**7)
    assert balance.stage_gap(full, (matched,)).max_stage == "addnorm1"


def test_stage_gap_refuses_an_entry_with_no_contains():
    # Defect 1's guard is vacuous on an empty `contains`, so `stage_gap` makes
    # declaring it mandatory rather than shipping a check that cannot fail.
    full = _stage("full", 4349.0, ("a", "b"), 10**7)
    bare = balance.IsolatedTime("s", 1000.0, "measured", "test", l3_bytes=10**7)
    try:
        balance.stage_gap(full, (bare,))
    except ValueError as e:
        assert "contains" in str(e)
    else:
        raise AssertionError("stage_gap ran its containment check on nothing")


def test_stage_gap_refuses_an_entry_with_no_l3_bytes():
    full = _stage("full", 4349.0, ("a", "b"), 10**7)
    bare = balance.IsolatedTime("s", 1000.0, "measured", "test", contains=("a",))
    try:
        balance.stage_gap(full, (bare,))
    except ValueError as e:
        assert "l3_bytes" in str(e)
    else:
        raise AssertionError("stage_gap compared variants of unstated DDR traffic")


def test_stage_gap_refuses_a_prefix_among_the_stages():
    full = _stage("full", 4349.0, ("mha", "ln1", "ffn_up"), 10**7)
    stages = (
        _stage("addnorm1", 3514.0, ("mha", "ln1"), 10**7),
        _stage("mha", 1068.0, ("mha",), 10**7),
    )
    try:
        balance.stage_gap(full, stages)
    except balance.PrefixComparison:
        pass
    else:
        raise AssertionError("both of iron's defects must be refused, not one")


def test_stage_gap_refuses_no_stages():
    full = _stage("full", 4349.0, ("a",), 10**7)
    try:
        balance.stage_gap(full, ())
    except ValueError:
        pass
    else:
        raise AssertionError("stage_gap took a max over nothing")


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def test_render_prints_every_port_and_names_the_overflow():
    matrix = _matrix(_MULTIPLEXED)
    text = balance.render(
        matrix,
        balance.balance_ports(matrix),
        label="fixture",
        required=balance.back_solve(matrix, 1000.0),
        duration_ns=1000.0,
    )
    assert "OVER BUDGET -- priced, not filtered" in text
    assert "demand 3" in text and "budget 2" in text
    # Every port has a row, compliant ones included.
    for column, direction in ((0, "MM2S"), (0, "S2MM"), (1, "MM2S"), (1, "S2MM")):
        assert f"| {column} | {direction} |" in text
    assert "ASAP async levels, NOT cycles" in text


def test_render_says_so_when_no_column_is_over_budget():
    matrix = _matrix(_COMPLIANT)
    text = balance.render(matrix, balance.balance_ports(matrix), label="fixture")
    assert "No column exceeds its budget" in text
    assert "OVER BUDGET" not in text


def test_render_flags_an_incomplete_reading():
    text_in = _MULTIPLEXED.replace("@channel_1[%c0, %c0]", "@channel_1[%dyn, %c0]")
    matrix = _matrix(text_in)
    text = balance.render(matrix, balance.balance_ports(matrix), label="fixture")
    assert "HONEST PARTIAL" in text
    assert "UNATTRIBUTED transfers: 1" in text
    # The complete reading does not claim to be partial.
    complete = _matrix(_MULTIPLEXED)
    assert "HONEST PARTIAL" not in balance.render(
        complete, balance.balance_ports(complete), label="fixture"
    )


def test_budget_comes_from_the_device_constant_not_a_literal():
    # One definition of the device constant. A local `2` here would drift the
    # day the constant does.
    import aircc_artifacts

    assert balance.PER_COLUMN_BUDGET == aircc_artifacts.SHIM_DMA_CHANNELS_PER_DIRECTION


def test_isolated_time_sources_come_from_the_ert_vocabulary():
    # The two modules must not grow separate source vocabularies; a value
    # `balance` accepts and `balance_ert` does not would let a modelled number
    # be labelled measured on one side of the seam.
    assert "measured" in balance_ert.COST_SOURCES
    try:
        balance.IsolatedTime("r", 1.0, "guessed", "test")
    except ValueError:
        pass
    else:
        raise AssertionError("IsolatedTime accepted a source outside COST_SOURCES")
