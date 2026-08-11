# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only checks on the case matrix.

    python3 study/test_cases.py

The FLOP tests compute the expected count a second way -- term by term, from the
shape -- rather than restating the module's own arithmetic, because a test that
copies the formula only proves the formula was typed twice.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import cases  # noqa: E402
import schema  # noqa: E402


def _raises(exc, match, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc as e:
        assert match in str(e), f"expected {match!r} in {str(e)!r}"
        return
    raise AssertionError(f"expected {exc.__name__} containing {match!r}")


def test_the_matrix_is_six_families_in_two_variants():
    assert len(cases.FAMILY_SPECS) == 6
    variants = {s.workload_variant for s in cases.FAMILY_SPECS.values()}
    assert variants == set(cases.WORKLOAD_VARIANTS)
    for variant in cases.WORKLOAD_VARIANTS:
        assert (
            sum(1 for s in cases.FAMILY_SPECS.values() if s.workload_variant == variant)
            == 3
        )


def test_family_ids_are_derived_from_the_specs_not_restated():
    assert cases.FAMILY_IDS == tuple(cases.FAMILY_SPECS)


def test_the_grid_positions_fill_the_declared_shape_exactly_once():
    """A duplicated cell would silently overplot one family with another."""
    rows, cols = cases.FAMILY_GRID_SHAPE
    positions = {(s.row_index, s.col_index) for s in cases.FAMILY_SPECS.values()}
    assert len(positions) == len(cases.FAMILY_SPECS)
    assert positions == {(r, c) for r in range(rows) for c in range(cols)}


def test_the_grid_rows_agree_with_the_row_labels():
    """Row 0 is the Encoder row and row 1 the Decoder row, or the labels lie."""
    for spec in cases.FAMILY_SPECS.values():
        expected = 0 if spec.workload_variant == "encoder_bert" else 1
        assert spec.row_index == expected, spec


def test_short_labels_are_unique():
    labels = [s.short_label for s in cases.FAMILY_SPECS.values()]
    assert len(set(labels)) == len(labels)


def test_head_size_divides():
    for fid in cases.FAMILY_IDS:
        c = cases.case(fid, 512)
        assert c.workload.hidden_size % c.workload.num_attention_heads == 0
        assert (
            c.workload.attention_head_size * c.workload.num_attention_heads
            == c.workload.hidden_size
        )


def test_flop_count_matches_a_term_by_term_derivation():
    seq, hidden, inter = 4096, 768, 3072
    expected = (
        6 * seq * hidden * hidden  # q, k, v
        + 4 * seq * seq * hidden  # scores + context
        + 2 * seq * hidden * hidden  # output projection
        + 4 * seq * hidden * inter  # up + down
    )
    assert (
        cases.effective_flop_count(
            seq_len=seq,
            hidden_size=hidden,
            intermediate_size=inter,
            num_attention_heads=12,
        )
        == expected
    )


def test_flop_count_grows_quadratically_in_sequence_once_attention_dominates():
    """The attention term is the only S^2 one; doubling S must more than double."""
    small = cases.case("baseline_768", 4096).workload.effective_flop_count
    large = cases.case("baseline_768", 8192).workload.effective_flop_count
    assert large > 2 * small


def test_flop_count_is_identical_across_execution_modes_by_construction():
    """It takes no mode argument at all -- pinned so nobody adds one."""
    import inspect

    params = set(inspect.signature(cases.effective_flop_count).parameters)
    assert params == {
        "seq_len",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
    }


def test_flop_count_refuses_an_indivisible_head_count():
    _raises(
        ValueError,
        "not divisible",
        cases.effective_flop_count,
        seq_len=64,
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=7,
    )


def test_gflops_is_none_for_an_unmeasured_or_impossible_latency():
    workload = cases.case("baseline_768", 1024).workload
    assert cases.effective_gflops_per_sec(workload, None) is None
    assert cases.effective_gflops_per_sec(workload, 0.0) is None
    assert cases.effective_gflops_per_sec(workload, -1.0) is None


def test_gflops_inverts_latency():
    workload = cases.case("baseline_768", 1024).workload
    fast = cases.effective_gflops_per_sec(workload, 100.0)
    slow = cases.effective_gflops_per_sec(workload, 200.0)
    assert fast == 2 * slow
    assert abs(fast - workload.effective_flop_count / 0.1 / 1e9) < 1e-6


def test_gflops_per_watt_needs_both_sides():
    assert cases.effective_gflops_per_sec_per_watt(100.0, None) is None
    assert cases.effective_gflops_per_sec_per_watt(None, 20.0) is None
    assert cases.effective_gflops_per_sec_per_watt(100.0, 0.0) is None
    assert cases.effective_gflops_per_sec_per_watt(100.0, 20.0) == 5.0


def test_canonical_execution_mode_accepts_both_sides_and_returns_the_csv_one():
    """Convention 7's direction, which is the easy one to get backwards."""
    assert cases.canonical_execution_mode("coarse") == "hybrid"
    assert cases.canonical_execution_mode("hybrid") == "hybrid"
    assert cases.canonical_execution_mode("fused") == "fused_elf"
    assert cases.canonical_execution_mode("fused_elf") == "fused_elf"


def test_canonical_execution_mode_defers_to_the_schema_for_the_domain():
    """No second list of the modes here; every CSV value must resolve."""
    for value in schema.EXECUTION_MODES:
        assert cases.canonical_execution_mode(value) == value
    for name, value in schema.EXECUTION_MODE_CSV.items():
        assert cases.canonical_execution_mode(name) == value


def test_canonical_execution_mode_raises_and_lists_both_vocabularies():
    _raises(ValueError, "code names are", cases.canonical_execution_mode, "nope")


def test_canonical_workload_variant_raises_on_an_unknown_one():
    _raises(
        ValueError, "unknown workload variant", cases.canonical_workload_variant, "x"
    )


def test_case_raises_on_an_unknown_family_rather_than_guessing():
    _raises(ValueError, "unknown family", cases.case, "baseline_999", 512)


def test_shape_columns_are_all_schema_columns():
    """A runner copies these straight into a row; an invented key would fail late."""
    names = {f.name for f in schema.fields_for("results")}
    assert set(cases.case("baseline_768", 512).shape_columns()) <= names


def test_iter_cases_walks_families_then_the_ladder():
    walked = cases.iter_cases(family="baseline_768")
    assert [c.seq_len for c in walked] == list(cases.SEQUENCE_LADDER)
    both = cases.iter_cases(seq_len=512)
    assert [c.study_case_id for c in both] == list(cases.FAMILY_IDS)


def test_iter_cases_filters_by_variant():
    encoder = cases.iter_cases(workload_variant="encoder_bert", seq_len=512)
    assert {c.workload_variant for c in encoder} == {"encoder_bert"}
    assert len(encoder) == 3


def test_the_ladder_is_not_filtered_by_what_the_hardware_can_build():
    """A case that cannot build must produce a failed row, not disappear."""
    assert 16384 in cases.SEQUENCE_LADDER
    assert cases.case("baseline_1024", 16384).seq_len == 16384


def test_ordered_family_ids_narrows_to_what_a_tree_actually_has():
    present = ["gpt2_512", "baseline_768"]
    assert cases.ordered_family_ids(present) == ("baseline_768", "gpt2_512")
    assert cases.ordered_family_ids(present, workload_variant="decoder_gpt2") == (
        "gpt2_512",
    )


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"case-matrix tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
