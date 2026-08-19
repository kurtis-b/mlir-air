# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only checks on the declaration-side cost bridge.

    python3 study/test_analytical_cost.py

THE VALIDATION THAT MATTERS IS THE FIRST GROUP
    ``analytical_cost``'s traffic half is not modelled: it is doc 31a's
    DRAM-crossing lens plus doc 53 section 6's band-serial term. Section 6
    derived those figures BY HAND from measured artifacts (devq 338/340), and
    this module reproduces every one of them TO THE BYTE from a formula written
    off the mechanism instead. Two independent derivations agreeing to the byte
    is what makes the bridge checkable rather than plausible.

    Everything else here is about keeping the modelled half honest -- above all
    that doc 47's "1,208 measured / 5 counted / **0 modelled**" survives this
    module existing, which is the property the operator's choice of an
    analytical model puts at risk.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import analytical_cost as ac  # noqa: E402
import balance_ert  # noqa: E402
import mapping_space  # noqa: E402

EMB, FFN = 768, 3072


# --- doc 53 section 6, reproduced to the byte ------------------------------


def test_weights_once_matches_the_emitted_runtime_sequence():
    """9,437,184 B, read off %arg1/%arg2 in devq 338/340's artifact."""
    assert ac.weight_bytes(EMB, FFN) == 9437184


def test_the_band_serial_weight_column_matches():
    assert ac.bands(512) == 8 and ac.bands(1024) == 16
    assert ac.bands(512) * ac.weight_bytes(EMB, FFN) == 75497472
    assert ac.bands(1024) * ac.weight_bytes(EMB, FFN) == 150994944


def test_the_tail_floor_row_matches_31a():
    assert ac.tail_floor_bytes(512, EMB, FFN) == 11799552
    assert ac.tail_floor_bytes(1024, EMB, FFN) == 14158848


def test_the_tail_band_serial_row_matches():
    assert ac.tail_band_serial_bytes(512, EMB, FFN) == 77859840
    assert ac.tail_band_serial_bytes(1024, EMB, FFN) == 155716608


def test_the_net_residency_row_matches():
    """The finding: band-serial residency is NET NEGATIVE at both lengths."""
    assert ac.residency_net_bytes(512, EMB, FFN) == -48758784
    assert ac.residency_net_bytes(1024, EMB, FFN) == -106954752


def test_the_crossover_is_83_rows_and_1_30_bands():
    rows = ac.residency_crossover_rows(EMB, FFN)
    assert abs(rows - 83.0) < 1.0, rows
    assert abs(rows / ac.BAND_ROWS - 1.30) < 0.01


def test_the_tail_exceeds_the_packaged_LAYER_at_both_lengths():
    for seq, want in ((512, 1.52), (1024, 1.77)):
        ratio = ac.tail_band_serial_bytes(seq, EMB, FFN) / ac.whole_layer_floor_bytes(
            seq
        )
        assert abs(ratio - want) < 0.005, (seq, ratio)


def test_the_floor_and_the_band_serial_form_AGREE_at_one_band():
    """The discrimination control for the whole band term.

    At a single band the two models must coincide exactly; if they did not,
    every figure above could be matched by a formula that is wrong everywhere
    else. They diverge from the second band on, which is the other direction.
    """
    assert ac.tail_floor_bytes(64, EMB, FFN) == ac.tail_band_serial_bytes(64, EMB, FFN)
    assert ac.tail_floor_bytes(128, EMB, FFN) != ac.tail_band_serial_bytes(
        128, EMB, FFN
    )


def test_whole_layer_floor_refuses_to_interpolate():
    """31a's layer floor is not linear in seq -- weights do not scale with it."""
    try:
        ac.whole_layer_floor_bytes(2048)
    except ValueError as exc:
        assert "interpolating" in str(exc)
        return
    raise AssertionError("an unrecorded sequence length must be refused")


# --- the provenance rule ----------------------------------------------------


def test_doc_47s_zero_modelled_property_SURVIVES_this_module():
    """The one that would matter most if it broke.

    Doc 47's ERT is 1,208 measured / 5 counted / 0 modelled, and it declined to
    import iron's bandwidth figure so that ports report `unpriced` rather than
    ranking against a constant wearing a measurement's label. This module
    introduces the project's first modelled numbers; they must live HERE.
    """
    ert = balance_ert.Ert()
    counts = ert.by_source()
    assert counts["modelled"] == 0, (
        "analytical_cost has leaked a modelled entry into the balance ERT; "
        f"by_source={counts}"
    )
    # And the separation is structural, not incidental. Read by `ast` and not
    # by string search: the docstring SAYS "nothing in this file writes into
    # balance_ert", so a substring scan flags the sentence that documents the
    # property it is checking. (It did, on first writing.) What must be absent
    # is an IMPORT, which is the only way to reach that table from here.
    import ast

    tree = ast.parse(
        open(os.path.join(_HERE, "analytical_cost.py"), encoding="utf-8").read()
    )
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "balance_ert" not in imported and "balance" not in imported, (
        f"analytical_cost imports {sorted(imported)}: the modelled costs must "
        "stay out of doc 47's table, which is what makes '0 modelled' mean "
        "anything"
    )


def test_every_term_declares_where_its_number_came_from():
    c = ac.cost_for_shape(1024, EMB, FFN)
    assert c.terms
    for t in c.terms:
        assert t.provenance in (ac.MEASURED, ac.COUNTED, ac.MODELLED)
        assert t.why, f"{t.resource} has no provenance string"


def test_a_single_port_cost_is_MEASURED_and_a_scaled_one_is_MODELLED():
    """The linear port scaling is the biggest assumption here and says so."""
    one = ac.cost_for_shape(1024, EMB, FFN, shim_ports=1)
    many = ac.cost_for_shape(1024, EMB, FFN, shim_ports=7)
    assert one.terms[0].provenance == ac.MEASURED
    assert many.terms[0].provenance == ac.MODELLED
    assert many.modelled_fraction == 1.0 and one.modelled_fraction == 0.0


# --- the bottleneck and the ranking ----------------------------------------


def test_the_bottleneck_names_a_resource():
    c = ac.cost_for_shape(1024, EMB, FFN)
    assert c.bottleneck == "shim_dram"
    assert c.ns > 0


def test_more_ports_is_faster_and_more_bytes_is_slower():
    """Sanity in both directions -- a cost model that ignored one would pass
    a test that only checked the other."""
    base = ac.cost_for_shape(1024, EMB, FFN, shim_ports=1)
    wide = ac.cost_for_shape(1024, EMB, FFN, shim_ports=4)
    assert wide.ns < base.ns
    longer = ac.cost_for_shape(2048, EMB, FFN, shim_ports=1)
    assert longer.ns > base.ns


def test_band_serial_past_the_crossover_carries_a_note():
    """The model must SAY when it is in the regime doc 53 found net-negative."""
    past = ac.cost_for_shape(1024, EMB, FFN, band_serial=True)
    assert any("crossover" in n for n in past.notes)
    within = ac.cost_for_shape(64, EMB, FFN, band_serial=True)
    assert not any("crossover" in n for n in within.notes)


def test_a_mapping_prices_through_its_own_declared_demand():
    """The bridge itself: a Mapping -> Demand -> ports -> cost, no compile."""
    m = mapping_space.Mapping()
    c = ac.cost(m, 1024)
    assert c.ns > 0
    assert any("shim_mm2s_slots" in n for n in c.notes)


def test_rank_orders_cheapest_first_and_is_stable():
    m = mapping_space.Mapping()
    wide = mapping_space.Mapping(gemm_herd_x=1)
    ranked = ac.rank([m, wide], 1024)
    times = [c.ns for _m, c in ranked]
    assert times == sorted(times)


# --- the measured bound on what the model may claim -------------------------


def test_a_traffic_only_time_UNDER_predicts_measured_latency():
    """Pinned so nobody reads an absolute `ns` from this module as a latency.

    Every recorded point is 8x-19x slower than shim bandwidth alone would say,
    because the layer is not shim-bound at these shapes. (`[2026-08-19]` the
    floor moved 9.0 -> 8.0 with the re-walk: offload@1024's ratio is 8.47 on
    the faster machine, same bytes.)
    """
    ratios = []
    for _mode, _seq, byts, meas_ms in ac.RECORDED_MODE_POINTS:
        pred_ms = byts / ac.SHIM_PORT_BYTES_PER_NS / 1e6
        ratios.append(meas_ms / pred_ms)
    assert min(ratios) > 8.0, min(ratios)
    assert max(ratios) < 21.0, max(ratios)


def test_BYTE_ORDER_DOES_NOT_PREDICT_LATENCY_ORDER_across_modes():
    """The bound itself, as a check rather than a paragraph.

    If this ever starts passing as "the orders agree", either the modes have
    changed or someone has taught the model reconfiguration -- and in both cases
    the licence in the module docstring needs rewriting rather than quietly
    widening.
    """
    at1024 = [p for p in ac.RECORDED_MODE_POINTS if p[1] == 1024]
    by_bytes = [m for m, _s, _b, _l in sorted(at1024, key=lambda p: p[2])]
    by_latency = [m for m, _s, _b, _l in sorted(at1024, key=lambda p: p[3])]
    assert by_bytes == ["runlist", "fused", "coarse", "offload"], by_bytes
    assert by_latency == ["fused", "coarse", "runlist", "offload"], by_latency
    assert by_bytes != by_latency, (
        "byte order now predicts latency order; the module's 'same structure "
        "only' licence rests on it NOT doing so"
    )


def test_the_bytes_to_time_ratio_is_stable_WITHIN_a_mode():
    """The other half: within one dispatch structure, bytes do track latency.

    This is what licenses ranking mappings that share a structure, and without
    it the test above would say the model is useless rather than bounded.
    """
    by_mode = {}
    for mode, _seq, byts, meas_ms in ac.RECORDED_MODE_POINTS:
        pred_ms = byts / ac.SHIM_PORT_BYTES_PER_NS / 1e6
        by_mode.setdefault(mode, []).append(meas_ms / pred_ms)
    spread = {m: max(r) / min(r) for m, r in by_mode.items()}

    # The three reconfiguration-light modes hold to ~10%.
    for mode in ("coarse", "fused", "runlist"):
        assert spread[mode] < 1.15, f"{mode} ratio spread {spread[mode]:.2f}"

    # `offload` is the outlier (`[2026-08-19]` re-measured at 1.41; the
    # 08-14 numbers gave 1.37), and it is the mode that pays 30 hw_context
    # loads: its bytes grow 2.25x from 512 to 1024 while its latency
    # grows 1.59x, so the fixed reconfiguration cost dominates a larger share at
    # the short end. Asserted as a BOUND rather than smoothed into the others,
    # because it is the measured signature of exactly the axis this module does
    # not model.
    assert 1.30 < spread["offload"] < 1.45, f"offload {spread['offload']:.2f}"

    # And the BETWEEN-mode spread must exceed every within-mode one, or the
    # "same structure only" licence would be arbitrary rather than derived.
    flat = [r for rs in by_mode.values() for r in rs]
    assert max(flat) / min(flat) > max(spread.values())


def test_within_workload_cost_is_slots_only_and_ties_where_slots_agree():
    """The row-31 owed validation, closed as the NEGATIVE finding it is.

    `cost()` consumes exactly three things: cols, itemsize, and the declared
    shim slot count. Two clauses, and BOTH are the finding:

    1. At fixed seams/routes/herds, every axis that distinguishes two designs
       which both build and run today -- norm-tail form, gemm fold, `tile_k`,
       `rows_per_call` -- leaves the slot count unchanged and therefore prices
       as an EXACT tie. Measured-consequential: C1 vs C2 (banded vs decomposed
       tail, byte-identical traffic) is cleanly RESOLVED at 4096 -- doc 30,
       min 455.2 vs 485.3 ms, no overlap -- and this module cannot see it,
       because the difference lives in dispatch structure (3x the runlist
       entries per submission), doc 03's OTHER axis, declared out of scope.

    2. Where one of those axes DOES move the slot count (Codex review found
       the case: under a Pipe/Pipe/Pipe/Seq seam scope, both nt1 forms are
       LEGAL and `fused` occupies 4 slots to `j7a`'s 6), the resulting order
       is 100% the MODELLED port term -- the one whose own docstring calls a
       port-only winner "unresolved". So a rank() order between buildable
       designs is never a traffic statement: within a workload the bytes are
       identical by construction.

    Pinned so a future term that changes either clause fails here first and
    gets its licence argued rather than inherited.
    """
    import dataclasses

    import mapping_space as ms

    base = ms.Mapping()
    baseline = ac.cost(base, 1024)
    for field, values in (
        ("nt1_form", ("fused", "j7a")),
        ("nt2_form", ("fused", "j7a")),
        ("gemm_fold", ("split", "folded")),
        ("tile_k", (8, 16, 24, 32)),
        ("rows_per_call", (1, 2, 4, 8)),
    ):
        for v in values:
            m = dataclasses.replace(base, **{field: v})
            got = ac.cost(m, 1024)
            slots = ms.whole_design(m).shim_mm2s_slots
            assert slots == ms.whole_design(base).shim_mm2s_slots, (field, v)
            assert got.ns == baseline.ns, (field, v, got.ns, baseline.ns)

    # Clause 2: the configuration the Codex review located, where the nt1
    # form DOES move the slot count. Split-GEMM seams are
    # (nt1,up)(up,gelu)(gelu,down)(down,nt2); scoping only (down,nt2) as Seq
    # with narrow herds makes BOTH nt1 forms legal, at 4 slots (fused) vs 6
    # (j7a). The order rank() would give them is 100% the modelled port term
    # -- the dram_bytes are asserted identical to make that visible.
    seam_cfg = (
        (("nt1", "up"), "Pipe"),
        (("up", "gelu"), "Pipe"),
        (("gelu", "down"), "Pipe"),
        (("down", "nt2"), "Seq"),
    )
    narrow = dataclasses.replace(
        base,
        nt2_form="j7a",
        gemm_fold="split",
        gemm_herd_x=1,
        norm_herd_x=2,
        seams=seam_cfg,
    )
    a = dataclasses.replace(narrow, nt1_form="fused")
    b = dataclasses.replace(narrow, nt1_form="j7a")
    assert ms.legality(a).legal and ms.legality(b).legal, (
        ms.legality(a),
        ms.legality(b),
    )
    ca, cb = ac.cost(a, 1024), ac.cost(b, 1024)
    assert ca.dram_bytes == cb.dram_bytes
    slots_a = ms.whole_design(a).shim_mm2s_slots
    slots_b = ms.whole_design(b).shim_mm2s_slots
    assert slots_a != slots_b, (slots_a, slots_b)
    assert ca.ns != cb.ns, "the port term stopped discriminating this pair"

    # Pricing of an over-budget declaration is LOUD, not silent: the default
    # mapping demands 32 slots against the machine's 16, legality refuses it,
    # and cost() carries that in its notes -- rank() prices declarations
    # without filtering (Codex review finding, 2026-08-19), so the note is
    # what a consumer has to filter on until rank() learns to.
    notes = " ".join(baseline.notes)
    assert "shim_mm2s_slots 32 of 16" in notes, notes


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"analytical-cost tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
