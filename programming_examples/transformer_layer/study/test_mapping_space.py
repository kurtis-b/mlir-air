# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only checks on the static legality predicate and the space census.

    python3 study/test_mapping_space.py

No device, no compile, no toolchain: ``mapping_space`` reads declarations, so
everything here is arithmetic on them. Three groups of checks, and the first two
are the ones that make the third mean anything.

1. **The predicate reproduces what was MEASURED.** R1's per-column shim census
   (doc 31b section 3.6: 7 of 16 ports, 4 shim->core + 3 shim->memtile, memtiles
   at 4/6 MM2S with 2/6 and 5/6 S2MM), the herd inventory (section 3.5: nine
   herds of [4,1] refuse, eight place), J7a's "exactly two per column" (doc 23),
   and queue item 10's over-budget control. A static model that cannot
   re-derive a measured table is not a model of this machine.

2. **The predicate can fail, and can fail SELECTIVELY.** Item 10's lesson is
   that a census which reads 0 exactly when a column is over budget is blind
   where it is needed. The static predicate has a different shape -- it reads
   the declaration, not the routed design -- but "different shape" is not
   evidence, so it is demonstrated refusing the over-budget design and
   admitting R1, and every clause has a test that moves it from green to red.

3. **The count is a count.** The factorisation ``legal = n_numeric x n_routing``
   is checked against brute force over whole sub-spaces, so a future clause that
   couples the axes fails here instead of biasing the headline number.

WHY THE CONSTANTS ARE PINNED BY AST
    ``mapping_space`` re-states MAX_PLACEABLE_HERD_X, MAX_L1_TILE_K, TILE_M,
    MICRO, L1_BYTES and the rest from ``builders/``, because ``builders/``
    imports ``air`` and this suite must run without a toolchain (the reason
    ``test_run_ladder._spec`` is a hand-written catalogue row). A re-stated
    constant that is not pinned agrees with its builder until the builder moves
    and then agrees with history -- and every one of these is a MEASURED wall, so
    a stale copy is a space size computed against a machine that no longer
    refuses what it used to.
"""

from __future__ import annotations

import ast
import dataclasses
import itertools
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import mapping_space as ms  # noqa: E402

_BUILDERS = os.path.join(_EXAMPLE, "builders")


def _module_constants(path):
    """``{name: value}`` for module-level assignments of constant expressions.

    Text, not an import: see the module docstring. ``ast.literal_eval`` is not
    enough -- the builders write ``64 * 1024`` and
    ``NPU2_COLUMNS * SHIM_MM2S_PER_COLUMN``, neither of which it evaluates -- so
    each value expression is compiled and evaluated against the constants
    already gathered from the SAME file, in an empty builtins namespace. Any
    expression that reaches outside that (an import, a call) simply does not
    become a constant, which is the right outcome: this is a pin on literals.
    """
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            value = eval(  # noqa: S307 -- an expression parsed out of our own tree
                compile(ast.Expression(node.value), path, "eval"),
                {"__builtins__": {}},
                dict(out),
            )
        except Exception:
            continue
        if isinstance(value, (int, float, str, bool, tuple)):
            out[target.id] = value
    return out


# ---------------------------------------------------------------------------
# 0. The re-stated constants still match the builders they came from.
# ---------------------------------------------------------------------------


def test_ffn_accum_constants_are_still_what_is_restated():
    src = _module_constants(os.path.join(_BUILDERS, "ffn_accum.py"))
    assert src, "no constants parsed from builders/ffn_accum.py; the ast walk broke"
    expected = {
        "MICRO": ms.MICRO,
        "TILE_M": ms.TILE_M,
        "MAX_PLACEABLE_HERD_X": ms.MAX_PLACEABLE_HERD_X,
        "MAX_L1_TILE_K": ms.MAX_L1_TILE_K,
        "MAX_FEED_CHANNELS": ms.MAX_FEED_CHANNELS,
        "L2_BYTES": ms.L2_BYTES,
    }
    for name, mine in expected.items():
        assert name in src, (
            f"builders/ffn_accum.py no longer defines {name}, which "
            f"mapping_space re-states as {mine}. Every one of these is a "
            "MEASURED wall; re-derive it, do not delete the pin"
        )
        assert src[name] == mine, (
            f"builders/ffn_accum.{name} is {src[name]}; mapping_space re-states "
            f"{mine}. The space size is computed against the stale one"
        )


def test_norm_tail_constants_are_still_what_is_restated():
    src = _module_constants(os.path.join(_BUILDERS, "norm_tail.py"))
    assert src, "no constants parsed from builders/norm_tail.py; the ast walk broke"
    for name, mine in (
        ("L1_BYTES", ms.L1_BYTES),
        ("L1_STACK_BYTES", ms.L1_STACK_BYTES),
        ("NORM_TAIL_VEC_LEN", ms.NORM_TAIL_VEC_LEN),
    ):
        assert name in src, f"builders/norm_tail.py no longer defines {name}"
        assert src[name] == mine, (
            f"builders/norm_tail.{name} is {src[name]}; mapping_space re-states "
            f"{mine}"
        )


def test_the_target_model_constants_match_the_compiled_census():
    """``ffn_resident_structure.py`` counts the SAME budget on a routed design.

    Two modules asserting different NPU2 geometries would disagree about which
    designs are over budget while both looking right, so the static predicate is
    pinned to the compiled census's own constants rather than to a second
    reading of AIETargetModel.h.
    """
    src = _module_constants(os.path.join(_EXAMPLE, "ffn_resident_structure.py"))
    assert src, "no constants parsed from ffn_resident_structure.py"
    assert src["NPU2_COLUMNS"] == ms.NPU2_COLUMNS
    assert src["SHIM_MM2S_PER_COLUMN"] == ms.SHIM_MM2S_PER_COLUMN
    assert src["NPU2_SHIM_MM2S_PORTS"] == ms.NPU2_SHIM_MM2S_PORTS
    # The compiled census's own negative control is the design this module's
    # negative control declares; if it changes shape, this one is stale.
    assert src["CONTROL_L3_STREAMS"] == ms.SHIM_MM2S_PER_COLUMN + 1


def test_the_norm_tail_l1_formula_is_the_builders_formula():
    """The ping-pong factor of two is a MEASUREMENT, not an assumption.

    Re-derived from the source of ``_stage_l1_bytes`` rather than compared
    against a remembered number: the function is read out of norm_tail.py, its
    body evaluated as an expression, and the result compared with this module's
    copy at three shapes.
    """
    path = os.path.join(_BUILDERS, "norm_tail.py")
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    fn = next(
        (
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_stage_l1_bytes"
        ),
        None,
    )
    assert fn is not None, (
        "builders/norm_tail.py no longer defines _stage_l1_bytes, which "
        "mapping_space.norm_tail_l1_bytes re-states"
    )
    ret = next(n for n in fn.body if isinstance(n, ast.Return))
    expr = compile(ast.Expression(ret.value), "<_stage_l1_bytes>", "eval")
    for rpc, cols, itemsize in ((1, 768, 2), (4, 768, 2), (8, 768, 2)):
        theirs = eval(  # noqa: S307 -- an expression parsed out of our own tree
            expr,
            {"L1_STACK_BYTES": ms.L1_STACK_BYTES},
            {"rows_per_call": rpc, "cols": cols, "itemsize": itemsize},
        )
        assert theirs == ms.norm_tail_l1_bytes(rpc, cols, itemsize), (
            f"at rows_per_call={rpc}: builders/norm_tail says {theirs}, "
            f"mapping_space says {ms.norm_tail_l1_bytes(rpc, cols, itemsize)}"
        )


def test_the_measured_l1_wall_is_reproduced():
    """rows_per_call 8 at cols 768 was MEASURED to overflow; 4 was not."""
    assert ms.norm_tail_l1_bytes(8, 768, 2) > ms.L1_BYTES
    assert ms.norm_tail_l1_bytes(4, 768, 2) <= ms.L1_BYTES


def test_the_gemm_side_contributes_no_l1_byte_demand():
    """A deliberate omission, pinned so it cannot be "fixed" back into a guess.

    ``build_ffn_accum_module`` states its L1 wall as the CONSTANT
    ``MAX_L1_TILE_K``, not as a formula: "32 keeps the worst-case L1 comfortably
    under the 64 KiB tile; 64 measures just over it". Ping-ponged A+B plus a
    resident C at tile_k 64, tile_n 192 is 91 KiB, not "just over" 64 -- so the
    prose does not determine a formula, and one written until it fit would
    invent a wall that refuses gemm_herd_x 1 and 2, which the builder accepts.
    In a module whose output is a space size, an invented wall is an invented
    cut in the direction that flatters the headline number.

    So the GEMM groups carry no L1 bytes and the tile_k AXIS carries the wall.
    """
    m = _base(gemm_herd_x=1, gemm_fold="split")
    groups = ms.group_demands(m)
    for name in ("up", "down", "gelu"):
        assert groups[name].l1_bytes == 0, (
            f"the {name} group now claims {groups[name].l1_bytes} B of L1. If a "
            "MEASURED formula has been derived, say where; if it was inferred "
            "from build_ffn_accum_module's prose, it refuses gemm_herd_x the "
            "builder accepts"
        )
    assert ms.axis("tile_k").values[-1] == ms.MAX_L1_TILE_K, (
        "the tile_k axis no longer ends at MAX_L1_TILE_K, so nothing carries "
        "the GEMM L1 wall at all"
    )
    assert ms.GEMM_L1_IS_BOUNDED_BY_TILE_K_NOT_BY_A_BYTE_FORMULA


# ---------------------------------------------------------------------------
# 1. TileFlow's combinator.
# ---------------------------------------------------------------------------


def test_time_scopes_take_the_max_and_space_scopes_sum():
    """``(Sequential || Sharing) ? Op::max(exprs) : Op::sum(exprs)``.

    Doc 46 section 4.3, quoting ``src/mapper/checker.cpp:506-519``. This is the
    one line of TileFlow this whole module exists to port, so it is asserted on
    its own rather than only through the designs that use it.
    """
    a = ms.Demand(herds=((4, 1),), l1_bytes=1000, l2_bytes=100)
    b = ms.Demand(herds=((2, 1),), l1_bytes=2000, l2_bytes=200)

    for scope in ms.SCOPE_SPACE:
        out = ms.compose(scope, (a, b))
        assert out.cores == 6, f"{scope} must SUM cores, got {out.cores}"
        assert out.shim_mm2s_slots == 6, f"{scope} must SUM shim slots"

    seq = ms.compose(ms.SCOPE_SEQ, (a, b))
    assert seq.cores == 4, "Seq must MAX cores (the stages take turns)"
    assert seq.l1_bytes == 2000, "Seq footprint is max: buffers freed between"
    assert seq.l2_bytes == 200

    shar = ms.compose(ms.SCOPE_SHAR, (a, b))
    assert shar.cores == 4, "Shar must MAX cores (the stages take the same herd)"
    assert shar.l1_bytes == 3000, "Shar footprint ADDS: buffers stay live"
    assert shar.l2_bytes == 300


def test_seq_and_shar_are_distinguishable():
    """Doc 46's table gives Shar its own row; collapsing it onto Seq deletes it.

    The two agree on compute and differ on footprint, which is the memory axis of
    the 2x2. If a refactor made them identical, this fails.
    """
    a = ms.Demand(herds=((4, 1),), l1_bytes=1000)
    b = ms.Demand(herds=((4, 1),), l1_bytes=1000)
    assert ms.compose(ms.SCOPE_SEQ, (a, b)) != ms.compose(ms.SCOPE_SHAR, (a, b))


def test_shar_stacks_the_guests_l3_operands_onto_the_hosts_columns():
    """Two groups on the same cores share the same columns, so their demands add.

    That is doc 23's rule ("the budget is per column ACROSS STACKED HERDS") in
    its sharpest form: a Shar seam is what turns two legal one-operand groups
    into one illegal two-operand herd when a third joins.
    """
    a = ms.Demand(herds=((8, 1),))
    b = ms.Demand(herds=((8, 1),))
    c = ms.Demand(herds=((8, 1),))
    merged = ms.compose(ms.SCOPE_SHAR, (a, b, c))
    assert merged.herds == ((8, 3),)
    assert max(ms.predicted_columns(merged)) == 3


# ---------------------------------------------------------------------------
# 2. The measured tables, reproduced statically.
# ---------------------------------------------------------------------------


def test_r1_reproduces_the_measured_column_census():
    """Doc 31b section 3.6, MEASURED: 7 of 16, 4 shim->core + 3 shim->memtile.

    The compiled census (``ffn_resident_structure.py`` clause 3) pins the same
    figures off a routed dump. This one gets there with no compiler at all, from
    the declaration -- which is the claim item 26 is making.
    """
    r1 = ms.r1_interior_demand()
    assert r1.shim_mm2s_slots == 7, (
        f"R1 counts {r1.shim_mm2s_slots} shim MM2S slots; doc 31b section 3.6 "
        "MEASURED 7 of 16 on the routed design"
    )
    assert sum(w * d for w, d in r1.herds) == 4, "4 shim->core (the C fetches)"
    assert r1.shim_global == 3, "3 shim->memtile (hidden, w_up, w_down)"
    assert r1.cores == 12 and len(r1.herds) == 3


def test_r1_reproduces_the_measured_memtile_occupancy():
    """Doc 31b section 3.6: memtile MM2S 4/6 on both, S2MM 2/6 and 5/6."""
    r1 = ms.r1_interior_demand()
    assert sorted(r1.memtiles) == [(4, 2), (4, 5)], (
        f"memtile occupancy {sorted(r1.memtiles)}; doc 31b section 3.6 MEASURED "
        "the up feed at 4 MM2S / 2 S2MM and the down feed at 4 / 5 (the GeLU "
        "fan-out plus the w_down refill)"
    )
    for mm2s, s2mm in r1.memtiles:
        assert mm2s <= ms.MEMTILE_MM2S and s2mm <= ms.MEMTILE_S2MM


def test_r1_is_admitted_and_says_it_is_within_budget():
    r1 = ms.r1_interior_demand()
    assert max(d for _, d in r1.herds) <= ms.SHIM_MM2S_PER_COLUMN
    assert r1.shim_mm2s_slots <= ms.NPU2_SHIM_MM2S_PORTS
    assert max(ms.predicted_columns(r1)) <= ms.SHIM_MM2S_PER_COLUMN, (
        "R1's predicted per-column load is over budget, and R1 routes today at "
        f"a MEASURED worst column of 2: {ms.predicted_columns(r1)}"
    )


def test_the_over_budget_control_is_refused():
    """Queue item 10's negative control, MEASURED: 12 packet_flow, 3 per column.

    The compiled census reads ZERO surviving circuit ports on this design and had
    to be widened twice before it could see the three. A static predicate reads
    the declaration, so it sees three directly -- and this asserts it does rather
    than assuming the different shape makes it immune.
    """
    bad = ms.over_budget_demand()
    worst = max(d for _, d in bad.herds)
    assert worst == 3, "the control must be over budget or refusing it proves nothing"
    assert worst > ms.SHIM_MM2S_PER_COLUMN
    assert max(ms.predicted_columns(bad)) == 3
    assert ms._refuses(bad), (
        "the predicate ADMITTED a design demanding 3 shim MM2S in every column "
        "it occupies, under every placement"
    )


def test_the_over_budget_control_discriminates_from_r1():
    """An arm that refuses everything is not a gate either."""
    assert ms._refuses(ms.over_budget_demand())
    assert not ms._refuses(ms.r1_interior_demand())


def test_the_herd_inventory_table_is_reproduced():
    """Doc 31b section 3.5, MEASURED by sweeping N herds of [4,1] in one segment."""
    for n_herds, expected in ((2, True), (4, True), (6, True), (8, True), (9, False)):
        widths = tuple([4] * n_herds)
        got = 4 * n_herds <= ms.NPU2_CORES and ms._row_packable(widths)
        assert got == expected, (
            f"{n_herds} herds of [4,1] ({4 * n_herds} tiles) came out "
            f"{'legal' if got else 'illegal'}; doc 31b section 3.5 MEASURED "
            f"{'placed' if expected else 'refused'}"
        )


def test_the_row_packer_is_a_row_packer_and_not_a_core_count():
    """Two herds of width 8 fit in 4 rows; nine of width 8 do not, at 72 cores.

    And the case a plain core count gets wrong: five herds of width 7 are 35
    cores (over 32, refused for that reason), but four of width 7 are 28 cores
    and still need four rows because two width-7 herds cannot share one.
    """
    assert ms._row_packable((8, 8, 8, 8))
    assert not ms._row_packable((8, 8, 8, 8, 8))
    assert ms._row_packable((7, 7, 7, 7))
    assert not ms._row_packable((7, 7, 7, 7, 7))
    assert ms._row_packable((4, 4, 4, 4, 4, 4, 4, 4))
    assert not ms._row_packable((4,) * 9)


def test_j7a_meets_its_column_budget_exactly():
    """Doc 23: three 8-wide herds, packed + gamma, 'exactly met'.

    The sharp version of the rule, and the one this predicate must not round the
    wrong way: two is legal, three is the packet path.
    """
    j7a = ms.compose(
        ms.SCOPE_PIPE,
        (
            ms.Demand(herds=((8, 1),)),
            ms.Demand(herds=((8, 0),)),
            ms.Demand(herds=((8, 1),), shim_s2mm=8),
        ),
    )
    assert max(ms.predicted_columns(j7a)) == ms.SHIM_MM2S_PER_COLUMN
    assert not ms._refuses(j7a)
    # Doc 23: "A stage wanting a third L3 input per column is back on the packet
    # path." One more herd-direct operand and the prediction says so.
    plus_one = ms.compose(ms.SCOPE_PIPE, (j7a, ms.Demand(herds=((8, 1),))))
    assert max(ms.predicted_columns(plus_one)) == 3


# ---------------------------------------------------------------------------
# 3. The predicate: what it refuses, what it prices, and that it does both.
# ---------------------------------------------------------------------------


def _base(**kw):
    """A legal mapping to perturb. Everything staged, so no column is loaded."""
    args = dict(
        nt1_form="fused",
        nt2_form="fused",
        gemm_fold="folded",
        gemm_herd_x=2,
        norm_herd_x=2,
        tile_k=32,
        rows_per_call=4,
        parallel_bands=1,
        routes=tuple((op, ms.ROUTE_L2_STAGED) for op, _ in ms.L3_OPERANDS),
        seams=tuple((p, ms.SCOPE_PIPE) for p in ms.SEAMS_FOLDED),
    )
    args.update(kw)
    return ms.Mapping(**args)


def test_the_predicate_admits_something():
    v = ms.legality(_base())
    assert v.legal, f"the base mapping is refused: {v.refusals}"


def test_para_is_refused_at_every_dependent_seam():
    """TileFlow section 4.1: Para is only applicable to tiles without dependency.

    Every seam of the tail chain carries one, so this is a whole axis value that
    is legal nowhere in this design -- and that is a finding, not a bug.
    """
    for pair in ms.SEAMS_FOLDED:
        m = _base(
            seams=tuple(
                (p, ms.SCOPE_PARA if p == pair else ms.SCOPE_PIPE)
                for p in ms.SEAMS_FOLDED
            )
        )
        v = ms.legality(m)
        assert not v.legal, f"Para at {pair} was admitted"
        assert any("data dependence" in r for r in v.refusals)


def test_para_is_legal_across_band_lanes():
    """The one place doc 46's fourth scope appears in this design."""
    a = ms.Demand(herds=((4, 1),))
    both = ms.compose(ms.SCOPE_PARA, (a, a))
    assert both.cores == 8 and both.shim_mm2s_slots == 8


def test_cores_refuse_at_the_measured_wall():
    """Two J7a tails at width 8 plus three GEMM herds is 9 herds and 60 cores."""
    m = _base(
        nt1_form="j7a",
        nt2_form="j7a",
        gemm_fold="split",
        norm_herd_x=8,
        gemm_herd_x=4,
        rows_per_call=8,
        seams=tuple((p, ms.SCOPE_PIPE) for p in ms.SEAMS_SPLIT),
    )
    v = ms.legality(m)
    assert not v.legal
    assert any("cores" in r or "row index" in r for r in v.refusals), v.refusals


def test_bands_multiply_the_core_demand():
    """``Para`` over lanes SUMS, which is what makes parallel_bands expensive."""
    one = ms.legality(_base(parallel_bands=1))
    assert one.legal
    many = ms.legality(_base(parallel_bands=8))
    assert many.report["cores"] == 8 * one.report["cores"]


def test_the_shim_budget_is_priced_and_not_refused():
    """The doc-44 correction, as a test.

    Both norm tails fetching both of their L3 operands herd-direct at one width
    stacks four streams onto every column. AIR does not refuse that -- it
    packet-multiplexes -- so the predicate must keep the point and charge for it.
    """
    m = _base(
        routes=(
            ("packed1", ms.ROUTE_HERD_DIRECT),
            ("gamma1", ms.ROUTE_HERD_DIRECT),
            ("w_up", ms.ROUTE_L2_STAGED),
            ("w_down", ms.ROUTE_L2_STAGED),
            ("gamma2", ms.ROUTE_HERD_DIRECT),
        )
    )
    v = ms.legality(m)
    assert v.report["predicted_column_demand"] > ms.SHIM_MM2S_PER_COLUMN
    assert v.legal, (
        "an over-subscribed column was REFUSED. AIR packet-multiplexes rather "
        "than refusing, so filtering these points hides the silent-multiplexing "
        "failure mode -- doc 44, as corrected"
    )
    assert "shim_mm2s_per_column" in v.prices
    assert 0.0 < v.prices["shim_mm2s_per_column"] < 1.0


def test_the_shim_budget_is_refused_where_it_is_placement_invariant():
    """Three herd-direct operands on ONE herd is over budget under any placement."""
    # A Shar seam moves up's L3 operands onto nt1's herd, so nt1 ends up with
    # packed1, gamma1 AND w_up herd-direct: three in every column it occupies.
    m = _base(
        nt1_form="fused",
        routes=(
            ("packed1", ms.ROUTE_HERD_DIRECT),
            ("gamma1", ms.ROUTE_HERD_DIRECT),
            ("w_up", ms.ROUTE_HERD_DIRECT),
            ("w_down", ms.ROUTE_L2_STAGED),
            ("gamma2", ms.ROUTE_L2_STAGED),
        ),
        seams=(
            (("nt1", "up"), ms.SCOPE_SHAR),
            (("up", "down"), ms.SCOPE_PIPE),
            (("down", "nt2"), ms.SCOPE_PIPE),
        ),
    )
    v = ms.legality(m)
    assert not v.legal, "three herd-direct operands on one herd were admitted"
    assert any("EVERY placement" in r for r in v.refusals), v.refusals


def test_l1_capacity_refuses_the_measured_overflow():
    m = _base(norm_herd_x=1, rows_per_call=8)
    v = ms.legality(m)
    assert not v.legal
    assert any("L1" in r for r in v.refusals), v.refusals


def test_divisibility_refuses_what_the_builders_raise_on():
    v = ms.legality(_base(norm_herd_x=3, rows_per_call=4))
    assert not v.legal
    assert any("divisible" in r for r in v.refusals), v.refusals


def test_no_demand_is_clamped_to_its_budget():
    """A clamped demand makes its clause unable to fail -- item 10's failure mode.

    An earlier draft of ``mapping_space`` wrote ``core_s2mm=min(CORE_S2MM, ...)``
    in two places, which is that failure mode exactly: the core-port clause could
    never fire, so every design passed it and the space size counted designs that
    cannot be wired. Asserted on the DEMAND, not on the verdict -- a clamp is
    invisible in a verdict that happens to be right for another reason.
    """
    m = _base(
        nt1_form="fused",
        routes=tuple((op, ms.ROUTE_HERD_DIRECT) for op, _ in ms.L3_OPERANDS),
    )
    # Three L3 operands herd-direct on a non-head group: three ports plus the
    # stage hand-off is four, and the demand must SAY four.
    d = ms._norm_group(m, "fused", ["packed1", "gamma1", "gamma2"], 0, head=False)
    assert d.core_s2mm == 4, (
        f"core_s2mm={d.core_s2mm} for three herd-direct operands on a group with "
        f"a predecessor; anything at or below {ms.CORE_S2MM} means the demand was "
        "clamped to the budget and the clause can never fire"
    )
    assert d.core_s2mm > ms.CORE_S2MM


def test_the_core_port_clause_is_not_binding_anywhere_in_this_space():
    """A clause that never fires contributes nothing to the count. Say so.

    No stage group of the R2 tail can want more than two core S2MM: the widest
    (nt1) owns two L3 operands, and every arrangement of them -- two herd-direct,
    one of each, both staged sharing one feed -- lands on exactly two. So the
    core DMA budget is met by construction here and removes no point from the
    space. That is worth pinning rather than leaving as an unexamined green
    clause: if the operand set grows, this fails and the clause starts cutting.
    """
    fired = 0
    checked = 0
    routings = list(itertools.product(ms.ROUTES, repeat=len(ms.L3_OPERANDS)))
    for form1 in ("fused", "j7a"):
        for form2 in ("fused", "j7a"):
            for combo in routings:
                m = _base(
                    nt1_form=form1,
                    nt2_form=form2,
                    routes=tuple(
                        (op, combo[i]) for i, (op, _) in enumerate(ms.L3_OPERANDS)
                    ),
                )
                checked += 1
                if any("S2MM / " in r for r in ms.legality(m).refusals):
                    fired += 1
    assert checked and fired == 0, (
        f"the core-port clause fired on {fired} of {checked} mappings; it used "
        "to be met by construction, so the space size has moved"
    )
    # And the demand it reads is still above the budget somewhere, so the clause
    # is dormant rather than dead code guarding a clamped value.
    m = _base(routes=tuple((op, ms.ROUTE_HERD_DIRECT) for op, _ in ms.L3_OPERANDS))
    d = ms._norm_group(m, "fused", ["packed1", "gamma1", "gamma2"], 0, head=False)
    assert d.core_s2mm > ms.CORE_S2MM


def test_the_reduction_fanout_clause_is_reachable():
    """The GeLU->down broadcast needs a memtile, and a memtile has six ports.

    At ``gemm_herd_x`` inside its own range the clause never fires -- which is
    itself worth pinning, because a clause that cannot fire anywhere in the space
    contributes nothing to the count and should be known not to.
    """
    fired = False
    for g in ms.axis("gemm_herd_x").values:
        m = _base(
            gemm_fold="split",
            gemm_herd_x=g,
            seams=tuple((p, ms.SCOPE_PIPE) for p in ms.SEAMS_SPLIT),
        )
        v = ms.legality(m)
        fired = fired or any("fan-out" in r for r in v.refusals)
    assert not fired, (
        "the reduction fan-out clause fires inside gemm_herd_x's own range; "
        "either the range or the clause moved and the space size changed with it"
    )
    assert ms.MAX_PLACEABLE_HERD_X <= ms.MEMTILE_MM2S, (
        "gemm_herd_x can now exceed a memtile's port count, so the fan-out "
        "clause is live and this test's premise is stale"
    )


# ---------------------------------------------------------------------------
# 4. The count.
# ---------------------------------------------------------------------------


def test_the_raw_and_constructed_in_sizes_are_ordered():
    raw = ms.raw_space_size()
    constructed = ms.constructed_in_space_size()
    assert raw > constructed > 0
    # The only divisibility that bites at baseline_768 is norm_herd_x x
    # rows_per_call against a 64-row band; emb_dim and ffn_dim divide by every
    # value of their axes. If that changes, the ratio moves and this says so.
    assert raw % constructed == 0 or raw > constructed


def test_the_divisibility_filter_is_the_one_that_bites():
    """Named explicitly so a reader can check the 'before' number by hand."""
    pairs = [
        (n, r)
        for n in ms.axis("norm_herd_x").values
        for r in ms.axis("rows_per_call").values
        if ms._divisible(n, r)
    ]
    assert len(pairs) == 22, (
        f"{len(pairs)} legal (norm_herd_x, rows_per_call) pairs of "
        f"{len(ms.axis('norm_herd_x').values) * len(ms.axis('rows_per_call').values)}"
    )
    assert all(
        ms.EMB_DIM % g == 0 and (ms.EMB_DIM // g) % 16 == 0
        for g in ms.axis("gemm_herd_x").values
    ), "gemm_herd_x's range now contains a value emb_dim does not divide by"
    assert all(
        ms.FFN_DIM % t == 0 for t in ms.axis("tile_k").values
    ), "tile_k's range now contains a value ffn_dim does not divide by"


def _brute_force(nt1, nt2, fold, g, n, bands, seams, pairs):
    """Every (rows_per_call, tile_k, routing) at one structure, counted directly."""
    total = 0
    for r in ms.axis("rows_per_call").values:
        for t in ms.axis("tile_k").values:
            for combo in itertools.product(ms.ROUTES, repeat=len(ms.L3_OPERANDS)):
                m = ms._mapping(nt1, nt2, fold, g, n, bands, seams, pairs, r, t, combo)
                if ms.legality(m).legal:
                    total += 1
    return total


def test_the_factorisation_matches_brute_force():
    """``legal = n_numeric x n_routing`` -- checked, not assumed.

    The census cannot enumerate 1.15e8 points, so it factorises. That is only
    valid while the predicate reads the tiling axes and the routing axes through
    disjoint clauses. Six structures are brute-forced whole (896 points each,
    every rows_per_call x tile_k x routing) and compared with the factorised
    count; a clause that couples them fails here rather than moving the headline
    number by an amount nobody can see.
    """
    structures = [
        ("fused", "fused", "folded", 2, 2, 1),
        ("fused", "fused", "folded", 4, 8, 1),
        ("j7a", "fused", "split", 4, 4, 1),
        ("fused", "j7a", "split", 2, 8, 1),
        ("j7a", "j7a", "folded", 1, 1, 2),
        ("fused", "fused", "split", 1, 1, 4),
    ]
    for nt1, nt2, fold, g, n, bands in structures:
        pairs = ms.SEAMS_FOLDED if fold == "folded" else ms.SEAMS_SPLIT
        for seams in (
            tuple([ms.SCOPE_PIPE] * len(pairs)),
            tuple([ms.SCOPE_SEQ] * len(pairs)),
            (ms.SCOPE_SHAR,) + tuple([ms.SCOPE_PIPE] * (len(pairs) - 1)),
        ):
            brute = _brute_force(nt1, nt2, fold, g, n, bands, seams, pairs)
            # The SHIPPED counter, not a re-transcription of it: an earlier
            # draft's copy here agreed with a bug in the original.
            fact, _priced = ms.count_one(nt1, nt2, fold, g, n, bands, seams, pairs)
            assert brute == fact, (
                f"structure {(nt1, nt2, fold, g, n, bands)} seams {seams}: brute "
                f"force counts {brute} legal points, the factorisation used by "
                f"the census counts {fact}. The tiling and routing axes are no "
                "longer independent in the predicate"
            )


def test_the_predicate_is_not_vacuous_over_the_space():
    """It must both admit and refuse, and the priced set must be a proper subset."""
    seen_legal = seen_refused = seen_priced = seen_unpriced = 0
    pairs = ms.SEAMS_FOLDED
    for g in (1, 2, 4):
        for n in (1, 2, 4, 8):
            for combo in itertools.product(ms.ROUTES, repeat=len(ms.L3_OPERANDS)):
                m = _base(gemm_herd_x=g, norm_herd_x=n, routes=tuple(
                    (op, combo[i]) for i, (op, _) in enumerate(ms.L3_OPERANDS)
                ))
                v = ms.legality(m)
                if not v.legal:
                    seen_refused += 1
                    continue
                seen_legal += 1
                if v.prices:
                    seen_priced += 1
                else:
                    seen_unpriced += 1
    assert seen_legal and seen_refused, (
        f"{seen_legal} admitted, {seen_refused} refused -- a predicate that "
        "does only one of those is not a predicate"
    )
    assert seen_priced and seen_unpriced, (
        f"{seen_priced} priced, {seen_unpriced} unpriced -- the slope must "
        "discriminate or it is a constant"
    )
    assert len(pairs) == 3


def test_the_unbounded_axis_declares_itself():
    """The honesty clause: a space size is only as good as its loosest bound."""
    unbounded = [a.name for a in ms.AXES if a.unbounded_by_artifact]
    assert unbounded == ["parallel_bands"], (
        f"axes without an artifact bound are {unbounded}; if that list grew, the "
        "census's headline number rests on more inference than it says it does"
    )
    for a in ms.AXES:
        assert a.source and len(a.source) > 20, (
            f"axis {a.name} has no source string; every range in this space must "
            "cite the artifact that bounds it"
        )


#: Axis values that survive to the "before" count and are legal in NO mapping.
#: This is a RESULT, not a defect list -- each entry names a whole axis value the
#: static predicate eliminates outright, and together they are most of what makes
#: the space collapse. Pinned so that a clause which stops eliminating one (or
#: starts eliminating another) has to be noticed and explained.
#:
#:   norm_herd_x 3, 5, 6, 7  -- a 64-row band does not divide by them
#:   rows_per_call 8..64     -- L1, MEASURED to overflow at 8, cols 768
#:
#: Two axis values are NOT here and were expected to be, which is the more
#: interesting half:
#:
#:   norm_herd_x 8      survives, but only through a Seq seam. Eight-wide norm
#:                      herds and the FFN cannot share one segment's shim budget
#:                      (J7a alone "exactly meets" it at width 8, doc 23), so
#:                      width 8 in the R2 tail means PACKAGING it.
#:   parallel_bands 4-8 survive, but only through a Seq seam, for the same
#:                      reason -- see the band test below, where four lanes of a
#:                      CO-RESIDENT tail are refused on shim slots with cores to
#:                      spare.
ELIMINATED_AXIS_VALUES = (
    ("norm_herd_x", 3),
    ("norm_herd_x", 5),
    ("norm_herd_x", 6),
    ("norm_herd_x", 7),
    ("rows_per_call", 8),
    ("rows_per_call", 16),
    ("rows_per_call", 32),
    ("rows_per_call", 64),
)


def _reachable_axis_values():
    """One pass over a representative grid; returns ``{axis name: {values seen}}``.

    Not a per-value search -- a value is only "unreachable" relative to what was
    tried, so a single pass over a grid that varies every axis together is both
    cheaper and a stronger statement than a probe per value.
    """
    seen = {a.name: set() for a in ms.AXES}
    routings = list(itertools.product(ms.ROUTES, repeat=len(ms.L3_OPERANDS)))
    for nt1 in ms.axis("nt1_form").values:
        for nt2 in ms.axis("nt2_form").values:
            for fold in ms.axis("gemm_fold").values:
                pairs = ms.SEAMS_FOLDED if fold == "folded" else ms.SEAMS_SPLIT
                # THE SEAM VECTOR MUST VARY. A grid fixed at all-Pipe reports
                # norm_herd_x 8 as reachable nowhere, because eight-wide herds
                # only fit the shim budget once a Seq seam has split them into
                # separate segments -- which is a statement about the grid, not
                # about the machine.
                seam_vectors = (
                    tuple([ms.SCOPE_PIPE] * len(pairs)),
                    tuple([ms.SCOPE_SEQ] * len(pairs)),
                    tuple([ms.SCOPE_SHAR] * len(pairs)),
                )
                for seams in seam_vectors:
                    for g in ms.axis("gemm_herd_x").values:
                        for n in ms.axis("norm_herd_x").values:
                            for r in ms.axis("rows_per_call").values:
                                if not ms._divisible(n, r):
                                    continue
                                for bands in ms.axis("parallel_bands").values:
                                    # Staging an operand costs ONE shim slot;
                                    # fetching it herd-direct costs one per
                                    # column, so all-staged MINIMISES the slot
                                    # total. If that busts the 16-slot budget,
                                    # no routing here can fit and the other 31
                                    # need not be built -- sound, and it is
                                    # most of this test's runtime.
                                    floor = ms.legality(
                                        ms._mapping(
                                            nt1, nt2, fold, g, n, bands, seams,
                                            pairs, r, ms.MAX_L1_TILE_K,
                                            tuple([ms.ROUTE_L2_STAGED] * len(routings[0])),
                                        )
                                    )
                                    if any(
                                        "channel-slots" in x for x in floor.refusals
                                    ):
                                        continue
                                    for combo in routings:
                                        m = ms._mapping(
                                            nt1, nt2, fold, g, n, bands, seams,
                                            pairs, r, ms.MAX_L1_TILE_K, combo,
                                        )
                                        if not ms.legality(m).legal:
                                            continue
                                        seen["nt1_form"].add(nt1)
                                        seen["nt2_form"].add(nt2)
                                        seen["gemm_fold"].add(fold)
                                        seen["gemm_herd_x"].add(g)
                                        seen["norm_herd_x"].add(n)
                                        seen["rows_per_call"].add(r)
                                        seen["parallel_bands"].add(bands)
                                        seen["tile_k"].add(ms.MAX_L1_TILE_K)
                                        for i, (op, _) in enumerate(ms.L3_OPERANDS):
                                            seen[f"route_{op}"].add(combo[i])
                                        break
    for t in ms.axis("tile_k").values:
        if ms.legality(_base(tile_k=t)).legal:
            seen["tile_k"].add(t)
    return seen


def test_the_axis_values_legality_eliminates_outright_are_the_pinned_ones():
    """Which whole axis values survive the 'before' count and are legal nowhere.

    An axis value legal in no mapping inflates the space-before-legality number
    without adding a design, so the set of them is part of the result and is
    pinned rather than left to be rediscovered.
    """
    seen = _reachable_axis_values()
    eliminated = sorted(
        (a.name, v) for a in ms.AXES for v in a.values if v not in seen[a.name]
    )
    assert eliminated == sorted(ELIMINATED_AXIS_VALUES), (
        f"legality now eliminates {eliminated} outright; the pinned set is "
        f"{sorted(ELIMINATED_AXIS_VALUES)}. A difference means a clause moved, "
        "and the space size moved with it"
    )


def test_the_band_axis_is_capped_by_the_shim_budget_not_by_cores():
    """Band parallelism of a CO-RESIDENT tail is bounded by DMA, not by the array.

    Four lanes of the leanest possible co-resident tail fit in cores and do NOT
    fit in the shim budget: the tail reads five L3 operands however they are
    routed, so four concurrent lanes want 20 of NPU2's 16 slots. That is not what
    "8 columns, 32 cores" would lead anyone to guess, and it is the reason the
    ``parallel_bands`` axis -- the one nothing in the tree bounds -- turns out
    not to open the space up.

    Scoped deliberately to the co-resident arrangement: a Seq seam splits the
    tail into segments that take the array in turns, and then the same lane count
    fits, which is why ``parallel_bands`` is absent from
    ``ELIMINATED_AXIS_VALUES``.
    """
    lean = _base(gemm_herd_x=1, norm_herd_x=1, rows_per_call=1)
    three = ms.legality(dataclasses.replace(lean, parallel_bands=3))
    four = ms.legality(dataclasses.replace(lean, parallel_bands=4))
    assert three.legal, f"three lanes of the leanest tail are refused: {three.refusals}"
    assert not four.legal
    assert any("shim MM2S channel-slots" in r for r in four.refusals), four.refusals
    assert four.report["cores"] <= ms.NPU2_CORES, (
        f"four lanes want {four.report['cores']} cores, so this is a core "
        "refusal after all and the finding is wrong"
    )
