# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The blend space's SHAPE, derived from the configs, with no device.

CONTRACT
    Resolves ``pattern/coarse/cells.py``'s cells at the catalogue's shape and
    at the second ladder rung, and asserts what the cells are before any of
    them runs: the predicted dispatch vectors the two lit recipes pin, the
    ordinal bracket the pair owns, the input list each front decides, and the
    refusal that keeps the four already-owned cells from being reimplemented
    here.

WHY A STRUCTURAL ARM AT ALL, AND WHY IT IS WRITTEN FIRST
    A cell's claim is about DISPATCH GRANULARITY, and a numeric gate cannot see
    it -- C2 and C3 compute the same layer as ``coarse`` and would pass their
    stage comparisons at any granularity, including one that quietly collapsed
    to a mode that already exists. The prediction has to be written down before
    the hardware run so the hardware run can be checked against it rather than
    described by it. That is the same reasoning behind
    ``ffn_accum_structure.py`` and ``norm_tail_structure.py``, one level up: the
    thing being claimed is structural, so something structural has to check it.

WHAT MAKES THE PREDICTION CREDIBLE RATHER THAN PLAUSIBLE
    ``cell_dispatch_prediction`` composes each half's contribution
    independently, so of its four front x tail combinations TWO are already
    pinned by shipped gates -- ``(block, banded)`` is ``coarse``'s recorded
    4 submissions / 131 entries and ``(runlist, decomposed)`` is ``runlist``'s
    17 / 427. Those two are checked here against literals taken from the lit
    recipes. A model that reproduces both endpoint cells from the same
    arithmetic that predicts the interior ones is a model, not a guess.

NO NPU, NO PEANO, NO COMPILE. Every claim here comes from ``block_config`` and
``runlist_config``, which resolve registry rows and derive band sizes without
building a module. Resolving them is also what exercises the two guards a cell
introduces: the cross-half GEMM object-collision check, and the assertion that
both halves derived the same band size.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_HERE)  # programming_examples/
for _p in (_PROJ_ROOT, os.path.join(_PROJ_ROOT, "llms"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from builders.block import BLOCK_INPUT_NAMES  # noqa: E402
from pattern.coarse.cells import (  # noqa: E402
    BUILDABLE_CELLS,
    CELLS,
    GOLDEN_INPUT_NAMES,
    FRONT_BLOCK,
    FRONT_RUNLIST,
    TAIL_BANDED,
    TAIL_DECOMPOSED,
    cell_config,
    cell_dispatch_prediction,
    cell_input_names,
)
from pattern.runlist.runlist import RUNLIST_INPUT_NAMES  # noqa: E402

#: The gate configuration, and the shape both new cells' catalogue rows claim.
GATE_SHAPE = {
    "seq_len": 4096,
    "emb_dim": 768,
    "ffn_dim": 3072,
    "num_heads": 12,
    "head_dim": 64,
}

#: The second rung `coarse` is to be measured at. 2048 is the shortest length
#: where `fused`'s stitched tail is already unbuildable (1365-row cap), which is
#: what makes it the shortest length where `coarse` is a mode rather than
#: `fused` under another name.
LADDER_SHAPE = dict(GATE_SHAPE, seq_len=2048)

#: What the two SHIPPED gates pin, at GATE_SHAPE, as (submissions, entries).
#: Read off run_npu2_coarse_peano.lit and run_npu2_runlist_peano.lit; these are
#: the two cells of the space that already have mode names and measurements.
MEASURED_ENDPOINTS = {
    (FRONT_BLOCK, TAIL_BANDED): (4, 131),
    (FRONT_RUNLIST, TAIL_DECOMPOSED): (17, 427),
}


def _norm_blocks(shape):
    """``norm_blocks`` at this shape, from the same derivation both halves use."""
    from builders.block import norm_rows

    return shape["seq_len"] // norm_rows(shape["seq_len"], shape["emb_dim"])


def test_the_prediction_reproduces_both_measured_endpoints():
    """The model's validation: two of its four combinations are already gated."""
    blocks = _norm_blocks(GATE_SHAPE)
    for (front, tail), expected in MEASURED_ENDPOINTS.items():
        name = next(n for n, c in CELLS.items() if c.front == front and c.tail == tail)
        got = cell_dispatch_prediction(name, GATE_SHAPE["num_heads"], blocks)
        assert got == expected, (
            f"cell {name} ({front} front, {tail} tail) predicts {got} but the "
            f"shipped gate for it pins {expected}. The prediction composes each "
            "half independently, so a mismatch on an endpoint means the model "
            "is wrong about a half and every interior cell it predicts is wrong "
            "with it."
        )


def test_the_two_new_cells_predict_the_bracket_they_claim():
    """coarse < C3 < C2 < runlist, at both measured lengths.

    Ordinal and never threshold, per 08-phase-e-execution-strategies.md: each
    cell refines exactly one half of C1 and neither refines both, so each must
    land strictly between the incumbent and the fully decomposed mode. A cell
    outside the bracket is not the cell it claims to be.
    """
    for shape in (GATE_SHAPE, LADDER_SHAPE):
        blocks = _norm_blocks(shape)
        heads = shape["num_heads"]
        entries = {
            name: cell_dispatch_prediction(name, heads, blocks)[1]
            for name in ("C1", "C3", "C2", "C6")
        }
        ordered = [entries["C1"], entries["C3"], entries["C2"], entries["C6"]]
        assert ordered == sorted(ordered) and len(set(ordered)) == 4, (
            f"at seq {shape['seq_len']} the entry counts are {entries}; the "
            "claim is coarse(C1) < C3 < C2 < runlist(C6), strictly"
        )


def test_the_front_decides_the_input_list_and_the_injection_index():
    """The measured ln1_weight target is READ from the front's own list."""
    assert cell_input_names("C2") == BLOCK_INPUT_NAMES
    assert cell_input_names("C3") == RUNLIST_INPUT_NAMES
    for name in BUILDABLE_CELLS:
        names = cell_input_names(name)
        assert "ln1_weight" in names, f"{name}: nothing to inject into"
        # Same target every whole-layer mode uses, at each list's own index --
        # 3 in the block front's fused-w_qkv list, 5 in the runlist front's
        # split q/k/v one. The index is a CONSEQUENCE of the front, which is
        # why a cell reads it rather than declaring it.
        expected = 3 if CELLS[name].front == FRONT_BLOCK else 5
        assert names.index("ln1_weight") == expected, (
            f"{name}: ln1_weight is at {names.index('ln1_weight')} in {names}, "
            f"expected {expected}"
        )
        # Every name the front asks for must be one the preparer can draw.
        # Otherwise this is a KeyError minutes into a hardware run, after the
        # ELFs have compiled.
        missing = [n for n in names if n not in GOLDEN_INPUT_NAMES]
        assert not missing, f"{name}: prepare_cell cannot draw {missing}"


def test_the_owned_cells_refuse_to_be_rebuilt_here():
    """Four of the six cells already exist; building one here would be a fork."""
    for name, cell in CELLS.items():
        if cell.buildable_here:
            continue
        try:
            cell_config(name, **GATE_SHAPE)
        except ValueError as exc:
            assert cell.owner.split()[0] in str(exc), (
                f"{name}'s refusal should name where the cell actually lives, "
                f"got: {exc}"
            )
        else:  # pragma: no cover - the failure path is the point
            raise AssertionError(
                f"cell_config built {name}, which is {cell.owner}. A second "
                "implementation measures something no gate validated."
            )


def test_the_buildable_cells_resolve_at_both_lengths():
    """Resolution is what runs the two guards a composed cell introduces.

    The cross-half GEMM object-collision check, and the assertion that both
    halves derived the same band size. Neither can be exercised without
    resolving both halves together, and both fail loudly by design.
    """
    for shape in (GATE_SHAPE, LADDER_SHAPE):
        for name in BUILDABLE_CELLS:
            cfgs = cell_config(name, **shape)
            assert cfgs["cell"] == name
            assert cfgs["block"]["norm_rows"] == cfgs["runlist"]["norm_rows"]
            cell = CELLS[name]
            for key in cell.block_keys:
                assert key in cfgs["block"]["artifacts"], f"{name}: block {key}"
            for key in cell.runlist_keys:
                assert key in cfgs["runlist"]["artifacts"], f"{name}: runlist {key}"


def test_every_cell_of_the_space_is_accounted_for():
    """Six cells, two axes, and four of them already have owners."""
    assert len(CELLS) == 6, "the space is front {2} x tail {3}"
    assert set(BUILDABLE_CELLS) == {"C2", "C3"}
    owned = {n for n, c in CELLS.items() if not c.buildable_here}
    assert owned == {"C1", "C4", "C5", "C6"}


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    blocks = _norm_blocks(GATE_SHAPE)
    for name in ("C1", "C3", "C2", "C6"):
        subs, entries = cell_dispatch_prediction(name, GATE_SHAPE["num_heads"], blocks)
        cell = CELLS[name]
        print(
            f"  [cell] {name} {cell.front:7s} front + {cell.tail:11s} tail: "
            f"{subs} submissions, {entries} entries"
        )
    print(f"coarse cell structure: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
