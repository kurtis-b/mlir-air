# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``coarse``'s blend space: the cells, and the two that are new.

CONTRACT
    ``prepare_cell(cell, shape, seed=..., cache_dir=..., label=...,
    extra=...)`` is the ``SPECS`` preparer for one interior cell of
    28-coarse-blend-space.md's blend space, presented to ``opcheck.py``'s
    ``dispatch`` seam exactly as every other whole-layer mode is: the shared
    golden model, the measured ``ln1_weight`` injection target, per-boundary
    comparisons at ``BLOCK_STAGE_ATOL``, and one ``DispatchVector`` recorded
    per dispatched sequence.

WHAT A CELL IS
    `28-coarse-blend-space.md` derives the blend space from the artifact plans
    rather than from the definition's wording. It is TWO axes, not a choice per
    operator, because ``fused`` and ``coarse`` build their front from the same
    two modules and differ in the tail alone::

        front  {block-form, runlist-form}  x  tail  {stitched, banded, decomposed}

    Six cells, and two of them already have mode names -- ``(block, stitched)``
    IS ``fused`` and ``(runlist, decomposed)`` IS ``runlist``. This module
    builds the two INTERIOR cells that do not::

        C2 = (block front, decomposed tail)
        C3 = (runlist front, banded tail)

    ``C1 = (block, banded)`` is today's ``coarse`` and is NOT built here -- see
    the footguns.

WHY THE MEASUREMENT BELONGS AT 2048/4096 AND NOT AT 1024
    ``fused``'s stitched tail needs a ``plane_major`` plane stride of
    ``rows * cols`` against the shim ``aie.dma_bd`` cap of 1,048,576, so it
    caps at 1365 rows (``builders/norm_tail.py``). At seq 2048 and above the
    whole stitched row of the table is unbuildable, and THAT is what makes
    ``coarse`` a mode rather than ``fused`` under another name: it is the blend
    you use where full fusion does not fit. At 1024 every cell is dominated by
    one that already has a mode name, so a winner declared there would collapse
    the taxonomy from four points to three. Do not measure the cells at 1024.

THE ENTRY COUNTS ARE DERIVED, AND THE DERIVATION IS ALREADY VALIDATED
    ``cell_dispatch_prediction`` composes each half's contribution from
    ``norm_blocks`` and ``num_heads`` rather than counting anything at run
    time, so the structure test can pin a cell's shape with no device. Two of
    its four combinations are already pinned by shipped gates, which is what
    says the model is right rather than plausible: ``(block, banded)`` predicts
    4 submissions / 131 entries, which is ``run_npu2_coarse_peano.lit``'s
    literal, and ``(runlist, decomposed)`` predicts 17 / 427, which is
    ``run_npu2_runlist_peano.lit``'s.

FOOTGUNS
    - **This module does not build C1.** ``(block, banded)`` is
      ``builders/block.py`` dispatched through ``pattern/coarse/coarse.py``,
      and a second implementation of it here would be a fork measuring
      something D2 never validated -- the failure ``coarse.py``'s own docstring
      warns about. ``CELLS`` describes all six cells because the blend space is
      six cells; ``run_cell`` builds two of them and points at the owning module
      for the rest.
    - **Each cell needs its OWN ELF cache directory.** ``KernelCache`` picks the
      directory by NAME, so two modes pointed at one can trade ELFs whose
      fingerprints happen to agree -- numerically valid output attributed to the
      wrong execution boundary, which no equivalence check would surface.
    - **A cell compiles a SUBSET of each half's artifacts**, passed through the
      ``keys`` parameter both ``compile_*_artifacts`` take. Compiling the union
      would build large ELFs the cell never dispatches, and would put GEMM
      objects on disk that the collision check below has no reason to consider.
    - **The two halves' GEMM objects can collide, and the guard is per config
      until a cell merges two.** ``ek.compile_gemm_mm`` names its object from
      ``(tile_m, tile_n)`` alone while ``tile_k_l1`` is a compile flag, so two
      GEMMs agreeing on the first two and differing on the third write one file
      with two micro-kernels and whichever compiles last wins -- silently, as
      D2 measured between the FFN's up-projection and the o-projection.
      ``runlist_config`` and ``offload_config`` each check their own specs;
      nothing checked ACROSS a block half and a runlist half until
      ``_check_cell_objects``.
    - **The gamma form differs between the tails and is adapted here, not in a
      callee.** The banded tail's ``addnorm`` takes the ``[emb]`` weight
      directly (``builders/addnorm.py`` declares ``l3_w_ty`` as ``[cols]``);
      the decomposed tail's ``elementwise_mul`` takes a host-materialized
      ``[norm_rows, emb]`` broadcast. Passing one where the other is expected
      is a shape error at best and a wrong broadcast at worst.
    - **The input list is decided by the FRONT**, because the block front takes
      one fused ``w_qkv`` and the runlist front takes ``w_q``/``w_k``/``w_v``
      separately. ``cell_input_names`` returns the right tuple, and the
      ``ln1_weight`` injection index is read from it rather than written down --
      the target itself is the measured one (``opcheck_layer.py``) and is not
      re-picked per cell.
    - ``execution_mode`` for every cell is ``coarse``'s CSV value: these are
      cells of one mode, not new taxonomy points. They separate in a results
      tree by ``study_case_label`` and by the per-mode CSV filename, and the
      cell is recorded in the artifact as ``blend_cell``.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE_ROOT = os.path.dirname(os.path.dirname(_HERE))  # transformer_layer/
_PROJ_ROOT = os.path.dirname(_EXAMPLE_ROOT)  # programming_examples/
for _p in (_PROJ_ROOT, os.path.join(_PROJ_ROOT, "llms"), _EXAMPLE_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The block half's region functions. They are module-private by name and are
# nonetheless the reuse surface 28-coarse-blend-space.md names -- "module-level
# and reusable as-is, which is what makes C1 and C2's front free". Imported
# rather than reimplemented: a copy here would be a fork of the D2 gate.
from builders.block import (  # noqa: E402
    BLOCK_INPUT_NAMES,
    _sequence_a,
    _sequence_ffn,
    _sequence_norm,
    block_config,
    compile_block_artifacts,
)
from builders.elementwise_mul import broadcast_row_weight  # noqa: E402
from opcheck_layer import (  # noqa: E402
    BLOCK_STAGE_ATOL,
    print_dispatch_totals,
    reconfiguration_delta,
)
from opcheck_prepare import _spec_digest  # noqa: E402
from pattern import EXECUTION_MODE_CSV  # noqa: E402
from pattern.blocked_attention import round_bf16  # noqa: E402

# The runlist half's region functions, extracted to module scope for exactly
# this composition. See that module's docstring for what each submission holds
# and why the per-head split is a memory bound rather than a schedule choice.
from pattern.offload.offload import _check_no_object_collision  # noqa: E402
from pattern.reference import (  # noqa: E402
    ENCODER_BOUNDARIES,
    fuse_qkv_weight,
    generate_golden_reference,
)
from pattern.runlist.runlist import (  # noqa: E402
    RUNLIST_INPUT_NAMES,
    compile_runlist_artifacts,
    run_attention_interior,
    run_ffn,
    run_norm_chain,
    run_o_proj,
    run_projections,
    runlist_config,
)

#: Front levels: how qkv -> attention -> o_proj is dispatched.
FRONT_BLOCK = "block"
FRONT_RUNLIST = "runlist"

#: Tail levels: how ln1 -> ffn -> ln2 is dispatched.
TAIL_STITCHED = "stitched"
TAIL_BANDED = "banded"
TAIL_DECOMPOSED = "decomposed"


class Cell:
    """One cell of the blend space, and who owns it.

    ``owner`` is ``None`` for the two interior cells this module builds, and
    otherwise names the module that already implements the cell -- which is the
    scoping finding stated as data: the space ``coarse`` blends over CONTAINS
    the two things it blends.
    """

    def __init__(self, name, front, tail, block_keys, runlist_keys, owner=None):
        self.name = name
        self.front = front
        self.tail = tail
        self.block_keys = block_keys
        self.runlist_keys = runlist_keys
        self.owner = owner

    @property
    def buildable_here(self):
        return self.owner is None

    def __repr__(self):  # pragma: no cover - diagnostics only
        return f"Cell({self.name}: {self.front} front, {self.tail} tail)"


#: The whole blend space, six cells. Only the two with ``owner=None`` are built
#: here; the other four are recorded so the space is data rather than prose, and
#: so a reader can see that four of the six already exist.
CELLS = {
    "C1": Cell(
        "C1",
        FRONT_BLOCK,
        TAIL_BANDED,
        ("qkv_proj", "mha_out_proj", "ffn", "addnorm"),
        (),
        owner="pattern/coarse/coarse.py (today's `coarse`, over builders/block.py)",
    ),
    "C2": Cell(
        "C2",
        FRONT_BLOCK,
        TAIL_DECOMPOSED,
        ("qkv_proj", "mha_out_proj"),
        ("add", "ln", "mul", "up", "gelu", "down"),
    ),
    "C3": Cell(
        "C3",
        FRONT_RUNLIST,
        TAIL_BANDED,
        ("ffn", "addnorm"),
        (
            "q_proj",
            "k_proj",
            "v_proj",
            "attn_scores",
            "softmax",
            "attn_output",
            "o_proj",
        ),
    ),
    "C4": Cell(
        "C4",
        FRONT_BLOCK,
        TAIL_STITCHED,
        (),
        (),
        owner="pattern/fused/fused.py (this cell IS `fused`; unbuildable at seq >= 2048)",
    ),
    "C5": Cell(
        "C5",
        FRONT_RUNLIST,
        TAIL_STITCHED,
        (),
        (),
        owner="nothing -- the expensive cell (the stitched tail reads a packed "
        "plane the front must write) and moot at seq >= 2048, where no "
        "stitched tail builds",
    ),
    "C6": Cell(
        "C6",
        FRONT_RUNLIST,
        TAIL_DECOMPOSED,
        (),
        (),
        owner="pattern/runlist/runlist.py (this cell IS `runlist`)",
    ),
}

#: The cells this module builds, which is what a catalogue row may name.
BUILDABLE_CELLS = tuple(name for name, cell in CELLS.items() if cell.buildable_here)

#: Every host tensor ``prepare_cell`` knows how to draw from the golden model.
#: A cell's input list must be a subset of this, and ``coarse_cells_structure.py``
#: asserts it with no device: a name a front asks for and the preparer cannot
#: fill is a ``KeyError`` minutes into a hardware run otherwise.
GOLDEN_INPUT_NAMES = (
    "x",
    "w_qkv",
    "w_q",
    "w_k",
    "w_v",
    "w_o",
    "ln1_weight",
    "w_up",
    "w_down",
    "ln2_weight",
)


def cell_input_names(cell_name):
    """The host tensors this cell's layer takes, in fault-injection order.

    Decided by the FRONT: the block form takes one fused ``w_qkv``, the runlist
    form takes ``w_q``/``w_k``/``w_v``. Returning the existing tuples rather
    than restating them keeps a cell's injection index tied to the same list
    the front's own mode uses.
    """
    front = CELLS[cell_name].front
    return BLOCK_INPUT_NAMES if front == FRONT_BLOCK else RUNLIST_INPUT_NAMES


def cell_dispatch_prediction(cell_name, num_heads, norm_blocks):
    """``(submissions, entries)`` this cell will dispatch, derived not counted.

    Each half contributes independently, which is what makes the blend space a
    product rather than a list:

    ==================  ===========  ==========================
    half                submissions  entries
    ==================  ===========  ==========================
    front ``block``     1            2
    front ``runlist``   2 + heads    4 + 3 * heads
    tail ``banded``     3            1 + 2 * norm_blocks
    tail ``decomposed`` 3            3 + 6 * norm_blocks
    ==================  ===========  ==========================

    Validated against two shipped gates rather than asserted: ``(block,
    banded)`` gives (4, 131) and ``(runlist, decomposed)`` gives (17, 427),
    which are ``coarse``'s and ``runlist``'s pinned lit literals at the gate
    configuration. A stitched tail has no entry here -- it is one entry in one
    submission, but no cell using it is buildable at the lengths ``coarse`` is
    measured at, and a number nothing can run is a number nobody can check.
    """
    cell = CELLS[cell_name]
    if cell.front == FRONT_BLOCK:
        submissions, entries = 1, 2
    else:
        submissions, entries = 2 + num_heads, 4 + 3 * num_heads
    if cell.tail == TAIL_BANDED:
        submissions, entries = submissions + 3, entries + 1 + 2 * norm_blocks
    elif cell.tail == TAIL_DECOMPOSED:
        submissions, entries = submissions + 3, entries + 3 + 6 * norm_blocks
    else:
        raise ValueError(
            f"cell {cell_name} has a {cell.tail} tail, which dispatches through "
            f"{cell.owner}; this prediction covers the banded and decomposed "
            "tails only"
        )
    return submissions, entries


def _check_cell_objects(cell_name, cfgs):
    """Raise if the cell's two halves compile two micro-kernels to one object.

    Each half's own config already checks itself; nothing checked across a
    block half and a runlist half until a cell put them in one working
    directory. The specs are collected per HALF and by the artifact keys the
    cell actually builds, so an unused GEMM cannot manufacture a collision.
    """
    cell = CELLS[cell_name]
    specs = {}
    if "qkv_proj" in cell.block_keys:
        specs["block:qkv_proj"] = (cfgs["block"]["qkv_spec"], None)
    if "mha_out_proj" in cell.block_keys:
        specs["block:o_proj"] = (cfgs["block"]["o_proj_spec"], None)
    if "ffn" in cell.block_keys:
        specs["block:ffn_up"] = (cfgs["block"]["ffn_up_spec"], None)
        specs["block:ffn_down"] = (cfgs["block"]["ffn_down_spec"], None)
    for key in cell.runlist_keys:
        gemms = cfgs["runlist"]["gemms"]
        if key in gemms:
            specs[f"runlist:{key}"] = cfgs["runlist"]["specs"][gemms[key]]
    _check_no_object_collision(specs)


def cell_config(cell_name, seq_len, emb_dim, ffn_dim, num_heads, head_dim):
    """Resolve both halves' configurations without building anything.

    The two halves are kept SIDE BY SIDE rather than merged into one flat
    config, and that is not style. Both carry ``artifacts``,
    ``backend_kwargs``, ``norm_rows`` and ``specs`` under those names, so a
    merge would need a rename policy, and every function this module composes
    already takes the config its own mode validated. Passing each half its own
    config untouched is what makes the composition a composition rather than a
    reimplementation.
    """
    if cell_name not in CELLS:
        raise KeyError(f"unknown cell {cell_name!r}; known: {sorted(CELLS)}")
    cell = CELLS[cell_name]
    if not cell.buildable_here:
        raise ValueError(
            f"cell {cell_name} ({cell.front} front, {cell.tail} tail) is not "
            f"built here: it is {cell.owner}. Measuring it means running that "
            "mode, not adding a second implementation of it."
        )
    cfgs = {
        "cell": cell_name,
        "block": block_config(seq_len, emb_dim, ffn_dim, num_heads, head_dim),
        "runlist": runlist_config(seq_len, emb_dim, ffn_dim, num_heads, head_dim),
    }
    if cfgs["block"]["norm_rows"] != cfgs["runlist"]["norm_rows"]:
        raise AssertionError(
            "the two halves resolved different band sizes "
            f"({cfgs['block']['norm_rows']} vs {cfgs['runlist']['norm_rows']}); "
            "both derive from builders.block.norm_rows, so this is a real "
            "divergence and the cell's tail would not be at coarse's schedule"
        )
    _check_cell_objects(cell_name, cfgs)
    return cfgs


def describe_cell(cfgs):
    """One line per resolved decision, for the run log and the lit gate."""
    cell = CELLS[cfgs["cell"]]
    block, runlist = cfgs["block"], cfgs["runlist"]
    submissions, entries = cell_dispatch_prediction(
        cell.name, block["num_heads"], block["norm_blocks"]
    )
    print(
        f"  coarse {cell.name} {block['seq_len']}x{block['emb_dim']} ffn "
        f"{block['ffn_dim']} {block['num_heads']}h x {block['head_dim']} "
        f"(encoder_bert, non-causal, {cell.front} front + {cell.tail} tail, "
        f"{entries} entries over {submissions} runlists)"
    )
    if cell.front == FRONT_BLOCK:
        print(
            f"    front: qkv_proj {block['qkv_spec']['method']} "
            f"({block['qkv_source']}), mha_out_proj + o_proj "
            f"{block['o_proj_spec']['method']} ({block['o_proj_source']}), "
            "q/k/v device-resident"
        )
    else:
        spec, (m, k, n) = runlist["specs"]["proj"]
        print(
            f"    front: q/k/v/o_proj {m}x{k}x{n} {spec['method']} (registry), "
            f"per head attn_scores + softmax + attn_output (injected tiles)"
        )
    if cell.tail == TAIL_BANDED:
        print(
            f"    tail: ffn {block['ffn_up_spec']['method']} up / "
            f"{block['ffn_down_spec']['method']} down ({block['ffn_source']}), "
            f"addnorm pre-add {block['norm_rows']}x{block['emb_dim']} "
            f"x{block['norm_blocks']} dispatches"
        )
    else:
        up_spec, _ = runlist["specs"]["up"]
        down_spec, _ = runlist["specs"]["down"]
        print(
            f"    tail: up {up_spec['method']} / gelu / down "
            f"{down_spec['method']} (registry), norm chains "
            f"{block['norm_rows']} rows x{block['norm_blocks']} of "
            "(add + layer_norm + mul)"
        )


def compile_cell_artifacts(cache, cfgs, run_only=False):
    """Compile only the artifacts this cell dispatches, from both halves.

    Each half compiles through its OWN function, at the subset the cell names.
    Both merge into the one fingerprint file the cache keeps
    (``builders/block_cache.py::save_fingerprints``), so a cell's cache records
    what it built and reuses only exact matches, exactly as a single-half mode
    does.
    """
    cell = CELLS[cfgs["cell"]]
    if cell.block_keys:
        compile_block_artifacts(
            cache, cfgs["block"], run_only=run_only, keys=cell.block_keys
        )
    if cell.runlist_keys:
        compile_runlist_artifacts(
            cache, cfgs["runlist"], run_only=run_only, keys=cell.runlist_keys
        )


def _run_front(cache, cfgs, inputs):
    """The cell's front: q/k/v, attention, output projection.

    Returns ``(boundaries, vectors)`` where ``boundaries`` holds ``q``, ``k``,
    ``v``, ``attn_context`` and ``attn_out`` under the names
    ``pattern/reference.py`` uses -- the same five from either level, which is
    what lets a tail be paired with either front.
    """
    cell = CELLS[cfgs["cell"]]
    if cell.front == FRONT_BLOCK:
        x, w_qkv, w_o = inputs["x"], inputs["w_qkv"], inputs["w_o"]
        print("  [front 1/1] qkv_proj + mha_out_proj (one sequence, q/k/v resident)")
        attn, vector = _sequence_a(cache, cfgs["block"], x, w_qkv, w_o)
        return dict(attn), [vector]

    cfg = cfgs["runlist"]
    num_heads = cfg["num_heads"]
    print("  [front 1/%d] q_proj + k_proj + v_proj (3 entries)" % (2 + num_heads))
    proj, vec_proj = run_projections(
        cache, cfg, inputs["x"], inputs["w_q"], inputs["w_k"], inputs["w_v"]
    )
    print(
        f"  [front 2..{1 + num_heads}/{2 + num_heads}] attention on device: "
        f"{num_heads} x (attn_scores + softmax + attn_output), 3 entries each"
    )
    attn_context, attn_vectors = run_attention_interior(
        cache, cfg, proj["q"], proj["k"], proj["v"]
    )
    print(f"  [front {2 + num_heads}/{2 + num_heads}] output_proj (1 entry)")
    attn_out, vec_o = run_o_proj(cache, cfg, round_bf16(attn_context), inputs["w_o"])
    boundaries = {
        "q": proj["q"],
        "k": proj["k"],
        "v": proj["v"],
        # bf16 straight from the device, widened for the comparison exactly as
        # `runlist` widens it. The widening is exact and adds nothing.
        "attn_context": attn_context.astype(np.float32),
        "attn_out": attn_out,
    }
    return boundaries, [vec_proj] + attn_vectors + [vec_o]


def _run_tail(cache, cfgs, inputs, attn_out):
    """The cell's tail: ln1, the FFN, ln2.

    ``attn_out`` is the front's output and the ONLY tensor that crosses the
    front/tail seam, alongside the layer input as the first residual. Returns
    ``(boundaries, vectors)`` with ``hidden``, ``ffn_up``, ``ffn_gelu``,
    ``ffn_out`` and ``output``.
    """
    cell = CELLS[cfgs["cell"]]
    x = inputs["x"]
    w_up, w_down = inputs["w_up"], inputs["w_down"]
    ln1_weight, ln2_weight = inputs["ln1_weight"], inputs["ln2_weight"]

    if cell.tail == TAIL_BANDED:
        cfg = cfgs["block"]
        blocks = cfg["norm_blocks"]
        # The banded `addnorm` takes the [emb] weight itself; the decomposed
        # multiply takes a materialized band broadcast. See the module footguns.
        print(f"  [tail 1/3] addnorm ln1 x{blocks} (pre-add)")
        hidden, vec_1 = _sequence_norm(cache, cfg, "ln1", attn_out, x, ln1_weight)
        print("  [tail 2/3] ffn")
        ffn, vec_2 = _sequence_ffn(cache, cfg, hidden, w_up, w_down)
        print(f"  [tail 3/3] addnorm ln2 x{blocks} (pre-add)")
        output, vec_3 = _sequence_norm(
            cache, cfg, "ln2", ffn["ffn_out"], hidden, ln2_weight
        )
    else:
        cfg = cfgs["runlist"]
        blocks = cfg["norm_blocks"]
        rows = cfg["norm_rows"]
        gamma1 = round_bf16(broadcast_row_weight(ln1_weight, rows))
        gamma2 = round_bf16(broadcast_row_weight(ln2_weight, rows))
        print(f"  [tail 1/3] {blocks} x (add + layer_norm + mul) ln1")
        hidden, vec_1 = run_norm_chain(cache, cfg, "ln1", attn_out, x, gamma1)
        print("  [tail 2/3] up_proj + gelu + down_proj (3 entries)")
        ffn, vec_2 = run_ffn(cache, cfg, hidden, w_up, w_down)
        print(f"  [tail 3/3] {blocks} x (add + layer_norm + mul) ln2")
        output, vec_3 = run_norm_chain(
            cache, cfg, "ln2", ffn["ffn_out"], hidden, gamma2
        )

    boundaries = dict(ffn)
    boundaries["hidden"] = hidden
    boundaries["output"] = output
    return boundaries, [vec_1, vec_2, vec_3]


def run_cell(cache, cfgs, inputs):
    """Dispatch the whole layer for one cell and return every device boundary.

    Args:
        cache: a ``KernelCache`` holding both halves' artifacts.
        cfgs: ``cell_config(...)``.
        inputs: the tensors named by ``cell_input_names(cell)``, in that order.
            Taken positionally because that is the list ``opcheck.py`` perturbs
            under ``--fault-inject``, and named here.

    Returns:
        ``(boundaries, vector_rows)``, with a COPY of each of
        ``ENCODER_BOUNDARIES``; nothing in it aliases pool memory.
    """
    names = cell_input_names(cfgs["cell"])
    if len(inputs) != len(names):
        raise ValueError(
            f"cell {cfgs['cell']} takes {len(names)} inputs {names}, got {len(inputs)}"
        )
    named = dict(zip(names, inputs))
    front, front_vectors = _run_front(cache, cfgs, named)
    tail, tail_vectors = _run_tail(cache, cfgs, named, front["attn_out"])
    boundaries = dict(front)
    boundaries.update(tail)
    missing = [n for n in ENCODER_BOUNDARIES if n not in boundaries]
    if missing:
        raise AssertionError(f"run_cell produced no value for {missing}")
    return boundaries, [v.as_row() for v in front_vectors + tail_vectors]


def prepare_cell(cell_name, shape, seed=42, cache_dir=None, label=None, extra=None):
    """One blend cell against the golden model, for ``opcheck.py``.

    The glue is the same contract ``prepare_layer_dispatch`` and
    ``prepare_runlist`` implement -- same golden model and draw order, same
    measured ``ln1_weight`` target, same per-boundary comparisons at
    ``BLOCK_STAGE_ATOL``, same unconditional dispatch-vector recording. A mode
    with its own device path imports the pieces rather than the whole
    (``opcheck_layer.py``'s contract), and this module is one device path
    serving both cells.
    """
    cell = CELLS[cell_name]
    label = label or f"coarse_{cell_name.lower()}"
    if cache_dir is None:
        raise ValueError(
            f"{label}: pass the cell's OWN cache_dir. Sharing a directory lets "
            "two modes trade ELFs whose fingerprints agree, which attributes "
            "valid numbers to the wrong execution boundary."
        )
    seq_len, emb_dim = shape["seq_len"], shape["emb_dim"]
    ffn_dim, num_heads = shape["ffn_dim"], shape["num_heads"]
    head_dim = shape["head_dim"]

    cfgs = cell_config(cell_name, seq_len, emb_dim, ffn_dim, num_heads, head_dim)
    describe_cell(cfgs)

    golden = generate_golden_reference(
        seq_len, emb_dim, ffn_dim, num_heads, seed=seed, workload_variant="encoder_bert"
    )
    weights = golden["weights"]
    reference = golden["boundaries"]
    names = cell_input_names(cell_name)
    # The order is `names`; `inject` below indexes into it. Keyed by
    # GOLDEN_INPUT_NAMES, which the structure check pins as a superset of every
    # cell's list.
    available = {
        "x": golden["input"],
        "w_qkv": fuse_qkv_weight(weights) if cell.front == FRONT_BLOCK else None,
        "w_q": weights["q_weight"],
        "w_k": weights["k_weight"],
        "w_v": weights["v_weight"],
        "w_o": weights["attn_output_weight"],
        "ln1_weight": weights["ln1_weight"],
        "w_up": weights["ffn_up_weight"],
        "w_down": weights["ffn_down_weight"],
        "ln2_weight": weights["ln2_weight"],
    }
    assert set(available) == set(GOLDEN_INPUT_NAMES), (
        "the drawn inputs and GOLDEN_INPUT_NAMES have drifted; the structure "
        "check pins a cell's list against the latter, so a name only in one of "
        "them is checked by nothing"
    )
    inputs = [available[name] for name in names]

    from shared.infra.cache import KernelCache, Profiler

    cache = KernelCache(
        cache_dir=cache_dir, verbose=False, profiler=Profiler(enabled=True)
    )
    compile_cell_artifacts(cache, cfgs, run_only=True)

    def dispatch(device_inputs, stage_stats, forward_done=None):
        cache.profiler.cpu_times.clear()
        reconfig_baseline = cache.reconfiguration_counts()
        boundaries, vector_rows = run_cell(cache, cfgs, device_inputs)
        stages = []
        # The forward is DONE here: every boundary is a host array. The study's
        # clock stops at this instant (operator rule, 2026-08-22); the per-boundary
        # comparison below is verification and runs outside it.
        if forward_done is not None:
            forward_done()
        for name in ENCODER_BOUNDARIES:
            atol = BLOCK_STAGE_ATOL[name]
            stats = stage_stats(boundaries[name], reference[name], atol=atol)
            stages.append(dict(stats, name=name, atol=atol))
            print(
                f"  [stage] {name:13s} {stats['n_elements']:>9d} elements  "
                f"mismatch {stats['n_mismatch']:>7d}  "
                f"mean_rel_L1 {stats['mean_rel_L1']:.3e}  "
                f"atol_required {stats['atol_required']:.3e} (atol {atol:.1e})"
            )
        clean = sum(1 for s in stages if s["n_mismatch"] == 0)
        print(f"[{label}] stages: {clean}/{len(stages)} clean")
        # On the fault path too -- the FAULT half of the lit recipe pins the
        # printed totals to the same literals as the clean half.
        print_dispatch_totals(label, vector_rows)
        return [boundaries["output"]], {
            "stages": stages,
            "stages_passed": clean == len(stages),
            "dispatch_vectors": vector_rows,
            # The latency decomposition convention 10 asks for. Every operator
            # is on the device in both cells, so host_cpu_ms is empty BY
            # CONSTRUCTION and the comparison against a host-mediated mode is
            # what recording it is for.
            "device_ms": sum(
                float(r.get("device_submission_ms", 0.0)) for r in vector_rows
            ),
            "sync_ms": sum(float(r.get("host_sync_ms", 0.0)) for r in vector_rows),
            "host_cpu_ms": {
                k: sum(v) * 1000.0 for k, v in cache.profiler.cpu_times.items()
            },
            # What THIS dispatch loaded and attached (schema v2's
            # reconfiguration columns). A cell with the runlist front pays that
            # front's per-head attention reloads in steady state; a cell with
            # the block front keeps every context standing and honestly
            # reports 0.
            **reconfiguration_delta(cache, reconfig_baseline),
        }

    block, runlist = cfgs["block"], cfgs["runlist"]
    submissions, entries = cell_dispatch_prediction(
        cell_name, block["num_heads"], block["norm_blocks"]
    )
    record_extra = {
        "variant": "encoder_bert",
        "causal": False,
        "golden_seed": seed,
        "execution_mode": EXECUTION_MODE_CSV["coarse"],
        "attention_path": "device_all",
        # The cell, and the two axes it is a point on. This is the provenance
        # the mode lacked: the artifact says which blend was dispatched.
        "blend_cell": cell_name,
        "blend_front": cell.front,
        "blend_tail": cell.tail,
        "norm_rows": block["norm_rows"],
        "norm_blocks": block["norm_blocks"],
        "predicted_submissions": submissions,
        "predicted_runlist_entries": entries,
    }
    if cell.front == FRONT_BLOCK:
        record_extra.update(
            {
                "gemm_spec_source": block["qkv_source"],
                "gemm_spec_qkv": _spec_digest(block["qkv_spec"]),
                "gemm_spec_o_proj": _spec_digest(block["o_proj_spec"]),
            }
        )
    else:
        record_extra.update(
            {
                # MIXED: the projections resolve in the registry, the two
                # attention shapes are injected measured tiles that resolve in
                # none. Same wording `runlist` records, for the same reason.
                "gemm_spec_source": "registry+injected",
                "gemm_spec_proj": _spec_digest(runlist["specs"]["proj"][0]),
                "gemm_spec_attn_scores": _spec_digest(
                    runlist["specs"]["attn_scores"][0]
                ),
                "gemm_spec_attn_output": _spec_digest(
                    runlist["specs"]["attn_output"][0]
                ),
            }
        )
    if cell.tail == TAIL_BANDED:
        record_extra.update(
            {
                "gemm_spec_ffn_up": _spec_digest(block["ffn_up_spec"]),
                "gemm_spec_ffn_down": _spec_digest(block["ffn_down_spec"]),
            }
        )
    else:
        record_extra.update(
            {
                "gemm_spec_ffn_up": _spec_digest(runlist["specs"]["up"][0]),
                "gemm_spec_ffn_down": _spec_digest(runlist["specs"]["down"][0]),
            }
        )
    record_extra.update(extra or {})
    return {
        "inputs": inputs,
        # ln1_weight -- the measured target, from the front's own input list.
        # See opcheck_layer.py for the measurement that rules out every
        # attention-side candidate.
        "inject": (names.index("ln1_weight"), (0,)),
        "expected": [reference["output"]],
        "dispatch": dispatch,
        "record_extra": record_extra,
    }
