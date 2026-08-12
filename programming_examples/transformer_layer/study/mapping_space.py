# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How big the resident-tail mapping space is, and which points cannot be routed.

    python3 study/mapping_space.py            # the census
    python3 study/mapping_space.py --axes     # the axis table with its sources

CONTRACT
    Two things, and the second is the point.

    1. ``legality(mapping)`` -- a STATIC verdict on one point of the mapping
       space. No compile, no device, no dump: it reads the declaration, not a
       routed design. It returns ``Verdict(legal, refusals, prices)``, where a
       refusal is a design that CANNOT BE ROUTED under any placement and a
       price is a design that routes and is degraded.
    2. ``count_space()`` -- the size of the space before and after that
       predicate, over the axes ``AXES`` declares, each carrying the artifact
       in this tree that bounds it.

    The deliverable is (2). Doc 38 measured that iron's legality prune left a
    space small enough that its placements are hand-authored tables with only
    SEVEN legal ``(parallel_heads, parallel_ffn)`` tails. If ours is also small
    after legality, enumeration plus a measured cost table beats any search and
    the mapping work reorders. So the number is the result and this module is
    the machinery that produces it.

THE MECHANISM, AND THE THREE CHANGES ON THE WAY IN
    TileFlow's ``ResourceConstraintParser::visitScope`` (doc 46 section 4.3) is
    a ~30-line visitor: a resource demand accumulated up a scope tree with a
    combinator keyed on the scope type -- ``max`` where stages share the
    hardware in time (``Seq``/``Shar``) and SIGMA where they occupy it together
    in space (``Para``/``Pipe``). That is a CARDINALITY resource shared across a
    whole fused group, which doc 44 records as the thing MAESTRO's byte-rate
    model could not state. ``compose`` below is that combinator.

    Doc 46 section 1 names three deliberate deviations, all forced by our
    machine, and all three are here:

    - **We count our quantity, not theirs.** TileFlow's ``core_usage_`` is
      always the product of spatial loop extents. Ours is a ``Demand``: cores,
      shim MM2S streams split into the part pinned to a herd's columns and the
      part the allocator may place anywhere, memtile ports, core DMA ports and
      L1/L2 bytes.
    - **The budget is per COLUMN, not one uniform pair per level.** Doc 23's
      rule is two shim MM2S per column ACROSS THE WHOLE SEGMENT, and doc 31b
      section 4 shows the two ways an operand reaches the array cost
      differently: a herd-direct fetch is one MM2S in EVERY column that herd
      occupies; an L2-staged refill is one MM2S TOTAL, wherever the allocator
      puts it.
    - **Slope, not exit.** TileFlow's ``TILEFLOW_ASSERT`` calls ``exit(1)`` on a
      violated SPATIAL constraint. Ours must not, because AIR does not refuse
      when the column budget is exceeded -- it PACKET-MULTIPLEXES, silently
      (doc 23, and queue item 10 measured it: 0 inbound ``aie.flow`` and 12
      ``aie.packet_flow`` on a design 50% over). A predicate that filtered those
      points out would hide the failure mode this study is trying to see.

WHERE THE LINE BETWEEN REFUSE AND PRICE FALLS, AND WHY IT FALLS THERE
    The shim budget is per column, and which column a stream lands in is the
    allocator's choice -- so most of the budget is NOT statically decidable.
    What IS statically decidable is its PLACEMENT-INVARIANT part, and that is
    exactly where this module refuses:

    - a herd with more than ``SHIM_MM2S_PER_COLUMN`` herd-direct L3 operands is
      over budget in EVERY column it occupies, under every placement -- REFUSE.
      (That is queue item 10's negative control, built and measured.)
    - a segment whose total demand exceeds ``NPU2_COLUMNS *
      SHIM_MM2S_PER_COLUMN`` slots cannot fit however it is spread -- REFUSE.
      (That is J1's failure: "8 columns x 2 = 16, already full before the third
      stream", doc 23.)
    - everything in between routes under SOME placement, and whether it routes
      under the one the tools pick is a question about ``air-place-herds`` and
      the shim allocator. R1 is the live example: an assignment exists with
      worst column 1, and the shipped one produced worst column 2 (doc 31b
      section 3.6, MEASURED). Both legal. So ``_predicted_columns`` models what
      the tools are known to do -- doc 23's "the budget is per column ACROSS
      STACKED HERDS ... their demands add" -- and over-subscription there is
      PRICED, ``min(1, budget/demand)`` per doc 46 section 1, never rejected.

    Everything else in the predicate is a cliff because it was MEASURED as one:
    nine herds of four refuse with ``'aie.tile' op row index (6) must be less
    than the number of rows in the device (6)`` (doc 31b section 3.5); a
    rows_per_call that overflows L1 refuses in aiecc's allocator; a memtile
    asked for a seventh feed sub-channel has no port to give.

WHAT THIS MODEL DOES NOT KNOW
    - **Bytes and cycles.** This is a cardinality model. It answers "can this be
      routed" and "how many ports does it want", never "how fast". The cost half
      is queue item 25's balance instrument.
    - **Which column.** See above. It answers "does a legal placement exist" and
      "is the placement the tools are known to produce within budget", not "what
      will ``air-place-herds`` do".
    - **The R1/R2 partitioning choice.** R1 partitions the FFN by output column,
      R2 by row; doc 31b section 6.1 records that this "is not a parameter, it
      is a different module". A different module is a different space, not an
      axis of this one.
    - **Anything about attention.** The tail is nt1 -> up -> gelu -> down ->
      nt2. iron's ``parallel_heads`` has no analogue here because the resident
      tail has no heads in it.

FOOTGUNS
    - **Every machine constant here is RE-STATED from a builder, and every one
      is pinned back to that builder's source by ast in test_mapping_space.py.**
      Restating rather than importing is forced: ``builders/`` imports ``air``
      and this suite must run without a toolchain (the reason
      ``test_run_ladder._spec`` is a hand-written row). A re-stated constant
      that is not pinned agrees with the builder until the builder moves and
      then agrees with history.
    - **The space size is a claim about the axes, not about the machine.** Widen
      an axis and the "before" number grows with no measurement behind it. Every
      range in ``AXES`` carries the artifact that bounds it, and the one axis no
      artifact bounds says so in its ``source`` and in ``--axes`` output.
    - **Never clamp a demand to its budget.** An earlier draft of this file
      wrote ``core_s2mm=min(CORE_S2MM, ...)``, which makes the core-port clause
      unable to fail -- queue item 10's failure mode exactly, in a second place.
      Demands are reported raw and compared against the budget by the predicate.
    - **A count is not a search.** ``count_space`` factorises: for one
      (structure, seam vector) the predicate treats ``(rows_per_call, tile_k)``
      and the routing vector independently, so the legal count is the product of
      the two sub-counts. ``test_the_factorisation_matches_brute_force``
      brute-forces a sub-space and asserts equality, so a future clause that
      couples them fails there rather than biasing this number quietly.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from dataclasses import dataclass, field
from functools import lru_cache

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import cases  # noqa: E402

# ---------------------------------------------------------------------------
# The machine. Re-stated from the sources named; pinned by ast in the tests.
# ---------------------------------------------------------------------------

#: mlir_aie AIETargetModel.h -- NPU2TargetModel::columns() = 8 and
#: BaseNPU2TargetModel::rows() = 6, of which one shim, one memtile, four core.
#: Re-stated in ffn_resident_structure.py as NPU2_COLUMNS; pinned to it.
NPU2_COLUMNS = 8
NPU2_CORE_ROWS = 4
NPU2_CORES = NPU2_COLUMNS * NPU2_CORE_ROWS

#: A shim NOC tile drives two MM2S and two S2MM DMA channels. Doc 23's rule is
#: per COLUMN across the whole segment. Re-stated from
#: ffn_resident_structure.SHIM_MM2S_PER_COLUMN.
SHIM_MM2S_PER_COLUMN = 2
SHIM_S2MM_PER_COLUMN = 2
NPU2_SHIM_MM2S_PORTS = NPU2_COLUMNS * SHIM_MM2S_PER_COLUMN

#: A memtile has six MM2S and six S2MM. builders/ffn_accum.MAX_FEED_CHANNELS is
#: the same number seen from the feed side ("one feed sub-channel per core, each
#: a memtile MM2S port; a memtile has 6").
MEMTILE_MM2S = 6
MEMTILE_S2MM = 6

#: A compute tile has two in and two out. This is the arithmetic
#: builders/ffn_resident.py's docstring runs to derive its memtile fan-out: "a
#: down core has two S2MM channels, both spoken for (A|B feed + hoisted C
#: fetch), and an air channel has one physical source".
CORE_S2MM = 2
CORE_MM2S = 2

#: AIE2P core-local memory and the stack aircc reserves inside it.
#: builders/norm_tail.L1_BYTES / L1_STACK_BYTES.
L1_BYTES = 64 * 1024
L1_STACK_BYTES = 1024

#: A memtile's memory. builders/ffn_accum.L2_BYTES.
L2_BYTES = 512 * 1024

#: builders/ffn_accum: the aie::mmul microtile edge, and DIM_M baked into the
#: kernel object.
MICRO = 8
TILE_M = 64

#: The two MEASURED walls builders/ffn_accum refuses at, with the measurement in
#: the message: aie-place-tiles refuses the accumulator pair's shim slots past 4
#: columns, and tile_k 64 puts the worst-case L1 over the 64 KiB tile.
MAX_PLACEABLE_HERD_X = 4
MAX_L1_TILE_K = 32
MAX_FEED_CHANNELS = 6

#: builders/norm_tail: every stage steps its vectors by 16 with no scalar tail.
NORM_TAIL_VEC_LEN = 16

#: The shape the whole study is scoped at. Taken from the case matrix rather
#: than restated -- study/cases.py needs no toolchain, so it imports here.
_BASELINE = cases.FAMILY_SPECS["baseline_768"]
EMB_DIM = _BASELINE.hidden_size
FFN_DIM = _BASELINE.intermediate_size

#: Doc 31 makes every increment band-serial: one TILE_M-row band per dispatch.
#: builders/ffn_resident refuses seq_len != TILE_M.
BAND_ROWS = TILE_M


# ---------------------------------------------------------------------------
# The scope types. Doc 46 section 6's 2x2, in our vocabulary.
# ---------------------------------------------------------------------------

#: Compute shared in TIME: the stages take the array in turns, so cores are
#: `max` over children. TileFlow's Sequential and Sharing.
SCOPE_SEQ = "Seq"  # packaged -- separate launches, buffers freed, DRAM between
SCOPE_SHAR = "Shar"  # resident-in-turns -- same cores, buffers stay live
#: Compute partitioned in SPACE: the stages occupy the array together, so every
#: resource is SIGMA over children. TileFlow's Parallel and Pipeline.
SCOPE_PARA = "Para"  # independent copies -- doc 46: "unnamed, and we have it"
SCOPE_PIPE = "Pipe"  # interleaved -- dependent stages on distinct cores

SCOPE_TIME = (SCOPE_SEQ, SCOPE_SHAR)
SCOPE_SPACE = (SCOPE_PARA, SCOPE_PIPE)
SCOPES = (SCOPE_SEQ, SCOPE_SHAR, SCOPE_PARA, SCOPE_PIPE)

#: Doc 46 section 6's mapping onto doc 03's words. Data, because the doc's table
#: is the definition and a second prose copy would drift from it.
SCOPE_WORDS = {
    SCOPE_SEQ: "packaged",
    SCOPE_SHAR: "resident-in-turns",
    SCOPE_PARA: "independent",
    SCOPE_PIPE: "interleaved",
}

#: TileFlow section 4.1: "Para ... is only applicable to tiles without data
#: dependency". Every seam of the tail chain carries one, so Para is legal
#: across replicated band lanes and nowhere else in this design.
SCOPE_NEEDS_INDEPENDENCE = (SCOPE_PARA,)

#: Only Seq puts a segment boundary in: each air.launch lowers to its own
#: aie.device, so its stages are time-multiplexed at segment granularity and
#: the intermediate crosses DRAM (builders/pipeline_spec, "WHY FUSING INTO ONE
#: SEGMENT IS THE OPERATION"). Shar keeps one segment and one set of cores.
SCOPE_SPLITS_SEGMENT = (SCOPE_SEQ,)


# ---------------------------------------------------------------------------
# The demand vector, and TileFlow's combinator over it.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Demand:
    """What one node of the scope tree asks of one segment's hardware.

    ``herds`` is the load-bearing field and the reason this is not a plain
    vector of scalars: a herd is ``(width, herd_direct_streams)``, and doc 31b
    section 4's cost model is that a herd-direct operand costs one shim MM2S in
    EVERY column the herd occupies. Keeping the multiset lets the budget be
    evaluated per column later; collapsing it to a scalar here is precisely the
    mistake doc 44 records Timeloop making ("one uniform (x,y) per level").

    No field is ever clamped to its budget. See FOOTGUNS.
    """

    #: ((width, herd_direct_L3_streams), ...) -- one entry per herd.
    herds: tuple = ()
    #: L2-staged L3 reads: one shim MM2S each, allocator-placed, no fixed column.
    shim_global: int = 0
    #: L3 writes. Same per-column budget, opposite direction.
    shim_s2mm: int = 0
    #: ((mm2s, s2mm), ...) -- one entry per memtile this node mediates through.
    memtiles: tuple = ()
    #: The worst single core's L1 use, in bytes, stack included.
    l1_bytes: int = 0
    #: The worst single memtile's L2 use, in bytes.
    l2_bytes: int = 0
    #: The worst single core's DMA port use.
    core_s2mm: int = 0
    core_mm2s: int = 0
    #: core -> core aie.flow edges, doc 03's space-multiplexed discriminator.
    core_core_edges: int = 0

    @property
    def cores(self) -> int:
        return sum(w for w, _ in self.herds)

    @property
    def shim_mm2s_slots(self) -> int:
        """Total shim MM2S channel-slots this node consumes, of NPU2's 16.

        Placement-INVARIANT: a herd-direct operand costs one slot in each of the
        herd's columns however the columns are chosen, and a staged refill costs
        exactly one wherever it lands. This is the quantity doc 31b section 3.6
        MEASURED as "7 of 16 ports" on R1.
        """
        return sum(w * d for w, d in self.herds) + self.shim_global


def _merge_onto(host: Demand, guest: Demand) -> Demand:
    """``Shar``: two groups take the SAME cores in turns, buffers staying live.

    Cores are ``max`` (the guest runs where the host does), footprints ADD (that
    is what "buffers stay live" means, and it is the whole difference between
    ``Shar`` and ``Seq`` in doc 46's table), and the guest's L3 operands become
    operands of the host's first herd -- so they land in the host's columns and
    the per-column budget sees them stacked.
    """
    if not host.herds:
        return guest
    if not guest.herds:
        return host
    extra_direct = sum(d for _, d in guest.herds)
    w0, d0 = host.herds[0]
    return Demand(
        herds=((max(w0, guest.herds[0][0]), d0 + extra_direct),) + host.herds[1:],
        shim_global=host.shim_global + guest.shim_global,
        shim_s2mm=host.shim_s2mm + guest.shim_s2mm,
        memtiles=host.memtiles + guest.memtiles,
        l1_bytes=host.l1_bytes + guest.l1_bytes,
        l2_bytes=host.l2_bytes + guest.l2_bytes,
        core_s2mm=max(host.core_s2mm, guest.core_s2mm),
        core_mm2s=max(host.core_mm2s, guest.core_mm2s),
        core_core_edges=host.core_core_edges + guest.core_core_edges,
    )


def compose(scope: str, children: tuple) -> Demand:
    """TileFlow's ``visitScope``, over our demand vector instead of their pair.

    ``core_usage_ = (Sequential || Sharing) ? Op::max(exprs) : Op::sum(exprs)``
    -- ``src/mapper/checker.cpp:506-519``, quoted in doc 46 section 4.3.

    The 2x2 doc 46 section 6 draws is that COMPUTE and FOOTPRINT are separate
    axes of it, so this is not one combinator but two:

      - compute (cores, ports, edges): ``max`` for ``Seq``/``Shar``, SIGMA for
        ``Para``/``Pipe``.
      - footprint (L1, L2): ``max`` for ``Seq`` alone -- the one scope whose
        buffers are freed between turns -- and SIGMA for ``Shar``. For
        ``Para``/``Pipe`` the children are on DIFFERENT cores, so L1 is per-core
        ``max`` there and only L2, which they share, adds.

    Collapsing those into one rule is how ``Seq`` and ``Shar`` would stop being
    distinguishable, and telling them apart is the whole content of doc 46's
    table row for ``Shar``.
    """
    if scope not in SCOPES:
        raise ValueError(f"{scope!r} is not one of {SCOPES}")
    if not children:
        return Demand()
    if len(children) == 1:
        return children[0]

    if scope == SCOPE_SHAR:
        out = children[0]
        for c in children[1:]:
            out = _merge_onto(out, c)
        return out

    if scope == SCOPE_SEQ:
        worst = max(children, key=lambda d: (d.cores, d.shim_mm2s_slots))
        return Demand(
            herds=worst.herds,
            shim_global=max(c.shim_global for c in children),
            shim_s2mm=max(c.shim_s2mm for c in children),
            memtiles=worst.memtiles,
            l1_bytes=max(c.l1_bytes for c in children),
            l2_bytes=max(c.l2_bytes for c in children),
            core_s2mm=max(c.core_s2mm for c in children),
            core_mm2s=max(c.core_mm2s for c in children),
            core_core_edges=max(c.core_core_edges for c in children),
        )

    # Para / Pipe: partitioned in space.
    return Demand(
        herds=tuple(itertools.chain.from_iterable(c.herds for c in children)),
        shim_global=sum(c.shim_global for c in children),
        shim_s2mm=sum(c.shim_s2mm for c in children),
        memtiles=tuple(itertools.chain.from_iterable(c.memtiles for c in children)),
        l1_bytes=max(c.l1_bytes for c in children),
        l2_bytes=sum(c.l2_bytes for c in children),
        core_s2mm=max(c.core_s2mm for c in children),
        core_mm2s=max(c.core_mm2s for c in children),
        core_core_edges=sum(c.core_core_edges for c in children),
    )


# ---------------------------------------------------------------------------
# The axes. Every range carries the artifact in this tree that bounds it.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Axis:
    name: str
    values: tuple
    source: str
    #: True when nothing in this tree bounds the range and the bound below is an
    #: inference. Reported separately by --axes and by the census, because a
    #: space size is only as honest as its loosest axis.
    unbounded_by_artifact: bool = False


#: The five L3 operands of the R2 tail, doc 31b section 4's table verbatim, each
#: paired with the stage group that fetches it. ``output`` is S2MM and in a
#: different budget, so it is not a routing axis.
L3_OPERANDS = (
    ("packed1", "nt1"),
    ("gamma1", "nt1"),
    ("w_up", "up"),
    ("w_down", "down"),
    ("gamma2", "nt2"),
)

ROUTE_HERD_DIRECT = "herd_direct"
ROUTE_L2_STAGED = "l2_staged"
ROUTES = (ROUTE_HERD_DIRECT, ROUTE_L2_STAGED)

#: The tail's operator boundaries. Doc 31b section 5: "the stages R2 is about
#: are the tail's operator boundaries -- nt1 -> up -> gelu -> down -> nt2 -- not
#: the internals of a normalization".
GROUPS = ("nt1", "up", "gelu", "down", "nt2")
SEAMS_SPLIT = (("nt1", "up"), ("up", "gelu"), ("gelu", "down"), ("down", "nt2"))
SEAMS_FOLDED = (("nt1", "up"), ("up", "down"), ("down", "nt2"))

AXES = (
    Axis(
        "nt1_form",
        ("fused", "j7a"),
        "doc 31b section 5's herd inventory: a norm tail is J7a's three-herd "
        "pipeline (builders/norm_tail.NORM_TAIL_STAGE_SPEC, three StageSpecs) "
        "or the one-herd form on the addnorm pre-add kernel",
    ),
    Axis(
        "nt2_form",
        ("fused", "j7a"),
        "as nt1_form: doc 31b section 5's herd inventory, which counts the two "
        "tails separately because its 7-herd row keeps J7a's pipeline on the "
        "first tail and folds only the second",
    ),
    Axis(
        "gemm_fold",
        ("split", "folded"),
        "doc 31b section 5 row 2: up+gelu fold into one herd, and the fold is "
        "free at the object level because ffn_accum_mm.o already exports "
        "ffn_gelu_bf16 (section 7.2)",
    ),
    Axis(
        "gemm_herd_x",
        tuple(range(1, MAX_PLACEABLE_HERD_X + 1)),
        "builders/ffn_accum.MAX_PLACEABLE_HERD_X = 4, MEASURED: aie-place-tiles "
        "refuses the accumulator pair's shim slots at 6 columns",
    ),
    Axis(
        "norm_herd_x",
        tuple(range(1, NPU2_COLUMNS + 1)),
        "NPU2TargetModel::columns() = 8; builders/norm_tail defaults herd_x=8 "
        "and its docstring records that air-place-herds does place it",
    ),
    Axis(
        "tile_k",
        tuple(range(MICRO, MAX_L1_TILE_K + 1, MICRO)),
        "builders/ffn_accum: multiples of MICRO = 8 up to MAX_L1_TILE_K = 32, "
        "MEASURED: tile_k 64 puts the worst-case L1 over the 64 KiB tile",
    ),
    Axis(
        "rows_per_call",
        tuple(r for r in range(1, TILE_M + 1) if TILE_M % r == 0),
        "divisors of builders/ffn_accum.TILE_M = 64, the band a dispatch "
        "carries. The L1 ceiling on it is a capacity REJECTION, not a range "
        "bound -- builders/norm_tail._stage_l1_bytes, MEASURED at 8",
    ),
    Axis(
        "parallel_bands",
        tuple(range(1, NPU2_COLUMNS + 1)),
        "NOT BOUNDED BY ANY ARTIFACT. A band lane needs at least one column so "
        "8 is an upper bound, but no measurement in this tree bounds it, and "
        "builders/ffn_resident refuses seq_len != TILE_M so the shipped builder "
        "is at 1. Doc 31: a multi-band module 'would need herd_y > 1 through "
        "the same memtile fan-out, which is unmeasured'",
        unbounded_by_artifact=True,
    ),
) + tuple(
    Axis(
        f"route_{operand}",
        ROUTES,
        "doc 31b section 4: herd-direct costs one shim MM2S in every column the "
        "herd occupies; L2-staged costs one MM2S total, allocator-placed",
    )
    for operand, _ in L3_OPERANDS
)

#: The seam axis is declared separately because its LENGTH depends on
#: gemm_fold: folding up and gelu deletes a boundary, so a folded design has
#: three seams and a split one four. Multiplying by 4**4 in both cases would
#: count a seam that does not exist.
SEAM_AXIS = Axis(
    "seam_<a>_<b>",
    SCOPES,
    "doc 46 section 6's 2x2 -- {compute shared in time vs partitioned in space} "
    "x {footprint exclusive vs additive} -- mapped onto doc 03's vocabulary. "
    "One choice per operator boundary of the tail",
)


def axis(name: str) -> Axis:
    for a in AXES:
        if a.name == name:
            return a
    raise KeyError(name)


# ---------------------------------------------------------------------------
# One point of the space.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Mapping:
    """One point: a structure, a tiling, a routing and a scope per seam."""

    nt1_form: str = "j7a"
    nt2_form: str = "j7a"
    gemm_fold: str = "split"
    gemm_herd_x: int = MAX_PLACEABLE_HERD_X
    norm_herd_x: int = NPU2_COLUMNS
    tile_k: int = MAX_L1_TILE_K
    rows_per_call: int = 4
    parallel_bands: int = 1
    #: ((operand, ROUTE_*), ...); an operand not named is herd-direct.
    routes: tuple = ()
    #: (((group_a, group_b), scope), ...); a seam not named is Pipe.
    seams: tuple = ()
    #: bf16 everywhere in the tail; a field so a future dtype is an axis rather
    #: than a literal.
    itemsize: int = 2
    cols: int = EMB_DIM

    @property
    def seam_pairs(self):
        return SEAMS_FOLDED if self.gemm_fold == "folded" else SEAMS_SPLIT

    def route(self, operand: str) -> str:
        for name, value in self.routes:
            if name == operand:
                return value
        return ROUTE_HERD_DIRECT

    def scope(self, seam) -> str:
        seam = tuple(seam)
        for pair, value in self.seams:
            if tuple(pair) == seam:
                return value
        return SCOPE_PIPE


def norm_tail_l1_bytes(rows_per_call: int, cols: int, itemsize: int) -> int:
    """builders/norm_tail._stage_l1_bytes, re-stated and pinned by ast.

    The factor of two is aircc's ping-pong of BOTH of stage_add's tiles, and it
    is a MEASUREMENT: four buffers of 24+24+12+12 KiB at rows_per_call=8,
    cols=768, refused by aiecc's allocator with "allocated buffers exceeded
    available memory". The fused one-herd form holds the same three planes (two
    packed, one out), so the same formula bounds it.
    """
    return 2 * 3 * rows_per_call * cols * itemsize + L1_STACK_BYTES


#: THE GEMM SIDE HAS NO L1 BYTE FORMULA HERE, AND THAT IS DELIBERATE.
#:
#: ``build_ffn_accum_module``'s docstring describes the composition ("ping-ponged
#: A and B tiles plus the resident C") but states the WALL as the constant
#: ``MAX_L1_TILE_K = 32`` -- "32 keeps the worst-case L1 comfortably under the
#: 64 KiB tile; 64 measures just over it". A formula written from that prose does
#: not reproduce "just over": ping-ponged A+B plus a resident C at tile_k 64 and
#: tile_n 192 comes to 91 KiB, not 66. So the prose does not determine the
#: formula, and a formula guessed until it fits would be an INVENTED WALL --
#: which in this module means an invented cut in the space size, in the
#: direction that flatters the headline number.
#:
#: What the tree does carry is the constant, and it is already the bound on the
#: ``tile_k`` axis. The GEMM groups therefore contribute no L1 byte demand and
#: the L1 capacity clause gates the norm side alone, where
#: ``_stage_l1_bytes`` is a real formula with a measured ping-pong factor
#: behind it. ``builders/ffn_accum`` accepts ``herd_x`` 1 and 2, so nothing in
#: this tree refuses them and neither does this predicate.
GEMM_L1_IS_BOUNDED_BY_TILE_K_NOT_BY_A_BYTE_FORMULA = True


# ---------------------------------------------------------------------------
# Leaf demands: what each stage group asks for.
# ---------------------------------------------------------------------------


def _norm_group(
    mapping: Mapping, form: str, operands, s2mm_streams: int, head: bool
) -> Demand:
    """One norm tail, as ``form`` builds it.

    J7a is three herds of ``[herd_x, 1]`` (builders/norm_tail: stage_add,
    stage_norm, stage_scale) with two core->core edges per column; the fused form
    is one herd. The herd-direct operands are distributed one per herd in J7a
    (``packed`` on the add stage, ``gamma`` on the scale stage -- doc 31b section
    4) and stacked on the single herd when fused, which is the whole difference
    the column budget sees between the two forms.
    """
    w = mapping.norm_herd_x
    direct = [op for op in operands if mapping.route(op) == ROUTE_HERD_DIRECT]
    staged = [op for op in operands if mapping.route(op) == ROUTE_L2_STAGED]
    l1 = norm_tail_l1_bytes(mapping.rows_per_call, mapping.cols, mapping.itemsize)

    if form == "j7a":
        n_herds = 3
        per_herd = [0] * n_herds
        for i, _ in enumerate(direct):
            per_herd[min(i, n_herds - 1)] += 1
        herds = tuple((w, d) for d in per_herd)
        edges = 2 * w
    elif form == "fused":
        herds = ((w, len(direct)),)
        edges = 0
    else:
        raise ValueError(f"norm tail form {form!r} is not one of ('fused', 'j7a')")

    # Each staged operand enters a memtile (one S2MM) and is fanned to the herd's
    # cores (w MM2S), the ffn_accum feed shape.
    memtiles = tuple((w, 1) for _ in staged)
    staged_bytes = len(staged) * mapping.rows_per_call * mapping.cols * mapping.itemsize
    return Demand(
        herds=herds,
        shim_global=len(staged),
        shim_s2mm=s2mm_streams,
        memtiles=memtiles,
        l1_bytes=l1,
        l2_bytes=staged_bytes,
        # A core's inbound ports: one for the previous stage's hand-off (the head
        # group has none -- its input IS packed1), one per herd-direct operand,
        # and ONE for the staged feed however many operands ride it. That last
        # is builders/ffn_accum's A|B discipline: "the memtile sends A's k-slice
        # then B's k-slice down the same channel because a core has only two
        # S2MM ports". NOT clamped to CORE_S2MM -- see FOOTGUNS.
        core_s2mm=(0 if head else 1) + len(direct) + (1 if staged else 0),
        core_mm2s=1,
        core_core_edges=edges,
    )


def _gemm_group(mapping: Mapping, operands, with_gelu: bool) -> Demand:
    """One GEMM herd -- up, down, or up+gelu folded.

    Its A|B feed comes off a memtile, one sub-channel per core: that is
    ``builders/ffn_accum.MAX_FEED_CHANNELS`` seen from the other end, and it is
    why a GEMM herd wider than six has nowhere to be fed from. A staged operand
    adds one memtile S2MM and rides the SAME sub-channels out (the memtile sends
    A's k-slice then B's k-slice down the same channel because a core has only
    two S2MM ports).

    ``with_gelu`` is doc 31b section 5's fold: the activation runs on the same
    core between the accumulate and the put, so it costs one more L1 chunk and
    no extra herd. The object already exports ``ffn_gelu_bf16`` (section 7.2), so
    it costs nothing at link time.
    """
    w = mapping.gemm_herd_x
    direct = [op for op in operands if mapping.route(op) == ROUTE_HERD_DIRECT]
    staged = [op for op in operands if mapping.route(op) == ROUTE_L2_STAGED]
    tile_n = mapping.cols // w if w and mapping.cols % w == 0 else 0
    return Demand(
        herds=((w, len(direct)),),
        shim_global=len(staged),
        memtiles=((w, len(staged)),),
        # No L1 byte demand: see
        # GEMM_L1_IS_BOUNDED_BY_TILE_K_NOT_BY_A_BYTE_FORMULA above.
        l1_bytes=0,
        l2_bytes=len(staged) * mapping.tile_k * (tile_n or 1) * mapping.itemsize,
        # The A|B feed channel carries the stage input and every staged operand;
        # each herd-direct operand takes a port of its own.
        core_s2mm=1 + len(direct),
        core_mm2s=1,
        core_core_edges=0,
    )


def group_demands(mapping: Mapping) -> dict:
    """Leaf demand per stage group, before any seam is composed."""
    by_group = {g: [] for g in GROUPS}
    for operand, owner in L3_OPERANDS:
        by_group[owner].append(operand)

    folded = mapping.gemm_fold == "folded"
    out = {
        "nt1": _norm_group(mapping, mapping.nt1_form, by_group["nt1"], 0, head=True),
        # nt2 writes the tail's output: one S2MM per column it occupies.
        "nt2": _norm_group(
            mapping, mapping.nt2_form, by_group["nt2"], mapping.norm_herd_x, head=False
        ),
        "up": _gemm_group(mapping, by_group["up"], with_gelu=folded),
        "down": _gemm_group(mapping, by_group["down"], with_gelu=False),
    }
    if folded:
        out["gelu"] = Demand()
    else:
        # The GeLU herd links encoder_ffn.o and holds one chunk in and one out;
        # no L3 operand of its own, and no L1 byte demand for the same reason
        # the GEMM herds have none.
        out["gelu"] = Demand(
            herds=((mapping.gemm_herd_x, 0),),
            l1_bytes=0,
            core_s2mm=1,
            core_mm2s=1,
        )
    return out


def segments(mapping: Mapping) -> list:
    """The scope tree, evaluated: one ``Demand`` per SEGMENT.

    A ``Seq`` seam puts a segment boundary in -- each ``air.launch`` lowers to
    its own ``aie.device``, so its stages are time-multiplexed at segment
    granularity (builders/pipeline_spec) -- and the intermediate therefore
    crosses DRAM: one shim S2MM on the producer, one herd-direct MM2S on the
    consumer. That is doc 31a's crossing table as a resource, and it is why
    packaging trades cores for shim ports rather than being free.

    ``Shar`` and ``Pipe`` both keep one segment; they differ in whether the two
    groups take the same cores in turns or different cores together, which is
    exactly the compute axis of doc 46's 2x2.
    """
    leaves = group_demands(mapping)
    folded = mapping.gemm_fold == "folded"
    order = [g for g in GROUPS if not (folded and g == "gelu")]

    blocks = [[leaves[order[0]]]]
    scopes_in_block = [[]]
    for pair in mapping.seam_pairs:
        scope = mapping.scope(pair)
        nxt = leaves[pair[1]]
        if scope in SCOPE_SPLITS_SEGMENT:
            blocks.append([nxt])
            scopes_in_block.append([])
        else:
            blocks[-1].append(nxt)
            scopes_in_block[-1].append(scope)

    out = []
    for members, scopes in zip(blocks, scopes_in_block):
        acc = members[0]
        for child, scope in zip(members[1:], scopes):
            acc = compose(scope, (acc, child))
        out.append(acc)

    # Each Seq boundary is one DRAM round trip between adjacent segments: the
    # producer's output leaves the array and the consumer's input comes back off
    # it. The CORE ports do not move -- the hand-off used one in each direction
    # on chip and uses one in each direction through L3 -- so only the SHIM
    # demand changes, which is the whole cost of packaging.
    for i in range(len(out) - 1):
        producer, consumer = out[i], out[i + 1]
        out[i] = Demand(
            herds=producer.herds,
            shim_global=producer.shim_global,
            shim_s2mm=producer.shim_s2mm
            + (producer.herds[0][0] if producer.herds else 0),
            memtiles=producer.memtiles,
            l1_bytes=producer.l1_bytes,
            l2_bytes=producer.l2_bytes,
            core_s2mm=producer.core_s2mm,
            core_mm2s=producer.core_mm2s,
            core_core_edges=producer.core_core_edges,
        )
        if consumer.herds:
            w0, d0 = consumer.herds[0]
            out[i + 1] = Demand(
                herds=((w0, d0 + 1),) + consumer.herds[1:],
                shim_global=consumer.shim_global,
                shim_s2mm=consumer.shim_s2mm,
                memtiles=consumer.memtiles,
                l1_bytes=consumer.l1_bytes,
                l2_bytes=consumer.l2_bytes,
                core_s2mm=consumer.core_s2mm,
                core_mm2s=consumer.core_mm2s,
                core_core_edges=consumer.core_core_edges,
            )
    return out


def whole_design(mapping: Mapping, per_segment=None) -> Demand:
    """The top of the tree: band lanes are ``Para`` over the composed chain.

    ``parallel_bands`` is the one axis that carries doc 46's fourth scope --
    independent copies with additive footprint, the cell doc 46 says we have been
    building "without a name". Every seam inside the tail is dependent, so
    ``Para`` is illegal at all of them and legal only here.
    """
    lanes = segments(mapping) if per_segment is None else per_segment
    # Segments are time-multiplexed, so the array must hold the largest one.
    one = compose(SCOPE_SEQ, tuple(lanes)) if lanes else Demand()
    if mapping.parallel_bands == 1:
        return one
    return compose(SCOPE_PARA, tuple([one] * mapping.parallel_bands))


# ---------------------------------------------------------------------------
# Legality.
# ---------------------------------------------------------------------------


@dataclass
class Verdict:
    legal: bool = True
    #: Structural: this design cannot be routed under ANY placement.
    refusals: list = field(default_factory=list)
    #: Legal-but-degraded: it routes, and it is slower for a named reason.
    prices: dict = field(default_factory=dict)
    #: Reported figures, so a caller can print demand beside budget (doc 46
    #: section 4.2's "used/limit", MAESTRO's refinement).
    report: dict = field(default_factory=dict)

    def refuse(self, reason: str) -> None:
        self.legal = False
        self.refusals.append(reason)


@lru_cache(maxsize=None)
def _row_packable(widths: tuple) -> bool:
    """Can these ``[w, 1]`` herds be packed into 4 rows of 8 columns?

    A herd of width w occupies w columns of ONE row, and two herds in one row may
    not overlap, so this is bin packing widths into ``NPU2_CORE_ROWS`` bins of
    ``NPU2_COLUMNS``. Exact, by DFS with the standard symmetry break (never open
    a second bin at a remaining capacity already tried at this depth).

    MEASURED against doc 31b section 3.5: eight herds of [4, 1] place and use
    every tile; nine refuse with "'aie.tile' op row index (6) must be less than
    the number of rows in the device (6)".
    """
    if any(w > NPU2_COLUMNS or w < 1 for w in widths):
        return False
    if sum(widths) > NPU2_CORES:
        return False
    order = sorted(widths, reverse=True)

    def go(i: int, caps: tuple) -> bool:
        if i == len(order):
            return True
        w = order[i]
        seen = set()
        for j, cap in enumerate(caps):
            if cap < w or cap in seen:
                continue
            seen.add(cap)
            nxt = list(caps)
            nxt[j] = cap - w
            if go(i + 1, tuple(sorted(nxt, reverse=True))):
                return True
        return False

    return go(0, tuple([NPU2_COLUMNS] * NPU2_CORE_ROWS))


def predicted_columns(demand: Demand, lanes: int = 1) -> list:
    """The per-column shim MM2S load the tools are known to produce.

    Doc 23: "the budget is per column ACROSS STACKED HERDS: three 8-wide herds
    put one tile of each into every column, so their demands add." So herds fill
    from column 0 and stack, which is what makes J7a's packed+gamma "exactly two
    per column" and what doc 31b section 4 warns R2 about. Staged refills are
    "allocated globally across shim columns" (builders/ffn_accum, confirmed by
    doc 31b section 3.6: R1's three landed on shim columns 1, 1 and 5 while its
    herd-direct fetches took 0, 2, 3, 4), so they go to the least-loaded column.

    This is a PREDICTION about the tools, not an invariant about the machine --
    R1's own measurement shows the allocator finding a different assignment than
    the ideal one -- which is why exceeding it is PRICED and never refused.
    """
    load = [0] * NPU2_COLUMNS
    for w, d in demand.herds:
        for c in range(min(w, NPU2_COLUMNS)):
            load[c] += d * lanes
    for _ in range(demand.shim_global * lanes):
        c = min(range(NPU2_COLUMNS), key=lambda i: load[i])
        load[c] += 1
    return load


def legality(mapping: Mapping) -> Verdict:
    """The static verdict on one mapping. No compile, no device.

    Refuses only what is unroutable under EVERY placement; prices the rest. See
    the module docstring for where that line falls and why.
    """
    v = Verdict()

    # --- constructed-in shape arithmetic (the builders raise on these) ------
    if mapping.cols % mapping.gemm_herd_x or (mapping.cols // mapping.gemm_herd_x) % 16:
        v.refuse(
            f"emb_dim {mapping.cols} does not split across gemm_herd_x "
            f"{mapping.gemm_herd_x} into a tile_n multiple of 16 "
            "(builders/ffn_accum)"
        )
    if FFN_DIM % mapping.tile_k or mapping.tile_k % MICRO:
        v.refuse(
            f"ffn_dim {FFN_DIM} does not divide by tile_k {mapping.tile_k}, "
            f"itself a multiple of {MICRO} (builders/ffn_accum)"
        )
    if BAND_ROWS % (mapping.norm_herd_x * mapping.rows_per_call):
        v.refuse(
            f"band rows {BAND_ROWS} not divisible by norm_herd_x*rows_per_call "
            f"({mapping.norm_herd_x * mapping.rows_per_call}) "
            "(builders/norm_tail)"
        )
    if mapping.cols % NORM_TAIL_VEC_LEN:
        v.refuse(
            f"cols {mapping.cols} is not a multiple of {NORM_TAIL_VEC_LEN} "
            "(builders/norm_tail)"
        )

    # --- scope legality: Para needs independent children -------------------
    for seam in mapping.seam_pairs:
        scope = mapping.scope(seam)
        if scope in SCOPE_NEEDS_INDEPENDENCE:
            v.refuse(
                f"seam {seam[0]}->{seam[1]} is {scope} but carries a data "
                "dependence; TileFlow section 4.1: Para is only applicable to "
                "tiles without data dependency"
            )

    if not v.legal:
        return v

    per_segment = segments(mapping)
    whole = whole_design(mapping, per_segment)
    lanes = mapping.parallel_bands

    # --- cores: a measured cliff -------------------------------------------
    widths = tuple(w for w, _ in whole.herds)
    v.report["cores"] = whole.cores
    v.report["herds"] = len(whole.herds)
    if whole.cores > NPU2_CORES:
        v.refuse(
            f"{whole.cores} cores over {len(whole.herds)} herds, above NPU2's "
            f"{NPU2_CORES}. MEASURED (doc 31b section 3.5): nine herds of "
            "[4, 1] refuse with \"'aie.tile' op row index (6) must be less "
            'than the number of rows in the device (6)"'
        )
    elif not _row_packable(tuple(sorted(widths, reverse=True))):
        v.refuse(
            f"herd widths {sorted(widths, reverse=True)} do not pack into "
            f"{NPU2_CORE_ROWS} rows of {NPU2_COLUMNS} columns; a [w, 1] herd "
            "takes w columns of ONE row"
        )

    # --- the shim budget ---------------------------------------------------
    # Segments are time-multiplexed, so each meets the whole budget on its own;
    # band lanes are concurrent, so they add.
    worst_segment = max(
        per_segment, key=lambda d: d.shim_mm2s_slots, default=Demand()
    )
    total_slots = worst_segment.shim_mm2s_slots * lanes
    v.report["shim_mm2s_slots"] = total_slots
    v.report["shim_mm2s_budget"] = NPU2_SHIM_MM2S_PORTS

    # REFUSE (a): placement-invariant -- every column this herd occupies carries
    # all of its herd-direct operands, whichever columns those are.
    for d in per_segment:
        worst_direct = max((direct for _, direct in d.herds), default=0)
        if worst_direct > SHIM_MM2S_PER_COLUMN:
            v.refuse(
                f"a herd fetches {worst_direct} L3 operands herd-direct, so "
                f"every column it occupies demands {worst_direct} of "
                f"{SHIM_MM2S_PER_COLUMN} shim MM2S under EVERY placement. "
                "MEASURED (queue item 10): at three, AIR emits zero inbound "
                "aie.flow and 12 aie.packet_flow"
            )
            break

    # REFUSE (b): placement-invariant -- there are only 16 slots.
    if total_slots > NPU2_SHIM_MM2S_PORTS:
        v.refuse(
            f"{total_slots} shim MM2S channel-slots against NPU2's "
            f"{NPU2_SHIM_MM2S_PORTS} (8 columns x {SHIM_MM2S_PER_COLUMN}); no "
            "placement fits. Doc 23: J1 failed exactly here -- '8 columns x 2 = "
            "16, already full before the third stream'"
        )

    # PRICE: the assignment the tools are known to produce may be over budget
    # while a legal one exists. AIR packet-multiplexes rather than refusing, so
    # this point stays in the space and carries its degradation.
    predicted = max(
        (max(predicted_columns(d, lanes)) for d in per_segment), default=0
    )
    v.report["predicted_column_demand"] = predicted
    if predicted > SHIM_MM2S_PER_COLUMN:
        v.prices["shim_mm2s_per_column"] = min(
            1.0, SHIM_MM2S_PER_COLUMN / float(predicted)
        )

    s2mm_worst = max((d.shim_s2mm for d in per_segment), default=0) * lanes
    v.report["shim_s2mm_slots"] = s2mm_worst
    if s2mm_worst > NPU2_COLUMNS * SHIM_S2MM_PER_COLUMN:
        v.refuse(
            f"{s2mm_worst} shim S2MM channel-slots against "
            f"{NPU2_COLUMNS * SHIM_S2MM_PER_COLUMN}"
        )

    # --- memtile ports: a cliff (there is no port to give) ------------------
    for d in per_segment:
        for mm2s, s2mm in d.memtiles:
            if mm2s > MEMTILE_MM2S:
                v.refuse(
                    f"a memtile feed of {mm2s} sub-channels over {MEMTILE_MM2S} "
                    "MM2S ports (builders/ffn_accum: 'one feed sub-channel per "
                    "core, each a memtile MM2S port; a memtile has 6')"
                )
                break
            if s2mm > MEMTILE_S2MM:
                v.refuse(f"a memtile takes {s2mm} inbound streams over {MEMTILE_S2MM}")
                break

    # --- the reduction fan-out: a routing cliff ----------------------------
    # Every down core consumes EVERY chunk the GeLU stage produces, so the stream
    # must be broadcast gemm_herd_x ways. A down core's two S2MM ports are
    # already spoken for (A|B feed + hoisted C fetch) and an air channel has one
    # physical source, so the only node that can serialise gemm_herd_x producers
    # into gemm_herd_x identical feeds is a memtile (builders/ffn_resident, "THE
    # HAND-OFF FANS THROUGH A MEMTILE BY PORT ARITHMETIC, not by choice").
    if mapping.gemm_fold == "split" and mapping.gemm_herd_x > 1:
        if mapping.scope(("gelu", "down")) in SCOPE_SPACE:
            need = mapping.gemm_herd_x
            if need > MEMTILE_MM2S or need > MEMTILE_S2MM:
                v.refuse(
                    f"the GeLU->down fan-out needs a memtile to serialise "
                    f"{need} producers into {need} feeds, over the memtile's "
                    f"{MEMTILE_MM2S}/{MEMTILE_S2MM} ports"
                )

    # --- core DMA ports ----------------------------------------------------
    for d in per_segment:
        if d.core_s2mm > CORE_S2MM or d.core_mm2s > CORE_MM2S:
            v.refuse(
                f"a core wants {d.core_s2mm} S2MM / {d.core_mm2s} MM2S over "
                f"{CORE_S2MM}/{CORE_MM2S}"
            )
            break

    # --- capacity ----------------------------------------------------------
    v.report["l1_bytes"] = whole.l1_bytes
    if whole.l1_bytes > L1_BYTES:
        v.refuse(
            f"L1 {whole.l1_bytes} B over the {L1_BYTES} B tile. MEASURED: "
            "rows_per_call 8 at cols 768 is refused by aiecc's allocator "
            "('allocated buffers exceeded available memory')"
        )
    v.report["l2_bytes"] = whole.l2_bytes
    if whole.l2_bytes > L2_BYTES:
        v.refuse(f"L2 {whole.l2_bytes} B over the {L2_BYTES} B memtile")

    return v


# ---------------------------------------------------------------------------
# The measurement: how big is the space, before and after.
# ---------------------------------------------------------------------------


def raw_space_size() -> int:
    """Every axis at its full declared range, nothing filtered."""
    n = 1
    for a in AXES:
        n *= len(a.values)
    # The seam axis has a different length per gemm_fold, so it is not a plain
    # factor: fold out of the product above and put both arms back.
    n //= len(axis("gemm_fold").values)
    return n * (len(SCOPES) ** len(SEAMS_SPLIT) + len(SCOPES) ** len(SEAMS_FOLDED))


def _divisible(norm_herd_x: int, rows_per_call: int) -> bool:
    return BAND_ROWS % (norm_herd_x * rows_per_call) == 0


def constructed_in_space_size() -> int:
    """After the divisibility the builders raise on -- Timeloop's first tier.

    Doc 46: "two-tier legality (divisors constructed-in, capacity rejected)". An
    enumerator that generated the raw product would be generating shapes no
    builder will build, so this is the honest "before legality" figure and the
    raw one is reported beside it.
    """
    n_pairs = sum(
        1
        for n in axis("norm_herd_x").values
        for r in axis("rows_per_call").values
        if _divisible(n, r)
    )
    n_gemm = sum(
        1
        for g in axis("gemm_herd_x").values
        if EMB_DIM % g == 0 and (EMB_DIM // g) % 16 == 0
    )
    n_tile_k = sum(
        1 for t in axis("tile_k").values if FFN_DIM % t == 0 and t % MICRO == 0
    )
    n_routes = len(ROUTES) ** len(L3_OPERANDS)
    n_forms = len(axis("nt1_form").values) * len(axis("nt2_form").values)
    n_bands = len(axis("parallel_bands").values)
    per_fold = n_forms * n_gemm * n_pairs * n_tile_k * n_bands * n_routes
    return per_fold * (
        len(SCOPES) ** len(SEAMS_SPLIT) + len(SCOPES) ** len(SEAMS_FOLDED)
    )


#: The scopes a tail seam may legally take. Para is refused at every one of
#: them (the chain is dependent), so enumerating it would only add zeroes.
LEGAL_SEAM_SCOPES = tuple(s for s in SCOPES if s not in SCOPE_NEEDS_INDEPENDENCE)


def _structures():
    for nt1 in axis("nt1_form").values:
        for nt2 in axis("nt2_form").values:
            for fold in axis("gemm_fold").values:
                for g in axis("gemm_herd_x").values:
                    for n in axis("norm_herd_x").values:
                        for bands in axis("parallel_bands").values:
                            yield nt1, nt2, fold, g, n, bands


def _mapping(nt1, nt2, fold, g, n, bands, seams, pairs, rpc, tk, combo=None):
    if combo is None:
        routes = tuple((op, ROUTE_L2_STAGED) for op, _ in L3_OPERANDS)
    else:
        routes = tuple((op, combo[i]) for i, (op, _) in enumerate(L3_OPERANDS))
    return Mapping(
        nt1_form=nt1,
        nt2_form=nt2,
        gemm_fold=fold,
        gemm_herd_x=g,
        norm_herd_x=n,
        tile_k=tk,
        rows_per_call=rpc,
        parallel_bands=bands,
        routes=routes,
        seams=tuple(zip(pairs, seams)),
    )


def _numeric_points(norm_herd_x):
    return [
        (r, t)
        for r in axis("rows_per_call").values
        if _divisible(norm_herd_x, r)
        for t in axis("tile_k").values
    ]


#: Refusal texts that cannot depend on the routing vector: cores and row
#: packing are functions of the herd widths alone, and the L1/divisibility
#: clauses of the tiling alone. Used only to skip work -- see ``count_one``.
_ROUTING_INDEPENDENT_REFUSALS = ("cores over", "do not pack into", "not divisible")


def _routing_vectors():
    """All routing vectors, all-staged FIRST.

    Order is a search heuristic and nothing else: staging minimises shim slots,
    so the all-staged vector is the likeliest witness, and finding the witness
    on the first try rather than the last is most of the census's runtime.
    """
    combos = list(itertools.product(ROUTES, repeat=len(L3_OPERANDS)))
    all_staged = tuple([ROUTE_L2_STAGED] * len(L3_OPERANDS))
    combos.remove(all_staged)
    return [all_staged] + combos


def count_one(nt1, nt2, fold, g, n, bands, seams, pairs):
    """(legal, priced) over ``(rows_per_call, tile_k) x routing`` at one structure.

    Factorised, and the factorisation needs a probe point that is itself LEGAL:
    counting the routing axis at an arbitrary tiling, or the tiling axis at an
    arbitrary routing, reads zero whenever that arbitrary choice happens to be
    the illegal one -- which is a silent undercount, not a visible failure. So a
    legal witness is found first and both sub-counts are taken through it.

    ``test_the_factorisation_matches_brute_force`` brute-forces whole structures
    against this, so a clause that couples the two axis groups fails there.
    """
    numeric = _numeric_points(n)
    if not numeric:
        return 0, 0
    routings = _routing_vectors()

    # Staging an operand costs ONE shim slot; fetching it herd-direct costs one
    # per column, so the all-staged routing MINIMISES the slot total -- and the
    # total does not depend on the tiling axes at all. If that minimum already
    # busts the 16-slot budget, nothing at this structure is legal and the
    # witness search need not run. Sound, and it is most of the census's runtime.
    r_probe, t_probe = numeric[0]
    floor = legality(
        _mapping(
            nt1, nt2, fold, g, n, bands, seams, pairs, r_probe, t_probe,
            tuple([ROUTE_L2_STAGED] * len(L3_OPERANDS)),
        )
    )
    if any("channel-slots" in x for x in floor.refusals):
        return 0, 0
    # Cores and row packing are functions of the herd widths alone, so a refusal
    # naming one of them holds for every routing at this structure too.
    if any(
        any(tag in x for tag in _ROUTING_INDEPENDENT_REFUSALS) for x in floor.refusals
    ):
        return 0, 0

    if floor.legal:
        witness = (r_probe, t_probe, routings[0])
    else:
        witness = None
        for r, t in numeric:
            for combo in routings:
                if legality(
                    _mapping(nt1, nt2, fold, g, n, bands, seams, pairs, r, t, combo)
                ).legal:
                    witness = (r, t, combo)
                    break
            if witness:
                break
    if witness is None:
        return 0, 0
    r0, t0, combo0 = witness

    n_num = sum(
        1
        for r, t in numeric
        if legality(
            _mapping(nt1, nt2, fold, g, n, bands, seams, pairs, r, t, combo0)
        ).legal
    )
    n_route = n_priced = 0
    for combo in routings:
        v = legality(_mapping(nt1, nt2, fold, g, n, bands, seams, pairs, r0, t0, combo))
        if not v.legal:
            continue
        n_route += 1
        if v.prices:
            n_priced += 1
    return n_num * n_route, n_num * n_priced


def enumerate_legal(report_every=0):
    """Walk the space, factorised.

    Returns a dict: the two counts, plus the projections that make the number
    mean something. A bare "3.7 million legal points" cannot be compared with
    doc 38's "SEVEN legal (parallel_heads, parallel_ffn) tails", because iron's
    seven is a TWO-AXIS SLICE of its space and not its size. So the same slices
    are taken here -- the spatial-replication pair, and the whole structural
    sub-space -- and it is those that answer whether enumeration beats search.

    For a fixed (structure, seam vector) the predicate reads ``(rows_per_call,
    tile_k)`` only through the capacity clauses and the routing vector only
    through the port clauses, so the legal count factorises -- see ``count_one``.
    """
    legal = priced = 0
    example = None
    seen = 0
    legal_structures = set()
    legal_shapes = 0
    replication = set()
    widths = set()
    for nt1, nt2, fold, g, n, bands in _structures():
        pairs = SEAMS_FOLDED if fold == "folded" else SEAMS_SPLIT
        if not _numeric_points(n):
            continue
        for seams in itertools.product(LEGAL_SEAM_SCOPES, repeat=len(pairs)):
            seen += 1
            n_legal, n_priced = count_one(nt1, nt2, fold, g, n, bands, seams, pairs)
            legal += n_legal
            priced += n_priced
            if not n_legal:
                continue
            legal_shapes += 1
            legal_structures.add((nt1, nt2, fold, g, n, bands))
            replication.add((g, bands))
            widths.add((g, n))
            if example is None:
                r0, t0 = _numeric_points(n)[0]
                for combo in _routing_vectors():
                    m = _mapping(nt1, nt2, fold, g, n, bands, seams, pairs, r0, t0, combo)
                    v = legality(m)
                    if v.legal and not v.prices:
                        example = m
                        break
            if report_every and seen % report_every == 0:
                print(f"[mapping-space]   ... {seen} (structure, seam) pairs walked")
    return {
        "legal": legal,
        "priced": priced,
        "example": example,
        # The structural sub-space: how many distinct (forms, fold, widths,
        # bands) survive at all, and how many of those x seam vector.
        "legal_structures": len(legal_structures),
        "legal_shapes": legal_shapes,
        # iron's slice, ours: the two spatial-replication axes.
        "legal_replication_pairs": sorted(replication),
        "legal_width_pairs": sorted(widths),
    }


def count_space():
    out = dict(enumerate_legal())
    out["raw"] = raw_space_size()
    out["constructed_in"] = constructed_in_space_size()
    return out


# ---------------------------------------------------------------------------
# The controls. Both run every time main() does.
# ---------------------------------------------------------------------------


def r1_interior_demand() -> Demand:
    """R1's FFN interior as one segment: the object doc 31b section 3.6 counted.

    Herds: up [4,1], gelu [4,1], down [4,1]. Herd-direct L3 streams: the down
    herd's hoisted C accumulator fetch, one per column. L2-staged: hidden, w_up,
    w_down. Memtiles: the up feed (4 MM2S out, 2 staged in) and the down feed
    (4 MM2S out, 1 staged + 4 GeLU puts in).

    Built from the declaration rather than through ``Mapping`` because
    ``Mapping`` describes R2's five-group tail and R1 is three of those groups
    with no norm tails at all.
    """
    w = 4
    up = Demand(herds=((w, 0),), shim_global=2, memtiles=((w, 2),))
    gelu = Demand(herds=((w, 0),))
    down = Demand(
        herds=((w, 1),), shim_global=1, shim_s2mm=w, memtiles=((w, 1 + w),)
    )
    return compose(SCOPE_PIPE, (up, gelu, down))


def over_budget_demand(n_l3: int = SHIM_MM2S_PER_COLUMN + 1, herd_x: int = 4) -> Demand:
    """The census negative control, as a declaration.

    ``ffn_resident_structure.build_over_budget_module``: one segment, one herd of
    [herd_x, 1], every lane fetching all ``n_l3`` L3 operands herd-direct.
    MEASURED through aircc on 2026-08-12: zero inbound ``aie.flow`` and 12
    ``aie.packet_flow``, three streams per column. The compiled census had to be
    widened twice before it could see that; a STATIC predicate reads the
    declaration, so it sees ``n_l3`` directly -- and this control exists to prove
    it does, on the same design, rather than to assume it.
    """
    return Demand(herds=((herd_x, n_l3),), shim_s2mm=herd_x)


def _refuses(demand: Demand) -> bool:
    """The shim clause alone, applied to a bare ``Demand``."""
    if max((d for _, d in demand.herds), default=0) > SHIM_MM2S_PER_COLUMN:
        return True
    return demand.shim_mm2s_slots > NPU2_SHIM_MM2S_PORTS


def check_controls() -> list:
    """Both controls plus the two measured tables. Empty return means all held."""
    problems = []

    # NEGATIVE: a design that is genuinely over budget must be REFUSED.
    bad = over_budget_demand()
    worst_direct = max(d for _, d in bad.herds)
    refused = _refuses(bad)
    print(
        f"[mapping-space] negative control: one herd [4,1] with {worst_direct} "
        f"herd-direct L3 operands -> {bad.shim_mm2s_slots} of "
        f"{NPU2_SHIM_MM2S_PORTS} slots, {worst_direct} per column, "
        f"predicted {predicted_columns(bad)} -> "
        f"{'REFUSED' if refused else 'ADMITTED'}"
    )
    if worst_direct <= SHIM_MM2S_PER_COLUMN:
        problems.append(
            "negative control: the control is not over budget, so refusing it "
            "proves nothing -- pick a control that is"
        )
    if not refused:
        problems.append(
            "negative control: the predicate ADMITTED a design demanding "
            f"{worst_direct} shim MM2S per column under every placement, so it "
            "cannot fail and is not gating anything"
        )

    # POSITIVE: R1 is NOT over budget and must be accepted, and the static count
    # must reproduce the measured one.
    r1 = r1_interior_demand()
    slots = r1.shim_mm2s_slots
    direct = sum(w * d for w, d in r1.herds)
    staged = r1.shim_global
    print(
        f"[mapping-space] positive control: R1 interior -> shim MM2S "
        f"{slots}/{NPU2_SHIM_MM2S_PORTS}, {direct} shim->core + {staged} "
        f"shim->memtile, {r1.cores} cores over {len(r1.herds)} herds -> "
        f"{'REFUSED' if _refuses(r1) else 'ADMITTED'}"
    )
    if (slots, direct, staged) != (7, 4, 3):
        problems.append(
            f"positive control: R1 statically counts {slots}/16 ({direct} "
            f"shim->core + {staged} shim->memtile); doc 31b section 3.6 "
            "MEASURED 7 of 16, 4 shim->core + 3 shim->memtile"
        )
    if _refuses(r1):
        problems.append("positive control: R1 was refused, and it routes today")
    mt = sorted(r1.memtiles)
    print(f"[mapping-space] positive control: R1 memtiles (MM2S, S2MM) {mt}")
    if mt != [(4, 2), (4, 5)]:
        problems.append(
            f"positive control: memtile occupancy {mt}; doc 31b section 3.6 "
            "MEASURED 4/6 MM2S with 2/6 S2MM on one and 4/6 with 5/6 on the "
            "other"
        )

    # The herd inventory, doc 31b section 5, MEASURED at section 3.5.
    for n_herds, expected in ((9, False), (8, True), (7, True), (5, True)):
        widths = tuple([4] * n_herds)
        ok = 4 * n_herds <= NPU2_CORES and _row_packable(widths)
        print(
            f"[mapping-space] herd inventory: {n_herds} herds of [4,1] = "
            f"{4 * n_herds} tiles -> {'places' if ok else 'REFUSED'}"
        )
        if ok != expected:
            problems.append(
                f"herd inventory: {n_herds} herds of [4,1] came out "
                f"{'legal' if ok else 'illegal'}; doc 31b section 3.5 MEASURED "
                f"{'placed' if expected else 'refused'}"
            )

    # J7a's own column budget, doc 23: three 8-wide herds, packed + gamma, and
    # the rule says that is EXACTLY two per column.
    j7a = compose(
        SCOPE_PIPE,
        (
            Demand(herds=((8, 1),)),  # stage_add fetches packed
            Demand(herds=((8, 0),)),  # stage_norm fetches nothing
            Demand(herds=((8, 1),), shim_s2mm=8),  # stage_scale fetches gamma
        ),
    )
    load = predicted_columns(j7a)
    print(f"[mapping-space] J7a control: predicted per-column MM2S {load}")
    if max(load) != SHIM_MM2S_PER_COLUMN or _refuses(j7a):
        problems.append(
            f"J7a control: predicted per-column demand {max(load)} and "
            f"{'refused' if _refuses(j7a) else 'admitted'}; doc 23 says the "
            "budget 'is exactly met by the packed fetch and the gamma fetch'"
        )
    return problems


# ---------------------------------------------------------------------------


def _print_axes():
    print("[mapping-space] axes, and what bounds each")
    for a in AXES:
        flag = "   UNBOUNDED BY ARTIFACT" if a.unbounded_by_artifact else ""
        print(f"  {a.name:<18} {len(a.values):>3} values {a.values}{flag}")
        print(f"       source: {a.source}")
    print(
        f"  {SEAM_AXIS.name:<18} {len(SEAM_AXIS.values):>3} values "
        f"{SEAM_AXIS.values} x {len(SEAMS_SPLIT)} seams (split) / "
        f"{len(SEAMS_FOLDED)} (folded)"
    )
    print(f"       source: {SEAM_AXIS.source}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--axes", action="store_true", help="print the axis table")
    args = parser.parse_args(argv)

    if args.axes:
        _print_axes()
        return 0

    problems = check_controls()
    counts = count_space()
    raw, constructed = counts["raw"], counts["constructed_in"]
    legal, priced = counts["legal"], counts["priced"]

    print(f"[mapping-space] space before legality: {constructed} (raw product {raw})")
    print(f"[mapping-space] space after legality: {legal}")
    if constructed:
        print(
            "[mapping-space] legality removes "
            f"{100.0 * (constructed - legal) / constructed:.4f}% of the space"
        )
    print(f"[mapping-space] of the legal points, {priced} are priced, not refused")
    # The slices that answer "does enumeration beat search". Doc 38's SEVEN is a
    # two-axis slice of iron's space, not its size, so the comparable figures are
    # these -- not the headline count.
    print(
        f"[mapping-space] legal structures: {counts['legal_structures']}, "
        f"legal (structure, seam vector): {counts['legal_shapes']}"
    )
    print(
        "[mapping-space] legal (gemm_herd_x, parallel_bands) pairs: "
        f"{len(counts['legal_replication_pairs'])} "
        f"{counts['legal_replication_pairs']}"
    )
    print(
        "[mapping-space] legal (gemm_herd_x, norm_herd_x) pairs: "
        f"{len(counts['legal_width_pairs'])} {counts['legal_width_pairs']}"
    )
    unbounded = [a.name for a in AXES if a.unbounded_by_artifact]
    print(f"[mapping-space] axes no artifact bounds: {unbounded or ['none']}")

    if legal <= 0:
        problems.append(
            "the legal space is empty, so the predicate rejects everything and "
            "the count means nothing"
        )
    if legal >= constructed:
        problems.append(
            "the predicate removed nothing, so it is not a filter and the "
            "'after' number is the 'before' number"
        )
    if priced <= 0:
        problems.append(
            "no legal point is priced, so the slope is dead code and this is a "
            "cliff-only predicate -- doc 44's correction is not implemented"
        )
    if priced >= legal:
        problems.append(
            "every legal point is priced, so the price does not discriminate"
        )

    if problems:
        print("[mapping-space] FAIL")
        for p in problems:
            print(f"  {p}")
        return 1
    print("[mapping-space] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
