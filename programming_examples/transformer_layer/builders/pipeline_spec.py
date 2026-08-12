# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The declarative surface for on-chip pipeline staging (H8).

WHAT THIS IS
    A builder that wants its stages CO-RESIDENT declares so, per stage:

        STAGE_SPEC = (
            StageSpec("add"),
            StageSpec("norm"),
            StageSpec("scale"),
        )

    and emits one ``air.launch`` per stage, stamping each with
    ``launch_attributes(group, index, spec)``. ``air-opt
    --air-fuse-pipeline-launches`` then co-locates the group into ONE
    ``air.launch`` holding ONE ``air.segment``.

WHY FUSING INTO ONE SEGMENT IS THE OPERATION, AND NOT AN OPTIMIZATION
    Each ``air.launch`` lowers to its own ``aie.device``, so stages in separate
    launches are time-multiplexed at segment granularity and ON-CHIP RESIDENCY
    HOLDS ONLY WITHIN A SEGMENT. A pipeline whose stages hand off through
    L1->L1 ``air.channel`` edges -- which is what ``builders/norm_tail.py``
    is -- therefore does not merely run slower unfused; its declared edges span
    devices. That is why the pass REFUSES a malformed group instead of leaving
    it alone (doc 23's refuse-versus-skip discriminator: declining must leave a
    program that is correct-but-slower, and here it does not).

WHY STAGING IS DECLARED HERE AND NOT DERIVED BY THE PASS
    ``staging`` picks a LOOP CONSTRUCTION, and no pass rewrites one into the
    other:

      "l1"              stage bodies hand off in L1; nothing staged through L2.
                        builders/norm_tail.py's three herds.
      "memtile"         one operand staged through L2 and fanned out on one
                        channel -- builders/ffn_accum.py's A|B feed, where the
                        memtile sends A's k-slice then B's k-slice down the
                        same channel because a core has only two S2MM ports.
      "accum_in_place"  the accumulator round trip written naively per K step,
                        for air-hoist-dma-in-accum-pattern to lift into an
                        L1-resident ring.

    Deriving that choice is the analysis doc 16 sized H8 "large" and made
    conditional on H2. Declaring it costs a builder one word.

    The declaration is NOT decorative: the pass CHECKS it against the emitted
    IR and refuses when it does not hold. That check is the only thing in the
    toolchain that can catch a staging claim the builder lost, because BOTH
    constructions compile and compute correct numbers -- an accumulator
    allocated at herd scope instead of inside the K loop still returns the
    right answer, at a DDR round trip per step, and doc 22 records that two of
    the four cells in its table "would ship as working code and pass every
    numerical gate".

WHAT IS NOT CHECKED HERE
    The per-column shim MM2S budget. Fusing N stages into one segment ADDS
    their per-column L3-facing demand (doc 23: the budget is per column ACROSS
    stacked herds), so it is a real constraint on the fused result -- but it is
    only countable on the ROUTED design. ``pipeline_fusion_structure.py``
    counts it with ``ffn_resident_structure._shim_mm2s_census``, the census
    that already exists, rather than approximating it a third time.
"""

from dataclasses import dataclass
from typing import Optional

from air.ir import IntegerAttr, IntegerType, StringAttr

# Attribute names. MUST match mlir/include/air/Dialect/AIR/AIRDialect.h's
# xilinx::air::attrs -- the dialect verifies all three (type and the op they
# may sit on), so a typo here fails loudly at air-opt rather than silently
# leaving the pipeline unfused.
PIPELINE_GROUP_ATTR = "air.pipeline_group"
PIPELINE_STAGE_ATTR = "air.pipeline_stage"
STAGING_ATTR = "air.staging"

# Legal air.staging values, mirroring attrs::StagingL1/Memtile/AccumInPlace.
STAGING_L1 = "l1"
STAGING_MEMTILE = "memtile"
STAGING_ACCUM_IN_PLACE = "accum_in_place"
STAGING_VALUES = (STAGING_L1, STAGING_MEMTILE, STAGING_ACCUM_IN_PLACE)


@dataclass(frozen=True)
class StageSpec:
    """One pipeline stage, as the builder declares it.

    Args:
        name: the stage's name, for diagnostics and for the herd it builds.
            Not used by the pass -- grouping is by ``group`` and ordering by
            the stage index -- so renaming a stage cannot change the fusion.
        pipeline: whether this stage joins the co-resident group. ``False``
            leaves the stage in its own launch, i.e. its own ``aie.device``,
            which is the pre-H8 arrangement and is how a stage that must NOT
            be co-resident (say, one that would push a column over the shim
            budget) is expressed.
        staging: how this stage's operands are staged on chip; one of
            ``STAGING_VALUES``, or ``None`` for "unstated". ``None`` and
            ``"l1"` are both unchecked -- state one only when the builder
            really emitted it, since the pass refuses a claim it cannot see.
    """

    name: str
    pipeline: bool = True
    staging: Optional[str] = None

    def __post_init__(self):
        if self.staging is not None and self.staging not in STAGING_VALUES:
            raise ValueError(
                f"stage {self.name!r}: staging={self.staging!r} is not one of "
                f"{STAGING_VALUES}. The dialect verifier refuses an unknown "
                "value too, but failing here names the stage."
            )


def launch_attributes(group, index, spec):
    """The ``attributes=`` dict for this stage's ``air.launch``.

    Returns an EMPTY dict for a stage with ``pipeline=False``, so an unfused
    stage carries no markers at all and the pass never sees it -- rather than
    carrying a group it then has to be excluded from.

    Args:
        group: the pipeline's name. Launches sharing it, within one parent
            op, are one group.
        index: this stage's position. A group must cover ``0..N-1`` exactly
            once; the pass refuses a gap or a duplicate rather than falling
            back on IR order, since the order decides which stage's output
            feeds which.
        spec: the ``StageSpec``.
    """
    if not spec.pipeline:
        return {}
    attrs = {
        PIPELINE_GROUP_ATTR: StringAttr.get(group),
        PIPELINE_STAGE_ATTR: IntegerAttr.get(IntegerType.get_signless(64), index),
    }
    if spec.staging is not None:
        attrs[STAGING_ATTR] = StringAttr.get(spec.staging)
    return attrs


def stage_indices(stage_specs):
    """The pipeline index of each spec, ``None`` for the non-pipelined ones.

    Indices count only pipelined stages and are contiguous from 0, which is
    what the pass requires. Computing them here rather than in each builder is
    what stops a builder from declaring ``pipeline=False`` on a middle stage
    and leaving a gap -- the failure the pass reports as "no
    air.pipeline_stage = k", several layers from its cause.
    """
    out, n = [], 0
    for spec in stage_specs:
        if spec.pipeline:
            out.append(n)
            n += 1
        else:
            out.append(None)
    return out
