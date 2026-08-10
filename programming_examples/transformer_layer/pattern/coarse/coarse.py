# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``coarse`` — few fused kernels over one runlist per sequence.

CONTRACT
    ``prepare_coarse(shape, seed=...)`` is this mode's entry in the ``SPECS``
    catalogue: the D2 layer prepared for ``opcheck.py``'s ``dispatch`` seam,
    with this mode's own ELF cache directory and its ``execution_mode`` CSV
    value. The device path is ``builders/block.py``, UNCHANGED — five operator
    launches over four ELFs through ``KernelCache.run_sequence``, which records
    one ``DispatchVector`` per sequence. This module adds the mode layer and
    nothing else; if logic from ``block.py`` starts appearing here, the mode
    has quietly become a fork and stopped measuring the code D2 validated.

WHY THE WRAPPER EXISTS AT ALL
    ``block`` is the integration gate: it proves the layer computes the right
    numbers. ``coarse`` is the same dispatch measured as an execution strategy:
    the coarse-runlist point of the Phase E taxonomy, whose recorded dispatch
    vectors are the calibration every other mode's distinguishability clause is
    defined against. Same code, different claim — and the claim needs its own
    operator name, artifact and cache so the driver can hold each to its own
    contract.

FOOTGUNS
    - ``COARSE_CACHE_DIR`` is this mode's OWN directory, and that is a
      correctness requirement, not tidiness: ``KernelCache`` picks the
      directory by NAME, so two modes sharing one can trade ELFs whose
      fingerprints happen to agree — numerically valid output attributed to the
      wrong execution boundary, invisible to every equivalence check. The
      directory must be in ``transformer_layer/.gitignore`` AND the Makefile
      ``clean`` target: the driver's negative control runs ``opcheck.py`` from
      the SOURCE directory, so the cache lands in the source tree exactly the
      way D2's ``block_cache/`` leak did.
    - ``execution_mode`` is read from ``pattern.EXECUTION_MODE_CSV`` — the one
      place convention rule 7 lets iron's old mode name survive, as the CSV
      value — and this import is the one place that mapping is applied for
      this mode. Do not inline the string.
    - The dispatch vectors are recorded by ``run_block`` on the fault-injected
      path too. The driver requires the fault artifact's summed totals to EQUAL
      the clean run's; anything conditional on the injected flag fails that.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE_ROOT = os.path.dirname(os.path.dirname(_HERE))  # transformer_layer/
_PROJ_ROOT = os.path.dirname(_EXAMPLE_ROOT)  # programming_examples/
for _p in (_PROJ_ROOT, os.path.join(_PROJ_ROOT, "llms"), _EXAMPLE_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from opcheck_layer import prepare_layer_dispatch  # noqa: E402

from pattern import EXECUTION_MODE_CSV  # noqa: E402

#: This mode's ELF cache, relative to the working directory — like
#: ``BLOCK_CACHE_DIR``, but the mode's own. See the module footguns.
COARSE_CACHE_DIR = "coarse_cache"

# ---------------------------------------------------------------------------
# WHICH BLEND THIS MODE IS, AND WHAT CHOSE IT
#
# `coarse` is defined as a per-workload BLEND of `runlist` and `fused`, and
# until 2026-08-09 nothing recorded which blend it was. 28-coarse-blend-space.md
# derives the space from the artifact plans -- front {block, runlist} x tail
# {stitched, banded, decomposed}, six cells, four of them already owned by an
# existing mode -- and 30-coarse-cells-built.md measures the four that build at
# seq >= 2048, which is where `fused`'s stitched tail cannot pack and therefore
# where `coarse` is a mode rather than `fused` under another name.
#
# The measurement, walked TWICE at 2048 and 4096: C1 < C2 < C3 < C6 on averages
# and on minimums at both lengths. This cell wins.
#
# THE DISPATCH DID NOT CHANGE. C1 is `builders/block.py`, which is what this
# module has always called. What changed is that the choice now has provenance:
# it was measured against its three siblings rather than inherited from D2
# having been built at 4096. If a future workload admits a stitched tail, or a
# retune moves the front axis, the answer is a fresh walk of the cells -- not an
# edit here.
#
# `coarse_cells_structure.py` and the two cell gates keep the alternatives
# runnable, so re-deciding costs a measurement rather than a rebuild.
# ---------------------------------------------------------------------------

#: The cell of 28-coarse-blend-space.md's space that this mode dispatches.
BLEND_CELL = "C1"
BLEND_FRONT = "block"
BLEND_TAIL = "banded"

#: How the cell was chosen, recorded into every results artifact so a row can
#: be traced to the measurement behind it rather than to a convention.
BLEND_SELECTED_BY = (
    "measured: fastest of the four cells that build at seq >= 2048, on averages "
    "and minimums, over two walks at 2048 and 4096 (30-coarse-cells-built.md)"
)


def prepare_coarse(shape, seed=42):
    """The ``coarse`` mode's ``SPECS`` preparer: the D2 layer, measured as a mode.

    Same golden model, same injection target, same per-boundary comparisons as
    ``prepare_block`` — by construction, not by parallel implementation. Only
    the cache directory, the stage-print label, the recorded ``execution_mode``
    and the recorded blend cell differ.
    """
    return prepare_layer_dispatch(
        shape,
        seed=seed,
        cache_dir=COARSE_CACHE_DIR,
        label="coarse",
        extra={
            "execution_mode": EXECUTION_MODE_CSV["coarse"],
            # THE BLEND, AND WHY IT IS THIS ONE. See the module note above.
            # Recorded rather than implicit: the dispatch is unchanged, and
            # what is new is that the artifact now says which cell of
            # 28-coarse-blend-space.md's space was run and what selected it.
            "blend_cell": BLEND_CELL,
            "blend_front": BLEND_FRONT,
            "blend_tail": BLEND_TAIL,
            "blend_selected_by": BLEND_SELECTED_BY,
        },
    )
