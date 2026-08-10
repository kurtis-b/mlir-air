# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``coarse`` cell C3 — the runlist front, the banded tail.

CONTRACT
    ``prepare_coarse_c3(shape, seed=...)`` is this cell's entry in the
    ``SPECS`` catalogue. The device path is ``pattern/coarse/cells.py``, which
    calls ``pattern/runlist``'s front regions and ``builders/block.py``'s tail
    regions; this module adds the cell's own ELF cache and its stage label and
    nothing else. If dispatch logic starts appearing here, the cell has become
    a fork and has stopped measuring the code the block and runlist gates
    validated.

WHAT THIS CELL IS FOR
    C3 is the other interior cell of `28-coarse-blend-space.md`'s two-axis
    space: it refines the FRONT into ``runlist``'s three projections and
    per-head ``attn_scores`` -> ``softmax`` -> ``attn_output``, and keeps
    ``coarse``'s banded tail. C2 varies the tail and holds the front; this
    varies the front and holds the tail, so the pair separates the two axes
    instead of moving both at once — the error the first ``runlist`` structure
    made and the reason its entry count measured a schedule change rather than
    a decomposition.

    It is a MEASUREMENT POINT, not a proposed default.

FOOTGUNS
    - ``COARSE_C3_CACHE_DIR`` is this cell's OWN directory, for the reason
      ``pattern/coarse_c2/coarse_c2.py`` states at length: ``KernelCache``
      picks the directory by NAME, and two modes sharing one can trade ELFs
      whose fingerprints agree.
    - This cell inherits the ``runlist`` front's memory bound: ONE SUBMISSION
      PER HEAD, because twelve heads in one runlist would need every score and
      probability matrix live at once (~800 MiB at the gate configuration
      against ~70 MiB per head). It is a memory bound and not a schedule
      choice, and it does not touch the entry count the blend space compares.
    - The recorded ``execution_mode`` is ``coarse``'s CSV value. A cell is a
      point INSIDE the mode, not a fifth taxonomy point.
    - The dispatch vectors are recorded on the fault-injected path too. The
      driver requires the fault artifact's summed totals to EQUAL the clean
      run's; anything conditional on the injected flag fails that.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE_ROOT = os.path.dirname(os.path.dirname(_HERE))  # transformer_layer/
_PROJ_ROOT = os.path.dirname(_EXAMPLE_ROOT)  # programming_examples/
for _p in (_PROJ_ROOT, os.path.join(_PROJ_ROOT, "llms"), _EXAMPLE_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pattern.coarse.cells import prepare_cell  # noqa: E402

#: This cell's ELF cache, relative to the working directory. Its OWN — see the
#: module footguns.
COARSE_C3_CACHE_DIR = "coarse_c3_cache"


def prepare_coarse_c3(shape, seed=42):
    """The C3 cell's ``SPECS`` preparer: runlist front, banded tail.

    Same golden model, same measured injection target and same per-boundary
    comparisons as every other whole-layer mode — by construction, not by
    parallel implementation.
    """
    return prepare_cell(
        "C3", shape, seed=seed, cache_dir=COARSE_C3_CACHE_DIR, label="coarse_c3"
    )
