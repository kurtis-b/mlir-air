# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``coarse`` cell C2 — the block front, the decomposed tail.

CONTRACT
    ``prepare_coarse_c2(shape, seed=...)`` is this cell's entry in the
    ``SPECS`` catalogue. The device path is ``pattern/coarse/cells.py``, which
    calls ``builders/block.py``'s front region and ``pattern/runlist``'s tail
    regions; this module adds the cell's own ELF cache and its stage label and
    nothing else. If dispatch logic starts appearing here, the cell has become
    a fork and has stopped measuring the code the block and runlist gates
    validated.

WHAT THIS CELL IS FOR
    `28-coarse-blend-space.md` derives ``coarse``'s blend space as two axes over
    six cells and finds that two of the six already have mode names. C2 is one
    of the two INTERIOR cells: it keeps ``coarse``'s front -- two ELFs, q/k/v
    device-resident -- and refines only the tail, into ``runlist``'s
    up/GeLU/down plus per-band add/LayerNorm/multiply. Paired with C3, which
    varies the other axis, it is what lets the blend be chosen by measurement
    rather than inherited from D2.

    It is a MEASUREMENT POINT, not a proposed default. What comes out of the
    walk decides which cell ``coarse`` selects at a given workload.

FOOTGUNS
    - ``COARSE_C2_CACHE_DIR`` is this cell's OWN directory, and that is a
      correctness requirement rather than tidiness: ``KernelCache`` picks the
      directory by NAME, so two modes sharing one can trade ELFs whose
      fingerprints happen to agree -- numerically valid output attributed to
      the wrong execution boundary, invisible to every equivalence check. It
      must be in ``transformer_layer/.gitignore`` AND the Makefile ``clean``
      target: the driver's negative control runs ``opcheck.py`` from the SOURCE
      directory, so the cache lands in the source tree exactly the way D2's
      ``block_cache/`` leak did.
    - The recorded ``execution_mode`` is ``coarse``'s CSV value. A cell is a
      point INSIDE the mode, not a fifth taxonomy point, and a results tree
      that gave it its own value would compare it against the other three as
      though it were one.
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
COARSE_C2_CACHE_DIR = "coarse_c2_cache"


def prepare_coarse_c2(shape, seed=42):
    """The C2 cell's ``SPECS`` preparer: block front, decomposed tail.

    Same golden model, same measured injection target and same per-boundary
    comparisons as every other whole-layer mode — by construction, not by
    parallel implementation.
    """
    return prepare_cell(
        "C2", shape, seed=seed, cache_dir=COARSE_C2_CACHE_DIR, label="coarse_c2"
    )
