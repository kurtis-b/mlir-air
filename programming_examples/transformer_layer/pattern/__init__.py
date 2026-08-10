# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The transformer-layer patterns: one shared golden model, then the strategies.

``reference.py`` is the correctness anchor for the whole block -- the tensor
draws (in iron's order, which is load-bearing), the layer's structure for both
workload variants, and every operator boundary it passes through. Phase E's
execution-strategy directories are built BESIDE it and import this one copy;
iron carries three 8-line ``reference.py`` re-exports of its shared model and
porting convention 8 deletes them rather than reproducing them.

Nothing here touches hardware or builds MLIR. The device side of a pattern is
``builders/block.py``.

``EXECUTION_MODE_CSV`` below is the ONE place a mode's code name maps to the
CSV value iron's result trees use. Porting convention 7 renames iron's
``hybrid`` module to ``coarse`` everywhere in code, directories and prose, and
keeps ``hybrid`` only as the CSV *value* so results remain diffable against
iron's; ``fused_elf`` is a new CSV value, not a new column
(03-measurement-model.md). A mode reads its value from here rather than
spelling it out per call site, so the rename cannot drift apart from the data
it labels.
"""

#: mode name in this repository -> ``execution_mode`` value in the study CSV.
#:
#: `[2026-08-09]` ``coarse_c2`` and ``coarse_c3`` are CELLS of ``coarse``
#: (28-coarse-blend-space.md), not new taxonomy points, so they record
#: ``coarse``'s value. Giving a cell its own value would put it in
#: ``schema.EXECUTION_MODES`` and invite a cross-mode table to compare four
#: modes against a fifth thing that is one of them measured differently. A cell
#: separates in a results tree by ``study_case_label`` and by the per-mode CSV
#: filename, and the cell itself is recorded in the artifact as ``blend_cell``.
EXECUTION_MODE_CSV = {
    "coarse": "hybrid",
    "coarse_c2": "hybrid",
    "coarse_c3": "hybrid",
    "offload": "offload",
    "runlist": "runlist",
    "fused": "fused_elf",
}
