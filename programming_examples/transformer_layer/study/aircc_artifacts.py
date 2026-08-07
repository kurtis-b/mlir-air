# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Where aircc leaves the artifacts the study reads, and which one is stable.

CONTRACT
    ``routed_design(project_dir)`` returns the post-``aircc`` MLIR file carrying
    the routed AIE design -- tiles, cores, buffers, locks, flows and DMA
    allocations -- for a compile that ran with an ``air_project`` output
    directory. That file is ``air_project/aie.air.mlir``.

WHY THIS MODULE EXISTS
    Doc 09 carried an open issue: ``resource_usage/analysis.py`` regex-parses
    iron's ``input_physical.mlir``, and "'same dialect' does not guarantee a
    stable equivalent artifact under aircc. Identify the concrete post-aircc
    artifact and pin its path before this phase starts. If none is stable, emit
    one deliberately rather than parsing an incidental intermediate."

    **Resolved: ``aie.air.mlir`` is that artifact, and nothing needs emitting.**
    Measured on a real compile (the J7a norm-tail pipeline, herd_x=8):

        aie.tile             40      aie.core             24
        aie.buffer           88      aie.lock            112
        aie.flow             40      shim_dma_allocation  17

    and iron's three regexes match it UNMODIFIED -- ``_CORE_RE`` finds 24 cores
    as ``%tile_r_c``, ``_DMA_ALLOCATION_RE`` finds 17 allocations and parses
    kind/tile/direction/channel correctly (``shim_dma_allocation``,
    ``%shim_noc_tile_3_0``, ``S2MM``, ``0``), and buffers are declared in the
    expected ``aie.buffer(%tile_7_4) ... : memref<768xbf16, 2 : i32>`` form. So
    the port changes the artifact's LOCATION and not the parsing, which moves
    the parsing half of ``analysis.py`` from ADAPT toward PORT.

    The normalization constants doc 09 names carry over unchanged, because they
    are properties of AIE2P and not of either toolchain:
    ``AIE_TILE_LOCAL_MEMORY_BYTES=65536``, ``MEM_TILE_LOCAL_MEMORY_BYTES=524288``,
    ``SHIM_DMA_CHANNELS_PER_DIRECTION=2``.

FOOTGUNS
    - **Do not read ``debug_ir/pass_*.mlir`` for resource usage.** Those exist
      only under ``debug_ir=True``, they are numbered by pass ordinal so the
      name moves whenever the pipeline changes, and the last one is whatever
      pass happened to run last. ``aie.air.mlir`` is written by every compile
      and is named for what it is. Doc 09's "rather than parsing an incidental
      intermediate" is exactly this distinction.
    - **The file is per-segment-name, not per-run.** A design compiled twice
      into the same project directory overwrites it. Resource usage must be read
      before the next compile, or into a copied project directory.
    - **A compile that fails at the core-ELF link still writes it.** aircc runs
      the whole MLIR pipeline before it builds ELFs, so resource usage is
      readable without Peano and without a kernel object -- the same property
      ``norm_tail_structure.py`` relies on. A missing file therefore means the
      compile died EARLIER than lowering, which is a real failure and not a
      reason to fall back to a debug dump.
"""

from __future__ import annotations

from pathlib import Path

#: aircc's routed-design artifact, relative to the project directory. The
#: post-aircc equivalent of iron's ``input_physical.mlir``.
ROUTED_DESIGN_NAME = "aie.air.mlir"

#: AIE2P device properties the resource normalization divides by. Properties of
#: the part, not of either toolchain, so they carry over from iron unchanged.
AIE_TILE_LOCAL_MEMORY_BYTES = 65536
MEM_TILE_LOCAL_MEMORY_BYTES = 524288
SHIM_DMA_CHANNELS_PER_DIRECTION = 2


class RoutedDesignMissing(FileNotFoundError):
    """Raised when the routed design is absent, with what that implies."""


def routed_design(project_dir: str | Path) -> Path:
    """Path to the routed AIE design under ``project_dir``.

    Raises ``RoutedDesignMissing`` rather than returning a path that does not
    exist: every caller here goes on to parse the file, and a late "no such
    file" from inside a regex loop is a worse error than an early one that says
    what its absence means.
    """
    path = Path(project_dir) / ROUTED_DESIGN_NAME
    if not path.is_file():
        raise RoutedDesignMissing(
            f"{path} is absent. aircc writes it after the MLIR pipeline and "
            "before it builds core ELFs, so even a compile that fails at the "
            "link leaves it behind -- its absence means the compile died before "
            "lowering completed. Do NOT substitute a debug_ir/pass_*.mlir dump: "
            "those are numbered by pass ordinal and only exist under "
            "debug_ir=True."
        )
    return path
