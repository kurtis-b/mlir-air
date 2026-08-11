# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read array occupancy off a compiled design and write a schema ``resource`` row.

    python3 study/resource_usage.py <air_project-dir> [...] --out results/resource.csv

CONTRACT
    ``parse_routed_design(path)`` turns the routed AIE artifact into a
    ``ResourceUsage``: how many compute tiles carry a core, how much L1 and L2
    each tile holds against its capacity, how many shim / memory-tile /
    compute-tile DMA channels are allocated, and how many ``aie.flow`` ops run
    core tile to core tile. ``resource_row`` puts one of those into a
    ``schema.RESOURCE_FIELDS`` row, and the CLI walks a list of project
    directories and writes them through ``results_io``.

    NO NPU, and no dispatch. Doc 09 lists ``resource_usage`` as "build tree
    only"; everything here is a property of a compiled artifact, so a row is
    reproducible from the file without the device being free.

WHY THIS IS A RETARGET AND NOT A PORT
    Doc 09 carried an open issue against iron's ``resource_usage/analysis.py``:
    it regex-parses ``input_physical.mlir``, and "'same dialect' does not
    guarantee a stable equivalent artifact under aircc". ``aircc_artifacts.py``
    resolved it -- ``air_project/aie.air.mlir`` is the stable equivalent -- and
    this module is what consumes that resolution. The FILE moved; the parsing
    did not, and iron's three regexes are kept in the shape it measured them
    matching, because a rewritten regex would have to be re-verified against the
    same artifacts for no gain.

    Two things ARE new, and both are the port earning its keep:

    - **``core_to_core_flows``.** Doc 03 makes the core->core flow count the
      discriminator for AIE role style (zero = time-multiplexed, at least the
      stage count = space-multiplexed) and records that
      ``norm_tail_structure.py``'s hand-written check is the tree's only
      instrument for it. It is one regex over the artifact this module already
      reads, so every design measured for occupancy also reports its role style.
    - **A schema table.** iron writes bare CSVs per study; here the row goes
      through ``schema``/``results_io``, so a failed parse writes a complete row
      with ``run_status=failed`` exactly as a failed measurement does, and the
      manifest and smoke gate can be pointed at it.

FOOTGUNS
    - **Utilization is POOLED over the tiles that hold something**, not over the
      array. A design on 4 tiles and one on 32 are each measured against their
      own footprint, so the ratio answers "how full is what it used" and NOT
      "how much of the device did it take". ``compute_tiles_used`` is the
      column that answers the second question, and a comparison that wants
      device occupancy must read both. iron's definition, kept so its trees
      stay comparable.
    - **An absent DMA allocation is not zero usage.** The artifact states shim
      allocations explicitly and often states nothing for memory or compute
      tiles; those columns come back ``None`` with a note saying so rather than
      0, because a 0 would read as "this design moves no data through memory
      tiles", which is a different and usually false claim.
    - **A buffer whose memref this cannot size is skipped, not counted as 0.**
      A dynamic or unrecognised element type would otherwise silently shrink
      the allocated total. Skipped buffers are counted and reported.
    - **The artifact is per-segment-name, not per-run.** Compiling twice into
      one project directory overwrites it -- see ``aircc_artifacts`` -- so parse
      before the next compile, or into a copied directory.

THE ONE DEPARTURE FROM IRON'S ARITHMETIC: ``shim_dma_channels_used``
    iron keys its combined shim set on the channel NUMBER alone
    (``analysis.py``'s ``shim_dma_channels``), so a shim tile allocating S2MM
    channel 0 and MM2S channel 0 counts **one** channel used out of four. Those
    are physically distinct hardware channels, so the combined column undercounts
    exactly where the per-direction columns -- which iron keys correctly, one set
    per direction -- say two are in use. It is not a rounding difference: a
    design using channel 0 both ways on every shim tile reads 25% occupied
    against an actual 50%.

    This keys the combined set on ``(direction, channel)``. The per-direction
    columns are unchanged and still match iron's, so a tree comparison sees a
    difference only in the one column that was wrong, and only where the two
    directions overlap on a number -- which the fixture in
    ``test_resource_usage.py`` pins in both shapes.
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import aircc_artifacts  # noqa: E402
import results_io  # noqa: E402
import schema  # noqa: E402

#: Channel budgets per tile. The per-direction shim figure and the two memory
#: capacities live in ``aircc_artifacts`` -- they are the constants doc 09 named
#: -- and are imported rather than restated, so there is one definition of each.
SHIM_DMA_CHANNELS_TOTAL = aircc_artifacts.SHIM_DMA_CHANNELS_PER_DIRECTION * 2
MEM_TILE_DMA_CHANNELS_TOTAL = 6
COMPUTE_TILE_DMA_CHANNELS_TOTAL = 2

# iron's three regexes, unmodified: `aircc_artifacts` measured all three
# matching `aie.air.mlir` as written, so rewriting them would mean re-verifying
# the same behaviour against the same files for no gain.
_CORE_RE = re.compile(r"aie\.core\(\s*(%tile_\d+_\d+)\s*\)")
_BUFFER_RE = re.compile(
    r"aie\.buffer\(\s*(%(?:tile|mem_tile)_\d+_\d+)\s*\).*?:\s*memref<([^>]+)>",
    re.DOTALL,
)
_DMA_ALLOCATION_RE = re.compile(
    r"aie\.(?P<kind>\w*dma_allocation)\s+@\w+\(\s*"
    r"(?P<tile>%(?:shim_noc_tile|mem_tile|tile)_\d+_\d+)\s*,\s*"
    r"(?P<direction>S2MM|MM2S)\s*,\s*(?P<channel>\d+)\s*\)"
)

# The two AIR-specific ones. `_TILE_RE` gives every tile its ROW, and the row is
# what says whether a flow endpoint is shim (0), memory tile (1) or compute
# (>= 2). Both are `norm_tail_structure.py`'s, which is the module doc 03 names
# as the tree's flow-count instrument; matching its definitions exactly is what
# lets a number from here be compared with one from there.
_TILE_RE = re.compile(r"%(\S+) = aie\.tile\((\d+),\s*(\d+)\)")
_FLOW_RE = re.compile(
    r"aie\.flow\(%(\S+),\s*\w+\s*:\s*\d+,\s*%(\S+),\s*\w+\s*:\s*\d+\)"
)

#: memref element type -> bytes. Anything absent makes the buffer unsizeable,
#: which is reported rather than silently treated as empty.
_DTYPE_BYTES = {
    "bf16": 2,
    "bfloat16": 2,
    "f16": 2,
    "f32": 4,
    "f64": 8,
    "i1": 1,
    "i8": 1,
    "ui8": 1,
    "i16": 2,
    "ui16": 2,
    "i32": 4,
    "ui32": 4,
    "i64": 8,
    "ui64": 8,
    "index": 8,
}


@dataclass(frozen=True)
class ResourceUsage:
    """One compiled design's array occupancy. Field meanings are in ``schema``."""

    compute_tiles_used: int
    aie_tiles_with_buffers: int
    aie_tile_allocated_bytes: int
    aie_tile_memory_utilization: float
    aie_tile_memory_utilization_min: float | None
    aie_tile_memory_utilization_max: float | None
    aie_tile_memory_utilization_mean: float | None
    aie_tile_memory_utilization_median: float | None
    mem_tiles_with_buffers: int
    mem_tile_allocated_bytes: int
    mem_tile_memory_utilization: float
    mem_tile_memory_utilization_min: float | None
    mem_tile_memory_utilization_max: float | None
    mem_tile_memory_utilization_mean: float | None
    mem_tile_memory_utilization_median: float | None
    shim_tiles_with_s2mm: int
    shim_s2mm_channels_used: int
    shim_s2mm_channel_utilization: float
    shim_tiles_with_mm2s: int
    shim_mm2s_channels_used: int
    shim_mm2s_channel_utilization: float
    shim_tiles_with_dma: int
    shim_dma_channels_used: int
    shim_dma_channel_utilization: float
    mem_tiles_with_dma: int | None
    mem_dma_channels_used: int | None
    mem_dma_channel_utilization: float | None
    mem_dma_channel_note: str | None
    compute_tiles_with_dma: int | None
    compute_dma_channels_used: int | None
    compute_dma_channel_utilization: float | None
    compute_dma_channel_note: str | None
    core_to_core_flows: int
    total_flows: int
    unsizeable_buffers: int

    def as_columns(self) -> dict[str, object]:
        """The schema columns this describes. ``unsizeable_buffers`` is NOT one.

        It is diagnostic -- it says how much of the artifact the parse could not
        account for -- and it belongs in the CLI's report rather than in a
        comparison surface, where a reader would have to know to subtract it.
        """
        columns = asdict(self)
        columns.pop("unsizeable_buffers")
        return columns


def memref_bytes(memref_spec: str) -> int | None:
    """Bytes for ``memref<...>``'s inside, or ``None`` if it cannot be sized.

    ``None`` rather than 0 on purpose: the caller counts these, and a 0 would
    join the sum indistinguishably from a genuinely empty buffer.
    """
    spec = memref_spec.split(",", 1)[0].strip()
    parts = [part.strip() for part in spec.split("x") if part.strip()]
    if len(parts) < 2:
        return None
    itemsize = _DTYPE_BYTES.get(parts[-1])
    if itemsize is None:
        return None
    count = 1
    for dim in parts[:-1]:
        if not dim.isdigit():
            return None
        count *= int(dim)
    return count * itemsize


def _fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _spread(
    bytes_by_tile: dict[str, int], capacity: int
) -> tuple[float | None, float | None, float | None, float | None]:
    """(min, max, mean, median) per-tile utilization; all None when empty."""
    if not bytes_by_tile:
        return (None, None, None, None)
    values = [n / float(capacity) for n in bytes_by_tile.values()]
    return (
        min(values),
        max(values),
        statistics.fmean(values),
        statistics.median(values),
    )


def _dma_summary(
    channels_by_tile: dict[str, set[int]], *, per_tile: int, absent_note: str
) -> tuple[int | None, int | None, float | None, str | None]:
    """Channel occupancy, or Nones plus a note when the artifact states none."""
    if not channels_by_tile:
        return (None, None, None, absent_note)
    tiles = len(channels_by_tile)
    channels = sum(len(c) for c in channels_by_tile.values())
    return (tiles, channels, _fraction(channels, tiles * per_tile), None)


def _tile_rows(text: str) -> dict[str, int]:
    """SSA name -> tile row. Row 0 is shim, 1 is a memory tile, >= 2 is compute."""
    return {name: int(row) for name, _col, row in _TILE_RE.findall(text)}


def count_core_to_core_flows(text: str) -> tuple[int, int]:
    """(core->core flows, all flows) using ``norm_tail_structure``'s definition.

    An endpoint whose tile was never declared is not a compute tile for this
    purpose, and is not guessed at: it cannot be classified, so it cannot be
    counted as an on-chip stage edge.
    """
    rows = _tile_rows(text)
    flows = _FLOW_RE.findall(text)
    core_to_core = sum(
        1 for src, dst in flows if rows.get(src, -1) >= 2 and rows.get(dst, -1) >= 2
    )
    return core_to_core, len(flows)


def parse_routed_design(path: str | Path) -> ResourceUsage:
    """Parse one routed AIE artifact. See ``aircc_artifacts`` for which file."""
    text = Path(path).read_text(encoding="utf-8")

    compute_tiles = {m.group(1) for m in _CORE_RE.finditer(text)}
    aie_bytes: dict[str, int] = {}
    mem_bytes: dict[str, int] = {}
    unsizeable = 0
    for match in _BUFFER_RE.finditer(text):
        tile = match.group(1)
        size = memref_bytes(match.group(2))
        if size is None:
            unsizeable += 1
            continue
        target = mem_bytes if tile.startswith("%mem_tile_") else aie_bytes
        target[tile] = target.get(tile, 0) + size

    shim_s2mm: dict[str, set[int]] = {}
    shim_mm2s: dict[str, set[int]] = {}
    # Keyed by (direction, channel), NOT by channel -- see the module docstring's
    # note on the one departure from iron's arithmetic.
    shim_any: dict[str, set[tuple[str, int]]] = {}
    mem_dma: dict[str, set[int]] = {}
    compute_dma: dict[str, set[int]] = {}
    for match in _DMA_ALLOCATION_RE.finditer(text):
        tile = match.group("tile")
        channel = int(match.group("channel"))
        direction = match.group("direction")
        if tile.startswith("%shim_noc_tile_"):
            shim_any.setdefault(tile, set()).add((direction, channel))
            side = shim_s2mm if direction == "S2MM" else shim_mm2s
            side.setdefault(tile, set()).add(channel)
        elif tile.startswith("%mem_tile_"):
            mem_dma.setdefault(tile, set()).add(channel)
        else:
            compute_dma.setdefault(tile, set()).add(channel)

    aie_total = sum(aie_bytes.values())
    mem_total = sum(mem_bytes.values())
    aie_spread = _spread(aie_bytes, aircc_artifacts.AIE_TILE_LOCAL_MEMORY_BYTES)
    mem_spread = _spread(mem_bytes, aircc_artifacts.MEM_TILE_LOCAL_MEMORY_BYTES)

    s2mm_used = sum(len(c) for c in shim_s2mm.values())
    mm2s_used = sum(len(c) for c in shim_mm2s.values())
    shim_used = sum(len(c) for c in shim_any.values())
    per_direction = aircc_artifacts.SHIM_DMA_CHANNELS_PER_DIRECTION

    mem_dma_summary = _dma_summary(
        mem_dma,
        per_tile=MEM_TILE_DMA_CHANNELS_TOTAL,
        absent_note="the routed design declares no memory-tile dma_allocation; "
        "absent is not zero traffic",
    )
    compute_dma_summary = _dma_summary(
        compute_dma,
        per_tile=COMPUTE_TILE_DMA_CHANNELS_TOTAL,
        absent_note="the routed design declares no compute-tile dma_allocation; "
        "absent is not zero traffic",
    )
    core_to_core, total_flows = count_core_to_core_flows(text)

    return ResourceUsage(
        compute_tiles_used=len(compute_tiles),
        aie_tiles_with_buffers=len(aie_bytes),
        aie_tile_allocated_bytes=aie_total,
        aie_tile_memory_utilization=_fraction(
            aie_total, len(aie_bytes) * aircc_artifacts.AIE_TILE_LOCAL_MEMORY_BYTES
        ),
        aie_tile_memory_utilization_min=aie_spread[0],
        aie_tile_memory_utilization_max=aie_spread[1],
        aie_tile_memory_utilization_mean=aie_spread[2],
        aie_tile_memory_utilization_median=aie_spread[3],
        mem_tiles_with_buffers=len(mem_bytes),
        mem_tile_allocated_bytes=mem_total,
        mem_tile_memory_utilization=_fraction(
            mem_total, len(mem_bytes) * aircc_artifacts.MEM_TILE_LOCAL_MEMORY_BYTES
        ),
        mem_tile_memory_utilization_min=mem_spread[0],
        mem_tile_memory_utilization_max=mem_spread[1],
        mem_tile_memory_utilization_mean=mem_spread[2],
        mem_tile_memory_utilization_median=mem_spread[3],
        shim_tiles_with_s2mm=len(shim_s2mm),
        shim_s2mm_channels_used=s2mm_used,
        shim_s2mm_channel_utilization=_fraction(
            s2mm_used, len(shim_s2mm) * per_direction
        ),
        shim_tiles_with_mm2s=len(shim_mm2s),
        shim_mm2s_channels_used=mm2s_used,
        shim_mm2s_channel_utilization=_fraction(
            mm2s_used, len(shim_mm2s) * per_direction
        ),
        shim_tiles_with_dma=len(shim_any),
        shim_dma_channels_used=shim_used,
        shim_dma_channel_utilization=_fraction(
            shim_used, len(shim_any) * SHIM_DMA_CHANNELS_TOTAL
        ),
        mem_tiles_with_dma=mem_dma_summary[0],
        mem_dma_channels_used=mem_dma_summary[1],
        mem_dma_channel_utilization=mem_dma_summary[2],
        mem_dma_channel_note=mem_dma_summary[3],
        compute_tiles_with_dma=compute_dma_summary[0],
        compute_dma_channels_used=compute_dma_summary[1],
        compute_dma_channel_utilization=compute_dma_summary[2],
        compute_dma_channel_note=compute_dma_summary[3],
        core_to_core_flows=core_to_core,
        total_flows=total_flows,
        unsizeable_buffers=unsizeable,
    )


def role_style(usage: ResourceUsage) -> str:
    """Doc 03's AIE role style for this design, from its core->core flow count.

    Only the two states the flow count can decide between. ``single-function``
    is a property of a MODE over a run -- "the array only ever runs one kernel
    family" -- and no single artifact can show it, so this deliberately never
    returns it rather than guessing from one design.
    """
    return "space-multiplexed" if usage.core_to_core_flows else "time-multiplexed"


def resource_row(
    usage: ResourceUsage | None,
    *,
    artifact_name: str,
    routed_design_path: str,
    study_id: str = "adhoc",
    study_case_id: str | None = None,
    study_case_label: str | None = None,
    execution_mode: str | None = None,
    failure_message: str = "",
) -> dict:
    """A schema ``resource`` row. ``usage=None`` writes a complete failed row.

    Same contract as a failed measurement: the row exists, every metric is
    ``None``, and the reason is in ``failure_message``. A parse that wrote
    nothing would make "the compile failed" and "nobody compiled it"
    indistinguishable in the results tree.
    """
    row = schema.empty_row("resource")
    row["study_id"] = study_id
    row["study_case_id"] = study_case_id or artifact_name
    row["study_case_label"] = study_case_label or artifact_name
    row["execution_mode"] = execution_mode
    row["artifact_name"] = artifact_name
    row["routed_design_path"] = routed_design_path
    if usage is None:
        row["run_status"] = "failed"
        row["failure_message"] = failure_message[:300]
        return row
    row.update(usage.as_columns())
    row["run_status"] = "passed"
    row["failure_message"] = ""
    return row


def rows_for_projects(
    project_dirs: list[str],
    *,
    study_id: str = "adhoc",
    execution_mode: str | None = None,
) -> list[dict]:
    """One row per project directory, named by the directory's parent.

    Named by the PARENT because every aircc project directory is called
    ``air_project``; using the leaf would give every row the same identity and
    a results CSV nothing could be joined on.
    """
    rows = []
    for project_dir in project_dirs:
        path = Path(project_dir)
        name = path.parent.name or path.name
        try:
            design = aircc_artifacts.routed_design(path)
            usage = parse_routed_design(design)
        except Exception as e:
            rows.append(
                resource_row(
                    None,
                    artifact_name=name,
                    routed_design_path=str(path / aircc_artifacts.ROUTED_DESIGN_NAME),
                    study_id=study_id,
                    execution_mode=execution_mode,
                    failure_message=f"{type(e).__name__}: {e}".splitlines()[0],
                )
            )
            continue
        rows.append(
            resource_row(
                usage,
                artifact_name=name,
                routed_design_path=str(design),
                study_id=study_id,
                execution_mode=execution_mode,
            )
        )
        print(
            f"[resource-usage] {name}: {usage.compute_tiles_used} cores, "
            f"L1 {usage.aie_tile_allocated_bytes:,} B over "
            f"{usage.aie_tiles_with_buffers} tiles "
            f"({usage.aie_tile_memory_utilization:.1%}), "
            f"L2 {usage.mem_tile_allocated_bytes:,} B over "
            f"{usage.mem_tiles_with_buffers} tiles "
            f"({usage.mem_tile_memory_utilization:.1%}), "
            f"shim {usage.shim_dma_channels_used}ch over "
            f"{usage.shim_tiles_with_dma} tiles, "
            f"flows {usage.core_to_core_flows}/{usage.total_flows} core->core "
            f"[{role_style(usage)}]"
        )
        if usage.unsizeable_buffers:
            print(
                f"[resource-usage]   {usage.unsizeable_buffers} buffer(s) could "
                "not be sized and are excluded from the byte totals"
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "project_dirs",
        nargs="+",
        help="aircc project directories (the ones holding "
        f"{aircc_artifacts.ROUTED_DESIGN_NAME})",
    )
    ap.add_argument("--out", required=True, help="resource CSV to write")
    ap.add_argument("--study-id", default="adhoc")
    ap.add_argument(
        "--execution-mode",
        default=None,
        help="the CSV execution_mode these artifacts belong to; omit for "
        "standalone operator compiles, which belong to no mode",
    )
    args = ap.parse_args(argv)

    rows = rows_for_projects(
        args.project_dirs,
        study_id=args.study_id,
        execution_mode=args.execution_mode,
    )
    results_io.write_rows(args.out, rows, table="resource")
    passed = sum(1 for r in rows if r["run_status"] == "passed")
    print(f"[resource-usage] wrote {args.out}: {passed}/{len(rows)} parsed")
    for row in rows:
        if row["run_status"] != "passed":
            print(
                f"[resource-usage] FAILED {row['artifact_name']}: "
                f"{row['failure_message']}"
            )
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
