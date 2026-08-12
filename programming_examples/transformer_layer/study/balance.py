# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The balance instrument: per-column channel demand, priced as a slope, off one artifact.

    python3 study/balance.py <air_project-dir> [--duration-ns N] [--json out.json]

CONTRACT
    ``parse_transfers(text)`` reads every shim-facing ``air.channel.put``/
    ``get`` out of a routed AIE artifact with its BD shape and its dependence
    level; ``parse_allocations(text)`` maps each ``aie.shim_dma_allocation``
    symbol to the ``(column, direction, channel)`` it landed on;
    ``demand_matrix`` joins them into a ``[step x port]`` matrix per column;
    ``back_solve`` says what bandwidth a stall-free run would have required;
    ``balance_ports`` prices overflow as a SLOPE with demand printed beside
    budget; ``bottleneck`` takes the max over per-resource isolated times and
    NAMES the argmax; ``stage_gap`` is iron's full-vs-isolated-stage metric
    with both of its defects made structurally impossible.

    NO NPU, NO DISPATCH, NO SIMULATOR. Everything here is a property of a
    compiled artifact plus a table of measured costs, which is what makes it
    affordable to run over a search space. Doc 44: the static back-solve "needs
    no simulator and no hardware run, which is what makes a search affordable."

WHERE EACH PART CAME FROM
    Doc 44 §The instrument specifies five parts, each taken from a different
    framework, and this module is them:

    1. **A ``[step x port]`` demand matrix per column** -- SCALE-Sim (doc 42).
       Its demand ACCOUNTING is borrowed; its SERVICE model is not. Doc 44:
       SCALE-Sim's global-lockstep stalling and non-contending ports are
       "exactly wrong for 32 independently scheduled cores sharing two channels
       per column."
    2. **A static back-solve of the required bandwidth** -- SCALE-Sim's
       ``InterfaceBandwidth: CALC``: assume stall-free, divide traffic by the
       duration that assumption implies.
    3. **Overflow priced as a slope**, ``min(1, budget/demand)``, never a
       cliff, with demand printed beside budget -- Timeloop (doc 41) plus
       MAESTRO (doc 43). See the next section; this is the one place doc 44
       corrects an earlier proposal of this project's.
    4. **Latency = max over per-resource isolated times, argmax names the
       resource** -- Timeloop's bottleneck model (doc 41).
    5. Costs come from ``balance_ert`` -- Accelergy's ERT pattern (doc 40)
       holding measured ns and counted bytes.

WHY THE BUDGET IS NOT A LEGALITY PREDICATE
    Earlier in this study, reasoning from LLMCompass, this project proposed
    making the per-column MM2S budget a legality predicate -- exclude any
    mapping demanding more than 2 per column. Doc 44 corrects it, and the
    correction is the reason this module has no ``is_legal``:

    Exceeding the budget does not break correctness. AIR packet-multiplexes
    onto one queue and the design runs slower, so a legal-but-degraded point
    must be MODELLED as degraded, not filtered out. Worse, filtering would have
    hidden the precise failure mode we are trying to see -- silent multiplexing
    would vanish from the search rather than show up as cost. Capacity is a
    cliff; bandwidth is a slope.

    So ``balance_ports`` returns a record for EVERY column including the
    over-budget ones, with ``demand`` beside ``budget`` (MAESTRO's warn-tier
    shape) and the shortfall also charged into runtime (MAESTRO charges both
    channels, not either).

THE DEMAND COUNT MUST INCLUDE PACKET FLOWS, AND MEASURABLY DOES
    Counting per-column ingress as shim->core ``aie.flow`` ops reads **zero**
    on a design that is over budget, because AIR's reaction to exceeding the
    budget is to emit ``aie.packet_flow`` instead. Measured on the shipped
    ``addnorm`` artifact: 8 ``aie.flow`` (all core->shim output drains, so
    shim->core is 0) against 17 ``aie.packet_flow`` whose sources are column 0
    three times and columns 1-7 twice each -- the exact 3-streams-on-column-0
    doc 23 records. A demand count that reads 0 exactly when the design is over
    budget is a check that cannot fail, and this project found six of those in
    one day.

    This module therefore counts DMA ALLOCATIONS, which exist in both forms:
    three ``aie.shim_dma_allocation`` symbols on ``(%shim_noc_tile_0_0, MM2S,
    0)`` is demand 3 on one physical channel, and the multiplexing depth is
    reported beside the demand. ``test_balance.py`` pins the failing direction.

TWO DEMAND NUMBERS, AND THEY ANSWER DIFFERENT QUESTIONS
    - ``static_demand`` -- distinct logical channels allocated to a column and
      direction over the WHOLE design. This is the number doc 23's rule is
      about: "the budget is per COLUMN across the whole segment", and it is
      what AIR actually allocated, since a shim allocation is static.
    - ``peak_concurrent_demand`` -- the max over the step axis of the matrix.
      Always <= the static one. It is what SCALE-Sim's ``[cycle x port]``
      reading gives, and it says WHERE in the program the pressure is.

    The budget check uses the static number. Reporting only the concurrent one
    would let a design look compliant because its two streams never overlap,
    when AIR has already committed both to one physical channel regardless.

``step`` IS NOT A HARDWARE CYCLE, AND NOTHING HERE PRETENDS IT IS
    SCALE-Sim's rows are simulated cycles. Ours are ASAP levels over the
    launch-level async dependence graph: level 0 is anything with no token
    dependence, and a transfer is one past the max of its dependences. Two
    transfers at the same level MAY overlap; two at different levels are
    ordered. That is strictly less information than a cycle count and it is all
    a static read of the artifact can support. Anything that needs real cycles
    needs a simulator, and doc 44's whole argument for this design is that not
    needing one is what makes a search affordable.

FOOTGUNS
    - **An unattributed transfer is reported, never dropped.** A put whose
      channel index is not a constant, or whose ``metadataArray`` names a
      symbol with no allocation, lands in ``unattributed`` with a reason. A
      demand total that silently omitted it would understate exactly the case
      that is hardest to lower.
    - **An unknown trip count is ``None``, never 1.** A put inside a loop whose
      bounds are not constants has unknown byte volume; assuming one trip would
      understate the traffic by the trip count.
    - **``bottleneck`` over an incomplete set says so.** A resource with no
      priced time lands in ``unpriced`` and ``is_complete`` reads False. This
      tree has NO measured AIR-native shim bandwidth (doc 33 deferred the
      memcpy operator), so a port resource is unpriced unless a caller supplies
      a byte rate explicitly -- and one supplied that way is ``modelled``, not
      measured, and must be labelled so wherever it is reported.
    - **The artifact is per-segment-name, not per-run.** Same footgun as
      ``resource_usage``: compiling twice into one project directory overwrites
      it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import aircc_artifacts  # noqa: E402
import balance_ert  # noqa: E402
import resource_usage  # noqa: E402

#: Channels of one direction a single column's shim tile has. Imported, not
#: restated -- ``aircc_artifacts`` is where the device constants live, and doc
#: 23's rule ("keep every column's L3-facing streams at two or fewer") is a
#: statement about this number.
PER_COLUMN_BUDGET = aircc_artifacts.SHIM_DMA_CHANNELS_PER_DIRECTION

DIRECTIONS = ("MM2S", "S2MM")

_ALLOCATION_RE = re.compile(
    r"aie\.(?:\w*)shim_dma_allocation\s+@(?P<symbol>\w+)\(\s*"
    r"%shim_noc_tile_(?P<col>\d+)_(?P<row>\d+)\s*,\s*"
    r"(?P<direction>S2MM|MM2S)\s*,\s*(?P<channel>\d+)\s*\)"
)
_CONSTANT_RE = re.compile(r"%(\S+) = arith\.constant (-?\d+) : index")
#: MULTILINE is load-bearing: without it ``$`` anchors to the end of the whole
#: artifact, so exactly one declaration in the file could ever match and every
#: channel's shape came back unknown. That made every multi-column put
#: unattributable while the report still printed a clean compliant table for the
#: one channel it did resolve -- a wrong answer that looked like a passing one.
_CHANNEL_DECL_RE = re.compile(
    r"air\.channel @(?P<name>\w+)\s*\[(?P<shape>[^\]]*)\](?P<attrs>.*)$",
    re.MULTILINE,
)
_CHANNEL_OP_RE = re.compile(
    r"air\.channel\.(?P<kind>put|get)\s+async\s*"
    r"(?:\[(?P<deps>[^\]]*)\])?\s*"
    r"@(?P<channel>\w+)\[(?P<indices>[^\]]*)\]"
)
_OPERAND_RE = re.compile(
    r"\((?P<operand>%[\w.]+)\[(?P<offsets>[^\]]*)\]\s*"
    r"\[(?P<sizes>[^\]]*)\]\s*\[(?P<strides>[^\]]*)\]\)"
)
_METADATA_RE = re.compile(r"metadataArray\s*=\s*\[(?P<body>[^\]]*)\]")
_METADATA_ENTRY_RE = re.compile(r'base\s*=\s*"(?P<base>\w+)",\s*index\s*=\s*(?P<index>\d+)')
_RESULT_RE = re.compile(r"^\s*%(?P<result>[\w.]+) = ")
_ASYNC_DEPS_RE = re.compile(r"\basync\s*\[(?P<deps>[^\]]*)\]")
_SCF_FOR_RE = re.compile(
    r"scf\.for %[\w.]+ = %(?P<lb>[\w.]+) to %(?P<ub>[\w.]+) step %(?P<step>[\w.]+)"
)
_AFFINE_FOR_RE = re.compile(r"affine\.for %[\w.]+ = (?P<lb>-?\d+) to (?P<ub>-?\d+)")
_ITER_ARGS_RE = re.compile(r"iter_args\((?P<body>[^)]*)\)")
#: An ``air.launch`` iteration space repeats its whole body, so it multiplies the
#: traffic of every shim transfer inside it exactly as a loop does. Missing this
#: understated the matmul artifact's A-operand traffic by 8x while the demand
#: table read correct, which is the shape of error a byte total hides best.
_LAUNCH_SIZES_RE = re.compile(r"air\.launch\s+(?:async\s+)?\([^)]*\)\s+in\s+\((?P<body>[^)]*)\)")
_MEMREF_TYPE_RE = re.compile(r":\s*\((memref<[^>]+>)\)\s*$")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Port:
    """One physical shim DMA channel."""

    column: int
    direction: str
    channel: int


@dataclass(frozen=True)
class Transfer:
    """One shim-facing channel op, with the BD shape that makes it an ERT action."""

    kind: str
    channel: str
    allocation: str | None
    n_words: int | None
    n_dims: int
    sizes: tuple[int, ...]
    strides: tuple[int, ...]
    element_bytes: int
    trip_count: int | None
    step: int
    packet: bool
    line: int
    reason: str | None = None

    @property
    def bytes(self) -> int | None:
        """Bytes this op moves in total, or ``None`` if anything about it is unknown.

        ``None`` rather than a partial number: an unknown trip count times a
        known descriptor is not a byte count, and a sum that quietly absorbed
        it would understate the traffic by exactly the factor nobody can see.
        """
        if self.n_words is None or self.trip_count is None:
            return None
        return self.n_words * self.element_bytes * self.trip_count


def routed_design_path(project_dir: str | Path) -> Path:
    """The routed artifact inside an aircc project directory."""
    return aircc_artifacts.routed_design(project_dir)


def parse_allocations(text: str) -> dict[str, Port]:
    """``shim_dma_allocation`` symbol -> the physical port it landed on."""
    out: dict[str, Port] = {}
    for match in _ALLOCATION_RE.finditer(text):
        out[match.group("symbol")] = Port(
            column=int(match.group("col")),
            direction=match.group("direction"),
            channel=int(match.group("channel")),
        )
    return out


def _int_list(spec: str) -> tuple[int, ...] | None:
    """``"8, 768"`` -> ``(8, 768)``; ``None`` if any entry is not a literal int."""
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    values = []
    for part in parts:
        try:
            values.append(int(part))
        except ValueError:
            return None
    return tuple(values)


def _element_bytes(memref_type: str) -> int | None:
    """Bytes per element of ``memref<...>``.

    Delegates to ``resource_usage.memref_bytes`` on a one-element memref rather
    than keeping a second element-size table: one definition of each constant is
    the rule that file's docstring already follows.
    """
    inner = memref_type[len("memref<") : -1]
    spec = inner.split(",", 1)[0].strip()
    parts = [p.strip() for p in spec.split("x") if p.strip()]
    if not parts:
        return None
    return resource_usage.memref_bytes(f"1x{parts[-1]}")


def _memref_shape(memref_type: str) -> tuple[int, ...] | None:
    inner = memref_type[len("memref<") : -1]
    spec = inner.split(",", 1)[0].strip()
    parts = [p.strip() for p in spec.split("x") if p.strip()]
    dims = []
    for part in parts[:-1]:
        if not part.lstrip("-").isdigit():
            return None
        dims.append(int(part))
    return tuple(dims)


def _linear_index(indices: tuple[int, ...], shape: tuple[int, ...]) -> int | None:
    """Row-major linearization of a channel index against the channel's shape."""
    if len(indices) != len(shape):
        return None
    linear = 0
    for value, extent in zip(indices, shape):
        if not 0 <= value < max(extent, 1):
            return None
        linear = linear * extent + value
    return linear


def parse_transfers(text: str) -> tuple[Transfer, ...]:
    """Every shim-facing channel op in a routed design, in program order.

    "Shim-facing" is decided by the presence of ``metadataArray`` on the op:
    that attribute is written by the lowering precisely for the ops that bind
    to a shim allocation, so herd-level and segment-level channel ops -- which
    move L1/L2 data and consume no shim channel -- are excluded by having none,
    rather than by a guess about their operand's memory space.
    """
    constants: dict[str, int] = {
        name: int(value) for name, value in _CONSTANT_RE.findall(text)
    }
    channel_shapes: dict[str, tuple[int, ...]] = {}
    channel_packet: dict[str, bool] = {}
    for match in _CHANNEL_DECL_RE.finditer(text):
        shape = _int_list(match.group("shape")) or ()
        channel_shapes[match.group("name")] = shape
        channel_packet[match.group("name")] = "npu_dma_packet" in (
            match.group("attrs") or ""
        )

    levels: dict[str, int] = {}
    trip_stack: list[int | None] = []
    transfers: list[Transfer] = []

    for lineno, line in enumerate(text.splitlines(), start=1):
        # -- dependence level of whatever this line defines ----------------
        result = _RESULT_RE.match(line)
        deps_match = _ASYNC_DEPS_RE.search(line)
        level = 0
        if deps_match:
            dep_names = [
                d.strip().lstrip("%")
                for d in deps_match.group("deps").split(",")
                if d.strip()
            ]
            if dep_names:
                level = 1 + max(levels.get(d, 0) for d in dep_names)
        if result:
            levels[result.group("result")] = level

        # A loop's iter_args carry a token in; the body's level is the init's.
        iter_args = _ITER_ARGS_RE.search(line)
        if iter_args:
            for pair in iter_args.group("body").split(","):
                if "=" not in pair:
                    continue
                name, init = (p.strip().lstrip("%") for p in pair.split("=", 1))
                levels[name] = levels.get(init, level)

        op = _CHANNEL_OP_RE.search(line)
        if op and _METADATA_RE.search(line):
            transfers.append(
                _transfer_from_line(
                    line,
                    op,
                    lineno=lineno,
                    level=level,
                    constants=constants,
                    channel_shapes=channel_shapes,
                    channel_packet=channel_packet,
                    trips=_trip_product(trip_stack),
                )
            )

        # -- block nesting, for the trip count -----------------------------
        opens = line.count("{") - line.count("}")
        if opens > 0:
            trip = _loop_trips(line, constants)
            for _ in range(opens - 1):
                trip_stack.append(1)
            trip_stack.append(trip)
        elif opens < 0:
            for _ in range(-opens):
                if trip_stack:
                    trip_stack.pop()

    return tuple(transfers)


def _trip_product(stack: list[int | None]) -> int | None:
    """Trips of the enclosing loop nest, or ``None`` if any level is unknown."""
    total = 1
    for trip in stack:
        if trip is None:
            return None
        total *= trip
    return total


def _loop_trips(line: str, constants: dict[str, int]) -> int | None:
    """Trip count of the loop this line opens; 1 for a non-loop block.

    ``None`` means "this IS a loop and its bounds are not constants" -- which
    the caller propagates as an unknown byte volume rather than assuming one
    trip.
    """
    match = _SCF_FOR_RE.search(line)
    if match:
        try:
            lb = constants[match.group("lb")]
            ub = constants[match.group("ub")]
            step = constants[match.group("step")]
        except KeyError:
            return None
        if step == 0:
            return None
        return max(0, math.ceil((ub - lb) / step))
    match = _AFFINE_FOR_RE.search(line)
    if match:
        return max(0, int(match.group("ub")) - int(match.group("lb")))
    match = _LAUNCH_SIZES_RE.search(line)
    if match:
        total = 1
        for pair in match.group("body").split(","):
            if "=" not in pair:
                continue
            value = pair.split("=", 1)[1].strip()
            if value.startswith("%"):
                if value[1:] not in constants:
                    return None
                total *= constants[value[1:]]
            elif value.lstrip("-").isdigit():
                total *= int(value)
            else:
                return None
        return total
    if "scf.for" in line or "affine.for" in line or "air.launch" in line:
        return None
    return 1


def _transfer_from_line(
    line: str,
    op: re.Match,
    *,
    lineno: int,
    level: int,
    constants: dict[str, int],
    channel_shapes: dict[str, tuple[int, ...]],
    channel_packet: dict[str, bool],
    trips: int | None,
) -> Transfer:
    channel = op.group("channel")
    reason: str | None = None

    type_match = _MEMREF_TYPE_RE.search(line.rstrip())
    memref_type = type_match.group(1) if type_match else None
    element_bytes = _element_bytes(memref_type) if memref_type else None

    operand = _OPERAND_RE.search(line)
    sizes: tuple[int, ...] = ()
    strides: tuple[int, ...] = ()
    n_words: int | None = None
    if operand:
        parsed_sizes = _int_list(operand.group("sizes"))
        parsed_strides = _int_list(operand.group("strides"))
        if parsed_sizes is None or parsed_strides is None:
            reason = "non-constant sizes or strides in the access pattern"
        elif parsed_sizes:
            sizes, strides = parsed_sizes, parsed_strides
            n_words = 1
            for extent in sizes:
                n_words *= extent
        else:
            # `(%arg[] [] [])` is AIR's whole-memref form: the access pattern is
            # the memref's own shape, contiguous.
            shape = _memref_shape(memref_type) if memref_type else None
            if shape is None:
                reason = "whole-memref transfer whose memref could not be sized"
            else:
                sizes = shape
                stride, built = 1, []
                for extent in reversed(shape):
                    built.append(stride)
                    stride *= extent
                strides = tuple(reversed(built))
                n_words = 1
                for extent in sizes:
                    n_words *= extent
    else:
        reason = "no access pattern on the channel op"

    if element_bytes is None:
        element_bytes = 0
        n_words = None
        reason = reason or "element type not sizeable"

    # -- which shim allocation this op binds to --------------------------
    allocation: str | None = None
    entries = []
    metadata = _METADATA_RE.search(line)
    if metadata:
        entries = [
            (m.group("base"), int(m.group("index")))
            for m in _METADATA_ENTRY_RE.finditer(metadata.group("body"))
        ]
    if len(entries) == 1:
        allocation = entries[0][0]
    elif entries:
        raw = [i.strip() for i in op.group("indices").split(",") if i.strip()]
        values = []
        for token in raw:
            if token.startswith("%"):
                name = token[1:]
                if name not in constants:
                    values = None
                    break
                values.append(constants[name])
            elif token.lstrip("-").isdigit():
                values.append(int(token))
            else:
                values = None
                break
        shape = channel_shapes.get(channel)
        linear = (
            _linear_index(tuple(values), shape)
            if values is not None and shape
            else None
        )
        if linear is None:
            reason = reason or (
                "channel index is not a constant, so which of "
                f"{len(entries)} shim allocations this binds to is undecidable"
            )
        else:
            by_index = {index: base for base, index in entries}
            allocation = by_index.get(linear)
            if allocation is None:
                reason = reason or (
                    f"channel index {linear} has no metadataArray entry"
                )

    return Transfer(
        kind=op.group("kind"),
        channel=channel,
        allocation=allocation,
        n_words=n_words,
        n_dims=len(sizes),
        sizes=sizes,
        strides=strides,
        element_bytes=element_bytes,
        trip_count=trips,
        step=level,
        packet=channel_packet.get(channel, False),
        line=lineno,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# 1. The [step x port] demand matrix
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PortDemand:
    """One ``(column, direction)`` pair's demand against its budget."""

    column: int
    direction: str
    budget: int
    static_demand: int
    peak_concurrent_demand: int
    channels: tuple[str, ...]
    multiplex_depth: tuple[tuple[int, int], ...]
    bytes: int | None
    transfers: int

    @property
    def over_budget(self) -> bool:
        return self.static_demand > self.budget

    @property
    def max_multiplex_depth(self) -> int:
        return max((d for _c, d in self.multiplex_depth), default=0)


@dataclass(frozen=True)
class DemandMatrix:
    """The matrix plus everything it could not attribute."""

    ports: tuple[PortDemand, ...]
    cells: tuple[tuple[int, int, str, tuple[str, ...]], ...]
    steps: tuple[int, ...]
    unattributed: tuple[Transfer, ...]
    unknown_trip_counts: int

    def port(self, column: int, direction: str) -> PortDemand | None:
        for demand in self.ports:
            if demand.column == column and demand.direction == direction:
                return demand
        return None

    def cell(self, step: int, column: int, direction: str) -> tuple[str, ...]:
        """The distinct logical channels demanding ``(column, direction)`` at ``step``."""
        for s, c, d, names in self.cells:
            if (s, c, d) == (step, column, direction):
                return names
        return ()

    @property
    def is_complete(self) -> bool:
        """True when every transfer was attributed and every trip count known."""
        return not self.unattributed and not self.unknown_trip_counts


def demand_matrix(
    transfers: tuple[Transfer, ...],
    allocations: dict[str, Port],
    *,
    budget: int = PER_COLUMN_BUDGET,
) -> DemandMatrix:
    """Join transfers onto ports and build the ``[step x port]`` matrix.

    Demand is counted in DISTINCT LOGICAL CHANNELS, not in transfers: eight
    puts on ``air_channel_1_3`` are one stream contending for column 3, and
    counting the puts would report a column as eight times over budget for
    doing exactly what the rule permits.
    """
    unattributed: list[Transfer] = []
    unknown_trips = 0
    per_port_channels: dict[tuple[int, str], set[str]] = {}
    per_port_physical: dict[tuple[int, str], dict[int, set[str]]] = {}
    per_port_bytes: dict[tuple[int, str], int | None] = {}
    per_port_count: dict[tuple[int, str], int] = {}
    per_cell: dict[tuple[int, int, str], set[str]] = {}
    steps: set[int] = set()

    for transfer in transfers:
        port = allocations.get(transfer.allocation) if transfer.allocation else None
        if port is None:
            unattributed.append(transfer)
            continue
        if transfer.trip_count is None:
            unknown_trips += 1
        key = (port.column, port.direction)
        per_port_channels.setdefault(key, set()).add(transfer.allocation)
        per_port_physical.setdefault(key, {}).setdefault(port.channel, set()).add(
            transfer.allocation
        )
        per_port_count[key] = per_port_count.get(key, 0) + 1
        size = transfer.bytes
        if key not in per_port_bytes:
            per_port_bytes[key] = 0
        if size is None or per_port_bytes[key] is None:
            per_port_bytes[key] = None
        else:
            per_port_bytes[key] += size
        per_cell.setdefault(
            (transfer.step, port.column, port.direction), set()
        ).add(transfer.allocation)
        steps.add(transfer.step)

    ports = []
    for key in sorted(per_port_channels):
        column, direction = key
        concurrent = max(
            (
                len(names)
                for (step, c, d), names in per_cell.items()
                if (c, d) == key
            ),
            default=0,
        )
        ports.append(
            PortDemand(
                column=column,
                direction=direction,
                budget=budget,
                static_demand=len(per_port_channels[key]),
                peak_concurrent_demand=concurrent,
                channels=tuple(sorted(per_port_channels[key])),
                multiplex_depth=tuple(
                    sorted(
                        (channel, len(names))
                        for channel, names in per_port_physical[key].items()
                    )
                ),
                bytes=per_port_bytes[key],
                transfers=per_port_count[key],
            )
        )

    cells = tuple(
        sorted(
            (step, column, direction, tuple(sorted(names)))
            for (step, column, direction), names in per_cell.items()
        )
    )
    return DemandMatrix(
        ports=tuple(ports),
        cells=cells,
        steps=tuple(sorted(steps)),
        unattributed=tuple(unattributed),
        unknown_trip_counts=unknown_trips,
    )


# ---------------------------------------------------------------------------
# 2. The static back-solve -- SCALE-Sim's InterfaceBandwidth: CALC
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RequiredBandwidth:
    """One port's traffic divided by a duration. READ THE UNIT NOTE.

    The arithmetic is one division either way, but the reading depends on where
    ``duration_ns`` came from and the two are not the same claim:

    - a **hypothetical stall-free** duration gives the bandwidth that execution
      would REQUIRE -- SCALE-Sim's ``InterfaceBandwidth: CALC`` exactly;
    - a **measured achieved** latency (an ERT entry, which is what this tree
      actually has) gives the average rate the run SUSTAINED, and that is a
      LOWER BOUND on the requirement, since a stall-free run is no longer than
      the achieved one and a shorter duration needs a higher rate.

    ``back_solve`` does not know which it was handed, so it names neither in the
    field and the caller must say. ``render`` prints the distinction.
    """

    column: int
    direction: str
    bytes: int | None
    duration_ns: float
    bytes_per_ns: float | None

    @property
    def gigabytes_per_second(self) -> float | None:
        """1 byte/ns is 1 GB/s exactly, so this is a rename, not a conversion."""
        return self.bytes_per_ns


def back_solve(
    matrix: DemandMatrix, duration_ns: float
) -> tuple[RequiredBandwidth, ...]:
    """Per port, the bandwidth a stall-free run of ``duration_ns`` would have required.

    This is SCALE-Sim's ``InterfaceBandwidth: CALC`` (doc 42): rather than
    simulating against a supplied bandwidth, assume the execution does not
    stall and report what bandwidth that assumption implies. It is one
    division, it needs no simulator and no hardware run, and that is what makes
    evaluating a whole mapping space affordable.

    ``duration_ns`` is the caller's -- typically a MEASURED latency out of the
    ERT. Nothing here supplies one, because a duration invented here would turn
    the whole result into a model wearing a measurement's units.
    """
    if not duration_ns > 0:
        raise ValueError(
            f"duration_ns={duration_ns!r}: the back-solve divides by it, and a "
            "zero or negative duration would report an infinite requirement as "
            "though it were a finding"
        )
    return tuple(
        RequiredBandwidth(
            column=port.column,
            direction=port.direction,
            bytes=port.bytes,
            duration_ns=duration_ns,
            bytes_per_ns=None if port.bytes is None else port.bytes / duration_ns,
        )
        for port in matrix.ports
    )


# ---------------------------------------------------------------------------
# 3. Overflow priced as a SLOPE, with demand printed beside budget
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PortBalance:
    """One port's overflow, priced. Never a rejection -- see the module docstring."""

    column: int
    direction: str
    demand: int
    budget: int
    slowdown: float
    multiplexed: bool

    @property
    def over_budget(self) -> bool:
        return self.demand > self.budget

    @property
    def inflation(self) -> float:
        """The runtime multiplier this shortfall charges. ``1 / slowdown``."""
        return 1.0 / self.slowdown

    def warning(self) -> str | None:
        """MAESTRO's warn tier: demand printed beside budget, or ``None``."""
        if not self.over_budget:
            return None
        return (
            f"column {self.column} {self.direction}: demand {self.demand} against "
            f"budget {self.budget} -- slowdown {self.slowdown:.3f}, runtime "
            f"charged x{self.inflation:.3f}"
            + (
                "; AIR has packet-multiplexed these onto one queue"
                if self.multiplexed
                else ""
            )
        )


def balance_ports(matrix: DemandMatrix) -> tuple[PortBalance, ...]:
    """Price every port's demand as a slope. Returns a record for EVERY port.

    ``slowdown = min(1, budget / demand)`` -- Timeloop's, and doc 44 corrects
    this project's earlier proposal to make the same quantity a legality
    predicate. A compliant port gets a record too, with ``slowdown`` 1.0: a
    result set containing only the violations cannot be told apart from a
    result set produced by a check that did not run.
    """
    return tuple(
        PortBalance(
            column=port.column,
            direction=port.direction,
            demand=port.static_demand,
            budget=port.budget,
            slowdown=min(1.0, port.budget / port.static_demand)
            if port.static_demand
            else 1.0,
            multiplexed=port.max_multiplex_depth > 1,
        )
        for port in matrix.ports
    )


def charged_ns(base_ns: float, balances: tuple[PortBalance, ...]) -> float:
    """``base_ns`` with the worst port's shortfall charged into it.

    The WORST, not the product: the columns are parallel resources, so a layer
    waits on the slowest one rather than on all of them in series. MAESTRO
    charges the rate shortfall into runtime as well as warning about it, and
    this is that half.
    """
    return base_ns * max((b.inflation for b in balances), default=1.0)


# ---------------------------------------------------------------------------
# 4. Latency = max over per-resource isolated times, and the argmax NAMES it
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IsolatedTime:
    """One resource's time in isolation, with what is inside the measurement."""

    resource: str
    ns: float | None
    source: str = "absent"
    provenance: str = ""
    #: The stages this measurement CONTAINS. Empty for a pure resource. For a
    #: stage-truncated build it is load-bearing -- see ``stage_gap``.
    contains: tuple[str, ...] = ()
    #: L3 bytes this variant issues. Load-bearing for ``stage_gap``.
    l3_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.source not in balance_ert.COST_SOURCES:
            raise ValueError(
                f"source={self.source!r} is not one of "
                f"{list(balance_ert.COST_SOURCES)}"
            )
        if (self.ns is None) != (self.source == "absent"):
            raise ValueError(
                f"{self.resource}: ns={self.ns!r} with source={self.source!r}. A "
                "value is present exactly when its source is not 'absent'; an "
                "unpriced resource reading as a free one is how a bottleneck "
                "search names the wrong resource."
            )


@dataclass(frozen=True)
class Bottleneck:
    """Timeloop's bottleneck model: the max, and the name of what caused it."""

    resource: str
    ns: float
    ranked: tuple[tuple[str, float], ...]
    unpriced: tuple[str, ...]
    containment_checked: bool

    @property
    def is_complete(self) -> bool:
        """False when some resource had no priced time. Read this before the name."""
        return not self.unpriced


class PrefixComparison(ValueError):
    """One entry's ``contains`` is a strict superset of another's -- iron defect 1."""


def _reject_prefixes(entries: tuple[IsolatedTime, ...]) -> bool:
    """Refuse a comparison set mixing prefixes with true single stages.

    This is doc 38 §3.3 defect 1, made structurally impossible. iron's
    ``debug=7`` "isolated stage" ``addnorm1`` kept ``mha_debug=0``, so the whole
    MHA front-end computed inside it: the reported "max isolated stage" already
    contained another entry in the same comparison, and ``full - max`` was a
    max over a mixed set of prefixes and true stages. Returns whether the check
    was able to run at all -- with every ``contains`` empty it is vacuous, and a
    caller reporting a bottleneck must say so rather than imply a guard fired.
    """
    populated = [e for e in entries if e.contains]
    if len(populated) < 2:
        return bool(populated)
    for outer in populated:
        for inner in populated:
            if outer.resource == inner.resource:
                continue
            if set(inner.contains) < set(outer.contains):
                raise PrefixComparison(
                    f"{outer.resource} contains {sorted(outer.contains)}, which "
                    f"strictly contains {inner.resource}'s "
                    f"{sorted(inner.contains)}. These are a prefix and a stage, "
                    "not two stages: doc 38 §3.3 defect 1 is iron reporting "
                    "exactly this max as an isolated-stage bottleneck."
                )
    return True


def bottleneck(entries: tuple[IsolatedTime, ...]) -> Bottleneck:
    """``max`` over per-resource isolated times; the argmax names the resource.

    Timeloop's analytical model (doc 41), which doc 44 sizes at ~100 lines and
    calls "precisely the 'which stage is the bottleneck' answer iron
    approximated by hand with truncated binaries".

    A resource with no priced time is reported in ``unpriced`` and excluded
    from the max -- NOT treated as zero. This tree has no measured AIR-native
    shim byte rate (doc 33), so the shim ports are genuinely unpriced today and
    a caller must be able to see that the max was taken over an incomplete set.
    """
    if not entries:
        raise ValueError("bottleneck() over no resources: there is no argmax")
    checked = _reject_prefixes(entries)
    priced = [e for e in entries if e.ns is not None]
    unpriced = tuple(e.resource for e in entries if e.ns is None)
    if not priced:
        raise ValueError(
            f"every resource is unpriced ({list(unpriced)}); a max over nothing "
            "would name a resource on no evidence"
        )
    ranked = tuple(
        sorted(((e.resource, float(e.ns)) for e in priced), key=lambda r: -r[1])
    )
    return Bottleneck(
        resource=ranked[0][0],
        ns=ranked[0][1],
        ranked=ranked,
        unpriced=unpriced,
        containment_checked=checked,
    )


# ---------------------------------------------------------------------------
# iron's full-vs-isolated-stage metric, with both of its defects fixed
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageGap:
    """``full``, ``max_stage``, the gap and the ratio -- always to an artifact."""

    full_ns: float
    max_stage: str
    max_stage_ns: float
    gap_ns: float
    ratio: float
    l3_bytes: int
    stages: tuple[tuple[str, float], ...]

    def to_json(self) -> dict:
        return {
            "full_ns": self.full_ns,
            "max_stage": self.max_stage,
            "max_stage_ns": self.max_stage_ns,
            "gap_ns": self.gap_ns,
            "ratio": self.ratio,
            "l3_bytes": self.l3_bytes,
            "stages": [list(s) for s in self.stages],
        }


class ElidedTraffic(ValueError):
    """A stage variant issues different L3 traffic from ``full`` -- iron defect 2."""


def stage_gap(full: IsolatedTime, stages: tuple[IsolatedTime, ...]) -> StageGap:
    """iron's balance metric, ported with doc 38 §3.3's two defects made impossible.

    iron compiled one binary per stage with the dataflow graph preserved and
    only the arithmetic neutralized, timed each, and reported
    ``full - max(stage)`` as exposed serialization. Doc 38 found two defects
    that together inflate that gap, and this function refuses the inputs that
    carry them:

    - **Defect 1, the prefix.** ``debug=7`` ("addnorm1") kept the whole MHA
      front-end computing, so the max was taken over a set mixing prefixes with
      true single stages. Every entry here must declare ``contains``, and one
      strictly containing another is a ``PrefixComparison``.
    - **Defect 2, the elided weights.** ``addnorm1`` set both
      ``need_bup_weights`` and ``need_bdown_weights`` False, so ~9.4 MB of
      B_Up/B_Down reads the ``full`` build performs were never fetched -- a
      meaningful part of the reported 3.2-4.7 ms gap is that traffic, not
      imbalance. Every entry must declare ``l3_bytes`` and every stage's must
      equal ``full``'s, which is doc 38 §3.4 step 3's recommendation ("do not
      elide, so every variant issues identical DDR traffic and the gap is
      purely serialization") turned into a precondition.

    A third fix is structural rather than a guard: the result is a dataclass
    with ``to_json``, so a caller cannot compute the gap without something to
    write. iron's ``--output-json`` defaulted to ``None`` and that is why none
    of its numbers has a file behind it.
    """
    if not stages:
        raise ValueError("stage_gap() with no stages: there is no max to take")
    entries = (full,) + tuple(stages)
    for entry in entries:
        if not entry.contains:
            raise ValueError(
                f"{entry.resource} declares no `contains`. Defect 1's guard is "
                "the containment check, and it is vacuous on an empty set -- a "
                "check that cannot fail is the defect class this study has "
                "removed six of."
            )
        if entry.l3_bytes is None:
            raise ValueError(
                f"{entry.resource} declares no `l3_bytes`. Defect 2 is exactly "
                "an unstated difference in DDR traffic between variants."
            )
        if entry.ns is None:
            raise ValueError(f"{entry.resource} has no measured time")
    for stage in stages:
        if stage.l3_bytes != full.l3_bytes:
            raise ElidedTraffic(
                f"{stage.resource} issues {stage.l3_bytes:,} L3 bytes against "
                f"full's {full.l3_bytes:,} -- a difference of "
                f"{full.l3_bytes - stage.l3_bytes:,}. That difference lands in "
                "the gap as though it were exposed serialization; doc 38 §3.3 "
                "defect 2 is iron's addnorm1 variant never fetching B_Up/B_Down."
            )
    _reject_prefixes(tuple(stages))

    ranked = tuple(
        sorted(((s.resource, float(s.ns)) for s in stages), key=lambda r: -r[1])
    )
    max_stage, max_ns = ranked[0]
    return StageGap(
        full_ns=float(full.ns),
        max_stage=max_stage,
        max_stage_ns=max_ns,
        gap_ns=float(full.ns) - max_ns,
        ratio=float(full.ns) / max_ns,
        l3_bytes=int(full.l3_bytes),
        stages=ranked,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render(
    matrix: DemandMatrix,
    balances: tuple[PortBalance, ...],
    *,
    label: str,
    required: tuple[RequiredBandwidth, ...] = (),
    duration_ns: float | None = None,
) -> str:
    """The report. Demand beside budget on EVERY row, compliant ones included."""
    lines = [f"## Balance -- {label}\n"]
    lines.append(
        f"per-column budget: {PER_COLUMN_BUDGET} channels per direction "
        "(aircc_artifacts.SHIM_DMA_CHANNELS_PER_DIRECTION); doc 23: the budget "
        "is per COLUMN across the whole segment\n"
    )
    lines.append(
        "| column | dir | demand | budget | concurrent | mux depth | slowdown "
        "| charged | bytes | channels |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    by_key = {(b.column, b.direction): b for b in balances}
    for port in matrix.ports:
        bal = by_key[(port.column, port.direction)]
        size = "—" if port.bytes is None else f"{port.bytes:,}"
        lines.append(
            f"| {port.column} | {port.direction} | **{port.static_demand}** | "
            f"{port.budget} | {port.peak_concurrent_demand} | "
            f"{port.max_multiplex_depth} | {bal.slowdown:.3f} | "
            f"x{bal.inflation:.3f} | {size} | {', '.join(port.channels)} |"
        )

    warnings = [b.warning() for b in balances if b.warning()]
    if warnings:
        lines.append("\nOVER BUDGET -- priced, not filtered:")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append(
            "  Doc 44: exceeding the budget does not break correctness, so a "
            "legal-but-degraded point is modelled as degraded rather than "
            "excluded. Filtering would hide silent multiplexing, which is the "
            "failure mode this instrument exists to see."
        )
    else:
        lines.append(
            "\nNo column exceeds its budget. Every port above carries a row, "
            "including the compliant ones -- an empty violation list and a "
            "check that did not run look identical otherwise."
        )

    if required and duration_ns:
        lines.append(
            f"\n### Back-solved bandwidth, assuming a stall-free {duration_ns:,.0f} ns\n"
        )
        lines.append("| column | dir | bytes | required GB/s |")
        lines.append("|---|---|---|---|")
        for req in required:
            size = "—" if req.bytes is None else f"{req.bytes:,}"
            rate = (
                "—"
                if req.gigabytes_per_second is None
                else f"{req.gigabytes_per_second:.3f}"
            )
            lines.append(f"| {req.column} | {req.direction} | {size} | {rate} |")
        lines.append(
            "\nThis is SCALE-Sim's `InterfaceBandwidth: CALC` -- traffic divided "
            "by an assumed stall-free duration, one division, no simulator and "
            "no hardware run. READ THE UNIT: if the duration handed in was a "
            "MEASURED achieved latency then these are the rates the run "
            "SUSTAINED, and they are a LOWER BOUND on what a stall-free run "
            "would require. It is NOT a ratio against a ceiling either: doc 33 "
            "records that the AIR-native bandwidth operator was DEFERRED, so "
            "this tree holds no measured shim byte rate to divide by, and "
            "iron's 67.9-70.9 GB/s is a cross-toolchain import doc 33 marks as "
            "order-of-magnitude only."
        )

    lines.append(f"\nsteps on the axis: {len(matrix.steps)} (ASAP async levels, NOT cycles)")
    if matrix.unattributed:
        lines.append(f"\nUNATTRIBUTED transfers: {len(matrix.unattributed)}")
        for transfer in matrix.unattributed[:10]:
            lines.append(
                f"- line {transfer.line} @{transfer.channel} {transfer.kind}: "
                f"{transfer.reason or 'no shim allocation named'}"
            )
    if matrix.unknown_trip_counts:
        lines.append(
            f"\nUNKNOWN trip counts: {matrix.unknown_trip_counts} transfer(s) sit "
            "in loops whose bounds are not constants, so their byte volume is "
            "unknown and every affected port's bytes read as '—' rather than as "
            "a total that quietly assumed one trip."
        )
    if not matrix.is_complete:
        lines.append(
            "\nThis reading is an HONEST PARTIAL: see the two lists above for "
            "what it could not account for."
        )
    return "\n".join(lines) + "\n"


def analyse(project_dir: str | Path) -> tuple[DemandMatrix, tuple[PortBalance, ...]]:
    """Parse one aircc project directory and price it. The whole instrument, once."""
    design = routed_design_path(project_dir)
    text = Path(design).read_text(encoding="utf-8")
    matrix = demand_matrix(parse_transfers(text), parse_allocations(text))
    return matrix, balance_ports(matrix)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "project_dirs",
        nargs="+",
        help="aircc project directories (the ones holding "
        f"{aircc_artifacts.ROUTED_DESIGN_NAME})",
    )
    ap.add_argument(
        "--duration-ns",
        type=float,
        default=None,
        help="a MEASURED stall-free duration to back-solve the required "
        "bandwidth against. Omitted by default: a duration invented here would "
        "turn a counted byte total into a modelled bandwidth.",
    )
    ap.add_argument("--json", default=None, help="write the readings here")
    args = ap.parse_args(argv)

    payload = []
    status = 0
    for project in args.project_dirs:
        label = Path(project).parent.name or Path(project).name
        try:
            matrix, balances = analyse(project)
        except Exception as e:
            print(f"[balance] {label}: {type(e).__name__}: {e}")
            status = 1
            continue
        required = (
            back_solve(matrix, args.duration_ns) if args.duration_ns else ()
        )
        print(
            render(
                matrix,
                balances,
                label=label,
                required=required,
                duration_ns=args.duration_ns,
            )
        )
        payload.append(
            {
                "label": label,
                "project_dir": str(project),
                "budget_per_column_per_direction": PER_COLUMN_BUDGET,
                "ports": [
                    {
                        "column": p.column,
                        "direction": p.direction,
                        "static_demand": p.static_demand,
                        "peak_concurrent_demand": p.peak_concurrent_demand,
                        "budget": p.budget,
                        "max_multiplex_depth": p.max_multiplex_depth,
                        "bytes": p.bytes,
                        "transfers": p.transfers,
                        "channels": list(p.channels),
                        "slowdown": b.slowdown,
                        "inflation": b.inflation,
                        "over_budget": b.over_budget,
                    }
                    for p, b in zip(matrix.ports, balances)
                ],
                "required_bandwidth_gbps": [
                    {
                        "column": r.column,
                        "direction": r.direction,
                        "bytes": r.bytes,
                        "duration_ns": r.duration_ns,
                        "gigabytes_per_second": r.gigabytes_per_second,
                    }
                    for r in required
                ],
                "steps": list(matrix.steps),
                "unattributed_transfers": len(matrix.unattributed),
                "unknown_trip_counts": matrix.unknown_trip_counts,
                "is_complete": matrix.is_complete,
            }
        )

    if args.json and payload:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"[balance] wrote {out}")
    return status


if __name__ == "__main__":
    sys.exit(main())
