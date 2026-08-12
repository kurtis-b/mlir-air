# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The R1 structural check: the FFN interior is RESIDENT, and stays resident.

CONTRACT
    Compiles ``build_ffn_resident_module`` at every shape the catalogue
    claims for ``ffn_resident`` (from ``opcheck_specs.SPECS``) through
    ``XRTBackend(debug_ir=True)`` -- the production aircc, the numeric arm's
    backend options -- and asserts on the per-pass dumps. Hermetic: no NPU,
    no kernel objects; the compile fails at the core-ELF link deliberately
    (aiecc writes every MLIR dump first, doc 23 section 5). Do not "fix"
    that failure by compiling kernels; it only slows the check down.

    Seven clauses plus the census's own negative control, each catching a
    regression the numeric arm cannot see:

    1. ONE SEGMENT. Exactly one tile-bearing ``aie.device`` in the routed
       design (plus the anonymous control device). This is the resident
       claim itself -- the stitched composition measures one device PER
       LAUNCH (three), and a change that re-splits the segment would
       compute identical numbers while deleting the phase.
    2. THE STAGE EDGES. core->core ``aie.flow`` count EXACTLY ``herd_x``:
       the up->GeLU hand-offs, one per column. Derived for THIS design, not
       copied from J7a: every other hand-off is memtile-mediated because a
       down core's two S2MM ports are spoken for and a channel has one
       physical source (the builder's docstring carries the arithmetic).
       Fewer means an edge silently rerouted through L3 or a memtile;
       MORE is equally wrong (a hand-off left the derived shape).
    3. THE COLUMN BUDGET, counted directly: a column has TWO shim MM2S
       channels and the budget is per column ACROSS THE WHOLE SEGMENT
       (doc 23). Measured on this design today: 7 of the 16 ports, worst
       column 2 -- shim column 1 carries both staged refills. Two things
       this count must NOT do, because the arm did both until
       `[2026-08-12]` and read 1 where the truth is 2:
         - count ``shim -> core`` flows only. An L2-staged refill is a
           ``shim -> memtile`` flow and burns the same MM2S port; on R1
           that IS the whole difference between 1 and 2 (doc 31b 3.6).
         - count circuit-switched ``aie.flow`` only. Over budget AIR does
           not refuse -- it PACKET-multiplexes the streams onto one queue,
           so the circuit census of an over-budget column reads ZERO and
           the clause goes blind exactly when it is needed. The rule
           bounds DEMAND, so a packet-multiplexed group counts once per
           stream. Measured on the control below: 3 streams, one queue.
       A broadcast is one stream on one port, so the circuit half counts
       distinct source (bundle, channel) pairs rather than flows.
       THE CENSUS NEGATIVE CONTROL runs first, every time: a one-segment
       herd with 3 herd-direct L3 operands is 3 of the 2-per-column budget
       and the census MUST refuse it. An arm that cannot fail is not a
       gate; this one proves it can, on the same code path, in ~0.5 s.
    4. ZERO packet-typed channels in every dump (the J7a/J7b rule; the
       packet path is BD-starved post-H9 at these trip counts).
    5. THE DOWN RING STILL FORMS INSIDE THE COMPOSITION: K-loop data
       movement 4 -> 2 across ``air-hoist-dma-in-accum-pattern``, scoped by
       brace-matching ``air.herd @ffn_res_down`` (three herds live in this
       module; the sibling checks' "first scf.for of the dump" convention
       would find the UP herd's sweep loop and count the wrong thing), with
       the in-place accumulate call and the guarded zero inside the loop.
    6. DISPATCH. The up herd's k' loop calls the SAME in-place entry point
       with its unguarded pre-loop zero, and the GeLU herd dispatches
       ``ffn_gelu_bf16`` between its channel get and put -- named literally
       so builder drift cannot satisfy this with something else.
    7. LIVENESS, so no count passes vacuously: the air-dma-to-channel dump
       exists with no residual ``air.dma_memcpy_nd`` and all four composed
       channels present by name; the final dump carries no ``air.channel``
       and at least one ``aie.flow``.

FOOTGUNS
    - Clause 2's constant moves with the design. If the hand-off topology
      legitimately changes (e.g. R2 attaches the norm tails), re-derive it
      from the port arithmetic and say so here -- do not widen it to "at
      least herd_x" to make a regression pass.
    - Clause 3's 7-of-16 total is a RE-DERIVED literal, not a carried one.
      It was taken from a dump written by ``build-xrt``'s aircc of
      2026-08-11 13:28:03 (sha256 5cb08407...), which carries the item-6a
      fuse-pass fix and its review round. A structural literal off any
      PRE-6a dump is invalid rather than merely unlucky -- the old pass's
      lucky-GREEN R1 compiles left an extra channel alive with pairwise
      2-slot wraps where one 4-slot multiplexed stream belongs. If this
      count ever has to move, re-derive it from a fresh dump of the
      compiler in hand and record that compiler; never confirm it against
      the previous value.
    - The probe twin (``agents/probes/probe_ffn_resident_interior.py``)
      reports the same census with the exploratory extras; THIS file is the
      gate. Verifying the probe standalone is not verifying this arm
      (doc 23 section 5, the lesson that cost two suite runs).
"""

import glob
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJ_ROOT = _HERE.parent  # programming_examples/
for _p in (str(_PROJ_ROOT), str(_PROJ_ROOT / "llms"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ml_dtypes import bfloat16  # noqa: E402

from air.ir import *  # noqa: E402,F403
from air.dialects.affine import apply as affine_apply  # noqa: E402
from air.dialects.air import *  # noqa: E402,F403
from air.dialects.func import FuncOp  # noqa: E402
from air.dialects.memref import AllocOp, DeallocOp  # noqa: E402
from air.backend.xrt import XRTBackend  # noqa: E402
from air.backend.xrt_runner import type_mapper  # noqa: E402

from builders.ffn_resident import (  # noqa: E402
    CHANNEL_DOWN_FEED,
    CHANNEL_G,
    CHANNEL_H,
    CHANNEL_UP_FEED,
    build_ffn_resident_module,
)
from builders.ffn_accum import (  # noqa: E402
    FFN_ACCUM_HERD_X,
    FFN_ACCUM_TILE_K,
)
from opcheck_specs import SPECS  # noqa: E402

# Clause 6 names the entry points literally (the ffn_accum_structure
# convention): a builder that drifts to other symbols fails here rather than
# quietly redefining what "the in-place kernel" or "the activation" means.
REQUIRED_MM_SYMBOL = "ffn_matmul_bf16_bf16_up_proj"
REQUIRED_ZERO_SYMBOL = "ffn_zero_bf16_up_proj"
REQUIRED_GELU_SYMBOL = "ffn_gelu_bf16"

_UP_HERD_MARKER = "air.herd @ffn_res_up"
_GELU_HERD_MARKER = "air.herd @ffn_res_gelu"
_DOWN_HERD_MARKER = "air.herd @ffn_res_down"

_HOIST_DUMP_SUFFIX = "_after_air-hoist-dma-in-accum-pattern.mlir"
_CHANNEL_DUMP_SUFFIX = "_after_air-dma-to-channel.mlir"

# From mlir_aie's AIETargetModel.h, quoted rather than assumed:
# NPU2TargetModel::columns() = 8, and a shim NOC tile drives TWO MM2S DMA
# channels. Doc 23's rule is per COLUMN across the whole segment.
NPU2_COLUMNS = 8
SHIM_MM2S_PER_COLUMN = 2
NPU2_SHIM_MM2S_PORTS = NPU2_COLUMNS * SHIM_MM2S_PER_COLUMN

# The census's own negative control: one more herd-direct L3 operand than a
# column has MM2S channels. Verified over-budget by construction (the herd
# occupies one column per lane and every lane fetches all three).
CONTROL_L3_STREAMS = SHIM_MM2S_PER_COLUMN + 1

_MOVEMENT = re.compile(r"air\.dma_memcpy_nd|air\.channel\.(?:put|get)")
_TILE_RE = re.compile(r"%(\S+) = aie\.tile\((\d+),\s*(\d+)\)")
_FLOW_RE = re.compile(
    r"aie\.flow\(%(\S+),\s*(\w+)\s*:\s*(\d+),\s*%(\S+),\s*(\w+)\s*:\s*(\d+)\)"
)
# aie.packet_flow(<id>) { aie.packet_source<...>  aie.packet_dest<...> ... }
# ONE source per flow (a flow may broadcast to many dests), so counting the
# sources counts the flows and needs no brace matching. Verified on the
# negative control's routed dump: 12 packet_flow, 12 packet_source.
_PACKET_SOURCE_RE = re.compile(r"aie\.packet_source<%(\S+),\s*(\w+)\s*:\s*(\d+)>")


def _aircc_debug_dumps(module):
    """The production pipeline's per-pass dumps for ``module``."""
    prev_cwd = os.getcwd()
    error = None
    with tempfile.TemporaryDirectory(prefix="ffn-resident-structure-") as work:
        os.chdir(work)
        try:
            backend = XRTBackend(
                omit_while_true_loop=False,
                output_format="elf",
                instance_name="ffn_resident",
                runtime_loop_tiling_sizes=[2, 2],
                target_device="npu2",
                debug_ir=True,
            )
            try:
                backend.compile(module)
            except Exception as exc:
                error = f"{type(exc).__name__}: {str(exc)[-200:]}"
            finally:
                try:
                    backend.unload()
                except Exception:
                    pass
            dumps = [
                (os.path.basename(p), Path(p).read_text())
                for p in sorted(glob.glob("air_project/debug_ir/pass_*.mlir"))
            ]
        finally:
            os.chdir(prev_cwd)
    return dumps, error


def _braced_block(text, marker):
    """Lines from the first line containing ``marker`` through its close."""
    if text is None:
        return None
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if marker in line), None)
    if start is None:
        return None
    depth, body = 0, []
    for line in lines[start:]:
        body.append(line)
        depth += line.count("{") - line.count("}")
        if depth <= 0 and len(body) > 1:
            break
    return "\n".join(body)


def _first_loop(text):
    return _braced_block(text, "scf.for")


def _inner_loop(loop_block):
    """The first scf.for INSIDE ``loop_block`` (whose own first line is one)."""
    if not loop_block:
        return None
    body = "\n".join(loop_block.splitlines()[1:])
    return _braced_block(body, "scf.for")


def _device_blocks(text):
    blocks = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if "aie.device" in lines[i]:
            m = re.search(r"aie\.device\(\w+\)\s*(@\S+)?", lines[i])
            name = m.group(1) if m and m.group(1) else "<anonymous>"
            depth, body = 0, []
            while i < len(lines):
                body.append(lines[i])
                depth += lines[i].count("{") - lines[i].count("}")
                i += 1
                if depth <= 0 and len(body) > 1:
                    break
            blocks.append((name, "\n".join(body)))
        else:
            i += 1
    return blocks


def _shim_mm2s_census(dev):
    """Per-column shim MM2S demand on one routed ``aie.device`` block.

    Returns ``(demand, detail)``: ``demand`` maps column -> the number of
    L3-facing streams that column's shim must drive, and ``detail`` maps
    column -> ``(circuit_ports, packet_streams, to_core, to_memtile, s2mm)``
    for the report. Doc 23's rule bounds ``demand`` at
    ``SHIM_MM2S_PER_COLUMN``.

    Both halves are load-bearing:

    - a CIRCUIT-switched ``aie.flow`` out of a shim tile takes one MM2S
      channel per distinct source (bundle, channel), whatever it is wired
      to -- a core (a herd-direct fetch) or a memtile (an L2-staged refill).
      Several flows off ONE source port are a broadcast: one stream, one
      port, counted once.
    - a PACKET flow is what AIR emits when the budget is already blown: k
      streams share one queue by packet id. Counting the surviving ports
      there would read 1 (or, with the whole design converted, 0) for a
      column carrying k, so each packet flow counts as the separate stream
      it is.

    The two halves are added. A design that somehow ran a circuit flow and
    a packet flow off the SAME shim port would be counted twice; that errs
    toward refusing, which is the safe direction for a budget, and no
    routed design in this tree does it (AIR converts wholesale).
    """
    tiles = {m[0]: (int(m[1]), int(m[2])) for m in _TILE_RE.findall(dev)}

    def kind(t):
        rc = tiles.get(t)
        if rc is None:
            return None
        return "shim" if rc[1] == 0 else "memtile" if rc[1] == 1 else "core"

    def col(t):
        rc = tiles.get(t)
        return None if rc is None else rc[0]

    circuit_ports = {}
    to_core, to_memtile, s2mm = Counter(), Counter(), Counter()
    for s, sb, sp, d, _db, _dp in _FLOW_RE.findall(dev):
        if kind(s) == "shim" and sb == "DMA":
            circuit_ports.setdefault(col(s), set()).add((sb, int(sp)))
            if kind(d) == "core":
                to_core[col(s)] += 1
            elif kind(d) == "memtile":
                to_memtile[col(s)] += 1
        if kind(d) == "shim" and _db == "DMA":
            s2mm[col(d)] += 1

    packet = Counter()
    for t, b, _p in _PACKET_SOURCE_RE.findall(dev):
        if kind(t) == "shim" and b == "DMA":
            packet[col(t)] += 1

    cols = set(circuit_ports) | set(packet) | set(s2mm)
    demand, detail = {}, {}
    for c in sorted(x for x in cols if x is not None):
        nc = len(circuit_ports.get(c, ()))
        np_ = packet.get(c, 0)
        demand[c] = nc + np_
        detail[c] = (nc, np_, to_core.get(c, 0), to_memtile.get(c, 0),
                     s2mm.get(c, 0))
    return demand, detail


@module_builder
def build_over_budget_module(n_l3=CONTROL_L3_STREAMS, herd_x=4, elems=512):
    """The census's negative control: ``n_l3`` herd-direct L3 fetches per column.

    One segment, one herd of ``[herd_x, 1]``, every lane fetching all
    ``n_l3`` L3 operands into its own L1 buffer and storing one back. Each
    fetch is an L3-facing MM2S in EVERY column the herd occupies, so the
    column demand is ``n_l3`` by construction and nothing about the shape of
    R1 is involved.

    Each lane takes its OWN L3 slice at both ends: giving every core the
    same slice makes the inbound copy a broadcast and the outbound one a
    fan-in, which would change what is being counted.
    """
    xrt_dtype = type_mapper(bfloat16)
    l3_ty = MemRefType.get([herd_x * elems], xrt_dtype)
    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_ty = MemRefType.get([elems], xrt_dtype, memory_space=l1_space)
    off_map = AffineMap.get(
        0,
        1,
        [AffineExpr.get_mul(AffineSymbolExpr.get(0), AffineConstantExpr.get(elems))],
    )

    @FuncOp.from_py_func(*([l3_ty] * (n_l3 + 1)))
    def over_budget(*a):

        @launch(operands=list(a))
        def ob_launch(*l):

            @segment(name="over_budget_seg", operands=list(l))
            def ob_seg(*s):

                def _stage(tx, ty, _sx, _sy, *h):
                    bufs = []
                    for i in range(n_l3):
                        buf = AllocOp(l1_ty, [], [])
                        dma_memcpy_nd(
                            buf,
                            h[i],
                            src_offsets=[affine_apply(off_map, [tx])],
                            src_sizes=[elems],
                            src_strides=[1],
                        )
                        bufs.append(buf)
                    dma_memcpy_nd(
                        h[n_l3],
                        bufs[0],
                        dst_offsets=[affine_apply(off_map, [tx])],
                        dst_sizes=[elems],
                        dst_strides=[1],
                    )
                    for buf in bufs:
                        DeallocOp(buf)

                herd(name="ob_stage", sizes=[herd_x, 1], operands=list(s))(_stage)


def check_census_control():
    """Clause 3's negative control. Returns problems; empty means it FAILED as required.

    MEASURED, 2026-08-12, build-xrt aircc 2026-08-11 13:28:03: at three
    herd-direct L3 operands AIR routes the design with ZERO inbound circuit
    flows and 12 ``aie.packet_flow`` -- three streams multiplexed onto each
    column's single shim queue. So the pre-widening count (``shim -> core``
    circuit flows) reads 0 on a design that is 50% over budget, and the
    widened one reads 3. Both numbers are asserted here, so reverting either
    half of the widening fails this control rather than passing quietly.
    """
    label = "census negative control"
    module = build_over_budget_module()
    dumps, compile_error = _aircc_debug_dumps(module)
    if not dumps:
        return [
            f"{label}: aircc wrote no debug dumps -- the census is unproven "
            f"({compile_error or 'compile reported success'})"
        ]
    routed = next(
        (t for _, t in reversed(dumps) if _TILE_RE.search(t)),
        None,
    )
    if routed is None:
        return [f"{label}: no routed dump -- the census is unproven"]
    devices = [d for _, d in _device_blocks(routed) if _TILE_RE.search(d)]
    demand, detail = {}, {}
    for dev in devices:
        d, x = _shim_mm2s_census(dev)
        demand.update(d)
        detail.update(x)
    worst = max(demand.values(), default=0)
    # what the pre-widening clause counted: circuit shim->core flows only
    old_worst = max((v[2] for v in detail.values()), default=0)
    print(
        f"[ffn-resident-structure] {label}: {CONTROL_L3_STREAMS} L3 streams "
        f"per column -> widened census {worst}, pre-widening shim->core "
        f"count {old_worst}, packet streams "
        f"{sum(v[1] for v in detail.values())}"
    )
    problems = []
    if worst <= SHIM_MM2S_PER_COLUMN:
        problems.append(
            f"{label}: the census ADMITTED a design demanding "
            f"{CONTROL_L3_STREAMS} shim MM2S per column (worst column read "
            f"{worst}, budget {SHIM_MM2S_PER_COLUMN}) -- clause 3 cannot "
            "fail, so it is not gating anything"
        )
    if worst != CONTROL_L3_STREAMS:
        problems.append(
            f"{label}: the census read {worst} where the construction "
            f"demands exactly {CONTROL_L3_STREAMS} -- it is counting "
            "something other than per-column L3-facing streams"
        )
    if old_worst > SHIM_MM2S_PER_COLUMN:
        problems.append(
            f"{label}: the pre-widening shim->core count already reads "
            f"{old_worst} here, so this control does not discriminate the "
            "widening -- pick a control the narrow count cannot see"
        )
    return problems


def check_shape(shape_key, seq_len, ffn_dim, emb_dim):
    """One shape's verdict. Returns the list of problems, empty on pass."""
    label = f"ffn_resident [{shape_key}]"
    herd_x = FFN_ACCUM_HERD_X
    module = build_ffn_resident_module(
        seq_len, ffn_dim, emb_dim, herd_x=herd_x, tile_k=FFN_ACCUM_TILE_K
    )
    dumps, compile_error = _aircc_debug_dumps(module)
    if not dumps:
        return [
            f"{label}: aircc wrote no debug dumps -- nothing structural is "
            f"measured ({compile_error or 'compile reported success'})"
        ]
    problems = []
    names = [n for n, _ in dumps]

    # 5. the down ring, scoped to its herd
    hoist_idx = next(
        (i for i, n in enumerate(names) if n.endswith(_HOIST_DUMP_SUFFIX)), None
    )
    before = after = -1
    if not hoist_idx:
        problems.append(
            f"{label}: no usable air-hoist-dma-in-accum-pattern dump -- the "
            "down ring's formation is unmeasured "
            f"({compile_error or 'no error'})"
        )
    else:
        down_before = _first_loop(
            _braced_block(dumps[hoist_idx - 1][1], _DOWN_HERD_MARKER)
        )
        down_after = _first_loop(_braced_block(dumps[hoist_idx][1], _DOWN_HERD_MARKER))
        before = len(_MOVEMENT.findall(down_before or ""))
        after = len(_MOVEMENT.findall(down_after or ""))
        mm_marker = f"call @{REQUIRED_MM_SYMBOL}("
        zero_marker = f"call @{REQUIRED_ZERO_SYMBOL}("
        if before != 4 or after != 2:
            problems.append(
                f"{label}: down-herd K loop data movement {before} -> {after} "
                f"across {names[hoist_idx]}, need 4 -> 2 -- the accumulator "
                "round trip did not hoist inside the composition"
            )
        if (
            mm_marker not in (down_after or "")
            or zero_marker not in (down_after or "")
            or "scf.if" not in (down_after or "")
        ):
            problems.append(
                f"{label}: the down K loop must dispatch "
                f"@{REQUIRED_MM_SYMBOL} with the scf.if-guarded "
                f"@{REQUIRED_ZERO_SYMBOL} -- a stale y BO is otherwise "
                "returned as y_old + a @ w and XRTRunner's zero-filled "
                "placeholders conceal it"
            )
        # 6. the up herd and GeLU herd dispatch, in the same dump
        up_body = _braced_block(dumps[hoist_idx][1], _UP_HERD_MARKER)
        up_sweep = _first_loop(up_body)
        up_loop = _inner_loop(up_sweep)  # the k' loop inside the sweep loop
        if mm_marker not in (up_loop or ""):
            problems.append(
                f"{label}: no call @{REQUIRED_MM_SYMBOL} inside the up "
                "herd's k' loop -- the up stage is not the in-place ring"
            )
        if zero_marker not in (up_sweep or "") or zero_marker in (up_loop or ""):
            problems.append(
                f"{label}: the up herd's group zero must sit in the sweep "
                "body BEFORE the k' loop (unguarded, once per group) -- "
                "inside the k' loop it re-zeroes partial sums"
            )
        gelu_body = _braced_block(dumps[hoist_idx][1], _GELU_HERD_MARKER)
        gelu_loop = _first_loop(gelu_body)
        if f"call @{REQUIRED_GELU_SYMBOL}(" not in (gelu_loop or ""):
            problems.append(
                f"{label}: no call @{REQUIRED_GELU_SYMBOL} inside the GeLU "
                "herd's loop -- the activation stage is not dispatching the "
                "kernel"
            )

    # 7. channel-lowering liveness
    chan = next((t for n, t in dumps if n.endswith(_CHANNEL_DUMP_SUFFIX)), None)
    if chan is None:
        problems.append(
            f"{label}: no air-dma-to-channel dump -- the packet count proves " "nothing"
        )
    else:
        if "air.dma_memcpy_nd" in chan:
            problems.append(
                f"{label}: {chan.count('air.dma_memcpy_nd')} air.dma_memcpy_nd "
                "left after air-dma-to-channel -- lowering incomplete"
            )
        missing = [
            c
            for c in (CHANNEL_UP_FEED, CHANNEL_H, CHANNEL_G, CHANNEL_DOWN_FEED)
            if f"@{c}" not in chan
        ]
        if missing:
            problems.append(
                f"{label}: composed channel(s) {missing} missing at "
                "air-dma-to-channel -- this is not the three-stage design"
            )

    # 4. packet census
    n_packet = sum(text.count("npu_dma_packet") for _, text in dumps)
    if n_packet:
        problems.append(
            f"{label}: {n_packet} packet-typed channel references across the "
            "dumps -- a per-column budget is exceeded somewhere"
        )

    # 1-3. the routed design
    final_name, final = dumps[-1]
    if "air.channel" in final:
        problems.append(
            f"{label}: air.channel still present in {final_name} -- not "
            f"lowered to AIE routing ({compile_error or 'no error'})"
        )
    devices = _device_blocks(final)
    tiled = [(n, d) for n, d in devices if _TILE_RE.search(d)]
    if len(tiled) != 1:
        problems.append(
            f"{label}: {len(tiled)} tile-bearing aie.device in {final_name}, "
            "need exactly 1 -- the one-segment resident claim itself"
        )
    total_core_core = 0
    any_flow = False
    mm2s_total = 0
    mm2s_worst = 0
    for dev_name, dev in devices:
        tiles = {m[0]: (int(m[1]), int(m[2])) for m in _TILE_RE.findall(dev)}
        if not tiles:
            continue
        rows_by = {k: v[1] for k, v in tiles.items()}

        def kind(t):
            r = rows_by.get(t)
            if r is None:
                return None
            return "shim" if r == 0 else "memtile" if r == 1 else "core"

        flows = _FLOW_RE.findall(dev)
        any_flow = any_flow or bool(flows)
        total_core_core += sum(
            1 for s, _sb, _sp, d, _db, _dp in flows
            if kind(s) == "core" and kind(d) == "core"
        )
        # clause 3: the per-column shim MM2S budget, BOTH flow kinds
        demand, detail = _shim_mm2s_census(dev)
        mm2s_total += sum(demand.values())
        mm2s_worst = max([mm2s_worst] + list(demand.values()))
        for c in sorted(demand):
            nc, np_, tc, tm, _s2 = detail[c]
            print(
                f"[ffn-resident-structure]   shim col {c}: MM2S {demand[c]} "
                f"(circuit {nc} = {tc} ->core + {tm} ->memtile, packet {np_})"
            )
        over = {c: n for c, n in demand.items() if n > SHIM_MM2S_PER_COLUMN}
        if over:
            problems.append(
                f"{label}: device {dev_name}: columns {sorted(over)} demand "
                f"more than {SHIM_MM2S_PER_COLUMN} shim MM2S ({over}) -- doc "
                "23's per-column budget; AIR packet-multiplexes past it "
                "instead of refusing"
            )
    if not any_flow:
        problems.append(
            f"{label}: no aie.flow in {final_name} -- nothing was routed "
            f"({compile_error or 'compile reported success'})"
        )
    elif tiled and total_core_core != herd_x:
        problems.append(
            f"{label}: {total_core_core} core->core flows, need exactly "
            f"{herd_x} (the up->GeLU edges; the derivation is in the "
            "builder's docstring) -- a stage edge moved"
        )

    verdict = "FAIL" if problems else "PASS"
    print(
        f"[ffn-resident-structure] {label}: {verdict} "
        f"({len(tiled)} device, {total_core_core} core->core, K-loop "
        f"{before} -> {after}, {n_packet} packet-typed channels, shim MM2S "
        f"{mm2s_total}/{NPU2_SHIM_MM2S_PORTS} worst column {mm2s_worst})"
    )
    return problems


def main():
    shapes = [s for s in SPECS if s["operator"] == "ffn_resident"]
    if not shapes:
        print("[ffn-resident-structure] FAIL: no ffn_resident shapes in SPECS")
        return 1
    # Clause 3's negative control FIRST and unconditionally: if the census
    # cannot refuse an over-budget design, nothing below it is a gate.
    problems = check_census_control()
    for spec in shapes:
        shape = spec["shape"]
        problems += check_shape(
            spec["shape_key"], shape["seq_len"], shape["ffn_dim"], shape["emb_dim"]
        )
    if problems:
        print("[ffn-resident-structure] FAIL")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print("[ffn-resident-structure] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
