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

    Seven clauses, each catching a regression the numeric arm cannot see:

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
    3. THE COLUMN BUDGET, counted directly: at most 2 shim-facing inbound
       flows per column on the routed design (measured today: 1 -- the
       down C fetch; the three refill streams land on separate columns).
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

from air.backend.xrt import XRTBackend  # noqa: E402

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

_MOVEMENT = re.compile(r"air\.dma_memcpy_nd|air\.channel\.(?:put|get)")
_TILE_RE = re.compile(r"%(\S+) = aie\.tile\((\d+),\s*(\d+)\)")
_FLOW_RE = re.compile(
    r"aie\.flow\(%(\S+),\s*\w+\s*:\s*\d+,\s*%(\S+),\s*\w+\s*:\s*\d+\)"
)


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
    for dev_name, dev in devices:
        tiles = {m[0]: (int(m[1]), int(m[2])) for m in _TILE_RE.findall(dev)}
        if not tiles:
            continue
        rows_by = {k: v[1] for k, v in tiles.items()}
        cols_by = {k: v[0] for k, v in tiles.items()}

        def kind(t):
            r = rows_by.get(t)
            if r is None:
                return None
            return "shim" if r == 0 else "memtile" if r == 1 else "core"

        flows = _FLOW_RE.findall(dev)
        any_flow = any_flow or bool(flows)
        total_core_core += sum(
            1 for s, d in flows if kind(s) == "core" and kind(d) == "core"
        )
        inbound = Counter(
            cols_by.get(d) for s, d in flows if kind(s) == "shim" and kind(d) == "core"
        )
        over = {c: n for c, n in inbound.items() if n > 2}
        if over:
            problems.append(
                f"{label}: device {dev_name}: columns {sorted(over)} take "
                f"more than 2 shim-facing inbound flows ({over})"
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
        f"{before} -> {after}, {n_packet} packet-typed channels)"
    )
    return problems


def main():
    shapes = [s for s in SPECS if s["operator"] == "ffn_resident"]
    if not shapes:
        print("[ffn-resident-structure] FAIL: no ffn_resident shapes in SPECS")
        return 1
    problems = []
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
