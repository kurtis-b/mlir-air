# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The tail pipeline's structural check: AN1 -> FFN -> AN2 as ONE routed segment.

CONTRACT
    Compiles ``build_tail_pipeline_module`` at two shapes -- iron's baseline
    (seq 64, emb 48, ffn 96, m 64, k 48, n 96, depth 1, n_b 1) and a
    16-row-tile chain (seq 64, emb 96, ffn 192, m 16, k 48, n 96, depth 2,
    n_b 2) -- through ``XRTBackend(debug_ir=True)``, the production aircc,
    and asserts on the per-pass dumps. Hermetic: no NPU, no kernel objects;
    the compile fails at the core-ELF link deliberately (aiecc writes every
    MLIR dump and the link scripts first). Do not "fix" that failure by
    compiling kernels; it only slows the check down. Clause 1 pins the
    failure to THAT link error: any other aircc refusal is a FAIL carrying
    the refusing text.

    The clauses, each catching a regression the numeric arm cannot see:

    1. NO REFUSAL. aircc runs to the routed design and the only error is
       the deliberate missing-object link.
    2. ONE SEGMENT. Exactly one tile-bearing ``aie.device`` (plus the
       anonymous control device). A composition that re-split into a device
       per herd would compute identical numbers while deleting the design.
    3. THE STAGE EDGES. core->core ``aie.flow`` count EXACTLY ``n_b``: the
       up->down H hand-offs, one per column. Derived for THIS design from
       the core DMA port budget (two S2MM, two MM2S per core): every other
       hop -- AN1->up (a 4-D retile a core BD cannot carry), the ring, the
       reduction relay, down->AN2 (the un-tile) -- is memtile-mediated. The
       builder's docstring carries the arithmetic. Fewer means an edge went
       through L3; more means one left the derived shape.
    4. ZERO packet-typed channels in every dump and zero ``aie.packet_flow``
       in the routed design (over a column's budget AIR packet-multiplexes
       rather than refusing, and that path is BD-starved at these trip
       counts).
    5. THE COLUMN BUDGET, counted directly with ffn_resident_structure's
       census (circuit ports AND packet streams, shim->core AND
       shim->memtile): at most 2 shim MM2S per column. Its negative control
       runs first, every time, so the clause is demonstrably able to fail.
    6. THE MEMTILE CAP. Fewer than 48 ``aie.dma_bd`` blocks in every
       ``aie.memtile_dma`` (the hardware has 48; ffn_resident's per-column
       staging crossed it), and at most 16 in every core ``aie.mem``.
    7. L1 RESIDENCY. Every core tile carries exactly the ``aie.buffer``
       set the builder's plan lists for its herd -- one copy each -- and
       their bytes plus the stack stay under the 64 KiB tile. A toolchain
       that starts rotating herd-scope buffers fails here, not in aiecc's
       allocator minutes later.
    8. THE RETILE LIVES ON A MEMTILE. Every ``aie.dma_bd`` with four
       dimensions sits inside an ``aie.memtile_dma`` (none in a core
       ``aie.mem``), and at rpc > 8 at least one exists: that is the
       row-major -> blocked seam, where the builder says it is.
    9. DISPATCH, by name, in the routed cores: every AN core calls
       ``fused_add_layer_norm_2outs``; every up core ``ffn_zero_bf16_up_proj``
       and ``ffn_matmul_bf16_bf16_up_proj``; every down core
       ``ffn_gelu_bf16``, ``ffn_zero_bf16_down_proj`` and
       ``ffn_matmul_with_acc_bf16_bf16_down_proj``; and EXACTLY
       ``max(n_b - 1, 1)`` down cores ``ffn_eltwise_add_bf16_vector``: the
       chain's last core has no tail (its finals are relayed from its
       ring), a lone core folds in its own zero (the chain's role
       specialization survived air-to-aie).
   10. LIVENESS, so no count passes vacuously: the air-dma-to-channel dump
       has no residual ``air.dma_memcpy_nd`` and every composed channel by
       name; the final dump carries no ``air.channel`` and at least one
       ``aie.flow``.
   11. ONE TOKEN. Every ``aie.use_lock`` inside a core tile -- its
       ``aie.mem`` BDs and its ``aie.core`` program -- acquires or releases
       exactly 1. The 2026-08-22 deadlock was a put BD acquiring 2 (an L1
       buffer that was two gets' destination and one put's source) against
       a core releasing 1: this clause reads it off the routed dump, no
       device needed (builders/tail_pipeline.py FOOTGUNS, the one-token
       rule).
   12. THE FEED ALTERNATES. In every down core's ``aie.mem``, the S2MM
       chain carrying the ``[tile_n, tile_k]`` w_down buffer never puts two
       consecutive BDs (cyclically) on one buffer, and its acquire lock is
       initialised to 2 -- the strict (b, acc) round-robin with two
       distinct destinations. A third destination (init 3) or a repeated
       destination is the measured wrong answer (FOOTGUNS, the feed's
       round-robin rule).

FOOTGUNS
    - Clause 3's constant moves with the design. If a hop legitimately
      changes (a core BD grows a fourth dimension, a second MM2S frees up),
      re-derive it from the port arithmetic and say so in the builder -- do
      not widen it to "at least n_b".
    - The baseline shape is built with ``allow_an_lane_truncation=True``:
      emb 48 is not a multiple of the AN kernel's 32 lanes, and the module
      routes identically either way. That flag is a statement that THIS
      check is structural; no numeric arm may pass it.
    - The census and its negative control are imported from
      ffn_resident_structure.py rather than copied, so there is one
      implementation of the column rule in the suite.
"""

import glob
import os
import re
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJ_ROOT = _HERE.parent  # programming_examples/
for _p in (str(_PROJ_ROOT), str(_PROJ_ROOT / "llms"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from air.backend.xrt import XRTBackend  # noqa: E402

from builders.tail_pipeline import (  # noqa: E402
    ADD_SYMBOL,
    AN_SYMBOL,
    CHANNEL_A_FEED,
    CHANNEL_ACC_STORE,
    CHANNEL_AN1_FEED,
    CHANNEL_AN1_OUT,
    CHANNEL_AN2_FEED,
    CHANNEL_DOWN_FEED,
    CHANNEL_DOWN_OUT,
    CHANNEL_H,
    CHANNEL_REDUCE,
    CHANNEL_WUP_FEED,
    DOWN_MM_SYMBOL,
    DOWN_ZERO_SYMBOL,
    GELU_SYMBOL,
    HERD_AN1,
    HERD_AN2,
    HERD_DOWN,
    HERD_UP,
    L1_BYTES,
    L1_STACK_BYTES,
    UP_MM_SYMBOL,
    UP_ZERO_SYMBOL,
    build_tail_pipeline_module,
    tail_pipeline_l1_bytes,
)
from ffn_resident_structure import (  # noqa: E402
    NPU2_SHIM_MM2S_PORTS,
    SHIM_MM2S_PER_COLUMN,
    _device_blocks,
    _FLOW_RE,
    _shim_mm2s_census,
    _TILE_RE,
    check_census_control,
)

TAG = "[tail-pipeline-structure]"

# The two shapes, as (label, kwargs). The baseline is iron's test.py row
# (64, 48, 96, m 64, k 48, n 96, depth 1, nA 1, nB 1); the second is the
# 16-row chain with a two-block ring and two columns.
SHAPES = [
    (
        "baseline 64x48x96 m64 k48 n96 d1 nb1",
        dict(
            seq_len=64, emb_dim=48, ffn_dim=96, tile_m=64, tile_k=48, tile_n=96,
            down_proj_depth=1, n_b=1, allow_an_lane_truncation=True,
        ),
    ),
    (
        "chain 64x96x192 m16 k48 n96 d2 nb2",
        dict(
            seq_len=64, emb_dim=96, ffn_dim=192, tile_m=16, tile_k=48, tile_n=96,
            down_proj_depth=2, n_b=2,
        ),
    ),
]

# From mlir_aie's AIETargetModel: a memtile has 48 BDs, a core tile 16. The
# memtile clause is stated as STRICTLY fewer than 48 (the spec's bound).
MEMTILE_BD_CAP = 48
CORE_BD_CAP = 16

_CHANNEL_DUMP_SUFFIX = "_after_air-dma-to-channel.mlir"
# The deliberate failure: aiecc's per-core link cannot find the objects the
# herds name. Anything else on stderr is a refusal.
_EXPECTED_LINK_ERROR = re.compile(r"unable to find air_project/tail_pipeline_(an|ffn)\.o")

_BD_RE = re.compile(r"aie\.dma_bd\(")
_BD_4D_RE = re.compile(r"aie\.dma_bd\([^\n]*sizes = \[\s*\d+,\s*\d+,\s*\d+,\s*\d+\s*\]")
_BUFFER_RE = re.compile(
    r"aie\.buffer\(%(\S+)\)[^\n]*memref<([0-9x]+)xbf16"
)
_CORE_RE = re.compile(r"aie\.core\(%(\S+)\)")
_USE_LOCK_RE = re.compile(r"aie\.use_lock\(%(\S+), (\w+), %c(\d+)_i32\)")
_HERD_NAME_RE = re.compile(r'air\.herd_name = "([^"]+)"')


def _aircc_debug_dumps(module):
    """The production pipeline's per-pass dumps for ``module``, plus its error."""
    prev_cwd = os.getcwd()
    error = None
    with tempfile.TemporaryDirectory(prefix="tail-pipeline-structure-") as work:
        os.chdir(work)
        try:
            backend = XRTBackend(
                omit_while_true_loop=False,
                output_format="elf",
                instance_name="tail_pipeline",
                runtime_loop_tiling_sizes=[2, 2],
                target_device="npu2",
                debug_ir=True,
            )
            try:
                backend.compile(module)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
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


def _regions(text, opener):
    """Every ``opener``-headed brace-matched region in ``text``: (header, body)."""
    out = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if opener in lines[i]:
            depth, body = 0, []
            while i < len(lines):
                body.append(lines[i])
                depth += lines[i].count("{") - lines[i].count("}")
                i += 1
                if depth <= 0 and len(body) > 1:
                    break
            out.append((body[0], "\n".join(body)))
        else:
            i += 1
    return out


def _memref_elems(shape):
    n = 1
    for d in shape.split("x"):
        n *= int(d)
    return n


def check_shape(label, kwargs):
    """One shape's verdict. Returns the list of problems, empty on pass."""
    n_b = kwargs["n_b"]
    rpc = kwargs["tile_m"] // 2
    problems = []
    module = build_tail_pipeline_module(**kwargs)
    dumps, compile_error = _aircc_debug_dumps(module)
    names = [n for n, _ in dumps]

    # 1. no refusal: the dumps reach a routed design, and the only error is
    #    the deliberate link failure.
    routed_idx = next(
        (i for i in range(len(dumps) - 1, -1, -1) if _TILE_RE.search(dumps[i][1])), None
    )
    if routed_idx is None:
        return [
            f"{label}: aircc wrote no routed dump ({len(dumps)} dumps, last "
            f"{names[-1] if names else 'none'}) -- REFUSED: "
            f"{compile_error or 'compile reported success'}"
        ]
    if compile_error and not _EXPECTED_LINK_ERROR.search(compile_error):
        tail = compile_error.strip().splitlines()
        tail = "\n      ".join(tail[-12:])
        problems.append(
            f"{label}: aircc failed for a reason other than the deliberate "
            f"missing-object link -- a REFUSAL:\n      {tail}"
        )
    final_name, final = dumps[routed_idx]

    # 10. liveness at air-dma-to-channel
    chan = next((t for n, t in dumps if n.endswith(_CHANNEL_DUMP_SUFFIX)), None)
    if chan is None:
        problems.append(f"{label}: no air-dma-to-channel dump -- nothing proves the lowering")
    else:
        if "air.dma_memcpy_nd" in chan:
            problems.append(
                f"{label}: {chan.count('air.dma_memcpy_nd')} air.dma_memcpy_nd left "
                "after air-dma-to-channel -- lowering incomplete"
            )
        expected_channels = [
            CHANNEL_AN1_FEED, CHANNEL_AN1_OUT, CHANNEL_A_FEED, CHANNEL_WUP_FEED,
            CHANNEL_H, CHANNEL_DOWN_FEED, CHANNEL_ACC_STORE, CHANNEL_DOWN_OUT,
            CHANNEL_AN2_FEED,
        ] + ([CHANNEL_REDUCE] if n_b > 2 else [])
        missing = [c for c in expected_channels if f"@{c}" not in chan]
        if missing:
            problems.append(
                f"{label}: composed channel(s) {missing} missing at "
                "air-dma-to-channel -- this is not the four-herd design"
            )

    # 4. packet census
    n_packet = sum(text.count("npu_dma_packet") for _, text in dumps)
    n_packet_flow = final.count("aie.packet_flow")
    if n_packet or n_packet_flow:
        problems.append(
            f"{label}: {n_packet} packet-typed channel references across the dumps "
            f"and {n_packet_flow} aie.packet_flow in {final_name} -- a per-column "
            "budget is exceeded somewhere"
        )

    # 2, 3, 5. the routed design
    if "air.channel" in final:
        problems.append(
            f"{label}: air.channel still present in {final_name} -- not lowered "
            "to AIE routing"
        )
    devices = _device_blocks(final)
    tiled = [(n, d) for n, d in devices if _TILE_RE.search(d)]
    if len(tiled) != 1:
        problems.append(
            f"{label}: {len(tiled)} tile-bearing aie.device in {final_name}, need "
            "exactly 1 -- the one-segment claim itself"
        )
    total_core_core = 0
    any_flow = False
    mm2s_total = mm2s_worst = 0
    for dev_name, dev in tiled:
        tiles = {m[0]: (int(m[1]), int(m[2])) for m in _TILE_RE.findall(dev)}

        def kind(t):
            rc = tiles.get(t)
            if rc is None:
                return None
            return "shim" if rc[1] == 0 else "memtile" if rc[1] == 1 else "core"

        flows = _FLOW_RE.findall(dev)
        any_flow = any_flow or bool(flows)
        total_core_core += sum(
            1 for s, _sb, _sp, d, _db, _dp in flows
            if kind(s) == "core" and kind(d) == "core"
        )
        demand, detail = _shim_mm2s_census(dev)
        mm2s_total += sum(demand.values())
        mm2s_worst = max([mm2s_worst] + list(demand.values()))
        for c in sorted(demand):
            nc, np_, tc, tm, _s2 = detail[c]
            print(
                f"{TAG}   shim col {c}: MM2S {demand[c]} (circuit {nc} = {tc} "
                f"->core + {tm} ->memtile, packet {np_})"
            )
        over = {c: n for c, n in demand.items() if n > SHIM_MM2S_PER_COLUMN}
        if over:
            problems.append(
                f"{label}: device {dev_name}: columns {sorted(over)} demand more "
                f"than {SHIM_MM2S_PER_COLUMN} shim MM2S ({over}) -- the per-column "
                "budget; AIR packet-multiplexes past it instead of refusing"
            )
    if not any_flow:
        problems.append(f"{label}: no aie.flow in {final_name} -- nothing was routed")
    elif tiled and total_core_core != n_b:
        problems.append(
            f"{label}: {total_core_core} core->core flows, need exactly {n_b} "
            "(the up->down H edges; every other hop is memtile-mediated by the "
            "core port budget -- the derivation is in the builder's docstring) "
            "-- a stage edge moved"
        )

    # 6. DMA block caps
    memtile_bds = {}
    for header, body in _regions(final, "aie.memtile_dma("):
        memtile_bds[header.split("=")[0].strip()] = len(_BD_RE.findall(body))
    core_bds = {}
    for header, body in _regions(final, "= aie.mem("):
        core_bds[header.split("=")[0].strip()] = len(_BD_RE.findall(body))
    over_mt = {k: v for k, v in memtile_bds.items() if v >= MEMTILE_BD_CAP}
    if over_mt:
        problems.append(
            f"{label}: memtile DMA block count(s) {over_mt} not under "
            f"{MEMTILE_BD_CAP} -- aiecc refuses \"'aie.memtile_dma' op has more "
            f"than {MEMTILE_BD_CAP} blocks\""
        )
    over_core = {k: v for k, v in core_bds.items() if v > CORE_BD_CAP}
    if over_core:
        problems.append(
            f"{label}: core DMA block count(s) {over_core} over {CORE_BD_CAP}"
        )
    if not memtile_bds:
        problems.append(f"{label}: no aie.memtile_dma in {final_name} -- nothing staged")

    # 7. L1 residency: buffers per core tile against the builder's plan
    plan = tail_pipeline_l1_bytes(
        kwargs["emb_dim"], kwargs["tile_m"], kwargs["tile_k"], kwargs["tile_n"], n_b
    )
    tile_herd, tile_body = {}, {}
    for header, body in _regions(final, "aie.core("):
        m = _CORE_RE.search(header)
        h = _HERD_NAME_RE.search(body)
        if m and h:
            tile_herd[m.group(1)] = h.group(1)
            tile_body[m.group(1)] = body
    bufs = {}
    for t, shape in _BUFFER_RE.findall(final):
        bufs.setdefault(t, []).append(_memref_elems(shape))
    blk = kwargs["tile_m"] * kwargs["tile_k"]
    l1_worst = 0
    for t, herd_name in sorted(tile_herd.items()):
        want = [e for _, e in plan[herd_name][1]]
        if herd_name == HERD_DOWN and f"call @{ADD_SYMBOL}(" not in tile_body[t]:
            # The chain's last core has no tail; air-to-aie drops its
            # unreferenced l1_red after role specialization.
            want = [e for n, e in plan[herd_name][1] if n != "l1_red"]
        want = sorted(want)
        got = sorted(bufs.get(t, []))
        nbytes = sum(got) * 2 + L1_STACK_BYTES
        l1_worst = max(l1_worst, nbytes)
        if got != want:
            problems.append(
                f"{label}: tile {t} ({herd_name}) carries L1 buffers {got} "
                f"(elements), the plan says {want} -- aircc is rotating or "
                "dropping a herd-scope buffer"
            )
        if nbytes >= L1_BYTES:
            problems.append(
                f"{label}: tile {t} ({herd_name}) needs {nbytes} B of L1, not "
                f"under {L1_BYTES}"
            )
    if len(tile_herd) != 2 + 2 * n_b + 2:
        problems.append(
            f"{label}: {len(tile_herd)} routed cores, need {4 + 2 * n_b} "
            "(2 AN1 + n_b up + n_b down + 2 AN2)"
        )

    # 8. the retile is a memtile BD
    core_4d = sum(len(_BD_4D_RE.findall(b)) for _, b in _regions(final, "= aie.mem("))
    mt_4d = sum(len(_BD_4D_RE.findall(b)) for _, b in _regions(final, "aie.memtile_dma("))
    if core_4d:
        problems.append(
            f"{label}: {core_4d} four-dimensional aie.dma_bd inside a core aie.mem "
            "-- a core BD carries at most 3 dimensions; the retile must be the "
            "memtile's"
        )
    if rpc > 8 and mt_4d == 0:
        problems.append(
            f"{label}: no four-dimensional memtile aie.dma_bd at rpc={rpc} -- the "
            "row-major -> blocked retile is not where the builder says it is"
        )

    # 9. dispatch by name in the routed cores
    per_herd = {
        HERD_AN1: [AN_SYMBOL],
        HERD_AN2: [AN_SYMBOL],
        HERD_UP: [UP_ZERO_SYMBOL, UP_MM_SYMBOL],
        HERD_DOWN: [GELU_SYMBOL, DOWN_ZERO_SYMBOL, DOWN_MM_SYMBOL],
    }
    adders = 0
    for header, body in _regions(final, "aie.core("):
        m = _CORE_RE.search(header)
        h = _HERD_NAME_RE.search(body)
        if not (m and h):
            continue
        for sym in per_herd.get(h.group(1), []):
            if f"call @{sym}(" not in body:
                problems.append(
                    f"{label}: core {m.group(1)} ({h.group(1)}) does not call "
                    f"@{sym} -- the stage is not dispatching its kernel"
                )
        if h.group(1) == HERD_DOWN and f"call @{ADD_SYMBOL}(" in body:
            adders += 1
    if adders != max(n_b - 1, 1):
        problems.append(
            f"{label}: {adders} down cores call @{ADD_SYMBOL}, need exactly "
            f"{max(n_b - 1, 1)} -- the chain's role specialization did not "
            "survive air-to-aie"
        )

    # 11. one token on every core-tile lock op
    lock_init = {
        m.group(1): int(m.group(2))
        for m in re.finditer(r"%(lock_\S+) = aie.lock\([^)]*\) \{init = (\d+)", final)
    }
    multi = []
    for opener in ("= aie.mem(", "aie.core("):
        for header, body in _regions(final, opener):
            for m in _USE_LOCK_RE.finditer(body):
                if m.group(3) != "1":
                    multi.append((header.strip()[:40], m.group(1), m.group(2), m.group(3)))
    if multi:
        problems.append(
            f"{label}: {len(multi)} core-tile lock op(s) with a count other than "
            f"1, e.g. {multi[:3]} -- an L1 buffer is a channel destination AND "
            "source, or is put on two channels (the one-token rule)"
        )

    # 12. the down feed alternates (b, acc) and its lock init is 2
    b_shape = f"{kwargs['tile_n']}x{kwargs['tile_k']}xbf16"
    feed_ok = 0
    for t, herd_name in sorted(tile_herd.items()):
        if herd_name != HERD_DOWN:
            continue
        mem = next((b for h, b in _regions(final, "= aie.mem(") if f"aie.mem(%{t})" in h), "")
        chains = re.split(r"aie\.dma_start\(", mem)[1:]
        feed = [c for c in chains if c.startswith("S2MM") and b_shape in c]
        if len(feed) != 1:
            problems.append(
                f"{label}: tile {t} (down) has {len(feed)} S2MM chains carrying the "
                f"[{b_shape}] w_down buffer, need exactly 1"
            )
            continue
        bufs = re.findall(r"aie\.dma_bd\(%(\S+) :", feed[0])
        acq = {m.group(1) for m in _USE_LOCK_RE.finditer(feed[0]) if m.group(2).startswith("Acquire")}
        consecutive = [
            (bufs[i], bufs[(i + 1) % len(bufs)])
            for i in range(len(bufs))
            if len(bufs) > 1 and bufs[i] == bufs[(i + 1) % len(bufs)]
        ]
        inits = sorted(lock_init.get(a, -1) for a in acq)
        if consecutive or inits != [2]:
            problems.append(
                f"{label}: tile {t} (down) feed chain {bufs} acquires {sorted(acq)} "
                f"(init {inits}); need a strict two-buffer alternation with one "
                "acquire lock initialised to 2 (the feed's round-robin rule)"
            )
        else:
            feed_ok += 1
    if feed_ok != n_b:
        problems.append(
            f"{label}: {feed_ok} down feed chains read as a (b, acc) round-robin, "
            f"need {n_b}"
        )

    verdict = "FAIL" if problems else "PASS"
    print(
        f"{TAG} tail_pipeline [{label}]: {verdict} ({len(tiled)} device, "
        f"{len(tile_herd)} cores, {total_core_core} core->core, {n_packet} "
        f"packet-typed channels, shim MM2S {mm2s_total}/{NPU2_SHIM_MM2S_PORTS} "
        f"worst column {mm2s_worst}, memtile BDs max "
        f"{max(memtile_bds.values(), default=0)}/{MEMTILE_BD_CAP}, core BDs max "
        f"{max(core_bds.values(), default=0)}/{CORE_BD_CAP}, L1 worst {l1_worst} B, "
        f"memtile 4-D BDs {mt_4d}, adders {adders}, multi-token core locks "
        f"{len(multi)}, alternating feeds {feed_ok}/{n_b})"
    )
    return problems


def main():
    # The column census's negative control first and unconditionally: if it
    # cannot refuse an over-budget design, clause 5 is not a gate.
    problems = check_census_control()
    for label, kwargs in SHAPES:
        problems += check_shape(label, kwargs)
    if problems:
        print(f"{TAG} FAIL")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print(f"{TAG} PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
