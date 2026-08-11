#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Doc 31b (R2): the two budgets a one-segment resident tail has to fit --
how many herds place, and how many shim MM2S ports each column really drives.

WHY THIS EXISTS
    R2 attaches nt1 and nt2 to R1's FFN interior. Doc 31 sized the risk as the
    per-column L3 budget and predicted "the standalone widths do NOT fit
    together". Two things had never been measured:

    1. THE HERD BUDGET. NPU2 is 8 columns x 4 CORE ROWS = 32 core tiles
       (AIETargetModel.h: BaseNPU2TargetModel::rows() = 6 -- 1 shim, 1 memtile,
       4 core -- and NPU2TargetModel::columns() = 8). J7a's norm tail is THREE
       herds; R1's interior is three more. Nine herds of herd_x = 4 want 36
       tiles. Whether ``air-place-herds`` refuses at 8 or at 9, and how, decides
       whether R2 can use J7a's pipelines verbatim or must fuse stages. Arm
       ``herds`` sweeps N and reports the first N that does not place.

    2. THE SHIM BUDGET, COUNTED WHERE IT LIVES. Doc 23's rule is two L3-facing
       MM2S per COLUMN across the whole segment. R1's own structural check
       (``probe_ffn_resident_interior.py`` clause F, doc 31 gate arm (c)) counts
       ``shim -> core`` flows only -- but an L2-staged refill is a
       ``shim -> memtile`` flow and consumes a shim MM2S port just the same.
       Measured on R1's routed design: 4 shim->core AND 3 shim->memtile, so the
       arm reports "max 1 per column" while column 1's shim actually drives 2.
       Not a violation there; a blind spot everywhere. Arm ``shim`` recounts
       BOTH directions per column on any module it is given, and reports the
       memtile port occupancy beside it (a memtile has 6 MM2S / 6 S2MM, and R1
       already sits at 5 of 6 S2MM on one of them).

    3. THE SYMBOL BUDGET. R1 found that ``-D``-baked symbols cannot coexist
       twice in one module. R2 changes ``DIM_M`` (the row partition) and adds a
       norm-tail object, so both halves need re-checking: does the microkernel
       accept ``DIM_M = band / herd_x``, and which GLOBAL symbols do the
       objects actually share? Arm ``objects`` compiles them and reads the
       symbol tables instead of trusting the source comments -- which turns out
       to matter, because one of those comments is wider than the truth.

ARMS
    ``herds``  N herds of [herd_x, 1] in one segment, each a trivial L1->L1
        stage, N swept. Reports placement and the row/column map. The first
        refusing N is the budget, and it is the negative control: an arm that
        places every N up to 12 would mean the array is bigger than the target
        model says and this probe is measuring nothing.
    ``shim``   Full per-column shim MM2S / S2MM census of R1's own
        ``build_ffn_resident_module``, printed beside what R1's shipped clause
        counts, so the undercount is a diff and not an assertion.
    ``objects``  Compiles the kernel objects R2 needs at the row-partitioned
        tiles and diffs their global symbol tables. Needs PEANO_INSTALL_DIR
        (the sandbox's ``llvm-aie``); skips loudly without it. Its negative
        control is ``DIM_M = MICRO``, which the source static_asserts against
        and which must be REFUSED -- if it compiled, ``DIM_M % 16`` would not
        be the constraint that forces every R2 herd to width 4.

HERMETIC. No NPU, no Peano, no kernel objects.
PROVENANCE. Prints the air-opt path and mtime. Item 6b changes shim BD
emission; anything derived here is PROVISIONAL until it lands.

RUN
    python3 agents/probes/probe_r2_segment_budget.py            # both arms
    python3 agents/probes/probe_r2_segment_budget.py --arm herds --max-herds 12
"""

import argparse
import glob
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

COMPILE_TIMEOUT_S = 900


def _on_alarm(_sig, _frame):
    subprocess.run(["pkill", "-P", str(os.getpid()), "aircc"], check=False)
    raise TimeoutError(f"compile exceeded {COMPILE_TIMEOUT_S}s")


_REPO = Path(__file__).resolve().parents[2]
_PE = _REPO / "programming_examples"
for _p in (str(_PE), str(_PE / "llms"), str(_PE / "transformer_layer")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from ml_dtypes import bfloat16  # noqa: E402

from air.ir import *  # noqa: E402,F403
from air.dialects.affine import apply as affine_apply  # noqa: E402
from air.dialects.air import *  # noqa: E402,F403
from air.dialects.memref import AllocOp, DeallocOp  # noqa: E402
from air.dialects.scf import for_, yield_  # noqa: E402
from air.dialects.func import FuncOp  # noqa: E402
from air.backend.xrt import XRTBackend  # noqa: E402
from air.backend.xrt_runner import type_mapper  # noqa: E402

from builders.ffn_accum import (  # noqa: E402
    FFN_ACCUM_HERD_X,
    FFN_ACCUM_TILE_K,
    MICRO,
    TILE_M,
)
from builders.ffn_resident import build_ffn_resident_module  # noqa: E402

range_ = for_

# From mlir_aie's AIETargetModel.h, quoted rather than assumed:
#   BaseNPU2TargetModel::rows() = 6   (1 shim + 1 memtile + 4 core)
#   NPU2TargetModel::columns()  = 8
#   getNumBDs: MemTile 48, everything else 16
#   getBDMaxDims: MemTile 4, core and shim 3
NPU2_COLUMNS = 8
NPU2_CORE_ROWS = 4
NPU2_CORE_TILES = NPU2_COLUMNS * NPU2_CORE_ROWS

_TILE_RE = re.compile(r"%(\S+) = aie\.tile\((\d+),\s*(\d+)\)")
_FLOW_RE = re.compile(
    r"aie\.flow\(%(\S+),\s*(\w+)\s*:\s*(\d+),\s*%(\S+),\s*(\w+)\s*:\s*(\d+)\)"
)


@module_builder
def build_n_herd_module(n_herds, herd_x=FFN_ACCUM_HERD_X, elems=1024):
    """N herds of [herd_x, 1] chained L1->L1, one segment, nothing else.

    Deliberately minimal: each stage gets a tile from the stage before and puts
    it on to the next, so the module has N herds, N+1 channels, ONE L3-facing
    stream in and one out, and no weights. If this does not place, nothing with
    N herds places -- the arm isolates the HERD budget from every other one.
    """
    xrt_dtype = type_mapper(bfloat16)
    l3_ty = MemRefType.get([herd_x * elems], xrt_dtype)
    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_ty = MemRefType.get([elems], xrt_dtype, memory_space=l1_space)

    names = [f"nb_chan_{i}" for i in range(n_herds - 1)]
    for nm in names:
        Channel(nm, size=[herd_x, 1])

    # Column tx owns its own L3 slice at both ends. Giving every core the same
    # slice makes the inbound DMA a broadcast and the outbound one a FAN-IN,
    # which aie.flow does not support -- the module then fails for a reason
    # that has nothing to do with the herd budget this arm measures.
    off_map = AffineMap.get(
        0,
        1,
        [
            AffineExpr.get_mul(
                AffineSymbolExpr.get(0), AffineConstantExpr.get(elems)
            )
        ],
    )

    @FuncOp.from_py_func(l3_ty, l3_ty)
    def n_herds_fn(a_in, a_out):

        @launch(operands=[a_in, a_out])
        def nh_launch(l_in, l_out):

            @segment(name="n_herd_seg", operands=[l_in, l_out])
            def nh_seg(s_in, s_out):
                for i in range(n_herds):
                    first, last = i == 0, i == n_herds - 1
                    ops = [s_in] if first else ([s_out] if last else [])

                    def _stage(tx, ty, _sx, _sy, *h_args, _i=i,
                               _first=first, _last=last):
                        buf = AllocOp(l1_ty, [], [])
                        if _first:
                            dma_memcpy_nd(
                                buf,
                                h_args[0],
                                src_offsets=[affine_apply(off_map, [tx])],
                                src_sizes=[elems],
                                src_strides=[1],
                            )
                        else:
                            ChannelGet(names[_i - 1], buf, indices=[tx, ty])
                        if _last:
                            dma_memcpy_nd(
                                h_args[0],
                                buf,
                                dst_offsets=[affine_apply(off_map, [tx])],
                                dst_sizes=[elems],
                                dst_strides=[1],
                            )
                        else:
                            ChannelPut(names[_i], buf, indices=[tx, ty])
                        DeallocOp(buf)

                    herd(name=f"nb_stage_{i}", sizes=[herd_x, 1], operands=ops)(_stage)


def _aircc_dumps(module, instance):
    prev_cwd = os.getcwd()
    error = None
    with tempfile.TemporaryDirectory(prefix="r2-budget-probe-") as work:
        os.chdir(work)
        try:
            backend = XRTBackend(
                omit_while_true_loop=False,
                output_format="elf",
                instance_name=instance,
                runtime_loop_tiling_sizes=[2, 2],
                target_device="npu2",
                debug_ir=True,
            )
            signal.signal(signal.SIGALRM, _on_alarm)
            signal.alarm(COMPILE_TIMEOUT_S)
            try:
                backend.compile(module)
            except Exception as exc:
                text = str(exc)
                diag = next(
                    (
                        ln.strip()
                        for ln in text.splitlines()
                        if "error:" in ln or "op has" in ln
                    ),
                    (text.splitlines() or [""])[0],
                )
                error = diag[:300]
            finally:
                signal.alarm(0)
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


def _placement(dumps):
    """Tile map of the last dump that has any, plus the herd-placement dump."""
    for name, text in reversed(dumps):
        tiles = _TILE_RE.findall(text)
        if tiles:
            rows = defaultdict(list)
            for _t, c, r in tiles:
                rows[int(r)].append(int(c))
            return name, {r: sorted(set(v)) for r, v in sorted(rows.items())}, text
    return None, {}, None


def arm_herds(max_herds, herd_x):
    print(f"\n[r2-budget] ===== arm herds (herd_x={herd_x}) =====")
    print(f"[r2-budget] NPU2 is {NPU2_COLUMNS} columns x {NPU2_CORE_ROWS} core rows "
          f"= {NPU2_CORE_TILES} core tiles (AIETargetModel.h)")
    first_fail = None
    for n in range(2, max_herds + 1):
        module = build_n_herd_module(n, herd_x=herd_x)
        t0 = time.monotonic()
        dumps, err = _aircc_dumps(module, f"n_herd_{n}")
        el = time.monotonic() - t0
        name, rows, text = _placement(dumps)
        core_rows = {r: v for r, v in rows.items() if r >= 2}
        placed = sum(len(v) for v in core_rows.values())
        ok = placed == n * herd_x
        print(f"[r2-budget]   N={n:2d} ({n * herd_x:2d} tiles wanted): "
              f"placed={placed:2d} rows={core_rows} {'OK' if ok else 'MISPLACED'} "
              f"({el:.1f}s){'' if not err else '  ' + err[:110]}")
        if not ok and first_fail is None:
            first_fail = (n, err)
            break
    return first_fail


def _shim_census(text):
    tiles = {m[0]: (int(m[1]), int(m[2])) for m in _TILE_RE.findall(text)}

    def kind(t):
        r = tiles.get(t, (None, None))[1]
        return None if r is None else ("shim" if r == 0 else
                                       "memtile" if r == 1 else "core")

    def col(t):
        return tiles.get(t, (None, None))[0]

    flows = _FLOW_RE.findall(text)
    mm2s_total, mm2s_to_core = Counter(), Counter()
    s2mm = Counter()
    mt_out, mt_in = Counter(), Counter()
    for s, _sb, _sp, d, _db, _dp in flows:
        if kind(s) == "shim":
            mm2s_total[col(s)] += 1
            if kind(d) == "core":
                mm2s_to_core[col(s)] += 1
        if kind(d) == "shim":
            s2mm[col(d)] += 1
        if kind(s) == "memtile":
            mt_out[col(s)] += 1
        if kind(d) == "memtile":
            mt_in[col(d)] += 1
    return mm2s_total, mm2s_to_core, s2mm, mt_out, mt_in, flows


def arm_shim():
    print("\n[r2-budget] ===== arm shim: R1's routed design, both flow kinds =====")
    module = build_ffn_resident_module(TILE_M, 3072, 768, herd_x=FFN_ACCUM_HERD_X)
    dumps, err = _aircc_dumps(module, "ffn_resident")
    name, rows, text = _placement(dumps)
    if text is None:
        print(f"[r2-budget]   no routed dump ({err})")
        return None
    mm2s, mm2s_core, s2mm, mt_out, mt_in, flows = _shim_census(text)
    print(f"[r2-budget]   dump {name}, {len(flows)} aie.flow, tile rows {rows}")
    print("[r2-budget]   shim column | MM2S total | of which -> core | S2MM")
    over = []
    for c in range(NPU2_COLUMNS):
        t, k, s = mm2s.get(c, 0), mm2s_core.get(c, 0), s2mm.get(c, 0)
        if t or s:
            flag = "  <-- OVER the 2-per-column budget" if t > 2 else ""
            print(f"[r2-budget]     col {c}      |     {t}      |        {k}         "
                  f"|  {s}{flag}")
        if t > 2:
            over.append(c)
    print(f"[r2-budget]   TOTAL shim MM2S {sum(mm2s.values())} of "
          f"{2 * NPU2_COLUMNS} ports; worst column {max(mm2s.values(), default=0)}")
    print(f"[r2-budget]   R1's shipped clause counts shim->core ONLY: worst column "
          f"{max(mm2s_core.values(), default=0)} -- the undercount is "
          f"{max(mm2s.values(), default=0)} vs {max(mm2s_core.values(), default=0)}")
    for c in sorted(set(list(mt_out) + list(mt_in))):
        print(f"[r2-budget]   memtile col {c}: MM2S(out) {mt_out.get(c, 0)}/6, "
              f"S2MM(in) {mt_in.get(c, 0)}/6")
    return over, mm2s, mm2s_core


def _global_syms(obj):
    """Global (externally-linkable) defined symbols only.

    --extern-only matters: without it llvm-nm also lists local assembler labels
    (.LBB*, .L_LEnd*), which every object built by the same compiler shares and
    which have nothing to do with link collisions. Counting those makes every
    pair look like it collides.
    """
    out = subprocess.run(
        ["llvm-nm", "--defined-only", "--extern-only", obj],
        capture_output=True, text=True,
    )
    return {ln.split()[-1] for ln in out.stdout.splitlines() if ln.split()}


def arm_objects(band, herd_x, tile_k, emb):
    print("\n[r2-budget] ===== arm objects: the -D and symbol budget =====")
    if not os.environ.get("PEANO_INSTALL_DIR"):
        print("[r2-budget]   SKIPPED: PEANO_INSTALL_DIR unset. Set it to the sandbox "
              "llvm-aie, e.g.\n[r2-budget]   PEANO_INSTALL_DIR=<repo>/sandbox/lib/"
              "python3.12/site-packages/llvm-aie")
        return None
    import shared.infra.external_kernels as ek

    rows_per_core = band // herd_x
    group_n = emb // herd_x
    prev = os.getcwd()
    work = tempfile.mkdtemp(prefix="r2-objects-")
    os.chdir(work)
    built, refused = {}, {}
    try:
        # DIM_M sweep: the design's value, the control below it, R1's.
        for tm in (rows_per_core, MICRO, TILE_M):
            name = f"mm_m{tm}.o"
            try:
                ek.compile_encoder(tile_m=tm, tile_k=tile_k, tile_n=group_n,
                                   build_ffn=True, build_addnorm=False, out_name=name)
                built[f"DIM_M={tm}"] = name if os.path.exists(name) else None
            except Exception as exc:
                line = next((l for l in str(exc).splitlines()
                             if "static assertion" in l or "error" in l),
                            (str(exc).splitlines() or [""])[0])
                refused[f"DIM_M={tm}"] = line.strip()[:150]
        ek.compile_encoder(build_ffn=True, build_addnorm=False,
                           out_name="encoder_ffn.o")
        ek.compile_layer_norm()
        ek.compile_addnorm_ffn(build_ffn=False, build_addnorm=True, pre_add=True,
                               out_name="addnorm_pre.o")
        for k, v in built.items():
            print(f"[r2-budget]   {k:12s} BUILT   {v}")
        for k, v in refused.items():
            print(f"[r2-budget]   {k:12s} REFUSED {v}")

        objs = [f"mm_m{rows_per_core}.o", "encoder_ffn.o", "layer_norm.o",
                "addnorm_pre.o"]
        objs = [o for o in objs if os.path.exists(o)]
        syms = {o: _global_syms(o) for o in objs}
        for o in objs:
            print(f"[r2-budget]   {o}: {len(syms[o])} GLOBAL symbols")
        collisions = {}
        for i, a in enumerate(objs):
            for b in objs[i + 1:]:
                inter = sorted(syms[a] & syms[b])
                collisions[(a, b)] = inter
                print(f"[r2-budget]   {a} ^ {b}: "
                      f"{len(inter)} shared{' -> ' + ', '.join(inter[:4]) if inter else ''}"
                      f"{' ...' if len(inter) > 4 else ''}")
        print("[r2-budget]   where each symbol R2 calls is defined:")
        for s in ("ffn_matmul_bf16_bf16_up_proj", "ffn_zero_bf16_up_proj",
                  "ffn_gelu_bf16", "layer_norm_rows"):
            where = [o for o in objs if s in syms[o]]
            print(f"[r2-budget]     {s:34s} {where}")
        return built, refused, collisions, syms, objs
    finally:
        os.chdir(prev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=None, choices=["herds", "shim", "objects"])
    ap.add_argument("--band", type=int, default=TILE_M)
    ap.add_argument("--emb", type=int, default=768)
    ap.add_argument("--max-herds", type=int, default=12)
    ap.add_argument("--herd-x", type=int, default=FFN_ACCUM_HERD_X)
    ns = ap.parse_args()

    air_opt = shutil.which("air-opt")
    stamp = (
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(air_opt)))
        if air_opt
        else "NOT ON PATH"
    )
    print(f"[r2-budget] air-opt: {air_opt} (mtime {stamp})")
    print("[r2-budget] PROVISIONAL pending queue item 6b: the shim BD fix changes "
          "shim BD emission; re-derive, do not compare.")

    problems = []
    if ns.arm in (None, "herds"):
        ff = arm_herds(ns.max_herds, ns.herd_x)
        if ff is None:
            problems.append(
                f"CONTROL: every N up to {ns.max_herds} placed {ns.herd_x} tiles "
                f"each. {ns.max_herds * ns.herd_x} tiles is more than the "
                f"{NPU2_CORE_TILES} the target model declares, so either the sweep "
                "is not measuring placement or the device model changed")
        else:
            n, err = ff
            print(f"[r2-budget]   HERD BUDGET at herd_x={ns.herd_x}: "
                  f"{n - 1} herds place, {n} does not ({err or 'misplaced'})")
    if ns.arm in (None, "shim"):
        res = arm_shim()
        if res is None:
            problems.append("shim arm produced no routed dump")
        else:
            over, mm2s, mm2s_core = res
            if over:
                problems.append(f"R1 columns {over} exceed 2 shim MM2S -- doc 23's "
                                "rule is broken on the SHIPPED module")
            if max(mm2s.values(), default=0) <= max(mm2s_core.values(), default=0):
                problems.append(
                    "CONTROL: the shim->core-only count is not an undercount on this "
                    "module, so the arm demonstrates nothing -- re-check before "
                    "citing it")
    if ns.arm in (None, "objects"):
        res = arm_objects(ns.band, ns.herd_x, FFN_ACCUM_TILE_K, ns.emb)
        if res is not None:
            built, refused, _coll, _syms, _objs = res
            rows_per_core = ns.band // ns.herd_x
            if f"DIM_M={rows_per_core}" not in built:
                problems.append(
                    f"the row-partitioned microkernel does NOT build at "
                    f"DIM_M={rows_per_core}: {refused.get(f'DIM_M={rows_per_core}')} "
                    "-- the whole row partition rests on this")
            if f"DIM_M={MICRO}" not in refused:
                problems.append(
                    f"CONTROL: DIM_M={MICRO} BUILT. The kernel's DIM_M % 16 assert is "
                    "what forces every R2 herd to width 4; if it no longer holds, "
                    "re-derive the herd width rather than inheriting it")

    print()
    if problems:
        for p in problems:
            print(f"[r2-budget] {p}")
        print("[r2-budget] FAIL")
        return 1
    print("[r2-budget] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
