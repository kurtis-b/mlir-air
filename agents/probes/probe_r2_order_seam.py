#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Doc 31b (R2): the order seam, measured -- which on-chip hand-off shapes
survive, and which three are traps.

THE QUESTION
    R1 settled seam 1 for an L3-resident producer: ``hidden`` arrives row-major
    and the SHIM's 4-D read pattern retiles it during the per-k' refill, the
    offset advancing on the L3 side where the runtime sequence materializes one
    per task. R2's producer is a norm tail ON THE ARRAY, so that trick is gone:
    an ``aie.dma_bd`` offset is static (doc 23 "Never read a staged buffer at a
    per-iteration offset"; ``probe_ffn_accum_bd_offset.py`` is the measurement).

    Doc 31 seam 2 named the problem and predicted the resolution would be
    "re-mapping the norm tail's row->tile assignment to band order". That
    prediction is too weak, and this probe is why: a norm is a ROW-wise
    operation, so a norm tail can only ever emit whole rows, while a GEMM's A
    operand is a BLOCKED COLUMN STRIP of all its rows. No row->tile re-mapping
    turns one into the other. The re-mapping that works is on the FFN side --
    partition the GEMM herds by ROWS (M) instead of by output columns (N), so
    producer core c and consumer core c own the SAME rows and nothing is
    reordered between them. What is left is a purely LOCAL retile inside each
    consumer core, which is compute and not a BD, so the frozen-BD rule does
    not reach it.

THE ARMS
    ``row_tiles``   THE DESIGN, and the only arm asserted green. Producer core
        c emits its band ``rows_per_call`` rows at a time; consumer core c
        takes each tile with a WHOLE-BUFFER get into its own L1 buffer (no
        offsets anywhere), and builds each blocked ``[rows_per_core, tile_k]``
        A operand by an in-core vector copy that reads whichever tile buffer
        holds the row it needs. Nothing on any BD depends on an induction
        variable; nothing needs an offset into a bigger buffer.

    ``row_band``    CONTROL 1 -- the obvious form, and a SILENT MISCOMPILE.
        Same design, except the four gets land at LITERAL offsets inside one
        ``rows_per_core * emb`` L1 band. Literal offsets are exactly what the
        frozen-BD rule permits, so this looks safe. Measured: pass 029
        ``air-shrink-memref-sizes-by-access`` rewrites the band's type down to
        ONE get's size and leaves the gets' offsets and the retile's reads
        addressing the full band -- out of bounds, no error, no warning. The
        arm asserts the breakage so the trap is a measurement, not a memory.

    ``l2_staged``   CONTROL 2 -- doc 31 seam 2's other named candidate: stage
        the band in a memtile and let the up feed read it per k' step. Doc 23
        says the read offset freezes; this arm re-measures it inside R2's own
        module rather than citing J7b's.

    ``--wloop nested``  CONTROL 3 -- a COMPILER CRASH, not a design choice.
        The natural ``(group, k')`` nest makes the w_up refill's L3-side offset
        a TWO-SYMBOL ``affine.apply``. When ``air-split-l2-memref`` decides to
        split the refill's L2 buffer it calls ``tileChannelOpByFactor``, which
        composes with that apply but builds the replacement map as
        ``AffineMap::get(0, 1, add)`` -- one symbol -- and MLIR asserts
        ``willBeValidAffineMap``. Flattening the nest to one loop (the address
        is linear in ``g*k_steps + k``) gives a one-symbol map, same transfers
        in the same order, and compiles. ``--wloop flat`` is the default for
        that reason; pass ``--wloop nested`` to see the crash.

WHAT IS ASSERTED
    A. ``row_tiles`` routes (final dump has ``aie.flow``, no ``air.channel``,
       exactly one tile-bearing ``aie.device``).
    B. ``row_tiles`` has zero packet-typed channels in EVERY dump.
    C. ``row_tiles`` core->core flows == herd_x -- the producer->consumer edge
       is L1->L1 per column and nothing round-trips a memtile or L3.
    D. ``row_tiles`` keeps ``trips`` band tile buffers of the full
       ``rows_per_call * emb`` extent on each consumer core: the hand-off is
       not silently shrunk the way ``row_band`` is.
    E. CONTROL ``row_band`` MUST be broken -- its band memref must come out
       smaller than the extent its retile reads. A control that passes is a
       failed control and this probe reports FAIL.
    F. CONTROL ``l2_staged`` must NOT carry k_steps distinct memtile offsets.

HERMETIC. No NPU, no Peano, no kernel objects: aiecc writes every MLIR pass
dump before it compiles core ELFs (doc 23 section 5), so the link failing for
want of ``ffn_accum_mm.o`` costs nothing.

PROVENANCE. Every run prints the ``air-opt`` path and mtime. Queue item 6b
(shim BD exhaustion) is being fixed concurrently and changes shim BD emission,
so any structural literal taken from a pre-6b dump is PROVISIONAL: re-derive,
do not compare.

RUN
    PYTHONPATH=<repo>/build-xrt/python
    PATH=<repo>/build-xrt/bin:<sandbox>/lib/python3.12/site-packages/mlir_aie/bin:$PATH
    python3 agents/probes/probe_r2_order_seam.py
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
from collections import Counter
from pathlib import Path

COMPILE_TIMEOUT_S = 900


class _CompileTimeout(Exception):
    pass


def _on_alarm(_sig, _frame):
    subprocess.run(["pkill", "-P", str(os.getpid()), "aircc"], check=False)
    raise _CompileTimeout(f"compile exceeded {COMPILE_TIMEOUT_S}s")


_REPO = Path(__file__).resolve().parents[2]
_PE = _REPO / "programming_examples"
# transformer_layer LAST so it ends up FIRST: its builders/ must shadow
# llms/shared/builders, not the other way around (the known cwd/sys.path trap).
for _p in (str(_PE), str(_PE / "llms"), str(_PE / "transformer_layer")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from ml_dtypes import bfloat16  # noqa: E402

from air.ir import *  # noqa: E402,F403
from air.dialects.affine import apply as affine_apply  # noqa: E402
from air.dialects.air import *  # noqa: E402,F403
from air.dialects import arith  # noqa: E402
from air.dialects.arith import ConstantOp  # noqa: E402
from air.dialects.func import CallOp, FuncOp  # noqa: E402
from air.dialects.memref import AllocOp, DeallocOp, subview  # noqa: E402
from air.dialects.scf import for_, yield_  # noqa: E402
from air.dialects.vector import transfer_read, transfer_write  # noqa: E402
from air.backend.xrt import XRTBackend  # noqa: E402
from air.backend.xrt_runner import type_mapper  # noqa: E402

from builders.ffn_accum import (  # noqa: E402
    FFN_ACCUM_KERNEL_OBJ,
    FFN_ACCUM_MM_SYMBOL,
    FFN_ACCUM_TILE_K,
    FFN_ACCUM_ZERO_SYMBOL,
    MICRO,
)

range_ = for_

# The study shape, one band. herd_x is R1's (MAX_PLACEABLE_HERD_X); emb/ffn are
# baseline_768. Every derived quantity is computed and printed, never pinned.
BAND = 64
EMB = 768
FFN = 3072
HERD_X = 4
ROWS_PER_CALL = 4
VEC = MICRO  # one microtile row of bf16 -- the retile's copy granule

CHANNEL_ROWS = "r2_rows"
CHANNEL_A = "r2_afeed"
CHANNEL_B = "r2_bfeed"

ARMS = ("row_tiles", "row_band", "l2_staged")


def _map(n_syms, terms, const=0):
    """AffineMap for sum(sym_i * factor_i) + const, n_syms symbols, 0 dims."""
    e = AffineConstantExpr.get(const)
    for i, f in terms:
        e = AffineExpr.get_add(
            e, AffineExpr.get_mul(AffineSymbolExpr.get(i), AffineConstantExpr.get(f))
        )
    return AffineMap.get(0, n_syms, [e])


@module_builder
def build_r2_seam_module(
    arm,
    herd_x=HERD_X,
    band=BAND,
    emb=EMB,
    ffn=FFN,
    tile_k=FFN_ACCUM_TILE_K,
    rows_per_call=ROWS_PER_CALL,
    groups=2,
    wloop="flat",
):
    """A faithful miniature of R2's norm-tail -> up-projection seam.

    ``groups`` is how many output-column groups the up core sweeps; the study
    value is ``ffn // group_n``. The probe defaults to 2 because the seam
    mechanism is per-(group, k') and the sweep count is not what is measured.
    """
    rows_per_core = band // herd_x
    trips = rows_per_core // rows_per_call
    group_n = emb // herd_x
    k_steps = emb // tile_k
    micro_rows = rows_per_core // MICRO
    micro_cols = tile_k // MICRO
    # sub-blocks of a microtile row that one producer tile covers
    if MICRO % rows_per_call:
        raise ValueError(
            f"MICRO ({MICRO}) must divide by rows_per_call ({rows_per_call}): the "
            "retile picks its source tile buffer at codegen time, so a producer "
            "tile must not straddle a microtile row boundary"
        )
    sub_blocks = MICRO // rows_per_call

    a_elems = rows_per_core * tile_k
    b_elems = tile_k * group_n
    c_elems = rows_per_core * group_n
    tile_elems = rows_per_call * emb
    band_elems = rows_per_core * emb

    xrt_dtype = type_mapper(bfloat16)
    l3_src_ty = MemRefType.get([band, emb], xrt_dtype)
    l3_w_ty = MemRefType.get([emb * ffn], xrt_dtype)
    l3_out_ty = MemRefType.get([band, emb], xrt_dtype)

    l2_space = IntegerAttr.get(T.i32(), MemorySpace.L2)
    l2_b_ty = MemRefType.get([herd_x * b_elems], xrt_dtype, memory_space=l2_space)
    l2_band_ty = MemRefType.get([band * emb], xrt_dtype, memory_space=l2_space)

    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_tile_ty = MemRefType.get([tile_elems], xrt_dtype, memory_space=l1_space)
    l1_band_ty = MemRefType.get([band_elems], xrt_dtype, memory_space=l1_space)
    # FLAT operands for the GEMM callee: the kernel takes a base pointer over
    # the C ABI, so a declared shape is an extent and not a layout claim
    # (builders/ffn_accum.py FOOTGUNS). Flat is also what lets an in-core
    # vector copy write the operand with NO subview reaching the callee --
    # air-to-aie normalizes external signatures to identity layout, so a
    # strided-with-offset operand cannot compile (builders/norm_tail.py).
    l1_a_ty = MemRefType.get([a_elems], xrt_dtype, memory_space=l1_space)
    l1_b_ty = MemRefType.get([b_elems], xrt_dtype, memory_space=l1_space)
    l1_c_ty = MemRefType.get([c_elems], xrt_dtype, memory_space=l1_space)

    mm_func = FuncOp(
        FFN_ACCUM_MM_SYMBOL, ([l1_a_ty, l1_b_ty, l1_c_ty], []), visibility="private"
    )
    mm_func.attributes["link_with"] = StringAttr.get(FFN_ACCUM_KERNEL_OBJ)
    mm_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()
    zero_func = FuncOp(FFN_ACCUM_ZERO_SYMBOL, ([l1_c_ty], []), visibility="private")
    zero_func.attributes["link_with"] = StringAttr.get(FFN_ACCUM_KERNEL_OBJ)
    zero_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    vec_ty = VectorType.get([VEC], xrt_dtype)
    identity_map = AffineMapAttr.get(AffineMap.get_identity(1))

    staged = arm == "l2_staged"
    if staged:
        Channel(CHANNEL_A, size=[herd_x, 1])
    else:
        Channel(CHANNEL_ROWS, size=[herd_x, 1])
    Channel(CHANNEL_B, size=[herd_x, 1])

    @FuncOp.from_py_func(l3_src_ty, l3_w_ty, l3_out_ty)
    def r2_seam(a0, a1, a2):

        @launch(operands=[a0, a1, a2])
        def r2_launch(l_src, l_w, l_out):

            @segment(name="r2_seam_seg", operands=[l_src, l_w, l_out])
            def r2_seg(s_src, s_w, s_out):
                l2_b = AllocOp(l2_b_ty, [], [])
                l2_band = AllocOp(l2_band_ty, [], []) if staged else None

                if not staged:
                    # ---- PRODUCER: stands for nt1's stage_scale. Core c owns
                    # rows [c*rows_per_core, (c+1)*rows_per_core) and emits
                    # them rows_per_call at a time -- ONE L3-facing MM2S per
                    # column, its offset on the L3 side.
                    row_map = _map(2, [(0, 1), (1, rows_per_core)])

                    @herd(name="r2_prod", sizes=[herd_x, 1], operands=[s_src])
                    def prod_herd(tx, ty, _sx, _sy, h_src):
                        l1_tile = AllocOp(l1_tile_ty, [], [])
                        for t in range_(0, rows_per_core, rows_per_call):
                            row = affine_apply(row_map, [t, tx])
                            dma_memcpy_nd(
                                l1_tile,
                                h_src,
                                src_offsets=[row, 0],
                                src_sizes=[rows_per_call, emb],
                                src_strides=[emb, 1],
                            )
                            ChannelPut(CHANNEL_ROWS, l1_tile, indices=[tx, ty])
                            yield_([])
                        DeallocOp(l1_tile)

                out_row_map = _map(1, [(0, rows_per_core)])

                @herd(name="r2_up", sizes=[herd_x, 1], operands=[s_out])
                def up_herd(tx, ty, _sx, _sy, h_out):
                    c0 = ConstantOp.create_index(0)
                    cst0 = arith.ConstantOp(xrt_dtype, 0.0)
                    out_row = affine_apply(out_row_map, [tx])

                    tile_bufs = []
                    l1_band = None
                    if arm == "row_tiles":
                        # THE DESIGN: one L1 buffer per producer tile, each
                        # filled by a WHOLE-BUFFER get. No offset appears on
                        # any BD, and no buffer is bigger than the transfer
                        # that fills it -- which is what keeps
                        # air-shrink-memref-sizes-by-access a no-op (see
                        # CONTROL row_band).
                        for _t in range(trips):
                            buf = AllocOp(l1_tile_ty, [], [])
                            ChannelGet(CHANNEL_ROWS, buf, indices=[tx, ty])
                            tile_bufs.append(buf)
                    elif arm == "row_band":
                        # CONTROL 1: the obvious form -- one band, filled at
                        # literal offsets. Literal offsets are what the
                        # frozen-BD rule permits, so this LOOKS safe.
                        l1_band = AllocOp(l1_band_ty, [], [])
                        for t in range(trips):
                            ChannelGet(
                                CHANNEL_ROWS,
                                l1_band,
                                offsets=[t * tile_elems],
                                sizes=[tile_elems],
                                strides=[1],
                                indices=[tx, ty],
                            )

                    for _g in range_(0, groups):
                        l1_c = AllocOp(l1_c_ty, [], [])
                        CallOp(zero_func, [l1_c])
                        for k in range_(0, k_steps):
                            l1_a = AllocOp(l1_a_ty, [], [])
                            l1_b = AllocOp(l1_b_ty, [], [])
                            ChannelGet(CHANNEL_B, l1_b, indices=[tx, ty])
                            if staged:
                                ChannelGet(CHANNEL_A, l1_a, indices=[tx, ty])
                            else:
                                # THE IN-CORE RETILE. Blocked A operand:
                                # microtile (mi, ki), sub-row sr, takes MICRO
                                # contiguous elements from band row
                                # (mi*MICRO + sr) at column k*tile_k+ki*MICRO.
                                # mi/ki/sub-block are Python literals -- which
                                # is what lets the source TILE BUFFER be
                                # chosen at codegen time -- and r and k are
                                # induction variables of the COMPUTE, never of
                                # a BD.
                                for mi in range(micro_rows):
                                    for ki in range(micro_cols):
                                        for sb in range(sub_blocks):
                                            src_map = _map(
                                                2,
                                                [(0, emb), (1, tile_k)],
                                                ki * MICRO,
                                            )
                                            dst_base = (
                                                (mi * micro_cols + ki) * MICRO * MICRO
                                                + sb * rows_per_call * MICRO
                                            )
                                            dst_map = _map(
                                                1, [(0, MICRO)], dst_base
                                            )
                                            if arm == "row_tiles":
                                                src_buf = tile_bufs[
                                                    mi * sub_blocks + sb
                                                ]
                                            else:
                                                src_buf = l1_band
                                                src_map = _map(
                                                    2,
                                                    [(0, emb), (1, tile_k)],
                                                    ki * MICRO
                                                    + (mi * MICRO + sb * rows_per_call)
                                                    * emb,
                                                )
                                            for r in range_(0, rows_per_call):
                                                s_off = affine_apply(src_map, [r, k])
                                                d_off = affine_apply(dst_map, [r])
                                                sub_s = subview(
                                                    src_buf.result, [s_off], [VEC], [1]
                                                )
                                                sub_d = subview(
                                                    l1_a.result, [d_off], [VEC], [1]
                                                )
                                                v = transfer_read(
                                                    vec_ty,
                                                    sub_s,
                                                    [c0],
                                                    identity_map,
                                                    cst0,
                                                    [True],
                                                )
                                                transfer_write(
                                                    None,
                                                    v,
                                                    sub_d,
                                                    [c0],
                                                    identity_map,
                                                    [True],
                                                )
                                                yield_([])
                            CallOp(mm_func, [l1_a, l1_b, l1_c])
                            DeallocOp(l1_a)
                            DeallocOp(l1_b)
                            yield_([])
                        # Per-CORE destination. Writing every core to the same
                        # slice makes the store a broadcast get at launch
                        # scope, which is not the shape R2 has.
                        dma_memcpy_nd(
                            h_out,
                            l1_c,
                            dst_offsets=[out_row, 0],
                            dst_sizes=[rows_per_core, group_n],
                            dst_strides=[emb, 1],
                        )
                        DeallocOp(l1_c)
                        yield_([])
                    for buf in tile_bufs:
                        DeallocOp(buf)
                    if l1_band is not None:
                        DeallocOp(l1_band)

                up_herd.attributes["link_with"] = StringAttr.get(FFN_ACCUM_KERNEL_OBJ)

                # ---- B FEED: L3-staged per k' step, exactly R1's shape; the
                # offset advances on a LAUNCH ARGUMENT, which the runtime
                # sequence materializes per task.
                def _emit_b_refill(off):
                    dma_memcpy_nd(
                        l2_b,
                        s_w,
                        src_offsets=[off],
                        src_sizes=[herd_x * b_elems],
                        src_strides=[1],
                    )
                    for c in range(herd_x):
                        ChannelPut(
                            CHANNEL_B,
                            l2_b,
                            offsets=[c * b_elems],
                            sizes=[b_elems],
                            strides=[1],
                            indices=[c, 0],
                        )

                if wloop == "nested":
                    # CONTROL 3: the natural (group, k') nest -- a TWO-SYMBOL
                    # L3-side offset, which crashes air-split-l2-memref.
                    w_map = _map(
                        2, [(0, k_steps * herd_x * b_elems), (1, herd_x * b_elems)]
                    )
                    for g in range_(0, groups):
                        for k in range_(0, k_steps):
                            _emit_b_refill(affine_apply(w_map, [g, k]))
                            yield_([])
                        yield_([])
                else:
                    # The address is linear in (g*k_steps + k), so one loop
                    # expresses it with a ONE-SYMBOL map: same transfers, same
                    # order, same L3-side advance, and it compiles.
                    w_map = _map(1, [(0, herd_x * b_elems)])
                    for i in range_(0, groups * k_steps):
                        _emit_b_refill(affine_apply(w_map, [i]))
                        yield_([])

                if staged:
                    # CONTROL 2: the band lives in a memtile and the A feed
                    # reads it per k' step -- doc 31 seam 2's "staging bands in
                    # L2", which doc 23 says freezes the read offset.
                    dma_memcpy_nd(
                        l2_band,
                        s_src,
                        src_offsets=[0, 0],
                        src_sizes=[band, emb],
                        src_strides=[emb, 1],
                    )
                    a_map = _map(1, [(0, tile_k // MICRO)])
                    for g in range_(0, groups):
                        for k in range_(0, k_steps):
                            a_col = affine_apply(a_map, [k])
                            for c in range(herd_x):
                                ChannelPut(
                                    CHANNEL_A,
                                    l2_band,
                                    offsets=[0, a_col, 0, 0],
                                    sizes=[
                                        rows_per_core // MICRO,
                                        tile_k // MICRO,
                                        MICRO,
                                        MICRO,
                                    ],
                                    strides=[MICRO * emb, MICRO, emb, 1],
                                    indices=[c, 0],
                                )
                            yield_([])
                        yield_([])
                    DeallocOp(l2_band)

                DeallocOp(l2_b)


def _aircc_debug_dumps(module, keep_dumps=None):
    prev_cwd = os.getcwd()
    error = None
    with tempfile.TemporaryDirectory(prefix="r2-seam-probe-") as work:
        os.chdir(work)
        try:
            backend = XRTBackend(
                omit_while_true_loop=False,
                output_format="elf",
                instance_name="r2_seam",
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
                # aircc wraps the real diagnostic in a wall of stack frames;
                # surface the first line that names it.
                diag = next(
                    (
                        ln.strip()
                        for ln in text.splitlines()
                        if "error:" in ln or "op has" in ln
                    ),
                    text.splitlines()[0] if text.splitlines() else "",
                )
                error = f"{type(exc).__name__}: {diag[:300]}"
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
            if keep_dumps and os.path.isdir("air_project/debug_ir"):
                shutil.copytree("air_project/debug_ir", keep_dumps, dirs_exist_ok=True)
        finally:
            os.chdir(prev_cwd)
    return dumps, error


_TILE_RE = re.compile(r"%(\S+) = aie\.tile\((\d+),\s*(\d+)\)")
_FLOW_RE = re.compile(r"aie\.flow\(%(\S+),\s*\w+\s*:\s*\d+,\s*%(\S+),\s*\w+\s*:\s*\d+\)")
_BUF_RE = re.compile(r"%(\S+) = aie\.buffer\(%(\S+)\)[^:]*:\s*memref<(\d+)x")


def _tile_kinds(text):
    tiles = {m[0]: (int(m[1]), int(m[2])) for m in _TILE_RE.findall(text)}
    return tiles, {
        t: ("shim" if r == 0 else "memtile" if r == 1 else "core")
        for t, (c, r) in tiles.items()
    }


def _bd_offsets(text, kinds):
    """dma_bd offsets grouped by the kind of tile owning the buffer."""
    buf_tile = dict(re.findall(r"%(\S+) = aie\.buffer\(%(\S+)\)", text))
    out = {"core": [], "memtile": [], "shim": []}
    for m in re.finditer(r"aie\.dma_bd\(%([\w_]+)\s*:[^)]*offset\s*=\s*(\d+)", text):
        k = kinds.get(buf_tile.get(m.group(1)))
        if k:
            out[k].append(int(m.group(2)))
    return out


def run_arm(arm, groups, wloop="flat", keep_dumps=None):
    print(f"\n[r2-seam] ===== arm {arm} (wloop={wloop}) =====")
    try:
        module = build_r2_seam_module(arm, groups=groups, wloop=wloop)
    except Exception as exc:
        print(f"[r2-seam] {arm}: module build failed: {type(exc).__name__}: {exc}")
        return {"arm": arm, "built": False, "error": str(exc)}
    src = str(module)
    t0 = time.monotonic()
    dumps, err = _aircc_debug_dumps(module, keep_dumps=keep_dumps)
    elapsed = time.monotonic() - t0
    res = {
        "arm": arm,
        "built": True,
        "lines": len(src.splitlines()),
        "dumps": len(dumps),
        "secs": elapsed,
        "error": err,
    }
    print(f"[r2-seam] {arm}: {res['lines']} source lines -> {len(dumps)} dumps "
          f"in {elapsed:.1f}s")
    if not dumps:
        print(f"[r2-seam] {arm}: NO DUMPS ({err})")
        return res
    res["packet_dumps"] = len([n for n, t in dumps if "npu_dma_packet" in t])
    final_name, final = dumps[-1]
    res["final"] = final_name
    tiles, kinds = _tile_kinds(final)
    res["routed"] = bool(tiles) and "aie.flow" in final and "air.channel" not in final
    res["tiled_devices"] = 1 if tiles else 0
    flows = _FLOW_RE.findall(final)
    edges = Counter((kinds.get(s), kinds.get(d)) for s, d in flows)
    res["core_core"] = edges.get(("core", "core"), 0)
    offs = _bd_offsets(final, kinds)
    res["core_offsets"] = sorted(set(offs["core"]))
    res["memtile_offsets"] = sorted(set(offs["memtile"]))
    # L1 buffer census per core tile, and the biggest 1-D L1 extent that
    # survived -- clause D's instrument.
    core_bufs = Counter()
    extents = Counter()
    per_core_extent = Counter()  # (tile, extent) -> count
    l1_bytes = Counter()
    for _b, tile, size in _BUF_RE.findall(final):
        if kinds.get(tile) == "core":
            core_bufs[tile] += 1
            extents[int(size)] += 1
            per_core_extent[(tile, int(size))] += 1
            l1_bytes[tile] += int(size) * 2
    res["core_buf_counts"] = dict(sorted(Counter(core_bufs.values()).items()))
    res["l1_extents"] = dict(sorted(extents.items()))
    res["per_core_extent"] = per_core_extent
    res["max_l1_bytes"] = max(l1_bytes.values(), default=0)
    print(f"[r2-seam] {arm}: worst core L1 {res['max_l1_bytes']} B of 65536 "
          f"({len(l1_bytes)} core tiles)")
    print(f"[r2-seam] {arm}: routed={res['routed']}, packet dumps="
          f"{res['packet_dumps']}, flows {dict(edges)}")
    print(f"[r2-seam] {arm}: L1 extents (elements x count) {res['l1_extents']}")
    print(f"[r2-seam] {arm}: core dma_bd offsets {res['core_offsets']}, "
          f"memtile {res['memtile_offsets']}")
    if err:
        print(f"[r2-seam] {arm}: compile ended: {err.splitlines()[0][:160]}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=None, choices=list(ARMS))
    ap.add_argument("--groups", type=int, default=2)
    ap.add_argument("--wloop", default="flat", choices=["nested", "flat"])
    ap.add_argument("--keep-dumps", default=None)
    ns = ap.parse_args()

    air_opt = shutil.which("air-opt")
    stamp = (
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(air_opt)))
        if air_opt
        else "NOT ON PATH"
    )
    print(f"[r2-seam] air-opt: {air_opt} (mtime {stamp})")
    rows_per_core = BAND // HERD_X
    trips = rows_per_core // ROWS_PER_CALL
    print(f"[r2-seam] band {BAND}x{FFN}x{EMB}, herd_x {HERD_X}, rows/core "
          f"{rows_per_core}, {trips} producer tiles of {ROWS_PER_CALL} rows, "
          f"tile_k {FFN_ACCUM_TILE_K}, groups {ns.groups}")
    print("[r2-seam] PROVISIONAL pending queue item 6b: the shim BD fix changes "
          "shim BD emission; re-derive, do not compare.")

    arms = [ns.arm] if ns.arm else list(ARMS)
    results = {a: run_arm(a, ns.groups, ns.wloop,
                          keep_dumps=ns.keep_dumps if ns.arm else None) for a in arms}

    problems = []
    d = results.get("row_tiles")
    if d and d.get("built"):
        if not d.get("routed"):
            problems.append(f"row_tiles did not route ({d.get('error')})")
        if d.get("packet_dumps"):
            problems.append(f"row_tiles: packet-typed channels in "
                            f"{d['packet_dumps']} dump(s) -- a column budget is over")
        if d.get("core_core") != HERD_X:
            problems.append(f"row_tiles: {d.get('core_core')} core->core flows, "
                            f"expected {HERD_X} (one producer->consumer edge per "
                            "column, L1->L1)")
        # Clause D, counted PER CORE. At the study shape the C accumulator
        # (rows_per_core * group_n) happens to have the same element count as a
        # band tile (rows_per_call * emb), so a whole-module total would pass on
        # a coincidence; requiring `trips` on each of `herd_x` distinct core
        # tiles does not.
        want = ROWS_PER_CALL * EMB
        pce = d.get("per_core_extent", Counter())
        cores_ok = [t for (t, e), n in pce.items() if e == want and n >= trips]
        if len(cores_ok) < HERD_X:
            problems.append(
                f"row_tiles: only {len(cores_ok)} core tiles carry {trips} buffers "
                f"of the {want}-element band-tile extent, expected {HERD_X} -- the "
                "hand-off buffers were merged, shrunk or rewritten")
    b = results.get("row_band")
    if b is not None and b.get("built") and b.get("routed"):
        want = rows_per_core * EMB
        if b.get("l1_extents", {}).get(want, 0):
            problems.append(
                f"CONTROL row_band did NOT fail: an L1 buffer of the full "
                f"{want}-element band survived. air-shrink-memref-sizes-by-access "
                "was measured to shrink it to one get's size while leaving the "
                "reads addressing the whole band; if that is fixed, re-derive this "
                "probe's design rather than trusting either arm")
    st = results.get("l2_staged")
    if st is not None and st.get("built"):
        k_steps = EMB // FFN_ACCUM_TILE_K
        if st.get("routed") and len(st.get("memtile_offsets", [])) >= k_steps:
            problems.append(
                f"CONTROL l2_staged did NOT fail: it routed with "
                f"{len(st['memtile_offsets'])} distinct memtile offsets against "
                f"{k_steps} k' steps -- doc 23's frozen-BD rule would be refuted, "
                "which is a bigger finding than this probe; re-check before "
                "believing it")
        # Measured 2026-08-11: it does not even reach the frozen-BD question --
        # 'aie.memtile_dma' op has more than 48 blocks, the memtile analogue of
        # wall 4 (getNumBDs(MemTile) = 48 against the shim's 16). Loud, not
        # silent, which is the failure shape H9/J1 prefer.
        print(f"[r2-seam] l2_staged control refused with: {st.get('error')}")

    print()
    for a in arms:
        r = results[a]
        print(f"[r2-seam] {a:11s} built={r.get('built')} routed={r.get('routed')} "
              f"core->core={r.get('core_core')} "
              f"L1 extents={r.get('l1_extents')} secs={r.get('secs', 0):.1f}")
    if problems:
        print()
        for p in problems:
            print(f"[r2-seam] {p}")
        print("[r2-seam] FAIL")
        return 1
    print("[r2-seam] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
