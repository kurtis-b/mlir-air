#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Minimal-shape hunt: AIRFuseChannels crash on sibling same-bounds nests.

THE DEFECT THIS CHARACTERIZES
    ``air-fuse-channels`` SEGVs inside ``air::isAsyncOp(mlir::Operation*)``
    called from ``AIRFuseChannels::runOnFunction`` on doc 31 R1's module
    (``builders/ffn_resident.py``) -- nondeterministically under aircc (the
    same binary and module: 2 clean compiles, then 2 crashes, 2026-08-11),
    deterministically (10/10) under
    ``air-opt --air-fuse-channels`` on the round-tripped pre-fuse dump. The
    frame pattern matches the NFL merge path: ``runOnFunction`` collects
    ``nfl_erased_ops``, calls ``wrapRegionsWithForLoops``, then queries
    ``air::isAsyncOp(e)`` on ops the wrapping may have invalidated -- a
    use-after-free profile, which is what makes the aircc outcome an ASLR
    coin toss. (Read, not fixed: doc 31 forbids touching ``mlir/``
    in-phase; this probe exists to hand the compiler phase a minimal shape,
    the way H9 and H10 were handed theirs.)

WHAT THIS PROBE BUILDS
    The suspected trigger, isolated: a segment whose body carries N SIBLING
    scf.for nests with the SAME bounds, each holding one refill
    ``dma_memcpy_nd`` (L3 -> one shared L2 buffer, offset affine in the IV)
    plus named-channel puts to a consumer herd -- the exact shape
    ``ffn_resident``'s down feed presents after its per-sub-channel unroll
    (channel indices are compile-time, H5, so the c dimension MUST be
    textually unrolled; each textual dma instance becomes its own auto
    channel under air-dma-to-channel; N sibling same-bounds auto channels
    are then temporal-fusion candidates).

MEASURED `[2026-08-11]` -- THE BOUNDARY IS BETWEEN 2 AND 3
    ``--nests 2``: aircc completes all 59 dumps, air-opt 5/5 clean.
    ``--nests 3``: air-opt SEGV 5/5 (rc=-11), aircc crashed too that run.
    ``--nests 4``: air-opt SEGV, identical stack to the R1 module's.
    Two mergeable channels fuse cleanly; the third is what makes the
    pairwise candidate loop revisit a channel whose ops an earlier merge of
    the SAME candidate set already erased -- consistent with the
    use-after-free reading, and with 2-clean/2-crash nondeterminism under
    aircc on identical input (ASLR decides whether the freed read faults).
    A 115-line pre-fuse IR from this ~40-line builder is the reproducer;
    the R1 module is the production witness (its down feed carries herd_x=4
    such nests).

    Compiles hermetically through XRTBackend(debug_ir=True) to get the
    pre-fuse dump, then runs ``air-opt --air-fuse-channels`` on that dump
    REPEATEDLY (default 5x) and reports the outcome distribution. air-opt
    on the dump is the deterministic instrument; aircc is the coin toss.

RUN
    python3 agents/probes/probe_fuse_channels_sibling_nests.py --nests 4
    python3 agents/probes/probe_fuse_channels_sibling_nests.py --nests 2
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_PE = _REPO / "programming_examples"
for _p in (str(_PE), str(_PE / "llms"), str(_PE / "transformer_layer")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import numpy as np  # noqa: E402
from ml_dtypes import bfloat16  # noqa: E402

from air.ir import *  # noqa: E402, F403
from air.dialects.affine import apply as affine_apply  # noqa: E402
from air.dialects.air import *  # noqa: E402, F403
from air.dialects.func import FuncOp  # noqa: E402
from air.dialects.memref import AllocOp, DeallocOp  # noqa: E402
from air.dialects.scf import for_, yield_  # noqa: E402
from air.backend.xrt import XRTBackend  # noqa: E402
from air.backend.xrt_runner import type_mapper  # noqa: E402

range_ = for_

CH = "sibling_feed"
ELEMS = 1024  # one L2 staging buffer's elements
TRIPS = 6  # per-nest trip count; same bounds across nests is the point


def build_module(nests):
    @module_builder
    def _build():
        xrt_dtype = type_mapper(bfloat16)
        l3_ty = MemRefType.get([nests * TRIPS * ELEMS], xrt_dtype)
        l2_space = IntegerAttr.get(T.i32(), MemorySpace.L2)
        l2_ty = MemRefType.get([ELEMS], xrt_dtype, memory_space=l2_space)
        l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
        l1_ty = MemRefType.get([ELEMS], xrt_dtype, memory_space=l1_space)

        Channel(CH, size=[1, 1])

        @FuncOp.from_py_func(l3_ty)
        def sibling(arg0):
            @launch(operands=[arg0])
            def sb_launch(l_w):
                @segment(name="sibling_seg", operands=[l_w])
                def sb_seg(s_w):
                    l2_buf = AllocOp(l2_ty, [], [])

                    @herd(name="sibling_herd", sizes=[1, 1])
                    def consumer(_tx, _ty, _sx, _sy):
                        l1_buf = AllocOp(l1_ty, [], [])
                        for _t in range_(0, nests * TRIPS):
                            ChannelGet(CH, l1_buf, indices=[_tx, _ty])
                            yield_([])
                        DeallocOp(l1_buf)

                    # N sibling nests, same bounds, one textual refill DMA
                    # each (-> one auto channel each), puts to ONE named
                    # channel. The c dimension of ffn_resident's down feed,
                    # with everything else stripped.
                    for c in range(nests):
                        off_map = AffineMap.get(
                            0,
                            1,
                            [
                                AffineExpr.get_add(
                                    AffineExpr.get_mul(
                                        AffineSymbolExpr.get(0),
                                        AffineConstantExpr.get(ELEMS),
                                    ),
                                    AffineConstantExpr.get(c * TRIPS * ELEMS),
                                )
                            ],
                        )
                        for jj in range_(0, TRIPS):
                            off = affine_apply(off_map, [jj])
                            dma_memcpy_nd(
                                l2_buf,
                                s_w,
                                src_offsets=[off],
                                src_sizes=[ELEMS],
                                src_strides=[1],
                            )
                            ChannelPut(CH, l2_buf, indices=[0, 0])
                            yield_([])

                    DeallocOp(l2_buf)

    return _build()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nests", type=int, default=4)
    ap.add_argument("--tries", type=int, default=5)
    ap.add_argument("--keep", default=None, help="copy the pre-fuse dump here")
    ns = ap.parse_args()

    module = build_module(ns.nests)
    prev = os.getcwd()
    prefuse = None
    with tempfile.TemporaryDirectory(prefix="fuse-sibling-probe-") as work:
        os.chdir(work)
        try:
            backend = XRTBackend(
                omit_while_true_loop=False,
                output_format="elf",
                instance_name="sibling",
                runtime_loop_tiling_sizes=[2, 2],
                target_device="npu2",
                debug_ir=True,
            )
            err = None
            try:
                backend.compile(module)
            except Exception as exc:
                err = f"{type(exc).__name__}: {str(exc)[-160:]}"
            finally:
                try:
                    backend.unload()
                except Exception:
                    pass
            dumps = sorted(glob.glob("air_project/debug_ir/pass_*.mlir"))
            pre = [
                d
                for d in dumps
                if re.search(r"pass_\d+_after_cse", d)
                and int(re.search(r"pass_(\d+)_", d).group(1))
                < next(
                    (
                        int(re.search(r"pass_(\d+)_", x).group(1))
                        for x in dumps
                        if "air-fuse-channels" in x
                    ),
                    len(dumps),
                )
            ]
            n_dumps = len(dumps)
            fuse_reached = any("air-fuse-channels" in d for d in dumps)
            if pre:
                prefuse = Path(pre[-1]).read_text()
            print(
                f"[fuse-sibling-probe] nests={ns.nests}: {n_dumps} dumps, "
                f"fuse dump {'written' if fuse_reached else 'MISSING'} "
                f"(aircc: {err or 'succeeded'})"
            )
        finally:
            os.chdir(prev)

    if prefuse is None:
        print("[fuse-sibling-probe] no pre-fuse dump; nothing to measure")
        return 1
    with tempfile.NamedTemporaryFile(
        "w", suffix=".mlir", prefix="fuse-sibling-", delete=False
    ) as f:
        f.write(prefuse)
        in_path = f.name
    if ns.keep:
        Path(ns.keep).write_text(prefuse)
    airopt = str(_REPO / "install-xrt" / "bin" / "air-opt")
    if not os.path.exists(airopt):
        airopt = "air-opt"
    outcomes = {}
    trace = None
    for _ in range(ns.tries):
        r = subprocess.run(
            [airopt, "--air-fuse-channels", in_path, "-o", os.devnull],
            capture_output=True,
            text=True,
        )
        key = "ok" if r.returncode == 0 else f"rc={r.returncode}"
        outcomes[key] = outcomes.get(key, 0) + 1
        if r.returncode != 0 and trace is None:
            trace = "\n".join(r.stderr.splitlines()[:8])
    print(f"[fuse-sibling-probe] air-opt --air-fuse-channels x{ns.tries}: {outcomes}")
    if trace:
        print(trace)
    crashed = any(k != "ok" for k in outcomes)
    print(
        f"[fuse-sibling-probe] nests={ns.nests}: "
        + ("REPRODUCES the crash" if crashed else "does not reproduce")
    )
    os.unlink(in_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
