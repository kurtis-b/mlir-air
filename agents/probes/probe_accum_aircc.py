#!/usr/bin/env python3
"""Confirm the accumulator ring at AIRCC altitude, not air-opt.

Same in-loop-alloc in-place shape probe_accum_hoist.py measured, but compiled
through XRTBackend with debug_ir=True so the dump comes from the REAL pipeline.
Linking fails (no kernel object) -- irrelevant: the hoist pass runs second, long
before the link, and its dump is written either way.
"""

import glob, os, re, sys, tempfile
from ml_dtypes import bfloat16

_PE = "/home/cj/mlir-air/programming_examples"
for _p in (_PE, os.path.join(_PE, "llms"), os.path.join(_PE, "transformer_layer")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from air.ir import *
from air.dialects.air import *
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.func import FuncOp, CallOp
from air.dialects.scf import for_, yield_
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import type_mapper

range_ = for_
M, N, K, TILE_K = 32, 32, 128, 32


@module_builder
def build():
    xrt = type_mapper(bfloat16)
    l3a = MemRefType.get([M, K], xrt)
    l3b = MemRefType.get([K, N], xrt)
    l3c = MemRefType.get([M, N], xrt)
    l1 = IntegerAttr.get(T.i32(), MemorySpace.L1)
    ta = MemRefType.get([M, TILE_K], xrt, memory_space=l1)
    tb = MemRefType.get([TILE_K, N], xrt, memory_space=l1)
    tc = MemRefType.get([M, N], xrt, memory_space=l1)
    mm = FuncOp("mm", ([ta, tb, tc], []), visibility="private")
    mm.attributes["link_with"] = StringAttr.get("mm.o")
    mm.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    @FuncOp.from_py_func(l3a, l3b, l3c)
    def acc(a0, a1, a2):
        @launch(operands=[a0, a1, a2])
        def lau(la, lb, lc):
            @segment(name="s", operands=[la, lb, lc])
            def seg(sa, sb, sc):
                @herd(name="h", sizes=[1, 1], operands=[sa, sb, sc])
                def hb(tx, ty, sx, sy, ha, hb_, hc):
                    for k in range_(0, K, TILE_K):
                        b1 = AllocOp(ta, [], [])
                        b2 = AllocOp(tb, [], [])
                        b3 = AllocOp(tc, [], [])
                        dma_memcpy_nd(
                            b1,
                            ha,
                            src_offsets=[0, k],
                            src_sizes=[M, TILE_K],
                            src_strides=[K, 1],
                        )
                        dma_memcpy_nd(
                            b2,
                            hb_,
                            src_offsets=[k, 0],
                            src_sizes=[TILE_K, N],
                            src_strides=[N, 1],
                        )
                        dma_memcpy_nd(
                            b3,
                            hc,
                            src_offsets=[0, 0],
                            src_sizes=[M, N],
                            src_strides=[N, 1],
                        )
                        CallOp(mm, [b1, b2, b3])
                        dma_memcpy_nd(
                            hc,
                            b3,
                            dst_offsets=[0, 0],
                            dst_sizes=[M, N],
                            dst_strides=[N, 1],
                        )
                        DeallocOp(b1)
                        DeallocOp(b2)
                        DeallocOp(b3)
                        yield_([])

                hb.attributes["link_with"] = StringAttr.get("mm.o")


w = tempfile.mkdtemp(prefix="accum-aircc-")
os.chdir(w)
mod = build()
be = XRTBackend(
    verbose=False,
    omit_while_true_loop=False,
    output_format="elf",
    instance_name="acc",
    debug_ir=True,
)
try:
    be.compile(mod)
    print("compiled (unexpected -- no kernel object)")
except Exception as e:
    print(f"link failed as expected: {type(e).__name__}")
finally:
    try:
        be.unload()
    except Exception:
        pass

dumps = sorted(glob.glob("air_project/debug_ir/*hoist-dma-in-accum*.mlir"))
print(f"aircc dumps for the hoist pass: {len(dumps)}")


def in_loop(text):
    lines = text.splitlines()
    i = next((j for j, l in enumerate(lines) if "scf.for" in l), None)
    if i is None:
        return None
    depth, body = 0, []
    for l in lines[i:]:
        body.append(l)
        depth += l.count("{") - l.count("}")
        if depth <= 0 and len(body) > 1:
            break
    return len(
        re.findall(r"air\.dma_memcpy_nd|air\.channel\.(put|get)", "\n".join(body))
    )


prev = sorted(glob.glob("air_project/debug_ir/*.mlir"))
for d in dumps:
    idx = prev.index(d)
    before = open(prev[idx - 1]).read() if idx else None
    after = open(d).read()
    print(f"  {os.path.basename(d)}")
    if before is not None:
        print(
            f"    data-movement ops inside the loop: {in_loop(before)} -> {in_loop(after)}"
        )
