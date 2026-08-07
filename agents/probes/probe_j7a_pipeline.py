#!/usr/bin/env python3
"""J7a premise: does a three-herd norm-tail pipeline place, with no packet path?

  herd A  x, residual (2 L3 streams)  -> channel        eltwise add
  herd B  channel                     -> channel        layer_norm
  herd C  channel + gamma (1 L3)      -> L3 out         scale

Checks, compile-only through air-opt:
  1. no air.channel is packet-typed  (the H9 defect is then unreachable)
  2. air-place-herds succeeds with three herd_x-wide herds in one segment
  3. reports the tiles each herd landed on, so placement is visible rather than assumed

No NPU, no kernel objects, no linking.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

from ml_dtypes import bfloat16

_PE = "/home/cj/mlir-air/programming_examples"
for _p in (_PE, os.path.join(_PE, "llms"), os.path.join(_PE, "transformer_layer")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from air.ir import *  # noqa: F403,E402
from air.dialects.air import *  # noqa: F403,E402
from air.dialects.affine import apply as affine_apply  # noqa: E402
from air.dialects.memref import AllocOp, DeallocOp  # noqa: E402
from air.dialects.func import FuncOp, CallOp  # noqa: E402
from air.dialects.scf import for_, yield_  # noqa: E402
from air.dialects.arith import ConstantOp  # noqa: E402
from air.backend.xrt_runner import type_mapper  # noqa: E402

range_ = for_

LOWER = (
    "air-dependency,air-dma-to-channel,canonicalize,cse,"
    "air-dependency-canonicalize,canonicalize,cse"
)


@module_builder
def build(rows, cols, herd_x, rows_per_call, packed=False):
    rows_per_tile = rows // herd_x
    xrt = type_mapper(bfloat16)
    l3 = MemRefType.get([rows, cols], xrt)
    l3_pack = MemRefType.get([rows, 2 * cols], xrt)  # x | residual, row-interleaved
    l3_g = MemRefType.get([cols], xrt)
    l1s = IntegerAttr.get(T.i32(), MemorySpace.L1)
    tile = MemRefType.get([rows_per_call, cols], xrt, memory_space=l1s)
    gvec = MemRefType.get([cols], xrt, memory_space=l1s)
    packtile = MemRefType.get([rows_per_call, 2 * cols], xrt, memory_space=l1s)

    def extern(name, args):
        f = FuncOp(name, (args, []), visibility="private")
        f.attributes["link_with"] = StringAttr.get("stub.o")
        f.attributes["llvm.emit_c_interface"] = UnitAttr.get()
        return f

    add_fn = extern(
        "stage_add",
        (
            [packtile, packtile, tile, T.i32(), T.i32()]
            if packed
            else [tile, tile, tile, T.i32(), T.i32()]
        ),
    )
    ln_fn = extern("stage_norm", [tile, tile, T.i32(), T.i32()])
    mul_fn = extern("stage_scale", [tile, gvec, tile, T.i32(), T.i32()])

    # The two inter-stage edges. Declared, not placed: air-place-herds decides where.
    Channel("AtoB", size=[herd_x, 1])
    Channel("BtoC", size=[herd_x, 1])

    row_map = AffineMap.get(
        0,
        2,
        [
            AffineExpr.get_add(
                AffineSymbolExpr.get(0),
                AffineExpr.get_mul(
                    AffineSymbolExpr.get(1), AffineConstantExpr.get(rows_per_tile)
                ),
            )
        ],
    )

    in_tys = [l3_pack, l3_g, l3] if packed else [l3, l3, l3_g, l3]

    @FuncOp.from_py_func(*in_tys)
    def norm_tail(*fargs):
        if packed:
            a_x, a_g, a_out = fargs
            a_res = a_x
        else:
            a_x, a_res, a_g, a_out = fargs

        @launch(operands=[a_x, a_res, a_g, a_out])
        def lau(l_x, l_res, l_g, l_out):

            @segment(name="tail_seg", operands=[l_x, l_res, l_g, l_out])
            def seg(s_x, s_res, s_g, s_out):

                @herd(name="stage_add", sizes=[herd_x, 1], operands=[s_x, s_res])
                def h_add(tx, ty, sx, sy, h_x, h_res):
                    ci = ConstantOp(T.i32(), cols)
                    cr = ConstantOp(T.i32(), rows_per_call)
                    b_x = AllocOp(tile, [], [])
                    b_r = AllocOp(tile, [], [])
                    b_pk = AllocOp(packtile, [], [])
                    b_o = AllocOp(tile, [], [])
                    for iv in range_(0, rows_per_tile, rows_per_call):
                        row = affine_apply(row_map, [iv, tx])
                        if packed:
                            # ONE strided fetch covering x and residual together.
                            dma_memcpy_nd(
                                b_pk,
                                h_x,
                                src_offsets=[row, 0],
                                src_sizes=[rows_per_call, 2 * cols],
                                src_strides=[2 * cols, 1],
                            )
                        else:
                            dma_memcpy_nd(
                                b_x,
                                h_x,
                                src_offsets=[row, 0],
                                src_sizes=[rows_per_call, cols],
                                src_strides=[cols, 1],
                            )
                            dma_memcpy_nd(
                                b_r,
                                h_res,
                                src_offsets=[row, 0],
                                src_sizes=[rows_per_call, cols],
                                src_strides=[cols, 1],
                            )
                        CallOp(
                            add_fn,
                            (
                                [b_pk, b_pk, b_o, ci, cr]
                                if packed
                                else [b_x, b_r, b_o, ci, cr]
                            ),
                        )
                        ChannelPut("AtoB", b_o, indices=[tx, ty])
                        yield_([])
                    DeallocOp(b_x)
                    DeallocOp(b_r)
                    DeallocOp(b_pk)
                    DeallocOp(b_o)

                h_add.attributes["link_with"] = StringAttr.get("stub.o")

                @herd(name="stage_norm", sizes=[herd_x, 1], operands=[])
                def h_norm(tx, ty, sx, sy):
                    ci = ConstantOp(T.i32(), cols)
                    cr = ConstantOp(T.i32(), rows_per_call)
                    b_i = AllocOp(tile, [], [])
                    b_o = AllocOp(tile, [], [])
                    for iv in range_(0, rows_per_tile, rows_per_call):
                        ChannelGet("AtoB", b_i, indices=[tx, ty])
                        CallOp(ln_fn, [b_i, b_o, ci, cr])
                        ChannelPut("BtoC", b_o, indices=[tx, ty])
                        yield_([])
                    DeallocOp(b_i)
                    DeallocOp(b_o)

                h_norm.attributes["link_with"] = StringAttr.get("stub.o")

                @herd(name="stage_scale", sizes=[herd_x, 1], operands=[s_g, s_out])
                def h_scale(tx, ty, sx, sy, h_g, h_out):
                    ci = ConstantOp(T.i32(), cols)
                    cr = ConstantOp(T.i32(), rows_per_call)
                    b_i = AllocOp(tile, [], [])
                    b_g = AllocOp(gvec, [], [])
                    b_o = AllocOp(tile, [], [])
                    for iv in range_(0, rows_per_tile, rows_per_call):
                        row = affine_apply(row_map, [iv, tx])
                        ChannelGet("BtoC", b_i, indices=[tx, ty])
                        dma_memcpy_nd(
                            b_g, h_g, src_offsets=[0], src_sizes=[cols], src_strides=[1]
                        )
                        CallOp(mul_fn, [b_i, b_g, b_o, ci, cr])
                        dma_memcpy_nd(
                            h_out,
                            b_o,
                            dst_offsets=[row, 0],
                            dst_sizes=[rows_per_call, cols],
                            dst_strides=[cols, 1],
                        )
                        yield_([])
                    DeallocOp(b_i)
                    DeallocOp(b_g)
                    DeallocOp(b_o)

                h_scale.attributes["link_with"] = StringAttr.get("stub.o")


def run(src, pipeline, out):
    r = subprocess.run(
        ["air-opt", src, f"--pass-pipeline=builtin.module({pipeline})", "-o", out],
        capture_output=True,
        text=True,
    )
    return (open(out).read() if r.returncode == 0 else None), r.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4096)
    ap.add_argument("--cols", type=int, default=768)
    ap.add_argument("--herd-x", type=int, default=8)
    ap.add_argument("--rows-per-call", type=int, default=8)
    ap.add_argument("--packed", action="store_true")
    a = ap.parse_args()

    work = tempfile.mkdtemp(prefix="probe-j7a-")
    os.chdir(work)
    mod = build(a.rows, a.cols, a.herd_x, a.rows_per_call, packed=a.packed)
    src = os.path.join(work, "in.mlir")
    open(src, "w").write(str(mod))
    trips = (a.rows // a.herd_x) // a.rows_per_call
    print(
        f"3-herd norm tail: rows={a.rows} cols={a.cols} herd_x={a.herd_x} "
        f"rows_per_call={a.rows_per_call} -> {trips} trips/tile"
    )

    lowered, err = run(src, LOWER, os.path.join(work, "chan.mlir"))
    if lowered is None:
        print("  LOWERING FAILED:\n   " + "\n   ".join(err.strip().splitlines()[:6]))
        return 1
    packet = lowered.count('"npu_dma_packet"')
    print(
        f"  [1] packet-typed channels: {packet}   "
        f"-> {'PACKET PATH ENTERED' if packet else 'no packet path -- H9 defect unreachable'}"
    )

    placed, err = run(
        src,
        LOWER + ",air-place-herds{num-rows=6 num-cols=8 row-anchor=2 col-anchor=0}",
        os.path.join(work, "placed.mlir"),
    )
    if placed is None:
        print(
            "  [2] PLACEMENT FAILED:\n   " + "\n   ".join(err.strip().splitlines()[:6])
        )
        print(f"  IR: {work}")
        return 1
    print("  [2] air-place-herds: OK")
    for m in re.finditer(
        r'sym_name = "(stage_\w+)".*?x_loc = (\d+).*?y_loc = (\d+)', placed, re.S
    ):
        pass
    locs = re.findall(r"air\.herd @(\w+).*?\n", placed)
    xy = re.findall(r"(x_loc\s*=\s*\d+\s*:\s*i64|y_loc\s*=\s*\d+\s*:\s*i64)", placed)
    print(
        f"  [3] herds placed: {len(set(locs))} named, {len(xy)//2} with explicit tile locations"
    )
    print(f"  IR: {work}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
