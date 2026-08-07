#!/usr/bin/env python3
"""Does air-hoist-dma-in-accum-pattern collapse the accumulator round-trip?

Builds a K-loop that fetches an L1 accumulator, calls an external matmul, and
stores it back -- in two shapes:

  inplace   one L1 buffer:  dma C_l1 <- C_l3 ; call(A,B,C_l1) ; dma C_l3 <- C_l1
            (the shape of ffn_matmul_bf16_bf16_up_proj, which reads C and adds)

  twobuf    two L1 buffers: dma Acc_l1 <- C_l3 ; call(A,B,Acc_l1,C_l1) ;
                            dma C_l3 <- C_l1
            (the shape of ffn_matmul_with_acc_bf16_bf16_down_proj, iron's
             of_o_acc_in / of_o_acc_out pair)

Then runs air-opt with just `air-dependency,air-hoist-dma-in-accum-pattern` and
reports whether the C DMAs are still inside the loop.

Compile-only, no NPU, no kernel object, no linking.
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
from air.dialects.memref import AllocOp, DeallocOp  # noqa: E402
from air.dialects.func import FuncOp, CallOp  # noqa: E402
from air.dialects.scf import for_, yield_  # noqa: E402
from air.backend.xrt_runner import type_mapper  # noqa: E402

range_ = for_

M, N, K, TILE_K = 32, 32, 128, 32


@module_builder
def build(two_buffer):
    xrt = type_mapper(bfloat16)
    l3_a = MemRefType.get([M, K], xrt)
    l3_b = MemRefType.get([K, N], xrt)
    l3_c = MemRefType.get([M, N], xrt)
    l1 = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_a_ty = MemRefType.get([M, TILE_K], xrt, memory_space=l1)
    l1_b_ty = MemRefType.get([TILE_K, N], xrt, memory_space=l1)
    l1_c_ty = MemRefType.get([M, N], xrt, memory_space=l1)

    sig = (
        [l1_a_ty, l1_b_ty, l1_c_ty, l1_c_ty]
        if two_buffer
        else [l1_a_ty, l1_b_ty, l1_c_ty]
    )
    mm = FuncOp("mm", (sig, []), visibility="private")
    mm.attributes["link_with"] = StringAttr.get("mm.o")
    mm.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    @FuncOp.from_py_func(l3_a, l3_b, l3_c)
    def acc(a0, a1, a2):

        @launch(operands=[a0, a1, a2])
        def lau(l_a, l_b, l_c):

            @segment(name="seg", operands=[l_a, l_b, l_c])
            def seg(s_a, s_b, s_c):

                @herd(name="h", sizes=[1, 1], operands=[s_a, s_b, s_c])
                def body(_tx, _ty, _sx, _sy, h_a, h_b, h_c):
                    l1_a = AllocOp(l1_a_ty, [], [])
                    l1_b = AllocOp(l1_b_ty, [], [])
                    l1_c = AllocOp(l1_c_ty, [], [])
                    l1_acc = AllocOp(l1_c_ty, [], []) if two_buffer else None

                    for k in range_(0, K, TILE_K):
                        dma_memcpy_nd(
                            l1_a,
                            h_a,
                            src_offsets=[0, k],
                            src_sizes=[M, TILE_K],
                            src_strides=[K, 1],
                        )
                        dma_memcpy_nd(
                            l1_b,
                            h_b,
                            src_offsets=[k, 0],
                            src_sizes=[TILE_K, N],
                            src_strides=[N, 1],
                        )
                        # The accumulator fetch. Mirrored by the store below.
                        acc_in = l1_acc if two_buffer else l1_c
                        dma_memcpy_nd(
                            acc_in,
                            h_c,
                            src_offsets=[0, 0],
                            src_sizes=[M, N],
                            src_strides=[N, 1],
                        )
                        operands = (
                            [l1_a, l1_b, l1_acc, l1_c]
                            if two_buffer
                            else [l1_a, l1_b, l1_c]
                        )
                        CallOp(mm, operands)
                        dma_memcpy_nd(
                            h_c,
                            l1_c,
                            dst_offsets=[0, 0],
                            dst_sizes=[M, N],
                            dst_strides=[N, 1],
                        )
                        yield_([])

                    DeallocOp(l1_a)
                    DeallocOp(l1_b)
                    DeallocOp(l1_c)
                    if l1_acc is not None:
                        DeallocOp(l1_acc)


@module_builder
def build_inloop(two_buffer=False):
    """Same accumulator, but every L1 buffer is allocated inside the K loop --
    the shape air-hoist-dma-in-accum-pattern's guards actually describe."""
    xrt = type_mapper(bfloat16)
    l3_a = MemRefType.get([M, K], xrt)
    l3_b = MemRefType.get([K, N], xrt)
    l3_c = MemRefType.get([M, N], xrt)
    l1 = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_a_ty = MemRefType.get([M, TILE_K], xrt, memory_space=l1)
    l1_b_ty = MemRefType.get([TILE_K, N], xrt, memory_space=l1)
    l1_c_ty = MemRefType.get([M, N], xrt, memory_space=l1)

    sig = (
        [l1_a_ty, l1_b_ty, l1_c_ty, l1_c_ty]
        if two_buffer
        else [l1_a_ty, l1_b_ty, l1_c_ty]
    )
    mm = FuncOp("mm", (sig, []), visibility="private")
    mm.attributes["link_with"] = StringAttr.get("mm.o")
    mm.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    @FuncOp.from_py_func(l3_a, l3_b, l3_c)
    def acc(a0, a1, a2):

        @launch(operands=[a0, a1, a2])
        def lau(l_a, l_b, l_c):

            @segment(name="seg", operands=[l_a, l_b, l_c])
            def seg(s_a, s_b, s_c):

                @herd(name="h", sizes=[1, 1], operands=[s_a, s_b, s_c])
                def body(_tx, _ty, _sx, _sy, h_a, h_b, h_c):
                    for k in range_(0, K, TILE_K):
                        l1_a = AllocOp(l1_a_ty, [], [])
                        l1_b = AllocOp(l1_b_ty, [], [])
                        l1_c = AllocOp(l1_c_ty, [], [])
                        l1_acc = AllocOp(l1_c_ty, [], []) if two_buffer else None
                        dma_memcpy_nd(
                            l1_a,
                            h_a,
                            src_offsets=[0, k],
                            src_sizes=[M, TILE_K],
                            src_strides=[K, 1],
                        )
                        dma_memcpy_nd(
                            l1_b,
                            h_b,
                            src_offsets=[k, 0],
                            src_sizes=[TILE_K, N],
                            src_strides=[N, 1],
                        )
                        acc_in = l1_acc if two_buffer else l1_c
                        dma_memcpy_nd(
                            acc_in,
                            h_c,
                            src_offsets=[0, 0],
                            src_sizes=[M, N],
                            src_strides=[N, 1],
                        )
                        CallOp(
                            mm,
                            (
                                [l1_a, l1_b, l1_acc, l1_c]
                                if two_buffer
                                else [l1_a, l1_b, l1_c]
                            ),
                        )
                        dma_memcpy_nd(
                            h_c,
                            l1_c,
                            dst_offsets=[0, 0],
                            dst_sizes=[M, N],
                            dst_strides=[N, 1],
                        )
                        DeallocOp(l1_a)
                        DeallocOp(l1_b)
                        DeallocOp(l1_c)
                        if l1_acc is not None:
                            DeallocOp(l1_acc)
                        yield_([])


def dmas_inside_loop(mlir_text):
    """Count air dma/channel ops textually inside the scf.for body."""
    lines = mlir_text.splitlines()
    start = next((i for i, l in enumerate(lines) if "scf.for" in l), None)
    if start is None:
        return None, "no scf.for survived"
    depth, inside = 0, []
    for l in lines[start:]:
        inside.append(l)
        depth += l.count("{") - l.count("}")
        if depth <= 0 and len(inside) > 1:
            break
    body = "\n".join(inside)
    return len(re.findall(r"air\.dma_memcpy_nd|air\.channel\.(put|get)", body)), body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--shape",
        required=True,
        choices=("inplace", "twobuf", "inloop", "inloop_twobuf"),
    )
    args = ap.parse_args()

    work = tempfile.mkdtemp(prefix="probe-accum-")
    os.chdir(work)
    if args.shape == "inloop":
        mod = build_inloop(False)
    elif args.shape == "inloop_twobuf":
        mod = build_inloop(True)
    else:
        mod = build(args.shape == "twobuf")
    src = os.path.join(work, "in.mlir")
    open(src, "w").write(str(mod))

    def run(passes, out):
        r = subprocess.run(
            ["air-opt", src, f"--pass-pipeline=builtin.module({passes})", "-o", out],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print("air-opt failed:\n" + r.stderr[:800])
            sys.exit(2)
        return open(out).read()

    before = run("air-dependency", os.path.join(work, "dep.mlir"))
    after = run(
        "air-dependency,air-hoist-dma-in-accum-pattern",
        os.path.join(work, "hoisted.mlir"),
    )

    nb, _ = dmas_inside_loop(before)
    na, _ = dmas_inside_loop(after)
    print(f"shape={args.shape}")
    print(f"  data-movement ops inside the K loop: {nb} before -> {na} after")
    if nb is None or na is None:
        print("  -> INCONCLUSIVE")
    elif na < nb:
        print(f"  -> HOISTED: the pass lifted {nb - na} op(s) out of the loop")
    else:
        print("  -> NOT HOISTED: the accumulator round-trip stays, once per K step")
    print(f"  IR: {work}")


if __name__ == "__main__":
    sys.exit(main())
