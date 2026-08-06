#!/usr/bin/env python3
"""Driver-owned fixture for Phase H's objective check. NOT part of the example.

This file lives under agents/scripts/port-loop/, which `guard_gate_files()` fingerprints and no
phase allowlist covers -- so a session that edits it to make the check easier halts the run. That
is the point: Phase H's claim is about the COMPILER, and the evidence for it must not come from
code the phase authored.

WHAT IT PROVES

`builders/addnorm.py` forbids more than one trip of its row loop. The cause is a missing dependency
edge, not ping-pong per se: `checkOpOperandReadOrWrite` (mlir/lib/Util/Util.cpp) classifies a memref
use by memory effects, ChannelPut, ChannelGet or linalg and returns 'u' otherwise, so an external
kernel `func.call` -- which registers no memory effects -- is invisible. Unknown uses are dropped,
and an empty producer/consumer set becomes a `WaitAllOp` with no operands instead of a rejection.
The ping/pong halves therefore get no reuse edge protecting a buffer until the kernel has read it.

Two variants, run at TWO trips, which is what the shipped builder refuses:

  inside   the weight DMA is inside the loop, so every L1 buffer is refilled each iteration.
           This is a LEGITIMATE program. Today it miscompiles (481-497 of 512 elements wrong).
           After H2 teaches the classifier about external calls, it must produce ZERO mismatches.

  hoisted  the weight DMA is lifted out of the loop, so `l1_w` carries data across iterations and
           rotating it is genuinely unsound. builders/addnorm.py documents this corrupting. After
           H1, the compiler must REFUSE it with a diagnostic rather than emit wrong numbers.

So `inside` proves the fix works and `hoisted` proves it still discriminates. A pass that simply
disabled ping-pong everywhere would satisfy `inside` and fail nothing -- which is why `hoisted`
exists, and why it demands a diagnostic rather than merely "not the right answer".

Usage:
    python3 addnorm_multitrip.py --variant inside      # exit 0 iff zero mismatches
    python3 addnorm_multitrip.py --variant hoisted     # exit 0 iff compilation is REFUSED

Run under: flock -x -w 1800 /tmp/mlir-air-npu.lock
"""

import argparse
import os
import shutil  # noqa: F401
import sys
import tempfile

import numpy as np
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
from air.backend.xrt_runner import XRTRunner, type_mapper  # noqa: E402

from builders.addnorm import (  # noqa: E402
    PRE_ADD_SYMBOL,
    addnorm_kernel_object,
    addnorm_pre_add_reference,
    compile_addnorm_kernel,
)

range_ = for_

# The shape builders/addnorm.py measured the miscompile at: 2 trips of 4 rows on one column.
ROWS, COLS, HERD_X, ROWS_PER_CALL = 8, 64, 1, 4
RTOL, ATOL = 1.6e-2, 1e-1


@module_builder
def build(rows, cols, herd_x, rows_per_call, hoist_weight, np_dtype=bfloat16):
    rows_per_tile = rows // herd_x
    xrt_dtype = type_mapper(np_dtype)
    l3_act_ty = MemRefType.get([rows, cols], xrt_dtype)
    l3_w_ty = MemRefType.get([cols], xrt_dtype)
    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_tile_ty = MemRefType.get([rows_per_call, cols], xrt_dtype, memory_space=l1_space)
    l1_w_ty = MemRefType.get([cols], xrt_dtype, memory_space=l1_space)
    kernel_obj = addnorm_kernel_object(True)

    fn = FuncOp(
        PRE_ADD_SYMBOL,
        ([l1_tile_ty, l1_tile_ty, l1_w_ty, l1_tile_ty, T.i32(), T.i32()], []),
        visibility="private",
    )
    fn.attributes["link_with"] = StringAttr.get(kernel_obj)
    fn.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    row_map = AffineMap.get(
        0, 2,
        [AffineExpr.get_add(
            AffineSymbolExpr.get(0),
            AffineExpr.get_mul(AffineSymbolExpr.get(1),
                               AffineConstantExpr.get(rows_per_tile)))],
    )

    @FuncOp.from_py_func(l3_act_ty, l3_act_ty, l3_w_ty, l3_act_ty)
    def addnorm(a0, a1, a2, a3):

        @launch(operands=[a0, a1, a2, a3])
        def lau(l_x, l_res, l_w, l_out):

            @segment(name="seg", operands=[l_x, l_res, l_w, l_out])
            def seg(s_x, s_res, s_w, s_out):

                @herd(name="h", sizes=[herd_x, 1], operands=[s_x, s_res, s_w, s_out])
                def body(_tx, _ty, _sx, _sy, h_x, h_res, h_w, h_out):
                    l1_x = AllocOp(l1_tile_ty, [], [])
                    l1_res = AllocOp(l1_tile_ty, [], [])
                    l1_w = AllocOp(l1_w_ty, [], [])
                    l1_out = AllocOp(l1_tile_ty, [], [])
                    c_cols = ConstantOp(T.i32(), cols)
                    c_rows = ConstantOp(T.i32(), rows_per_call)

                    # The whole difference between the two variants.
                    if hoist_weight:
                        dma_memcpy_nd(l1_w, h_w, src_offsets=[0], src_sizes=[cols],
                                      src_strides=[1])

                    for iv in range_(0, rows_per_tile, rows_per_call):
                        row = affine_apply(row_map, [iv, _tx])
                        if not hoist_weight:
                            dma_memcpy_nd(l1_w, h_w, src_offsets=[0], src_sizes=[cols],
                                          src_strides=[1])
                        dma_memcpy_nd(l1_x, h_x, src_offsets=[row, 0],
                                      src_sizes=[rows_per_call, cols], src_strides=[cols, 1])
                        dma_memcpy_nd(l1_res, h_res, src_offsets=[row, 0],
                                      src_sizes=[rows_per_call, cols], src_strides=[cols, 1])
                        CallOp(fn, [l1_x, l1_res, l1_w, l1_out, c_cols, c_rows])
                        dma_memcpy_nd(h_out, l1_out, dst_offsets=[row, 0],
                                      dst_sizes=[rows_per_call, cols], dst_strides=[cols, 1])
                        yield_([])

                    DeallocOp(l1_x)
                    DeallocOp(l1_res)
                    DeallocOp(l1_w)
                    DeallocOp(l1_out)

    return addnorm


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", required=True, choices=("inside", "hoisted"))
    args = ap.parse_args()
    hoisted = args.variant == "hoisted"

    # RUN IN A TEMP DIRECTORY, ALWAYS.
    #
    # compile_addnorm_kernel writes its .o to the CWD and aircc writes air.mlir / air.elf /
    # air_project/ there too. This file lives under agents/scripts/port-loop/, which
    # guard_gate_files() fingerprints and no phase allowlist covers -- so when a session ran it
    # from this directory, commit_step's `git add -A` committed air.mlir and addnorm_pre_add.o
    # into the driver's own tree, and the tamper check would have halted the run for it.
    #
    # 15-environment-notes.md predicted exactly this ("anything new that runs from that directory
    # will leak artifacts there too") and it was still missed when this fixture was written. A
    # caller-supplied CWD is not a fix, because the leak depends on where the caller happened to
    # stand; owning the working directory here is.
    workdir = tempfile.mkdtemp(prefix="pl-addnorm-fixture-")
    os.chdir(workdir)
    print(f"fixture workdir: {workdir}")

    compile_addnorm_kernel(pre_add=True)
    trips = (ROWS // HERD_X) // ROWS_PER_CALL
    print(f"fixture: variant={args.variant} rows={ROWS} cols={COLS} herd_x={HERD_X} "
          f"rows_per_call={ROWS_PER_CALL} -> {trips} trips")

    rng = np.random.default_rng(7)
    x = (rng.standard_normal((ROWS, COLS)) * 0.5).astype(bfloat16)
    res = (rng.standard_normal((ROWS, COLS)) * 0.5).astype(bfloat16)
    w = (rng.random((COLS,)) + 0.5).astype(bfloat16)
    expected = addnorm_pre_add_reference(x, res, w)

    try:
        module = build(ROWS, COLS, HERD_X, ROWS_PER_CALL, hoisted)
        runner = XRTRunner(verbose=False, omit_while_true_loop=False,
                           output_format="elf", instance_name="addnorm")
        rc = runner.run_test(module, inputs=[x, res, w], expected_outputs=[expected],
                             rtol=RTOL, atol=ATOL)
        compiled, correct = True, (rc == 0)
    except Exception as exc:  # noqa: BLE001 - a refusal to compile IS the expected result for one arm
        print(f"  compilation/run raised {type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:200]}")
        compiled, correct = False, False

    if hoisted:
        # The weight buffer carries data across iterations, so rotating it is unsound. The compiler
        # must say so. Silently producing ANY answer -- right or wrong -- is the failure here.
        if not compiled:
            print("  -> PASS: the compiler refused the unsound program, as it must")
            return 0
        print("  -> FAIL: the compiler accepted a program it cannot prove safe "
              f"({'and got the right answer by luck' if correct else 'and miscompiled it'}). "
              "A bail-out is the minimum; silence is what this phase exists to remove.")
        return 1

    if correct:
        print("  -> PASS: a legitimate multi-trip loop is now correct")
        return 0
    print("  -> FAIL: the legitimate multi-trip loop is still wrong "
          f"({'refused to compile' if not compiled else 'compiled but mismatched'})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
