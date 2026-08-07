#!/usr/bin/env python3
"""Driver-owned fixture for the ping-pong safety proof. NOT part of the example.

This file lives under agents/scripts/port-loop/, which `guard_gate_files()` fingerprints and no
phase allowlist covers -- so a session that edits it to make the check easier halts the run. That
is the point: the claim is about the COMPILER, and the evidence for it must not come from code the
phase authored.

WHAT WENT WRONG, AND WHAT ACTUALLY FIXED IT
`[2026-08-06] This section replaces one that was wrong.` It blamed a missing dependency edge --
`checkOpOperandReadOrWrite` returning 'u' for an external `func.call`, an empty producer set
becoming a dependency-free `air.wait_all`, and the ping/pong halves losing their reuse edge.
Measurement disproved it: compiling the `inside` variant with `--omit-ping-pong-transform=all`
reproduced the IDENTICAL 481/512 corruption. A cause you can disable without changing the result is
not the cause.

The real defect was the shim feed order under packet multiplexing:

  1. `air-dma-to-channel` hoists each L3-side DMA into its OWN launch-scope loop, so a herd that
     fills N buffers per iteration produces N sibling per-channel put loops.
  2. Packet-multiplexed onto one shim MM2S queue (`channel_type = "npu_dma_packet"`), that queue
     serializes in task order -- whole channel after whole channel.
  3. The consuming tile's BD ring is built from the herd's per-iteration get order, so it expects
     the streams INTERLEAVED per iteration. At one trip the two orders coincide. At two or more,
     every packet after the first iteration lands in the wrong buffer.

Fixed by `air-fuse-packet-put-loops` (commit bfb647d9), which fuses sibling per-channel put loops
that share a block, share static bounds and all target packet-typed channels into one loop doing
the puts in program order -- plus modelling packet-typed channels as one shared stream resource in
`CanonicalizeAsyncOpDeps` so the token chain survives pruning.

The classifier defect was real but separate, and H2 fixed it: a callee carrying
`llvm.emit_c_interface` now classifies its memref operands from `llvm.readonly` / `llvm.writeonly`
argument attributes, and an unannotated operand stays unknown rather than being guessed at.

WHAT THE FIVE VARIANTS PROVE

All five run at TWO trips of the row loop -- the count `builders/addnorm.py` forbids. The first
four sit on ONE column, at cols=64, rows=8, rows_per_call=4, the exact shape the miscompile was
measured at; they differ in two independent bits: whether the callee's memref arguments are
annotated (so the rotation can be PROVEN safe), and whether the weight DMA sits inside the loop
(refilled every iteration) or is hoisted out of it (carrying data across iterations).

The fifth varies the thing none of the other four do: the herd WIDTH.

  variant             columns  callee        weight DMA   compiles   exact   labeled   wt rotated
  ------------------  -------  ------------  -----------  ---------  ------  --------  ----------
  inside                    1  unannotated   in loop      yes        yes     no        --
  hoisted                   1  unannotated   hoisted      yes        yes     no        --
  annotated                 1  annotated     in loop      yes        yes     YES       YES
  annotated_hoisted         1  annotated     hoisted      yes        yes     YES       NO
  multicolumn               8  unannotated   in loop      yes        yes     no        --

Each row kills a different way of passing without doing the work:

  - Every variant must COMPILE. `[2026-08-06]` H1 was originally specified as "hard-fails
    compilation with a diagnostic", and that was wrong: the prior art it cited
    (`memref::multiBuffer`, IREE, Triton, TVM) all decline the TRANSFORM and leave the code alone.
    Refusing broke three shipped models -- llama32_1b_int4, qwen3_0_6b, qwen3_1_7b -- that were
    always correct. A refusal reintroduced here fails all four rows.
  - Every variant must be NUMERICALLY EXACT. This is what a miscompile fails.
  - `annotated` must be LABELED. Without this clause the proof can be narrowed until it never
    fires -- which passes every correctness check ever written and silently costs the optimization.
    That is not hypothetical: dropping ping-pong regressed a shipped model 12.4 -> 7.8 tok/s
    (llms/shared/infra/backend_presets.py). gate-h.sh leg 4 watches the same risk from the
    throughput side; this watches it structurally.
  - `annotated_hoisted` must be labeled AND must leave the weight buffer OUT of the rotation set.
    This is the safety clause. The weight is loop-invariant there, so rotating it would hand later
    iterations the wrong half; the compiler is correct today precisely because it excludes it
    (measured: three `hoist_alloc` tile buffers, and the `memref<64xbf16, 2>` weight not among
    them). If a change ever adds it, this row fails structurally AND numerically.
  - `multicolumn` must be EXACT, and today it is not: 3747+ of 4096 elements wrong. Phase H's
    packet put-loop fusion walks only air.LaunchOp's immediate body blocks, and at herd_x >= 2 the
    per-tile put loops are wrapped in an scf.parallel the pass never visits, so the shim
    feed-order corruption returns from the second trip on. Four green single-column variants
    coexisted with that for a whole phase; this row is what stops the next one repeating it.
  - `inside` and `hoisted` must NOT be labeled. An unannotated external call leaves the direction
    of its memref operands unknowable, and the compiler must decline rather than guess. These two
    also record something worth not forgetting: `inside` is correct because of the packet-loop
    fusion, NOT because of ping-pong -- it is single-buffered.

Usage:
    python3 addnorm_multitrip.py --variant inside
    python3 addnorm_multitrip.py --variant hoisted
    python3 addnorm_multitrip.py --variant annotated
    python3 addnorm_multitrip.py --variant annotated_hoisted
    python3 addnorm_multitrip.py --variant multicolumn

Exit 0 iff every clause for that variant holds.

Run under: flock -x -w 1800 /tmp/mlir-air-npu.lock
"""

import argparse
import glob
import os
import re
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
# Every variant runs at cols=64 so the weight buffer's alloc type is one constant below.
COLS = 64
RTOL, ATOL = 1.6e-2, 1e-1

# What the weight buffer's alloc looks like in the labeled IR, as distinct from the three
# [rows_per_call, cols] activation tiles. Used to decide whether the rotation privatized it.
WEIGHT_ALLOC_TYPE = f"memref<{COLS}xbf16, 2 : i32>"

# (rows, cols, herd_x, rows_per_call). Trips per tile = (rows // herd_x) // rows_per_call.
_ONE_COLUMN = (8, 64, 1, 4)    # 2 trips on a single column -- the shape H1s ran
_EIGHT_COLUMN = (64, 64, 8, 4)  # 2 trips on each of 8 columns

#            shape,        callee annotated, weight hoisted, must be labeled, weight rotated
VARIANTS = {
    "inside":            (_ONE_COLUMN,   False, False, False, None),
    "hoisted":           (_ONE_COLUMN,   False, True,  False, None),
    "annotated":         (_ONE_COLUMN,   True,  False, True,  True),
    "annotated_hoisted": (_ONE_COLUMN,   True,  True,  True,  False),
    # `[2026-08-06]` The width every other variant does not cover, and the reason J1 is blocked.
    #
    # Phase H's fix for the two-trip miscompile worked only on ONE column. air-dma-to-channel
    # emits one scf.parallel wrapper PER hoisted put loop, and air-fuse-packet-put-loops walked
    # only air.LaunchOp's immediate body blocks -- so at herd_x >= 2 its output IR was
    # byte-identical to its input and the shim feed-order corruption returned from the second trip
    # on, silently: measured 4070 of 4096 elements wrong at exactly this shape.
    #
    # `[2026-08-07]` H9 fixed it by SEQUENTIALIZING each eligible wrapper into per-iteration
    # clones before grouping -- walking nested blocks alone would have fused nothing, since each
    # holds a single loop and the groups that matter span the wrappers. This clause is what proves
    # it: it went from 3747+/4096 wrong to numerically exact.
    #
    # Every clause above runs at herd_x=1, which is the only width H1s's fixture ever ran -- so
    # four green variants coexisted with a live silent miscompile one column wider. That is the
    # gap this row closes, and it is why it must be the DRIVER's fixture rather than a session's
    # recorded measurement: J1's own Codex review flagged the wall as supported only by prose.
    "multicolumn":       (_EIGHT_COLUMN, False, False, False, None),
}


@module_builder
def build(rows, cols, herd_x, rows_per_call, hoist_weight, annotate_callee,
          np_dtype=bfloat16):
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
    if annotate_callee:
        # x, residual and weight are read; the output is written. These are the attributes H2's
        # classifier reads; without them every memref operand of this call stays unclassifiable and
        # the proof declines the loop. The kernel really does have these directions -- the point of
        # the unannotated variants is that the COMPILER cannot know that.
        fn.arg_attrs = ArrayAttr.get([
            DictAttr.get({"llvm.readonly": UnitAttr.get()}),
            DictAttr.get({"llvm.readonly": UnitAttr.get()}),
            DictAttr.get({"llvm.readonly": UnitAttr.get()}),
            DictAttr.get({"llvm.writeonly": UnitAttr.get()}),
            DictAttr.get({}),
            DictAttr.get({}),
        ])

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

                    # The whole difference between the hoisted and non-hoisted variants.
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


def read_labeled_ir():
    """The IR as `air-label-scf-for-to-ping-pong` left it, from aircc's --debug-ir dump.

    Returns (text, path) or (None, reason). aircc writes one file per pass into
    <tmpdir>/debug_ir/, and tmpdir defaults to `air_project` relative to the CWD -- which this
    fixture owns, having chdir'd into a temp directory of its own.

    Reading the labeled IR rather than matching diagnostic text is deliberate: the label IS the
    decision. A warning can be emitted or suppressed independently of what the pass actually did,
    and the pair of clauses this fixture cares about ("was it labeled" and "was the weight buffer
    in the rotation set") are both only answerable from the IR.
    """
    hits = sorted(glob.glob("air_project/debug_ir/*label-scf-for-to-ping-pong*.mlir"))
    if not hits:
        return None, ("no labeled-IR dump under air_project/debug_ir/ -- either --debug-ir did "
                      "not reach aircc, or the pass was removed from the pipeline")
    if len(hits) > 1:
        return None, (f"expected one labeled-IR dump, found {len(hits)}: "
                      f"{[os.path.basename(h) for h in hits]}. The pipeline now runs the labeling "
                      "pass more than once and this check no longer knows which decision it is "
                      "reading.")
    return open(hits[0]).read(), hits[0]


def check_labeling(text, want_labeled, want_weight_rotated):
    """Clauses about what the pass DID. Returns a list of failure strings (empty means pass)."""
    fails = []
    labeled = re.search(r"\bunroll\s*=", text) is not None
    if labeled != want_labeled:
        fails.append(
            f"expected the loop to be {'LABELED' if want_labeled else 'left alone'}, "
            f"but `unroll` is {'present' if labeled else 'absent'} in the labeled IR"
        )

    if want_weight_rotated is None:
        return fails

    rotated = [
        ln for ln in text.splitlines()
        if "hoist_alloc" in ln and WEIGHT_ALLOC_TYPE in ln
    ]
    weight_rotated = bool(rotated)
    if weight_rotated != want_weight_rotated:
        if want_weight_rotated:
            fails.append(
                f"expected the weight buffer ({WEIGHT_ALLOC_TYPE}) to be in the rotation set -- it "
                "is refilled every iteration, so rotating it is both safe and the point"
            )
        else:
            fails.append(
                f"the weight buffer ({WEIGHT_ALLOC_TYPE}) was put IN the rotation set, but it is "
                "filled once before the loop and read by every iteration. Rotating it hands later "
                "iterations the wrong half. This is the hazard the proof exists to exclude."
            )
    return fails


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    args = ap.parse_args()
    shape, annotate, hoisted, want_labeled, want_weight_rotated = VARIANTS[args.variant]
    rows, cols, herd_x, rows_per_call = shape

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
    trips = (rows // herd_x) // rows_per_call
    print(f"fixture: variant={args.variant} rows={rows} cols={cols} herd_x={herd_x} "
          f"rows_per_call={rows_per_call} -> {trips} trips on {herd_x} column(s); "
          f"callee={'annotated' if annotate else 'unannotated'}, "
          f"weight DMA {'hoisted out of' if hoisted else 'inside'} the loop")

    rng = np.random.default_rng(7)
    x = (rng.standard_normal((rows, cols)) * 0.5).astype(bfloat16)
    res = (rng.standard_normal((rows, cols)) * 0.5).astype(bfloat16)
    w = (rng.random((cols,)) + 0.5).astype(bfloat16)
    expected = addnorm_pre_add_reference(x, res, w)

    try:
        module = build(rows, cols, herd_x, rows_per_call, hoisted, annotate)
        runner = XRTRunner(verbose=False, omit_while_true_loop=False,
                           output_format="elf", instance_name="addnorm",
                           debug_ir=True)
        rc = runner.run_test(module, inputs=[x, res, w], expected_outputs=[expected],
                             rtol=RTOL, atol=ATOL)
        compiled, correct = True, (rc == 0)
    except Exception as exc:  # noqa: BLE001
        print(f"  compilation/run raised {type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:200]}")
        compiled, correct = False, False

    fails = []
    if not compiled:
        # Every variant here is a correct program. Declining to TRANSFORM one is the intended
        # behaviour; declining to COMPILE one is the spec error that broke three shipped models.
        fails.append("the compiler did not produce a binary. Every variant of this fixture is a "
                     "correct program: when the rotation cannot be proven safe the pass must SKIP "
                     "-- leave the loop single-buffered and warn -- not abort the build.")
    elif not correct:
        fails.append("the program compiled but the output is outside tolerance")

    if compiled:
        text, where = read_labeled_ir()
        if text is None:
            fails.append(where)
        else:
            print(f"  labeled IR: {where}")
            fails.extend(check_labeling(text, want_labeled, want_weight_rotated))

    if fails:
        print(f"  -> FAIL ({args.variant}):")
        for f in fails:
            print(f"       - {f}")
        return 1

    shape = "labeled" if want_labeled else "left single-buffered"
    extra = ""
    if want_weight_rotated is not None:
        extra = (", weight buffer rotated" if want_weight_rotated
                 else ", weight buffer correctly excluded from the rotation")
    print(f"  -> PASS ({args.variant}): compiles, numerically exact, {shape}{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
