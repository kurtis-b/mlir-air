# Can two AIR kernels of DIFFERENT shape live in one xclbin and run from one `hw_context`?
#
# WHAT THIS ESTABLISHED `[2026-08-09]`: THE ANSWER IS NO, AND `offload`'s N-STREAMS HALF
# IS BLOCKED ON IT. Two GEMMs chained with `--xclbin-input`, at seq 1024, run from a
# single `hw_context`:
#
#   compile order                 kernels in xclbin      first            second
#   q_proj then up_proj           ['mm_qproj','mm_up']   9.5676e-03 OK    ERT_CMD_STATE_TIMEOUT
#   up_proj then q_proj (reverse) ['mm_up','mm_qproj']   9.5770e-03 OK    1.4124e+00 WRONG
#
# THE FAILURE IS POSITIONAL, NOT SHAPE-SPECIFIC. Whichever kernel is compiled FIRST runs
# correctly from the shared xclbin; whichever is SECOND does not. Reversing the order
# reverses which one breaks. Both kernels are present in the xclbin by name either way --
# `get_kernels()` lists both -- so the packaging succeeds and the execution does not.
#
# THE SECOND KERNEL'S FAILURE MODE DEPENDS ON ITS SHAPE, AND ONE OF THE TWO IS SILENT.
# `up_proj` (1024x768x3072) times out; `q_proj` (1024x768x768) returns garbage at
# mean_rel_L1 1.41 with NO error raised. Wrong numbers with no signal is the worse of the
# two and it is the one a gate would miss.
#
# THE CONTROLS THAT MAKE THIS A PROPERTY OF CHAINING RATHER THAN OF EITHER KERNEL:
# compiled STANDALONE as its own xclbin, each runs correctly and reuses its context
# cleanly -- `q_proj` 9.5676e-03 and `up_proj` 9.5770e-03, bit-identical across runs
# (`probe_context_reuse.py reuse-xclbin --op ...`). The only variable is `--xclbin-input`.
#
# CONSEQUENCE FOR THE PLAN. Doc 26 sizes "N instruction streams under one xclbin" as its
# own phase and records the mechanism as "already plumbed (`xclbin_input`, `xrt.py:80`,
# `:372-374`) but never dispatched". The plumbing is there and it DOES NOT WORK: aircc
# packages the kernel into the existing xclbin, and the result executes only in the first
# slot. So the phase is not a dispatch-layer change gated on a split rule -- it is blocked
# on the packaging path, which is `aircc --xclbin-input` and below. The `plan_submissions`
# split rule and the `ensure_loaded` context sharing remain correct designs with nothing
# to run on.
#
# WHAT IS NOT YET KNOWN: whether the second kernel's instruction stream, its shim BD
# programming, or the merged xclbin's partition metadata is what goes wrong. That is the
# next question and it is a compiler-side one.
#
# WHY THIS EXISTS.  `offload`'s remaining half is "N instruction streams under one xclbin"
# -- 03's mechanism for the mode's reconfiguration-minimizing claim, and the thing that
# makes it match iron. Doc 26 records the plumbing as "already plumbed (`xclbin_input`,
# `xrt.py:80`, `:372-374`) but never dispatched", which is a statement about options
# existing, not about the mechanism working. Nothing in this tree has ever produced one.
#
# The implementation it gates is not small -- it reaches `cache.ensure_loaded` (which
# creates one XRTBackend and one `hw_context` PER ARTIFACT NAME), `dispatch.py`'s split
# rule, and `pattern/offload`. So the mechanism gets proven before any of that is written.
#
# WHAT IT DOES.  Compiles two GEMMs of DIFFERENT shape, chaining the second onto the
# first with `xclbin_input`, then loads the resulting xclbin ONCE and runs both kernels
# from that single context, checking both against a numpy f32 reference.
#
# THE NAMING TRAP THIS PROBE EXISTS TO SURFACE.  `aircc --xclbin-input` is documented as
# "Generate kernel into existing xclbin file", and the loader finds a kernel by
# SUBSTRING match:
#
#     xkernel = [k for k in kernels if artifact.kernel in k.get_name()][0]   (xrt.py:634)
#
# `pattern/offload` names every drain GEMM `matmul_bf16` via
# `instance_name=_METHOD_FUNC[spec["method"]]`, so all three of its shapes would carry
# the SAME kernel name into a shared xclbin, and the substring match would return
# whichever came first -- the wrong program with the right buffers, which is the failure
# mode `dispatch.py`'s own footguns describe as executing without error and returning
# wrong numbers. So N streams needs N DISTINCT instance names, and this probe gives them
# distinct names deliberately. An arm that reuses one name is included to demonstrate the
# collision rather than assert it.
#
# Usage:  python3 probe_one_xclbin_n_streams.py [--distinct|--collide] [--seq N]
import argparse
import os
import sys

import numpy as np

_TL = "/home/cj/mlir-air/programming_examples/transformer_layer"
_PROJ = "/home/cj/mlir-air/programming_examples"
for _p in (_PROJ, os.path.join(_PROJ, "llms"), _TL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ml_dtypes import bfloat16  # noqa: E402

import shared.infra.external_kernels as ek  # noqa: E402
from air.backend.xrt import XRTBackend, XRTCompileArtifact  # noqa: E402
from pattern.offload.offload import _build_offload_module, offload_config  # noqa: E402

SCRATCH = "/home/cj/.claude/jobs/e75c34c9/tmp/nstreams"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument(
        "--collide",
        action="store_true",
        help="give both kernels the SAME instance name, to demonstrate the "
        "substring-match collision rather than assert it",
    )
    ap.add_argument(
        "--only",
        default="",
        help="execute only this op from the shared xclbin (both are still "
        "compiled into it). Discriminates 'the second kernel is broken' from "
        "'running two different kernels in one context is broken'.",
    )
    ap.add_argument(
        "--reverse",
        action="store_true",
        help="compile up_proj FIRST and q_proj second, to tell 'the second "
        "kernel in a chained xclbin is broken' from 'up_proj is broken'",
    )
    args = ap.parse_args()
    os.makedirs(SCRATCH, exist_ok=True)

    cfg = offload_config(args.seq, 768, 3072, 12, 64)

    # Two DIFFERENT shapes, so a mix-up is visible as wrong numbers rather than as a
    # coincidence: q_proj is seq x768x768 and up is seq x768x3072.
    order = (("q_proj", "mm_qproj"), ("up_proj", "mm_up"))
    if args.reverse:
        order = tuple(reversed(order))
    picks = []
    for op, inst in order:
        key = cfg["gemms"][op]
        spec, mkn = cfg["specs"][key]
        name = "shared_kernel" if args.collide else inst
        picks.append((op, key, spec, mkn, name))

    print(f"[probe] seq        : {args.seq}")
    print(f"[probe] instance   : {'COLLIDING (same name)' if args.collide else 'distinct'}")
    for op, _, spec, (m, k, n), inst in picks:
        print(f"[probe]   {op:8s} {m}x{k}x{n} {spec['method']} -> instance '{inst}'")

    # ---- build: chain the second compile onto the first's xclbin -------------------
    xclbin_in, artifacts = "", []
    for op, key, spec, (m, k, n), inst in picks:
        backend = XRTBackend(
            output_format="xclbin",
            instance_name=inst,
            kernel_name=inst,
            xclbin_input=xclbin_in,
            runtime_loop_tiling_sizes=[2, 2],
            omit_while_true_loop=False,
        )
        out = os.path.join(SCRATCH, f"{op}_{args.seq}")
        os.makedirs(out, exist_ok=True)
        module = _build_offload_module(cfg, key)
        cwd = os.getcwd()
        os.chdir(out)
        try:
            # The micro-kernel object must exist in THIS working directory before
            # aircc links: it resolves `air_project/mm_*.o` relative to cwd, and the
            # mode's own compile path builds it first for the same reason
            # (`compile_offload_artifacts` calls this before `compile_and_cache`).
            print(f"[probe] external object for {op}: {spec['obj']}", flush=True)
            ek.compile_gemm_mm(
                tile_m=spec["tile_m"],
                tile_n=spec["tile_n"],
                tile_k_l1=spec["tile_k_l1"],
                sym_suffix=spec["sym_suffix"],
                out_name=spec["obj"],
            )
            print(f"[probe] compiling {op} (xclbin_input={xclbin_in or 'NONE'}) ...",
                  flush=True)
            art = backend.compile(module)
        finally:
            os.chdir(cwd)
        art = XRTCompileArtifact(
            os.path.abspath(os.path.join(out, os.path.basename(art.output_binary))),
            art.kernel,
            os.path.abspath(os.path.join(out, os.path.basename(art.insts)))
            if art.insts
            else None,
        )
        # The xclbin names its kernels by INSTANCE name -- `art.kernel` comes back as
        # aircc's default `MLIR_AIE` regardless -- so the instance name is what the
        # loader's substring match has to be given. Carrying the wrong one here is
        # what makes a shared xclbin silently select the wrong program.
        print(f"[probe]   -> {art.output_binary}  "
              f"artifact.kernel={art.kernel!r} instance={inst!r}")
        artifacts.append((op, spec, (m, k, n), art, inst))
        # The NEXT compile packages itself into THIS xclbin, so the last one written
        # holds every kernel.
        xclbin_in = art.output_binary

    final_xclbin = artifacts[-1][3].output_binary
    print(f"[probe] final xclbin: {final_xclbin}")

    # ---- load ONCE, run both kernels from that one context ------------------------
    import pyxrt as xrt

    device = xrt.device(0)
    xb = xrt.xclbin(final_xclbin)
    device.register_xclbin(xb)
    context = xrt.hw_context(device, xb.get_uuid())
    present = [k.get_name() for k in xb.get_kernels()]
    print(f"[probe] kernels in the final xclbin: {present}")
    print(f"[probe] hw_contexts created: 1")

    if len(present) < len(picks) and not args.collide:
        print("[probe] RESULT: FAILED -- the chained compile did not accumulate "
              f"kernels ({len(present)} of {len(picks)})")
        return 1

    ok = True
    for op, spec, (m, k, n), art, inst in artifacts:
        if args.only and op != args.only:
            print(f"[probe] {op:8s} SKIPPED (--only {args.only})")
            continue
        matches = [name for name in present if inst in name]
        if not matches:
            print(f"[probe] {op}: kernel '{inst}' NOT in the shared xclbin")
            ok = False
            continue
        kern = xrt.kernel(context, matches[0])

        with open(art.insts, "rb") as f:
            instr = np.frombuffer(f.read(), dtype=np.uint32).copy()
        bo_instr = xrt.bo(device, len(instr) * 4, xrt.bo.cacheable, kern.group_id(1))
        bo_instr.write(instr.tobytes(), 0)
        bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        rng = np.random.default_rng(0)
        a = rng.standard_normal((m, k)).astype(np.float32).astype(bfloat16)
        b = (rng.standard_normal((k, n)) / np.sqrt(k)).astype(np.float32).astype(bfloat16)
        c = np.zeros((m, n), dtype=bfloat16)
        reference = a.astype(np.float32) @ b.astype(np.float32)

        bos = []
        for i, arr in enumerate((a, b, c)):
            nbytes = arr.size * arr.itemsize
            bo = xrt.bo(device, nbytes, xrt.bo.host_only, kern.group_id(i + 3))
            np.frombuffer(bo.map(), dtype=np.uint8)[:nbytes] = np.frombuffer(
                np.ascontiguousarray(arr).view(np.int16).tobytes(), dtype=np.uint8
            )
            bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            bos.append(bo)

        run = xrt.run(kern)
        run.set_arg(0, 3)  # OPCODE_DPU
        run.set_arg(1, bo_instr)
        run.set_arg(2, len(instr))
        for i, bo in enumerate(bos):
            run.set_arg(i + 3, bo)
        run.start()
        state = run.wait2()

        bos[2].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        got = np.frombuffer(bos[2].map(), dtype=np.int16, count=m * n)
        got = got.view(bfloat16).reshape(m, n).astype(np.float32)
        rel = np.abs(got - reference).mean() / np.abs(reference).mean()
        good = rel < 5e-2
        ok &= good
        print(f"[probe] {op:8s} {m}x{k}x{n} on the SHARED context: "
              f"state={state} mean_rel_L1={rel:.4e} {'OK' if good else 'WRONG'}")

    print(f"[probe] RESULT {'distinct' if not args.collide else 'collide'}: "
          f"one_xclbin_two_kernels_one_context={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
