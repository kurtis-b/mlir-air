# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# llama.cpp `q4_0` GEMV on NPU2, in two variants.
#
#   D[M] = dequant_q4_0(W)[M, K] @ B[K],   dequant_q4_0(w) = d * (q - 8)
#
# ASYMMETRIC (default) -- the *unmodified* int4-AWQ kernel. q4_0 is a strict
# special case of the kernel's `(q - z) * s` with the zero point pinned to 8
# and the group size pinned to 32, so the whole bridge is a host-side repack
# that emits `Z` as an all-8s plane. Nothing in `mv_int4_bf16.cc` changes.
#
# SYMMETRIC (`--symmetric`) -- `matvec_int4_bf16_packed_sym`, which subtracts
# the immediate 8 and never loads a Z plane. The packed BO then carries Q and S
# only. The two variants compute the same function; the difference is DRAM
# traffic, K*M/32 bytes per weight tensor (0.250 of the 4.750 bits/weight).
#
# Build/run (GS is forced to 32 -- it is q4_0's block size):
#   make run_q4_0     GGUF=/path/to/model.gguf TENSOR=blk.0.attn_q.weight
#   make run_q4_0_sym GGUF=/path/to/model.gguf TENSOR=blk.0.attn_q.weight
#
# The negative controls exist so the PASS is not vacuous. Each one breaks
# exactly one property and must FAIL on device:
#   no-uninterleave : feed q4_0's own nibble order straight through, skipping
#                     the [j, j+16] -> [2b, 2b+1] re-pairing.   (both variants)
#   fp16-scale-bits : reinterpret d's fp16 bits as bf16 without converting.
#                                                               (both variants)
#   wrong-zero      : Z = 7 instead of 8 (a plausible off-by-one).
#                     ASYMMETRIC ONLY -- it is meaningless once there is no
#                     zero plane, and is refused under --symmetric rather than
#                     silently passing. Its symmetric replacement lives where
#                     the constant moved: build the kernel with -DQ4_0_ZP=7
#                     (`make run_q4_0_sym_wrongzp`).
#   s-transposed    : write the per-tile S plane in [M_TILE, n_groups] order
#                     instead of [n_groups, M_TILE]. Same bytes, same tile
#                     size, wrong axis order -- the failure a packer makes when
#                     S becomes the tail plane and nothing downstream of it
#                     would notice.                              (both variants)

import argparse
import os
import sys

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from air.backend.xrt_runner import XRTRunner

from gguf_q4_0 import (  # noqa: E402
    GGUFFile,
    QK4_0,
    q4_0_blocks,
    q4_0_traffic_bytes,
    repack_q4_0_linear,
    scale_rounding_error,
    _synthesize_q4_0,
)
from matvec_int4_packed import build_module, cpu_reference, pack_inputs  # noqa: E402

GS = QK4_0  # 32 -- not a free parameter, it is q4_0's block size


def fast_reference(A_q, A_s, A_z, B):
    """Vectorized fp32 equivalent of `matvec_int4_packed.cpu_reference`.

    `cpu_reference` is a Python double loop over M*K, which is minutes at
    2048x2048. This is the same arithmetic in numpy; `--check-reference`
    proves the two agree on a sub-block before the fast one is trusted.
    """
    M, K_half = A_q.shape
    K = K_half * 2
    gs = K // A_s.shape[0]
    nibs = np.empty((M, K), dtype=np.uint8)
    nibs[:, 0::2] = A_q & 0x0F
    nibs[:, 1::2] = (A_q >> 4) & 0x0F
    w = nibs.astype(np.float32) - A_z.astype(np.float32).T.repeat(gs, axis=1)
    w *= A_s.astype(np.float32).T.repeat(gs, axis=1)
    return (w @ B.astype(np.float32)).astype(bfloat16)


def fast_reference_sym(A_q, A_s, B, zp=8):
    """Same, for the symmetric variant: `(q - zp) * s`, no Z array anywhere.

    Deliberately NOT `fast_reference(..., A_z=full(zp))` -- the symmetric gate's
    reference must not be able to pass by way of a zero plane the device never
    receives.
    """
    M, K_half = A_q.shape
    K = K_half * 2
    gs = K // A_s.shape[0]
    nibs = np.empty((M, K), dtype=np.uint8)
    nibs[:, 0::2] = A_q & 0x0F
    nibs[:, 1::2] = (A_q >> 4) & 0x0F
    w = nibs.astype(np.float32) - np.float32(zp)
    w *= A_s.astype(np.float32).T.repeat(gs, axis=1)
    return (w @ B.astype(np.float32)).astype(bfloat16)


def load_tensor(args):
    """Return (raw q4_0 payload, K, M, provenance string)."""
    if args.gguf:
        g = GGUFFile(args.gguf)
        if args.tensor not in g.tensors:
            raise SystemExit("no tensor %r in %s" % (args.tensor, args.gguf))
        ti = g.tensors[args.tensor]
        if ti.type_name != "Q4_0":
            raise SystemExit(
                "tensor %r is %s, not Q4_0. llama.cpp's Q4_0 file type is "
                "mixed: token_embd is usually Q6_K and some ffn_down layers "
                "are Q4_1. Pick a pure-q4_0 tensor (see `gguf_q4_0.py "
                "--list`)." % (args.tensor, ti.type_name)
            )
        K, M = int(ti.ne[0]), int(ti.ne[1])
        return np.asarray(g.raw_bytes(args.tensor)), K, M, "%s:%s" % (
            os.path.basename(args.gguf),
            args.tensor,
        )
    raw, _, _ = _synthesize_q4_0(args.k, args.m, seed=args.seed)
    return raw, args.k, args.m, "synthetic(seed=%d)" % args.seed


# Controls that corrupt an input plane before packing.
PLANE_CONTROLS = ("no-uninterleave", "wrong-zero", "fp16-scale-bits")
# Controls that corrupt the packed BO's layout after packing.
PACKED_CONTROLS = ("s-transposed",)
ALL_CONTROLS = ("none",) + PLANE_CONTROLS + PACKED_CONTROLS
# Refused rather than silently vacuous under --symmetric.
ASYMMETRIC_ONLY_CONTROLS = ("wrong-zero",)


def apply_negative_control(mode, raw, A_q, A_s, A_z, K, M):
    if mode == "none" or mode in PACKED_CONTROLS:
        return A_q, A_s, A_z
    if mode == "no-uninterleave":
        # Hand the kernel q4_0's own byte order untouched.
        _, qs = q4_0_blocks(raw, M * (K // QK4_0))
        return np.ascontiguousarray(qs.reshape(M, K // 2)), A_s, A_z
    if mode == "wrong-zero":
        if A_z is None:
            raise SystemExit(
                "negative control 'wrong-zero' is meaningless for the symmetric "
                "variant: there is no zero plane to corrupt, so the run would "
                "PASS and the control would prove nothing. Use "
                "`make run_q4_0_sym_wrongzp`, which moves the same off-by-one "
                "to where the constant now lives (-DQ4_0_ZP=7 in the kernel)."
            )
        return A_q, A_s, np.full_like(A_z, 7)
    if mode == "fp16-scale-bits":
        d, _ = q4_0_blocks(raw, M * (K // QK4_0))
        bad = np.asarray(d).view(np.uint16).view(bfloat16)
        return A_q, np.ascontiguousarray(bad.reshape(M, K // QK4_0).T), A_z
    raise SystemExit("unknown negative control %r" % mode)


def apply_packed_control(mode, PACKED, M_TILE, K_CHUNK, GS):
    """Corrupt the packed BO in place, after packing. Returns (PACKED, note).

    Layout-level faults have to be injected here: they are properties of the
    tile, not of any input plane.
    """
    if mode not in PACKED_CONTROLS:
        return PACKED, None
    n_gpc = K_CHUNK // GS
    q_bytes = M_TILE * (K_CHUNK // 2)
    s_bytes = n_gpc * M_TILE * 2
    if mode == "s-transposed":
        if n_gpc == M_TILE:
            raise SystemExit(
                "NEGATIVE CONTROL VACUOUS: n_groups_per_chunk == M_TILE == %d, "
                "so transposing S is a no-op at this shape" % n_gpc
            )
        before = PACKED[:, q_bytes : q_bytes + s_bytes].copy()
        for t in range(PACKED.shape[0]):
            s = PACKED[t, q_bytes : q_bytes + s_bytes].view(bfloat16)
            PACKED[t, q_bytes : q_bytes + s_bytes] = (
                np.ascontiguousarray(s.reshape(n_gpc, M_TILE).T)
                .view(np.uint8)
                .reshape(-1)
            )
        if np.array_equal(before, PACKED[:, q_bytes : q_bytes + s_bytes]):
            raise SystemExit(
                "NEGATIVE CONTROL VACUOUS: the transposed S plane is "
                "byte-identical to the original"
            )
        return PACKED, "S plane transposed to [M_TILE, n_groups] in all %d tiles" % (
            PACKED.shape[0]
        )
    raise SystemExit("unknown packed control %r" % mode)


def main():
    p = argparse.ArgumentParser(
        prog="matvec_q4_0.py",
        description="llama.cpp q4_0 GEMV on the unmodified int4-AWQ kernel.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("-p", "--print-module-only", action="store_true")
    p.add_argument("--gguf", type=str, default=None)
    p.add_argument("--tensor", type=str, default="blk.0.attn_q.weight")
    p.add_argument("--m", type=int, default=2048, help="synthetic-only")
    p.add_argument("--k", type=int, default=2048, help="synthetic-only")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--m-tile", type=int, default=8, dest="m_tile")
    p.add_argument("--k-chunk", type=int, default=2048, dest="k_chunk")
    p.add_argument("--n-cores", type=int, default=8, dest="n_cores")
    p.add_argument("--m-per-launch", type=int, default=None, dest="m_per_launch")
    p.add_argument(
        "--symmetric",
        action="store_true",
        help="drop the all-8s Z plane; dispatch matvec_int4_bf16_packed_sym",
    )
    p.add_argument(
        "--negative-control",
        default="none",
        choices=list(ALL_CONTROLS),
        dest="neg",
    )
    p.add_argument(
        "--expect-mismatch",
        default=None,
        dest="expect_mismatch",
        metavar="WHY",
        help="invert the verdict for a fault injected OUTSIDE this script "
        "(e.g. a kernel built with the wrong immediate). Kept here rather than "
        "in the caller so that a crash, a bad argument or a missing checkpoint "
        "still fails -- only a numeric mismatch counts as the control firing.",
    )
    p.add_argument(
        "--dump-output",
        default=None,
        dest="dump_output",
        help="write the raw device output to this .npy instead of gating on it "
        "(used to compare the two variants against each other)",
    )
    p.add_argument(
        "--check-reference",
        action="store_true",
        help="prove fast_reference == cpu_reference on a sub-block first",
    )
    p.add_argument("--host-only", action="store_true", help="no device run")
    p.add_argument("--output-format", choices=["xclbin", "elf"], default="elf")
    p.add_argument(
        "--compile-mode",
        choices=["compile-and-run", "compile-only"],
        default="compile-and-run",
        dest="compile_mode",
    )
    args = p.parse_args()

    raw, K, M, prov = load_tensor(args)
    m_per_launch = args.m_per_launch if args.m_per_launch is not None else M

    # Argument validation up front, before anything is built: a control that is
    # inapplicable, doubled, or attached to a run with no verdict must refuse
    # rather than produce a run someone could read as green.
    if args.symmetric and args.neg in ASYMMETRIC_ONLY_CONTROLS:
        raise SystemExit(
            "negative control %r does not apply to the symmetric variant "
            "(there is no zero plane); see the module docstring for its "
            "replacement" % args.neg
        )
    if args.expect_mismatch and args.neg != "none":
        raise SystemExit(
            "--expect-mismatch and --negative-control together would make a "
            "passing run of either look like a firing control"
        )
    if args.expect_mismatch and args.dump_output:
        raise SystemExit("--expect-mismatch has no verdict to invert in a dump run")

    module = build_module(
        M,
        K,
        GS=GS,
        M_TILE=args.m_tile,
        K_CHUNK=args.k_chunk,
        N_CORES=args.n_cores,
        M_PER_LAUNCH=m_per_launch,
        symmetric=args.symmetric,
    )
    if args.print_module_only:
        print(module)
        return 0

    if args.compile_mode == "compile-only":
        # Compile without dispatching. This branch must exist and must be
        # taken before any XRTRunner call: XRTRunner.run_test compiles AND
        # runs, so a missing branch here silently touches the device.
        from air.backend.xrt import XRTBackend

        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="matvec_int4_packed",
            use_lock_race_condition_fix=False,
            stack_size=4096,
        )
        backend.compile(module)
        backend.unload()
        print("compile-only: no device dispatch")
        return 0

    A_q, A_s, A_z = repack_q4_0_linear(raw, K, M)
    assert np.all(A_z == 8), "q4_0 zero-point plane must be all 8s"

    d, _ = q4_0_blocks(raw, M * (K // QK4_0))
    st = scale_rounding_error(np.asarray(d, dtype=np.float32))
    tr = q4_0_traffic_bytes(K, M, gs=GS, with_zeros=True)
    tr_sym = q4_0_traffic_bytes(K, M, gs=GS, with_zeros=False)

    variant = "SYMMETRIC (no Z plane)" if args.symmetric else "asymmetric (Z = all 8s)"
    live = tr_sym if args.symmetric else tr
    print(
        "q4_0 GEMV  source=%s  M=%d K=%d GS=%d  variant=%s"
        % (prov, M, K, GS, variant)
    )
    print(
        "  packed bytes THIS RUN: Q=%d S=%d Z=%d total=%d (%.3f bits/weight)"
        % (
            live["q_bytes"],
            live["s_bytes"],
            live["z_bytes"],
            live["total_bytes"],
            live["bits_per_weight"],
        )
    )
    print(
        "  asymmetric %d B (%.3f b/w) vs symmetric %d B (%.3f b/w); the all-8s "
        "Z plane costs %d B = K*M/%d"
        % (
            tr["total_bytes"],
            tr["bits_per_weight"],
            tr_sym["total_bytes"],
            tr_sym["bits_per_weight"],
            tr["total_bytes"] - tr_sym["total_bytes"],
            GS,
        )
    )
    print(
        "  scale fp16->bf16: max_rel=%.6f mean_rel=%.6f rms_rel=%.6f "
        "exact=%.4f over %d blocks"
        % (st["max_rel"], st["mean_rel"], st["rms_rel"], st["exact_frac"], st["n"])
    )

    rng = np.random.default_rng(args.seed + 7)
    B = rng.standard_normal(K).astype(bfloat16)

    if args.check_reference:
        sub_m, sub_k = 16, min(K, 256)
        ng = sub_k // GS
        r1 = cpu_reference(
            A_q[:sub_m, : sub_k // 2], A_s[:ng, :sub_m], A_z[:ng, :sub_m], B[:sub_k]
        ).astype(np.float32)
        r2 = fast_reference(
            A_q[:sub_m, : sub_k // 2], A_s[:ng, :sub_m], A_z[:ng, :sub_m], B[:sub_k]
        ).astype(np.float32)
        if not np.array_equal(r1, r2):
            raise SystemExit(
                "fast_reference disagrees with cpu_reference: max |d| = %g"
                % float(np.max(np.abs(r1 - r2)))
            )
        print("  reference cross-check: fast_reference == cpu_reference "
              "(%dx%d sub-block, bit-identical)" % (sub_m, sub_k))

        if args.symmetric:
            # The symmetric reference must reproduce the asymmetric one on the
            # same sub-block; if it does not, the equivalence claim is already
            # broken on the host and there is no point going to the device.
            r3 = fast_reference_sym(
                A_q[:sub_m, : sub_k // 2], A_s[:ng, :sub_m], B[:sub_k]
            ).astype(np.float32)
            if not np.array_equal(r1, r3):
                raise SystemExit(
                    "fast_reference_sym disagrees with cpu_reference: max |d| = %g"
                    % float(np.max(np.abs(r1 - r3)))
                )
            print("  reference cross-check: fast_reference_sym == cpu_reference "
                  "(same sub-block, bit-identical)")

    if args.symmetric:
        D_ref = fast_reference_sym(A_q, A_s, B)
    else:
        D_ref = fast_reference(A_q, A_s, A_z, B)

    A_q, A_s, A_z = apply_negative_control(args.neg, raw, A_q, A_s, A_z, K, M)
    if args.neg != "none":
        print("  NEGATIVE CONTROL '%s' active -- this run MUST fail." % args.neg)
    if args.expect_mismatch:
        print(
            "  NEGATIVE CONTROL (external: %s) active -- this run MUST fail."
            % args.expect_mismatch
        )

    PACKED = pack_inputs(
        A_q,
        A_s,
        None if args.symmetric else A_z,
        M,
        K,
        GS,
        args.m_tile,
        args.k_chunk,
        args.n_cores,
        m_per_launch,
        symmetric=args.symmetric,
    )
    PACKED, note = apply_packed_control(
        args.neg, PACKED, args.m_tile, args.k_chunk, GS
    )
    if note:
        print("  control detail: %s" % note)
    print("  packed BO: %s = %d B" % (PACKED.shape, PACKED.nbytes))
    if PACKED.nbytes != live["total_bytes"]:
        raise SystemExit(
            "packed BO is %d B but the traffic model says %d B -- the layout "
            "and the accounting disagree" % (PACKED.nbytes, live["total_bytes"])
        )

    if args.host_only:
        print("host-only: skipping device run")
        return 0

    if args.dump_output:
        # Raw device output, for comparing the two variants against EACH OTHER
        # rather than each against its own host reference. XRTRunner.run_test
        # returns only a verdict, so this mirrors its dispatch (same backend
        # options, same npu.lock) and keeps the array.
        import tempfile

        import filelock
        from air.backend.xrt import XRTBackend

        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="matvec_int4_packed",
            use_lock_race_condition_fix=False,
            stack_size=4096,
        )
        compiled = backend.compile(module)
        with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
            fn = backend.load(compiled)
            outs = fn(PACKED, B, np.zeros(D_ref.shape, D_ref.dtype))
        backend.unload()
        D_dev = np.asarray(outs[2])
        np.save(args.dump_output, D_dev)
        np.save(args.dump_output.replace(".npy", "") + ".ref.npy", np.asarray(D_ref))
        print("  device output -> %s (%s %s)"
              % (args.dump_output, D_dev.shape, D_dev.dtype))
        # Still report the gate metrics so a dump run is not a silent run.
        a = D_dev.astype(np.float64)
        e = np.asarray(D_ref).astype(np.float64)
        ok = np.isclose(a, e, rtol=0.1, atol=0.05)
        corr = float(np.corrcoef(a, e)[0, 1])
        print("  (dump run) isclose %d/%d, correlation %.6f"
              % (int(ok.sum()), ok.size, corr))
        return 0

    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format=args.output_format,
        instance_name="matvec_int4_packed",
        use_lock_race_condition_fix=False,
        stack_size=4096,
    )
    rc = runner.run_test(
        module,
        inputs=[PACKED, B],
        expected_outputs=[D_ref],
        rtol=0.1,
        atol=0.05,
        # Explicit, not defaulted: every element must pass isclose. A gate that
        # tolerates a few percent of mismatches is a different gate.
        max_mismatch_percentage=0,
        min_correlation=0.999,
    )
    label = args.neg if args.neg != "none" else args.expect_mismatch
    if label:
        # Invert: the control is correct exactly when the device run fails.
        if rc == 0:
            print("NEGATIVE CONTROL DID NOT FAIL -- the gate is vacuous.")
            return 1
        print("NEGATIVE CONTROL '%s' failed as required." % label)
        return 0
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
