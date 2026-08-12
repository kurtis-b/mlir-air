# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# llama.cpp `q4_0` GEMV on NPU2, through the *unmodified* int4-AWQ kernel.
#
#   D[M] = dequant_q4_0(W)[M, K] @ B[K],   dequant_q4_0(w) = d * (q - 8)
#
# q4_0 is a strict special case of the AWQ kernel's `(q - z) * s` with the
# zero-point pinned to 8 and the group size pinned to 32. Nothing in
# `mv_int4_bf16.cc` changes; the device module is `matvec_int4_packed`'s,
# instantiated at GS=32. Only the host-side pack differs, and that lives in
# `gguf_q4_0.py`.
#
# Build/run (GS is forced to 32 -- it is q4_0's block size):
#   make run_q4_0 GGUF=/path/to/model.gguf TENSOR=blk.0.attn_q.weight
#
# The negative controls exist so the PASS is not vacuous. Each one breaks
# exactly one property of the pack and must FAIL on device:
#   no-uninterleave : feed q4_0's own nibble order straight through, skipping
#                     the [j, j+16] -> [2b, 2b+1] re-pairing.
#   wrong-zero      : Z = 7 instead of 8 (a plausible off-by-one).
#   fp16-scale-bits : reinterpret d's fp16 bits as bf16 without converting.

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


def apply_negative_control(mode, raw, A_q, A_s, A_z, K, M):
    if mode == "none":
        return A_q, A_s, A_z
    if mode == "no-uninterleave":
        # Hand the kernel q4_0's own byte order untouched.
        _, qs = q4_0_blocks(raw, M * (K // QK4_0))
        return np.ascontiguousarray(qs.reshape(M, K // 2)), A_s, A_z
    if mode == "wrong-zero":
        return A_q, A_s, np.full_like(A_z, 7)
    if mode == "fp16-scale-bits":
        d, _ = q4_0_blocks(raw, M * (K // QK4_0))
        bad = np.asarray(d).view(np.uint16).view(bfloat16)
        return A_q, np.ascontiguousarray(bad.reshape(M, K // QK4_0).T), A_z
    raise SystemExit("unknown negative control %r" % mode)


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
        "--negative-control",
        default="none",
        choices=["none", "no-uninterleave", "wrong-zero", "fp16-scale-bits"],
        dest="neg",
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

    module = build_module(
        M,
        K,
        GS=GS,
        M_TILE=args.m_tile,
        K_CHUNK=args.k_chunk,
        N_CORES=args.n_cores,
        M_PER_LAUNCH=m_per_launch,
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

    print("q4_0 GEMV  source=%s  M=%d K=%d GS=%d" % (prov, M, K, GS))
    print(
        "  packed bytes: Q=%d S=%d Z=%d total=%d (%.3f bits/weight)"
        % (
            tr["q_bytes"],
            tr["s_bytes"],
            tr["z_bytes"],
            tr["total_bytes"],
            tr["bits_per_weight"],
        )
    )
    print(
        "  symmetric variant would be %d B (%.3f bits/weight); the all-8s Z "
        "plane costs %d B" % (
            tr_sym["total_bytes"],
            tr_sym["bits_per_weight"],
            tr["total_bytes"] - tr_sym["total_bytes"],
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

    D_ref = fast_reference(A_q, A_s, A_z, B)

    A_q, A_s, A_z = apply_negative_control(args.neg, raw, A_q, A_s, A_z, K, M)
    if args.neg != "none":
        print("  NEGATIVE CONTROL '%s' active -- this run MUST fail." % args.neg)

    PACKED = pack_inputs(
        A_q, A_s, A_z, M, K, GS, args.m_tile, args.k_chunk, args.n_cores, m_per_launch
    )
    print("  packed BO: %s = %d B" % (PACKED.shape, PACKED.nbytes))

    if args.host_only:
        print("host-only: skipping device run")
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
        min_correlation=0.999,
    )
    if args.neg != "none":
        # Invert: the control is correct exactly when the device run fails.
        if rc == 0:
            print("NEGATIVE CONTROL DID NOT FAIL -- the gate is vacuous.")
            return 1
        print("NEGATIVE CONTROL '%s' failed as required." % args.neg)
        return 0
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
