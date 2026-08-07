#!/usr/bin/env python3
"""The fused addnorm's one-pass variance collapses on large-mean rows -- and where.

Doc 23 open item 2. The regime that exposed the one-pass defect (a row whose mean is
large next to its spread) is covered for `norm_tail` by the 128x768_offset spec row.
`layer_norm` and the fused `addnorm` had no equivalent, and `addnorm` DELIBERATELY
keeps one-pass statistics. Four execution paths dispatch it.

THREE MODES, THREE CLAIMS

  --mode compare        Does it break? layer_norm vs both addnorm variants on the same
                        regime prepare_norm_tail uses (x ~ 8.0 + 0.25*randn, residual
                        identically zero so the bf16 sum is exact and the row isolates
                        the STATISTICS), plus a zero-mean control so a failure is
                        attributable to the regime and not the shape.

                            layer_norm            0/49152      (mean_rel_L1 1.1e-4)
                            addnorm  pre-add  43058/49152      (mean_rel_L1 33.1)
                            addnorm post-add  28170/32768      (mean_rel_L1 22.2)

                        J7a's f32 two-pass fix covers layer_norm completely. Both addnorm
                        variants share the one-pass path: E[x^2] and E[x]^2 become the same
                        bf16 number, variance clamps, the row normalizes by 1/sqrt(eps)~316.

  --mode sweep          WHERE the cliff is, measured rather than derived. Compiles ONCE and
                        varies only the input, so the boundary is attributable to the data:

                            |mean|/sigma  2 -> clean      4 -> COLLAPSED (1386/49152)
                                          8 -> 24061     32 -> 43058

                        A mantissa-only derivation gives 2^(8/2) = 16 and is WRONG BY ~4x:
                        it counts the final cancellation but not the 768-term row sums,
                        which lose mantissa in bf16 first. Measure these bounds; do not
                        derive them.

  --mode reachability   Is it reached HERE? Host-only, no NPU: per-row |mean|/sigma of both
                        tensors encoder_bert's addnorms normalize, from the FP32 golden
                        reference. Median 0.025, max 0.115, zero rows over 4 -- a ~35x
                        margin. So the defect is LATENT for this workload and the recorded
                        block/coarse/runlist/fused figures are NOT in question.
                        (encoder_bert only; decoder_gpt2 is pre-norm and no mode dispatches
                        it -- that is J5.)

NOT a test. Nothing runs this in CI. Writes nothing to the repo.
"""

import argparse
import os
import sys
import tempfile

_PE = "/home/cj/mlir-air/programming_examples"
for _p in (_PE, os.path.join(_PE, "llms"), os.path.join(_PE, "transformer_layer")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RTOL, ATOL = 1.6e-2, 5e-2  # opcheck's RTOL and the norm tier's atol

# MEASURED, not derived: clean at |mean|/sigma 2, collapsed at 4. The mantissa-only
# derivation (2^(8/2) = 16) overestimates it ~4x -- see the sweep mode.
CLIFF = 4.0


def _stats(got, exp):
    import numpy as np

    g = got.astype(np.float32)
    e = exp.astype(np.float32)
    err = np.abs(g - e)
    den = np.abs(e)
    nz = den > 0
    mean_rel = float(err[nz].sum() / den[nz].sum()) if nz.any() else float("nan")
    bad = int((err > ATOL + RTOL * den).sum())
    return mean_rel, bad, g.size, float(err.max())


def _draw(rows, cols, ratio, sigma=0.25, seed=7):
    """x with the given |mean|/sigma. ratio 0 is the zero-mean control."""
    import numpy as np
    from ml_dtypes import bfloat16

    rng = np.random.default_rng(seed)
    if ratio == 0:
        return rng.standard_normal((rows, cols)).astype(bfloat16)
    return (ratio * sigma + sigma * rng.standard_normal((rows, cols))).astype(bfloat16)


def _run_layer_norm(rows, cols, ratio):
    import numpy as np
    from ml_dtypes import bfloat16
    from shared.infra import external_kernels as ek
    from builders.layer_norm import build_layer_norm_module, layer_norm_reference
    from air.backend.xrt import XRTBackend

    ek.compile_layer_norm()
    x = _draw(rows, cols, ratio)
    exp = layer_norm_reference(x)
    be = XRTBackend(
        verbose=False,
        omit_while_true_loop=False,
        output_format="xclbin",
        instance_name="ln_cliff",
    )
    try:
        inv = be.compile_and_load(build_layer_norm_module(rows, cols, bfloat16))
        out = inv(x, np.zeros((rows, cols), dtype=bfloat16))[-1]
        return _stats(out.reshape(rows, cols), exp)
    finally:
        try:
            be.unload()
        except Exception:
            pass


def _addnorm_pieces(rows, cols, pre_add):
    import numpy as np
    from ml_dtypes import bfloat16
    from builders.addnorm import (
        addnorm_pre_add_reference,
        addnorm_reference,
        compile_addnorm_kernel,
    )

    compile_addnorm_kernel(pre_add=pre_add)
    # Residual identically zero: the bf16 sum is exact, so the row isolates the
    # STATISTICS rather than the addition.
    residual = np.zeros((rows, cols), dtype=bfloat16)
    weight = np.random.default_rng(99).uniform(0.5, 1.5, size=cols).astype(bfloat16)
    ref = addnorm_pre_add_reference if pre_add else addnorm_reference
    return residual, weight, ref


def _run_addnorm(rows, cols, ratio, pre_add=True):
    import numpy as np
    from ml_dtypes import bfloat16
    from builders.addnorm import build_addnorm_module
    from air.backend.xrt import XRTBackend

    residual, weight, ref = _addnorm_pieces(rows, cols, pre_add)
    x = _draw(rows, cols, ratio)
    exp = ref(x, residual, weight)
    be = XRTBackend(
        verbose=False,
        omit_while_true_loop=False,
        output_format="xclbin",
        instance_name="an_cliff",
    )
    try:
        mod = build_addnorm_module(rows, cols, bfloat16, pre_add=pre_add)
        inv = be.compile_and_load(mod)
        out = inv(x, residual, weight, np.zeros((rows, cols), dtype=bfloat16))[-1]
        return _stats(out.reshape(rows, cols), exp)
    finally:
        try:
            be.unload()
        except Exception:
            pass


def mode_compare(args):
    print(f"shape {args.rows}x{args.cols}  rtol={RTOL} atol={ATOL}")
    print("offset regime = x ~ 8.0 + 0.25*randn (|mean|/sigma 32), residual 0\n")
    hdr = f"{'operator':14s} {'regime':8s} {'mean_rel_L1':>12s} {'mismatch':>14s}  verdict"
    print(hdr)
    print("-" * len(hdr))
    cases = [
        ("layer_norm", lambda r: _run_layer_norm(args.rows, args.cols, r)),
        ("addnorm pre", lambda r: _run_addnorm(args.rows, args.cols, r, True)),
        ("addnorm post", lambda r: _run_addnorm(args.rows, args.cols, r, False)),
    ]
    for name, fn in cases:
        for label, ratio in (("control", 0), ("offset", 32)):
            mr, bad, n, _ = fn(ratio)
            print(
                f"{name:14s} {label:8s} {mr:12.4g} {bad:7d}/{n:<6d}  "
                f"{'ok' if bad == 0 else 'COLLAPSED'}"
            )


def mode_sweep(args):
    import numpy as np
    from ml_dtypes import bfloat16
    from builders.addnorm import build_addnorm_module
    from air.backend.xrt import XRTBackend

    rows, cols = args.rows, args.cols
    residual, weight, ref = _addnorm_pieces(rows, cols, True)
    print(f"addnorm (pre-add) {rows}x{cols}, sigma=0.25, rtol={RTOL} atol={ATOL}")
    print("ONE binary; only the input distribution varies.\n")
    be = XRTBackend(
        verbose=False,
        omit_while_true_loop=False,
        output_format="xclbin",
        instance_name="an_sweep",
    )
    try:
        inv = be.compile_and_load(
            build_addnorm_module(rows, cols, bfloat16, pre_add=True)
        )
        hdr = f"{'|mean|/sigma':>13s} {'mean':>8s} {'mean_rel_L1':>12s} {'mismatch':>14s}  verdict"
        print(hdr)
        print("-" * len(hdr))
        for ratio in args.ratios:
            x = _draw(rows, cols, ratio)
            exp = ref(x, residual, weight)
            out = inv(x, residual, weight, np.zeros((rows, cols), dtype=bfloat16))[-1]
            mr, bad, n, _ = _stats(out.reshape(rows, cols), exp)
            print(
                f"{ratio:13.1f} {ratio * 0.25:8.2f} {mr:12.4g} {bad:7d}/{n:<6d}  "
                f"{'ok' if bad == 0 else 'COLLAPSED'}"
            )
    finally:
        try:
            be.unload()
        except Exception:
            pass


def mode_reachability(args):
    import numpy as np
    from pattern.reference import generate_golden_reference

    print(f"encoder_bert seq={args.seq} hidden=768 ffn=3072")
    print(
        f"addnorm collapses above |mean|/sigma = {CLIFF:.0f} (MEASURED, see --mode sweep)\n"
    )
    ref = generate_golden_reference(
        args.seq, 768, 3072, 12, workload_variant="encoder_bert", include_output=True
    )
    b = ref["boundaries"]
    x = np.asarray(ref["input"], dtype=np.float32)

    def summarize(name, arr):
        a = np.asarray(arr, dtype=np.float32)
        ratio = np.abs(a.mean(axis=1)) / np.maximum(a.std(axis=1), 1e-30)
        over = int((ratio > CLIFF).sum())
        print(
            f"  {name:24s} median={np.median(ratio):7.3f} "
            f"p99={np.percentile(ratio, 99):7.3f} max={ratio.max():7.3f}  "
            f"rows over {CLIFF:.0f}: {over}"
        )
        return over

    print("tensors the block's two addnorm dispatches take statistics of:")
    total = summarize("norm1 (attn_out + x)", np.asarray(b["attn_out"], np.float32) + x)
    total += summarize(
        "norm2 (ffn_out + hidden)",
        np.asarray(b["ffn_out"], np.float32) + np.asarray(b["hidden"], np.float32),
    )
    print()
    if total:
        print(f"REACHABLE: {total} row(s) over the cliff -- the defect is LIVE here.")
    else:
        print(
            "NOT reachable at this configuration -- the collapse is LATENT, and the\n"
            "recorded block/coarse/runlist/fused figures are not in question. It is\n"
            "still unpinned by any test, and a larger-mean workload would hit it silently."
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode", default="compare", choices=["compare", "sweep", "reachability"]
    )
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--cols", type=int, default=768)
    ap.add_argument("--seq", type=int, default=512, help="reachability only")
    ap.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=[0, 1, 2, 4, 8, 16, 32],
        help="sweep only",
    )
    args = ap.parse_args()

    if args.mode != "reachability":
        work = tempfile.mkdtemp(prefix=f"an-cliff-{args.mode}-")
        os.chdir(work)
        print(f"scratch: {work}")
    {"compare": mode_compare, "sweep": mode_sweep, "reachability": mode_reachability}[
        args.mode
    ](args)


if __name__ == "__main__":
    main()
