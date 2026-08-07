#!/usr/bin/env python3
"""What did layer_norm's two-pass f32 variance cost? Measured: ~13%.

J7a's round-3 fix (commit 46451171) changed programming_examples/layer_norm/layer_norm.cc
from one-pass bf16 statistics to two-pass f32 -- a ~26x accuracy win that reads each row
TWICE, with the throughput cost unmeasured. Two execution modes dispatch this kernel and
Phase F will report latency built on it. Doc 23 open item 1.

RESULT (50 timed invocations each)

    shape       one-pass    two-pass    cost (min)   accuracy
    4096x768    4.835 ms    5.474 ms    +13.2%       1.969e-3 -> 7.117e-5  (27.7x)
    512x512     0.406 ms    0.461 ms    +13.5%       2.009e-3 -> 8.082e-5  (24.9x)

COMPARE MINIMUMS, NOT MEDIANS. The one-pass runs carry more host jitter (p90 6.447 ms vs the
two-pass 5.828 ms at 4096x768), which flatters them on the median to +9.2%. The min-to-min
figures agree to 0.3 points across a 12x shape range: ~13% is the kernel, the rest is dispatch.

METHOD
    Compile BOTH kernel sources to layer_norm.o in a scratch cwd, one at a time; build the same
    module via builders/layer_norm.py; compile_and_load ONCE; then time N invocations of the
    loaded artifact. Compilation is outside the timed region, so what is measured is dispatch
    plus kernel, not aiecc.

THE PROVENANCE CROSS-CHECK, which is the part worth keeping
    The one-pass build must reproduce mean_rel_L1 ~1.969e-3 at 4096x768 -- the exact figure
    opcheck_specs.py records for the run that sized that row's atol -- and ~2.0e-3 -> 8.1e-5 at
    512x512, matching doc 23. Both landed. Without it, a "before" build that is not actually
    the old kernel yields a plausible, meaningless number, and nothing would say so.

NOT a test. Nothing runs this in CI. It exists so the number can be re-derived.
"""

import argparse, os, statistics, subprocess, sys, tempfile, time
from pathlib import Path

REPO = Path("/home/cj/mlir-air")
PE = REPO / "programming_examples"
for p in (PE, PE / "llms", PE / "transformer_layer"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# The commit whose PARENT holds the one-pass kernel.
J7A_FIX = "46451171"
KERNEL_REL = "programming_examples/layer_norm/layer_norm.cc"


def old_kernel_source(dest: Path) -> Path:
    """Extract the pre-J7a one-pass kernel from git history."""
    out = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{J7A_FIX}^:{KERNEL_REL}"],
        capture_output=True,
        text=True,
        check=True,
    )
    dest.write_text(out.stdout)
    return dest


def measure(label, src_path, rows, cols, reps, warmup):
    """Compile src_path -> layer_norm.o in CWD, build, load, time `reps` invocations."""
    import numpy as np
    from ml_dtypes import bfloat16
    from shared.infra import external_kernels as ek
    from builders.layer_norm import build_layer_norm_module, layer_norm_reference
    from air.backend.xrt import XRTBackend

    print(f"\n=== {label} ===")
    print(f"  source: {src_path}")
    ek._compile_kernel(
        str(src_path), "layer_norm.o", extra_flags=["-DLN_VEC_LEN=16"], force=True
    )

    rng = np.random.default_rng(2)  # same seed as prepare_layer_norm
    x = rng.standard_normal((rows, cols)).astype(bfloat16)
    expected = layer_norm_reference(x)

    module = build_layer_norm_module(rows, cols, bfloat16)
    backend = XRTBackend(
        verbose=False,
        omit_while_true_loop=False,
        output_format="xclbin",
        instance_name="layer_norm_bench",
    )
    try:
        invoker = backend.compile_and_load(module)

        # Correctness + the provenance cross-check, before any timing.
        outs = invoker(x, np.zeros_like(expected))
        got = outs[-1].reshape(rows, cols)
        err = np.abs(got.astype(np.float32) - expected.astype(np.float32))
        denom = np.abs(expected.astype(np.float32))
        nz = denom > 0
        mean_rel_L1 = (
            float(err[nz].sum() / denom[nz].sum()) if nz.any() else float("nan")
        )

        for _ in range(warmup):
            invoker(x, np.zeros_like(expected))

        samples = []
        for _ in range(reps):
            buf = np.zeros_like(expected)
            t0 = time.perf_counter()
            invoker(x, buf)
            samples.append((time.perf_counter() - t0) * 1000.0)
    finally:
        try:
            backend.unload()
        except Exception:
            pass

    samples.sort()
    return {
        "label": label,
        "mean_rel_L1": mean_rel_L1,
        "min_ms": samples[0],
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "p90_ms": samples[int(0.9 * (len(samples) - 1))],
        "n": len(samples),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4096)
    ap.add_argument("--cols", type=int, default=768)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix="ln-bench-"))
    os.chdir(work)
    print(
        f"scratch: {work}   shape: {args.rows}x{args.cols}   "
        f"reps: {args.reps} (+{args.warmup} warmup)"
    )

    old_src = old_kernel_source(work / "layer_norm_onepass.cc")
    cur_src = REPO / KERNEL_REL

    before = measure(
        "BEFORE — one-pass bf16 (pre-J7a)",
        old_src,
        args.rows,
        args.cols,
        args.reps,
        args.warmup,
    )
    after = measure(
        "AFTER — two-pass f32 (HEAD)",
        cur_src,
        args.rows,
        args.cols,
        args.reps,
        args.warmup,
    )

    print("\n" + "=" * 72)
    print(f"layer_norm {args.rows}x{args.cols}, {before['n']} timed invocations each")
    print("=" * 72)
    hdr = f"{'variant':34s} {'median':>9s} {'min':>9s} {'p90':>9s} {'mean_rel_L1':>12s}"
    print(hdr)
    for r in (before, after):
        print(
            f"{r['label']:34s} {r['median_ms']:8.3f}m {r['min_ms']:8.3f}m "
            f"{r['p90_ms']:8.3f}m {r['mean_rel_L1']:12.4g}"
        )
    ratio = after["median_ms"] / before["median_ms"]
    print(
        f"\ntwo-pass / one-pass median: {ratio:.3f}x  "
        f"({(ratio - 1) * 100:+.1f}% wall per dispatch)"
    )
    print(
        f"accuracy: {before['mean_rel_L1']:.4g} -> {after['mean_rel_L1']:.4g}  "
        f"({before['mean_rel_L1'] / max(after['mean_rel_L1'], 1e-30):.1f}x better)"
    )
    print(
        "\nCROSS-CHECK: the one-pass mean_rel_L1 should land near 1.969e-3 at 4096x768 "
        "(opcheck_specs.py).\nIf it does not, the BEFORE build is not the old kernel and "
        "the timing above is meaningless."
    )


if __name__ == "__main__":
    main()
