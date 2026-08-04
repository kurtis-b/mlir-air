# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase B hardware gate: runlist aggregation over separately-compiled artifacts.

Builds the separately-compiled GEMM ELFs the study's `offload` and `runlist` modes
dispatch — projection GEMMs of one Llama-3.2-1B decoder layer at seq_len 2048,
with tiles and method taken from `kernel_registry` — and then runs four legs
against them:

  A. **cross-artifact runlist** — the gate's requirement. Two *different* ELFs in
     one runlist. This is what 05-phase-b assumed would work.
  B. **same-artifact aggregation** — the gate and up projections share a shape,
     hence one ELF, so their two dispatches can share one runlist. Checked
     bit-exact against sequential dispatch and timed against it.
  C. **`KernelCache.run_sequence`** — the whole layer through the new seam, with
     BO pooling and dirty-bit sync, bit-exact against per-GEMM `load_and_run`.
  D. informational only: the pre-existing single-shot defect in the standalone
     drain GEMM ELF. See `LAYER_GEMMS` for why it is not part of the gate.

Every leg is anchored first: the baseline `load_and_run` results are checked
element-wise against an FP32 numpy oracle at the repository's bf16 GEMM
tolerance, because "identical to sequential dispatch" proves nothing if
sequential dispatch is itself wrong.

Leg A does not pass on XRT 2.21.0 / NPU2 and this script says so and exits
non-zero. The measurements behind that are written up in
`docs/plans/transformer-layer-execution-studies/05a-phase-b-runlist-spike-result.md`;
the short version is that an AIR ELF is a *full* ELF — it carries its own array
configuration — and a `hw_context` accepts exactly one of those. Legs B and C are
what the seam can actually deliver and are reported separately so the difference
is visible rather than averaged away.

Run it under the repository's NPU lock, which is a different inode from the
`/tmp/npu.lock` `KernelCache` serializes on:

    flock -x -w 1800 /tmp/mlir-air-npu.lock make runlist-gate

Footguns:

- **Compilation is not free.** Four distinct ELFs at seq_len 2048; budget a couple
  of minutes on a warm `mm_*.o`. `--run-only` reuses the cached manifest.
- **`instance_name` must match the module's entry function**, which differs by GEMM
  method: `matmul_bf16` for drain, `gemm_cast_bf16` for fused-cast. Getting it
  wrong produces an ELF that loads and then times out.
- **`runtime_loop_tiling_sizes` and `stack_size` are not optional** — see
  `backend_kwargs_for`. Omitting them yields an ELF that runs and returns numbers
  that change from call to call.
- **Leg B needs two dispatches of one artifact to be a fair comparison.** Gate and
  up are that pair in a real layer; do not substitute two runs of the same
  buffers, which would measure a warm cache rather than aggregation.
- **Leg A must run last.** Probing the multi-ELF API costs an extra `hw_context`
  on a device that already holds one per loaded artifact, and leaving it alive
  moves both the numerics and the timings of anything measured after it.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_HERE = Path(__file__).resolve().parent
_EXAMPLES = _HERE.parent
for _p in (_EXAMPLES, _EXAMPLES / "llms", _EXAMPLES / "llms" / "shared"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from shared.builders.gemm_builder import (  # noqa: E402
    _build_gemm_module,
    gemm_registry_config,
)
from shared.infra.bo_pool import BufferSpec, DispatchStep  # noqa: E402
from shared.infra.cache import KernelCache, Profiler  # noqa: E402
from shared.infra.dispatch import RunlistSplitError, plan_submissions  # noqa: E402

#: Llama-3.2-1B decoder-layer projection GEMMs at seq_len 2048, as (label, M, K, N).
#: q/o and gate/up each share a shape, so these five compile to three distinct
#: ELFs — which is what makes same-artifact aggregation measurable even though
#: cross-artifact aggregation is unavailable.
#:
#: The layer's K and V projections (2048x2048x512) are **not** here. The registry
#: picks the `drain` method for that shape, and a standalone drain ELF is correct
#: only on its first invocation after load; every call after that is wrong,
#: whatever BO set it is given. That is a pre-existing defect on a path the
#: shipped deployments do not use — they reach drain GEMMs only inside fused
#: multi-launch ELFs — and it is unrelated to runlist aggregation. `leg_d` below
#: measures it every run so it stays visible instead of rotting in a document.
LAYER_GEMMS = [
    ("q_proj", 2048, 2048, 2048),
    ("o_proj", 2048, 2048, 2048),
    ("gate_proj", 2048, 2048, 8192),
    ("up_proj", 2048, 2048, 8192),
    ("down_proj", 2048, 8192, 2048),
]

#: The drain shape leg D probes. Kept out of LAYER_GEMMS; see the note above.
DRAIN_SHAPE = ("kv_proj", 2048, 2048, 512)

WARMUP_ITERS = 3
TIMED_ITERS = 15


def artifact_name(m, k, n):
    """Artifacts are keyed by shape: same shape, same compiled ELF."""
    return f"gemm_{m}x{k}x{n}"


def backend_kwargs_for(name, entry):
    """Backend preset for a standalone GEMM ELF.

    `runtime_loop_tiling_sizes` and `stack_size` are not optional: they match
    `matrix_multiplication/bf16_in_bf16_out/run.py`'s own runner and the shipped
    `O_FFN_BACKEND` preset. Dropping them compiles and loads fine and then
    produces results that are wrong on the first call and *different* on every
    call after it — a nondeterminism that reads as a runlist bug when it is
    nothing of the kind.
    """
    return {
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": entry,
        "runtime_loop_tiling_sizes": [2, 2],
        "stack_size": 2048,
    }


def build_artifacts(cache, run_only=False):
    """Compile one ELF per distinct GEMM shape in the layer.

    Returns dict artifact name -> (config, entry function name).
    """
    from shared.infra.external_kernels import compile_gemm_mm

    shapes = {}
    for _, m, k, n in LAYER_GEMMS + [DRAIN_SHAPE]:
        shapes.setdefault(artifact_name(m, k, n), (m, k, n))

    specs = {}
    for name, (m, k, n) in shapes.items():
        cfg = gemm_registry_config(m, k, n)
        entry = "gemm_cast_bf16" if cfg["needs_f32_scratch"] else "matmul_bf16"
        specs[name] = (cfg, entry)

    if run_only and cache.load_manifest():
        missing = [n for n in shapes if n not in cache.artifacts]
        if not missing:
            print("  reusing cached artifacts")
            return specs
        print(f"  cache miss for {missing}; compiling")

    # The suffixed mm.o variants must exist before any compile_and_cache, so
    # prepare_air_project stages them into air_project/ for every ELF that links
    # them: drain links _m32, fused-cast links _m64.
    compile_gemm_mm(
        tile_m=32, tile_n=128, tile_k_l1=32, sym_suffix="_m32", out_name="mm_m32.o"
    )
    compile_gemm_mm(
        tile_m=64, tile_n=128, tile_k_l1=32, sym_suffix="_m64", out_name="mm_m64.o"
    )

    for name, (m, k, n) in shapes.items():
        cfg, entry = specs[name]
        module = _build_gemm_module(
            m,
            k,
            n,
            cfg["tile_m"],
            cfg["tile_k_l2"],
            cfg["tile_k_l1"],
            cfg["tile_n"],
            **cfg["build_kwargs"],
        )
        cache.compile_and_cache(name, module, backend_kwargs_for(name, entry))
    cache._save_manifest()
    return specs


def gemm_args(rng, cfg, m, k, n):
    """Host arrays for one GEMM, in the artifact's argument order.

    fused-cast takes an extra f32 C scratch between B and the bf16 output; drain
    does not. Passing the wrong count is an ERT timeout, not an error message.

    Inputs are scaled by 1/sqrt(K), matching the example's own harness, so the
    accumulated magnitudes stay in the range the tiles were measured at.
    """
    scale = 1.0 / np.sqrt(k)
    args = [
        (rng.standard_normal((m, k)) * scale).astype(bfloat16),
        (rng.standard_normal((k, n)) * scale).astype(bfloat16),
    ]
    if cfg["needs_f32_scratch"]:
        args.append(np.zeros((m, n), dtype=np.float32))
    args.append(np.zeros((m, n), dtype=bfloat16))
    return args


def sequential_reference(cache, specs, arrays):
    """Per-GEMM `load_and_run`, the baseline every leg is compared against.

    Also checks each result against an FP32 numpy oracle at the repository's bf16
    GEMM tolerance. "Numerically identical to sequential dispatch" is worth
    nothing if sequential dispatch is itself wrong, so the baseline is anchored
    before anything is compared to it.

    Returns (results, baseline_is_correct).
    """
    # matrix_multiplication/bf16_in_bf16_out/run.py's high-precision tier: rtol
    # anchors to torch's bf16 standard, atol to the measured worst-case abs err.
    rtol, atol = 1.6e-2, 1.5e-3
    out = {}
    ok = True
    for label, m, k, n in LAYER_GEMMS:
        name = artifact_name(m, k, n)
        cfg, entry = specs[name]
        res = cache.load_and_run(
            name,
            backend_kwargs_for(name, entry),
            *arrays[label],
            bo_key=label,
        )
        got = np.array(res[-1], copy=True).reshape(m, n)
        out[label] = got
        oracle = arrays[label][0].astype(np.float32) @ arrays[label][1].astype(
            np.float32
        )
        close = np.isclose(got.astype(np.float32), oracle, rtol=rtol, atol=atol)
        frac = close.mean()
        if not close.all():
            ok = False
        print(
            f"    {label} ({cfg['method']}): {frac * 100:.3f}% of elements within "
            f"rtol={rtol} atol={atol} of the fp32 oracle"
        )
    return out, ok


def leg_a_cross_artifact_runlist(cache, specs):
    """The gate's requirement: two different ELFs, one runlist.

    Returns (passed, detail). Never builds the runlist when the artifacts differ:
    05a §4 measured that such a runlist executes and returns wrong numbers with no
    error raised, so emitting it to "see what happens" would corrupt the very
    comparison the gate exists to make.

    Runs **last**, and drops its XRT objects before returning. Probing the API
    means creating an extra `hw_context` on a device that already holds one per
    loaded artifact; leaving it alive perturbs both the numerics and the timing of
    anything measured afterwards.
    """
    import gc

    import pyxrt as xrt

    a, b = artifact_name(2048, 2048, 2048), artifact_name(2048, 2048, 512)
    detail = []

    # 1. What the seam does: refuse, with the reason.
    steps = [
        DispatchStep(a, ("x", "y", "z", "w"), (2, 3)),
        DispatchStep(b, ("p", "q", "r"), (2,)),
    ]
    try:
        plan_submissions(
            steps, lambda k: cache.artifacts[k].output_binary, require_single=True
        )
        detail.append("plan_submissions did NOT refuse a cross-artifact runlist")
        return False, detail
    except RunlistSplitError:
        detail.append("plan_submissions refuses the cross-artifact runlist (rule E2)")

    # 2. Why it refuses: XRT will not put the second ELF into the first's context.
    unexpected = False
    try:
        dev = xrt.device(0)
        elf_a = xrt.elf(cache.artifacts[a].output_binary)
        elf_b = xrt.elf(cache.artifacts[b].output_binary)
        ctx = xrt.hw_context(dev, elf_a)
        for label, elf in (
            ("module(elf_b)->ctx", elf_b),
            ("module(elf_a)->ctx", elf_a),
        ):
            try:
                xrt.module(xrt.module(elf), ctx)
                detail.append(
                    f"UNEXPECTED: {label} succeeded — re-check 05a, the "
                    f"platform may have gained multi-ELF contexts"
                )
                unexpected = True
            except Exception as exc:
                detail.append(f"{label}: {exc}")
        try:
            xrt.ext.kernel(ctx, xrt.module(elf_b), cache.artifacts[b].kernel)
            detail.append(
                "UNEXPECTED: ext.kernel(ctx, module, name) accepted a full ELF"
            )
            unexpected = True
        except Exception as exc:
            detail.append(f"ext.kernel(ctx, module(elf_b), name): {exc}")
    finally:
        ctx = elf_a = elf_b = dev = None
        gc.collect()

    return unexpected, detail


def leg_d_drain_single_shot(cache, specs, rng):
    """Informational: is the standalone drain ELF still single-shot?

    Not a Phase B gate — it measures a pre-existing defect on a path no shipped
    deployment uses. It runs anyway so that the finding is re-measured on every
    gate run rather than living only in a document, and so that a fix is noticed.
    """
    label, m, k, n = DRAIN_SHAPE
    name = artifact_name(m, k, n)
    cfg, entry = specs[name]
    args = gemm_args(rng, cfg, m, k, n)
    oracle = args[0].astype(np.float32) @ args[1].astype(np.float32)

    fracs = []
    for _ in range(2):
        res = cache.load_and_run(
            name, backend_kwargs_for(name, entry), *args, bo_key=label
        )
        got = np.array(res[-1], copy=True).reshape(m, n).astype(np.float32)
        fracs.append(
            float(np.isclose(got, oracle, rtol=1.6e-2, atol=1.5e-3).mean()) * 100
        )
    print(
        f"    {name} ({cfg['method']}): call 1 {fracs[0]:.2f}% correct, "
        f"call 2 {fracs[1]:.2f}% correct"
    )
    if fracs[1] > 99.0:
        print("    the single-shot drain defect appears to be FIXED — put the K/V")
        print("    projections back into LAYER_GEMMS and delete this leg.")
    else:
        print("    still single-shot: correct once per load, wrong thereafter.")
    return fracs


def leg_b_same_artifact_aggregation(cache, specs, arrays, reference):
    """Gate and Up share an ELF, so their dispatches can share one runlist.

    Checks bit-exactness against `reference` and times aggregated vs sequential.
    """
    import pyxrt as xrt

    name = artifact_name(2048, 2048, 8192)
    cfg, entry = specs[name]
    backend, _ = cache.ensure_loaded(name, backend_kwargs_for(name, entry))

    pairs = []
    for label in ("gate_proj", "up_proj"):
        bos = []
        for arr in arrays[label]:
            bo = xrt.ext.bo(backend.device, arr.size * arr.itemsize)
            src = np.frombuffer(
                arr.view(np.int16) if arr.dtype == bfloat16 else arr, dtype=np.uint8
            )
            np.copyto(
                np.frombuffer(bo.map(), dtype=np.uint8, count=len(src)),
                src,
                casting="no",
            )
            bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            bos.append(bo)
        pairs.append((label, bos))

    def make_run(bos):
        run = xrt.run(backend.kernel)
        for i, bo in enumerate(bos):
            run.set_arg(i, bo)
        return run

    def sequential():
        for _, bos in pairs:
            run = make_run(bos)
            run.start()
            run.wait2()

    runlist = xrt.runlist(backend.context)
    runs = [make_run(bos) for _, bos in pairs]
    for run in runs:
        runlist.add(run)

    def aggregated():
        runlist.execute()
        runlist.wait()

    def once(fn):
        t0 = time.perf_counter()
        fn()
        return (time.perf_counter() - t0) * 1000.0

    # Interleaved, median of N. The saving is one host submission — a fixed cost
    # of order 100 us — so on 9 ms kernels it is ~1% and thermal drift over a
    # back-to-back A-then-B benchmark is larger than the effect being measured.
    # Alternating and taking medians removes the drift; a mean of two separate
    # blocks does not.
    for _ in range(WARMUP_ITERS):
        sequential()
        aggregated()
    seq_samples, agg_samples = [], []
    for _ in range(TIMED_ITERS):
        seq_samples.append(once(sequential))
        agg_samples.append(once(aggregated))
    seq_ms = float(np.median(seq_samples))
    agg_ms = float(np.median(agg_samples))

    exact = True
    for label, bos in pairs:
        bos[-1].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        got = np.frombuffer(
            bos[-1].map(), dtype=bfloat16, count=reference[label].size
        ).reshape(reference[label].shape)
        same = np.array_equal(got.view(np.int16), reference[label].view(np.int16))
        print(f"    {label}: bit-identical to sequential dispatch -> {same}")
        exact &= bool(same)

    saved = seq_ms - agg_ms
    print(
        f"    sequential (2 submissions) {seq_ms:.3f} ms   "
        f"runlist (1 submission) {agg_ms:.3f} ms   "
        f"saved {saved * 1000:.0f} us ({seq_ms / agg_ms:.3f}x)"
    )
    print(
        f"    the saving is one host submission, so it is a fixed cost: it is "
        f"{saved / seq_ms * 100:.1f}% of a {seq_ms / 2:.1f} ms/GEMM pair here and"
    )
    print(
        "    grows as the entries get smaller — which is the axis the study measures."
    )
    faster = agg_ms < seq_ms
    return exact and faster, {"seq_ms": seq_ms, "agg_ms": agg_ms, "saved_ms": saved}


def leg_c_run_sequence(cache, specs, arrays, reference):
    """The whole layer through `KernelCache.run_sequence`, bit-exact vs the baseline.

    Exercises the pooled BOs, the dirty-bit sync and the dispatch vector on the
    real artifacts. The sequence is not aggregatable into one submission (the
    seven GEMMs span four ELFs), so this also checks that the vector reports the
    split honestly instead of claiming one submission.
    """
    steps = []
    buf_specs = {}
    seq_arrays = {}
    for label, m, k, n in LAYER_GEMMS:
        name = artifact_name(m, k, n)
        cfg, _ = specs[name]
        arg_names = [f"{label}_a", f"{label}_b"]
        if cfg["needs_f32_scratch"]:
            arg_names.append(f"{label}_scratch")
        arg_names.append(f"{label}_out")
        writes = (
            (len(arg_names) - 2, len(arg_names) - 1)
            if cfg["needs_f32_scratch"]
            else (len(arg_names) - 1,)
        )
        steps.append(DispatchStep(name, tuple(arg_names), writes))
        for arg_name, arr in zip(arg_names, arrays[label]):
            buf_specs[arg_name] = BufferSpec(
                name=arg_name,
                nbytes=arr.size * arr.itemsize,
                host_output=arg_name.endswith("_out"),
            )
            seq_arrays[arg_name] = arr

    results, vector = cache.run_sequence(
        steps,
        buf_specs,
        {name: backend_kwargs_for(name, entry) for name, (_, entry) in specs.items()},
        seq_arrays,
    )

    exact = True
    for label, m, n in ((l, m, n) for l, m, _, n in LAYER_GEMMS):
        got = results[f"{label}_out"]
        same = np.array_equal(got.view(np.int16), reference[label].view(np.int16))
        print(f"    {label}: bit-identical to load_and_run -> {same}")
        exact &= bool(same)

    row = vector.as_row()
    print(f"    dispatch vector: {row}")
    print(
        f"    submissions {vector.host_submissions} over {vector.runlist_entries} "
        f"entries ({vector.per_submission_entries}), "
        f"submit {vector.submission_ms:.2f} ms, sync {vector.sync_ms:.2f} ms"
    )
    honest = vector.host_submissions == len(vector.per_submission_entries)
    if not honest:
        print("    FAIL: the vector's submission count does not match what ran")
    return exact and honest, row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-only",
        action="store_true",
        help="reuse cached ELFs if the manifest has them",
    )
    parser.add_argument("--cache-dir", default=str(_HERE / "runlist_gate_cache"))
    args = parser.parse_args()

    cache = KernelCache(
        cache_dir=args.cache_dir, verbose=True, profiler=Profiler(enabled=False)
    )
    print("== compiling the layer's separately-compiled GEMM ELFs ==")
    specs = build_artifacts(cache, run_only=args.run_only)

    rng = np.random.default_rng(0)
    arrays = {}
    for label, m, k, n in LAYER_GEMMS:
        cfg, _ = specs[artifact_name(m, k, n)]
        arrays[label] = gemm_args(rng, cfg, m, k, n)

    print("\n== baseline: per-GEMM load_and_run, checked against an fp32 oracle ==")
    reference, baseline_ok = sequential_reference(cache, specs, arrays)
    if not baseline_ok:
        print(
            "    FAIL: the sequential baseline does not match the fp32 oracle, so\n"
            "    'identical to sequential dispatch' would prove nothing. Stopping."
        )
        return 1

    print("\n== leg B: same-artifact aggregation (gate_proj + up_proj share an ELF) ==")
    b_ok, _ = leg_b_same_artifact_aggregation(cache, specs, arrays, reference)

    print("\n== leg C: KernelCache.run_sequence over the whole layer ==")
    c_ok, _ = leg_c_run_sequence(cache, specs, arrays, reference)

    # Last: probing the multi-ELF API costs an extra hw_context on a device that
    # already holds one per loaded artifact, which moves the numbers above.
    print("\n== leg A: cross-artifact (multi-ELF) runlist — the gate's requirement ==")
    a_ok, a_detail = leg_a_cross_artifact_runlist(cache, specs)
    for line in a_detail:
        print(f"    {line}")

    print("\n== leg D (informational): standalone drain ELF single-shot defect ==")
    leg_d_drain_single_shot(cache, specs, rng)

    print("\n" + "=" * 68)
    print(f"  leg A  cross-artifact multi-ELF runlist : {'PASS' if a_ok else 'FAIL'}")
    print(f"  leg B  same-artifact aggregation        : {'PASS' if b_ok else 'FAIL'}")
    print(f"  leg C  run_sequence over the layer      : {'PASS' if c_ok else 'FAIL'}")
    print("=" * 68)
    if not a_ok:
        print(
            "\nPHASE B GATE: FAIL — the multi-ELF runlist the gate requires is not\n"
            "available on this stack. Legs B and C show what the seam does deliver.\n"
            "See docs/plans/transformer-layer-execution-studies/\n"
            "05a-phase-b-runlist-spike-result.md for the measurements and the three\n"
            "remaining routes, none of which is a Phase B change."
        )
    elif b_ok and c_ok:
        print("\nPHASE B GATE: PASS")
    return 0 if (a_ok and b_ok and c_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
