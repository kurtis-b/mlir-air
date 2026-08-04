# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase B hardware gate: runlist aggregation over separately-compiled artifacts.

Builds the separately-compiled GEMM ELFs the study's `offload` and `runlist` modes
dispatch — projection GEMMs of one Llama-3.2-1B decoder layer at seq_len 2048,
with tiles and method taken from `kernel_registry` — and then runs five legs
against them:

  A. **cross-artifact runlist** — the gate. Three *different* ELFs (q, gate and
     down projections, three distinct shapes) in one runlist, bit-exact against
     sequential dispatch and timed against it.
  A2. the seam still **refuses** the one aggregation that is silently wrong: a
     cross-artifact runlist under the xclbin ABI.
  B. **same-artifact aggregation** — the gate and up projections share a shape,
     hence one ELF, so their two dispatches can share one runlist. Isolates the
     submission saving from the cost of building the entries.
  C. **`KernelCache.run_sequence`** — the whole layer through the new seam, with
     BO pooling and dirty-bit sync, in one submission, bit-exact against per-GEMM
     `load_and_run`, run twice to prove the pool is reused.
  D. informational only: the pre-existing single-shot defect in the standalone
     drain GEMM ELF. See `LAYER_GEMMS` for why it is not part of the gate.

Every leg is anchored first: the baseline `load_and_run` results are checked
element-wise against an FP32 numpy oracle at the repository's bf16 GEMM
tolerance, because "identical to sequential dispatch" proves nothing if
sequential dispatch is itself wrong.

The one thing that does **not** work is what 05-phase-b proposed: binding several
full ELFs into a single `hw_context`. XRT rejects that three separate ways, and
`docs/plans/transformer-layer-execution-studies/05a-phase-b-runlist-spike-result.md`
records each. Aggregation does not need it — `xrt.runlist` dispatches each entry
against the context its kernel came from, so N ELFs means N contexts and still
one runlist.

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
- **Clobber the outputs before checking bit-identity.** Both aggregation legs
  reuse the BOs the sequential baseline wrote, so an entry the runlist silently
  skipped would still hold the right bytes. Leg A fills them with 0xA5 first.
- **A runlist saves one host submission, a fixed cost of order 100 us.** On these
  multi-millisecond GEMMs that is a few percent, which is smaller than thermal
  drift across a back-to-back benchmark. `interleaved_medians` alternates and
  reports the win count for that reason; do not replace it with two blocks.
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
from shared.infra.bo_pool import BufferSpec, DispatchStep, content_key  # noqa: E402
from shared.infra.cache import KernelCache, Profiler  # noqa: E402
from shared.infra.dispatch import RunlistSplitError, plan_submissions  # noqa: E402

#: Llama-3.2-1B decoder-layer projection GEMMs at seq_len 2048, as (label, M, K, N).
#: q/o and gate/up each share a shape, so these five compile to three distinct
#: ELFs. That mix is deliberate: leg A needs genuinely different artifacts and
#: leg B needs two dispatches of one, and both have to come from a real layer
#: rather than from a shape chosen to make the measurement convenient.
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


def upload_bos(backend, arrays_for_label):
    """One `xrt.ext.bo` per host array, filled and synced to device.

    `xrt.ext.bo` carries no memory group, so a BO allocated against one
    artifact's device is usable by any artifact's kernel — which is what lets the
    cross-artifact runlist below share the pool the sequential baseline used.
    """
    import pyxrt as xrt

    bos = []
    for arr in arrays_for_label:
        bo = xrt.ext.bo(backend.device, arr.size * arr.itemsize)
        src = np.frombuffer(
            arr.view(np.int16) if arr.dtype == bfloat16 else arr, dtype=np.uint8
        )
        np.copyto(
            np.frombuffer(bo.map(), dtype=np.uint8, count=len(src)), src, casting="no"
        )
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        bos.append(bo)
    return bos


def interleaved_medians(sequential, aggregated):
    """Median of alternating (sequential, aggregated) pairs, plus the win count.

    The saving is one host submission — a fixed cost of order 100 us — so on
    multi-millisecond GEMMs it is a low single-digit percentage, and thermal
    drift across a back-to-back A-then-B benchmark is larger than the effect.
    Alternating and taking medians removes the drift; two separate blocks do not.
    The win count is reported alongside because a median that moved by less than
    the sample spread is not a measurement.
    """

    def once(fn):
        t0 = time.perf_counter()
        fn()
        return (time.perf_counter() - t0) * 1000.0

    for _ in range(WARMUP_ITERS):
        sequential()
        aggregated()
    seq_samples, agg_samples = [], []
    for _ in range(TIMED_ITERS):
        seq_samples.append(once(sequential))
        agg_samples.append(once(aggregated))
    wins = sum(1 for s, a in zip(seq_samples, agg_samples) if a < s)
    return float(np.median(seq_samples)), float(np.median(agg_samples)), wins


def report_timing(seq_ms, agg_ms, wins, entries):
    saved = seq_ms - agg_ms
    print(
        f"    sequential ({entries} submissions) {seq_ms:.3f} ms   "
        f"runlist (1 submission) {agg_ms:.3f} ms   "
        f"saved {saved * 1000:.0f} us ({seq_ms / agg_ms:.4f}x)"
    )
    print(f"    runlist faster in {wins}/{TIMED_ITERS} interleaved pairs")
    return saved


def leg_a_cross_artifact_runlist(cache, specs, arrays, reference):
    """The gate: several *different* ELFs in one runlist.

    Three of the layer's projections — q, gate and down — have three distinct
    shapes, so they are three separately compiled artifacts. Each is loaded into
    its own `hw_context`; the runlist is built on one of those contexts and
    carries a run of every artifact's kernel. Checked bit-exact against the
    sequential `load_and_run` baseline and timed against sequential dispatch of
    the same three runs.

    This is what 05-phase-b assumed and what 05a first reported as impossible.
    Both are half right: the ELFs cannot be merged into *one* context (05a §§1-3
    stand), but they do not need to be. `xrt.runlist` dispatches each entry
    against the context its kernel came from, so N contexts and one runlist
    aggregate correctly. Every ordering, and every choice of which context hosts
    the runlist, was measured bit-identical.
    """
    import pyxrt as xrt

    labels = ("q_proj", "gate_proj", "down_proj")
    shape_of = {label: (m, k, n) for label, m, k, n in LAYER_GEMMS}
    names = [artifact_name(*shape_of[label]) for label in labels]
    if len(set(names)) != len(names):
        raise AssertionError(
            f"leg A needs distinct artifacts, got {names} — this leg is the gate "
            "and cannot be run on one ELF"
        )

    backends = {}
    for label, name in zip(labels, names):
        _, entry = specs[name]
        backends[label], _ = cache.ensure_loaded(name, backend_kwargs_for(name, entry))
    print(f"    {len(names)} separately-compiled ELFs, one hw_context each: {names}")

    # All BOs against one device wrapper, the way the pool allocates them.
    bosets = {label: upload_bos(backends[labels[0]], arrays[label]) for label in labels}

    def make_run(lab):
        run = xrt.run(backends[lab].kernel)
        for i, bo in enumerate(bosets[lab]):
            run.set_arg(i, bo)
        return run

    def sequential():
        for lab in labels:
            run = make_run(lab)
            run.start()
            run.wait2()

    def aggregated():
        # Rebuilt per call rather than hoisted: a hoisted runlist would measure
        # re-execution of a prepared object, but the study's modes build their
        # entries each layer. Leg B hoists on purpose, so the two bracket the
        # cost of construction.
        runlist = xrt.runlist(backends[labels[0]].context)
        for lab in labels:
            runlist.add(make_run(lab))
        runlist.execute()
        runlist.wait()

    # Clobber the outputs first, so bit-identity cannot be satisfied by a
    # leftover from the sequential baseline that the runlist never overwrote.
    for lab in labels:
        out = bosets[lab][-1]
        np.frombuffer(out.map(), dtype=np.uint8, count=reference[lab].nbytes)[:] = 0xA5
        out.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    aggregated()

    exact = True
    for lab in labels:
        bosets[lab][-1].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        got = np.frombuffer(
            bosets[lab][-1].map(), dtype=bfloat16, count=reference[lab].size
        ).reshape(reference[lab].shape)
        same = np.array_equal(got.view(np.int16), reference[lab].view(np.int16))
        print(f"    {lab}: bit-identical to sequential dispatch -> {same}")
        exact &= bool(same)

    seq_ms, agg_ms, wins = interleaved_medians(sequential, aggregated)
    report_timing(seq_ms, agg_ms, wins, len(labels))
    faster = agg_ms < seq_ms
    return exact and faster, {"seq_ms": seq_ms, "agg_ms": agg_ms, "wins": wins}


def leg_a2_seam_refuses_the_xclbin_case(cache):
    """The seam still refuses to aggregate across artifacts under the xclbin ABI.

    Not a hardware measurement — `plan_submissions` is pure — but it belongs
    next to leg A, because leg A is exactly the reason someone will be tempted to
    delete the refusal. 05a §4 measured the xclbin case: one context, entries
    from several artifacts, executes silently and returns wrong numbers for every
    entry but one. The ELF case is safe because each entry brings its own
    context; the xclbin case is not because the configuration is in the xclbin.
    """
    steps = [
        DispatchStep("art_a", ("x", "y", "z", "w"), (2, 3)),
        DispatchStep("art_b", ("p", "q", "r"), (2,)),
    ]
    binaries = {"art_a": "a.xclbin", "art_b": "b.xclbin"}
    try:
        plan_submissions(
            steps, binaries.__getitem__, require_single=True, elf_abi=False
        )
    except RunlistSplitError as exc:
        print(f"    xclbin ABI: refused, {str(exc).split('.')[0]}")
        return True
    print("    FAIL: the xclbin cross-artifact runlist was NOT refused (rule E2)")
    return False


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

    pairs = [
        (label, upload_bos(backend, arrays[label]))
        for label in ("gate_proj", "up_proj")
    ]

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

    # Hoisted on purpose, unlike leg A's: this leg isolates the submission cost
    # from the cost of building the entries, and leg A pays both.
    runlist = xrt.runlist(backend.context)
    runs = [make_run(bos) for _, bos in pairs]
    for run in runs:
        runlist.add(run)

    def aggregated():
        runlist.execute()
        runlist.wait()

    seq_ms, agg_ms, wins = interleaved_medians(sequential, aggregated)

    exact = True
    for label, bos in pairs:
        bos[-1].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        got = np.frombuffer(
            bos[-1].map(), dtype=bfloat16, count=reference[label].size
        ).reshape(reference[label].shape)
        same = np.array_equal(got.view(np.int16), reference[label].view(np.int16))
        print(f"    {label}: bit-identical to sequential dispatch -> {same}")
        exact &= bool(same)

    saved = report_timing(seq_ms, agg_ms, wins, len(pairs))
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
    real artifacts. The five GEMMs span three ELFs, and this asks for
    `require_single_submission=True`: the seam has to deliver on the real layer
    what leg A shows the hardware can do, not merely avoid crashing while
    splitting into three. A split here raises `RunlistSplitError` rather than
    quietly reporting three submissions.

    The B operand of each GEMM is declared **static**, which is what it is in a
    real layer: a weight. That puts it in the content-keyed pool, so it is
    uploaded on the first pass and never again — which only holds if the pool
    survives between dispatches.

    Runs the sequence twice. The second pass must be bit-identical to the first,
    must reuse the pool, and must move strictly fewer bytes. `run_sequence`
    builds a fresh `PoolPlan` every call, so if pools were keyed on plan object
    identity the second pass would silently allocate a second set of BOs and
    re-upload every weight — the defect this leg exists to catch.
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
            static = arg_name.endswith("_b")
            buf_specs[arg_name] = BufferSpec(
                name=arg_name,
                nbytes=arr.size * arr.itemsize,
                static=static,
                host_output=arg_name.endswith("_out"),
                content_key=content_key(arr) if static else None,
            )
            seq_arrays[arg_name] = arr

    kwargs = {
        name: backend_kwargs_for(name, entry) for name, (_, entry) in specs.items()
    }

    def run_once():
        return cache.run_sequence(
            steps, buf_specs, kwargs, seq_arrays, require_single_submission=True
        )

    results, vector = run_once()
    exact = True
    for label, _, _, _ in LAYER_GEMMS:
        got = results[f"{label}_out"]
        same = np.array_equal(got.view(np.int16), reference[label].view(np.int16))
        print(f"    {label}: bit-identical to load_and_run -> {same}")
        exact &= bool(same)

    # Copy before the second pass: the returned arrays are views into pool memory.
    first_pass = {k: np.array(v, copy=True) for k, v in results.items()}
    pools_after_first = set(cache._pools)
    results2, vector2 = run_once()
    pools_after_second = set(cache._pools)
    repeatable = all(
        np.array_equal(first_pass[k].view(np.int16), results2[k].view(np.int16))
        for k in first_pass
    )
    # One pool, the same pool. Keyed on `id(plan)` this would be two, because
    # every call builds a fresh plan object.
    reused = pools_after_second == pools_after_first and len(pools_after_first) == 1
    slots = max(p.stats()["slots"] for p in cache._pools.values())
    print(
        f"    second pass: bit-identical -> {repeatable}; one pool reused -> "
        f"{reused} ({len(pools_after_second)} pool(s), {slots} slots)"
    )

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
    aggregated = vector.host_submissions == 1
    if not aggregated:
        print(
            f"    FAIL: the layer took {vector.host_submissions} submissions, not one"
        )
    # The static-weight rule (S2) on hardware: the B operands are uploaded once,
    # so pass 2 moves strictly fewer bytes. If this ever reads equal, the pool
    # is being rebuilt per dispatch and every weight is going over the bus again.
    saved_mb = (vector.bytes_transferred - vector2.bytes_transferred) / 1e6
    print(
        f"    sync boundaries: {vector.sync_boundaries} on pass 1, "
        f"{vector2.sync_boundaries} on pass 2; "
        f"pass 2 moved {saved_mb:.0f} MB less (static weights already resident)"
    )
    weights_resident = vector2.bytes_transferred < vector.bytes_transferred
    if not weights_resident:
        print("    FAIL: pass 2 re-uploaded the static weights (rule S2)")
    return (
        exact and honest and aggregated and repeatable and reused and weights_resident,
        row,
    )


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

    print("\n== leg A: cross-artifact (multi-ELF) runlist — the gate's requirement ==")
    a_ok, _ = leg_a_cross_artifact_runlist(cache, specs, arrays, reference)

    print("\n== leg A2: the seam still refuses the xclbin cross-artifact runlist ==")
    a2_ok = leg_a2_seam_refuses_the_xclbin_case(cache)

    print("\n== leg B: same-artifact aggregation (gate_proj + up_proj share an ELF) ==")
    b_ok, _ = leg_b_same_artifact_aggregation(cache, specs, arrays, reference)

    print("\n== leg C: KernelCache.run_sequence over the whole layer ==")
    c_ok, _ = leg_c_run_sequence(cache, specs, arrays, reference)

    print("\n== leg D (informational): standalone drain ELF single-shot defect ==")
    leg_d_drain_single_shot(cache, specs, rng)

    print("\n" + "=" * 68)
    print(f"  leg A  cross-artifact multi-ELF runlist : {'PASS' if a_ok else 'FAIL'}")
    print(f"  leg A2 xclbin cross-artifact refused    : {'PASS' if a2_ok else 'FAIL'}")
    print(f"  leg B  same-artifact aggregation        : {'PASS' if b_ok else 'FAIL'}")
    print(f"  leg C  run_sequence over the layer      : {'PASS' if c_ok else 'FAIL'}")
    print("=" * 68)
    ok = a_ok and a2_ok and b_ok and c_ok
    if ok:
        print("\nPHASE B GATE: PASS")
    else:
        print(
            "\nPHASE B GATE: FAIL. Do not relax a leg to clear it — see\n"
            "docs/plans/transformer-layer-execution-studies/\n"
            "05a-phase-b-runlist-spike-result.md for what each leg measures and\n"
            "which failure means what."
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
