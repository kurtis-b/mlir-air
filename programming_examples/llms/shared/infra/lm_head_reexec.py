# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The LM-head GEMV re-execution gate, shared by the model drivers.

One kernel compiled fresh from a model's production builder, dispatched seven
times (first; back-to-back x2; after 0.5 s idle; new input; after another ELF;
back-to-back again), every dispatch judged fail-closed against an f32
reference: clean = finite, no element beyond 1e-2 of the output scale (the
honest bf16 error is 2.5e-3) and bit-identical to the first dispatch for the
same input. PASS requires 7/7. The history (the 19 x 8192 family, the LOAD_PDI
parity mechanism and its compiler fix) is in qwen3_0_6b/lm_head_reexec_gate.py
and doc 57 section 1.5. `[2026-08-23]` factored out of that script so the
Llama-3.2-1B head (item 12) runs the same gate.
"""

import sys
import time

import numpy as np
from ml_dtypes import bfloat16

TINY_M = 64


def run_gate(
    cache_dir, build_module, parts, backend, emb, tag="production lm_head_gemv"
):
    """`build_module()` -> the model's production LM-head module over `parts`
    (the partition row counts, in ELF arg order); `backend` its aircc kwargs;
    `emb` the input width. Returns the process exit code (0 = 7/7 clean)."""
    from matvec import build_module as build_gemv
    from shared.infra.cache import KernelCache

    parts = list(parts)
    offs = [0]
    for r in parts:
        offs.append(offs[-1] + r)
    n_rows = offs[-1]
    other_backend = {**backend, "instance_name": "matvec_bf16"}

    cache = KernelCache(cache_dir=str(cache_dir))
    t0 = time.perf_counter()
    cache.compile_and_cache("lm_head_gemv", build_module(), backend)
    cache.compile_and_cache(
        "other_gemv",
        build_gemv(TINY_M, emb, 8, 4, 8, bfloat16, bfloat16),
        other_backend,
    )
    cache._save_manifest()
    print(
        f"[reexec] compiled {tag} (partitions {parts}) + a 1-launch other ELF in "
        f"{time.perf_counter() - t0:.0f}s",
        flush=True,
    )

    rng = np.random.default_rng(0)
    W = (rng.standard_normal((n_rows, emb), dtype=np.float32) * 0.02).astype(bfloat16)
    xs = [rng.standard_normal(emb, dtype=np.float32).astype(bfloat16) for _ in range(2)]
    refs = [x.astype(np.float32) @ W.astype(np.float32).T for x in xs]
    Wt = (rng.standard_normal((TINY_M, emb), dtype=np.float32) * 0.02).astype(bfloat16)
    n = len(parts)

    def run_head(x):
        ins = [x]
        for p, rows in enumerate(parts):
            ins.append(np.ascontiguousarray(W[offs[p] : offs[p + 1]]))
            ins.append(np.zeros(rows, dtype=bfloat16))
        res = cache.load_and_run(
            "lm_head_gemv",
            backend,
            *ins,
            output_indices=[2 + 2 * p for p in range(n)],
            static_input_indices={1 + 2 * p for p in range(n)},
            intermediate_indices={2 + 2 * p for p in range(n)},
        )
        return np.concatenate(
            [np.asarray(res[2 + 2 * p]).astype(np.float32) for p in range(n)]
        )

    def run_other(x):
        cache.load_and_run(
            "other_gemv",
            other_backend,
            Wt,
            x,
            np.zeros(TINY_M, dtype=bfloat16),
            output_indices=[2],
            static_input_indices={0},
            intermediate_indices={2},
        )

    def judge(label, out, ref, base):
        scale = np.max(np.abs(ref)) + 1e-9
        rel = np.abs(out - ref) / scale
        bad = np.nonzero(rel > 1e-2)[0]
        finite = bool(np.all(np.isfinite(out)))
        same = bool(np.array_equal(out, base)) if base is not None else True
        bad_parts = sorted(
            {int(np.searchsorted(offs, i, side="right") - 1) for i in bad}
        )
        clean = finite and bad.size == 0 and same
        print(
            f"[reexec] {label}: {'clean' if clean else 'WRONG'} -- max_rel {rel.max():.2e} "
            f"n_bad {bad.size}/{ref.size} partitions {bad_parts} "
            f"first_rows {[int(i - offs[np.searchsorted(offs, i, side='right') - 1]) for i in bad[:6]]}"
            f"{'' if base is None else f' bit_identical_to_first {same}'} finite {finite}",
            flush=True,
        )
        return clean

    # Non-vacuity control, run every time and before anything is judged for
    # real: a gate that has never been seen to say WRONG is not evidence when
    # it says clean. Corrupt one element of a known-good vector by 10x the
    # threshold and require the same `judge` to reject it. Costs no device
    # time -- it reuses the reference as the "output".
    probe = refs[0].copy()
    probe[0] += 10e-2 * (np.max(np.abs(refs[0])) + 1e-9)
    if judge("c0 CONTROL (deliberately corrupted)", probe, refs[0], None):
        print(
            "[reexec] CONTROL DID NOT FIRE -- the judge cannot detect a bad "
            "element, so a clean verdict below would mean nothing"
        )
        return 2

    v = []
    d1 = run_head(xs[0])
    v.append(judge("d1 first dispatch", d1, refs[0], None))
    d2 = run_head(xs[0])
    v.append(judge("d2 back-to-back", d2, refs[0], d1))
    d3 = run_head(xs[0])
    v.append(judge("d3 back-to-back", d3, refs[0], d1))
    print(
        f"[reexec] d2 vs d3 max abs diff {np.max(np.abs(d2 - d3)):.3e} (non-zero = non-deterministic)"
    )
    time.sleep(0.5)
    d4 = run_head(xs[0])
    v.append(judge("d4 after 0.5 s idle", d4, refs[0], d1))
    d5 = run_head(xs[1])
    v.append(judge("d5 new input back-to-back", d5, refs[1], None))
    run_other(xs[0])
    d6 = run_head(xs[0])
    v.append(judge("d6 after other ELF", d6, refs[0], d1))
    d7 = run_head(xs[0])
    v.append(judge("d7 back-to-back after d6", d7, refs[0], d1))
    ok = sum(v)
    print(f"[reexec] {ok}/{len(v)} dispatches clean")
    print("[reexec] PASS" if ok == len(v) else "[reexec] FAIL")
    return 0 if ok == len(v) else 1
