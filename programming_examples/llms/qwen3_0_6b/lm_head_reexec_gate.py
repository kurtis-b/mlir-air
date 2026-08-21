# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The LM-head GEMV's RE-EXECUTION gate: the production ELF, dispatched back-to-back.

`[2026-08-21]` Doc 57 section 1.4 found (devq 446-448) and this gate confirms
on the production artifact (devq 452) that the 19 x 8192 `lm_head_gemv`
multi-launch ELF returns WRONG, NON-DETERMINISTIC values when it executes
immediately after itself: partition 0 (the first launch), ~700-2,800 of its
8192 rows (rows 24, 32, 40, 64.. -- tile boundaries), max error ~0.9 of the
output scale, different on every repeat. It is exact on the first dispatch
after load, exact after ANY other ELF has run in between, and wrong again on
the very next back-to-back dispatch. A 0.5 s host idle between dispatches
does NOT heal it, so it is stale partition/context state, not a race with
in-flight work. Production decode never runs the head twice in a row (an
`o_gemv_ffn` always precedes it) and the verify gate judges token sets, not
logits, so `make verify` cannot see it; any batched-logits, re-scoring or
per-token runlist design that makes the head adjacent to itself will.

Shape of `study/fused_reexec_gate.py`: ONE kernel, compiled fresh from the
production builder (`build_lm_head_gemv_qwen_module`, so a fix anywhere
between builder and device is what flips this), dispatched seven times in the
patterns above, every dispatch judged fail-closed against an f32 reference
(clean = finite, no element beyond 1e-2 of the output scale -- the honest
bf16 error is 2.5e-3 -- and bit-identical to the first dispatch for the same
input). PASS requires 7/7.

`[2026-08-21, 03:40]` With the production head re-partitioned to
9 x 16384 + 4480 (doc 57 section 5 item 5b) this gate reads 7/7 clean (devq
482, 484): the shipped artifact no longer shows the defect, the lit's XFAIL is
gone, and the gate guards against the family's return. The defect is avoided,
not understood -- doc 57 section 1.5's table has the seven configurations.

Usage: python3 lm_head_reexec_gate.py [CACHE_DIR]   (default ./lm_head_reexec_cache)
"""

import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent, _HERE.parent.parent, _HERE.parent.parent / "matrix_vector_multiplication" / "bf16"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from shared.infra.cache import KernelCache  # noqa: E402
from qwen3_0_6b_decode import (  # noqa: E402
    _LM_PARTS,
    _lm_gemv_backend,
    build_lm_head_gemv_qwen_module,
)

_OFFS = [0]
for _r in _LM_PARTS:
    _OFFS.append(_OFFS[-1] + _r)
_N_ROWS = _OFFS[-1]

EMB = 1024
TINY_M = 64


def _other_module():
    """A one-launch GEMV of the same kernel: the 'different ELF' that heals."""
    from matvec import build_module as build_gemv
    return build_gemv(TINY_M, EMB, 8, 4, 8, bfloat16, bfloat16)


def main():
    cache_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd() / "lm_head_reexec_cache"
    cache = KernelCache(cache_dir=str(cache_dir))
    t0 = time.perf_counter()
    cache.compile_and_cache("lm_head_gemv", build_lm_head_gemv_qwen_module(EMB), _lm_gemv_backend())
    cache.compile_and_cache("other_gemv", _other_module(),
                            {**_lm_gemv_backend(), "instance_name": "matvec_bf16"})
    cache._save_manifest()
    print(f"[reexec] compiled production lm_head_gemv (partitions {list(_LM_PARTS)}) "
          f"+ a 1-launch other ELF in {time.perf_counter() - t0:.0f}s", flush=True)

    rng = np.random.default_rng(0)
    W = (rng.standard_normal((_N_ROWS, EMB), dtype=np.float32) * 0.02).astype(bfloat16)
    xs = [rng.standard_normal(EMB, dtype=np.float32).astype(bfloat16) for _ in range(2)]
    refs = [x.astype(np.float32) @ W.astype(np.float32).T for x in xs]
    Wt = (rng.standard_normal((TINY_M, EMB), dtype=np.float32) * 0.02).astype(bfloat16)

    def run_head(x):
        ins = [x]
        n = len(_LM_PARTS)
        for p, rows in enumerate(_LM_PARTS):
            ins.append(np.ascontiguousarray(W[_OFFS[p]:_OFFS[p + 1]]))
            ins.append(np.zeros(rows, dtype=bfloat16))
        res = cache.load_and_run("lm_head_gemv", _lm_gemv_backend(), *ins,
                                 output_indices=[2 + 2 * p for p in range(n)],
                                 static_input_indices={1 + 2 * p for p in range(n)},
                                 intermediate_indices={2 + 2 * p for p in range(n)})
        return np.concatenate([np.asarray(res[2 + 2 * p]).astype(np.float32) for p in range(n)])

    def run_other(x):
        cache.load_and_run("other_gemv", {**_lm_gemv_backend(), "instance_name": "matvec_bf16"},
                           Wt, x, np.zeros(TINY_M, dtype=bfloat16),
                           output_indices=[2], static_input_indices={0}, intermediate_indices={2})

    def judge(tag, out, ref, base):
        scale = np.max(np.abs(ref)) + 1e-9
        rel = np.abs(out - ref) / scale
        bad = np.nonzero(rel > 1e-2)[0]
        finite = bool(np.all(np.isfinite(out)))
        same = bool(np.array_equal(out, base)) if base is not None else True
        parts = sorted({int(np.searchsorted(_OFFS, i, side="right") - 1) for i in bad})
        clean = finite and bad.size == 0 and same
        print(f"[reexec] {tag}: {'clean' if clean else 'WRONG'} -- max_rel {rel.max():.2e} "
              f"n_bad {bad.size}/{ref.size} partitions {parts} "
              f"first_rows {[int(i - _OFFS[np.searchsorted(_OFFS, i, side='right') - 1]) for i in bad[:6]]}"
              f"{'' if base is None else f' bit_identical_to_first {same}'} finite {finite}", flush=True)
        return clean

    v = []
    d1 = run_head(xs[0]); v.append(judge("d1 first dispatch", d1, refs[0], None))
    d2 = run_head(xs[0]); v.append(judge("d2 back-to-back", d2, refs[0], d1))
    d3 = run_head(xs[0]); v.append(judge("d3 back-to-back", d3, refs[0], d1))
    print(f"[reexec] d2 vs d3 max abs diff {np.max(np.abs(d2 - d3)):.3e} (non-zero = non-deterministic)")
    time.sleep(0.5)
    d4 = run_head(xs[0]); v.append(judge("d4 after 0.5 s idle", d4, refs[0], d1))
    d5 = run_head(xs[1]); v.append(judge("d5 new input back-to-back", d5, refs[1], None))
    run_other(xs[0])
    d6 = run_head(xs[0]); v.append(judge("d6 after other ELF", d6, refs[0], d1))
    d7 = run_head(xs[0]); v.append(judge("d7 back-to-back after d6", d7, refs[0], d1))
    n = sum(v)
    print(f"[reexec] {n}/{len(v)} dispatches clean")
    print("[reexec] PASS" if n == len(v) else "[reexec] FAIL")
    return 0 if n == len(v) else 1


if __name__ == "__main__":
    sys.exit(main())
