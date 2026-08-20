# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The fused decoder's RE-EXECUTION gate: two dispatches of one prepared stitch.

`[2026-08-19]` This gate exists because the failure it pins was silent and
survived every single-dispatch check: the causal mha kept its Q-block base in
an UNINITIALIZED L1 counter behind a boot flag that only fires on zeroed
memory, and the base never wrapped -- so dispatch 1 of the fused decoder was
12/12 boundaries clean while every later dispatch computed UNMASKED attention
(corr 0.9994 with the non-causal reference, devq 382-384) with q/k/v still
clean. The fix wraps the counter's advance modulo the execution's total Q
blocks (attn_npu2*.py), restoring the boot state at the end of every complete
execution. Removing that wrap turns dispatch 2 unmasked again, which this
gate reports as FAIL: it is the checked-in falsifier the fix's review asked
for.

Two dispatches are the load-bearing count -- one dispatch cannot see the
defect, and dispatch 3 was measured byte-identical to dispatch 2 when the
wall stood and when it fell, so a third buys nothing here (the deeper
sweep lives in the study's decoder profile walks).

A boundary is "clean" here by the SAME rule the fused driver prints its own
``12/12`` under -- ``n_mismatch == 0`` from ``run_mode._stage_stats``, the
abs+rel band ``atol + 1.6e-2 * |expected|`` -- taken from the very callback
values the dispatch computes, against the driver's own reference and atol.
Two fail-closed clauses sit on top, because ``n_mismatch`` is a ``>``
comparison and NaN compares False: a dispatch that runs fewer boundary
comparisons than ``DECODER_BOUNDARIES`` names is a FAIL, and a boundary with
any non-finite element is a FAIL. The correlation clause keeps its own
independently generated causal reference -- it is the defect's discriminator
(0.44 causal when the wall stood, ~0.9995 healthy) and should not inherit a
hypothetically wrong driver reference.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_TL = os.path.dirname(_HERE)
for _p in (_TL, os.path.join(os.path.dirname(os.path.dirname(_HERE)), "llms"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main() -> int:
    from ml_dtypes import bfloat16  # noqa: F401  (env preflight)

    import run_mode
    from pattern.fused.fused import prepare_fused
    from pattern.reference import DECODER_BOUNDARIES, generate_golden_reference

    shape = {
        "seq_len": 512,
        "emb_dim": 768,
        "ffn_dim": 3072,
        "num_heads": 12,
        "head_dim": 64,
        "workload_variant": "decoder_gpt2",
    }
    prepared = prepare_fused(shape)
    dispatch = prepared["dispatch"]

    golden = generate_golden_reference(
        512, 768, 3072, 12, seed=42, workload_variant="decoder_gpt2"
    )
    causal_ref = np.asarray(
        golden["boundaries"]["attn_context"], np.float32
    ).ravel()

    n_bounds = len(DECODER_BOUNDARIES)
    captured = {}
    recorded = []

    def capturing_stats(actual, expected, atol):
        stats = run_mode._stage_stats(actual, expected, atol)
        recorded.append((stats, np.array(np.asarray(actual, np.float32), copy=True)))
        return stats

    ok = True
    for run in (1, 2):
        recorded.clear()
        dispatch(prepared["inputs"], capturing_stats)
        captured[run] = list(recorded)
        bad = []
        if len(captured[run]) != n_bounds:
            bad.append(
                f"only {len(captured[run])}/{n_bounds} boundary comparisons ran"
            )
        for name, (stats, arr) in zip(DECODER_BOUNDARIES, captured[run]):
            if not np.isfinite(arr).all():
                bad.append(f"{name} has non-finite elements")
            elif stats["n_mismatch"]:
                bad.append(f"{name} n_mismatch {stats['n_mismatch']}")
        verdict = (
            f"{n_bounds}/{n_bounds} boundaries clean"
            if not bad
            else "FAIL: " + "; ".join(bad[:3])
        )
        ok = ok and not bad
        print(f"[reexec] dispatch {run}: {verdict}", flush=True)

    ctx_idx = DECODER_BOUNDARIES.index("attn_context")
    if len(captured[2]) <= ctx_idx:
        print("[reexec] dispatch 2 attn_context never compared; corr not computable")
        ok = False
    else:
        d2 = captured[2][ctx_idx][1].ravel()
        corr = float(np.corrcoef(d2, causal_ref)[0, 1])
        # The defect's signature was corr 0.44 here (and 0.9994 against the
        # UNMASKED reference); a healthy re-execution reads ~0.9995 causal.
        print(f"[reexec] dispatch 2 attn_context corr vs causal reference: {corr:.4f}")
        if not (corr >= 0.99):  # NaN-safe: a NaN corr fails
            ok = False
    print(f"[reexec] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
