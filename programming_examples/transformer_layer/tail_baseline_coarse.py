# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""``coarse``'s AN1 / FFN / AN2 submissions timed beside its whole forward, at one shape.

    python3 tail_baseline_coarse.py --seq 512 [--mode coarse|fused] [--warmup 1] [--samples 10]

The comparison row for ``tail_pipeline_timed.py``: the same seam ``run_mode``
measures (``SPECS`` preparer -> ``dispatch``), the forward-only clock (the
``forward_done`` instant), and, per iteration, the dispatch vector rows' own
``device_submission_ms`` / ``host_sync_ms`` for the tail's submissions --
``coarse`` dispatches ``[attention, AN1, FFN, AN2]`` (``builders/block.py::
run_block``), so its tail is rows 1..3; ``fused`` is one submission and has no
per-stage split (its whole forward is reported). Turbo is required.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "study"), os.path.dirname(_HERE), os.path.join(os.path.dirname(_HERE), "llms")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_mode  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--mode", default="coarse", choices=["coarse", "fused"])
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--tag", default="baseline")
    args = ap.parse_args(argv)
    run_mode.require_turbo()
    spec = run_mode._spec_for(args.mode)
    shape, _case, _fam = run_mode._shape_for(spec, args.seq)
    prepared = spec["prepare"](shape)
    dispatch, inputs = prepared["dispatch"], prepared["inputs"]
    for _ in range(args.warmup):
        dispatch(inputs, run_mode._stage_stats)
    tail_rows = (1, 2, 3) if args.mode == "coarse" else (0,)
    lat, tail_dev, tail_sync, passed = [], [], [], []
    for i in range(args.samples):
        done = []
        t0 = time.perf_counter()
        outputs, extra = dispatch(inputs, run_mode._stage_stats, forward_done=lambda: done.append(time.perf_counter()))
        sec = done[0] - t0
        rows = extra["dispatch_vectors"]
        dev = sum(float(rows[r].get("device_submission_ms", 0.0)) for r in tail_rows)
        syn = sum(float(rows[r].get("host_sync_ms", 0.0)) for r in tail_rows)
        ok = bool(extra.get("stages_passed", True))
        lat.append(sec * 1000.0); tail_dev.append(dev); tail_sync.append(syn); passed.append(ok)
        per = " ".join(f"r{r}:{float(rows[r].get('device_submission_ms', 0.0)):.3f}" for r in range(len(rows)))
        print(f"[forward {i + 1}] {sec * 1000.0:8.3f} ms  tail device {dev:7.3f} ms  tail sync {syn:6.3f}  "
              f"submissions {per}  {'PASS' if ok else 'FAIL'}")
    rec = {"tag": args.tag, "mode": args.mode, "seq": args.seq, "avg_latency_ms": float(np.mean(lat)),
           "min_latency_ms": float(np.min(lat)), "max_latency_ms": float(np.max(lat)), "latency_samples_ms": lat,
           "tail_device_ms_avg": float(np.mean(tail_dev)), "tail_sync_ms_avg": float(np.mean(tail_sync)),
           "tail_rows": list(tail_rows), "all_passed": all(passed)}
    with open(f"{args.tag}_{args.mode}_{args.seq}.json", "w") as f:
        json.dump(rec, f, indent=1)
    print(f"BASELINE {args.mode} @{args.seq}: forward avg {rec['avg_latency_ms']:.3f} ms (min {rec['min_latency_ms']:.3f}); "
          f"tail rows {tail_rows}: device {rec['tail_device_ms_avg']:.3f} ms + sync {rec['tail_sync_ms_avg']:.3f} ms; "
          f"{'ALL PASS' if rec['all_passed'] else 'NOT ALL PASS'}")
    return 0 if rec["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
