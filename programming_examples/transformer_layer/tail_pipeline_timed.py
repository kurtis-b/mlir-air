# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Time ``builders/tail_pipeline.py``'s whole-layer forward under the study's clock.

    python3 tail_pipeline_timed.py --seq 512 --emb 768 --ffn 3072 --tile-m 16 --tile-k 192 \\
        --tile-n 96 --depth 4 --n-b 2 --omit-pingpong L2 [--warmup 1] [--samples 10] [--tag T]

ONE hardware context for the whole run; the weights and gammas are uploaded once
(static, rule S2) before the clock; a forward is the ``seq / tile_m`` bands in
sequence, each: write the band's x/residual rows into the input BOs, sync them to
the device, run, sync the band's y back (the forward-only clock, 2026-08-22:
from the first band's input write to the last band's output being readable).
Correctness is checked on EVERY forward against the reference (outside the clock)
and the row is ``passed`` only if all verify. Also reported: the sum of
execute+wait per forward (``device_ms``, the dispatch seam's T1 rule) and the
per-band mean. Re-running the module on one context is itself a measurement --
the re-execution shape (queue item 10) in miniature.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from ml_dtypes import bfloat16

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE), os.path.join(os.path.dirname(_HERE), "llms")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tail_pipeline_rung as rung  # noqa: E402  (inputs, compile_all, compare)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--emb", type=int, default=768)
    ap.add_argument("--ffn", type=int, default=3072)
    ap.add_argument("--tile-m", type=int, default=16)
    ap.add_argument("--tile-k", type=int, default=192)
    ap.add_argument("--tile-n", type=int, default=96)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--n-b", type=int, default=2)
    ap.add_argument("--omit-pingpong", default="L2", choices=["", "L1", "L2", "all"])
    ap.add_argument("--atol", type=float, default=1e-1)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--tag", default="timed")
    args = ap.parse_args(argv)
    args.band_serial = True
    args.stop_after = None
    args.allow_an_lane_truncation = False
    args.dump_npz = None
    bands = args.seq // args.tile_m
    geom = (f"{args.seq}x{args.emb}x{args.ffn} m{args.tile_m} k{args.tile_k} n{args.tile_n} "
            f"d{args.depth} nb{args.n_b} bands{bands}")
    print(f"[timed] {args.tag}: {geom}")

    inputs, expected, _host = rung._inputs(args, np.random.default_rng(args.seed))
    _backend, artifact = rung.compile_all(args)

    import filelock
    import tempfile
    import pyxrt as xrt

    x, residual = inputs[0], inputs[1]
    static = list(inputs[2:])  # gamma1, w_up_p, w_down_p, gamma2
    band_shape = (args.tile_m, args.emb)
    y = np.zeros(expected.shape, expected.dtype)

    def _bytes(a):
        v = a.view(np.int16) if a.dtype == bfloat16 else a
        return np.frombuffer(np.ascontiguousarray(v).tobytes(), dtype=np.uint8)

    latencies, device_ms, verified = [], [], []
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        device = xrt.device(0)
        elf = xrt.elf(artifact.output_binary)
        context = xrt.hw_context(device, elf)
        kernel = xrt.ext.kernel(context, artifact.kernel)
        band_bytes = args.tile_m * args.emb * 2
        sizes = [band_bytes, band_bytes] + [a.size * a.itemsize for a in static] + [band_bytes]
        bos = [xrt.ext.bo(device, s) for s in sizes]
        maps = [np.frombuffer(bos[i].map(), dtype=np.uint8) for i in range(len(bos))]
        for i, a in enumerate(static):  # static uploads, once, outside the clock
            maps[2 + i][: sizes[2 + i]] = _bytes(a)
            bos[2 + i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        y_idx = len(bos) - 1
        run = xrt.run(kernel)
        for i, bo in enumerate(bos):
            run.set_arg(i, bo)

        def forward():
            """One whole-layer forward: bands in sequence. Returns (sec, device_sec)."""
            dev = 0.0
            t0 = time.perf_counter()
            for b in range(bands):
                rows = slice(b * args.tile_m, (b + 1) * args.tile_m)
                maps[0][:band_bytes] = _bytes(x[rows])
                maps[1][:band_bytes] = _bytes(residual[rows])
                bos[0].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
                bos[1].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
                d0 = time.perf_counter()
                run.start()
                run.wait2()
                dev += time.perf_counter() - d0
                bos[y_idx].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
                y[rows] = np.frombuffer(maps[y_idx][:band_bytes].tobytes(), dtype=expected.dtype).reshape(band_shape)
            return time.perf_counter() - t0, dev

        for _ in range(args.warmup):
            forward()
        for i in range(args.samples):
            sec, dev = forward()
            stats = rung.compare(y, expected, args.atol)  # outside the clock
            ok = stats["n_mismatch"] == 0 and stats["finite"]
            latencies.append(sec * 1000.0)
            device_ms.append(dev * 1000.0)
            verified.append(ok)
            print(f"[forward {i + 1}] {sec * 1000.0:8.3f} ms  device {dev * 1000.0:8.3f} ms  "
                  f"{'PASS' if ok else 'FAIL'} mismatch {stats['n_mismatch']}/{stats['n_elements']} "
                  f"corr {stats['corr']:.6f}")
        del run, bos, maps, kernel, context, elf, device

    rec = {
        "tag": args.tag, "geom": geom, "warmup": args.warmup, "samples": args.samples,
        "avg_latency_ms": float(np.mean(latencies)), "min_latency_ms": float(np.min(latencies)),
        "max_latency_ms": float(np.max(latencies)), "latency_samples_ms": latencies,
        "device_ms_avg": float(np.mean(device_ms)), "device_ms_per_band_avg": float(np.mean(device_ms)) / bands,
        "all_verified": all(verified), "bands": bands,
    }
    with open(f"{args.tag}_timed.json", "w") as f:
        json.dump(rec, f, indent=1)
    print(f"TIMED: avg {rec['avg_latency_ms']:.3f} ms  min {rec['min_latency_ms']:.3f}  max {rec['max_latency_ms']:.3f}  "
          f"device {rec['device_ms_avg']:.3f} ms ({rec['device_ms_per_band_avg']:.3f}/band x {bands})  "
          f"verified {'ALL' if rec['all_verified'] else 'NOT ALL'} ({args.tag} [{geom}])")
    return 0 if rec["all_verified"] else 1


if __name__ == "__main__":
    sys.exit(main())
