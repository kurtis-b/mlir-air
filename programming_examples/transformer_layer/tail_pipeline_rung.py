# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""One hardware rung of ``builders/tail_pipeline.py``: compile, dispatch, compare.

    python3 tail_pipeline_rung.py --seq 64 --emb 96 --ffn 192 --tile-m 16 --tile-k 48 \\
        --tile-n 96 --depth 2 --n-b 2 [--stop-after an1|up|down] [--atol 5e-2] \\
        [--dispatches 3] [--band-serial] [--omit-pingpong L2] [--tag T]

BAND-SERIAL. With several sweeps (ffn_dim > n_b * tile_n) the module is ONE
band (the builder's rule; its FOOTGUNS say why) and ``--band-serial`` makes
the rung iterate the bands on the host: the module is built at
``seq_len = tile_m``, each band's rows of x and residual go in their own BOs,
and the per-band y are assembled before the compare -- one hardware context
per band per "dispatch", the band loop on launch arguments that
builders/ffn_resident.py follows. ``--omit-pingpong L2`` hands the backend's
option through (single-slot L2 staging: at the layer's width the memtile's
48 BDs do not hold the double-buffered programs).

``--stop-after`` is the bisection instrument (``build_tail_pipeline_module``'s
``stop_after``): the module stops at that herd and routes its output to ``y``;
the comparison is against ``tail_pipeline_stage_reference`` for the same cut
(h1 rows / H tiles / reduced C blocks). Omit it for the whole pipeline.

TOLERANCE. The oracle is FP32 end to end (the house rule); the device rounds
to bf16 at every hop. Measured on the chain and the emb-32 baseline geometry
(devq 516, 2026-08-22): each cut sits inside ``5e-2 + 1.6e-2*|ref|`` (an1
needs 6e-3, up 4.3e-2, down 3.4e-2 of absolute slack), and the WHOLE
pipeline's ``y`` equals AN2 applied to the device's own C and h1 within
1.05e-2 + 1.6e-2*|ref| -- so the full output's extra error is the FFN's bf16
error (up to the down cut's 5e-2) passed through LayerNorm's 1/std(c) (~1.4
at these inputs) plus the AN envelope: 6.5e-2 needed, 1e-1 is the default
for the whole pipeline, 5e-2 at a cut. ``--atol`` overrides both.

RUN FROM A SCRATCH DIRECTORY inside a devq measure job: the kernel objects,
``air.mlir``, ``air.elf`` and ``air_project/`` land in the cwd, and the device
lock is taken per dispatch as ``probe_r1_rung.py`` does. Verdict per dispatch:
PASS (every element within ``atol + 1.6e-2 * |ref|``), FAIL (ran, mismatches),
TIMEOUT (``ERT_CMD_STATE_TIMEOUT``; the output is still read back and its
sentinel fraction reported -- a hang is a datum with a shape). A JSON record
per dispatch is written beside the artifacts. Item 9 of the 2026-08-21 queue.
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

from builders.tail_pipeline import (  # noqa: E402
    STOP_STAGES,
    build_tail_pipeline_module,
    compile_tail_pipeline_kernels,
    tail_pipeline_device_inputs,
    tail_pipeline_stage_reference,
)

SENTINEL_BYTE = 0x7F  # bf16 0x7F7F = a large finite value no LayerNorm output reaches


def _inputs(args, rng):
    s, e, f = args.seq, args.emb, args.ffn
    x = rng.standard_normal((s, e)).astype(np.float32)
    residual = (0.5 * rng.standard_normal((s, e))).astype(np.float32)
    gamma1 = (1.0 + 0.1 * rng.standard_normal(e)).astype(np.float32)
    gamma2 = (1.0 + 0.1 * rng.standard_normal(e)).astype(np.float32)
    w_up = (rng.standard_normal((e, f)) / np.sqrt(e)).astype(np.float32)
    w_down = (rng.standard_normal((f, e)) / np.sqrt(f)).astype(np.float32)
    b = lambda a: a.astype(bfloat16)  # noqa: E731
    host = dict(x=b(x), residual=b(residual), gamma1=b(gamma1), gamma2=b(gamma2),
                w_up=b(w_up), w_down=b(w_down))
    expected = tail_pipeline_stage_reference(
        host["x"], host["residual"], host["gamma1"], host["w_up"], host["w_down"],
        host["gamma2"], args.tile_m, args.tile_k, args.tile_n, args.n_b, args.depth,
        stop_after=args.stop_after)
    dev = tail_pipeline_device_inputs(host["x"], host["residual"], host["gamma1"], host["w_up"],
                                      host["w_down"], host["gamma2"], args.tile_k, args.tile_n,
                                      args.n_b, args.depth)
    return dev, expected, host


def compile_all(args):
    from air.backend.xrt import XRTBackend

    compile_tail_pipeline_kernels(tile_m=args.tile_m, tile_k=args.tile_k, tile_n=args.tile_n)
    module = build_tail_pipeline_module(
        seq_len=args.tile_m if args.band_serial else args.seq, emb_dim=args.emb,
        ffn_dim=args.ffn, tile_m=args.tile_m, tile_k=args.tile_k, tile_n=args.tile_n,
        down_proj_depth=args.depth, n_b=args.n_b,
        allow_an_lane_truncation=args.allow_an_lane_truncation, stop_after=args.stop_after,
    )
    backend = XRTBackend(
        verbose=False, output_format="elf", instance_name="tail_pipeline",
        runtime_loop_tiling_sizes=[2, 2], target_device="npu2",
        omit_pingpong=args.omit_pingpong,
    )
    t0 = time.time()
    artifact = backend.compile(module)
    print(f"[compile] ok in {time.time() - t0:.1f}s -> {artifact.output_binary} kernel {artifact.kernel}")
    return backend, artifact


def dispatch(artifact, inputs, expected):
    """One hardware context, one run; returns (record, y or None)."""
    import filelock
    import tempfile
    import pyxrt as xrt

    args_np = list(inputs) + [np.zeros(expected.shape, expected.dtype)]
    sizes = [a.size * a.itemsize for a in args_np]
    y_idx = len(args_np) - 1
    res = {"verdict": "ERROR"}
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        device = xrt.device(0)
        elf = xrt.elf(artifact.output_binary)
        context = xrt.hw_context(device, elf)
        kernel = xrt.ext.kernel(context, artifact.kernel)
        bos = [xrt.ext.bo(device, s) for s in sizes]
        maps = [np.frombuffer(bos[i].map(), dtype=np.uint8) for i in range(len(bos))]
        for i, a in enumerate(args_np):
            if i == y_idx:
                maps[i][: sizes[i]] = SENTINEL_BYTE
            else:
                v = a.view(np.int16) if a.dtype == bfloat16 else a
                maps[i][: sizes[i]] = np.frombuffer(np.ascontiguousarray(v).tobytes(), dtype=np.uint8)
            bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        run = xrt.run(kernel)
        for i, bo in enumerate(bos):
            run.set_arg(i, bo)
        t0 = time.time()
        try:
            run.start()
            run.wait2()
            res["verdict"] = "RAN"
        except Exception as e:  # noqa: BLE001 -- the timeout arrives as RuntimeError
            res["verdict"] = "TIMEOUT"
            res["exception"] = str(e).strip()[:300]
        res["wait_s"] = round(time.time() - t0, 4)
        y_bytes = None
        try:
            bos[y_idx].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            y_bytes = np.array(maps[y_idx][: sizes[y_idx]], copy=True)
        except Exception as e:  # noqa: BLE001
            res["sync_back_error"] = str(e).strip()[:200]
        del run, bos, maps, kernel, context, elf, device
    if y_bytes is None:
        return res, None
    res["y_sentinel_fraction"] = round(float((y_bytes == SENTINEL_BYTE).mean()), 6)
    y = np.frombuffer(y_bytes.tobytes(), dtype=expected.dtype).reshape(expected.shape)
    return res, y


def dispatch_bands(artifact, inputs, expected, args):
    """Band-serial: one context per band, rows sliced on the host, y assembled.

    Every cut's y layout has the band outermost (rows for an1/full, the
    (band, sweep, column) run for up, the (band, d) run for down), so the
    per-band outputs concatenate. The record is the worst band's verdict;
    wait_s sums the bands.
    """
    if not args.band_serial:
        return dispatch(artifact, inputs, expected)
    bands = args.seq // args.tile_m
    per_band = expected.size // bands
    y = np.zeros(expected.shape, expected.dtype).reshape(bands, -1)
    rec = {"verdict": "RAN", "wait_s": 0.0, "y_sentinel_fraction": 0.0, "bands": bands}
    for b in range(bands):
        rows = slice(b * args.tile_m, (b + 1) * args.tile_m)
        band_inputs = [np.ascontiguousarray(inputs[0][rows]), np.ascontiguousarray(inputs[1][rows])] + list(inputs[2:])
        band_expected = expected.reshape(bands, -1)[b].reshape(
            (args.tile_m, args.emb) if expected.ndim == 2 else (per_band,))
        r, yb = dispatch(artifact, band_inputs, band_expected)
        rec["wait_s"] += r.get("wait_s", 0.0)
        if r["verdict"] != "RAN":
            # A hung band says everything the later bands would; stop here.
            rec["verdict"] = r["verdict"]
            rec["exception"] = r.get("exception", "")
            rec["failed_band"] = b
            rec["y_sentinel_fraction"] = r.get("y_sentinel_fraction", 1.0)
            return rec, None
        if yb is None:
            rec["y_sentinel_fraction"] = 1.0
            return rec, None
        rec["y_sentinel_fraction"] = max(rec["y_sentinel_fraction"], r.get("y_sentinel_fraction", 0.0))
        y[b] = yb.reshape(-1)
    rec["wait_s"] = round(rec["wait_s"], 4)
    return rec, y.reshape(expected.shape)


def compare(y, expected, atol):
    a = y.astype(np.float32)
    e = expected.astype(np.float32)
    err = np.abs(a - e)
    den = np.abs(e)
    bad = err > atol + 1.6e-2 * den
    nz = den > 0
    finite = bool(np.isfinite(a).all())
    return {
        "n_elements": int(a.size), "n_mismatch": int(bad.sum()), "finite": finite,
        "abs_err_max": float(err.max()),
        "mean_rel_L1": float(err[nz].sum() / den[nz].sum()) if nz.any() else 0.0,
        "atol_required": float(np.max(np.maximum(err - 1.6e-2 * den, 0.0))),
        "corr": float(np.corrcoef(a.ravel(), e.ravel())[0, 1]) if a.std() > 0 else 0.0,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--emb", type=int, default=96)
    ap.add_argument("--ffn", type=int, default=192)
    ap.add_argument("--tile-m", type=int, default=16)
    ap.add_argument("--tile-k", type=int, default=48)
    ap.add_argument("--tile-n", type=int, default=96)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--n-b", type=int, default=2)
    ap.add_argument("--allow-an-lane-truncation", action="store_true")
    ap.add_argument("--stop-after", choices=[s for s in STOP_STAGES if s], default=None,
                    help="bisection cut: stop the module at this herd and compare its output")
    ap.add_argument("--atol", type=float, default=None,
                    help="absolute tolerance on top of 1.6e-2*|ref| (default: 5e-2 at a cut, "
                         "1e-1 for the whole pipeline; see the docstring)")
    ap.add_argument("--dispatches", type=int, default=3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--band-serial", action="store_true",
                    help="build the module at one band (seq_len = tile_m) and iterate bands on the host")
    ap.add_argument("--omit-pingpong", default="", choices=["", "L1", "L2", "all"],
                    help="XRTBackend(omit_pingpong=...)")
    ap.add_argument("--tag", default="rung")
    ap.add_argument("--dump-npz", action="store_true",
                    help="also save y, expected and the host inputs per dispatch (<tag>_dispatch<i>.npz)")
    args = ap.parse_args(argv)
    if args.atol is None:
        args.atol = 5e-2 if args.stop_after else 1e-1
    geom = f"{args.seq}x{args.emb}x{args.ffn} m{args.tile_m} k{args.tile_k} n{args.tile_n} d{args.depth} nb{args.n_b}"
    if args.stop_after:
        geom += f" stop_after={args.stop_after}"
    if args.band_serial:
        geom += f" band-serial x{args.seq // args.tile_m}"
    if args.omit_pingpong:
        geom += f" omit-pingpong={args.omit_pingpong}"
    print(f"[rung] {args.tag}: {geom}")
    inputs, expected, host = _inputs(args, np.random.default_rng(args.seed))
    backend, artifact = compile_all(args)
    verdicts = []
    for i in range(args.dispatches):
        res, y = dispatch_bands(artifact, inputs, expected, args)
        rec = {"tag": args.tag, "geom": geom, "dispatch": i + 1, **res}
        if y is not None and res["verdict"] == "RAN":
            rec.update(compare(y, expected, args.atol))
            rec["verdict"] = "PASS" if (rec["n_mismatch"] == 0 and rec["finite"]) else "FAIL"
        print(f"[dispatch {i + 1}] {rec['verdict']} wait {rec.get('wait_s')}s "
              + (f"mismatch {rec['n_mismatch']}/{rec['n_elements']} corr {rec['corr']:.6f} "
                 f"abs_err_max {rec['abs_err_max']:.3e} mean_rel_L1 {rec['mean_rel_L1']:.3e} "
                 f"atol_required {rec['atol_required']:.3e}" if "n_mismatch" in rec else
                 f"sentinel {rec.get('y_sentinel_fraction')} {rec.get('exception', '')[:120]}"))
        with open(f"{args.tag}_dispatch{i + 1}.json", "w") as f:
            json.dump(rec, f, indent=1)
        if args.dump_npz and y is not None:
            np.savez(f"{args.tag}_dispatch{i + 1}.npz", y=y.view(np.int16), expected=expected.view(np.int16),
                     **{k: v.view(np.int16) for k, v in host.items()})
        verdicts.append(rec["verdict"])
    try:
        backend.unload()
    except Exception:  # noqa: BLE001
        pass
    print(f"RUNG VERDICT: {'PASS' if all(v == 'PASS' for v in verdicts) else 'FAIL'} ({args.tag} [{geom}] {verdicts})")
    return 0 if all(v == "PASS" for v in verdicts) else 1


if __name__ == "__main__":
    sys.exit(main())
