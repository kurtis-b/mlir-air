#!/usr/bin/env python3
"""One R1 (ffn_resident) rung: build -> compile -> (optionally) dispatch -> report.

WHY THIS EXISTS (queue item 21, wall 7).  The shipped gate
(`run_npu2_ffn_resident_peano.lit`) has no working discriminator: both its
numeric arm and its negative control die inside ``run.wait2()`` with
``ERT_CMD_STATE_TIMEOUT``, so the control never reaches the comparison it exists
to prove.  A control that cannot run is not a control that passed.  This script
is the discriminator hunt: it runs ONE rung of the R1 design at a chosen shape
and a chosen backend tiling preset, and it reports one of four verdicts --

    PASS      the device completed and the output matched the FP32 reference
    FAIL      the device completed and the output did NOT match
    TIMEOUT   the command timed out (ctx health data captured verbatim)
    ERROR     anything else

so that a (PASS, FAIL) pair over the clean/fault twin at the SAME rung is a
demonstrated discriminator, and a TIMEOUT is a datum with a shape attached
rather than a dead end.

THREE THINGS IT ADDS OVER THE GATE

1.  SHAPE.  ``--emb-dim/--ffn-dim/--herd-x/--tile-k`` are the builder's own
    knobs.  The gate is pinned at 64x3072x768 = herd_x 4, sweeps 4, k_steps 24,
    chunks_per_group 6.  Smaller rungs are the same design with smaller trip
    counts, so a rung that passes proves the composition is sound and localizes
    the wall to a count rather than to the structure.

2.  A PARTIAL-PROGRESS READOUT.  ``y`` is pre-filled on the device with a
    sentinel bit pattern BEFORE dispatch, and read back even when the command
    times out (the shipped path raises out of ``wait2`` and never syncs).  The
    down herd's C store is hoisted out of the K loop, so core ``tx`` writes its
    whole ``[TILE_M, group_n]`` column strip exactly once, at the end of its K
    loop -- therefore "strip tx still holds the sentinel" == "core tx never
    finished".  That is herd_x bits of progress out of a hang.

3.  THE CTX HEALTH DATA, PARSED.  ``ctx_pc`` and friends are captured per rung
    so they can be COMPARED across rungs.  A ctx_pc that is identical across
    unrelated hangs is a firmware constant and names nothing; one that moves
    with the program is a real thread.  Neither is assumed here.

FAIL CLOSED.  ``--mode`` has no default and no fallback.  ``compile`` asserts
that pyxrt was never imported and never constructs a device; ``dispatch``
asserts the mode before touching XRT.  There is no code path that dispatches
without ``--mode dispatch``.

DEVICE DISCIPLINE.  Dispatch takes /tmp/npu.lock (the inner lock XRTRunner
takes), never /tmp/mlir-air-npu.lock (devq's, taking it here self-deadlocks).
Run it under `devq.sh run --class measure`.  Compile OFF-queue (`--mode
compile` writes air.elf + kernel.txt) and dispatch with `--reuse-elf`, so the
device lock is held for the dispatch only.

WHAT IT FOUND (devq 273 / 274 / 275 / 276, and the park note in
programming_examples/transformer_layer/run_npu2_ffn_resident_peano.lit):
R1's dataflow RUNS and is correct at herd_x 1 / 64x32x32 (5/5 identical PASS);
at herd_x 2 the SAME ELF with identical inputs gives PASS, TIMEOUT, PASS,
TIMEOUT, PASS; at herd_x 4 it is 5/5 TIMEOUT at every shape and every backend
setting tried.  Wall 7 is a race, and the shapes are valid probes because
`probe_r1_emulate_shape.py` calls the built module element-exact at all of
them.

USAGE
    # off-queue
    python3 agents/probes/probe_r1_rung.py --mode compile \\
        --emb-dim 64 --ffn-dim 64 --herd-x 2 --tile-k 32 --tag h2
    # on the queue, five times, to see the race
    agents/scripts/devq.sh run --class measure -- \\
        python3 agents/probes/probe_r1_rung.py --mode dispatch \\
        --emb-dim 64 --ffn-dim 64 --herd-x 2 --tile-k 32 \\
        --atol 5e-2 --reuse-elf ./air.elf
"""

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_WT = Path(__file__).resolve().parents[2]
_PROJ_ROOT = _WT / "programming_examples"
_TL = _PROJ_ROOT / "transformer_layer"
# Same order opcheck.py uses: transformer_layer first, so `builders` resolves
# here and not to llms/shared/builders (the documented shadowing trap).
for _p in (str(_PROJ_ROOT), str(_PROJ_ROOT / "llms"), str(_TL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MODE = None  # set once in main(); the fail-closed gate reads it

# 0x5A5A as bf16 is ~1.54e16 -- three orders outside anything this operator can
# produce, and a bit pattern no zeroed or partially-written buffer lands on.
SENTINEL_BYTE = 0x5A


def _fail_closed(expected):
    if MODE != expected:
        raise SystemExit(
            f"REFUSED: this code path requires --mode {expected}, mode is {MODE!r}"
        )


def build(args, need_module=True):
    """Build the AIR module + host-side operands for this rung. No device."""
    from ml_dtypes import bfloat16

    from builders.ffn_accum import MICRO, TILE_M
    from builders.ffn_resident import (
        build_ffn_resident_module,
        compile_ffn_resident_kernels,
        ffn_resident_device_inputs,
        ffn_resident_reference,
    )
    from opcheck_prepare import _registry_gemm_scale, _unit_output_scale, _GELU_SECOND_MOMENT

    seq_len = TILE_M
    emb_dim, ffn_dim = args.emb_dim, args.ffn_dim
    herd_x, tile_k = args.herd_x, args.tile_k
    group_n = emb_dim // herd_x

    # The kernel object bakes DIM_M/K/N in as -D flags; both GEMM herds link the
    # one object, so tile_n IS group_n for this rung (the builder's hard
    # -D-symbol constraint). Only aircc needs them, so a reuse-elf dispatch
    # skips the peano invocations entirely.
    if need_module:
        compile_ffn_resident_kernels(tile_k=tile_k, tile_n=group_n, emb_dim=emb_dim)

    rng = np.random.default_rng(args.seed)
    up_scale = _unit_output_scale(emb_dim)
    down_scale = _registry_gemm_scale(ffn_dim) / np.sqrt(ffn_dim * _GELU_SECOND_MOMENT)
    hidden = (rng.standard_normal((seq_len, emb_dim)) * up_scale).astype(bfloat16)
    w_up = (rng.standard_normal((emb_dim, ffn_dim)) * up_scale).astype(bfloat16)
    w_down = (rng.standard_normal((ffn_dim, emb_dim)) * down_scale).astype(bfloat16)

    expected = ffn_resident_reference(hidden, w_up, w_down)

    inputs = ffn_resident_device_inputs(
        hidden, w_up, w_down, herd_x=herd_x, tile_k=tile_k
    )
    if args.fault_inject:
        # opcheck's own injection: input 0 (row-major hidden) at (0, 0), after
        # the reference is computed. FAULT_DELTA is opcheck.py's 2.0.
        inputs[0] = np.array(inputs[0], copy=True)
        before = float(inputs[0][0, 0])
        inputs[0][0, 0] = np.asarray(before + 2.0, dtype=bfloat16)
        print(
            f"[fault-inject] input 0 at (0, 0): {before} -> {float(inputs[0][0, 0])}"
        )

    module = (
        build_ffn_resident_module(
            seq_len, ffn_dim, emb_dim, herd_x=herd_x, tile_k=tile_k
        )
        if need_module
        else None
    )
    geom = {
        "seq_len": seq_len,
        "emb_dim": emb_dim,
        "ffn_dim": ffn_dim,
        "herd_x": herd_x,
        "tile_k": tile_k,
        "group_n": group_n,
        "sweeps": ffn_dim // (herd_x * group_n),
        "k_steps_up": emb_dim // tile_k,
        "chunks_per_group": group_n // tile_k,
        "micro": MICRO,
    }
    return module, inputs, expected, geom


def compile_module(module, args, workdir):
    from air.backend.xrt import XRTBackend

    tiling = [] if args.tiling == "none" else [int(x) for x in args.tiling.split(",")]
    backend = XRTBackend(
        verbose=False,
        runtime_loop_tiling_sizes=tiling,
        output_format="elf",
        # opcheck.py passes instance_name = spec["operator"]. It is NOT
        # cosmetic: for ELF output the kernel XRT looks up is
        # f"main:{instance_name}", and xrt.py's own validation warns that the
        # wrong name "may cause ERT_CMD_STATE_TIMEOUT or a runtime deadlock".
        # Matching the gate exactly is the whole point of this rung.
        instance_name="ffn_resident",
        # Wall 7 is a measured RACE (devq 275: the same ELF gives PASS on 3 of
        # 5 dispatches and ERT_CMD_STATE_TIMEOUT on the other 2). These are the
        # two knobs in the backend that name that class, and R1's gate sets
        # neither.
        use_lock_race_condition_fix=(args.lockfix == "v1"),
        use_lock_race_condition_fix_v2=(args.lockfix == "v2"),
        omit_pingpong=args.omit_pingpong,
        # Explicit, so compile() never shells out to xrt-smi to autodetect.
        target_device="npu2",
    )
    t0 = time.time()
    artifact = backend.compile(module)
    # Record the kernel name beside the ELF so a later --reuse-elf dispatch
    # cannot guess it wrong (it did, once: "MLIR_AIE" -> "no module found with
    # given kernel name in ctx", 22 legs of devq 272).
    Path("kernel.txt").write_text(artifact.kernel)
    return backend, artifact, time.time() - t0, tiling


_CTX_FIELDS = (
    "txn_op_idx",
    "ctx_pc",
    "fatal_error_type",
    "fatal_error_exception_type",
    "fatal_error_exception_pc",
    "fatal_error_app_module",
)


def _parse_ctx_health(text):
    out = {}
    for f in _CTX_FIELDS:
        m = re.search(rf"{f}\s*=\s*(0x[0-9A-Fa-f]+)", text)
        if m:
            out[f] = m.group(1).lower()
    return out


def dump_npz(path, inputs, expected, y, geom, args):
    """Persist EXACTLY the bytes this dispatch used, for off-line analysis.

    Queue item 23 needs two things a printed summary cannot give: a byte-level
    equality test across repeated dispatches (``5/5 identical`` is a claim
    about bytes, not about a correlation that rounds the same), and the raw
    ``y`` beside the operands that produced it, so a host-side model of the
    dataflow can be scored against the device element by element.  The
    operands are saved rather than re-derived from ``--seed`` so the analysis
    can never drift from the run.
    """
    from ml_dtypes import bfloat16

    payload = {
        "y_raw": np.asarray(y).view(np.uint16) if y is not None else np.zeros(0, np.uint16),
        "expected": np.asarray(expected, dtype=np.float64),
        "geom": np.array(json.dumps(geom)),
        "argv": np.array(json.dumps(sys.argv[1:])),
    }
    for i, a in enumerate(inputs):
        a = np.asarray(a)
        payload[f"in{i}"] = a.view(np.uint16) if a.dtype == bfloat16 else a
    np.savez(path, **payload)


def dispatch(backend, artifact, inputs, expected, geom, args):
    """Load, fill BOs (y with a sentinel), run, and read back WHATEVER happened."""
    _fail_closed("dispatch")
    import filelock
    import tempfile
    import pyxrt as xrt
    from ml_dtypes import bfloat16

    y_shape = expected.shape
    y_dtype = expected.dtype
    args_np = list(inputs) + [np.zeros(y_shape, y_dtype)]
    sizes = [a.size * a.itemsize for a in args_np]
    y_idx = len(args_np) - 1

    res = {"verdict": "ERROR", "ctx_health": {}, "wait_s": None}

    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        device = xrt.device(0)
        elf = xrt.elf(artifact.output_binary)
        context = xrt.hw_context(device, elf)
        kernel = xrt.ext.kernel(context, artifact.kernel)

        bos = [xrt.ext.bo(device, s) for s in sizes]
        maps = [np.frombuffer(bos[i].map(), dtype=np.uint8) for i in range(len(bos))]

        for i, a in enumerate(args_np):
            if i == y_idx:
                maps[i][: sizes[i]] = SENTINEL_BYTE  # the progress sentinel
            else:
                v = a.view(np.int16) if a.dtype == bfloat16 else a
                maps[i][: sizes[i]] = np.frombuffer(
                    np.ascontiguousarray(v).tobytes(), dtype=np.uint8
                )
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
            res["exception"] = str(e).strip()
            res["ctx_health"] = _parse_ctx_health(str(e))
        res["wait_s"] = round(time.time() - t0, 3)

        # THE POINT: read back regardless. The shipped path raises out of
        # wait2() and never syncs, so nothing has ever looked at what the
        # device actually produced under this hang.
        y_bytes = None
        try:
            bos[y_idx].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            y_bytes = np.array(maps[y_idx][: sizes[y_idx]], copy=True)
        except Exception as e:  # noqa: BLE001
            res["sync_back_error"] = str(e).strip()

        del run, bos, maps, kernel, context, elf, device

    if y_bytes is None:
        return res, None

    # --- progress readout -------------------------------------------------
    untouched = y_bytes == SENTINEL_BYTE
    res["y_sentinel_fraction"] = round(float(untouched.mean()), 6)
    y = np.frombuffer(y_bytes.tobytes(), dtype=y_dtype).reshape(y_shape)
    # Down core tx owns columns [tx*group_n, (tx+1)*group_n); its C store is
    # hoisted out of the K loop, so the strip is written once, at the end.
    per_elem_untouched = untouched.reshape(y_shape[0], y_shape[1], -1).all(axis=2)
    strips = []
    for tx in range(geom["herd_x"]):
        lo, hi = tx * geom["group_n"], (tx + 1) * geom["group_n"]
        frac = float(per_elem_untouched[:, lo:hi].mean())
        strips.append({"core": tx, "cols": [lo, hi], "sentinel_fraction": round(frac, 6)})
    res["y_strips"] = strips
    res["cores_finished"] = sum(1 for s in strips if s["sentinel_fraction"] == 0.0)
    return res, y


def compare(y, expected, args):
    """opcheck's comparison: full-output np.isclose at the row's rtol/atol."""
    a = np.asarray(y, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    rtol, atol = 1.6e-2, args.atol
    close = np.isclose(a, e, rtol=rtol, atol=atol)
    mism = int((~close).sum())
    abs_err = np.abs(a - e)
    denom = np.abs(e).mean()
    af, ef = a.ravel(), e.ravel()
    corr = (
        float(np.corrcoef(af, ef)[0, 1])
        if af.std() > 0 and ef.std() > 0
        else float("nan")
    )
    stats = {
        "mismatches": mism,
        "total": int(a.size),
        "correlation": corr,
        "abs_err_max": float(abs_err.max()),
        "mean_rel_L1": float(abs_err.mean() / denom) if denom else float("nan"),
        # atol that WOULD have been required, opcheck's own definition
        "atol_required": float(np.max(abs_err - rtol * np.abs(e)).clip(min=0.0)),
        "rtol": rtol,
        "atol": atol,
    }
    return (mism == 0), stats


def main():
    global MODE
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", required=True, choices=("compile", "dispatch"))
    p.add_argument("--emb-dim", type=int, default=768)
    p.add_argument("--ffn-dim", type=int, default=3072)
    p.add_argument("--herd-x", type=int, default=4)
    p.add_argument("--tile-k", type=int, default=32)
    p.add_argument(
        "--tiling",
        default="2,2",
        help="runtime_loop_tiling_sizes; 'none' for an empty list",
    )
    p.add_argument("--lockfix", default="none", choices=("none", "v1", "v2"))
    p.add_argument("--omit-pingpong", default="", choices=("", "L1", "L2", "all"))
    p.add_argument("--fault-inject", action="store_true")
    p.add_argument("--atol", type=float, default=5e-3)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--tag", default="rung")
    p.add_argument(
        "--reuse-elf",
        default=None,
        help="dispatch a PRE-COMPILED air.elf (compiled off-queue) instead of "
        "recompiling; refuses if the path does not exist",
    )
    p.add_argument("--json-out", default=None)
    p.add_argument(
        "--dump-npz",
        default=None,
        help="write the raw device y AND the operands that produced it to this "
        ".npz (queue item 23: determinism is a claim about bytes, and the "
        "element-wise comparison needs the operands beside the output)",
    )
    args = p.parse_args()
    MODE = args.mode

    label = (
        f"{args.tag} [{args.emb_dim}x{args.ffn_dim} herd_x={args.herd_x} "
        f"tile_k={args.tile_k} tiling={args.tiling}"
        f"{' FAULT' if args.fault_inject else ''}]"
    )
    rec = {"tag": args.tag, "label": label, "mode": args.mode, "argv": sys.argv[1:]}
    print(f"=== {label} ===", flush=True)

    reuse = args.reuse_elf and Path(args.reuse_elf).is_file()
    if args.reuse_elf and not reuse:
        raise SystemExit(f"REFUSED: --reuse-elf {args.reuse_elf} does not exist")
    try:
        module, inputs, expected, geom = build(args, need_module=not reuse)
        rec["geom"] = geom
        print(
            "[geom] sweeps={sweeps} k_steps_up={k_steps_up} "
            "chunks_per_group={chunks_per_group} group_n={group_n}".format(**geom),
            flush=True,
        )
        if reuse:
            from air.backend.xrt import XRTCompileArtifact

            kname_file = Path(args.reuse_elf).with_name("kernel.txt")
            if not kname_file.is_file():
                raise SystemExit(
                    f"REFUSED: {kname_file} missing -- the compile step must "
                    "record the kernel name; guessing it is how devq 272 lost "
                    "22 legs"
                )
            kname = kname_file.read_text().strip()
            backend, artifact = None, XRTCompileArtifact(args.reuse_elf, kname, None)
            rec["kernel"] = kname
            rec["compile_s"] = 0.0
            rec["reused_elf"] = args.reuse_elf
            rec["tiling"] = args.tiling
            rec["binary"] = artifact.output_binary
            print(f"[compile] REUSED {args.reuse_elf}", flush=True)
        else:
            backend, artifact, secs, tiling = compile_module(module, args, os.getcwd())
            rec["compile_s"] = round(secs, 2)
            rec["tiling"] = tiling
            rec["binary"] = artifact.output_binary
            print(f"[compile] ok in {secs:.1f}s -> {artifact.output_binary}", flush=True)
    except Exception:  # noqa: BLE001
        rec["verdict"] = "ERROR"
        rec["error"] = traceback.format_exc()
        print(rec["error"], flush=True)
        _emit(rec, args)
        return 3

    if args.mode == "compile":
        assert "pyxrt" not in sys.modules, "compile mode imported pyxrt"
        rec["verdict"] = "COMPILED"
        print(f"[compile-only] no device touched (pyxrt imported: False)", flush=True)
        _emit(rec, args)
        return 0

    try:
        res, y = dispatch(backend, artifact, inputs, expected, geom, args)
    except Exception:  # noqa: BLE001
        rec["verdict"] = "ERROR"
        rec["error"] = traceback.format_exc()
        print(rec["error"], flush=True)
        _emit(rec, args)
        return 3

    rec.update(res)
    if res.get("ctx_health"):
        print("[ctx-health] " + " ".join(f"{k}={v}" for k, v in res["ctx_health"].items()), flush=True)
    if "y_strips" in res:
        parts = []
        for s in res["y_strips"]:
            f = s["sentinel_fraction"]
            parts.append(f"c{s['core']}=" + ("DONE" if f == 0.0 else f"{f:.3f}untouched"))
        print(
            "[progress] y sentinel fraction {:.4f}; cores finished {}/{}: {}".format(
                res["y_sentinel_fraction"],
                res["cores_finished"],
                geom["herd_x"],
                " ".join(parts),
            ),
            flush=True,
        )
    print(f"[wait] {res['wait_s']}s, dispatch verdict {res['verdict']}", flush=True)

    if args.dump_npz:
        dump_npz(args.dump_npz, inputs, expected, y, geom, args)
        print(f"[dump] {args.dump_npz}", flush=True)

    if y is not None:
        ok, stats = compare(y, expected, args)
        rec["stats"] = stats
        print(
            "[compare] mismatches {}/{} corr {:.9f} abs_err_max {:.4e} "
            "mean_rel_L1 {:.4e} atol_required {:.4e} (atol {:.1e})".format(
                stats["mismatches"],
                stats["total"],
                stats["correlation"],
                stats["abs_err_max"],
                stats["mean_rel_L1"],
                stats["atol_required"],
                stats["atol"],
            ),
            flush=True,
        )
        if res["verdict"] == "RAN":
            rec["verdict"] = "PASS" if ok else "FAIL"
        else:
            # The device did not complete; the comparison is reported for the
            # progress it shows, NOT as a verdict.
            rec["verdict"] = "TIMEOUT"

    print(f"RUNG VERDICT: {rec['verdict']}  ({label})", flush=True)
    _emit(rec, args)
    return 0


def _emit(rec, args):
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rec, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
