# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Walk the sequence ladder for one or more execution modes (J3).

    python3 study/run_ladder.py --modes coarse,offload,runlist,fused \
        --seqs 512,1024,2048,4096 --out-dir <results-root> --study-id j3-ladder

CONTRACT
    One CSV per mode, named ``<mode>.csv``, holding one schema-v1 row per rung.
    Delegates every rung to ``run_mode.run`` -- the timing, the dispatch-vector
    totals and the pass/fail verdict all come from the one implementation the
    single-shape path already uses, so a ladder row and a single-shape row are
    the same measurement at different lengths.

WHY A LADDER AT ALL
    Doc 16's J3: "A tradeoff analysis at a single shape has no curves and
    therefore no crossover -- which is the result the study exists to produce."
    Four modes at one length rank four numbers; the same four across lengths is
    where a mode that wins at 512 and loses at 4096 becomes visible.

FOOTGUNS
    - **Hold the NPU lock around the whole invocation**, not per rung:

          flock -x -w 1800 /tmp/mlir-air-npu.lock python3 study/run_ladder.py ...

      and never alongside a ``port-loop`` gate. Gate leg 4 is a decode-throughput
      measurement against a floor, so interleaved dispatches corrupt it even
      though they stay correct.
    - **Nothing CPU-heavy may run beside this.** Compilation is outside the
      clock, but the timed region includes host-side dispatch, so a concurrent
      build inflates latency for whichever rung is unlucky. That is a silent
      distortion, not a failure.
    - **The working directory decides where caches land**, because aircc and
      KernelCache write relative to cwd. Run from
      ``programming_examples/transformer_layer`` so the ``*_cache/`` and ``*.o``
      debris falls under the .gitignore rules written for it. Cache entries are
      keyed by an ELF name that embeds the shape (``blk_ffn_512x768x3072``), so
      rungs of different lengths share a cache directory safely.
    - **A rung that cannot build is a result, not a crash.** It writes a row with
      ``run_status=failed`` and the compiler's own message, and the ladder keeps
      going: a length failing does not predict longer ones failing, and the
      refusal message is usually the interesting part. The exit code is nonzero
      if any rung failed, so a gate still sees it.
    - Rows are rewritten after **every** rung, so a killed run leaves the rungs
      it finished rather than nothing.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import results_io  # noqa: E402
import run_mode  # noqa: E402


def walk(
    modes: list[str],
    seqs: list[int],
    out_dir: str,
    study_id: str,
    warmup: int,
    samples: int,
    runs_per_sample: int,
) -> list[dict]:
    """Run every (mode, seq) rung, writing each mode's CSV as it fills."""
    every = []
    for mode in modes:
        rows = []
        out = os.path.join(out_dir, f"{mode}.csv")
        for seq in seqs:
            t0 = time.perf_counter()
            row = run_mode.run(mode, warmup, samples, runs_per_sample, seq)
            row["study_id"] = study_id
            rows.append(row)
            every.append(row)
            # Rewrite the whole mode after each rung: cheap, and a killed run
            # keeps what it measured.
            results_io.write_rows(out, rows)
            wall = time.perf_counter() - t0
            if row["run_status"] == "passed":
                print(
                    f"[ladder] {mode:9s} seq {seq:5d}  avg {row['avg_latency_ms']:9.3f} ms"
                    f"  subs {row['host_submissions_per_layer']:3d}"
                    f"  herd {row['herd_launches']:4d}"
                    f"  sync {row['sync_boundaries']:4d}"
                    f"  ({wall:.0f}s wall)",
                    flush=True,
                )
            else:
                print(
                    f"[ladder] {mode:9s} seq {seq:5d}  FAILED  "
                    f"{row['failure_message']}  ({wall:.0f}s wall)",
                    flush=True,
                )
        print(f"[ladder] wrote {out}", flush=True)
    return every


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--modes", default="coarse,offload,runlist,fused")
    ap.add_argument("--seqs", default="512,1024,2048,4096")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--study-id", default="j3-ladder")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--runs-per-sample", type=int, default=1)
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    modes = [m for m in args.modes.split(",") if m]
    seqs = [int(s) for s in args.seqs.split(",") if s]

    rows = walk(
        modes,
        seqs,
        args.out_dir,
        args.study_id,
        args.warmup,
        args.samples,
        args.runs_per_sample,
    )

    passed = [r for r in rows if r["run_status"] == "passed"]
    print(f"[ladder] {len(passed)}/{len(rows)} rungs passed")
    for r in rows:
        if r["run_status"] != "passed":
            print(f"[ladder]   FAILED {r['study_case_label']}: {r['failure_message']}")
    return 0 if len(passed) == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
