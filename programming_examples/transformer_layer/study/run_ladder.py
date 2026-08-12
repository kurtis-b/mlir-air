# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Walk the sequence ladder for one or more execution modes (J3).

    python3 study/run_ladder.py --modes coarse,offload,runlist,fused \
        --seqs 512,1024,2048,4096 --out-dir <results-root> --study-id j3-ladder

CONTRACT
    One CSV per mode, named ``<mode>.csv``, holding one current-schema row per rung.
    Delegates every rung to ``run_mode.py`` -- the timing, the dispatch-vector
    totals and the pass/fail verdict all come from the one implementation the
    single-shape path already uses, so a ladder row and a single-shape row are
    the same measurement at different lengths.

ONE PROCESS PER RUNG, AND THIS IS NOT OPTIONAL
    `[2026-08-08]` The first version of this module called ``run_mode.run`` in
    process, looping rungs. ``coarse`` and ``offload`` walked all four lengths;
    ``runlist`` passed 512 and 1024 and then failed 2048 **and** 4096, each on a
    *different* ELF, with "Failed to load ELF kernel for XRT ... contains a
    kernel symbol matching the provided name". The symbol was present in the
    file. The same mode at the same 4096 had passed alone in a fresh process
    minutes earlier.

    What separates them is that ``runlist`` loads roughly ten kernels per rung
    against ``coarse``'s four, so by the third rung one process had accumulated
    the most XRT kernel and context handles -- and the load that happened to be
    next is the one that failed. It is process-level resource exhaustion, not a
    shape limit and not a compiler defect, and in-process looping reports it as
    "this mode cannot run at this length", which is false and is exactly the
    kind of wrong conclusion a study publishes.

    So each rung runs as a child process that exits before the next begins. This
    is what iron's ``end_to_end/modes.py`` subprocess isolation is for; it reads
    like defensive scaffolding until a ladder walks far enough to need it.
    Isolation also makes a ladder row and a single-shape row identical in
    provenance, since both now come from a fresh ``run_mode.py`` invocation.

WHY A LADDER AT ALL
    Doc 16's J3: "A tradeoff analysis at a single shape has no curves and
    therefore no crossover -- which is the result the study exists to produce."
    Four modes at one length rank four numbers; the same four across lengths is
    where a mode that wins at 512 and loses at 4096 becomes visible.

FOOTGUNS
    - **Reserve the device around the whole invocation**, not per rung:

          agents/scripts/devq.sh run --class measure -- \
              python3 study/run_ladder.py ...

      ``run``, never ``submit``: ``submit`` diverts output to the job log and
      returns an id, so a gate that substitutes it blanks its own FileCheck and
      still exits 0. Never alongside a ``port-loop`` gate either -- gate leg 4 is
      a decode-throughput measurement against a floor, so interleaved dispatches
      corrupt it even though they stay correct.
    - **Nothing CPU-heavy may run beside this.** Compilation is outside the
      clock, but the timed region includes host-side dispatch, so a concurrent
      build inflates latency for whichever rung is unlucky. That is a silent
      distortion, not a failure. The ``measure`` class is an absolute barrier in
      the queue -- later builds are not admitted until it has run -- so this is
      enforced rather than left to discipline. **Compile before the walk**, as
      separate ``--class build`` jobs, so no aircc runs inside the reservation.
      Those builds must be SEQUENTIAL: aircc stages intermediates in
      ``air_project/`` relative to cwd, so concurrent invocations from one
      directory delete each other's object files, and one of them can "succeed"
      out of the wreckage.
    - **Walk it TWICE and compare the walks.** `[2026-08-09]` A single walk of
      512/1024 produced a clean `offload`/`runlist` crossover, reported by
      ``ladder_report``; a second walk under identical conditions reported no
      crossover at all. `offload` drifts up to 120% within one walk, which is
      enough to invert a ranking on its own. One walk cannot tell a ranking from
      noise -- see 27-common-ladder-result.md.
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

A RUNG THAT CANNOT APPLY IS `skipped`, AND THAT IS A DIFFERENT THING
    ``walk(..., skip_reason=fn)`` takes a callable ``fn(mode, seq) -> str|None``.
    Where it returns text the rung is NOT run: a row is written with
    ``run_status="skipped"`` carrying the reason, and no child process starts.

    ``schema.RUN_STATUSES`` has held ``"skipped"`` since v1 and nothing had ever
    emitted it, so a structurally inapplicable rung -- ``fused`` outside its
    256..1024 packing bound -- was recorded as ``failed``, identical to a mode
    that regressed. The one-passed-row gate survives that; a COUNT-based gate
    cannot, which is why this landed with ``manifest``'s row counts rather than
    before them. The applicability rule itself is NOT here: it belongs to the
    profile (``study/profiles.py``), because it is a claim about what the modes
    support and this module only walks what it is given.

    Default ``None`` means no skips, which is what every existing caller gets.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import results_io  # noqa: E402
import schema  # noqa: E402


def _f(value) -> float:
    """A CSV cell as a float. Cells are strings, and a blank one is None."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _i(value) -> int:
    """A CSV cell as an int; -1 marks 'not recorded' in the progress line only."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return -1


def _rung(
    mode: str,
    seq: int,
    study_id: str,
    warmup: int,
    samples: int,
    runs_per_sample: int,
    scratch: str,
) -> dict:
    """One rung, in its own process. See ONE PROCESS PER RUNG in the docstring."""
    out = os.path.join(scratch, f"{mode}_{seq}.csv")
    cmd = [
        sys.executable,
        os.path.join(_HERE, "run_mode.py"),
        "--mode",
        mode,
        "--seq",
        str(seq),
        "--out",
        out,
        "--study-id",
        study_id,
        "--warmup",
        str(warmup),
        "--samples",
        str(samples),
        "--runs-per-sample",
        str(runs_per_sample),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)

    if os.path.isfile(out):
        rows = results_io.read_rows(out)
        if rows:
            return rows[0]

    # The child died before writing. run_mode never raises for a *measurement*
    # failure -- it writes a failed row -- so reaching here means the process
    # itself died: a signal, an OOM kill, a hard XRT fault. Synthesize the row
    # rather than dropping the rung, or the ladder would silently have a hole
    # where its worst failure was.
    row = schema.empty_row("results")
    row["execution_mode"] = schema.EXECUTION_MODE_CSV.get(mode, mode)
    row["study_id"] = study_id
    row["seq_len"] = seq
    row["study_case_id"] = f"{seq}x?_encoder_bert"
    row["study_case_label"] = f"{mode} seq {seq}"
    row["run_status"] = "failed"
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    row["failure_message"] = f"child exited {proc.returncode} without writing a row" + (
        f": {tail[-1][:200]}" if tail else ""
    )
    return row


def _skipped_row(mode: str, seq: int, study_id: str, reason: str) -> dict:
    """A rung that cannot apply. Written, never run. See the docstring section.

    The reason rides ``failure_message`` because a new column is a schema
    version bump; the ``skipped:`` prefix keeps it readable wherever that field
    surfaces, including the smoke gate's verbatim quote.
    """
    row = schema.empty_row("results")
    row["execution_mode"] = schema.EXECUTION_MODE_CSV.get(mode, mode)
    row["study_id"] = study_id
    row["seq_len"] = seq
    row["study_case_id"] = f"{seq}x?_encoder_bert"
    row["study_case_label"] = f"{mode} seq {seq}"
    row["run_status"] = "skipped"
    row["failure_message"] = f"skipped: {reason}"
    return row


def walk(
    modes: list[str],
    seqs: list[int],
    out_dir: str,
    study_id: str,
    warmup: int,
    samples: int,
    runs_per_sample: int,
    skip_reason=None,
) -> list[dict]:
    """Run every (mode, seq) rung, writing each mode's CSV as it fills.

    ``skip_reason(mode, seq)`` returning text records the rung as ``skipped``
    and starts no child process.
    """
    every = []
    scratch = tempfile.mkdtemp(prefix="ladder-rungs-")
    for mode in modes:
        rows = []
        out = os.path.join(out_dir, f"{mode}.csv")
        for seq in seqs:
            t0 = time.perf_counter()
            reason = skip_reason(mode, seq) if skip_reason is not None else None
            if reason:
                row = _skipped_row(mode, seq, study_id, reason)
            else:
                row = _rung(
                    mode, seq, study_id, warmup, samples, runs_per_sample, scratch
                )
            rows.append(row)
            every.append(row)
            # Rewrite the whole mode after each rung: cheap, and a killed run
            # keeps what it measured.
            results_io.write_rows(out, rows)
            wall = time.perf_counter() - t0
            if row["run_status"] == "passed":
                print(
                    f"[ladder] {mode:9s} seq {seq:5d}"
                    f"  avg {_f(row['avg_latency_ms']):9.3f} ms"
                    f"  subs {_i(row['host_submissions_per_layer']):3d}"
                    f"  herd {_i(row['herd_launches']):4d}"
                    f"  sync {_i(row['sync_boundaries']):4d}"
                    f"  ({wall:.0f}s wall)",
                    flush=True,
                )
            elif row["run_status"] == "skipped":
                print(
                    f"[ladder] {mode:9s} seq {seq:5d}  SKIPPED  "
                    f"{row['failure_message']}",
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
    skipped = [r for r in rows if r["run_status"] == "skipped"]
    print(
        f"[ladder] {len(passed)}/{len(rows)} rungs passed"
        + (f", {len(skipped)} skipped" if skipped else "")
    )
    for r in rows:
        if r["run_status"] == "failed":
            print(f"[ladder]   FAILED {r['study_case_label']}: {r['failure_message']}")
    # A skipped rung is not a failure: it is a rung the mode does not support,
    # and reporting it as one is the confusion `skipped` exists to remove.
    return 0 if len(passed) + len(skipped) == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
