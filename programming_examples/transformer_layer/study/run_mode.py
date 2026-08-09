# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run one execution mode on hardware and write a schema-v1 results row.

    python3 study/run_mode.py --mode coarse --out results/coarse.csv

CONTRACT
    Prepares a mode through the SAME seam ``opcheck.py`` uses -- the mode's own
    ``prepare_*`` from ``opcheck_specs.SPECS`` -- times its dispatch, compares
    the layer output against the FP32 golden reference, and writes one row via
    ``results_io``. It reimplements no part of a mode; a mode that changes
    changes here for free, which is the point of going through the catalogue.

THE TIMED REGION, matching what ``schema.py`` declares
    Cache construction, ELF compilation and golden-reference construction all
    happen BEFORE the clock starts, and their cost is reported separately in
    ``compile_setup_time_ms``. The clock covers ``run_block``-equivalent
    dispatch only: warmup iterations first and discarded, then
    ``runs_per_sample`` iterations per sample, ``latency_sample_count`` samples,
    ``timed_total_sec`` their sum. Folding compile into latency would flatter a
    fused mode for a reason that has nothing to do with its execution boundary.

WHY A FAILED RUN STILL WRITES A ROW
    ``run_status``/``failure_message`` carry the failure and every metric stays
    ``None``. Phase F's gate distinguishes "no passed row" from "no rows", and
    it cannot do that if a failure writes nothing. A crash before dispatch is
    the one case that writes nothing, because there is no row to attribute.

FOOTGUNS
    - **``runlist_entries_per_submission`` is a MEAN, not a total.**
      ``dispatch_vector_totals`` returns the total; doc 03 defines the column as
      a per-submission mean, so this divides. Summing the column across rows
      does NOT give a mode's entry count -- that is
      ``sum(round(mean * submissions))``.
    - The mode's CSV value is not its code name: ``coarse`` records as
      ``hybrid`` (convention 7). Read it from ``schema.EXECUTION_MODE_CSV``.
    - This needs the NPU. Serialize it like every other device job:
      ``flock -x -w 1800 /tmp/mlir-air-npu.lock python3 study/run_mode.py ...``
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.dirname(_HERE)
_PE = os.path.dirname(_EXAMPLE)
for _p in (_PE, os.path.join(_PE, "llms"), _EXAMPLE, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import results_io  # noqa: E402
import schema  # noqa: E402

# Which modes run attention on the host. Derived from a fact the code states:
# `runlist` imports `blocked_attention` from `pattern/blocked_attention.py` and
# calls it; `coarse`, `fused` and `offload` import at most `round_bf16` from that
# module and dispatch attention to the device. `test_attention_path.py`
# re-derives this from the pattern modules' imports and fails if the map drifts,
# because the map is a claim about other files.
#
# It is recorded per row because it is the single covariate that decides how a
# latency curve reads -- the sequence ladder's host-attention modes fit a
# log-log slope of 1.25 against the device-attention modes' 1.03 and 1.15 -- and
# it was None in every row of the first ladder, which is how the schema had a
# field for it and the data did not.
#
# `[2026-08-08]` offload moved host_torch -> device, and `[2026-08-09]` runlist
# followed. Both are the corrected taxonomy landing: offload's two attention
# matmuls are LINEAR so they belong on the NPU, and runlist is defined as every
# operator on the device with nothing on the host, softmax included.
# test_attention_path.py caught this map going stale on the first of those,
# which is the job it was written for.
#
# ATTENTION_PATH IS NO LONGER A COVARIATE IN THIS STUDY. All four modes are now
# on the device side and the field is constant across every row a run can
# produce. That is the designed end state, not a loss -- but it has a
# consequence worth stating where the map lives, because it invalidates a
# published reading: the first sequence ladder's headline was that its slopes
# split on attention placement (host-attention modes at 1.23-1.27 against
# device-attention modes at 1.03-1.17), and that split CANNOT BE REPRODUCED
# from any future run, because no mode sits on the host side any more. A rerun
# showing separated slopes is measuring something else and needs a new
# explanation, not the old one restated.
ATTENTION_PATH_BY_MODE = {
    "coarse": "device",
    "fused": "device",
    "offload": "device",
    "runlist": "device",
}


def _spec_for(mode: str) -> dict:
    """The catalogue row for ``mode``, so shapes are not restated here."""
    from opcheck_specs import SPECS

    for spec in SPECS:
        if spec["operator"] == mode:
            return spec
    known = sorted({s["operator"] for s in SPECS})
    raise SystemExit(f"no SPECS row for mode {mode!r}; known operators: {known}")


def _stage_stats(actual, expected, atol):
    """The per-boundary comparison the dispatch seam calls. Numbers only."""
    import numpy as np

    a = np.asarray(actual, dtype=np.float32)
    e = np.asarray(expected, dtype=np.float32)
    err = np.abs(a - e)
    den = np.abs(e)
    nz = den > 0
    return {
        "n_elements": int(a.size),
        "n_mismatch": int((err > atol + 1.6e-2 * den).sum()),
        "mean_rel_L1": float(err[nz].sum() / den[nz].sum()) if nz.any() else 0.0,
        "atol_required": float(np.max(np.maximum(err - 1.6e-2 * den, 0.0))),
    }


def _shape_for(spec: dict, seq_len: int | None) -> tuple[dict, str]:
    """The spec's shape, optionally at a different sequence length (J3's ladder).

    Returns a COPY. Mutating ``spec["shape"]`` would rewrite the module-level
    ``SPECS`` row, so a second rung in the same process would silently inherit
    the first rung's length -- a ladder that reports several lengths and
    measured one.
    """
    shape = dict(spec["shape"])
    if seq_len is None:
        return shape, spec["shape_key"]
    shape["seq_len"] = seq_len
    # Built, never parsed back out, per opcheck_specs' own note on shape_key.
    emb = shape.get("emb_dim", shape.get("hidden_size", "?"))
    return shape, f"{seq_len}x{emb}_encoder_bert"


def run(
    mode: str,
    warmup: int,
    samples: int,
    runs_per_sample: int,
    seq_len: int | None = None,
) -> dict:
    """Run ``mode`` and return a schema row. Never raises for a run failure."""
    row = schema.empty_row("results")
    row["execution_mode"] = schema.EXECUTION_MODE_CSV.get(mode, mode)
    row["backend"] = "xrt"
    row["workload_variant"] = "encoder_bert"
    row["attention_path"] = ATTENTION_PATH_BY_MODE.get(mode)
    row["batch_size"] = 1
    row["dtype"] = "bf16"
    row["warmup_runs"] = warmup
    row["runs_per_sample"] = runs_per_sample
    row["latency_sample_count"] = samples

    spec = _spec_for(mode)
    shape, shape_key = _shape_for(spec, seq_len)
    row["study_case_id"] = shape_key
    row["study_case_label"] = f"{mode} {shape_key}"
    for key in ("seq_len", "hidden_size", "intermediate_size"):
        if key in shape:
            row[key] = shape[key]
    if "emb_dim" in shape:
        row["hidden_size"] = shape["emb_dim"]
    if "ffn_dim" in shape:
        row["intermediate_size"] = shape["ffn_dim"]
    if "num_heads" in shape:
        row["num_attention_heads"] = shape["num_heads"]
    if "head_dim" in shape:
        row["attention_head_size"] = shape["head_dim"]

    try:
        # --- everything here is OUTSIDE the timed region -------------------
        setup_t0 = time.perf_counter()
        prepared = spec["prepare"](shape)
        dispatch = prepared.get("dispatch")
        if dispatch is None:
            raise RuntimeError(
                f"{mode} has no dispatch seam; run_mode measures whole-layer "
                "modes, not single-operator specs"
            )
        inputs = prepared["inputs"]
        expected = prepared["expected"][0]
        row["compile_setup_time_ms"] = (time.perf_counter() - setup_t0) * 1000.0

        for _ in range(warmup):
            dispatch(inputs, _stage_stats)

        durations = []
        outputs = None
        extra = {}
        for _ in range(samples):
            t0 = time.perf_counter()
            for _ in range(runs_per_sample):
                outputs, extra = dispatch(inputs, _stage_stats)
            durations.append(time.perf_counter() - t0)
        # --- clock stops ---------------------------------------------------

        total_sec = sum(durations)
        per_inference = [d / runs_per_sample for d in durations]
        row["timed_total_sec"] = total_sec
        row["measured_inference_count"] = samples * runs_per_sample
        row["avg_latency_ms"] = (total_sec / (samples * runs_per_sample)) * 1000.0
        row["min_latency_ms"] = min(per_inference) * 1000.0
        row["max_latency_ms"] = max(per_inference) * 1000.0

        totals = _dispatch_totals(extra)
        if totals:
            subs = totals["host_submissions"] or 1
            row["host_submissions_per_layer"] = totals["host_submissions"]
            # A MEAN, per doc 03 -- not the total.
            row["runlist_entries_per_submission"] = totals["runlist_entries"] / subs
            row["air_launches_per_elf"] = totals["air_launches"]
            row["herd_launches"] = totals["herd_launches"]
            row["sync_boundaries"] = totals["sync_boundaries"]
            row["bytes_transferred"] = totals["bytes_transferred"]

        stats = _stage_stats(outputs[0], expected, atol=spec["atol"])
        row["validation_error_count"] = stats["n_mismatch"]
        stages_ok = bool(extra.get("stages_passed", True))
        if stats["n_mismatch"] == 0 and stages_ok:
            row["run_status"] = "passed"
            row["failure_message"] = ""
        else:
            row["run_status"] = "failed"
            row["failure_message"] = (
                f"{stats['n_mismatch']} element(s) outside tolerance "
                f"(atol {spec['atol']}), stages_passed={stages_ok}"
            )
    except Exception as e:  # a failed measurement still writes a complete row
        row["run_status"] = "failed"
        row["failure_message"] = f"{type(e).__name__}: {e}".splitlines()[0][:300]
        traceback.print_exc()
    return row


def _dispatch_totals(extra: dict) -> dict | None:
    """Sum the mode's recorded vectors through the one shared implementation."""
    rows = extra.get("dispatch_vectors")
    if not rows:
        return None
    from opcheck_layer import dispatch_vector_totals

    return dispatch_vector_totals(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", default="coarse")
    ap.add_argument("--out", required=True, help="results CSV to write")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--runs-per-sample", type=int, default=1)
    ap.add_argument("--study-id", default="adhoc")
    ap.add_argument(
        "--seq",
        type=int,
        default=None,
        help="sequence length override; omit for the spec's own (J3's ladder)",
    )
    args = ap.parse_args(argv)

    row = run(args.mode, args.warmup, args.samples, args.runs_per_sample, args.seq)
    row["study_id"] = args.study_id
    results_io.write_rows(args.out, [row])

    status = row["run_status"]
    print(f"[run-mode] {args.mode} @ {row['study_case_id']}: {status}")
    if status == "passed":
        print(
            f"[run-mode]   avg {row['avg_latency_ms']:.3f} ms  "
            f"min {row['min_latency_ms']:.3f} ms  "
            f"submissions {row['host_submissions_per_layer']}  "
            f"sync {row['sync_boundaries']}"
        )
    else:
        print(f"[run-mode]   {row['failure_message']}")
    print(f"[run-mode] wrote {args.out}")
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
