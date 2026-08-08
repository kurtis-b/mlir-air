# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read iron result rows into this study's schema, refusing what cannot compare.

CONTRACT
    ``adapt_iron_row(row)`` takes one row of iron's ``RESULTS_CSV_FIELDNAMES`` and
    returns a row of ``schema.RESULTS_FIELDNAMES``. Fields whose meaning survives
    the crossing are carried over; fields whose meaning does not are left
    ``None`` and recorded in ``failure_message``. Asking for one of those
    explicitly raises ``IncomparableField``.

    That refusal is the module's whole reason to exist. Doc 03: "write an
    explicit adapter rather than pretending the schemas are identical. The
    adapter's job is to make compare_results_roots.py able to read both, and to
    fail loudly on fields whose semantics differ rather than silently comparing
    incomparable numbers."

WHAT DOES NOT CROSS, AND WHY -- read before adding anything to _SAFE
    **The latency fields.** iron builds ``timed_total_sec`` two different ways.
    Its plain path sums per-sample durations (``modes.py``: ``timed_total_sec =
    sum(latencies_sec)``). Its power path instead runs ``forward_once()`` in a
    ``while True`` until the POWER SAMPLER is satisfied -- minimum runs, minimum
    duration, minimum sample count -- and takes one ``perf_counter()`` span
    across the lot, with a ``time.sleep(0)`` yield inside the timed region. So
    ``measured_inference_count`` is an output of power sampling, not an input,
    and the two paths do not even agree with each other. Neither matches this
    schema's declared region (post-warmup, buffers resident, through final sync,
    compile and weight upload excluded).

    **The power block.** Both filter outliers at modified-Z >= 3.5, but a
    filtered mean is only comparable if the sample populations are, and iron's
    are collected over that dynamic window.

    **The dispatch counts.** iron records ``npu_dispatch_count`` -- one scalar.
    This study records a six-field vector precisely because doc 03 found a
    scalar cannot distinguish the taxonomy's points. One number does not become
    six by renaming it.

    **Tolerance-derived verdicts.** iron's end-to-end gate is ``FINAL_REL_TOL``
    0.1 / ``FINAL_ABS_TOL`` 0.5 with a 5% mismatch budget, and above
    ``seq_len`` 512 it degrades to a finite-output check. This port uses an FP32
    reference at the registry's rtol/atol with ZERO permitted mismatches. A
    ``run_status`` of ``passed`` therefore means something materially weaker on
    iron's side, so it crosses only as ``iron_run_status`` in the failure text,
    never into this schema's ``run_status``.

FOOTGUNS
    - **``hybrid`` is iron's name for ``coarse``.** Handled here because porting
      convention 7 confines that mapping to the schema module and this adapter;
      it must not appear anywhere else.
    - **Carrying a field over is a claim about its semantics, not its name.**
      Every entry in ``_SAFE`` below should be defensible as "this means the
      same thing on both sides". When unsure, leave it out: a missing column is
      visible, a wrongly-carried one is not.
    - The adapter does not read CSV files. Give it dicts; file handling and
      schema-version dispatch belong to the caller.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import schema  # noqa: E402


class IncomparableField(ValueError):
    """Raised when a caller asks for a field whose meaning does not cross."""


#: iron column -> this schema's column, for fields that mean the same thing on
#: both sides: identity, shape, and counts that are inputs rather than outputs.
_SAFE: dict[str, str] = {
    "study_id": "study_id",
    "study_case_id": "study_case_id",
    "study_case_label": "study_case_label",
    "workload_variant": "workload_variant",
    "backend": "backend",
    "execution_mode": "execution_mode",
    "seq_len": "seq_len",
    "hidden_size": "hidden_size",
    "intermediate_size": "intermediate_size",
    "num_attention_heads": "num_attention_heads",
    "attention_head_size": "attention_head_size",
    "batch_size": "batch_size",
    "dtype": "dtype",
    "use_bias": "use_bias",
    "weights_source": "weights_source",
    "warmup_runs": "warmup_runs",
    "process_model": "process_model",
    "selected_config_json": "selected_config_json",
    "selected_candidate_ids_json": "selected_candidate_ids_json",
    "is_best": "is_best",
}

#: iron column -> why it does not cross. Asking for one of these raises.
_INCOMPARABLE: dict[str, str] = {
    "timed_total_sec": "iron builds this two ways -- sum of per-sample durations "
    "on the plain path, and one perf_counter span over a power-sampler-chosen "
    "run count (with a sleep(0) inside) on the power path. Neither matches this "
    "schema's declared region.",
    "avg_latency_ms": "derived from timed_total_sec; inherits its ambiguity.",
    "min_latency_ms": "derived from timed_total_sec; inherits its ambiguity.",
    "max_latency_ms": "derived from timed_total_sec; inherits its ambiguity.",
    "measured_inference_count": "on iron's power path this is an OUTPUT of "
    "power sampling, not a configured input as it is here.",
    "latency_sample_count": "same: sample count is sampler-driven on iron's "
    "power path.",
    "runs_per_sample": "same.",
    "compile_setup_time_ms": "iron's boundary between compile and dispatch "
    "setup is not this one; both exclude it from latency but they do not "
    "measure the same span.",
    "host_qkv_precompute_ms": "only meaningful against a mode that precomputes "
    "the same way.",
    "effective_gflops_per_sec": "derived from latency.",
    "effective_gflops_per_sec_per_watt": "derived from latency and power.",
    "npu_dispatch_count": "one scalar; this study records a six-field dispatch "
    "vector because doc 03 established a scalar cannot separate the taxonomy.",
    "npu_unique_instruction_binary_count": "packaging detail, not a taxonomy "
    "point; no counterpart here.",
    "npu_unique_xclbin_count": "same.",
    "run_status": "iron's gate is FINAL_REL_TOL 0.1 / FINAL_ABS_TOL 0.5 with a "
    "5% mismatch budget, degrading to a finite-output check above seq_len 512. "
    "This port permits ZERO mismatches at the registry tolerance, so `passed` "
    "is a materially weaker claim on iron's side.",
    "validation_error_count": "counted against a different tolerance.",
}

# The power block crosses as a unit or not at all: a filtered mean is comparable
# only if the sample populations are, and iron's come from a dynamic window.
for _f in schema.POWER_FIELDNAMES:
    _INCOMPARABLE.setdefault(
        _f,
        "iron samples power across a window whose length the sampler chooses; "
        "the filtered statistics are not drawn from a comparable population.",
    )


def incomparable_reason(iron_field: str) -> str | None:
    """Why ``iron_field`` does not cross, or None if it does."""
    return _INCOMPARABLE.get(iron_field)


def adapt_iron_row(
    row: dict[str, object], *, require: tuple[str, ...] = ()
) -> dict[str, object]:
    """One iron results row as a schema-v1 row.

    ``require`` names iron fields the caller insists on carrying over; each one
    that does not cross raises ``IncomparableField``. Callers that just want the
    comparable subset pass nothing and read ``failure_message`` to see what was
    dropped.
    """
    for field in require:
        reason = incomparable_reason(field)
        if reason:
            raise IncomparableField(f"{field} does not cross: {reason}")
        if field not in _SAFE:
            raise IncomparableField(
                f"{field} is not a known iron results column; it is neither "
                "carried over nor recorded as incomparable, so nothing can be "
                "claimed about it"
            )

    out = schema.empty_row("results")
    for iron_name, ours in _SAFE.items():
        if iron_name in row:
            out[ours] = row[iron_name]

    # execution_mode needs NO translation. Convention 7 keeps `hybrid` as this
    # study's CSV value for coarse precisely so iron's trees stay diffable, so
    # the value crosses unchanged. An earlier revision rewrote `hybrid` to
    # `coarse` here, which turned a valid CSV value into one the schema rejects
    # -- the code name and the CSV value are different on purpose.

    # iron rows carry no attention_path; leaving it None is honest, and
    # validate_row permits it. Inferring one from execution_mode would be a
    # guess about where attention ran, which is the confound that column exists
    # to expose.
    dropped = sorted(f for f in row if f in _INCOMPARABLE)
    iron_status = row.get("run_status")
    note = f"adapted from iron schema; iron_run_status={iron_status!r}"
    if dropped:
        note += f"; {len(dropped)} incomparable field(s) not carried: " + ", ".join(
            dropped
        )
    out["failure_message"] = note

    schema.validate_row(out, "results")
    return out
