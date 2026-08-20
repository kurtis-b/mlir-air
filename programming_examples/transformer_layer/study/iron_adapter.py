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

    `[2026-08-20]` The tree level, and the port validation doc 00's success
    criterion 3 asks for ("comparable -- through an explicit adapter -- to the
    iron result trees, so the port can be validated rather than merely run"):

    - ``read_iron_results(root)`` reads an iron results root
      (``end_to_end/results_all_power.csv``) into adapted rows, and
      ``split_by_mode_csv(rows)`` files them under the per-mode CSV names this
      study's roots use, so the two layouts meet on ``compare_roots.row_key``.
    - ``validate_port(iron_root, our_root, csvs)`` is the validation itself:
      for every row this port measured that iron also measured, the SHAPE
      must agree field for field -- that is the claim "the same layer" rests
      on, and it is the only cross-toolchain claim the refusals above leave
      standing. It FAILS on a shape disagreement, reports coverage both ways,
      and never compares a latency, because the refusals are not optional.

    The first real-tree run found the join itself broken: this port stamps
    ``study_case_id`` with the SPECS shape key (``512x768_encoder_bert`` --
    the resume key, pinned by tests and every results root) where iron stamps
    its family id (``baseline_768``), so the declared key matched 0 rows of a
    162-row iron tree. The adapter now TRANSLATES the identity through
    ``cases.FAMILY_SPECS`` (the authority for iron's family ids and their
    shapes), refusing a family it does not know or a row whose own width
    contradicts its family, and keeps iron's id in ``study_case_label``, the
    field the schema declares human-readable and never parsed.

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
    - ``adapt_iron_row`` does not read CSV files. Give it dicts; file handling
      and schema-version dispatch belong to the caller. ``read_iron_results``
      is the one file reader, and it reads ONE iron layout -- the end-to-end
      results file -- not iron's tuning, block or component tables.
    - **The identity translation is a claim too.** iron's ``baseline_768`` is
      this port's ``{seq}x768_encoder_bert`` only because ``cases.FAMILY_SPECS``
      says so and the iron row's own ``hidden_size``/``workload_variant`` agree.
      A family outside that table is refused, not transliterated.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import schema  # noqa: E402

#: Where an iron results root keeps the rows this adapter reads. iron's
#: unattended runner writes one file for all modes and all cases; this study
#: writes one CSV per mode.
IRON_RESULTS_REL = "end_to_end/results_all_power.csv"

#: The shape fields whose agreement is the whole of "the same layer". Names are
#: identical on both sides and every one is in _SAFE.
SHAPE_FIELDS: tuple[str, ...] = (
    "workload_variant",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "attention_head_size",
    "batch_size",
    "dtype",
)


class IncomparableField(ValueError):
    """Raised when a caller asks for a field whose meaning does not cross."""


class UnknownIronCase(ValueError):
    """Raised for an iron ``study_case_id`` this port has no family for."""


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
    # `[2026-08-09]` This entry used to read "same", deferring to the line above
    # it, and its REASON was wrong even though its verdict was right. It was
    # written under the SUPERSEDED taxonomy, where the modes were distinguished
    # by who sequences the work and packaging really was incidental. Under the
    # corrected axis -- reconfiguration cost against DRAM traffic (doc 03) --
    # this is the opposite of incidental: it is `offload`'s OWN axis, the mode
    # is defined as the one that minimizes it, and its two packaging paths
    # differ in this column and in no other. This port now populates it.
    #
    # That makes crossing it MORE dangerous, not less, which is why the verdict
    # does not change. This port's value is defined per measured layer as the
    # count of distinct binaries; what iron's is counted over is not established
    # anywhere in this tree, and every other entry here that claims something
    # about iron's internals cites the source for it. Carrying a definitional
    # field across on an unverified assumption would silently corrupt the very
    # axis it names -- exactly the failure the contract at the top of this
    # module exists to prevent. Establish iron's unit first, with a source; then
    # this can move to _SAFE and not before.
    "npu_unique_xclbin_count": "under the corrected taxonomy this is NOT a "
    "packaging detail -- it is `offload`'s defining axis, and this port records "
    "it per measured layer as the count of distinct binaries. What iron counts "
    "it over is not established in this tree, and a definitional field carried "
    "across on an assumption would corrupt the axis it names.",
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
    """One iron results row as a current-schema row.

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

    # Identity: iron's family id becomes this port's shape key, through the
    # one table that knows both, and only when the row's own width agrees.
    iron_case = row.get("study_case_id")
    if iron_case is not None and str(iron_case) != "":
        out["study_case_id"] = case_key_for(row)
        out["study_case_label"] = f"iron:{iron_case}"

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


def case_key_for(row: dict[str, object]) -> str:
    """This port's ``study_case_id`` for an iron row, or raise.

    The key is ``{seq_len}x{hidden_size}_{workload_variant}`` -- what
    ``run_mode._shape_for`` derives for a measured row and
    ``run_ladder._case_identity`` for a synthesized one. The family comes from
    ``cases.FAMILY_SPECS`` keyed by iron's id; the row's own ``hidden_size`` and
    ``workload_variant``, when present, must agree with it or the row is
    refused -- a tree whose case ids and widths disagree is corrupt, and
    translating it would launder that into a clean-looking join.
    """
    import cases

    iron_case = str(row.get("study_case_id"))
    fam = cases.FAMILY_SPECS.get(iron_case)
    if fam is None:
        raise UnknownIronCase(
            f"iron study_case_id {iron_case!r} is not a family in "
            f"cases.FAMILY_SPECS ({sorted(cases.FAMILY_SPECS)}); it cannot be "
            "translated into this port's case key without guessing"
        )
    seq = row.get("seq_len")
    if seq is None or str(seq) == "":
        raise UnknownIronCase(
            f"iron row for {iron_case!r} carries no seq_len; the case key "
            "needs one"
        )
    for field, expected in (
        ("hidden_size", fam.hidden_size),
        ("workload_variant", fam.workload_variant),
    ):
        got = row.get(field)
        if got is not None and str(got) != "" and str(got) != str(expected):
            raise UnknownIronCase(
                f"iron row says study_case_id={iron_case!r} but "
                f"{field}={got!r} where the family has {expected!r}; refusing "
                "to translate a row that disagrees with its own case"
            )
    return f"{int(seq)}x{fam.hidden_size}_{fam.workload_variant}"


def is_iron_root(root) -> bool:
    """Does ``root`` hold the one iron layout this adapter reads?"""
    from pathlib import Path

    return (Path(root) / IRON_RESULTS_REL).is_file()


def read_iron_results(root) -> list[dict[str, object]]:
    """Every row of ``root``'s end-to-end results file, adapted.

    Raises ``FileNotFoundError`` for a root without the file -- a missing tree
    must not read as an empty one -- and lets ``UnknownIronCase`` propagate,
    because one untranslatable row means the tree is not the grid this port
    declares and the caller should know before comparing anything.
    """
    import csv
    from pathlib import Path

    path = Path(root) / IRON_RESULTS_REL
    if not path.is_file():
        raise FileNotFoundError(
            f"{root} is not an iron results root: no {IRON_RESULTS_REL}"
        )
    with path.open(newline="", encoding="utf-8") as handle:
        return [adapt_iron_row(r) for r in csv.DictReader(handle)]


#: iron CSV mode value -> the per-mode file this study's roots write. `hybrid`
#: maps to the canonical `coarse` cell, not to `coarse_c2`/`coarse_c3`: iron
#: has ONE hybrid design and doc 30 chose C1 as this port's `coarse` by
#: measurement, so C1 is the cell that answers to iron's row.
MODE_CSV_NAMES: dict[str, str] = {
    "hybrid": "coarse.csv",
    "runlist": "runlist.csv",
    "offload": "offload.csv",
}


def split_by_mode_csv(rows: list[dict[str, object]]) -> dict[str, list[dict]]:
    """File adapted rows under this study's per-mode CSV names."""
    out: dict[str, list[dict]] = {}
    for row in rows:
        mode = str(row.get("execution_mode"))
        name = MODE_CSV_NAMES.get(mode)
        if name is None:
            raise IncomparableField(
                f"iron execution_mode {mode!r} has no per-mode CSV in this "
                f"study ({sorted(MODE_CSV_NAMES)})"
            )
        out.setdefault(name, []).append(row)
    return out


def validate_port(iron_root, our_root, csvs: list[str]):
    """Does this port measure the layers iron measured? A ``compare_roots.Report``.

    For each of this port's rows (in ``csvs`` under ``our_root``) with a
    counterpart in the iron tree on ``compare_roots.row_key``, every field in
    ``SHAPE_FIELDS`` must agree -- FAIL otherwise. A row with no counterpart is
    reported with the reason: ``fused_elf`` has no iron mode at all; a (family,
    seq_len) outside iron's grid is outside it. Coverage of iron's grid by this
    port is printed as counts. iron's ``run_status`` rides along as information,
    never as a verdict (it is a weaker claim -- see _INCOMPARABLE). No latency
    or power field is read on either side, which is the contract, not a gap.

    An empty ``csvs`` FAILS, as in ``compare_roots``: a comparison over nothing
    must not print OK.
    """
    from pathlib import Path

    import compare_roots
    import results_io

    report = compare_roots.Report()
    report.say(f"iron root : {iron_root}")
    report.say(f"port root : {our_root}")
    if not csvs:
        report.fail(
            "no CSVs were named, so this validation proved nothing. Name the "
            "files the port run produced."
        )
        report.say("")
        report.say(f"warnings: {report.warnings}   failures: {report.failures}")
        report.say("VERDICT: PROBLEM")
        return report

    try:
        iron_rows = read_iron_results(iron_root)
    except (FileNotFoundError, ValueError) as e:
        # UnknownIronCase and IncomparableField are ValueErrors; so are a
        # malformed seq_len and a schema validation failure. All four mean
        # the tree cannot be read, and a tree that cannot be read is a
        # PROBLEM report, not a traceback.
        report.fail(f"iron tree unreadable -- {e}")
        report.say("")
        report.say(f"warnings: {report.warnings}   failures: {report.failures}")
        report.say("VERDICT: PROBLEM")
        return report
    iron_by_key = {compare_roots.row_key(r): r for r in iron_rows}
    iron_modes = {str(r.get("execution_mode")) for r in iron_rows}
    iron_cases = {str(r.get("study_case_label")) for r in iron_rows}
    iron_seqs = {str(r.get("seq_len")) for r in iron_rows}
    report.say(
        f"iron rows: {len(iron_rows)} over {len(iron_cases)} families, "
        f"{len(iron_seqs)} lengths, modes {sorted(iron_modes)}"
    )

    covered: set[tuple[str, ...]] = set()
    for rel in csvs:
        report.say(f"\n=== {rel} ===")
        path = Path(our_root) / rel
        if not path.exists():
            report.fail("missing in the port root; it was named, so its absence fails")
            continue
        try:
            # The strict-prefix reader: the shape block is in the v1 prefix,
            # and schema v2 silently took ladder_report out on 15 v1 trees
            # before plots.py adopted this reader (README, figure tier row).
            ours = results_io.read_rows_compatible(path)
        except Exception as e:
            report.fail(f"unreadable through the schema's prefix reader -- {e}")
            continue
        if not ours:
            report.fail(
                "holds no rows; it was named as a measurement file, so an "
                "empty one is a failure rather than something to skip"
            )
            continue
        for row in ours:
            key = compare_roots.row_key(row)
            mode = str(row.get("execution_mode"))
            status = row.get("run_status")
            other = iron_by_key.get(key)
            if other is None:
                if mode not in iron_modes:
                    report.say(
                        f"  info  {key}: no iron counterpart -- iron has no "
                        f"{mode!r} mode"
                    )
                elif str(row.get("seq_len")) not in iron_seqs:
                    report.say(
                        f"  info  {key}: no iron counterpart -- seq_len outside "
                        "iron's ladder"
                    )
                else:
                    report.warn(
                        f"{key}: iron measured this mode and length but has no "
                        "row under this case; check the family translation"
                    )
                continue
            covered.add(key)
            disagreements = []
            unset = []
            for field in SHAPE_FIELDS:
                mine, theirs = row.get(field), other.get(field)
                if mine is None or theirs is None:
                    unset.append(
                        f"{field} ({'port' if mine is None else 'iron'} unset)"
                    )
                    continue
                if str(mine) != str(theirs):
                    disagreements.append(f"{field}: port={mine!r} iron={theirs!r}")
            compared = len(SHAPE_FIELDS) - len(unset)
            note = str(other.get("failure_message") or "")
            iron_status = next(
                (s.strip() for s in note.split(";") if "iron_run_status" in s),
                "iron_run_status unknown",
            )
            if disagreements:
                report.fail(f"{key} shape disagrees -- " + "; ".join(disagreements))
                continue
            if compared == 0:
                report.fail(
                    f"{key}: no shape field is set on both sides, so nothing "
                    "was compared"
                )
                continue
            report.say(
                f"  ok    {key}: shape agrees on {compared}/{len(SHAPE_FIELDS)} "
                f"fields; port run_status={status!r}; {iron_status}"
            )
            if unset:
                # A partial agreement is reported as partial: "agrees on 7"
                # over a row with three fields unset is the false OK the
                # review flagged.
                report.warn(f"{key}: {len(unset)} shape field(s) not compared -- "
                            + ", ".join(unset))

    report.say("")
    report.say(
        f"iron grid covered by this port root: {len(covered)}/{len(iron_by_key)} "
        "(coverage is information; the verdict is shape agreement)"
    )
    if not covered:
        # Zero shared points means zero comparisons, and a validation that
        # compared nothing must not print OK -- the same rule as the empty
        # --csv case. Wholly mis-keyed roots, a header-only iron file and a
        # port root outside iron's grid all land here, each already named
        # above; this line is what turns them from information into a verdict.
        report.fail(
            "no shared (case, mode, seq_len) point was compared, so this "
            "validation proved nothing about the port"
        )
    report.say(f"warnings: {report.warnings}   failures: {report.failures}")
    report.say(f"VERDICT: {'OK' if report.failures == 0 else 'PROBLEM'}")
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(
        description="Validate a port results root against an iron results tree."
    )
    ap.add_argument("--iron-root", required=True, type=Path)
    ap.add_argument("--root", required=True, type=Path, help="this port's root")
    ap.add_argument(
        "--csv",
        action="append",
        default=[],
        dest="csvs",
        metavar="REL/PATH.csv",
        help="a results CSV under --root. Repeatable. Required: with none, "
        "the validation checks nothing.",
    )
    args = ap.parse_args(argv)
    report = validate_port(
        args.iron_root.expanduser().resolve(),
        args.root.expanduser().resolve(),
        args.csvs,
    )
    print(report.render())
    return 0 if report.failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
