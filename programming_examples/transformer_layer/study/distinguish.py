# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The cross-mode distinguishability gate over a profile's recorded dispatch vectors.

    python3 study/distinguish.py <results_root> [--seq-len N]

Ported from the retired port-loop Phase E check (``phase_e_checks.distinguish``, at git tag
``pre-cleanup-20260821``) so the one cross-mode claim the four modes make stays gated after the
harness that carried it was removed in the 2026-08-21 cleanup. Doc 08e defines the criterion; it
is ORDINAL over the recorded vectors, never an absolute threshold -- ``coarse`` measures ~131
runlist entries, 128 of them one operator's row blocking, so any threshold would measure
``build_addnorm_module``'s L1 capacity rather than the taxonomy.

Input is a results root written by ``run_profile.py`` (one ``<mode>.csv`` per mode). A sequence
length is gated only where all four modes have a ``passed`` row; lengths where a mode is skipped
are reported, not failed -- the skips are derived from the refusing builders (doc 54 §3).

The four clauses:
  1. distinctness -- no two modes record the same vector (the floor: an identical vector is not
     measuring the boundary);
  2. ``offload`` submits more than every other mode and aggregates nothing
     (``runlist_entries_per_submission == 1``);
  3. ``runlist`` EXECUTES more than ``coarse`` (``herd_launches``; entries would be true by
     construction, since runlist is coarse's schedule decomposed -- doc 08e's J4 note);
  4. ``fused`` crosses fewer sync boundaries than ``coarse``.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import results_io  # noqa: E402

MODES = ("offload", "runlist", "coarse", "fused")
VECTOR_KEYS = (
    "host_submissions_per_layer",
    "runlist_entries_per_submission",
    "air_launches_per_elf",
    "herd_launches",
    "sync_boundaries",
    "bytes_transferred",
)


COUNT_KEYS = tuple(k for k in VECTOR_KEYS if k != "runlist_entries_per_submission")

#: Doc 54 walk 2 at seq 512 (results/g-full-baseline768-w2, devq 435), verbatim -- the measured
#: four-mode vectors the host tests use as their passing reference.
WALK2_512_VECTORS = {
    "offload": dict(host_submissions_per_layer=30, runlist_entries_per_submission=1.0,
                    air_launches_per_elf=30, herd_launches=90, sync_boundaries=90,
                    bytes_transferred=44040192),
    "runlist": dict(host_submissions_per_layer=17, runlist_entries_per_submission=91 / 17,
                    air_launches_per_elf=49, herd_launches=151, sync_boundaries=106,
                    bytes_transferred=20447232),
    "coarse": dict(host_submissions_per_layer=4, runlist_entries_per_submission=4.75,
                   air_launches_per_elf=11, herd_launches=33, sync_boundaries=59,
                   bytes_transferred=22020096),
    "fused": dict(host_submissions_per_layer=1, runlist_entries_per_submission=3.0,
                  air_launches_per_elf=11, herd_launches=23, sync_boundaries=13,
                  bytes_transferred=21233664),
}


def _num(v):
    if v in (None, "") or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def validate(mode: str, vector: dict) -> list[str]:
    """The retired gate's `vector_totals` rules, per vector: fail CLOSED on anything that is not
    a finite, non-negative number; counts whole; at least one submission; mean x submissions a
    whole number of runlist entries; some bytes moved. A malformed vector must never satisfy a
    clause by accident (NaN compares unequal to everything, so it would pass distinctness)."""
    problems = []
    for k in VECTOR_KEYS:
        raw = vector.get(k)
        x = _num(raw)
        if x is None:
            problems.append(f"{mode}.{k}={raw!r} is not a number")
        elif not math.isfinite(x):
            problems.append(f"{mode}.{k}={raw!r} is not finite")
        elif x < 0:
            problems.append(f"{mode}.{k}={raw!r} is negative")
        elif k in COUNT_KEYS and x != int(x):
            problems.append(f"{mode}.{k}={raw!r} is a count and must be a whole number")
    if problems:
        return problems
    subs = int(_num(vector["host_submissions_per_layer"]))
    if subs < 1:
        problems.append(f"{mode} records {subs} host submissions; every recorded sequence submitted at least once")
    product = _num(vector["runlist_entries_per_submission"]) * subs
    if abs(product - round(product)) > 1e-9:
        problems.append(f"{mode}: {product!r} runlist entries over {subs} submission(s) is not a whole number")
    if _num(vector["bytes_transferred"]) <= 0:
        problems.append(f"{mode} moved zero bytes; nothing was transferred to a device")
    return problems


def canonical(vector: dict) -> dict:
    """A validated vector as whole-number totals: the derived mean becomes ``runlist_entries``
    (round(mean x submissions)), so two vectors that differ only in float noise of the mean
    compare EQUAL here, as the retired gate's summed totals did."""
    subs = int(_num(vector["host_submissions_per_layer"]))
    out = {k: int(_num(vector[k])) for k in COUNT_KEYS}
    out["runlist_entries"] = int(round(_num(vector["runlist_entries_per_submission"]) * subs))
    return out


def distinguish(vectors_by_mode: dict) -> list[str]:
    """Return the failed clauses (empty means the four modes are separated)."""
    if set(vectors_by_mode) != set(MODES):
        return [f"need all four modes, have {sorted(vectors_by_mode)}"]
    failures = []
    for m in MODES:
        failures.extend(validate(m, vectors_by_mode[m]))
    if failures:
        return failures
    v = {m: canonical(vectors_by_mode[m]) for m in MODES}
    o, r, c, f = (v[m] for m in MODES)
    for i, a in enumerate(MODES):
        for b in MODES[i + 1:]:
            if v[a] == v[b]:
                failures.append(f"{a} and {b} have identical dispatch vectors")
    for other in ("runlist", "coarse", "fused"):
        if not o["host_submissions_per_layer"] > v[other]["host_submissions_per_layer"]:
            failures.append(
                f"offload host_submissions_per_layer {o['host_submissions_per_layer']} does not "
                f"exceed {other}'s {v[other]['host_submissions_per_layer']}")
    if o["runlist_entries"] != o["host_submissions_per_layer"]:
        failures.append(
            f"offload aggregates: {o['runlist_entries']} entries over "
            f"{o['host_submissions_per_layer']} submissions, but the mode dispatches one GEMM per submission")
    if not r["herd_launches"] > c["herd_launches"]:
        failures.append(
            f"runlist herd_launches {r['herd_launches']} does not exceed coarse's {c['herd_launches']}")
    if not f["sync_boundaries"] < c["sync_boundaries"]:
        failures.append(
            f"fused sync_boundaries {f['sync_boundaries']} is not below coarse's {c['sync_boundaries']}")
    return failures


def rows_from_root(root: str) -> tuple[dict, list[str]]:
    """``{seq_len: {mode: (run_status, vector)}}`` over every row of the root's mode CSVs, plus
    the lines for any (mode, seq_len) recorded more than once -- a duplicate would otherwise
    overwrite its predecessor silently and could stand in for a length that was never walked."""
    out: dict = {}
    problems = []
    for m in MODES:
        path = os.path.join(root, f"{m}.csv")
        if not os.path.exists(path):
            continue
        for row in results_io.read_rows_compatible(path):
            n = int(row["seq_len"])
            if m in out.get(n, {}):
                problems.append(f"seq {n}: FAIL {m} has more than one row")
            out.setdefault(n, {})[m] = (
                row.get("run_status"), {k: row.get(k) for k in VECTOR_KEYS})
    return out, problems


def gate_root(root: str, seq_len: int | None = None,
              expected_seqs=None, expected_skips=None) -> tuple[int, list[str]]:
    """Gate every length where all four modes passed. Returns (lengths gated, lines); a line
    containing ``FAIL`` is a failure. A mode that is ``skipped`` at a length makes that length
    "not gated" (skips are derived from the refusing builders, doc 54 §3); a mode that is
    absent, or recorded with any OTHER status (failed, timeout, interrupted...), is a FAILURE
    at that length -- a run that did not pass is not a structural skip. ``expected_seqs`` (a
    profile's declared lengths) are gated whether or not any mode recorded them, so a length
    that vanished from every CSV is a failure rather than an absence. ``expected_skips`` (the
    profile's declared structural skips, as ``{(mode, seq_len)}``) pins skip IDENTITY: a row
    skipped where the profile declares the rung measurable is a failure, and so is a row passed
    where the profile declares a skip -- the builder refuses there, so a passed row cannot be a
    measurement. Without it a skip and a failing length could trade places and keep every count."""
    by_len, lines = rows_from_root(root)
    gated = 0
    lengths = set(by_len) | set(expected_seqs or ())
    for n in sorted(lengths):
        if seq_len is not None and n != seq_len:
            continue
        have = by_len.get(n, {})
        if expected_skips is not None:
            for m, (status, _) in sorted(have.items()):
                declared = (m, n) in expected_skips
                if status == "skipped" and not declared:
                    lines.append(f"seq {n}: FAIL {m} is skipped but the profile declares it measurable")
                elif status == "passed" and declared:
                    lines.append(f"seq {n}: FAIL {m} is passed where the profile declares a structural skip")
        missing = sorted(set(MODES) - set(have))
        bad = sorted(m for m, (status, _) in have.items() if status not in ("passed", "skipped"))
        skipped = sorted(m for m, (status, _) in have.items() if status == "skipped")
        identity_failed = expected_skips is not None and any(
            line.startswith(f"seq {n}: FAIL") and ("declares" in line) for line in lines)
        if missing or bad:
            what = (["no mode has a row"] if not have else
                    [f"{m} has no row" for m in missing] + [f"{m} run_status={have[m][0]}" for m in bad])
            lines.append(f"seq {n}: FAIL {'; '.join(what)}")
            continue
        if skipped or identity_failed:
            if skipped and not identity_failed:
                lines.append(f"seq {n}: not gated ({', '.join(skipped)} skipped)")
            continue
        gated += 1
        for fail in distinguish({m: vec for m, (_, vec) in have.items()}):
            lines.append(f"seq {n}: FAIL {fail}")
    if seq_len is not None and seq_len not in lengths:
        lines.append(f"seq {seq_len}: FAIL no mode has a row")
    return gated, lines


def declared(profile) -> tuple[tuple, set]:
    """A profile's (declared lengths, declared structural skips as {(mode, seq)})."""
    return tuple(profile.seqs), {(r.mode, r.seq) for r in profile.rungs() if r.skip_reason}


def summary(gated: int, fails: int) -> str:
    if gated == 0:
        return "distinguish: no sequence length has all four modes passed -- nothing gated"
    return f"distinguish: {gated} length(s) gated, {fails} failure(s)"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root")
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--profile", default=None,
                    help="gate against this profile's declared lengths and structural skips")
    args = ap.parse_args(argv)
    expected_seqs = expected_skips = None
    if args.profile:
        import profiles
        prof = profiles.profile(args.profile)
        expected_seqs, expected_skips = declared(prof)
    gated, lines = gate_root(args.root, args.seq_len, expected_seqs, expected_skips)
    for line in lines:
        print(line)
    fails = sum(1 for line in lines if ": FAIL " in line)
    print(summary(gated, fails))
    return 2 if gated == 0 else (1 if fails else 0)


if __name__ == "__main__":
    sys.exit(main())
