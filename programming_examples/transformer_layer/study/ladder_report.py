# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Turn a sequence-ladder results tree into the comparison J3 exists to produce.

    python3 study/ladder_report.py <ladder-results-dir> [--modes a,b] [--md out.md]

CONTRACT
    Reads ``<mode>.csv`` per mode, each holding one row per rung, and reports
    three things text-only: the latency and dispatch-structure table, a
    **log-log scaling exponent** per mode, and every **crossover** -- a pair of
    modes whose ranking swaps between adjacent rungs.

WHY THE EXPONENT IS THE POINT
    Ranking four modes at one length ranks four numbers. The exponent says what
    each mode's cost is *made of*: fit ``log(latency)`` against ``log(seq_len)``
    and a slope near 1 is a mode whose time scales with the token count -- GEMMs,
    transfers, per-launch overhead -- while a slope near 2 is one paying
    attention's O(seq^2). The four modes do not all run attention in the same
    place (see this doc's confound section), so the exponent is where that
    difference stops being an asterisk and becomes a measurement.

FOOTGUNS
    - **A crossover is only real if both sides pass.** A mode that fails a rung
      has no latency there, and drawing a line across the gap invents a
      measurement. Failed rows are excluded from fits and named in the output;
      a mode with fewer than three passing rungs gets no exponent at all rather
      than a two-point slope presented as a trend.
    - **The exponent is descriptive, not a model.** Three or four rungs over one
      decade cannot separate ``n^2`` from ``n log n`` from a linear term plus a
      constant, and a fixed per-launch cost pulls every slope toward 1 at short
      lengths. Read it as "closer to linear" or "closer to quadratic".
    - **A reported crossover is a hypothesis until a second walk repeats it.**
      `[2026-08-09]` This module reported an `offload`/`runlist` crossover from
      one walk of 512/1024; a second walk under identical conditions reported
      none. `offload` drifts up to 120% within a single walk, which inverts a
      ranking without anything changing. The §Crossovers section says what THIS
      tree of CSVs contains, not what reproduces -- see
      docs/plans/transformer-layer-execution-studies/27-common-ladder-result.md.
    - **Attention placement is no longer the confound it was.** `[2026-08-09]`
      All four modes run attention on the device, so `attention_path` is
      constant across every row a run can produce and no longer explains a
      slope split. Latency comparisons across modes remain descriptive: this
      module reports what was measured; it is not evidence that one mode is
      faster for a reason it names.
    - Rows are matched on ``seq_len``, not on row order, so a partially written
      or reordered CSV lines up correctly.
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

_STRUCT = [
    ("host_submissions_per_layer", "subs"),
    ("herd_launches", "herd"),
    ("sync_boundaries", "sync"),
    ("bytes_transferred", "bytes"),
]


def load(root: str, modes: list[str]) -> dict[str, list[dict]]:
    """{mode: [rows sorted by seq_len]}, skipping modes with no CSV.

    Reads through ``read_rows_compatible``: this is the ANALYSIS tier, and its
    inputs are archived trees. `[2026-08-14]` The v2 bump silently took this
    module out on every v1 tree -- including ``j3_ladder_full``, the nine-rung
    ladder, whose own ``report.md`` this module wrote on 2026-08-08 and could
    not have regenerated the day after v2 landed. Writers stay on the exact
    ``read_rows``; nothing about reporting on a finished measurement needs it.
    """
    out = {}
    for mode in modes:
        path = os.path.join(root, f"{mode}.csv")
        if not os.path.isfile(path):
            continue
        rows = results_io.read_rows_compatible(path)
        for row in rows:
            row["_seq"] = int(row["seq_len"])
            row["_ok"] = row["run_status"] == "passed"
            row["_ms"] = float(row["avg_latency_ms"]) if row["avg_latency_ms"] else None
        out[mode] = sorted(rows, key=lambda r: r["_seq"])
    return out


def exponent(rows: list[dict]) -> float | None:
    """Least-squares slope of log(latency) vs log(seq) over PASSING rungs only.

    None below three points: a two-point slope is an interpolation between two
    measurements, and printing it in the same column as a fitted trend invites
    reading it as one.
    """
    pts = [
        (math.log(r["_seq"]), math.log(r["_ms"])) for r in rows if r["_ok"] and r["_ms"]
    ]
    if len(pts) < 3:
        return None
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    den = sum((x - mx) ** 2 for x, _ in pts)
    if den == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in pts) / den


def crossovers(data: dict[str, list[dict]]) -> list[str]:
    """Pairs whose ranking swaps between adjacent rungs, both sides passing."""
    found = []
    modes = sorted(data)
    for i, a in enumerate(modes):
        for b in modes[i + 1 :]:
            ms_a = {r["_seq"]: r["_ms"] for r in data[a] if r["_ok"] and r["_ms"]}
            ms_b = {r["_seq"]: r["_ms"] for r in data[b] if r["_ok"] and r["_ms"]}
            shared = sorted(set(ms_a) & set(ms_b))
            for lo, hi in zip(shared, shared[1:]):
                before = ms_a[lo] - ms_b[lo]
                after = ms_a[hi] - ms_b[hi]
                if before == 0 or after == 0 or (before < 0) == (after < 0):
                    continue
                first, second = (a, b) if before < 0 else (b, a)
                found.append(
                    f"{first} leads {second} at seq {lo} "
                    f"({ms_a[lo]:.1f} vs {ms_b[lo]:.1f} ms) and trails it at seq {hi} "
                    f"({ms_a[hi]:.1f} vs {ms_b[hi]:.1f} ms)"
                )
    return found


def render(data: dict[str, list[dict]]) -> str:
    lines = []
    seqs = sorted({r["_seq"] for rows in data.values() for r in rows})

    lines.append("## Latency by sequence length (ms, avg)\n")
    lines.append("| mode | " + " | ".join(str(s) for s in seqs) + " | log-log slope |")
    lines.append("|---" * (len(seqs) + 2) + "|")
    for mode in sorted(data):
        by_seq = {r["_seq"]: r for r in data[mode]}
        cells = []
        for s in seqs:
            r = by_seq.get(s)
            if r is None:
                cells.append("—")
            elif not r["_ok"]:
                cells.append("**FAILED**")
            else:
                cells.append(f"{r['_ms']:.1f}")
        exp = exponent(data[mode])
        cells.append(f"{exp:.2f}" if exp is not None else "n/a")
        lines.append(f"| `{mode}` | " + " | ".join(cells) + " |")

    for key, label in _STRUCT:
        lines.append(f"\n## {label} by sequence length\n")
        lines.append("| mode | " + " | ".join(str(s) for s in seqs) + " |")
        lines.append("|---" * (len(seqs) + 1) + "|")
        for mode in sorted(data):
            by_seq = {r["_seq"]: r for r in data[mode]}
            cells = []
            for s in seqs:
                r = by_seq.get(s)
                cells.append(str(r[key]) if r and r["_ok"] else "—")
            lines.append(f"| `{mode}` | " + " | ".join(cells) + " |")

    failed = [
        f"{m} @ {r['_seq']}: {r['failure_message']}"
        for m, rows in sorted(data.items())
        for r in rows
        if not r["_ok"]
    ]
    lines.append("\n## Failed rungs\n")
    lines.extend([f"- {f}" for f in failed] or ["- none"])

    lines.append("\n## Crossovers\n")
    found = crossovers(data)
    lines.extend(
        [f"- {c}" for c in found] or ["- none: the ranking is the same at every rung"]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root")
    ap.add_argument("--modes", default="coarse,offload,runlist,fused")
    ap.add_argument("--md", default=None, help="also write the report here")
    args = ap.parse_args(argv)

    data = load(args.root, [m for m in args.modes.split(",") if m])
    if not data:
        print(f"[ladder-report] no <mode>.csv found under {args.root}")
        return 1
    report = render(data)
    print(report)
    if args.md:
        with open(args.md, "w", encoding="utf-8") as fh:
            fh.write(report)
        print(f"[ladder-report] wrote {args.md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
