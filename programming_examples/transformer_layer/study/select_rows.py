# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pick rows out of a results tree, once, so every report agrees what counts.

    python3 study/select_rows.py results/ --mode coarse --seq 1024

WHY NOT ``select.py``, WHICH IS WHAT IRON CALLS IT
    ``select`` is a STDLIB MODULE. This directory goes on ``sys.path`` (every
    module here starts by putting it there), so a file named ``select.py``
    shadows it for anything that imports it afterwards -- and, worse, loses to
    it when the stdlib one is already in ``sys.modules``, which is how this was
    found: ``import select`` returned the stdlib module and every attribute
    lookup failed with a message that says nothing about shadowing. iron avoids
    it only by importing package-relative, which this tier deliberately does
    not. The name is the fix; see the same trap in the repository's
    ``PYTHONPATH`` note.

CONTRACT
    ``load_tree(root)`` reads every schema results CSV under a root.
    ``select(rows, ...)`` filters them by family, variant, sequence length,
    mode and status, and returns them in matrix order -- family in
    ``cases.FAMILY_IDS`` order, then mode, then the ladder -- so two reports
    over one tree list their rows the same way. ``numeric(row, field)`` is the
    one place a CSV string becomes a number.

WHY THIS IS A MODULE AND NOT THREE PRIVATE HELPERS
    ``ladder_report``, ``compare_roots`` and any future report each need "the
    rows that count", and three answers to that question is three different
    tables from one tree. iron hit this: its ``select.py`` defines
    ``_eligible_row`` and its comparator re-derives eligibility separately, so
    a row can be in one report and not the other with nothing saying why.

    The eligibility rule is ONE predicate here, and it is deliberately weak:
    a row is eligible if it parses as this schema and its ``execution_mode``
    resolves. **Passing is NOT part of eligibility** -- ``status="all"`` is the
    default, because a failed rung is a measurement and hiding it is how a
    ladder comes to look complete. Reports that need passing rows ask for them.

FOOTGUNS
    - **Sorting is by the matrix, not by the file.** A tree missing a family
      still lists what it has in matrix order; a family the matrix does not know
      sorts last rather than raising, because a results tree from an older run
      is worth reading.
    - **Numeric fields come back as strings** from ``results_io``, by design
      (its docstring says why). ``numeric`` returns ``None`` rather than raising
      on an empty or malformed cell -- a failed row has empty metrics and that
      is valid.
    - **``load_tree`` skips a CSV it cannot read as this schema, and SAYS SO.**
      A tree usually holds derived and foreign files beside the results; a
      loader that raised on the first one would be unusable, and one that
      skipped silently would quietly drop a real results file whose schema
      version moved. The skipped list is returned, not printed and discarded.
    - **A results tree is not a results file.** ``load_tree`` globs ``*.csv``
      one level deep and in the root; it does not recurse without limit, so a
      nested archive directory does not silently join the comparison.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import cases  # noqa: E402
import results_io  # noqa: E402
import schema  # noqa: E402


def numeric(row: dict, field: str) -> float | None:
    """A results cell as a float, or ``None`` for empty or malformed."""
    value = row.get(field)
    if value is None:
        return None
    text = str(value).strip()
    if text in ("", "None", "nan", "NaN"):
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed if parsed == parsed else None


def integer(row: dict, field: str) -> int | None:
    """A results cell as an int, via ``numeric`` so ``'1024.0'`` still works."""
    value = numeric(row, field)
    return None if value is None else int(value)


def is_eligible(row: dict) -> bool:
    """Whether a row is a measurement this study can talk about.

    Weak on purpose -- see the module docstring. It checks identity, not
    outcome: a failed rung is eligible and reports decide whether to show it.
    """
    if str(row.get("execution_mode") or "") not in schema.EXECUTION_MODES:
        return False
    return integer(row, "seq_len") is not None


def _sort_key(row: dict) -> tuple[int, int, int]:
    family = str(row.get("study_case_id") or "")
    mode = str(row.get("execution_mode") or "")
    seq = integer(row, "seq_len")
    family_order = (
        cases.FAMILY_IDS.index(family)
        if family in cases.FAMILY_IDS
        else len(cases.FAMILY_IDS)
    )
    mode_order = (
        schema.EXECUTION_MODES.index(mode)
        if mode in schema.EXECUTION_MODES
        else len(schema.EXECUTION_MODES)
    )
    seq_order = (
        cases.SEQUENCE_LADDER.index(seq)
        if seq in cases.SEQUENCE_LADDER
        else len(cases.SEQUENCE_LADDER)
    )
    return (family_order, mode_order, seq_order)


def select(
    rows: list[dict],
    *,
    family: str = "all",
    workload_variant: str = "all",
    seq_len: int | str = "all",
    mode: str = "all",
    status: str = "all",
) -> list[dict]:
    """The eligible rows matching every filter, in matrix order.

    ``mode`` accepts either side of convention 7's mapping and compares on the
    CSV value, so ``--mode coarse`` finds the rows written as ``hybrid``. That
    is the single most common way to select nothing by accident.
    """
    wanted_mode = None if mode == "all" else cases.canonical_execution_mode(mode)
    wanted_variant = (
        None
        if workload_variant == "all"
        else cases.canonical_workload_variant(workload_variant)
    )
    wanted_seq = None if seq_len == "all" else int(seq_len)

    picked = []
    for row in rows:
        if not is_eligible(row):
            continue
        if family != "all" and str(row.get("study_case_id") or "") != family:
            continue
        if wanted_variant is not None:
            if str(row.get("workload_variant") or "") != wanted_variant:
                continue
        if wanted_seq is not None and integer(row, "seq_len") != wanted_seq:
            continue
        if wanted_mode is not None:
            if str(row.get("execution_mode") or "") != wanted_mode:
                continue
        if status != "all" and str(row.get("run_status") or "") != status:
            continue
        picked.append(row)
    return sorted(picked, key=_sort_key)


def load_tree(root: str | Path) -> tuple[list[dict], list[str]]:
    """(rows, skipped) for every readable results CSV under ``root``.

    Returns the skipped files rather than printing them: a caller deciding
    whether a comparison is trustworthy needs to know a file was passed over,
    and a message on stdout is not something a caller can check.
    """
    root = Path(root)
    paths = sorted(set(root.glob("*.csv")) | set(root.glob("*/*.csv")))
    rows: list[dict] = []
    skipped: list[str] = []
    for path in paths:
        try:
            rows.extend(results_io.read_rows(path))
        except Exception as e:
            skipped.append(f"{path.relative_to(root)}: {e}")
    return rows, skipped


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root")
    ap.add_argument("--family", default="all")
    ap.add_argument("--variant", default="all")
    ap.add_argument("--seq", default="all")
    ap.add_argument("--mode", default="all", help="code name or CSV value")
    ap.add_argument("--status", default="all", choices=("all",) + schema.RUN_STATUSES)
    args = ap.parse_args(argv)

    rows, skipped = load_tree(args.root)
    picked = select(
        rows,
        family=args.family,
        workload_variant=args.variant,
        seq_len=args.seq,
        mode=args.mode,
        status=args.status,
    )
    for line in skipped:
        print(f"[select] skipped {line}")
    print(f"[select] {len(picked)} of {len(rows)} row(s)")
    for row in picked:
        latency = numeric(row, "avg_latency_ms")
        print(
            f"  {row['study_case_id']:<22} {row['execution_mode']:<10} "
            f"seq {row['seq_len']:<6} {row['run_status']:<7} "
            + ("—" if latency is None else f"{latency:.3f} ms")
        )
    return 0 if picked else 1


if __name__ == "__main__":
    sys.exit(main())
