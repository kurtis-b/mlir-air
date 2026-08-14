# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read and write current-schema result CSVs. The mechanism to ``schema.py``'s catalogue.

CONTRACT
    ``write_rows(path, rows, table)`` writes a CSV whose header is exactly that
    table's field order, validating every row first. ``read_rows(path, table)``
    reads one back, checking the header matches and the schema version is this
    one. Nothing here decides what a column MEANS -- that is ``schema.py``, and
    keeping the split is what stops a writer inventing a column.

WHY VALIDATE ON WRITE RATHER THAN ON READ
    A malformed row is cheap to fix at the moment it is produced and expensive
    to diagnose weeks later out of a results tree. ``validate_row`` costs
    microseconds against measurements that cost seconds.

FOOTGUNS
    - **A failed measurement still writes a COMPLETE row**, every metric
      ``None``. That is required, not tolerated: Phase F's gate distinguishes
      "no passed row" from "no rows", and it cannot do that if failures write
      nothing. ``write_rows`` therefore refuses an empty ``rows`` list rather
      than creating a header-only file that reads as "ran and measured nothing".
    - **CSV has no None.** ``None`` is written as the empty string and read back
      as ``None``; a genuinely empty string in a text field is indistinguishable
      from unset, which is why no field's meaning depends on that difference.
    - Numeric fields come back as **strings**. Callers that compare numerically
      must convert; the gate below only tests ``run_status`` and reports
      ``failure_message``, both text, so it does not.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import schema  # noqa: E402


def write_rows(path: str | Path, rows: list[dict], table: str = "results") -> Path:
    """Write ``rows`` as a schema CSV. Validates each row; refuses an empty list."""
    if not rows:
        raise ValueError(
            f"refusing to write {path} with no rows: a header-only CSV reads as "
            "'ran and measured nothing', which is exactly what the smoke gate "
            "must be able to distinguish. A failed measurement writes a "
            "complete row with run_status=failed."
        )
    fieldnames = [f.name for f in schema.fields_for(table)]
    for i, row in enumerate(rows):
        try:
            schema.validate_row(row, table)
        except ValueError as e:
            raise ValueError(f"row {i} is not writable: {e}") from None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
    return path


def read_rows(path: str | Path, table: str = "results") -> list[dict]:
    """Read a schema CSV back, with ``None`` restored for empty cells."""
    expected = [f.name for f in schema.fields_for(table)]
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        if header != expected:
            missing = sorted(set(expected) - set(header))
            extra = sorted(set(header) - set(expected))
            raise ValueError(
                f"{path} header does not match schema v{schema.SCHEMA_VERSION} "
                f"{table} (missing={missing}, extra={extra}). Read it through "
                "iron_adapter if it came from iron."
            )
        rows = [{k: (v if v != "" else None) for k, v in r.items()} for r in reader]

    # Restore schema_version as an int so read -> write round-trips. csv gives
    # back the string "1", and validate_row compares against the int 1, so
    # without this a row read from a results CSV cannot be written to another
    # one -- it fails with "use the adapter", pointing at iron, for a file this
    # module wrote itself. Found by run_ladder, which reads each rung's row from
    # a child process and rewrites them as one CSV per mode.
    #
    # Only this field is coerced. The measurement columns stay strings because
    # the schema declares meanings and timing boundaries, not types, and
    # inventing a type per column here would put that decision in the wrong
    # module. Callers doing arithmetic convert explicitly -- see ladder_report.
    for row in rows:
        if row.get("schema_version") is not None:
            try:
                row["schema_version"] = int(row["schema_version"])
            except (TypeError, ValueError):
                pass  # leave it; the check below reports it far better

    for i, row in enumerate(rows):
        version = row.get("schema_version")
        if str(version) != str(schema.SCHEMA_VERSION):
            raise ValueError(
                f"{path} row {i} carries schema_version {version!r}; this module "
                f"is v{schema.SCHEMA_VERSION}"
            )
    return rows


def read_rows_compatible(path: str | Path, table: str = "results") -> list[dict]:
    """Read a CSV written by THIS schema or an older one. Analysis tier only.

    ``read_rows`` is exact by design and every writer must stay on it: a
    measurement that cannot be written at the current version is a bug, and
    accepting a near-miss header is how a column silently changes meaning.

    Reading an ARCHIVED tree is the other case. ``schema.py`` states that v2's
    five columns "are appended AFTER all of them, so a v1-shaped reader keeps
    working column for column" -- a property deliberately engineered and, until
    this function, never used. Fifteen of this project's nineteen result trees
    are v1, including the nine-rung ladder and doc 30's coarse-cell result, and
    they are not re-measurable at will.

    So the rule is a STRICT PREFIX and nothing looser: the file's header must be
    exactly the first N names of the current table, in order. Columns the file
    predates are filled with ``None`` -- which every caller must then treat as
    "not measured", never as zero. A reordered header, an unknown column, or a
    version NEWER than this module still raises, unchanged.

    FOOTGUN
        The ``None`` fill is indistinguishable from a v2 row whose measurement
        failed, and that is intentional -- both mean "no number here". A caller
        that needs to tell an old tree from a failed measurement must read
        ``schema_version``, not the absent column.
    """
    expected = [f.name for f in schema.fields_for(table)]
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        if header != expected[: len(header)]:
            missing = sorted(set(expected) - set(header))
            extra = sorted(set(header) - set(expected))
            raise ValueError(
                f"{path} header is not a prefix of schema v{schema.SCHEMA_VERSION} "
                f"{table} (missing={missing}, extra={extra}). Only columns "
                "appended at the end are readable across versions; a reordered "
                "or foreign header is not. Read it through iron_adapter if it "
                "came from iron."
            )
        absent = expected[len(header) :]
        rows = []
        for record in reader:
            row = {k: (v if v != "" else None) for k, v in record.items()}
            row.update({name: None for name in absent})
            rows.append(row)

    for row in rows:
        if row.get("schema_version") is not None:
            try:
                row["schema_version"] = int(row["schema_version"])
            except (TypeError, ValueError):
                pass

    for i, row in enumerate(rows):
        version = row.get("schema_version")
        try:
            numeric = int(version)
        except (TypeError, ValueError):
            raise ValueError(
                f"{path} row {i} carries an unreadable schema_version {version!r}"
            ) from None
        if numeric > schema.SCHEMA_VERSION:
            raise ValueError(
                f"{path} row {i} carries schema_version {numeric}, NEWER than "
                f"this module's v{schema.SCHEMA_VERSION}. Reading it here would "
                "drop columns this code does not know about."
            )
    return rows
