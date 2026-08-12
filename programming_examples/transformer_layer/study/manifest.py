# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The results manifest: what a run produced, and whether it actually measured.

    python3 study/manifest.py <results-root> --expect rel/path.csv ... -o manifest.json

CONTRACT
    ``build_manifest`` describes a results tree -- which expected CSVs are
    present, how big and how recent, the toolchain and git provenance to
    reproduce it, and a single ``complete`` verdict. ``write_manifest`` writes it
    as sorted JSON so two runs diff cleanly.

COMPLETENESS MEANS MEASURED, NOT PRESENT -- the one deliberate departure
    iron's ``results_manifest.py`` defines ``"complete": not missing_files``, so
    a tree whose every measurement failed is "complete" as long as the files
    exist. That is the same hole doc 09 records in its smoke test, which
    "reported 21/21 passed on an environment where every measurement had
    failed", and Phase G's gate -- "a full profile run completes with a complete
    results_manifest.json" -- would inherit it verbatim.

    So ``complete`` here is ``files present AND every one of them has a row with
    run_status=passed``, delegated to ``smoke_gate.check_results_root`` rather
    than reimplemented. The manifest carries the gate's problem list under
    ``incomplete_reasons``, so "not complete" always says why.

AND IT VALIDATES ROWS, NOT ONLY FILES
    That rule alone is still a FILE rule wearing a row's clothes: one passed row
    satisfies it, so a CSV that should hold nine rungs and holds one reported
    ``complete: True``. Doc 10's gate asks for "no missing files **or rows** --
    against counts derived from the profile", and ``expected_rows`` is that.

    Pass ``expected_rows={rel: {"rows": N, "measured": M, "skipped": K}}`` --
    ``study/profiles.py`` derives it from the profile, so no number is typed by
    hand and retargeting a profile retargets its gate. Three clauses, each its
    own reason line:

      rows      every expected row exists. Catches the truncated walk.
      measured  every rung that was meant to be attempted PASSED. Catches the
                walk that ran and regressed.
      skipped   every structurally inapplicable rung is recorded ``skipped``.
                This is the clause that needs ``run_status="skipped"`` to be
                emitted at all: without it an inapplicable rung lands as
                ``failed``, trips two clauses, and reads as a regression.

    Omitting ``expected_rows`` leaves every count key ``None`` and the verdict
    exactly as it was, so an existing caller is unaffected -- but a caller that
    omits it is back to file-level evidence and should say why.

FOOTGUNS
    - **``expected`` is the contract**, exactly as in ``smoke_gate``. A manifest
      built with no expected files is marked incomplete rather than trivially
      complete -- describing an empty tree as a finished run is the failure this
      module exists to avoid.
    - Git provenance is best-effort: a tree with no git, or a dirty one, still
      produces a manifest, with ``dirty: true`` or ``sha: null``. A run is worth
      recording even when its provenance is imperfect; silently omitting the
      field would hide that.
    - ``created_at_utc`` makes two manifests of the same tree differ. Diff on
      the rest, or drop that key first.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import results_io  # noqa: E402
import schema  # noqa: E402
import smoke_gate  # noqa: E402

#: The three count clauses, as (expectation key, matching predicate, label).
#: ``rows`` counts everything; the other two count a run_status.
_ROW_CLAUSES = (
    ("rows", None, "row(s)"),
    ("measured", "passed", "passed row(s)"),
    ("skipped", "skipped", "skipped row(s)"),
)


def _git(repo: Path) -> dict:
    """Best-effort git provenance; never raises."""

    def run(*args):
        try:
            out = subprocess.run(
                ["git", "-C", str(repo), *args],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None

    sha = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "sha": sha,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def _system() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def _file_record(root: Path, rel: str) -> dict:
    path = root / rel
    exists = path.is_file()
    return {
        "path": rel,
        "exists": exists,
        "bytes": path.stat().st_size if exists else None,
        "mtime_utc": (
            datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
            if exists
            else None
        ),
    }


def _observed_status_counts(path: Path) -> dict[str, int] | None:
    """Rows by ``run_status``, or ``None`` if the file cannot be read as one.

    Unreadable is ``None`` rather than an empty tally: ``smoke_gate`` already
    reports a missing or foreign CSV with the better message, and a tally of
    zeros here would read as "it was read and it was empty", which is a
    different fact.
    """
    try:
        rows = results_io.read_rows(path)
    except Exception:
        return None
    counts = {status: 0 for status in schema.RUN_STATUSES}
    counts["total"] = len(rows)
    for row in rows:
        status = str(row.get("run_status") or "")
        if status in counts:
            counts[status] += 1
    return counts


def _row_count_problems(
    rel: str, expectation: dict, observed: dict | None
) -> list[str]:
    """The count clauses for one file. Empty means every declared count is met."""
    if observed is None:
        # Nothing to say that smoke_gate has not already said better.
        return []
    problems = []
    for key, status, label in _ROW_CLAUSES:
        if key not in expectation:
            continue
        want = int(expectation[key])
        got = observed["total"] if status is None else observed[status]
        if got == want:
            continue
        detail = (
            f"{rel}: expected {want} {label}, found {got} "
            f"(passed {observed['passed']}, failed {observed['failed']}, "
            f"skipped {observed['skipped']} of {observed['total']})"
        )
        if key == "skipped" and got < want:
            detail += (
                ". A structurally inapplicable rung must be recorded "
                "run_status=skipped; recorded as failed it is indistinguishable "
                "from a regression"
            )
        problems.append(detail)
    return problems


def _repo_root() -> Path:
    """Ask git, rather than counting directories up from this file.

    Counting is wrong twice over: it was off by one for the normal layout, and
    even corrected it names the wrong tree when this module runs from a git
    WORKTREE, where the checkout root is not the main repository. ``git
    rev-parse --show-toplevel`` is worktree-correct by construction. Falls back
    to the directory arithmetic only if git is unavailable, because a manifest
    with imperfect provenance still beats no manifest.
    """
    try:
        out = subprocess.run(
            ["git", "-C", _HERE, "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return Path(out.stdout.strip())
    except Exception:
        pass
    return Path(_HERE).resolve().parents[2]


def build_manifest(
    results_root, expected: list[str], repo=None, expected_rows: dict | None = None
) -> dict:
    """Describe ``results_root``; ``complete`` means every expected CSV measured.

    ``expected_rows`` adds the row-count clauses -- see the docstring section.
    """
    root = Path(results_root)
    repo = Path(repo) if repo else _repo_root()
    expected_rows = expected_rows or {}

    problems = smoke_gate.check_results_root(root, expected)
    files = [_file_record(root, rel) for rel in expected]
    for record in files:
        rel = record["path"]
        expectation = expected_rows.get(rel)
        observed = _observed_status_counts(root / rel) if record["exists"] else None
        record["expected_rows"] = expectation
        record["observed_rows"] = observed
        if expectation:
            problems.extend(_row_count_problems(rel, expectation, observed))
    unexpected = sorted(set(expected_rows) - {r["path"] for r in files})
    if unexpected:
        # A count for a file nobody expects is a profile/gate disagreement, and
        # silently ignoring it is how a gate ends up validating three of four.
        problems.append(
            f"row counts were given for file(s) that are not in the expected "
            f"list: {unexpected}. The expected list is the contract."
        )
    return {
        "study_id": "transformer_layer_results_manifest",
        "schema_version": schema.SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_root": str(root),
        "repo_root": str(repo),
        "git": _git(repo),
        "system": _system(),
        "expected_files": files,
        "missing_files": [f["path"] for f in files if not f["exists"]],
        # Recorded so a reader can tell "the counts were met" from "no counts
        # were checked" -- the two used to look identical, which is M3.
        "row_counts_checked": bool(expected_rows),
        # NOT `not missing_files` -- see the module docstring.
        "complete": not problems,
        "incomplete_reasons": problems,
    }


def write_manifest(output, manifest: dict) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results_root")
    ap.add_argument("--expect", action="append", default=[], metavar="REL/PATH.csv")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args(argv)

    manifest = build_manifest(args.results_root, args.expect)
    write_manifest(args.output, manifest)

    print(f"[manifest] wrote {args.output}")
    print(f"[manifest] complete: {manifest['complete']}")
    for reason in manifest["incomplete_reasons"]:
        print(f"[manifest]   {reason}")
    return 0 if manifest["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
