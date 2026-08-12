# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The results manifest: what a run produced, and whether it actually measured.

    python3 study/manifest.py <results-root> --expect rel/path.csv ... -o manifest.json

CONTRACT
    ``build_manifest`` describes a results tree -- which expected CSVs are
    present, how big and how recent, the git provenance and the MEASUREMENT
    CONDITION to reproduce it, and a single ``complete`` verdict.
    ``write_manifest`` writes it as sorted JSON so two runs diff cleanly.

THE MEASUREMENT CONDITION `[2026-08-12]` (doc 34 M4)
    A latency is a number measured under conditions, and the NPU power mode is
    the condition that has actually cost a day: it is non-persistent, it resets
    on every reboot and every ``amdxdna`` reload, and at ``Default`` this host
    measures ~15-20x slow (README trap 0, doc 32). Nothing recorded it, so the
    rule "08-10's records are Default-conditional, pre-08-10's are Turbo-
    conditional" had to live in README prose. ``conditions`` puts it in the
    artifact, and ``compare_roots`` refuses to splice a comparison across it.

    ``schema.CONDITION_FIELDS`` declares the block and says why it is a manifest
    block rather than a ``results`` column (a column is a version bump, and a
    bump makes every recorded CSV unreadable -- it already did that to 56 of
    them once).

    **The observation is the CALLER's, not this module's.** ``build_manifest``
    records what it is given and records ``unknown`` otherwise; it does not
    query the device behind the caller's back. Two reasons: the honest one is
    that a manifest is built AFTER a run, so a probe here observes the mode now
    and not the mode the rows were measured at -- a distinction the
    ``npu_power_mode_source`` field exists to keep. The practical one is that
    this module is imported by a host-only test suite that must stay hermetic
    and under a second, and a device query in ``build_manifest`` would put
    ``xrt-smi`` on its critical path. ``observe_conditions()`` is the one-line
    probe for a caller that wants it, and the CLI runs it by default.

    Not recorded, and named so it is not mistaken for an oversight:
    ``xrt_version`` and the LLVM/mlir-aie/Peano pin. Doc 34 M4 asks for those
    too and doc 03 records an XRT version change alone moving ``offload``
    19-39%. They are a declaration in ``schema.CONDITION_FIELDS`` away. Note
    while you are there that ``compare_roots.compare_manifests`` diffs a
    ``toolchain`` block that this module has NEVER written -- the diff loop has
    been iterating an empty key since it was written.

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
    - **An `unknown` power mode is not a synonym for turbo.** Every root
      recorded before `[2026-08-12]` has no conditions block and CANNOT be
      stamped after the fact -- the mode is not recoverable from the files. Do
      not backfill one to clear a warning; re-walk, or read the comparison
      knowing it is unconditioned. Same call, and the same reasoning, as the
      throughput floor's `unknown` in ``port-loop/pmode_guard.py``.
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


def observe_conditions(mode: str | None = None) -> dict:
    """The measurement condition as this host reports it, best-effort.

    ``mode`` supplied  -- stamp it as ``observed``: the caller watched the mode
                          at measurement time, which is the only source whose
                          clock is the measurement's clock. Mirrors what
                          ``seed-throughput-baseline.sh`` does for the floor.
    ``mode`` omitted   -- probe the device now and stamp
                          ``probed_at_manifest_build``, which is a weaker claim
                          and says so.

    NO SECOND PARSER, and never raises. Detection is ``registry_sweep``'s
    ``npu_power_mode()`` -- the one implementation ``run_mode.py``,
    ``component_groups.py`` and ``port-loop/pmode_guard.py`` all already use.
    Two parsers of one device's output disagree eventually, and here the
    disagreement would show up as a comparison being refused or allowed. An
    import failure, a missing ``xrt-smi`` or an unparsable report all degrade to
    ``unknown`` WITH THE REASON ATTACHED, exactly as ``_git`` degrades: a
    manifest with imperfect provenance still beats no manifest. That is the
    opposite of ``pmode_guard``'s fail-closed, and deliberately so -- this
    module RECORDS a condition and must not refuse to describe a tree, while
    that one GATES on it and must not let an unknown pass as a green.
    """
    block = schema.empty_conditions()
    block["observed_at_utc"] = datetime.now(timezone.utc).isoformat()

    if mode is not None:
        block["npu_power_mode"] = schema.normalise_power_mode(mode)
        block["npu_power_mode_source"] = "observed"
        block["npu_power_mode_detail"] = "supplied by the caller that measured"
        return block

    try:
        example = Path(_HERE).parent
        if str(example) not in sys.path:
            sys.path.insert(0, str(example))
        from sweep.registry_sweep import npu_power_mode

        detected, detail = npu_power_mode()
    except Exception as exc:  # pragma: no cover - harness breakage, not a task
        block["npu_power_mode_detail"] = (
            f"could not run the shared power-mode check "
            f"({type(exc).__name__}: {exc})"
        )
        return block

    if detected is None:
        block["npu_power_mode_detail"] = detail
        return block
    block["npu_power_mode"] = schema.normalise_power_mode(detected)
    block["npu_power_mode_source"] = "probed_at_manifest_build"
    block["npu_power_mode_detail"] = detail
    return block


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
    results_root,
    expected: list[str],
    repo=None,
    conditions=None,
    expected_rows: dict | None = None,
) -> dict:
    """Describe ``results_root``; ``complete`` means every expected CSV measured.

    ``conditions`` is the measurement condition -- a block from
    ``observe_conditions()`` -- and omitting it records ``unknown`` WITH a
    reason rather than probing the device. See the module docstring for why the
    observation is the caller's; ``manifest.py``'s own CLI probes by default.

    A supplied block is validated, so a hand-assembled dict with a typo'd key
    fails here rather than being written and read back as a silent ``unknown``.

    ``expected_rows`` adds the row-count clauses -- see the docstring section.

    Both kwargs default to None and are therefore back-compatible: a caller
    that passes neither gets the pre-2026-08-12 behaviour, an unconditioned
    manifest whose completeness is judged on files alone.
    """
    root = Path(results_root)
    repo = Path(repo) if repo else _repo_root()
    expected_rows = expected_rows or {}

    if conditions is None:
        conditions = schema.empty_conditions()
        conditions["npu_power_mode_detail"] = (
            "no measurement condition was supplied to build_manifest. The mode "
            "is NOT assumed: a comparison against this root is unconditioned on "
            "the axis that costs 15-20x (README trap 0). Pass "
            "manifest.observe_conditions(), or run this module's CLI, which "
            "probes by default."
        )
    schema.validate_conditions(conditions)

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
        # The measurement condition. NOT part of `complete`: a tree that
        # measured everything it was asked to did so, whatever mode it ran at.
        # Whether two such trees may be COMPARED is compare_roots' question, and
        # keeping the two verdicts apart is what stops "unknown pmode" from
        # making an otherwise finished run report as unfinished.
        schema.CONDITIONS_KEY: conditions,
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
    ap.add_argument(
        "--npu-power-mode",
        default=None,
        metavar="MODE",
        help="the power mode the run was MEASURED at, if you observed it "
        "(e.g. turbo). Stamped as `observed`, the strongest source. Without "
        "it the device is probed now and stamped "
        "`probed_at_manifest_build`, which is a weaker claim: a manifest is "
        "built after a run and a driver reload in between resets the mode.",
    )
    ap.add_argument(
        "--no-probe",
        action="store_true",
        help="record the condition as `unknown` rather than querying the "
        "device. For a manifest built on a host that did not do the measuring.",
    )
    args = ap.parse_args(argv)

    conditions = None
    if not args.no_probe or args.npu_power_mode is not None:
        conditions = observe_conditions(args.npu_power_mode)

    manifest = build_manifest(args.results_root, args.expect, conditions=conditions)
    write_manifest(args.output, manifest)

    print(f"[manifest] wrote {args.output}")
    print(f"[manifest] complete: {manifest['complete']}")
    block = manifest[schema.CONDITIONS_KEY]
    print(
        f"[manifest] npu_power_mode: {block['npu_power_mode']} "
        f"({block['npu_power_mode_source']})"
    )
    if block["npu_power_mode"] == schema.UNKNOWN_CONDITION:
        # Loud, and never fatal: an unrecorded condition does not make a tree
        # that measured everything incomplete. compare_roots decides what an
        # unknown means for a COMPARISON; this only refuses to be quiet.
        print(f"[manifest]   {block['npu_power_mode_detail']}")
    for reason in manifest["incomplete_reasons"]:
        print(f"[manifest]   {reason}")
    return 0 if manifest["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
