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

THE TOOLCHAIN CONDITION `[2026-08-12]` (queue item 16)
    The other half. ``compare_roots.compare_manifests`` has always diffed a
    ``toolchain`` manifest block, and this module never wrote one, so that half
    of every root comparison compared nothing -- for as long as it existed.

    ``observe_toolchain()`` writes it: XRT version, the mlir-aie and Peano wheel
    pins, and which AIR tree (``build-xrt`` vs ``install-xrt``) the interpreter
    resolves. Declared in ``schema.TOOLCHAIN_FIELDS``, under
    ``schema.TOOLCHAIN_KEY`` -- a name that is not a free choice, since matching
    the key the diff already reads is the whole point.

    Same rules as the conditions block, for the same reasons: the observation is
    the CALLER's, ``build_manifest`` records ``unknown`` rather than probing
    behind its back, the host suite stays hermetic, ``SCHEMA_VERSION`` STAYS 2,
    and a manifest older than the block reads back as ``absent`` rather than
    crashing or quietly matching.

    Where they differ is what ``compare_roots`` does with a mismatch: it REFUSES
    a pmode mismatch and FLAGS a toolchain one. That asymmetry is argued in
    ``compare_roots.compare_toolchain``, not here.

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

AND IT SAYS WHICH SESSION WALKED WHICH RUNG `[2026-08-12]` (resume, item 8)
    Once a walk can be resumed, one CSV can hold rows measured days apart, and
    nothing in a results row says so. ``walk`` is a block from
    ``resume.walk_block()`` recording the sessions, what each measured, and the
    axes they disagree on.

    Its ``attribution_problems`` are merged into ``incomplete_reasons``, which
    is the half that matters: a ledger claiming rungs the CSVs do not hold, or a
    row whose hash no longer matches the session that recorded it, makes a run
    INCOMPLETE rather than merely annotated. The row counts alone cannot see
    either -- they count rows, and both defects leave the count right.

    Omitting ``walk`` records ``absent`` and adds no problems, exactly as
    ``expected_rows=None`` leaves the counts unchecked. That is what keeps every
    root recorded before the ledger existed readable, and it is why
    ``walk_attribution_checked`` is recorded beside the verdict: "attributed"
    and "nobody looked" used to be the same absence, which is M3's shape again.

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


#: Where XRT records its own version. A JSON file the runtime ships, read rather
#: than shelled out to: `xrt-smi --version` is a device-touching binary and this
#: module must not put one on a host-only critical path.
XRT_VERSION_JSON = Path("/opt/xilinx/xrt/version.json")

#: The two AIR trees doc 15 keeps distinct, longest-first so `install-xrt` cannot
#: be matched by a prefix of `build-xrt` or vice versa.
_AIR_TREES = ("install-xrt", "build-xrt")


def _wheel_version(distribution: str) -> tuple[str | None, str | None]:
    """``(version, why_not)`` for an installed wheel, without importing it.

    ``importlib.metadata`` reads the dist-info of whatever the CURRENT
    interpreter would import, which is the point: it reports the wheel that
    actually resolves rather than a path this module guessed. Importing the
    package instead would be both slow and, for these two, capable of failing on
    a machine whose toolchain is exactly the thing in question.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version(distribution), None
    except PackageNotFoundError:
        return None, f"no {distribution} distribution is visible to this interpreter"
    except Exception as exc:  # pragma: no cover - harness breakage, not a task
        return None, f"could not read {distribution} ({type(exc).__name__}: {exc})"


def _xrt_version() -> tuple[str | None, str | None]:
    """``(version, why_not)`` from XRT's own ``version.json``.

    ``BUILD_VERSION+VERSION_HASH[:12]``: the release number alone has been
    identical across rebuilt runtimes, and doc 03's 19-39% ``offload`` swing is
    attributed to an XRT change -- so the build hash is carried too, truncated
    because a full 40-char hash makes the diff line unreadable and the first 12
    already separate every build this host has seen.
    """
    try:
        payload = json.loads(XRT_VERSION_JSON.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, f"{XRT_VERSION_JSON} does not exist on this host"
    except (OSError, ValueError) as exc:
        return None, f"could not read {XRT_VERSION_JSON} ({type(exc).__name__}: {exc})"
    if not isinstance(payload, dict):
        return None, f"{XRT_VERSION_JSON} is not a JSON object"
    build = str(payload.get("BUILD_VERSION") or "").strip()
    digest = str(payload.get("VERSION_HASH") or "").strip()
    if not build:
        return None, f"{XRT_VERSION_JSON} records no BUILD_VERSION"
    return (f"{build}+{digest[:12]}" if digest else build), None


def _air_resolution() -> tuple[str | None, str | None]:
    """Which AIR tree this interpreter would import: build-xrt, install-xrt, ...

    Uses ``find_spec``, which locates a top-level package WITHOUT executing its
    ``__init__``. Importing ``air`` here would be the wrong tool twice over: it
    is slow, and on a host whose bindings are stale it raises -- and a stale
    binding is precisely the condition this field exists to make visible, so the
    probe for it must not be the thing that fails.
    """
    try:
        import importlib.util

        spec = importlib.util.find_spec("air")
    except Exception as exc:
        return None, f"could not locate the air package ({type(exc).__name__}: {exc})"
    if spec is None:
        return None, "no `air` package is visible to this interpreter"
    origin = spec.origin or (list(spec.submodule_search_locations or []) or [None])[0]
    if not origin:
        return None, "the `air` package has no locatable origin"
    for tree in _AIR_TREES:
        if f"/{tree}/" in str(origin):
            return tree, None
    # Verbatim rather than "other": doc 15's whole point is that WHICH tree is
    # resolved decides whether a run saw a compiler fix, and collapsing an
    # unrecognised path to a label throws away the only clue.
    return str(Path(origin).parent), None


def observe_toolchain() -> dict:
    """The toolchain this host would build and dispatch with, best-effort.

    Never raises, and every field degrades to ``unknown`` WITH the reason
    attached -- ``_git``'s rule and ``observe_conditions``' rule, for the same
    reason: a manifest with imperfect provenance still beats no manifest, and an
    `unknown` that does not say why is indistinguishable from one nobody tried
    to fill.

    Stamped ``probed_at_manifest_build`` whenever anything at all was read. There
    is no ``observed`` variant and no caller-supplied override, which is the one
    place this differs from ``observe_conditions``: a pmode can reset between the
    run and the manifest build, so who observed it and when is load-bearing,
    whereas a toolchain changes only when somebody deliberately rebuilds it. If
    that ever stops being true the field's domain (CONDITION_SOURCES) already has
    room for ``observed``.
    """
    block = schema.empty_toolchain()
    reasons = []
    for name, probe in (
        ("xrt_version", _xrt_version),
        ("mlir_aie_version", lambda: _wheel_version("mlir_aie")),
        ("peano_version", lambda: _wheel_version("llvm-aie")),
        ("air_resolution", _air_resolution),
    ):
        try:
            value, why_not = probe()
        except Exception as exc:  # pragma: no cover - a probe must never raise
            value, why_not = None, f"{type(exc).__name__}: {exc}"
        if value:
            block[name] = str(value)
        else:
            reasons.append(f"{name}: {why_not}")

    known = sum(
        1
        for name in schema.TOOLCHAIN_IDENTITY_FIELDNAMES
        if block[name] != schema.UNKNOWN_CONDITION
    )
    if known:
        block["toolchain_source"] = "probed_at_manifest_build"
    block["toolchain_detail"] = (
        f"probed {known}/{len(schema.TOOLCHAIN_IDENTITY_FIELDNAMES)} field(s) "
        f"at manifest build"
        + (f"; not determined -- {'; '.join(reasons)}" if reasons else "")
    )
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
    toolchain=None,
    walk=None,
) -> dict:
    """Describe ``results_root``; ``complete`` means every expected CSV measured.

    ``conditions`` is the measurement condition -- a block from
    ``observe_conditions()`` -- and omitting it records ``unknown`` WITH a
    reason rather than probing the device. See the module docstring for why the
    observation is the caller's; ``manifest.py``'s own CLI probes by default.

    A supplied block is validated, so a hand-assembled dict with a typo'd key
    fails here rather than being written and read back as a silent ``unknown``.

    ``toolchain`` is the build condition -- a block from ``observe_toolchain()``
    -- and behaves identically: omitting it records ``unknown`` with a reason
    rather than probing, and a hand-assembled block is validated so a typo'd key
    fails here rather than being read back as a silent ``unknown``.

    ``expected_rows`` adds the row-count clauses -- see the docstring section.

    ``walk`` is the session attribution -- a block from ``resume.walk_block()``
    -- and omitting it records ``absent`` and checks nothing, which is what
    every root recorded before the ledger existed gets.

    All four kwargs default to None and are therefore back-compatible: a caller
    that passes none gets the pre-2026-08-12 behaviour, an unconditioned
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

    if toolchain is None:
        toolchain = schema.empty_toolchain()
        toolchain["toolchain_detail"] = (
            "no toolchain was supplied to build_manifest. It is NOT assumed: a "
            "comparison against this root is unconditioned on the axis doc 03 "
            "records moving `offload` 19-39% on an XRT change alone. Pass "
            "manifest.observe_toolchain(), or run this module's CLI, which "
            "probes by default."
        )
    schema.validate_toolchain(toolchain)

    walk_checked = walk is not None
    if walk is None:
        walk = schema.empty_walk()
        walk["walk_detail"] = (
            "no walk ledger was supplied to build_manifest. Whether these rows "
            "came from one session is NOT assumed: a resumed walk can hold rows "
            "measured days apart at two toolchains, and nothing in a results "
            "row says so. Pass resume.walk_block()."
        )
    schema.validate_walk(walk)

    problems = smoke_gate.check_results_root(root, expected)
    # Merged rather than reported beside, so one list answers "why is this run
    # not complete" -- a ledger that describes a walk the files do not hold is
    # not an annotation on a finished run, it is an unfinished run.
    problems.extend(walk["attribution_problems"])
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
        # The build condition, and NOT part of `complete` for the conditions
        # block's reason. Written under the key `compare_manifests` has always
        # diffed -- before this, that loop iterated `{}` on both sides and the
        # toolchain half of every comparison compared nothing (queue item 16).
        schema.TOOLCHAIN_KEY: toolchain,
        # Which session walked which rung. NOT part of the conditions block: a
        # pmode is one value for a run, while this is a value PER ROW, and the
        # whole reason it exists is that a resumed run has more than one.
        schema.WALK_KEY: walk,
        "expected_files": files,
        "missing_files": [f["path"] for f in files if not f["exists"]],
        # Recorded so a reader can tell "the counts were met" from "no counts
        # were checked" -- the two used to look identical, which is M3.
        "row_counts_checked": bool(expected_rows),
        # Same distinction one level up: "every row is attributed" and "nobody
        # looked" are different facts and must not read the same.
        "walk_attribution_checked": walk_checked,
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
        "device. For a manifest built on a host that did not do the measuring. "
        "Suppresses the toolchain probe too: both describe THIS host, and a "
        "manifest built elsewhere should claim neither.",
    )
    args = ap.parse_args(argv)

    conditions = None
    toolchain = None
    if not args.no_probe or args.npu_power_mode is not None:
        conditions = observe_conditions(args.npu_power_mode)
    if not args.no_probe:
        # No `--toolchain-*` override to match `--npu-power-mode`: nothing about
        # a toolchain is observable at measurement time and not at manifest
        # build, so an override would only be a way to write a claim by hand.
        toolchain = observe_toolchain()

    manifest = build_manifest(
        args.results_root, args.expect, conditions=conditions, toolchain=toolchain
    )
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

    tools = manifest[schema.TOOLCHAIN_KEY]
    print(
        "[manifest] toolchain: "
        + " ".join(
            f"{name}={tools[name]}" for name in schema.TOOLCHAIN_IDENTITY_FIELDNAMES
        )
        + f" ({tools['toolchain_source']})"
    )
    if any(
        tools[name] == schema.UNKNOWN_CONDITION
        for name in schema.TOOLCHAIN_IDENTITY_FIELDNAMES
    ):
        print(f"[manifest]   {tools['toolchain_detail']}")
    for reason in manifest["incomplete_reasons"]:
        print(f"[manifest]   {reason}")
    return 0 if manifest["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
