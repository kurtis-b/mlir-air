# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Measure tile configurations for the case matrix's projection GEMMs and record
the winners in ``kernel_registry``.

``registry_lookup.gemm_config()`` RAISES on an unmeasured ``(M, K, N)`` rather
than guessing, deliberately -- hand-copied tile configs previously caused drift
and stale-config bugs. This is the tool that makes a shape resolvable, and the
only sanctioned way to do it.

CONTRACT

    python3 registry_sweep.py --family baseline_768 --list
        Every shape the family needs, as JSON. No hardware.

    python3 registry_sweep.py --family baseline_768 --plan
        Every (shape, candidate) that would be measured. No hardware.

    python3 registry_sweep.py --family baseline_768
        Measure. Resumable and checkpointed per candidate. NEEDS AN NPU.

    python3 registry_sweep.py --family baseline_768 --write-registry
        Fold the measured results into the registry JSON and both markdown
        pages. Append-only; raises on a shape already registered. No hardware.
        A ``--role`` / ``--seq`` filter narrows what is APPENDED to the JSON.
        It does not narrow the markdown: that section is replaced whole, so it
        is always rebuilt from every shape of the family (see
        ``sweep_report.write_family_markdown``).

    python3 registry_sweep.py --family baseline_768 --verify-resolution
        Assert every shape in the family resolves THROUGH THE OPERATOR BUILDER
        THAT OWNS IT -- not through the generic lookup, which answers a weaker
        question (see ``sweep_report.verify_resolution``). No hardware; this is
        what the lit test runs.

    ``--role`` / ``--seq`` / ``--shape`` narrow any of the above.
    ``--remeasure-registered`` measures a shape that is already registered,
    which is how a registered shape's MISSING METHOD gets a candidate; the
    write stays append-only either way.

APPEND ONLY
    The shapes already in the registry are what the ten shipped LLM deployments
    resolve against. Re-measuring one into a different winner would change their
    behaviour without anyone asking. ``registry_writer`` refuses, and this tool
    skips an already-registered shape rather than queueing it unless
    ``--remeasure-registered`` says otherwise -- and even then, measuring only
    fills the results directory; the write is where the rule is enforced.

    The one edit permitted to an existing entry is ADDING A METHOD it has no row
    for, through ``registry_writer.add_missing_methods``, and only to an entry
    whose ``used_by`` says this sweep wrote it. A shape can be registered and
    still unbuildable by the operator that needs it -- ``64x768x2304`` was
    registered with ``drain`` and ``direct`` only, and ``qkv_proj`` pins
    ``fused-cast`` -- so "the shape is present" is not the same claim as "the
    operator resolves". ``--verify-resolution`` is what tells the two apart.

LOCKING -- READ THIS BEFORE ADDING A flock ANYWHERE
    This tool takes NO lock. Wrap the whole invocation instead:

        flock -x -w 1800 /tmp/mlir-air-npu.lock python3 registry_sweep.py ...

    Taking ``/tmp/mlir-air-npu.lock`` inside the tool as well would DEADLOCK
    against that wrapper: both are BSD ``flock(2)``, which treats a second
    ``open()`` of the same inode as a foreign lock. And never take
    ``/tmp/npu.lock`` -- ``XRTRunner`` holds that one across binary load and
    dispatch inside each candidate subprocess, so taking it here would deadlock
    against this tool's own children.

WHY EACH CANDIDATE IS ITS OWN PROCESS
    Two reasons, and the second is the one that bites. The AIR/MLIR context and
    the XRT device state accumulate across builds in a way that makes a
    long-running parent's later measurements not comparable with its earlier
    ones. And a candidate that hangs the NPU takes the process down with it; a
    subprocess can be killed on a timeout and recorded as ``failed_timeout``,
    where an in-process hang ends the sweep. iron's equivalent has no timeout at
    all, and a hang there is unrecoverable without external intervention.

RESUME, AND WHAT INVALIDATES IT
    Each candidate's verdict is checkpointed to its own file under
    ``results/<family>/`` the moment it is reached, so a kill mid-sweep loses one
    candidate rather than the run. The filename carries a signature over the
    shape, the full candidate configuration AND ``MEASUREMENT_CONTRACT`` -- so
    editing a candidate, or changing what a measurement means, invalidates reuse
    instead of silently keeping a stale winner.

    ``failed_build`` and ``failed_precision`` verdicts ARE reused. Both are
    deterministic: placement is a property of the configuration, and the inputs
    are regenerated from a fixed seed. Re-running them would spend a full aircc
    compile per resume to re-derive an answer already on disk. ``failed_timeout``
    and ``failed_no_comparison`` are NOT reused -- those are statements about the
    machine, not about the candidate. (iron retries every failure forever, which
    is why its FFN sweep never converged.)

TURBO IS ENFORCED, FAIL-CLOSED
    A non-turbo measurement is not comparable with a turbo one, and silently
    mixing them corrupts the registry permanently -- the numbers stay in the file
    long after the run that produced them is forgotten. The power mode is checked
    before EVERY candidate, and anything other than a positively-identified
    ``turbo`` aborts the run. Undetectable is a failure, not a pass: if
    ``xrt-smi`` is missing or its output cannot be parsed, the run stops.

FOOTGUNS
    - Each candidate runs in its own scratch directory under ``--workdir``,
      because ``mm.o`` is rebuilt per candidate with that candidate's tile in its
      ``-DDIM_*`` flags and aiecc searches the CURRENT directory for it. Sharing
      one directory does not fail to link, it links the wrong microkernel.
    - The scratch directories are deleted after each candidate. A completed
      sweep of this family writes ~3 GB of aircc artifacts otherwise.
    - ``--write-registry`` is a separate invocation from the sweep on purpose:
      the sweep is hours long and the registry write is seconds, and coupling
      them would mean a crash in the write costs the measurements.

WHERE THE REST OF IT LIVES
    This module is the ORCHESTRATOR: fan candidates out to subprocesses,
    checkpoint each verdict, resume, elect winners, and the CLI over all of it.
    Its neighbours in this package each own one other thing, and the split is
    porting convention 5's ~800-line cap made concrete:

        sweep_families.py  which shapes and candidates exist at all
        sweep_measure.py   ONE candidate: build it, check it, time it. The
                           process this module forks per candidate.
        sweep_report.py    what is done with a finished sweep -- the resolution
                           assertion and the family markdown
        registry_writer.py the append-only registry JSON write

    ``sweep_report`` imports two functions from here, so this module imports it
    inside ``main()`` rather than at module scope. The dependency has a direction
    and the CLI is the only place that needs both halves.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_EXAMPLE_DIR = _HERE.parent
_PROJ_ROOT = _EXAMPLE_DIR.parent
# _EXAMPLE_DIR ahead of _PROJ_ROOT so `builders` is this example's package.
# PYTHONPATH here ends in a colon, which puts the CWD on sys.path, so the name
# is resolvable to whatever `builders/` the caller happened to stand in.
for _p in (str(_PROJ_ROOT), str(_EXAMPLE_DIR), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import registry_writer  # noqa: E402
from sweep_families import (  # noqa: E402
    FAMILY_HIDDEN,
    METHODS,
    ROLES,
    SEQ_LADDER,
    candidates_for_shape,
    shapes_for_family,
)

# Bump when what a measurement MEANS changes -- the tolerance rule, the input
# generation, the timing method. It is part of every result's signature, so a
# bump invalidates every checkpoint rather than mixing two contracts in one
# registry row.
MEASUREMENT_CONTRACT = "2026-08-c4-v1"

DEFAULT_RESULTS = _HERE / "results"
DEFAULT_WORKDIR = _HERE / "work"
DEFAULT_PERF_ITERS = 20
DEFAULT_TIMEOUT_S = 900

# Verdicts that describe the candidate (deterministic, safe to reuse) versus the
# machine (must be re-run). See the resume section of the module docstring.
REUSABLE_STATUSES = {"passed", "failed_build", "failed_precision", "failed_tier"}


class TurboNotEnforced(RuntimeError):
    """The NPU power mode is not, or cannot be shown to be, turbo."""


def npu_power_mode():
    """(mode, detail). ``mode`` is None when it could not be determined."""
    xrt_smi = shutil.which("xrt-smi")
    if not xrt_smi:
        return None, "xrt-smi is not on PATH"
    for report in ("platform", "all"):
        try:
            proc = subprocess.run(
                [xrt_smi, "--batch", "examine", "-r", report],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return None, f"xrt-smi examine -r {report} failed: {exc}"
        match = re.search(
            r"Power Mode\s*:\s*([A-Za-z][A-Za-z0-9_-]*)",
            proc.stdout + proc.stderr,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).strip().lower(), f"xrt-smi examine -r {report}"
    return None, "no 'Power Mode' line in any xrt-smi examine report"


def require_turbo():
    mode, detail = npu_power_mode()
    if mode == "turbo":
        return
    if mode is None:
        raise TurboNotEnforced(
            f"could not determine the NPU power mode ({detail}). A measurement "
            f"that cannot be shown to be at turbo must not enter the registry; "
            f"set it with `sudo xrt-smi configure --pmode turbo` and verify with "
            f"`xrt-smi examine -r platform`."
        )
    raise TurboNotEnforced(
        f"NPU power mode is `{mode}`, not turbo ({detail}). A non-turbo number "
        f"is not comparable with the turbo numbers already in the registry; set "
        f"it with `sudo xrt-smi configure --pmode turbo`."
    )


def signature(shape, candidate, perf_iters):
    """Stable hash over everything that changes what a measurement means."""
    payload = {
        "shape": [shape.M, shape.K, shape.N],
        "candidate": candidate.as_dict(),
        "perf_iters": perf_iters,
        "contract": MEASUREMENT_CONTRACT,
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def result_path(results_dir, shape, candidate, sig):
    return results_dir / f"{shape.label}__{candidate.method}__{sig}.json"


def _write_atomic(path, text):
    """Checkpoint without a window where the file is half-written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def load_checkpoint(path, sig):
    """A reusable prior verdict for this exact configuration, or None."""
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if record.get("signature") != sig:
        return None
    if record.get("status") not in REUSABLE_STATUSES:
        return None
    return record


def run_candidate(shape, candidate, sig, workdir, perf_iters, timeout_s, verbose):
    """Measure one candidate in its own process and directory."""
    scratch = Path(
        tempfile.mkdtemp(
            prefix=f"{shape.label}__{candidate.method}__", dir=str(workdir)
        )
    )
    payload = {
        "shape": {
            "M": shape.M,
            "K": shape.K,
            "N": shape.N,
            "family": shape.family,
            "role": shape.role,
            "seq": shape.seq,
        },
        "candidate": candidate.as_dict(),
        "perf_iters": perf_iters,
    }
    payload_path = scratch / "payload.json"
    result_file = scratch / "result.json"
    payload_path.write_text(json.dumps(payload, indent=2))

    cmd = [
        sys.executable,
        str(_HERE / "sweep_measure.py"),
        "--payload",
        str(payload_path),
        "--result",
        str(result_file),
    ]
    if verbose:
        cmd.append("--verbose")

    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(scratch),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        proc, timed_out = exc, True

    elapsed = time.time() - started

    if timed_out:
        record = {
            "shape": payload["shape"],
            "candidate": payload["candidate"],
            "tier": METHODS[candidate.method]["tier"],
            "status": "failed_timeout",
            "gflops": None,
            "error": f"no verdict within {timeout_s}s",
        }
    elif result_file.exists():
        record = json.loads(result_file.read_text())
    else:
        # The worker exits non-zero without a result file only when the harness
        # itself broke, which says nothing about the candidate.
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        record = {
            "shape": payload["shape"],
            "candidate": payload["candidate"],
            "tier": METHODS[candidate.method]["tier"],
            "status": "failed_no_comparison",
            "gflops": None,
            "error": f"worker exited {proc.returncode} with no result:\n{tail}",
        }

    record["signature"] = sig
    record["wall_seconds"] = round(elapsed, 1)
    shutil.rmtree(scratch, ignore_errors=True)
    return record


def sweep(
    shapes,
    results_dir,
    workdir,
    perf_iters=DEFAULT_PERF_ITERS,
    timeout_s=DEFAULT_TIMEOUT_S,
    resume=True,
    verbose=False,
    candidate_kwargs=None,
):
    """Measure every candidate of every shape. Returns {shape.key: [records]}."""
    results_dir.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)
    all_records = {}
    total = sum(
        len(candidates_for_shape(s, **(candidate_kwargs or {}))) for s in shapes
    )
    done = 0

    for shape in shapes:
        records = []
        for candidate in candidates_for_shape(shape, **(candidate_kwargs or {})):
            sig = signature(shape, candidate, perf_iters)
            path = result_path(results_dir, shape, candidate, sig)
            done += 1
            prefix = f"[{done}/{total}] {shape.label} {candidate.label}"

            cached = load_checkpoint(path, sig) if resume else None
            if cached is not None:
                print(f"{prefix}: reuse {cached['status']}")
                records.append(cached)
                continue

            # Fail-closed, before every candidate: see the module docstring.
            require_turbo()

            print(f"{prefix}: measuring", flush=True)
            record = run_candidate(
                shape, candidate, sig, workdir, perf_iters, timeout_s, verbose
            )
            _write_atomic(path, json.dumps(record, indent=2) + "\n")
            detail = (
                f"{record['gflops']:.0f} GFLOP/s"
                if record.get("gflops")
                else (record.get("error") or "").splitlines()[:1]
            )
            print(f"{prefix}: {record['status']} {detail}", flush=True)
            records.append(record)
        all_records[shape.key] = records
    return all_records


def _failure_summary(method_records):
    """Why a method has no row, in the terms a later reader needs.

    "failed_precision" alone is not enough to act on. A candidate that missed the
    bound by 15% is a tolerance edge; one that needed 140x it computed something
    else entirely, and the two want opposite responses from whoever reads the row
    next. So the closest ``atol_required`` any candidate reached is quoted
    alongside the bound it was checked at.
    """
    statuses = sorted({r["status"] for r in method_records})
    needed = [
        r["atol_required"]
        for r in method_records
        if r["status"] == "failed_precision" and r.get("atol_required") is not None
    ]
    if not needed:
        return ", ".join(statuses)
    atol = min(r["atol"] for r in method_records if r["status"] == "failed_precision")
    return (
        f"{', '.join(statuses)}; closest needed atol {min(needed):.2e} "
        f"vs {atol:.2e}"
    )


def select_winners(shape, records, candidates):
    """Fastest passing candidate per method, and the best method per tier.

    Ranking is max GFLOP/s. Ties break toward the earlier candidate in
    ``candidates_for_shape`` order, which is deterministic -- iron's equivalent
    ties on whatever order the resume file happened to list, which is not.
    """
    order = {
        json.dumps(c.as_dict(), sort_keys=True): i for i, c in enumerate(candidates)
    }

    def rank(record):
        key = json.dumps(record["candidate"], sort_keys=True)
        return (-record["gflops"], order.get(key, len(order)))

    methods, failed = {}, {}
    for method in METHODS:
        method_records = [r for r in records if r["candidate"]["method"] == method]
        if not method_records:
            continue
        passing = [
            r for r in method_records if r["status"] == "passed" and r.get("gflops")
        ]
        if not passing:
            failed[method] = _failure_summary(method_records)
            continue
        methods[method] = min(passing, key=rank)

    best = {}
    for tier in ("high", "low"):
        tier_methods = [m for m in methods if methods[m]["tier"] == tier]
        if tier_methods:
            best[tier] = min(tier_methods, key=lambda m: rank(methods[m]))

    notes = []
    herds = {
        (r["candidate"]["herd_m"], r["candidate"]["herd_n"]) for r in methods.values()
    }
    non_default = sorted(h for h in herds if list(h) != registry_writer.DEFAULT_HERD)
    if non_default:
        notes.append(
            "M="
            + str(shape.M)
            + " is smaller than tile_m x 8, so the herd shrinks: "
            + ", ".join(
                f"{m} {r['candidate']['herd_m']}x{r['candidate']['herd_n']}"
                for m, r in sorted(methods.items())
                if [r["candidate"]["herd_m"], r["candidate"]["herd_n"]]
                != registry_writer.DEFAULT_HERD
            )
            + " (per-method `herd` overrides the file-level 8x4)."
        )
    for method, summary in sorted(failed.items()):
        notes.append(f"{method}: no candidate passed ({summary}).")

    return {
        "shape": shape,
        "methods": methods,
        "best": best,
        "failed": failed,
        "note": " ".join(notes) or None,
    }


def collect_rows(shapes, all_records, candidate_kwargs=None):
    rows = []
    for shape in shapes:
        records = all_records.get(shape.key)
        if not records:
            continue
        rows.append(
            select_winners(
                shape, records, candidates_for_shape(shape, **(candidate_kwargs or {}))
            )
        )
    return rows


def load_all_results(shapes, results_dir, perf_iters, candidate_kwargs=None):
    """Re-read the checkpoints for a set of shapes, without measuring anything."""
    all_records = {}
    for shape in shapes:
        records = []
        for candidate in candidates_for_shape(shape, **(candidate_kwargs or {})):
            sig = signature(shape, candidate, perf_iters)
            path = result_path(results_dir, shape, candidate, sig)
            if path.exists():
                record = json.loads(path.read_text())
                if record.get("signature") == sig:
                    records.append(record)
        if records:
            all_records[shape.key] = records
    return all_records


def _select_shapes(args):
    shapes = shapes_for_family(args.family)
    if args.role:
        shapes = [s for s in shapes if s.role in args.role]
    if args.seq:
        shapes = [s for s in shapes if s.seq in args.seq]
    if args.shape:
        wanted = set(args.shape)
        shapes = [s for s in shapes if s.label in wanted]
    return shapes


def main():
    # Imported HERE and not at module scope: `sweep_report` consumes this
    # module's `collect_rows` / `load_all_results`, so a top-level import in both
    # directions is a cycle. The dependency has a direction -- the orchestrator
    # produces checkpoints and the report reads them -- and the CLI is the one
    # place that needs both halves, so this is where the edge belongs.
    from sweep_report import verify_resolution, write_family_markdown

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--family",
        default="baseline_768",
        choices=sorted(FAMILY_HIDDEN),
        help="which width of the case matrix to sweep",
    )
    parser.add_argument("--role", action="append", choices=ROLES)
    parser.add_argument("--seq", action="append", type=int, choices=SEQ_LADDER)
    parser.add_argument("--shape", action="append", help="one MxKxN label")
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--perf-iters", type=int, default=DEFAULT_PERF_ITERS)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--list", action="store_true", help="shapes as JSON; no hardware"
    )
    parser.add_argument("--plan", action="store_true", help="candidates; no hardware")
    parser.add_argument("--write-registry", action="store_true")
    parser.add_argument("--verify-resolution", action="store_true")
    parser.add_argument("--tile-n-options", type=int, default=None)
    parser.add_argument("--tile-k-l2-options", type=int, default=None)
    parser.add_argument("--cast-tile-n-options", type=int, default=None)
    parser.add_argument(
        "--remeasure-registered",
        action="store_true",
        help="measure shapes that are already registered (measuring writes "
        "only to --results; the registry write stays append-only)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    shapes = _select_shapes(args)
    results_dir = args.results or (DEFAULT_RESULTS / args.family)
    workdir = args.workdir or (DEFAULT_WORKDIR / args.family)
    candidate_kwargs = {}
    if args.tile_n_options is not None:
        candidate_kwargs["tile_n_options"] = args.tile_n_options
    if args.tile_k_l2_options is not None:
        candidate_kwargs["tile_k_l2_options"] = args.tile_k_l2_options
    if args.cast_tile_n_options is not None:
        candidate_kwargs["cast_tile_n_options"] = args.cast_tile_n_options

    if args.list:
        print(
            json.dumps(
                [
                    {
                        "role": s.role,
                        "seq": s.seq,
                        "M": s.M,
                        "K": s.K,
                        "N": s.N,
                        "label": s.label,
                    }
                    for s in shapes
                ],
                indent=2,
            )
        )
        return 0

    if args.plan:
        total = 0
        for shape in shapes:
            candidates = candidates_for_shape(shape, **candidate_kwargs)
            total += len(candidates)
            print(f"{shape.label} ({shape.role}, seq={shape.seq}):")
            for candidate in candidates:
                print(f"    {candidate.label}")
            if not candidates:
                print("    (no legal candidate)")
        print(f"{len(shapes)} shapes, {total} candidates")
        return 0

    if args.verify_resolution:
        failures = verify_resolution(shapes)
        if failures:
            print(f"\nFAIL: {len(failures)}/{len(shapes)} shapes do not resolve:")
            for shape, message in failures:
                print(f"  {shape.label} ({shape.role}, seq={shape.seq}): {message}")
            return 1
        print(f"\nPASS: all {len(shapes)} {args.family} shapes resolve")
        return 0

    if args.write_registry:
        already = registry_writer.registered_shape_keys()
        all_records = load_all_results(
            shapes, results_dir, args.perf_iters, candidate_kwargs
        )
        rows = collect_rows(shapes, all_records, candidate_kwargs)
        missing = [s.label for s in shapes if s.key not in all_records]
        if missing:
            print(f"WARNING: no measurements for {len(missing)} shapes: {missing}")
        unusable = [r["shape"].label for r in rows if not r["best"]]
        if unusable:
            print(f"WARNING: no passing candidate for {len(unusable)}: {unusable}")
        fresh = [r for r in rows if r["shape"].key not in already]
        written = registry_writer.append_shapes(fresh)
        print(f"appended {len(written)} shapes to {registry_writer.GEMM_JSON.name}")
        # A shape this sweep already wrote can still be missing a METHOD, if the
        # grid of the day had no passing candidate for it. Adding one is the
        # single edit `registry_writer` permits to an existing entry, and it is
        # guarded there; everything else about the entry stays byte-identical.
        report = registry_writer.add_missing_methods(
            [r for r in rows if r["shape"].key in already]
        )
        for key, methods in report["added"]:
            print(f"added {methods} to the existing {key} entry")
        if report["not_owned"]:
            print(
                f"already registered by someone else, left untouched: "
                f"{report['not_owned']}"
            )
        if report["unchanged"]:
            print(f"already complete, left untouched: {report['unchanged']}")
        # Deliberately not `rows`: the markdown section is replaced whole, so it
        # is rebuilt from the whole family however the append was narrowed.
        mirrored = write_family_markdown(
            args.family, results_dir, args.perf_iters, candidate_kwargs
        )
        print(
            f"mirrored {len(mirrored)} {args.family} rows into "
            f"GEMM_bf16_in_bf16_out.md and supported_kernels.md"
        )
        return 0

    already = registry_writer.registered_shape_keys()
    if args.remeasure_registered:
        queued = list(shapes)
        registered = [s.label for s in shapes if s.key in already]
        if registered:
            print(
                f"measuring {len(registered)} already-registered shapes on "
                f"request: {registered}. Measuring writes only to the results "
                f"directory; what the registry accepts back is still "
                f"append-only, plus a method an owned entry is missing."
            )
    else:
        queued = [s for s in shapes if s.key not in already]
        if len(queued) != len(shapes):
            print(
                f"skipping {len(shapes) - len(queued)} already-registered shapes "
                f"(this sweep is append-only)"
            )
    try:
        all_records = sweep(
            queued,
            results_dir,
            workdir,
            perf_iters=args.perf_iters,
            timeout_s=args.timeout,
            resume=not args.no_resume,
            verbose=args.verbose,
            candidate_kwargs=candidate_kwargs,
        )
    except TurboNotEnforced as exc:
        print(f"\nABORT: {exc}", file=sys.stderr)
        return 2

    rows = collect_rows(queued, all_records, candidate_kwargs)
    unresolved = [r["shape"].label for r in rows if not r["best"]]
    print(f"\nswept {len(rows)} shapes; {len(rows) - len(unresolved)} have a winner")
    if unresolved:
        print(f"no passing candidate: {unresolved}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
