# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One profile, one command, one manifest -- the unattended runner's core (G0).

    agents/scripts/devq.sh run --class measure -- \
        python3 study/run_profile.py --profile smoke --out-dir results/g0-smoke

CONTRACT
    Takes a named profile from ``profiles.py`` and does the five things that
    were previously done by hand in five invocations, in one process that
    either finishes with a complete manifest or says which clause failed:

      1. refuses off-Turbo, before anything is prepared;
      2. takes ``run_lock`` on the results root, so a second runner cannot
         overwrite the first one's rows;
      3. samples SoC power for the whole walk (``power.py``);
      4. walks every rung through ``run_ladder.walk``, recording structurally
         inapplicable rungs as ``run_status=skipped`` rather than running them;
      5. runs ``smoke_gate`` and writes ``results_manifest.json`` with the
         expected FILE list and the expected ROW COUNTS both derived from the
         profile -- doc 10's gate sentence, "no missing files or rows, against
         counts derived from the profile itself, not hard-coded".

    It writes one artifact of its own, ``profile_run.json``: the plan, the
    per-rung outcome, the wall clock, the power block, and the families the
    profile could not reach. The manifest says whether the run is complete; this
    says what the run was.

WHAT IT DELIBERATELY DOES NOT DO
    No reboot orchestration, no ``@reboot`` crontab hook, no TTM page-limit
    transitions, no thermal gate, no ``turbostat``. Doc 34 §4.4 recommends
    dropping all four on measured grounds and doc 10 now records them as
    dropped with those reasons. What survives of doc 10's privileged-setup
    block is a single binary, ``xrt-smi configure``, and it is the operator's
    action, never this script's.

    No device serialization either. ``agents/scripts/devq.sh`` is the queue and
    it is strictly more than the ``flock -x -w 1800 /tmp/mlir-air-npu.lock`` doc
    10 asks for -- FIFO order, a measure/build barrier, and liveness
    reconciliation immune to pid reuse. This script takes NO device lock and
    names no device lock path: ``run_lock`` here is per output file, and
    pointing it at either device inode would deadlock against the wrapper that
    launched the run.

FOOTGUNS
    - **``run``, never ``submit``.** ``submit`` diverts output to the job log and
      returns an id, so a gate that substitutes it blanks its own FileCheck and
      still exits 0.
    - **This is a laptop.** Lid close suspends it. Wrap a long profile:
      ``systemd-inhibit --what=handle-lid-switch:sleep:idle agents/scripts/devq.sh
      run --class measure -- python3 study/run_profile.py ...``
    - **Run it from ``programming_examples/transformer_layer``**, because aircc
      and ``KernelCache`` write relative to cwd and only that directory's
      ``.gitignore`` covers the debris. The run reports any tracked-tree dirt it
      finds afterwards rather than leaving a human to notice -- doc 15's rule
      that "a new artifact directory is the default outcome of adding a
      KernelCache-backed gate, not an exception".
    - **A devq job is a bare shell.** Source ``agents/scripts/port-loop/lib-env.sh``
      and call BOTH ``pl_env_ensure`` and ``pl_env_ensure_xrt``. Skipping the
      second compiles everything and then dies at the first dispatch on
      ``ModuleNotFoundError: No module named 'pyxrt'``, minutes in, looking like
      a model regression.
    - **The power block is SoC watts over the WHOLE walk, compilation
      included.** No sensor on this platform measures the NPU (doc 09), and
      compilation dominates a cold walk 20x over measurement, so this number is
      a run-level condition and NOT a per-mode or per-rung energy figure. It is
      reported at run level for exactly that reason and is not written into any
      results row.
    - **One walk is not a result.** README trap 1: a single walk published a
      crossover a second walk refuted. Walk twice into two roots and compare
      with ``compare_roots.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.dirname(_HERE)
_PE = os.path.dirname(_EXAMPLE)
for _p in (_PE, os.path.join(_PE, "llms"), _EXAMPLE, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import manifest  # noqa: E402
import power  # noqa: E402
import profiles  # noqa: E402
import run_lock  # noqa: E402
import smoke_gate  # noqa: E402

MANIFEST_NAME = "results_manifest.json"
RUN_REPORT_NAME = "profile_run.json"


def _require_turbo() -> None:
    """The measurement precondition. Imported, never re-derived.

    ``sweep.registry_sweep.require_turbo`` is the single implementation
    ``run_mode.py``, ``component_groups.py`` and
    ``agents/scripts/port-loop/pmode_guard.py`` all fail closed on -- the last
    of those imports it rather than parsing ``xrt-smi`` a second time, and so
    does this. Two parsers of one device's output disagree eventually, and the
    disagreement shows up as a verdict.

    ``run_mode`` re-takes this per rung because it runs as a fresh process, so a
    driver reload mid-walk is caught at the next rung. This call is the
    fail-fast one: it refuses before a results root exists.
    """
    from sweep.registry_sweep import TurboNotEnforced, require_turbo

    try:
        require_turbo()
    except TurboNotEnforced as exc:
        raise SystemExit(
            f"[run-profile] refused: {exc}\n"
            "[run-profile] The power mode is non-persistent and resets on every "
            "reboot and every amdxdna reload; at `Default` this host measures "
            "~15-20x slow, so a walk taken there is not comparable with any "
            "recorded number (README trap 0). Setting it is the operator's "
            "action and needs root."
        ) from None


def _tree_dirt(repo: Path) -> list[str]:
    """Tracked-tree paths a run left behind. Best effort; never raises.

    Ignored files do not appear, which is the point: a gate that leaks a new
    cache directory shows up here until that directory joins .gitignore and the
    clean target. Reported, never refused -- an operator editing files during a
    walk is normal and is not the runner's business to veto.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:
        return []
    if out.returncode != 0:
        return []
    return [line for line in out.stdout.splitlines() if line.strip()]


def _rung_outcomes(rows: list[dict]) -> list[dict]:
    """Per-rung outcome for the run report. Counts and status, no latency."""
    return [
        {
            "execution_mode": row.get("execution_mode"),
            "seq_len": row.get("seq_len"),
            "run_status": row.get("run_status"),
            "failure_message": row.get("failure_message") or None,
        }
        for row in rows
    ]


def gate(profile: profiles.Profile, out_dir: str | Path, repo=None) -> dict:
    """Run the gate over an existing tree and write its manifest. No device.

    Separated from the walk so a recorded results root can be re-verified
    without re-measuring it -- which is how doc 34 discovered that the recorded
    Phase F gate artifact no longer verifies against schema v2. A gate you
    cannot re-run over an old tree is a gate whose past verdicts are hearsay.
    """
    out_dir = Path(out_dir)
    expected = profile.expected_files()
    expected_rows = profile.expected_rows()

    problems = smoke_gate.check_results_root(out_dir, expected)
    for line in problems:
        print(f"[smoke-gate] {line}")
    print(f"[smoke-gate] {'FAIL' if problems else 'PASS'} " f"({len(expected)} CSV(s))")

    built = manifest.build_manifest(
        out_dir, expected, repo=repo, expected_rows=expected_rows
    )
    manifest.write_manifest(out_dir / MANIFEST_NAME, built)
    print(f"[manifest] wrote {out_dir / MANIFEST_NAME}")
    print(f"[manifest] complete: {built['complete']}")
    for reason in built["incomplete_reasons"]:
        print(f"[manifest]   {reason}")
    for record in built["expected_files"]:
        want, got = record.get("expected_rows"), record.get("observed_rows")
        if want and got:
            print(
                f"[manifest]   {record['path']:<14} "
                f"rows {got['total']}/{want['rows']}  "
                f"passed {got['passed']}/{want['measured']}  "
                f"skipped {got['skipped']}/{want['skipped']}"
            )
    return built


def run(
    profile: profiles.Profile,
    out_dir: str | Path,
    *,
    study_id: str,
    warmup: int,
    samples: int,
    runs_per_sample: int,
    power_backend: str = "auto",
    walker=None,
    repo=None,
) -> dict:
    """Walk the profile, gate it, and write both artifacts. Returns the report.

    ``walker`` is injectable so the host tests can exercise every step of this
    function without a device; production leaves it ``None`` and gets
    ``run_ladder.walk``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if walker is None:
        import run_ladder

        walker = run_ladder.walk

    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    with power.open_monitor(power_backend) as monitor:
        rows = walker(
            list(profile.modes),
            list(profile.seqs),
            str(out_dir),
            study_id,
            warmup,
            samples,
            runs_per_sample,
            skip_reason=profiles.skip_reason,
        )
        power_columns = monitor.stats()
    wall_sec = time.perf_counter() - t0

    built = gate(profile, out_dir, repo=repo)

    by_status: dict[str, int] = {}
    for row in rows:
        key = str(row.get("run_status"))
        by_status[key] = by_status.get(key, 0) + 1

    report = {
        "study_id": study_id,
        "profile": profile.summary(),
        "started_utc": started.isoformat(),
        "wall_clock_sec": round(wall_sec, 3),
        "rungs": _rung_outcomes(rows),
        "rungs_by_status": by_status,
        "complete": built["complete"],
        "incomplete_reasons": built["incomplete_reasons"],
        # SoC watts over the WHOLE walk, compilation included. See the footgun.
        "power_over_whole_walk": power_columns,
        "devq_job_id": os.environ.get("DEVQ_JOB_ID"),
        "tree_dirt_after_run": _tree_dirt(Path(repo) if repo else Path(_PE).parent),
    }
    (out_dir / RUN_REPORT_NAME).write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"[run-profile] wrote {out_dir / RUN_REPORT_NAME}")
    return report


def _print_plan(profile: profiles.Profile) -> None:
    print(f"[run-profile] profile {profile.name}: {profile.description}")
    for rung in profile.rungs():
        reason = rung.skip_reason
        print(
            f"[run-profile]   {'SKIP' if reason else 'run '} "
            f"{rung.mode:<9} seq {rung.seq:<6}" + (f"  {reason}" if reason else "")
        )
    for rel, counts in profile.expected_rows().items():
        print(
            f"[run-profile] expect {rel:<14} rows {counts['rows']:>3}  "
            f"passed {counts['measured']:>3}  skipped {counts['skipped']:>3}"
        )
    for fid, why in profile.unreachable().items():
        print(f"[run-profile] NOT WALKED {fid}: {why}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--profile", required=True, help=f"one of {sorted(profiles.PROFILES)}"
    )
    ap.add_argument("--out-dir", required=True, help="results root for this walk")
    ap.add_argument("--study-id", default=None, help="defaults to g0-<profile>")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--runs-per-sample", type=int, default=1)
    ap.add_argument("--power-backend", default="auto")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and the expected counts; touch no device",
    )
    ap.add_argument(
        "--gate-only",
        action="store_true",
        help="re-verify an existing results root and rewrite its manifest",
    )
    args = ap.parse_args(argv)

    try:
        prof = profiles.profile(args.profile)
    except ValueError as exc:
        print(f"[run-profile] {exc}")
        return 2
    study_id = args.study_id or f"g0-{prof.name}"

    if args.dry_run:
        _print_plan(prof)
        return 0

    if args.gate_only:
        built = gate(prof, args.out_dir)
        return 0 if built["complete"] else 1

    _print_plan(prof)
    _require_turbo()

    if not os.environ.get("DEVQ_JOB_ID"):
        # Not a refusal: a host-only rehearsal is legitimate. But the device is
        # scheduled by devq and a walk taken beside another job's dispatches is
        # a silently distorted measurement, not a failed one.
        print(
            "[run-profile] WARNING not running under devq. The device is "
            "scheduled by agents/scripts/devq.sh; take it with "
            "`devq.sh run --class measure -- ...` so no build or gate runs "
            "beside the timed region."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = run_lock.lock_path_for(out_dir / MANIFEST_NAME)
    try:
        with run_lock.hold(lock, study=f"profile {prof.name}"):
            report = run(
                prof,
                out_dir,
                study_id=study_id,
                warmup=args.warmup,
                samples=args.samples,
                runs_per_sample=args.runs_per_sample,
                power_backend=args.power_backend,
            )
    except run_lock.StudyAlreadyRunning as exc:
        print(f"[run-profile] refused: {exc}")
        return 2

    counts = report["rungs_by_status"]
    print(
        f"[run-profile] {prof.name}: "
        + "  ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        + f"  ({report['wall_clock_sec']:.0f}s wall)"
    )
    if report["tree_dirt_after_run"]:
        print(
            f"[run-profile] WARNING the run left {len(report['tree_dirt_after_run'])} "
            "tracked-tree change(s); a new cache directory joins .gitignore AND "
            "the clean target in the same commit (doc 15)"
        )
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
