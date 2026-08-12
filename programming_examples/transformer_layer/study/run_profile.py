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

RESUME `[2026-08-12]` -- doc 10 work item 8
    ``--resume`` carries forward every rung this root already holds a ``passed``
    row for and walks the rest. A ``ladder`` walk is ~45 min cold and this host
    is a laptop that suspends on a lid close; without resume a reboot at minute
    40 costs the whole thing.

    Three things make it a resume rather than a shortcut, and the third is the
    one that matters:

      - ``study/resume.py`` decides reuse. ``failed`` is never reused, only
        ``passed``; a skip is re-derived from the profile every session.
      - the ledger (``walk_sessions.json``) is opened before the walk and
        flushed after EVERY rung, so a session killed halfway leaves an
        ``interrupted`` record naming what it attributed, which the next session
        closes out. That record reaches the manifest as the ``walk`` block.
      - the run AUDITS ITSELF. Every carried-forward rung is re-hashed after the
        walk; a rung the plan called reused whose row moved is a ``RESUME
        DEFECT`` and makes the run INCOMPLETE. A resume that silently redoes
        work empties every downstream diff, and bookkeeping that only counts
        what the runner intended could never see it.

    Completeness is unaffected by resuming: ``manifest``'s row-count clauses
    read the CSVs and know nothing about sessions, so a resumed walk that is
    short is incomplete exactly as a fresh one is. **A resume cannot report a
    complete walk that is not one.** What it cannot do is make two sessions one
    measurement -- see ``resume.py`` §WHAT IT CANNOT.

    A results root that already holds CSVs is REFUSED without ``--resume``.

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

    IT DOES ASK, THOUGH `[2026-08-12]` (queue item 19). The lock is advisory --
    nothing outside the broker consults it -- which is how a run that meant to
    be compile-only dispatched beside job 252's 65-minute regression. Off-queue,
    this script now calls ``devq.sh preflight`` and REFUSES to start when the
    device is held by another job. Asking is not taking: preflight holds nothing
    and cannot acquire, so this still takes no device lock.

    The split is the codebase's existing one, not a new rule. A DEFINITE "held"
    refuses, because the exit is actionable and obvious -- re-run under ``devq.sh
    run --class measure``. An INDETERMINATE answer (no broker, unrunnable) warns
    and continues, exactly as ``compare_roots`` flags an unrecorded power mode:
    refusing on an unknown would make a broken broker block all measurement. And
    there is no ``--ignore-device-lock``, for ``pmode_guard``'s reason: nothing
    added for a guard may become the way to defeat it. Wanting to dispatch
    anyway is wanting ``devq.sh run``.

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
import resume as resume_mod  # noqa: E402
import run_lock  # noqa: E402
import schema  # noqa: E402
import smoke_gate  # noqa: E402

MANIFEST_NAME = "results_manifest.json"
RUN_REPORT_NAME = "profile_run.json"


def _require_turbo() -> str:
    """The measurement precondition, and it RETURNS what it observed.

    `[2026-08-12]` The return value is new and it closes a hole opened by the
    conditions block landing before this caller did: ``gate`` never passed
    ``conditions=`` to ``build_manifest``, so every profile manifest recorded
    ``npu_power_mode: unknown`` -- on a run that had just REFUSED to start
    unless the mode was turbo. The rule is "never stamp a condition you did not
    observe"; this was its inverse, observing and then discarding, which is the
    worse half because the artifact then looks like nobody could tell.

    Stamped ``observed`` rather than ``probed_at_manifest_build`` because this
    call happens before the walk, on the clock of the measurement, which is what
    that source value means. ``seed-throughput-baseline.sh`` stamps the
    throughput floor the same way for the same reason.

    Imported, never re-derived.

    ``sweep.registry_sweep.require_turbo`` is the single implementation
    ``run_mode.py``, ``component_groups.py`` and
    ``agents/scripts/port-loop/pmode_guard.py`` all fail closed on -- the last
    of those imports it rather than parsing ``xrt-smi`` a second time, and so
    does this. Two parsers of one device's output disagree eventually, and the
    disagreement shows up as a verdict.

    ``run_mode`` re-takes this per rung because it runs as a fresh process, so a
    driver reload mid-walk is caught at the next rung. This call is the
    fail-fast one: it refuses before a results root exists.

    **Exit 2, matching ``run_mode``**, because a refused precondition is not a
    failed measurement and a caller must be able to tell them apart: 1 means the
    profile ran and did not complete, 2 means it never started.
    """
    from sweep.registry_sweep import TurboNotEnforced, npu_power_mode, require_turbo

    try:
        require_turbo()
        mode, _detail = npu_power_mode()
        return schema.normalise_power_mode(mode)
    except TurboNotEnforced as exc:
        print(f"[run-profile] refused: {exc}")
        print(
            "[run-profile] The power mode is non-persistent and resets on every "
            "reboot and every amdxdna reload; at `Default` this host measures "
            "~15-20x slow, so a walk taken there is not comparable with any "
            "recorded number (README trap 0). Setting it is the operator's "
            "action and needs root."
        )
        raise SystemExit(2) from None


#: The broker, resolved from this file rather than from cwd or PATH -- doc 15's
#: rule that a probe must not depend on where it was launched from.
DEVQ = Path(_HERE).resolve().parents[2] / "agents" / "scripts" / "devq.sh"


def device_preflight(devq: Path | None = None) -> bool | None:
    """Is the NPU free to dispatch to right now? ``None`` means indeterminate.

    `[2026-08-12]`, queue item 19. The device lock is advisory: nothing outside
    ``devq.sh`` consults it, so an off-queue dispatch runs beside whatever holds
    it. This ASKS -- ``devq.sh preflight`` is read-only, holds nothing and cannot
    acquire -- so the module's "takes no device lock" contract is intact.

      True   the lock is free, or the caller is already inside a devq job.
      False  another job holds it. The caller must not dispatch.
      None   the broker could not answer. Warn, do not refuse.

    THE THREE-VALUED RETURN IS THE POINT. A bool would collapse "the device is
    busy" into "I could not tell", and those need opposite responses: the first
    is a definite finding with an obvious exit (re-run under ``devq.sh run
    --class measure``), the second is the guard itself being broken, and
    refusing there would let a missing broker block all measurement. That is
    ``compare_roots``' refuse-known / flag-unknown split, applied to a device
    rather than to a recorded condition.

    Never raises: a guard that dies is a guard that stops the run it was meant
    to protect, for a reason that has nothing to do with the device.
    """
    devq = Path(devq) if devq else DEVQ
    if not devq.is_file():
        print(f"[run-profile] WARNING cannot preflight the device: no {devq}")
        return None
    try:
        proc = subprocess.run(
            ["bash", str(devq), "preflight"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        print(f"[run-profile] WARNING cannot preflight the device ({exc})")
        return None
    if proc.returncode == 0:
        return True
    if proc.returncode == 3:
        for line in (proc.stderr or "").strip().splitlines():
            print(f"[run-profile] {line}")
        print(
            "[run-profile] refused: the NPU is held by another job. Dispatching "
            "now would corrupt its measurement and distort this one -- devq job "
            "252's ten-model regression survived exactly this only because "
            "contention causes false FAILURES rather than false passes. Re-run "
            "under `devq.sh run --class measure -- ...`; there is deliberately "
            "no flag to skip this."
        )
        return False
    print(
        f"[run-profile] WARNING the device preflight was inconclusive "
        f"(devq.sh preflight exited {proc.returncode}): "
        f"{(proc.stderr or '').strip()[:200]}"
    )
    return None


#: Modules a rung needs that a bare shell does not have, with what a caller who
#: is missing one sees instead. Both are LATE failures: they surface at the first
#: dispatch or the first builder import, minutes into a cold walk, looking like a
#: model regression rather than like a shell that was never set up.
_REQUIRED_MODULES = {
    "pyxrt": (
        "pyxrt lives beside the XRT install and `env_setup.sh` does NOT add it. "
        "Without it XRTBackend.load() raises at the FIRST DISPATCH -- after "
        "every kernel has compiled -- and the traceback says "
        "ModuleNotFoundError, which reads as a broken model. Source "
        "agents/scripts/port-loop/lib-env.sh and call BOTH pl_env_ensure and "
        "pl_env_ensure_xrt"
    ),
    "ml_dtypes": (
        "every builder imports bfloat16 from ml_dtypes at module scope, so the "
        "first rung dies importing opcheck_specs. Same fix: the port-loop "
        "environment, not a pip install beside a live gate"
    ),
}


def environment_problems(cwd: Path | None = None, importable=None) -> list[str]:
    """What would make this walk die mid-suite. Empty means nothing found.

    `[2026-08-12]` DOC 10's WORK ITEM 5, TAKEN AS A CHECK RATHER THAN AS PROSE.
    That item asks for "the prerequisites and recovery sections of the example
    README", because "the runner shells out to all of them, and a missing tool
    fails mid-suite rather than at start unless checked". The last four words are
    the requirement; the README is one way to meet it, and the weaker one -- a
    paragraph cannot fail, and this project's own record is of prose rules that
    were true when written and silently stopped being true (README trap 0 lived
    in prose for exactly that reason until the conditions block moved it into the
    artifact).

    The TABLE doc 10 specifies is separately obsolete and is recorded as dropped:
    of its six tools, `amd-ttm`, `turbostat`, `sensors`, `rocm-smi` and `crontab`
    are all in doc 10 §Deliberately dropped with a measurement behind each, and
    the sixth, `xrt-smi`, is only ever READ here -- `require_turbo` already
    refuses when it is missing or unparsable. So the prerequisite that is
    actually unguarded is not a binary at all: it is the two Python modules a
    bare devq shell lacks, both of which fail LATE.

    ``importable`` is injectable so the host tests can drive both directions
    without unloading a module out from under the interpreter running them.
    """
    import importlib.util

    if importable is None:

        def importable(name):
            try:
                return importlib.util.find_spec(name) is not None
            except Exception:
                return False

    problems = []
    for name, why in _REQUIRED_MODULES.items():
        if not importable(name):
            problems.append(f"`{name}` is not importable. {why}")

    # aircc and KernelCache write relative to cwd, and only the example
    # directory's .gitignore covers the debris -- doc 15's rule that "a new
    # artifact directory is the DEFAULT OUTCOME of adding a KernelCache-backed
    # gate, not an exception". A walk from anywhere else leaks .o files,
    # air_project/ and four *_cache/ directories into whatever it was launched
    # from, which is how eleven artifacts were committed by mistake once.
    cwd = Path(cwd) if cwd else Path.cwd()
    if cwd.resolve() != Path(_EXAMPLE).resolve():
        problems.append(
            f"the working directory is {cwd}, not {_EXAMPLE}. aircc and "
            "KernelCache write relative to cwd and only the example's own "
            ".gitignore covers what they write, so a walk from here leaves "
            "*.o, air_project/ and the per-mode *_cache/ directories loose in "
            "the tree (doc 15)"
        )
    return problems


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


def _rung_sources(ledger) -> dict[str, int]:
    """This session's rungs by ``schema.RUNG_SOURCES``, read off the ledger.

    Off the LEDGER and not off the plan, deliberately: the ledger is appended by
    the walker as each rung lands, so it says what happened, while the plan says
    what was intended. A report built from the plan would agree with itself no
    matter what the walk did, which is the shape of check G0 closed twice.
    """
    counts = {source: 0 for source in schema.RUNG_SOURCES}
    for rung in ledger.sessions[-1]["rungs"]:
        counts[rung["source"]] += 1
    return counts


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


def gate(
    profile: profiles.Profile,
    out_dir: str | Path,
    repo=None,
    conditions=None,
    toolchain=None,
    walk=None,
) -> dict:
    """Run the gate over an existing tree and write its manifest. No device.

    Separated from the walk so a recorded results root can be re-verified
    without re-measuring it -- which is how doc 34 discovered that the recorded
    Phase F gate artifact no longer verifies against schema v2. A gate you
    cannot re-run over an old tree is a gate whose past verdicts are hearsay.

    EVERY CONDITION BLOCK IS THE CALLER'S, AND ALL THREE DEFAULT TO NONE.
    `[2026-08-12]`, queue item 16 and resume. This function has two callers with
    opposite entitlements. ``run`` measured on this host just now, so it hands
    over what it observed -- the power mode it refused to start without, the
    toolchain it dispatched through, and the ledger it just wrote.
    ``--gate-only`` re-verifies a tree measured at some earlier time, possibly
    against a toolchain since overwritten and a power mode since reset; probing
    there would write TODAY's conditions onto SOMEONE ELSE's measurement, which
    is the "never stamp a condition you did not observe" rule broken by a
    helpful default. So a re-gate records ``unknown``, loudly, and the walk
    block it reads back is whatever the tree already carries.
    """
    out_dir = Path(out_dir)
    expected = profile.expected_files()
    expected_rows = profile.expected_rows()

    problems = smoke_gate.check_results_root(out_dir, expected)
    for line in problems:
        print(f"[smoke-gate] {line}")
    print(f"[smoke-gate] {'FAIL' if problems else 'PASS'} " f"({len(expected)} CSV(s))")

    built = manifest.build_manifest(
        out_dir,
        expected,
        repo=repo,
        expected_rows=expected_rows,
        conditions=conditions,
        toolchain=toolchain,
        walk=walk,
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
    block = built[schema.WALK_KEY]
    if built["walk_attribution_checked"]:
        print(f"[manifest]   walk: {block['walk_detail']}")
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
    resume: bool = False,
    npu_power_mode: str | None = None,
) -> dict:
    """Walk the profile, gate it, and write both artifacts. Returns the report.

    ``walker`` is injectable so the host tests can exercise every step of this
    function without a device; production leaves it ``None`` and gets
    ``run_ladder.walk``.

    ``resume`` carries forward every rung this root already holds a ``passed``
    row for. The ledger is opened BEFORE the walk and flushed after every rung,
    so a session killed halfway leaves an ``interrupted`` record naming exactly
    the rungs it attributed -- not a hole where the reason for the resume was.

    ``npu_power_mode`` is what the caller OBSERVED before starting (``main``
    passes what ``_require_turbo`` saw). It is stamped ``observed``, and it is
    also what a later session's rows are checked against for a pmode splice.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if walker is None:
        import run_ladder

        walker = run_ladder.walk

    expected_files = profile.expected_files()
    repo_root = Path(repo) if repo else Path(_PE).parent

    # Scanned BEFORE the ledger is opened: what the root holds is a fact about
    # the previous sessions, and opening a session first would make this
    # session's own (empty) attribution part of the input to its own plan.
    prior = resume_mod.scan(out_dir, expected_files)
    ledger = resume_mod.Ledger.load(out_dir)
    conditions = manifest.observe_conditions(npu_power_mode)
    toolchain = manifest.observe_toolchain()

    plan = resume_mod.plan(profile, prior, enabled=resume)
    if resume:
        for line in resume_mod.describe(plan):
            print(f"[run-profile] {line}")
        print(
            f"[run-profile] resume: {len(plan.reuse)} rung(s) carried forward, "
            f"{len(plan.remeasure)} to walk, {len(plan.skipped)} skipped"
        )

    started = datetime.now(timezone.utc)
    ledger.open_session(
        profile=profile.name,
        started_utc=started.isoformat(),
        devq_job_id=os.environ.get("DEVQ_JOB_ID"),
        git_sha=resume_mod.git_sha(repo_root),
        npu_power_mode=conditions["npu_power_mode"],
        toolchain_fingerprint=resume_mod.toolchain_fingerprint(toolchain),
    )
    t0 = time.perf_counter()
    try:
        with power.open_monitor(power_backend) as monitor:
            rows = walker(
                list(profile.modes),
                list(profile.seqs),
                str(out_dir),
                study_id,
                warmup,
                samples,
                runs_per_sample,
                # BOUND to this profile's family. The bare module function
                # would apply the 768 packing bound to a 512 walk, skipping
                # `fused` rungs it supports -- and a skipped rung is not a
                # failure, so the walk would report complete having never
                # attempted them.
                skip_reason=profile.skip_rule(),
                reuse=plan.reuse_for_walk,
                on_rung=ledger.record_rung,
                family=profile.family,
            )
            power_columns = monitor.stats()
    finally:
        # Closed in a `finally` so a walk that raises still leaves a readable
        # session -- an exception mid-walk is the ordinary case here, not the
        # exceptional one, and an unclosed session is what the NEXT run reports
        # as `interrupted`. Nothing is invented: `ended_utc` is set because this
        # process did observe the end.
        ledger.close_session()
    wall_sec = time.perf_counter() - t0

    # THE CHECK AGAINST A RESUME THAT SILENTLY REDOES WORK. `plan` said which
    # rungs would be carried forward; this re-hashes what the walk actually
    # produced for them. Bookkeeping alone would report whatever the plan said.
    fidelity = resume_mod.fidelity_problems(plan.reuse, rows)
    for line in fidelity:
        print(f"[run-profile] RESUME DEFECT {line}")

    # Re-scanned AFTER the walk, so the audit compares the ledger against the
    # files as they now are rather than against the rows this process happens to
    # be holding -- the two differing is precisely what it is looking for.
    after = resume_mod.scan(out_dir, expected_files)
    walk_block = resume_mod.walk_block(
        ledger, after, profile=profile, fidelity=fidelity
    )

    # The walk just happened, on this host: the conditions and the toolchain
    # observed here describe the build that produced these rows. See `gate`'s
    # docstring for why the `--gate-only` caller below deliberately does not.
    built = gate(
        profile,
        out_dir,
        repo=repo,
        conditions=conditions,
        toolchain=toolchain,
        walk=walk_block,
    )

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
        # Counted from the LEDGER, which is written per rung by the walker --
        # not from the plan, which is what this process intended. The two
        # disagreeing is `resume_defects` below.
        "rungs_by_source": _rung_sources(ledger),
        "resume_requested": resume,
        "resume_defects": fidelity,
        "session_id": ledger.sessions[-1]["session_id"],
        "session_count": len(ledger.sessions),
        "condition_splices": walk_block["condition_splices"],
        "complete": built["complete"],
        "incomplete_reasons": built["incomplete_reasons"],
        # SoC watts over the WHOLE walk, compilation included. See the footgun.
        "power_over_whole_walk": power_columns,
        "devq_job_id": os.environ.get("DEVQ_JOB_ID"),
        "tree_dirt_after_run": _tree_dirt(repo_root),
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
    ap.add_argument(
        "--family",
        default=None,
        help=f"retarget the profile to another case-matrix family; one of "
        f"{list(profiles.REACHABLE_FAMILIES)}. Every expected count is "
        f"re-derived, including `fused`'s applicability bound, which moves "
        f"with the width.",
    )
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
    ap.add_argument(
        "--resume",
        action="store_true",
        help="carry forward every rung this root already has a PASSED row for, "
        "and walk the rest. Required to write into a root that already holds "
        "results: without it a populated root is refused rather than half "
        "overwritten. Failed rungs are always re-run -- see study/resume.py.",
    )
    args = ap.parse_args(argv)

    try:
        prof = profiles.profile(args.profile)
        if args.family:
            prof = prof.retarget(args.family)
    except ValueError as exc:
        print(f"[run-profile] {exc}")
        return 2
    study_id = args.study_id or f"g0-{prof.name}"
    if args.family:
        study_id = args.study_id or f"g0-{prof.name}-{args.family}"

    if args.dry_run:
        _print_plan(prof)
        return 0

    if args.gate_only:
        built = gate(prof, args.out_dir)
        return 0 if built["complete"] else 1

    _print_plan(prof)

    # A POPULATED ROOT IS REFUSED UNLESS A RESUME WAS ASKED FOR, and this guard
    # points both ways. Walking a used root without `--resume` half-overwrites
    # it: `run_ladder` rewrites a mode's CSV from the rungs of THIS walk, so a
    # root would end up holding one mode's old rows beside another mode's new
    # ones with nothing recording the seam. And resuming a root by accident is
    # the opposite failure -- last week's rows presented as today's walk. There
    # is no flag to overwrite, because wanting to overwrite a recorded walk is
    # wanting a different --out-dir, which costs nothing and keeps both.
    populated = [
        rel for rel in prof.expected_files() if (Path(args.out_dir) / rel).is_file()
    ]
    if populated and not args.resume:
        print(
            f"[run-profile] refused: {args.out_dir} already holds "
            f"{', '.join(populated)}. Pass --resume to carry forward the rungs "
            "that already have passed rows, or give a fresh --out-dir. Walking "
            "over a recorded root would leave one CSV from each walk and "
            "nothing saying which is which."
        )
        return 2

    # Doc 10 item 5: refuse at START, not mid-suite. AFTER the argument checks
    # above -- an out-dir mistake is the caller's and is instant to fix -- and
    # BEFORE the pmode check, because a shell with no pyxrt cannot dispatch
    # whatever the power mode is.
    environment = environment_problems()
    for problem in environment:
        print(f"[run-profile] refused: {problem}")
    if environment:
        return 2

    mode = _require_turbo()

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
        # ...and being off-queue is only a warning while the device is FREE.
        # See the module docstring for the refuse/flag split.
        if device_preflight() is False:
            return 3

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
                resume=args.resume,
                npu_power_mode=mode,
            )
    except run_lock.StudyAlreadyRunning as exc:
        print(f"[run-profile] refused: {exc}")
        return 2

    counts = report["rungs_by_status"]
    sources = report["rungs_by_source"]
    print(
        f"[run-profile] {prof.name}: "
        + "  ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        + f"  ({report['wall_clock_sec']:.0f}s wall)"
    )
    print(
        f"[run-profile] {report['session_id']} of {report['session_count']}: "
        + "  ".join(f"{k} {v}" for k, v in sorted(sources.items()))
    )
    for axis in report["condition_splices"]:
        print(
            f"[run-profile] WARNING this root's rows were measured across a "
            f"{axis} change; they were not all produced by one tree"
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
