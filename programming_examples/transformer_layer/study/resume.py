# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resume an interrupted profile walk, and say honestly what was walked when.

    python3 study/resume.py <results-root> --profile ladder      # what a resume would do
    python3 study/resume.py <results-root> --profile ladder --audit

CONTRACT
    Three things, and they are three because a resume that does only the first
    is the trap this module exists to avoid:

      1. PLAN -- ``plan(profile, prior)`` decides, per rung, reuse or re-measure.
      2. LEDGER -- ``Ledger`` records which SESSION produced which rung, flushed
         after every rung so a killed session's attribution survives it.
      3. AUDIT -- ``walk_block(...)`` turns the ledger plus the rows on disk into
         the manifest's ``walk`` block, and every disagreement between the two
         becomes an ``attribution_problem``, which ``manifest.build_manifest``
         merges into ``incomplete_reasons``.

    Step 3 is the point. Steps 1 and 2 are bookkeeping, and bookkeeping that
    only counts what the runner says it did cannot catch a runner that says the
    wrong thing. G0's two closed defects were both checks that COULD NOT FAIL --
    a `run_status="skipped"` nothing emitted, and a manifest that validated
    files while a CSV holding 1 of 9 rungs reported ``complete: True``. So every
    attributed rung carries a ``row_digest`` and the audit re-hashes the file.

WHAT RESUME GUARANTEES
    - A rung with a ``passed`` row from an earlier session is not re-run.
    - Every rung of the profile appears in its CSV exactly once, whether it was
      measured now, carried forward, or skipped.
    - The completeness verdict is UNCHANGED by resuming: ``manifest``'s three
      row-count clauses read the CSVs on disk and know nothing about sessions,
      so a resumed walk that is short is incomplete exactly as a fresh one is.
    - A rung the plan declared ``reused`` whose final row does not hash to the
      prior row's digest is reported as a problem. That is the check against a
      resume that SILENTLY REDOES work -- the failure mode that empties a
      downstream diff and makes a tamper check vacuous (doc 14 §A seventh).
    - A row on disk that no session claims is ``rungs_unattributed`` and a
      problem, so a hand-dropped or hand-edited CSV cannot pass as a walk.
    - A splice across power modes is a problem; across a toolchain or a git sha
      it is flagged. That is ``compare_roots``' refuse-known / flag-unknown
      split, applied WITHIN one results root instead of between two.

WHAT IT CANNOT
    - **It cannot make a spliced walk one measurement.** Rows from two sessions
      are two populations, taken hours or days apart on a laptop whose thermal
      and load state nobody recorded. The block says so; it does not fix it.
      Two walks into two roots and ``compare_roots`` remains the standing rule
      (README trap 1) and resume does not weaken it.
    - **It cannot resume mid-rung.** The granularity is one ``run_mode`` child
      process. A rung killed at second 300 of 340 is re-run from zero.
    - **It cannot detect a distorted measurement.** A rung measured beside
      another job's dispatches produces a perfectly valid row with a perfectly
      good digest. Contention is devq's problem and stays devq's problem.
    - **It cannot attribute a row to a session at row level.** Attribution is
      keyed by ``(execution_mode, seq_len)`` and lives OUTSIDE the CSV, because
      a ``session_id`` column would bump ``SCHEMA_VERSION`` to 3 and take every
      surviving v2 root out of every reader (item 15's decision, unchanged).
      The digest is what buys back the confidence that key-based attribution
      loses -- it is a claim about a row, checked against the row.

THE REUSE POLICY, AND WHY `failed` IS NOT REUSABLE
    ``sweep/registry_sweep.py``'s ``REUSABLE_STATUSES`` splits verdicts that
    describe the CANDIDATE from verdicts that describe the MACHINE, and doc 10
    §Resume idempotence points here. The split lands differently in this tier,
    for a reason worth stating rather than inheriting:

      passed   REUSED. A completed measurement of a rung nobody is re-measuring.
      skipped  RE-DERIVED, every session, free. A skip is the PROFILE's current
               claim about what a mode supports, not a recorded observation; if
               `fused`'s packing bound moves, the old skip is a stale claim and
               reusing it would freeze a rule the profile has since changed.
      failed   RE-RUN. Deliberately, and it is the one place this is stricter
               than the registry sweep. Between two sessions the tree can change
               -- that is WHY the walk was interrupted often enough to need
               resume -- and a retained failure is a claim about code that may
               no longer be there. The sweep can reuse `failed_build` because a
               registry row is keyed by a MEASUREMENT_CONTRACT hash that changes
               when the meaning does; a results CSV has no such key, so the
               conservative direction is the honest one. Re-running a failure
               costs time and can only ever produce a truer row.

    There is no ``--reuse-failed``. Wanting one is wanting a fresh walk into a
    fresh root, which is one flag away and leaves both artifacts intact.

FOOTGUNS
    - **A ledger that cannot be read is a PROBLEM, not a fresh start.** Starting
      over on a corrupt ledger would silently erase the attribution of every row
      already on disk and let the resumed walk look clean. The exit is a new
      ``--out-dir``, not a repaired file.
    - **The ledger is flushed after every rung**, so it is exactly as durable as
      the CSVs ``run_ladder`` rewrites after every rung. Batching it at the end
      would lose precisely the session a resume exists to recover from.
    - **A session that walked a different profile into the same root is a splice
      of two plans**, and the block records the profile per session so it is
      visible. Nothing here merges two plans; the manifest's counts are the
      CALLER's profile and a root walked under two is judged against one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import results_io  # noqa: E402
import schema  # noqa: E402

#: The ledger's filename inside a results root. Beside the manifest, not inside
#: it: the manifest is DERIVED and rewritten whole on every gate, and an
#: append-only record cannot live in a file something else truncates.
LEDGER_NAME = "walk_sessions.json"

#: Row statuses a later session may carry forward untouched. One entry, on
#: purpose -- see THE REUSE POLICY in the module docstring.
REUSABLE_STATUSES: frozenset[str] = frozenset({"passed"})


def row_digest(row: dict) -> str:
    """A stable hash of one results row, in schema field order.

    Hashes the CSV SERIALISATION rather than the dict, so a row hashes the same
    whether it was just built in memory (floats, ints, None) or read back from
    the file (strings, None for empty) -- which is the whole reason it can be
    compared across a process boundary at all. ``results_io`` writes ``None`` as
    the empty string and reads the empty string back as ``None``, so that pair
    round-trips, and every other value goes through ``str`` exactly as ``csv``
    would write it.

    Truncated to 16 hex characters: the collision risk against a few dozen rows
    per root is nil, and a full digest makes the ledger unreadable in a diff.
    """
    payload = "\x1f".join(
        "" if row.get(name) is None else str(row.get(name))
        for name in schema.RESULTS_FIELDNAMES
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


#: `[2026-08-23]` The columns that, with ``(execution_mode, seq_len)``, identify
#: a MODEL row (schema v3, doc 56 section 3.6). A layer row's key is the pair
#: alone -- unchanged, so every recorded ledger keeps meaning what it meant.
MODEL_KEY_FIELDS: tuple[str, ...] = (
    "measurement_scope",
    "model_id",
    "phase",
    "ubatch_tokens",
    "context_end_tokens",
    "precision_plan_id",
)


def model_key(row: dict) -> tuple[str, ...] | None:
    """The MODEL_KEY_FIELDS of a model-scope row as strings; None for a layer row.

    Integers are normalised through ``int(float())`` for ``row_key``'s reason:
    a CSV hands back ``"512"`` where memory holds ``512``.
    """
    if row.get("measurement_scope") != "model":
        return None
    out = []
    for name in MODEL_KEY_FIELDS:
        value = row.get(name)
        try:
            value = int(float(value))
        except (TypeError, ValueError):
            pass
        out.append("" if value is None else str(value))
    return tuple(out)


def row_key(row: dict) -> tuple:
    """``(execution_mode, seq_len)`` -- the rung a row belongs to -- extended by
    MODEL_KEY_FIELDS for a model-scope row.

    ``seq_len`` is coerced because a row read from a CSV carries it as a string
    and a row built in memory carries an int; keying on the raw value would make
    the same rung two different keys depending on which side of a file it came
    from, and the reuse lookup would silently miss every time.

    A layer row keys on the pair alone, exactly as before v3. A model row
    appends ``model_key(row)``: three decode rows at seq_len 1 and three
    contexts are three rungs, and the pair would fold them into one.
    """
    try:
        seq = int(float(row.get("seq_len")))
    except (TypeError, ValueError):
        seq = -1
    key = (str(row.get("execution_mode")), seq)
    extra = model_key(row)
    return key if extra is None else key + extra


def rung_key(mode: str, seq: int, extra=None) -> tuple:
    """The key of a rung named by its CODE mode, as ``row_key`` would key its row."""
    key = (schema.EXECUTION_MODE_CSV.get(mode, mode), int(seq))
    return key if extra is None else key + tuple(extra)


@dataclass(frozen=True)
class PriorRow:
    """One row already on disk, with the evidence needed to reuse it."""

    row: dict
    digest: str
    status: str


@dataclass
class PriorWalk:
    """What a results root already holds. ``unreadable`` is not an empty root."""

    rows: dict[tuple[str, int], PriorRow] = field(default_factory=dict)
    unreadable: dict[str, str] = field(default_factory=dict)

    def get(self, mode: str, seq: int, extra=None) -> PriorRow | None:
        return self.rows.get(rung_key(mode, seq, extra))


def scan(results_root, expected_files: list[str]) -> PriorWalk:
    """Index the rows a results root already holds, by rung.

    A file that exists and cannot be read as the current schema is recorded in
    ``unreadable`` rather than treated as absent. The difference matters: absent
    means "re-measure it", unreadable means "something is here that this reader
    does not understand", and silently re-measuring over a v1 CSV would destroy
    a recorded walk to make a gate green.
    """
    root = Path(results_root)
    prior = PriorWalk()
    for rel in expected_files:
        path = root / rel
        if not path.is_file():
            continue
        try:
            rows = results_io.read_rows(path)
        except Exception as exc:
            prior.unreadable[rel] = f"{type(exc).__name__}: {exc}"
            continue
        for row in rows:
            prior.rows[row_key(row)] = PriorRow(
                row=row,
                digest=row_digest(row),
                status=str(row.get("run_status") or ""),
            )
    return prior


@dataclass(frozen=True)
class ResumePlan:
    """Per rung: carry it forward, or walk it. Keyed by the CODE mode name."""

    #: ``(code_mode, seq) -> (row, digest)`` handed to ``run_ladder.walk``. A
    #: model rung's key is ``(code_mode, seq, *MODEL_KEY_FIELDS values)``.
    reuse: dict[tuple, tuple[dict, str]]
    #: the keys the walk must actually attempt.
    remeasure: tuple[tuple, ...]
    #: the keys the profile's applicability rule refuses.
    skipped: tuple[tuple, ...]
    #: Why each rung was not reused, for the plan printout. Never silent.
    reasons: dict[tuple, str]

    @property
    def reuse_for_walk(self) -> dict[tuple, dict]:
        return {key: row for key, (row, _) in self.reuse.items()}


def plan(profile, prior: PriorWalk, *, enabled: bool = True) -> ResumePlan:
    """Decide every rung. ``enabled=False`` plans a full fresh walk.

    Skips are decided FIRST and are never reused: the applicability rule is the
    profile's current claim about what a mode supports, and a rung the profile
    now refuses must not be resurrected from a row measured when it did not.
    """
    reuse: dict[tuple[str, int], tuple[dict, str]] = {}
    remeasure: list[tuple[str, int]] = []
    skipped: list[tuple[str, int]] = []
    reasons: dict[tuple[str, int], str] = {}

    for rung in profile.rungs():
        # A model rung (model_profiles.ModelRung) carries the v3 key columns as
        # `extra`; a layer rung has none and keys on the pair, as always.
        extra = getattr(rung, "extra", None)
        key = (rung.mode, rung.seq) if extra is None else (rung.mode, rung.seq) + tuple(extra)
        if rung.skip_reason:
            skipped.append(key)
            continue
        if not enabled:
            remeasure.append(key)
            reasons[key] = "not a resume: every rung is walked"
            continue
        found = prior.get(rung.mode, rung.seq, extra)
        if found is None:
            remeasure.append(key)
            reasons[key] = "no row on disk for this rung"
        elif found.status in REUSABLE_STATUSES:
            reuse[key] = (found.row, found.digest)
        else:
            remeasure.append(key)
            reasons[key] = (
                f"the recorded row is `{found.status}`, which is re-run rather "
                "than carried forward -- a retained failure is a claim about "
                "code that may no longer be there"
            )
    return ResumePlan(
        reuse=reuse,
        remeasure=tuple(remeasure),
        skipped=tuple(skipped),
        reasons=reasons,
    )


def git_sha(repo) -> str | None:
    """HEAD, best effort. ``_git``'s rule: imperfect provenance beats none."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def toolchain_fingerprint(block: dict | None) -> str:
    """The toolchain identity fields as one comparable string.

    One string rather than four compared fields because a splice check asks a
    yes/no question -- did the toolchain move between sessions -- and
    ``schema.toolchain_differences`` already exists for the question of WHICH
    field moved, between roots. Unknowns are carried verbatim so two sessions
    that both failed to probe do not compare equal to a real match by accident;
    they compare equal to each other, which is correct and is why the splice
    check below ignores any comparison involving an unknown.
    """
    if not block:
        return schema.UNKNOWN_CONDITION
    return "|".join(
        schema.normalise_power_mode(block.get(name))
        for name in schema.TOOLCHAIN_IDENTITY_FIELDNAMES
    )


class Ledger:
    """The append-only session record for one results root.

    Flushed after EVERY rung. A batched write would lose exactly the session a
    resume exists to recover from -- the one the machine died in the middle of.
    """

    def __init__(self, path, sessions: list[dict], unreadable: str | None = None):
        self.path = Path(path)
        self.sessions = sessions
        #: Why a ledger file present on disk could not be parsed. Never cleared
        #: by writing a new one: the rows it described are still there.
        self.unreadable = unreadable
        self._open: dict | None = None

    # -- persistence --------------------------------------------------------

    @classmethod
    def load(cls, results_root) -> "Ledger":
        """Read a root's ledger. A missing file is empty; a corrupt one is NOT.

        A corrupt ledger degrades to an empty session list WITH ``unreadable``
        set, which ``walk_block`` turns into an attribution problem. Treating it
        as absent would silently erase the provenance of every row already on
        disk and let the resumed walk report clean.
        """
        path = Path(results_root) / LEDGER_NAME
        if not path.is_file():
            return cls(path, [])
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            sessions = payload["sessions"]
            if not isinstance(sessions, list):
                raise ValueError("`sessions` is not a list")
            for record in sessions:
                # `[2026-08-23]` A ledger written before `model_key` existed
                # names layer rungs only; it reads back as exactly that, and
                # the field is filled with None -- "a layer rung", which is the
                # only thing a rung recorded then could have been.
                for rung in record.get("rungs") or []:
                    if isinstance(rung, dict):
                        rung.setdefault("model_key", None)
                schema.validate_session(record)
        except Exception as exc:
            return cls(path, [], unreadable=f"{type(exc).__name__}: {exc}")
        return cls(path, sessions)

    def flush(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"sessions": self.sessions}, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        return self.path

    # -- one session --------------------------------------------------------

    def open_session(self, **facts) -> dict:
        """Start a session, closing out any predecessor that never ended.

        A session still ``running`` when a new one starts did not finish, and
        the new one is the only party in a position to observe that. Relabelling
        it ``interrupted`` is how "the machine rebooted mid-walk" ends up in the
        artifact instead of in somebody's memory.
        """
        for record in self.sessions:
            if record["status"] == "running":
                record["status"] = "interrupted"
        record = {name: None for name in schema.SESSION_FIELDNAMES}
        record.update(facts)
        record["session_id"] = f"s{len(self.sessions) + 1:03d}"
        record["status"] = "running"
        record["rungs"] = []
        schema.validate_session(record)
        self.sessions.append(record)
        self._open = record
        self.flush()
        return record

    def record_rung(self, mode: str, seq: int, row: dict, source: str) -> None:
        """Attribute one rung and flush. Called from the walker's per-rung hook."""
        if self._open is None:
            raise RuntimeError("record_rung before open_session")
        if source not in schema.RUNG_SOURCES:
            raise ValueError(f"rung source {source!r} not in {schema.RUNG_SOURCES}")
        extra = model_key(row)
        self._open["rungs"].append(
            {
                "execution_mode": schema.EXECUTION_MODE_CSV.get(mode, mode),
                "seq_len": int(seq),
                "model_key": None if extra is None else list(extra),
                "source": source,
                "run_status": row.get("run_status"),
                "row_digest": row_digest(row),
            }
        )
        self.flush()

    def close_session(self) -> None:
        if self._open is None:
            return
        self._open["status"] = "complete"
        self._open["ended_utc"] = datetime.now(timezone.utc).isoformat()
        schema.validate_session(self._open)
        self.flush()
        self._open = None


def _measuring_sessions(sessions: list[dict]) -> list[dict]:
    """Sessions that actually measured something. A pure-reuse session's
    conditions describe nothing and must not create a phantom splice."""
    return [
        s for s in sessions if any(r["source"] == "measured" for r in s["rungs"])
    ]


def condition_splices(sessions: list[dict]) -> list[str]:
    """Axes on which the MEASURING sessions disagree, in declaration order.

    An ``unknown`` on either side is never a difference -- ``toolchain_
    differences``' rule -- because "we do not know" and "these differ" are
    different findings and only one of them is evidence.
    """
    measuring = _measuring_sessions(sessions)
    out = []
    for axis, key in (
        ("npu_power_mode", "npu_power_mode"),
        ("toolchain", "toolchain_fingerprint"),
        ("git_sha", "git_sha"),
    ):
        seen = {
            schema.normalise_power_mode(s.get(key))
            for s in measuring
            if schema.normalise_power_mode(s.get(key)) != schema.UNKNOWN_CONDITION
        }
        if len(seen) > 1:
            out.append(axis)
    return out


def walk_block(
    ledger: Ledger,
    prior: PriorWalk,
    *,
    profile=None,
    fidelity: list[str] | None = None,
) -> dict:
    """The manifest's ``walk`` block, audited against the rows on disk.

    ``prior`` must be a scan taken AFTER the walk: this re-hashes what is
    actually in the files and compares it against what the ledger claims, which
    is the only part of this module that can catch the ledger being wrong.

    ``fidelity`` carries problems the runner itself detected in-flight (a rung
    it declared reused whose final row moved). They are merged here rather than
    reported separately so there is one list an operator reads.
    """
    block = schema.empty_walk()
    problems: list[str] = list(fidelity or [])

    if ledger.unreadable is not None:
        problems.append(
            f"{LEDGER_NAME} exists and cannot be read ({ledger.unreadable}). The "
            "rows in this root are still there and are now unattributable: "
            "which session measured which rung is lost. Walk into a fresh "
            "--out-dir rather than repairing it."
        )

    sessions = ledger.sessions
    block["sessions"] = sessions
    block["session_count"] = len(sessions)
    block["walk_source"] = "resumed" if len(sessions) > 1 else "single_session"

    claimed: dict[tuple, tuple[str, str]] = {}
    for record in sessions:
        for rung in record["rungs"]:
            key = (str(rung["execution_mode"]), int(rung["seq_len"]))
            if rung.get("model_key") is not None:
                key = key + tuple(str(v) for v in rung["model_key"])
            if rung["source"] == "measured":
                block["rungs_measured"] += 1
            elif rung["source"] == "reused":
                block["rungs_reused"] += 1
            claimed[key] = (record["session_id"], str(rung["row_digest"]))

    # (a) the ledger claims a rung whose row is not in the files.
    for key, (session_id, _) in sorted(claimed.items()):
        if key not in prior.rows:
            problems.append(
                f"{session_id} claims rung {key[0]} seq {key[1]} and no such row "
                "is in the results CSVs. The ledger describes a walk the files "
                "do not hold."
            )
            continue
        # (b) the row moved since the session that recorded it.
        if prior.rows[key].digest != claimed[key][1]:
            problems.append(
                f"rung {key[0]} seq {key[1]} was recorded by {session_id} with "
                f"digest {claimed[key][1]} and the row on disk hashes to "
                f"{prior.rows[key].digest}. It was re-measured or edited behind "
                "the ledger, so the session it is attributed to did not produce "
                "the row that is there."
            )

    # (c) a row nobody claims. Scoped to the profile's own rungs when one is
    # given, so a root that also holds an unrelated CSV is not indicted for it.
    wanted = None
    if profile is not None:
        wanted = {
            rung_key(r.mode, r.seq, getattr(r, "extra", None)) for r in profile.rungs()
        }
    orphans = sorted(
        key for key in prior.rows if key not in claimed and (wanted is None or key in wanted)
    )
    block["rungs_unattributed"] = len(orphans)
    if orphans:
        problems.append(
            f"{len(orphans)} row(s) belong to no session: "
            f"{', '.join(describe_key(k) for k in orphans[:6])}"
            + (" ..." if len(orphans) > 6 else "")
            + ". A row nobody measured has unknown provenance; the ledger exists "
            "so that there are none of those."
        )

    for rel, why in sorted(prior.unreadable.items()):
        problems.append(
            f"{rel} exists and cannot be read as the current schema ({why}), so "
            "its rows can be neither attributed nor reused."
        )

    splices = condition_splices(sessions)
    block["condition_splices"] = splices
    if "npu_power_mode" in splices:
        # compare_roots REFUSES a pmode mismatch between two roots. Inside one
        # CSV it is strictly worse: there is not even a root boundary to warn a
        # reader that two populations were joined.
        modes = sorted(
            {
                schema.normalise_power_mode(s.get("npu_power_mode"))
                for s in _measuring_sessions(sessions)
            }
        )
        problems.append(
            f"this root's rows were measured at more than one NPU power mode "
            f"({', '.join(modes)}). At `Default` this host measures ~15-20x slow "
            "(README trap 0), so these rows are not one measurement and must not "
            "be read as a curve. Re-walk into a fresh root."
        )

    block["attribution_problems"] = problems
    flagged = [a for a in splices if a != "npu_power_mode"]
    block["walk_detail"] = (
        f"{len(sessions)} session(s); "
        f"{block['rungs_measured']} measured, {block['rungs_reused']} reused"
        + (
            f". FLAGGED: the measuring sessions differ on {', '.join(flagged)} -- "
            "resuming across a commit or a rebuild is normal and is not refused, "
            "but these rows were not all produced by one tree"
            if flagged
            else ""
        )
    )
    return block


def fidelity_problems(
    plan_reuse: dict[tuple[str, int], tuple[dict, str]], walked: list[dict]
) -> list[str]:
    """Rungs the plan declared reused whose final row is not the prior row.

    THE CHECK AGAINST A RESUME THAT SILENTLY REDOES WORK. It cannot fire while
    the walker honours ``reuse``; it fires the moment one does not, which is the
    regression a bookkeeping-only resume could never see. Doc 14's lesson, in a
    different tier: "a halt must record WHERE it halted, or resume throws away
    committed work and makes the tamper check vacuous".
    """
    final = {row_key(row): row for row in walked}
    out = []
    for key, (_, digest) in sorted(plan_reuse.items()):
        mode, seq, *extra = key
        csv_key = rung_key(mode, seq, extra or None)
        row = final.get(csv_key)
        if row is None:
            out.append(
                f"rung {describe_key(csv_key)} was planned as REUSED and is absent "
                "from the walk's output; a carried-forward row must still be "
                "written into its mode's CSV."
            )
        elif row_digest(row) != digest:
            out.append(
                f"rung {describe_key(csv_key)} was planned as REUSED and its row "
                f"changed ({digest} -> {row_digest(row)}). The walker re-ran a "
                "rung the plan said it would carry forward, so the run's own "
                "account of what it measured is wrong."
            )
    return out


def describe_key(key: tuple) -> str:
    """A rung key as prose: ``hybrid seq 512`` for a layer rung, the model
    columns appended for a model rung."""
    text = f"{key[0]} seq {key[1]}"
    if len(key) > 2:
        text += " " + " ".join(f"{n}={v}" for n, v in zip(MODEL_KEY_FIELDS, key[2:]))
    return text


def describe(plan_: ResumePlan) -> list[str]:
    """The plan as lines. Every rung named, because a silent reuse is the trap."""
    lines = []

    def name(key):
        extra = " " + " ".join(str(v) for v in key[2:]) if len(key) > 2 else ""
        return f"{key[0]:<9} seq {key[1]:<6}{extra}"

    for key, (_, digest) in sorted(plan_.reuse.items()):
        lines.append(f"REUSE  {name(key)} row {digest}")
    for key in plan_.remeasure:
        lines.append(f"WALK   {name(key)} {plan_.reasons.get(key, 'planned')}")
    for key in plan_.skipped:
        lines.append(f"SKIP   {name(key)} the profile refuses it")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results_root")
    ap.add_argument("--profile", required=True)
    ap.add_argument(
        "--audit",
        action="store_true",
        help="check the ledger against the rows on disk rather than planning",
    )
    args = ap.parse_args(argv)

    import profiles

    prof = profiles.profile(args.profile)
    prior = scan(args.results_root, prof.expected_files())

    if args.audit:
        block = walk_block(Ledger.load(args.results_root), prior, profile=prof)
        print(f"[resume] {block['walk_detail']}")
        for record in block["sessions"]:
            print(
                f"[resume]   {record['session_id']} {record['status']:<11} "
                f"{record['profile']}  {len(record['rungs'])} rung(s)  "
                f"pmode {record['npu_power_mode']}  job {record['devq_job_id']}"
            )
        for problem in block["attribution_problems"]:
            print(f"[resume]   PROBLEM {problem}")
        return 1 if block["attribution_problems"] else 0

    plan_ = plan(prof, prior)
    for line in describe(plan_):
        print(f"[resume] {line}")
    print(
        f"[resume] {len(plan_.reuse)} reusable, {len(plan_.remeasure)} to walk, "
        f"{len(plan_.skipped)} skipped"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
