# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Named suite profiles: the rungs a run walks, and the counts its manifest must meet.

    python3 study/profiles.py --list
    python3 study/profiles.py --profile ladder            # the plan, rung by rung
    python3 study/profiles.py --profile ladder --expect   # the manifest expectation

CONTRACT
    A ``Profile`` is a name, a set of execution modes, a set of sequence
    lengths, and the one model family the runner can reach. From those four
    things everything else is DERIVED and nothing is restated: the rung list,
    the expected CSV names, and the per-CSV row counts a manifest validates
    against. ``expected_files`` and ``expected_rows`` are what
    ``run_profile.py`` hands to ``smoke_gate`` and ``manifest``.

    Doc 10 §Job counts: "Do not hard-code iron's 888/834/21/3 counts as
    acceptance criteria ... Generate expected counts from the checked-in suite
    profile and validate them in the manifest." That is this module plus
    ``manifest.expected_rows``; the counts below are computed from the tables,
    never typed out, so retargeting a profile retargets its gate.

WHAT THIS PROFILE CANNOT REACH, SAID OUT LOUD
    ``cases.py`` declares six families x a nine-point ladder. The runner can
    reach **one** family. ``run_mode.run`` writes ``workload_variant =
    "encoder_bert"`` unconditionally and ``run_mode._shape_for`` varies only
    ``seq_len``, building the case key as ``f"{seq_len}x{emb}_encoder_bert"``
    from the SPECS row's own ``emb_dim`` -- and every whole-layer SPECS row is
    ``emb_dim 768``. So ``baseline_768`` is reachable and the other five are
    not, at any length, by construction rather than by omission.

    Widening that is a coverage sweep, not a profile edit: new registry rows at
    hidden 512/1024 plus a decoder variant. The precedent (C4) cost 504 + 66
    minutes of gate time. Until then ``UNREACHABLE_FAMILIES`` carries each one
    with its reason, ``run_profile`` copies them into the run report, and no
    profile silently presents a one-family walk as a matrix walk. This follows
    ``component_groups.py``, which reports ``0/12`` rather than putting a mode
    total under a group label.

STRUCTURAL SKIPS ARE ``run_status="skipped"``, NOT FAILURES
    ``schema.RUN_STATUSES`` has carried ``"skipped"`` since v1 and nothing had
    ever emitted it, so a rung that CANNOT apply was recorded identically to a
    rung that broke. The existing gate survived that -- it only needs one passed
    row per CSV -- but a COUNT-based gate cannot: 9 expected rows arriving as 3
    passed and 6 failed is indistinguishable from a mode that regressed at every
    length but one.

    So ``skip_reason(mode, seq)`` is the applicability rule, it returns the
    reason as text, and ``run_ladder.walk`` writes a ``skipped`` row carrying it
    without spawning a child process. The reason rides ``failure_message``
    because adding a column is a schema version bump; it is prefixed
    ``skipped:`` so it still reads correctly wherever that field surfaces.

    **Only artifact-backed bounds belong here.** ``fused`` is the one that
    qualifies: ``pattern/fused/fused.py`` is "BOUNDED TO 256..1024" and states
    the mechanism -- its plane-major packing caps the mode at
    ``rows*cols <= 2^20``, which at emb 768 is 1365 rows, and the SPECS row
    itself was moved 4096 -> 1024 on 2026-08-08 after ``make check-fused``
    raised ``plane stride ... over the shim aie.dma_bd cap of 1048576`` before
    reaching the device. 8192 and 16384 are NOT skipped for any mode: cases.py
    is explicit that "which of them a mode can build is a separate question that
    only a run answers", and pre-declaring a failure is how a matrix stops being
    a measurement.

FOOTGUNS
    - **A profile is a plan, not a promise.** ``expected_rows`` says how many
      rows each CSV must hold and how many of them must have passed. It says
      nothing about latency, and a profile completing is not a result.
    - **`full` is not expected to be green today, and that is deliberate.** It
      attempts 64, 128, 8192 and 16384, which no mode has ever been measured at.
      Truncating it to the four points that are known to work would make it a
      synonym for `ladder` and would quietly convert "we have not measured this"
      into "this is not in the matrix". The rungs that CANNOT apply are
      `skipped`; the rungs nobody has tried are RUN, and their refusal messages
      are the result. `smoke` and `ladder` are the profiles to gate on.
    - **Two walks is the standing rule** (README trap 1: a single walk published
      a crossover a second walk refuted). ``run_profile`` walks once; run it
      twice into two roots and compare with ``compare_roots.py``. A profile that
      walked itself twice would hide which walk a row came from.
    - **The ladder profile's cost is dominated by COMPILATION, not measurement**
      -- 631 s cold against 32 s warm for the same 8 rungs (devq job 224). Size a
      window off the cold number.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import cases  # noqa: E402
import schema  # noqa: E402

#: The family every reachable rung runs at. Not a preference -- see the module
#: docstring; ``test_profiles.py`` re-derives it from ``opcheck_specs.py`` and
#: ``run_mode.py`` rather than trusting this line.
REACHABLE_FAMILY = "baseline_768"

#: Why each declared family the runner cannot reach is out of reach. Keyed by
#: ``cases.FAMILY_IDS`` so a family added to the matrix and forgotten here fails
#: a test rather than quietly disappearing from the report.
UNREACHABLE_FAMILIES: dict[str, str] = {
    "tinybert_512": (
        "no whole-layer SPECS row at emb_dim 512, and run_mode._shape_for "
        "varies only seq_len -- needs registry coverage at hidden 512"
    ),
    "baseline_1024": (
        "no whole-layer SPECS row at emb_dim 1024, and run_mode._shape_for "
        "varies only seq_len -- needs registry coverage at hidden 1024"
    ),
    "gpt2_512": (
        "decoder_gpt2: run_mode.run writes workload_variant='encoder_bert' "
        "unconditionally, and there is no emb_dim 512 row either"
    ),
    "gpt2_small_768": (
        "decoder_gpt2: run_mode.run writes workload_variant='encoder_bert' "
        "unconditionally, so the causal-mask path is unreachable from a profile"
    ),
    "gpt2_medium_1024": (
        "decoder_gpt2: run_mode.run writes workload_variant='encoder_bert' "
        "unconditionally, and there is no emb_dim 1024 row either"
    ),
}

#: ``fused``'s supported sequence range, from ``pattern/fused/fused.py``'s own
#: "BOUNDED TO 256..1024" and the packing derivation behind it. Inclusive.
FUSED_SEQ_RANGE: tuple[int, int] = (256, 1024)

#: The four whole-layer execution modes, in taxonomy order. Validated against
#: the schema rather than redeclared -- convention 7 keeps the mode vocabulary
#: in one module and this is not it.
PROFILE_MODES: tuple[str, ...] = ("coarse", "offload", "runlist", "fused")


def skip_reason(mode: str, seq: int) -> str | None:
    """Why ``(mode, seq)`` cannot apply, or ``None`` if it can be attempted.

    Structural inapplicability only. A rung that MIGHT fail is not skipped --
    it is run, and its failure is the result.
    """
    if mode == "fused":
        low, high = FUSED_SEQ_RANGE
        if seq < low or seq > high:
            return (
                f"fused is bounded to {low}..{high} (pattern/fused/fused.py): "
                f"its plane-major packing caps the mode at rows*cols <= 2^20, "
                f"which is {2 ** 20 // 768} rows at emb 768, and the builder "
                f"raises before aircc is reached. seq {seq} is outside it"
            )
    return None


@dataclass(frozen=True)
class Rung:
    """One point of a profile: an execution mode, a family, a length."""

    mode: str
    family: str
    seq: int

    @property
    def csv_name(self) -> str:
        """The CSV ``run_ladder`` writes this rung into: one file per mode."""
        return f"{self.mode}.csv"

    @property
    def skip_reason(self) -> str | None:
        return skip_reason(self.mode, self.seq)


@dataclass(frozen=True)
class Profile:
    """A named plan. Everything a gate needs is derived from these fields."""

    name: str
    description: str
    modes: tuple[str, ...]
    seqs: tuple[int, ...]
    family: str = REACHABLE_FAMILY

    def __post_init__(self) -> None:
        for mode in self.modes:
            # Validates through the schema's mapping; does not redeclare it.
            cases.canonical_execution_mode(mode)
        if self.family not in cases.FAMILY_SPECS:
            raise ValueError(
                f"profile {self.name!r} names family {self.family!r}, which is "
                f"not in the case matrix; known are {list(cases.FAMILY_IDS)}"
            )

    def rungs(self) -> tuple[Rung, ...]:
        """Mode-major, then ladder order -- the order ``run_ladder`` walks."""
        return tuple(
            Rung(mode=mode, family=self.family, seq=seq)
            for mode in self.modes
            for seq in self.seqs
        )

    def expected_files(self) -> list[str]:
        """The CSVs this profile must produce, in mode order. The gate's contract."""
        return [f"{mode}.csv" for mode in self.modes]

    def expected_rows(self) -> dict[str, dict[str, int]]:
        """Per CSV: rows to exist, rows to pass, rows to be skipped.

        ``rows == measured + skipped`` by construction, which is the invariant
        that makes a count-based manifest able to tell an inapplicable rung from
        a broken one.
        """
        expectation: dict[str, dict[str, int]] = {}
        for rung in self.rungs():
            entry = expectation.setdefault(
                rung.csv_name, {"rows": 0, "measured": 0, "skipped": 0}
            )
            entry["rows"] += 1
            entry["skipped" if rung.skip_reason else "measured"] += 1
        return expectation

    def unreachable(self) -> dict[str, str]:
        """The declared families this profile does not walk, with reasons."""
        return {
            fid: UNREACHABLE_FAMILIES.get(fid, "not walked by this profile")
            for fid in cases.FAMILY_IDS
            if fid != self.family
        }

    def summary(self) -> dict[str, object]:
        """The plan as JSON, for the run report. No measurement in it."""
        rungs = self.rungs()
        return {
            "name": self.name,
            "description": self.description,
            "family": self.family,
            "modes": list(self.modes),
            "sequence_lengths": list(self.seqs),
            "rung_count": len(rungs),
            "expected_files": self.expected_files(),
            "expected_rows": self.expected_rows(),
            "skipped_rungs": [
                {
                    "mode": r.mode,
                    "seq": r.seq,
                    "reason": r.skip_reason,
                }
                for r in rungs
                if r.skip_reason
            ],
            "families_not_walked": self.unreachable(),
        }


#: The three profiles, smallest first. ``full`` is the DECLARED ladder over the
#: one reachable family -- deliberately not a six-family claim, and deliberately
#: not truncated to the lengths that are known to work either.
PROFILES: dict[str, Profile] = {
    "smoke": Profile(
        name="smoke",
        description=(
            "every mode once, at the one length all four support. Minutes warm, "
            "~5 min cold. The per-change profile."
        ),
        modes=PROFILE_MODES,
        seqs=(1024,),
    ),
    "ladder": Profile(
        name="ladder",
        description=(
            "the four measured ladder points (J3). ~45 min cold, ~3 min warm. "
            "Walk it twice into two roots and compare -- README trap 1."
        ),
        modes=PROFILE_MODES,
        seqs=(512, 1024, 2048, 4096),
    ),
    "full": Profile(
        name="full",
        description=(
            "the whole declared nine-point ladder over the ONE reachable "
            "family. Not a six-family matrix walk and does not claim to be; "
            "the other five families are recorded as unreachable with reasons. "
            "EXPECT IT TO FAIL TODAY -- 64, 128, 8192 and 16384 have never been "
            "measured for any mode, and this profile deliberately attempts them."
        ),
        modes=PROFILE_MODES,
        seqs=cases.SEQUENCE_LADDER,
    ),
}


def profile(name: str) -> Profile:
    """Resolve a profile by name. Raises on an unknown one, never guesses."""
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(
            f"unknown profile {name!r}; known are {sorted(PROFILES)}"
        ) from None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profile", default=None)
    ap.add_argument("--list", action="store_true", help="names and sizes only")
    ap.add_argument(
        "--expect",
        action="store_true",
        help="print the manifest expectation rather than the rung plan",
    )
    args = ap.parse_args(argv)

    if args.list or not args.profile:
        for name, prof in PROFILES.items():
            rungs = prof.rungs()
            skipped = sum(1 for r in rungs if r.skip_reason)
            print(
                f"{name:<8} {len(rungs):>3} rungs "
                f"({len(rungs) - skipped} measured, {skipped} skipped)  "
                f"{prof.description}"
            )
        return 0

    prof = profile(args.profile)

    if args.expect:
        for rel, counts in prof.expected_rows().items():
            print(
                f"[profile] {rel:<14} rows {counts['rows']:>3}  "
                f"measured {counts['measured']:>3}  skipped {counts['skipped']:>3}"
            )
        return 0

    print(f"[profile] {prof.name}: {prof.description}")
    print(
        f"[profile] family {prof.family} ({cases.FAMILY_SPECS[prof.family].display_label})"
    )
    for rung in prof.rungs():
        reason = rung.skip_reason
        verdict = "SKIP" if reason else "run "
        print(
            f"[profile]   {verdict} {rung.mode:<9} seq {rung.seq:<6}"
            + (f"  {reason}" if reason else "")
        )
    for fid, why in prof.unreachable().items():
        print(f"[profile] NOT WALKED {fid}: {why}")
    print(
        f"[profile] {len(prof.rungs())} rung(s) over "
        f"{len(prof.expected_files())} CSV(s); "
        f"schema v{schema.SCHEMA_VERSION}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
