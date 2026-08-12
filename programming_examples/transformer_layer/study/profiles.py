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

WHAT THIS PROFILE CAN AND CANNOT REACH, SAID OUT LOUD
    ``cases.py`` declares six families x a nine-point ladder. The runner reaches
    **three**: the encoder widths. The three decoder families are refused with a
    reason rather than omitted.

    `[2026-08-12]` THE PREVIOUS TEXT HERE WAS WRONG, AND ITS WRONGNESS COST THE
    SCOPING OF A WHOLE PHASE. It said ``tinybert_512`` and ``baseline_1024``
    "need registry coverage at hidden 512/1024" and that widening the matrix was
    a Phase-C-sized sweep -- doc 34 M5 sized it against C4's 504 + 66 minutes.
    Measured instead of assumed: ``kernel_registry/details/
    GEMM_bf16_in_bf16_out.json`` holds **36 of 36** projection triples for each
    of hidden 512, 768 and 1024, landed 2026-08-07 by the baseline_512 and
    baseline_1024 sweeps (69 -> 103 -> 136 rows). The sweep the estimate was
    sized against **had already been run**, and nothing since read the file.

    So the real blocker was never coverage, it was that ``run_mode._shape_for``
    overrode ``seq_len`` and not the width. That is now a parameter, and the
    cost of two more families is zero device-hours of sweeping.

    The decoder half of the old text stands and is unchanged: ``decoder_gpt2``
    is a different LAYER GRAPH, not a flag -- see
    ``run_mode.UNBUILDABLE_VARIANTS``, which refuses it. Refusing matters more
    than omitting here: overriding the width alone and stamping the row
    ``decoder_gpt2`` would produce a valid-looking bidirectional measurement
    under a causal name, and nothing downstream could detect it.

    ``UNREACHABLE_FAMILIES`` carries each unreachable family with its reason,
    ``run_profile`` copies them into the run report, and no profile silently
    presents a partial walk as a matrix walk -- following
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
from dataclasses import dataclass, replace

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import cases  # noqa: E402
import schema  # noqa: E402

#: The default family, kept as the name every existing caller and every recorded
#: root already uses. NOT "the only one" any more -- see REACHABLE_FAMILIES.
REACHABLE_FAMILY = "baseline_768"

#: Every family a profile can walk today. Derived from the two things that
#: actually decide it -- the layer graph must be buildable, and every projection
#: GEMM must resolve -- rather than listed; ``test_profiles.py`` re-derives both
#: halves from the sources, so a family that stops resolving fails a test rather
#: than making this line a lie.
REACHABLE_FAMILIES: tuple[str, ...] = (
    "tinybert_512",
    "baseline_768",
    "baseline_1024",
)

#: Why each declared family the runner cannot reach is out of reach. Keyed by
#: ``cases.FAMILY_IDS`` so a family added to the matrix and forgotten here fails
#: a test rather than quietly disappearing from the report.
#:
#: `[2026-08-12]` All three surviving entries are the SAME reason, and it is a
#: layer-graph reason rather than a coverage one. The two width entries that
#: used to sit here ("needs registry coverage at hidden 512/1024") were false:
#: that coverage landed 2026-08-07 and nothing had read the file since.
UNREACHABLE_FAMILIES: dict[str, str] = {
    "gpt2_512": (
        "decoder_gpt2 is a different layer graph, not a width -- see "
        "run_mode.UNBUILDABLE_VARIANTS. Its width (512) IS reachable, so this "
        "family is one graph away rather than one sweep away"
    ),
    "gpt2_small_768": (
        "decoder_gpt2 is a different layer graph, not a width -- see "
        "run_mode.UNBUILDABLE_VARIANTS. Its width is the default 768, so this "
        "is the cheapest of the three decoders: the graph is the whole cost"
    ),
    "gpt2_medium_1024": (
        "decoder_gpt2 is a different layer graph, not a width -- see "
        "run_mode.UNBUILDABLE_VARIANTS. Its width (1024) is reachable, so like "
        "the other two decoders the graph is the whole cost"
    ),
}

#: Which METHOD each projection role's owning consumer pins, keyed by the
#: ``(K multiple, N multiple)`` of the family's hidden size. Full TRIPLE coverage
#: is not full METHOD coverage, and this is the table that says which of the two
#: a walk actually needs.
#:
#: The value is ``(method, the module that pins it)``. Both pins are real and
#: neither is a preference:
#:
#:   - ``qkv_proj`` resolves ``(seq, h, 3h)`` with ``method=SCRATCH_METHOD`` and
#:     raises if it is absent -- its three-way C split rides fused-cast's
#:     separate cast launch, so no other method substitutes.
#:   - ``offload``'s shared-xclbin chain re-resolves any ``fused-cast`` winner to
#:     ``drain`` (``_chain_spec``), because a two-launch module faults the
#:     firmware's in-stream ``load_pdi``. That pin applies to the three shapes
#:     its chain resolves -- ``proj`` ``(seq, h, h)``, ``up`` ``(seq, h, 4h)``
#:     and ``down`` ``(seq, 4h, h)`` -- and to no others.
#:
#: ``test_profiles.py`` re-derives both pins from those two modules by ``ast``
#: and checks this table against them, then checks the registry against this
#: table for every reachable family at every ladder point. Three sources, so a
#: pin that moves fails a test rather than making this comment a lie.
PINNED_PROJECTION_METHODS: dict[tuple[int, int], tuple[str, str]] = {
    (1, 3): ("fused-cast", "builders/qkv_proj.py::SCRATCH_METHOD"),
    (1, 1): ("drain", "pattern/offload/offload.py::_chain_spec"),
    (1, 4): ("drain", "pattern/offload/offload.py::_chain_spec"),
    (4, 1): ("drain", "pattern/offload/offload.py::_chain_spec"),
}

#: Known holes in the registry that make a REACHABLE family's rung fail. Not
#: skips: a skip is a claim about what a MODE supports, and this is a missing
#: measurement -- so the rung is run and its refusal is the result, which is
#: `cases.py`'s rule that "pre-declaring a failure is how a matrix stops being a
#: measurement". Recorded here so an operator reading an incomplete walk finds
#: the cause in one line instead of in a traceback.
#:
#: `[2026-08-12]` EMPTY, AND THE ONE ENTRY IT HELD WAS NEVER A GAP. G1 recorded
#: ``2048x1024x3072`` / ``drain`` against ``offload`` and ``runlist``, and the
#: registry fact was true -- that triple has ``direct`` and ``fused-cast`` and no
#: ``drain`` row. The CONSEQUENCE was false. Neither mode resolves a ``3h`` shape
#: at all: ``offload``'s chain is ``proj`` ``(seq, h, h)``, ``up``
#: ``(seq, h, 4h)`` and ``down`` ``(seq, 4h, h)``, and ``runlist``'s is the same
#: three. The only consumer of ``(seq, h, 3h)`` is ``qkv_proj``, which pins
#: ``fused-cast`` -- present. So nothing anywhere asks for ``drain`` at that
#: triple, and ``baseline_1024`` was 36/36 the whole time rather than 35/36.
#:
#: The mis-recording is the same failure the module docstring above records, one
#: level in: G1 corrected M5 by reading the REGISTRY instead of
#: ``opcheck_specs.py``, and then modelled ``offload``'s drain pin against the
#: qkv shape, which ``offload`` never resolves. A re-derivation is only as good
#: as its choice of source, and that applies to the fix as well as to the bug.
#:
#: So the test no longer only asks "is this method still missing". A missing
#: method is not a gap unless something DEMANDS it, and the demand is
#: ``PINNED_PROJECTION_METHODS`` above. An entry that names a (triple, method)
#: nothing pins now fails -- which is what the deleted entry would do.
KNOWN_REGISTRY_GAPS: tuple[dict, ...] = ()

#: ``fused``'s packing cap, from ``builders/norm_tail.py``: plane-major staging
#: needs a plane stride of ``rows*cols``, and the shim ``aie.dma_bd`` field caps
#: it at 2^20. This is the NUMBER the mode's "BOUNDED TO 256..1024" was derived
#: from at emb 768, and carrying the derivation rather than its answer is what
#: makes the bound follow a width change.
FUSED_PLANE_STRIDE_CAP = 2**20

#: The bottom of ``fused``'s range, which is a tiling floor rather than a
#: packing one and therefore does NOT move with width.
FUSED_SEQ_MIN = 256


def fused_seq_range(family: str = REACHABLE_FAMILY) -> tuple[int, int]:
    """``fused``'s supported (min, max) sequence at a family's width. Inclusive.

    DERIVED, not tabulated. At emb 768 this returns (256, 1024), which is the
    range ``fused.py`` states in prose and which the previous hardcoded
    ``FUSED_SEQ_RANGE`` carried -- so the 768 profiles are unchanged. At emb 512
    it returns (256, 2048) and at emb 1024 (256, 1024), because the cap is on
    ``rows*cols`` and cols is the width. A tabulated bound would have been right
    at one width and silently wrong at the other two, which is the same class of
    error as the row this module's docstring records: a number that was correct
    when written and became a claim about somebody else's shape.
    """
    import cases

    emb = cases.FAMILY_SPECS[family].hidden_size
    # The largest power-of-two ladder point whose plane fits.
    return (FUSED_SEQ_MIN, max(s for s in cases.SEQUENCE_LADDER if s * emb <= FUSED_PLANE_STRIDE_CAP))

#: The four whole-layer execution modes, in taxonomy order. Validated against
#: the schema rather than redeclared -- convention 7 keeps the mode vocabulary
#: in one module and this is not it.
PROFILE_MODES: tuple[str, ...] = ("coarse", "offload", "runlist", "fused")


def skip_reason(mode: str, seq: int, family: str = REACHABLE_FAMILY) -> str | None:
    """Why ``(mode, seq)`` cannot apply, or ``None`` if it can be attempted.

    Structural inapplicability only. A rung that MIGHT fail is not skipped --
    it is run, and its failure is the result.
    """
    if mode == "fused":
        import cases

        emb = cases.FAMILY_SPECS[family].hidden_size
        low, high = fused_seq_range(family)
        if seq < low or seq > high:
            return (
                f"fused is bounded to {low}..{high} at {family} (emb {emb}): "
                f"its plane-major packing caps the mode at rows*cols <= "
                f"{FUSED_PLANE_STRIDE_CAP}, which is "
                f"{FUSED_PLANE_STRIDE_CAP // emb} rows at this width, and the "
                f"builder raises before aircc is reached. seq {seq} is outside it"
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
        return skip_reason(self.mode, self.seq, self.family)


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
        if self.family not in REACHABLE_FAMILIES:
            # Refused at construction rather than walked and failed 36 times.
            # A decoder profile would otherwise cost a full cold walk to learn
            # what UNREACHABLE_FAMILIES already says in one line.
            raise ValueError(
                f"profile {self.name!r} names family {self.family!r}, which no "
                f"whole-layer mode can build: "
                f"{UNREACHABLE_FAMILIES.get(self.family, 'no reason recorded')}"
            )

    def retarget(self, family: str) -> "Profile":
        """The same plan at another family's width, with its own derived counts.

        The point of the profile table's discipline: retargeting retargets the
        gate. ``fused``'s applicability bound moves with the width, so the
        expected skipped-row count for `ladder` at ``tinybert_512`` differs from
        the one at ``baseline_768`` WITHOUT a number being edited anywhere.
        """
        return replace(self, family=family)

    def skip_rule(self):
        """``skip_reason`` bound to THIS profile's family, for the walker.

        The walker takes a two-argument callable and knows nothing about
        families. Handing it the bare module function would silently apply the
        768 bound to a 512 walk -- `fused` at seq 2048 would be skipped where it
        is supported, and a skipped rung is not a failure, so the walk would
        report complete having never attempted it.
        """
        return lambda mode, seq: skip_reason(mode, seq, self.family)

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
