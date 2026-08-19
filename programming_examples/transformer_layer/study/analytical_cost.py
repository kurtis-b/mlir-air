# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Price a mapping WITHOUT compiling it. Queue row 31's missing bridge.

    python3 study/analytical_cost.py --self-test
    python3 study/analytical_cost.py --table

WHY THIS MODULE EXISTS
    Doc 53 section 8 found that row 31's selector cannot be assembled from the
    parts the row names, because the two instruments live on opposite sides of
    the compile:

        mapping_space.legality(mapping)   reads a DECLARATION   -- no compile
        balance.demand_matrix(...)        reads a ROUTED ARTIFACT -- compile

    So pricing the legal space by doc 47's instrument means compiling it:
    3,721,772 points is ~90 days, 15,347 (structure, seam) pairs ~9 hours, 428
    structures ~15 minutes. Every route needs a modelled step, and doc 53 left
    the choice to the operator. **The operator chose an analytical model**, so
    this is the declaration-side `Demand -> cost` bridge that did not exist.

THE PROVENANCE RULE, WHICH IS THE WHOLE REASON THIS IS A SEPARATE MODULE
    Doc 47's central property is **1,208 measured / 5 counted / 0 modelled**,
    and it deliberately declined to import iron's cross-toolchain bandwidth
    figure so that ports report `unpriced` rather than ranking against a
    constant wearing a measurement's label. An analytical model introduces the
    first modelled numbers in this project.

    They are kept HERE, in this module's own table, and nothing in this file
    writes into ``balance_ert``. ``balance.py``'s 0-modelled property survives
    unchanged, and a caller can always ask which half a number came from:
    every term carries a ``provenance`` of ``measured``, ``counted`` or
    ``modelled``, and ``Cost.provenance_counts`` reports the split the way doc
    47 reports its own.

WHAT IS MODELLED AND WHAT IS NOT
    The TRAFFIC half is **not** modelled -- it is doc 31a's DRAM-crossing lens
    plus doc 53 section 6's band-serial term, both of which are arithmetic over
    the builder's own constants, and ``test_analytical_cost.py`` reproduces
    every figure in section 6's two tables to the byte, including the 1.30-band
    crossover. That is what makes this a bridge rather than a guess: the byte
    counts are checkable against a hand-derivation that came from measured
    artifacts (devq 338/340).

    The RATE half is where the modelling is. Turning bytes into nanoseconds
    needs a bytes-per-nanosecond per resource, and the honest position is that
    this project has ONE such number from its own hardware -- doc 47's
    back-solve, 655,360 B against a measured 122.81 us -> 5.336 GB/s sustained
    on one shim port (devq 293). Everything else here is derived from it or
    declared ``modelled`` with its reasoning attached.

FOOTGUNS
    - **A cost from this module is not a measurement and must never be written
      into a results CSV.** The schema's latency columns mean "a clock ran".
      ``Cost`` is deliberately not a schema row and has no ``avg_latency_ms``.
    - **The bottleneck is only as good as the resource list.** An argmax over
      three resources cannot name a fourth. ``Cost.terms`` is complete for what
      it models and silent about what it does not, so read ``modelled_fraction``
      before believing a ranking.
    - **Ranking is safer than absolute time.** Two mappings' costs share every
      modelled constant, so their RATIO is far better founded than either
      number. ``rank()`` exists so callers do the comparison this supports.
    - **AND IT MAY ONLY RANK WITHIN ONE DISPATCH STRUCTURE.** This is measured,
      not cautionary: on the recorded four-mode comparison the byte order at
      1024 is ``runlist < fused < coarse < offload`` and the LATENCY order is
      ``fused < coarse < runlist < offload``. ``runlist`` moves the fewest bytes
      and is third slowest, because it pays 24-32 ``hw_context`` loads --
      reconfiguration, the taxonomy's other axis, which this module does not
      model. The bytes-to-time ratio is stable within a mode (`[2026-08-19]`
      coarse 9.89/10.03, runlist 18.44/17.91) and differs between modes, so the
      licence is exactly: same structure, different mapping. See
      ``RECORDED_MODE_POINTS`` and the test that pins it.
    - The traffic model covers the FFN tail (doc 31b's nt1 -> up -> gelu ->
      down -> nt2), which is what ``mapping_space`` enumerates. It is not a
      whole-layer model and ``whole_layer_floor`` is provided only to reproduce
      31a's comparison row.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import mapping_space  # noqa: E402

MEASURED = "measured"
COUNTED = "counted"
MODELLED = "modelled"

#: The one bytes-per-second figure this project measured on its own hardware:
#: doc 47's back-solve closed 655,360 B against 122.81 us on one shim port
#: (devq 293, a build-class job with no dispatch). Everything rate-shaped here
#: is this number or is declared `modelled` beside it.
SHIM_PORT_BYTES_PER_NS = 5.336  # 5.336 GB/s == 5.336 B/ns
SHIM_PORT_SOURCE = "doc 47 back-solve, devq 293: 655360 B / 122.81 us"

#: NPU2's per-column shim budget, from mapping_space (counted, not modelled).
SHIM_MM2S_PER_COLUMN = mapping_space.SHIM_MM2S_PER_COLUMN

#: Rows per band. The design is band-serial: one dispatch per TILE_M rows, the
#: band loop advancing on launch arguments (builders/ffn_resident.py FOOTGUNS).
BAND_ROWS = mapping_space.BAND_ROWS

#: `[2026-08-14]` THE MEASURED BOUND ON WHAT THIS MODULE MAY CLAIM.
#:
#: ``(mode, seq, bytes_transferred, avg_latency_ms)`` from the standing
#: cross-mode comparison (results/postflip-ladder-w1, Turbo, doc 32's post-flip
#: walk). Pinned as literals rather than read from the tree because result roots
#: are gitignored, and a check that silently skips on a fresh clone is the
#: failure mode this project keeps finding.
#:
#: Two things fall out, and both bound the model rather than support it:
#:
#:   1. A traffic-only time UNDER-PREDICTS by 8.5x to 18.4x (`[2026-08-19]`
#:      re-measured; was 9.7x-20.3x on the 08-14 numbers). The layer is
#:      nowhere near shim-bandwidth-bound at these shapes, so an absolute ``ns``
#:      from this module is not a latency and must not be read as one.
#:   2. **Byte order does NOT predict latency order.** By bytes at 1024:
#:      runlist < fused < coarse < offload. By latency: fused < coarse <
#:      runlist < offload. `runlist` moves the FEWEST bytes and is third
#:      slowest, because it pays 24-32 hw_context loads -- reconfiguration,
#:      which is the OTHER axis of the corrected taxonomy (doc 03) and which
#:      this module does not model at all.
#:
#: The ratio is stable WITHIN a mode (`[2026-08-19]` coarse 9.89/10.03,
#: fused 9.61/9.71, runlist 18.44/17.91; offload's 11.97/8.47 spread is the
#: measured signature of its 30 context loads) and differs BETWEEN modes. That is the whole licence
#: this module has: it may rank mappings that share a dispatch structure, and it
#: may not rank across structures that differ in reconfiguration.
#: `[2026-08-19]` REFRESHED from doc 32's re-walk (devq 353, Turbo verified
#: in-job, warmup 2 / samples 5, results/rewalk-doc32-w1 pinned here with w2
#: beside it in the comment): every latency moved DOWN 9-24% from the
#: 2026-08-14 values while the BYTES are identical -- the machine got faster
#: (context-load cost), the traffic did not move, so every under-prediction
#: ratio tightened. Both walks agree on every ordering. w2's avgs:
#: coarse 43.14/78.41, fused 39.11/77.65, offload 101.74/153.24,
#: runlist 73.33/129.98 -- within 6% of w1 at every point.
RECORDED_MODE_POINTS = (
    ("coarse", 512, 22020096, 40.8),
    ("coarse", 1024, 44040192, 82.8),
    ("fused", 512, 21233664, 38.2),
    ("fused", 1024, 42467328, 77.3),
    ("offload", 512, 44040192, 98.8),
    ("offload", 1024, 99090432, 157.3),
    ("runlist", 512, 20447232, 70.7),
    ("runlist", 1024, 40894464, 137.3),
)

#: Doc 31a's per-scope split puts tail-internal removable crossings at
#: 17,301,504 B @512 and 34,603,008 B @1024 -- linear, and identical at both
#: lengths, at this many bytes per row. COUNTED off 31a's table, not modelled.
REMOVABLE_BYTES_PER_ROW = 33792


@dataclass(frozen=True)
class Term:
    """One resource's contribution, with where its number came from."""

    resource: str
    bytes_moved: int
    rate_bytes_per_ns: float
    provenance: str
    why: str

    @property
    def ns(self) -> float:
        return (
            self.bytes_moved / self.rate_bytes_per_ns
            if self.rate_bytes_per_ns
            else float("inf")
        )


@dataclass(frozen=True)
class Cost:
    """A modelled cost, its bottleneck, and an honest provenance split."""

    terms: tuple = ()
    #: Bytes crossing DRAM for one execution of the tail.
    dram_bytes: int = 0
    notes: tuple = field(default_factory=tuple)

    @property
    def ns(self) -> float:
        """The bottleneck's time. Resources overlap, so max, not sum.

        Doc 41's note on Timeloop: the value of a bottleneck model is that its
        argmax NAMES the offending resource, which a sum cannot do.
        """
        return max((t.ns for t in self.terms), default=0.0)

    @property
    def bottleneck(self) -> str:
        if not self.terms:
            return "none"
        return max(self.terms, key=lambda t: t.ns).resource

    @property
    def provenance_counts(self) -> dict:
        out = {MEASURED: 0, COUNTED: 0, MODELLED: 0}
        for t in self.terms:
            out[t.provenance] = out.get(t.provenance, 0) + 1
        return out

    @property
    def modelled_fraction(self) -> float:
        if not self.terms:
            return 0.0
        return self.provenance_counts[MODELLED] / len(self.terms)


# ---------------------------------------------------------------------------
# Traffic. NOT modelled -- doc 31a's lens and doc 53 section 6's band term.
# ---------------------------------------------------------------------------


def weight_bytes(emb: int, ffn: int, itemsize: int = 2) -> int:
    """``w_up`` + ``w_down`` for one fetch of both. 9,437,184 B at 768x3072.

    Doc 53 section 6 measured this off the emitted runtime sequence (devq
    338/340): each band dispatch carries ``%arg1 : memref<2359296xbf16>`` and
    the same for ``%arg2``, so a band fetches BOTH weights whole.
    """
    return 2 * emb * ffn * itemsize


def bands(seq: int) -> int:
    """Band dispatches for a sequence. Band-serial, so ``ceil`` not ``floor``."""
    return -(-seq // BAND_ROWS)


def tail_activation_bytes(seq: int, emb: int, itemsize: int = 2) -> int:
    """The tail's non-weight crossings: three activation planes plus two gammas.

    Derived to match doc 31a's tail floor exactly -- see the test module, which
    pins 11,799,552 @512 and 14,158,848 @1024 against this plus `weight_bytes`.
    """
    return 3 * seq * emb * itemsize + 2 * emb * itemsize


def tail_floor_bytes(seq: int, emb: int, ffn: int, itemsize: int = 2) -> int:
    """Doc 31a's tail floor: every static weight counted ONCE per layer."""
    return weight_bytes(emb, ffn, itemsize) + tail_activation_bytes(seq, emb, itemsize)


def tail_band_serial_bytes(seq: int, emb: int, ffn: int, itemsize: int = 2) -> int:
    """The same tail, fetching both weights once PER BAND. Doc 53 section 6."""
    return bands(seq) * weight_bytes(emb, ffn, itemsize) + tail_activation_bytes(
        seq, emb, itemsize
    )


def residency_net_bytes(seq: int, emb: int, ffn: int, itemsize: int = 2) -> int:
    """What band-serial residency nets against packaged. NEGATIVE is worse.

    ``removed - added``: doc 31a's tail-internal removable crossings against
    doc 53 section 6's extra weight fetches. Reproduces -48,758,784 @512 and
    -106,954,752 @1024.
    """
    removed = REMOVABLE_BYTES_PER_ROW * seq
    added = (bands(seq) - 1) * weight_bytes(emb, ffn, itemsize)
    return removed - added


def residency_crossover_rows(emb: int, ffn: int, itemsize: int = 2) -> float:
    """The sequence length where residency stops paying for itself.

    ``33792*S = W*(S/64 - 1)``. Doc 53 section 6 gets 83 rows / 1.30 bands at
    768x3072, and the test module pins that this function agrees.
    """
    w = weight_bytes(emb, ffn, itemsize)
    denom = w / BAND_ROWS - REMOVABLE_BYTES_PER_ROW
    if denom <= 0:
        return float("inf")
    return w / denom


def whole_layer_floor_bytes(seq: int) -> int:
    """Doc 31a's derived packaged-layer floor, for section 6's comparison row.

    COUNTED off 31a rather than re-derived: this module models the tail, and
    the layer figure exists here only so `tail alone is 1.52x / 1.77x the
    packaged LAYER` stays checkable in one place.
    """
    table = {512: 51121152, 1024: 88083456}
    if seq not in table:
        raise ValueError(
            f"whole_layer_floor_bytes has 31a's figure for {sorted(table)} only; "
            f"asked for {seq}. Re-derive from 31a rather than interpolating -- "
            "the floor is not linear in seq (weights do not scale with it)."
        )
    return table[seq]


# ---------------------------------------------------------------------------
# Cost. The rate half, where the modelling is.
# ---------------------------------------------------------------------------


def _shim_rate(parallel_ports: int) -> float:
    """Aggregate shim bytes/ns across ``parallel_ports``.

    MODELLED, and this is the single biggest assumption in the file: it takes
    the one measured single-port figure and scales it LINEARLY with the number
    of ports a mapping occupies. Doc 33's imported memcpy ladder is evidence
    AGAINST perfect linearity -- it is non-monotonic and peaks at 4 shim tiles
    rather than 8 -- so this over-credits wide designs, and it is declared
    rather than hidden. A design that wins here only by occupying more ports
    should be treated as unresolved, not as a winner.
    """
    return SHIM_PORT_BYTES_PER_NS * max(1, parallel_ports)


def cost_for_shape(
    seq: int,
    emb: int,
    ffn: int,
    *,
    itemsize: int = 2,
    band_serial: bool = True,
    shim_ports: int = 1,
) -> Cost:
    """Price one tail at a shape, without a mapping. The traffic-only view."""
    dram = (
        tail_band_serial_bytes(seq, emb, ffn, itemsize)
        if band_serial
        else tail_floor_bytes(seq, emb, ffn, itemsize)
    )
    rate = _shim_rate(shim_ports)
    terms = (
        Term(
            resource="shim_dram",
            bytes_moved=dram,
            rate_bytes_per_ns=rate,
            provenance=MODELLED if shim_ports > 1 else MEASURED,
            why=(
                SHIM_PORT_SOURCE
                if shim_ports == 1
                else f"{SHIM_PORT_SOURCE}, scaled linearly to {shim_ports} ports "
                "-- see _shim_rate, which over-credits wide designs"
            ),
        ),
    )
    notes = []
    if (
        band_serial
        and bands(seq) > residency_crossover_rows(emb, ffn, itemsize) / BAND_ROWS
    ):
        notes.append(
            f"band-serial past the {residency_crossover_rows(emb, ffn, itemsize)/BAND_ROWS:.2f}-band "
            "crossover: residency nets NEGATIVE here (doc 53 section 6)"
        )
    return Cost(terms=terms, dram_bytes=dram, notes=tuple(notes))


def cost(mapping, seq: int, ffn: int | None = None) -> Cost:
    """Price a ``mapping_space.Mapping`` at a sequence length.

    The mapping decides how many shim ports the design occupies -- that is
    ``Demand.shim_mm2s_slots``, which ``mapping_space`` documents as
    placement-INVARIANT and which doc 31b section 3.6 measured as 7 of 16 on
    R1. So the declaration really does carry the one structural quantity the
    rate half needs, which is why this bridge is possible at all.
    """
    ffn = FFN_DIM if ffn is None else ffn
    # The PEAK over time-multiplexed segments, times concurrent band lanes --
    # the same authority the legality check consumes. Reading
    # whole_design(...).shim_mm2s_slots here under-counted Seq compositions
    # (mapping_space.peak_shim_mm2s_slots has the measured case).
    ports = max(1, mapping_space.peak_shim_mm2s_slots(mapping))
    out = cost_for_shape(
        seq, mapping.cols, ffn, itemsize=mapping.itemsize, shim_ports=ports
    )
    return Cost(
        terms=out.terms,
        dram_bytes=out.dram_bytes,
        notes=out.notes
        + (
            f"shim_mm2s_slots {ports} of {mapping_space.NPU2_COLUMNS * SHIM_MM2S_PER_COLUMN}",
        ),
    )


FFN_DIM = mapping_space.FFN_DIM


def rank(mappings, seq: int, ffn: int | None = None):
    """``[(mapping, Cost), ...]`` cheapest first.

    Prefer this to reading an absolute ``ns``: every entry shares the same
    modelled constants, so the ORDER is far better founded than any single
    number. See the module footguns.

    `[2026-08-19]` TWO LIMITS, both pinned in the test suite. This ranks
    DECLARATIONS: legality is a separate question, an over-budget mapping is
    priced (its Cost.notes carry "shim_mm2s_slots N of 16"), and nothing here
    filters -- run ``mapping_space.legality`` first. And within one workload
    the only discriminator is the MODELLED port term: every axis on which two
    buildable-and-runnable designs differ today prices as an exact tie, so a
    rank() order between such designs is a statement about port occupancy
    under the disclaimed linear scaling, never about dispatch structure --
    doc 30's resolved C1/C2 6.6% is invisible to this function.
    """
    priced = [(m, cost(m, seq, ffn)) for m in mappings]
    priced.sort(key=lambda pair: pair[1].ns)
    return priced


# ---------------------------------------------------------------------------


def _self_test():
    """Reproduce doc 53 section 6's two tables to the byte."""
    emb, ffn = 768, 3072
    ok = True

    def check(label, got, want):
        nonlocal ok
        good = got == want
        ok = ok and good
        print(
            f"  [{'PASS' if good else 'FAIL'}] {label}: {got:,}"
            + ("" if good else f" != {want:,}")
        )

    check("weights once", weight_bytes(emb, ffn), 9437184)
    check("weights band-serial @512", bands(512) * weight_bytes(emb, ffn), 75497472)
    check("weights band-serial @1024", bands(1024) * weight_bytes(emb, ffn), 150994944)
    check("tail floor @512", tail_floor_bytes(512, emb, ffn), 11799552)
    check("tail floor @1024", tail_floor_bytes(1024, emb, ffn), 14158848)
    check("tail band-serial @512", tail_band_serial_bytes(512, emb, ffn), 77859840)
    check("tail band-serial @1024", tail_band_serial_bytes(1024, emb, ffn), 155716608)
    check("net residency @512", residency_net_bytes(512, emb, ffn), -48758784)
    check("net residency @1024", residency_net_bytes(1024, emb, ffn), -106954752)

    rows = residency_crossover_rows(emb, ffn)
    good = abs(rows - 83.0) < 1.0 and abs(rows / BAND_ROWS - 1.30) < 0.01
    ok = ok and good
    print(
        f"  [{'PASS' if good else 'FAIL'}] crossover: {rows:.1f} rows = {rows/BAND_ROWS:.2f} bands"
    )

    for seq, want in ((512, 1.52), (1024, 1.77)):
        ratio = tail_band_serial_bytes(seq, emb, ffn) / whole_layer_floor_bytes(seq)
        good = abs(ratio - want) < 0.005
        ok = ok and good
        print(
            f"  [{'PASS' if good else 'FAIL'}] tail vs packaged layer @{seq}: {ratio:.2f}x"
        )

    print("analytical-cost self-test: " + ("All PASSED." if ok else "FAILED."))
    return 0 if ok else 1


def _main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    ap.add_argument("--table", action="store_true", help="print the traffic table")
    args = ap.parse_args(argv)
    if args.table:
        emb, ffn = 768, 3072
        print(
            f"{'seq':>6} {'bands':>6} {'floor B':>14} {'band-serial B':>16} {'net B':>14}"
        )
        for seq in (64, 128, 256, 512, 1024, 2048, 4096):
            print(
                f"{seq:>6} {bands(seq):>6} {tail_floor_bytes(seq,emb,ffn):>14,} "
                f"{tail_band_serial_bytes(seq,emb,ffn):>16,} {residency_net_bytes(seq,emb,ffn):>14,}"
            )
        return 0
    return _self_test()


if __name__ == "__main__":
    sys.exit(_main())
