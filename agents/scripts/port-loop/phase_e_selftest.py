#!/usr/bin/env python3
"""Both-directions fixtures for `phase_e_checks.py`, split out at the ~800-line convention.

The same mechanism-versus-catalogue seam `opcheck.py`/`opcheck_specs.py` and
`registry_sweep.py`/`sweep_families.py` already draw: the checks live next door, the synthetic
artifact sets that exercise them live here.

Run it through the module it tests, which is the only supported entry point:

    python3 agents/scripts/port-loop/phase_e_checks.py selftest

Every case builds a conforming four-mode artifact set in a temp directory, applies exactly one
mutation, and asserts the verdict flips. A clause with no failing case is a clause nobody has
shown discriminates -- which is how Phase B shipped a hardware gate that ran no hardware.
"""

import json
import os
import pathlib
import shutil
import tempfile

from phase_e_checks import (
    FFN,
    HEAD_DIM,
    HEADS,
    HID,
    MEAN_KEY,
    MIN_STAGES,
    MODES,
    RTOL,
    SEQ,
    main,
)

def _vec(subs, entries, air, herd, sync, nbytes):
    return {
        "host_submissions_per_layer": subs,
        MEAN_KEY: entries / subs,
        "air_launches_per_elf": air,
        "herd_launches": herd,
        "sync_boundaries": sync,
        "bytes_transferred": nbytes,
    }


def _artifact(operator, vectors, key="4096x768_encoder_bert"):
    # Deep-copy the vectors. Sharing the module-level lists by reference let one case's mutation
    # leak into every later case, so a clause could "fail as expected" for the previous case's
    # reason and never be tested at all -- a selftest that reports green while proving nothing,
    # which is the precise failure mode this file exists to prevent.
    vectors = json.loads(json.dumps(vectors))
    return {
        "operator": operator,
        "shape_key": key,
        "shape": {
            "seq_len": SEQ,
            "emb_dim": HID,
            "ffn_dim": FFN,
            "num_heads": HEADS,
            "head_dim": HEAD_DIM,
        },
        "rtol": RTOL,
        "atol": 1e-1,
        "ref_dtype": "float32",
        "n_elements": SEQ * HID,
        "n_mismatch": 0,
        "passed": True,
        "fault_injected": None,
        "stages": [
            {"name": f"b{i}", "n_elements": SEQ * HID, "n_mismatch": 0, "passed": True}
            for i in range(MIN_STAGES)
        ],
        "dispatch_vectors": vectors,
    }


# Shaped after what the block actually measured, with each mode placed where the taxonomy says it
# belongs. coarse's row IS D2's measurement: 4 / 131 / 12 / 146 / 402 / 202,902,528.
_SELFTEST_VECTORS = {
    "coarse": [
        _vec(1, 2, 6, 10, 9, 80216064),
        _vec(1, 64, 1, 64, 193, 18875904),
        _vec(1, 1, 4, 8, 7, 84934656),
        _vec(1, 64, 1, 64, 193, 18875904),
    ],
    "offload": [_vec(1, 1, 1, 8, 3, 12000000) for _ in range(6)],
    "runlist": [_vec(1, 150, 20, 160, 420, 210000000)],
    "fused": [_vec(1, 3, 30, 150, 11, 190000000)],
}


def _write_set(root, mutate=None):
    """Build a conforming four-mode artifact set, then apply one mutation to it."""
    results = root / "results"
    fault = results / "fault"
    fault.mkdir(parents=True, exist_ok=True)
    stamp = root / ".gate-started"
    stamp.write_text("")

    listing = [{"operator": m, "shape_key": "4096x768_encoder_bert"} for m in MODES]
    data = {m: _artifact(m, _SELFTEST_VECTORS[m]) for m in MODES}
    faults = {}
    for m in MODES:
        fd = json.loads(json.dumps(data[m]))
        fd["fault_injected"] = "input"
        fd["passed"] = False
        fd["n_mismatch"] = 12345
        faults[m] = fd

    if mutate:
        mutate(data, faults, listing, results, fault)

    # Filenames come from the artifact's own operator/shape_key, not from the dict key, so a
    # mutation can add a second artifact for an existing operator.
    for d in data.values():
        (results / f"{d['operator']}__{d['shape_key']}.json").write_text(json.dumps(d))
    for d in faults.values():
        (fault / f"{d['operator']}__{d['shape_key']}.json").write_text(json.dumps(d))

    # Everything must be newer than the stamp on an honest run.
    now = os.path.getmtime(stamp) + 10
    for p in list(results.glob("*.json")) + list(fault.glob("*.json")):
        os.utime(p, (now, now))
    return results, fault, stamp, listing


def _run_set(mutate=None, cmd=None):
    """Build a (possibly mutated) artifact set in a tempdir and run one subcommand against it."""
    root = pathlib.Path(tempfile.mkdtemp(prefix="pl-e-selftest-"))
    try:
        results, fault, stamp, listing = _write_set(root, mutate)
        argv = list(cmd or ["distinguish"])
        argv += [
            "--results", str(results),
            "--fault-results", str(fault),
            "--stamp", str(stamp),
            "--listing", str(_dump(root, listing)),
        ]
        return main(argv)
    finally:
        shutil.rmtree(root, ignore_errors=True)


_OFFLOAD_CMD = ["mode", "--operator", "offload", "--expect-no-aggregation", "--min-submissions", "6"]
# J1's clause: coarse's runlist entries must COLLAPSE. The fixture's coarse row is D2's real
# measurement (131 entries, 128 of them addnorm's row blocking), so the same bound must reject it
# before the change and accept it after -- which is what makes the clause evidence rather than
# decoration. The mode agreeing with the oracle proves nothing here: it agreed at 131 too.
_COLLAPSE_CMD = ["mode", "--operator", "coarse",
                 "--max-field", "runlist_entries", "--max-value", "10"]
_COMPARE_CMD = ["compare", "--left", "runlist", "--right", "coarse",
                "--field", "runlist_entries", "--relation", "gt"]


def _ladder_case(seq_lens, want_pass):
    """The E1 ladder clause has its own fixture: `ffn` artifacts, not mode artifacts."""
    root = pathlib.Path(tempfile.mkdtemp(prefix="pl-e-ladder-"))
    try:
        results = root / "results"
        results.mkdir(parents=True)
        stamp = root / ".gate-started"
        stamp.write_text("")
        listing = []
        for seq in seq_lens:
            key = f"{seq}x768x3072"
            listing.append({"operator": "ffn", "shape_key": key})
            d = {
                "operator": "ffn",
                "shape_key": key,
                "shape": {"seq_len": seq, "emb_dim": HID, "ffn_dim": FFN},
                "rtol": RTOL,
                "atol": 1e-1,
                "ref_dtype": "float32",
                "n_elements": seq * FFN,
                "n_mismatch": 0,
                "passed": True,
                "fault_injected": None,
            }
            (results / f"ffn__{key}.json").write_text(json.dumps(d))
        now = os.path.getmtime(stamp) + 10
        for p in results.glob("*.json"):
            os.utime(p, (now, now))
        rc = main([
            "ladder", "--results", str(results), "--stamp", str(stamp),
            "--listing", str(_dump(root, listing)),
        ])
        return (rc == 0) if want_pass else (rc != 0)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _dump(root, obj):
    p = root / "listing.json"
    p.write_text(json.dumps(obj))
    return p


def selftest():
    """Exercise every clause in both directions. Touches no hardware, writes only to a tempdir."""
    cases = []

    def case(name, mutate, want_pass, cmd=None):
        cases.append((name, mutate, want_pass, cmd))

    case("a conforming four-mode set", None, True)

    # --- J1's collapse clause, both directions -------------------------------------------------
    case("coarse at 131 entries against J1's bound of 10", None, False, _COLLAPSE_CMD)

    def collapse_norms(d, f, l, r, fr):
        """What lifting addnorm's one-trip guard should do: the two 64-entry row-blocked
        normalization points become one launch each, leaving the GEMM boundaries untouched."""
        for artifacts in (d, f):
            for v in artifacts["coarse"]["dispatch_vectors"]:
                if v[MEAN_KEY] == 64.0:
                    v[MEAN_KEY] = 1.0
    case("coarse with the norm dispatches collapsed", collapse_norms, True, _COLLAPSE_CMD)

    def collapse_but_break_numerics(d, f, l, r, fr):
        collapse_norms(d, f, l, r, fr)
        d["coarse"]["stages"][2]["n_mismatch"] = 7
        d["coarse"]["stages"][2]["passed"] = False
    case("collapsed entries but a boundary that mismatches",
         collapse_but_break_numerics, False, _COLLAPSE_CMD)

    def drop_stages(d, f, l, r, fr):
        d["coarse"]["stages"] = d["coarse"]["stages"][:3]
    case("fewer than 8 stages", drop_stages, False)

    def repeat_stage_names(d, f, l, r, fr):
        for st in d["fused"]["stages"]:
            st["name"] = "same"
    case("repeated stage names", repeat_stage_names, False)

    def slice_output(d, f, l, r, fr):
        d["runlist"]["n_elements"] = SEQ * HID // 4
    case("a comparison over a slice", slice_output, False)

    def wrong_shape(d, f, l, r, fr):
        d["offload"]["shape"]["seq_len"] = 512
    case("a mode run at the wrong sequence length", wrong_shape, False)

    def widened_atol(d, f, l, r, fr):
        d["coarse"]["atol"] = 0.5
    case("an atol above the registry ceiling", widened_atol, False)

    def trust_passed(d, f, l, r, fr):
        d["fused"]["n_mismatch"] = 7  # `passed` still True: the verdict is re-derived
    case("mismatches with passed=True", trust_passed, False)

    def no_vectors(d, f, l, r, fr):
        del d["runlist"]["dispatch_vectors"]
    case("no dispatch_vectors", no_vectors, False)

    def fractional_entries(d, f, l, r, fr):
        d["coarse"]["dispatch_vectors"][0][MEAN_KEY] = 2.5
        f["coarse"]["dispatch_vectors"][0][MEAN_KEY] = 2.5
    case("a non-integral entries-per-submission product", fractional_entries, False)

    def zero_bytes(d, f, l, r, fr):
        for v in d["fused"]["dispatch_vectors"]:
            v["bytes_transferred"] = 0
        for v in f["fused"]["dispatch_vectors"]:
            v["bytes_transferred"] = 0
    case("a mode that moved no bytes", zero_bytes, False)

    def fabricated_vector(d, f, l, r, fr):
        # The clean artifact claims numbers the driver's own fault run did not measure.
        d["runlist"]["dispatch_vectors"][0]["sync_boundaries"] = 99999
    case("a clean vector the fault run disagrees with", fabricated_vector, False)

    def passing_fault(d, f, l, r, fr):
        f["coarse"]["passed"] = True
    case("a fault-injected run that passed", passing_fault, False)

    def undeclared(d, f, l, r, fr):
        l[:] = [e for e in l if e["operator"] != "fused"]
    case("an artifact for an undeclared shape", undeclared, False)

    def two_conforming(d, f, l, r, fr):
        extra = json.loads(json.dumps(d["coarse"]))
        extra["shape_key"] = "4096x768_second"
        l.append({"operator": "coarse", "shape_key": "4096x768_second"})
        d["coarse_second"] = extra
        f2 = json.loads(json.dumps(f["coarse"]))
        f2["shape_key"] = "4096x768_second"
        f["coarse_second"] = f2
    case("two conforming results for one mode", two_conforming, False)

    def offload_not_extreme(d, f, l, r, fr):
        v = [_vec(1, 1, 1, 8, 3, 12000000)]
        d["offload"]["dispatch_vectors"] = v
        f["offload"]["dispatch_vectors"] = v
    case("offload not the host-mediated extreme", offload_not_extreme, False)

    def offload_aggregates(d, f, l, r, fr):
        v = [_vec(6, 42, 6, 48, 18, 72000000)]
        d["offload"]["dispatch_vectors"] = v
        f["offload"]["dispatch_vectors"] = v
    case("offload aggregating into a runlist", offload_aggregates, False)

    def runlist_not_finer(d, f, l, r, fr):
        v = [_vec(1, 10, 20, 160, 420, 210000000)]
        d["runlist"]["dispatch_vectors"] = v
        f["runlist"]["dispatch_vectors"] = v
    case("runlist no finer than coarse", runlist_not_finer, False)

    def fused_syncs_as_much(d, f, l, r, fr):
        v = [_vec(1, 3, 30, 150, 500, 190000000)]
        d["fused"]["dispatch_vectors"] = v
        f["fused"]["dispatch_vectors"] = v
    case("fused not removing intermediate sync", fused_syncs_as_much, False)

    def identical_modes(d, f, l, r, fr):
        v = _SELFTEST_VECTORS["coarse"]
        d["fused"]["dispatch_vectors"] = json.loads(json.dumps(v))
        f["fused"]["dispatch_vectors"] = json.loads(json.dumps(v))
    case("two modes with identical vectors", identical_modes, False)

    # E3's clause: offload's defining property, checkable from its own artifact alone.
    case("offload aggregating nothing (E3's own clause)", None, True, _OFFLOAD_CMD)
    case("offload batched into a runlist", offload_aggregates, False, _OFFLOAD_CMD)
    case("offload below its submission floor", offload_not_extreme, False, _OFFLOAD_CMD)

    # E4's clause: the one ordinal comparison that sub-phase owns.
    case("runlist finer than coarse (E4's own clause)", None, True, _COMPARE_CMD)
    case("runlist no finer than coarse, compared directly", runlist_not_finer, False, _COMPARE_CMD)

    failures = 0
    for name, mutate, want_pass, cmd in cases:
        # The mutation callbacks receive (data, faults, listing, results_dir, fault_dir); the
        # two-conforming case adds a second artifact for an existing operator, which is why files
        # are named from each artifact's own operator/shape_key.
        rc = _run_set(mutate, cmd)
        ok = (rc == 0) if want_pass else (rc != 0)
        verdict = "ok" if ok else "WRONG"
        expect = "pass" if want_pass else "fail"
        print(f"  [{verdict:>5}] expected to {expect}: {name}")
        if not ok:
            failures += 1

    # E1's ladder clause, which needs `ffn` artifacts rather than mode artifacts.
    for name, seqs, want_pass in (
        ("the ladder pinned to 4096 alone", [SEQ], False),
        ("no ffn results at all", [], False),
        ("the ladder moved to a second point", [SEQ, 64], True),
    ):
        ok = _ladder_case(seqs, want_pass)
        expect = "pass" if want_pass else "fail"
        print(f"  [{'ok' if ok else 'WRONG':>5}] expected to {expect}: {name}")
        if not ok:
            failures += 1
    total = len(cases) + 3

    print()
    if failures:
        print(f"selftest: {failures} of {total} clause(s) did not behave as specified")
        return 1
    print(f"selftest: {total}/{total} clauses behave in both directions")
    return 0
