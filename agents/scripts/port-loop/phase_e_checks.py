#!/usr/bin/env python3
"""Driver-side objective checks for Phase E, the four execution strategies.

WHY THIS IS A MODULE AND NOT A HEREDOC IN phases.sh
---------------------------------------------------
Every other phase's objective check is embedded python inside `phases.sh`, and at forty lines
apiece that is the right call. Phase E's are not forty lines: four modes are held to one artifact
contract, and the fourth adds a cross-mode assertion over all of them. Embedded, that would push
`phases.sh` well past the ~800-line convention the plan gates on and -- the part that actually
matters -- it could not be run in both directions.

The harness has been bitten twice by a check nobody exercised. Phase C4 halted on an objective
check no honest run could pass, because only its failure direction had been tried. Phase B passed a
"hardware" gate that ran for sixteen seconds without touching the NPU. `run-one <phase>
hardware-check [gate-log]` exists so `pl_assert_gate_ran_hardware` can be aimed at a known-good and
a known-bad log. This module's `selftest` subcommand is the same idea taken further: it synthesizes
conforming and violating artifact sets in a temp directory and asserts the verdict both ways, for
every clause.

    python3 agents/scripts/port-loop/phase_e_checks.py selftest

That command touches no hardware, writes nothing outside a temp directory, and is the thing to run
after editing anything here.

WHAT IT CHECKS, AND WHAT IT DELIBERATELY DOES NOT
-------------------------------------------------
It does NOT re-derive the numerics contract from scratch -- `phase_c_operator_check` in `phases.sh`
already does that for every declared (operator, shape) and is reused verbatim for each mode. It is
re-checked here anyway on the one artifact this module selects, cheaply, because selecting the
wrong artifact and then trusting it is the failure this module could uniquely introduce.

What it adds on top of `phase_c_operator_check`:

  * FULL-LAYER SCOPE, generalized from D2's block clause to any mode: the forced configuration,
    the whole 4096x768 output, at least eight distinctly-named per-boundary stages with no
    mismatches. A comparison over one tile, or a mode with no per-boundary breakdown, fails here.
  * THE DISPATCH VECTOR CONTRACT: the six `DispatchVector.as_row()` keys, finite, non-negative,
    count fields integer-valued, and `entries_per_submission * host_submissions` exactly integral.
    That last one is not pedantry: the field is a derived MEAN (dispatch.py:166), so a fabricated
    number is most likely to show up as a product that is not a whole number of runlist entries.
  * VECTOR PROVENANCE, which is the only real defence the numbers have. `results/` is gitignored,
    so a hand-written `dispatch_vectors` block is invisible to `guard_fingerprint`,
    `guard_check_tamper`, `guard_check_destructive` and every Codex diff review; freshness alone
    does not stop it. But the DRIVER runs each mode itself under `--fault-inject input` and
    requires that run to fail, and that run writes its own artifact. So the fault artifact's summed
    totals must EQUAL the clean run's. Injection perturbs one input element after the reference
    exists and does not change the dispatch path, so they are identical on an honest run -- the
    block's clean and fault artifacts today both total 4 / 131 / 12 / 146 / 402 / 202,902,528. A
    session cannot know those six numbers without actually dispatching.
  * DISTINGUISHABILITY, the cross-mode assertion Phase E exists to establish. See `distinguish`.

FRESHNESS AT PHASE E5
---------------------
Every clause requires artifacts newer than the gate's start stamp, and it is worth writing down why
that is satisfiable for a cross-mode check running at the LAST sub-phase's gate. Lit enrolment
under `transformer_layer/` is path-based, so the gate runs every mode's `.lit`, and each of those
runs `opcheck.py`, which writes into the SOURCE tree's `results/` regardless of its working
directory. All four modes' artifacts are therefore rewritten by E5's own gate. This is exactly the
assumption C4's objective check got wrong in the other direction, so it is stated rather than
assumed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys

# The forced gate configuration. Identical to PL_D_* in phases.sh, and forced for the same reason:
# a mode quietly run at a smaller sequence or a narrower family would satisfy every numeric clause
# and prove nothing about the case matrix.
SEQ = 4096
HID = 768
FFN = 3072
HEADS = 12
HEAD_DIM = 64
MIN_STAGES = 8

# The canonical bf16 rtol, fixed repository-wide (kernel_registry/README.md), and the loosest atol
# anywhere in the registry. A session cannot widen either; the driver rejects both.
RTOL = 1.6e-2
ATOL_MAX = 1e-1

# `DispatchVector.as_row()` (llms/shared/infra/dispatch.py:172). Five counts and one derived MEAN.
COUNT_KEYS = (
    "host_submissions_per_layer",
    "air_launches_per_elf",
    "herd_launches",
    "sync_boundaries",
    "bytes_transferred",
)
MEAN_KEY = "runlist_entries_per_submission"
VECTOR_KEYS = COUNT_KEYS + (MEAN_KEY,)

# The six fields a mode is summarized by. `runlist_entries` is reconstructed rather than read.
TOTAL_KEYS = (
    "host_submissions",
    "runlist_entries",
    "air_launches",
    "herd_launches",
    "sync_boundaries",
    "bytes_transferred",
)

MODES = ("offload", "runlist", "coarse", "fused")


class CheckError(Exception):
    """A clause failed. The message is what the operator reads in the driver log."""


# --- artifact loading ---------------------------------------------------------------------------


def _load_json(path: pathlib.Path):
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001 - any unreadable artifact is a failure, not a crash
        raise CheckError(f"{path.name} is not readable JSON: {exc}") from exc


def _declared_keys(listing, operator):
    """The shape keys `opcheck.py --list` declares for this operator.

    Only a declared shape counts as evidence. `results/` is gitignored, so an artifact for a shape
    nothing declares is validated by nothing at all -- no fingerprint, no tamper check, no review
    diff sees it. This is D1's rule and it is reused unchanged.
    """
    keys = [e["shape_key"] for e in listing if e.get("operator") == operator]
    if not keys:
        raise CheckError(
            f"opcheck.py --list declares no shape for operator {operator!r}; "
            f"it declares {sorted({e.get('operator') for e in listing})}"
        )
    return keys


def _fresh(path: pathlib.Path, cutoff):
    return cutoff is None or path.stat().st_mtime >= cutoff


def _numerics_problems(d):
    """Re-derive the verdict rather than trusting `passed`. Same contract phases.sh enforces."""
    problems = []
    if d.get("fault_injected") is not None:
        problems.append(f"is a fault-injected run (fault_injected={d['fault_injected']!r})")
    if d.get("ref_dtype") != "float32":
        problems.append(f"ref_dtype={d.get('ref_dtype')!r}, not float32")
    if d.get("n_mismatch") != 0:
        problems.append(f"n_mismatch={d.get('n_mismatch')!r}, not 0")
    try:
        if abs(float(d["rtol"]) - RTOL) > 1e-12:
            problems.append(f"rtol={d['rtol']!r}, not the canonical {RTOL:g}")
    except (KeyError, TypeError, ValueError):
        problems.append("rtol missing or not a number")
    try:
        if float(d["atol"]) > ATOL_MAX:
            problems.append(f"atol={d['atol']!r} exceeds the registry maximum {ATOL_MAX:g}")
    except (KeyError, TypeError, ValueError):
        problems.append("atol missing or not a number")
    return problems


def _scope_problems(d):
    """Full-layer scope and the per-boundary breakdown, generalized from D2's block clause."""
    problems = []
    shape = d.get("shape") or {}
    for key, want in (
        ("seq_len", SEQ),
        ("emb_dim", HID),
        ("ffn_dim", FFN),
        ("num_heads", HEADS),
        ("head_dim", HEAD_DIM),
    ):
        if shape.get(key) != want:
            problems.append(f"shape[{key!r}]={shape.get(key)!r}, not {want!r}")

    want_elems = SEQ * HID
    if d.get("n_elements") != want_elems:
        problems.append(
            f"n_elements={d.get('n_elements')!r}, not the full {SEQ} x {HID} = {want_elems}"
        )

    stages = d.get("stages")
    if not isinstance(stages, list):
        problems.append("no `stages` list: the per-boundary comparison is a work item, not optional")
        return problems
    if len(stages) < MIN_STAGES:
        problems.append(
            f"only {len(stages)} stage(s); at least {MIN_STAGES} operator boundaries "
            "must be compared"
        )
        return problems

    seen = set()
    for i, st in enumerate(stages):
        if not isinstance(st, dict):
            problems.append(f"stage {i} is not an object")
            continue
        name = st.get("name")
        # Distinct, non-empty names. Without this the clause is satisfiable by repeating one
        # trivial entry MIN_STAGES times, which localizes nothing -- and localization is the entire
        # reason the stage list is required.
        if not isinstance(name, str) or not name.strip():
            problems.append(f"stage {i} has no name")
            name = f"#{i}"
        elif name in seen:
            problems.append(f"stage name {name!r} repeats; the boundaries must be distinct")
        else:
            seen.add(name)
        n = st.get("n_elements")
        if not isinstance(n, int) or n < SEQ * HID:
            problems.append(
                f"stage {name}: n_elements={n!r}, smaller than one {SEQ} x {HID} boundary tensor"
            )
        if st.get("n_mismatch") != 0:
            problems.append(f"stage {name}: n_mismatch={st.get('n_mismatch')!r}, not 0")
    return problems


# --- the dispatch vector ------------------------------------------------------------------------


def vector_totals(d, where):
    """Validate `dispatch_vectors` and sum it. The DRIVER does this arithmetic, never the session.

    `runlist_entries_per_submission` is a derived mean (dispatch.py:166), so the total number of
    runlist entries is the sum of `mean * submissions` over the recorded rows -- not the sum of the
    means. For the block's four rows (1, 2.0), (1, 64.0), (1, 1.0), (1, 64.0) that is 131, which is
    the number the plan documents quote.
    """
    vectors = d.get("dispatch_vectors")
    if not isinstance(vectors, list) or not vectors:
        raise CheckError(f"{where}: no non-empty `dispatch_vectors` list")

    totals = {k: 0 for k in TOTAL_KEYS}
    for i, v in enumerate(vectors):
        if not isinstance(v, dict):
            raise CheckError(f"{where}: dispatch_vectors[{i}] is not an object")
        missing = [k for k in VECTOR_KEYS if k not in v]
        if missing:
            raise CheckError(
                f"{where}: dispatch_vectors[{i}] is missing {', '.join(missing)}; "
                "record DispatchVector.as_row() rather than a hand-built dict"
            )
        for k in VECTOR_KEYS:
            x = v[k]
            if isinstance(x, bool) or not isinstance(x, (int, float)):
                raise CheckError(f"{where}: dispatch_vectors[{i}][{k!r}]={x!r} is not a number")
            if not math.isfinite(x):
                raise CheckError(f"{where}: dispatch_vectors[{i}][{k!r}]={x!r} is not finite")
            if x < 0:
                raise CheckError(f"{where}: dispatch_vectors[{i}][{k!r}]={x!r} is negative")
        for k in COUNT_KEYS:
            if float(v[k]) != int(v[k]):
                raise CheckError(
                    f"{where}: dispatch_vectors[{i}][{k!r}]={v[k]!r} is a count and must be a "
                    "whole number"
                )
        subs = int(v["host_submissions_per_layer"])
        if subs < 1:
            raise CheckError(
                f"{where}: dispatch_vectors[{i}] records {subs} host submissions; every recorded "
                "sequence submitted at least once"
            )
        product = float(v[MEAN_KEY]) * subs
        if abs(product - round(product)) > 1e-9:
            raise CheckError(
                f"{where}: dispatch_vectors[{i}] has {v[MEAN_KEY]!r} entries per submission over "
                f"{subs} submission(s) = {product!r}, which is not a whole number of runlist "
                "entries"
            )
        totals["runlist_entries"] += int(round(product))
        totals["host_submissions"] += subs
        totals["air_launches"] += int(v["air_launches_per_elf"])
        totals["herd_launches"] += int(v["herd_launches"])
        totals["sync_boundaries"] += int(v["sync_boundaries"])
        totals["bytes_transferred"] += int(v["bytes_transferred"])

    if totals["bytes_transferred"] <= 0:
        raise CheckError(
            f"{where}: every recorded vector moved zero bytes; nothing was transferred to a device"
        )
    return totals


# --- one mode -----------------------------------------------------------------------------------


def check_mode(operator, results, fault_results, cutoff, listing, quiet=False):
    """Resolve, validate and summarize one mode's artifact pair. Returns its six totals."""
    declared = set(_declared_keys(listing, operator))

    candidates = sorted(results.glob(f"{operator}__*.json"))
    if not candidates:
        raise CheckError(
            f"{operator}: no results/{operator}__*.json at all; the mode produced no artifact"
        )

    conforming, rejected = [], []
    for path in candidates:
        key = path.name[len(operator) + 2 : -len(".json")]
        if key not in declared:
            rejected.append((path.name, [f"shape_key {key!r} is not declared by opcheck.py --list"]))
            continue
        if not _fresh(path, cutoff):
            rejected.append((path.name, ["predates the gate; it was not produced by this run"]))
            continue
        d = _load_json(path)
        problems = _numerics_problems(d) + _scope_problems(d)
        if problems:
            rejected.append((path.name, problems))
        else:
            conforming.append((path, key, d))

    if not conforming:
        lines = [
            f"{operator}: no fresh, declared result at the gate configuration "
            f"(seq_len {SEQ}, emb_dim {HID}, ffn_dim {FFN}, {HEADS} heads) with a full-layer, "
            "per-boundary comparison behind it:"
        ]
        for name, problems in rejected:
            lines.append(f"    {name}: {'; '.join(problems)}")
        raise CheckError("\n".join(lines))

    # Exactly one. Two conforming artifacts would make every downstream cross-mode comparison
    # depend on which one the driver happened to sort first, and that is not a thing to guess at.
    if len(conforming) > 1:
        raise CheckError(
            f"{operator}: {len(conforming)} results conform to the gate configuration "
            f"({', '.join(p.name for p, _, _ in conforming)}); refusing to guess which one the "
            "mode's dispatch vector describes"
        )

    path, key, d = conforming[0]
    clean = vector_totals(d, path.name)

    # Provenance. The fault run is the driver's own, so its vector cannot have been written ahead
    # of time; requiring equality ties the clean numbers to a dispatch that actually happened.
    fault_path = fault_results / f"{operator}__{key}.json"
    if not fault_path.exists():
        raise CheckError(
            f"{operator}: no fault-injected artifact at {fault_path}. The driver re-runs every mode "
            "with --fault-inject input; its artifact is what proves the recorded dispatch vector "
            "came from hardware rather than from a text editor."
        )
    if not _fresh(fault_path, cutoff):
        raise CheckError(f"{operator}: {fault_path.name} predates the gate")
    fd = _load_json(fault_path)
    if fd.get("fault_injected") is None:
        raise CheckError(
            f"{operator}: {fault_path.name} records fault_injected=None; it is not an injected run"
        )
    if fd.get("passed") is not False:
        raise CheckError(
            f"{operator}: {fault_path.name} records passed={fd.get('passed')!r}; the injected run "
            "must fail, or the check does not discriminate"
        )
    fault = vector_totals(fd, fault_path.name)

    differing = [k for k in TOTAL_KEYS if clean[k] != fault[k]]
    if differing:
        detail = ", ".join(f"{k}: clean {clean[k]} vs fault {fault[k]}" for k in differing)
        raise CheckError(
            f"{operator}: the fault-injected run's dispatch vector differs from the clean run's "
            f"({detail}). Injection perturbs one input element after the reference exists and does "
            "not change the dispatch path, so these must agree. Either the instrumentation is "
            "conditional on the injected flag, or one of the two vectors was not measured."
        )

    if not quiet:
        for name, problems in rejected:
            print(f"  ({name} is not the gate configuration, ignored: {problems[0]})")
        print(
            f"  {operator} [{key}]: {d['n_elements']} elements over the full {SEQ}x{HID} layer, "
            f"{len(d['stages'])} distinct clean stage boundaries"
        )
        print(f"    vector {_fmt_totals(clean)}  (fault run agrees)")
    return clean


def _fmt_totals(t):
    return (
        f"submissions {t['host_submissions']}, entries {t['runlist_entries']}, "
        f"air {t['air_launches']}, herd {t['herd_launches']}, "
        f"sync {t['sync_boundaries']}, bytes {t['bytes_transferred']}"
    )


# --- distinguishability -------------------------------------------------------------------------


def _table(totals_by_mode):
    head = f"{'mode':<10}" + "".join(f"{k:>20}" for k in TOTAL_KEYS)
    rows = [head, "-" * len(head)]
    for mode in MODES:
        t = totals_by_mode.get(mode)
        if t is None:
            rows.append(f"{mode:<10}" + f"{'(no artifact)':>20}")
            continue
        rows.append(f"{mode:<10}" + "".join(f"{t[k]:>20,}" for k in TOTAL_KEYS))
    return "\n".join(rows)


def distinguish(totals_by_mode):
    """The cross-mode assertion. Returns a list of failed gating clauses (empty means separated).

    Ordinal over driver-summed totals, never absolute thresholds. `coarse` already measures 131
    runlist entries, 128 of them one operator's row blocking, so any threshold here would be
    measuring `build_addnorm_module`'s L1 capacity rather than the taxonomy. Both 08 and 03 warn
    that "runlist many, coarse few" may otherwise be a comparison between a number and itself.
    """
    failures = []
    o, r, c, f = (totals_by_mode[m] for m in ("offload", "runlist", "coarse", "fused"))

    # 1. Distinctness -- the floor. If two modes produce the same vector, the vector is not
    #    measuring the boundary, whatever else holds.
    for i, a in enumerate(MODES):
        for b in MODES[i + 1 :]:
            if all(totals_by_mode[a][k] == totals_by_mode[b][k] for k in TOTAL_KEYS):
                failures.append(f"{a} and {b} have identical dispatch vectors")

    # 2. offload is the host-mediated extreme, and aggregates nothing. No absolute count is
    #    demanded: attention stays in host torch (see 08c), so it is six GEMM dispatches and not
    #    the eight the plan originally predicted.
    for other in ("runlist", "coarse", "fused"):
        if not o["host_submissions"] > totals_by_mode[other]["host_submissions"]:
            failures.append(
                f"offload host_submissions {o['host_submissions']} does not exceed "
                f"{other}'s {totals_by_mode[other]['host_submissions']}"
            )
    if o["runlist_entries"] != o["host_submissions"]:
        failures.append(
            f"offload aggregates: {o['runlist_entries']} entries over {o['host_submissions']} "
            "submissions, but the mode dispatches one GEMM per submission"
        )

    # 3. runlist EXECUTES more than coarse. The one ordinal claim that mode owns.
    #
    #    `[2026-08-08]` This asserted `runlist_entries > coarse entries` and could not fail:
    #    `runlist` IS `coarse`'s schedule decomposed into single-operator entries, so more entries
    #    is true by construction, and a gate that cannot fail measures nothing (J4). Measured
    #    through Phase F's runner at baseline_768 seq 4096: entries 391 vs 131 -- guaranteed --
    #    against herd_launches 404 vs 146, which counts executed work rather than dispatch
    #    packaging and which neither mode fixes by construction.
    if not r["herd_launches"] > c["herd_launches"]:
        failures.append(
            f"runlist herd_launches {r['herd_launches']} does not exceed coarse's "
            f"{c['herd_launches']}"
        )

    # 4. fused removes intermediate host sync, which is what MLIR-level fusion IS.
    if not f["sync_boundaries"] < c["sync_boundaries"]:
        failures.append(
            f"fused sync_boundaries {f['sync_boundaries']} is not below coarse's "
            f"{c['sync_boundaries']}"
        )
    return failures


def predicted_but_not_gating(totals_by_mode):
    """Clauses recorded with a verdict but never halting.

    Both assume a particular fused decomposition. A faithful whole-layer stitch may still row-block
    its normalization, and `air_launches` is counted once per distinct ELF while `herd_launches`
    accumulates per step (dispatch.py:122-153) -- a deliberate asymmetry that makes the second
    claim weaker than it looks. If either is false that is a finding for the README, not a halt.
    """
    c, f = totals_by_mode["coarse"], totals_by_mode["fused"]
    return [
        ("fused runlist_entries < coarse", f["runlist_entries"] < c["runlist_entries"]),
        ("fused air_launches >= coarse", f["air_launches"] >= c["air_launches"]),
    ]


# --- the E1 ladder ------------------------------------------------------------------------------


def check_naming(repo):
    """The `(method, tile_n)` naming fix, resolved through the real registry.

    The FFN's up-projection and the o-projection are both `drain` at the gate configuration, at
    tile_n 128 and 96 -- which is the collision itself: same method, different tile_n, one symbol
    suffix and one object name. After E1 they must differ.
    """
    pe = os.path.join(repo, "programming_examples")
    for p in (pe, os.path.join(pe, "llms")):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        from shared.builders.gemm_builder import gemm_registry_config
    except Exception as exc:  # noqa: BLE001
        raise CheckError(f"cannot import shared.builders.gemm_builder: {exc}") from exc

    try:
        up = gemm_registry_config(SEQ, HID, FFN)  # ffn_up   4096x768x3072
        op = gemm_registry_config(SEQ, HID, HID)  # o_proj   4096x768x768
    except Exception as exc:  # noqa: BLE001
        raise CheckError(f"gemm_registry_config raised on a registered shape: {exc}") from exc

    # The precondition IS the test. If the registry ever puts these two on different methods their
    # names would differ for an unrelated reason, and this clause would pass while proving nothing.
    if up["method"] != op["method"]:
        raise CheckError(
            f"this check no longer tests what it was written to test: ffn_up resolves to "
            f"{up['method']!r} and o_proj to {op['method']!r}, so they were never going to collide. "
            "Pick two same-method, different-tile_n shapes and update this clause."
        )
    if up["tile_n"] == op["tile_n"]:
        raise CheckError(
            f"this check no longer tests what it was written to test: both shapes resolve to "
            f"tile_n {up['tile_n']}, so there is no collision to fix."
        )

    problems = []
    if up["sym_suffix"] == op["sym_suffix"]:
        problems.append(
            f"both mint sym_suffix {up['sym_suffix']!r} at tile_n {up['tile_n']} and "
            f"{op['tile_n']} -- stitch_elf rejects the redefinition"
        )
    if up["obj"] == op["obj"]:
        problems.append(
            f"both mint object {up['obj']!r} at tile_n {up['tile_n']} and {op['tile_n']} -- "
            "one silently overwrites the other's micro-kernel"
        )
    if problems:
        raise CheckError(
            f"{up['method']} at tile_n {up['tile_n']} and {op['tile_n']} do not have distinct "
            "names: " + "; ".join(problems)
        )
    print(
        f"  {up['method']} tile_n {up['tile_n']} -> {up['sym_suffix']} / {up['obj']}; "
        f"tile_n {op['tile_n']} -> {op['sym_suffix']} / {op['obj']}"
    )


def check_ladder(results, cutoff, listing, pairs_out=None):
    """`ffn` must pass at a baseline_768 sequence length other than 4096.

    Before E1, `build_ffn_module` cannot build at any other point on the ladder at all: its two
    projections collide on `f32_to_bf16_mn_<suffix>` everywhere the registry puts them on the same
    method. So this is not a clause a laxer test can produce -- it needs the fix.

    Dimensions are read from the recorded `shape` dict, never from the shape_key string, because
    the key is a name the session chooses.
    """
    declared = set(_declared_keys(listing, "ffn"))
    hits = []
    for path in sorted(results.glob("ffn__*.json")):
        key = path.name[len("ffn__") : -len(".json")]
        if key not in declared or not _fresh(path, cutoff):
            continue
        d = _load_json(path)
        if _numerics_problems(d):
            continue
        shape = d.get("shape") or {}
        if (
            shape.get("emb_dim") == HID
            and shape.get("ffn_dim") == FFN
            and isinstance(shape.get("seq_len"), int)
            and shape["seq_len"] != SEQ
        ):
            hits.append((key, shape["seq_len"]))

    if not hits:
        raise CheckError(
            "the sequence ladder did not move: no fresh, declared, contract-satisfying `ffn` "
            f"result at a baseline_768 shape whose seq_len is not {SEQ}. Add an opcheck spec at "
            "another ladder point -- seq 64 is the cheapest, and its two registry rows both "
            "resolve to `drain`, which is precisely the collision E1 removes."
        )

    for key, seq in hits:
        print(f"  ffn [{key}]: seq_len {seq}, off the {SEQ} pin")
    if pairs_out:
        with open(pairs_out, "w") as fh:
            for key, _ in hits:
                fh.write(f"ffn {key}\n")
    return hits


# --- CLI ----------------------------------------------------------------------------------------


def _read_listing(path):
    """The `opcheck.py --list` output. Never optional: it is the declared-shape intersection.

    An empty or unparseable file is reported rather than raised as a traceback. In practice it
    means `opcheck.py --list` produced nothing -- almost always because the venv and env_setup.sh
    were not sourced, so `import air` failed -- and a stack trace there reads like a defect in the
    check instead of a missing environment.
    """
    if not path:
        raise CheckError("no --listing given; the declared-shape intersection cannot be skipped")
    p = pathlib.Path(path)
    if not p.exists():
        raise CheckError(f"listing {path} does not exist")
    text = p.read_text().strip()
    if not text:
        raise CheckError(
            f"listing {path} is empty: 'opcheck.py --list' declared nothing. It imports `air`, so "
            "this is usually the venv and utils/env_setup.sh not having been sourced."
        )
    try:
        listing = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CheckError(f"listing {path} is not readable JSON: {exc}") from exc
    if not isinstance(listing, list) or not listing:
        raise CheckError(f"listing {path} is not a non-empty JSON array of declared shapes")
    return listing


def _cutoff(stamp):
    if not stamp:
        raise CheckError("no --stamp given; artifact freshness cannot be proven")
    if not os.path.exists(stamp):
        raise CheckError(f"gate stamp {stamp} does not exist; artifact freshness cannot be proven")
    return os.path.getmtime(stamp)


def _dirs(args):
    results = pathlib.Path(args.results)
    fault = pathlib.Path(args.fault_results) if args.fault_results else results / "fault"
    return results, fault


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ("mode", "distinguish", "ladder", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--results", required=True)
        p.add_argument("--fault-results")
        p.add_argument("--stamp")
        p.add_argument("--listing")
        if name == "mode":
            p.add_argument("--operator", required=True)
            # A mode whose defining property is checkable without the other three. `offload` is
            # the case: it aggregates nothing, and that is true of its own artifact alone. The
            # cross-mode ordering waits for E5, when all four exist.
            p.add_argument("--expect-no-aggregation", action="store_true")
            p.add_argument("--min-submissions", type=int, default=0)
            # An upper bound on one vector field, for a phase whose claim is that a number
            # COLLAPSED. J1 uses it: lifting addnorm's one-trip guard must take coarse's
            # runlist entries from 131 to a handful, and "the mode still agrees with the
            # oracle" is true either way -- the collapse is the whole result, so it has to be
            # asserted separately or the phase can pass having changed nothing.
            p.add_argument("--max-field", choices=TOTAL_KEYS)
            p.add_argument("--max-value", type=int)
        if name == "ladder":
            p.add_argument("--pairs-out")
        if name == "compare":
            # One ordinal claim between two modes, for the sub-phase that owns it. E4 uses it for
            # `runlist entries > coarse entries` -- the reason coarse had to be measured first.
            p.add_argument("--left", required=True)
            p.add_argument("--right", required=True)
            p.add_argument("--field", required=True, choices=TOTAL_KEYS)
            p.add_argument("--relation", required=True, choices=("gt", "lt"))

    p = sub.add_parser("naming")
    p.add_argument("--repo", required=True)

    sub.add_parser("selftest")

    args = ap.parse_args(argv)

    try:
        if args.cmd == "naming":
            check_naming(args.repo)
            return 0
        if args.cmd == "selftest":
            # Imported lazily: the fixtures import back from this module, and nothing but the
            # selftest itself needs them.
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from phase_e_selftest import selftest

            return selftest()

        results, fault = _dirs(args)
        cutoff = _cutoff(args.stamp)
        listing = _read_listing(args.listing)

        if args.cmd == "mode":
            totals = check_mode(args.operator, results, fault, cutoff, listing)
            if args.expect_no_aggregation and totals["runlist_entries"] != totals["host_submissions"]:
                raise CheckError(
                    f"{args.operator} aggregates: {totals['runlist_entries']} runlist entries over "
                    f"{totals['host_submissions']} submissions. This mode dispatches one kernel per "
                    "submission; batching them into a runlist would make it `coarse`."
                )
            if totals["host_submissions"] < args.min_submissions:
                raise CheckError(
                    f"{args.operator} records {totals['host_submissions']} host submission(s), "
                    f"fewer than the {args.min_submissions} its decomposition requires"
                )
            if args.max_field is not None:
                if args.max_value is None:
                    raise CheckError("--max-field requires --max-value")
                got = totals[args.max_field]
                if got > args.max_value:
                    raise CheckError(
                        f"{args.operator} {args.max_field} is {got:,}, above the {args.max_value:,} "
                        "this phase claims to have collapsed it to. The mode agreeing with the "
                        "oracle does not establish that: it agreed before the change too."
                    )
                print(f"  {args.operator} {args.max_field} {got:,} <= {args.max_value:,}")
            return 0
        if args.cmd == "ladder":
            check_ladder(results, cutoff, listing, args.pairs_out)
            return 0
        if args.cmd == "compare":
            left = check_mode(args.left, results, fault, cutoff, listing)
            right = check_mode(args.right, results, fault, cutoff, listing)
            lv, rv = left[args.field], right[args.field]
            ok = lv > rv if args.relation == "gt" else lv < rv
            word = "exceed" if args.relation == "gt" else "be below"
            if not ok:
                raise CheckError(
                    f"{args.left} {args.field} {lv:,} does not {word} {args.right}'s {rv:,}"
                )
            print(f"  {args.left} {args.field} {lv:,} {'>' if args.relation == 'gt' else '<'} "
                  f"{args.right} {rv:,}")
            return 0

        totals = {}
        missing = []
        for mode in MODES:
            try:
                totals[mode] = check_mode(mode, results, fault, cutoff, listing, quiet=True)
            except CheckError as exc:
                missing.append(f"{mode}: {exc}")
        if missing:
            print("distinguishability: not every mode produced a usable artifact:", file=sys.stderr)
            for m in missing:
                print(f"  {m}", file=sys.stderr)
            return 1

        print(_table(totals))
        for label, ok in predicted_but_not_gating(totals):
            print(f"  predicted (recorded, not gating): {label} -- {'holds' if ok else 'FALSE'}")

        failures = distinguish(totals)
        if failures:
            print("", file=sys.stderr)
            print("DISTINGUISHABILITY FAILED. The four modes do not separate as the taxonomy",
                  file=sys.stderr)
            print("predicts, which means the measurement model is not measuring what it claims:",
                  file=sys.stderr)
            for f in failures:
                print(f"  - {f}", file=sys.stderr)
            print("", file=sys.stderr)
            print("Per 08 this must be resolved BEFORE Phase F consumes these numbers. The answer",
                  file=sys.stderr)
            print("is a revised measurement model or a corrected mode -- never a mode tuned until",
                  file=sys.stderr)
            print("the inequality holds. The full table is above.", file=sys.stderr)
            return 1
        print("  all four gating clauses hold: the modes separate as the taxonomy predicts")
        return 0
    except CheckError as exc:
        print(f"objective check: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())

