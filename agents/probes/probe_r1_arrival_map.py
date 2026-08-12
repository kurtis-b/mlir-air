#!/usr/bin/env python3
"""Read an R1 device output back as an ARRIVAL MAP: which piece of which chunk
actually reached the down herd.

WHY THIS EXISTS (queue item 23).  `probe_r1_rung.py --mode dispatch` reports one
correlation and one mismatch count.  Those say a rung is wrong; they never say
*which transfer* was wrong, so R1's deterministic failure at
`herd_x=1, k'=4, chunks=4` sat unchased as "an eighth wall".  It is not a wall:
it is one identifiable transfer arriving truncated, and the way to see that is to
stop scoring the output and start DECOMPOSING it.

WHAT IT DOES.  R1's down herd computes, by design,

    y = sum_j  H[:, j*tile_k : (j+1)*tile_k] @ w_down[j*tile_k : (j+1)*tile_k, :]

with H = gelu(hidden @ w_up), one term per down-K step.  The up herd hands each
H chunk over in `TILE_M//MICRO` RUNS of `tile_k*MICRO` elements (the strided
`air.channel.put` out of the blocked L1 accumulator).  So build one basis matrix
per (down-K step j, row-run r) -- `H_j` with every row outside run r zeroed,
times `Wd_j` -- and least-squares the measured `y` over all of them.  Each
coefficient is 1.0 if that run arrived and 0.0 if it did not, and every
coefficient is identifiable because the runs occupy disjoint output rows.

No matrix inversion, so no noise amplification by cond(w_down); the residual is
reported and on a correct rung it sits at the bf16 noise floor (~0.016).

CALIBRATE IT BEFORE YOU BELIEVE IT.  Run it on a rung that PASSES first.  Every
coefficient must read 1.00.  A run of this probe on `--emb-dim 32 --ffn-dim 32`
is the negative control for the instrument itself, and `--self-check` makes that
mandatory by refusing any input whose operands do not byte-match the probe's own
regeneration from `--seed`.

INPUT.  The `.npz` written by `probe_r1_rung.py --dump-npz`, which carries the
raw device `y` beside the exact operands that produced it.

    python3 agents/probes/probe_r1_rung.py --mode dispatch ... \\
        --dump-npz rung.npz
    python3 agents/probes/probe_r1_arrival_map.py rung.npz

WHAT IT FOUND (devq 278 / 294, doc 49).  At every rung with
`chunks_per_group > 1` the LAST chunk of each up-herd group arrives as its first
run only, the remaining runs zeroed -- the core's accumulator memset overtaking
the last chunk's BD, because `air-to-aie`'s shared-staging-buffer lock placement
leaves the core's own writes unguarded.

herd_x >= 2 (queue item 21, wall 7)
-----------------------------------
The single-strip refusal this probe used to carry was over-conservative.  At
`herd_x > 1` down core `tx` owns y columns `[tx*group_n, (tx+1)*group_n)` and
computes `y[:, strip] = sum_j H_j @ wd_j[:, strip]` -- the strips are DISJOINT,
so the same basis is identifiable **per strip**, and the per-strip maps answer
"which core, which chunk" directly.

A second model is fitted alongside, and at herd_x >= 2 it is the one that fires:
the full `{H_i @ Wd_j}` PAIRING dictionary.  An arrival map can only see a
transfer arriving truncated; it is blind to a transfer arriving WHOLE but
MATCHED WITH THE WRONG PARTNER.  Wall 7 is the second kind -- every chunk
arrives intact and the down herd pairs it with another K step's `w_down` --
so the arrival model residual blows up (0.66-0.84) while the pairing model sits
at the bf16 noise floor (0.015).  Read the pairing matrix first whenever the
arrival residual is large; the arrival map is not meaningful then.

The pairing matrix is reported as a permutation `position -> H chunk`, together
with whether that permutation is an INTERLEAVING of the per-up-core chunk
streams (each core's own chunks still in order, only the merge between cores
varying).  That is wall 7's signature: herd_x S2MM channels racing for one
single-slot memtile staging buffer, which can reorder the cores against each
other but never reorder one core against itself.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_WT = Path(__file__).resolve().parents[2]
_PROJ = _WT / "programming_examples"
for _p in (str(_PROJ), str(_PROJ / "llms"), str(_PROJ / "transformer_layer")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# Above this relative-L1 residual a least-squares model does not explain the
# output and its coefficients must not be read. The bf16 noise floor measured on
# passing rungs is ~0.016 (doc 49); 0.05 leaves room for a rung's own rounding
# without admitting a model that is merely the best of a bad basis.
_ARRIVAL_RESID_OK = 0.05


def _gelu(x):
    x = np.asarray(x, np.float64)
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def _operands(emb, ffn, tile_m, seed):
    """Regenerate EXACTLY probe_r1_rung.build()'s operands for this rung."""
    from ml_dtypes import bfloat16
    from opcheck_prepare import (
        _GELU_SECOND_MOMENT,
        _registry_gemm_scale,
        _unit_output_scale,
    )

    rng = np.random.default_rng(seed)
    up_scale = _unit_output_scale(emb)
    down_scale = _registry_gemm_scale(ffn) / np.sqrt(ffn * _GELU_SECOND_MOMENT)
    hidden = (rng.standard_normal((tile_m, emb)) * up_scale).astype(bfloat16)
    w_up = (rng.standard_normal((emb, ffn)) * up_scale).astype(bfloat16)
    w_down = (rng.standard_normal((ffn, emb)) * down_scale).astype(bfloat16)
    return hidden, w_up, w_down


def _synth(emb, ffn, hx, tile_k, sigma, path, seed=13):
    """Write a --dump-npz-shaped file whose y has stream position p consuming H
    chunk sigma[p]. Used only by --self-test."""
    from ml_dtypes import bfloat16
    from builders.ffn_accum import MICRO, TILE_M, ffn_accum_pack_w
    from builders.ffn_resident import ffn_resident_pack_w_up

    hidden, w_up, w_down = _operands(emb, ffn, TILE_M, seed)
    H = _gelu(hidden.astype(np.float64) @ w_up.astype(np.float64))
    wd = w_down.astype(np.float64)
    y = np.zeros((TILE_M, emb), np.float64)
    for pos, i in enumerate(sigma):
        y += H[:, i * tile_k : (i + 1) * tile_k] @ wd[
            pos * tile_k : (pos + 1) * tile_k, :
        ]
    group_n = emb // hx
    geom = {
        "seq_len": TILE_M, "emb_dim": emb, "ffn_dim": ffn, "herd_x": hx,
        "tile_k": tile_k, "group_n": group_n,
        "sweeps": ffn // (hx * group_n), "k_steps_up": emb // tile_k,
        "chunks_per_group": group_n // tile_k, "micro": MICRO,
    }
    np.savez(
        path,
        y_raw=np.asarray(y.astype(bfloat16)).view(np.uint16),
        expected=np.zeros((TILE_M, emb), np.float64),
        geom=np.array(json.dumps(geom)),
        argv=np.array(json.dumps(["--self-test"])),
        in0=np.ascontiguousarray(hidden).view(np.uint16),
        in1=ffn_resident_pack_w_up(w_up, hx, tile_k).view(np.uint16),
        in2=ffn_accum_pack_w(w_down, hx, tile_k).view(np.uint16),
    )


def _self_test():
    """Show BOTH new clauses failing on a deliberately broken input.

    A check that cannot fail is not a check (queue items 10, 14, 17, 19, 20).
    The pairing model and the interleaving clause are both asserted here against
    synthetic outputs whose permutation is known by construction.
    """
    import io
    import contextlib
    import tempfile

    cases = [
        # (label, sigma, must appear, must NOT appear, extra argv)
        ("cross-column interleaving is recovered and named",
         [0, 2, 1, 3], "IS an interleaving", "is NOT an interleaving", []),
        ("a WITHIN-column swap is refused as an interleaving",
         [1, 0, 2, 3], "is NOT an interleaving", None, []),
        ("the identity raises no alarm",
         [0, 1, 2, 3], "in step", "NOT an interleaving", ["--always-pairing"]),
    ]
    ok = True
    with tempfile.TemporaryDirectory() as td:
        for label, sigma, want, unwanted, extra in cases:
            p = Path(td) / "c.npz"
            _synth(128, 128, 2, 32, sigma, p)
            buf = io.StringIO()
            argv = list(sys.argv)
            sys.argv = ["probe", *extra, str(p)]
            try:
                with contextlib.redirect_stdout(buf):
                    main()
            finally:
                sys.argv = argv
            out = buf.getvalue()
            good = want in out and f"position -> H chunk: {sigma}" in out
            if unwanted:
                good = good and unwanted not in out
            ok &= good
            print(f"  [{'PASS' if good else 'FAIL'}] {label}  (sigma {sigma})")
    print("SELF-TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    if "--self-test" in sys.argv[1:]:
        return _self_test()
    ap.add_argument("npz", help="written by probe_r1_rung.py --dump-npz")
    ap.add_argument("--seed", type=int, default=13, help="the dispatch's --seed")
    ap.add_argument(
        "--no-self-check",
        action="store_true",
        help="skip the operand byte-match; use only when --seed is unknown, and "
        "say so in whatever you write down",
    )
    ap.add_argument(
        "--always-pairing",
        action="store_true",
        help="fit the {H_i @ Wd_j} pairing dictionary even when the arrival "
        "model already explains the output (on a correct rung it must come "
        "back as the identity -- that is this model's own calibration)",
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="run the pairing and interleaving clauses against synthetic "
        "outputs whose permutation is known by construction, including one the "
        "interleaving clause must REFUSE; takes no npz and touches no device",
    )
    a = ap.parse_args()

    from ml_dtypes import bfloat16
    from builders.ffn_accum import MICRO, TILE_M, ffn_accum_pack_w
    from builders.ffn_resident import ffn_resident_pack_w_up

    z = np.load(a.npz, allow_pickle=False)
    geom = json.loads(str(z["geom"]))
    emb, ffn = geom["emb_dim"], geom["ffn_dim"]
    herd_x, tile_k = geom["herd_x"], geom["tile_k"]
    sweeps, cpg = geom["sweeps"], geom["chunks_per_group"]
    group_n = geom["group_n"]

    hidden, w_up, w_down = _operands(emb, ffn, TILE_M, a.seed)
    if not a.no_self_check:
        for name, have, want in (
            ("hidden", z["in0"].view(bfloat16), hidden),
            (
                "w_up",
                z["in1"].view(bfloat16),
                ffn_resident_pack_w_up(w_up, herd_x, tile_k),
            ),
            (
                "w_down",
                z["in2"].view(bfloat16),
                ffn_accum_pack_w(w_down, herd_x, tile_k),
            ),
        ):
            if not (have == want).all():
                raise SystemExit(
                    f"REFUSED: regenerated {name} does not byte-match the dispatch's "
                    f"operand. --seed {a.seed} is wrong, or the builder's packing "
                    "changed since the run. Scoring a model against the wrong "
                    "operands is how a confident wrong story gets written."
                )

    y = z["y_raw"].view(bfloat16).astype(np.float64).reshape(TILE_M, emb)
    H = _gelu(hidden.astype(np.float64) @ w_up.astype(np.float64))
    wd = w_down.astype(np.float64)
    nkd, nrun = ffn // tile_k, TILE_M // MICRO

    print(
        f"=== {Path(a.npz).name}  emb={emb} ffn={ffn} herd_x={herd_x} "
        f"tile_k={tile_k} sweeps={sweeps} cpg={cpg} down_K={nkd} ==="
    )

    # Chunk j is produced by up/GeLU column (j // cpg) % herd_x in sweep
    # (j // cpg) // herd_x -- group g = s*herd_x + c spans K steps
    # [g*cpg, (g+1)*cpg).
    def producer(j):
        g = j // cpg
        return g % herd_x, g // herd_x  # (column, sweep)

    rc = 0
    for tx in range(herd_x):
        lo, hi = tx * group_n, (tx + 1) * group_n
        ys = y[:, lo:hi]
        if herd_x > 1:
            print(f"--- down core {tx}, y columns [{lo}, {hi}) ---")

        # ---- model A: per (chunk, row-run) arrival (diagonal pairing) ----
        basis = []
        for j in range(nkd):
            Hj = H[:, j * tile_k : (j + 1) * tile_k]
            Wj = wd[j * tile_k : (j + 1) * tile_k, lo:hi]
            for r in range(nrun):
                m = np.zeros_like(Hj)
                m[r * MICRO : (r + 1) * MICRO] = Hj[r * MICRO : (r + 1) * MICRO]
                basis.append((m @ Wj).ravel())
        B = np.stack(basis, axis=1)
        coef, *_ = np.linalg.lstsq(B, ys.ravel(), rcond=None)
        resid = np.abs(ys.ravel() - B @ coef).mean() / np.abs(ys).mean()

        print(
            f"[fit] {B.shape[1]} (chunk, row-run) basis vectors over {ys.size} "
            f"equations; residual relL1 = {resid:.4f}  "
            f"(bf16 noise floor is ~0.016; a large residual means the defect is NOT "
            f"a per-run arrival and this map should not be read)"
        )
        print("[map] 1.00 = that run arrived, 0.00 = it did not")
        lbl = "(s,c,jj)" if herd_x > 1 else "(s,jj)"
        print(
            "    "
            + f"{'j':>2} {lbl:>9}  "
            + " ".join(f"{'r%d' % r:>5}" for r in range(nrun))
        )
        worst = []
        for j in range(nkd):
            c = coef[j * nrun : (j + 1) * nrun]
            tag = (
                "  <== LAST CHUNK OF GROUP"
                if cpg > 1 and (j % cpg) == cpg - 1
                else ""
            )
            col, swp = producer(j)
            name = (
                "(%d,%d,%d)" % (swp, col, j % cpg)
                if herd_x > 1
                else "(%d,%d)" % (j // cpg, j % cpg)
            )
            print(
                f"    {j:>2} {name:>9}  "
                + " ".join(f"{v:5.2f}" for v in c)
                + tag
            )
            worst.append((float(np.abs(c - 1.0).max()), j))
        bad = [j for d, j in worst if d > 0.25]
        print(
            "[verdict] "
            + (
                "every run of every chunk arrived"
                if not bad
                else f"chunks {bad} lost runs; "
                f"{'ALL are the last chunk of a group' if all((j % cpg) == cpg - 1 for j in bad) and cpg > 1 else 'the pattern is NOT last-chunk-of-group'}"
            )
        )
        if bad:
            rc = 1

        # ---- model B: the full {H_i @ Wd_j} pairing dictionary ----
        # Only meaningful when the arrival model does NOT explain the output.
        # A whole chunk matched to the wrong K step is invisible to model A.
        if resid <= _ARRIVAL_RESID_OK and not a.always_pairing:
            continue
        pb = []
        for i in range(nkd):
            Hi = H[:, i * tile_k : (i + 1) * tile_k]
            for j in range(nkd):
                pb.append((Hi @ wd[j * tile_k : (j + 1) * tile_k, lo:hi]).ravel())
        PB = np.stack(pb, axis=1)
        pcoef, *_ = np.linalg.lstsq(PB, ys.ravel(), rcond=None)
        presid = np.abs(ys.ravel() - PB @ pcoef).mean() / np.abs(ys).mean()
        M = pcoef.reshape(nkd, nkd)
        print(
            f"[pairing] full {{H_i @ Wd_j}} dictionary, residual relL1 = "
            f"{presid:.4f}"
            + (
                "  <== THIS model explains the output: whole chunks, wrong partners"
                if presid <= _ARRIVAL_RESID_OK < resid
                else ""
            )
        )
        for i in range(nkd):
            print("      " + " ".join(f"{v:6.2f}" for v in M[i]))
        # Read it as a permutation: stream position j consumed H chunk sigma[j].
        if presid <= _ARRIVAL_RESID_OK:
            sigma = [int(np.argmax(np.abs(M[:, j]))) for j in range(nkd)]
            ok_perm = sorted(sigma) == list(range(nkd))
            print(
                f"[pairing] position -> H chunk: {sigma}"
                + ("" if ok_perm else "  (NOT a permutation)")
            )
            if ok_perm:
                if sigma == list(range(nkd)):
                    print("[pairing] identity -- the streams are in step")
                else:
                    rc = 1
                    # Interleaving test: within each producing column, are that
                    # column's own chunks still in increasing order?
                    per_col = {}
                    for pos, ch in enumerate(sigma):
                        per_col.setdefault(producer(ch)[0], []).append(ch)
                    inter = all(
                        v == sorted(v) for v in per_col.values()
                    )
                    print(
                        "[pairing] per-column arrival order "
                        + " ".join(f"c{k}={v}" for k, v in sorted(per_col.items()))
                    )
                    print(
                        "[pairing] "
                        + (
                            "IS an interleaving of the per-column streams "
                            "(wall 7's signature: the columns race each other, "
                            "never themselves)"
                            if inter
                            else "is NOT an interleaving -- a column's own "
                            "chunks arrived out of order, which the "
                            "single-slot writer race cannot do"
                        )
                    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
