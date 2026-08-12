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


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("npz", help="written by probe_r1_rung.py --dump-npz")
    ap.add_argument("--seed", type=int, default=13, help="the dispatch's --seed")
    ap.add_argument(
        "--no-self-check",
        action="store_true",
        help="skip the operand byte-match; use only when --seed is unknown, and "
        "say so in whatever you write down",
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
    if herd_x != 1:
        raise SystemExit(
            f"REFUSED: herd_x is {herd_x}. The single-strip algebra below assumes "
            "one down core owning every output column; at herd_x>1 y is a sum over "
            "strips and the basis is not identifiable this way."
        )

    hidden, w_up, w_down = _operands(emb, ffn, TILE_M, a.seed)
    if not a.no_self_check:
        for name, have, want in (
            ("hidden", z["in0"].view(bfloat16), hidden),
            ("w_up", z["in1"].view(bfloat16), ffn_resident_pack_w_up(w_up, 1, tile_k)),
            ("w_down", z["in2"].view(bfloat16), ffn_accum_pack_w(w_down, 1, tile_k)),
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

    basis = []
    for j in range(nkd):
        Hj = H[:, j * tile_k : (j + 1) * tile_k]
        Wj = wd[j * tile_k : (j + 1) * tile_k, :]
        for r in range(nrun):
            m = np.zeros_like(Hj)
            m[r * MICRO : (r + 1) * MICRO] = Hj[r * MICRO : (r + 1) * MICRO]
            basis.append((m @ Wj).ravel())
    B = np.stack(basis, axis=1)
    coef, *_ = np.linalg.lstsq(B, y.ravel(), rcond=None)
    resid = np.abs(y.ravel() - B @ coef).mean() / np.abs(y).mean()

    print(
        f"=== {Path(a.npz).name}  emb={emb} ffn={ffn} herd_x={herd_x} "
        f"tile_k={tile_k} sweeps={sweeps} cpg={cpg} down_K={nkd} ==="
    )
    print(
        f"[fit] {B.shape[1]} (chunk, row-run) basis vectors over {y.size} "
        f"equations; residual relL1 = {resid:.4f}  "
        f"(bf16 noise floor is ~0.016; a large residual means the defect is NOT "
        f"a per-run arrival and this map should not be read)"
    )
    print("[map] 1.00 = that run arrived, 0.00 = it did not")
    print("    " + f"{'j':>2} {'(s,jj)':>7}  " + " ".join(f"{'r%d'%r:>5}" for r in range(nrun)))
    worst = []
    for j in range(nkd):
        c = coef[j * nrun : (j + 1) * nrun]
        tag = "  <== LAST CHUNK OF GROUP" if cpg > 1 and (j % cpg) == cpg - 1 else ""
        print(
            f"    {j:>2} {'(%d,%d)' % (j // cpg, j % cpg):>7}  "
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
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
