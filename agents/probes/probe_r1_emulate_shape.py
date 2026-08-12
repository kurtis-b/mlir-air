#!/usr/bin/env python3
"""Run R1's MODULE-INTERPRETING emulation arm (queue item 17) at an ARBITRARY
shape, so the device ladder's off-gate rungs can be validated as probes.

WHY.  devq 273/274 walked R1 across shapes on hardware and found three regimes:
correct at the fully degenerate 1x1x1 rung, WRONG ANSWER at two others, and
ERT_CMD_STATE_TIMEOUT everywhere else.  Before any of that is read as a
statement about the hardware, the builder has to be shown correct AT THOSE
SHAPES -- otherwise a garbage rung is a builder-at-that-shape artifact and the
ladder is not a bisection of anything.

`builders/test_ffn_resident.py` pins SEQ/FFN/EMB/HERD_X/TILE_K as module
globals and derives everything else from them, so re-pointing those globals and
re-deriving the rest runs the same interpreter at another shape.  Clause 5 (end
to end, f64, patterns read off the built module) is what is reproduced here;
the packing clauses come along because they are pure functions of the same
constants.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

_WT = Path(__file__).resolve().parents[2]
_PROJ = _WT / "programming_examples"
for _p in (str(_PROJ), str(_PROJ / "llms"), str(_PROJ / "transformer_layer")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

p = argparse.ArgumentParser()
p.add_argument("--emb-dim", type=int, required=True)
p.add_argument("--ffn-dim", type=int, required=True)
p.add_argument("--herd-x", type=int, required=True)
p.add_argument("--tile-k", type=int, default=32)
p.add_argument("--tag", default="")
a = p.parse_args()

import builders.test_ffn_resident as T  # noqa: E402
from builders.ffn_accum import TILE_M, ffn_accum_pack_w  # noqa: E402
from builders.ffn_resident import (  # noqa: E402
    build_ffn_resident_module,
    ffn_resident_pack_w_up,
)

# Re-point the arm's globals, re-deriving every dependent one exactly as the
# module does at import.
T.SEQ, T.FFN, T.EMB = TILE_M, a.ffn_dim, a.emb_dim
T.HERD_X, T.TILE_K = a.herd_x, a.tile_k
T.GROUP_N = T.EMB // T.HERD_X
T.SWEEPS = T.FFN // (T.HERD_X * T.GROUP_N)
T.CPG = T.GROUP_N // T.TILE_K
T.K_UP = T.EMB // T.TILE_K
T.CHUNK = TILE_M * T.TILE_K
T.UP_B = T.TILE_K * T.GROUP_N
T.DOWN_CHUNK = T.TILE_K * T.EMB

rng = np.random.default_rng(0)
hidden = rng.standard_normal((T.SEQ, T.EMB))
w_up = rng.standard_normal((T.EMB, T.FFN))
w_down = rng.standard_normal((T.FFN, T.EMB))
wup_packed = ffn_resident_pack_w_up(w_up, T.HERD_X, T.TILE_K)
wdown_packed = ffn_accum_pack_w(w_down, T.HERD_X, T.TILE_K)

module = build_ffn_resident_module(
    T.SEQ, T.FFN, T.EMB, herd_x=T.HERD_X, tile_k=T.TILE_K
)


def hosts():
    return [
        np.ascontiguousarray(hidden.reshape(-1)),
        wup_packed.astype(np.float64),
        wdown_packed.astype(np.float64),
        rng.standard_normal(T.SEQ * T.EMB),  # y enters as noise, per the arm
    ]


ref = T._gelu(hidden @ w_up) @ w_down
args = hosts()
ctx = T.emulate(module, args)
err = float(np.abs(args[3].reshape(T.SEQ, T.EMB) - ref).max())
scale = float(np.abs(ref).max())
label = a.tag or f"{T.EMB}x{T.FFN} hx={T.HERD_X} tk={T.TILE_K}"
print(
    f"{label:<14} sweeps={T.SWEEPS} k_up={T.K_UP} cpg={T.CPG} gn={T.GROUP_N} "
    f"| max|y-ref| = {err:.4e}  (|ref|max {scale:.3e})  "
    f"-> {'EXACT' if err < 1e-9 else 'MISMATCH'}"
)
sys.exit(0 if err < 1e-9 else 1)
