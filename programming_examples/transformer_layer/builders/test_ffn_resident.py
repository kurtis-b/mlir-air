# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only: ffn_resident's addressing arithmetic, emulated to exactness.

WHAT THIS PINS, AND WHY IT EXISTS AT ALL
    ``builders/ffn_resident.py`` moves every byte through hand-derived
    patterns: the w_up (sweep, k', column)-major packing, the shim's 4-D
    row-major->blocked retile of ``hidden``, the strided chunk extraction
    from the blocked H group, the w_down per-K-step slices, and a global K
    order assembled from three herds' loops. A single off-by-one in any of
    them produces plausible wrong numbers on the device, hours after the
    mistake. This file emulates THE DEVICE'S EXACT ADDRESS ARITHMETIC in
    numpy -- every DMA pattern element-by-element, every channel op in its
    declared order, f64 throughout so ONLY ordering and addressing are
    tested (bf16 rounding is the device's business and the numeric arm's)
    -- and requires the result to equal ``gelu(hidden @ w_up) @ w_down``
    exactly.

    It is also, deliberately, the one behavioral check that stays LIVE
    while the operator's device gate is parked: every aircc-compiling arm
    is blocked by the air-fuse-channels crash (doc 31, "R1's gate is
    BLOCKED"), and this file needs no aircc, no XRT, no kernel objects.

    No framework dependency -- pytest is not in the sandbox venv -- so it
    runs as a plain script and prints a named pass count, exactly as
    ``test_block_cache.py`` does; the lit arm FileChecks the count so a
    check that stops running fails loudly.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_TL = os.path.dirname(_HERE)  # transformer_layer/
_PE = os.path.dirname(_TL)  # programming_examples/
for _p in (_PE, os.path.join(_PE, "llms"), _TL):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from builders.ffn_accum import MICRO, TILE_M, ffn_accum_pack_w  # noqa: E402
from builders.ffn_resident import ffn_resident_pack_w_up  # noqa: E402

SEQ, FFN, EMB = 64, 3072, 768
HERD_X, TILE_K = 4, 32
GROUP_N = EMB // HERD_X  # 192
SWEEPS = FFN // (HERD_X * GROUP_N)  # 4
CPG = GROUP_N // TILE_K  # 6
K_UP = EMB // TILE_K  # 24
CHUNK = TILE_M * TILE_K  # 2048
UP_B = TILE_K * GROUP_N  # 6144
DOWN_CHUNK = TILE_K * EMB  # 24576

_passed = 0
_failed = 0


def _check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1
    else:
        _failed += 1
        print(f"FAIL: {name}")


def _unblock(flat, rows, cols):
    g = flat.reshape(rows // MICRO, cols // MICRO, MICRO, MICRO)
    return g.transpose(0, 2, 1, 3).reshape(rows, cols)


def _block(mat):
    r, c = mat.shape
    return np.ascontiguousarray(
        mat.reshape(r // MICRO, MICRO, c // MICRO, MICRO)
        .transpose(0, 2, 1, 3)
        .reshape(-1)
    )


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))


def _shim_retile(hidden, kp):
    """The up feed's 4-D refill: offsets [0, kp*(TILE_K//MICRO), 0, 0],
    sizes [8, 4, 8, 8], strides [8*EMB, MICRO, EMB, 1] over row-major
    hidden -- element for element, as the BD walks it."""
    flat = hidden.reshape(-1)
    out = np.empty(CHUNK)
    idx = 0
    base = kp * TILE_K
    for d0 in range(TILE_M // MICRO):
        for d1 in range(TILE_K // MICRO):
            for d2 in range(MICRO):
                for d3 in range(MICRO):
                    out[idx] = flat[
                        d0 * MICRO * EMB + base + d1 * MICRO + d2 * EMB + d3
                    ]
                    idx += 1
    return out


def _chunk_put(cflat, jj):
    """The up core's strided chunk put: offsets [jj, 0, 0], sizes
    [1, 8, TILE_K*MICRO], strides [TILE_K*MICRO, GROUP_N*MICRO, 1]."""
    out = np.empty(CHUNK)
    i = 0
    for d1 in range(TILE_M // MICRO):
        for d2 in range(TILE_K * MICRO):
            out[i] = cflat[jj * TILE_K * MICRO + d1 * GROUP_N * MICRO + d2]
            i += 1
    return out


def main():
    rng = np.random.default_rng(0)
    hidden = rng.standard_normal((SEQ, EMB))
    w_up = rng.standard_normal((EMB, FFN))
    w_down = rng.standard_normal((FFN, EMB))
    wup_packed = ffn_resident_pack_w_up(w_up, HERD_X, TILE_K)
    wdown_packed = ffn_accum_pack_w(w_down, HERD_X, TILE_K)

    # 1. The shim retile delivers exactly the k' column slice, blocked.
    ok = all(
        np.array_equal(
            _unblock(_shim_retile(hidden, kp), TILE_M, TILE_K),
            hidden[:, kp * TILE_K : (kp + 1) * TILE_K],
        )
        for kp in range(K_UP)
    )
    _check("shim 4-D retile == k' column slice, all k'", ok)

    # 2. The w_up packing puts group g's k'-slice where the feed reads it.
    ok = True
    for s in range(SWEEPS):
        for kp in range(K_UP):
            w_off = (s * K_UP + kp) * HERD_X * UP_B
            for c in range(HERD_X):
                g = s * HERD_X + c
                got = _unblock(
                    wup_packed[w_off + c * UP_B : w_off + (c + 1) * UP_B],
                    TILE_K,
                    GROUP_N,
                )
                want = w_up[
                    kp * TILE_K : (kp + 1) * TILE_K, g * GROUP_N : (g + 1) * GROUP_N
                ]
                ok = ok and np.array_equal(got, want)
    _check("w_up pack: every (s, k', c) slice", ok)

    # 3. The chunk put extracts column block jj of the blocked group.
    group = rng.standard_normal((TILE_M, GROUP_N))
    ok = all(
        np.array_equal(
            _chunk_put(_block(group), jj),
            _block(group[:, jj * TILE_K : (jj + 1) * TILE_K]),
        )
        for jj in range(CPG)
    )
    _check("chunk put == blocked column block, all jj", ok)

    # 4. The w_down refill at K step j is rows [32j, 32j+32), per-tx sliced.
    ok = True
    for j in range(FFN // TILE_K):
        wchunk = wdown_packed[j * DOWN_CHUNK : (j + 1) * DOWN_CHUNK]
        for tx in range(HERD_X):
            got = _unblock(wchunk[tx * UP_B : (tx + 1) * UP_B], TILE_K, GROUP_N)
            want = w_down[
                j * TILE_K : (j + 1) * TILE_K, tx * GROUP_N : (tx + 1) * GROUP_N
            ]
            ok = ok and np.array_equal(got, want)
    _check("w_down pack: every (j, tx) slice", ok)

    # 5. End to end: the three herds' loops, chunk order and all, EXACTLY
    # reproduce gelu(hidden @ w_up) @ w_down. Any ordering or addressing
    # slip breaks exactness; f64 keeps rounding out of the verdict.
    streams = [[] for _ in range(HERD_X)]
    for s in range(SWEEPS):
        groups = [np.zeros((TILE_M, GROUP_N)) for _ in range(HERD_X)]
        for kp in range(K_UP):
            a_mat = _unblock(_shim_retile(hidden, kp), TILE_M, TILE_K)
            w_off = (s * K_UP + kp) * HERD_X * UP_B
            for c in range(HERD_X):
                b_mat = _unblock(
                    wup_packed[w_off + c * UP_B : w_off + (c + 1) * UP_B],
                    TILE_K,
                    GROUP_N,
                )
                groups[c] += a_mat @ b_mat
        for c in range(HERD_X):
            cflat = _block(groups[c])
            for jj in range(CPG):
                streams[c].append(_gelu(_chunk_put(cflat, jj)))
    y = np.zeros((TILE_M, EMB))
    consumed = [0] * HERD_X
    j_global = 0
    for s in range(SWEEPS):
        for c in range(HERD_X):
            for jj in range(CPG):
                a_mat = _unblock(streams[c][consumed[c]], TILE_M, TILE_K)
                consumed[c] += 1
                wchunk = wdown_packed[
                    j_global * DOWN_CHUNK : (j_global + 1) * DOWN_CHUNK
                ]
                for tx in range(HERD_X):
                    b_mat = _unblock(
                        wchunk[tx * UP_B : (tx + 1) * UP_B], TILE_K, GROUP_N
                    )
                    y[:, tx * GROUP_N : (tx + 1) * GROUP_N] += a_mat @ b_mat
                j_global += 1
    ref = _gelu(hidden @ w_up) @ w_down
    err = float(np.abs(y - ref).max())
    _check(f"end-to-end dataflow exact (max err {err:.2e})", err < 1e-9)

    total = _passed + _failed
    print(f"ffn_resident emulation tests: {_passed}/{total} passed")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
