# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only checks on the query-blocked attention ``offload`` and ``runlist`` share.

``pattern/blocked_attention.py`` is the one implementation of the host
attention boundary for the two modes that decompose attention, and it is
written to be arithmetically INDEPENDENT of the oracle
(``builders/mha_attention.py::chunked_attention_reference``) — torch on the
mode path, numpy on the check path, the same convention implemented twice on
purpose. That independence is exactly what makes a test here meaningful:
comparing the two implementations on the same data is a real check, not a
value compared against itself, and it is the check no hardware gate performs
(the hardware gate compares a whole LAYER whose attention inputs already
carry device GEMM error).

    python3 pattern/test_blocked_attention.py

No hardware, no MLIR, a couple of seconds. It is a ``test_*.py`` so ordinary
pytest discovery finds it too (porting convention 11).

FOOTGUN
    The blocked/unblocked agreement below is checked at 1e-6, NOT exact
    equality: partitioning the query rows never changes any row's softmax or
    its reduction over V mathematically, but torch may order a matmul's
    reductions differently at different operand shapes. Exact equality would
    test torch's scheduler, not the blocking.
"""

import math
import os
import sys

import numpy as np
from ml_dtypes import bfloat16

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE_ROOT = os.path.dirname(_HERE)
if _EXAMPLE_ROOT not in sys.path:
    sys.path.insert(0, _EXAMPLE_ROOT)

from builders.mha_attention import chunked_attention_reference  # noqa: E402
from pattern.blocked_attention import (  # noqa: E402
    MAX_ATTENTION_SCRATCH_BUFFER_BYTES,
    MIN_BLOCKED_QUERY_BLOCK_SIZE,
    blocked_attention,
    resolve_query_block_size,
)

SEQ, HEADS, HEAD_DIM = 128, 4, 32


def _qkv(seed=0):
    rng = np.random.default_rng(seed)
    shape = (SEQ, HEADS * HEAD_DIM)
    return tuple(rng.standard_normal(shape).astype(bfloat16) for _ in range(3))


def test_blocking_does_not_change_the_arithmetic():
    """A forced small block agrees with the unblocked computation."""
    q, k, v = _qkv()
    whole = blocked_attention(q, k, v, HEADS, query_block_size=SEQ)
    blocked = blocked_attention(q, k, v, HEADS, query_block_size=32)
    assert whole.dtype == np.float32
    assert np.allclose(whole, blocked, rtol=0, atol=1e-6)


def test_agrees_with_the_independent_oracle():
    """torch mode path vs numpy oracle path, same bf16 inputs, both variants.

    The two implementations share no code; agreement to f32 noise on the same
    data is what says they compute the same function.
    """
    q, k, v = _qkv(seed=1)
    for causal in (False, True):
        mode = blocked_attention(q, k, v, HEADS, causal=causal, query_block_size=32)
        oracle = chunked_attention_reference(
            q, k, v, HEADS, causal=causal, query_block_size=64
        )
        assert np.allclose(mode, oracle, rtol=1e-5, atol=1e-5), f"causal={causal}"


def test_causal_first_row_attends_only_to_itself():
    """Row 0 under a causal mask is exactly V's row 0, per head."""
    q, k, v = _qkv(seed=2)
    out = blocked_attention(q, k, v, HEADS, causal=True, query_block_size=32)
    assert np.allclose(out[0], np.asarray(v[0], dtype=np.float32), rtol=0, atol=1e-6)


def test_query_block_size_thresholds():
    """Unblocked under the scratch cap; the documented block above it.

    The two rungs 08c quotes: 4096 x 12 heads is ~805 MB of f32 scores, under
    the 3 GiB cap, so no blocking; 16384 x 12 is ~12.9 GB, over it, and the
    largest divisor at or above the minimum that fits the cap is 4096.
    """
    assert resolve_query_block_size(4096, 12) == 4096
    block = resolve_query_block_size(16384, 12)
    assert block == 4096
    assert block >= MIN_BLOCKED_QUERY_BLOCK_SIZE
    assert 16384 % block == 0
    assert block * 16384 * 12 * 4 <= MAX_ATTENTION_SCRATCH_BUFFER_BYTES


def test_scale_is_rsqrt_head_dim():
    """One head, one position: softmax over a single key is 1, so out == v.

    And with two positions the logit gap is scaled by 1/sqrt(head_dim):
    checked against a direct computation with math.exp, not against either
    implementation.
    """
    head_dim = 16
    rng = np.random.default_rng(3)
    q = rng.standard_normal((2, head_dim)).astype(bfloat16)
    k = rng.standard_normal((2, head_dim)).astype(bfloat16)
    v = rng.standard_normal((2, head_dim)).astype(bfloat16)
    out = blocked_attention(q, k, v, 1, query_block_size=2)

    qf, kf, vf = (np.asarray(t, dtype=np.float64) for t in (q, k, v))
    logits = qf[0] @ kf.T / math.sqrt(head_dim)
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    assert np.allclose(out[0], weights @ vf, rtol=1e-5, atol=1e-5)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"blocked attention tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
