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

#: The tolerances the checks below assert AT. Named once so a negative control
#: cannot drift to a looser bar than the clean check it is the control for --
#: a control that rejects only at 1e-2 says nothing about a check asserting at
#: 1e-5. Every ``_agrees``/``_rejects`` call takes these, never a literal.
ORACLE_RTOL = ORACLE_ATOL = 1e-5
BLOCKING_ATOL = 1e-6


def _qkv(seed=0):
    rng = np.random.default_rng(seed)
    shape = (SEQ, HEADS * HEAD_DIM)
    return tuple(rng.standard_normal(shape).astype(bfloat16) for _ in range(3))


def _max_abs(a, b):
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def _agrees(a, b, rtol, atol, what):
    """The comparison, with its margin reported rather than swallowed."""
    delta = _max_abs(a, b)
    assert np.allclose(a, b, rtol=rtol, atol=atol), f"{what}: max|diff| {delta:.3e}"
    return delta


def _rejects(a, b, rtol, atol, what):
    """The SAME comparison, asserted to FAIL. Returns its margin.

    This is the half the module was missing: every check above compares two
    things that agree, so none of them had ever been shown able to disagree.
    A tolerance wide enough to swallow the defect, a mask argument silently
    ignored, or the two implementations folded into one would leave every
    clean check green -- and each is caught here instead.
    """
    delta = _max_abs(a, b)
    assert not np.allclose(a, b, rtol=rtol, atol=atol), (
        f"{what}: the comparison ACCEPTED a difference it must reject "
        f"(max|diff| {delta:.3e} at rtol={rtol} atol={atol}). The check this "
        "controls for cannot fail, so its passing means nothing."
    )
    return delta


def test_blocking_does_not_change_the_arithmetic():
    """A forced small block agrees with the unblocked computation."""
    q, k, v = _qkv()
    whole = blocked_attention(q, k, v, HEADS, query_block_size=SEQ)
    blocked = blocked_attention(q, k, v, HEADS, query_block_size=32)
    assert whole.dtype == np.float32
    _agrees(whole, blocked, 0, BLOCKING_ATOL, "blocked vs unblocked")


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
        _agrees(mode, oracle, ORACLE_RTOL, ORACLE_ATOL, f"causal={causal}")


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


# ---------------------------------------------------------------------------
# NEGATIVE CONTROLS
#
# Everything above compares two things that agree. On this project a check that
# has never been shown able to fail is not evidence, so each control below feeds
# the SAME comparison at the SAME tolerance a difference a real defect would
# produce, and asserts it is REJECTED. They use only the production entry
# points -- no third implementation is written here, because a hand-written
# "defective variant" is one more thing that can silently stop matching what it
# models.
# ---------------------------------------------------------------------------


def test_negative_control_the_oracle_comparison_rejects_a_wrong_mask():
    """A mask argument silently ignored is the defect the causal arm exists to
    catch. Feeding the comparison a differently-masked oracle must fail it."""
    q, k, v = _qkv(seed=1)
    for causal in (False, True):
        mode = blocked_attention(q, k, v, HEADS, causal=causal, query_block_size=32)
        wrong = chunked_attention_reference(
            q, k, v, HEADS, causal=not causal, query_block_size=64
        )
        _rejects(mode, wrong, ORACLE_RTOL, ORACLE_ATOL, f"mask flipped, causal={causal}")


def test_negative_control_the_oracle_comparison_rejects_a_wrong_head_split():
    """The layout half. ``[seq, heads*head_dim]`` reinterpreted at the wrong
    head count is this codebase's documented integration failure, and it
    changes the arithmetic without changing the shape -- so the comparison,
    not the shape check, is what has to catch it."""
    q, k, v = _qkv(seed=1)
    for causal in (False, True):
        mode = blocked_attention(q, k, v, HEADS, causal=causal, query_block_size=32)
        wrong = chunked_attention_reference(
            q, k, v, HEADS * 2, causal=causal, query_block_size=64
        )
        _rejects(mode, wrong, ORACLE_RTOL, ORACLE_ATOL, f"head split, causal={causal}")


def test_negative_control_the_blocking_comparison_rejects_a_real_difference():
    """``test_blocking_does_not_change_the_arithmetic`` asserts at atol 1e-6 and
    the two sides agree EXACTLY (max|diff| 0.0 at this size), so that assertion
    has never been near its own bound. This shows the bound still discriminates:
    two genuinely different computations are rejected at the same 1e-6."""
    q, k, v = _qkv()
    whole = blocked_attention(q, k, v, HEADS, query_block_size=SEQ)
    masked = blocked_attention(q, k, v, HEADS, causal=True, query_block_size=32)
    _rejects(whole, masked, 0, BLOCKING_ATOL, "unblocked vs a differently-masked block")


def test_negative_control_the_scale_check_rejects_a_dropped_rsqrt():
    """``test_scale_is_rsqrt_head_dim`` compares against a direct ``math.exp``
    computation. Recomputing that reference WITHOUT the 1/sqrt(head_dim) must
    disagree, or the check would pass on an implementation that dropped it."""
    head_dim = 16
    rng = np.random.default_rng(3)
    q = rng.standard_normal((2, head_dim)).astype(bfloat16)
    k = rng.standard_normal((2, head_dim)).astype(bfloat16)
    v = rng.standard_normal((2, head_dim)).astype(bfloat16)
    out = blocked_attention(q, k, v, 1, query_block_size=2)

    qf, kf, vf = (np.asarray(t, dtype=np.float64) for t in (q, k, v))
    unscaled = qf[0] @ kf.T  # the 1/sqrt(head_dim) dropped
    weights = np.exp(unscaled - unscaled.max())
    weights /= weights.sum()
    _rejects(out[0], weights @ vf, ORACLE_RTOL, ORACLE_ATOL, "scale dropped")


def test_the_two_implementations_are_not_the_same_arithmetic():
    """The module docstring's independence rule, asserted rather than assumed.

    ``blocked_attention`` is torch-f32 and ``chunked_attention_reference`` is
    numpy-f32, deliberately implemented twice; folding them into one helper
    would compare a value against itself and pass no matter what is wrong. That
    collapse is invisible to every agreement check -- it makes them agree
    BETTER -- so the tell is the margin going to exactly zero. It is 3-5e-7
    here, from f32 reduction order, and it is not allowed to be 0.
    """
    q, k, v = _qkv(seed=1)
    mode = blocked_attention(q, k, v, HEADS, query_block_size=32)
    oracle = chunked_attention_reference(q, k, v, HEADS, query_block_size=64)
    delta = _max_abs(mode, oracle)
    assert delta > 0.0, (
        "the mode path and the oracle now agree BIT-EXACTLY, which two "
        "independent f32 implementations at different block sizes do not. "
        "Check that blocked_attention has not been folded onto "
        "chunked_attention_reference -- see this module's docstring."
    )
    assert delta < ORACLE_ATOL, f"clean margin {delta:.3e} is at the tolerance wall"
    print(
        f"  oracle-agreement margin: max|diff| {delta:.3e} "
        f"against atol {ORACLE_ATOL:.0e}"
    )


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"blocked attention tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
