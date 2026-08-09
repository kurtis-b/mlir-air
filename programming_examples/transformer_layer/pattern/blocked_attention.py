# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Query-blocked host attention, shared by the ``offload`` and ``runlist`` modes.

CONTRACT
    ``blocked_attention(q, k, v, num_heads, ...)`` takes the three projections
    as ``[seq, num_heads * head_dim]`` bf16 numpy arrays -- exactly what a
    device GEMM hands back -- and returns the attention context in the same
    layout as FLOAT32 numpy, unrounded, computed in torch f32 on the host.
    Above a scratch threshold the computation blocks over query rows so a long
    sequence never materializes the full ``[heads, seq, seq]`` score tensor;
    ``resolve_query_block_size(seq_len, num_heads)`` is that decision, exposed
    so a mode can record the block size it ran at.

    Ported from iron's ``pattern/{offload,runlist}/op.py``, which share one
    ``_blocked_attention`` / ``_resolve_query_block_size`` pair so the two
    modes block identically. This module is that sharing, kept: it lives in
    ``pattern/`` rather than in ``offload/`` so ``runlist`` imports the same
    blocking, per 08c.

WHY THIS IS HOST MATH -- AND WHY THAT IS NO LONGER THE ONLY OPTION
    ``[2026-08-08]`` This module's original rationale said the two attention
    GEMMs "cannot be dispatched on this device", because at the gate
    configuration they are ``4096 x 64 x 4096`` and ``4096 x 4096 x 64`` and no
    ``K = 64`` or ``N = 64`` bf16-out row exists or can be swept. **The premise
    is right and the conclusion was wrong.** No such row exists and
    ``sweep_families.py`` genuinely cannot stage one -- it derives K and N from
    ``FAMILY_HIDDEN x ROLE_KN_MULTIPLES`` with a minimum hidden of 512 -- but
    that is a CATALOGUE constraint, not a hardware one. Both shapes are
    measured passing on real NPU2 at every rung of the study's ladder, and
    ``attn_output`` passes by all three GEMM methods. The route around the
    catalogue is the ``gemm_spec_fn`` escape hatch every builder here ships:
    inject the measured tiles, record ``gemm_spec_source: injected``.

    (A related claim that circulated with it -- that ``attn_scores`` "would
    need K = 64 against a minimum ``tile_k_l2`` of 256, which does not tile" --
    is also false. ``tile_k_l2 = 64`` is what passes, and at K = 64 it is
    forced, because K admits no other L2 tile.)

    So ``offload`` no longer calls this module: ``pattern/offload/offload.py``
    dispatches both matmuls and keeps only the softmax on the host, which is
    what the corrected taxonomy's linearity rule requires. This module remains
    the host-attention implementation for ``runlist``, which has not been
    rebuilt yet, and as the reference for what the host boundary cost.

WHY THE THRESHOLD IS SCRATCH BYTES, NOT iron's ``seq_len >= 16384``
    iron gates blocking on a hardcoded sequence length and then sizes the
    block from ``MAX_ATTENTION_SCRATCH_BUFFER_BYTES``. The port keys the
    trigger off the same constant instead: block exactly when the full f32
    score tensor -- ``seq * seq * num_heads * 4`` bytes, which is what this
    implementation would otherwise allocate -- exceeds the cap. At every
    rung of the study's ladder the two rules agree (4096 x 12 heads is
    ~805 MB, under the 3 GiB cap; 16384 x 12 is ~12.9 GB, over it), and the
    scratch form says WHY the threshold sits where it does on a 31 GB host.

THE ORACLE INDEPENDENCE RULE, WHICH IS NOT OPTIONAL
    The modes that call this are checked against
    ``builders/mha_attention.py::chunked_attention_reference`` through the
    golden model. This function must therefore never call it, nor share
    arithmetic with it: this one is torch on the host path, the oracle is
    numpy on the check path, and the same convention (``scale =
    1/sqrt(head_dim)``, causal masks ``kv_position > q_position``) is
    implemented twice ON PURPOSE. Folding them into one helper would compare
    a value against itself and pass no matter what is wrong with it -- the
    exact hazard 08c names.

FOOTGUNS
    - The bf16 -> torch bridge goes through an int16 view. numpy has no
      native bfloat16 and ``torch.from_numpy`` refuses ``ml_dtypes``' one, so
      the bits are reinterpreted, never value-converted.
    - When no divisor of ``seq_len`` at or above
      ``MIN_BLOCKED_QUERY_BLOCK_SIZE`` fits the cap, iron falls back to the
      LARGEST such divisor -- which is ``seq_len`` itself, i.e. unblocked.
      That fallback is preserved as-is: it only triggers where even minimum
      blocks overflow the cap, and silently choosing a different policy here
      would make the two repositories block differently at exactly the
      sequence lengths where the choice matters.
"""

import math

import numpy as np
from ml_dtypes import bfloat16

#: iron's cap on the attention score scratch, and its minimum useful block.
MAX_ATTENTION_SCRATCH_BUFFER_BYTES = 3 * 1024**3
MIN_BLOCKED_QUERY_BLOCK_SIZE = 256

#: The score scratch is materialized in f32 (see the module docstring).
_SCORE_ITEMSIZE = 4


def _descending_divisors(value):
    """Every divisor of ``value``, largest first."""
    divisors = set()
    for candidate in range(1, int(math.isqrt(value)) + 1):
        if value % candidate == 0:
            divisors.add(candidate)
            divisors.add(value // candidate)
    return tuple(sorted(divisors, reverse=True))


def resolve_query_block_size(seq_len, num_heads):
    """Query rows per attention block: ``seq_len`` until the scratch overflows.

    The scratch is the f32 score tensor for one block over every head,
    ``block * seq_len * num_heads * 4`` bytes. Unblocked (``block ==
    seq_len``) until that exceeds ``MAX_ATTENTION_SCRATCH_BUFFER_BYTES``;
    then the largest divisor of ``seq_len`` that is at least
    ``MIN_BLOCKED_QUERY_BLOCK_SIZE`` and fits the cap; then iron's fallback
    (see the module footguns).
    """
    per_row = _SCORE_ITEMSIZE * seq_len * num_heads
    if seq_len * per_row <= MAX_ATTENTION_SCRATCH_BUFFER_BYTES:
        return seq_len
    cap = MAX_ATTENTION_SCRATCH_BUFFER_BYTES // per_row
    for candidate in _descending_divisors(seq_len):
        if MIN_BLOCKED_QUERY_BLOCK_SIZE <= candidate <= cap:
            return candidate
    for candidate in _descending_divisors(seq_len):
        if candidate >= MIN_BLOCKED_QUERY_BLOCK_SIZE:
            return candidate
    return seq_len


def _heads_f32(array, num_heads):
    """``[seq, heads * head_dim]`` bf16 numpy -> ``[heads, seq, head_dim]`` torch f32."""
    import torch

    seq_len, emb_dim = array.shape
    bits = torch.from_numpy(np.ascontiguousarray(array).view(np.int16))
    tensor = bits.view(torch.bfloat16).to(torch.float32)
    return tensor.view(seq_len, num_heads, emb_dim // num_heads).transpose(0, 1)


def blocked_attention(q, k, v, num_heads, causal=False, query_block_size=None):
    """Scaled-dot-product attention on the host, blocked over query rows.

    Args:
        q, k, v: ``[seq, num_heads * head_dim]`` bf16 numpy arrays, seq-first
            -- the layout the device GEMMs produce.
        num_heads: head count. MHA only; the layer this serves has no GQA.
        causal: mask ``kv_position > q_position``.
        query_block_size: rows per block. ``None`` resolves it from the
            scratch cap; a mode passes its recorded value so the run and the
            artifact cannot disagree.

    Returns:
        ``[seq, num_heads * head_dim]`` FLOAT32 numpy, unrounded. The caller
        owns any bf16 rounding, because what it feeds downstream (a device
        GEMM's ``A`` operand) is where that rounding actually happens.
    """
    import torch

    seq_len, emb_dim = q.shape
    head_dim = emb_dim // num_heads
    if query_block_size is None:
        query_block_size = resolve_query_block_size(seq_len, num_heads)

    q_heads = _heads_f32(q, num_heads)
    k_heads = _heads_f32(k, num_heads)
    v_heads = _heads_f32(v, num_heads)
    scale = 1.0 / math.sqrt(head_dim)

    context = torch.empty_like(q_heads)
    for start in range(0, seq_len, query_block_size):
        end = start + query_block_size
        # The scratch the block size exists to bound: [heads, block, seq] f32.
        scores = q_heads[:, start:end, :] @ k_heads.transpose(-2, -1) * scale
        if causal:
            kv_positions = torch.arange(seq_len)
            q_positions = torch.arange(start, end)
            scores = scores.masked_fill(
                kv_positions.unsqueeze(0) > q_positions.unsqueeze(1), float("-inf")
            )
        context[:, start:end, :] = torch.softmax(scores, dim=-1) @ v_heads

    return context.transpose(0, 1).reshape(seq_len, emb_dim).numpy()


def round_bf16(array_f32):
    """f32 numpy -> bf16 numpy, the single rounding a device operand takes."""
    return np.asarray(array_f32).astype(bfloat16)
