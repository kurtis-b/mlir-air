# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""NumPy CPU attention reference shared by the llms/ prefill drivers.

`attention_reference` is the `cpu_attn=True` prefill fallback: full GQA
attention in F32, for the models whose semantics are plain causal GQA over
already-RoPE'd Q/K. Nine model dirs carried a byte-identical copy of it (two
spellings of one implementation, differing only in whether the causal mask is
added in one statement or two) plus a private `softmax` that nothing else
called; this is that single copy.

Not every model belongs here. `lfm2_1_2b_q4nx` keeps its own
`lfm2_1_2b_q4nx_cpu_attn.attention_reference` because its implementation
differs, and `smolvla` needs bidirectional MHA rather than causal GQA. RMSNorm
stays per-model on purpose: its `eps` is a config value (Qwen 1e-6, Llama and
SmolLM2 1e-5) that callers take from the default, so a shared default would
silently change a model's numerics.
"""

import numpy as np


def softmax(x, axis=-1):
    """Numerically stable softmax (F32)."""
    x = np.asarray(x, dtype=np.float32)
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def attention_reference(q, k, v, n_heads, n_kv_heads):
    """GQA attention with a causal mask (F32).

    Args:
        q: (seq_len, n_heads * head_dim) -- already projected and RoPE'd
           (and bias-added, where the model has QKV bias).
        k: (seq_len, n_kv_heads * head_dim) -- already projected and RoPE'd.
        v: (seq_len, n_kv_heads * head_dim) -- already projected.
        n_heads: number of query heads.
        n_kv_heads: number of key/value heads (for GQA).

    Returns:
        (seq_len, n_heads * head_dim) attention output (F32).
    """
    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    seq_len = q.shape[0]
    head_dim = q.shape[1] // n_heads
    group_size = n_heads // n_kv_heads
    q = q.reshape(seq_len, n_heads, head_dim).transpose(1, 0, 2)
    k = k.reshape(seq_len, n_kv_heads, head_dim).transpose(1, 0, 2)
    v = v.reshape(seq_len, n_kv_heads, head_dim).transpose(1, 0, 2)
    scale = 1.0 / np.sqrt(head_dim)
    causal_mask = np.triu(np.full((seq_len, seq_len), -np.inf, dtype=np.float32), k=1)
    out_heads = np.empty((n_heads, seq_len, head_dim), dtype=np.float32)
    for h in range(n_heads):
        kv_idx = h // group_size
        scores = q[h] @ k[kv_idx].T * scale
        scores = scores + causal_mask
        probs = softmax(scores, axis=-1)
        out_heads[h] = probs @ v[kv_idx]
    return out_heads.transpose(1, 0, 2).reshape(seq_len, n_heads * head_dim)
