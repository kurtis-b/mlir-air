# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Small NumPy CPU helpers shared by production prefill/decode + verify.

Mirrors llama32_1b_cpu_helpers.py. Kept helpers are the ones production
prefill/decode import at runtime:

  - rms_norm           : LM-head GEMV final-norm + (Qwen3) QK-norm building block.
  - qk_norm_per_head   : Qwen3 per-head RMSNorm over head_dim, applied to Q and K
                         after projection and before RoPE.
"""

import numpy as np


def rms_norm(x, weight, eps=1e-6):
    """RMS normalization: x / sqrt(mean(x^2) + eps) * weight.

    Qwen3 uses eps=1e-6 (rms_norm_eps in config.json).
    """
    x = np.asarray(x, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def qk_norm_per_head(x, weight, n_heads, head_dim, eps=1e-6):
    """Qwen3 per-head RMSNorm.

    Args:
        x: (seq, n_heads*head_dim) projected Q or K (pre-RoPE).
        weight: (head_dim,) q_norm or k_norm weight.
    Returns:
        (seq, n_heads*head_dim) normed, same layout.
    """
    x = np.asarray(x, dtype=np.float32)
    seq = x.shape[0]
    xh = x.reshape(seq, n_heads, head_dim)
    rms = np.sqrt(np.mean(xh * xh, axis=-1, keepdims=True) + eps)
    xh = (xh / rms) * np.asarray(weight, dtype=np.float32)
    return xh.reshape(seq, n_heads * head_dim)
