# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Small NumPy CPU helpers for Qwen2.5-0.5B production prefill/decode + verify.

Mirrors llama32_1b_cpu_helpers.py. Qwen2.5 has QKV bias (no QK-norm).
Kept helper: rms_norm (attention_reference now lives in
`shared/infra/cpu_attn.py`). The QKV bias is fused
on-device inside the rms_qkv_bias_rope(_gemv) ELF (see prefill/decode), so no
dedicated host bias helper is needed here.
"""

import numpy as np


def rms_norm(x, weight, eps=1e-6):
    """RMS norm; Qwen2.5 uses eps=1e-6 (rms_norm_eps)."""
    x = np.asarray(x, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight
