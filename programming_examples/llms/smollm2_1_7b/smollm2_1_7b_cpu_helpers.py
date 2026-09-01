# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Small NumPy CPU helpers shared by production prefill/decode + verify.

This file used to be a full F32 CPU forward-pass implementation of the model
(plus a standalone `--verify` CLI that compared the F32 forward against HF
transformers F32). With the verify subsystem rewritten to compare directly
against HF transformers in bf16 (see verify/), that whole F32 reference
chain became redundant. What is kept here is the small set of NumPy helpers
that production still imports:

  - rms_norm           : LM-head GEMV final-norm (inference.py prefill end,
                         and every decode step).
"""

import numpy as np


def rms_norm(x, weight, eps=1e-5):
    """RMS normalization: x / sqrt(mean(x^2) + eps) * weight.

    Args:
        x: (M, N) input array in F32.
        weight: (N,) learned scale parameter.
        eps: Small constant for numerical stability.

    Returns:
        (M, N) normalized and scaled array in F32.
    """
    x = np.asarray(x, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight
