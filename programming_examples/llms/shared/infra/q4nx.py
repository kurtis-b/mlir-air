# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Q4NX bundle helpers shared by the q4nx model dirs.

Three leaf helpers that the q4nx family carried one copy of each:

- `requant_q4k` — the affine re-quantizer, `w ~= sc*q + mn`, q in [0,15], one
  (sc, mn) per group along the reduction axis. It is the ONLY quantizer in
  these examples: prefill weights and the decode cascade cache both go through
  it, so the two see bit-identical values.
- `proj_dims` — logical (out, K) per projection, for I8-packed Q4NX headers.
- `resolve_q4nx_model` — resolve a repo id / directory / file path to a local
  `model.q4nx`.

`llama32_1b_q4nx` deliberately keeps its own `resolve_q4nx_model`: it adds
revision pinning (`_PINNED_Q4NX_REVISION`) and a cached-snapshot fallback,
which are model-specific policy rather than a different spelling of this.
"""

import os

import numpy as np


def requant_q4k(Wm, group):
    """Per-group min/max 4-bit re-quant of a matrix [M, K] -> (q, sc, mn)."""
    M, Kc = Wm.shape
    Wg = Wm.reshape(M, Kc // group, group)
    mn = Wg.min(2)
    mx = Wg.max(2)
    sc = (mx - mn) / 15.0
    sc = np.where(sc <= 0, 1.0, sc).astype(np.float32)
    q = np.clip(np.round((Wg - mn[..., None]) / sc[..., None]), 0, 15).astype(np.uint8)
    return q.reshape(M, Kc), sc, mn.astype(np.float32)


def proj_dims(c):
    """Logical (out, K) per projection, for I8-packed Q4NX headers."""
    dq = c.n_heads * c.head_dim
    dkv = c.n_kv_heads * c.head_dim
    return {
        "q": (dq, c.emb_dim),
        "k": (dkv, c.emb_dim),
        "v": (dkv, c.emb_dim),
        "o": (c.emb_dim, dq),
        "gate": (c.hidden_dim, c.emb_dim),
        "up": (c.hidden_dim, c.emb_dim),
        "down": (c.emb_dim, c.hidden_dim),
    }


def resolve_q4nx_model(model):
    """Resolve `model` to a local model.q4nx path.

    `model` may be an HF repo id (contains '/'), a directory containing
    model.q4nx, or a direct file path.
    """
    if os.path.isfile(model):
        return model
    if os.path.isdir(model):
        p = os.path.join(model, "model.q4nx")
        if os.path.isfile(p):
            return p
    from huggingface_hub import hf_hub_download

    return hf_hub_download(model, "model.q4nx")
