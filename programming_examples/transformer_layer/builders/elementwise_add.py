# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""2-D element-wise add, and the causal mask that rides on it.

CONTRACT
    ``build_elementwise_add_module(rows, cols, np_dtype, ...)`` returns an
    ``air.ir.Module`` whose single function takes three ``[rows, cols]`` bf16
    memrefs and computes ``out = a + b`` element-wise. All three arguments stay
    2-D at the L3 boundary; the collapse to 1-D happens inside the launch, so a
    downstream launch reads the output without an ``expand_shape``.

    ``causal_mask=True`` does not change the device design at all. It asserts
    the square shape an attention-score tensor has and records the intent; the
    mask itself is host data, produced by ``causal_mask_bias`` and bound as the
    ``b`` operand.

WHY causal_mask IS A KEYWORD AND NOT AN OPERATOR
    iron's ``AIECausalMask`` subclasses ``AIEElementwiseAdd`` and adds a
    precomputed triangular tensor as a static second input; it has no
    ``design.py``, i.e. no device design whatsoever. Porting convention 8 says
    to delete that redundancy on the way in, so it is a keyword on this builder
    instead of a sixth operator. iron's own full-suite sweep has zero rows for
    it, which is the other half of the argument.

FOOTGUNS
    - The mask fills with ``-10000.0``, not ``-inf``. This is deliberate and is
      iron's choice too: the mask is *added* to the scores, and ``-inf`` in bf16
      propagates to NaN through that add as soon as a score is ``+inf`` or the
      softmax's row max is itself ``-inf`` (an all-masked row). ``-10000.0``
      underflows the softmax to zero just as effectively and stays finite.
    - ``rows`` must be divisible by ``herd_x * herd_y`` -- each tile takes a
      contiguous ``rows*cols / tiles`` chunk and walks it a row at a time, so a
      remainder is silently unprocessed rather than diagnosed.
    - ``herd_y`` must be 1. Each tile uses three independent shim DMAs (a in,
      b in, out); ``herd_y > 1`` exhausts the shim DMA channels and aircc fails
      to place with "out of channels". This is the same limit the standalone
      ``eltwise_add`` example hit and recorded in the registry.
    - The device add is a single bf16 rounding of an f32 sum, so the reference
      below must round exactly once too. Adding in bf16 in the reference would
      make the check agree with the device for the wrong reason.
"""

import os
import sys

import numpy as np
import torch
from ml_dtypes import bfloat16

_PROJ_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if os.path.join(_PROJ_ROOT, "llms") not in sys.path:
    sys.path.insert(0, os.path.join(_PROJ_ROOT, "llms"))

from shared.builders.o_ffn_multi import _build_add_2d_to_2d  # noqa: E402

# iron's fill value (iron/operators/causal_mask/op.py). See the module footgun
# note: -inf in bf16 turns the subsequent add into NaN, -10000.0 does not.
CAUSAL_MASK_FILL = -10000.0


def build_elementwise_add_module(
    rows,
    cols,
    np_dtype=bfloat16,
    causal_mask=False,
    vector_size=16,
    herd_x=8,
    herd_y=1,
):
    """Build the 2-D element-wise add ``out = a + b`` over an ``herd_x`` herd.

    Args:
        rows, cols: the L3 tensor shape. ``rows`` must divide by
            ``herd_x * herd_y`` and ``cols`` by ``vector_size``.
        np_dtype: element type; bf16 in every transformer-block use.
        causal_mask: when True, ``b`` is an additive causal mask rather than a
            residual. Asserts the square shape attention scores have. The
            emitted MLIR is identical either way -- see the module docstring.
        vector_size: AIE vector lane count for the bf16 elementwise add.
        herd_x, herd_y: AIE tile grid. ``herd_y`` must be 1.

    Returns:
        air.ir.Module with one function ``eltwise_add_2d(a, b, out)``.
    """
    if causal_mask and rows != cols:
        raise ValueError(
            f"causal_mask=True needs a square [seq, seq] score tensor, got "
            f"[{rows}, {cols}]"
        )
    if herd_y != 1:
        raise ValueError(
            f"herd_y must be 1 (three shim DMAs per tile); got herd_y={herd_y}"
        )
    tiles = herd_x * herd_y
    if rows % tiles:
        raise ValueError(f"rows ({rows}) must be divisible by herd_x*herd_y ({tiles})")
    if cols % vector_size:
        raise ValueError(f"cols ({cols}) must be divisible by vector_size")

    # Reused verbatim from llms/shared/builders/o_ffn_multi.py rather than
    # re-derived: this is the same 2-D-in / 2-D-out add llama's prefill residual
    # already runs, and porting convention 8 says to import the shared one.
    return _build_add_2d_to_2d(
        rows, cols, np_dtype, vector_size=vector_size, herd_x=herd_x, herd_y=herd_y
    )


def causal_mask_bias(seq_len, np_dtype=bfloat16, fill=CAUSAL_MASK_FILL):
    """The precomputed additive causal mask, ``[seq_len, seq_len]``.

    Zero on and below the diagonal, ``fill`` above it, so that adding it to a
    score row leaves the causal entries untouched and drives the rest to zero
    through the softmax.

    Precomputed with torch (as iron does) and constant for a given
    ``seq_len`` -- in a ``KernelCache`` deployment it is registered once as a
    content-keyed static buffer and never re-uploaded, which is the whole
    reason it is a static second input rather than a runtime argument.
    """
    mask = torch.full((seq_len, seq_len), float(fill), dtype=torch.float32)
    mask = torch.triu(mask, diagonal=1)
    return mask.numpy().astype(np_dtype)


def elementwise_add_reference(a, b):
    """FP32 reference for ``out = a + b``, rounded once to the input dtype.

    Matches ``torch.add`` on bf16 tensors: an f32 sum of two exact-bf16
    operands with a single rounding back. There is no accumulation here, so
    this is the whole numerical story -- the registry records
    ``mean_rel_L1 = 1.9e-3`` for it, the cleanest kernel it holds.
    """
    return (a.astype(np.float32) + b.astype(np.float32)).astype(a.dtype)
