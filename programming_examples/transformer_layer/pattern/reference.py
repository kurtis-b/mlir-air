# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The transformer-layer golden model: iron's structure, this repository's numerics.

CONTRACT
    ``generate_golden_reference(seq_len, hidden_size, intermediate_size,
    num_heads, ...)`` returns one dict::

        "input"           bf16 [seq_len, hidden_size]
        "attention_mask"  bf16 [seq_len, seq_len], or None
        "weights"         dict of bf16 tensors under iron's own names
        "output"          bf16 [seq_len, hidden_size], or None
        "boundaries"      dict name -> array, one entry per operator boundary,
                          in execution order, or None

    Every tensor is a numpy array (``ml_dtypes.bfloat16`` where the device sees
    bf16), because everything downstream of this file -- the builders, their
    oracles, ``opcheck.py`` -- is numpy. Only the random draws are torch; see
    below.

WHY THE DRAWS ARE torch AND THE ARITHMETIC IS NOT
    The draw ORDER is load-bearing: ``torch.manual_seed(seed)`` then, in order,
    ``input``, ``q_weight``, ``k_weight``, ``v_weight``, ``attn_output_weight``,
    ``ln1_weight``, ``ffn_up_weight``, ``ffn_down_weight``, ``ln2_weight``.
    Reordering any of them changes every tensor after it, so a study run here
    would stop being comparable with the same configuration run in iron. Keeping
    torch's generator is what preserves that.

    ``ln*_weight`` is ``torch.rand`` (uniform on [0, 1)), NOT ``randn``, and
    ``val_range = 0.05`` scales the ``randn`` draws. The biases are
    ``torch.zeros``, which is why the device operators having no bias term is not
    a gap -- and because a zeros tensor consumes no RNG, the bias draws do not
    appear in the order above and must not be added to it.

WHY THIS IS NOT A VERBATIM PORT OF iron's reference.py
    iron's takes ``dtype: str = "bf16"`` and builds EVERY tensor at that dtype,
    then runs the whole layer in it. That is the defect corrected in
    ``docs/plans/transformer-layer-execution-studies/06-phase-c-operators.md``,
    and it is worse for a whole layer than it was per operator: this chain is
    eight GEMMs, two LayerNorms and a softmax, so a bf16 oracle accumulates error
    in the same direction as the device and the comparison flatters itself the
    whole way down. Here the draws are f32, rounded ONCE to bf16 -- which is what
    the device is actually given -- and the arithmetic is f32.

    Two more differences from iron's, both deliberate:

    - Its GeLU is ``torch.nn.functional.gelu``, the **erf** form. The device
      kernel is the **tanh** approximation, and the two differ by ~1e-3 absolute
      around |x| = 2. At iron's 4e-2 tolerance that hides; at this repository's
      1.6e-2 it does not. ``builders/gelu.py::gelu_tanh_reference`` is what the
      FFN leg actually approximates, and it is what is called below.
    - It returns only the final output. This returns every operator boundary,
      because an end-to-end comparison over a layer that ENDS in a LayerNorm can
      absorb a great deal of upstream damage before it trips -- C4 found a GEMM
      that returned zeros for two of the nine sub-tiles of each cast worker while
      still producing a plausibly-shaped output.

WHY THE ARITHMETIC IS DELEGATED TO THE OPERATOR ORACLES
    Each boundary below is computed by the SAME reference function that
    ``opcheck.py`` validated that operator against on hardware in Phase D1 --
    ``qkv_proj_reference``, ``chunked_attention_reference``,
    ``o_proj_reference``, ``addnorm_pre_add_reference``, ``ffn_reference``'s two
    halves, ``elementwise_add_reference``. Re-deriving the same arithmetic here
    would be a second implementation to keep in step with the first, and the
    failure mode of letting them drift is a golden model that disagrees with the
    per-operator gate for reasons nobody can localize.

    ``test_reference.py`` beside this file pins the composition against a
    straight-line torch transcription of iron's structure, so "delegated" cannot
    silently become "different".

WHERE THE bf16 ROUNDINGS SIT, AND WHY IT MATTERS
    The device stages every OPERATOR boundary in a bf16 DDR buffer, and nothing
    inside an operator. This chain rounds in exactly those places and nowhere
    else, because each operator oracle already rounds once on the way out. Two
    boundaries are therefore f32 here and bf16 on the device:

        attn_context   the attention half's output, which the fused
                       ``mha_out_proj`` ELF stages in bf16 before its projection
        ffn_up/ffn_gelu the FFN's interior, staged in bf16 twice over

    That gap is MEASURED rather than defined away -- it is a real term in those
    operators' recorded error, and rounding it here would make the check agree
    with the device for the wrong reason. See ``builders/ffn.py`` and
    ``builders/mha_out_proj.py``, which make the same choice for the same reason.

FOOTGUNS
    - ``include_output=False`` is an ESCAPE HATCH, not dead code: it skips
      materializing the attention pipeline so the 16384 rung stays tractable for
      Phase F's runner. It still draws every weight, in the same order, so the
      tensors it does return are identical to the ``True`` case.
    - ``encoder_bert`` is POST-norm and non-causal; ``decoder_gpt2`` is PRE-norm
      and causal, and its residual stream is the raw sum rather than the
      normalized one. They share almost no control flow past the draws; do not
      collapse them into one path with two flags.
    - The device's ``addnorm`` operator is the PRE-ADD form,
      ``LayerNorm(x + residual) * weight``. iron writes the same thing as an add
      followed by ``F.layer_norm``. A block wired to the post-add form
      (``LayerNorm(x) * weight + residual``) produces a plausible activation that
      is wrong by one residual add; the two references live in separate functions
      in ``builders/addnorm.py`` for exactly that reason.
    - The fused QKV weight the device wants is ``fuse_qkv_weight(weights)``, in
      the column order ``(q, k, v)``. Building it by hand from the dict in a
      different order is not a shape error anywhere; it silently swaps two
      projections.
"""

import os
import sys

import numpy as np
from ml_dtypes import bfloat16

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE_ROOT = os.path.dirname(_HERE)  # programming_examples/transformer_layer/
_PROJ_ROOT = os.path.dirname(_EXAMPLE_ROOT)  # programming_examples/
for _p in (_PROJ_ROOT, os.path.join(_PROJ_ROOT, "llms"), _EXAMPLE_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from builders.addnorm import addnorm_pre_add_reference  # noqa: E402
from builders.elementwise_add import elementwise_add_reference  # noqa: E402
from builders.gelu import gelu_tanh_reference  # noqa: E402
from builders.mha_attention import chunked_attention_reference  # noqa: E402
from builders.o_proj import o_proj_reference  # noqa: E402
from builders.qkv_proj import qkv_proj_reference  # noqa: E402

#: iron's scale on every ``randn`` draw. Not applied to the ``rand`` LayerNorm
#: weights, which are uniform on [0, 1) as a trained gamma roughly is.
VAL_RANGE = 0.05

#: The weight names, in the order they are drawn. The order IS the contract; see
#: the module docstring.
WEIGHT_DRAW_ORDER = (
    "q_weight",
    "k_weight",
    "v_weight",
    "attn_output_weight",
    "ln1_weight",
    "ffn_up_weight",
    "ffn_down_weight",
    "ln2_weight",
)

#: Boundaries an ``encoder_bert`` layer passes through, in execution order.
ENCODER_BOUNDARIES = (
    "q",
    "k",
    "v",
    "attn_context",
    "attn_out",
    "hidden",
    "ffn_up",
    "ffn_gelu",
    "ffn_out",
    "output",
)

#: Same for ``decoder_gpt2``. ``ln_in`` is the pre-norm this variant applies to
#: the attention input and ``ffn_in`` the second pre-norm; ``encoder_bert`` has
#: neither, and its ``hidden`` has no counterpart here.
DECODER_BOUNDARIES = (
    "ln_in",
    "q",
    "k",
    "v",
    "attn_context",
    "attn_out",
    "residual",
    "ffn_in",
    "ffn_up",
    "ffn_gelu",
    "ffn_out",
    "output",
)

WORKLOAD_VARIANTS = ("encoder_bert", "decoder_gpt2")


def _to_numpy(tensor):
    """A torch tensor as a numpy array, bf16 included.

    ``numpy`` has no native bfloat16 and ``torch.Tensor.numpy()`` refuses the
    dtype, so the bits are reinterpreted through int16 into ``ml_dtypes``'.
    """
    import torch

    if tensor.dtype is torch.bfloat16:
        return tensor.view(torch.int16).numpy().view(bfloat16)
    return tensor.numpy()


def _draw(shape, uniform=False):
    """One draw in the fixed order, at f32, rounded once to bf16.

    f32 then a single rounding, not iron's draw-at-bf16: the device is given
    exact bf16 and the oracle computes in f32 from it, so this is where that
    single rounding belongs.
    """
    import torch

    if uniform:
        drawn = torch.rand(*shape, dtype=torch.float32)
    else:
        drawn = torch.randn(*shape, dtype=torch.float32) * VAL_RANGE
    return _to_numpy(drawn.to(torch.bfloat16))


def generate_golden_reference(
    seq_len,
    hidden_size,
    intermediate_size,
    num_heads,
    seed=42,
    workload_variant="encoder_bert",
    include_output=True,
    include_attention_mask=None,
):
    """The layer's inputs, weights and -- optionally -- every boundary it passes.

    Args:
        seq_len, hidden_size, intermediate_size, num_heads: the configuration.
            ``head_dim`` is ``hidden_size // num_heads`` and must divide evenly.
        seed: seeds torch's global generator, as iron's does.
        workload_variant: ``encoder_bert`` (post-norm, non-causal) or
            ``decoder_gpt2`` (pre-norm, causal).
        include_output: materialize the pipeline. ``False`` returns the same
            input and weights with ``output`` and ``boundaries`` set to ``None``,
            for a caller that only needs the tensors -- Phase F's runner at the
            long end of the sequence ladder.
        include_attention_mask: build the ``[seq_len, seq_len]`` mask. Defaults
            to ``include_output``. It is iron's dense mask and is descriptive
            only: the device path takes causality as a builder flag rather than
            as a tensor, and ``encoder_bert``'s mask is all ones.

    Returns:
        dict as described in the module docstring.
    """
    import torch

    if workload_variant not in WORKLOAD_VARIANTS:
        raise ValueError(
            f"unsupported workload_variant {workload_variant!r}; expected one of "
            f"{WORKLOAD_VARIANTS}"
        )
    if hidden_size % num_heads:
        raise ValueError(
            f"hidden_size ({hidden_size}) must be divisible by num_heads "
            f"({num_heads})"
        )

    if include_attention_mask is None:
        include_attention_mask = include_output

    torch.manual_seed(seed)
    # THE ORDER BELOW IS THE CONTRACT. See the module docstring.
    input_tensor = _draw((seq_len, hidden_size))
    attention_mask = None
    if include_attention_mask:
        # Drawn from no RNG, so its position here is free. It stays beside the
        # input because that is where iron builds it and a reader comparing the
        # two files should not have to hunt for it.
        if workload_variant == "decoder_gpt2":
            mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.float32))
        else:
            mask = torch.ones(seq_len, seq_len, dtype=torch.float32)
        attention_mask = _to_numpy(mask.to(torch.bfloat16))

    weights = {}
    for name in WEIGHT_DRAW_ORDER:
        if name in ("ln1_weight", "ln2_weight"):
            weights[name] = _draw((hidden_size,), uniform=True)
        elif name == "ffn_up_weight":
            weights[name] = _draw((hidden_size, intermediate_size))
        elif name == "ffn_down_weight":
            weights[name] = _draw((intermediate_size, hidden_size))
        else:
            weights[name] = _draw((hidden_size, hidden_size))

    result = {
        "input": input_tensor,
        "attention_mask": attention_mask,
        "weights": weights,
        "output": None,
        "boundaries": None,
    }
    if not include_output:
        return result

    boundaries = (
        _encoder_bert_boundaries(input_tensor, weights, num_heads)
        if workload_variant == "encoder_bert"
        else _decoder_gpt2_boundaries(input_tensor, weights, num_heads)
    )
    result["boundaries"] = boundaries
    result["output"] = boundaries["output"]
    return result


def fuse_qkv_weight(weights):
    """The ``[hidden, 3 * hidden]`` weight ``build_qkv_proj_module`` takes.

    Column order ``(q, k, v)``, matching ``builders/qkv_proj.py::QKV_NAMES`` and
    the split-cast launches' column offsets. A different order is not a shape
    error anywhere; it silently swaps two projections.
    """
    return np.concatenate(
        [weights["q_weight"], weights["k_weight"], weights["v_weight"]], axis=1
    )


def _attention(x, weights, num_heads, causal):
    """``q, k, v, attn_context, attn_out`` from one activation.

    Shared by both variants, which differ only in what ``x`` is and whether the
    mask is causal. ``attn_context`` is left in f32 -- see the module docstring
    on where the bf16 roundings sit.
    """
    q, k, v = qkv_proj_reference(x, fuse_qkv_weight(weights))
    context = chunked_attention_reference(q, k, v, num_heads, causal=causal)
    attn_out = o_proj_reference(context, weights["attn_output_weight"]).astype(x.dtype)
    return q, k, v, context, attn_out


def _ffn(x, weights):
    """``ffn_up, ffn_gelu, ffn_out`` from one activation.

    The interior stays f32 for the same reason ``builders/ffn.py``'s oracle does:
    the device stages it in bf16 twice over and measuring that gap is the point.
    """
    up = x.astype(np.float32) @ weights["ffn_up_weight"].astype(np.float32)
    activated = gelu_tanh_reference(up)
    out = (activated @ weights["ffn_down_weight"].astype(np.float32)).astype(x.dtype)
    return up, activated, out


def _encoder_bert_boundaries(x, weights, num_heads):
    """Post-norm, non-causal. Both normalization points are PRE-ADD addnorm."""
    q, k, v, context, attn_out = _attention(x, weights, num_heads, causal=False)
    hidden = addnorm_pre_add_reference(attn_out, x, weights["ln1_weight"])
    up, activated, ffn_out = _ffn(hidden, weights)
    output = addnorm_pre_add_reference(ffn_out, hidden, weights["ln2_weight"])
    return dict(
        zip(
            ENCODER_BOUNDARIES,
            (q, k, v, context, attn_out, hidden, up, activated, ffn_out, output),
        )
    )


def _decoder_gpt2_boundaries(x, weights, num_heads):
    """Pre-norm, causal, and the residual stream is the RAW sum.

    Two things differ from ``encoder_bert`` beyond the mask. The norms come
    BEFORE the sublayers they feed, so they normalize ``x`` alone -- expressed
    here as the pre-add addnorm with a zero residual, which is the same function
    the device dispatches and not a second LayerNorm implementation. And the
    layer's residual stream is ``attn_out + x``, unnormalized, so the final
    boundary is a plain elementwise add rather than a second norm.
    """
    zeros = np.zeros_like(x)
    ln_in = addnorm_pre_add_reference(x, zeros, weights["ln1_weight"])
    q, k, v, context, attn_out = _attention(ln_in, weights, num_heads, causal=True)
    residual = elementwise_add_reference(attn_out, x)
    ffn_in = addnorm_pre_add_reference(residual, zeros, weights["ln2_weight"])
    up, activated, ffn_out = _ffn(ffn_in, weights)
    output = elementwise_add_reference(ffn_out, residual)
    return dict(
        zip(
            DECODER_BOUNDARIES,
            (
                ln_in,
                q,
                k,
                v,
                context,
                attn_out,
                residual,
                ffn_in,
                up,
                activated,
                ffn_out,
                output,
            ),
        )
    )
