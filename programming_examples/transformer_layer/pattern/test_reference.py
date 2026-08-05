# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only checks that the golden model is the layer it claims to be.

``reference.py`` computes each boundary by calling the operator oracle that
``opcheck.py`` validated that operator against on hardware. That removes one
failure mode -- two implementations of the same arithmetic drifting apart -- and
introduces another: nothing in the composition itself says the operators were
composed in the right ORDER, with the right tensor in each slot. A layer wired
with the post-add addnorm, or with q and k swapped in the fused weight, or with
the erf GeLU, is a perfectly well-typed composition that is not this layer.

So this file transcribes iron's structure straight through, in torch, with no
reference to ``reference.py``'s internals, and pins the composition against it.
Then it checks the three substitutions that would survive that comparison
because they are small: erf vs tanh GeLU, post-add vs pre-add, and the QKV
column order.

    python3 pattern/test_reference.py

No hardware, no MLIR, a couple of seconds. It is a ``test_*.py`` so ordinary
pytest discovery finds it too (porting convention 11).

FOOTGUN
    The straight-line transcription below rounds NOTHING between operators,
    while the composition rounds each operator boundary to bf16 -- that is the
    device's DDR staging and the whole reason the composition is built that way.
    So they agree to a bf16 rounding and not to floating-point equality, and the
    tolerance here is sized to that gap. Tightening it to exact equality does not
    make the test stronger; it makes it fail for the one reason it should not.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE_ROOT = os.path.dirname(_HERE)
if _EXAMPLE_ROOT not in sys.path:
    sys.path.insert(0, _EXAMPLE_ROOT)

from pattern.reference import (  # noqa: E402
    ENCODER_BOUNDARIES,
    generate_golden_reference,
)

# The block's own widths at a short sequence. The widths are not a free choice:
# they set the up-projection's scale, hence where on the GeLU curve the FFN
# operates, hence whether the erf/tanh substitution below is visible at all. At
# hidden 128 the up-projection lands inside +/-1, where the two GeLU forms agree
# to 2e-5 and the check proves nothing. Only `seq_len` is cut, and it costs
# nothing: every operator here is row-parallel.
SHAPE = dict(seq_len=64, hidden_size=768, intermediate_size=3072, num_heads=12)

# The composition rounds every operator boundary to bf16; the transcription
# rounds nothing. bf16 carries 8 mantissa bits, so a single rounding is 2^-9
# relative -- about 2e-3 -- and this layer stacks eight of them. 2e-2 is that
# with room, and it is still 100x tighter than the substitutions below move the
# output.
COMPOSITION_RTOL = 2e-2
COMPOSITION_ATOL = 2e-2


def _layer_norm(x, weight, eps=1e-5):
    """Two-pass f32 LayerNorm times a weight, no bias. torch, unrounded."""
    import torch

    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps) * weight


def _transcribe_encoder_bert(x, w, num_heads, gelu="tanh", residual="pre_add"):
    """iron's encoder_bert, straight through in f32 torch. No rounding.

    ``gelu`` and ``residual`` exist so the substitution checks below can ask
    this same function for the WRONG layer and measure how far off it is.
    """
    import torch

    seq_len, hidden = x.shape
    head_dim = hidden // num_heads

    def heads(t):
        return t.view(seq_len, num_heads, head_dim).transpose(0, 1)

    q, k, v = (heads(x @ w[n]) for n in ("q_weight", "k_weight", "v_weight"))
    scores = (q @ k.transpose(-2, -1)) / (head_dim**0.5)
    context = torch.softmax(scores, dim=-1) @ v
    context = context.transpose(0, 1).contiguous().view(seq_len, hidden)
    attn_out = context @ w["attn_output_weight"]

    if residual == "pre_add":
        hidden_states = _layer_norm(attn_out + x, w["ln1_weight"])
    else:
        hidden_states = _layer_norm(attn_out, w["ln1_weight"]) + x

    up = hidden_states @ w["ffn_up_weight"]
    if gelu == "tanh":
        activated = torch.nn.functional.gelu(up, approximate="tanh")
    else:
        activated = torch.nn.functional.gelu(up)
    ffn_out = activated @ w["ffn_down_weight"]

    if residual == "pre_add":
        return _layer_norm(ffn_out + hidden_states, w["ln2_weight"])
    return _layer_norm(ffn_out, w["ln2_weight"]) + hidden_states


def _as_torch(golden):
    """The golden model's bf16 tensors as f32 torch, for the transcription."""
    import torch

    def to_t(array):
        return torch.from_numpy(array.astype(np.float32))

    return to_t(golden["input"]), {k: to_t(v) for k, v in golden["weights"].items()}


def _rel(a, b):
    """Worst relative deviation of ``a`` from ``b``, ignoring near-zero ``b``."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    scale = max(float(np.abs(b).max()), 1e-30)
    return float(np.abs(a - b).max() / scale)


def test_composition_matches_a_straight_line_transcription():
    """The delegated composition IS iron's encoder_bert, order and all."""
    golden = generate_golden_reference(**SHAPE)
    x, w = _as_torch(golden)
    expected = _transcribe_encoder_bert(x, w, SHAPE["num_heads"]).numpy()
    actual = golden["output"].astype(np.float32)
    assert np.allclose(
        actual, expected, rtol=COMPOSITION_RTOL, atol=COMPOSITION_ATOL
    ), f"composition deviates from the transcription by {_rel(actual, expected):.3e}"


def test_the_gelu_is_the_tanh_approximation():
    """The activation is tanh, pinned by IDENTITY because no tolerance can pin it.

    iron's reference calls ``torch.nn.functional.gelu``'s erf default while the
    device kernel is the tanh approximation, and this is the substitution the
    port was most likely to make. It is also one that NO numerical check in this
    repository would have caught. Measured at these widths:

        boundary       max |erf - tanh|    atol the substitution needs
        ffn_gelu           4.73e-4                  3.65e-4
        layer output       6.06e-4                  2.89e-4

    Every ``atol`` this example uses is at least 1e-3, and most are 100x that,
    so a wrong activation would sail through the stage list AND the end-to-end
    comparison. "The numerical check would have caught it" is exactly the wrong
    conclusion to draw from a passing gate.

    So the assertions below are equality against ``gelu_tanh_reference``'s own
    output and inequality against the erf form -- identity, not tolerance. The
    lower bound on the substitution size is there only to notice if a future
    SHAPE drifts onto a part of the curve where the two forms coincide, which
    would make the inequality vacuous.
    """
    import torch

    golden = generate_golden_reference(**SHAPE)
    up = torch.from_numpy(golden["boundaries"]["ffn_up"])
    tanh_form = torch.nn.functional.gelu(up, approximate="tanh").numpy()
    erf_form = torch.nn.functional.gelu(up).numpy()
    actual = np.asarray(golden["boundaries"]["ffn_gelu"], dtype=np.float64)

    separation = float(np.abs(erf_form.astype(np.float64) - tanh_form).max())
    assert separation > 1e-4, (
        f"erf and tanh GeLU differ by only {separation:.3e} at this shape, so "
        f"the inequality below proves nothing -- the widths in SHAPE have "
        f"drifted off the part of the curve where the two forms disagree"
    )
    assert _rel(actual, tanh_form) < 1e-6, "the composition is not using tanh GeLU"
    assert (
        float(np.abs(actual - erf_form.astype(np.float64)).max()) > 1e-4
    ), "the composition matches the erf form"


def test_the_residual_ordering_is_pre_add():
    """``encoder_bert`` normalizes the SUM. The other ordering is a real error."""
    golden = generate_golden_reference(**SHAPE)
    x, w = _as_torch(golden)
    num_heads = SHAPE["num_heads"]
    pre = _transcribe_encoder_bert(x, w, num_heads, residual="pre_add").numpy()
    post = _transcribe_encoder_bert(x, w, num_heads, residual="post_add").numpy()
    assert _rel(post, pre) > COMPOSITION_RTOL
    assert _rel(golden["output"].astype(np.float32), pre) < COMPOSITION_RTOL


def test_the_fused_qkv_weight_is_in_q_k_v_column_order():
    """``fuse_qkv_weight`` must agree with the boundaries the model reports.

    The device's ``qkv_proj`` slices one fused weight into three column bands in
    ``QKV_NAMES`` order. A wrong order is not a shape error anywhere and swaps
    two projections silently, so it is checked against the q/k/v boundaries the
    golden model itself produced.
    """
    from pattern.reference import fuse_qkv_weight

    golden = generate_golden_reference(**SHAPE)
    hidden = SHAPE["hidden_size"]
    fused = fuse_qkv_weight(golden["weights"]).astype(np.float32)
    xf = golden["input"].astype(np.float32)
    for i, name in enumerate(("q", "k", "v")):
        band = xf @ fused[:, i * hidden : (i + 1) * hidden]
        assert np.array_equal(
            band.astype(golden["boundaries"][name].dtype),
            golden["boundaries"][name],
        ), f"fused column band {i} is not the {name} projection"


def test_boundaries_are_complete_and_shaped():
    """Every declared boundary exists, at the width the block dispatches."""
    golden = generate_golden_reference(**SHAPE)
    boundaries = golden["boundaries"]
    assert tuple(boundaries) == ENCODER_BOUNDARIES
    seq_len, hidden = SHAPE["seq_len"], SHAPE["hidden_size"]
    # Only the FFN's INTERIOR is intermediate-wide; ffn_out is back at hidden.
    interior = ("ffn_up", "ffn_gelu")
    for name, array in boundaries.items():
        width = SHAPE["intermediate_size"] if name in interior else hidden
        assert array.shape == (seq_len, width), f"{name} is {array.shape}"


def test_include_output_false_draws_the_same_tensors():
    """The escape hatch skips the pipeline, never the draws."""
    full = generate_golden_reference(**SHAPE)
    tensors_only = generate_golden_reference(**SHAPE, include_output=False)
    assert tensors_only["output"] is None
    assert tensors_only["boundaries"] is None
    assert np.array_equal(tensors_only["input"], full["input"])
    for name, weight in full["weights"].items():
        assert np.array_equal(tensors_only["weights"][name], weight), name


def test_decoder_gpt2_is_a_different_layer():
    """The second variant is pre-norm and causal, and is not the encoder."""
    encoder = generate_golden_reference(**SHAPE)
    decoder = generate_golden_reference(**SHAPE, workload_variant="decoder_gpt2")
    assert np.array_equal(decoder["input"], encoder["input"])
    assert (
        _rel(decoder["output"].astype(np.float32), encoder["output"].astype(np.float32))
        > COMPOSITION_RTOL
    )
    # iron builds a `tril` mask for this variant and an all-ones one for the
    # encoder; the device path takes causality as a builder flag, so the mask is
    # descriptive and this is the only thing that checks it.
    assert (
        decoder["attention_mask"].astype(np.float32).sum()
        < encoder["attention_mask"].astype(np.float32).sum()
    )


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"reference tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
