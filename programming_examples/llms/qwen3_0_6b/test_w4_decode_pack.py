# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host tests for `w4_decode_pack.py`, re-homed from the study branch's
`transformer_layer/study/test_w4_decode_pack.py` (study commit 0d08d195):
the selection flag (default, typo refusal, and that `QWEN3_W4_DECODE=0`
selects nothing -- the loader's ONE `quantize_decode_weights` call sits
under `w4_decode_selected()`), the quant contract columns, and the
round-trip dequant band. Study tests bound to the decode driver, the lit
gates, or the int4 cascade builder are NOT ported here: their subjects
land in later slices."""

import ast
import os
import sys

import numpy as np
from ml_dtypes import bfloat16

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import w4_decode_pack as wp  # noqa: E402


def _selected(value):
    """Run the production predicate with QWEN3_W4_DECODE set to `value`
    (None = unset), restoring the environment afterwards."""
    old = os.environ.pop(wp.W4_ENV, None)
    try:
        if value is not None:
            os.environ[wp.W4_ENV] = value
        return wp.w4_decode_selected()
    finally:
        os.environ.pop(wp.W4_ENV, None)
        if old is not None:
            os.environ[wp.W4_ENV] = old


def test_flag_zero_selects_nothing():
    # R3a ships the pack default-OFF: the flip to w4_decode is R3c's, landing
    # with the three-arm verify gate (review of PR #32, P1).
    assert wp.W4_DEFAULT is False
    assert _selected(None) is False
    assert _selected("") is False  # the `FOO= cmd` shell idiom
    assert _selected("1") is True
    assert _selected("0") is False, "QWEN3_W4_DECODE=0 must select bf16"
    try:
        _selected("yes")
    except ValueError as e:
        assert wp.W4_ENV in str(e)
    else:
        raise AssertionError("a typo flag value was silently accepted")
    # ... and the loader's ONLY quantize_decode_weights call is guarded by
    # that same predicate, so False really does select nothing.
    with open(os.path.join(_HERE, "qwen3_0_6b_weights.py")) as f:
        tree = ast.parse(f.read())
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "load_weights"
    )
    calls = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", None) == "quantize_decode_weights"
    ]
    guards = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Call)
        and getattr(n.test.func, "id", None) == "w4_decode_selected"
    ]
    assert len(calls) == 1 and len(guards) == 1, (len(calls), len(guards))
    assert any(n is calls[0] for n in ast.walk(guards[0])), "call not under the guard"


def test_quant_contract_columns():
    c = wp.quant_contract()
    assert c["quant_group_size"] == wp.GROUP_SIZE == 128
    assert "asymmetric" in c["quant_zero_point_layout"]
    assert "uint8 [K/gs, M]" in c["quant_zero_point_layout"]
    assert c["quant_gemv_contract_name"] == (
        "awq_u4_asym_g128_bf16s_u8z_dequant_in_kernel"
    )
    assert "(q - z) * s" in c["quant_gemv_contract"]
    assert "QKV and LM head stay bf16" in c["quant_gemv_contract"]
    try:
        wp.quant_contract(group_size=64)
    except ValueError:
        pass
    else:
        raise AssertionError("a contradicting group_size was accepted")


def test_roundtrip_dequant_band():
    """Quantize + dequantize through the production functions; cosine vs the
    original must stay >= 0.99 (measured 0.9949 at these seeds/shapes; a
    zero-point off by one group step lands near 0.69)."""
    rng = np.random.default_rng(7)
    for M, K in ((48, 256), (32, 512)):
        W = (rng.standard_normal((M, K)) * 0.02).astype(bfloat16)
        q, s, z = wp._fake_quantize(W)
        Wd = wp.dequant_rows(q, s, z)
        assert Wd.shape == W.shape and Wd.dtype == W.dtype
        a = W.astype(np.float64).ravel()
        b = Wd.astype(np.float64).ravel()
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        assert cos >= 0.99, f"round-trip cosine {cos:.5f} < 0.99 band ({M}x{K})"
        # non-vacuity: quantization actually changed the bytes.
        assert not np.array_equal(W.view(np.uint16), Wd.view(np.uint16))


def test_quantize_packs_production_shapes_bit_exact_and_idempotent():
    """quantize_decode_weights through the production `_fake_quantize` /
    `_pack` at qwen3-0.6b shapes: the packed BOs match the int4 cascade
    builder's own `_packed_dims` arithmetic (one disagreement and the ELF
    strides into garbage), the substituted bf16 fields ARE the dequantized
    copies bit-exact, QKV is untouched, and a second call is a no-op."""
    from llama32_1b_int4.multi_launch_builder.o_gemv_ffn_int4_multi import (
        _packed_dims,
    )

    class _NS:
        pass

    emb, hidden, kv = 1024, 3072, 1024
    cfg = _NS()
    cfg.emb_dim, cfg.hidden_dim, cfg.n_layers = emb, hidden, 1
    cfg.n_heads, cfg.head_dim = 16, 128
    q_dim = cfg.n_heads * cfg.head_dim
    rng = np.random.default_rng(11)
    lw = _NS()
    shapes = (
        ("wq", (emb, q_dim)),
        ("wk", (emb, kv)),
        ("wv", (emb, kv)),
        ("wo", (q_dim, emb)),
        ("w_gate", (emb, hidden)),
        ("w_up", (emb, hidden)),
        ("w_down", (hidden, emb)),
    )
    for name, shape in shapes:
        setattr(lw, name, (rng.standard_normal(shape) * 0.02).astype(bfloat16))
    weights = _NS()
    weights.layers = [lw]
    orig = {name: np.array(getattr(lw, name)) for name, _ in shapes}

    wp.quantize_decode_weights(weights, cfg)
    assert weights._w4_decode_applied is True
    for k in ("wq", "wk", "wv"):  # QKV untouched, bit for bit
        assert np.array_equal(
            getattr(lw, k).view(np.uint16), orig[k].view(np.uint16)
        ), k
    for packed, (M, K) in (
        (lw._wo_packed, (emb, q_dim)),
        (lw._wgateup_packed, (2 * hidden, emb)),
        (lw._wdown_packed, (emb, hidden)),
    ):
        want_dims = _packed_dims(
            M, K, wp.GROUP_SIZE, wp.M_TILE, wp.K_CHUNK, wp.N_CORES, M
        )
        assert packed.dtype == np.uint8 and packed.shape == want_dims, (M, K)
    # the qwen dims, concretely (the study's recorded byte arithmetic)
    assert lw._wo_packed.shape == (256, 4288)
    assert lw._wgateup_packed.shape == (768, 4288)
    assert lw._wdown_packed.shape == (384, 4288)
    for k in ("wo", "w_gate", "w_up", "w_down"):
        q, s, z = wp._fake_quantize(orig[k].T)
        want = wp.dequant_rows(q, s, z).T
        got = getattr(lw, k)
        assert np.array_equal(got.view(np.uint16), want.view(np.uint16)), k
        # non-vacuity: quantization actually changed the field
        assert not np.array_equal(got.view(np.uint16), orig[k].view(np.uint16)), k
    snap = lw.wo.copy()
    wp.quantize_decode_weights(weights, cfg)  # idempotent
    assert np.array_equal(lw.wo.view(np.uint16), snap.view(np.uint16))


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"w4_decode_pack tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
