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
    assert _selected(None) is wp.W4_DEFAULT
    assert _selected("") is wp.W4_DEFAULT  # the `FOO= cmd` shell idiom
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


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"w4_decode_pack tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
