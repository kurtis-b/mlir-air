# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host tests for `qwen3_0_6b/w4_decode_pack.py` (doc 56 H2b, queue item 18):
the flag default, the packed-BO dims against the BUILDER's own arithmetic
(the two must agree or the ELF reads garbage), the dequant substitution's
bit-exactness (prefill and the HF oracle compute on EXACTLY what the kernel
dequants), and idempotence."""

from __future__ import annotations

import os
import sys

import numpy as np
from ml_dtypes import bfloat16

_HERE = os.path.dirname(os.path.abspath(__file__))
_PE = os.path.dirname(os.path.dirname(_HERE))
for _p in (_PE, os.path.join(_PE, "llms"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qwen3_0_6b import w4_decode_pack as wp  # noqa: E402


def test_flag_defaults_off_and_reads_env_at_call_time():
    old = os.environ.pop(wp.W4_ENV, None)
    try:
        assert wp.w4_decode_selected() is False, "bf16 must stay the default"
        os.environ[wp.W4_ENV] = "1"
        assert wp.w4_decode_selected() is True
        os.environ[wp.W4_ENV] = "0"
        assert wp.w4_decode_selected() is False
    finally:
        if old is None:
            os.environ.pop(wp.W4_ENV, None)
        else:
            os.environ[wp.W4_ENV] = old


def test_packed_bo_dims_equal_the_builders_arithmetic():
    """The packer's output shape for each of the three matrices equals the
    int4 cascade builder's `_packed_dims` at the SAME (gs, m_tile, k_chunk,
    n_cores) -- one disagreement and the ELF strides into garbage."""
    from llama32_1b_int4.multi_launch_builder.o_gemv_ffn_int4_multi import _packed_dims

    emb, q_dim, hidden = 1024, 2048, 3072
    rng = np.random.default_rng(3)
    for M, K in ((emb, q_dim), (2 * hidden, emb), (emb, hidden)):
        W = (rng.standard_normal((M, K)) * 0.02).astype(bfloat16)
        q, s, z = wp._fake_quantize(W)
        packed = wp._pack(q, s, z)
        tiles, tile_bytes = _packed_dims(M, K, wp.GROUP_SIZE, wp.M_TILE, wp.K_CHUNK, wp.N_CORES, M)
        assert packed.shape == (tiles, tile_bytes), (M, K, packed.shape, tiles, tile_bytes)
        assert packed.dtype == np.uint8
    # the qwen dims, concretely (PREDICTION.md section 1's byte arithmetic)
    assert _packed_dims(emb, q_dim, 128, 8, 1024, 8, emb) == (256, 4288)
    assert _packed_dims(2 * hidden, emb, 128, 8, 1024, 8, 2 * hidden) == (768, 4288)
    assert _packed_dims(emb, hidden, 128, 8, 1024, 8, emb) == (384, 4288)


def test_quantize_substitutes_bit_exact_dequant_and_is_idempotent():
    """After `quantize_decode_weights`: the loader fields ARE the dequantized
    copies of what the packed BOs hold (bit-exact vs an independent
    elementwise dequant), QKV is untouched, the packed attrs exist with the
    right shapes, and a second call is a no-op."""

    class _NS:
        pass

    emb, q_dim, hidden, kv = 1024, 2048, 3072, 1024
    cfg = _NS()
    cfg.emb_dim, cfg.hidden_dim, cfg.n_layers = emb, hidden, 2
    cfg.n_heads, cfg.head_dim = 16, 128

    rng = np.random.default_rng(11)
    weights = _NS()
    weights.layers = []
    for _ in range(cfg.n_layers):
        lw = _NS()
        lw.wq = (rng.standard_normal((emb, q_dim)) * 0.02).astype(bfloat16)
        lw.wk = (rng.standard_normal((emb, kv)) * 0.02).astype(bfloat16)
        lw.wv = (rng.standard_normal((emb, kv)) * 0.02).astype(bfloat16)
        lw.wo = (rng.standard_normal((q_dim, emb)) * 0.02).astype(bfloat16)
        lw.w_gate = (rng.standard_normal((emb, hidden)) * 0.02).astype(bfloat16)
        lw.w_up = (rng.standard_normal((emb, hidden)) * 0.02).astype(bfloat16)
        lw.w_down = (rng.standard_normal((hidden, emb)) * 0.02).astype(bfloat16)
        weights.layers.append(lw)
    orig = [{k: np.array(getattr(lw, k)) for k in ("wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down")}
            for lw in weights.layers]

    wp.quantize_decode_weights(weights, cfg)
    assert weights._w4_decode_applied is True
    for lw, o in zip(weights.layers, orig):
        # QKV untouched, bit for bit
        for k in ("wq", "wk", "wv"):
            assert np.array_equal(getattr(lw, k).view(np.uint16), o[k].view(np.uint16)), k
        # packed BOs at the builder's dims
        assert lw._wo_packed.shape == (256, 4288)
        assert lw._wgateup_packed.shape == (768, 4288)
        assert lw._wdown_packed.shape == (384, 4288)
        # the substituted fields ARE dequant(quant(orig)): recompute
        # independently from the original and compare bit-exact
        for k, key in (("wo", "wo"), ("w_gate", "w_gate"), ("w_up", "w_up"), ("w_down", "w_down")):
            q, s, z = wp._fake_quantize(o[key].T)
            want = wp.dequant_rows(q, s, z).T
            got = getattr(lw, k)
            assert np.array_equal(got.view(np.uint16), want.view(np.uint16)), k
            # ... and quantization actually happened (a no-op substitute would
            # silently verify against the WRONG oracle)
            assert not np.array_equal(got.view(np.uint16), o[key].view(np.uint16)), k
    # idempotent: the second call must not quantize the dequant copies again
    snap = weights.layers[0].wo.copy()
    wp.quantize_decode_weights(weights, cfg)
    assert np.array_equal(weights.layers[0].wo.view(np.uint16), snap.view(np.uint16))


def test_llama_default_ir_is_byte_identical_under_the_q_dim_parameter():
    """`[2026-08-26]` item-18 review round, non-blocking (b): the q_dim /
    k_chunk generalization of the llama int4 cascade builder must leave the
    LLAMA default IR byte-identical -- previously only an ignored evidence
    control (results/item18-h2b-20260826/controls/llama-byte-identity.txt),
    now a tracked regression. Two invariants: (1) parameter-neutrality --
    the default args and an explicit q_dim=emb_dim render the same bytes,
    which survives any legitimate stage-builder change; (2) the golden sha
    of the reviewed control's bytes, pinned the way test_plan pins its
    goldens -- a stage-builder (or MLIR printer) change moves it
    DELIBERATELY, with the commit that moves it updating the constant."""
    import hashlib

    from llama32_1b_int4.multi_launch_builder.o_gemv_ffn_int4_multi import (
        build_o_gemv_ffn_int4_module,
    )

    default = str(build_o_gemv_ffn_int4_module(emb_dim=2048, hidden_dim=8192))
    explicit = str(build_o_gemv_ffn_int4_module(emb_dim=2048, hidden_dim=8192, q_dim=2048))
    assert default == explicit, "q_dim=emb_dim must be the default's identity"
    sha = hashlib.sha256(default.encode()).hexdigest()
    assert sha == "36d7c7d6f77c8de5b5a0da5cece9413921903a965af8df6a92b626c1c889a3a1", (
        f"llama o_gemv_ffn_int4 default IR sha moved to {sha}; if a deliberate "
        "builder/printer change did this, update the golden here in the same commit")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"w4_decode_pack tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
