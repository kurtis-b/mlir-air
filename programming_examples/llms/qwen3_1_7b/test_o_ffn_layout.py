# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Host layout tests for the mixed-method o_ffn_qwen builder (the qwen3_1_7b
analogue of the qwen3_0_6b rewire): the O/FFN cascade now builds through the
shared `shared.builders.o_ffn_multi._build_o_ffn` seam, so each GEMM
independently resolves drain vs fused-cast per the registry. For Qwen3-1.7B
(q_dim == emb_dim == 2048, hidden 6144) the production seq (2048) resolves
all four GEMMs to fused-cast -- the prior all-fused layout, pinned here --
while below 2048 the O rows resolve drain but the Down shape
(seq x 6144 x 2048) has no short-seq rows yet, so the mixed arm pins that
boundary as future registry data. IR/compile level only: no NPU, no aircc,
no downloads.

    python3 test_o_ffn_layout.py
"""

import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_LLMS = os.path.dirname(_HERE)
_PE = os.path.dirname(_LLMS)
for _p in (_PE, _LLMS, _HERE):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from qwen3_1_7b_weights import LlamaConfig  # noqa: E402

_CFG = LlamaConfig()
_EMB, _HID = _CFG.emb_dim, _CFG.hidden_dim
_QD = _CFG.n_heads * _CFG.head_dim

# One build per seq_len (each is a full stitch + parse); tests share these.
_BUILT = {}


def _build(seq):
    if seq not in _BUILT:
        import qwen3_1_7b_prefill as qp

        module, scratch_for = qp.build_o_ffn_qwen_module(seq, _EMB, _QD, _HID)
        _BUILT[seq] = (str(module), scratch_for)
    return _BUILT[seq]


def _func_arg_types(text):
    """The @o_ffn_qwen func's arg types, in order, from the module text."""
    m = re.search(r"func\.func @o_ffn_qwen\((.*?)\)\s*\{", text, re.S)
    assert m, "module has no @o_ffn_qwen func"
    return [a.split(":", 1)[1].strip() for a in m.group(1).split(",") if a.strip()]


def _base_types(seq):
    """The 15 always-present args (attn_out..output) of the o_ffn_qwen ABI."""
    return (
        [f"memref<{seq}x{_QD}xbf16>", f"memref<{_QD}x{_EMB}xbf16>"]
        + [f"memref<{seq}x{_EMB}xbf16>"] * 3
        + [f"memref<{_EMB}xbf16>", f"memref<{seq}x{_EMB}xbf16>"]
        + [f"memref<{_EMB}x{_HID}xbf16>", f"memref<{seq}x{_HID}xbf16>"] * 2
        + [f"memref<{seq}x{_HID}xbf16>", f"memref<{_HID}x{_EMB}xbf16>"]
        + [f"memref<{seq}x{_EMB}xbf16>", f"memref<{seq * _EMB}xbf16>"]
    )


def test_seq2048_layout_is_the_prior_all_fused_one():
    """At seq=2048 all four GEMMs stay fused-cast: the exact 19-arg signature
    the pre-rewire builder emitted (byte-identity is pinned in the PR's
    evidence; this pins the ABI those bytes imply)."""
    text, scratch_for = _build(2048)
    assert scratch_for == [15, 16, 17, 18], scratch_for
    args = _func_arg_types(text)
    assert args == _base_types(2048) + [
        "memref<2048x2048xf32>",
        "memref<2048x6144xf32>",
        "memref<2048x6144xf32>",
        "memref<2048x2048xf32>",
    ], args
    # all-fused: no drain (_m32) object referenced, and 12 launches (four
    # 2-launch fused-cast GEMMs + the 4 non-GEMM stages).
    assert "_m32" not in text, "unexpected drain symbol at seq=2048"
    assert "@zero_f32_mn_m64" in text, "fused-cast _m64 symbols missing"
    assert text.count("air.launch") == 12, text.count("air.launch")


def test_short_seq_mixed_rows_are_future_data():
    """seq 512/1024: the registry HAS mixed-method rows for this model's O
    (drain) and Gate/Up (fused-cast), but none yet for Down
    (seq x 6144 x 2048), so the cascade is registry-resolvable only at the
    production seq -- builder and plan both refuse with the KeyError naming
    the missing Down shape (no silent fallback), and the current
    single-method (all-fused) production build still works. When the Down
    rows land this arm goes red: upgrade it to the real mixed-method build
    test (the qwen3_0_6b suite's shape)."""
    from kernel_registry.registry_lookup import gemm_config

    import qwen3_1_7b_prefill as qp

    for seq in (512, 1024):
        # The short rows that DO exist resolve to a genuine method mix...
        assert gemm_config(seq, _QD, _EMB)["method"] == "drain", seq  # O
        assert gemm_config(seq, _EMB, _HID)["method"] == "fused-cast", seq  # G/U
        # ...but Down's shape has no short-seq row: builder and plan raise
        # the same registry KeyError naming it (future data, not a fallback).
        for what, fn in (
            ("builder", lambda: qp.build_o_ffn_qwen_module(seq, _EMB, _QD, _HID)),
            ("plan", lambda: qp._o_ffn_scratch_plan(seq, _CFG)),
        ):
            try:
                fn()
            except KeyError as e:
                assert f"{seq}x{_HID}x{_EMB}" in str(e), (seq, what, str(e))
            else:
                raise AssertionError(
                    f"seq={seq}: {what} built -- the Down rows landed; upgrade "
                    "this arm to the real mixed-method build test"
                )
    # The single-method production build is unaffected by the refusals above.
    _, scratch_for = _build(2048)
    assert scratch_for == [15, 16, 17, 18], scratch_for


def test_the_host_scratch_plan_matches_the_builder_contract():
    """_o_ffn_scratch_plan (what preload + the block runner allocate and mark
    intermediate) must equal the builder's returned scratch_for and the
    module's actual trailing signature at every length that builds (2048;
    at 512/1024 plan and builder raise from the SAME o_ffn_gemm_layout
    lookup -- pinned by the mixed arm -- so they cannot drift there
    either)."""
    import qwen3_1_7b_prefill as qp

    for seq in (2048,):
        text, scratch_for = _build(seq)
        plan_for, shapes, inter = qp._o_ffn_scratch_plan(seq, _CFG)
        assert plan_for == scratch_for, (seq, plan_for, scratch_for)
        args = _func_arg_types(text)
        assert len(args) == 15 + len(shapes), (seq, len(args), shapes)
        assert inter == {i for i in scratch_for if i is not None}, (seq, inter)
        for idx, (rows, cols) in zip(sorted(inter), shapes):
            assert args[idx] == f"memref<{rows}x{cols}xf32>", (seq, idx, args[idx])


def test_checked_plan_binds_the_cache_to_the_registry():
    """Review of #51, P1 (mirrored): a loaded cache must match the recomputed
    plan. The unverifiable no-sidecar refusal is exercised at seq=4096 -- the
    registry resolves it (all-fused) but it is NOT the provably-safe legacy
    2048 shape; at 512/1024 the registry KeyError fires before the sidecar
    check (pinned by the mixed arm), so no short-seq ELF can exist to bind."""
    import json
    import tempfile
    from pathlib import Path

    import qwen3_1_7b_prefill as qp

    class _C:
        def __init__(self, d):
            self.cache_dir = d

    config = _CFG
    plan = qp._o_ffn_scratch_plan(2048, config)[0]
    with tempfile.TemporaryDirectory() as d:
        # round-trip: matching sidecar passes
        (Path(d) / qp._SCRATCH_SIDECAR).write_text(
            json.dumps({"seq_len": 2048, "scratch_for": list(plan)})
        )
        qp._checked_o_ffn_plan(_C(d), 2048, config)
        # mismatch: a GENUINELY different recorded plan is refused with a
        # sentence ([15, 16, 17, 18] is the correct all-fused plan here).
        (Path(d) / qp._SCRATCH_SIDECAR).write_text(
            json.dumps({"seq_len": 2048, "scratch_for": [None, None, None, 15]})
        )
        try:
            qp._checked_o_ffn_plan(_C(d), 2048, config)
        except RuntimeError as e:
            assert "argument layout does not match" in str(e)
        else:
            raise AssertionError("a mismatched cached plan was accepted")
    with tempfile.TemporaryDirectory() as d:
        # no sidecar: legacy all-fused 2048 proceeds; a non-legacy layout
        # (all-fused seq=4096 resolves in the registry) is refused.
        qp._checked_o_ffn_plan(_C(d), 2048, config)
        try:
            qp._checked_o_ffn_plan(_C(d), 4096, config)
        except RuntimeError as e:
            assert "no scratch" in str(e)
        else:
            raise AssertionError("an unverifiable non-legacy cache was accepted")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"o_ffn layout tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
