# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Host layout tests for the mixed-method o_ffn_qwen builder (review of #50,
P0): the qwen prefill O/FFN cascade now builds through the shared
`shared.builders.o_ffn_multi._build_o_ffn` seam, so each GEMM independently
resolves drain vs fused-cast per the registry -- the decoupled O
(seq x 2048 x 1024) is drain at seq 512/1024 while Down stays fused-cast,
where the old all-fused-cast copy refused to build. IR/compile level only:
no NPU, no aircc, no downloads.

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

from qwen3_0_6b_weights import LlamaConfig  # noqa: E402

_CFG = LlamaConfig()
_EMB, _HID = _CFG.emb_dim, _CFG.hidden_dim
_QD = _CFG.n_heads * _CFG.head_dim

# One build per seq_len (each is a full stitch + parse); tests share these.
_BUILT = {}


def _build(seq):
    if seq not in _BUILT:
        import qwen3_0_6b_prefill as qp

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
        "memref<2048x1024xf32>",
        "memref<2048x3072xf32>",
        "memref<2048x3072xf32>",
        "memref<2048x1024xf32>",
    ], args
    # all-fused: no drain (_m32) object referenced, and 12 launches (four
    # 2-launch fused-cast GEMMs + the 4 non-GEMM stages).
    assert "_m32" not in text, "unexpected drain symbol at seq=2048"
    assert "@zero_f32_mn_m64" in text, "fused-cast _m64 symbols missing"
    assert text.count("air.launch") == 12, text.count("air.launch")


def test_short_seq_mixed_methods_build():
    """seq 512/1024: O/Gate/Up resolve drain, Down fused-cast -- the module
    BUILDS (the old copy raised at its all-fused assert) and co-links _m32
    drain GEMMs with the _m64 fused-cast Down in one func."""
    for seq in (512, 1024):
        text, scratch_for = _build(seq)
        assert scratch_for == [None, None, None, 15], (seq, scratch_for)
        args = _func_arg_types(text)
        # 15 base args + exactly ONE f32 scratch (Down's); the old hardcoded
        # four-scratch host layout would overrun this signature by three.
        assert args == _base_types(seq) + [f"memref<{seq}x{_EMB}xf32>"], (seq, args)
        assert "@op_has_no_registered_library_name_m32" in text, seq  # drain GEMMs
        assert "@zero_f32_mn_m64" in text, seq  # fused-cast Down
        # 9 launches: O/Gate/Up drain (1 each) + Down fused-cast (2) + the 4
        # non-GEMM stages. All-fused would be 12 (the seq=2048 shape).
        assert text.count("air.launch") == 9, (seq, text.count("air.launch"))


def test_the_host_scratch_plan_matches_the_builder_contract():
    """_o_ffn_scratch_plan (what preload + the block runner allocate and mark
    intermediate) must equal the builder's returned scratch_for and the
    module's actual trailing signature at every supported length."""
    import qwen3_0_6b_prefill as qp

    for seq in (512, 1024, 2048):
        text, scratch_for = _build(seq)
        plan_for, shapes, inter = qp._o_ffn_scratch_plan(seq, _CFG)
        assert plan_for == scratch_for, (seq, plan_for, scratch_for)
        args = _func_arg_types(text)
        assert len(args) == 15 + len(shapes), (seq, len(args), shapes)
        assert inter == {i for i in scratch_for if i is not None}, (seq, inter)
        for idx, (rows, cols) in zip(sorted(inter), shapes):
            assert args[idx] == f"memref<{rows}x{cols}xf32>", (seq, idx, args[idx])


def test_checked_plan_binds_the_cache_to_the_registry():
    """Review of #51, P1: a loaded cache must match the recomputed plan."""
    import json
    import tempfile
    from pathlib import Path

    import qwen3_0_6b_prefill as qp

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
        # mismatch: a stale plan is refused with a sentence
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
        # no sidecar: legacy all-fused 2048 proceeds; short seq is refused
        qp._checked_o_ffn_plan(_C(d), 2048, config)
        try:
            qp._checked_o_ffn_plan(_C(d), 512, config)
        except RuntimeError as e:
            assert "no scratch" in str(e)
        else:
            raise AssertionError("an unverifiable short-seq cache was accepted")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"o_ffn layout tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
