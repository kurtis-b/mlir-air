# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Host tests for the W4 decode driver seams (review of #33, P2): the flag
selects the artifact set, the sibling-precision manifest carry is narrow AND
toolchain-gated, the qwen int4 FFN carries the model's 1e-6 RMS eps, and the
llama int4 builder's new parameters leave its default IR byte-identical.
Compact variants of the study's driver tests (pre-port-20260829,
transformer_layer/study/test_w4_decode_pack.py), against main's manifest
schema and the production KernelCache."""

import importlib
import json
import os
import sys
import tempfile
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_LLMS = os.path.dirname(_HERE)
for _p in (_HERE, _LLMS):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _decode_at(flag):
    """qwen3_0_6b_decode reloaded with QWEN3_W4_DECODE set to `flag` (None =
    unset), environment restored; the module import IS the production read."""
    old = os.environ.pop("QWEN3_W4_DECODE", None)
    try:
        if flag is not None:
            os.environ["QWEN3_W4_DECODE"] = flag
        import qwen3_0_6b_decode as d

        return importlib.reload(d)
    finally:
        os.environ.pop("QWEN3_W4_DECODE", None)
        if old is not None:
            os.environ["QWEN3_W4_DECODE"] = old


def test_selected_artifact_set_follows_the_flag():
    bf16 = ("rms_qkv_qknorm_rope_gemv2", "o_gemv_ffn", "lm_head_gemv")
    int4 = ("rms_qkv_qknorm_rope_gemv2", "o_gemv_ffn_int4", "lm_head_gemv")
    assert _decode_at(None).required_decode_artifacts() == bf16  # default OFF
    assert _decode_at("0").required_decode_artifacts() == bf16
    assert _decode_at("1").required_decode_artifacts() == int4


def test_sibling_carry_is_narrow_and_toolchain_gated():
    from shared.infra.cache import KernelCache

    d = _decode_at("1")  # sibling of w4 is the bf16 o_gemv_ffn
    try:

        def cache_with(entries_names, toolchain):
            tmp = Path(tempfile.mkdtemp(prefix="w4_carry_"))
            cache = KernelCache(cache_dir=tmp)
            entries = {}
            for n in entries_names:
                b = tmp / f"{n}.elf"
                b.write_bytes(b"stub")
                entries[n] = {"output_binary": str(b), "kernel": "MLIR_AIE"}
            man = {"_toolchain": toolchain, "entries": entries}
            (tmp / KernelCache.MANIFEST_FILE).write_text(json.dumps(man))
            return cache

        names = ("o_gemv_ffn", "rms_qkv_qknorm_rope_gemv4")
        good = cache_with(names, KernelCache(cache_dir=".")._toolchain_id())
        carried = d._sibling_o_ffn_entry(good)
        assert carried is not None and carried[0] == "o_gemv_ffn", carried
        # narrow: the stale launch-count variant is never the carried name
        assert "gemv4" not in carried[0]
        # toolchain-gated (review P1): a stale-toolchain manifest carries nothing
        stale = cache_with(names, "peano-0.0.0-stale")
        assert d._sibling_o_ffn_entry(stale) is None, "stale toolchain carried"
        # and a missing binary carries nothing
        gone = cache_with(("o_gemv_ffn",), KernelCache(cache_dir=".")._toolchain_id())
        os.remove(Path(gone.cache_dir) / "o_gemv_ffn.elf")
        assert d._sibling_o_ffn_entry(gone) is None
    finally:
        _decode_at(None)


def test_qwen_int4_ffn_carries_the_model_eps():
    """The reused llama builder hard-coded llama's 1e-5 RMS eps (review P0);
    the qwen delegate must thread the model contract's 1e-6 into the IR."""
    from llama32_1b_int4.multi_launch_builder.o_gemv_ffn_int4_multi import (
        build_o_gemv_ffn_int4_module,
    )

    d = _decode_at("1")
    try:
        qwen_ir = str(d.build_o_gemv_ffn_int4_qwen_module(1024, 2048, 3072))
        kw = dict(
            emb_dim=1024,
            hidden_dim=3072,
            gs=128,
            m_tile=8,
            k_chunk=1024,
            n_cores=8,
            q_dim=2048,
        )
        # printer-agnostic A/B: eps is the only free variable between these
        assert qwen_ir == str(
            build_o_gemv_ffn_int4_module(eps=1e-6, **kw)
        ), "the qwen delegate is not the builder at eps=1e-6"
        assert qwen_ir != str(
            build_o_gemv_ffn_int4_module(**kw)
        ), "eps never reached the IR -- qwen module equals the 1e-5 default"
    finally:
        _decode_at(None)


def test_llama_default_ir_is_neutral_under_new_params():
    """q_dim (R3b) and eps (this fix) must leave the LLAMA default IR
    byte-identical: default args == explicit q_dim=emb_dim, eps=1e-5."""
    from llama32_1b_int4.multi_launch_builder.o_gemv_ffn_int4_multi import (
        build_o_gemv_ffn_int4_module,
    )

    default = str(build_o_gemv_ffn_int4_module(emb_dim=2048, hidden_dim=8192))
    explicit = str(
        build_o_gemv_ffn_int4_module(
            emb_dim=2048, hidden_dim=8192, q_dim=2048, eps=1.0e-5
        )
    )
    assert default == explicit, "new kwargs moved the llama default IR"
    moved = str(build_o_gemv_ffn_int4_module(emb_dim=2048, hidden_dim=8192, eps=1e-6))
    assert default != moved, "eps kwarg is dead -- it never reaches the IR"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"w4_decode_driver tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
