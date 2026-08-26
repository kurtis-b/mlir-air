# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host tests for `qwen3_0_6b/w4_decode_pack.py` (doc 56 H2b, queue items 18
and 24): the flag default, the packed-BO dims against the BUILDER's own
arithmetic (the two must agree or the ELF reads garbage), the dequant
substitution's bit-exactness (prefill and the HF oracle compute on EXACTLY what
the kernel dequants), and idempotence.

`[2026-08-26]` queue item 24 flipped the default to `w4_decode`. What that adds
here: the default is now pinned ON from ONE constant (`W4_DEFAULT`), an
unparseable flag value is a refusal rather than a silent bf16, the decode cache
is checked against the SELECTED precision before any device work, and the
standing lit is asserted to pin BOTH precisions plus the quantization bar --
because a flipped default whose gate is inherited rather than pinned is a gate
that silently changed meaning."""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
from ml_dtypes import bfloat16

_HERE = os.path.dirname(os.path.abspath(__file__))
_PE = os.path.dirname(os.path.dirname(_HERE))
for _p in (_PE, os.path.join(_PE, "llms"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qwen3_0_6b import w4_decode_pack as wp  # noqa: E402


def test_flag_defaults_on_and_reads_env_at_call_time():
    """`[2026-08-26]` queue item 24: the default is `w4_decode`, it comes from
    ONE constant, and the flag keeps its name and its sense (`1` = w4 on) so
    there are never two ways to say the same thing."""
    old = os.environ.pop(wp.W4_ENV, None)
    try:
        assert wp.W4_DEFAULT is True, "queue item 24 flipped the default ON"
        assert wp.w4_decode_selected() is wp.W4_DEFAULT, "unset must mean W4_DEFAULT"
        assert wp.w4_decode_selected() is True
        os.environ[wp.W4_ENV] = "1"
        assert wp.w4_decode_selected() is True
        os.environ[wp.W4_ENV] = "0"
        assert wp.w4_decode_selected() is False
        os.environ[wp.W4_ENV] = ""  # the `FOO= cmd` idiom == unset
        assert wp.w4_decode_selected() is wp.W4_DEFAULT
    finally:
        if old is None:
            os.environ.pop(wp.W4_ENV, None)
        else:
            os.environ[wp.W4_ENV] = old


def test_unparseable_flag_value_is_refused_not_silently_bf16():
    """A typo must not quietly pick the other precision: one string decides
    which ELF is compiled, which weights are packed and which oracle the gate
    builds, and a silent `!= "1"` would move all three with nothing in any log."""
    old = os.environ.get(wp.W4_ENV)
    try:
        for bad in ("true", "yes", "ON", "2", " 1"):
            os.environ[wp.W4_ENV] = bad
            try:
                wp.w4_decode_selected()
            except ValueError as e:
                assert wp.W4_ENV in str(e) and repr(bad) in str(e)
            else:
                raise AssertionError(f"{bad!r} was accepted")
    finally:
        if old is None:
            os.environ.pop(wp.W4_ENV, None)
        else:
            os.environ[wp.W4_ENV] = old


def _qwen_dir():
    return os.path.join(_PE, "llms", "qwen3_0_6b")


def test_decode_cache_is_checked_against_the_selected_precision():
    """The flip's most likely consequence is a cache compiled before it. That
    must be a sentence naming the fix, at the ONE place every entry point
    passes through (`prepare_runtime`), not a bare KeyError from
    `load_and_run` after a prefill already ran."""
    import importlib

    old = os.environ.get(wp.W4_ENV)
    # the driver modules import each other by bare name; scope that sys.path
    # entry to this test so `verify_adapter` & friends cannot shadow another
    # example's module for the rest of the suite.
    qdir = _qwen_dir()
    added = qdir not in sys.path
    if added:
        sys.path.insert(0, qdir)
    try:
        os.environ[wp.W4_ENV] = "1"
        dec = importlib.import_module("qwen3_0_6b_decode")
        dec = importlib.reload(dec)
        assert dec.required_decode_artifacts() == (
            "rms_qkv_qknorm_rope_gemv2", "o_gemv_ffn_int4", "lm_head_gemv")

        class _Stale:
            cache_dir = "/tmp/stale_decode_kernel_cache"
            artifacts = {"rms_qkv_qknorm_rope_gemv2": 1, "o_gemv_ffn": 1,
                         "lm_head_gemv": 1}

        try:
            dec.require_decode_artifacts(_Stale())
        except RuntimeError as e:
            msg = str(e)
            assert "o_gemv_ffn_int4" in msg and "QWEN3_W4_DECODE=0" in msg, msg
        else:
            raise AssertionError("a bf16-only cache was accepted under w4_decode")

        class _Good:
            cache_dir = "/tmp/good"
            artifacts = {n: 1 for n in dec.required_decode_artifacts()}

        dec.require_decode_artifacts(_Good())  # must not raise
        # and the driver calls it, at the one chokepoint
        with open(os.path.join(_qwen_dir(), "qwen3_0_6b_inference.py")) as f:
            src = f.read()
        head = src[src.index("def prepare_runtime("):]
        assert "require_decode_artifacts(decode_cache)" in head[: head.index("print(")]
    finally:
        if old is None:
            os.environ.pop(wp.W4_ENV, None)
        else:
            os.environ[wp.W4_ENV] = old
        importlib.reload(importlib.import_module("qwen3_0_6b_decode"))
        if added:
            sys.path.remove(qdir)


def test_a_decode_compile_carries_the_sibling_precisions_o_ffn_entry():
    """`[2026-08-26]` queue item 24, found on the device (devq 655): after the
    flip, a default `make verify`/`make compile` rewrote the shipped decode
    manifest with the w4 set only, and `QWEN3_W4_DECODE=0 make profile` then
    refused -- the bf16 ELF was still on disk, just no longer named. Since the
    flag is documented as an A/B knob, ONE build tree has to serve both, so a
    decode compile carries the SIBLING precision's O+FFN entry across.

    Narrow on purpose: exactly that one name, never the whole previous
    manifest. The same cache also holds `rms_qkv_qknorm_rope_gemv` / `_gemv4`
    ELFs from an older launch-count selection, and resurrecting those into the
    manifest is the hazard this avoids."""
    import importlib
    import json
    import tempfile

    old = os.environ.get(wp.W4_ENV)
    qdir = _qwen_dir()
    added = qdir not in sys.path
    if added:
        sys.path.insert(0, qdir)
    try:
        os.environ[wp.W4_ENV] = "1"
        dec = importlib.reload(importlib.import_module("qwen3_0_6b_decode"))

        class _Cache:
            MANIFEST_FILE = "manifest.json"

            def __init__(self, d):
                self.cache_dir = d
                self.artifacts, self.launch_counts, self.arg_counts = {}, {}, {}

        with tempfile.TemporaryDirectory() as d:
            for name in ("o_gemv_ffn", "rms_qkv_qknorm_rope_gemv4"):
                (pathlib.Path(d) / f"{name}.elf").write_bytes(b"\x7fELF")
            (pathlib.Path(d) / "manifest.json").write_text(json.dumps({
                "o_gemv_ffn": {"output_binary": os.path.join(d, "o_gemv_ffn.elf"),
                               "kernel": "MLIR_AIE", "insts": None,
                               "launches": {"air_launches": 3, "herd_launches": 5},
                               "n_args": 15},
                "rms_qkv_qknorm_rope_gemv4": {
                    "output_binary": os.path.join(d, "rms_qkv_qknorm_rope_gemv4.elf"),
                    "kernel": "MLIR_AIE", "insts": None,
                    "launches": {"air_launches": 4, "herd_launches": 4}, "n_args": 18},
            }))
            cache = _Cache(d)
            carried = dec._sibling_o_ffn_entry(cache)
            assert carried is not None and carried[0] == "o_gemv_ffn", carried
            # the compile writes its own three artifacts...
            cache.artifacts = {n: object() for n in dec.required_decode_artifacts()}
            dec._restore_sibling_o_ffn(cache, carried)
            assert set(cache.artifacts) == set(dec.required_decode_artifacts()) | {"o_gemv_ffn"}
            assert "rms_qkv_qknorm_rope_gemv4" not in cache.artifacts, (
                "only the sibling O+FFN entry is carried; a stale launch-count "
                "variant must NOT come back")
            assert cache.launch_counts["o_gemv_ffn"] == {"air_launches": 3, "herd_launches": 5}
            assert cache.arg_counts["o_gemv_ffn"] == 15
            # a manifest naming a binary that is gone is not carried
            os.remove(os.path.join(d, "o_gemv_ffn.elf"))
            assert dec._sibling_o_ffn_entry(_Cache(d)) is None

        # and the compile path reads BEFORE it writes and restores AFTER
        src = pathlib.Path(qdir, "qwen3_0_6b_decode.py").read_text()
        body = src[src.index("def compile_decode_kernels("):]
        body = body[: body.index("\ndef ", 1)]
        assert body.index("_sibling_o_ffn_entry(cache)") < body.index("compile_and_cache")
        assert body.index("compile_and_cache") < body.index("_restore_sibling_o_ffn(cache, carried)")
        assert body.index("_restore_sibling_o_ffn(cache, carried)") < body.index("_save_manifest()")
    finally:
        if old is None:
            os.environ.pop(wp.W4_ENV, None)
        else:
            os.environ[wp.W4_ENV] = old
        importlib.reload(importlib.import_module("qwen3_0_6b_decode"))
        if added:
            sys.path.remove(qdir)


def test_standing_lit_pins_both_precisions_and_the_quantization_bar():
    """`[2026-08-26]` queue item 24. The suite gate the w4 path never had.
    Three properties, each of which a plausible future edit would break:
    (1) BOTH arms pin `QWEN3_W4_DECODE` explicitly, so the test's meaning does
    not move the next time a default does; (2) the quantization bar runs --
    the token-set gate against the UNPATCHED checkpoint, which is the only arm
    that can see quantization error at all; (3) the lit does not use `make
    clean`, whose `rm -rf ../verify/reports` escapes the test's cwd (item 20's
    recorded hazard)."""
    with open(os.path.join(_qwen_dir(), "run_npu2_verify.lit")) as f:
        lit = f.read()
    runs = [ln for ln in lit.splitlines() if ln.startswith("// RUN:")]
    joined = "\n".join(runs)
    bf16 = [ln for ln in runs if "QWEN3_W4_DECODE=0" in ln and "Makefile verify " in ln]
    w4 = [ln for ln in runs if "QWEN3_W4_DECODE=1" in ln and "Makefile verify " in ln]
    qbar = [ln for ln in runs if "verify-quant-bar" in ln]
    assert len(bf16) == 1, "no pinned bf16 arm"
    assert len(w4) == 1, "no pinned w4 arm"
    assert len(qbar) == 1, "no quantization bar arm"
    assert "make -f %S/Makefile clean-build" in joined
    assert "Makefile clean\n" not in joined + "\n", "the unscoped clean is back"
    # (4) the arms SHARE the artifact set. `compile_and_cache` never skips and
    # the adapter compiles into the source-tree build_peano, so three plain
    # `make verify` runs would compile the whole set three times (~7.5 min each,
    # devq 652). Arm 2 LOADS arm 1's prefill and compiles only the decode set --
    # which is also what makes the A/B exact, since the precision lives entirely
    # in the decode cascade; arm 3 LOADS both, so the quantization bar is
    # measured on precisely the ELFs arm 2 gated.
    assert "LLMS_VERIFY_PREFILL_CACHE=%S/build_peano/prefill_kernel_cache" in w4[0], (
        "arm 2 must LOAD arm 1's prefill set: identical prefill bytes are what "
        "make this an A/B of the decode precision and nothing else")
    assert "LLMS_VERIFY_DECODE_CACHE" not in w4[0], (
        "arm 2 must COMPILE the decode set -- that is where the precision lives")
    assert "LLMS_VERIFY_PREFILL_CACHE=%S/build_peano/prefill_kernel_cache" in qbar[0]
    assert "LLMS_VERIFY_DECODE_CACHE=%S/build_peano/decode_kernel_cache" in qbar[0], (
        "the quantization bar must judge the SAME bytes arm 2 gated")
    # and the LOADED arms say so in their output, so a silent recompile is red
    assert lit.count("artifact set: prefill_M=2048 max_seq=2048 LOADED") == 2
    # the arms must be distinguishable in the OUTPUT, not just the command line
    assert "W4: HF reference patched with the w4_decode dequant O+FFN weights" in lit
    assert "QUANTBAR-NOT: HF reference patched with the w4_decode dequant" in lit

    with open(os.path.join(_qwen_dir(), "Makefile")) as f:
        mk = f.read()
    bar = mk[mk.index("verify-quant-bar:"):]
    bar = bar[: bar.index("\n\n")]
    assert "QWEN3_W4_DECODE=1" in bar and "--gate-phase capture-npu" in bar
    assert "QWEN3_W4_DECODE=0" in bar and "--gate-phase compare-hf" in bar, (
        "the compare phase must run with the flag OFF -- that is what makes "
        "build_hf_model return None and the oracle the plain checkpoint")
    assert "clean-build:" in mk


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
