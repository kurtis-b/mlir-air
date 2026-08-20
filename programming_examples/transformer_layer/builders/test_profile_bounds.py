# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The profile's derived bounds agree with the builders, at every ladder point.

    python3 builders/test_profile_bounds.py

`study/profiles.skip_reason` carries three bounds the first ``full`` profile
found (devq 427): the FlashAttention ``parallel_seq`` floor, the attention
GEMMs' tile multiple, and the softmax's L1 row width. ``study/test_profiles``
pins each constant to its builder's SOURCE by ast, without importing air.
This module is the other altitude: WITH air, for every (mode, length) in the
nine-point ladder, the predicate must say "skip" exactly when the builder
refuses before aircc -- a config function raising, or the GEMM builder's
assertion on module construction -- and "run" exactly when every module the
mode needs at that length constructs. Builds IR only; never compiles, never
touches a device.

A bound that skipped a buildable rung would report a walk complete having
never attempted a length the mode supports; a bound that missed a refusal
leaves the rung to fail a device-hour in. Both directions are checked.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, os.path.join(os.path.dirname(_ROOT), "llms"), os.path.join(_ROOT, "study"), _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import cases  # noqa: E402
import profiles  # noqa: E402

EMB, FFN, HEADS, HD = 768, 3072, 12, 64


def _refuses(fn):
    """Did ``fn`` refuse before aircc (ValueError or AssertionError)?"""
    try:
        fn()
    except (ValueError, AssertionError):
        return True
    return False


def _coarse_refuses(seq):
    from builders.block import block_config

    return _refuses(lambda: block_config(seq, EMB, FFN, HEADS, HD))


def _offload_refuses(seq):
    from pattern.offload.offload import _build_offload_module, offload_config

    def build_all():
        cfg = offload_config(seq, EMB, FFN, HEADS, HD)
        for key in cfg["specs"]:
            _build_offload_module(cfg, key)

    return _refuses(build_all)


def _runlist_refuses(seq):
    from pattern.runlist.runlist import _build_runlist_module, runlist_config

    def build_all():
        cfg = runlist_config(seq, EMB, FFN, HEADS, HD)
        for key in cfg["artifacts"]:
            _build_runlist_module(cfg, key)

    return _refuses(build_all)


def _fused_refuses(seq):
    # fused's packing bound refuses in the tail module's builder (the plane
    # stride against the shim dma_bd cap), not in its config, so the tail is
    # the module to construct.
    from pattern.fused.fused import build_fused_tail_module, fused_config

    return _refuses(lambda: build_fused_tail_module(fused_config(seq, EMB, FFN, HEADS, HD)))


REFUSES = {
    "coarse": _coarse_refuses,
    "offload": _offload_refuses,
    "runlist": _runlist_refuses,
    "fused": _fused_refuses,
}


def test_every_mode_at_every_ladder_point_skips_iff_its_builder_refuses():
    disagreements = []
    for mode, refuses in REFUSES.items():
        for seq in cases.SEQUENCE_LADDER:
            predicted = profiles.skip_reason(mode, seq, "baseline_768") is not None
            actual = refuses(seq)
            if predicted != actual:
                disagreements.append((mode, seq, "skip" if predicted else "run", "refuses" if actual else "builds"))
    assert not disagreements, disagreements


def test_the_agreement_is_not_vacuous_in_either_direction():
    """At least one skip and one run per bounded mode, so the test above
    could fail both ways."""
    for mode in ("coarse", "offload", "runlist"):
        verdicts = {profiles.skip_reason(mode, s, "baseline_768") is not None for s in cases.SEQUENCE_LADDER}
        assert verdicts == {True, False}, (mode, verdicts)


def test_the_refusal_messages_are_the_ones_the_bounds_cite():
    """The bound names a mechanism; the builder's message must be that one."""
    import re

    def message(fn):
        try:
            fn()
        except (ValueError, AssertionError) as e:
            return str(e)
        raise AssertionError("expected a refusal")

    from builders.block import block_config
    from pattern.runlist.runlist import runlist_config

    assert "parallel_seq" in message(lambda: block_config(64, EMB, FFN, HEADS, HD))
    assert "even rows_per_call=1 needs" in message(lambda: runlist_config(16384, EMB, FFN, HEADS, HD))
    # The GEMM builder's assertion carries (dim, tile, herd); at 256 it is the
    # n check on attn_scores: 256 against tile_n 128 x herd_n 4.
    from pattern.offload.offload import _build_offload_module, offload_config

    cfg = offload_config(256, EMB, FFN, HEADS, HD)
    assert re.fullmatch(r"\(256, 128, 4\)", message(lambda: _build_offload_module(cfg, "attn_scores")))


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"profile bound tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
