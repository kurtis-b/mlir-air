# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only checks on the sequence-ladder shape override.

    python3 study/test_run_ladder.py

No device and no compile: these cover ``run_mode._shape_for``, which is the one
place a ladder can go wrong quietly. The load-bearing one is
``test_override_does_not_mutate_the_spec``. ``SPECS`` rows are module-level
dicts, so overriding ``seq_len`` in place would rewrite the catalogue, and every
later rung in the same process would inherit the first rung's length -- a ladder
that reports four lengths and measured one, with four plausible latencies to
prove it. That failure is invisible in the output CSV, which is why it is
pinned here rather than left to review.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import run_mode  # noqa: E402


def _spec(seq=4096, emb=768):
    """A SPECS-shaped row. Deliberately not imported: opcheck_specs pulls in the
    builders, which need ml_dtypes and a toolchain, and these checks need
    neither."""
    return {
        "operator": "coarse",
        "shape_key": f"{seq}x{emb}_encoder_bert",
        "shape": {"seq_len": seq, "emb_dim": emb, "ffn_dim": 4 * emb},
    }


def test_no_override_returns_the_specs_own_shape():
    spec = _spec()
    shape, key = run_mode._shape_for(spec, None)
    assert shape == spec["shape"]
    assert key == spec["shape_key"], "the catalogue's own key must survive verbatim"


def test_override_sets_the_length_and_derives_a_key():
    shape, key = run_mode._shape_for(_spec(), 512)
    assert shape["seq_len"] == 512
    assert key == "512x768_encoder_bert"
    assert shape["emb_dim"] == 768, "only the length moves"
    assert shape["ffn_dim"] == 3072


def test_override_does_not_mutate_the_spec():
    """The bug this module exists for: a ladder reporting N lengths, measuring 1."""
    spec = _spec()
    run_mode._shape_for(spec, 512)
    assert spec["shape"]["seq_len"] == 4096, "the catalogue row was rewritten"
    assert spec["shape_key"] == "4096x768_encoder_bert"


def test_successive_rungs_are_independent():
    spec = _spec()
    keys = [run_mode._shape_for(spec, s)[1] for s in (512, 1024, 2048)]
    lens = [run_mode._shape_for(spec, s)[0]["seq_len"] for s in (512, 1024, 2048)]
    assert keys == [
        "512x768_encoder_bert",
        "1024x768_encoder_bert",
        "2048x768_encoder_bert",
    ]
    assert lens == [512, 1024, 2048]


def test_key_falls_back_when_the_shape_names_hidden_size():
    """Some rows carry hidden_size rather than emb_dim; the key must not say '?'."""
    spec = {
        "operator": "coarse",
        "shape_key": "4096x768_encoder_bert",
        "shape": {"seq_len": 4096, "hidden_size": 768},
    }
    _, key = run_mode._shape_for(spec, 1024)
    assert key == "1024x768_encoder_bert"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"ladder tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
