# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""`herd_rows` is inert at its default -- checked, not asserted.

Ten call sites in ``programming_examples/llms/`` parse ``str(build_module(...))``
as MLIR text, so the day the default gains a herd dimension they all change.
This pins that it has not.

Three of the four cases are self-contained; the fourth compares against the
builder as it stood *before* the parameter existed, read straight out of git,
so the comparison is against real predecessor source rather than a recorded
hash. A hash would rot the first time an unrelated air.api change moved the
emitted text for everyone; the predecessor source moves with it.
"""

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matvec  # noqa: E402

# The commit whose matvec.py is the parameter-less predecessor: origin/main
# immediately before the herd_rows parameter landed.
PRE_PARAM_COMMIT = "3f2dc181"
REPO_RELATIVE = "programming_examples/matrix_vector_multiplication/bf16/matvec.py"

# One shape, used by every case. Divisible by tile_m * herd_m * herd_rows for
# herd_rows up to 4, so no case is skipped by the divisibility assert.
SHAPE = dict(m=2048, k=8192, tile_m=2, m_input=1, herd_m=4)


def _build(herd_rows=None, module=matvec):
    """Emit the module as MLIR text. `herd_rows=None` omits the argument."""
    kwargs = dict(SHAPE)
    if herd_rows is not None:
        kwargs["herd_rows"] = herd_rows
    return str(
        module.build_module(
            kwargs.pop("m"),
            kwargs.pop("k"),
            kwargs.pop("tile_m"),
            kwargs.pop("m_input"),
            kwargs.pop("herd_m"),
            bfloat16,
            bfloat16,
            target="npu2",
            **kwargs,
        )
    )


def test_omitting_herd_rows_matches_passing_one():
    omitted, explicit = _build(), _build(1)
    assert omitted == explicit, (
        "omitting herd_rows and passing 1 emit different IR; the default is "
        "no longer inert"
    )


def test_two_rows_differs_from_the_default():
    # The control. Without it, the assertions above would also pass if
    # `herd_rows` were ignored entirely.
    assert _build() != _build(2), (
        "herd_rows=2 emits the same IR as the default, so the parameter is "
        "not reaching the builder and the other cases prove nothing"
    )


def test_two_rows_differs_by_gaining_a_herd_dimension():
    # Pin WHY it differs, so the control above cannot pass for an incidental
    # reason. A second herd dimension is what puts a third symbol in the
    # offset map -- the shape air-split-l2-memref cannot yet split.
    one, two = _build(1), _build(2)
    assert "()[s0, s1, s2]" not in one, "the 1-row form already has 3 symbols"
    assert "()[s0, s1, s2]" in two, (
        "herd_rows=2 did not produce a 3-symbol offset map; the 2-D herd is "
        "not being emitted"
    )


def test_default_matches_the_pre_parameter_builder():
    src = subprocess.run(
        ["git", "show", f"{PRE_PARAM_COMMIT}:{REPO_RELATIVE}"],
        cwd=HERE,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "matvec_pre_param.py"
        path.write_text(src)
        spec = importlib.util.spec_from_file_location("matvec_pre_param", path)
        pre = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pre)

    assert "herd_rows" not in src, (
        f"{PRE_PARAM_COMMIT} already knows herd_rows; it is not the "
        "pre-parameter builder and this case compares nothing"
    )
    assert _build() == _build(module=pre), (
        "today's default emits different IR than the builder did before the "
        "parameter existed"
    )


if __name__ == "__main__":
    tests = sorted(k for k in globals() if k.startswith("test_"))
    n_pass = 0
    for name in tests:
        try:
            globals()[name]()
            n_pass += 1
            print("PASS  %s" % name)
        except Exception as e:  # report every failure shape, keep going
            print("FAIL  %s: %r" % (name, e))
    print("herd_rows inertness tests: %d/%d passed" % (n_pass, len(tests)))
    sys.exit(0 if n_pass == len(tests) else 1)
