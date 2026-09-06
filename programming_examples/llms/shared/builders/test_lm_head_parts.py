# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""`parts` gives the LM-head builder mixed partition sizes, inert when omitted.

The equal-sized path is what every caller on main uses today, so the first two
cases pin that it did not move: omitting `parts` matches the builder as it
stood before the parameter (read out of git), and spelling the same partitions
out explicitly matches omitting them. The rest exercise the actual feature and
its validation.

Host-only: builds MLIR text, no device.
"""

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import lm_head_gemv_multi as builder  # noqa: E402

# origin/main immediately before `parts` landed.
PRE_PARTS_COMMIT = "f02f2855"
REPO_RELATIVE = "programming_examples/llms/shared/builders/lm_head_gemv_multi.py"

# Small enough to build quickly, still on the tile grid (tile_m * herd_m = 64).
SHAPE = dict(emb_dim=256, n_partitions=2, n_part=128, tile_m=8, m_input=4, herd_m=8)


def _build(module=builder, **over):
    kwargs = dict(SHAPE)
    kwargs.update(over)
    return str(module.build_lm_head_gemv_module(**kwargs))


def test_omitting_parts_matches_the_pre_parameter_builder():
    src = subprocess.run(
        ["git", "show", f"{PRE_PARTS_COMMIT}:{REPO_RELATIVE}"],
        cwd=HERE,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "parts" not in src.split('"""')[2], (
        f"{PRE_PARTS_COMMIT} already knows `parts`; it is not the "
        "pre-parameter builder and this case compares nothing"
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "lm_head_pre_parts.py"
        path.write_text(src)
        spec = importlib.util.spec_from_file_location("lm_head_pre_parts", path)
        pre = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pre)
        before = _build(module=pre)
    assert _build() == before, (
        "omitting `parts` emits different IR than the builder did before the "
        "parameter existed"
    )


def test_spelling_out_equal_parts_matches_omitting_them():
    assert _build() == _build(parts=[SHAPE["n_part"]] * SHAPE["n_partitions"]), (
        "an explicit list of equal partitions differs from the "
        "n_partitions/n_part shorthand that is supposed to expand to it"
    )


def test_mixed_parts_emit_both_shapes_and_one_launch_each():
    ir = _build(parts=[128, 64])
    # The control for the two cases above: if `parts` were ignored they would
    # still pass, and this would not.
    assert (
        f"memref<128x{SHAPE['emb_dim']}xbf16>" in ir
    ), "the 128-row weight arg is missing"
    assert (
        f"memref<64x{SHAPE['emb_dim']}xbf16>" in ir
    ), "the 64-row weight arg is missing"
    assert (
        ir.count("air.launch") == 2
    ), f"expected one air.launch per partition, found {ir.count('air.launch')}"


def test_a_partition_off_the_tile_grid_is_refused():
    # 100 is not a multiple of tile_m * herd_m = 64. The message must name the
    # offending partition, which is the reason for checking here at all.
    try:
        _build(parts=[128, 100])
    except ValueError as e:
        assert "partition 1" in str(e), f"the message does not name partition 1: {e}"
        return
    raise AssertionError("a partition off the tile grid was accepted")


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
    print("lm_head parts tests: %d/%d passed" % (n_pass, len(tests)))
    sys.exit(0 if n_pass == len(tests) else 1)
