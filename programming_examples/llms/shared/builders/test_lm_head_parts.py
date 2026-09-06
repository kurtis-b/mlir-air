# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""`parts` gives the LM-head builder mixed partition sizes, inert when omitted.

The equal-sized path is what every caller on main uses today, so the first two
cases pin that it did not move: omitting `parts` matches the builder as it
stood before the parameter (read out of git), and spelling the same partitions
out explicitly matches omitting them. The rest exercise the actual feature and
its validation.

The four cases above are host-only (MLIR text, no device) and are what the lit
runner collects. `--device` adds a fifth thing they cannot answer: whether a
module whose launches have DIFFERENT shapes actually LOWERS. That is the real
risk in this change -- emitting the right text is not the same as compiling,
which is exactly how the herd_rows path failed -- so it compiles two arms
through the LM head's own backend preset:

    equal  parts=[256, 256]   control: the shape main ships today
    mixed  parts=[256, 128]   the new capability

The control is load-bearing. Two earlier attempts failed BOTH arms and so said
nothing about the change: devq 929 because the herds carry `link_with="mv.o"`
and the working directory had none, and devq 931 because the xclbin path
cannot express a multi-launch module at all (`aiecc: edge 'air.insts.bin'
produced duplicate output path`) -- the drivers use `output_format="elf"` via
`LM_GEMV_BACKEND`, which is why this reuses that preset rather than
hand-rolled kwargs. **devq 932: both arms COMPILED OK**, exit 0, producing a
204,984-byte `air.elf` whose `air.mlir` carries both `memref<128x2048xbf16>`
and `memref<256x2048xbf16>` across 2 `air.launch` ops. That the same script
reported FAIL at 931 and PASS at 932 on a one-line change is the evidence
that it can go red. **devq 933 is the same run from THIS file** (same builder
md5 348a7f3b), so what is recorded here is what the committed code does.

    source agents/.state/tlenv.sh
    python3 test_lm_head_parts.py --device
"""

import hashlib
import importlib.util
import os
import subprocess
import sys
import tempfile
import traceback
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


def device_gate():
    """Compile the equal and mixed arms through aircc. Needs the toolchain."""
    root = Path(__file__).resolve().parents[4]
    matvec = root / "programming_examples/matrix_vector_multiplication/bf16"
    tile_m = 8

    src = Path(__file__).with_name("lm_head_gemv_multi.py")
    print("builder md5: %s" % hashlib.md5(src.read_bytes()).hexdigest())

    # The herds carry link_with="mv.o", so aiecc needs that object in the
    # working directory. Build it exactly as the example's own Makefile does.
    subprocess.run(
        ["make", "-C", str(matvec), "compile-kernel", "TILE_M=%d" % tile_m],
        check=True,
    )
    workdir = matvec / "build_peano"
    assert (workdir / "mv.o").exists(), "mv.o was not produced"
    os.chdir(workdir)

    sys.path.insert(0, str(root / "programming_examples/llms"))
    from shared.infra.backend_presets import LM_GEMV_BACKEND
    from air.backend.xrt import XRTBackend

    shape = dict(emb_dim=2048, tile_m=tile_m, m_input=4, herd_m=8)
    results = {}
    for name, parts in [("equal", [256, 256]), ("mixed", [256, 128])]:
        print("\n=== arm %s: parts=%s ===" % (name, parts), flush=True)
        try:
            module = builder.build_lm_head_gemv_module(parts=parts, **shape)
            backend = XRTBackend(verbose=False, **LM_GEMV_BACKEND)
            backend.compile(module)
            backend.unload()
            results[name] = True
            print("  %s: COMPILED OK" % name, flush=True)
        except Exception:
            results[name] = False
            print("  %s: FAILED" % name, flush=True)
            traceback.print_exc()

    print("\nresults: %s" % results, flush=True)
    if not results.get("equal"):
        # Without the control the mixed arm's outcome is uninterpretable.
        print("CONTROL FAILED -- this run says nothing about the change")
        return 2
    print("lm_head parts device gate: %s" % ("PASS" if results["mixed"] else "FAIL"))
    return 0 if results["mixed"] else 1


if __name__ == "__main__":
    if "--device" in sys.argv:
        sys.exit(device_gate())

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
