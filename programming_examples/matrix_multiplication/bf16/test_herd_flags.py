# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Host regression for the mmult runners' herd flags (no device, no Peano).

    python3 test_herd_flags.py

Each runner CLI is driven end to end (parse -> forward -> place) via runpy with
--herd-m 2 --herd-n 2 on a small module built by run.py's build_module; the
placed IR written to air_ir_debug.mlir must carry x_size/y_size = 2. The
pre-existing simulator-schema failure downstream of the placement is swallowed:
the gate is the IR, which is written before that step. A 4x4 default arm pins
the default path.
"""

import os
import re
import runpy
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from ml_dtypes import bfloat16  # noqa: E402

import run as run_mod  # noqa: E402


def _input_ir():
    launch = run_mod.build_module(
        128, 128, 128, 32, 64, 32, 32, 4, 4, bfloat16, bfloat16
    )
    return str(launch.build(target="npu2"))


def _placed_sizes(script, ir_path, herd_args):
    """Run the script's CLI in a temp cwd; return the segment x/y sizes."""
    d = tempfile.mkdtemp()
    old_cwd, old_argv = os.getcwd(), sys.argv
    os.chdir(d)
    sys.argv = [script, "--input-file", ir_path] + herd_args
    try:
        runpy.run_path(os.path.join(_HERE, script), run_name="__main__")
    except SystemExit:
        pass
    except Exception:
        pass  # the simulator's arch-schema failure is pre-existing; the IR gate is below
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv
    with open(os.path.join(d, "air_ir_debug.mlir")) as f:
        placed = f.read()
    m = re.search(r"x_size = (\d+) : i64, y_loc = \d+ : i64, y_size = (\d+)", placed)
    assert m, "no placed segment attributes in air_ir_debug.mlir"
    return int(m.group(1)), int(m.group(2))


def main():
    ir = _input_ir()
    f = tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False)
    f.write(ir)
    f.close()
    n_pass = 0
    for script in ("mmult_aie2.py", "mmult_aie2p.py"):
        x, y = _placed_sizes(script, f.name, ["--herd-m", "2", "--herd-n", "2"])
        assert (x, y) == (2, 2), (
            f"{script}: --herd-m 2 --herd-n 2 placed a {x}x{y} segment -- "
            "the flags are parsed but not reaching the placer again"
        )
        xd, yd = _placed_sizes(script, f.name, [])
        assert (xd, yd) == (4, 4), f"{script}: default placement changed: {xd}x{yd}"
        print(
            f"PASS  {script}: herd flags reach the placer (2x2), default intact (4x4)"
        )
        n_pass += 1
    print(f"mmult herd-flag tests: {n_pass}/2 passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
