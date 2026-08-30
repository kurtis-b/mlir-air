# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Host regression for the mmult runners' herd flags (no device, no Peano).

    python3 test_herd_flags.py

Each runner CLI is driven end to end (parse -> forward -> place) as a
subprocess on a small module printed by run.py -p; the placed IR the runner
writes to air_ir_debug.mlir must carry x_size/y_size matching --herd-m 2
--herd-n 2, and the default arm must stay the runner's own default (4x4 / 8x4). Exit codes are ignored: the
runner fails AFTER the placement (the pre-existing simulator arch-schema
failure), and the IR on disk is the gate.
"""

import os
import re
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIZES = re.compile(r"x_size = (\d+) : i64, y_loc = \d+ : i64, y_size = (\d+)")


def _input_ir(arch):
    cmd = [sys.executable, os.path.join(_HERE, "run.py"), "-p", "--arch", arch]
    cmd += ["--m", "512", "--k", "512", "--n", "512", "--herd-m", "2", "--herd-n", "2"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    assert "air.segment" in out, f"run.py -p ({arch}) printed no AIR module"
    return out


def _placed_sizes(script, ir_path, herd_args):
    """Run the runner CLI in a temp cwd; return the placed segment x/y sizes."""
    d = tempfile.mkdtemp()
    cmd = [sys.executable, os.path.join(_HERE, script), "--input-file", ir_path]
    subprocess.run(cmd + herd_args, cwd=d, capture_output=True, timeout=600)
    debug = os.path.join(d, "air_ir_debug.mlir")
    assert os.path.exists(debug), f"{script}: no air_ir_debug.mlir (died pre-placement)"
    with open(debug) as f:
        m = _SIZES.search(f.read())
    assert m, f"{script}: no placed segment attributes in air_ir_debug.mlir"
    return int(m.group(1)), int(m.group(2))


def main():
    n_pass = 0
    for script, arch, dflt in (
        ("mmult_aie2.py", "aie2", (4, 4)),
        ("mmult_aie2p.py", "aie2p", (8, 4)),
    ):
        f = tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False)
        f.write(_input_ir(arch))
        f.close()
        x, y = _placed_sizes(script, f.name, ["--herd-m", "2", "--herd-n", "2"])
        assert (x, y) == (2, 2), (
            f"{script}: --herd-m 2 --herd-n 2 placed a {x}x{y} segment -- "
            "the flags are parsed but no longer reach the placer"
        )
        xd, yd = _placed_sizes(script, f.name, [])
        assert (xd, yd) == dflt, f"{script}: default placement changed: {xd}x{yd}"
        print(
            f"PASS  {script}: herd flags reach the placer (2x2), default intact ({xd}x{yd})"
        )
        n_pass += 1
    print(f"mmult herd-flag tests: {n_pass}/2 passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
