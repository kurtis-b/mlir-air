# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only checks on the routed-design locator, and on the CLAIM behind it.

    python3 study/test_aircc_artifacts.py [--project DIR]

Without ``--project`` these are pure unit tests over a synthetic fixture: no
NPU, no MLIR, no aircc. With ``--project`` pointing at a real ``air_project``
directory they additionally re-derive the resolution doc 09 asked for, by
running iron's own three regexes against the artifact and reporting the counts.
That second mode is how the claim gets checked against a fresh toolchain rather
than trusted from this file's docstring.

Generate a project directory to point it at with any compile, e.g.

    python3 agents/probes/probe_ffn_accum_bd_offset.py --ksteps 2
"""

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import aircc_artifacts  # noqa: E402

# iron's regexes, verbatim from resource_usage/analysis.py. Copied rather than
# imported because that module is not ported yet; when it is, import them from
# there so there is one copy.
_CORE_RE = re.compile(r"aie\.core\(\s*(%tile_\d+_\d+)\s*\)")
_DMA_ALLOCATION_RE = re.compile(
    r"aie\.(?P<kind>\w*dma_allocation)\s+@\w+\(\s*"
    r"(?P<tile>%(?:shim_noc_tile|mem_tile|tile)_\d+_\d+)\s*,\s*"
    r"(?P<direction>S2MM|MM2S)\s*,\s*(?P<channel>\d+)\s*\)"
)
_BUFFER_RE = re.compile(r"aie\.buffer\(\s*(%tile_\d+_\d+)\s*\)")


def _raises(exc, match, fn, *args):
    try:
        fn(*args)
    except exc as e:
        assert match in str(e), f"expected {match!r} in {str(e)!r}"
        return
    raise AssertionError(f"expected {exc.__name__} containing {match!r}")


def test_locator_returns_the_named_artifact():
    with tempfile.TemporaryDirectory() as d:
        target = Path(d) / aircc_artifacts.ROUTED_DESIGN_NAME
        target.write_text("// routed design\n")
        assert aircc_artifacts.routed_design(d) == target


def test_missing_artifact_says_what_its_absence_means():
    """A path that does not exist is worse than an error that explains why."""
    with tempfile.TemporaryDirectory() as d:
        _raises(
            aircc_artifacts.RoutedDesignMissing,
            "died before lowering completed",
            aircc_artifacts.routed_design,
            d,
        )


def test_missing_artifact_warns_against_the_debug_dump_fallback():
    """Doc 09: emit or name a stable artifact, never parse an incidental one."""
    with tempfile.TemporaryDirectory() as d:
        _raises(
            aircc_artifacts.RoutedDesignMissing,
            "debug_ir",
            aircc_artifacts.routed_design,
            d,
        )


def test_a_directory_named_like_the_artifact_is_not_the_artifact():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / aircc_artifacts.ROUTED_DESIGN_NAME).mkdir()
        _raises(
            aircc_artifacts.RoutedDesignMissing,
            "absent",
            aircc_artifacts.routed_design,
            d,
        )


def test_device_constants_are_the_aie2p_ones():
    """These are part properties; they carry over from iron unchanged."""
    assert aircc_artifacts.AIE_TILE_LOCAL_MEMORY_BYTES == 65536
    assert aircc_artifacts.MEM_TILE_LOCAL_MEMORY_BYTES == 524288
    assert aircc_artifacts.SHIM_DMA_CHANNELS_PER_DIRECTION == 2


def check_real_project(project_dir):
    """Re-derive doc 09's resolution against a real compile. Returns 0 or 1."""
    path = aircc_artifacts.routed_design(project_dir)
    src = path.read_text()
    cores = _CORE_RE.findall(src)
    dmas = [m.groupdict() for m in _DMA_ALLOCATION_RE.finditer(src)]
    buffers = _BUFFER_RE.findall(src)

    print(f"  artifact: {path}")
    print(f"  iron _CORE_RE            -> {len(cores)} cores")
    print(f"  iron _BUFFER_RE          -> {len(buffers)} buffers")
    print(f"  iron _DMA_ALLOCATION_RE  -> {len(dmas)} allocations")
    if dmas:
        d = dmas[0]
        print(f"      e.g. {d['kind']} {d['tile']} {d['direction']} ch{d['channel']}")

    problems = []
    if not cores:
        problems.append("no aie.core matched -- the artifact is not a routed design")
    if not buffers:
        problems.append("no aie.buffer matched")
    if not dmas:
        problems.append(
            "no dma_allocation matched -- resource usage cannot be normalized "
            "against SHIM_DMA_CHANNELS_PER_DIRECTION without them"
        )
    for p in problems:
        print(f"  FAIL {p}")
    if problems:
        return 1
    print("  PASS  iron's regexes match this artifact unmodified")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--project",
        help="a real air_project directory; re-derives doc 09's resolution",
    )
    args = ap.parse_args()

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"aircc-artifact tests: {len(tests)}/{len(tests)} passed")

    if args.project:
        print(f"\nre-deriving the resolution against {args.project}")
        return check_real_project(args.project)
    return 0


if __name__ == "__main__":
    sys.exit(main())
