# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Host-only regression for the int4 group-size plumbing (no Peano, no NPU).

    python3 shared/infra/test_int4_gs.py   (from programming_examples/llms)

1. A gs=32 build after a gs=128 build in the SAME CWD must stage the gs=32
   kernel as the canonical `mv_int4_bf16.o`. `_compile_kernel` skips an
   existing .o by NAME, so an object name that does not carry the group size
   hands every later caller the first variant built there -- the stub below
   keeps exactly that skip-by-name rule.
2. `int4_gs` in `compile_and_cache`'s backend kwargs must reach
   `prepare_air_project` and must NOT reach `XRTBackend` (which rejects it).
Both fail on the implementation before this file was added.
"""

import os
import sys
import tempfile
from pathlib import Path

import shared.infra.cache as cache_mod
import shared.infra.external_kernels as ek


def _stub_compile_kernel(src_path, output_name, extra_flags=None, force=False):
    """Stand-in for the Peano compile: the object's bytes are its flags."""
    if not force and Path(output_name).exists():
        return  # the real skip-by-name rule, verbatim
    Path(output_name).write_text(" ".join(extra_flags or []))


def test_canonical_object_follows_the_last_group_size():
    real = ek._compile_kernel
    ek._compile_kernel = _stub_compile_kernel
    cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            ek.compile_mv_int4_bf16(gs=128)
            first = Path("mv_int4_bf16.o").read_text()
            ek.compile_mv_int4_bf16(gs=32)
            canonical = Path("mv_int4_bf16.o").read_text()
            assert "-DDIM_GS=128" in first, first
            assert "-DDIM_GS=32" in canonical, (
                "a gs=32 request after a gs=128 build left the gs=128 kernel "
                f"staged as the canonical object: {canonical!r}"
            )
    finally:
        os.chdir(cwd)
        ek._compile_kernel = real


def test_int4_gs_is_forwarded_to_the_preparer_and_kept_from_xrt():
    import air.backend.xrt as xrt_mod

    seen = {}

    def fake_prepare(quant="bf16", **kw):
        seen["prepare"] = {"quant": quant, **kw}

    class FakeBackend:
        def __init__(self, **kwargs):
            seen["backend"] = dict(kwargs)

        def compile(self, module, output_binary_name="air"):
            f = tempfile.NamedTemporaryFile(suffix=".elf", delete=False)
            f.write(b"stub")
            f.close()
            return xrt_mod.XRTCompileArtifact(f.name, "stub", None)

        def unload(self):
            pass

    real_prepare, real_backend = cache_mod.prepare_air_project, xrt_mod.XRTBackend
    cache_mod.prepare_air_project, xrt_mod.XRTBackend = fake_prepare, FakeBackend
    try:
        kw = {"int4_gs": 32, "output_format": "elf"}
        cache = cache_mod.KernelCache(cache_dir=tempfile.mkdtemp(), verbose=False)
        cache.compile_and_cache("x_int4", "<module>", kw)
    finally:
        cache_mod.prepare_air_project, xrt_mod.XRTBackend = real_prepare, real_backend
    assert seen["prepare"].get("int4_gs") == 32, seen
    assert "int4_gs" not in seen["backend"], seen
    assert kw["int4_gs"] == 32, "the caller's dict was mutated"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"int4 gs tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
