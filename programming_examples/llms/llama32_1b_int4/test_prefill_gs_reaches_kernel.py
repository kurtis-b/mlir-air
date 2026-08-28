# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Behavioural audit: the prefill driver's `--gs` must reach the KERNEL, not
only the host packer.

    python3 llama32_1b_int4/test_prefill_gs_reaches_kernel.py

WHY THIS FILE EXISTS
    `[2026-08-27]` `llama32_1b_int4_prefill.py` parsed `--gs`, passed it to
    `load_awq_weights` (the host packer), and compiled the int4 GEMM
    micro-kernel at a hardcoded `gs=128`. `--gs 64` therefore PACKED the weights
    in groups of 64 and DEQUANTIZED them on device in groups of 128. Every
    scale after the first is read from the wrong place; the output is wrong; and
    NOTHING FAILS -- both halves are individually well formed, the ELF builds,
    the run completes, and the number is simply not the model's.

WHY THE EXISTING AUDIT COULD NOT HAVE CAUGHT IT
    `verify/test_verify_runner.py` walks the AST for flags that are parsed and
    never read. `--gs` IS read -- by the packer. That file states the gap in its
    own SCOPE section: "What this cannot see is a flag that is read and then has
    no effect. That is a different defect and needs a behavioural test." This is
    that behavioural test, for that defect, in the file where it occurred.

WHY IT CAN RUN WITHOUT A DEVICE
    It never compiles anything. `_compile_kernel` and the three sibling kernel
    builders are replaced with recorders, so what is asserted is the FLAGS the
    driver would have passed to Peano -- which is exactly where the defect was.
    The module imports numpy and ml_dtypes but no XRT and no checkpoint.
"""

import os
import sys
import tempfile
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent))
sys.path.insert(0, str(_THIS_DIR))

import llama32_1b_int4_prefill as P  # noqa: E402


def _captured_gemm_names(gs, seed=(), prepare=None):
    """Like `_captured_gemm_flags` but returns every (name, flags) the driver
    asked for, and pre-seeds the cwd with `seed` filenames first."""
    return _run_prepare(gs, seed=seed, prepare=prepare)


def _captured_gemm_flags(gs, prepare=None):
    """Run the driver's air_project preparation with every compiler stubbed out
    and return the -D flags it would have handed the GEMM micro-kernel."""
    seen = _run_prepare(gs, prepare=prepare)
    gemm = [flags for out, flags in seen if "matmul" in out and flags is not None]
    assert len(gemm) == 1, (
        f"expected exactly one GEMM micro-kernel compile, saw {len(gemm)}: "
        f"{[o for o, _ in seen]}. If the driver changed shape, re-anchor this "
        "test rather than deleting it."
    )
    return gemm[0]


def _run_prepare(gs, seed=(), prepare=None):
    seen = []

    def fake_compile_kernel(src, out, extra_flags=None, force=False):
        # FAITHFUL to the real `_compile_kernel`: it skips an existing object BY
        # NAME. An always-recording fake cannot see the cache-reuse defect at
        # all, which is exactly what the review found wrong with the first
        # version of this file.
        if not force and Path(out).exists():
            seen.append((str(out), None))  # skipped, no flags
            return
        seen.append((str(out), list(extra_flags or [])))
        Path(out).write_bytes(b"")  # the driver copies this file

    # `_INT4_GS` may not EXIST on the source under audit -- that is precisely
    # the pre-fix state -- so save/restore has to tolerate a missing attribute.
    # If it required one, this test would die with AttributeError on the very
    # source it exists to indict, and would be reporting "cannot run" as if it
    # were "detected the bug".
    _MISSING = object()
    names = ("_compile_kernel", "compile_rope", "compile_silu_and_mul",
             "compile_attn_npu2", "_INT4_GS")
    saved = {n: getattr(P, n, _MISSING) for n in names}
    try:
        P._compile_kernel = fake_compile_kernel
        P.compile_rope = lambda *a, **k: None
        P.compile_silu_and_mul = lambda *a, **k: None
        P.compile_attn_npu2 = lambda *a, **k: None
        P._INT4_GS = gs
        with tempfile.TemporaryDirectory() as td:
            cwd = os.getcwd()
            os.chdir(td)
            try:
                for f in seed:
                    Path(f).write_bytes(b"stale")
                (prepare or P._prepare_air_project_int4)()
            finally:
                os.chdir(cwd)
    finally:
        for n, v in saved.items():
            if v is _MISSING:
                delattr(P, n)
            else:
                setattr(P, n, v)

    return seen


def _dim_gs(flags):
    vals = [f.split("=", 1)[1] for f in flags if f.startswith("-DDIM_GS=")]
    assert len(vals) == 1, f"expected exactly one -DDIM_GS in {flags}"
    return int(vals[0])


def test_the_recorder_sees_a_real_compile():
    """Non-vacuity, run first: if the stub captured nothing, every assertion
    below would pass by describing an empty list."""
    flags = _captured_gemm_flags(128)
    assert any(f.startswith("-DDIM_M=") for f in flags), flags
    assert any(f.startswith("-DDIM_K_CHUNK=") for f in flags), flags
    assert any(f.startswith("-DDIM_GS=") for f in flags), flags


def test_gs_reaches_the_gemm_kernel():
    """THE DEFECT. The group size the host packed with must be the group size
    the device dequantizes with. On the pre-fix source this fails at gs=64 and
    gs=256, which both report -DDIM_GS=128."""
    wrong = []
    for gs in (64, 128, 256):
        got = _dim_gs(_captured_gemm_flags(gs))
        if got != gs:
            wrong.append(f"--gs {gs} compiled the kernel at -DDIM_GS={got}")
    assert not wrong, (
        "the driver packs at one group size and dequantizes at another: "
        + "; ".join(wrong)
        + ". This is a wrong answer with no failure -- the ELF builds and the "
        "run completes. Wire --gs to the kernel compile, or make the driver "
        "refuse a group size it cannot honour."
    )


def test_the_audit_can_actually_fail():
    """Negative control, run on every invocation.

    Re-creates the pre-fix call -- the GEMM compile that ignores its caller's
    group size -- and asserts this test catches it. An audit that cannot detect
    the bug it was written for is the same class of defect as the bug."""
    def prefix_prepare():
        P._compile_mv_int4_bf16_matmul(tile_n=P._INT4_TILE_N)  # no gs= : the bug

    got = _dim_gs(_captured_gemm_flags(64, prepare=prefix_prepare))
    assert got == 128, (
        "STALE: the pre-fix call no longer produces -DDIM_GS=128, so this "
        f"control proved nothing (it produced {got}). Re-anchor it."
    )


def test_disagreeing_channels_are_refused():
    """`compile_and_cache` carries its own `int4_gs`. If a caller ever routes a
    non-default value that contradicts `--gs`, the driver must refuse rather
    than silently pick one -- which is how the original defect behaved."""
    _MISSING = object()
    saved = getattr(P, "_INT4_GS", _MISSING)
    try:
        P._INT4_GS = 64
        try:
            P._prepare_air_project_int4(int4_gs=256)
        except ValueError as exc:
            assert "256" in str(exc) and "64" in str(exc), str(exc)
        else:
            raise AssertionError(
                "routing int4_gs=256 against --gs 64 was accepted; two channels "
                "disagreed and the driver chose one silently"
            )
        # and the stock default must NOT trip it, or every compile refuses
        P._INT4_GS = 128
        assert _dim_gs(_captured_gemm_flags(128)) == 128
    finally:
        if saved is _MISSING:
            delattr(P, "_INT4_GS")
        else:
            P._INT4_GS = saved


# Order matters: the recorder's non-vacuity check first, then THE DEFECT, then
# the controls. Alphabetical order would run `test_disagreeing_channels...`
# first, and on a pre-fix source that dies with AttributeError -- reporting
# "this test cannot run" in the position where "this source is wrong" belongs.
_ORDER = (
    "test_the_recorder_sees_a_real_compile",
    "test_gs_reaches_the_gemm_kernel",
    "test_a_stale_gemm_object_is_not_reused_across_group_sizes",
    "test_the_cache_object_name_carries_every_build_affecting_flag",
    "test_a_stale_ELF_is_refused_at_a_different_group_size",
    "test_the_audit_can_actually_fail",
    "test_disagreeing_channels_are_refused",
)


def test_a_stale_gemm_object_is_not_reused_across_group_sizes():
    """CACHE LAYER 1 (review finding 4). `_compile_kernel` skips an existing
    object BY NAME. Under the pre-fix constant name `mv_int4_bf16_matmul.o`, a
    cwd left over from a gs=128 run satisfied a gs=64 request without
    rebuilding -- so `--gs 64` linked the gs=128 kernel even once the group size
    was threaded correctly. Wiring a value through is not enough when the cache
    key cannot see it."""
    seen = _captured_gemm_names(64, seed=("mv_int4_bf16_matmul.o",))
    compiled = [(n, f) for n, f in seen if "matmul" in n and f is not None]
    skipped = [n for n, f in seen if "matmul" in n and f is None]
    assert compiled, (
        "a stale `mv_int4_bf16_matmul.o` from a previous group size suppressed "
        f"the GEMM compile entirely (skipped: {skipped}); --gs 64 would have "
        "linked the gs=128 kernel"
    )
    name, flags = compiled[0]
    assert _dim_gs(flags) == 64, flags
    assert "gs64" in name, (
        f"the GEMM object name {name!r} does not carry the group size, so the "
        "next run at another gs will reuse it by name"
    )


def test_the_cache_object_name_carries_every_build_affecting_flag():
    """The invariant behind the fix, not just the one case: every `-D` the
    driver passes must appear in the object's filename, or that flag is
    invisible to a cache keyed on the name."""
    seen = _captured_gemm_names(64)
    name, flags = [(n, f) for n, f in seen if "matmul" in n and f is not None][0]
    base = Path(name).stem
    missing = []
    for f in flags:
        if not f.startswith("-D") or "=" not in f:
            continue
        val = f.split("=", 1)[1]
        if not val.isdigit():
            continue  # non-numeric defines (e.g. feature switches) are not keys
        if val not in base:
            missing.append(f)
    assert not missing, (
        f"{name!r} omits {missing}; `_compile_kernel` skips by name, so a build "
        "that differs only in those flags silently reuses this object"
    )


def test_a_stale_ELF_is_refused_at_a_different_group_size():
    """CACHE LAYER 2 (review finding 3). Even with the object cache fixed, a
    cached ELF was loaded by filename alone, so a gs=128 cache directory served
    a `--gs 64` run and neither repair was reached."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        elf = Path(td) / "o_ffn_int4.elf"
        elf.write_bytes(b"\x7fELF")

        # (a) no sidecar => legacy, provably gs=128 => must refuse at 64
        try:
            P.check_cached_elf_gs(elf, 64, int4=True)
        except SystemExit as exc:
            assert "128" in str(exc) and "64" in str(exc), str(exc)
        else:
            raise AssertionError(
                "a legacy ELF (no build record, therefore gs=128) was accepted "
                "for a --gs 64 run"
            )

        # (b) ... and must be accepted at 128
        assert P.check_cached_elf_gs(elf, 128, int4=True) == 128

        # (c) a sidecar naming 64 is accepted at 64 and refused at 128
        elf.with_suffix(".build.json").write_text('{"int4_gs": 64}')
        assert P.check_cached_elf_gs(elf, 64, int4=True) == 64
        try:
            P.check_cached_elf_gs(elf, 128, int4=True)
        except SystemExit:
            pass
        else:
            raise AssertionError("a gs=64 ELF was accepted for a --gs 128 run")

        # (d) a NON-int4 ELF is never refused -- the bf16/bfp16 stitchers do not
        #     link the int4 GEMV and must not be blocked by its group size
        assert P.check_cached_elf_gs(elf, 128, int4=False) == 64


def main():
    g = globals()
    named = {k for k in g if k.startswith("test_")}
    assert named == set(_ORDER), (
        f"_ORDER is stale: {sorted(named ^ set(_ORDER))} is not listed. A test "
        "missing from _ORDER would never run."
    )
    tests = [g[n] for n in _ORDER]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"prefill gs-reaches-kernel tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
