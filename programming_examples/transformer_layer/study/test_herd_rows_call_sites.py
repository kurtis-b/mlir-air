# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host tests for WHERE `herd_rows` is enabled (queue item 28).

Item 27 (commit `2e14f533`) added `herd_rows` to the bf16 GEMV builder and
measured the curve -- 35.72 / 44.43 / 50.03 GB/s over 8 / 16 / 32 cores -- and
also measured what rows COST: about +6.5 us of fixed time per added core, so
rows pay only above a byte threshold (9.06 MB of bf16 weights for 2 rows,
20.82 MB for 4). Item 28 turns it on at the one family of call sites above that
threshold, the LM-head partitions, and leaves every other call site alone.

Two things can silently undo that, and this module pins both:

  * a call site quietly acquiring `herd_rows` when its shape does not pay for
    it -- a regression that is invisible in `make verify` and shows up only as
    slower tokens; and
  * the LM head's partitioning changing under the parameter, so that a
    partition no longer divides by `tile_m * herd_m * herd_rows` (the 4480-row
    tail divides by 128 but NOT by 256, which is exactly why the row count is
    per-partition rather than one number for the head).

The lock-race coupling that a multi-row herd REQUIRES is pinned separately, in
`llms/shared/infra/test_dispatch.py` (run by `run_seam_tests.lit`), because it
lives in the compile chokepoint rather than at a call site.

No toolchain: the builders import `air` and this suite must not, so the call
sites are read with `ast` -- the same rule and the same technique as
`test_profiles._module_constant`.
"""

from __future__ import annotations

import ast
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PE = os.path.dirname(os.path.dirname(_HERE))
_LLMS = os.path.join(_PE, "llms")
for _p in (_PE, _LLMS, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Item 27 RESULTS.md section 5.2, `crossover.py`: bytes above which more rows
# beat one row, for the bf16 GEMV family.
BF16_CROSSOVER_MB = {2: 9.06, 4: 20.82}

#: The ONE enabled call site, and the environment variable that A/Bs it.
ENABLED = {"qwen3_0_6b/qwen3_0_6b_decode.py": ("QWEN3_LM_HERD_ROWS", 4)}


def _parse(rel):
    path = os.path.join(_LLMS, rel)
    with open(path, encoding="utf-8") as fh:
        return ast.parse(fh.read(), filename=path)


def _calls(tree, func_name):
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if name == func_name:
                out.append(node)
    return out


def _kwarg(call, name):
    for kw in call.keywords:
        if kw.arg == name:
            return kw
    return None


def _llms_sources():
    for root, dirs, files in os.walk(_LLMS):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", "build_peano", "build_chess")]
        for f in files:
            if f.endswith(".py"):
                yield os.path.relpath(os.path.join(root, f), _LLMS)


def test_only_the_priced_call_site_passes_herd_rows_to_the_lm_head_builder():
    """Every other caller of `build_lm_head_gemv_module` must not pass it.

    Their partitions are the same 16384 x emb shape, so they would probably
    win too -- but "probably" is not a measurement, and each one needs its own
    `make verify` before its bytes change. Enabling one by accident is the
    failure this catches.
    """
    passers = []
    for rel in _llms_sources():
        tree = _parse(rel)
        for call in _calls(tree, "build_lm_head_gemv_module"):
            if _kwarg(call, "herd_rows") is not None:
                passers.append(rel)
    assert sorted(set(passers)) == sorted(ENABLED), (
        f"herd_rows reaches the LM-head builder from {sorted(set(passers))}; "
        f"item 28 enabled exactly {sorted(ENABLED)}. A new call site needs its "
        "own byte-threshold arithmetic and its own make verify."
    )


def test_the_qkv_gemvs_are_left_at_one_row():
    """Item 27 section 5.2 prices them on the LOSING side: a 1-4 MB GEMV is
    below the 9.06 MB two-row crossover, so rows would cost it fixed time and
    buy it nothing. The three QKV stitchers must keep the default."""
    for rel in (
        "shared/builders/rms_gemv_rope_multi.py",
        "shared/builders/rms_qkv_bias_rope_multi.py",
        "shared/builders/rms_qkv_qknorm_rope_multi.py",
        "shared/infra/lm_head_reexec.py",
    ):
        tree = _parse(rel)
        for call in _calls(tree, "build_gemv"):
            assert _kwarg(call, "herd_rows") is None, (
                f"{rel} passes herd_rows to a GEMV whose shape item 27 prices "
                "below the crossover"
            )


def test_the_enabled_call_site_states_its_row_count_in_one_place():
    """One constant, read from one env, feeding BOTH the builder and the
    backend. Two numbers would be two ways to say it, and the second one is
    the one that hangs the device."""
    rel = "qwen3_0_6b/qwen3_0_6b_decode.py"
    tree = _parse(rel)
    env, want = ENABLED[rel]

    default = None
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        tgt = node.targets[0]
        if not (isinstance(tgt, ast.Name) and tgt.id == "_LM_HERD_ROWS"):
            continue
        call = node.value
        assert isinstance(call, ast.Call) and getattr(call.func, "id", "") == "int"
        get = call.args[0]
        assert get.args[0].value == env, get.args[0].value
        default = int(get.args[1].value)
    assert default == want, f"_LM_HERD_ROWS default is {default}, expected {want}"

    # The builder call and the backend both go through the same helper.
    builder = _calls(tree, "build_lm_head_gemv_module")
    assert len(builder) == 1
    rows_kw = _kwarg(builder[0], "herd_rows")
    assert rows_kw is not None and getattr(rows_kw.value.func, "id", "") == "lm_head_herd_rows"
    # `[2026-08-27]` the backend derives NOTHING any more. `matvec.py` marks its
    # own herd at herd_rows > 1 and the compile chokepoint supplies the lock-race
    # fix for that mark alone; a row-count-driven helper in the presets was a
    # SECOND injection trigger and is deleted, because over-broad injection is
    # what faulted the device (devq 812/813).
    assert not _calls(tree, "with_herd_rows"), (
        "with_herd_rows is deleted; the builder-emitted mark is the only trigger"
    )
    src = open(os.path.join(_LLMS, rel), encoding="utf-8").read()
    code = [ln for ln in src.splitlines() if not ln.lstrip().startswith("#")]
    assert not [ln for ln in code if "use_lock_race_condition_fix" in ln], (
        "this driver sets the lock-race flag by hand; the chokepoint owns it"
    )


def test_the_partitioning_is_derived_from_the_row_count():
    """The BD repeat cap on the activation broadcast is
    `M / (herd_m * m_input * herd_rows) - 1 <= 255`, so a partition may carry
    16384 / 32768 / 65536 rows at 1 / 2 / 4 core rows and the head needs
    10 / 5 / 3 launches. Item 28's measurement (devq 691) says that is where
    the win is -- 7.663 / 6.844 / 6.470 ms for the same 311.16 MB -- and NOT in
    the bandwidth (devq 688: rows alone move the shipped head by -0.4 %).

    So the partitioning is derived rather than written down. This test is the
    derivation, independently: if the driver's rule and this one ever disagree,
    one of them is wrong about which partitions the device will accept.
    """
    tile_m, herd_m, m_input, cap, vocab = 8, 8, 8, 255, 151936
    want = {
        1: (16384,) * 9 + (4480,),
        2: (32768,) * 4 + (20864,),
        4: (65536,) * 2 + (20864,),
    }
    for rows, expect in want.items():
        limit = (cap + 1) * herd_m * m_input * rows
        grid = tile_m * herd_m
        full, rem = divmod(vocab, limit)
        parts = tuple([limit] * full + ([-(-rem // grid) * grid] if rem else []))
        assert parts == expect, (rows, parts, expect)
        assert sum(parts) >= vocab
        for p in parts:
            r = rows
            while r > 1 and p % (tile_m * herd_m * r):
                r //= 2
            # every partition is legal at the row count it ends up with ...
            assert p % (tile_m * herd_m * r) == 0
            # ... and its broadcast repeat is inside the hardware range.
            assert p // (herd_m * m_input * r) - 1 <= cap, (p, r)
    # Zero pad rows at 2 and 4 rows; 64 at one row.
    assert sum(want[2]) == vocab and sum(want[4]) == vocab
    assert sum(want[1]) == vocab
    # The tail is why the row count is per-partition, not one number.
    assert 20864 % (tile_m * herd_m * 2) == 0
    assert 20864 % (tile_m * herd_m * 4) != 0


def test_the_one_row_arm_reproduces_the_pre_item_28_head():
    """`QWEN3_LM_HERD_ROWS=1` is the A/B arm AND the byte-identity control: it
    must give back exactly the partitioning the driver shipped before this
    item, or the A/B is between two different kernels."""
    tile_m, herd_m, m_input, cap, vocab = 8, 8, 8, 255, 151936
    limit = (cap + 1) * herd_m * m_input * 1
    grid = tile_m * herd_m
    full, rem = divmod(vocab, limit)
    parts = tuple([limit] * full + [-(-rem // grid) * grid])
    assert parts == tuple([16384] * 9 + [4480]), parts
    assert len(parts) == 10 and sum(parts) == vocab


def test_the_enabled_shapes_are_above_the_measured_byte_threshold():
    """Rows are not a blanket win. Pin the arithmetic that says these shapes
    pay and the QKV ones do not, so a partition or emb change that moves a
    shape across the threshold fails here instead of silently costing time."""
    mb = lambda rows, emb: rows * emb * 2 / 1e6  # noqa: E731
    assert mb(16384, 1024) == 33.554432
    assert mb(16384, 1024) > BF16_CROSSOVER_MB[2]
    assert mb(16384, 2048) > BF16_CROSSOVER_MB[4]
    # The 4480 tail is within 2 % of the two-row crossover -- a wash, taken for
    # the simplicity of one row count over the head rather than for its time.
    assert 0.98 < mb(4480, 1024) / BF16_CROSSOVER_MB[2] < 1.05
    # The QKV GEMVs, on the losing side at every shipped shape.
    for rows, emb in ((2048, 1024), (512, 1024), (2048, 2048), (256, 1024)):
        assert mb(rows, emb) < BF16_CROSSOVER_MB[2], (rows, emb)


def test_no_preset_or_driver_carries_the_lock_race_flag():
    """`[2026-08-27]` One trigger, in one place. The presets used to expose
    `with_herd_rows`, which injected the flag from a row count -- a second way in
    beside the builder's mark, and over-broad injection is what faulted the
    device on the study's QKV split-cast form (devq 809/812/813)."""
    import importlib

    bp = importlib.import_module("shared.infra.backend_presets")
    assert not hasattr(bp, "with_herd_rows")
    for name in dir(bp):
        value = getattr(bp, name)
        if isinstance(value, dict):
            assert "use_lock_race_condition_fix" not in value, name


def test_the_two_caps_pull_opposite_ways_and_the_tail_can_fall_between_them():
    """`[2026-08-27, restored]` Round 6 deleted this test with no justification
    in its diff and no replacement anywhere, in a round about the lock-race flag
    -- while the invariant it pins stayed live: the assert is still in
    `qwen3_0_6b_decode.py`'s `_lm_partitions`, and the README still says "the
    driver now asserts it". Restored verbatim; the original docstring follows.

    Halving a tail's `herd_rows` for divisibility DOUBLES its broadcast
    repeat, so `M % (tile_m*herd_m*r) == 0` and `repeat <= 255` can be
    unsatisfiable together. It does not bite at Qwen3-0.6B's m_input 8 (the
    20864 tail reads 162) and it DOES at m_input 4 (325), which is why the
    driver asserts rather than commenting. The fix, when it fires, is to round
    the tail to `tile_m*herd_m*want` instead of to the 64-row tile grid.
    """
    tile_m, herd_m, cap, vocab = 8, 8, 255, 151936

    def tail(m_input, want):
        limit = (cap + 1) * herd_m * m_input * want
        rem = vocab % limit
        p = -(-rem // (tile_m * herd_m)) * (tile_m * herd_m)
        r = want
        while r > 1 and p % (tile_m * herd_m * r):
            r //= 2
        return p, r, p // (herd_m * m_input * r) - 1

    # the shipped call site: legal
    assert tail(8, 4) == (20864, 2, 162)
    assert tail(8, 2) == (20864, 2, 162)
    # the same vocab at m_input 4: the tail falls between the two caps
    p, r, repeat = tail(4, 4)
    assert (p, r) == (20864, 2) and repeat == 325 > cap, (p, r, repeat)
    # and rounding it to the want-row grid fixes it, for 128 pad rows
    padded = -(-p // (tile_m * herd_m * 4)) * (tile_m * herd_m * 4)
    assert padded == 20992 and padded - p == 128
    assert padded % (tile_m * herd_m * 4) == 0
    assert padded // (herd_m * 4 * 4) - 1 <= cap
