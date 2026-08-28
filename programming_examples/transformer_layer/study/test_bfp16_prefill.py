# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host tests for `w_bfp16_prefill` (doc 56 H4, queue item 20).

What is pinned here, and why each one exists:

- **The packed layout IS the builders' tile geometry.** `pack_b_bfp16ebs8`
  emits `[N/tile_n, K/tile_k_l1, tile_bytes]`, and the two bfp16 stitchers
  build every GEMM at their own defaults. If the two ever disagree the ELF
  reads the weight BO as garbage and the only symptom is a wrong token -- so
  the packer's constants are checked against the BUILDERS' signature defaults,
  not against a copy of them.
- **9 bits per element, not 4.5.** The brief for this item priced the whole
  hypothesis at "16 -> 4.5 bits/elt, ~3.5x". `bfp16ebs8` is 9 bytes per 8
  elements. The byte arithmetic is asserted against a real packed array.
- **The contract has ONE owner.** `awq_bfp_pack.quant_contract` is it; the
  plan package mirrors only the NAME (it stays dependency-free), and this
  test pins the agreement -- the `fa_cache_name` / `W4_GEMV_CONTRACT` pattern.
- **The flag hops exist, by source**: the binding's `precision_env_map` ->
  `prepare` (before the driver import) -> the driver's module-level read ->
  the layer dispatch -> the verify adapter's refusal to compile bf16 ELFs
  under a bfp16 plan.
- **The artifact set is keyed by the plan**, and a `prefill`-phase plan takes
  the prefill ELFs while keeping the shipped decode set.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from ml_dtypes import bfloat16

_HERE = os.path.dirname(os.path.abspath(__file__))
_PE = os.path.dirname(os.path.dirname(_HERE))
for _p in (_PE, os.path.join(_PE, "llms"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import model_profiles as mp  # noqa: E402
import resume  # noqa: E402
from llama32_1b_int4 import awq_bfp_pack as pk  # noqa: E402
from shared import model_adapter as ma  # noqa: E402
from shared.plan import LLAMA32_1B_INT4, NPU2_CAPS, QWEN3_0_6B, Workload, decoder_graph, plan  # noqa: E402
# `shared.plan.plan` the MODULE, not the `plan` function the package re-exports
import importlib  # noqa: E402

importlib.import_module("shared.plan.plan")
planmod = sys.modules["shared.plan.plan"]  # noqa: E402


def _builder_defaults(module_name, func_name):
    """The builder function's default (tile_n, tile_k_l1), read from ITS OWN
    signature -- no `air` import, no copy of the numbers."""
    path = Path(_PE) / "llms" / "llama32_1b_int4" / "multi_launch_builder" / f"{module_name}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == func_name)
    names = [a.arg for a in fn.args.args]
    defaults = dict(zip(names[len(names) - len(fn.args.defaults):],
                        [ast.literal_eval(d) if isinstance(d, ast.Constant) else None for d in fn.args.defaults]))
    return defaults


def test_the_packed_layout_is_the_builders_tile_geometry():
    """`[2026-08-27]` queue item 22 narrowed this to the DEFAULT width.

    The N tile is a parameter now (the builders always took it; the packer and
    the plan now read `LLAMA32_1B_INT4_BFP16_TILE_N`), so the invariant that can
    be pinned statically is the DEFAULT: builder signature default == packer
    default == plan default. The LIVE agreement -- this packer against THESE
    ELFs -- cannot be read from source at all, because the width the ELFs were
    built at is not in any Python file; it is enforced at run time by
    `assert_layout_agrees`, which the tests below pin.
    """
    for mod, fn in (("rms_gemms_rope_bfp16_multi", "build_rms_gemms_rope_bfp16_module"),
                    ("o_ffn_bfp16_multi", "build_o_ffn_bfp16_module")):
        d = _builder_defaults(mod, fn)
        assert d["tile_n"] == pk.BFP16_N_TILE_DEFAULT, (mod, d["tile_n"])
        assert d["tile_k_l1"] == pk.BFP16_K_CHUNK_DEFAULT, (mod, d["tile_k_l1"])
        assert d["tile_m"] == planmod.BFP16_TILE_M, (mod, d["tile_m"])
    assert (planmod.BFP16_TILE_N_DEFAULT, planmod.BFP16_TILE_K_L1) == (
        pk.BFP16_N_TILE_DEFAULT, pk.BFP16_K_CHUNK_DEFAULT)
    # the packer and the plan read ONE environment variable, under one name,
    # and admit exactly the same widths -- the mirroring pattern's pin
    assert pk.BFP16_TILE_N_ENV == planmod.BFP16_TILE_N_ENV
    assert pk.BFP16_TILE_N_SUPPORTED == planmod.BFP16_TILE_N_SUPPORTED
    assert pk.BFP16_N_TILE == planmod.BFP16_TILE_N  # both read the live env


def test_the_air_project_hook_accepts_what_the_cache_passes():
    """`[2026-08-27]` queue item 22. The int4/bfp16 prefill drivers REPLACE
    `cache.prepare_air_project` with their own function. When the seam grew
    `int4_gs` / `int4_k_chunk`, the two replacements were not updated, so every
    compile through that driver died with `TypeError: unexpected keyword
    argument 'int4_gs'` before a single kernel was built -- and nothing pinned
    the agreement. This does: the keywords the CALLER passes are read from
    `cache.compile_and_cache`'s source, and each replacement must accept them.
    """
    import ast

    cache_src = (Path(_PE) / "llms" / "shared" / "infra" / "cache.py").read_text(encoding="utf-8")
    tree = ast.parse(cache_src)
    # EVERY call site, unioned. cache.py has two, and they do not pass the same
    # keywords; taking the last one found made the first version of this test
    # vacuous (it passed against the very signature that breaks the driver).
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "prepare_air_project"]
    assert calls, "cache.py no longer calls prepare_air_project by that name"
    passed = {kw.arg for c in calls for kw in c.keywords if kw.arg}
    assert "quant" in passed and len(passed) > 1, (calls, passed)

    drv = (Path(_PE) / "llms" / "llama32_1b_int4" / "llama32_1b_int4_prefill.py").read_text(encoding="utf-8")
    dtree = ast.parse(drv)
    seen = 0
    for fn in dtree.body:
        if not (isinstance(fn, ast.FunctionDef) and fn.name.startswith("_prepare_air_project")):
            continue
        seen += 1
        names = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
        missing = passed - names
        assert not missing or fn.args.kwarg is not None, (
            f"{fn.name} cannot accept {sorted(missing)}, which "
            f"cache.compile_and_cache always passes")
    assert seen == 2, f"expected the int4 and bfp16 hooks, found {seen}"


def test_a_width_nobody_compiled_is_refused_not_packed():
    """Both readers refuse the same set, and both accept the same set."""
    # this suite runs its modules in ONE process, so the caller's value is saved
    # and put back -- the first version deleted it, which silently changed what
    # every later test in the file saw
    env_before = os.environ.get(pk.BFP16_TILE_N_ENV)
    try:
        for reader in (pk._read_bfp16_tile_n, planmod._read_bfp16_tile_n):
            for bad in ("48", "16", "0", "wide", "-32"):
                os.environ[pk.BFP16_TILE_N_ENV] = bad
                with pytest.raises(ValueError):
                    reader()
            for good in pk.BFP16_TILE_N_SUPPORTED:
                os.environ[pk.BFP16_TILE_N_ENV] = str(good)
                assert reader() == good
            os.environ.pop(pk.BFP16_TILE_N_ENV, None)
            assert reader() == pk.BFP16_N_TILE_DEFAULT  # unset -> the default
    finally:
        if env_before is None:
            os.environ.pop(pk.BFP16_TILE_N_ENV, None)
        else:
            os.environ[pk.BFP16_TILE_N_ENV] = env_before


def _packed_weights(tile_n, n_layers=2, K=256, N=512, k_chunk=None):
    """A weights-shaped object whose per-layer buffers really are packed at
    `tile_n` -- the tests derive from the buffers, exactly as the guard does."""
    k_chunk = pk.BFP16_K_CHUNK_DEFAULT if k_chunk is None else k_chunk

    class _L:
        pass

    class _W:
        pass

    w = _W()
    w.layers = []
    for _ in range(n_layers):
        layer = _L()
        for f in pk.BFP16_PREFILL_FIELDS:
            dense = np.zeros((K, N), dtype=bfloat16)
            setattr(layer, f, dense)
        layer._bfp_packed = {f: pk.pack_b_bfp16ebs8(getattr(layer, f), tile_n, k_chunk)
                             for f in pk.BFP16_PREFILL_FIELDS}
        w.layers.append(layer)
    return w


def test_the_packed_layout_is_derived_from_the_buffers_not_recorded_beside_them():
    """BYPASS: a recorded width goes stale or is forged (second review, #4).

    There is no stamp. The layout is solved out of the packed array against the
    dense array it came from -- `[N/tile_n, K/tile_k_l1, tile_n*tile_k_l1//8*9]`
    -- so a repacked or swapped layer is a REFUSAL rather than a stale record,
    and no object can be certified without buffers to certify.
    """
    for tn in pk.BFP16_TILE_N_SUPPORTED:
        assert pk.derive_packed_layout(_packed_weights(tn)) == (tn, pk.BFP16_K_CHUNK_DEFAULT)

    # ONE layer repacked at another width: the whole model refuses
    w = _packed_weights(32)
    w.layers[1]._bfp_packed["wq"] = pk.pack_b_bfp16ebs8(
        w.layers[1].wq, 128, pk.BFP16_K_CHUNK_DEFAULT)
    with pytest.raises(pk.Bfp16LayoutError):
        pk.derive_packed_layout(w)

    # nothing packed at all is not "the default", it is undecidable
    class _Empty:
        layers = []

    with pytest.raises(pk.Bfp16LayoutError):
        pk.derive_packed_layout(_Empty())

    # a buffer whose record axis contradicts its own tiling is refused
    w = _packed_weights(32)
    bad = w.layers[0]._bfp_packed["wq"]
    w.layers[0]._bfp_packed["wq"] = bad[:, :, : bad.shape[2] - 9]
    with pytest.raises(pk.Bfp16LayoutError):
        pk.derive_packed_layout(w)


def test_the_derivation_looks_where_the_packed_buffers_actually_are():
    """The derivation is only fail-CLOSED in the useful direction if it can see
    the buffers: a field-name drift would make it find none and refuse the
    LEGITIMATE path, which is the too-strict failure this round was about.

    Both producers key their packed dicts by the loader's own field names, so
    that agreement is pinned rather than assumed.
    """
    from awq_pack import _HF_AWQ_LAYER_MAP

    assert set(_HF_AWQ_LAYER_MAP.values()) == set(pk.BFP16_PREFILL_FIELDS)
    # and the dense array a field holds is the (K, N) the packer consumes, so
    # the derivation solves against the same orientation it was packed in
    K, N = 256, 512
    dense = np.zeros((K, N), dtype=bfloat16)
    packed = pk.pack_b_bfp16ebs8(dense, 64, pk.BFP16_K_CHUNK_DEFAULT)
    assert pk.derive_layout_from_packed(packed, dense.shape) == (64, pk.BFP16_K_CHUNK_DEFAULT)
    # WHAT THE CROSS-CHECK DOES AND DOES NOT CONSTRAIN, stated precisely in
    # BOTH directions: the record axis pins the PRODUCT tile_n * tile_k_l1,
    # while the split comes from the dense shape.
    #   NON-SQUARE: a transposed dense shape yields a different, internally
    #   consistent pair -- (32, 256) here instead of (64, 128), same 9216-byte
    #   record -- which the invariant then refuses only because (32, 256) is not
    #   buildable, not because it noticed the transpose.
    assert pk.derive_layout_from_packed(packed, (N, K)) == (32, 256)
    assert not pk.layout_is_buildable(32, 256)
    #   SQUARE: transposing preserves the shape, so the SAME buildable pair is
    #   derived and the geometry comparison cannot see the transpose at all.
    #   This guard is about tile WIDTH, not about content orientation, and it
    #   does not claim otherwise.
    sq = np.zeros((512, 512), dtype=bfloat16)
    sq_packed = pk.pack_b_bfp16ebs8(sq, 128, pk.BFP16_K_CHUNK_DEFAULT)
    assert (pk.derive_layout_from_packed(sq_packed, sq.shape)
            == pk.derive_layout_from_packed(sq_packed, sq.shape[::-1])
            == (128, pk.BFP16_K_CHUNK_DEFAULT))


def test_a_declaration_is_evidence_not_authority():
    """BYPASS: unsupported geometry self-certifies through a sidecar (#3).

    A declaration is usable only if it names a layout THIS BUILD CAN PRODUCE.
    Otherwise a hand-written sidecar plus a packer told the same numbers agrees
    with itself and admits a kernel nobody ever built -- the item-28 shape:
    certifying that a declaration EXISTS while claiming the layout MATCHES.
    """
    for tn in pk.BFP16_TILE_N_SUPPORTED:
        assert pk.layout_is_buildable(tn, pk.BFP16_K_CHUNK_DEFAULT)
    for tn, tk in ((256, 128), (48, 128), (16, 128), (0, 128), ("32", 128),
                   (32, 64), (32, 256), (32, None), (None, None)):
        assert not pk.layout_is_buildable(tn, tk), (tn, tk)
    with tempfile.TemporaryDirectory() as d:
        # writing an unbuildable declaration is refused at the source
        with pytest.raises(pk.Bfp16LayoutError):
            pk.write_declared_layout(d, 256, 128, why="forged")
        # and a hand-written one is not read as a declaration
        Path(d, pk.BFP16_GEOMETRY_SIDECAR).write_text(
            json.dumps({"tile_n": 48, "tile_k_l1": 128}), encoding="utf-8")
        assert pk.read_declared_layout(d) is None
        with pytest.raises(pk.Bfp16LayoutError):
            pk.assert_layout_agrees(_packed_weights(32), d)
        # malformed / non-dict / missing keys are all "no declaration"
        for body in ("{not json", "[]", '{"tile_n": 32}', '{"tile_k_l1": 128}'):
            Path(d, pk.BFP16_GEOMETRY_SIDECAR).write_text(body, encoding="utf-8")
            assert pk.read_declared_layout(d) is None


def test_the_one_invariant_admits_only_when_both_facts_agree():
    """The guard itself: derived layout == declared, buildable layout."""
    with tempfile.TemporaryDirectory() as d32, tempfile.TemporaryDirectory() as d128, \
            tempfile.TemporaryDirectory() as undeclared:
        pk.write_declared_layout(d32, 32, 128, why="test")
        pk.write_declared_layout(d128, 128, 128, why="test")
        w32, w128 = _packed_weights(32), _packed_weights(128)
        assert pk.assert_layout_agrees(w32, d32)["tile_n"] == 32
        assert pk.assert_layout_agrees(w128, d128)["tile_n"] == 128
        # the mismatch this whole guard exists for -- silent otherwise
        with pytest.raises(pk.Bfp16LayoutError):
            pk.assert_layout_agrees(w32, d128)
        with pytest.raises(pk.Bfp16LayoutError):
            pk.assert_layout_agrees(w128, d32)
        # an undeclared set, and an unnamed one
        with pytest.raises(pk.Bfp16LayoutError):
            pk.assert_layout_agrees(w32, undeclared)
        with pytest.raises(pk.Bfp16LayoutError):
            pk.assert_layout_agrees(w32, None)
        # and the live environment does NOT enter into it: what the buffers
        # ARE is what is compared, whatever the packer is currently set to
        env_before = os.environ.get(pk.BFP16_TILE_N_ENV)
        os.environ[pk.BFP16_TILE_N_ENV] = "128"
        try:
            assert pk.assert_layout_agrees(w32, d32)["tile_n"] == 32
            with pytest.raises(pk.Bfp16LayoutError):
                pk.assert_layout_agrees(w32, d128)
        finally:
            if env_before is None:
                os.environ.pop(pk.BFP16_TILE_N_ENV, None)
            else:
                os.environ[pk.BFP16_TILE_N_ENV] = env_before


def test_an_existing_sets_declaration_is_read_only():
    """BLOCKING, final review: cached ELFs were RELABELLED with the requested
    width.

    32-wide ELFs present + width 128 requested: both compiles are skipped as
    "already built", the declaration was rewritten to 128, the host then packed
    at 128, and the guard read its OWN new label and admitted -- a rubber stamp
    on the exact mismatch it exists to catch, and worse than no declaration.
    Only the act that BUILDS a set may write its declaration.
    """
    with tempfile.TemporaryDirectory() as d:
        pk.write_declared_layout(d, 32, 128, why="built at 32")
        # re-recording the SAME layout is idempotent
        assert pk.write_declared_layout(d, 32, 128, why="again")["tile_n"] == 32
        # relabelling to another width refuses
        with pytest.raises(pk.Bfp16LayoutError):
            pk.write_declared_layout(d, 128, 128, why="relabel")
        assert pk.read_declared_layout(d)["tile_n"] == 32
        # ... unless the caller states it just rebuilt the ELFs
        assert pk.write_declared_layout(d, 128, 128, why="rebuilt",
                                        allow_replace=True)["tile_n"] == 128


def test_a_populated_set_cannot_be_built_into_at_another_width():
    """The same defect at the other end: the CREATE-path preflight.

    A set may be built into only when it is EMPTY of bfp16 ELFs or already
    declares exactly this layout -- otherwise a partially populated set becomes
    a MIXED-width set carrying one label.
    """
    with tempfile.TemporaryDirectory() as d:
        # empty: fine
        assert pk.assert_can_build_into(d, 128, 128)
        # declared 32, building 32: fine (the resume case)
        pk.write_declared_layout(d, 32, 128, why="built at 32")
        assert pk.assert_can_build_into(d, 32, 128)
        # declared 32, building 128: refuse
        with pytest.raises(pk.Bfp16LayoutError):
            pk.assert_can_build_into(d, 128, 128)
    with tempfile.TemporaryDirectory() as d:
        # ELFs present, nothing declared: their width is unknown -> refuse
        Path(d, pk.BFP16_SET_ELFS[0]).write_bytes(b"\x7fELF stub")
        with pytest.raises(pk.Bfp16LayoutError):
            pk.assert_can_build_into(d, 32, 128)
    with tempfile.TemporaryDirectory() as d:
        # PARTIALLY populated at another width -> refuse
        pk.write_declared_layout(d, 32, 128, why="built at 32")
        Path(d, pk.BFP16_SET_ELFS[0]).write_bytes(b"\x7fELF stub")
        with pytest.raises(pk.Bfp16LayoutError):
            pk.assert_can_build_into(d, 128, 128)


def test_the_driver_only_declares_a_set_it_actually_built():
    """And the driver must run that preflight BEFORE compiling and write the
    declaration only when it compiled something -- read structurally, because
    the defect was a write on a path that compiled nothing."""
    import ast

    src = (Path(_PE) / "llms" / "llama32_1b_int4"
           / "llama32_1b_int4_prefill.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    main = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    alias = {}
    for node in ast.walk(main):
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("awq_bfp_pack"):
            for a in node.names:
                alias[a.asname or a.name] = a.name
    real = lambda n: alias.get(n, n)  # noqa: E731

    pre_ln = write_ln = None
    for node in ast.walk(main):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if real(node.func.id) == "assert_can_build_into":
                pre_ln = node.lineno if pre_ln is None else min(pre_ln, node.lineno)
            if real(node.func.id) == "write_declared_layout":
                write_ln = node.lineno if write_ln is None else min(write_ln, node.lineno)
    assert pre_ln, "the compile path must run the create-path preflight"
    assert write_ln and pre_ln < write_ln, (pre_ln, write_ln)
    # the write must be guarded by a flag the compiles set
    guarded = False
    for node in ast.walk(main):
        if isinstance(node, ast.If):
            names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
            calls = {real(c.func.id) for c in ast.walk(node)
                     if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
            if "write_declared_layout" in calls and names:
                guarded = True
    assert guarded, (
        "the declaration write must be conditional on this invocation having "
        "actually compiled ELFs; an unconditional write relabels a cached set")


def test_compiling_a_fresh_set_establishes_the_declaration_and_proceeds():
    """REGRESSION (second review, #1): a guard that blocks correct use is not
    fail-closed, it is broken.

    Create-vs-consume, read from the driver's source: nothing may check an
    artifact set's declaration BEFORE the compile block that writes it, and the
    consume-time check must sit after the `--compile-only` return.
    """
    import ast

    src = (Path(_PE) / "llms" / "llama32_1b_int4"
           / "llama32_1b_int4_prefill.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    main = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    # resolve `from awq_bfp_pack import X as Y` so an ALIAS cannot evade this
    # (the first version matched literal names and a `_early` alias walked past
    # it -- caught by tamper-checking the test itself)
    alias = {}
    for node in ast.walk(main):
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("awq_bfp_pack"):
            for a in node.names:
                alias[a.asname or a.name] = a.name
    def real(name):
        return alias.get(name, name)

    write_ln = check_ln = None
    for node in ast.walk(main):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            nm = real(node.func.id)
            if nm == "write_declared_layout":
                write_ln = node.lineno if write_ln is None else min(write_ln, node.lineno)
            if nm == "assert_layout_agrees":
                check_ln = node.lineno if check_ln is None else min(check_ln, node.lineno)
    assert write_ln, "the bfp16 compile path must WRITE the set's declaration"
    assert check_ln, "the run path must CHECK the layout invariant"
    assert write_ln < check_ln, (
        f"the consume-time check (line {check_ln}) must come AFTER the compile "
        f"path that establishes the declaration (line {write_ln}); otherwise a "
        f"fresh cache can never be compiled")
    # and no OTHER geometry check may precede the compile block
    for node in ast.walk(main):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and real(node.func.id) in ("assert_layout_agrees", "read_declared_layout")
                and node.lineno < write_ln):
            raise AssertionError(
                f"line {node.lineno} checks the set's declaration before the "
                f"compile block writes it -- this is the bootstrap lock")
    # the loader itself must take no cache argument at all
    load = next(n for n in ast.parse(
        (Path(_PE) / "llms" / "llama32_1b_int4" / "awq_bfp_pack.py").read_text(
            encoding="utf-8")).body
        if isinstance(n, ast.FunctionDef) and n.name == "load_awq_weights_bfp")
    names = {a.arg for a in load.args.args} | {a.arg for a in load.args.kwonlyargs}
    assert "prefill_cache_dir" not in names, (
        "loading a checkpoint is not where the weights meet an ELF; a check "
        "there is a second answer to what a missing declaration means")


def test_the_inference_driver_checks_per_artifact_set_not_per_pack():
    """The transcode is idempotent -- it returns early once the weights are
    packed -- so a check inside it only ever sees the FIRST artifact set. The
    invariant must be enforced from `prepare_runtime`, which runs per (weights,
    set) pair."""
    import ast

    src = (Path(_PE) / "llms" / "llama32_1b_int4"
           / "llama32_1b_int4_inference.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "prepare_runtime" in fns and "_pack_prefill_weights_bfp16" in fns

    def calls(node):
        return {c.func.id for c in ast.walk(node)
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}

    assert "assert_layout_agrees" in calls(fns["prepare_runtime"])
    assert "assert_layout_agrees" not in calls(fns["_pack_prefill_weights_bfp16"]), (
        "the check must not live inside the idempotent transcode")
    body = fns["_pack_prefill_weights_bfp16"].body
    first_stmt = next(st for st in body if not isinstance(st, ast.Expr))
    assert isinstance(first_stmt, ast.If) and any(
        isinstance(x, ast.Return) for x in ast.walk(first_stmt)), (
        "the pack is expected to be idempotent; if it stops being so, revisit "
        "where the layout check belongs")


def test_every_packer_entry_point_defaults_to_the_declared_default_width():
    """BYPASS: `awq_pack_for_npu_bfp16` and `load_awq_weights_bfp` defaulted
    `n_tile` to **64** while the builders, the shipped cache and
    `BFP16_N_TILE_DEFAULT` are all 32 -- so their default output was silently
    misread by the default ELF. Read from the SIGNATURES so a future edit
    cannot drift them apart."""
    import ast

    src = (Path(_PE) / "llms" / "llama32_1b_int4" / "awq_bfp_pack.py").read_text(
        encoding="utf-8")
    tree = ast.parse(src)
    checked = 0
    for fn in tree.body:
        if not (isinstance(fn, ast.FunctionDef)
                and fn.name in ("awq_pack_for_npu_bfp16", "load_awq_weights_bfp",
                                "pack_layer_bfp16")):
            continue
        names = [a.arg for a in fn.args.args]
        defaults = dict(zip(names[len(names) - len(fn.args.defaults):], fn.args.defaults))
        for arg in ("n_tile", "k_chunk"):
            if arg not in defaults:
                continue
            checked += 1
            node = defaults[arg]
            assert isinstance(node, ast.Constant) and node.value is None, (
                f"{fn.name}({arg}=...) must default to None -- i.e. follow the "
                f"live packer geometry -- not to a hardcoded width "
                f"({ast.dump(node)})")
    assert checked >= 5, checked
    # and with NOTHING selected, the live width IS the declared default
    env_before = os.environ.pop(pk.BFP16_TILE_N_ENV, None)
    try:
        assert pk._read_bfp16_tile_n() == pk.BFP16_N_TILE_DEFAULT
    finally:
        if env_before is not None:
            os.environ[pk.BFP16_TILE_N_ENV] = env_before


def test_pack_layer_follows_the_live_width_not_the_default():
    """The gate the first round of this item weakened: with an env-selected
    width, `pack_layer_bfp16` called WITHOUT an explicit tile must follow it."""
    import importlib

    # restore whatever the caller's environment was: this suite runs its
    # modules in ONE process, so leaking a reloaded module or a stray variable
    # changes what every later test sees.
    env_before = os.environ.get(pk.BFP16_TILE_N_ENV)
    os.environ[pk.BFP16_TILE_N_ENV] = "128"
    try:
        live = importlib.reload(pk)
        assert live.BFP16_N_TILE == 128 and live.BFP16_N_TILE_DEFAULT == 32

        class _L:
            pass

        layer = _L()
        for f in live.BFP16_PREFILL_FIELDS:
            setattr(layer, f, np.zeros((256, 256), dtype=bfloat16))
        out = live.pack_layer_bfp16(layer)
        # [N/tile_n, K/tile_k_l1, tile_bytes] -- the first axis IS the width
        assert out["wq"].shape[0] == 256 // 128, out["wq"].shape
        assert np.array_equal(out["wq"], live.pack_b_bfp16ebs8(
            layer.wq, 128, live.BFP16_K_CHUNK))
    finally:
        if env_before is None:
            os.environ.pop(pk.BFP16_TILE_N_ENV, None)
        else:
            os.environ[pk.BFP16_TILE_N_ENV] = env_before
        importlib.reload(pk)
    expected = int(env_before) if env_before else pk.BFP16_N_TILE_DEFAULT
    assert pk.BFP16_N_TILE == expected, (pk.BFP16_N_TILE, expected)


def test_nine_bits_per_element_not_four_and_a_half():
    """And the byte count is INVARIANT in the tile width, which is why a
    packer/ELF width mismatch cannot be caught by any size check (queue item
    22) -- the run-time refusal below is the only thing that can catch it."""
    K, N = 512, 256
    W = (np.arange(K * N, dtype=np.float32).reshape(K, N) / (K * N) - 0.5).astype(bfloat16)
    for tile_n in pk.BFP16_TILE_N_SUPPORTED:
        packed = pk.pack_b_bfp16ebs8(W, tile_n, pk.BFP16_K_CHUNK_DEFAULT)
        assert packed.dtype == np.uint8
        assert packed.shape[:2] == (N // tile_n, K // pk.BFP16_K_CHUNK_DEFAULT)
        assert packed.nbytes == K * N * 9 // 8, (tile_n, packed.nbytes, K * N)
        assert packed.nbytes / (K * N) == 1.125
    # the whole Llama-3.2-1B prefill weight set, in both formats
    elts = 16 * (2048 * 2048 * 2 + 2048 * 512 * 2 + 3 * 2048 * 8192)
    assert round(elts * 2 / 1e6, 1) == 1946.2
    assert round(elts * 1.125 / 1e6, 1) == 1094.7
    assert round(2 / 1.125, 3) == 1.778  # NOT 3.5


def test_pack_layer_transcodes_the_dense_bf16_the_bf16_arm_uses():
    """Both arms must compute over bit-identical weights: the bfp16 BO is a
    transcode of the SAME dequantized array, not a second dequantization."""
    class _L:
        pass

    layer = _L()
    for f in pk.BFP16_PREFILL_FIELDS:
        setattr(layer, f, np.zeros((128, 64), dtype=bfloat16))
    layer.wq = (np.random.default_rng(0).standard_normal((128, 64)) / 8).astype(bfloat16)
    out = pk.pack_layer_bfp16(layer, n_tile=pk.BFP16_N_TILE_DEFAULT,
                              k_chunk=pk.BFP16_K_CHUNK_DEFAULT)
    assert sorted(out) == sorted(pk.BFP16_PREFILL_FIELDS)
    assert np.array_equal(out["wq"], pk.pack_b_bfp16ebs8(
        layer.wq, pk.BFP16_N_TILE_DEFAULT, pk.BFP16_K_CHUNK_DEFAULT))
    # the source array is untouched -- the bf16 arm still sees exactly it
    assert layer.wq.dtype == bfloat16


def test_the_contract_has_one_owner_and_the_plan_mirrors_only_the_name():
    c = pk.quant_contract(128)
    assert c["quant_gemm_contract_name"] == planmod.W_BFP16_GEMM_CONTRACT
    assert c["quant_weight_bytes_per_element"] == 1.125
    assert c["quant_group_size"] == 8  # the FORMAT's group, not the checkpoint's 128
    assert c["checkpoint_group_size"] == 128
    # the plan package IMPORTS nothing from a model directory (it may name one
    # in a comment; what it must not do is depend on one)
    tree = ast.parse(Path(_PE, "llms", "shared", "plan", "plan.py").read_text(encoding="utf-8"))
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            imported.add(n.module.split(".")[0])
    assert not imported & {"awq_bfp_pack", "llama32_1b_int4", "awq_repacker", "qwen3_0_6b"}, imported


def test_quant_columns_are_the_bfp16_owners_and_differ_from_w4s():
    bfp = ma.quant_columns("llama32_1b_int4", "w_bfp16_prefill")
    w4 = ma.quant_columns("llama32_1b_int4", "w4_decode")
    assert bfp["quant_gemm_contract"] != w4["quant_gemm_contract"]
    # decode is UNTOUCHED by this plan: the same int4 GEMV contract on both
    assert bfp["quant_gemv_contract"] == w4["quant_gemv_contract"]
    assert "bfp16ebs8" in bfp["quant_packing_scheme"]
    assert "no scale plane" in bfp["quant_scale_layout"].lower() or "none" in bfp["quant_scale_layout"].lower()
    assert ma.quant_columns("llama32_1b", "bf16") == {}


def test_the_bfp16_plan_is_the_shipped_stitchers_launch_for_launch():
    g = decoder_graph(LLAMA32_1B_INT4)
    base = plan(g, Workload("prefill", 2048, 2048, 2048, "w4_decode"), NPU2_CAPS)
    bfp = plan(g, Workload("prefill", 2048, 2048, 2048, "w_bfp16_prefill"), NPU2_CAPS)
    names = {s.name: s for s in bfp.stages}
    assert "rms_gemms_rope_bfp16" in names and "o_ffn_bfp16" in names
    assert names["rms_gemms_rope_bfp16"].launches == 6
    assert names["o_ffn_bfp16"].launches == 8
    assert names["flash_attn"].launches == 1
    # 16 x (6 + 1 + 8) + the 8-launch bf16 LM head
    assert bfp.total_launches == 248 and base.total_launches == 328
    assert bfp.total_submissions == base.total_submissions == 49
    assert bfp.total_host_ops == base.total_host_ops == 18
    # weight bytes: 9/16 of bf16 on the GEMM weights, norms unchanged
    gemm_bf16 = sum(s.weight_bytes for s in base.stages if s.name in ("rms_gemms_rope", "o_ffn"))
    gemm_bfp = sum(s.weight_bytes for s in bfp.stages if s.name in ("rms_gemms_rope_bfp16", "o_ffn_bfp16"))
    assert 1.77 < gemm_bf16 / gemm_bfp < 1.79
    # every bfp16 GEMM candidate is analytical_unmeasured, at the builder's tiles
    cands = [c for s in bfp.stages for c in s.candidates.values()]
    assert len(cands) == 7 and all(c["source"] == "analytical_unmeasured" for c in cands)
    assert all(c["method"] == "bfp16-direct" and c["gflops"] is None for c in cands)
    assert all(c["tile"]["tile_n"] == planmod.BFP16_TILE_N for c in cands)
    assert all(c["contract"] == planmod.W_BFP16_GEMM_CONTRACT for c in cands)
    # the registry policy is a RECORDED rejected alternative, with its cost
    why = " ".join(r[1] for r in bfp.rejected)
    assert "no quant parameter" in why and "tile_n 32" in why
    assert bfp.sha != base.sha


def test_the_bfp16_plan_refuses_what_has_no_stitcher():
    g4 = decoder_graph(LLAMA32_1B_INT4)
    for wl, needle in (
        (Workload("decode", 1, 1, 2048, "w_bfp16_prefill"), "names the PREFILL GEMM"),
        (Workload("prefill", 2048, 2048, 2048, "w_bfp16_prefill"), None),
    ):
        if needle is None:
            plan(g4, wl, NPU2_CAPS)
            continue
        try:
            plan(g4, wl, NPU2_CAPS)
        except ValueError as exc:
            assert needle in str(exc)
        else:
            raise AssertionError(f"expected a refusal for {wl}")
    # a qk-norm model has no bfp16 prefill sibling
    try:
        plan(decoder_graph(QWEN3_0_6B), Workload("prefill", 2048, 2048, 2048, "w_bfp16_prefill"), NPU2_CAPS)
    except ValueError as exc:
        assert "qk-norm prefill stitcher" in str(exc)
    else:
        raise AssertionError("expected a refusal for a qk-norm model")


def test_bfp16_prefill_is_the_h4_row_of_doc_56():
    prof = mp.profile("bfp16-prefill")
    assert prof.models == ("llama32_1b_int4",)
    assert prof.decode_ctxs == () and prof.prefill_Ms == {"llama32_1b_int4": ()}
    rungs = prof.rungs()
    assert len(rungs) == 2 and all(r.phase == "prefill" and r.curve == mp.KERNEL_SCALING for r in rungs)
    assert [r.precision_plan for r in rungs] == ["w4_decode", "w_bfp16_prefill"]
    assert [r.case_id for r in rungs] == ["llama32_1b_int4/prefill/M2048/ctx2048/w4_decode",
                                          "llama32_1b_int4/prefill/M2048/ctx2048/w_bfp16_prefill"]
    # SAME M, SAME prompt length, SAME gate capacity: only the plan differs
    a, b = rungs
    assert a.M == b.M == 2048 and a.prompt_tokens == b.prompt_tokens == 2048
    assert a.gate_prompt_tokens == b.gate_prompt_tokens == 2048
    assert a.gate_max_seq == b.gate_max_seq == 2048 + mp.GATE_N_TOKENS
    # distinct resume identities, distinct artifact sets, one CSV
    assert resume.rung_key(a.mode, a.seq, a.extra) != resume.rung_key(b.mode, b.seq, b.extra)
    assert prof.expected_files() == ["model_llama32_1b_int4.csv"]
    assert prof.precision_plans_used() == ("w4_decode", "w_bfp16_prefill")
    bound = prof.bind({("llama32_1b_int4", 2048, "w4_decode"): {"prefill_cache": "/c/p", "decode_cache": "/c/d"},
                       ("llama32_1b_int4", 2048, "w_bfp16_prefill"): {"prefill_cache": "/c/bfp", "decode_cache": "/c/d"}})
    assert bound.artifact_sets() == [("llama32_1b_int4", 2048, "w4_decode"),
                                     ("llama32_1b_int4", 2048, "w_bfp16_prefill")]
    assert bound.expected_rows() == {"model_llama32_1b_int4.csv": {"rows": 2, "measured": 2, "skipped": 0}}
    # one arm compiled, the other not: a complete SKIP naming what to build
    half = prof.bind({("llama32_1b_int4", 2048, "w4_decode"): {"prefill_cache": "/c/p", "decode_cache": "/c/d"}},
                     {("llama32_1b_int4", 2048, "w_bfp16_prefill"): "assembled from the shipped bfp16 ELFs"})
    skips = {r.case_id: r.skip_reason for r in half.rungs() if r.skip_reason}
    assert list(skips) == ["llama32_1b_int4/prefill/M2048/ctx2048/w_bfp16_prefill"]
    assert "w_bfp16_prefill" in skips["llama32_1b_int4/prefill/M2048/ctx2048/w_bfp16_prefill"]


def test_a_prefill_phase_plan_takes_the_prefill_set_and_keeps_the_shipped_decode():
    with tempfile.TemporaryDirectory() as d:
        llms = Path(d) / "llms"
        for c in ("prefill_kernel_cache", "decode_kernel_cache"):
            (llms / "llama32_1b_int4" / "build_peano" / c).mkdir(parents=True)
            (llms / "llama32_1b_int4" / "build_peano" / c / "manifest.json").write_text("{}")
        root = Path(d) / "compiled"
        plans = ("w4_decode", "w_bfp16_prefill")
        compiled, notes = mp.discover_compiled(("llama32_1b_int4",), root, llms_dir=llms, precision_plans=plans)
        # the shipped plan binds; the bfp16 one is a skip that names WHERE it goes
        assert ("llama32_1b_int4", 2048, "w4_decode") in compiled
        assert ("llama32_1b_int4", 2048, "w_bfp16_prefill") not in compiled
        note = notes[("llama32_1b_int4", 2048, "w_bfp16_prefill")]
        assert "w_bfp16_prefill/M<M>/prefill_kernel_cache" in note and "PREFILL ELFs" in note
        # once assembled, it binds with the SHIPPED decode cache -- decode is untouched
        bfp = root / "llama32_1b_int4" / "w_bfp16_prefill" / "M2048" / "prefill_kernel_cache"
        bfp.mkdir(parents=True)
        (bfp / "manifest.json").write_text("{}")
        compiled, _ = mp.discover_compiled(("llama32_1b_int4",), root, llms_dir=llms, precision_plans=plans)
        got = compiled[("llama32_1b_int4", 2048, "w_bfp16_prefill")]
        assert got["prefill_cache"] == str(bfp)
        assert got["decode_cache"].endswith("build_peano/decode_kernel_cache")
        assert got["decode_cache"] == compiled[("llama32_1b_int4", 2048, "w4_decode")]["decode_cache"]


def test_the_flag_hops_exist_by_source():
    drv = Path(_PE, "llms", "llama32_1b_int4", "llama32_1b_int4_inference.py").read_text(encoding="utf-8")
    tree = ast.parse(drv)
    # the flag is read ONCE, at module level, and validated
    assert 'PREFILL_DTYPE_ENV = "LLAMA32_1B_INT4_PREFILL_DTYPE"' in drv
    assigns = [n for n in tree.body if isinstance(n, ast.Assign)
               and any(getattr(t, "id", None) == "PREFILL_DTYPE" for t in n.targets)]
    assert len(assigns) == 1, "PREFILL_DTYPE must be read once, at import"
    # prepare_runtime branches to the bfp16 pack, run_npu_prefill to the bfp16 block
    for fn_name, needle in (("prepare_runtime", "_pack_prefill_weights_bfp16"),
                            ("run_npu_prefill", "_bfp16_layer_runner")):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn_name)
        body = ast.unparse(fn)
        assert "PREFILL_DTYPE" in body and needle in body, fn_name
    # the KV append needs k_roped/v from the bfp16 block: with_kv must be passed
    assert "with_kv=True" in drv
    # `[2026-08-26, review round]` THE TWO ARMS MUST MATCH ON BO RESIDENCY.
    # The bf16 branch runs `run_transformer_block`, which passes
    # shared_nonstatic=True on both fused stages and gives flash_attn no
    # per-layer bo_key; the bfp16 branch must ask for the same policy, or the
    # H4 A/B differs in residency as well as in its two GEMM ELFs and the
    # difference lands in the measured GEMM delta (Codex review of a3a8f9f3).
    assert "shared_nonstatic=True" in drv, "the bfp16 arm must run the bf16 arm's residency policy"
    pf = Path(_PE, "llms", "llama32_1b_int4", "llama32_1b_int4_prefill.py").read_text(encoding="utf-8")
    ptree = ast.parse(pf)
    fn = next(n for n in ptree.body if isinstance(n, ast.FunctionDef) and n.name == "_run_layer_bfp16")
    assert "shared_nonstatic" in [a.arg for a in fn.args.args + fn.args.kwonlyargs]
    body = ast.unparse(fn)
    # both fused stages honour it, and the FA call drops its per-layer bo_key
    assert body.count("if shared_nonstatic else") >= 2, body.count("if shared_nonstatic else")
    assert "_fa_kw" in body and "shared_nonstatic" in body
    assert "'bo_key': f'flash_attn_L{layer_idx}'" in body or 'bo_key": f"flash_attn_L{layer_idx}"' in body
    # and the DEFAULT is off, so the standalone verify / diagnosis paths, which
    # persist per-layer intermediates, never receive pooled views
    d = dict(zip([a.arg for a in fn.args.args][-len(fn.args.defaults):],
                 [ast.literal_eval(x) for x in fn.args.defaults]))
    assert d["shared_nonstatic"] is False, d
    # the binding drives the flag, and pins BOTH directions
    b = ma.MODELS["llama32_1b_int4"]
    assert set(b.precision_env_map) == {"w4_decode", "w_bfp16_prefill"}
    assert all(list(v) == ["LLAMA32_1B_INT4_PREFILL_DTYPE"] for v in b.precision_env_map.values())
    # prepare applies the env BEFORE importing the driver (the flag is read at import)
    src = inspect.getsource(ma.ModelAdapter.prepare)
    assert src.index("precision_env") < src.index("_import_driver")
    # the gate subprocess gets the same env
    rm = Path(_HERE, "run_model.py").read_text(encoding="utf-8")
    assert "precision_env(gate.get(" in rm
    # and the verify adapter refuses to compile bf16 ELFs under a bfp16 plan
    va = Path(_PE, "llms", "llama32_1b_int4", "verify_adapter.py").read_text(encoding="utf-8")
    assert "inference_prefill_dtype() != \"bf16\"" in va
    assert "compile_prefill_kernels" in va


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    # `[2026-08-27, second review]` this suite runs its modules in ONE process,
    # and a test that leaves LLAMA32_1B_INT4_BFP16_TILE_N changed silently
    # weakens every test after it -- which is exactly how a suite launched at
    # 64 came to run the rest at the default. Enforced here rather than
    # remembered at each site.
    _env_at_start = os.environ.get(pk.BFP16_TILE_N_ENV)
    for t in tests:
        try:
            t()
            _now = os.environ.get(pk.BFP16_TILE_N_ENV)
            if _now != _env_at_start:
                os.environ.pop(pk.BFP16_TILE_N_ENV, None)
                if _env_at_start is not None:
                    os.environ[pk.BFP16_TILE_N_ENV] = _env_at_start
                raise AssertionError(
                    f"{t.__name__} left {pk.BFP16_TILE_N_ENV}={_now!r} "
                    f"(was {_env_at_start!r}); a leaked width weakens every "
                    f"test after it in this single-process suite")
            print(f"  PASS  {t.__name__}")
        except Exception as exc:
            failed += 1
            import traceback

            traceback.print_exc()
            print(f"  FAIL  {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"bfp16_prefill tests: {len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
