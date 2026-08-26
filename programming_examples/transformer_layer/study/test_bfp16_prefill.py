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
    for mod, fn in (("rms_gemms_rope_bfp16_multi", "build_rms_gemms_rope_bfp16_module"),
                    ("o_ffn_bfp16_multi", "build_o_ffn_bfp16_module")):
        d = _builder_defaults(mod, fn)
        assert d["tile_n"] == pk.BFP16_N_TILE, (mod, d["tile_n"], pk.BFP16_N_TILE)
        assert d["tile_k_l1"] == pk.BFP16_K_CHUNK, (mod, d["tile_k_l1"], pk.BFP16_K_CHUNK)
        assert d["tile_m"] == planmod.BFP16_TILE_M, (mod, d["tile_m"])
    assert (planmod.BFP16_TILE_N, planmod.BFP16_TILE_K_L1) == (pk.BFP16_N_TILE, pk.BFP16_K_CHUNK)


def test_nine_bits_per_element_not_four_and_a_half():
    K, N = 256, 64
    W = (np.arange(K * N, dtype=np.float32).reshape(K, N) / (K * N) - 0.5).astype(bfloat16)
    packed = pk.pack_b_bfp16ebs8(W, pk.BFP16_N_TILE, pk.BFP16_K_CHUNK)
    assert packed.dtype == np.uint8
    assert packed.shape[:2] == (N // pk.BFP16_N_TILE, K // pk.BFP16_K_CHUNK)
    assert packed.nbytes == K * N * 9 // 8, (packed.nbytes, K * N)
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
    out = pk.pack_layer_bfp16(layer)
    assert sorted(out) == sorted(pk.BFP16_PREFILL_FIELDS)
    assert np.array_equal(out["wq"], pk.pack_b_bfp16ebs8(layer.wq, pk.BFP16_N_TILE, pk.BFP16_K_CHUNK))
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
    for t in tests:
        try:
            t()
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
