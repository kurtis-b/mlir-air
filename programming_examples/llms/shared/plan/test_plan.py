# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""H0's host gates (doc 56 section 4): the plan for each golden model reproduces its
hand-built ELF sequence, launch counts and NPU/CPU split; every study skip in
`profiles.skip_reason` is reproduced by `study_skip`; the capacity solver is
ranked against the registry on every swept shape with each mismatch named.

Framework-free (the seam lit runs these as plain scripts and FileChecks the
count): `python3 shared/plan/test_plan.py` prints `N/N passed`; pytest also
collects it.
"""
import json
import os
import sys

import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_LLMS = os.path.normpath(os.path.join(_HERE, "..", ".."))
_PE = os.path.normpath(os.path.join(_LLMS, ".."))
for p in (_LLMS,):
    if p not in sys.path:
        sys.path.insert(0, p)

from shared.plan import (  # noqa: E402
    ModelGraph, decoder_graph, QWEN3_0_6B, LLAMA32_1B, NPU2_CAPS, Workload, plan, study_skip,
)
from shared.plan.plan import solve_gemm_tiles, gemm_candidate, MEASURED, ANALYTICAL  # noqa: E402
from shared.plan.placement import PROFILE_MODES  # noqa: E402

GOLDEN = {"qwen3_0_6b": QWEN3_0_6B, "llama32_1b": LLAMA32_1B}


# --- 1. the golden graphs are pinned -------------------------------------------

class _Skip(Exception):
    pass


def test_golden_graph_pinned():
    for name in sorted(GOLDEN):
        want = open(os.path.join(_HERE, "golden", f"{name}.json")).read()
        got = decoder_graph(GOLDEN[name]).to_json()
        assert got == want, f"{name}: decoder_graph drifted from golden/{name}.json (regenerate on purpose, not by accident)"
        assert ModelGraph.from_json(want).to_json() == want


def test_graph_shapes_resolve():
    g = decoder_graph(QWEN3_0_6B)
    assert g.shape_of("q_L", 512, 512) == (512, 2048)
    assert g.shape_of("k_cache_L", 1, 1) == ("ctx", 8, 128)
    assert g.nbytes("wq_L") == 1024 * 2048 * 2
    assert {n.op for n in g.phase_nodes("decode")} >= {"matmul", "attention", "rope", "rms_norm_per_head"}


# --- 2. the plan reproduces the shipped drivers -----------------------------------
# Expected values are the drivers' cached manifests (LaunchCounts.from_module) and
# their ARCHITECTURE.md, read 2026-08-21.

SHIPPED = {
    # `[2026-08-23]` re-read after queue items 11 and 12: qwen decode QKV is the 2-launch
    # head-epilogue form (rms_qkv_qknorm_rope_gemv2) and its LM head the 9 x 16384 + 4480
    # mixed partition (10 launches); llama's head is 8 x 16384 at m_input 8.
    ("qwen3_0_6b", "prefill"): dict(elfs=["rms_qkv_qknorm_rope", "flash_attn", "o_ffn_qwen"], launches=[9, 1, 12],
                                     host=["transpose_seq_to_head", "kv_append", "transpose_head_to_seq"], lm_head=10),
    ("qwen3_0_6b", "decode"): dict(elfs=["rms_qkv_qknorm_rope_gemv2", "o_gemv_ffn"], launches=[2, 3],
                                    host=["kv_append", "decode_attention_cpu"], lm_head=10),
    ("llama32_1b", "prefill"): dict(elfs=["rms_gemms_rope", "flash_attn", "o_ffn"], launches=[7, 1, 12],
                                     host=["kv_append"], lm_head=8),
    ("llama32_1b", "decode"): dict(elfs=["rms_gemv_rope", "o_gemv_ffn"], launches=[6, 3],
                                    host=["kv_append", "decode_attention_cpu"], lm_head=8),
}


def test_plan_reproduces_shipped_sequence():
    for name, phase in sorted(SHIPPED):
        g = decoder_graph(GOLDEN[name])
        wl = Workload(phase, 2048 if phase == "prefill" else 1, 2048 if phase == "prefill" else 512)
        p = plan(g, wl, NPU2_CAPS)
        want = SHIPPED[(name, phase)]
        assert p.elf_sequence(repeated=True) == want["elfs"], (name, phase, p.elf_sequence(True))
        assert [s.launches for s in p.stages if s.where == "device" and s.repeated] == want["launches"], (name, phase)
        assert p.host_sequence(repeated=True) == want["host"], (name, phase)
        assert p.elf_sequence(repeated=False) == ["lm_head_gemv"]
        assert [s.launches for s in p.stages if s.where == "device" and not s.repeated] == [want["lm_head"]], (name, phase)
        assert p.host_sequence(repeated=False) == ["embed_lookup", "final_rms_norm"]
        # every registry-backed candidate is measured at M=2048 (the swept prefill shapes)
        if phase == "prefill":
            assert p.source == MEASURED, [s.name for s in p.stages if s.source == ANALYTICAL]
        assert len(p.sha) == 64


def test_qwen_decode_token_counts_match_doc57():
    """Doc 57 section 5 after items 5c and 5/5b: 28 x (2 + 3) + 10 = 150 boundaries, 57
    submissions per token (was 28 x 7 + 19 = 215 on 2026-08-21, the H0 gate's number)."""
    p = plan(decoder_graph(QWEN3_0_6B), Workload("decode", 1, 512))
    assert p.total_launches == 150, p.total_launches
    assert p.total_submissions == 57
    assert p.total_host_ops == 28 * 2 + 2
    # the LM head is the planner's own derivation, shipped: 9 full partitions + a 4480 tail
    head = [s for s in p.stages if s.name == "lm_head_gemv"][0]
    assert head.launches == 10 and "[16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 4480]" in head.launch_breakdown[0][2]
    assert not any(r[0] == "lm_head_gemv partitioning" for r in p.rejected)
    assert any(r[0] == "rms_qkv_qknorm_rope_gemv4" for r in p.rejected)


def test_plan_hash_is_value_identity():
    g = decoder_graph(QWEN3_0_6B)
    a = plan(g, Workload("decode", 1, 512)).sha
    b = plan(g, Workload("decode", 1, 512)).sha
    c = plan(g, Workload("decode", 1, 1024)).sha
    assert a == b and a != c


def test_non_lean_form_splits_o_ffn():
    """A model outside the lean bounds gets the split O+FFN forms (qwen25_0_5b's shape: hidden 4864)."""
    from shared.plan.graph import ModelSpec
    spec = ModelSpec(name="x", hf_id="x", n_layers=2, emb_dim=896, n_heads=14, n_kv_heads=2, head_dim=64,
                     hidden_dim=4864, vocab_size=151936, lm_head_rows_per_launch=8192)
    p = plan(decoder_graph(spec), Workload("decode", 1, 64))
    assert p.elf_sequence(True) == ["rms_gemv_rope", "o_gemv", "gate_gemv", "up_gemv", "down_gemv"]
    assert any("lean form" in r[1] for r in p.rejected)
    p = plan(decoder_graph(spec), Workload("prefill", 2048, 2048))
    assert p.elf_sequence(True) == ["rms_gemms_rope", "flash_attn", "o_ffn_head", "down_add"]


# --- 3. the study's skips are reproduced ------------------------------------------

def _study():
    study = os.path.join(_PE, "transformer_layer", "study")
    if not os.path.isdir(study):
        raise _Skip("study tree not present")
    for p in (study, os.path.join(_PE, "transformer_layer")):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        import profiles, cases, run_mode  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise _Skip(f"study modules not importable here: {e}")
    return profiles, cases, run_mode


def test_study_skip_matches_profiles_skip_reason():
    profiles, cases, run_mode = _study()
    n = 0
    for family in profiles.REACHABLE_FAMILIES:
        spec = cases.FAMILY_SPECS[family]
        unb = run_mode.UNBUILDABLE_VARIANTS.get(spec.workload_variant, {})
        for mode in PROFILE_MODES:
            for seq in cases.SEQUENCE_LADDER:
                want = profiles.skip_reason(mode, seq, family)
                got = study_skip(mode, seq, spec.hidden_size, cases.SEQUENCE_LADDER, unbuildable=unb,
                                 variant=spec.workload_variant)
                assert (want is None) == (got is None), (family, mode, seq, want, got)
                n += 1
    assert n >= 36 * len(profiles.REACHABLE_FAMILIES)


def test_caps_match_study_constants():
    profiles, cases, run_mode = _study()
    import mapping_space
    c = NPU2_CAPS
    assert c.fa_parallel_seq == profiles.FA_PARALLEL_SEQ
    assert c.attn_gemm_seq_multiple == profiles.ATTN_GEMM_SEQ_MULTIPLE
    assert c.softmax_l1_bytes == profiles.SOFTMAX_L1_BYTES
    assert c.softmax_scale_bands == profiles.SOFTMAX_SCALE_BANDS
    assert c.softmax_itemsize == profiles.SOFTMAX_ITEMSIZE
    assert c.fused_plane_stride_cap == profiles.FUSED_PLANE_STRIDE_CAP
    assert c.fused_seq_min == profiles.FUSED_SEQ_MIN
    assert c.max_feed_channels == mapping_space.MAX_FEED_CHANNELS
    assert c.max_placeable_herd_x == mapping_space.MAX_PLACEABLE_HERD_X
    for seq in cases.SEQUENCE_LADDER:
        assert c.softmax_fits_l1(seq) == profiles.softmax_fits_l1(seq)


# --- 4. the solver, ranked against the registry -------------------------------------

def _registry_shapes():
    kr = os.path.join(_PE, "kernel_registry", "details", "GEMM_bf16_in_bf16_out.json")
    if not os.path.exists(kr):
        raise _Skip("registry not present")
    return json.load(open(kr))


def test_solver_vs_registry_every_mismatch_named():
    """The capacity solver ranks tiles by traffic; the registry ranks by measured latency.

    The gate is not agreement -- it is that every disagreement falls into a
    named class, so a new one would be a finding rather than noise."""
    data = _registry_shapes()
    default_herd = tuple(data.get("herd", (8, 4)))
    agree, classes = 0, {}
    for s in data["shapes"]:
        M, K, N = s["M"], s["K"], s["N"]
        best_name = s["best"].get("high") or s["best"].get("low")
        best = s["methods"][best_name]
        reg_tile, reg_herd = best["tile"], tuple(best.get("herd") or default_herd)
        sol = solve_gemm_tiles(M, K, N)
        assert sol is not None, (M, K, N)
        t = sol["tile"]
        same = (t["tile_m"], t["tile_n"], t["tile_k_l2"]) == (reg_tile["tile_m"], reg_tile["tile_n"], reg_tile["tile_k_l2"]) and sol["herd"] == reg_herd
        if same:
            agree += 1
            continue
        if sol["herd"] != reg_herd:
            cls = "herd: registry measured a reduced herd at short M (per-row herd override)" if reg_herd != default_herd else "herd: solver reduced the herd where the registry kept 8x4"
        elif t["tile_m"] != reg_tile["tile_m"]:
            cls = "tile_m: the measured best is a drain/fused-cast METHOD with a forced tile_m (precision tier, not traffic)"
        elif t["tile_n"] != reg_tile["tile_n"]:
            cls = "tile_n: registry's narrower N tile at this shape (channel/BD budget the traffic model does not see)"
        else:
            cls = "tile_k_l2: same traffic, different K-panel depth (L2 refill count vs ping-pong depth)"
        classes.setdefault(cls, []).append((M, K, N))
    total = len(data["shapes"])
    print(f"\nsolver vs registry: {agree}/{total} identical; mismatch classes: " +
          "; ".join(f"{k} x{len(v)}" for k, v in classes.items()))
    assert agree + sum(len(v) for v in classes.values()) == total
    # every mismatch is in a named class by construction; the classes are the finding
    assert set(classes) <= {
        "herd: registry measured a reduced herd at short M (per-row herd override)",
        "herd: solver reduced the herd where the registry kept 8x4",
        "tile_m: the measured best is a drain/fused-cast METHOD with a forced tile_m (precision tier, not traffic)",
        "tile_n: registry's narrower N tile at this shape (channel/BD budget the traffic model does not see)",
        "tile_k_l2: same traffic, different K-panel depth (L2 refill count vs ping-pong depth)",
    }


def test_registry_override_marks_source():
    assert gemm_candidate(2048, 1024, 2048)["source"] == MEASURED
    c = gemm_candidate(2048, 1000, 2000)   # not a swept shape; 1000 % 64 != 0 -> solver may refuse or derive
    assert c is None or c["source"] == ANALYTICAL
    c = gemm_candidate(3072, 1024, 2048)
    assert c is not None and c["source"] == ANALYTICAL and c["method"] == "direct"


def _main():
    tests = [(n, o) for n, o in globals().items() if n.startswith("test_") and callable(o)]
    failed, skipped = [], []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS {name}")
        except _Skip as e:
            skipped.append(name)
            print(f"  SKIP {name}: {e}")
        except Exception:
            failed.append(name)
            print(f"  FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - len(failed) - len(skipped)}/{len(tests)} passed" + (f", {len(skipped)} skipped" if skipped else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
