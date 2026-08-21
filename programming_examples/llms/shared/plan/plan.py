# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""`plan(graph, workload, caps) -> Plan` (doc 56 section 3.3).

Stages, each pure and testable alone:
  1. candidates  -- per matmul: the registry's measured config where the exact
                    shape exists, else the capacity solver's pick marked
                    `analytical_unmeasured`;
  2. placement   -- placement.place;
  3. fusion      -- consecutive device ops grouped into the builder patterns the
                    drivers actually ship, with their launch counts DERIVED
                    (a GEMM is one launch plus one cast launch when its measured
                    method is fused-cast; a GEMV's rows per launch are capped by
                    the BD repeat count; the decode O+FFN cascade is three);
  4. dispatch    -- launches grouped into submissions (one xrt.run per ELF today,
                    split at every host op) and costed with doc 57's constants.
The Plan carries every stage's output, each rejected alternative with its
reason, the prediction source per number, and a SHA-256 over all of it that is
the artifact cache key.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, asdict

from .caps import DeviceCaps, NPU2_CAPS
from .placement import Workload, place, DEVICE, HOST, REFUSE

MEASURED, ANALYTICAL, FORCED = "measured", "analytical_unmeasured", "forced"


# ---------------------------------------------------------------------------
# 1. candidates: the GEMM registry and the capacity solver
# ---------------------------------------------------------------------------

def _registry_lookup(M, K, N):
    """kernel_registry.gemm_config or None when the shape is unmeasured / registry absent."""
    try:
        import sys, os
        here = os.path.dirname(os.path.abspath(__file__))
        kr = os.path.normpath(os.path.join(here, "..", "..", "..", "kernel_registry"))
        if kr not in sys.path:
            sys.path.insert(0, kr)
        from registry_lookup import gemm_config
    except Exception:
        return None
    try:
        return gemm_config(M, K, N)
    except KeyError:
        return None


TILE_M_CHOICES = (32, 64)
TILE_N_CHOICES = (32, 64, 96, 128)
TILE_K_L2_CHOICES = (64, 128, 256, 512)
TILE_K_L1 = 32


def solve_gemm_tiles(M, K, N, caps=NPU2_CAPS, out_itemsize=4):
    """Capacity solver (`compute_chunks` analog): minimize reload traffic under spatial legality.

    Traffic = weight-panel reloads ceil(M / (tile_m*herd_m)) * K*N*2 + activation
    reloads ceil(N / (tile_n*herd_n)) * M*K*2. Legality: M % (tile_m*herd_m) == 0,
    N % (tile_n*herd_n) == 0, K % tile_k_l2 == 0, per-column L2 holds the A and B
    panels double-buffered plus the C tile, L1 holds the A/B L1 tiles ping-pong
    plus C. Ties break toward the registry's measured preference: a 256-deep K
    panel (the registry's modal tile_k_l2 -- deeper panels buy nothing once the
    L2 ping-pong hides the refill), then the wider N tile, then the wider M
    tile. Returns {"tile", "herd", "traffic_bytes", "source"}.
    """
    best = None
    for herd_m in (8, 4, 2, 1):
        for herd_n in (4, 2, 1):
            for tm in TILE_M_CHOICES:
                if M % (tm * herd_m):
                    continue
                for tn in TILE_N_CHOICES:
                    if N % (tn * herd_n):
                        continue
                    for tk2 in TILE_K_L2_CHOICES:
                        if K % tk2 or tk2 % TILE_K_L1:
                            continue
                        l2 = 2 * (tm * tk2 * 2) + 2 * (tk2 * tn * 2) + tm * tn * out_itemsize
                        l1 = 2 * (tm * TILE_K_L1 * 2) + 2 * (TILE_K_L1 * tn * 2) + tm * tn * out_itemsize
                        if l2 > caps.l2_bytes_per_column or l1 > caps.l1_bytes:
                            continue
                        traffic = (math.ceil(M / (tm * herd_m)) * K * N * 2
                                   + math.ceil(N / (tn * herd_n)) * M * K * 2)
                        key = (traffic, -(herd_m * herd_n), abs(tk2 - 256), -tn, -tm)
                        if best is None or key < best[0]:
                            best = (key, dict(tile_m=tm, tile_k_l2=tk2, tile_k_l1=TILE_K_L1, tile_n=tn),
                                    (herd_m, herd_n), traffic)
    if best is None:
        return None
    return {"tile": best[1], "herd": best[2], "traffic_bytes": best[3], "source": ANALYTICAL}


def gemm_candidate(M, K, N, caps=NPU2_CAPS):
    reg = _registry_lookup(M, K, N)
    if reg is not None:
        return {"tile": reg["tile"], "herd": tuple(reg["herd"]), "method": reg["method"],
                "gflops": reg["gflops"], "source": MEASURED}
    sol = solve_gemm_tiles(M, K, N, caps)
    if sol is None:
        return None
    sol["method"] = "direct"   # an unmeasured shape gets the simplest form; policy must opt in
    sol["gflops"] = None
    return sol


# ---------------------------------------------------------------------------
# 3. fusion -- the builder patterns the drivers ship, with derived launch counts
# ---------------------------------------------------------------------------

@dataclass
class Stage:
    name: str                      # builder pattern / ELF name
    where: str                     # device | host
    ops: tuple                     # node ids (template ids, "L" = layer)
    launches: int = 0              # air.launch count (device) -- derived
    launch_breakdown: tuple = ()   # ((op, launches, why), ...)
    repeated: bool = True          # once per layer
    weight_bytes: int = 0
    boundary_bytes: int = 0        # host<->device bytes per instance
    candidates: dict = field(default_factory=dict)   # matmul op -> candidate
    source: str = MEASURED
    note: str = ""


def _lean_form(spec, caps):
    return spec.emb_dim < caps.lean_form_emb_max and spec.hidden_dim % caps.lean_form_hidden_multiple == 0


def _gemm_launches(cand):
    return 2 if (cand and cand.get("method") == "fused-cast") else 1


def _gemv_partitions(rows, caps, herd_m=8, m_input=4, tile_m=8):
    """(launch count, partition rows) for a GEMV of `rows`: full partitions at the BD
    repeat cap plus ONE tail partition sized to the remainder (rounded up to the
    tile grid) -- stitching accepts launches of different shapes, so padding the
    vocab to a whole number of full partitions is never required."""
    cap = (caps.bd_repeat_cap + 1) * herd_m * m_input
    grid = tile_m * herd_m
    full, rem = divmod(rows, cap)
    parts = [cap] * full + ([math.ceil(rem / grid) * grid] if rem else [])
    return len(parts), tuple(parts)


def fuse(graph, wl, placements, caps=NPU2_CAPS):
    """Group the phase's ops into stages in execution order, deriving launch counts."""
    spec = graph.spec
    g = graph
    stages, rejected = [], []
    lean = _lean_form(spec, caps)
    Mq = wl.M

    def wbytes(*tids):
        return sum(g.nbytes(t, Mq, wl.kv_len) for t in tids)

    def gemm(op, w):
        Min, Nout = g.shape_of(w)
        return gemm_candidate(Mq, Min, Nout, caps)

    # embed (host)
    stages.append(Stage("embed_lookup", HOST, ("embed",), repeated=False,
                        boundary_bytes=g.nbytes("x_0", Mq), note="host table lookup, x_0 uploaded"))

    qk = spec.qk_norm
    if wl.phase == "prefill":
        # --- QKV stage ---
        cands = {op: gemm(op, w) for op, w in (("q_proj_L", "wq_L"), ("k_proj_L", "wk_L"), ("v_proj_L", "wv_L"))}
        ops = ["attn_norm_L", "q_proj_L", "k_proj_L", "v_proj_L"] + (["q_norm_L", "k_norm_L"] if qk else []) + ["rope_q_L", "rope_k_L"]
        bd = [("attn_norm_L", 1, "RMSNorm launch")]
        for op in ("q_proj_L", "k_proj_L", "v_proj_L"):
            n = _gemm_launches(cands[op]); bd.append((op, n, f"GEMM {cands[op]['method'] if cands[op] else '?'}" + (" + cast launch" if n == 2 else "")))
        if qk:
            bd += [("q_norm_L", 1, "per-head RMSNorm"), ("k_norm_L", 1, "per-head RMSNorm")]
        bd += [("rope_q_L", 1, "RoPE"), ("rope_k_L", 1, "RoPE")]
        stages.append(Stage("rms_qkv_qknorm_rope" if qk else "rms_gemms_rope", DEVICE, tuple(ops),
                            launches=sum(b[1] for b in bd), launch_breakdown=tuple(bd),
                            weight_bytes=wbytes("attn_norm_w_L", "wq_L", "wk_L", "wv_L") + (wbytes("q_norm_w_L", "k_norm_w_L") if qk else 0),
                            candidates=cands, source=MEASURED if all(c and c["source"] == MEASURED for c in cands.values()) else ANALYTICAL))
        # --- attention ---
        pa = placements["attention_L"]
        if pa.where == REFUSE:
            rejected.append(("attention_L", pa.reason))
        for hop in pa.extra_host_ops[:1]:
            stages.append(Stage(hop, HOST, ("attention_L",), boundary_bytes=g.nbytes("q_roped_L", Mq) + 2 * g.nbytes("k_roped_L", Mq),
                                note=pa.reason))
        stages.append(Stage("kv_append", HOST, ("kv_append_L",), boundary_bytes=g.nbytes("k_roped_L", Mq) + g.nbytes("v_L", Mq),
                            note=placements["kv_append_L"].reason))
        stages.append(Stage("flash_attn", pa.where, ("attention_L",), launches=1,
                            launch_breakdown=(("attention_L", 1, "FlashAttention, one launch"),), note=pa.reason))
        for hop in pa.extra_host_ops[1:]:
            stages.append(Stage(hop, HOST, ("attention_L",), boundary_bytes=g.nbytes("attn_out_L", Mq), note=pa.reason))
        # --- O + FFN ---
        cands = {op: gemm(op, w) for op, w in (("o_proj_L", "wo_L"), ("gate_proj_L", "w_gate_L"),
                                                ("up_proj_L", "w_up_L"), ("down_proj_L", "w_down_L"))}
        def gl(op, what):
            n = _gemm_launches(cands[op]); return (op, n, f"{what} GEMM {cands[op]['method'] if cands[op] else '?'}" + (" + cast" if n == 2 else ""))
        if lean:
            bd = [gl("o_proj_L", "O"), ("residual_1_L", 1, "add"), ("ffn_norm_L", 1, "RMSNorm"), gl("gate_proj_L", "gate"),
                  gl("up_proj_L", "up"), ("swiglu_L", 1, "SwiGLU"), gl("down_proj_L", "down"), ("residual_2_L", 1, "add")]
            stages.append(Stage("o_ffn_qwen" if qk else "o_ffn", DEVICE,
                                ("o_proj_L", "residual_1_L", "ffn_norm_L", "gate_proj_L", "up_proj_L", "swiglu_L", "down_proj_L", "residual_2_L"),
                                launches=sum(b[1] for b in bd), launch_breakdown=tuple(bd),
                                weight_bytes=wbytes("wo_L", "ffn_norm_w_L", "w_gate_L", "w_up_L", "w_down_L"), candidates=cands,
                                note=f"lean fused O+FFN: emb {spec.emb_dim} < {caps.lean_form_emb_max} and hidden {spec.hidden_dim} % {caps.lean_form_hidden_multiple} == 0"))
        else:
            bd1 = [gl("o_proj_L", "O"), ("residual_1_L", 1, "add"), ("ffn_norm_L", 1, "RMSNorm"), gl("gate_proj_L", "gate"), gl("up_proj_L", "up"), ("swiglu_L", 1, "SwiGLU")]
            stages.append(Stage("o_ffn_head", DEVICE, ("o_proj_L", "residual_1_L", "ffn_norm_L", "gate_proj_L", "up_proj_L", "swiglu_L"),
                                launches=sum(b[1] for b in bd1), launch_breakdown=tuple(bd1),
                                weight_bytes=wbytes("wo_L", "ffn_norm_w_L", "w_gate_L", "w_up_L"), candidates={k: cands[k] for k in ("o_proj_L", "gate_proj_L", "up_proj_L")},
                                note=f"split O+FFN: emb {spec.emb_dim} / hidden {spec.hidden_dim} outside the lean form's bounds"))
            bd2 = [gl("down_proj_L", "down"), ("residual_2_L", 1, "add")]
            stages.append(Stage("down_add", DEVICE, ("down_proj_L", "residual_2_L"), launches=sum(b[1] for b in bd2), launch_breakdown=tuple(bd2),
                                weight_bytes=wbytes("w_down_L"), candidates={"down_proj_L": cands["down_proj_L"]}))
            rejected.append(("o_ffn fused", "spatial: the fused cascade needs the lean form"))
    else:  # decode
        if qk:
            bd = (("attn_norm_L", 1, "RMSNorm"), ("qkv_proj", 1, "ONE GEMV over [wq; wk; wv]"),
                  ("qk_norm", 1, "ONE per-row-weighted QK-norm over Q|K"), ("rope", 1, "ONE RoPE over Q|K"))
            name = "rms_qkv_qknorm_rope_gemv4"
        else:
            bd = (("attn_norm_L", 1, "RMSNorm"), ("q_proj_L", 1, "GEMV"), ("k_proj_L", 1, "GEMV"), ("v_proj_L", 1, "GEMV"),
                  ("rope_q_L", 1, "RoPE"), ("rope_k_L", 1, "RoPE"))
            name = "rms_gemv_rope"
        ops = ["attn_norm_L", "q_proj_L", "k_proj_L", "v_proj_L"] + (["q_norm_L", "k_norm_L"] if qk else []) + ["rope_q_L", "rope_k_L"]
        stages.append(Stage(name, DEVICE, tuple(ops), launches=sum(b[1] for b in bd), launch_breakdown=bd,
                            weight_bytes=wbytes("attn_norm_w_L", "wq_L", "wk_L", "wv_L") + (wbytes("q_norm_w_L", "k_norm_w_L") if qk else 0),
                            boundary_bytes=g.nbytes("x_in_L") + (spec.n_heads + spec.n_kv_heads) * spec.head_dim * 2))  # x_in + the position LUT
        stages.append(Stage("kv_append", HOST, ("kv_append_L",), boundary_bytes=g.nbytes("k_roped_L") + g.nbytes("v_L") + g.nbytes("q_roped_L"),
                            note=placements["kv_append_L"].reason))
        stages.append(Stage("decode_attention_cpu", HOST, ("attention_L",), note=placements["attention_L"].reason,
                            boundary_bytes=g.nbytes("attn_out_L")))
        if lean:
            bd = (("o_proj_L+residual_1_L", 1, "matvec_2tile_add"), ("ffn_norm_L+gate_proj_L+up_proj_L+swiglu_L", 1, "matvec_swiglu_rms cascade"),
                  ("down_proj_L+residual_2_L", 1, "matvec_2tile_add"))
            stages.append(Stage("o_gemv_ffn", DEVICE, ("o_proj_L", "residual_1_L", "ffn_norm_L", "gate_proj_L", "up_proj_L", "swiglu_L", "down_proj_L", "residual_2_L"),
                                launches=3, launch_breakdown=bd, weight_bytes=wbytes("wo_L", "ffn_norm_w_L", "w_gate_L", "w_up_L", "w_down_L"),
                                boundary_bytes=g.nbytes("x_out_L"), note="lean fused O+FFN cascade (3 launches)"))
        else:
            for nm, ops_, ws in (("o_gemv", ("o_proj_L", "residual_1_L"), ("wo_L",)), ("gate_gemv", ("ffn_norm_L", "gate_proj_L"), ("ffn_norm_w_L", "w_gate_L")),
                                 ("up_gemv", ("up_proj_L",), ("w_up_L",)), ("down_gemv", ("swiglu_L", "down_proj_L", "residual_2_L"), ("w_down_L",))):
                stages.append(Stage(nm, DEVICE, ops_, launches=1, launch_breakdown=((nm, 1, "GEMV"),), weight_bytes=wbytes(*ws)))
            rejected.append(("o_gemv_ffn fused", "spatial: the cascade needs the lean form"))
    # --- once: final norm (host) + LM head ---
    stages.append(Stage("final_rms_norm", HOST, ("final_norm",), repeated=False, note=placements["final_norm"].reason,
                        boundary_bytes=g.nbytes("x_final_normed")))
    # LM head: the shipped partitioning is a driver fact (spec.lm_head_rows_per_launch); the
    # BD-repeat derivation is reported beside it, as an alternative when it differs.
    derived_count, derived_parts = _gemv_partitions(spec.vocab_size, caps, herd_m=8, m_input=8)
    if spec.lm_head_rows_per_launch:
        n_part = spec.lm_head_rows_per_launch
        n_part_count = math.ceil(spec.vocab_size / n_part)
        parts = (n_part,) * n_part_count
        why = f"driver pins {n_part} rows per launch"
        if derived_count != n_part_count or sum(derived_parts) != sum(parts):
            saved_b = n_part_count - derived_count
            saved_rows = sum(parts) - sum(derived_parts)
            rejected.append(("lm_head_gemv partitioning",
                             f"derived: {derived_count} launches {list(derived_parts)} fit the BD repeat cap "
                             f"({caps.bd_repeat_cap}) at m_input 8 -- {saved_b} boundaries fewer "
                             f"(~{saved_b * caps.launch_boundary_us / 1e3:.1f} ms/token) and {saved_rows} fewer pad rows "
                             f"({saved_rows * spec.emb_dim * 2 / 1e6:.1f} MB, ~{saved_rows * spec.emb_dim * 2 / (caps.gemv_stream_gbs * 1e6):.2f} ms) "
                             f"than the shipped {n_part_count} x {n_part}; untested, an O3 knob"))
    else:
        n_part_count, parts = derived_count, derived_parts
        why = f"BD repeat cap {caps.bd_repeat_cap} x herd 8 x m_input 8, tail partition on the tile grid"
    rows_total = sum(parts)
    stages.append(Stage("lm_head_gemv", DEVICE, ("lm_head",), launches=n_part_count, repeated=False,
                        launch_breakdown=(("lm_head", n_part_count, f"GEMV partitions {list(parts)}: {why}"),),
                        weight_bytes=rows_total * spec.emb_dim * 2, boundary_bytes=rows_total * 2,
                        note=f"{rows_total - spec.vocab_size} pad rows; logits truncated on the host"))
    return stages, rejected


# ---------------------------------------------------------------------------
# 4. dispatch + cost, and the Plan
# ---------------------------------------------------------------------------

@dataclass
class Plan:
    spec_name: str
    workload: dict
    caps: dict
    placements: dict
    stages: list
    rejected: list
    per_layer_launches: int = 0
    per_layer_submissions: int = 0
    per_layer_host_ops: int = 0
    total_launches: int = 0
    total_submissions: int = 0
    total_host_ops: int = 0
    resident_weight_bytes: int = 0
    boundary_bytes: int = 0
    est_us: float = 0.0
    est_breakdown: dict = field(default_factory=dict)
    source: str = MEASURED
    planner_version: str = "h0.1"
    sha: str = ""

    def elf_sequence(self, repeated=None):
        return [s.name for s in self.stages if s.where == DEVICE and (repeated is None or s.repeated == repeated)]

    def host_sequence(self, repeated=None):
        return [s.name for s in self.stages if s.where == HOST and (repeated is None or s.repeated == repeated)]

    def to_json(self):
        d = asdict(self)
        return json.dumps(d, indent=1, default=str, sort_keys=True)


def plan(graph, wl, caps=NPU2_CAPS):
    spec = graph.spec
    placements = place(graph, wl, caps)
    stages, rejected = fuse(graph, wl, placements, caps)
    L = spec.n_layers
    per_layer = [s for s in stages if s.repeated]
    once = [s for s in stages if not s.repeated]
    dev = lambda ss: [s for s in ss if s.where == DEVICE]
    hst = lambda ss: [s for s in ss if s.where == HOST]
    p = Plan(spec.name, asdict(wl), caps.as_dict(), {k: asdict(v) for k, v in placements.items()}, stages, rejected)
    p.per_layer_launches = sum(s.launches for s in dev(per_layer))
    p.per_layer_submissions = len(dev(per_layer))         # one xrt.run per ELF, split at host ops
    p.per_layer_host_ops = len(hst(per_layer))
    p.total_launches = L * p.per_layer_launches + sum(s.launches for s in dev(once))
    p.total_submissions = L * p.per_layer_submissions + len(dev(once))
    p.total_host_ops = L * p.per_layer_host_ops + len(hst(once))
    p.resident_weight_bytes = L * sum(s.weight_bytes for s in per_layer) + sum(s.weight_bytes for s in once)
    p.boundary_bytes = L * sum(s.boundary_bytes for s in per_layer) + sum(s.boundary_bytes for s in once)
    # Cost: decode is weight-stream + boundaries + submissions (doc 57); prefill uses the
    # registry's gflops where measured. Analytical everywhere -- the prediction, never the result.
    bnd = p.total_launches * caps.launch_boundary_us
    sub = p.total_submissions * caps.run_fixed_us
    if wl.phase == "decode":
        stream = p.resident_weight_bytes / (caps.gemv_stream_gbs * 1e3)   # bytes / (GB/s) -> us
        p.est_breakdown = {"boundaries_us": bnd, "submissions_us": sub, "weight_stream_us": stream}
    else:
        gemm_us = 0.0
        for s in stages:
            for op, c in s.candidates.items():
                if c and c.get("gflops"):
                    w = graph.tensors[graph.nodes[[n.id for n in graph.nodes].index(op)].inputs[1]]
                    K_, N_ = graph.shape_of(w.id)
                    gemm_us += (2.0 * wl.M * K_ * N_) / (c["gflops"] * 1e3) * (L if s.repeated else 1)
        p.est_breakdown = {"boundaries_us": bnd, "submissions_us": sub, "gemm_us_from_registry": gemm_us}
    p.est_us = sum(p.est_breakdown.values())
    p.source = ANALYTICAL if any(s.source == ANALYTICAL for s in stages) else MEASURED
    body = json.dumps({"spec": asdict(spec), "workload": p.workload, "caps": p.caps, "placements": p.placements,
                       "stages": [asdict(s) for s in stages], "planner_version": p.planner_version},
                      sort_keys=True, default=str)
    p.sha = hashlib.sha256(body.encode()).hexdigest()
    return p
