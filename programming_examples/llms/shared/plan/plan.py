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


def _registry_lookup_method(M, K, N, method):
    """A SPECIFIC registry method's measured row, or None."""
    try:
        from registry_lookup import gemm_config_method
        return gemm_config_method(M, K, N, "bf16", method)
    except Exception:
        return None


def gemm_candidate(M, K, N, caps=NPU2_CAPS, method=None):
    """`method` `[2026-08-23]` FORCES a registry method (its measured row for this
    shape) over the tier's best: source `forced`, so a plan built for an
    artifact whose builder supports one form only (o_ffn is fused-cast-only)
    hashes as that plan and derives that artifact's launch count."""
    if method is not None:
        reg = _registry_lookup_method(M, K, N, method)
        if reg is None:
            return None
        return {"tile": reg["tile"], "herd": tuple(reg["herd"]), "method": reg["method"],
                "gflops": reg["gflops"], "source": FORCED}
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


#: `[2026-08-26]` doc 56 H2a: the w4_decode GEMV quantization contract's NAME.
#: The one OWNER of the contract is the packing code
#: (`llama32_1b_int4/awq_repacker.quant_contract`), which the study's quant_*
#: columns read; this literal mirrors its `quant_gemv_contract_name` so the
#: plan's stages name the contract without importing the model dir (the plan
#: package stays dependency-free -- the fa_cache_name pattern; a host test
#: pins the agreement).
W4_GEMV_CONTRACT = "awq_u4_asym_g128_bf16s_u8z_dequant_in_kernel"
W4_BYTES_NOTE = ("weight_bytes are int4 nibbles (bf16 // 4); per-group bf16 scales + u8 zeros "
                 "ride on top (~+4.7 %); the exact packed BO bytes are the driver's")


def _w4_bytes(bf16_bytes):
    return bf16_bytes // 4


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


def fuse(graph, wl, placements, caps=NPU2_CAPS, forced=None):
    """Group the phase's ops into stages in execution order, deriving launch counts.

    `forced` `[2026-08-23]`: {stage name: GEMM method} -- every matmul of that
    stage takes the named registry method (source `forced`). The deviation the
    kernel-scaling curve needed at Qwen3-0.6B M=1024 (`o_ffn_qwen` fused-cast
    where the registry best is drain) is expressed HERE, in the plan, so the
    plan's hash and launch count describe the artifact that was built."""
    spec = graph.spec
    g = graph
    stages, rejected = [], []
    lean = _lean_form(spec, caps)
    Mq = wl.M
    forced = dict(forced or {})

    def wbytes(*tids):
        return sum(g.nbytes(t, Mq, wl.kv_len) for t in tids)

    def gemm(op, w, stage=None):
        Min, Nout = g.shape_of(w)
        return gemm_candidate(Mq, Min, Nout, caps, method=forced.get(stage))

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
        # `[2026-08-25]` doc 56 H1b: kv_len > M is a chunked-prefill chunk whose
        # queries attend to the whole context -- a RECTANGULAR FA artifact,
        # named per context length. Mirrors fa_headfirst.fa_cache_name (the
        # plan package stays dependency-free; a host test pins the agreement).
        fa_name = "flash_attn" if wl.kv_len == Mq else f"flash_attn_ctx{wl.kv_len}"
        stages.append(Stage(fa_name, pa.where, ("attention_L",), launches=1,
                            launch_breakdown=(("attention_L", 1, "FlashAttention, one launch"),),
                            note=pa.reason + ("" if wl.kv_len == Mq else f"; rectangular (Lq={Mq}, Lk={wl.kv_len}), causal base {wl.kv_len - Mq}")))
        for hop in pa.extra_host_ops[1:]:
            stages.append(Stage(hop, HOST, ("attention_L",), boundary_bytes=g.nbytes("attn_out_L", Mq), note=pa.reason))
        # --- O + FFN ---
        o_ffn_stage = ("o_ffn_qwen" if qk else "o_ffn") if lean else "o_ffn_head"
        cands = {op: gemm(op, w, o_ffn_stage) for op, w in (("o_proj_L", "wo_L"), ("gate_proj_L", "w_gate_L"),
                                                            ("up_proj_L", "w_up_L"), ("down_proj_L", "w_down_L"))}
        if o_ffn_stage in forced:
            for op, c in cands.items():
                if c is None:
                    rejected.append((op, f"forced method {forced[o_ffn_stage]!r} has no registry row at M={Mq}"))
        def gl(op, what):
            n = _gemm_launches(cands[op]); return (op, n, f"{what} GEMM {cands[op]['method'] if cands[op] else '?'}" + (" + cast" if n == 2 else ""))
        if lean:
            bd = [gl("o_proj_L", "O"), ("residual_1_L", 1, "add"), ("ffn_norm_L", 1, "RMSNorm"), gl("gate_proj_L", "gate"),
                  gl("up_proj_L", "up"), ("swiglu_L", 1, "SwiGLU"), gl("down_proj_L", "down"), ("residual_2_L", 1, "add")]
            stages.append(Stage("o_ffn_qwen" if qk else "o_ffn", DEVICE,
                                ("o_proj_L", "residual_1_L", "ffn_norm_L", "gate_proj_L", "up_proj_L", "swiglu_L", "down_proj_L", "residual_2_L"),
                                launches=sum(b[1] for b in bd), launch_breakdown=tuple(bd),
                                weight_bytes=wbytes("wo_L", "ffn_norm_w_L", "w_gate_L", "w_up_L", "w_down_L"), candidates=cands,
                                source=FORCED if o_ffn_stage in forced else MEASURED,
                                note=f"lean fused O+FFN: emb {spec.emb_dim} < {caps.lean_form_emb_max} and hidden {spec.hidden_dim} % {caps.lean_form_hidden_multiple} == 0"
                                + (f"; GEMM method FORCED to {forced[o_ffn_stage]!r} (the cascade builder's only form)" if o_ffn_stage in forced else "")))
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
        w4 = wl.precision_plan == "w4_decode"
        if w4 and not lean:
            # `[2026-08-26]` doc 56 H2b (queue item 18) lifted the qk-norm
            # refusal: the int4 GEMV family covers the lean qk-norm form
            # (Qwen3-0.6B: o_gemv_ffn_int4 at q_dim/k_chunk=emb). The
            # NON-lean form still refuses -- no int4 split-form driver
            # exists, and refusing beats naming bf16 stages a w4 plan.
            raise ValueError(
                f"w4_decode decode plan exists only for the lean fused form "
                f"(llama32_1b_int4's 3-launch cascade; qwen3_0_6b's decoupled sibling, doc 56 H2b); "
                f"{spec.name} is outside the lean bounds")
        if qk:
            # `[2026-08-23]` doc 57 section 5 item 5c (queue item 11): ONE head-aligned GEMV whose
            # cores apply QK-norm + RoPE in L1 (mv_heads.cc), so the stage is RMSNorm + GEMV.
            bd = (("attn_norm_L", 1, "RMSNorm"),
                  ("qkv_proj+qk_norm+rope", 1, "ONE head-aligned GEMV over [wq; wk; wv] with the QK-norm + RoPE epilogue in L1 (mv_heads)"))
            name = "rms_qkv_qknorm_rope_gemv2"
            rejected.append(("rms_qkv_qknorm_rope_gemv4", "4 launches (GEMV, QK-norm, RoPE as separate launches): superseded by the "
                             "head-epilogue GEMV, 0.672 -> 0.494 ms per layer (devq 555); kept behind QWEN3_RMS_QKV_LAUNCHES=4 for A/B"))
            if w4:
                # `[2026-08-26]` doc 56 H2b (queue item 18): under w4_decode
                # the qk-norm QKV stage STAYS bf16 -- both int4 forms priced
                # negative (results/item18-h2b-20260826/PREDICTION.md s2.B):
                rejected.append(("rms_qkv int4 (launch-structure fallback)",
                                 "int4 QKV without the in-core epilogue needs RMSNorm + int4 GEMV + QK-norm + RoPE "
                                 "launches: +2 boundaries = +0.214 ms/layer against a stream ceiling of 0.152 ms/layer "
                                 "(6.19 MB/layer at the boundary-free 40.8 GB/s) -- the measured 107 us boundary "
                                 "(devq 450) beats the byte ceiling at ANY dequant speed; priced negative, doc 56 H2b"))
                rejected.append(("rms_qkv int4 (in-core epilogue)",
                                 "an int4 mv_heads (dequant + persistent-head accumulate + QK-norm/RoPE epilogue) is a "
                                 "NEW kernel family, ceiling <= 2.3-4.5 ms/token at the measured 11-19 GB/s int4 rates, "
                                 "with Q/K quantization error injected directly into attention; H2b's deliverable is "
                                 "over the EXISTING int4 builders -- deferred, priced in PREDICTION.md s2.B1"))
                rejected.append(("lm_head int4",
                                 "doc 57 s5 item 6 (O4) measured the ten-launch int4 head on this model: -0.46 ms/token "
                                 "ceiling (devq 488, 11 GB/s dequant-bound), the one-launch form cannot compile past 2 "
                                 "iterations (push_queue repeat cap, devq 468), and the top-5 accuracy question is "
                                 "deliberately unasked until a faster int4 GEMV exists; head stays bf16"))
        elif w4:
            # `[2026-08-26]` doc 56 H2a (queue item 17): the SHIPPED int4 decode
            # stage -- same 6-launch shape as rms_gemv_rope, the three GEMVs
            # dequanting in-kernel. The contract NAME mirrors the packing code's
            # (llama32_1b_int4/awq_repacker.quant_contract; the plan package
            # stays dependency-free, a host test pins the agreement -- the
            # fa_cache_name pattern).
            gemv = f"int4 GEMV ({W4_GEMV_CONTRACT})"
            bd = (("attn_norm_L", 1, "RMSNorm"), ("q_proj_L", 1, gemv), ("k_proj_L", 1, gemv), ("v_proj_L", 1, gemv),
                  ("rope_q_L", 1, "RoPE"), ("rope_k_L", 1, "RoPE"))
            name = "rms_qkv_int4_rope"
        else:
            bd = (("attn_norm_L", 1, "RMSNorm"), ("q_proj_L", 1, "GEMV"), ("k_proj_L", 1, "GEMV"), ("v_proj_L", 1, "GEMV"),
                  ("rope_q_L", 1, "RoPE"), ("rope_k_L", 1, "RoPE"))
            name = "rms_gemv_rope"
        ops = ["attn_norm_L", "q_proj_L", "k_proj_L", "v_proj_L"] + (["q_norm_L", "k_norm_L"] if qk else []) + ["rope_q_L", "rope_k_L"]
        # `[2026-08-26]` doc 56 H2b: under w4_decode a qk-norm model's QKV
        # weights STAY bf16 (both int4 QKV forms priced negative, see the
        # rejected entries above) -- only the non-qk llama form is int4 here.
        qkv_w4 = w4 and not qk
        qkv_wb = wbytes("attn_norm_w_L") + (_w4_bytes(wbytes("wq_L", "wk_L", "wv_L")) if qkv_w4 else wbytes("wq_L", "wk_L", "wv_L")) \
            + (wbytes("q_norm_w_L", "k_norm_w_L") if qk else 0)
        stages.append(Stage(name, DEVICE, tuple(ops), launches=sum(b[1] for b in bd), launch_breakdown=bd,
                            weight_bytes=qkv_wb,
                            boundary_bytes=g.nbytes("x_in_L") + (spec.n_heads + spec.n_kv_heads) * spec.head_dim * 2,  # x_in + the position LUT
                            note=W4_BYTES_NOTE if qkv_w4 else
                            ("w4_decode: QKV stays bf16 (int4 QKV priced negative, doc 56 H2b)" if w4 else "")))
        stages.append(Stage("kv_append", HOST, ("kv_append_L",), boundary_bytes=g.nbytes("k_roped_L") + g.nbytes("v_L") + g.nbytes("q_roped_L"),
                            note=placements["kv_append_L"].reason))
        stages.append(Stage("decode_attention_cpu", HOST, ("attention_L",), note=placements["attention_L"].reason,
                            boundary_bytes=g.nbytes("attn_out_L")))
        if lean and w4:
            bd = (("o_proj_L+residual_1_L", 1, f"int4 matvec_add ({W4_GEMV_CONTRACT})"),
                  ("ffn_norm_L+gate_proj_L+up_proj_L+swiglu_L", 1, f"int4 matvec_swiglu_rms cascade, gate/up nibble-interleaved ({W4_GEMV_CONTRACT})"),
                  ("down_proj_L+residual_2_L", 1, f"int4 matvec_add ({W4_GEMV_CONTRACT})"))
            stages.append(Stage("o_gemv_ffn_int4", DEVICE, ("o_proj_L", "residual_1_L", "ffn_norm_L", "gate_proj_L", "up_proj_L", "swiglu_L", "down_proj_L", "residual_2_L"),
                                launches=3, launch_breakdown=bd,
                                weight_bytes=wbytes("ffn_norm_w_L") + _w4_bytes(wbytes("wo_L", "w_gate_L", "w_up_L", "w_down_L")),
                                boundary_bytes=g.nbytes("x_out_L"),
                                note="lean fused int4 O+FFN cascade (3 launches); " + W4_BYTES_NOTE
                                + ("; O GEMV decoupled (K=q_dim), k_chunk=emb -- doc 56 H2b" if qk else "")))
        elif lean:
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
    #: h0.1 -> h0.2 `[2026-08-23]`: decode QKV 2-launch form, derived Qwen LM head;
    #: h0.2 -> h0.3: `forced` GEMM methods are part of the plan and its hash.
    planner_version: str = "h0.3"
    forced: dict = field(default_factory=dict)
    sha: str = ""

    def elf_sequence(self, repeated=None):
        return [s.name for s in self.stages if s.where == DEVICE and (repeated is None or s.repeated == repeated)]

    def host_sequence(self, repeated=None):
        return [s.name for s in self.stages if s.where == HOST and (repeated is None or s.repeated == repeated)]

    def to_json(self):
        d = asdict(self)
        return json.dumps(d, indent=1, default=str, sort_keys=True)


def plan(graph, wl, caps=NPU2_CAPS, forced=None):
    spec = graph.spec
    placements = place(graph, wl, caps)
    forced = dict(forced or {})
    stages, rejected = fuse(graph, wl, placements, caps, forced=forced)
    L = spec.n_layers
    per_layer = [s for s in stages if s.repeated]
    once = [s for s in stages if not s.repeated]
    dev = lambda ss: [s for s in ss if s.where == DEVICE]
    hst = lambda ss: [s for s in ss if s.where == HOST]
    p = Plan(spec.name, asdict(wl), caps.as_dict(), {k: asdict(v) for k, v in placements.items()}, stages, rejected, forced=forced)
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
    p.source = (FORCED if forced else ANALYTICAL) if any(s.source != MEASURED for s in stages) else MEASURED
    body = json.dumps({"spec": asdict(spec), "workload": p.workload, "caps": p.caps, "placements": p.placements,
                       "stages": [asdict(s) for s in stages], "planner_version": p.planner_version, "forced": forced},
                      sort_keys=True, default=str)
    p.sha = hashlib.sha256(body.encode()).hexdigest()
    return p


# ---------------------------------------------------------------------------
# 5. `[2026-08-25]` doc 56 H1b: the chunked-prefill plan -- per-chunk stages.
# ---------------------------------------------------------------------------

def plan_ubatch_prefill(graph, logical_tokens, ubatch, caps=NPU2_CAPS, ctx=2048,
                        precision_plan="bf16", forced=None):
    """The plan of an INCREMENTAL prefill: a `logical_tokens` prompt in
    `ubatch`-token chunks, chunk-outer / layer-inner (doc 56 sections 3.4 / 5).

    Chunk i is the ordinary prefill plan at Workload("prefill", M=ubatch,
    kv_len=(i+1)*ubatch) -- its FA stage is the square "flash_attn" for chunk
    1 and the rectangular "flash_attn_ctx<kv>" after. The composed Plan holds
    every chunk's stages IN ORDER (each chunk: its embed_lookup of the chunk's
    rows, then the per-layer stages; the LAST chunk also final_rms_norm + the
    LM head), so `total_host_ops` / `total_launches` / `total_submissions`
    are what the driver must dispatch and the runner's live checks enforce --
    the chunked form is MODELLED, never special-cased at the check.

    logical_tokens == ubatch composes one chunk and returns the base plan
    unchanged (same sha as the whole-prompt plan: it IS the same execution).

    A prompt that is not a whole number of chunks is a refusal -- the chunked
    scheduler has no padding path (H1b: EOS padding gone; a padded tail would
    need masking and exclusion from the numerator, and nothing ships that).
    """
    logical_tokens, ubatch = int(logical_tokens), int(ubatch)
    if ubatch <= 0 or logical_tokens <= 0 or logical_tokens % ubatch != 0:
        raise ValueError(
            f"ubatch prefill: prompt of {logical_tokens} tokens is not a whole "
            f"number of ubatch={ubatch} chunks (no padding path; doc 56 H1b)")
    n_chunks = logical_tokens // ubatch
    chunk_plans = [
        plan(graph, Workload("prefill", ubatch, (i + 1) * ubatch, ctx, precision_plan), caps, forced=forced)
        for i in range(n_chunks)
    ]
    if n_chunks == 1:
        return chunk_plans[0]

    from dataclasses import replace as _replace

    spec = graph.spec
    L = spec.n_layers
    stages, rejected = [], list(chunk_plans[0].rejected)
    for i, cp in enumerate(chunk_plans):
        tag = f"chunk {i}: context {i * ubatch}->{(i + 1) * ubatch}"
        for s in cp.stages:
            if s.repeated or s.name == "embed_lookup":
                stages.append(_replace(s, note=(tag + ("; " + s.note if s.note else ""))))
        if i == n_chunks - 1:
            for s in cp.stages:
                if not s.repeated and s.name != "embed_lookup":
                    tail_tag = tag + " (after the last chunk)"
                    stages.append(_replace(s, note=(tail_tag + ("; " + s.note if s.note else ""))))

    p = Plan(spec.name,
             {**chunk_plans[0].workload, "kv_len": logical_tokens,
              "logical_tokens": logical_tokens, "ubatch_tokens": ubatch, "n_chunks": n_chunks},
             caps.as_dict(), chunk_plans[0].placements, stages, rejected, forced=dict(forced or {}))
    dev = [s for s in stages if s.where == DEVICE]
    hst = [s for s in stages if s.where == HOST]
    reps = lambda s: L if s.repeated else 1
    p.per_layer_launches = chunk_plans[0].per_layer_launches      # per layer per chunk
    p.per_layer_submissions = chunk_plans[0].per_layer_submissions
    p.per_layer_host_ops = chunk_plans[0].per_layer_host_ops
    p.total_launches = sum(s.launches * reps(s) for s in dev)
    p.total_submissions = sum(reps(s) for s in dev)
    p.total_host_ops = sum(reps(s) for s in hst)
    # weights are RESIDENT once, not per chunk; boundary bytes cross per chunk
    p.resident_weight_bytes = chunk_plans[0].resident_weight_bytes
    p.boundary_bytes = sum(s.boundary_bytes * reps(s) for s in stages)
    bnd = p.total_launches * caps.launch_boundary_us
    sub = p.total_submissions * caps.run_fixed_us
    gemm_us = sum(cp.est_breakdown.get("gemm_us_from_registry", 0.0) for cp in chunk_plans)
    p.est_breakdown = {"boundaries_us": bnd, "submissions_us": sub, "gemm_us_from_registry": gemm_us}
    p.est_us = sum(p.est_breakdown.values())
    p.source = (FORCED if forced else ANALYTICAL) if any(cp.source != MEASURED for cp in chunk_plans) else MEASURED
    body = json.dumps({"composition": "ubatch_prefill", "chunks": [cp.sha for cp in chunk_plans],
                       "logical_tokens": logical_tokens, "ubatch": ubatch,
                       "planner_version": p.planner_version, "forced": dict(forced or {})}, sort_keys=True)
    p.sha = hashlib.sha256(body.encode()).hexdigest()
    return p
