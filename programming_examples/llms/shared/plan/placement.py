# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Placement: `device | host | refuse` per node, from the lifted bounds.

The `supports_op` analog (doc 56 section 3.3 stage 2). `refuse` is a derived
skip (the study's `skip_reason` reproduced by `study_skip`), `host` is counted
with its boundary bytes. Nothing here predicts latency; it says what can run
where and why.
"""
from __future__ import annotations

from dataclasses import dataclass

from .caps import DeviceCaps, NPU2_CAPS

DEVICE, HOST, REFUSE = "device", "host", "refuse"


@dataclass(frozen=True)
class Workload:
    phase: str            # prefill | decode
    M: int                # query rows (prompt/chunk length in prefill, 1 in decode)
    kv_len: int           # keys visible (== M for a single-chunk prefill; ctx in decode)
    ctx: int = 2048       # KV capacity
    precision_plan: str = "bf16"   # bf16 | w4_decode | w_bfp16_prefill (doc 56 section 3.5)


@dataclass(frozen=True)
class Placement:
    where: str
    reason: str
    extra_host_ops: tuple = ()    # host glue the device form forces (e.g. FA transposes)
    boundary_bytes: int = 0       # bytes that cross host<->device because of this placement


def _attention_placement(node, spec, wl, caps):
    hd = node.attr("head_dim")
    if wl.phase == "decode":
        return Placement(HOST, "single-token GQA over the host-owned KV cache: an NPU FA launch's "
                               "boundary (~107 us) exceeds the one-query compute", ())
    if hd == caps.headfirst_fa_head_dim:
        return Placement(DEVICE, "head-first FlashAttention (hd=128); the BF16 DMA stride-1 rule and the "
                                 "seq-first dk_chunks>1 bug force host seq<->head transposes around it",
                         ("transpose_seq_to_head", "transpose_head_to_seq"))
    if hd == caps.seqfirst_fa_head_dim:
        return Placement(DEVICE, "seq-first FlashAttention (hd=64), no host transposes", ())
    return Placement(REFUSE, f"no FlashAttention variant for head_dim {hd} (have {caps.seqfirst_fa_head_dim}, "
                             f"{caps.headfirst_fa_head_dim})", ())


def place(graph, wl, caps=NPU2_CAPS):
    """Per-node placement for one phase. Returns {node.id: Placement}."""
    spec = graph.spec
    out = {}
    for node in graph.phase_nodes(wl.phase):
        op = node.op
        if op == "embed_lookup":
            p = Placement(HOST, "table lookup; the row is the host's to send")
        elif op == "kv_append":
            p = Placement(HOST, "the KV cache is host-owned in every shipped driver (device-owned KV is doc 56 H3.4)")
        elif op == "attention":
            p = _attention_placement(node, spec, wl, caps)
        elif node.id == "final_norm":
            p = Placement(HOST, "one row of RMSNorm before the LM head; both drivers do it on the host")
        elif op == "matmul":
            p = Placement(DEVICE, "GEMM (prefill) / GEMV (decode) on the array")
        elif op in ("rms_norm", "rms_norm_per_head", "rope", "swiglu", "add"):
            p = Placement(DEVICE, "glue fused into the enclosing builder pattern")
        else:
            p = Placement(REFUSE, f"unknown op {op}")
        out[node.id] = p
    return out


# --- the study's skip predicate, reproduced from the lifted caps --------------

PROFILE_MODES = ("coarse", "offload", "runlist", "fused")
FA_MODES = ("coarse", "coarse_c2", "coarse_c3", "fused")
ATTN_GEMM_MODES = ("offload", "runlist")
DEVICE_SOFTMAX_MODES = ("runlist",)


def study_skip(mode, seq, emb, ladder, unbuildable=None, variant=None, caps=NPU2_CAPS):
    """`profiles.skip_reason` from the planner's side: same four clauses, same order.

    `unbuildable` is run_mode.UNBUILDABLE_VARIANTS[variant] (a {mode: reason}
    dict) when the caller has it; the planner does not import run_mode.
    Returns a short tag (not the study's prose) or None.
    """
    if unbuildable and mode in unbuildable:
        return f"variant:{variant}:{mode}"
    if mode == "fused":
        low, high = caps.fused_seq_range(emb, ladder)
        if seq < low or seq > high:
            return "fused:plane-stride"
    if mode in FA_MODES and seq % caps.fa_parallel_seq:
        return "fa:parallel_seq"
    if mode in ATTN_GEMM_MODES and seq % caps.attn_gemm_seq_multiple:
        return "attn-gemm:seq-multiple"
    if mode in DEVICE_SOFTMAX_MODES and not caps.softmax_fits_l1(seq):
        return "softmax:l1"
    return None
