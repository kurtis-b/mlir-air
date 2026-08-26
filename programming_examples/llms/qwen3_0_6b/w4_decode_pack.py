# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""`w4_decode` weight path for Qwen3-0.6B (doc 56 H2b, queue item 18).

ONE owner of, in this order:

* the selection flag (`QWEN3_W4_DECODE`, default OFF -- bf16 stays the
  production default until the operator flips it);
* the quantization parameters (RTN asymmetric uint4, GROUP_SIZE=128 --
  the same in-kernel `(q - z) * s` contract as the llama AWQ path; the
  packing tiling K_CHUNK=1024 = qwen's emb, the int4 cascade's DIM_K);
* `quantize_decode_weights`: fake-quantize + pack the three decode O+FFN
  matrices per layer (`_wo_packed` / `_wgateup_packed` / `_wdown_packed`,
  the llama attribute names) AND substitute the bf16 fields with the
  DEQUANTIZED copy -- doc 56 section 3.5's "bf16 prefill weights resident
  separately": prefill's bf16 GEMMs and decode's in-kernel dequant then
  compute from numerically ONE model, which is what lets the verify
  oracle be patched once (the adapter's `build_hf_model` patches HF with
  these same substituted fields);
* the `quant_*` schema columns (`quant_contract`), derived from the llama
  packing owner (`llama32_1b_int4.awq_repacker.quant_contract` -- the
  kernel-side contract IS the same: same packer format, same
  `mv_int4_bf16.cc`) with the provenance fields rewritten to this model's
  reality: RTN fake-quantize of the bf16 checkpoint, no AWQ checkpoint.

Quantized here: wo, w_gate, w_up, w_down (the token's largest weight mass;
o_gemv_ffn_int4). NOT quantized -- priced negatives in
results/item18-h2b-20260826/PREDICTION.md section 2: the QKV stage (the
2-launch in-core QK-norm+RoPE epilogue cannot host in-kernel dequant
without a new kernel, and every fallback launch structure loses more to
boundaries than the bytes save) and the LM head (doc 57 section 5 item 6:
-0.46 ms measured ceiling, dequant-bound at 11 GB/s, one-launch form
impossible).
"""

import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_THIS_DIR = Path(__file__).resolve().parent
_LLMS_DIR = _THIS_DIR.parent
if str(_LLMS_DIR) not in sys.path:
    sys.path.insert(0, str(_LLMS_DIR))

#: The selection flag. Read at call time (not import time) so one process can
#: decide before importing the driver; `model_adapter.prepare` sets it from
#: `precision_plan` before the driver import.
W4_ENV = "QWEN3_W4_DECODE"

#: RTN asymmetric uint4 group size. 128 mirrors the llama AWQ contract (the
#: one owner of the contract NAME is `llama32_1b_int4.awq_repacker`); the
#: kernel dequants `(q - z) * s` per group either way.
GROUP_SIZE = 128

#: int4 GEMV packing tiling for this model's cascade: K_CHUNK == emb_dim so
#: stage 2 (swiglu_rms) is single-chunk while O (K=q_dim=2048) and down
#: (K=hidden=3072) split into 2 / 3 chunks against ONE mv_int4_bf16.o at
#: DIM_K=K_CHUNK.
M_TILE = 8
K_CHUNK = 1024
N_CORES = 8


def w4_decode_selected() -> bool:
    """True when the operator (or the study runner) selected the w4_decode
    path. Default OFF: bf16 is the production default (doc 56 H2b)."""
    return os.environ.get(W4_ENV, "0") == "1"


def _fake_quantize(W_out_in):
    """RTN asym uint4 gs=GROUP_SIZE of a [M=out, K=in] bf16 matrix ->
    (W_q [M, K/2] u8, W_s [K/gs, M] bf16, W_z [K/gs, M] u8), the
    `pack_inputs` layout. Reuses the llama int4 example's quantizer."""
    from llama32_1b_int4.awq_pack import fake_quantize_awq_int4

    return fake_quantize_awq_int4(np.ascontiguousarray(W_out_in), gs=GROUP_SIZE)


def dequant_rows(W_q, W_s, W_z, gs=GROUP_SIZE):
    """Dequantize the (W_q [M, K/2], W_s [K/gs, M], W_z [K/gs, M]) triplet
    back to a dense [M, K] bf16 -- the copy prefill computes on and the HF
    oracle is patched with. Mirrors the kernel's `(q - z) * s`."""
    M, K_half = W_q.shape
    K = 2 * K_half
    q = np.empty((M, K), dtype=np.int32)
    q[:, 0::2] = W_q & 0x0F
    q[:, 1::2] = (W_q >> 4) & 0x0F
    s = np.repeat(W_s.astype(np.float32).T, gs, axis=1)  # [M, K]
    z = np.repeat(W_z.astype(np.int32).T, gs, axis=1)  # [M, K]
    return ((q - z) * s).astype(bfloat16)


def _ensure_int4_path():
    """matvec_int4_packed lives in the int4_awq example dir; resolve it the
    way the llama int4 loader does (lazily -- only the w4 path pays the
    sys.path insert)."""
    int4_dir = _LLMS_DIR.parent / "matrix_vector_multiplication" / "int4_awq"
    if str(int4_dir) not in sys.path:
        sys.path.insert(0, str(int4_dir))


def _pack(W_q, W_s, W_z):
    _ensure_int4_path()
    from matvec_int4_packed import pack_inputs

    M, K_half = W_q.shape
    K = 2 * K_half
    return pack_inputs(
        W_q, W_s, W_z, M, K, GROUP_SIZE, M_TILE, K_CHUNK, N_CORES, M
    )


def quantize_decode_weights(weights, config):
    """Apply the w4_decode weight transformation in place.

    Per layer: wo / w_gate / w_up / w_down are RTN-quantized in the GEMV
    orientation ([out, in] = the loader field transposed), packed into the
    decode BOs (`_wo_packed`, `_wgateup_packed` gate/up row-interleaved as
    the int4 FFN ELF consumes, `_wdown_packed`), and the loader's bf16
    fields are REPLACED by the dequantized copy (transposed back to the
    loader's (in, out) orientation) so prefill and the HF oracle see the
    same numbers the decode kernel dequants. QKV, norms, embed and the LM
    head are untouched (bf16). Idempotent via `_w4_decode_applied`.
    """
    if getattr(weights, "_w4_decode_applied", False):
        return weights
    _ensure_int4_path()

    emb = config.emb_dim
    q_dim = config.n_heads * config.head_dim
    hidden = config.hidden_dim

    for li, lw in enumerate(weights.layers):
        assert lw.wo.shape == (q_dim, emb) and lw.w_gate.shape == (emb, hidden)
        assert lw.w_up.shape == (emb, hidden) and lw.w_down.shape == (hidden, emb)

        # O: GEMV orientation [out=emb, in=q_dim].
        q, s, z = _fake_quantize(lw.wo.astype(bfloat16).T)
        lw._wo_packed = _pack(q, s, z)
        lw.wo = np.ascontiguousarray(dequant_rows(q, s, z).T)  # (q_dim, emb)

        # gate / up: [out=hidden, in=emb], rows interleaved 2i=gate, 2i+1=up
        # (the layout matvec_int4_swiglu_rms consumes; llama's loader shape).
        gq, gs_, gz = _fake_quantize(lw.w_gate.astype(bfloat16).T)
        uq, us, uz = _fake_quantize(lw.w_up.astype(bfloat16).T)
        gu_q = np.empty((2 * hidden, emb // 2), dtype=np.uint8)
        gu_q[0::2], gu_q[1::2] = gq, uq
        n_groups = gs_.shape[0]
        gu_s = np.empty((n_groups, 2 * hidden), dtype=gs_.dtype)
        gu_s[:, 0::2], gu_s[:, 1::2] = gs_, us
        gu_z = np.empty((n_groups, 2 * hidden), dtype=np.uint8)
        gu_z[:, 0::2], gu_z[:, 1::2] = gz, uz
        lw._wgateup_packed = _pack(gu_q, gu_s, gu_z)
        lw.w_gate = np.ascontiguousarray(dequant_rows(gq, gs_, gz).T)  # (emb, hidden)
        lw.w_up = np.ascontiguousarray(dequant_rows(uq, us, uz).T)

        # down: [out=emb, in=hidden].
        dq, ds, dz = _fake_quantize(lw.w_down.astype(bfloat16).T)
        lw._wdown_packed = _pack(dq, ds, dz)
        lw.w_down = np.ascontiguousarray(dequant_rows(dq, ds, dz).T)  # (hidden, emb)

        if li == 0 or (li + 1) % 7 == 0:
            print(f"  w4_decode: layer {li + 1}/{config.n_layers} quantized+packed")

    weights._w4_decode_applied = True
    return weights


def quant_contract(group_size=None):
    """The w4_decode quant_* schema columns for this model.

    Derived from the llama packing owner's contract (same packer layout,
    same kernel, same group size -- `quant_gemv_contract_name` is BY
    CONSTRUCTION the name the plan package mirrors as `W4_GEMV_CONTRACT`),
    with the checkpoint-provenance fields rewritten: this model has no AWQ
    checkpoint; the weights are RTN fake-quantized from the bf16 checkpoint
    by THIS module and prefill runs on the dequantized copy it substitutes.
    """
    if group_size is None:
        group_size = GROUP_SIZE
    elif group_size != GROUP_SIZE:
        raise ValueError(
            f"quant_contract: caller-supplied group_size {group_size} contradicts "
            f"w4_decode_pack.GROUP_SIZE {GROUP_SIZE}; this module is the owner"
        )
    from llama32_1b_int4.awq_repacker import quant_contract as _llama_contract

    c = dict(_llama_contract(group_size=group_size))
    c["quant_gemm_contract"] = (
        "prefill: bf16 GEMMs on the dequantized RTN copy this module "
        "substitutes ((q - z) * s, w4_decode_pack.quantize_decode_weights; "
        "no AWQ checkpoint -- RTN fake-quantize of the bf16 checkpoint via "
        "awq_pack.fake_quantize_awq_int4)"
    )
    c["quant_gemv_contract"] = (
        f"decode: {c['quant_gemv_contract_name']} -- in-kernel (q - z) * s per "
        f"group of {group_size}, bf16 compute, accfloat accumulate, bf16 "
        "visible (matvec_int4_packed / mv_int4_bf16.cc at DIM_K="
        f"{K_CHUNK}); quantized matrices: wo, gate/up (nibble-row-"
        "interleaved), down; QKV and LM head stay bf16 (priced negatives, "
        "doc 56 H2b)"
    )
    return c
