# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AWQ uint4 -> bf16 dequant -> bfp16ebs8 packed BO loader."""

import os
import sys
from typing import Optional

import numpy as np
from ml_dtypes import bfloat16

_HERE = os.path.dirname(os.path.abspath(__file__))
_BFP_GEMM_DIR = os.path.normpath(
    os.path.join(_HERE, "..", "..", "matrix_multiplication", "bf16_x_bfp16")
)
# `[2026-08-26]` this module is also the `w_bfp16_prefill` contract OWNER
# (`quant_contract`, read by the study's quant_* columns), so it must import
# from anywhere -- not only from a process that has already put this model's
# directory and the bf16 sibling's on sys.path.
_LLAMA_BF16 = os.path.normpath(os.path.join(_HERE, "..", "llama32_1b"))
for _p in (_BFP_GEMM_DIR, _LLAMA_BF16, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from awq_pack import (  # noqa: E402
    _HF_AWQ_LAYER_MAP,
    _resolve_safetensor_files,
    awq_dequant_layer,
)
from llama32_1b_weights import LayerWeights, LlamaConfig, LlamaWeights  # noqa: E402
from matmul_bf16_x_bfp16 import pack_b_bfp16ebs8  # noqa: E402


def awq_pack_for_npu_bfp16(
    qweight_i32, qzeros_i32, scales_bf16, gs=128, n_tile=64, k_chunk=128, M_seq=2048
):
    """AWQ qweight/qzeros/scales -> bfp16ebs8 packed BO uint8."""
    del M_seq  # unused; kept for signature parity with the int4 packer
    W_dense_bf16 = awq_dequant_layer(qweight_i32, qzeros_i32, scales_bf16, gs=gs)
    return pack_b_bfp16ebs8(W_dense_bf16, n_tile, k_chunk)


def load_awq_weights_bfp(
    model_path: str,
    config: Optional[LlamaConfig] = None,
    n_tile: int = 64,
    k_chunk: int = 128,
    seq_len: int = 2048,
):
    """AWQ HF checkpoint -> (LlamaWeights bf16, list[dict] of bfp16 BOs).

    Drop-in replacement for awq_pack.load_awq_weights when the prefill
    driver wants bfp16 weight BOs. Same LlamaWeights output (bf16
    dequantized projections) for the CPU/HF reference path; the per-layer
    packed dict carries bfp16ebs8 BOs instead of int4 Q+S+Z BOs.
    """
    from safetensors import safe_open
    import torch

    if config is None:
        config = LlamaConfig()

    files = _resolve_safetensor_files(model_path)
    key_to_file = {}
    for fp in files:
        with safe_open(fp, framework="pt") as f:
            for k in f.keys():
                key_to_file[k] = fp

    def _get(k, as_int32=False):
        with safe_open(key_to_file[k], framework="pt") as f:
            t = f.get_tensor(k)
        if as_int32:
            return t.numpy().astype(np.int32)
        if t.dtype == torch.bfloat16:
            return t.view(torch.int16).numpy().view(bfloat16)
        return t.numpy()

    embed = _get("model.embed_tokens.weight")
    assert embed.shape == (config.vocab_size, config.emb_dim), embed.shape
    final_norm = _get("model.norm.weight")
    assert final_norm.shape == (config.emb_dim,)

    layers_bf16 = []
    layers_packed = []
    for li in range(config.n_layers):
        base = f"model.layers.{li}"
        layer_kw = {
            "attn_norm": _get(f"{base}.input_layernorm.weight"),
            "ffn_norm": _get(f"{base}.post_attention_layernorm.weight"),
        }
        packed_kw = {}
        for hf_suffix, field in _HF_AWQ_LAYER_MAP.items():
            qw = _get(f"{base}.{hf_suffix}.qweight", as_int32=True)
            qz = _get(f"{base}.{hf_suffix}.qzeros", as_int32=True)
            sc = _get(f"{base}.{hf_suffix}.scales")
            layer_kw[field] = awq_dequant_layer(qw, qz, sc, gs=128)
            packed_kw[field] = awq_pack_for_npu_bfp16(
                qw, qz, sc, gs=128, n_tile=n_tile, k_chunk=k_chunk, M_seq=seq_len
            )
        layers_bf16.append(LayerWeights(**layer_kw))
        layers_packed.append(packed_kw)

    weights = LlamaWeights(
        embed_table=embed,
        layers=layers_bf16,
        final_norm=final_norm,
        lm_head=embed,  # tied, matches the int4 path
    )
    return weights, layers_packed


# ---------------------------------------------------------------------------
# `[2026-08-26]` doc 56 H4 (queue item 20): the seam the e2e driver and the
# study's quant_* columns use.
# ---------------------------------------------------------------------------

#: The seven GEMM weight fields the bfp16 prefill stitchers consume, in the
#: order they appear in the two ELFs' argument lists.
BFP16_PREFILL_FIELDS = ("wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down")

#: The builders' shared tile geometry -- `rms_gemms_rope_bfp16_multi` and
#: `o_ffn_bfp16_multi` both build every GEMM at tile_n = 32 / tile_k_l1 = 128,
#: and the packed BO layout IS that geometry (`pack_b_bfp16ebs8` emits
#: [N/tile_n, K/tile_k_l1, tile_bytes]). Packing at any other pair produces a
#: BO the ELF will read as garbage, so the two constants live here, once.
BFP16_N_TILE = 32
BFP16_K_CHUNK = 128

#: Elements sharing one exponent in `bfp16ebs8` -- the format's own group.
BFP16_BLOCK = 8


def pack_layer_bfp16(layer, fields=BFP16_PREFILL_FIELDS, n_tile=BFP16_N_TILE,
                     k_chunk=BFP16_K_CHUNK):
    """`{field: bfp16ebs8 uint8 BO}` from a `LayerWeights`' DENSE bf16 fields.

    The e2e driver's loader (`llama32_1b_int4_weights.load_weights_awq`) has
    already dequantized every AWQ projection to dense bf16 `[in, out]` -- the
    array the bf16 prefill stitchers consume. This transcodes THAT array, so
    both prefill arms compute over bit-identical weights and the HF reference
    the verify gate builds is the reference for both. There is no second
    dequantization and no second scale: in `bfp16ebs8` the block's shared
    8-bit exponent IS the scale, applied by the MMUL itself.
    """
    return {f: pack_b_bfp16ebs8(np.asarray(getattr(layer, f), dtype=bfloat16),
                                n_tile, k_chunk) for f in fields}


def quant_contract(group_size=None):
    """The study's `quant_*` column values for `precision_plan_id =
    w_bfp16_prefill`, owned HERE -- beside the packing code that implements it,
    the way `awq_repacker.quant_contract` owns AWQ's (doc 56 H2a). The study
    never hand-types these; `model_adapter.quant_columns` keeps only the keys
    that ARE schema columns, so the extra `*_name` / `bits` keys below are for
    the plan and the evidence, not the CSV.

    `group_size` is accepted for signature parity with the AWQ contract and is
    recorded rather than used: block floating point's group is a property of
    the FORMAT (8 elements share one exponent), not of the checkpoint, and the
    AWQ g128 grouping is consumed and discarded by the dequantization that
    precedes this transcode.
    """
    return {
        # -- the seven schema columns (study/schema.py:272-283)
        "quant_packing_scheme": (
            "bfp16ebs8: per 8 K-contiguous elements of one N row, one 9-byte record "
            "[shared uint8 exponent | 8 x int8 mantissa] = 1.125 B/elt = 9 bits/elt. "
            "BO is uint8 [N/32, K/128, tile_bytes] -- the builders' shared "
            "tile_n=32 / tile_k_l1=128 geometry, part of the contract, not a "
            "caller's choice (matmul_bf16_x_bfp16.pack_b_bfp16ebs8)"
        ),
        "quant_group_size": BFP16_BLOCK,
        "quant_scale_layout": (
            "none: the block's shared 8-bit exponent IS the scale and the MMUL "
            "applies it -- there is no scale plane and NO DEQUANT PASS "
            "(doc 57 s5b: Hexagon prices its own at HTP_MM_HMX_COST_W_DEQUANT = 3)"
        ),
        "quant_zero_point_layout": "none (symmetric: signed mantissas)",
        "quant_accum_type": "f32 (aie::accum<accfloat>, mm_bf16_x_bfp16.cc); bf16 epilogue",
        "quant_gemm_contract": (
            f"prefill: {BFP16_GEMM_CONTRACT_NAME} -- AWQ uint4 asym g{group_size} "
            "dequantized to the SAME dense bf16 array the bf16 arm's GEMMs consume, "
            "then transcoded to bfp16ebs8 at load time; B is a native mac_8x8_8x8T "
            "operand, A is upconverted in-core, accumulate f32, store bf16"
        ),
        # Decode is untouched by this plan: the same int4 AWQ GEMV the
        # w4_decode plan runs. The two columns differ, which is what they are
        # for (doc 56 sections 3.5 / 3.6).
        "quant_gemv_contract": _awq_gemv_contract(group_size)["quant_gemv_contract"],
        # -- not schema columns; the plan and the evidence read these
        "quant_gemm_contract_name": BFP16_GEMM_CONTRACT_NAME,
        "quant_weight_bits": 9.0,
        "quant_weight_bytes_per_element": 1.125,
        "checkpoint_group_size": group_size,
    }


#: The GEMM contract's NAME, mirrored by `shared/plan/plan.py`'s
#: `W_BFP16_GEMM_CONTRACT` so the plan can name it without importing this
#: model directory (the `fa_cache_name` / `W4_GEMV_CONTRACT` pattern; a host
#: test pins the agreement).
BFP16_GEMM_CONTRACT_NAME = "bfp16ebs8_shared_exp8_mantissa8_native_mmul_operand"


def _awq_gemv_contract(group_size):
    from awq_repacker import quant_contract as _awq_contract

    return _awq_contract(group_size or 128)
