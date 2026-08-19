# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GGUF q4_0 weight loader for the int4 SmolLM2-1.7B example.

Loads the bartowski SmolLM2-1.7B-Instruct-Q4_0 GGUF checkpoint once and
populates BOTH layouts, mirroring `llama32_1b_int4_weights.load_weights_awq`:

* bf16 dequant fields on each LayerWeights (wq / wk / wv / wo / w_gate /
  w_up / w_down) — consumed by the NPU **bf16 prefill** path. The dequant is
  taken FROM THE Q4_0 PAYLOAD (`dequant_q4_0_reference`, bf16 scales), never
  from the original bf16 weights, so prefill runs the same numbers the int4
  decode path carries and the verify gate exercises the quantized model.
* per-layer decode-side packed BO attributes (`_wq_packed`, `_wk_packed`,
  `_wv_packed`, `_wo_packed`, `_wgateup_packed`, `_wdown_packed`) —
  consumed by the NPU **int4 decode** ELFs, at this checkpoint's group size
  `gs = 32` (q4_0's block size; the AWQ example runs gs = 128).

WHAT COMES FROM WHERE, AND WHY BOTH CHECKPOINTS ARE OPEN
    The GGUF supplies the seven quantized linears per layer. The bf16 HF
    checkpoint (loaded through the sibling bf16 example's own
    `smollm2_1_7b_weights.load_weights`) supplies:

    * embeddings / tied lm_head — Q6_K in the GGUF and NEVER consumed from
      it (`gguf_q4_0.py`'s recorded policy; same as the AWQ example's
      "reusing embed_table as lm_head"), plus every norm;
    * the ORIGINAL float weights for the checkpoint's three promoted
      tensors — `blk.{0,1,10}.ffn_down.weight` are Q4_1, whose fractional
      zero-point the kernel's uint8 Z plane cannot represent, so
      `q4_0_payload_for` re-quantizes them from the original bf16 (route (d),
      measured 0.087-0.088 rms/rms against the accepted q4_0 band
      0.083-0.085; transcoding measured 0.111 and was REFUSED).

THE RoPE UN-PERMUTE IS NOT OPTIONAL
    llama.cpp stores `attn_q`/`attn_k` with rows permuted for its own RoPE
    layout; measured cosine vs the HF originals is ~0.03 as stored and ~0.996
    after `llama_unpermute_rows` (gguf_q4_0.py). It applies to the OUTPUT
    rows: A_q's rows and the M axis of the S/Z planes for the packed BOs, and
    the [M, K] dequant before its transpose for the dense fields. SmolLM2 is
    full MHA, so q and k both un-permute with n_head = n_kv_head = 32.
"""

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from ml_dtypes import bfloat16

_THIS_DIR = Path(__file__).resolve().parent
_LLMS_DIR = _THIS_DIR.parent
_SMOLLM2_BF16_DIR = _LLMS_DIR / "smollm2_1_7b"
_PROG_EXAMPLES = _LLMS_DIR.parent
for p in (str(_LLMS_DIR), str(_SMOLLM2_BF16_DIR), str(_THIS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from smollm2_1_7b_weights import (  # noqa: E402
    LlamaConfig,
    LlamaWeights,
    load_weights,
)

sys.path.insert(
    0,
    str(_PROG_EXAMPLES / "matrix_vector_multiplication" / "int4_awq"),
)
from gguf_q4_0 import (  # noqa: E402
    GGUFFile,
    QK4_0,
    dequant_q4_0_reference,
    llama_unpermute_rows,
    q4_0_payload_for,
    repack_q4_0_linear,
)
from matvec_int4_packed import pack_inputs as _pack_inputs  # noqa: E402

#: GGUF tensor suffix -> (dataclass field, un-permute head count or None).
#: SmolLM2 is full MHA, so q and k share the same 32; v/o/gate/up/down are
#: stored un-permuted.
_GGUF_LINEARS = (
    ("attn_q", "wq", 32),
    ("attn_k", "wk", 32),
    ("attn_v", "wv", None),
    ("attn_output", "wo", None),
    ("ffn_gate", "w_gate", None),
    ("ffn_up", "w_up", None),
    ("ffn_down", "w_down", None),
)

#: Field name -> per-layer packed-BO attribute used by the decode ELFs.
#: w_gate / w_up are interleaved at the plane level into one
#: `_wgateup_packed` BO, same as the AWQ example.
_PACKED_ATTR = {
    "wq": "_wq_packed",
    "wk": "_wk_packed",
    "wv": "_wv_packed",
    "wo": "_wo_packed",
    "w_down": "_wdown_packed",
}


def _unpermute_planes(A_q, A_s, A_z, n_head):
    """Undo llama.cpp's RoPE row permutation on all three packed planes.

    The permutation reorders OUTPUT rows: A_q's rows directly, and the M axis
    of the per-group S/Z planes (which are [n_groups, M], hence the double
    transpose). Identical to `gguf_q4_0.repack_q4_0_for_gemv`'s treatment,
    exposed here because gate/up interleaving needs the planes BEFORE
    `pack_inputs`.
    """
    A_q = np.ascontiguousarray(llama_unpermute_rows(A_q, n_head))
    A_s = np.ascontiguousarray(llama_unpermute_rows(A_s.T, n_head).T)
    A_z = np.ascontiguousarray(llama_unpermute_rows(A_z.T, n_head).T)
    return A_q, A_s, A_z


def load_weights_gguf_q4_0(
    gguf_path: str,
    hf_model_name_or_path: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    config: Optional[LlamaConfig] = None,
    m_tile: int = 8,
    k_chunk: int = 2048,
    n_cores: int = 8,
) -> LlamaWeights:
    """Load the GGUF q4_0 checkpoint into a dual-layout LlamaWeights.

    Args:
        gguf_path: the Q4_0 GGUF file.
        hf_model_name_or_path: the bf16 HF checkpoint — embeddings, norms,
            tied lm_head, and the promoted tensors' quantization references
            come from it (see the module docstring).
        config: model hyperparameters (defaults to SmolLM2-1.7B).
        m_tile, k_chunk, n_cores: GEMV packed-BO tiling parameters; defaults
            match the int4 decode ELF builders.

    Returns:
        LlamaWeights with bf16 dequant fields AND per-layer decode-side
        packed-BO attributes attached, plus a ``_gguf_provenance`` dict
        naming every tensor that was re-quantized from the reference.
    """
    if config is None:
        config = LlamaConfig()

    print("  loading bf16 HF base (embeddings, norms, references)...")
    weights = load_weights(hf_model_name_or_path, config=config)
    g = GGUFFile(gguf_path)

    promoted: dict = {}
    for layer_idx, layer in enumerate(weights.layers):
        for suffix, field_name, n_head in _GGUF_LINEARS:
            name = f"blk.{layer_idx}.{suffix}.weight"
            if name not in g.tensors:
                raise KeyError(f"Missing GGUF tensor: {name}")
            info = g.tensors[name]
            K, M = info.ne[0], info.ne[1]
            # The base loader stores [in, out]; the payload reference wants
            # [out, in] as HF stores it. f32 for the quantizer.
            base_field = getattr(layer, field_name)
            if base_field.shape != (K, M):
                raise ValueError(
                    f"{name}: base bf16 field is {base_field.shape}, "
                    f"GGUF says [in, out] = {(K, M)}"
                )
            reference = np.ascontiguousarray(base_field.T).astype(np.float32)
            payload, provenance = q4_0_payload_for(g, name, reference=reference)
            if provenance != "checkpoint":
                promoted[name] = provenance

            # The un-permute applies ONLY to the checkpoint's own bytes:
            # llama.cpp permutes q/k rows on conversion, but a payload
            # re-quantized from the HF reference is already in HF row order,
            # and un-permuting it would silently shuffle the output channels
            # (Codex review finding; latent here -- this checkpoint's q/k are
            # all Q4_0 -- and fatal on one where they are not).
            unpermute = n_head if provenance == "checkpoint" else None

            # (a) dense bf16 for the bf16 prefill path — FROM THE PAYLOAD,
            # with bf16 scales, so prefill computes on exactly the numbers
            # the packed BOs carry.
            w = dequant_q4_0_reference(payload, K, M, scale_dtype="bf16")
            if unpermute is not None:
                w = llama_unpermute_rows(w, unpermute)
            setattr(layer, field_name, np.ascontiguousarray(w.T).astype(bfloat16))

            # (b) decode-side packed planes; gate/up deferred for interleave.
            A_q, A_s, A_z = repack_q4_0_linear(payload, K, M)
            if unpermute is not None:
                A_q, A_s, A_z = _unpermute_planes(A_q, A_s, A_z, unpermute)
            if field_name == "w_gate":
                gate_quants = (A_q, A_s, A_z)
            elif field_name == "w_up":
                up_quants = (A_q, A_s, A_z)
            else:
                setattr(
                    layer,
                    _PACKED_ATTR[field_name],
                    _pack_inputs(
                        A_q, A_s, A_z, M, K, QK4_0, m_tile, k_chunk, n_cores, M
                    ),
                )

        # Interleave gate/up at the (A_q, A_s, A_z) level: row 2i = gate[i],
        # row 2i+1 = up[i]. One BO for the int4 FFN ELF, same as the AWQ
        # example (`llama32_1b_int4_weights.py`).
        g_q, g_s, g_z = gate_quants
        u_q, u_s, u_z = up_quants
        h_out, k_half = g_q.shape
        if u_q.shape != (h_out, k_half):
            raise RuntimeError("ffn_gate and ffn_up have different shapes")
        gu_q = np.empty((2 * h_out, k_half), dtype=np.uint8)
        gu_q[0::2] = g_q
        gu_q[1::2] = u_q
        n_groups = g_s.shape[0]
        gu_s = np.empty((n_groups, 2 * h_out), dtype=g_s.dtype)
        gu_s[:, 0::2] = g_s
        gu_s[:, 1::2] = u_s
        gu_z = np.empty((n_groups, 2 * h_out), dtype=np.uint8)
        gu_z[:, 0::2] = g_z
        gu_z[:, 1::2] = u_z
        M_gateup = 2 * h_out
        K_full = k_half * 2
        layer._wgateup_packed = _pack_inputs(
            gu_q,
            gu_s,
            gu_z,
            M_gateup,
            K_full,
            QK4_0,
            m_tile,
            k_chunk,
            n_cores,
            M_gateup,
        )

        if (layer_idx + 1) % 4 == 0 or layer_idx == 0:
            print(f"  GGUF q4_0 layer {layer_idx + 1}/{config.n_layers} loaded")

    if promoted:
        print(
            f"  {len(promoted)} promoted tensor(s) re-quantized from the bf16 "
            f"reference: {sorted(promoted)}"
        )
    weights._gguf_provenance = promoted
    return weights
