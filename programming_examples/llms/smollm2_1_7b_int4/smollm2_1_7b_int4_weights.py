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
    The GGUF supplies the seven quantized linears per layer (`_GGUF_LINEARS`)
    and nothing else. The bf16 HF checkpoint (the sibling bf16 example's own
    `smollm2_1_7b_weights.load_weights`) supplies embeddings / tied lm_head
    and every norm -- `token_embd.weight` is Q6_K in the GGUF and NEVER read
    from it (`gguf_q4_0.promoted_tensors` lists it beside the Q4_1 linears;
    the tied head follows the AWQ example's policy in `awq_pack.py`) -- and
    the ORIGINAL float weights for the promoted tensors:
    `blk.{0,1,10}.ffn_down.weight` are Q4_1, whose fractional zero-point the
    kernel's uint8 Z plane cannot represent, so `q4_0_payload_for`
    re-quantizes them from the bf16 (route (d) of the attributed record in
    `gguf_q4_0.py`; transcoding from q4_1 is refused there).
    `_gguf_provenance` names every tensor that took this route.

THE RoPE UN-PERMUTE IS NOT OPTIONAL
    llama.cpp stores `attn_q`/`attn_k` with rows permuted for its own RoPE
    layout, so as stored they are row-shuffled against the HF originals
    (`test_int4_weights.py` pins the band: cosine >= 0.99 after
    `llama_unpermute_rows`, < 0.1 as stored). It applies to the OUTPUT rows:
    A_q's rows and the M axis of the S/Z planes for the packed BOs, and the
    [M, K] dequant before its transpose for the dense fields. SmolLM2 is
    full MHA, so q and k both un-permute with n_head = n_kv_head = 32.

The checkpoint path comes from `resolve_gguf_path`: an explicit path, else
`$SMOLLM2_GGUF`, else the hub file (`GGUF_REPO`/`GGUF_FILE`).
"""

import os
import sys
from pathlib import Path
from typing import Optional

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
    unpermute_planes,
)
from matvec_int4_packed import pack_inputs as _pack_inputs  # noqa: E402

HF_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
GGUF_REPO = "bartowski/SmolLM2-1.7B-Instruct-GGUF"
GGUF_FILE = "SmolLM2-1.7B-Instruct-Q4_0.gguf"

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


def resolve_gguf_path(gguf_path: Optional[str] = None, download: bool = True) -> str:
    """The Q4_0 checkpoint to load: `gguf_path`, else `$SMOLLM2_GGUF`, else
    the hub file -- from the HF cache when present (which also lights the
    `hfweights_bartowski_smollm2_1_7b_instruct_gguf` lit feature), otherwise
    downloaded, or `FileNotFoundError` when `download` is False."""
    path = gguf_path or os.environ.get("SMOLLM2_GGUF")
    if path:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"GGUF not found: {path} (gguf_path argument / $SMOLLM2_GGUF)"
            )
        return path
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return hf_hub_download(GGUF_REPO, GGUF_FILE, local_files_only=True)
    except LocalEntryNotFoundError:
        if not download:
            raise FileNotFoundError(
                f"$SMOLLM2_GGUF is unset and {GGUF_REPO}/{GGUF_FILE} is not in "
                "the HF cache"
            )
        return hf_hub_download(GGUF_REPO, GGUF_FILE)


def load_weights_gguf_q4_0(
    gguf_path: Optional[str] = None,
    hf_model_name_or_path: str = HF_MODEL,
    config: Optional[LlamaConfig] = None,
    m_tile: int = 8,
    k_chunk: int = 2048,
    n_cores: int = 8,
) -> LlamaWeights:
    """Load the GGUF q4_0 checkpoint into a dual-layout LlamaWeights.

    Args:
        gguf_path: the Q4_0 GGUF file (see `resolve_gguf_path` for None).
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
    g = GGUFFile(resolve_gguf_path(gguf_path))

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
            # (latent here -- this checkpoint's q/k are all Q4_0 -- and fatal
            # on one where they are not).
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
                A_q, A_s, A_z = unpermute_planes(A_q, A_s, A_z, unpermute)
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
