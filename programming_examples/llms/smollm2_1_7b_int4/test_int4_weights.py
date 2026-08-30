# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host test of the GGUF q4_0 loader: one layer, CPU, seconds.

(i)   every layer-0 dense field is the HF bf16 weight up to q4_0 error
      (cosine >= 0.99 against the safetensors tensor), and the as-stored
      llama.cpp row order of q and k is not (cosine < 0.1) -- the un-permute
      is load-bearing, not a no-op;
(ii)  each decode-side packed BO has the shape `pack_inputs`'s arithmetic
      gives at gs = 32 for the ELF builders' m_tile / k_chunk / n_cores;
(iii) `_gguf_provenance` names exactly this layer's promoted linear
      (`blk.0.ffn_down.weight`, Q4_1) and not the Q6_K embedding, which
      `promoted_tensors` also lists and the loader never reads.

Needs the HF snapshot in the cache and the GGUF (`resolve_gguf_path` without
the download arm); prints a SKIP line and exits 0 when neither `$SMOLLM2_GGUF`
nor the cached hub file provides it -- the lit's REQUIRES gate on the same two.
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smollm2_1_7b_int4_weights import (  # noqa: E402
    _GGUF_LINEARS,
    _PACKED_ATTR,
    HF_MODEL,
    load_weights_gguf_q4_0,
    resolve_gguf_path,
)
from smollm2_1_7b_weights import LlamaConfig, _resolve_safetensor_files  # noqa: E402
from gguf_q4_0 import QK4_0, GGUFFile, promoted_tensors  # noqa: E402

#: HF module per `_GGUF_LINEARS` entry (same order).
_HF_NAME = dict(
    zip(
        (f for _, f, _ in _GGUF_LINEARS),
        ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
        + ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"),
    )
)


def cosine(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def llama_permute_rows(w, n_head):
    """`convert_hf_to_gguf.py`'s forward permute: the as-stored row order."""
    out, inn = w.shape
    return w.reshape(n_head, 2, out // n_head // 2, inn).swapaxes(1, 2).reshape(w.shape)


def packed_shape(M, K, m_tile=8, k_chunk=2048, n_cores=8):
    """`pack_inputs`'s docstring arithmetic at gs = QK4_0, one launch."""
    tile_bytes = m_tile * (k_chunk // 2) + (k_chunk // QK4_0) * m_tile * 3
    return (n_cores * (M // n_cores // m_tile) * (K // k_chunk), tile_bytes)


def test_dense_fields_match_hf_and_the_unpermute_is_load_bearing(layer, hf):
    for _, field, n_head in _GGUF_LINEARS:
        w = getattr(layer, field).T  # [out, in], as HF stores it
        ref = hf.get_tensor(f"model.layers.0.{_HF_NAME[field]}.weight")
        c = cosine(w, ref)
        print(f"    {field:7s} cosine vs HF {c:.4f}")
        assert c >= 0.99, f"{field}: cosine {c} < 0.99"
        if n_head is not None:  # q and k: the as-stored order must NOT match
            c_stored = cosine(llama_permute_rows(w, n_head), ref)
            print(f"    {field:7s} as-stored (permuted) cosine {c_stored:.4f}")
            assert c_stored < 0.1, f"{field}: as-stored cosine {c_stored} >= 0.1"


def test_packed_bos_have_the_pack_inputs_shape(layer, cfg):
    dims = {a: getattr(layer, f).shape[::-1] for f, a in _PACKED_ATTR.items()}
    dims["_wgateup_packed"] = (2 * cfg.hidden_dim, cfg.emb_dim)
    for attr, (M, K) in dims.items():
        packed, want = getattr(layer, attr), packed_shape(M, K)
        assert packed.dtype == np.uint8, attr
        assert packed.shape == want, f"{attr}: {packed.shape} != {want}"
        print(f"    {attr:16s} {packed.shape} for [M, K] = {(M, K)}")


def test_provenance_names_the_promoted_linear_and_not_the_embedding(weights, g):
    prov = weights._gguf_provenance
    assert prov == {"blk.0.ffn_down.weight": "quantized_from_reference"}, prov
    listed = promoted_tensors(g)
    assert listed.get("token_embd.weight") == "Q6_K", listed
    assert {n for n in listed if n.startswith("blk.0.")} == set(prov), listed
    print(f"    provenance {prov}; promoted_tensors lists {sorted(listed)}")


def main():
    try:
        gguf = resolve_gguf_path(download=False)
    except FileNotFoundError as e:
        if os.environ.get("SMOLLM2_GGUF"):
            raise  # a wrong path is an error, never a silent skip
        print(f"SKIP: {e}")
        return 0
    from safetensors import safe_open

    cfg = LlamaConfig(n_layers=1)
    weights = load_weights_gguf_q4_0(gguf, HF_MODEL, config=cfg)
    (shard,) = _resolve_safetensor_files(HF_MODEL)
    layer, passed = weights.layers[0], 0
    with safe_open(shard, framework="numpy") as hf:
        runs = (
            (test_dense_fields_match_hf_and_the_unpermute_is_load_bearing, layer, hf),
            (test_packed_bos_have_the_pack_inputs_shape, layer, cfg),
            (
                test_provenance_names_the_promoted_linear_and_not_the_embedding,
                weights,
                GGUFFile(gguf),
            ),
        )
        for fn, *args in runs:
            try:
                fn(*args)
            except AssertionError as e:
                print(f"FAIL  {fn.__name__}: {e}")
                continue
            print(f"PASS  {fn.__name__}")
            passed += 1
    print(f"int4 weights tests: {passed}/{len(runs)} passed")
    return 0 if passed == len(runs) else 1


if __name__ == "__main__":
    sys.exit(main())
