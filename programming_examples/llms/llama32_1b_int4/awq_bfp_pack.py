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
from matmul_bf16_x_bfp16 import (  # noqa: E402
    BFP16_BLOCK,
    BFP16_BYTES_PER_BLOCK,
    pack_b_bfp16ebs8,
)


def awq_pack_for_npu_bfp16(
    qweight_i32, qzeros_i32, scales_bf16, gs=128, n_tile=None, k_chunk=None,
    M_seq=2048
):
    """AWQ qweight/qzeros/scales -> bfp16ebs8 packed BO uint8.

    `[2026-08-27, review of queue item 22]` `n_tile` defaulted to **64** while
    the builders, the shipped cache and `BFP16_N_TILE_DEFAULT` are all 32, so
    this entry point's default output was silently misread by the default ELF.
    It follows the live packer geometry now (`BFP16_N_TILE`), which is the ONE
    value the artifact-set check is made against.
    """
    del M_seq  # unused; kept for signature parity with the int4 packer
    n_tile = BFP16_N_TILE if n_tile is None else n_tile
    k_chunk = BFP16_K_CHUNK if k_chunk is None else k_chunk
    W_dense_bf16 = awq_dequant_layer(qweight_i32, qzeros_i32, scales_bf16, gs=gs)
    return pack_b_bfp16ebs8(W_dense_bf16, n_tile, k_chunk)


def load_awq_weights_bfp(
    model_path: str,
    config: Optional[LlamaConfig] = None,
    n_tile: Optional[int] = None,
    k_chunk: Optional[int] = None,
    seq_len: int = 2048,
):
    """AWQ HF checkpoint -> (LlamaWeights bf16, list[dict] of bfp16 BOs).

    Drop-in replacement for awq_pack.load_awq_weights when the prefill
    driver wants bfp16 weight BOs. Same LlamaWeights output (bf16
    dequantized projections) for the CPU/HF reference path; the per-layer
    packed dict carries bfp16ebs8 BOs instead of int4 Q+S+Z BOs.

    `[2026-08-27, review of queue item 22]` `n_tile` defaulted to 64 while the
    builders, the shipped cache and `BFP16_N_TILE_DEFAULT` were all 32, so a
    caller that omitted it got BOs the default ELF misreads -- silently, since
    every width has the same BO size. It follows the live packer geometry now.

    It does NOT check anything against an artifact set, and the
    `prefill_cache_dir` parameter the first fix added here is GONE
    `[2026-08-27, second review]`: loading a checkpoint is not the moment the
    weights meet an ELF, and a check here was a second place answering "what
    does a missing declaration mean?" -- which is how the guard ended up with
    four answers. The layout invariant is enforced once, by
    `assert_layout_agrees`, immediately before the first dispatch that consumes
    these buffers, and it DERIVES their layout rather than trusting anything
    recorded here.
    """
    from safetensors import safe_open
    import torch

    if config is None:
        config = LlamaConfig()
    n_tile = BFP16_N_TILE if n_tile is None else n_tile
    k_chunk = BFP16_K_CHUNK if k_chunk is None else k_chunk

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
#: These are the BUILDERS' SIGNATURE DEFAULTS and a host test pins them there.
BFP16_N_TILE_DEFAULT = 32
BFP16_K_CHUNK_DEFAULT = 128

#: `[2026-08-27]` queue item 22: the N tile is now selectable, because it is
#: worth 1.8x on the GEMM (item 22's measurement) and the builders take it as a
#: parameter. Only widths this item compiled AND measured are admitted; anything
#: else REFUSES rather than packing a BO nobody has run.
BFP16_TILE_N_ENV = "LLAMA32_1B_INT4_BFP16_TILE_N"
BFP16_TILE_N_SUPPORTED = (32, 64, 128)


def _read_bfp16_tile_n():
    raw = (os.environ.get(BFP16_TILE_N_ENV) or "").strip()
    if not raw:
        return BFP16_N_TILE_DEFAULT
    if not raw.isdigit() or int(raw) not in BFP16_TILE_N_SUPPORTED:
        raise ValueError(
            f"{BFP16_TILE_N_ENV}={raw!r}: the bfp16 prefill N tile must be one of "
            f"{BFP16_TILE_N_SUPPORTED}. It is not a free number: the linked "
            f"mm_bf16_x_bfp16.o bakes -DDIM_N, every GEMM's N must divide by "
            f"tile_n * herd_n(4), and a width nobody compiled has no ELF to run "
            f"against. Refusing rather than packing weights for a kernel that "
            f"does not exist."
        )
    return int(raw)


#: The LIVE geometry the packer will emit at. Equal to the default unless the
#: env names another admitted width.
BFP16_N_TILE = _read_bfp16_tile_n()
BFP16_K_CHUNK = BFP16_K_CHUNK_DEFAULT

# ---------------------------------------------------------------------------
# THE ONE INVARIANT
#
#     the layout the weights were PACKED IN
#         ==
#     the layout the ELF about to consume them was BUILT FOR
#
# Two facts have to meet, and they are established very differently.
#
#   fact A -- what the packed buffers ARE. **Derived from the buffers**, not
#       recorded beside them: `pack_b_bfp16ebs8` emits
#       `[N/tile_n, K/tile_k_l1, tile_n*tile_k_l1//8*9]`, so with the dense
#       array the pair is solvable AND the third axis cross-checks it. Nothing
#       is stamped, so nothing can go stale when a layer is repacked or
#       replaced, and nothing can be forged short of forging the arrays.
#
#   fact B -- what the ELF EXPECTS. **Not derivable at all**: the ELF's weight
#       argument is the same number of bytes at every width, which is the whole
#       hazard. So it is DECLARED by whatever built the set -- and a declaration
#       is EVIDENCE, NOT AUTHORITY: it is usable only if it names a layout this
#       build can actually produce.
#
# Enforced ONCE, by `assert_layout_agrees`, at the point where the two meet:
# immediately before the first dispatch that consumes a packed weight BO. Not
# at checkpoint load, not inside the idempotent transcode, not at four call
# sites with four different answers to what a missing fact means.
#
# CREATE vs CONSUME. Compiling a set is the act that ESTABLISHES fact B, so the
# compile path WRITES the declaration and proceeds; it cannot be asked to find
# one first. Consuming a set requires the fact to be there already.
# `[2026-08-27, second review of queue item 22]` the previous fix inverted that
# and locked bfp16 bootstrapping -- `make compile-bfp16` on a clean cache
# refused before it could build the ELFs whose declaration it would then write.
# A guard that blocks correct use is not fail-closed, it is broken.
# ---------------------------------------------------------------------------

#: Name of the sidecar an artifact set carries to declare the tile geometry its
#: ELFs were BUILT at. Written by whatever compiles the set.
BFP16_GEOMETRY_SIDECAR = "bfp16_geometry.json"


class Bfp16LayoutError(ValueError):
    """Raised for every refusal below, so a caller can catch this one class."""


def layout_is_buildable(tile_n, tile_k_l1):
    """Can this build actually produce that layout? Fact B's validator.

    A declaration naming anything else is not a fact about any ELF we could
    have built, so it is refused rather than believed. Without this, a
    hand-written sidecar declaring `tile_n = 256` (or an arbitrary
    `tile_k_l1`) self-certifies simply by matching a packer told to use the
    same numbers -- the shape of gate item 28's review condemned: certifying
    that a declaration EXISTS while claiming the layout MATCHES.
    """
    # STRICT about the type, not just the value: a declaration is machine
    # written (`write_declared_layout` emits ints), so accepting anything that
    # merely coerces would admit `32.9` and `True` as "32" and "1". Coercion is
    # a class of wrong answer this guard has no reason to take on.
    for v in (tile_n, tile_k_l1):
        if not isinstance(v, int) or isinstance(v, bool):
            return False
    tn, tk = tile_n, tile_k_l1
    if tn not in BFP16_TILE_N_SUPPORTED:
        return False
    # tile_k_l1 is not a swept axis: the builders and the linked micro-kernel
    # take it from one constant, so exactly one value is buildable.
    return tk == BFP16_K_CHUNK_DEFAULT


def assert_layout_buildable(tile_n, tile_k_l1, what):
    if not layout_is_buildable(tile_n, tile_k_l1):
        raise Bfp16LayoutError(
            f"{what} names tile_n={tile_n} / tile_k_l1={tile_k_l1}, which this "
            f"build cannot produce: tile_n must be one of "
            f"{BFP16_TILE_N_SUPPORTED} and tile_k_l1 must be "
            f"{BFP16_K_CHUNK_DEFAULT}. A layout nobody can build is not "
            f"evidence about any ELF -- refusing."
        )
    return int(tile_n), int(tile_k_l1)


def derive_layout_from_packed(packed, dense_shape):
    """FACT A, for one buffer: solve the layout out of the array itself.

    `packed.shape == (N // tile_n, K // tile_k_l1, tile_n * tile_k_l1 // 8 * 9)`
    against a dense `(K, N)`, so both tiles are determined and the third axis
    is an independent cross-check. Returns (tile_n, tile_k_l1); raises if the
    array is not a `pack_b_bfp16ebs8` product of that dense array.

    WHAT THIS GUARANTEES AND WHAT IT DOES NOT. The record axis pins the PRODUCT
    `tile_n * tile_k_l1`; the split between the two comes from the dense shape.
    So this is exact for the array the buffer was actually packed from, and it
    is NOT a transpose detector: given a transposed dense shape it returns a
    different, internally consistent pair. That pair then fails
    `layout_is_buildable` at the comparison, which is where such a case is
    caught -- but the derivation alone does not claim to catch it.
    """
    import numpy as _np

    a = _np.asarray(packed)
    if a.ndim != 3 or a.dtype != _np.uint8:
        raise Bfp16LayoutError(
            f"packed weight buffer is {a.ndim}-D {a.dtype}, not the 3-D uint8 "
            f"`pack_b_bfp16ebs8` emits -- its layout cannot be derived.")
    K, N = int(dense_shape[0]), int(dense_shape[1])
    nb, kb, tile_bytes = (int(x) for x in a.shape)
    if nb <= 0 or kb <= 0 or N % nb or K % kb:
        raise Bfp16LayoutError(
            f"packed buffer {a.shape} does not tile a dense ({K}, {N}) -- "
            f"layout underivable.")
    tile_n, tile_k_l1 = N // nb, K // kb
    expect = tile_n * tile_k_l1 // BFP16_BLOCK * BFP16_BYTES_PER_BLOCK
    if tile_bytes != expect:
        raise Bfp16LayoutError(
            f"packed buffer's record axis is {tile_bytes} bytes but "
            f"tile_n={tile_n} / tile_k_l1={tile_k_l1} over a dense ({K}, {N}) "
            f"requires {expect}. The buffer is not what its own shape claims "
            f"-- refusing.")
    return tile_n, tile_k_l1


def derive_packed_layout(weights, packed_layers=None, fields=BFP16_PREFILL_FIELDS):
    """FACT A, for a whole model: the ONE layout every packed buffer is in.

    Every (layer, field) pair is solved independently and they must agree. A
    layer repacked or swapped at another width is therefore a REFUSAL, not a
    stale record -- which is what a stamp beside the weights could not do.
    """
    seen = {}
    n_checked = 0
    for i, layer in enumerate(getattr(weights, "layers", []) or []):
        packed = (packed_layers[i] if packed_layers is not None
                  else getattr(layer, "_bfp_packed", None))
        if not packed:
            continue
        for f in fields:
            if f not in packed:
                continue
            dense = getattr(layer, f, None)
            if dense is None or getattr(dense, "shape", None) is None:
                raise Bfp16LayoutError(
                    f"layer {i} field {f!r} has a packed buffer but no dense "
                    f"array to derive its layout against -- refusing.")
            seen.setdefault(derive_layout_from_packed(packed[f], dense.shape),
                            []).append((i, f))
            n_checked += 1
    if not n_checked:
        raise Bfp16LayoutError(
            "no packed bfp16 weight buffers were found on these weights, so "
            "the layout they are in cannot be derived -- refusing rather than "
            "assuming one.")
    if len(seen) > 1:
        detail = "; ".join(f"{k}: {len(v)} buffers e.g. {v[0]}" for k, v in seen.items())
        raise Bfp16LayoutError(
            f"the packed weight buffers are not all in one layout ({detail}). "
            f"One ELF consumes all of them -- refusing.")
    return next(iter(seen))


def read_declared_layout(cache_dir):
    """FACT B as recorded, or None when the set declares nothing usable.

    Reading only, and deliberately strict about what counts as a declaration:
    a file that does not name a BUILDABLE layout is not one.
    """
    import json
    from pathlib import Path

    if cache_dir is None:
        return None
    path = Path(cache_dir) / BFP16_GEOMETRY_SIDECAR
    if not path.is_file():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    if not isinstance(d, dict):
        return None
    if not layout_is_buildable(d.get("tile_n"), d.get("tile_k_l1")):
        return None
    return d


#: The ELFs a bfp16 prefill artifact set is made of. Their presence is what
#: makes a set "already built" for `assert_can_build_into`.
BFP16_SET_ELFS = ("rms_gemms_rope_bfp16.elf", "o_ffn_bfp16.elf")


def assert_can_build_into(cache_dir, tile_n, tile_k_l1):
    """CREATE-path preflight: may this invocation add width-`tile_n` ELFs here?

    `[2026-08-27, final review of queue item 22]` The write below used to be
    unconditional, which turned the guard into a RUBBER STAMP on the one case
    it exists to catch: with 32-wide ELFs already cached and the width set to
    128, both compiles are skipped as "already built", the declaration is
    rewritten to 128, the host then packs at 128, the guard reads its own new
    label and admits -- and the unchanged 32-wide ELFs silently consume
    128-permuted bytes of the same length. That is worse than no declaration.

    A set may be built into only when it is EMPTY of bfp16 ELFs, or already
    declares exactly this layout. Anything else refuses, including a set whose
    ELFs exist but declare nothing (their width is unknown, so adding to it
    would produce a mixed set nobody can label) and a PARTIALLY populated set at
    another width.

    Returns True when this invocation may write the declaration afterwards.
    """
    from pathlib import Path

    tn, tk = assert_layout_buildable(tile_n, tile_k_l1, "this compile")
    d = read_declared_layout(cache_dir)
    present = [e for e in BFP16_SET_ELFS if Path(cache_dir, e).is_file()]
    if d is not None:
        built = (int(d["tile_n"]), int(d["tile_k_l1"]))
        if built != (tn, tk):
            raise Bfp16LayoutError(
                f"artifact set {cache_dir} already declares tile_n={built[0]} / "
                f"tile_k_l1={built[1]}"
                + (f" and holds {', '.join(present)}" if present else "")
                + f", but this compile is at tile_n={tn} / tile_k_l1={tk}. An "
                f"existing set's declaration is READ-ONLY -- relabelling it "
                f"would leave ELFs of one width labelled as another, which is "
                f"silent. Build into an empty directory, or set "
                f"{BFP16_TILE_N_ENV}={built[0]}.")
        return True
    if present:
        raise Bfp16LayoutError(
            f"artifact set {cache_dir} already holds {', '.join(present)} but "
            f"declares no layout, so the width those ELFs were built at is "
            f"unknown. Adding tile_n={tn} ELFs would make it a mixed set with "
            f"one label. Refusing: build into an empty directory, or record the "
            f"existing set's width first with `write_declared_layout` if it can "
            f"be established.")
    return True


def write_declared_layout(cache_dir, tile_n, tile_k_l1, why=None,
                          allow_replace=False, **extra):
    """Record FACT B. Called by whatever COMPILES a set -- the one act
    authorised to establish it -- and once by hand per set that predates the
    declaration, where `why` is the evidence for the recorded width.

    An existing declaration is READ-ONLY: re-recording the SAME layout is
    idempotent, and re-recording a different one refuses unless the caller
    explicitly passes `allow_replace` (which only the person who just rebuilt
    the set's ELFs is entitled to do).
    """
    import json
    from pathlib import Path

    tn, tk = assert_layout_buildable(tile_n, tile_k_l1, "this declaration")
    prev = read_declared_layout(cache_dir)
    if prev is not None and not allow_replace:
        was = (int(prev["tile_n"]), int(prev["tile_k_l1"]))
        if was != (tn, tk):
            raise Bfp16LayoutError(
                f"artifact set {cache_dir} already declares tile_n={was[0]} / "
                f"tile_k_l1={was[1]}; refusing to relabel it as {tn} / {tk}. "
                f"The declaration describes ELFs that are already there -- "
                f"rewriting it does not rebuild them, it only makes the guard "
                f"agree with the wrong thing.")
    d = {"tile_n": tn, "tile_k_l1": tk}
    if why:
        d["why"] = why
    d.update(extra)
    Path(cache_dir, BFP16_GEOMETRY_SIDECAR).write_text(
        json.dumps(d, indent=2), encoding="utf-8")
    return d


def assert_layout_agrees(weights, cache_dir, packed_layers=None):
    """THE guard. Call it once, immediately before the first dispatch that
    consumes a packed weight BO.

    Licenses a "yes" only when: fact A is derivable from the buffers, fact B is
    declared by the set AND buildable, and the two are equal. Every other case
    -- unnamed set, undeclared set, unbuildable declaration, underivable or
    self-inconsistent buffers, buffers in more than one layout, A != B --
    refuses, because the failure it guards against is silent: the packed BO is
    1.125 B/elt at every width, so nothing downstream can catch a mismatch and
    the only symptom is a wrong token.
    """
    packed_tn, packed_tk = derive_packed_layout(weights, packed_layers)
    if cache_dir is None:
        raise Bfp16LayoutError(
            f"the weights are packed for tile_n={packed_tn} / "
            f"tile_k_l1={packed_tk}, but no artifact set was named, so what "
            f"the ELF expects cannot be read -- refusing.")
    d = read_declared_layout(cache_dir)
    if d is None:
        raise Bfp16LayoutError(
            f"artifact set {cache_dir} does not declare a buildable tile "
            f"layout (no usable {BFP16_GEOMETRY_SIDECAR}), so what its ELFs "
            f"expect cannot be read -- and the weights beside it are packed "
            f"for tile_n={packed_tn} / tile_k_l1={packed_tk}. A wrong guess is "
            f"silent (same BO size at every width). Refusing. A set COMPILED "
            f"from now on declares this itself; for an older set, record it "
            f"from whatever establishes its width with "
            f"`awq_bfp_pack.write_declared_layout(<set>, <tile_n>, "
            f"{BFP16_K_CHUNK_DEFAULT}, why=...)`.")
    built = (int(d["tile_n"]), int(d["tile_k_l1"]))
    if built != (packed_tn, packed_tk):
        raise Bfp16LayoutError(
            f"LAYOUT MISMATCH: the weights are packed for tile_n={packed_tn} / "
            f"tile_k_l1={packed_tk}, the ELFs in {cache_dir} were built for "
            f"tile_n={built[0]} / tile_k_l1={built[1]}. Same BO size either "
            f"way, so this would be silent -- refusing. Set "
            f"{BFP16_TILE_N_ENV}={built[0]} and re-pack, or point at a set "
            f"built for the packed width.")
    return d


#: Elements sharing one exponent in `bfp16ebs8` -- the format's own group.
#: Imported from the packer above rather than re-declared: `[2026-08-27]` the
#: layout derivation solves against these two numbers, so a second copy of them
#: is a second answer waiting to drift.


def pack_layer_bfp16(layer, fields=BFP16_PREFILL_FIELDS, n_tile=None,
                     k_chunk=None):
    """`{field: bfp16ebs8 uint8 BO}` from a `LayerWeights`' DENSE bf16 fields.

    The e2e driver's loader (`llama32_1b_int4_weights.load_weights_awq`) has
    already dequantized every AWQ projection to dense bf16 `[in, out]` -- the
    array the bf16 prefill stitchers consume. This transcodes THAT array, so
    both prefill arms compute over bit-identical weights and the HF reference
    the verify gate builds is the reference for both. There is no second
    dequantization and no second scale: in `bfp16ebs8` the block's shared
    8-bit exponent IS the scale, applied by the MMUL itself.
    """
    n_tile = BFP16_N_TILE if n_tile is None else n_tile
    k_chunk = BFP16_K_CHUNK if k_chunk is None else k_chunk
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
            f"BO is uint8 [N/{BFP16_N_TILE}, K/{BFP16_K_CHUNK}, tile_bytes] -- the "
            f"builders' shared tile_n={BFP16_N_TILE} / tile_k_l1={BFP16_K_CHUNK} "
            "geometry, which the ELF bakes into mm_bf16_x_bfp16.o's -DDIM_N and the "
            "packer must match exactly (matmul_bf16_x_bfp16.pack_b_bfp16ebs8)"
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
