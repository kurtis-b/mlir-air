# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""llama.cpp GGUF container reader for the int4-AWQ GEMV bridge.

Parses a GGUF (v2/v3) file -- header, metadata key/values, tensor infos with
absolute byte offsets -- and exposes each tensor's on-disk payload as a
zero-copy `np.memmap`.

On top of the reader, the q4_0 codec repacks a payload into the int4-AWQ
GEMV's `(A_q, A_s, A_z)` planes (`matvec_int4_packed.pack_inputs` form) with
no kernel change: q4_0's `d * (q - 8)` is the kernel's `(q - z) * s` with the
zero point pinned to 8 and the group size to 32, so
  * `Z` is an all-8s plane (`q4_0_traffic_bytes` counts what it costs),
  * `S` is q4_0's per-block `d`, fp16 -> bf16 (`scale_rounding_error`),
  * `Q` is un-interleaved: q4_0 packs elements `j` and `j+16` of a 32-block
    into byte `j`; the kernel's `Q` is row-major (byte `b` = elements `2b`,
    `2b+1`).
`llama_unpermute_rows` undoes llama.cpp's RoPE row permute of q/k.
`q4_0_payload_for` is the packer's one entry point: Q4_0 bytes as stored;
tensors llama.cpp promoted to q4_1 (fractional zero point, not in the
kernel's uint8 Z plane) re-quantised from the source weights, never
transcoded silently -- the attributed decision record below says why.
`--self-test` proves the codec, the repack, the un-permute and that route.

No dependency on the `gguf` Python package: the container is simple enough to
parse directly, and no package may be installed while gates run.

Self-test (synthetic, no checkpoint; leg (d) imports `matvec_int4_packed`):
    python3 gguf_q4_0.py --self-test
Inspect a checkpoint (`--gguf` defaults to `$SMOLLM2_GGUF`):
    python3 gguf_q4_0.py --gguf <file.gguf> --list   # tensors + payload check
    python3 gguf_q4_0.py --gguf <file.gguf> --meta   # metadata KVs
"""

import argparse
import os
import struct
import sys
from collections import Counter

import numpy as np
from ml_dtypes import bfloat16

GGUF_MAGIC = b"GGUF"

# gguf_metadata_value_type -> struct format (8 = string, 9 = array, handled
# separately): u8 i8 u16 i16 u32 i32 f32 bool ... u64 i64 f64.
_T_STRING, _T_ARRAY = 8, 9
_FMTS = "<B <b <H <h <I <i <f <? <Q <q <d".split()
_SCALAR_FMT = dict(zip((0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12), _FMTS))

# ggml_type -> (name, block_size, type_size). Only the types this bridge needs
# to *identify*; only Q4_0 and F32/F16 can actually be decoded here.
GGML_TYPES = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),  # struct { fp16 d; uint8 qs[16]; }
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
}

GGML_TYPE_Q4_0 = 2
QK4_0 = 32


class TensorInfo:
    """One GGUF tensor_info record, plus its resolved absolute byte offset."""

    __slots__ = ("name", "ne", "ggml_type", "rel_offset", "abs_offset", "nbytes")

    def __init__(self, name, ne, ggml_type, rel_offset, data_start):
        self.name = name
        self.ne = ne  # ne[0] is the fastest-moving (contiguous) dimension
        self.ggml_type = ggml_type
        self.rel_offset = rel_offset
        self.abs_offset = data_start + rel_offset
        _, blck, tsize = GGML_TYPES.get(ggml_type, ("?", 1, 0))
        self.nbytes = int(np.prod(ne, dtype=np.int64)) // blck * tsize

    @property
    def type_name(self):
        return GGML_TYPES.get(self.ggml_type, ("UNKNOWN_%d" % self.ggml_type,))[0]


class GGUFFile:
    """Minimal read-only GGUF parser: header, metadata KVs, tensor infos.

    Payloads are read lazily with `np.memmap`, so opening a 1 GB checkpoint
    costs nothing. Raises `ValueError` on a bad magic, an unsupported version,
    or a header cut short.
    """

    def __init__(self, path):
        self.path = path
        self.metadata = {}
        self.tensors = {}
        self._parse()

    def _read(self, f, n):
        raw = f.read(n)
        if len(raw) != n:
            raise ValueError("truncated GGUF header at byte %d" % f.tell())
        return raw

    def _u32(self, f):
        return struct.unpack("<I", self._read(f, 4))[0]

    def _u64(self, f):
        return struct.unpack("<Q", self._read(f, 8))[0]

    def _string(self, f):
        return self._read(f, self._u64(f)).decode("utf-8", errors="replace")

    def _value(self, f, vtype):
        if vtype == _T_STRING:
            return self._string(f)
        if vtype == _T_ARRAY:
            elem_type, n = self._u32(f), self._u64(f)
            if elem_type == _T_STRING:
                return [self._string(f) for _ in range(n)]
            fmt = _SCALAR_FMT[elem_type]
            raw = self._read(f, n * struct.calcsize(fmt))
            return [v for (v,) in struct.iter_unpack(fmt, raw)]
        fmt = _SCALAR_FMT[vtype]
        return struct.unpack(fmt, self._read(f, struct.calcsize(fmt)))[0]

    def _parse(self):
        with open(self.path, "rb") as f:
            magic = f.read(4)
            if magic != GGUF_MAGIC:
                raise ValueError("not a GGUF file: magic=%r" % magic)
            self.version = self._u32(f)
            # v1 has 32-bit counts; anything above v3 has a layout this reader
            # has not seen -- parsing it as v3 would return wrong offsets.
            if self.version not in (2, 3):
                raise ValueError(
                    "unsupported GGUF v%d (this reader knows v2 and v3)" % self.version
                )
            n_tensors = self._u64(f)
            n_kv = self._u64(f)
            for _ in range(n_kv):
                key = self._string(f)
                self.metadata[key] = self._value(f, self._u32(f))
            raw_infos = []
            for _ in range(n_tensors):
                name = self._string(f)
                ne = [self._u64(f) for _ in range(self._u32(f))]
                raw_infos.append((name, ne, self._u32(f), self._u64(f)))
            # Tensor data starts at the header end rounded up to the alignment
            # (a tensor-less file, e.g. a vocab-only GGUF, ends unpadded).
            self.alignment = self.metadata.get("general.alignment", 32)
            self.header_end = f.tell()
            self.data_start = -(-self.header_end // self.alignment) * self.alignment
        for name, ne, ggml_type, offset in raw_infos:
            self.tensors[name] = TensorInfo(
                name, ne, ggml_type, offset, self.data_start
            )

    def raw_bytes(self, name):
        """The tensor's on-disk payload as a uint8 memmap (no copy)."""
        ti = self.tensors[name]
        return np.memmap(self.path, np.uint8, "r", ti.abs_offset, (ti.nbytes,))

    def check_payloads(self):
        """Every payload memmap has its `nbytes`, none overlap, and the last one
        ends at the file size (up to one alignment pad); a wrong `data_start`
        or a truncated file raises `ValueError`. Returns the data end offset."""
        end = self.header_end
        for ti in sorted(self.tensors.values(), key=lambda t: t.abs_offset):
            if ti.abs_offset < end or self.raw_bytes(ti.name).shape[0] != ti.nbytes:
                raise ValueError("payload of %s overlaps or is short" % ti.name)
            end = ti.abs_offset + ti.nbytes
        size = os.path.getsize(self.path)
        if not end <= size < end + self.alignment:
            raise ValueError("payloads end at %d, file size is %d" % (end, size))
        return end

    def architecture(self):
        return self.metadata.get("general.architecture", "?")


# ---------------------------------------------------------------------------
# q4_0 codec -> the kernel's (A_q, A_s, A_z)
# ---------------------------------------------------------------------------


def q4_0_blocks(raw, n_blocks):
    """Split a q4_0 payload into its per-block `d` (fp16) and `qs` (16 uint8).

    Returns (d[n_blocks] float16 view, qs[n_blocks, 16] uint8).
    """
    if raw.size != n_blocks * 18:
        raise ValueError("q4_0 payload %d B != %d blocks x 18 B" % (raw.size, n_blocks))
    blocks = np.asarray(raw).reshape(n_blocks, 18)
    # `d` is little-endian fp16 in the first two bytes of each 18-byte block.
    d = blocks[:, 0:2].copy().view(np.float16).reshape(n_blocks)
    qs = blocks[:, 2:18]
    return d, qs


def q4_0_nibbles(qs):
    """Un-interleave q4_0's nibble order into plain element order.

    q4_0 stores element `j` in the low nibble of byte `j` and element `j+16` in
    the high nibble, for j in [0, 16). Returns uint8 [n_blocks, 32] in element
    order, values in [0, 16).
    """
    n_blocks = qs.shape[0]
    out = np.empty((n_blocks, QK4_0), dtype=np.uint8)
    out[:, 0:16] = qs & 0x0F
    out[:, 16:32] = (qs >> 4) & 0x0F
    return out


def repack_q4_0_linear(raw, K, M):
    """A q4_0 tensor payload -> (A_q, A_s, A_z) in mlir-air `pack_inputs` form.

    Args:
        raw: uint8 payload of a GGUF q4_0 tensor with ne = [K, M]; blocks run
             along the contiguous K axis, so row `m` owns blocks
             [m*K/32, (m+1)*K/32).
        K:   reduction dim (in_features)  == ne[0]
        M:   output dim   (out_features)  == ne[1]

    Returns:
        A_q: uint8 [M, K/2]  -- byte b of row m holds elements 2b (low nibble)
             and 2b+1 (high nibble), matching the kernel's row-major Q.
        A_s: bf16  [K/32, M] -- q4_0's per-block `d`, fp16 -> bf16.
        A_z: uint8 [K/32, M] -- all 8s (q4_0 is `d * (q - 8)`).
    """
    if K % QK4_0 != 0:
        raise ValueError("K=%d is not a multiple of the q4_0 block size 32" % K)
    bpr = K // QK4_0  # blocks per row
    d, qs = q4_0_blocks(raw, M * bpr)

    # Elements in plain order, [M, K].
    nibs = q4_0_nibbles(qs).reshape(M, K)

    # Re-pack to the kernel's pairing: byte b <- (e[2b] | e[2b+1] << 4).
    A_q = (nibs[:, 0::2] | (nibs[:, 1::2] << 4)).astype(np.uint8)

    # d is [M, bpr] in memory; the kernel wants [n_groups, M].
    A_s = np.ascontiguousarray(d.reshape(M, bpr).T).astype(bfloat16)
    A_z = np.full((bpr, M), 8, dtype=np.uint8)
    return A_q, A_s, A_z


# ---------------------------------------------------------------------------
# q4_1 -> q4_0, because the kernel's zero-point is an INTEGER
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS, AND WHY IT IS NOT A REPACK LIKE EVERYTHING ABOVE.
#
# llama.cpp promotes a few tensors to `q4_1` when symmetric q4_0 fits them
# badly. In `bartowski/SmolLM2-1.7B-Instruct-Q4_0.gguf` exactly three do --
# `blk.{0,1,10}.ffn_down.weight` -- against 165 pure q4_0 linears (`--list`
# re-derives the histogram; `promoted_tensors` names them).
#
# q4_1 dequantises as `w = d*q + m` (two fp16 per 32-element block: a scale and
# a MINIMUM). The device kernel computes `(q - z) * s`. Matching the two:
#
#     d*q + m = s*q - s*z   =>   s = d,  z = -m/d
#
# and `-m/d` is a real number, while the kernel's zero-point plane is
# `uint8_t *a_z` (`mv_int4_bf16.cc`) -- an INTEGER. So q4_1 is not
# representable in this kernel's form, and no repack can make it so. That is a
# statement about the kernel's signature, not about effort.
#
# Three routes were considered:
#
#   (a) Round `z` to the nearest integer. REJECTED, and this is the one that
#       looks cheapest and is worst. The residual is a per-group ADDITIVE
#       constant of up to d/2 on every weight in the group, and a GEMV sums
#       over the group -- so it does not cancel like rounding noise, it
#       accumulates as `shift * sum(b)`. Systematically biased, and biased per
#       group, which is the shape of an error that survives averaging.
#
#   (b) Run those three `ffn_down` in bf16. Exact, but it makes the decode path
#       mixed-dtype for 3 of 168 linears, which costs a second builder
#       instantiation to avoid a 1.8% traffic difference.
#
#   (c) TRANSCODE q4_1 -> q4_0 by dequantising and re-quantising with
#       llama.cpp's own rule, on the argument that the result lands in the same
#       error class as the 165 tensors already quantised that way. PROPOSED
#       HERE AND THEN REFUSED BY ITS OWN MEASUREMENT -- see the table below. It
#       does not land in that class; it lands about a third outside it, because
#       re-quantising already-quantised data compounds two roundings. q4_0's
#       scale is set from the element of largest magnitude divided by -8, so
#       the round trip is exact only when a block contains a `q = 0` element;
#       otherwise the grid shifts (measured on already-q4_0 tensors:
#       rms/rms 3.6e-2 added by a transcode that should be free).
#
#   (d) QUANTISE THE THREE FROM THE ORIGINAL bf16 WEIGHTS instead of from the
#       checkpoint's q4_1 -- one rounding, not two. SHIPS.
#
# The figures here were measured on the study branch, commit 8d67c1f3
# (2026-08-14), on `bartowski/SmolLM2-1.7B-Instruct-Q4_0.gguf` vs the bf16
# `HuggingFaceTB/SmolLM2-1.7B-Instruct` weights; not re-measured on main.
# They are re-derivable with `requantization_error` (the transcode arm) and
# `quantize_q4_0` of the HF weights (the source arm); the mechanism -- two
# roundings lose to one -- is reproduced synthetically by self-test leg (l).
# rms error over rms weight:
#
#     the pure-q4_0 ffn_down tensors (the error llama.cpp ACCEPTED)
#                                              0.0828 .. 0.0853
#     the three as q4_1 (why they were promoted) 0.0720 .. 0.0729
#     (c) transcoded q4_1 -> q4_0                0.1109 .. 0.1124   OUT of family
#     (d) quantised from the bf16 source         0.0869 .. 0.0884   in family
#
# So (d) keeps ONE uniform int4 path with no kernel and no builder change --
# the property the whole q4_0 bridge exists to preserve -- and pays about 4%
# more error than the average accepted tensor rather than 32% more. It needs
# the bf16 weights at pack time, which costs nothing: the pipeline already
# loads them for the tied embedding / lm_head, which is Q6_K in the checkpoint
# and never consumed from it (`awq_pack.py:270-277` does the same for
# Llama-3.2-1B).
#
# `requantize_q4_1_to_q4_0` is KEPT rather than deleted. It is the refused
# route, and the comparison above is only checkable because both are here.


def q4_1_blocks(raw, n_blocks):
    """Split a q4_1 payload into per-block `d`, `m` (both fp16) and `qs`.

    A `block_q4_1` is 20 bytes: fp16 delta, fp16 min, then the same 16 packed
    bytes q4_0 uses with the same nibble interleave.
    """
    if raw.size != n_blocks * 20:
        raise ValueError("q4_1 payload %d B != %d blocks x 20 B" % (raw.size, n_blocks))
    blocks = np.asarray(raw).reshape(n_blocks, 20)
    d = blocks[:, 0:2].copy().view(np.float16).reshape(n_blocks)
    m = blocks[:, 2:4].copy().view(np.float16).reshape(n_blocks)
    qs = blocks[:, 4:20]
    return d, m, qs


def dequant_q4_1_reference(raw, K, M):
    """Exact q4_1 dequantisation to float32 [M, K]. `w = d*q + m`."""
    if K % QK4_0 != 0:
        raise ValueError("K=%d is not a multiple of the block size 32" % K)
    bpr = K // QK4_0
    d, m, qs = q4_1_blocks(raw, M * bpr)
    nibs = q4_0_nibbles(qs).astype(np.float32)
    w = nibs * d.astype(np.float32)[:, None] + m.astype(np.float32)[:, None]
    return w.reshape(M, K)


def quantize_q4_0(w):
    """float32 [n_blocks, 32] -> a q4_0 payload, by llama.cpp's own rule.

    Transcribed from `quantize_row_q4_0_ref`: the scale is set from the element
    of LARGEST MAGNITUDE (signed, divided by -8, so the sign convention puts
    that element at nibble 0 or 15), and each element is
    ``min(15, floor(x/d + 8.5))``. The C cast `(int8_t)(x0 + 8.5f)` truncates
    toward zero and its argument is non-negative over the representable range,
    so `floor` is the faithful spelling.

    Deviating from this rule -- rounding differently, or fitting the scale by
    least squares -- would produce a tensor that is not what llama.cpp would
    have produced for the same weights, and the point of re-quantising is to
    land in the SAME error class as the checkpoint's other 165 tensors.
    """
    w = np.asarray(w, dtype=np.float32)
    n_blocks = w.shape[0]
    idx = np.argmax(np.abs(w), axis=1)
    amax_signed = w[np.arange(n_blocks), idx]
    d = (amax_signed / -8.0).astype(np.float32)
    d16 = d.astype(np.float16)  # stored as fp16, so quantise against that
    dq = d16.astype(np.float32)
    inv = np.where(dq != 0, 1.0 / np.where(dq != 0, dq, 1.0), 0.0)

    q = np.floor(w * inv[:, None] + 8.5)
    q = np.clip(q, 0, 15).astype(np.uint8)

    out = np.zeros((n_blocks, 18), dtype=np.uint8)
    out[:, 0:2] = d16.view(np.uint8).reshape(n_blocks, 2)
    out[:, 2:18] = q[:, 0:16] | (q[:, 16:32] << 4)
    return out.reshape(-1)


def requantize_q4_1_to_q4_0(raw, K, M):
    """A q4_1 tensor payload -> an equivalent q4_0 payload. Lossy; measured.

    Returns the q4_0 payload only. Use ``requantization_error`` for what it
    cost -- the two are separate so a caller cannot get the bytes without the
    error being computable from the same inputs.
    """
    w = dequant_q4_1_reference(raw, K, M)
    return quantize_q4_0(w.reshape(-1, QK4_0))


def requantization_error(raw_q4_1, K, M):
    """What re-quantising this tensor cost, as numbers rather than adjectives.

    Compares the q4_0 round trip against the EXACT q4_1 dequantisation, and
    also reports q4_1's own distance from... nothing, because the original
    float weights are not in the checkpoint. That bound is the honest one: this
    is the error ADDED on top of whatever q4_1 already carried, not a total.
    """
    exact = dequant_q4_1_reference(raw_q4_1, K, M)
    requant = requantize_q4_1_to_q4_0(raw_q4_1, K, M)
    got = dequant_q4_0_reference(requant, K, M)

    a, b = exact.reshape(-1), got.reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    cos = float(np.dot(a, b) / denom) if denom else 1.0
    scale = float(np.max(np.abs(a))) or 1.0
    return {
        "cosine": cos,
        "max_abs_err": float(np.max(np.abs(a - b))),
        "max_err_over_tensor_max": float(np.max(np.abs(a - b)) / scale),
        "rms_err": float(np.sqrt(np.mean((a - b) ** 2))),
        "rms_over_rms": float(
            np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(a**2)) or 1.0)
        ),
    }


def q4_0_payload_for(g, name, reference=None):
    """The q4_0 payload for `name`, whatever the checkpoint stored it as.

    The one entry point a model packer should call, so the promoted-tensor
    policy lives in exactly one place instead of at every call site.

      * already `q4_0` -> the checkpoint's own bytes, untouched.
      * promoted (`q4_1` and anything else) -> quantised from `reference`,
        which must be the ORIGINAL float weights for this tensor, shaped
        [out_features, in_features] as HF stores them (== [M, K] here).

    Raises rather than transcoding when a promoted tensor arrives with no
    reference: route (c) in the record above compounds two roundings, and
    falling back to it silently would degrade three layers of the model in a
    way nothing downstream could see -- the packer would succeed and the
    tokens would just be slightly worse.

    Returns (payload, provenance) where provenance is "checkpoint" or
    "quantized_from_reference", so a caller can report which tensors it had to
    re-derive instead of asserting it did not.
    """
    info = g.tensors[name]
    kind = GGML_TYPES.get(info.ggml_type, ("?",))[0]
    if kind == "Q4_0":
        return np.asarray(g.raw_bytes(name)), "checkpoint"
    if reference is None:
        raise ValueError(
            f"{name} is {kind}, not Q4_0, and no reference weights were given. "
            f"The kernel's zero-point plane is uint8 (mv_int4_bf16.cc), so "
            f"{kind}'s fractional zero-point is not representable, and "
            "transcoding compounds two roundings (the route-(c) record in "
            "gguf_q4_0.py). Pass the original bf16 weights for this tensor."
        )
    K, M = info.ne[0], info.ne[1]
    ref = np.asarray(reference, dtype=np.float32)
    if ref.shape != (M, K):
        raise ValueError(
            f"{name}: reference has shape {ref.shape}, expected {(M, K)} "
            "([out_features, in_features], as HF stores it)"
        )
    return quantize_q4_0(ref.reshape(-1, QK4_0)), "quantized_from_reference"


def promoted_tensors(g):
    """Every 2-D tensor the checkpoint did NOT store as q4_0, with its type.

    The tensors a packer needs reference weights for (`q4_0_payload_for`).
    """
    out = {}
    for name, info in g.tensors.items():
        if len(info.ne) != 2:
            continue
        kind = GGML_TYPES.get(info.ggml_type, ("?",))[0]
        if kind != "Q4_0":
            out[name] = kind
    return out


def llama_unpermute_rows(w, n_head):
    """Undo llama.cpp's RoPE row permutation on a [out_features, in_features] matrix.

    `convert_hf_to_gguf.py`'s `LlamaModel.permute` reorders the rows of
    `q_proj` and `k_proj` so that llama.cpp's RoPE can treat each head's
    `head_dim` as two contiguous halves instead of HF's interleaved pairs:

        W.reshape(n_head, 2, head_dim//2, in).swapaxes(1, 2).reshape(out, in)

    This is its inverse. It is not optional: a packer that skips it feeds the
    kernel a row-shuffled Q/K projection (self-test leg (f) checks the inverse
    against the forward transform above).

    Apply with `n_head` for q_proj and `n_kv_head` for k_proj. Not applied to
    v/o/gate/up/down, which llama.cpp stores unpermuted.
    """
    out, inn = w.shape
    if out % n_head != 0:
        raise ValueError("out=%d not divisible by n_head=%d" % (out, n_head))
    head_dim = out // n_head
    if head_dim % 2 != 0:
        raise ValueError("head_dim=%d must be even" % head_dim)
    return w.reshape(n_head, head_dim // 2, 2, inn).swapaxes(1, 2).reshape(out, inn)


def unpermute_planes(A_q, A_s, A_z, n_head):
    """`llama_unpermute_rows` on a repacked triple: the permutation reorders
    OUTPUT rows, so it moves A_q's rows and the M axis of the per-group S/Z
    planes together (self-test leg (g))."""
    return (
        np.ascontiguousarray(llama_unpermute_rows(A_q, n_head)),
        np.ascontiguousarray(llama_unpermute_rows(A_s.T, n_head).T),
        np.ascontiguousarray(llama_unpermute_rows(A_z.T, n_head).T),
    )


def dequant_q4_0_reference(raw, K, M, scale_dtype="fp16"):
    """Direct fp32 dequant of a q4_0 tensor, independent of the repack path.

    `scale_dtype="fp16"` reproduces llama.cpp exactly (`d` used at fp16
    precision). `scale_dtype="bf16"` rounds `d` to bf16 first, which is what
    the NPU packed BO actually carries. The difference between the two is the
    packer's scale-precision loss and nothing else.

    Returns float32 [M, K].
    """
    bpr = K // QK4_0
    d, qs = q4_0_blocks(raw, M * bpr)
    nibs = q4_0_nibbles(qs).astype(np.int32)  # [M*bpr, 32]
    if scale_dtype == "bf16":
        dd = d.astype(bfloat16).astype(np.float32)
    elif scale_dtype == "fp16":
        dd = d.astype(np.float32)
    else:
        raise ValueError("scale_dtype must be 'fp16' or 'bf16'")
    w = (nibs - 8) * dd[:, None]
    return w.reshape(M, K)


# ---------------------------------------------------------------------------
# Cost accounting (bytes, not adjectives)
# ---------------------------------------------------------------------------


def q4_0_traffic_bytes(K, M, gs=32, with_zeros=True):
    """Packed-BO bytes for one [M, K] weight, and the implied bits/weight.

    `with_zeros=False` models a symmetric variant that drops the all-8s Z
    plane entirely -- the difference between the two is pure redundant traffic.
    """
    n_groups = K // gs
    q = M * (K // 2)
    s = n_groups * M * 2
    z = n_groups * M if with_zeros else 0
    total = q + s + z
    return {
        "q_bytes": q,
        "s_bytes": s,
        "z_bytes": z,
        "total_bytes": total,
        "bits_per_weight": total * 8.0 / (M * K),
    }


def scale_rounding_error(d_fp16):
    """Relative error introduced by rounding q4_0's fp16 `d` to bf16.

    Every weight in a block inherits its block's scale error exactly, so this
    per-block statistic *is* the per-weight statistic.
    """
    a = np.asarray(d_fp16, dtype=np.float32)
    b = np.asarray(d_fp16).astype(bfloat16).astype(np.float32)
    nz = a != 0
    rel = np.zeros_like(a)
    rel[nz] = np.abs(b[nz] - a[nz]) / np.abs(a[nz])
    return {
        "n": int(a.size),
        "max_rel": float(rel.max()) if a.size else 0.0,
        "mean_rel": float(rel.mean()) if a.size else 0.0,
        "rms_rel": float(np.sqrt((rel**2).mean())) if a.size else 0.0,
        "exact_frac": float((rel == 0).mean()) if a.size else 0.0,
    }


# ---------------------------------------------------------------------------
# Self-test (synthetic; no checkpoint required)
# ---------------------------------------------------------------------------


def _synthesize_q4_0(K, M, seed=42):
    """Build a byte-exact q4_0 payload from known nibbles + scales."""
    rng = np.random.default_rng(seed)
    bpr = K // QK4_0
    n_blocks = M * bpr
    nibs = rng.integers(0, 16, size=(n_blocks, QK4_0), dtype=np.uint8)
    d = (rng.uniform(0.005, 0.02, size=n_blocks)).astype(np.float16)

    blocks = np.zeros((n_blocks, 18), dtype=np.uint8)
    blocks[:, 0:2] = d.view(np.uint8).reshape(n_blocks, 2)
    # Interleave back into q4_0's own order: byte j = e[j] | e[j+16] << 4.
    blocks[:, 2:18] = nibs[:, 0:16] | (nibs[:, 16:32] << 4)
    return blocks.reshape(-1), nibs, d


def self_test(K=256, M=64, seed=42, verbose=True):
    raw, nibs_ref, d_ref = _synthesize_q4_0(K, M, seed=seed)
    bpr = K // QK4_0

    # (a) block split recovers d and qs.
    d, qs = q4_0_blocks(raw, M * bpr)
    if not np.array_equal(d.view(np.uint16), d_ref.view(np.uint16)):
        raise AssertionError("q4_0 `d` round-trip mismatch")
    if verbose:
        print("  [a] q4_0 block split (d, qs): PASS (%d blocks)" % (M * bpr))

    # (b) nibble un-interleave recovers element order exactly.
    nibs = q4_0_nibbles(qs)
    if not np.array_equal(nibs, nibs_ref):
        wrong = int((nibs != nibs_ref).sum())
        raise AssertionError("nibble un-interleave: %d wrong" % wrong)
    if verbose:
        print("  [b] q4_0 nibble un-interleave: PASS (%d nibbles)" % nibs.size)

    # (c) repack -> the kernel's own Q convention recovers the same elements.
    A_q, A_s, A_z = repack_q4_0_linear(raw, K, M)
    if A_q.shape != (M, K // 2):
        raise AssertionError("A_q shape %s" % (A_q.shape,))
    if A_s.shape != (bpr, M) or A_z.shape != (bpr, M):
        raise AssertionError("A_s/A_z shape %s %s" % (A_s.shape, A_z.shape))
    if not np.all(A_z == 8):
        raise AssertionError("A_z is not all 8s")
    unpacked = np.zeros((M, K), dtype=np.uint8)
    unpacked[:, 0::2] = A_q & 0x0F
    unpacked[:, 1::2] = (A_q >> 4) & 0x0F
    if not np.array_equal(unpacked, nibs_ref.reshape(M, K)):
        raise AssertionError("A_q does not decode to the source nibbles")
    if verbose:
        print("  [c] repack -> kernel Q convention: PASS")

    # (d) the harness's own cpu_reference on (A_q, A_s, A_z) must equal a
    #     direct q4_0 dequant that never touches the repack path.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from matvec_int4_packed import cpu_reference  # type: ignore

    rng = np.random.default_rng(seed + 1)
    x = rng.standard_normal(K).astype(bfloat16)
    y_repacked = cpu_reference(A_q, A_s, A_z, x).astype(np.float32)
    w_direct = dequant_q4_0_reference(raw, K, M, scale_dtype="bf16")
    y_direct = (w_direct @ x.astype(np.float32)).astype(bfloat16).astype(np.float32)
    if not np.allclose(y_repacked, y_direct, rtol=1e-2, atol=1e-3):
        mx = float(np.max(np.abs(y_repacked - y_direct)))
        raise AssertionError("repack vs direct q4_0 dequant: max |d| = %g" % mx)
    if verbose:
        print("  [d] cpu_reference(repack) vs direct q4_0 dequant: PASS")

    # (e) negative control: a packer that forgets to un-interleave must FAIL
    #     (c). If this ever passes, the un-interleave check is vacuous.
    bad_q = (qs & 0x0F) | ((qs >> 4) << 4)  # == qs, i.e. "no un-interleave"
    bad_unpacked = np.zeros((M, K), dtype=np.uint8)
    bad_q_rows = bad_q.reshape(M, K // 2)
    bad_unpacked[:, 0::2] = bad_q_rows & 0x0F
    bad_unpacked[:, 1::2] = (bad_q_rows >> 4) & 0x0F
    if np.array_equal(bad_unpacked, nibs_ref.reshape(M, K)):
        raise AssertionError(
            "NEGATIVE CONTROL FAILED: the un-interleaved and raw orders agree, "
            "so test (c) proves nothing"
        )
    n_diff = int((bad_unpacked != nibs_ref.reshape(M, K)).sum())
    if verbose:
        print(
            "  [e] negative control (skip un-interleave): correctly DIFFERS "
            "on %d / %d nibbles" % (n_diff, nibs_ref.size)
        )

    # (f) llama_unpermute_rows inverts convert_hf_to_gguf's LlamaModel.permute.
    n_head = 4
    w = np.arange(M * 8, dtype=np.float32).reshape(M, 8)
    hd = M // n_head
    permuted = (
        w.reshape(n_head, 2, hd // 2, 8).swapaxes(1, 2).reshape(M, 8)
    )  # the forward transform, verbatim from convert_hf_to_gguf.py
    if np.array_equal(permuted, w):
        raise AssertionError(
            "NEGATIVE CONTROL FAILED: permute is a no-op at this shape, so "
            "test (f) proves nothing"
        )
    if not np.array_equal(llama_unpermute_rows(permuted, n_head), w):
        raise AssertionError("llama_unpermute_rows does not invert permute")
    if verbose:
        print("  [f] llama_unpermute_rows inverts llama.cpp's RoPE permute: PASS")

    # (g) the un-permute must move Q's rows and S/Z's M axis TOGETHER, or every
    #     row silently gets another row's scale. Dequantizing the permuted
    #     triple must equal un-permuting the dense dequant.
    A_qp, A_sp, A_zp = unpermute_planes(A_q, A_s, A_z, n_head)
    nib_p = np.zeros((M, K), dtype=np.uint8)
    nib_p[:, 0::2] = A_qp & 0x0F
    nib_p[:, 1::2] = (A_qp >> 4) & 0x0F
    got = (
        nib_p.astype(np.float32) - A_zp.astype(np.float32).T.repeat(QK4_0, axis=1)
    ) * (A_sp.astype(np.float32).T.repeat(QK4_0, axis=1))
    want = llama_unpermute_rows(
        dequant_q4_0_reference(raw, K, M, scale_dtype="bf16"), n_head
    )
    if not np.allclose(got, want, rtol=1e-3, atol=1e-6):
        raise AssertionError(
            "permuted repack disagrees with un-permuted dense dequant: max |d| = %g"
            % float(np.max(np.abs(got - want)))
        )
    if verbose:
        print("  [g] un-permute moves Q rows and S/Z together: PASS")

    # (k) q4_1 round trip: the block split and the exact dequant, against a
    #     payload built here so the arithmetic is checked rather than assumed.
    rng = np.random.default_rng(seed + 7)
    n_blk = M * bpr
    d41 = (rng.random(n_blk).astype(np.float32) * 0.05 + 0.01).astype(np.float16)
    m41 = ((rng.random(n_blk).astype(np.float32) - 0.5) * 0.2).astype(np.float16)
    q41 = rng.integers(0, 16, size=(n_blk, QK4_0), dtype=np.uint8)
    blk = np.zeros((n_blk, 20), dtype=np.uint8)
    blk[:, 0:2] = d41.view(np.uint8).reshape(n_blk, 2)
    blk[:, 2:4] = m41.view(np.uint8).reshape(n_blk, 2)
    blk[:, 4:20] = q41[:, 0:16] | (q41[:, 16:32] << 4)
    raw41 = blk.reshape(-1)

    want = (
        q41.astype(np.float32) * d41.astype(np.float32)[:, None]
        + m41.astype(np.float32)[:, None]
    )
    got = dequant_q4_1_reference(raw41, K, M).reshape(n_blk, QK4_0)
    if not np.allclose(want, got, rtol=0, atol=0):
        raise AssertionError("q4_1 dequant does not match d*q + m exactly")
    if verbose:
        print("  [k] q4_1 block split + exact `d*q + m` dequant: PASS")

    # (l) THE ONE THAT DECIDED THE ROUTE. Quantising from the float source must
    #     beat transcoding the q4_1, on the same weights. If this ever inverts,
    #     the comment block above is wrong and the packer should be revisited --
    #     so it is asserted, not described.
    #
    #     THE FIXTURE IS THE SUBTLE PART, and getting it wrong made this leg
    #     pass vacuously on first writing: if the source is generated AS
    #     `d*q + m` it is exactly q4_1-representable, the q4_1 step is lossless,
    #     and both routes are then byte-identical by construction (the two rms
    #     values come out equal). The transcode penalty exists only because
    #     q4_1 is ALREADY an approximation of the float source, so the source
    #     here is unconstrained noise and is quantised to q4_1 by llama.cpp's
    #     own rule (`d = (max-min)/15`, `m = min`).
    src = rng.standard_normal((n_blk, QK4_0)).astype(np.float32) * 0.05
    lo = src.min(axis=1)
    hi = src.max(axis=1)
    d_f = ((hi - lo) / 15.0).astype(np.float16)
    m_f = lo.astype(np.float16)
    dq_ = d_f.astype(np.float32)
    q_f = np.clip(
        np.round(
            (src - m_f.astype(np.float32)[:, None])
            / np.where(dq_ != 0, dq_, 1.0)[:, None]
        ),
        0,
        15,
    ).astype(np.uint8)
    blk2 = np.zeros((n_blk, 20), dtype=np.uint8)
    blk2[:, 0:2] = d_f.view(np.uint8).reshape(n_blk, 2)
    blk2[:, 2:4] = m_f.view(np.uint8).reshape(n_blk, 2)
    blk2[:, 4:20] = q_f[:, 0:16] | (q_f[:, 16:32] << 4)
    raw41b = blk2.reshape(-1)

    direct = dequant_q4_0_reference(quantize_q4_0(src), K, M).reshape(n_blk, QK4_0)
    transcoded = dequant_q4_0_reference(
        requantize_q4_1_to_q4_0(raw41b, K, M), K, M
    ).reshape(n_blk, QK4_0)

    def _rms(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    e_direct, e_trans = _rms(src, direct), _rms(src, transcoded)
    if not e_direct < e_trans:
        raise AssertionError(
            "quantising from the source (%.6g) did not beat transcoding the "
            "q4_1 (%.6g); the promoted-tensor route rests on this" % (e_direct, e_trans)
        )
    if verbose:
        print(
            "  [l] quantise-from-source beats transcode-from-q4_1: PASS "
            "(rms %.4g < %.4g)" % (e_direct, e_trans)
        )

    # (m) negative control for (l): a promoted tensor with no reference must be
    #     REFUSED, not silently transcoded. Without this the packer could
    #     degrade three layers and still report success.
    class _FakeInfo:
        ne = (K, M)
        ggml_type = 3  # Q4_1

    class _FakeGGUF:
        tensors = {"promoted": _FakeInfo()}

        def raw_bytes(self, name):
            return raw41

    try:
        q4_0_payload_for(_FakeGGUF(), "promoted")
        raise AssertionError(
            "NEGATIVE CONTROL FAILED: a promoted tensor was transcoded with no "
            "reference weights"
        )
    except ValueError:
        pass
    payload, prov = q4_0_payload_for(
        _FakeGGUF(), "promoted", reference=src.reshape(M, K)
    )
    if prov != "quantized_from_reference" or payload.size != n_blk * 18:
        raise AssertionError("promoted tensor with a reference did not quantise")
    if verbose:
        print("  [m] promoted tensor without reference: correctly REFUSED")


def _main():
    ap = argparse.ArgumentParser(prog="gguf_q4_0.py", description=__doc__)
    ap.add_argument(
        "--gguf",
        default=os.environ.get("SMOLLM2_GGUF"),
        help="checkpoint to inspect (default: $SMOLLM2_GGUF)",
    )
    ap.add_argument("--list", action="store_true", help="tensors + payload check")
    ap.add_argument("--meta", action="store_true", help="dump metadata KVs")
    ap.add_argument("--self-test", action="store_true", dest="self_test_")
    ap.add_argument("--k", type=int, default=256, help="self-test K")
    ap.add_argument("--m", type=int, default=64, help="self-test M")
    ap.add_argument("--seed", type=int, default=42, help="self-test seed")
    args = ap.parse_args()

    if args.self_test_:
        print("gguf_q4_0 self-test: K=%d, M=%d, seed=%d" % (args.k, args.m, args.seed))
        self_test(K=args.k, M=args.m, seed=args.seed)
        print("All self-tests PASSED.")
        return 0
    if args.gguf is None:
        ap.error("--gguf is required (or set SMOLLM2_GGUF)")

    g = GGUFFile(args.gguf)
    print(
        "GGUF v%d  arch=%s  tensors=%d  kv=%d  data_start=%d"
        % (g.version, g.architecture(), len(g.tensors), len(g.metadata), g.data_start)
    )
    if args.meta:
        for k, v in g.metadata.items():
            print("  %-44s %s" % (k, repr(v)[:100]))
    if args.list:
        c = Counter(t.type_name for t in g.tensors.values())
        print("  type histogram: %s" % dict(c))
        for name, t in g.tensors.items():
            print("  %-34s ne=%-16s %-6s %10d B" % (name, t.ne, t.type_name, t.nbytes))
        print(
            "  payload check: OK, %d tensors end at %d B"
            % (len(g.tensors), g.check_payloads())
        )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
