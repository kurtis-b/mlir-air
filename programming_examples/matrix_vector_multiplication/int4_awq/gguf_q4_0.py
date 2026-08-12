# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""llama.cpp GGUF `q4_0` checkpoint -> the packed BO layout used by mlir-air's
int4-AWQ GEMV kernel (`mv_int4_bf16.cc`).

Why this works with no kernel change
------------------------------------
The device kernel computes

    dequant(A)[r, k] = (q[r, k] - z[r, g(k)]) * s_a[r, g(k)]

`q4_0`'s dequant is `d * (q - 8)` with `q` in [0, 16) -- the *same* expression
with the zero-point pinned to the constant 8 and the group size pinned to 32.
Group size is already a kernel parameter (`DIM_GS`, plumbed `GS ?= 128` in the
Makefile), and `gs=32` is legal: the GEMV instantiates
`matvec_int4_bf16_impl<DIM_M, DIM_K, DIM_GS, 32>`, so the inner vector `r` is
32 and `static_assert(gs % r == 0)` holds with `NSUB = gs/r = 1`.

So the whole bridge is host-side repacking:

  * `Z` becomes an all-8s plane (`k/32 x m` bytes of a constant -- see
    `q4_0_traffic_bytes` for what that costs),
  * `S` is q4_0's per-block `d`, fp16 -> bf16 (loses 3 mantissa bits; the
    kernel's packed BO carries bf16 scales -- `scale_rounding_error` measures
    it),
  * `Q` must be un-interleaved: q4_0 packs element `i` and element `i+16` of a
    32-element block into byte `i` (low/high nibble), while the kernel's `Q` is
    plain row-major `k/2` bytes (byte `b` holds elements `2b`, `2b+1`).

This module deliberately carries no dependency on the `gguf` Python package.
The container is simple enough to parse directly, and the study's standing rule
is that packages must not be installed while gates run.

Self-test: `python3 gguf_q4_0.py --self-test` (synthetic, no checkpoint).
Checkpoint dump: `python3 gguf_q4_0.py --gguf <file.gguf> --list`.
"""

import argparse
import os
import struct
import sys

import numpy as np
from ml_dtypes import bfloat16

# ---------------------------------------------------------------------------
# GGUF container
# ---------------------------------------------------------------------------

GGUF_MAGIC = b"GGUF"

# gguf_metadata_value_type
(
    _T_UINT8,
    _T_INT8,
    _T_UINT16,
    _T_INT16,
    _T_UINT32,
    _T_INT32,
    _T_FLOAT32,
    _T_BOOL,
    _T_STRING,
    _T_ARRAY,
    _T_UINT64,
    _T_INT64,
    _T_FLOAT64,
) = range(13)

_SCALAR_FMT = {
    _T_UINT8: "<B",
    _T_INT8: "<b",
    _T_UINT16: "<H",
    _T_INT16: "<h",
    _T_UINT32: "<I",
    _T_INT32: "<i",
    _T_FLOAT32: "<f",
    _T_BOOL: "<?",
    _T_UINT64: "<Q",
    _T_INT64: "<q",
    _T_FLOAT64: "<d",
}

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
        tname, blck, tsize = GGML_TYPES.get(ggml_type, ("?", 1, 0))
        nelem = 1
        for d in ne:
            nelem *= d
        self.nbytes = (nelem // blck) * tsize if blck else 0

    @property
    def type_name(self):
        return GGML_TYPES.get(self.ggml_type, ("UNKNOWN_%d" % self.ggml_type,))[0]

    def __repr__(self):
        return "TensorInfo(%s, ne=%s, %s, %d B @ %d)" % (
            self.name,
            self.ne,
            self.type_name,
            self.nbytes,
            self.abs_offset,
        )


class GGUFFile:
    """Minimal read-only GGUF parser: header, metadata KVs, tensor infos.

    Tensor payloads are read lazily with `np.memmap`, so opening a 1 GB
    checkpoint costs nothing.
    """

    def __init__(self, path):
        self.path = path
        self.metadata = {}
        self.tensors = {}
        self._parse()

    # -- primitive readers --------------------------------------------------

    def _u32(self, f):
        return struct.unpack("<I", f.read(4))[0]

    def _u64(self, f):
        return struct.unpack("<Q", f.read(8))[0]

    def _string(self, f):
        n = self._u64(f)
        return f.read(n).decode("utf-8", errors="replace")

    def _value(self, f, vtype):
        if vtype == _T_STRING:
            return self._string(f)
        if vtype == _T_ARRAY:
            elem_type = self._u32(f)
            n = self._u64(f)
            # Long arrays here are tokenizer vocabularies; keep the length but
            # do not materialize hundreds of thousands of Python strings.
            if elem_type == _T_STRING:
                if n > 1024:
                    for _ in range(n):
                        f.seek(self._u64(f), 1)
                    return ["<%d strings elided>" % n]
                return [self._string(f) for _ in range(n)]
            fmt = _SCALAR_FMT[elem_type]
            width = struct.calcsize(fmt)
            if n > 1024:
                f.seek(n * width, 1)
                return ["<%d values elided>" % n]
            raw = f.read(n * width)
            return [struct.unpack_from(fmt, raw, i * width)[0] for i in range(n)]
        fmt = _SCALAR_FMT[vtype]
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

    # -- container ----------------------------------------------------------

    def _parse(self):
        with open(self.path, "rb") as f:
            magic = f.read(4)
            if magic != GGUF_MAGIC:
                raise ValueError("not a GGUF file: magic=%r" % magic)
            self.version = self._u32(f)
            if self.version < 2:
                raise ValueError(
                    "GGUF v%d uses 32-bit counts; only v2+ supported" % self.version
                )
            n_tensors = self._u64(f)
            n_kv = self._u64(f)

            for _ in range(n_kv):
                key = self._string(f)
                vtype = self._u32(f)
                self.metadata[key] = self._value(f, vtype)

            raw_infos = []
            for _ in range(n_tensors):
                name = self._string(f)
                n_dims = self._u32(f)
                ne = [self._u64(f) for _ in range(n_dims)]
                ggml_type = self._u32(f)
                offset = self._u64(f)
                raw_infos.append((name, ne, ggml_type, offset))

            alignment = self.metadata.get("general.alignment", 32)
            pos = f.tell()
            pad = (alignment - (pos % alignment)) % alignment
            self.data_start = pos + pad

        for name, ne, ggml_type, offset in raw_infos:
            self.tensors[name] = TensorInfo(
                name, ne, ggml_type, offset, self.data_start
            )

    # -- payload ------------------------------------------------------------

    def raw_bytes(self, name):
        """The tensor's on-disk payload as a uint8 memmap (no copy)."""
        ti = self.tensors[name]
        return np.memmap(
            self.path,
            dtype=np.uint8,
            mode="r",
            offset=ti.abs_offset,
            shape=(ti.nbytes,),
        )

    def architecture(self):
        return self.metadata.get("general.architecture", "?")


# ---------------------------------------------------------------------------
# q4_0 -> (A_q, A_s, A_z)
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


def llama_unpermute_rows(w, n_head):
    """Undo llama.cpp's RoPE row permutation on a [out_features, in_features] matrix.

    `convert_hf_to_gguf.py`'s `LlamaModel.permute` reorders the rows of
    `q_proj` and `k_proj` so that llama.cpp's RoPE can treat each head's
    `head_dim` as two contiguous halves instead of HF's interleaved pairs:

        W.reshape(n_head, 2, head_dim//2, in).swapaxes(1, 2).reshape(out, in)

    This is its inverse. **It is not optional**: measured on layer 0 of both
    Llama-3.2-1B and SmolLM2-1.7B, the as-stored GGUF `attn_q`/`attn_k` have
    cosine ~0.03 against the HF originals and ~0.996 after un-permuting. A
    packer that skips it feeds the kernel a row-shuffled Q/K projection.

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


def repack_q4_0_for_gemv(raw, K, M, n_head=None, M_TILE=8, K_CHUNK=2048, N_CORES=8):
    """A q4_0 tensor -> the [total_tiles, tile_bytes] uint8 BO the decode ELFs take.

    Mirrors `llama32_1b_int4/awq_repacker.repack_for_gemv`, single-launch
    (`M_PER_LAUNCH = M`). Pass `n_head` (q_proj) or `n_kv_head` (k_proj) to undo
    llama.cpp's RoPE row permutation; leave it None for v/o/gate/up/down.
    """
    from matvec_int4_packed import pack_inputs  # same directory

    A_q, A_s, A_z = repack_q4_0_linear(raw, K, M)
    if n_head is not None:
        # The permutation reorders OUTPUT rows, so it applies to A_q's rows and
        # to the M axis of the per-group S/Z planes alike.
        A_q = np.ascontiguousarray(llama_unpermute_rows(A_q, n_head))
        A_s = np.ascontiguousarray(llama_unpermute_rows(A_s.T, n_head).T)
        A_z = np.ascontiguousarray(llama_unpermute_rows(A_z.T, n_head).T)
    return pack_inputs(A_q, A_s, A_z, M, K, QK4_0, M_TILE, K_CHUNK, N_CORES, M)


# ---------------------------------------------------------------------------
# Cost accounting (bytes, not adjectives)
# ---------------------------------------------------------------------------


def q4_0_traffic_bytes(K, M, gs=32, with_zeros=True):
    """Packed-BO bytes for one [M, K] weight, and the implied bits/weight.

    `with_zeros=False` models the symmetric variant that drops the all-8s Z
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


def checkpoint_traffic(g, gs=QK4_0, roles=None):
    """Sum the packed-BO traffic over every pure-q4_0 2-D tensor in a GGUF.

    Measured off the checkpoint's own tensor_info records, not off a config:
    a `Q4_0` file type is mixed in practice (llama.cpp promotes some tensors to
    Q4_1/Q6_K), and the promoted ones cannot go through this kernel at all.
    Both the pure-q4_0 subset and the all-q4_0 idealization are returned, since
    the study's DRAM axis wants the second and the hardware only admits the
    first.
    """
    per_type = {}
    q4 = []
    for name, t in g.tensors.items():
        if len(t.ne) != 2:
            continue
        nelem = int(t.ne[0]) * int(t.ne[1])
        per_type.setdefault(t.type_name, [0, 0])
        per_type[t.type_name][0] += 1
        per_type[t.type_name][1] += nelem
        if roles is not None and not any(r in name for r in roles):
            continue
        if t.type_name == "Q4_0":
            q4.append((name, int(t.ne[0]), int(t.ne[1]), nelem))

    n_q4 = sum(r[3] for r in q4)
    n_all = sum(v[1] for v in per_type.values())
    z_q4 = n_q4 // gs
    return {
        "per_type": per_type,
        "n_tensors_q4_0": len(q4),
        "n_weights_q4_0": n_q4,
        "n_weights_all_2d": n_all,
        "asym_bytes": n_q4 // 2 + 2 * (n_q4 // gs) + z_q4,
        "sym_bytes": n_q4 // 2 + 2 * (n_q4 // gs),
        "z_bytes": z_q4,
        "tensors": q4,
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
    A_q2, A_s2, A_z2 = repack_q4_0_linear(raw, K, M)
    A_qp = np.ascontiguousarray(llama_unpermute_rows(A_q2, n_head))
    A_sp = np.ascontiguousarray(llama_unpermute_rows(A_s2.T, n_head).T)
    A_zp = np.ascontiguousarray(llama_unpermute_rows(A_z2.T, n_head).T)
    nib_p = np.zeros((M, K), dtype=np.uint8)
    nib_p[:, 0::2] = A_qp & 0x0F
    nib_p[:, 1::2] = (A_qp >> 4) & 0x0F
    got = (nib_p.astype(np.float32) - A_zp.astype(np.float32).T.repeat(QK4_0, axis=1)) * (
        A_sp.astype(np.float32).T.repeat(QK4_0, axis=1)
    )
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

    # (h) the SYMMETRIC pack is the asymmetric pack with the Z plane removed
    #     and nothing else moved, and the byte delta is exactly K*M/gs.
    from matvec_int4_packed import pack_inputs  # type: ignore

    M_TILE, K_CHUNK, N_CORES = 8, K, 8
    n_gpc = K_CHUNK // QK4_0
    q_bytes = M_TILE * (K_CHUNK // 2)
    s_bytes = n_gpc * M_TILE * 2
    p_asym = pack_inputs(
        A_q, A_s, A_z, M, K, QK4_0, M_TILE, K_CHUNK, N_CORES, M
    )
    p_sym = pack_inputs(
        A_q, A_s, None, M, K, QK4_0, M_TILE, K_CHUNK, N_CORES, M, symmetric=True
    )
    if not np.array_equal(p_sym, p_asym[:, : q_bytes + s_bytes]):
        raise AssertionError("symmetric pack is not the asymmetric Q+S prefix")
    if not np.all(p_asym[:, q_bytes + s_bytes :] == 8):
        raise AssertionError("the region the symmetric pack drops is not all 8s")
    delta = p_asym.nbytes - p_sym.nbytes
    if delta != K * M // QK4_0:
        raise AssertionError(
            "symmetric saving %d B != K*M/gs = %d B" % (delta, K * M // QK4_0)
        )
    if verbose:
        print(
            "  [h] symmetric pack == asymmetric minus the all-8s Z plane: PASS "
            "(%d -> %d B, saved %d = K*M/%d)"
            % (p_asym.nbytes, p_sym.nbytes, delta, QK4_0)
        )

    # (i) negative control for (h): the symmetric pack must REFUSE a Z plane.
    #     Without this, a caller could believe a zero point shipped when the
    #     kernel is using an immediate.
    try:
        pack_inputs(
            A_q, A_s, A_z, M, K, QK4_0, M_TILE, K_CHUNK, N_CORES, M, symmetric=True
        )
        raise AssertionError(
            "NEGATIVE CONTROL FAILED: the symmetric pack accepted a Z plane"
        )
    except ValueError:
        pass
    if verbose:
        print("  [i] negative control (symmetric pack + Z plane): correctly REFUSED")

    # (j) the traffic model must agree with the packed BO it describes, both
    #     ways, and must reproduce llama.cpp's canonical bits/weight.
    t_a = q4_0_traffic_bytes(K, M, gs=QK4_0, with_zeros=True)
    t_s = q4_0_traffic_bytes(K, M, gs=QK4_0, with_zeros=False)
    if t_a["total_bytes"] != p_asym.nbytes or t_s["total_bytes"] != p_sym.nbytes:
        raise AssertionError(
            "traffic model %d/%d B disagrees with the packed BOs %d/%d B"
            % (t_a["total_bytes"], t_s["total_bytes"], p_asym.nbytes, p_sym.nbytes)
        )
    if abs(t_a["bits_per_weight"] - 4.75) > 1e-9 or abs(
        t_s["bits_per_weight"] - 4.50
    ) > 1e-9:
        raise AssertionError(
            "bits/weight %.6f / %.6f, expected 4.750 / 4.500"
            % (t_a["bits_per_weight"], t_s["bits_per_weight"])
        )
    if verbose:
        print(
            "  [j] traffic model matches the packed BOs, 4.750 -> 4.500 "
            "bits/weight: PASS"
        )


# ---------------------------------------------------------------------------

def _main():
    ap = argparse.ArgumentParser(
        prog="gguf_q4_0.py",
        description="GGUF q4_0 -> mlir-air packed-BO bridge + self-test.",
    )
    ap.add_argument("--gguf", type=str, default=None, help="checkpoint to inspect")
    ap.add_argument("--list", action="store_true", help="list tensors")
    ap.add_argument("--meta", action="store_true", help="dump metadata KVs")
    ap.add_argument(
        "--traffic",
        action="store_true",
        help="packed-BO bytes over the whole checkpoint, both variants",
    )
    ap.add_argument("--self-test", action="store_true", dest="self_test_")
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.self_test_ or args.gguf is None:
        print("gguf_q4_0 self-test: K=%d, M=%d, seed=%d" % (args.k, args.m, args.seed))
        self_test(K=args.k, M=args.m, seed=args.seed)
        print("All self-tests PASSED.")
        return 0

    g = GGUFFile(args.gguf)
    print("GGUF v%d  arch=%s  tensors=%d  kv=%d  data_start=%d"
          % (g.version, g.architecture(), len(g.tensors), len(g.metadata),
             g.data_start))
    if args.meta:
        for k, v in g.metadata.items():
            s = repr(v)
            print("  %-44s %s" % (k, s[:100]))
    if args.list:
        from collections import Counter
        c = Counter(t.type_name for t in g.tensors.values())
        print("  type histogram: %s" % dict(c))
        for name, t in g.tensors.items():
            print("  %-34s ne=%-16s %-6s %10d B" % (name, t.ne, t.type_name, t.nbytes))
    if args.traffic:
        MiB = 1024.0 * 1024.0
        LIN = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up",
               "ffn_down")
        tr = checkpoint_traffic(g, roles=LIN)
        print()
        print("packed-BO traffic for ONE full pass over the linear weights")
        print("  2-D tensor types in this checkpoint:")
        for tn, (cnt, ne) in sorted(tr["per_type"].items()):
            print("    %-6s %4d tensors %15d weights" % (tn, cnt, ne))
        print("  pure-q4_0 linears: %d tensors, %d weights"
              % (tr["n_tensors_q4_0"], tr["n_weights_q4_0"]))
        print("  asymmetric (Z present): %d B = %.2f MiB  (%.3f bits/weight)"
              % (tr["asym_bytes"], tr["asym_bytes"] / MiB,
                 tr["asym_bytes"] * 8.0 / tr["n_weights_q4_0"]))
        print("  symmetric  (no Z)     : %d B = %.2f MiB  (%.3f bits/weight)"
              % (tr["sym_bytes"], tr["sym_bytes"] / MiB,
                 tr["sym_bytes"] * 8.0 / tr["n_weights_q4_0"]))
        print("  Z plane                : %d B = %.2f MiB  (%.3f bits/weight)"
              % (tr["z_bytes"], tr["z_bytes"] / MiB,
                 tr["z_bytes"] * 8.0 / tr["n_weights_q4_0"]))
        # And the idealization the study quotes: every linear q4_0, including
        # the ones llama.cpp promoted. Reported separately, never merged.
        nl = g.metadata.get("llama.block_count")
        emb = g.metadata.get("llama.embedding_length")
        ff = g.metadata.get("llama.feed_forward_length")
        nh = g.metadata.get("llama.attention.head_count")
        nkv = g.metadata.get("llama.attention.head_count_kv")
        if None not in (nl, emb, ff, nh, nkv):
            hd = emb // nh
            per_layer = (emb * emb                 # q
                         + emb * (nkv * hd)        # k
                         + emb * (nkv * hd)        # v
                         + emb * emb               # o
                         + 3 * emb * ff)           # gate, up, down
            tot = per_layer * nl
            print("  --- idealization: all %d x 7 linears q4_0 ---" % nl)
            print("  weights: %d (%d per layer x %d layers)" % (tot, per_layer, nl))
            print("  asymmetric %d B = %.2f MiB (%.3f b/w) | symmetric %d B = "
                  "%.2f MiB (%.3f b/w) | Z %d B = %.2f MiB"
                  % (tot // 2 + 2 * (tot // QK4_0) + tot // QK4_0,
                     (tot // 2 + 2 * (tot // QK4_0) + tot // QK4_0) / MiB,
                     (tot // 2 + 2 * (tot // QK4_0) + tot // QK4_0) * 8.0 / tot,
                     tot // 2 + 2 * (tot // QK4_0),
                     (tot // 2 + 2 * (tot // QK4_0)) / MiB,
                     (tot // 2 + 2 * (tot // QK4_0)) * 8.0 / tot,
                     tot // QK4_0, (tot // QK4_0) / MiB))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
