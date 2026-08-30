# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""llama.cpp GGUF container reader for the int4-AWQ GEMV bridge.

Parses a GGUF (v2/v3) file -- header, metadata key/values, tensor infos with
absolute byte offsets -- and exposes each tensor's on-disk payload as a
zero-copy `np.memmap`. The q4_0 codec and the repack into the kernel's
packed-BO layout (`mv_int4_bf16.cc`) are layered on top of this reader.

No dependency on the `gguf` Python package: the container is simple enough to
parse directly, and no package may be installed while gates run.

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
            if self.version < 2:
                raise ValueError("GGUF v%d has 32-bit counts; need v2+" % self.version)
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


def _main():
    ap = argparse.ArgumentParser(prog="gguf_q4_0.py", description=__doc__)
    ap.add_argument(
        "--gguf",
        default=os.environ.get("SMOLLM2_GGUF"),
        help="checkpoint to inspect (default: $SMOLLM2_GGUF)",
    )
    ap.add_argument("--list", action="store_true", help="tensors + payload check")
    ap.add_argument("--meta", action="store_true", help="dump metadata KVs")
    args = ap.parse_args()
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
