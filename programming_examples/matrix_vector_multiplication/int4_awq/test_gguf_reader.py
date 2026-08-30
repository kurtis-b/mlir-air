# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host test for the GGUF container reader (`gguf_q4_0.GGUFFile`).

    python3 test_gguf_reader.py

Writes a minimal GGUF v3 file per the spec (header, scalar/string/array KVs,
three tensor infos, alignment padding, payloads) and checks the parsed table
-- names, shapes, types, absolute offsets, payload bytes -- against what was
written. Negative controls: a wrong magic, a tensor table cut short, a payload cut
short, and a v1 or v4 header must each raise. No checkpoint, no device, no toolchain.
"""

import os
import struct
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_q4_0 import GGML_TYPE_Q4_0, GGUFFile  # noqa: E402

ALIGN = 32
# (name, ne, ggml_type, nbytes): F32, Q4_0 (6 blocks x 18 B), F16.
TENSORS = [
    ("token_embd.weight", [8, 2], 0, 64),
    ("blk.0.attn_q.weight", [64, 3], GGML_TYPE_Q4_0, 108),
    ("output_norm.weight", [5], 1, 10),
]
_U = struct.pack


def _s(text):
    return _U("<Q", len(text)) + text.encode()


def _kv(key, vtype, payload):
    return _s(key) + _U("<I", vtype) + payload


def _payload(i, nbytes):
    return bytes((7 * j + 13 * i) & 0xFF for j in range(nbytes))


def build_gguf():
    """Returns (bytes, per-tensor relative offsets, header length)."""
    kvs = [
        _kv("general.architecture", 8, _s("llama")),
        _kv("general.alignment", 4, _U("<I", ALIGN)),
        _kv("llama.rope.freq_base", 6, _U("<f", 10000.0)),
        _kv("tokenizer.ggml.tokens", 9, _U("<IQ", 8, 2) + _s("a") + _s("b")),
        _kv("tokenizer.ggml.token_type", 9, _U("<IQ", 5, 3) + _U("<iii", 1, -2, 3)),
    ]
    head = b"GGUF" + _U("<IQQ", 3, len(TENSORS), len(kvs)) + b"".join(kvs)
    offsets, off = [], 0
    for name, ne, ty, nbytes in TENSORS:
        dims = b"".join(_U("<Q", d) for d in ne)
        head += _s(name) + _U("<I", len(ne)) + dims + _U("<IQ", ty, off)
        offsets.append(off)
        off += -(-nbytes // ALIGN) * ALIGN  # every payload padded to ALIGN
    body = bytearray(off)
    for i, (name, ne, ty, nbytes) in enumerate(TENSORS):
        body[offsets[i] : offsets[i] + nbytes] = _payload(i, nbytes)
    return head + bytes((-len(head)) % ALIGN) + bytes(body), offsets, len(head)


def _write(tmp, raw):
    path = os.path.join(tmp, "t.gguf")
    with open(path, "wb") as f:
        f.write(raw)
    return path


def test_header_and_metadata(tmp):
    g = GGUFFile(_write(tmp, build_gguf()[0]))
    assert (g.version, g.architecture()) == (3, "llama"), (g.version, g.architecture())
    assert g.metadata["general.alignment"] == ALIGN, g.metadata
    assert abs(g.metadata["llama.rope.freq_base"] - 10000.0) < 1e-3, g.metadata
    assert g.metadata["tokenizer.ggml.tokens"] == ["a", "b"], g.metadata
    assert g.metadata["tokenizer.ggml.token_type"] == [1, -2, 3], g.metadata


def test_tensor_table_offsets_and_payloads(tmp):
    raw, offsets, head_len = build_gguf()
    g = GGUFFile(_write(tmp, raw))
    assert (g.header_end, g.data_start) == (head_len, -(-head_len // ALIGN) * ALIGN)
    assert list(g.tensors) == [t[0] for t in TENSORS], list(g.tensors)
    for i, ((name, ne, ty, nbytes), off) in enumerate(zip(TENSORS, offsets)):
        t = g.tensors[name]
        assert (t.ne, t.ggml_type, t.nbytes) == (ne, ty, nbytes), (name, t.ne, t.nbytes)
        assert (t.rel_offset, t.abs_offset) == (off, g.data_start + off), (name, off)
        assert g.raw_bytes(name).tobytes() == _payload(i, nbytes), name
    assert g.tensors["blk.0.attn_q.weight"].type_name == "Q4_0"
    end = g.data_start + offsets[-1] + TENSORS[-1][3]
    assert g.check_payloads() == end and 0 <= len(raw) - end < ALIGN, (end, len(raw))


def test_bad_magic_and_truncations_raise(tmp):
    raw, _, head_len = build_gguf()
    for what, broken in [
        ("bad magic", b"GGML" + raw[4:]),
        ("tensor table cut short", raw[: head_len - 20]),
        ("last payload cut short", raw[:-40]),
    ]:
        try:
            GGUFFile(_write(tmp, broken)).check_payloads()
        except ValueError:
            continue
        raise AssertionError("%s: the reader did not raise" % what)


def test_unsupported_versions_raise(tmp):
    raw = build_gguf()[0]
    for version in (1, 4):
        broken = raw[:4] + struct.pack("<I", version) + raw[8:]
        try:
            GGUFFile(_write(tmp, broken))
        except ValueError:
            continue
        raise AssertionError("GGUF v%d: the reader did not raise" % version)


if __name__ == "__main__":
    tests = sorted(k for k in globals() if k.startswith("test_"))
    n_pass = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name in tests:
            try:
                globals()[name](tmp)
                n_pass += 1
                print("PASS  %s" % name)
            except Exception as e:  # report every failure shape, keep going
                print("FAIL  %s: %r" % (name, e))
    print("gguf reader tests: %d/%d passed" % (n_pass, len(tests)))
    sys.exit(0 if n_pass == len(tests) else 1)
