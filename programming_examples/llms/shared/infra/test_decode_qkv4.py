# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only tests for the Qwen3 QKV stage's two host ABIs (decode_qkv4).

The 2-launch form (doc 57 O1 second half, 2026-08-23) and the 4-launch form
are selected by `QWEN3_RMS_QKV_LAUNCHES` in the 0.6B driver; these pin that
the driver's argument helper follows the selection, the shapes each ELF
expects, and that the weight augmentation + slot gather are each other's
inverse (the device never reorders rows the host cannot undo). No NPU.

No test-framework dependency — see `test_bo_pool.py` for why.

    python shared/infra/test_decode_qkv4.py
"""

import importlib
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ml_dtypes import bfloat16

_LLMS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_LLMS))
sys.path.insert(0, str(_LLMS / "qwen3_0_6b"))

from shared.builders.rms_qkv_qknorm_rope_multi import (  # noqa: E402
    K_PAD,
    QKV2_HERD_M,
    QKV2_TILE_M,
    qkv2_gather,
    qkv2_out_total,
    qkv2_prep_weight,
    qkv_heads_slot_gather,
    qkv_heads_store_perm,
)
from shared.infra import decode_qkv4 as q4  # noqa: E402

# Qwen3-0.6B's geometry: the shapes the gates ran on.
CFG = SimpleNamespace(emb_dim=1024, n_heads=16, n_kv_heads=8, head_dim=128)
Q_DIM, KV_DIM = CFG.n_heads * CFG.head_dim, CFG.n_kv_heads * CFG.head_dim
QKV_DIM = Q_DIM + 2 * KV_DIM


def _layer(seed=0):
    rng = np.random.default_rng(seed)
    f = lambda *s: (rng.standard_normal(s, dtype=np.float32) * 0.02).astype(bfloat16)
    return SimpleNamespace(
        attn_norm=f(CFG.emb_dim), q_norm=f(CFG.head_dim), k_norm=f(CFG.head_dim),
        _wq_t=f(Q_DIM, CFG.emb_dim), _wk_t=f(KV_DIM, CFG.emb_dim), _wv_t=f(KV_DIM, CFG.emb_dim),
    )


def _driver(launches):
    os.environ["QWEN3_RMS_QKV_LAUNCHES"] = str(launches)
    name = "qwen3_0_6b_decode"
    if name in sys.modules:
        return importlib.reload(sys.modules[name])
    return importlib.import_module(name)


def test_driver_args_follow_the_launch_count():
    """`rms_qkv4_args` returns the ABI of the form `_RMS_QKV_KERNEL` names
    (the review of 460beadc found it pinned to the 4-launch ABI)."""
    x = np.zeros(CFG.emb_dim, bfloat16)
    lut = np.zeros((CFG.n_heads + CFG.n_kv_heads) * CFG.head_dim, bfloat16)
    for launches, n_args, out_idx in ((2, 5, q4.OUTPUT_INDICES_2), (4, 9, q4.OUTPUT_INDICES)):
        d = _driver(launches)
        assert d._RMS_QKV_KERNEL == f"rms_qkv_qknorm_rope_gemv{launches}", d._RMS_QKV_KERNEL
        ins, outs, statics, inters = d.rms_qkv4_args(_layer(), x, lut, CFG)
        assert len(ins) == n_args, (launches, len(ins))
        assert list(outs) == list(out_idx)
        assert max(statics | inters | set(outs)) < n_args
    del os.environ["QWEN3_RMS_QKV_LAUNCHES"]


def test_two_launch_args_have_the_elf_shapes():
    lw = _layer()
    ins = q4.call_args_2(lw, np.zeros(CFG.emb_dim, bfloat16), np.ones(CFG.head_dim, bfloat16), CFG)
    assert [a.shape for a in ins] == [
        (CFG.emb_dim,), (CFG.emb_dim,), (CFG.emb_dim + 3 * CFG.head_dim,),
        (QKV_DIM, CFG.emb_dim + K_PAD), (qkv2_out_total(QKV_DIM, CFG.head_dim),),
    ], [a.shape for a in ins]
    assert all(a.dtype == bfloat16 for a in ins)
    bvec = ins[2]
    e, h = CFG.emb_dim, CFG.head_dim
    assert np.all(bvec[e:e + h] == 1) and np.array_equal(bvec[e + h:e + 2 * h], lw.q_norm)
    assert np.array_equal(bvec[e + 2 * h:], lw.k_norm)
    # idempotent, and the logical copy is released once the augmented one exists
    assert ins[3] is lw._wqkv2 and getattr(lw, "_wqkv_t", None) is None
    assert q4.call_args_2(lw, np.zeros(CFG.emb_dim, bfloat16), np.ones(h, bfloat16), CFG)[3] is lw._wqkv2


def test_augmented_weight_rows_tag_and_kind():
    """Every stored row carries (chunk tag, Q/K/V kind) of its logical row and
    the weight bytes of that row; the 64-element pad is otherwise zero."""
    lw = _layer()
    w = np.concatenate([lw._wq_t, lw._wk_t, lw._wv_t], axis=0)
    aug = qkv2_prep_weight(w, Q_DIM, Q_DIM + KV_DIM, CFG.head_dim)
    perm = qkv_heads_store_perm(QKV_DIM, QKV2_HERD_M, QKV2_TILE_M)
    assert sorted(perm.tolist()) == list(range(QKV_DIM))
    assert np.array_equal(aug[:, :CFG.emb_dim], w[perm])
    logical = perm
    tag = aug[:, CFG.emb_dim].astype(np.int64)
    kind = aug[:, CFG.emb_dim + 1].astype(np.int64)
    assert np.array_equal(tag, (logical % CFG.head_dim) // QKV2_TILE_M)
    assert np.array_equal(kind, np.where(logical < Q_DIM, 0, np.where(logical < Q_DIM + KV_DIM, 1, 2)))
    assert not aug[:, CFG.emb_dim + 2:].any()
    # a column owns whole heads, in order: stored block (iteration i, column tx)
    # is logical rows tx*rows_per_col + i*tile_m ...
    rows_per_col = QKV_DIM // QKV2_HERD_M
    assert rows_per_col % CFG.head_dim == 0
    for i in range(rows_per_col // QKV2_TILE_M):
        for tx in range(QKV2_HERD_M):
            base = i * QKV2_HERD_M * QKV2_TILE_M + tx * QKV2_TILE_M
            assert perm[base] == tx * rows_per_col + i * QKV2_TILE_M


def test_slot_gather_inverts_the_device_slot_layout():
    """Simulate the device: head h of column tx is final in the slot of its last
    chunk's iteration; gathering the slots yields q | k | v in logical order."""
    h, herd, tile = CFG.head_dim, QKV2_HERD_M, QKV2_TILE_M
    rows_per_col = QKV_DIM // herd
    cph = h // tile
    slots = np.full(qkv2_out_total(QKV_DIM, h), np.nan, dtype=np.float32)
    # 4096 distinct bf16-exact values (32 exponents x 128 mantissas): the row
    # index itself is not representable in bf16 above 256.
    i = np.arange(QKV_DIM)
    logical = (2.0 ** (i // 128) * (1 + (i % 128) / 128)).astype(np.float32)
    assert np.array_equal(logical.astype(bfloat16).astype(np.float32), logical)
    for tx in range(herd):
        for hd in range(rows_per_col // h):
            it = hd * cph + cph - 1
            lo = tx * rows_per_col + hd * h
            slots[it * herd * h + tx * h: it * herd * h + (tx + 1) * h] = logical[lo:lo + h]
    g = qkv2_gather(QKV_DIM, h)
    assert np.array_equal(g, qkv_heads_slot_gather(QKV_DIM, herd, h, tile))
    assert np.array_equal(slots[g], logical)
    res = {4: slots.astype(bfloat16)}
    v, q, k = q4.split_outputs_2(res, CFG)
    assert np.array_equal(q.astype(np.float32), logical[:Q_DIM])
    assert np.array_equal(k.astype(np.float32), logical[Q_DIM:Q_DIM + KV_DIM])
    assert np.array_equal(v.astype(np.float32), logical[Q_DIM + KV_DIM:])


def _main():
    tests = [(n, o) for n, o in globals().items() if n.startswith("test_") and callable(o)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS {name}")
        except Exception:
            failed.append(name)
            print(f"  FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
