# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Head-aligned GEMV C[M] = A[M,K] @ B[K] with an in-core per-head epilogue.

The builder for `mv_heads.cc` (see its header for the kernel contract: TAG/KIND
row padding, the packed B, the persistent head accumulator). Raw-bindings port
of the study branch's `_build_qkv_heads_gemv` (tag pre-port-20260829), kept as
measured; `matvec.py` is the air.api form of the plain GEMV it descends from.

Row -> core mapping. matvec.py interleaves 8-row tiles across the herd_m
columns, so a head's 128 rows land on 16 cores. Here column tx owns the
CONTIGUOUS logical block [tx*rows_per_col, +rows_per_col) and launch iteration
i gives it rows tx*rows_per_col + i*tile_m + [0, tile_m): chunk
(i mod chunks_per_head) of head (i // chunks_per_head), chunks_per_head =
head_dim / tile_m. A core sees a head as consecutive iterations, accumulates
them in a persistent L1 head buffer and runs the epilogue on the last chunk.
The host stores A iteration-major (`qkv2_layout.qkv_heads_store_perm`) so the
L3 -> L2 fetch stays matvec.py's contiguous [herd_m, tile_m, K] block.

Output. Every iteration writes its [herd_m, head_dim] heads to its OWN slot of
the [n_iter * herd_m * head_dim] output (`qkv2_layout.qkv_heads_slot_gather`
picks each head's last-chunk slot on the host). The compiler requires it: a
per-head logical slot has 16 iterations writing one region, which
`air-verify-hierarchy-locality` (strict, aircc's default) rejects.

Func signature: (A: [M, K + K_PAD], B: [K + 3*head_dim], OUT: [n_iter*herd_m*head_dim]).
"""

import os
import sys

from air.ir import (
    MemRefType,
    IntegerAttr,
    AffineMap,
    AffineExpr,
    AffineSymbolExpr,
    AffineConstantExpr,
    F32Type,
    FloatAttr,
    StringAttr,
    UnitAttr,
)
from air.dialects.air import (
    module_builder,
    launch,
    segment,
    herd,
    dma_memcpy_nd,
    MemorySpace,
    T,
)
from air.dialects.affine import apply as affine_apply
from air.dialects import arith
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.func import FuncOp, CallOp
from air.backend.xrt_runner import type_mapper

# The kernel's host-side layout contract owns K_PAD (llms/shared/infra).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "llms"))
from shared.infra.external_kernels import mv_heads_object_name  # noqa: E402
from shared.infra.qkv2_layout import K_PAD  # noqa: E402


@module_builder
def build_module(
    m,
    k,
    head_dim,
    np_dtype,
    tile_m=8,
    herd_m=8,
    eps=1e-6,
    link_with=None,
):
    # HEAD_DIM is baked into the kernel object, so the object must be the one
    # compiled for THIS head_dim (a 128-wide object walks 128 elements through
    # a narrower head's L1 buffers).
    if link_with is None:
        link_with = mv_heads_object_name(head_dim)
    assert m % herd_m == 0, (m, herd_m)
    rows_per_col = m // herd_m
    assert rows_per_col % head_dim == 0, (rows_per_col, head_dim)
    assert head_dim % tile_m == 0, (head_dim, tile_m)
    # The epilogue's RoPE walks each HALF head in 16-lane vectors with no tail
    # (mv_heads.cc), so a half head must be a whole number of vectors.
    assert head_dim % 32 == 0, (head_dim, "half head must be a multiple of 16")
    n_iter = rows_per_col // tile_m
    assert k % 64 == 0, k
    k_pad = k + K_PAD
    b_total = k + 3 * head_dim

    xrt_dtype = type_mapper(np_dtype)
    f32 = F32Type.get()

    memrefTyA = MemRefType.get([m, k_pad], xrt_dtype)
    memrefTyB = MemRefType.get([b_total], xrt_dtype)
    memrefTyOut = MemRefType.get([n_iter * herd_m * head_dim], xrt_dtype)

    l2_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L2)
    l2MemrefTyA = MemRefType.get(
        [herd_m, tile_m, k_pad], xrt_dtype, memory_space=l2_mem_space
    )
    l2MemrefTyOut = MemRefType.get(
        [herd_m, head_dim], xrt_dtype, memory_space=l2_mem_space
    )

    l1_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1MemrefTyA = MemRefType.get([tile_m, k_pad], xrt_dtype, memory_space=l1_mem_space)
    l1MemrefTyB = MemRefType.get([b_total], xrt_dtype, memory_space=l1_mem_space)
    l1MemrefTyHead = MemRefType.get([head_dim], xrt_dtype, memory_space=l1_mem_space)

    chunk_func = FuncOp(
        "qkv_heads_chunk_bf16",
        (
            [
                T.i32(),
                T.i32(),
                l1MemrefTyA,
                l1MemrefTyB,
                l1MemrefTyHead,
                l1MemrefTyHead,
                f32,
            ],
            [],
        ),
        visibility="private",
    )
    chunk_func.attributes["link_with"] = StringAttr.get(link_with)
    chunk_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    @FuncOp.from_py_func(memrefTyA, memrefTyB, memrefTyOut)
    def matvec_heads(arg_a, arg_b, arg_out):
        @launch(operands=[arg_a, arg_b, arg_out], sizes=[n_iter, 1])
        def launch_body(l_ivx, l_ivy, l_sx, l_sy, l3_a, l3_b, l3_out):
            @segment(name="qkvh_seg", operands=[l_ivx, l3_a, l3_b, l3_out])
            def segment_body(ivx_s, l3_a_s, l3_b_s, l3_out_s):
                # stored row offset of this iteration's contiguous block
                stored_map = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0),
                            AffineConstantExpr.get(herd_m * tile_m),
                        )
                    ],
                )
                stored_offset_m = affine_apply(stored_map, [ivx_s])

                l2_a = AllocOp(l2MemrefTyA, [], [])
                l2_out = AllocOp(l2MemrefTyOut, [], [])
                l1_a = AllocOp(l1MemrefTyA, [], [])
                l1_b = AllocOp(l1MemrefTyB, [], [])
                l1_c = AllocOp(l1MemrefTyHead, [], [])
                l1_out = AllocOp(l1MemrefTyHead, [], [])

                # L3 -> L2: the iteration's contiguous [herd_m, tile_m, k_pad] block.
                dma_memcpy_nd(
                    l2_a,
                    l3_a_s,
                    src_offsets=[0, stored_offset_m, 0],
                    src_sizes=[herd_m, tile_m, k_pad],
                    src_strides=[tile_m * k_pad, k_pad, 1],
                )

                @herd(
                    name="qkvh_herd",
                    sizes=[herd_m, 1],
                    operands=[l1_a, l1_b, l1_c, l1_out, l2_a, l3_b_s, l2_out],
                )
                def herd_body(
                    _tx,
                    _ty,
                    _sx,
                    _sy,
                    _l1_a,
                    _l1_b,
                    _l1_c,
                    _l1_out,
                    _l2_a,
                    _l3_b,
                    _l2_out,
                ):
                    # B (packed): L3 -> L1 (broadcast, repeat channel).
                    dma_memcpy_nd(
                        _l1_b,
                        _l3_b,
                        src_offsets=[],
                        src_sizes=[b_total],
                        src_strides=[1],
                    )
                    # A: L2 -> L1, this core's column slice.
                    dma_memcpy_nd(
                        _l1_a,
                        _l2_a,
                        src_offsets=[_tx, 0, 0],
                        src_sizes=[1, tile_m, k_pad],
                        src_strides=[tile_m * k_pad, k_pad, 1],
                    )
                    m_const = arith.ConstantOp(IntegerAttr.get(T.i32(), tile_m), None)
                    k_const = arith.ConstantOp(IntegerAttr.get(T.i32(), k), None)
                    eps_c = arith.ConstantOp(f32, FloatAttr.get(f32, eps))
                    CallOp(
                        chunk_func,
                        [m_const, k_const, _l1_a, _l1_b, _l1_c, _l1_out, eps_c],
                    )
                    # OUT (the head, final on the last chunk): L1 -> L2 row tx.
                    dma_memcpy_nd(
                        _l2_out,
                        _l1_out,
                        dst_offsets=[_tx, 0],
                        dst_sizes=[1, head_dim],
                        dst_strides=[head_dim, 1],
                        src_offsets=[],
                        src_sizes=[head_dim],
                        src_strides=[1],
                    )

                herd_body.attributes["link_with"] = StringAttr.get(link_with)

                # L2 -> L3: this iteration's own [herd_m * head_dim] slot.
                slot_map = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0),
                            AffineConstantExpr.get(herd_m * head_dim),
                        )
                    ],
                )
                slot_offset = affine_apply(slot_map, [ivx_s])
                dma_memcpy_nd(
                    l3_out_s,
                    l2_out,
                    dst_offsets=[slot_offset],
                    dst_sizes=[herd_m * head_dim],
                    dst_strides=[1],
                    src_offsets=[0, 0],
                    src_sizes=[herd_m, head_dim],
                    src_strides=[head_dim, 1],
                )

                for buf in (l2_a, l2_out, l1_a, l1_b, l1_c, l1_out):
                    DeallocOp(buf)


if __name__ == "__main__":
    # Standalone device check of the GEMV launch + epilogue against numpy, on
    # the Qwen3-0.6B geometry by default: PASS! when every gathered head
    # matches rope(qknorm(W x)) (Q/K) or W x (V) within a bf16 tolerance.
    import argparse

    import numpy as np
    from air.backend.xrt import XRTBackend
    from ml_dtypes import bfloat16
    from shared.infra.external_kernels import compile_mv_heads
    from shared.infra.qkv2_layout import qkv2_gather, qkv2_out_total, qkv2_prep_weight

    p = argparse.ArgumentParser(
        description="head-aligned QKV GEMV + epilogue, on the NPU"
    )
    p.add_argument("--k", type=int, default=1024)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=16)
    p.add_argument("--n-kv-heads", type=int, default=8)
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args()
    hd, k = a.head_dim, a.k
    q_dim, kv_dim = a.n_heads * hd, a.n_kv_heads * hd
    m, half, eps = q_dim + 2 * kv_dim, hd // 2, 1e-6
    rng = np.random.default_rng(0)
    w = (rng.standard_normal((m, k)) * 0.05).astype(bfloat16)
    x = rng.standard_normal(k).astype(bfloat16)
    qn = (1 + 0.1 * rng.standard_normal(hd)).astype(bfloat16)
    kn = (1 + 0.1 * rng.standard_normal(hd)).astype(bfloat16)
    ang = rng.uniform(0, 2 * np.pi, half)
    lut = np.concatenate([np.cos(ang), np.sin(ang)]).astype(bfloat16)  # [cos | sin]
    f32 = np.float32
    # The kernel's arithmetic, step by step in bf16: bf16 dot, rstd over the
    # head, times the norm weight, then the half-split rotation.
    y = (w.astype(f32) @ x.astype(f32)).astype(bfloat16)
    ref = y.copy()
    for h in range(m // hd):
        lo = h * hd
        kind = 0 if lo < q_dim else (1 if lo < q_dim + kv_dim else 2)
        if kind == 2:
            continue
        c = y[lo : lo + hd].astype(f32)
        t = (c / np.sqrt(np.mean(c * c) + eps)).astype(bfloat16).astype(f32)
        t = (t * (qn if kind == 0 else kn).astype(f32)).astype(bfloat16).astype(f32)
        t1, t2 = t[:half], t[half:]
        cs, sn = lut[:half].astype(f32), lut[half:].astype(f32)
        ref[lo : lo + half] = (t1 * cs - t2 * sn).astype(bfloat16)
        ref[lo + half : lo + hd] = (t1 * sn + t2 * cs).astype(bfloat16)
    obj = compile_mv_heads(hd)  # in cwd, like the driver's build dir
    mod = build_module(m, k, hd, bfloat16, eps=eps, link_with=obj)
    be = XRTBackend(
        verbose=a.verbose,
        omit_while_true_loop=False,
        output_format="elf",
        instance_name="matvec_heads",  # ELF mode: must name the module's func
    )
    fn = be.compile_and_load(mod)
    A = qkv2_prep_weight(w, q_dim, q_dim + kv_dim, hd)
    B = np.concatenate([x, lut, qn, kn]).astype(bfloat16)
    out = np.zeros(qkv2_out_total(m, hd), dtype=bfloat16)
    got = fn(A, B, out)[2][qkv2_gather(m, hd)].astype(f32)
    be.unload()
    err = np.abs(got - ref.astype(f32))
    bound = 0.05 * np.abs(ref.astype(f32)) + 0.02
    bad = int((err > bound).sum())
    print(
        f"max abs err {err.max():.4f}, {bad} of {m} outside 5% + 0.02 (Q/K roped, V plain)"
    )
    print("PASS!" if bad == 0 else "FAIL")
    raise SystemExit(0 if bad == 0 else 1)
