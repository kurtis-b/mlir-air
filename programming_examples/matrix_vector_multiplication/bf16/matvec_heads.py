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
    link_with="mv_heads_hd128.o",
):
    assert m % herd_m == 0, (m, herd_m)
    rows_per_col = m // herd_m
    assert rows_per_col % head_dim == 0, (rows_per_col, head_dim)
    assert head_dim % tile_m == 0, (head_dim, tile_m)
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
