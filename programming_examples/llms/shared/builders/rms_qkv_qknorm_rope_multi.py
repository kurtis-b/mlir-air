# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RMSNorm + QKV GEMMs + per-head QK-norm + RoPE Q+K — 8-launch prefill ELF.

This is the Qwen3 generalization of `rms_gemms_rope_multi.build_rms_gemms_rope_module`.
Qwen3 applies a per-head RMSNorm (QK-norm) to Q and K AFTER the projection
GEMM and BEFORE RoPE. RoPE's nonlinearity-free rotation does NOT commute past
the (nonlinear) QK-norm, so the QK-norm must sit physically between the GEMM
and RoPE. We express it with the existing `weighted_rms_norm` row-wise RMSNorm
kernel: q[seq, n_heads*head_dim] viewed as [seq*n_heads, head_dim] rows gives a
row-wise RMSNorm over head_dim with weight q_norm (head_dim,) broadcast across
all rows — exactly per-head QK-norm. No new C kernel needed.

8 launches (vs the 6 of rms_gemms_rope):
  1. RMSNorm      x_in x norm_w -> normed
  2. Q GEMM       normed x wq -> q          (seq, q_dim)
  3. K GEMM       normed x wk -> k          (seq, kv_dim)
  4. V GEMM       normed x wv -> v          (seq, kv_dim)
  5. QK-norm Q    q  x q_norm -> q_n        (per-head RMSNorm, head_dim)  <-- NEW
  6. QK-norm K    k  x k_norm -> k_n        (per-head RMSNorm, head_dim)  <-- NEW
  7. RoPE Q       q_n(2D->1D) x lut_q -> q_roped(1D->2D)
  8. RoPE K       k_n(2D->1D) x lut_k -> k_roped(1D->2D)

The QK-norm slices reshape the 2D Q/K GEMM-output buffers (seq, *_dim) into
[seq*heads, head_dim] views via a collapse_shape -> expand_shape prelude and
route in/out operands onto those SSA values with `arg_aliases`, so no extra
func arg is needed for the reshaped view. QK-norm writes to dedicated output
buffers (q_n, k_n) that RoPE then consumes, keeping the data flow explicit
(not in-place).

The QK-norm RMSNorm kernel reads `weighted_rms_norm.EPS`; Qwen3 uses eps=1e-6,
so we temporarily override that module global during the QK-norm build (same
pattern the decode `rms_gemv_rope_multi.EPS` override uses).
"""

import numpy as np
from ml_dtypes import bfloat16

# ---------------------------------------------------------------------------
# Per-head RMSNorm (QK-norm) with 2D in/out args (collapse to 1D inside launch,
# process head_dim-wide rows). Modeled on rms_gemms_rope_multi._build_rope_2d so
# the L1 DMA reads a collapse_shape of a block argument — the allowed AIE
# dma_bd chain (subview/cast/collapse of a block arg). expand_shape on a func
# arg is REJECTED by the AIE backend, which is why we cannot just reshape the
# 2D GEMM-output buffer to [rows, head_dim] and feed weighted_rms_norm.
#
# Math mirrors weighted_rms_norm: sum(x^2) accumulated in f32, rstd = rsqrt(
# mean + eps) in f32, epilogue y = x * rstd * weight in bf16 vectors. eps is a
# build-time arg (Qwen3 = 1e-6).
# ---------------------------------------------------------------------------


from air.ir import (
    MemRefType,
    IntegerAttr,
    AffineMap,
    AffineExpr,
    AffineSymbolExpr,
    AffineConstantExpr,
    AffineMapAttr,
    VectorType,
    F32Type,
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
from air.dialects import arith, math as math_dialect
from air.dialects.memref import AllocOp, DeallocOp, subview
from air.dialects.memref import collapse_shape as memref_collapse_shape
from air.dialects.vector import (
    transfer_read,
    transfer_write,
    BroadcastOp,
    reduction as vector_reduction,
)
from air.dialects.func import FuncOp
from air.dialects.scf import for_ as range_, yield_
from air.backend.xrt_runner import type_mapper


@module_builder
def _build_qknorm_2d(
    outer_rows, outer_cols, head_dim, np_dtype, eps, herd_x, vector_size=16
):
    """Build a per-head RMSNorm launch with 2D in/out args.

    The outer 2D shape (outer_rows=seq_len, outer_cols=q_dim or kv_dim) matches
    the GEMM output type. Inside the launch the buffers are collapse_shape'd to
    1D and the herd processes total/head_dim rows of head_dim each, RMSNorm-ing
    each row with the shared weight (head_dim,).

    Func signature:
      (in_2d: [outer_rows, outer_cols], weight_1d: [head_dim], out_2d: [outer_rows, outer_cols])
    """
    xrt_dtype = type_mapper(np_dtype)
    total = outer_rows * outer_cols
    rope_rows = total // head_dim  # n_heads * seq_len
    herd_y = 1
    total_tiles = herd_x * herd_y
    assert head_dim % vector_size == 0
    assert total % head_dim == 0
    assert rope_rows % total_tiles == 0
    rows_per_tile = rope_rows // total_tiles

    f32 = F32Type.get()
    vecTy = VectorType.get([vector_size], xrt_dtype)
    vecTyF32 = VectorType.get([vector_size], f32)
    identity_map = AffineMapAttr.get(AffineMap.get_identity(1))

    l3_2d_ty = MemRefType.get([outer_rows, outer_cols], xrt_dtype)
    l3_1d_ty = MemRefType.get([total], xrt_dtype)
    l3_w_ty = MemRefType.get([head_dim], xrt_dtype)

    l1_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1RowTy = MemRefType.get([head_dim], xrt_dtype, memory_space=l1_mem_space)
    l1VecTyF32 = MemRefType.get([vector_size], f32, memory_space=l1_mem_space)
    l1SqTy = MemRefType.get([vector_size], xrt_dtype, memory_space=l1_mem_space)

    # row_offset = (local_row + tile_id * rows_per_tile) * head_dim
    row_offset_map = AffineMap.get(
        0,
        3,
        [
            AffineExpr.get_mul(
                AffineExpr.get_add(
                    AffineSymbolExpr.get(0),
                    AffineExpr.get_mul(
                        AffineExpr.get_add(
                            AffineExpr.get_mul(
                                AffineSymbolExpr.get(1),
                                AffineConstantExpr.get(herd_y),
                            ),
                            AffineSymbolExpr.get(2),
                        ),
                        AffineConstantExpr.get(rows_per_tile),
                    ),
                ),
                AffineConstantExpr.get(head_dim),
            )
        ],
    )

    @FuncOp.from_py_func(l3_2d_ty, l3_w_ty, l3_2d_ty)
    def qknorm_2d(arg0_2d, arg1_w, arg2_2d):
        @launch(operands=[arg0_2d, arg1_w, arg2_2d])
        def qkn_launch(l_in_2d, l_w, l_out_2d):
            in_flat = memref_collapse_shape(l3_1d_ty, l_in_2d, [[0, 1]])
            out_flat = memref_collapse_shape(l3_1d_ty, l_out_2d, [[0, 1]])

            @segment(name="qkn_seg", operands=[in_flat, l_w, out_flat])
            def qkn_seg(s_in, s_w, s_out):
                @herd(
                    name="qkn_herd", sizes=[herd_x, herd_y], operands=[s_in, s_w, s_out]
                )
                def qkn_body(_tx, _ty, _sx, _sy, h_in, h_w, h_out):
                    l1_in = AllocOp(l1RowTy, [], [])
                    l1_out = AllocOp(l1RowTy, [], [])
                    l1_w = AllocOp(l1RowTy, [], [])
                    l1_acc = AllocOp(l1VecTyF32, [], [])
                    l1_sq = AllocOp(l1SqTy, [], [])

                    c0 = arith.ConstantOp.create_index(0)
                    cst0 = arith.ConstantOp(xrt_dtype, 0.0)
                    cst0_f32 = arith.ConstantOp(f32, 0.0)
                    n_f = arith.ConstantOp(f32, float(head_dim))
                    eps_f = arith.ConstantOp(f32, eps)
                    v_zero_f32 = BroadcastOp(vecTyF32, cst0_f32)

                    # weight DMA once per tile (broadcast across rows).
                    dma_memcpy_nd(
                        l1_w,
                        h_w,
                        src_offsets=[0],
                        src_sizes=[head_dim],
                        src_strides=[1],
                    )

                    for local_row in range_(rows_per_tile):
                        row_off = affine_apply(row_offset_map, [local_row, _tx, _ty])
                        dma_memcpy_nd(
                            l1_in,
                            h_in,
                            src_offsets=[row_off],
                            src_sizes=[head_dim],
                            src_strides=[1],
                        )

                        # sum of x^2 in f32.
                        transfer_write(
                            None, v_zero_f32, l1_acc, [c0], identity_map, [True]
                        )
                        for j in range_(0, head_dim, vector_size):
                            sub_in = subview(l1_in.result, [j], [vector_size], [1])
                            v_x = transfer_read(
                                vecTy, sub_in, [c0], identity_map, cst0, [True]
                            )
                            v_sq = arith.mulf(v_x, v_x)
                            transfer_write(
                                None, v_sq, l1_sq, [c0], identity_map, [True]
                            )
                            v_sq_rd = transfer_read(
                                vecTy, l1_sq, [c0], identity_map, cst0, [True]
                            )
                            v_sq_f32 = arith.extf(vecTyF32, v_sq_rd)
                            v_acc = transfer_read(
                                vecTyF32, l1_acc, [c0], identity_map, cst0_f32, [True]
                            )
                            v_sum = arith.addf(v_acc, v_sq_f32)
                            transfer_write(
                                None, v_sum, l1_acc, [c0], identity_map, [True]
                            )
                            yield_([])

                        v_final = transfer_read(
                            vecTyF32, l1_acc, [c0], identity_map, cst0_f32, [True]
                        )
                        total_sum = vector_reduction(f32, "add", v_final)
                        rms = arith.divf(total_sum, n_f)
                        rms_eps = arith.addf(rms, eps_f)
                        rstd_f32 = math_dialect.rsqrt(rms_eps)
                        rstd = arith.truncf(xrt_dtype, rstd_f32)
                        v_rstd = BroadcastOp(vecTy, rstd)

                        for j in range_(0, head_dim, vector_size):
                            sub_in = subview(l1_in.result, [j], [vector_size], [1])
                            sub_w = subview(l1_w.result, [j], [vector_size], [1])
                            sub_out = subview(l1_out.result, [j], [vector_size], [1])
                            v_x = transfer_read(
                                vecTy, sub_in, [c0], identity_map, cst0, [True]
                            )
                            v_w = transfer_read(
                                vecTy, sub_w, [c0], identity_map, cst0, [True]
                            )
                            v_normed = arith.mulf(v_x, v_rstd)
                            v_weighted = arith.mulf(v_normed, v_w)
                            transfer_write(
                                None, v_weighted, sub_out, [c0], identity_map, [True]
                            )
                            yield_([])

                        dma_memcpy_nd(
                            h_out,
                            l1_out,
                            dst_offsets=[row_off],
                            dst_sizes=[head_dim],
                            dst_strides=[1],
                        )
                        yield_([])

                    DeallocOp(l1_in)
                    DeallocOp(l1_out)
                    DeallocOp(l1_w)
                    DeallocOp(l1_acc)
                    DeallocOp(l1_sq)


@module_builder
def _build_qknorm_1d(
    n_rows,
    head_dim,
    np_dtype,
    eps,
    herd_x=8,
    vector_size=16,
    per_row_weight=False,
    in_total=None,
):
    """Decode per-head RMSNorm with 1D func args (M=1 token).

    Func signature: (in_1d: [n_rows*head_dim], weight: [head_dim], out_1d: [n_rows*head_dim]).
    The herd processes n_rows rows (= n_heads or n_kv_heads) of head_dim each.
    Mirrors _build_qknorm_2d math but with no collapse (args are already 1D).

    per_row_weight=True `[2026-08-21]` (doc 57 O1): the weight arg is
    [n_rows*head_dim] and row r is normalized with weight row r. That lets ONE
    launch normalize Q's n_heads rows with q_norm and K's n_kv_heads rows with
    k_norm from a host-tiled [q_norm x n_heads; k_norm x n_kv_heads] buffer,
    which is static, so two launches become one with no kernel change.

    in_total (default n_rows*head_dim): length of the INPUT buffer. Larger than
    the rows processed when the input is a packed [q | k | v] GEMV output whose
    first n_rows*head_dim elements are Q|K -- the launch then takes the whole
    func arg and never reads past row n_rows. (A `memref.subview` + `cast`
    prelude, the o_gemv_ffn pattern, fails here: with a single use the cast is
    sunk into the launch region and its subview operand is left outside --
    "'memref.cast' op using value defined outside the region", devq 459.)
    """
    xrt_dtype = type_mapper(np_dtype)
    total = n_rows * head_dim
    in_total = total if in_total is None else in_total
    assert in_total >= total
    herd_y = 1
    total_tiles = herd_x * herd_y
    assert head_dim % vector_size == 0
    assert n_rows % total_tiles == 0
    rows_per_tile = n_rows // total_tiles

    f32 = F32Type.get()
    vecTy = VectorType.get([vector_size], xrt_dtype)
    vecTyF32 = VectorType.get([vector_size], f32)
    identity_map = AffineMapAttr.get(AffineMap.get_identity(1))

    l3_1d_ty = MemRefType.get([total], xrt_dtype)
    l3_in_ty = MemRefType.get([in_total], xrt_dtype)
    l3_w_ty = MemRefType.get([total if per_row_weight else head_dim], xrt_dtype)
    l1_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1RowTy = MemRefType.get([head_dim], xrt_dtype, memory_space=l1_mem_space)
    l1VecTyF32 = MemRefType.get([vector_size], f32, memory_space=l1_mem_space)
    l1SqTy = MemRefType.get([vector_size], xrt_dtype, memory_space=l1_mem_space)

    row_offset_map = AffineMap.get(
        0,
        3,
        [
            AffineExpr.get_mul(
                AffineExpr.get_add(
                    AffineSymbolExpr.get(0),
                    AffineExpr.get_mul(
                        AffineExpr.get_add(
                            AffineExpr.get_mul(
                                AffineSymbolExpr.get(1),
                                AffineConstantExpr.get(herd_y),
                            ),
                            AffineSymbolExpr.get(2),
                        ),
                        AffineConstantExpr.get(rows_per_tile),
                    ),
                ),
                AffineConstantExpr.get(head_dim),
            )
        ],
    )

    @FuncOp.from_py_func(l3_in_ty, l3_w_ty, l3_1d_ty)
    def qknorm_1d(arg0_in, arg1_w, arg2_out):
        @launch(operands=[arg0_in, arg1_w, arg2_out])
        def qkn_launch(l_in, l_w, l_out):
            @segment(name="qkn1_seg", operands=[l_in, l_w, l_out])
            def qkn_seg(s_in, s_w, s_out):
                @herd(
                    name="qkn1_herd",
                    sizes=[herd_x, herd_y],
                    operands=[s_in, s_w, s_out],
                )
                def qkn_body(_tx, _ty, _sx, _sy, h_in, h_w, h_out):
                    l1_in = AllocOp(l1RowTy, [], [])
                    l1_out = AllocOp(l1RowTy, [], [])
                    l1_w = AllocOp(l1RowTy, [], [])
                    l1_acc = AllocOp(l1VecTyF32, [], [])
                    l1_sq = AllocOp(l1SqTy, [], [])

                    c0 = arith.ConstantOp.create_index(0)
                    cst0 = arith.ConstantOp(xrt_dtype, 0.0)
                    cst0_f32 = arith.ConstantOp(f32, 0.0)
                    n_f = arith.ConstantOp(f32, float(head_dim))
                    eps_f = arith.ConstantOp(f32, eps)
                    v_zero_f32 = BroadcastOp(vecTyF32, cst0_f32)

                    if not per_row_weight:
                        dma_memcpy_nd(
                            l1_w,
                            h_w,
                            src_offsets=[0],
                            src_sizes=[head_dim],
                            src_strides=[1],
                        )

                    for local_row in range_(rows_per_tile):
                        row_off = affine_apply(row_offset_map, [local_row, _tx, _ty])
                        dma_memcpy_nd(
                            l1_in,
                            h_in,
                            src_offsets=[row_off],
                            src_sizes=[head_dim],
                            src_strides=[1],
                        )
                        if per_row_weight:
                            dma_memcpy_nd(
                                l1_w,
                                h_w,
                                src_offsets=[row_off],
                                src_sizes=[head_dim],
                                src_strides=[1],
                            )

                        transfer_write(
                            None, v_zero_f32, l1_acc, [c0], identity_map, [True]
                        )
                        for j in range_(0, head_dim, vector_size):
                            sub_in = subview(l1_in.result, [j], [vector_size], [1])
                            v_x = transfer_read(
                                vecTy, sub_in, [c0], identity_map, cst0, [True]
                            )
                            v_sq = arith.mulf(v_x, v_x)
                            transfer_write(
                                None, v_sq, l1_sq, [c0], identity_map, [True]
                            )
                            v_sq_rd = transfer_read(
                                vecTy, l1_sq, [c0], identity_map, cst0, [True]
                            )
                            v_sq_f32 = arith.extf(vecTyF32, v_sq_rd)
                            v_acc = transfer_read(
                                vecTyF32, l1_acc, [c0], identity_map, cst0_f32, [True]
                            )
                            v_sum = arith.addf(v_acc, v_sq_f32)
                            transfer_write(
                                None, v_sum, l1_acc, [c0], identity_map, [True]
                            )
                            yield_([])

                        v_final = transfer_read(
                            vecTyF32, l1_acc, [c0], identity_map, cst0_f32, [True]
                        )
                        total_sum = vector_reduction(f32, "add", v_final)
                        rms = arith.divf(total_sum, n_f)
                        rms_eps = arith.addf(rms, eps_f)
                        rstd_f32 = math_dialect.rsqrt(rms_eps)
                        rstd = arith.truncf(xrt_dtype, rstd_f32)
                        v_rstd = BroadcastOp(vecTy, rstd)

                        for j in range_(0, head_dim, vector_size):
                            sub_in = subview(l1_in.result, [j], [vector_size], [1])
                            sub_w = subview(l1_w.result, [j], [vector_size], [1])
                            sub_out = subview(l1_out.result, [j], [vector_size], [1])
                            v_x = transfer_read(
                                vecTy, sub_in, [c0], identity_map, cst0, [True]
                            )
                            v_w = transfer_read(
                                vecTy, sub_w, [c0], identity_map, cst0, [True]
                            )
                            v_normed = arith.mulf(v_x, v_rstd)
                            v_weighted = arith.mulf(v_normed, v_w)
                            transfer_write(
                                None, v_weighted, sub_out, [c0], identity_map, [True]
                            )
                            yield_([])

                        dma_memcpy_nd(
                            h_out,
                            l1_out,
                            dst_offsets=[row_off],
                            dst_sizes=[head_dim],
                            dst_strides=[1],
                        )
                        yield_([])

                    DeallocOp(l1_in)
                    DeallocOp(l1_out)
                    DeallocOp(l1_w)
                    DeallocOp(l1_acc)
                    DeallocOp(l1_sq)


def build_rms_qkv_qknorm_rope_module(
    seq_len,
    emb_dim,
    q_dim,
    kv_dim,
    n_heads,
    n_kv_heads,
    head_dim,
    herd_m=8,
    herd_n=4,
    rope_herd_x=8,
    qknorm_eps=1e-6,
    qknorm_herd_x=8,
    gemm_spec_fn=None,
):
    """Build the 8-launch fused prefill attention-input ELF.

    Func args:
      %arg0  x_in     (seq_len, emb_dim)
      %arg1  norm_w   (emb_dim,)
      %arg2  normed   (seq_len, emb_dim)
      %arg3  wq       (emb_dim, q_dim)
      %arg4  q        (seq_len, q_dim)            Q GEMM out (pre-QK-norm)
      %arg5  wk       (emb_dim, kv_dim)
      %arg6  k        (seq_len, kv_dim)           K GEMM out (pre-QK-norm)
      %arg7  wv       (emb_dim, kv_dim)
      %arg8  v        (seq_len, kv_dim)           V GEMM out (final)
      %arg9  q_norm   (head_dim,)                 QK-norm Q weight
      %arg10 k_norm   (head_dim,)                 QK-norm K weight
      %arg11 q_n      (seq_len, q_dim)            QK-norm Q out (RoPE input)
      %arg12 k_n      (seq_len, kv_dim)           QK-norm K out (RoPE input)
      %arg13 lut_q    (n_heads*seq_len*head_dim,) RoPE Q LUT (1D, seq-first)
      %arg14 lut_k    (n_kv_heads*seq_len*head_dim,) RoPE K LUT (1D)
      %arg15 q_roped  (seq_len, q_dim)            final RoPE Q
      %arg16 k_roped  (seq_len, kv_dim)           final RoPE K
      [+ registry-driven f32 C-scratch tail args for fused-cast GEMMs]

    gemm_spec_fn: optional callable (m, k, n) -> spec dict shaped like
      gemm_registry_config() (keys: method, tile_m/k_l2/k_l1/n, sym_suffix,
      build_kwargs, needs_f32_scratch). When None (default), the per-GEMM spec
      is looked up from the kernel registry via gemm_registry_config — used by
      qwen3_0_6b / qwen3_1_7b whose shapes are in the registry. Models whose
      attention-input GEMM shapes are NOT in the registry (e.g. qwen3_4b with
      emb=2560) pass their own gemm_spec so the Q/K/V GEMMs use the same
      validated method+tiles the model's split rms_qkv ELF already used.

    Returns (module, scratch_for).
    """
    from shared.builders.gemm_builder import _build_gemm_module, gemm_registry_config
    from shared.builders.rms_gemms_rope_multi import _build_rope_2d
    from shared.infra.stitching import (
        _wrap_ir_in_launch,
        stitch_elf,
        KernelSlice,
        FuncArg,
        alloc_gemm_scratch,
    )
    from weighted_rms_norm.weighted_rms_norm import build_module as build_rms

    q_total = seq_len * q_dim
    k_total = seq_len * kv_dim
    assert q_dim == n_heads * head_dim, (q_dim, n_heads, head_dim)
    assert kv_dim == n_kv_heads * head_dim, (kv_dim, n_kv_heads, head_dim)

    # Per-GEMM config: caller-injected spec fn (off-registry shapes) or the
    # registry lookup (default).
    if gemm_spec_fn is not None:
        q_spec = gemm_spec_fn(seq_len, emb_dim, q_dim)
        k_spec = gemm_spec_fn(seq_len, emb_dim, kv_dim)
        v_spec = gemm_spec_fn(seq_len, emb_dim, kv_dim)
    else:
        q_spec = gemm_registry_config(seq_len, emb_dim, q_dim, "bf16", "high")
        k_spec = gemm_registry_config(seq_len, emb_dim, kv_dim, "bf16", "high")
        v_spec = gemm_registry_config(seq_len, emb_dim, kv_dim, "bf16", "high")

    def _kw_tiles(spec):
        return (
            dict(spec["build_kwargs"]),
            spec["tile_m"],
            spec["tile_k_l2"],
            spec["tile_k_l1"],
            spec["tile_n"],
        )

    # ---- Build sub-kernels ----
    print("  [1/8] RMSNorm...")
    rms_ir = _wrap_ir_in_launch(
        str(build_rms(seq_len, emb_dim, bfloat16, 16, herd_x=8))
    )

    _q_kw, _q_tm, _q_k2, _q_k1, _q_tn = _kw_tiles(q_spec)
    _k_kw, _k_tm, _k_k2, _k_k1, _k_tn = _kw_tiles(k_spec)
    _v_kw, _v_tm, _v_k2, _v_k1, _v_tn = _kw_tiles(v_spec)
    print(f"  [2/8] Q GEMM ({q_spec['method']})  {seq_len}x{emb_dim}x{q_dim}...")
    q_ir = str(
        _build_gemm_module(
            seq_len, emb_dim, q_dim, _q_tm, _q_k2, _q_k1, _q_tn, herd_m, herd_n, **_q_kw
        )
    )
    print(f"  [3/8] K GEMM ({k_spec['method']})  {seq_len}x{emb_dim}x{kv_dim}...")
    k_ir = str(
        _build_gemm_module(
            seq_len,
            emb_dim,
            kv_dim,
            _k_tm,
            _k_k2,
            _k_k1,
            _k_tn,
            herd_m,
            herd_n,
            **_k_kw,
        )
    )
    print(f"  [4/8] V GEMM ({v_spec['method']})  {seq_len}x{emb_dim}x{kv_dim}...")
    v_ir = str(
        _build_gemm_module(
            seq_len,
            emb_dim,
            kv_dim,
            _v_tm,
            _v_k2,
            _v_k1,
            _v_tn,
            herd_m,
            herd_n,
            **_v_kw,
        )
    )

    # 5-6. QK-norm: per-head RMSNorm over head_dim with eps=1e-6. Uses a
    #   dedicated 2D-in/out builder (collapse inside launch) — see
    #   _build_qknorm_2d for why expand_shape on the func arg is illegal.
    qn_rows = seq_len * n_heads
    kn_rows = seq_len * n_kv_heads
    print(f"  [5/8] QK-norm Q (rows={qn_rows} dim={head_dim} eps={qknorm_eps})...")
    qkn_q_ir = str(
        _build_qknorm_2d(seq_len, q_dim, head_dim, bfloat16, qknorm_eps, qknorm_herd_x)
    )
    print(f"  [6/8] QK-norm K (rows={kn_rows} dim={head_dim} eps={qknorm_eps})...")
    qkn_k_ir = str(
        _build_qknorm_2d(seq_len, kv_dim, head_dim, bfloat16, qknorm_eps, qknorm_herd_x)
    )

    # 7-8. RoPE Q/K (2D in/out, head_dim wide).
    print(f"  [7/8] RoPE Q (outer={seq_len}x{q_dim}, dim={head_dim})...")
    rope_q_ir = str(_build_rope_2d(seq_len, q_dim, head_dim, bfloat16, rope_herd_x))
    print(f"  [8/8] RoPE K (outer={seq_len}x{kv_dim}, dim={head_dim})...")
    rope_k_ir = str(_build_rope_2d(seq_len, kv_dim, head_dim, bfloat16, rope_herd_x))

    # ---- Scratch (fused-cast GEMM f32 C-tail) ----
    scratch_args, scratch_for = alloc_gemm_scratch(
        [
            (q_spec, seq_len, q_dim),
            (k_spec, seq_len, kv_dim),
            (v_spec, seq_len, kv_dim),
        ],
        base_arg_count=17,
    )

    def _gemm_arg_map(in_idx, w_idx, out_idx, sc):
        if sc is not None:
            return {0: in_idx, 1: w_idx, 2: sc, 3: out_idx}
        return {0: in_idx, 1: w_idx, 2: out_idx}

    def _gemm_externs(spec):
        sfx = spec["sym_suffix"]
        return {
            "@op_has_no_registered_library_name" + sfx,
            "@zero_f32_mn" + sfx,
            "@f32_to_bf16_mn" + sfx,
        }

    base_args = [
        FuncArg("%arg0", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg1", f"memref<{emb_dim}xbf16>"),
        FuncArg("%arg2", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg3", f"memref<{emb_dim}x{q_dim}xbf16>"),
        FuncArg("%arg4", f"memref<{seq_len}x{q_dim}xbf16>"),
        FuncArg("%arg5", f"memref<{emb_dim}x{kv_dim}xbf16>"),
        FuncArg("%arg6", f"memref<{seq_len}x{kv_dim}xbf16>"),
        FuncArg("%arg7", f"memref<{emb_dim}x{kv_dim}xbf16>"),
        FuncArg("%arg8", f"memref<{seq_len}x{kv_dim}xbf16>"),
        FuncArg("%arg9", f"memref<{head_dim}xbf16>"),
        FuncArg("%arg10", f"memref<{head_dim}xbf16>"),
        FuncArg("%arg11", f"memref<{seq_len}x{q_dim}xbf16>"),
        FuncArg("%arg12", f"memref<{seq_len}x{kv_dim}xbf16>"),
        FuncArg("%arg13", f"memref<{q_total}xbf16>"),
        FuncArg("%arg14", f"memref<{k_total}xbf16>"),
        FuncArg("%arg15", f"memref<{seq_len}x{q_dim}xbf16>"),
        FuncArg("%arg16", f"memref<{seq_len}x{kv_dim}xbf16>"),
    ]

    slices = [
        KernelSlice(
            rms_ir, "r", {0: 0, 1: 1, 2: 2}, extern_syms={"@zero_vectorized_bf16"}
        ),
        KernelSlice(
            q_ir,
            "q",
            _gemm_arg_map(2, 3, 4, scratch_for[0]),
            extern_syms={"@matmul_bf16"} | _gemm_externs(q_spec),
        ),
        KernelSlice(
            k_ir,
            "k",
            _gemm_arg_map(2, 5, 6, scratch_for[1]),
            extern_syms={"@matmul_bf16"} | _gemm_externs(k_spec),
        ),
        KernelSlice(
            v_ir,
            "v",
            _gemm_arg_map(2, 7, 8, scratch_for[2]),
            extern_syms={"@matmul_bf16"} | _gemm_externs(v_spec),
        ),
        # QK-norm Q: in=q(arg4), weight=q_norm(arg9), out=q_n(arg11).
        KernelSlice(qkn_q_ir, "qn", {0: 4, 1: 9, 2: 11}, private_from=False),
        # QK-norm K: in=k(arg6), weight=k_norm(arg10), out=k_n(arg12).
        KernelSlice(qkn_k_ir, "kn", {0: 6, 1: 10, 2: 12}, private_from=False),
        # RoPE consumes the QK-norm outputs (arg11/arg12), not the raw GEMM outs.
        KernelSlice(rope_q_ir, "rq", {0: 11, 1: 13, 2: 15}, extern_syms={"@rope"}),
        KernelSlice(rope_k_ir, "rk", {0: 12, 1: 14, 2: 16}, extern_syms={"@rope"}),
    ]

    module = stitch_elf(
        "rms_qkv_qknorm_rope",
        base_args,
        slices,
        scratch_args=scratch_args,
        debug_dump_path="/tmp/debug_rms_qkv_qknorm_rope.mlir",
    )
    print(
        f"  rms_qkv_qknorm_rope module: {len(str(module).splitlines())} lines, parsed OK"
    )
    return module, scratch_for


# ===========================================================================
# DECODE (M=1) fused builder: RMSNorm + Q/K/V GEMV + per-head QK-norm + RoPE.
# 8-launch 1D ELF. Mirrors the prefill builder at M=1 (GEMV instead of GEMM).
# ===========================================================================


def build_rms_qkv_qknorm_rope_gemv_module(
    emb_dim,
    q_dim,
    kv_dim,
    n_heads,
    n_kv_heads,
    head_dim,
    tile_m=8,
    m_input=4,
    herd_m=8,
    qknorm_eps=1e-6,
    qknorm_herd_x=8,
):
    """8-launch decode ELF (all 1D — M=1 token):

    %arg0  x_in     (emb_dim,)
    %arg1  norm_w   (emb_dim,)
    %arg2  normed   (emb_dim,)
    %arg3  wq       (q_dim, emb_dim)    GEMV weight (out, in)
    %arg4  q        (q_dim,)            Q GEMV out (pre-QK-norm)
    %arg5  wk       (kv_dim, emb_dim)
    %arg6  k        (kv_dim,)
    %arg7  wv       (kv_dim, emb_dim)
    %arg8  v        (kv_dim,)           V GEMV out (final)
    %arg9  q_norm   (head_dim,)
    %arg10 k_norm   (head_dim,)
    %arg11 q_n      (q_dim,)            QK-norm Q out (RoPE input)
    %arg12 k_n      (kv_dim,)           QK-norm K out (RoPE input)
    %arg13 lut_q    (q_dim,)            RoPE Q LUT (n_heads*head_dim, position-dependent)
    %arg14 lut_k    (kv_dim,)           RoPE K LUT
    %arg15 q_roped  (q_dim,)            final RoPE Q
    %arg16 k_roped  (kv_dim,)           final RoPE K
    """
    import shared.builders.rms_gemv_rope_multi as rgr
    from shared.infra.stitching import stitch_elf, KernelSlice, FuncArg
    from matvec import build_module as build_gemv

    assert q_dim == n_heads * head_dim
    assert kv_dim == n_kv_heads * head_dim

    # RMSNorm (decode 1D) reads rgr.EPS; Qwen3 = 1e-6.
    _saved_eps = rgr.EPS
    rgr.EPS = qknorm_eps
    try:
        print("  [1/8] RMSNorm (decode 1D, eps=%g)..." % qknorm_eps)
        rms_ir = str(rgr._build_rms_1d(emb_dim, bfloat16, 16))
    finally:
        rgr.EPS = _saved_eps

    print(f"  [2/8] Q GEMV M={q_dim} K={emb_dim}...")
    q_ir = str(build_gemv(q_dim, emb_dim, tile_m, m_input, herd_m, bfloat16, bfloat16))
    print(f"  [3/8] K GEMV M={kv_dim} K={emb_dim}...")
    k_ir = str(build_gemv(kv_dim, emb_dim, tile_m, m_input, herd_m, bfloat16, bfloat16))
    print(f"  [4/8] V GEMV M={kv_dim} K={emb_dim}...")
    v_ir = str(build_gemv(kv_dim, emb_dim, tile_m, m_input, herd_m, bfloat16, bfloat16))

    print(f"  [5/8] QK-norm Q (rows={n_heads} dim={head_dim} eps={qknorm_eps})...")
    qkn_q_ir = str(
        _build_qknorm_1d(n_heads, head_dim, bfloat16, qknorm_eps, qknorm_herd_x)
    )
    print(f"  [6/8] QK-norm K (rows={n_kv_heads} dim={head_dim} eps={qknorm_eps})...")
    qkn_k_ir = str(
        _build_qknorm_1d(
            n_kv_heads,
            head_dim,
            bfloat16,
            qknorm_eps,
            herd_x=min(qknorm_herd_x, n_kv_heads),
        )
    )

    print(f"  [7/8] RoPE Q (rows={n_heads} dim={head_dim})...")
    rope_q_ir = str(
        rgr._build_rope_1d(
            n_heads, head_dim, bfloat16, herd_x=min(qknorm_herd_x, n_heads)
        )
    )
    print(f"  [8/8] RoPE K (rows={n_kv_heads} dim={head_dim})...")
    rope_k_ir = str(
        rgr._build_rope_1d(
            n_kv_heads, head_dim, bfloat16, herd_x=min(qknorm_herd_x, n_kv_heads)
        )
    )

    base_args = [
        FuncArg("%arg0", f"memref<{emb_dim}xbf16>"),
        FuncArg("%arg1", f"memref<{emb_dim}xbf16>"),
        FuncArg("%arg2", f"memref<{emb_dim}xbf16>"),
        FuncArg("%arg3", f"memref<{q_dim}x{emb_dim}xbf16>"),
        FuncArg("%arg4", f"memref<{q_dim}xbf16>"),
        FuncArg("%arg5", f"memref<{kv_dim}x{emb_dim}xbf16>"),
        FuncArg("%arg6", f"memref<{kv_dim}xbf16>"),
        FuncArg("%arg7", f"memref<{kv_dim}x{emb_dim}xbf16>"),
        FuncArg("%arg8", f"memref<{kv_dim}xbf16>"),
        FuncArg("%arg9", f"memref<{head_dim}xbf16>"),
        FuncArg("%arg10", f"memref<{head_dim}xbf16>"),
        FuncArg("%arg11", f"memref<{q_dim}xbf16>"),
        FuncArg("%arg12", f"memref<{kv_dim}xbf16>"),
        FuncArg("%arg13", f"memref<{q_dim}xbf16>"),
        FuncArg("%arg14", f"memref<{kv_dim}xbf16>"),
        FuncArg("%arg15", f"memref<{q_dim}xbf16>"),
        FuncArg("%arg16", f"memref<{kv_dim}xbf16>"),
    ]
    # GEMV func args: {0: weight (MxK), 1: input (K,), 2: output (M,)}.
    slices = [
        KernelSlice(rms_ir, "r", {0: 0, 1: 1, 2: 2}, private_from=False),
        KernelSlice(q_ir, "q", {0: 3, 1: 2, 2: 4}),
        KernelSlice(k_ir, "k", {0: 5, 1: 2, 2: 6}, private_from=False),
        KernelSlice(v_ir, "v", {0: 7, 1: 2, 2: 8}, private_from=False),
        KernelSlice(qkn_q_ir, "qn", {0: 4, 1: 9, 2: 11}, private_from=False),
        KernelSlice(qkn_k_ir, "kn", {0: 6, 1: 10, 2: 12}, private_from=False),
        KernelSlice(rope_q_ir, "rq", {0: 11, 1: 13, 2: 15}, extern_syms={"@rope"}),
        KernelSlice(rope_k_ir, "rk", {0: 12, 1: 14, 2: 16}, extern_syms={"@rope"}),
    ]
    module = stitch_elf(
        "rms_qkv_qknorm_rope_gemv",
        base_args,
        slices,
        extra_externs={
            "@zero_vectorized_bf16",
            "@matvec_vectorized_bf16_bf16",
            "@linalg_fill_bf16",
            "@rope",
        },
        debug_dump_path="/tmp/debug_rms_qkv_qknorm_rope_gemv.mlir",
    )
    print(
        f"  rms_qkv_qknorm_rope_gemv module: {len(str(module).splitlines())} lines, parsed OK"
    )
    return module


# ===========================================================================
# DECODE (M=1) fused builder, 4 launches `[2026-08-21]` (doc 57 O1, first cut).
# ===========================================================================


def build_rms_qkv_qknorm_rope_gemv4_module(
    emb_dim,
    q_dim,
    kv_dim,
    n_heads,
    n_kv_heads,
    head_dim,
    tile_m=8,
    m_input=4,
    herd_m=8,
    qknorm_eps=1e-6,
    herd_x=8,
    host_rmsnorm=False,
):
    """4-launch decode ELF: the 8-launch `build_rms_qkv_qknorm_rope_gemv_module`
    with its three GEMVs, two QK-norms and two RoPEs each collapsed to one.

    host_rmsnorm=True `[2026-08-21]` (doc 57 O1, second cut): the RMSNorm launch
    is dropped and arg0 is the already-normalized vector -- the host computes
    bf16(x * rsqrt(mean(x^2) + eps) * norm_w) over emb_dim elements (~10 us)
    instead of the device paying a ~107 us launch boundary for it. 3 launches,
    7 args: 0 normed, 1 wqkv, 2 qkv, 3 qk_norm_w, 4 qk_n, 5 lut, 6 qk_roped.

    Doc 57 measured every `air.launch` boundary at ~107 us (section 1.5), so
    the 8-launch form spends ~0.75 ms of its 1.03 ms per layer in boundaries.
    This form keeps every kernel (no new C code) and halves the count:

      1. RMSNorm   x_in x norm_w                      -> normed   (emb_dim,)
      2. QKV GEMV  normed x [wq; wk; wv]              -> qkv      (q_dim+2*kv_dim,)
      3. QK-norm   qkv[0 : q_dim+kv_dim] viewed as (n_heads+n_kv_heads, head_dim)
                   rows, weight row r = q_norm (r < n_heads) else k_norm
                                                       -> qk_n     (q_dim+kv_dim,)
      4. RoPE      qk_n x lut (the same position LUT tiled n_heads+n_kv_heads x)
                                                       -> qk_roped (q_dim+kv_dim,)

    The weights are concatenated ROW-wise by the host ONCE (`wqkv = [wq_t; wk_t;
    wv_t]`, static), the QK-norm weight is tiled once (static), and the LUT is
    one (n_heads+n_kv_heads) x head_dim tile per position instead of two. The
    QK-norm launch takes the whole qkv arg and processes its first
    n_heads+n_kv_heads rows (`in_total`), so V rides in the tail untouched.

    9 args:
    %arg0  x_in      (emb_dim,)
    %arg1  norm_w    (emb_dim,)                 static
    %arg2  normed    (emb_dim,)                 intermediate
    %arg3  wqkv      (q_dim+2*kv_dim, emb_dim)  static, rows [wq; wk; wv]
    %arg4  qkv       (q_dim+2*kv_dim,)          q | k | v  (v = qkv[q_dim+kv_dim:], host output)
    %arg5  qk_norm_w (q_dim+kv_dim,)            static, [q_norm x n_heads; k_norm x n_kv_heads]
    %arg6  qk_n      (q_dim+kv_dim,)            intermediate
    %arg7  lut       (q_dim+kv_dim,)            position LUT tiled (n_heads+n_kv_heads) x
    %arg8  qk_roped  (q_dim+kv_dim,)            q_roped | k_roped (host output)
    """
    import shared.builders.rms_gemv_rope_multi as rgr
    from shared.infra.stitching import stitch_elf, KernelSlice, FuncArg
    from matvec import build_module as build_gemv

    assert q_dim == n_heads * head_dim
    assert kv_dim == n_kv_heads * head_dim
    qk_dim = q_dim + kv_dim
    qkv_dim = q_dim + 2 * kv_dim
    n_qk_rows = n_heads + n_kv_heads
    assert n_qk_rows % herd_x == 0, (n_qk_rows, herd_x)

    if not host_rmsnorm:
        _saved_eps = rgr.EPS
        rgr.EPS = qknorm_eps
        try:
            print("  [1/4] RMSNorm (decode 1D, eps=%g)..." % qknorm_eps)
            rms_ir = str(rgr._build_rms_1d(emb_dim, bfloat16, 16))
        finally:
            rgr.EPS = _saved_eps

    print(f"  [2/4] QKV GEMV M={qkv_dim} K={emb_dim} (one launch over [wq; wk; wv])...")
    qkv_ir = str(
        build_gemv(qkv_dim, emb_dim, tile_m, m_input, herd_m, bfloat16, bfloat16)
    )
    print(
        f"  [3/4] QK-norm Q|K (rows={n_qk_rows} dim={head_dim} eps={qknorm_eps}, per-row weight)..."
    )
    qkn_ir = str(
        _build_qknorm_1d(
            n_qk_rows,
            head_dim,
            bfloat16,
            qknorm_eps,
            herd_x,
            per_row_weight=True,
            in_total=qkv_dim,
        )
    )
    print(f"  [4/4] RoPE Q|K (rows={n_qk_rows} dim={head_dim})...")
    rope_ir = str(rgr._build_rope_1d(n_qk_rows, head_dim, bfloat16, herd_x=herd_x))

    if host_rmsnorm:
        base_args = [
            FuncArg("%arg0", f"memref<{emb_dim}xbf16>"),  # normed (host)
            FuncArg("%arg1", f"memref<{qkv_dim}x{emb_dim}xbf16>"),
            FuncArg("%arg2", f"memref<{qkv_dim}xbf16>"),
            FuncArg("%arg3", f"memref<{qk_dim}xbf16>"),
            FuncArg("%arg4", f"memref<{qk_dim}xbf16>"),
            FuncArg("%arg5", f"memref<{qk_dim}xbf16>"),
            FuncArg("%arg6", f"memref<{qk_dim}xbf16>"),
        ]
        slices = [
            KernelSlice(qkv_ir, "qkv", {0: 1, 1: 0, 2: 2}),
            KernelSlice(qkn_ir, "qkn", {0: 2, 1: 3, 2: 4}, private_from=False),
            KernelSlice(rope_ir, "rp", {0: 4, 1: 5, 2: 6}, extern_syms={"@rope"}),
        ]
    else:
        base_args = [
            FuncArg("%arg0", f"memref<{emb_dim}xbf16>"),
            FuncArg("%arg1", f"memref<{emb_dim}xbf16>"),
            FuncArg("%arg2", f"memref<{emb_dim}xbf16>"),
            FuncArg("%arg3", f"memref<{qkv_dim}x{emb_dim}xbf16>"),
            FuncArg("%arg4", f"memref<{qkv_dim}xbf16>"),
            FuncArg("%arg5", f"memref<{qk_dim}xbf16>"),
            FuncArg("%arg6", f"memref<{qk_dim}xbf16>"),
            FuncArg("%arg7", f"memref<{qk_dim}xbf16>"),
            FuncArg("%arg8", f"memref<{qk_dim}xbf16>"),
        ]
        slices = [
            KernelSlice(rms_ir, "r", {0: 0, 1: 1, 2: 2}, private_from=False),
            KernelSlice(qkv_ir, "qkv", {0: 3, 1: 2, 2: 4}),
            # QK-norm takes the WHOLE qkv arg (in_total=qkv_dim) and reads rows
            # [0, n_heads+n_kv_heads) only; V in the tail is never touched.
            KernelSlice(qkn_ir, "qkn", {0: 4, 1: 5, 2: 6}, private_from=False),
            KernelSlice(rope_ir, "rp", {0: 6, 1: 7, 2: 8}, extern_syms={"@rope"}),
        ]
    module = stitch_elf(
        "rms_qkv_qknorm_rope_gemv",
        base_args,
        slices,
        extra_externs={
            "@zero_vectorized_bf16",
            "@matvec_vectorized_bf16_bf16",
            "@linalg_fill_bf16",
            "@rope",
        },
        debug_dump_path="/tmp/debug_rms_qkv_qknorm_rope_gemv4.mlir",
    )
    print(
        f"  rms_qkv_qknorm_rope_gemv{3 if host_rmsnorm else 4} module: "
        f"{len(str(module).splitlines())} lines, {3 if host_rmsnorm else 4} launches, parsed OK"
    )
    return module


# ===========================================================================
# DECODE (M=1) fused builder, 2 launches `[2026-08-22]` (doc 57 O1, second half):
# the QKV GEMV with a HEAD-ALIGNED row -> column mapping and QK-norm + RoPE
# as an in-core epilogue.
# ===========================================================================

K_PAD = 64  # row padding of the augmented weight matrix: [tag, kind, 0 ...]


@module_builder
def _build_qkv_heads_gemv(
    m,
    k,
    head_dim,
    np_dtype,
    tile_m=8,
    herd_m=8,
    eps=1e-6,
    link_with="mv_heads_hd128.o",
    out_slots=True,
):
    """GEMV C[M] = A[M,K] @ B[K] whose rows are distributed so that every
    core owns WHOLE heads, with the per-head QK-norm + RoPE epilogue applied
    in L1 before the head leaves the core (kernel mv_heads.cc).

    out_slots=True: the per-iteration head write goes to its OWN slot of a
    [n_iter, herd_m, head_dim] output (`qkv_heads_slot_gather` picks each
    head's last-chunk slot on the host). Required by the compiler: with the
    head's logical slot as the target, 16 iterations write the same region
    and `air-verify-hierarchy-locality` (strict, aircc's default) rejects
    the launch -- "iteration variable does not appear in any offset of this
    access; iterations cannot be disjoint". out_slots=False keeps that
    (rejected) logical-slot form for reference.

    Row -> core mapping. matvec.py's L2-staged tiles interleave 8-row tiles
    across the herd_m columns (launch iteration i gives column tx the LOGICAL
    rows [i*herd_m*tile_m + tx*tile_m, +tile_m)), so a head's 128 rows land on
    16 different cores. Here column tx owns the CONTIGUOUS logical block
    [tx*rows_per_col, (tx+1)*rows_per_col), rows_per_col = M / herd_m, and
    launch iteration i gives it the rows

        logical(tx, i) = tx*rows_per_col + i*tile_m + [0, tile_m)

    -- chunk (i mod chunks_per_head) of head (tx*heads_per_col + i // chunks_per_head),
    chunks_per_head = head_dim / tile_m. A core therefore sees a head as
    chunks_per_head CONSECUTIVE launch iterations, accumulates them into a
    persistent L1 head buffer, and runs the epilogue on the last chunk.

    Storage. The host stores A ITERATION-MAJOR (`qkv_heads_store_perm`): stored
    row i*herd_m*tile_m + tx*tile_m + r holds logical(tx, i)[r], so the L3 -> L2
    fetch is matvec.py's contiguous [herd_m, tile_m, K] block and the L2 stage
    keeps its 16 KB granularity (a whole-head 256 KB tile serializes the fill
    against the core's drain: 0.577 vs 0.444 ms per 8 MB single-launch GEMV,
    results/o1-epilogue-20260822/probe_gemv_variants*.json). The OUTPUT is
    written in LOGICAL order (the L2 -> L3 write is column-strided), so the
    host un-permutes nothing.

    Tag and kind. The core does not see the launch iteration (its program is a
    while(true) over lock handshakes), and a core tile has two inbound DMA
    channels, both taken (A from the memtile, B from the shim: a third stream
    fails in aiecc's router, "'aie.connect' op ... targets same dst as another
    connect op"). So every weight row carries, in a K_PAD-element padding the
    host bakes once, its chunk index within the head (TAG, a[k]) and its head
    kind (KIND, a[k+1]: 0 Q, 1 K, 2 V). The kernel writes the chunk at
    c[tag*tile_m] and, at the last tag, runs the epilogue with the kind's
    weight. The per-iteration output DMA writes the head's 128 outputs to its
    logical slot every chunk; only the last chunk's write carries the final
    values and it is the last one issued (same channel, in order).

    B is the PACKED vector [normed (K) | lut (head_dim) | q_norm (head_dim) |
    k_norm (head_dim)], fetched whole per iteration (broadcast).

    Func signature: (A: [M, K + K_PAD], B: [K + 3*head_dim], OUT: [M]).
    """
    assert m % herd_m == 0, (m, herd_m)
    rows_per_col = m // herd_m
    assert rows_per_col % head_dim == 0, (rows_per_col, head_dim)
    assert head_dim % tile_m == 0, (head_dim, tile_m)
    chunks_per_head = head_dim // tile_m
    n_iter = rows_per_col // tile_m
    assert k % 64 == 0, k
    k_pad = k + K_PAD
    tail = 3 * head_dim
    b_total = k + tail

    from air.dialects.func import CallOp
    from air.ir import StringAttr, UnitAttr, FloatAttr

    xrt_dtype = type_mapper(np_dtype)
    f32 = F32Type.get()

    memrefTyA = MemRefType.get([m, k_pad], xrt_dtype)
    memrefTyB = MemRefType.get([b_total], xrt_dtype)
    out_total = n_iter * herd_m * head_dim if out_slots else m
    memrefTyOut = MemRefType.get([out_total], xrt_dtype)

    l2_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L2)
    l2MemrefTyA = MemRefType.get([herd_m, tile_m, k_pad], xrt_dtype, memory_space=l2_mem_space)
    l2MemrefTyOut = MemRefType.get([herd_m, head_dim], xrt_dtype, memory_space=l2_mem_space)

    l1_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1MemrefTyA = MemRefType.get([tile_m, k_pad], xrt_dtype, memory_space=l1_mem_space)
    l1MemrefTyB = MemRefType.get([b_total], xrt_dtype, memory_space=l1_mem_space)
    l1MemrefTyHead = MemRefType.get([head_dim], xrt_dtype, memory_space=l1_mem_space)

    chunk_func = FuncOp(
        "qkv_heads_chunk_bf16",
        ([T.i32(), T.i32(), l1MemrefTyA, l1MemrefTyB, l1MemrefTyHead, l1MemrefTyHead, f32], []),
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
                    0, 1,
                    [AffineExpr.get_mul(AffineSymbolExpr.get(0), AffineConstantExpr.get(herd_m * tile_m))],
                )
                stored_offset_m = affine_apply(stored_map, [ivx_s])
                # logical row offset WITHIN a column of this iteration's head: (i // chunks_per_head) * head_dim
                head_map = AffineMap.get(
                    0, 1,
                    [AffineExpr.get_mul(
                        AffineExpr.get_floor_div(AffineSymbolExpr.get(0), AffineConstantExpr.get(chunks_per_head)),
                        AffineConstantExpr.get(head_dim))],
                )
                head_offset_m = affine_apply(head_map, [ivx_s])

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
                def herd_body(_tx, _ty, _sx, _sy, _l1_a, _l1_b, _l1_c, _l1_out, _l2_a, _l3_b, _l2_out):
                    # B (packed): L3 -> L1 (broadcast, repeat channel).
                    dma_memcpy_nd(_l1_b, _l3_b, src_offsets=[], src_sizes=[b_total], src_strides=[1])
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
                    CallOp(chunk_func, [m_const, k_const, _l1_a, _l1_b, _l1_c, _l1_out, eps_c])
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

                if out_slots:
                    # L2 -> L3: this iteration's own [herd_m * head_dim] slot.
                    slot_map = AffineMap.get(
                        0, 1,
                        [AffineExpr.get_mul(AffineSymbolExpr.get(0), AffineConstantExpr.get(herd_m * head_dim))],
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
                else:
                    # L2 -> L3: every column's head slot, logical order (column stride rows_per_col).
                    dma_memcpy_nd(
                        l3_out_s,
                        l2_out,
                        dst_offsets=[0, head_offset_m],
                        dst_sizes=[herd_m, head_dim],
                        dst_strides=[rows_per_col, 1],
                        src_offsets=[0, 0],
                        src_sizes=[herd_m, head_dim],
                        src_strides=[head_dim, 1],
                    )

                for buf in (l2_a, l2_out, l1_a, l1_b, l1_c, l1_out):
                    DeallocOp(buf)


@module_builder
def _build_qkv_heads_gemv_wholehead(
    m,
    k,
    head_dim,
    n_q_rows,
    n_qk_rows,
    np_dtype,
    m_input=8,
    herd_m=8,
    eps=1e-6,
    link_with="mv_heads_hd128.o",
    fill_chunks=True,
):
    """The whole-head-per-iteration form of the head-aligned GEMV (study
    variant): launch iteration i gives column tx the logical rows
    [tx*rows_per_col + i*head_dim, +head_dim) -- ONE head -- through a
    [herd_m, head_dim, K] L2 tile (256 KB per memtile at K = 1024), the L1 C
    tile is the head, and the epilogue runs after the head_dim/m_input chunk
    calls. The epilogue kind is per column (n_q_rows / n_qk_rows must be whole
    columns). No weight permutation or padding; outputs in logical order.

    fill_chunks=True: the L3 -> L2 fill is a loop of head_dim/m_input
    sub-tile DMAs (one per m_input rows) instead of one whole-tile DMA, meant
    to let the memtile hand sub-tiles to the core while the rest of the tile
    streams in. MEASURED NO DIFFERENT (devq 552: 0.588 ms either way vs
    matvec.py's 0.444; the memtile keeps one lock cycle per tile), which is
    why production is the tagged-chunk form `_build_qkv_heads_gemv` (0.492).
    Kept as the record of the study (results/o1-epilogue-20260822/).

    Func signature: (A: [M, K], B: [K + 3*head_dim], OUT: [M]).
    """
    assert m % herd_m == 0, (m, herd_m)
    rows_per_col = m // herd_m
    assert rows_per_col % head_dim == 0, (rows_per_col, head_dim)
    heads_per_col = rows_per_col // head_dim
    assert head_dim % m_input == 0, (head_dim, m_input)
    assert k % 64 == 0, k
    for name, rows in (("n_q_rows", n_q_rows), ("n_qk_rows", n_qk_rows)):
        assert rows % rows_per_col == 0, (name, rows, rows_per_col)
    n_q_cols = n_q_rows // rows_per_col
    n_qk_cols = n_qk_rows // rows_per_col
    tile_m = head_dim
    n_chunks = tile_m // m_input
    tail = 3 * head_dim
    b_total = k + tail

    from air.dialects.func import CallOp
    from air.ir import StringAttr, UnitAttr, FloatAttr

    xrt_dtype = type_mapper(np_dtype)
    f32 = F32Type.get()
    memrefTyA = MemRefType.get([m, k], xrt_dtype)
    memrefTyB = MemRefType.get([b_total], xrt_dtype)
    memrefTyOut = MemRefType.get([m], xrt_dtype)
    l2_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L2)
    l2MemrefTyA = MemRefType.get([herd_m, tile_m, k], xrt_dtype, memory_space=l2_mem_space)
    l2MemrefTyOut = MemRefType.get([herd_m, tile_m], xrt_dtype, memory_space=l2_mem_space)
    l1_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1MemrefTyA = MemRefType.get([m_input, k], xrt_dtype, memory_space=l1_mem_space)
    l1MemrefTyB = MemRefType.get([b_total], xrt_dtype, memory_space=l1_mem_space)
    l1MemrefTyHead = MemRefType.get([tile_m], xrt_dtype, memory_space=l1_mem_space)
    l1MemrefTyScratch = MemRefType.get([tail], xrt_dtype, memory_space=l1_mem_space)

    chunk_func = FuncOp(
        "qkv_heads_chunk_scratch_bf16",
        ([T.i32(), T.i32(), T.i32(), l1MemrefTyA, l1MemrefTyB, l1MemrefTyHead, l1MemrefTyScratch], []),
        visibility="private",
    )
    epilogue_func = FuncOp(
        "qknorm_rope_head_bf16",
        ([l1MemrefTyHead, l1MemrefTyScratch, l1MemrefTyHead, f32, T.i32()], []),
        visibility="private",
    )
    for func in (chunk_func, epilogue_func):
        func.attributes["link_with"] = StringAttr.get(link_with)
        func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    @FuncOp.from_py_func(memrefTyA, memrefTyB, memrefTyOut)
    def matvec_heads(arg_a, arg_b, arg_out):
        @launch(operands=[arg_a, arg_b, arg_out], sizes=[heads_per_col, 1])
        def launch_body(l_ivx, l_ivy, l_sx, l_sy, l3_a, l3_b, l3_out):
            @segment(name="qkvh_seg", operands=[l_ivx, l3_a, l3_b, l3_out])
            def segment_body(ivx_s, l3_a_s, l3_b_s, l3_out_s):
                ivx_map = AffineMap.get(
                    0, 1, [AffineExpr.get_mul(AffineSymbolExpr.get(0), AffineConstantExpr.get(tile_m))]
                )
                launch_offset_m = affine_apply(ivx_map, [ivx_s])
                l2_a = AllocOp(l2MemrefTyA, [], [])
                l2_out = AllocOp(l2MemrefTyOut, [], [])
                l1_a = AllocOp(l1MemrefTyA, [], [])
                l1_b = AllocOp(l1MemrefTyB, [], [])
                l1_c = AllocOp(l1MemrefTyHead, [], [])
                l1_scratch = AllocOp(l1MemrefTyScratch, [], [])
                l1_out = AllocOp(l1MemrefTyHead, [], [])

                if fill_chunks:
                    # head rows of every column, m_input rows at a time
                    for j in range_(0, n_chunks):
                        sub_map = AffineMap.get(
                            0, 2,
                            [AffineExpr.get_add(
                                AffineSymbolExpr.get(0),
                                AffineExpr.get_mul(AffineSymbolExpr.get(1), AffineConstantExpr.get(m_input)))],
                        )
                        row_off = affine_apply(sub_map, [launch_offset_m, j])
                        j_off = affine_apply(
                            AffineMap.get(0, 1, [AffineExpr.get_mul(AffineSymbolExpr.get(0), AffineConstantExpr.get(m_input))]),
                            [j],
                        )
                        dma_memcpy_nd(
                            l2_a,
                            l3_a_s,
                            dst_offsets=[0, j_off, 0],
                            dst_sizes=[herd_m, m_input, k],
                            dst_strides=[tile_m * k, k, 1],
                            src_offsets=[0, row_off, 0],
                            src_sizes=[herd_m, m_input, k],
                            src_strides=[rows_per_col * k, k, 1],
                        )
                        yield_([])
                else:
                    dma_memcpy_nd(
                        l2_a,
                        l3_a_s,
                        src_offsets=[0, launch_offset_m, 0],
                        src_sizes=[herd_m, tile_m, k],
                        src_strides=[rows_per_col * k, k, 1],
                    )

                @herd(
                    name="qkvh_herd",
                    sizes=[herd_m, 1],
                    operands=[l1_a, l1_b, l1_c, l1_scratch, l1_out, l2_a, l3_b_s, l2_out],
                )
                def herd_body(_tx, _ty, _sx, _sy, _l1_a, _l1_b, _l1_c, _l1_scratch, _l1_out, _l2_a, _l3_b, _l2_out):
                    for j_m in range_(0, n_chunks):
                        j_m_offset = affine_apply(
                            AffineMap.get(0, 1, [AffineExpr.get_mul(AffineSymbolExpr.get(0), AffineConstantExpr.get(m_input))]),
                            [j_m],
                        )
                        dma_memcpy_nd(_l1_b, _l3_b, src_offsets=[], src_sizes=[b_total], src_strides=[1])
                        dma_memcpy_nd(
                            _l1_a, _l2_a,
                            src_offsets=[_tx, j_m_offset, 0],
                            src_sizes=[1, m_input, k],
                            src_strides=[tile_m * k, k, 1],
                        )
                        row_offset_i32 = arith.index_cast(T.i32(), j_m_offset)
                        m_const = arith.ConstantOp(IntegerAttr.get(T.i32(), m_input), None)
                        k_const = arith.ConstantOp(IntegerAttr.get(T.i32(), k), None)
                        CallOp(chunk_func, [m_const, k_const, row_offset_i32, _l1_a, _l1_b, _l1_c, _l1_scratch])
                        yield_([])
                    two = arith.ConstantOp(IntegerAttr.get(T.i32(), 2), None)
                    c_q = arith.ConstantOp.create_index(n_q_cols)
                    c_qk = arith.ConstantOp.create_index(n_qk_cols)
                    is_q = arith.extui(T.i32(), arith.cmpi(arith.CmpIPredicate.ult, _tx, c_q))
                    is_qk = arith.extui(T.i32(), arith.cmpi(arith.CmpIPredicate.ult, _tx, c_qk))
                    kind = arith.subi(arith.subi(two, is_qk), is_q)
                    eps_c = arith.ConstantOp(f32, FloatAttr.get(f32, eps))
                    CallOp(epilogue_func, [_l1_c, _l1_scratch, _l1_out, eps_c, kind])
                    dma_memcpy_nd(
                        _l2_out, _l1_out,
                        dst_offsets=[_tx, 0], dst_sizes=[1, tile_m], dst_strides=[tile_m, 1],
                        src_offsets=[], src_sizes=[tile_m], src_strides=[1],
                    )

                herd_body.attributes["link_with"] = StringAttr.get(link_with)

                dma_memcpy_nd(
                    l3_out_s, l2_out,
                    dst_offsets=[0, launch_offset_m], dst_sizes=[herd_m, tile_m], dst_strides=[rows_per_col, 1],
                    src_offsets=[0, 0], src_sizes=[herd_m, tile_m], src_strides=[tile_m, 1],
                )
                for buf in (l2_a, l2_out, l1_a, l1_b, l1_c, l1_scratch, l1_out):
                    DeallocOp(buf)


def qkv_heads_store_perm(m, herd_m, tile_m):
    """Index array P with A_stored = A_logical[P]: stored row
    i*herd_m*tile_m + tx*tile_m + r = logical row tx*rows_per_col + i*tile_m + r."""
    rows_per_col = m // herd_m
    n_iter = rows_per_col // tile_m
    perm = np.empty(m, dtype=np.int64)
    for i in range(n_iter):
        for tx in range(herd_m):
            base = i * herd_m * tile_m + tx * tile_m
            perm[base:base + tile_m] = tx * rows_per_col + i * tile_m + np.arange(tile_m)
    return perm


def qkv_heads_slot_gather(m, herd_m, head_dim, tile_m=8):
    """Index array G with out_logical = slots[G] for out_slots=True: head h of
    column tx is final in the slot of its last chunk's iteration
    (h*chunks_per_head + chunks_per_head - 1), at [tx*head_dim, +head_dim)."""
    rows_per_col = m // herd_m
    cph = head_dim // tile_m
    g = np.empty(m, dtype=np.int64)
    for tx in range(herd_m):
        for h in range(rows_per_col // head_dim):
            it = h * cph + cph - 1
            lo = tx * rows_per_col + h * head_dim
            g[lo:lo + head_dim] = it * herd_m * head_dim + tx * head_dim + np.arange(head_dim)
    return g


def qkv_heads_row_map(m, herd_m, head_dim, tile_m=8):
    """The logical row -> (column, launch iteration) mapping of
    `_build_qkv_heads_gemv`, as (column, iteration, row_lo, row_hi) blocks."""
    rows_per_col = m // herd_m
    blocks = []
    for tx in range(herd_m):
        for i in range(rows_per_col // tile_m):
            lo = tx * rows_per_col + i * tile_m
            blocks.append((tx, i, lo, lo + tile_m))
    return blocks


def qkv_heads_augment_weight(w_logical, q_rows, qk_rows, head_dim, herd_m=8, tile_m=8):
    """The static A of `_build_qkv_heads_gemv` from the logical [wq; wk; wv]
    (M, K): rows permuted iteration-major and padded by K_PAD with
    [tag, kind, 0...] per row (tag = the row's chunk index within its head,
    kind = 0 Q / 1 K / 2 V). Done once per layer by the host."""
    m, k = w_logical.shape
    perm = qkv_heads_store_perm(m, herd_m, tile_m)
    aug = np.zeros((m, k + K_PAD), dtype=bfloat16)
    aug[:, :k] = np.asarray(w_logical, dtype=bfloat16)[perm]
    logical = perm  # logical row of each stored row
    tag = (logical % head_dim) // tile_m
    kind = np.where(logical < q_rows, 0, np.where(logical < qk_rows, 1, 2))
    aug[:, k] = tag.astype(bfloat16)
    aug[:, k + 1] = kind.astype(bfloat16)
    return np.ascontiguousarray(aug)


def build_rms_qkv_qknorm_rope_gemv2_module(
    emb_dim,
    q_dim,
    kv_dim,
    n_heads,
    n_kv_heads,
    head_dim,
    tile_m=None,
    herd_m=None,
    qknorm_eps=1e-6,
    link_with=None,
):
    """2-launch decode ELF (doc 57 O1, second half): RMSNorm, then ONE GEMV
    over [wq; wk; wv] whose columns own whole heads and whose cores apply
    QK-norm + RoPE to the Q|K heads in L1 before the head leaves the core
    (`_build_qkv_heads_gemv`, kernel mv_heads.cc). The 4-launch form's
    separate QK-norm and RoPE launches -- two ~107 us boundaries -- are gone,
    and so are its qk_n intermediate and the 24x-tiled LUT.

    5 args:
    %arg0  x_in   (emb_dim,)
    %arg1  norm_w (emb_dim,)                         static
    %arg2  bvec   (emb_dim + 3*head_dim,)            [normed | lut | q_norm | k_norm]:
                                                     the RMSNorm launch writes [0, emb_dim);
                                                     the host fills the tail every call
    %arg3  wqkv_aug (q_dim+2*kv_dim, emb_dim+K_PAD)  static: `qkv_heads_augment_weight`
                                                     of the row-packed [wq; wk; wv]
    %arg4  out    (qkv2_out_total,)                  per-iteration head slots; the host's
                                                     `qkv2_gather` reads q_roped | k_roped | v
    """
    import shared.builders.rms_gemv_rope_multi as rgr
    from shared.infra.stitching import stitch_elf, KernelSlice, FuncArg
    from shared.infra.external_kernels import mv_heads_object_name

    assert q_dim == n_heads * head_dim
    assert kv_dim == n_kv_heads * head_dim
    qkv_dim = q_dim + 2 * kv_dim
    b_total = emb_dim + 3 * head_dim
    if link_with is None:
        link_with = mv_heads_object_name(head_dim)
    tile_m = QKV2_TILE_M if tile_m is None else tile_m
    herd_m = QKV2_HERD_M if herd_m is None else herd_m
    assert (tile_m, herd_m) == (QKV2_TILE_M, QKV2_HERD_M), "the host hooks (qkv2_*) assume these"

    _saved_eps = rgr.EPS
    rgr.EPS = qknorm_eps
    try:
        print("  [1/2] RMSNorm (decode 1D, eps=%g, into the packed B head)..." % qknorm_eps)
        rms_ir = str(rgr._build_rms_1d(emb_dim, bfloat16, 16, out_total=b_total))
    finally:
        rgr.EPS = _saved_eps

    print(
        f"  [2/2] head-aligned QKV GEMV M={qkv_dim} K={emb_dim}(+{K_PAD} pad) "
        f"tile_m={tile_m} (+ in-core QK-norm/RoPE on the Q|K heads)..."
    )
    gemv_ir = str(
        _build_qkv_heads_gemv(
            qkv_dim, emb_dim, head_dim, bfloat16,
            tile_m=tile_m, herd_m=herd_m, eps=qknorm_eps, link_with=link_with,
        )
    )

    base_args = [
        FuncArg("%arg0", f"memref<{emb_dim}xbf16>"),
        FuncArg("%arg1", f"memref<{emb_dim}xbf16>"),
        FuncArg("%arg2", f"memref<{b_total}xbf16>"),
        FuncArg("%arg3", f"memref<{qkv_dim}x{emb_dim + K_PAD}xbf16>"),
        FuncArg("%arg4", f"memref<{qkv2_out_total(qkv_dim, head_dim)}xbf16>"),
    ]
    slices = [
        KernelSlice(rms_ir, "r", {0: 0, 1: 1, 2: 2}, private_from=False),
        # GEMV func args: {0: A (wqkv_aug), 1: B (packed bvec), 2: OUT}.
        KernelSlice(gemv_ir, "qh", {0: 3, 1: 2, 2: 4}),
    ]
    module = stitch_elf(
        "rms_qkv_qknorm_rope_gemv",
        base_args,
        slices,
        extra_externs={"@zero_vectorized_bf16", "@qkv_heads_chunk_bf16"},
        debug_dump_path="/tmp/debug_rms_qkv_qknorm_rope_gemv2.mlir",
    )
    print(
        f"  rms_qkv_qknorm_rope_gemv2 module: {len(str(module).splitlines())} lines, "
        f"2 launches, parsed OK"
    )
    return module


# ---------------------------------------------------------------------------
# The 2-launch form's host-side hooks (what shared.infra.decode_qkv4's
# prep_weights_2 / call_args_2 / split_outputs_2 use). They describe the GEMV
# form `build_rms_qkv_qknorm_rope_gemv2_module` builds: the tagged-chunk GEMV
# (`_build_qkv_heads_gemv`, tile_m 8, iteration-major storage with the
# [tag, kind] row padding, per-iteration output slots).
# ---------------------------------------------------------------------------

QKV2_TILE_M = 8
QKV2_HERD_M = 8


def qkv2_prep_weight(w_logical, q_dim, qk_dim, head_dim):
    """Static weight of the 2-launch ELF from the logical [wq; wk; wv] (M, K)."""
    return qkv_heads_augment_weight(w_logical, q_dim, qk_dim, head_dim, QKV2_HERD_M, QKV2_TILE_M)


def qkv2_out_total(qkv_dim, head_dim):
    """Length of the ELF's output arg (per-iteration head slots)."""
    rows_per_col = qkv_dim // QKV2_HERD_M
    return (rows_per_col // QKV2_TILE_M) * QKV2_HERD_M * head_dim


def qkv2_gather(qkv_dim, head_dim):
    """Index array mapping the output arg to q_roped | k_roped | v (or None)."""
    return qkv_heads_slot_gather(qkv_dim, QKV2_HERD_M, head_dim, QKV2_TILE_M)
