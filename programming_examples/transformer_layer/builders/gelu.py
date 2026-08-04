# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The FFN activation stage: tanh-approximation GeLU over the external kernel.

CONTRACT
    ``build_gelu_module(rows, cols, ...)`` returns an ``air.ir.Module`` whose
    single function takes two ``[rows, cols]`` bf16 memrefs and computes
    ``out = gelu(in)`` element-wise. Both arguments stay 2-D at the L3 boundary
    and are collapsed to 1-D inside the launch, so the GEMM launches on either
    side of it read and write the same buffers as 2-D without an
    ``expand_shape``.

WHY THIS IS ITS OWN MODULE
    It is the seam ``ffn.py`` splits on (porting convention 5). iron's
    ``ffn/design.py`` is 1096 lines with up-projection, activation and
    down-projection interleaved in one file; the activation is the one stage
    that is neither a GEMM nor registry-tiled, so it lifts out cleanly and is
    reusable by any other staged FFN.

IT IS THE TANH APPROXIMATION, NOT ERF GeLU
    ``ffn_gelu_bf16`` calls ``gelu_tanh_approx_bf16``:

        0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    what HuggingFace calls ``gelu_new`` / ``gelu_pytorch_tanh``. iron's oracle
    calls ``torch.nn.functional.gelu``, whose DEFAULT is the exact erf form; at
    iron's 4e-2 tolerance the difference hides, at ``rtol = 1.6e-2`` it does not.
    ``gelu_tanh_reference`` below is therefore the tanh form deliberately. Do
    not "fix" it back to erf.

FOOTGUNS
    - The device kernel carries its intermediates in bf16 -- ``x^2``, ``x^3``,
      the inner sum and ``1 + tanh`` are each rounded. The reference here is
      pure FP32 and does NOT reproduce that, on purpose: reproducing it would
      hide exactly the error it introduces. The gap is real and it is what the
      ``atol`` is sized against. It is worst where ``1 + tanh(u)`` cancels
      (``x`` around -2 to -3): bf16 near 1.0 has a spacing of 2^-8, so a
      ``1 + tanh`` of 0.012 carries ~8% relative error while the absolute error
      stays around 1e-3.
    - ``cols`` must be a multiple of ``vector_size`` (16). The kernel truncates
      its trailing partial vector and leaves those elements UNTOUCHED -- stale
      bytes from whatever last used the buffer, not zeros.
    - ``rows * cols`` must divide by ``herd_x``, and ``herd_y`` is fixed at 1:
      two shim DMAs per tile across all 8 columns is already the channel budget.
    - The object named by ``KERNEL_OBJ`` must be in the working directory when
      aiecc links. It is ``encoder.cc``'s FFN half only -- built with
      ``build_addnorm=False`` so it cannot collide with ``addnorm_ffn.o``, which
      defines ``ffn_gelu_bf16`` too.
"""

import numpy as np
from ml_dtypes import bfloat16

from air.ir import *  # noqa: F403
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *  # noqa: F403
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.memref import collapse_shape as memref_collapse_shape
from air.dialects.func import FuncOp, CallOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import type_mapper

range_ = for_

# encoder.cc built with -DBUILD_FFN and NOT -DBUILD_ADDNORM. The addnorm half
# would collide with addnorm_ffn.o on ffn_gelu_bf16 and
# ffn_eltwise_add_bf16_vector, so the FFN path links its own narrower object.
KERNEL_OBJ = "encoder_ffn.o"

GELU_SYMBOL = "ffn_gelu_bf16"

# The vector width gelu_tanh_approx_bf16 steps by. A `cols` that is not a
# multiple of this silently leaves the remainder unwritten.
GELU_VEC_LEN = 16

# sqrt(2/pi) and the cubic coefficient, as the kernel broadcasts them.
_SQRT_2_OVER_PI = 0.7978845608028654
_GELU_CUBIC_COEFF = 0.044715


@module_builder
def build_gelu_module(rows, cols, np_dtype=bfloat16, herd_x=8, tile_n=None):
    """Build the element-wise GeLU launch over an ``herd_x x 1`` herd.

    Args:
        rows, cols: L3 tensor shape, 2-D in and 2-D out. ``cols`` must be a
            multiple of ``GELU_VEC_LEN``; ``rows * cols`` must divide by
            ``herd_x``.
        np_dtype: element type; bf16, matching the kernel's C linkage.
        herd_x: AIE columns. Two shim DMAs per tile (in, out).
        tile_n: elements resident in L1 per ``ffn_gelu_bf16`` call. Defaults to
            one row. ``2 * tile_n * 2`` bytes must fit L1 with ping-pong.

    Returns:
        air.ir.Module with one function ``ffn_gelu_2d(x, y)``.
    """
    if cols % GELU_VEC_LEN:
        raise ValueError(
            f"cols ({cols}) must be a multiple of GELU_VEC_LEN ({GELU_VEC_LEN}); "
            "the kernel truncates its trailing partial vector"
        )
    n = rows * cols
    if n % herd_x:
        raise ValueError(f"rows*cols ({n}) must be divisible by herd_x ({herd_x})")
    chunk_size = n // herd_x
    if tile_n is None:
        tile_n = cols
    if chunk_size % tile_n:
        raise ValueError(
            f"per-tile chunk ({chunk_size}) must be divisible by tile_n ({tile_n})"
        )

    xrt_dtype = type_mapper(np_dtype)
    l3_2d_ty = MemRefType.get([rows, cols], xrt_dtype)
    l3_1d_ty = MemRefType.get([n], xrt_dtype)

    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_ty = MemRefType.get([tile_n], xrt_dtype, memory_space=l1_space)

    gelu_func = FuncOp(
        GELU_SYMBOL,
        ([l1_ty, l1_ty, T.i32()], []),
        visibility="private",
    )
    gelu_func.attributes["link_with"] = StringAttr.get(KERNEL_OBJ)
    gelu_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    # flat element offset = loop_iv + tx * chunk_size
    offset_map = AffineMap.get(
        0,
        2,
        [
            AffineExpr.get_add(
                AffineSymbolExpr.get(0),
                AffineExpr.get_mul(
                    AffineSymbolExpr.get(1), AffineConstantExpr.get(chunk_size)
                ),
            )
        ],
    )

    @FuncOp.from_py_func(l3_2d_ty, l3_2d_ty)
    def ffn_gelu_2d(arg0, arg1):

        @launch(operands=[arg0, arg1])
        def gelu_launch(l_in, l_out):
            in_flat = memref_collapse_shape(l3_1d_ty, l_in, [[0, 1]])
            out_flat = memref_collapse_shape(l3_1d_ty, l_out, [[0, 1]])

            @segment(name="gelu_seg", operands=[in_flat, out_flat])
            def gelu_seg(s_in, s_out):

                @herd(name="gelu_herd", sizes=[herd_x, 1], operands=[s_in, s_out])
                def gelu_body(_tx, _ty, _sx, _sy, h_in, h_out):
                    l1_in = AllocOp(l1_ty, [], [])
                    l1_out = AllocOp(l1_ty, [], [])
                    size_i32 = ConstantOp(T.i32(), tile_n)

                    for loop_iv in range_(0, chunk_size, tile_n):
                        offset = affine_apply(offset_map, [loop_iv, _tx])
                        dma_memcpy_nd(
                            l1_in,
                            h_in,
                            src_offsets=[offset],
                            src_sizes=[tile_n],
                            src_strides=[1],
                        )
                        CallOp(gelu_func, [l1_in, l1_out, size_i32])
                        dma_memcpy_nd(
                            h_out,
                            l1_out,
                            dst_offsets=[offset],
                            dst_sizes=[tile_n],
                            dst_strides=[1],
                        )
                        yield_([])

                    DeallocOp(l1_in)
                    DeallocOp(l1_out)

                gelu_body.attributes["link_with"] = StringAttr.get(KERNEL_OBJ)


def gelu_tanh_reference(x):
    """FP32 tanh-approximation GeLU, rounded once to the input dtype.

    The TANH approximation, matching ``gelu_tanh_approx_bf16`` and HuggingFace's
    ``gelu_pytorch_tanh`` -- not ``torch.nn.functional.gelu``'s default erf
    form, which differs by up to ~1e-3 absolute around |x| = 2 and would not
    survive ``rtol = 1.6e-2`` on its own.

    FP32 throughout, so the bf16 intermediates the device carries show up as
    error rather than being cancelled out. See the module footgun note.
    """
    xf = x.astype(np.float32)
    inner = np.float32(_SQRT_2_OVER_PI) * (
        xf + np.float32(_GELU_CUBIC_COEFF) * xf * xf * xf
    )
    return (np.float32(0.5) * xf * (np.float32(1.0) + np.tanh(inner))).astype(x.dtype)
