//===- encoder.cc -----------------------------------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// AIE2P device kernels for one transformer encoder layer: the FFN block
// (bf16 blocked matmul + GeLU + residual add) and the AddNorm block
// (layer-norm reductions and the fused add/normalize/scale variants).
//
// The two blocks are independently gated so a single source file can be
// compiled into two different objects that live in the same ELF without
// dragging each other's code in.
//
// FILE LAYOUT
//   This file holds the contract, the build gating and the extern "C" entry
//   points. The template bodies they call live in two sibling sources included
//   below -- encoder_matmul.cc (the 2x2-expanded aie::mmul microkernels) and
//   encoder_layer_norm.cc (the LayerNorm reductions, fused and staged). The
//   split keeps each source inside the ~800-line module-size convention; it is
//   still ONE translation unit and ONE object, the same way
//   matrix_multiplication/bf16_in_fp32_out/mm_aie2p.cc includes zero.cc.
//   Everything below about flags, tile shapes and ABI applies to all three.
//
// CONTRACT
//   - Every entry point has C linkage and a block-specific prefix (ffn_* /
//     ln_* / fused_*). The prefixes are the ABI; the AIR/AIE lowering binds to
//     these exact strings, so renaming one is a breaking change even though
//     nothing in this file references them.
//   - The FFN matmuls consume PRE-TILED operands. A, B and C are sequences of
//     row-major r x s / s x t / r x t microtiles, and the microtiles themselves
//     are laid out row-major. Handing these kernels plain row-major matrices
//     produces silently wrong numbers, not a crash.
//   - The accumulating matmuls (ffn_matmul_bf16_bf16_up_proj,
//     ffn_matmul_with_acc_bf16_bf16_down_proj) READ C/pAcc before writing it.
//     C must have been zeroed (ffn_zero_bf16_*) or hold a valid partial sum.
//     The ffn_matmul_init_* variants zero C themselves and so do not.
//   - All B operands are row-major and all C operands are row-major. This file
//     deliberately carries no transposed-operand path.
//
// FOOTGUNS
//   - Nothing is emitted unless -DBUILD_FFN and/or -DBUILD_ADDNORM is passed on
//     the command line. With neither, this file compiles cleanly to an object
//     that exports ZERO symbols and links silently as a no-op -- the failure
//     shows up much later as a device buffer full of stale data, not as a link
//     error. Always check the built object with `llvm-nm --defined-only`.
//   - -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 is MANDATORY for BUILD_FFN; a
//     static_assert fires without it. It must be a -D on the compiler command
//     line and NOT a #define in this file, because aie_api reads it while
//     selecting the mmul implementation -- i.e. it must already be visible when
//     <aie_api/aie.hpp> is preprocessed, which is before any line of this file.
//     Defining it after the include is silently ineffective for aie_api even
//     though it satisfies the static_assert.
//   - DEBUG_AIE_KERNELS, IF DEFINED AT ALL, must be defined to 0 or 1. The
//     debug bodies are selected with `#if DEBUG_AIE_KERNELS == 0`, so a bare
//     -DDEBUG_AIE_KERNELS (which expands to the empty token sequence) makes
//     that test ill-formed. Use -DDEBUG_AIE_KERNELS=0 or =1, or leave it
//     undefined for the real numerics. Note that defining it at all replaces
//     the layer-norm math with a passthrough and turns ffn_gelu_bf16 into a
//     no-op that does not even write its output buffer.
//   - DIM_M / DIM_K / DIM_N are the L1 microkernel tile dims, defaulting to
//     64/64/64. They are compile-time only and the static_asserts enforce:
//       BUILD_FFN     : DIM_M % 16 == 0, DIM_N % 16 == 0, DIM_K % 8 == 0
//                       (the matmuls expand 2x in m and n over 8x8x8 mmuls)
//       BUILD_ADDNORM : DIM_M % 8 == 0, DIM_K % 8 == 0
//     Do not relax a static_assert to make a shape fit; change the -D values.
//   - DIM_* are used ONLY by the BUILD_FFN entry points and by the tile-shape
//     asserts of the BUILD_ADDNORM ones. The AddNorm kernels take their actual
//     working extents (cols, rows_to_process, tileWidth, tileHeight, col_idx)
//     as RUNTIME arguments, so recompiling with different DIM_* does not change
//     how much data they touch.
//   - "zero.cc" resolves via -I to
//     matrix_multiplication/bf16_in_fp32_out/zero.cc, whose zero_vectorized
//     takes FOUR template parameters <T, M, N, r> -- the vector width r is
//     explicit here, not derived from sizeof(T). Pass 32 for bfloat16 and 16
//     for float; getting it wrong is a compile error at best and a partially
//     zeroed buffer at worst.
//
//===----------------------------------------------------------------------===//

#include <aie_kernels/aie_kernel_utils.h>

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "elementwise.cc"
#include "zero.cc"

// The template bodies live in two sibling sources, included here so this stays
// one translation unit. The seam is by role: matmul microkernels, LayerNorm
// reductions, then the extern "C" entry points below. They are independent of
// each other, so the include order does not matter.
#include "encoder_layer_norm.cc"
#include "encoder_matmul.cc"

extern "C" {

// If you want to compile microkernels with different inner tile sizes,
// define DIM_M, DIM_K and DIM_N at compile time using -DDIM_M 32 etc.
// These dimensions must be divisible by the r, s, t dimensions used in
// the kernels.

#ifndef DIM_M
#define DIM_M 64
#endif

#ifndef DIM_K
#define DIM_K 64
#endif

#ifndef DIM_N
#define DIM_N 64
#endif

#ifdef BUILD_FFN
void ffn_zero_bf16_up_proj(bfloat16 *C) {
  zero_vectorized<bfloat16, DIM_M, DIM_N, 32>(C);
}

void ffn_zero_bf16_down_proj(bfloat16 *C) {
  zero_vectorized<bfloat16, DIM_M, DIM_K, 32>(C);
}

void ffn_matmul_init_bf16_bf16_up_proj(const bfloat16 *A, const bfloat16 *B,
                                       bfloat16 *C) {
#ifndef AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
  static_assert(false,
                "AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 must be defined for "
                "this kernel");
#endif
  constexpr int r = 8;
  constexpr int s = 8;
  constexpr int t = 8;

  static_assert(DIM_M % (2 * r) == 0);
  static_assert(DIM_N % (2 * t) == 0);
  static_assert(DIM_K % s == 0);

  ::aie::set_rounding(aie::rounding_mode::conv_even);

  zero_vectorized<bfloat16, DIM_M, DIM_N, 32>(C);
  matmul_vectorized_2x2_mmul<bfloat16, bfloat16, (DIM_M / r), (DIM_K / s),
                             (DIM_N / t), r, s, t>(A, B, C);
}

void ffn_matmul_bf16_bf16_up_proj(const bfloat16 *A, const bfloat16 *B,
                                  bfloat16 *C) {
#ifndef AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
  static_assert(false,
                "AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 must be defined for "
                "this kernel");
#endif
  constexpr int r = 8;
  constexpr int s = 8;
  constexpr int t = 8;

  static_assert(DIM_M % (2 * r) == 0);
  static_assert(DIM_N % (2 * t) == 0);
  static_assert(DIM_K % s == 0);

  ::aie::set_rounding(aie::rounding_mode::conv_even);

  matmul_vectorized_2x2_mmul<bfloat16, bfloat16, (DIM_M / r), (DIM_K / s),
                             (DIM_N / t), r, s, t>(A, B, C);
}

void ffn_matmul_init_bf16_bf16_down_proj(const bfloat16 *A, const bfloat16 *B,
                                         bfloat16 *C) {
#ifndef AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
  static_assert(false,
                "AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 must be defined for "
                "this kernel");
#endif
  constexpr int r = 8;
  constexpr int s = 8;
  constexpr int t = 8;

  static_assert(DIM_M % (2 * r) == 0);
  static_assert(DIM_N % (2 * t) == 0);
  static_assert(DIM_K % s == 0);

  ::aie::set_rounding(aie::rounding_mode::conv_even);

  zero_vectorized<bfloat16, DIM_M, DIM_K, 32>(C);
  matmul_vectorized_2x2_mmul<bfloat16, bfloat16, (DIM_M / r), (DIM_N / t),
                             (DIM_K / s), r, s, t>(A, B, C);
}

void ffn_matmul_with_acc_bf16_bf16_down_proj(const bfloat16 *A,
                                             const bfloat16 *B, bfloat16 *pAcc,
                                             bfloat16 *C) {
#ifndef AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
  static_assert(false,
                "AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 must be defined for "
                "this kernel");
#endif
  constexpr int r = 8;
  constexpr int s = 8;
  constexpr int t = 8;

  static_assert(DIM_M % (2 * r) == 0);
  static_assert(DIM_N % (2 * t) == 0);
  static_assert(DIM_K % s == 0);

  ::aie::set_rounding(aie::rounding_mode::conv_even);

  matmul_with_acc_vectorized_2x2_mmul<bfloat16, bfloat16, (DIM_M / r),
                                      (DIM_N / t), (DIM_K / s), r, s, t>(
      A, B, pAcc, C);
}

void ffn_gelu_bf16(bfloat16 *__restrict input, bfloat16 *__restrict output,
                   int input_size) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  // Skip GeLU calculations in debug mode
#ifndef DEBUG_AIE_KERNELS
  gelu_tanh_approx_bf16(input, output, input_size);
#endif
}

void ffn_eltwise_add_bf16_vector(bfloat16 *a_in, bfloat16 *b_in,
                                 bfloat16 *c_out, int size) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  eltwise_vadd<bfloat16, bfloat16>(a_in, b_in, c_out, size);
}

#endif

#ifdef BUILD_ADDNORM
void ln_passThroughTile_in(const bfloat16 *input, bfloat16 *output,
                           const int32_t cols, const int32_t tileWidth,
                           const int32_t tileHeight, const int32_t col_idx) {
  event0();
  const bfloat16 *pIn = input + col_idx * tileWidth;
  bfloat16 *pOut = output;
  AIE_PREPARE_FOR_PIPELINING
  for (int row = 0; row < tileHeight; ++row) {
    const bfloat16 *pInRow = pIn + row * cols;
    for (int col = 0; col < tileWidth; col += 32) {
      auto reg = ::aie::load_v<32>(pInRow + col);
      ::aie::store_v(pOut + col, reg);
    }
    pOut += tileWidth;
  }
  event1();
}

void fused_add_layer_norm_1outs(const bfloat16 *input, const bfloat16 *residual,
                                const bfloat16 *weights, const float *sum,
                                const float *sumsq, bfloat16 *output,
                                const int32_t cols, const int32_t col_idx) {
  constexpr int r = 8;
  constexpr int s = 8;

  static_assert(DIM_M % r == 0);
  static_assert(DIM_K % s == 0);

  ::aie::set_rounding(aie::rounding_mode::conv_even);
  fused_add_layer_norm_1<bfloat16, (DIM_M / r), (DIM_K / s), r, s>(
      input, residual, weights, sum, sumsq, output, cols, col_idx);
}

void fused_layer_norm_1outs(const bfloat16 *input, const float *sum,
                            const float *sumsq, bfloat16 *output,
                            const int32_t cols) {
  constexpr int r = 8;
  constexpr int s = 8;

  static_assert(DIM_M % r == 0);
  static_assert(DIM_K % s == 0);

  ::aie::set_rounding(aie::rounding_mode::conv_even);
  fused_layer_norm_1<bfloat16, (DIM_M / r), (DIM_K / s), r, s>(
      input, sum, sumsq, output, cols);
}

void ln_mul_weights_1outs(const bfloat16 *input, const bfloat16 *weights,
                          bfloat16 *output, const int32_t col_idx) {
  constexpr int r = 8;
  constexpr int s = 8;

  static_assert(DIM_M % r == 0);
  static_assert(DIM_K % s == 0);

  ::aie::set_rounding(aie::rounding_mode::conv_even);
  ln_mul_weights_1<bfloat16, (DIM_M / r), (DIM_K / s), r, s>(input, weights,
                                                             output, col_idx);
}

void ln_mul_add_1outs(const bfloat16 *input, const bfloat16 *residual,
                      const bfloat16 *weights, bfloat16 *output,
                      const int32_t col_idx) {
  constexpr int r = 8;
  constexpr int s = 8;

  static_assert(DIM_M % r == 0);
  static_assert(DIM_K % s == 0);

  ::aie::set_rounding(aie::rounding_mode::conv_even);
  ln_mul_add_1<bfloat16, (DIM_M / r), (DIM_K / s), r, s>(
      input, residual, weights, output, col_idx);
}

void fused_add_layer_norm_2outs(const bfloat16 *input, const bfloat16 *residual,
                                const bfloat16 *weights, bfloat16 *output1,
                                bfloat16 *output2, const int32_t cols,
                                const int32_t rows_to_process) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  fused_add_layer_norm_2<bfloat16, 32>(input, residual, weights, output1,
                                       output2, cols, rows_to_process);
}

void ln_zero_bf16(bfloat16 *C, int size) {
  ln_zero_vectorized<bfloat16, 16>(C, size);
}

void ln_zero_f32(float *C, int size) { ln_zero_vectorized<float, 8>(C, size); }

void ln_calc_sum_sumsq(const bfloat16 *A, float *pSum, float *pSumSq) {
  constexpr int r = 8;
  constexpr int s = 8;

  static_assert(DIM_M % r == 0);
  static_assert(DIM_K % s == 0);

  ::aie::set_rounding(aie::rounding_mode::conv_even);

  ln_calc_sum_sumsq_vectorized<bfloat16, (DIM_M / r), (DIM_K / s), r, s>(
      A, pSum, pSumSq);
}

#endif
}
