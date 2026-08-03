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

/* Blocked MatMul kernel (vectorized) utilizing the aie::mmul class.
 * The matrices are assumed to be pre-tiled with the following shapes
 * for the aie::mmul class: A => rxs, B => sxt, C => rxt.
 *
 * The matrix dimensions of the kernel are defined by rowA, colA and colB.
 * In this kernel we expand the aie::mmul two times in both the 'm' dimension
 * and the 'n' dimension, producing a 2x2 block in the output matrix.
 *
 * Data within each tile (rxs, sxt and rxt) are assumed to be in row-major
 * order. Also, the entire tiles themselves are stored in row-major order.
 *
 * C is accumulated into: the prior contents of C are loaded and added to.
 */
template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t>
void matmul_vectorized_2x2_mmul(const T_in *__restrict pA,
                                const T_in *__restrict pB,
                                T_out *__restrict pC) {
  using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

  event0();

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (unsigned z = 0; z < rowA; z += 2) {

    T_out *__restrict pC1 = pC + (z * colB) * MMUL::size_C;
    T_out *__restrict pC2 = pC + ((z + 1) * colB) * MMUL::size_C;

    for (unsigned j = 0; j < colB; j += 2)
#ifdef OPT_PERF_ENABLED
      AIE_LOOP_FLATTEN
#endif
      {
        const T_in *__restrict pA1 = pA + (z * colA) * MMUL::size_A;
        const T_in *__restrict pA2 = pA + ((z + 1) * colA) * MMUL::size_A;
        const T_in *__restrict pB1 = pB + (j)*MMUL::size_B;
        const T_in *__restrict pB2 = pB + (j + 1) * MMUL::size_B;
        aie::vector<T_in, MMUL::size_A> A0;
        aie::vector<T_in, MMUL::size_A> A1;
        aie::vector<T_in, MMUL::size_B> B0;
        aie::vector<T_in, MMUL::size_B> B1;

        aie::vector<T_out, MMUL::size_C> acc_C00 =
            aie::load_v<MMUL::size_C>(pC1);
        aie::vector<T_out, MMUL::size_C> acc_C01 =
            aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C);
        aie::vector<T_out, MMUL::size_C> acc_C10 =
            aie::load_v<MMUL::size_C>(pC2);
        aie::vector<T_out, MMUL::size_C> acc_C11 =
            aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C);

        MMUL C00(acc_C00);
        MMUL C01(acc_C01);
        MMUL C10(acc_C10);
        MMUL C11(acc_C11);

        for (unsigned i = 0; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
          AIE_LOOP_FLATTEN
#endif
          {
            A0 = aie::load_v<MMUL::size_A>(pA1);
            pA1 += MMUL::size_A;
            A1 = aie::load_v<MMUL::size_A>(pA2);
            pA2 += MMUL::size_A;
            B0 = aie::load_v<MMUL::size_B>(pB1);
            pB1 += MMUL::size_B * colB;
            B1 = aie::load_v<MMUL::size_B>(pB2);
            pB2 += MMUL::size_B * colB;

            C00.mac(A0, B0);
            C01.mac(A0, B1);
            C10.mac(A1, B0);
            C11.mac(A1, B1);
          }

        aie::store_v(pC1, C00.template to_vector<T_out>());
        pC1 += MMUL::size_C;
        aie::store_v(pC1, C01.template to_vector<T_out>());
        pC1 += MMUL::size_C;
        aie::store_v(pC2, C10.template to_vector<T_out>());
        pC2 += MMUL::size_C;
        aie::store_v(pC2, C11.template to_vector<T_out>());
        pC2 += MMUL::size_C;
      }
  }

  event1();
}

/* Blocked MatMul kernel (vectorized) utilizing the aie::mmul class.
 * The matrices are assumed to be pre-tiled with the following shapes
 * for the aie:mmul class: A => rxs, B => sxt, C => rxt.
 *
 * The matrix dimensions of the kernel are defined by rowA, colA and colB.
 * In this particular kernel we expand the aie::mmul two times in each
 * input matrices A (in 'm' dimension, or rowA) and B (in 'n' dimension, or
 * ColB), leading to a 1x4 expansion in output matrix C (see C00, C01, C02, C03
 * below). This expansion helps with accumulator registers usage, which leads in
 * attaining high kernel efficiency (SIMD utilization).
 *
 * Data within each tile (rxs, sxt and rxt) are assumed to be in row-major
 * order. Also, the entire tiles themselves are stored in row-major order, as
 * shown in the example below for matrix A:
 *
 *      <-s->
 *    _  ________________________
 * 	  r |  1 |  2 |  3 | ...
 * 	  _ |____|____|____|
 * 	    |  x | x+1| x+2| ...
 * 	    |____|____|____|
 * 	    |.
 * 	    |.
 * 	    |.
 *
 * A simplified example of this kernel can be found in the AIE-API
 * documentation: https://xilinx.github.io/aie_api/group__group__mmul.html
 *
 * The running sum is read from pAcc and the result written to pC, so the
 * accumulator and the destination may be distinct buffers.
 */
template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t>
void matmul_with_acc_vectorized_2x2_mmul(const T_in *__restrict pA,
                                         const T_in *__restrict pB,
                                         T_out *__restrict pAcc,
                                         T_out *__restrict pC) {
  using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

  event0();

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (unsigned z = 0; z < rowA; z += 2) {

    T_out *__restrict pAcc1 = pAcc + (z * colB) * MMUL::size_C;
    T_out *__restrict pAcc2 = pAcc + ((z + 1) * colB) * MMUL::size_C;
    T_out *__restrict pC1 = pC + (z * colB) * MMUL::size_C;
    T_out *__restrict pC2 = pC + ((z + 1) * colB) * MMUL::size_C;

    for (unsigned j = 0; j < colB; j += 2)
#ifdef OPT_PERF_ENABLED
      AIE_LOOP_FLATTEN
#endif
      {
        const T_in *__restrict pA1 = pA + (z * colA) * MMUL::size_A;
        const T_in *__restrict pA2 = pA + ((z + 1) * colA) * MMUL::size_A;
        const T_in *__restrict pB1 = pB + (j)*MMUL::size_B;
        const T_in *__restrict pB2 = pB + (j + 1) * MMUL::size_B;
        aie::vector<T_in, MMUL::size_A> A0;
        aie::vector<T_in, MMUL::size_A> A1;
        aie::vector<T_in, MMUL::size_B> B0;
        aie::vector<T_in, MMUL::size_B> B1;

        aie::vector<T_out, MMUL::size_C> acc_C00 =
            aie::load_v<MMUL::size_C>(pAcc1);
        pAcc1 += MMUL::size_C;
        aie::vector<T_out, MMUL::size_C> acc_C01 =
            aie::load_v<MMUL::size_C>(pAcc1);
        pAcc1 += MMUL::size_C;
        aie::vector<T_out, MMUL::size_C> acc_C10 =
            aie::load_v<MMUL::size_C>(pAcc2);
        pAcc2 += MMUL::size_C;
        aie::vector<T_out, MMUL::size_C> acc_C11 =
            aie::load_v<MMUL::size_C>(pAcc2);
        pAcc2 += MMUL::size_C;

        MMUL C00(acc_C00);
        MMUL C01(acc_C01);
        MMUL C10(acc_C10);
        MMUL C11(acc_C11);

        for (unsigned i = 0; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
          AIE_LOOP_FLATTEN
#endif
          {
            A0 = aie::load_v<MMUL::size_A>(pA1);
            pA1 += MMUL::size_A;
            A1 = aie::load_v<MMUL::size_A>(pA2);
            pA2 += MMUL::size_A;
            B0 = aie::load_v<MMUL::size_B>(pB1);
            pB1 += MMUL::size_B * colB;
            B1 = aie::load_v<MMUL::size_B>(pB2);
            pB2 += MMUL::size_B * colB;

            C00.mac(A0, B0);
            C01.mac(A0, B1);
            C10.mac(A1, B0);
            C11.mac(A1, B1);
          }

        aie::store_v(pC1, C00.template to_vector<T_out>());
        pC1 += MMUL::size_C;
        aie::store_v(pC1, C01.template to_vector<T_out>());
        pC1 += MMUL::size_C;
        aie::store_v(pC2, C10.template to_vector<T_out>());
        pC2 += MMUL::size_C;
        aie::store_v(pC2, C11.template to_vector<T_out>());
        pC2 += MMUL::size_C;
      }
  }

  event1();
}

template <typename T, unsigned rowA, unsigned colA, unsigned r, unsigned s>
void fused_add_layer_norm_1(const T *__restrict input,
                            const T *__restrict residual,
                            const T *__restrict weight,
                            const float *__restrict sum,
                            const float *__restrict sumsq, T *__restrict output,
                            const int32_t cols,
                            const int32_t col_idx) // For offset into weight
                                                   // vector
{
  event0();

  constexpr float epsilon = 1e-5f;
  constexpr unsigned mmul_c_size =
      r * s; // This is the number of elements in each C tile (microtile)

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (unsigned z = 0; z < rowA; z += 1) {

    const T *__restrict pA1 = input + (z * colA) * mmul_c_size;
    const T *__restrict pR1 = residual + (z * colA) * mmul_c_size;
    T *__restrict pC1 = output + (z * colA) * mmul_c_size;

    // Each row has an accumulator for sum and sum-squared, so offset by z*r to
    // get to the correct row, and then we'll index into the correct one within
    // that row with the loop below
    const float *__restrict pSum1 = sum + z * r;
    const float *__restrict pSumSq1 = sumsq + z * r;

    // For each row within this microtile, apply layer norm and add residual
    for (unsigned ri = 0; ri < r; ri++) {
      float mean = aie::div(*pSum1, aie::to_float(cols));
      float mean_sq = mean * mean;
      float variance = aie::div(*pSumSq1, aie::to_float(cols)) - mean_sq;
      if (variance < 0.0f) {
        variance = 0.0f;
      }
      float inv_std = aie::invsqrt(variance + epsilon);

      const T *__restrict pW = weight + (col_idx * colA * s);

      for (unsigned j = 0; j < colA; j += 2) {
        aie::vector<T, s> A0 = aie::load_v<s>(pA1);
        pA1 += mmul_c_size; // Move pointer to the start of the next microtile
                            // in the same row
        aie::vector<T, s> A1 = aie::load_v<s>(pA1);
        pA1 += mmul_c_size; // Move pointer to the start of the next microtile
                            // in the same row
        auto A01 = aie::concat(A0, A1);
        aie::vector<T, s> R0 = aie::load_v<s>(pR1);
        pR1 += mmul_c_size; // Move pointer to the start of the next microtile
                            // in the same row
        aie::vector<T, s> R1 = aie::load_v<s>(pR1);
        pR1 += mmul_c_size; // Move pointer to the start of the next microtile
                            // in the same row
        auto R01 = aie::concat(R0, R1);

        aie::accum<accfloat, 2 * s> a_acc;
        a_acc.from_vector(A01);
        aie::accum<accfloat, 2 * s> diff_acc = aie::sub(a_acc, mean);
        aie::accum<accfloat, 2 * s> norm_acc =
            aie::mul(diff_acc.template to_vector<float>(), inv_std);
        aie::vector<T, 2 * s> weight_v = aie::load_v<2 * s>(pW);
        pW += 2 * s; // Move weight pointer to the columns
        aie::vector<T, 2 * s> scaled_acc =
            aie::mul(norm_acc.template to_vector<T>(), weight_v);
        // aie::accum<accfloat, 2 * s> out_v = aie::add(scaled_v, beta_v);
        aie::vector<T, 2 * s> out_acc = aie::add(scaled_acc, R01);

#ifndef DEBUG_AIE_KERNELS
        // Write s elements to one row of four r x s tiles in output
        aie::store_v(pC1, out_acc.template extract<s>(0));
        pC1 += mmul_c_size; // Move pointer to the start of the next microtile
                            // in the same row
        aie::store_v(pC1, out_acc.template extract<s>(1));
#else
#if DEBUG_AIE_KERNELS == 0
        // Input values
        aie::store_v(pC1, A0);
        pC1 += mmul_c_size; // Move pointer to the start of the next microtile
                            // in the same row
        aie::store_v(pC1, A1);
#elif DEBUG_AIE_KERNELS == 1
        // Residual values
        aie::store_v(pC1, R0);
        pC1 += mmul_c_size; // Move pointer to the start of the next microtile
                            // in the same row
        aie::store_v(pC1, R1);
#endif
#endif
        pC1 += mmul_c_size; // Move pointer to the start of the next microtile
                            // in the same row
      }

      pSum1++;
      pSumSq1++;
      pA1 -= colA * mmul_c_size; // Move pointer back to the start of the row
      pA1 += s;                  // Move pointer to the next row within the same
                                 // microtile
      pR1 -= colA * mmul_c_size; // Move pointer back to the start of the row
      pR1 += s;                  // Move pointer to the next row within the same
                                 // microtile
      pC1 -= colA * mmul_c_size; // Move pointer back to the start of the row
      pC1 += s;                  // Move pointer to the next row within the same
                                 // microtile
    }
  }
  event1();
}

template <typename T, unsigned rowA, unsigned colA, unsigned r, unsigned s>
void fused_layer_norm_1(const T *__restrict input, const float *__restrict sum,
                        const float *__restrict sumsq, T *__restrict output,
                        const int32_t cols) {
  event0();

  constexpr float epsilon = 1e-5f;
  constexpr unsigned mmul_c_size =
      r * s; // Number of elements in each C microtile

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (unsigned z = 0; z < rowA; z += 1) {
    const T *__restrict pA1 = input + (z * colA) * mmul_c_size;
    T *__restrict pC1 = output + (z * colA) * mmul_c_size;

    const float *__restrict pSum1 = sum + z * r;
    const float *__restrict pSumSq1 = sumsq + z * r;

    // For each row within this microtile, apply layer norm (no
    // weight/residual).
    for (unsigned ri = 0; ri < r; ri++) {
      float mean = aie::div(*pSum1, aie::to_float(cols));
      float mean_sq = mean * mean;
      float variance = aie::div(*pSumSq1, aie::to_float(cols)) - mean_sq;
      if (variance < 0.0f) {
        variance = 0.0f;
      }
      float inv_std = aie::invsqrt(variance + epsilon);

      for (unsigned j = 0; j < colA; j += 2) {
        aie::vector<T, s> A0 = aie::load_v<s>(pA1);
        pA1 += mmul_c_size;
        aie::vector<T, s> A1 = aie::load_v<s>(pA1);
        pA1 += mmul_c_size;
        auto A01 = aie::concat(A0, A1);

        aie::accum<accfloat, 2 * s> a_acc;
        a_acc.from_vector(A01);
        aie::accum<accfloat, 2 * s> diff_acc = aie::sub(a_acc, mean);
        aie::accum<accfloat, 2 * s> norm_acc =
            aie::mul(diff_acc.template to_vector<float>(), inv_std);
        aie::vector<T, 2 * s> out_acc = norm_acc.template to_vector<T>();

        aie::store_v(pC1, out_acc.template extract<s>(0));
        pC1 += mmul_c_size;
        aie::store_v(pC1, out_acc.template extract<s>(1));
        pC1 += mmul_c_size;
      }

      pSum1++;
      pSumSq1++;
      pA1 -= colA * mmul_c_size;
      pA1 += s;
      pC1 -= colA * mmul_c_size;
      pC1 += s;
    }
  }
  event1();
}

template <typename T, unsigned rowA, unsigned colA, unsigned r, unsigned s>
void ln_mul_weights_1(const T *__restrict input, const T *__restrict weight,
                      T *__restrict output, const int32_t col_idx) {
  event0();

  constexpr unsigned mmul_c_size =
      r * s; // Number of elements in each C microtile

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (unsigned z = 0; z < rowA; z += 1) {
    const T *__restrict pA1 = input + (z * colA) * mmul_c_size;
    T *__restrict pC1 = output + (z * colA) * mmul_c_size;
    const T *__restrict pW_row = weight + (col_idx * colA * s);

    for (unsigned ri = 0; ri < r; ri++) {
      const T *__restrict pW = pW_row;
      for (unsigned j = 0; j < colA; j += 2) {
        aie::vector<T, s> A0 = aie::load_v<s>(pA1);
        pA1 += mmul_c_size;
        aie::vector<T, s> A1 = aie::load_v<s>(pA1);
        pA1 += mmul_c_size;
        auto A01 = aie::concat(A0, A1);

        aie::vector<T, 2 * s> weight_v = aie::load_v<2 * s>(pW);
        pW += 2 * s;
        aie::vector<T, 2 * s> out_acc = aie::mul(A01, weight_v);

        aie::store_v(pC1, out_acc.template extract<s>(0));
        pC1 += mmul_c_size;
        aie::store_v(pC1, out_acc.template extract<s>(1));
        pC1 += mmul_c_size;
      }

      pA1 -= colA * mmul_c_size;
      pA1 += s;
      pC1 -= colA * mmul_c_size;
      pC1 += s;
    }
  }
  event1();
}

template <typename T, unsigned rowA, unsigned colA, unsigned r, unsigned s>
void ln_mul_add_1(const T *__restrict input, const T *__restrict residual,
                  const T *__restrict weight, T *__restrict output,
                  const int32_t col_idx) {
  event0();

  constexpr unsigned mmul_c_size =
      r * s; // Number of elements in each C microtile

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (unsigned z = 0; z < rowA; z += 1) {
    const T *__restrict pA1 = input + (z * colA) * mmul_c_size;
    const T *__restrict pR1 = residual + (z * colA) * mmul_c_size;
    T *__restrict pC1 = output + (z * colA) * mmul_c_size;
    const T *__restrict pW_row = weight + (col_idx * colA * s);

    for (unsigned ri = 0; ri < r; ri++) {
      const T *__restrict pW = pW_row;
      for (unsigned j = 0; j < colA; j += 2) {
        aie::vector<T, s> A0 = aie::load_v<s>(pA1);
        pA1 += mmul_c_size;
        aie::vector<T, s> A1 = aie::load_v<s>(pA1);
        pA1 += mmul_c_size;
        auto A01 = aie::concat(A0, A1);

        aie::vector<T, s> R0 = aie::load_v<s>(pR1);
        pR1 += mmul_c_size;
        aie::vector<T, s> R1 = aie::load_v<s>(pR1);
        pR1 += mmul_c_size;
        auto R01 = aie::concat(R0, R1);

        aie::vector<T, 2 * s> weight_v = aie::load_v<2 * s>(pW);
        pW += 2 * s;
        aie::vector<T, 2 * s> scaled = aie::mul(A01, weight_v);
        aie::vector<T, 2 * s> out_acc = aie::add(scaled, R01);

        aie::store_v(pC1, out_acc.template extract<s>(0));
        pC1 += mmul_c_size;
        aie::store_v(pC1, out_acc.template extract<s>(1));
        pC1 += mmul_c_size;
      }

      pA1 -= colA * mmul_c_size;
      pA1 += s;
      pR1 -= colA * mmul_c_size;
      pR1 += s;
      pC1 -= colA * mmul_c_size;
      pC1 += s;
    }
  }
  event1();
}

template <typename T, int N>
void fused_add_layer_norm_2(const T *__restrict input,
                            const T *__restrict residual,
                            const T *__restrict weight, T *__restrict output1,
                            T *__restrict output2, const int32_t cols,
                            const int32_t rows_to_process) {
  event0();
#ifndef DEBUG_AIE_KERNELS
  constexpr float epsilon = 1e-5f;
  int vector_chunks = cols / N;

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(4)
  for (int row = 0; row < rows_to_process; row++) {

    ::aie::vector<T, N> sum_acc = ::aie::zeros<T, N>();
    ::aie::vector<float, N> sum_sq_acc = ::aie::zeros<float, N>();
    int input_idx = row * cols;

    for (int i = 0; i < vector_chunks; i++) {

      ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input + input_idx);
      sum_acc = ::aie::add(sum_acc, reg_a);
      ::aie::vector<float, N> sq_acc = ::aie::mul(reg_a, reg_a);
      sum_sq_acc = ::aie::add(sum_sq_acc, sq_acc);
      input_idx += N;
    }

    input_idx -= cols; // reset pointer to beginning of the row

    float sum_of_vals = ::aie::reduce_add(sum_acc);
    float sum_of_sq_vals = ::aie::reduce_add(sum_sq_acc);

    float mean = sum_of_vals / float(cols);
    float mean_sq = mean * mean;
    float variance = (sum_of_sq_vals / float(cols)) - mean_sq;
    if (variance < 0.0f) {
      variance = 0.0f;
    }
    float inv_std = aie::invsqrt(variance + epsilon);

    ::aie::vector<T, N> mean_v = ::aie::broadcast<T, N>(mean);
    ::aie::vector<T, N> inv_std_v = ::aie::broadcast<T, N>(inv_std);

    const T *__restrict pW = weight;
    const T *__restrict pRes = residual + row * cols;
    T *__restrict pOut1 = output1 + row * cols;
    T *__restrict pOut2 = output2 + row * cols;
    for (int i = 0; i < vector_chunks; i++) {

      ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input + input_idx);
      ::aie::vector<T, N> reg_weight = ::aie::load_v<N>(pW);
      ::aie::vector<T, N> reg_res = ::aie::load_v<N>(pRes);
      ::aie::vector<T, N> diff_v = ::aie::sub(reg_a, mean_v);
      ::aie::vector<T, N> norm_v = ::aie::mul(diff_v, inv_std_v);
      ::aie::vector<T, N> scaled_v = aie::mul(norm_v, reg_weight);
      // ::aie::vector<T, N> out_v = ::aie::add(scaled_v, beta_v);
      ::aie::vector<T, N> out_v = ::aie::add(scaled_v, reg_res);
      ::aie::store_v(pOut1, out_v);
      ::aie::store_v(pOut2, out_v);
      input_idx += N;
      pW += N;
      pRes += N;
      pOut1 += N;
      pOut2 += N;
    }
  }
#else
#if DEBUG_AIE_KERNELS == 0

  // In debug mode, just copy input to output
  int total_elements = rows_to_process * cols;
  const T *__restrict pIn = input;
  T *__restrict pOut1 = output1;
  T *__restrict pOut2 = output2;
  AIE_PREPARE_FOR_PIPELINING
  // AIE_LOOP_MIN_ITERATION_COUNT(4)
  for (int i = 0; i < total_elements; i += N) {
    ::aie::vector<T, N> reg_a = ::aie::load_v<N>(pIn);
    ::aie::store_v(pOut1, reg_a);
    ::aie::store_v(pOut2, reg_a);
    pIn += N;
    pOut1 += N;
    pOut2 += N;
  }
#elif DEBUG_AIE_KERNELS == 1
  // In debug mode, just copy residual to output
  int total_elements = rows_to_process * cols;
  const T *__restrict pRes = residual;
  T *__restrict pOut1 = output1;
  T *__restrict pOut2 = output2;
  AIE_PREPARE_FOR_PIPELINING
  // AIE_LOOP_MIN_ITERATION_COUNT(4)
  for (int i = 0; i < total_elements; i += N) {
    ::aie::vector<T, N> reg_a = ::aie::load_v<N>(pRes);
    ::aie::store_v(pOut1, reg_a);
    ::aie::store_v(pOut2, reg_a);
    pRes += N;
    pOut1 += N;
    pOut2 += N;
  }
#endif
#endif
  event1();
}

template <typename T, int N>
void ln_zero_vectorized(T *__restrict c, int size) {
  event0();

  T *__restrict pOut1 = c;
  auto zero_v = ::aie::zeros<T, N>();

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (int j = 0; j < size; j += N) {
    ::aie::store_v(pOut1, zero_v);
    pOut1 += N;
  }

  event1();
}

template <typename T, unsigned rowA, unsigned colA, unsigned r, unsigned s>
void ln_calc_sum_sumsq_vectorized(const T *__restrict pA,
                                  float *__restrict pSum,
                                  float *__restrict pSumSq) {
  event0();

  constexpr unsigned mmul_c_size =
      r * s; // This is the number of elements in each C tile (microtile)

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (unsigned z = 0; z < rowA; z += 1) {
    const T *__restrict pA1 = pA + (z * colA) * mmul_c_size;
    // Each row has an accumulator for sum and sum-squared, so offset by z*r to
    // get to the correct row, and then we'll index into the correct one within
    // that row with the loop below
    float *__restrict pSum1 = pSum + z * r;
    float *__restrict pSumSq1 = pSumSq + z * r;

    // Per-row accumulators for sum and sum-squared across all column tiles
    for (unsigned i = 0; i < r; i++) {
      float row_sum = *pSum1;
      float row_sumsq = *pSumSq1;
      for (unsigned j = 0; j < colA; j += 2) {
        aie::vector<T, s> A0 = aie::load_v<s>(pA1);
        pA1 += mmul_c_size; // Move pointer to the start of the next microtile
                            // in the same row
        aie::vector<T, s> A1 = aie::load_v<s>(pA1);
        pA1 += mmul_c_size; // Move pointer to the start of the next microtile
                            // in the same row

        auto A01 = aie::concat(A0, A1);
        aie::accum<accfloat, 2 * s> sum_acc;
        sum_acc.from_vector(A01);
        float sum = aie::reduce_add(sum_acc.template to_vector<float>());
        row_sum += sum;

        aie::vector<float, s> a_acc_sq0 = aie::mul(A0, A0);
        float sumsq = aie::reduce_add(a_acc_sq0);
        row_sumsq += sumsq;
        aie::vector<float, s> a_acc_sq1 = aie::mul(A1, A1);
        sumsq = aie::reduce_add(a_acc_sq1);
        row_sumsq += sumsq;
      }
      *pSum1 = row_sum;
      *pSumSq1 = row_sumsq;
      pSum1++;
      pSumSq1++;
      pA1 -= colA * mmul_c_size; // Move pointer back to the start of the row
      pA1 += s;                  // Move pointer to the next row within the same
                                 // microtile
    }
  }

  event1();
}

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
