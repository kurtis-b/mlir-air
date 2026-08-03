//===- mm_aie2p.cc -----------------------------------------------*- C++
//-*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
// This file is an updated version of mm.cc for AIE2P architecture
// using 8x8x8 matmul shape with BFP16 emulation
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define REL_WRITE 0
#define REL_READ 1

#include <aie_api/aie.hpp>

#include "zero.cc"

// Set rounding mode to convergent even for bf16 matmul accuracy.
// Without this, BFP16 emulation uses floor rounding which introduces
// a systematic negative bias (~-0.07 per K element).
constexpr aie::rounding_mode round_mode = aie::rounding_mode::conv_even;

template <typename T_in, typename T_out, int rowA, int colA, int colB>
static inline void matmul_scalar(T_in *a, T_in *b, T_out *c) {
  event0();
  for (int row = 0; row < rowA; row++) {
    for (int col = 0; col < colB; col++) {
      T_out running_sum = 0;
      for (int i = 0; i < colA; i++) {
        running_sum += a[row * colA + i] * b[i * colB + col];
      }
      c[row * colB + col] += running_sum;
    }
  }
  event1();
}

template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t>
static inline void matmul_vectorized_2x2_mmul(const T_in *__restrict pA,
                                              const T_in *__restrict pB,
                                              T_out *__restrict pC) {

  using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

  event0();

  for (unsigned z = 0; z < rowA; z += 2)
    chess_prepare_for_pipelining chess_loop_range(2, ) {
      T_out *__restrict pC1 = pC + (z)*MMUL::size_C;
      T_out *__restrict pC2 = pC + ((z + 1)) * MMUL::size_C;

      for (unsigned j = 0; j < colB; j += 2)
#ifdef OPT_PERF_ENABLED
        chess_flatten_loop
#endif
        {
          const T_in *__restrict pA1 = pA + (z)*MMUL::size_A;
          const T_in *__restrict pA2 = pA + ((z + 1)) * MMUL::size_A;
          const T_in *__restrict pB1 = pB + (j)*colA * MMUL::size_B;
          const T_in *__restrict pB2 = pB + (j + 1) * colA * MMUL::size_B;

          aie::vector<T_out, MMUL::size_C> acc_C00 =
              aie::load_v<MMUL::size_C>(pC1);
          aie::vector<T_out, MMUL::size_C> acc_C01 =
              aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C * rowA);
          aie::vector<T_out, MMUL::size_C> acc_C10 =
              aie::load_v<MMUL::size_C>(pC2);
          aie::vector<T_out, MMUL::size_C> acc_C11 =
              aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C * rowA);

          MMUL C00(acc_C00);
          MMUL C01(acc_C01);
          MMUL C10(acc_C10);
          MMUL C11(acc_C11);

          for (unsigned i = 0; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
            chess_flatten_loop
#endif
            {
              aie::vector<T_in, MMUL::size_A> A0 =
                  aie::load_v<MMUL::size_A>(pA1);
              pA1 += rowA * MMUL::size_A;
              aie::vector<T_in, MMUL::size_A> A1 =
                  aie::load_v<MMUL::size_A>(pA2);
              pA2 += rowA * MMUL::size_A;
              aie::vector<T_in, MMUL::size_B> B0 =
                  aie::load_v<MMUL::size_B>(pB1);
              pB1 += MMUL::size_B;
              aie::vector<T_in, MMUL::size_B> B1 =
                  aie::load_v<MMUL::size_B>(pB2);
              pB2 += MMUL::size_B;

              C00.mac(A0, B0);
              C01.mac(A0, B1);
              C10.mac(A1, B0);
              C11.mac(A1, B1);
            }

          aie::store_v(pC1, C00.template to_vector<T_out>());
          pC1 += MMUL::size_C * rowA;
          aie::store_v(pC1, C01.template to_vector<T_out>());
          pC1 += MMUL::size_C * rowA;
          aie::store_v(pC2, C10.template to_vector<T_out>());
          pC2 += MMUL::size_C * rowA;
          aie::store_v(pC2, C11.template to_vector<T_out>());
          pC2 += MMUL::size_C * rowA;
        }
    }

  event1();
}

// ---------------------------------------------------------------------------
// "init" family: zero-then-multiply.
//
// Identical tile addressing to matmul_vectorized_*_mmul above, but the mmul
// accumulators start at zero instead of being seeded from pC. This fuses the
// separate zero_* call that otherwise has to precede the first K-tile into the
// matmul itself, which matters for staged pipelines that re-enter the same L1
// C tile once per output block.
//
// FOOTGUN: because C is overwritten rather than accumulated into, calling this
// for any K-tile other than the first silently discards the partial sums. The
// K-loop must call the init variant exactly once, at i_k == 0, and the plain
// accumulating variant for every subsequent K-tile.
// ---------------------------------------------------------------------------

template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t>
static inline void matmul_init_vectorized_2x2_mmul(const T_in *__restrict pA,
                                                   const T_in *__restrict pB,
                                                   T_out *__restrict pC) {

  using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

  event0();

  const aie::vector<T_out, MMUL::size_C> zero_c =
      aie::zeros<T_out, MMUL::size_C>();

  for (unsigned z = 0; z < rowA; z += 2)
    chess_prepare_for_pipelining chess_loop_range(2, ) {
      T_out *__restrict pC1 = pC + (z)*MMUL::size_C;
      T_out *__restrict pC2 = pC + ((z + 1)) * MMUL::size_C;

      for (unsigned j = 0; j < colB; j += 2)
#ifdef OPT_PERF_ENABLED
        chess_flatten_loop
#endif
        {
          const T_in *__restrict pA1 = pA + (z)*MMUL::size_A;
          const T_in *__restrict pA2 = pA + ((z + 1)) * MMUL::size_A;
          const T_in *__restrict pB1 = pB + (j)*colA * MMUL::size_B;
          const T_in *__restrict pB2 = pB + (j + 1) * colA * MMUL::size_B;

          MMUL C00(zero_c);
          MMUL C01(zero_c);
          MMUL C10(zero_c);
          MMUL C11(zero_c);

          for (unsigned i = 0; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
            chess_flatten_loop
#endif
            {
              aie::vector<T_in, MMUL::size_A> A0 =
                  aie::load_v<MMUL::size_A>(pA1);
              pA1 += rowA * MMUL::size_A;
              aie::vector<T_in, MMUL::size_A> A1 =
                  aie::load_v<MMUL::size_A>(pA2);
              pA2 += rowA * MMUL::size_A;
              aie::vector<T_in, MMUL::size_B> B0 =
                  aie::load_v<MMUL::size_B>(pB1);
              pB1 += MMUL::size_B;
              aie::vector<T_in, MMUL::size_B> B1 =
                  aie::load_v<MMUL::size_B>(pB2);
              pB2 += MMUL::size_B;

              C00.mac(A0, B0);
              C01.mac(A0, B1);
              C10.mac(A1, B0);
              C11.mac(A1, B1);
            }

          aie::store_v(pC1, C00.template to_vector<T_out>());
          pC1 += MMUL::size_C * rowA;
          aie::store_v(pC1, C01.template to_vector<T_out>());
          pC1 += MMUL::size_C * rowA;
          aie::store_v(pC2, C10.template to_vector<T_out>());
          pC2 += MMUL::size_C * rowA;
          aie::store_v(pC2, C11.template to_vector<T_out>());
          pC2 += MMUL::size_C * rowA;
        }
    }

  event1();
}

template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t>
static inline void matmul_init_vectorized_2x1_mmul(const T_in *__restrict pA,
                                                   const T_in *__restrict pB,
                                                   T_out *__restrict pC) {

  using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

  event0();

  const aie::vector<T_out, MMUL::size_C> zero_c =
      aie::zeros<T_out, MMUL::size_C>();

  for (unsigned z = 0; z < rowA; z += 2)
    chess_prepare_for_pipelining chess_loop_range(2, ) {
      T_out *__restrict pC1 = pC + (z)*MMUL::size_C;
      T_out *__restrict pC2 = pC + ((z + 1)) * MMUL::size_C;

      for (unsigned j = 0; j < colB; ++j)
#ifdef OPT_PERF_ENABLED
        chess_flatten_loop
#endif
        {
          const T_in *__restrict pA1 = pA + (z)*MMUL::size_A;
          const T_in *__restrict pA2 = pA + ((z + 1)) * MMUL::size_A;
          const T_in *__restrict pB1 = pB + (j)*colA * MMUL::size_B;

          MMUL C00(zero_c);
          MMUL C10(zero_c);

          for (unsigned i = 0; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
            chess_flatten_loop
#endif
            {
              aie::vector<T_in, MMUL::size_A> A0 =
                  aie::load_v<MMUL::size_A>(pA1);
              pA1 += rowA * MMUL::size_A;
              aie::vector<T_in, MMUL::size_A> A1 =
                  aie::load_v<MMUL::size_A>(pA2);
              pA2 += rowA * MMUL::size_A;
              aie::vector<T_in, MMUL::size_B> B0 =
                  aie::load_v<MMUL::size_B>(pB1);
              pB1 += MMUL::size_B;

              C00.mac(A0, B0);
              C10.mac(A1, B0);
            }

          aie::store_v(pC1, C00.template to_vector<T_out>());
          pC1 += MMUL::size_C * rowA;
          aie::store_v(pC2, C10.template to_vector<T_out>());
          pC2 += MMUL::size_C * rowA;
        }
    }

  event1();
}

template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t>
static inline void matmul_init_vectorized_1x1_mmul(const T_in *__restrict pA,
                                                   const T_in *__restrict pB,
                                                   T_out *__restrict pC) {

  using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

  event0();

  const aie::vector<T_out, MMUL::size_C> zero_c =
      aie::zeros<T_out, MMUL::size_C>();

  for (unsigned z = 0; z < rowA; ++z)
    chess_prepare_for_pipelining chess_loop_range(2, ) {
      T_out *__restrict pC1 = pC + (z)*MMUL::size_C;

      for (unsigned j = 0; j < colB; ++j)
#ifdef OPT_PERF_ENABLED
        chess_flatten_loop
#endif
        {
          const T_in *__restrict pA1 = pA + (z)*MMUL::size_A;
          const T_in *__restrict pB1 = pB + (j)*colA * MMUL::size_B;

          MMUL C00(zero_c);

          for (unsigned i = 0; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
            chess_flatten_loop
#endif
            {
              aie::vector<T_in, MMUL::size_A> A0 =
                  aie::load_v<MMUL::size_A>(pA1);
              pA1 += rowA * MMUL::size_A;
              aie::vector<T_in, MMUL::size_B> B0 =
                  aie::load_v<MMUL::size_B>(pB1);
              pB1 += MMUL::size_B;

              C00.mac(A0, B0);
            }

          aie::store_v(pC1, C00.template to_vector<T_out>());
          pC1 += MMUL::size_C * rowA;
        }
    }

  event1();
}

// ---------------------------------------------------------------------------
// "with_acc" family: accumulate into an explicit pAcc, write to a distinct pC.
//
// The plain matmul_vectorized_* variants accumulate in place (pC is both the
// partial-sum source and the destination). These take the running partials from
// a separate pAcc buffer, which lets a staged pipeline keep the accumulator
// resident in one L1 buffer while streaming each block's result out through a
// different one.
//
// FOOTGUN: pAcc and pC must have the SAME tile layout (tiles stored
// column-major, see matmul_vectorized_2x2_mmul) and the SAME element type.
// Passing pAcc == pC is legal and degenerates to the in-place variant, but the
// compiler is told both are __restrict, so any *partial* overlap is UB.
// ---------------------------------------------------------------------------

template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t>
static inline void matmul_with_acc_vectorized_2x2_mmul(
    const T_in *__restrict pA, const T_in *__restrict pB,
    const T_out *__restrict pAcc, T_out *__restrict pC) {

  using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

  event0();

  for (unsigned z = 0; z < rowA; z += 2)
    chess_prepare_for_pipelining chess_loop_range(2, ) {
      const T_out *__restrict pAcc1 = pAcc + (z)*MMUL::size_C;
      const T_out *__restrict pAcc2 = pAcc + ((z + 1)) * MMUL::size_C;
      T_out *__restrict pC1 = pC + (z)*MMUL::size_C;
      T_out *__restrict pC2 = pC + ((z + 1)) * MMUL::size_C;

      for (unsigned j = 0; j < colB; j += 2)
#ifdef OPT_PERF_ENABLED
        chess_flatten_loop
#endif
        {
          const T_in *__restrict pA1 = pA + (z)*MMUL::size_A;
          const T_in *__restrict pA2 = pA + ((z + 1)) * MMUL::size_A;
          const T_in *__restrict pB1 = pB + (j)*colA * MMUL::size_B;
          const T_in *__restrict pB2 = pB + (j + 1) * colA * MMUL::size_B;

          aie::vector<T_out, MMUL::size_C> acc_C00 =
              aie::load_v<MMUL::size_C>(pAcc1);
          aie::vector<T_out, MMUL::size_C> acc_C01 =
              aie::load_v<MMUL::size_C>(pAcc1 + MMUL::size_C * rowA);
          aie::vector<T_out, MMUL::size_C> acc_C10 =
              aie::load_v<MMUL::size_C>(pAcc2);
          aie::vector<T_out, MMUL::size_C> acc_C11 =
              aie::load_v<MMUL::size_C>(pAcc2 + MMUL::size_C * rowA);
          pAcc1 += 2 * MMUL::size_C * rowA;
          pAcc2 += 2 * MMUL::size_C * rowA;

          MMUL C00(acc_C00);
          MMUL C01(acc_C01);
          MMUL C10(acc_C10);
          MMUL C11(acc_C11);

          for (unsigned i = 0; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
            chess_flatten_loop
#endif
            {
              aie::vector<T_in, MMUL::size_A> A0 =
                  aie::load_v<MMUL::size_A>(pA1);
              pA1 += rowA * MMUL::size_A;
              aie::vector<T_in, MMUL::size_A> A1 =
                  aie::load_v<MMUL::size_A>(pA2);
              pA2 += rowA * MMUL::size_A;
              aie::vector<T_in, MMUL::size_B> B0 =
                  aie::load_v<MMUL::size_B>(pB1);
              pB1 += MMUL::size_B;
              aie::vector<T_in, MMUL::size_B> B1 =
                  aie::load_v<MMUL::size_B>(pB2);
              pB2 += MMUL::size_B;

              C00.mac(A0, B0);
              C01.mac(A0, B1);
              C10.mac(A1, B0);
              C11.mac(A1, B1);
            }

          aie::store_v(pC1, C00.template to_vector<T_out>());
          pC1 += MMUL::size_C * rowA;
          aie::store_v(pC1, C01.template to_vector<T_out>());
          pC1 += MMUL::size_C * rowA;
          aie::store_v(pC2, C10.template to_vector<T_out>());
          pC2 += MMUL::size_C * rowA;
          aie::store_v(pC2, C11.template to_vector<T_out>());
          pC2 += MMUL::size_C * rowA;
        }
    }

  event1();
}

template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t>
static inline void matmul_with_acc_vectorized_2x1_mmul(
    const T_in *__restrict pA, const T_in *__restrict pB,
    const T_out *__restrict pAcc, T_out *__restrict pC) {

  using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

  event0();

  for (unsigned z = 0; z < rowA; z += 2)
    chess_prepare_for_pipelining chess_loop_range(2, ) {
      const T_out *__restrict pAcc1 = pAcc + (z)*MMUL::size_C;
      const T_out *__restrict pAcc2 = pAcc + ((z + 1)) * MMUL::size_C;
      T_out *__restrict pC1 = pC + (z)*MMUL::size_C;
      T_out *__restrict pC2 = pC + ((z + 1)) * MMUL::size_C;

      for (unsigned j = 0; j < colB; ++j)
#ifdef OPT_PERF_ENABLED
        chess_flatten_loop
#endif
        {
          const T_in *__restrict pA1 = pA + (z)*MMUL::size_A;
          const T_in *__restrict pA2 = pA + ((z + 1)) * MMUL::size_A;
          const T_in *__restrict pB1 = pB + (j)*colA * MMUL::size_B;

          aie::vector<T_out, MMUL::size_C> acc_C00 =
              aie::load_v<MMUL::size_C>(pAcc1);
          aie::vector<T_out, MMUL::size_C> acc_C10 =
              aie::load_v<MMUL::size_C>(pAcc2);
          pAcc1 += MMUL::size_C * rowA;
          pAcc2 += MMUL::size_C * rowA;

          MMUL C00(acc_C00);
          MMUL C10(acc_C10);

          for (unsigned i = 0; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
            chess_flatten_loop
#endif
            {
              aie::vector<T_in, MMUL::size_A> A0 =
                  aie::load_v<MMUL::size_A>(pA1);
              pA1 += rowA * MMUL::size_A;
              aie::vector<T_in, MMUL::size_A> A1 =
                  aie::load_v<MMUL::size_A>(pA2);
              pA2 += rowA * MMUL::size_A;
              aie::vector<T_in, MMUL::size_B> B0 =
                  aie::load_v<MMUL::size_B>(pB1);
              pB1 += MMUL::size_B;

              C00.mac(A0, B0);
              C10.mac(A1, B0);
            }

          aie::store_v(pC1, C00.template to_vector<T_out>());
          pC1 += MMUL::size_C * rowA;
          aie::store_v(pC2, C10.template to_vector<T_out>());
          pC2 += MMUL::size_C * rowA;
        }
    }

  event1();
}

template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t>
static inline void matmul_with_acc_vectorized_1x1_mmul(
    const T_in *__restrict pA, const T_in *__restrict pB,
    const T_out *__restrict pAcc, T_out *__restrict pC) {

  using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

  event0();

  for (unsigned z = 0; z < rowA; ++z)
    chess_prepare_for_pipelining chess_loop_range(2, ) {
      const T_out *__restrict pAcc1 = pAcc + (z)*MMUL::size_C;
      T_out *__restrict pC1 = pC + (z)*MMUL::size_C;

      for (unsigned j = 0; j < colB; ++j)
#ifdef OPT_PERF_ENABLED
        chess_flatten_loop
#endif
        {
          const T_in *__restrict pA1 = pA + (z)*MMUL::size_A;
          const T_in *__restrict pB1 = pB + (j)*colA * MMUL::size_B;

          aie::vector<T_out, MMUL::size_C> acc_C00 =
              aie::load_v<MMUL::size_C>(pAcc1);
          pAcc1 += MMUL::size_C * rowA;

          MMUL C00(acc_C00);

          for (unsigned i = 0; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
            chess_flatten_loop
#endif
            {
              aie::vector<T_in, MMUL::size_A> A0 =
                  aie::load_v<MMUL::size_A>(pA1);
              pA1 += rowA * MMUL::size_A;
              aie::vector<T_in, MMUL::size_B> B0 =
                  aie::load_v<MMUL::size_B>(pB1);
              pB1 += MMUL::size_B;

              C00.mac(A0, B0);
            }

          aie::store_v(pC1, C00.template to_vector<T_out>());
          pC1 += MMUL::size_C * rowA;
        }
    }

  event1();
}

// bf16 MatMul kernel with bf16 outputs using 8x8x8 shape.
// Not used by default (combos selects bf16_f32 variant instead).
// Available for configurations that require bf16 output.
template <unsigned m, unsigned k, unsigned n>
static inline void
matmul_vectorized_8x8x8_bf16_bf16(const bfloat16 *__restrict pA,
                                  const bfloat16 *__restrict pB,
                                  bfloat16 *__restrict pC) {

  constexpr int r = 8;
  constexpr int s = 8;
  constexpr int t = 8;

  static_assert(m % (2 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (2 * t) == 0);

  ::aie::set_rounding(round_mode);
  return matmul_vectorized_2x2_mmul<bfloat16, bfloat16, (m / r), (k / s),
                                    (n / t), r, s, t>(pA, pB, pC);
}

// bf16 MatMul kernel with f32 accumulation output using 8x8x8 shape.
// Keeps the accumulator as f32 between K-tile iterations, avoiding bf16
// truncation of partial sums. This gives higher precision at the cost of
// 2x L1 memory for the C buffer.
template <unsigned m, unsigned k, unsigned n>
static inline void
matmul_vectorized_8x8x8_bf16_f32(const bfloat16 *__restrict pA,
                                 const bfloat16 *__restrict pB,
                                 float *__restrict pC) {

  constexpr int r = 8;
  constexpr int s = 8;
  constexpr int t = 8;

  static_assert(m % (2 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (2 * t) == 0);

  ::aie::set_rounding(round_mode);
  return matmul_vectorized_2x2_mmul<bfloat16, float, (m / r), (k / s), (n / t),
                                    r, s, t>(pA, pB, pC);
}

// Shape dispatchers for the init / with_acc families. Unlike the plain
// matmul_vectorized_8x8x8_* wrappers above, which hard-require the 2x2
// expansion, these fall back to 2x1 and then 1x1 so tile shapes with an odd
// number of r-rows or t-columns still compile. The selection is `if constexpr`,
// so exactly one body survives and there is no runtime dispatch cost.

#define MATMUL_SHAPE_DISPATCH_8x8x8(family, ctype_in, ctype_out, ...)          \
  do {                                                                         \
    constexpr int r = 8;                                                       \
    constexpr int s = 8;                                                       \
    constexpr int t = 8;                                                       \
    static_assert(m % r == 0);                                                 \
    static_assert(k % s == 0);                                                 \
    static_assert(n % t == 0);                                                 \
    ::aie::set_rounding(round_mode);                                           \
    if constexpr ((m % (2 * r) == 0) && (n % (2 * t) == 0)) {                  \
      return family##_2x2_mmul<ctype_in, ctype_out, (m / r), (k / s), (n / t), \
                               r, s, t>(__VA_ARGS__);                          \
    } else if constexpr (m % (2 * r) == 0) {                                   \
      return family##_2x1_mmul<ctype_in, ctype_out, (m / r), (k / s), (n / t), \
                               r, s, t>(__VA_ARGS__);                          \
    } else {                                                                   \
      return family##_1x1_mmul<ctype_in, ctype_out, (m / r), (k / s), (n / t), \
                               r, s, t>(__VA_ARGS__);                          \
    }                                                                          \
  } while (0)

template <unsigned m, unsigned k, unsigned n>
static inline void
matmul_init_vectorized_8x8x8_bf16_bf16(const bfloat16 *__restrict pA,
                                       const bfloat16 *__restrict pB,
                                       bfloat16 *__restrict pC) {
  MATMUL_SHAPE_DISPATCH_8x8x8(matmul_init_vectorized, bfloat16, bfloat16, pA,
                              pB, pC);
}

template <unsigned m, unsigned k, unsigned n>
static inline void
matmul_init_vectorized_8x8x8_bf16_f32(const bfloat16 *__restrict pA,
                                      const bfloat16 *__restrict pB,
                                      float *__restrict pC) {
  MATMUL_SHAPE_DISPATCH_8x8x8(matmul_init_vectorized, bfloat16, float, pA, pB,
                              pC);
}

template <unsigned m, unsigned k, unsigned n>
static inline void matmul_with_acc_vectorized_8x8x8_bf16_bf16(
    const bfloat16 *__restrict pA, const bfloat16 *__restrict pB,
    const bfloat16 *__restrict pAcc, bfloat16 *__restrict pC) {
  MATMUL_SHAPE_DISPATCH_8x8x8(matmul_with_acc_vectorized, bfloat16, bfloat16,
                              pA, pB, pAcc, pC);
}

template <unsigned m, unsigned k, unsigned n>
static inline void matmul_with_acc_vectorized_8x8x8_bf16_f32(
    const bfloat16 *__restrict pA, const bfloat16 *__restrict pB,
    const float *__restrict pAcc, float *__restrict pC) {
  MATMUL_SHAPE_DISPATCH_8x8x8(matmul_with_acc_vectorized, bfloat16, float, pA,
                              pB, pAcc, pC);
}

extern "C" {

// If you want to compile microkernels with different inner tile sizes,
// define DIM_M, DIM_K and DIM_N at compile time using -DDIM_M 32 etc.
// These dimensions must be divisible by the r, s, t dimensions used in
// the kernels.
//
// For 4x8x4 shape with 2x2 expansion:
// - DIM_M must be divisible by 2*4 = 8
// - DIM_K must be divisible by 8
// - DIM_N must be divisible by 2*4 = 8

#ifndef DIM_M
#define DIM_M 128
#define DIM_M_DIV_4 32
#define DIM_M_DIV_8 16
#endif

#ifndef DIM_K
#define DIM_K 32
#endif

#ifndef DIM_N
#define DIM_N 64
#define DIM_N_DIV_4 16
#define DIM_N_DIV_8 8
#endif

#ifndef combos
#define combos(X) X(bfloat16, bf16, float, f32, 8, 8, 8)
#endif

// Optional per-tile_m symbol suffix so multiple mm.o variants (e.g. DIM_M=32
// and DIM_M=64) can be linked into ONE ELF without symbol collision. Default
// empty (back-compat: the symbols keep their original names). Pass
// -DSYM_SUFFIX=_m64 etc. to disambiguate. SYM(name) -> name##SYM_SUFFIX via
// token paste.
#ifndef SYM_SUFFIX
#define SYM_SUFFIX
#endif
#define SYM_CAT2(a, b) a##b
#define SYM_CAT(a, b) SYM_CAT2(a, b)
#define SYM(name) SYM_CAT(name, SYM_SUFFIX)

#define matmul_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,            \
                                 mlir_type_out, r, s, t)                       \
  void SYM(op_has_no_registered_library_name)(                                 \
      ctype_in * a_in, ctype_in * b_in, ctype_out * c_out) {                   \
    matmul_vectorized_##r##x##s##x##t##_##mlir_type_in##_##mlir_type_out<      \
        DIM_M, DIM_K, DIM_N>(a_in, b_in, c_out);                               \
  }

#define matmul_scalar_c_func(ctype_in, mlir_type_in, ctype_out, mlir_type_out, \
                             r, s, t)                                          \
  void matmul_scalar_##mlir_type_in##_##mlir_type_out(                         \
      ctype_in *a_in, ctype_in *b_in, ctype_out *c_out) {                      \
    matmul_scalar<ctype_in, ctype_out, DIM_M, DIM_K, DIM_N>(a_in, b_in,        \
                                                            c_out);            \
  }

#define CAT2(a, b) a##b
#define CAT(a, b) CAT2(a, b)
#define MAKE_LINALG_FILL_NAME(mlir_in, mlir_out, N_div, M_div, tile)           \
  CAT(CAT(CAT(CAT(CAT(CAT(CAT(CAT(linalg_fill_, mlir_in), _view1x1x), N_div),  \
                      x),                                                      \
                  M_div),                                                      \
              x),                                                              \
          tile),                                                               \
      x##mlir_out##as2)

#define zero_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,              \
                               mlir_type_out, r, s, t)                         \
  void MAKE_LINALG_FILL_NAME(mlir_type_out, mlir_type_out, DIM_N_DIV_##t,      \
                             DIM_M_DIV_##r, r##x##t)(ctype_out * c_out) {      \
    zero_vectorized<ctype_out, DIM_M, DIM_N, 32>(c_out);                       \
  }

// Opt-in families. Both are OFF by default so the ten shipped LLM deployments
// that link this file keep byte-identical objects; pass
// -DGENERATE_MATMUL_INIT_KERNELS / -DGENERATE_MATMUL_WITH_ACC_KERNELS to emit
// them. Names follow `combos`, so the bf16-in/f32-out combo yields
// matmul_init_bf16_f32 and matmul_with_acc_bf16_f32, both SYM_SUFFIX-aware.
#define matmul_init_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,       \
                                      mlir_type_out, r, s, t)                  \
  void SYM(matmul_init_##mlir_type_in##_##mlir_type_out)(                      \
      ctype_in * a_in, ctype_in * b_in, ctype_out * c_out) {                   \
    matmul_init_vectorized_##r##x##s##x##t##_##mlir_type_in##_##mlir_type_out< \
        DIM_M, DIM_K, DIM_N>(a_in, b_in, c_out);                               \
  }

#define matmul_with_acc_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,       \
                                          mlir_type_out, r, s, t)                  \
  void SYM(matmul_with_acc_##mlir_type_in##_##mlir_type_out)(                      \
      ctype_in * a_in, ctype_in * b_in, ctype_out * c_acc,                         \
      ctype_out * c_out) {                                                         \
    matmul_with_acc_vectorized_##r##x##s##x##t##_##mlir_type_in##_##mlir_type_out< \
        DIM_M, DIM_K, DIM_N>(a_in, b_in, c_acc, c_out);                            \
  }

combos(matmul_vectorized_c_func) combos(matmul_scalar_c_func)
    combos(zero_vectorized_c_func)

#ifdef GENERATE_MATMUL_INIT_KERNELS
        combos(matmul_init_vectorized_c_func)
#endif

#ifdef GENERATE_MATMUL_WITH_ACC_KERNELS
            combos(matmul_with_acc_vectorized_c_func)
#endif

    // ---------------------------------------------------------------------------
    // Fixed-name wrappers for the manual-CallOp external-GEMM path (used when
    // the GEMM lives in a fused multi-launch ELF, where the air-linalg-to-func
    // pass can't run). Unlike the shape-named linalg_fill_* above, these names
    // are constant so the stitcher's _rename_all_with_externs can preserve
    // them. The accumulator C is f32 (FP32 partials across the whole K loop,
    // matching the registry BF16-GEMM standard); f32_to_bf16_mn does the single
    // epilogue cast to bf16 after the host's K-loop completes.

    // Zero the f32 L1 C accumulator (one call before the K-loop).
    void SYM(zero_f32_mn)(float *c) {
  zero_vectorized<float, DIM_M, DIM_N, 32>(c);
}

// f32 -> bf16 narrowing of the L1 C tile (one call after the K-loop).
void SYM(f32_to_bf16_mn)(float *src, bfloat16 *dst) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  constexpr unsigned VW = 16;
  constexpr unsigned NTOT = DIM_M * DIM_N;
  static_assert(NTOT % VW == 0, "DIM_M*DIM_N must be a multiple of 16");
  for (unsigned i = 0; i < NTOT; i += VW) {
    aie::vector<float, VW> v = aie::load_v<VW>(src + i);
    aie::vector<bfloat16, VW> vb;
    for (unsigned j = 0; j < VW; j++)
      vb[j] = (bfloat16)v[j];
    aie::store_v(dst + i, vb);
  }
}

// Length-parameterized f32 -> bf16 narrowing of a contiguous chunk of `n`
// elements. Used by the chunked-drain path: the bf16-out external GEMM casts
// the f32 C accumulator in G segments (one segment per call) into a small bf16
// drain buffer, so the full bf16 tile never has to coexist with the f32
// accumulator + A/B ping-pong in L1 (lets tile_m stay 64). `n` must be a
// multiple of 16.
void SYM(f32_to_bf16_n)(float *src, bfloat16 *dst, int n) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  constexpr unsigned VW = 16;
  for (int i = 0; i < n; i += VW) {
    aie::vector<float, VW> v = aie::load_v<VW>(src + i);
    aie::vector<bfloat16, VW> vb;
    for (unsigned j = 0; j < VW; j++)
      vb[j] = (bfloat16)v[j];
    aie::store_v(dst + i, vb);
  }
}

} // extern "C"
