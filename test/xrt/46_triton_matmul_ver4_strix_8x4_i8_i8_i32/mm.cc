//===- mm.cc -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>

#include <aie_api/aie.hpp>

#ifndef EXTERNAL_K_PACKS
#define EXTERNAL_K_PACKS 9
#endif

#ifndef EXTERNAL_BLOCK_M
#define EXTERNAL_BLOCK_M 3
#endif

#ifndef EXTERNAL_BLOCK_N
#define EXTERNAL_BLOCK_N 2
#endif

#ifndef EXTERNAL_CORE_M_PACKS
#define EXTERNAL_CORE_M_PACKS 18
#endif

#ifndef EXTERNAL_ACTIVE_M_PACKS
#define EXTERNAL_ACTIVE_M_PACKS 18
#endif

#ifndef EXTERNAL_CORE_N_PACKS
#define EXTERNAL_CORE_N_PACKS 18
#endif

#ifndef EXTERNAL_C_STRIDE_M_PACKS
#define EXTERNAL_C_STRIDE_M_PACKS 18
#endif

#define EXTERNAL_KERNEL_STYLE_HAND_SCHEDULED 1

#ifndef EXTERNAL_KERNEL_STYLE
#define EXTERNAL_KERNEL_STYLE EXTERNAL_KERNEL_STYLE_HAND_SCHEDULED
#endif

#define EXTERNAL_SCHEDULE_SOFTWARE_PIPELINE 3

#ifndef EXTERNAL_SCHEDULE
#define EXTERNAL_SCHEDULE EXTERNAL_SCHEDULE_SOFTWARE_PIPELINE
#endif

#if EXTERNAL_SCHEDULE != EXTERNAL_SCHEDULE_SOFTWARE_PIPELINE
#error "only the retained software-pipelined SOTA INT8 GEMM kernel is supported"
#endif

#define EXTERNAL_M_LOOP_ATTR(MIN, MAX)                                           \
  chess_prepare_for_pipelining chess_loop_range(MIN, MAX)
#define EXTERNAL_N_LOOP_ATTR(MIN, MAX)                                           \
  chess_prepare_for_pipelining chess_loop_range(MIN, MAX)
#define EXTERNAL_K_LOOP_ATTR(MIN, MAX)                                           \
  chess_prepare_for_pipelining chess_loop_range(MIN, MAX)

#if EXTERNAL_KERNEL_STYLE != EXTERNAL_KERNEL_STYLE_HAND_SCHEDULED
#error "only the retained hand-scheduled SOTA INT8 GEMM kernel is supported"
#endif

template <typename MMUL>
static inline void matmul_pack_2x2_streamed(int8 *__restrict pA,
                                            int8 *__restrict pB,
                                            int8 *__restrict pC, unsigned m,
                                            unsigned n) {
  constexpr unsigned kKPackCount = EXTERNAL_K_PACKS;
  constexpr unsigned kActiveMPackCount = EXTERNAL_ACTIVE_M_PACKS;
  constexpr unsigned kAStrideK = kActiveMPackCount * MMUL::size_A;
  constexpr unsigned kBStrideN = kKPackCount * MMUL::size_B;
  constexpr unsigned kCStrideN = EXTERNAL_C_STRIDE_M_PACKS * MMUL::size_C;

  MMUL C00(aie::load_v<MMUL::size_C>(pC + n * kCStrideN + m * MMUL::size_C));
  MMUL C01(
      aie::load_v<MMUL::size_C>(pC + (n + 1) * kCStrideN + m * MMUL::size_C));
  MMUL C10(
      aie::load_v<MMUL::size_C>(pC + n * kCStrideN + (m + 1) * MMUL::size_C));
  MMUL C11(aie::load_v<MMUL::size_C>(
      pC + (n + 1) * kCStrideN + (m + 1) * MMUL::size_C));

  const int8 __aie_dm_resource_a *__restrict pA0 =
      (const int8 __aie_dm_resource_a *)(pA + m * MMUL::size_A);
  const int8 __aie_dm_resource_a *__restrict pA1 =
      (const int8 __aie_dm_resource_a *)(pA + (m + 1) * MMUL::size_A);
  const int8 __aie_dm_resource_b *__restrict pB0 =
      (const int8 __aie_dm_resource_b *)(pB + n * kBStrideN);
  const int8 __aie_dm_resource_b *__restrict pB1 =
      (const int8 __aie_dm_resource_b *)(pB + (n + 1) * kBStrideN);

  for (unsigned k = 0; k < kKPackCount; ++k)
    EXTERNAL_K_LOOP_ATTR(EXTERNAL_K_PACKS, EXTERNAL_K_PACKS) {
      aie::vector<int8, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB0);
      aie::vector<int8, MMUL::size_B> B1 = aie::load_v<MMUL::size_B>(pB1);
      pB0 += MMUL::size_B;
      pB1 += MMUL::size_B;

      {
        aie::vector<int8, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA0);
        pA0 += kAStrideK;
        C00.mac(A0, B0);
        C01.mac(A0, B1);
      }
      {
        aie::vector<int8, MMUL::size_A> A1 = aie::load_v<MMUL::size_A>(pA1);
        pA1 += kAStrideK;
        C10.mac(A1, B0);
        C11.mac(A1, B1);
      }
    }

  aie::store_v(pC + n * kCStrideN + m * MMUL::size_C,
               C00.template to_vector<int8>());
  aie::store_v(pC + (n + 1) * kCStrideN + m * MMUL::size_C,
               C01.template to_vector<int8>());
  aie::store_v(pC + n * kCStrideN + (m + 1) * MMUL::size_C,
               C10.template to_vector<int8>());
  aie::store_v(pC + (n + 1) * kCStrideN + (m + 1) * MMUL::size_C,
               C11.template to_vector<int8>());
}

template <typename MMUL>
static inline void matmul_pack_1x2_streamed(int8 *__restrict pA,
                                            int8 *__restrict pB,
                                            int8 *__restrict pC, unsigned m,
                                            unsigned n) {
  constexpr unsigned kKPackCount = EXTERNAL_K_PACKS;
  constexpr unsigned kActiveMPackCount = EXTERNAL_ACTIVE_M_PACKS;
  constexpr unsigned kAStrideK = kActiveMPackCount * MMUL::size_A;
  constexpr unsigned kBStrideN = kKPackCount * MMUL::size_B;
  constexpr unsigned kCStrideN = EXTERNAL_C_STRIDE_M_PACKS * MMUL::size_C;

  MMUL C00(aie::load_v<MMUL::size_C>(pC + n * kCStrideN + m * MMUL::size_C));
  MMUL C01(
      aie::load_v<MMUL::size_C>(pC + (n + 1) * kCStrideN + m * MMUL::size_C));

  const int8 __aie_dm_resource_a *__restrict pA0 =
      (const int8 __aie_dm_resource_a *)(pA + m * MMUL::size_A);
  const int8 __aie_dm_resource_b *__restrict pB0 =
      (const int8 __aie_dm_resource_b *)(pB + n * kBStrideN);
  const int8 __aie_dm_resource_b *__restrict pB1 =
      (const int8 __aie_dm_resource_b *)(pB + (n + 1) * kBStrideN);

  for (unsigned k = 0; k < kKPackCount; ++k)
    EXTERNAL_K_LOOP_ATTR(EXTERNAL_K_PACKS, EXTERNAL_K_PACKS) {
      aie::vector<int8, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB0);
      aie::vector<int8, MMUL::size_B> B1 = aie::load_v<MMUL::size_B>(pB1);
      aie::vector<int8, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA0);
      pB0 += MMUL::size_B;
      pB1 += MMUL::size_B;
      pA0 += kAStrideK;
      C00.mac(A0, B0);
      C01.mac(A0, B1);
    }

  aie::store_v(pC + n * kCStrideN + m * MMUL::size_C,
               C00.template to_vector<int8>());
  aie::store_v(pC + (n + 1) * kCStrideN + m * MMUL::size_C,
               C01.template to_vector<int8>());
}

extern "C" void matmul_i8_i8_i8_acc32_strix(int8 *__restrict pA,
                                             int8 *__restrict pB,
                                             int8 *__restrict pC) {
  using MMUL = aie::mmul<8, 8, 8, int8, int8, acc32>;

  static_assert(MMUL::size_A == 64);
  static_assert(MMUL::size_B == 64);
  static_assert(MMUL::size_C == 64);
  constexpr bool kLegacySotaProfile =
      EXTERNAL_K_PACKS == 9 && EXTERNAL_BLOCK_M == 3 &&
      EXTERNAL_BLOCK_N == 2 && EXTERNAL_CORE_M_PACKS == 18 &&
      EXTERNAL_ACTIVE_M_PACKS == 18 && EXTERNAL_CORE_N_PACKS == 18 &&
      EXTERNAL_C_STRIDE_M_PACKS == 18;
  constexpr bool kPowerOfTwoProfile =
      EXTERNAL_K_PACKS == 8 && EXTERNAL_BLOCK_M == 2 &&
      EXTERNAL_BLOCK_N == 2 && EXTERNAL_CORE_M_PACKS == 16 &&
      EXTERNAL_ACTIVE_M_PACKS == 16 && EXTERNAL_CORE_N_PACKS == 16 &&
      EXTERNAL_C_STRIDE_M_PACKS == 16;
  static_assert(kLegacySotaProfile || kPowerOfTwoProfile,
                "supported profiles are legacy 18x18/K9 and power2 16x16/K8");
  static_assert(EXTERNAL_BLOCK_N == 2,
                "retained hand-scheduled kernel writes N in 2-pack groups");
  static_assert(EXTERNAL_CORE_M_PACKS == EXTERNAL_ACTIVE_M_PACKS,
                "partial active-M profiles need a separate C routing contract");
  static_assert(EXTERNAL_CORE_M_PACKS == EXTERNAL_C_STRIDE_M_PACKS,
                "C stride must match the full core M pack count");

  constexpr unsigned kActiveMPackCount = EXTERNAL_ACTIVE_M_PACKS;
  constexpr unsigned kNPackCount = EXTERNAL_CORE_N_PACKS;
  static_assert(EXTERNAL_BLOCK_M != 3 || kActiveMPackCount % 3 == 0,
                "3x2 profile requires M packs to be divisible by 3");
  static_assert(EXTERNAL_BLOCK_M != 2 ||
                    (kActiveMPackCount % 2 == 0 && kNPackCount % 2 == 0),
                "2x2 profile requires even M and N pack counts");

  aie::set_saturation(aie::saturation_mode::none);

  if constexpr (EXTERNAL_BLOCK_M == 3) {
    for (unsigned m = 0; m < kActiveMPackCount; m += 3)
      EXTERNAL_M_LOOP_ATTR(kActiveMPackCount / 3, kActiveMPackCount / 3) {
        for (unsigned n = 0; n < kNPackCount; n += 2)
          EXTERNAL_N_LOOP_ATTR(kNPackCount / 2, kNPackCount / 2) {
            matmul_pack_2x2_streamed<MMUL>(pA, pB, pC, m, n);
            matmul_pack_1x2_streamed<MMUL>(pA, pB, pC, m + 2, n);
          }
      }
  } else {
    for (unsigned m = 0; m < kActiveMPackCount; m += 2)
      EXTERNAL_M_LOOP_ATTR(kActiveMPackCount / 2, kActiveMPackCount / 2) {
        for (unsigned n = 0; n < kNPackCount; n += 2)
          EXTERNAL_N_LOOP_ATTR(kNPackCount / 2, kNPackCount / 2) {
            matmul_pack_2x2_streamed<MMUL>(pA, pB, pC, m, n);
          }
      }
  }
}
