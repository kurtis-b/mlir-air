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
#define EXTERNAL_BLOCK_M 2
#endif

#ifndef EXTERNAL_BLOCK_N
#define EXTERNAL_BLOCK_N 2
#endif

extern "C" void matmul_i8_i8_i8_acc32_strix(int8 *__restrict pA,
                                             int8 *__restrict pB,
                                             int8 *__restrict pC) {
  using MMUL = aie::mmul<8, 8, 8, int8, int8, acc32>;

  static_assert(MMUL::size_A == 64);
  static_assert(MMUL::size_B == 64);
  static_assert(MMUL::size_C == 64);

  constexpr unsigned kKPackCount = EXTERNAL_K_PACKS;
  constexpr unsigned kBlockM = EXTERNAL_BLOCK_M;
  constexpr unsigned kBlockN = EXTERNAL_BLOCK_N;
  constexpr unsigned kMPackCount = 18;
  constexpr unsigned kNPackCount = 18;
  constexpr unsigned kAStrideK = kMPackCount * MMUL::size_A;
  constexpr unsigned kBStrideN = kKPackCount * MMUL::size_B;
  constexpr unsigned kCStrideN = kMPackCount * MMUL::size_C;

  static_assert(kKPackCount > 0);
  static_assert(kKPackCount <= 18, "EXTERNAL_K_PACKS exceeds L1 budget");
  static_assert(kBlockN == 2, "only two-column N blocking is implemented");
  static_assert(kBlockM == 2 || kBlockM == 3,
                "only 2x2 and 3x2 blocking are implemented");
  static_assert(kMPackCount % kBlockM == 0);
  static_assert(kNPackCount % kBlockN == 0);

  aie::set_saturation(aie::saturation_mode::none);

  if constexpr (kBlockM == 3) {
    for (unsigned m = 0; m < kMPackCount; m += 3)
      chess_prepare_for_pipelining chess_loop_range(6, 6) {
        for (unsigned n = 0; n < kNPackCount; n += 2)
          chess_prepare_for_pipelining chess_loop_range(9, 9) {
            int8 *__restrict pC00 = pC + n * kCStrideN + m * MMUL::size_C;
            int8 *__restrict pC01 = pC + (n + 1) * kCStrideN + m * MMUL::size_C;
            int8 *__restrict pC10 = pC + n * kCStrideN + (m + 1) * MMUL::size_C;
            int8 *__restrict pC11 =
                pC + (n + 1) * kCStrideN + (m + 1) * MMUL::size_C;
            int8 *__restrict pC20 = pC + n * kCStrideN + (m + 2) * MMUL::size_C;
            int8 *__restrict pC21 =
                pC + (n + 1) * kCStrideN + (m + 2) * MMUL::size_C;

            MMUL C00(aie::load_v<MMUL::size_C>(pC00));
            MMUL C01(aie::load_v<MMUL::size_C>(pC01));
            MMUL C10(aie::load_v<MMUL::size_C>(pC10));
            MMUL C11(aie::load_v<MMUL::size_C>(pC11));
            MMUL C20(aie::load_v<MMUL::size_C>(pC20));
            MMUL C21(aie::load_v<MMUL::size_C>(pC21));

            const int8 __aie_dm_resource_a *__restrict pA0 =
                (const int8 __aie_dm_resource_a *)(pA + m * MMUL::size_A);
            const int8 __aie_dm_resource_a *__restrict pA1 =
                (const int8 __aie_dm_resource_a *)(pA + (m + 1) * MMUL::size_A);
            const int8 __aie_dm_resource_a *__restrict pA2 =
                (const int8 __aie_dm_resource_a *)(pA + (m + 2) * MMUL::size_A);
            const int8 __aie_dm_resource_b *__restrict pB0 =
                (const int8 __aie_dm_resource_b *)(pB + n * kBStrideN);
            const int8 __aie_dm_resource_b *__restrict pB1 =
                (const int8 __aie_dm_resource_b *)(pB + (n + 1) * kBStrideN);

            for (unsigned k = 0; k < kKPackCount; ++k)
              chess_prepare_for_pipelining chess_loop_range(EXTERNAL_K_PACKS,
                                                            EXTERNAL_K_PACKS) {
                aie::vector<int8, MMUL::size_A> A0 =
                    aie::load_v<MMUL::size_A>(pA0);
                aie::vector<int8, MMUL::size_A> A1 =
                    aie::load_v<MMUL::size_A>(pA1);
                aie::vector<int8, MMUL::size_A> A2 =
                    aie::load_v<MMUL::size_A>(pA2);
                aie::vector<int8, MMUL::size_B> B0 =
                    aie::load_v<MMUL::size_B>(pB0);
                aie::vector<int8, MMUL::size_B> B1 =
                    aie::load_v<MMUL::size_B>(pB1);

                pA0 += kAStrideK;
                pA1 += kAStrideK;
                pA2 += kAStrideK;
                pB0 += MMUL::size_B;
                pB1 += MMUL::size_B;

                C00.mac(A0, B0);
                C01.mac(A0, B1);
                C10.mac(A1, B0);
                C11.mac(A1, B1);
                C20.mac(A2, B0);
                C21.mac(A2, B1);
              }

            aie::store_v(pC00, C00.template to_vector<int8>());
            aie::store_v(pC01, C01.template to_vector<int8>());
            aie::store_v(pC10, C10.template to_vector<int8>());
            aie::store_v(pC11, C11.template to_vector<int8>());
            aie::store_v(pC20, C20.template to_vector<int8>());
            aie::store_v(pC21, C21.template to_vector<int8>());
          }
      }
  } else {
    for (unsigned m = 0; m < kMPackCount; m += 2)
      chess_prepare_for_pipelining chess_loop_range(9, 9) {
        for (unsigned n = 0; n < kNPackCount; n += 2)
          chess_prepare_for_pipelining chess_loop_range(9, 9) {
            int8 *__restrict pC00 = pC + n * kCStrideN + m * MMUL::size_C;
            int8 *__restrict pC01 = pC + (n + 1) * kCStrideN + m * MMUL::size_C;
            int8 *__restrict pC10 = pC + n * kCStrideN + (m + 1) * MMUL::size_C;
            int8 *__restrict pC11 =
                pC + (n + 1) * kCStrideN + (m + 1) * MMUL::size_C;

            MMUL C00(aie::load_v<MMUL::size_C>(pC00));
            MMUL C01(aie::load_v<MMUL::size_C>(pC01));
            MMUL C10(aie::load_v<MMUL::size_C>(pC10));
            MMUL C11(aie::load_v<MMUL::size_C>(pC11));

            const int8 __aie_dm_resource_a *__restrict pA0 =
                (const int8 __aie_dm_resource_a *)(pA + m * MMUL::size_A);
            const int8 __aie_dm_resource_a *__restrict pA1 =
                (const int8 __aie_dm_resource_a *)(pA + (m + 1) * MMUL::size_A);
            const int8 __aie_dm_resource_b *__restrict pB0 =
                (const int8 __aie_dm_resource_b *)(pB + n * kBStrideN);
            const int8 __aie_dm_resource_b *__restrict pB1 =
                (const int8 __aie_dm_resource_b *)(pB + (n + 1) * kBStrideN);

            for (unsigned k = 0; k < kKPackCount; ++k)
              chess_prepare_for_pipelining chess_loop_range(EXTERNAL_K_PACKS,
                                                            EXTERNAL_K_PACKS) {
                aie::vector<int8, MMUL::size_A> A0 =
                    aie::load_v<MMUL::size_A>(pA0);
                aie::vector<int8, MMUL::size_A> A1 =
                    aie::load_v<MMUL::size_A>(pA1);
                aie::vector<int8, MMUL::size_B> B0 =
                    aie::load_v<MMUL::size_B>(pB0);
                aie::vector<int8, MMUL::size_B> B1 =
                    aie::load_v<MMUL::size_B>(pB1);

                pA0 += kAStrideK;
                pA1 += kAStrideK;
                pB0 += MMUL::size_B;
                pB1 += MMUL::size_B;

                C00.mac(A0, B0);
                C01.mac(A0, B1);
                C10.mac(A1, B0);
                C11.mac(A1, B1);
              }

            aie::store_v(pC00, C00.template to_vector<int8>());
            aie::store_v(pC01, C01.template to_vector<int8>());
            aie::store_v(pC10, C10.template to_vector<int8>());
            aie::store_v(pC11, C11.template to_vector<int8>());
          }
      }
  }
}
