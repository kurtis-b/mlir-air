//===- mm.cc -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>

#include <aie_api/aie.hpp>

extern "C" void matmul_i8_i8_i8_acc32_strix(int8 *__restrict pA,
                                             int8 *__restrict pB,
                                             int8 *__restrict pC) {
  using MMUL = aie::mmul<8, 8, 8, int8, int8, acc32>;

  static_assert(MMUL::size_A == 64);
  static_assert(MMUL::size_B == 64);
  static_assert(MMUL::size_C == 64);

  constexpr unsigned kKPackCount = 9;
  constexpr unsigned kMPackCount = 18;
  constexpr unsigned kNPackCount = 18;
  constexpr unsigned kAStrideK = kMPackCount * MMUL::size_A;
  constexpr unsigned kBStrideN = kKPackCount * MMUL::size_B;
  constexpr unsigned kCStrideN = kMPackCount * MMUL::size_C;

  aie::set_saturation(aie::saturation_mode::none);

  for (unsigned m = 0; m < kMPackCount; m += 2)
    chess_prepare_for_pipelining chess_loop_range(9, 9) {
      for (unsigned n = 0; n < kNPackCount; n += 2)
        chess_prepare_for_pipelining chess_loop_range(9, 9) {
          int8 *__restrict pC00 = pC + n * kCStrideN + m * MMUL::size_C;
          int8 *__restrict pC01 = pC + (n + 1) * kCStrideN + m * MMUL::size_C;
          int8 *__restrict pC10 = pC + n * kCStrideN + (m + 1) * MMUL::size_C;
          int8 *__restrict pC11 = pC + (n + 1) * kCStrideN + (m + 1) * MMUL::size_C;

          MMUL C00(aie::load_v<MMUL::size_C>(pC00));
          MMUL C01(aie::load_v<MMUL::size_C>(pC01));
          MMUL C10(aie::load_v<MMUL::size_C>(pC10));
          MMUL C11(aie::load_v<MMUL::size_C>(pC11));

          for (unsigned k = 0; k < kKPackCount; ++k)
            chess_prepare_for_pipelining chess_loop_range(9, 9) {
              const int8 *__restrict pA0 = pA + k * kAStrideK + m * MMUL::size_A;
              const int8 *__restrict pA1 =
                  pA + k * kAStrideK + (m + 1) * MMUL::size_A;
              const int8 *__restrict pB0 = pB + n * kBStrideN + k * MMUL::size_B;
              const int8 *__restrict pB1 =
                  pB + (n + 1) * kBStrideN + k * MMUL::size_B;

              aie::vector<int8, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA0);
              aie::vector<int8, MMUL::size_A> A1 = aie::load_v<MMUL::size_A>(pA1);
              aie::vector<int8, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB0);
              aie::vector<int8, MMUL::size_B> B1 = aie::load_v<MMUL::size_B>(pB1);

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
