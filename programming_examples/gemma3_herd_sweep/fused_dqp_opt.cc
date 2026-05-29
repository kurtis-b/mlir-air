// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Optimized fused Q4NX dequantization and projection. The paper-style entry
// point accumulates multiple Q4NX 32x256 column blocks into one output vector;
// the smoke-test entry point keeps the original one-block overwrite behavior.

#include <aie_api/aie.hpp>
#include <cstdint>

#ifndef Q4NX_ROWS
#define Q4NX_ROWS 32
#endif

#ifndef Q4NX_COLS
#define Q4NX_COLS 256
#endif

constexpr aie::rounding_mode kDqpRoundMode = aie::rounding_mode::conv_even;

static inline bfloat16 q4_to_bf16(uint8_t packed, bool high) {
  uint8_t nibble = high ? ((packed >> 4) & 0x0F) : (packed & 0x0F);
  return static_cast<bfloat16>(static_cast<float>(nibble));
}

template <bool Accumulate>
static void fused_dqp_impl(const uint8_t *__restrict packed,
                           const bfloat16 *__restrict scale,
                           const bfloat16 *__restrict min_offset,
                           const bfloat16 *__restrict activation,
                           bfloat16 *__restrict out) {
  constexpr int VecLen = 8;
  constexpr int RowSubBlock = 16;
  static_assert(Q4NX_COLS % VecLen == 0,
                "optimized FusedDQP expects column count divisible by 8");
  static_assert(Q4NX_ROWS % RowSubBlock == 0,
                "optimized FusedDQP expects row count divisible by 16");

  ::aie::set_rounding(kDqpRoundMode);

  for (unsigned row_base = 0; row_base < Q4NX_ROWS; row_base += RowSubBlock) {
    float acc[RowSubBlock];
    for (unsigned r = 0; r < RowSubBlock; ++r) {
      acc[r] = Accumulate ? static_cast<float>(out[row_base + r]) : 0.0f;
    }

    for (unsigned col = 0; col < Q4NX_COLS; col += VecLen)
      chess_prepare_for_pipelining chess_loop_range(4, ) {
        aie::vector<bfloat16, VecLen> s = aie::load_v<VecLen>(scale + col);
        aie::vector<bfloat16, VecLen> m = aie::load_v<VecLen>(min_offset + col);
        aie::vector<bfloat16, VecLen> a = aie::load_v<VecLen>(activation + col);

        for (unsigned r = 0; r < RowSubBlock; ++r) {
          const unsigned idx = (row_base + r) * Q4NX_COLS + col;
          const uint8_t p0 = packed[(idx >> 1) + 0];
          const uint8_t p1 = packed[(idx >> 1) + 1];
          const uint8_t p2 = packed[(idx >> 1) + 2];
          const uint8_t p3 = packed[(idx >> 1) + 3];
          const bfloat16 q_buf[VecLen] = {
              q4_to_bf16(p0, false), q4_to_bf16(p0, true),
              q4_to_bf16(p1, false), q4_to_bf16(p1, true),
              q4_to_bf16(p2, false), q4_to_bf16(p2, true),
              q4_to_bf16(p3, false), q4_to_bf16(p3, true),
          };

          aie::vector<bfloat16, VecLen> q = aie::load_v<VecLen>(q_buf);
          aie::accum<accfloat, VecLen> w = aie::mul(q, s);
          w = aie::add(w, m);
          aie::accum<accfloat, VecLen> prod =
              aie::mul(w.to_vector<bfloat16>(), a);
          acc[r] += aie::reduce_add(prod.to_vector<float>());
        }
      }

    for (unsigned r = 0; r < RowSubBlock; ++r) {
      out[row_base + r] = static_cast<bfloat16>(acc[r]);
    }
  }
}

extern "C" {

void fused_dqp_block_opt(const uint8_t *__restrict packed,
                         const bfloat16 *__restrict scale,
                         const bfloat16 *__restrict min_offset,
                         const bfloat16 *__restrict activation,
                         bfloat16 *__restrict out) {
  fused_dqp_impl<false>(packed, scale, min_offset, activation, out);
}

void fused_dqp_accum_block_opt(const uint8_t *__restrict packed,
                               const bfloat16 *__restrict scale,
                               const bfloat16 *__restrict min_offset,
                               const bfloat16 *__restrict activation,
                               bfloat16 *__restrict out) {
  fused_dqp_impl<true>(packed, scale, min_offset, activation, out);
}

} // extern "C"
