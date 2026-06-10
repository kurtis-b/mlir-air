// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Optimized Q4NX block dequantization for one 32x256-style block.
// The microkernel expands packed int4 values in 8-column vectors, then uses
// BF16 vector multiply-add against per-column scale and minimum vectors.

#include <aie_api/aie.hpp>
#include <cstdint>

#ifndef Q4NX_ROWS
#define Q4NX_ROWS 32
#endif

#ifndef Q4NX_COLS
#define Q4NX_COLS 256
#endif

constexpr aie::rounding_mode kQ4nxRoundMode = aie::rounding_mode::conv_even;

static inline bfloat16 q4_to_bf16(uint8_t packed, bool high) {
  uint8_t nibble = high ? ((packed >> 4) & 0x0F) : (packed & 0x0F);
  return static_cast<bfloat16>(static_cast<float>(nibble));
}

extern "C" {

void q4nx_dequant_block_opt(const uint8_t *__restrict packed,
                            const bfloat16 *__restrict scale,
                            const bfloat16 *__restrict min_offset,
                            bfloat16 *__restrict out) {
  constexpr int VecLen = 8;
  constexpr unsigned elems = Q4NX_ROWS * Q4NX_COLS;
  static_assert(Q4NX_COLS % VecLen == 0,
                "optimized Q4NX expects column count divisible by 8");

  ::aie::set_rounding(kQ4nxRoundMode);

  for (unsigned i = 0; i < elems; i += VecLen)
    chess_prepare_for_pipelining chess_loop_range(4, ) {
      const unsigned col = i % Q4NX_COLS;
      const uint8_t p0 = packed[(i >> 1) + 0];
      const uint8_t p1 = packed[(i >> 1) + 1];
      const uint8_t p2 = packed[(i >> 1) + 2];
      const uint8_t p3 = packed[(i >> 1) + 3];
      const bfloat16 q_buf[VecLen] = {
          q4_to_bf16(p0, false), q4_to_bf16(p0, true),
          q4_to_bf16(p1, false), q4_to_bf16(p1, true),
          q4_to_bf16(p2, false), q4_to_bf16(p2, true),
          q4_to_bf16(p3, false), q4_to_bf16(p3, true),
      };

      aie::vector<bfloat16, VecLen> q = aie::load_v<VecLen>(q_buf);
      aie::vector<bfloat16, VecLen> s = aie::load_v<VecLen>(scale + col);
      aie::vector<bfloat16, VecLen> m = aie::load_v<VecLen>(min_offset + col);
      aie::accum<accfloat, VecLen> result = aie::mul(q, s);
      result = aie::add(result, m);
      aie::store_v(out + i, result.to_vector<bfloat16>());
    }
}

} // extern "C"
