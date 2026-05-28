// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Q4NX block dequantization for one 32x256-style block:
//   out[row, col] = scale[col] * q4[row, col] + min[col]

#include <aie_api/aie.hpp>
#include <cstdint>

#ifndef Q4NX_ROWS
#define Q4NX_ROWS 32
#endif

#ifndef Q4NX_COLS
#define Q4NX_COLS 256
#endif

extern "C" {

void q4nx_dequant_block(const uint8_t *__restrict packed,
                        const bfloat16 *__restrict scale,
                        const bfloat16 *__restrict min_offset,
                        bfloat16 *__restrict out) {
  constexpr unsigned elems = Q4NX_ROWS * Q4NX_COLS;
  for (unsigned i = 0; i < elems; i += 2)
    chess_prepare_for_pipelining {
      uint8_t p = packed[i >> 1];
      unsigned col0 = i % Q4NX_COLS;
      unsigned col1 = (i + 1) % Q4NX_COLS;
      float q0 = static_cast<float>(p & 0x0F);
      float q1 = static_cast<float>((p >> 4) & 0x0F);
      float s0 = static_cast<float>(scale[col0]);
      float s1 = static_cast<float>(scale[col1]);
      float m0 = static_cast<float>(min_offset[col0]);
      float m1 = static_cast<float>(min_offset[col1]);
      out[i] = static_cast<bfloat16>(q0 * s0 + m0);
      out[i + 1] = static_cast<bfloat16>(q1 * s1 + m1);
    }
}

} // extern "C"
