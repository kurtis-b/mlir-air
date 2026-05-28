// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Fused Q4NX dequantization and projection for one 32x256 block.

#include <aie_api/aie.hpp>
#include <cstdint>

#ifndef Q4NX_ROWS
#define Q4NX_ROWS 32
#endif

#ifndef Q4NX_COLS
#define Q4NX_COLS 256
#endif

extern "C" {

void fused_dqp_block(const uint8_t *__restrict packed,
                     const bfloat16 *__restrict scale,
                     const bfloat16 *__restrict min_offset,
                     const bfloat16 *__restrict activation,
                     bfloat16 *__restrict out) {
  for (unsigned row = 0; row < Q4NX_ROWS; ++row) {
    float acc = 0.0f;
    const unsigned row_base = row * Q4NX_COLS;
    for (unsigned col = 0; col < Q4NX_COLS; ++col)
      chess_prepare_for_pipelining {
        unsigned idx = row_base + col;
        uint8_t p = packed[idx >> 1];
        unsigned q = (idx & 1) ? ((p >> 4) & 0x0F) : (p & 0x0F);
        float w = static_cast<float>(q) * static_cast<float>(scale[col]) +
                  static_cast<float>(min_offset[col]);
        acc += w * static_cast<float>(activation[col]);
      }
    out[row] = static_cast<bfloat16>(acc);
  }
}

} // extern "C"
