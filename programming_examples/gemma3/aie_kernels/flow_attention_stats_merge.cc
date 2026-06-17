// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Merge tiled FlowQKV attention statistics into one attention output row.
// Each invocation consumes all KV-tile statistics for a single query row:
//   stats[tile][max, denom, numerator[HEAD_DIM]]
// and writes HEAD_DIM bf16 values.

#include <aie_api/aie.hpp>
#include <cstdint>

#ifndef TILE_COUNT
#define TILE_COUNT 32
#endif

#ifndef HEAD_DIM
#define HEAD_DIM 256
#endif

static inline float fast_exp_approx(float x) {
  if (x < -20.0f)
    return 0.0f;
  if (x > 20.0f)
    x = 20.0f;

  constexpr float log2e = 1.4426950408889634f;
  constexpr float ln2 = 0.6931471805599453f;
  float y = x * log2e;
  int exponent = static_cast<int>(y);
  if (static_cast<float>(exponent) > y)
    --exponent;

  const float r = x - static_cast<float>(exponent) * ln2;
  float scale =
      1.0f +
      r * (1.0f +
           r * (0.5f +
                r * (0.1666666716f +
                     r * (0.0416666679f + r * 0.0083333338f))));
  while (exponent < 0) {
    scale *= 0.5f;
    ++exponent;
  }
  while (exponent > 0) {
    scale *= 2.0f;
    --exponent;
  }
  return scale;
}

extern "C" {

void flowqkv_tiled_stats_merge_bf16(const float *__restrict stats,
                                    bfloat16 *__restrict out) {
  constexpr unsigned stats_stride = HEAD_DIM + 2;

  float global_max = -3.402823466e38f;
  for (unsigned tile = 0; tile < TILE_COUNT; ++tile) {
    const float max_score = stats[tile * stats_stride];
    global_max = max_score > global_max ? max_score : global_max;
  }

  float denom = 0.0f;
  for (unsigned tile = 0; tile < TILE_COUNT; ++tile) {
    const float *row = stats + tile * stats_stride;
    denom += row[1] * fast_exp_approx(row[0] - global_max);
  }
  const float inv_denom = denom > 0.0f ? (1.0f / denom) : 0.0f;

  for (unsigned d = 0; d < HEAD_DIM; ++d) {
    float acc = 0.0f;
    for (unsigned tile = 0; tile < TILE_COUNT; ++tile) {
      const float *row = stats + tile * stats_stride;
      const float scale = fast_exp_approx(row[0] - global_max);
      acc += row[d + 2] * scale;
    }
    out[d] = static_cast<bfloat16>(acc * inv_denom);
  }
}

} // extern "C"
