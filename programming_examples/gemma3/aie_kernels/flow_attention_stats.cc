// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Tiled FlowQKV attention-stat kernel. Each invocation consumes one KV tile and
// emits the per-query softmax maximum, denominator, and unnormalized numerator.
// The host or a future reduction kernel can merge tile statistics into the full
// attention result without staging the full KV cache in tile-local memory.

#include <aie_api/aie.hpp>
#include <cstdint>

#ifndef Q_CHUNK
#define Q_CHUNK 4
#endif

#ifndef KV_TILE
#define KV_TILE 32
#endif

#ifndef HEAD_DIM
#define HEAD_DIM 256
#endif

static inline float fast_exp_approx(float x) {
  if (x < -20.0f)
    return 0.0f;
  if (x > 20.0f)
    x = 20.0f;
  union {
    uint32_t i;
    float f;
  } v;
  v.i = static_cast<uint32_t>(12102203.0f * x + 1064866805.0f);
  return v.f;
}

extern "C" {

void flowqkv_tile_stats_bf16(const bfloat16 *__restrict q,
                             const bfloat16 *__restrict k,
                             const bfloat16 *__restrict v,
                             float *__restrict stats) {
  constexpr unsigned stats_stride = HEAD_DIM + 2;
  const float inv_sqrt_d = 1.0f / __builtin_sqrtf((float)HEAD_DIM);

  for (unsigned qi = 0; qi < Q_CHUNK; ++qi) {
    float max_score = -3.402823466e38f;
    for (unsigned kk = 0; kk < KV_TILE; ++kk) {
      float score = 0.0f;
      for (unsigned d = 0; d < HEAD_DIM; ++d)
        score += static_cast<float>(q[qi * HEAD_DIM + d]) *
                 static_cast<float>(k[kk * HEAD_DIM + d]);
      score *= inv_sqrt_d;
      max_score = score > max_score ? score : max_score;
    }

    float denom = 0.0f;
    for (unsigned kk = 0; kk < KV_TILE; ++kk) {
      float score = 0.0f;
      for (unsigned d = 0; d < HEAD_DIM; ++d)
        score += static_cast<float>(q[qi * HEAD_DIM + d]) *
                 static_cast<float>(k[kk * HEAD_DIM + d]);
      denom += fast_exp_approx(score * inv_sqrt_d - max_score);
    }

    float *row = stats + qi * stats_stride;
    row[0] = max_score;
    row[1] = denom;

    for (unsigned d = 0; d < HEAD_DIM; ++d) {
      float acc = 0.0f;
      for (unsigned kk = 0; kk < KV_TILE; ++kk) {
        float score = 0.0f;
        for (unsigned kd = 0; kd < HEAD_DIM; ++kd)
          score += static_cast<float>(q[qi * HEAD_DIM + kd]) *
                   static_cast<float>(k[kk * HEAD_DIM + kd]);
        float weight = fast_exp_approx(score * inv_sqrt_d - max_score);
        acc += weight * static_cast<float>(v[kk * HEAD_DIM + d]);
      }
      row[d + 2] = acc;
    }
  }
}

} // extern "C"
