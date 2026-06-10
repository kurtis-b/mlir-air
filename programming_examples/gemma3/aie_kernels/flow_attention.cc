// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Correctness-first FlowQKV/FlowKV attention kernels. The implementation uses
// online softmax semantics over a streamed KV range. It is intentionally simple
// and is meant to be replaced by a multi-CT scheduled version after validation.

#include <aie_api/aie.hpp>
#include <cstdint>

#ifndef Q_CHUNK
#define Q_CHUNK 4
#endif

#ifndef KV_LEN
#define KV_LEN 32
#endif

#ifndef HEAD_DIM
#define HEAD_DIM 64
#endif

#ifndef QUERY_BASE
#define QUERY_BASE 0
#endif

#ifndef WINDOW_LEN
#define WINDOW_LEN 0
#endif

#ifndef CAUSAL
#define CAUSAL 0
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

template <unsigned QRows>
static void attention_chunk(const bfloat16 *__restrict q,
                            const bfloat16 *__restrict k,
                            const bfloat16 *__restrict v,
                            bfloat16 *__restrict out) {
  const float inv_sqrt_d = 1.0f / __builtin_sqrtf((float)HEAD_DIM);

  for (unsigned qi = 0; qi < QRows; ++qi) {
    unsigned end = KV_LEN;
    if constexpr (CAUSAL != 0) {
      unsigned causal_end = QUERY_BASE + qi + 1;
      end = causal_end < end ? causal_end : end;
    }
    unsigned start = 0;
    if constexpr (WINDOW_LEN > 0) {
      start = (end > WINDOW_LEN) ? (end - WINDOW_LEN) : 0;
    }

    float max_score = -3.402823466e38f;
    for (unsigned kk = start; kk < end; ++kk) {
      float score = 0.0f;
      for (unsigned d = 0; d < HEAD_DIM; ++d)
        score += static_cast<float>(q[qi * HEAD_DIM + d]) *
                 static_cast<float>(k[kk * HEAD_DIM + d]);
      score *= inv_sqrt_d;
      max_score = score > max_score ? score : max_score;
    }

    float denom = 0.0f;
    for (unsigned kk = start; kk < end; ++kk) {
      float score = 0.0f;
      for (unsigned d = 0; d < HEAD_DIM; ++d)
        score += static_cast<float>(q[qi * HEAD_DIM + d]) *
                 static_cast<float>(k[kk * HEAD_DIM + d]);
      denom += fast_exp_approx(score * inv_sqrt_d - max_score);
    }
    float inv_denom = denom > 0.0f ? (1.0f / denom) : 0.0f;

    for (unsigned d = 0; d < HEAD_DIM; ++d) {
      float acc = 0.0f;
      for (unsigned kk = start; kk < end; ++kk) {
        float score = 0.0f;
        for (unsigned kd = 0; kd < HEAD_DIM; ++kd)
          score += static_cast<float>(q[qi * HEAD_DIM + kd]) *
                   static_cast<float>(k[kk * HEAD_DIM + kd]);
        float weight = fast_exp_approx(score * inv_sqrt_d - max_score) * inv_denom;
        acc += weight * static_cast<float>(v[kk * HEAD_DIM + d]);
      }
      out[qi * HEAD_DIM + d] = static_cast<bfloat16>(acc);
    }
  }
}

extern "C" {

void flowqkv_chunk_bf16(const bfloat16 *__restrict q,
                        const bfloat16 *__restrict k,
                        const bfloat16 *__restrict v,
                        bfloat16 *__restrict out) {
  attention_chunk<Q_CHUNK>(q, k, v, out);
}

void flowkv_decode_bf16(const bfloat16 *__restrict q,
                        const bfloat16 *__restrict k,
                        const bfloat16 *__restrict v,
                        bfloat16 *__restrict out) {
  attention_chunk<1>(q, k, v, out);
}

} // extern "C"
