// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Optimized FlowQKV/FlowKV attention microkernels. These kernels keep the
// correctness-first online-softmax semantics while vectorizing the BF16 dot
// products and value accumulation on AIE2P vector lanes.

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

constexpr aie::rounding_mode kFlowRoundMode = aie::rounding_mode::conv_even;

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

static inline float dot_bf16_v8(const bfloat16 *__restrict lhs,
                                const bfloat16 *__restrict rhs) {
  constexpr int VecLen = 8;
  static_assert(HEAD_DIM % VecLen == 0,
                "optimized Flow attention expects head_dim divisible by 8");

  float sum = 0.0f;
  for (unsigned d = 0; d < HEAD_DIM; d += VecLen)
    chess_prepare_for_pipelining chess_loop_range(4, ) {
      aie::vector<bfloat16, VecLen> qv = aie::load_v<VecLen, aie_dm_resource::a>(lhs + d);
      aie::vector<bfloat16, VecLen> kv = aie::load_v<VecLen, aie_dm_resource::b>(rhs + d);
      aie::accum<accfloat, VecLen> prod = aie::mul(qv, kv);
      sum += aie::reduce_add(prod.to_vector<float>());
    }
  return sum;
}

template <unsigned QRows>
static void attention_chunk_opt(const bfloat16 *__restrict q,
                                const bfloat16 *__restrict k,
                                const bfloat16 *__restrict v,
                                bfloat16 *__restrict out) {
  constexpr int VecLen = 8;
  static_assert(HEAD_DIM % VecLen == 0,
                "optimized Flow attention expects head_dim divisible by 8");
  const float inv_sqrt_d = 1.0f / __builtin_sqrtf((float)HEAD_DIM);

  ::aie::set_rounding(kFlowRoundMode);

  for (unsigned qi = 0; qi < QRows; ++qi) {
    unsigned end = KV_LEN;
    if constexpr (CAUSAL != 0) {
      const unsigned causal_end = QUERY_BASE + qi + 1;
      end = causal_end < end ? causal_end : end;
    }
    unsigned start = 0;
    if constexpr (WINDOW_LEN > 0) {
      start = (end > WINDOW_LEN) ? (end - WINDOW_LEN) : 0;
    }

    float scores[KV_LEN];
    float max_score = -3.402823466e38f;
    const bfloat16 *__restrict q_row = q + qi * HEAD_DIM;
    for (unsigned kk = start; kk < end; ++kk) {
      const float score = dot_bf16_v8(q_row, k + kk * HEAD_DIM) * inv_sqrt_d;
      scores[kk] = score;
      max_score = score > max_score ? score : max_score;
    }

    float denom = 0.0f;
    for (unsigned kk = start; kk < end; ++kk) {
      const float weight = fast_exp_approx(scores[kk] - max_score);
      scores[kk] = weight;
      denom += weight;
    }
    const float inv_denom = denom > 0.0f ? (1.0f / denom) : 0.0f;

    for (unsigned d = 0; d < HEAD_DIM; d += VecLen) {
      aie::accum<accfloat, VecLen> acc = aie::zeros<accfloat, VecLen>();
      for (unsigned kk = start; kk < end; ++kk)
        chess_prepare_for_pipelining chess_loop_range(4, ) {
          const bfloat16 weight = static_cast<bfloat16>(scores[kk] * inv_denom);
          aie::vector<bfloat16, VecLen> w =
              aie::broadcast<bfloat16, VecLen>(weight);
          aie::vector<bfloat16, VecLen> vv =
              aie::load_v<VecLen, aie_dm_resource::c>(v + kk * HEAD_DIM + d);
          acc = mac(acc, w, vv);
        }
      aie::store_v(out + qi * HEAD_DIM + d, acc.to_vector<bfloat16>());
    }
  }
}

extern "C" {

void flowqkv_chunk_bf16_opt(const bfloat16 *__restrict q,
                            const bfloat16 *__restrict k,
                            const bfloat16 *__restrict v,
                            bfloat16 *__restrict out) {
  attention_chunk_opt<Q_CHUNK>(q, k, v, out);
}

void flowkv_decode_bf16_opt(const bfloat16 *__restrict q,
                            const bfloat16 *__restrict k,
                            const bfloat16 *__restrict v,
                            bfloat16 *__restrict out) {
  attention_chunk_opt<1>(q, k, v, out);
}

} // extern "C"
