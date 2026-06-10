// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Optimized FlowQKV/FlowKV attention microkernels. The implementation uses
// chunked online-softmax accumulation, matching the paper equations for m/l/Y
// state while keeping the source-level AIR wrapper compact.

#include <aie_api/aie.hpp>
#include <cstdint>

#ifndef Q_CHUNK
#define Q_CHUNK 4
#endif

#ifndef KV_LEN
#define KV_LEN 32
#endif

#ifndef KV_CHUNK
#define KV_CHUNK 32
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
      aie::vector<bfloat16, VecLen> qv =
          aie::load_v<VecLen, aie_dm_resource::a>(lhs + d);
      aie::vector<bfloat16, VecLen> kv =
          aie::load_v<VecLen, aie_dm_resource::b>(rhs + d);
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
  static_assert(KV_CHUNK > 0, "KV_CHUNK must be positive");
  static_assert(KV_LEN % KV_CHUNK == 0,
                "optimized Flow attention expects kv_len divisible by kv_chunk");
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

    float y[HEAD_DIM];
    for (unsigned d = 0; d < HEAD_DIM; ++d) {
      y[d] = 0.0f;
    }
    float max_score = -3.402823466e38f;
    float denom = 0.0f;

    const bfloat16 *__restrict q_row = q + qi * HEAD_DIM;
    for (unsigned chunk = 0; chunk < KV_LEN; chunk += KV_CHUNK) {
      const unsigned chunk_end = chunk + KV_CHUNK;
      const unsigned active_begin = start > chunk ? start : chunk;
      const unsigned active_end = end < chunk_end ? end : chunk_end;
      if (active_end <= active_begin)
        continue;

      for (unsigned kk = active_begin; kk < active_end; ++kk)
        chess_prepare_for_pipelining chess_loop_range(4, ) {
          const float score = dot_bf16_v8(q_row, k + kk * HEAD_DIM) * inv_sqrt_d;
          const float new_max = score > max_score ? score : max_score;
          const float carry = denom > 0.0f ? fast_exp_approx(max_score - new_max) : 0.0f;
          const float weight = fast_exp_approx(score - new_max);

          const bfloat16 *__restrict v_row = v + kk * HEAD_DIM;
          for (unsigned d = 0; d < HEAD_DIM; ++d) {
            y[d] = y[d] * carry + weight * static_cast<float>(v_row[d]);
          }
          denom = denom * carry + weight;
          max_score = new_max;
        }
    }

    const float inv_denom = denom > 0.0f ? (1.0f / denom) : 0.0f;
    for (unsigned d = 0; d < HEAD_DIM; ++d) {
      out[qi * HEAD_DIM + d] = static_cast<bfloat16>(y[d] * inv_denom);
    }
  }
}


template <unsigned QRows>
static void attention_chunk_qbase_opt(int query_base,
                                      const bfloat16 *__restrict q,
                                      const bfloat16 *__restrict k,
                                      const bfloat16 *__restrict v,
                                      bfloat16 *__restrict out) {
  static_assert(KV_CHUNK > 0, "KV_CHUNK must be positive");
  static_assert(KV_LEN % KV_CHUNK == 0,
                "optimized Flow attention expects kv_len divisible by kv_chunk");
  const float inv_sqrt_d = 1.0f / __builtin_sqrtf((float)HEAD_DIM);

  ::aie::set_rounding(kFlowRoundMode);

  for (unsigned qi = 0; qi < QRows; ++qi) {
    unsigned end = KV_LEN;
    if constexpr (CAUSAL != 0) {
      const unsigned causal_end = static_cast<unsigned>(query_base) + qi + 1;
      end = causal_end < end ? causal_end : end;
    }
    unsigned start = 0;
    if constexpr (WINDOW_LEN > 0) {
      start = (end > WINDOW_LEN) ? (end - WINDOW_LEN) : 0;
    }

    float y[HEAD_DIM];
    for (unsigned d = 0; d < HEAD_DIM; ++d) {
      y[d] = 0.0f;
    }
    float max_score = -3.402823466e38f;
    float denom = 0.0f;

    const bfloat16 *__restrict q_row = q + qi * HEAD_DIM;
    for (unsigned chunk = 0; chunk < KV_LEN; chunk += KV_CHUNK) {
      const unsigned chunk_end = chunk + KV_CHUNK;
      const unsigned active_begin = start > chunk ? start : chunk;
      const unsigned active_end = end < chunk_end ? end : chunk_end;
      if (active_end <= active_begin)
        continue;

      for (unsigned kk = active_begin; kk < active_end; ++kk)
        chess_prepare_for_pipelining chess_loop_range(4, ) {
          const float score = dot_bf16_v8(q_row, k + kk * HEAD_DIM) * inv_sqrt_d;
          const float new_max = score > max_score ? score : max_score;
          const float carry = denom > 0.0f ? fast_exp_approx(max_score - new_max) : 0.0f;
          const float weight = fast_exp_approx(score - new_max);

          const bfloat16 *__restrict v_row = v + kk * HEAD_DIM;
          for (unsigned d = 0; d < HEAD_DIM; ++d) {
            y[d] = y[d] * carry + weight * static_cast<float>(v_row[d]);
          }
          denom = denom * carry + weight;
          max_score = new_max;
        }
    }

    const float inv_denom = denom > 0.0f ? (1.0f / denom) : 0.0f;
    for (unsigned d = 0; d < HEAD_DIM; ++d) {
      out[qi * HEAD_DIM + d] = static_cast<bfloat16>(y[d] * inv_denom);
    }
  }
}

static void flowqkv_scores_impl(int query_base,
                                const bfloat16 *__restrict q,
                                const bfloat16 *__restrict k,
                                bfloat16 *__restrict attn) {
  const float inv_sqrt_d = 1.0f / __builtin_sqrtf((float)HEAD_DIM);

  ::aie::set_rounding(kFlowRoundMode);

  for (unsigned qi = 0; qi < Q_CHUNK; ++qi) {
    bfloat16 *__restrict attn_row = attn + qi * KV_LEN;
    for (unsigned kk = 0; kk < KV_LEN; ++kk) {
      attn_row[kk] = static_cast<bfloat16>(0.0f);
    }

    unsigned end = KV_LEN;
    if constexpr (CAUSAL != 0) {
      const unsigned causal_end = static_cast<unsigned>(query_base) + qi + 1;
      end = causal_end < end ? causal_end : end;
    }
    unsigned start = 0;
    if constexpr (WINDOW_LEN > 0) {
      start = (end > WINDOW_LEN) ? (end - WINDOW_LEN) : 0;
    }
    if (end <= start)
      continue;

    float scores[KV_LEN];
    float max_score = -3.402823466e38f;
    const bfloat16 *__restrict q_row = q + qi * HEAD_DIM;
    for (unsigned kk = start; kk < end; ++kk)
      chess_prepare_for_pipelining chess_loop_range(4, ) {
        const float score = dot_bf16_v8(q_row, k + kk * HEAD_DIM) * inv_sqrt_d;
        scores[kk] = score;
        max_score = score > max_score ? score : max_score;
      }

    float denom = 0.0f;
    for (unsigned kk = start; kk < end; ++kk)
      chess_prepare_for_pipelining chess_loop_range(4, ) {
        const float weight = fast_exp_approx(scores[kk] - max_score);
        scores[kk] = weight;
        denom += weight;
      }

    const float inv_denom = denom > 0.0f ? (1.0f / denom) : 0.0f;
    for (unsigned kk = start; kk < end; ++kk)
      chess_prepare_for_pipelining chess_loop_range(4, ) {
        attn_row[kk] = static_cast<bfloat16>(scores[kk] * inv_denom);
      }
  }
}

static void flowqkv_apply_impl(const bfloat16 *__restrict attn,
                               const bfloat16 *__restrict v,
                               bfloat16 *__restrict out) {
  constexpr int VecLen = 8;
  static_assert(HEAD_DIM % VecLen == 0,
                "optimized FlowQKV apply expects head_dim divisible by 8");

  ::aie::set_rounding(kFlowRoundMode);

  for (unsigned qi = 0; qi < Q_CHUNK; ++qi) {
    const bfloat16 *__restrict attn_row = attn + qi * KV_LEN;
    for (unsigned d = 0; d < HEAD_DIM; d += VecLen) {
      aie::accum<accfloat, VecLen> acc = aie::zeros<accfloat, VecLen>();
      for (unsigned kk = 0; kk < KV_LEN; ++kk)
        chess_prepare_for_pipelining chess_loop_range(4, ) {
          aie::vector<bfloat16, VecLen> w =
              aie::broadcast<bfloat16, VecLen>(attn_row[kk]);
          aie::vector<bfloat16, VecLen> vv =
              aie::load_v<VecLen, aie_dm_resource::c>(v + kk * HEAD_DIM + d);
          acc = mac(acc, w, vv);
        }
      aie::store_v(out + qi * HEAD_DIM + d, acc.to_vector<bfloat16>());
    }
  }
}

static void flowqkv_scores_chunk_impl(int32_t query_base, int32_t chunk_offset,
                                      const bfloat16 *__restrict q,
                                      const bfloat16 *__restrict k,
                                      bfloat16 *__restrict attn_chunk) {
  const float inv_sqrt_d = 1.0f / __builtin_sqrtf((float)HEAD_DIM);

  ::aie::set_rounding(kFlowRoundMode);

  for (unsigned qi = 0; qi < Q_CHUNK; ++qi) {
    bfloat16 *__restrict attn_row = attn_chunk + qi * KV_CHUNK;
    for (unsigned j = 0; j < KV_CHUNK; ++j) {
      attn_row[j] = static_cast<bfloat16>(0.0f);
    }

    unsigned end = KV_LEN;
    if constexpr (CAUSAL != 0) {
      const unsigned causal_end = static_cast<unsigned>(query_base) + qi + 1;
      end = causal_end < end ? causal_end : end;
    }
    unsigned start = 0;
    if constexpr (WINDOW_LEN > 0) {
      start = (end > WINDOW_LEN) ? (end - WINDOW_LEN) : 0;
    }
    if (end <= start)
      continue;

    float scores[KV_LEN];
    float max_score = -3.402823466e38f;
    const bfloat16 *__restrict q_row = q + qi * HEAD_DIM;
    for (unsigned kk = start; kk < end; ++kk)
      chess_prepare_for_pipelining chess_loop_range(4, ) {
        const float score = dot_bf16_v8(q_row, k + kk * HEAD_DIM) * inv_sqrt_d;
        scores[kk] = score;
        max_score = score > max_score ? score : max_score;
      }

    float denom = 0.0f;
    for (unsigned kk = start; kk < end; ++kk)
      chess_prepare_for_pipelining chess_loop_range(4, ) {
        const float weight = fast_exp_approx(scores[kk] - max_score);
        scores[kk] = weight;
        denom += weight;
      }

    const float inv_denom = denom > 0.0f ? (1.0f / denom) : 0.0f;
    const unsigned base = static_cast<unsigned>(chunk_offset);
    for (unsigned j = 0; j < KV_CHUNK; ++j) {
      const unsigned kk = base + j;
      if (kk >= start && kk < end) {
        attn_row[j] = static_cast<bfloat16>(scores[kk] * inv_denom);
      }
    }
  }
}

static void flowqkv_apply_chunk_impl(int32_t chunk_offset,
                                     const bfloat16 *__restrict attn_chunk,
                                     const bfloat16 *__restrict v_chunk,
                                     bfloat16 *__restrict out) {
  constexpr int VecLen = 8;
  static_assert(HEAD_DIM % VecLen == 0,
                "optimized FlowQKV apply expects head_dim divisible by 8");

  ::aie::set_rounding(kFlowRoundMode);

  const unsigned base = static_cast<unsigned>(chunk_offset);
  for (unsigned qi = 0; qi < Q_CHUNK; ++qi) {
    const bfloat16 *__restrict attn_row = attn_chunk + qi * KV_CHUNK;
    for (unsigned d = 0; d < HEAD_DIM; d += VecLen) {
      aie::accum<accfloat, VecLen> acc = aie::zeros<accfloat, VecLen>();
      if (base != 0) {
        aie::vector<bfloat16, VecLen> out_vec =
            aie::load_v<VecLen, aie_dm_resource::a>(out + qi * HEAD_DIM + d);
        acc = aie::accum<accfloat, VecLen>(out_vec);
      }
      for (unsigned j = 0; j < KV_CHUNK; ++j)
        chess_prepare_for_pipelining chess_loop_range(4, ) {
          aie::vector<bfloat16, VecLen> w =
              aie::broadcast<bfloat16, VecLen>(attn_row[j]);
          aie::vector<bfloat16, VecLen> vv = aie::load_v<VecLen, aie_dm_resource::c>(
              v_chunk + j * HEAD_DIM + d);
          acc = mac(acc, w, vv);
        }
      aie::store_v(out + qi * HEAD_DIM + d, acc.to_vector<bfloat16>());
    }
  }
}
static void flowkv_scores_impl(const bfloat16 *__restrict q,
                               const bfloat16 *__restrict k,
                               bfloat16 *__restrict attn) {
  const float inv_sqrt_d = 1.0f / __builtin_sqrtf((float)HEAD_DIM);

  ::aie::set_rounding(kFlowRoundMode);

  unsigned end = KV_LEN;
  if constexpr (CAUSAL != 0) {
    const unsigned causal_end = QUERY_BASE + 1;
    end = causal_end < end ? causal_end : end;
  }
  unsigned start = 0;
  if constexpr (WINDOW_LEN > 0) {
    start = (end > WINDOW_LEN) ? (end - WINDOW_LEN) : 0;
  }

  for (unsigned kk = 0; kk < KV_LEN; ++kk) {
    attn[kk] = static_cast<bfloat16>(0.0f);
  }

  float scores[KV_LEN];
  float max_score = -3.402823466e38f;
  for (unsigned kk = start; kk < end; ++kk)
    chess_prepare_for_pipelining chess_loop_range(4, ) {
      const float score = dot_bf16_v8(q, k + kk * HEAD_DIM) * inv_sqrt_d;
      scores[kk] = score;
      max_score = score > max_score ? score : max_score;
    }

  float denom = 0.0f;
  for (unsigned kk = start; kk < end; ++kk)
    chess_prepare_for_pipelining chess_loop_range(4, ) {
      const float weight = fast_exp_approx(scores[kk] - max_score);
      scores[kk] = weight;
      denom += weight;
    }

  const float inv_denom = denom > 0.0f ? (1.0f / denom) : 0.0f;
  for (unsigned kk = start; kk < end; ++kk)
    chess_prepare_for_pipelining chess_loop_range(4, ) {
      attn[kk] = static_cast<bfloat16>(scores[kk] * inv_denom);
    }
}

static void flowkv_apply_impl(const bfloat16 *__restrict attn,
                              const bfloat16 *__restrict v,
                              bfloat16 *__restrict out) {
  constexpr int VecLen = 8;
  static_assert(HEAD_DIM % VecLen == 0,
                "optimized FlowKV apply expects head_dim divisible by 8");

  ::aie::set_rounding(kFlowRoundMode);

  for (unsigned d = 0; d < HEAD_DIM; d += VecLen) {
    aie::accum<accfloat, VecLen> acc = aie::zeros<accfloat, VecLen>();
    for (unsigned kk = 0; kk < KV_LEN; ++kk)
      chess_prepare_for_pipelining chess_loop_range(4, ) {
        aie::vector<bfloat16, VecLen> w =
            aie::broadcast<bfloat16, VecLen>(attn[kk]);
        aie::vector<bfloat16, VecLen> vv =
            aie::load_v<VecLen, aie_dm_resource::c>(v + kk * HEAD_DIM + d);
        acc = mac(acc, w, vv);
      }
    aie::store_v(out + d, acc.to_vector<bfloat16>());
  }
}

static void flowkv_scores_chunk_impl(int32_t chunk_offset,
                                     const bfloat16 *__restrict q,
                                     const bfloat16 *__restrict k,
                                     bfloat16 *__restrict attn_chunk) {
  const float inv_sqrt_d = 1.0f / __builtin_sqrtf((float)HEAD_DIM);

  ::aie::set_rounding(kFlowRoundMode);

  for (unsigned j = 0; j < KV_CHUNK; ++j) {
    attn_chunk[j] = static_cast<bfloat16>(0.0f);
  }

  unsigned end = KV_LEN;
  if constexpr (CAUSAL != 0) {
    const unsigned causal_end = QUERY_BASE + 1;
    end = causal_end < end ? causal_end : end;
  }
  unsigned start = 0;
  if constexpr (WINDOW_LEN > 0) {
    start = (end > WINDOW_LEN) ? (end - WINDOW_LEN) : 0;
  }
  if (end <= start)
    return;

  float scores[KV_LEN];
  float max_score = -3.402823466e38f;
  for (unsigned kk = start; kk < end; ++kk)
    chess_prepare_for_pipelining chess_loop_range(4, ) {
      const float score = dot_bf16_v8(q, k + kk * HEAD_DIM) * inv_sqrt_d;
      scores[kk] = score;
      max_score = score > max_score ? score : max_score;
    }

  float denom = 0.0f;
  for (unsigned kk = start; kk < end; ++kk)
    chess_prepare_for_pipelining chess_loop_range(4, ) {
      const float weight = fast_exp_approx(scores[kk] - max_score);
      scores[kk] = weight;
      denom += weight;
    }

  const float inv_denom = denom > 0.0f ? (1.0f / denom) : 0.0f;
  const unsigned base = static_cast<unsigned>(chunk_offset);
  for (unsigned j = 0; j < KV_CHUNK; ++j) {
    const unsigned kk = base + j;
    if (kk >= start && kk < end) {
      attn_chunk[j] = static_cast<bfloat16>(scores[kk] * inv_denom);
    }
  }
}

static void flowkv_apply_chunk_impl(int32_t chunk_offset,
                                    const bfloat16 *__restrict attn_chunk,
                                    const bfloat16 *__restrict v,
                                    bfloat16 *__restrict out) {
  constexpr int VecLen = 8;
  static_assert(HEAD_DIM % VecLen == 0,
                "optimized FlowKV apply expects head_dim divisible by 8");

  ::aie::set_rounding(kFlowRoundMode);

  const unsigned base = static_cast<unsigned>(chunk_offset);
  for (unsigned d = 0; d < HEAD_DIM; d += VecLen) {
    aie::accum<accfloat, VecLen> acc = aie::zeros<accfloat, VecLen>();
    if (base != 0) {
      aie::vector<bfloat16, VecLen> out_vec =
          aie::load_v<VecLen, aie_dm_resource::a>(out + d);
      acc = aie::accum<accfloat, VecLen>(out_vec);
    }
    for (unsigned j = 0; j < KV_CHUNK; ++j)
      chess_prepare_for_pipelining chess_loop_range(4, ) {
        aie::vector<bfloat16, VecLen> w =
            aie::broadcast<bfloat16, VecLen>(attn_chunk[j]);
        aie::vector<bfloat16, VecLen> vv = aie::load_v<VecLen, aie_dm_resource::c>(
            v + (base + j) * HEAD_DIM + d);
        acc = mac(acc, w, vv);
      }
    aie::store_v(out + d, acc.to_vector<bfloat16>());
  }
}

extern "C" {

void flowqkv_chunk_bf16_opt(const bfloat16 *__restrict q,
                            const bfloat16 *__restrict k,
                            const bfloat16 *__restrict v,
                            bfloat16 *__restrict out) {
  attention_chunk_opt<Q_CHUNK>(q, k, v, out);
}

void flowqkv_chunk_qbase_bf16_opt(int query_base,
                                  const bfloat16 *__restrict q,
                                  const bfloat16 *__restrict k,
                                  const bfloat16 *__restrict v,
                                  bfloat16 *__restrict out) {
  attention_chunk_qbase_opt<Q_CHUNK>(query_base, q, k, v, out);
}

void flowqkv_scores_bf16_opt(int query_base,
                             const bfloat16 *__restrict q,
                             const bfloat16 *__restrict k,
                             bfloat16 *__restrict attn) {
  flowqkv_scores_impl(query_base, q, k, attn);
}

void flowqkv_apply_bf16_opt(const bfloat16 *__restrict attn,
                            const bfloat16 *__restrict v,
                            bfloat16 *__restrict out) {
  flowqkv_apply_impl(attn, v, out);
}

void flowqkv_scores_chunk_bf16_opt(int32_t query_base, int32_t chunk_offset,
                                   const bfloat16 *__restrict q,
                                   const bfloat16 *__restrict k,
                                   bfloat16 *__restrict attn_chunk) {
  flowqkv_scores_chunk_impl(query_base, chunk_offset, q, k, attn_chunk);
}

void flowqkv_apply_chunk_bf16_opt(int32_t chunk_offset,
                                  const bfloat16 *__restrict attn_chunk,
                                  const bfloat16 *__restrict v_chunk,
                                  bfloat16 *__restrict out) {
  flowqkv_apply_chunk_impl(chunk_offset, attn_chunk, v_chunk, out);
}

void flowkv_decode_bf16_opt(const bfloat16 *__restrict q,
                            const bfloat16 *__restrict k,
                            const bfloat16 *__restrict v,
                            bfloat16 *__restrict out) {
  attention_chunk_opt<1>(q, k, v, out);
}

void flowkv_scores_bf16_opt(const bfloat16 *__restrict q,
                            const bfloat16 *__restrict k,
                            bfloat16 *__restrict attn) {
  flowkv_scores_impl(q, k, attn);
}

void flowkv_apply_bf16_opt(const bfloat16 *__restrict attn,
                           const bfloat16 *__restrict v,
                           bfloat16 *__restrict out) {
  flowkv_apply_impl(attn, v, out);
}

void flowkv_scores_chunk_bf16_opt(int32_t chunk_offset,
                                  const bfloat16 *__restrict q,
                                  const bfloat16 *__restrict k,
                                  bfloat16 *__restrict attn_chunk) {
  flowkv_scores_chunk_impl(chunk_offset, q, k, attn_chunk);
}

void flowkv_apply_chunk_bf16_opt(int32_t chunk_offset,
                                 const bfloat16 *__restrict attn_chunk,
                                 const bfloat16 *__restrict v,
                                 bfloat16 *__restrict out) {
  flowkv_apply_chunk_impl(chunk_offset, attn_chunk, v, out);
}

} // extern "C"
