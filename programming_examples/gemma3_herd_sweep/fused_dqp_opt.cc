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

#ifndef FUSED_DQP_COL_BLOCKS
#define FUSED_DQP_COL_BLOCKS 1
#endif

#ifndef FUSED_DQP_COL_CHUNK
#define FUSED_DQP_COL_CHUNK 32
#endif

#ifndef FUSED_DQP_PIPE_ROW_CHUNK
#define FUSED_DQP_PIPE_ROW_CHUNK 16
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

static void fused_dqp_dequant_tile_impl(
    int32_t block_index, int32_t row_offset, int32_t col_offset,
    const uint8_t *__restrict packed_all, bfloat16 *__restrict deq_tile) {
  constexpr int VecLen = 8;
  constexpr unsigned PackedElems = Q4NX_ROWS * Q4NX_COLS / 2;
  constexpr unsigned ParamElems = 2 * Q4NX_COLS;
  constexpr unsigned BlockBytes = PackedElems + ParamElems * sizeof(bfloat16);
  static_assert(Q4NX_COLS % VecLen == 0,
                "optimized FusedDQP expects column count divisible by 8");
  static_assert(FUSED_DQP_COL_CHUNK % VecLen == 0,
                "optimized FusedDQP column chunk must be divisible by 8");
  static_assert(Q4NX_ROWS % FUSED_DQP_PIPE_ROW_CHUNK == 0,
                "optimized FusedDQP row chunk must divide row count");

  ::aie::set_rounding(kDqpRoundMode);

  const uint8_t *__restrict packed =
      packed_all + static_cast<unsigned>(block_index) * BlockBytes;
  const bfloat16 *__restrict params =
      reinterpret_cast<const bfloat16 *>(packed + PackedElems);
  const bfloat16 *__restrict scale = params;
  const bfloat16 *__restrict min_offset = params + Q4NX_COLS;
  const unsigned row_base = static_cast<unsigned>(row_offset);
  const unsigned col_base = static_cast<unsigned>(col_offset);

  for (unsigned r = 0; r < FUSED_DQP_PIPE_ROW_CHUNK; ++r) {
    const unsigned row = row_base + r;
    for (unsigned c = 0; c < FUSED_DQP_COL_CHUNK; c += VecLen)
      chess_prepare_for_pipelining chess_loop_range(4, ) {
        const unsigned col = col_base + c;
        const unsigned idx = row * Q4NX_COLS + col;
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
        aie::vector<bfloat16, VecLen> s = aie::load_v<VecLen>(scale + col);
        aie::vector<bfloat16, VecLen> m = aie::load_v<VecLen>(min_offset + col);
        aie::accum<accfloat, VecLen> w = aie::mul(q, s);
        w = aie::add(w, m);
        aie::store_v(deq_tile + r * FUSED_DQP_COL_CHUNK + c,
                     w.to_vector<bfloat16>());
      }
  }
}

static void fused_dqp_project_tile_impl(
    int32_t row_offset, const bfloat16 *__restrict deq_tile,
    const bfloat16 *__restrict activation_chunk, bfloat16 *__restrict out) {
  constexpr int VecLen = 8;
  static_assert(FUSED_DQP_COL_CHUNK % VecLen == 0,
                "optimized FusedDQP column chunk must be divisible by 8");
  static_assert(Q4NX_ROWS % FUSED_DQP_PIPE_ROW_CHUNK == 0,
                "optimized FusedDQP row chunk must divide row count");

  ::aie::set_rounding(kDqpRoundMode);

  const unsigned row_base = static_cast<unsigned>(row_offset);
  for (unsigned r = 0; r < FUSED_DQP_PIPE_ROW_CHUNK; ++r) {
    float acc = static_cast<float>(out[row_base + r]);
    for (unsigned col = 0; col < FUSED_DQP_COL_CHUNK; col += VecLen)
      chess_prepare_for_pipelining chess_loop_range(4, ) {
        aie::vector<bfloat16, VecLen> w = aie::load_v<VecLen>(
            deq_tile + r * FUSED_DQP_COL_CHUNK + col);
        aie::vector<bfloat16, VecLen> a =
            aie::load_v<VecLen>(activation_chunk + col);
        aie::accum<accfloat, VecLen> prod = aie::mul(w, a);
        acc += aie::reduce_add(prod.to_vector<float>());
      }
    out[row_base + r] = static_cast<bfloat16>(acc);
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

void fused_dqp_dequant_tile_opt(int32_t block_index, int32_t row_offset,
                                int32_t col_offset,
                                const uint8_t *__restrict packed_all,
                                bfloat16 *__restrict deq_tile) {
  fused_dqp_dequant_tile_impl(block_index, row_offset, col_offset, packed_all,
                              deq_tile);
}

void fused_dqp_project_tile_opt(int32_t row_offset,
                                const bfloat16 *__restrict deq_tile,
                                const bfloat16 *__restrict activation_chunk,
                                bfloat16 *__restrict out) {
  fused_dqp_project_tile_impl(row_offset, deq_tile, activation_chunk, out);
}

void fused_dqp_accum_block_opt(const uint8_t *__restrict packed,
                               const bfloat16 *__restrict scale,
                               const bfloat16 *__restrict min_offset,
                               const bfloat16 *__restrict activation,
                               bfloat16 *__restrict out) {
  fused_dqp_impl<true>(packed, scale, min_offset, activation, out);
}

} // extern "C"
