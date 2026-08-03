//===- softmax.cc -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#define __AIENGINE__ 2
#define NOCPP
#define __AIEARCH__ 20

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define REL_WRITE 0
#define REL_READ 1

#include <aie_api/aie.hpp>

#include "lut_based_ops.h"
#include "zero.cc"

template <unsigned VecLen, typename VecIteratorIn, typename VecIteratorOut>
float exp_bf16(int num_elems, VecIteratorIn &in, VecIteratorOut &out) {
  bfloat16 __aie_dm_resource_a *ilut_ab =
      (bfloat16 __aie_dm_resource_a *)softmax_ilut_ab;
  bfloat16 __aie_dm_resource_b *ilut_cd =
      (bfloat16 __aie_dm_resource_b *)softmax_ilut_cd;
  bfloat16 __aie_dm_resource_a *flut_ab =
      (bfloat16 __aie_dm_resource_a *)softmax_flut_ab;
  bfloat16 __aie_dm_resource_b *flut_cd =
      (bfloat16 __aie_dm_resource_b *)softmax_flut_cd;
  using lut_type = aie::lut<4, bfloat16, bfloat16>;
  const int LUT_elems = 256;
  const int step_i = 8;
  const int step_f = 0;

  constexpr int SM_SCALE_FAC =
      8; // Use 8-bit fractional part for LUTs when converting from bfloat16 to
         // int, adjust any input scale factor using this.

  const int elem_iters =
      num_elems / VecLen +
      (num_elems % VecLen != 0); // number of iterations need to be performed
  aie::vector<bfloat16, VecLen> I_val_vec, F_val_vec, res0, input_bf16;
  aie::accum<accfloat, VecLen> exp_val_accum;
  aie::accum<accfloat, VecLen> exp_val_accum_shift;
  exp_val_accum = aie::zeros<accfloat, VecLen>();
  // Maximum value computation
  bfloat16 max_value;
  aie::vector<bfloat16, VecLen> max_bfloat16;
  aie::accum<accfloat, VecLen> acc0, acc1, acc_res;
  aie::vector<int16, VecLen> input;
  aie::vector<int16, 2 * VecLen> input0;

  lut_type lut_i(LUT_elems, ilut_ab, ilut_cd);
  lut_type lut_f(LUT_elems, flut_ab, flut_cd);
  aie::parallel_lookup<uint16, lut_type, aie::lut_oor_policy::truncate>
      lookup_i(lut_i, step_i);
  aie::parallel_lookup<uint16, lut_type, aie::lut_oor_policy::truncate>
      lookup_f(lut_f, step_f);
  aie::accum<accfloat, VecLen> exp_val;

  // if constexpr(maxsub_en == 1){
  auto input_max = in;
  uint16 neg_infinity = (uint16)0xff80;
  bfloat16 *bf_neg_infinity = (bfloat16 *)&neg_infinity;
  aie::vector<bfloat16, VecLen> max_vec =
      aie::broadcast<bfloat16, VecLen>((*bf_neg_infinity));
  aie::vector<bfloat16, VecLen> temp;
  for (int i = 0; i < elem_iters; i++)
    chess_prepare_for_pipelining chess_loop_range(4, ) {
      temp = aie::load_v<VecLen>(input_max);
      max_vec = aie::max(max_vec, temp);
    }
  max_value = aie::reduce_max(max_vec);
  max_bfloat16 = aie::broadcast<bfloat16, VecLen>(max_value);

  for (int i = 0; i < elem_iters; i++)
    chess_prepare_for_pipelining chess_loop_range(4, ) {
      aie::vector<bfloat16, VecLen> input_org = aie::load_v<VecLen>(in);
      in += VecLen;
      acc0.from_vector(input_org, 0);
      acc1.from_vector(max_bfloat16, 0);
      acc_res = sub(acc0, acc1);
      input_bf16 = to_v16bfloat16(acc_res);
      input0 = v32int16(bfloat16_to_int(input_bf16, SM_SCALE_FAC));
#ifndef SM_USE_MSB
      input = filter_even(input0);
#else
      input = filter_odd(input0);
#endif

      I_val_vec = lookup_i.fetch(input.template cast_to<uint16>());
      F_val_vec = lookup_f.fetch(input.template cast_to<uint16>());
      exp_val = aie::mul(I_val_vec, F_val_vec);
      exp_val_accum = add(exp_val_accum, exp_val);
      aie::store_v(out, exp_val.template to_vector<bfloat16>());
      out += VecLen;
    }
  // Variant not using emulated FP32 for the mul reduce, off by +/- 1 in final
  // result and 10 cycles slower
  aie::vector<float, VecLen> reduce = exp_val_accum.template to_vector<float>();
  float res = aie::reduce_add(reduce);
  return res;
}

bfloat16 __attribute__((always_inline)) compute_inv_as_bf16(float x) {
  unsigned int *B_x;
  unsigned int exp_mask = 0x7F800000;
  unsigned int mantissa_mask = 0x007FFFFF;
  unsigned int mantissa_Q = 0x00008000;
  unsigned char exponent, mantissa;
  unsigned inv_exponent;
  unsigned short inv_x_val;
  unsigned int B_Q;
  bfloat16 *inv_x;
  B_x = (unsigned int *)&x;
  B_Q = *B_x + mantissa_Q;
  exponent = (B_Q & exp_mask) >> 23;
  mantissa = (B_Q & mantissa_mask) >> 16;
  inv_exponent = (mantissa == 0) + (253 - exponent);
  inv_x_val = (inv_exponent << 7) + m_inv_lut[mantissa];
  inv_x = (bfloat16 *)&inv_x_val;
  return *inv_x;
}

#ifdef SOFTMAX_STREAMING

// ---------------------------------------------------------------------------
// Two-pass streaming softmax (opt-in, -DSOFTMAX_STREAMING).
//
// softmax_bf16 above is single-shot: one row, LUT-based exp, normalized in
// place. The streaming family below instead splits softmax across repeated
// calls over successive K blocks, carrying running max/sum state in a
// caller-owned `scale_buffer` -- the FlashAttention online-softmax recurrence,
// exposed as separate device entry points so a host-side K loop can drive it.
//
// SCALE BUFFER LAYOUT -- this is the footgun. `scale_buffer` is one bf16 array
// of 4 * num_rows elements, read as four bands of num_rows:
//
//   band 0  [0*num_rows + row]  m_prev   running row max, previous block
//   band 1  [1*num_rows + row]  m_cur    running row max, including this block
//   band 2  [2*num_rows + row]  l        running denominator sum
//   band 3  [3*num_rows + row]  scratch  per-row exp sum on the way IN to
//                                        partial_softmax_rows_bf16; on the way
//                                        OUT, the rescale factor
//                                        exp2(m_prev - m_cur) that the caller
//                                        must apply to the running O
//                                        accumulator before adding this
//                                        block's PV product
//
// Band 3 changing meaning inside a single call is deliberate (it saves a
// buffer) and is the easiest thing to get wrong. Call order per K block is
// fixed: init_softmax_scale_buffer once before the K loop, then
// partial_softmax_rows_bf16 per block, then normalize_softmax_rows_bf16 once
// after the loop. Calling normalize before the loop finishes divides by a
// partial denominator.
//
// All exponentials are base-2 (log2e is folded into the scale), so this family
// does NOT share the LUT tables that softmax_bf16 uses.
// ---------------------------------------------------------------------------

#ifndef SM_VEC_LEN
#define SM_VEC_LEN 64
#endif

#ifndef SM_LOG2E
#define SM_LOG2E 1.4453125f // bf16-representable 1.44269504089
#endif

// One K block of one row: rescale by `scale`, track the running max in
// scale_buffer bands 0/1, write exp2(scaled - max) to `output` and this block's
// exp sum to band 3. Does not normalize.
static void partial_softmax_alias_bf16(
    const bfloat16 *__restrict input_vector, bfloat16 *__restrict output_vector,
    bfloat16 *__restrict scale_buffer, const int32_t vector_size,
    const int32_t row_idx, const int32_t num_rows, const bfloat16 scale) {
  event0();
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  auto it_max_in = aie::cbegin_restrict_vector<SM_VEC_LEN>(input_vector);
  auto it_exp_in = aie::cbegin_restrict_vector<SM_VEC_LEN>(input_vector);
  auto it_exp_out = aie::begin_restrict_vector<SM_VEC_LEN>(output_vector);

  aie::vector<bfloat16, SM_VEC_LEN> exp_val, input_bf16, scale_vec, max_val_vec;
  aie::accum<accfloat, SM_VEC_LEN> exp_val_accum, scaled_accum, exp_in_accum;

  float max_val = std::numeric_limits<float>::lowest();
  const int elem_iters = vector_size / SM_VEC_LEN;

  exp_val_accum = aie::zeros<accfloat, SM_VEC_LEN>();
  scale_vec = aie::broadcast<bfloat16, SM_VEC_LEN>(scale);

  for (int i = 0; i < elem_iters; i++) {
    input_bf16 = *it_max_in++;
    scaled_accum = aie::mul(input_bf16, scale_vec);
    float running_max = aie::reduce_max(scaled_accum.to_vector<bfloat16>());
    if (running_max > max_val) {
      max_val = running_max;
    }
  }

  // Band 1 becomes max(previous running max, this block's max); a block whose
  // max does not beat the running one reuses the running value so the
  // exponentials stay on the same reference.
  if (max_val > (float)scale_buffer[row_idx]) {
    scale_buffer[num_rows + row_idx] = (bfloat16)max_val;
  } else {
    scale_buffer[num_rows + row_idx] = scale_buffer[row_idx];
    max_val = (float)scale_buffer[row_idx];
  }

  max_val_vec = aie::broadcast<bfloat16, SM_VEC_LEN>((bfloat16)max_val);

  for (int i = 0; i < elem_iters; i++) {
    input_bf16 = *it_exp_in++;
    scaled_accum = aie::mul(input_bf16, scale_vec);
    exp_in_accum = aie::sub(scaled_accum, max_val_vec);
    exp_val = aie::exp2<bfloat16>(exp_in_accum.to_vector<float>());
    exp_val_accum = aie::add(exp_val_accum, exp_val);
    *it_exp_out++ = exp_val;
  }

  aie::vector<float, SM_VEC_LEN> reduce = exp_val_accum.to_vector<float>();
  scale_buffer[3 * num_rows + row_idx] = (bfloat16)aie::reduce_add(reduce);

  event1();
}

#endif // SOFTMAX_STREAMING

extern "C" {

#ifdef SOFTMAX_STREAMING

// Reset the running softmax state: max to -inf, everything else to zero. Must
// be called once before the first partial_softmax_rows_bf16 of a K loop.
void init_softmax_scale_buffer(bfloat16 *scale_buffer, const int32_t num_rows) {
  for (int32_t row = 0; row < num_rows; ++row) {
    scale_buffer[row] = std::numeric_limits<bfloat16>::lowest();
    scale_buffer[num_rows + row] = (bfloat16)0.0f;
    scale_buffer[2 * num_rows + row] = (bfloat16)0.0f;
    scale_buffer[3 * num_rows + row] = (bfloat16)0.0f;
  }
}

// Straight bf16 copy. Exists so the host can snapshot or relocate the scale
// buffer between launches without a round trip through L3.
void copy_softmax_scale_bf16(const bfloat16 *__restrict input,
                             bfloat16 *__restrict output,
                             const int32_t num_elements) {
  for (int32_t idx = 0; idx < num_elements; ++idx) {
    output[idx] = input[idx];
  }
}

// One K block of `num_rows` rows. Writes exp2(scaled - m_cur) to `output`,
// folds this block into the running denominator (band 2), and leaves the O
// rescale factor exp2(m_prev - m_cur) in band 3.
void partial_softmax_rows_bf16(const bfloat16 *__restrict input,
                               bfloat16 *__restrict output,
                               bfloat16 *__restrict scale_buffer,
                               const int32_t row_width,
                               const int32_t num_rows) {
  for (int32_t row = 0; row < num_rows; ++row) {
    partial_softmax_alias_bf16(input + row * row_width,
                               output + row * row_width, scale_buffer,
                               row_width, row, num_rows, (bfloat16)SM_LOG2E);
  }

  for (int32_t row = 0; row < num_rows; ++row) {
    const float m_prev = (float)scale_buffer[row];
    const float m_cur = (float)scale_buffer[num_rows + row];
    const float l_prev = (float)scale_buffer[2 * num_rows + row];
    const float block_sum = (float)scale_buffer[3 * num_rows + row];
    // reduce_max over a broadcast vector is a scalar exp2 -- aie::exp2 has no
    // scalar overload, so the value is broadcast, exponentiated and reduced.
    const bfloat16 max_diff_exp = aie::reduce_max(
        aie::exp2<bfloat16>(aie::broadcast<float, SM_VEC_LEN>(m_prev - m_cur)));
    scale_buffer[3 * num_rows + row] = max_diff_exp;
    scale_buffer[2 * num_rows + row] =
        (bfloat16)((float)max_diff_exp * l_prev + block_sum);
    scale_buffer[row] = scale_buffer[num_rows + row];
  }
}

// Divide each row by its accumulated denominator (band 2). Call once, after
// the last partial_softmax_rows_bf16 of the K loop.
void normalize_softmax_rows_bf16(const bfloat16 *__restrict input,
                                 const bfloat16 *__restrict scale_buffer,
                                 bfloat16 *__restrict output,
                                 const int32_t row_width,
                                 const int32_t num_rows) {
  for (int32_t row = 0; row < num_rows; ++row) {
    const bfloat16 inv_sum =
        (bfloat16)aie::inv((float)scale_buffer[2 * num_rows + row]);
    auto it_in =
        aie::cbegin_restrict_vector<SM_VEC_LEN>(input + row * row_width);
    auto it_out =
        aie::begin_restrict_vector<SM_VEC_LEN>(output + row * row_width);
    const int32_t elem_iters = row_width / SM_VEC_LEN;
    for (int32_t i = 0; i < elem_iters; ++i) {
      aie::vector<bfloat16, SM_VEC_LEN> in_vec = *it_in++;
      auto out_acc =
          aie::mul(in_vec, aie::broadcast<bfloat16, SM_VEC_LEN>(inv_sum));
      *it_out++ = out_acc.to_vector<bfloat16>();
    }
  }
}

#endif // SOFTMAX_STREAMING

void softmax_bf16(const bfloat16 *__restrict in, const int pos,
                  bfloat16 *__restrict out) {
  const bfloat16 *__restrict pIn = in;
  bfloat16 *__restrict pExp =
      out; // Reusing output buffer to buffer intermediate exp results.
  bfloat16 *__restrict pOut = out;
  zero_vectorized<bfloat16, 1, 256, 16>(out);
  float accum_exp_val = exp_bf16<16>(pos + 1, pIn, pExp);
  bfloat16 accum_inv = compute_inv_as_bf16(accum_exp_val);
  int num_elems = pos + 1;
  for (unsigned i = 0; i < num_elems / 16 + (num_elems % 16 != 0); i++)
    chess_prepare_for_pipelining chess_loop_range(4, ) {
      aie::vector<bfloat16, 16> in_elems = aie::load_v<16>(pOut);
      aie::accum<accfloat, 16> out_vals = aie::mul(in_elems, accum_inv);
      aie::store_v(pOut, out_vals.template to_vector<bfloat16>());
      pOut += 16;
    }
}

} // extern "C"
