//===- layer_norm.cc --------------------------------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// External (Peano-compiled) LayerNorm kernels for AIE2P.
//
// layer_norm.py in this directory builds the same normalization by direct
// codegen from the vector dialect, one row per launch. These entry points
// exist for the multi-row case: they normalize `rows_to_process` consecutive
// rows of an L1 tile in a single call, which is what a transformer block needs
// when a whole activation tile is resident.
//
// They are NOT bit-equivalent to layer_norm.py -- that builder keeps its
// statistics in bf16, these kernels keep them in f32 (below). Do not use one
// as the numerical oracle for the other.
//
// CONTRACT
//   - Element type is bf16; statistics accumulate in f32, and the variance is
//     TWO-PASS: mean first, then E[(x - mean)^2]. The one-pass
//     E[x^2] - E[x]^2 form this file used to ship cancels catastrophically on
//     a row whose mean is large next to its spread -- at mean/sigma ~ 30 it
//     loses the variance entirely, clamps at zero and normalizes by
//     1/sqrt(eps) -- and a bf16 sum accumulator loses the addends' low bits
//     once the running sum outgrows them, so the mean itself was wrong on
//     exactly those rows. Offset rows are valid inputs under every builder
//     that links this object; the statistics have to be exact enough for
//     them, not just for zero-mean activations.
//   - The normalization applies (x - mean) * inv_std in f32 and rounds ONCE,
//     at the store. The per-element error against an f32 reference is a
//     single bf16 rounding plus ~2^-9 relative on inv_std (the squared
//     deviations round to bf16 before accumulating; the deviations themselves
//     are exact, since every bf16 value is exact in f32).
//   - `cols` must be a multiple of the vector width (16). There is no scalar
//     tail: a `cols` that is not a multiple of 16 silently drops the
//     remainder, because vector_chunks truncates.
//   - Rows are contiguous and `cols` elements apart. A tile with padding
//     between rows must pass the padded stride as `cols`, which then also
//     normalizes over the padding.
//   - No gamma/beta. These are the unweighted forms; the weighted variant
//     lives in weighted_rms_norm/ and in the transformer_layer kernels.
//
// The two-pass variance is non-negative by construction, but the clamp before
// invsqrt stays: aie::invsqrt of a negative operand returns NaN and would
// poison the whole row, and the clamp is one compare against that class of
// surprise.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

namespace {

constexpr float kEpsilon = 1e-5f;

// Normalize `rows_to_process` rows of `cols` elements each, in place across
// separate input/output buffers. Statistics in f32, variance two-pass; see the
// file header for why neither is optional.
template <typename T, int N>
void layer_norm_rows_impl(const T *__restrict input, T *__restrict output,
                          int32_t cols, int32_t rows_to_process) {
  event0();

  const int vector_chunks = cols / N;
  for (int row = 0; row < rows_to_process; row++) {
    int input_index = row * cols;

    // Pass 1: the row sum, widened to f32 lane-by-lane before accumulating.
    ::aie::vector<float, N> sum_acc = ::aie::zeros<float, N>();
    for (int i = 0; i < vector_chunks; i++) {
      ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input + input_index);
      ::aie::accum<accfloat, N> a_acc;
      a_acc.from_vector(reg_a);
      sum_acc = ::aie::add(sum_acc, a_acc.template to_vector<float>());
      input_index += N;
    }
    input_index -= cols;
    const float mean = ::aie::reduce_add(sum_acc) / float(cols);

    // Pass 2: E[(x - mean)^2]. The deviation is exact in f32 and rounds to
    // bf16 only for the squaring, so the variance carries ~2^-9 relative
    // error at any mean -- where the one-pass form's error grows with
    // (mean/sigma)^2 and swallows the variance whole on offset rows.
    ::aie::vector<float, N> sum_sq_acc = ::aie::zeros<float, N>();
    for (int i = 0; i < vector_chunks; i++) {
      ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input + input_index);
      ::aie::accum<accfloat, N> a_acc;
      a_acc.from_vector(reg_a);
      ::aie::accum<accfloat, N> diff_acc = ::aie::sub(a_acc, mean);
      ::aie::vector<T, N> diff_v = diff_acc.template to_vector<T>();
      ::aie::vector<float, N> sq_v = ::aie::mul(diff_v, diff_v);
      sum_sq_acc = ::aie::add(sum_sq_acc, sq_v);
      input_index += N;
    }
    input_index -= cols;
    float variance = ::aie::reduce_add(sum_sq_acc) / float(cols);
    // Non-negative by construction; kept because invsqrt of a negative
    // operand returns NaN and would poison the whole row.
    if (variance < 0.0f) {
      variance = 0.0f;
    }
    const float inv_std = ::aie::invsqrt(variance + kEpsilon);

    // Pass 3: (x - mean) * inv_std in f32, one rounding to bf16 at the store.
    for (int i = 0; i < vector_chunks; i++) {
      ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input + input_index);
      ::aie::accum<accfloat, N> a_acc;
      a_acc.from_vector(reg_a);
      ::aie::accum<accfloat, N> diff_acc = ::aie::sub(a_acc, mean);
      ::aie::accum<accfloat, N> norm_acc =
          ::aie::mul(diff_acc.template to_vector<float>(), inv_std);
      ::aie::store_v(output + input_index, norm_acc.template to_vector<T>());
      input_index += N;
    }
  }

  event1();
}

// Residual add fused into the same passes: statistics and normalization both
// run over (input1 + input2), recomputed per pass rather than materialized.
// The elementwise sum itself is bf16 -- the same single rounding a caller that
// materializes the sum (norm_tail's stage_add) pays -- and everything after it
// follows layer_norm_rows_impl.
template <typename T, int N>
void add_layer_norm_rows_impl(const T *__restrict input1,
                              const T *__restrict input2, T *__restrict output,
                              int32_t cols, int32_t rows_to_process) {
  event0();

  const int vector_chunks = cols / N;
  for (int row = 0; row < rows_to_process; row++) {
    int input_index = row * cols;

    // Pass 1: the row sum of (input1 + input2), widened to f32.
    ::aie::vector<float, N> sum_acc = ::aie::zeros<float, N>();
    for (int i = 0; i < vector_chunks; i++) {
      ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input1 + input_index);
      ::aie::vector<T, N> reg_b = ::aie::load_v<N>(input2 + input_index);
      ::aie::vector<T, N> reg_sum = ::aie::add(reg_a, reg_b);
      ::aie::accum<accfloat, N> s_acc;
      s_acc.from_vector(reg_sum);
      sum_acc = ::aie::add(sum_acc, s_acc.template to_vector<float>());
      input_index += N;
    }
    input_index -= cols;
    const float mean = ::aie::reduce_add(sum_acc) / float(cols);

    // Pass 2: E[(sum - mean)^2]; see layer_norm_rows_impl.
    ::aie::vector<float, N> sum_sq_acc = ::aie::zeros<float, N>();
    for (int i = 0; i < vector_chunks; i++) {
      ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input1 + input_index);
      ::aie::vector<T, N> reg_b = ::aie::load_v<N>(input2 + input_index);
      ::aie::vector<T, N> reg_sum = ::aie::add(reg_a, reg_b);
      ::aie::accum<accfloat, N> s_acc;
      s_acc.from_vector(reg_sum);
      ::aie::accum<accfloat, N> diff_acc = ::aie::sub(s_acc, mean);
      ::aie::vector<T, N> diff_v = diff_acc.template to_vector<T>();
      ::aie::vector<float, N> sq_v = ::aie::mul(diff_v, diff_v);
      sum_sq_acc = ::aie::add(sum_sq_acc, sq_v);
      input_index += N;
    }
    input_index -= cols;
    float variance = ::aie::reduce_add(sum_sq_acc) / float(cols);
    // Same clamp as layer_norm_rows_impl, for the same NaN reason.
    if (variance < 0.0f) {
      variance = 0.0f;
    }
    const float inv_std = ::aie::invsqrt(variance + kEpsilon);

    // Pass 3: (sum - mean) * inv_std in f32, one rounding at the store.
    for (int i = 0; i < vector_chunks; i++) {
      ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input1 + input_index);
      ::aie::vector<T, N> reg_b = ::aie::load_v<N>(input2 + input_index);
      ::aie::vector<T, N> reg_sum = ::aie::add(reg_a, reg_b);
      ::aie::accum<accfloat, N> s_acc;
      s_acc.from_vector(reg_sum);
      ::aie::accum<accfloat, N> diff_acc = ::aie::sub(s_acc, mean);
      ::aie::accum<accfloat, N> norm_acc =
          ::aie::mul(diff_acc.template to_vector<float>(), inv_std);
      ::aie::store_v(output + input_index, norm_acc.template to_vector<T>());
      input_index += N;
    }
  }

  event1();
}

} // namespace

#ifndef LN_VEC_LEN
#define LN_VEC_LEN 16
#endif

extern "C" {

// Single row -- the same function the direct-codegen builder in layer_norm.py
// computes, with f32 statistics where that builder keeps bf16 ones, so not the
// same bits. See the note at the top of this file.
void layer_norm(bfloat16 *input, bfloat16 *output, int32_t cols) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  layer_norm_rows_impl<bfloat16, LN_VEC_LEN>(input, output, cols, 1);
}

void layer_norm_rows(bfloat16 *input, bfloat16 *output, int32_t cols,
                     int32_t rows_to_process) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  layer_norm_rows_impl<bfloat16, LN_VEC_LEN>(input, output, cols,
                                             rows_to_process);
}

void add_layer_norm_rows(bfloat16 *input1, bfloat16 *input2, bfloat16 *output,
                         int32_t cols, int32_t rows_to_process) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  add_layer_norm_rows_impl<bfloat16, LN_VEC_LEN>(input1, input2, output, cols,
                                                 rows_to_process);
}

} // extern "C"
