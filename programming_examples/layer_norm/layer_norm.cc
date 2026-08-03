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
// CONTRACT
//   - Element type is bf16; statistics accumulate in the same vector type,
//     with the sum-of-squares in f32. This matches the AIR bf16 kernel
//     standard and is NOT the f32-statistics variant.
//   - `cols` must be a multiple of the vector width (16). There is no scalar
//     tail: a `cols` that is not a multiple of 16 silently drops the
//     remainder, because vector_chunks truncates.
//   - Rows are contiguous and `cols` elements apart. A tile with padding
//     between rows must pass the padded stride as `cols`, which then also
//     normalizes over the padding.
//   - No gamma/beta. These are the unweighted forms; the weighted variant
//     lives in weighted_rms_norm/ and in the transformer_layer kernels.
//
// FOOTGUN: variance is computed as E[x^2] - E[x]^2, which cancels
// catastrophically when the row mean is large relative to its spread. That is
// the same formulation the direct-codegen builder uses, so the two agree, but
// it is less accurate than a two-pass mean-then-variance for inputs with a
// large DC offset.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

namespace {

constexpr float kEpsilon = 1e-5f;

// Normalize `rows_to_process` rows of `cols` elements each, in place across
// separate input/output buffers.
template <typename T, int N>
void layer_norm_rows_impl(const T *__restrict input, T *__restrict output,
                          int32_t cols, int32_t rows_to_process) {
  event0();

  const int vector_chunks = cols / N;
  for (int row = 0; row < rows_to_process; row++) {
    ::aie::vector<T, N> sum_acc = ::aie::zeros<T, N>();
    ::aie::vector<float, N> sum_sq_acc = ::aie::zeros<float, N>();
    int input_index = row * cols;
    for (int i = 0; i < vector_chunks; i++) {
      ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input + input_index);
      sum_acc = ::aie::add(sum_acc, reg_a);
      ::aie::vector<float, N> sq_acc = ::aie::mul(reg_a, reg_a);
      sum_sq_acc = ::aie::add(sum_sq_acc, sq_acc);
      input_index += N;
    }
    input_index -= cols;

    const float sum_of_vals = ::aie::reduce_add(sum_acc);
    const float sum_of_sq_vals = ::aie::reduce_add(sum_sq_acc);

    const float mean = sum_of_vals / float(cols);
    const float variance = (sum_of_sq_vals / float(cols)) - mean * mean;
    const float inv_std = ::aie::invsqrt(variance + kEpsilon);

    ::aie::vector<T, N> mean_v = ::aie::broadcast<T, N>(mean);
    ::aie::vector<T, N> inv_std_v = ::aie::broadcast<T, N>(inv_std);

    for (int i = 0; i < vector_chunks; i++) {
      ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input + input_index);
      ::aie::vector<T, N> diff_v = ::aie::sub(reg_a, mean_v);
      ::aie::vector<T, N> norm_v = ::aie::mul(diff_v, inv_std_v);
      ::aie::store_v(output + input_index, norm_v);
      input_index += N;
    }
  }

  event1();
}

// Residual add fused into the same pass: statistics and normalization both run
// over (input1 + input2), so the sum is read twice rather than materialized.
template <typename T, int N>
void add_layer_norm_rows_impl(const T *__restrict input1,
                              const T *__restrict input2, T *__restrict output,
                              int32_t cols, int32_t rows_to_process) {
  event0();

  const int vector_chunks = cols / N;
  for (int row = 0; row < rows_to_process; row++) {
    ::aie::vector<T, N> sum_acc = ::aie::zeros<T, N>();
    ::aie::vector<float, N> sum_sq_acc = ::aie::zeros<float, N>();
    int input_index = row * cols;
    for (int i = 0; i < vector_chunks; i++) {
      ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input1 + input_index);
      ::aie::vector<T, N> reg_b = ::aie::load_v<N>(input2 + input_index);
      ::aie::vector<T, N> reg_sum = ::aie::add(reg_a, reg_b);
      sum_acc = ::aie::add(sum_acc, reg_sum);
      ::aie::vector<float, N> sq_acc = ::aie::mul(reg_sum, reg_sum);
      sum_sq_acc = ::aie::add(sum_sq_acc, sq_acc);
      input_index += N;
    }
    input_index -= cols;

    const float sum_of_vals = ::aie::reduce_add(sum_acc);
    const float sum_of_sq_vals = ::aie::reduce_add(sum_sq_acc);

    const float mean = sum_of_vals / float(cols);
    const float variance = (sum_of_sq_vals / float(cols)) - mean * mean;
    const float inv_std = ::aie::invsqrt(variance + kEpsilon);

    ::aie::vector<T, N> mean_v = ::aie::broadcast<T, N>(mean);
    ::aie::vector<T, N> inv_std_v = ::aie::broadcast<T, N>(inv_std);

    for (int i = 0; i < vector_chunks; i++) {
      ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input1 + input_index);
      ::aie::vector<T, N> reg_b = ::aie::load_v<N>(input2 + input_index);
      ::aie::vector<T, N> reg_sum = ::aie::add(reg_a, reg_b);
      ::aie::vector<T, N> diff_v = ::aie::sub(reg_sum, mean_v);
      ::aie::vector<T, N> norm_v = ::aie::mul(diff_v, inv_std_v);
      ::aie::store_v(output + input_index, norm_v);
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

// Single row. Same result as the direct-codegen builder in layer_norm.py.
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
