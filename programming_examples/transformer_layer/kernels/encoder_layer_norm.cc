//===- encoder_layer_norm.cc ------------------------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// The encoder block's LayerNorm reductions and their staged variants.
//
// Textually included by encoder.cc, the same way
// matrix_multiplication/bf16_in_fp32_out/mm_aie2p.cc includes zero.cc. There is
// no separately compiled object for this file. The split is by role -- matmul
// microkernels in encoder_matmul.cc, LayerNorm reductions here, extern "C"
// entry points in encoder.cc -- so that no source exceeds the ~800-line
// module-size convention. All three still compile as a single translation unit.
//
// CONTRACT
//   - Nothing here has C linkage. encoder.cc wraps these in its own
//     `ln_`-prefixed extern "C" entry points.
//   - Include only AFTER <aie_api/aie.hpp> and
//   <aie_kernels/aie_kernel_utils.h>.
//   - The file offers both the fused form (fused_add_layer_norm_1/_2, which
//     reduce and normalize in one call) and the staged form
//     (ln_calc_sum_sumsq_vectorized -> fused_layer_norm_1 -> ln_mul_weights_1
//     -> ln_mul_add_1), where the reduction is done elsewhere and its sum /
//     sum-of-squares are passed in. The staged form exists so a herd can split
//     the reduction across cores; it is not a drop-in alternative.
//   - Include guarded, so including it twice in one translation unit is safe.
//
// FOOTGUN: variance is E[x^2] - E[x]^2, which cancels when a row's mean is
// large next to its spread and can round below zero. Every site clamps it at
// zero before aie::invsqrt, because invsqrt of a negative operand returns NaN
// and would poison the whole row. Keep the clamp if you add another site.
//
//===----------------------------------------------------------------------===//

#ifndef TRANSFORMER_LAYER_ENCODER_LAYER_NORM_CC
#define TRANSFORMER_LAYER_ENCODER_LAYER_NORM_CC

template <typename T, unsigned rowA, unsigned colA, unsigned r, unsigned s>
void fused_add_layer_norm_1(const T *__restrict input,
                            const T *__restrict residual,
                            const T *__restrict weight,
                            const float *__restrict sum,
                            const float *__restrict sumsq, T *__restrict output,
                            const int32_t cols,
                            const int32_t col_idx) // For offset into weight
                                                   // vector
{
  event0();

  constexpr float epsilon = 1e-5f;
  constexpr unsigned mmul_c_size =
      r * s; // This is the number of elements in each C tile (microtile)

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (unsigned z = 0; z < rowA; z += 1) {

    const T *__restrict pA1 = input + (z * colA) * mmul_c_size;
    const T *__restrict pR1 = residual + (z * colA) * mmul_c_size;
    T *__restrict pC1 = output + (z * colA) * mmul_c_size;

    // Each row has an accumulator for sum and sum-squared, so offset by z*r to
    // get to the correct row, and then we'll index into the correct one within
    // that row with the loop below
    const float *__restrict pSum1 = sum + z * r;
    const float *__restrict pSumSq1 = sumsq + z * r;

    // For each row within this microtile, apply layer norm and add residual
    for (unsigned ri = 0; ri < r; ri++) {
      float mean = aie::div(*pSum1, aie::to_float(cols));
      float mean_sq = mean * mean;
      float variance = aie::div(*pSumSq1, aie::to_float(cols)) - mean_sq;
      if (variance < 0.0f) {
        variance = 0.0f;
      }
      float inv_std = aie::invsqrt(variance + epsilon);

      const T *__restrict pW = weight + (col_idx * colA * s);

      for (unsigned j = 0; j < colA; j += 2) {
        aie::vector<T, s> A0 = aie::load_v<s>(pA1);
        pA1 += mmul_c_size; // Move pointer to the start of the next microtile
                            // in the same row
        aie::vector<T, s> A1 = aie::load_v<s>(pA1);
        pA1 += mmul_c_size; // Move pointer to the start of the next microtile
                            // in the same row
        auto A01 = aie::concat(A0, A1);
        aie::vector<T, s> R0 = aie::load_v<s>(pR1);
        pR1 += mmul_c_size; // Move pointer to the start of the next microtile
                            // in the same row
        aie::vector<T, s> R1 = aie::load_v<s>(pR1);
        pR1 += mmul_c_size; // Move pointer to the start of the next microtile
                            // in the same row
        auto R01 = aie::concat(R0, R1);

        aie::accum<accfloat, 2 * s> a_acc;
        a_acc.from_vector(A01);
        aie::accum<accfloat, 2 * s> diff_acc = aie::sub(a_acc, mean);
        aie::accum<accfloat, 2 * s> norm_acc =
            aie::mul(diff_acc.template to_vector<float>(), inv_std);
        aie::vector<T, 2 * s> weight_v = aie::load_v<2 * s>(pW);
        pW += 2 * s; // Move weight pointer to the columns
        aie::vector<T, 2 * s> scaled_acc =
            aie::mul(norm_acc.template to_vector<T>(), weight_v);
        // aie::accum<accfloat, 2 * s> out_v = aie::add(scaled_v, beta_v);
        aie::vector<T, 2 * s> out_acc = aie::add(scaled_acc, R01);

#ifndef DEBUG_AIE_KERNELS
        // Write s elements to one row of four r x s tiles in output
        aie::store_v(pC1, out_acc.template extract<s>(0));
        pC1 += mmul_c_size; // Move pointer to the start of the next microtile
                            // in the same row
        aie::store_v(pC1, out_acc.template extract<s>(1));
#else
#if DEBUG_AIE_KERNELS == 0
        // Input values
        aie::store_v(pC1, A0);
        pC1 += mmul_c_size; // Move pointer to the start of the next microtile
                            // in the same row
        aie::store_v(pC1, A1);
#elif DEBUG_AIE_KERNELS == 1
        // Residual values
        aie::store_v(pC1, R0);
        pC1 += mmul_c_size; // Move pointer to the start of the next microtile
                            // in the same row
        aie::store_v(pC1, R1);
#endif
#endif
        pC1 += mmul_c_size; // Move pointer to the start of the next microtile
                            // in the same row
      }

      pSum1++;
      pSumSq1++;
      pA1 -= colA * mmul_c_size; // Move pointer back to the start of the row
      pA1 += s;                  // Move pointer to the next row within the same
                                 // microtile
      pR1 -= colA * mmul_c_size; // Move pointer back to the start of the row
      pR1 += s;                  // Move pointer to the next row within the same
                                 // microtile
      pC1 -= colA * mmul_c_size; // Move pointer back to the start of the row
      pC1 += s;                  // Move pointer to the next row within the same
                                 // microtile
    }
  }
  event1();
}

template <typename T, unsigned rowA, unsigned colA, unsigned r, unsigned s>
void fused_layer_norm_1(const T *__restrict input, const float *__restrict sum,
                        const float *__restrict sumsq, T *__restrict output,
                        const int32_t cols) {
  event0();

  constexpr float epsilon = 1e-5f;
  constexpr unsigned mmul_c_size =
      r * s; // Number of elements in each C microtile

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (unsigned z = 0; z < rowA; z += 1) {
    const T *__restrict pA1 = input + (z * colA) * mmul_c_size;
    T *__restrict pC1 = output + (z * colA) * mmul_c_size;

    const float *__restrict pSum1 = sum + z * r;
    const float *__restrict pSumSq1 = sumsq + z * r;

    // For each row within this microtile, apply layer norm (no
    // weight/residual).
    for (unsigned ri = 0; ri < r; ri++) {
      float mean = aie::div(*pSum1, aie::to_float(cols));
      float mean_sq = mean * mean;
      float variance = aie::div(*pSumSq1, aie::to_float(cols)) - mean_sq;
      if (variance < 0.0f) {
        variance = 0.0f;
      }
      float inv_std = aie::invsqrt(variance + epsilon);

      for (unsigned j = 0; j < colA; j += 2) {
        aie::vector<T, s> A0 = aie::load_v<s>(pA1);
        pA1 += mmul_c_size;
        aie::vector<T, s> A1 = aie::load_v<s>(pA1);
        pA1 += mmul_c_size;
        auto A01 = aie::concat(A0, A1);

        aie::accum<accfloat, 2 * s> a_acc;
        a_acc.from_vector(A01);
        aie::accum<accfloat, 2 * s> diff_acc = aie::sub(a_acc, mean);
        aie::accum<accfloat, 2 * s> norm_acc =
            aie::mul(diff_acc.template to_vector<float>(), inv_std);
        aie::vector<T, 2 * s> out_acc = norm_acc.template to_vector<T>();

        aie::store_v(pC1, out_acc.template extract<s>(0));
        pC1 += mmul_c_size;
        aie::store_v(pC1, out_acc.template extract<s>(1));
        pC1 += mmul_c_size;
      }

      pSum1++;
      pSumSq1++;
      pA1 -= colA * mmul_c_size;
      pA1 += s;
      pC1 -= colA * mmul_c_size;
      pC1 += s;
    }
  }
  event1();
}

template <typename T, unsigned rowA, unsigned colA, unsigned r, unsigned s>
void ln_mul_weights_1(const T *__restrict input, const T *__restrict weight,
                      T *__restrict output, const int32_t col_idx) {
  event0();

  constexpr unsigned mmul_c_size =
      r * s; // Number of elements in each C microtile

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (unsigned z = 0; z < rowA; z += 1) {
    const T *__restrict pA1 = input + (z * colA) * mmul_c_size;
    T *__restrict pC1 = output + (z * colA) * mmul_c_size;
    const T *__restrict pW_row = weight + (col_idx * colA * s);

    for (unsigned ri = 0; ri < r; ri++) {
      const T *__restrict pW = pW_row;
      for (unsigned j = 0; j < colA; j += 2) {
        aie::vector<T, s> A0 = aie::load_v<s>(pA1);
        pA1 += mmul_c_size;
        aie::vector<T, s> A1 = aie::load_v<s>(pA1);
        pA1 += mmul_c_size;
        auto A01 = aie::concat(A0, A1);

        aie::vector<T, 2 * s> weight_v = aie::load_v<2 * s>(pW);
        pW += 2 * s;
        aie::vector<T, 2 * s> out_acc = aie::mul(A01, weight_v);

        aie::store_v(pC1, out_acc.template extract<s>(0));
        pC1 += mmul_c_size;
        aie::store_v(pC1, out_acc.template extract<s>(1));
        pC1 += mmul_c_size;
      }

      pA1 -= colA * mmul_c_size;
      pA1 += s;
      pC1 -= colA * mmul_c_size;
      pC1 += s;
    }
  }
  event1();
}

template <typename T, unsigned rowA, unsigned colA, unsigned r, unsigned s>
void ln_mul_add_1(const T *__restrict input, const T *__restrict residual,
                  const T *__restrict weight, T *__restrict output,
                  const int32_t col_idx) {
  event0();

  constexpr unsigned mmul_c_size =
      r * s; // Number of elements in each C microtile

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (unsigned z = 0; z < rowA; z += 1) {
    const T *__restrict pA1 = input + (z * colA) * mmul_c_size;
    const T *__restrict pR1 = residual + (z * colA) * mmul_c_size;
    T *__restrict pC1 = output + (z * colA) * mmul_c_size;
    const T *__restrict pW_row = weight + (col_idx * colA * s);

    for (unsigned ri = 0; ri < r; ri++) {
      const T *__restrict pW = pW_row;
      for (unsigned j = 0; j < colA; j += 2) {
        aie::vector<T, s> A0 = aie::load_v<s>(pA1);
        pA1 += mmul_c_size;
        aie::vector<T, s> A1 = aie::load_v<s>(pA1);
        pA1 += mmul_c_size;
        auto A01 = aie::concat(A0, A1);

        aie::vector<T, s> R0 = aie::load_v<s>(pR1);
        pR1 += mmul_c_size;
        aie::vector<T, s> R1 = aie::load_v<s>(pR1);
        pR1 += mmul_c_size;
        auto R01 = aie::concat(R0, R1);

        aie::vector<T, 2 * s> weight_v = aie::load_v<2 * s>(pW);
        pW += 2 * s;
        aie::vector<T, 2 * s> scaled = aie::mul(A01, weight_v);
        aie::vector<T, 2 * s> out_acc = aie::add(scaled, R01);

        aie::store_v(pC1, out_acc.template extract<s>(0));
        pC1 += mmul_c_size;
        aie::store_v(pC1, out_acc.template extract<s>(1));
        pC1 += mmul_c_size;
      }

      pA1 -= colA * mmul_c_size;
      pA1 += s;
      pR1 -= colA * mmul_c_size;
      pR1 += s;
      pC1 -= colA * mmul_c_size;
      pC1 += s;
    }
  }
  event1();
}

template <typename T, int N>
void fused_add_layer_norm_2(const T *__restrict input,
                            const T *__restrict residual,
                            const T *__restrict weight, T *__restrict output1,
                            T *__restrict output2, const int32_t cols,
                            const int32_t rows_to_process) {
  event0();
#ifndef DEBUG_AIE_KERNELS
  constexpr float epsilon = 1e-5f;
  int vector_chunks = cols / N;

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(4)
  for (int row = 0; row < rows_to_process; row++) {

    ::aie::vector<T, N> sum_acc = ::aie::zeros<T, N>();
    ::aie::vector<float, N> sum_sq_acc = ::aie::zeros<float, N>();
    int input_idx = row * cols;

    for (int i = 0; i < vector_chunks; i++) {

      ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input + input_idx);
      sum_acc = ::aie::add(sum_acc, reg_a);
      ::aie::vector<float, N> sq_acc = ::aie::mul(reg_a, reg_a);
      sum_sq_acc = ::aie::add(sum_sq_acc, sq_acc);
      input_idx += N;
    }

    input_idx -= cols; // reset pointer to beginning of the row

    float sum_of_vals = ::aie::reduce_add(sum_acc);
    float sum_of_sq_vals = ::aie::reduce_add(sum_sq_acc);

    float mean = sum_of_vals / float(cols);
    float mean_sq = mean * mean;
    float variance = (sum_of_sq_vals / float(cols)) - mean_sq;
    if (variance < 0.0f) {
      variance = 0.0f;
    }
    float inv_std = aie::invsqrt(variance + epsilon);

    ::aie::vector<T, N> mean_v = ::aie::broadcast<T, N>(mean);
    ::aie::vector<T, N> inv_std_v = ::aie::broadcast<T, N>(inv_std);

    const T *__restrict pW = weight;
    const T *__restrict pRes = residual + row * cols;
    T *__restrict pOut1 = output1 + row * cols;
    T *__restrict pOut2 = output2 + row * cols;
    for (int i = 0; i < vector_chunks; i++) {

      ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input + input_idx);
      ::aie::vector<T, N> reg_weight = ::aie::load_v<N>(pW);
      ::aie::vector<T, N> reg_res = ::aie::load_v<N>(pRes);
      ::aie::vector<T, N> diff_v = ::aie::sub(reg_a, mean_v);
      ::aie::vector<T, N> norm_v = ::aie::mul(diff_v, inv_std_v);
      ::aie::vector<T, N> scaled_v = aie::mul(norm_v, reg_weight);
      // ::aie::vector<T, N> out_v = ::aie::add(scaled_v, beta_v);
      ::aie::vector<T, N> out_v = ::aie::add(scaled_v, reg_res);
      ::aie::store_v(pOut1, out_v);
      ::aie::store_v(pOut2, out_v);
      input_idx += N;
      pW += N;
      pRes += N;
      pOut1 += N;
      pOut2 += N;
    }
  }
#else
#if DEBUG_AIE_KERNELS == 0

  // In debug mode, just copy input to output
  int total_elements = rows_to_process * cols;
  const T *__restrict pIn = input;
  T *__restrict pOut1 = output1;
  T *__restrict pOut2 = output2;
  AIE_PREPARE_FOR_PIPELINING
  // AIE_LOOP_MIN_ITERATION_COUNT(4)
  for (int i = 0; i < total_elements; i += N) {
    ::aie::vector<T, N> reg_a = ::aie::load_v<N>(pIn);
    ::aie::store_v(pOut1, reg_a);
    ::aie::store_v(pOut2, reg_a);
    pIn += N;
    pOut1 += N;
    pOut2 += N;
  }
#elif DEBUG_AIE_KERNELS == 1
  // In debug mode, just copy residual to output
  int total_elements = rows_to_process * cols;
  const T *__restrict pRes = residual;
  T *__restrict pOut1 = output1;
  T *__restrict pOut2 = output2;
  AIE_PREPARE_FOR_PIPELINING
  // AIE_LOOP_MIN_ITERATION_COUNT(4)
  for (int i = 0; i < total_elements; i += N) {
    ::aie::vector<T, N> reg_a = ::aie::load_v<N>(pRes);
    ::aie::store_v(pOut1, reg_a);
    ::aie::store_v(pOut2, reg_a);
    pRes += N;
    pOut1 += N;
    pOut2 += N;
  }
#endif
#endif
  event1();
}

template <typename T, int N>
void ln_zero_vectorized(T *__restrict c, int size) {
  event0();

  T *__restrict pOut1 = c;
  auto zero_v = ::aie::zeros<T, N>();

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (int j = 0; j < size; j += N) {
    ::aie::store_v(pOut1, zero_v);
    pOut1 += N;
  }

  event1();
}

template <typename T, unsigned rowA, unsigned colA, unsigned r, unsigned s>
void ln_calc_sum_sumsq_vectorized(const T *__restrict pA,
                                  float *__restrict pSum,
                                  float *__restrict pSumSq) {
  event0();

  constexpr unsigned mmul_c_size =
      r * s; // This is the number of elements in each C tile (microtile)

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (unsigned z = 0; z < rowA; z += 1) {
    const T *__restrict pA1 = pA + (z * colA) * mmul_c_size;
    // Each row has an accumulator for sum and sum-squared, so offset by z*r to
    // get to the correct row, and then we'll index into the correct one within
    // that row with the loop below
    float *__restrict pSum1 = pSum + z * r;
    float *__restrict pSumSq1 = pSumSq + z * r;

    // Per-row accumulators for sum and sum-squared across all column tiles
    for (unsigned i = 0; i < r; i++) {
      float row_sum = *pSum1;
      float row_sumsq = *pSumSq1;
      for (unsigned j = 0; j < colA; j += 2) {
        aie::vector<T, s> A0 = aie::load_v<s>(pA1);
        pA1 += mmul_c_size; // Move pointer to the start of the next microtile
                            // in the same row
        aie::vector<T, s> A1 = aie::load_v<s>(pA1);
        pA1 += mmul_c_size; // Move pointer to the start of the next microtile
                            // in the same row

        auto A01 = aie::concat(A0, A1);
        aie::accum<accfloat, 2 * s> sum_acc;
        sum_acc.from_vector(A01);
        float sum = aie::reduce_add(sum_acc.template to_vector<float>());
        row_sum += sum;

        aie::vector<float, s> a_acc_sq0 = aie::mul(A0, A0);
        float sumsq = aie::reduce_add(a_acc_sq0);
        row_sumsq += sumsq;
        aie::vector<float, s> a_acc_sq1 = aie::mul(A1, A1);
        sumsq = aie::reduce_add(a_acc_sq1);
        row_sumsq += sumsq;
      }
      *pSum1 = row_sum;
      *pSumSq1 = row_sumsq;
      pSum1++;
      pSumSq1++;
      pA1 -= colA * mmul_c_size; // Move pointer back to the start of the row
      pA1 += s;                  // Move pointer to the next row within the same
                                 // microtile
    }
  }

  event1();
}

#endif // TRANSFORMER_LAYER_ENCODER_LAYER_NORM_CC
