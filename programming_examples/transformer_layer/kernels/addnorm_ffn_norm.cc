//===- addnorm_ffn_norm.cc --------------------------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// The fused residual-add + LayerNorm templates and the tile passthroughs.
//
// Textually included by addnorm_ffn.cc, the same way
// matrix_multiplication/bf16_in_fp32_out/mm_aie2p.cc includes zero.cc. There is
// no separately compiled object for this file. The split is by role -- matmul
// microkernels in addnorm_ffn_matmul.cc, add-norm and passthrough templates
// here, extern "C" entry points in addnorm_ffn.cc -- so that no source exceeds
// the ~800-line module-size convention. All three still compile as a single
// translation unit.
//
// CONTRACT
//   - Nothing here has C linkage. addnorm_ffn.cc wraps these in its own
//     extern "C" entry points.
//   - Include only AFTER <aie_api/aie.hpp> and
//   <aie_kernels/aie_kernel_utils.h>.
//   - -DADDNORM_PRE_ADD is read INSIDE the two fused_add_layer_norm bodies in
//     this file. It is a correctness switch, not a performance one; the table
//     at the top of addnorm_ffn.cc is the authority on what each build
//     computes. Nothing in addnorm_ffn.cc's extern "C" layer varies with it,
//     so this is the only file that has to be recompiled to change modes --
//     which is exactly why the compile gate compares the two objects byte for
//     byte.
//   - Include guarded, so including it twice in one translation unit is safe.
//
// FOOTGUN: variance is E[x^2] - E[x]^2, which cancels when a row's mean is
// large next to its spread and can round below zero. Both sites clamp it at
// zero before aie::invsqrt, because invsqrt of a negative operand returns NaN
// and would poison the whole row. Keep the clamp if you add another site.
//
//===----------------------------------------------------------------------===//

#ifndef TRANSFORMER_LAYER_ADDNORM_FFN_NORM_CC
#define TRANSFORMER_LAYER_ADDNORM_FFN_NORM_CC

// Fused residual-add + LayerNorm, single output.
//
// Default build:            output = gamma * norm(input) + residual
// -DADDNORM_PRE_ADD build:  output = gamma * norm(input + residual)
//
// See the ADDNORM_PRE_ADD table at the top of this file. The flag also moves
// the mean/variance reduction from `input` to `input + residual`.
template <typename T, int N>
void fused_add_layer_norm_1(const T *__restrict input,
                            const T *__restrict residual,
                            const T *__restrict weight, T *__restrict output,
                            const int32_t cols, const int32_t rows_to_process) {
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

    // Pass 1: accumulate sum and sum-of-squares for this row.
    for (int i = 0; i < vector_chunks; i++) {

      ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input + input_idx);
#ifdef ADDNORM_PRE_ADD
      // Statistics are taken over (input + residual), not over input alone.
      ::aie::vector<T, N> reg_res = ::aie::load_v<N>(residual + input_idx);
      ::aie::vector<T, N> reg_preadd = ::aie::add(reg_a, reg_res);
      sum_acc = ::aie::add(sum_acc, reg_preadd);
      ::aie::vector<float, N> sq_acc = ::aie::mul(reg_preadd, reg_preadd);
#else
      sum_acc = ::aie::add(sum_acc, reg_a);
      ::aie::vector<float, N> sq_acc = ::aie::mul(reg_a, reg_a);
#endif
      sum_sq_acc = ::aie::add(sum_sq_acc, sq_acc);
      input_idx += N;
    }

    float sum_of_vals = ::aie::reduce_add(sum_acc);
    float sum_of_sq_vals = ::aie::reduce_add(sum_sq_acc);

    float mean = sum_of_vals / float(cols);
    float mean_sq = mean * mean;
    float variance = (sum_of_sq_vals / float(cols)) - mean_sq;
    // E[x^2] - E[x]^2 can round below zero when the row's spread is small next
    // to its mean; invsqrt of a negative operand is NaN, which would poison the
    // whole row. Clamp to the true zero-variance case instead.
    if (variance < 0.0f) {
      variance = 0.0f;
    }
    float inv_std = aie::invsqrt(variance + epsilon);

    ::aie::vector<T, N> mean_v = ::aie::broadcast<T, N>(mean);
    ::aie::vector<T, N> inv_std_v = ::aie::broadcast<T, N>(inv_std);

    // Pass 2: normalize, scale, and emit. pIn restarts at the head of the row.
    const T *__restrict pW = weight;
    const T *__restrict pIn = input + row * cols;
    const T *__restrict pRes = residual + row * cols;
    T *__restrict pOut = output + row * cols;
    for (int i = 0; i < vector_chunks; i++) {

      ::aie::vector<T, N> reg_a = ::aie::load_v<N>(pIn);
      ::aie::vector<T, N> reg_weight = ::aie::load_v<N>(pW);
      ::aie::vector<T, N> reg_res = ::aie::load_v<N>(pRes);
#ifdef ADDNORM_PRE_ADD
      ::aie::vector<T, N> reg_preadd = ::aie::add(reg_a, reg_res);
      ::aie::vector<T, N> diff_v = ::aie::sub(reg_preadd, mean_v);
#else
      ::aie::vector<T, N> diff_v = ::aie::sub(reg_a, mean_v);
#endif
      ::aie::vector<T, N> norm_v = ::aie::mul(diff_v, inv_std_v);
      ::aie::vector<T, N> scaled_v = aie::mul(norm_v, reg_weight);
#ifdef ADDNORM_PRE_ADD
      // Residual is already inside scaled_v; do NOT add it a second time.
      ::aie::store_v(pOut, scaled_v);
#else
      // ::aie::vector<T, N> out_v = ::aie::add(scaled_v, beta_v);
      ::aie::vector<T, N> out_v = ::aie::add(scaled_v, reg_res);
      ::aie::store_v(pOut, out_v);
#endif
      pIn += N;
      pW += N;
      pRes += N;
      pOut += N;
    }
  }
#else
#if DEBUG_AIE_KERNELS == 0
  // In debug mode, just copy input to output
  int total_elements = rows_to_process * cols;
  const T *__restrict pIn = input;
  T *__restrict pOut = output;
  AIE_PREPARE_FOR_PIPELINING
  // AIE_LOOP_MIN_ITERATION_COUNT(4)
  for (int i = 0; i < total_elements; i += N) {
    ::aie::vector<T, N> reg_a = ::aie::load_v<N>(pIn);
    ::aie::store_v(pOut, reg_a);
    pIn += N;
    pOut += N;
  }
#elif DEBUG_AIE_KERNELS == 1
  // In debug mode, just copy residual to output
  int total_elements = rows_to_process * cols;
  const T *__restrict pRes = residual;
  T *__restrict pOut = output;
  AIE_PREPARE_FOR_PIPELINING
  // AIE_LOOP_MIN_ITERATION_COUNT(4)
  for (int i = 0; i < total_elements; i += N) {
    ::aie::vector<T, N> reg_a = ::aie::load_v<N>(pRes);
    ::aie::store_v(pOut, reg_a);
    pRes += N;
    pOut += N;
  }
#endif
#endif
  event1();
}

// Fused residual-add + LayerNorm, two outputs.
//
// Default build:            output1 = output2 = gamma * norm(input) + residual
//                           (the two outputs are bit-identical)
// -DADDNORM_PRE_ADD build:  output1 = gamma * norm(input + residual)
//                           output2 = input + residual  (raw pre-add sum, which
//                           downstream uses as the next block's residual)
//
// See the ADDNORM_PRE_ADD table at the top of this file.
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

    // Pass 1: accumulate sum and sum-of-squares for this row.
    for (int i = 0; i < vector_chunks; i++) {

      ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input + input_idx);
#ifdef ADDNORM_PRE_ADD
      // Statistics are taken over (input + residual), not over input alone.
      ::aie::vector<T, N> reg_res = ::aie::load_v<N>(residual + input_idx);
      ::aie::vector<T, N> reg_preadd = ::aie::add(reg_a, reg_res);
      sum_acc = ::aie::add(sum_acc, reg_preadd);
      ::aie::vector<float, N> sq_acc = ::aie::mul(reg_preadd, reg_preadd);
#else
      sum_acc = ::aie::add(sum_acc, reg_a);
      ::aie::vector<float, N> sq_acc = ::aie::mul(reg_a, reg_a);
#endif
      sum_sq_acc = ::aie::add(sum_sq_acc, sq_acc);
      input_idx += N;
    }

    float sum_of_vals = ::aie::reduce_add(sum_acc);
    float sum_of_sq_vals = ::aie::reduce_add(sum_sq_acc);

    float mean = sum_of_vals / float(cols);
    float mean_sq = mean * mean;
    float variance = (sum_of_sq_vals / float(cols)) - mean_sq;
    // Same clamp as the 1-output form: without it a row whose variance rounds
    // negative emits NaN for every element.
    if (variance < 0.0f) {
      variance = 0.0f;
    }
    float inv_std = aie::invsqrt(variance + epsilon);

    ::aie::vector<T, N> mean_v = ::aie::broadcast<T, N>(mean);
    ::aie::vector<T, N> inv_std_v = ::aie::broadcast<T, N>(inv_std);

    // Pass 2: normalize, scale, and emit. pIn restarts at the head of the row.
    const T *__restrict pW = weight;
    const T *__restrict pIn = input + row * cols;
    const T *__restrict pRes = residual + row * cols;
    T *__restrict pOut1 = output1 + row * cols;
    T *__restrict pOut2 = output2 + row * cols;
    for (int i = 0; i < vector_chunks; i++) {

      ::aie::vector<T, N> reg_a = ::aie::load_v<N>(pIn);
      ::aie::vector<T, N> reg_weight = ::aie::load_v<N>(pW);
      ::aie::vector<T, N> reg_res = ::aie::load_v<N>(pRes);
#ifdef ADDNORM_PRE_ADD
      ::aie::vector<T, N> reg_preadd = ::aie::add(reg_a, reg_res);
      ::aie::vector<T, N> diff_v = ::aie::sub(reg_preadd, mean_v);
#else
      ::aie::vector<T, N> diff_v = ::aie::sub(reg_a, mean_v);
#endif
      ::aie::vector<T, N> norm_v = ::aie::mul(diff_v, inv_std_v);
      ::aie::vector<T, N> scaled_v = aie::mul(norm_v, reg_weight);
#ifdef ADDNORM_PRE_ADD
      // Residual is already inside scaled_v; do NOT add it a second time.
      // output2 exports the raw pre-add sum for the next block's residual path.
      ::aie::store_v(pOut1, scaled_v);
      ::aie::store_v(pOut2, reg_preadd);
#else
      // ::aie::vector<T, N> out_v = ::aie::add(scaled_v, beta_v);
      ::aie::vector<T, N> out_v = ::aie::add(scaled_v, reg_res);
      ::aie::store_v(pOut1, out_v);
      ::aie::store_v(pOut2, out_v);
#endif
      pIn += N;
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

template <typename T, int N, int numMmulBlocks, int blockSize>
void ffn_passThrough_aie_MLessThanr(const T *__restrict in, T *__restrict out,
                                    const int32_t row_offset) {
  event0();

  T *__restrict pOut = out;
  const T *__restrict pIn = in + row_offset * N;

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(4)
  for (unsigned block = 0; block < numMmulBlocks; block++) {
    // Write half of the block to output for each block
    auto in_vec = ::aie::load_v<N>(pIn);
    ::aie::store_v(pOut, in_vec);
    pIn += blockSize;
    pOut += N;
  }

  event1();
}

template <typename T, int N, int numMmulBlocks, int blockSize>
void ffn_passThrough_aie(const T *__restrict in, T *__restrict out,
                         const int32_t row_offset) {
  event0();

  T *__restrict pOut = out;
  const T *__restrict pIn = in + row_offset * numMmulBlocks / 2 * blockSize;
  constexpr int iters = numMmulBlocks / 2;
  constexpr int blockIters = blockSize / N;

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(4)
  for (unsigned block = 0; block < iters; block++) {
    for (unsigned i = 0; i < blockIters; i++) {
      auto in_vec = ::aie::load_v<N>(pIn);
      ::aie::store_v(pOut, in_vec);
      pIn += N;
      pOut += N;
    }
  }

  event1();
}

template <typename T, int N>
void ln_passThrough_in_aie(const T *__restrict in, T *__restrict out,
                           const int32_t K, const int32_t k,
                           const int32_t rows_to_process,
                           const int32_t col_offset) {
  event0();

  T *__restrict pOut = out;

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(4)
  for (unsigned row = 0; row < rows_to_process; row++) {

    const T *__restrict pIn = in + row * K + col_offset * k;

    for (int j = 0; j < k; j += N) { // Nx samples per loop

      ::aie::vector<T, N> reg_a = ::aie::load_v<N>(pIn);
      ::aie::store_v(pOut, reg_a);
      pOut += N;
      pIn += N;
    }
  }

  event1();
}

template <typename T, int N>
void ln_passThrough_out_aie(const T *__restrict in, T *__restrict out,
                            const int32_t K, const int32_t k,
                            const int32_t rows_to_process,
                            const int32_t col_offset) {
  event0();

  const T *__restrict pIn = in;

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(4)
  for (unsigned row = 0; row < rows_to_process; row++) {

    T *__restrict pOut = out + row * K + col_offset * k;

    for (int j = 0; j < k; j += N) {

      ::aie::vector<T, N> reg_a = ::aie::load_v<N>(pIn);
      ::aie::store_v(pOut, reg_a);
      pOut += N;
      pIn += N;
    }
  }

  event1();
}

#endif // TRANSFORMER_LAYER_ADDNORM_FFN_NORM_CC
