//===- elementwise.cc -------------------------------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Elementwise primitives shared by the transformer-layer device kernels.
//
// Textually included by encoder.cc and addnorm_ffn.cc, in the same way
// matrix_multiplication/bf16_in_fp32_out/mm_aie2p.cc includes zero.cc. There is
// no separately compiled object for this file.
//
// CONTRACT
//   - Nothing here has C linkage. The kernels that include it wrap these in
//     their own prefixed extern "C" entry points (ffn_gelu_bf16,
//     ffn_eltwise_add_bf16_vector). That is deliberate: two kernels in the same
//     ELF would otherwise both define an unprefixed gelu_bf16 and the link
//     would fail on a duplicate symbol.
//   - Include guarded, so including it twice in one translation unit is safe.
//
// FOOTGUN: eltwise_vadd and gelu_tanh_approx_bf16 both truncate their trailing
// partial vector. `size` must be a multiple of 16; a size of 17 processes 16
// elements and leaves the 17th untouched (not zeroed -- untouched), which reads
// as stale data from whatever last used the buffer.
//
//===----------------------------------------------------------------------===//

#ifndef TRANSFORMER_LAYER_ELEMENTWISE_CC
#define TRANSFORMER_LAYER_ELEMENTWISE_CC

#include <aie_api/aie.hpp>
#include <stdint.h>

// c[i] = a[i] + b[i] over `size` elements, 16 at a time.
template <typename T_in, typename T_out>
void eltwise_vadd(const T_in *a, const T_in *b, T_out *c, int size) {
  constexpr int vec_factor = 16;
  event0();
  const T_in *__restrict pA1 = a;
  const T_in *__restrict pB1 = b;
  T_out *__restrict pC1 = c;
  const int F = size / vec_factor;
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(16)
  for (int i = 0; i < F; i++) {
    aie::vector<T_in, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
    pA1 += vec_factor;
    aie::vector<T_in, vec_factor> B0 = aie::load_v<vec_factor>(pB1);
    pB1 += vec_factor;
    aie::vector<T_out, vec_factor> cout = aie::add(A0, B0);
    aie::store_v(pC1, cout);
    pC1 += vec_factor;
  }
  event1();
}

// GeLU via the tanh approximation:
//   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// This is what HuggingFace calls "gelu_new" / "gelu_pytorch_tanh", NOT the
// exact erf-based GeLU. Checking it against an erf reference at bf16 tolerance
// will fail on the tails; compare against the tanh approximation.
inline void gelu_tanh_approx_bf16(const bfloat16 *__restrict input_vector,
                                  bfloat16 *__restrict output_vector,
                                  const int32_t vector_size) {
  event0();

  constexpr int VecLen = 16;

  auto it_in = aie::cbegin_restrict_vector<VecLen>(input_vector);
  auto it_out = aie::begin_restrict_vector<VecLen>(output_vector);

  const aie::vector<bfloat16, VecLen> v05 =
      aie::broadcast<bfloat16, VecLen>((bfloat16)0.5f);
  const aie::vector<bfloat16, VecLen> v1 =
      aie::broadcast<bfloat16, VecLen>((bfloat16)1.0f);
  const aie::vector<bfloat16, VecLen> vs2opi =
      aie::broadcast<bfloat16, VecLen>((bfloat16)0.79788456f); // sqrt(2/pi)
  const aie::vector<bfloat16, VecLen> vBeta =
      aie::broadcast<bfloat16, VecLen>((bfloat16)0.044715f);

  const int iters = vector_size / VecLen;
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (int i = 0; i < iters; i++) {
    aie::vector<bfloat16, VecLen> x = *it_in++;

    aie::vector<bfloat16, VecLen> x2 = aie::mul(x, x).to_vector<bfloat16>();
    aie::vector<bfloat16, VecLen> x3 = aie::mul(x, x2).to_vector<bfloat16>();

    // inner = sqrt(2/pi) * (x + 0.044715 * x^3)
    aie::vector<bfloat16, VecLen> x3_beta =
        aie::mul(x3, vBeta).to_vector<bfloat16>();
    aie::vector<bfloat16, VecLen> inner = aie::add(x, x3_beta);
    auto inner1 = aie::mul(inner, vs2opi);

    auto tanh_out = aie::tanh<bfloat16>(inner1.to_vector<float>());

    // 0.5 * x * (1 + tanh(inner))
    aie::vector<bfloat16, VecLen> one_plus_tanh = aie::add(tanh_out, v1);
    aie::vector<bfloat16, VecLen> mul_v05 =
        aie::mul(v05, one_plus_tanh).to_vector<bfloat16>();
    *it_out++ = aie::mul(x, mul_v05).to_vector<bfloat16>();
  }

  event1();
}

#endif // TRANSFORMER_LAYER_ELEMENTWISE_CC
