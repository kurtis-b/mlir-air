//===- addnorm_ffn.cc -------------------------------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Device kernels for the transformer layer's feed-forward network (FFN) and its
// fused residual-add + LayerNorm. Two independently gated halves, and one flag
// that changes the *numerics* of the LayerNorm half.
//
// FILE LAYOUT
//   This file holds the contract, the build gating and the extern "C" entry
//   points. The template bodies they call live in two sibling sources included
//   below -- addnorm_ffn_matmul.cc (the 1x4-expanded aie::mmul microkernels)
//   and addnorm_ffn_norm.cc (the fused add-norm templates and the tile
//   passthroughs). The split keeps each source inside the ~800-line
//   module-size convention; it is still ONE translation unit and ONE object,
//   the same way matrix_multiplication/bf16_in_fp32_out/mm_aie2p.cc includes
//   zero.cc. Everything below about flags, tile shapes and ABI applies to all
//   three -- in particular ADDNORM_PRE_ADD, whose whole effect is inside
//   addnorm_ffn_norm.cc even though it is documented here.
//
//===----------------------------------------------------------------------===//
//
// ADDNORM_PRE_ADD -- READ THIS FIRST
//
// The two fused_add_layer_norm entry points compute a different function
// depending on whether -DADDNORM_PRE_ADD was passed. This is not a performance
// switch; it is a correctness switch. Picking the wrong one produces plausible
// looking output that is silently the wrong math.
//
// clang-format off
//
//   +---------------------+----------------------------+------------------------+
//   |                     | flag ABSENT (default)      | -DADDNORM_PRE_ADD      |
//   +---------------------+----------------------------+------------------------+
//   | statistics over     | input                      | input + residual       |
//   | _1outs: output      | gamma*norm(input)+residual | gamma*norm(in+res)     |
//   | _2outs: output1     | gamma*norm(input)+residual | gamma*norm(in+res)     |
//   | _2outs: output2     | same as output1            | input + residual       |
//   |                     |                            | (the raw pre-add sum)  |
//   +---------------------+----------------------------+------------------------+
//
// clang-format on
//
// In words:
//   - Default: normalize the input, scale by gamma, and *then* add the
//   residual.
//     Statistics never see the residual.
//   - -DADDNORM_PRE_ADD: a true add-then-LayerNorm. The residual is folded in
//     before the statistics are taken, so mean/variance are those of the sum,
//     and the residual is NOT added again afterwards. The 2-output form
//     additionally exports the pre-add sum (input + residual) through output2,
//     which is what the next block wants as *its* residual.
//
// Note the asymmetry in the 2-output form: in the default build output1 and
// output2 are bit-identical (two stores of the same vector); under
// ADDNORM_PRE_ADD they are different tensors. Host code that assumes the two
// outputs are interchangeable is correct only in the default build.
//
// There is no beta/bias term in either mode. The commented-out `beta_v` add is
// preserved from the original kernel as a marker, not as a feature.
//
//===----------------------------------------------------------------------===//
//
// BUILD GATING -- this file emits NOTHING by default
//
// Every extern "C" symbol lives behind -DBUILD_FFN or -DBUILD_ADDNORM.
// Compiling with neither yields a valid but empty object file, which links fine
// and then fails at run time with an unresolved kernel. Pass at least one:
//
//   -DBUILD_FFN      zero_bf16_up_proj
//                    zero_bf16_down_proj
//                    matmul_bf16_bf16_up_proj_half_inps
//                    matmul_with_acc_bf16_bf16_down_proj
//                    ffn_passThroughTile_out
//                    ffn_gelu_bf16
//                    ffn_eltwise_add_bf16_vector
//
//   -DBUILD_ADDNORM  fused_add_layer_norm_1outs
//                    fused_add_layer_norm_2outs
//                    ln_passThroughTile_out
//                    ln_passThroughTile_in
//
// Both may be passed together; that is the 11-symbol build.
//
//===----------------------------------------------------------------------===//
//
// REQUIRED FLAGS AND THEIR FOOTGUNS
//
// -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
//     MANDATORY for -DBUILD_FFN. Both matmul entry points carry a
//     `static_assert(false, ...)` that fires if it is missing, so a bad build
//     is a compile error rather than a numerical surprise. It MUST be supplied
//     on the command line as a -D. Writing `#define
//     AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16` in this file would be too late:
//     aie_api reads it while <aie_api/aie.hpp> is being included, and that
//     include happens above any source-level #define. A source #define would
//     satisfy the static_assert while leaving aie::mmul configured the other
//     way -- the worst of both worlds.
//     It is not needed (and is harmless) for an ADDNORM-only build.
//
// -DDEBUG_AIE_KERNELS=0 | -DDEBUG_AIE_KERNELS=1
//     Optional. If defined AT ALL it must have the value 0 or 1; the LayerNorm
//     bodies dispatch on `#if DEBUG_AIE_KERNELS == 0 / == 1` with no else, so a
//     bare -DDEBUG_AIE_KERNELS (which expands to 1... but only by accident of
//     the preprocessor) or any other value compiles to an empty function body
//     that writes nothing to the output buffer and returns stale data.
//       0 -> LayerNorm becomes a copy of `input` to the output(s).
//       1 -> LayerNorm becomes a copy of `residual` to the output(s).
//     When defined, ffn_gelu_bf16 also degenerates to a no-op (GeLU skipped),
//     because the debug path assumes an identity weight matrix.
//
// LayerNorm `cols` (a run-time argument, not a -D)
//     Must be a multiple of 32 (the bf16 vector width used by both fused
//     entry points). The row loops step by whole vectors and silently drop a
//     trailing partial vector, so a `cols` of 33 normalizes over the first 32
//     lanes and leaves lane 33 of the output untouched -- stale, not zeroed.
//     Same truncation footgun as eltwise_vadd/gelu in elementwise.cc.
//
// -DDIM_M / -DDIM_K / -DDIM_N  (default 64 each)
//     Inner matmul tile shape. The kernels use r=s=t=8 and a 1x4 unroll, and
//     the static_asserts enforce:
//       up_proj   (M x K) * (K x N):  DIM_M % 8 == 0
//                                     DIM_K % 8 == 0
//                                     DIM_N % 32 == 0   (4*t, the 1x4 unroll)
//       down_proj (M x N) * (N x K):  DIM_M % 8 == 0
//                                     DIM_N % 8 == 0
//                                     DIM_K % 32 == 0   (4*s, the 1x4 unroll)
//     Note that the two are NOT the same constraint set: the dimension needing
//     the %32 swaps between N (up) and K (down). See the comment at the
//     matmul_with_acc_bf16_bf16_down_proj call site for why.
//
//===----------------------------------------------------------------------===//
//
// LINKING CONFLICT WITH encoder.cc
//
// This kernel and encoder.cc both export ffn_gelu_bf16 and
// ffn_eltwise_add_bf16_vector. The two objects therefore CANNOT be linked into
// a single ELF as-is -- the link fails on duplicate symbols. If a multi-launch
// design needs both, one side's symbols must be renamed (or objcopy'd) first.
// The shared implementations themselves live in elementwise.cc and have no C
// linkage precisely so that renaming stays a one-line change here.
//
//===----------------------------------------------------------------------===//

#include <aie_kernels/aie_kernel_utils.h>

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "elementwise.cc"
#include "zero.cc"

// The template bodies live in two sibling sources, included here so this stays
// one translation unit. The seam is by role: matmul microkernels, the add-norm
// and passthrough templates, then the extern "C" entry points below. They are
// independent of each other, so the include order does not matter.
//
// ADDNORM_PRE_ADD is consumed entirely inside addnorm_ffn_norm.cc.
#include "addnorm_ffn_matmul.cc"
#include "addnorm_ffn_norm.cc"

extern "C" {

// If you want to compile microkernels with different inner tile sizes,
// define DIM_M, DIM_K and DIM_N at compile time using -DDIM_M 32 etc.
// These dimensions must be divisible by the r, s, t dimensions used in
// the kernels.

#ifndef DIM_M
#define DIM_M 64
#endif

#ifndef DIM_K
#define DIM_K 64
#endif

#ifndef DIM_N
#define DIM_N 64
#endif

#ifdef BUILD_FFN
// NOTE: the trailing template argument of zero_vectorized is the vector store
// width r. This repo's zero.cc takes it explicitly (unlike the upstream
// version, which derived r = 512 / (sizeof(T) * 8) internally). For a 512-bit
// store unit that is 32 for bfloat16 and 16 for float.
void zero_bf16_up_proj(bfloat16 *C) {
  zero_vectorized<bfloat16, DIM_M, DIM_N, 32>(C);
}

void zero_bf16_down_proj(bfloat16 *C) {
  zero_vectorized<bfloat16, DIM_M, DIM_K, 32>(C);
}

void matmul_bf16_bf16_up_proj_half_inps(const bfloat16 *A1, const bfloat16 *A2,
                                        const bfloat16 *B, bfloat16 *C) {
#ifndef AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
  static_assert(false,
                "AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 must be defined for "
                "this kernel");
#endif
  constexpr int r = 8;
  constexpr int s = 8;
  constexpr int t = 8;

  static_assert(DIM_M % r == 0);
  static_assert(DIM_K % s == 0);
  static_assert(DIM_N % (4 * t) == 0);

  ::aie::set_rounding(aie::rounding_mode::conv_even);

  if constexpr (DIM_M <= r) {
    // The inputs need to be concatenated inside the kernel for this case
    matmul_vectorized_1x4_mmul_MLessThanr<bfloat16, bfloat16, (DIM_M / r),
                                          (DIM_K / s), (DIM_N / t), r, s, t>(
        A1, A2, B, C);
  } else {
    matmul_vectorized_1x4_mmul<bfloat16, bfloat16, (DIM_M / r / 2), (DIM_K / s),
                               (DIM_N / t), r, s, t>(A1, A2, B, C);
  }
}

void matmul_with_acc_bf16_bf16_down_proj(const bfloat16 *A, const bfloat16 *B,
                                         bfloat16 *pAcc, bfloat16 *C) {
#ifndef AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
  static_assert(false,
                "AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 must be defined for "
                "this kernel");
#endif
  constexpr int r = 8;
  constexpr int s = 8;
  constexpr int t = 8;

  static_assert(DIM_M % r == 0);
  static_assert(DIM_N % t == 0);
  static_assert(DIM_K % (4 * s) == 0);

  ::aie::set_rounding(aie::rounding_mode::conv_even);

  // NOTE: K and N, s and t are swapped here compared to the up projection since
  // up projection computes MxK with KxN, while down projection computes MxN
  // with NxK. The template argument list below therefore reads
  // <..., (DIM_M/r), (DIM_N/t), (DIM_K/s), r, t, s> -- N before K and t before
  // s. That is deliberate, not a typo: the template's "colA" is the reduction
  // extent (here N) and its "colB" is the output extent (here K). This swap is
  // also why the %32 divisibility requirement lands on DIM_K for down_proj but
  // on DIM_N for up_proj.
  matmul_with_acc_vectorized_1x4_mmul<bfloat16, bfloat16, (DIM_M / r),
                                      (DIM_N / t), (DIM_K / s), r, t, s>(
      A, B, pAcc, C);
}

void ffn_passThroughTile_out(const bfloat16 *in, bfloat16 *out,
                             const int32_t row_offset) {
  // NOTE: The 8 * 8 below assumes r=8 and s=8 for the matmul kernel
  // configuration, since the input here is the output of down proj matmul which
  // stores the data contiguously in blocks of (r x s)
  constexpr int r = 8;
  constexpr int s = 8;

  if constexpr (DIM_M <= r) {
    ffn_passThrough_aie_MLessThanr<bfloat16, 32, (DIM_M * DIM_K) / (r * s),
                                   r * s>(
        in, out,
        row_offset); // Assumes input sz is larger than output sz, e.g. 8x96
                     // input to 4x96 output
  } else {
    ffn_passThrough_aie<bfloat16, 32, (DIM_M * DIM_K) / (r * s), r * s>(
        in, out,
        row_offset); // Assumes input sz is larger than output sz, e.g. 8x96
                     // input to 4x96 output
  }
}

// NOTE: encoder.cc also exports ffn_gelu_bf16 and ffn_eltwise_add_bf16_vector.
// The two objects cannot be linked into one ELF without renaming one side.
void ffn_gelu_bf16(bfloat16 *__restrict input, bfloat16 *__restrict output,
                   int input_size) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  // Skip GeLU calculations in debug mode
#ifndef DEBUG_AIE_KERNELS
  gelu_tanh_approx_bf16(input, output, input_size);
#endif
}

void ffn_eltwise_add_bf16_vector(bfloat16 *a_in, bfloat16 *b_in,
                                 bfloat16 *c_out, int size) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  eltwise_vadd<bfloat16, bfloat16>(a_in, b_in, c_out, size);
}

#endif

#ifdef BUILD_ADDNORM
void fused_add_layer_norm_1outs(const bfloat16 *input, const bfloat16 *residual,
                                const bfloat16 *weights, bfloat16 *output,
                                const int32_t cols,
                                const int32_t rows_to_process) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  fused_add_layer_norm_1<bfloat16, 32>(input, residual, weights, output, cols,
                                       rows_to_process);
}

void fused_add_layer_norm_2outs(const bfloat16 *input, const bfloat16 *residual,
                                const bfloat16 *weights, bfloat16 *output1,
                                bfloat16 *output2, const int32_t cols,
                                const int32_t rows_to_process) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  fused_add_layer_norm_2<bfloat16, 32>(input, residual, weights, output1,
                                       output2, cols, rows_to_process);
}

void ln_passThroughTile_out(const int16_t *in, int16_t *out, const int32_t cols,
                            const int32_t cols_to_process,
                            const int32_t rows_to_process,
                            const int32_t col_offset) {
  ln_passThrough_out_aie<int16_t, 32>(in, out, cols, cols_to_process,
                                      rows_to_process, col_offset);
}

void ln_passThroughTile_in(const int16_t *in, int16_t *out, const int32_t cols,
                           const int32_t cols_to_process,
                           const int32_t rows_to_process,
                           const int32_t col_offset) {
  ln_passThrough_in_aie<int16_t, 32>(in, out, cols, cols_to_process,
                                     rows_to_process, col_offset);
}
#endif
}
