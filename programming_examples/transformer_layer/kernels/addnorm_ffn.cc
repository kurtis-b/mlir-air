//===- addnorm_ffn.cc -------------------------------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Device kernels for the transformer layer's feed-forward network (FFN) and its
// fused residual-add + LayerNorm. One source, two independently gated halves,
// and one flag that changes the *numerics* of the LayerNorm half.
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

/* Blocked MatMul kernel (vectorized) utilizing the aie::mmul class.
 * The matrices are assumed to be pre-tiled with the following shapes
 * for the aie:mmul class: A => rxs, B => sxt, C => rxt.
 *
 * The matrix dimensions of the kernel are defined by rowA, colA and colB.
 * In this particular kernel we expand the aie::mmul two times in each
 * input matrices A (in 'm' dimension, or rowA) and B (in 'n' dimension, or
 * ColB), leading to a 1x4 expansion in output matrix C (see C00, C01, C02, C03
 * below). This expansion helps with accumulator registers usage, which leads in
 * attaining high kernel efficiency (SIMD utilization).
 *
 * Data within each tile (rxs, sxt and rxt) are assumed to be in row-major
 * order. Also, the entire tiles themselves are stored in row-major order, as
 * shown in the example below for matrix A:
 *
 *      <-s->
 *    _  ________________________
 * 	  r |  1 |  2 |  3 | ...
 * 	  _ |____|____|____|
 * 	    |  x | x+1| x+2| ...
 * 	    |____|____|____|
 * 	    |.
 * 	    |.
 * 	    |.
 *
 * A simplified example of this kernel can be found in the AIE-API
 * documentation: https://xilinx.github.io/aie_api/group__group__mmul.html
 */
template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t>
void matmul_vectorized_1x4_mmul(const T_in *__restrict pHalfA1,
                                const T_in *__restrict pHalfA2,
                                const T_in *__restrict pB,
                                T_out *__restrict pC) {
  // Don't change functionality with debug mode since the weight matrix is an
  // identity matrix
  using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;
  const int szA =
      MMUL::size_A; // The tiles have half of the rows expected for mmul api

  event0();

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (unsigned z = 0; z < rowA; z += 1) {
    // Process the first input A
    T_out *__restrict pC1 = pC + (z * colB) * MMUL::size_C;

    for (unsigned j = 0; j < colB; j += 4)
#ifdef OPT_PERF_ENABLED
      AIE_LOOP_FLATTEN
#endif
      {

        const T_in *__restrict pHA1 = pHalfA1 + (z * colA) * szA;
        const T_in *__restrict pB1 = pB + (j)*MMUL::size_B;
        const T_in *__restrict pB2 = pB + (j + 1) * MMUL::size_B;
        const T_in *__restrict pB3 = pB + (j + 2) * MMUL::size_B;
        const T_in *__restrict pB4 = pB + (j + 3) * MMUL::size_B;
        aie::vector<T_in, szA> HA10;
        aie::vector<T_in, MMUL::size_B> B0;
        aie::vector<T_in, MMUL::size_B> B1;
        aie::vector<T_in, MMUL::size_B> B2;
        aie::vector<T_in, MMUL::size_B> B3;

        // Load partial results from C buffer for accumulation in-place. The
        // zero.cc function handles the zeroing of data when a new
        // accumulation is needed (after the 'K' reduction dimension)
        aie::vector<T_out, MMUL::size_C> acc_C00 =
            aie::load_v<MMUL::size_C>(pC1);
        aie::vector<T_out, MMUL::size_C> acc_C01 =
            aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C);
        aie::vector<T_out, MMUL::size_C> acc_C02 =
            aie::load_v<MMUL::size_C>(pC1 + 2 * MMUL::size_C);
        aie::vector<T_out, MMUL::size_C> acc_C03 =
            aie::load_v<MMUL::size_C>(pC1 + 3 * MMUL::size_C);

        MMUL C00(acc_C00);
        MMUL C01(acc_C01);
        MMUL C02(acc_C02);
        MMUL C03(acc_C03);

        for (unsigned i = 0; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
          AIE_LOOP_FLATTEN
#endif
          {
            HA10 = aie::load_v<szA>(pHA1);
            pHA1 += szA;
            B0 = aie::load_v<MMUL::size_B>(pB1);
            pB1 += MMUL::size_B * colB;
            B1 = aie::load_v<MMUL::size_B>(pB2);
            pB2 += MMUL::size_B * colB;
            B2 = aie::load_v<MMUL::size_B>(pB3);
            pB3 += MMUL::size_B * colB;
            B3 = aie::load_v<MMUL::size_B>(pB4);
            pB4 += MMUL::size_B * colB;

            C00.mac(HA10, B0);
            C01.mac(HA10, B1);
            C02.mac(HA10, B2);
            C03.mac(HA10, B3);
          }

        // TODO make shift right here to keep most significat bits
        // when lowering the output
        // example below shows how to shift right 10 bits
        // #define SHIFT 10
        // aie::store_v(pC1, C00.template to_vector<T_out>(SHIFT));

        aie::store_v(pC1, C00.template to_vector<T_out>());
        pC1 += MMUL::size_C;
        aie::store_v(pC1, C01.template to_vector<T_out>());
        pC1 += MMUL::size_C;
        aie::store_v(pC1, C02.template to_vector<T_out>());
        pC1 += MMUL::size_C;
        aie::store_v(pC1, C03.template to_vector<T_out>());
        pC1 += MMUL::size_C;
      }

    // Process the second input A
    T_out *__restrict pC2 = pC + ((z + 1) * colB) * MMUL::size_C;

    for (unsigned j = 0; j < colB; j += 4)
#ifdef OPT_PERF_ENABLED
      AIE_LOOP_FLATTEN
#endif
      {

        const T_in *__restrict pHA2 = pHalfA2 + (z * colA) * szA;
        const T_in *__restrict pB1 = pB + (j)*MMUL::size_B;
        const T_in *__restrict pB2 = pB + (j + 1) * MMUL::size_B;
        const T_in *__restrict pB3 = pB + (j + 2) * MMUL::size_B;
        const T_in *__restrict pB4 = pB + (j + 3) * MMUL::size_B;
        aie::vector<T_in, szA> HA20;
        aie::vector<T_in, MMUL::size_B> B0;
        aie::vector<T_in, MMUL::size_B> B1;
        aie::vector<T_in, MMUL::size_B> B2;
        aie::vector<T_in, MMUL::size_B> B3;

        // Load partial results from C buffer for accumulation in-place. The
        // zero.cc function handles the zeroing of data when a new
        // accumulation is needed (after the 'K' reduction dimension)
        aie::vector<T_out, MMUL::size_C> acc_C10 =
            aie::load_v<MMUL::size_C>(pC2);
        aie::vector<T_out, MMUL::size_C> acc_C11 =
            aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C);
        aie::vector<T_out, MMUL::size_C> acc_C12 =
            aie::load_v<MMUL::size_C>(pC2 + 2 * MMUL::size_C);
        aie::vector<T_out, MMUL::size_C> acc_C13 =
            aie::load_v<MMUL::size_C>(pC2 + 3 * MMUL::size_C);

        MMUL C10(acc_C10);
        MMUL C11(acc_C11);
        MMUL C12(acc_C12);
        MMUL C13(acc_C13);

        for (unsigned i = 0; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
          AIE_LOOP_FLATTEN
#endif
          {
            HA20 = aie::load_v<szA>(pHA2);
            pHA2 += szA;
            B0 = aie::load_v<MMUL::size_B>(pB1);
            pB1 += MMUL::size_B * colB;
            B1 = aie::load_v<MMUL::size_B>(pB2);
            pB2 += MMUL::size_B * colB;
            B2 = aie::load_v<MMUL::size_B>(pB3);
            pB3 += MMUL::size_B * colB;
            B3 = aie::load_v<MMUL::size_B>(pB4);
            pB4 += MMUL::size_B * colB;

            C10.mac(HA20, B0);
            C11.mac(HA20, B1);
            C12.mac(HA20, B2);
            C13.mac(HA20, B3);
          }

        // TODO make shift right here to keep most significat bits
        // when lowering the output
        // example below shows how to shift right 10 bits
        // #define SHIFT 10
        // aie::store_v(pC2, C00.template to_vector<T_out>(SHIFT));

        aie::store_v(pC2, C10.template to_vector<T_out>());
        pC2 += MMUL::size_C;
        aie::store_v(pC2, C11.template to_vector<T_out>());
        pC2 += MMUL::size_C;
        aie::store_v(pC2, C12.template to_vector<T_out>());
        pC2 += MMUL::size_C;
        aie::store_v(pC2, C13.template to_vector<T_out>());
        pC2 += MMUL::size_C;
      }
  }

  event1();
}

/* Blocked MatMul kernel (vectorized) utilizing the aie::mmul class.
 * The matrices are assumed to be pre-tiled with the following shapes
 * for the aie:mmul class: A => rxs, B => sxt, C => rxt.
 *
 * The matrix dimensions of the kernel are defined by rowA, colA and colB.
 * In this particular kernel we expand the aie::mmul two times in each
 * input matrices A (in 'm' dimension, or rowA) and B (in 'n' dimension, or
 * ColB), leading to a 1x4 expansion in output matrix C (see C00, C01, C02, C03
 * below). This expansion helps with accumulator registers usage, which leads in
 * attaining high kernel efficiency (SIMD utilization).
 *
 * Data within each tile (rxs, sxt and rxt) are assumed to be in row-major
 * order. Also, the entire tiles themselves are stored in row-major order, as
 * shown in the example below for matrix A:
 *
 *      <-s->
 *    _  ________________________
 * 	  r |  1 |  2 |  3 | ...
 * 	  _ |____|____|____|
 * 	    |  x | x+1| x+2| ...
 * 	    |____|____|____|
 * 	    |.
 * 	    |.
 * 	    |.
 *
 * A simplified example of this kernel can be found in the AIE-API
 * documentation: https://xilinx.github.io/aie_api/group__group__mmul.html
 */
template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t>
void matmul_vectorized_1x4_mmul_MLessThanr(const T_in *__restrict pHalfA1,
                                           const T_in *__restrict pHalfA2,
                                           const T_in *__restrict pB,
                                           T_out *__restrict pC) {
  // Don't change functionality with debug mode since the weight matrix is an
  // identity matrix
  using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;
  const int szA =
      MMUL::size_A / 2; // The tiles have half of the rows expected for mmul api

  event0();

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (unsigned z = 0; z < rowA; z += 1) {

    T_out *__restrict pC1 = pC + (z * colB) * MMUL::size_C;

    for (unsigned j = 0; j < colB; j += 4)
#ifdef OPT_PERF_ENABLED
      AIE_LOOP_FLATTEN
#endif
      {

        const T_in *__restrict pHA1 = pHalfA1 + (z * colA) * szA;
        const T_in *__restrict pHA2 = pHalfA2 + (z * colA) * szA;
        const T_in *__restrict pB1 = pB + (j)*MMUL::size_B;
        const T_in *__restrict pB2 = pB + (j + 1) * MMUL::size_B;
        const T_in *__restrict pB3 = pB + (j + 2) * MMUL::size_B;
        const T_in *__restrict pB4 = pB + (j + 3) * MMUL::size_B;
        aie::vector<T_in, szA> HA10;
        aie::vector<T_in, szA> HA20;
        aie::vector<T_in, MMUL::size_B> B0;
        aie::vector<T_in, MMUL::size_B> B1;
        aie::vector<T_in, MMUL::size_B> B2;
        aie::vector<T_in, MMUL::size_B> B3;

        // Load partial results from C buffer for accumulation in-place. The
        // zero.cc function handles the zeroing of data when a new
        // accumulation is needed (after the 'K' reduction dimension)
        aie::vector<T_out, MMUL::size_C> acc_C00 =
            aie::load_v<MMUL::size_C>(pC1);
        aie::vector<T_out, MMUL::size_C> acc_C01 =
            aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C);
        aie::vector<T_out, MMUL::size_C> acc_C02 =
            aie::load_v<MMUL::size_C>(pC1 + 2 * MMUL::size_C);
        aie::vector<T_out, MMUL::size_C> acc_C03 =
            aie::load_v<MMUL::size_C>(pC1 + 3 * MMUL::size_C);

        MMUL C00(acc_C00);
        MMUL C01(acc_C01);
        MMUL C02(acc_C02);
        MMUL C03(acc_C03);

        for (unsigned i = 0; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
          AIE_LOOP_FLATTEN
#endif
          {
            HA10 = aie::load_v<szA>(pHA1);
            pHA1 += szA;
            HA20 = aie::load_v<szA>(pHA2);
            pHA2 += szA;
            auto A0 = ::aie::concat(HA10, HA20); // Expects HA10 to be the first
                                                 // rows and HA20 the next rows
            B0 = aie::load_v<MMUL::size_B>(pB1);
            pB1 += MMUL::size_B * colB;
            B1 = aie::load_v<MMUL::size_B>(pB2);
            pB2 += MMUL::size_B * colB;
            B2 = aie::load_v<MMUL::size_B>(pB3);
            pB3 += MMUL::size_B * colB;
            B3 = aie::load_v<MMUL::size_B>(pB4);
            pB4 += MMUL::size_B * colB;

            C00.mac(A0, B0);
            C01.mac(A0, B1);
            C02.mac(A0, B2);
            C03.mac(A0, B3);
          }

        // TODO make shift right here to keep most significat bits
        // when lowering the output
        // example below shows how to shift right 10 bits
        // #define SHIFT 10
        // aie::store_v(pC1, C00.template to_vector<T_out>(SHIFT));

        aie::store_v(pC1, C00.template to_vector<T_out>());
        pC1 += MMUL::size_C;
        aie::store_v(pC1, C01.template to_vector<T_out>());
        pC1 += MMUL::size_C;
        aie::store_v(pC1, C02.template to_vector<T_out>());
        pC1 += MMUL::size_C;
        aie::store_v(pC1, C03.template to_vector<T_out>());
        pC1 += MMUL::size_C;
      }
  }

  event1();
}

/* Blocked MatMul kernel (vectorized) utilizing the aie::mmul class.
 * The matrices are assumed to be pre-tiled with the following shapes
 * for the aie:mmul class: A => rxs, B => sxt, C => rxt.
 *
 * The matrix dimensions of the kernel are defined by rowA, colA and colB.
 * In this particular kernel we expand the aie::mmul two times in each
 * input matrices A (in 'm' dimension, or rowA) and B (in 'n' dimension, or
 * ColB), leading to a 1x4 expansion in output matrix C (see C00, C01, C02, C03
 * below). This expansion helps with accumulator registers usage, which leads in
 * attaining high kernel efficiency (SIMD utilization).
 *
 * Data within each tile (rxs, sxt and rxt) are assumed to be in row-major
 * order. Also, the entire tiles themselves are stored in row-major order, as
 * shown in the example below for matrix A:
 *
 *      <-s->
 *    _  ________________________
 * 	  r |  1 |  2 |  3 | ...
 * 	  _ |____|____|____|
 * 	    |  x | x+1| x+2| ...
 * 	    |____|____|____|
 * 	    |.
 * 	    |.
 * 	    |.
 *
 * A simplified example of this kernel can be found in the AIE-API
 * documentation: https://xilinx.github.io/aie_api/group__group__mmul.html
 */
template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t>
void matmul_with_acc_vectorized_1x4_mmul(const T_in *__restrict pA,
                                         const T_in *__restrict pB,
                                         T_out *__restrict pAcc,
                                         T_out *__restrict pC) {
  // Don't change functionality with debug mode since the weight matrix is an
  // identity matrix
  using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

  event0();

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (unsigned z = 0; z < rowA; z += 1) {

    T_out *__restrict pAcc1 = pAcc + (z * colB) * MMUL::size_C;
    T_out *__restrict pC1 = pC + (z * colB) * MMUL::size_C;

    for (unsigned j = 0; j < colB; j += 4)
#ifdef OPT_PERF_ENABLED
      AIE_LOOP_FLATTEN
#endif
      {

        const T_in *__restrict pA1 = pA + (z * colA) * MMUL::size_A;
        const T_in *__restrict pB1 = pB + (j)*MMUL::size_B;
        const T_in *__restrict pB2 = pB + (j + 1) * MMUL::size_B;
        const T_in *__restrict pB3 = pB + (j + 2) * MMUL::size_B;
        const T_in *__restrict pB4 = pB + (j + 3) * MMUL::size_B;
        aie::vector<T_in, MMUL::size_A> A0;
        aie::vector<T_in, MMUL::size_B> B0;
        aie::vector<T_in, MMUL::size_B> B1;
        aie::vector<T_in, MMUL::size_B> B2;
        aie::vector<T_in, MMUL::size_B> B3;

        // Load partial results from C buffer for accumulation in-place. The
        // zero.cc function handles the zeroing of data when a new
        // accumulation is needed (after the 'K' reduction dimension)
        aie::vector<T_out, MMUL::size_C> acc_C00 =
            aie::load_v<MMUL::size_C>(pAcc1);
        pAcc1 += MMUL::size_C;
        aie::vector<T_out, MMUL::size_C> acc_C01 =
            aie::load_v<MMUL::size_C>(pAcc1);
        pAcc1 += MMUL::size_C;
        aie::vector<T_out, MMUL::size_C> acc_C02 =
            aie::load_v<MMUL::size_C>(pAcc1);
        pAcc1 += MMUL::size_C;
        aie::vector<T_out, MMUL::size_C> acc_C03 =
            aie::load_v<MMUL::size_C>(pAcc1);
        pAcc1 += MMUL::size_C;

        MMUL C00(acc_C00);
        MMUL C01(acc_C01);
        MMUL C02(acc_C02);
        MMUL C03(acc_C03);

        for (unsigned i = 0; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
          AIE_LOOP_FLATTEN
#endif
          {
            A0 = aie::load_v<MMUL::size_A>(pA1);
            pA1 += MMUL::size_A;
            B0 = aie::load_v<MMUL::size_B>(pB1);
            pB1 += MMUL::size_B * colB;
            B1 = aie::load_v<MMUL::size_B>(pB2);
            pB2 += MMUL::size_B * colB;
            B2 = aie::load_v<MMUL::size_B>(pB3);
            pB3 += MMUL::size_B * colB;
            B3 = aie::load_v<MMUL::size_B>(pB4);
            pB4 += MMUL::size_B * colB;

            C00.mac(A0, B0);
            C01.mac(A0, B1);
            C02.mac(A0, B2);
            C03.mac(A0, B3);
          }

        // TODO make shift right here to keep most significat bits
        // when lowering the output
        // example below shows how to shift right 10 bits
        // #define SHIFT 10
        // aie::store_v(pC1, C00.template to_vector<T_out>(SHIFT));

        aie::store_v(pC1, C00.template to_vector<T_out>());
        pC1 += MMUL::size_C;
        aie::store_v(pC1, C01.template to_vector<T_out>());
        pC1 += MMUL::size_C;
        aie::store_v(pC1, C02.template to_vector<T_out>());
        pC1 += MMUL::size_C;
        aie::store_v(pC1, C03.template to_vector<T_out>());
        pC1 += MMUL::size_C;
      }
  }

  event1();
}

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
