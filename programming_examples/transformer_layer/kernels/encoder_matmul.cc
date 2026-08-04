//===- encoder_matmul.cc ----------------------------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// The encoder block's bf16 matmul microkernels, 2x2-expanded over aie::mmul.
//
// Textually included by encoder.cc, the same way
// matrix_multiplication/bf16_in_fp32_out/mm_aie2p.cc includes zero.cc. There is
// no separately compiled object for this file. The split is by role -- matmul
// microkernels here, LayerNorm reductions in encoder_layer_norm.cc, extern "C"
// entry points in encoder.cc -- so that no source exceeds the ~800-line
// module-size convention. All three still compile as a single translation unit.
//
// CONTRACT
//   - Nothing here has C linkage. encoder.cc wraps these in its own
//     `ffn_`-prefixed extern "C" entry points, which is what keeps encoder.o
//     and addnorm_ffn.o from colliding on an unprefixed name.
//   - Include only AFTER <aie_api/aie.hpp> and
//   <aie_kernels/aie_kernel_utils.h>:
//     this file uses aie::mmul and the AIE_PREPARE_FOR_PIPELINING /
//     AIE_LOOP_MIN_ITERATION_COUNT / AIE_LOOP_FLATTEN macros without including
//     either header itself.
//   - Include guarded, so including it twice in one translation unit is safe.
//
// FOOTGUN: the two templates differ in where the running sum comes from.
// matmul_vectorized_2x2_mmul accumulates in place -- it loads the prior
// contents of pC and adds. matmul_with_acc_vectorized_2x2_mmul reads the
// running sum from a separate pAcc and writes to pC, so accumulator and
// destination may be distinct buffers. Swapping them silently changes which
// buffer the previous K-step's partial products were expected to be in.
//
//===----------------------------------------------------------------------===//

#ifndef TRANSFORMER_LAYER_ENCODER_MATMUL_CC
#define TRANSFORMER_LAYER_ENCODER_MATMUL_CC

/* Blocked MatMul kernel (vectorized) utilizing the aie::mmul class.
 * The matrices are assumed to be pre-tiled with the following shapes
 * for the aie::mmul class: A => rxs, B => sxt, C => rxt.
 *
 * The matrix dimensions of the kernel are defined by rowA, colA and colB.
 * In this kernel we expand the aie::mmul two times in both the 'm' dimension
 * and the 'n' dimension, producing a 2x2 block in the output matrix.
 *
 * Data within each tile (rxs, sxt and rxt) are assumed to be in row-major
 * order. Also, the entire tiles themselves are stored in row-major order.
 *
 * C is accumulated into: the prior contents of C are loaded and added to.
 */
template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t>
void matmul_vectorized_2x2_mmul(const T_in *__restrict pA,
                                const T_in *__restrict pB,
                                T_out *__restrict pC) {
  using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

  event0();

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (unsigned z = 0; z < rowA; z += 2) {

    T_out *__restrict pC1 = pC + (z * colB) * MMUL::size_C;
    T_out *__restrict pC2 = pC + ((z + 1) * colB) * MMUL::size_C;

    for (unsigned j = 0; j < colB; j += 2)
#ifdef OPT_PERF_ENABLED
      AIE_LOOP_FLATTEN
#endif
      {
        const T_in *__restrict pA1 = pA + (z * colA) * MMUL::size_A;
        const T_in *__restrict pA2 = pA + ((z + 1) * colA) * MMUL::size_A;
        const T_in *__restrict pB1 = pB + (j)*MMUL::size_B;
        const T_in *__restrict pB2 = pB + (j + 1) * MMUL::size_B;
        aie::vector<T_in, MMUL::size_A> A0;
        aie::vector<T_in, MMUL::size_A> A1;
        aie::vector<T_in, MMUL::size_B> B0;
        aie::vector<T_in, MMUL::size_B> B1;

        aie::vector<T_out, MMUL::size_C> acc_C00 =
            aie::load_v<MMUL::size_C>(pC1);
        aie::vector<T_out, MMUL::size_C> acc_C01 =
            aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C);
        aie::vector<T_out, MMUL::size_C> acc_C10 =
            aie::load_v<MMUL::size_C>(pC2);
        aie::vector<T_out, MMUL::size_C> acc_C11 =
            aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C);

        MMUL C00(acc_C00);
        MMUL C01(acc_C01);
        MMUL C10(acc_C10);
        MMUL C11(acc_C11);

        for (unsigned i = 0; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
          AIE_LOOP_FLATTEN
#endif
          {
            A0 = aie::load_v<MMUL::size_A>(pA1);
            pA1 += MMUL::size_A;
            A1 = aie::load_v<MMUL::size_A>(pA2);
            pA2 += MMUL::size_A;
            B0 = aie::load_v<MMUL::size_B>(pB1);
            pB1 += MMUL::size_B * colB;
            B1 = aie::load_v<MMUL::size_B>(pB2);
            pB2 += MMUL::size_B * colB;

            C00.mac(A0, B0);
            C01.mac(A0, B1);
            C10.mac(A1, B0);
            C11.mac(A1, B1);
          }

        aie::store_v(pC1, C00.template to_vector<T_out>());
        pC1 += MMUL::size_C;
        aie::store_v(pC1, C01.template to_vector<T_out>());
        pC1 += MMUL::size_C;
        aie::store_v(pC2, C10.template to_vector<T_out>());
        pC2 += MMUL::size_C;
        aie::store_v(pC2, C11.template to_vector<T_out>());
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
 *
 * The running sum is read from pAcc and the result written to pC, so the
 * accumulator and the destination may be distinct buffers.
 */
template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t>
void matmul_with_acc_vectorized_2x2_mmul(const T_in *__restrict pA,
                                         const T_in *__restrict pB,
                                         T_out *__restrict pAcc,
                                         T_out *__restrict pC) {
  using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

  event0();

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (unsigned z = 0; z < rowA; z += 2) {

    T_out *__restrict pAcc1 = pAcc + (z * colB) * MMUL::size_C;
    T_out *__restrict pAcc2 = pAcc + ((z + 1) * colB) * MMUL::size_C;
    T_out *__restrict pC1 = pC + (z * colB) * MMUL::size_C;
    T_out *__restrict pC2 = pC + ((z + 1) * colB) * MMUL::size_C;

    for (unsigned j = 0; j < colB; j += 2)
#ifdef OPT_PERF_ENABLED
      AIE_LOOP_FLATTEN
#endif
      {
        const T_in *__restrict pA1 = pA + (z * colA) * MMUL::size_A;
        const T_in *__restrict pA2 = pA + ((z + 1) * colA) * MMUL::size_A;
        const T_in *__restrict pB1 = pB + (j)*MMUL::size_B;
        const T_in *__restrict pB2 = pB + (j + 1) * MMUL::size_B;
        aie::vector<T_in, MMUL::size_A> A0;
        aie::vector<T_in, MMUL::size_A> A1;
        aie::vector<T_in, MMUL::size_B> B0;
        aie::vector<T_in, MMUL::size_B> B1;

        aie::vector<T_out, MMUL::size_C> acc_C00 =
            aie::load_v<MMUL::size_C>(pAcc1);
        pAcc1 += MMUL::size_C;
        aie::vector<T_out, MMUL::size_C> acc_C01 =
            aie::load_v<MMUL::size_C>(pAcc1);
        pAcc1 += MMUL::size_C;
        aie::vector<T_out, MMUL::size_C> acc_C10 =
            aie::load_v<MMUL::size_C>(pAcc2);
        pAcc2 += MMUL::size_C;
        aie::vector<T_out, MMUL::size_C> acc_C11 =
            aie::load_v<MMUL::size_C>(pAcc2);
        pAcc2 += MMUL::size_C;

        MMUL C00(acc_C00);
        MMUL C01(acc_C01);
        MMUL C10(acc_C10);
        MMUL C11(acc_C11);

        for (unsigned i = 0; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
          AIE_LOOP_FLATTEN
#endif
          {
            A0 = aie::load_v<MMUL::size_A>(pA1);
            pA1 += MMUL::size_A;
            A1 = aie::load_v<MMUL::size_A>(pA2);
            pA2 += MMUL::size_A;
            B0 = aie::load_v<MMUL::size_B>(pB1);
            pB1 += MMUL::size_B * colB;
            B1 = aie::load_v<MMUL::size_B>(pB2);
            pB2 += MMUL::size_B * colB;

            C00.mac(A0, B0);
            C01.mac(A0, B1);
            C10.mac(A1, B0);
            C11.mac(A1, B1);
          }

        aie::store_v(pC1, C00.template to_vector<T_out>());
        pC1 += MMUL::size_C;
        aie::store_v(pC1, C01.template to_vector<T_out>());
        pC1 += MMUL::size_C;
        aie::store_v(pC2, C10.template to_vector<T_out>());
        pC2 += MMUL::size_C;
        aie::store_v(pC2, C11.template to_vector<T_out>());
        pC2 += MMUL::size_C;
      }
  }

  event1();
}

#endif // TRANSFORMER_LAYER_ENCODER_MATMUL_CC
