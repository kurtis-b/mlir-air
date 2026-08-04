//===- addnorm_ffn_matmul.cc ------------------------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// The FFN's bf16 matmul microkernels, 1x4-expanded over aie::mmul.
//
// Textually included by addnorm_ffn.cc, the same way
// matrix_multiplication/bf16_in_fp32_out/mm_aie2p.cc includes zero.cc. There is
// no separately compiled object for this file. The split is by role -- matmul
// microkernels here, add-norm and passthrough templates in
// addnorm_ffn_norm.cc, extern "C" entry points in addnorm_ffn.cc -- so that no
// source exceeds the ~800-line module-size convention. All three still compile
// as a single translation unit.
//
// CONTRACT
//   - Nothing here has C linkage. addnorm_ffn.cc wraps these in its own
//     extern "C" entry points.
//   - Include only AFTER <aie_api/aie.hpp> and
//   <aie_kernels/aie_kernel_utils.h>:
//     this file uses aie::mmul and the AIE_PREPARE_FOR_PIPELINING /
//     AIE_LOOP_MIN_ITERATION_COUNT / AIE_LOOP_FLATTEN macros without including
//     either header itself.
//   - Include guarded, so including it twice in one translation unit is safe.
//
// FOOTGUN: matmul_vectorized_1x4_mmul takes A as TWO half-tiles (pHalfA1,
// pHalfA2) rather than one contiguous buffer -- it is the up-projection form,
// where the two halves arrive on different DMA channels. The `_MLessThanr`
// variant is for the tail case rowA < r and reads only a prefix of each half;
// using it on a full tile drops rows silently.
//
//===----------------------------------------------------------------------===//

#ifndef TRANSFORMER_LAYER_ADDNORM_FFN_MATMUL_CC
#define TRANSFORMER_LAYER_ADDNORM_FFN_MATMUL_CC

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

#endif // TRANSFORMER_LAYER_ADDNORM_FFN_MATMUL_CC
