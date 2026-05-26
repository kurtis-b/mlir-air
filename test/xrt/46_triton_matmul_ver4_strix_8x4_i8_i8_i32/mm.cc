//===- mm.cc -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>

#include <aie_api/aie.hpp>

#ifndef EXTERNAL_K_PACKS
#define EXTERNAL_K_PACKS 9
#endif

#ifndef EXTERNAL_BLOCK_M
#define EXTERNAL_BLOCK_M 2
#endif

#ifndef EXTERNAL_BLOCK_N
#define EXTERNAL_BLOCK_N 2
#endif

#ifndef EXTERNAL_CORE_M_PACKS
#define EXTERNAL_CORE_M_PACKS 18
#endif

#ifndef EXTERNAL_ACTIVE_M_PACKS
#define EXTERNAL_ACTIVE_M_PACKS EXTERNAL_CORE_M_PACKS
#endif

#ifndef EXTERNAL_CORE_N_PACKS
#define EXTERNAL_CORE_N_PACKS 18
#endif

#ifndef EXTERNAL_C_STRIDE_M_PACKS
#define EXTERNAL_C_STRIDE_M_PACKS EXTERNAL_CORE_M_PACKS
#endif

#ifndef EXTERNAL_ATB_C_OFFSET
#define EXTERNAL_ATB_C_OFFSET 0
#endif

#define EXTERNAL_KERNEL_STYLE_PEANO_MMUL 0
#define EXTERNAL_KERNEL_STYLE_HAND_SCHEDULED 1
#define EXTERNAL_KERNEL_STYLE_NATIVE_MMUL 2
#define EXTERNAL_KERNEL_STYLE_ASM_MICROKERNEL 3
#define EXTERNAL_KERNEL_STYLE_NATIVE_MMUL_ATB_REF 4

#ifndef EXTERNAL_KERNEL_STYLE
#define EXTERNAL_KERNEL_STYLE EXTERNAL_KERNEL_STYLE_PEANO_MMUL
#endif

#define EXTERNAL_SCHEDULE_BASELINE 0
#define EXTERNAL_SCHEDULE_FLAT 1
#define EXTERNAL_SCHEDULE_MANUAL_UNROLL 2
#define EXTERNAL_SCHEDULE_SOFTWARE_PIPELINE 3

#ifndef EXTERNAL_SCHEDULE
#define EXTERNAL_SCHEDULE EXTERNAL_SCHEDULE_SOFTWARE_PIPELINE
#endif

#if EXTERNAL_SCHEDULE == EXTERNAL_SCHEDULE_BASELINE
#define EXTERNAL_M_LOOP_ATTR(MIN, MAX) chess_loop_range(MIN, MAX)
#define EXTERNAL_N_LOOP_ATTR(MIN, MAX) chess_loop_range(MIN, MAX)
#define EXTERNAL_K_LOOP_ATTR(MIN, MAX) chess_loop_range(MIN, MAX)
#elif EXTERNAL_SCHEDULE == EXTERNAL_SCHEDULE_FLAT
#define EXTERNAL_M_LOOP_ATTR(MIN, MAX) chess_loop_range(MIN, MAX)
#define EXTERNAL_N_LOOP_ATTR(MIN, MAX) chess_flatten_loop chess_loop_range(MIN, MAX)
#define EXTERNAL_K_LOOP_ATTR(MIN, MAX) chess_flatten_loop chess_loop_range(MIN, MAX)
#elif EXTERNAL_SCHEDULE == EXTERNAL_SCHEDULE_MANUAL_UNROLL
#define EXTERNAL_M_LOOP_ATTR(MIN, MAX)                                           \
  chess_prepare_for_pipelining chess_loop_range(MIN, MAX)
#define EXTERNAL_N_LOOP_ATTR(MIN, MAX)                                           \
  chess_prepare_for_pipelining chess_loop_range(MIN, MAX)
#define EXTERNAL_K_LOOP_ATTR(MIN, MAX)                                           \
  chess_prepare_for_pipelining chess_loop_range(MIN, MAX) chess_unroll_loop(3)
#elif EXTERNAL_SCHEDULE == EXTERNAL_SCHEDULE_SOFTWARE_PIPELINE
#define EXTERNAL_M_LOOP_ATTR(MIN, MAX)                                           \
  chess_prepare_for_pipelining chess_loop_range(MIN, MAX)
#define EXTERNAL_N_LOOP_ATTR(MIN, MAX)                                           \
  chess_prepare_for_pipelining chess_loop_range(MIN, MAX)
#define EXTERNAL_K_LOOP_ATTR(MIN, MAX)                                           \
  chess_prepare_for_pipelining chess_loop_range(MIN, MAX)
#else
#error "unknown EXTERNAL_SCHEDULE"
#endif

#if EXTERNAL_KERNEL_STYLE != EXTERNAL_KERNEL_STYLE_PEANO_MMUL &&                 \
    EXTERNAL_KERNEL_STYLE != EXTERNAL_KERNEL_STYLE_HAND_SCHEDULED &&            \
    EXTERNAL_KERNEL_STYLE != EXTERNAL_KERNEL_STYLE_NATIVE_MMUL &&               \
    EXTERNAL_KERNEL_STYLE != EXTERNAL_KERNEL_STYLE_ASM_MICROKERNEL &&           \
    EXTERNAL_KERNEL_STYLE != EXTERNAL_KERNEL_STYLE_NATIVE_MMUL_ATB_REF
#error "unknown EXTERNAL_KERNEL_STYLE"
#endif

#if EXTERNAL_KERNEL_STYLE == EXTERNAL_KERNEL_STYLE_ASM_MICROKERNEL
#error "asm-microkernel is reserved: dense 3x2 needs six v64acc32 accumulators, but this AIE2P target exposes five 2048-bit DM accumulator registers"
#endif

static inline v64int8 native_load_i8x64(const int8 *__restrict p) {
  return (v64int8)aie::load_v<64>(p);
}

static inline v64acc32 native_load_acc_i8(const int8 *__restrict p) {
  aie::accum<acc32, 64> acc(aie::load_v<64>(p));
  return (v64acc32)acc;
}

static inline void native_store_acc_i8(int8 *__restrict p, v64acc32 acc) {
  aie::store_v(p, aie::accum<acc32, 64>(acc).template to_vector<int8>());
}

static inline v64acc32 native_mac_i8(v64acc32 acc, v64int8 a, v64int8 b) {
  return ::mac_8x8_8x8_conf(a, true, b, true, acc, false, 0, 0, 0);
}

template <typename MMUL>
static inline void matmul_pack_2x2_streamed(int8 *__restrict pA,
                                            int8 *__restrict pB,
                                            int8 *__restrict pC, unsigned m,
                                            unsigned n) {
  constexpr unsigned kKPackCount = EXTERNAL_K_PACKS;
  constexpr unsigned kActiveMPackCount = EXTERNAL_ACTIVE_M_PACKS;
  constexpr unsigned kCoreMPackCount = EXTERNAL_CORE_M_PACKS;
  constexpr unsigned kAStrideK = kActiveMPackCount * MMUL::size_A;
  constexpr unsigned kBStrideN = kKPackCount * MMUL::size_B;
  constexpr unsigned kCStrideN = EXTERNAL_C_STRIDE_M_PACKS * MMUL::size_C;

  MMUL C00(aie::load_v<MMUL::size_C>(pC + n * kCStrideN + m * MMUL::size_C));
  MMUL C01(
      aie::load_v<MMUL::size_C>(pC + (n + 1) * kCStrideN + m * MMUL::size_C));
  MMUL C10(
      aie::load_v<MMUL::size_C>(pC + n * kCStrideN + (m + 1) * MMUL::size_C));
  MMUL C11(aie::load_v<MMUL::size_C>(
      pC + (n + 1) * kCStrideN + (m + 1) * MMUL::size_C));

  const int8 __aie_dm_resource_a *__restrict pA0 =
      (const int8 __aie_dm_resource_a *)(pA + m * MMUL::size_A);
  const int8 __aie_dm_resource_a *__restrict pA1 =
      (const int8 __aie_dm_resource_a *)(pA + (m + 1) * MMUL::size_A);
  const int8 __aie_dm_resource_b *__restrict pB0 =
      (const int8 __aie_dm_resource_b *)(pB + n * kBStrideN);
  const int8 __aie_dm_resource_b *__restrict pB1 =
      (const int8 __aie_dm_resource_b *)(pB + (n + 1) * kBStrideN);

  for (unsigned k = 0; k < kKPackCount; ++k)
    EXTERNAL_K_LOOP_ATTR(EXTERNAL_K_PACKS, EXTERNAL_K_PACKS) {
      aie::vector<int8, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB0);
      aie::vector<int8, MMUL::size_B> B1 = aie::load_v<MMUL::size_B>(pB1);
      pB0 += MMUL::size_B;
      pB1 += MMUL::size_B;

      {
        aie::vector<int8, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA0);
        pA0 += kAStrideK;
        C00.mac(A0, B0);
        C01.mac(A0, B1);
      }
      {
        aie::vector<int8, MMUL::size_A> A1 = aie::load_v<MMUL::size_A>(pA1);
        pA1 += kAStrideK;
        C10.mac(A1, B0);
        C11.mac(A1, B1);
      }
    }

  aie::store_v(pC + n * kCStrideN + m * MMUL::size_C,
               C00.template to_vector<int8>());
  aie::store_v(pC + (n + 1) * kCStrideN + m * MMUL::size_C,
               C01.template to_vector<int8>());
  aie::store_v(pC + n * kCStrideN + (m + 1) * MMUL::size_C,
               C10.template to_vector<int8>());
  aie::store_v(pC + (n + 1) * kCStrideN + (m + 1) * MMUL::size_C,
               C11.template to_vector<int8>());
}

template <typename MMUL>
static inline void matmul_pack_1x2_streamed(int8 *__restrict pA,
                                            int8 *__restrict pB,
                                            int8 *__restrict pC, unsigned m,
                                            unsigned n) {
  constexpr unsigned kKPackCount = EXTERNAL_K_PACKS;
  constexpr unsigned kActiveMPackCount = EXTERNAL_ACTIVE_M_PACKS;
  constexpr unsigned kCoreMPackCount = EXTERNAL_CORE_M_PACKS;
  constexpr unsigned kAStrideK = kActiveMPackCount * MMUL::size_A;
  constexpr unsigned kBStrideN = kKPackCount * MMUL::size_B;
  constexpr unsigned kCStrideN = EXTERNAL_C_STRIDE_M_PACKS * MMUL::size_C;

  MMUL C00(aie::load_v<MMUL::size_C>(pC + n * kCStrideN + m * MMUL::size_C));
  MMUL C01(
      aie::load_v<MMUL::size_C>(pC + (n + 1) * kCStrideN + m * MMUL::size_C));

  const int8 __aie_dm_resource_a *__restrict pA0 =
      (const int8 __aie_dm_resource_a *)(pA + m * MMUL::size_A);
  const int8 __aie_dm_resource_b *__restrict pB0 =
      (const int8 __aie_dm_resource_b *)(pB + n * kBStrideN);
  const int8 __aie_dm_resource_b *__restrict pB1 =
      (const int8 __aie_dm_resource_b *)(pB + (n + 1) * kBStrideN);

  for (unsigned k = 0; k < kKPackCount; ++k)
    EXTERNAL_K_LOOP_ATTR(EXTERNAL_K_PACKS, EXTERNAL_K_PACKS) {
      aie::vector<int8, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB0);
      aie::vector<int8, MMUL::size_B> B1 = aie::load_v<MMUL::size_B>(pB1);
      aie::vector<int8, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA0);
      pB0 += MMUL::size_B;
      pB1 += MMUL::size_B;
      pA0 += kAStrideK;
      C00.mac(A0, B0);
      C01.mac(A0, B1);
    }

  aie::store_v(pC + n * kCStrideN + m * MMUL::size_C,
               C00.template to_vector<int8>());
  aie::store_v(pC + (n + 1) * kCStrideN + m * MMUL::size_C,
               C01.template to_vector<int8>());
}

template <typename MMUL>
static inline void matmul_pack_2x1_streamed(int8 *__restrict pA,
                                            int8 *__restrict pB,
                                            int8 *__restrict pC, unsigned m,
                                            unsigned n) {
  constexpr unsigned kKPackCount = EXTERNAL_K_PACKS;
  constexpr unsigned kActiveMPackCount = EXTERNAL_ACTIVE_M_PACKS;
  constexpr unsigned kCoreMPackCount = EXTERNAL_CORE_M_PACKS;
  constexpr unsigned kAStrideK = kActiveMPackCount * MMUL::size_A;
  constexpr unsigned kBStrideN = kKPackCount * MMUL::size_B;
  constexpr unsigned kCStrideN = EXTERNAL_C_STRIDE_M_PACKS * MMUL::size_C;

  MMUL C00(aie::load_v<MMUL::size_C>(pC + n * kCStrideN + m * MMUL::size_C));
  MMUL C10(
      aie::load_v<MMUL::size_C>(pC + n * kCStrideN + (m + 1) * MMUL::size_C));

  const int8 __aie_dm_resource_a *__restrict pA0 =
      (const int8 __aie_dm_resource_a *)(pA + m * MMUL::size_A);
  const int8 __aie_dm_resource_a *__restrict pA1 =
      (const int8 __aie_dm_resource_a *)(pA + (m + 1) * MMUL::size_A);
  const int8 __aie_dm_resource_b *__restrict pB0 =
      (const int8 __aie_dm_resource_b *)(pB + n * kBStrideN);

  for (unsigned k = 0; k < kKPackCount; ++k)
    EXTERNAL_K_LOOP_ATTR(EXTERNAL_K_PACKS, EXTERNAL_K_PACKS) {
      aie::vector<int8, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB0);
      pB0 += MMUL::size_B;
      {
        aie::vector<int8, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA0);
        pA0 += kAStrideK;
        C00.mac(A0, B0);
      }
      {
        aie::vector<int8, MMUL::size_A> A1 = aie::load_v<MMUL::size_A>(pA1);
        pA1 += kAStrideK;
        C10.mac(A1, B0);
      }
    }

  aie::store_v(pC + n * kCStrideN + m * MMUL::size_C,
               C00.template to_vector<int8>());
  aie::store_v(pC + n * kCStrideN + (m + 1) * MMUL::size_C,
               C10.template to_vector<int8>());
}

template <typename MMUL>
static inline void matmul_pack_3x2_streamed(int8 *__restrict pA,
                                            int8 *__restrict pB,
                                            int8 *__restrict pC, unsigned m,
                                            unsigned n) {
  constexpr unsigned kKPackCount = EXTERNAL_K_PACKS;
  constexpr unsigned kActiveMPackCount = EXTERNAL_ACTIVE_M_PACKS;
  constexpr unsigned kCoreMPackCount = EXTERNAL_CORE_M_PACKS;
  constexpr unsigned kAStrideK = kActiveMPackCount * MMUL::size_A;
  constexpr unsigned kBStrideN = kKPackCount * MMUL::size_B;
  constexpr unsigned kCStrideN = EXTERNAL_C_STRIDE_M_PACKS * MMUL::size_C;

  MMUL C00(aie::load_v<MMUL::size_C>(pC + n * kCStrideN + m * MMUL::size_C));
  MMUL C01(
      aie::load_v<MMUL::size_C>(pC + (n + 1) * kCStrideN + m * MMUL::size_C));
  MMUL C10(
      aie::load_v<MMUL::size_C>(pC + n * kCStrideN + (m + 1) * MMUL::size_C));
  MMUL C11(aie::load_v<MMUL::size_C>(
      pC + (n + 1) * kCStrideN + (m + 1) * MMUL::size_C));
  MMUL C20(
      aie::load_v<MMUL::size_C>(pC + n * kCStrideN + (m + 2) * MMUL::size_C));
  MMUL C21(aie::load_v<MMUL::size_C>(
      pC + (n + 1) * kCStrideN + (m + 2) * MMUL::size_C));

  const int8 __aie_dm_resource_a *__restrict pA0 =
      (const int8 __aie_dm_resource_a *)(pA + m * MMUL::size_A);
  const int8 __aie_dm_resource_a *__restrict pA1 =
      (const int8 __aie_dm_resource_a *)(pA + (m + 1) * MMUL::size_A);
  const int8 __aie_dm_resource_a *__restrict pA2 =
      (const int8 __aie_dm_resource_a *)(pA + (m + 2) * MMUL::size_A);
  const int8 __aie_dm_resource_b *__restrict pB0 =
      (const int8 __aie_dm_resource_b *)(pB + n * kBStrideN);
  const int8 __aie_dm_resource_b *__restrict pB1 =
      (const int8 __aie_dm_resource_b *)(pB + (n + 1) * kBStrideN);

  for (unsigned k = 0; k < kKPackCount; ++k)
    EXTERNAL_K_LOOP_ATTR(EXTERNAL_K_PACKS, EXTERNAL_K_PACKS) {
      aie::vector<int8, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB0);
      aie::vector<int8, MMUL::size_B> B1 = aie::load_v<MMUL::size_B>(pB1);
      pB0 += MMUL::size_B;
      pB1 += MMUL::size_B;

      {
        aie::vector<int8, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA0);
        pA0 += kAStrideK;
        C00.mac(A0, B0);
        C01.mac(A0, B1);
      }
      {
        aie::vector<int8, MMUL::size_A> A1 = aie::load_v<MMUL::size_A>(pA1);
        pA1 += kAStrideK;
        C10.mac(A1, B0);
        C11.mac(A1, B1);
      }
      {
        aie::vector<int8, MMUL::size_A> A2 = aie::load_v<MMUL::size_A>(pA2);
        pA2 += kAStrideK;
        C20.mac(A2, B0);
        C21.mac(A2, B1);
      }
    }

  aie::store_v(pC + n * kCStrideN + m * MMUL::size_C,
               C00.template to_vector<int8>());
  aie::store_v(pC + (n + 1) * kCStrideN + m * MMUL::size_C,
               C01.template to_vector<int8>());
  aie::store_v(pC + n * kCStrideN + (m + 1) * MMUL::size_C,
               C10.template to_vector<int8>());
  aie::store_v(pC + (n + 1) * kCStrideN + (m + 1) * MMUL::size_C,
               C11.template to_vector<int8>());
  aie::store_v(pC + n * kCStrideN + (m + 2) * MMUL::size_C,
               C20.template to_vector<int8>());
  aie::store_v(pC + (n + 1) * kCStrideN + (m + 2) * MMUL::size_C,
               C21.template to_vector<int8>());
}



static inline void matmul_pack_2x2_native(int8 *__restrict pA,
                                          int8 *__restrict pB,
                                          int8 *__restrict pC, unsigned m,
                                          unsigned n) {
  constexpr unsigned kKPackCount = EXTERNAL_K_PACKS;
  constexpr unsigned kActiveMPackCount = EXTERNAL_ACTIVE_M_PACKS;
  constexpr unsigned kCoreMPackCount = EXTERNAL_CORE_M_PACKS;
  constexpr unsigned kPackBytes = 64;
  constexpr unsigned kAStrideK = kActiveMPackCount * kPackBytes;
  constexpr unsigned kBStrideN = kKPackCount * kPackBytes;
  constexpr unsigned kCStrideN = EXTERNAL_C_STRIDE_M_PACKS * kPackBytes;

  int8 *__restrict pC00 = pC + n * kCStrideN + m * kPackBytes;
  int8 *__restrict pC01 = pC + (n + 1) * kCStrideN + m * kPackBytes;
  int8 *__restrict pC10 = pC + n * kCStrideN + (m + 1) * kPackBytes;
  int8 *__restrict pC11 = pC + (n + 1) * kCStrideN + (m + 1) * kPackBytes;

  v64acc32 C00 = native_load_acc_i8(pC00);
  v64acc32 C01 = native_load_acc_i8(pC01);
  v64acc32 C10 = native_load_acc_i8(pC10);
  v64acc32 C11 = native_load_acc_i8(pC11);

  const int8 __aie_dm_resource_a *__restrict pA0 =
      (const int8 __aie_dm_resource_a *)(pA + m * kPackBytes);
  const int8 __aie_dm_resource_a *__restrict pA1 =
      (const int8 __aie_dm_resource_a *)(pA + (m + 1) * kPackBytes);
  const int8 __aie_dm_resource_b *__restrict pB0 =
      (const int8 __aie_dm_resource_b *)(pB + n * kBStrideN);
  const int8 __aie_dm_resource_b *__restrict pB1 =
      (const int8 __aie_dm_resource_b *)(pB + (n + 1) * kBStrideN);

#define NATIVE_2X2_STEP()                                                     \
  do {                                                                        \
    v64int8 B0 = native_load_i8x64(pB0);                                      \
    v64int8 B1 = native_load_i8x64(pB1);                                      \
    pB0 += kPackBytes;                                                        \
    pB1 += kPackBytes;                                                        \
                                                                               \
    v64int8 A0 = native_load_i8x64(pA0);                                      \
    pA0 += kAStrideK;                                                         \
    C00 = native_mac_i8(C00, A0, B0);                                         \
    C01 = native_mac_i8(C01, A0, B1);                                         \
                                                                               \
    v64int8 A1 = native_load_i8x64(pA1);                                      \
    pA1 += kAStrideK;                                                         \
    C10 = native_mac_i8(C10, A1, B0);                                         \
    C11 = native_mac_i8(C11, A1, B1);                                         \
  } while (false)

#if EXTERNAL_K_PACKS == 9
  NATIVE_2X2_STEP();
  NATIVE_2X2_STEP();
  NATIVE_2X2_STEP();
  NATIVE_2X2_STEP();
  NATIVE_2X2_STEP();
  NATIVE_2X2_STEP();
  NATIVE_2X2_STEP();
  NATIVE_2X2_STEP();
  NATIVE_2X2_STEP();
#elif EXTERNAL_K_PACKS == 18
  for (unsigned kk = 0; kk < 2; ++kk)
    chess_prepare_for_pipelining chess_loop_range(2, 2) {
      NATIVE_2X2_STEP();
      NATIVE_2X2_STEP();
      NATIVE_2X2_STEP();
      NATIVE_2X2_STEP();
      NATIVE_2X2_STEP();
      NATIVE_2X2_STEP();
      NATIVE_2X2_STEP();
      NATIVE_2X2_STEP();
      NATIVE_2X2_STEP();
    }
#else
  for (unsigned k = 0; k < kKPackCount; ++k)
    EXTERNAL_K_LOOP_ATTR(EXTERNAL_K_PACKS, EXTERNAL_K_PACKS) {
      NATIVE_2X2_STEP();
    }
#endif

#undef NATIVE_2X2_STEP

  native_store_acc_i8(pC00, C00);
  native_store_acc_i8(pC01, C01);
  native_store_acc_i8(pC10, C10);
  native_store_acc_i8(pC11, C11);
}

static inline void matmul_pack_2x2_native_atb_ref(int8 *__restrict pA,
                                                  int8 *__restrict pB,
                                                  int8 *__restrict pC,
                                                  unsigned m, unsigned n) {
  constexpr unsigned kActiveMPackCount = EXTERNAL_ACTIVE_M_PACKS;
  constexpr unsigned kPackBytes = 64;
  constexpr unsigned kAStrideK = kActiveMPackCount * kPackBytes;
  constexpr unsigned kBStrideN = EXTERNAL_K_PACKS * kPackBytes;
  constexpr unsigned kCStrideN = EXTERNAL_C_STRIDE_M_PACKS * kPackBytes;

  int8 *__restrict pC00 = pC + n * kCStrideN + m * kPackBytes;
  int8 *__restrict pC01 = pC + (n + 1) * kCStrideN + m * kPackBytes;
  int8 *__restrict pC10 = pC + n * kCStrideN + (m + 1) * kPackBytes;
  int8 *__restrict pC11 = pC + (n + 1) * kCStrideN + (m + 1) * kPackBytes;

  v64acc32 C00 = native_load_acc_i8(pC00);
  v64acc32 C01 = native_load_acc_i8(pC01);
  v64acc32 C10 = native_load_acc_i8(pC10);
  v64acc32 C11 = native_load_acc_i8(pC11);

  const int8 __aie_dm_resource_a *__restrict pA0 =
      (const int8 __aie_dm_resource_a *)(pA + m * kPackBytes);
  const int8 __aie_dm_resource_a *__restrict pA1 =
      (const int8 __aie_dm_resource_a *)(pA + (m + 1) * kPackBytes);
  const int8 __aie_dm_resource_b *__restrict pB0 =
      (const int8 __aie_dm_resource_b *)(pB + n * kBStrideN);
  const int8 __aie_dm_resource_b *__restrict pB1 =
      (const int8 __aie_dm_resource_b *)(pB + (n + 1) * kBStrideN);

#define ATB_REF_STEP(TAG)                                                      \
  do {                                                                        \
    v64int8 A0_##TAG = native_load_i8x64(pA0);                                \
    pA0 += kAStrideK;                                                         \
    v64int8 B0_##TAG = native_load_i8x64(pB0);                                \
    pB0 += kPackBytes;                                                        \
    v64int8 A1_##TAG = native_load_i8x64(pA1);                                \
    pA1 += kAStrideK;                                                         \
    v64int8 B1_##TAG = native_load_i8x64(pB1);                                \
    pB1 += kPackBytes;                                                        \
    C00 = native_mac_i8(C00, A0_##TAG, B0_##TAG);                             \
    C01 = native_mac_i8(C01, A0_##TAG, B1_##TAG);                             \
    C10 = native_mac_i8(C10, A1_##TAG, B0_##TAG);                             \
    C11 = native_mac_i8(C11, A1_##TAG, B1_##TAG);                             \
  } while (false)

  for (unsigned kk = 0; kk < 2; ++kk)
    chess_prepare_for_pipelining chess_loop_range(2, 2) {
      ATB_REF_STEP(ping);
      ATB_REF_STEP(pong);
      ATB_REF_STEP(ping);
      ATB_REF_STEP(pong);
      ATB_REF_STEP(ping);
      ATB_REF_STEP(pong);
      ATB_REF_STEP(ping);
      ATB_REF_STEP(pong);
      ATB_REF_STEP(ping);
    }

#undef ATB_REF_STEP

  native_store_acc_i8(pC00, C00);
  native_store_acc_i8(pC01, C01);
  native_store_acc_i8(pC10, C10);
  native_store_acc_i8(pC11, C11);
}

static inline void matmul_pack_3x2_native(int8 *__restrict pA,
                                          int8 *__restrict pB,
                                          int8 *__restrict pC, unsigned m,
                                          unsigned n) {
  constexpr unsigned kKPackCount = EXTERNAL_K_PACKS;
  constexpr unsigned kActiveMPackCount = EXTERNAL_ACTIVE_M_PACKS;
  constexpr unsigned kCoreMPackCount = EXTERNAL_CORE_M_PACKS;
  constexpr unsigned kPackBytes = 64;
  constexpr unsigned kAStrideK = kActiveMPackCount * kPackBytes;
  constexpr unsigned kBStrideN = kKPackCount * kPackBytes;
  constexpr unsigned kCStrideN = EXTERNAL_C_STRIDE_M_PACKS * kPackBytes;

  int8 *__restrict pC00 = pC + n * kCStrideN + m * kPackBytes;
  int8 *__restrict pC01 = pC + (n + 1) * kCStrideN + m * kPackBytes;
  int8 *__restrict pC10 = pC + n * kCStrideN + (m + 1) * kPackBytes;
  int8 *__restrict pC11 = pC + (n + 1) * kCStrideN + (m + 1) * kPackBytes;
  int8 *__restrict pC20 = pC + n * kCStrideN + (m + 2) * kPackBytes;
  int8 *__restrict pC21 = pC + (n + 1) * kCStrideN + (m + 2) * kPackBytes;

  v64acc32 C00 = native_load_acc_i8(pC00);
  v64acc32 C01 = native_load_acc_i8(pC01);
  v64acc32 C10 = native_load_acc_i8(pC10);
  v64acc32 C11 = native_load_acc_i8(pC11);
  v64acc32 C20 = native_load_acc_i8(pC20);
  v64acc32 C21 = native_load_acc_i8(pC21);

  const int8 __aie_dm_resource_a *__restrict pA0 =
      (const int8 __aie_dm_resource_a *)(pA + m * kPackBytes);
  const int8 __aie_dm_resource_a *__restrict pA1 =
      (const int8 __aie_dm_resource_a *)(pA + (m + 1) * kPackBytes);
  const int8 __aie_dm_resource_a *__restrict pA2 =
      (const int8 __aie_dm_resource_a *)(pA + (m + 2) * kPackBytes);
  const int8 __aie_dm_resource_b *__restrict pB0 =
      (const int8 __aie_dm_resource_b *)(pB + n * kBStrideN);
  const int8 __aie_dm_resource_b *__restrict pB1 =
      (const int8 __aie_dm_resource_b *)(pB + (n + 1) * kBStrideN);

  for (unsigned k = 0; k < kKPackCount; ++k)
    EXTERNAL_K_LOOP_ATTR(EXTERNAL_K_PACKS, EXTERNAL_K_PACKS) {
      v64int8 B0 = native_load_i8x64(pB0);
      v64int8 B1 = native_load_i8x64(pB1);
      pB0 += kPackBytes;
      pB1 += kPackBytes;

      v64int8 A0 = native_load_i8x64(pA0);
      pA0 += kAStrideK;
      C00 = native_mac_i8(C00, A0, B0);
      C01 = native_mac_i8(C01, A0, B1);

      v64int8 A1 = native_load_i8x64(pA1);
      pA1 += kAStrideK;
      C10 = native_mac_i8(C10, A1, B0);
      C11 = native_mac_i8(C11, A1, B1);

      v64int8 A2 = native_load_i8x64(pA2);
      pA2 += kAStrideK;
      C20 = native_mac_i8(C20, A2, B0);
      C21 = native_mac_i8(C21, A2, B1);
    }

  native_store_acc_i8(pC00, C00);
  native_store_acc_i8(pC01, C01);
  native_store_acc_i8(pC10, C10);
  native_store_acc_i8(pC11, C11);
  native_store_acc_i8(pC20, C20);
  native_store_acc_i8(pC21, C21);
}

template <typename MMUL>
static inline void matmul_pack_2x3_streamed(int8 *__restrict pA,
                                            int8 *__restrict pB,
                                            int8 *__restrict pC, unsigned m,
                                            unsigned n) {
  constexpr unsigned kKPackCount = EXTERNAL_K_PACKS;
  constexpr unsigned kActiveMPackCount = EXTERNAL_ACTIVE_M_PACKS;
  constexpr unsigned kCoreMPackCount = EXTERNAL_CORE_M_PACKS;
  constexpr unsigned kAStrideK = kActiveMPackCount * MMUL::size_A;
  constexpr unsigned kBStrideN = kKPackCount * MMUL::size_B;
  constexpr unsigned kCStrideN = EXTERNAL_C_STRIDE_M_PACKS * MMUL::size_C;

  MMUL C00(aie::load_v<MMUL::size_C>(pC + n * kCStrideN + m * MMUL::size_C));
  MMUL C01(
      aie::load_v<MMUL::size_C>(pC + (n + 1) * kCStrideN + m * MMUL::size_C));
  MMUL C02(
      aie::load_v<MMUL::size_C>(pC + (n + 2) * kCStrideN + m * MMUL::size_C));
  MMUL C10(
      aie::load_v<MMUL::size_C>(pC + n * kCStrideN + (m + 1) * MMUL::size_C));
  MMUL C11(aie::load_v<MMUL::size_C>(
      pC + (n + 1) * kCStrideN + (m + 1) * MMUL::size_C));
  MMUL C12(aie::load_v<MMUL::size_C>(
      pC + (n + 2) * kCStrideN + (m + 1) * MMUL::size_C));

  const int8 __aie_dm_resource_a *__restrict pA0 =
      (const int8 __aie_dm_resource_a *)(pA + m * MMUL::size_A);
  const int8 __aie_dm_resource_a *__restrict pA1 =
      (const int8 __aie_dm_resource_a *)(pA + (m + 1) * MMUL::size_A);
  const int8 __aie_dm_resource_b *__restrict pB0 =
      (const int8 __aie_dm_resource_b *)(pB + n * kBStrideN);
  const int8 __aie_dm_resource_b *__restrict pB1 =
      (const int8 __aie_dm_resource_b *)(pB + (n + 1) * kBStrideN);
  const int8 __aie_dm_resource_b *__restrict pB2 =
      (const int8 __aie_dm_resource_b *)(pB + (n + 2) * kBStrideN);

  for (unsigned k = 0; k < kKPackCount; ++k)
    EXTERNAL_K_LOOP_ATTR(EXTERNAL_K_PACKS, EXTERNAL_K_PACKS) {
      aie::vector<int8, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB0);
      aie::vector<int8, MMUL::size_B> B1 = aie::load_v<MMUL::size_B>(pB1);
      aie::vector<int8, MMUL::size_B> B2 = aie::load_v<MMUL::size_B>(pB2);
      pB0 += MMUL::size_B;
      pB1 += MMUL::size_B;
      pB2 += MMUL::size_B;

      {
        aie::vector<int8, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA0);
        pA0 += kAStrideK;
        C00.mac(A0, B0);
        C01.mac(A0, B1);
        C02.mac(A0, B2);
      }
      {
        aie::vector<int8, MMUL::size_A> A1 = aie::load_v<MMUL::size_A>(pA1);
        pA1 += kAStrideK;
        C10.mac(A1, B0);
        C11.mac(A1, B1);
        C12.mac(A1, B2);
      }
    }

  aie::store_v(pC + n * kCStrideN + m * MMUL::size_C,
               C00.template to_vector<int8>());
  aie::store_v(pC + (n + 1) * kCStrideN + m * MMUL::size_C,
               C01.template to_vector<int8>());
  aie::store_v(pC + (n + 2) * kCStrideN + m * MMUL::size_C,
               C02.template to_vector<int8>());
  aie::store_v(pC + n * kCStrideN + (m + 1) * MMUL::size_C,
               C10.template to_vector<int8>());
  aie::store_v(pC + (n + 1) * kCStrideN + (m + 1) * MMUL::size_C,
               C11.template to_vector<int8>());
  aie::store_v(pC + (n + 2) * kCStrideN + (m + 1) * MMUL::size_C,
               C12.template to_vector<int8>());
}

template <typename MMUL>
static inline void matmul_pack_4x2_streamed(int8 *__restrict pA,
                                            int8 *__restrict pB,
                                            int8 *__restrict pC, unsigned m,
                                            unsigned n) {
  constexpr unsigned kKPackCount = EXTERNAL_K_PACKS;
  constexpr unsigned kActiveMPackCount = EXTERNAL_ACTIVE_M_PACKS;
  constexpr unsigned kCoreMPackCount = EXTERNAL_CORE_M_PACKS;
  constexpr unsigned kAStrideK = kActiveMPackCount * MMUL::size_A;
  constexpr unsigned kBStrideN = kKPackCount * MMUL::size_B;
  constexpr unsigned kCStrideN = EXTERNAL_C_STRIDE_M_PACKS * MMUL::size_C;

  MMUL C00(aie::load_v<MMUL::size_C>(pC + n * kCStrideN + m * MMUL::size_C));
  MMUL C01(
      aie::load_v<MMUL::size_C>(pC + (n + 1) * kCStrideN + m * MMUL::size_C));
  MMUL C10(
      aie::load_v<MMUL::size_C>(pC + n * kCStrideN + (m + 1) * MMUL::size_C));
  MMUL C11(aie::load_v<MMUL::size_C>(
      pC + (n + 1) * kCStrideN + (m + 1) * MMUL::size_C));
  MMUL C20(
      aie::load_v<MMUL::size_C>(pC + n * kCStrideN + (m + 2) * MMUL::size_C));
  MMUL C21(aie::load_v<MMUL::size_C>(
      pC + (n + 1) * kCStrideN + (m + 2) * MMUL::size_C));
  MMUL C30(
      aie::load_v<MMUL::size_C>(pC + n * kCStrideN + (m + 3) * MMUL::size_C));
  MMUL C31(aie::load_v<MMUL::size_C>(
      pC + (n + 1) * kCStrideN + (m + 3) * MMUL::size_C));

  const int8 __aie_dm_resource_a *__restrict pA0 =
      (const int8 __aie_dm_resource_a *)(pA + m * MMUL::size_A);
  const int8 __aie_dm_resource_a *__restrict pA1 =
      (const int8 __aie_dm_resource_a *)(pA + (m + 1) * MMUL::size_A);
  const int8 __aie_dm_resource_a *__restrict pA2 =
      (const int8 __aie_dm_resource_a *)(pA + (m + 2) * MMUL::size_A);
  const int8 __aie_dm_resource_a *__restrict pA3 =
      (const int8 __aie_dm_resource_a *)(pA + (m + 3) * MMUL::size_A);
  const int8 __aie_dm_resource_b *__restrict pB0 =
      (const int8 __aie_dm_resource_b *)(pB + n * kBStrideN);
  const int8 __aie_dm_resource_b *__restrict pB1 =
      (const int8 __aie_dm_resource_b *)(pB + (n + 1) * kBStrideN);

  for (unsigned k = 0; k < kKPackCount; ++k)
    EXTERNAL_K_LOOP_ATTR(EXTERNAL_K_PACKS, EXTERNAL_K_PACKS) {
      aie::vector<int8, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB0);
      aie::vector<int8, MMUL::size_B> B1 = aie::load_v<MMUL::size_B>(pB1);
      pB0 += MMUL::size_B;
      pB1 += MMUL::size_B;

      {
        aie::vector<int8, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA0);
        pA0 += kAStrideK;
        C00.mac(A0, B0);
        C01.mac(A0, B1);
      }
      {
        aie::vector<int8, MMUL::size_A> A1 = aie::load_v<MMUL::size_A>(pA1);
        pA1 += kAStrideK;
        C10.mac(A1, B0);
        C11.mac(A1, B1);
      }
      {
        aie::vector<int8, MMUL::size_A> A2 = aie::load_v<MMUL::size_A>(pA2);
        pA2 += kAStrideK;
        C20.mac(A2, B0);
        C21.mac(A2, B1);
      }
      {
        aie::vector<int8, MMUL::size_A> A3 = aie::load_v<MMUL::size_A>(pA3);
        pA3 += kAStrideK;
        C30.mac(A3, B0);
        C31.mac(A3, B1);
      }
    }

  aie::store_v(pC + n * kCStrideN + m * MMUL::size_C,
               C00.template to_vector<int8>());
  aie::store_v(pC + (n + 1) * kCStrideN + m * MMUL::size_C,
               C01.template to_vector<int8>());
  aie::store_v(pC + n * kCStrideN + (m + 1) * MMUL::size_C,
               C10.template to_vector<int8>());
  aie::store_v(pC + (n + 1) * kCStrideN + (m + 1) * MMUL::size_C,
               C11.template to_vector<int8>());
  aie::store_v(pC + n * kCStrideN + (m + 2) * MMUL::size_C,
               C20.template to_vector<int8>());
  aie::store_v(pC + (n + 1) * kCStrideN + (m + 2) * MMUL::size_C,
               C21.template to_vector<int8>());
  aie::store_v(pC + n * kCStrideN + (m + 3) * MMUL::size_C,
               C30.template to_vector<int8>());
  aie::store_v(pC + (n + 1) * kCStrideN + (m + 3) * MMUL::size_C,
               C31.template to_vector<int8>());
}

template <unsigned kBlockM, unsigned kBlockN, typename MMUL>
static inline void matmul_diagnostic_block(int8 *__restrict pA,
                                           int8 *__restrict pB,
                                           int8 *__restrict pC) {
  constexpr unsigned kKPackCount = EXTERNAL_K_PACKS;
  constexpr unsigned kActiveMPackCount = EXTERNAL_ACTIVE_M_PACKS;
  constexpr unsigned kCoreMPackCount = EXTERNAL_CORE_M_PACKS;
  constexpr unsigned kNPackCount = EXTERNAL_CORE_N_PACKS;
  constexpr unsigned kAStrideK = kActiveMPackCount * MMUL::size_A;
  constexpr unsigned kBStrideN = kKPackCount * MMUL::size_B;
  constexpr unsigned kCStrideN = EXTERNAL_C_STRIDE_M_PACKS * MMUL::size_C;

  static_assert(kBlockM >= 2 && kBlockM <= 4);
  static_assert(kBlockN >= 2 && kBlockN <= 3);
  static_assert(kActiveMPackCount % kBlockM == 0);
  static_assert(kNPackCount % kBlockN == 0);

  for (unsigned m = 0; m < kActiveMPackCount; m += kBlockM)
    EXTERNAL_M_LOOP_ATTR(kActiveMPackCount / kBlockM,
                         kActiveMPackCount / kBlockM) {
      for (unsigned n = 0; n < kNPackCount; n += kBlockN)
        EXTERNAL_N_LOOP_ATTR(kNPackCount / kBlockN, kNPackCount / kBlockN) {
          int8 *__restrict pCBlock[kBlockM][kBlockN];
          MMUL C[kBlockM][kBlockN];
          const int8 __aie_dm_resource_a *__restrict pABlock[kBlockM];
          const int8 __aie_dm_resource_b *__restrict pBBlock[kBlockN];

          for (unsigned mi = 0; mi < kBlockM; ++mi)
            chess_unroll_loop() {
              pABlock[mi] = (const int8 __aie_dm_resource_a *)(
                  pA + (m + mi) * MMUL::size_A);
              for (unsigned ni = 0; ni < kBlockN; ++ni)
                chess_unroll_loop() {
                  pCBlock[mi][ni] =
                      pC + (n + ni) * kCStrideN + (m + mi) * MMUL::size_C;
                  C[mi][ni] =
                      MMUL(aie::load_v<MMUL::size_C>(pCBlock[mi][ni]));
                }
            }

          for (unsigned ni = 0; ni < kBlockN; ++ni)
            chess_unroll_loop() {
              pBBlock[ni] =
                  (const int8 __aie_dm_resource_b *)(pB + (n + ni) * kBStrideN);
            }

          for (unsigned k = 0; k < kKPackCount; ++k)
            EXTERNAL_K_LOOP_ATTR(EXTERNAL_K_PACKS, EXTERNAL_K_PACKS) {
              aie::vector<int8, MMUL::size_A> A[kBlockM];
              aie::vector<int8, MMUL::size_B> B[kBlockN];

              for (unsigned mi = 0; mi < kBlockM; ++mi)
                chess_unroll_loop() {
                  A[mi] = aie::load_v<MMUL::size_A>(pABlock[mi]);
                  pABlock[mi] += kAStrideK;
                }
              for (unsigned ni = 0; ni < kBlockN; ++ni)
                chess_unroll_loop() {
                  B[ni] = aie::load_v<MMUL::size_B>(pBBlock[ni]);
                  pBBlock[ni] += MMUL::size_B;
                }
              for (unsigned mi = 0; mi < kBlockM; ++mi)
                chess_unroll_loop() {
                  for (unsigned ni = 0; ni < kBlockN; ++ni)
                    chess_unroll_loop() { C[mi][ni].mac(A[mi], B[ni]); }
                }
            }

          for (unsigned mi = 0; mi < kBlockM; ++mi)
            chess_unroll_loop() {
              for (unsigned ni = 0; ni < kBlockN; ++ni)
                chess_unroll_loop() {
                  aie::store_v(pCBlock[mi][ni],
                               C[mi][ni].template to_vector<int8>());
                }
            }
        }
    }
}

extern "C" void matmul_i8_i8_i8_acc32_strix(int8 *__restrict pA,
                                             int8 *__restrict pB,
                                             int8 *__restrict pC) {
  using MMUL = aie::mmul<8, 8, 8, int8, int8, acc32>;

  static_assert(MMUL::size_A == 64);
  static_assert(MMUL::size_B == 64);
  static_assert(MMUL::size_C == 64);

  constexpr unsigned kKPackCount = EXTERNAL_K_PACKS;
  constexpr unsigned kBlockM = EXTERNAL_BLOCK_M;
  constexpr unsigned kBlockN = EXTERNAL_BLOCK_N;
  constexpr unsigned kActiveMPackCount = EXTERNAL_ACTIVE_M_PACKS;
  constexpr unsigned kCoreMPackCount = EXTERNAL_CORE_M_PACKS;
  constexpr unsigned kNPackCount = EXTERNAL_CORE_N_PACKS;
  constexpr unsigned kAStrideK = kActiveMPackCount * MMUL::size_A;
  constexpr unsigned kBStrideN = kKPackCount * MMUL::size_B;
  constexpr unsigned kCStrideN = EXTERNAL_C_STRIDE_M_PACKS * MMUL::size_C;
  (void)kAStrideK;
  (void)kBStrideN;
  (void)kCStrideN;

  static_assert(kKPackCount > 0);
  static_assert(kKPackCount <= 18, "EXTERNAL_K_PACKS exceeds L1 budget");
  static_assert(kCoreMPackCount > 0);
  static_assert(kActiveMPackCount > 0);
  static_assert(kActiveMPackCount <= kCoreMPackCount,
                "EXTERNAL_ACTIVE_M_PACKS must fit within EXTERNAL_CORE_M_PACKS");
  static_assert(EXTERNAL_C_STRIDE_M_PACKS >= kActiveMPackCount,
                "C stride must cover active M packs");
  static_assert(kNPackCount > 0);
  static_assert(kBlockM >= 2 && kBlockM <= 4,
                "EXTERNAL_BLOCK_M must be 2, 3, or 4");
  static_assert(kBlockN >= 2 && kBlockN <= 3,
                "EXTERNAL_BLOCK_N must be 2 or 3");
  static_assert(kBlockM != 4 || kBlockN == 2,
                "EXTERNAL_BLOCK_M=4 currently supports EXTERNAL_BLOCK_N=2");
  static_assert((kBlockM == 4 && kBlockN == 2 && kActiveMPackCount == 18) ||
                kActiveMPackCount % kBlockM == 0,
                "active M packs must be divisible by EXTERNAL_BLOCK_M");
  static_assert(kNPackCount % kBlockN == 0);
#if EXTERNAL_KERNEL_STYLE == EXTERNAL_KERNEL_STYLE_NATIVE_MMUL_ATB_REF
  static_assert(EXTERNAL_BLOCK_M == 2 && EXTERNAL_BLOCK_N == 2,
                "native-mmul-atb-ref is only implemented for the ATB v2 2x2 block");
  static_assert(EXTERNAL_ACTIVE_M_PACKS == 6,
                "native-mmul-atb-ref expects ATB v2 active M packs = 6");
  static_assert(EXTERNAL_CORE_M_PACKS == 18 && EXTERNAL_CORE_N_PACKS == 18,
                "native-mmul-atb-ref expects the ATB v2 18x18 core tile");
  static_assert(EXTERNAL_C_STRIDE_M_PACKS == EXTERNAL_CORE_M_PACKS,
                "native-mmul-atb-ref expects full-core C stride");
  static_assert(EXTERNAL_ATB_C_OFFSET,
                "native-mmul-atb-ref is only for ATB v2 full-C offset scheduling");
#endif
#if EXTERNAL_ATB_C_OFFSET
  static_assert(kCoreMPackCount % kActiveMPackCount == 0,
                "ATB C offset requires active M packs to divide core M packs");
  static_assert(EXTERNAL_C_STRIDE_M_PACKS == kCoreMPackCount,
                "ATB C offset requires full-core C stride");
  constexpr unsigned kAtbActiveMBands = kCoreMPackCount / kActiveMPackCount;
  static unsigned atb_active_m_band = 0;
  pC += atb_active_m_band * kActiveMPackCount * MMUL::size_C;
  atb_active_m_band =
      (atb_active_m_band + 1 == kAtbActiveMBands) ? 0 : atb_active_m_band + 1;
#endif

  aie::set_saturation(aie::saturation_mode::none);

  if constexpr (kBlockM == 3 && kBlockN == 2) {
    for (unsigned m = 0; m < kActiveMPackCount; m += 3)
      EXTERNAL_M_LOOP_ATTR(kActiveMPackCount / 3, kActiveMPackCount / 3) {
        for (unsigned n = 0; n < kNPackCount; n += 2)
          EXTERNAL_N_LOOP_ATTR(kNPackCount / 2, kNPackCount / 2) {
            if constexpr (EXTERNAL_KERNEL_STYLE ==
                          EXTERNAL_KERNEL_STYLE_HAND_SCHEDULED) {
              matmul_pack_2x2_streamed<MMUL>(pA, pB, pC, m, n);
              matmul_pack_1x2_streamed<MMUL>(pA, pB, pC, m + 2, n);
            } else if constexpr (EXTERNAL_KERNEL_STYLE ==
                                     EXTERNAL_KERNEL_STYLE_NATIVE_MMUL) {
              matmul_pack_3x2_native(pA, pB, pC, m, n);
            } else {
              matmul_pack_3x2_streamed<MMUL>(pA, pB, pC, m, n);
            }
          }
      }
  } else if constexpr (kBlockM == 2 && kBlockN == 3) {
    for (unsigned m = 0; m < kActiveMPackCount; m += 2)
      EXTERNAL_M_LOOP_ATTR(kActiveMPackCount / 2, kActiveMPackCount / 2) {
        for (unsigned n = 0; n < kNPackCount; n += 3)
          EXTERNAL_N_LOOP_ATTR(kNPackCount / 3, kNPackCount / 3) {
            if constexpr (EXTERNAL_KERNEL_STYLE ==
                          EXTERNAL_KERNEL_STYLE_HAND_SCHEDULED) {
              matmul_pack_2x2_streamed<MMUL>(pA, pB, pC, m, n);
              matmul_pack_2x1_streamed<MMUL>(pA, pB, pC, m, n + 2);
            } else {
              matmul_pack_2x3_streamed<MMUL>(pA, pB, pC, m, n);
            }
          }
      }
  } else if constexpr (kBlockM == 4 && kBlockN == 2) {
    for (unsigned m = 0; m < 16; m += 4)
      EXTERNAL_M_LOOP_ATTR(4, 4) {
        for (unsigned n = 0; n < kNPackCount; n += 2)
          EXTERNAL_N_LOOP_ATTR(kNPackCount / 2, kNPackCount / 2) {
            if constexpr (EXTERNAL_KERNEL_STYLE ==
                          EXTERNAL_KERNEL_STYLE_HAND_SCHEDULED) {
              matmul_pack_2x2_streamed<MMUL>(pA, pB, pC, m, n);
              matmul_pack_2x2_streamed<MMUL>(pA, pB, pC, m + 2, n);
            } else {
              matmul_pack_4x2_streamed<MMUL>(pA, pB, pC, m, n);
            }
          }
      }
    for (unsigned n = 0; n < kNPackCount; n += 2)
      EXTERNAL_N_LOOP_ATTR(kNPackCount / 2, kNPackCount / 2) {
        matmul_pack_2x2_streamed<MMUL>(pA, pB, pC, 16, n);
      }
  } else if constexpr (kBlockM == 2 && kBlockN == 2) {
    for (unsigned m = 0; m < kActiveMPackCount; m += 2)
      EXTERNAL_M_LOOP_ATTR(kActiveMPackCount / 2, kActiveMPackCount / 2) {
        for (unsigned n = 0; n < kNPackCount; n += 2)
          EXTERNAL_N_LOOP_ATTR(kNPackCount / 2, kNPackCount / 2) {
            if constexpr (EXTERNAL_KERNEL_STYLE ==
                          EXTERNAL_KERNEL_STYLE_NATIVE_MMUL_ATB_REF) {
              matmul_pack_2x2_native_atb_ref(pA, pB, pC, m, n);
            } else if constexpr (EXTERNAL_KERNEL_STYLE ==
                                 EXTERNAL_KERNEL_STYLE_NATIVE_MMUL) {
              matmul_pack_2x2_native(pA, pB, pC, m, n);
            } else {
              matmul_pack_2x2_streamed<MMUL>(pA, pB, pC, m, n);
            }
          }
      }
  } else {
    matmul_diagnostic_block<kBlockM, kBlockN, MMUL>(pA, pB, pC);
  }
}
