//===- mv_int4_bf16_r64.cc - the bf16 GEMV body at r=64, as a control -----===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// PROTOTYPE CONTROL (queue item 23). Nothing here is on a shipped path.
//
// `mv_int4_bf16.cc`'s `matvec_int4_bf16_impl` instantiated at inner vector
// width r = 64 explicitly, with its own entry symbols. The arithmetic is
// character-for-character the shipped kernel's; only the vector width
// differs. (Since the DIM_R predicate landed, the shipped file itself takes
// r = 64 where gs % 64 == 0; this file remains the standalone control with
// distinct symbols, so an A/B can link both widths into one session.)
//
// Why this control exists. At r = 32 the weight load is
// `aie::load_v<16>` -- 16 bytes -- and Peano emits `vldb.128` + a separate
// `vunpack`. At r = 64 the same source expression becomes a single fused
// `vldb.unpack`, because the load unit's int4->int8 unpack mode wants a
// 32-byte source. Measured on one gs=128 group (study commit 20fad8c8):
//
//     r = 32 (pre-r64 shipped) : 47 vector ops, 0 vldb.unpack, 4 vunpack
//     r = 64                   : 34 vector ops, 2 vldb.unpack, 0 vunpack
//
// So 13 of the 19 ops the integer kernel saves are available to the bf16
// kernel too, for a one-token change and no format, accuracy or layout
// consequence. Without this arm the item would credit integer compute with a
// win that is really a vector-width choice, which is the whole reason it is
// measured rather than argued.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_M
#define DIM_M 8
#endif
#ifndef DIM_K
#define DIM_K 1024
#endif
#ifndef DIM_GS
#define DIM_GS 128
#endif

static_assert(DIM_K % DIM_GS == 0, "DIM_K must be a multiple of DIM_GS");
static_assert(DIM_GS % 64 == 0,
              "this control requires gs to be a multiple of 64");

template <unsigned m, unsigned k, unsigned gs, unsigned r>
static void
matvec_int4_bf16_r_impl(uint8_t *__restrict a_q, bfloat16 *__restrict a_s,
                        uint8_t *__restrict a_z, bfloat16 *__restrict b,
                        bfloat16 *__restrict c) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  static_assert(gs % r == 0, "group size must be multiple of inner vector r");
  constexpr unsigned NSUB = gs / r;

  for (unsigned row = 0; row < m; row++) {
    aie::accum<accfloat, r> acc;
    acc.from_vector(aie::zeros<float, r>());
    const uint8_t *__restrict aq = a_q + row * (k / 2);

    for (unsigned g = 0; g < k / gs; g++) {
      aie::vector<int8, r> zv =
          aie::broadcast<int8, r>((int8_t)a_z[g * m + row]);
      bfloat16 sa = a_s[g * m + row];

      aie::accum<accfloat, r> g_acc;
      g_acc.from_vector(aie::zeros<float, r>());

#pragma clang loop unroll(full)
      for (unsigned i = 0; i < NSUB; i++) {
        const unsigned off = (g * gs + i * r) / 2;
        aie::vector<uint8, r / 2> packed = aie::load_v<r / 2>(aq + off);
        aie::vector<int8, r> w_int8 =
            packed.template cast_to<uint4>().template unpack_sign<int8>(false);
        w_int8 = aie::sub(w_int8, zv);
        aie::vector<bfloat16, r> w_bf16 = aie::to_float<bfloat16>(w_int8, 0);
        aie::vector<bfloat16, r> b_vec = aie::load_v<r>(b + g * gs + i * r);
        g_acc = aie::mac(g_acc, w_bf16, b_vec);
      }

      aie::vector<bfloat16, r> g_bf16 = g_acc.template to_vector<bfloat16>();
      acc = aie::mac(acc, g_bf16, sa);
    }

    float s = aie::reduce_add(acc.template to_vector<float>());
    c[row] = (bfloat16)((float)c[row] + s);
  }
}

extern "C" {

void matvec_int4_bf16_packed_r64(uint8_t *packed, bfloat16 *b, bfloat16 *c) {
  constexpr unsigned Q_BYTES = DIM_M * (DIM_K / 2);
  constexpr unsigned S_BYTES = (DIM_K / DIM_GS) * DIM_M * 2;
  uint8_t *a_q = packed;
  bfloat16 *a_s = reinterpret_cast<bfloat16 *>(packed + Q_BYTES);
  uint8_t *a_z = packed + Q_BYTES + S_BYTES;
  matvec_int4_bf16_r_impl<DIM_M, DIM_K, DIM_GS, 64>(a_q, a_s, a_z, b, c);
}

void zero_vectorized_bf16_r64(bfloat16 *c) {
  for (unsigned i = 0; i < DIM_M; i++)
    c[i] = (bfloat16)0.0f;
}

} // extern "C"
