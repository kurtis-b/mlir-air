//===- mv_int4_int8.cc - AWQ uint4 weight x int8 activation matvec ---------===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// PROTOTYPE (queue item 23). Nothing here is on a shipped path.
//
// The integer-compute counterpart of `mv_int4_bf16.cc`'s GEMV. Same packed
// Q/S/Z BO, byte for byte; same tile shape; same entry signature. The ONLY
// thing that changes is the arithmetic:
//
//   mv_int4_bf16.cc :  unpack nibbles -> subtract z -> to_float<bfloat16>
//                      -> aie::mac(accfloat, bf16, bf16)
//   this file       :  vldb.unpack (nibbles widen in the LOAD unit)
//                      -> aie::mac(acc32, int8, int8)
//                      -> acc32 -> bf16 (native narrowing) -> scale fold
//
// and the activation is dynamically quantised to int8 per group on device,
// which is per-token work the bf16 path does not do.
//
// Why this is faster, measured by instruction count on AIE2P (see
// results/item23-int-gemv-20260826/ISA_FINDINGS.md, probe_group.dis): one
// gs=128 group costs 47 vector ops in the bf16 kernel and 16-19 here. The MAC
// itself is NOT the win -- elementwise int8 and elementwise bf16 are both 64
// MACs/instruction on AIE2P. The win is (i) `vldb.unpack` folding the int4->
// int8 widening into the load, (ii) deleting the int8->bf16 conversion, which
// is 17 of the bf16 kernel's 47 ops, and (iii) the per-group fold staying
// cheap: a real SRS acc32->int16 plus the same int->bf16 widening the bf16
// kernel already uses. (`acc32 -> bf16` LOOKS like a 1-instruction fold and
// is not value-correct -- see the comment at the fold below.)
//
// Entries (all take the SAME (packed, b_bf16, c_bf16) signature as the bf16
// kernel, so no builder change is needed to swap one in):
//
//   matvec_int4_int8_packed        - zero point subtracted in the inner loop.
//                                    Numerically the closest match to the bf16
//                                    kernel: the value the group accumulator
//                                    truncates to bf16 is the same centred
//                                    (q - z) * b it truncates there.
//   matvec_int4_int8_packed_zhoist - zero point hoisted out of the inner loop
//                                    via  sum (q-z)b = sum qb - z sum b.
//                                    3 fewer ops per group; the group
//                                    accumulator now holds the UNCENTRED
//                                    sum q*b, ~1.9x larger in magnitude, so
//                                    its bf16 truncation costs ~1.9x more
//                                    absolute error. Priced, not assumed.
//   matvec_int4_int8_packed_noquant- TIMING ABLATION ONLY. Skips the activation
//                                    quantisation and reinterprets the bf16 B
//                                    buffer's bytes as int8. The result is
//                                    NUMERICALLY MEANINGLESS by construction
//                                    and must never be used for a correctness
//                                    claim; it exists solely to price the
//                                    quantisation on device.
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
static_assert(DIM_GS % 64 == 0, "DIM_GS must be a multiple of the 64-lane int8 vector");

// Vector width of the integer MAC. 64 is the native int8 elementwise width on
// AIE2P (`mac_elem_64_conf`, aie_api/detail/aie2p/mul_acc32.hpp:70); the bf16
// kernel runs at 32 because a 64-lane bf16 vector is 128 B and it needs two
// loads per MAC.
static constexpr unsigned R = 64;

// ---------------------------------------------------------------------------
// Dynamic int8 activation quantisation, per group of DIM_GS.
//   s_b[g] = max_{k in g} |b[k]| / 127        (bf16)
//   bq[k]  = round(b[k] / s_b[g])             (int8, conv_even)
//   bsum[g]= sum_{k in g} bq[k]               (int32, for the zero-point fold)
// Called once per kernel call and shared by all DIM_M rows.
// ---------------------------------------------------------------------------
template <unsigned k, unsigned gs>
static void quantize_activation(const bfloat16 *__restrict b,
                                int8_t *__restrict bq,
                                bfloat16 *__restrict sb,
                                int32_t *__restrict bsum) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  constexpr unsigned NSUB = gs / R;
  const aie::vector<int8, R> ones = aie::broadcast<int8, R>((int8_t)1);

  for (unsigned g = 0; g < k / gs; g++) {
    // max |b| over the group
    bfloat16 mx = (bfloat16)0.0f;
    for (unsigned i = 0; i < NSUB; i++) {
      aie::vector<bfloat16, R> v = aie::load_v<R>(b + g * gs + i * R);
      bfloat16 m = aie::reduce_max(aie::abs(v));
      mx = (float)m > (float)mx ? m : mx;
    }
    float s = (float)mx * (1.0f / 127.0f);
    float inv = ((float)mx == 0.0f) ? 0.0f : (127.0f / (float)mx);
    sb[g] = (bfloat16)s;

    aie::accum<acc32, R> sacc;
    sacc.from_vector(aie::zeros<int32, R>());
    for (unsigned i = 0; i < NSUB; i++) {
      aie::vector<bfloat16, R> v = aie::load_v<R>(b + g * gs + i * R);
      aie::accum<accfloat, R> sc = aie::mul(v, (bfloat16)inv);
      // TRAP, same class as the fold below: `sc.to_vector<int8>(0)` takes an
      // accum<accfloat> through accum.hpp's get_srs<int8>, i.e. the INTEGER
      // SRS `srs_to_v64int8`, on a FLOAT accumulator. It compiles (with only a
      // deprecation warning about srs_to_v64int8) and returns garbage/zero;
      // measured on device the whole GEMV came back ~0 (devq 657 self-gate).
      // The tree's proven route is FP accumulator -> bf16 vector -> to_fixed,
      // exactly as llama2_mha/mha.cc:137 does it.
      aie::vector<bfloat16, R> vs = sc.template to_vector<bfloat16>();
      aie::vector<int8, R> q = aie::to_fixed<int8>(vs, 0);
      aie::store_v(bq + g * gs + i * R, q);
      sacc = aie::mac(sacc, q, ones);
    }
    bsum[g] = aie::reduce_add(sacc.template to_vector<int32>());
  }
}

// ---------------------------------------------------------------------------
// The GEMV itself. c[row] += sum_g s_a[g,row]*s_b[g] * sum_{k in g} (q-z)*bq
// ---------------------------------------------------------------------------
template <unsigned m, unsigned k, unsigned gs, bool ZHOIST>
static void matvec_int4_int8_impl(const uint8_t *__restrict a_q,
                                  const bfloat16 *__restrict a_s,
                                  const uint8_t *__restrict a_z,
                                  const int8_t *__restrict bq,
                                  const bfloat16 *__restrict sb,
                                  const int32_t *__restrict bsum,
                                  bfloat16 *__restrict c) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  constexpr unsigned NSUB = gs / R;

  for (unsigned row = 0; row < m; row++) {
    aie::accum<accfloat, R> acc;
    acc.from_vector(aie::zeros<float, R>());
    const uint8_t *__restrict aq = a_q + row * (k / 2);
    float corr = 0.0f;

    for (unsigned g = 0; g < k / gs; g++) {
      const int8_t zs = (int8_t)a_z[g * m + row];
      // s_a[g,row] * s_b[g], folded once per (row, group) in the scalar unit.
      const float sab_f = (float)a_s[g * m + row] * (float)sb[g];
      const bfloat16 sab = (bfloat16)sab_f;

      aie::accum<acc32, R> g_acc;
      g_acc.from_vector(aie::zeros<int32, R>());

      if constexpr (ZHOIST) {
#pragma clang loop unroll(full)
        for (unsigned i = 0; i < NSUB; i++) {
          const unsigned off = (g * gs + i * R) / 2;
          aie::vector<uint8, R / 2> packed = aie::load_v<R / 2>(aq + off);
          aie::vector<int8, R> w =
              packed.template cast_to<uint4>().template unpack_sign<int8>(false);
          aie::vector<int8, R> bv = aie::load_v<R>(bq + g * gs + i * R);
          g_acc = aie::mac(g_acc, w, bv);
        }
        // - z * sum(bq) folded on the scalar side, once per (row, group).
        corr += sab_f * (float)zs * (float)bsum[g];
      } else {
        const aie::vector<int8, R> zv = aie::broadcast<int8, R>(zs);
#pragma clang loop unroll(full)
        for (unsigned i = 0; i < NSUB; i++) {
          const unsigned off = (g * gs + i * R) / 2;
          aie::vector<uint8, R / 2> packed = aie::load_v<R / 2>(aq + off);
          aie::vector<int8, R> w =
              packed.template cast_to<uint4>().template unpack_sign<int8>(false);
          w = aie::sub(w, zv);
          aie::vector<int8, R> bv = aie::load_v<R>(bq + g * gs + i * R);
          g_acc = aie::mac(g_acc, w, bv);
        }
      }

      // Group fold. NOTE: `g_acc.to_vector<bfloat16>()` is a TRAP on an
      // INTEGER accumulator -- aie_api's get_srs<bfloat16>
      // (detail/aie2p/accum.hpp:1090-1095) dispatches to ::to_v32bfloat16(acc)
      // without constraining the accumulator Class, so an acc32 is
      // BIT-REINTERPRETED as fp32 (a `vconv.bf16.fp32` on integer bits), not
      // converted. It compiles, it is 1 instruction, and it silently returns
      // denormal-zero for every realistic accumulator value. Measured on
      // device: the whole GEMV output came back ~0 (walks/smoke, devq 651).
      //
      // The value-correct route is a real SRS to int16 (accum.hpp's
      // SRS_FN(v32int16), which honours shift and sign) followed by the same
      // integer->float widening the shipped bf16 kernel already uses. Lane
      // values here are bounded by gs/R * 15 * 127 = 3810, well inside int16.
      aie::vector<int16, R> g_i16 = g_acc.template to_vector<int16>(0);
      aie::vector<bfloat16, R> g_bf16 = aie::to_float<bfloat16>(g_i16, 0);
      acc = aie::mac(acc, g_bf16, sab);
    }

    float s = aie::reduce_add(acc.template to_vector<float>());
    c[row] = (bfloat16)((float)c[row] + s - corr);
  }
}

extern "C" {

// Primary entry: zero point subtracted in the inner loop (accuracy-matched to
// mv_int4_bf16.cc's group accumulator magnitude).
void matvec_int4_int8_packed(uint8_t *packed, bfloat16 *b, bfloat16 *c) {
  constexpr unsigned Q_BYTES = DIM_M * (DIM_K / 2);
  constexpr unsigned S_BYTES = (DIM_K / DIM_GS) * DIM_M * 2;
  const uint8_t *a_q = packed;
  const bfloat16 *a_s = reinterpret_cast<const bfloat16 *>(packed + Q_BYTES);
  const uint8_t *a_z = packed + Q_BYTES + S_BYTES;

  alignas(32) int8_t bq[DIM_K];
  alignas(32) bfloat16 sb[DIM_K / DIM_GS];
  alignas(32) int32_t bsum[DIM_K / DIM_GS];
  quantize_activation<DIM_K, DIM_GS>(b, bq, sb, bsum);
  matvec_int4_int8_impl<DIM_M, DIM_K, DIM_GS, false>(a_q, a_s, a_z, bq, sb,
                                                     bsum, c);
}

// Zero point hoisted out of the inner loop.
void matvec_int4_int8_packed_zhoist(uint8_t *packed, bfloat16 *b, bfloat16 *c) {
  constexpr unsigned Q_BYTES = DIM_M * (DIM_K / 2);
  constexpr unsigned S_BYTES = (DIM_K / DIM_GS) * DIM_M * 2;
  const uint8_t *a_q = packed;
  const bfloat16 *a_s = reinterpret_cast<const bfloat16 *>(packed + Q_BYTES);
  const uint8_t *a_z = packed + Q_BYTES + S_BYTES;

  alignas(32) int8_t bq[DIM_K];
  alignas(32) bfloat16 sb[DIM_K / DIM_GS];
  alignas(32) int32_t bsum[DIM_K / DIM_GS];
  quantize_activation<DIM_K, DIM_GS>(b, bq, sb, bsum);
  matvec_int4_int8_impl<DIM_M, DIM_K, DIM_GS, true>(a_q, a_s, a_z, bq, sb, bsum,
                                                    c);
}

// TIMING ABLATION ONLY -- numerically meaningless. Reinterprets the bf16 B
// buffer's raw bytes as int8 activations and unit scales, so the weight-side
// instruction stream is identical to `_zhoist` with the quantisation removed.
void matvec_int4_int8_packed_noquant(uint8_t *packed, bfloat16 *b,
                                     bfloat16 *c) {
  constexpr unsigned Q_BYTES = DIM_M * (DIM_K / 2);
  constexpr unsigned S_BYTES = (DIM_K / DIM_GS) * DIM_M * 2;
  const uint8_t *a_q = packed;
  const bfloat16 *a_s = reinterpret_cast<const bfloat16 *>(packed + Q_BYTES);
  const uint8_t *a_z = packed + Q_BYTES + S_BYTES;

  const int8_t *bq = reinterpret_cast<const int8_t *>(b);
  alignas(32) bfloat16 sb[DIM_K / DIM_GS];
  alignas(32) int32_t bsum[DIM_K / DIM_GS];
#pragma clang loop unroll(full)
  for (unsigned g = 0; g < DIM_K / DIM_GS; g++) {
    sb[g] = (bfloat16)1.0f;
    bsum[g] = 0;
  }
  matvec_int4_int8_impl<DIM_M, DIM_K, DIM_GS, true>(a_q, a_s, a_z, bq, sb, bsum,
                                                    c);
}


// DIAGNOSTIC ENTRY -- writes the quantiser's intermediates into c instead of a
// GEMV result, so one device run localises a failure instead of a bisect per
// devq round-trip. c has DIM_M=8 slots per tile; tile 0 lands in D[0..7].
void matvec_int4_int8_packed_diag(uint8_t *packed, bfloat16 *b, bfloat16 *c) {
  constexpr unsigned Q_BYTES = DIM_M * (DIM_K / 2);
  constexpr unsigned S_BYTES = (DIM_K / DIM_GS) * DIM_M * 2;
  const uint8_t *a_q = packed;
  const bfloat16 *a_s = reinterpret_cast<const bfloat16 *>(packed + Q_BYTES);
  const uint8_t *a_z = packed + Q_BYTES + S_BYTES;

  alignas(32) int8_t bq[DIM_K];
  alignas(32) bfloat16 sb[DIM_K / DIM_GS];
  alignas(32) int32_t bsum[DIM_K / DIM_GS];
  quantize_activation<DIM_K, DIM_GS>(b, bq, sb, bsum);

  // one group of row 0, integer MAC, to see whether the product path works
  aie::accum<acc32, R> g;
  g.from_vector(aie::zeros<int32, R>());
  aie::vector<int8, R> zv = aie::broadcast<int8, R>((int8_t)a_z[0]);
  aie::vector<uint8, R / 2> pk = aie::load_v<R / 2>(a_q);
  aie::vector<int8, R> w =
      pk.template cast_to<uint4>().template unpack_sign<int8>(false);
  w = aie::sub(w, zv);
  aie::vector<int8, R> bv = aie::load_v<R>(bq);
  g = aie::mac(g, w, bv);
  aie::vector<int16, R> gi = g.template to_vector<int16>(0);
  aie::vector<bfloat16, R> gb = aie::to_float<bfloat16>(gi, 0);

  c[0] = sb[0];                       // group-0 activation scale
  c[1] = (bfloat16)(float)bq[0];      // first quantised activation
  c[2] = (bfloat16)(float)bq[1];
  c[3] = (bfloat16)(float)bsum[0];    // group-0 sum of bq
  c[4] = a_s[0];                      // weight scale (sanity: BO readable?)
  c[5] = (bfloat16)(float)a_z[0];     // zero point
  c[6] = (bfloat16)(float)(int)w[0];  // first unpacked, centred weight
  c[7] = gb[0];                       // first lane of the folded group product
}

// Same zero epilogue the bf16 kernel exports, so the builder's zero call
// resolves against this object too.
void zero_vectorized_bf16_i8(bfloat16 *c) {
  for (unsigned i = 0; i < DIM_M; i++)
    c[i] = (bfloat16)0.0f;
}

} // extern "C"
