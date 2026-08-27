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
//   (The entry that subtracted the zero point inside the inner loop,
//    `matvec_int4_int8_packed`, was REMOVED on 2026-08-26 -- it measured
//    rel RMS 0.18 on device against an unexplained cause and no gated arm
//    used it. See the note above its former position.)
//
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
template <unsigned m, unsigned k, unsigned gs, bool ZHOIST,
          bool WITH_CORR = true>
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
        // WITH_CORR=false DELETES this at compile time (arm B1) rather than
        // zeroing an input at run time -- zeroing leaves the two scalar float
        // multiplies and the add in the instruction stream, which is exactly
        // why arm C could not ablate what it claimed to.
        if constexpr (WITH_CORR)
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


// Per-K-CHUNK activation quantisation: ONE scale for the whole chunk instead of
// one per group of gs. This is the repair item 23 proposed and never built. It
// exists so that `s_b` factors out of the group loop entirely --
//     c[row] = s_b * sum_g s_a[g,row] * L_g
// -- leaving the inner fold a single bf16 LOAD feeding a vector MAC (exactly
// the shipped bf16 kernel's fold, zero scalar float) and ONE scalar multiply
// per ROW instead of 64 per call. It costs accuracy: the activation scale is
// now set by the chunk maximum, not the group maximum.
template <unsigned k>
static void quantize_activation_chunk(const bfloat16 *__restrict b,
                                      int8_t *__restrict bq, float *sb_out) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  bfloat16 mx = (bfloat16)0.0f;
  for (unsigned i = 0; i < k; i += R) {
    aie::vector<bfloat16, R> v = aie::load_v<R>(b + i);
    bfloat16 mm = aie::reduce_max(aie::abs(v));
    mx = (float)mm > (float)mx ? mm : mx;
  }
  float s = (float)mx * (1.0f / 127.0f);
  float inv = ((float)mx == 0.0f) ? 0.0f : (127.0f / (float)mx);
  *sb_out = s;
  for (unsigned i = 0; i < k; i += R) {
    aie::vector<bfloat16, R> v = aie::load_v<R>(b + i);
    aie::accum<accfloat, R> sc = aie::mul(v, (bfloat16)inv);
    aie::vector<bfloat16, R> vs = sc.template to_vector<bfloat16>();
    aie::store_v(bq + i, aie::to_fixed<int8>(vs, 0));
  }
}

// The repaired GEMV. Per (row, group) the ONLY scalar work is a bf16 load --
// byte for byte the shipped bf16 kernel's fold. WIDE=true swaps aie::mac for
// the wide elementwise form `::mac_elem_64_2_conf(v128int8, v128int8, v64acc32)`
// = 128 int8 MACs in ONE instruction, which aie::mac cannot reach (it is pinned
// to the half-width I512 form and chunks by 64: aie_api mul_acc32.hpp:64-72,
// 79-102; verified by disassembly in isa_probe/probe_wide.dis -- aie::mac over
// 128 int8 emits TWO vmac, the intrinsic emits ONE).
template <unsigned m, unsigned k, unsigned gs, bool WIDE>
static void matvec_int4_int8_chunk_impl(const uint8_t *__restrict a_q,
                                        const bfloat16 *__restrict a_s,
                                        const uint8_t *__restrict a_z,
                                        const int8_t *__restrict bq,
                                        float sb, bfloat16 *__restrict c) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  constexpr unsigned RW = WIDE ? 128 : R;   // elements consumed per MAC
  constexpr unsigned NSUB = gs / RW;

  for (unsigned row = 0; row < m; row++) {
    aie::accum<accfloat, R> acc;
    acc.from_vector(aie::zeros<float, R>());
    const uint8_t *__restrict aq = a_q + row * (k / 2);

    for (unsigned g = 0; g < k / gs; g++) {
      aie::accum<acc32, R> g_acc;
      g_acc.from_vector(aie::zeros<int32, R>());

      if constexpr (WIDE) {
        const aie::vector<int8, 128> zv = aie::broadcast<int8, 128>(
            (int8_t)a_z[g * m + row]);
#pragma clang loop unroll(full)
        for (unsigned i = 0; i < NSUB; i++) {
          const unsigned off = (g * gs + i * 128) / 2;
          aie::vector<uint8, 64> packed = aie::load_v<64>(aq + off);
          aie::vector<int8, 128> w =
              packed.template cast_to<uint4>().template unpack_sign<int8>(false);
          w = aie::sub(w, zv);
          aie::vector<int8, 128> bv = aie::load_v<128>(bq + g * gs + i * 128);
          g_acc = aie::accum<acc32, R>(::mac_elem_64_2_conf(
              (v128int8)w, (v128int8)bv, (v64acc32)g_acc, 0, 0, 0, 0));
        }
      } else {
        const aie::vector<int8, R> zv =
            aie::broadcast<int8, R>((int8_t)a_z[g * m + row]);
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

      aie::vector<int16, R> g_i16 = g_acc.template to_vector<int16>(0);
      aie::vector<bfloat16, R> g_bf16 = aie::to_float<bfloat16>(g_i16, 0);
      // THE POINT OF THIS VARIANT: a bf16 LOAD, no scalar float at all.
      acc = aie::mac(acc, g_bf16, a_s[g * m + row]);
    }
    // one scalar float multiply per ROW, not 64 per call
    float s = aie::reduce_add(acc.template to_vector<float>());
    c[row] = (bfloat16)((float)c[row] + sb * s);
  }
}

extern "C" {

// NOTE (2026-08-26, after the DAM-RS cross-check): the entry that subtracted
// the zero point inside the inner loop, `matvec_int4_int8_packed`, HAS BEEN
// REMOVED. It measured relative RMS 0.18 on device (devq 661) against an
// unexplained cause, while every gated arm invoked `_zhoist`. Exporting a
// symbol that is known-wrong and ungated is a trap for any caller that picks
// the nominal name, so it is gone rather than documented. The in-loop subtract
// itself is not the suspect -- `matvec_int4_int8_chunk` below uses it and is
// gated -- so the discrepancy remains an open lead, not a known defect.

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


// B1 ablation -- TIMING ONLY, NUMERICALLY WRONG BY CONSTRUCTION. `_zhoist`
// with the zero-point correction chain deleted (the `corr` accumulation, two
// scalar float multiplies and an add per (row,group)). It drops the z term, so
// its OUTPUT IS NOT THE GEMV and no correctness claim is made or accepted for
// it. It exists to price the corr chain, which arm C did NOT ablate.
void matvec_int4_int8_packed_nocorr(uint8_t *packed, bfloat16 *b, bfloat16 *c) {
  constexpr unsigned Q_BYTES = DIM_M * (DIM_K / 2);
  constexpr unsigned S_BYTES = (DIM_K / DIM_GS) * DIM_M * 2;
  const uint8_t *a_q = packed;
  const bfloat16 *a_s = reinterpret_cast<const bfloat16 *>(packed + Q_BYTES);
  const uint8_t *a_z = packed + Q_BYTES + S_BYTES;
  alignas(32) int8_t bq[DIM_K];
  alignas(32) bfloat16 sb[DIM_K / DIM_GS];
  alignas(32) int32_t bsum[DIM_K / DIM_GS];
  quantize_activation<DIM_K, DIM_GS>(b, bq, sb, bsum);
  matvec_int4_int8_impl<DIM_M, DIM_K, DIM_GS, true, /*WITH_CORR=*/false>(
      a_q, a_s, a_z, bq, sb, bsum, c);
}

// B2 -- per-K-chunk activation scale, in-loop zero subtract, no scalar float in
// the group loop. NUMERICALLY CORRECT (a coarser but legitimate activation
// quantisation) and gated.
void matvec_int4_int8_chunk(uint8_t *packed, bfloat16 *b, bfloat16 *c) {
  constexpr unsigned Q_BYTES = DIM_M * (DIM_K / 2);
  constexpr unsigned S_BYTES = (DIM_K / DIM_GS) * DIM_M * 2;
  const uint8_t *a_q = packed;
  const bfloat16 *a_s = reinterpret_cast<const bfloat16 *>(packed + Q_BYTES);
  const uint8_t *a_z = packed + Q_BYTES + S_BYTES;
  alignas(32) int8_t bq[DIM_K];
  float sb;
  quantize_activation_chunk<DIM_K>(b, bq, &sb);
  matvec_int4_int8_chunk_impl<DIM_M, DIM_K, DIM_GS, false>(a_q, a_s, a_z, bq,
                                                           sb, c);
}

// B3 -- B2 plus the WIDE elementwise int8 MAC at 128 elements/instruction.
void matvec_int4_int8_chunk_wide(uint8_t *packed, bfloat16 *b, bfloat16 *c) {
  constexpr unsigned Q_BYTES = DIM_M * (DIM_K / 2);
  constexpr unsigned S_BYTES = (DIM_K / DIM_GS) * DIM_M * 2;
  const uint8_t *a_q = packed;
  const bfloat16 *a_s = reinterpret_cast<const bfloat16 *>(packed + Q_BYTES);
  const uint8_t *a_z = packed + Q_BYTES + S_BYTES;
  alignas(32) int8_t bq[DIM_K];
  float sb;
  quantize_activation_chunk<DIM_K>(b, bq, &sb);
  matvec_int4_int8_chunk_impl<DIM_M, DIM_K, DIM_GS, true>(a_q, a_s, a_z, bq, sb,
                                                          c);
}

// Same zero epilogue the bf16 kernel exports, so the builder's zero call
// resolves against this object too.
void zero_vectorized_bf16_i8(bfloat16 *c) {
  for (unsigned i = 0; i < DIM_M; i++)
    c[i] = (bfloat16)0.0f;
}

} // extern "C"
