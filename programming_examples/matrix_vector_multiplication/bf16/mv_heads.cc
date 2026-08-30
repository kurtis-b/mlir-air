//===- mv_heads.cc -----------------------------------------------*- C++
//-*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// GEMV rows + an in-core per-head epilogue.
//
// The decode QKV stage projects x (K) onto [wq; wk; wv] (M rows) and then
// applies, to the Q and K heads only, a per-head RMSNorm (QK-norm: RMS over the
// head's HEAD_DIM outputs, times a per-element weight) and the half-split RoPE
// rotation with one position's cos|sin LUT. When a core owns a WHOLE head's
// rows, both can run on the L1 output tile before it leaves the core, which
// removes the separate QK-norm and RoPE launches of the multi-launch forms:
// two `air.launch` boundaries (PDI loads) per layer. (Measured on the study
// branch only; the driver PR's profile vs main's 8-launch form is the number.)
//
// One call per m-row CHUNK of a head (m = 8 rows, the matvec.py tile, so the
// L2 pipeline keeps matvec.py's 16 KB granularity; a whole-head tile
// serializes its fill against the core's drain -- chunking avoids that):
//
//   qkv_heads_chunk_bf16(m, k, a, b, c, out, eps)
//     a   : [m, k + K_PAD] the chunk's weight rows; element a[r][k] is the
//           chunk's TAG (its index within the head, 0 .. HEAD_DIM/m - 1) and
//           a[r][k+1] its KIND (0 Q head, 1 K head, 2 V head), both baked
//           into the padding of the static weight matrix by the host.
//     b   : [k + 3*HEAD_DIM] = [normed | lut | q_norm | k_norm]
//     c   : [HEAD_DIM] the head accumulator, PERSISTENT across calls
//     out : [HEAD_DIM] the head's final output, written by the last chunk
//
//     c[tag*m + i] = dot(a[i, 0:k], b[0:k]);  if tag is the last chunk:
//     out = kind < 2 ? rope(qknorm(c) * w_kind, lut) : c
//
// Why the tag rides in A: the core does not see the launch iteration (its
// program is a while(true) over lock handshakes), and a core tile has only two
// inbound DMA channels, both taken (A from the memtile, B from the shim) --
// a third stream fails in aiecc's router ("'aie.connect' op ... targets same
// dst as another connect op"). The weight rows are static, so the host bakes
// the per-row tag/kind into a 64-element row padding once (+6 % bytes).
//
// The QK-norm arithmetic mirrors the 4-launch stage's vector-dialect kernel
// (`_build_qknorm_1d`): sum of squares in f32, rstd = rsqrt(mean + eps) in f32
// and TRUNCATED to bf16 before the multiply, y = bf16(bf16(x * rstd) * w). The
// RoPE half is rope_halfsplit.cc's loop verbatim (f32 products, one rounding).
//
//===----------------------------------------------------------------------===//

#define __AIENGINE__ 2
#define NOCPP
#define __AIEARCH__ 20

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

// HEAD_DIM and K_PAD are baked in at compile time
// (external_kernels.compile_mv_heads): the per-head loops unroll fully and the
// mean needs no float division (the core has no FP divider).
#ifndef HEAD_DIM
#define HEAD_DIM 128
#endif
#ifndef K_PAD
#define K_PAD 64
#endif

// mv.cc's kernel with a row stride (the K_PAD tail of every row is skipped).
template <uint32_t r>
void matvec_heads_vectorized(uint32_t m, uint32_t k, uint32_t row_stride,
                             const bfloat16 *__restrict a,
                             const bfloat16 *__restrict b,
                             bfloat16 *__restrict c) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  bfloat16 *c_end = c + m;
  const bfloat16 *b_end = b + k;
  for (; c < c_end; c++, a += row_stride - k) {
    aie::accum acc = aie::zeros<accfloat, r>();
    for (const bfloat16 *__restrict b_cur = b; b_cur < b_end;
         b_cur += r, a += r) {
      aie::vector<bfloat16, r> a_vec = aie::load_v<r>(a);
      aie::vector<bfloat16, r> b_vec = aie::load_v<r>(b_cur);
      acc = aie::mac(acc, a_vec, b_vec);
    }
    *c =
        static_cast<bfloat16>(aie::reduce_add(acc.template to_vector<float>()));
  }
}

// One head: QK-norm (weighted RMSNorm over `dims`) then half-split RoPE.
template <typename T, int N, int dims>
void qknorm_rope_head(const T *__restrict c, const T *__restrict w,
                      const T *__restrict lut, T *__restrict out, float eps) {
  // --- QK-norm: sum of squares in f32 (16-lane accumulator, then reduce) ---
  ::aie::accum<accfloat, N> acc = ::aie::zeros<accfloat, N>();
  for (int j = 0; j < dims; j += N) {
    ::aie::vector<T, N> x = ::aie::load_v<N>(c + j);
    acc = ::aie::mac(acc, x, x);
  }
  float sum_sq = ::aie::reduce_add(acc.template to_vector<float>());
  float rms = sum_sq * (1.0f / static_cast<float>(dims)) + eps;
  float rstd_f32 = ::aie::invsqrt(rms);
  // The 4-launch kernel truncates rstd to bf16 before the multiply
  // (`arith.truncf` in _build_qknorm_1d); mirrored so the two agree.
  ::aie::vector<T, N> rstd_v = ::aie::broadcast<T, N>(static_cast<T>(rstd_f32));

  // normed = bf16(x * rstd); weighted = bf16(normed * w) into `out`; RoPE
  // below reads `out` and writes it back in place (each pair (i, i+half) is
  // loaded before either is stored).
  for (int j = 0; j < dims; j += N) {
    ::aie::vector<T, N> x = ::aie::load_v<N>(c + j);
    ::aie::vector<T, N> wv = ::aie::load_v<N>(w + j);
    ::aie::vector<T, N> normed = ::aie::mul(x, rstd_v);
    ::aie::vector<T, N> weighted = ::aie::mul(normed, wv);
    ::aie::store_v(out + j, weighted);
  }

  // --- RoPE, half-split (rope_halfsplit.cc): LUT = [cos_0..cos_{h-1}, sin_0..]
  // ---
  const int half = dims / 2;
  for (int v = 0; v < half; v += N) {
    ::aie::vector<T, N> x1 = ::aie::load_v<N>(out + v);
    ::aie::vector<T, N> x2 = ::aie::load_v<N>(out + v + half);
    ::aie::vector<T, N> cos_v = ::aie::load_v<N>(lut + v);
    ::aie::vector<T, N> sin_v = ::aie::load_v<N>(lut + v + half);
    ::aie::vector<T, N> out1 =
        ::aie::sub(::aie::mul(x1, cos_v), ::aie::mul(x2, sin_v));
    ::aie::vector<T, N> out2 =
        ::aie::add(::aie::mul(x1, sin_v), ::aie::mul(x2, cos_v));
    ::aie::store_v(out + v, out1);
    ::aie::store_v(out + v + half, out2);
  }
}

template <typename T, int N, int dims>
void copy_head(const T *__restrict c, T *__restrict out) {
  for (int j = 0; j < dims; j += N)
    ::aie::store_v(out + j, ::aie::load_v<N>(c + j));
}

extern "C" {

void qkv_heads_chunk_bf16(uint32_t m, uint32_t k, const bfloat16 *__restrict a,
                          const bfloat16 *__restrict b, bfloat16 *__restrict c,
                          bfloat16 *__restrict out, float eps) {
  const uint32_t row_stride = k + K_PAD;
  const int32_t tag = static_cast<int32_t>(static_cast<float>(a[k]));
  const int32_t kind = static_cast<int32_t>(static_cast<float>(a[k + 1]));
  matvec_heads_vectorized<64>(m, k, row_stride, a, b, c + tag * m);
  if (static_cast<uint32_t>(tag + 1) * m == HEAD_DIM) { // the head's last chunk
    if (kind < 2)
      qknorm_rope_head<bfloat16, 16, HEAD_DIM>(c, b + k + HEAD_DIM * (1 + kind),
                                               b + k, out, eps);
    else
      copy_head<bfloat16, 16, HEAD_DIM>(c, out);
  }
}

} // extern "C"
