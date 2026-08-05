// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// bf16 tile transpose. Copied from
// programming_examples/data_transfer_transpose/dma_bf16/transpose.cc so the
// transformer-layer fingerprint (builders/block_cache.py hashes this
// directory) covers the source the transpose ELF actually links; the original
// stays the standalone example's.
//
// Transposes one DIM_M x DIM_N row-major L1 tile to DIM_N x DIM_M row-major.
// Scalar element access, not VSHUFFLE-optimized: a DMA-stride transpose is not
// available for sub-32-bit types (the innermost DMA stride must be 1 below
// 32 bits), so the movement is contiguous and the reordering happens here.
//
// DIM_M / DIM_N are the L1 TILE shape, not the L3 matrix shape, and they must
// be -D flags on the compile: builders/transpose.py bakes them per tile shape
// and names the object transpose_m<M>n<N>.o, so two tile shapes cannot
// overwrite each other's object (the E1 naming lesson).

#include <cstdint>

#ifndef DIM_M
#define DIM_M 64
#endif
#ifndef DIM_N
#define DIM_N 32
#endif

using DTYPE = uint16_t;

extern "C" {

void transpose_bf16(DTYPE *__restrict__ in_ptr, DTYPE *__restrict__ out_ptr) {
  for (unsigned i = 0; i < DIM_M; i++) {
    for (unsigned j = 0; j < DIM_N; j++) {
      out_ptr[j * DIM_M + i] = in_ptr[i * DIM_N + j];
    }
  }
}

} // extern "C"
