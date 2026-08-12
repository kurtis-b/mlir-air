//===- shrink_memref_multi_get_band.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// air-shrink-memref-sizes-by-access: THE EXTENT IS OFFSET + SIZE, NOT SIZE.
//
// The pass sized an allocation from air::getDataAccessShapeFromMemcpyOp, which
// derived its bound from each user's SIZES and STRIDES and never looked at the
// OFFSETS. A band assembled by N channel gets that each move the same volume
// into a different literal offset therefore measured as one get's worth, and
// the memref was rewritten to that -- while every get kept its original
// offset, because getUpdatedOffsetsAfterShrinkage rebases an offset to 0 only
// when it varies over a SPATIAL iteration space (an air.herd tile index or an
// scf.parallel induction variable, where the buffer is replicated per PE) and
// leaves a literal offset alone.
//
// The result compiled clean, routed, and returned wrong data: no error, no
// warning, no diagnostic on any decline path -- the same class as the frozen-BD
// trap. Measured shape, from 31b-r2-order-seam.md 3.2 (R1's pass_029):
//
//   before: air.channel.get @r2_rows[...] (%b[6144] [3072] [1]) : memref<12288xbf16, 2>
//   after:  air.channel.get @r2_rows[...] (%b[6144] [3072] [1]) : memref<3072xbf16, 2>
//
// i.e. a get reaching element 12,256 of a 3,072-element buffer.
//
// The extent now counts offset + size, so this band measures 12288 and is left
// alone. Case 2 pins that this is a real extent computation and not a blanket
// refusal: the same four gets landing on only two distinct offsets still shrink
// -- to 6144, the extent they actually reach, not 3072 and not 12288.
// Case 3 pins the per-PE case the pass exists to serve, which must keep
// shrinking. Case 4 pins the loud decline: an offset from an iteration space
// the pass does not model used to fall off the end of an empty SmallVector
// returned by getUpdatedOffsetsAfterShrinkage; it now declines with a
// diagnostic instead, per doc 23 "Silence is the wrong default".

// RUN: air-opt %s -air-shrink-memref-sizes-by-access -split-input-file | FileCheck %s
// RUN: air-opt %s -air-shrink-memref-sizes-by-access -split-input-file -verify-diagnostics

// A 12288-element L1 band filled by four gets at literal offsets 0 / 3072 /
// 6144 / 9216. The extent is 9216 + 3072 = 12288, so nothing shrinks and every
// get keeps an offset that is still inside the buffer.

// CHECK-LABEL: @multi_get_band_is_not_shrunk
// CHECK: memref.alloc() : memref<12288xbf16, 2 : i32>
// CHECK-NOT: memref<3072xbf16, 2 : i32>
// CHECK: air.channel.get {{.*}} (%{{.*}}[%c0] [%c3072] [%c1{{.*}}]) {id = 1 : i32} : (memref<12288xbf16, 2 : i32>)
// CHECK: air.channel.get {{.*}} (%{{.*}}[%c3072] [%c3072] [%c1{{.*}}]) {id = 2 : i32} : (memref<12288xbf16, 2 : i32>)
// CHECK: air.channel.get {{.*}} (%{{.*}}[%c6144] [%c3072] [%c1{{.*}}]) {id = 3 : i32} : (memref<12288xbf16, 2 : i32>)
// CHECK: air.channel.get {{.*}} (%{{.*}}[%c9216] [%c3072] [%c1{{.*}}]) {id = 4 : i32} : (memref<12288xbf16, 2 : i32>)
module {
  air.channel @r2_rows [1, 1]
  func.func @multi_get_band_is_not_shrunk() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg0) in (%arg1=%c1) {
      %1 = air.segment @seg async {
        %c1_0 = arith.constant 1 : index
        %2 = air.herd @herd_0 async tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) {
          %c0 = arith.constant 0 : index
          %c1_1 = arith.constant 1 : index
          %c3072 = arith.constant 3072 : index
          %c6144 = arith.constant 6144 : index
          %c9216 = arith.constant 9216 : index
          %async_token, %results = air.execute -> (memref<12288xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<12288xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<12288xbf16, 2 : i32>
          }
          %g0 = air.channel.get async [%async_token] @r2_rows[%c0, %c0] (%results[%c0] [%c3072] [%c1_1]) {id = 1 : i32} : (memref<12288xbf16, 2 : i32>)
          %g1 = air.channel.get async [%async_token] @r2_rows[%c0, %c0] (%results[%c3072] [%c3072] [%c1_1]) {id = 2 : i32} : (memref<12288xbf16, 2 : i32>)
          %g2 = air.channel.get async [%async_token] @r2_rows[%c0, %c0] (%results[%c6144] [%c3072] [%c1_1]) {id = 3 : i32} : (memref<12288xbf16, 2 : i32>)
          %g3 = air.channel.get async [%async_token] @r2_rows[%c0, %c0] (%results[%c9216] [%c3072] [%c1_1]) {id = 4 : i32} : (memref<12288xbf16, 2 : i32>)
          %async_token_2 = air.execute [%g0, %g1, %g2, %g3] {
            memref.dealloc %results : memref<12288xbf16, 2 : i32>
          }
        }
      }
    }
    return
  }
}

// -----

// The same four gets, but landing on only two distinct offsets. The extent is
// 3072 + 3072 = 6144, so the band DOES shrink -- to the extent it reaches, not
// to one get's size and not to the declared 12288.

// CHECK-LABEL: @partial_band_shrinks_to_its_extent
// CHECK: memref.alloc() {{.*}} : memref<6144xbf16, 2 : i32>
// CHECK: air.channel.get {{.*}} (%{{.*}}[%c0] [%c3072] {{.*}} : (memref<6144xbf16, 2 : i32>)
// CHECK: air.channel.get {{.*}} (%{{.*}}[%c3072] [%c3072] {{.*}} : (memref<6144xbf16, 2 : i32>)
module {
  air.channel @r2_rows [1, 1]
  func.func @partial_band_shrinks_to_its_extent() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg0) in (%arg1=%c1) {
      %1 = air.segment @seg async {
        %c1_0 = arith.constant 1 : index
        %2 = air.herd @herd_0 async tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) {
          %c0 = arith.constant 0 : index
          %c1_1 = arith.constant 1 : index
          %c3072 = arith.constant 3072 : index
          %async_token, %results = air.execute -> (memref<12288xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<12288xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<12288xbf16, 2 : i32>
          }
          %g0 = air.channel.get async [%async_token] @r2_rows[%c0, %c0] (%results[%c0] [%c3072] [%c1_1]) {id = 1 : i32} : (memref<12288xbf16, 2 : i32>)
          %g1 = air.channel.get async [%async_token] @r2_rows[%c0, %c0] (%results[%c3072] [%c3072] [%c1_1]) {id = 2 : i32} : (memref<12288xbf16, 2 : i32>)
          %g2 = air.channel.get async [%async_token] @r2_rows[%c0, %c0] (%results[%c0] [%c3072] [%c1_1]) {id = 3 : i32} : (memref<12288xbf16, 2 : i32>)
          %g3 = air.channel.get async [%async_token] @r2_rows[%c0, %c0] (%results[%c3072] [%c3072] [%c1_1]) {id = 4 : i32} : (memref<12288xbf16, 2 : i32>)
          %async_token_2 = air.execute [%g0, %g1, %g2, %g3] {
            memref.dealloc %results : memref<12288xbf16, 2 : i32>
          }
        }
      }
    }
    return
  }
}

// -----

// The per-PE case the pass exists to serve, which must keep shrinking: the get
// offset is an affine.apply on a herd tile index, so the buffer is replicated
// per PE, the offset is rebased to 0, and it contributes nothing to the extent.

// CHECK-LABEL: @herd_variant_offset_still_shrinks
// CHECK: memref.alloc() {{.*}} : memref<3072xbf16, 2 : i32>
// CHECK: air.channel.get {{.*}} (%{{.*}}[0] [%c3072] {{.*}} : (memref<3072xbf16, 2 : i32>)
#map = affine_map<()[s0] -> (s0 * 3072)>
module {
  air.channel @r2_rows [4]
  func.func @herd_variant_offset_still_shrinks() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg0) in (%arg1=%c1) {
      %1 = air.segment @seg async {
        %c4 = arith.constant 4 : index
        %2 = air.herd @herd_0 async tile (%tx) in (%sx=%c4) {
          %c3072 = arith.constant 3072 : index
          %c1_1 = arith.constant 1 : index
          %off = affine.apply #map()[%tx]
          %async_token, %results = air.execute -> (memref<12288xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<12288xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<12288xbf16, 2 : i32>
          }
          %g0 = air.channel.get async [%async_token] @r2_rows[%tx] (%results[%off] [%c3072] [%c1_1]) {id = 1 : i32} : (memref<12288xbf16, 2 : i32>)
          %async_token_2 = air.execute [%g0] {
            memref.dealloc %results : memref<12288xbf16, 2 : i32>
          }
        }
      }
    }
    return
  }
}

// -----

// An offset the pass cannot bound: an scf.for induction variable whose trip
// count is not static, feeding a channel get (whose offsets, unlike a subview's
// or a vector transfer's, are never folded into the access sizes by
// updateAccessPatternByScfForNest). How far the sweep reaches is unknown, so
// shrinking on it would be a guess. The pass declines -- and says so. Leaving
// the memref at its declared size is always correct here; only the
// optimization is lost.

// CHECK-LABEL: @unbounded_offset_declines_loudly
// CHECK: memref.alloc() {{.*}} : memref<12288xbf16, 2 : i32>
// CHECK: air.channel.get {{.*}} : (memref<12288xbf16, 2 : i32>)
module {
  air.channel @r2_rows [1, 1]
  func.func @unbounded_offset_declines_loudly(%trips: index) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg0) in (%arg1=%c1) args(%n=%trips) : index {
      %1 = air.segment @seg async args(%n2=%n) : index {
        %c1_0 = arith.constant 1 : index
        %2 = air.herd @herd_0 async tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%n3=%n2) : index {
          %c0 = arith.constant 0 : index
          %c1_1 = arith.constant 1 : index
          %c3072 = arith.constant 3072 : index
          %async_token, %results = air.execute -> (memref<12288xbf16, 2 : i32>) {
            // expected-warning@+1 {{declining to shrink}}
            %alloc = memref.alloc() : memref<12288xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<12288xbf16, 2 : i32>
          }
          // Upper bound is a runtime value, so the sweep has no static bound.
          %loop = scf.for %i = %c0 to %n3 step %c3072 iter_args(%t = %async_token) -> (!air.async.token) {
            %g = air.channel.get async [%t] @r2_rows[%c0, %c0] (%results[%i] [%c3072] [%c1_1]) {id = 1 : i32} : (memref<12288xbf16, 2 : i32>)
            scf.yield %g : !air.async.token
          }
          %async_token_2 = air.execute [%loop] {
            memref.dealloc %results : memref<12288xbf16, 2 : i32>
          }
        }
      }
    }
    return
  }
}
