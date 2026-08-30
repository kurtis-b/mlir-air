//===- air_split_l2_memref_repeated_feed.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s --air-split-l2-memref="tiles-per-l2-tile=1" --split-input-file | FileCheck %s

// `tileChannelOpByFactor` splits the far side of the channel CONTIGUOUSLY:
// split i takes offset += i * (wrap / factor) with wrap /= factor. That inverts
// the L2-side split only when the two accesses run in step, one buffer for one
// buffer -- the pattern the pass was written for, where both sides walk the same
// tile in matching loops.
//
// It does not hold when the L2 buffer is re-staged per iteration while the far
// side delivers the whole loop's worth in ONE access. Below, the L3 put moves
// 24 x 8576 bytes once while the L2 buffer stages 8576 bytes 24 times, and the
// buffer's two halves are INTERLEAVED through that stream (iteration i holds
// rows 2i and 2i+1) rather than laid out as its first and second halves.
// Splitting the far side contiguously hands sub-buffer 0 rows 0..23 and
// sub-buffer 1 rows 24..47, so every core reads the wrong tile: this compiles
// and runs, and a per-element bound on an int4 GEMV at this shape failed on
// 6031 of 6144 outputs.
//
// Expressing the split here would need a STRIDED far-side split (offset += i,
// with an extra wrap-and-stride dimension), which the pass only arms from an
// scf.for step. Until it does, decline -- the buffer must come out intact.

// -----

// Case 1 -- MM2S. One L3 put covering 24 stagings; the L2 buffer is split two
// ways per staging. Must not split.

// CHECK-LABEL: func.func @repeated_feed_mm2s
// CHECK: memref.alloc() : memref<8576xi8, 1 : i32>
// CHECK-NOT: memref.alloc() : memref<4288xi8, 1 : i32>

air.channel @inL3 [1]
air.channel @aL2ToL1 [1, 2]
func.func @repeated_feed_mm2s(%arg0: memref<48x4288xi8>) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%a0=%arg0) : memref<48x4288xi8> attributes {id = 1 : i32} {
    %c0 = arith.constant 0 : index
    %c1_0 = arith.constant 1 : index
    %c48 = arith.constant 48 : index
    %c4288 = arith.constant 4288 : index
    %p = air.channel.put async @inL3[%c0] (%a0[%c0, %c0] [%c48, %c4288] [%c4288, %c1_0]) {id = 1 : i32} : (memref<48x4288xi8>)
    %s = air.segment @seg async attributes {id = 2 : i32} {
      %c0_1 = arith.constant 0 : index
      %c1_1 = arith.constant 1 : index
      %c2_1 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c24 = arith.constant 24 : index
      %c4288_1 = arith.constant 4288 : index
      %w0 = air.wait_all async
      %f = scf.for %i = %c0_1 to %c24 step %c1_1 iter_args(%it = %w0) -> (!air.async.token) {
        %tok, %buf = air.execute -> (memref<8576xi8, 1 : i32>) {
          %alloc = memref.alloc() : memref<8576xi8, 1 : i32>
          air.execute_terminator %alloc : memref<8576xi8, 1 : i32>
        }
        %g = air.channel.get async [%it, %tok] @inL3[%c0_1] (%buf[] [] []) {id = 2 : i32} : (memref<8576xi8, 1 : i32>)
        %p0 = air.channel.put async [%g] @aL2ToL1[%c0_1, %c0_1] (%buf[%c0_1] [%c4288_1] [%c1_1]) {id = 3 : i32} : (memref<8576xi8, 1 : i32>)
        %p1 = air.channel.put async [%g] @aL2ToL1[%c0_1, %c1_1] (%buf[%c4288_1] [%c4288_1] [%c1_1]) {id = 4 : i32} : (memref<8576xi8, 1 : i32>)
        %dt = air.execute [%p0, %p1] {
          memref.dealloc %buf : memref<8576xi8, 1 : i32>
        }
        %wa = air.wait_all async [%p0, %p1]
        scf.yield %wa : !air.async.token
      }
      %h = air.herd @h async tile (%hx, %hy) in (%hsx=%c4, %hsy=%c2_1) attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
        %c0_2 = arith.constant 0 : index
        %c1_2 = arith.constant 1 : index
        %c24_2 = arith.constant 24 : index
        %wh = air.wait_all async
        %fh = scf.for %j = %c0_2 to %c24_2 step %c1_2 iter_args(%ith = %wh) -> (!air.async.token) {
          %tokl, %bufl = air.execute -> (memref<4288xi8, 2 : i32>) {
            %alloc = memref.alloc() : memref<4288xi8, 2 : i32>
            air.execute_terminator %alloc : memref<4288xi8, 2 : i32>
          }
          %gl = air.channel.get async [%ith, %tokl] @aL2ToL1[%hx, %hy] (%bufl[] [] []) {id = 5 : i32} : (memref<4288xi8, 2 : i32>)
          %dl = air.execute [%gl] {
            memref.dealloc %bufl : memref<4288xi8, 2 : i32>
          }
          scf.yield %gl : !air.async.token
        }
      }
    }
  }
  return
}

// -----

// Case 2 -- S2MM. Same mismatch in the other direction: one L3 get drains 24
// stagings of a two-way-joined L2 buffer. Must not split.

// CHECK-LABEL: func.func @repeated_feed_s2mm
// CHECK: memref.alloc() : memref<16xbf16, 1 : i32>
// CHECK-NOT: memref.alloc() : memref<8xbf16, 1 : i32>

air.channel @dL1ToL2 [1, 2]
air.channel @outD [1]
func.func @repeated_feed_s2mm(%arg0: memref<384xbf16>) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%a0=%arg0) : memref<384xbf16> attributes {id = 1 : i32} {
    %c0 = arith.constant 0 : index
    %c1_0 = arith.constant 1 : index
    %c384 = arith.constant 384 : index
    %g = air.channel.get async @outD[%c0] (%a0[%c0] [%c384] [%c1_0]) {id = 1 : i32} : (memref<384xbf16>)
    %s = air.segment @seg async attributes {id = 2 : i32} {
      %c0_1 = arith.constant 0 : index
      %c1_1 = arith.constant 1 : index
      %c2_1 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c24 = arith.constant 24 : index
      %h = air.herd @h async tile (%hx, %hy) in (%hsx=%c4, %hsy=%c2_1) attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
        %c0_2 = arith.constant 0 : index
        %c1_2 = arith.constant 1 : index
        %c24_2 = arith.constant 24 : index
        %wh = air.wait_all async
        %fh = scf.for %j = %c0_2 to %c24_2 step %c1_2 iter_args(%ith = %wh) -> (!air.async.token) {
          %tokl, %bufl = air.execute -> (memref<8xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<8xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<8xbf16, 2 : i32>
          }
          %pl = air.channel.put async [%ith, %tokl] @dL1ToL2[%hx, %hy] (%bufl[] [] []) {id = 2 : i32} : (memref<8xbf16, 2 : i32>)
          %dl = air.execute [%pl] {
            memref.dealloc %bufl : memref<8xbf16, 2 : i32>
          }
          scf.yield %pl : !air.async.token
        }
      }
      %w0 = air.wait_all async [%h]
      %f = scf.for %i = %c0_1 to %c24 step %c1_1 iter_args(%it = %w0) -> (!air.async.token) {
        %tok, %buf = air.execute -> (memref<16xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<16xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<16xbf16, 1 : i32>
        }
        %g0 = air.channel.get async [%it, %tok] @dL1ToL2[%c0_1, %c0_1] (%buf[%c0_1] [%c8] [%c1_1]) {id = 3 : i32} : (memref<16xbf16, 1 : i32>)
        %g1 = air.channel.get async [%it, %tok] @dL1ToL2[%c0_1, %c1_1] (%buf[%c8] [%c8] [%c1_1]) {id = 4 : i32} : (memref<16xbf16, 1 : i32>)
        %pp = air.channel.put async [%g0, %g1] @outD[%c0_1] (%buf[] [] []) {id = 5 : i32} : (memref<16xbf16, 1 : i32>)
        %dt = air.execute [%pp] {
          memref.dealloc %buf : memref<16xbf16, 1 : i32>
        }
        scf.yield %pp : !air.async.token
      }
    }
  }
  return
}

// -----

// Case 3 -- the same repeated feed with the far side's leading size SSA-DYNAMIC
// rather than a constant. `%dyn` is 48 at runtime, exactly case 1's shape, but
// `getConstantIntValue` cannot see it.
//
// The volume test that rejects case 1 returned a -1 sentinel here and the guard
// read it as "no larger than the L2 side", so the split went ahead -- a safety
// decline that treats "I cannot compute this" as "safe to proceed" is not a
// safety decline. On the pass as first fixed this input reached the same
// contiguous cut that failed 6031 of 6144 elements on device, and aborted in
// `runOnOperation` on a std::optional deref (exit 134) on the way there.
// Undecidable must decline.

// CHECK-LABEL: func.func @repeated_feed_dynamic_far_size
// CHECK: memref.alloc() : memref<8576xi8, 1 : i32>
// CHECK-NOT: memref.alloc() : memref<4288xi8, 1 : i32>

air.channel @inL3 [1]
air.channel @aL2ToL1 [1, 2]
func.func @repeated_feed_dynamic_far_size(%arg0: memref<48x4288xi8>) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%a0=%arg0) : memref<48x4288xi8> attributes {id = 1 : i32} {
    %c0 = arith.constant 0 : index
    %c1_0 = arith.constant 1 : index
    %c48 = arith.constant 48 : index
    %c4288 = arith.constant 4288 : index
    // 48 at runtime; not a constant to the pass.
    %dyn = arith.muli %c48, %sx : index
    %p = air.channel.put async @inL3[%c0] (%a0[%c0, %c0] [%dyn, %c4288] [%c4288, %c1_0]) {id = 1 : i32} : (memref<48x4288xi8>)
    %s = air.segment @seg async attributes {id = 2 : i32} {
      %c0_1 = arith.constant 0 : index
      %c1_1 = arith.constant 1 : index
      %c2_1 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c24 = arith.constant 24 : index
      %c4288_1 = arith.constant 4288 : index
      %w0 = air.wait_all async
      %f = scf.for %i = %c0_1 to %c24 step %c1_1 iter_args(%it = %w0) -> (!air.async.token) {
        %tok, %buf = air.execute -> (memref<8576xi8, 1 : i32>) {
          %alloc = memref.alloc() : memref<8576xi8, 1 : i32>
          air.execute_terminator %alloc : memref<8576xi8, 1 : i32>
        }
        %g = air.channel.get async [%it, %tok] @inL3[%c0_1] (%buf[] [] []) {id = 2 : i32} : (memref<8576xi8, 1 : i32>)
        %p0 = air.channel.put async [%g] @aL2ToL1[%c0_1, %c0_1] (%buf[%c0_1] [%c4288_1] [%c1_1]) {id = 3 : i32} : (memref<8576xi8, 1 : i32>)
        %p1 = air.channel.put async [%g] @aL2ToL1[%c0_1, %c1_1] (%buf[%c4288_1] [%c4288_1] [%c1_1]) {id = 4 : i32} : (memref<8576xi8, 1 : i32>)
        %dt = air.execute [%p0, %p1] {
          memref.dealloc %buf : memref<8576xi8, 1 : i32>
        }
        %wa = air.wait_all async [%p0, %p1]
        scf.yield %wa : !air.async.token
      }
      %h = air.herd @h async tile (%hx, %hy) in (%hsx=%c4, %hsy=%c2_1) attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
        %c0_2 = arith.constant 0 : index
        %c1_2 = arith.constant 1 : index
        %c24_2 = arith.constant 24 : index
        %wh = air.wait_all async
        %fh = scf.for %j = %c0_2 to %c24_2 step %c1_2 iter_args(%ith = %wh) -> (!air.async.token) {
          %tokl, %bufl = air.execute -> (memref<4288xi8, 2 : i32>) {
            %alloc = memref.alloc() : memref<4288xi8, 2 : i32>
            air.execute_terminator %alloc : memref<4288xi8, 2 : i32>
          }
          %gl = air.channel.get async [%ith, %tokl] @aL2ToL1[%hx, %hy] (%bufl[] [] []) {id = 5 : i32} : (memref<4288xi8, 2 : i32>)
          %dl = air.execute [%gl] {
            memref.dealloc %bufl : memref<4288xi8, 2 : i32>
          }
          scf.yield %gl : !air.async.token
        }
      }
    }
  }
  return
}
