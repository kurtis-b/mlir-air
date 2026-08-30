//===- air_split_l2_memref_short_offset_list.mlir -------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s --air-split-l2-memref="tiles-per-l2-tile=1" --split-input-file | FileCheck %s

// An air.channel put/get may legally carry FEWER offsets than sizes/strides:
// the verifier (verifySizesStridesRank) only ties sizes to strides, and the
// shipped builders emit exactly that for an L3 operand --
//   air.channel.get @outD[%c0] (%arg[%off] [24, 2, 8] [8, 192, 1])
// is one offset into a rank-1 memref carrying a three-dimensional wrap-and-
// stride pattern.
//
// `tileChannelOpByFactor` indexes each participating op's offset list at the
// split dimension, which the pass derives from the L2 side and reuses verbatim
// on the far side of the channel. When the L2 buffer is split on a non-leading
// dimension the split dimension is 1, and on such an op offset #1 does not
// exist: the pass read past the end of the SmallVector and air-opt died with
//   Assertion `idx < size()' failed
//   #10 xilinx::tileChannelOpByFactor(...)
//   #11 xilinx::AIRSplitL2MemrefForBufferConstraintPass::runOnOperation()
// (SIGABRT, aircc reports it as "returncode -6").
//
// A split it cannot express is not an error; the pass already declines several
// shapes. Leave the buffer intact.

// CHECK-LABEL: func.func @short_offset_list_on_far_side
// CHECK: memref.alloc() : memref<24x16xbf16, 1 : i32>
// CHECK-NOT: memref.alloc() : memref<24x8xbf16, 1 : i32>

air.channel @dL1ToL2 [1, 2]
air.channel @outD [1]
func.func @short_offset_list_on_far_side(%arg0: memref<384xbf16>) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%a0=%arg0) : memref<384xbf16> attributes {id = 1 : i32} {
    %c0 = arith.constant 0 : index
    %c1_0 = arith.constant 1 : index
    %c2_0 = arith.constant 2 : index
    %c8_0 = arith.constant 8 : index
    %c24_0 = arith.constant 24 : index
    %c192 = arith.constant 192 : index
    // One offset, three sizes and three strides.
    %g = air.channel.get async @outD[%c0] (%a0[%c0] [%c24_0, %c2_0, %c8_0] [%c8_0, %c192, %c1_0]) {id = 1 : i32} : (memref<384xbf16>)
    %s = air.segment @seg async attributes {id = 2 : i32} {
      %c0_1 = arith.constant 0 : index
      %c1_1 = arith.constant 1 : index
      %c2_1 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
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
      // Two gets on the leading dimension's full extent, differing at dim 1:
      // the split dimension is 1, not 0.
      %tok, %buf = air.execute -> (memref<24x16xbf16, 1 : i32>) {
        %alloc = memref.alloc() : memref<24x16xbf16, 1 : i32>
        air.execute_terminator %alloc : memref<24x16xbf16, 1 : i32>
      }
      %g0 = air.channel.get async [%h, %tok] @dL1ToL2[%c0_1, %c0_1] (%buf[%c0_1, %c0_1] [%c24, %c8] [%c16, %c1_1]) {id = 3 : i32} : (memref<24x16xbf16, 1 : i32>)
      %g1 = air.channel.get async [%h, %tok] @dL1ToL2[%c0_1, %c1_1] (%buf[%c0_1, %c8] [%c24, %c8] [%c16, %c1_1]) {id = 4 : i32} : (memref<24x16xbf16, 1 : i32>)
      %pp = air.channel.put async [%g0, %g1] @outD[%c0_1] (%buf[] [] []) {id = 5 : i32} : (memref<24x16xbf16, 1 : i32>)
      %dt = air.execute [%pp] {
        memref.dealloc %buf : memref<24x16xbf16, 1 : i32>
      }
    }
  }
  return
}

// -----

// Case 2 -- the degenerate end of the same freedom: a far-side op carrying ZERO
// offsets and one size. Legal for the same reason (the verifier ties sizes to
// strides, not offsets), and here the reference side is a rank-2 buffer split on
// dimension 0.
//
// The first version of the precheck credited the rank-matching step in
// `runOnOperation` with growing this op's offset list from 0 to 1 and approved
// it. That step cannot run at all on an empty list: it starts at
// `offsets.size() - 1`, which underflows to -1 and indexes off the front of the
// SmallVector -- `Assertion 'idx < size()' failed`, exit 134, on the very
// compiler the earlier fix had produced. Nothing to model means decline.

// CHECK-LABEL: func.func @zero_offset_far_side
// CHECK: memref.alloc() : memref<2x4288xi8, 1 : i32>
// CHECK-NOT: memref.alloc() : memref<1x4288xi8, 1 : i32>

air.channel @inL3 [1]
air.channel @aL2ToL1 [1, 2]
func.func @zero_offset_far_side(%arg0: memref<8576xi8>) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%a0=%arg0) : memref<8576xi8> attributes {id = 1 : i32} {
    %c0 = arith.constant 0 : index
    %c1_0 = arith.constant 1 : index
    %c8576 = arith.constant 8576 : index
    // Sizes and strides agree in rank; offsets is empty.
    %p = air.channel.put async @inL3[%c0] (%a0[] [%c8576] [%c1_0]) {id = 1 : i32} : (memref<8576xi8>)
    %s = air.segment @seg async attributes {id = 2 : i32} {
      %c0_1 = arith.constant 0 : index
      %c1_1 = arith.constant 1 : index
      %c2_1 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c4288_1 = arith.constant 4288 : index
      %tok, %buf = air.execute -> (memref<2x4288xi8, 1 : i32>) {
        %alloc = memref.alloc() : memref<2x4288xi8, 1 : i32>
        air.execute_terminator %alloc : memref<2x4288xi8, 1 : i32>
      }
      %g = air.channel.get async [%tok] @inL3[%c0_1] (%buf[] [] []) {id = 2 : i32} : (memref<2x4288xi8, 1 : i32>)
      %p0 = air.channel.put async [%g] @aL2ToL1[%c0_1, %c0_1] (%buf[%c0_1, %c0_1] [%c1_1, %c4288_1] [%c4288_1, %c1_1]) {id = 3 : i32} : (memref<2x4288xi8, 1 : i32>)
      %p1 = air.channel.put async [%g] @aL2ToL1[%c0_1, %c1_1] (%buf[%c1_1, %c0_1] [%c1_1, %c4288_1] [%c4288_1, %c1_1]) {id = 4 : i32} : (memref<2x4288xi8, 1 : i32>)
      %dt = air.execute [%p0, %p1] {
        memref.dealloc %buf : memref<2x4288xi8, 1 : i32>
      }
      %h = air.herd @h async tile (%hx, %hy) in (%hsx=%c4, %hsy=%c2_1) attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
        %tokl, %bufl = air.execute -> (memref<4288xi8, 2 : i32>) {
          %alloc = memref.alloc() : memref<4288xi8, 2 : i32>
          air.execute_terminator %alloc : memref<4288xi8, 2 : i32>
        }
        %gl = air.channel.get async [%tokl] @aL2ToL1[%hx, %hy] (%bufl[] [] []) {id = 5 : i32} : (memref<4288xi8, 2 : i32>)
        %dl = air.execute [%gl] {
          memref.dealloc %bufl : memref<4288xi8, 2 : i32>
        }
      }
    }
  }
  return
}

// -----

// A far-side op whose offset list is short AND whose rank matching against the
// L2 side's [24, 16] stops early: one offset, one size 24. The rank-matching
// step in `runOnOperation` breaks at 24 % 16, so no offset is ever inserted and
// offset #1 still does not exist when `wraps[1]` is read -- before the tiler's
// guard. The plan-time check must model the match exactly, not bound it.

// CHECK-LABEL: func.func @rank_match_stops_short
// CHECK: memref.alloc() : memref<24x16xbf16, 1 : i32>
// CHECK-NOT: memref.alloc() : memref<24x8xbf16, 1 : i32>

air.channel @dL1ToL2 [1, 2]
air.channel @outD [1]
func.func @rank_match_stops_short(%arg0: memref<384xbf16>) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%a0=%arg0) : memref<384xbf16> attributes {id = 1 : i32} {
    %c0 = arith.constant 0 : index
    %c1_0 = arith.constant 1 : index
    %c2_0 = arith.constant 2 : index
    %c8_0 = arith.constant 8 : index
    %c24_0 = arith.constant 24 : index
    %c192 = arith.constant 192 : index
    // One offset and one size: rank matching against [24, 16] stops at 24 % 16.
    %g = air.channel.get async @outD[%c0] (%a0[%c0] [%c24_0] [%c1_0]) {id = 1 : i32} : (memref<384xbf16>)
    %s = air.segment @seg async attributes {id = 2 : i32} {
      %c0_1 = arith.constant 0 : index
      %c1_1 = arith.constant 1 : index
      %c2_1 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
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
      // Two gets on the leading dimension's full extent, differing at dim 1:
      // the split dimension is 1, not 0.
      %tok, %buf = air.execute -> (memref<24x16xbf16, 1 : i32>) {
        %alloc = memref.alloc() : memref<24x16xbf16, 1 : i32>
        air.execute_terminator %alloc : memref<24x16xbf16, 1 : i32>
      }
      %g0 = air.channel.get async [%h, %tok] @dL1ToL2[%c0_1, %c0_1] (%buf[%c0_1, %c0_1] [%c24, %c8] [%c16, %c1_1]) {id = 3 : i32} : (memref<24x16xbf16, 1 : i32>)
      %g1 = air.channel.get async [%h, %tok] @dL1ToL2[%c0_1, %c1_1] (%buf[%c0_1, %c8] [%c24, %c8] [%c16, %c1_1]) {id = 4 : i32} : (memref<24x16xbf16, 1 : i32>)
      %pp = air.channel.put async [%g0, %g1] @outD[%c0_1] (%buf[] [] []) {id = 5 : i32} : (memref<24x16xbf16, 1 : i32>)
      %dt = air.execute [%pp] {
        memref.dealloc %buf : memref<24x16xbf16, 1 : i32>
      }
    }
  }
  return
}
