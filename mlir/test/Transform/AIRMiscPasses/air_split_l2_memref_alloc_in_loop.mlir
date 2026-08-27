//===- air_split_l2_memref_alloc_in_loop.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s --air-split-l2-memref="tiles-per-l2-tile=1" --split-input-file | FileCheck %s

// An L2 staging buffer may be allocated INSIDE the scf.for that stages it --
// one buffer per iteration, which is what a double-buffered per-column feed
// looks like once air-isolate-async-dma-loop-nests has run. The pass's own
// documented example allocates the L2 buffer above the loop, and that
// undocumented assumption was load-bearing in the dependency repair:
//
// `partitionMemref` finishes by calling `traceDependencyFromScfForOp` on every
// scf.for whose channel ops it rewrote. That helper puts an empty air.wait_all
// immediately BEFORE the loop, makes it the loop's iter_arg init, then walks the
// loop BODY attaching the producer of every buffer the body touches. With the
// buffer allocated above the loop the producer is above it too and the result
// dominates; with the buffer allocated in the body the producers are in-loop
// ops, so the init operand becomes a use its own definition does not dominate
// and air-opt rejects the module with
//   error: operand #0 does not dominate this use
// before FileCheck ever runs. Those in-loop producers are already dependences of
// the in-loop consumers, so only tokens defined above the loop belong on the
// init.
//
// Cases 1 and 2 cover both directions, because the failure is in the shared
// dependency repair rather than in either channel direction. Case 3 is the
// control that must keep splitting as before. Case 4 covers the far-side lookup
// at more than one column.
//
// Every case here stages one far-side execution for one L2-side execution, and
// splits the L2 buffer along a real memref dimension whose extent the far side
// carries too. The shape where the far side instead delivers the whole stage
// loop's worth at once is one the pass declines; it is in
// air_split_l2_memref_repeated_feed.mlir.

// -----

// Case 1 -- MM2S side. Per-iteration L3->L2 get of a [2, 4288] tile pair, two
// L2->L1 puts one row each, L2 buffer allocated per iteration inside the
// scf.for. Split factor 2, along memref dimension 0.

// CHECK-LABEL: func.func @alloc_in_loop_mm2s
// CHECK: scf.for
// CHECK-COUNT-2: memref.alloc() : memref<1x4288xi8, 1 : i32>
// CHECK-NOT: memref.alloc() : memref<2x4288xi8, 1 : i32>

#map = affine_map<()[s0] -> (s0 * 2)>
air.channel @inL3 [1]
air.channel @aL2ToL1 [1, 2]
func.func @alloc_in_loop_mm2s(%arg0: memref<48x4288xi8>) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%a0=%arg0) : memref<48x4288xi8> attributes {id = 1 : i32} {
    %c0 = arith.constant 0 : index
    %c1_0 = arith.constant 1 : index
    %c2_0 = arith.constant 2 : index
    %c24_0 = arith.constant 24 : index
    %c4288 = arith.constant 4288 : index
    %w = air.wait_all async
    %fl = scf.for %i = %c0 to %c24_0 step %c1_0 iter_args(%it = %w) -> (!air.async.token) {
      %tk, %off = air.execute -> (index) {
        %e = affine.apply #map()[%i]
        air.execute_terminator %e : index
      }
      %p = air.channel.put async [%it, %tk] @inL3[%c0] (%a0[%off, %c0] [%c2_0, %c4288] [%c4288, %c1_0]) {id = 1 : i32} : (memref<48x4288xi8>)
      scf.yield %p : !air.async.token
    }
    %s = air.segment @seg async attributes {id = 2 : i32} {
      %c0_1 = arith.constant 0 : index
      %c1_1 = arith.constant 1 : index
      %c2_1 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c24 = arith.constant 24 : index
      %c4288_1 = arith.constant 4288 : index
      %w0 = air.wait_all async
      %f = scf.for %i = %c0_1 to %c24 step %c1_1 iter_args(%it = %w0) -> (!air.async.token) {
        %tok, %buf = air.execute -> (memref<2x4288xi8, 1 : i32>) {
          %alloc = memref.alloc() : memref<2x4288xi8, 1 : i32>
          air.execute_terminator %alloc : memref<2x4288xi8, 1 : i32>
        }
        %g = air.channel.get async [%it, %tok] @inL3[%c0_1] (%buf[] [] []) {id = 2 : i32} : (memref<2x4288xi8, 1 : i32>)
        %p0 = air.channel.put async [%g] @aL2ToL1[%c0_1, %c0_1] (%buf[%c0_1, %c0_1] [%c1_1, %c4288_1] [%c4288_1, %c1_1]) {id = 3 : i32} : (memref<2x4288xi8, 1 : i32>)
        %p1 = air.channel.put async [%g] @aL2ToL1[%c0_1, %c1_1] (%buf[%c1_1, %c0_1] [%c1_1, %c4288_1] [%c4288_1, %c1_1]) {id = 4 : i32} : (memref<2x4288xi8, 1 : i32>)
        %dt = air.execute [%p0, %p1] {
          memref.dealloc %buf : memref<2x4288xi8, 1 : i32>
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

// Case 2 -- S2MM side. Two L1->L2 gets one row each, per-iteration L2->L3 put,
// L2 buffer allocated per iteration inside the scf.for. Split factor 2.

// CHECK-LABEL: func.func @alloc_in_loop_s2mm
// CHECK: scf.for
// CHECK-COUNT-2: memref.alloc() : memref<1x8xbf16, 1 : i32>
// CHECK-NOT: memref.alloc() : memref<2x8xbf16, 1 : i32>

#map = affine_map<()[s0] -> (s0 * 2)>
air.channel @dL1ToL2 [1, 2]
air.channel @outD [1]
func.func @alloc_in_loop_s2mm(%arg0: memref<48x8xbf16>) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%a0=%arg0) : memref<48x8xbf16> attributes {id = 1 : i32} {
    %c0 = arith.constant 0 : index
    %c1_0 = arith.constant 1 : index
    %c2_0 = arith.constant 2 : index
    %c8_0 = arith.constant 8 : index
    %c24_0 = arith.constant 24 : index
    %w = air.wait_all async
    %fl = scf.for %i = %c0 to %c24_0 step %c1_0 iter_args(%it = %w) -> (!air.async.token) {
      %tk, %off = air.execute -> (index) {
        %e = affine.apply #map()[%i]
        air.execute_terminator %e : index
      }
      %g = air.channel.get async [%it, %tk] @outD[%c0] (%a0[%off, %c0] [%c2_0, %c8_0] [%c8_0, %c1_0]) {id = 1 : i32} : (memref<48x8xbf16>)
      scf.yield %g : !air.async.token
    }
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
        %tok, %buf = air.execute -> (memref<2x8xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<2x8xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<2x8xbf16, 1 : i32>
        }
        %g0 = air.channel.get async [%it, %tok] @dL1ToL2[%c0_1, %c0_1] (%buf[%c0_1, %c0_1] [%c1_1, %c8] [%c8, %c1_1]) {id = 3 : i32} : (memref<2x8xbf16, 1 : i32>)
        %g1 = air.channel.get async [%it, %tok] @dL1ToL2[%c0_1, %c1_1] (%buf[%c1_1, %c0_1] [%c1_1, %c8] [%c8, %c1_1]) {id = 4 : i32} : (memref<2x8xbf16, 1 : i32>)
        %pp = air.channel.put async [%g0, %g1] @outD[%c0_1] (%buf[] [] []) {id = 5 : i32} : (memref<2x8xbf16, 1 : i32>)
        %dt = air.execute [%pp] {
          memref.dealloc %buf : memref<2x8xbf16, 1 : i32>
        }
        scf.yield %pp : !air.async.token
      }
    }
  }
  return
}

// -----

// Case 3 -- CONTROL. Same MM2S shape as case 1 with the L2 buffer allocated
// once ABOVE the scf.for. This is the shape the pass was written for; it must
// keep splitting into two one-row sub-buffers.

// CHECK-LABEL: func.func @alloc_above_loop_mm2s
// CHECK-COUNT-2: memref.alloc() : memref<1x4288xi8, 1 : i32>
// CHECK-NOT: memref.alloc() : memref<2x4288xi8, 1 : i32>

#map = affine_map<()[s0] -> (s0 * 2)>
air.channel @inL3 [1]
air.channel @aL2ToL1 [1, 2]
func.func @alloc_above_loop_mm2s(%arg0: memref<48x4288xi8>) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%a0=%arg0) : memref<48x4288xi8> attributes {id = 1 : i32} {
    %c0 = arith.constant 0 : index
    %c1_0 = arith.constant 1 : index
    %c2_0 = arith.constant 2 : index
    %c24_0 = arith.constant 24 : index
    %c4288 = arith.constant 4288 : index
    %w = air.wait_all async
    %fl = scf.for %i = %c0 to %c24_0 step %c1_0 iter_args(%it = %w) -> (!air.async.token) {
      %tk, %off = air.execute -> (index) {
        %e = affine.apply #map()[%i]
        air.execute_terminator %e : index
      }
      %p = air.channel.put async [%it, %tk] @inL3[%c0] (%a0[%off, %c0] [%c2_0, %c4288] [%c4288, %c1_0]) {id = 1 : i32} : (memref<48x4288xi8>)
      scf.yield %p : !air.async.token
    }
    %s = air.segment @seg async attributes {id = 2 : i32} {
      %c0_1 = arith.constant 0 : index
      %c1_1 = arith.constant 1 : index
      %c2_1 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c24 = arith.constant 24 : index
      %c4288_1 = arith.constant 4288 : index
      %tok, %buf = air.execute -> (memref<2x4288xi8, 1 : i32>) {
        %alloc = memref.alloc() : memref<2x4288xi8, 1 : i32>
        air.execute_terminator %alloc : memref<2x4288xi8, 1 : i32>
      }
      %w0 = air.wait_all async [%tok]
      %f = scf.for %i = %c0_1 to %c24 step %c1_1 iter_args(%it = %w0) -> (!air.async.token) {
        %g = air.channel.get async [%it, %tok] @inL3[%c0_1] (%buf[] [] []) {id = 2 : i32} : (memref<2x4288xi8, 1 : i32>)
        %p0 = air.channel.put async [%g] @aL2ToL1[%c0_1, %c0_1] (%buf[%c0_1, %c0_1] [%c1_1, %c4288_1] [%c4288_1, %c1_1]) {id = 3 : i32} : (memref<2x4288xi8, 1 : i32>)
        %p1 = air.channel.put async [%g] @aL2ToL1[%c0_1, %c1_1] (%buf[%c1_1, %c0_1] [%c1_1, %c4288_1] [%c4288_1, %c1_1]) {id = 4 : i32} : (memref<2x4288xi8, 1 : i32>)
        %wa = air.wait_all async [%p0, %p1]
        scf.yield %wa : !air.async.token
      }
      %dt = air.execute [%f] {
        memref.dealloc %buf : memref<2x4288xi8, 1 : i32>
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

// Case 4 -- TWO columns, one L2 buffer each, both staged through the same
// channel SYMBOL at different bundle indices. This is the per-column staging
// pattern, and it is where the far-side lookup being symbol-scoped shows:
// `getTheOtherChannelOpThroughSymbol` hands back every put on @inL3 for each of
// the two buffers, so each put is tiled twice over, and the copies made while
// processing the first buffer are still in the IR (they are only queued for
// erasure and removed at the end of the pass) so the second buffer tiles those
// too. Four launch-level puts became twelve, against four gets; at four columns
// it was 120 against 8, which compiles and then hangs the device with
// ERT_CMD_STATE_TIMEOUT. Exactly one tiled put per tiled get is the invariant.

// CHECK-LABEL: func.func @two_columns_one_symbol
// CHECK-COUNT-4: air.channel.put{{.*}}@channel_0
// CHECK-NOT: air.channel.put{{.*}}@channel_0

#map = affine_map<()[s0] -> (s0 * 2)>
#map1 = affine_map<()[s0] -> (s0 * 2 + 48)>
air.channel @inL3 [2]
air.channel @aL2ToL1 [2, 2]
func.func @two_columns_one_symbol(%arg0: memref<96x4288xi8>) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%a0=%arg0) : memref<96x4288xi8> attributes {id = 1 : i32} {
    %c0 = arith.constant 0 : index
    %c1_0 = arith.constant 1 : index
    %c2_0 = arith.constant 2 : index
    %c24_0 = arith.constant 24 : index
    %c4288 = arith.constant 4288 : index
    %w = air.wait_all async
    %fl0 = scf.for %i = %c0 to %c24_0 step %c1_0 iter_args(%it = %w) -> (!air.async.token) {
      %tk, %off = air.execute -> (index) {
        %e = affine.apply #map()[%i]
        air.execute_terminator %e : index
      }
      %p = air.channel.put async [%it, %tk] @inL3[%c0] (%a0[%off, %c0] [%c2_0, %c4288] [%c4288, %c1_0]) {id = 1 : i32} : (memref<96x4288xi8>)
      scf.yield %p : !air.async.token
    }
    %fl1 = scf.for %i = %c0 to %c24_0 step %c1_0 iter_args(%it = %w) -> (!air.async.token) {
      %tk, %off = air.execute -> (index) {
        %e = affine.apply #map1()[%i]
        air.execute_terminator %e : index
      }
      %p = air.channel.put async [%it, %tk] @inL3[%c1_0] (%a0[%off, %c0] [%c2_0, %c4288] [%c4288, %c1_0]) {id = 2 : i32} : (memref<96x4288xi8>)
      scf.yield %p : !air.async.token
    }
    %s = air.segment @seg async attributes {id = 2 : i32} {
      %c0_1 = arith.constant 0 : index
      %c1_1 = arith.constant 1 : index
      %c2_1 = arith.constant 2 : index
      %c24 = arith.constant 24 : index
      %c4288_1 = arith.constant 4288 : index
      %w0 = air.wait_all async
      %f0 = scf.for %i = %c0_1 to %c24 step %c1_1 iter_args(%it = %w0) -> (!air.async.token) {
        %tok, %buf = air.execute -> (memref<2x4288xi8, 1 : i32>) {
          %alloc = memref.alloc() : memref<2x4288xi8, 1 : i32>
          air.execute_terminator %alloc : memref<2x4288xi8, 1 : i32>
        }
        %g = air.channel.get async [%it, %tok] @inL3[%c0_1] (%buf[] [] []) {id = 3 : i32} : (memref<2x4288xi8, 1 : i32>)
        %q0 = air.channel.put async [%g] @aL2ToL1[%c0_1, %c0_1] (%buf[%c0_1, %c0_1] [%c1_1, %c4288_1] [%c4288_1, %c1_1]) {id = 4 : i32} : (memref<2x4288xi8, 1 : i32>)
        %q1 = air.channel.put async [%g] @aL2ToL1[%c0_1, %c1_1] (%buf[%c1_1, %c0_1] [%c1_1, %c4288_1] [%c4288_1, %c1_1]) {id = 5 : i32} : (memref<2x4288xi8, 1 : i32>)
        %dt = air.execute [%q0, %q1] {
          memref.dealloc %buf : memref<2x4288xi8, 1 : i32>
        }
        %wa = air.wait_all async [%q0, %q1]
        scf.yield %wa : !air.async.token
      }
      %w1 = air.wait_all async
      %f1 = scf.for %i = %c0_1 to %c24 step %c1_1 iter_args(%it = %w1) -> (!air.async.token) {
        %tok, %buf = air.execute -> (memref<2x4288xi8, 1 : i32>) {
          %alloc = memref.alloc() : memref<2x4288xi8, 1 : i32>
          air.execute_terminator %alloc : memref<2x4288xi8, 1 : i32>
        }
        %g = air.channel.get async [%it, %tok] @inL3[%c1_1] (%buf[] [] []) {id = 6 : i32} : (memref<2x4288xi8, 1 : i32>)
        %q0 = air.channel.put async [%g] @aL2ToL1[%c1_1, %c0_1] (%buf[%c0_1, %c0_1] [%c1_1, %c4288_1] [%c4288_1, %c1_1]) {id = 7 : i32} : (memref<2x4288xi8, 1 : i32>)
        %q1 = air.channel.put async [%g] @aL2ToL1[%c1_1, %c1_1] (%buf[%c1_1, %c0_1] [%c1_1, %c4288_1] [%c4288_1, %c1_1]) {id = 8 : i32} : (memref<2x4288xi8, 1 : i32>)
        %dt = air.execute [%q0, %q1] {
          memref.dealloc %buf : memref<2x4288xi8, 1 : i32>
        }
        %wa = air.wait_all async [%q0, %q1]
        scf.yield %wa : !air.async.token
      }
      %h = air.herd @h async tile (%hx, %hy) in (%hsx=%c2_1, %hsy=%c2_1) attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
        %c0_2 = arith.constant 0 : index
        %c1_2 = arith.constant 1 : index
        %c24_2 = arith.constant 24 : index
        %wh = air.wait_all async
        %fh = scf.for %j = %c0_2 to %c24_2 step %c1_2 iter_args(%ith = %wh) -> (!air.async.token) {
          %tokl, %bufl = air.execute -> (memref<4288xi8, 2 : i32>) {
            %alloc = memref.alloc() : memref<4288xi8, 2 : i32>
            air.execute_terminator %alloc : memref<4288xi8, 2 : i32>
          }
          %gl = air.channel.get async [%ith, %tokl] @aL2ToL1[%hx, %hy] (%bufl[] [] []) {id = 9 : i32} : (memref<4288xi8, 2 : i32>)
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
