//===- non_constant_bd_offset.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" -verify-diagnostics -split-input-file

// A channel put whose offset depends on a loop induction variable cannot
// lower to a tile-side (L1/L2) BD: an aie.dma_bd offset is static and cannot
// advance per iteration. air-to-aie used to substitute constant 0 for the
// offset silently — every BD in the chain then addressed block 0 forever, and
// the design compiled, placed, routed, and hung on hardware with no hint. It
// must refuse instead, and the diagnostic must say what to do: stage the
// operand per iteration from L3, whose runtime-sequence-programmed transfers
// can materialize a moving offset.

#map = affine_map<()[s0] -> (s0 * 32)>
air.channel @channel_iv [1, 1]
func.func @l2_iv_offset_refused() {
  %c1 = arith.constant 1 : index
  air.launch (%arg0, %arg1) in (%arg2=%c1, %arg3=%c1) {
    air.segment @seg {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c1024 = arith.constant 1024 : index
      %alloc = memref.alloc() : memref<4096xbf16, 1>
      // expected-note@+1 {{the value varies with the induction variable of this loop}}
      scf.for %iv = %c0 to %c4 step %c1_0 {
        // expected-note@+1 {{non-constant offset produced by 'affine.apply' here}}
        %off = affine.apply #map()[%iv]
        // expected-error@+1 {{channel @channel_iv: BD offset is not a compile-time constant}}
        air.channel.put @channel_iv[] (%alloc[%off] [%c1024] [%c1_0]) : (memref<4096xbf16, 1>)
      }
      air.herd @herd_0 tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) {
        %buf = memref.alloc() : memref<1024xbf16, 2>
        air.channel.get @channel_iv[] (%buf[] [] []) : (memref<1024xbf16, 2>)
        memref.dealloc %buf : memref<1024xbf16, 2>
      }
      memref.dealloc %alloc : memref<4096xbf16, 1>
    }
  }
  return
}

// -----

// Control: the same shape with a compile-time-constant offset still lowers.
// No diagnostics expected.

air.channel @channel_const [1, 1]
func.func @l2_const_offset_lowers() {
  %c1 = arith.constant 1 : index
  air.launch (%arg0, %arg1) in (%arg2=%c1, %arg3=%c1) {
    air.segment @seg {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c1024 = arith.constant 1024 : index
      %alloc = memref.alloc() : memref<4096xbf16, 1>
      scf.for %iv = %c0 to %c4 step %c1_0 {
        air.channel.put @channel_const[] (%alloc[%c1024] [%c1024] [%c1_0]) : (memref<4096xbf16, 1>)
      }
      air.herd @herd_0 tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) {
        %buf = memref.alloc() : memref<1024xbf16, 2>
        air.channel.get @channel_const[] (%buf[] [] []) : (memref<1024xbf16, 2>)
        memref.dealloc %buf : memref<1024xbf16, 2>
      }
      memref.dealloc %alloc : memref<4096xbf16, 1>
    }
  }
  return
}

// -----

// Scope: an IV-dependent offset on an *L3* operand still compiles. L3-side
// transfers are programmed per task by the runtime sequence, which can
// materialize a moving offset — this is the form every shipped design that
// walks a staged buffer uses, and it must stay untouched. No diagnostics
// expected.

#map = affine_map<()[s0] -> (s0 * 1024)>
air.channel @channel_l3 [1, 1]
func.func @l3_iv_offset_compiles(%ext: memref<4096xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%arg0, %arg1) in (%arg2=%c1, %arg3=%c1) args(%l3 = %ext) : memref<4096xbf16> {
    %c0 = arith.constant 0 : index
    %c1_0 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c1024 = arith.constant 1024 : index
    scf.for %iv = %c0 to %c4 step %c1_0 {
      %off = affine.apply #map()[%iv]
      air.channel.put @channel_l3[] (%l3[%off] [%c1024] [%c1_0]) : (memref<4096xbf16>)
    }
    air.segment @seg {
      %c1_1 = arith.constant 1 : index
      air.herd @herd_0 tile (%tx, %ty) in (%sx=%c1_1, %sy=%c1_1) {
        %buf = memref.alloc() : memref<1024xbf16, 2>
        air.channel.get @channel_l3[] (%buf[] [] []) : (memref<1024xbf16, 2>)
        memref.dealloc %buf : memref<1024xbf16, 2>
      }
    }
  }
  return
}
