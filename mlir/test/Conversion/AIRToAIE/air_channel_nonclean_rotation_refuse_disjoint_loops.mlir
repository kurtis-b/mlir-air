//===- air_channel_nonclean_rotation_refuse_disjoint_loops.mlir -*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// REFUSAL case for the non-clean-rotation plan (queue row 28(a)): sites in
// DISJOINT loops. A0 rotates alone in its own trip-2 loop, two equivalent B
// sites share a SEPARATE trip-2 loop, A1 is peeled after both. Program order
// [A0,B,B,A1] with trips [2,2,2,1] passes the staircase and every group
// check -- the A group supplies rotation evidence, the B group supplies
// shared-loop evidence -- but the true firing order is bursts per loop
// (A0,A0,B,B,B,B,A1), not the cycle interleave, so a cycle+remainder program
// would silently reorder. The plan must refuse (every run site's one loop
// ancestor must be one shared steady loop) and leave the legacy bucketing
// intact: a 3-BD task with repeat_count = 1, then a 1-BD task.

// CHECK: aie.device
// CHECK-DAG:   %[[TILE:.*]] = aie.tile(2, 3)
// CHECK:       aie.mem(%[[TILE]])
// CHECK:         aie.dma_start(S2MM, 0, ^[[T1BB1:[^,]+]], ^{{[^,)]+}}, repeat_count = 1)
// CHECK:       ^[[T1BB1]]:
// CHECK:         aie.dma_bd(%{{[^ ]+}} : memref<32x32xbf16, 2>
// CHECK:         aie.dma_bd(%[[B:[^ ]+]] : memref<16x32xbf16, 2>
// CHECK:         aie.dma_bd(%[[B]] : memref<16x32xbf16, 2>
// CHECK-NOT:     aie.dma_bd
// CHECK:         aie.dma_start(S2MM, 0, ^[[T2BB1:[^,]+]], ^{{[^,)]+}})
// CHECK:       ^[[T2BB1]]:
// CHECK:         aie.dma_bd(%{{[^ ]+}} : memref<32x32xbf16, 2>
// CHECK-NOT:     aie.dma_bd

air.channel @channel_0 [1, 1]
func.func @refuse_disjoint_loops() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%a, %b) in (%c=%c1, %d=%c1) {
    %1 = air.segment async {
      %c1_0 = arith.constant 1 : index
      %2 = air.herd @herd_0 async tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) {
        %c0 = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %t0, %bufa0 = air.execute -> (memref<32x32xbf16, 2>) {
          %m = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %m : memref<32x32xbf16, 2>
        }
        %t1, %bufa1 = air.execute -> (memref<32x32xbf16, 2>) {
          %m = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %m : memref<32x32xbf16, 2>
        }
        %t2, %bufb = air.execute -> (memref<16x32xbf16, 2>) {
          %m = memref.alloc() : memref<16x32xbf16, 2>
          air.execute_terminator %m : memref<16x32xbf16, 2>
        }
        %s0 = scf.for %i = %c0 to %c2 step %c1_h iter_args(%dep = %t0) -> (!air.async.token) {
          %g0 = air.channel.get async [%dep, %t0] @channel_0[] (%bufa0[] [] []) : (memref<32x32xbf16, 2>)
          scf.yield %g0 : !air.async.token
        }
        %s1 = scf.for %i = %c0 to %c2 step %c1_h iter_args(%dep = %s0) -> (!air.async.token) {
          %g0 = air.channel.get async [%dep, %t2] @channel_0[] (%bufb[] [] []) : (memref<16x32xbf16, 2>)
          %g1 = air.channel.get async [%g0] @channel_0[] (%bufb[] [] []) : (memref<16x32xbf16, 2>)
          scf.yield %g1 : !air.async.token
        }
        %p0 = air.channel.get async [%s1, %t1] @channel_0[] (%bufa1[] [] []) : (memref<32x32xbf16, 2>)
        %d0 = air.execute [%p0] { memref.dealloc %bufa0 : memref<32x32xbf16, 2> }
        %d1 = air.execute [%p0] { memref.dealloc %bufa1 : memref<32x32xbf16, 2> }
        %d2 = air.execute [%p0] { memref.dealloc %bufb : memref<16x32xbf16, 2> }
      }
    }
  }
  return
}
