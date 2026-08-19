//===- air_channel_nonclean_rotation_refuse_mixed_offsets.mlir --*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// REFUSAL case for the non-clean-rotation plan (queue row 28(a)): the same
// interleaved shape as air_channel_nonclean_rotation.mlir -- a 3-buffer A
// rotation multiplexed with a single-buffer B stream of a different memref
// type -- but B's sites read DIFFERENT offsets of the one buffer (the two
// steady sites read rows 0-7, the peel site reads rows 8-15). B's sites land
// in both plan tasks, so their mutual order is load-bearing -- a cycle+
// remainder program would deliver B's chunks to the wrong halves. The plan
// must refuse (single-buffer groups require pairwise-equivalent BDs, offsets
// included) and leave the legacy bucketing intact: a 4-BD task with
// repeat_count = 1, then a 2-BD task, with the row-8 access only in the
// second task.

// CHECK: aie.device
// CHECK-DAG:   %[[TILE:.*]] = aie.tile(2, 3)
// CHECK:       aie.mem(%[[TILE]])
// CHECK:         aie.dma_start(S2MM, 0, ^[[T1BB1:[^,]+]], ^{{[^,)]+}}, repeat_count = 1)
// CHECK:       ^[[T1BB1]]:
// CHECK:         aie.dma_bd(%{{[^ ]+}} : memref<32x32xbf16, 2>
// CHECK:         aie.dma_bd(%[[B:[^ ]+]] : memref<16x32xbf16, 2> offset = 0 len = 256
// CHECK:         aie.dma_bd(%{{[^ ]+}} : memref<32x32xbf16, 2>
// CHECK:         aie.dma_bd(%[[B]] : memref<16x32xbf16, 2> offset = 0 len = 256
// CHECK-NOT:     aie.dma_bd
// CHECK:         aie.dma_start(S2MM, 0, ^[[T2BB1:[^,]+]], ^{{[^,)]+}})
// CHECK:       ^[[T2BB1]]:
// CHECK:         aie.dma_bd(%{{[^ ]+}} : memref<32x32xbf16, 2>
// CHECK:         aie.dma_bd(%[[B]] : memref<16x32xbf16, 2> offset = 256 len = 256
// CHECK-NOT:     aie.dma_bd

air.channel @channel_0 [1, 1]
func.func @refuse_mixed_offsets() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%a, %b) in (%c=%c1, %d=%c1) {
    %1 = air.segment async {
      %c1_0 = arith.constant 1 : index
      %2 = air.herd @herd_0 async tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) {
        %c0 = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c8 = arith.constant 8 : index
        %c32 = arith.constant 32 : index
        %t0, %bufa0 = air.execute -> (memref<32x32xbf16, 2>) {
          %m = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %m : memref<32x32xbf16, 2>
        }
        %t1, %bufa1 = air.execute -> (memref<32x32xbf16, 2>) {
          %m = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %m : memref<32x32xbf16, 2>
        }
        %t2, %bufa2 = air.execute -> (memref<32x32xbf16, 2>) {
          %m = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %m : memref<32x32xbf16, 2>
        }
        %t3, %bufb = air.execute -> (memref<16x32xbf16, 2>) {
          %m = memref.alloc() : memref<16x32xbf16, 2>
          air.execute_terminator %m : memref<16x32xbf16, 2>
        }
        %s = scf.for %i = %c0 to %c2 step %c1_h iter_args(%dep = %t0) -> (!air.async.token) {
          %g0 = air.channel.get async [%dep, %t0] @channel_0[] (%bufa0[] [] []) : (memref<32x32xbf16, 2>)
          %g1 = air.channel.get async [%g0, %t3] @channel_0[] (%bufb[%c0, %c0] [%c8, %c32] [%c32, %c1_h]) : (memref<16x32xbf16, 2>)
          %g2 = air.channel.get async [%g1, %t1] @channel_0[] (%bufa1[] [] []) : (memref<32x32xbf16, 2>)
          %g3 = air.channel.get async [%g2] @channel_0[] (%bufb[%c0, %c0] [%c8, %c32] [%c32, %c1_h]) : (memref<16x32xbf16, 2>)
          scf.yield %g3 : !air.async.token
        }
        %p0 = air.channel.get async [%s, %t2] @channel_0[] (%bufa2[] [] []) : (memref<32x32xbf16, 2>)
        %p1 = air.channel.get async [%p0] @channel_0[] (%bufb[%c8, %c0] [%c8, %c32] [%c32, %c1_h]) : (memref<16x32xbf16, 2>)
        %d0 = air.execute [%p1] { memref.dealloc %bufa0 : memref<32x32xbf16, 2> }
        %d1 = air.execute [%p1] { memref.dealloc %bufa1 : memref<32x32xbf16, 2> }
        %d2 = air.execute [%p1] { memref.dealloc %bufa2 : memref<32x32xbf16, 2> }
        %d3 = air.execute [%p1] { memref.dealloc %bufb : memref<16x32xbf16, 2> }
      }
    }
  }
  return
}
