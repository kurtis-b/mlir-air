//===- air_channel_nonclean_rotation_refuse_repeated_buffer.mlir -*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// REFUSAL case for the non-clean-rotation plan (queue row 28(a)): two steady
// sites on the SAME buffer plus a peeled second buffer. Trip counts form the
// {q, q+1} prefix staircase ([2,2,1]) and the buffers are >= 2, but the group
// has three sites over two buffers -- NOT one full cycle of distinct buffers
// -- so a cycle+remainder program would reorder the A0 sites' firings. The
// plan must refuse and leave the legacy per-repeat-count bucketing byte-for-
// byte intact: a [A0,A0] task with repeat_count = 1, then a [A1] task.

// CHECK: aie.device
// CHECK-DAG:   %[[TILE:.*]] = aie.tile(2, 3)
// CHECK:       aie.mem(%[[TILE]])
// CHECK:         aie.dma_start(S2MM, 0, ^[[T1BB1:[^,]+]], ^{{[^,)]+}}, repeat_count = 1)
// CHECK:       ^[[T1BB1]]:
// CHECK:         aie.dma_bd(%[[A0:[^ ]+]] : memref<32x32xbf16, 2>
// CHECK:         aie.dma_bd(%[[A0]] : memref<32x32xbf16, 2>
// CHECK-NOT:     aie.dma_bd
// CHECK:         aie.dma_start(S2MM, 0, ^[[T2BB1:[^,]+]], ^{{[^,)]+}})
// CHECK:       ^[[T2BB1]]:
// CHECK:         aie.dma_bd(%{{[^ ]+}} : memref<32x32xbf16, 2>
// CHECK-NOT:     aie.dma_bd

air.channel @channel_0 [1, 1]
func.func @refuse_repeated_buffer() {
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
        %s = scf.for %i = %c0 to %c2 step %c1_h iter_args(%dep = %t0) -> (!air.async.token) {
          %g0 = air.channel.get async [%dep, %t0] @channel_0[] (%bufa0[] [] []) : (memref<32x32xbf16, 2>)
          %g1 = air.channel.get async [%g0] @channel_0[] (%bufa0[] [] []) : (memref<32x32xbf16, 2>)
          scf.yield %g1 : !air.async.token
        }
        %p0 = air.channel.get async [%s, %t1] @channel_0[] (%bufa1[] [] []) : (memref<32x32xbf16, 2>)
        %d0 = air.execute [%p0] { memref.dealloc %bufa0 : memref<32x32xbf16, 2> }
        %d1 = air.execute [%p0] { memref.dealloc %bufa1 : memref<32x32xbf16, 2> }
      }
    }
  }
  return
}
