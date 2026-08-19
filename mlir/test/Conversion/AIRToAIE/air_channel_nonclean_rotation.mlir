//===- air_channel_nonclean_rotation.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s --implicit-check-not=repeat_count

// NON-CLEAN rotation on a MULTIPLEXED channel (queue row 28(a), doc 52 §10).
// One bundled channel carries two interleaved payload streams -- a 3-buffer
// rotation of A chunks and a single-buffer B stream -- with the steady loop
// (trip 2) carrying (A0, B, A1, B) and an epilogue peel carrying (A2, B).
// The total firing count (10) is not a multiple of the cycle (6), and the
// mixed payload types stop detectNBufferRotation's circular-chain path at its
// memref-type check, so before the fix this fell to per-repeat-count tasks:
// a [A0,B,A1,B] task with repeat_count = 1 (executes twice) followed by an
// [A2,B] task -- replaying a PREFIX of the rotation out of phase against the
// producer's chain. Measured on hardware as a byte-deterministic permutation
// of the delivered stream (delivered order [0,1,3,4,2] at down_K = 5).
//
// The only order-preserving program is the WHOLE cycle once (q = 1) followed
// by the first r = 4 BDs once: visits A0,A1,A2,A0,A1 -- matching the
// producer's rotation for exactly 10 firings. Two tasks, NO repeat_count
// anywhere (q - 1 = 0), and the remainder task re-visits the cycle's first
// two A buffers -- the buffer identity is the whole point, so it is pinned
// by capture below.

// The dma_start block-arg patterns exclude ',' after the entry successor so a
// trailing ", repeat_count = N" cannot be swallowed by a greedy match, and
// --implicit-check-not=repeat_count rejects it anywhere between matches.
// CHECK: aie.device
// CHECK-DAG:   %[[TILE:.*]] = aie.tile(2, 3)
// CHECK:       aie.mem(%[[TILE]])
// CHECK:         aie.dma_start(S2MM, 0, ^[[T1BB1:[^,]+]], ^{{[^,)]+}})
// Task 1: one full cycle, six BDs, terminated (no cycling back). The B
// stream is ONE buffer, so every B BD must carry the same captured value.
// CHECK:       ^[[T1BB1]]:
// CHECK:         aie.dma_bd(%[[A0:[^ ]+]] : memref<32x32xbf16, 2>
// CHECK:         aie.dma_bd(%[[B:[^ ]+]] : memref<16x32xbf16, 2>
// CHECK:         aie.dma_bd(%[[A1:[^ ]+]] : memref<32x32xbf16, 2>
// CHECK:         aie.dma_bd(%[[B]] : memref<16x32xbf16, 2>
// CHECK:         aie.dma_bd(%[[A2:[^ ]+]] : memref<32x32xbf16, 2>
// CHECK:         aie.dma_bd(%[[B]] : memref<16x32xbf16, 2>
// CHECK-NOT:     aie.dma_bd
// Task 2: the remainder -- the first four BDs of the SAME cycle, so the A
// buffers must be exactly A0 then A1 again (not A2, and not fresh buffers).
// CHECK:         aie.dma_start(S2MM, 0, ^[[T2BB1:[^,]+]], ^{{[^,)]+}})
// CHECK:       ^[[T2BB1]]:
// CHECK:         aie.dma_bd(%[[A0]] : memref<32x32xbf16, 2>
// CHECK:         aie.dma_bd(%[[B]] : memref<16x32xbf16, 2>
// CHECK:         aie.dma_bd(%[[A1]] : memref<32x32xbf16, 2>
// CHECK:         aie.dma_bd(%[[B]] : memref<16x32xbf16, 2>
// CHECK-NOT:     aie.dma_bd

air.channel @channel_0 [1, 1]
func.func @nonclean_rotation() {
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
        %t2, %bufa2 = air.execute -> (memref<32x32xbf16, 2>) {
          %m = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %m : memref<32x32xbf16, 2>
        }
        %t3, %bufb = air.execute -> (memref<16x32xbf16, 2>) {
          %m = memref.alloc() : memref<16x32xbf16, 2>
          air.execute_terminator %m : memref<16x32xbf16, 2>
        }
        // steady: (A0, B, A1, B) inside one shared loop (trip 2)
        %s = scf.for %i = %c0 to %c2 step %c1_h iter_args(%dep = %t0) -> (!air.async.token) {
          %g0 = air.channel.get async [%dep, %t0] @channel_0[] (%bufa0[] [] []) : (memref<32x32xbf16, 2>)
          %g1 = air.channel.get async [%g0, %t3] @channel_0[] (%bufb[] [] []) : (memref<16x32xbf16, 2>)
          %g2 = air.channel.get async [%g1, %t1] @channel_0[] (%bufa1[] [] []) : (memref<32x32xbf16, 2>)
          %g3 = air.channel.get async [%g2] @channel_0[] (%bufb[] [] []) : (memref<16x32xbf16, 2>)
          scf.yield %g3 : !air.async.token
        }
        // epilogue peel: (A2, B) once, outside the loop
        %p0 = air.channel.get async [%s, %t2] @channel_0[] (%bufa2[] [] []) : (memref<32x32xbf16, 2>)
        %p1 = air.channel.get async [%p0] @channel_0[] (%bufb[] [] []) : (memref<16x32xbf16, 2>)
        %d0 = air.execute [%p1] { memref.dealloc %bufa0 : memref<32x32xbf16, 2> }
        %d1 = air.execute [%p1] { memref.dealloc %bufa1 : memref<32x32xbf16, 2> }
        %d2 = air.execute [%p1] { memref.dealloc %bufa2 : memref<32x32xbf16, 2> }
        %d3 = air.execute [%p1] { memref.dealloc %bufb : memref<16x32xbf16, 2> }
      }
    }
  }
  return
}
