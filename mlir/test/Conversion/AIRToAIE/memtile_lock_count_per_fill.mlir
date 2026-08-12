//===- memtile_lock_count_per_fill.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" --split-input-file | FileCheck %s

// A legacy counted memtile lock must be sized by how many reads happen PER
// FILL, not by how many reader ops appear in the IR.
//
// When a reader nest is unrolled over an outer dimension that the single
// producer op is not unrolled over, the same consumer endpoint reads the same
// sub-region once per fill and so appears N times statically. air-to-aie
// collapses those into ONE BD in that channel's chain -- both cases below emit
// exactly one MM2S BD per endpoint -- so counting them separately makes the
// producer acquire and release more tokens per fill than the consumers can
// ever return. Such a design still compiles: with acquire R against C consumer
// releases per fill it simply stops refilling after init/(R-C) rounds, which
// on hardware is a hang (ERT_CMD_STATE_TIMEOUT), not a diagnostic.

// -----

// CASE 1 -- one writer op; two consumer endpoints, each read TWICE statically.
// Reads per fill is 2 (one per endpoint), not 4.

// CHECK-LABEL: aie.device(xcve2802) @seg_single
// CHECK: %[[MT:.*]] = aie.logical_tile<MemTile>(?, ?)
// CHECK: %[[WLOCK:.*]] = aie.lock(%[[MT]], {{[0-9]+}}) {init = 2 : i32}
// CHECK: %[[RLOCK:.*]] = aie.lock(%[[MT]], {{[0-9]+}}) {init = 0 : i32}
// CHECK: aie.memtile_dma(%[[MT]])
// CHECK: %[[C2:.*]] = arith.constant 2 : i32
// CHECK: %[[C1:.*]] = arith.constant 1 : i32
// The four reader ops emit exactly TWO BDs, one per endpoint, each taking one
// token -- this is why four is the wrong number for the producer to wait on.
// CHECK: aie.use_lock(%[[RLOCK]], AcquireGreaterEqual, %[[C1]])
// CHECK: aie.dma_bd({{.*}} offset = 0 len = 8)
// CHECK: aie.use_lock(%[[WLOCK]], Release, %[[C1]])
// CHECK: aie.use_lock(%[[RLOCK]], AcquireGreaterEqual, %[[C1]])
// CHECK: aie.dma_bd({{.*}} offset = 8 len = 8)
// CHECK: aie.use_lock(%[[WLOCK]], Release, %[[C1]])
// The fill takes and returns 2 -- matching the two reads it enables. Pre-fix
// this pair was 4, and the buffer could refill only init/(4-2) = 2 times.
// CHECK: aie.use_lock(%[[WLOCK]], AcquireGreaterEqual, %[[C2]])
// CHECK: aie.dma_bd({{.*}} offset = 0 len = 32)
// CHECK: aie.use_lock(%[[RLOCK]], Release, %[[C2]])

air.channel @w0 [1, 1]
air.channel @r0 [1, 1]
air.channel @r1 [1, 1]
func.func @per_fill_single_writer() {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) {
    air.segment @seg_single {
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %t, %l2 = air.execute -> (memref<4x8xbf16, 1>) {
        %alloc = memref.alloc() {air.no_split} : memref<4x8xbf16, 1>
        air.execute_terminator %alloc : memref<4x8xbf16, 1>
      }
      // ONE full-buffer write.
      air.channel.get @w0[] (%l2[] [] []) : (memref<4x8xbf16, 1>)
      // Two endpoints, each reading its own row -- each written TWICE, exactly
      // as a reader nest unrolled over a dimension the writer is not.
      air.channel.put @r0[] (%l2[%c0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<4x8xbf16, 1>)
      air.channel.put @r1[] (%l2[%c1_0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<4x8xbf16, 1>)
      air.channel.put @r0[] (%l2[%c0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<4x8xbf16, 1>)
      air.channel.put @r1[] (%l2[%c1_0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<4x8xbf16, 1>)
      %dd = air.execute {
        memref.dealloc %l2 : memref<4x8xbf16, 1>
      }
      air.herd @hw tile (%txw, %tyw) in (%sxw=%c1_0, %syw=%c1_0)
            attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<32xbf16, 2>) {
          %aa = memref.alloc() : memref<32xbf16, 2>
          air.execute_terminator %aa : memref<32xbf16, 2>
        }
        air.channel.put @w0[] (%l1[] [] []) : (memref<32xbf16, 2>)
        %dw = air.execute {memref.dealloc %l1 : memref<32xbf16, 2>}
      }
      air.herd @h0 tile (%tx0, %ty0) in (%sx0=%c1_0, %sy0=%c1_0)
            attributes {x_loc = 3 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.get @r0[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.get @r0[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %d0 = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      air.herd @h1 tile (%tx1, %ty1) in (%sx1=%c1_0, %sy1=%c1_0)
            attributes {x_loc = 4 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.get @r1[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.get @r1[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %d1 = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
    }
  }
  return
}

// -----

// CASE 2 -- the CONTROL, and the reason the collapse is scoped to a single
// writer op. Two writer endpoints share the buffer, so they are
// time-multiplexed: one fires per fill while both consumer endpoints read. The
// raw ratio 4/2 is already correct here, precisely because BOTH sides carry
// the same static replication; collapsing only the read side would give
// 2/2 = 1 and starve the second consumer. This case must be UNCHANGED, and it
// is what fails if the per-fill collapse is ever applied unconditionally.

// CHECK-LABEL: aie.device(xcve2802) @seg_multi
// CHECK: %[[MT2:.*]] = aie.logical_tile<MemTile>(?, ?)
// CHECK: %[[WLOCK2:.*]] = aie.lock(%[[MT2]], {{[0-9]+}}) {init = 2 : i32}
// CHECK: %[[RLOCK2:.*]] = aie.lock(%[[MT2]], {{[0-9]+}}) {init = 0 : i32}
// CHECK: aie.memtile_dma(%[[MT2]])
// CHECK: %[[C2B:.*]] = arith.constant 2 : i32
// CHECK: %[[C1B:.*]] = arith.constant 1 : i32
// Two readers at 1 each ...
// CHECK: aie.use_lock(%[[RLOCK2]], AcquireGreaterEqual, %[[C1B]])
// CHECK: aie.use_lock(%[[WLOCK2]], Release, %[[C1B]])
// CHECK: aie.use_lock(%[[RLOCK2]], AcquireGreaterEqual, %[[C1B]])
// CHECK: aie.use_lock(%[[WLOCK2]], Release, %[[C1B]])
// ... and BOTH writers still at 2, not 1.
// CHECK: aie.use_lock(%[[WLOCK2]], AcquireGreaterEqual, %[[C2B]])
// CHECK: aie.use_lock(%[[RLOCK2]], Release, %[[C2B]])
// CHECK: aie.use_lock(%[[WLOCK2]], AcquireGreaterEqual, %[[C2B]])
// CHECK: aie.use_lock(%[[RLOCK2]], Release, %[[C2B]])

air.channel @w0 [1, 1]
air.channel @w1 [1, 1]
air.channel @r0 [1, 1]
air.channel @r1 [1, 1]
func.func @per_fill_multi_writer_control() {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) {
    air.segment @seg_multi {
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %t, %l2 = air.execute -> (memref<4x8xbf16, 1>) {
        %alloc = memref.alloc() {air.no_split} : memref<4x8xbf16, 1>
        air.execute_terminator %alloc : memref<4x8xbf16, 1>
      }
      // TWO writer endpoints, each writing its own row.
      air.channel.get @w0[] (%l2[%c0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<4x8xbf16, 1>)
      air.channel.get @w1[] (%l2[%c1_0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<4x8xbf16, 1>)
      // Two reader endpoints, each written twice -- the same replication the
      // single-writer case above carries.
      air.channel.put @r0[] (%l2[%c0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<4x8xbf16, 1>)
      air.channel.put @r1[] (%l2[%c1_0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<4x8xbf16, 1>)
      air.channel.put @r0[] (%l2[%c0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<4x8xbf16, 1>)
      air.channel.put @r1[] (%l2[%c1_0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<4x8xbf16, 1>)
      %dd = air.execute {
        memref.dealloc %l2 : memref<4x8xbf16, 1>
      }
      air.herd @hw0 tile (%txw, %tyw) in (%sxw=%c1_0, %syw=%c1_0)
            attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.put @w0[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %dw = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      air.herd @hw1 tile (%txw1, %tyw1) in (%sxw1=%c1_0, %syw1=%c1_0)
            attributes {x_loc = 5 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.put @w1[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %dw1 = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      air.herd @h0 tile (%tx0, %ty0) in (%sx0=%c1_0, %sy0=%c1_0)
            attributes {x_loc = 3 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.get @r0[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.get @r0[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %d0 = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      air.herd @h1 tile (%tx1, %ty1) in (%sx1=%c1_0, %sy1=%c1_0)
            attributes {x_loc = 4 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.get @r1[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.get @r1[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %d1 = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
    }
  }
  return
}
