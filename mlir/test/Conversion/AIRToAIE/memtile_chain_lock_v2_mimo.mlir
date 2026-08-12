//===- memtile_chain_lock_v2_mimo.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// MIMO: a shared L2 memtile buffer with 2 writers AND 2 readers, all touching
// the WHOLE buffer -- one slot, time-multiplexed. This is wall 7's shape (doc
// 52): R1's down feed stages every GeLU column's H chunk through one L2
// buffer, so at herd_x > 1 the buffer takes one S2MM channel per column and
// one MM2S channel per consumer.
//
// It is NOT the fan-in shape. Fan-in is N writers filling DISJOINT
// sub-regions, read once when assembled (memtile_chain_lock_v2_fanin.mlir);
// its daisy chain is correct precisely because no reader runs between two
// writers. Here every writer fills the whole slot and every reader must
// consume each fill before the next lands.
//
// That combination is not expressible. An AIE2 BD descriptor carries exactly
// one acquire-lock field and one release-lock field, so a writer's single
// release must go EITHER to the next writer (ordering the writers, leaving the
// readers unsignalled) OR to the readers (binding the readers, leaving the
// writers racing on a counting semaphore, which is the legacy template and is
// wall 7). Doc 52 §8 closes the remaining freedom -- asymmetric acquire and
// release counts -- with a counting argument, and measures both arms with
// `agents/probes/probe_aie_buffer_writer_race.py --check-order`.
//
// So the contract pinned here is: v2 REFUSES this shape by name rather than
// falling through to the legacy counted-lock template. That silent
// fall-through is why `use_lock_race_condition_fix_v2` A/B'd byte-identical to
// baseline five times on hardware -- it was never reached, not inert.

// 1. v2 refuses, naming the buffer and the shape.
// RUN: not air-opt %s -air-to-aie="use-lock-race-condition-fix-v2=true row-offset=3 col-offset=2 device=xcve2802" 2>&1 | FileCheck %s --check-prefix=REFUSE

// REFUSE: 2 writers and 2 readers (MIMO) on a single slot
// REFUSE-SAME: one acquire and one release

// 2. The falsifier arm (mimo-chain-lock) emits the two-chain form: 1 cap lock
//    plus nW + nR - 1 = 3 signal locks, and -- the point of the arm -- it
//    stays SINGLE-SLOT, so it does not silently become the per-buffer fix.
//    It is measured UNSOUND (ORDERED writers, but OVERWRITE on the read side);
//    this only pins what it emits so the measurement is reproducible.
// RUN: air-opt %s -air-to-aie="use-lock-race-condition-fix-v2=true mimo-chain-lock=true row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s --check-prefix=CHAIN

// CHAIN: aie.device
// CHAIN-DAG: %[[MT:.*]] = aie.logical_tile<MemTile>(?, ?)
// 1 cap lock (init=1, single slot) + 3 signal locks (init=0).
// CHAIN-DAG: aie.lock(%[[MT]], {{[0-9]+}}) {init = 1 : i32}
// CHAIN-DAG: aie.lock(%[[MT]], {{[0-9]+}}) {init = 0 : i32}
// CHAIN-DAG: aie.lock(%[[MT]], {{[0-9]+}}) {init = 0 : i32}
// CHAIN-DAG: aie.lock(%[[MT]], {{[0-9]+}}) {init = 0 : i32}
// Exactly ONE L2 buffer instance: no ping-pong twin was spliced in, so the
// buffer count is unchanged from the legacy arm below.
// CHAIN: aie.buffer(%[[MT]]) {{.*}} : memref<8xbf16, 1
// CHAIN-NOT: aie.buffer(%[[MT]]) {{.*}} : memref<8xbf16, 1

// 3. The SHIPPED default (v2 off) is untouched: legacy counted-lock template,
//    one buffer, one lock pair. Nothing shipped moves because of this change.
// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s --check-prefix=LEGACY

// LEGACY: aie.device
// LEGACY-DAG: %[[MT2:.*]] = aie.logical_tile<MemTile>(?, ?)
// LEGACY: aie.buffer(%[[MT2]]) {{.*}} : memref<8xbf16, 1
// LEGACY-NOT: aie.buffer(%[[MT2]]) {{.*}} : memref<8xbf16, 1

air.channel @w0 [1, 1]
air.channel @w1 [1, 1]
air.channel @r0 [1, 1]
air.channel @r1 [1, 1]
func.func @memtile_mimo_single_slot() {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) {
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      // Shared L2 staging buffer carrying air.no_split: ONE slot, and every
      // participant below touches all of it.
      %t, %l2 = air.execute -> (memref<8xbf16, 1>) {
        %alloc = memref.alloc() {air.no_split} : memref<8xbf16, 1>
        air.execute_terminator %alloc : memref<8xbf16, 1>
      }
      // 2 segment-side gets: two producers, each filling the WHOLE buffer.
      air.channel.get @w0[] (%l2[] [] []) : (memref<8xbf16, 1>)
      air.channel.get @w1[] (%l2[] [] []) : (memref<8xbf16, 1>)
      // 2 segment-side puts: two consumers, each reading the WHOLE buffer.
      air.channel.put @r0[] (%l2[] [] []) : (memref<8xbf16, 1>)
      air.channel.put @r1[] (%l2[] [] []) : (memref<8xbf16, 1>)
      %d_ = air.execute {
        memref.dealloc %l2 : memref<8xbf16, 1>
      }
      air.herd @h0 tile (%tx0, %ty0) in (%sx0=%c1_0, %sy0=%c1_0)
            attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.put @w0[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %d0 = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      air.herd @h1 tile (%tx1, %ty1) in (%sx1=%c1_0, %sy1=%c1_0)
            attributes {x_loc = 3 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.put @w1[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %d1 = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      air.herd @hr0 tile (%txr0, %tyr0) in (%sxr0=%c1_0, %syr0=%c1_0)
            attributes {x_loc = 4 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.get @r0[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %dr0 = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      air.herd @hr1 tile (%txr1, %tyr1) in (%sxr1=%c1_0, %syr1=%c1_0)
            attributes {x_loc = 5 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.get @r1[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %dr1 = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
    }
  }
  return
}
