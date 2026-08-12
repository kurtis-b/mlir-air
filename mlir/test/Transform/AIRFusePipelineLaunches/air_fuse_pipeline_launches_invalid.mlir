//===- air_fuse_pipeline_launches_invalid.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-fuse-pipeline-launches -split-input-file -verify-diagnostics

// A MALFORMED GROUP IS REFUSED, NEVER SILENTLY LEFT UNFUSED.
//
// This is the opposite of air-label-scf-for-to-ping-pong's skip-and-warn rule,
// and the discriminator is doc 23's: declining to ping-pong leaves a CORRECT
// single-buffered loop, so only the optimization is lost. Declining to fuse a
// pipeline leaves each stage in its own air.launch and therefore its own
// aie.device -- so the L1->L1 air.channel edges the stages declare span
// devices. The untransformed program is the broken one, exactly as with
// air-fuse-packet-put-loops.
//
// The positive controls live in air_fuse_pipeline_launches.mlir; without them
// every clause here could be passing because the pass refuses everything.

// A duplicated stage index: the fused order would be an artifact of IR order.
func.func @duplicate_stage(%arg0: memref<64xi32>) {
  air.launch () in () args(%a=%arg0) : memref<64xi32> attributes {air.pipeline_group = "p", air.pipeline_stage = 0 : i64} {
    air.segment args(%sa=%a) : memref<64xi32> {
      %c1 = arith.constant 1 : index
      air.herd @h0 tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%ha=%sa) : memref<64xi32> {
      }
    }
  }
  // expected-error@+1 {{declares air.pipeline_stage = 0 more than once}}
  air.launch () in () args(%b=%arg0) : memref<64xi32> attributes {air.pipeline_group = "p", air.pipeline_stage = 0 : i64} {
    air.segment args(%sb=%b) : memref<64xi32> {
      %c1 = arith.constant 1 : index
      air.herd @h1 tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%hb=%sb) : memref<64xi32> {
      }
    }
  }
  return
}

// -----

// A gap in the stage indices: a stage of the pipeline is missing, which most
// likely means a builder emitted one launch fewer than it declared.
func.func @gap_in_stages(%arg0: memref<64xi32>) {
  air.launch () in () args(%a=%arg0) : memref<64xi32> attributes {air.pipeline_group = "p", air.pipeline_stage = 0 : i64} {
    air.segment args(%sa=%a) : memref<64xi32> {
      %c1 = arith.constant 1 : index
      air.herd @h0 tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%ha=%sa) : memref<64xi32> {
      }
    }
  }
  // expected-error@+1 {{has no air.pipeline_stage = 1}}
  air.launch () in () args(%b=%arg0) : memref<64xi32> attributes {air.pipeline_group = "p", air.pipeline_stage = 2 : i64} {
    air.segment args(%sb=%b) : memref<64xi32> {
      %c1 = arith.constant 1 : index
      air.herd @h1 tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%hb=%sb) : memref<64xi32> {
      }
    }
  }
  return
}

// -----

// A group member with no air.pipeline_stage. The pass will not fall back on IR
// order, because the order decides which stage's output feeds which.
func.func @missing_stage(%arg0: memref<64xi32>) {
  // expected-error@+1 {{but no air.pipeline_stage}}
  air.launch () in () args(%a=%arg0) : memref<64xi32> attributes {air.pipeline_group = "p"} {
    air.segment args(%sa=%a) : memref<64xi32> {
      %c1 = arith.constant 1 : index
      air.herd @h0 tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%ha=%sa) : memref<64xi32> {
      }
    }
  }
  return
}

// -----

// A stage with a launch grid. The fused launch has ONE iteration space, so
// merging stages that each declare their own is not defined.
func.func @stage_has_grid(%arg0: memref<64xi32>) {
  %c2 = arith.constant 2 : index
  // expected-error@+1 {{declares a launch grid of 1 dimension(s)}}
  air.launch (%i) in (%sz=%c2) args(%a=%arg0) : memref<64xi32> attributes {air.pipeline_group = "p", air.pipeline_stage = 0 : i64} {
    air.segment args(%sa=%a) : memref<64xi32> {
      %c1 = arith.constant 1 : index
      air.herd @h0 tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%ha=%sa) : memref<64xi32> {
      }
    }
  }
  air.launch () in () args(%b=%arg0) : memref<64xi32> attributes {air.pipeline_group = "p", air.pipeline_stage = 1 : i64} {
    air.segment args(%sb=%b) : memref<64xi32> {
      %c1 = arith.constant 1 : index
      air.herd @h1 tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%hb=%sb) : memref<64xi32> {
      }
    }
  }
  return
}

// -----

// A stage whose launch body holds two segments: which one joins the fused
// segment, and in what order relative to the other stages, is not declared.
func.func @two_segments(%arg0: memref<64xi32>) {
  // expected-error@+1 {{holds more than one air.segment}}
  air.launch () in () args(%a=%arg0) : memref<64xi32> attributes {air.pipeline_group = "p", air.pipeline_stage = 0 : i64} {
    air.segment args(%sa=%a) : memref<64xi32> {
      %c1 = arith.constant 1 : index
      air.herd @h0 tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%ha=%sa) : memref<64xi32> {
      }
    }
    air.segment args(%sa2=%a) : memref<64xi32> {
      %c1 = arith.constant 1 : index
      air.herd @h0b tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%ha2=%sa2) : memref<64xi32> {
      }
    }
  }
  return
}

// -----

// A stage whose launch body holds L3-scope work beside its segment. This pass
// has no declared ordering for that work across stages, and inventing one is
// exactly the derivation H8 was scoped away from.
func.func @extra_launch_scope_op(%arg0: memref<64xi32>) {
  // expected-error@+1 {{beside its air.segment}}
  air.launch () in () args(%a=%arg0) : memref<64xi32> attributes {air.pipeline_group = "p", air.pipeline_stage = 0 : i64} {
    %l3 = memref.alloc() : memref<64xi32, 1 : i32>
    air.segment args(%sa=%a) : memref<64xi32> {
      %c1 = arith.constant 1 : index
      air.herd @h0 tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%ha=%sa) : memref<64xi32> {
      }
    }
    memref.dealloc %l3 : memref<64xi32, 1 : i32>
  }
  return
}

// -----

// THE STAGING CLAIM IS CHECKED, NOT CARRIED.
//
// "accum_in_place" declared on a stage whose loop does NOT allocate a buffer
// that is both the destination and the source of an air.dma_memcpy_nd. This is
// the case doc 22 measured as shipping silently: the accumulator allocated at
// herd scope instead of inside the K loop compiles, places, and returns the
// right numbers at one DDR round trip per K step -- 4 -> 4 rather than 4 -> 2 --
// and no numeric gate can see it.
func.func @accum_alloc_outside_loop(%arg0: memref<64x64xi32>) {
  // expected-error@+1 {{declares air.staging = "accum_in_place" but no loop in its segment}}
  air.launch () in () args(%a=%arg0) : memref<64x64xi32> attributes {air.pipeline_group = "p", air.pipeline_stage = 0 : i64, air.staging = "accum_in_place"} {
    air.segment args(%sa=%a) : memref<64x64xi32> {
      %c1 = arith.constant 1 : index
      air.herd @acc tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%ha=%sa) : memref<64x64xi32> {
        %c0 = arith.constant 0 : index
        %hc1 = arith.constant 1 : index
        %c8 = arith.constant 8 : index
        %c64 = arith.constant 64 : index
        // The alloc is HOISTED out of the loop -- the natural way to write it,
        // and the one that does not ring.
        %buf = memref.alloc() : memref<8x8xi32, 2 : i32>
        scf.for %k = %c0 to %c64 step %c8 {
          air.dma_memcpy_nd (%buf[][][], %ha[%c0, %c0][%c8, %c8][%c64, %hc1]) : (memref<8x8xi32, 2 : i32>, memref<64x64xi32>)
          air.dma_memcpy_nd (%ha[%c0, %c0][%c8, %c8][%c64, %hc1], %buf[][][]) : (memref<64x64xi32>, memref<8x8xi32, 2 : i32>)
        }
        memref.dealloc %buf : memref<8x8xi32, 2 : i32>
      }
    }
  }
  return
}

// -----

// "accum_in_place" where the in-loop buffer is only ever a DMA destination:
// the TWO-BUFFER kernel shape (pAcc in, C out), which doc 22 measured as never
// matching areSymmetricDmaOps at any alloc site.
func.func @accum_not_in_place(%arg0: memref<64x64xi32>) {
  // expected-error@+1 {{declares air.staging = "accum_in_place" but no loop in its segment}}
  air.launch () in () args(%a=%arg0) : memref<64x64xi32> attributes {air.pipeline_group = "p", air.pipeline_stage = 0 : i64, air.staging = "accum_in_place"} {
    air.segment args(%sa=%a) : memref<64x64xi32> {
      %c1 = arith.constant 1 : index
      air.herd @acc tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%ha=%sa) : memref<64x64xi32> {
        %c0 = arith.constant 0 : index
        %hc1 = arith.constant 1 : index
        %c8 = arith.constant 8 : index
        %c64 = arith.constant 64 : index
        scf.for %k = %c0 to %c64 step %c8 {
          %pacc = memref.alloc() : memref<8x8xi32, 2 : i32>
          %c = memref.alloc() : memref<8x8xi32, 2 : i32>
          air.dma_memcpy_nd (%pacc[][][], %ha[%c0, %c0][%c8, %c8][%c64, %hc1]) : (memref<8x8xi32, 2 : i32>, memref<64x64xi32>)
          air.dma_memcpy_nd (%ha[%c0, %c0][%c8, %c8][%c64, %hc1], %c[][][]) : (memref<64x64xi32>, memref<8x8xi32, 2 : i32>)
          memref.dealloc %pacc : memref<8x8xi32, 2 : i32>
          memref.dealloc %c : memref<8x8xi32, 2 : i32>
        }
      }
    }
  }
  return
}

// -----

// "memtile" declared on a stage whose segment allocates no L2 buffer.
func.func @memtile_without_l2(%arg0: memref<64xi32>) {
  // expected-error@+1 {{declares air.staging = "memtile" but its segment allocates no L2 buffer}}
  air.launch () in () args(%a=%arg0) : memref<64xi32> attributes {air.pipeline_group = "p", air.pipeline_stage = 0 : i64, air.staging = "memtile"} {
    air.segment args(%sa=%a) : memref<64xi32> {
      %c1 = arith.constant 1 : index
      air.herd @h0 tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%ha=%sa) : memref<64xi32> {
      }
    }
  }
  return
}
