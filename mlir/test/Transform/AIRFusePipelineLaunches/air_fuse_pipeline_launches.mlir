//===- air_fuse_pipeline_launches.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-fuse-pipeline-launches --split-input-file | FileCheck %s

// A two-stage declared pipeline collapses into ONE launch holding ONE segment
// with both herds, in declared stage order. The operands of the two launches
// are unioned; the pipeline markers are erased on consume.

// CHECK-LABEL: func.func @two_stage
// CHECK: air.launch () in () args(%[[A:.*]]=%{{.*}}, %[[B:.*]]=%{{.*}}) : memref<64xi32>, memref<64xi32> {
// CHECK-NOT: air.pipeline_group
// CHECK-NOT: air.pipeline_stage
// CHECK: air.segment @seg args(%[[SA:.*]]=%[[A]], %[[SB:.*]]=%[[B]]) : memref<64xi32>, memref<64xi32> {
// CHECK: air.herd @stage0
// CHECK: air.herd @stage1
// CHECK-NOT: air.launch
func.func @two_stage(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
  air.launch () in () args(%a=%arg0) : memref<64xi32> attributes {air.pipeline_group = "p", air.pipeline_stage = 0 : i64} {
    air.segment @seg args(%sa=%a) : memref<64xi32> {
      %c1 = arith.constant 1 : index
      air.herd @stage0 tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%ha=%sa) : memref<64xi32> {
      }
    }
  }
  air.launch () in () args(%b=%arg1) : memref<64xi32> attributes {air.pipeline_group = "p", air.pipeline_stage = 1 : i64} {
    air.segment @seg args(%sb=%b) : memref<64xi32> {
      %c1 = arith.constant 1 : index
      air.herd @stage1 tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%hb=%sb) : memref<64xi32> {
      }
    }
  }
  return
}

// -----

// A value fed to TWO stages becomes ONE operand of the fused launch. Keeping
// duplicates would grow the fused launch's shim-facing ABI and would stop the
// fused module from reproducing a hand-written one that passes each buffer
// once.

// CHECK-LABEL: func.func @shared_operand
// CHECK: air.launch () in () args(%[[A:.*]]=%{{.*}}) : memref<64xi32> {
// CHECK: air.segment @seg args(%{{.*}}=%[[A]]) : memref<64xi32> {
// CHECK: air.herd @stage0
// CHECK: air.herd @stage1
func.func @shared_operand(%arg0: memref<64xi32>) {
  air.launch () in () args(%a=%arg0) : memref<64xi32> attributes {air.pipeline_group = "p", air.pipeline_stage = 0 : i64} {
    air.segment @seg args(%sa=%a) : memref<64xi32> {
      %c1 = arith.constant 1 : index
      air.herd @stage0 tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%ha=%sa) : memref<64xi32> {
      }
    }
  }
  air.launch () in () args(%b=%arg0) : memref<64xi32> attributes {air.pipeline_group = "p", air.pipeline_stage = 1 : i64} {
    air.segment @seg args(%sb=%b) : memref<64xi32> {
      %c1 = arith.constant 1 : index
      air.herd @stage1 tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%hb=%sb) : memref<64xi32> {
      }
    }
  }
  return
}

// -----

// Stage order is the DECLARED order, not IR order: stage 1 appears first in
// the input and second in the output. Without this the fused pipeline's
// producer/consumer direction would be an artifact of how the builder happened
// to emit its launches.

// CHECK-LABEL: func.func @out_of_order
// CHECK: air.segment
// CHECK: air.herd @first
// CHECK: air.herd @second
func.func @out_of_order(%arg0: memref<64xi32>) {
  air.launch () in () args(%b=%arg0) : memref<64xi32> attributes {air.pipeline_group = "p", air.pipeline_stage = 1 : i64} {
    air.segment @seg args(%sb=%b) : memref<64xi32> {
      %c1 = arith.constant 1 : index
      air.herd @second tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%hb=%sb) : memref<64xi32> {
      }
    }
  }
  air.launch () in () args(%a=%arg0) : memref<64xi32> attributes {air.pipeline_group = "p", air.pipeline_stage = 0 : i64} {
    air.segment @seg args(%sa=%a) : memref<64xi32> {
      %c1 = arith.constant 1 : index
      air.herd @first tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%ha=%sa) : memref<64xi32> {
      }
    }
  }
  return
}

// -----

// Two DIFFERENT groups in one function stay separate: grouping is by declared
// name, so a module holding two pipelines does not collapse into one segment.

// CHECK-LABEL: func.func @two_groups
// CHECK: air.launch
// CHECK: air.herd @p0
// CHECK: air.herd @p1
// CHECK: air.launch
// CHECK: air.herd @q0
// CHECK: air.herd @q1
func.func @two_groups(%arg0: memref<64xi32>) {
  air.launch () in () args(%a=%arg0) : memref<64xi32> attributes {air.pipeline_group = "p", air.pipeline_stage = 0 : i64} {
    air.segment args(%sa=%a) : memref<64xi32> {
      %c1 = arith.constant 1 : index
      air.herd @p0 tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%ha=%sa) : memref<64xi32> {
      }
    }
  }
  air.launch () in () args(%c=%arg0) : memref<64xi32> attributes {air.pipeline_group = "q", air.pipeline_stage = 0 : i64} {
    air.segment args(%sc=%c) : memref<64xi32> {
      %c1 = arith.constant 1 : index
      air.herd @q0 tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%hc=%sc) : memref<64xi32> {
      }
    }
  }
  air.launch () in () args(%b=%arg0) : memref<64xi32> attributes {air.pipeline_group = "p", air.pipeline_stage = 1 : i64} {
    air.segment args(%sb=%b) : memref<64xi32> {
      %c1 = arith.constant 1 : index
      air.herd @p1 tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%hb=%sb) : memref<64xi32> {
      }
    }
  }
  air.launch () in () args(%d=%arg0) : memref<64xi32> attributes {air.pipeline_group = "q", air.pipeline_stage = 1 : i64} {
    air.segment args(%sd=%d) : memref<64xi32> {
      %c1 = arith.constant 1 : index
      air.herd @q1 tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%hd=%sd) : memref<64xi32> {
      }
    }
  }
  return
}

// -----

// An UNATTRIBUTED launch is untouched. Every shipped design is unattributed,
// so this is the clause that says the pass is a no-op for them.

// CHECK-LABEL: func.func @untouched
// CHECK: air.launch
// CHECK: air.launch
func.func @untouched(%arg0: memref<64xi32>) {
  air.launch () in () args(%a=%arg0) : memref<64xi32> {
    air.segment args(%sa=%a) : memref<64xi32> {
      %c1 = arith.constant 1 : index
      air.herd @h0 tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%ha=%sa) : memref<64xi32> {
      }
    }
  }
  air.launch () in () args(%b=%arg0) : memref<64xi32> {
    air.segment args(%sb=%b) : memref<64xi32> {
      %c1 = arith.constant 1 : index
      air.herd @h1 tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%hb=%sb) : memref<64xi32> {
      }
    }
  }
  return
}

// -----

// A declared "accum_in_place" stage whose segment DOES hold the shape --
// a loop allocating a buffer that is both the destination and the source of an
// air.dma_memcpy_nd -- is accepted. This is the positive control for the
// staging check below: without it, that check could be refusing everything.

// CHECK-LABEL: func.func @accum_ok
// CHECK: air.launch
// CHECK-NOT: air.staging
// CHECK: air.herd @acc
func.func @accum_ok(%arg0: memref<64x64xi32>) {
  air.launch () in () args(%a=%arg0) : memref<64x64xi32> attributes {air.pipeline_group = "p", air.pipeline_stage = 0 : i64, air.staging = "accum_in_place"} {
    air.segment args(%sa=%a) : memref<64x64xi32> {
      %c1 = arith.constant 1 : index
      air.herd @acc tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%ha=%sa) : memref<64x64xi32> {
        %c0 = arith.constant 0 : index
        %hc1 = arith.constant 1 : index
        %c8 = arith.constant 8 : index
        %c64 = arith.constant 64 : index
        scf.for %k = %c0 to %c64 step %c8 {
          %buf = memref.alloc() : memref<8x8xi32, 2 : i32>
          air.dma_memcpy_nd (%buf[][][], %ha[%c0, %c0][%c8, %c8][%c64, %hc1]) : (memref<8x8xi32, 2 : i32>, memref<64x64xi32>)
          air.dma_memcpy_nd (%ha[%c0, %c0][%c8, %c8][%c64, %hc1], %buf[][][]) : (memref<64x64xi32>, memref<8x8xi32, 2 : i32>)
          memref.dealloc %buf : memref<8x8xi32, 2 : i32>
        }
      }
    }
  }
  return
}
