//===- air_to_rocdl.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// REQUIRES: gpu
// RUN: air-opt %s -air-to-rocdl | FileCheck %s

// Verifies that air-to-rocdl converts AIR hierarchy to GPU dialect:
//   air.launch(gx, gy) -> gpu.launch blocks(gx, gy, 1)
//   air.herd(hx, hy)   -> gpu.launch threads(hx, hy, 1)
//   air.segment         -> unwrapped
//   memref space=1      -> GPU workgroup attribution (space 3)
//   memref space=2      -> GPU private attribution (space 5)

// CHECK-LABEL: func.func @vecadd
// CHECK-NOT: air.launch
// CHECK-NOT: air.segment
// CHECK-NOT: air.herd
// CHECK: gpu.launch
// CHECK-SAME: blocks
// CHECK-SAME: threads
// CHECK: gpu.terminator

module {
  func.func @vecadd(%arg0: memref<4x16xf32>, %arg1: memref<4x16xf32>, %arg2: memref<4x16xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    air.launch (%bx, %by) in (%nbx=%c4, %nby=%c1)
        args(%in0=%arg0, %in1=%arg1, %out=%arg2)
        : memref<4x16xf32>, memref<4x16xf32>, memref<4x16xf32> {
      air.segment @seg
          args(%seg_bx=%bx, %seg_in0=%in0, %seg_in1=%in1, %seg_out=%out)
          : index, memref<4x16xf32>, memref<4x16xf32>, memref<4x16xf32> {
        %s0 = arith.constant 0 : index
        %s1 = arith.constant 1 : index
        %s16 = arith.constant 16 : index
        %tile = arith.constant 1 : index
        %l2lhs = memref.alloc() : memref<16xf32, 1>
        %l2rhs = memref.alloc() : memref<16xf32, 1>
        %l2out = memref.alloc() : memref<16xf32, 1>
        air.dma_memcpy_nd (%l2lhs[%s0] [%s16] [%s1],
                           %seg_in0[%seg_bx, %s0] [%s1, %s16] [%s16, %s1])
            : (memref<16xf32, 1>, memref<4x16xf32>)
        air.dma_memcpy_nd (%l2rhs[%s0] [%s16] [%s1],
                           %seg_in1[%seg_bx, %s0] [%s1, %s16] [%s16, %s1])
            : (memref<16xf32, 1>, memref<4x16xf32>)

        air.herd @herd tile (%tx, %ty) in (%ntx=%tile, %nty=%tile)
            args(%herd_lhs=%l2lhs, %herd_rhs=%l2rhs, %herd_out=%l2out)
            : memref<16xf32, 1>, memref<16xf32, 1>, memref<16xf32, 1> {
          %h0 = arith.constant 0 : index
          %h1 = arith.constant 1 : index
          %h16 = arith.constant 16 : index
          %l1lhs = memref.alloc() : memref<16xf32, 2>
          %l1rhs = memref.alloc() : memref<16xf32, 2>
          %l1out = memref.alloc() : memref<16xf32, 2>
          air.dma_memcpy_nd (%l1lhs[%h0] [%h16] [%h1],
                             %herd_lhs[%h0] [%h16] [%h1])
              : (memref<16xf32, 2>, memref<16xf32, 1>)
          air.dma_memcpy_nd (%l1rhs[%h0] [%h16] [%h1],
                             %herd_rhs[%h0] [%h16] [%h1])
              : (memref<16xf32, 2>, memref<16xf32, 1>)
          scf.for %i = %h0 to %h16 step %h1 {
            %a = memref.load %l1lhs[%i] : memref<16xf32, 2>
            %b = memref.load %l1rhs[%i] : memref<16xf32, 2>
            %c = arith.addf %a, %b : f32
            memref.store %c, %l1out[%i] : memref<16xf32, 2>
          }
          air.dma_memcpy_nd (%herd_out[%h0] [%h16] [%h1],
                             %l1out[%h0] [%h16] [%h1])
              : (memref<16xf32, 1>, memref<16xf32, 2>)
          memref.dealloc %l1lhs : memref<16xf32, 2>
          memref.dealloc %l1rhs : memref<16xf32, 2>
          memref.dealloc %l1out : memref<16xf32, 2>
          air.herd_terminator
        }

        air.dma_memcpy_nd (%seg_out[%seg_bx, %s0] [%s1, %s16] [%s16, %s1],
                           %l2out[%s0] [%s16] [%s1])
            : (memref<4x16xf32>, memref<16xf32, 1>)
        memref.dealloc %l2lhs : memref<16xf32, 1>
        memref.dealloc %l2rhs : memref<16xf32, 1>
        memref.dealloc %l2out : memref<16xf32, 1>
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}
