//===- air_to_rocdl_launch_ids.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// REQUIRES: gpu
// RUN: air-opt %s -air-to-rocdl | FileCheck %s

// Verifies that air-to-rocdl remaps launch ids and kernel operands before
// erasing the air.launch region. This covers AIR kernels that forward the
// launch induction var through air.segment into DMA indexing, which previously
// crashed with "Cannot destroy a value that still has uses!".

// CHECK-LABEL: func.func @launch_id_copy
// CHECK-NOT: air.launch
// CHECK-NOT: air.segment
// CHECK-NOT: air.herd
// CHECK: gpu.launch
// CHECK: memref.load
// CHECK: memref.store
// CHECK: gpu.terminator

module {
  func.func @launch_id_copy(%src: memref<4x8xf32>, %dst: memref<4x8xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    air.launch (%bx, %by) in (%nbx=%c4, %nby=%c1)
        args(%launch_src=%src, %launch_dst=%dst)
        : memref<4x8xf32>, memref<4x8xf32> {
      air.segment @seg
          args(%seg_bx=%bx, %seg_src=%launch_src, %seg_dst=%launch_dst)
          : index, memref<4x8xf32>, memref<4x8xf32> {
        %s0 = arith.constant 0 : index
        %s1 = arith.constant 1 : index
        %s8 = arith.constant 8 : index
        %seg_tile = arith.constant 1 : index
        %l2 = memref.alloc() : memref<8xf32, 1>
        air.dma_memcpy_nd (%l2[%s0] [%s8] [%s1],
                           %seg_src[%seg_bx, %s0] [%s1, %s8] [%s8, %s1])
            : (memref<8xf32, 1>, memref<4x8xf32>)

        air.herd @copy_herd tile (%tx, %ty) in (%sx=%seg_tile, %sy=%seg_tile)
            args(%herd_buf=%l2)
            : memref<8xf32, 1> {
          %h0 = arith.constant 0 : index
          %h1 = arith.constant 1 : index
          %h8 = arith.constant 8 : index
          %l1 = memref.alloc() : memref<8xf32, 2>
          air.dma_memcpy_nd (%l1[%h0] [%h8] [%h1],
                             %herd_buf[%h0] [%h8] [%h1])
              : (memref<8xf32, 2>, memref<8xf32, 1>)
          air.dma_memcpy_nd (%herd_buf[%h0] [%h8] [%h1],
                             %l1[%h0] [%h8] [%h1])
              : (memref<8xf32, 1>, memref<8xf32, 2>)
          air.herd_terminator
        }

        air.dma_memcpy_nd (%seg_dst[%seg_bx, %s0] [%s1, %s8] [%s8, %s1],
                           %l2[%s0] [%s8] [%s1])
            : (memref<4x8xf32>, memref<8xf32, 1>)
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}
