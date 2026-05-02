//===- air_gpu_host_staging.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// REQUIRES: gpu
// RUN: air-opt %s -air-to-rocdl -air-gpu-outlining -air-gpu-host-staging | FileCheck %s

// Verifies that AIR kernels outlined for the GPU get compiler-generated host
// staging instead of relying on textual IR rewriting in the runtime harness.

// CHECK-LABEL: func.func @copy_with_host_staging
// CHECK: %[[DEV0:.*]] = gpu.alloc
// CHECK: gpu.memcpy %[[DEV0]], %arg0
// CHECK: %[[DEV1:.*]] = gpu.alloc
// CHECK: gpu.memcpy %[[DEV1]], %arg1
// CHECK: gpu.launch_func
// CHECK: gpu.memcpy %arg0, %[[DEV0]]
// CHECK: gpu.memcpy %arg1, %[[DEV1]]
// CHECK: gpu.dealloc %[[DEV0]]
// CHECK: gpu.dealloc %[[DEV1]]

module {
  func.func @copy_with_host_staging(%src: memref<4x8xf32>,
                                    %dst: memref<4x8xf32>) {
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
