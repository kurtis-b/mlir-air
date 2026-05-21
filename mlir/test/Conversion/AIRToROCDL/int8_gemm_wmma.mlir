//===- int8_gemm_wmma.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// REQUIRES: gpu
// RUN: air-opt %s -air-to-rocdl -air-gpu-outlining | FileCheck %s

// CHECK: gpu.launch_func @{{.*}}::@{{.*}} blocks in (%c16, %c8, %c1) threads in (%c256, %c1, %c1)
// CHECK: gpu.module @
// CHECK: gpu.func @{{.*}} kernel
// CHECK-SAME: air.gpu.int8_gemm_variant = "lds_128x64_wmma4"
// CHECK: rocdl.wmma.i32.16x16x16.iu8
// CHECK-NOT: rocdl.wmma.i32.16x16x64
// CHECK-NOT: swmmac

module {
  func.func @forward(%arg0: memref<1024x1024xi8>, %arg1: memref<1024x1024xi8>, %arg2: memref<1024x1024xi32>) {
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    air.launch (%bx, %by) in (%gx=%c32, %gy=%c16) args(%a=%arg0, %b=%arg1, %c=%arg2) : memref<1024x1024xi8>, memref<1024x1024xi8>, memref<1024x1024xi32> attributes {air.gpu.int8_gemm_wmma} {
      air.launch_terminator
    }
    return
  }
}
