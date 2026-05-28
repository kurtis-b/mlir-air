//===- int8_gemm_wmma.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// REQUIRES: gpu
// RUN: air-opt %s -air-to-rocdl -air-gpu-outlining | FileCheck %s --check-prefixes=CHECK
// RUN: not air-opt %s -air-to-rocdl="int8-gemm-variant=not_a_variant" 2>&1 | FileCheck %s --check-prefix=BAD
// RUN: not air-opt %s -air-to-rocdl="int8-gemm-group-size=6" 2>&1 | FileCheck %s --check-prefix=BADGROUP
// RUN: not air-opt %s -air-to-rocdl="int8-gemm-group-size=4" -air-gpu-outlining="int8-gemm-group-size=4" 2>&1 | FileCheck %s --check-prefix=BADGROUP4

// CHECK: gpu.launch_func @{{.*}}::@{{.*}} blocks in (%c8{{(_[0-9]+)?}}, %c8{{(_[0-9]+)?}}, %c1) threads in (%c32, %c4, %c1)
// CHECK: gpu.module @
// CHECK: gpu.func @{{.*}} kernel
// CHECK-SAME: air.gpu.int8_gemm_group_m = 8 : i32
// CHECK-SAME: air.gpu.int8_gemm_variant = "lds_128x128_rocmlir_k32_pipe3"
// CHECK: memref<128x32xi8, 3>
// CHECK: gpu.barrier
// CHECK: rocdl.wmma.i32.16x16x16.iu8
// CHECK-NOT: rocdl.wmma.i32.16x16x64
// CHECK-NOT: swmmac
// BAD: unsupported variant 'not_a_variant'
// BADGROUP: unsupported group size '6'
// BADGROUP4: unsupported group size '4'

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
