//===- int8_gemm_wmma.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// REQUIRES: gpu
// RUN: air-opt %s -air-to-rocdl -air-gpu-outlining | FileCheck %s --check-prefixes=CHECK,DEFAULT
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=lds_128x64_bpack" -air-gpu-outlining | FileCheck %s --check-prefixes=CHECK,BPACK
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=lds_128x64_bpack_swizzle" -air-gpu-outlining | FileCheck %s --check-prefixes=CHECK,SWIZZLE
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=lds_128x64_bpack_pipe2" -air-gpu-outlining | FileCheck %s --check-prefixes=CHECK,PIPE2
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=lds_128x64_bpack_pipe2_grouped" -air-gpu-outlining | FileCheck %s --check-prefixes=CHECK,GROUPED
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=lds_128x64_bpack_swizzle_grouped int8-gemm-group-size=8" -air-gpu-outlining="int8-gemm-group-size=8" | FileCheck %s --check-prefixes=CHECK,SWIZZLEGROUP
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=lds_128x64_bpack_frag" -air-gpu-outlining | FileCheck %s --check-prefixes=CHECK,FRAG
// RUN: not air-opt %s -air-to-rocdl="int8-gemm-variant=not_a_variant" 2>&1 | FileCheck %s --check-prefix=BAD
// RUN: not air-opt %s -air-to-rocdl="int8-gemm-group-size=6" 2>&1 | FileCheck %s --check-prefix=BADGROUP

// CHECK: gpu.launch_func @{{.*}}::@{{.*}} blocks in (%c16, %c8, %c1) threads in (%c256, %c1, %c1)
// CHECK: gpu.module @
// CHECK: gpu.func @{{.*}} kernel
// DEFAULT-SAME: air.gpu.int8_gemm_variant = "lds_128x64_wmma4"
// BPACK-SAME: air.gpu.int8_gemm_variant = "lds_128x64_bpack"
// SWIZZLE-SAME: air.gpu.int8_gemm_variant = "lds_128x64_bpack_swizzle"
// PIPE2-SAME: air.gpu.int8_gemm_variant = "lds_128x64_bpack_pipe2"
// GROUPED-SAME: air.gpu.int8_gemm_variant = "lds_128x64_bpack_pipe2_grouped"
// SWIZZLEGROUP-SAME: air.gpu.int8_gemm_group_m = 8 : i32
// SWIZZLEGROUP-SAME: air.gpu.int8_gemm_variant = "lds_128x64_bpack_swizzle_grouped"
// FRAG-SAME: air.gpu.int8_gemm_variant = "lds_128x64_bpack_frag"
// CHECK: rocdl.wmma.i32.16x16x16.iu8
// CHECK-NOT: rocdl.wmma.i32.16x16x64
// CHECK-NOT: swmmac
// BAD: unsupported variant 'not_a_variant'
// BADGROUP: unsupported group size '6'

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
