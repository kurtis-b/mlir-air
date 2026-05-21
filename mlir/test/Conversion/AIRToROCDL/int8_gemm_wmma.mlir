//===- int8_gemm_wmma.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// REQUIRES: gpu
// RUN: air-opt %s -air-to-rocdl -air-gpu-outlining | FileCheck %s --check-prefixes=CHECK,GRID64,DEFAULT
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=lds_128x64_bpack" -air-gpu-outlining | FileCheck %s --check-prefixes=CHECK,GRID64,BPACK
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=lds_128x64_bpack_swizzle" -air-gpu-outlining | FileCheck %s --check-prefixes=CHECK,GRID64,SWIZZLE
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=lds_128x64_bpack_pipe2" -air-gpu-outlining | FileCheck %s --check-prefixes=CHECK,GRID64,PIPE2
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=lds_128x64_bpack_pipe2_grouped" -air-gpu-outlining | FileCheck %s --check-prefixes=CHECK,GRID64,GROUPED
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=lds_128x64_bpack_swizzle_grouped int8-gemm-group-size=8" -air-gpu-outlining="int8-gemm-group-size=8" | FileCheck %s --check-prefixes=CHECK,GRID64,SWIZZLEGROUP
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=lds_128x64_bpack_frag" -air-gpu-outlining | FileCheck %s --check-prefixes=CHECK,GRID64,FRAG
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=lds_128x128_bpack_swizzle_pipe2" -air-gpu-outlining | FileCheck %s --check-prefixes=CHECK,GRID128,PIPE128
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=lds_128x64_bpack_swizzle_pipe2_looped" -air-gpu-outlining | FileCheck %s --check-prefixes=CHECK,GRID64,PIPE64LOOP
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=lds_128x128_bpack_swizzle_looped" -air-gpu-outlining | FileCheck %s --check-prefixes=CHECK,GRID128,WIDELOOP
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=lds_64x128_bpack_swizzle_pipe2_looped" -air-gpu-outlining | FileCheck %s --check-prefixes=CHECK,GRID64X128,SHORTPIPE
// RUN: not air-opt %s -air-to-rocdl="int8-gemm-variant=not_a_variant" 2>&1 | FileCheck %s --check-prefix=BAD
// RUN: not air-opt %s -air-to-rocdl="int8-gemm-group-size=6" 2>&1 | FileCheck %s --check-prefix=BADGROUP

// GRID64: gpu.launch_func @{{.*}}::@{{.*}} blocks in (%c16, %c8, %c1) threads in (%c256, %c1, %c1)
// GRID128: gpu.launch_func @{{.*}}::@{{.*}} blocks in (%c8{{(_[0-9]+)?}}, %c8{{(_[0-9]+)?}}, %c1) threads in (%c256, %c1, %c1)
// GRID64X128: gpu.launch_func @{{.*}}::@{{.*}} blocks in (%c8{{(_[0-9]+)?}}, %c16{{(_[0-9]+)?}}, %c1) threads in (%c128, %c1, %c1)
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
// PIPE128-SAME: air.gpu.int8_gemm_variant = "lds_128x128_bpack_swizzle_pipe2"
// PIPE64LOOP-SAME: air.gpu.int8_gemm_variant = "lds_128x64_bpack_swizzle_pipe2_looped"
// WIDELOOP-SAME: air.gpu.int8_gemm_variant = "lds_128x128_bpack_swizzle_looped"
// SHORTPIPE-SAME: air.gpu.int8_gemm_variant = "lds_64x128_bpack_swizzle_pipe2_looped"
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
