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
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=lds_128x64_bpack_swizzle_pipe2_k32_looped" -air-gpu-outlining | FileCheck %s --check-prefixes=CHECK,GRID64,PIPE64K32
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=lds_128x64_bpack_swizzle_pipe2_k128_looped" -air-gpu-outlining | FileCheck %s --check-prefixes=CHECK,GRID64,PIPE64K128
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=lds_128x128_bpack_swizzle_k32_looped" -air-gpu-outlining | FileCheck %s --check-prefixes=CHECK,GRID128,WIDEK32
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=lds_128x128_bpack_swizzle_k128_looped" -air-gpu-outlining | FileCheck %s --check-prefixes=CHECK,GRID128,WIDEK128
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=lds_128x64_bpack_swizzle_breg_k64_looped" -air-gpu-outlining | FileCheck %s --check-prefixes=CHECK,GRID64,BREG64
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=lds_128x64_bpack_swizzle_k32_w4_pipe2 int8-gemm-group-size=8" -air-gpu-outlining="int8-gemm-group-size=8" | FileCheck %s --check-prefixes=CHECK,GRID64W4,TLIKE64
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=lds_128x64_bpack_swizzle_k32_w4_pipe2_pad int8-gemm-group-size=8" -air-gpu-outlining="int8-gemm-group-size=8" | FileCheck %s --check-prefixes=CHECK,GRID64W4,TLIKE64PAD
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=lds_64x128_bpack_swizzle_k32_w4_pipe2 int8-gemm-group-size=8" -air-gpu-outlining="int8-gemm-group-size=8" | FileCheck %s --check-prefixes=CHECK,GRID64X128W4,TLIKE64X128
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=lds_64x128_bpack_swizzle_k32_w4_pipe2_pad int8-gemm-group-size=8" -air-gpu-outlining="int8-gemm-group-size=8" | FileCheck %s --check-prefixes=CHECK,GRID64X128W4,TLIKE64X128PAD
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=lds_128x128_bpack_swizzle_k32_w4_pipe2 int8-gemm-group-size=8" -air-gpu-outlining="int8-gemm-group-size=8" | FileCheck %s --check-prefixes=CHECK,GRID128W4,TLIKE
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=lds_128x128_bpack_swizzle_k32_w4_pipe2_pad int8-gemm-group-size=8" -air-gpu-outlining="int8-gemm-group-size=8" | FileCheck %s --check-prefixes=CHECK,GRID128W4,TLIKEPAD
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=lds_128x64_bpack_swizzle_k32_w4_pipe2_short int8-gemm-group-size=8" -air-gpu-outlining="int8-gemm-group-size=8" | FileCheck %s --check-prefixes=CHECK,GRID64W4,TSHORT64
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=lds_128x64_bpack_swizzle_k32_w4_pipe2_short_pad int8-gemm-group-size=8" -air-gpu-outlining="int8-gemm-group-size=8" | FileCheck %s --check-prefixes=CHECK,GRID64W4,TSHORT64PAD
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=lds_64x128_bpack_swizzle_k32_w4_pipe2_short int8-gemm-group-size=8" -air-gpu-outlining="int8-gemm-group-size=8" | FileCheck %s --check-prefixes=CHECK,GRID64X128W4,TSHORT64X128
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=lds_64x128_bpack_swizzle_k32_w4_pipe2_short_pad int8-gemm-group-size=8" -air-gpu-outlining="int8-gemm-group-size=8" | FileCheck %s --check-prefixes=CHECK,GRID64X128W4,TSHORT64X128PAD
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=lds_128x128_bpack_swizzle_k32_w4_pipe2_short int8-gemm-group-size=8" -air-gpu-outlining="int8-gemm-group-size=8" | FileCheck %s --check-prefixes=CHECK,GRID128W4,TSHORT
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=lds_128x128_bpack_swizzle_k32_w4_pipe2_short_pad int8-gemm-group-size=8" -air-gpu-outlining="int8-gemm-group-size=8" | FileCheck %s --check-prefixes=CHECK,GRID128W4,TSHORTPAD
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=lds_128x128_rocmlir_k32_pipe3 int8-gemm-group-size=8" -air-gpu-outlining="int8-gemm-group-size=8" | FileCheck %s --check-prefixes=CHECK,GRID128W4,ROCMLIRLIKE
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=lds_128x128_tensile_k32_pipe3 int8-gemm-group-size=8" -air-gpu-outlining="int8-gemm-group-size=8" | FileCheck %s --check-prefixes=CHECK,GRID128W4,TENSILEPIPE3
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=lds_128x128_tensile_k32_pipe3_pad int8-gemm-group-size=8" -air-gpu-outlining="int8-gemm-group-size=8" | FileCheck %s --check-prefixes=CHECK,GRID128W4,TENSILEPIPE3PAD
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=lds_128x128_tensile_k32_pipe3_wpe2 int8-gemm-group-size=8" -air-gpu-outlining="int8-gemm-group-size=8" | FileCheck %s --check-prefixes=CHECK,GRID128W4,TENSILEPIPE3WPE2
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=lds_128x128_tensile_k32_pipe2 int8-gemm-group-size=8" -air-gpu-outlining="int8-gemm-group-size=8" | FileCheck %s --check-prefixes=CHECK,GRID128W4,TENSILEPIPE2
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=lds_128x128_tensile_k32_pipe2_pad int8-gemm-group-size=8" -air-gpu-outlining="int8-gemm-group-size=8" | FileCheck %s --check-prefixes=CHECK,GRID128W4,TENSILEPIPE2PAD
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=global_128x128_bpack_w4_direct int8-gemm-group-size=8" -air-gpu-outlining="int8-gemm-group-size=8" | FileCheck %s --check-prefixes=CHECK,GRID128DIRECT,DIRECT
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=global_128x128_bpack_w4_direct_canonical int8-gemm-group-size=8" -air-gpu-outlining="int8-gemm-group-size=8" | FileCheck %s --check-prefixes=CHECK,GRID128DIRECT,DIRECTCANON
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=global_128x128_bpack_w4_prefetch int8-gemm-group-size=8" -air-gpu-outlining="int8-gemm-group-size=8" | FileCheck %s --check-prefixes=CHECK,GRID128DIRECT,DIRECTPREFETCH
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=global_128x128_bpack_w4_direct_rawptr int8-gemm-group-size=8" -air-gpu-outlining="int8-gemm-group-size=8" | FileCheck %s --check-prefixes=CHECK,GRID128RAWPTR,RAWPTR
// RUN: air-opt %s -air-to-rocdl="int8-gemm-variant=global_128x128_bpack_w4_direct_rawptr_u2 int8-gemm-group-size=8" -air-gpu-outlining="int8-gemm-group-size=8" | FileCheck %s --check-prefixes=CHECK,GRID128RAWPTR,RAWPTRU2
// RUN: not air-opt %s -air-to-rocdl="int8-gemm-variant=global_128x128_bpack_w4_direct int8-gemm-group-size=4" -air-gpu-outlining="int8-gemm-group-size=4" 2>&1 | FileCheck %s --check-prefix=BADDIRECTGROUP
// RUN: not air-opt %s -air-to-rocdl="int8-gemm-variant=lds_128x128_rocmlir_k32_pipe3 int8-gemm-group-size=4" -air-gpu-outlining="int8-gemm-group-size=4" 2>&1 | FileCheck %s --check-prefix=BADROCMLIRGROUP
// RUN: not air-opt %s -air-to-rocdl="int8-gemm-variant=lds_128x128_tensile_k32_pipe3 int8-gemm-group-size=4" -air-gpu-outlining="int8-gemm-group-size=4" 2>&1 | FileCheck %s --check-prefix=BADTENSILEPIPE3GROUP
// RUN: not air-opt %s -air-to-rocdl="int8-gemm-variant=lds_128x128_tensile_k32_pipe3_pad int8-gemm-group-size=4" -air-gpu-outlining="int8-gemm-group-size=4" 2>&1 | FileCheck %s --check-prefix=BADTENSILEPIPE3GROUP
// RUN: not air-opt %s -air-to-rocdl="int8-gemm-variant=lds_128x128_tensile_k32_pipe3_wpe2 int8-gemm-group-size=4" -air-gpu-outlining="int8-gemm-group-size=4" 2>&1 | FileCheck %s --check-prefix=BADTENSILEPIPE3GROUP
// RUN: not air-opt %s -air-to-rocdl="int8-gemm-variant=not_a_variant" 2>&1 | FileCheck %s --check-prefix=BAD
// RUN: not air-opt %s -air-to-rocdl="int8-gemm-group-size=6" 2>&1 | FileCheck %s --check-prefix=BADGROUP

// DIRECT: #llvm.loop_unroll<count = 4 : i32>
// DIRECT-NEXT: #llvm.loop_annotation<disableNonforced = true, unroll = #loop_unroll>
// RAWPTR: #llvm.loop_unroll<disable = true>
// RAWPTR-NEXT: #llvm.loop_annotation<disableNonforced = true, unroll = #loop_unroll>
// RAWPTRU2: #llvm.loop_unroll<count = 2 : i32>
// RAWPTRU2-NEXT: #llvm.loop_annotation<disableNonforced = true, unroll = #loop_unroll>
// GRID64: gpu.launch_func @{{.*}}::@{{.*}} blocks in (%c16, %c8, %c1) threads in (%c256, %c1, %c1)
// GRID64W4: gpu.launch_func @{{.*}}::@{{.*}} blocks in (%c16, %c8, %c1) threads in (%c32, %c4, %c1)
// GRID128: gpu.launch_func @{{.*}}::@{{.*}} blocks in (%c8{{(_[0-9]+)?}}, %c8{{(_[0-9]+)?}}, %c1) threads in (%c256, %c1, %c1)
// GRID128W4: gpu.launch_func @{{.*}}::@{{.*}} blocks in (%c8{{(_[0-9]+)?}}, %c8{{(_[0-9]+)?}}, %c1) threads in (%c32, %c4, %c1)
// GRID128DIRECT: gpu.launch_func @{{.*}}::@{{.*}} blocks in (%c8{{(_[0-9]+)?}}, %c8{{(_[0-9]+)?}}, %c1) threads in (%c128, %c1, %c1)
// GRID128RAWPTR: gpu.launch_func @{{.*}}::@{{.*}} blocks in (%c64{{(_[0-9]+)?}}, %c1{{(_[0-9]+)?}}, %c1{{(_[0-9]+)?}}) threads in (%c128, %c1, %c1)
// GRID64X128: gpu.launch_func @{{.*}}::@{{.*}} blocks in (%c8{{(_[0-9]+)?}}, %c16{{(_[0-9]+)?}}, %c1) threads in (%c128, %c1, %c1)
// GRID64X128W4: gpu.launch_func @{{.*}}::@{{.*}} blocks in (%c8{{(_[0-9]+)?}}, %c16{{(_[0-9]+)?}}, %c1) threads in (%c32, %c4, %c1)
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
// PIPE64K32-SAME: air.gpu.int8_gemm_variant = "lds_128x64_bpack_swizzle_pipe2_k32_looped"
// PIPE64K128-SAME: air.gpu.int8_gemm_variant = "lds_128x64_bpack_swizzle_pipe2_k128_looped"
// WIDEK32-SAME: air.gpu.int8_gemm_variant = "lds_128x128_bpack_swizzle_k32_looped"
// WIDEK128-SAME: air.gpu.int8_gemm_variant = "lds_128x128_bpack_swizzle_k128_looped"
// BREG64-SAME: air.gpu.int8_gemm_variant = "lds_128x64_bpack_swizzle_breg_k64_looped"
// TLIKE64-SAME: air.gpu.int8_gemm_group_m = 8 : i32
// TLIKE64-SAME: air.gpu.int8_gemm_variant = "lds_128x64_bpack_swizzle_k32_w4_pipe2"
// TLIKE64PAD-SAME: air.gpu.int8_gemm_group_m = 8 : i32
// TLIKE64PAD-SAME: air.gpu.int8_gemm_variant = "lds_128x64_bpack_swizzle_k32_w4_pipe2_pad"
// TLIKE64X128-SAME: air.gpu.int8_gemm_group_m = 8 : i32
// TLIKE64X128-SAME: air.gpu.int8_gemm_variant = "lds_64x128_bpack_swizzle_k32_w4_pipe2"
// TLIKE64X128PAD-SAME: air.gpu.int8_gemm_group_m = 8 : i32
// TLIKE64X128PAD-SAME: air.gpu.int8_gemm_variant = "lds_64x128_bpack_swizzle_k32_w4_pipe2_pad"
// TLIKE-SAME: air.gpu.int8_gemm_group_m = 8 : i32
// TLIKE-SAME: air.gpu.int8_gemm_variant = "lds_128x128_bpack_swizzle_k32_w4_pipe2"
// TLIKEPAD-SAME: air.gpu.int8_gemm_group_m = 8 : i32
// TLIKEPAD-SAME: air.gpu.int8_gemm_variant = "lds_128x128_bpack_swizzle_k32_w4_pipe2_pad"
// TSHORT64-SAME: air.gpu.int8_gemm_group_m = 8 : i32
// TSHORT64-SAME: air.gpu.int8_gemm_variant = "lds_128x64_bpack_swizzle_k32_w4_pipe2_short"
// TSHORT64PAD-SAME: air.gpu.int8_gemm_group_m = 8 : i32
// TSHORT64PAD-SAME: air.gpu.int8_gemm_variant = "lds_128x64_bpack_swizzle_k32_w4_pipe2_short_pad"
// TSHORT64X128-SAME: air.gpu.int8_gemm_group_m = 8 : i32
// TSHORT64X128-SAME: air.gpu.int8_gemm_variant = "lds_64x128_bpack_swizzle_k32_w4_pipe2_short"
// TSHORT64X128PAD-SAME: air.gpu.int8_gemm_group_m = 8 : i32
// TSHORT64X128PAD-SAME: air.gpu.int8_gemm_variant = "lds_64x128_bpack_swizzle_k32_w4_pipe2_short_pad"
// TSHORT-SAME: air.gpu.int8_gemm_group_m = 8 : i32
// TSHORT-SAME: air.gpu.int8_gemm_variant = "lds_128x128_bpack_swizzle_k32_w4_pipe2_short"
// TSHORTPAD-SAME: air.gpu.int8_gemm_group_m = 8 : i32
// TSHORTPAD-SAME: air.gpu.int8_gemm_variant = "lds_128x128_bpack_swizzle_k32_w4_pipe2_short_pad"
// ROCMLIRLIKE-SAME: air.gpu.int8_gemm_group_m = 8 : i32
// ROCMLIRLIKE-SAME: air.gpu.int8_gemm_variant = "lds_128x128_rocmlir_k32_pipe3"
// ROCMLIRLIKE: memref<128x32xi8, 3>
// ROCMLIRLIKE: gpu.barrier
// TENSILEPIPE3-SAME: air.gpu.int8_gemm_group_m = 8 : i32
// TENSILEPIPE3-SAME: air.gpu.int8_gemm_variant = "lds_128x128_tensile_k32_pipe3"
// TENSILEPIPE3: memref<128x32xi8, 3>
// TENSILEPIPE3: gpu.barrier
// TENSILEPIPE3PAD-SAME: air.gpu.int8_gemm_group_m = 8 : i32
// TENSILEPIPE3PAD-SAME: air.gpu.int8_gemm_variant = "lds_128x128_tensile_k32_pipe3_pad"
// TENSILEPIPE3PAD: memref<128x48xi8, 3>
// TENSILEPIPE3PAD: gpu.barrier
// TENSILEPIPE3WPE2-SAME: air.gpu.int8_gemm_group_m = 8 : i32
// TENSILEPIPE3WPE2-SAME: air.gpu.int8_gemm_variant = "lds_128x128_tensile_k32_pipe3_wpe2"
// TENSILEPIPE3WPE2-SAME: rocdl.waves_per_eu = 2 : i32
// TENSILEPIPE3WPE2: memref<128x32xi8, 3>
// TENSILEPIPE3WPE2: gpu.barrier
// TENSILEPIPE2-SAME: air.gpu.int8_gemm_group_m = 8 : i32
// TENSILEPIPE2-SAME: air.gpu.int8_gemm_variant = "lds_128x128_tensile_k32_pipe2"
// TENSILEPIPE2: memref<128x32xi8, 3>
// TENSILEPIPE2: gpu.barrier
// TENSILEPIPE2PAD-SAME: air.gpu.int8_gemm_group_m = 8 : i32
// TENSILEPIPE2PAD-SAME: air.gpu.int8_gemm_variant = "lds_128x128_tensile_k32_pipe2_pad"
// TENSILEPIPE2PAD: memref<128x48xi8, 3>
// TENSILEPIPE2PAD: gpu.barrier
// DIRECT: air.gpu.int8_gemm_group_m = 8 : i32
// DIRECT-SAME: air.gpu.int8_gemm_variant = "global_128x128_bpack_w4_direct"
// DIRECTCANON: air.gpu.int8_gemm_group_m = 8 : i32
// DIRECTCANON-SAME: air.gpu.int8_gemm_variant = "global_128x128_bpack_w4_direct_canonical"
// DIRECTPREFETCH: air.gpu.int8_gemm_group_m = 8 : i32
// DIRECTPREFETCH-SAME: air.gpu.int8_gemm_variant = "global_128x128_bpack_w4_prefetch"
// RAWPTR: air.gpu.int8_gemm_group_m = 8 : i32
// RAWPTR-SAME: air.gpu.int8_gemm_variant = "global_128x128_bpack_w4_direct_rawptr"
// RAWPTRU2: air.gpu.int8_gemm_group_m = 8 : i32
// RAWPTRU2-SAME: air.gpu.int8_gemm_variant = "global_128x128_bpack_w4_direct_rawptr_u2"
// DIRECT-NOT: memref<{{.*}}, 3>
// DIRECT-NOT: gpu.barrier
// DIRECT: scf.for
// DIRECT-SAME: step %c32{{.*}}
// DIRECTCANON-NOT: memref<{{.*}}, 3>
// DIRECTCANON-NOT: gpu.barrier
// DIRECTCANON: scf.for
// DIRECTCANON-SAME: step %c16{{.*}}
// DIRECTPREFETCH-NOT: memref<{{.*}}, 3>
// DIRECTPREFETCH-NOT: gpu.barrier
// DIRECTPREFETCH: scf.for
// DIRECTPREFETCH-SAME: step %c32{{.*}}
// RAWPTR-NOT: memref<{{.*}}, 3>
// RAWPTR-NOT: gpu.barrier
// RAWPTR: scf.for
// RAWPTR-SAME: step %c32{{.*}}
// RAWPTRU2-NOT: memref<{{.*}}, 3>
// RAWPTRU2-NOT: gpu.barrier
// RAWPTRU2: scf.for
// RAWPTRU2-SAME: step %c16{{.*}}
// DIRECT-NOT: gpu.barrier
// CHECK: rocdl.wmma.i32.16x16x16.iu8
// CHECK-NOT: rocdl.wmma.i32.16x16x64
// CHECK-NOT: swmmac
// DIRECT: rocdl.wmma.i32.16x16x16.iu8
// DIRECT: } {loop_annotation = #loop_annotation}
// RAWPTR: rocdl.wmma.i32.16x16x16.iu8
// RAWPTR: } {loop_annotation = #loop_annotation}
// RAWPTRU2: rocdl.wmma.i32.16x16x16.iu8
// RAWPTRU2: } {loop_annotation = #loop_annotation}
// DIRECT-NOT: gpu.barrier
// DIRECT: gpu.return
// RAWPTR-NOT: gpu.barrier
// RAWPTR: gpu.return
// RAWPTRU2-NOT: gpu.barrier
// RAWPTRU2: gpu.return
// BAD: unsupported variant 'not_a_variant'
// BADGROUP: unsupported group size '6'
// BADDIRECTGROUP: direct global variant requires group size 8
// BADROCMLIRGROUP: rocMLIR-like variant requires group size 8
// BADTENSILEPIPE3GROUP: Tensile-like pipe3 variant requires group size 8

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
