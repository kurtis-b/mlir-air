//===- air_split_l2_memref_no_l2_split.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s --air-split-l2-memref="tiles-per-l2-tile=1" | FileCheck %s

#map = affine_map<()[s0] -> (s0 * 256)>
#map1 = affine_map<()[s0] -> (s0 * 64)>
air.channel @channel_1 [1, 1]
air.channel @channel_0 [4, 4]

// CHECK-LABEL: func.func @no_l2_split
// CHECK: memref.alloc() {air.no_l2_split = true, air.shrinkage = false} : memref<256x256xbf16, 1 : i32>
// CHECK-NOT: memref.alloc() : memref<64x256xbf16, 1 : i32>
func.func @no_l2_split(%arg0: memref<512x512xbf16>) {
  %c2 = arith.constant 2 : index
  %0 = air.launch async (%arg1, %arg2) in (%arg3=%c2, %arg4=%c2) args(%arg5=%arg0) : memref<512x512xbf16> attributes {id = 1 : i32} {
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %async_token, %results = air.execute -> (index) {
      %3 = affine.apply #map()[%arg1]
      air.execute_terminator %3 : index
    }
    %async_token_0, %results_1 = air.execute -> (index) {
      %3 = affine.apply #map()[%arg2]
      air.execute_terminator %3 : index
    }
    %1 = air.channel.get async [%async_token, %async_token_0] @channel_1[] (%arg5[%results, %results_1] [%c256, %c256] [%c512, %c1]) {id = 3 : i32} : (memref<512x512xbf16>)
    %2 = air.segment @segment_0 async {
      %c64 = arith.constant 64 : index
      %c1_2 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c256_3 = arith.constant 256 : index
      %3 = air.wait_all async
      %async_token_4, %results_5 = air.execute -> (memref<256x256xbf16, 1 : i32>) {
        %alloc = memref.alloc() {air.no_l2_split = true, air.shrinkage = false} : memref<256x256xbf16, 1 : i32>
        air.execute_terminator %alloc : memref<256x256xbf16, 1 : i32>
      }
      %5 = scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%c4, %c4) step (%c1_2, %c1_2) init (%async_token_4) -> !air.async.token {
        %async_token_7, %results_8 = air.execute -> (index) {
          %9 = affine.apply #map1()[%arg6]
          air.execute_terminator %9 : index
        }
        %async_token_9, %results_10 = air.execute -> (index) {
          %9 = affine.apply #map1()[%arg7]
          air.execute_terminator %9 : index
        }
        %8 = air.channel.get async [%async_token_4, %async_token_9, %async_token_7] @channel_0[%arg6, %arg7] (%results_5[%results_8, %results_10] [%c64, %c64] [%c256_3, %c1_2]) {id = 24 : i32} : (memref<256x256xbf16, 1 : i32>)
        scf.reduce(%8 : !air.async.token) {
        ^bb0(%arg8: !air.async.token, %arg9: !air.async.token):
          %9 = air.wait_all async [%arg8, %arg9]
          scf.reduce.return %9 : !air.async.token
        }
      }
      %6 = air.herd @herd_0 async [%async_token_4] tile (%arg6, %arg7) in (%arg8=%c4, %arg9=%c4) attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
        %c64_7 = arith.constant 64 : index
        %c256_8 = arith.constant 256 : index
        %c4_9 = arith.constant 4 : index
        %c16 = arith.constant 16 : index
        %c1_10 = arith.constant 1 : index
        %c0_11 = arith.constant 0 : index
        %async_token_12, %results_13 = air.execute -> (memref<16x16x4x4xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<16x16x4x4xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<16x16x4x4xbf16, 2 : i32>
        }
        %8 = air.channel.put async [%async_token_12] @channel_0[%arg6, %arg7] (%results_13[%c0_11, %c0_11, %c0_11] [%c64_7, %c16, %c4_9] [%c4_9, %c256_8, %c1_10]) {id = 41 : i32} : (memref<16x16x4x4xbf16, 2 : i32>)
        %async_token_14 = air.execute [%8] {
          memref.dealloc %results_13 : memref<16x16x4x4xbf16, 2 : i32>
        }
      }
      %7 = air.channel.put async [%3, %6] @channel_1[] (%results_5[] [] []) {id = 42 : i32} : (memref<256x256xbf16, 1 : i32>)
      %async_token_6 = air.execute [%7] {
        memref.dealloc %results_5 : memref<256x256xbf16, 1 : i32>
      }
    }
  }
  return
}
