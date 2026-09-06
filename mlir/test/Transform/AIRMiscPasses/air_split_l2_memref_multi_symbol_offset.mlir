//===- air_split_l2_memref_multi_symbol_offset.mlir -------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s --air-split-l2-memref="tiles-per-l2-tile=1 max-launch-channels-mm2s=16 max-launch-channels-s2mm=16" --split-input-file | FileCheck %s

// An L3-side channel offset produced by a multi-level loop nest is a
// MULTI-symbol affine.apply -- advancing a staged L3 buffer over a (group, step)
// nest gives affine_map<()[s0, s1] -> (s0 * A + s1 * B)>. When the pass splits
// the L2 buffer such an access feeds, tileChannelOpByFactor rebuilds that offset
// per split by composing with the existing apply.
//
// The rebuild used to hardcode the replacement map to exactly one symbol --
// AffineMap::get(0, 1, add) at all three of its construction sites -- while the
// composed expression still referenced every symbol the original apply used. On
// any nest deeper than one level that names a symbol position the map does not
// declare, so MLIR's willBeValidAffineMap assertion fires and air-opt SIGABRTs
// (exit 134) inside AIRSplitL2MemrefForBufferConstraintPass. Both cases below
// abort the unfixed pass; the fix sizes the replacement map from the map the
// offset was lifted out of.
//
// This is not a corner: a two-level nest over an L3 operand is the ordinary
// shape, and one symbol only ever sufficed because the pass happened to decline
// the split wherever a deeper nest appeared.

// Case 1 -- contiguous access, offset carried in a bare affine.apply, so the
// pass composes from the split sizes alone. A 1024-element L2 buffer split
// 8 ways gives eight 128-element slices, and each slice offset must be the
// launch's own two-symbol base PLUS its slice displacement. Dropping either
// launch symbol would collapse every launch iteration onto one base.

// CHECK-DAG: #[[$BASE:.+]] = affine_map<()[s0, s1] -> (s0 * 2048 + s1 * 1024)>
// CHECK-DAG: #[[$SLICE1:.+]] = affine_map<()[s0, s1] -> (s0 * 2048 + s1 * 1024 + 128)>
// CHECK-DAG: #[[$SLICE7:.+]] = affine_map<()[s0, s1] -> (s0 * 2048 + s1 * 1024 + 896)>

// CHECK-LABEL: func.func @two_symbol_offset_contiguous
// CHECK: air.launch async (%[[X:[a-z0-9_]+]], %[[Y:[a-z0-9_]+]],
// Both launch induction variables still reach every split offset.
// CHECK-DAG: affine.apply #[[$BASE]]()[%[[X]], %[[Y]]]
// CHECK-DAG: affine.apply #[[$SLICE1]]()[%[[X]], %[[Y]]]
// CHECK-DAG: affine.apply #[[$SLICE7]]()[%[[X]], %[[Y]]]
// Eight puts, one per memtile column, each moving a 1024/8 = 128 element slice.
// CHECK: air.channel.put {{.*}}[%c0{{.*}}, %c0{{.*}}] (%{{.*}}[%{{.*}}] [%c128{{.*}}] [
// CHECK: air.channel.put {{.*}}[%c1{{.*}}, %c0{{.*}}] (%{{.*}}[%{{.*}}] [%c128{{.*}}] [
// CHECK: air.channel.put {{.*}}[%c7{{.*}}, %c0{{.*}}] (%{{.*}}[%{{.*}}] [%c128{{.*}}] [

#map = affine_map<()[s0] -> (s0 * 128)>
#map2 = affine_map<()[s0, s1] -> (s0 * 2048 + s1 * 1024)>
module {
  air.channel @channel_0 []
  air.channel @channel_1 []
  air.channel @channel_2 [8, 1]
  air.channel @channel_3 [8, 1]
  air.channel @channel_4 [8, 1]
  air.channel @channel_5 []
  func.func @two_symbol_offset_contiguous(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: memref<*xbf16>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = air.launch async (%arg9, %arg10, %arg11) in (%arg12=%c2, %arg13=%c2, %arg14=%c1) args(%arg15=%arg0, %arg16=%arg1, %arg17=%arg2) : memref<*xbf16>, memref<*xbf16>, memref<*xbf16> attributes {id = 1 : i32} {
      %c1024 = arith.constant 1024 : index
      %c1_0 = arith.constant 1 : index
      %5 = affine.apply #map2()[%arg9, %arg10]
      %6 = air.channel.put async  @channel_0[] (%arg15[%5] [%c1024] [%c1_0]) {id = 1 : i32} : (memref<*xbf16>)
      %7 = air.channel.put async  @channel_1[] (%arg16[%5] [%c1024] [%c1_0]) {id = 2 : i32} : (memref<*xbf16>)
      %8 = air.channel.get async  @channel_5[] (%arg17[%5] [%c1024] [%c1_0]) {id = 3 : i32} : (memref<*xbf16>)
      %9 = air.segment @vecadd_0 async  attributes {id = 2 : i32} {
        %c128 = arith.constant 128 : index
        %c0 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c1_1 = arith.constant 1 : index
        %async_token, %results = air.execute -> (memref<1024xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<1024xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<1024xbf16, 1 : i32>
        }
        %10 = air.channel.get async [%async_token]  @channel_0[] (%results[] [] []) {id = 4 : i32} : (memref<1024xbf16, 1 : i32>)
        %async_token_2, %results_3 = air.execute -> (memref<1024xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<1024xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<1024xbf16, 1 : i32>
        }
        %11 = air.channel.get async [%async_token_2]  @channel_1[] (%results_3[] [] []) {id = 5 : i32} : (memref<1024xbf16, 1 : i32>)
        %async_token_4, %results_5 = air.execute -> (memref<1024xbf16, 1>) {
          %alloc = memref.alloc() : memref<1024xbf16, 1>
          air.execute_terminator %alloc : memref<1024xbf16, 1>
        }
        %12 = air.wait_all async [%10, %async_token_4]
        %13 = scf.parallel (%arg18) = (%c0) to (%c8) step (%c1_1) init (%12) -> !air.async.token {
          %19 = affine.apply #map()[%arg18]
          %20 = air.channel.put async [%10, %async_token_4]  @channel_2[%arg18, %c0] (%results[%19] [%c128] [%c1_1]) {id = 6 : i32} : (memref<1024xbf16, 1 : i32>)
          scf.reduce(%20 : !air.async.token) {
          ^bb0(%arg19: !air.async.token, %arg20: !air.async.token):
            %21 = air.wait_all async [%arg19, %arg20]
            scf.reduce.return %21 : !air.async.token
          }
        }
        %14 = air.wait_all async [%11, %async_token_4]
        %15 = scf.parallel (%arg18) = (%c0) to (%c8) step (%c1_1) init (%14) -> !air.async.token {
          %19 = affine.apply #map()[%arg18]
          %20 = air.channel.put async [%11, %async_token_4]  @channel_3[%arg18, %c0] (%results_3[%19] [%c128] [%c1_1]) {id = 7 : i32} : (memref<1024xbf16, 1 : i32>)
          scf.reduce(%20 : !air.async.token) {
          ^bb0(%arg19: !air.async.token, %arg20: !air.async.token):
            %21 = air.wait_all async [%arg19, %arg20]
            scf.reduce.return %21 : !air.async.token
          }
        }
        %16 = scf.parallel (%arg18) = (%c0) to (%c8) step (%c1_1) init (%async_token_4) -> !air.async.token {
          %19 = affine.apply #map()[%arg18]
          %20 = air.channel.get async [%async_token_4]  @channel_4[%arg18, %c0] (%results_5[%19] [%c128] [%c1_1]) {id = 8 : i32} : (memref<1024xbf16, 1>)
          scf.reduce(%20 : !air.async.token) {
          ^bb0(%arg19: !air.async.token, %arg20: !air.async.token):
            %21 = air.wait_all async [%arg19, %arg20]
            scf.reduce.return %21 : !air.async.token
          }
        }
        %17 = air.herd @herd_0 async [%async_token_4]  tile (%arg18, %arg19) in (%arg20=%c8, %arg21=%c1_1) attributes {id = 3 : i32} {
          %19 = ub.poison : bf16
          %c0_7 = arith.constant 0 : index
          %c128_8 = arith.constant 128 : index
          %c32 = arith.constant 32 : index
          %async_token_9, %results_10 = air.execute -> (memref<128xbf16, 2>) {
            %alloc = memref.alloc() : memref<128xbf16, 2>
            air.execute_terminator %alloc : memref<128xbf16, 2>
          }
          %20 = air.channel.get async [%async_token_9]  @channel_2[%arg18, %arg19] (%results_10[] [] []) {id = 9 : i32} : (memref<128xbf16, 2>)
          %async_token_11, %results_12 = air.execute -> (memref<128xbf16, 2>) {
            %alloc = memref.alloc() : memref<128xbf16, 2>
            air.execute_terminator %alloc : memref<128xbf16, 2>
          }
          %21 = air.channel.get async [%async_token_11]  @channel_3[%arg18, %arg19] (%results_12[] [] []) {id = 10 : i32} : (memref<128xbf16, 2>)
          %async_token_13, %results_14 = air.execute -> (memref<128xbf16, 2>) {
            %alloc = memref.alloc() : memref<128xbf16, 2>
            air.execute_terminator %alloc : memref<128xbf16, 2>
          }
          %22 = air.wait_all async [%20, %21, %async_token_13]
          %23 = scf.for %arg22 = %c0_7 to %c128_8 step %c32 iter_args(%arg23 = %22) -> (!air.async.token) {
            %subview = memref.subview %results_10[%arg22] [32] [1] : memref<128xbf16, 2> to memref<32xbf16, strided<[1], offset: ?>, 2>
            %subview_18 = memref.subview %results_12[%arg22] [32] [1] : memref<128xbf16, 2> to memref<32xbf16, strided<[1], offset: ?>, 2>
            %subview_19 = memref.subview %results_14[%arg22] [32] [1] : memref<128xbf16, 2> to memref<32xbf16, strided<[1], offset: ?>, 2>
            %async_token_20, %results_21 = air.execute [%arg23] -> (vector<32xbf16>) {
              %27 = vector.transfer_read %subview[%c0_7], %19 {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2>, vector<32xbf16>
              air.execute_terminator %27 : vector<32xbf16>
            }
            %async_token_22, %results_23 = air.execute [%arg23] -> (vector<32xbf16>) {
              %27 = vector.transfer_read %subview_18[%c0_7], %19 {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2>, vector<32xbf16>
              air.execute_terminator %27 : vector<32xbf16>
            }
            %25 = arith.addf %results_21, %results_23 : vector<32xbf16>
            %async_token_24 = air.execute [%arg23] {
              vector.transfer_write %25, %subview_19[%c0_7] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2>
            }
            %26 = air.wait_all async [%async_token_20, %async_token_22, %async_token_24]
            scf.yield %26 : !air.async.token
          }
          %24 = air.channel.put async [%async_token_13]  @channel_4[%arg18, %arg19] (%results_14[] [] []) {id = 11 : i32} : (memref<128xbf16, 2>)
          %async_token_15 = air.execute [%20] {
            memref.dealloc %results_10 : memref<128xbf16, 2>
          }
          %async_token_16 = air.execute [%21] {
            memref.dealloc %results_12 : memref<128xbf16, 2>
          }
          %async_token_17 = air.execute [%24] {
            memref.dealloc %results_14 : memref<128xbf16, 2>
          }
        }
        %18 = air.channel.put async [%17]  @channel_5[] (%results_5[] [] []) {id = 12 : i32} : (memref<1024xbf16, 1>)
        %async_token_6 = air.execute [%18] {
          memref.dealloc %results_5 : memref<1024xbf16, 1>
        }
      }
    }
    return
  }
}

// -----

// Case 2 -- overlapping (conv 3x3 stride 2) access, and the two-symbol offset is
// wrapped in an air.execute. Here the pass has BOTH a split affine_map to
// compose with and a split offset to substitute into it, which is the other
// composition path; it too built a one-symbol replacement. The four split
// offsets keep the overlap stride of 2 on top of the two-symbol base, and the
// base's second symbol (%arg5) must survive the rebuild.

// CHECK-DAG: #[[$CONVBASE:.+]] = affine_map<()[s0, s1] -> (s0 * 2 + s1 * 128)>
// CHECK-DAG: #[[$CONV1:.+]] = affine_map<()[s0, s1] -> (s0 * 2 + s1 * 128 + 2)>
// CHECK-DAG: #[[$CONV2:.+]] = affine_map<()[s0, s1] -> (s0 * 2 + s1 * 128 + 4)>
// CHECK-DAG: #[[$CONV3:.+]] = affine_map<()[s0, s1] -> (s0 * 2 + s1 * 128 + 6)>

// CHECK-LABEL: func.func @two_symbol_offset_overlapping
// CHECK: air.launch async (%[[U:[a-z0-9_]+]], %{{[a-z0-9_]+}}, %[[W:[a-z0-9_]+]])
// CHECK-DAG: %[[Q0:.*]] = affine.apply #[[$CONVBASE]]()[%[[U]], %[[W]]]
// CHECK-DAG: %[[Q1:.*]] = affine.apply #[[$CONV1]]()[%[[U]], %[[W]]]
// CHECK-DAG: %[[Q2:.*]] = affine.apply #[[$CONV2]]()[%[[U]], %[[W]]]
// CHECK-DAG: %[[Q3:.*]] = affine.apply #[[$CONV3]]()[%[[U]], %[[W]]]
// One put per split, each reading its own slice of the L3 operand.
//
// The third offset is deliberately NOT pinned to %c0 here, and that is a
// difference from the branch this test came from. `getOriginalApplyOperands`
// propagates a non-zero base on the split dim rather than zeroing it "just
// because the access happens to be contiguous" (AIRMiscPasses.cpp), so this
// pass emits the base's own apply in that slot. What the test pins is what it
// was written to pin: one put per split, each carrying ITS OWN Q_i on the split
// dimension, in split order.
// CHECK: air.channel.put {{.*}}[%c0{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %[[Q0]], %{{.*}},
// CHECK: air.channel.put {{.*}}[%c1{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %[[Q1]], %{{.*}},
// CHECK: air.channel.put {{.*}}[%c2{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %[[Q2]], %{{.*}},
// CHECK: air.channel.put {{.*}}[%c3{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %[[Q3]], %{{.*}},

#map = affine_map<()[s0, s1] -> (s0 * 2 + s1 * 128)>
#map1 = affine_map<()[s0] -> (s0 * 32)>
#map2 = affine_map<()[s0, s1] -> (s0 + s1 * 2)>
#map3 = affine_map<()[s0, s1] -> (s0 + s1 * 8)>
module {
  air.channel @channel_3 [4, 4]
  air.channel @channel_1 [1, 1]
  func.func @two_symbol_offset_overlapping(%arg0: memref<1x513x513x16xi8>, %arg1: memref<3x3x16x32xi8>, %arg2: memref<1x256x256x32xi32>) {
    %c64 = arith.constant 64 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %0 = air.launch async (%arg3, %arg4, %arg5) in (%arg6=%c64, %arg7=%c16, %arg8=%c4) args(%arg9=%arg0) : memref<1x513x513x16xi8> attributes {id = 1 : i32} {
      %c8208 = arith.constant 8208 : index
      %c4210704 = arith.constant 4210704 : index
      %c16_0 = arith.constant 16 : index
      %c33 = arith.constant 33 : index
      %c9 = arith.constant 9 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %async_token, %results = air.execute -> (index) {
        %3 = affine.apply #map()[%arg3, %arg5]
        air.execute_terminator %3 : index
      }
      %async_token_1, %results_2 = air.execute -> (index) {
        %3 = affine.apply #map1()[%arg4]
        air.execute_terminator %3 : index
      }
      %1 = air.channel.put async [%async_token, %async_token_1]  @channel_1[] (%arg9[%c0, %results, %results_2, %c0] [%c1, %c9, %c33, %c16_0] [%c4210704, %c8208, %c16_0, %c1]) {id = 1 : i32} : (memref<1x513x513x16xi8>)
      %2 = air.segment @segment_0 async  attributes {id = 2 : i32} {
        %c7 = arith.constant 7 : index
        %c4752 = arith.constant 4752 : index
        %c528 = arith.constant 528 : index
        %c8 = arith.constant 8 : index
        %c3 = arith.constant 3 : index
        %c16_3 = arith.constant 16 : index
        %c1_4 = arith.constant 1 : index
        %c0_5 = arith.constant 0 : index
        %c4_6 = arith.constant 4 : index
        %3 = air.wait_all async
        %4 = air.wait_all async
        %async_token_7, %results_8 = air.execute -> (memref<1x9x33x16xi8, 1 : i32>) {
          %alloc = memref.alloc() : memref<1x9x33x16xi8, 1 : i32>
          air.execute_terminator %alloc : memref<1x9x33x16xi8, 1 : i32>
        }
        %5 = air.channel.get async [%3, %4, %async_token_7]  @channel_1[] (%results_8[] [] []) {id = 4 : i32} : (memref<1x9x33x16xi8, 1 : i32>)
        %6 = scf.parallel (%arg10, %arg11) = (%c0_5, %c0_5) to (%c4_6, %c4_6) step (%c1_4, %c1_4) init (%5) -> !air.async.token {
          %8 = scf.for %arg12 = %c0_5 to %c3 step %c1_4 iter_args(%arg13 = %5) -> (!air.async.token) {
            %9 = scf.for %arg14 = %c0_5 to %c3 step %c1_4 iter_args(%arg15 = %arg13) -> (!air.async.token) {
              %10 = scf.for %arg16 = %c0_5 to %c16_3 step %c8 iter_args(%arg17 = %arg15) -> (!air.async.token) {
                %async_token_10, %results_11 = air.execute [%arg17] -> (index) {
                  %12 = affine.apply #map2()[%arg12, %arg10]
                  air.execute_terminator %12 : index
                }
                %async_token_12, %results_13 = air.execute [%arg17] -> (index) {
                  %12 = affine.apply #map3()[%arg14, %arg11]
                  air.execute_terminator %12 : index
                }
                %11 = air.channel.put async [%async_token_10, %async_token_12]  @channel_3[%arg10, %arg11] (%results_8[%c0_5, %results_11, %results_13, %arg16] [%c1_4, %c1_4, %c7, %c8] [%c4752, %c528, %c16_3, %c1_4]) {id = 7 : i32} : (memref<1x9x33x16xi8, 1 : i32>)
                scf.yield %11 : !air.async.token
              }
              scf.yield %10 : !air.async.token
            }
            scf.yield %9 : !air.async.token
          }
          scf.reduce(%8 : !air.async.token) {
          ^bb0(%arg12: !air.async.token, %arg13: !air.async.token):
            %9 = air.wait_all async [%arg12, %arg13]
            scf.reduce.return %9 : !air.async.token
          }
        }
        %7 = air.herd @herd_0 async [%5]  tile (%arg10, %arg11) in (%arg12=%c4_6, %arg13=%c4_6) attributes {id = 3 : i32} {
          %c0_10 = arith.constant 0 : index
          %c16_11 = arith.constant 16 : index
          %c8_12 = arith.constant 8 : index
          %c3_13 = arith.constant 3 : index
          %c1_14 = arith.constant 1 : index
          %8 = air.wait_all async
          %9 = scf.for %arg14 = %c0_10 to %c3_13 step %c1_14 iter_args(%arg15 = %8) -> (!air.async.token) {
            %10 = scf.for %arg16 = %c0_10 to %c3_13 step %c1_14 iter_args(%arg17 = %arg15) -> (!air.async.token) {
              %11 = scf.for %arg18 = %c0_10 to %c16_11 step %c8_12 iter_args(%arg19 = %arg17) -> (!air.async.token) {
                %async_token_15, %results_16 = air.execute -> (memref<1x1x7x8xi8, 2 : i32>) {
                  %alloc = memref.alloc() : memref<1x1x7x8xi8, 2 : i32>
                  air.execute_terminator %alloc : memref<1x1x7x8xi8, 2 : i32>
                }
                %12 = air.channel.get async [%arg19, %async_token_15]  @channel_3[%arg10, %arg11] (%results_16[] [] []) {id = 9 : i32} : (memref<1x1x7x8xi8, 2 : i32>)
                %async_token_17 = air.execute {
                  memref.dealloc %results_16 : memref<1x1x7x8xi8, 2 : i32>
                }
                scf.yield %12 : !air.async.token
              }
              scf.yield %11 : !air.async.token
            }
            scf.yield %10 : !air.async.token
          }
        }
        %async_token_9 = air.execute [%5] {
          memref.dealloc %results_8 : memref<1x9x33x16xi8, 1 : i32>
        }
      }
    }
    return
  }
}
