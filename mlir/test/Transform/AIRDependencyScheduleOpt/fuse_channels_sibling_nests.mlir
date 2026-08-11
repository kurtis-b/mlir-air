//===- fuse_channels_sibling_nests.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-fuse-channels="aggressive-mode=false" --split-input-file | FileCheck %s

// N sibling same-bounds scf.for nests, each carrying one put of its own
// channel, all N channels mutually NFL-mergeable. This shape used to be a
// use-after-free: the pairwise candidate loop had no merge roles, so on a
// 3-clique the third pair put one channel's ops into both the
// fuse-destination and the erased sets, and wrapRegionsWithForLoops
// clones-and-erases what the erase loop then reads (SEGV/hang under air-opt,
// an ASLR coin toss under aircc). Both cases below crash the unfixed pass.
//
// Case 1 pins the SEMANTIC boundary, per side: the NFL merge clones only the
// destination's transfer body and erases the sources on any side it fuses,
// so the put side -- whose three nests read DIFFERENT dynamic offsets (+6144
// / +12288, the shape ffn_resident's down feed presents after
// air-dma-to-channel unrolls its sub-channel dimension) -- must NOT be
// loop-fused: all three puts survive on the merged channel, each keeping its
// own offsets, so every slice's data still moves. Only the
// pattern-identical get side is wrapped in the 1 + k = 3-trip multiplexing
// loop (6 x 3 = 18 gets against 3 x 6 = 18 surviving puts). The pre-fix
// pass wrapped the put side too, transferring the destination's slice three
// times and dropping the other two slices' data.
//
// Case 2 pins the N-way multiplex where BOTH sides are identical: four
// channels of 6 whole-buffer puts each fuse into one 4 x 6 = 24-slot
// stream, matching the consumer -- not the pairwise 2 the wrap hardcoded.
//
// Minimized by agents/probes/probe_fuse_channels_sibling_nests.py, which
// measured the N=2 clean / N=3 crash boundary on the unfixed pass.

// CHECK-LABEL: func.func @heterogeneous_offsets_keep_puts
// CHECK: air.launch
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK: air.channel.put {{.*}} @chan_a
// CHECK: scf.for
// CHECK: air.channel.put {{.*}} @chan_a
// CHECK: scf.for
// CHECK: air.channel.put {{.*}} @chan_a
// CHECK-NOT: air.channel.put {{.*}} @chan_b
// CHECK-NOT: air.channel.put {{.*}} @chan_c
// CHECK: air.segment
// CHECK: %[[C3:.*]] = arith.constant 3 : index
// CHECK: scf.for %{{.*}} = %{{.*}} to %[[C3]] step
// CHECK-NEXT: air.channel.get {{.*}} @chan_a
// CHECK-NOT: air.channel.get {{.*}} @chan_b
// CHECK-NOT: air.channel.get {{.*}} @chan_c
#map = affine_map<()[s0] -> (s0 * 1024)>
#map1 = affine_map<()[s0] -> (s0 * 1024 + 6144)>
#map2 = affine_map<()[s0] -> (s0 * 1024 + 12288)>
module {
  air.channel @feed [1, 1]
  air.channel @chan_a []
  air.channel @chan_b []
  air.channel @chan_c []
  func.func @heterogeneous_offsets_keep_puts(%arg0: memref<18432xbf16>) {
    %0 = air.launch async () in () args(%arg1=%arg0) : memref<18432xbf16> {
      %c6 = arith.constant 6 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %1 = air.wait_all async
      %2 = scf.for %arg2 = %c0 to %c6 step %c1 iter_args(%arg3 = %1) -> (!air.async.token) {
        %11 = affine.apply #map()[%arg2]
        %12 = air.channel.put async [%arg3]  @chan_a[] (%arg1[%11] [1024] [1]) : (memref<18432xbf16>)
        scf.yield %12 : !air.async.token
      }
      %3 = air.wait_all async
      %5 = scf.for %arg2 = %c0 to %c6 step %c1 iter_args(%arg3 = %3) -> (!air.async.token) {
        %11 = affine.apply #map1()[%arg2]
        %12 = air.channel.put async [%arg3]  @chan_b[] (%arg1[%11] [1024] [1]) : (memref<18432xbf16>)
        scf.yield %12 : !air.async.token
      }
      %6 = air.wait_all async
      %9 = scf.for %arg2 = %c0 to %c6 step %c1 iter_args(%arg3 = %6) -> (!air.async.token) {
        %11 = affine.apply #map2()[%arg2]
        %12 = air.channel.put async [%arg3]  @chan_c[] (%arg1[%11] [1024] [1]) : (memref<18432xbf16>)
        scf.yield %12 : !air.async.token
      }
      %10 = air.segment @seg async {
        %c6_0 = arith.constant 6 : index
        %c0_1 = arith.constant 0 : index
        %c1_2 = arith.constant 1 : index
        %async_token, %results = air.execute -> (memref<1024xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<1024xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<1024xbf16, 1 : i32>
        }
        %11 = air.herd @consumer async tile (%arg2, %arg3) in (%arg4=%c1_2, %arg5=%c1_2) {
          %c1_4 = arith.constant 1 : index
          %c18 = arith.constant 18 : index
          %c0_5 = arith.constant 0 : index
          %async_token_6, %results_7 = air.execute -> (memref<1024xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<1024xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<1024xbf16, 2 : i32>
          }
          %15 = scf.for %arg6 = %c0_5 to %c18 step %c1_4 iter_args(%arg7 = %async_token_6) -> (!air.async.token) {
            %16 = air.channel.get async [%arg7]  @feed[%arg2, %arg3] (%results_7[] [] []) : (memref<1024xbf16, 2 : i32>)
            scf.yield %16 : !air.async.token
          }
          %async_token_8 = air.execute [%15] {
            memref.dealloc %results_7 : memref<1024xbf16, 2 : i32>
          }
        }
        %12 = scf.for %arg2 = %c0_1 to %c6_0 step %c1_2 iter_args(%arg3 = %async_token) -> (!air.async.token) {
          %15 = air.channel.get async [%arg3]  @chan_a[] (%results[] [] []) : (memref<1024xbf16, 1 : i32>)
          %16 = air.channel.put async [%15]  @feed[%c0_1, %c0_1] (%results[] [] []) : (memref<1024xbf16, 1 : i32>)
          scf.yield %16 : !air.async.token
        }
        %13 = scf.for %arg2 = %c0_1 to %c6_0 step %c1_2 iter_args(%arg3 = %12) -> (!air.async.token) {
          %15 = air.channel.get async [%arg3]  @chan_b[] (%results[] [] []) : (memref<1024xbf16, 1 : i32>)
          %16 = air.channel.put async [%15]  @feed[%c0_1, %c0_1] (%results[] [] []) : (memref<1024xbf16, 1 : i32>)
          scf.yield %16 : !air.async.token
        }
        %14 = scf.for %arg2 = %c0_1 to %c6_0 step %c1_2 iter_args(%arg3 = %13) -> (!air.async.token) {
          %15 = air.channel.get async [%arg3]  @chan_c[] (%results[] [] []) : (memref<1024xbf16, 1 : i32>)
          %16 = air.channel.put async [%15]  @feed[%c0_1, %c0_1] (%results[] [] []) : (memref<1024xbf16, 1 : i32>)
          scf.yield %16 : !air.async.token
        }
        %async_token_3 = air.execute [%14] {
          memref.dealloc %results : memref<1024xbf16, 1 : i32>
        }
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: func.func @fuse_identical_four_way
// CHECK: air.launch
// CHECK: %[[C4:.*]] = arith.constant 4 : index
// CHECK: scf.for %{{.*}} = %{{.*}} to %[[C4]] step
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK: air.channel.put {{.*}} @chan_a
// CHECK-NOT: air.channel.put {{.*}} @chan_b
// CHECK-NOT: air.channel.put {{.*}} @chan_c
// CHECK-NOT: air.channel.put {{.*}} @chan_d
// CHECK: air.segment
module {
  air.channel @feed [1, 1]
  air.channel @chan_a []
  air.channel @chan_b []
  air.channel @chan_c []
  air.channel @chan_d []
  func.func @fuse_identical_four_way(%arg0: memref<1024xbf16>) {
    %0 = air.launch async () in () args(%arg1=%arg0) : memref<1024xbf16> {
      %c6 = arith.constant 6 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %1 = air.wait_all async
      %2 = scf.for %arg2 = %c0 to %c6 step %c1 iter_args(%arg3 = %1) -> (!air.async.token) {
        %12 = air.channel.put async [%arg3]  @chan_a[] (%arg1[] [] []) : (memref<1024xbf16>)
        scf.yield %12 : !air.async.token
      }
      %3 = air.wait_all async
      %4 = scf.for %arg2 = %c0 to %c6 step %c1 iter_args(%arg3 = %3) -> (!air.async.token) {
        %12 = air.channel.put async [%arg3]  @chan_b[] (%arg1[] [] []) : (memref<1024xbf16>)
        scf.yield %12 : !air.async.token
      }
      %5 = air.wait_all async
      %6 = scf.for %arg2 = %c0 to %c6 step %c1 iter_args(%arg3 = %5) -> (!air.async.token) {
        %12 = air.channel.put async [%arg3]  @chan_c[] (%arg1[] [] []) : (memref<1024xbf16>)
        scf.yield %12 : !air.async.token
      }
      %7 = air.wait_all async
      %8 = scf.for %arg2 = %c0 to %c6 step %c1 iter_args(%arg3 = %7) -> (!air.async.token) {
        %12 = air.channel.put async [%arg3]  @chan_d[] (%arg1[] [] []) : (memref<1024xbf16>)
        scf.yield %12 : !air.async.token
      }
      %10 = air.segment @seg async {
        %c6_0 = arith.constant 6 : index
        %c0_1 = arith.constant 0 : index
        %c1_2 = arith.constant 1 : index
        %async_token, %results = air.execute -> (memref<1024xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<1024xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<1024xbf16, 1 : i32>
        }
        %11 = air.herd @consumer async tile (%arg2, %arg3) in (%arg4=%c1_2, %arg5=%c1_2) {
          %c1_4 = arith.constant 1 : index
          %c24 = arith.constant 24 : index
          %c0_5 = arith.constant 0 : index
          %async_token_6, %results_7 = air.execute -> (memref<1024xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<1024xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<1024xbf16, 2 : i32>
          }
          %15 = scf.for %arg6 = %c0_5 to %c24 step %c1_4 iter_args(%arg7 = %async_token_6) -> (!air.async.token) {
            %16 = air.channel.get async [%arg7]  @feed[%arg2, %arg3] (%results_7[] [] []) : (memref<1024xbf16, 2 : i32>)
            scf.yield %16 : !air.async.token
          }
          %async_token_8 = air.execute [%15] {
            memref.dealloc %results_7 : memref<1024xbf16, 2 : i32>
          }
        }
        %12 = scf.for %arg2 = %c0_1 to %c6_0 step %c1_2 iter_args(%arg3 = %async_token) -> (!air.async.token) {
          %15 = air.channel.get async [%arg3]  @chan_a[] (%results[] [] []) : (memref<1024xbf16, 1 : i32>)
          %16 = air.channel.put async [%15]  @feed[%c0_1, %c0_1] (%results[] [] []) : (memref<1024xbf16, 1 : i32>)
          scf.yield %16 : !air.async.token
        }
        %13 = scf.for %arg2 = %c0_1 to %c6_0 step %c1_2 iter_args(%arg3 = %12) -> (!air.async.token) {
          %15 = air.channel.get async [%arg3]  @chan_b[] (%results[] [] []) : (memref<1024xbf16, 1 : i32>)
          %16 = air.channel.put async [%15]  @feed[%c0_1, %c0_1] (%results[] [] []) : (memref<1024xbf16, 1 : i32>)
          scf.yield %16 : !air.async.token
        }
        %14 = scf.for %arg2 = %c0_1 to %c6_0 step %c1_2 iter_args(%arg3 = %13) -> (!air.async.token) {
          %15 = air.channel.get async [%arg3]  @chan_c[] (%results[] [] []) : (memref<1024xbf16, 1 : i32>)
          %16 = air.channel.put async [%15]  @feed[%c0_1, %c0_1] (%results[] [] []) : (memref<1024xbf16, 1 : i32>)
          scf.yield %16 : !air.async.token
        }
        %17 = scf.for %arg2 = %c0_1 to %c6_0 step %c1_2 iter_args(%arg3 = %14) -> (!air.async.token) {
          %15 = air.channel.get async [%arg3]  @chan_d[] (%results[] [] []) : (memref<1024xbf16, 1 : i32>)
          %16 = air.channel.put async [%15]  @feed[%c0_1, %c0_1] (%results[] [] []) : (memref<1024xbf16, 1 : i32>)
          scf.yield %16 : !air.async.token
        }
        %async_token_3 = air.execute [%17] {
          memref.dealloc %results : memref<1024xbf16, 1 : i32>
        }
      }
    }
    return
  }
}
