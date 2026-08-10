//===- fuse_packet_put_loops_decline_warns.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// air-fuse-packet-put-loops: DECLINING IS LOUD. For this pass the
// untransformed program is the broken one (doc 23 "Silence is the wrong
// default"): past one trip, unfused same-bounds packet put loops deliver
// whole channels back to back against a consumer ring built for
// per-iteration interleave -- silently wrong data whenever the channels
// share a packet queue, which is unknowable before air-to-aie. So a decline
// that leaves two or more same-bounds candidates at trip count > 1 warns,
// naming the loops, the channels and the trip count; the one-trip shape and
// different-bounds pairs stay silent, because there the unfused form is
// correct (orders coincide) or no group ever existed.

// RUN: air-opt %s -air-fuse-packet-put-loops -split-input-file -verify-diagnostics

// Fusion declined by the dominance rule: the first loop's token is consumed
// BETWEEN the loops, so rebuilding the group at the last member's position
// would break dominance. Two same-bounds candidates remain at trip count 2:
// warn, with both channels named and the second loop noted.
module {
  air.channel @pw0 [1, 1] {channel_type = "npu_dma_packet"}
  air.channel @pw1 [1, 1] {channel_type = "npu_dma_packet"}
  func.func @dominance_decline_warns(%arg0: memref<8x64xbf16>, %arg1: memref<8x64xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx) in (%sx=%c1) args(%a=%arg0, %b=%arg1) : memref<8x64xbf16>, memref<8x64xbf16> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %t0 = air.wait_all async
      // expected-warning @below {{2 same-bounds packet put loops left unfused at trip count 2 (channels: pw0, pw1)}}
      %l0 = scf.for %i = %c0 to %c8 step %c4 iter_args(%t = %t0) -> (!air.async.token) {
        %p = air.channel.put async [%t] @pw0[] (%a[%i, %c0] [%c4, %c64] [%c64, %c1_l]) : (memref<8x64xbf16>)
        scf.yield %p : !air.async.token
      }
      %mid = air.wait_all async [%l0]
      %t1 = air.wait_all async
      // expected-note @below {{unfused sibling put loop here}}
      %l1 = scf.for %i = %c0 to %c8 step %c4 iter_args(%t = %t1) -> (!air.async.token) {
        %p = air.channel.put async [%t] @pw1[] (%b[%i, %c0] [%c4, %c64] [%c64, %c1_l]) : (memref<8x64xbf16>)
        scf.yield %p : !air.async.token
      }
      %done = air.wait_all async [%l0, %l1]
    }
    return
  }
}

// -----

// A bare packet put between the loops seals the group (fusing across it
// would reorder the shared stream), leaving two singleton groups the fusion
// correctly skips. The hazard is still standing -- two same-bounds put loops
// at trip count 2 -- so the decline still warns.
module {
  air.channel @ps0 [1, 1] {channel_type = "npu_dma_packet"}
  air.channel @ps1 [1, 1] {channel_type = "npu_dma_packet"}
  air.channel @psx [1, 1] {channel_type = "npu_dma_packet"}
  func.func @sealed_groups_warn(%arg0: memref<8x64xbf16>, %arg1: memref<8x64xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx) in (%sx=%c1) args(%a=%arg0, %b=%arg1) : memref<8x64xbf16>, memref<8x64xbf16> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %t0 = air.wait_all async
      // expected-warning @below {{2 same-bounds packet put loops left unfused at trip count 2}}
      %l0 = scf.for %i = %c0 to %c8 step %c4 iter_args(%t = %t0) -> (!air.async.token) {
        %p = air.channel.put async [%t] @ps0[] (%a[%i, %c0] [%c4, %c64] [%c64, %c1_l]) : (memref<8x64xbf16>)
        scf.yield %p : !air.async.token
      }
      %ts = air.wait_all async
      %bp = air.channel.put async [%ts] @psx[] (%a[%c0, %c0] [%c4, %c64] [%c64, %c1_l]) : (memref<8x64xbf16>)
      %t1 = air.wait_all async
      // expected-note @below {{unfused sibling put loop here}}
      %l1 = scf.for %i = %c0 to %c8 step %c4 iter_args(%t = %t1) -> (!air.async.token) {
        %p = air.channel.put async [%t] @ps1[] (%b[%i, %c0] [%c4, %c64] [%c64, %c1_l]) : (memref<8x64xbf16>)
        scf.yield %p : !air.async.token
      }
      %done = air.wait_all async [%l0, %l1, %bp]
    }
    return
  }
}

// -----

// One trip: the whole-channel and per-iteration orders coincide, so the
// unfused form is correct and the same dominance decline stays SILENT. An
// unconditional diagnostic would fire on most shipped designs; this case is
// what keeps it conditional.
module {
  air.channel @po0 [1, 1] {channel_type = "npu_dma_packet"}
  air.channel @po1 [1, 1] {channel_type = "npu_dma_packet"}
  func.func @one_trip_decline_is_silent(%arg0: memref<8x64xbf16>, %arg1: memref<8x64xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx) in (%sx=%c1) args(%a=%arg0, %b=%arg1) : memref<8x64xbf16>, memref<8x64xbf16> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c64 = arith.constant 64 : index
      %t0 = air.wait_all async
      %l0 = scf.for %i = %c0 to %c4 step %c4 iter_args(%t = %t0) -> (!air.async.token) {
        %p = air.channel.put async [%t] @po0[] (%a[%i, %c0] [%c4, %c64] [%c64, %c1_l]) : (memref<8x64xbf16>)
        scf.yield %p : !air.async.token
      }
      %mid = air.wait_all async [%l0]
      %t1 = air.wait_all async
      %l1 = scf.for %i = %c0 to %c4 step %c4 iter_args(%t = %t1) -> (!air.async.token) {
        %p = air.channel.put async [%t] @po1[] (%b[%i, %c0] [%c4, %c64] [%c64, %c1_l]) : (memref<8x64xbf16>)
        scf.yield %p : !air.async.token
      }
      %done = air.wait_all async [%l0, %l1]
    }
    return
  }
}

// -----

// Different bounds never form a group, at any trip count: silent. The
// hazard this pass polices is a same-bounds group's interleave order; two
// loops with different bounds were never candidates for one consumer ring.
module {
  air.channel @pd0 [1, 1] {channel_type = "npu_dma_packet"}
  air.channel @pd1 [1, 1] {channel_type = "npu_dma_packet"}
  func.func @different_bounds_silent(%arg0: memref<8x64xbf16>, %arg1: memref<8x64xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx) in (%sx=%c1) args(%a=%arg0, %b=%arg1) : memref<8x64xbf16>, memref<8x64xbf16> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %t0 = air.wait_all async
      %l0 = scf.for %i = %c0 to %c8 step %c4 iter_args(%t = %t0) -> (!air.async.token) {
        %p = air.channel.put async [%t] @pd0[] (%a[%i, %c0] [%c4, %c64] [%c64, %c1_l]) : (memref<8x64xbf16>)
        scf.yield %p : !air.async.token
      }
      %mid = air.wait_all async [%l0]
      %t1 = air.wait_all async
      %l1 = scf.for %i = %c0 to %c8 step %c2 iter_args(%t = %t1) -> (!air.async.token) {
        %p = air.channel.put async [%t] @pd1[] (%b[%i, %c0] [%c2, %c64] [%c64, %c1_l]) : (memref<8x64xbf16>)
        scf.yield %p : !air.async.token
      }
      %done = air.wait_all async [%l0, %l1]
    }
    return
  }
}
