//===- fuse_packet_put_loops.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// air-fuse-packet-put-loops: sibling launch-scope put loops feeding
// packet-multiplexed channels fuse into ONE loop whose body performs the puts
// in original program order, chained by async tokens. The consuming tile
// ring expects the streams interleaved per iteration; the per-channel loops
// air-dma-to-channel hoists deliver them whole-channel-after-whole-channel
// instead, which misdelivers every packet after the first trip (measured on
// NPU2: 481/512 elements wrong at two trips). Also locks the contract that
// the interleaving chain SURVIVES -canonicalize: CanonicalizeAsyncOpDeps
// models packet-typed channels as one shared stream resource, so the edge
// between puts on two different packet channels is never pruned as
// "no shared resource".

// RUN: air-opt %s -air-fuse-packet-put-loops -split-input-file | FileCheck %s
// RUN: air-opt %s -air-fuse-packet-put-loops -canonicalize -split-input-file | FileCheck %s --check-prefix=CANON

// Two sibling packet put loops, same static bounds: fused into one loop, puts
// token-chained in program order (@pk0 before @pk1, the herd's per-iteration
// get order). The fused loop's incoming token joins both source inits.
// CHECK-LABEL: @two_packet_loops
// CHECK: %[[JOIN:.*]] = air.wait_all async [%{{.*}}, %{{.*}}]
// CHECK: scf.for {{.*}} iter_args(%[[T:.*]] = %[[JOIN]])
// CHECK: %[[P0:.*]] = air.channel.put async [%[[T]]] @pk0
// CHECK: %[[P1:.*]] = air.channel.put async [%[[P0]]] @pk1
// CHECK: scf.yield %[[P1]]
// CHECK-NOT: scf.for

// The chain survives canonicalization: @pk1's put still waits on @pk0's.
// CANON-LABEL: @two_packet_loops
// CANON: %[[P0:.*]] = air.channel.put async {{.*}} @pk0
// CANON: air.channel.put async [%[[P0]]] @pk1
module {
  air.channel @pk0 [1, 1] {channel_type = "npu_dma_packet"}
  air.channel @pk1 [1, 1] {channel_type = "npu_dma_packet"}
  func.func @two_packet_loops(%arg0: memref<8x64xbf16>, %arg1: memref<8x64xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx) in (%sx=%c1) args(%a=%arg0, %b=%arg1) : memref<8x64xbf16>, memref<8x64xbf16> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %t0 = air.wait_all async
      %l0 = scf.for %i = %c0 to %c8 step %c4 iter_args(%t = %t0) -> (!air.async.token) {
        %p = air.channel.put async [%t] @pk0[] (%a[%i, %c0] [%c4, %c64] [%c64, %c1_l]) : (memref<8x64xbf16>)
        scf.yield %p : !air.async.token
      }
      %t1 = air.wait_all async
      %l1 = scf.for %i = %c0 to %c8 step %c4 iter_args(%t = %t1) -> (!air.async.token) {
        %p = air.channel.put async [%t] @pk1[] (%b[%i, %c0] [%c4, %c64] [%c64, %c1_l]) : (memref<8x64xbf16>)
        scf.yield %p : !air.async.token
      }
      %done = air.wait_all async [%l0, %l1]
    }
    return
  }
}

// -----

// Different static bounds land in different groups: nothing fuses.
// CHECK-LABEL: @different_bounds_not_fused
// CHECK: scf.for
// CHECK: @pk2
// CHECK: scf.for
// CHECK: @pk3
module {
  air.channel @pk2 [1, 1] {channel_type = "npu_dma_packet"}
  air.channel @pk3 [1, 1] {channel_type = "npu_dma_packet"}
  func.func @different_bounds_not_fused(%arg0: memref<8x64xbf16>, %arg1: memref<8x64xbf16>) {
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
        %p = air.channel.put async [%t] @pk2[] (%a[%i, %c0] [%c4, %c64] [%c64, %c1_l]) : (memref<8x64xbf16>)
        scf.yield %p : !air.async.token
      }
      %t1 = air.wait_all async
      %l1 = scf.for %i = %c0 to %c8 step %c2 iter_args(%t = %t1) -> (!air.async.token) {
        %p = air.channel.put async [%t] @pk3[] (%b[%i, %c0] [%c2, %c64] [%c64, %c1_l]) : (memref<8x64xbf16>)
        scf.yield %p : !air.async.token
      }
      %done = air.wait_all async [%l0, %l1]
    }
    return
  }
}

// -----

// Circuit-routed (non-packet) channels do not share a shim queue, so their
// loops are not candidates: nothing fuses.
// CHECK-LABEL: @non_packet_not_fused
// CHECK: scf.for
// CHECK: @cir0
// CHECK: scf.for
// CHECK: @cir1
module {
  air.channel @cir0 [1, 1]
  air.channel @cir1 [1, 1]
  func.func @non_packet_not_fused(%arg0: memref<8x64xbf16>, %arg1: memref<8x64xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx) in (%sx=%c1) args(%a=%arg0, %b=%arg1) : memref<8x64xbf16>, memref<8x64xbf16> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %t0 = air.wait_all async
      %l0 = scf.for %i = %c0 to %c8 step %c4 iter_args(%t = %t0) -> (!air.async.token) {
        %p = air.channel.put async [%t] @cir0[] (%a[%i, %c0] [%c4, %c64] [%c64, %c1_l]) : (memref<8x64xbf16>)
        scf.yield %p : !air.async.token
      }
      %t1 = air.wait_all async
      %l1 = scf.for %i = %c0 to %c8 step %c4 iter_args(%t = %t1) -> (!air.async.token) {
        %p = air.channel.put async [%t] @cir1[] (%b[%i, %c0] [%c4, %c64] [%c64, %c1_l]) : (memref<8x64xbf16>)
        scf.yield %p : !air.async.token
      }
      %done = air.wait_all async [%l0, %l1]
    }
    return
  }
}

// -----

// Dominance: the first loop's result is consumed BEFORE the last group
// member, so building the fused loop at the last member's position would
// break dominance for that user. The group must be left alone.
// CHECK-LABEL: @early_user_blocks_fusion
// CHECK: scf.for
// CHECK: @pk4
// CHECK: air.wait_all async
// CHECK: scf.for
// CHECK: @pk5
module {
  air.channel @pk4 [1, 1] {channel_type = "npu_dma_packet"}
  air.channel @pk5 [1, 1] {channel_type = "npu_dma_packet"}
  func.func @early_user_blocks_fusion(%arg0: memref<8x64xbf16>, %arg1: memref<8x64xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx) in (%sx=%c1) args(%a=%arg0, %b=%arg1) : memref<8x64xbf16>, memref<8x64xbf16> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %t0 = air.wait_all async
      %l0 = scf.for %i = %c0 to %c8 step %c4 iter_args(%t = %t0) -> (!air.async.token) {
        %p = air.channel.put async [%t] @pk4[] (%a[%i, %c0] [%c4, %c64] [%c64, %c1_l]) : (memref<8x64xbf16>)
        scf.yield %p : !air.async.token
      }
      %early = air.wait_all async [%l0]
      %l1 = scf.for %i = %c0 to %c8 step %c4 iter_args(%t = %early) -> (!air.async.token) {
        %p = air.channel.put async [%t] @pk5[] (%b[%i, %c0] [%c4, %c64] [%c64, %c1_l]) : (memref<8x64xbf16>)
        scf.yield %p : !air.async.token
      }
      %done = air.wait_all async [%l1]
    }
    return
  }
}

// -----

// Same-bounds packet loops @pkA and @pkC separated by packet loop @pkB with
// different bounds. All three feed the one shared packet stream, so fusing A
// and C at C's position would emit B's packets before A's, changing the
// stream order from A0,B0,C0,A1,C1 to B0,A0,C0,A1,C1. An intervening op that
// occupies the packet stream seals the group: nothing fuses, and every put
// keeps its program-order position.
// CHECK-LABEL: @intervening_packet_loop_seals_group
// CHECK: scf.for
// CHECK: @pkA
// CHECK: scf.for
// CHECK: @pkB
// CHECK: scf.for
// CHECK: @pkC
module {
  air.channel @pkA [1, 1] {channel_type = "npu_dma_packet"}
  air.channel @pkB [1, 1] {channel_type = "npu_dma_packet"}
  air.channel @pkC [1, 1] {channel_type = "npu_dma_packet"}
  func.func @intervening_packet_loop_seals_group(%arg0: memref<8x64xbf16>, %arg1: memref<8x64xbf16>, %arg2: memref<8x64xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx) in (%sx=%c1) args(%a=%arg0, %b=%arg1, %c=%arg2) : memref<8x64xbf16>, memref<8x64xbf16>, memref<8x64xbf16> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %t0 = air.wait_all async
      %l0 = scf.for %i = %c0 to %c8 step %c4 iter_args(%t = %t0) -> (!air.async.token) {
        %p = air.channel.put async [%t] @pkA[] (%a[%i, %c0] [%c4, %c64] [%c64, %c1_l]) : (memref<8x64xbf16>)
        scf.yield %p : !air.async.token
      }
      %t1 = air.wait_all async
      %l1 = scf.for %i = %c0 to %c8 step %c2 iter_args(%t = %t1) -> (!air.async.token) {
        %p = air.channel.put async [%t] @pkB[] (%b[%i, %c0] [%c2, %c64] [%c64, %c1_l]) : (memref<8x64xbf16>)
        scf.yield %p : !air.async.token
      }
      %t2 = air.wait_all async
      %l2 = scf.for %i = %c0 to %c8 step %c4 iter_args(%t = %t2) -> (!air.async.token) {
        %p = air.channel.put async [%t] @pkC[] (%c[%i, %c0] [%c4, %c64] [%c64, %c1_l]) : (memref<8x64xbf16>)
        scf.yield %p : !air.async.token
      }
      %done = air.wait_all async [%l0, %l1, %l2]
    }
    return
  }
}

// -----

// A bare (non-loop) packet put between two same-bounds packet loops also
// occupies the shared stream, and fusing across it would move the first
// loop's packets after it. The group is sealed: nothing fuses.
// CHECK-LABEL: @bare_packet_put_seals_group
// CHECK: scf.for
// CHECK: @pkD
// CHECK: air.channel.put {{.*}} @pkE
// CHECK: scf.for
// CHECK: @pkF
module {
  air.channel @pkD [1, 1] {channel_type = "npu_dma_packet"}
  air.channel @pkE [1, 1] {channel_type = "npu_dma_packet"}
  air.channel @pkF [1, 1] {channel_type = "npu_dma_packet"}
  func.func @bare_packet_put_seals_group(%arg0: memref<8x64xbf16>, %arg1: memref<8x64xbf16>, %arg2: memref<8x64xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx) in (%sx=%c1) args(%a=%arg0, %b=%arg1, %c=%arg2) : memref<8x64xbf16>, memref<8x64xbf16>, memref<8x64xbf16> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %t0 = air.wait_all async
      %l0 = scf.for %i = %c0 to %c8 step %c4 iter_args(%t = %t0) -> (!air.async.token) {
        %p = air.channel.put async [%t] @pkD[] (%a[%i, %c0] [%c4, %c64] [%c64, %c1_l]) : (memref<8x64xbf16>)
        scf.yield %p : !air.async.token
      }
      %t1 = air.wait_all async
      %p1 = air.channel.put async [%t1] @pkE[] (%b[%c0, %c0] [%c8, %c64] [%c64, %c1_l]) : (memref<8x64xbf16>)
      %t2 = air.wait_all async
      %l2 = scf.for %i = %c0 to %c8 step %c4 iter_args(%t = %t2) -> (!air.async.token) {
        %p = air.channel.put async [%t] @pkF[] (%c[%i, %c0] [%c4, %c64] [%c64, %c1_l]) : (memref<8x64xbf16>)
        scf.yield %p : !air.async.token
      }
      %done = air.wait_all async [%l0, %p1, %l2]
    }
    return
  }
}

// -----

// A circuit-routed put between the two loops does NOT occupy the packet
// stream, so it does not seal the group: the packet loops still fuse, and
// the circuit put keeps its own position and ordering.
// CHECK-LABEL: @intervening_circuit_put_does_not_seal
// CHECK: air.channel.put {{.*}} @cirX
// CHECK: scf.for
// CHECK: @pkG
// CHECK: @pkH
// CHECK-NOT: scf.for
module {
  air.channel @pkG [1, 1] {channel_type = "npu_dma_packet"}
  air.channel @pkH [1, 1] {channel_type = "npu_dma_packet"}
  air.channel @cirX [1, 1]
  func.func @intervening_circuit_put_does_not_seal(%arg0: memref<8x64xbf16>, %arg1: memref<8x64xbf16>, %arg2: memref<8x64xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx) in (%sx=%c1) args(%a=%arg0, %b=%arg1, %c=%arg2) : memref<8x64xbf16>, memref<8x64xbf16>, memref<8x64xbf16> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %t0 = air.wait_all async
      %l0 = scf.for %i = %c0 to %c8 step %c4 iter_args(%t = %t0) -> (!air.async.token) {
        %p = air.channel.put async [%t] @pkG[] (%a[%i, %c0] [%c4, %c64] [%c64, %c1_l]) : (memref<8x64xbf16>)
        scf.yield %p : !air.async.token
      }
      %t1 = air.wait_all async
      %p1 = air.channel.put async [%t1] @cirX[] (%b[%c0, %c0] [%c8, %c64] [%c64, %c1_l]) : (memref<8x64xbf16>)
      %t2 = air.wait_all async
      %l2 = scf.for %i = %c0 to %c8 step %c4 iter_args(%t = %t2) -> (!air.async.token) {
        %p = air.channel.put async [%t] @pkH[] (%c[%i, %c0] [%c4, %c64] [%c64, %c1_l]) : (memref<8x64xbf16>)
        scf.yield %p : !air.async.token
      }
      %done = air.wait_all async [%l0, %p1, %l2]
    }
    return
  }
}

// -----

// A put that does not chain from the loop's iter token cannot be
// order-threaded by fusion: remapping the (unused) iter arg to the previous
// put's token would add nothing, leaving the fused puts unchained and the
// packet order whole-channel. Such a loop is not the shape
// air-dma-to-channel emits and is not a candidate: nothing fuses.
// CHECK-LABEL: @unchained_put_not_fused
// CHECK: scf.for
// CHECK: @pk6
// CHECK: scf.for
// CHECK: @pk7
module {
  air.channel @pk6 [1, 1] {channel_type = "npu_dma_packet"}
  air.channel @pk7 [1, 1] {channel_type = "npu_dma_packet"}
  func.func @unchained_put_not_fused(%arg0: memref<8x64xbf16>, %arg1: memref<8x64xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx) in (%sx=%c1) args(%a=%arg0, %b=%arg1) : memref<8x64xbf16>, memref<8x64xbf16> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %t0 = air.wait_all async
      %l0 = scf.for %i = %c0 to %c8 step %c4 iter_args(%t = %t0) -> (!air.async.token) {
        %p = air.channel.put async [%t] @pk6[] (%a[%i, %c0] [%c4, %c64] [%c64, %c1_l]) : (memref<8x64xbf16>)
        scf.yield %p : !air.async.token
      }
      %t1 = air.wait_all async
      %l1 = scf.for %i = %c0 to %c8 step %c4 iter_args(%t = %t1) -> (!air.async.token) {
        %p = air.channel.put async [%t1] @pk7[] (%b[%i, %c0] [%c4, %c64] [%c64, %c1_l]) : (memref<8x64xbf16>)
        scf.yield %p : !air.async.token
      }
      %done = air.wait_all async [%l0, %l1]
    }
    return
  }
}
