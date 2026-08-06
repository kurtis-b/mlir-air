//===- ping_pong_shared_resident_ring_mixed.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-label-scf-for-to-ping-pong -air-ping-pong-transform -verify-diagnostics | FileCheck %s --implicit-check-not=hoist_alloc --implicit-check-not=unroll

// Each sibling loop mixes a MARKED get (@inX, air.shared_resident_ring) with
// an UNMARKED get (@inW) -- the shape whose partial ring sharing must never
// happen. Here the consuming compute step is a call to an UNANNOTATED
// external callee, so the read of each per-iteration buffer cannot be
// classified and the H1 safety proof cannot build the reuse edge that would
// protect any rotation at all: both loops must be SKIPPED with a warning and
// keep their correct single-buffered schedule -- no rings, so the
// marked/unmarked mixing question is never reached. The
// per-loop-rings-never-partially-shared coverage this input carried before H2
// changed the unannotated-callee outcome lives in
// ping_pong_shared_resident_ring_mixed_annotated.mlir, identical but for the
// callee argument attributes that make the same loops provable.

// SKIP: the loops stay untransformed -- every alloc remains inside its own
// loop (2 + 2, no hoisted rings), and each loop keeps its original single
// async-token iter arg. The implicit check-nots prove no loop and no alloc
// was labeled.
// CHECK-LABEL: mixed_marked_unmarked
// CHECK: scf.for {{.*}} -> (!air.async.token) {
// CHECK: scf.for {{.*}} -> (!air.async.token) {
// CHECK-COUNT-2: memref.alloc()
// CHECK: scf.for {{.*}} -> (!air.async.token) {
// CHECK-COUNT-2: memref.alloc()
// CHECK-NOT: memref.alloc()

module {
  air.channel @inX [1] {air.shared_resident_ring}
  air.channel @inW [1]
  func.func @mixed_marked_unmarked() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%a, %b) in (%c=%c1, %d=%c1) {
      %1 = air.segment async {
        %c0 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c8 = arith.constant 8 : index
        %c1s = arith.constant 1 : index
        %2 = air.wait_all async
        %3 = scf.for %v1 = %c0 to %c4 step %c1s iter_args(%t = %2) -> (!air.async.token) {
          // expected-warning@+1 {{is a ping-pong candidate that cannot be proven safe to transform}}
          %g0 = scf.for %j = %c0 to %c8 step %c1s iter_args(%tt = %t) -> (!air.async.token) {
            %tx, %bx = air.execute [%tt] -> (memref<256xi8, 2>) {
              %al = memref.alloc() : memref<256xi8, 2>
              air.execute_terminator %al : memref<256xi8, 2>
            }
            %gx = air.channel.get async [%tx] @inX[] (%bx[] [] []) : (memref<256xi8, 2>)
            %tw, %bw = air.execute [%tt] -> (memref<2560xi8, 2>) {
              %al = memref.alloc() : memref<2560xi8, 2>
              air.execute_terminator %al : memref<2560xi8, 2>
            }
            %gw = air.channel.get async [%tw] @inW[] (%bw[] [] []) : (memref<2560xi8, 2>)
            %cc = air.execute [%gx, %gw] {
              func.call @acc(%bx, %bw) : (memref<256xi8, 2>, memref<2560xi8, 2>) -> ()
            }
            %dx = air.execute [%cc] { memref.dealloc %bx : memref<256xi8, 2> }
            %dw = air.execute [%cc] { memref.dealloc %bw : memref<2560xi8, 2> }
            %w = air.wait_all async [%dx, %dw]
            scf.yield %w : !air.async.token
          }
          // expected-warning@+1 {{is a ping-pong candidate that cannot be proven safe to transform}}
          %g1 = scf.for %j = %c0 to %c8 step %c1s iter_args(%tt = %g0) -> (!air.async.token) {
            %tx, %bx = air.execute [%tt] -> (memref<256xi8, 2>) {
              %al = memref.alloc() : memref<256xi8, 2>
              air.execute_terminator %al : memref<256xi8, 2>
            }
            %gx = air.channel.get async [%tx] @inX[] (%bx[] [] []) : (memref<256xi8, 2>)
            %tw, %bw = air.execute [%tt] -> (memref<2560xi8, 2>) {
              %al = memref.alloc() : memref<2560xi8, 2>
              air.execute_terminator %al : memref<2560xi8, 2>
            }
            %gw = air.channel.get async [%tw] @inW[] (%bw[] [] []) : (memref<2560xi8, 2>)
            %cc = air.execute [%gx, %gw] {
              func.call @acc(%bx, %bw) : (memref<256xi8, 2>, memref<2560xi8, 2>) -> ()
            }
            %dx = air.execute [%cc] { memref.dealloc %bx : memref<256xi8, 2> }
            %dw = air.execute [%cc] { memref.dealloc %bw : memref<2560xi8, 2> }
            %w = air.wait_all async [%dx, %dw]
            scf.yield %w : !air.async.token
          }
          scf.yield %g1 : !air.async.token
        }
      }
    }
    return
  }
  func.func private @acc(%a: memref<256xi8, 2>, %b: memref<2560xi8, 2>)
}
