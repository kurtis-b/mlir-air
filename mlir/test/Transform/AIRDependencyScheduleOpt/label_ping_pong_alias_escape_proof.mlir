//===- label_ping_pong_alias_escape_proof.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// The safety proof must follow a duplicated buffer through EVERY memref
// value derived from it, not only ViewLikeOpInterface results and
// air.execute yields. A memory-effect-free forwarding op the alias map
// did not model (arith.select is the canonical case) used to launder the
// buffer into a fresh SSA value: the reader behind it vanished from the
// proof, the buffer was declared vacuously safe (no consumer), and the
// transform could overwrite a half while the kernel was still reading it.
// A pure region-free op cannot allocate, so a memref result provably
// forwards one of its memref operands: when every memref operand aliases
// the SAME candidate alloc the result is tracked as that buffer; anything
// the map cannot attribute to exactly one buffer (arms aliasing different
// allocs, an arm the map does not track, a loop-like op recirculating the
// buffer, a call returning a memref) makes the loop unprovable and it is
// skipped, never labeled.

// RUN: air-opt %s -air-label-scf-for-to-ping-pong -split-input-file -verify-diagnostics | FileCheck %s

// The reader reaches the buffer only through arith.select %b, %b. The
// alias is unambiguous, so the proof sees the consumer, the per-iteration
// channel.get is its producer, and the loop is provably safe: labeled.
// CHECK-LABEL: func.func @select_alias_reader_with_refill
// CHECK: scf.for
// CHECK: hoist_alloc
// CHECK: } {unroll = 2 : i32}
module {
  air.channel @cf1 [1, 1]
  func.func private @knl_ro(memref<64xbf16, 2> {llvm.readonly})
      attributes {llvm.emit_c_interface}
  func.func @select_alias_reader_with_refill() {
    %c1 = arith.constant 1 : index
    %0 = air.herd @h async tile (%tx, %ty) in (%sx=%c1, %sy=%c1) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %t0 = air.wait_all async
      %1 = scf.for %i = %c0 to %c8 step %c4 iter_args(%t = %t0) -> (!air.async.token) {
        %tb, %bb = air.execute -> (memref<64xbf16, 2>) {
          %a = memref.alloc() : memref<64xbf16, 2>
          air.execute_terminator %a : memref<64xbf16, 2>
        }
        %g = air.channel.get async [%tb, %t] @cf1[%tx, %ty] (%bb[] [] []) : (memref<64xbf16, 2>)
        %cond = arith.cmpi eq, %i, %c0 : index
        %alias = arith.select %cond, %bb, %bb : memref<64xbf16, 2>
        %tc = air.execute [%g] {
          func.call @knl_ro(%alias) : (memref<64xbf16, 2>) -> ()
        }
        %d = air.execute [%tc] { memref.dealloc %bb : memref<64xbf16, 2> }
        scf.yield %d : !air.async.token
      }
    }
    return
  }
}

// -----

// The same select-laundered reader, but nothing refills the buffer inside
// the loop. Before alias-following, the reader was invisible and this
// loop was labeled as vacuously safe -- the exact laundering that lets
// ping-pong rebuild the dependency graph without the kernel's read and
// overwrite a buffer half still in use. With the consumer visible and no
// per-iteration producer, the loop is unprovable: SKIPPED with a warning,
// left on its correct single-buffered schedule, never labeled. (This used
// to be a hard compile error; the prior art the proof cites -- upstream
// memref::multiBuffer, IREE, Triton, TVM -- declines the transform, not
// the build.)
// CHECK-LABEL: func.func @select_alias_reader_no_refill
// CHECK-NOT: hoist_alloc
// CHECK-NOT: unroll
module {
  func.func private @knl_ro2(memref<64xbf16, 2> {llvm.readonly})
      attributes {llvm.emit_c_interface}
  func.func @select_alias_reader_no_refill() {
    %c1 = arith.constant 1 : index
    %0 = air.herd @h async tile (%tx, %ty) in (%sx=%c1, %sy=%c1) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %t0 = air.wait_all async
      // expected-warning@+1 {{is a ping-pong candidate that cannot be proven safe to transform}}
      %1 = scf.for %i = %c0 to %c8 step %c4 iter_args(%t = %t0) -> (!air.async.token) {
        %tb, %bb = air.execute -> (memref<64xbf16, 2>) {
          %a = memref.alloc() : memref<64xbf16, 2>
          air.execute_terminator %a : memref<64xbf16, 2>
        }
        %cond = arith.cmpi eq, %i, %c0 : index
        %alias = arith.select %cond, %bb, %bb : memref<64xbf16, 2>
        %tc = air.execute [%tb] {
          func.call @knl_ro2(%alias) : (memref<64xbf16, 2>) -> ()
        }
        %d = air.execute [%tc] { memref.dealloc %bb : memref<64xbf16, 2> }
        scf.yield %d : !air.async.token
      }
    }
    return
  }
}

// -----

// A select whose arms alias two DIFFERENT duplicated buffers cannot be
// attributed to one buffer, so the derived value is untrackable and the
// loop is unprovable: skipped with a warning, never labeled -- even
// though both buffers are refilled every iteration and would otherwise
// have looked (vacuously) safe.
// CHECK-LABEL: func.func @select_alias_two_buffers
// CHECK-NOT: hoist_alloc
// CHECK-NOT: unroll
module {
  air.channel @cf2 [1, 1]
  air.channel @cf3 [1, 1]
  func.func private @knl_ro3(memref<64xbf16, 2> {llvm.readonly})
      attributes {llvm.emit_c_interface}
  func.func @select_alias_two_buffers() {
    %c1 = arith.constant 1 : index
    %0 = air.herd @h async tile (%tx, %ty) in (%sx=%c1, %sy=%c1) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %t0 = air.wait_all async
      // expected-warning@+1 {{is a ping-pong candidate that cannot be proven safe to transform}}
      %1 = scf.for %i = %c0 to %c8 step %c4 iter_args(%t = %t0) -> (!air.async.token) {
        %ta, %ba = air.execute -> (memref<64xbf16, 2>) {
          %a = memref.alloc() : memref<64xbf16, 2>
          air.execute_terminator %a : memref<64xbf16, 2>
        }
        %tb, %bb = air.execute -> (memref<64xbf16, 2>) {
          %a = memref.alloc() : memref<64xbf16, 2>
          air.execute_terminator %a : memref<64xbf16, 2>
        }
        %ga = air.channel.get async [%ta, %t] @cf2[%tx, %ty] (%ba[] [] []) : (memref<64xbf16, 2>)
        %gb = air.channel.get async [%tb, %t] @cf3[%tx, %ty] (%bb[] [] []) : (memref<64xbf16, 2>)
        %cond = arith.cmpi eq, %i, %c0 : index
        %alias = arith.select %cond, %ba, %bb : memref<64xbf16, 2>
        %tc = air.execute [%ga, %gb] {
          func.call @knl_ro3(%alias) : (memref<64xbf16, 2>) -> ()
        }
        %da = air.execute [%tc] { memref.dealloc %ba : memref<64xbf16, 2> }
        %db = air.execute [%tc] { memref.dealloc %bb : memref<64xbf16, 2> }
        %w = air.wait_all async [%da, %db]
        scf.yield %w : !air.async.token
      }
    }
    return
  }
}
