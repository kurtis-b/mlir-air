//===- label_ping_pong_external_call_proof.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// H1+H2: a ping-pong candidate whose compute is an external kernel call
// (llvm.emit_c_interface) is PROVEN safe when the callee's argument
// attributes (llvm.readonly / llvm.writeonly) classify every use of every
// buffer the rotation would duplicate, each duplicated buffer that is read
// has a per-iteration producer, and is SKIPPED with a warning when a
// duplicated buffer's use is unannotated (read-versus-write is not
// established, so the loop keeps its correct untransformed schedule). The
// proof's subject is EXACTLY the buffers the rotation privatizes: a call
// touching any other memref -- a herd-argument or call-zero-filled
// accumulator, a weight buffer alloc'd and filled outside the loop,
// untouched scratch -- must NOT block the transform, because that buffer
// stays one physical buffer and every edge ordering it survives the
// rebuild. Two prior drafts drew that line wider and both broke shipped
// designs: refusing on any unclassifiable use failed 8 of 24
// transformer-layer hardware tests, and refusing on unprivatized memrefs
// whose data carries across iterations failed 3 of 10 shipped LLM
// deployments. Silence on an unprovable DUPLICATED buffer was a measured
// 481/512-wrong miscompile; that is what Skip (and Refuse, for a read
// with no per-iteration producer) still guards.

// RUN: air-opt %s -air-label-scf-for-to-ping-pong -split-input-file -verify-diagnostics | FileCheck %s

// All call operands are annotated read-only and refilled inside the loop:
// provable, labeled.
// CHECK: scf.for
// CHECK: hoist_alloc
// CHECK: } {unroll = 2 : i32}
module {
  air.channel @cw [1, 1]
  air.channel @cx [1, 1]
  func.func private @knl(memref<4x64xbf16, 2> {llvm.readonly},
                         memref<64xbf16, 2> {llvm.readonly}, i32)
      attributes {llvm.emit_c_interface}
  func.func @inside() {
    %c1 = arith.constant 1 : index
    %0 = air.herd @h async tile (%tx, %ty) in (%sx=%c1, %sy=%c1) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c64_i32 = arith.constant 64 : i32
      %t0 = air.wait_all async
      %1 = scf.for %i = %c0 to %c8 step %c4 iter_args(%t = %t0) -> (!air.async.token) {
        %tw, %bw = air.execute -> (memref<64xbf16, 2>) {
          %a = memref.alloc() : memref<64xbf16, 2>
          air.execute_terminator %a : memref<64xbf16, 2>
        }
        %tx2, %bx = air.execute -> (memref<4x64xbf16, 2>) {
          %a = memref.alloc() : memref<4x64xbf16, 2>
          air.execute_terminator %a : memref<4x64xbf16, 2>
        }
        %gw = air.channel.get async [%tw, %t] @cw[%tx, %ty] (%bw[] [] []) : (memref<64xbf16, 2>)
        %gx = air.channel.get async [%tx2, %t] @cx[%tx, %ty] (%bx[] [] []) : (memref<4x64xbf16, 2>)
        %tc = air.execute [%gw, %gx] {
          func.call @knl(%bx, %bw, %c64_i32) : (memref<4x64xbf16, 2>, memref<64xbf16, 2>, i32) -> ()
        }
        %dw = air.execute [%tc] { memref.dealloc %bw : memref<64xbf16, 2> }
        %dx = air.execute [%tc] { memref.dealloc %bx : memref<4x64xbf16, 2> }
        %w = air.wait_all async [%dw, %dx]
        scf.yield %w : !air.async.token
      }
    }
    return
  }
}

// -----

// The weight buffer is filled ONCE outside the loop and read by the call
// inside it. The rotation does not privatize that buffer -- it is not one
// of the loop's own per-iteration allocs -- so its unclassified use must
// not, by itself, block anything: single-buffered execution of this loop
// is correct, and an earlier draft that hard-refused this shape also
// refused three shipped LLM deployments that read a loop-invariant buffer
// the same way. The loop IS skipped (warning), because its own candidate
// alloc %bx is read by the same unannotated call.
// CHECK-LABEL: func.func @hoisted
// CHECK-NOT: hoist_alloc
// CHECK-NOT: unroll
module {
  air.channel @cw2 [1, 1]
  air.channel @cx2 [1, 1]
  func.func private @knl2(memref<4x64xbf16, 2>, memref<64xbf16, 2>, i32)
      attributes {llvm.emit_c_interface}
  func.func @hoisted() {
    %c1 = arith.constant 1 : index
    %0 = air.herd @h async tile (%tx, %ty) in (%sx=%c1, %sy=%c1) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c64_i32 = arith.constant 64 : i32
      %tw, %bw = air.execute -> (memref<64xbf16, 2>) {
        %a = memref.alloc() : memref<64xbf16, 2>
        air.execute_terminator %a : memref<64xbf16, 2>
      }
      %gw = air.channel.get async [%tw] @cw2[%tx, %ty] (%bw[] [] []) : (memref<64xbf16, 2>)
      // expected-warning@+1 {{is a ping-pong candidate that cannot be proven safe to transform}}
      %1 = scf.for %i = %c0 to %c8 step %c4 iter_args(%t = %gw) -> (!air.async.token) {
        %tx2, %bx = air.execute -> (memref<4x64xbf16, 2>) {
          %a = memref.alloc() : memref<4x64xbf16, 2>
          air.execute_terminator %a : memref<4x64xbf16, 2>
        }
        %gx = air.channel.get async [%tx2, %t] @cx2[%tx, %ty] (%bx[] [] []) : (memref<4x64xbf16, 2>)
        %tc = air.execute [%gx] {
          func.call @knl2(%bx, %bw, %c64_i32) : (memref<4x64xbf16, 2>, memref<64xbf16, 2>, i32) -> ()
        }
        %dx = air.execute [%tc] { memref.dealloc %bx : memref<4x64xbf16, 2> }
        scf.yield %dx : !air.async.token
      }
      %td = air.execute [%1] { memref.dealloc %bw : memref<64xbf16, 2> }
    }
    return
  }
}

// -----

// The same legitimate per-iteration shape as the first case, but the callee's
// memref arguments carry NO llvm.readonly/llvm.writeonly attributes.
// Read-versus-write is not established, so the loop cannot be proven safe;
// it is left untransformed (which is correct) with a warning, never labeled.
// CHECK-LABEL: func.func @unannotated_inside
// CHECK-NOT: hoist_alloc
// CHECK-NOT: unroll
module {
  air.channel @cw3 [1, 1]
  air.channel @cx3 [1, 1]
  func.func private @knl3(memref<4x64xbf16, 2>, memref<64xbf16, 2>, i32)
      attributes {llvm.emit_c_interface}
  func.func @unannotated_inside() {
    %c1 = arith.constant 1 : index
    %0 = air.herd @h async tile (%tx, %ty) in (%sx=%c1, %sy=%c1) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c64_i32 = arith.constant 64 : i32
      %t0 = air.wait_all async
      // expected-warning@+1 {{is a ping-pong candidate that cannot be proven safe to transform}}
      %1 = scf.for %i = %c0 to %c8 step %c4 iter_args(%t = %t0) -> (!air.async.token) {
        %tw, %bw = air.execute -> (memref<64xbf16, 2>) {
          %a = memref.alloc() : memref<64xbf16, 2>
          air.execute_terminator %a : memref<64xbf16, 2>
        }
        %tx2, %bx = air.execute -> (memref<4x64xbf16, 2>) {
          %a = memref.alloc() : memref<4x64xbf16, 2>
          air.execute_terminator %a : memref<4x64xbf16, 2>
        }
        %gw = air.channel.get async [%tw, %t] @cw3[%tx, %ty] (%bw[] [] []) : (memref<64xbf16, 2>)
        %gx = air.channel.get async [%tx2, %t] @cx3[%tx, %ty] (%bx[] [] []) : (memref<4x64xbf16, 2>)
        %tc = air.execute [%gw, %gx] {
          func.call @knl3(%bx, %bw, %c64_i32) : (memref<4x64xbf16, 2>, memref<64xbf16, 2>, i32) -> ()
        }
        %dw = air.execute [%tc] { memref.dealloc %bw : memref<64xbf16, 2> }
        %dx = air.execute [%tc] { memref.dealloc %bx : memref<4x64xbf16, 2> }
        %w = air.wait_all async [%dw, %dx]
        scf.yield %w : !air.async.token
      }
    }
    return
  }
}

// -----

// The shipped elementwise / causal-mask shape: compute reads the input
// buffer and writes the output buffer THROUGH memref.subview aliases inside
// a nested static-trip loop. The alias-aware proof must see the
// transfer_read as the input's consumer and the transfer_write as the
// output's per-iteration producer, and label the loop; missing the alias
// made this working design refuse to compile.
// CHECK-LABEL: func.func @subview_alias_compute
// CHECK: scf.for
// CHECK: hoist_alloc
// CHECK: } {unroll = 2 : i32}
module {
  air.channel @cin [1, 1]
  air.channel @cout [1, 1]
  func.func @subview_alias_compute() {
    %c1 = arith.constant 1 : index
    %0 = air.herd @h async tile (%tx, %ty) in (%sx=%c1, %sy=%c1) {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c64 = arith.constant 64 : index
      %c512 = arith.constant 512 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %t0 = air.wait_all async
      %1 = scf.for %i = %c0 to %c512 step %c64 iter_args(%t = %t0) -> (!air.async.token) {
        %ta, %ba = air.execute -> (memref<64xbf16, 2>) {
          %a = memref.alloc() : memref<64xbf16, 2>
          air.execute_terminator %a : memref<64xbf16, 2>
        }
        %to, %bo = air.execute -> (memref<64xbf16, 2>) {
          %a = memref.alloc() : memref<64xbf16, 2>
          air.execute_terminator %a : memref<64xbf16, 2>
        }
        %g = air.channel.get async [%ta, %t] @cin[%tx, %ty] (%ba[] [] []) : (memref<64xbf16, 2>)
        %2 = scf.for %j = %c0 to %c64 step %c16 iter_args(%tt = %g) -> (!air.async.token) {
          %sva = memref.subview %ba[%j] [16] [1] : memref<64xbf16, 2> to memref<16xbf16, strided<[1], offset: ?>, 2>
          %svo = memref.subview %bo[%j] [16] [1] : memref<64xbf16, 2> to memref<16xbf16, strided<[1], offset: ?>, 2>
          %tr, %v = air.execute [%tt] -> (vector<16xbf16>) {
            %r = vector.transfer_read %sva[%c0], %cst {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2>, vector<16xbf16>
            air.execute_terminator %r : vector<16xbf16>
          }
          %tw = air.execute [%tr, %to] {
            vector.transfer_write %v, %svo[%c0] {in_bounds = [true]} : vector<16xbf16>, memref<16xbf16, strided<[1], offset: ?>, 2>
          }
          scf.yield %tw : !air.async.token
        }
        %p = air.channel.put async [%2] @cout[%tx, %ty] (%bo[] [] []) : (memref<64xbf16, 2>)
        %da = air.execute [%2] { memref.dealloc %ba : memref<64xbf16, 2> }
        %do = air.execute [%p] { memref.dealloc %bo : memref<64xbf16, 2> }
        %w = air.wait_all async [%da, %do]
        scf.yield %w : !air.async.token
      }
    }
    return
  }
}

// -----

// The refusal is SCOPED to buffers whose data provably carries INTO the
// loop through a classified out-of-loop fill. Here the call also
// accumulates into a buffer alloc'd outside the loop -- zero-filled by
// another unclassified kernel call, drained by a channel.put after the
// loop. No classified WRITE fills it outside the loop, so there is no
// producer edge the rebuilt token graph could sever: the loop must be
// skipped (its own candidates are unannotated), never refused. Attempt 1
// hard-refused this exact shape and broke every GEMM-bearing operator in
// the transformer-layer suite (8 of 24 hardware tests).
// CHECK-LABEL: func.func @accumulator_outside_loop
// CHECK-NOT: hoist_alloc
// CHECK-NOT: unroll
module {
  air.channel @ca [1, 1]
  air.channel @cb [1, 1]
  air.channel @cacc [1, 1]
  func.func private @zero_acc(memref<4x64xf32, 2>) attributes {llvm.emit_c_interface}
  func.func private @mm_knl(memref<4x64xbf16, 2>, memref<4x64xbf16, 2>, memref<4x64xf32, 2>, i32)
      attributes {llvm.emit_c_interface}
  func.func @accumulator_outside_loop() {
    %c1 = arith.constant 1 : index
    %0 = air.herd @h async tile (%tx, %ty) in (%sx=%c1, %sy=%c1) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
      %c64_i32 = arith.constant 64 : i32
      %tacc, %bacc = air.execute -> (memref<4x64xf32, 2>) {
        %a = memref.alloc() : memref<4x64xf32, 2>
        air.execute_terminator %a : memref<4x64xf32, 2>
      }
      %tz = air.execute [%tacc] {
        func.call @zero_acc(%bacc) : (memref<4x64xf32, 2>) -> ()
      }
      // expected-warning@+1 {{is a ping-pong candidate that cannot be proven safe to transform}}
      %1 = scf.for %i = %c0 to %c16 step %c4 iter_args(%t = %tz) -> (!air.async.token) {
        %ta, %ba = air.execute -> (memref<4x64xbf16, 2>) {
          %a = memref.alloc() : memref<4x64xbf16, 2>
          air.execute_terminator %a : memref<4x64xbf16, 2>
        }
        %tb, %bb = air.execute -> (memref<4x64xbf16, 2>) {
          %a = memref.alloc() : memref<4x64xbf16, 2>
          air.execute_terminator %a : memref<4x64xbf16, 2>
        }
        %ga = air.channel.get async [%ta, %t] @ca[%tx, %ty] (%ba[] [] []) : (memref<4x64xbf16, 2>)
        %gb = air.channel.get async [%tb, %t] @cb[%tx, %ty] (%bb[] [] []) : (memref<4x64xbf16, 2>)
        %tc = air.execute [%ga, %gb] {
          func.call @mm_knl(%ba, %bb, %bacc, %c64_i32) : (memref<4x64xbf16, 2>, memref<4x64xbf16, 2>, memref<4x64xf32, 2>, i32) -> ()
        }
        %da = air.execute [%tc] { memref.dealloc %ba : memref<4x64xbf16, 2> }
        %db = air.execute [%tc] { memref.dealloc %bb : memref<4x64xbf16, 2> }
        %w = air.wait_all async [%da, %db]
        scf.yield %w : !air.async.token
      }
      %p = air.channel.put async [%1] @cacc[%tx, %ty] (%bacc[] [] []) : (memref<4x64xf32, 2>)
      %td = air.execute [%p] { memref.dealloc %bacc : memref<4x64xf32, 2> }
    }
    return
  }
}

// -----

// The refusal must also be scoped in TIME. Here the call accumulates into
// an external buffer through an UNANNOTATED operand, and the only
// classified WRITE to that buffer is a channel.get sequenced AFTER the
// loop, refilling it for a later phase. That write waits on the loop's
// result token, which the rebuild preserves; it cannot feed data into the
// loop's iterations, so there is no edge to sever and it must not block
// the transform. The loop's own candidates are annotated and refilled per
// iteration, so the loop must be LABELED. An earlier draft counted any
// out-of-loop write regardless of its position and hard-refused this
// working shape.
// CHECK-LABEL: func.func @write_after_loop
// CHECK: scf.for
// CHECK: hoist_alloc
// CHECK: } {unroll = 2 : i32}
module {
  air.channel @ca2 [1, 1]
  air.channel @cb2 [1, 1]
  air.channel @cnext [1, 1]
  func.func private @mm_knl2(memref<4x64xbf16, 2> {llvm.readonly},
                             memref<4x64xbf16, 2> {llvm.readonly},
                             memref<4x64xf32, 2>, i32)
      attributes {llvm.emit_c_interface}
  func.func @write_after_loop() {
    %c1 = arith.constant 1 : index
    %0 = air.herd @h async tile (%tx, %ty) in (%sx=%c1, %sy=%c1) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
      %c64_i32 = arith.constant 64 : i32
      %tacc, %bacc = air.execute -> (memref<4x64xf32, 2>) {
        %a = memref.alloc() : memref<4x64xf32, 2>
        air.execute_terminator %a : memref<4x64xf32, 2>
      }
      %1 = scf.for %i = %c0 to %c16 step %c4 iter_args(%t = %tacc) -> (!air.async.token) {
        %ta, %ba = air.execute -> (memref<4x64xbf16, 2>) {
          %a = memref.alloc() : memref<4x64xbf16, 2>
          air.execute_terminator %a : memref<4x64xbf16, 2>
        }
        %tb, %bb = air.execute -> (memref<4x64xbf16, 2>) {
          %a = memref.alloc() : memref<4x64xbf16, 2>
          air.execute_terminator %a : memref<4x64xbf16, 2>
        }
        %ga = air.channel.get async [%ta, %t] @ca2[%tx, %ty] (%ba[] [] []) : (memref<4x64xbf16, 2>)
        %gb = air.channel.get async [%tb, %t] @cb2[%tx, %ty] (%bb[] [] []) : (memref<4x64xbf16, 2>)
        %tc = air.execute [%ga, %gb] {
          func.call @mm_knl2(%ba, %bb, %bacc, %c64_i32) : (memref<4x64xbf16, 2>, memref<4x64xbf16, 2>, memref<4x64xf32, 2>, i32) -> ()
        }
        %da = air.execute [%tc] { memref.dealloc %ba : memref<4x64xbf16, 2> }
        %db = air.execute [%tc] { memref.dealloc %bb : memref<4x64xbf16, 2> }
        %w = air.wait_all async [%da, %db]
        scf.yield %w : !air.async.token
      }
      %g = air.channel.get async [%1] @cnext[%tx, %ty] (%bacc[] [] []) : (memref<4x64xf32, 2>)
      %td = air.execute [%g] { memref.dealloc %bacc : memref<4x64xf32, 2> }
    }
    return
  }
}

// -----

// The same time-scoping with UNANNOTATED candidates: identical to
// accumulator_outside_loop except the accumulator is also refilled by a
// channel.get AFTER the loop (buffer reuse for the next phase). The
// after-loop write still cannot carry data INTO the loop, so the verdict
// must stay Skip -- a warning because the loop's own candidates are
// unannotated -- and must not escalate to a hard refusal.
// CHECK-LABEL: func.func @write_after_loop_unannotated
// CHECK-NOT: hoist_alloc
// CHECK-NOT: unroll
module {
  air.channel @ca3 [1, 1]
  air.channel @cb3 [1, 1]
  air.channel @cnext3 [1, 1]
  func.func private @zero_acc3(memref<4x64xf32, 2>) attributes {llvm.emit_c_interface}
  func.func private @mm_knl3(memref<4x64xbf16, 2>, memref<4x64xbf16, 2>, memref<4x64xf32, 2>, i32)
      attributes {llvm.emit_c_interface}
  func.func @write_after_loop_unannotated() {
    %c1 = arith.constant 1 : index
    %0 = air.herd @h async tile (%tx, %ty) in (%sx=%c1, %sy=%c1) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
      %c64_i32 = arith.constant 64 : i32
      %tacc, %bacc = air.execute -> (memref<4x64xf32, 2>) {
        %a = memref.alloc() : memref<4x64xf32, 2>
        air.execute_terminator %a : memref<4x64xf32, 2>
      }
      %tz = air.execute [%tacc] {
        func.call @zero_acc3(%bacc) : (memref<4x64xf32, 2>) -> ()
      }
      // expected-warning@+1 {{is a ping-pong candidate that cannot be proven safe to transform}}
      %1 = scf.for %i = %c0 to %c16 step %c4 iter_args(%t = %tz) -> (!air.async.token) {
        %ta, %ba = air.execute -> (memref<4x64xbf16, 2>) {
          %a = memref.alloc() : memref<4x64xbf16, 2>
          air.execute_terminator %a : memref<4x64xbf16, 2>
        }
        %tb, %bb = air.execute -> (memref<4x64xbf16, 2>) {
          %a = memref.alloc() : memref<4x64xbf16, 2>
          air.execute_terminator %a : memref<4x64xbf16, 2>
        }
        %ga = air.channel.get async [%ta, %t] @ca3[%tx, %ty] (%ba[] [] []) : (memref<4x64xbf16, 2>)
        %gb = air.channel.get async [%tb, %t] @cb3[%tx, %ty] (%bb[] [] []) : (memref<4x64xbf16, 2>)
        %tc = air.execute [%ga, %gb] {
          func.call @mm_knl3(%ba, %bb, %bacc, %c64_i32) : (memref<4x64xbf16, 2>, memref<4x64xbf16, 2>, memref<4x64xf32, 2>, i32) -> ()
        }
        %da = air.execute [%tc] { memref.dealloc %ba : memref<4x64xbf16, 2> }
        %db = air.execute [%tc] { memref.dealloc %bb : memref<4x64xbf16, 2> }
        %w = air.wait_all async [%da, %db]
        scf.yield %w : !air.async.token
      }
      %g = air.channel.get async [%1] @cnext3[%tx, %ty] (%bacc[] [] []) : (memref<4x64xf32, 2>)
      %td = air.execute [%g] { memref.dealloc %bacc : memref<4x64xf32, 2> }
    }
    return
  }
}

// -----

// A herd-level buffer refilled by an enclosing loop, read inside the
// candidate loop through an unannotated operand. The rotation does not
// privatize %bw -- the only buffer it duplicates is %bx, whose use is
// annotated read-only and whose channel.get refills it every iteration --
// so the loop must be LABELED: %bw stays one physical buffer, the calls
// reach it through edges chained from the loop's init token, and the
// outer loop's refill waits on the loop's result token, all of which the
// rebuild preserves. An earlier draft treated the outer loop's
// recirculation of the refill as data carried INTO the loop and
// hard-refused this shape; that same trigger refused three shipped LLM
// deployments reading a loop-invariant buffer through an external call.
// CHECK-LABEL: func.func @write_after_loop_recirculated
// CHECK: scf.for
// CHECK: hoist_alloc
// CHECK: } {unroll = 2 : i32}
module {
  air.channel @cx4 [1, 1]
  air.channel @cw4 [1, 1]
  func.func private @knl4(memref<4x64xbf16, 2> {llvm.readonly},
                          memref<64xbf16, 2>, i32)
      attributes {llvm.emit_c_interface}
  func.func @write_after_loop_recirculated() {
    %c1 = arith.constant 1 : index
    %0 = air.herd @h async tile (%tx, %ty) in (%sx=%c1, %sy=%c1) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c64_i32 = arith.constant 64 : i32
      %tw, %bw = air.execute -> (memref<64xbf16, 2>) {
        %a = memref.alloc() : memref<64xbf16, 2>
        air.execute_terminator %a : memref<64xbf16, 2>
      }
      %2 = scf.for %k = %c0 to %c8 step %c4 iter_args(%tk = %tw) -> (!air.async.token) {
        %1 = scf.for %i = %c0 to %c8 step %c4 iter_args(%t = %tk) -> (!air.async.token) {
          %tx2, %bx = air.execute -> (memref<4x64xbf16, 2>) {
            %a = memref.alloc() : memref<4x64xbf16, 2>
            air.execute_terminator %a : memref<4x64xbf16, 2>
          }
          %gx = air.channel.get async [%tx2, %t] @cx4[%tx, %ty] (%bx[] [] []) : (memref<4x64xbf16, 2>)
          %tc = air.execute [%gx] {
            func.call @knl4(%bx, %bw, %c64_i32) : (memref<4x64xbf16, 2>, memref<64xbf16, 2>, i32) -> ()
          }
          %dx = air.execute [%tc] { memref.dealloc %bx : memref<4x64xbf16, 2> }
          scf.yield %dx : !air.async.token
        }
        %gw = air.channel.get async [%1] @cw4[%tx, %ty] (%bw[] [] []) : (memref<64xbf16, 2>)
        scf.yield %gw : !air.async.token
      }
      %td = air.execute [%2] { memref.dealloc %bw : memref<64xbf16, 2> }
    }
    return
  }
}

// -----

// A duplicated buffer with NO recognized consumer is vacuously safe: the
// buffer is filled each iteration but provably nothing reads it (every use
// is classified), so there is no reader for a reuse edge to protect and
// each iteration's fill lands in its own half unobserved. The pass must
// LABEL this loop. An earlier draft of the proof refused it, which rejected
// bare alloc/dealloc and fill-only loops the compiler's pre-existing tests
// have always accepted; this case is the regression guard against that
// refusal coming back.
// CHECK-LABEL: func.func @no_consumer
// CHECK: hoist_alloc
// CHECK: } {unroll = 2 : i32}
module {
  air.channel @cn [1, 1]
  func.func @no_consumer() {
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
        %g = air.channel.get async [%tb, %t] @cn[%tx, %ty] (%bb[] [] []) : (memref<64xbf16, 2>)
        %d = air.execute [%g] { memref.dealloc %bb : memref<64xbf16, 2> }
        scf.yield %d : !air.async.token
      }
    }
    return
  }
}

// -----

// A read-modify-write of a duplicated buffer (memref.atomic_rmw declares
// BOTH memory effects on its memref operand) is a READER whose write
// depends on the buffer's prior contents -- the loop-carried dependence
// rotation breaks. The classifier used to answer 'w' for it (Write was
// queried before Read), so the proof counted the accumulator as its own
// producer, saw no reader, and declared the buffer vacuously safe --
// labeling a loop whose rotation splits the accumulation across two
// halves. It must REFUSE: the buffer is read but nothing refills it each
// iteration.
module {
  air.channel @cdrain [1, 1]
  func.func @rmw_accumulator() {
    %c1 = arith.constant 1 : index
    %0 = air.herd @h async tile (%tx, %ty) in (%sx=%c1, %sy=%c1) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %ci = arith.constant 0 : index
      %cst = arith.constant 1.0 : f32
      %t0 = air.wait_all async
      // expected-error@+1 {{is a ping-pong candidate that cannot be proven safe to transform}}
      %1 = scf.for %i = %c0 to %c8 step %c4 iter_args(%t = %t0) -> (!air.async.token) {
        %tb, %bb = air.execute -> (memref<64xf32, 2>) {
          %a = memref.alloc() : memref<64xf32, 2>
          air.execute_terminator %a : memref<64xf32, 2>
        }
        %tc = air.execute [%tb, %t] {
          %old = memref.atomic_rmw addf %cst, %bb[%ci] : (f32, memref<64xf32, 2>) -> f32
        }
        %p = air.channel.put async [%tc] @cdrain[%tx, %ty] (%bb[] [] []) : (memref<64xf32, 2>)
        %d = air.execute [%p] { memref.dealloc %bb : memref<64xf32, 2> }
        scf.yield %d : !air.async.token
      }
    }
    return
  }
}

// -----

// The same read-modify-write, but the buffer is refilled by a channel.get
// on EVERY iteration before the accumulation touches it: each iteration's
// read observes that iteration's own fill, so rotation is provably safe
// and the loop must still LABEL. This is the guard against the 'b'
// classification over-refusing -- a read-modify-write is only unsafe when
// the data it reads carries across iterations.
// CHECK-LABEL: func.func @rmw_with_refill
// CHECK: scf.for
// CHECK: hoist_alloc
// CHECK: } {unroll = 2 : i32}
module {
  air.channel @cfill [1, 1]
  air.channel @cdrain [1, 1]
  func.func @rmw_with_refill() {
    %c1 = arith.constant 1 : index
    %0 = air.herd @h async tile (%tx, %ty) in (%sx=%c1, %sy=%c1) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %ci = arith.constant 0 : index
      %cst = arith.constant 1.0 : f32
      %t0 = air.wait_all async
      %1 = scf.for %i = %c0 to %c8 step %c4 iter_args(%t = %t0) -> (!air.async.token) {
        %tb, %bb = air.execute -> (memref<64xf32, 2>) {
          %a = memref.alloc() : memref<64xf32, 2>
          air.execute_terminator %a : memref<64xf32, 2>
        }
        %g = air.channel.get async [%tb, %t] @cfill[%tx, %ty] (%bb[] [] []) : (memref<64xf32, 2>)
        %tc = air.execute [%g] {
          %old = memref.atomic_rmw addf %cst, %bb[%ci] : (f32, memref<64xf32, 2>) -> f32
        }
        %p = air.channel.put async [%tc] @cdrain[%tx, %ty] (%bb[] [] []) : (memref<64xf32, 2>)
        %d = air.execute [%p] { memref.dealloc %bb : memref<64xf32, 2> }
        scf.yield %d : !air.async.token
      }
    }
    return
  }
}

// -----

// One duplicated buffer passed to TWO formals of the same call -- one
// llvm.writeonly, one llvm.readonly. At op level that is a read-modify-write
// of the buffer, exactly like memref.atomic_rmw: the callee's internal order
// of the two accesses is unknowable, so the write must not count as the
// refill protecting the call's own read. Classifying only the first operand
// position ('w') let the proof record the call as both producer and
// consumer -- vouching for itself -- and label a loop whose rotation hands
// the readonly formal a stale half. It must REFUSE: the buffer is read but
// nothing else refills it each iteration.
module {
  air.channel @cdrain2 [1, 1]
  func.func private @knl_wr(memref<64xf32, 2> {llvm.writeonly},
                            memref<64xf32, 2> {llvm.readonly}, i32)
      attributes {llvm.emit_c_interface}
  func.func @same_buffer_two_formals() {
    %c1 = arith.constant 1 : index
    %0 = air.herd @h async tile (%tx, %ty) in (%sx=%c1, %sy=%c1) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c64_i32 = arith.constant 64 : i32
      %t0 = air.wait_all async
      // expected-error@+1 {{is a ping-pong candidate that cannot be proven safe to transform}}
      %1 = scf.for %i = %c0 to %c8 step %c4 iter_args(%t = %t0) -> (!air.async.token) {
        %tb, %bb = air.execute -> (memref<64xf32, 2>) {
          %a = memref.alloc() : memref<64xf32, 2>
          air.execute_terminator %a : memref<64xf32, 2>
        }
        %tc = air.execute [%tb, %t] {
          func.call @knl_wr(%bb, %bb, %c64_i32) : (memref<64xf32, 2>, memref<64xf32, 2>, i32) -> ()
        }
        %p = air.channel.put async [%tc] @cdrain2[%tx, %ty] (%bb[] [] []) : (memref<64xf32, 2>)
        %d = air.execute [%p] { memref.dealloc %bb : memref<64xf32, 2> }
        scf.yield %d : !air.async.token
      }
    }
    return
  }
}

// -----

// The same two-formal read-modify-write, but the buffer is refilled by a
// channel.get on EVERY iteration before the call touches it: each
// iteration's read observes that iteration's own fill, so rotation is
// provably safe and the loop must still LABEL. Guard against the op-level
// aggregation over-refusing -- a call that reads and writes the same buffer
// is only unsafe when the data it reads carries across iterations.
// CHECK-LABEL: func.func @same_buffer_two_formals_with_refill
// CHECK: scf.for
// CHECK: hoist_alloc
// CHECK: } {unroll = 2 : i32}
module {
  air.channel @cfill3 [1, 1]
  air.channel @cdrain3 [1, 1]
  func.func private @knl_wr2(memref<64xf32, 2> {llvm.writeonly},
                             memref<64xf32, 2> {llvm.readonly}, i32)
      attributes {llvm.emit_c_interface}
  func.func @same_buffer_two_formals_with_refill() {
    %c1 = arith.constant 1 : index
    %0 = air.herd @h async tile (%tx, %ty) in (%sx=%c1, %sy=%c1) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c64_i32 = arith.constant 64 : i32
      %t0 = air.wait_all async
      %1 = scf.for %i = %c0 to %c8 step %c4 iter_args(%t = %t0) -> (!air.async.token) {
        %tb, %bb = air.execute -> (memref<64xf32, 2>) {
          %a = memref.alloc() : memref<64xf32, 2>
          air.execute_terminator %a : memref<64xf32, 2>
        }
        %g = air.channel.get async [%tb, %t] @cfill3[%tx, %ty] (%bb[] [] []) : (memref<64xf32, 2>)
        %tc = air.execute [%g] {
          func.call @knl_wr2(%bb, %bb, %c64_i32) : (memref<64xf32, 2>, memref<64xf32, 2>, i32) -> ()
        }
        %p = air.channel.put async [%tc] @cdrain3[%tx, %ty] (%bb[] [] []) : (memref<64xf32, 2>)
        %d = air.execute [%p] { memref.dealloc %bb : memref<64xf32, 2> }
        scf.yield %d : !air.async.token
      }
    }
    return
  }
}
