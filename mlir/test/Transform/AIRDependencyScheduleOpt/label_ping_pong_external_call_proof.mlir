//===- label_ping_pong_external_call_proof.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// H1+H2: a ping-pong candidate whose compute is an external kernel call
// (llvm.emit_c_interface) is PROVEN safe when the callee's argument
// attributes (llvm.readonly / llvm.writeonly) classify every memref the call
// touches and each such memref is one of the loop's own per-iteration
// buffers; SKIPPED with a warning when an operand is unannotated
// (read-versus-write is not established, so the loop keeps its correct
// untransformed schedule); and REFUSED with a hard pass failure when the
// call may access a buffer that receives a classified WRITE before the
// loop within the loop's own scope (its data carries into iterations, and
// the only edge ordering that fill against the invisible use is the one
// the transform rebuilds). A classified write sequenced AFTER the loop
// waits on the loop's result token, which the rebuild preserves, and must
// not trigger the refusal. The refusal is scoped to exactly that: a call touching any
// other unprivatized memref -- a herd-argument or call-zero-filled
// accumulator, untouched scratch -- must NOT block the transform; the
// unscoped version of this refusal broke 8 of 24 shipped hardware designs.
// Silence here was a measured 481/512-wrong miscompile.

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
// inside it. No argument attribute proves the call read-only on it, so the
// candidate cannot be proven safe and the pass must refuse, not guess.
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
      // expected-error@+1 {{is a ping-pong candidate that cannot be proven safe to transform}}
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

// The one shape where an "after the loop" write still carries data INTO
// the loop: an enclosing loop re-executes both the candidate loop and the
// write without re-allocating the buffer, so outer-iteration k's refill is
// dynamically BEFORE outer-iteration k+1's unclassified in-loop read. The
// weight buffer lives at herd level, the call reads it through an
// unannotated operand, and the ONLY classified write to it is the
// channel.get refill after the candidate loop but inside the outer loop:
// the pass must still REFUSE. Without the recirculation check this labels
// cleanly (no fill precedes the loop anywhere), so this case guards the
// after-loop escape from being widened into exactly the hoisted-weight
// hole it coexists with.
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
        // expected-error@+1 {{is a ping-pong candidate that cannot be proven safe to transform}}
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
