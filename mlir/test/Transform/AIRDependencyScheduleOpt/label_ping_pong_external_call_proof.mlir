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
// call may access a buffer defined outside the loop (the data carries across
// iterations; rotation would corrupt it) or when a duplicated buffer has no
// recognized consumer. Silence here was a measured 481/512-wrong miscompile.

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

// A duplicated buffer with NO recognized consumer: the buffer is filled each
// iteration but nothing ever reads it, so the reuse edge protecting it until
// its readers finish cannot be built. H1 requires at least one recognized
// consumer for every duplicated buffer; the pass must refuse, not label.
module {
  air.channel @cn [1, 1]
  func.func @no_consumer() {
    %c1 = arith.constant 1 : index
    %0 = air.herd @h async tile (%tx, %ty) in (%sx=%c1, %sy=%c1) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %t0 = air.wait_all async
      // expected-error@+1 {{is a ping-pong candidate that cannot be proven safe to transform}}
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
