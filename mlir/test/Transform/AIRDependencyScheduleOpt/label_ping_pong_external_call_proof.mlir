//===- label_ping_pong_external_call_proof.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// H1+H2: a ping-pong candidate whose compute is an external kernel call
// (llvm.emit_c_interface) is PROVEN safe when every memref the call touches
// is one of the loop's own per-iteration buffers, and REFUSED with a
// diagnostic when the call touches a buffer defined outside the loop (the
// data carries across iterations; rotation would corrupt it). The refusal is
// a hard pass failure: silence here was a measured 481/512-wrong miscompile.

// RUN: air-opt %s -air-label-scf-for-to-ping-pong -split-input-file -verify-diagnostics | FileCheck %s

// All call operands are refilled inside the loop: provable, labeled.
// CHECK: scf.for
// CHECK: hoist_alloc
// CHECK: } {unroll = 2 : i32}
module {
  air.channel @cw [1, 1]
  air.channel @cx [1, 1]
  func.func private @knl(memref<4x64xbf16, 2>, memref<64xbf16, 2>, i32)
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
