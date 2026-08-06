//===- label_ping_pong_loop_invariant_not_rotated.mlir ----------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// H1s: the transform fires where it is provable, and privatizes PER BUFFER,
// not per loop. The callee is fully annotated (every memref formal carries
// llvm.readonly or llvm.writeonly), so the loop is provably safe and MUST
// be labeled -- this is the clause a predicate narrowed until it never
// fires cannot pass. And the weight buffer is filled ONCE, before the
// loop: it is loop-invariant, one physical buffer read by every iteration.
// It is not one of the loop's own per-iteration allocs, so the rotation
// must leave it OUT of the hoist_alloc set -- rotating it would hand later
// iterations the wrong half. The weight is the only memref<64xbf16, 2> in
// the function, so the CHECK-NOTs on that type bracket the whole labeled
// output: if the weight alloc ever acquires hoist_alloc, they fail.
//
// This pins on the compiler's own suite what the driver's
// addnorm_multitrip.py fixture measures on hardware as its
// `annotated_hoisted` arrangement (cols=64, rows=8, rows_per_call=4 -- two
// trips of the row loop): labeled, exact, weight excluded from rotation.

// RUN: air-opt %s -air-label-scf-for-to-ping-pong -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @annotated_callee_hoisted_weight
// CHECK-NOT: hoist_alloc = true} : memref<64xbf16, 2>
// CHECK: memref.alloc() {hoist_alloc = true} : memref<4x64xbf16, 2>
// CHECK: memref.alloc() {hoist_alloc = true} : memref<4x64xbf16, 2>
// CHECK: } {unroll = 2 : i32}
// CHECK-NOT: hoist_alloc = true} : memref<64xbf16, 2>
module {
  air.channel @cw_inv [1, 1]
  air.channel @cx_inv [1, 1]
  air.channel @co_inv [1, 1]
  func.func private @knl_inv(memref<4x64xbf16, 2> {llvm.readonly},
                             memref<64xbf16, 2> {llvm.readonly},
                             memref<4x64xbf16, 2> {llvm.writeonly}, i32)
      attributes {llvm.emit_c_interface}
  func.func @annotated_callee_hoisted_weight() {
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
      %gw = air.channel.get async [%tw] @cw_inv[%tx, %ty] (%bw[] [] []) : (memref<64xbf16, 2>)
      %1 = scf.for %i = %c0 to %c8 step %c4 iter_args(%t = %gw) -> (!air.async.token) {
        %tx2, %bx = air.execute -> (memref<4x64xbf16, 2>) {
          %a = memref.alloc() : memref<4x64xbf16, 2>
          air.execute_terminator %a : memref<4x64xbf16, 2>
        }
        %to, %bo = air.execute -> (memref<4x64xbf16, 2>) {
          %a = memref.alloc() : memref<4x64xbf16, 2>
          air.execute_terminator %a : memref<4x64xbf16, 2>
        }
        %gx = air.channel.get async [%tx2, %t] @cx_inv[%tx, %ty] (%bx[] [] []) : (memref<4x64xbf16, 2>)
        %tc = air.execute [%gx, %to] {
          func.call @knl_inv(%bx, %bw, %bo, %c64_i32) : (memref<4x64xbf16, 2>, memref<64xbf16, 2>, memref<4x64xbf16, 2>, i32) -> ()
        }
        %p = air.channel.put async [%tc] @co_inv[%tx, %ty] (%bo[] [] []) : (memref<4x64xbf16, 2>)
        %dx = air.execute [%tc] { memref.dealloc %bx : memref<4x64xbf16, 2> }
        %do = air.execute [%p] { memref.dealloc %bo : memref<4x64xbf16, 2> }
        %w = air.wait_all async [%dx, %do]
        scf.yield %w : !air.async.token
      }
      %td = air.execute [%1] { memref.dealloc %bw : memref<64xbf16, 2> }
    }
    return
  }
}
