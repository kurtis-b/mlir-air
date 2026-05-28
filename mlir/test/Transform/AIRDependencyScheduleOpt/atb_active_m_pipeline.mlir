//===- atb_active_m_pipeline.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-ping-pong-transform | FileCheck %s

// CHECK-LABEL: atb_active_m_two_buffer
// CHECK: %[[PING_TOK:[[:alnum:]_]+]], %[[PING_BUF:[[:alnum:]_]+]] = air.execute {{.*}} -> (memref<18x6x8x8xi8, 2>)
// CHECK: %[[PONG_TOK:[[:alnum:]_]+]], %[[PONG_BUF:[[:alnum:]_]+]] = air.execute {{.*}} -> (memref<18x6x8x8xi8, 2>)
// CHECK: %[[LOOP:[[:alnum:]_]+]]:4 = scf.for {{.*}} iter_args(%[[PREV:[[:alnum:]_]+]] = {{.*}}, %[[PING_FREE:[[:alnum:]_]+]] = %[[PING_TOK]], %[[PONG_FREE:[[:alnum:]_]+]] = %[[PONG_TOK]], %[[DONE:[[:alnum:]_]+]] = {{.*}}) -> (!air.async.token, !air.async.token, !air.async.token, !air.async.token)
// CHECK: scf.if {{.*}} -> (!air.async.token)
// CHECK: scf.if {{.*}} -> (!air.async.token)
// CHECK: air.channel.get async [%[[PREV]], %[[PING_FREE]]]
// CHECK-SAME: @channel_0
// CHECK-SAME: (%[[PING_BUF]][] [] [])
// CHECK: air.channel.get async [%[[PREV]], %[[PONG_FREE]]]
// CHECK-SAME: @channel_0
// CHECK-SAME: (%[[PONG_BUF]][] [] [])
// CHECK: scf.if {{.*}} -> (!air.async.token, !air.async.token, !air.async.token)
// CHECK: scf.if {{.*}} -> (!air.async.token, !air.async.token, !air.async.token)
// CHECK: air.execute [%[[DONE]], %[[PREV]]]
// CHECK: func.call @matmul_i8_i8_i8_acc32_strix(%[[PING_BUF]],
// CHECK: air.execute [%[[DONE]], %[[PREV]]]
// CHECK: func.call @matmul_i8_i8_i8_acc32_strix(%[[PONG_BUF]],
// CHECK: scf.yield {{.*}} : !air.async.token, !air.async.token, !air.async.token, !air.async.token
// CHECK: atb_two_buffer_pipeline = true
// CHECK: memref.dealloc %[[PING_BUF]]
// CHECK: memref.dealloc %[[PONG_BUF]]
// CHECK: air.wait_all [%[[LOOP]]#3]

air.channel @channel_0 [1, 1]

module {
  func.func private @matmul_i8_i8_i8_acc32_strix(
      memref<18x6x8x8xi8, 2>, memref<18x18x8x8xi8, 2>,
      memref<18x18x8x8xi8, 2>) attributes {link_with = "mm.o", llvm.emit_c_interface}

  func.func @atb_active_m_two_buffer() {
    %c0 = arith.constant 0 : index
    %c6 = arith.constant 6 : index
    %c18 = arith.constant 18 : index
    %start = air.wait_all async
    %b_tok, %b = air.execute [%start] -> (memref<18x18x8x8xi8, 2>) {
      %alloc = memref.alloc() : memref<18x18x8x8xi8, 2>
      air.execute_terminator %alloc : memref<18x18x8x8xi8, 2>
    }
    %c_tok, %c = air.execute [%b_tok] -> (memref<18x18x8x8xi8, 2>) {
      %alloc = memref.alloc() : memref<18x18x8x8xi8, 2>
      air.execute_terminator %alloc : memref<18x18x8x8xi8, 2>
    }
    %loop = scf.for %iv = %c0 to %c18 step %c6 iter_args(%tok = %c_tok) -> (!air.async.token) {
      %a_tok, %a = air.execute -> (memref<18x6x8x8xi8, 2>) {
        %alloc = memref.alloc() : memref<18x6x8x8xi8, 2>
        air.execute_terminator %alloc : memref<18x6x8x8xi8, 2>
      }
      %get = air.channel.get async [%tok, %a_tok] @channel_0[] (%a[] [] []) {id = 0 : i32} : (memref<18x6x8x8xi8, 2>)
      %compute = air.execute [%get] {
        func.call @matmul_i8_i8_i8_acc32_strix(%a, %b, %c) : (memref<18x6x8x8xi8, 2>, memref<18x18x8x8xi8, 2>, memref<18x18x8x8xi8, 2>) -> ()
      }
      %dealloc = air.execute [%compute] {
        memref.dealloc %a : memref<18x6x8x8xi8, 2>
      }
      scf.yield %compute : !air.async.token
    }
    air.wait_all [%loop]
    return
  }
}
