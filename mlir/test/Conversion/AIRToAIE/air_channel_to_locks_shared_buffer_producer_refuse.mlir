//===- air_channel_to_locks_shared_buffer_producer_refuse.mlir -*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: not air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" 2>&1 | FileCheck %s

// Two puts share one L1 buffer, but put 1 sits inside an scf.for and put 2 is
// its sibling. The guard's backward scan sees only siblings: hoisting put 2's
// write-lock acquire above the loop would let it consume the lock that put 1's
// own acquire (inside the loop, invisible to the scan) must take first -- a
// deadlock on the device. The pass refuses instead of guessing.

// CHECK: error: 'air.channel.put' op puts sharing one L1 buffer must sit in one block

air.channel @channel_0 [1, 1]
func.func private @producer(memref<64x64xbf16, 2>) attributes {link_with = "kernel.o", llvm.emit_c_interface}
func.func @shared_buffer_cross_block_puts() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) {
    %1 = air.segment async {
      %c0_0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %async_token_0, %l2_buf = air.execute -> (memref<2048xbf16, 1>) {
        %alloc = memref.alloc() : memref<2048xbf16, 1>
        air.execute_terminator %alloc : memref<2048xbf16, 1>
      }
      %3 = air.channel.get async @channel_0[] (%l2_buf[] [] []) : (memref<2048xbf16, 1>)
      %2 = air.herd @herd_0 async tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c1_0) attributes {link_with = "kernel.o"} {
        %c0_1 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %async_token_2, %buf = air.execute -> (memref<64x64xbf16, 2>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2>
          air.execute_terminator %alloc : memref<64x64xbf16, 2>
        }
        %async_token_w = air.execute [%async_token_2] {
          func.call @producer(%buf) : (memref<64x64xbf16, 2>) -> ()
        }
        %tok_loop = scf.for %i = %c0_1 to %c1_1 step %c1_1 iter_args(%t = %async_token_w) -> (!air.async.token) {
          %tok_1 = air.channel.put async [%t] @channel_0[] (%buf[0, 0, 0] [1, 16, 128] [128, 256, 1]) : (memref<64x64xbf16, 2>)
          scf.yield %tok_1 : !air.async.token
        }
        %tok_2 = air.channel.put async [%tok_loop] @channel_0[] (%buf[1, 0, 0] [1, 16, 128] [128, 256, 1]) : (memref<64x64xbf16, 2>)
        %async_token_3 = air.execute [%tok_2] {
          memref.dealloc %buf : memref<64x64xbf16, 2>
        }
      }
    }
  }
  return
}
