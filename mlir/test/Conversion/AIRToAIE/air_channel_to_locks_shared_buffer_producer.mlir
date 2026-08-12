//===- air_channel_to_locks_shared_buffer_producer.mlir --------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" --split-input-file | FileCheck %s

// A core that PRODUCES into an L1 buffer and then sends it out in N>1 strided
// pieces on one channel -- the resident-FFN "up herd" shape (an accumulator
// written once per round, handed on in chunks_per_group slices).
//
// The interleaved lock placement for shared staging buffers (#1515) puts an
// acquire immediately before EVERY put.  That paces put i+1 against put i, but
// it leaves the core's own WRITES unprotected: after the last put's release the
// core falls straight into the next round and overwrites the buffer while the
// last BD is still streaming out of it.  Measured on NPU2 as a DETERMINISTIC
// wrong answer -- the last slice arrives with only its first run intact and the
// remaining runs zeroed, because the core's memset overtakes the DMA mid-BD.
//
// The requirement this pins: the FIRST acquire of the round must dominate the
// core's first write to the buffer, exactly as the non-interleaved path
// achieves by hoisting the acquire to block start.  With N acquires and N
// releases per round in strict alternation, the round-r+1 block-start acquire
// can only be satisfied by the round-r LAST BD's release.
//
// The companion test air_channel_to_locks_shared_buffer.mlir covers the same
// mode with no producing write, and pins the per-put interleave that must be
// preserved.

// CHECK: aie.device
// CHECK-DAG:         %[[TILE:.*]] = aie.tile(2, 3)
// CHECK-DAG:         %[[WLOCK:.*]] = aie.lock(%[[TILE]], {{[0-9]+}}) {init = 1 : i32}
// CHECK-DAG:         %[[RLOCK:.*]] = aie.lock(%[[TILE]], {{[0-9]+}}) {init = 0 : i32}
// CHECK-DAG:         %[[BUF:.*]] = aie.buffer(%[[TILE]]) {{{.*}}} : memref<64x64xbf16, 2>

// Two BDs, one per slice, at the two literal offsets.
// CHECK:    aie.mem(%[[TILE]])  {
// CHECK:           aie.dma_start(MM2S, 0
// CHECK:           aie.use_lock(%[[RLOCK]], AcquireGreaterEqual, %{{.*}})
// CHECK:           aie.dma_bd(%[[BUF]] : memref<64x64xbf16, 2> offset = 0 len = 2048
// CHECK:           aie.use_lock(%[[WLOCK]], Release, %{{.*}})
// CHECK:           aie.use_lock(%[[RLOCK]], AcquireGreaterEqual, %{{.*}})
// CHECK:           aie.dma_bd(%[[BUF]] : memref<64x64xbf16, 2> offset = 256 len = 2048
// CHECK:           aie.use_lock(%[[WLOCK]], Release, %{{.*}})
// CHECK:         }

// THE CLAUSE THAT FAILS PRE-FIX: the acquire must come BEFORE the producing
// call, not after it.
// CHECK:    aie.core(%[[TILE]])  {
// CHECK:           aie.use_lock(%[[WLOCK]], AcquireGreaterEqual, %{{.*}})
// CHECK:           func.call @producer(%[[BUF]])
// CHECK:           aie.use_lock(%[[RLOCK]], Release, %{{.*}})
// CHECK:           aie.use_lock(%[[WLOCK]], AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT:      aie.use_lock(%[[RLOCK]], Release, %{{.*}})
// CHECK:           aie.end
// CHECK:         }

air.channel @channel_0 [1, 1]
func.func private @producer(memref<64x64xbf16, 2>) attributes {link_with = "kernel.o", llvm.emit_c_interface}
func.func @shared_buffer_producer_then_puts() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) {
    %1 = air.segment async {
      %c1_0 = arith.constant 1 : index
      %async_token_0, %l2_buf = air.execute -> (memref<2048xbf16, 1>) {
        %alloc = memref.alloc() : memref<2048xbf16, 1>
        air.execute_terminator %alloc : memref<2048xbf16, 1>
      }
      %3 = air.channel.get async @channel_0[] (%l2_buf[] [] []) : (memref<2048xbf16, 1>)
      %2 = air.herd @herd_0 async tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c1_0) attributes {link_with = "kernel.o"} {
        %async_token_2, %buf = air.execute -> (memref<64x64xbf16, 2>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2>
          air.execute_terminator %alloc : memref<64x64xbf16, 2>
        }
        // The core writes the whole buffer, ONCE, before either put.
        %async_token_w = air.execute [%async_token_2] {
          func.call @producer(%buf) : (memref<64x64xbf16, 2>) -> ()
        }
        // Two strided slices of that one buffer, at literal offsets.
        %tok_1 = air.channel.put async [%async_token_w] @channel_0[] (%buf[0, 0, 0] [1, 8, 256] [256, 512, 1]) : (memref<64x64xbf16, 2>)
        %tok_2 = air.channel.put async [%tok_1] @channel_0[] (%buf[1, 0, 0] [1, 8, 256] [256, 512, 1]) : (memref<64x64xbf16, 2>)
        %async_token_3 = air.execute [%tok_2] {
          memref.dealloc %buf : memref<64x64xbf16, 2>
        }
      }
    }
  }
  return
}
