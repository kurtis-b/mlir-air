//===- direct_shim_s2mm_channel_exhaustion.mlir -------------*- MLIR -*-===//
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: not air-opt %s -air-to-aie="row-offset=2 col-offset=0 device=npu1" 2>&1 | FileCheck %s

// CHECK: failed to map to shim dma channels: S2MM direct L3 route exceeded 2 physical channels per shim column across
// CHECK: use an L2 gather/output aggregation route or a validated packet route

module {
  func.func @direct_s2mm_fanout(%arg0: memref<4x3x64xbf16>) {
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    air.herd @herd tile(%tx, %ty) in (%sx = %c4, %sy = %c3)
        args(%out = %arg0) : memref<4x3x64xbf16> {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c192 = arith.constant 192 : index
      %buf = memref.alloc() : memref<64xbf16, 2>
      air.dma_memcpy_nd (
        %out[%tx, %ty, %c0] [%c1, %c1, %c64] [%c192, %c64, %c1],
        %buf[] [] []
      ) {id = 1 : i32} : (memref<4x3x64xbf16>, memref<64xbf16, 2>)
      memref.dealloc %buf : memref<64xbf16, 2>
    }
    return
  }
}
