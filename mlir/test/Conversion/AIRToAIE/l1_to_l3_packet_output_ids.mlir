//===- l1_to_l3_packet_output_ids.mlir ---------------------*- MLIR -*-===//
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=2 col-offset=0 device=npu1" --split-input-file | FileCheck %s

// CHECK-LABEL: aie.device(npu1) @seg
// CHECK-DAG: aie.packet_flow(0)
// CHECK-DAG: aie.packet_flow(1)
// CHECK-DAG: aie.shim_dma_allocation @air_pkt_out_0{{.*}}S2MM, 0, <pkt_type = 0, pkt_id = 0>
// CHECK-DAG: aie.shim_dma_allocation @air_pkt_out_1{{.*}}S2MM, 0, <pkt_type = 0, pkt_id = 1>
// CHECK-DAG: air.channel.get{{.*}}@pkt_out{{.*}}metadataArray{{.*}}packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>
// CHECK-DAG: air.channel.get{{.*}}@pkt_out{{.*}}metadataArray{{.*}}packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>

module {
  air.channel @pkt_out [1, 2] {channel_type = "npu_dma_packet"}

  func.func @l1_to_l3_two_packet_outputs(%arg0: memref<2x64xbf16>) {
    %0 = air.launch async () in () args(%out=%arg0) : memref<2x64xbf16> attributes {id = 1 : i32} {
      %c64 = arith.constant 64 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %1 = air.channel.get async @pkt_out[%c0, %c0] (%out[%c0, %c0] [%c1, %c64] [%c64, %c1]) {id = 1 : i32} : (memref<2x64xbf16>)
      %2 = air.channel.get async @pkt_out[%c0, %c1] (%out[%c1, %c0] [%c1, %c64] [%c64, %c1]) {id = 2 : i32} : (memref<2x64xbf16>)
      %3 = air.segment @seg async attributes {id = 2 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
        %sc2 = arith.constant 2 : index
        %sc1 = arith.constant 1 : index
        %4 = air.herd @herd async tile (%tx, %ty) in (%sx=%sc1, %sy=%sc2) attributes {id = 3 : i32} {
          %async_token, %buf = air.execute -> (memref<64xbf16, 2>) {
            %alloc = memref.alloc() : memref<64xbf16, 2>
            air.execute_terminator %alloc : memref<64xbf16, 2>
          }
          %put = air.channel.put async [%async_token] @pkt_out[%tx, %ty] (%buf[] [] []) {id = 3 : i32} : (memref<64xbf16, 2>)
          %5 = air.execute [%put] {
            memref.dealloc %buf : memref<64xbf16, 2>
          }
        }
      }
    }
    return
  }
}
