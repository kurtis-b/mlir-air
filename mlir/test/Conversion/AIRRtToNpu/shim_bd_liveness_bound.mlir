//===- shim_bd_liveness_bound.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu %s | FileCheck %s

// A shim tile has a fixed buffer-descriptor pool (16). mlir-aie's BD allocator
// is a linear program-order walk: a dma_configure_task holds an ID until a
// dma_free_task / dma_await_task gives it back, and AIR emits that give-back at
// the airrt.wait_all that joined the transfer's token. A feed whose N tokens are
// all joined by ONE terminal wait_all therefore lowers to N configures followed
// by N clustered frees -- N live BDs at once, and past the pool the allocator
// refuses outright ("Too many simultaneously active buffer descriptors on tile
// (0,0), which supports up to 16"). That refusal is what blocks doc 31's
// resident FFN interior, whose 4-D shim retile of `hidden` runs 96 iterations.
//
// @bulk below is that shape at 20 transfers. airrt-to-npu must bound it:
//
//  1. every @bulk task issues a completion token and is awaited, so its BD is
//     provably idle before the ID is reused -- a dma_free_task would only be a
//     claim (mlir-aie: "using dma_free_task(X) before task X has completed will
//     lead to a race condition"), and no compiler-inserted recycle can make it;
//  2. the run is SUNK past @weights, the one-shot feed the untransformed
//     sequence issued AFTER it. An await that blocks before a consumer's other
//     operand has even been issued is a hang, so the pacing is only sound once
//     every other feed is already started -- the hazard air.runtime_hoist exists
//     for, closed from the other side.
//
// @out (S2MM) keeps its terminal drain await, which is also the sink anchor.

// CHECK-LABEL: aie.runtime_sequence @bd_liveness
// The output S2MM stays armed first, and the one-shot weight feed now precedes
// the bulk run rather than trailing it.
// CHECK: aiex.dma_configure_task_for @out
// CHECK: aiex.dma_configure_task_for @weights
// CHECK: %[[B0:.*]] = aiex.dma_configure_task_for @bulk
// CHECK: issue_token = true
// CHECK: aiex.dma_start_task(%[[B0]])
// The bounded in-flight set: the run gives BDs back with completion-token
// awaits instead of one clustered free per transfer, and the await that makes
// room runs BEFORE the configure that takes the ID -- the allocator hands the ID
// out at the configure, so an await one op later is one ID too late.
// CHECK: aiex.dma_await_task(%[[B0]])
// CHECK-NEXT: aiex.dma_configure_task_for @bulk
// Every token this step creates is consumed exactly once: the in-flight tail is
// drained after the last start.
// CHECK: aiex.dma_await_task
// CHECK-NOT: aiex.dma_free_task(%[[B0]])

module {
  aie.device(npu1) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    aie.shim_dma_allocation @bulk(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @weights(%shim_noc_tile_1_0, MM2S, 0)
    aie.shim_dma_allocation @out(%shim_noc_tile_2_0, S2MM, 0)
  } {sym_name = "bd_liveness_seg"}
  airrt.module_metadata{}
  func.func @bd_liveness(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>, %arg2: memref<1024xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c32_i64 = arith.constant 32 : i64
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %c64_i64 = arith.constant 64 : i64
    %c96_i64 = arith.constant 96 : i64
    %c128_i64 = arith.constant 128 : i64
    %c160_i64 = arith.constant 160 : i64
    %c192_i64 = arith.constant 192 : i64
    %c224_i64 = arith.constant 224 : i64
    %c256_i64 = arith.constant 256 : i64
    %c288_i64 = arith.constant 288 : i64
    %c320_i64 = arith.constant 320 : i64
    %c352_i64 = arith.constant 352 : i64
    %c384_i64 = arith.constant 384 : i64
    %c416_i64 = arith.constant 416 : i64
    %c448_i64 = arith.constant 448 : i64
    %c480_i64 = arith.constant 480 : i64
    %c512_i64 = arith.constant 512 : i64
    %c544_i64 = arith.constant 544 : i64
    %c576_i64 = arith.constant 576 : i64
    %c608_i64 = arith.constant 608 : i64
    %p = airrt.segment_load "bd_liveness_seg" : i64
    %out = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c32_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out} : (i32, i64, i64, memref<1024xi32>) : !airrt.event
    %b0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c32_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @bulk} : (i32, i64, i64, memref<1024xi32>) : !airrt.event
    %b1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c32_i64], [%c1_i64, %c1_i64, %c1_i64, %c32_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @bulk} : (i32, i64, i64, memref<1024xi32>) : !airrt.event
    %b2 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c64_i64], [%c1_i64, %c1_i64, %c1_i64, %c32_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @bulk} : (i32, i64, i64, memref<1024xi32>) : !airrt.event
    %b3 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c96_i64], [%c1_i64, %c1_i64, %c1_i64, %c32_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @bulk} : (i32, i64, i64, memref<1024xi32>) : !airrt.event
    %b4 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c128_i64], [%c1_i64, %c1_i64, %c1_i64, %c32_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @bulk} : (i32, i64, i64, memref<1024xi32>) : !airrt.event
    %b5 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c160_i64], [%c1_i64, %c1_i64, %c1_i64, %c32_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @bulk} : (i32, i64, i64, memref<1024xi32>) : !airrt.event
    %b6 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c192_i64], [%c1_i64, %c1_i64, %c1_i64, %c32_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @bulk} : (i32, i64, i64, memref<1024xi32>) : !airrt.event
    %b7 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c224_i64], [%c1_i64, %c1_i64, %c1_i64, %c32_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @bulk} : (i32, i64, i64, memref<1024xi32>) : !airrt.event
    %b8 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c256_i64], [%c1_i64, %c1_i64, %c1_i64, %c32_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @bulk} : (i32, i64, i64, memref<1024xi32>) : !airrt.event
    %b9 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c288_i64], [%c1_i64, %c1_i64, %c1_i64, %c32_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @bulk} : (i32, i64, i64, memref<1024xi32>) : !airrt.event
    %b10 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c320_i64], [%c1_i64, %c1_i64, %c1_i64, %c32_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @bulk} : (i32, i64, i64, memref<1024xi32>) : !airrt.event
    %b11 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c352_i64], [%c1_i64, %c1_i64, %c1_i64, %c32_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @bulk} : (i32, i64, i64, memref<1024xi32>) : !airrt.event
    %b12 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c384_i64], [%c1_i64, %c1_i64, %c1_i64, %c32_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @bulk} : (i32, i64, i64, memref<1024xi32>) : !airrt.event
    %b13 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c416_i64], [%c1_i64, %c1_i64, %c1_i64, %c32_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @bulk} : (i32, i64, i64, memref<1024xi32>) : !airrt.event
    %b14 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c448_i64], [%c1_i64, %c1_i64, %c1_i64, %c32_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @bulk} : (i32, i64, i64, memref<1024xi32>) : !airrt.event
    %b15 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c480_i64], [%c1_i64, %c1_i64, %c1_i64, %c32_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @bulk} : (i32, i64, i64, memref<1024xi32>) : !airrt.event
    %b16 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c512_i64], [%c1_i64, %c1_i64, %c1_i64, %c32_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @bulk} : (i32, i64, i64, memref<1024xi32>) : !airrt.event
    %b17 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c544_i64], [%c1_i64, %c1_i64, %c1_i64, %c32_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @bulk} : (i32, i64, i64, memref<1024xi32>) : !airrt.event
    %b18 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c576_i64], [%c1_i64, %c1_i64, %c1_i64, %c32_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @bulk} : (i32, i64, i64, memref<1024xi32>) : !airrt.event
    %b19 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c608_i64], [%c1_i64, %c1_i64, %c1_i64, %c32_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @bulk} : (i32, i64, i64, memref<1024xi32>) : !airrt.event
    %w = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c32_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @weights} : (i32, i64, i64, memref<1024xi32>) : !airrt.event
    airrt.wait_all %out, %b0, %b1, %b2, %b3, %b4, %b5, %b6, %b7, %b8, %b9, %b10, %b11, %b12, %b13, %b14, %b15, %b16, %b17, %b18, %b19, %w
    return
  }
}
