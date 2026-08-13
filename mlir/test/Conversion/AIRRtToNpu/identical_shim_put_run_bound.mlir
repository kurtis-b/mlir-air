//===- identical_shim_put_run_bound.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu %s | FileCheck %s

// A launch-level loop whose body is one `air.channel.put` of the SAME slice of
// the SAME operand lowers to `trip` byte-identical airrt.dma_memcpy_nd ops,
// hence `trip` byte-identical configure/start pairs on ONE shim channel with no
// await between them, every release clustered at the terminal wait_all.
// `aiex.npu.push_queue` is a bare register write to that channel's Start_Queue
// and nothing in this compiler accounts for how many are outstanding, so the
// outstanding count is simply the design's trip count.
//
// Doc 52 SS10.6 measures that shape on R1's `hidden` refill: `down_K`
// outstanding starts on %shim_noc_tile_0_0 / MM2S 0, pushed BEFORE the `w_up`
// and `w_down` feeds the consumers need in order to drain them. It separates
// all 21 rungs -- PASS at 2/3/4, FAIL at 5, TIMEOUT at 6+ -- with BD length,
// total task count, tiling and the L2 rotation each excluded against the
// artifact.
//
// THIS IS NOT REACHABLE BY shim_bd_liveness_bound.mlir's STEP. Six live BDs on
// a 16-BD tile is comfortably under budget, so the BD-liveness bound is a no-op
// here; that is exactly why the shape survived it.
//
// WHY NOT FOLD THE RUN INTO `repeat_count` (doc 52 SS10.9's suggestion). A
// task's repeat_count IS the descriptor's iteration dimension: airrt-to-npu
// sets it to `sizes[0] - 1` and, when `strides[0] != 0`, also emits dim 0 in
// the BD layout to carry the iteration stride, so `repeat_count + 1` executions
// ADVANCE the address and cannot restart it. Concatenating `trip` identical
// copies needs an outer dimension of size `trip` at stride 0 on top of that --
// a fifth hardware dimension. On R1's own emitted module the `hidden`
// descriptor is `sizes = [8, 4, 8, 8] strides = [256, 8, 32, 1]` with
// `repeat_count = 7` and no adjacent pair is mergeable, so all four dimensions
// are already in use and the fold is arithmetically unavailable.
//
// So the run is paced instead, with the same mechanism and the same two
// invariants the BD-liveness bound uses.
//
// THE TRIGGER IS STRUCTURAL AND NO QUEUE DEPTH IS CLAIMED. What fires the step
// is that the run is a loop of IDENTICAL puts -- a trip-shaped occupancy the
// design can raise without bound and that the compiler never folded. @varying
// below is the negative control: six transfers on their own channel with
// DISTINCT offsets are a real multi-part transfer, not an unfolded loop, and
// they must come out untouched.

// CHECK-LABEL: aie.runtime_sequence @identical_put_run

// The output S2MM stays armed first.
// CHECK: aiex.dma_configure_task_for @out

// The negative control is NOT rewritten and NOT moved: six distinct-offset
// transfers stay fire-and-forget, with no completion token and no paced await,
// in the place the untransformed sequence put them.
// CHECK: %[[V0:.*]] = aiex.dma_configure_task_for @varying
// CHECK-NOT: issue_token
// CHECK: aiex.dma_start_task(%[[V0]])

// Only the identical run is SUNK, and it is sunk past @weights -- the one-shot
// feed the untransformed sequence issued AFTER it. An await that blocks before
// a consumer's other operand has even been issued is a hang, so the pacing is
// only sound once every other feed is already started. This is the clause the
// pre-fix binary fails: without the step, @weights trails all six refills.
// CHECK: aiex.dma_configure_task_for @weights
// CHECK: aiex.dma_start_task

// The identical run: every task now issues a completion token so its BD is
// provably idle before the ID is reused. A dma_free_task would only be a claim
// (mlir-aie: "using dma_free_task(X) before task X has completed will lead to a
// race condition"), so awaits are the only recycle a compiler can argue for.
// CHECK: %[[R0:.*]] = aiex.dma_configure_task_for @refill
// CHECK: issue_token = true
// CHECK: aiex.dma_start_task(%[[R0]])

// Bounded in flight at depth 2, and the await that makes room runs BEFORE the
// configure that takes the ID -- the allocator hands the ID out at the
// configure, so an await one op later is one ID too late.
// CHECK: aiex.dma_await_task(%[[R0]])
// CHECK-NEXT: aiex.dma_configure_task_for @refill

// Every token this step creates is consumed exactly once: the in-flight tail is
// drained after the last start, and the clustered per-transfer free is gone.
// CHECK: aiex.dma_await_task
// CHECK-NOT: aiex.dma_free_task(%[[R0]])

module {
  aie.device(npu1) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    aie.shim_dma_allocation @refill(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @varying(%shim_noc_tile_1_0, MM2S, 0)
    aie.shim_dma_allocation @weights(%shim_noc_tile_3_0, MM2S, 0)
    aie.shim_dma_allocation @out(%shim_noc_tile_2_0, S2MM, 0)
  } {sym_name = "identical_put_run_seg"}
  airrt.module_metadata{}
  func.func @identical_put_run(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>, %arg2: memref<1024xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c32_i64 = arith.constant 32 : i64
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %c5_i32 = arith.constant 5 : i32
    %c64_i64 = arith.constant 64 : i64
    %c96_i64 = arith.constant 96 : i64
    %c128_i64 = arith.constant 128 : i64
    %c160_i64 = arith.constant 160 : i64
    %p = airrt.segment_load "identical_put_run_seg" : i64
    %out = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c32_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out} : (i32, i64, i64, memref<1024xi32>) : !airrt.event
    // Six transfers on @varying, DISTINCT offsets: a real multi-part transfer.
    %v0 = airrt.dma_memcpy_nd(%c5_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c32_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @varying} : (i32, i64, i64, memref<1024xi32>) : !airrt.event
    %v1 = airrt.dma_memcpy_nd(%c5_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c32_i64], [%c1_i64, %c1_i64, %c1_i64, %c32_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @varying} : (i32, i64, i64, memref<1024xi32>) : !airrt.event
    %v2 = airrt.dma_memcpy_nd(%c5_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c64_i64], [%c1_i64, %c1_i64, %c1_i64, %c32_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @varying} : (i32, i64, i64, memref<1024xi32>) : !airrt.event
    %v3 = airrt.dma_memcpy_nd(%c5_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c96_i64], [%c1_i64, %c1_i64, %c1_i64, %c32_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @varying} : (i32, i64, i64, memref<1024xi32>) : !airrt.event
    %v4 = airrt.dma_memcpy_nd(%c5_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c128_i64], [%c1_i64, %c1_i64, %c1_i64, %c32_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @varying} : (i32, i64, i64, memref<1024xi32>) : !airrt.event
    %v5 = airrt.dma_memcpy_nd(%c5_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c160_i64], [%c1_i64, %c1_i64, %c1_i64, %c32_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @varying} : (i32, i64, i64, memref<1024xi32>) : !airrt.event
    // Six transfers on @refill, IDENTICAL in every field: the unfolded loop.
    %r0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c32_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @refill} : (i32, i64, i64, memref<1024xi32>) : !airrt.event
    %r1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c32_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @refill} : (i32, i64, i64, memref<1024xi32>) : !airrt.event
    %r2 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c32_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @refill} : (i32, i64, i64, memref<1024xi32>) : !airrt.event
    %r3 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c32_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @refill} : (i32, i64, i64, memref<1024xi32>) : !airrt.event
    %r4 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c32_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @refill} : (i32, i64, i64, memref<1024xi32>) : !airrt.event
    %r5 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c32_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @refill} : (i32, i64, i64, memref<1024xi32>) : !airrt.event
    %w = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c32_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @weights} : (i32, i64, i64, memref<1024xi32>) : !airrt.event
    airrt.wait_all %out, %v0, %v1, %v2, %v3, %v4, %v5, %r0, %r1, %r2, %r3, %r4, %r5, %w
    return
  }
}
