//===- load_pdi_parity_pad.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// The same-ELF re-execution defect (doc 57 section 1.5; results/reexec-matrix-
// 20260822, devq 528/529): on the full-ELF path every aiex.configure and every
// aiex.npu.load_pdi reset becomes one LOAD_PDI that aiecc's expand-load-pdi
// turns into "load an EMPTY pdi (the partition reset), then write the device's
// configuration inline", alternating TWO empty pdis by position in the control
// stream -- because the NPU2 firmware skips a LOAD_PDI whose pdi_id equals the
// one it loaded last (aie2ps ISA, LOAD_PDI: "consecutive loading of same pdi
// results in following loading skipped by the uC").  The alternation restarts
// at 0 in every dispatch, so a dispatch that issues an ODD number of loads ends
// on the empty pdi it starts with, and the NEXT dispatch of the same ELF begins
// with a LOAD_PDI the firmware skips: launch 0 then runs on the previous
// dispatch's final DMA-channel / lock state (measured: 19 x GEMV(8192) wrong
// in partition 0 from dispatch 2, 20 x the same launch clean; 3-launch QKV
// hangs at dispatch 2, 4-launch clean; a SINGLE multi-iteration GEMV launch
// wrong at dispatch 2, two of them clean).
//
// The pass therefore keeps the per-dispatch LOAD_PDI count EVEN: when the main
// runtime sequence's configure count plus the reset load_pdi count inside the
// configured sequences is odd, it appends `aiex.configure @air_dispatch_end_reset {}`
// -- a tile-less device whose expansion is exactly one empty-pdi load and no
// configuration writes -- so consecutive dispatches never begin with the pdi
// they ended on.  ELF mode only (no load_pdi is emitted otherwise).

// RUN: air-opt -airrt-to-npu="output-elf=true" -canonicalize -cse --split-input-file %s | FileCheck %s --check-prefix=ELF
// RUN: air-opt -airrt-to-npu="output-elf=false" -canonicalize -cse --split-input-file %s | FileCheck %s --check-prefix=XCLBIN

// Case 1: three launches -> 3 loads (odd) -> the pad is appended.
// ELF-LABEL: aie.device(npu2) @air_dispatch_end_reset {
// ELF-NEXT:  }
// ELF:       aie.runtime_sequence @three_launches(
// ELF:         aiex.configure @seg_a {
// ELF:         aiex.configure @seg_b {
// ELF:         aiex.configure @seg_c {
// ELF:         aiex.configure @air_dispatch_end_reset {
// ELF-NEXT:    }
// ELF-NEXT:  }
// XCLBIN-LABEL: aie.runtime_sequence @three_launches(
// XCLBIN-NOT:   air_dispatch_end_reset

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @in_a(%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @out_a(%tile_0_0, S2MM, 0)
  } {sym_name = "seg_a"}
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @in_b(%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @out_b(%tile_0_0, S2MM, 0)
  } {sym_name = "seg_b"}
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @in_c(%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @out_c(%tile_0_0, S2MM, 0)
  } {sym_name = "seg_c"}
  airrt.module_metadata {
  }
  func.func @three_launches(%arg0: memref<512xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c512_i64 = arith.constant 512 : i64
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    affine.for %arg1 = 0 to 1 {
      %0 = airrt.dma_memcpy_nd(%c1_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c512_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @in_a} : (i32, i64, i64, memref<512xi32>) : !airrt.event
      %1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c512_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @out_a} : (i32, i64, i64, memref<512xi32>) : !airrt.event
      airrt.wait_all %0, %1 {"air.launch_end"}
      %p = airrt.segment_load "seg_a" : i64
    } {affine_opt_label = "tiling"}
    affine.for %arg1 = 0 to 1 {
      %0 = airrt.dma_memcpy_nd(%c1_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c512_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @in_b} : (i32, i64, i64, memref<512xi32>) : !airrt.event
      %1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c512_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @out_b} : (i32, i64, i64, memref<512xi32>) : !airrt.event
      airrt.wait_all %0, %1 {"air.launch_end"}
      %p = airrt.segment_load "seg_b" : i64
    } {affine_opt_label = "tiling"}
    affine.for %arg1 = 0 to 1 {
      %0 = airrt.dma_memcpy_nd(%c1_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c512_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @in_c} : (i32, i64, i64, memref<512xi32>) : !airrt.event
      %1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c512_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @out_c} : (i32, i64, i64, memref<512xi32>) : !airrt.event
      airrt.wait_all %0, %1 {"air.launch_end"}
      %p = airrt.segment_load "seg_c" : i64
    } {affine_opt_label = "tiling"}
    return
  }
}

// -----

// Case 2: two launches -> 2 loads (even) -> no pad.
// ELF-LABEL: aie.runtime_sequence @two_launches(
// ELF:         aiex.configure @seg_a {
// ELF:         aiex.configure @seg_b {
// ELF-NOT:     air_dispatch_end_reset
// XCLBIN-LABEL: aie.runtime_sequence @two_launches(
// XCLBIN-NOT:   air_dispatch_end_reset

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @in_a(%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @out_a(%tile_0_0, S2MM, 0)
  } {sym_name = "seg_a"}
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @in_b(%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @out_b(%tile_0_0, S2MM, 0)
  } {sym_name = "seg_b"}
  airrt.module_metadata {
  }
  func.func @two_launches(%arg0: memref<512xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c512_i64 = arith.constant 512 : i64
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    affine.for %arg1 = 0 to 1 {
      %0 = airrt.dma_memcpy_nd(%c1_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c512_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @in_a} : (i32, i64, i64, memref<512xi32>) : !airrt.event
      %1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c512_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @out_a} : (i32, i64, i64, memref<512xi32>) : !airrt.event
      airrt.wait_all %0, %1 {"air.launch_end"}
      %p = airrt.segment_load "seg_a" : i64
    } {affine_opt_label = "tiling"}
    affine.for %arg1 = 0 to 1 {
      %0 = airrt.dma_memcpy_nd(%c1_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c512_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @in_b} : (i32, i64, i64, memref<512xi32>) : !airrt.event
      %1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c512_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @out_b} : (i32, i64, i64, memref<512xi32>) : !airrt.event
      airrt.wait_all %0, %1 {"air.launch_end"}
      %p = airrt.segment_load "seg_b" : i64
    } {affine_opt_label = "tiling"}
    return
  }
}

// -----

// Case 3: one launch with no repeat_count DMAs -> 1 load (odd) -> the pad is
// appended (every single-launch ELF re-dispatched in one context hits this).
// ELF-LABEL: aie.device(npu2) @air_dispatch_end_reset {
// ELF:       aie.runtime_sequence @one_launch(
// ELF:         aiex.configure @seg_a {
// ELF:         aiex.configure @air_dispatch_end_reset {
// XCLBIN-LABEL: aie.runtime_sequence @one_launch(
// XCLBIN-NOT:   air_dispatch_end_reset

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.shim_dma_allocation @in_a(%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @out_a(%tile_0_0, S2MM, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:
      aie.end
    ^bb2:
      aie.end
    }
  } {sym_name = "seg_a"}
  airrt.module_metadata {
  }
  func.func @one_launch(%arg0: memref<512xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c512_i64 = arith.constant 512 : i64
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    affine.for %arg1 = 0 to 1 {
      %0 = airrt.dma_memcpy_nd(%c1_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c512_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @in_a} : (i32, i64, i64, memref<512xi32>) : !airrt.event
      %1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c512_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @out_a} : (i32, i64, i64, memref<512xi32>) : !airrt.event
      airrt.wait_all %0, %1 {"air.launch_end"}
      %p = airrt.segment_load "seg_a" : i64
    } {affine_opt_label = "tiling"}
    return
  }
}

// -----

// Case 4: one launch whose device already gets the per-launch reset
// (repeat_count DMA -> aiex.npu.load_pdi @seg_a_reset) -> 2 loads -> no pad.
// ELF-LABEL: aie.device(npu2) @seg_a_reset {
// ELF:       aie.runtime_sequence @seg_a_sequence(
// ELF:         aiex.npu.load_pdi {device_ref = @seg_a_reset}
// ELF:       aie.runtime_sequence @one_launch(
// ELF:         aiex.configure @seg_a {
// ELF-NOT:     air_dispatch_end_reset
// XCLBIN-LABEL: aie.runtime_sequence @one_launch(
// XCLBIN-NOT:   air_dispatch_end_reset

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.shim_dma_allocation @in_a(%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @out_a(%tile_0_0, S2MM, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2, repeat_count = 3)
    ^bb1:
      aie.end
    ^bb2:
      aie.end
    }
  } {sym_name = "seg_a"}
  airrt.module_metadata {
  }
  func.func @one_launch(%arg0: memref<512xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c512_i64 = arith.constant 512 : i64
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    affine.for %arg1 = 0 to 1 {
      %0 = airrt.dma_memcpy_nd(%c1_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c512_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @in_a} : (i32, i64, i64, memref<512xi32>) : !airrt.event
      %1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c512_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @out_a} : (i32, i64, i64, memref<512xi32>) : !airrt.event
      airrt.wait_all %0, %1 {"air.launch_end"}
      %p = airrt.segment_load "seg_a" : i64
    } {affine_opt_label = "tiling"}
    return
  }
}
