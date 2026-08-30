//===- air_attr_verify_invalid.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// airDialect::verifyOperationAttribute: every air.* discardable attribute is
// checked for its TYPE and for the OP it may sit on, on every op after every
// pass. These cases lock the mistyped and misplaced forms of the two
// documented user-facing attributes; the value-range cases for
// air.shim_dma_tile_sizes on a correct air.launch live in
// Transform/AIRDependencyScheduleOpt/opt_shim_dma_bds_per_launch_attr_invalid.mlir.

// RUN: air-opt %s -split-input-file -verify-diagnostics

// air.disable_ping_pong must be a UnitAttr, even on the right op.
func.func @mistyped_disable_ping_pong() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // expected-error@+1 {{air.disable_ping_pong must be a unit attribute}}
  scf.for %i = %c0 to %c4 step %c1 {
  } {air.disable_ping_pong = 1 : i32}
  return
}

// -----

// air.disable_ping_pong may only sit on a loop the labeling pass reads.
// expected-error@+1 {{air.disable_ping_pong may only be attached to an scf.for or scf.parallel loop}}
func.func @misplaced_disable_ping_pong() attributes {air.disable_ping_pong} {
  return
}

// -----

// air.shim_dma_tile_sizes may only sit on an air.launch.
func.func @misplaced_shim_dma_tile_sizes() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // expected-error@+1 {{air.shim_dma_tile_sizes may only be attached to an air.launch}}
  scf.for %i = %c0 to %c4 step %c1 {
  } {air.shim_dma_tile_sizes = array<i64: 64>}
  return
}

// -----

// air.shim_dma_tile_sizes must be a DenseI64ArrayAttr.
func.func @mistyped_shim_dma_tile_sizes() {
  %c1 = arith.constant 1 : index
  // expected-error@+1 {{air.shim_dma_tile_sizes must be a dense i64 array attribute}}
  air.launch (%a) in (%s=%c1) attributes {air.shim_dma_tile_sizes = "foo"} {
  }
  return
}

// -----

// Positive control: both attributes, correctly typed and placed, verify.
func.func @well_formed() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
  } {air.disable_ping_pong}
  scf.parallel (%p) = (%c0) to (%c4) step (%c1) {
    scf.reduce
  } {air.disable_ping_pong}
  air.launch (%a) in (%s=%c1) attributes {air.shim_dma_tile_sizes = array<i64: 64, 32>} {
  }
  return
}
