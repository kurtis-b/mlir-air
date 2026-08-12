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

// air.pipeline_group may only sit on an air.launch. On anything else it would
// simply never be collected, and the pipeline would stay silently unfused --
// which for a pipeline is a broken program, not a slow one.
func.func @misplaced_pipeline_group() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // expected-error@+1 {{air.pipeline_group may only be attached to an air.launch}}
  scf.for %i = %c0 to %c4 step %c1 {
  } {air.pipeline_group = "p"}
  return
}

// -----

// air.pipeline_group must be a string.
func.func @mistyped_pipeline_group() {
  %c1 = arith.constant 1 : index
  // expected-error@+1 {{air.pipeline_group must be a string attribute}}
  air.launch (%a) in (%s=%c1) attributes {air.pipeline_group = 3 : i64} {
  }
  return
}

// -----

// air.pipeline_stage must be an integer.
func.func @mistyped_pipeline_stage() {
  %c1 = arith.constant 1 : index
  // expected-error@+1 {{air.pipeline_stage must be an integer attribute}}
  air.launch (%a) in (%s=%c1) attributes {air.pipeline_group = "p", air.pipeline_stage = "first"} {
  }
  return
}

// -----

// A negative stage index cannot be part of a 0..N-1 cover.
func.func @negative_pipeline_stage() {
  %c1 = arith.constant 1 : index
  // expected-error@+1 {{air.pipeline_stage must be non-negative}}
  air.launch (%a) in (%s=%c1) attributes {air.pipeline_group = "p", air.pipeline_stage = -1 : i64} {
  }
  return
}

// -----

// air.pipeline_stage may only sit on an air.launch.
func.func @misplaced_pipeline_stage() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // expected-error@+1 {{air.pipeline_stage may only be attached to an air.launch}}
  scf.for %i = %c0 to %c4 step %c1 {
  } {air.pipeline_stage = 0 : i64}
  return
}

// -----

// air.staging's value set is closed: an unknown staging is a builder claiming
// a construction the fusion pass has no check for, which would make the
// declaration decorative exactly where it is meant to be load-bearing.
func.func @unknown_staging() {
  %c1 = arith.constant 1 : index
  // expected-error@+1 {{air.staging must be one of "l1", "memtile" or "accum_in_place"; got "l2_ring"}}
  air.launch (%a) in (%s=%c1) attributes {air.pipeline_group = "p", air.pipeline_stage = 0 : i64, air.staging = "l2_ring"} {
  }
  return
}

// -----

// air.staging must be a string.
func.func @mistyped_staging() {
  %c1 = arith.constant 1 : index
  // expected-error@+1 {{air.staging must be a string attribute}}
  air.launch (%a) in (%s=%c1) attributes {air.staging = 1 : i64} {
  }
  return
}

// -----

// Positive control: every attribute, correctly typed and placed, verifies.
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
  air.launch (%b) in (%s2=%c1) attributes {air.pipeline_group = "p", air.pipeline_stage = 0 : i64, air.staging = "l1"} {
  }
  air.launch (%c) in (%s3=%c1) attributes {air.pipeline_group = "p", air.pipeline_stage = 1 : i64, air.staging = "memtile"} {
  }
  air.launch (%d) in (%s4=%c1) attributes {air.pipeline_group = "p", air.pipeline_stage = 2 : i64, air.staging = "accum_in_place"} {
  }
  return
}
