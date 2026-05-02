//===- air_to_rocdl_invalid_dims.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// REQUIRES: gpu
// RUN: not air-opt %s -air-to-rocdl 2>&1 | FileCheck %s

// CHECK: expected air.launch rank between 1 and 3 for GPU lowering, got 4
func.func @invalid_launch_rank() {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b, %c, %d) in
      (%sa = %c1, %sb = %c1, %sc = %c1, %sd = %c1) {
    air.segment @seg {
      %s1 = arith.constant 1 : index
      air.herd @herd tile (%tx) in (%sx = %s1) {
        air.herd_terminator
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}
