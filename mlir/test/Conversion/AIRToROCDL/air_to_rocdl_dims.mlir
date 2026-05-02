//===- air_to_rocdl_dims.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// REQUIRES: gpu
// RUN: air-opt %s -air-to-rocdl | FileCheck %s

// CHECK-LABEL: func.func @launch_herd_1d
// CHECK: gpu.launch blocks(%{{[^,]+}}, %{{[^,]+}}, %{{[^)]+}}){{.*}}threads(%{{[^,]+}}, %{{[^,]+}}, %{{[^)]+}})
func.func @launch_herd_1d() {
  %c4 = arith.constant 4 : index
  air.launch (%bx) in (%sx = %c4) {
    air.segment @seg {
      %c8 = arith.constant 8 : index
      air.herd @herd tile (%tx) in (%hx = %c8) {
        air.herd_terminator
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}

// CHECK-LABEL: func.func @launch_herd_3d
// CHECK: gpu.launch blocks(%{{[^,]+}}, %{{[^,]+}}, %{{[^)]+}}){{.*}}threads(%{{[^,]+}}, %{{[^,]+}}, %{{[^)]+}})
func.func @launch_herd_3d() {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  air.launch (%bx, %by, %bz) in (%sx = %c2, %sy = %c3, %sz = %c4) {
    air.segment @seg {
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      air.herd @herd tile (%tx, %ty, %tz) in
          (%hx = %c5, %hy = %c6, %hz = %c7) {
        air.herd_terminator
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}
