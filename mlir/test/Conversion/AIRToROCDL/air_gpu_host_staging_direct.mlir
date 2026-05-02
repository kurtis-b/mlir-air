//===- air_gpu_host_staging_direct.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// REQUIRES: gpu
// RUN: air-opt %s -air-gpu-host-staging | FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @dynamic_kernel(%arg0: memref<?x?xf32>) kernel {
      gpu.return
    }
    gpu.func @async_kernel(%arg0: memref<4xf32>) kernel {
      gpu.return
    }
  }

  // CHECK-LABEL: func.func @dynamic_host_staging
  // CHECK: %[[D0:.*]] = memref.dim %arg0, %{{.*}}
  // CHECK: %[[D1:.*]] = memref.dim %arg0, %{{.*}}
  // CHECK: %[[DEV:.*]] = gpu.alloc{{ ?}}(%[[D0]], %[[D1]]) : memref<?x?xf32>
  // CHECK: gpu.memcpy %[[DEV]], %arg0
  // CHECK: gpu.launch_func
  // CHECK-SAME: args(%[[DEV]] : memref<?x?xf32>)
  // CHECK: gpu.memcpy %arg0, %[[DEV]]
  // CHECK: gpu.dealloc %[[DEV]]
  func.func @dynamic_host_staging(%arg0: memref<?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    gpu.launch_func @kernels::@dynamic_kernel
        blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
        args(%arg0 : memref<?x?xf32>)
    return
  }

  // CHECK-LABEL: func.func @async_host_staging
  // CHECK: %[[WAIT:.*]] = gpu.wait async
  // CHECK: %[[DEV:.*]] = gpu.alloc
  // CHECK: gpu.memcpy %[[DEV]], %arg0
  // CHECK: %[[LAUNCH:.*]] = gpu.launch_func async [%[[WAIT]]]
  // CHECK-SAME: args(%[[DEV]] : memref<4xf32>)
  // CHECK: %[[COPY:.*]] = gpu.memcpy async [%[[LAUNCH]]] %arg0, %[[DEV]]
  // CHECK: %[[DEALLOC:.*]] = gpu.dealloc async [%[[COPY]]] %[[DEV]]
  // CHECK: return %[[DEALLOC]]
  func.func @async_host_staging(%arg0: memref<4xf32>) -> !gpu.async.token {
    %c1 = arith.constant 1 : index
    %wait = gpu.wait async
    %launch = gpu.launch_func async [%wait] @kernels::@async_kernel
        blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
        args(%arg0 : memref<4xf32>)
    return %launch : !gpu.async.token
  }
}
