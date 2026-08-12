//===- AIRFusePipelineLaunches.h --------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_FUSE_PIPELINE_LAUNCHES_H
#define AIR_FUSE_PIPELINE_LAUNCHES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRFusePipelineLaunchesPass();

} // namespace air
} // namespace xilinx

#endif // AIR_FUSE_PIPELINE_LAUNCHES_H
