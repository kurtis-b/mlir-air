//===- GPUHostStagingPass.cpp ---------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Conversion/GPUHostStagingPass.h"
#include "air/Conversion/GPUPassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::air;

namespace {
#define GEN_PASS_DEF_CONVERTGPUHOSTSTAGING
#include "air/Conversion/GPUPasses.h.inc"

static bool shouldStageOperand(Value operand) {
  if (!isa<MemRefType>(operand.getType()))
    return false;
  Operation *definingOp = operand.getDefiningOp();
  return !definingOp || !isa<gpu::AllocOp>(definingOp);
}

static SmallVector<Value> getDynamicSizes(OpBuilder &builder, Location loc,
                                          Value memref) {
  auto type = cast<MemRefType>(memref.getType());
  SmallVector<Value> dynamicSizes;
  for (auto [idx, dim] : llvm::enumerate(type.getShape()))
    if (ShapedType::isDynamic(dim))
      dynamicSizes.push_back(memref::DimOp::create(builder, loc, memref, idx));
  return dynamicSizes;
}

struct ConvertGPUHostStagingPass
    : public xilinx::air::impl::ConvertGPUHostStagingBase<
          ConvertGPUHostStagingPass> {

  ConvertGPUHostStagingPass() = default;
  ConvertGPUHostStagingPass(const ConvertGPUHostStagingPass &pass) {}

  LogicalResult stageLaunch(gpu::LaunchFuncOp launchFunc) {
    OpBuilder builder(launchFunc);
    Location loc = launchFunc.getLoc();

    SmallVector<Value> stagedOperands;
    SmallVector<std::pair<Value, Value>> stagedMemrefs;
    stagedOperands.reserve(launchFunc.getKernelOperands().size());

    for (Value operand : launchFunc.getKernelOperands()) {
      if (!shouldStageOperand(operand)) {
        stagedOperands.push_back(operand);
        continue;
      }

      auto memrefType = cast<MemRefType>(operand.getType());
      auto dynamicSizes = getDynamicSizes(builder, loc, operand);
      auto alloc = gpu::AllocOp::create(builder, loc, memrefType, Type(),
                                        ValueRange(), dynamicSizes,
                                        ValueRange(), false);
      Value deviceMemref = alloc.getMemref();
      gpu::MemcpyOp::create(builder, loc, Type(), ValueRange(), deviceMemref,
                            operand);
      stagedOperands.push_back(deviceMemref);
      stagedMemrefs.emplace_back(operand, deviceMemref);
    }

    if (stagedMemrefs.empty())
      return success();

    builder.setInsertionPoint(launchFunc);
    std::optional<gpu::KernelDim3> clusterSize =
        launchFunc.hasClusterSize()
            ? std::optional<gpu::KernelDim3>(
                  launchFunc.getClusterSizeOperandValues())
            : std::nullopt;
    gpu::LaunchFuncOp newLaunch;
    if (Value asyncObject = launchFunc.getAsyncObject()) {
      newLaunch = gpu::LaunchFuncOp::create(
          builder, loc, launchFunc.getKernel(),
          launchFunc.getGridSizeOperandValues(),
          launchFunc.getBlockSizeOperandValues(),
          launchFunc.getDynamicSharedMemorySize(), stagedOperands, asyncObject,
          clusterSize);
    } else {
      newLaunch = gpu::LaunchFuncOp::create(
          builder, loc, launchFunc.getKernel(),
          launchFunc.getGridSizeOperandValues(),
          launchFunc.getBlockSizeOperandValues(),
          launchFunc.getDynamicSharedMemorySize(), stagedOperands,
          launchFunc.getAsyncToken() ? launchFunc.getAsyncToken().getType()
                                     : Type(),
          launchFunc.getAsyncDependencies(),
          clusterSize);
    }

    builder.setInsertionPointAfter(newLaunch);
    SmallVector<Value> asyncDependencies;
    if (Value asyncToken = newLaunch.getAsyncToken())
      asyncDependencies.push_back(asyncToken);
    for (auto [hostMemref, deviceMemref] : stagedMemrefs)
      gpu::MemcpyOp::create(builder, loc, Type(), asyncDependencies, hostMemref,
                            deviceMemref);
    for (auto [hostMemref, deviceMemref] : stagedMemrefs)
      gpu::DeallocOp::create(builder, loc, Type(), ValueRange(), deviceMemref);

    if (launchFunc->getNumResults() != 0)
      launchFunc->replaceAllUsesWith(newLaunch->getResults());
    launchFunc.erase();
    return success();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<gpu::LaunchFuncOp> launchFuncs;
    module.walk([&](gpu::LaunchFuncOp launchFunc) {
      if (!launchFunc->getParentOfType<gpu::GPUFuncOp>())
        launchFuncs.push_back(launchFunc);
    });

    for (gpu::LaunchFuncOp launchFunc : launchFuncs)
      if (failed(stageLaunch(launchFunc))) {
        signalPassFailure();
        return;
      }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> xilinx::air::createGPUHostStagingPass() {
  return std::make_unique<ConvertGPUHostStagingPass>();
}
