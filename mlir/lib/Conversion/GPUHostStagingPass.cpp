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

struct StagedMemref {
  Value hostMemref;
  Value deviceMemref;
};

static bool shouldStageOperand(Value operand) {
  if (!isa<MemRefType>(operand.getType()))
    return false;
  // v1 policy: AIR kernel operand mutability is not inferred yet, so every
  // host memref is staged as inout. TODO: replace this with read/write
  // inference once AIR-to-GPU preserves operand access information.
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

static Value createStagingAllocation(OpBuilder &builder, Location loc,
                                     Value hostMemref) {
  auto memrefType = cast<MemRefType>(hostMemref.getType());
  auto dynamicSizes = getDynamicSizes(builder, loc, hostMemref);
  auto alloc =
      gpu::AllocOp::create(builder, loc, memrefType, Type(), ValueRange(),
                           dynamicSizes, ValueRange(), false);
  return alloc.getMemref();
}

static void copyInputToDevice(OpBuilder &builder, Location loc,
                              StagedMemref staged) {
  gpu::MemcpyOp::create(builder, loc, Type(), ValueRange(), staged.deviceMemref,
                        staged.hostMemref);
}

static Value copyOutputToHost(OpBuilder &builder, Location loc,
                              StagedMemref staged, Value dependency) {
  SmallVector<Value> dependencies;
  Type asyncTokenType;
  if (dependency) {
    dependencies.push_back(dependency);
    asyncTokenType = dependency.getType();
  }
  auto copy = gpu::MemcpyOp::create(builder, loc, asyncTokenType, dependencies,
                                    staged.hostMemref, staged.deviceMemref);
  return copy.getAsyncToken();
}

static Value cleanupStagingAllocation(OpBuilder &builder, Location loc,
                                      StagedMemref staged, Value dependency) {
  SmallVector<Value> dependencies;
  Type asyncTokenType;
  if (dependency) {
    dependencies.push_back(dependency);
    asyncTokenType = dependency.getType();
  }
  auto dealloc = gpu::DeallocOp::create(builder, loc, asyncTokenType,
                                        dependencies, staged.deviceMemref);
  return dealloc.getAsyncToken();
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
    SmallVector<StagedMemref> stagedMemrefs;
    stagedOperands.reserve(launchFunc.getKernelOperands().size());

    for (Value operand : launchFunc.getKernelOperands()) {
      if (!shouldStageOperand(operand)) {
        stagedOperands.push_back(operand);
        continue;
      }

      Value deviceMemref = createStagingAllocation(builder, loc, operand);
      copyInputToDevice(builder, loc, {operand, deviceMemref});
      stagedOperands.push_back(deviceMemref);
      stagedMemrefs.push_back({operand, deviceMemref});
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
      newLaunch =
          gpu::LaunchFuncOp::create(builder, loc, launchFunc.getKernel(),
                                    launchFunc.getGridSizeOperandValues(),
                                    launchFunc.getBlockSizeOperandValues(),
                                    launchFunc.getDynamicSharedMemorySize(),
                                    stagedOperands, asyncObject, clusterSize);
    } else {
      newLaunch = gpu::LaunchFuncOp::create(
          builder, loc, launchFunc.getKernel(),
          launchFunc.getGridSizeOperandValues(),
          launchFunc.getBlockSizeOperandValues(),
          launchFunc.getDynamicSharedMemorySize(), stagedOperands,
          launchFunc.getAsyncToken() ? launchFunc.getAsyncToken().getType()
                                     : Type(),
          launchFunc.getAsyncDependencies(), clusterSize);
    }

    builder.setInsertionPointAfter(newLaunch);
    Value completionToken = newLaunch.getAsyncToken();
    for (StagedMemref staged : stagedMemrefs)
      completionToken = copyOutputToHost(builder, loc, staged, completionToken);
    for (StagedMemref staged : stagedMemrefs)
      completionToken =
          cleanupStagingAllocation(builder, loc, staged, completionToken);

    if (launchFunc->getNumResults() != 0) {
      if (!completionToken)
        return launchFunc.emitOpError()
               << "expected staged async launch to produce a completion token";
      launchFunc.getAsyncToken().replaceAllUsesWith(completionToken);
    }
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
