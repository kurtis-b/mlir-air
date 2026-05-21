//===- GPUKernelOutlinePass.cpp --------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------===//
#include "air/Conversion/GPUKernelOutlinePass.h"
#include "air/Conversion/GPUPassDetail.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Util/Util.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
using namespace mlir;
using namespace xilinx;
using namespace xilinx::air;

namespace {
#define GEN_PASS_DEF_CONVERTGPUKERNELOUTLINE
#include "air/Conversion/Passes.h.inc"

static constexpr const char kInt8GemmWmmaAttr[] = "air.gpu.int8_gemm_wmma";
static constexpr const char kInt8GemmVariantAttr[] =
    "air.gpu.int8_gemm_variant";
static constexpr const char kInt8GemmGroupAttr[] =
    "air.gpu.int8_gemm_group_m";
static constexpr const char kInt8GemmDefaultVariant[] = "lds_128x64_wmma4";
static constexpr const char kInt8GemmBPackVariant[] = "lds_128x64_bpack";
static constexpr const char kInt8GemmBPackSwizzleVariant[] =
    "lds_128x64_bpack_swizzle";
static constexpr const char kInt8GemmBPackPipe2Variant[] =
    "lds_128x64_bpack_pipe2";
static constexpr const char kInt8GemmBPackPipe2GroupedVariant[] =
    "lds_128x64_bpack_pipe2_grouped";
static constexpr const char kInt8GemmBPackSwizzleGroupedVariant[] =
    "lds_128x64_bpack_swizzle_grouped";
static constexpr const char kInt8GemmBPackFragVariant[] =
    "lds_128x64_bpack_frag";

static bool isSupportedInt8GemmVariant(StringRef variant) {
  return variant == kInt8GemmDefaultVariant ||
         variant == kInt8GemmBPackVariant ||
         variant == kInt8GemmBPackSwizzleVariant ||
         variant == kInt8GemmBPackPipe2Variant ||
         variant == kInt8GemmBPackPipe2GroupedVariant ||
         variant == kInt8GemmBPackSwizzleGroupedVariant ||
         variant == kInt8GemmBPackFragVariant;
}

static bool isSupportedInt8GemmGroupSize(uint32_t groupSize) {
  return groupSize == 2 || groupSize == 4 || groupSize == 8;
}

SmallVector<mlir::BlockArgument, 4> gpuArgs;
class AffineApplyToSubPattern
    : public mlir::OpRewritePattern<mlir::affine::AffineApplyOp> {
  using OpRewritePattern<mlir::affine::AffineApplyOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::affine::AffineApplyOp affineOp,
                  mlir::PatternRewriter &rewriter) const override {

    for (unsigned j = 0; j < affineOp.getNumOperands(); ++j) {
      mlir::Value nestedOperand = affineOp.getOperand(j);

      if (auto blockArg =
              mlir::dyn_cast_if_present<mlir::BlockArgument>(nestedOperand)) {
        mlir::Block *parentBlock = blockArg.getOwner();

        mlir::Operation *parentOp = parentBlock->getParentOp();
        if (llvm::isa<air::SegmentOp>(parentOp)) {
          unsigned argNumber = blockArg.getArgNumber();
          affineOp.setOperand(j, gpuArgs[j + argNumber]);
        } else if (llvm::isa<air::HerdOp>(parentOp)) {
          unsigned argNumber = blockArg.getArgNumber();
          affineOp.setOperand(j, gpuArgs[j + 3 + argNumber]);
        }
      }
    }

    return mlir::success();
  }
};

class SCFForToSubPattern : public mlir::OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(scf::ForOp forOp,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Region &region = forOp.getRegion();

    mlir::Block &entryBlock = region.front();
    for (mlir::Operation &nestedOp : entryBlock.getOperations()) {
      if (auto storeOp =
              llvm::dyn_cast_if_present<mlir::memref::StoreOp>(nestedOp)) {
        mlir::Value memref = storeOp.getMemRef();
        llvm::SmallVector<mlir::Value> indices(storeOp.getIndices().begin(),
                                               storeOp.getIndices().end());

        if (auto blockArg =
                mlir::dyn_cast_if_present<mlir::BlockArgument>(memref)) {
          mlir::Block *parentBlock = blockArg.getOwner();
          mlir::Operation *parentOp = parentBlock->getParentOp();
          Type operandType = memref.getType();
          if (auto memRefType =
                  mlir::dyn_cast_if_present<MemRefType>(operandType)) {
            unsigned argNumber = blockArg.getArgNumber();
            if (auto segmentOp =
                    mlir::dyn_cast_if_present<air::SegmentOp>(parentOp)) {
              if (auto memRefType =
                      mlir::dyn_cast_if_present<MemRefType>(operandType)) {
                mlir::Value correspondingValue = segmentOp.getKernelOperand(
                    argNumber); // Map back to the original value
                storeOp->setOperand(1, correspondingValue);
              }
            } else if (auto herdOp =
                           mlir::dyn_cast_if_present<air::HerdOp>(parentOp)) {
              if (auto memRefType =
                      mlir::dyn_cast_if_present<MemRefType>(operandType)) {
                mlir::Value correspondingValue =
                    herdOp.getKernelOperand(argNumber - 4);
                storeOp->setOperand(1, correspondingValue);
              }
            } else if (auto launchOp =
                           mlir::dyn_cast_if_present<air::LaunchOp>(parentOp)) {
              if (auto memRefType =
                      mlir::dyn_cast_if_present<MemRefType>(operandType)) {
                mlir::Value correspondingValue =
                    launchOp.getKernelOperand(argNumber - 4);
                storeOp->setOperand(1, correspondingValue);
              }
            }
          }
        }
      }
      for (mlir::Value operand : nestedOp.getOperands()) {
        if (auto opResult = dyn_cast_if_present<mlir::OpResult>(operand)) {
          mlir::Operation *op = operand.getDefiningOp();
          for (unsigned j = 0; j < op->getNumOperands(); ++j) {
            mlir::Value nestedOperand = op->getOperand(j);

            if (auto blockArg = mlir::dyn_cast_if_present<mlir::BlockArgument>(
                    nestedOperand)) {
              mlir::Block *parentBlock = blockArg.getOwner();
              mlir::Operation *parentOp = parentBlock->getParentOp();
              Type operandType = nestedOperand.getType();
              if (auto memRefType =
                      mlir::dyn_cast_if_present<MemRefType>(operandType)) {
                unsigned argNumber = blockArg.getArgNumber();
                if (auto segmentOp =
                        mlir::dyn_cast_if_present<air::SegmentOp>(parentOp)) {
                  if (auto memRefType =
                          mlir::dyn_cast_if_present<MemRefType>(operandType)) {
                    mlir::Value correspondingValue = segmentOp.getKernelOperand(
                        argNumber); // Map back to the original value
                    op->setOperand(j, correspondingValue);
                  }
                } else if (auto herdOp = mlir::dyn_cast_if_present<air::HerdOp>(
                               parentOp)) {
                  if (auto memRefType =
                          mlir::dyn_cast_if_present<MemRefType>(operandType)) {
                    mlir::Value correspondingValue =
                        herdOp.getKernelOperand(argNumber - 4);
                    op->setOperand(j, correspondingValue);
                  }
                } else if (auto launchOp =
                               mlir::dyn_cast_if_present<air::LaunchOp>(
                                   parentOp)) {
                  if (auto memRefType =
                          mlir::dyn_cast_if_present<MemRefType>(operandType)) {
                    mlir::Value correspondingValue =
                        launchOp.getKernelOperand(argNumber - 4);
                    op->setOperand(j, correspondingValue);
                  }
                }

              } else {
                if (llvm::isa<air::SegmentOp>(parentOp)) {
                  unsigned argNumber = blockArg.getArgNumber();
                  op->setOperand(j, gpuArgs[j + argNumber]);
                } else if (llvm::isa<air::HerdOp>(parentOp)) {
                  unsigned argNumber = blockArg.getArgNumber();
                  op->setOperand(j, gpuArgs[j + 3 + argNumber]);
                }
              }
            }
          }
        }
      }
    }

    return mlir::success();
  }
};
class DMAMemcpyToSubPattern
    : public mlir::OpRewritePattern<air::DmaMemcpyNdOp> {
  using OpRewritePattern<air::DmaMemcpyNdOp>::OpRewritePattern;

  void replaceDMAOperand(BlockArgument blockArg,
                         air::DmaMemcpyNdOp dmaOp) const {
    int numOp = -1;
    mlir::Block *parentBlock = blockArg.getOwner();
    Type operandType = blockArg.getType();
    unsigned argNumber = blockArg.getArgNumber(); // Get the argument number

    for (unsigned i = 0; i < dmaOp->getNumOperands(); ++i) {
      if (dmaOp->getOperand(i) == blockArg) {
        numOp = i; // Found the operand index
      }
    }
    // Get the parent operation of the block
    mlir::Operation *parentOp = parentBlock->getParentOp();
    if (auto segmentOp = mlir::dyn_cast_if_present<air::SegmentOp>(parentOp)) {
      if (auto memRefType =
              mlir::dyn_cast_if_present<MemRefType>(operandType)) {
        mlir::Value correspondingValue = segmentOp.getKernelOperand(
            argNumber); // Map back to the original value
        dmaOp.setOperand(numOp, correspondingValue);
      }
    } else if (auto herdOp = mlir::dyn_cast_if_present<air::HerdOp>(parentOp)) {
      if (auto memRefType =
              mlir::dyn_cast_if_present<MemRefType>(operandType)) {
        mlir::Value correspondingValue = herdOp.getKernelOperand(argNumber - 4);
        dmaOp.setOperand(numOp, correspondingValue);
      }
    } else if (auto launchOp =
                   mlir::dyn_cast_if_present<air::LaunchOp>(parentOp)) {
      if (auto memRefType =
              mlir::dyn_cast_if_present<MemRefType>(operandType)) {
        mlir::Value correspondingValue =
            launchOp.getKernelOperand(argNumber - 4);
        dmaOp.setOperand(numOp, correspondingValue);
      }
    }
  }

  mlir::LogicalResult
  matchAndRewrite(air::DmaMemcpyNdOp dmaOp,
                  mlir::PatternRewriter &rewriter) const override {

    mlir::Value sourceMemRef = dmaOp.getSrcMemref();
    mlir::Value dstMemRef = dmaOp.getDstMemref();
    if (auto blockArg =
            mlir::dyn_cast_if_present<BlockArgument>(sourceMemRef)) {
      replaceDMAOperand(blockArg, dmaOp);
    }
    if (auto blockArg = mlir::dyn_cast_if_present<BlockArgument>(dstMemRef)) {
      replaceDMAOperand(blockArg, dmaOp);
    }

    return mlir::success();
  }
};

struct ConvertGPUKernelOutlinePass
    : public xilinx::air::impl::ConvertGPUKernelOutlineBase<
          ConvertGPUKernelOutlinePass> {

  ConvertGPUKernelOutlinePass() = default;
  ConvertGPUKernelOutlinePass(const ConvertGPUKernelOutlinePass &pass) {}
  SmallVector<Value, 4> blkIdx;
  SmallVector<Value, 4> gridIdx;

  static bool hasStaticMemRef(Value value, ArrayRef<int64_t> shape,
                              unsigned elementWidth) {
    auto type = dyn_cast<MemRefType>(value.getType());
    return type && type.getShape() == shape &&
           type.getElementType().isInteger(elementWidth);
  }

  static Value cidx(OpBuilder &builder, Location loc, int64_t value) {
    return arith::ConstantIndexOp::create(builder, loc, value);
  }

  static Value ci32(OpBuilder &builder, Location loc, int32_t value) {
    return arith::ConstantIntOp::create(builder, loc, value, 32);
  }

  static Value loadPackedContiguousI8x16AsI32x4(OpBuilder &builder,
                                                Location loc, Value memref,
                                                Value row, Value colBase) {
    auto i8Type = IntegerType::get(builder.getContext(), 8);
    auto i32Type = builder.getI32Type();
    auto vec16i8 = VectorType::get({16}, i8Type);
    auto vec4i32 = VectorType::get({4}, i32Type);
    Value bytes = vector::LoadOp::create(builder, loc, vec16i8, memref,
                                         ValueRange{row, colBase});
    return vector::BitCastOp::create(builder, loc, vec4i32, bytes);
  }

  static LogicalResult emitInt8GemmWmmaKernel(gpu::GPUFuncOp func,
                                             StringRef variant,
                                             uint32_t groupSize) {
    Block &body = func.getBody().front();
    if (func.getFunctionType().getNumInputs() != 3)
      return func.emitOpError(kInt8GemmWmmaAttr)
             << " expects exactly three kernel arguments";

    Value a = body.getArgument(0);
    Value b = body.getArgument(1);
    Value c = body.getArgument(2);
    if (!hasStaticMemRef(a, {1024, 1024}, 8) ||
        !hasStaticMemRef(b, {1024, 1024}, 8) ||
        !hasStaticMemRef(c, {1024, 1024}, 32))
      return func.emitOpError(kInt8GemmWmmaAttr)
             << " only supports memref<1024x1024xi8>, "
                "memref<1024x1024xi8>, memref<1024x1024xi32>";
    if (!isSupportedInt8GemmVariant(variant))
      return func.emitOpError(kInt8GemmWmmaAttr)
             << " unsupported variant '" << variant << "'";
    if (!isSupportedInt8GemmGroupSize(groupSize))
      return func.emitOpError(kInt8GemmWmmaAttr)
             << " unsupported group size '" << groupSize << "'";

    bool packedB = variant != kInt8GemmDefaultVariant;
    bool swizzledLds = variant == kInt8GemmBPackSwizzleVariant ||
                       variant == kInt8GemmBPackSwizzleGroupedVariant;
    bool pipelined = variant == kInt8GemmBPackPipe2Variant ||
                     variant == kInt8GemmBPackPipe2GroupedVariant;
    bool groupedBlocks = variant == kInt8GemmBPackPipe2GroupedVariant ||
                         variant == kInt8GemmBPackSwizzleGroupedVariant;
    bool directBFragments = variant == kInt8GemmBPackFragVariant;

    Location loc = func.getLoc();
    MLIRContext *ctx = func.getContext();
    while (!body.getOperations().empty())
      body.back().erase();

    auto i8Type = IntegerType::get(ctx, 8);
    auto i32Type = IntegerType::get(ctx, 32);
    auto aLdsType = MemRefType::get({128, 64}, i8Type, {}, 3);
    auto bLdsType = MemRefType::get({64, 64}, i8Type, {}, 3);
    Value aLds = func.addWorkgroupAttribution(aLdsType, loc);
    Value bLds;
    Value aLdsNext;
    Value bLdsNext;
    if (!directBFragments)
      bLds = func.addWorkgroupAttribution(bLdsType, loc);
    if (pipelined) {
      aLdsNext = func.addWorkgroupAttribution(aLdsType, loc);
      bLdsNext = func.addWorkgroupAttribution(bLdsType, loc);
    }

    OpBuilder builder(ctx);
    builder.setInsertionPointToStart(&body);

    Value c0 = cidx(builder, loc, 0);
    Value c2 = cidx(builder, loc, 2);
    Value c4 = cidx(builder, loc, 4);
    Value cGroupM = cidx(builder, loc, groupSize);
    Value cGroupTileCount = cidx(builder, loc, groupSize * 16);
    Value c16 = cidx(builder, loc, 16);
    Value c32 = cidx(builder, loc, 32);
    Value c64 = cidx(builder, loc, 64);
    Value c128 = cidx(builder, loc, 128);
    Value c1024 = cidx(builder, loc, 1024);
    Value c4096 = cidx(builder, loc, 4096);

    Value physicalBlockX = gpu::BlockIdOp::create(
        builder, loc, builder.getIndexType(), gpu::Dimension::x);
    Value physicalBlockY = gpu::BlockIdOp::create(
        builder, loc, builder.getIndexType(), gpu::Dimension::y);
    Value tx = gpu::ThreadIdOp::create(builder, loc, builder.getIndexType(),
                                       gpu::Dimension::x);

    Value blockX = physicalBlockX;
    Value blockY = physicalBlockY;
    if (groupedBlocks) {
      Value linearBlock = arith::AddIOp::create(
          builder, loc, arith::MulIOp::create(builder, loc, physicalBlockY, c16),
          physicalBlockX);
      Value groupId = arith::DivSIOp::create(builder, loc, linearBlock,
                                             cGroupTileCount);
      Value pidInGroup = arith::RemSIOp::create(builder, loc, linearBlock,
                                                cGroupTileCount);
      Value firstM = arith::MulIOp::create(builder, loc, groupId, cGroupM);
      blockY = arith::AddIOp::create(
          builder, loc, firstM,
          arith::RemSIOp::create(builder, loc, pidInGroup, cGroupM));
      blockX = arith::DivSIOp::create(builder, loc, pidInGroup, cGroupM);
    }

    Value wave = arith::DivSIOp::create(builder, loc, tx, c32);
    Value lane = arith::RemSIOp::create(builder, loc, tx, c32);
    Value lane16 = arith::RemSIOp::create(builder, loc, lane, c16);
    Value laneHalf = arith::DivSIOp::create(builder, loc, lane, c16);
    Value blockRowBase = arith::MulIOp::create(builder, loc, blockY, c128);
    Value blockColBase = arith::MulIOp::create(builder, loc, blockX, c64);
    Value waveRowBase = arith::MulIOp::create(builder, loc, wave, c16);
    Value tileRow = arith::AddIOp::create(builder, loc, blockRowBase,
                                          waveRowBase);
    Value tileCol = blockColBase;

    auto vec8i32 = VectorType::get({8}, i32Type);
    Value initAcc = arith::ConstantOp::create(builder, loc,
                                              builder.getZeroAttr(vec8i32));
    auto vec16i8 = VectorType::get({16}, i8Type);
    Value copyIdx16 = arith::MulIOp::create(builder, loc, tx, c16);

    auto createWmma = [&](Value aPacked, Value bPacked, Value acc) {
      OperationState wmmaState(loc, "rocdl.wmma.i32.16x16x16.iu8");
      wmmaState.addTypes(vec8i32);
      wmmaState.addOperands({aPacked, bPacked, acc});
      wmmaState.addAttribute("signA", builder.getBoolAttr(true));
      wmmaState.addAttribute("signB", builder.getBoolAttr(true));
      wmmaState.addAttribute("clamp", builder.getBoolAttr(false));
      return builder.create(wmmaState)->getResult(0);
    };

    auto swizzleColBase = [&](Value row, Value colBase) -> Value {
      Value group = arith::DivSIOp::create(builder, loc, colBase, c16);
      Value rowGroup = arith::RemSIOp::create(builder, loc, row, c4);
      Value swizzledGroup =
          arith::XOrIOp::create(builder, loc, rowGroup, group);
      return arith::MulIOp::create(builder, loc, swizzledGroup, c16);
    };

    auto maybeSwizzledColBase = [&](Value row, Value colBase) -> Value {
      return swizzledLds ? swizzleColBase(row, colBase) : colBase;
    };

    auto storeContiguousI8x16 = [&](Value memref, Value row, Value colBase,
                                    Value bytes) {
      vector::StoreOp::create(builder, loc, bytes, memref,
                              ValueRange{row,
                                         maybeSwizzledColBase(row, colBase)});
    };

    auto loadPackedFromLds = [&](Value memref, Value row, Value colBase) {
      return loadPackedContiguousI8x16AsI32x4(
          builder, loc, memref, row, maybeSwizzledColBase(row, colBase));
    };

    auto copyATile = [&](Value dst, Value kBase, Value linearIndex) {
      Value copyARow = arith::DivSIOp::create(builder, loc, linearIndex, c64);
      Value copyACol = arith::RemSIOp::create(builder, loc, linearIndex, c64);
      Value globalARow =
          arith::AddIOp::create(builder, loc, blockRowBase, copyARow);
      Value globalACol = arith::AddIOp::create(builder, loc, kBase, copyACol);
      Value aVec = vector::LoadOp::create(builder, loc, vec16i8, a,
                                          ValueRange{globalARow, globalACol});
      storeContiguousI8x16(dst, copyARow, copyACol, aVec);
    };

    auto copyBTile = [&](Value dst, Value kBase, Value linearIndex) {
      if (directBFragments)
        return;
      if (packedB) {
        Value bTileCol = arith::DivSIOp::create(builder, loc, linearIndex, c64);
        Value bTileK = arith::RemSIOp::create(builder, loc, linearIndex, c64);
        Value globalBRow =
            arith::AddIOp::create(builder, loc, blockColBase, bTileCol);
        Value globalBCol = arith::AddIOp::create(builder, loc, kBase, bTileK);
        Value bVec = vector::LoadOp::create(
            builder, loc, vec16i8, b, ValueRange{globalBRow, globalBCol});
        storeContiguousI8x16(dst, bTileCol, bTileK, bVec);
        return;
      }

      Value bRow = arith::DivSIOp::create(builder, loc, linearIndex, c64);
      Value bCol = arith::RemSIOp::create(builder, loc, linearIndex, c64);
      Value globalBRow = arith::AddIOp::create(builder, loc, kBase, bRow);
      Value globalBCol =
          arith::AddIOp::create(builder, loc, blockColBase, bCol);
      Value bVec = vector::LoadOp::create(builder, loc, vec16i8, b,
                                          ValueRange{globalBRow, globalBCol});
      for (int byte = 0; byte < 16; ++byte) {
        Value byteIdx = cidx(builder, loc, byte);
        Value ldsRow = arith::AddIOp::create(builder, loc, bCol, byteIdx);
        Value value = vector::ExtractOp::create(builder, loc, bVec, byte);
        memref::StoreOp::create(builder, loc, value, dst,
                                ValueRange{ldsRow, bRow});
      }
    };

    auto copyTile = [&](Value aDst, Value bDst, Value kBase) {
      copyATile(aDst, kBase, copyIdx16);
      copyATile(aDst, kBase,
                arith::AddIOp::create(builder, loc, copyIdx16, c4096));
      copyBTile(bDst, kBase, copyIdx16);
    };

    auto computeTile = [&](Value aSrc, Value bSrc, Value kBase,
                           SmallVectorImpl<Value> &accs) {
      for (int kk = 0; kk < 64; kk += 16) {
        Value kkOffset = cidx(builder, loc, kk);
        Value aRow = arith::AddIOp::create(builder, loc, waveRowBase, lane16);
        Value aPacked = loadPackedFromLds(aSrc, aRow, kkOffset);
        for (int col = 0; col < 64; col += 16) {
          Value bRow = col == 0
                           ? lane16
                           : arith::AddIOp::create(builder, loc, lane16,
                                                  cidx(builder, loc, col));
          Value bPacked;
          if (directBFragments) {
            Value globalBRow =
                arith::AddIOp::create(builder, loc, blockColBase, bRow);
            Value globalBCol =
                arith::AddIOp::create(builder, loc, kBase, kkOffset);
            bPacked = loadPackedContiguousI8x16AsI32x4(
                builder, loc, b, globalBRow, globalBCol);
          } else {
            bPacked = loadPackedFromLds(bSrc, bRow, kkOffset);
          }
          accs[col / 16] = createWmma(aPacked, bPacked, accs[col / 16]);
        }
      }
    };

    SmallVector<Value, 4> finalAccs;
    finalAccs.reserve(4);
    if (pipelined) {
      finalAccs.assign(4, initAcc);
      copyTile(aLds, bLds, c0);
      gpu::BarrierOp::create(builder, loc);
      for (int tile = 0; tile < 16; ++tile) {
        Value currentA = (tile % 2 == 0) ? aLds : aLdsNext;
        Value currentB = (tile % 2 == 0) ? bLds : bLdsNext;
        Value nextA = (tile % 2 == 0) ? aLdsNext : aLds;
        Value nextB = (tile % 2 == 0) ? bLdsNext : bLds;
        Value kBase = cidx(builder, loc, tile * 64);
        if (tile + 1 < 16)
          copyTile(nextA, nextB, cidx(builder, loc, (tile + 1) * 64));
        computeTile(currentA, currentB, kBase, finalAccs);
        if (tile + 1 < 16)
          gpu::BarrierOp::create(builder, loc);
      }
    } else {
      SmallVector<Value, 4> initAccs(4, initAcc);
      auto kLoop = scf::ForOp::create(builder, loc, c0, c1024, c64,
                                      ValueRange(initAccs));
      builder.setInsertionPointToStart(kLoop.getBody());
      Value k = kLoop.getInductionVar();
      copyTile(aLds, bLds, k);
      gpu::BarrierOp::create(builder, loc);

      SmallVector<Value, 4> nextAccs;
      nextAccs.reserve(4);
      for (unsigned i = 0; i < 4; ++i)
        nextAccs.push_back(kLoop.getRegionIterArg(i));
      computeTile(aLds, bLds, k, nextAccs);

      gpu::BarrierOp::create(builder, loc);
      scf::YieldOp::create(builder, loc, ValueRange(nextAccs));

      builder.setInsertionPointAfter(kLoop);
      for (unsigned i = 0; i < 4; ++i)
        finalAccs.push_back(kLoop.getResult(i));
    }

    for (int element = 0; element < 8; ++element) {
      Value elemIdx = cidx(builder, loc, element);
      Value rowInTile = arith::AddIOp::create(
          builder, loc,
          arith::MulIOp::create(builder, loc, elemIdx, c2), laneHalf);
      Value dstRow = arith::AddIOp::create(builder, loc, tileRow, rowInTile);
      Value dstColBase = arith::AddIOp::create(builder, loc, tileCol, lane16);
      for (int col = 0; col < 64; col += 16) {
        Value dstCol = col == 0
                           ? dstColBase
                           : arith::AddIOp::create(builder, loc, dstColBase,
                                                  cidx(builder, loc, col));
        Value value = vector::ExtractOp::create(builder, loc,
                                                finalAccs[col / 16], element);
        memref::StoreOp::create(builder, loc, value, c,
                                ValueRange{dstRow, dstCol});
      }
    }

    gpu::ReturnOp::create(builder, loc);
    return success();
  }

  static DenseI32ArrayAttr maybeConstantDimsAttr(gpu::KernelDim3 dims) {
    SmallVector<int32_t, 3> constants;
    MLIRContext *ctx = dims.x.getContext();
    for (Value v : {dims.x, dims.y, dims.z}) {
      APInt constValue;
      if (!matchPattern(v, m_ConstantInt(&constValue)))
        return nullptr;
      // In the event someone called for a too-large block or grid dimension,
      // don't set bounds as it is likely to cause more confusing behavior.
      if (constValue.ugt(std::numeric_limits<uint32_t>::max()))
        return nullptr;
      constants.push_back(
          constValue.getLimitedValue(std::numeric_limits<uint32_t>::max()));
    }
    return DenseI32ArrayAttr::get(ctx, constants);
  }

  template <typename OpTy>
  static void createForAllDimensions(OpBuilder &builder, Location loc,
                                     SmallVectorImpl<Value> &values) {
    for (auto dim : {gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z})
      values.push_back(OpTy::create(builder, loc, builder.getIndexType(), dim));
  }

  /// Adds operations generating block/thread ids and grid/block dimensions at
  /// the beginning of the `launchFuncOpBody` region. Add mapping from argument
  /// in entry block of `launchOpBody`, to the corresponding result value of the
  /// added operations.
  static void injectGpuIndexOperations(Location loc, Region &launchFuncOpBody,
                                       Region &launchOpBody, IRMapping &map,
                                       bool hasCluster = false) {
    OpBuilder builder(loc->getContext());
    Block &firstBlock = launchOpBody.front();
    builder.setInsertionPointToStart(&launchFuncOpBody.front());
    SmallVector<Value> indexOps;
    // The order is important here, as it must match the order of the arguments
    createForAllDimensions<gpu::BlockIdOp>(builder, loc, indexOps);
    createForAllDimensions<gpu::ThreadIdOp>(builder, loc, indexOps);
    createForAllDimensions<gpu::GridDimOp>(builder, loc, indexOps);
    createForAllDimensions<gpu::BlockDimOp>(builder, loc, indexOps);
    if (hasCluster) {
      createForAllDimensions<gpu::ClusterIdOp>(builder, loc, indexOps);
      createForAllDimensions<gpu::ClusterDimOp>(builder, loc, indexOps);
    }
    // Replace the leading 12 function args with the respective thread/block
    // index operations. Iterate backwards since args are erased and indices
    // change.
    for (const auto &indexOp : enumerate(indexOps))
      map.map(firstBlock.getArgument(indexOp.index()), indexOp.value());
  }

  static gpu::GPUFuncOp
  outlineKernelFuncImpl(gpu::LaunchOp launchOp, StringRef kernelFnName,
                        SetVector<Value> &operands,
                        SetVector<Value> &filteredOperands) {
    Location loc = launchOp.getLoc();
    // Create a builder with no insertion point, insertion will happen
    // separately due to symbol table manipulation.
    OpBuilder builder(launchOp.getContext());
    Region &launchOpBody = launchOp.getBody();

    // Identify uses from values defined outside of the scope of the launch
    // operation.
    mlir::getUsedValuesDefinedAbove(launchOpBody, operands);

    DenseMap<Value, Attribute> constantValues;

    for (Value operand : operands) {
      if (auto defOp = operand.getDefiningOp()) {
        if (auto constOp = dyn_cast_if_present<arith::ConstantOp>(defOp)) {
          // Record the constant value for later inlining
          constantValues[operand] = constOp.getValue();
          continue; // Don't pass this constant as an argument
        }
      }
      filteredOperands.insert(operand); // Only non-constant operands are passed
    }

    // Create the gpu.func operation.
    SmallVector<Type, 4> kernelOperandTypes;
    kernelOperandTypes.reserve(filteredOperands.size());
    for (Value operand : filteredOperands) {
      kernelOperandTypes.push_back(operand.getType());
    }
    FunctionType type =
        FunctionType::get(launchOp.getContext(), kernelOperandTypes, {});
    auto outlinedFunc = gpu::GPUFuncOp::create(
        builder, loc, kernelFnName, type,
        TypeRange(ValueRange(launchOp.getWorkgroupAttributions())),
        TypeRange(ValueRange(launchOp.getPrivateAttributions())));
    outlinedFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                          builder.getUnitAttr());

    // If we can infer bounds on the grid and/or block sizes from the arguments
    // to the launch op, propagate them to the generated kernel. This is safe
    // because multiple launches with the same body are not deduplicated.
    if (auto blockBounds =
            maybeConstantDimsAttr(launchOp.getBlockSizeOperandValues()))
      outlinedFunc.setKnownBlockSizeAttr(blockBounds);
    if (auto gridBounds =
            maybeConstantDimsAttr(launchOp.getGridSizeOperandValues()))
      outlinedFunc.setKnownGridSizeAttr(gridBounds);

    IRMapping map;

    // Map the arguments corresponding to the launch parameters like blockIdx,
    // threadIdx, etc. If cluster is present, then we also generate clusterIdx
    // and clusterDim.
    Region &outlinedFuncBody = outlinedFunc.getBody();
    injectGpuIndexOperations(loc, outlinedFuncBody, launchOpBody, map,
                             launchOp.hasClusterSize());

    // Map memory attributions from the LaunOp op to the GPUFuncOp attributions.
    for (const auto &[launchArg, funcArg] :
         llvm::zip(launchOp.getWorkgroupAttributions(),
                   outlinedFunc.getWorkgroupAttributions()))
      map.map(launchArg, funcArg);
    for (const auto &[launchArg, funcArg] :
         llvm::zip(launchOp.getPrivateAttributions(),
                   outlinedFunc.getPrivateAttributions()))
      map.map(launchArg, funcArg);

    // Map arguments from gpu.launch region to the arguments of the gpu.func
    // operation.
    Block &entryBlock = outlinedFuncBody.front();
    for (const auto &operand : enumerate(filteredOperands))
      map.map(operand.value(), entryBlock.getArgument(operand.index()));

    for (auto [originalValue, constAttr] : constantValues) {
      // Get the location of original constant for accurate IR tracing
      Location constLoc = originalValue.getLoc();
      OpBuilder constBuilder =
          OpBuilder::atBlockBegin(&outlinedFunc.getBody().front());

      Value newConst = arith::ConstantOp::create(
          constBuilder, constLoc, llvm::cast<TypedAttr>(constAttr));

      // Update the mapping so that cloned uses map to new constant
      map.map(originalValue, newConst);
    }

    launchOpBody.cloneInto(&outlinedFuncBody, map);

    // Replace the terminator op with returns.
    for (Block &block : launchOpBody) {
      Block *clonedBlock = map.lookup(&block);
      auto terminator =
          dyn_cast_if_present<gpu::TerminatorOp>(clonedBlock->getTerminator());
      if (!terminator)
        continue;
      OpBuilder replacer(terminator);
      gpu::ReturnOp::create(replacer, terminator->getLoc());
      terminator->erase();
    }

    // Splice now the entry block of the gpu.launch operation at the end of the
    // gpu.func entry block and erase the redundant block.
    Block *clonedLaunchOpEntry = map.lookup(&launchOpBody.front());
    entryBlock.getOperations().splice(entryBlock.getOperations().end(),
                                      clonedLaunchOpEntry->getOperations());
    clonedLaunchOpEntry->erase();

    return outlinedFunc;
  }

  static void convertToLaunchFuncOp(gpu::LaunchOp launchOp,
                                    gpu::GPUFuncOp kernelFunc,
                                    ValueRange operands) {
    OpBuilder builder(launchOp);
    // The launch op has an optional dynamic shared memory size. If it doesn't
    // exist, we use zero.
    Value asyncToken = launchOp.getAsyncToken();
    std::optional<gpu::KernelDim3> clusterSize =
        launchOp.getClusterSizeOperandValues();
    auto launchFunc = gpu::LaunchFuncOp::create(
        builder, launchOp.getLoc(), kernelFunc,
        launchOp.getGridSizeOperandValues(),
        launchOp.getBlockSizeOperandValues(),
        launchOp.getDynamicSharedMemorySize(), operands,
        asyncToken ? asyncToken.getType() : nullptr,
        launchOp.getAsyncDependencies(), clusterSize);
    launchOp.replaceAllUsesWith(launchFunc);
    launchOp.erase();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    PassManager pm(module.getContext());
    OpBuilder builder(module.getContext());

    pm.addPass(mlir::createLowerAffinePass());
    pm.addPass(mlir::createConvertLinalgToLoopsPass());
    pm.addPass(mlir::createSCFToControlFlowPass());

    if (failed(pm.run(module))) {
      module.emitError("Sub-pipeline failed in LowerRockOpsToGPUPass");
      signalPassFailure();
    }

    SymbolTable symbolTable(getOperation());
    module.walk([&](func::FuncOp func) {
      func.walk([&](gpu::LaunchOp launchOp) {
        ModuleOp op = getOperation();
        MLIRContext *ctx = op.getContext();
        OpBuilder b(ctx);
        Location loc = op.getLoc();

        // Annotate this module as a container module.
        op->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
                    UnitAttr::get(ctx));

        SetVector<Value> operands;
        SetVector<Value> filteredOperands;
        std::string gfname = func.getName().str();
        gfname += "_module";
        // create a GPUModuleOp in case the GPU module specified does not exist.
        auto gpuModule = gpu::GPUModuleOp::create(b, loc, gfname);

        // add the GPUModuleOp into the symbol table.
        SymbolTable symbolTable(op);
        symbolTable.insert(gpuModule);

        gpu::GPUFuncOp outlinedFunc =
            outlineKernelFuncImpl(launchOp, gfname, operands, filteredOperands);
        bool useInt8Wmma = launchOp->hasAttr(kInt8GemmWmmaAttr);
        StringRef variant = clInt8GemmVariant;
        if (auto variantAttr =
                launchOp->getAttrOfType<StringAttr>(kInt8GemmVariantAttr))
          variant = variantAttr.getValue();
        uint32_t groupSize = clInt8GemmGroupSize;
        if (auto groupAttr =
                launchOp->getAttrOfType<IntegerAttr>(kInt8GemmGroupAttr))
          groupSize = groupAttr.getInt();
        if (useInt8Wmma) {
          if (!isSupportedInt8GemmVariant(variant)) {
            launchOp.emitOpError(kInt8GemmWmmaAttr)
                << " unsupported variant '" << variant << "'";
            signalPassFailure();
            return;
          }
          if (!isSupportedInt8GemmGroupSize(groupSize)) {
            launchOp.emitOpError(kInt8GemmWmmaAttr)
                << " unsupported group size '" << groupSize << "'";
            signalPassFailure();
            return;
          }
          outlinedFunc->setAttr(kInt8GemmWmmaAttr, UnitAttr::get(ctx));
          outlinedFunc->setAttr(kInt8GemmVariantAttr,
                                builder.getStringAttr(variant));
          outlinedFunc->setAttr(kInt8GemmGroupAttr,
                                builder.getI32IntegerAttr(groupSize));
        }
        SymbolTable gpuModuleSymbolTable(gpuModule);
        // insert the GPUFuncOp into GPUModuleOp.
        gpuModuleSymbolTable.insert(outlinedFunc);
        if (useInt8Wmma &&
            failed(emitInt8GemmWmmaKernel(outlinedFunc, variant, groupSize))) {
          signalPassFailure();
          return;
        }
        convertToLaunchFuncOp(launchOp, outlinedFunc,
                              filteredOperands.getArrayRef());
      });
    });
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::affine::AffineDialect>(),
        registry.insert<mlir::scf::SCFDialect>(),
        registry.insert<mlir::arith::ArithDialect>(),
        registry.insert<mlir::cf::ControlFlowDialect>(),
        registry.insert<mlir::memref::MemRefDialect>(),
        registry.insert<mlir::func::FuncDialect>(),
        registry.insert<mlir::vector::VectorDialect>(), // If used anywhere
        registry.insert<mlir::ROCDL::ROCDLDialect>();
    registry.insert<mlir::LLVM::LLVMDialect>();
    registry.insert<mlir::gpu::GPUDialect>();
    registry.insert<scf::SCFDialect>();
  }

  // Function to print detailed information about a Value (including block
  // arguments)
  void printValueDetails(Value val) {
    if (auto blockArg = mlir::dyn_cast_if_present<BlockArgument>(val)) {
      llvm::outs() << "Block argument: index=" << blockArg.getArgNumber()
                   << " type=" << blockArg.getType() << "\n";
    } else {
      llvm::outs() << "Operation result: " << val << "\n";
    }
  }

  void deleteAirSegment(air::LaunchOp launchOp, OpBuilder &builder,
                        gpu::LaunchOp gpuLaunchOp) {
    launchOp.walk([&](Operation *childOp) {
      if (auto segmentOp =
              dyn_cast_if_present<xilinx::air::SegmentOp>(childOp)) {
        if (!segmentOp.getRegion().empty()) {
          Block &segmentBlock =
              segmentOp.getRegion()
                  .front(); // Get the first block (body) of the herd operation

          for (auto &operation :
               llvm::make_early_inc_range(segmentBlock.without_terminator())) {
            operation.moveBefore(segmentOp);
          }
          segmentBlock.getTerminator()->erase(); // Erase the terminator
          segmentOp.erase();                     // Erase the herd operation
        }
      }
    });
  }

  void deleteAirHerd(xilinx::air::SegmentOp segmentOp, OpBuilder &builder,
                     gpu::LaunchOp gpuLaunchOp) {
    // Traverse the children of the segment to find the air.herd
    segmentOp.walk([&](Operation *childOp) {
      if (auto herdOp = dyn_cast_if_present<xilinx::air::HerdOp>(childOp)) {
        // Check if the herd operation has a region (body)
        if (!herdOp.getRegion().empty()) {
          Block &herdBlock =
              herdOp.getRegion()
                  .front(); // Get the first block (body) of the herd operation

          for (auto &operation :
               llvm::make_early_inc_range(herdBlock.without_terminator())) {
            operation.moveBefore(herdOp);
          }
          herdBlock.getTerminator()->erase();
          herdOp.erase();
        }
      }
    });
  }

  // Convert air.launch -> gpu.launch with thread block tuning
  gpu::LaunchOp convertLaunchToGPULaunch(xilinx::air::LaunchOp launchOp,
                                         OpBuilder &builder, Value gridXVal,
                                         Value gridYVal) {
    Location loc = launchOp.getLoc();
    // Define grid and block sizes (modify these values as needed for your use
    // case)
    int64_t gridSizeZ = 1;
    int64_t blockSizeZ = 1;

    builder.setInsertionPoint(launchOp);
    Value gridZVal = arith::ConstantOp::create(builder, loc,
                                               builder.getIndexAttr(gridSizeZ));
    Value blockZVal = arith::ConstantOp::create(
        builder, loc, builder.getIndexAttr(blockSizeZ));

    Operation *blockXValOp = blkIdx[0].getDefiningOp();
    blockXValOp->moveBefore(launchOp);
    Operation *blockYValOp = blkIdx[1].getDefiningOp();
    blockYValOp->moveBefore(launchOp);

    // Create the gpu.launch operation
    auto gpuLaunchOp =
        gpu::LaunchOp::create(builder, loc, gridXVal, gridYVal, gridZVal,
                              blkIdx[0], blkIdx[1], blockZVal);

    // Get thread indices for use within the gpu.launch body
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&gpuLaunchOp.getBody().front());

    gpu::TerminatorOp::create(builder, loc);
    return gpuLaunchOp;
  }

  void hoistAlloc(gpu::LaunchOp launchOp, OpBuilder &builder) {
    Location loc = launchOp.getLoc();
    launchOp.walk([&](memref::AllocOp allocOp) {
      // workgroup
      MemRefType memRefType = allocOp.getType();
      if (air::isL2(memRefType)) {
        mlir::Type elementType = memRefType.getElementType();
        llvm::ArrayRef<int64_t> shape = memRefType.getShape();

        // Create a new MemRefType with the same shape, element type, but with a
        // different memory space
        mlir::MemRefType newType =
            mlir::MemRefType::get(shape, elementType, /*affineMap=*/{}, 3);

        auto wg = launchOp.addWorkgroupAttribution(newType, loc);
        allocOp.replaceAllUsesWith(wg); // Replace row with globalRow
        allocOp.erase();
      } else if (air::isL1(memRefType)) {
        mlir::Type elementType = memRefType.getElementType();
        llvm::ArrayRef<int64_t> shape = memRefType.getShape();

        // Create a new MemRefType with the same shape, element type, but with a
        // different memory space
        mlir::MemRefType newType =
            mlir::MemRefType::get(shape, elementType, /*affineMap=*/{}, 5);
        auto wg = launchOp.addPrivateAttribution(newType, loc);
        allocOp.replaceAllUsesWith(wg); // Replace row with globalRow
        allocOp.erase();
      }
    });
    launchOp.walk([&](memref::DeallocOp deallocOp) { deallocOp.erase(); });
  }

  // Convert air.dma_memcpy_nd -> llvm.memcpy or gpu.global_to_shared for memory
  // operations
  void convertDMAToGPUMemcpy(xilinx::air::DmaMemcpyNdOp dmaMemcpyOp,
                             OpBuilder &builder) {
    builder.setInsertionPointAfter(dmaMemcpyOp);

    // Extract the operands (memrefs)
    Value destMemref = dmaMemcpyOp.getDstMemref(); // Destination memref
    Value srcMemref = dmaMemcpyOp.getSrcMemref();  // Source memref
    SmallVector<Value, 4> src_offsets = dmaMemcpyOp.getSrcOffsets();
    SmallVector<Value, 4> dst_offsets = dmaMemcpyOp.getDstOffsets();
    SmallVector<Value, 4> src_sizes = dmaMemcpyOp.getSrcSizes();
    SmallVector<Value, 4> dst_sizes = dmaMemcpyOp.getDstSizes();
    SmallVector<Value, 4> src_strides = dmaMemcpyOp.getSrcStrides();
    SmallVector<Value, 4> dst_strides = dmaMemcpyOp.getDstStrides();

    // Create SCF loops to simulate the memory copy
    // We'll assume a 2D memory region and use sizes for both dimensions

    // Extract the precomputed indices from the dma_memcpy_nd operation (i.e.,
    // %1 and %0) Constants for SCF loop bounds and steps
    Value c0 = arith::ConstantIndexOp::create(builder, dmaMemcpyOp.getLoc(), 0);
    Value c1 = arith::ConstantIndexOp::create(builder, dmaMemcpyOp.getLoc(), 1);
    // Extract the number of dimensions from the memref type
    auto srcMemrefType =
        mlir::dyn_cast_or_null<MemRefType>(srcMemref.getType());
    auto destMemrefType =
        mlir::dyn_cast_or_null<MemRefType>(destMemref.getType());
    int64_t destRank = destMemrefType.getRank();

    // Generate SCF loops for the destination memref (64x64xi32)
    SmallVector<scf::ForOp, 4> loops;
    // Generate loops for each dimension of the destination
    for (int i = 0; i < destRank; ++i) {
      Value loopStart = c0; // Start of loop (e.g., 0)
      Value loopEnd =
          (src_sizes.size() > 0)
              ? src_sizes[i]
              : dst_sizes[i]; // Determine the loop end based on the dimensions
      Value loopStep = c1;    // Step size (e.g., 1)

      // Create a loop for each dimension
      auto loop = scf::ForOp::create(builder, dmaMemcpyOp.getLoc(), loopStart,
                                     loopEnd, loopStep);
      loops.push_back(loop);

      // Set insertion point to inside the loop body
      builder.setInsertionPointToStart(loop.getBody());
    }

    // Generate load/store operations inside the innermost loop
    builder.setInsertionPointToStart(loops.back().getBody());
    SmallVector<Value, 4> indices;

    // Collect the loop induction variables to create indices
    for (int i = 0; i < destRank; ++i) {
      indices.push_back(loops[i].getInductionVar());
    }

    // Check for dimension mismatch and adjust indices accordingly
    // Handle the case where the destination is larger than the source
    if (destMemrefType.getShape()[0] > srcMemrefType.getShape()[0] &&
        destMemrefType.getShape()[1] > srcMemrefType.getShape()[1]) {
      // Both x and y dimensions are larger in the destination, need to adjust
      // for chunks
      Value idxRow = arith::AddIOp::create(
          builder, dmaMemcpyOp.getLoc(), dst_offsets[0],
          indices[0]); // Adjust row index for source access
      Value idxCol = arith::AddIOp::create(
          builder, dmaMemcpyOp.getLoc(), dst_offsets[1],
          indices[1]); // Adjust column index for source access

      // Load from the source memref (e.g., memref<64x64xi32>)
      Value loadSrc = memref::LoadOp::create(builder, dmaMemcpyOp.getLoc(),
                                             srcMemref, indices);

      // Store to the destination memref (e.g., memref<32x32xi32>)
      memref::StoreOp::create(builder, dmaMemcpyOp.getLoc(), loadSrc,
                              destMemref, ValueRange{idxRow, idxCol});
    } else if (srcMemrefType.getShape()[0] > destMemrefType.getShape()[0] &&
               srcMemrefType.getShape()[1] > destMemrefType.getShape()[1]) {
      // Both x and y dimensions are larger in the source matrix, need to adjust
      // for chunks
      Value idxRow =
          arith::AddIOp::create(builder, dmaMemcpyOp.getLoc(), src_offsets[0],
                                indices[0]); // Adjust row index
      Value idxCol =
          arith::AddIOp::create(builder, dmaMemcpyOp.getLoc(), src_offsets[1],
                                indices[1]); // Adjust column index

      // Load from the source memref (e.g., memref<64x64xi32>)
      Value loadSrc = memref::LoadOp::create(
          builder, dmaMemcpyOp.getLoc(), srcMemref, ValueRange{idxRow, idxCol});

      // Store to the destination memref (e.g., memref<32x32xi32>)
      memref::StoreOp::create(builder, dmaMemcpyOp.getLoc(), loadSrc,
                              destMemref, indices);

    } else if (srcMemrefType.getShape()[0] > destMemrefType.getShape()[0]) {
      // Only the x dimension differs, adjust only the x index
      Value idxRow =
          arith::AddIOp::create(builder, dmaMemcpyOp.getLoc(), src_offsets[0],
                                indices[0]); // Adjust row index for larger x

      // Load from the source memref (e.g., memref<64x64xi32>)
      Value loadSrc =
          memref::LoadOp::create(builder, dmaMemcpyOp.getLoc(), srcMemref,
                                 ValueRange{idxRow, indices[1]});

      // Store to the destination memref (e.g., memref<32x32xi32>)
      memref::StoreOp::create(builder, dmaMemcpyOp.getLoc(), loadSrc,
                              destMemref, indices);

    } else if (srcMemrefType.getShape()[1] > destMemrefType.getShape()[1]) {
      // Only the y dimension differs, adjust only the y index
      Value idxCol =
          arith::AddIOp::create(builder, dmaMemcpyOp.getLoc(), src_offsets[1],
                                indices[1]); // Adjust column index for larger y

      // Load from the source memref (e.g., memref<64x64xi32>)
      Value loadSrc =
          memref::LoadOp::create(builder, dmaMemcpyOp.getLoc(), srcMemref,
                                 ValueRange{indices[0], idxCol});

      // Store to the destination memref (e.g., memref<32x32xi32>)
      memref::StoreOp::create(builder, dmaMemcpyOp.getLoc(), loadSrc,
                              destMemref, indices);
    }
    // Remove the air.dma_memcpy_nd operation
    dmaMemcpyOp.erase();
  }
};
} // namespace

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createGPUKernelOutlinePass() {
  return std::make_unique<ConvertGPUKernelOutlinePass>();
}

} // namespace air
} // namespace xilinx
