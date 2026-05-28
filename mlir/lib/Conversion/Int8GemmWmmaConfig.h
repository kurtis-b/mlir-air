//===- Int8GemmWmmaConfig.h -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_AIR_LIB_CONVERSION_INT8GEMMWMMACONFIG_H
#define MLIR_AIR_LIB_CONVERSION_INT8GEMMWMMACONFIG_H

#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <optional>

namespace xilinx::air::gpu_int8_gemm {

static constexpr const char kInt8GemmWmmaAttr[] = "air.gpu.int8_gemm_wmma";
static constexpr const char kInt8GemmVariantAttr[] =
    "air.gpu.int8_gemm_variant";
static constexpr const char kInt8GemmGroupAttr[] =
    "air.gpu.int8_gemm_group_m";

static constexpr const char kInt8GemmDefaultVariant[] =
    "lds_128x128_rocmlir_k32_pipe3";

enum class PipelineKind {
  SingleBufferLoop,
  Pipe2UnrolledCopy,
  Pipe2UnrolledPrefetch,
  Pipe2LoopedPrefetch,
  TensileLikePipe2,
  TensileLikePipe2ShortLived,
  TensileLikePipe3,
  RocmlirLikePipe3,
  AirTunedDirect,
  AirTunedDirectCanonical,
  AirTunedDirectPrefetch,
  AirTunedDirectRawPtr,
  AirTunedDirectRawPtrU2,
};

struct Int8GemmKernelConfig {
  llvm::StringRef variant;
  int64_t blockRows;
  int64_t blockCols;
  int64_t kPerBlock;
  int64_t blockThreads;
  int64_t ldsStages;
  bool swizzledLds;
  bool packedB;
  bool groupedBlocks;
  bool directBFromGlobal;
  PipelineKind pipeline;
  uint32_t defaultGroupM;
  int64_t waveTileRows = 16;
  int64_t waveTileCols = 0;
  int64_t ldsKPadding = 0;
  int64_t blockDimX = 0;
  int64_t blockDimY = 1;
  int64_t wavesPerEu = 0;

  int64_t waveCount() const { return blockThreads / 32; }
  int64_t effectiveWaveTileCols() const {
    return waveTileCols == 0 ? blockCols : waveTileCols;
  }
  int64_t wavesM() const { return blockRows / waveTileRows; }
  int64_t wavesN() const { return blockCols / effectiveWaveTileCols(); }
  int64_t effectiveBlockDimX() const {
    return blockDimX == 0 ? blockThreads : blockDimX;
  }
  int64_t effectiveBlockDimY() const {
    return blockDimX == 0 ? 1 : blockDimY;
  }
};

static const Int8GemmKernelConfig kInt8GemmKernelConfigs[] = {
    {kInt8GemmDefaultVariant, 128, 128, 32, 128, 3, false, true,
     true, false, PipelineKind::RocmlirLikePipe3, 8, 64, 64, 0, 32, 4},
};


inline std::optional<Int8GemmKernelConfig>
getInt8GemmKernelConfig(llvm::StringRef variant) {
  for (const auto &config : kInt8GemmKernelConfigs)
    if (variant == config.variant)
      return config;
  return std::nullopt;
}

inline bool isSupportedInt8GemmVariant(llvm::StringRef variant) {
  return getInt8GemmKernelConfig(variant).has_value();
}

inline bool isSupportedInt8GemmGroupSize(uint32_t groupSize) {
  return groupSize == 8;
}

inline bool usesRawPtrKernelAbi(const Int8GemmKernelConfig &config) {
  return config.pipeline == PipelineKind::AirTunedDirectRawPtr ||
         config.pipeline == PipelineKind::AirTunedDirectRawPtrU2;
}

inline bool usesLinearGroupedGrid(const Int8GemmKernelConfig &config) {
  return usesRawPtrKernelAbi(config);
}

} // namespace xilinx::air::gpu_int8_gemm

#endif // MLIR_AIR_LIB_CONVERSION_INT8GEMMWMMACONFIG_H
