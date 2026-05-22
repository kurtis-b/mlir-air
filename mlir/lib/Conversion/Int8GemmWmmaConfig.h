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
static constexpr const char kInt8GemmBPackSwizzlePipe2_128x128Variant[] =
    "lds_128x128_bpack_swizzle_pipe2";
static constexpr const char kInt8GemmBPackSwizzlePipe2LoopedVariant[] =
    "lds_128x64_bpack_swizzle_pipe2_looped";
static constexpr const char kInt8GemmBPackSwizzleLooped_128x128Variant[] =
    "lds_128x128_bpack_swizzle_looped";
static constexpr const char kInt8GemmBPackSwizzlePipe2Looped_64x128Variant[] =
    "lds_64x128_bpack_swizzle_pipe2_looped";
static constexpr const char kInt8GemmBPackSwizzlePipe2LoopedK32Variant[] =
    "lds_128x64_bpack_swizzle_pipe2_k32_looped";
static constexpr const char kInt8GemmBPackSwizzlePipe2LoopedK128Variant[] =
    "lds_128x64_bpack_swizzle_pipe2_k128_looped";
static constexpr const char kInt8GemmBPackSwizzleLooped_128x128K32Variant[] =
    "lds_128x128_bpack_swizzle_k32_looped";
static constexpr const char kInt8GemmBPackSwizzleLooped_128x128K128Variant[] =
    "lds_128x128_bpack_swizzle_k128_looped";
static constexpr const char kInt8GemmBPackSwizzleBRegK64LoopedVariant[] =
    "lds_128x64_bpack_swizzle_breg_k64_looped";
static constexpr const char kInt8GemmBPackSwizzleK32W4Pipe2_128x64Variant[] =
    "lds_128x64_bpack_swizzle_k32_w4_pipe2";
static constexpr const char kInt8GemmBPackSwizzleK32W4Pipe2Pad_128x64Variant[] =
    "lds_128x64_bpack_swizzle_k32_w4_pipe2_pad";
static constexpr const char kInt8GemmBPackSwizzleK32W4Pipe2_64x128Variant[] =
    "lds_64x128_bpack_swizzle_k32_w4_pipe2";
static constexpr const char kInt8GemmBPackSwizzleK32W4Pipe2Pad_64x128Variant[] =
    "lds_64x128_bpack_swizzle_k32_w4_pipe2_pad";
static constexpr const char kInt8GemmBPackSwizzleK32W4Pipe2Variant[] =
    "lds_128x128_bpack_swizzle_k32_w4_pipe2";
static constexpr const char kInt8GemmBPackSwizzleK32W4Pipe2PadVariant[] =
    "lds_128x128_bpack_swizzle_k32_w4_pipe2_pad";
static constexpr const char
    kInt8GemmBPackSwizzleK32W4Pipe2Short_128x64Variant[] =
        "lds_128x64_bpack_swizzle_k32_w4_pipe2_short";
static constexpr const char
    kInt8GemmBPackSwizzleK32W4Pipe2ShortPad_128x64Variant[] =
        "lds_128x64_bpack_swizzle_k32_w4_pipe2_short_pad";
static constexpr const char
    kInt8GemmBPackSwizzleK32W4Pipe2Short_64x128Variant[] =
        "lds_64x128_bpack_swizzle_k32_w4_pipe2_short";
static constexpr const char
    kInt8GemmBPackSwizzleK32W4Pipe2ShortPad_64x128Variant[] =
        "lds_64x128_bpack_swizzle_k32_w4_pipe2_short_pad";
static constexpr const char kInt8GemmBPackSwizzleK32W4Pipe2ShortVariant[] =
    "lds_128x128_bpack_swizzle_k32_w4_pipe2_short";
static constexpr const char kInt8GemmBPackSwizzleK32W4Pipe2ShortPadVariant[] =
    "lds_128x128_bpack_swizzle_k32_w4_pipe2_short_pad";
static constexpr const char kInt8GemmRocmlirLikePipe3Variant[] =
    "lds_128x128_rocmlir_k32_pipe3";
static constexpr const char kInt8GemmAirTunedDirectVariant[] =
    "global_128x128_bpack_w4_direct";

enum class PipelineKind {
  SingleBufferLoop,
  Pipe2UnrolledCopy,
  Pipe2UnrolledPrefetch,
  Pipe2LoopedPrefetch,
  TensileLikePipe2,
  TensileLikePipe2ShortLived,
  RocmlirLikePipe3,
  AirTunedDirect,
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
    {kInt8GemmDefaultVariant, 128, 64, 64, 256, 1, false, false, false, false,
     PipelineKind::SingleBufferLoop, 4},
    {kInt8GemmBPackVariant, 128, 64, 64, 256, 1, false, true, false, false,
     PipelineKind::SingleBufferLoop, 4},
    {kInt8GemmBPackSwizzleVariant, 128, 64, 64, 256, 1, true, true, false,
     false, PipelineKind::SingleBufferLoop, 4},
    {kInt8GemmBPackPipe2Variant, 128, 64, 64, 256, 2, false, true, false,
     false, PipelineKind::Pipe2UnrolledCopy, 4},
    {kInt8GemmBPackPipe2GroupedVariant, 128, 64, 64, 256, 2, false, true,
     true, false, PipelineKind::Pipe2UnrolledCopy, 4},
    {kInt8GemmBPackSwizzleGroupedVariant, 128, 64, 64, 256, 1, true, true,
     true, false, PipelineKind::SingleBufferLoop, 4},
    {kInt8GemmBPackFragVariant, 128, 64, 64, 256, 1, false, true, false, true,
     PipelineKind::SingleBufferLoop, 4},
    {kInt8GemmBPackSwizzlePipe2_128x128Variant, 128, 128, 64, 256, 2, true,
     true, false, false, PipelineKind::Pipe2UnrolledPrefetch, 4},
    {kInt8GemmBPackSwizzlePipe2LoopedVariant, 128, 64, 64, 256, 2, true, true,
     false, false, PipelineKind::Pipe2LoopedPrefetch, 4},
    {kInt8GemmBPackSwizzleLooped_128x128Variant, 128, 128, 64, 256, 1, true,
     true, false, false, PipelineKind::SingleBufferLoop, 4},
    {kInt8GemmBPackSwizzlePipe2Looped_64x128Variant, 64, 128, 64, 128, 2,
     true, true, false, false, PipelineKind::Pipe2LoopedPrefetch, 4},
    {kInt8GemmBPackSwizzlePipe2LoopedK32Variant, 128, 64, 32, 256, 2, true,
     true, false, false, PipelineKind::Pipe2LoopedPrefetch, 4},
    {kInt8GemmBPackSwizzlePipe2LoopedK128Variant, 128, 64, 128, 256, 2, true,
     true, false, false, PipelineKind::Pipe2LoopedPrefetch, 4},
    {kInt8GemmBPackSwizzleLooped_128x128K32Variant, 128, 128, 32, 256, 1,
     true, true, false, false, PipelineKind::SingleBufferLoop, 4},
    {kInt8GemmBPackSwizzleLooped_128x128K128Variant, 128, 128, 128, 256, 1,
     true, true, false, false, PipelineKind::SingleBufferLoop, 4},
    {kInt8GemmBPackSwizzleBRegK64LoopedVariant, 128, 64, 64, 256, 2, true,
     true, false, true, PipelineKind::Pipe2LoopedPrefetch, 4},
    {kInt8GemmBPackSwizzleK32W4Pipe2_128x64Variant, 128, 64, 32, 128, 2,
     true, true, true, false, PipelineKind::TensileLikePipe2, 8, 32, 64, 0, 32, 4},
    {kInt8GemmBPackSwizzleK32W4Pipe2Pad_128x64Variant, 128, 64, 32, 128, 2,
     true, true, true, false, PipelineKind::TensileLikePipe2, 8, 32, 64, 16, 32, 4},
    {kInt8GemmBPackSwizzleK32W4Pipe2_64x128Variant, 64, 128, 32, 128, 2,
     true, true, true, false, PipelineKind::TensileLikePipe2, 8, 32, 64, 0, 32, 4},
    {kInt8GemmBPackSwizzleK32W4Pipe2Pad_64x128Variant, 64, 128, 32, 128, 2,
     true, true, true, false, PipelineKind::TensileLikePipe2, 8, 32, 64, 16, 32, 4},
    {kInt8GemmBPackSwizzleK32W4Pipe2Variant, 128, 128, 32, 128, 2, true,
     true, true, false, PipelineKind::TensileLikePipe2, 8, 64, 64, 0, 32, 4},
    {kInt8GemmBPackSwizzleK32W4Pipe2PadVariant, 128, 128, 32, 128, 2, true,
     true, true, false, PipelineKind::TensileLikePipe2, 8, 64, 64, 16, 32, 4},
    {kInt8GemmBPackSwizzleK32W4Pipe2Short_128x64Variant, 128, 64, 32, 128, 2,
     true, true, true, false, PipelineKind::TensileLikePipe2ShortLived, 8, 32,
     64, 0, 32, 4},
    {kInt8GemmBPackSwizzleK32W4Pipe2ShortPad_128x64Variant, 128, 64, 32, 128,
     2, true, true, true, false, PipelineKind::TensileLikePipe2ShortLived, 8,
     32, 64, 16, 32, 4},
    {kInt8GemmBPackSwizzleK32W4Pipe2Short_64x128Variant, 64, 128, 32, 128, 2,
     true, true, true, false, PipelineKind::TensileLikePipe2ShortLived, 8, 32,
     64, 0, 32, 4},
    {kInt8GemmBPackSwizzleK32W4Pipe2ShortPad_64x128Variant, 64, 128, 32, 128,
     2, true, true, true, false, PipelineKind::TensileLikePipe2ShortLived, 8,
     32, 64, 16, 32, 4},
    {kInt8GemmBPackSwizzleK32W4Pipe2ShortVariant, 128, 128, 32, 128, 2, true,
     true, true, false, PipelineKind::TensileLikePipe2ShortLived, 8, 64, 64,
     0, 32, 4},
    {kInt8GemmBPackSwizzleK32W4Pipe2ShortPadVariant, 128, 128, 32, 128, 2,
     true, true, true, false, PipelineKind::TensileLikePipe2ShortLived, 8, 64,
     64, 16, 32, 4},
    {kInt8GemmRocmlirLikePipe3Variant, 128, 128, 32, 128, 3, false, true,
     true, false, PipelineKind::RocmlirLikePipe3, 8, 64, 64, 0, 32, 4},
    {kInt8GemmAirTunedDirectVariant, 128, 128, 32, 128, 0, false, true, true,
     true, PipelineKind::AirTunedDirect, 8, 64, 64},
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
  return groupSize == 2 || groupSize == 4 || groupSize == 8;
}

} // namespace xilinx::air::gpu_int8_gemm

#endif // MLIR_AIR_LIB_CONVERSION_INT8GEMMWMMACONFIG_H
