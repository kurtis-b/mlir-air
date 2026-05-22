//===- hip_int8_gemm_baseline.cpp -----------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include "hip/hip_runtime.h"
#include "rocblas/rocblas.h"
#include "rocwmma/rocwmma.hpp"

namespace {

constexpr int M = 1024;
constexpr int N = 1024;
constexpr int K = 1024;
constexpr int Tile = 16;
constexpr int WavesPerBlock = 8;
constexpr int ThreadsPerBlock = WavesPerBlock * 32;
constexpr int TilesM = M / Tile;
constexpr int TilesN = N / Tile;
constexpr int TileCount = TilesM * TilesN;
constexpr int AirTunedBlockM = 128;
constexpr int AirTunedBlockN = 128;
constexpr int AirTunedWaveM = 64;
constexpr int AirTunedWaveN = 64;
constexpr int AirTunedWaveTilesM = AirTunedWaveM / Tile;
constexpr int AirTunedWaveTilesN = AirTunedWaveN / Tile;
constexpr int AirTunedWavesPerBlock = 4;
constexpr int AirTunedThreadsPerBlock = AirTunedWavesPerBlock * 32;
constexpr int AirTunedBlocksM = M / AirTunedBlockM;
constexpr int AirTunedBlocksN = N / AirTunedBlockN;
constexpr int AirTunedGroupM = 8;
constexpr double Ops = 2.0 * double(M) * double(N) * double(K);

using I32x4 = int32_t __attribute__((ext_vector_type(4)));
using I32x8 = int32_t __attribute__((ext_vector_type(8)));

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t status = (expr);                                                \
    if (status != hipSuccess) {                                                \
      std::fprintf(stderr, "HIP error: %s (%d) at %s:%d\n",                   \
                   hipGetErrorString(status), static_cast<int>(status),        \
                   __FILE__, __LINE__);                                        \
      std::exit(2);                                                            \
    }                                                                          \
  } while (false)

#define ROCBLAS_CHECK(expr)                                                    \
  do {                                                                         \
    rocblas_status status = (expr);                                            \
    if (status != rocblas_status_success) {                                    \
      std::fprintf(stderr, "rocBLAS error: %d at %s:%d\n",                    \
                   static_cast<int>(status), __FILE__, __LINE__);              \
      std::exit(3);                                                            \
    }                                                                          \
  } while (false)

struct Options {
  std::string provider = "hip_wmma";
  int warmups = 10;
  int iterations = 20;
  int repetitions = 3;
  int validationSamples = 256;
};

struct Summary {
  double medianMeanMs = 0.0;
  double minMeanMs = 0.0;
  double meanMeanMs = 0.0;
  double stddevMeanMs = 0.0;
  double cvMeanMsPct = 0.0;
  double bestKernelMinMs = 0.0;
  double medianTops = 0.0;
  double meanTops = 0.0;
  double stddevTops = 0.0;
  double cvTopsPct = 0.0;
  double maxTops = 0.0;
};

void usage(const char *argv0) {
  std::fprintf(stderr,
               "Usage: %s --provider hip_wmma|rocwmma|air_tuned|rocblas_tensile "
               "[--warmups N] [--iterations N] [--repetitions N] "
               "[--validation-samples N]\n",
               argv0);
}

int parsePositive(const char *value, const char *name) {
  char *end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (!end || *end != '\0' || parsed <= 0 || parsed > 1'000'000) {
    std::fprintf(stderr, "invalid %s: %s\n", name, value);
    std::exit(2);
  }
  return static_cast<int>(parsed);
}

Options parseOptions(int argc, char **argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    auto requireValue = [&](const char *name) -> const char * {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", name);
        usage(argv[0]);
        std::exit(2);
      }
      return argv[++i];
    };
    if (std::strcmp(argv[i], "--provider") == 0) {
      opts.provider = requireValue("--provider");
    } else if (std::strcmp(argv[i], "--warmups") == 0) {
      opts.warmups = parsePositive(requireValue("--warmups"), "--warmups");
    } else if (std::strcmp(argv[i], "--iterations") == 0) {
      opts.iterations =
          parsePositive(requireValue("--iterations"), "--iterations");
    } else if (std::strcmp(argv[i], "--repetitions") == 0) {
      opts.repetitions =
          parsePositive(requireValue("--repetitions"), "--repetitions");
    } else if (std::strcmp(argv[i], "--validation-samples") == 0) {
      opts.validationSamples = parsePositive(requireValue("--validation-samples"),
                                             "--validation-samples");
    } else if (std::strcmp(argv[i], "--help") == 0 ||
               std::strcmp(argv[i], "-h") == 0) {
      usage(argv[0]);
      std::exit(0);
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
      usage(argv[0]);
      std::exit(2);
    }
  }
  if (opts.provider != "hip_wmma" && opts.provider != "rocwmma" &&
      opts.provider != "air_tuned" && opts.provider != "rocblas_tensile") {
    std::fprintf(stderr, "unsupported provider: %s\n", opts.provider.c_str());
    std::exit(2);
  }
  return opts;
}

void initInputs(std::vector<int8_t> &a, std::vector<int8_t> &b) {
  for (int i = 0; i < M; ++i)
    for (int k = 0; k < K; ++k)
      a[i * K + k] = static_cast<int8_t>(((i + 3 * k) & 15) - 8);

  for (int k = 0; k < K; ++k)
    for (int j = 0; j < N; ++j)
      b[k * N + j] = static_cast<int8_t>(((5 * k + j) & 15) - 8);
}

void packB(const std::vector<int8_t> &b, std::vector<int8_t> &bPacked) {
  for (int j = 0; j < N; ++j)
    for (int k = 0; k < K; ++k)
      bPacked[j * K + k] = b[k * N + j];
}

int validateSamples(const std::vector<int32_t> &c, const std::vector<int8_t> &a,
                    const std::vector<int8_t> &b, int samples) {
  int mismatches = 0;
  for (int s = 0; s < samples; ++s) {
    int i = (s * 131 + 17) % M;
    int j = (s * 197 + 29) % N;
    int32_t expected = 0;
    for (int k = 0; k < K; ++k)
      expected += static_cast<int32_t>(a[i * K + k]) *
                  static_cast<int32_t>(b[k * N + j]);
    int32_t actual = c[i * N + j];
    if (actual != expected) {
      if (mismatches < 8)
        std::printf("mismatch[%d] row=%d col=%d expected=%d actual=%d\n", s, i,
                    j, expected, actual);
      ++mismatches;
    }
  }
  return mismatches;
}

double mean(const std::vector<double> &values) {
  return std::accumulate(values.begin(), values.end(), 0.0) /
         static_cast<double>(values.size());
}

double stddev(const std::vector<double> &values, double avg) {
  if (values.size() < 2)
    return 0.0;
  double sum = 0.0;
  for (double value : values) {
    double delta = value - avg;
    sum += delta * delta;
  }
  return std::sqrt(sum / static_cast<double>(values.size() - 1));
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  size_t mid = values.size() / 2;
  if (values.size() % 2)
    return values[mid];
  return 0.5 * (values[mid - 1] + values[mid]);
}

Summary summarize(const std::vector<double> &meanMs,
                  const std::vector<double> &minMs,
                  const std::vector<double> &tops) {
  Summary summary;
  summary.medianMeanMs = median(meanMs);
  summary.minMeanMs = *std::min_element(meanMs.begin(), meanMs.end());
  summary.meanMeanMs = mean(meanMs);
  summary.stddevMeanMs = stddev(meanMs, summary.meanMeanMs);
  summary.cvMeanMsPct =
      summary.meanMeanMs ? (summary.stddevMeanMs / summary.meanMeanMs) * 100.0
                         : 0.0;
  summary.bestKernelMinMs = *std::min_element(minMs.begin(), minMs.end());
  summary.medianTops = median(tops);
  summary.meanTops = mean(tops);
  summary.stddevTops = stddev(tops, summary.meanTops);
  summary.cvTopsPct =
      summary.meanTops ? (summary.stddevTops / summary.meanTops) * 100.0 : 0.0;
  summary.maxTops = *std::max_element(tops.begin(), tops.end());
  return summary;
}

__device__ I32x4 loadAContiguous(const int8_t *a, int row, int kBase) {
  const int32_t *packed = reinterpret_cast<const int32_t *>(a + row * K + kBase);
  I32x4 out;
  out[0] = packed[0];
  out[1] = packed[1];
  out[2] = packed[2];
  out[3] = packed[3];
  return out;
}

__device__ I32x4 loadBColumnPacked(const int8_t *b, int col, int kBase) {
  I32x4 out;
#pragma unroll
  for (int word = 0; word < 4; ++word) {
    uint32_t packed = 0;
#pragma unroll
    for (int byte = 0; byte < 4; ++byte) {
      int kk = kBase + word * 4 + byte;
      auto value = static_cast<uint8_t>(b[kk * N + col]);
      packed |= static_cast<uint32_t>(value) << (byte * 8);
    }
    out[word] = static_cast<int32_t>(packed);
  }
  return out;
}

__device__ I32x4 loadBPackedContiguous(const int8_t *bPacked, int col,
                                       int kBase) {
  const int32_t *packed =
      reinterpret_cast<const int32_t *>(bPacked + col * K + kBase);
  I32x4 out;
  out[0] = packed[0];
  out[1] = packed[1];
  out[2] = packed[2];
  out[3] = packed[3];
  return out;
}

__device__ I32x8 zeroI32x8() {
  I32x8 out;
#pragma unroll
  for (int i = 0; i < 8; ++i)
    out[i] = 0;
  return out;
}

__device__ int2 groupedMacroTile(int pid) {
  constexpr int groupTiles = AirTunedGroupM * AirTunedBlocksN;
  int group = pid / groupTiles;
  int firstM = group * AirTunedGroupM;
  int groupSizeM = min(AirTunedBlocksM - firstM, AirTunedGroupM);
  int inGroup = pid - group * groupTiles;
  int pidM = firstM + (inGroup % groupSizeM);
  int pidN = inGroup / groupSizeM;
  return make_int2(pidM, pidN);
}

__global__ __launch_bounds__(ThreadsPerBlock, 2) void hipWmmaKernel(
    const int8_t *__restrict__ a, const int8_t *__restrict__ b,
    int32_t *__restrict__ c) {
  int lane = threadIdx.x & 31;
  int wave = threadIdx.x >> 5;
  int tileLinear = blockIdx.x * WavesPerBlock + wave;
  if (tileLinear >= TileCount)
    return;

  int tileRow = tileLinear / TilesN;
  int tileCol = tileLinear - tileRow * TilesN;
  int lane16 = lane & 15;
  int laneHalf = lane >> 4;
  int aRow = tileRow * Tile + lane16;
  int bCol = tileCol * Tile + lane16;
  I32x8 acc = {0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll 4
  for (int kBase = 0; kBase < K; kBase += Tile) {
    I32x4 aPacked = loadAContiguous(a, aRow, kBase);
    I32x4 bPacked = loadBColumnPacked(b, bCol, kBase);
    acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
        true, aPacked, true, bPacked, acc, false);
  }

#pragma unroll
  for (int element = 0; element < 8; ++element) {
    int row = tileRow * Tile + element * 2 + laneHalf;
    int col = tileCol * Tile + lane16;
    c[row * N + col] = acc[element];
  }
}

__global__ __launch_bounds__(AirTunedThreadsPerBlock, 1) void
airTuned128x128Kernel(const int8_t *__restrict__ a,
                      const int8_t *__restrict__ bPacked,
                      int32_t *__restrict__ c) {
  int lane = threadIdx.x & 31;
  int wave = threadIdx.x >> 5;
  int2 macro = groupedMacroTile(blockIdx.x);
  int waveM = wave >> 1;
  int waveN = wave & 1;
  int rowBase = macro.x * AirTunedBlockM + waveM * AirTunedWaveM;
  int colBase = macro.y * AirTunedBlockN + waveN * AirTunedWaveN;
  int lane16 = lane & 15;
  int laneHalf = lane >> 4;

  I32x8 acc[AirTunedWaveTilesM][AirTunedWaveTilesN];
#pragma unroll
  for (int mi = 0; mi < AirTunedWaveTilesM; ++mi) {
#pragma unroll
    for (int nj = 0; nj < AirTunedWaveTilesN; ++nj)
      acc[mi][nj] = zeroI32x8();
  }

#pragma unroll 2
  for (int kBase = 0; kBase < K; kBase += Tile) {
    I32x4 aFrag[AirTunedWaveTilesM];
    I32x4 bFrag[AirTunedWaveTilesN];
#pragma unroll
    for (int mi = 0; mi < AirTunedWaveTilesM; ++mi) {
      int row = rowBase + mi * Tile + lane16;
      aFrag[mi] = loadAContiguous(a, row, kBase);
    }
#pragma unroll
    for (int nj = 0; nj < AirTunedWaveTilesN; ++nj) {
      int col = colBase + nj * Tile + lane16;
      bFrag[nj] = loadBPackedContiguous(bPacked, col, kBase);
    }
#pragma unroll
    for (int mi = 0; mi < AirTunedWaveTilesM; ++mi) {
#pragma unroll
      for (int nj = 0; nj < AirTunedWaveTilesN; ++nj) {
        acc[mi][nj] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
            true, aFrag[mi], true, bFrag[nj], acc[mi][nj], false);
      }
    }
  }

#pragma unroll
  for (int mi = 0; mi < AirTunedWaveTilesM; ++mi) {
#pragma unroll
    for (int nj = 0; nj < AirTunedWaveTilesN; ++nj) {
#pragma unroll
      for (int element = 0; element < 8; ++element) {
        int row = rowBase + mi * Tile + element * 2 + laneHalf;
        int col = colBase + nj * Tile + lane16;
        c[row * N + col] = acc[mi][nj][element];
      }
    }
  }
}

__global__ __launch_bounds__(ThreadsPerBlock, 2) void rocwmmaKernel(
    const int8_t *__restrict__ a, const int8_t *__restrict__ b,
    int32_t *__restrict__ c) {
  int wave = threadIdx.x >> 5;
  int tileLinear = blockIdx.x * WavesPerBlock + wave;
  if (tileLinear >= TileCount)
    return;

  int tileRow = tileLinear / TilesN;
  int tileCol = tileLinear - tileRow * TilesN;
  using FragA = rocwmma::fragment<rocwmma::matrix_a, Tile, Tile, Tile, int8_t,
                                  rocwmma::row_major>;
  using FragB = rocwmma::fragment<rocwmma::matrix_b, Tile, Tile, Tile, int8_t,
                                  rocwmma::row_major>;
  using FragC =
      rocwmma::fragment<rocwmma::accumulator, Tile, Tile, Tile, int32_t>;

  FragA fragA;
  FragB fragB;
  FragC fragC;
  rocwmma::fill_fragment(fragC, 0);

  for (int kBase = 0; kBase < K; kBase += Tile) {
    rocwmma::load_matrix_sync(fragA, a + (tileRow * Tile) * K + kBase, K,
                              rocwmma::mem_row_major);
    rocwmma::load_matrix_sync(fragB, b + kBase * N + tileCol * Tile, N,
                              rocwmma::mem_row_major);
    rocwmma::mma_sync(fragC, fragA, fragB, fragC);
  }

  rocwmma::store_matrix_sync(c + (tileRow * Tile) * N + tileCol * Tile, fragC,
                             N, rocwmma::mem_row_major);
}

void launchHipWmma(const Options &, const int8_t *a, const int8_t *b,
                   int32_t *c, hipStream_t stream) {
  dim3 block(ThreadsPerBlock);
  dim3 grid((TileCount + WavesPerBlock - 1) / WavesPerBlock);
  hipWmmaKernel<<<grid, block, 0, stream>>>(a, b, c);
  HIP_CHECK(hipGetLastError());
}

void launchRocwmma(const Options &, const int8_t *a, const int8_t *b,
                   int32_t *c, hipStream_t stream) {
  dim3 block(ThreadsPerBlock);
  dim3 grid((TileCount + WavesPerBlock - 1) / WavesPerBlock);
  rocwmmaKernel<<<grid, block, 0, stream>>>(a, b, c);
  HIP_CHECK(hipGetLastError());
}

void launchAirTuned(const Options &, const int8_t *a, const int8_t *bPacked,
                    int32_t *c, hipStream_t stream) {
  dim3 block(AirTunedThreadsPerBlock);
  dim3 grid(AirTunedBlocksM * AirTunedBlocksN);
  airTuned128x128Kernel<<<grid, block, 0, stream>>>(a, bPacked, c);
  HIP_CHECK(hipGetLastError());
}

struct RocblasContext {
  rocblas_handle handle = nullptr;
  hipStream_t stream = nullptr;

  explicit RocblasContext(hipStream_t s) : stream(s) {
    ROCBLAS_CHECK(rocblas_create_handle(&handle));
    ROCBLAS_CHECK(rocblas_set_stream(handle, stream));
    ROCBLAS_CHECK(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
  }

  ~RocblasContext() {
    if (handle)
      rocblas_destroy_handle(handle);
  }
};

void runRocblasGemm(rocblas_handle handle, const int8_t *a, const int8_t *b,
                    int32_t *c) {
  int32_t alpha = 1;
  int32_t beta = 0;
  ROCBLAS_CHECK(rocblas_gemm_ex(
      handle, rocblas_operation_none, rocblas_operation_none, N, M, K, &alpha,
      b, rocblas_datatype_i8_r, N, a, rocblas_datatype_i8_r, K, &beta, c,
      rocblas_datatype_i32_r, N, c, rocblas_datatype_i32_r, N,
      rocblas_datatype_i32_r, rocblas_gemm_algo_standard, 0, 0));
}

void runProvider(const Options &opts, const int8_t *dA, const int8_t *dB,
                 const int8_t *dBPacked, int32_t *dC, hipStream_t stream) {
  if (opts.provider == "hip_wmma") {
    launchHipWmma(opts, dA, dB, dC, stream);
  } else if (opts.provider == "rocwmma") {
    launchRocwmma(opts, dA, dB, dC, stream);
  } else if (opts.provider == "air_tuned") {
    launchAirTuned(opts, dA, dBPacked, dC, stream);
  } else if (opts.provider == "rocblas_tensile") {
    static RocblasContext ctx(stream);
    runRocblasGemm(ctx.handle, dA, dB, dC);
  }
}

float timeIterations(const Options &opts, const int8_t *dA, const int8_t *dB,
                     const int8_t *dBPacked, int32_t *dC,
                     hipStream_t stream) {
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  HIP_CHECK(hipEventRecord(start, stream));
  for (int iter = 0; iter < opts.iterations; ++iter)
    runProvider(opts, dA, dB, dBPacked, dC, stream);
  HIP_CHECK(hipEventRecord(stop, stream));
  HIP_CHECK(hipEventSynchronize(stop));
  float elapsedMs = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&elapsedMs, start, stop));
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipEventDestroy(start));
  return elapsedMs / static_cast<float>(opts.iterations);
}

} // namespace

int main(int argc, char **argv) {
  Options opts = parseOptions(argc, argv);
  std::vector<int8_t> hostA(M * K);
  std::vector<int8_t> hostB(K * N);
  std::vector<int8_t> hostBPacked(K * N);
  std::vector<int32_t> hostC(M * N, 0);
  initInputs(hostA, hostB);
  packB(hostB, hostBPacked);

  int8_t *dA = nullptr;
  int8_t *dB = nullptr;
  int8_t *dBPacked = nullptr;
  int32_t *dC = nullptr;
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipMalloc(&dA, hostA.size() * sizeof(int8_t)));
  HIP_CHECK(hipMalloc(&dB, hostB.size() * sizeof(int8_t)));
  HIP_CHECK(hipMalloc(&dBPacked, hostBPacked.size() * sizeof(int8_t)));
  HIP_CHECK(hipMalloc(&dC, hostC.size() * sizeof(int32_t)));
  HIP_CHECK(hipMemcpyAsync(dA, hostA.data(), hostA.size() * sizeof(int8_t),
                           hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipMemcpyAsync(dB, hostB.data(), hostB.size() * sizeof(int8_t),
                           hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipMemcpyAsync(dBPacked, hostBPacked.data(),
                           hostBPacked.size() * sizeof(int8_t),
                           hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipMemsetAsync(dC, 0, hostC.size() * sizeof(int32_t), stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  for (int warmup = 0; warmup < opts.warmups; ++warmup)
    runProvider(opts, dA, dB, dBPacked, dC, stream);
  HIP_CHECK(hipStreamSynchronize(stream));

  std::vector<double> meanMs;
  std::vector<double> minMs;
  std::vector<double> tops;
  meanMs.reserve(opts.repetitions);
  minMs.reserve(opts.repetitions);
  tops.reserve(opts.repetitions);
  for (int rep = 0; rep < opts.repetitions; ++rep) {
    float meanMsRep = timeIterations(opts, dA, dB, dBPacked, dC, stream);
    double topsRep = (Ops / (double(meanMsRep) * 1.0e-3)) / 1.0e12;
    meanMs.push_back(meanMsRep);
    minMs.push_back(meanMsRep);
    tops.push_back(topsRep);
    std::printf("provider=%s repetition=%d mean_ms=%.6f tops=%.6f\n",
                opts.provider.c_str(), rep + 1, double(meanMsRep), topsRep);
  }

  HIP_CHECK(hipMemcpyAsync(hostC.data(), dC, hostC.size() * sizeof(int32_t),
                           hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  int mismatches = validateSamples(hostC, hostA, hostB, opts.validationSamples);
  Summary summary = summarize(meanMs, minMs, tops);
  const char *validation = mismatches == 0 ? "PASS" : "FAIL";
  const char *status = mismatches == 0 ? "PASS" : "FAIL";
  std::printf(
      "provider=%s status=%s validation=%s mismatches=%d "
      "warmups=%d iterations=%d repetitions=%d validation_samples=%d "
      "timing_domain=hip_event median_mean_ms=%.6f min_mean_ms=%.6f "
      "mean_mean_ms=%.6f stddev_mean_ms=%.6f cv_mean_ms_pct=%.6f "
      "best_kernel_min_ms=%.6f median_tops=%.6f mean_tops=%.6f "
      "stddev_tops=%.6f cv_tops_pct=%.6f max_tops=%.6f\n",
      opts.provider.c_str(), status, validation, mismatches, opts.warmups,
      opts.iterations, opts.repetitions, opts.validationSamples,
      summary.medianMeanMs, summary.minMeanMs, summary.meanMeanMs,
      summary.stddevMeanMs, summary.cvMeanMsPct, summary.bestKernelMinMs,
      summary.medianTops, summary.meanTops, summary.stddevTops,
      summary.cvTopsPct, summary.maxTops);

  HIP_CHECK(hipFree(dC));
  HIP_CHECK(hipFree(dBPacked));
  HIP_CHECK(hipFree(dB));
  HIP_CHECK(hipFree(dA));
  HIP_CHECK(hipStreamDestroy(stream));
  return mismatches == 0 ? 0 : 1;
}
