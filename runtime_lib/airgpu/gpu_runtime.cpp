// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Drop-in replacement for LLVM's libmlir_rocm_runtime.so.
// Implements the same mgpu* C ABI but uses VMem-backed allocation
// (hipMemCreate/hipMemMap/hipMemSetAccess) instead of hipMalloc.
//
// Usage:
//   mlir-runner --entry-point-result=void \
//       --shared-libs=libairgpu.so \
//       final.mlir

#include "symmetric_heap.h"
#include "vmem_allocator.h"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>
#include <limits>
#include <mutex>

static void reportHipError(const char *expr, hipError_t result) {
  if (result != hipSuccess)
    fprintf(stderr, "'%s' failed with '%s'\n", expr,
            hipGetErrorString(result));
}

#define HIP_REPORT_IF_ERROR(expr) reportHipError(#expr, (expr))

// Matches LLVM's StridedMemRefType used in mlir-runner
template <typename T, int N>
struct StridedMemRefType {
  T *basePtr;
  T *data;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

// ---------------------------------------------------------------------------
// Global allocator instance (constructor/destructor for library load/unload)
// ---------------------------------------------------------------------------

static VMemAllocator *g_allocator = nullptr;
static SymmetricHeap *g_symmetric_heap = nullptr;

struct BenchmarkStats {
  uint64_t count = 0;
  double totalMs = 0.0;
  double minMs = std::numeric_limits<double>::infinity();
  double maxMs = 0.0;

  void reset() {
    count = 0;
    totalMs = 0.0;
    minMs = std::numeric_limits<double>::infinity();
    maxMs = 0.0;
  }

  void record(double ms) {
    ++count;
    totalMs += ms;
    if (ms < minMs)
      minMs = ms;
    if (ms > maxMs)
      maxMs = ms;
  }
};

static std::atomic<bool> g_kernel_profiling_enabled{false};
static std::mutex g_benchmark_mutex;
static BenchmarkStats g_kernel_event_stats;
static BenchmarkStats g_host_dispatch_wait_stats;
static hipStream_t g_benchmark_stream = nullptr;
static hipEvent_t g_benchmark_start_event = nullptr;
static hipEvent_t g_benchmark_stop_event = nullptr;
static uint64_t g_benchmark_launch_count = 0;
static bool g_benchmark_batch_failed = false;

static void recordKernelEventMs(double ms) {
  if (ms < 0.0)
    return;
  std::lock_guard<std::mutex> lock(g_benchmark_mutex);
  g_kernel_event_stats.record(ms);
}

static void recordKernelEventBatchMs(double averageMs, uint64_t count) {
  if (averageMs < 0.0 || count == 0)
    return;
  std::lock_guard<std::mutex> lock(g_benchmark_mutex);
  for (uint64_t i = 0; i < count; ++i)
    g_kernel_event_stats.record(averageMs);
}

static bool isBenchmarkStream(hipStream_t stream) {
  return stream && stream == g_benchmark_stream;
}

static bool usePersistentBenchmarkStream() {
  const char *env = std::getenv("AIRGPU_BENCHMARK_STREAM");
  return env && std::strcmp(env, "0") != 0;
}

static void resetBenchmarkBatch(bool destroyStream) {
  if (g_benchmark_start_event) {
    HIP_REPORT_IF_ERROR(hipEventDestroy(g_benchmark_start_event));
    g_benchmark_start_event = nullptr;
  }
  if (g_benchmark_stop_event) {
    HIP_REPORT_IF_ERROR(hipEventDestroy(g_benchmark_stop_event));
    g_benchmark_stop_event = nullptr;
  }
  if (destroyStream && g_benchmark_stream) {
    HIP_REPORT_IF_ERROR(hipStreamDestroy(g_benchmark_stream));
    g_benchmark_stream = nullptr;
  }
  g_benchmark_launch_count = 0;
  g_benchmark_batch_failed = false;
}

static void finalizeBenchmarkBatch() {
  if (!g_benchmark_stream)
    return;

  if (!g_benchmark_batch_failed && g_benchmark_launch_count > 0 &&
      g_benchmark_stop_event) {
    hipError_t result = hipEventSynchronize(g_benchmark_stop_event);
    if (result != hipSuccess) {
      reportHipError("hipEventSynchronize(g_benchmark_stop_event)", result);
    } else {
      float elapsedMs = 0.0f;
      result = hipEventElapsedTime(&elapsedMs, g_benchmark_start_event,
                                   g_benchmark_stop_event);
      if (result == hipSuccess) {
        recordKernelEventBatchMs(
            static_cast<double>(elapsedMs) /
                static_cast<double>(g_benchmark_launch_count),
            g_benchmark_launch_count);
      } else {
        reportHipError("hipEventElapsedTime(&elapsedMs, g_benchmark_start_event, "
                       "g_benchmark_stop_event)",
                       result);
      }
    }
  } else {
    HIP_REPORT_IF_ERROR(hipStreamSynchronize(g_benchmark_stream));
  }

  resetBenchmarkBatch(!usePersistentBenchmarkStream());
}

static hipStream_t getBenchmarkStream() {
  if (!g_benchmark_stream)
    HIP_REPORT_IF_ERROR(hipStreamCreate(&g_benchmark_stream));
  return g_benchmark_stream;
}

static void printBenchmarkStats(const char *domain, const BenchmarkStats &stats,
                                double ops) {
  if (stats.count == 0) {
    printf("timing_domain=%s count=0 min_ms=nan mean_ms=nan max_ms=nan "
           "tops=nan\n",
           domain);
    return;
  }

  double meanMs = stats.totalMs / static_cast<double>(stats.count);
  double tops = ops / (meanMs * 1.0e9);
  printf("timing_domain=%s count=%llu min_ms=%.6f mean_ms=%.6f max_ms=%.6f "
         "tops=%.6f\n",
         domain, static_cast<unsigned long long>(stats.count), stats.minMs,
         meanMs, stats.maxMs, tops);
}

static double meanTops(const BenchmarkStats &stats, double ops) {
  if (stats.count == 0)
    return std::numeric_limits<double>::quiet_NaN();
  double meanMs = stats.totalMs / static_cast<double>(stats.count);
  return ops / (meanMs * 1.0e9);
}

// Lazy-init the standalone allocator (only when mgpuMemAlloc is called
// without a symmetric heap).  Avoids pinning device 0 at library load time.
static std::once_flag g_allocator_flag;
static VMemAllocator *getDefaultAllocator() {
  std::call_once(g_allocator_flag, [] { g_allocator = new VMemAllocator(); });
  return g_allocator;
}

__attribute__((destructor)) static void airgpu_runtime_shutdown() {
  delete g_symmetric_heap;
  g_symmetric_heap = nullptr;
  delete g_allocator;
  g_allocator = nullptr;
}

// ===========================================================================
// Module Management
// ===========================================================================

extern "C" hipModule_t mgpuModuleLoad(void *data, size_t /*gpuBlobSize*/) {
  hipModule_t module = nullptr;
  HIP_REPORT_IF_ERROR(hipModuleLoadData(&module, data));
  return module;
}

extern "C" hipModule_t mgpuModuleLoadJIT(void *data, int /*optLevel*/) {
  hipModule_t module = nullptr;
  HIP_REPORT_IF_ERROR(hipModuleLoadData(&module, data));
  return module;
}

extern "C" void mgpuModuleUnload(hipModule_t module) {
  HIP_REPORT_IF_ERROR(hipModuleUnload(module));
}

extern "C" hipFunction_t mgpuModuleGetFunction(hipModule_t module,
                                               const char *name) {
  hipFunction_t function = nullptr;
  HIP_REPORT_IF_ERROR(hipModuleGetFunction(&function, module, name));
  return function;
}

// ===========================================================================
// Kernel Launch
// ===========================================================================

extern "C" void mgpuLaunchKernel(hipFunction_t function, intptr_t gridX,
                                 intptr_t gridY, intptr_t gridZ,
                                 intptr_t blockX, intptr_t blockY,
                                 intptr_t blockZ, int32_t smem,
                                 hipStream_t stream, void **params,
                                 void **extra, size_t /*paramsCount*/) {
  if (!g_kernel_profiling_enabled.load(std::memory_order_acquire)) {
    HIP_REPORT_IF_ERROR(hipModuleLaunchKernel(function, gridX, gridY, gridZ,
                                              blockX, blockY, blockZ, smem,
                                              stream, params, extra));
    return;
  }

  if (isBenchmarkStream(stream)) {
    if (!g_benchmark_start_event)
      HIP_REPORT_IF_ERROR(hipEventCreate(&g_benchmark_start_event));
    if (!g_benchmark_stop_event)
      HIP_REPORT_IF_ERROR(hipEventCreate(&g_benchmark_stop_event));
    if (!g_benchmark_start_event || !g_benchmark_stop_event)
      g_benchmark_batch_failed = true;

    if (!g_benchmark_batch_failed && g_benchmark_launch_count == 0) {
      hipError_t result = hipEventRecord(g_benchmark_start_event, stream);
      if (result != hipSuccess) {
        reportHipError("hipEventRecord(g_benchmark_start_event, stream)",
                       result);
        g_benchmark_batch_failed = true;
      }
    }

    hipError_t launchResult = hipModuleLaunchKernel(
        function, gridX, gridY, gridZ, blockX, blockY, blockZ, smem, stream,
        params, extra);
    reportHipError("hipModuleLaunchKernel(function, gridX, gridY, gridZ, blockX, "
                   "blockY, blockZ, smem, stream, params, extra)",
                   launchResult);

    if (launchResult == hipSuccess)
      ++g_benchmark_launch_count;
    else
      g_benchmark_batch_failed = true;

    if (!g_benchmark_batch_failed) {
      hipError_t result = hipEventRecord(g_benchmark_stop_event, stream);
      if (result != hipSuccess) {
        reportHipError("hipEventRecord(g_benchmark_stop_event, stream)",
                       result);
        g_benchmark_batch_failed = true;
      }
    }
    return;
  }

  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  hipError_t result = hipEventCreate(&start);
  if (result != hipSuccess) {
    reportHipError("hipEventCreate(&start)", result);
    HIP_REPORT_IF_ERROR(hipModuleLaunchKernel(function, gridX, gridY, gridZ,
                                              blockX, blockY, blockZ, smem,
                                              stream, params, extra));
    return;
  }

  result = hipEventCreate(&stop);
  if (result != hipSuccess) {
    reportHipError("hipEventCreate(&stop)", result);
    HIP_REPORT_IF_ERROR(hipEventDestroy(start));
    HIP_REPORT_IF_ERROR(hipModuleLaunchKernel(function, gridX, gridY, gridZ,
                                              blockX, blockY, blockZ, smem,
                                              stream, params, extra));
    return;
  }

  bool canProfile = true;
  result = hipEventRecord(start, stream);
  if (result != hipSuccess) {
    reportHipError("hipEventRecord(start, stream)", result);
    canProfile = false;
  }

  hipError_t launchResult = hipModuleLaunchKernel(
      function, gridX, gridY, gridZ, blockX, blockY, blockZ, smem, stream,
      params, extra);
  reportHipError("hipModuleLaunchKernel(function, gridX, gridY, gridZ, blockX, "
                 "blockY, blockZ, smem, stream, params, extra)",
                 launchResult);

  result = hipEventRecord(stop, stream);
  if (result != hipSuccess) {
    reportHipError("hipEventRecord(stop, stream)", result);
    canProfile = false;
  }

  if (canProfile && launchResult == hipSuccess) {
    result = hipEventSynchronize(stop);
    if (result != hipSuccess) {
      reportHipError("hipEventSynchronize(stop)", result);
    } else {
      float elapsedMs = 0.0f;
      result = hipEventElapsedTime(&elapsedMs, start, stop);
      if (result == hipSuccess)
        recordKernelEventMs(static_cast<double>(elapsedMs));
      else
        reportHipError("hipEventElapsedTime(&elapsedMs, start, stop)", result);
    }
  }

  HIP_REPORT_IF_ERROR(hipEventDestroy(stop));
  HIP_REPORT_IF_ERROR(hipEventDestroy(start));
}

extern "C" void mgpuBenchmarkReset() {
  resetBenchmarkBatch(!usePersistentBenchmarkStream());
  std::lock_guard<std::mutex> lock(g_benchmark_mutex);
  g_kernel_event_stats.reset();
  g_host_dispatch_wait_stats.reset();
}

extern "C" void mgpuBenchmarkSetKernelProfiling(int32_t enabled) {
  if (enabled != 0) {
    resetBenchmarkBatch(!usePersistentBenchmarkStream());
    g_kernel_profiling_enabled.store(true, std::memory_order_release);
    return;
  }
  g_kernel_profiling_enabled.store(false, std::memory_order_release);
  finalizeBenchmarkBatch();
}

extern "C" int64_t mgpuHostTimeNs() {
  auto now = std::chrono::steady_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
}

extern "C" void mgpuBenchmarkRecordHostNs(int64_t elapsedNs) {
  if (elapsedNs < 0)
    return;

  double elapsedMs = static_cast<double>(elapsedNs) / 1.0e6;
  std::lock_guard<std::mutex> lock(g_benchmark_mutex);
  g_host_dispatch_wait_stats.record(elapsedMs);
}

extern "C" void mgpuBenchmarkPrintI8I32(int64_t m, int64_t n, int64_t k,
                                        int64_t warmups,
                                        int64_t iterations) {
  BenchmarkStats kernelEventStats;
  BenchmarkStats hostDispatchWaitStats;
  {
    std::lock_guard<std::mutex> lock(g_benchmark_mutex);
    kernelEventStats = g_kernel_event_stats;
    hostDispatchWaitStats = g_host_dispatch_wait_stats;
  }

  double ops = 2.0 * static_cast<double>(m) * static_cast<double>(n) *
               static_cast<double>(k);

  printf("backend=gpu\n");
  printf("shape=%lldx%lldx%lld\n", static_cast<long long>(m),
         static_cast<long long>(n), static_cast<long long>(k));
  printf("warmups=%lld iterations=%lld\n", static_cast<long long>(warmups),
         static_cast<long long>(iterations));
  printBenchmarkStats("kernel_event", kernelEventStats, ops);
  printBenchmarkStats("host_dispatch_wait", hostDispatchWaitStats, ops);
  double kernelTops = meanTops(kernelEventStats, ops);
  constexpr double kRadeon890MInt8PeakTops = 23.7568;
  printf("gpu_peak_int8_tops=%.4f kernel_event_peak_pct=%.2f "
         "kernel_event_gap_tops=%.4f\n",
         kRadeon890MInt8PeakTops,
         (kernelTops / kRadeon890MInt8PeakTops) * 100.0,
         kRadeon890MInt8PeakTops - kernelTops);
}

// ===========================================================================
// Stream Management
// ===========================================================================

extern "C" hipStream_t mgpuStreamCreate() {
  if (g_kernel_profiling_enabled.load(std::memory_order_acquire) ||
      usePersistentBenchmarkStream())
    return getBenchmarkStream();
  hipStream_t stream = nullptr;
  HIP_REPORT_IF_ERROR(hipStreamCreate(&stream));
  return stream;
}

extern "C" void mgpuStreamDestroy(hipStream_t stream) {
  if (isBenchmarkStream(stream))
    return;
  HIP_REPORT_IF_ERROR(hipStreamDestroy(stream));
}

extern "C" void mgpuStreamSynchronize(hipStream_t stream) {
  if (g_kernel_profiling_enabled.load(std::memory_order_acquire) &&
      isBenchmarkStream(stream))
    return;
  HIP_REPORT_IF_ERROR(hipStreamSynchronize(stream));
}

extern "C" void mgpuStreamWaitEvent(hipStream_t stream, hipEvent_t event) {
  HIP_REPORT_IF_ERROR(hipStreamWaitEvent(stream, event, 0));
}

// ===========================================================================
// Event Management
// ===========================================================================

extern "C" hipEvent_t mgpuEventCreate() {
  hipEvent_t event = nullptr;
  HIP_REPORT_IF_ERROR(hipEventCreate(&event));
  return event;
}

extern "C" void mgpuEventDestroy(hipEvent_t event) {
  HIP_REPORT_IF_ERROR(hipEventDestroy(event));
}

extern "C" void mgpuEventSynchronize(hipEvent_t event) {
  HIP_REPORT_IF_ERROR(hipEventSynchronize(event));
}

extern "C" void mgpuEventRecord(hipEvent_t event, hipStream_t stream) {
  HIP_REPORT_IF_ERROR(hipEventRecord(event, stream));
}

// ===========================================================================
// Memory — VMem-backed (the key difference from LLVM's runtime)
// ===========================================================================

static bool useHipMallocAllocator() {
  const char *env = std::getenv("AIRGPU_USE_HIP_MALLOC");
  return env && std::strcmp(env, "0") != 0;
}

extern "C" void *mgpuMemAlloc(uint64_t sizeBytes, hipStream_t /*stream*/,
                              bool /*isHostShared*/) {
  if (useHipMallocAllocator()) {
    void *ptr = nullptr;
    HIP_REPORT_IF_ERROR(hipMalloc(&ptr, static_cast<size_t>(sizeBytes)));
    return ptr;
  }
  return getDefaultAllocator()->allocate(static_cast<size_t>(sizeBytes));
}

extern "C" void mgpuMemFree(void *ptr, hipStream_t /*stream*/) {
  if (!ptr)
    return;
  if (useHipMallocAllocator()) {
    HIP_REPORT_IF_ERROR(hipFree(ptr));
    return;
  }
  getDefaultAllocator()->free(ptr);
}

// ===========================================================================
// Memory Operations (standard HIP, same as LLVM)
// ===========================================================================

extern "C" void mgpuMemcpy(void *dst, void *src, size_t sizeBytes,
                           hipStream_t stream) {
  HIP_REPORT_IF_ERROR(
      hipMemcpyAsync(dst, src, sizeBytes, hipMemcpyDefault, stream));
}

extern "C" void mgpuMemset32(void *dst, int value, size_t count,
                             hipStream_t stream) {
  HIP_REPORT_IF_ERROR(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(dst),
                                        value, count, stream));
}

extern "C" void mgpuMemset16(void *dst, short value, size_t count,
                             hipStream_t stream) {
  HIP_REPORT_IF_ERROR(hipMemsetD16Async(reinterpret_cast<hipDeviceptr_t>(dst),
                                        value, count, stream));
}

// ===========================================================================
// Host Memory Registration
// ===========================================================================

extern "C" void mgpuMemHostRegister(void *ptr, uint64_t sizeBytes) {
  HIP_REPORT_IF_ERROR(hipHostRegister(ptr, sizeBytes, hipHostRegisterDefault));
}

extern "C" void
mgpuMemHostRegisterMemRef(int64_t rank, StridedMemRefType<char, 1> *descriptor,
                          int64_t elementSizeBytes) {
  if (rank > 0 && descriptor) {
    int64_t size = descriptor->sizes[0] * elementSizeBytes;
    HIP_REPORT_IF_ERROR(
        hipHostRegister(descriptor->data, size, hipHostRegisterDefault));
  }
}

extern "C" void mgpuMemHostUnregister(void *ptr) {
  HIP_REPORT_IF_ERROR(hipHostUnregister(ptr));
}

extern "C" void
mgpuMemHostUnregisterMemRef(int64_t rank,
                            StridedMemRefType<char, 1> *descriptor,
                            int64_t elementSizeBytes) {
  if (descriptor) {
    HIP_REPORT_IF_ERROR(hipHostUnregister(descriptor->data));
  }
}

// ===========================================================================
// Device Management & MemRef Helpers
// ===========================================================================

extern "C" void mgpuSetDefaultDevice(int32_t device) {
  HIP_REPORT_IF_ERROR(hipSetDevice(device));
}

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif
extern "C" StridedMemRefType<float, 1>
mgpuMemGetDeviceMemRef1dFloat(float *allocated, float *aligned, int64_t offset,
                              int64_t size, int64_t stride) {
  StridedMemRefType<float, 1> result;
  result.basePtr = allocated;
  result.data = aligned;
  result.offset = offset;
  result.sizes[0] = size;
  result.strides[0] = stride;
  return result;
}

extern "C" StridedMemRefType<int32_t, 1>
mgpuMemGetDeviceMemRef1dInt32(int32_t *allocated, int32_t *aligned,
                              int64_t offset, int64_t size, int64_t stride) {
  StridedMemRefType<int32_t, 1> result;
  result.basePtr = allocated;
  result.data = aligned;
  result.offset = offset;
  result.sizes[0] = size;
  result.strides[0] = stride;
  return result;
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif

// ===========================================================================
// Symmetric Heap — multi-GPU memory sharing
// ===========================================================================

extern "C" void mgpuSymmetricHeapInit(uint64_t heap_size) {
  if (g_symmetric_heap) {
    fprintf(stderr, "airgpu: symmetric heap already initialized\n");
    return;
  }
  g_symmetric_heap = new SymmetricHeap(static_cast<size_t>(heap_size));
}

extern "C" void mgpuSymmetricHeapDestroy() {
  delete g_symmetric_heap;
  g_symmetric_heap = nullptr;
}

extern "C" int32_t mgpuGetRank() {
  if (!g_symmetric_heap)
    return 0;
  return g_symmetric_heap->getRank();
}

extern "C" int32_t mgpuGetWorldSize() {
  if (!g_symmetric_heap)
    return 1;
  return g_symmetric_heap->getWorldSize();
}

extern "C" void *mgpuSymmetricAlloc(uint64_t sizeBytes,
                                    hipStream_t /*stream*/) {
  if (!g_symmetric_heap) {
    fprintf(stderr, "airgpu: symmetric heap not initialized\n");
    abort();
  }
  return g_symmetric_heap->allocate(static_cast<size_t>(sizeBytes));
}

extern "C" void mgpuSymmetricFree(void *ptr, hipStream_t /*stream*/) {
  if (g_symmetric_heap && ptr)
    g_symmetric_heap->free(ptr);
}

extern "C" void *mgpuGetHeapBase(int32_t rank) {
  if (!g_symmetric_heap)
    return nullptr;
  return g_symmetric_heap->getHeapBase(rank);
}

extern "C" void **mgpuGetHeapBases() {
  if (!g_symmetric_heap)
    return nullptr;
  return g_symmetric_heap->getHeapBases();
}

extern "C" void mgpuBarrier() {
  if (g_symmetric_heap)
    g_symmetric_heap->barrier();
}

extern "C" void mgpuSetDevice(int32_t device_id) {
  HIP_REPORT_IF_ERROR(hipSetDevice(device_id));
}
