//===- int8_gemm.cpp -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <pthread.h>
#if defined(__linux__)
#include <sched.h>
#endif
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <immintrin.h>

namespace {

constexpr int kM = 1024;
constexpr int kN = 1024;
constexpr int kK = 1024;
constexpr int kMR = 4;
constexpr int kNR = 32;
constexpr int kKGroup = 4;
constexpr int kDefaultThreads = 12;

struct Options {
  int warmups = 0;
  int iterations = 1;
  int threads = kDefaultThreads;
  bool verify = true;
};

void usage(const char *argv0) {
  std::cout << "Usage: " << argv0
            << " [--warmups N] [--iterations N] [--threads N]"
            << " [--no-verify]\n";
}

int parseInt(const char *value, const char *name, bool allowZero = true) {
  char *end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  long minValue = allowZero ? 0 : 1;
  if (*value == '\0' || *end != '\0' || parsed < minValue ||
      parsed > std::numeric_limits<int>::max()) {
    throw std::runtime_error(std::string("invalid ") + name + ": " + value);
  }
  return static_cast<int>(parsed);
}

Options parseOptions(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
      std::exit(0);
    }
    if (arg == "--warmups") {
      if (++i == argc)
        throw std::runtime_error("missing value for --warmups");
      options.warmups = parseInt(argv[i], "--warmups");
      continue;
    }
    if (arg == "--iterations") {
      if (++i == argc)
        throw std::runtime_error("missing value for --iterations");
      options.iterations = parseInt(argv[i], "--iterations");
      continue;
    }
    if (arg == "--threads") {
      if (++i == argc)
        throw std::runtime_error("missing value for --threads");
      options.threads = parseInt(argv[i], "--threads", false);
      continue;
    }
    if (arg == "--no-verify") {
      options.verify = false;
      continue;
    }
    throw std::runtime_error("unknown option: " + arg);
  }
  if (options.iterations == 0)
    throw std::runtime_error("--iterations must be greater than zero");
  return options;
}

void initializeInputs(std::vector<int8_t> &a, std::vector<int8_t> &b,
                      int8_t *bPacked) {
  for (int i = 0; i < kM; ++i) {
    for (int k = 0; k < kK; ++k) {
      a[i * kK + k] = static_cast<int8_t>((i * 3 + k * 5 + 1) & 7);
    }
  }
  for (int k = 0; k < kK; ++k) {
    for (int j = 0; j < kN; ++j) {
      int8_t value = static_cast<int8_t>((k * 7 + j * 11 + 3) & 7);
      b[k * kN + j] = value;
    }
  }

  for (int panel = 0; panel < kN; panel += kNR) {
    int8_t *panelBase = bPacked + (panel / kNR) * kK * kNR;
    for (int k = 0; k < kK; k += kKGroup) {
      int8_t *kBase = panelBase + (k / kKGroup) * kNR * kKGroup;
      for (int col = 0; col < kNR; ++col) {
        for (int byte = 0; byte < kKGroup; ++byte)
          kBase[col * kKGroup + byte] = b[(k + byte) * kN + panel + col];
      }
    }
  }
}

int32_t referenceElement(const std::vector<int8_t> &a,
                         const std::vector<int8_t> &b, int row, int col) {
  int32_t sum = 0;
  for (int k = 0; k < kK; ++k)
    sum += static_cast<int32_t>(a[row * kK + k]) *
           static_cast<int32_t>(b[k * kN + col]);
  return sum;
}

bool verifySamples(const std::vector<int8_t> &a, const std::vector<int8_t> &b,
                   const int32_t *c) {
  constexpr int samples[][2] = {{0, 0},     {0, 31},    {3, 5},
                                {17, 19},   {127, 64},  {255, 255},
                                {511, 17},  {700, 901}, {1023, 1023}};
  bool ok = true;
  for (const auto &sample : samples) {
    int row = sample[0];
    int col = sample[1];
    int32_t expected = referenceElement(a, b, row, col);
    int32_t observed = c[row * kN + col];
    if (expected == observed)
      continue;
    std::cerr << "mismatch at (" << row << ", " << col
              << "): expected=" << expected << " observed=" << observed
              << "\n";
    ok = false;
  }
  return ok;
}

struct AlignedFree {
  void operator()(void *ptr) const { _mm_free(ptr); }
};

template <typename T>
using AlignedPtr = std::unique_ptr<T, AlignedFree>;

template <typename T>
AlignedPtr<T> makeAlignedBuffer(std::size_t count) {
  void *ptr = _mm_malloc(count * sizeof(T), 64);
  if (!ptr)
    throw std::bad_alloc();
  return AlignedPtr<T>(static_cast<T *>(ptr));
}

static inline __attribute__((always_inline)) uint32_t
loadPackedA4(const int8_t *ptr) {
  uint32_t packed;
  std::memcpy(&packed, ptr, sizeof(packed));
  return packed;
}

bool cpuSupportsVnni() {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx512f") &&
         __builtin_cpu_supports("avx512bw") &&
         __builtin_cpu_supports("avx512vnni");
#else
  return true;
#endif
}

} // namespace

extern "C" __attribute__((noinline, used,
                          target("avx512f,avx512bw,avx512vl,avx512vnni")))
void cpu_i8_gemm_vnni(const int8_t *__restrict__ a,
                      const int8_t *__restrict__ bPacked,
                      int32_t *__restrict__ c, int rowBegin, int rowEnd) {
  for (int i = rowBegin; i < rowEnd; i += kMR) {
    const int8_t *aRow0 = a + (i + 0) * kK;
    const int8_t *aRow1 = a + (i + 1) * kK;
    const int8_t *aRow2 = a + (i + 2) * kK;
    const int8_t *aRow3 = a + (i + 3) * kK;
    for (int j = 0; j < kN; j += kNR) {
      const int8_t *bPanel = bPacked + (j / kNR) * kK * kNR;
      __m512i acc00 = _mm512_setzero_si512();
      __m512i acc01 = _mm512_setzero_si512();
      __m512i acc10 = _mm512_setzero_si512();
      __m512i acc11 = _mm512_setzero_si512();
      __m512i acc20 = _mm512_setzero_si512();
      __m512i acc21 = _mm512_setzero_si512();
      __m512i acc30 = _mm512_setzero_si512();
      __m512i acc31 = _mm512_setzero_si512();

      const int8_t *a0Ptr = aRow0;
      const int8_t *a1Ptr = aRow1;
      const int8_t *a2Ptr = aRow2;
      const int8_t *a3Ptr = aRow3;
      const int8_t *bK = bPanel;
      for (int k = 0; k < kK; k += 2 * kKGroup,
               bK += 2 * kNR * kKGroup, a0Ptr += 2 * kKGroup,
               a1Ptr += 2 * kKGroup, a2Ptr += 2 * kKGroup,
               a3Ptr += 2 * kKGroup) {
        for (int u = 0; u < 2; ++u) {
          const int8_t *bStep = bK + u * kNR * kKGroup;
          const int8_t *a0Step = a0Ptr + u * kKGroup;
          const int8_t *a1Step = a1Ptr + u * kKGroup;
          const int8_t *a2Step = a2Ptr + u * kKGroup;
          const int8_t *a3Step = a3Ptr + u * kKGroup;
          __m512i b0 =
              _mm512_load_si512(reinterpret_cast<const __m512i *>(bStep));
          __m512i b1 =
              _mm512_load_si512(reinterpret_cast<const __m512i *>(bStep + 64));
          __m512i a0 =
              _mm512_set1_epi32(static_cast<int>(loadPackedA4(a0Step)));
          __m512i a1 =
              _mm512_set1_epi32(static_cast<int>(loadPackedA4(a1Step)));
          __m512i a2 =
              _mm512_set1_epi32(static_cast<int>(loadPackedA4(a2Step)));
          __m512i a3 =
              _mm512_set1_epi32(static_cast<int>(loadPackedA4(a3Step)));
          acc00 = _mm512_dpbusd_epi32(acc00, a0, b0);
          acc01 = _mm512_dpbusd_epi32(acc01, a0, b1);
          acc10 = _mm512_dpbusd_epi32(acc10, a1, b0);
          acc11 = _mm512_dpbusd_epi32(acc11, a1, b1);
          acc20 = _mm512_dpbusd_epi32(acc20, a2, b0);
          acc21 = _mm512_dpbusd_epi32(acc21, a2, b1);
          acc30 = _mm512_dpbusd_epi32(acc30, a3, b0);
          acc31 = _mm512_dpbusd_epi32(acc31, a3, b1);
        }
      }

      int32_t *cRow0 = c + (i + 0) * kN + j;
      int32_t *cRow1 = c + (i + 1) * kN + j;
      int32_t *cRow2 = c + (i + 2) * kN + j;
      int32_t *cRow3 = c + (i + 3) * kN + j;
      _mm512_store_si512(reinterpret_cast<__m512i *>(cRow0), acc00);
      _mm512_store_si512(reinterpret_cast<__m512i *>(cRow0 + 16), acc01);
      _mm512_store_si512(reinterpret_cast<__m512i *>(cRow1), acc10);
      _mm512_store_si512(reinterpret_cast<__m512i *>(cRow1 + 16), acc11);
      _mm512_store_si512(reinterpret_cast<__m512i *>(cRow2), acc20);
      _mm512_store_si512(reinterpret_cast<__m512i *>(cRow2 + 16), acc21);
      _mm512_store_si512(reinterpret_cast<__m512i *>(cRow3), acc30);
      _mm512_store_si512(reinterpret_cast<__m512i *>(cRow3 + 16), acc31);
    }
  }
}

void pinCurrentThreadToCpu(int cpu);

class GemmThreadTeam {
public:
  GemmThreadTeam(const int8_t *a, const int8_t *bPacked, int32_t *c,
                 int requestedThreads)
      : a(a), bPacked(bPacked), c(c),
        numThreads(std::min(std::max(1, requestedThreads), kM / kMR)),
        stop(false) {
    if (numThreads == 1)
      return;
    if (pthread_barrier_init(&startBarrier, nullptr, numThreads + 1) != 0 ||
        pthread_barrier_init(&doneBarrier, nullptr, numThreads + 1) != 0)
      throw std::runtime_error("failed to initialize pthread barriers");
    workers.reserve(numThreads);
    int rowBlocks = kM / kMR;
    for (int thread = 0; thread < numThreads; ++thread) {
      int blockBegin = (rowBlocks * thread) / numThreads;
      int blockEnd = (rowBlocks * (thread + 1)) / numThreads;
      int rowBegin = blockBegin * kMR;
      int rowEnd = blockEnd * kMR;
      workers.emplace_back([this, thread, rowBegin, rowEnd]() {
        pinCurrentThreadToCpu(thread);
        while (true) {
          pthread_barrier_wait(&startBarrier);
          if (stop.load(std::memory_order_acquire))
            break;
          cpu_i8_gemm_vnni(this->a, this->bPacked, this->c, rowBegin,
                            rowEnd);
          pthread_barrier_wait(&doneBarrier);
        }
      });
    }
  }

  GemmThreadTeam(const GemmThreadTeam &) = delete;
  GemmThreadTeam &operator=(const GemmThreadTeam &) = delete;

  ~GemmThreadTeam() {
    if (numThreads == 1)
      return;
    stop.store(true, std::memory_order_release);
    pthread_barrier_wait(&startBarrier);
    for (std::thread &worker : workers)
      worker.join();
    pthread_barrier_destroy(&startBarrier);
    pthread_barrier_destroy(&doneBarrier);
  }

  void runOnce() {
    if (numThreads == 1) {
      cpu_i8_gemm_vnni(a, bPacked, c, 0, kM);
      return;
    }
    pthread_barrier_wait(&startBarrier);
    pthread_barrier_wait(&doneBarrier);
  }

  int threads() const { return numThreads; }

private:
  const int8_t *a;
  const int8_t *bPacked;
  int32_t *c;
  int numThreads;
  std::atomic<bool> stop;
  pthread_barrier_t startBarrier;
  pthread_barrier_t doneBarrier;
  std::vector<std::thread> workers;
};

void clearOutput(int32_t *c) {
  std::fill(c, c + static_cast<std::size_t>(kM) * kN, 0);
}

void pinCurrentThreadToCpu(int cpu) {
#if defined(__linux__)
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(cpu, &cpuset);
  (void)pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
#else
  (void)cpu;
#endif
}

int main(int argc, char **argv) {
  try {
    Options options = parseOptions(argc, argv);
    if (!cpuSupportsVnni()) {
      std::cerr << "CPU does not report AVX-512 VNNI support\n";
      return 2;
    }

    std::vector<int8_t> a(kM * kK);
    std::vector<int8_t> b(kK * kN);
    AlignedPtr<int8_t> bPacked =
        makeAlignedBuffer<int8_t>(static_cast<std::size_t>(kN) * kK);
    AlignedPtr<int32_t> c =
        makeAlignedBuffer<int32_t>(static_cast<std::size_t>(kM) * kN);
    initializeInputs(a, b, bPacked.get());
    clearOutput(c.get());

    GemmThreadTeam team(a.data(), bPacked.get(), c.get(), options.threads);

    for (int i = 0; i < options.warmups; ++i)
      team.runOnce();

    double totalUs = 0.0;
    double minUs = std::numeric_limits<double>::infinity();
    double maxUs = 0.0;
    for (int i = 0; i < options.iterations; ++i) {
      auto start = std::chrono::steady_clock::now();
      team.runOnce();
      auto stop = std::chrono::steady_clock::now();
      double us = std::chrono::duration<double, std::micro>(stop - start).count();
      totalUs += us;
      minUs = std::min(minUs, us);
      maxUs = std::max(maxUs, us);
    }

    bool valid = !options.verify || verifySamples(a, b, c.get());
    double avgUs = totalUs / static_cast<double>(options.iterations);
    double gigaOps = (2.0 * kM * kN * kK) / (avgUs * 1000.0);

    std::cout << "backend=cpu\n";
    std::cout << "kernel=cpu_i8_gemm_vnni\n";
    std::cout << "shape=" << kM << "x" << kN << "x" << kK << "\n";
    std::cout << "dtype=i8xi8_to_i32\n";
    std::cout << "layout=A_row_major,B_packed_NR32_K4\n";
    std::cout << "threads=" << team.threads() << "\n";
    std::cout << "warmups=" << options.warmups << "\n";
    std::cout << "iterations=" << options.iterations << "\n";
    std::cout << "timing_domain=host_steady_clock\n";
    std::cout << "avg_us=" << avgUs << "\n";
    std::cout << "min_us=" << minUs << "\n";
    std::cout << "max_us=" << maxUs << "\n";
    std::cout << "gops=" << gigaOps << "\n";
    std::cout << "validation=" << (valid ? "PASS" : "FAIL") << "\n";
    return valid ? 0 : 1;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    usage(argv[0]);
    return 2;
  }
}
