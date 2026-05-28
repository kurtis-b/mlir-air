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
#include <utility>
#include <vector>

#include <immintrin.h>

namespace {

constexpr int kDefaultSize = 1024;
constexpr int kMR = 4;
constexpr int kNR = 32;
constexpr int kKGroup = 4;
constexpr int kDefaultThreads = 12;

struct Options {
  int m = kDefaultSize;
  int n = kDefaultSize;
  int k = kDefaultSize;
  int warmups = 0;
  int iterations = 1;
  int threads = kDefaultThreads;
  bool verify = true;
};

void usage(const char *argv0) {
  std::cout << "Usage: " << argv0 << " [--size N] [--m M] [--n N] [--k K]"
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

bool isPowerOfTwoGreaterThan512(int value) {
  return value > 512 && (value & (value - 1)) == 0;
}

void validateShape(const Options &options) {
  if (!isPowerOfTwoGreaterThan512(options.m) ||
      !isPowerOfTwoGreaterThan512(options.n) ||
      !isPowerOfTwoGreaterThan512(options.k))
    throw std::runtime_error(
        "M, N, and K must be powers of two greater than 512");
  if (options.m % kMR != 0 || options.n % kNR != 0 ||
      options.k % (2 * kKGroup) != 0)
    throw std::runtime_error("shape is not divisible by the VNNI tile shape");
}

Options parseOptions(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
      std::exit(0);
    }
    if (arg == "--size") {
      if (++i == argc)
        throw std::runtime_error("missing value for --size");
      int size = parseInt(argv[i], "--size", false);
      options.m = size;
      options.n = size;
      options.k = size;
      continue;
    }
    if (arg == "--m") {
      if (++i == argc)
        throw std::runtime_error("missing value for --m");
      options.m = parseInt(argv[i], "--m", false);
      continue;
    }
    if (arg == "--n") {
      if (++i == argc)
        throw std::runtime_error("missing value for --n");
      options.n = parseInt(argv[i], "--n", false);
      continue;
    }
    if (arg == "--k") {
      if (++i == argc)
        throw std::runtime_error("missing value for --k");
      options.k = parseInt(argv[i], "--k", false);
      continue;
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
  validateShape(options);
  return options;
}

void initializeInputs(std::vector<int8_t> &a, std::vector<int8_t> &b,
                      int8_t *bPacked, int m, int n, int kDim) {
  for (int i = 0; i < m; ++i) {
    for (int k = 0; k < kDim; ++k)
      a[static_cast<std::size_t>(i) * kDim + k] =
          static_cast<int8_t>((i * 3 + k * 5 + 1) & 7);
  }
  for (int k = 0; k < kDim; ++k) {
    for (int j = 0; j < n; ++j) {
      int8_t value = static_cast<int8_t>((k * 7 + j * 11 + 3) & 7);
      b[static_cast<std::size_t>(k) * n + j] = value;
    }
  }

  for (int panel = 0; panel < n; panel += kNR) {
    int8_t *panelBase =
        bPacked + static_cast<std::size_t>(panel / kNR) * kDim * kNR;
    for (int k = 0; k < kDim; k += kKGroup) {
      int8_t *kBase =
          panelBase + static_cast<std::size_t>(k / kKGroup) * kNR * kKGroup;
      for (int col = 0; col < kNR; ++col) {
        for (int byte = 0; byte < kKGroup; ++byte)
          kBase[col * kKGroup + byte] =
              b[static_cast<std::size_t>(k + byte) * n + panel + col];
      }
    }
  }
}

int32_t referenceElement(const std::vector<int8_t> &a,
                         const std::vector<int8_t> &b, int n, int kDim, int row,
                         int col) {
  int32_t sum = 0;
  for (int k = 0; k < kDim; ++k)
    sum += static_cast<int32_t>(a[static_cast<std::size_t>(row) * kDim + k]) *
           static_cast<int32_t>(b[static_cast<std::size_t>(k) * n + col]);
  return sum;
}

bool verifySamples(const std::vector<int8_t> &a, const std::vector<int8_t> &b,
                   const int32_t *c, int m, int n, int kDim) {
  std::vector<std::pair<int, int>> samples = {
      {0, 0},
      {0, std::min(n - 1, 31)},
      {std::min(m - 1, 3), std::min(n - 1, 5)},
      {m / 8, n / 16},
      {m / 4, n / 4},
      {m / 2, n / 2},
      {(3 * m) / 4, (7 * n) / 8},
      {m - 1, n - 1},
  };

  bool ok = true;
  for (const auto &sample : samples) {
    int row = sample.first;
    int col = sample.second;
    int32_t expected = referenceElement(a, b, n, kDim, row, col);
    int32_t observed = c[static_cast<std::size_t>(row) * n + col];
    if (expected == observed)
      continue;
    std::cerr << "mismatch at (" << row << ", " << col
              << "): expected=" << expected << " observed=" << observed << "\n";
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
                          target("avx512f,avx512bw,avx512vl,avx512vnni"))) void
cpu_i8_gemm_vnni(const int8_t *__restrict__ a,
                 const int8_t *__restrict__ bPacked, int32_t *__restrict__ c,
                 int n, int kDim, int rowBegin, int rowEnd) {
  for (int i = rowBegin; i < rowEnd; i += kMR) {
    const int8_t *aRow0 = a + static_cast<std::size_t>(i + 0) * kDim;
    const int8_t *aRow1 = a + static_cast<std::size_t>(i + 1) * kDim;
    const int8_t *aRow2 = a + static_cast<std::size_t>(i + 2) * kDim;
    const int8_t *aRow3 = a + static_cast<std::size_t>(i + 3) * kDim;
    for (int j = 0; j < n; j += kNR) {
      const int8_t *bPanel =
          bPacked + static_cast<std::size_t>(j / kNR) * kDim * kNR;
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
      for (int k = 0; k < kDim; k += 2 * kKGroup, bK += 2 * kNR * kKGroup,
               a0Ptr += 2 * kKGroup, a1Ptr += 2 * kKGroup, a2Ptr += 2 * kKGroup,
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

      int32_t *cRow0 = c + static_cast<std::size_t>(i + 0) * n + j;
      int32_t *cRow1 = c + static_cast<std::size_t>(i + 1) * n + j;
      int32_t *cRow2 = c + static_cast<std::size_t>(i + 2) * n + j;
      int32_t *cRow3 = c + static_cast<std::size_t>(i + 3) * n + j;
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
  GemmThreadTeam(const int8_t *a, const int8_t *bPacked, int32_t *c, int m,
                 int n, int kDim, int requestedThreads)
      : a(a), bPacked(bPacked), c(c), n(n), kDim(kDim), mRows(m),
        numThreads(std::min(std::max(1, requestedThreads), m / kMR)),
        stop(false) {
    if (numThreads == 1)
      return;
    if (pthread_barrier_init(&startBarrier, nullptr, numThreads + 1) != 0 ||
        pthread_barrier_init(&doneBarrier, nullptr, numThreads + 1) != 0)
      throw std::runtime_error("failed to initialize pthread barriers");
    workers.reserve(numThreads);
    int rowBlocks = m / kMR;
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
          cpu_i8_gemm_vnni(this->a, this->bPacked, this->c, this->n, this->kDim,
                           rowBegin, rowEnd);
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
      cpu_i8_gemm_vnni(a, bPacked, c, n, kDim, 0, mRows);
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
  int n;
  int kDim;
  int mRows = 0;
  int numThreads;
  std::atomic<bool> stop;
  pthread_barrier_t startBarrier;
  pthread_barrier_t doneBarrier;
  std::vector<std::thread> workers;
};

void clearOutput(int32_t *c, int m, int n) {
  std::fill(c, c + static_cast<std::size_t>(m) * n, 0);
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

    std::vector<int8_t> a(static_cast<std::size_t>(options.m) * options.k);
    std::vector<int8_t> b(static_cast<std::size_t>(options.k) * options.n);
    AlignedPtr<int8_t> bPacked = makeAlignedBuffer<int8_t>(
        static_cast<std::size_t>(options.n) * options.k);
    AlignedPtr<int32_t> c = makeAlignedBuffer<int32_t>(
        static_cast<std::size_t>(options.m) * options.n);
    initializeInputs(a, b, bPacked.get(), options.m, options.n, options.k);
    clearOutput(c.get(), options.m, options.n);

    GemmThreadTeam team(a.data(), bPacked.get(), c.get(), options.m, options.n,
                        options.k, options.threads);

    for (int i = 0; i < options.warmups; ++i)
      team.runOnce();

    double totalUs = 0.0;
    double minUs = std::numeric_limits<double>::infinity();
    double maxUs = 0.0;
    for (int i = 0; i < options.iterations; ++i) {
      auto start = std::chrono::steady_clock::now();
      team.runOnce();
      auto stop = std::chrono::steady_clock::now();
      double us =
          std::chrono::duration<double, std::micro>(stop - start).count();
      totalUs += us;
      minUs = std::min(minUs, us);
      maxUs = std::max(maxUs, us);
    }

    bool valid = !options.verify ||
                 verifySamples(a, b, c.get(), options.m, options.n, options.k);
    double avgUs = totalUs / static_cast<double>(options.iterations);
    double gigaOps =
        (2.0 * options.m * options.n * options.k) / (avgUs * 1000.0);

    std::cout << "backend=cpu\n";
    std::cout << "kernel=cpu_i8_gemm_vnni\n";
    std::cout << "shape=" << options.m << "x" << options.n << "x" << options.k
              << "\n";
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
