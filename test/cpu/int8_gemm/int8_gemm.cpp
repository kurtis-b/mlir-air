//===- int8_gemm.cpp -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <immintrin.h>

namespace {

constexpr int kM = 1024;
constexpr int kN = 1024;
constexpr int kK = 1024;

struct Options {
  int warmups = 0;
  int iterations = 1;
  bool verify = true;
};

void usage(const char *argv0) {
  std::cout << "Usage: " << argv0 << " [--warmups N] [--iterations N]"
            << " [--no-verify]\n";
}

int parseInt(const char *value, const char *name) {
  char *end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (*value == '\0' || *end != '\0' || parsed < 0 ||
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
                      std::vector<int8_t> &bt) {
  for (int i = 0; i < kM; ++i) {
    for (int k = 0; k < kK; ++k) {
      a[i * kK + k] = static_cast<int8_t>((i * 3 + k * 5 + 1) & 7);
    }
  }
  for (int k = 0; k < kK; ++k) {
    for (int j = 0; j < kN; ++j) {
      int8_t value = static_cast<int8_t>((k * 7 + j * 11 + 3) & 7);
      b[k * kN + j] = value;
      bt[j * kK + k] = value;
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
                   const std::vector<int32_t> &c) {
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
void cpu_i8_gemm_vnni(const int8_t *a, const int8_t *bt, int32_t *c) {
  for (int i = 0; i < kM; ++i) {
    const int8_t *aRow = a + i * kK;
    for (int j = 0; j < kN; ++j) {
      const int8_t *bCol = bt + j * kK;
      __m512i acc = _mm512_setzero_si512();

      for (int k = 0; k < kK; k += 64) {
        __m512i avec =
            _mm512_loadu_si512(reinterpret_cast<const __m512i *>(aRow + k));
        __m512i bvec =
            _mm512_loadu_si512(reinterpret_cast<const __m512i *>(bCol + k));
        acc = _mm512_dpbusd_epi32(acc, avec, bvec);
      }

      c[i * kN + j] = _mm512_reduce_add_epi32(acc);
    }
  }
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
    std::vector<int8_t> bt(kN * kK);
    std::vector<int32_t> c(kM * kN);
    initializeInputs(a, b, bt);

    for (int i = 0; i < options.warmups; ++i)
      cpu_i8_gemm_vnni(a.data(), bt.data(), c.data());

    double totalUs = 0.0;
    double minUs = std::numeric_limits<double>::infinity();
    double maxUs = 0.0;
    for (int i = 0; i < options.iterations; ++i) {
      auto start = std::chrono::steady_clock::now();
      cpu_i8_gemm_vnni(a.data(), bt.data(), c.data());
      auto stop = std::chrono::steady_clock::now();
      double us = std::chrono::duration<double, std::micro>(stop - start).count();
      totalUs += us;
      minUs = std::min(minUs, us);
      maxUs = std::max(maxUs, us);
    }

    bool valid = !options.verify || verifySamples(a, b, c);
    double avgUs = totalUs / static_cast<double>(options.iterations);
    double gigaOps = (2.0 * kM * kN * kK) / (avgUs * 1000.0);

    std::cout << "backend=cpu\n";
    std::cout << "kernel=cpu_i8_gemm_vnni\n";
    std::cout << "shape=" << kM << "x" << kN << "x" << kK << "\n";
    std::cout << "dtype=i8xi8_to_i32\n";
    std::cout << "layout=A_row_major,B_transposed_for_cpu_hot_loop\n";
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
