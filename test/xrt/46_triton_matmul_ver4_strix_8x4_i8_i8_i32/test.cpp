//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "cxxopts.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "test_utils.h"

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

using A_DATATYPE = int8_t;
using B_DATATYPE = int8_t;

constexpr double kDefaultAcceptanceTops = 36.15;

enum class BLayout { RowMajor, ColumnMajor };
enum class OutputType { Int32, Int8 };
enum class ValidationMode { None, Samples, Full };

static BLayout parse_b_layout(const std::string &layout) {
  if (layout == "row")
    return BLayout::RowMajor;
  if (layout == "column")
    return BLayout::ColumnMajor;
  throw std::runtime_error("--b-layout must be 'row' or 'column'");
}

static OutputType parse_output_type(const std::string &type) {
  if (type == "int32")
    return OutputType::Int32;
  if (type == "int8")
    return OutputType::Int8;
  throw std::runtime_error("--output-type must be 'int32' or 'int8'");
}

static ValidationMode parse_validation(const std::string &mode) {
  if (mode == "none")
    return ValidationMode::None;
  if (mode == "samples")
    return ValidationMode::Samples;
  if (mode == "full")
    return ValidationMode::Full;
  throw std::runtime_error("--validation must be 'none', 'samples', or 'full'");
}

static const char *layout_name(BLayout layout) {
  return layout == BLayout::RowMajor ? "row" : "column";
}

static const char *output_type_name(OutputType type) {
  return type == OutputType::Int32 ? "int32" : "int8";
}

static const char *validation_name(ValidationMode mode) {
  switch (mode) {
  case ValidationMode::None:
    return "none";
  case ValidationMode::Samples:
    return "samples";
  case ValidationMode::Full:
    return "full";
  }
  return "unknown";
}

template <typename T> static std::string printable_value(T value) {
  if constexpr (std::is_same_v<T, int8_t>)
    return std::to_string(static_cast<int>(value));
  else
    return std::to_string(value);
}

template <typename C_DATATYPE>
static C_DATATYPE cast_reference_value(int64_t sum) {
  if constexpr (std::is_same_v<C_DATATYPE, int8_t>) {
    auto wrapped = static_cast<uint8_t>(sum & 0xff);
    return std::bit_cast<int8_t>(wrapped);
  } else {
    return static_cast<C_DATATYPE>(sum);
  }
}

template <typename C_DATATYPE>
static C_DATATYPE reference_element(const std::vector<A_DATATYPE> &a,
                                    const std::vector<B_DATATYPE> &b, int row,
                                    int col, int K, int N) {
  int64_t sum = 0;
  for (int k = 0; k < K; ++k)
    sum += static_cast<int64_t>(a[row * K + k]) *
           static_cast<int64_t>(b[k * N + col]);
  return cast_reference_value<C_DATATYPE>(sum);
}

static std::vector<std::pair<int, int>> sample_points(int M, int N,
                                                       unsigned requested,
                                                       unsigned seed) {
  std::vector<std::pair<int, int>> points = {
      {0, 0},
      {0, std::min(N - 1, 31)},
      {std::min(M - 1, 3), std::min(N - 1, 5)},
      {std::min(M - 1, 17), std::min(N - 1, 19)},
      {std::min(M - 1, 127), std::min(N - 1, 64)},
      {std::min(M - 1, 255), std::min(N - 1, 255)},
      {std::min(M - 1, 511), std::min(N - 1, 17)},
      {std::min(M - 1, 700), std::min(N - 1, 901)},
      {M - 1, N - 1},
  };
  std::mt19937 rng(seed ^ 0x9e3779b9U);
  std::uniform_int_distribution<int> row_dist(0, M - 1);
  std::uniform_int_distribution<int> col_dist(0, N - 1);
  while (points.size() < requested)
    points.emplace_back(row_dist(rng), col_dist(rng));
  std::sort(points.begin(), points.end());
  points.erase(std::unique(points.begin(), points.end()), points.end());
  return points;
}

template <typename C_DATATYPE>
static bool verify_outputs(const std::vector<A_DATATYPE> &a,
                           const std::vector<B_DATATYPE> &b,
                           const std::vector<C_DATATYPE> &c, int M, int K,
                           int N, ValidationMode mode,
                           unsigned validation_samples, unsigned seed) {
  if (mode == ValidationMode::None)
    return true;

  bool valid = true;
  auto check = [&](int row, int col) {
    C_DATATYPE expected = reference_element<C_DATATYPE>(a, b, row, col, K, N);
    C_DATATYPE observed = c[row * N + col];
    if (expected == observed)
      return;
    std::cerr << "mismatch at (" << row << ", " << col
              << "): expected=" << printable_value(expected)
              << " observed=" << printable_value(observed) << "\n";
    valid = false;
  };

  if (mode == ValidationMode::Full) {
    for (int row = 0; row < M; ++row)
      for (int col = 0; col < N; ++col)
        check(row, col);
    return valid;
  }

  for (auto [row, col] : sample_points(M, N, validation_samples, seed))
    check(row, col);
  return valid;
}

void add_default_options(cxxopts::Options &options) {
  options.add_options()("help,h", "produce help message")(
      "xclbin,x", "the input xclbin path", cxxopts::value<std::string>())(
      "kernel,k", "the kernel name in the XCLBIN (for instance PP_PRE_FD)",
      cxxopts::value<std::string>())("verbosity,v",
                                     "the verbosity of the output",
                                     cxxopts::value<int>()->default_value("0"))(
      "instr,i",
      "path of file containing userspace instructions to be sent to the LX6",
      cxxopts::value<std::string>())("size_m,M", "Matrix size M",
                                     cxxopts::value<int>())(
      "size_n,N", "Matrix size N", cxxopts::value<int>())(
      "size_k,K", "Matrix size K", cxxopts::value<int>())(
      "warmups", "Warmup iterations",
      cxxopts::value<unsigned>()->default_value("10"))(
      "iterations", "Timed iterations",
      cxxopts::value<unsigned>()->default_value("20"))(
      "b-layout", "Device B buffer layout: row or column",
      cxxopts::value<std::string>()->default_value("row"))(
      "output-type", "Device C element type: int32 or int8",
      cxxopts::value<std::string>()->default_value("int32"))(
      "validation", "Validation mode: samples, full, or none",
      cxxopts::value<std::string>()->default_value("samples"))(
      "validation-samples", "Sample count for sampled validation",
      cxxopts::value<unsigned>()->default_value("64"))(
      "seed", "Deterministic input RNG seed",
      cxxopts::value<unsigned>()->default_value("42"))(
      "acceptance-tops", "Acceptance threshold in TOPS",
      cxxopts::value<double>()->default_value("36.15"))(
      "require-acceptance", "Fail when average TOPS is below acceptance",
      cxxopts::value<bool>()->default_value("false"));
}

template <typename C_DATATYPE>
static int run_profile(const cxxopts::ParseResult &vm, OutputType output_type,
                       BLayout b_layout, ValidationMode validation_mode,
                       int verbosity) {
  int M = vm["size_m"].as<int>();
  int K = vm["size_k"].as<int>();
  int N = vm["size_n"].as<int>();
  if (M <= 0 || K <= 0 || N <= 0) {
    std::cerr << "M, K, and N must be greater than zero\n";
    return 2;
  }

  size_t A_VOLUME = static_cast<size_t>(M) * K;
  size_t B_VOLUME = static_cast<size_t>(K) * N;
  size_t B_STORAGE_VOLUME =
      b_layout == BLayout::ColumnMajor ? B_VOLUME * 4 : B_VOLUME;
  size_t C_VOLUME = static_cast<size_t>(M) * N;

  size_t A_SIZE = A_VOLUME * sizeof(A_DATATYPE);
  size_t B_SIZE = B_STORAGE_VOLUME * sizeof(B_DATATYPE);
  size_t C_SIZE = C_VOLUME * sizeof(C_DATATYPE);

  unsigned seed = vm["seed"].as<unsigned>();
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(0, 7);

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  if (verbosity >= 1)
    std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << "\n";
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Kernel opcode: " << vm["kernel"].as<std::string>() << "\n";
  std::string Node = vm["kernel"].as<std::string>();

  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node, verbosity](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 if (verbosity >= 1)
                                   std::cout << "Name: " << name << std::endl;
                                 return name.rfind(Node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  if (verbosity >= 1)
    std::cout << "Registering xclbin: " << vm["xclbin"].as<std::string>()
              << "\n";

  device.register_xclbin(xclbin);

  if (verbosity >= 1)
    std::cout << "Getting hardware context.\n";
  xrt::hw_context context(device, xclbin.get_uuid());

  if (verbosity >= 1)
    std::cout << "Getting handle to kernel:" << kernelName << "\n";
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_a = xrt::bo(device, A_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_b = xrt::bo(device, B_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_c = xrt::bo(device, C_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  A_DATATYPE *bufA = bo_a.map<A_DATATYPE *>();
  std::vector<A_DATATYPE> AVec(A_VOLUME);
  for (auto &value : AVec)
    value = static_cast<A_DATATYPE>(dist(rng));
  memcpy(bufA, AVec.data(), AVec.size() * sizeof(A_DATATYPE));

  B_DATATYPE *bufB = bo_b.map<B_DATATYPE *>();
  std::vector<B_DATATYPE> BVec(B_VOLUME);
  for (auto &value : BVec)
    value = static_cast<B_DATATYPE>(dist(rng));
  std::vector<B_DATATYPE> BDevice(B_STORAGE_VOLUME, 0);
  if (b_layout == BLayout::RowMajor) {
    BDevice = BVec;
  } else {
    for (int k = 0; k < K; ++k)
      for (int n = 0; n < N; ++n)
        BDevice[(static_cast<size_t>(n) * K + k) * 4] = BVec[k * N + n];
  }
  memcpy(bufB, BDevice.data(), BDevice.size() * sizeof(B_DATATYPE));

  C_DATATYPE *bufC = bo_c.map<C_DATATYPE *>();
  std::vector<C_DATATYPE> CVec(C_VOLUME, 0);
  memcpy(bufC, CVec.data(), CVec.size() * sizeof(C_DATATYPE));

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned n_iterations = vm["iterations"].as<unsigned>();
  unsigned n_warmup_iterations = vm["warmups"].as<unsigned>();
  if (n_iterations == 0) {
    std::cerr << "--iterations must be greater than zero\n";
    return 2;
  }
  unsigned num_iter = n_iterations + n_warmup_iterations;
  double npu_time_total = 0;
  double npu_time_min = std::numeric_limits<double>::infinity();
  double npu_time_max = 0;

  double ops = 2.0 * double(M) * double(K) * double(N);

  for (unsigned iter = 0; iter < num_iter; iter++) {
    if (verbosity >= 1)
      std::cout << "Running Kernel.\n";
    auto start = std::chrono::steady_clock::now();
    unsigned int opcode = 3;
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_a, bo_b, bo_c);
    run.wait();
    auto stop = std::chrono::steady_clock::now();

    if (iter < n_warmup_iterations)
      continue;

    double npu_time =
        std::chrono::duration<double, std::micro>(stop - start).count();
    npu_time_total += npu_time;
    npu_time_min = std::min(npu_time_min, npu_time);
    npu_time_max = std::max(npu_time_max, npu_time);
  }

  bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  memcpy(CVec.data(), bufC, CVec.size() * sizeof(C_DATATYPE));
  bool valid = verify_outputs(AVec, BVec, CVec, M, K, N, validation_mode,
                              vm["validation-samples"].as<unsigned>(), seed);

  double avg_us = npu_time_total / n_iterations;
  double avg_gops = ops / (1000 * avg_us);
  double max_gops = ops / (1000 * npu_time_min);
  double min_gops = ops / (1000 * npu_time_max);
  double avg_tops = avg_gops / 1000.0;
  double max_tops = max_gops / 1000.0;
  double min_tops = min_gops / 1000.0;
  double acceptance_tops = vm["acceptance-tops"].as<double>();
  bool meets_acceptance = avg_tops >= acceptance_tops;

  std::cout << std::endl
            << "Avg NPU matmul time: " << avg_us << "us." << std::endl;
  std::cout << "Avg NPU gflops: " << avg_gops << std::endl;
  std::cout << "Avg NPU TOPS: " << avg_tops << std::endl;

  std::cout << std::endl
            << "Min NPU matmul time: " << npu_time_min << "us." << std::endl;
  std::cout << "Max NPU gflops: " << max_gops << std::endl;
  std::cout << "Max NPU TOPS: " << max_tops << std::endl;

  std::cout << std::endl
            << "Max NPU matmul time: " << npu_time_max << "us." << std::endl;
  std::cout << "Min NPU gflops: " << min_gops << std::endl;
  std::cout << "Min NPU TOPS: " << min_tops << std::endl;

  std::cout << "backend=npu\n";
  std::cout << "shape=" << M << "x" << K << "x" << N << "\n";
  std::cout << "output_type=" << output_type_name(output_type) << "\n";
  std::cout << "b_layout=" << layout_name(b_layout) << "\n";
  std::cout << "warmups=" << n_warmup_iterations << "\n";
  std::cout << "iterations=" << n_iterations << "\n";
  std::cout << "validation_mode=" << validation_name(validation_mode) << "\n";
  std::cout << "validation_samples=" << vm["validation-samples"].as<unsigned>()
            << "\n";
  std::cout << "rng_seed=" << seed << "\n";
  std::cout << "timing_domain=host_run_wait\n";
  std::cout << "avg_us=" << avg_us << "\n";
  std::cout << "min_us=" << npu_time_min << "\n";
  std::cout << "max_us=" << npu_time_max << "\n";
  std::cout << "gops=" << avg_gops << "\n";
  std::cout << "avg_tops=" << avg_tops << "\n";
  std::cout << "min_tops=" << min_tops << "\n";
  std::cout << "max_tops=" << max_tops << "\n";
  std::cout << "acceptance_tops=" << acceptance_tops << "\n";
  std::cout << "meets_acceptance=" << (meets_acceptance ? "yes" : "no")
            << "\n";
  std::cout << "validation=" << (valid ? "PASS" : "FAIL") << "\n";

  if (!valid)
    return 1;
  if (vm["require-acceptance"].as<bool>() && !meets_acceptance)
    return 1;
  return 0;
}

int main(int argc, const char *argv[]) {
  cxxopts::Options options("Triton Matmul Profiling");
  cxxopts::ParseResult vm;
  add_default_options(options);
  test_utils::parse_options(argc, argv, options, vm);
  int verbosity = vm["verbosity"].as<int>();

  BLayout b_layout = parse_b_layout(vm["b-layout"].as<std::string>());
  OutputType output_type = parse_output_type(vm["output-type"].as<std::string>());
  ValidationMode validation_mode =
      parse_validation(vm["validation"].as<std::string>());

  if (output_type == OutputType::Int8)
    return run_profile<int8_t>(vm, output_type, b_layout, validation_mode,
                               verbosity);
  return run_profile<int32_t>(vm, output_type, b_layout, validation_mode,
                              verbosity);
}
