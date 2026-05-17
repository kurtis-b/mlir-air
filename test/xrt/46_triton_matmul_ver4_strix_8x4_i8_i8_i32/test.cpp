//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "cxxopts.hpp"
#include <bits/stdc++.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdfloat>

#include "test_utils.h"

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

using A_DATATYPE = int8_t;
using B_DATATYPE = int8_t;
using C_DATATYPE = int32_t;

static inline int8_t random_int8_t() { return (int8_t)(rand() % 8); }

enum class BLayout { RowMajor, ColumnMajor };

static BLayout parse_b_layout(const std::string &layout) {
  if (layout == "row")
    return BLayout::RowMajor;
  if (layout == "column")
    return BLayout::ColumnMajor;
  throw std::runtime_error("--b-layout must be 'row' or 'column'");
}

static C_DATATYPE reference_element(const std::vector<A_DATATYPE> &a,
                                    const std::vector<B_DATATYPE> &b, int row,
                                    int col, int K, int N) {
  C_DATATYPE sum = 0;
  for (int k = 0; k < K; ++k)
    sum += static_cast<C_DATATYPE>(a[row * K + k]) *
           static_cast<C_DATATYPE>(b[k * N + col]);
  return sum;
}

static bool verify_samples(const std::vector<A_DATATYPE> &a,
                           const std::vector<B_DATATYPE> &b,
                           const std::vector<C_DATATYPE> &c, int M, int K,
                           int N) {
  const std::array<std::pair<int, int>, 9> samples = {
      {{0, 0},
       {0, std::min(N - 1, 31)},
       {std::min(M - 1, 3), std::min(N - 1, 5)},
       {std::min(M - 1, 17), std::min(N - 1, 19)},
       {std::min(M - 1, 127), std::min(N - 1, 64)},
       {std::min(M - 1, 255), std::min(N - 1, 255)},
       {std::min(M - 1, 511), std::min(N - 1, 17)},
       {std::min(M - 1, 700), std::min(N - 1, 901)},
       {M - 1, N - 1}}};
  bool valid = true;
  for (auto [row, col] : samples) {
    C_DATATYPE expected = reference_element(a, b, row, col, K, N);
    C_DATATYPE observed = c[row * N + col];
    if (expected == observed)
      continue;
    std::cerr << "mismatch at (" << row << ", " << col
              << "): expected=" << expected << " observed=" << observed
              << "\n";
    valid = false;
  }
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
      cxxopts::value<std::string>()->default_value("row"));
}

int main(int argc, const char *argv[]) {

  // Program arguments parsing
  cxxopts::Options options("Triton Matmul Profiling");
  cxxopts::ParseResult vm;
  add_default_options(options);
  test_utils::parse_options(argc, argv, options, vm);
  int verbosity = vm["verbosity"].as<int>();

  int M = vm["size_m"].as<int>();
  int K = vm["size_k"].as<int>();
  int N = vm["size_n"].as<int>();
  BLayout b_layout = parse_b_layout(vm["b-layout"].as<std::string>());

  int A_VOLUME = M * K;
  int B_VOLUME = K * N;
  int C_VOLUME = M * N;

  int A_SIZE = (A_VOLUME * sizeof(A_DATATYPE));
  int B_SIZE = (B_VOLUME * sizeof(B_DATATYPE));
  int C_SIZE = (C_VOLUME * sizeof(C_DATATYPE));

  srand(time(NULL));

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  // Start the XRT test code
  // Get a device handle
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  // Load the xclbin
  if (verbosity >= 1)
    std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << "\n";
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Kernel opcode: " << vm["kernel"].as<std::string>() << "\n";
  std::string Node = vm["kernel"].as<std::string>();

  // Get the kernel from the xclbin
  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node, verbosity](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 if (verbosity >= 1) {
                                   std::cout << "Name: " << name << std::endl;
                                 }
                                 return name.rfind(Node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  if (verbosity >= 1)
    std::cout << "Registering xclbin: " << vm["xclbin"].as<std::string>()
              << "\n";

  device.register_xclbin(xclbin);

  // get a hardware context
  if (verbosity >= 1)
    std::cout << "Getting hardware context.\n";
  xrt::hw_context context(device, xclbin.get_uuid());

  // get a kernel handle
  if (verbosity >= 1)
    std::cout << "Getting handle to kernel:" << kernelName << "\n";
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_a =
      xrt::bo(device, A_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_b =
      xrt::bo(device, B_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_c =
      xrt::bo(device, C_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  // Initialize input matrices with random int8 values
  A_DATATYPE *bufA = bo_a.map<A_DATATYPE *>();
  std::vector<A_DATATYPE> AVec(A_VOLUME);
  for (int i = 0; i < A_VOLUME; i++) {
    AVec[i] = random_int8_t();
  }
  memcpy(bufA, AVec.data(), (AVec.size() * sizeof(A_DATATYPE)));

  B_DATATYPE *bufB = bo_b.map<B_DATATYPE *>();
  std::vector<B_DATATYPE> BVec(B_VOLUME);
  for (int i = 0; i < B_VOLUME; i++) {
    BVec[i] = random_int8_t();
  }
  std::vector<B_DATATYPE> BDevice(B_VOLUME);
  if (b_layout == BLayout::RowMajor) {
    BDevice = BVec;
  } else {
    for (int k = 0; k < K; ++k)
      for (int n = 0; n < N; ++n)
        BDevice[n * K + k] = BVec[k * N + n];
  }
  memcpy(bufB, BDevice.data(), (BDevice.size() * sizeof(B_DATATYPE)));

  C_DATATYPE *bufC = bo_c.map<C_DATATYPE *>();
  std::vector<C_DATATYPE> CVec(C_VOLUME, 0);
  memcpy(bufC, CVec.data(), (CVec.size() * sizeof(C_DATATYPE)));

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

  // For int8 matmul: 2 ops (multiply + add) per element
  double macs = 2.0 * double(M) * double(K) * double(N);

  for (unsigned iter = 0; iter < num_iter; iter++) {

    if (verbosity >= 1) {
      std::cout << "Running Kernel.\n";
    }
    auto start = std::chrono::steady_clock::now();
    unsigned int opcode = 3;
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_a, bo_b, bo_c);
    run.wait();
    auto stop = std::chrono::steady_clock::now();

    if (iter < n_warmup_iterations) {
      /* Warmup iterations do not count towards average runtime. */
      continue;
    }

    double npu_time =
        std::chrono::duration<double, std::micro>(stop - start).count();

    npu_time_total += npu_time;
    npu_time_min = (npu_time < npu_time_min) ? npu_time : npu_time_min;
    npu_time_max = (npu_time > npu_time_max) ? npu_time : npu_time_max;
  }

  bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  memcpy(CVec.data(), bufC, (CVec.size() * sizeof(C_DATATYPE)));
  bool valid = verify_samples(AVec, BVec, CVec, M, K, N);

  double avg_us = npu_time_total / n_iterations;
  double avg_gops = macs / (1000 * avg_us);
  double max_gops = macs / (1000 * npu_time_min);
  double min_gops = macs / (1000 * npu_time_max);

  std::cout << std::endl
            << "Avg NPU matmul time: " << avg_us << "us." << std::endl;
  std::cout << "Avg NPU gflops: " << avg_gops << std::endl;

  std::cout << std::endl
            << "Min NPU matmul time: " << npu_time_min << "us." << std::endl;
  std::cout << "Max NPU gflops: " << max_gops << std::endl;

  std::cout << std::endl
            << "Max NPU matmul time: " << npu_time_max << "us." << std::endl;
  std::cout << "Min NPU gflops: " << min_gops << std::endl;

  std::cout << "backend=npu\n";
  std::cout << "shape=" << M << "x" << N << "x" << K << "\n";
  std::cout << "b_layout="
            << (b_layout == BLayout::RowMajor ? "row" : "column") << "\n";
  std::cout << "warmups=" << n_warmup_iterations << "\n";
  std::cout << "iterations=" << n_iterations << "\n";
  std::cout << "timing_domain=host_run_wait\n";
  std::cout << "avg_us=" << avg_us << "\n";
  std::cout << "min_us=" << npu_time_min << "\n";
  std::cout << "max_us=" << npu_time_max << "\n";
  std::cout << "gops=" << avg_gops << "\n";
  std::cout << "validation=" << (valid ? "PASS" : "FAIL") << "\n";

  return valid ? 0 : 1;
}
