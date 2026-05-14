// SPDX-License-Identifier: MIT

#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_hip(hipError_t status, const char *what) {
  if (status == hipSuccess)
    return;
  std::ostringstream os;
  os << what << ": " << hipGetErrorName(status) << " ("
     << hipGetErrorString(status) << ")";
  throw std::runtime_error(os.str());
}

void check_rocblas(rocblas_status status, const char *what) {
  if (status == rocblas_status_success)
    return;
  std::ostringstream os;
  os << what << ": rocblas_status=" << static_cast<int>(status);
  throw std::runtime_error(os.str());
}

struct Args {
  int M = 0;
  int K = 0;
  int H = 0;
  int N = 0;
  int warmup = 0;
  int iterations = 1;
  std::string input;
  std::string prefill_weight;
  std::string decode_weight;
  std::string prefill_output;
  std::string output;
};

int parse_int(const std::map<std::string, std::string> &values,
              const std::string &name) {
  auto it = values.find(name);
  if (it == values.end())
    throw std::runtime_error("missing required argument: --" + name);
  return std::stoi(it->second);
}

std::string parse_string(const std::map<std::string, std::string> &values,
                         const std::string &name) {
  auto it = values.find(name);
  if (it == values.end())
    throw std::runtime_error("missing required argument: --" + name);
  return it->second;
}

Args parse_args(int argc, char **argv) {
  std::map<std::string, std::string> values;
  for (int i = 1; i < argc; i += 2) {
    std::string key = argv[i];
    if (key.rfind("--", 0) != 0 || i + 1 >= argc)
      throw std::runtime_error("arguments must be --name value pairs");
    values[key.substr(2)] = argv[i + 1];
  }
  Args args;
  args.M = parse_int(values, "M");
  args.K = parse_int(values, "K");
  args.H = parse_int(values, "H");
  args.N = parse_int(values, "N");
  args.warmup = parse_int(values, "warmup");
  args.iterations = parse_int(values, "iterations");
  args.input = parse_string(values, "input");
  args.prefill_weight = parse_string(values, "prefill-weight");
  args.decode_weight = parse_string(values, "decode-weight");
  args.prefill_output = parse_string(values, "prefill-output");
  args.output = parse_string(values, "output");
  if (args.M <= 0 || args.K <= 0 || args.H <= 0 || args.N <= 0 ||
      args.iterations <= 0 || args.warmup < 0)
    throw std::runtime_error("shape and iteration arguments must be positive");
  return args;
}

std::vector<uint16_t> read_u16(const std::string &path, size_t expected_count) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("failed to open input file: " + path);
  std::vector<uint16_t> data(expected_count);
  in.read(reinterpret_cast<char *>(data.data()),
          static_cast<std::streamsize>(data.size() * sizeof(uint16_t)));
  if (in.gcount() !=
      static_cast<std::streamsize>(data.size() * sizeof(uint16_t)))
    throw std::runtime_error("input file has unexpected byte count: " + path);
  return data;
}

void write_u16(const std::string &path, const std::vector<uint16_t> &data) {
  std::ofstream out(path, std::ios::binary);
  if (!out)
    throw std::runtime_error("failed to open output file: " + path);
  out.write(reinterpret_cast<const char *>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(uint16_t)));
}

template <typename T>
T *device_alloc_copy(const std::vector<T> &host, const char *label) {
  T *device = nullptr;
  check_hip(hipMalloc(&device, host.size() * sizeof(T)), label);
  check_hip(hipMemcpy(device, host.data(), host.size() * sizeof(T),
                      hipMemcpyHostToDevice),
            "hipMemcpy H2D");
  return device;
}

void copy_to_host(std::vector<uint16_t> &host, const uint16_t *device) {
  check_hip(hipMemcpy(host.data(), device, host.size() * sizeof(uint16_t),
                      hipMemcpyDeviceToHost),
            "hipMemcpy D2H");
}

std::string json_escape(const std::string &value) {
  std::ostringstream os;
  for (char ch : value) {
    switch (ch) {
    case '\\':
      os << "\\\\";
      break;
    case '"':
      os << "\\\"";
      break;
    case '\n':
      os << "\\n";
      break;
    default:
      os << ch;
    }
  }
  return os.str();
}

void gemm_row_major_bf16(rocblas_handle handle, int m, int k, int n,
                         const uint16_t *a_row_major,
                         const uint16_t *b_row_major, uint16_t *c_row_major) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  // rocBLAS is column-major. Row-major C[M,N] = A[M,K] * B[K,N]
  // is computed as C_col[N,M] = B_col[N,K] * A_col[K,M].
  check_rocblas(
      rocblas_gemm_ex(handle, rocblas_operation_none, rocblas_operation_none, n,
                      m, k, &alpha, b_row_major, rocblas_datatype_bf16_r, n,
                      a_row_major, rocblas_datatype_bf16_r, k, &beta,
                      c_row_major, rocblas_datatype_bf16_r, n, c_row_major,
                      rocblas_datatype_bf16_r, n, rocblas_datatype_f32_r,
                      rocblas_gemm_algo_standard, 0, 0),
      "rocblas_gemm_ex");
}

struct Sample {
  float prefill_ms = 0.0f;
  float decode_ms = 0.0f;
  float end_to_end_ms = 0.0f;
};

Sample run_pipeline(rocblas_handle handle, const Args &args, uint16_t *input,
                    uint16_t *prefill_weight, uint16_t *decode_weight,
                    uint16_t *prefill_output, uint16_t *output, bool timed) {
  hipEvent_t start = nullptr;
  hipEvent_t mid = nullptr;
  hipEvent_t end = nullptr;
  if (timed) {
    check_hip(hipEventCreate(&start), "hipEventCreate(start)");
    check_hip(hipEventCreate(&mid), "hipEventCreate(mid)");
    check_hip(hipEventCreate(&end), "hipEventCreate(end)");
    check_hip(hipEventRecord(start, nullptr), "hipEventRecord(start)");
  }
  gemm_row_major_bf16(handle, args.M, args.K, args.H, input, prefill_weight,
                      prefill_output);
  if (timed)
    check_hip(hipEventRecord(mid, nullptr), "hipEventRecord(mid)");
  const uint16_t *decode_input =
      prefill_output + static_cast<size_t>(args.M - 1) * args.H;
  gemm_row_major_bf16(handle, 1, args.H, args.N, decode_input, decode_weight,
                      output);
  if (!timed) {
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize");
    return {};
  }
  check_hip(hipEventRecord(end, nullptr), "hipEventRecord(end)");
  check_hip(hipEventSynchronize(end), "hipEventSynchronize(end)");
  Sample sample;
  check_hip(hipEventElapsedTime(&sample.prefill_ms, start, mid),
            "hipEventElapsedTime(prefill)");
  check_hip(hipEventElapsedTime(&sample.decode_ms, mid, end),
            "hipEventElapsedTime(decode)");
  check_hip(hipEventElapsedTime(&sample.end_to_end_ms, start, end),
            "hipEventElapsedTime(end_to_end)");
  check_hip(hipEventDestroy(start), "hipEventDestroy(start)");
  check_hip(hipEventDestroy(mid), "hipEventDestroy(mid)");
  check_hip(hipEventDestroy(end), "hipEventDestroy(end)");
  return sample;
}

float mean(const std::vector<float> &values) {
  if (values.empty())
    return 0.0f;
  return std::accumulate(values.begin(), values.end(), 0.0f) /
         static_cast<float>(values.size());
}

} // namespace

int main(int argc, char **argv) {
  try {
    Args args = parse_args(argc, argv);
    check_hip(hipSetDevice(0), "hipSetDevice");
    hipDeviceProp_t prop{};
    check_hip(hipGetDeviceProperties(&prop, 0), "hipGetDeviceProperties");

    auto input_host =
        read_u16(args.input, static_cast<size_t>(args.M) * args.K);
    auto prefill_weight_host =
        read_u16(args.prefill_weight, static_cast<size_t>(args.K) * args.H);
    auto decode_weight_host =
        read_u16(args.decode_weight, static_cast<size_t>(args.H) * args.N);
    std::vector<uint16_t> prefill_output_host(static_cast<size_t>(args.M) *
                                              args.H);
    std::vector<uint16_t> output_host(args.N);

    uint16_t *input = device_alloc_copy(input_host, "hipMalloc(input)");
    uint16_t *prefill_weight =
        device_alloc_copy(prefill_weight_host, "hipMalloc(prefill_weight)");
    uint16_t *decode_weight =
        device_alloc_copy(decode_weight_host, "hipMalloc(decode_weight)");
    uint16_t *prefill_output = nullptr;
    uint16_t *output = nullptr;
    check_hip(hipMalloc(&prefill_output,
                        prefill_output_host.size() * sizeof(uint16_t)),
              "hipMalloc(prefill_output)");
    check_hip(hipMalloc(&output, output_host.size() * sizeof(uint16_t)),
              "hipMalloc(output)");

    rocblas_handle handle = nullptr;
    check_rocblas(rocblas_create_handle(&handle), "rocblas_create_handle");
    check_rocblas(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host),
                  "rocblas_set_pointer_mode");

    for (int i = 0; i < args.warmup; ++i)
      run_pipeline(handle, args, input, prefill_weight, decode_weight,
                   prefill_output, output, false);

    std::vector<float> prefill_samples;
    std::vector<float> decode_samples;
    std::vector<float> e2e_samples;
    for (int i = 0; i < args.iterations; ++i) {
      Sample sample = run_pipeline(handle, args, input, prefill_weight,
                                   decode_weight, prefill_output, output, true);
      prefill_samples.push_back(sample.prefill_ms);
      decode_samples.push_back(sample.decode_ms);
      e2e_samples.push_back(sample.end_to_end_ms);
    }

    copy_to_host(prefill_output_host, prefill_output);
    copy_to_host(output_host, output);
    write_u16(args.prefill_output, prefill_output_host);
    write_u16(args.output, output_host);

    check_rocblas(rocblas_destroy_handle(handle), "rocblas_destroy_handle");
    check_hip(hipFree(input), "hipFree(input)");
    check_hip(hipFree(prefill_weight), "hipFree(prefill_weight)");
    check_hip(hipFree(decode_weight), "hipFree(decode_weight)");
    check_hip(hipFree(prefill_output), "hipFree(prefill_output)");
    check_hip(hipFree(output), "hipFree(output)");

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "{";
    std::cout << "\"framework\":\"rocBLAS\",";
    std::cout << "\"device\":\"gpu\",";
    std::cout << "\"hip_device_name\":\"" << json_escape(prop.name) << "\",";
    std::cout << "\"mean_end_to_end_ms\":" << mean(e2e_samples) << ",";
    std::cout << "\"mean_prefill_ms\":" << mean(prefill_samples) << ",";
    std::cout << "\"mean_decode_ms\":" << mean(decode_samples) << ",";
    std::cout << "\"iterations\":" << args.iterations << ",";
    std::cout << "\"warmup\":" << args.warmup << ",";
    std::cout
        << "\"device_execution_proof\":\"rocBLAS BF16 GEMM calls on HIP device "
        << json_escape(prop.name)
        << "; weights and inputs copied before timed loop; final readback "
           "after timed loop\"";
    std::cout << "}\n";
    return 0;
  } catch (const std::exception &exc) {
    std::cerr << exc.what() << "\n";
    return 1;
  }
}
