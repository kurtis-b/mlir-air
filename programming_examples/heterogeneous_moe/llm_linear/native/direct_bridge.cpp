// SPDX-License-Identifier: MIT
//
// Native bridge for the llm_linear direct GPU/NPU handoff path.
//
// The bridge deliberately makes XRT BOs the owning allocations.  HIP imports
// those BO export handles and GPU kernels receive the mapped device pointers in
// their memref descriptors.  This keeps the interstage tensor in one shared BO
// across GPU->NPU and NPU->GPU runs; host readback is only used for initial
// input/weight setup and final/reporting outputs.

#include <hip/hip_runtime_api.h>

#include <xrt/experimental/xrt_hw_context.h>
#include <xrt/experimental/xrt_xclbin.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

extern "C" {

struct LlmLinearDirectRunConfig {
  uint32_t abi_version;
  uint32_t direction;
  uint32_t dtype;
  uint32_t reserved;
  uint64_t m;
  uint64_t k;
  uint64_t h;
  uint64_t n;
  const void *input;
  const void *prefill_weights;
  const void *decode_weights;
  void *output;
  void *prefill_output;
  void *decode_input;
  const char *gpu_prefill_so;
  const char *gpu_decode_so;
  const char *npu_prefill_xclbin;
  const char *npu_prefill_insts;
  const char *npu_decode_xclbin;
  const char *npu_decode_insts;
  const char *npu_kernel_name;
};

struct LlmLinearDirectRunResult {
  uint32_t abi_version;
  uint32_t bo_flag;
  uint32_t import_method;
  uint32_t reserved;
  uint64_t prefill_us;
  uint64_t handoff_us;
  uint64_t decode_us;
  uint64_t direct_bytes;
  uint64_t subview_offset_bytes;
  char mechanism[128];
  char sync_events[512];
  char diagnostic[512];
};

int llm_linear_direct_bridge_probe();
int llm_linear_direct_bridge_last_error(char *buffer, uint64_t capacity);
int llm_linear_direct_bridge_run(const LlmLinearDirectRunConfig *config,
                                 LlmLinearDirectRunResult *result);
}

namespace {

constexpr uint32_t kAbiVersion = 1;
constexpr uint32_t kDirectionGpuPrefillNpuDecode = 0;
constexpr uint32_t kDirectionNpuPrefillGpuDecode = 1;
constexpr uint32_t kImportHipVmem = 2;

thread_local std::string g_last_error;

void set_last_error(std::string message) { g_last_error = std::move(message); }

uint64_t now_us() {
  using clock = std::chrono::steady_clock;
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          clock::now().time_since_epoch())
          .count());
}

void copy_cstr(char *dst, size_t capacity, const std::string &src) {
  if (!dst || capacity == 0)
    return;
  const size_t n = std::min(capacity - 1, src.size());
  std::memcpy(dst, src.data(), n);
  dst[n] = '\0';
}

void check_hip(hipError_t err, const char *what) {
  if (err == hipSuccess)
    return;
  std::ostringstream os;
  os << what << ": " << hipGetErrorName(err) << " (" << hipGetErrorString(err)
     << ")";
  throw std::runtime_error(os.str());
}

struct BridgeMode {
  xrt::bo::flags flag;
  const char *flag_name;
};

const BridgeMode kBridgeModes[] = {
    {xrt::bo::flags::p2p, "p2p"},
    {xrt::bo::flags::device_only, "device_only"},
    {xrt::bo::flags::carveout, "carveout"},
    {xrt::bo::flags::normal, "normal"},
};

uint32_t flag_to_result(xrt::bo::flags flag) {
  return static_cast<uint32_t>(flag);
}

size_t align_up(size_t value, size_t alignment) {
  if (alignment == 0)
    return value;
  return (value + alignment - 1) & ~(alignment - 1);
}

size_t hip_vmem_granularity() {
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = 0;
  prop.requestedHandleType = hipMemHandleTypePosixFileDescriptor;
  size_t granularity = 0;
  check_hip(hipMemGetAllocationGranularity(&granularity, &prop,
                                           hipMemAllocationGranularityMinimum),
            "hipMemGetAllocationGranularity");
  return granularity == 0 ? 4096 : granularity;
}

struct ExternalMapping {
  hipMemGenericAllocationHandle_t handle{};
  void *ptr = nullptr;
  size_t size = 0;

  ExternalMapping() = default;
  ExternalMapping(const ExternalMapping &) = delete;
  ExternalMapping &operator=(const ExternalMapping &) = delete;

  ExternalMapping(ExternalMapping &&other) noexcept
      : handle(other.handle), ptr(other.ptr), size(other.size) {
    other.handle = {};
    other.ptr = nullptr;
    other.size = 0;
  }

  ExternalMapping &operator=(ExternalMapping &&other) noexcept {
    if (this == &other)
      return *this;
    reset();
    handle = other.handle;
    ptr = other.ptr;
    size = other.size;
    other.handle = {};
    other.ptr = nullptr;
    other.size = 0;
    return *this;
  }

  ~ExternalMapping() { reset(); }

  void reset() {
    // On the Ryzen XRT/HIP stack tested for this bridge, tearing down imported
    // XRT BO handles can trip ownership bugs. The benchmark process is the
    // lifetime boundary for now; keep the imported handle alive after a
    // successful import rather than crashing while cleaning up.
    handle = {};
    ptr = nullptr;
    size = 0;
  }
};

ExternalMapping import_xrt_bo_to_hip(xrt::bo &bo, size_t size) {
  int exported_fd = static_cast<int>(bo.export_buffer());
  int import_fd = dup(exported_fd);
  if (import_fd < 0)
    throw std::runtime_error("dup() failed for XRT BO export handle");

  ExternalMapping mapping;
  hipError_t err = hipMemImportFromShareableHandle(
      &mapping.handle,
      reinterpret_cast<void *>(static_cast<intptr_t>(import_fd)),
      hipMemHandleTypePosixFileDescriptor);
  close(import_fd);
  if (err != hipSuccess) {
    check_hip(err, "hipMemImportFromShareableHandle(XRT BO fd)");
  }

  hipDeviceptr_t va = 0;
  size_t granularity = hip_vmem_granularity();
  check_hip(hipMemAddressReserve(&va, size, granularity, 0, 0),
            "hipMemAddressReserve(XRT BO import)");
  check_hip(hipMemMap(va, size, 0, mapping.handle, 0),
            "hipMemMap(XRT BO import)");
  hipMemAccessDesc access_desc{};
  access_desc.location.type = hipMemLocationTypeDevice;
  access_desc.location.id = 0;
  access_desc.flags = hipMemAccessFlagsProtReadWrite;
  check_hip(hipMemSetAccess(va, size, &access_desc, 1),
            "hipMemSetAccess(XRT BO import)");
  mapping.ptr = reinterpret_cast<void *>(va);
  mapping.size = size;
  return mapping;
}

struct SharedBo {
  xrt::bo bo;
  size_t size = 0;
  size_t allocation_size = 0;
  ExternalMapping hip;

  SharedBo() = default;

  SharedBo(const xrt::device &device, size_t size_bytes, xrt::bo::flags flag,
           xrt::memory_group group)
      : bo(device, align_up(size_bytes, hip_vmem_granularity()), flag, group),
        size(size_bytes),
        allocation_size(align_up(size_bytes, hip_vmem_granularity())) {}

  void write_host(const void *src) {
    if (!src || size == 0)
      return;
    bo.write(src, size, 0);
    bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, size, 0);
  }

  void read_host(void *dst) {
    if (!dst || size == 0)
      return;
    bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, size, 0);
    bo.read(dst, size, 0);
  }

  void map_hip() {
    if (!hip.ptr)
      hip = import_xrt_bo_to_hip(bo, allocation_size);
  }

  void *hip_ptr(uint64_t offset = 0) {
    map_hip();
    return static_cast<void *>(static_cast<uint8_t *>(hip.ptr) + offset);
  }
};

template <int Rank>
struct MemRef {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t sizes[Rank];
  int64_t strides[Rank];
};

MemRef<1> memref1(void *ptr, int64_t d0) {
  return MemRef<1>{ptr, ptr, 0, {d0}, {1}};
}

MemRef<2> memref2(void *ptr, int64_t d0, int64_t d1) {
  return MemRef<2>{ptr, ptr, 0, {d0, d1}, {d1, 1}};
}

using PrefillFn = void (*)(MemRef<2> *, MemRef<2> *, MemRef<2> *);
using DecodeFn = void (*)(MemRef<1> *, MemRef<2> *, MemRef<1> *);

class SharedLibrary {
public:
  explicit SharedLibrary(const char *path) {
    if (!path || !path[0])
      throw std::runtime_error("missing GPU direct shared-library path");
    handle_ = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
    if (!handle_) {
      std::ostringstream os;
      os << "dlopen(" << path << ") failed: " << dlerror();
      throw std::runtime_error(os.str());
    }
  }

  SharedLibrary(const SharedLibrary &) = delete;
  SharedLibrary &operator=(const SharedLibrary &) = delete;

  ~SharedLibrary() {
    if (handle_)
      dlclose(handle_);
  }

  template <typename Fn>
  Fn symbol(const char *entry) {
    std::string name = std::string("_mlir_ciface_") + entry;
    dlerror();
    void *sym = dlsym(handle_, name.c_str());
    const char *err = dlerror();
    if (err || !sym) {
      std::ostringstream os;
      os << "dlsym(" << name << ") failed";
      if (err)
        os << ": " << err;
      throw std::runtime_error(os.str());
    }
    return reinterpret_cast<Fn>(sym);
  }

private:
  void *handle_ = nullptr;
};

std::vector<uint32_t> read_instructions(const char *path) {
  if (!path || !path[0])
    throw std::runtime_error("missing NPU instruction path");
  std::ifstream file(path, std::ios::binary);
  if (!file)
    throw std::runtime_error(std::string("failed to open NPU instructions: ") +
                             path);
  std::vector<char> bytes((std::istreambuf_iterator<char>(file)),
                          std::istreambuf_iterator<char>());
  if (bytes.empty() || (bytes.size() % sizeof(uint32_t)) != 0)
    throw std::runtime_error(
        "NPU instruction file is empty or not uint32 aligned");
  std::vector<uint32_t> words(bytes.size() / sizeof(uint32_t));
  std::memcpy(words.data(), bytes.data(), bytes.size());
  return words;
}

class NpuKernel {
public:
  NpuKernel(const char *xclbin_path, const char *insts_path,
            const char *kernel_substr)
      : device_(0), xclbin_(std::string(xclbin_path)),
        uuid_(device_.register_xclbin(xclbin_)), context_(device_, uuid_) {
    std::string needle =
        kernel_substr && kernel_substr[0] ? kernel_substr : "MLIR_AIE";
    std::string kernel_name;
    for (const auto &candidate : xclbin_.get_kernels()) {
      std::string name = candidate.get_name();
      if (name.find(needle) != std::string::npos) {
        kernel_name = name;
        break;
      }
    }
    if (kernel_name.empty())
      throw std::runtime_error("NPU kernel not found in xclbin: " + needle);
    kernel_ = xrt::kernel(context_, kernel_name);
    instructions_ = read_instructions(insts_path);
    bo_instr_ = xrt::bo(device_, instructions_.size() * sizeof(uint32_t),
                        xrt::bo::flags::cacheable, kernel_.group_id(1));
    bo_instr_.write(instructions_.data(),
                    instructions_.size() * sizeof(uint32_t), 0);
  }

  xrt::device &device() { return device_; }
  xrt::memory_group group_id(int argno) const {
    return kernel_.group_id(argno);
  }

  void run3(SharedBo &arg0, SharedBo &arg1, SharedBo &arg2) {
    bo_instr_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    xrt::run run(kernel_);
    int opcode = 3;
    uint32_t instr_count = static_cast<uint32_t>(instructions_.size());
    run.set_arg(0, opcode);
    run.set_arg(1, bo_instr_);
    run.set_arg(2, instr_count);
    run.set_arg(3, arg0.bo);
    run.set_arg(4, arg1.bo);
    run.set_arg(5, arg2.bo);
    run.start();
    auto state = run.wait();
    if (state != ERT_CMD_STATE_COMPLETED) {
      std::ostringstream os;
      os << "NPU run completed with non-completed ERT state " << state;
      throw std::runtime_error(os.str());
    }
  }

private:
  xrt::device device_;
  xrt::xclbin xclbin_;
  xrt::uuid uuid_;
  xrt::hw_context context_;
  xrt::kernel kernel_;
  std::vector<uint32_t> instructions_;
  xrt::bo bo_instr_;
};

void try_bridge_mode_once(const BridgeMode &mode, xrt::memory_group group) {
  xrt::device device(0);
  SharedBo candidate(device, 4096, mode.flag, group);
  candidate.map_hip();
  check_hip(hipMemset(candidate.hip.ptr, 0x5a, 4096),
            "hipMemset(imported XRT BO)");
  check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize(probe)");
}

bool probe_bridge_mode_in_child(const BridgeMode &mode, xrt::memory_group group,
                                std::string &diagnostic) {
  pid_t child = fork();
  if (child < 0) {
    diagnostic = "fork() failed";
    return false;
  }
  if (child == 0) {
    int devnull = open("/dev/null", O_WRONLY);
    if (devnull >= 0) {
      dup2(devnull, STDERR_FILENO);
      close(devnull);
    }
    try {
      check_hip(hipSetDevice(0), "hipSetDevice(0)");
      try_bridge_mode_once(mode, group);
      _exit(0);
    } catch (...) {
      _exit(2);
    }
  }
  int status = 0;
  if (waitpid(child, &status, 0) < 0) {
    diagnostic = "waitpid() failed";
    return false;
  }
  if (WIFEXITED(status) && WEXITSTATUS(status) == 0)
    return true;
  std::ostringstream os;
  if (WIFSIGNALED(status))
    os << "signal " << WTERMSIG(status);
  else
    os << "exit " << (WIFEXITED(status) ? WEXITSTATUS(status) : status);
  diagnostic = os.str();
  return false;
}

BridgeMode select_bridge_mode(xrt::memory_group group) {
  std::string errors;
  for (const BridgeMode &mode : kBridgeModes) {
    std::string diagnostic;
    if (probe_bridge_mode_in_child(mode, group, diagnostic))
      return mode;
    errors += std::string(mode.flag_name) + ": " + diagnostic + "; ";
  }
  throw std::runtime_error("no XRT BO mode could be imported into HIP: " +
                           errors);
}

size_t elem_size(uint32_t dtype) {
  if (dtype <= 1)
    return 2;
  throw std::runtime_error("unsupported dtype enum in direct bridge");
}

void validate_config(const LlmLinearDirectRunConfig &cfg) {
  if (cfg.abi_version != kAbiVersion)
    throw std::runtime_error("unsupported direct bridge ABI version");
  if (cfg.direction != kDirectionGpuPrefillNpuDecode &&
      cfg.direction != kDirectionNpuPrefillGpuDecode)
    throw std::runtime_error("unsupported direct bridge direction");
  if (!cfg.input || !cfg.prefill_weights || !cfg.decode_weights || !cfg.output)
    throw std::runtime_error("direct bridge run received null host buffer");
  if (cfg.m == 0 || cfg.k == 0 || cfg.h == 0 || cfg.n == 0)
    throw std::runtime_error("direct bridge run received zero shape dimension");
  (void)elem_size(cfg.dtype);
}

void run_gpu_prefill(const LlmLinearDirectRunConfig &cfg, SharedBo &input,
                     SharedBo &weights, SharedBo &handoff) {
  SharedLibrary lib(cfg.gpu_prefill_so);
  PrefillFn prefill = lib.symbol<PrefillFn>("llm_linear_prefill");
  auto input_ref = memref2(input.hip_ptr(), static_cast<int64_t>(cfg.m),
                           static_cast<int64_t>(cfg.k));
  auto weight_ref = memref2(weights.hip_ptr(), static_cast<int64_t>(cfg.k),
                            static_cast<int64_t>(cfg.h));
  auto output_ref = memref2(handoff.hip_ptr(), static_cast<int64_t>(cfg.m),
                            static_cast<int64_t>(cfg.h));
  prefill(&input_ref, &weight_ref, &output_ref);
  check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize(gpu prefill)");
}

void run_gpu_decode(const LlmLinearDirectRunConfig &cfg, SharedBo &handoff,
                    SharedBo &weights, SharedBo &output,
                    uint64_t offset_bytes) {
  SharedLibrary lib(cfg.gpu_decode_so);
  DecodeFn decode = lib.symbol<DecodeFn>("llm_linear_decode");
  auto input_ref =
      memref1(handoff.hip_ptr(offset_bytes), static_cast<int64_t>(cfg.h));
  auto weight_ref = memref2(weights.hip_ptr(), static_cast<int64_t>(cfg.h),
                            static_cast<int64_t>(cfg.n));
  auto output_ref = memref1(output.hip_ptr(), static_cast<int64_t>(cfg.n));
  decode(&input_ref, &weight_ref, &output_ref);
  check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize(gpu decode)");
}

void run_gpu_to_npu(const LlmLinearDirectRunConfig &cfg, const BridgeMode &mode,
                    LlmLinearDirectRunResult &result) {
  NpuKernel decode_kernel(cfg.npu_decode_xclbin, cfg.npu_decode_insts,
                          cfg.npu_kernel_name);
  const size_t es = elem_size(cfg.dtype);
  const size_t input_bytes = cfg.m * cfg.k * es;
  const size_t prefill_weight_bytes = cfg.k * cfg.h * es;
  const size_t handoff_bytes = cfg.m * cfg.h * es;
  const size_t decode_weight_bytes = cfg.h * cfg.n * es;
  const size_t output_bytes = cfg.n * es;

  auto &device = decode_kernel.device();
  SharedBo input(device, input_bytes, mode.flag, 0);
  SharedBo prefill_weights(device, prefill_weight_bytes, mode.flag, 0);
  SharedBo handoff(device, handoff_bytes, mode.flag, decode_kernel.group_id(3));
  SharedBo decode_weights(device, decode_weight_bytes, mode.flag,
                          decode_kernel.group_id(4));
  SharedBo output(device, output_bytes, mode.flag, decode_kernel.group_id(5));

  input.write_host(cfg.input);
  prefill_weights.write_host(cfg.prefill_weights);
  decode_weights.write_host(cfg.decode_weights);

  const uint64_t prefill_start = now_us();
  run_gpu_prefill(cfg, input, prefill_weights, handoff);
  result.prefill_us = now_us() - prefill_start;
  result.handoff_us = 0;

  const uint64_t decode_start = now_us();
  decode_kernel.run3(handoff, decode_weights, output);
  result.decode_us = now_us() - decode_start;

  output.read_host(cfg.output);
  if (cfg.prefill_output)
    handoff.read_host(cfg.prefill_output);
  if (cfg.decode_input) {
    const uint64_t offset = (cfg.m - 1) * cfg.h * es;
    handoff.bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, cfg.h * es, offset);
    handoff.bo.read(cfg.decode_input, cfg.h * es, offset);
  }
  result.direct_bytes = handoff_bytes;
  result.subview_offset_bytes = (cfg.m - 1) * cfg.h * es;
  copy_cstr(result.sync_events, sizeof(result.sync_events),
            "hipDeviceSynchronize:generation=gpu_prefill;"
            "xrtRunWait:consumer=npu_decode;"
            "xrtBoSyncFromDevice:final_output");
}

void run_npu_to_gpu(const LlmLinearDirectRunConfig &cfg, const BridgeMode &mode,
                    LlmLinearDirectRunResult &result) {
  NpuKernel prefill_kernel(cfg.npu_prefill_xclbin, cfg.npu_prefill_insts,
                           cfg.npu_kernel_name);
  const size_t es = elem_size(cfg.dtype);
  const size_t input_bytes = cfg.m * cfg.k * es;
  const size_t prefill_weight_bytes = cfg.k * cfg.h * es;
  const size_t handoff_bytes = cfg.m * cfg.h * es;
  const size_t decode_weight_bytes = cfg.h * cfg.n * es;
  const size_t output_bytes = cfg.n * es;

  auto &device = prefill_kernel.device();
  SharedBo input(device, input_bytes, mode.flag, prefill_kernel.group_id(3));
  SharedBo prefill_weights(device, prefill_weight_bytes, mode.flag,
                           prefill_kernel.group_id(4));
  SharedBo handoff(device, handoff_bytes, mode.flag,
                   prefill_kernel.group_id(5));
  SharedBo decode_weights(device, decode_weight_bytes, mode.flag, 0);
  SharedBo output(device, output_bytes, mode.flag, 0);

  input.write_host(cfg.input);
  prefill_weights.write_host(cfg.prefill_weights);
  decode_weights.write_host(cfg.decode_weights);

  const uint64_t prefill_start = now_us();
  prefill_kernel.run3(input, prefill_weights, handoff);
  result.prefill_us = now_us() - prefill_start;
  result.handoff_us = 0;

  const uint64_t offset = (cfg.m - 1) * cfg.h * es;
  const uint64_t decode_start = now_us();
  run_gpu_decode(cfg, handoff, decode_weights, output, offset);
  result.decode_us = now_us() - decode_start;

  output.read_host(cfg.output);
  if (cfg.prefill_output)
    handoff.read_host(cfg.prefill_output);
  if (cfg.decode_input) {
    handoff.bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, cfg.h * es, offset);
    handoff.bo.read(cfg.decode_input, cfg.h * es, offset);
  }
  result.direct_bytes = handoff_bytes;
  result.subview_offset_bytes = offset;
  copy_cstr(result.sync_events, sizeof(result.sync_events),
            "xrtRunWait:producer=npu_prefill;"
            "hipDeviceSynchronize:consumer=gpu_decode;"
            "xrtBoSyncFromDevice:final_output");
}

} // namespace

extern "C" int llm_linear_direct_bridge_probe() {
  try {
    BridgeMode mode = select_bridge_mode(0);
    std::ostringstream ok;
    ok << "direct bridge probe succeeded with XRT BO flag " << mode.flag_name
       << " and HIP VMem fd import";
    set_last_error(ok.str());
    return 0;
  } catch (const std::exception &exc) {
    set_last_error(exc.what());
    return 1;
  }
}

extern "C" int llm_linear_direct_bridge_last_error(char *buffer,
                                                   uint64_t capacity) {
  copy_cstr(buffer, static_cast<size_t>(capacity), g_last_error);
  return 0;
}

extern "C" int
llm_linear_direct_bridge_run(const LlmLinearDirectRunConfig *config,
                             LlmLinearDirectRunResult *result) {
  if (!config || !result) {
    set_last_error("null direct bridge run config/result");
    return 1;
  }
  std::memset(result, 0, sizeof(*result));
  result->abi_version = kAbiVersion;
  try {
    validate_config(*config);
    BridgeMode mode = select_bridge_mode(0);
    check_hip(hipSetDevice(0), "hipSetDevice(0)");
    result->bo_flag = flag_to_result(mode.flag);
    result->import_method = kImportHipVmem;
    copy_cstr(result->mechanism, sizeof(result->mechanism),
              std::string("xrt_bo_export_import_hip_vmem_fd:") +
                  mode.flag_name);

    if (config->direction == kDirectionGpuPrefillNpuDecode)
      run_gpu_to_npu(*config, mode, *result);
    else
      run_npu_to_gpu(*config, mode, *result);

    copy_cstr(result->diagnostic, sizeof(result->diagnostic), "ok");
    set_last_error("ok");
    return 0;
  } catch (const std::exception &exc) {
    set_last_error(exc.what());
    copy_cstr(result->diagnostic, sizeof(result->diagnostic), exc.what());
    return 1;
  }
}
