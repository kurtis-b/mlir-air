// SPDX-License-Identifier: MIT
//
// Native bridge for the llm_linear direct GPU/NPU handoff path.
//
// The bridge deliberately makes HIP VMem the owning allocation for tensors that
// cross the GPU/NPU boundary.  HIP exports a POSIX fd for each allocation and
// XRT imports that fd as an xrt::bo view.  This matches the viable probe result
// on the Ryzen AI stack and keeps the HIP handle, fd, VA, and XRT BO alive in a
// process-lifetime pool.

#ifndef __AMDGCN_WAVEFRONT_SIZE
#define __AMDGCN_WAVEFRONT_SIZE 32
#endif
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <xrt/experimental/xrt_hw_context.h>
#include <xrt/experimental/xrt_xclbin.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <exception>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <unistd.h>

extern "C" {

struct LlmLinearDirectRunConfig {
  uint32_t abi_version;
  uint32_t direction;
  uint32_t dtype;
  uint32_t decode_storage;
  uint32_t decode_block_size;
  uint32_t decode_quant_axis;
  uint32_t decode_tile_h;
  uint32_t decode_tile_n;
  uint64_t m;
  uint64_t k;
  uint64_t h;
  uint64_t n;
  uint64_t decode_packed_bytes;
  uint64_t decode_scale_bytes;
  const void *input;
  const void *prefill_weights;
  const void *decode_weights;
  const void *decode_packed_weights;
  const void *decode_scales;
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

struct LlmLinearGpuStageRunConfig {
  uint32_t abi_version;
  uint32_t stage;
  uint32_t dtype;
  uint32_t decode_storage;
  uint32_t decode_block_size;
  uint32_t decode_quant_axis;
  uint32_t reserved;
  uint32_t reserved1;
  uint64_t m;
  uint64_t k;
  uint64_t h;
  uint64_t n;
  uint64_t decode_packed_bytes;
  uint64_t decode_scale_bytes;
  const void *input;
  const void *weights;
  const void *scales;
  void *output;
  const char *gpu_so;
};

struct LlmLinearGpuStageRunResult {
  uint32_t abi_version;
  uint32_t reserved0;
  uint32_t reserved1;
  uint32_t reserved2;
  uint64_t stage_us;
  char diagnostic[512];
};

struct LlmLinearResidentPipelinePrepareConfig {
  uint32_t abi_version;
  uint32_t backend;
  uint32_t dtype;
  uint32_t decode_storage;
  uint32_t decode_block_size;
  uint32_t decode_quant_axis;
  uint32_t gpu_decode_tile_h;
  uint32_t gpu_decode_tile_n;
  uint32_t npu_decode_tile_h;
  uint32_t npu_decode_tile_n;
  uint64_t m;
  uint64_t k;
  uint64_t h;
  uint64_t n;
  uint64_t decode_packed_bytes;
  uint64_t decode_scale_bytes;
  const void *prefill_weights;
  const void *decode_weights;
  const void *decode_packed_weights;
  const void *decode_scales;
  const char *gpu_prefill_so;
  const char *gpu_decode_so;
  const char *npu_prefill_xclbin;
  const char *npu_prefill_insts;
  const char *npu_decode_xclbin;
  const char *npu_decode_insts;
  const char *npu_kernel_name;
};

struct LlmLinearResidentPipelinePrepareResult {
  uint32_t abi_version;
  uint32_t reserved0;
  uint32_t reserved1;
  uint32_t reserved2;
  uint64_t handle;
  uint64_t static_upload_bytes;
  char storage_kind[64];
  char diagnostic[512];
};

struct LlmLinearResidentPipelineRunConfig {
  uint32_t abi_version;
  uint32_t capture_details;
  uint32_t reserved0;
  uint32_t reserved1;
  uint64_t handle;
  const void *input;
  void *output;
  void *prefill_output;
  void *decode_input;
};

struct LlmLinearResidentPipelineRunResult {
  uint32_t abi_version;
  uint32_t reserved0;
  uint32_t reserved1;
  uint32_t reserved2;
  uint64_t prefill_us;
  uint64_t decode_us;
  uint64_t input_upload_us;
  uint64_t output_readback_us;
  uint64_t launch_count;
  uint64_t timed_allocation_count;
  char diagnostic[512];
};

int llm_linear_direct_bridge_probe();
int llm_linear_direct_bridge_probe_report(char *buffer, uint64_t capacity);
int llm_linear_direct_bridge_last_error(char *buffer, uint64_t capacity);
int llm_linear_direct_bridge_cleanup();
int llm_linear_direct_bridge_run(const LlmLinearDirectRunConfig *config,
                                 LlmLinearDirectRunResult *result);
int llm_linear_direct_bridge_run_gpu_stage(
    const LlmLinearGpuStageRunConfig *config,
    LlmLinearGpuStageRunResult *result);
int llm_linear_direct_bridge_prepare_resident_pipeline(
    const LlmLinearResidentPipelinePrepareConfig *config,
    LlmLinearResidentPipelinePrepareResult *result);
int llm_linear_direct_bridge_run_resident_pipeline(
    const LlmLinearResidentPipelineRunConfig *config,
    LlmLinearResidentPipelineRunResult *result);
int llm_linear_direct_bridge_release_resident_pipeline(uint64_t handle);
}

namespace {

constexpr uint32_t kAbiVersion = 2;
constexpr uint32_t kDirectionGpuPrefillNpuDecode = 0;
constexpr uint32_t kDirectionNpuPrefillGpuDecode = 1;
constexpr uint32_t kGpuStagePrefill = 0;
constexpr uint32_t kGpuStageDecode = 1;
constexpr uint32_t kResidentBackendGpu = 0;
constexpr uint32_t kResidentBackendNpu = 1;
constexpr uint32_t kDecodeStorageDense = 0;
constexpr uint32_t kDecodeStorageInt4 = 1;
constexpr uint32_t kImportXrtBoFromHipVmemFd = 3;
constexpr const char *kContractNoHostCopies = "no_host_copies";
constexpr const char *kHipVmemMechanism = "hip_vmem_export_xrt_bo_import_fd";
constexpr const char *kDeviceResidentDirectClass =
    "device_resident_zero_host_copy";
constexpr const char *kSharedHostDirectClass = "shared_host_zero_copy";
constexpr const char *kHostStagedDirectClass = "host_staged_copy";
constexpr const char *kUnsupportedDirectClass = "unsupported";

thread_local std::string g_last_error;

void set_last_error(std::string message) { g_last_error = std::move(message); }

void trace_step(const char *step) {
  const char *enabled = std::getenv("LLM_LINEAR_DIRECT_BRIDGE_TRACE");
  if (!enabled || !enabled[0] || enabled[0] == '0')
    return;
  std::fprintf(stderr, "[direct_bridge] %s\n", step);
  std::fflush(stderr);
}

class ScopedUnsetEnv {
public:
  explicit ScopedUnsetEnv(const char *name) : name_(name ? name : "") {
    if (name_.empty())
      return;
    const char *value = std::getenv(name_.c_str());
    if (!value)
      return;
    had_value_ = true;
    value_ = value;
    unsetenv(name_.c_str());
  }

  ~ScopedUnsetEnv() {
    if (name_.empty())
      return;
    if (had_value_)
      setenv(name_.c_str(), value_.c_str(), 1);
  }

  ScopedUnsetEnv(const ScopedUnsetEnv &) = delete;
  ScopedUnsetEnv &operator=(const ScopedUnsetEnv &) = delete;

private:
  std::string name_;
  std::string value_;
  bool had_value_ = false;
};

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

std::string json_escape(const std::string &src) {
  std::string out;
  out.reserve(src.size() + 8);
  for (unsigned char ch : src) {
    switch (ch) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\b':
      out += "\\b";
      break;
    case '\f':
      out += "\\f";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      if (ch < 0x20) {
        char buffer[8];
        std::snprintf(buffer, sizeof(buffer), "\\u%04x", ch);
        out += buffer;
      } else {
        out.push_back(static_cast<char>(ch));
      }
      break;
    }
  }
  return out;
}

std::string json_string(const std::string &src) {
  return "\"" + json_escape(src) + "\"";
}

const char *json_bool(bool value) { return value ? "true" : "false"; }

void check_hip(hipError_t err, const char *what) {
  if (err == hipSuccess)
    return;
  std::ostringstream os;
  os << what << ": " << hipGetErrorName(err) << " (" << hipGetErrorString(err)
     << ")";
  throw std::runtime_error(os.str());
}

size_t align_up(size_t value, size_t alignment) {
  if (alignment == 0)
    return value;
  return (value + alignment - 1) & ~(alignment - 1);
}

void verify_uniform_bytes(const void *ptr, uint8_t value, size_t count) {
  const auto *bytes = static_cast<const uint8_t *>(ptr);
  for (size_t i = 0; i < count; ++i) {
    if (bytes[i] != value) {
      std::ostringstream os;
      os << "probe byte mismatch at " << i << ": got "
         << static_cast<int>(bytes[i]) << ", expected "
         << static_cast<int>(value);
      throw std::runtime_error(os.str());
    }
  }
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

struct HipVmemXrtBuffer {
  hipDeviceptr_t va = 0;
  hipMemGenericAllocationHandle_t handle{};
  int fd = -1;
  size_t size = 0;
  size_t allocation_size = 0;
  std::unique_ptr<xrt::bo> xrt_bo;
  bool in_use = false;

  HipVmemXrtBuffer(const xrt::device &device, size_t size_bytes)
      : size(size_bytes),
        allocation_size(
            align_up(std::max<size_t>(size_bytes, 1), hip_vmem_granularity())) {
    hipMemAllocationProp prop{};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = 0;
    prop.requestedHandleType = hipMemHandleTypePosixFileDescriptor;
    check_hip(hipMemCreate(&handle, allocation_size, &prop, 0),
              "hipMemCreate(HIP-owned handoff)");
    check_hip(hipMemAddressReserve(&va, allocation_size, hip_vmem_granularity(),
                                   0, 0),
              "hipMemAddressReserve(HIP-owned handoff)");
    check_hip(hipMemMap(va, allocation_size, 0, handle, 0),
              "hipMemMap(HIP-owned handoff)");
    hipMemAccessDesc access{};
    access.location.type = hipMemLocationTypeDevice;
    access.location.id = 0;
    access.flags = hipMemAccessFlagsProtReadWrite;
    check_hip(hipMemSetAccess(va, allocation_size, &access, 1),
              "hipMemSetAccess(HIP-owned handoff)");
    check_hip(hipMemExportToShareableHandle(
                  &fd, handle, hipMemHandleTypePosixFileDescriptor, 0),
              "hipMemExportToShareableHandle(HIP-owned handoff)");
    xrt_bo = std::make_unique<xrt::bo>(device,
                                       static_cast<xrt::bo::export_handle>(fd));
  }

  HipVmemXrtBuffer(const HipVmemXrtBuffer &) = delete;
  HipVmemXrtBuffer &operator=(const HipVmemXrtBuffer &) = delete;

  ~HipVmemXrtBuffer() {
    xrt_bo.reset();
    if (fd >= 0) {
      (void)close(fd);
      fd = -1;
    }
    if (va) {
      (void)hipDeviceSynchronize();
      (void)hipMemUnmap(va, allocation_size);
      (void)hipMemAddressFree(va, allocation_size);
      va = 0;
    }
    if (handle) {
      (void)hipMemRelease(handle);
      handle = {};
    }
  }

  void *hip_ptr(uint64_t offset = 0) {
    return static_cast<void *>(static_cast<uint8_t *>(va) + offset);
  }

  xrt::bo &bo() { return *xrt_bo; }

  void hip_write_host(const void *src) {
    if (!src || size == 0)
      return;
    check_hip(hipMemcpy(hip_ptr(), src, size, hipMemcpyHostToDevice),
              "hipMemcpy H2D(HIP-owned BO)");
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize(H2D)");
  }

  void hip_read_host(void *dst) {
    if (!dst || size == 0)
      return;
    check_hip(hipMemcpy(dst, hip_ptr(), size, hipMemcpyDeviceToHost),
              "hipMemcpy D2H(HIP-owned BO)");
  }

  void hip_read_host(void *dst, size_t bytes, size_t offset) {
    if (!dst || bytes == 0)
      return;
    check_hip(hipMemcpy(dst, hip_ptr(offset), bytes, hipMemcpyDeviceToHost),
              "hipMemcpy D2H(HIP-owned BO subrange)");
  }

  void xrt_write_host(const void *src) {
    if (!src || size == 0)
      return;
    bo().write(src, size, 0);
    bo().sync(XCL_BO_SYNC_BO_TO_DEVICE, size, 0);
  }

  void xrt_read_host(void *dst) {
    if (!dst || size == 0)
      return;
    bo().sync(XCL_BO_SYNC_BO_FROM_DEVICE, size, 0);
    bo().read(dst, size, 0);
  }

  void xrt_read_host(void *dst, size_t bytes, size_t offset) {
    if (!dst || bytes == 0)
      return;
    bo().sync(XCL_BO_SYNC_BO_FROM_DEVICE, bytes, offset);
    bo().read(dst, bytes, offset);
  }

  void sync_to_xrt(size_t bytes, size_t offset = 0) {
    if (bytes == 0)
      return;
    bo().sync(XCL_BO_SYNC_BO_TO_DEVICE, bytes, offset);
  }

  void sync_from_xrt(size_t bytes, size_t offset = 0) {
    if (bytes == 0)
      return;
    bo().sync(XCL_BO_SYNC_BO_FROM_DEVICE, bytes, offset);
  }
};

struct HipDeviceBuffer {
  void *ptr = nullptr;
  size_t size = 0;

  explicit HipDeviceBuffer(size_t size_bytes) : size(size_bytes) {
    check_hip(hipMalloc(&ptr, std::max<size_t>(size, 1)),
              "hipMalloc(native GPU stage)");
  }

  ~HipDeviceBuffer() {
    if (ptr)
      (void)hipFree(ptr);
  }

  HipDeviceBuffer(const HipDeviceBuffer &) = delete;
  HipDeviceBuffer &operator=(const HipDeviceBuffer &) = delete;

  void *hip_ptr(uint64_t offset = 0) {
    return static_cast<void *>(static_cast<uint8_t *>(ptr) + offset);
  }

  void hip_write_host(const void *src) {
    if (!src || size == 0)
      return;
    check_hip(hipMemcpy(hip_ptr(), src, size, hipMemcpyHostToDevice),
              "hipMemcpy H2D(native GPU stage)");
  }

  void hip_read_host(void *dst) {
    if (!dst || size == 0)
      return;
    check_hip(hipMemcpy(dst, hip_ptr(), size, hipMemcpyDeviceToHost),
              "hipMemcpy D2H(native GPU stage)");
  }

  void hip_read_host(void *dst, size_t bytes, size_t offset) {
    if (!dst || bytes == 0)
      return;
    check_hip(hipMemcpy(dst, hip_ptr(offset), bytes, hipMemcpyDeviceToHost),
              "hipMemcpy D2H(native GPU stage subrange)");
  }
};

std::mutex &pool_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::vector<HipVmemXrtBuffer *> &hip_vmem_pool() {
  static auto *pool = new std::vector<HipVmemXrtBuffer *>();
  return *pool;
}

struct PooledHipVmemBo {
  HipVmemXrtBuffer *buffer = nullptr;

  PooledHipVmemBo() = default;
  explicit PooledHipVmemBo(HipVmemXrtBuffer *value) : buffer(value) {}
  PooledHipVmemBo(const PooledHipVmemBo &) = delete;
  PooledHipVmemBo &operator=(const PooledHipVmemBo &) = delete;

  PooledHipVmemBo(PooledHipVmemBo &&other) noexcept : buffer(other.buffer) {
    other.buffer = nullptr;
  }

  PooledHipVmemBo &operator=(PooledHipVmemBo &&other) noexcept {
    if (this == &other)
      return *this;
    release();
    buffer = other.buffer;
    other.buffer = nullptr;
    return *this;
  }

  ~PooledHipVmemBo() { release(); }

  void release() {
    if (!buffer)
      return;
    std::lock_guard<std::mutex> lock(pool_mutex());
    buffer->in_use = false;
    buffer = nullptr;
  }

  HipVmemXrtBuffer *operator->() { return buffer; }
  HipVmemXrtBuffer &operator*() { return *buffer; }
};

PooledHipVmemBo acquire_hip_vmem_xrt_bo(const xrt::device &device,
                                        size_t size) {
  std::lock_guard<std::mutex> lock(pool_mutex());
  for (HipVmemXrtBuffer *buffer : hip_vmem_pool()) {
    if (!buffer->in_use && buffer->size == size) {
      buffer->in_use = true;
      return PooledHipVmemBo(buffer);
    }
  }
  auto *buffer = new HipVmemXrtBuffer(device, size);
  buffer->in_use = true;
  hip_vmem_pool().push_back(buffer);
  return PooledHipVmemBo(buffer);
}

void cleanup_hip_vmem_pool() {
  std::lock_guard<std::mutex> lock(pool_mutex());
  auto &pool = hip_vmem_pool();
  for (HipVmemXrtBuffer *buffer : pool) {
    if (buffer && buffer->in_use)
      throw std::runtime_error(
          "cannot cleanup direct bridge pool during a run");
  }
  for (HipVmemXrtBuffer *buffer : pool)
    delete buffer;
  pool.clear();
  (void)hipDeviceSynchronize();
  (void)hipDeviceReset();
}

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

MemRef<2> memref2_strided(void *ptr, int64_t d0, int64_t d1, int64_t s0,
                          int64_t s1) {
  return MemRef<2>{ptr, ptr, 0, {d0, d1}, {s0, s1}};
}

using PrefillFn = void (*)(MemRef<2> *, MemRef<2> *, MemRef<2> *);
using DecodeFn = void (*)(MemRef<1> *, MemRef<2> *, MemRef<1> *);
using DecodeInt4Fn = void (*)(MemRef<1> *, MemRef<2> *, MemRef<2> *,
                              MemRef<1> *);

class SharedLibrary {
public:
  explicit SharedLibrary(const char *path) {
    if (!path || !path[0])
      throw std::runtime_error("missing GPU direct shared-library path");
    int flags = RTLD_NOW | RTLD_LOCAL;
#ifdef RTLD_DEEPBIND
    flags |= RTLD_DEEPBIND;
#endif
    handle_ = dlopen(path, flags);
    if (!handle_) {
      std::ostringstream os;
      os << "dlopen(" << path << ") failed: " << dlerror();
      throw std::runtime_error(os.str());
    }
  }

  SharedLibrary(const SharedLibrary &) = delete;
  SharedLibrary &operator=(const SharedLibrary &) = delete;

  // Keep HIP code objects registered for the process lifetime; unloading after
  // a direct decode call can race normal-process teardown on this ROCm stack.
  ~SharedLibrary() = default;

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
            const char *kernel_substr,
            std::shared_ptr<xrt::device> shared_device = nullptr)
      : device_(shared_device ? std::move(shared_device)
                              : std::make_shared<xrt::device>(0)),
        xclbin_(std::string(xclbin_path)),
        uuid_(device_->register_xclbin(xclbin_)), context_(*device_, uuid_) {
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
    bo_instr_ = xrt::bo(*device_, instructions_.size() * sizeof(uint32_t),
                        xrt::bo::flags::cacheable, kernel_.group_id(1));
    bo_instr_.write(instructions_.data(),
                    instructions_.size() * sizeof(uint32_t), 0);
  }

  xrt::device &device() { return *device_; }
  void run3(xrt::bo &arg0, xrt::bo &arg1, xrt::bo &arg2) {
    bo_instr_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    unsigned int opcode = 3;
    unsigned int instr_count = static_cast<unsigned int>(instructions_.size());
    auto run = kernel_(opcode, bo_instr_, instr_count, arg0, arg1, arg2);
    run.wait();
  }

  void run4(xrt::bo &arg0, xrt::bo &arg1, xrt::bo &arg2, xrt::bo &arg3) {
    bo_instr_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    unsigned int opcode = 3;
    unsigned int instr_count = static_cast<unsigned int>(instructions_.size());
    auto run = kernel_(opcode, bo_instr_, instr_count, arg0, arg1, arg2, arg3);
    run.wait();
  }

  void run3(PooledHipVmemBo &arg0, PooledHipVmemBo &arg1,
            PooledHipVmemBo &arg2) {
    run3(arg0->bo(), arg1->bo(), arg2->bo());
  }

  void run4(PooledHipVmemBo &arg0, PooledHipVmemBo &arg1, PooledHipVmemBo &arg2,
            PooledHipVmemBo &arg3) {
    run4(arg0->bo(), arg1->bo(), arg2->bo(), arg3->bo());
  }

private:
  std::shared_ptr<xrt::device> device_;
  xrt::xclbin xclbin_;
  xrt::uuid uuid_;
  xrt::hw_context context_;
  xrt::kernel kernel_;
  std::vector<uint32_t> instructions_;
  xrt::bo bo_instr_;
};

struct MechanismProbeReport {
  std::string mechanism;
  bool supported = false;
  bool direct_eligible = false;
  std::string direct_class = kUnsupportedDirectClass;
  std::string ownership;
  std::string handle_type;
  std::string import_view;
  bool bidirectional_visibility = false;
  bool npu_kernel_verification = false;
  std::vector<std::string> sync_events;
  int host_materialization_count = 0;
  bool zero_host_copy = false;
  bool device_resident_buffers = false;
  std::string diagnostic;
};

std::string sync_events_json(const std::vector<std::string> &events) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < events.size(); ++i) {
    if (i)
      os << ",";
    os << "{\"event\":" << json_string(events[i]) << "}";
  }
  os << "]";
  return os.str();
}

std::string mechanism_report_json(const MechanismProbeReport &report) {
  std::ostringstream os;
  os << "{" << "\"mechanism\":" << json_string(report.mechanism) << ","
     << "\"supported\":" << json_bool(report.supported) << ","
     << "\"direct_eligible\":" << json_bool(report.direct_eligible) << ","
     << "\"direct_class\":" << json_string(report.direct_class) << ","
     << "\"ownership\":" << json_string(report.ownership) << ","
     << "\"handle_type\":" << json_string(report.handle_type) << ","
     << "\"import_view\":" << json_string(report.import_view) << ","
     << "\"bidirectional_visibility\":"
     << json_bool(report.bidirectional_visibility) << ","
     << "\"npu_kernel_verification\":"
     << json_bool(report.npu_kernel_verification) << ","
     << "\"sync_events\":" << sync_events_json(report.sync_events) << ","
     << "\"host_materialization_count\":" << report.host_materialization_count
     << "," << "\"zero_host_copy\":" << json_bool(report.zero_host_copy) << ","
     << "\"device_resident_buffers\":"
     << json_bool(report.device_resident_buffers) << ","
     << "\"diagnostic\":" << json_string(report.diagnostic) << "}";
  return os.str();
}

std::string probe_report_json(const std::vector<MechanismProbeReport> &reports,
                              const std::string &selected_mechanism,
                              const std::string &diagnostic) {
  const bool direct_supported = !selected_mechanism.empty();
  std::ostringstream os;
  os << "{" << "\"schema_version\":1,"
     << "\"contract\":" << json_string(kContractNoHostCopies) << ","
     << "\"direct_supported\":" << json_bool(direct_supported) << ","
     << "\"selected_mechanism\":";
  if (direct_supported)
    os << json_string(selected_mechanism);
  else
    os << "null";
  os << ",\"diagnostic\":" << json_string(diagnostic) << ","
     << "\"mechanisms\":[";
  for (size_t i = 0; i < reports.size(); ++i) {
    if (i)
      os << ",";
    os << mechanism_report_json(reports[i]);
  }
  os << "]}";
  return os.str();
}

void probe_hip_owned_import_once() {
  ScopedUnsetEnv scoped_pythonpath("PYTHONPATH");
  ScopedUnsetEnv scoped_ld_library_path("LD_LIBRARY_PATH");
  trace_step("probe: before hipSetDevice");
  check_hip(hipSetDevice(0), "hipSetDevice(0)");
  trace_step("probe: before xrt device");
  xrt::device device(0);
  trace_step("probe: before acquire HIP VMem BO");
  auto candidate = acquire_hip_vmem_xrt_bo(device, 4096);
  trace_step("probe: before hipMemset");
  check_hip(hipMemset(candidate->hip_ptr(), 0x5a, 4096),
            "hipMemset(HIP-owned handoff)");
  trace_step("probe: before hipDeviceSynchronize");
  check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize(probe)");
  std::vector<uint8_t> host(4096, 0);
  candidate->sync_from_xrt(4096);
  candidate->bo().read(host.data(), host.size(), 0);
  verify_uniform_bytes(host.data(), 0x5a, host.size());
  std::fill(host.begin(), host.end(), 0xa5);
  candidate->xrt_write_host(host.data());
  std::fill(host.begin(), host.end(), 0);
  candidate->hip_read_host(host.data());
  verify_uniform_bytes(host.data(), 0xa5, host.size());
  trace_step("probe: done");
}

std::string build_direct_probe_report_json() {
  std::vector<MechanismProbeReport> reports;

  MechanismProbeReport hip_vmem;
  hip_vmem.mechanism = kHipVmemMechanism;
  hip_vmem.direct_class = kDeviceResidentDirectClass;
  hip_vmem.ownership = "hip_vmem";
  hip_vmem.handle_type = "posix_fd";
  hip_vmem.import_view = "xrt_bo";
  hip_vmem.sync_events = {"hipDeviceSynchronize", "xrtBoSyncFromDevice",
                          "xrtBoSyncToDevice", "hipMemcpyDtoH"};
  try {
    probe_hip_owned_import_once();
    hip_vmem.supported = true;
    hip_vmem.direct_eligible = true;
    hip_vmem.bidirectional_visibility = true;
    hip_vmem.zero_host_copy = true;
    hip_vmem.device_resident_buffers = true;
    hip_vmem.diagnostic =
        "HIP VMem POSIX fd export imported into XRT BO with bidirectional "
        "visibility; workload NPU kernel verification is recorded by the "
        "direct run";
  } catch (const std::exception &exc) {
    hip_vmem.diagnostic = exc.what();
  }
  reports.push_back(hip_vmem);

  reports.push_back(MechanismProbeReport{
      "xrt_bo_export_hip_vmem_import_fd",
      false,
      false,
      kDeviceResidentDirectClass,
      "xrt_bo",
      "posix_fd",
      "hip_vmem",
      false,
      false,
      {},
      0,
      true,
      false,
      "Not selected by the runtime bridge; transfer_probe covers per-flag "
      "experiments and this direction failed, timed out, or signaled on the "
      "audited stack."});

  reports.push_back(MechanismProbeReport{
      "xrt_bo_export_hip_external_memory_import_fd",
      false,
      false,
      kDeviceResidentDirectClass,
      "xrt_bo",
      "posix_fd",
      "hip_external_memory",
      false,
      false,
      {},
      0,
      true,
      false,
      "Not selected by the runtime bridge; HIP external-memory import did not "
      "produce a stable direct mapping for XRT-owned BOs on the audited "
      "stack."});

  reports.push_back(MechanismProbeReport{
      "xrt_host_userptr_hip_registered_shared_host",
      false,
      false,
      kSharedHostDirectClass,
      "host_userptr",
      "host_pointer",
      "hip_registered_host_pointer",
      false,
      false,
      {},
      0,
      true,
      false,
      "Shared host mappings are tracked as a zero-copy research class, but are "
      "not accepted as the current device-resident direct bridge."});

  reports.push_back(MechanismProbeReport{
      "numpy_host_staged_baseline",
      true,
      false,
      kHostStagedDirectClass,
      "host_numpy",
      "host_pointer",
      "host_array",
      true,
      false,
      {},
      1,
      false,
      false,
      "Comparison baseline only; transfer_mode=direct must never select this "
      "mechanism."});

  std::string selected;
  for (const auto &report : reports) {
    if (report.supported && report.direct_eligible && report.zero_host_copy &&
        report.host_materialization_count == 0) {
      selected = report.mechanism;
      break;
    }
  }

  std::string diagnostic =
      selected.empty()
          ? "no audited zero-host-copy GPU/NPU bridge mechanism was validated"
          : "direct bridge probe selected " + selected;
  return probe_report_json(reports, selected, diagnostic);
}

size_t elem_size(uint32_t dtype) {
  if (dtype <= 1)
    return 2;
  throw std::runtime_error("unsupported dtype enum in direct bridge");
}

float bf16_to_float(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

uint16_t float_to_bf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffU + ((bits >> 16) & 1U);
  return static_cast<uint16_t>(bits >> 16);
}

__device__ float device_bf16_to_float(uint16_t value) {
  union {
    uint32_t u;
    float f;
  } bits;
  bits.u = static_cast<uint32_t>(value) << 16;
  return bits.f;
}

__device__ uint16_t device_float_to_bf16(float value) {
  union {
    uint32_t u;
    float f;
  } bits;
  bits.f = value;
  bits.u += 0x7fffU + ((bits.u >> 16) & 1U);
  return static_cast<uint16_t>(bits.u >> 16);
}

__global__ void accumulate_bf16_to_f32_kernel(const uint16_t *src, float *dst,
                                              uint64_t count) {
  uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count)
    dst[i] += device_bf16_to_float(src[i]);
}

__global__ void f32_to_bf16_kernel(const float *src, uint16_t *dst,
                                   uint64_t count) {
  uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count)
    dst[i] = device_float_to_bf16(src[i]);
}

void launch_accumulate_bf16_to_f32(void *src, void *dst, uint64_t count) {
  if (count == 0)
    return;
  const uint32_t threads = 256;
  const uint32_t blocks =
      static_cast<uint32_t>((count + threads - 1) / threads);
  hipLaunchKernelGGL(accumulate_bf16_to_f32_kernel, dim3(blocks), dim3(threads),
                     0, 0, static_cast<const uint16_t *>(src),
                     static_cast<float *>(dst), count);
  check_hip(hipGetLastError(), "launch accumulate_bf16_to_f32_kernel");
}

void launch_f32_to_bf16(void *src, void *dst, uint64_t count) {
  if (count == 0)
    return;
  const uint32_t threads = 256;
  const uint32_t blocks =
      static_cast<uint32_t>((count + threads - 1) / threads);
  hipLaunchKernelGGL(f32_to_bf16_kernel, dim3(blocks), dim3(threads), 0, 0,
                     static_cast<const float *>(src),
                     static_cast<uint16_t *>(dst), count);
  check_hip(hipGetLastError(), "launch f32_to_bf16_kernel");
}

bool decode_is_int4(uint32_t storage) {
  if (storage == kDecodeStorageDense)
    return false;
  if (storage == kDecodeStorageInt4)
    return true;
  throw std::runtime_error("unsupported decode storage enum in direct bridge");
}

struct DecodeLayout {
  bool int4 = false;
  uint64_t h = 0;
  uint64_t n = 0;
  size_t element_bytes = 0;
  uint32_t block_size = 0;
  uint32_t quant_axis = 0;
  uint64_t block_count = 0;
  uint64_t packed_word_cols = 0;
  uint64_t tile_h = 0;
  uint64_t tile_n = 0;

  size_t dense_weight_bytes() const { return h * n * element_bytes; }
  size_t dense_tile_bytes() const { return tile_h * tile_n * element_bytes; }
  size_t packed_bytes() const { return h * (n / 2); }
  size_t scale_bytes() const { return block_count * n * sizeof(float); }
  size_t packed_tile_bytes() const { return h * (tile_n / 2); }
  size_t scale_tile_bytes() const {
    return block_count * tile_n * sizeof(float);
  }
  bool tiled_dense() const { return !int4 && (tile_h < h || tile_n < n); }
  bool tiled_int4() const { return int4 && tile_n < n; }
  bool tiled_decode() const { return tile_h < h || tile_n < n; }
};

DecodeLayout make_decode_layout(uint64_t h, uint64_t n, uint32_t dtype,
                                uint32_t storage, uint32_t block_size,
                                uint32_t quant_axis,
                                uint32_t requested_tile_h = 0,
                                uint32_t requested_tile_n = 0) {
  DecodeLayout layout;
  layout.int4 = decode_is_int4(storage);
  layout.h = h;
  layout.n = n;
  layout.element_bytes = elem_size(dtype);
  layout.tile_h =
      std::min<uint64_t>(h, requested_tile_h ? requested_tile_h : h);
  layout.tile_n =
      std::min<uint64_t>(n, requested_tile_n ? requested_tile_n : n);
  if (layout.tile_h == 0 || (h % layout.tile_h) != 0)
    throw std::runtime_error("direct NPU decode requires H % tile_h == 0");
  if (layout.tile_n == 0 || (n % layout.tile_n) != 0)
    throw std::runtime_error("direct NPU decode requires N % tile_n == 0");
  if (!layout.int4)
    return layout;

  if (block_size == 0)
    throw std::runtime_error("int4 decode block_size must be positive");
  if (quant_axis != 0)
    throw std::runtime_error("int4 decode hardware requires quant_axis == 0");
  if ((h % block_size) != 0)
    throw std::runtime_error(
        "int4 decode hardware requires H % block_size == 0");
  if ((n % 8) != 0)
    throw std::runtime_error("int4 decode hardware requires N divisible by 8");
  layout.block_size = block_size;
  layout.quant_axis = quant_axis;
  layout.block_count = h / block_size;
  layout.packed_word_cols = n / 8;
  if ((layout.tile_n % 8) != 0)
    throw std::runtime_error("int4 direct NPU decode requires N % tile_n == 0");
  return layout;
}

void stage_decode_dense_weight_tile(const LlmLinearDirectRunConfig &cfg,
                                    const DecodeLayout &layout,
                                    uint64_t start_h, uint64_t start_n,
                                    std::vector<uint8_t> &dst) {
  const auto *src = static_cast<const uint8_t *>(cfg.decode_weights);
  for (uint64_t row = 0; row < layout.tile_h; ++row) {
    const uint8_t *row_src =
        src + ((start_h + row) * cfg.n + start_n) * layout.element_bytes;
    uint8_t *row_dst = dst.data() + row * layout.tile_n * layout.element_bytes;
    std::memcpy(row_dst, row_src, layout.tile_n * layout.element_bytes);
  }
}

void stage_decode_int4_packed_tile(const LlmLinearDirectRunConfig &cfg,
                                   const DecodeLayout &layout, uint64_t start_n,
                                   std::vector<uint8_t> &dst) {
  const uint64_t tile_words = layout.tile_n / 8;
  const uint64_t word_start = start_n / 8;
  const auto *src = static_cast<const uint32_t *>(cfg.decode_packed_weights);
  auto *out = reinterpret_cast<uint32_t *>(dst.data());
  for (uint64_t row = 0; row < cfg.h; ++row) {
    std::memcpy(out + row * tile_words,
                src + row * layout.packed_word_cols + word_start,
                tile_words * sizeof(uint32_t));
  }
}

void stage_decode_int4_scale_tile(const LlmLinearDirectRunConfig &cfg,
                                  const DecodeLayout &layout, uint64_t start_n,
                                  std::vector<uint8_t> &dst) {
  const auto *src = static_cast<const float *>(cfg.decode_scales);
  auto *out = reinterpret_cast<float *>(dst.data());
  for (uint64_t block = 0; block < layout.block_count; ++block) {
    std::memcpy(out + block * layout.tile_n, src + block * cfg.n + start_n,
                layout.tile_n * sizeof(float));
  }
}

void validate_config(const LlmLinearDirectRunConfig &cfg) {
  if (cfg.abi_version != kAbiVersion)
    throw std::runtime_error("unsupported direct bridge ABI version");
  if (cfg.direction != kDirectionGpuPrefillNpuDecode &&
      cfg.direction != kDirectionNpuPrefillGpuDecode)
    throw std::runtime_error("unsupported direct bridge direction");
  if (!cfg.input || !cfg.prefill_weights || !cfg.output)
    throw std::runtime_error("direct bridge run received null host buffer");
  if (cfg.m == 0 || cfg.k == 0 || cfg.h == 0 || cfg.n == 0)
    throw std::runtime_error("direct bridge run received zero shape dimension");
  const DecodeLayout layout = make_decode_layout(
      cfg.h, cfg.n, cfg.dtype, cfg.decode_storage, cfg.decode_block_size,
      cfg.decode_quant_axis, cfg.decode_tile_h, cfg.decode_tile_n);
  if (layout.int4) {
    if (!cfg.decode_packed_weights || !cfg.decode_scales)
      throw std::runtime_error(
          "int4 direct bridge run received null packed/scales");
    if (cfg.decode_packed_bytes != layout.packed_bytes() ||
        cfg.decode_scale_bytes != layout.scale_bytes())
      throw std::runtime_error(
          "int4 direct bridge run received invalid packed/scales byte counts");
  } else if (!cfg.decode_weights) {
    throw std::runtime_error(
        "dense direct bridge run received null decode weights");
  }
}

void validate_gpu_stage_config(const LlmLinearGpuStageRunConfig &cfg) {
  if (cfg.abi_version != kAbiVersion)
    throw std::runtime_error("unsupported GPU stage bridge ABI version");
  if (cfg.stage != kGpuStagePrefill && cfg.stage != kGpuStageDecode)
    throw std::runtime_error("unsupported GPU stage enum");
  if (!cfg.input || !cfg.weights || !cfg.output || !cfg.gpu_so ||
      !cfg.gpu_so[0])
    throw std::runtime_error("GPU stage run received null buffer or library");
  if (cfg.stage == kGpuStagePrefill && (cfg.m == 0 || cfg.k == 0 || cfg.h == 0))
    throw std::runtime_error("GPU prefill stage received zero shape dimension");
  if (cfg.stage == kGpuStageDecode && (cfg.h == 0 || cfg.n == 0))
    throw std::runtime_error("GPU decode stage received zero shape dimension");
  const DecodeLayout layout =
      make_decode_layout(cfg.h, cfg.n, cfg.dtype, cfg.decode_storage,
                         cfg.decode_block_size, cfg.decode_quant_axis);
  if (cfg.stage == kGpuStageDecode && layout.int4) {
    if (!cfg.scales)
      throw std::runtime_error("int4 GPU decode stage received null scales");
    if (cfg.decode_packed_bytes != layout.packed_bytes() ||
        cfg.decode_scale_bytes != layout.scale_bytes())
      throw std::runtime_error(
          "int4 GPU decode stage received invalid packed/scales byte counts");
  }
}

void run_gpu_prefill(const LlmLinearDirectRunConfig &cfg,
                     PooledHipVmemBo &input, PooledHipVmemBo &weights,
                     PooledHipVmemBo &handoff) {
  SharedLibrary lib(cfg.gpu_prefill_so);
  PrefillFn prefill = lib.symbol<PrefillFn>("llm_linear_prefill");
  auto input_ref = memref2(input->hip_ptr(), static_cast<int64_t>(cfg.m),
                           static_cast<int64_t>(cfg.k));
  auto weight_ref = memref2(weights->hip_ptr(), static_cast<int64_t>(cfg.k),
                            static_cast<int64_t>(cfg.h));
  auto output_ref = memref2(handoff->hip_ptr(), static_cast<int64_t>(cfg.m),
                            static_cast<int64_t>(cfg.h));
  prefill(&input_ref, &weight_ref, &output_ref);
  check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize(gpu prefill)");
}

void run_gpu_decode(const LlmLinearDirectRunConfig &cfg,
                    PooledHipVmemBo &handoff, PooledHipVmemBo &weights,
                    PooledHipVmemBo &output, uint64_t offset_bytes) {
  SharedLibrary lib(cfg.gpu_decode_so);
  DecodeFn decode = lib.symbol<DecodeFn>("llm_linear_decode");
  auto input_ref =
      memref1(handoff->hip_ptr(offset_bytes), static_cast<int64_t>(cfg.h));
  auto weight_ref = memref2(weights->hip_ptr(), static_cast<int64_t>(cfg.h),
                            static_cast<int64_t>(cfg.n));
  auto output_ref = memref1(output->hip_ptr(), static_cast<int64_t>(cfg.n));
  decode(&input_ref, &weight_ref, &output_ref);
  check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize(gpu decode)");
}

void run_gpu_decode_dense_tile(DecodeFn decode,
                               const LlmLinearDirectRunConfig &cfg,
                               const DecodeLayout &layout,
                               PooledHipVmemBo &handoff,
                               PooledHipVmemBo &weights,
                               PooledHipVmemBo &output, uint64_t offset_bytes) {
  auto input_ref = memref1(handoff->hip_ptr(offset_bytes),
                           static_cast<int64_t>(layout.tile_h));
  auto weight_ref =
      memref2(weights->hip_ptr(), static_cast<int64_t>(layout.tile_h),
              static_cast<int64_t>(layout.tile_n));
  auto output_ref =
      memref1(output->hip_ptr(), static_cast<int64_t>(layout.tile_n));
  decode(&input_ref, &weight_ref, &output_ref);
  check_hip(hipDeviceSynchronize(),
            "hipDeviceSynchronize(gpu dense decode tile)");
}

void run_gpu_decode_int4(const LlmLinearDirectRunConfig &cfg,
                         const DecodeLayout &layout, PooledHipVmemBo &handoff,
                         PooledHipVmemBo &packed, PooledHipVmemBo &scales,
                         PooledHipVmemBo &output, uint64_t offset_bytes) {
  SharedLibrary lib(cfg.gpu_decode_so);
  DecodeInt4Fn decode = lib.symbol<DecodeInt4Fn>("llm_linear_decode_int4");
  auto input_ref =
      memref1(handoff->hip_ptr(offset_bytes), static_cast<int64_t>(cfg.h));
  auto packed_ref = memref2(packed->hip_ptr(), static_cast<int64_t>(cfg.h),
                            static_cast<int64_t>(layout.packed_word_cols));
  auto scales_ref =
      memref2(scales->hip_ptr(), static_cast<int64_t>(layout.block_count),
              static_cast<int64_t>(layout.n));
  auto output_ref = memref1(output->hip_ptr(), static_cast<int64_t>(cfg.n));
  decode(&input_ref, &packed_ref, &scales_ref, &output_ref);
  check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize(gpu int4 decode)");
}

void run_gpu_prefill_stage(const LlmLinearGpuStageRunConfig &cfg,
                           HipDeviceBuffer &input, HipDeviceBuffer &weights,
                           HipDeviceBuffer &output) {
  SharedLibrary lib(cfg.gpu_so);
  PrefillFn prefill = lib.symbol<PrefillFn>("llm_linear_prefill");
  auto input_ref = memref2(input.hip_ptr(), static_cast<int64_t>(cfg.m),
                           static_cast<int64_t>(cfg.k));
  auto weight_ref = memref2(weights.hip_ptr(), static_cast<int64_t>(cfg.k),
                            static_cast<int64_t>(cfg.h));
  auto output_ref = memref2(output.hip_ptr(), static_cast<int64_t>(cfg.m),
                            static_cast<int64_t>(cfg.h));
  prefill(&input_ref, &weight_ref, &output_ref);
  check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize(native gpu prefill)");
}

void run_gpu_decode_stage(const LlmLinearGpuStageRunConfig &cfg,
                          HipDeviceBuffer &input, HipDeviceBuffer &weights,
                          HipDeviceBuffer &output) {
  SharedLibrary lib(cfg.gpu_so);
  DecodeFn decode = lib.symbol<DecodeFn>("llm_linear_decode");
  auto input_ref = memref1(input.hip_ptr(), static_cast<int64_t>(cfg.h));
  auto weight_ref = memref2(weights.hip_ptr(), static_cast<int64_t>(cfg.h),
                            static_cast<int64_t>(cfg.n));
  auto output_ref = memref1(output.hip_ptr(), static_cast<int64_t>(cfg.n));
  decode(&input_ref, &weight_ref, &output_ref);
  check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize(native gpu decode)");
}

void run_gpu_decode_int4_stage(const LlmLinearGpuStageRunConfig &cfg,
                               const DecodeLayout &layout,
                               HipDeviceBuffer &input, HipDeviceBuffer &packed,
                               HipDeviceBuffer &scales,
                               HipDeviceBuffer &output) {
  SharedLibrary lib(cfg.gpu_so);
  DecodeInt4Fn decode = lib.symbol<DecodeInt4Fn>("llm_linear_decode_int4");
  auto input_ref = memref1(input.hip_ptr(), static_cast<int64_t>(cfg.h));
  auto packed_ref = memref2(packed.hip_ptr(), static_cast<int64_t>(cfg.h),
                            static_cast<int64_t>(layout.packed_word_cols));
  auto scales_ref =
      memref2(scales.hip_ptr(), static_cast<int64_t>(layout.block_count),
              static_cast<int64_t>(layout.n));
  auto output_ref = memref1(output.hip_ptr(), static_cast<int64_t>(cfg.n));
  decode(&input_ref, &packed_ref, &scales_ref, &output_ref);
  check_hip(hipDeviceSynchronize(),
            "hipDeviceSynchronize(native gpu int4 decode)");
}

void copy_gpu_decode_row(PooledHipVmemBo &handoff, PooledHipVmemBo &row,
                         size_t bytes, uint64_t offset_bytes) {
  if (bytes == 0)
    return;
  check_hip(hipMemcpy(row->hip_ptr(), handoff->hip_ptr(offset_bytes), bytes,
                      hipMemcpyDeviceToDevice),
            "hipMemcpy D2D(decode row)");
  check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize(decode row D2D)");
}

void run_gpu_to_npu(const LlmLinearDirectRunConfig &cfg,
                    LlmLinearDirectRunResult &result) {
  NpuKernel decode_kernel(cfg.npu_decode_xclbin, cfg.npu_decode_insts,
                          cfg.npu_kernel_name);
  const size_t es = elem_size(cfg.dtype);
  const size_t input_bytes = cfg.m * cfg.k * es;
  const size_t prefill_weight_bytes = cfg.k * cfg.h * es;
  const size_t handoff_bytes = cfg.m * cfg.h * es;
  const DecodeLayout decode_layout = make_decode_layout(
      cfg.h, cfg.n, cfg.dtype, cfg.decode_storage, cfg.decode_block_size,
      cfg.decode_quant_axis, cfg.decode_tile_h, cfg.decode_tile_n);
  const size_t decode_weight_bytes =
      decode_layout.int4
          ? decode_layout.packed_tile_bytes()
          : (decode_layout.tiled_dense() ? decode_layout.dense_tile_bytes()
                                         : decode_layout.dense_weight_bytes());
  const size_t decode_scale_bytes =
      decode_layout.int4 ? decode_layout.scale_tile_bytes() : 0;
  const size_t output_bytes = cfg.n * es;
  const size_t output_tile_bytes = decode_layout.tile_n * es;

  auto &device = decode_kernel.device();
  auto input = acquire_hip_vmem_xrt_bo(device, input_bytes);
  auto prefill_weights = acquire_hip_vmem_xrt_bo(device, prefill_weight_bytes);
  auto handoff = acquire_hip_vmem_xrt_bo(device, handoff_bytes);
  auto decode_weights = acquire_hip_vmem_xrt_bo(device, decode_weight_bytes);
  auto decode_scales = decode_layout.int4
                           ? acquire_hip_vmem_xrt_bo(device, decode_scale_bytes)
                           : PooledHipVmemBo();
  auto output = acquire_hip_vmem_xrt_bo(
      device, decode_layout.tiled_decode() ? output_tile_bytes : output_bytes);

  input->hip_write_host(cfg.input);
  prefill_weights->hip_write_host(cfg.prefill_weights);
  if (!decode_layout.tiled_decode())
    decode_weights->xrt_write_host(
        decode_layout.int4 ? cfg.decode_packed_weights : cfg.decode_weights);
  if (decode_layout.int4 && !decode_layout.tiled_int4())
    decode_scales->xrt_write_host(cfg.decode_scales);

  const uint64_t prefill_start = now_us();
  run_gpu_prefill(cfg, input, prefill_weights, handoff);
  result.prefill_us = now_us() - prefill_start;
  const uint64_t offset = (cfg.m - 1) * cfg.h * es;
  const size_t full_decode_input_bytes = cfg.h * es;
  const size_t decode_input_bytes = decode_layout.tiled_dense()
                                        ? decode_layout.tile_h * es
                                        : full_decode_input_bytes;
  auto decode_input = acquire_hip_vmem_xrt_bo(device, decode_input_bytes);
  if (!decode_layout.tiled_dense()) {
    const uint64_t handoff_start = now_us();
    copy_gpu_decode_row(handoff, decode_input, decode_input_bytes, offset);
    decode_input->sync_to_xrt(decode_input_bytes);
    result.handoff_us = now_us() - handoff_start;
  } else {
    result.handoff_us = 0;
  }

  const uint64_t decode_start = now_us();
  if (decode_layout.tiled_dense()) {
    if (cfg.dtype != 0)
      throw std::runtime_error("tiled dense direct NPU decode requires bf16");
    std::vector<uint8_t> staged_decode_weights(decode_weight_bytes, 0);
    std::vector<uint8_t> partial_output(output_tile_bytes, 0);
    std::vector<uint8_t> zero_output(output_tile_bytes, 0);
    std::vector<float> accum(decode_layout.tile_n, 0.0f);
    for (uint64_t start_n = 0; start_n < cfg.n;
         start_n += decode_layout.tile_n) {
      std::fill(accum.begin(), accum.end(), 0.0f);
      for (uint64_t start_h = 0; start_h < cfg.h;
           start_h += decode_layout.tile_h) {
        const uint64_t handoff_start = now_us();
        copy_gpu_decode_row(handoff, decode_input, decode_input_bytes,
                            offset + start_h * es);
        decode_input->sync_to_xrt(decode_input_bytes);
        result.handoff_us += now_us() - handoff_start;
        stage_decode_dense_weight_tile(cfg, decode_layout, start_h, start_n,
                                       staged_decode_weights);
        decode_weights->xrt_write_host(staged_decode_weights.data());
        output->xrt_write_host(zero_output.data());
        decode_kernel.run3(decode_input, decode_weights, output);
        output->xrt_read_host(partial_output.data(), output_tile_bytes, 0);
        const auto *partial =
            reinterpret_cast<const uint16_t *>(partial_output.data());
        for (uint64_t lane = 0; lane < decode_layout.tile_n; ++lane)
          accum[lane] += bf16_to_float(partial[lane]);
      }
      auto *dst = reinterpret_cast<uint16_t *>(
          static_cast<uint8_t *>(cfg.output) + start_n * es);
      for (uint64_t lane = 0; lane < decode_layout.tile_n; ++lane)
        dst[lane] = float_to_bf16(accum[lane]);
    }
  } else if (decode_layout.tiled_int4()) {
    std::vector<uint8_t> staged_decode_weights(decode_weight_bytes, 0);
    std::vector<uint8_t> staged_decode_scales(decode_scale_bytes, 0);
    for (uint64_t start_n = 0; start_n < cfg.n;
         start_n += decode_layout.tile_n) {
      stage_decode_int4_packed_tile(cfg, decode_layout, start_n,
                                    staged_decode_weights);
      stage_decode_int4_scale_tile(cfg, decode_layout, start_n,
                                   staged_decode_scales);
      decode_weights->xrt_write_host(staged_decode_weights.data());
      decode_scales->xrt_write_host(staged_decode_scales.data());
      decode_kernel.run4(decode_input, decode_weights, decode_scales, output);
      output->xrt_read_host(static_cast<uint8_t *>(cfg.output) + start_n * es);
    }
  } else if (decode_layout.int4) {
    decode_kernel.run4(decode_input, decode_weights, decode_scales, output);
  } else {
    decode_kernel.run3(decode_input, decode_weights, output);
  }
  result.decode_us = now_us() - decode_start;

  if (cfg.prefill_output)
    handoff->sync_to_xrt(handoff_bytes);
  if (!decode_layout.tiled_decode())
    output->xrt_read_host(cfg.output);
  if (cfg.prefill_output)
    handoff->xrt_read_host(cfg.prefill_output);
  if (cfg.decode_input) {
    if (decode_layout.tiled_dense())
      handoff->xrt_read_host(cfg.decode_input, full_decode_input_bytes, offset);
    else
      decode_input->xrt_read_host(cfg.decode_input);
  }
  result.direct_bytes = full_decode_input_bytes;
  result.subview_offset_bytes = offset;
  copy_cstr(result.sync_events, sizeof(result.sync_events),
            "hipDeviceSynchronize:producer=gpu_prefill;"
            "hipMemcpyDtoD:stage=decode_input_row;"
            "xrtBoSyncToDevice:handoff=gpu_to_npu_row;"
            "xrtRunWait:consumer=npu_decode;"
            "xrtBoSyncFromDevice:final_output");
}

void run_npu_to_gpu(const LlmLinearDirectRunConfig &cfg,
                    LlmLinearDirectRunResult &result) {
  NpuKernel prefill_kernel(cfg.npu_prefill_xclbin, cfg.npu_prefill_insts,
                           cfg.npu_kernel_name);
  const size_t es = elem_size(cfg.dtype);
  const uint64_t prefill_tile_h = std::min<uint64_t>(cfg.h, 512);
  const size_t row_input_bytes = cfg.k * es;
  const size_t prefill_weight_tile_bytes = cfg.k * prefill_tile_h * es;
  const size_t handoff_bytes = cfg.m * cfg.h * es;
  const DecodeLayout decode_layout = make_decode_layout(
      cfg.h, cfg.n, cfg.dtype, cfg.decode_storage, cfg.decode_block_size,
      cfg.decode_quant_axis, cfg.decode_tile_h, cfg.decode_tile_n);
  const size_t decode_weight_bytes =
      decode_layout.int4
          ? decode_layout.packed_bytes()
          : (decode_layout.tiled_dense() ? decode_layout.dense_tile_bytes()
                                         : decode_layout.dense_weight_bytes());
  const size_t decode_scale_bytes =
      decode_layout.int4 ? decode_layout.scale_bytes() : 0;
  const size_t output_bytes = cfg.n * es;
  const size_t output_tile_bytes = decode_layout.tile_n * es;

  auto &device = prefill_kernel.device();
  auto input = acquire_hip_vmem_xrt_bo(device, row_input_bytes);
  auto prefill_weights =
      acquire_hip_vmem_xrt_bo(device, prefill_weight_tile_bytes);
  auto handoff = acquire_hip_vmem_xrt_bo(device, handoff_bytes);
  auto prefill_temp = acquire_hip_vmem_xrt_bo(device, prefill_tile_h * es);
  auto decode_weights = acquire_hip_vmem_xrt_bo(device, decode_weight_bytes);
  auto decode_scales = decode_layout.int4
                           ? acquire_hip_vmem_xrt_bo(device, decode_scale_bytes)
                           : PooledHipVmemBo();
  auto output = acquire_hip_vmem_xrt_bo(
      device, decode_layout.tiled_decode() ? output_tile_bytes : output_bytes);

  if (!decode_layout.tiled_decode())
    decode_weights->hip_write_host(
        decode_layout.int4 ? cfg.decode_packed_weights : cfg.decode_weights);
  if (decode_layout.int4)
    decode_scales->hip_write_host(cfg.decode_scales);

  const size_t row_output_bytes = cfg.h * es;
  std::vector<uint8_t> staged_weights(prefill_weight_tile_bytes, 0);
  const uint64_t prefill_start = now_us();
  for (uint64_t row = 0; row < cfg.m; ++row) {
    input->xrt_write_host(static_cast<const uint8_t *>(cfg.input) +
                          row * row_input_bytes);
    for (uint64_t tile = 0; tile < cfg.h; tile += prefill_tile_h) {
      const uint64_t width = std::min<uint64_t>(prefill_tile_h, cfg.h - tile);
      std::fill(staged_weights.begin(), staged_weights.end(), 0);
      for (uint64_t k = 0; k < cfg.k; ++k) {
        const uint8_t *src = static_cast<const uint8_t *>(cfg.prefill_weights) +
                             (k * cfg.h + tile) * es;
        uint8_t *dst = staged_weights.data() + k * prefill_tile_h * es;
        std::memcpy(dst, src, width * es);
      }
      prefill_weights->xrt_write_host(staged_weights.data());
      trace_step("n2g: before npu prefill tile");
      prefill_kernel.run3(input, prefill_weights, prefill_temp);
      trace_step("n2g: after npu prefill tile");
      prefill_temp->sync_from_xrt(width * es, 0);
      check_hip(hipMemcpy(handoff->hip_ptr(row * row_output_bytes + tile * es),
                          prefill_temp->hip_ptr(), width * es,
                          hipMemcpyDeviceToDevice),
                "hipMemcpy D2D(npu prefill tile handoff)");
    }
  }
  check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize(npu prefill rows)");
  result.prefill_us = now_us() - prefill_start;
  const uint64_t handoff_start = now_us();
  check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize(npu_to_gpu handoff)");
  result.handoff_us = now_us() - handoff_start;

  const uint64_t offset = (cfg.m - 1) * cfg.h * es;
  trace_step("n2g: before gpu decode");
  const uint64_t decode_start = now_us();
  if (decode_layout.tiled_dense()) {
    if (cfg.dtype != 0)
      throw std::runtime_error("tiled dense direct GPU decode requires bf16");
    std::vector<uint8_t> staged_decode_weights(decode_weight_bytes, 0);
    std::vector<uint8_t> partial_output(output_tile_bytes, 0);
    std::vector<float> accum(decode_layout.tile_n, 0.0f);
    SharedLibrary lib(cfg.gpu_decode_so);
    DecodeFn decode = lib.symbol<DecodeFn>("llm_linear_decode");
    for (uint64_t start_n = 0; start_n < cfg.n;
         start_n += decode_layout.tile_n) {
      std::fill(accum.begin(), accum.end(), 0.0f);
      for (uint64_t start_h = 0; start_h < cfg.h;
           start_h += decode_layout.tile_h) {
        stage_decode_dense_weight_tile(cfg, decode_layout, start_h, start_n,
                                       staged_decode_weights);
        decode_weights->hip_write_host(staged_decode_weights.data());
        run_gpu_decode_dense_tile(decode, cfg, decode_layout, handoff,
                                  decode_weights, output,
                                  offset + start_h * es);
        output->hip_read_host(partial_output.data());
        const auto *partial =
            reinterpret_cast<const uint16_t *>(partial_output.data());
        for (uint64_t lane = 0; lane < decode_layout.tile_n; ++lane)
          accum[lane] += bf16_to_float(partial[lane]);
      }
      auto *dst = reinterpret_cast<uint16_t *>(
          static_cast<uint8_t *>(cfg.output) + start_n * es);
      for (uint64_t lane = 0; lane < decode_layout.tile_n; ++lane)
        dst[lane] = float_to_bf16(accum[lane]);
    }
  } else if (decode_layout.int4) {
    run_gpu_decode_int4(cfg, decode_layout, handoff, decode_weights,
                        decode_scales, output, offset);
  } else {
    run_gpu_decode(cfg, handoff, decode_weights, output, offset);
  }
  result.decode_us = now_us() - decode_start;
  trace_step("n2g: after gpu decode");

  if (!decode_layout.tiled_decode()) {
    trace_step("n2g: before output hip_read_host");
    output->hip_read_host(cfg.output);
    trace_step("n2g: after output hip_read_host");
  }
  if (cfg.prefill_output)
    handoff->hip_read_host(cfg.prefill_output);
  if (cfg.decode_input) {
    handoff->hip_read_host(cfg.decode_input, cfg.h * es, offset);
  }
  result.direct_bytes = cfg.h * es;
  result.subview_offset_bytes = offset;
  copy_cstr(result.sync_events, sizeof(result.sync_events),
            "xrtRunWait:producer=npu_prefill_rows;"
            "xrtBoSyncFromDevice:handoff=npu_to_gpu_rows;"
            "hipMemcpyDtoD:stage=prefill_rows;"
            "hipDeviceSynchronize:consumer=gpu_decode;"
            "hipMemcpyDtoH:final_output");
}

DecodeLayout
make_resident_decode_layout(const LlmLinearResidentPipelinePrepareConfig &cfg) {
  const bool gpu = cfg.backend == kResidentBackendGpu;
  return make_decode_layout(cfg.h, cfg.n, cfg.dtype, cfg.decode_storage,
                            cfg.decode_block_size, cfg.decode_quant_axis,
                            gpu ? cfg.gpu_decode_tile_h : cfg.npu_decode_tile_h,
                            gpu ? cfg.gpu_decode_tile_n
                                : cfg.npu_decode_tile_n);
}

void validate_resident_prepare_config(
    const LlmLinearResidentPipelinePrepareConfig &cfg) {
  if (cfg.abi_version != kAbiVersion)
    throw std::runtime_error("unsupported resident pipeline ABI version");
  if (cfg.backend != kResidentBackendGpu && cfg.backend != kResidentBackendNpu)
    throw std::runtime_error("unsupported resident pipeline backend");
  if (!cfg.prefill_weights || cfg.m == 0 || cfg.k == 0 || cfg.h == 0 ||
      cfg.n == 0)
    throw std::runtime_error(
        "resident pipeline received invalid shape/weights");
  const DecodeLayout layout = make_resident_decode_layout(cfg);
  if (layout.int4) {
    if (!cfg.decode_packed_weights || !cfg.decode_scales)
      throw std::runtime_error("resident int4 pipeline missing packed/scales");
    if (cfg.decode_packed_bytes != layout.packed_bytes() ||
        cfg.decode_scale_bytes != layout.scale_bytes())
      throw std::runtime_error(
          "resident int4 pipeline received invalid packed/scales byte counts");
  } else if (!cfg.decode_weights) {
    throw std::runtime_error("resident dense pipeline missing decode weights");
  }
  if (cfg.backend == kResidentBackendGpu) {
    if (!cfg.gpu_prefill_so || !cfg.gpu_prefill_so[0] || !cfg.gpu_decode_so ||
        !cfg.gpu_decode_so[0])
      throw std::runtime_error("resident GPU pipeline missing GPU libraries");
  } else {
    if (!cfg.npu_prefill_xclbin || !cfg.npu_prefill_insts ||
        !cfg.npu_decode_xclbin || !cfg.npu_decode_insts)
      throw std::runtime_error("resident NPU pipeline missing NPU artifacts");
  }
}

class ResidentPipeline {
public:
  virtual ~ResidentPipeline() = default;
  virtual void run(const LlmLinearResidentPipelineRunConfig &cfg,
                   LlmLinearResidentPipelineRunResult &result) = 0;
  virtual const char *storage_kind() const = 0;
  uint64_t static_upload_bytes() const { return static_upload_bytes_; }

protected:
  uint64_t static_upload_bytes_ = 0;
};

class GpuResidentPipeline final : public ResidentPipeline {
public:
  explicit GpuResidentPipeline(
      const LlmLinearResidentPipelinePrepareConfig &cfg)
      : m_(cfg.m), k_(cfg.k), h_(cfg.h), n_(cfg.n), dtype_(cfg.dtype),
        es_(elem_size(cfg.dtype)), layout_(make_resident_decode_layout(cfg)),
        prefill_lib_(std::make_unique<SharedLibrary>(cfg.gpu_prefill_so)),
        decode_lib_(std::make_unique<SharedLibrary>(cfg.gpu_decode_so)),
        input_(std::make_unique<HipDeviceBuffer>(cfg.m * cfg.k * es_)),
        prefill_weights_(
            std::make_unique<HipDeviceBuffer>(cfg.k * cfg.h * es_)),
        handoff_(std::make_unique<HipDeviceBuffer>(cfg.m * cfg.h * es_)),
        output_(std::make_unique<HipDeviceBuffer>(cfg.n * es_)) {
    check_hip(hipSetDevice(0), "hipSetDevice(0)");
    prefill_fn_ = prefill_lib_->symbol<PrefillFn>("llm_linear_prefill");
    prefill_weights_->hip_write_host(cfg.prefill_weights);
    static_upload_bytes_ += cfg.k * cfg.h * es_;
    if (layout_.int4) {
      decode_int4_fn_ =
          decode_lib_->symbol<DecodeInt4Fn>("llm_linear_decode_int4");
      decode_weights_ =
          std::make_unique<HipDeviceBuffer>(layout_.packed_bytes());
      decode_scales_ = std::make_unique<HipDeviceBuffer>(layout_.scale_bytes());
      decode_weights_->hip_write_host(cfg.decode_packed_weights);
      decode_scales_->hip_write_host(cfg.decode_scales);
      static_upload_bytes_ += layout_.packed_bytes() + layout_.scale_bytes();
    } else {
      decode_fn_ = decode_lib_->symbol<DecodeFn>("llm_linear_decode");
      decode_weights_ =
          std::make_unique<HipDeviceBuffer>(layout_.dense_weight_bytes());
      decode_weights_->hip_write_host(cfg.decode_weights);
      static_upload_bytes_ += layout_.dense_weight_bytes();
      if (layout_.tiled_dense()) {
        partial_ = std::make_unique<HipDeviceBuffer>(layout_.tile_n * es_);
        accum_ =
            std::make_unique<HipDeviceBuffer>(layout_.tile_n * sizeof(float));
      }
    }
  }

  const char *storage_kind() const override { return "hip_device_allocation"; }

  void run(const LlmLinearResidentPipelineRunConfig &cfg,
           LlmLinearResidentPipelineRunResult &result) override {
    if (!cfg.input || !cfg.output)
      throw std::runtime_error("resident GPU run missing input/output");
    const uint64_t upload_start = now_us();
    input_->hip_write_host(cfg.input);
    result.input_upload_us = now_us() - upload_start;

    const uint64_t prefill_start = now_us();
    for (uint64_t row = 0; row < m_; ++row) {
      auto input_ref =
          memref2(input_->hip_ptr(row * k_ * es_), 1, static_cast<int64_t>(k_));
      auto weight_ref =
          memref2(prefill_weights_->hip_ptr(), static_cast<int64_t>(k_),
                  static_cast<int64_t>(h_));
      auto output_ref = memref2(handoff_->hip_ptr(row * h_ * es_), 1,
                                static_cast<int64_t>(h_));
      prefill_fn_(&input_ref, &weight_ref, &output_ref);
      result.launch_count += 1;
    }
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize(resident gpu prefill)");
    result.prefill_us = now_us() - prefill_start;

    const uint64_t decode_start = now_us();
    const uint64_t offset = (m_ - 1) * h_ * es_;
    if (layout_.int4) {
      auto input_ref =
          memref1(handoff_->hip_ptr(offset), static_cast<int64_t>(h_));
      auto packed_ref =
          memref2(decode_weights_->hip_ptr(), static_cast<int64_t>(h_),
                  static_cast<int64_t>(layout_.packed_word_cols));
      auto scales_ref = memref2(decode_scales_->hip_ptr(),
                                static_cast<int64_t>(layout_.block_count),
                                static_cast<int64_t>(n_));
      auto output_ref = memref1(output_->hip_ptr(), static_cast<int64_t>(n_));
      decode_int4_fn_(&input_ref, &packed_ref, &scales_ref, &output_ref);
      result.launch_count += 1;
      check_hip(hipDeviceSynchronize(),
                "hipDeviceSynchronize(resident gpu int4 decode)");
    } else if (layout_.tiled_dense()) {
      check_hip(hipMemset(accum_->hip_ptr(), 0, layout_.tile_n * sizeof(float)),
                "hipMemset(resident gpu decode accum)");
      for (uint64_t start_h = 0; start_h < h_; start_h += layout_.tile_h) {
        auto input_ref = memref1(handoff_->hip_ptr(offset + start_h * es_),
                                 static_cast<int64_t>(layout_.tile_h));
        auto weight_ref = memref2_strided(
            decode_weights_->hip_ptr((start_h * n_) * es_),
            static_cast<int64_t>(layout_.tile_h),
            static_cast<int64_t>(layout_.tile_n), static_cast<int64_t>(n_), 1);
        auto partial_ref =
            memref1(partial_->hip_ptr(), static_cast<int64_t>(layout_.tile_n));
        decode_fn_(&input_ref, &weight_ref, &partial_ref);
        result.launch_count += 1;
        check_hip(hipDeviceSynchronize(),
                  "hipDeviceSynchronize(resident gpu dense decode tile)");
        launch_accumulate_bf16_to_f32(partial_->hip_ptr(), accum_->hip_ptr(),
                                      layout_.tile_n);
      }
      launch_f32_to_bf16(accum_->hip_ptr(), output_->hip_ptr(), layout_.tile_n);
      check_hip(hipDeviceSynchronize(),
                "hipDeviceSynchronize(resident gpu dense decode reduction)");
    } else {
      auto input_ref =
          memref1(handoff_->hip_ptr(offset), static_cast<int64_t>(h_));
      auto weight_ref =
          memref2(decode_weights_->hip_ptr(), static_cast<int64_t>(h_),
                  static_cast<int64_t>(n_));
      auto output_ref = memref1(output_->hip_ptr(), static_cast<int64_t>(n_));
      decode_fn_(&input_ref, &weight_ref, &output_ref);
      result.launch_count += 1;
      check_hip(hipDeviceSynchronize(),
                "hipDeviceSynchronize(resident gpu dense decode)");
    }
    result.decode_us = now_us() - decode_start;

    const uint64_t read_start = now_us();
    output_->hip_read_host(cfg.output);
    result.output_readback_us = now_us() - read_start;
    if (cfg.capture_details && cfg.prefill_output)
      handoff_->hip_read_host(cfg.prefill_output);
    if (cfg.capture_details && cfg.decode_input)
      handoff_->hip_read_host(cfg.decode_input, h_ * es_, offset);
  }

private:
  uint64_t m_;
  uint64_t k_;
  uint64_t h_;
  uint64_t n_;
  uint32_t dtype_;
  size_t es_;
  DecodeLayout layout_;
  std::unique_ptr<SharedLibrary> prefill_lib_;
  std::unique_ptr<SharedLibrary> decode_lib_;
  PrefillFn prefill_fn_ = nullptr;
  DecodeFn decode_fn_ = nullptr;
  DecodeInt4Fn decode_int4_fn_ = nullptr;
  std::unique_ptr<HipDeviceBuffer> input_;
  std::unique_ptr<HipDeviceBuffer> prefill_weights_;
  std::unique_ptr<HipDeviceBuffer> handoff_;
  std::unique_ptr<HipDeviceBuffer> decode_weights_;
  std::unique_ptr<HipDeviceBuffer> decode_scales_;
  std::unique_ptr<HipDeviceBuffer> output_;
  std::unique_ptr<HipDeviceBuffer> partial_;
  std::unique_ptr<HipDeviceBuffer> accum_;
};

class NpuResidentPipeline final : public ResidentPipeline {
public:
  explicit NpuResidentPipeline(
      const LlmLinearResidentPipelinePrepareConfig &cfg)
      : m_(cfg.m), k_(cfg.k), h_(cfg.h), n_(cfg.n), dtype_(cfg.dtype),
        es_(elem_size(cfg.dtype)), layout_(make_resident_decode_layout(cfg)),
        prefill_tile_h_(std::min<uint64_t>(cfg.h, 512)),
        device_(std::make_shared<xrt::device>(0)),
        prefill_kernel_(std::make_unique<NpuKernel>(
            cfg.npu_prefill_xclbin, cfg.npu_prefill_insts, cfg.npu_kernel_name,
            device_)),
        decode_kernel_(std::make_unique<NpuKernel>(
            cfg.npu_decode_xclbin, cfg.npu_decode_insts, cfg.npu_kernel_name,
            device_)) {
    check_hip(hipSetDevice(0), "hipSetDevice(0)");
    auto &device = prefill_kernel_->device();
    input_row_ = acquire_hip_vmem_xrt_bo(device, k_ * es_);
    handoff_ = acquire_hip_vmem_xrt_bo(device, m_ * h_ * es_);
    prefill_temp_ = acquire_hip_vmem_xrt_bo(device, prefill_tile_h_ * es_);
    output_full_ = acquire_hip_vmem_xrt_bo(device, n_ * es_);
    output_tile_ = acquire_hip_vmem_xrt_bo(device, layout_.tile_n * es_);

    prepare_prefill_weight_tiles(cfg, device);
    if (layout_.int4)
      prepare_int4_decode_tiles(cfg, device);
    else
      prepare_dense_decode_tiles(cfg, device);
  }

  const char *storage_kind() const override { return "xrt_static_bo"; }

  void run(const LlmLinearResidentPipelineRunConfig &cfg,
           LlmLinearResidentPipelineRunResult &result) override {
    if (!cfg.input || !cfg.output)
      throw std::runtime_error("resident NPU run missing input/output");
    const uint64_t prefill_start = now_us();
    for (uint64_t row = 0; row < m_; ++row) {
      const uint64_t upload_start = now_us();
      input_row_->xrt_write_host(static_cast<const uint8_t *>(cfg.input) +
                                 row * k_ * es_);
      result.input_upload_us += now_us() - upload_start;
      for (uint64_t tile = 0; tile < prefill_weight_tiles_.size(); ++tile) {
        const uint64_t start_h = tile * prefill_tile_h_;
        const uint64_t width =
            std::min<uint64_t>(prefill_tile_h_, h_ - start_h);
        prefill_kernel_->run3(input_row_, prefill_weight_tiles_[tile],
                              prefill_temp_);
        result.launch_count += 1;
        prefill_temp_->sync_from_xrt(width * es_, 0);
        check_hip(hipMemcpy(handoff_->hip_ptr(row * h_ * es_ + start_h * es_),
                            prefill_temp_->hip_ptr(), width * es_,
                            hipMemcpyDeviceToDevice),
                  "hipMemcpy D2D(resident npu prefill tile)");
      }
    }
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize(resident npu prefill)");
    result.prefill_us = now_us() - prefill_start;

    const uint64_t decode_start = now_us();
    if (layout_.int4)
      run_int4_decode(cfg, result);
    else
      run_dense_decode(cfg, result);
    result.decode_us = now_us() - decode_start;

    const uint64_t read_start = now_us();
    output_full_->hip_read_host(cfg.output);
    result.output_readback_us = now_us() - read_start;
    const uint64_t offset = (m_ - 1) * h_ * es_;
    if (cfg.capture_details && cfg.prefill_output)
      handoff_->hip_read_host(cfg.prefill_output);
    if (cfg.capture_details && cfg.decode_input)
      handoff_->hip_read_host(cfg.decode_input, h_ * es_, offset);
  }

private:
  void prepare_prefill_weight_tiles(
      const LlmLinearResidentPipelinePrepareConfig &cfg, xrt::device &device) {
    const uint64_t tiles = (h_ + prefill_tile_h_ - 1) / prefill_tile_h_;
    prefill_weight_tiles_.reserve(static_cast<size_t>(tiles));
    std::vector<uint8_t> staged(k_ * prefill_tile_h_ * es_, 0);
    const auto *src = static_cast<const uint8_t *>(cfg.prefill_weights);
    for (uint64_t tile = 0; tile < tiles; ++tile) {
      const uint64_t start_h = tile * prefill_tile_h_;
      const uint64_t width = std::min<uint64_t>(prefill_tile_h_, h_ - start_h);
      std::fill(staged.begin(), staged.end(), 0);
      for (uint64_t k = 0; k < k_; ++k) {
        std::memcpy(staged.data() + k * prefill_tile_h_ * es_,
                    src + (k * h_ + start_h) * es_, width * es_);
      }
      auto bo = acquire_hip_vmem_xrt_bo(device, staged.size());
      bo->xrt_write_host(staged.data());
      static_upload_bytes_ += staged.size();
      prefill_weight_tiles_.push_back(std::move(bo));
    }
  }

  void
  prepare_dense_decode_tiles(const LlmLinearResidentPipelinePrepareConfig &cfg,
                             xrt::device &device) {
    decode_input_ = acquire_hip_vmem_xrt_bo(device, layout_.tile_h * es_);
    accum_ = std::make_unique<HipDeviceBuffer>(layout_.tile_n * sizeof(float));
    const uint64_t h_tiles = h_ / layout_.tile_h;
    const uint64_t n_tiles = n_ / layout_.tile_n;
    decode_weight_tiles_.reserve(static_cast<size_t>(h_tiles * n_tiles));
    std::vector<uint8_t> staged(layout_.dense_tile_bytes(), 0);
    const auto *src = static_cast<const uint8_t *>(cfg.decode_weights);
    for (uint64_t start_n = 0; start_n < n_; start_n += layout_.tile_n) {
      for (uint64_t start_h = 0; start_h < h_; start_h += layout_.tile_h) {
        for (uint64_t row = 0; row < layout_.tile_h; ++row) {
          std::memcpy(staged.data() + row * layout_.tile_n * es_,
                      src + ((start_h + row) * n_ + start_n) * es_,
                      layout_.tile_n * es_);
        }
        auto bo = acquire_hip_vmem_xrt_bo(device, staged.size());
        bo->xrt_write_host(staged.data());
        static_upload_bytes_ += staged.size();
        decode_weight_tiles_.push_back(std::move(bo));
      }
    }
  }

  void
  prepare_int4_decode_tiles(const LlmLinearResidentPipelinePrepareConfig &cfg,
                            xrt::device &device) {
    decode_input_ = acquire_hip_vmem_xrt_bo(device, h_ * es_);
    const uint64_t n_tiles = n_ / layout_.tile_n;
    decode_weight_tiles_.reserve(static_cast<size_t>(n_tiles));
    decode_scale_tiles_.reserve(static_cast<size_t>(n_tiles));
    std::vector<uint8_t> packed(layout_.packed_tile_bytes(), 0);
    std::vector<uint8_t> scales(layout_.scale_tile_bytes(), 0);
    const auto *packed_src =
        static_cast<const uint32_t *>(cfg.decode_packed_weights);
    const auto *scale_src = static_cast<const float *>(cfg.decode_scales);
    const uint64_t packed_words_per_tile = layout_.tile_n / 8;
    for (uint64_t start_n = 0; start_n < n_; start_n += layout_.tile_n) {
      const uint64_t word_start = start_n / 8;
      auto *packed_dst = reinterpret_cast<uint32_t *>(packed.data());
      for (uint64_t row = 0; row < h_; ++row) {
        std::memcpy(packed_dst + row * packed_words_per_tile,
                    packed_src + row * layout_.packed_word_cols + word_start,
                    packed_words_per_tile * sizeof(uint32_t));
      }
      auto *scale_dst = reinterpret_cast<float *>(scales.data());
      for (uint64_t block = 0; block < layout_.block_count; ++block) {
        std::memcpy(scale_dst + block * layout_.tile_n,
                    scale_src + block * n_ + start_n,
                    layout_.tile_n * sizeof(float));
      }
      auto packed_bo = acquire_hip_vmem_xrt_bo(device, packed.size());
      auto scale_bo = acquire_hip_vmem_xrt_bo(device, scales.size());
      packed_bo->xrt_write_host(packed.data());
      scale_bo->xrt_write_host(scales.data());
      static_upload_bytes_ += packed.size() + scales.size();
      decode_weight_tiles_.push_back(std::move(packed_bo));
      decode_scale_tiles_.push_back(std::move(scale_bo));
    }
  }

  void run_dense_decode(const LlmLinearResidentPipelineRunConfig &,
                        LlmLinearResidentPipelineRunResult &result) {
    const uint64_t offset = (m_ - 1) * h_ * es_;
    uint64_t weight_index = 0;
    for (uint64_t start_n = 0; start_n < n_; start_n += layout_.tile_n) {
      check_hip(hipMemset(accum_->hip_ptr(), 0, layout_.tile_n * sizeof(float)),
                "hipMemset(resident npu decode accum)");
      for (uint64_t start_h = 0; start_h < h_; start_h += layout_.tile_h) {
        check_hip(hipMemcpy(decode_input_->hip_ptr(),
                            handoff_->hip_ptr(offset + start_h * es_),
                            layout_.tile_h * es_, hipMemcpyDeviceToDevice),
                  "hipMemcpy D2D(resident npu decode input tile)");
        check_hip(hipDeviceSynchronize(),
                  "hipDeviceSynchronize(resident npu decode input tile)");
        decode_input_->sync_to_xrt(layout_.tile_h * es_);
        decode_kernel_->run3(decode_input_, decode_weight_tiles_[weight_index],
                             output_tile_);
        result.launch_count += 1;
        output_tile_->sync_from_xrt(layout_.tile_n * es_);
        launch_accumulate_bf16_to_f32(output_tile_->hip_ptr(),
                                      accum_->hip_ptr(), layout_.tile_n);
        ++weight_index;
      }
      launch_f32_to_bf16(accum_->hip_ptr(),
                         output_full_->hip_ptr(start_n * es_), layout_.tile_n);
      check_hip(hipDeviceSynchronize(),
                "hipDeviceSynchronize(resident npu decode reduction)");
    }
  }

  void run_int4_decode(const LlmLinearResidentPipelineRunConfig &,
                       LlmLinearResidentPipelineRunResult &result) {
    const uint64_t offset = (m_ - 1) * h_ * es_;
    check_hip(hipMemcpy(decode_input_->hip_ptr(), handoff_->hip_ptr(offset),
                        h_ * es_, hipMemcpyDeviceToDevice),
              "hipMemcpy D2D(resident npu int4 decode input)");
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize(resident npu int4 decode input)");
    decode_input_->sync_to_xrt(h_ * es_);
    uint64_t tile = 0;
    for (uint64_t start_n = 0; start_n < n_; start_n += layout_.tile_n) {
      decode_kernel_->run4(decode_input_, decode_weight_tiles_[tile],
                           decode_scale_tiles_[tile], output_tile_);
      result.launch_count += 1;
      output_tile_->sync_from_xrt(layout_.tile_n * es_);
      check_hip(hipMemcpy(output_full_->hip_ptr(start_n * es_),
                          output_tile_->hip_ptr(), layout_.tile_n * es_,
                          hipMemcpyDeviceToDevice),
                "hipMemcpy D2D(resident npu int4 output tile)");
      ++tile;
    }
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize(resident npu int4 output)");
  }

  uint64_t m_;
  uint64_t k_;
  uint64_t h_;
  uint64_t n_;
  uint32_t dtype_;
  size_t es_;
  DecodeLayout layout_;
  uint64_t prefill_tile_h_;
  std::shared_ptr<xrt::device> device_;
  std::unique_ptr<NpuKernel> prefill_kernel_;
  std::unique_ptr<NpuKernel> decode_kernel_;
  PooledHipVmemBo input_row_;
  PooledHipVmemBo handoff_;
  PooledHipVmemBo prefill_temp_;
  PooledHipVmemBo decode_input_;
  PooledHipVmemBo output_tile_;
  PooledHipVmemBo output_full_;
  std::vector<PooledHipVmemBo> prefill_weight_tiles_;
  std::vector<PooledHipVmemBo> decode_weight_tiles_;
  std::vector<PooledHipVmemBo> decode_scale_tiles_;
  std::unique_ptr<HipDeviceBuffer> accum_;
};

std::mutex &resident_pipeline_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<uint64_t, std::unique_ptr<ResidentPipeline>> &
resident_pipelines() {
  static auto *pipelines =
      new std::unordered_map<uint64_t, std::unique_ptr<ResidentPipeline>>();
  return *pipelines;
}

uint64_t next_resident_pipeline_handle() {
  static uint64_t next = 1;
  return next++;
}

} // namespace

extern "C" int llm_linear_direct_bridge_probe() {
  try {
    probe_hip_owned_import_once();
    set_last_error(
        "direct bridge probe succeeded with HIP VMem fd export and XRT BO "
        "import");
    return 0;
  } catch (const std::exception &exc) {
    set_last_error(exc.what());
    return 1;
  }
}

extern "C" int llm_linear_direct_bridge_probe_report(char *buffer,
                                                     uint64_t capacity) {
  try {
    std::string report = build_direct_probe_report_json();
    copy_cstr(buffer, static_cast<size_t>(capacity), report);
    const bool ok =
        report.find("\"direct_supported\":true") != std::string::npos;
    set_last_error(ok ? "direct bridge probe report selected a direct path"
                      : "direct bridge probe report found no direct path");
    return ok ? 0 : 1;
  } catch (const std::exception &exc) {
    std::string message = exc.what();
    set_last_error(message);
    std::string report = probe_report_json(
        {}, "", "direct bridge probe report failed: " + message);
    copy_cstr(buffer, static_cast<size_t>(capacity), report);
    return 1;
  }
}

extern "C" int llm_linear_direct_bridge_last_error(char *buffer,
                                                   uint64_t capacity) {
  copy_cstr(buffer, static_cast<size_t>(capacity), g_last_error);
  return 0;
}

extern "C" int llm_linear_direct_bridge_cleanup() {
  try {
    cleanup_hip_vmem_pool();
    set_last_error("ok");
    return 0;
  } catch (const std::exception &exc) {
    set_last_error(exc.what());
    return 1;
  }
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
    ScopedUnsetEnv scoped_pythonpath("PYTHONPATH");
    ScopedUnsetEnv scoped_ld_library_path("LD_LIBRARY_PATH");
    check_hip(hipSetDevice(0), "hipSetDevice(0)");
    result->bo_flag = 0;
    result->import_method = kImportXrtBoFromHipVmemFd;
    copy_cstr(result->mechanism, sizeof(result->mechanism), kHipVmemMechanism);

    if (config->direction == kDirectionGpuPrefillNpuDecode)
      run_gpu_to_npu(*config, *result);
    else
      run_npu_to_gpu(*config, *result);

    copy_cstr(result->diagnostic, sizeof(result->diagnostic), "ok");
    set_last_error("ok");
    return 0;
  } catch (const std::exception &exc) {
    set_last_error(exc.what());
    copy_cstr(result->diagnostic, sizeof(result->diagnostic), exc.what());
    return 1;
  }
}

extern "C" int
llm_linear_direct_bridge_run_gpu_stage(const LlmLinearGpuStageRunConfig *config,
                                       LlmLinearGpuStageRunResult *result) {
  if (!config || !result) {
    set_last_error("null GPU stage bridge run config/result");
    return 1;
  }
  std::memset(result, 0, sizeof(*result));
  result->abi_version = kAbiVersion;
  try {
    validate_gpu_stage_config(*config);
    ScopedUnsetEnv scoped_pythonpath("PYTHONPATH");
    ScopedUnsetEnv scoped_ld_library_path("LD_LIBRARY_PATH");
    check_hip(hipSetDevice(0), "hipSetDevice(0)");
    const size_t es = elem_size(config->dtype);
    const bool is_prefill = config->stage == kGpuStagePrefill;
    const DecodeLayout decode_layout = make_decode_layout(
        config->h, config->n, config->dtype, config->decode_storage,
        config->decode_block_size, config->decode_quant_axis);
    const bool int4_decode = !is_prefill && decode_layout.int4;
    const size_t input_bytes =
        (is_prefill ? config->m * config->k : config->h) * es;
    const size_t weight_bytes =
        is_prefill ? config->k * config->h * es
                   : (int4_decode ? decode_layout.packed_bytes()
                                  : config->h * config->n * es);
    const size_t scale_bytes = int4_decode ? decode_layout.scale_bytes() : 0;
    const size_t output_bytes =
        (is_prefill ? config->m * config->h : config->n) * es;

    HipDeviceBuffer input(input_bytes);
    HipDeviceBuffer weights(weight_bytes);
    auto scales =
        int4_decode ? std::make_unique<HipDeviceBuffer>(scale_bytes) : nullptr;
    HipDeviceBuffer output(output_bytes);
    input.hip_write_host(config->input);
    weights.hip_write_host(config->weights);
    if (int4_decode)
      scales->hip_write_host(config->scales);

    const uint64_t stage_start = now_us();
    if (is_prefill)
      run_gpu_prefill_stage(*config, input, weights, output);
    else if (int4_decode)
      run_gpu_decode_int4_stage(*config, decode_layout, input, weights, *scales,
                                output);
    else
      run_gpu_decode_stage(*config, input, weights, output);
    result->stage_us = now_us() - stage_start;
    output.hip_read_host(config->output);

    copy_cstr(result->diagnostic, sizeof(result->diagnostic), "ok");
    set_last_error("ok");
    return 0;
  } catch (const std::exception &exc) {
    set_last_error(exc.what());
    copy_cstr(result->diagnostic, sizeof(result->diagnostic), exc.what());
    return 1;
  }
}

extern "C" int llm_linear_direct_bridge_prepare_resident_pipeline(
    const LlmLinearResidentPipelinePrepareConfig *config,
    LlmLinearResidentPipelinePrepareResult *result) {
  if (!config || !result) {
    set_last_error("null resident pipeline prepare config/result");
    return 1;
  }
  std::memset(result, 0, sizeof(*result));
  result->abi_version = kAbiVersion;
  try {
    validate_resident_prepare_config(*config);
    ScopedUnsetEnv scoped_pythonpath("PYTHONPATH");
    ScopedUnsetEnv scoped_ld_library_path("LD_LIBRARY_PATH");
    check_hip(hipSetDevice(0), "hipSetDevice(0)");
    std::unique_ptr<ResidentPipeline> pipeline;
    if (config->backend == kResidentBackendGpu)
      pipeline = std::make_unique<GpuResidentPipeline>(*config);
    else
      pipeline = std::make_unique<NpuResidentPipeline>(*config);

    const uint64_t handle = next_resident_pipeline_handle();
    result->handle = handle;
    result->static_upload_bytes = pipeline->static_upload_bytes();
    copy_cstr(result->storage_kind, sizeof(result->storage_kind),
              pipeline->storage_kind());
    {
      std::lock_guard<std::mutex> lock(resident_pipeline_mutex());
      resident_pipelines()[handle] = std::move(pipeline);
    }
    copy_cstr(result->diagnostic, sizeof(result->diagnostic), "ok");
    set_last_error("ok");
    return 0;
  } catch (const std::exception &exc) {
    set_last_error(exc.what());
    copy_cstr(result->diagnostic, sizeof(result->diagnostic), exc.what());
    return 1;
  }
}

extern "C" int llm_linear_direct_bridge_run_resident_pipeline(
    const LlmLinearResidentPipelineRunConfig *config,
    LlmLinearResidentPipelineRunResult *result) {
  if (!config || !result) {
    set_last_error("null resident pipeline run config/result");
    return 1;
  }
  std::memset(result, 0, sizeof(*result));
  result->abi_version = kAbiVersion;
  try {
    if (config->abi_version != kAbiVersion)
      throw std::runtime_error("unsupported resident pipeline run ABI version");
    ResidentPipeline *pipeline = nullptr;
    {
      std::lock_guard<std::mutex> lock(resident_pipeline_mutex());
      auto it = resident_pipelines().find(config->handle);
      if (it == resident_pipelines().end())
        throw std::runtime_error("unknown resident pipeline handle");
      pipeline = it->second.get();
    }
    pipeline->run(*config, *result);
    copy_cstr(result->diagnostic, sizeof(result->diagnostic), "ok");
    set_last_error("ok");
    return 0;
  } catch (const std::exception &exc) {
    set_last_error(exc.what());
    copy_cstr(result->diagnostic, sizeof(result->diagnostic), exc.what());
    return 1;
  }
}

extern "C" int
llm_linear_direct_bridge_release_resident_pipeline(uint64_t handle) {
  try {
    std::lock_guard<std::mutex> lock(resident_pipeline_mutex());
    auto it = resident_pipelines().find(handle);
    if (it != resident_pipelines().end())
      resident_pipelines().erase(it);
    set_last_error("ok");
    return 0;
  } catch (const std::exception &exc) {
    set_last_error(exc.what());
    return 1;
  }
}
