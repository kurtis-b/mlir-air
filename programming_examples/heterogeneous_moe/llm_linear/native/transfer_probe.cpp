// SPDX-License-Identifier: MIT
//
// Probe possible Ryzen iGPU/NPU data-sharing paths. Run one method per process
// so unsupported XRT/HIP interop paths can abort without stopping the matrix.

#include <hip/hip_runtime_api.h>

#include <xrt/experimental/xrt_hw_context.h>
#include <xrt/experimental/xrt_xclbin.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr size_t kBytes = 4096;
void check_hip(hipError_t err, const char *what) {
  if (err == hipSuccess)
    return;
  std::ostringstream os;
  os << what << ": " << hipGetErrorName(err) << " (" << hipGetErrorString(err)
     << ")";
  throw std::runtime_error(os.str());
}

size_t align_up(size_t value, size_t alignment) {
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

void verify_bytes(const void *ptr, uint8_t value, size_t count) {
  const auto *bytes = static_cast<const uint8_t *>(ptr);
  for (size_t i = 0; i < count; ++i) {
    if (bytes[i] != value) {
      std::ostringstream os;
      os << "byte mismatch at " << i << ": got " << static_cast<int>(bytes[i])
         << ", expected " << static_cast<int>(value);
      throw std::runtime_error(os.str());
    }
  }
}

uint8_t iteration_pattern(size_t iteration, uint8_t base) {
  return static_cast<uint8_t>(base + (iteration % 97));
}

void gpu_fill(void *device_ptr, uint8_t value) {
  check_hip(hipMemset(device_ptr, value, kBytes), "hipMemset");
  check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize");
}

struct BoMode {
  xrt::bo::flags flag;
  const char *name;
};

const BoMode kBoModes[] = {
    {xrt::bo::flags::normal, "normal"},
    {xrt::bo::flags::cacheable, "cacheable"},
    {xrt::bo::flags::host_only, "host_only"},
    {xrt::bo::flags::device_only, "device_only"},
    {xrt::bo::flags::p2p, "p2p"},
    {xrt::bo::flags::svm, "svm"},
    {xrt::bo::flags::carveout, "carveout"},
};

void hip_device_baseline() {
  check_hip(hipSetDevice(0), "hipSetDevice");
  void *device = nullptr;
  check_hip(hipMalloc(&device, kBytes), "hipMalloc");
  gpu_fill(device, 0x11);
  std::vector<uint8_t> host(kBytes);
  check_hip(hipMemcpy(host.data(), device, kBytes, hipMemcpyDeviceToHost),
            "hipMemcpy D2H");
  verify_bytes(host.data(), 0x11, kBytes);
}

void xrt_bo_baseline() {
  xrt::device device(0);
  xrt::bo bo(device, kBytes, xrt::bo::flags::host_only, 0);
  std::vector<uint8_t> host(kBytes, 0x22);
  bo.write(host.data(), kBytes, 0);
  bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, kBytes, 0);
  std::fill(host.begin(), host.end(), 0x00);
  bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, kBytes, 0);
  bo.read(host.data(), kBytes, 0);
  verify_bytes(host.data(), 0x22, kBytes);
}

void hip_host_malloc_xrt_userptr() {
  check_hip(hipSetDevice(0), "hipSetDevice");
  void *host = nullptr;
  check_hip(hipHostMalloc(&host, kBytes, hipHostMallocMapped),
            "hipHostMallocMapped");
  void *device_ptr = nullptr;
  check_hip(hipHostGetDevicePointer(&device_ptr, host, 0),
            "hipHostGetDevicePointer");
  gpu_fill(device_ptr, 0x33);
  verify_bytes(host, 0x33, kBytes);
  xrt::device device(0);
  xrt::bo user_bo(device, host, kBytes, 0);
  user_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, kBytes, 0);
}

void xrt_mapped_bo_hip_register(xrt::bo::flags flag) {
  check_hip(hipSetDevice(0), "hipSetDevice");
  xrt::device device(0);
  xrt::bo bo(device, kBytes, flag, 0);
  auto *host = bo.map<void *>();
  check_hip(hipHostRegister(host, kBytes, hipHostRegisterMapped),
            "hipHostRegister(XRT BO map)");
  void *device_ptr = nullptr;
  check_hip(hipHostGetDevicePointer(&device_ptr, host, 0),
            "hipHostGetDevicePointer(XRT BO map)");
  gpu_fill(device_ptr, 0x44);
  verify_bytes(host, 0x44, kBytes);
  bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, kBytes, 0);
}

void xrt_export_to_hip_vmem(xrt::bo::flags flag) {
  check_hip(hipSetDevice(0), "hipSetDevice");
  auto *device = new xrt::device(0);
  auto *bo =
      new xrt::bo(*device, align_up(kBytes, hip_vmem_granularity()), flag, 0);
  int exported_fd = static_cast<int>(bo->export_buffer());
  int import_fd = dup(exported_fd);
  if (import_fd < 0)
    throw std::runtime_error("dup export fd failed");
  hipMemGenericAllocationHandle_t handle{};
  check_hip(hipMemImportFromShareableHandle(
                &handle,
                reinterpret_cast<void *>(static_cast<intptr_t>(import_fd)),
                hipMemHandleTypePosixFileDescriptor),
            "hipMemImportFromShareableHandle(XRT fd)");
  close(import_fd);
  hipDeviceptr_t va = 0;
  size_t size = align_up(kBytes, hip_vmem_granularity());
  check_hip(hipMemAddressReserve(&va, size, hip_vmem_granularity(), 0, 0),
            "hipMemAddressReserve");
  check_hip(hipMemMap(va, size, 0, handle, 0), "hipMemMap");
  hipMemAccessDesc access{};
  access.location.type = hipMemLocationTypeDevice;
  access.location.id = 0;
  access.flags = hipMemAccessFlagsProtReadWrite;
  check_hip(hipMemSetAccess(va, size, &access, 1), "hipMemSetAccess");
  gpu_fill(reinterpret_cast<void *>(va), 0x55);
}

void xrt_export_to_hip_external(xrt::bo::flags flag) {
  check_hip(hipSetDevice(0), "hipSetDevice");
  auto *device = new xrt::device(0);
  auto *bo = new xrt::bo(*device, kBytes, flag, 0);
  int exported_fd = static_cast<int>(bo->export_buffer());
  int import_fd = dup(exported_fd);
  if (import_fd < 0)
    throw std::runtime_error("dup export fd failed");
  hipExternalMemoryHandleDesc handle_desc{};
  handle_desc.type = hipExternalMemoryHandleTypeOpaqueFd;
  handle_desc.handle.fd = import_fd;
  handle_desc.size = kBytes;
  hipExternalMemory_t external = nullptr;
  check_hip(hipImportExternalMemory(&external, &handle_desc),
            "hipImportExternalMemory(XRT fd)");
  hipExternalMemoryBufferDesc buffer_desc{};
  buffer_desc.size = kBytes;
  void *device_ptr = nullptr;
  check_hip(
      hipExternalMemoryGetMappedBuffer(&device_ptr, external, &buffer_desc),
      "hipExternalMemoryGetMappedBuffer");
  gpu_fill(device_ptr, 0x66);
}

void hip_vmem_export_to_xrt_import(bool with_pid) {
  check_hip(hipSetDevice(0), "hipSetDevice");
  size_t size = align_up(kBytes, hip_vmem_granularity());
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = 0;
  prop.requestedHandleType = hipMemHandleTypePosixFileDescriptor;
  hipMemGenericAllocationHandle_t handle{};
  check_hip(hipMemCreate(&handle, size, &prop, 0), "hipMemCreate");
  int fd = -1;
  check_hip(hipMemExportToShareableHandle(
                &fd, handle, hipMemHandleTypePosixFileDescriptor, 0),
            "hipMemExportToShareableHandle");
  auto *device = new xrt::device(0);
  if (with_pid)
    (void)new xrt::bo(*device, xrt::pid_type{getpid()},
                      static_cast<xrt::bo::export_handle>(fd));
  else
    (void)new xrt::bo(*device, static_cast<xrt::bo::export_handle>(fd));
}

struct HipExportedAllocation {
  hipDeviceptr_t va = 0;
  hipMemGenericAllocationHandle_t handle{};
  int fd = -1;
  size_t size = 0;
};

HipExportedAllocation make_exported_hip_vmem() {
  check_hip(hipSetDevice(0), "hipSetDevice");
  HipExportedAllocation allocation;
  allocation.size = align_up(kBytes, hip_vmem_granularity());
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = 0;
  prop.requestedHandleType = hipMemHandleTypePosixFileDescriptor;
  check_hip(hipMemCreate(&allocation.handle, allocation.size, &prop, 0),
            "hipMemCreate");
  check_hip(hipMemAddressReserve(&allocation.va, allocation.size,
                                 hip_vmem_granularity(), 0, 0),
            "hipMemAddressReserve");
  check_hip(hipMemMap(allocation.va, allocation.size, 0, allocation.handle, 0),
            "hipMemMap");
  hipMemAccessDesc access{};
  access.location.type = hipMemLocationTypeDevice;
  access.location.id = 0;
  access.flags = hipMemAccessFlagsProtReadWrite;
  check_hip(hipMemSetAccess(allocation.va, allocation.size, &access, 1),
            "hipMemSetAccess");
  check_hip(hipMemExportToShareableHandle(&allocation.fd, allocation.handle,
                                          hipMemHandleTypePosixFileDescriptor,
                                          0),
            "hipMemExportToShareableHandle");
  return allocation;
}

void hip_vmem_gpu_write_xrt_read_iterations(size_t iterations) {
  auto allocation = make_exported_hip_vmem();
  auto *device = new xrt::device(0);
  auto *bo =
      new xrt::bo(*device, static_cast<xrt::bo::export_handle>(allocation.fd));
  std::vector<uint8_t> host(kBytes, 0);
  for (size_t i = 0; i < iterations; ++i) {
    uint8_t value = iteration_pattern(i, 0x30);
    gpu_fill(reinterpret_cast<void *>(allocation.va), value);
    bo->sync(XCL_BO_SYNC_BO_TO_DEVICE, kBytes, 0);
    bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE, kBytes, 0);
    bo->read(host.data(), kBytes, 0);
    verify_bytes(host.data(), value, kBytes);
  }
}

void hip_vmem_gpu_write_xrt_read() {
  hip_vmem_gpu_write_xrt_read_iterations(1);
}

void hip_vmem_xrt_write_gpu_read_iterations(size_t iterations) {
  auto allocation = make_exported_hip_vmem();
  auto *device = new xrt::device(0);
  auto *bo =
      new xrt::bo(*device, static_cast<xrt::bo::export_handle>(allocation.fd));
  std::vector<uint8_t> host(kBytes, 0);
  for (size_t i = 0; i < iterations; ++i) {
    uint8_t value = iteration_pattern(i, 0x60);
    std::fill(host.begin(), host.end(), value);
    bo->write(host.data(), kBytes, 0);
    bo->sync(XCL_BO_SYNC_BO_TO_DEVICE, kBytes, 0);
    std::fill(host.begin(), host.end(), 0);
    check_hip(hipMemcpy(host.data(), reinterpret_cast<void *>(allocation.va),
                        kBytes, hipMemcpyDeviceToHost),
              "hipMemcpy imported HIP VMem to host");
    verify_bytes(host.data(), value, kBytes);
  }
}

void hip_vmem_xrt_write_gpu_read() {
  hip_vmem_xrt_write_gpu_read_iterations(1);
}

std::vector<uint32_t> read_instructions(const char *path) {
  std::ifstream file(path, std::ios::binary);
  if (!file)
    throw std::runtime_error(std::string("failed to open insts: ") + path);
  std::vector<char> bytes((std::istreambuf_iterator<char>(file)),
                          std::istreambuf_iterator<char>());
  if (bytes.empty() || (bytes.size() % sizeof(uint32_t)) != 0)
    throw std::runtime_error("instruction file is empty or not uint32 aligned");
  std::vector<uint32_t> words(bytes.size() / sizeof(uint32_t));
  std::memcpy(words.data(), bytes.data(), bytes.size());
  return words;
}

void hip_vmem_imported_xrt_npu_vecadd_iterations(size_t iterations) {
  const char *xclbin_path = std::getenv("TRANSFER_PROBE_VECADD_XCLBIN");
  const char *insts_path = std::getenv("TRANSFER_PROBE_VECADD_INSTS");
  if (!xclbin_path || !insts_path)
    throw std::runtime_error(
        "set TRANSFER_PROBE_VECADD_XCLBIN and TRANSFER_PROBE_VECADD_INSTS");

  constexpr size_t elements = 1024;
  constexpr size_t bytes = elements * sizeof(uint16_t);
  auto in0_alloc = make_exported_hip_vmem();
  auto in1_alloc = make_exported_hip_vmem();
  auto out_alloc = make_exported_hip_vmem();

  std::vector<uint16_t> in0(elements, 0x3f80); // bf16 1.0
  std::vector<uint16_t> in1(elements, 0x4000); // bf16 2.0
  std::vector<uint16_t> out(elements, 0);
  xrt::device device(0);
  xrt::xclbin xclbin{std::string(xclbin_path)};
  auto uuid = device.register_xclbin(xclbin);
  xrt::hw_context context(device, uuid);
  std::string kernel_name;
  for (const auto &candidate : xclbin.get_kernels()) {
    std::string name = candidate.get_name();
    if (name.find("MLIR_AIE") != std::string::npos) {
      kernel_name = name;
      break;
    }
  }
  if (kernel_name.empty())
    throw std::runtime_error("MLIR_AIE kernel not found in vecadd xclbin");
  xrt::kernel kernel(context, kernel_name);
  std::vector<uint32_t> insts = read_instructions(insts_path);
  xrt::bo bo_instr(device, insts.size() * sizeof(uint32_t),
                   xrt::bo::flags::cacheable, kernel.group_id(1));
  bo_instr.write(insts.data(), insts.size() * sizeof(uint32_t), 0);
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto *in0_bo =
      new xrt::bo(device, static_cast<xrt::bo::export_handle>(in0_alloc.fd));
  auto *in1_bo =
      new xrt::bo(device, static_cast<xrt::bo::export_handle>(in1_alloc.fd));
  auto *out_bo =
      new xrt::bo(device, static_cast<xrt::bo::export_handle>(out_alloc.fd));

  unsigned int opcode = 3;
  unsigned int inst_count = static_cast<unsigned int>(insts.size());

  for (size_t iter = 0; iter < iterations; ++iter) {
    std::fill(out.begin(), out.end(), 0);
    check_hip(hipMemcpy(reinterpret_cast<void *>(in0_alloc.va), in0.data(),
                        bytes, hipMemcpyHostToDevice),
              "hipMemcpy input0 H2D");
    check_hip(hipMemcpy(reinterpret_cast<void *>(in1_alloc.va), in1.data(),
                        bytes, hipMemcpyHostToDevice),
              "hipMemcpy input1 H2D");
    check_hip(hipMemcpy(reinterpret_cast<void *>(out_alloc.va), out.data(),
                        bytes, hipMemcpyHostToDevice),
              "hipMemcpy output H2D");
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize(vecadd inputs)");
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    in0_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE, bytes, 0);
    in1_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE, bytes, 0);
    out_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE, bytes, 0);

    auto run = kernel(opcode, bo_instr, inst_count, *in0_bo, *in1_bo, *out_bo);
    run.wait();

    in0_bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE, bytes, 0);
    in1_bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE, bytes, 0);
    out_bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE, bytes, 0);
    check_hip(hipMemcpy(out.data(), reinterpret_cast<void *>(out_alloc.va),
                        bytes, hipMemcpyDeviceToHost),
              "hipMemcpy output D2H");
    for (size_t i = 0; i < elements; ++i) {
      if (out[i] != 0x4040) {
        std::ostringstream os;
        os << "vecadd output mismatch at iteration " << iter << ", element "
           << i << ": got 0x" << std::hex << out[i] << ", expected 0x4040";
        throw std::runtime_error(os.str());
      }
    }
  }
}

void hip_vmem_imported_xrt_npu_vecadd() {
  hip_vmem_imported_xrt_npu_vecadd_iterations(1);
}

void xrt_host_bo_npu_vecadd_iterations(size_t iterations) {
  const char *xclbin_path = std::getenv("TRANSFER_PROBE_VECADD_XCLBIN");
  const char *insts_path = std::getenv("TRANSFER_PROBE_VECADD_INSTS");
  if (!xclbin_path || !insts_path)
    throw std::runtime_error(
        "set TRANSFER_PROBE_VECADD_XCLBIN and TRANSFER_PROBE_VECADD_INSTS");

  constexpr size_t elements = 1024;
  constexpr size_t bytes = elements * sizeof(uint16_t);
  std::vector<uint16_t> in0(elements, 0x3f80); // bf16 1.0
  std::vector<uint16_t> in1(elements, 0x4000); // bf16 2.0
  std::vector<uint16_t> out(elements, 0);

  xrt::device device(0);
  xrt::xclbin xclbin{std::string(xclbin_path)};
  auto uuid = device.register_xclbin(xclbin);
  xrt::hw_context context(device, uuid);
  std::string kernel_name;
  for (const auto &candidate : xclbin.get_kernels()) {
    std::string name = candidate.get_name();
    if (name.find("MLIR_AIE") != std::string::npos) {
      kernel_name = name;
      break;
    }
  }
  if (kernel_name.empty())
    throw std::runtime_error("MLIR_AIE kernel not found in vecadd xclbin");
  xrt::kernel kernel(context, kernel_name);
  std::vector<uint32_t> insts = read_instructions(insts_path);
  xrt::bo bo_instr(device, insts.size() * sizeof(uint32_t),
                   xrt::bo::flags::cacheable, kernel.group_id(1));
  bo_instr.write(insts.data(), insts.size() * sizeof(uint32_t), 0);

  xrt::bo in0_bo(device, bytes, xrt::bo::flags::host_only, kernel.group_id(3));
  xrt::bo in1_bo(device, bytes, xrt::bo::flags::host_only, kernel.group_id(4));
  xrt::bo out_bo(device, bytes, xrt::bo::flags::host_only, kernel.group_id(5));

  unsigned int opcode = 3;
  unsigned int inst_count = static_cast<unsigned int>(insts.size());
  for (size_t iter = 0; iter < iterations; ++iter) {
    std::fill(out.begin(), out.end(), 0);
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    in0_bo.write(in0.data(), bytes, 0);
    in1_bo.write(in1.data(), bytes, 0);
    out_bo.write(out.data(), bytes, 0);
    in0_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, bytes, 0);
    in1_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, bytes, 0);
    out_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, bytes, 0);

    auto run = kernel(opcode, bo_instr, inst_count, in0_bo, in1_bo, out_bo);
    run.wait();

    out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, bytes, 0);
    out_bo.read(out.data(), bytes, 0);
    for (size_t i = 0; i < elements; ++i) {
      if (out[i] != 0x4040) {
        std::ostringstream os;
        os << "host BO vecadd output mismatch at iteration " << iter
           << ", element " << i << ": got 0x" << std::hex << out[i]
           << ", expected 0x4040";
        throw std::runtime_error(os.str());
      }
    }
  }
}

void add_stress_methods(
    std::vector<std::pair<std::string, std::function<void()>>> &methods) {
  constexpr size_t kIterations[] = {1, 10, 100, 1000};
  for (size_t iterations : kIterations) {
    methods.push_back(
        {"xrt_host_bo_npu_vecadd_stress_" + std::to_string(iterations),
         [iterations] { xrt_host_bo_npu_vecadd_iterations(iterations); }});
    methods.push_back(
        {"hip_vmem_gpu_write_xrt_read_stress_" + std::to_string(iterations),
         [iterations] { hip_vmem_gpu_write_xrt_read_iterations(iterations); }});
    methods.push_back(
        {"hip_vmem_xrt_write_gpu_read_stress_" + std::to_string(iterations),
         [iterations] { hip_vmem_xrt_write_gpu_read_iterations(iterations); }});
    methods.push_back({"hip_vmem_imported_xrt_npu_vecadd_stress_" +
                           std::to_string(iterations),
                       [iterations] {
                         hip_vmem_imported_xrt_npu_vecadd_iterations(
                             iterations);
                       }});
  }
}

} // namespace

int main(int argc, char **argv) {
  std::vector<std::pair<std::string, std::function<void()>>> methods = {
      {"hip_device_baseline", hip_device_baseline},
      {"xrt_bo_host_staged_baseline", xrt_bo_baseline},
      {"hip_host_malloc_mapped_to_xrt_userptr", hip_host_malloc_xrt_userptr},
      {"xrt_host_only_bo_map_registered_to_hip",
       [] { xrt_mapped_bo_hip_register(xrt::bo::flags::host_only); }},
      {"xrt_normal_bo_map_registered_to_hip",
       [] { xrt_mapped_bo_hip_register(xrt::bo::flags::normal); }},
      {"hip_vmem_export_to_xrt_import",
       [] { hip_vmem_export_to_xrt_import(false); }},
      {"hip_vmem_export_to_xrt_import_pid",
       [] { hip_vmem_export_to_xrt_import(true); }},
      {"hip_vmem_gpu_write_xrt_read", hip_vmem_gpu_write_xrt_read},
      {"hip_vmem_xrt_write_gpu_read", hip_vmem_xrt_write_gpu_read},
      {"hip_vmem_imported_xrt_npu_vecadd", hip_vmem_imported_xrt_npu_vecadd},
  };
  add_stress_methods(methods);
  for (const BoMode &mode : kBoModes) {
    methods.push_back({std::string("xrt_export_to_hip_vmem_") + mode.name,
                       [mode] { xrt_export_to_hip_vmem(mode.flag); }});
  }
  for (const BoMode &mode : kBoModes) {
    methods.push_back({std::string("xrt_export_to_hip_external_") + mode.name,
                       [mode] { xrt_export_to_hip_external(mode.flag); }});
  }

  if (argc == 1 || std::string_view(argv[1]) == "--list") {
    for (const auto &method : methods)
      std::cout << method.first << "\n";
    return 0;
  }

  std::string_view requested(argv[1]);
  for (const auto &method : methods) {
    if (method.first != requested)
      continue;
    try {
      method.second();
      std::cout << "PASS\tok\n";
      std::cout.flush();
      _exit(0);
    } catch (const std::exception &exc) {
      std::cout << "FAIL\t" << exc.what() << "\n";
      std::cout.flush();
      _exit(2);
    }
  }
  std::cout << "FAIL\tunknown method: " << requested << "\n";
  std::cout.flush();
  _exit(2);
}
