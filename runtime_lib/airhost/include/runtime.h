// SPDX-License-Identifier: MIT
// Copyright (C) 2023, Advanced Micro Devices, Inc.

#ifndef RUNTIME_H_
#define RUNTIME_H_

#include <vector>

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

namespace air {
namespace rocm {

constexpr hsa_device_type_t kHsaDeviceTypeAie =
#ifdef HSA_DEVICE_TYPE_AIE
    HSA_DEVICE_TYPE_AIE;
#else
    static_cast<hsa_device_type_t>(3);
#endif

inline hsa_status_t signalCreateOnAgent(hsa_signal_value_t initialValue,
                                        hsa_agent_t agent,
                                        hsa_signal_t *signal) {
  return hsa_amd_signal_create(initialValue, 1, &agent, 0, signal);
}

class Runtime {
public:
  Runtime() = default;
  static void Init();
  static void ShutDown();

  void *AllocateMemory(size_t size);
  void FreeMemory(void *ptr);

  static Runtime *runtime_;

private:
  static hsa_status_t IterateAgents(hsa_agent_t agent, void *data);
  static hsa_status_t IterateMemPool(hsa_amd_memory_pool_t pool, void *data);
  void FindAieAgents();
  void InitMemSegments();

  hsa_amd_memory_pool_t global_mem_pool_;
  std::vector<hsa_agent_t> aie_agents_;
};

} // namespace rocm
} // namespace air

#endif // RUNTIME_H_
