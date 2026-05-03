# NPU/GPU Transfer Method Probe

Date: May 3, 2026

Target stack used for this probe:

- Ryzen AI Strix NPU through XRT 2.23.0 / amdxdna 2.23.0.
- Radeon 890M `gfx1150` iGPU through HIP/ROCm 7.2.
- XRT device `0` and HIP device `0`.

The goal was to determine which data-transfer or shared-allocation mechanisms
can plausibly support direct GPU/NPU handoff for `llm_linear`.

## Probe

Build the probe from the repository root:

```bash
programming_examples/heterogeneous_moe/llm_linear/native/build_transfer_probe.sh /tmp/npu_gpu_transfer_probe
ldd /tmp/npu_gpu_transfer_probe | rg 'amdhip|xrt'
```

The explicit ROCm library path matters. When the probe was accidentally linked
against the system `libamdhip64.so.5` instead of the ROCm 7.2 HIP runtime,
otherwise valid HIP baseline cases crashed during process teardown.

List available methods:

```bash
/tmp/npu_gpu_transfer_probe --list
```

Run selected methods after XRT setup:

```bash
source /opt/xilinx/xrt/setup.sh
timeout 20s /tmp/npu_gpu_transfer_probe hip_device_baseline
timeout 20s /tmp/npu_gpu_transfer_probe xrt_bo_host_staged_baseline
timeout 20s /tmp/npu_gpu_transfer_probe hip_vmem_export_to_xrt_import
timeout 20s /tmp/npu_gpu_transfer_probe hip_vmem_gpu_write_xrt_read
timeout 20s /tmp/npu_gpu_transfer_probe hip_vmem_xrt_write_gpu_read
timeout 20s /tmp/npu_gpu_transfer_probe hip_vmem_gpu_write_xrt_read_stress_100
timeout 20s /tmp/npu_gpu_transfer_probe hip_vmem_xrt_write_gpu_read_stress_100
```

The NPU kernel smoke uses the vecadd artifact built by the XRT test suite:

```bash
source /opt/xilinx/xrt/setup.sh
TRANSFER_PROBE_VECADD_XCLBIN=/home/cj/mlir-air/build-xrt/test/xrt/40_triton_vec_add/test_npu2_peano/air.xclbin \
TRANSFER_PROBE_VECADD_INSTS=/home/cj/mlir-air/build-xrt/test/xrt/40_triton_vec_add/test_npu2_peano/air.insts.bin \
  timeout 20s /tmp/npu_gpu_transfer_probe hip_vmem_imported_xrt_npu_vecadd
```

Persistent-context stress variants are available for 1, 10, 100, and 1000
iterations:

```bash
timeout 120s /tmp/npu_gpu_transfer_probe hip_vmem_gpu_write_xrt_read_stress_1000
timeout 120s /tmp/npu_gpu_transfer_probe hip_vmem_xrt_write_gpu_read_stress_1000
timeout 60s /tmp/npu_gpu_transfer_probe hip_vmem_imported_xrt_npu_vecadd_stress_1000
timeout 60s /tmp/npu_gpu_transfer_probe xrt_host_bo_npu_vecadd_stress_1000
```

## Results

| Method family | Result | Interpretation |
| --- | --- | --- |
| HIP device allocation baseline | Passed | HIP device execution and memcpy are healthy when linked to ROCm 7.2. |
| XRT host-staged BO baseline | Passed | Current host-staged path remains a valid fallback and baseline. |
| HIP VMem allocation, fd export, XRT BO import | Passed | Best direct-handoff candidate. XRT can import a HIP-owned POSIX fd as a BO. |
| HIP VMem GPU write, XRT BO read | Passed serial 1000-iteration stress | GPU-produced bytes were visible through the imported XRT BO in the persistent-context probe. |
| XRT BO write, HIP VMem GPU read | Passed serial 1000-iteration stress | XRT-produced bytes were visible through the HIP virtual address in the persistent-context probe. |
| XRT host BOs for NPU vecadd | Passed 1000 iterations | Control proving the vecadd xclbin is relaunchable with normal XRT-owned BOs. |
| HIP VMem imported as XRT BOs for NPU vecadd | Passed 1000 iterations after launch/sync fix | An NPU kernel can consume and produce HIP-owned imported BOs in a persistent-context loop. |
| XRT `host_only` BO map plus `hipHostRegister` | Passed in isolation; later repeated matrix runs timed out | Usable as a shared host-memory path, not device-resident handoff. |
| XRT normal BO map plus `hipHostRegister` | Failed | XRT normal BO maps were not registerable by HIP on this stack. |
| HIP mapped host allocation passed to XRT userptr BO | Failed or signaled | Not a reliable bridge path. |
| XRT BO export into HIP VMem import | Failed, timed out, or signaled for all tested BO flags | Current XRT-owned BO direct bridge direction is not viable on this stack. |
| XRT BO export into HIP external memory import | Failed, timed out, or signaled for all tested BO flags | HIP external memory import does not rescue the XRT-owned direction. |

The tested XRT BO flags for export into HIP were `normal`, `cacheable`,
`host_only`, `device_only`, `p2p`, `svm`, and `carveout`. None produced a
stable importable HIP device mapping on the current machine.

## Direct Contract And Platform Requirements

`transfer_mode=direct` means `contract=no_host_copies`: no CPU or NumPy data
materialization may occur between the GPU producer and NPU consumer, or between
the NPU producer and GPU consumer. CPU orchestration, descriptor construction,
and explicit synchronization are allowed. Host-staged transfer remains a
baseline only and must never satisfy direct mode.

The native probe reports a mechanism matrix instead of a single boolean. Each
mechanism records ownership, exported handle type, import view, direct class,
bidirectional visibility, NPU-kernel verification state, sync events,
host-materialization count, diagnostics, `zero_host_copy`, and
`device_resident_buffers`. Current direct results select
`hip_vmem_export_xrt_bo_import_fd` with direct class
`device_resident_zero_host_copy`. Shared host mappings, such as XRT
`host_only` maps registered with HIP, are tracked separately as
`shared_host_zero_copy`; they are useful research evidence but are not the
current device-resident bridge. `numpy_host_staged_baseline` has
`zero_host_copy=false` and `host_materialization_count>0`.

Shared DRAM on a Ryzen package is not sufficient by itself. A valid direct
handoff also requires compatible export/import APIs in both drivers, IOMMU/PASID
mappings that let the GPU and NPU address the same allocation, allocation and
fd lifetime rules that keep both runtime views alive, coherency behavior that
matches the synchronization model, and explicit sync at every producer/consumer
edge. Without those properties, two devices can physically share memory and
still force a host copy, reject an imported handle, or observe stale data.

The pid-taking XRT import constructor for HIP VMem fd import was less stable
than the non-pid `xrt::bo(device, export_handle)` constructor. The non-pid form
is the preferred path for the next implementation attempt.

Fork-based isolation is not valid for classifying HIP behavior. HIP baseline
cases crashed when run in a forked child after runtime initialization. The probe
therefore runs one method per process.

## Implementation Implication

The direct bridge uses HIP-owned VMem handoff allocations:

1. Allocate handoff tensors with `hipMemCreate` using
   `hipMemHandleTypePosixFileDescriptor`.
2. Reserve and map a HIP virtual address with `hipMemAddressReserve`,
   `hipMemMap`, and `hipMemSetAccess`.
3. Export the HIP allocation fd with `hipMemExportToShareableHandle`.
4. Import the fd into XRT with `xrt::bo(device, export_handle)`.
5. Keep the HIP handle, HIP virtual address, fd, and XRT BO alive for the full
   lifetime of every GPU and NPU use.
6. Use explicit synchronization at handoff edges: GPU stream/device completion
   before XRT kernel launch, XRT run completion plus imported BO sync before GPU
   consumption.

This is still low-level runtime work, not a compiler-only fix. MLIR-AIR can
continue to describe device-resident tensors and edge metadata, but the actual
handoff mechanism depends on runtime allocation ownership, fd lifetime,
coherency, and XRT/HIP synchronization.

The LLM-linear native direct bridge keeps the HIP VMem allocation, exported fd,
mapped VA, and imported XRT BO in a process-lifetime allocation pool. NPU
launches use the same `kernel(...)` invocation pattern as the passing XRT tests,
and the runtime syncs imported BOs back from the XRT side after NPU use before
they can be reused by HIP.

The LLM-linear workload path keeps tiny direct runs as regression coverage. The
Milestone 2 hardware gate is accepted as of May 3, 2026 with:
`source /opt/xilinx/xrt/setup.sh && ../../sandbox/bin/python run_llm_linear_milestone2.py`.
It writes evidence under `llm_linear/artifacts/benchmarks/milestone2_e2e` and
covers `medium_m8_k512_h512_n256` in both direct directions plus matching
host-mixed baselines, not the broader `medium` ladder. Full CPU/GPU/NPU crossover
and `llm_like` studies remain future work. Direct result JSONs must report HIP
VMem ownership, XRT BO imported views, POSIX fd export, zero NumPy host
materializations, and explicit sync events.

The G2N bridge no longer uses `xrt::bo::copy` to extract the decode row. It
writes the selected decode row into a row-sized HIP VMem allocation with a HIP
device-to-device row copy, imports that row allocation into XRT, and passes the
row BO to NPU decode. This preserves a direct imported XRT BO handoff while
avoiding the XRT warning path:
`Reverting to host copy of buffers (exec_buf: Operation not supported)`.
