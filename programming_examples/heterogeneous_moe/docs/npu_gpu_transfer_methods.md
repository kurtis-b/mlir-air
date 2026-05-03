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
```

The NPU kernel smoke uses the vecadd artifact built by the XRT test suite:

```bash
source /opt/xilinx/xrt/setup.sh
TRANSFER_PROBE_VECADD_XCLBIN=/home/cj/mlir-air/build-xrt/test/xrt/40_triton_vec_add/test_npu2_peano/air.xclbin \
TRANSFER_PROBE_VECADD_INSTS=/home/cj/mlir-air/build-xrt/test/xrt/40_triton_vec_add/test_npu2_peano/air.insts.bin \
  timeout 20s /tmp/npu_gpu_transfer_probe hip_vmem_imported_xrt_npu_vecadd
```

## Results

| Method family | Result | Interpretation |
| --- | --- | --- |
| HIP device allocation baseline | Passed | HIP device execution and memcpy are healthy when linked to ROCm 7.2. |
| XRT host-staged BO baseline | Passed | Current host-staged path remains a valid fallback and baseline. |
| HIP VMem allocation, fd export, XRT BO import | Passed | Best direct-handoff candidate. XRT can import a HIP-owned POSIX fd as a BO. |
| HIP VMem GPU write, XRT BO read | Passed in isolated runs; later repeated run hung | GPU-produced bytes were visible through the imported XRT BO, but process-level lifecycle is not stable enough yet. |
| XRT BO write, HIP VMem GPU read | Passed in isolated runs; later repeated runs were unstable | XRT-produced bytes were visible through the HIP virtual address, but this needs persistent runtime ownership before acceptance. |
| HIP VMem imported as XRT BOs for NPU vecadd | Passed once in a fresh run; later repeated runs were unstable | Proof that an NPU kernel can consume and produce HIP-owned imported BOs, but lifecycle handling needs production cleanup. |
| XRT `host_only` BO map plus `hipHostRegister` | Passed in isolation; later repeated matrix runs timed out | Usable as a shared host-memory path, not device-resident handoff. |
| XRT normal BO map plus `hipHostRegister` | Failed | XRT normal BO maps were not registerable by HIP on this stack. |
| HIP mapped host allocation passed to XRT userptr BO | Failed or signaled | Not a reliable bridge path. |
| XRT BO export into HIP VMem import | Failed, timed out, or signaled for all tested BO flags | Current XRT-owned BO direct bridge direction is not viable on this stack. |
| XRT BO export into HIP external memory import | Failed, timed out, or signaled for all tested BO flags | HIP external memory import does not rescue the XRT-owned direction. |

The tested XRT BO flags for export into HIP were `normal`, `cacheable`,
`host_only`, `device_only`, `p2p`, `svm`, and `carveout`. None produced a
stable importable HIP device mapping on the current machine.

The pid-taking XRT import constructor for HIP VMem fd import was less stable
than the non-pid `xrt::bo(device, export_handle)` constructor. The non-pid form
is the preferred path for the next implementation attempt.

Fork-based isolation is not valid for classifying HIP behavior. HIP baseline
cases crashed when run in a forked child after runtime initialization. The probe
therefore runs one method per process.

## Implementation Implication

The direct bridge should pivot from XRT-owned allocations to HIP-owned VMem
handoff allocations:

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

The next accepted Milestone 2 implementation should use a persistent allocation
pool and persistent XRT hardware context. The repeated-run instability seen in
the probe is consistent with per-process/per-method HIP VMem and XRT object
teardown being too fragile for the current driver stack.
