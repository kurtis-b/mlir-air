# LLM-Linear Direct Bridge

This directory contains native C++ helpers for investigating
`transfer_mode=direct`.

`direct_bridge.cpp` is the runtime bridge currently called by the Python
LLM-linear harness. It owns shared handoff buffers as HIP VMem allocations,
exports POSIX fds, imports those fds into XRT as `xrt::bo` views, and launches
the no-host-staging GPU shared libraries plus XRT NPU kernels. The bridge keeps
the HIP handle, fd, VA, and imported BO in a process-lifetime allocation pool.

Build:

```bash
llm_linear/native/build_direct_bridge.sh /tmp/libllm_linear_direct_bridge.so
```

Probe from `programming_examples/heterogeneous_moe`:

```bash
source /opt/xilinx/xrt/setup.sh
LLM_LINEAR_DIRECT_BRIDGE_SO=/tmp/libllm_linear_direct_bridge.so \
  ../../sandbox/bin/python -c "from llm_linear.direct_bridge import probe_direct_bridge; print(probe_direct_bridge())"
```

The bridge probe exercises the lightweight HIP-owned import path and fails
closed if the native library cannot allocate HIP VMem, export a POSIX fd, import
it as an XRT BO, and prove bidirectional visibility through HIP and XRT views.
Python consumes the structured probe report from
`llm_linear_direct_bridge_probe_report`, including the selected mechanism,
direct class, handle type, import view, zero-host-copy status, host
materialization count, and diagnostics. The workload result marks NPU-kernel
verification after the direct run records the XRT kernel wait. Use
`transfer_probe.cpp` for the stronger looped visibility and vecadd stress checks
through both runtime views.

Transfer-method probe:

```bash
llm_linear/native/build_transfer_probe.sh /tmp/npu_gpu_transfer_probe
source /opt/xilinx/xrt/setup.sh
/tmp/npu_gpu_transfer_probe --list
timeout 20s /tmp/npu_gpu_transfer_probe hip_vmem_export_to_xrt_import
timeout 20s /tmp/npu_gpu_transfer_probe hip_vmem_gpu_write_xrt_read
timeout 20s /tmp/npu_gpu_transfer_probe hip_vmem_xrt_write_gpu_read
timeout 20s /tmp/npu_gpu_transfer_probe hip_vmem_gpu_write_xrt_read_stress_100
timeout 20s /tmp/npu_gpu_transfer_probe hip_vmem_xrt_write_gpu_read_stress_100
export TRANSFER_PROBE_VECADD_XCLBIN=/home/cj/mlir-air/build-xrt/test/xrt/40_triton_vec_add/test_npu2_peano/air.xclbin
export TRANSFER_PROBE_VECADD_INSTS=/home/cj/mlir-air/build-xrt/test/xrt/40_triton_vec_add/test_npu2_peano/air.insts.bin
timeout 60s /tmp/npu_gpu_transfer_probe xrt_host_bo_npu_vecadd_stress_1000
timeout 60s /tmp/npu_gpu_transfer_probe hip_vmem_imported_xrt_npu_vecadd_stress_1000
```

The transfer probe found HIP-owned allocation to be the viable candidate:
allocate handoff tensors as HIP VMem, export a POSIX fd, and import that fd into
XRT as a BO. See
[`../../docs/npu_gpu_transfer_methods.md`](../../docs/npu_gpu_transfer_methods.md)
for the method matrix and implementation recommendation.

The NPU vecadd stress path uses XRT's `kernel(...)` invocation form rather than
manual `xrt::run::set_arg` setup. On this stack that matches the passing XRT
tests and avoids relaunch instability with imported BOs. The probe also syncs
all imported BO views back from the XRT side after each NPU run before HIP
rewrites or reads them.

For direct handoff workload acceptance, use the Milestone 2 wrapper from
`programming_examples/heterogeneous_moe`:

```bash
source /opt/xilinx/xrt/setup.sh && ../../sandbox/bin/python run_llm_linear_milestone2.py
```

As of May 3, 2026, this wrapper is the accepted Milestone 2 hardware gate. It
builds `/tmp/libllm_linear_direct_bridge.so`, runs tiny direct regressions,
runs the `medium_m8_k512_h512_n256` direct workload in both directions in fresh
subprocesses, runs matching host-mixed baselines, captures logs under
`llm_linear/artifacts/benchmarks/milestone2_e2e/logs/`, and fails if any log
contains `Reverting to host copy of buffers` or
`exec_buf: Operation not supported`.

For fused int4 decode hardware acceptance, use the Milestone 3 wrapper:

```bash
source /opt/xilinx/xrt/setup.sh && ../../sandbox/bin/python run_llm_linear_milestone3.py
```

As of May 3, 2026, this wrapper is the accepted Milestone 3 hardware gate. It
uses the same bridge ABI version and also passes quantized decode metadata,
packed signed-int4 weights, and scale buffers for GPU/NPU decode. It writes logs
and results under `llm_linear/artifacts/benchmarks/milestone3_int4_hw/` and
covers the full `tiny_ci`, `medium`, and `llm_like` LLM-linear suites.

The G2N path audits the selected decode row as the handoff tensor. It avoids
`xrt::bo::copy`; when imported-parent sub-buffering is not viable, the bridge
uses a HIP device-to-device row copy into a row-sized HIP VMem allocation and
passes that imported row BO to NPU decode. Result JSONs must report row-sized
`direct_bytes`, the expected row offset `(M - 1) * H * 2`, and zero NumPy host
materializations.

For NPU int4 decode wider than the compiled tile width, the bridge stages static
packed-weight and scale tiles from the host-side static weight buffers into the
NPU decode BOs. That staging is static weight staging for the decode kernel; it
is not a prefill-to-decode handoff materialization and does not weaken the
Milestone 2 direct handoff contract.
