# LLM-Linear Direct Bridge

This directory contains native C++ helpers for investigating
`transfer_mode=direct`.

`direct_bridge.cpp` is the runtime bridge currently called by the Python
LLM-linear harness. It owns handoff buffers as XRT BOs, exports their file
descriptors, imports them into HIP VMem, and launches the no-host-staging GPU
shared libraries plus XRT NPU kernels. That ownership direction is not accepted
on the current Ryzen AI stack.

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

On the current Ryzen AI test machine, the bridge builds but the probe fails
closed because each XRT BO flag tested for HIP VMem import aborts in an isolated
child process. Direct benchmark cases must not be treated as accepted until this
probe succeeds on the target stack.

Transfer-method probe:

```bash
llm_linear/native/build_transfer_probe.sh /tmp/npu_gpu_transfer_probe
source /opt/xilinx/xrt/setup.sh
/tmp/npu_gpu_transfer_probe --list
timeout 20s /tmp/npu_gpu_transfer_probe hip_vmem_export_to_xrt_import
timeout 20s /tmp/npu_gpu_transfer_probe hip_vmem_gpu_write_xrt_read
timeout 20s /tmp/npu_gpu_transfer_probe hip_vmem_xrt_write_gpu_read
```

The transfer probe found the opposite ownership direction to be the viable
candidate: allocate handoff tensors as HIP VMem, export a POSIX fd, and import
that fd into XRT as a BO. See
[`../../docs/npu_gpu_transfer_methods.md`](../../docs/npu_gpu_transfer_methods.md)
for the method matrix and implementation recommendation.
