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
verification after the direct run records the XRT kernel wait.

Run a tiny direct case from `programming_examples/heterogeneous_moe`:

```bash
source /opt/xilinx/xrt/setup.sh
LLM_LINEAR_DIRECT_BRIDGE_SO=/tmp/libllm_linear_direct_bridge.so \
  ../../sandbox/bin/python run_llm_linear_suite.py \
    --suite tiny_ci \
    --case-filter gpu_prefill_npu_decode_direct \
    --iterations 1 \
    --warmup 0 \
    --allow-npu \
    --transfer-mode direct \
    --require-correctness
```

The direct bridge ABI also passes quantized decode metadata, packed signed-int4
weights, and scale buffers for GPU/NPU decode.

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
direct handoff contract.
