# LLM-Linear Direct Bridge

This directory contains the native C++ bridge used by
`transfer_mode=direct`. The bridge owns handoff buffers as XRT BOs, exports
their file descriptors, imports them into HIP VMem, and launches the
no-host-staging GPU shared libraries plus XRT NPU kernels.

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
