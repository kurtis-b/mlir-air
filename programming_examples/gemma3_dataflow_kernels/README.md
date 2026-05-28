# Gemma3 Dataflow Kernel Sketches

This directory contains first-pass MLIR-AIR/NPU2 examples for the kernel
patterns described in "Mapping Gemma3 onto an Edge Dataflow Architecture".
The goal is to make each kernel concrete, testable, and easy to iterate on
before integrating them into an end-to-end Gemma3 runtime.

The examples are intentionally correctness-first:

- `q4nx.py` / `q4nx.cc`: dequantize one Q4NX 32x256 int4 block to bf16.
- `bf16_tiled_mm.py`: run the existing AIE2P tiled bf16 matrix-multiply
  generator with Gemma-style defaults.
- `fused_dqp.py` / `fused_dqp.cc`: fuse Q4NX dequantization with one 32x256
  matrix-vector block projection.
- `flowqkv.py` / `flow_attention.cc`: chunked prefill attention over one KV
  group using online softmax accumulation.
- `flowkv.py` / `flow_attention.cc`: decode attention as the Q-chunk-size-1
  specialization of the same chunked attention.

These are not FastFlowLM binary reproductions and do not use disassembly.
They are source-level kernels built from the paper's public description and
nearby MLIR-AIR examples.

## Quick Start

From this directory, with the usual MLIR-AIR NPU environment active:

```bash
make run-q4nx
make run-mm
make run-fused-dqp
make run-flowqkv
make run-flowkv
```

Print generated AIR without compiling:

```bash
make print-q4nx
make print-flowqkv
```

The default shapes are deliberately small. Override them from `make`, for
example:

```bash
make run-flowqkv Q_CHUNK=4 KV_LEN=64 HEAD_DIM=64 KV_CHUNK=16
make run-fused-dqp Q4NX_ROWS=32 Q4NX_COLS=256
```

## Implementation Notes

Q4NX uses the paper's block shape: 32 rows by 256 columns. Every two int4
weights are packed into one byte, low nibble first. Each column has one bf16
scale and one bf16 minimum offset, and dequantization is:

```text
w_bf16[row, col] = scale[col] * q4[row, col] + min[col]
```

`FlowQKV` and `FlowKV` currently use a single herd and scalar C++ loops for
the attention math. They preserve the algorithmic semantics and validation
surface while leaving the full 8-column/CT-pair paper mapping as the next
optimization step.
