# Gemma3 Dataflow Kernel Sketches

This directory contains first-pass MLIR-AIR/NPU2 examples for the kernel
patterns described in "Mapping Gemma3 onto an Edge Dataflow Architecture".
The goal is to make each kernel concrete, testable, and easy to iterate on
before integrating them into an end-to-end Gemma3 runtime.

The examples include correctness-first and optimized Peano variants:

- `q4nx.py` / `q4nx.cc` / `q4nx_opt.cc`: dequantize Q4NX 32x256 int4
  block grids to bf16.
- `bf16_tiled_mm.py`: run the existing AIE2P tiled bf16 matrix-multiply
  generator with Gemma-style defaults; optimized targets reuse
  `../matrix_multiplication/bf16/mm_aie2p.cc` with `OPT_PERF_ENABLED`.
- `fused_dqp.py` / `fused_dqp.cc` / `fused_dqp_opt.cc`: fuse Q4NX
  dequantization with 32x256 matrix-vector block projections.
- `flowqkv.py` / `flow_attention.cc` / `flow_attention_opt.cc`: chunked prefill
  attention over one or more KV groups using online softmax accumulation.
- `flowkv.py` / `flow_attention.cc` / `flow_attention_opt.cc`: decode attention as
  the Q-chunk-size-1 specialization of the same grouped attention.
- `residual_add.py`: BF16 elementwise residual add for Gemma attention and MLP
  residual paths.

These are not FastFlowLM binary reproductions and do not use disassembly.
They are source-level kernels built from the public paper description and
nearby MLIR-AIR examples.

## Quick Start

From this directory, with the usual MLIR-AIR NPU environment active:

```bash
make run-q4nx
make run-mm
make run-fused-dqp
make run-flowqkv
make run-flowkv
python3 residual_add.py --compile-mode compile-only --output-format elf
```

Optimized Peano compile/run targets use the `*-opt` suffix:

```bash
make all-opt COMPILE_MODE=compile-only OUTPUT_FORMAT=elf
make run-q4nx-opt COMPILE_MODE=compile-only OUTPUT_FORMAT=elf
make run-mm-opt COMPILE_MODE=compile-only OUTPUT_FORMAT=elf
make run-fused-dqp-opt COMPILE_MODE=compile-only OUTPUT_FORMAT=elf
make run-flowqkv-opt COMPILE_MODE=compile-only OUTPUT_FORMAT=elf
make run-flowkv-opt COMPILE_MODE=compile-only OUTPUT_FORMAT=elf
```

Generate optimized Peano disassembly:

```bash
make dump-asm-opt COMPILE_MODE=compile-only OUTPUT_FORMAT=elf
```

Print generated AIR without compiling:

```bash
make print-q4nx
make print-flowqkv
```

Correctness defaults are deliberately small. Optimized defaults use scaled
physical-herd mappings. Override either set from `make`, for example:

```bash
make run-flowqkv Q_CHUNK=4 KV_LEN=64 HEAD_DIM=64 KV_CHUNK=16
make run-fused-dqp Q4NX_ROWS=32 Q4NX_COLS=256
```

## Scaled Mapping

The Python AIR wrappers separate the logical problem grid from the physical herd.
`--row-blocks`, `--col-blocks`, and `--groups` choose the logical Q4NX,
projection, or KV-group tiles; `--herd-rows` and `--herd-cols` choose the number
of simultaneously mapped CTs. Additional logical tiles can be covered by the
`air.launch` grid for compile-only exploration, following the launch/herd tiling
pattern used by the nearby GEMM and attention examples. The hardware-validated
defaults below keep these direct wrappers to one launch tile because multi-launch
Q4NX launch-tiled variants currently return corrupt data, while FusedDQP
launch-tiled or wider-herd variants currently corrupt data or time out on this
stack.

Optimized defaults currently use:

- Q4NX: 4x2 logical Q4NX blocks mapped through a 4x2 physical herd.
- FusedDQP: 2 logical projection row blocks mapped through a 2x1 physical herd.
- BF16 tiled MM: the existing 256x256x256 optimized GEMM with an 8x4 herd.
- FlowQKV: 4 KV groups mapped across a 4x1 herd.
- FlowKV: 1 KV group mapped on a 1x1 herd; grouped decode currently returns
  NaNs for groups beyond 0 in this direct wrapper.

This captures the paper block and KV-group decomposition while staying within
the direct L3-to-L1 DMA packet/channel, runtime launch, and FusedDQP fanout
limits seen in this source-level AIR mapping. The full paper schedule still has
further work: shared activation/KV staging, more explicit stream overlap, and
the FlowKV two-CT-per-KV-group split.

## Implementation Notes

Q4NX uses the paper block shape: 32 rows by 256 columns. Every two int4
weights are packed into one byte, low nibble first. Each column has one bf16
scale and one bf16 minimum offset, and dequantization is:

```text
w_bf16[row, col] = scale[col] * q4[row, col] + min[col]
```

The optimized Q4NX and FusedDQP microkernels expand packed int4 values in
8-column groups and use BF16 vector multiply-add/reduce operations. The
optimized FlowQKV/FlowKV microkernels vectorize BF16 dot products and value
accumulation while preserving the same online-softmax semantics.

The optimized attention kernels are still one group per CT. They are not yet
the full paper multi-CT, KV-partitioned schedule with explicit inter-CT
reductions and stream overlap; that remains the next integration step.
