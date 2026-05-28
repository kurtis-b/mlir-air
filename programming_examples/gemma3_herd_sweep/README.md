# Gemma3 Herd Sweep Kernels

This programming example collects source-level MLIR-AIR/NPU2 kernels for a
Gemma3-style transformer dataflow and makes their physical herd shape selectable
from the command line. The supported shapes are `2x4`, `4x4`, and `8x4`.

The example uses Peano/AIE2P microkernels throughout:

- `q4nx.py` with `q4nx_opt.cc`: Q4NX int4 block dequantization for 32x256 blocks.
- `bf16_tiled_mm.py`: a Gemma-style wrapper around the optimized BF16 tiled GEMM
  implementation in `../matrix_multiplication/bf16/mm_aie2p.cc`.
- `fused_dqp.py` with `fused_dqp_opt.cc`: fused Q4NX dequantization and
  projection over 32x256 row blocks.
- `flowqkv.py` with `flow_attention_opt.cc`: grouped prefill attention over a
  query chunk and per-group KV cache rows.
- `flowkv.py` with `flow_attention_opt.cc`: decode attention as the q-chunk-1
  specialization of the grouped attention path.

## Quick Start

From this directory, with the usual MLIR-AIR NPU environment active:

```bash
make run-q4nx HERD_SHAPE=2x4 COMPILE_MODE=compile-only OUTPUT_FORMAT=elf
make run-mm HERD_SHAPE=4x4 COMPILE_MODE=compile-only OUTPUT_FORMAT=elf
make run-fused-dqp HERD_SHAPE=8x4 COMPILE_MODE=compile-only OUTPUT_FORMAT=elf
make run-flowqkv HERD_SHAPE=4x4 COMPILE_MODE=compile-only OUTPUT_FORMAT=elf
make run-flowkv HERD_SHAPE=2x4 COMPILE_MODE=compile-only OUTPUT_FORMAT=elf
```

Run the full compile sweep:

```bash
make sweep COMPILE_MODE=compile-only OUTPUT_FORMAT=elf
```

Run a hardware validation target by changing `COMPILE_MODE`:

```bash
make run-q4nx HERD_SHAPE=2x4 COMPILE_MODE=compile-and-run OUTPUT_FORMAT=elf
```

For hardware runs, source the MLIR-AIR environment first and XRT setup after it so
`pyxrt` remains on `PYTHONPATH`. Do not wrap these targets in an outer
`flock /tmp/npu.lock`; `XRTRunner` already takes that lock internally.

## Herd Shapes

`HERD_SHAPE` controls both the physical herd dimensions and the default logical
workload size. The Python drivers expose the same setting as `--herd-shape`.

| Shape | CTs | Default workload |
| --- | ---: | --- |
| `2x4` | 8 | 8 Q4NX/projection/attention groups, MM M=64 N=128 |
| `4x4` | 16 | 16 Q4NX/projection/attention groups, MM M=128 N=128 |
| `8x4` | 32 | 32 Q4NX/projection/attention groups, MM M=256 N=128 |

The problem dimensions remain intentionally compact so the example is usable as
a compile and runtime smoke test. Override `Q_CHUNK`, `KV_LEN`, `HEAD_DIM`,
`MM_M`, `MM_N`, and related Make variables for larger experiments.

## Mapping Notes

Q4NX maps one 32x256 dequantization block per CT. Packed weights and the BF16
scale/min parameter pack are staged from L3 to L2, then each CT pulls its tile to
L1. Scale and min are copied into contiguous L1 buffers before calling the Peano
microkernel. Results are gathered through L2 before returning to L3.

BF16 tiled MM reuses the optimized AIE2P GEMM path. Matrix rows scale with herd
rows and matrix columns scale with herd columns. The wrapper keeps the GEMM
L3/L2/L1 tiling, Peano microkernel, and runtime tiling behavior from the
reference programming example.

FusedDQP maps one 32x256 projection row block per CT. The wrapper packs scale,
min, and the activation vector into a single BF16 per-block input so each CT uses
one BF16 staged input plus one packed-weight staged input. The 2x4 and 4x4 shapes
write the small projection result directly to L3; 8x4 gathers results through L2
to stay within shim-channel limits.

FlowQKV and FlowKV map one attention group per CT in this source-level example.
Q, K, and V are packed as one BF16 input per group, staged through L2, and split
into contiguous Q/K/V L1 buffers before the Peano attention microkernel runs.
The 8x4 shape gathers outputs through L2; smaller shapes write outputs directly.

These examples are meant to be readable, sweepable AIR mappings. They do not use
binary disassembly, generated instruction traces, or model-runtime integration.
